"""
Compare GPTModel with trivial baselines on small, learnable synthetic data.

The data come from a Markov chain over the first N_ACTIVE tokens of the
vocabulary: the next token follows a fixed rule with probability P_RULE and is
uniform over the active tokens otherwise. Its unigram distribution is far from
uniform over the vocabulary (unigram beats uniform), and the next token depends
on the current one (a sequence model beats unigram). An untrained model is at
chance level and loses to both baselines.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from llamlam.config import Config
from llamlam.model import GPTModel

N_ACTIVE = 20  # tokens actually used, out of vocab_size
P_RULE = 0.9  # probability that the next token follows the rule
MARGIN = 0.5  # nats by which the trained model must beat a baseline


def make_config():
    return Config(
        max_seq_length=16,
        vocab_size=100,
        n_layers=2,
        n_heads=2,
        dim_head=8,
        dropout=0.0,  # Disable dropout for testing
    )


def markov_batches(n_batches, seed, batch_size=16, seq_length=16):
    """Sequences where token t+1 = (7 * token t + 3) % N_ACTIVE w.p. P_RULE."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        ids = torch.empty(batch_size, seq_length, dtype=torch.long)
        ids[:, 0] = torch.randint(0, N_ACTIVE, (batch_size,), generator=g)
        for t in range(1, seq_length):
            follow = torch.rand(batch_size, generator=g) < P_RULE
            noise = torch.randint(0, N_ACTIVE, (batch_size,), generator=g)
            ids[:, t] = torch.where(follow, (7 * ids[:, t - 1] + 3) % N_ACTIVE, noise)
        batches.append({"input_ids": ids})
    return batches


def train(model, batches, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in batches:
            loss = model(batch["input_ids"])["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    return model


def mean_loss(model, batches):
    with torch.no_grad():
        return sum(model(b["input_ids"])["loss"].item() for b in batches) / len(batches)


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return GPTModel(config)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def sample_data():
    """Training data: structured, non-uniform token sequences."""
    return markov_batches(n_batches=4, seed=0)


@pytest.fixture
def heldout_data():
    """Fresh sequences from the same process, never trained on."""
    return markov_batches(n_batches=2, seed=1)


@pytest.fixture(scope="module")
def trained_model():
    """A model briefly trained (40 Adam steps) on the Markov training data."""
    torch.manual_seed(42)
    model = GPTModel(make_config())
    return train(model, markov_batches(n_batches=4, seed=0), epochs=10, lr=1e-2)


class UniformBaseline(nn.Module):
    """Baseline model that predicts uniform distribution over vocabulary."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.ones(
            batch_size, seq_len, self.vocab_size, device=input_ids.device
        )
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {"loss": loss, "logits": logits}


class UnigramBaseline(nn.Module):
    """Baseline model that learns token frequencies."""

    def __init__(self, vocab_size):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(vocab_size))

    @torch.no_grad()
    def fit(self, batches, smoothing=1e-3):
        """Set logits to the log target-token frequencies (the unigram optimum)."""
        counts = torch.full_like(self.logits, smoothing)
        for batch in batches:
            targets = batch["input_ids"][..., 1:].flatten()
            counts += torch.bincount(targets, minlength=counts.numel())
        self.logits.copy_(torch.log(counts / counts.sum()))
        return self

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = self.logits.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {"loss": loss, "logits": logits}


def test_against_uniform_baseline(trained_model, heldout_data):
    """Test that a briefly trained model beats the uniform distribution baseline."""
    uniform_baseline = UniformBaseline(trained_model.config.vocab_size)

    model_loss = mean_loss(trained_model, heldout_data)
    baseline_loss = mean_loss(uniform_baseline, heldout_data)

    assert model_loss < baseline_loss - MARGIN, (
        f"Model loss {model_loss:.3f} is not better than uniform baseline "
        f"loss {baseline_loss:.3f}"
    )


def test_against_unigram_baseline(trained_model, sample_data, heldout_data):
    """Test that a briefly trained model beats the unigram frequency baseline."""
    unigram_baseline = UnigramBaseline(trained_model.config.vocab_size)
    unigram_baseline.fit(sample_data)  # same training data as the model

    model_loss = mean_loss(trained_model, heldout_data)
    baseline_loss = mean_loss(unigram_baseline, heldout_data)

    # The baseline is meaningful: it is clearly better than uniform
    uniform_loss = mean_loss(
        UniformBaseline(trained_model.config.vocab_size), heldout_data
    )
    assert baseline_loss < uniform_loss - MARGIN

    assert model_loss < baseline_loss - MARGIN, (
        f"Model loss {model_loss:.3f} is not better than unigram baseline "
        f"loss {baseline_loss:.3f}"
    )


def test_perplexity_improvement(model, sample_data):
    """Test that model perplexity improves with training."""
    # Calculate initial perplexity
    initial_loss = 0
    with torch.no_grad():
        for batch in sample_data:
            input_ids = batch["input_ids"]
            initial_loss += model(input_ids)["loss"].item()
    initial_loss /= len(sample_data)
    initial_perplexity = torch.exp(torch.tensor(initial_loss)).item()

    # Train for a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(5):
        for batch in sample_data:
            input_ids = batch["input_ids"]
            loss = model(input_ids)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Calculate final perplexity
    final_loss = 0
    with torch.no_grad():
        for batch in sample_data:
            input_ids = batch["input_ids"]
            final_loss += model(input_ids)["loss"].item()
    final_loss /= len(sample_data)
    final_perplexity = torch.exp(torch.tensor(final_loss)).item()

    assert final_perplexity < initial_perplexity, (
        f"Perplexity did not improve: initial {initial_perplexity:.3f}, "
        f"final {final_perplexity:.3f}"
    )


def test_next_token_prediction(trained_model, heldout_data):
    """Test model's ability to predict next tokens on unseen sequences.

    The Bayes-optimal top-1 accuracy is P_RULE + (1 - P_RULE) / N_ACTIVE ~ 0.905;
    chance level over the vocabulary is 0.01.
    """
    correct = total = 0
    with torch.no_grad():
        for batch in heldout_data:
            input_ids = batch["input_ids"]
            predicted_tokens = trained_model(input_ids)["logits"].argmax(dim=-1)
            correct += (predicted_tokens[:, :-1] == input_ids[:, 1:]).sum().item()
            total += input_ids[:, 1:].numel()

    accuracy = correct / total
    assert accuracy > 0.8, f"Next-token accuracy {accuracy:.3f} is too low"


def test_generate_coherent_text(tokenizer):
    """After memorising a sentence, greedy generation from its start completes it."""
    sentence = "Once upon a time there was a little cat who lived in a big red house."
    prompt = "Once upon a time"
    sentence_ids = tokenizer.encode(sentence, return_tensors="pt")
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    n_prompt, n_sentence = prompt_ids.size(1), sentence_ids.size(1)
    assert torch.equal(sentence_ids[:, :n_prompt], prompt_ids)  # precondition

    torch.manual_seed(42)
    config = Config(
        max_seq_length=32,
        vocab_size=len(tokenizer),
        n_layers=2,
        n_heads=2,
        dim_head=8,
        dropout=0.0,
    )
    model = train(GPTModel(config), [{"input_ids": sentence_ids}], epochs=100, lr=1e-2)

    generated_text = model.generate(
        tokenizer, prompt, max_new_tokens=n_sentence - n_prompt
    )

    assert generated_text == sentence, f"Generated {generated_text!r}"


def test_model_memorization(model):
    """Test that model can memorize a very simple pattern."""
    # Create a simple repeating pattern
    pattern = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])

    # Train model to memorize pattern
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = model(pattern)["loss"].item()

    for _ in range(100):
        loss = model(pattern)["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    final_loss = model(pattern)["loss"].item()

    assert final_loss < initial_loss, "Model failed to memorize simple pattern"
    assert final_loss < 0.1, f"Final loss {final_loss:.3f} is too high"


if __name__ == "__main__":
    pytest.main([__file__])
