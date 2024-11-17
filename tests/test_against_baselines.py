import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from llamlam.config import Config
from llamlam.model import GPTModel


@pytest.fixture
def config():
    return Config(
        max_length=16,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        head_width=8,
        dropout=0.0,  # Disable dropout for testing
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return GPTModel(config)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def sample_data(config):
    """Generate synthetic data for testing."""
    torch.manual_seed(42)
    batch_size = 4
    seq_length = 16
    n_batches = 5
    return [
        {"input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_length))}
        for _ in range(n_batches)
    ]  # TODO: use real data


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

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = self.logits.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {"loss": loss, "logits": logits}


def test_against_uniform_baseline(model, sample_data):
    """Test that model performs better than uniform distribution baseline."""
    uniform_baseline = UniformBaseline(model.config.vocab_size)

    model_loss = 0
    baseline_loss = 0

    for batch in sample_data:
        input_ids = batch["input_ids"]
        with torch.no_grad():
            model_output = model(input_ids)
            baseline_output = uniform_baseline(input_ids)

        model_loss += model_output["loss"].item()
        baseline_loss += baseline_output["loss"].item()

    model_loss /= len(sample_data)
    baseline_loss /= len(sample_data)

    assert model_loss < baseline_loss, (
        f"Model loss {model_loss:.3f} is not better than uniform baseline "
        f"loss {baseline_loss:.3f}"
    )


def test_against_unigram_baseline(model, sample_data):
    """Test that model performs better than unigram frequency baseline."""
    unigram_baseline = UnigramBaseline(model.config.vocab_size)
    optimizer = torch.optim.Adam(unigram_baseline.parameters(), lr=0.01)

    # Train unigram baseline
    for _ in range(5):
        for batch in sample_data:
            input_ids = batch["input_ids"]
            loss = unigram_baseline(input_ids)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model_loss = 0
    baseline_loss = 0

    with torch.no_grad():
        for batch in sample_data:
            input_ids = batch["input_ids"]
            model_output = model(input_ids)
            baseline_output = unigram_baseline(input_ids)

            model_loss += model_output["loss"].item()
            baseline_loss += baseline_output["loss"].item()

    model_loss /= len(sample_data)
    baseline_loss /= len(sample_data)

    assert model_loss < baseline_loss, (
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


def test_next_token_prediction(model):
    """Test model's ability to predict next tokens."""
    # Create a simple sequence
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

    with torch.no_grad():
        logits = model(input_ids)["logits"]

    # Get predicted tokens
    predicted_tokens = logits.argmax(dim=-1)

    # Check if any predicted token matches the actual next token
    matches = (predicted_tokens[:, :-1] == input_ids[:, 1:]).any()

    assert matches, "Model failed to correctly predict any next token"


def test_generate_coherent_text(model, tokenizer):
    """Test model's text generation capabilities."""
    prompt = "Once upon a time"
    generated_text = model.generate(tokenizer, prompt, max_new_tokens=20)

    # Basic coherence checks
    assert len(generated_text.split()) > len(
        prompt.split()
    ), "Model failed to generate new tokens"
    assert generated_text.startswith(
        prompt
    ), "Generated text does not start with the prompt"
    assert len(set(generated_text.split())) > 1, "Generated text has no variety"

    # Additional checks for generation quality
    words = generated_text.split()
    assert len(words) >= 5, "Generated text is too short"
    assert len(set(words)) / len(words) > 0.3, "Generated text has too much repetition"


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
