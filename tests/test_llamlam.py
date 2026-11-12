import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from llamlam.config import Config
from llamlam.model import GPTModel
from llamlam.utils import set_seed


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)


@pytest.fixture
def config():
    return Config(
        max_seq_length=128,
        vocab_size=50257,
        n_layers=4,
        n_heads=4,
        dim_head=16,
    )


@pytest.fixture
def train_config():
    return Config(
        seed=42,
        n_epochs=3,
        learning_rate=5e-4,
        weight_decay=0.01,
        batch_size=4,
    )


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def model(config):
    return GPTModel(config)


@pytest.fixture
def sample_batch(config):
    batch_size = 2
    seq_length = 64  # Shorter than max_seq_length for testing efficiency
    return {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
    }


def test_model_initialization(config):
    """Test model initialization and architecture."""
    model = GPTModel(config)

    # Test basic model structure
    assert isinstance(model, GPTModel)
    assert model.config.n_layers == config.n_layers
    assert model.config.n_heads == config.n_heads
    assert model.config.dim_head == config.dim_head
    assert model.config.dim_embd == config.n_heads * config.dim_head

    # Test component dimensions
    assert model.embed.weight.shape == (config.vocab_size, config.dim_embd)
    assert model.pos_embed.shape == (1, config.max_seq_length, config.dim_embd)
    assert model.head.weight.shape == (config.vocab_size, config.dim_embd)
    assert len(model.blocks) == config.n_layers


def test_forward_pass(model, sample_batch):
    """Test the forward pass of the model."""
    model.eval()
    with torch.no_grad():
        outputs = model(sample_batch["input_ids"])

    assert "logits" in outputs
    assert "loss" in outputs

    batch_size, seq_length = sample_batch["input_ids"].shape
    assert outputs["logits"].shape == (batch_size, seq_length, model.config.vocab_size)
    assert isinstance(outputs["loss"].item(), float)


def test_loss_calculation(model, sample_batch):
    """Test loss calculation and shape."""
    model.train()
    outputs = model(sample_batch["input_ids"])

    assert "loss" in outputs
    assert isinstance(outputs["loss"].item(), float)
    assert outputs["loss"].item() > 0

    # Test that loss is properly reduced
    assert outputs["loss"].dim() == 0

    # Verify loss calculation manually
    logits = outputs["logits"][:, :-1, :]
    targets = sample_batch["input_ids"][:, 1:]
    manual_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )
    assert torch.allclose(outputs["loss"], manual_loss)


@pytest.mark.slow
def test_overfit_single_batch(model, sample_batch, train_config):
    """Test model's ability to overfit a single batch."""
    set_seed(train_config.seed)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    initial_loss = model(sample_batch["input_ids"])["loss"].item()

    for _ in range(500):
        optimizer.zero_grad()
        outputs = model(sample_batch["input_ids"])
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

    final_loss = model(sample_batch["input_ids"])["loss"].item()
    assert final_loss < initial_loss
    assert final_loss < 0.1


def test_generate(model, tokenizer):
    """generate() appends max_new_tokens greedy tokens to the prompt, in place.

    The expected text comes from greedy decoding written out in the test, so a
    wrong token count, non-greedy choice or lost prompt changes the result.
    """
    prompt = "Once upon a time"
    max_new_tokens = 20
    generated_text = model.generate(tokenizer, prompt, max_new_tokens=max_new_tokens)

    assert isinstance(generated_text, str)
    assert len(generated_text) > len(prompt)
    assert generated_text.startswith(prompt)
    # generate() runs where the model lives (CPU here) and does not move it
    assert {p.device.type for p in model.parameters()} == {"cpu"}

    ids = tokenizer.encode(prompt, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids[:, -model.config.max_seq_length :])["logits"]
            ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    expected = tokenizer.decode(ids[0], skip_special_tokens=True)

    assert generated_text == expected
    assert model.generate(tokenizer, prompt, max_new_tokens) == generated_text


def test_save_and_load(model, config, tmp_path):
    """Test model state saving and loading."""
    save_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), save_path)

    loaded_model = GPTModel(config)
    # precondition: a fresh model differs, so equality below comes from loading
    assert not torch.allclose(model.embed.weight, loaded_model.embed.weight)
    loaded_model.load_state_dict(torch.load(save_path, weights_only=True))

    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)


def test_reproducibility(config):
    """Test model initialization reproducibility."""
    set_seed(42)
    model1 = GPTModel(config)

    set_seed(42)
    model2 = GPTModel(config)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_attention_mask(model):
    """attention_mask drops padded targets from the loss; it never changes logits.

    Attention is causal, so right padding cannot influence real positions; the
    mask (1 = real token, 0 = padding) only removes padded targets from the mean.
    """
    model.eval()  # no dropout: the two forward passes must be comparable
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))

    # Create a mask that masks out the second half of the sequence
    attention_mask = torch.ones((batch_size, seq_length))
    attention_mask[:, seq_length // 2 :] = 0

    with torch.no_grad():
        outputs_with_mask = model(input_ids, attention_mask=attention_mask)
        outputs_without_mask = model(input_ids)

    # Logits are identical with and without the mask
    torch.testing.assert_close(
        outputs_with_mask["logits"], outputs_without_mask["logits"], rtol=0, atol=0
    )

    # The loss is the mean over targets whose own mask entry is 1
    logits = outputs_without_mask["logits"][:, :-1]
    targets = input_ids[:, 1:]
    real = attention_mask[:, 1:].bool()
    expected = F.cross_entropy(logits[real], targets[real])
    torch.testing.assert_close(outputs_with_mask["loss"], expected)
    assert not torch.allclose(outputs_with_mask["loss"], outputs_without_mask["loss"])

    # A mask that hides every target leaves nothing to average
    no_targets = torch.zeros_like(attention_mask)
    no_targets[:, 0] = 1  # position 0 is never a target
    with pytest.raises(ValueError):
        model(input_ids, attention_mask=no_targets)


def test_gradient_flow(model, sample_batch):
    """Test gradient computation and flow."""
    model.train()
    model.zero_grad()

    outputs = model(sample_batch["input_ids"])
    loss = outputs["loss"]
    loss.backward()

    # Check gradients for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
        # Check that at least some gradients are non-zero
        assert param.grad.abs().sum() > 0, f"Zero gradients for {name}"


def test_hidden_states_output(model, sample_batch):
    """Test hidden states output functionality."""
    outputs = model(sample_batch["input_ids"], output_hidden_states=True)

    assert "hidden_states" in outputs
    hidden_states = outputs["hidden_states"]

    # Check number of hidden states (input + one per layer)
    assert len(hidden_states) == model.config.n_layers + 1

    # Check dimensions of each hidden state
    batch_size, seq_length = sample_batch["input_ids"].shape
    for hidden_state in hidden_states:
        assert hidden_state.shape == (batch_size, seq_length, model.config.dim_embd)


if __name__ == "__main__":
    pytest.main()
