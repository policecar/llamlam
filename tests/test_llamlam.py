import pytest
import torch
from transformers import AutoTokenizer

from llamlam.config import Config
from llamlam.model import GPTModel
from llamlam.utils import set_seed


@pytest.fixture
def config():
    return Config(
        max_length=128,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        head_width=16,
    )


@pytest.fixture
def train_config():
    return Config(
        seed=42,
        num_epochs=3,
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
    seq_length = 64  # Shorter than max_length for testing efficiency
    return {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
    }


def test_model_initialization(config):
    """Test model initialization and architecture."""
    model = GPTModel(config)

    # Test basic model structure
    assert isinstance(model, GPTModel)
    assert model.config.n_layer == config.n_layer
    assert model.config.n_head == config.n_head
    assert model.config.head_width == config.head_width
    assert model.config.n_embd == config.n_head * config.head_width

    # Test component dimensions
    assert model.embed.weight.shape == (config.vocab_size, config.n_embd)
    assert model.pos_embed.shape == (1, config.max_length, config.n_embd)
    assert model.head.weight.shape == (config.vocab_size, config.n_embd)
    assert len(model.blocks) == config.n_layer


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
    """Test text generation capability."""
    prompt = "Once upon a time"
    generated_text = model.generate(tokenizer, prompt, max_new_tokens=20)

    assert isinstance(generated_text, str)
    assert len(generated_text) > len(prompt)
    assert generated_text.startswith(prompt)


def test_save_and_load(model, config, tmp_path):
    """Test model state saving and loading."""
    save_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), save_path)

    loaded_model = GPTModel(config)
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
    """Test attention mask functionality."""
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))

    # Create a mask that masks out the second half of the sequence
    attention_mask = torch.ones((batch_size, seq_length))
    attention_mask[:, seq_length // 2 :] = 0

    outputs_with_mask = model(input_ids, attention_mask=attention_mask)
    outputs_without_mask = model(input_ids)

    # Outputs should differ when using mask
    assert not torch.allclose(
        outputs_with_mask["logits"], outputs_without_mask["logits"]
    )


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
    assert len(hidden_states) == model.config.n_layer + 1

    # Check dimensions of each hidden state
    batch_size, seq_length = sample_batch["input_ids"].shape
    for hidden_state in hidden_states:
        assert hidden_state.shape == (batch_size, seq_length, model.config.n_embd)


if __name__ == "__main__":
    pytest.main()
