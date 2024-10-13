import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llamlam.config import LMConfig, TrainConfig
from llamlam.data import CustomDataset
from llamlam.model import GPTModel
from llamlam.utils import set_seed


@pytest.fixture
def model_config():
    return LMConfig(
        max_length=128,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        head_width=16,
    )


@pytest.fixture
def train_config():
    return TrainConfig(
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
def model(model_config):
    return GPTModel(model_config)


@pytest.fixture
def dataset(tokenizer):
    return CustomDataset(tokenizer, type_path="train", max_length=128, debug=True)


def test_model_initialization(model_config):
    model = GPTModel(model_config)
    assert isinstance(model, GPTModel)
    assert model.config.n_layer == model_config.n_layer
    assert model.config.n_head == model_config.n_head
    assert model.config.head_width == model_config.head_width
    assert model.config.n_embd == model_config.n_head * model_config.head_width


def test_forward_pass(model, dataset):
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda x: {"input_ids": torch.stack([item["input_ids"] for item in x])},
    )
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (2, 128, model.config.vocab_size)


def test_loss_calculation(model, dataset):
    model.train()
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda x: {"input_ids": torch.stack([item["input_ids"] for item in x])},
    )
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]

    outputs = model(input_ids)
    assert "loss" in outputs
    assert isinstance(outputs["loss"].item(), float)
    assert outputs["loss"].item() > 0


def test_overfit_single_batch(model, dataset, train_config):
    set_seed(train_config.seed)
    model.train()
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda x: {"input_ids": torch.stack([item["input_ids"] for item in x])},
    )
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]

    optimizer = torch.optim.AdamW(  # type: ignore
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    initial_loss = model(input_ids)["loss"].item()

    for _ in range(100):  # Increase this number if the model doesn't overfit
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

    final_loss = model(input_ids)["loss"].item()
    assert final_loss < initial_loss
    assert final_loss < 0.1  # This threshold might need adjustment


def test_generate(model, tokenizer):
    prompt = "Once upon a time"
    generated_text = model.generate(tokenizer, prompt, max_new_tokens=20)
    assert isinstance(generated_text, str)
    assert len(generated_text) > len(prompt)
    assert generated_text.startswith(prompt)


def test_save_and_load(model, model_config, tmp_path):
    # Save the model
    save_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), save_path)

    # Load the model
    loaded_model = GPTModel(model_config)
    loaded_model.load_state_dict(torch.load(save_path))

    # Compare model parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)


def test_attention_mask(model):
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    attention_mask[:, 5:] = 0  # Mask out the last 5 tokens

    outputs_with_mask = model(input_ids, attention_mask=attention_mask)
    outputs_without_mask = model(input_ids)

    # The outputs should be different when using an attention mask
    assert not torch.allclose(outputs_with_mask["logits"], outputs_without_mask["logits"])


def test_gradient_flow(model, dataset):
    model.train()
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda x: {"input_ids": torch.stack([item["input_ids"] for item in x])},
    )
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]

    outputs = model(input_ids)
    loss = outputs["loss"]
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"


if __name__ == "__main__":
    pytest.main()
