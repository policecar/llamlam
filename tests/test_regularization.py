import pytest
import torch
from torch.optim import Adam

from llamlam.config import Config, Config
from llamlam.model import GPTModel


@pytest.fixture
def model():
    config = Config(max_seq_length=16, vocab_size=100, n_layers=2, n_heads=2, dim_head=8)
    return GPTModel(config)


@pytest.fixture
def train_config():
    return Config()


def test_weight_decay(model, train_config):
    input_ids = torch.randint(0, 100, (4, 16))
    optimizer = Adam(
        model.parameters(), lr=1e-3, weight_decay=train_config.weight_decay
    )

    initial_norm = sum(p.norm().item() for p in model.parameters())

    for _ in range(100):
        optimizer.zero_grad()
        loss = model(input_ids)["loss"]
        loss.backward()
        optimizer.step()

    final_norm = sum(p.norm().item() for p in model.parameters())

    assert final_norm < initial_norm, "Weight decay did not reduce parameter norms"


def test_gradient_clipping(model, train_config):
    input_ids = torch.randint(0, 100, (4, 16))
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Compute gradients once
    model(input_ids)["loss"].backward()

    # Get initial norm before clipping
    initial_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    # Now clip the same gradients
    clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), train_config.gradient_clipping
    )

    assert (
        clipped_grad_norm <= train_config.gradient_clipping + 1e-5
    ), "Gradient clipping did not limit gradient norm"
    assert (
        clipped_grad_norm <= initial_grad_norm + 1e-5
    ), "Gradient clipping increased gradient norm"


def test_dropout(model):
    model.eval()
    input_ids = torch.randint(0, 100, (4, 16))

    with torch.no_grad():
        output1 = model(input_ids)["logits"]
        output2 = model(input_ids)["logits"]

    assert torch.allclose(output1, output2), "Model outputs differ in eval mode"

    model.train()
    output1 = model(input_ids)["logits"]
    output2 = model(input_ids)["logits"]

    assert not torch.allclose(
        output1, output2
    ), "Model outputs are identical in train mode"


def test_layer_norm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            input_shape = (4, 16, module.normalized_shape[0])
            x = torch.randn(input_shape)
            output = module(x)

            assert torch.allclose(
                output.mean(dim=-1), torch.zeros(4, 16), atol=1e-5
            ), "LayerNorm output mean is not close to zero"
            # std() uses Bessel's correction, so we need higher tolerance
            assert torch.allclose(
                output.std(dim=-1), torch.ones(4, 16), atol=5e-2
            ), "LayerNorm output std is not close to one"


def test_weight_initialization(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            if "norm" not in name:  # Skip LayerNorm weights
                assert param.mean().abs() < 0.1, f"Weight {name} has high mean"
                # Model uses custom initialization with alpha * (1/dim_embd)**0.5
                # which gives smaller std values (0.05 - 0.5 range)
                assert 0.01 < param.std() < 2, f"Weight {name} has unusual std: {param.std()}"
        elif "bias" in name:
            assert param.mean().abs() < 0.1, f"Bias {name} has high mean"


def test_overfitting_small_dataset(model):
    input_ids = torch.randint(0, 100, (2, 16))
    optimizer = Adam(model.parameters(), lr=1e-3)

    initial_loss = model(input_ids)["loss"].item()

    for _ in range(1000):
        optimizer.zero_grad()
        loss = model(input_ids)["loss"]
        loss.backward()
        optimizer.step()

    final_loss = model(input_ids)["loss"].item()

    assert final_loss < initial_loss, "Model failed to overfit small dataset"
    assert final_loss < 0.1, "Model did not achieve near-zero loss on small dataset"
