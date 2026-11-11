import pytest
import torch
from torch.optim import Adam, AdamW

from llamlam.config import Config
from llamlam.model import GPTModel, LayerNorm
from llamlam.utils import get_grouped_params


def small_config(**overrides):
    kwargs = dict(max_seq_length=16, vocab_size=100, n_layers=2, n_heads=2, dim_head=8)
    kwargs.update(overrides)
    return Config(**kwargs)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def model():
    return GPTModel(small_config())  # default dropout (0.1), needed by test_dropout


@pytest.fixture
def det_model():
    """Same architecture without dropout, for deterministic optimisation tests."""
    return GPTModel(small_config(dropout=0.0))


@pytest.fixture
def train_config():
    return Config()


def test_weight_decay(det_model, train_config):
    """Weight decay shrinks the decayed weights relative to training without it.

    Training on a fixed batch grows the weights, so the comparison is against an
    identical run with weight_decay=0, using the repo's optimizer param groups.
    """
    input_ids = torch.randint(0, 100, (4, 16))
    init_state = det_model.state_dict()

    def decayed_norm_after_training(weight_decay):
        model = GPTModel(det_model.config)
        model.load_state_dict(init_state)
        groups = get_grouped_params(model, weight_decay=weight_decay)
        optimizer = AdamW(groups, lr=1e-3)
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(input_ids)["loss"]
            loss.backward()
            optimizer.step()
        return sum(p.norm().item() for p in groups[0]["params"])

    assert train_config.weight_decay > 0
    norm_without = decayed_norm_after_training(0.0)
    norm_with = decayed_norm_after_training(train_config.weight_decay)

    assert norm_with < norm_without, "Weight decay did not reduce parameter norms"


def test_gradient_clipping(det_model, train_config):
    max_norm = train_config.gradient_clipping
    input_ids = torch.randint(0, 100, (4, 16))

    # Scale the loss so the gradient norm is well above the clipping threshold
    (10 * det_model(input_ids)["loss"]).backward()
    initial_grads = [p.grad.clone() for p in det_model.parameters()]
    initial_grad_norm = torch.nn.utils.clip_grad_norm_(
        det_model.parameters(), float("inf")
    )
    assert initial_grad_norm > 2 * max_norm, "precondition: gradients need clipping"

    torch.nn.utils.clip_grad_norm_(det_model.parameters(), max_norm)
    clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
        det_model.parameters(), float("inf")
    )

    assert (
        clipped_grad_norm <= max_norm + 1e-5
    ), "Gradient clipping did not limit gradient norm"
    assert clipped_grad_norm == pytest.approx(max_norm, rel=1e-4)

    # Clipping rescales, it does not change the gradient direction
    scale = (max_norm / initial_grad_norm).item()
    for g0, p in zip(initial_grads, det_model.parameters()):
        torch.testing.assert_close(p.grad, g0 * scale, rtol=1e-4, atol=1e-8)


def test_dropout(model):
    assert model.config.dropout > 0
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
    norms = [
        m for m in model.modules() if isinstance(m, (torch.nn.LayerNorm, LayerNorm))
    ]
    # norm_1 and norm_2 in every block (llamlam LayerNorm) plus ln_f (torch)
    assert len(norms) == 2 * model.config.n_layers + 1

    for module in norms:
        dim = module.weight.shape[0]
        x = 3 * torch.randn(4, 16, dim) + 5
        output = module(x)

        assert torch.allclose(
            output.mean(dim=-1), torch.zeros(4, 16), atol=1e-5
        ), "LayerNorm output mean is not close to zero"
        # LayerNorm uses the biased variance; eps=1e-5 lowers the std by ~1e-6
        assert torch.allclose(
            output.std(dim=-1, unbiased=False), torch.ones(4, 16), atol=1e-4
        ), "LayerNorm output std is not close to one"


def test_weight_initialization():
    """Parameters follow the init scheme in llamlam/model.py (alpha = 0.5).

    embed ~ N(0, (3.3 alpha)^2), head ~ N(0, (alpha / d)^2),
    attention and feed-forward weights ~ N(0, alpha^2 / d) for d = dim_embd,
    norm gains = 1, biases = 0, pos_embed = 0.
    """
    model = GPTModel(small_config(bias=True))  # bias=True covers the bias branch
    alpha, d = 0.5, model.config.dim_embd
    expected_std = {
        "embed.weight": 3.3 * alpha,
        "head.weight": alpha / d,
    }

    for name, param in model.named_parameters():
        is_norm = ".norm_" in name or name.startswith("ln_f.")
        if name.endswith(".bias"):
            assert torch.all(param == 0), f"Bias {name} is not zero"
        elif is_norm:
            assert torch.all(param == 1), f"Norm gain {name} is not one"
        elif name == "pos_embed":
            assert torch.all(param == 0), "pos_embed is not zero"
        else:
            assert name.endswith(".weight"), f"Unexpected parameter {name}"
            std = expected_std.get(name, alpha / d**0.5)
            assert param.std().item() == pytest.approx(
                std, rel=0.15
            ), f"Weight {name} has std {param.std().item():.4f}, expected {std:.4f}"
            # mean within 5 standard errors of zero
            assert (
                param.mean().abs() < 5 * std / param.numel() ** 0.5
            ), f"Weight {name} has high mean"


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
