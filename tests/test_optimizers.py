"""The custom optimizers should run on CPU and reduce the loss.

Regression coverage for: GrokAdamW's group['state'] KeyError + CUDA-only state,
and Muon's hardcoded 'cuda' device.
"""

import pytest
import torch

from llamlam.config import Config
from llamlam.model import GPTModel
from llamlam.utils import build_optimizer


@pytest.fixture
def model():
    cfg = Config(
        max_seq_length=16, vocab_size=64, n_layers=2, n_heads=2, dim_head=8, dropout=0.0
    )
    torch.manual_seed(0)
    return GPTModel(cfg)


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randint(0, 64, (2, 12))


@pytest.mark.parametrize("optimizer", ["adamw", "grokadamw", "muon"])
def test_optimizer_reduces_loss(model, data, optimizer):
    model.config.optimizer = optimizer
    opt = build_optimizer(model, model.config)

    initial = model(data)["loss"].item()
    for _ in range(5):
        opt.zero_grad()
        loss = model(data)["loss"]
        loss.backward()
        opt.step()
    final = model(data)["loss"].item()

    assert final < initial, f"{optimizer}: {initial:.3f} -> {final:.3f}"


def test_unknown_optimizer_raises(model):
    model.config.optimizer = "nope"
    with pytest.raises(ValueError):
        build_optimizer(model, model.config)


def test_weight_decay_shrinks_params(model, data):
    """With everything else equal, weight decay yields smaller final norms."""
    import copy

    def final_norm(weight_decay):
        m = copy.deepcopy(model)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=weight_decay)
        for _ in range(50):
            opt.zero_grad()
            m(data)["loss"].backward()
            opt.step()
        return sum(p.norm().item() for p in m.parameters())

    assert final_norm(0.5) < final_norm(0.0)


def test_gradient_clipping_bounds_norm(model, data):
    model(data)["loss"].backward()
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    # The returned value is the *pre-clip* norm; re-measure post-clip.
    post = (
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    )
    assert post <= 0.1 + 1e-4
    assert clipped >= post
