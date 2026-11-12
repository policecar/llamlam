import math

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from llamlam.config import Config
from llamlam.model import GPTModel


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def model():
    # dropout=0: loss comparisons across steps must not be dropout noise
    config = Config(
        max_seq_length=16,
        vocab_size=100,
        n_layers=2,
        n_heads=2,
        dim_head=8,
        dropout=0.0,
    )
    return GPTModel(config)


def test_learning_rate_impact(model):
    """Loss after a fixed budget is U-shaped in the learning rate.

    Too small barely moves, a moderate rate learns most, far too large is worse.
    """
    input_ids = torch.randint(0, 100, (4, 16))
    initial_loss = model(input_ids)["loss"].item()

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    losses = []

    for lr in learning_rates:
        model_copy = GPTModel(model.config)
        model_copy.load_state_dict(model.state_dict())
        optimizer = Adam(model_copy.parameters(), lr=lr)

        for _ in range(10):
            optimizer.zero_grad()
            loss = model_copy(input_ids)["loss"]
            loss.backward()
            optimizer.step()

        final_loss = model_copy(input_ids)["loss"].item()
        losses.append(final_loss)

    best = min(range(len(losses)), key=losses.__getitem__)
    assert losses[best] < initial_loss - 1.0, "No learning rate improved the loss"
    assert abs(losses[0] - initial_loss) < 0.05, "A tiny learning rate moved the loss"
    assert 0 < best < len(losses) - 1, (
        f"Best learning rate is at the edge of the sweep: {learning_rates[best]} "
        f"(losses {losses})"
    )
    assert losses[-1] > losses[best] + 1.0, "A far too large learning rate did no harm"


def test_learning_rate_scheduler(model):
    input_ids = torch.randint(0, 100, (4, 16))
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)

    initial_lr = optimizer.param_groups[0]["lr"]
    losses = []

    for epoch in range(10):
        optimizer.zero_grad()
        loss = model(input_ids)["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    final_lr = optimizer.param_groups[0]["lr"]

    assert final_lr < initial_lr, "Learning rate did not decrease"
    assert final_lr == pytest.approx(initial_lr * 0.95**10)
    assert losses[-1] < losses[0], "Loss did not decrease with scheduled learning rate"


def test_extremely_small_learning_rate(model):
    input_ids = torch.randint(0, 100, (4, 16))
    initial_loss = model(input_ids)["loss"].item()
    initial_params = [p.clone() for p in model.parameters()]

    optimizer = Adam(model.parameters(), lr=1e-12)

    for _ in range(100):
        optimizer.zero_grad()
        loss = model(input_ids)["loss"]
        loss.backward()
        optimizer.step()

    final_loss = model(input_ids)["loss"].item()
    final_params = list(model.parameters())

    assert (
        abs(final_loss - initial_loss) < 1e-6
    ), "Loss changed significantly with extremely small learning rate"
    for p1, p2 in zip(initial_params, final_params):
        assert torch.allclose(
            p1, p2, atol=1e-6
        ), "Parameters changed significantly with extremely small learning rate"


def test_extremely_large_learning_rate(model):
    """One Adam step at lr=100 makes the loss diverge.

    The loss stays finite here (logits are computed in float32 and a step of
    ~100 per weight does not overflow), so divergence means "non-finite or
    orders of magnitude above the initial loss".
    """
    input_ids = torch.randint(0, 100, (4, 16))
    initial_loss = model(input_ids)["loss"].item()

    optimizer = Adam(model.parameters(), lr=100)

    optimizer.zero_grad()
    loss = model(input_ids)["loss"]
    loss.backward()
    optimizer.step()

    final_loss = model(input_ids)["loss"].item()

    assert not math.isfinite(final_loss) or final_loss > 10 * initial_loss, (
        f"Loss did not diverge with extremely large learning rate: "
        f"{initial_loss:.3f} -> {final_loss:.3f}"
    )
