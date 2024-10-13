import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from llamlam.config import LMConfig
from llamlam.model import GPTModel


@pytest.fixture
def model():
    config = LMConfig(max_length=16, vocab_size=100, n_layer=2, n_head=2, head_width=8)
    return GPTModel(config)


def test_learning_rate_impact(model):
    input_ids = torch.randint(0, 100, (4, 16))
    initial_loss = model(input_ids)["loss"].item()

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
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

    assert min(losses) < initial_loss, "No learning rate improved the loss"
    assert losses != sorted(losses), "Loss decreased monotonically with learning rate"


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
    input_ids = torch.randint(0, 100, (4, 16))
    initial_loss = model(input_ids)["loss"].item()

    optimizer = Adam(model.parameters(), lr=100)

    optimizer.zero_grad()
    loss = model(input_ids)["loss"]
    loss.backward()
    optimizer.step()

    final_loss = model(input_ids)["loss"].item()

    assert not torch.isfinite(
        final_loss
    ), "Loss remained finite with extremely large learning rate"
