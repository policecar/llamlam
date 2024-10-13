import pytest
import torch
import torch.nn.functional as F

from llamlam.config import LMConfig
from llamlam.model import GPTModel


@pytest.fixture
def model():
    config = LMConfig(max_length=16, vocab_size=100, n_layer=2, n_head=2, head_width=8)
    return GPTModel(config)


def test_loss_shape_and_type(model):
    input_ids = torch.randint(0, 100, (4, 16))
    output = model(input_ids)

    assert "loss" in output, "Loss not present in model output"
    assert output["loss"].shape == (), "Loss is not a scalar"
    assert output["loss"].dtype == torch.float32, "Loss is not float32"


def test_loss_nonnegativity(model):
    input_ids = torch.randint(0, 100, (4, 16))
    loss = model(input_ids)["loss"]

    assert loss >= 0, "Loss is negative"


def test_loss_reduction(model):
    input_ids = torch.randint(0, 100, (4, 16))
    loss = model(input_ids)["loss"]

    # Calculate loss manually
    logits = model(input_ids)["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    manual_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )

    assert torch.allclose(
        loss, manual_loss
    ), "Model loss doesn't match manually calculated loss"


def test_loss_backpropagation(model):
    input_ids = torch.randint(0, 100, (4, 16))
    loss = model(input_ids)["loss"]

    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "Some parameters don't have gradients"
        assert not torch.allclose(
            param.grad, torch.zeros_like(param.grad)
        ), "Some gradients are zero"


def test_loss_with_padding(model):
    input_ids = torch.randint(0, 100, (4, 16))
    input_ids[:, -4:] = model.config.vocab_size - 1  # Assume last token is padding

    loss_with_padding = model(input_ids)["loss"]

    # Manually mask out padding
    logits = model(input_ids)["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    mask = (shift_labels != model.config.vocab_size - 1).float()
    manual_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    manual_loss = (manual_loss * mask.view(-1)).sum() / mask.sum()

    assert torch.allclose(
        loss_with_padding, manual_loss
    ), "Loss doesn't handle padding correctly"


def test_loss_with_uniform_distribution(model):
    input_ids = torch.randint(0, 100, (4, 16))

    # Set logits to uniform distribution
    uniform_logits = torch.ones(4, 16, model.config.vocab_size)
    model.head.weight.data.zero_()
    model.head.bias.data.zero_()

    loss = F.cross_entropy(
        uniform_logits.view(-1, uniform_logits.size(-1)), input_ids.view(-1)
    )
    expected_loss = torch.log(torch.tensor(model.config.vocab_size, dtype=torch.float))

    assert torch.allclose(
        loss, expected_loss
    ), "Loss for uniform distribution is incorrect"


def test_loss_with_perfect_prediction(model):
    input_ids = torch.randint(0, 100, (4, 16))

    # Set logits to perfect prediction
    perfect_logits = torch.zeros(4, 16, model.config.vocab_size)
    perfect_logits.scatter_(2, input_ids.unsqueeze(-1), 1e9)
    model.head.weight.data.zero_()
    model.head.bias.data = perfect_logits[0, 0]

    loss = model(input_ids)["loss"]

    assert torch.allclose(
        loss, torch.tensor(0.0)
    ), "Loss for perfect prediction is not zero"


def test_loss_gradient_norm(model):
    input_ids = torch.randint(0, 100, (4, 16))
    loss = model(input_ids)["loss"]

    loss.backward()

    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm**0.5

    assert (
        1e-5 < total_norm < 1e5
    ), f"Gradient norm {total_norm} is outside reasonable range"
