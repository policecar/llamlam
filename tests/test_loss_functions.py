import math

import pytest
import torch
import torch.nn.functional as F

from llamlam.config import Config
from llamlam.model import GPTModel


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def model():
    # dropout=0: tests compare the loss with logits from a separate forward pass
    config = Config(
        max_seq_length=16,
        vocab_size=100,
        n_layers=2,
        n_heads=2,
        dim_head=8,
        dropout=0.0,
    )
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
    """Targets whose attention_mask entry is 0 are excluded from the mean loss."""
    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)
    attention_mask[:, -4:] = 0  # right padding
    attention_mask[1, -7:] = 0  # sequences of different lengths

    loss_with_padding = model(input_ids, attention_mask=attention_mask)["loss"]

    # Manually mask out padding: target t is dropped iff attention_mask[t] == 0
    logits = model(input_ids)["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    mask = attention_mask[..., 1:].float()
    manual_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    manual_loss = (manual_loss * mask.view(-1)).sum() / mask.sum()

    assert torch.allclose(
        loss_with_padding, manual_loss
    ), "Loss doesn't handle padding correctly"
    assert not torch.allclose(
        loss_with_padding, model(input_ids)["loss"]
    ), "Padding mask had no effect on the loss"


def test_loss_with_uniform_distribution(model):
    """A head that outputs constant logits gives a loss of exactly ln(vocab_size)."""
    input_ids = torch.randint(0, 100, (4, 16))

    # Zero the head (it has no bias): every logit is 0, i.e. uniform predictions
    with torch.no_grad():
        model.head.weight.zero_()

    output = model(input_ids)
    assert torch.equal(output["logits"], torch.zeros_like(output["logits"]))

    expected_loss = torch.tensor(math.log(model.config.vocab_size))
    assert torch.allclose(
        output["loss"], expected_loss
    ), "Loss for uniform distribution is incorrect"


def test_loss_with_perfect_prediction(model):
    """Logits that put all mass on the true next token give a loss of zero.

    The head's output is replaced by one-hot logits for input_ids[:, t + 1] at
    position t, so this also fails if the loss aligns logits and labels wrongly.
    """
    input_ids = torch.randint(0, 100, (4, 16))
    vocab_size = model.config.vocab_size

    perfect_logits = torch.zeros(4, 16, vocab_size)
    perfect_logits[:, :-1].scatter_(2, input_ids[:, 1:].unsqueeze(-1), 1e4)
    handle = model.head.register_forward_hook(lambda m, i, o: perfect_logits)
    try:
        loss = model(input_ids)["loss"]
    finally:
        handle.remove()

    assert torch.allclose(
        loss, torch.tensor(0.0)
    ), f"Loss for perfect prediction is not zero: {loss.item()}"


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
