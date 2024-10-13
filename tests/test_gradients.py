"""
Check https://github.com/dscamiss/blog/blob/b0a45c81f0c91a378b51cd16e4b8275b98e1175e/content/posts/attention-explicit.md?plain=1#L428
"""

import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from llamlam.config import LMConfig
from llamlam.model import GPTModel, Attention, Block


@pytest.fixture
def small_config():
    return LMConfig(max_length=16, vocab_size=100, n_layer=2, n_head=2, head_width=8)


@pytest.fixture
def model(small_config):
    return GPTModel(small_config)


def test_attention_gradient(small_config):
    attention = Attention(small_config.n_embd, small_config.n_head)
    attention.double()  # Convert to double precision for numerical stability

    input = torch.randn(
        2, 16, small_config.n_embd, dtype=torch.double, requires_grad=True
    )

    assert gradcheck(attention, (input,), eps=1e-6, atol=1e-4)


def test_block_gradient(small_config):
    block = Block(small_config)
    block.double()

    input = torch.randn(
        2, 16, small_config.n_embd, dtype=torch.double, requires_grad=True
    )

    assert gradcheck(block, (input,), eps=1e-6, atol=1e-4)


def test_model_gradient(model):
    model.double()
    input_ids = torch.randint(0, 100, (2, 16), dtype=torch.long)

    def loss_fn(model, input_ids):
        output = model(input_ids)
        return output["loss"]

    input_ids.requires_grad_(True)
    assert gradcheck(lambda x: loss_fn(model, x), (input_ids,), eps=1e-6, atol=1e-4)


def test_gradient_magnitude(model):
    input_ids = torch.randint(0, 100, (4, 16))
    output = model(input_ids)
    loss = output["loss"]
    loss.backward()

    for name, param in model.named_parameters():
        grad_norm = param.grad.norm()
        assert (
            1e-6 < grad_norm < 1e3
        ), f"Unusual gradient magnitude for {name}: {grad_norm}"


def test_learning_rate_sensitivity(model):
    input_ids = torch.randint(0, 100, (4, 16))
    initial_loss = model(input_ids)["loss"].item()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # type: ignore

    for _ in range(10):
        optimizer.zero_grad()
        loss = model(input_ids)["loss"]
        loss.backward()
        optimizer.step()

    final_loss = model(input_ids)["loss"].item()
    assert final_loss < initial_loss, "Loss did not decrease after training"


def test_gradient_accumulation(model):
    input_ids = torch.randint(0, 100, (8, 16))

    # Single large batch
    large_batch_output = model(input_ids)
    large_batch_loss = large_batch_output["loss"]
    large_batch_loss.backward()
    large_batch_grads = [param.grad.clone() for param in model.parameters()]

    # Reset gradients
    model.zero_grad()

    # Accumulate gradients from smaller batches
    for i in range(0, 8, 4):
        small_batch_output = model(input_ids[i : i + 4])
        small_batch_loss = (
            small_batch_output["loss"] / 2
        )  # Divide by 2 to match the large batch
        small_batch_loss.backward()

    # Compare gradients
    for large_grad, param in zip(large_batch_grads, model.parameters()):
        assert torch.allclose(large_grad, param.grad, atol=1e-5)


def test_gradient_clipping(model):
    input_ids = torch.randint(0, 100, (4, 16))
    output = model(input_ids)
    loss = output["loss"]
    loss.backward()

    # Get the original gradient norm
    original_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    # Reset gradients
    model.zero_grad()
    loss.backward()

    # Clip gradients
    max_norm = 1.0
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # Check if the new gradient norm is at most max_norm
    new_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    assert (
        new_norm <= max_norm + 1e-6
    ), f"Gradient norm {new_norm} exceeds max_norm {max_norm}"

    # If original norm was smaller than max_norm, it shouldn't have changed
    if original_norm < max_norm:
        assert torch.isclose(original_norm, new_norm, atol=1e-6)
