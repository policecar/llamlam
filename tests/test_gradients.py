"""
Check https://github.com/dscamiss/blog/blob/b0a45c81f0c91a378b51cd16e4b8275b98e1175e/content/posts/attention-explicit.md?plain=1#L428
"""

import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck
from torch.func import functional_call

from llamlam.config import Config
from llamlam.model import Block, Context, GPTModel


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def small_config():
    # dropout=0: gradcheck and gradient comparisons need a deterministic forward
    return Config(
        max_seq_length=16,
        vocab_size=100,
        n_layers=2,
        n_heads=2,
        dim_head=8,
        dropout=0.0,
    )


@pytest.fixture
def model(small_config):
    return GPTModel(small_config)


def test_attention_gradient(small_config):
    attention = Context(small_config.dim_embd, small_config.n_heads)
    attention.double()  # Convert to double precision for numerical stability

    input = torch.randn(
        2, 16, small_config.dim_embd, dtype=torch.double, requires_grad=True
    )

    assert gradcheck(attention, (input,), eps=1e-6, atol=1e-4)


def test_block_gradient(small_config):
    block = Block(small_config)
    block.double()

    input = torch.randn(
        2, 16, small_config.dim_embd, dtype=torch.double, requires_grad=True
    )

    assert gradcheck(block, (input,), eps=1e-6, atol=1e-4)


def test_model_gradient(model):
    """Analytic gradients through the whole stack of blocks match finite differences.

    Token ids are integers and cannot be differentiated, so the check is taken
    w.r.t. the positional embedding, which enters before the first block. The
    checked output is the hidden state after the last block: GPTModel casts its
    logits (and so the loss) to float32 with .float(), which is too coarse for a
    double-precision finite-difference check.
    """
    model.double()
    input_ids = torch.randint(0, 100, (2, 16), dtype=torch.long)
    pos_embed = torch.randn_like(model.pos_embed).requires_grad_(True)

    def last_hidden_state(pos_embed):
        outputs = functional_call(
            model,
            {"pos_embed": pos_embed},
            (input_ids,),
            {"output_hidden_states": True},
        )
        return outputs["hidden_states"][-1]

    assert last_hidden_state(pos_embed).dtype == torch.double
    assert gradcheck(last_hidden_state, (pos_embed,), eps=1e-6, atol=1e-4)


def test_gradient_magnitude(model):
    input_ids = torch.randint(0, 100, (4, 16))
    output = model(input_ids)
    loss = output["loss"]
    loss.backward()

    for name, param in model.named_parameters():
        grad_norm = param.grad.norm()
        assert 1e-6 < grad_norm < 1e3, (
            f"Unusual gradient magnitude for {name}: {grad_norm}"
        )


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
    model(input_ids)["loss"].backward()
    original_grads = [p.grad.clone() for p in model.parameters()]

    # Get the original gradient norm (max_norm=inf never rescales)
    original_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    # Clip to half the original norm so that clipping actually takes effect
    max_norm = original_norm.item() / 2
    returned_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    assert torch.isclose(returned_norm, original_norm), "returns the pre-clip norm"

    # The clipped gradient norm is exactly max_norm ...
    new_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    assert torch.isclose(new_norm, torch.tensor(max_norm), rtol=1e-4), (
        f"Gradient norm {new_norm} is not max_norm {max_norm}"
    )

    # ... reached by rescaling every gradient by the same factor
    for g0, p in zip(original_grads, model.parameters()):
        torch.testing.assert_close(p.grad, g0 * 0.5, rtol=1e-4, atol=1e-8)

    # A max_norm above the gradient norm leaves the gradients unchanged
    before = [p.grad.clone() for p in model.parameters()]
    nn.utils.clip_grad_norm_(model.parameters(), 10 * max_norm)
    for g0, p in zip(before, model.parameters()):
        torch.testing.assert_close(p.grad, g0)
