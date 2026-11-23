"""
Regression tests for DiffTransformer review fixes: lambda init, reference
layout / parameter parity with GPTModel, residual dropout, __main__ demo.
"""

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer, MultiHeadDifferentialAttention
from llamlam.model import GPTModel

REPO_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_PREFIXES = ("embed.", "pos_embed", "token_emb.", "pos_emb.", "head.")


def non_embedding_params(model):
    return sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith(EMBEDDING_PREFIXES)
    )


def small_config(**overrides):
    kwargs = dict(
        max_seq_length=16,
        vocab_size=50,
        n_layers=2,
        n_heads=2,
        dim_head=16,
        dropout=0.0,
        bias=False,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


# --------------------------------------------------------------------------- #
# Item 2: lambda ~= lambda_init at init
# --------------------------------------------------------------------------- #


def test_lambda_near_lambda_init_at_init():
    torch.manual_seed(0)
    config = Config(
        max_seq_length=16, vocab_size=50, n_layers=12, n_heads=12, dim_head=64
    )
    model = DiffTransformer(config)
    for layer in model.layers:
        attn = layer.attn
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            std = getattr(attn, name).std().item()
            assert 0.07 < std < 0.13, f"{name} std {std:.3f}"
        lam = attn.lambda_full()
        assert lam.shape == ()  # one lambda per layer
        # std of (lambda - lambda_init) is ~0.08 here; std-1 init gave >> 1
        deviation = (lam - attn.lambda_init).abs().max().item()
        assert deviation < 0.5, f"lambda deviates from lambda_init by {deviation}"


def test_one_lambda_per_layer_shared_by_all_heads():
    """Reference: lambda_q1/k1/q2/k2 are (head_dim,), so lambda is one scalar."""
    config = Config()
    with torch.device("meta"):
        model = DiffTransformer(config)
    for layer in model.layers:
        attn = layer.attn
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            assert getattr(attn, name).shape == (attn.dim_head,), name
        assert attn.lambda_full().shape == ()
    lambda_params = [n for n, _ in model.named_parameters() if ".attn.lambda_" in n]
    assert len(lambda_params) == 4 * config.n_layers


def test_lambda_init_schedule_matches_reference():
    """Reference: lambda_init = 0.8 - 0.6 * exp(-0.3 * depth), depth from 0."""
    model = DiffTransformer(small_config(n_layers=4))
    for depth, layer in enumerate(model.layers):
        expected = 0.8 - 0.6 * math.exp(-0.3 * depth)
        assert layer.attn.lambda_init == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Item 3: reference layout and parameter parity
# --------------------------------------------------------------------------- #


def test_parameter_parity_with_gpt_at_default_config():
    config = Config()
    with torch.device("meta"):
        gpt = GPTModel(config)
        diff = DiffTransformer(config)
    n_gpt = non_embedding_params(gpt)
    n_diff = non_embedding_params(diff)
    assert abs(n_diff - n_gpt) / n_gpt < 0.02, (n_gpt, n_diff)


def test_reference_projection_layout():
    config = Config()
    d = config.dim_embd
    with torch.device("meta"):
        model = DiffTransformer(config)
    attn = model.layers[0].attn
    # Reference: num_heads = half the baseline's; each softmax map has the
    # width of a baseline head (6 differential heads of 2 x 64 at the default)
    assert attn.n_heads == config.n_heads // 2 == 6
    assert attn.dim_head == d // config.n_heads == config.dim_head == 64
    for proj in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        assert proj.weight.shape == (d, d)
    assert attn.rms_scale.shape == (2 * attn.dim_head,)
    assert attn.lambda_q1.shape == (attn.dim_head,)
    ff = model.layers[0].ff
    assert ff.WG.weight.shape == (8 * d // 3, d)
    assert ff.W1.weight.shape == (8 * d // 3, d)
    assert ff.W2.weight.shape == (d, 8 * d // 3)


@pytest.mark.parametrize("model_cls", [GPTModel, DiffTransformer])
def test_bias_false_gives_no_bias_parameters(model_cls):
    model = model_cls(small_config(bias=False))
    assert [n for n, _ in model.named_parameters() if "bias" in n] == []


@pytest.mark.parametrize("bias", [False, True])
def test_feedforward_bias_follows_config_in_both_models(bias):
    config = small_config(bias=bias)
    gpt_ff = [
        m
        for block in GPTModel(config).blocks
        for m in block.feedforward
        if isinstance(m, nn.Linear)
    ]
    diff_ff = [
        m for layer in DiffTransformer(config).layers for m in layer.ff.children()
    ]
    assert len(gpt_ff) == 2 * config.n_layers and len(diff_ff) == 3 * config.n_layers
    for linear in gpt_ff + diff_ff:
        assert isinstance(linear, nn.Linear)
        assert (linear.bias is not None) == bias


@pytest.mark.parametrize("n_heads", [0, 1, 3, 13])
def test_odd_baseline_n_heads_raises(n_heads):
    with pytest.raises(ValueError, match="n_heads .* must be even"):
        DiffTransformer(small_config(n_heads=n_heads))


def test_attention_dim_embd_not_divisible_by_two_heads_raises():
    # dim_embd = 30 is not divisible by 2 * n_heads = 8 differential maps
    with pytest.raises(ValueError, match="divisible by 2 \\* n_heads"):
        MultiHeadDifferentialAttention(dim_embd=30, n_heads=4, lambda_init=0.5)


def _reference_forward(attn, x):
    """Direct port of unilm multihead_diffattn.py forward (no RoPE, no GQA)."""
    bsz, n, _ = x.shape
    h, hd = attn.n_heads, attn.dim_head
    q = attn.W_q(x).view(bsz, n, 2 * h, hd).transpose(1, 2) * hd**-0.5
    k = attn.W_k(x).view(bsz, n, 2 * h, hd).transpose(1, 2)
    v = attn.W_v(x).view(bsz, n, h, 2 * hd).transpose(1, 2)
    w = q @ k.transpose(-1, -2)
    w = w + torch.triu(torch.full((n, n), float("-inf")), 1)
    w = F.softmax(w, dim=-1).view(bsz, h, 2, n, n)
    # Reference: lambda vectors are (head_dim,) -> one scalar for all heads.
    # Computed from the parameters here, independently of lambda_full().
    for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
        assert getattr(attn, name).shape == (hd,), name
    lam_1 = torch.exp(torch.sum(attn.lambda_q1 * attn.lambda_k1, dim=-1).float())
    lam_2 = torch.exp(torch.sum(attn.lambda_q2 * attn.lambda_k2, dim=-1).float())
    lam = lam_1 - lam_2 + attn.lambda_init
    assert lam.shape == ()
    w = w[:, :, 0] - lam * w[:, :, 1]
    out = w @ v
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + attn.eps)
    out = out * attn.rms_scale * (1 - attn.lambda_init)
    return attn.W_o(out.transpose(1, 2).reshape(bsz, n, 2 * h * hd))


def test_attention_matches_reference_forward():
    torch.manual_seed(0)
    attn = MultiHeadDifferentialAttention(dim_embd=64, n_heads=4, lambda_init=0.5)
    with torch.no_grad():
        # Make lambda vectors non-trivial so a wrong lambda would show
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            getattr(attn, name).normal_(0, 0.5)
        attn.rms_scale.uniform_(0.5, 1.5)
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        torch.testing.assert_close(attn(x), _reference_forward(attn, x))


# --------------------------------------------------------------------------- #
# Item 4: residual dropout
# --------------------------------------------------------------------------- #


def _two_passes(model, input_ids):
    with torch.no_grad():
        return model(input_ids)["logits"], model(input_ids)["logits"]


def test_dropout_active_in_train_mode():
    torch.manual_seed(0)
    model = DiffTransformer(small_config(dropout=0.1))
    input_ids = torch.randint(0, 50, (2, 16))
    model.train()
    a, b = _two_passes(model, input_ids)
    assert not torch.allclose(a, b)
    model.eval()
    a, b = _two_passes(model, input_ids)
    assert torch.equal(a, b)


def test_dropout_zero_is_deterministic_in_train_mode():
    torch.manual_seed(0)
    model = DiffTransformer(small_config(dropout=0.0))
    model.train()
    a, b = _two_passes(model, torch.randint(0, 50, (2, 16)))
    assert torch.equal(a, b)


def test_dropout_sits_on_residual_branches():
    """With p = 1 both residual branches are zeroed, so every layer is identity."""
    torch.manual_seed(0)
    model = DiffTransformer(small_config(dropout=1.0))
    model.train()
    input_ids = torch.randint(0, 50, (2, 16))
    positions = torch.arange(16).unsqueeze(0).expand(2, 16)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        embedded = model.token_emb(input_ids) + model.pos_emb(positions)
        expected = model.head(model.norm(embedded))
    for hidden in out["hidden_states"]:
        assert torch.equal(hidden, embedded)
    torch.testing.assert_close(out["logits"], expected)


# --------------------------------------------------------------------------- #
# Item 5: __main__ demo runs
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_main_demo_runs():
    result = subprocess.run(
        [sys.executable, "-m", "llamlam.difftransformer"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "torch.Size([2, 128, 30522])" in result.stdout
