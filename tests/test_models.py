"""Architecture, forward pass, and training-dynamics tests for both models.

Generic behavior is parametrized over GPTModel and DiffTransformer via the
`model` fixture; structure tests that depend on a specific module layout are
kept per-model.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer
from llamlam.model import Block, Context, GPTModel
from llamlam.utils import set_seed


# --------------------------------------------------------------------------- #
# Forward pass / shapes / loss (both models)
# --------------------------------------------------------------------------- #
def test_forward_shapes(model, batch):
    model.eval()
    with torch.no_grad():
        out = model(batch["input_ids"])
    b, t = batch["input_ids"].shape
    assert out["logits"].shape == (b, t, model.config.vocab_size)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])


def test_loss_matches_manual(model, batch):
    model.eval()
    with torch.no_grad():
        out = model(batch["input_ids"], labels=batch["labels"])
    logits = out["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    manual = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out["loss"], manual, atol=1e-5)


def test_hidden_states_output(model, batch):
    out = model(batch["input_ids"], output_hidden_states=True)
    hs = out["hidden_states"]
    assert len(hs) == model.config.n_layers + 1  # embeddings + one per layer
    b, t = batch["input_ids"].shape
    for h in hs:
        assert h.shape == (b, t, model.config.dim_embd)


def test_reproducible_init(model_name, config):
    set_seed(42)
    m1 = {"gpt": GPTModel, "diff": DiffTransformer}[model_name](config)
    set_seed(42)
    m2 = {"gpt": GPTModel, "diff": DiffTransformer}[model_name](config)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_save_and_load(model, config, tmp_path):
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    reloaded = type(model)(config)
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    for p1, p2 in zip(model.parameters(), reloaded.parameters()):
        assert torch.allclose(p1, p2)


# --------------------------------------------------------------------------- #
# Training dynamics (both models)
# --------------------------------------------------------------------------- #
def test_gradient_flow_is_finite_and_nonzero(model, batch):
    model.train()
    model.zero_grad()
    model(batch["input_ids"], labels=batch["labels"])["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"


def test_lr_step_reduces_loss(model, batch):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = model(batch["input_ids"], labels=batch["labels"])["loss"].item()
    for _ in range(10):
        opt.zero_grad()
        loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
        loss.backward()
        opt.step()
    assert model(batch["input_ids"], labels=batch["labels"])["loss"].item() < initial


@pytest.mark.slow
def test_overfit_single_batch(model, batch):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    initial = model(batch["input_ids"], labels=batch["labels"])["loss"].item()
    for _ in range(600):
        opt.zero_grad()
        loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
        loss.backward()
        opt.step()
    final = model(batch["input_ids"], labels=batch["labels"])["loss"].item()
    assert final < initial
    assert final < 0.5


def test_beats_uniform_baseline(model, batch):
    """An initialized model's loss should be no worse than log(vocab_size)."""
    model.eval()
    with torch.no_grad():
        loss = model(batch["input_ids"], labels=batch["labels"])["loss"].item()
    uniform = torch.log(torch.tensor(float(model.config.vocab_size))).item()
    assert loss <= uniform + 0.5  # near uniform at init, not wildly above


# --------------------------------------------------------------------------- #
# GPTModel-specific structure
# --------------------------------------------------------------------------- #
def test_gpt_structure(config):
    model = GPTModel(config)
    assert isinstance(model.embed, nn.Embedding)
    assert model.embed.weight.shape == (config.vocab_size, config.dim_embd)
    assert len(model.blocks) == config.n_layers
    assert isinstance(model.blocks[0], Block)
    assert isinstance(model.head, nn.Linear)
    assert model.head.weight.shape == (config.vocab_size, config.dim_embd)


def test_gpt_weight_tying(config):
    m = GPTModel(dataclasses.replace(config, tie_word_embeddings=True))
    assert m.head.weight is m.embed.weight

    m2 = GPTModel(dataclasses.replace(config, tie_word_embeddings=False))
    assert m2.head.weight is not m2.embed.weight


def test_gpt_init_std_is_sane(config):
    """Init should be ~init_std, not the old ~1.65 embedding blow-up."""
    m = GPTModel(config)
    assert m.embed.weight.std().item() < 5 * config.init_std


def test_context_output_shape(config):
    ctx = Context(config.dim_embd, config.n_heads)
    x = torch.randn(4, 16, config.dim_embd)
    assert ctx(x).shape == (4, 16, config.dim_embd)


def test_gpt_dropout_active_in_train_only(config):
    model = GPTModel(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids)["logits"], model(ids)["logits"])
    model.train()
    assert not torch.allclose(model(ids)["logits"], model(ids)["logits"])


def test_gpt_layernorm_normalizes(config):
    model = GPTModel(config)
    x = torch.randn(4, 16, config.dim_embd)
    out = model.ln_f(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4, 16), atol=1e-5)


# --------------------------------------------------------------------------- #
# DiffTransformer-specific structure
# --------------------------------------------------------------------------- #
def test_diff_structure(config):
    model = DiffTransformer(config)
    assert isinstance(model.token_emb, nn.Embedding)
    assert len(model.layers) == config.n_layers
    assert model.head.weight.shape == (config.vocab_size, config.dim_embd)
