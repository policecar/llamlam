"""
build_optimizer: optimizer selection, parameter routing and weight decay,
learning rates under the LR scheduler, and real steps on tiny models.
"""

import copy

import pytest
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch.optim.adamw import AdamW
from transformers import get_scheduler

from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel
from llamlam.opt import GrokAdamW, Muon
from llamlam.utils import (
    build_optimizer,
    current_learning_rates,
    get_grouped_params,
    muon_param_split,
)

MODELS = [GPTModel, DiffTransformer]
OPTIMIZER_CLASSES = {"adamw": AdamW, "grokadamw": GrokAdamW, "muon": Muon}


def tiny_config(**overrides):
    params = dict(
        n_layers=2,
        n_heads=2,
        dim_head=8,
        max_seq_length=16,
        vocab_size=64,
        dropout=0.0,
        bias=True,
        learning_rate=3e-3,
        weight_decay=0.1,
    )
    params.update(overrides)
    return Config(**params)


def tiny_model(cls, **overrides):
    torch.manual_seed(0)
    return cls(tiny_config(**overrides))


def names(model, params):
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {by_id[id(p)] for p in params}


def data():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 64, (4, 12), generator=g)


##########################################
# Selection
##########################################


@pytest.mark.parametrize("name, cls", OPTIMIZER_CLASSES.items())
def test_builds_named_class(name, cls):
    model = tiny_model(GPTModel, optimizer=name)
    optimizer = build_optimizer(model, model.config)
    assert type(optimizer) is cls
    # every trainable parameter is optimized exactly once
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


@pytest.mark.parametrize("name", ["sgd", "AdamW", "", None])
def test_unknown_optimizer_raises(name):
    model = tiny_model(GPTModel, optimizer=name)
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer(model, model.config)


@pytest.mark.parametrize("name", ["adamw", "grokadamw"])
def test_adam_variants_use_grouped_params(name):
    model = tiny_model(DiffTransformer, optimizer=name)
    optimizer = build_optimizer(model, model.config)
    expected = get_grouped_params(model, weight_decay=0.1)
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.1, 0.0]
    for group, want in zip(optimizer.param_groups, expected):
        assert [id(p) for p in group["params"]] == [id(p) for p in want["params"]]


@pytest.mark.parametrize(
    "clipping, expected", [(1.0, 1.0), (0.5, 0.5), (0, 0.0), (None, 0.0)]
)
def test_grokadamw_per_tensor_clipping_follows_config(clipping, expected):
    model = tiny_model(GPTModel, optimizer="grokadamw", gradient_clipping=clipping)
    optimizer = build_optimizer(model, model.config)
    assert all(g["gradient_clipping"] == expected for g in optimizer.param_groups)


@pytest.mark.parametrize("name", OPTIMIZER_CLASSES)
def test_negative_gradient_clipping_raises_for_every_optimizer(name):
    # a negative max norm makes clip_grad_norm_'s coefficient negative, which
    # flips every gradient under AdamW and Muon; 0 and None disable clipping
    for clipping in (0, None):
        build_optimizer(
            tiny_model(GPTModel),
            tiny_config(optimizer=name, gradient_clipping=clipping),
        )
    with pytest.raises(ValueError, match="gradient_clipping must be >= 0"):
        build_optimizer(
            tiny_model(GPTModel), tiny_config(optimizer=name, gradient_clipping=-1.0)
        )


##########################################
# Muon routing
##########################################


@pytest.mark.parametrize("tie", [True, False])
def test_muon_split_gpt(tie):
    model = tiny_model(GPTModel, tie_word_embeddings=tie)
    muon, adamw = muon_param_split(model)
    expected_muon = {
        f"blocks.{i}.{n}.weight"
        for i in range(2)
        for n in (
            "context.qkv_proj",
            "context.out_proj",
            "feedforward.0",
            "feedforward.2",
        )
    }
    assert names(model, muon) == expected_muon
    all_names = {n for n, _ in model.named_parameters()}
    assert names(model, adamw) == all_names - expected_muon
    assert {"embed.weight", "pos_embed", "ln_f.weight"} <= names(model, adamw)
    # the head goes to AdamW; tied, it is the embedding matrix, listed once
    assert sum(p is model.head.weight for p in adamw) == 1
    assert ("head.weight" in all_names) is not tie
    assert (model.head.weight is model.embed.weight) is tie


@pytest.mark.parametrize("tie", [True, False])
def test_muon_split_difftransformer(tie):
    model = tiny_model(DiffTransformer, tie_word_embeddings=tie)
    muon, adamw = muon_param_split(model)
    expected_muon = {
        f"layers.{i}.{n}.weight"
        for i in range(2)
        for n in (
            "attn.W_q",
            "attn.W_k",
            "attn.W_v",
            "attn.W_o",
            "ff.WG",
            "ff.W1",
            "ff.W2",
        )
    }
    assert names(model, muon) == expected_muon
    all_names = {n for n, _ in model.named_parameters()}
    assert names(model, adamw) == all_names - expected_muon
    assert all(p.ndim == 2 for p in muon)
    assert sum(p is model.head.weight for p in adamw) == 1
    assert (model.head.weight is model.token_emb.weight) is tie


def test_muon_flags_match_split():
    model = tiny_model(DiffTransformer, optimizer="muon")
    optimizer = build_optimizer(model, model.config)
    muon, adamw = muon_param_split(model)
    assert all(optimizer.state[p]["use_muon"] for p in muon)
    assert not any(optimizer.state[p]["use_muon"] for p in adamw)


def test_muon_split_tied_head_goes_to_adamw_once():
    model = tiny_model(GPTModel)
    model.head.weight = model.embed.weight
    muon, adamw = muon_param_split(model)
    assert sum(p is model.embed.weight for p in adamw) == 1
    assert not any(p is model.embed.weight for p in muon)


def test_muon_requires_a_head():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="model.head"):
        muon_param_split(model)


##########################################
# Weight decay: decided by get_grouped_params for every optimizer
##########################################


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("name", OPTIMIZER_CLASSES)
def test_weight_decay_applies_exactly_to_decay_group(cls, name):
    # With zero gradients every update term is 0, so a step changes a parameter
    # only by decoupled weight decay: p *= 1 - lr * wd.
    model = tiny_model(cls, optimizer=name)
    optimizer = build_optimizer(model, model.config)
    decayed = {id(p) for p in get_grouped_params(model, 0.1)[0]["params"]}
    muon_ids = {id(p) for p in muon_param_split(model)[0]} if name == "muon" else set()
    before = copy.deepcopy(model)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()

    lr = model.config.learning_rate
    for (n, p), (_, p0) in zip(model.named_parameters(), before.named_parameters()):
        if id(p) in decayed and id(p) not in muon_ids:  # Muon has no weight decay
            # Compare the decrement p0 - p = p0 * lr * wd (3e-4 * p0), not p: the
            # default tolerance on p (atol 1e-5) exceeds the whole decay of a
            # 0.02-scale weight, so wd 0.08 passed. In float32, rounding 1 - 3e-4
            # and then p0 * (1 - 3e-4) costs <= 2 * 2**-24 * |p0| = 1.2e-7 * |p0|,
            # and p0 - p is exact, so the decrement is off by <= 4e-4 of itself
            # (2.3e-4 measured); rtol 1e-2 leaves 25x margin, fails a 5% error.
            torch.testing.assert_close(p0 - p, p0 * lr * 0.1, rtol=1e-2, atol=0, msg=n)
        else:
            torch.testing.assert_close(p, p0, rtol=0, atol=0, msg=n)


##########################################
# Learning rates and the LR scheduler
##########################################


def effective_lrs(optimizer):
    """The step size each parameter group actually uses, per path."""
    if isinstance(optimizer, Muon):
        return [
            (g["lr"], g["adamw_lr_ratio"] * g["lr"]) for g in optimizer.param_groups
        ]
    return [(g["lr"],) for g in optimizer.param_groups]


def base_lrs(config):
    """Per-group base rates: (Muon path, AdamW path) for Muon, (lr,) otherwise."""
    if config.optimizer == "muon":
        return (config.muon_learning_rate, config.learning_rate)
    return (config.learning_rate,)


@pytest.mark.parametrize("name", OPTIMIZER_CLASSES)
def test_learning_rates_follow_config_and_scheduler(name):
    # muon_learning_rate differs from both defaults, so a mix-up shows
    model = tiny_model(GPTModel, optimizer=name, muon_learning_rate=0.05)
    optimizer = build_optimizer(model, model.config)
    base = base_lrs(model.config)
    for lrs in effective_lrs(optimizer):
        assert lrs == pytest.approx(base, rel=1e-12)

    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=2, num_training_steps=6
    )
    factors = [0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0]  # warmup 2, linear decay to 6
    for step, factor in enumerate(factors):
        for lrs in effective_lrs(optimizer):
            expected = tuple(b * factor for b in base)
            assert lrs == pytest.approx(expected, rel=1e-12, abs=1e-15), (step, lrs)
        optimizer.step()
        scheduler.step()


def test_muon_default_rates_and_paths():
    """At the defaults Muon steps hidden matrices at 0.02, the rest at 6e-4."""
    assert Config().muon_learning_rate == 0.02
    assert Config().learning_rate == 6e-4
    model = tiny_model(
        GPTModel, optimizer="muon", learning_rate=6e-4, muon_learning_rate=0.02
    )
    optimizer = build_optimizer(model, model.config)
    assert len(optimizer.param_groups) == 2  # decayed / non-decayed AdamW params
    for g in optimizer.param_groups:
        assert g["lr"] == 0.02
        assert g["adamw_lr_ratio"] * g["lr"] == pytest.approx(6e-4, rel=1e-12)


def test_muon_rates_reach_the_parameters():
    """One step with zero momentum history: Muon-path update size scales with
    muon_learning_rate, AdamW-path update size with learning_rate, independently."""

    def deltas(lr, muon_lr):
        model = tiny_model(
            GPTModel,
            optimizer="muon",
            learning_rate=lr,
            muon_learning_rate=muon_lr,
            weight_decay=0.0,
        )
        optimizer = build_optimizer(model, model.config)
        before = copy.deepcopy(model)
        model(data())["loss"].backward()
        optimizer.step()
        return {
            n: (p - p0).abs().max().item()
            for (n, p), (_, p0) in zip(
                model.named_parameters(), before.named_parameters()
            )
        }

    base = deltas(1e-3, 0.02)
    more_muon = deltas(1e-3, 0.04)
    more_adamw = deltas(2e-3, 0.02)
    reference = tiny_model(GPTModel)
    muon_names = names(reference, muon_param_split(reference)[0])
    assert muon_names  # both paths are exercised
    for n, d in base.items():
        assert d > 0, n
        if n in muon_names:
            assert more_muon[n] == pytest.approx(2 * d, rel=1e-2), n
            assert more_adamw[n] == pytest.approx(d, rel=1e-2), n
        else:
            assert more_adamw[n] == pytest.approx(2 * d, rel=1e-2), n
            assert more_muon[n] == pytest.approx(d, rel=1e-2), n


@pytest.mark.parametrize("name", ["adamw", "grokadamw"])
def test_muon_learning_rate_has_no_effect_on_other_optimizers(name):
    groups = []
    for muon_lr in (0.02, 0.5):
        model = tiny_model(GPTModel, optimizer=name, muon_learning_rate=muon_lr)
        optimizer = build_optimizer(model, model.config)
        groups.append(
            [
                {k: v for k, v in g.items() if k != "params"}
                for g in optimizer.param_groups
            ]
        )
    assert groups[0] == groups[1]
    assert all(g["lr"] == model.config.learning_rate for g in groups[0])


@pytest.mark.parametrize("name", OPTIMIZER_CLASSES)
def test_current_learning_rates_names_each_path(name):
    model = tiny_model(GPTModel, optimizer=name, muon_learning_rate=0.05)
    optimizer = build_optimizer(model, model.config)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=2, num_training_steps=6
    )
    optimizer.step()
    scheduler.step()  # warmup factor 0.5
    wrapped = Accelerator(cpu=True).prepare(optimizer)
    assert isinstance(wrapped, AcceleratedOptimizer)
    for opt in (optimizer, wrapped):
        rates = current_learning_rates(opt)
        if name == "muon":
            assert set(rates) == {"learning_rate", "muon_learning_rate"}
            assert rates["muon_learning_rate"] == pytest.approx(0.5 * 0.05)
        else:
            assert set(rates) == {"learning_rate"}
        assert rates["learning_rate"] == pytest.approx(0.5 * 3e-3)


##########################################
# Real steps reduce the loss
##########################################


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("name", OPTIMIZER_CLASSES)
def test_optimizer_reduces_loss(cls, name):
    model = tiny_model(cls, optimizer=name)
    optimizer = build_optimizer(model, model.config)
    x = data()
    with torch.no_grad():
        initial = model(x)["loss"].item()
    for _ in range(10):
        optimizer.zero_grad()
        model(x)["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        final = model(x)["loss"].item()
    assert (
        final < initial - 0.05
    ), f"{name}/{cls.__name__}: {initial:.3f} -> {final:.3f}"
    # every parameter moved (no group silently left out of the update)
    stepped = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(stepped) == len(list(model.parameters()))
