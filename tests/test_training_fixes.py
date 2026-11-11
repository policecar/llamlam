"""
Training-loop fixes from the code review: padding masks in train/eval (item 1),
gradient clipping (2), unscaled train-loss logging (3), gradient accumulation (4),
best/last checkpoints (5, see also tests/test_checkpointing.py), weight-decay
groups (7), sampling via unwrap (10).
"""

import copy
import math
from types import SimpleNamespace

import accelerate.scheduler
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from safetensors.torch import load_file
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import get_scheduler

from llamlam.checkpoint import CheckpointManager
from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel
from llamlam.train import (
    accumulation_group_size,
    is_accumulation_boundary,
    lr_schedule_steps,
    num_training_steps,
    run_training,
    sample_text,
)
from llamlam.utils import evaluate, get_grouped_params

SEQ_LEN = 12


def tiny_config(**overrides):
    params = dict(
        n_layers=1,
        n_heads=2,
        dim_head=8,
        max_seq_length=16,
        vocab_size=50,
        dropout=0.0,
        bias=True,
        n_epochs=1,
        eval_steps=10**6,  # only the step-0 and end-of-epoch evaluations
        gradient_accumulation_steps=1,
        gradient_clipping=0,
    )
    params.update(overrides)
    return Config(**params)


def tiny_model(config, cls=DiffTransformer, seed=0):
    torch.manual_seed(seed)
    return cls(config)


def make_batches(n, config, batch_size=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(
                0, config.vocab_size, (batch_size, SEQ_LEN), generator=g
            ),
            "attention_mask": torch.ones(batch_size, SEQ_LEN, dtype=torch.long),
        }
        for _ in range(n)
    ]


def pad_batch(input_ids, total_len, pad_id=0):
    """Right-pad [B, L] input_ids to total_len, returning a collator-style batch."""
    b, n = input_ids.shape
    padded = torch.full((b, total_len), pad_id, dtype=input_ids.dtype)
    padded[:, :n] = input_ids
    mask = torch.zeros(b, total_len, dtype=torch.long)
    mask[:, :n] = 1
    return {"input_ids": padded, "attention_mask": mask}


@pytest.fixture
def accelerator():
    return Accelerator(cpu=True)


def train(model, optimizer, batches, config, run_dir, accelerator, log=None, val=None):
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    result = run_training(
        model,
        optimizer,
        scheduler,
        batches,
        val if val is not None else batches[:1],
        accelerator,
        config,
        run_dir,
        log=log,
    )
    return result, scheduler


def params_close(a, b, **kw):
    return all(
        torch.allclose(pa, pb, **kw) for pa, pb in zip(a.parameters(), b.parameters())
    )


##########################################
# Item 4: gradient accumulation
##########################################


def test_accumulation_boundaries():
    def boundaries(n, accum):
        return [s for s in range(n) if is_accumulation_boundary(s, n, accum)]

    assert boundaries(10, 4) == [3, 7, 9]  # trailing partial group flushed
    assert boundaries(8, 4) == [3, 7]
    assert boundaries(3, 4) == [2]
    assert boundaries(4, 1) == [0, 1, 2, 3]


def test_accumulation_group_sizes():
    sizes = [accumulation_group_size(s, 10, 4) for s in range(10)]
    assert sizes == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]


def test_num_training_steps_counts_partial_groups():
    assert num_training_steps(2, 10, 4) == 6
    assert num_training_steps(1, 8, 4) == 2


def test_optimizer_and_scheduler_step_counts(tmp_path, accelerator):
    config = tiny_config(gradient_accumulation_steps=4, n_epochs=2)
    model = tiny_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result, scheduler = train(
        model, optimizer, make_batches(10, config), config, tmp_path, accelerator
    )
    expected = num_training_steps(2, 10, 4)
    assert result["n_optimizer_steps"] == expected
    assert scheduler.last_epoch == expected
    for p in model.parameters():
        assert int(optimizer.state[p]["step"]) == expected


def test_full_group_is_one_step_on_mean_gradient(tmp_path, accelerator):
    config = tiny_config(gradient_accumulation_steps=2)
    model = tiny_model(config)
    reference = copy.deepcopy(model)
    batches = make_batches(2, config)

    train(
        model,
        torch.optim.SGD(model.parameters(), lr=0.5),
        batches,
        config,
        tmp_path,
        accelerator,
    )

    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.5)
    losses = [
        reference(b["input_ids"], attention_mask=b["attention_mask"])["loss"]
        for b in batches
    ]
    (sum(losses) / 2).backward()
    ref_opt.step()

    assert params_close(model, reference, atol=1e-6)


def test_partial_group_averages_over_its_real_size(tmp_path, accelerator):
    # 2 micro-batches with accum=3 (one partial group) must match accum=2
    batches = make_batches(2, tiny_config())
    results = []
    for accum in (2, 3):
        config = tiny_config(gradient_accumulation_steps=accum)
        model = tiny_model(config)
        train(
            model,
            torch.optim.SGD(model.parameters(), lr=0.5),
            batches,
            config,
            tmp_path / str(accum),
            accelerator,
        )
        results.append(model)
    assert params_close(*results, atol=1e-6)


##########################################
# Item 2: gradient clipping
##########################################


def grad_norm_and_delta(clip, tmp_path, accelerator):
    config = tiny_config(gradient_clipping=clip)
    model = tiny_model(config)
    before = copy.deepcopy(model)
    batch = make_batches(1, config)

    probe = copy.deepcopy(model)
    probe(batch[0]["input_ids"], attention_mask=batch[0]["attention_mask"])[
        "loss"
    ].backward()
    raw_norm = torch.norm(torch.stack([p.grad.norm() for p in probe.parameters()]))

    train(
        model,
        torch.optim.SGD(model.parameters(), lr=1.0),
        batch,
        config,
        tmp_path,
        accelerator,
    )
    delta = torch.norm(
        torch.stack(
            [(a - b).norm() for a, b in zip(model.parameters(), before.parameters())]
        )
    )
    return raw_norm.item(), delta.item()


def test_gradient_clipping_limits_update_norm(tmp_path, accelerator):
    raw_norm, delta = grad_norm_and_delta(1e-3, tmp_path, accelerator)
    assert raw_norm > 1e-2  # clipping is actually active
    assert delta == pytest.approx(1e-3, rel=1e-3)


@pytest.mark.parametrize("disabled", [0, None])
def test_gradient_clipping_disabled(disabled, tmp_path, accelerator):
    raw_norm, delta = grad_norm_and_delta(disabled, tmp_path, accelerator)
    assert delta == pytest.approx(raw_norm, rel=1e-4)


##########################################
# Item 3: logged train loss is unscaled
##########################################


def test_logged_train_loss_is_unscaled(tmp_path, accelerator):
    config = tiny_config(gradient_accumulation_steps=4)
    model = tiny_model(config)
    batches = make_batches(4, config)
    with torch.no_grad():
        first_loss = copy.deepcopy(model)(
            batches[0]["input_ids"], attention_mask=batches[0]["attention_mask"]
        )["loss"].item()

    logged = []
    result, _ = train(
        model,
        torch.optim.AdamW(model.parameters()),
        batches,
        config,
        tmp_path,
        accelerator,
        log=lambda metrics, step=None: logged.append(metrics),
    )
    train_losses = [m["train_loss"] for m in logged if "train_loss" in m]
    assert len(train_losses) == 4
    # no optimizer step before micro-batch 0's loss, so it equals the initial loss
    assert train_losses[0] == pytest.approx(first_loss, rel=1e-5)
    assert result["avg_train_loss"] == pytest.approx(sum(train_losses) / 4)


##########################################
# Item 1: padding excluded from train and eval loss
##########################################


def test_training_ignores_padding(tmp_path, accelerator):
    config = tiny_config()
    real = make_batches(1, config)[0]["input_ids"][:, :8]
    unpadded = [{"input_ids": real, "attention_mask": torch.ones_like(real)}]
    padded = [pad_batch(real, SEQ_LEN)]

    models = []
    for name, batches in (("unpadded", unpadded), ("padded", padded)):
        model = tiny_model(config)
        train(
            model,
            torch.optim.SGD(model.parameters(), lr=0.5),
            batches,
            config,
            tmp_path / name,
            accelerator,
        )
        models.append(model)
    assert params_close(*models, atol=1e-5)


@pytest.mark.parametrize("cls", [GPTModel, DiffTransformer])
def test_eval_padded_equals_unpadded(cls):
    config = tiny_config()
    model = tiny_model(config, cls)
    real = make_batches(1, config)[0]["input_ids"][:, :6]
    loss_real, ppl_real = evaluate(
        model, [{"input_ids": real, "attention_mask": torch.ones_like(real)}]
    )
    loss_padded, ppl_padded = evaluate(model, [pad_batch(real, SEQ_LEN)])
    assert loss_padded == pytest.approx(loss_real, rel=1e-5)
    assert ppl_padded == pytest.approx(math.exp(loss_real), rel=1e-5)


def manual_token_loss(model, sequences):
    """Mean next-token cross entropy over all targets of unpadded sequences."""
    total, count = 0.0, 0
    with torch.no_grad():
        for seq in sequences:
            logits = model(seq)["logits"][:, :-1].reshape(-1, model.config.vocab_size)
            total += F.cross_entropy(
                logits.float(), seq[:, 1:].reshape(-1), reduction="sum"
            ).item()
            count += seq[:, 1:].numel()
    return total / count


def test_eval_is_token_weighted_across_batches(accelerator):
    config = tiny_config()
    model = tiny_model(config).eval()
    ids = make_batches(2, config, batch_size=1)
    short, full = ids[0]["input_ids"][:, :3], ids[1]["input_ids"]  # 2 vs 11 targets
    batches = [pad_batch(short, SEQ_LEN), pad_batch(full, SEQ_LEN)]

    expected = manual_token_loss(model, [short, full])
    loss, ppl = evaluate(model, batches)
    assert loss == pytest.approx(expected, rel=1e-5)
    assert ppl == pytest.approx(math.exp(expected), rel=1e-5)
    assert evaluate(model, batches, accelerator=accelerator) == pytest.approx(
        (loss, ppl)
    )


def test_eval_without_attention_mask_counts_all_targets():
    config = tiny_config()
    model = tiny_model(config).eval()
    x = make_batches(1, config)[0]["input_ids"]
    loss, _ = evaluate(model, [{"input_ids": x}])
    assert loss == pytest.approx(manual_token_loss(model, [x]), rel=1e-5)


def test_eval_skips_all_padding_batches_and_rejects_empty():
    config = tiny_config()
    model = tiny_model(config)
    real = make_batches(1, config)[0]
    empty = {
        "input_ids": torch.zeros(2, SEQ_LEN, dtype=torch.long),
        "attention_mask": torch.zeros(2, SEQ_LEN, dtype=torch.long),
    }
    assert evaluate(model, [empty, real]) == pytest.approx(evaluate(model, [real]))
    with pytest.raises(ValueError, match="no non-padded targets"):
        evaluate(model, [empty])


##########################################
# Item 5: best and last checkpoints
##########################################


def test_checkpoint_manager_keeps_best_separate_from_last(tmp_path):
    accelerator = Accelerator(cpu=True)
    model = accelerator.prepare(tiny_model(tiny_config()))
    checkpoints = CheckpointManager(accelerator, tmp_path, keep_k=1)

    assert checkpoints.save_best(3.0, {"global_step": 1}) is not None
    with torch.no_grad():
        model.head.weight.fill_(1.0)
    best_dir = checkpoints.save_best(2.0, {"global_step": 2})  # improvement: saved
    assert best_dir == tmp_path / "ckpt_000002"
    with torch.no_grad():
        model.head.weight.fill_(3.0)
    assert checkpoints.save_best(2.5, {"global_step": 3}) is None  # worse: untouched
    checkpoints.save_last({"global_step": 3})

    best = load_file(best_dir / "model.safetensors")
    last = load_file(tmp_path / "last" / "model.safetensors")
    assert torch.all(best["head.weight"] == 1.0)
    assert torch.all(last["head.weight"] == 3.0)
    assert checkpoints.best == [(2.0, "ckpt_000002")]
    assert not (tmp_path / "ckpt_000001").exists()  # pruned to keep_k=1


def test_run_training_best_includes_end_of_epoch_evals(tmp_path, accelerator):
    config = tiny_config(n_epochs=2)
    model = tiny_model(config)
    batches = make_batches(4, config)
    val = make_batches(2, config, seed=7)
    step0_val, _ = evaluate(copy.deepcopy(model), val)

    logged = []
    result, _ = train(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-2),
        batches,
        config,
        tmp_path,
        accelerator,
        log=lambda m, step=None: logged.append(m),
        val=val,
    )
    epoch_vals = [m["val_loss"] for m in logged if "val_loss" in m]
    assert len(epoch_vals) == 2
    assert result["best_val_loss"] == pytest.approx(min([step0_val, *epoch_vals]))
    # keep_k=3 keeps all three evaluations: after micro-batch 1, epoch 0 (4), epoch 1 (8)
    kept = sorted(p.name for p in tmp_path.glob("ckpt_*"))
    assert kept == ["ckpt_000001", "ckpt_000004", "ckpt_000008"]
    assert (tmp_path / "last").is_dir()


##########################################
# Item 7: weight-decay parameter groups
##########################################


def decay_by_name(model, weight_decay=0.1):
    groups = get_grouped_params(model, weight_decay=weight_decay)
    by_id = {}
    for g in groups:
        for p in g["params"]:
            assert id(p) not in by_id, "parameter in more than one group"
            by_id[id(p)] = g["weight_decay"]
    return {n: by_id[id(p)] for n, p in model.named_parameters()}


def test_gpt_weight_decay_groups():
    decay = decay_by_name(tiny_model(tiny_config(), GPTModel))
    for name, wd in decay.items():
        no_decay = (
            name.endswith(".bias")
            or ".norm_1." in name
            or ".norm_2." in name
            or name.startswith("ln_f.")
            or name == "pos_embed"
        )
        assert wd == (0.0 if no_decay else 0.1), name
    assert decay["blocks.0.norm_1.weight"] == 0.0
    assert decay["ln_f.weight"] == 0.0
    assert decay["embed.weight"] == decay["head.weight"] == 0.1
    assert decay["blocks.0.context.qkv_proj.weight"] == 0.1


def test_difftransformer_weight_decay_groups():
    decay = decay_by_name(tiny_model(tiny_config(), DiffTransformer))
    for name, wd in decay.items():
        no_decay = (
            name.endswith(".bias")
            or name.endswith(".scale")
            or name.endswith(".rms_scale")
            or ".attn.lambda_" in name
        )
        assert wd == (0.0 if no_decay else 0.1), name
    for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2", "rms_scale"):
        assert decay[f"layers.0.attn.{name}"] == 0.0
    assert decay["layers.0.norm1.scale"] == decay["norm.scale"] == 0.0
    assert decay["token_emb.weight"] == decay["pos_emb.weight"] == 0.1
    assert decay["layers.0.attn.W_q.weight"] == decay["head.weight"] == 0.1


def test_grouped_params_handle_tied_and_frozen_weights():
    model = tiny_model(tiny_config(), GPTModel)
    model.head.weight = model.embed.weight  # tie
    model.ln_f.weight.requires_grad_(False)
    groups = get_grouped_params(model, weight_decay=0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) == len(expected)
    assert set(ids) == expected
    torch.optim.AdamW(groups)  # raises on duplicated parameters


##########################################
# Item 10: sampling from a wrapped model
##########################################


def test_sample_text_unwraps_model():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = tiny_config(vocab_size=len(tokenizer))
    wrapped = nn.DataParallel(tiny_model(config))
    assert not hasattr(wrapped, "generate")  # the old direct call would fail
    text = sample_text(Accelerator(cpu=True), wrapped, tokenizer, prompt="Once upon")
    assert isinstance(text, str) and text.startswith("Once upon")


##########################################
# LR scheduler step counts with several processes
##########################################


def test_lr_schedule_steps():
    # single process: unchanged from the optimizer-step count
    assert lr_schedule_steps(2, 10, 4, 3, 1, False) == (3, num_training_steps(2, 10, 4))
    # each step() call advances the wrapped scheduler num_processes times
    # (2 epochs * ceil(5 / 4) = 4 optimizer steps) * 2 processes
    assert lr_schedule_steps(2, 5, 4, 3, 2, False) == (6, 8)
    assert lr_schedule_steps(1, 7, 2, 2, 3, False) == (6, 12)
    # split_batches: one advance per call
    assert lr_schedule_steps(2, 5, 4, 3, 2, True) == (3, 4)
    with pytest.raises(ValueError, match="num_processes"):
        lr_schedule_steps(1, 4, 1, 0, 0, False)


def lrs_per_optimizer_step(
    accelerator,
    monkeypatch,
    scheduler_type,
    n_epochs,
    n_micro_batches,
    accum,
    counts,
    P,
):
    """
    LR after each optimizer step of one process, stepping a real
    AcceleratedScheduler at the run_training accumulation boundaries while it
    believes there are P processes. Returns (lrs, underlying scheduler).
    """
    monkeypatch.setattr(
        accelerate.scheduler,
        "AcceleratorState",
        lambda: SimpleNamespace(num_processes=P),
    )
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = get_scheduler(
        scheduler_type,
        optimizer,
        num_warmup_steps=counts[0],
        num_training_steps=counts[1],
    )
    accelerated_optimizer = AcceleratedOptimizer(optimizer, device_placement=False)
    wrapped = AcceleratedScheduler(scheduler, accelerated_optimizer)
    assert not wrapped.split_batches and wrapped.step_with_optimizer
    assert wrapped.gradient_state.sync_gradients  # no accelerator.accumulate
    lrs = []
    for _ in range(n_epochs):
        for step in range(n_micro_batches):
            if is_accumulation_boundary(step, n_micro_batches, accum):
                accelerated_optimizer.step()  # no grads: a no-op, keeps the order
                wrapped.step()
                lrs.append(wrapped.get_last_lr()[0])
    return lrs, scheduler


SCHEDULE_CASES = [
    # n_epochs, sharded loader length, accum, n_warmup_steps, num_processes
    (2, 5, 4, 1, 2),
    (1, 7, 2, 2, 3),
    (3, 4, 1, 2, 4),
    (2, 10, 4, 3, 1),
]


@pytest.mark.parametrize("scheduler_type", ["linear", "cosine"])
@pytest.mark.parametrize("n_epochs, n_micro, accum, warmup, P", SCHEDULE_CASES)
def test_multi_process_schedule_matches_single_process(
    accelerator, monkeypatch, scheduler_type, n_epochs, n_micro, accum, warmup, P
):
    n_steps = num_training_steps(n_epochs, n_micro, accum)
    args = (accelerator, monkeypatch, scheduler_type, n_epochs, n_micro, accum)
    expected, _ = lrs_per_optimizer_step(*args, (warmup, n_steps), 1)
    counts = lr_schedule_steps(n_epochs, n_micro, accum, warmup, P, False)
    lrs, scheduler = lrs_per_optimizer_step(*args, counts, P)
    assert len(lrs) == n_steps
    assert lrs == pytest.approx(expected)
    # the schedule ends exactly on the last optimizer step, not before
    assert scheduler.last_epoch == counts[1]
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)
    assert lrs[-2] > 1e-6


@pytest.mark.parametrize("n_epochs, n_micro, accum, warmup, P", SCHEDULE_CASES[:3])
def test_unscaled_schedule_counts_are_wrong_with_several_processes(
    accelerator, monkeypatch, n_epochs, n_micro, accum, warmup, P
):
    # the pre-fix counts: un-sharded loader length (P * sharded), no scaling
    n_steps = num_training_steps(n_epochs, n_micro, accum)
    args = (accelerator, monkeypatch, "linear", n_epochs, n_micro, accum)
    expected, _ = lrs_per_optimizer_step(*args, (warmup, n_steps), 1)
    old_counts = (warmup, num_training_steps(n_epochs, n_micro * P, accum))
    lrs, _ = lrs_per_optimizer_step(*args, old_counts, P)
    assert lrs != pytest.approx(expected)


def test_prepared_scheduler_ends_on_last_optimizer_step(tmp_path, accelerator):
    # the main() pattern on one process: prepare the loader, count, then prepare
    # the scheduler; the LR must reach the end of its schedule on the last step
    config = tiny_config(gradient_accumulation_steps=4, n_epochs=2)
    model = tiny_model(config)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(make_batches(10, config), batch_size=None)
    model, optimizer, loader = accelerator.prepare(model, base_optimizer, loader)
    assert len(loader) == 10
    counts = lr_schedule_steps(
        config.n_epochs,
        len(loader),
        config.gradient_accumulation_steps,
        2,
        accelerator.num_processes,
        accelerator.split_batches,
    )
    assert counts == (2, num_training_steps(2, 10, 4))
    scheduler = accelerator.prepare(
        get_scheduler("linear", base_optimizer, counts[0], counts[1])
    )
    lrs = []
    result = run_training(
        model,
        optimizer,
        scheduler,
        loader,
        make_batches(1, config),
        accelerator,
        config,
        tmp_path,
        log=lambda m, step=None: lrs.append(m["learning_rate"])
        if "learning_rate" in m
        else None,
    )
    assert result["n_optimizer_steps"] == counts[1]
    assert scheduler.scheduler.last_epoch == counts[1]
    assert lrs[-1] == 0.0
    assert lrs[-2] > 0.0  # after the 5th of 6 optimizer steps
