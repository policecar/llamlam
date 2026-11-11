"""
Checkpoints (llamlam.checkpoint): keep-k-best pruning, last/ round trip, resume
directory resolution, main-process-only writes, and an interrupted-and-resumed
run matching an uninterrupted one.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from llamlam.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    read_progress,
    resolve_resume_dir,
    select_best,
)
from llamlam.config import Config
from llamlam.train import (
    build_model,
    lr_schedule_steps,
    num_training_steps,
    run_training,
)
from llamlam.utils import build_optimizer


class FakeAccelerator:
    """Stands in for accelerate.Accelerator: persists a single integer 'state'."""

    def __init__(self, state=0, is_main_process=True, fail_save=False):
        self.state = state
        self.is_main_process = is_main_process
        self.fail_save = fail_save
        self.device = torch.device("cpu")
        self.saved_to = []
        self.loaded_from = []

    def wait_for_everyone(self):
        pass

    def save_state(self, output_dir):
        self.saved_to.append(Path(output_dir).name)
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        if self.fail_save:
            raise RuntimeError("killed while saving")
        (d / "state.bin").write_text(str(self.state))

    def load_state(self, input_dir):
        self.loaded_from.append(Path(input_dir))
        self.state = int((Path(input_dir) / "state.bin").read_text())


def ckpt_dirs(run_dir):
    return sorted(p.name for p in Path(run_dir).glob("ckpt_*"))


##########################################
# keep-k-best
##########################################


def test_select_best_keeps_k_lowest():
    items = [(0.5, "a"), (0.1, "b"), (0.9, "c"), (0.3, "d")]
    keep, drop = select_best(items, 2)
    assert keep == [(0.1, "b"), (0.3, "d")]
    assert sorted(drop) == [(0.5, "a"), (0.9, "c")]


@pytest.mark.parametrize("keep_k", [1, 2, 3])
def test_save_best_keeps_exactly_the_k_lowest_losses(tmp_path, keep_k):
    # 0.4 is not a new best but is among the 2 lowest when it arrives
    losses = [0.5, 0.3, 0.4, 0.6, 0.2, 0.35, 0.9]
    acc = FakeAccelerator()
    manager = CheckpointManager(acc, tmp_path, keep_k=keep_k)
    for step, loss in enumerate(losses, start=1):
        acc.state = step
        manager.save_best(loss, {"global_step": step})
        seen = list(enumerate(losses[:step], start=1))
        lowest = sorted(seen, key=lambda s: s[1])[:keep_k]
        assert ckpt_dirs(tmp_path) == sorted(f"ckpt_{s:06d}" for s, _ in lowest)
        assert manager.best == [(loss_, f"ckpt_{s:06d}") for s, loss_ in lowest]
        for s, loss_ in lowest:  # each kept dir holds its own state and loss
            d = tmp_path / f"ckpt_{s:06d}"
            assert (d / "state.bin").read_text() == str(s)
            assert read_progress(d)["val_loss"] == loss_
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.old"))


def test_save_best_skips_non_qualifying_without_writing(tmp_path):
    acc = FakeAccelerator()
    manager = CheckpointManager(acc, tmp_path, keep_k=2)
    manager.save_best(0.2, {"global_step": 1})
    manager.save_best(0.3, {"global_step": 2})
    acc.saved_to.clear()
    assert manager.save_best(0.5, {"global_step": 3}) is None
    assert manager.save_best(0.3, {"global_step": 4}) is None  # ties keep the earlier
    assert manager.save_best(float("nan"), {"global_step": 5}) is None
    assert acc.saved_to == []
    assert ckpt_dirs(tmp_path) == ["ckpt_000001", "ckpt_000002"]


def test_keep_k_zero_keeps_only_last(tmp_path):
    manager = CheckpointManager(FakeAccelerator(), tmp_path, keep_k=0)
    assert manager.save_best(0.1, {"global_step": 1}) is None
    manager.save_last({"global_step": 1, "epoch": 1})
    assert ckpt_dirs(tmp_path) == [] and (tmp_path / "last").is_dir()
    with pytest.raises(ValueError, match="keep_k"):
        CheckpointManager(FakeAccelerator(), tmp_path, keep_k=-1)


def test_restored_best_list_keeps_pruning(tmp_path):
    first = CheckpointManager(FakeAccelerator(), tmp_path, keep_k=2)
    first.save_best(0.5, {"global_step": 1})
    first.save_best(0.4, {"global_step": 2})
    first.save_last({"global_step": 2, "epoch": 1})

    resumed = CheckpointManager(FakeAccelerator(), tmp_path, keep_k=2)
    resumed.restore(read_progress(tmp_path / "last"))
    assert resumed.best == [(0.4, "ckpt_000002"), (0.5, "ckpt_000001")]
    assert resumed.save_best(0.6, {"global_step": 3}) is None
    resumed.save_best(0.1, {"global_step": 4})
    assert ckpt_dirs(tmp_path) == ["ckpt_000002", "ckpt_000004"]  # 0.5 pruned


##########################################
# last/ and resume
##########################################


def test_save_last_overwrites_and_round_trips(tmp_path):
    acc = FakeAccelerator(state=7)
    manager = CheckpointManager(acc, tmp_path, keep_k=2)
    manager.save_best(0.25, {"global_step": 5})
    manager.save_last({"global_step": 10, "tokens_seen": 999, "epoch": 1})
    acc.state = 8
    manager.save_last({"global_step": 20, "tokens_seen": 1998, "epoch": 2})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_000005", "last"]

    restorer = FakeAccelerator(state=0)
    progress = load_checkpoint(restorer, tmp_path / "last")
    assert restorer.state == 8  # accelerator state restored, latest save
    assert progress == {
        "global_step": 20,
        "tokens_seen": 1998,
        "epoch": 2,
        "resumable": True,
        "best_checkpoints": [[0.25, "ckpt_000005"]],
    }


def test_failed_save_keeps_previous_last(tmp_path):
    acc = FakeAccelerator(state=1)
    manager = CheckpointManager(acc, tmp_path)
    manager.save_last({"global_step": 4, "epoch": 1})
    acc.state, acc.fail_save = 2, True
    with pytest.raises(RuntimeError, match="killed"):
        manager.save_last({"global_step": 8, "epoch": 2})
    restorer = FakeAccelerator()
    assert load_checkpoint(restorer, tmp_path / "last")["global_step"] == 4
    assert restorer.state == 1
    # the next successful save clears the leftover temporary directory
    acc.fail_save = False
    manager.save_last({"global_step": 8, "epoch": 2})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last"]
    assert load_checkpoint(restorer, tmp_path / "last")["global_step"] == 8


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_rng_state_round_trips(tmp_path):
    # accelerate's save_state omits the MPS generator; dropout on MPS uses it
    acc = FakeAccelerator()
    acc.device = torch.device("mps")
    CheckpointManager(acc, tmp_path).save_last({"global_step": 1, "epoch": 1})
    assert (tmp_path / "last" / "mps_rng_state.pt").is_file()
    expected = torch.rand(8, device="mps")
    torch.rand(100, device="mps")  # move the generator on
    load_checkpoint(acc, tmp_path / "last")
    assert torch.equal(torch.rand(8, device="mps"), expected)


def test_no_mps_rng_file_off_mps(tmp_path):
    CheckpointManager(FakeAccelerator(), tmp_path).save_last({"epoch": 1})
    assert not (tmp_path / "last" / "mps_rng_state.pt").exists()


def test_best_checkpoints_are_not_resumable(tmp_path):
    manager = CheckpointManager(FakeAccelerator(), tmp_path)
    path = manager.save_best(0.1, {"global_step": 3, "epoch": 0})
    restorer = FakeAccelerator()
    with pytest.raises(ValueError, match="mid-epoch"):
        load_checkpoint(restorer, path)
    assert restorer.loaded_from == []  # refused before touching any state


def test_incomplete_checkpoint_is_rejected(tmp_path):
    (tmp_path / "last").mkdir()
    (tmp_path / "last" / "state.bin").write_text("3")
    restorer = FakeAccelerator()
    with pytest.raises(FileNotFoundError, match="progress.json"):
        load_checkpoint(restorer, tmp_path / "last")
    assert restorer.loaded_from == []


def test_resolve_resume_dir(tmp_path):
    assert resolve_resume_dir(tmp_path, None) is None
    assert resolve_resume_dir(tmp_path, "") is None
    # "latest" before any epoch finished: nothing to resume, explicit error
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        resolve_resume_dir(tmp_path, "latest")
    (tmp_path / "ckpt_000100").mkdir()
    with pytest.raises(FileNotFoundError):  # ckpt_* are not "latest"
        resolve_resume_dir(tmp_path, "latest")
    (tmp_path / "last").mkdir()
    assert resolve_resume_dir(tmp_path, "latest") == tmp_path / "last"

    explicit = tmp_path / "elsewhere" / "last"
    with pytest.raises(FileNotFoundError, match="elsewhere"):
        resolve_resume_dir(tmp_path, str(explicit))
    explicit.mkdir(parents=True)
    assert resolve_resume_dir(tmp_path, str(explicit)) == explicit


def test_only_main_process_writes(tmp_path):
    main_dir, worker_dir = tmp_path / "main", tmp_path / "worker"
    for run_dir in (main_dir, worker_dir):  # a kept checkpoint that gets pruned
        (run_dir / "ckpt_000001").mkdir(parents=True)
    main_acc = FakeAccelerator()
    worker_acc = FakeAccelerator(is_main_process=False)
    main = CheckpointManager(main_acc, main_dir, keep_k=1)
    worker = CheckpointManager(worker_acc, worker_dir, keep_k=1)
    for manager in (main, worker):
        manager.best = [(0.5, "ckpt_000001")]
        manager.save_best(0.1, {"global_step": 2})
        manager.save_best(0.9, {"global_step": 3})
        manager.save_last({"global_step": 3, "epoch": 1})

    # the worker joins every collective save and tracks the same kept list ...
    assert worker_acc.saved_to == main_acc.saved_to == ["ckpt_000002.tmp", "last.tmp"]
    assert worker.best == main.best == [(0.1, "ckpt_000002")]
    # ... but writes no progress, renames nothing and deletes nothing
    assert not list(worker_dir.rglob("progress.json"))
    assert sorted(p.name for p in worker_dir.iterdir()) == [
        "ckpt_000001",
        "ckpt_000002.tmp",
        "last.tmp",
    ]
    # while the main process does all three
    assert sorted(p.name for p in main_dir.iterdir()) == ["ckpt_000002", "last"]
    assert (main_dir / "last" / "progress.json").is_file()


##########################################
# End to end: interrupt during epoch 1, resume, compare with uninterrupted
##########################################

N_MICRO = 5
ACCUM = 2


class Interrupted(Exception):
    pass


def e2e_config(**overrides):
    params = dict(
        n_layers=1,
        n_heads=2,
        dim_head=8,
        max_seq_length=16,
        vocab_size=50,
        dropout=0.1,  # exercises the RNG state restored on resume
        n_epochs=2,
        eval_steps=3,
        gradient_accumulation_steps=ACCUM,
        n_warmup_steps=2,
        learning_rate=1e-2,
        keep_k_checkpoints=2,
    )
    params.update(overrides)
    return Config(**params)


def e2e_batches(config):
    """Dynamically padded batches: rows of different real lengths."""
    g = torch.Generator().manual_seed(3)
    batches = []
    for i in range(N_MICRO):
        length = 6 + i
        ids = torch.randint(0, config.vocab_size, (2, length), generator=g)
        mask = torch.ones_like(ids)
        mask[1, length - 1 - i % 3 :] = 0
        batches.append({"input_ids": ids, "attention_mask": mask})
    return batches


def setup_training(config, init_seed):
    """The main() pattern: prepare, count from the prepared loader, then scheduler."""
    accelerator = Accelerator(cpu=True)
    torch.manual_seed(init_seed)
    model = build_model(config)
    base_optimizer = build_optimizer(model, config)
    batches = e2e_batches(config)
    train_loader = DataLoader(batches, batch_size=None)
    val_loader = DataLoader(batches[:2], batch_size=None)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, base_optimizer, train_loader, val_loader
    )
    counts = lr_schedule_steps(
        config.n_epochs,
        len(train_loader),
        ACCUM,
        config.n_warmup_steps,
        accelerator.num_processes,
        accelerator.split_batches,
    )
    scheduler = accelerator.prepare(
        get_scheduler("linear", base_optimizer, counts[0], counts[1])
    )
    return model, optimizer, scheduler, train_loader, val_loader, accelerator, counts


def train_run(config, run_dir, init_seed=0, interrupt_epoch=None, resume_dir=None):
    model, optimizer, scheduler, train_loader, val_loader, accelerator, counts = (
        setup_training(config, init_seed)
    )
    logged = []

    def log(metrics, step=None):
        if metrics.get("epoch") == interrupt_epoch and "train_loss" in metrics:
            raise Interrupted
        logged.append((step, metrics))

    try:
        result = run_training(
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            accelerator,
            config,
            run_dir,
            log=log,
            resume_dir=resume_dir,
        )
    except Interrupted:
        result = None
    step_counts = {
        int(s["step"]) for s in optimizer.optimizer.state.values() if "step" in s
    }
    return {
        "result": result,
        "logged": logged,
        "model": accelerator.unwrap_model(model),
        "scheduler_last_epoch": scheduler.scheduler.last_epoch,
        "optimizer_steps": step_counts,
        "counts": counts,
    }


def lrs(logged):
    return [(s, m["learning_rate"]) for s, m in logged if "learning_rate" in m]


@pytest.mark.parametrize(
    "model_type, optimizer",
    [("gpt", "adamw"), ("diff", "adamw"), ("gpt", "muon"), ("diff", "grokadamw")],
)
def test_interrupted_and_resumed_run_matches_uninterrupted(
    tmp_path, model_type, optimizer
):
    config = e2e_config(model_type=model_type, optimizer=optimizer)
    full = train_run(config, tmp_path / "full")

    run_dir = tmp_path / "resumed"
    first = train_run(config, run_dir, interrupt_epoch=1)
    assert first["result"] is None  # stopped at the first micro-batch of epoch 1
    progress = read_progress(run_dir / "last")
    steps_per_epoch = math.ceil(N_MICRO / ACCUM)
    assert progress["epoch"] == 1
    assert progress["global_step"] == N_MICRO
    assert progress["n_optimizer_steps"] == steps_per_epoch

    # a differently initialised model: everything must come from the checkpoint
    second = train_run(
        config,
        run_dir,
        init_seed=1,
        resume_dir=resolve_resume_dir(run_dir, "latest"),
    )

    # counters continue where they left off and end where the full run ends
    resumed_lrs = lrs(second["logged"])
    assert resumed_lrs[0][0] == N_MICRO + 1  # first logged step of epoch 1
    assert resumed_lrs == lrs(full["logged"])[N_MICRO:]  # same LR position
    for key in ("n_optimizer_steps", "global_step", "tokens_seen", "best_val_loss"):
        assert second["result"][key] == full["result"][key], key
    total_steps = num_training_steps(2, N_MICRO, ACCUM)
    assert second["result"]["n_optimizer_steps"] == total_steps
    assert second["scheduler_last_epoch"] == full["scheduler_last_epoch"]
    assert second["scheduler_last_epoch"] == second["counts"][1] == total_steps
    assert lrs(full["logged"])[-1][1] == 0.0  # the schedule ends on the last step
    # every per-parameter step counter (AdamW, GrokAdamW, Muon's AdamW path)
    assert second["optimizer_steps"] == full["optimizer_steps"] == {total_steps}
    tokens = sum(int(b["attention_mask"].sum()) for b in e2e_batches(config))
    assert second["result"]["tokens_seen"] == 2 * tokens

    # identical training trajectory: same data, same restored RNG for dropout
    for (n, p), (_, q) in zip(
        second["model"].named_parameters(), full["model"].named_parameters()
    ):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6, msg=n)
    # and the same kept-best checkpoints
    assert ckpt_dirs(run_dir) == ckpt_dirs(tmp_path / "full")
    assert json.loads((run_dir / "last" / "progress.json").read_text())["epoch"] == 2


def test_resume_into_another_run_dir_starts_a_new_best_list(tmp_path):
    config = e2e_config(keep_k_checkpoints=10)
    train_run(config, tmp_path / "orig", interrupt_epoch=1)
    fork = train_run(config, tmp_path / "fork", resume_dir=tmp_path / "orig" / "last")
    assert fork["result"]["global_step"] == 2 * N_MICRO
    # only the evaluations of the resumed epoch (after micro-batches 7, 10; end 10)
    assert ckpt_dirs(tmp_path / "fork") == ["ckpt_000007", "ckpt_000010"]
    # the kept list names only this run's checkpoints, never the original's
    kept = read_progress(tmp_path / "fork" / "last")["best_checkpoints"]
    assert sorted(name for _, name in kept) == ["ckpt_000007", "ckpt_000010"]
    assert len(ckpt_dirs(tmp_path / "orig")) == 3  # untouched: 1, 4, 5


def test_resume_at_final_epoch_trains_nothing(tmp_path):
    config = e2e_config(n_epochs=1)
    done = train_run(config, tmp_path)
    again = train_run(config, tmp_path, resume_dir=tmp_path / "last")
    assert again["logged"] == []
    for key in ("n_optimizer_steps", "global_step", "tokens_seen", "best_val_loss"):
        assert again["result"][key] == done["result"][key]
    assert math.isnan(again["result"]["avg_train_loss"])


def test_resume_counters_cover_every_progress_field(tmp_path):
    # every counter run_training restores is written to last/
    config = e2e_config(n_epochs=1)
    train_run(config, tmp_path)
    keys = set(read_progress(tmp_path / "last"))
    assert keys == {
        "epoch",
        "global_step",
        "n_optimizer_steps",
        "tokens_seen",
        "best_val_loss",
        "resumable",
        "best_checkpoints",
    }
