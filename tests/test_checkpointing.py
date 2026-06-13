"""Tests for checkpoint save/resume and keep-k-best pruning (utils.CheckpointManager)."""

from pathlib import Path

import pytest

from llamlam.utils import (
    CheckpointManager,
    _select_best,
    load_checkpoint,
    resolve_resume_dir,
)


class FakeAccelerator:
    """Stands in for accelerate.Accelerator: persists a single integer 'state'."""

    is_main_process = True

    def __init__(self, state=0):
        self.state = state

    def save_state(self, output_dir):
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.bin").write_text(str(self.state))

    def load_state(self, output_dir):
        self.state = int((Path(output_dir) / "state.bin").read_text())


def test_select_best_keeps_k_lowest():
    items = [(0.5, "a"), (0.1, "b"), (0.9, "c"), (0.3, "d")]
    keep, drop = _select_best(items, 2)
    assert [p for _, p in keep] == ["b", "d"]
    assert {p for _, p in drop} == {"a", "c"}


def test_save_best_prunes_to_keep_k(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_k=2)
    acc = FakeAccelerator()
    losses = {100: 0.5, 200: 0.4, 300: 0.6, 400: 0.2}
    for step, loss in losses.items():
        mgr.save_best(acc, loss, {"global_step": step, "epoch": 0})

    remaining = sorted(p.name for p in tmp_path.glob("ckpt_*"))
    # Lowest-loss two are steps 400 (0.2) and 200 (0.4).
    assert remaining == ["ckpt_000200", "ckpt_000400"]
    assert mgr.best_path == tmp_path / "ckpt_000400"


def test_save_last_overwrites_and_round_trips(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_k=3)
    acc = FakeAccelerator(state=7)

    mgr.save_last(acc, {"global_step": 10, "tokens_seen": 999, "epoch": 1})
    mgr.save_last(acc, {"global_step": 20, "tokens_seen": 1998, "epoch": 2})

    # Only one 'last' dir exists and it holds the most recent progress.
    last = tmp_path / "last"
    assert last.exists()

    restorer = FakeAccelerator(state=0)
    progress = load_checkpoint(restorer, last)
    assert restorer.state == 7  # accelerate state restored
    assert progress == {"global_step": 20, "tokens_seen": 1998, "epoch": 2}


def test_resolve_resume_dir_variants(tmp_path):
    assert resolve_resume_dir(tmp_path, None) is None
    assert resolve_resume_dir(tmp_path, "") is None
    # "latest" with no checkpoints yet -> None
    assert resolve_resume_dir(tmp_path, "latest") is None

    # An explicit path is returned as-is.
    explicit = tmp_path / "ckpt_000123"
    explicit.mkdir()
    assert resolve_resume_dir(tmp_path, str(explicit)) == Path(str(explicit))

    # "latest" prefers the 'last' dir when present.
    (tmp_path / "last").mkdir()
    assert resolve_resume_dir(tmp_path, "latest") == tmp_path / "last"


def test_resolve_latest_falls_back_to_highest_ckpt(tmp_path):
    for step in (100, 300, 200):
        (tmp_path / f"ckpt_{step:06d}").mkdir()
    assert resolve_resume_dir(tmp_path, "latest") == tmp_path / "ckpt_000300"


def test_non_main_process_does_not_write_progress(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_k=2)
    acc = FakeAccelerator()
    acc.is_main_process = False
    mgr.save_best(acc, 0.1, {"global_step": 1, "epoch": 0})
    # State is written by all ranks, but progress.json / pruning only by main.
    assert not (tmp_path / "ckpt_000001" / "progress.json").exists()
    assert mgr.best == []


if __name__ == "__main__":
    pytest.main([__file__])
