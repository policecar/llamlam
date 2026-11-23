"""
Checkpoints: a resumable `last/` plus the keep-k best by validation loss.

Layout under a run directory:

    last/            accelerator state at the end of the latest finished epoch
    ckpt_<step>/     accelerator state at an evaluation, kept while its validation
                     loss is among the keep_k lowest seen in the run

Every checkpoint holds what `accelerator.save_state` writes (model, optimizer,
LR scheduler, RNG states, grad scaler), the MPS RNG state when training on MPS
(which save_state omits), plus `progress.json` with the counters
Accelerate does not track (epoch, global_step, n_optimizer_steps, tokens_seen,
best_val_loss), the list of kept best checkpoints and the run's config
(`Config.to_dict()`, which a resume continues with; see train.resume_config).

Resume is epoch-granular: only `last/` (written at an epoch boundary) is
resumable, and an epoch interrupted part-way is restarted from its beginning.
The `ckpt_<step>/` directories are model snapshots taken mid-epoch (with
gradients possibly half-accumulated), so load_checkpoint refuses to resume from
them; use `accelerator.load_state` on them directly to evaluate or sample.

@decision DEC-CKPT-001: keep-k is decided at every evaluation (save iff the loss
    would be among the k lowest), not only when the loss beats the best so far
@rationale: the PR saved only on a new best and pruned to k, so with k=2 and losses
    0.5, 0.3, 0.4 it kept {0.5, 0.3}, not the two lowest {0.3, 0.4}. Deciding up
    front also never writes a checkpoint only to delete it again.
@decision DEC-CKPT-002: every process tracks the kept list and calls save_state;
    only the main process writes progress.json, renames and deletes directories
@rationale: save_state is a collective call (each rank writes its RNG state, and
    DeepSpeed/FSDP gather weights), so all ranks must agree on whether to save.
    In the PR non-main ranks kept an empty list and would always try to save.
@decision DEC-CKPT-003: write into `<name>.tmp/`, then rename over `<name>/`
@rationale: a run killed while saving would otherwise leave `last/` with new
    weights and old (or missing) progress, which resumes silently wrong; now the
    previous `last/` survives until the new one is complete.
@decision DEC-CKPT-004: around save_state, a save-state pre-hook drops the head's
    alias of the tied embedding (`head.weight`) from the model state dict
@rationale: safetensors refuses aliased tensors, so Accelerate's
    clean_state_dict_for_safetensors deletes `head.weight` itself and logs
    "Removed shared tensor {'head.weight'} while saving" on every save of a tied
    model. Dropping that one key first writes the same file without the message,
    and needs no log filter: an untied head is not touched, and any other shared
    tensor is still removed and reported by Accelerate. Loading is unchanged
    (safetensors' load_model restores the alias from the model's own sharing).
"""

import json
import logging
import math
import shutil
from pathlib import Path

import torch

from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel

logger = logging.getLogger(__name__)

LAST_DIR = "last"
PROGRESS_FILE = "progress.json"
# accelerate's save_state (1.0.1 and 1.15.0) keeps the CPU, CUDA, XPU, ... RNG
# states but not the MPS one, so dropout on MPS would draw different masks after a
# resume
MPS_RNG_FILE = "mps_rng_state.pt"


# (embedding, head) parameters that tie_word_embeddings makes one tensor
TIED_WEIGHTS = {
    GPTModel: ("embed.weight", "head.weight"),
    DiffTransformer: ("token_emb.weight", "head.weight"),
}


def tied_alias_key(model, state_dict):
    """
    The state_dict key of `model`'s head when it is tied to its embedding, else None.

    `model` is the unwrapped model; state_dict may come from a wrapper (DDP,
    torch.compile) and so carry a key prefix. Only a head that is the embedding
    parameter, and whose entry shares the embedding entry's storage, qualifies.
    """
    names = TIED_WEIGHTS.get(type(model))
    if names is None:
        return None
    embedding, head = names
    if model.get_parameter(head) is not model.get_parameter(embedding):
        return None  # untied
    for key, tensor in state_dict.items():
        if key != head and not key.endswith("." + head):
            continue
        target = state_dict.get(key[: len(key) - len(head)] + embedding)
        if (
            isinstance(target, torch.Tensor)
            and isinstance(tensor, torch.Tensor)
            and tensor.data_ptr() == target.data_ptr()
            and tensor.shape == target.shape
        ):
            return key
    return None


def drop_tied_head(accelerator):
    """A save-state pre-hook that drops each tied head's alias key (DEC-CKPT-004)."""

    def hook(models, weights, output_dir):
        if len(weights) != len(models):
            return  # FSDP / DeepSpeed / Megatron save the models themselves
        for model, state_dict in zip(models, weights):
            key = tied_alias_key(accelerator.unwrap_model(model), state_dict)
            if key is not None:
                del state_dict[key]

    return hook


def best_checkpoint_name(global_step):
    return f"ckpt_{global_step:06d}"


def select_best(checkpoints, keep_k):
    """Split (val_loss, name) entries into the keep_k lowest-loss ones and the rest."""
    ordered = sorted(checkpoints, key=lambda c: c[0])  # stable: earlier wins ties
    return ordered[:keep_k], ordered[keep_k:]


class CheckpointManager:
    """
    Saves `last/` and keep-k-best checkpoints of `accelerator`'s state in run_dir.

    `best` is the list of kept (val_loss, directory name) pairs, lowest loss first.
    """

    def __init__(self, accelerator, run_dir, keep_k=3):
        if keep_k < 0:
            raise ValueError(f"keep_k must be >= 0, got {keep_k}")
        self.accelerator = accelerator
        self.run_dir = Path(run_dir)
        self.keep_k = keep_k
        self.best = []

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def qualifies(self, val_loss):
        """Whether a checkpoint with val_loss would be among the keep_k kept."""
        if self.keep_k == 0 or math.isnan(val_loss):
            return False
        return len(self.best) < self.keep_k or val_loss < self.best[-1][0]

    def save_last(self, progress):
        """Overwrite last/ with the current state; progress must describe an epoch end."""
        return self._write(LAST_DIR, {**progress, "resumable": True})

    def save_best(self, val_loss, progress):
        """
        Save a ckpt_<global_step>/ checkpoint if val_loss is among the keep_k lowest,
        then delete the checkpoints that dropped out. Returns its path, or None.
        """
        if not self.qualifies(val_loss):
            return None
        name = best_checkpoint_name(progress["global_step"])
        entries = [(loss, n) for loss, n in self.best if n != name]
        self.best, dropped = select_best(entries + [(val_loss, name)], self.keep_k)
        path = self._write(name, {**progress, "val_loss": val_loss, "resumable": False})
        if self.is_main_process:
            for _, old in dropped:
                shutil.rmtree(self.run_dir / old, ignore_errors=True)
        return path

    def restore(self, progress):
        """Take over the kept-best list recorded in a checkpoint's progress."""
        self.best, _ = select_best(
            [(loss, name) for loss, name in progress.get("best_checkpoints", [])],
            self.keep_k,
        )

    def _write(self, name, progress):
        final = self.run_dir / name
        tmp = self.run_dir / f"{name}.tmp"
        old = self.run_dir / f"{name}.old"
        accelerator = self.accelerator
        accelerator.wait_for_everyone()
        if self.is_main_process:  # left over from a killed save or failed cleanup
            for leftover in (tmp, old):
                shutil.rmtree(leftover, ignore_errors=True)
        accelerator.wait_for_everyone()
        handle = accelerator.register_save_state_pre_hook(drop_tied_head(accelerator))
        try:
            accelerator.save_state(str(tmp))  # collective: every rank writes its part
        finally:
            handle.remove()
        accelerator.wait_for_everyone()
        if self.is_main_process:
            if accelerator.device.type == "mps":  # single-process only
                torch.save(torch.mps.get_rng_state(), tmp / MPS_RNG_FILE)
            progress = {**progress, "best_checkpoints": [list(b) for b in self.best]}
            with open(tmp / PROGRESS_FILE, "w") as f:
                json.dump(progress, f, indent=1)
            if final.exists():
                final.rename(old)
            tmp.rename(final)
            shutil.rmtree(old, ignore_errors=True)
        accelerator.wait_for_everyone()
        return final


def resolve_resume_dir(run_dir, resume_from):
    """
    Resolve --resume_from to a checkpoint directory, or None if not resuming.

    "latest" means `<run_dir>/last`; anything else is a path to a checkpoint
    directory. Raises FileNotFoundError if the directory does not exist.
    """
    if resume_from in (None, ""):
        return None
    if resume_from == "latest":
        path = Path(run_dir) / LAST_DIR
        hint = (
            f" (--resume_from latest resumes the run in --output_dir/--run_name, "
            f"here {run_dir}; no epoch of it has finished yet, or the name is wrong)"
        )
    else:
        path = Path(resume_from)
        hint = ""
    old = path.with_name(f"{path.name}.old")
    if not path.is_dir() and old.is_dir():  # killed between _write's two renames
        hint += f"; {old} is the intact previous checkpoint, rename it to {path}"
    if not path.is_dir():
        raise FileNotFoundError(f"No checkpoint directory {path} to resume from{hint}")
    return path


def read_resumable_progress(ckpt_dir):
    """The progress of an epoch-boundary checkpoint; raises for a mid-epoch one."""
    progress = read_progress(ckpt_dir)
    if not progress.get("resumable", False):
        raise ValueError(
            f"{ckpt_dir} was saved mid-epoch (step {progress.get('global_step')}); "
            f"resume is epoch-granular, so resume from a {LAST_DIR}/ checkpoint"
        )
    return progress


def read_progress(ckpt_dir):
    """The progress dict saved with a checkpoint; raises if it has none."""
    progress_file = Path(ckpt_dir) / PROGRESS_FILE
    if not progress_file.is_file():
        raise FileNotFoundError(
            f"{ckpt_dir} has no {PROGRESS_FILE}: not a complete checkpoint"
        )
    with open(progress_file) as f:
        return json.load(f)


def load_checkpoint(accelerator, ckpt_dir):
    """
    Restore accelerator state (model, optimizer, scheduler, RNG) from an
    epoch-boundary checkpoint and return its progress dict.

    Every object must already be prepared, the LR scheduler included, so that
    load_state restores into it. Raises ValueError for a mid-epoch (best)
    checkpoint, which cannot be resumed exactly.
    """
    progress = read_resumable_progress(ckpt_dir)
    accelerator.load_state(str(ckpt_dir))
    mps_rng = Path(ckpt_dir) / MPS_RNG_FILE
    if mps_rng.is_file() and accelerator.device.type == "mps":
        torch.mps.set_rng_state(torch.load(mps_rng, weights_only=True))
    return progress
