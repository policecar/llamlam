import json
import shutil

import numpy as np
import torch
import random

from pathlib import Path

from torch.optim.adamw import AdamW


def set_seed(seed: int = 137):
    """Set the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, config):
    """Construct the optimizer named by ``config.optimizer``.

    AdamW uses parameter groups (no weight decay on biases / norms). Muon splits
    2D weight matrices (orthogonalized updates) from everything else (its internal
    AdamW). GrokAdamW takes the same decay grouping as AdamW.
    """
    name = config.optimizer.lower()
    if name == "adamw":
        return AdamW(
            get_grouped_params(
                model, weight_decay=config.weight_decay, no_decay=config.no_decay
            ),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if name == "grokadamw":
        from llamlam.opt import GrokAdamW

        return GrokAdamW(
            get_grouped_params(
                model, weight_decay=config.weight_decay, no_decay=config.no_decay
            ),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clipping=config.gradient_clipping,
        )
    if name == "muon":
        from llamlam.opt import Muon

        # Muon handles >=2D weight matrices; biases, norms, embeddings and the
        # (tied) head go to the internal AdamW.
        muon_params, adamw_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and "emb" not in n and "head" not in n:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return Muon(
            muon_params,
            lr=config.learning_rate,
            adamw_params=adamw_params,
            adamw_lr=config.learning_rate,
            adamw_wd=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def evaluate(model, dataloader, device=None, accelerator=None):
    """
    Evaluate the model on the given dataloader.

    Assumes that the model computes its own labels from the inputs
    and returns its loss.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader to evaluate the model on.
        accelerator: Optional; The accelerator to use. Defaults to None.

    Returns:
        A tuple containing the loss and the perplexity.
    """
    losses = []

    model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
        loss = outputs["loss"]
        if accelerator is not None:
            losses.append(accelerator.gather(loss.unsqueeze(0)))
        else:
            losses.append(loss.unsqueeze(0))
    loss = torch.mean(torch.cat(losses))
    model.train()

    try:
        perplexity = torch.exp(loss).item()
    except OverflowError:
        perplexity = float("inf")

    return loss.item(), perplexity


def get_grouped_params(model, weight_decay=0.1, no_decay=[]):
    """
    Get grouped parameters for the optimizer.

    This function groups model parameters for optimization, applying different
    hyperparameters based on parameter type. E.g., embedding parameters may
    require different learning rates, biases might be excluded from weight decay.

    This approach allows for fine-grained control over parameter optimization.

    Args:
        model: The model to get the grouped parameters for.
        weight_decay: The weight decay to use. Defaults to 0.1.
        no_decay: The parameters to not apply weight decay to.

    Returns:
        A list of dictionaries containing the grouped parameters.
    """
    opt_grouped_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            group_parameters = {}

            # Define weight decay for all params except those in no_decay
            if any(nd in n for nd in no_decay):
                group_parameters["weight_decay"] = 0.0
            else:
                group_parameters["weight_decay"] = weight_decay

            # # Define custom learning rate for specific types of params
            # is_embed = "embed" in n
            # if "embed" in n or any(nd in n for nd in no_decay):
            #     group_parameters["lr"] = learning_rate * (3.3 if is_embed else 1.0)
            # else:
            #     group_parameters["lr"] = learning_rate * (1 / dim_head)

            group_parameters["params"] = [p]
            opt_grouped_params.append(group_parameters)

    return opt_grouped_params


def get_device():
    """Pick the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _filter_logits(logits, top_k=None, top_p=None):
    """Apply top-k and/or nucleus (top-p) filtering to a (batch, vocab) logits tensor."""
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p
        # keep at least the top token; shift the mask right by one
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    do_sample=False,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """Autoregressively generate text. Greedy by default; set do_sample=True for
    temperature / top-k / top-p sampling. Shared by GPTModel and DiffTransformer.
    """
    device = get_device()
    model.eval()
    model.to(device)

    token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_len = model.config.max_seq_length
    past = None

    for _ in range(max_new_tokens):
        if past is None:
            # (Re)prefill the cropped context window.
            out = model(token_ids[:, -max_len:], use_cache=True)
        else:
            # Incremental step: feed only the new token, reuse the cache.
            out = model(token_ids[:, -1:], past_key_values=past, use_cache=True)
        logits = out["logits"][:, -1, :]  # last-step logits
        past = out["past_key_values"]

        if not do_sample:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        token_ids = torch.cat((token_ids, idx_next), dim=1)

        # Learned positional embeddings only span max_len; once we reach it, drop
        # the cache so the next step re-prefills the cropped window.
        if token_ids.size(1) >= max_len:
            past = None

    return tokenizer.decode(token_ids[0], skip_special_tokens=True)


def _select_best(checkpoints, keep_k):
    """Split (val_loss, path) entries into the keep_k lowest-loss and the rest."""
    ordered = sorted(checkpoints, key=lambda c: c[0])
    return ordered[:keep_k], ordered[keep_k:]


class CheckpointManager:
    """Save/restore Accelerate training state with keep-k-best pruning.

    - ``save_last`` overwrites a single ``last/`` checkpoint, always available to
      resume the most recent state.
    - ``save_best`` writes a per-step checkpoint and prunes to the ``keep_k``
      lowest-validation-loss ones.

    Each checkpoint dir holds whatever ``accelerator.save_state`` writes (model,
    optimizer, scheduler, RNG) plus a ``progress.json`` with the training
    counters that Accelerate does not track.
    """

    def __init__(self, output_dir, keep_k=3):
        self.output_dir = Path(output_dir)
        self.keep_k = keep_k
        self.best = []  # list of (val_loss, Path)

    def _write(self, accelerator, ckpt_dir, progress):
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(ckpt_dir))
        if getattr(accelerator, "is_main_process", True):
            with open(ckpt_dir / "progress.json", "w") as f:
                json.dump(progress, f)
        return ckpt_dir

    def save_last(self, accelerator, progress):
        return self._write(accelerator, self.output_dir / "last", progress)

    def save_best(self, accelerator, val_loss, progress):
        ckpt_dir = self._write(
            accelerator,
            self.output_dir / f"ckpt_{progress['global_step']:06d}",
            progress,
        )
        if getattr(accelerator, "is_main_process", True):
            self.best = [(loss, p) for (loss, p) in self.best if p != ckpt_dir]
            self.best.append((val_loss, ckpt_dir))
            self.best, dropped = _select_best(self.best, self.keep_k)
            for _, path in dropped:
                shutil.rmtree(path, ignore_errors=True)
        return ckpt_dir

    @property
    def best_path(self):
        return self.best[0][1] if self.best else None


def resolve_resume_dir(output_dir, resume_from):
    """Resolve ``resume_from`` to a checkpoint dir.

    None/"" -> no resume; an explicit path -> that path; "latest" -> the ``last/``
    checkpoint if present, else the highest-numbered ``ckpt_*`` directory.
    """
    if resume_from in (None, ""):
        return None
    if resume_from != "latest":
        return Path(resume_from)
    last = Path(output_dir) / "last"
    if last.exists():
        return last
    ckpts = sorted(Path(output_dir).glob("ckpt_*"))
    return ckpts[-1] if ckpts else None


def load_checkpoint(accelerator, ckpt_dir):
    """Restore Accelerate state from ``ckpt_dir`` and return its progress dict."""
    ckpt_dir = Path(ckpt_dir)
    accelerator.load_state(str(ckpt_dir))
    progress_file = ckpt_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {}
