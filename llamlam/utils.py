import math
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 137):
    """Set the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device=None, accelerator=None):
    """
    Evaluate the model on the given dataloader.

    Assumes that the model computes its own labels from the inputs and returns
    the mean loss over its non-padded targets when given an attention_mask.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader to evaluate the model on.
        accelerator: Optional; The accelerator to use. Defaults to None.

    Returns:
        A tuple (loss, perplexity): the mean loss over all real (non-padded)
        target tokens, and exp of that loss.
    """
    # @decision DEC-001: token-weighted mean, reduced once across processes
    # @rationale: batches have different numbers of real targets after padding is
    #   masked, so a mean of per-batch means over-weights short batches. Summing
    #   (loss * n_targets) and n_targets locally and reducing once keeps the result
    #   identical with and without an accelerator.
    loss_sum = None
    n_targets = None

    model.eval()
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        # Targets are positions 1..L-1; position t counts iff attention_mask[t] == 1
        if attention_mask is not None:
            batch_targets = attention_mask[:, 1:].sum()
        else:
            batch_targets = torch.tensor(
                input_ids.size(0) * (input_ids.size(1) - 1), device=input_ids.device
            )
        batch_targets = batch_targets.to(torch.float32)
        if batch_targets == 0:
            continue  # all-padding batch: no targets, loss is undefined
        with torch.no_grad():
            loss = model(input_ids, attention_mask=attention_mask)["loss"]
        weighted = loss.detach().to(torch.float32) * batch_targets
        loss_sum = weighted if loss_sum is None else loss_sum + weighted
        n_targets = batch_targets if n_targets is None else n_targets + batch_targets
    model.train()

    if loss_sum is None:  # this process saw no targets; still join the reduce
        device = accelerator.device if accelerator is not None else "cpu"
        loss_sum = n_targets = torch.zeros((), dtype=torch.float32, device=device)
    totals = torch.stack([loss_sum, n_targets])
    if accelerator is not None:
        totals = accelerator.reduce(totals, reduction="sum")
    if totals[1] == 0:
        raise ValueError("evaluate(): dataloader produced no non-padded targets")
    loss = (totals[0] / totals[1]).item()

    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return loss, perplexity


def get_grouped_params(model, weight_decay=0.1):
    """
    Get parameter groups for the optimizer.

    Only the `weight` of nn.Linear and nn.Embedding modules is weight-decayed.
    Every other parameter (biases, norm gains/scales, DiffTransformer lambdas,
    raw nn.Parameter tensors such as GPTModel.pos_embed) gets weight_decay 0.

    Args:
        model: The model to get the grouped parameters for.
        weight_decay: The weight decay for Linear/Embedding weights. Defaults to 0.1.

    Returns:
        A list of two optimizer param-group dicts: decayed and non-decayed.
    """
    # @decision DEC-002: decide weight decay by module type, not by name pattern
    # @rationale: the old substring list ("LayerNorm.weight") matched no parameter,
    #   so norm gains were decayed. Whitelisting Linear/Embedding weights holds for
    #   any module naming (norm_1, ln_f, RMSNorm.scale, rms_scale, lambda_*), and a
    #   new parameter type defaults to no decay instead of silently being decayed.
    decay, no_decay = [], []
    seen = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue  # skip frozen and tied (already grouped) parameters
            seen.add(id(param))
            if name == "weight" and isinstance(module, (nn.Linear, nn.Embedding)):
                decay.append(param)
            else:
                no_decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(
    model, optimizer, config, global_step, val_loss, tag, output_dir, max_ckpts=3
):
    """Save checkpoint to output directory."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "iter_num": global_step,
        "best_val_loss": val_loss,
    }
    torch.save(checkpoint, output_dir / f"ckpt_{tag}.pt")
