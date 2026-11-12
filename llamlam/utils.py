import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW

from llamlam.config import OPTIMIZERS, check_choice
from llamlam.opt import GrokAdamW, Muon


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


def muon_param_split(model):
    """
    Split trainable parameters into (muon, adamw) lists for the Muon optimizer.

    Muon gets the weight matrices of hidden nn.Linear layers. The AdamW path gets
    everything else: embeddings (nn.Embedding weights, raw positional parameters
    such as GPTModel.pos_embed), the output head (`model.head`), biases, norm
    gains and other 0/1-D parameters. Tied parameters appear once.
    """
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Linear):
        raise ValueError(
            "Muon routing needs the output projection at `model.head` (nn.Linear) "
            f"to keep it out of the Muon group; {type(model).__name__} has none"
        )
    muon, adamw = [], []
    seen = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            is_hidden_matrix = (
                isinstance(module, nn.Linear)
                and name == "weight"
                and param.ndim == 2
                and param is not head.weight  # the head's, or a weight tied to it
            )
            if is_hidden_matrix:
                muon.append(param)
            else:
                adamw.append(param)
    return muon, adamw


# @decision DEC-003: one constructor for all optimizers, with the weight-decay
#   grouping of get_grouped_params everywhere it applies, and every learning rate
#   held in param_group["lr"] (or a fixed ratio of it) so the LR scheduler drives it
# @rationale: the PR gave Muon's internal AdamW a single adamw_wd for all its
#   parameters, which decays norm gains and biases; here those go into a second
#   Muon param group with adamw_wd=0, matching AdamW/GrokAdamW. Muon stores its
#   AdamW lr as adamw_lr / lr times group["lr"], so a LambdaLR scales both paths.
# @decision DEC-004: Muon's orthogonalised updates use config.muon_learning_rate
#   (default 0.02, Muon's documented default), its AdamW path config.learning_rate
# @rationale: the orthogonalised update has spectral norm ~lr, a different scale
#   from an Adam step, so one shared rate left Muon ~33x below its recommended
#   0.02 at the default 6e-4. group["lr"] holds the Muon rate and adamw_lr_ratio
#   = learning_rate / muon_learning_rate, so the schedule still scales both.
def build_optimizer(model, config):
    """
    Construct the optimizer named by config.optimizer ("adamw", "muon", "grokadamw").

    - adamw: torch AdamW over get_grouped_params(model, config.weight_decay).
    - grokadamw: llamlam.opt.GrokAdamW over the same groups; its per-tensor
      gradient clipping uses config.gradient_clipping (0 or None disables it).
    - muon: llamlam.opt.Muon with muon_param_split(model); Muon steps use
      lr=config.muon_learning_rate (no weight decay, as in Muon itself), the
      AdamW path lr=config.learning_rate, with config.weight_decay on the
      Linear / Embedding weights it holds (embeddings, head) and 0 on the rest.

    config.muon_learning_rate is read only for "muon".

    Raises ValueError for an unknown optimizer name.
    """
    name = check_choice("optimizer", config.optimizer, OPTIMIZERS)
    lr, weight_decay = config.learning_rate, config.weight_decay

    if name == "adamw":
        return AdamW(
            get_grouped_params(model, weight_decay=weight_decay),
            lr=lr,
            weight_decay=weight_decay,
        )

    if name == "grokadamw":
        return GrokAdamW(
            get_grouped_params(model, weight_decay=weight_decay),
            lr=lr,
            weight_decay=weight_decay,
            gradient_clipping=config.gradient_clipping or 0.0,
        )

    # muon
    muon_params, adamw_params = muon_param_split(model)
    decay_ids = {
        id(p) for p in get_grouped_params(model, weight_decay=weight_decay)[0]["params"]
    }
    adamw_decay = [p for p in adamw_params if id(p) in decay_ids]
    adamw_no_decay = [p for p in adamw_params if id(p) not in decay_ids]
    optimizer = Muon(
        muon_params,
        lr=config.muon_learning_rate,
        adamw_params=adamw_decay,
        adamw_lr=lr,
        adamw_wd=weight_decay,
    )
    if adamw_no_decay:
        # Same defaults (lr, adamw_lr_ratio, betas) apart from the weight decay;
        # Muon reads state["use_muon"] for every parameter it steps.
        optimizer.add_param_group({"params": adamw_no_decay, "adamw_wd": 0.0})
        for p in adamw_no_decay:
            optimizer.state[p]["use_muon"] = False
    return optimizer


def current_learning_rates(optimizer):
    """
    The learning rates in effect now, for logging (the scheduler's factor included).

    "learning_rate" is always the rate of the parameters configured by
    config.learning_rate: every parameter for AdamW/GrokAdamW, Muon's AdamW path
    (embeddings, head, norms, biases) for Muon. Muon adds "muon_learning_rate",
    the rate of its orthogonalised updates. `optimizer` may be wrapped by Accelerate.
    """
    base = getattr(optimizer, "optimizer", optimizer)  # AcceleratedOptimizer
    group = base.param_groups[0]
    if isinstance(base, Muon):
        return {
            "learning_rate": group["adamw_lr_ratio"] * group["lr"],
            "muon_learning_rate": group["lr"],
        }
    return {"learning_rate": group["lr"]}
