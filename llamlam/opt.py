"""
Up-and-coming optimizers:

- GrokAdamW from https://github.com/cognitivecomputations/grokadamw
- Muon from https://github.com/KellerJordan/Muon/ (the Nov-2024 version with the
  built-in AdamW backup, commit d7ee0fe; fixes follow upstream's later 28c793b)

@decision DEC-OPT-001: Optimizer state is created lazily, on each parameter's device.
@rationale: Matches torch.optim; eager CUDA allocation broke CPU/MPS, add_param_group
    and load_state_dict, and allocated state for parameters that never get gradients.
@decision DEC-OPT-002: Parameters whose .grad is None are skipped (no state, no weight
    decay), as in torch.optim and upstream Muon 28c793b.
@rationale: Unused parameters (e.g. frozen or conditionally used modules) must not
    move. In distributed Muon the owner's slot stays zero, so all ranks agree.
@decision DEC-OPT-003: GrokAdamW bias-corrects the first moment with the layer-wise
    beta1 it was accumulated with, not the plain beta1 (differs from upstream).
@rationale: Upstream's mismatch makes early steps far too large for decayed layers;
    see "Differences from upstream" in the GrokAdamW docstring.
@decision DEC-OPT-004: GrokAdamW's layer-wise beta1 decay uses the parameter's position
    normalised by its group size, so gamma spans the whole group (differs from upstream).
@rationale: Upstream's raw position drives beta1 to ~0 for most tensors of a real
    model; see "Differences from upstream" in the GrokAdamW docstring.
@decision DEC-OPT-005: In distributed Muon every rank updates every Muon parameter's
    momentum buffer; only the orthogonalisation is split across ranks (differs from
    upstream, which keeps each buffer on its owning rank only).
@rationale: Accelerate saves the optimizer state of one process and every rank loads
    it, so per-rank buffers restarted from zero on resume for the parameters other
    ranks own. Gradients are identical on all ranks after DDP's all-reduce (train.py
    syncs on every backward and clips the synced gradients), so the buffers are too.
    Cost: one full copy of Muon's momentum per rank.
"""

import math
import os
from typing import Callable, Iterable, Optional

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


class GrokAdamW(Optimizer):
    """AdamW with a Grokfast gradient filter and layer-wise beta1 decay.

    Args:
        params: Parameters or parameter groups to optimize.
        lr, betas, eps, weight_decay: As in AdamW (decoupled weight decay).
        alpha_init: Rate of the Grokfast gradient EMA (higher = slower EMA).
        lamb: Weight of the Grokfast EMA added to the gradient.
        gamma: Total fractional reduction of beta1 across a param group: the
            group's first parameter uses beta1, its last beta1 * (1 - gamma),
            geometrically in between. gamma=0 gives plain beta1 everywhere.
        grokking_signal_fns: Callables whose mean (ignoring None) slows the EMA:
            alpha = alpha_init * exp(-grokking_signal_decay_rate * signal).
        grokking_signal_decay_rate: See grokking_signal_fns.
        gradient_clipping: Per-tensor max gradient norm; 0 disables clipping.

    Differences from upstream:
        A. Bias correction (DEC-OPT-003). Before: the first moment was averaged
           with the layer-wise beta1 but bias-corrected with the plain beta1
           (1 - beta1 ** step). Now: it is corrected with 1 - layer_beta1 ** step.
           Why: the correction must match the average it corrects. With upstream's
           decay at the defaults, parameter index 10 has layer_beta1 ~= 0.314, so
           its first step was (1 - 0.314) / (1 - 0.9) ~= 6.9x too large (fading
           as steps accumulate).
        B. Layer-wise decay (DEC-OPT-004). Before: layer_beta1 = beta1 *
           (1 - gamma) ** i, with i the raw position in the param group. Now: the
           exponent is i / (n - 1) for a group of n parameters (0 if n == 1).
           Why: real groups hold dozens to hundreds of tensors and 0.9 ** 50 ~=
           0.005, so most parameters trained with almost no momentum, depending on
           how many tensors preceded them. Now the decay is gentle and spans the
           group whatever its length.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        alpha_init: float = 0.98,
        lamb: float = 2.0,
        gamma: float = 0.1,
        grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
        grokking_signal_decay_rate: float = 0.1,
        gradient_clipping: float = 1.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError(f"Invalid alpha_init value: {alpha_init}")
        if not 0.0 <= lamb:
            raise ValueError(f"Invalid lamb value: {lamb}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= grokking_signal_decay_rate:
            raise ValueError(
                f"Invalid grokking_signal_decay_rate value: {grokking_signal_decay_rate}"
            )
        if not 0.0 <= gradient_clipping:
            raise ValueError(f"Invalid gradient_clipping value: {gradient_clipping}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha_init=alpha_init,
            lamb=lamb,
            gamma=gamma,
            grokking_signal_fns=grokking_signal_fns,
            grokking_signal_decay_rate=grokking_signal_decay_rate,
            gradient_clipping=gradient_clipping,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure, use_amp=False)

    @torch.no_grad()
    def step_amp(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        return self._step_impl(closure, use_amp=True)

    def _step_impl(
        self, closure: Optional[Callable[[], float]], use_amp: bool
    ) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            use_amp (bool): Run the update under torch.autocast for the parameters'
                device type. The update is in-place elementwise math, which autocast
                does not recast, so this matches step().

        Returns:
            Optional[float]: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)

            # Keep each parameter's index in the group: layer-wise beta1 decay is by
            # position and must not shift when an earlier parameter has no grad.
            indexed = [
                (i, p) for i, p in enumerate(group["params"]) if p.grad is not None
            ]
            if not indexed:
                continue

            if use_amp:
                with torch.autocast(device_type=indexed[0][1].device.type):
                    self._update_group(group, indexed, grokking_signal)
            else:
                self._update_group(group, indexed, grokking_signal)

        return loss

    def _compute_grokking_signal(self, group: dict) -> Optional[float]:
        """Mean of the signals returned by grokking_signal_fns, ignoring None.

        Exceptions raised by a signal function propagate: a broken signal is a
        misconfiguration and must not silently turn Grokfast amplification off.
        """
        if group["grokking_signal_fns"] is None:
            return None

        signals = [
            s for s in (fn() for fn in group["grokking_signal_fns"]) if s is not None
        ]
        if not signals:
            return None
        return sum(signals) / len(signals)

    def _update_group(
        self,
        group: dict,
        indexed_params: list[tuple[int, torch.Tensor]],
        grokking_signal: Optional[float],
    ) -> None:
        beta1, beta2 = group["betas"]
        alpha = group["alpha_init"]
        if grokking_signal is not None:
            alpha = alpha * math.exp(
                -group["grokking_signal_decay_rate"] * grokking_signal
            )
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    f"Grokking signal {grokking_signal} gives EMA alpha {alpha} outside "
                    "[0, 1]; signals must be non-negative."
                )

        # Positions run over the whole group, with or without grads (see _step_impl).
        last_index = len(group["params"]) - 1

        for i, p in indexed_params:
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["grok_ema"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            state["step"] += 1

            # Per-tensor gradient clipping (upstream behaviour; modifies p.grad in place)
            if group["gradient_clipping"] > 0:
                torch.nn.utils.clip_grad_norm_(p, group["gradient_clipping"])
            grad = p.grad

            # Layer-wise momentum decay, normalised by group size (upstream diff B)
            position = i / last_index if last_index > 0 else 0.0
            layer_beta1 = beta1 * (1 - group["gamma"]) ** position

            # Grokfast component
            grok_ema = state["grok_ema"]
            grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
            grok_grad = grad + group["lamb"] * grok_ema

            # AdamW update with Grokfast-amplified gradient
            exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

            # Bias correction must use the beta1 that exp_avg was accumulated with
            # (upstream diff A)
            bias_correction1 = 1 - layer_beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

            # Decoupled weight decay (from AdamW)
            p.mul_(1 - group["lr"] * group["weight_decay"])

            # Update parameters
            p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Runs in bfloat16 on every device (as upstream). Not wrapped in torch.compile so it
    works without a compiler toolchain; callers may compile it themselves.
    """
    if G.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor, got shape {tuple(G.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)  # ensure top singular value <= 1; never mutates G
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.

    Distributed: if WORLD_SIZE is set, each rank orthogonalizes every WORLD_SIZE-th Muon
    parameter and the updates are summed with dist.all_reduce (process group must exist).
    Every rank updates the momentum buffer of every Muon parameter, so each rank holds
    a full copy of Muon's momentum (as large as the Muon parameters, in the gradients'
    dtype, as in single-process training) rather than about 1/WORLD_SIZE of it; in
    exchange any rank's state_dict resumes all ranks (DEC-OPT-005). Requires identical
    gradients on all ranks at step(), as after DDP's gradient all-reduce.

    Raises ValueError for invalid hyperparameters, in add_param_group too: lr must be
    > 0 in the constructor (>= 0 in an added group); the AdamW lr, eps and weight
    decay >= 0; the AdamW betas in [0, 1).
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
    ):
        if not lr > 0.0:
            # adamw lr is stored as a ratio of lr so that lr schedulers scale both
            raise ValueError(f"Invalid learning rate: {lr} (must be > 0)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not ns_steps >= 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        # Materialize first: both arguments may be one-shot iterators (model.parameters()).
        muon_params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        muon_ids = {id(p) for p in muon_params}
        if any(id(p) in muon_ids for p in adamw_params):
            raise ValueError("A parameter is in both muon_params and adamw_params")
        super().__init__(muon_params + adamw_params, defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def add_param_group(self, param_group):
        """torch's add_param_group, after checking the group's own values or the
        defaults it will take. Also runs for the constructor's group."""
        value = {k: param_group.get(k, v) for k, v in self.defaults.items()}
        beta1, beta2 = value["adamw_betas"]
        if not 0.0 <= value["lr"]:
            raise ValueError(f"Invalid learning rate: {value['lr']}")
        if not 0.0 <= value["adamw_lr_ratio"]:
            raise ValueError(
                f"Invalid adamw_lr: {value['adamw_lr_ratio'] * value['lr']} "
                f"(adamw_lr_ratio={value['adamw_lr_ratio']})"
            )
        if not 0.0 <= value["adamw_eps"]:
            raise ValueError(f"Invalid adamw_eps value: {value['adamw_eps']}")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid adamw_betas: {value['adamw_betas']}")
        if not 0.0 <= value["adamw_wd"]:
            raise ValueError(f"Invalid adamw_wd value: {value['adamw_wd']}")
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]

            if params:
                # generate weight updates in distributed fashion
                total_params = sum(p.numel() for p in params)
                updates_flat = torch.zeros(
                    total_params, device=params[0].device, dtype=torch.bfloat16
                )
                curr_idx = 0
                for i, p in enumerate(params):
                    if p.grad is None:
                        curr_idx += p.numel()
                        continue
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    # Every rank keeps every buffer (DEC-OPT-005): p.grad is the same
                    # on all ranks, so the buffers are too, and any rank's state resumes.
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                    if i % self.world_size == self.rank:
                        if group["nesterov"]:
                            g = g.add(buf, alpha=momentum)
                        else:
                            g = buf
                        g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                        updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                    curr_idx += p.numel()

                # sync updates across devices. we are not memory-constrained so can do this simple deserialization
                if self.world_size > 1:
                    dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

                # deserialize and apply updates. Applied to every param regardless of the
                # local .grad: the owning rank decided, and a skipped param's slot is zero.
                curr_idx = 0
                for p in params:
                    g = (
                        updates_flat[curr_idx : curr_idx + p.numel()]
                        .view_as(p)
                        .type_as(p)
                    )
                    p.add_(g, alpha=-lr)
                    curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = (
                group["adamw_lr_ratio"] * group["lr"]
            )  # in order for lr schedule to work
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.mul_(1 - lr * weight_decay)
                p.add_(g, alpha=-lr / scale)

        return loss
