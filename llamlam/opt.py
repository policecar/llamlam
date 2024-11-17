"""
Up-and-coming optimizers:

- GrokAdamW from https://github.com/cognitivecomputations/grokadamw
- Muon from https://github.com/KellerJordan/Muon/

"""

import os
import torch

import torch.distributed as dist

from torch.optim.optimizer import Optimizer
from torch.cuda.amp import autocast
from typing import Iterable, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrokAdamW(Optimizer):
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
        super(GrokAdamW, self).__init__(params, defaults)

        # Pre-allocate state tensors and move to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p] = {}
                state["step"] = 0
                state["exp_avg"] = torch.empty_like(
                    p, memory_format=torch.preserve_format
                ).to(device)
                state["exp_avg_sq"] = torch.empty_like(
                    p, memory_format=torch.preserve_format
                ).to(device)
                state["grok_ema"] = torch.empty_like(
                    p, memory_format=torch.preserve_format
                ).to(device)

                # Initialize tensors
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                state["grok_ema"].zero_()

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
            use_amp (bool): Whether to use automatic mixed precision (AMP).

        Returns:
            Optional[float]: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)

            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue

            grads = [p.grad for p in params_with_grad]

            # Function to apply parameter updates
            def _apply_updates():
                self._update_group(group, params_with_grad, grads, grokking_signal)

            if use_amp:
                with autocast():
                    _apply_updates()
            else:
                _apply_updates()

        return loss

    def _compute_grokking_signal(self, group: dict) -> Optional[float]:
        """Computes a combined grokking signal from multiple functions."""
        if group["grokking_signal_fns"] is None:
            return None

        signals = []
        for fn in group["grokking_signal_fns"]:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(
                    f"Error in grokking_signal_fn: {e}. Ignoring this function."
                )

        if not signals:
            return None

        # Example: Taking the mean of all valid signals
        return sum(signals) / len(signals)

    @staticmethod
    def _update_group(
        group: dict,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        grokking_signal: Optional[float],
    ) -> None:
        for i, (p, grad) in enumerate(zip(params, grads)):
            state = group["state"][p]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Apply gradient clipping if enabled
            if group["gradient_clipping"] > 0:
                torch.nn.utils.clip_grad_norm_(p, group["gradient_clipping"])

            # Layer-wise momentum decay
            layer_beta1 = beta1 * (1 - group["gamma"]) ** i

            # Grokfast component
            grok_grad = GrokAdamW._update_grok_ema(grad, state, group, grokking_signal)

            # AdamW update with Grokfast-amplified gradient
            exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

            # AdamW bias correction
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = group["lr"] * torch.sqrt(bias_correction2) / bias_correction1

            # Decoupled weight decay (from AdamW)
            p.mul_(1 - group["lr"] * group["weight_decay"])

            # Update parameters
            p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)

    @staticmethod
    def _update_grok_ema(
        grad: torch.Tensor, state: dict, group: dict, grokking_signal: Optional[float]
    ) -> torch.Tensor:
        grok_ema = state["grok_ema"]
        alpha = group["alpha_init"]
        if grokking_signal is not None:
            alpha = alpha * torch.exp(
                -group["grokking_signal_decay_rate"] * grokking_signal
            )
        grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
        return grad + group["lamb"] * grok_ema


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
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

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

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

    def step(self):
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(g, alpha=-lr)
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
                assert g is not None
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
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
