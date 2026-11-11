"""Tests for llamlam.opt (GrokAdamW, Muon). All run on CPU; MPS is exercised if present.

Built on plain torch.nn modules so they do not depend on the repo's models.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn

from llamlam.opt import GrokAdamW, Muon, zeropower_via_newtonschulz5


def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
    )


def _data(seed=1, n=256):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    w = torch.randn(8, 1, generator=g)
    return x, torch.sin(x @ w)


def _loss(model, x, y):
    return nn.functional.mse_loss(model(x), y)


def _train(model, opt, steps, x, y):
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = _loss(model, x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def _snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def _available_devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


# --------------------------------------------------------------------------- #
# GrokAdamW
# --------------------------------------------------------------------------- #


def test_grokadamw_repeated_steps_reduce_loss():
    model = _mlp()
    x, y = _data()
    opt = GrokAdamW(model.parameters(), lr=1e-2)
    losses = _train(model, opt, 200, x, y)
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < 0.2 * losses[0], (losses[0], losses[-1])


def test_grokadamw_matches_adamw_with_grokfast_and_layer_decay_off():
    """lamb=0 removes the Grokfast term, gamma=0 removes layer-wise beta1 decay,
    gradient_clipping=0 removes per-tensor clipping: what remains is AdamW.

    The only intended difference is eps placement: GrokAdamW uses the Adam-paper
    form lr*sqrt(bc2)/bc1 * m / (sqrt(v) + eps) (as upstream does), torch uses
    lr/bc1 * m / (sqrt(v)/sqrt(bc2) + eps). With eps=0 they are algebraically
    identical, so compare with eps=0 tightly (tanh MLP => no exactly-zero grads).
    """
    x, y = _data()
    ref, grok = _mlp(), _mlp()
    kw = dict(lr=1e-2, betas=(0.9, 0.99), eps=0.0, weight_decay=0.1)
    opt_ref = torch.optim.AdamW(ref.parameters(), **kw)
    opt_grok = GrokAdamW(
        grok.parameters(), lamb=0.0, gamma=0.0, gradient_clipping=0.0, **kw
    )
    for _ in range(5):
        _train(ref, opt_ref, 1, x, y)
        _train(grok, opt_grok, 1, x, y)
        for a, b in zip(ref.parameters(), grok.parameters()):
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
    # Sanity: the parameters actually moved, so the comparison is not vacuous.
    assert any(
        (a - b).abs().max() > 1e-3
        for a, b in zip(ref.parameters(), _mlp().parameters())
    )


def test_grokadamw_first_step_is_lr_sign_grad_for_every_layer():
    """Adam's bias-corrected first step is exactly lr*sign(g) (eps=0, no wd).
    That must hold with layer-wise beta1 decay too, i.e. bias correction must
    use the same (decayed) beta1 that the first moment was accumulated with."""
    model = _mlp()
    x, y = _data()
    lr = 1e-2
    opt = GrokAdamW(
        model.parameters(),
        lr=lr,
        eps=0.0,
        weight_decay=0.0,
        lamb=0.0,
        gamma=0.5,
        gradient_clipping=0.0,
    )
    before = _snapshot(model)
    opt.zero_grad()
    _loss(model, x, y).backward()
    grads = [p.grad.clone() for p in model.parameters()]
    opt.step()
    for p0, p1, g in zip(before, model.parameters(), grads):
        torch.testing.assert_close(p1 - p0, -lr * torch.sign(g), rtol=1e-4, atol=1e-7)


def test_grokadamw_layer_beta_does_not_depend_on_other_params_grads():
    """A parameter's layer-wise beta1 is tied to its position in the group, not
    to how many earlier parameters happened to have a gradient this step.
    (Gradients change between steps; with a constant gradient bias-corrected
    Adam is independent of beta1 and the test could not tell betas apart.)"""
    torch.manual_seed(0)
    g1, g2 = torch.randn(5), torch.randn(5)
    finals = []
    for first_has_grad in (True, False):
        params = [nn.Parameter(torch.zeros(5)) for _ in range(3)]
        opt = GrokAdamW(
            params,
            lr=1e-2,
            gamma=0.5,
            lamb=0.0,
            weight_decay=0.0,
            gradient_clipping=0.0,
        )
        for g in (g1, g2):
            params[0].grad = g.clone() if first_has_grad else None
            params[1].grad, params[2].grad = g.clone(), g.clone()
            opt.step()
        finals.append([p.detach().clone() for p in params])
        if not first_has_grad:
            assert len(opt.state[params[0]]) == 0
    # Non-vacuous: positions 1 and 2 use different betas, so they end up apart.
    assert (finals[0][1] - finals[0][2]).abs().max() > 1e-4
    torch.testing.assert_close(finals[1][2], finals[0][2], rtol=0, atol=0)


# Layer-wise beta1 is observed through exp_avg after two steps with gradients g1, g2
# (lamb=0, no clipping => the Grokfast gradient is the raw gradient):
# exp_avg = b * (1 - b) * g1 + (1 - b) * g2, which identifies b. float64, exact maths.
_PLAIN = dict(lr=1e-2, eps=0.0, weight_decay=0.0, lamb=0.0, gradient_clipping=0.0)


def _exp_avg_after_two_steps(b, g1, g2):
    return b * (1 - b) * g1 + (1 - b) * g2


def _two_steps(opt, params, grads1, grads2):
    """grads*: one gradient (or None) per parameter, per step."""
    for grads in (grads1, grads2):
        for p, g in zip(params, grads):
            p.grad = None if g is None else g.clone()
        opt.step()


def _randn64(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).double()


def test_grokadamw_layer_decay_spans_a_large_group():
    """beta1 decays from beta1 (first param) to beta1 * (1 - gamma) (last param)
    however many tensors the group holds. Upstream's raw-index exponent gave the
    last of 200 params beta1 * 0.9 ** 199 ~= 7e-10, i.e. no momentum at all."""
    n, beta1, beta2, gamma, lr = 200, 0.9, 0.999, 0.1, _PLAIN["lr"]
    g1, g2 = _randn64(n, 3, seed=0), _randn64(n, 3, seed=1)
    params = [nn.Parameter(torch.zeros(3, dtype=torch.float64)) for _ in range(n)]
    opt = GrokAdamW(params, betas=(beta1, beta2), gamma=gamma, **_PLAIN)
    _two_steps(opt, params, g1, g2)

    for i in (0, n // 2, n - 1):
        b = beta1 * (1 - gamma) ** (i / (n - 1))
        exp_avg = opt.state[params[i]]["exp_avg"]
        expected = _exp_avg_after_two_steps(b, g1[i], g2[i])
        torch.testing.assert_close(exp_avg, expected, rtol=1e-12, atol=1e-15)
        b_old = beta1 * (1 - gamma) ** i
        if i > 0:  # non-vacuous: the old formula gives a clearly different state
            old = _exp_avg_after_two_steps(b_old, g1[i], g2[i])
            assert (expected - old).abs().max() > 0.1

    # Bias correction uses the same beta1: replay Adam (eps=0) for the last param.
    b, g1l, g2l = beta1 * (1 - gamma), g1[-1], g2[-1]
    v2 = beta2 * (1 - beta2) * g1l**2 + (1 - beta2) * g2l**2
    m2 = _exp_avg_after_two_steps(b, g1l, g2l)
    expected_p = (
        -lr * torch.sign(g1l)
        - lr * math.sqrt(1 - beta2**2) / (1 - b**2) * m2 / v2.sqrt()
    )
    torch.testing.assert_close(params[-1].detach(), expected_p, rtol=1e-12, atol=1e-15)


def test_grokadamw_layer_decay_counts_params_without_grad():
    """Position and group size cover the whole group, not just the params that
    have a gradient this step: with only the last two of five params trained,
    they still use exponents 3/4 and 1."""
    n, beta1, gamma = 5, 0.9, 0.5
    g1, g2 = _randn64(n, 4, seed=2), _randn64(n, 4, seed=3)
    params = [nn.Parameter(torch.zeros(4, dtype=torch.float64)) for _ in range(n)]
    opt = GrokAdamW(params, betas=(beta1, 0.999), gamma=gamma, **_PLAIN)
    mask = [None, None, None]
    _two_steps(opt, params, mask + [g1[3], g1[4]], mask + [g2[3], g2[4]])

    for p in params[:3]:
        assert len(opt.state[p]) == 0
    for i in (3, 4):
        b = beta1 * (1 - gamma) ** (i / (n - 1))
        expected = _exp_avg_after_two_steps(b, g1[i], g2[i])
        torch.testing.assert_close(
            opt.state[params[i]]["exp_avg"], expected, rtol=1e-12, atol=1e-15
        )


def test_grokadamw_single_param_group_uses_plain_beta1():
    """Decay is per param group; a one-parameter group (exponent 0) uses beta1."""
    beta1, gamma = 0.9, 0.5
    g1, g2 = _randn64(4, 2, seed=4), _randn64(4, 2, seed=5)
    params = [nn.Parameter(torch.zeros(2, dtype=torch.float64)) for _ in range(4)]
    opt = GrokAdamW(
        [{"params": params[:3]}, {"params": params[3:]}],
        betas=(beta1, 0.999),
        gamma=gamma,
        **_PLAIN,
    )
    _two_steps(opt, params, g1, g2)

    for i, b in ((2, beta1 * (1 - gamma)), (3, beta1)):
        expected = _exp_avg_after_two_steps(b, g1[i], g2[i])
        torch.testing.assert_close(
            opt.state[params[i]]["exp_avg"], expected, rtol=1e-12, atol=1e-15
        )


def test_grokadamw_grokking_signal_fns_set_ema_rate():
    model = nn.Linear(4, 3)
    x = torch.randn(16, 4)
    calls = []

    def signal():
        calls.append(1)
        return 2.0

    def no_signal():
        return None

    alpha_init, rate = 0.9, 0.5
    opt = GrokAdamW(
        model.parameters(),
        alpha_init=alpha_init,
        grokking_signal_fns=[signal, no_signal],
        grokking_signal_decay_rate=rate,
        gradient_clipping=0.0,
    )
    model(x).square().mean().backward()
    g = model.weight.grad.clone()
    opt.step()
    alpha = alpha_init * math.exp(-rate * 2.0)  # None signals are ignored
    torch.testing.assert_close(opt.state[model.weight]["grok_ema"], (1 - alpha) * g)
    assert len(calls) == 1
    # Repeated steps keep working.
    for _ in range(3):
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
    assert len(calls) == 4


def test_grokadamw_signal_fn_errors_propagate():
    model = nn.Linear(4, 3)

    def broken():
        raise RuntimeError("eval loss unavailable")

    opt = GrokAdamW(model.parameters(), grokking_signal_fns=[broken])
    model(torch.randn(2, 4)).sum().backward()
    with pytest.raises(RuntimeError, match="eval loss unavailable"):
        opt.step()


def test_grokadamw_signal_outside_valid_alpha_range_raises():
    model = nn.Linear(4, 3)
    opt = GrokAdamW(model.parameters(), grokking_signal_fns=[lambda: -10.0])
    model(torch.randn(2, 4)).sum().backward()
    with pytest.raises(ValueError, match="alpha"):
        opt.step()


def test_grokadamw_skips_params_without_grad():
    used, unused = nn.Linear(4, 3), nn.Linear(4, 3)
    opt = GrokAdamW(list(used.parameters()) + list(unused.parameters()), lr=1e-2)
    before_used, before_unused = _snapshot(used), _snapshot(unused)
    for _ in range(3):
        opt.zero_grad()
        used(torch.randn(8, 4)).square().mean().backward()
        opt.step()
    for p0, p in zip(before_unused, unused.parameters()):
        assert p.grad is None
        assert torch.equal(p0, p)  # no weight decay applied either
        assert len(opt.state[p]) == 0
    for p0, p in zip(before_used, used.parameters()):
        assert not torch.equal(p0, p)
        assert opt.state[p]["step"] == 3


def test_grokadamw_step_amp_matches_step_on_cpu():
    x, y = _data()
    a, b = _mlp(), _mlp()
    opt_a, opt_b = (
        GrokAdamW(a.parameters(), lr=1e-2),
        GrokAdamW(b.parameters(), lr=1e-2),
    )
    for _ in range(3):
        for model, opt, use_amp in ((a, opt_a, False), (b, opt_b, True)):
            opt.zero_grad()
            _loss(model, x, y).backward()
            opt.step_amp() if use_amp else opt.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


def test_grokadamw_closure_returns_loss():
    model = _mlp()
    x, y = _data()
    opt = GrokAdamW(model.parameters())

    def closure():
        opt.zero_grad()
        loss = _loss(model, x, y)
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lr=-1.0),
        dict(gamma=-0.1),
        dict(gamma=1.5),
        dict(lamb=-1.0),
        dict(gradient_clipping=-1.0),
        dict(grokking_signal_decay_rate=-0.1),
    ],
)
def test_grokadamw_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        GrokAdamW(nn.Linear(2, 2).parameters(), **kwargs)


# --------------------------------------------------------------------------- #
# Muon
# --------------------------------------------------------------------------- #


def test_muon_accepts_generator_and_reduces_loss():
    """model.parameters() is a generator; Muon must classify every parameter."""
    model = _mlp()
    x, y = _data()
    opt = Muon(model.parameters(), lr=0.02, adamw_lr=3e-3)
    for p in model.parameters():
        assert opt.state[p]["use_muon"] == (p.ndim >= 2)
    losses = _train(model, opt, 200, x, y)
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < 0.2 * losses[0], (losses[0], losses[-1])


@pytest.mark.parametrize("shape", [(32, 64), (48, 48), (64, 32)])
def test_muon_update_is_approximately_orthogonal(shape):
    """Upstream: the update is ~ U S' V^T with S'_ii in roughly (0.5, 1.5),
    scaled by sqrt(max(1, rows/cols)), and applied with step size lr."""
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(*shape))
    lr = 0.1
    opt = Muon([w], lr=lr)
    w.grad = 10.0 * torch.randn(*shape)
    before = w.detach().clone()
    opt.step()
    update = (before - w.detach()) / lr
    scale = max(1.0, shape[0] / shape[1]) ** 0.5
    sv = torch.linalg.svdvals(update.double()) / scale
    assert sv.min() > 0.5 and sv.max() < 1.5, sv
    # Centred on 1 (catches a missing or wrong rows/cols scale for tall matrices).
    assert 0.85 < sv.median() < 1.15, sv
    assert "momentum_buffer" in opt.state[w]


def test_muon_routes_1d_and_adamw_params_to_adamw():
    torch.manual_seed(0)
    hidden = nn.Linear(16, 16)  # weight -> Muon, bias (1-D) -> AdamW
    head = nn.Linear(16, 4)  # passed via adamw_params -> AdamW for both
    big = nn.Parameter(
        torch.randn(10000, 2)
    )  # embedding-like (size(0) >= 10000) -> AdamW
    adamw_lr = 1e-3
    opt = Muon(
        list(hidden.parameters()) + [big],
        lr=0.02,
        adamw_params=head.parameters(),
        adamw_lr=adamw_lr,
        adamw_eps=0.0,
    )
    adamw_routed = [hidden.bias, head.weight, head.bias, big]
    for p in adamw_routed:
        assert opt.state[p]["use_muon"] is False
    assert opt.state[hidden.weight]["use_muon"] is True

    x = torch.randn(32, 16)
    loss = head(torch.tanh(hidden(x))).square().mean() + big.square().mean()
    loss.backward()
    grads = {p: p.grad.clone() for p in adamw_routed}
    before = {p: p.detach().clone() for p in adamw_routed}
    opt.step()
    for p in adamw_routed:
        # Adam's first step is exactly lr * sign(g) (eps=0, no weight decay).
        torch.testing.assert_close(
            p.detach() - before[p],
            -adamw_lr * torch.sign(grads[p]),
            rtol=1e-4,
            atol=1e-8,
        )
        assert set(opt.state[p]) == {"use_muon", "step", "moment1", "moment2"}
    assert "momentum_buffer" in opt.state[hidden.weight]
    assert "moment1" not in opt.state[hidden.weight]


def test_muon_without_nesterov_orthogonalizes_the_momentum_buffer():
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(8, 8))
    momentum, lr = 0.9, 0.1
    opt = Muon([w], lr=lr, momentum=momentum, nesterov=False)
    g1, g2 = torch.randn(8, 8), torch.randn(8, 8)
    w.grad = g1.clone()
    opt.step()
    before = w.detach().clone()
    w.grad = g2.clone()
    opt.step()
    expected = (
        zeropower_via_newtonschulz5(momentum * g1 + g2, steps=6).bfloat16().float()
    )
    torch.testing.assert_close(
        (before - w.detach()) / lr, expected, rtol=2e-2, atol=2e-2
    )
    # The raw gradient alone would be a clearly different update.
    raw = zeropower_via_newtonschulz5(g2, steps=6).float()
    assert (raw - expected).abs().max() > 0.1


def test_muon_skips_params_without_grad():
    used, unused = nn.Linear(6, 6), nn.Linear(6, 6)
    opt = Muon(list(used.parameters()) + list(unused.parameters()), lr=0.02)
    before_used, before_unused = _snapshot(used), _snapshot(unused)
    for _ in range(3):
        opt.zero_grad()
        used(torch.randn(16, 6)).square().mean().backward()
        opt.step()
    for p0, p in zip(before_unused, unused.parameters()):
        assert p.grad is None
        assert torch.equal(p0, p)
        assert set(opt.state[p]) == {"use_muon"}  # no buffers created
    for p0, p in zip(before_used, used.parameters()):
        assert not torch.equal(p0, p)


def test_muon_closure_and_no_grad():
    model = _mlp()
    x, y = _data()
    opt = Muon(model.parameters())

    def closure():
        opt.zero_grad()
        loss = _loss(model, x, y)
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    # Parameters stay leaves without grad history after an in-place update.
    for p in model.parameters():
        assert p.is_leaf and p.grad_fn is None


def test_zeropower_does_not_mutate_bfloat16_input_and_rejects_1d():
    G = torch.randn(8, 4, dtype=torch.bfloat16)
    G_copy = G.clone()
    zeropower_via_newtonschulz5(G, steps=3)
    assert torch.equal(G, G_copy)
    with pytest.raises(ValueError):
        zeropower_via_newtonschulz5(torch.randn(4), steps=3)


def test_muon_rejects_invalid_configuration():
    lin = nn.Linear(2, 2)
    with pytest.raises(ValueError):
        Muon(lin.parameters(), lr=0.0)
    with pytest.raises(ValueError):
        Muon(lin.parameters(), momentum=1.0)
    with pytest.raises(ValueError):
        Muon(lin.parameters(), ns_steps=0)
    with pytest.raises(ValueError, match="both"):
        Muon([lin.weight], adamw_params=[lin.weight])


# --------------------------------------------------------------------------- #
# Shared: lr schedulers, devices
# --------------------------------------------------------------------------- #


def _grokadamw(params):
    return GrokAdamW(params, lr=1e-2)


def _muon(params):
    return Muon(params, lr=0.02, adamw_lr=1e-2)


@pytest.mark.parametrize("make_opt", [_grokadamw, _muon], ids=["grokadamw", "muon"])
def test_lr_scheduler_controls_step_size(make_opt):
    """Step 1 at the base lr moves every parameter; after the scheduler drops lr
    to 0, step 2 must leave every parameter (Muon and AdamW paths) untouched."""
    model = _mlp()
    x, y = _data()
    opt = make_opt(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 if s == 0 else 0.0)
    before = _snapshot(model)
    _train(model, opt, 1, x, y)
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    sched.step()
    assert opt.param_groups[0]["lr"] == 0.0
    after_first = _snapshot(model)
    _train(model, opt, 1, x, y)
    for a, b in zip(after_first, model.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("make_opt", [_grokadamw, _muon], ids=["grokadamw", "muon"])
def test_optimizers_run_on_parameter_device(make_opt, device):
    model = _mlp().to(device)
    x, y = (t.to(device) for t in _data())
    opt = make_opt(model.parameters())
    losses = _train(model, opt, 20, x, y)
    assert losses[-1] < losses[0]
    for p in model.parameters():
        for v in opt.state[p].values():
            if isinstance(v, torch.Tensor):
                assert v.device == p.device


def test_state_dict_roundtrip_continues_identically():
    x, y = _data()
    for make_opt in (_grokadamw, _muon):
        model = _mlp()
        opt = make_opt(model.parameters())
        _train(model, opt, 3, x, y)
        clone = copy.deepcopy(model)
        opt2 = make_opt(clone.parameters())
        # deepcopy, as torch.save/load would: load_state_dict reuses the tensors when
        # dtype/device already match, so the two optimizers would share buffers.
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        _train(model, opt, 2, x, y)
        _train(clone, opt2, 2, x, y)
        for a, b in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
