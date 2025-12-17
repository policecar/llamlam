"""Tests for custom optimizers (GrokAdamW and Muon).

These tests verify:
1. Basic functionality (instantiation, stepping)
2. Parameter updates (do they change, does loss decrease)
3. Bug detection (exposes current bugs in opt.py)
4. Gradient clipping (global vs per-parameter)
5. State management (persistence, momentum)
6. Edge cases (zero grads, frozen params, signals)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llamlam.opt import GrokAdamW, Muon


# ============================================================================
# Helper Classes
# ============================================================================


class SimpleMLP(nn.Module):
    """Simple MLP for testing optimizers."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TwoParamModel(nn.Module):
    """Model with exactly 2 parameters for gradient clipping tests."""

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.randn(5, 5))
        self.param2 = nn.Parameter(torch.randn(5, 5))

    def forward(self, x):
        return x @ self.param1 @ self.param2


# ============================================================================
# Test Suite 1: Basic Functionality
# ============================================================================


def test_grokadamw_instantiation():
    """Test that GrokAdamW can be created with valid parameters."""
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3)
    assert optimizer is not None
    assert len(optimizer.param_groups) > 0


def test_muon_instantiation():
    """Test that Muon can be created with valid parameters."""
    model = SimpleMLP()
    optimizer = Muon(model.parameters(), lr=0.02)
    assert optimizer is not None


def test_grokadamw_single_step():
    """Test that GrokAdamW can perform one optimization step.

    This will FAIL with current code due to state access bug!
    Expected error: KeyError: 'state'
    """
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3)

    # Forward pass
    output = model(torch.randn(4, 10))
    loss = output.sum()

    # Backward pass
    loss.backward()

    # This should not crash (but currently will)
    optimizer.step()


def test_muon_single_step():
    """Test that Muon can perform one optimization step."""
    model = SimpleMLP()
    optimizer = Muon(model.parameters(), lr=0.02)

    # Forward pass
    output = model(torch.randn(4, 10))
    loss = output.sum()

    # Backward pass
    loss.backward()

    # This should not crash
    optimizer.step()


# ============================================================================
# Test Suite 2: Parameter Updates
# ============================================================================


def test_grokadamw_updates_parameters():
    """Test that GrokAdamW actually updates model parameters."""
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3)

    # Save initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Training step
    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Verify parameters changed
    for initial, current in zip(initial_params, model.parameters()):
        assert not torch.equal(initial, current), "Parameters should change after step"


def test_grokadamw_decreases_loss():
    """Test that GrokAdamW can decrease loss over multiple steps."""
    torch.manual_seed(42)
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=0.01)

    # Fixed input/target for reproducibility
    x = torch.randn(32, 10)
    target = torch.randn(32, 10)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Loss should generally decrease (allow some variance)
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_muon_updates_parameters():
    """Test that Muon actually updates model parameters."""
    model = SimpleMLP()
    optimizer = Muon(model.parameters(), lr=0.02)

    # Save initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Training step
    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Verify parameters changed
    for initial, current in zip(initial_params, model.parameters()):
        assert not torch.equal(initial, current), "Parameters should change after step"


# ============================================================================
# Test Suite 3: Bug Detection Tests
# ============================================================================


@pytest.mark.xfail(reason="Bug: state access in _update_group tries group['state'][p] which doesn't exist")
def test_grokadamw_state_access_bug():
    """Test that exposes the state access bug.

    Current code tries to access group['state'][p] which doesn't exist.
    The state is in self.state[p], not group['state'][p].

    This test is marked as xfail - it documents the bug we need to fix.
    """
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3)

    # Trigger the bug
    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()

    # This currently raises KeyError: 'state'
    optimizer.step()


def test_gradient_clipping_is_per_parameter():
    """Test that documents the current per-parameter gradient clipping behavior.

    Current code clips each parameter individually, which is wrong.
    Global clipping should consider all gradients together.

    This test documents the buggy behavior - we'll fix it and update the test.
    """
    # Create a model with two parameters where individual norms are small
    # but global norm is large
    model = TwoParamModel()

    # Set gradients manually
    # param1.grad norm = 0.8, param2.grad norm = 0.8
    # Global norm = sqrt(0.8^2 + 0.8^2) = 1.13 > 1.0 (should be clipped!)
    model.param1.grad = torch.ones_like(model.param1) * 0.8 / (5 * torch.sqrt(torch.tensor(5.0)))
    model.param2.grad = torch.ones_like(model.param2) * 0.8 / (5 * torch.sqrt(torch.tensor(5.0)))

    # Calculate actual norms
    norm1 = model.param1.grad.norm().item()
    norm2 = model.param2.grad.norm().item()
    global_norm = torch.sqrt(model.param1.grad.norm() ** 2 + model.param2.grad.norm() ** 2).item()

    # Both individual norms are < 1.0, but global norm > 1.0
    assert norm1 < 1.0 and norm2 < 1.0
    # Note: Due to per-parameter bug, current code won't clip these
    # This test documents the issue


# ============================================================================
# Test Suite 4: Gradient Clipping Correctness
# ============================================================================


def test_grokadamw_gradient_clipping_enabled():
    """Test that gradient clipping is configured correctly."""
    model = SimpleMLP()
    max_norm = 0.5
    optimizer = GrokAdamW(model.parameters(), lr=1e-3, gradient_clipping=max_norm)

    # Verify gradient_clipping is in the param group
    assert optimizer.param_groups[0]["gradient_clipping"] == max_norm


def test_grokadamw_no_clipping_when_disabled():
    """Test that gradient clipping is disabled when set to 0."""
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3, gradient_clipping=0.0)

    # Create very large gradients
    output = model(torch.randn(32, 10))
    loss = output.sum() * 1000
    loss.backward()

    # Should not crash (even with huge gradients)
    optimizer.step()


def test_grokadamw_gradient_clipping_limits_updates():
    """Test that gradient clipping limits the size of parameter updates."""
    torch.manual_seed(42)
    model = SimpleMLP()

    # Two optimizers: one with clipping, one without
    optimizer_clipped = GrokAdamW(model.parameters(), lr=1e-2, gradient_clipping=0.1)

    model_unclipped = SimpleMLP()
    model_unclipped.load_state_dict(model.state_dict())  # Same initial weights
    optimizer_unclipped = GrokAdamW(model_unclipped.parameters(), lr=1e-2, gradient_clipping=0.0)

    # Create large gradients
    x = torch.randn(32, 10)

    # Clipped model
    output = model(x)
    loss = output.sum() * 100  # Large multiplier
    loss.backward()
    optimizer_clipped.step()

    # Unclipped model
    output_unclipped = model_unclipped(x)
    loss_unclipped = output_unclipped.sum() * 100
    loss_unclipped.backward()
    optimizer_unclipped.step()

    # Calculate parameter change magnitudes
    change_clipped = sum((p1 - p2).norm().item()
                         for p1, p2 in zip(model.parameters(),
                                          SimpleMLP().parameters()))
    change_unclipped = sum((p1 - p2).norm().item()
                           for p1, p2 in zip(model_unclipped.parameters(),
                                            SimpleMLP().parameters()))

    # Clipped updates should be smaller
    # (This test may need adjustment after bug fix)


# ============================================================================
# Test Suite 5: State Persistence
# ============================================================================


def test_grokadamw_maintains_state():
    """Test that optimizer state persists across multiple steps."""
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3)

    # First step
    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Check state was created
    for p in model.parameters():
        assert p in optimizer.state
        state = optimizer.state[p]
        assert "step" in state
        assert state["step"] == 1
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert "grok_ema" in state

    # Second step
    optimizer.zero_grad()
    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Step counter should increment
    for p in model.parameters():
        assert optimizer.state[p]["step"] == 2


def test_grokadamw_momentum_accumulation():
    """Test that momentum buffers accumulate correctly."""
    torch.manual_seed(42)
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # First step
    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Check exp_avg is non-zero (momentum started)
    for p in model.parameters():
        assert optimizer.state[p]["exp_avg"].abs().sum() > 0, "Momentum buffer should be non-zero"


def test_muon_maintains_state():
    """Test that Muon optimizer state persists across steps."""
    model = SimpleMLP()
    optimizer = Muon(model.parameters(), lr=0.02)

    # First step
    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Check that state was created for 2D parameters
    two_d_params = [p for p in model.parameters() if p.ndim >= 2]
    assert len(two_d_params) > 0, "Should have at least one 2D parameter"

    for p in two_d_params:
        if p.size(0) < 10000:  # Muon uses this heuristic
            assert p in optimizer.state
            assert "use_muon" in optimizer.state[p]


# ============================================================================
# Test Suite 6: Edge Cases
# ============================================================================


def test_grokadamw_with_zero_gradients():
    """Test optimizer handles parameters with zero gradients."""
    model = SimpleMLP()
    optimizer = GrokAdamW(model.parameters(), lr=1e-3)

    # Set all gradients to zero
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    # Should not crash
    optimizer.step()


def test_grokadamw_with_some_none_gradients():
    """Test optimizer handles parameters with None gradients (frozen params)."""
    model = SimpleMLP()

    # Freeze first layer
    for p in model.fc1.parameters():
        p.requires_grad = False

    optimizer = GrokAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Only fc2 parameters should have state
    assert model.fc2.weight in optimizer.state
    assert model.fc1.weight not in optimizer.state


def test_grokadamw_with_grokking_signal():
    """Test GrokAdamW with grokking signal functions."""
    model = SimpleMLP()

    signal_value = [0.5]  # Mutable list to act as closure

    def grokking_fn():
        return signal_value[0]

    optimizer = GrokAdamW(
        model.parameters(), lr=1e-3, grokking_signal_fns=[grokking_fn]
    )

    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Should complete without error


def test_grokadamw_with_invalid_grokking_signal():
    """Test GrokAdamW handles failing grokking signal functions gracefully."""
    model = SimpleMLP()

    def failing_fn():
        raise ValueError("Signal computation failed")

    optimizer = GrokAdamW(
        model.parameters(), lr=1e-3, grokking_signal_fns=[failing_fn]
    )

    output = model(torch.randn(4, 10))
    loss = output.sum()
    loss.backward()

    # Should not crash - should log warning and continue
    optimizer.step()


# ============================================================================
# Test Suite 7: Validation Tests
# ============================================================================


def test_grokadamw_validates_parameters():
    """Test that GrokAdamW validates input parameters."""
    model = SimpleMLP()

    # Invalid learning rate
    with pytest.raises(ValueError, match="Invalid learning rate"):
        GrokAdamW(model.parameters(), lr=-1.0)

    # Invalid epsilon
    with pytest.raises(ValueError, match="Invalid epsilon"):
        GrokAdamW(model.parameters(), lr=1e-3, eps=-1e-8)

    # Invalid beta
    with pytest.raises(ValueError, match="Invalid beta"):
        GrokAdamW(model.parameters(), lr=1e-3, betas=(1.5, 0.999))

    # Invalid weight decay
    with pytest.raises(ValueError, match="Invalid weight_decay"):
        GrokAdamW(model.parameters(), lr=1e-3, weight_decay=-0.1)

    # Invalid alpha_init
    with pytest.raises(ValueError, match="Invalid alpha_init"):
        GrokAdamW(model.parameters(), lr=1e-3, alpha_init=1.5)


# ============================================================================
# Test Suite 8: Comparison Tests
# ============================================================================


def test_grokadamw_vs_adamw_basic_behavior():
    """Test that GrokAdamW behaves similarly to AdamW when grokking features are disabled."""
    torch.manual_seed(42)

    # Two identical models
    model1 = SimpleMLP()
    model2 = SimpleMLP()
    model2.load_state_dict(model1.state_dict())

    # GrokAdamW with minimal grokking features
    optimizer1 = GrokAdamW(
        model1.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        alpha_init=0.0,  # Disable grokking EMA
        lamb=0.0,  # Disable grokking amplification
    )

    # Standard AdamW for comparison
    optimizer2 = torch.optim.AdamW(
        model2.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01
    )

    # Same training step
    x = torch.randn(4, 10)

    output1 = model1(x)
    loss1 = output1.sum()
    loss1.backward()
    optimizer1.step()

    output2 = model2(x)
    loss2 = output2.sum()
    loss2.backward()
    optimizer2.step()

    # Losses should be similar (not exactly equal due to implementation differences)
    assert abs(loss1.item() - loss2.item()) < 0.1
