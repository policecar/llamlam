# Implementation Plan for Improvements

This document outlines the plan to implement improvements identified in the code review.
Each phase represents a logical PR that builds on the previous one.

## Phase 1: Critical Bug Fixes (PR #2)

**Goal**: Fix bugs that affect training correctness
**Estimated files**: 4-5
**Risk**: Medium (changes training behavior)
**Dependencies**: Smoke test (PR #1)

### Changes:

1. **Fix gradient accumulation timing** (train.py:233)
   ```python
   # Before:
   if step % config.gradient_accumulation_steps == 0:

   # After:
   if (step + 1) % config.gradient_accumulation_steps == 0:
   ```
   - Add test to verify optimizer is called at correct intervals
   - Update smoke test to verify gradients accumulate properly

2. **Fix padding parameter** (data.py:11)
   ```python
   # Before:
   padding="True",

   # After:
   padding=True,
   ```
   - Add test for DataCollator with pad_to_multiple_of

3. **Fix checkpoint saving logic** (train.py:250-254)
   ```python
   # Before:
   if (global_step == 0) or (val_loss < min(val_losses)):
       best_val_loss = min(val_losses) if len(val_losses) else val_loss

   # After:
   if len(val_losses) == 0 or val_loss < min(val_losses):
       best_val_loss = val_loss
   ```

4. **Remove dead code** (model.py:53, 163; difftransformer.py:317)
   - Remove unused `self.scaling` or actually use it
   - Remove `if input_ids is not None:` checks that are always true
   - Remove `if x is not None:` checks that are always true

5. **Fix or remove attention_mask functionality** (model.py:143, 229)
   - Option A: Remove attention_mask parameter entirely (breaking change)
   - Option B: Implement attention_mask properly in Context module
   - Recommendation: Option A for now, add proper implementation in future PR
   - Update test_llamlam.py to remove test_attention_mask

### Testing Strategy:
- Run smoke test before and after, verify loss trajectories match
- Add unit test for gradient accumulation
- Add unit test for DataCollator
- Run full test suite

### Success Criteria:
- All tests pass
- Smoke test shows expected loss decrease
- No breaking changes to existing functionality (except attention_mask removal)

---

## Phase 2: Code Quality & Structure (PR #3)

**Goal**: Improve code quality without changing behavior
**Estimated files**: 8-10
**Risk**: Low (mostly refactoring)
**Dependencies**: Phase 1

### Changes:

1. **Centralize device management** (utils.py)
   ```python
   def get_device() -> torch.device:
       """Get the best available device (CUDA > MPS > CPU)."""
       if torch.cuda.is_available():
           return torch.device("cuda")
       elif torch.backends.mps.is_available():
           return torch.device("mps")
       return torch.device("cpu")
   ```
   - Replace duplicated device logic in train.py, model.py, difftransformer.py

2. **Add type hints throughout**
   - Add to all function signatures in:
     - model.py (forward, generate)
     - difftransformer.py (forward, generate)
     - utils.py (all functions)
     - data.py (DataCollator)

3. **Add config validation** (config.py)
   ```python
   def __post_init__(self):
       self.dim_embd = self.n_heads * self.dim_head
       self._validate()

   def _validate(self):
       assert self.n_heads > 0, "n_heads must be positive"
       assert self.dim_head > 0, "dim_head must be positive"
       assert self.dim_embd % self.n_heads == 0, "dim_embd must be divisible by n_heads"
       assert self.learning_rate > 0, "learning_rate must be positive"
       # ... more validations
   ```

4. **Add docstrings** (config.py, various)
   - Add comprehensive docstring to Config class
   - Document magic numbers (alpha values, initialization schemes)
   - Add module-level docstrings

5. **Remove nested torch.no_grad()** (model.py:229)
   - Remove redundant inner no_grad context

6. **Handle unused args.run_name** (train.py:84-86)
   - Either use it or remove the argument

### Testing Strategy:
- All existing tests should still pass
- Add tests for config validation
- Add tests for device selection utility
- Verify smoke test still works

### Success Criteria:
- No behavior changes
- All tests pass
- Type hints added to public APIs
- Config validation prevents invalid configurations

---

## Phase 3: Optimizer Fixes (PR #4)

**Goal**: Fix bugs in custom optimizers
**Estimated files**: 2
**Risk**: Medium (if anyone uses these optimizers)
**Dependencies**: Phase 2

### Changes:

1. **Fix GrokAdamW state access** (opt.py:165)
   ```python
   # Before:
   state = group["state"][p]

   # After:
   # This needs to be fixed in the _update_group signature
   # Change from static method to instance method, or pass state dict
   ```
   - Refactor _update_group to be instance method
   - Or pass self.state as parameter

2. **Fix gradient clipping** (opt.py:173)
   ```python
   # Before: (clips per-parameter, ineffective)
   torch.nn.utils.clip_grad_norm_(p, group["gradient_clipping"])

   # After: (collect all params, clip globally)
   # Move clipping outside the per-parameter loop
   if group["gradient_clipping"] > 0:
       params = [p for p in group["params"] if p.grad is not None]
       torch.nn.utils.clip_grad_norm_(params, group["gradient_clipping"])
   ```

3. **Add tests for custom optimizers**
   - Test GrokAdamW basic functionality
   - Test Muon basic functionality
   - Test that they actually update parameters
   - Test gradient clipping works

### Testing Strategy:
- Add tests/test_optimizers.py
- Test that optimizers can train a simple model
- Verify gradient clipping works as expected
- Run smoke test with each optimizer

### Success Criteria:
- GrokAdamW and Muon work correctly
- Tests demonstrate proper functionality
- Gradient clipping is effective

---

## Phase 4: Training Infrastructure (PR #5)

**Goal**: Improve training loop robustness and features
**Estimated files**: 3-4
**Risk**: Low
**Dependencies**: Phase 3

### Changes:

1. **Implement checkpoint cleanup** (train.py:253)
   ```python
   def cleanup_old_checkpoints(output_dir: Path, keep_best_k: int = 3):
       """Keep only the k best checkpoints based on validation loss."""
       # Implementation
   ```
   - Track validation loss for each checkpoint
   - Delete old checkpoints when exceeding keep_best_k

2. **Add proper error handling**
   - Wrap training loop in try-catch
   - Handle CUDA OOM errors gracefully
   - Add cleanup on keyboard interrupt
   - Log errors properly

3. **Fix tokens_seen usage** (train.py:239)
   - Either log it to wandb or remove it

4. **Add training resumption**
   - Implement TODO at train.py:218
   - Save/load global_step, optimizer state, scheduler state
   - Add --resume flag

5. **Better logging structure**
   ```python
   # Log to both wandb and logger consistently
   # Create utility function for logging metrics
   ```

### Testing Strategy:
- Test checkpoint cleanup with multiple checkpoints
- Test training resumption
- Verify error handling doesn't break training
- Update smoke test to test resumption (optional)

### Success Criteria:
- Old checkpoints are cleaned up automatically
- Training can resume from checkpoint
- Errors are handled gracefully
- All metrics logged consistently

---

## Phase 5: Model Architecture Improvements (PR #6)

**Goal**: Improve model architecture consistency
**Estimated files**: 3-4
**Risk**: Medium (changes model behavior)
**Dependencies**: Phase 4

### Changes:

1. **Standardize normalization**
   - Document why GPTModel uses LayerNorm vs DiffTransformer uses RMSNorm
   - Or standardize to one approach
   - Add config option to choose normalization type

2. **Consistent initialization**
   - Ensure all layers in Block are initialized
   - Document initialization strategy
   - Add tests to verify initialization

3. **Document/fix SwiGLU implementation** (activation.py:28-62)
   - Document why implementation differs from standard
   - Or align with standard: `W2(silu(W1(x)) * W3(x))`

4. **Add model comparison tests**
   - Test GPTModel vs DiffTransformer on same data
   - Verify both can overfit small dataset
   - Document performance characteristics

### Testing Strategy:
- Add tests for model initialization
- Test both models can train successfully
- Verify parameter counts are as expected
- Run smoke test with both models

### Success Criteria:
- Models are well-documented
- Initialization is consistent and tested
- Both models have test coverage

---

## Phase 6: Testing & Documentation (PR #7)

**Goal**: Complete test coverage and documentation
**Estimated files**: 5-10
**Risk**: Low
**Dependencies**: Phase 5

### Changes:

1. **Add missing tests**
   - tests/test_difftransformer.py (mirror test_llamlam.py)
   - tests/test_data.py (test DataCollator)
   - tests/test_config.py (test validation)
   - Improve existing tests

2. **Documentation improvements**
   - Expand README with:
     - Project overview
     - Architecture explanation
     - Training results
     - Model comparisons
   - Add CONTRIBUTING.md
   - Add architecture diagrams (optional)
   - Document magic numbers in code

3. **Remove or implement TeuxDeux comments**
   - model.py:216-219
   - difftransformer.py:372-375
   - README.md:63-65
   - Either implement these features or remove the TODOs

### Testing Strategy:
- Achieve >80% test coverage
- All tests pass
- Documentation is clear and accurate

### Success Criteria:
- Comprehensive test coverage
- Clear, complete documentation
- No hanging TODOs
- Easy for new contributors to understand codebase

---

## Summary Timeline

| Phase | Description | Files | Risk | Priority |
|-------|-------------|-------|------|----------|
| 1 | Critical bug fixes | 4-5 | Medium | P0 |
| 2 | Code quality | 8-10 | Low | P0 |
| 3 | Optimizer fixes | 2 | Medium | P1 |
| 4 | Training infrastructure | 3-4 | Low | P1 |
| 5 | Architecture improvements | 3-4 | Medium | P2 |
| 6 | Testing & docs | 5-10 | Low | P2 |

**Total estimated**: 25-35 files changed across 6 PRs

## Testing Strategy Across All Phases

For each phase:
1. Run smoke test before changes (baseline)
2. Make changes incrementally
3. Run unit tests after each change
4. Run smoke test after all changes
5. Compare smoke test results to baseline
6. Run full test suite
7. Manual verification where needed

## Rollout Strategy

1. **Phase 1 & 2** should be done together or back-to-back (critical fixes)
2. **Phase 3 & 4** can be done in parallel if needed
3. **Phase 5 & 6** should wait until earlier phases stabilize
4. Each PR should be reviewed and tested independently
5. Use feature flags for risky changes if needed

## Success Metrics

- All tests pass
- Smoke test completes in < 2 minutes
- Training runs complete successfully
- No regressions in model performance
- Code quality improves (linting, type checking)
- Documentation is comprehensive
