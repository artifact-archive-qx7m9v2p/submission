# CRITICAL FINDING: Simulation-Based Validation Detected Implementation Bug

**Date**: 2025-10-30
**Experiment**: 2 (AR(1) Log-Normal with Regime-Switching)
**Status**: FAIL (as designed)

## What Happened

Simulation-based calibration successfully detected a **critical implementation error** in the AR(1) model before it could be applied to real data.

### The Evidence

1. **Parameter Recovery Failure**: MLE beta_1 = 0.390, True beta_1 = 0.860 (54.6% error)
2. **Likelihood Inconsistency**:
   - NLL at true parameters: 13.64
   - NLL at MLE parameters: 10.82
   - Δ NLL = 2.82 (HUGE difference)

3. **Multiple Initialization Test**: 9/10 different starting points converged to the SAME wrong solution
   - Not an optimization problem
   - Not an identifiability issue
   - **Data generation and likelihood are inconsistent**

## Root Cause Analysis

The bug is in the AR(1) initialization/structure. Specifically:

**In data generation** (`simulation_validation_simple.py` lines 85-88):
```python
epsilon[0] = np.random.normal(0, sigma_init)
log_C[0] = mu_trend[0] + epsilon[0]
epsilon[0] = log_C[0] - mu_trend[0]  # ← OVERWRITES epsilon[0]
```

This creates a **circular definition** where:
- First we draw epsilon[0] from N(0, sigma_init)
- Then we use it to generate log_C[0]
- Then we REDEFINE epsilon[0] as the residual

But the residual epsilon[0] is NOT the same as the draw from N(0, sigma_init) unless we got extremely lucky!

**In likelihood** (`simulation_validation_simple.py` lines 183-187):
```python
sigma_init = sigma_regime[regime_idx[0]] / np.sqrt(1 - phi**2)
epsilon_0 = log_C[0] - mu_trend[0]
ll += norm.logpdf(epsilon_0, loc=0, scale=sigma_init)
```

This correctly evaluates the density of the RESIDUAL, but the residual was not actually drawn from this distribution in data generation!

## Why This Matters

This is **exactly what simulation-based calibration is designed to catch**:
- If the model can't recover parameters from data it supposedly generated
- There's a fundamental problem with the model implementation
- Applying this buggy model to real data would give **nonsense results**

## The Correct Implementation

For AR(1), we should either:

**Option A**: Generate epsilon first, then observations
```python
epsilon[0] ~ N(0, sigma / sqrt(1 - phi^2))  # Draw from stationary distribution
log_C[0] = mu_trend[0] + epsilon[0]          # Use the drawn epsilon
# Don't redefine epsilon[0]!
```

**Option B**: Generate log_C directly
```python
log_C[0] ~ N(mu_trend[0], sigma / sqrt(1 - phi^2))  # Marginal distribution
epsilon[0] = log_C[0] - mu_trend[0]                   # Compute residual
```

Both are equivalent, but Option A makes the AR structure clearer.

## Decision

**VALIDATION RESULT**: ✗ FAIL (correctly!)

**Actions Required**:
1. ✓ Fix data generation to be consistent with likelihood
2. ✓ Re-run simulation-based validation with corrected implementation
3. Only proceed to real data after validation PASSES

## Lessons Learned

This demonstrates the **critical value of simulation-based calibration**:
- Caught a subtle but critical bug before real data analysis
- Prevented publication of incorrect results
- Validated the validation framework itself (it works!)

**Time saved**: Hours of debugging why real data fit gives weird results
**Credibility saved**: Not publishing results from buggy model

## Status

Proceeding to implement corrected version and re-validate.
