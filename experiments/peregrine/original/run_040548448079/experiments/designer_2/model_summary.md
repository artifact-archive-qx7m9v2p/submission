# Quick Reference: Smooth Nonlinear Models

## Three Model Classes

### 1. Gaussian Process Regression
- **Strength**: Maximum flexibility, captures arbitrary smooth functions
- **Weakness**: O(N³) computation, may over-smooth discrete breaks
- **Abandon if**: Lengthscale → 0, LOO >> 20 worse than changepoint

### 2. Penalized B-Spline Regression
- **Strength**: Local flexibility, computationally efficient
- **Weakness**: Knot sensitivity, may show artifacts
- **Abandon if**: Derivative discontinuous at obs 17, knot-dependent results

### 3. Polynomial Regression (Baseline)
- **Strength**: Simple, fast, interpretable
- **Weakness**: Limited flexibility, may miss local patterns
- **Abandon if**: LOO >> 20 worse than GP, systematic bias at obs 17

## Key Decision Criteria

| LOO-ELPD Difference | Interpretation |
|---------------------|----------------|
| Smooth ≈ Changepoint (±10) | Smooth acceleration sufficient |
| Smooth -10 to -20 worse | Borderline, check residuals |
| Smooth < -20 worse | Discrete break real, use changepoint |

## Implementation Order

1. **Polynomial** (2-3 hrs): Fast baseline
2. **GP or Spline** (4-6 hrs): Flexible alternatives
3. **Comparison** (2-3 hrs): LOO + diagnostics
4. **Decision** (1-2 hrs): Select or reject smooth models

## Critical Tests

1. **Leave-future-out CV**: Hold out last 5 obs
2. **Residual ACF**: Should be white noise
3. **First derivative**: Should be continuous
4. **LOO comparison**: vs changepoint models

## Expected Outcome

**Most likely**: Smooth models fail, discrete break confirmed.
**Reason**: 730% growth rate change at obs 17 suggests true regime shift.
**Action**: Document failure, recommend changepoint models.

---
See `/workspace/experiments/designer_2/proposed_models.md` for full specifications.
