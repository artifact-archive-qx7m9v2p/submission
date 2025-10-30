# Designer 2: Non-parametric/Flexible Models

**Modeling Philosophy**: Let the data reveal its functional form without imposing rigid parametric assumptions.

## Quick Summary

I propose **3 Bayesian models** with increasing flexibility:

1. **GP-NegBin**: Gaussian Process with negative binomial likelihood
2. **P-spline GLM**: Penalized B-splines with automatic smoothness tuning
3. **Semi-parametric**: Logistic growth + GP deviations + time-varying overdispersion

## Key Distinctions from Other Designers

- **Designer 1 (Parametric GLMs)**: I avoid assuming specific functional forms (quadratic, exponential)
- **Designer 3 (Hierarchical/Temporal)**: I focus on flexible mean functions rather than temporal correlation structure

## Expected Outcome

**My prediction**: P-splines (Model 2) will perform best for n=40
- GP might overfit (too flexible)
- Semi-parametric might be too complex (convergence issues)
- Splines offer good flexibility/simplicity tradeoff

## Falsification Mindset

I will abandon flexible models if:
- Simple parametric models (Designer 1) have better LOO-CV
- Posterior concentrates on near-linear functions (flexibility wasted)
- Computational failures despite tuning (geometry too difficult)
- Holdout predictions are terrible (overfitting)

## Files

- `proposed_models.md`: Full model specifications with priors, implementation code, falsification criteria
- `README.md`: This file

## Implementation Priority

1. Fit Model 2 (P-splines) first - most likely to succeed
2. Fit Model 1 (GP) second - diagnostic comparison
3. Fit Model 3 (Semi-parametric) if time allows - most complex

## Contact Points with Other Designers

**Compare with Designer 1**:
- If my models don't beat their quadratic NegBin, flexibility is unjustified

**Compare with Designer 3**:
- If temporal models beat mine, autocorrelation structure more important than flexible mean
- Consider hybrid: flexible trend + AR errors

---

**Status**: Design complete, ready for implementation phase
**Date**: 2025-10-29
