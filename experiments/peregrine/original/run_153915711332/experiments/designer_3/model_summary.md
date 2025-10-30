# Quick Reference: Model Designer 3 Proposals

## Three Competing Hypotheses

### Model 1: Hierarchical Changepoint (PRIORITY 1)
**Story:** Discrete regime shift at unknown time point
**Key Feature:** Estimates changepoint location, regime-specific parameters
**Test:** Is there a real structural break or just smooth acceleration?
**Abandon if:** Changepoint posterior is diffuse (SD > 1.0) or not identified

### Model 2: Gaussian Process (PRIORITY 2)
**Story:** Smooth acceleration without discrete breaks
**Key Feature:** Flexible nonparametric function with NB likelihood
**Test:** Is growth smooth or discontinuous?
**Abandon if:** GP posterior shows sharp jumps at consistent locations

### Model 3: Latent State-Space (PRIORITY 3)
**Story:** Unobserved latent state evolving smoothly, counts are noisy realizations
**Key Feature:** Separates signal (state evolution) from noise (observation error)
**Test:** Is there meaningful distinction between latent and observed dynamics?
**Abandon if:** Latent state is just smoothed observed data (correlation > 0.99)

## Critical Decision Points

1. **After initial fits:** Are models comparable (ΔELPD < 10) or is one clearly better?
2. **After refinement:** Has performance improved or hit ceiling?
3. **After validation:** Does any model pass all stress tests?

## Red Flags for Major Pivot

- All models fail convergence (Rhat > 1.01)
- All models fail posterior predictive checks
- Dispersion parameter phi at extremes (< 1 or > 100) across all models
- Computational issues persist despite tuning

## Expected Outcomes

| Parameter | Expected Range | Model |
|-----------|---------------|--------|
| tau (changepoint) | [0.0, 0.5] | Model 1 |
| beta1_2 / beta1_1 | > 1.5 | Model 1 |
| rho (length scale) | [0.5, 2.0] | Model 2 |
| alpha (GP amplitude) | [0.3, 1.5] | Model 2 |
| sigma_w (state SD) | [0.05, 0.3] | Model 3 |
| phi (dispersion) | [5, 50] | All models |

## Success Criteria

- Rhat < 1.01 for all parameters
- ESS > 100 for key parameters
- Divergent transitions < 1%
- Posterior predictive checks pass
- LOO provides clear ranking

## Failure Response

If all three models fail:
1. Try simpler parametric NB GLM (quadratic or exponential)
2. Consider frequentist GEE with AR(1)
3. Question whether Bayesian inference is appropriate for N=40

**Remember:** Discovering that all models fail is scientific success—we learn what doesn't work!
