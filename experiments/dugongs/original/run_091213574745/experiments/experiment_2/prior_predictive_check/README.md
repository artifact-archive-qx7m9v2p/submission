# Prior Predictive Check - Experiment 2 (Student-t Model)

**Status**: PASS
**Date**: 2025-10-28

## Quick Summary

All validation checks passed for the Student-t logarithmic model:
- Domain violations: 0.06% extreme, 0.08% moderate (PASS)
- Negative slopes: 2.3% (PASS)
- Large sigma: 0.0% (PASS)
- Nu exploration: Balanced across tail regimes (PASS)
- Coverage: Observed data within prior envelope (PASS)

**Key Finding**: Nu prior Gamma(2, 0.1) successfully explores both robust (nu<20: 57%) and near-Normal (nu>30: 21%) behavior.

## Model Specification

```
Y_i ~ StudentT(nu, mu_i, sigma)
mu_i = beta_0 + beta_1 * log(x_i)

Priors:
beta_0 ~ Normal(2.3, 0.5)
beta_1 ~ Normal(0.29, 0.15)
sigma ~ Exponential(10)
nu ~ Gamma(2, 0.1)
```

## Files

- `code/run_prior_predictive_studentt.py`: Prior sampling and validation
- `code/visualize_prior_predictive.py`: Plotting code
- `code/prior_samples.npz`: Saved samples (1000 draws)
- `plots/`: 7 diagnostic plots
- `summary_stats.json`: Numerical results
- `findings.md`: Comprehensive analysis

## Key Plots

1. **parameter_plausibility.png**: All 4 parameter marginals and pairs
2. **nu_tail_behavior_diagnostic.png**: How nu affects tail behavior (6 panels)
3. **prior_predictive_coverage.png**: 100 curves color-coded by nu regime
4. **studentt_vs_normal_comparison.png**: Comparison to Model 1 (Normal)

## Decision

**PROCEED** to model fitting with current priors.

## Next Steps

1. Fit model using MCMC (Stan or PyMC)
2. Monitor posterior nu - key diagnostic for model comparison
3. Compare to Model 1 using LOO
4. If nu > 30, prefer Model 1 (simpler)
5. If nu < 10, Student-t justified

## Comparison to Model 1

- Same functional form (log-linear)
- Similar priors for beta_0, beta_1, sigma
- Added nu parameter for tail flexibility
- Prior predictive ~17% wider ranges (appropriate for heavy tails)
- Both models pass all validation checks

See `findings.md` for full analysis.

## IMPORTANT UPDATE

**Recommended Prior Modification**: Use nu ~ Gamma(2, 0.1) **truncated to [3, Inf]** instead of unbounded.

**Reason**: 0.9% of draws have nu < 1, causing numerical instabilities (extreme outliers). Truncating at nu=3 removes pathological cases while retaining all scientifically meaningful tail behaviors.

**Implementation** (Stan):
```stan
real<lower=3> nu;
nu ~ gamma(2, 0.1);
```

See ADDENDUM in findings.md for full details.
