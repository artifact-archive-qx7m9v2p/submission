# Prior Predictive Check - Experiment 2

**Model**: Log-Linear Heteroscedastic Model
**Status**: CONDITIONAL PASS
**Date**: 2025-10-27

## Quick Summary

The prior specification is adequate for model fitting. Key results:
- 29.4% of prior samples generate data similar to observed (exceeds 20% threshold)
- 82.7% correctly predict decreasing variance with x
- No computational failures (0% negative sigma)
- Only 1.6% of samples generate negative Y values

**Main Concern**: Variance ratio distribution has heavy tails (median 21x vs observed 8.8x), but this will be constrained by the likelihood during fitting.

**Recommendation**: Proceed with fitting, monitor MCMC diagnostics.

## Files

```
prior_predictive_check/
├── README.md (this file)
├── findings.md (detailed analysis)
├── code/
│   ├── prior_predictive_check.py (main analysis)
│   ├── create_visualizations.py (plotting)
│   └── prior_predictive_samples.npz (saved data)
└── plots/
    ├── parameter_distributions.png
    ├── prior_predictive_coverage.png
    ├── variance_structure_diagnostic.png
    ├── mean_structure_diagnostic.png
    └── edge_cases_diagnostic.png
```

## Key Findings by Plot

### parameter_distributions.png
- All priors sample correctly from specified distributions
- gamma_1 allows 17.3% of samples with wrong-direction heteroscedasticity

### prior_predictive_coverage.png
- Observed data falls within prior predictive distribution
- 29.9% of samples cover both observed min and max

### variance_structure_diagnostic.png
- **Main concern**: Variance ratios show heavy right tail
- Median ratio: 21x (reasonable), but outliers reach 4762x
- 82.7% show correct decreasing variance

### mean_structure_diagnostic.png
- Log-linear relationship well-expressed
- Mean change prior: 1.08 vs observed: 0.77
- No concerning parameter interactions

### edge_cases_diagnostic.png
- Minimal pathological samples (1.6% with negative Y)
- No extreme sigma values (all < 10)
- Coefficient of variation decreases with x (correct pattern)

## Model Specification

```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = gamma_0 + gamma_1 * x_i

Priors:
  beta_0 ~ Normal(1.8, 0.5)
  beta_1 ~ Normal(0.3, 0.2)
  gamma_0 ~ Normal(-2, 1)
  gamma_1 ~ Normal(-0.05, 0.05)
```

## Next Steps

1. Fit model using MCMC
2. Check for divergent transitions
3. Verify posterior shrinkage (especially gamma parameters)
4. Perform posterior predictive checks
5. If issues arise, consider tightening gamma priors

## Contact

For questions about this analysis, see detailed findings in `findings.md`.
