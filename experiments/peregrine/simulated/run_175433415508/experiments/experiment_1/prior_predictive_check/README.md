# Prior Predictive Check - Experiment 1

**Model:** Negative Binomial Linear Model (Baseline)
**Status:** PASS ✓
**Date:** 2025-10-29

---

## Quick Summary

The prior predictive check validates that the specified priors generate scientifically plausible data before model fitting. All validation criteria passed successfully.

**Decision: Proceed to model fitting with current prior specification.**

---

## Model Under Test

```
C_t ~ NegativeBinomial(mu_t, phi)
log(mu_t) = beta_0 + beta_1 * year_t

Priors:
  beta_0 ~ Normal(4.69, 1.0)    # Intercept (log scale)
  beta_1 ~ Normal(1.0, 0.5)      # Growth rate
  phi ~ Gamma(2, 0.1)            # Overdispersion parameter
```

---

## Key Results

### Validation Criteria (6/6 PASS)

- ✓ No negative counts (0 violations)
- ✓ 99.2% of counts in reasonable range [0, 5000]
- ✓ Only 0.3% extreme outliers (>10,000)
- ✓ 99% of samples show positive growth
- ✓ Prior covers observed mean (109.4)
- ✓ Prior covers observed maximum (269)

### Prior Predictive Statistics

| Statistic | Prior Mean | Prior Range | Observed |
|-----------|------------|-------------|----------|
| Dataset mean | 353.5 | [4.8, 3220] | 109.4 |
| Dataset max | 1842.7 | [12, 23593] | 269 |
| Growth factor | 158.6x | [0.7x, 3007x] | ~9x |

### Parameter Samples

| Parameter | Mean | SD | Range |
|-----------|------|----|----|
| beta_0 | 4.66 | 1.00 | [1.46, 6.66] |
| beta_1 | 1.06 | 0.51 | [-0.13, 2.40] |
| phi | 19.43 | 12.81 | [1.41, 56.32] |

---

## Visualizations

Six diagnostic plots generated in `/workspace/experiments/experiment_1/prior_predictive_check/plots/`:

1. **`parameter_plausibility.png`** - Prior distributions sampling correctly
2. **`prior_predictive_coverage.png`** - Prior predictions cover observed data range
3. **`prior_summary_diagnostics.png`** - Dataset-level statistics (mean, var, min, max)
4. **`count_distribution_diagnostic.png`** - Overall count distribution
5. **`endpoint_diagnostics.png`** - Expected counts at time endpoints
6. **`comprehensive_diagnostic.png`** - Single-page overview (recommended starting point)

**Recommended viewing order:** Start with `comprehensive_diagnostic.png` for overview, then examine specific plots as needed.

---

## Key Findings

### What's Working Well

1. **Parameters are plausible**: All three priors generate reasonable parameter values
2. **Strong growth signal**: 99% of samples favor positive growth as expected
3. **No domain violations**: No negative counts or impossible values
4. **Good coverage**: Observed data well within prior predictive range

### Expected Behavior

The prior generates some extreme growth trajectories (up to 23,000 counts), which represents:
- **<1% of simulations** (long right tail)
- **Appropriate prior uncertainty** before seeing data
- **Will be constrained** by the posterior during fitting

This is exactly what good priors should look like: diffuse enough to avoid overconfidence, but structured enough to encode domain knowledge.

### What to Monitor Post-Fitting

1. Check if posterior maintains extreme tails (would indicate misspecification)
2. Watch for divergences during MCMC sampling
3. Compare prior vs posterior width (expect substantial shrinkage)

---

## Files Generated

```
/workspace/experiments/experiment_1/prior_predictive_check/
├── README.md                           # This file
├── findings.md                         # Detailed analysis
├── code/
│   ├── prior_predictive_check.py      # Reproducible script
│   ├── prior_predictive.stan          # Stan model (reference)
│   └── results.json                   # Numerical results
└── plots/
    ├── comprehensive_diagnostic.png    # Overview (start here)
    ├── parameter_plausibility.png
    ├── prior_predictive_coverage.png
    ├── prior_summary_diagnostics.png
    ├── count_distribution_diagnostic.png
    └── endpoint_diagnostics.png
```

---

## Reproducibility

Run the check again:
```bash
python /workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_check.py
```

All results are reproducible (random seed = 42).

---

## Conclusion

**PASS** - The model is ready for fitting. The priors appropriately encode domain knowledge while maintaining sufficient uncertainty. No modifications to prior specification or model structure are needed.

**Next step:** Fit the model using MCMC and conduct posterior predictive checks.

---

For detailed analysis and visual evidence, see `findings.md`.
