# Prior Predictive Check - Experiment 1

This directory contains the prior predictive check for **Experiment 1: Robust Logarithmic Regression**.

## Status: FAILED

The prior predictive check identified critical issues with the prior specification that must be addressed before model fitting.

## Quick Start

**To view results:**
1. Read `findings.md` for comprehensive analysis
2. View plots in `plots/` directory (start with `comprehensive_summary.png`)
3. Check `code/diagnostics.json` for numerical results

**To reproduce:**
```bash
cd code/
python run_prior_predictive_numpy.py
```

## Directory Structure

```
prior_predictive_check/
├── README.md                           (this file)
├── findings.md                         (comprehensive analysis and recommendations)
├── code/
│   ├── run_prior_predictive_numpy.py  (main analysis script)
│   ├── prior_predictive.stan          (Stan model - not used due to installation)
│   └── diagnostics.json               (numerical results)
└── plots/
    ├── comprehensive_summary.png       (START HERE - multi-panel overview)
    ├── parameter_plausibility.png      (marginal prior distributions)
    ├── prior_predictive_curves.png     (100 prior predictive curves)
    ├── prior_predictive_coverage.png   (prior predictive intervals)
    ├── predictions_at_key_x_values.png (distributions at x_min, x_mid, x_max)
    ├── extrapolation_diagnostic.png    (behavior at x=50)
    └── monotonicity_diagnostic.png     (increasing vs decreasing curves)
```

## Key Findings

### Problems Identified

1. **Sigma prior too wide:** Half-Cauchy(0, 0.2) generates extreme values
   - 34% of datasets contain values outside [0.5, 4.5]
   - 12.1% of samples have Y < 0 (impossible for this data)
   - Compound effect: heavy-tailed prior + heavy-tailed Student-t likelihood

2. **Beta prior too diffuse:** Normal(0.3, 0.3) allows 16% negative slopes
   - EDA shows strong positive relationship (R² = 0.888)
   - 13.9% of prior curves decrease (should be <10%)

3. **Poor scale alignment:** Only 39.3% of prior means near observed data
   - Indicates priors not concentrated on plausible region

### Checks Failed (4/7)

- Check 1: Predictions in [0.5, 4.5] - 65.9% (target: ≥80%) ❌
- Check 2: Monotonically increasing - 86.1% (target: ≥90%) ❌
- Check 5: No extreme negative - 12.1% (target: <5%) ❌
- Check 7: Mean within ±2 SD - 39.3% (target: ≥70%) ❌

## Recommended Prior Adjustments

```stan
// ORIGINAL (FAILED)
alpha ~ normal(2.0, 0.5);
beta ~ normal(0.3, 0.3);
c ~ gamma(2, 2);
nu ~ gamma(2, 0.1);
sigma ~ half_cauchy(0, 0.2);

// REVISED (RECOMMENDED)
alpha ~ normal(2.0, 0.5);         // unchanged - working well
beta ~ normal(0.3, 0.2);          // tightened: 0.3 → 0.2
c ~ gamma(2, 2);                  // unchanged - working well
nu ~ gamma(2, 0.1);               // unchanged - working well
sigma ~ normal(0, 0.15);          // changed: half_cauchy → half_normal
                                  // (with lower bound: real<lower=0> sigma;)
```

### Justification

**Sigma change (primary fix):**
- Half-Normal has lighter tails than Half-Cauchy
- SD=0.15 gives E[sigma]≈0.12 (~45% of observed Y_SD=0.27)
- 95% of mass below 0.3 (eliminates extreme values)
- Still flexible for model misspecification

**Beta change (secondary fix):**
- Reduces P(beta < 0) from 16% to ~7%
- Maintains mean at EDA estimate (0.3)
- Still allows beta up to 0.7 in 95% interval

## Next Steps

1. **Update model specification** with revised priors
2. **Re-run prior predictive check** with new priors
3. **Verify all checks pass** (target: 7/7 passing)
4. **Proceed to simulation-based calibration**
5. **Fit to real data** only after validation passes

## Why This Matters

Prior predictive checks are cheap (2 minutes runtime) but catch expensive problems:
- Prevents wasting time fitting a misspecified model
- Identifies numerical instabilities before MCMC sampling
- Ensures priors encode genuine domain knowledge, not arbitrary defaults

**DO NOT skip this step** - it's foundational to principled Bayesian workflow.

## Technical Notes

- Used NumPy/SciPy instead of Stan (CmdStan installation issues)
- Results equivalent for prior predictive checks (no MCMC needed)
- 1000 prior samples with random seed 42
- Runtime: ~2 minutes

## References

- Gabry et al. (2019). "Visualization in Bayesian workflow"
- Gelman et al. (2020). "Bayesian Workflow"
- Prior predictive checking: sample from priors → generate data → assess plausibility

---

**Status:** FAIL - Prior adjustment required before proceeding
**Date:** 2025-10-27
