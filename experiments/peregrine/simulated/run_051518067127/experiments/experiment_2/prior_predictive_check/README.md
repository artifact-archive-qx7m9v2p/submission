# Prior Predictive Check - Experiment 2: AR(1) Log-Normal with Regime-Switching

**Status**: FAIL - Critical issues identified
**Date**: 2025-10-30
**Prior Draws**: 1,000

---

## Quick Summary

### Decision: FAIL

**Three critical issues must be addressed before fitting:**

1. **Autocorrelation prior mismatch**: phi ~ U(-0.95, 0.95) generates ACF centered at 0, but observed ACF = 0.961
2. **Extreme predictions**: 5.8% of predictions exceed 1,000 (max: 348 million!)
3. **Low coverage**: Only 2.8% of prior draws produce plausible data [10, 500]

### Recommended Action

Implement revised priors:
- **phi ~ Beta(20, 2) rescaled to (0, 0.95)** instead of U(-0.95, 0.95)
- **sigma_regime ~ HalfNormal(0, 0.5)** instead of HalfNormal(0, 1)
- **beta_1 ~ Normal(0.86, 0.15)** instead of Normal(0.86, 0.2)

Then re-run prior predictive check before proceeding to fitting.

---

## Files in This Directory

### Documentation
- **`findings.md`** - Full detailed analysis (read this for complete assessment)
- **`README.md`** - This file (quick reference)

### Code
- **`code/prior_predictive_check.py`** - Main implementation with AR(1) sequential generation
- **`code/create_summary_plot.py`** - Summary visualization generator

### Key Visualizations

1. **`plots/summary_critical_issues.png`** - START HERE - Shows all 3 critical issues
2. **`plots/prior_autocorrelation_diagnostic.png`** - Most critical: ACF mismatch
3. **`plots/prior_predictive_coverage.png`** - Shows extreme prediction intervals
4. **`plots/parameter_plausibility.png`** - All 7 parameter prior distributions
5. **`plots/prior_trajectories.png`** - Sample paths showing AR temporal smoothness
6. **`plots/regime_variance_diagnostic.png`** - 4-panel regime structure analysis
7. **`plots/log_scale_diagnostic.png`** - Count vs log-scale comparison

---

## Critical Statistics

```
Domain Violations:
  - Extreme high (>1000):     5.8% of predictions
  - Extreme low (<1):         1.2% of predictions

Plausibility:
  - Fully in plausible range: 2.8% (FAIL - should be >50%)

Autocorrelation:
  - Observed ACF lag-1:       0.961
  - Prior ACF median:         -0.059
  - Prior 90% CI:             [-0.860, 0.772]
  - Covers observed:          NO (CRITICAL FAIL)
```

---

## What's Working

- AR(1) structure correctly generates temporally smooth trajectories
- Log-scale trend is well-specified
- No computational red flags (no NaN/Inf)
- Regime boundaries don't create discontinuities
- Implementation quality is correct

## What Needs Fixing

- Phi prior doesn't encode domain knowledge of high autocorrelation
- Sigma priors too wide, creating heavy right tail
- Prior-data disconnect will cause poor convergence if fitted as-is

---

## Next Steps

1. Update model specification with revised priors
2. Re-run: `python code/prior_predictive_check.py`
3. Verify improvements in `findings.md`
4. Proceed to simulation validation only after PASS

**DO NOT proceed to fitting with current priors!**

---

For detailed analysis and full recommendations, see **`findings.md`**.
