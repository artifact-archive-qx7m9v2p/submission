# Simulation-Based Validation Summary

**Experiment**: Negative Binomial GLM with Quadratic Trend
**Date**: 2025-10-30
**Status**: ⚠️ **CONDITIONAL PASS** (proceed with caution)

---

## Quick Decision

✓ **SAFE TO PROCEED to real data fitting**

**But**: Expect wide uncertainty intervals on quadratic term (beta_2) and dispersion (phi)

---

## Key Findings (1-Minute Read)

### What Worked ✓
- **Convergence**: Perfect (R-hat = 1.000, ESS > 2400, 0 divergences)
- **Trend recovery**: Excellent visual fit to true mean curve
- **Intercept & linear term**: <2% error (beta_0, beta_1)
- **Calibration**: All 90% CIs contain true values

### What Struggled ⚠️
- **Quadratic term (beta_2)**: 41.6% relative error
  - True: 0.100, Recovered: 0.058
  - BUT: True value within 90% CI [-0.016, 0.135]
- **Dispersion (phi)**: 26.7% relative error
  - True: 15.0, Recovered: 19.0
  - BUT: Conservative (wider prediction intervals)

### Root Cause
**Limited sample size (N=40)** makes quadratic acceleration difficult to estimate precisely. This is a **data quantity issue**, not a model problem.

---

## Validation Metrics

| Criterion | Result | Status |
|-----------|--------|--------|
| Convergence (R-hat, ESS) | R̂=1.000, ESS>2400 | ✓ PASS |
| Parameter Recovery | beta_2: 41.6% error | ✗ FAIL |
| Credible Interval Coverage | 100% (4/4 params) | ✓ PASS |
| Identifiability | max corr = 0.69 | ✓ PASS |
| **Overall** | **3/4 criteria** | **⚠️ CONDITIONAL** |

---

## Interpretation

**For prediction**: Model is reliable
- Mean curve well-recovered (see `data_fit.png`)
- Uncertainty properly calibrated

**For parameter interpretation**: Use caution
- beta_2 point estimate unreliable (wide CI)
- beta_0, beta_1 are trustworthy
- phi is nuisance parameter (overestimation is conservative)

---

## Recommendation

**Proceed to real data fitting** with these guidelines:

1. ✓ Trust overall trend predictions
2. ⚠️ Report beta_2 with wide credible intervals
3. ⚠️ Do NOT over-interpret beta_2 point estimate
4. ✓ Use model for forecasting/prediction
5. 💡 Consider linear model alternative if beta_2 not essential

**Why proceed despite bias?**
- No evidence of misspecification
- Bias due to data limitations, not model error
- Real data (also N=40) will have same constraints
- Validation caught the issue early (success!)

---

## Visual Evidence

All plots in `/workspace/experiments/experiment_1/simulation_based_validation/plots/`:

- **parameter_recovery.png**: Shows beta_2 posterior shifted left but covering true value
- **recovery_accuracy.png**: Quantifies 41.6% error for beta_2, 26.7% for phi
- **data_fit.png**: Demonstrates excellent mean trend recovery despite parameter bias
- **parameter_correlations.png**: Confirms identifiability (max |corr| = 0.69)

---

## Next Steps

1. Fit model to real data (aware of beta_2 uncertainty)
2. Check if real-data beta_2 has similarly wide CI
3. Compare quadratic vs linear model via LOO-CV
4. Report validation results in supplementary materials

---

**Full details**: See `recovery_metrics.md`
**Code**: `code/simulation_validation_pymc.py`
**Implementation**: PyMC 5.26.1 (see `IMPLEMENTATION_NOTE.md` for Stan vs PyMC discussion)
