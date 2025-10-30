# Posterior Predictive Check - Experiment 1

## Overview

Comprehensive posterior predictive checks for the Log-Linear Negative Binomial model fitted in Experiment 1.

## Files

### Analysis Code
- `code/ppc.py` - Full posterior predictive check implementation with 1000 samples

### Results
- `ppc_results.json` - Quantitative metrics and falsification test results
- `ppc_findings.md` - Comprehensive findings report with interpretation

### Diagnostic Plots (7 plots)

1. **timeseries_fit.png** - Observed vs predicted time series with 90% prediction intervals
2. **residuals.png** - Four-panel residual diagnostics (vs time, vs fitted, Q-Q plot, distribution)
3. **early_vs_late_fit.png** - Side-by-side comparison showing degraded late-period performance
4. **var_mean_recovery.png** - Variance-to-mean ratio: posterior predictive vs observed
5. **calibration.png** - Prediction interval coverage calibration curve
6. **distribution_overlay.png** - Overall distribution comparison
7. **arviz_ppc.png** - ArviZ posterior predictive check

## Key Findings

**Overall Assessment: FAIL** (1 of 4 criteria passed)

### Falsification Criteria Results

| Criterion | Result | Value |
|-----------|--------|-------|
| Var/Mean in [50, 90] | FAIL | 95% CI [54.8, 130.9] extends beyond target |
| Coverage > 80% | PASS | 100% (over-conservative) |
| Late/Early MAE < 2 | FAIL | Ratio = 4.17 |
| No strong curvature | FAIL | Quadratic coefficient = -5.22 (inverted-U) |

### Major Deficiencies

1. **Systematic curvature**: Inverted-U residual pattern (coef = -5.22) indicates linear growth assumption violated
2. **Late-period failure**: MAE increases 4.2× from early to late period
3. **Overdispersion overestimated**: Var/Mean predicted as 84.5 vs observed 68.7

### Recommendations

1. Add quadratic term: `log(μ) = β₀ + β₁ × year + β₂ × year²`
2. Consider alternative growth functions
3. Re-run posterior predictive checks on improved model

## Visual Summary

The time series plot shows all observations falling within 90% prediction intervals (suggesting good coverage), but residual diagnostics reveal systematic model misspecification. The inverted-U pattern in residuals vs time indicates the model assumes constant exponential growth when the data exhibit accelerating growth.

## Detailed Report

See `ppc_findings.md` for complete analysis, visual evidence citations, and interpretation.

---

**Generated**: 2025-10-29
**Model**: Log-Linear Negative Binomial (β₀=4.355±0.049, β₁=0.863±0.050, φ=13.835±3.449)
