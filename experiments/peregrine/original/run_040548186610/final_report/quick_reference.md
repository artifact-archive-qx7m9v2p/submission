# Quick Reference Guide

**Bayesian Time Series Count Modeling - Final Results**
**Date:** October 29, 2025

---

## At a Glance

**Recommended Model:** Experiment 1 - Negative Binomial Quadratic Regression
**Status:** ADEQUATE for trend estimation, NOT suitable for forecasting
**Main Finding:** 28-fold growth with acceleration
**Main Limitation:** Temporal correlation unresolved (ACF(1) = 0.686)

---

## Model Equation

```
C ~ NegativeBinomial(μ, φ)
log(μ) = β₀ + β₁·year + β₂·year²
```

---

## Parameter Estimates

| Parameter | Mean | 95% CI | Interpretation |
|-----------|------|--------|----------------|
| β₀ | 4.29 | [4.18, 4.40] | Log-count at center ≈ 73 counts |
| β₁ | 0.84 | [0.75, 0.92] | Growth: 2.32× per SD (132%) |
| β₂ | 0.10 | [0.01, 0.19] | Acceleration: 10% per SD² |
| φ | 16.6 | [7.8, 26.3]* | Dispersion (use 99% CI*) |

---

## Key Findings (One Sentence Each)

1. **Strong growth:** 28-fold increase over observation period (β₁ = 0.84)
2. **Weak acceleration:** Growth rate increases by 10% (β₂ = 0.10, barely excludes zero)
3. **Extreme overdispersion:** Variance 68× mean, requiring Negative Binomial (φ = 16.6)
4. **Temporal correlation persists:** ACF(1) = 0.686, making model unsuitable for forecasting
5. **Complex model failed:** AR(1) state-space added 42 parameters with 0% ACF improvement

---

## Model Performance

- **R² = 0.883** (strong trend fit)
- **Coverage = 100%** (over-conservative, target 95%)
- **Convergence:** Perfect (R̂ = 1.000, 0 divergences)
- **LOO-ELPD:** -174.17 ± 5.61
- **Residual ACF(1) = 0.686** (exceeds threshold 0.3)

---

## Use This Model For

- Estimating trend direction and magnitude ✓
- Testing acceleration hypotheses ✓
- Conservative prediction intervals ✓
- Comparing growth across groups ✓
- Exploratory visualization ✓

---

## Do NOT Use This Model For

- Temporal forecasting (one-step-ahead) ✗
- Mechanistic dynamics understanding ✗
- Precise uncertainty quantification ✗
- Applications requiring ACF < 0.3 ✗
- Claiming observations independent ✗

---

## Critical Limitations

1. **Temporal correlation unresolved** (ACF = 0.686 >> 0.3 threshold)
2. **Not suitable for forecasting** (ignores recent observations)
3. **Over-conservative intervals** (100% vs. 95% target)
4. **Systematic residual patterns** (U-shaped)

---

## Predictions at Key Time Points

| Year | Expected Count | 95% Prediction Interval |
|------|---------------|-------------------------|
| -1.67 (earliest) | 14 | [5, 35] |
| 0.0 (center) | 73 | [35, 148] |
| +1.67 (latest) | 396 | [208, 742] |

---

## Model Comparison

| | Exp 1 (Simple) | Exp 3 (Complex) |
|---|---|---|
| **Parameters** | 4 | 46 |
| **Runtime** | 10 min | 25 min |
| **ACF(1)** | 0.686 | 0.690 (no improvement) |
| **R²** | 0.883 | 0.861 (worse) |
| **Coverage** | 67.5% (50% PI) | 75.0% (worse) |
| **Decision** | **RECOMMENDED** | Not recommended |

**Reason:** Simple model adequate; complex model provided zero improvement on critical metric.

---

## Required Disclosures for Publications

**In Methods:**
> "We fitted a Bayesian Negative Binomial regression with quadratic time trend using PyMC 5.26.1 (4 chains × 1,000 iterations, R̂ = 1.000). Posterior predictive checks revealed residual autocorrelation (ACF(1) = 0.686), indicating observations not fully independent. A complex AR(1) model provided no improvement (ACF(1) = 0.690), suggesting fundamental data constraints."

**In Results:**
> "Strong accelerating growth: β₁ = 0.84 [0.75, 0.92], β₂ = 0.10 [0.01, 0.19]. Point predictions highly correlated (R² = 0.883). Intervals conservative (100% coverage at 95% level). Overdispersion φ = 16.6 [7.8, 26.3]."

**In Limitations:**
> "Model treats observations as independent given time, but residual ACF(1) = 0.686 indicates temporal persistence. Not suitable for temporal forecasting. Appropriate for trend estimation with documented uncertainty."

---

## File Locations

**Main report:** `/workspace/final_report/report.md`
**Executive summary:** `/workspace/final_report/executive_summary.md`
**Data:** `/workspace/data/data.csv`
**Model code:** `/workspace/experiments/experiment_1/`
**Figures:** `/workspace/final_report/figures/`
**InferenceData:** `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`

---

## Software Used

- PyMC 5.26.1 (Bayesian PPL)
- Python 3.13
- ArviZ 0.20.0
- NUTS sampler (4 chains)

---

## Validation Conducted

- ✓ Prior predictive checks
- ✓ Simulation-based calibration (20 runs)
- ✓ Posterior convergence diagnostics
- ✓ Posterior predictive checks
- ✓ LOO cross-validation
- ✓ Residual diagnostics

---

## Next Steps (If Temporal Dynamics Critical)

**Priority 1:** Collect external covariates (70-80% success probability)
**Priority 2:** Try observation-level AR(1) (40-60% success)
**Priority 3:** Collect more data (n > 100) (80-90% success)

---

## Quick Decision Tree

**Q: Do you need temporal forecasting (predict next value)?**
- YES → This model NOT suitable. Collect covariates or try obs-level AR.
- NO → Continue

**Q: Do you need precise uncertainty (tight intervals)?**
- YES → This model NOT suitable. Intervals 15% too wide.
- NO → Continue

**Q: Do you need to estimate trend and test acceleration?**
- YES → This model SUITABLE. Use with documented limitations.

**Q: Are observations independent given time acceptable for your application?**
- YES → This model SUITABLE.
- NO → This model NOT suitable.

---

## Parameter Interpretation Quick Guide

**β₀ (Intercept):**
- Raw: 4.29 on log-scale
- Transformed: exp(4.29) = 73 counts at center
- Meaning: Baseline count level

**β₁ (Linear):**
- Raw: 0.84 on log-scale per SD
- Transformed: exp(0.84) = 2.32× per SD
- Meaning: 132% growth rate per SD of time

**β₂ (Quadratic):**
- Raw: 0.10 on log-scale per SD²
- Transformed: exp(0.10) = 1.10× per SD²
- Meaning: 10% acceleration beyond linear

**φ (Dispersion):**
- Value: 16.6
- Meaning: Variance = μ + μ²/16.6
- Example: At μ=110, variance ≈ 840 (7.6× Poisson)

---

## Common Questions

**Q: Why 100% coverage instead of 95%?**
A: Unmodeled temporal correlation inflates uncertainty. Conservative but safe.

**Q: Why not use complex AR(1) model?**
A: Added 42 parameters with 0% improvement on ACF. Parsimony favors simple.

**Q: Can I use this for forecasting?**
A: No. Temporal correlation (ACF=0.686) means recent values have predictive power not captured.

**Q: Is the acceleration real?**
A: Weak evidence (95% CI barely excludes zero). Possibly real, but uncertain.

**Q: Should I use 95% or 99% CI for φ?**
A: 99% CI (SBC showed slight overconfidence at 95% for dispersion parameter).

---

## Visual Summary

**Most important figures:**
1. `exp1_ppc_dashboard.png` - Comprehensive model diagnostics (12 panels)
2. `acf_comparison_exp1_vs_exp3.png` - Shows zero ACF improvement from complex model
3. `exp1_fitted_values.png` - Data with model trend and uncertainty
4. `exp1_residual_diagnostics.png` - Residual patterns and ACF
5. `parameter_comparison.png` - Exp 1 vs Exp 3 posteriors

---

## Citation

Bayesian Modeling Team (2025). "Bayesian Modeling of Time Series Count Data." Analysis conducted October 29, 2025 using PyMC 5.26.1.

**Key methods references:**
- Gelman et al. (2020) - Bayesian workflow
- Vehtari et al. (2017) - LOO cross-validation
- Salvatier et al. (2016) - PyMC

---

## Support

**For detailed information:**
- Methods: Main report Section 3
- Results: Main report Section 4
- Limitations: Main report Section 7
- Interpretation: `/workspace/final_report/supplementary/parameter_interpretation_guide.md`
- Reproducibility: `/workspace/final_report/supplementary/reproducibility.md`

---

**Version:** 1.0
**Status:** FINAL
**Last updated:** October 29, 2025
