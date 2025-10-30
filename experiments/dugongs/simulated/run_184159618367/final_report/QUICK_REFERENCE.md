# Quick Reference Guide
## Log-Log Power Law Model for Y-x Relationship

**One-Page Cheat Sheet for Practitioners**

---

## Model Equation

### Power Law Form (Use This)
```
Y = 1.773 × x^0.126
```

### Mathematical Form
```
log(Y) = 0.572 + 0.126 × log(x)
```

---

## Parameter Estimates

| Parameter | Value | 95% Credible Interval | Meaning |
|-----------|-------|------------------------|---------|
| **Exponent (β)** | **0.126** | **[0.106, 0.148]** | **Elasticity** |
| Scaling (exp(α)) | 1.773 | [1.694, 1.859] | Y when x=1 |
| Log-scale SD (σ) | 0.055 | [0.041, 0.070] | Residual noise |

---

## Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R²** | **0.81** | **Explains 81% of variance** |
| RMSE | 0.12 | Typical error: 5% of Y range |
| MAE | 0.10 | Average absolute error |
| Coverage (95%) | 100% | All data within prediction bands |
| ELPD | 38.85 ± 3.29 | Out-of-sample prediction quality |
| Pareto k (max) | 0.399 | No influential outliers |

---

## How to Use

### 1. Point Prediction

**For a new x value** (within 1-32):
```
Y_predicted = 1.773 × x^0.126
```

**Example**: x = 15
```
Y = 1.773 × 15^0.126 = 2.50
```

### 2. Prediction Interval (95%)

**Method**: Use posterior predictive samples from InferenceData

**Typical Width**: ±0.2 to ±0.3 units

**Example**: x = 15
- Point: 2.50
- 95% Interval: [2.23, 2.77] (approximately)

### 3. Elasticity Calculation

**Question**: How much does Y change when x changes by P%?

**Answer**: Y changes by approximately 0.126 × P%

**Example**: x increases by 10%
- Y increases by 0.126 × 10% = 1.26%

---

## Validated Range

| Aspect | Range | Status |
|--------|-------|--------|
| **Reliable** | **x ∈ [1.0, 31.5]** | **✓ Full validation** |
| Adequate data | x ∈ [1.0, 20] | 24/27 observations |
| Sparse data | x ∈ (20, 32] | Only 3 observations |
| **Caution** | **x < 1 or x > 35** | **⚠ Extrapolation** |

---

## Quick Interpretation

### What β = 0.126 Means

**Sublinear Growth** (β < 1):
- Y increases with x, but at a decreasing rate
- Strong diminishing returns
- Growth rate decreases by 86% from x=1 to x=30

**Examples**:
| x Change | Y Change | Percentage |
|----------|----------|------------|
| 1 → 2 (×2) | 1.77 → 1.93 | +9.1% |
| 5 → 10 (×2) | 2.18 → 2.36 | +9.1% |
| 10 → 20 (×2) | 2.36 → 2.58 | +9.1% |

**Pattern**: Same percentage increase for same proportional change in x

---

## Dos and Don'ts

### DO Use For:
- ✓ Predictions for x ∈ [1, 32]
- ✓ Understanding diminishing returns
- ✓ Quantifying elasticity
- ✓ Scientific inference about power law
- ✓ Comparing to theoretical expectations

### DON'T Use For:
- ✗ Predictions for x < 0.5 or x > 35
- ✗ Causal inference (descriptive only)
- ✗ Exact 90% intervals (under-calibrated)
- ✗ Mechanistic understanding (what causes this?)
- ✗ Time series prediction (no temporal model)

---

## Model Comparison Results

| Model | ELPD | R² | RMSE | Verdict |
|-------|------|-----|------|---------|
| **Power Law (Exp3)** | **38.85** | **0.81** | 0.12 | **WINNER** ✓ |
| Exponential (Exp1) | 22.19 | 0.89 | 0.09 | Overfits |

**Winner Reason**: 75% better out-of-sample prediction (ELPD)

---

## Uncertainty Levels

| Confidence Level | Use This Interval | Status |
|------------------|-------------------|--------|
| 50% | ✗ Don't use | Under-calibrated (41% actual) |
| 80% | ~ Use with caution | Acceptable (82% actual) |
| **95%** | **✓ Recommended** | **Perfect (100% actual)** |

**Recommendation**: Always use 95% prediction intervals

---

## Common Use Cases

### Case 1: Predict Y for x = 7.5
```
Point: Y = 1.773 × 7.5^0.126 = 2.28
95% PI: [2.05, 2.51]
```

### Case 2: What if x increases from 10 to 15?
```
Before: Y = 1.773 × 10^0.126 = 2.36
After:  Y = 1.773 × 15^0.126 = 2.50
Change: +0.14 units (+5.9%)
```

### Case 3: How much x needed to reach Y = 2.40?
```
Solve: 2.40 = 1.773 × x^0.126
x = (2.40 / 1.773)^(1/0.126) ≈ 13.6
```

---

## Files and Code

### Load Model Results
```python
import arviz as az
idata = az.from_netcdf('/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf')
```

### Make Predictions
```python
import numpy as np

# New x value
x_new = 15.0

# Point prediction
y_pred = 1.773 * x_new**0.126

# With uncertainty (from posterior)
alpha = idata.posterior['alpha'].values.flatten()
beta = idata.posterior['beta'].values.flatten()
sigma = idata.posterior['sigma'].values.flatten()

# Sample predictions
log_y_samples = alpha + beta * np.log(x_new) + np.random.normal(0, sigma, size=len(alpha))
y_samples = np.exp(log_y_samples)

# 95% credible interval
y_lower = np.percentile(y_samples, 2.5)
y_upper = np.percentile(y_samples, 97.5)
```

### Key Files
- **InferenceData**: `/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf`
- **Full Report**: `/workspace/final_report/report.md`
- **Executive Summary**: `/workspace/final_report/EXECUTIVE_SUMMARY.md`
- **Technical Details**: `/workspace/final_report/supplementary/technical_details.md`

---

## Diagnostic Checklist

Before using model, verify:
- [x] R-hat ≤ 1.01 for all parameters (max: 1.010) ✓
- [x] ESS > 400 for all parameters (min: 1383) ✓
- [x] Zero divergences (0/4000) ✓
- [x] 95% coverage ≥ 90% (100%) ✓
- [x] All Pareto k < 0.7 (max: 0.399) ✓
- [x] Residuals normal (Shapiro p = 0.94) ✓

**Status**: All checks passed ✓

---

## Contact and Documentation

**Questions About**:
- Predictions: Section 10.2 of full report
- Limitations: Section 9 of full report
- Implementation: Technical supplement
- Validation: Model files in `/workspace/experiments/experiment_3/`

**Visual Reference**:
- Model fit: `/workspace/final_report/figures/main_model_fit.png`
- Diagnostics: `/workspace/final_report/figures/convergence_diagnostics.png`
- Comparison: `/workspace/final_report/figures/model_comparison_loo.png`

---

## Model Status

**Adequacy**: ADEQUATE ✓
**Convergence**: EXCELLENT ✓
**Validation**: PASSED ✓
**Recommended**: YES ✓

**Use with confidence for x ∈ [1, 32]**

---

**Last Updated**: October 27, 2025
**Version**: 1.0 (Final)
