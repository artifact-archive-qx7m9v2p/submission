# Exploratory Data Analysis - Time Series Count Data

**Analysis Date:** 2025-10-29
**Dataset:** `/workspace/data/data.csv` (40 observations, 2 variables)

---

## Quick Start

### For Immediate Insights
**Read this first:** `eda_report.md` - Comprehensive report with modeling recommendations

### For Detailed Exploration Process
**Read this second:** `eda_log.md` - Step-by-step analysis log with all intermediate findings

---

## Directory Structure

```
eda/
├── README.md                          # This file - Navigation guide
├── eda_report.md                      # Main report with recommendations
├── eda_log.md                         # Detailed exploration log
├── code/                              # Reproducible analysis scripts
│   ├── 01_initial_exploration.py      # Data loading and descriptive stats
│   ├── 02_distribution_analysis.py    # Overdispersion and distribution tests
│   ├── 03_temporal_analysis.py        # Trend analysis and model comparison
│   ├── 04_advanced_diagnostics.py     # ACF, PACF, changepoint detection
│   └── 05_summary_findings.py         # Executive summary script
└── visualizations/                    # All plots (multi-panel figures)
    ├── 01_distribution_analysis.png   # Distribution properties
    ├── 02_temporal_analysis.png       # Temporal trends and heteroscedasticity
    └── 03_advanced_diagnostics.png    # Time series structure and autocorrelation
```

---

## Key Findings at a Glance

### 5 Critical Findings:

1. **EXTREME OVERDISPERSION**
   - Variance/Mean ratio: 67.99 (Poisson expects ≈1)
   - Index of Dispersion: 2651.69 (far outside 95% CI)
   - **Implication:** Poisson models will fail. Use Negative Binomial.

2. **STRONG EXPONENTIAL GROWTH**
   - 8.45× growth factor over time period
   - 745% percentage increase
   - Exponential model R² = 0.935 (better than linear)
   - **Implication:** Use log-link with polynomial or exponential trend.

3. **MASSIVE AUTOCORRELATION**
   - ACF at lag-1: 0.9886 (near-perfect)
   - Durbin-Watson: 0.195 (severe positive autocorrelation)
   - Lag-1 R²: 0.9773
   - **Implication:** Standard GLM independence violated. Use GEE/robust SE.

4. **SEVERE HETEROSCEDASTICITY**
   - Variance ratio (late/early): 26.19×
   - F-test p-value < 0.0001
   - Mean-variance relationship: Var ∝ Mean²⁺
   - **Implication:** Negative Binomial variance structure appropriate.

5. **POTENTIAL CHANGEPOINT**
   - CUSUM detects regime shift at year ≈ 0.3
   - Mean increases 4.5× (45.67 → 205.12)
   - T-test p < 0.0001
   - **Implication:** Consider piecewise or regime-switching models.

---

## Model Recommendations (Prioritized)

### Top 3 Model Classes:

**1. Negative Binomial GLM with Nonlinear Trend** ⭐⭐⭐
```
C ~ NegativeBinomial(μ, θ)
log(μ) = β₀ + β₁×year + β₂×year²
```
- Addresses overdispersion and nonlinearity
- May need GEE or robust SE for autocorrelation

**2. Negative Binomial GLM with Exponential Trend** ⭐⭐⭐
```
C ~ NegativeBinomial(μ, θ)
log(μ) = β₀ + β₁×year
```
- Natural for exponential growth
- Good interpretability (growth rate = exp(β₁) - 1)

**3. Changepoint Negative Binomial Model** ⭐⭐
```
Two-regime model with structural break at year ≈ 0.3
```
- Tests regime shift hypothesis
- Separate growth parameters before/after break

---

## Visualization Guide

### Figure 1: Distribution Analysis (`01_distribution_analysis.png`)
**6-panel multi-panel figure showing:**
- Histogram with KDE (right-skewed, bimodal)
- Q-Q plot (S-curve, non-normal)
- Box plot with individual points (no outliers)
- Empirical CDF (quartile reference)
- Log-scale histogram (bimodal)
- Count frequency histogram (33 unique values)

**Key insight:** Neither normal nor log-normal; extreme overdispersion confirmed.

### Figure 2: Temporal Analysis (`02_temporal_analysis.png`)
**6-panel multi-panel figure showing:**
- Scatter with linear trend (R² = 0.885)
- Polynomial fits comparison (quadratic best balances fit/complexity)
- Log-scale plot (exponential R² = 0.935)
- Residual plot (U-shaped pattern → nonlinearity)
- Q-Q plot of residuals (approximately normal)
- Variance structure over time (increasing variance)

**Key insight:** Exponential/polynomial trend fits better than linear; heteroscedasticity confirmed.

### Figure 3: Advanced Diagnostics (`03_advanced_diagnostics.png`)
**9-panel multi-panel figure showing:**
- ACF plot (all lags 1-10+ significant)
- PACF plot (lag-1 dominant → AR(1))
- First differences (more stationary)
- Squared residuals over time (heterogeneity)
- Rolling statistics (exponential increase)
- CUSUM (changepoint at year ≈ 0.3)
- Growth rate over time (variable but stable median)
- Mean-variance relationship (quadratic, above Poisson)
- Lag-1 scatter plot (R² = 0.977)

**Key insight:** Time series structure dominates; autocorrelation and changepoint must be modeled.

---

## Mandatory Modeling Requirements

### Must Do:
1. ✓ Use **Negative Binomial** or **Quasi-Poisson** (NOT Poisson)
2. ✓ Address **autocorrelation** (GEE with AR(1), robust SE, or lagged DV)
3. ✓ Include **nonlinear trend** (quadratic, cubic, or exponential)

### Should Do:
4. Test **changepoint** at year ≈ 0.3
5. Validate with **out-of-sample prediction**
6. Check **ACF of residuals** (target: < 0.3)

### Must Avoid:
7. ✗ Standard Poisson GLM (will severely underfit)
8. ✗ Ignoring autocorrelation (standard errors too small)
9. ✗ Linear trend only (residuals show curvature)

---

## Reproducibility

All analyses are fully reproducible. To re-run:

```bash
# Run individual analyses
python /workspace/eda/code/01_initial_exploration.py
python /workspace/eda/code/02_distribution_analysis.py
python /workspace/eda/code/03_temporal_analysis.py
python /workspace/eda/code/04_advanced_diagnostics.py

# Or run summary
python /workspace/eda/code/05_summary_findings.py
```

---

## Technical Details

**Software:** Python 3.x with pandas, numpy, scipy, matplotlib, seaborn

**Statistical Tests:**
- Shapiro-Wilk (normality)
- Kolmogorov-Smirnov (distribution)
- Index of Dispersion (Poisson assumption)
- F-test (variance equality)
- T-test (mean equality)
- Durbin-Watson (autocorrelation)

**Modeling Frameworks Recommended:**
- GLM with Negative Binomial family
- GEE (Generalized Estimating Equations) with AR(1)
- Time series count models (ARIMA-like, state-space)
- Bayesian hierarchical models

---

## Next Steps for Modeling

1. **Baseline:** Fit NB-GLM with linear, quadratic, and exponential trends
2. **Compare:** Use AIC/BIC to select best trend specification
3. **Diagnose:** Check ACF of residuals for remaining autocorrelation
4. **Adjust:** If ACF significant, add GEE-AR(1) or robust SE
5. **Test:** Fit changepoint model and compare to smooth model
6. **Validate:** Out-of-sample prediction on last 5 observations
7. **Finalize:** Choose model balancing fit, parsimony, diagnostics

**Expected Model Parameters:**
- θ (NB dispersion): 5-50
- β₁ (linear trend): ≈81 (raw) or ≈0.85 (log scale)
- β₂ (quadratic term): test if significant
- Changepoint: test year ≈ 0.3

---

## Summary

This is a **complex, high-quality dataset** with:
- ✓ No missing data or outliers
- ✓ Strong, predictable patterns (R² > 0.93)
- ✓ Clear growth trend (exponential/polynomial)
- ⚠ Extreme overdispersion requiring Negative Binomial
- ⚠ Strong autocorrelation requiring special handling
- ⚠ Heteroscedasticity increasing over time

**The data is highly structured and modelable**, but standard Poisson regression is completely inappropriate. Proper handling of overdispersion and autocorrelation is essential for valid inference.

---

**Analyst:** EDA Specialist
**Contact:** See analysis code for methodology details
**Date:** 2025-10-29
