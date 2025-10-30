# EDA Analyst 3 - Model Assumptions & Diagnostics

**Focus Area**: Model assumption exploration, residual diagnostics, count model suitability
**Dataset**: `/workspace/data/data_analyst_3.csv` (40 observations, variables: year, C)
**Date**: 2025-10-29

## Quick Summary

This analysis systematically evaluated model assumptions for count data, focusing on:
- Poisson distributional assumptions (equidispersion)
- Heteroscedasticity patterns across transformations
- Residual diagnostics from simple models
- Variance-mean relationships
- Temporal autocorrelation

## Key Findings

### 1. Severe Overdispersion (CRITICAL)
- **Variance/Mean ratio**: 70.43 (should be ~1 for Poisson)
- **Formal test**: χ² = 2746.6, p < 0.001
- **Conclusion**: Standard Poisson regression is **NOT appropriate**
- **Recommendation**: Use Negative Binomial regression

### 2. Strong Temporal Autocorrelation (CRITICAL)
- **Lag-1 correlation**: 0.971 (very high)
- **Durbin-Watson statistic**: 0.47 (indicates strong positive autocorrelation)
- **Implication**: Standard errors will be underestimated without correlated error structure
- **Recommendation**: Use GEE with AR(1) or GLMM with temporal random effects

### 3. Heteroscedasticity in Transformations
- **Linear model**: Homoscedastic (BP p = 0.332) ✓
- **Log transformation**: Heteroscedastic (BP p = 0.003) ✗
- **Sqrt transformation**: Heteroscedastic (BP p = 0.015) ✗
- **Implication**: Use GLM with log link instead of transforming response

### 4. Excellent Model Fit on Log Scale
- **R² (linear)**: 0.881
- **R² (log)**: 0.937 (best)
- **R² (sqrt)**: 0.924
- **Pattern**: Clear exponential growth

### 5. No Data Quality Issues
- **Missing values**: 0
- **Outliers**: None (Cook's D all < 0.1)
- **Data integrity**: All counts are positive integers
- **Verdict**: Data ready for modeling

## Recommended Models (Priority Order)

### 1. Negative Binomial Regression (PRIMARY)
```
log(μ) = β₀ + β₁ × year
var(Y) = μ + α × μ²
```
- Handles overdispersion naturally
- Dispersion parameter α accounts for extra-Poisson variation
- Add AR(1) correlation structure for autocorrelation

### 2. Quasi-Poisson (ALTERNATIVE)
- Simpler than Negative Binomial
- Adjusts standard errors for overdispersion
- Good for prediction, limited for inference

### 3. Gaussian GLM with robust SEs (FALLBACK)
- Flexible but doesn't respect count constraints

### NOT RECOMMENDED
- ✗ Standard Poisson (equidispersion violated)
- ✗ Simple linear regression (predictions can be negative)

## Directory Structure

```
/workspace/eda/analyst_3/
├── README.md (this file)
├── findings.md (comprehensive 16KB report)
├── eda_log.md (detailed exploration log, 12KB)
├── code/
│   ├── 01_initial_exploration.py
│   ├── 02_residual_diagnostics.py
│   ├── 03_visualizations.py
│   ├── 04_count_model_tests.py
│   ├── 05_summary_stats.py
│   └── model_results.npz
└── visualizations/ (all 300 dpi PNG)
    ├── residual_diagnostics_all_models.png (684KB)
    ├── variance_mean_relationship.png (462KB)
    ├── dispersion_analysis.png (415KB)
    ├── transformation_comparison.png (501KB)
    └── influence_diagnostics.png (248KB)
```

## Visualization Guide

### 1. residual_diagnostics_all_models.png
**3×3 panel comparing three models (Linear, Log, Sqrt)**
- Left column: Residuals vs Fitted
- Middle column: Residuals vs Year
- Right column: Q-Q plots

**Key insight**: Linear model has best residual properties; log model shows funnel pattern (heteroscedasticity)

### 2. variance_mean_relationship.png
**2 panels showing variance structure**
- Left: Sliding window variance vs mean with Poisson reference
- Right: Scale-location plot (squared residuals vs fitted)

**Key insight**: Variance far exceeds mean; not Poisson-distributed; quadratic relationship suggests Negative Binomial

### 3. dispersion_analysis.png
**2×2 panel on distribution and dispersion**
- Top-left: Observed histogram
- Top-right: Observed vs Poisson distribution
- Bottom-left: Boxplots by time period (Q1-Q4)
- Bottom-right: Time series with ±2 SD bands

**Key insight**: Distribution much wider than Poisson; variance heterogeneous over time (Levene's test p = 0.010)

### 4. transformation_comparison.png
**2×2 panel comparing transformations**
- Top-left: Original scale with linear fit
- Top-right: Log scale with linear fit
- Bottom-left: Sqrt scale with linear fit
- Bottom-right: Model metrics comparison (R², BP p-value, SW p-value)

**Key insight**: Log transformation gives best fit (R² = 0.937) but worst heteroscedasticity (BP p = 0.003)

### 5. influence_diagnostics.png
**2 panels on influential observations**
- Left: Cook's distance for each observation
- Right: Leverage vs residuals

**Key insight**: No influential outliers (all Cook's D < 0.1); results are robust

## Statistical Tests Summary

| Test | Purpose | Statistic | p-value | Result |
|------|---------|-----------|---------|--------|
| Equidispersion (χ²) | Poisson suitability | 2746.6 | < 0.001 | REJECT Poisson |
| Breusch-Pagan (Linear) | Heteroscedasticity | 0.94 | 0.332 | Homoscedastic ✓ |
| Breusch-Pagan (Log) | Heteroscedasticity | 8.91 | 0.003 | Heteroscedastic ✗ |
| Breusch-Pagan (Sqrt) | Heteroscedasticity | 5.94 | 0.015 | Heteroscedastic ✗ |
| Shapiro-Wilk (Linear) | Normality | 0.955 | 0.112 | Normal ✓ |
| Shapiro-Wilk (Log) | Normality | 0.983 | 0.796 | Normal ✓ |
| Shapiro-Wilk (Sqrt) | Normality | 0.969 | 0.325 | Normal ✓ |
| Levene's Test | Variance equality | 4.39 | 0.010 | Heterogeneous ✗ |

## Reproducibility

All analyses are fully reproducible. To regenerate:

```bash
cd /workspace/eda/analyst_3/code
python 01_initial_exploration.py
python 02_residual_diagnostics.py
python 03_visualizations.py
python 04_count_model_tests.py
python 05_summary_stats.py
```

## Next Steps

1. **Fit Negative Binomial model** with log(μ) = β₀ + β₁×year
2. **Estimate dispersion parameter** α and test significance
3. **Add AR(1) correlation** using GEE or GLMM
4. **Validate with randomized quantile residuals**
5. **Compare models** using AIC/BIC
6. **Generate prediction intervals** accounting for overdispersion

## Contact

**Analyst**: EDA Analyst 3
**Specialization**: Model diagnostics, residual analysis, count model assumptions
**Output location**: `/workspace/eda/analyst_3/`
