# Analyst 3: Feature Engineering & Transformation Analysis

## Overview

This directory contains a comprehensive analysis of transformations and feature engineering approaches for the time-series count data. The focus is on identifying optimal data representations for modeling and understanding the underlying data-generating process.

## Executive Summary

**Key Finding**: Log transformation is optimal across multiple criteria (linearity, variance stabilization, normality). The data exhibit exponential/polynomial growth with severe variance heterogeneity (34x increase), strongly suggesting a **GLM with log link (Poisson or Negative Binomial) and quadratic predictor structure**.

## Directory Structure

```
/workspace/eda/analyst_3/
├── README.md                          # This file
├── findings.md                        # Main findings and recommendations (562 lines)
├── eda_log.md                        # Detailed exploration log (381 lines)
├── code/                             # Reproducible analysis scripts
│   ├── 00_run_all_analyses.py       # Master script to reproduce all analyses
│   ├── 01_initial_exploration.py    # Data quality and initial patterns
│   ├── 02_transformation_analysis.py # Transformation performance comparison
│   ├── 02b_polynomial_analysis.py   # Polynomial vs exponential models
│   ├── 03_visualization_transformations.py  # Core plots
│   └── 04_advanced_visualizations.py        # Advanced diagnostics
└── visualizations/                   # 10 multi-panel and single plots
    ├── 01_transformation_comparison.png      # 6-panel: All transformations
    ├── 02_residual_diagnostics.png          # 9-panel: Residual diagnostics
    ├── 03_variance_stabilization.png        # 4-panel: Variance assessment
    ├── 04_polynomial_vs_exponential.png     # 4-panel: Model comparison
    ├── 05_all_models_comparison.png         # Single: Overlaid fits
    ├── 06_boxcox_optimization.png           # 3-panel: Box-Cox parameter sweep
    ├── 07_feature_correlation_matrix.png    # Single: Feature correlations
    ├── 08_model_selection_criteria.png      # 2-panel: AIC/BIC comparison
    ├── 09_model_fits_with_intervals.png     # 4-panel: Fits with uncertainty
    └── 10_scale_location_plots.png          # 4-panel: Homoscedasticity check
```

## Quick Start

To reproduce all analyses:

```bash
cd /workspace/eda/analyst_3
python code/00_run_all_analyses.py
```

To view main findings:
```bash
cat /workspace/eda/analyst_3/findings.md
```

## Key Results

### 1. Transformation Performance

| Transformation | Linearity (r) | Variance Ratio | Normality (p) | Rank |
|----------------|---------------|----------------|---------------|------|
| **Log**        | **0.9672**    | **0.58**       | **0.9446**    | **1** |
| Box-Cox (λ=-0.04) | 0.9667     | 0.50           | 0.9712        | 2    |
| Square Root    | 0.9618        | 4.49           | 0.1334        | 3    |
| Original       | 0.9405        | **34.72**      | 0.0712        | 4    |

**Conclusion**: Log transformation wins across criteria; Box-Cox confirms λ≈0 optimal.

### 2. Functional Form

| Model       | R²    | AIC    | Residual Quality | Recommendation |
|-------------|-------|--------|------------------|----------------|
| Quadratic   | 0.961 | 231.78 | Poor (heteroscedastic) | For fit quality |
| Exponential | 0.929 | 253.99 | **Excellent** (normal, homoscedastic) | **For inference** |

**Conclusion**: Both viable; exponential preferred for robust inference, quadratic for pure predictive accuracy.

### 3. Model Recommendation

**PRIMARY**: GLM with log link
```
C ~ Poisson(μ) or NegativeBinomial(μ, θ)
log(μ) = β₀ + β₁×year + β₂×year²
```

**Rationale**:
- Respects count data structure
- Log link matches optimal transformation
- Built-in variance modeling
- Captures polynomial growth
- Valid statistical inference

**ALTERNATIVE**: Log-linear OLS (if simplicity needed)
```
log(C) = β₀ + β₁×year + β₂×year² + ε
```

## Visualization Guide

### Core Transformations
- **`01_transformation_comparison.png`**: Compare 6 transformations side-by-side
  - Shows log and Box-Cox achieve best linearity
  - R² annotated for each transformation

- **`02_residual_diagnostics.png`**: Residual patterns for Original, Log, Box-Cox
  - Q-Q plots show log produces normal residuals
  - Shapiro-Wilk p-values confirm normality

- **`03_variance_stabilization.png`**: Absolute residuals vs fitted values
  - Log reduces variance ratio from 34.7 to 0.58
  - Trend lines show homoscedasticity achieved

### Model Comparison
- **`04_polynomial_vs_exponential.png`**: Four model classes individually
  - Linear, Quadratic, Cubic, Exponential
  - R² progression: 0.88 → 0.96 → 0.98 → 0.93

- **`05_all_models_comparison.png`**: All models overlaid
  - Models converge in observed range
  - Divergence at extrapolation boundaries

- **`08_model_selection_criteria.png`**: AIC/BIC trends
  - Quadratic optimal by AIC (231.78)
  - Diminishing returns after degree 3

### Advanced Diagnostics
- **`06_boxcox_optimization.png`**: Parameter sweep λ ∈ [-2, 2]
  - Three criteria (linearity, variance, normality) all favor λ≈0
  - Robust evidence for log transformation

- **`07_feature_correlation_matrix.png`**: Derived feature correlations
  - exp(0.5×year) achieves highest correlation (0.973)
  - Suggests dampened exponential growth

- **`09_model_fits_with_intervals.png`**: Prediction uncertainty
  - ±2 SE bands grow with predictions
  - Illustrates heteroscedasticity on original scale

- **`10_scale_location_plots.png`**: Scale-location diagnostics
  - Flat trend for log = homoscedastic
  - Increasing trend for original = heteroscedastic

## Main Findings Summary

### Robust Conclusions (High Confidence)
1. ✅ Log transformation is optimal across multiple criteria
2. ✅ Severe variance heterogeneity on original scale (34x ratio)
3. ✅ Growth pattern exceeds simple linear (quadratic/exponential)
4. ✅ Count data characteristics strongly evident
5. ✅ Quadratic term significantly improves fit (ΔAIC = -41)

### Model Recommendations
1. **Tier 1**: Poisson or Negative Binomial GLM with log link + quadratic terms
2. **Tier 2**: Log-linear OLS (simpler, good residuals, requires back-transformation)
3. **Tier 3**: Quadratic OLS with robust SE (best raw fit, challenging inference)

### Not Recommended
- ❌ Simple linear model on original scale (heteroscedastic, poor fit)
- ❌ High-degree polynomials (overfitting risk with n=40)
- ❌ Inverse or square transformations (poor linearity)
- ❌ Power law specification (poor fit, R²=0.70)

## Key Insights for Modeling

### Data Structure
- **Nature**: Count data with exponential/polynomial growth
- **Variance pattern**: Increases with mean (Poisson-like)
- **Growth rate**: 2.34x per standardized year unit
- **Acceleration**: Positive quadratic term indicates increasing growth rate

### Transformation Impact
- **Original scale**: Variance ratio = 34.7 (severe heteroscedasticity)
- **Log scale**: Variance ratio = 0.58 (near-homoscedastic)
- **Residuals**: Shapiro-Wilk p = 0.94 on log scale (excellent normality)
- **Linearity**: r = 0.967 on log scale vs 0.941 on original

### Model Selection Criteria
- **By AIC**: Quadratic wins (231.78 vs 253.99 exponential)
- **By residual diagnostics**: Exponential wins (normal, homoscedastic)
- **By interpretability**: Exponential wins (growth rate parameter)
- **By flexibility**: GLM wins (respects count structure)

### Feature Engineering
- **Essential**: year, year²
- **Promising**: exp(0.5×year) [highest correlation: 0.973]
- **Avoid**: year³⁺ (overfitting), inverse terms (poor linearity)

## Implementation Notes

### Starting Point
```python
import statsmodels.api as sm
import pandas as pd

# Load data
df = pd.read_csv('/workspace/data/data_analyst_3.csv')

# Create features
X = sm.add_constant(pd.DataFrame({
    'year': df['year'],
    'year2': df['year']**2
}))

# Fit Poisson GLM
model = sm.GLM(df['C'], X, family=sm.families.Poisson()).fit()
print(model.summary())

# Check overdispersion
overdispersion = model.deviance / model.df_resid
print(f"Overdispersion parameter: {overdispersion:.2f}")

# If > 1.5, use Negative Binomial
if overdispersion > 1.5:
    model_nb = sm.GLM(df['C'], X, family=sm.families.NegativeBinomial()).fit()
    print(model_nb.summary())
```

### Validation Checklist
- [ ] Check deviance residuals for patterns
- [ ] Assess overdispersion parameter
- [ ] Compare AIC across Poisson, NegBin, Quasi-Poisson
- [ ] Validate predictions on holdout set
- [ ] Check for influential observations (n=40 is small)
- [ ] Test sensitivity to polynomial degree (2 vs 3)

## Competing Hypotheses Tested

### Hypothesis 1: Exponential Growth ✓ Supported
- **Evidence FOR**: Log-linear R² = 0.935, growth rate = 2.34x/year
- **Evidence AGAINST**: Quadratic fits better (ΔAIC = -22)
- **Verdict**: Strong approximate model, slight acceleration present

### Hypothesis 2: Polynomial Growth (Quadratic) ✓ Strongly Supported
- **Evidence FOR**: R² = 0.961, AIC = 231.78, significant improvement over linear
- **Evidence AGAINST**: Heteroscedastic residuals
- **Verdict**: Best fit on original scale, captures acceleration

### Hypothesis 3: Count Data Process ✓ Strongly Supported
- **Evidence FOR**: Variance-mean relationship, log-link effectiveness, integer outcomes
- **Evidence AGAINST**: None substantial
- **Verdict**: GLM framework appropriate

### Hypothesis 4: Power Law ✗ Rejected
- **Evidence FOR**: None
- **Evidence AGAINST**: R² = 0.70 (poor), log-log plot not linear
- **Verdict**: Not appropriate for this data

## Limitations and Caveats

1. **Sample size**: n=40 limits complexity of models
2. **Temporal structure**: Potential autocorrelation not assessed
3. **Extrapolation**: Polynomial models unstable outside data range
4. **Back-transformation**: Log-scale models require bias correction for predictions
5. **Model uncertainty**: Both quadratic and exponential fit well; choice affects extrapolation

## Next Steps for Modeling

1. **Immediate**: Fit Poisson GLM with quadratic terms, check overdispersion
2. **If overdispersed**: Upgrade to Negative Binomial GLM
3. **Validation**: Cross-validation for degree selection (2 vs 3)
4. **Diagnostics**: Check deviance residuals, influential points
5. **Comparison**: Benchmark against log-linear OLS for simplicity
6. **Temporal**: Test for autocorrelation if treating as time series

## Contact & Questions

This analysis was conducted by **Analyst 3** focusing on feature engineering and transformations. For questions about:
- Transformation methodology → see `eda_log.md`
- Specific recommendations → see `findings.md`
- Reproducing results → run `code/00_run_all_analyses.py`

---

**Last updated**: 2025-10-29
**Dataset**: `/workspace/data/data_analyst_3.csv` (40 observations)
**Key recommendation**: GLM with log link (Poisson/NegBin) + quadratic terms
