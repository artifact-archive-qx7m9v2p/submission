# EDA Analyst 2 - Output Index

## Overview
This directory contains a comprehensive exploratory data analysis of the x-Y relationship dataset, with focus on **functional form exploration**, **local vs global trends**, and **model class recommendations**.

---

## Key Findings at a Glance

- **Relationship Type**: Non-linear, diminishing returns / asymptotic pattern
- **Best Model**: Asymptotic exponential (Y = 2.57 - 1.02*exp(-0.20*x), R²=0.89)
- **Regime Shift**: Strong positive correlation for x<10 (r=0.94), plateau for x≥10 (r≈0)
- **Transformation**: Log-log achieves near-perfect linearity (r=0.92)
- **Recommended Approach**: Gaussian Process Regression or Non-Linear Least Squares

---

## Main Documents

### 📊 Start Here: Quick Summary
- **`visualizations/00_SUMMARY_comprehensive.png`** - Single-page visual summary of all key findings

### 📝 Written Reports
- **`findings.md`** - Main EDA report with model recommendations and interpretation
- **`eda_log.md`** - Detailed exploration process, intermediate findings, and hypotheses tested

---

## Analysis Scripts (code/)

All scripts are numbered in execution order:

1. **`01_initial_exploration.py`** - Basic statistics, correlations, outliers, data quality
2. **`02_basic_visualizations.py`** - Overview plots, distributions, initial residuals
3. **`03_functional_form_exploration.py`** - Fit 7 functional forms, compare AIC/R²/RMSE
4. **`04_visualize_functional_forms.py`** - Create plots for all candidate models
5. **`05_smoothing_analysis.py`** - LOWESS, moving averages, derivative analysis
6. **`06_segmentation_analysis.py`** - Regime detection, change points, local correlations
7. **`07_transformation_analysis.py`** - Test log, sqrt, power transformations
8. **`08_correlation_structure.py`** - Bootstrap CI, prediction intervals, influence analysis
9. **`09_final_summary_plot.py`** - Comprehensive summary visualization

---

## Visualizations (visualizations/)

### Summary & Overview
- **`00_SUMMARY_comprehensive.png`** ⭐ - Complete analysis summary (START HERE)
- `01_data_overview.png` - Scatter, distributions, Q-Q plots, boxplots
- `02_residual_analysis.png` - Linear model residual diagnostics
- `03_repeated_x_variability.png` - Noise at repeated x values

### Functional Form Analysis
- `04_all_functional_forms.png` - All 7 models compared side-by-side
- `05_top_models_comparison.png` - Top 3 models with residuals highlighted
- `06_residual_comparison.png` - Residual patterns for top models

### Local Trends & Smoothing
- `07_smoothing_methods.png` - LOWESS, moving averages, splines
- `08_lowess_comparison.png` - Different LOWESS bandwidths
- `09_derivative_analysis.png` - Rate of change and curvature

### Segmentation & Regimes
- `10_segmentation_analysis.png` - Quantile, equal-width, piecewise fits

### Transformations
- `11_transformations.png` - Effect of log, sqrt, reciprocal transforms
- `12_transformation_residuals.png` - Residuals after transformation

### Statistical Properties
- `13_correlation_structure.png` - Bootstrap, prediction intervals, Cook's distance

---

## Key Findings by Question

### Q1: What functional form fits best?
**Answer**: Asymptotic exponential (R²=0.89) or cubic polynomial (R²=0.90)
- See: `04_all_functional_forms.png`, `05_top_models_comparison.png`

### Q2: Is the relationship linear?
**Answer**: No - linear R²=0.52, clear residual patterns
- See: `02_residual_analysis.png`, `04_all_functional_forms.png` (Linear panel)

### Q3: Are there regime shifts?
**Answer**: Yes - strong growth for x<10, plateau for x≥10
- See: `10_segmentation_analysis.png`, Section 3 in `findings.md`

### Q4: What transformations help?
**Answer**: Log-log transformation achieves r=0.92 (nearly linear)
- See: `11_transformations.png`, `12_transformation_residuals.png`

### Q5: How strong is predictive power?
**Answer**: Very strong for x<10 (r=0.94), minimal for x≥10 (r≈0)
- See: `13_correlation_structure.png`, Section 3 in `findings.md`

### Q6: What model should I use?
**Answer**: Gaussian Process or Non-Linear Least Squares (asymptotic)
- See: Section 4 in `findings.md`

---

## Model Comparison Table

| Model | R² | RMSE | AIC | Parameters | Recommendation |
|-------|-----|------|-----|------------|----------------|
| Cubic | 0.898 | 0.089 | -122.63 | 4 | Use if max accuracy needed |
| **Asymptotic** | **0.889** | **0.093** | **-122.38** | **3** | **RECOMMENDED** |
| Quadratic | 0.862 | 0.103 | -116.56 | 3 | Good balance |
| Logarithmic | 0.829 | 0.115 | -112.88 | 2 | Simple alternative |
| Power Law | 0.810 | 0.121 | -110.00 | 2 | After log-log transform |
| Square Root | 0.707 | 0.151 | -98.25 | 2 | Moderate fit |
| Linear | 0.518 | 0.193 | -84.87 | 2 | INADEQUATE |

---

## Correlation by X Range

| X Range | n | Correlation | Interpretation |
|---------|---|-------------|----------------|
| x < 10 | 14 | **r = 0.94** | Very strong positive |
| 10 ≤ x < 20 | 10 | r = -0.26 | No relationship |
| x ≥ 20 | 3 | r = -0.78 | Uncertain (small n) |

---

## Data Quality Notes

### Strengths
- No missing values
- 7 repeated x values (good for variance assessment)
- Wide x range (1.0 to 31.5)

### Issues Identified
1. **Outlier**: x=31.5 (Cook's D = 0.81) - highly influential
2. **Heteroscedasticity**: Variance at repeated x ranges from σ=0.019 to 0.157
3. **Sparse high-x region**: Only 3 points for x>20
4. **Non-normality**: Both x and Y fail Shapiro-Wilk test

---

## Recommended Next Steps

### For Immediate Modeling:
1. Fit asymptotic model: Y = a - b*exp(-c*x)
2. Compare to quadratic polynomial as alternative
3. Use cross-validation for model selection
4. Check residual diagnostics

### For Additional Data Collection:
1. More observations at x>20 to confirm plateau
2. Replicates at all x values to characterize variance
3. Dense sampling around x=10 to pinpoint transition

### For Further Analysis:
1. Sensitivity analysis without x=31.5 outlier
2. Weighted regression to handle heteroscedasticity
3. Bootstrap confidence intervals
4. Bayesian approach with informative priors

---

## Files Generated

**Total**: 9 Python scripts, 13 visualizations, 3 documentation files

**Size**: All code is reproducible and well-documented

**Dependencies**: pandas, numpy, matplotlib, seaborn, scipy

---

## Contact

For questions about this analysis, refer to:
- `findings.md` for interpretation and recommendations
- `eda_log.md` for detailed methodology
- Individual visualization files for specific plots

---

**Analysis Completed**: 2025-10-27
**Analyst**: EDA Analyst 2
**Focus**: Functional form exploration and model class recommendations
