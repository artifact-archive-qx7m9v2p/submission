# Exploratory Data Analysis: Y vs x Relationship

**Dataset**: `/workspace/data/data.csv` (N=27, 2 variables)
**Analysis Date**: 2025-10-28
**Goal**: Understand relationship to inform Bayesian model building

---

## Quick Start

### For the impatient:
1. **Main finding**: Strong positive non-linear relationship, logarithmic model recommended
2. **View**: `visualizations/eda_summary_simple.png` for one-page summary
3. **Read**: First 2 pages of `eda_report.md` for executive summary

### For thorough review:
1. **Read**: `eda_report.md` - Comprehensive 14-section report with all findings
2. **View**: `visualizations/eda_summary_comprehensive.png` - Multi-panel summary
3. **Browse**: All 13 visualizations in `visualizations/` directory
4. **Deep dive**: `eda_log.md` - Detailed exploration process (7 rounds)

---

## Directory Structure

```
/workspace/eda/
├── README.md                           # This file
├── eda_report.md                       # Main comprehensive report
├── eda_log.md                          # Detailed exploration log
├── data_quality_summary.json           # Quantitative statistics
├── code/                               # Reproducible analysis scripts
│   ├── 01_data_loading_and_quality.py
│   ├── 02_univariate_analysis.py
│   ├── 03_bivariate_analysis.py
│   ├── 04_hypothesis_testing.py
│   ├── 05_additional_insights.py
│   └── 06_summary_visualization.py
└── visualizations/                     # All plots (300 dpi)
    ├── eda_summary_comprehensive.png   # ⭐ Multi-panel summary
    ├── eda_summary_simple.png          # ⭐ One-panel summary
    ├── univariate_x_distribution.png
    ├── univariate_y_distribution.png
    ├── univariate_combined_distributions.png
    ├── bivariate_scatter_various_fits.png
    ├── bivariate_residual_analysis.png
    ├── hypothesis_all_models_comparison.png
    ├── hypothesis_residuals_comparison.png
    ├── variance_structure_analysis.png
    ├── replicate_analysis.png
    ├── transformation_analysis.png
    └── rate_of_change_analysis.png
```

---

## Key Findings Summary

### Data Quality
- **Excellent**: No missing values, no errors
- N=27 observations, 20 unique x values
- 6 x-values have replicates (multiple Y observations)
- One influential point at x=31.5 (requires sensitivity check)

### Relationship
- **Strong positive monotonic** (Spearman ρ = 0.78, p < 0.001)
- **Non-linear**: Logarithmic or quadratic form
- **Saturation pattern**: Diminishing returns at high x
- Linear model inadequate (R²=0.52 vs 0.83 for log)

### Variance Structure
- **Approximately homoscedastic** (Breusch-Pagan p=0.55)
- Trend toward decreasing variance at high x (ratio 4.6:1)
- Residuals normally distributed (important for inference)

### Model Recommendation
**Primary**: Logarithmic model
```
Y ~ Normal(μ, σ)
μ = α + β·log(x)
R² = 0.83, RMSE = 0.11

Recommended priors:
α ~ Normal(1.75, 0.5)
β ~ Normal(0.27, 0.15)
σ ~ HalfNormal(0.2)
```

**Alternative**: Quadratic model (R²=0.86, but risk of overfitting)

---

## Main Documents

### eda_report.md (Comprehensive Report)
**14 sections, ~40 pages**

1. Executive Summary
2. Data Quality Assessment
3. Univariate Analysis
4. Bivariate Relationship Analysis
5. Functional Form Investigation ⭐
6. Variance Structure Analysis
7. Saturation and Plateau Analysis
8. Data Coverage and Gaps
9. Influential Points and Sensitivity
10. Modeling Recommendations ⭐⭐
11. Summary of Key Findings
12. Recommended Next Steps
13. Visualization Index
14. Conclusion

**Use this for**: Complete understanding, modeling decisions, manuscript preparation

### eda_log.md (Detailed Exploration Process)
**7 rounds of analysis**

Round 1: Initial Data Quality Assessment
Round 2: Univariate Analysis
Round 3: Bivariate Analysis
Round 4: Competing Hypotheses Testing ⭐
Round 5: Deep Dive - Variance Structure
Round 6: Transformation Analysis
Round 7: Saturation Analysis

**Use this for**: Understanding analysis decisions, replication, audit trail

---

## Visualization Guide

### Summary Visualizations (Start Here!)

**eda_summary_comprehensive.png** - 6-panel comprehensive summary
- Main scatter with recommended model + CI band
- Residual plot
- Model comparison (R² values)
- Y distribution
- Variance by region
- Key statistics and recommendations

**eda_summary_simple.png** - Single-panel presentation-ready plot
- Clean scatter plot with logarithmic fit
- Confidence band
- Suitable for talks/presentations

### Detailed Visualizations

**Univariate (3 files)**
- `univariate_x_distribution.png` - 4-panel analysis of x
- `univariate_y_distribution.png` - 4-panel analysis of Y
- `univariate_combined_distributions.png` - Side-by-side comparison

**Bivariate (2 files)**
- `bivariate_scatter_various_fits.png` - 4 functional forms compared
- `bivariate_residual_analysis.png` - Residual diagnostics (4 panels)

**Model Comparison (2 files)**
- `hypothesis_all_models_comparison.png` - All 5 models + overlay (6 panels)
- `hypothesis_residuals_comparison.png` - Residuals for each model (6 panels)

**Variance Structure (2 files)**
- `variance_structure_analysis.png` - Heteroscedasticity check
- `replicate_analysis.png` - Replicate locations and variability

**Additional (2 files)**
- `transformation_analysis.png` - Y vs log(Y) comparison
- `rate_of_change_analysis.png` - dY/dx pattern

---

## Code Scripts

All scripts are self-contained and can be run independently. They are numbered in logical order.

### 01_data_loading_and_quality.py
- Loads data and validates structure
- Descriptive statistics for both variables
- Missing value and duplicate checks
- Outlier detection (IQR method)
- X-variable spacing analysis
- Outputs: Console report + `data_quality_summary.json`

### 02_univariate_analysis.py
- Distribution analysis for x and Y
- Histograms, KDE, boxplots, Q-Q plots
- Normality tests (Shapiro-Wilk, KS)
- Outputs: 3 visualization files

### 03_bivariate_analysis.py
- Scatter plots with various fits
- Correlation analysis (Pearson, Spearman, Kendall)
- Residual analysis for linear model
- Heteroscedasticity testing (Breusch-Pagan)
- Influential points (Cook's distance)
- Outputs: 2 visualization files

### 04_hypothesis_testing.py
- Tests 5 competing functional forms
  1. Linear: Y = a + bx
  2. Logarithmic: Y = a + b·ln(x)
  3. Asymptotic: Y = a + b/x
  4. Quadratic: Y = a + bx + cx²
  5. Square root: Y = a + b·√x
- Model comparison (R², RMSE, MAE)
- Residual comparison across models
- Outputs: 2 visualization files

### 05_additional_insights.py
- Variance structure analysis (rolling window)
- Replicate analysis (multiple Y at same x)
- Data coverage and gap analysis
- Log transformation testing
- Saturation/plateau analysis
- Rate of change computation
- Outputs: 4 visualization files

### 06_summary_visualization.py
- Creates comprehensive 6-panel summary
- Creates simple 1-panel presentation plot
- Outputs: 2 summary visualization files

---

## Reproducibility

### To reproduce entire analysis:
```bash
cd /workspace/eda/code
python 01_data_loading_and_quality.py
python 02_univariate_analysis.py
python 03_bivariate_analysis.py
python 04_hypothesis_testing.py
python 05_additional_insights.py
python 06_summary_visualization.py
```

### Requirements:
- Python 3.x
- pandas, numpy, scipy
- matplotlib, seaborn
- pathlib (standard library)

All visualizations use 300 dpi for publication quality.

---

## Model Comparison Results

| Model | Equation | R² | RMSE | Parameters | Recommendation |
|-------|----------|---------|--------|------------|----------------|
| Linear | Y = 2.04 + 0.026x | 0.518 | 0.193 | 2 | ❌ Poor fit |
| Square Root | Y = 1.72 + 0.19√x | 0.707 | 0.151 | 2 | ⚠️ Moderate |
| Asymptotic | Y = 2.52 - 0.99/x | 0.755 | 0.138 | 2 | ⚠️ Good |
| **Logarithmic** | **Y = 1.75 + 0.27·ln(x)** | **0.829** | **0.115** | **2** | **✅ Recommended** |
| Quadratic | Y = 1.75 + 0.09x - 0.002x² | 0.862 | 0.103 | 3 | ⚠️ Best fit, may overfit |

**Winner**: Logarithmic model (parsimonious, theoretically sound, strong fit)

---

## Bayesian Modeling Checklist

### Before Fitting
- [ ] Decide on primary model (logarithmic recommended)
- [ ] Choose variance structure (constant vs heteroscedastic)
- [ ] Set up weakly informative priors (values provided in report)
- [ ] Run prior predictive checks

### During Fitting
- [ ] Fit primary model (logarithmic)
- [ ] Fit alternative model (quadratic) for comparison
- [ ] Check convergence diagnostics (Rhat, ESS)
- [ ] Examine posterior distributions

### After Fitting
- [ ] Posterior predictive checks (marginal and conditional)
- [ ] Model comparison (LOO-CV or WAIC)
- [ ] Sensitivity analysis: refit without x=31.5
- [ ] Check predictions in gap region (x ∈ [23, 29])

### Red Flags to Watch
- [ ] Poor fit in gap region (may need more flexible model)
- [ ] Extreme sensitivity to x=31.5 point
- [ ] Heteroscedasticity not captured
- [ ] Posterior predictive checks fail

---

## Data Quality Issues to Address

### Critical
- None! Data quality is excellent.

### For Consideration
1. **Influential point at x=31.5**: Sensitivity analysis required
2. **Data gap at x ∈ [23, 29]**: Creates prediction uncertainty
3. **High replicate variability at x=15.5**: Investigate if possible

### If Collecting More Data
**Priority 1**: High x region (x > 20) - confirm plateau
**Priority 2**: Gap region (x ∈ [23, 29]) - reduce uncertainty
**Priority 3**: More replicates - better estimate variance structure

---

## Questions Answered

✅ **Is there a relationship?** YES - strong positive (ρ=0.78)
✅ **Is it linear?** NO - logarithmic or quadratic
✅ **Is there saturation?** YES - diminishing returns pattern
✅ **Is variance constant?** APPROXIMATELY - slight decrease at high x
✅ **Are there outliers?** NO major outliers, one influential point
✅ **What model to use?** Logarithmic recommended
✅ **Good data quality?** EXCELLENT - no missing, no errors

---

## Questions Remaining

❓ **Exact plateau level?** Need more high-x data
❓ **True asymptote or slow growth?** Logarithmic → slow growth, Asymptotic → true plateau
❓ **Heteroscedasticity real?** Trend visible but not significant (n=27 may be too small)
❓ **Behavior beyond x=31.5?** No data, extrapolation risky
❓ **Why high variability at x=15.5?** Unknown - measurement issue?

---

## Contact and Citation

**Analysis performed by**: EDA Specialist Agent
**Date**: 2025-10-28
**Framework**: Systematic EDA with competing hypotheses testing

**To cite this analysis**:
```
Exploratory Data Analysis: Y vs x Relationship
Date: 2025-10-28
Dataset: data.csv (N=27)
Analysis: Systematic EDA with 5 functional forms tested
Recommendation: Logarithmic model (R²=0.829)
```

---

## Acknowledgments

Analysis followed systematic EDA principles:
- Iterative exploration (7 rounds)
- Multiple hypothesis testing (5 functional forms)
- Skeptical approach (sensitivity checks recommended)
- Focus on practical significance
- Documentation of tentative vs robust findings

All visualizations created at 300 dpi for publication quality.
All code is reproducible and well-documented.

---

**For questions or clarifications, see detailed reports or re-run analysis scripts.**

**END OF README**
