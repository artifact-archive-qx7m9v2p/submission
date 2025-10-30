# Exploratory Data Analysis: Time Series Count Data

**Analysis Date:** 2025-10-29
**Dataset:** `/workspace/data/data.csv` (40 observations)
**Status:** ✅ Complete

---

## Quick Summary

This directory contains a comprehensive exploratory data analysis of time series count data showing strong non-linear growth with extreme overdispersion.

### Key Findings
- **Strong non-linear trend:** Accelerating growth (quadratic R² = 0.961)
- **Extreme overdispersion:** Variance-to-mean ratio = 68.0
- **High temporal correlation:** Lag-1 autocorrelation = 0.989
- **Clean data:** No missing values or problematic outliers

### Recommended Model
**Negative Binomial GLM with Quadratic Trend**
```
C ~ NegBinomial(μ, φ)
log(μ) = β₀ + β₁·year + β₂·year²
```

---

## Directory Structure

```
/workspace/eda/
├── README.md                           # This file
├── eda_report.md                       # Comprehensive findings report (main deliverable)
├── eda_log.md                          # Detailed exploration process log
├── code/
│   ├── 01_initial_exploration.py       # Data loading and descriptive stats
│   ├── 02_visualization_analysis.py    # All visualizations
│   └── 03_hypothesis_testing.py        # Statistical tests and model comparison
└── visualizations/                     # All plots (300 DPI PNG)
    ├── timeseries_plot.png
    ├── count_distribution.png
    ├── scatter_with_smoothing.png
    ├── residual_diagnostics.png
    ├── variance_analysis.png
    ├── boxplot_by_period.png
    ├── autocorrelation_plot.png
    └── log_transformation_analysis.png
```

---

## How to Use This Analysis

### 1. For a Quick Overview
- Read the **Executive Summary** in `eda_report.md`
- View key visualizations:
  - `timeseries_plot.png` - Overall pattern
  - `scatter_with_smoothing.png` - Non-linearity evidence
  - `variance_analysis.png` - Overdispersion evidence

### 2. For Modeling Decisions
- See **Section 9: Modeling Recommendations** in `eda_report.md`
- Focus on model comparison table (Section 3.2)
- Review challenges (Section 8)

### 3. For Detailed Understanding
- Read `eda_log.md` for step-by-step exploration process
- Review all hypothesis tests in Section 5 of `eda_report.md`
- Examine residual diagnostics (Section 4)

### 4. To Reproduce Analysis
- Run scripts in `code/` directory in order:
  ```bash
  python /workspace/eda/code/01_initial_exploration.py
  python /workspace/eda/code/02_visualization_analysis.py
  python /workspace/eda/code/03_hypothesis_testing.py
  ```

---

## Key Visualizations

### Primary Plots

1. **`timeseries_plot.png`** - Shows clear accelerating growth pattern over time

2. **`scatter_with_smoothing.png`** - Compares linear, polynomial, and exponential fits
   - Linear R² = 0.885 (inadequate)
   - Exponential R² = 0.929 (good)
   - Quadratic R² = 0.961 (best)

3. **`variance_analysis.png`** - Two-panel plot showing:
   - Left: Mean-variance relationship (far above Poisson line)
   - Right: Overdispersion by time period (ranges 1.6-13.5×)

4. **`residual_diagnostics.png`** - Four-panel residual analysis revealing:
   - U-shaped pattern (non-linearity)
   - Acceptable normality
   - Strong temporal structure

### Supporting Plots

5. **`count_distribution.png`** - Histogram showing right-skewed distribution
6. **`boxplot_by_period.png`** - Shows increasing mean and variance over time
7. **`autocorrelation_plot.png`** - Demonstrates strong temporal dependence
8. **`log_transformation_analysis.png`** - Shows log-linear relationship

---

## Statistical Evidence Summary

### Non-linearity
- **AIC comparison:** Quadratic vs Linear Δ = 41.4 (decisive)
- **Visual:** U-shaped residual pattern from linear fit
- **Changepoint analysis:** 6× acceleration in growth rate (p < 0.001)

### Overdispersion
- **Overall:** Variance-to-mean ratio = 68.0 (Poisson expectation = 1.0)
- **By period:** Ranges from 1.6× to 13.5× across time
- **Levene's test:** F = 9.24, p = 0.0001 (significant heterogeneity)

### Temporal Dependence
- **Lag-1 correlation:** r = 0.989, p < 0.001
- **Durbin-Watson:** 0.195 (strong positive autocorrelation)
- **All lags 1-15:** Significant positive autocorrelation

---

## Modeling Implications

### Must Address
1. ✅ **Overdispersion** → Use Negative Binomial (not Poisson)
2. ✅ **Non-linearity** → Include polynomial or exponential terms
3. ✅ **Count data** → Use appropriate distribution (NegBin, Poisson with φ)

### Should Consider
4. **Temporal correlation** → May need AR(1) errors or state space model
5. **Heteroscedastic variance** → Could model time-varying dispersion
6. **Small sample** → Bayesian approach with informative priors

### Can Ignore (for now)
7. Outliers (none detected)
8. Missing data (none present)
9. Measurement error (appears clean)

---

## Model Comparison Results

| Model Type | R² | RMSE | AIC | Recommendation |
|------------|-----|------|-----|----------------|
| Linear | 0.885 | 28.94 | 273.2 | ❌ Inadequate |
| Exponential | 0.929 | 22.76 | 254.0 | ✓ Good alternative |
| **Quadratic** | **0.961** | **16.81** | **231.8** | ✅ **Recommended** |

---

## Data Characteristics

| Property | Value | Interpretation |
|----------|-------|----------------|
| Sample size | 40 | Small (limits model complexity) |
| Time range | -1.668 to 1.668 | Standardized, evenly spaced |
| Count range | 19 to 272 | Wide (14× fold change) |
| Mean count | 109.45 | Moderate level |
| Variance | 7441.74 | Very high |
| Var/Mean ratio | 67.99 | Extreme overdispersion |
| Correlation (r) | 0.941 | Very strong positive trend |
| Autocorr (lag-1) | 0.989 | Extremely high temporal dependence |

---

## Next Steps

### For Modeling Team
1. Implement recommended model in Bayesian framework (Stan/PyMC)
2. Specify weakly informative priors based on data scale
3. Run posterior predictive checks
4. Compare with exponential alternative
5. Check residual autocorrelation; add AR(1) if needed

### For Further Analysis
1. **If available:** Seek domain knowledge to explain acceleration
2. **If possible:** Obtain covariates to explain variation
3. **For forecasting:** Consider saturation/ceiling effects for long-term predictions
4. **For robustness:** Try alternative specifications (splines, GAMs)

---

## Contact

**Analysis performed by:** EDA Specialist Agent
**Framework:** Claude Agent SDK
**Date:** 2025-10-29

For questions about methodology or findings, refer to:
- `eda_report.md` - Comprehensive technical report
- `eda_log.md` - Detailed decision process and reasoning

---

## Software Environment

**Python packages used:**
- pandas - Data manipulation
- numpy - Numerical computing
- matplotlib - Plotting
- seaborn - Statistical visualization
- scipy - Statistical tests and distributions

**All code is reproducible** - Run scripts in `code/` directory in numerical order.

---

**Analysis Status:** ✅ Complete and ready for modeling phase
