# Exploratory Data Analysis Output

## Quick Navigation

### Main Reports
- **[EDA Report](eda_report.md)** - Comprehensive findings and modeling recommendations (START HERE)
- **[EDA Log](eda_log.md)** - Detailed exploration process and iterative findings

### Deliverables Summary

**Dataset**: 27 observations of Y vs x relationship

**Key Finding**: Strong evidence for nonlinear, asymptotic relationship with two distinct regimes (growth and plateau phases)

**Recommended Models**:
1. Logarithmic: Y ~ β₀ + β₁*log(x) [BEST FIT, R²=0.897]
2. Piecewise linear with changepoint at x≈7 [MOST INTERPRETABLE]
3. Asymptotic saturation model [THEORETICALLY MOTIVATED]

**Data Quality**: Excellent (no missing values, minimal outliers, 1 influential point)

---

## Directory Structure

```
/workspace/eda/
├── README.md              # This file
├── eda_report.md          # Main comprehensive report
├── eda_log.md            # Detailed exploration log
├── code/                  # Reproducible analysis scripts
│   ├── 01_initial_exploration.py
│   ├── 02_univariate_analysis.py
│   ├── 03_bivariate_analysis.py
│   ├── 04_nonlinearity_investigation.py
│   ├── 05_changepoint_visualization.py
│   └── 06_outlier_influence_analysis.py
└── visualizations/        # All plots (9 total)
    ├── 01_x_distribution.png
    ├── 02_Y_distribution.png
    ├── 03_bivariate_analysis.png
    ├── 04_variance_analysis.png
    ├── 05_functional_forms.png
    ├── 06_transformations.png
    ├── 07_changepoint_analysis.png
    ├── 08_rate_of_change.png
    └── 09_outlier_influence.png
```

---

## Visualization Guide

### Understanding the Data

| File | Description | Key Insights |
|------|-------------|--------------|
| `01_x_distribution.png` | Predictor distribution (6 panels: histogram, KDE, boxplot, Q-Q, ECDF, sequential) | Right-skewed, sparse at high values |
| `02_Y_distribution.png` | Response distribution (same 6-panel layout) | Approximately normal, narrow range |

### Relationship Analysis

| File | Description | Key Insights |
|------|-------------|--------------|
| `03_bivariate_analysis.png` | Scatter plots, fits, residuals (6 panels) | Clear nonlinearity, systematic residual patterns |
| `04_variance_analysis.png` | Heteroscedasticity check, rate of change (2 panels) | Constant variance, decreasing rate of change |

### Model Comparison

| File | Description | Key Insights |
|------|-------------|--------------|
| `05_functional_forms.png` | 6 different models compared (2×3 grid) | Logarithmic model best R² (0.897) |
| `06_transformations.png` | Log-log, semi-log, reciprocal (2×2 grid) | Log-log transformation linearizes relationship |

### Regime Analysis

| File | Description | Key Insights |
|------|-------------|--------------|
| `07_changepoint_analysis.png` | Piecewise model details (2×2 grid) | Strong evidence for x≈7 changepoint |
| `08_rate_of_change.png` | Local slopes by region (2 panels) | Confirms two-regime structure |

### Data Quality

| File | Description | Key Insights |
|------|-------------|--------------|
| `09_outlier_influence.png` | Diagnostics: Cook's D, leverage, DFFITS (2×3 grid) | One influential outlier at x=31.5 |

---

## Code Execution

All scripts are standalone and can be run independently:

```bash
cd /workspace
python eda/code/01_initial_exploration.py
python eda/code/02_univariate_analysis.py
python eda/code/03_bivariate_analysis.py
python eda/code/04_nonlinearity_investigation.py
python eda/code/05_changepoint_visualization.py
python eda/code/06_outlier_influence_analysis.py
```

**Dependencies**: pandas, numpy, matplotlib, seaborn, scipy

**Total runtime**: ~10 seconds

---

## Key Statistics at a Glance

### Data Quality
- Sample size: 27
- Missing values: 0
- Duplicates: 1
- Outliers: 1 (3.7%)

### Simple Linear Model (Inadequate)
- R² = 0.677
- RMSE = 0.153
- p < 0.000001

### Logarithmic Model (RECOMMENDED)
- R² = 0.897
- RMSE = 0.087
- Formula: Y = 2.020 + 0.290*log(x)

### Piecewise Linear Model
- Changepoint: x = 7.0
- Regime 1 slope: 0.113 (steep)
- Regime 2 slope: 0.017 (flat)
- F-statistic: 22.4 (p < 0.0001)

### Correlations
- Pearson r = 0.823
- Spearman ρ = 0.920

---

## Next Steps

1. Fit recommended models in Bayesian framework
2. Compare using WAIC/LOO-CV
3. Run sensitivity analysis excluding x=31.5 observation
4. Perform posterior predictive checks
5. Consider model averaging if appropriate

---

## Questions?

Refer to:
- **[eda_report.md](eda_report.md)** for comprehensive findings
- **[eda_log.md](eda_log.md)** for detailed exploration process

**Generated**: 2025-10-28
**Analyst**: EDA Specialist
