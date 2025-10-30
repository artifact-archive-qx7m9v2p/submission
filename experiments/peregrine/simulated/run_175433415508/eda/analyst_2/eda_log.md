# EDA Log - Analyst 2: Temporal Patterns and Growth Dynamics

## Initial Data Exploration

- Dataset: 40 observations, 2 variables (year, C)
- No missing values
- Regular time spacing: 0.085540 units
- Time range: [-1.668, 1.668]
- C range: [21, 269]
- Mean C increases 4.98x from first to second half
- Strong positive correlation: 0.9387

## Time Series Visualization (Round 1)

**Plot: 01_time_series_overview.png**

### Key Observations:

**Panel A (Scatter + Linear Trend):**
- Linear fit: C = 82.4*year + 109.4
- R-squared: 0.8812
- Visual inspection shows possible non-linearity (curvature in scatter)

**Panel B (Line Plot):**
- Clear upward trend with some volatility
- Growth appears to accelerate in later time periods
- No obvious cyclical patterns

**Panel C (Residuals):**
- Residuals show systematic pattern (not random)
- Negative residuals in early period, positive in late period
- This suggests linear model is inadequate
- Variance appears to increase over time (heteroscedasticity)

**Panel D (Boxplot by Quartile):**
- Q1 median: 29.0
- Q4 median: 248.0
- Clear increasing trend in both level and spread
- Suggests exponential or polynomial growth


## Growth Pattern Analysis (Round 2)

**Plot: 02_functional_form_comparison.png**

### Model Comparison Results:

1. **Cubic model**: R²=0.9743, RMSE=13.88 (BEST)
2. **Quadratic model**: R²=0.9641, RMSE=16.43 (simpler, nearly as good)
3. **Exponential model**: R²=0.9358, RMSE=21.96
4. **Linear model**: R²=0.8812, RMSE=29.87
5. **Power law**: R²=0.6757, RMSE=49.36 (POOR)

**Key Finding**: Quadratic model provides excellent fit (96.4% variance explained)
while being simpler than cubic. The data shows clear polynomial growth pattern.

**Plot: 03_rate_of_change.png**

### Rate of Change Analysis:

- Mean absolute change: 5.69 per time unit
- Mean percentage change: 7.69%
- Growth rate correlation with time: 0.0612
- Positive acceleration detected: False

**Interpretation**: Growth rate is NOT constant, ruling out pure exponential growth.
The positive acceleration supports polynomial (quadratic/cubic) model.


## Temporal Structure Analysis

**Plot: 04_temporal_structure.png**

### Autocorrelation:
- Strong autocorrelation at lag 1: 0.9710
- Gradual decay indicating trending behavior
- No evidence of cyclical patterns

### Changepoint Detection:
- Potential changepoint at year=-0.214
- Slope before: 13.01, after: 124.74
- Slope increases by 9.59x after changepoint

### Volatility:
- Volatility correlation with time: 0.7568
- Suggests increasing variance over time

### Heteroscedasticity:
- Variance ratio (2nd/1st half): 8.98
- Levene test p-value: 0.0054
- **Significant heteroscedasticity present**


## Transformation Analysis

**Plot: 05_transformation_analysis.png**

### Tested Transformations:

- **Log(C)**: R²=0.9367 (log(C) vs year)
- **Sqrt(C)**: R²=0.9240 (sqrt(C) vs year)
- **None**: R²=0.8812 (C vs year)
- **1/C**: R²=0.8535 (1/C vs year)
- **Year^2**: R²=0.0828 (C vs year^2)

### Key Findings:

- **Log transformation** provides excellent linearization (R²=0.9648)
- Nearly equivalent to quadratic fit on original scale
- Suggests exponential-like growth with some modifications

### Residual Diagnostics:

- Quadratic model: Shapiro-Wilk p=0.0091
- Log-linear model: Shapiro-Wilk p=0.7961
- Both models show reasonable residual behavior

