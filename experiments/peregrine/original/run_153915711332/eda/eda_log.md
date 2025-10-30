# Exploratory Data Analysis Log

## Dataset: Time Series Count Data
**Date:** 2025-10-29
**Analyst:** EDA Specialist
**Dataset:** `/workspace/data/data.csv`

---

## Phase 1: Initial Data Loading and Validation

### Data Structure
- **Observations:** 40
- **Variables:** 2 (year, C)
- **Data types:** year (float64), C (int64)
- **Missing values:** None
- **Duplicates:** None
- **Data quality:** Excellent - no missing or invalid values

### Time Variable (year)
- Standardized time variable (mean ≈ 0, std ≈ 1)
- Range: [-1.668, 1.668]
- **Evenly spaced:** Yes (spacing = 0.0855)
- Represents approximately 3.34 time units

### Count Variable (C)
- Range: [19, 272]
- All values are positive integers
- No zeros or negative counts
- 33 unique values (high diversity)

**Key Finding:** Clean dataset with proper count structure. Ready for analysis.

---

## Phase 2: Distribution Analysis

### Descriptive Statistics
- **Mean:** 109.45
- **Median:** 74.50
- **Mode:** 19
- **Std Dev:** 86.27
- **Variance:** 7441.74
- **Skewness:** 0.602 (right-skewed)
- **Kurtosis:** -1.233 (light tails, platykurtic)

### Critical Finding: Massive Overdispersion
**Variance-to-Mean Ratio: 67.99**

This is the most important finding for modeling:
- Poisson distribution assumes Var = Mean (ratio ≈ 1)
- Observed ratio of 68 indicates **extreme overdispersion**
- Index of Dispersion Test: χ² = 2651.69, well outside 95% CI [23.65, 58.12]
- **Conclusion:** Strongly reject Poisson assumption

### Normality Tests
**On raw counts (C):**
- Shapiro-Wilk: p = 5.37e-05 → **Reject normality**
- K-S Test: p = 0.053 → Borderline (cannot reject at α=0.05)

**On log-transformed counts:**
- Shapiro-Wilk: p = 0.003 → **Reject log-normality**

**Interpretation:** Neither normal nor log-normal distribution fits well. This is expected for count data with overdispersion.

### Outlier Detection
- **IQR method (1.5×IQR):** 0 outliers
- **Z-score method (|z| > 3):** 0 outliers
- No extreme values requiring removal

**Visualization:** `01_distribution_analysis.png`
- Histogram shows bimodal pattern (low counts early, high counts late)
- Q-Q plot shows S-shaped curve (non-normal)
- Box plot confirms no outliers
- Log-scale histogram also shows bimodality

---

## Phase 3: Temporal Trend Analysis

### Linear Model Performance
**Equation:** C = 81.133 × year + 109.450

- **R²:** 0.8846 (88.5% variance explained)
- **Correlation:** 0.9405 (very strong)
- **P-value:** 2.09e-19 (highly significant)
- **Slope 95% CI:** [71.81, 90.45]

**Growth Metrics:**
- Initial count (year = -1.67): 29
- Final count (year = 1.67): 245
- **Absolute growth:** 216 counts
- **Percentage growth:** 745%
- **Growth factor:** 8.45×

### Residual Diagnostics
**Linear Model Residuals:**
- Mean: 0 (as expected)
- Std Dev: 29.31
- Range: [-43.10, 54.98]
- Shapiro-Wilk: p = 0.0712 → Residuals approximately normal ✓
- **Durbin-Watson: 0.195** → **Strong positive autocorrelation** ⚠️

**Critical Finding:** The very low D-W statistic (<<2) indicates strong serial correlation in residuals. This violates independence assumption of standard regression.

### Residual Pattern Analysis
From `02_temporal_analysis.png` (Residual Plot):
- Clear U-shaped pattern in residuals vs fitted values
- Smoothed trend shows systematic curvature
- Early predictions: overestimate
- Middle predictions: underestimate
- Late predictions: overestimate again

**Conclusion:** Linear model is inadequate. Relationship is nonlinear.

### Exponential Model Performance
**Equation:** C = 77.14 × exp(0.850 × year)

- **R² (on log scale):** 0.9354 (93.5% variance explained)
- **Better fit than linear model** in log-space
- **Growth rate:** 134% multiplicative increase per standardized year
- Log-scale plot shows better linearity

**Interpretation:** Exponential growth pattern is present and strong.

### Polynomial Model Comparison

| Degree | R² | AIC | BIC |
|--------|------|------|------|
| 1 (Linear) | 0.8846 | 273.2 | 276.6 |
| 2 (Quadratic) | 0.9610 | 231.8 | 236.8 |
| 3 (Cubic) | 0.9757 | 214.9 | 221.7 |
| 4 (Quartic) | 0.9900 | 181.2 | 189.6 |

**Key Findings:**
- Each polynomial degree improves fit substantially
- Quadratic model: ΔR² = +0.076, ΔAIC = -41.4
- Degree 4 achieves R² = 0.99 but risks overfitting
- **Recommendation:** Quadratic or cubic model balances fit and complexity

### Heteroscedasticity Analysis

**Breusch-Pagan Test (informal):**
- Slope of (residuals² vs year): -159.50
- P-value: 0.243 → No evidence from this test

**However, period comparison reveals strong heteroscedasticity:**

| Period | Mean | Variance |
|--------|------|----------|
| Early (year < 0) | 37.20 | 157.64 |
| Late (year ≥ 0) | 181.70 | 4127.91 |

- **Variance ratio:** 26.19 (late/early)
- **F-test:** p < 0.0001 → **Variances differ significantly**

**Visualization Finding:** The variance structure plot shows both mean and variance increase dramatically over time, with variance growing faster than mean.

### Mean-Variance Relationship
**Regression:** Variance = 3.151 × Mean + 83.186
- R² = 0.349 (weak relationship in bins)
- Slope > 1 suggests **quadratic mean-variance relationship**
- Typical of Negative Binomial or Gamma distributions

**Visualization:** `02_temporal_analysis.png`

---

## Phase 4: Advanced Diagnostic Analysis

### Autocorrelation Structure (ACF/PACF)

**Autocorrelation Function:**
- ACF(1) = 0.9886 (extremely high!)
- ACF(2) = 0.9829
- ACF(3) = 0.9713
- Significant lags: 1-10+ (all exceed confidence bands)

**Critical Finding:** Strong, persistent autocorrelation indicates:
1. Non-independent observations
2. Time series structure is dominant
3. Standard regression assumptions violated
4. ARIMA or state-space models may be appropriate

**Partial Autocorrelation:**
- PACF(1) = 0.989 (very high)
- PACF(2-20) mostly within confidence bands
- Suggests **AR(1) process** may be suitable

### Stationarity Analysis

**Original Series:**
- Mean: 109.45
- Variance: 7441.74
- Clearly non-stationary (strong trend)

**First Differences:**
- Mean: 5.54
- Variance: 170.04 (much smaller!)
- Std Dev: 13.04

**Finding:** Differencing reduces variance by 98%, suggesting series is **integrated of order 1 (I(1))**.

### Changepoint Detection

**CUSUM Analysis:**
- Maximum |CUSUM| at index 23 (year = 0.299)
- Count at changepoint: 95

**Before vs After Comparison:**
- Before (n=24): Mean = 45.67, SD = 22.52
- After (n=16): Mean = 205.12, SD = 47.93
- **T-test:** t = -14.18, p < 0.0001 → **Highly significant difference**

**Interpretation:** Clear regime shift around year = 0.3. The process transitions from slow growth (mean ~46) to rapid growth (mean ~205).

### Growth Rate Dynamics

**Period-over-Period Growth:**
- Mean: 7.72%
- Median: 6.25%
- Std Dev: 21.64% (high variability)
- Range: [-47.22%, 84.21%]

**Growth Comparison:**
- First half mean: 8.02%
- Second half mean: 7.43%
- No clear acceleration pattern (roughly constant growth rate)

**Finding:** Growth rate is relatively stable but highly variable, consistent with multiplicative process with noise.

### Lag-1 Dependence

**C(t+1) vs C(t) Regression:**
- Correlation: 0.9886
- R²: 0.9773
- **Equation:** C(t+1) = 1.011 × C(t) + 4.417

**Interpretation:**
- Near-perfect momentum (slope ≈ 1)
- Past values are extremely predictive of future
- Suggests **random walk with drift** or **autoregressive process**

**Visualization:** `03_advanced_diagnostics.png`
- Lag-1 scatter plot shows tight linear relationship
- Points cluster tightly around regression line
- Minimal deviation except at extremes

---

## Phase 5: Hypothesis Testing

### Hypothesis 1: Poisson Process
**Status:** REJECTED ✗

**Evidence:**
- Variance/Mean ratio = 68 (should be ≈1)
- Index of dispersion test strongly rejects
- Coefficient of variation = 0.79 (too high for Poisson)

### Hypothesis 2: Linear Growth
**Status:** PARTIALLY SUPPORTED ~

**Evidence:**
- High R² = 0.885 suggests linear component
- But residual patterns show nonlinearity
- Polynomial models provide better fit
- Exponential model also performs better

**Conclusion:** Linear trend exists but is insufficient alone.

### Hypothesis 3: Exponential Growth
**Status:** STRONGLY SUPPORTED ✓

**Evidence:**
- Log-linear R² = 0.935 (better than linear)
- Log-scale plot shows good linearity
- Growth factor of 8.45× over time period
- Consistent with multiplicative process

### Hypothesis 4: Changepoint/Regime Shift
**Status:** SUPPORTED ✓

**Evidence:**
- CUSUM shows clear inflection at year ≈ 0.3
- Means differ significantly (p < 0.0001)
- Visual inspection confirms transition point

### Hypothesis 5: Time Series Dependencies
**Status:** STRONGLY SUPPORTED ✓

**Evidence:**
- ACF shows persistent autocorrelation
- Durbin-Watson = 0.195 (strong positive correlation)
- Lag-1 R² = 0.977
- Series is integrated I(1)

---

## Key Data Characteristics Summary

### What We Know with High Confidence:
1. **Extreme overdispersion** (Var/Mean = 68)
2. **Strong temporal trend** (8.45× growth)
3. **Nonlinear growth pattern** (exponential or polynomial)
4. **Heteroscedasticity** (variance increases with time/mean)
5. **Strong autocorrelation** (ACF > 0.98 at lag-1)
6. **Non-stationary** but becomes stationary after differencing
7. **Changepoint around year = 0.3**
8. **No outliers or data quality issues**

### What Remains Uncertain:
1. Exact functional form (exponential vs polynomial vs other)
2. Whether changepoint is real structural break or smooth transition
3. Source of overdispersion (true Negative Binomial vs other mechanisms)
4. Whether growth will continue or plateau (extrapolation risk)

---

## Modeling Implications

### Distributions to Consider:
1. **Negative Binomial** (for overdispersion) ✓✓✓
2. **Quasi-Poisson** (for overdispersion with Poisson-like structure) ✓✓
3. **Zero-Inflated models** - NOT needed (no zeros observed)
4. Poisson - REJECT

### Regression Structures:
1. **Exponential growth:** log(λ) = β₀ + β₁×year
2. **Polynomial trend:** log(λ) = β₀ + β₁×year + β₂×year²
3. **Piecewise/changepoint model** at year ≈ 0.3
4. **Time series models:** ARIMA(p,1,q) given I(1) structure

### Variance Modeling:
- **Heteroscedastic models** recommended
- Mean-variance relationship suggests Var ∝ Mean²
- Consider robust standard errors

### Critical Considerations:
- **Autocorrelation** violates standard GLM assumptions
- May need **GEE** (Generalized Estimating Equations)
- Or **mixed effects** with temporal correlation structure
- Or explicit **time series approach** (ARIMA, state-space)

---

## Recommendations for Modeling Phase

### Primary Model Classes (Ranked):

**1. Negative Binomial GLM with Exponential/Polynomial Trend**
- Address overdispersion directly
- Model: NB(μ, θ) with log(μ) = β₀ + β₁×year + β₂×year²
- May need autocorrelation adjustment (GEE or robust SE)

**2. Time Series Count Model (ARIMA-like)**
- Leverage strong AR(1) structure
- Model first differences with count distribution
- State-space model with Poisson/NB observation equation

**3. Changepoint/Piecewise Model**
- Two regimes: before/after year = 0.3
- Separate growth rates for each period
- Negative Binomial distribution within regimes

### Model Comparison Strategy:
- Use AIC/BIC for nested models
- Cross-validation for predictive performance
- Residual diagnostics (especially autocorrelation)
- Posterior predictive checks (if Bayesian)

### Variables/Transformations:
- **Primary predictor:** year (standardized time)
- **Consider:** year², year³ for nonlinearity
- **Consider:** changepoint indicator (year ≥ 0.3)
- **Consider:** lagged counts C(t-1) as predictor
- **Do NOT log-transform C** (loses count structure)

---

## Conclusion

This is a **complex time series count dataset** with:
- Extreme overdispersion requiring Negative Binomial or similar
- Strong exponential/nonlinear growth trend
- Significant autocorrelation requiring special handling
- Possible structural changepoint
- Increasing variance over time

**Standard Poisson regression is completely inappropriate.** The ideal model will combine:
1. Count distribution with overdispersion (NB)
2. Nonlinear trend (exponential/polynomial)
3. Temporal dependence structure (AR, GEE, or similar)

The high R² values and strong patterns suggest **the data is highly structured and predictable**, but proper handling of overdispersion and autocorrelation is essential for valid inference.
