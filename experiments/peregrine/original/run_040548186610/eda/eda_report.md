# Exploratory Data Analysis Report: Time Series Count Data

**Date:** 2025-10-29
**Dataset:** `/workspace/data/data.csv`
**Observations:** 40
**Analysis performed by:** EDA Specialist

---

## Executive Summary

This EDA analyzed 40 observations of time series count data with variables `year` (normalized time) and `C` (count observations). The analysis reveals a **strong non-linear growth trend** with **significant overdispersion** and **high temporal autocorrelation**. The data exhibits characteristics consistent with exponential or accelerating growth, requiring specialized count data models that account for overdispersion and non-linear trends.

**Key Findings:**
- Strong positive trend (Pearson r = 0.941, p < 0.001)
- Extreme overdispersion: variance-to-mean ratio = 68.0 (far exceeding Poisson assumption of 1.0)
- Non-linear growth: quadratic and exponential models substantially outperform linear (AIC improvement: 41 points)
- Accelerating trend: growth rate increases 6x from early to late period
- Strong temporal autocorrelation (lag-1 r = 0.989)
- Heteroscedastic variance across time periods (Levene test p < 0.001)

---

## 1. Data Quality Assessment

### 1.1 Structure and Completeness
- **Shape:** 40 observations × 2 variables
- **Variables:**
  - `year`: normalized/standardized time variable (continuous, range: -1.668 to 1.668)
  - `C`: count observations (integer, range: 19 to 272)
- **Missing values:** None (0%)
- **Duplicates:** None
- **Data types:** Appropriate (float64 for year, int64 for counts)

**Visual Evidence:** See `timeseries_plot.png`

### 1.2 Time Variable Properties
- **Spacing:** Perfectly evenly spaced (interval = 0.0855)
- **Normalization:** Centered at 0 (mean = 0.000, SD = 1.000)
- **Interpretation:** Standardized time variable facilitating model interpretation

### 1.3 Data Quality Conclusion
**Status: EXCELLENT** - Clean dataset with no missing values, outliers, or data quality issues. Ready for modeling.

---

## 2. Univariate Distributions

### 2.1 Count Variable (C)

**Descriptive Statistics:**
| Statistic | Value |
|-----------|-------|
| Mean | 109.45 |
| Median | 74.50 |
| Std Dev | 86.27 |
| Variance | 7441.74 |
| Min | 19 |
| Max | 272 |
| Range | 253 |
| Skewness | 0.602 |
| Kurtosis | -1.233 |

**Key Observations:**
- **Right-skewed distribution** (mean > median by 47%)
- **High dispersion:** coefficient of variation = 0.79
- **Wide range:** 14.3× spread from min to max
- **Negative kurtosis:** flatter than normal (platykurtic)

**Visual Evidence:** See `count_distribution.png`

**Interpretation:** The right-skewed distribution with large variance suggests the data may follow a negative binomial or similar overdispersed count distribution rather than Poisson.

### 2.2 Overdispersion Analysis

**Variance-to-Mean Ratio: 67.99**

This is **extremely high** compared to Poisson (ratio = 1). This indicates:
1. Simple Poisson regression will be inadequate
2. Need negative binomial, quasi-Poisson, or zero-inflated models
3. Variance increases faster than mean (multiplicative dispersion)

**Visual Evidence:** See `variance_analysis.png`

**Variance by Time Period:**
| Period | Mean | Variance | Var/Mean Ratio |
|--------|------|----------|----------------|
| Q1 (early) | 28.3 | 46.7 | 1.65 |
| Q2 | 46.1 | 110.1 | 2.39 |
| Q3 | 127.9 | 1733.2 | **13.55** |
| Q4 (late) | 235.5 | 549.2 | 2.33 |

**Critical Finding:** Overdispersion is NOT constant across time. Q3 shows extreme overdispersion (13.5×), while other periods show moderate overdispersion (1.6-2.4×). This suggests the variance structure may be more complex than simple constant overdispersion.

---

## 3. Relationship Analysis: Year vs Count

### 3.1 Correlation Analysis

**Strength of Association:**
- **Pearson correlation:** r = 0.9405, p = 2.09 × 10⁻¹⁹
- **Spearman correlation:** ρ = 0.9664, p = 5.17 × 10⁻²⁴

Both correlations are **extremely strong** and highly significant, but Spearman > Pearson suggests some non-linearity.

**Visual Evidence:** See `scatter_with_smoothing.png`

### 3.2 Linearity Assessment

**Model Comparison:**

| Model | R² | RMSE | AIC | Delta AIC |
|-------|-----|------|-----|-----------|
| Linear | 0.8846 | 28.94 | 273.22 | +41.44 |
| Exponential | 0.9286 | 22.76 | 253.99 | +22.21 |
| **Quadratic** | **0.9610** | **16.81** | **231.78** | **0.00** |

**Equations:**
- **Linear:** C = 81.13 × year + 109.45
- **Exponential:** C = 77.14 × exp(0.850 × year)  [growth rate: 134% per unit year]
- **Quadratic:** C = 27.04 × year² + 81.13 × year + 83.09

**Conclusion:** The relationship is **strongly non-linear**. The quadratic model provides the best fit (41-point AIC improvement over linear), suggesting **accelerating growth**. The exponential model also fits well (19-point improvement), consistent with exponential/multiplicative growth.

**Visual Evidence:** In `scatter_with_smoothing.png`, the LOWESS smooth and polynomial curves closely follow the data, while the linear fit systematically under-predicts at both extremes.

### 3.3 Changepoint Analysis

Testing for a structural break at the median year:

**Early vs Late Period:**
| Period | N | Mean | SD | Slope |
|--------|---|------|-----|-------|
| Early (year < 0) | 20 | 37.2 | 12.6 | 20.18 counts/year |
| Late (year ≥ 0) | 20 | 181.7 | 64.3 | 122.06 counts/year |

**Statistical Tests:**
- **t-test:** t = -9.87, p = 4.89 × 10⁻¹² (highly significant difference)
- **Mann-Whitney U:** U = 0.0, p = 6.73 × 10⁻⁸ (highly significant)

**Key Finding:** The growth rate **accelerates by 6× in the later period**. This provides strong evidence for non-linear (accelerating) growth rather than constant linear growth.

---

## 4. Residual Analysis (Linear Model)

### 4.1 Residual Patterns

**Visual Evidence:** See `residual_diagnostics.png`

**Observations:**

1. **Residuals vs Fitted:** Shows clear **U-shaped pattern** (quadratic trend), confirming non-linearity
   - Under-prediction at low and high fitted values
   - Over-prediction at middle fitted values
   - This validates the need for quadratic or exponential terms

2. **Q-Q Plot:** Residuals are **approximately normal** with slight deviations at tails
   - Shapiro-Wilk test: W = 0.949, p = 0.071 (marginally non-significant)
   - Acceptable for most modeling purposes

3. **Residuals vs Year:** Strong quadratic pattern visible
   - Negative residuals in middle period
   - Positive residuals at both ends
   - Confirms time-dependent systematic bias

### 4.2 Heteroscedasticity

**Tests:**
- **Correlation(|residuals|, fitted):** r = -0.174, p = 0.284 (not significant with linear model)
- **Levene's test (quartiles):** F = 9.24, p = 0.0001 (highly significant)

**Interpretation:** While residuals from the linear model don't show obvious heteroscedasticity, the **raw data variance differs significantly across time periods**. This is obscured in the linear residuals due to model misspecification. The true variance structure is heteroscedastic.

---

## 5. Temporal Patterns

### 5.1 Autocorrelation

**Visual Evidence:** See `autocorrelation_plot.png`

**Key Metrics:**
- **Lag-1 autocorrelation:** r = 0.989, p < 0.001 (extremely high)
- **Durbin-Watson statistic:** 0.195 (strong positive autocorrelation; expect ~2 for independence)
- **Runs test:** 2 observed runs vs 21 expected (highly non-random)

**Interpretation:** The counts exhibit **very strong temporal dependence**. Each observation is highly correlated with adjacent observations. This violates independence assumptions of standard regression and suggests:
1. Need for time series models (ARIMA, state space)
2. Or models with temporal structure (random walks, GP regression)
3. Standard errors from naive models will be underestimated

### 5.2 Trend Decomposition

The observed pattern suggests the data generating process has:
- **Strong deterministic trend** (non-linear growth)
- **High persistence/momentum** (strong autocorrelation)
- **Multiplicative noise** (variance increases with level)

This is consistent with:
- Population growth models
- Epidemic spread
- Technology adoption curves
- Financial time series (with positive drift)

---

## 6. Log Transformation Analysis

**Visual Evidence:** See `log_transformation_analysis.png`

**Log-Linear Model:**
- **Equation:** log(C) = 0.850 × year + 4.346
- **R² on log scale:** 0.9354
- **Interpretation:** Exponential growth at 134% per unit year

**Distribution of log(C):**
- More symmetric than raw counts
- Closer to normal distribution
- Reduces heteroscedasticity

**Modeling Implication:** Working on log scale may stabilize variance and improve residual properties. This supports using:
- Log-normal models
- Poisson/negative binomial with log link (GLMs)
- Exponential family models

---

## 7. Outliers and Influential Points

### 7.1 Outlier Detection

**Standardized residuals > 2.5:** None detected

**Conclusion:** No extreme outliers that would unduly influence model fitting.

### 7.2 Influential Points

**Cook's Distance > 4/n (0.100):** 3 points identified
- **Indices:** 0, 1, 35
- **Interpretation:** First two and one late observation have higher influence
  - Expected at series endpoints due to leverage
  - Not concerning as values appear reasonable

**Recommendation:** No data points require removal. All observations appear genuine.

---

## 8. Key Modeling Challenges

### 8.1 Primary Challenges

1. **Overdispersion (Var/Mean = 68)**
   - Cannot use simple Poisson regression
   - Need negative binomial or quasi-likelihood approach

2. **Non-linear Growth**
   - Linear models inadequate (large residual patterns)
   - Need polynomial, exponential, or spline terms

3. **Temporal Autocorrelation (r = 0.989)**
   - Violates independence assumption
   - Need time series structure or correlated errors

4. **Heteroscedastic Variance**
   - Variance changes across time
   - May need variance modeling (e.g., dispersion parameter as function of mean)

5. **Small Sample Size (n=40)**
   - Limited data for complex models
   - Need to balance flexibility vs overfitting

### 8.2 Data Generation Process Hypotheses

**Most Likely:** Exponential/accelerating growth process with:
- Multiplicative error structure
- Temporal momentum/persistence
- Positive feedback mechanisms

**Possible Mechanisms:**
- Population growth (birth > death, compound growth)
- Contagion/diffusion process (awareness spreading)
- Technology adoption (S-curve, currently in growth phase)
- Accumulation with feedback

---

## 9. Modeling Recommendations

### 9.1 Recommended Model Families (Ranked)

#### **Tier 1: Best Candidates**

1. **Negative Binomial Regression with Non-linear Trend**
   - **Model:** `C ~ NegBinomial(μ, φ)`
   - **Link:** `log(μ) = β₀ + β₁·year + β₂·year²`
   - **Rationale:** Handles overdispersion + non-linear trend
   - **Bayesian implementation:** Easy in Stan/PyMC
   - **Prior considerations:**
     - Weakly informative priors on β coefficients
     - Half-normal prior on dispersion parameter φ

2. **Poisson/Negative Binomial with Exponential Trend + AR Errors**
   - **Model:** `log(μ) = β₀ + β₁·year` with AR(1) structure
   - **Rationale:** Captures exponential growth + temporal correlation
   - **Bayesian implementation:** State space model
   - **Challenge:** More complex, requires careful prior specification

3. **Log-Normal Regression with Polynomial Trend**
   - **Model:** `log(C) ~ Normal(μ, σ)`
   - **Link:** `μ = β₀ + β₁·year + β₂·year²`
   - **Rationale:** Log transformation reduces heteroscedasticity
   - **Advantage:** Simpler than GLMs, but still flexible
   - **Note:** Technically C is discrete, but approximation may work well

#### **Tier 2: Alternative Approaches**

4. **Gaussian Process Regression on Log Scale**
   - **Model:** `log(C) ~ GP(mean_function, kernel)`
   - **Rationale:** Flexible non-parametric approach, handles non-linearity naturally
   - **Advantage:** Doesn't assume functional form
   - **Challenge:** Small sample (n=40) may limit GP benefits

5. **Dynamic Linear Model (State Space)**
   - **Model:** Bayesian state space with time-varying level/trend
   - **Rationale:** Explicitly models temporal evolution
   - **Advantage:** Can capture changepoints and evolving dynamics
   - **Challenge:** More parameters, may overfit with n=40

#### **Tier 3: Not Recommended**

6. ❌ **Simple Poisson Regression** - Fails due to extreme overdispersion
7. ❌ **Linear Regression** - Ignores non-linearity and count nature
8. ❌ **Independent Models** - Ignores temporal correlation

### 9.2 Recommended Starting Point

**Model:** Negative Binomial Regression with Quadratic Trend

```
C_i ~ NegBinomial(μ_i, φ)
log(μ_i) = β₀ + β₁·year_i + β₂·year_i²

Priors:
  β₀ ~ Normal(log(100), 1)     # Based on observed mean
  β₁ ~ Normal(0, 1)             # Positive trend expected
  β₂ ~ Normal(0, 0.5)           # Acceleration term
  φ ~ Half-Normal(0, 5)         # Dispersion parameter
```

**Justification:**
- Accounts for overdispersion via negative binomial
- Captures non-linear growth via quadratic term
- Maintains interpretability
- Computationally tractable
- Can be extended with AR structure if needed

### 9.3 Model Validation Strategy

**Cross-validation approach:**
1. Use leave-one-out cross-validation (LOO-CV) for small sample
2. Check posterior predictive distributions
3. Validate on last 20% of time series (out-of-sample forecast)

**Diagnostics to check:**
1. Posterior predictive checks (PPC)
2. Residual patterns (should be random)
3. Dispersion adequacy (variance-to-mean ratio captured)
4. Forecast accuracy on held-out data

### 9.4 Model Comparison Criteria

**Primary:**
- **WAIC/LOO-IC:** For Bayesian model comparison
- **Posterior predictive coverage:** % of obs in 95% credible intervals

**Secondary:**
- **RMSE:** Absolute prediction error
- **MAE:** Robust to outliers
- **Calibration plots:** Predicted vs observed quantiles

---

## 10. Limitations and Cautions

### 10.1 Data Limitations

1. **Small sample size (n=40):** Limits complexity of models that can be reliably fit
2. **Single time series:** Cannot assess between-series variation or external validity
3. **No covariates:** Cannot explain WHY counts are increasing
4. **Limited history:** May not capture full dynamic range of process

### 10.2 Analysis Limitations

1. **Tentative findings:**
   - Exact functional form (quadratic vs exponential) uncertain with n=40
   - Variance structure may be more complex than detected
   - Could be early phase of S-curve (logistic) that appears exponential

2. **Extrapolation risk:**
   - Exponential/quadratic trends cannot continue indefinitely
   - Predictions outside observed range highly uncertain
   - May need saturation/ceiling effects for long-term forecasts

3. **Assumption violations:**
   - High autocorrelation complicates inference
   - Some model assumptions (independence) clearly violated
   - Need robust standard errors or fully specified temporal model

---

## 11. Summary: Modeling Checklist

### Critical Requirements
- ✅ **Must handle overdispersion** (Var/Mean = 68)
- ✅ **Must model non-linear trend** (quadratic or exponential)
- ✅ **Should consider temporal correlation** (lag-1 r = 0.989)
- ✅ **Should use count-appropriate distributions** (negative binomial, Poisson)

### Nice to Have
- Heteroscedastic variance modeling
- Changepoint detection
- Saturation/ceiling effects for long-term prediction

### Starting Model Recommendation
**Negative Binomial GLM with quadratic trend in Bayesian framework**
- Addresses: overdispersion ✓, non-linearity ✓, count data ✓
- Can be extended: Add AR(1) errors if needed, time-varying dispersion
- Interpretable: Clear parameters for trend and dispersion

---

## 12. Files Generated

### Code
- `/workspace/eda/code/01_initial_exploration.py` - Data loading and descriptive statistics
- `/workspace/eda/code/02_visualization_analysis.py` - All visualizations and graphics
- `/workspace/eda/code/03_hypothesis_testing.py` - Statistical tests and model comparisons

### Visualizations (all 300 DPI PNG)
- `timeseries_plot.png` - Time series of counts over time
- `count_distribution.png` - Histogram with density of count variable
- `scatter_with_smoothing.png` - Scatter plot with linear, polynomial, and smooth fits
- `residual_diagnostics.png` - Four-panel residual analysis from linear model
- `variance_analysis.png` - Mean-variance relationship and overdispersion by period
- `boxplot_by_period.png` - Distribution of counts by time quartile
- `autocorrelation_plot.png` - ACF showing temporal dependence
- `log_transformation_analysis.png` - Log-scale analysis for exponential growth

### Reports
- `/workspace/eda/eda_report.md` - This comprehensive report
- `/workspace/eda/eda_log.md` - Detailed exploration process log

---

## Conclusion

This dataset presents a **clear and strong non-linear growth pattern** with **extreme overdispersion** and **high temporal correlation**. The most appropriate modeling approach is a **negative binomial GLM with quadratic or exponential trend**, implemented in a Bayesian framework to facilitate uncertainty quantification. The strong autocorrelation suggests consideration of time series extensions (AR errors or state space models) may further improve model fit.

The data quality is excellent with no concerning outliers or missing values. All findings are statistically robust. The primary modeling challenge is balancing model complexity against the limited sample size (n=40) while adequately capturing the three key features: overdispersion, non-linearity, and temporal dependence.

**Recommended next step:** Fit negative binomial regression with quadratic trend as baseline, then evaluate need for temporal correlation structure based on residual diagnostics.

---

**Report prepared by:** EDA Specialist Agent
**Date:** 2025-10-29
**All code and visualizations available in:** `/workspace/eda/`
