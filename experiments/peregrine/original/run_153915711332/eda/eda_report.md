# Exploratory Data Analysis Report
## Time Series Count Data Analysis

**Dataset:** `/workspace/data/data.csv`
**Analysis Date:** 2025-10-29
**Observations:** 40
**Variables:** year (standardized time), C (count outcome)

---

## Executive Summary

This EDA reveals a **complex time series count dataset** with extreme overdispersion and strong temporal dependencies. The data exhibits an 8.45× growth over the observation period, with variance increasing 26× from early to late periods. **Standard Poisson models are completely inappropriate** for this data. The analysis strongly recommends Negative Binomial or quasi-Poisson models with exponential/polynomial trends and explicit modeling of temporal autocorrelation.

### Critical Findings:
1. **Extreme overdispersion:** Variance/Mean ratio = 67.99 (Poisson would be ≈1)
2. **Strong growth:** 745% increase, exponential pattern dominates
3. **Massive autocorrelation:** ACF(1) = 0.989, violating independence
4. **Heteroscedasticity:** Variance ratio (late/early) = 26.19
5. **Probable changepoint:** Regime shift detected at year ≈ 0.3
6. **No data quality issues:** Clean dataset, no missing values or outliers

---

## 1. Data Quality and Structure

### 1.1 Dataset Characteristics
- **Complete data:** No missing values, no duplicates
- **Valid counts:** All values are positive integers [19, 272]
- **Time structure:** Evenly spaced observations (Δt = 0.0855)
- **Sample size:** n = 40 (adequate for GLM, limited for complex models)

### 1.2 Basic Statistics

| Variable | Mean | Median | SD | Min | Max | Range |
|----------|------|--------|-----|-----|-----|-------|
| year | 0.00 | 0.00 | 1.00 | -1.67 | 1.67 | 3.34 |
| C | 109.45 | 74.50 | 86.27 | 19 | 272 | 253 |

**Data Quality Assessment:** ✓ Excellent - Ready for modeling

---

## 2. Distribution Analysis

### 2.1 Count Distribution Properties

**See visualization:** `01_distribution_analysis.png`

The count distribution exhibits:
- **Right skewness:** 0.602 (moderate positive skew)
- **Light tails:** Kurtosis = -1.233 (platykurtic)
- **Bimodal pattern:** Low counts cluster at 20-50, high counts at 180-270
- **No outliers:** Both IQR and z-score methods confirm no extreme values

### 2.2 Overdispersion Analysis

**CRITICAL FINDING: Extreme Overdispersion**

```
Mean:                    109.45
Variance:               7441.74
Variance/Mean Ratio:      67.99
Coefficient of Variation:  0.79
```

**Index of Dispersion Test:**
- Test statistic: 2651.69
- 95% CI (under Poisson): [23.65, 58.12]
- **Result:** Strongly reject Poisson assumption (p << 0.001)

**Interpretation:** The variance is 68 times larger than the mean, indicating massive overdispersion. This is one of the most extreme cases of overdispersion in count data. **Poisson models will severely underestimate uncertainty and provide invalid inference.**

### 2.3 Normality Tests

| Test | Raw Counts (C) | Log(C) |
|------|----------------|--------|
| Shapiro-Wilk | p = 5.37e-05 (reject) | p = 0.003 (reject) |
| K-S Test | p = 0.053 (borderline) | - |

**Conclusion:** Data is neither normal nor log-normal. This is expected and appropriate for count data.

---

## 3. Temporal Trend Analysis

### 3.1 Growth Characteristics

**See visualization:** `02_temporal_analysis.png`

**Overall Growth:**
- Initial count (year = -1.67): 29
- Final count (year = 1.67): 245
- **Growth factor:** 8.45×
- **Percentage increase:** 745%

**Growth Rate (Period-over-Period):**
- Mean: 7.72% per period
- Median: 6.25%
- Range: [-47%, +84%]
- High variability (SD = 21.64%)

### 3.2 Model Comparison

| Model Type | R² | AIC | Key Finding |
|------------|-----|-----|-------------|
| Linear | 0.8846 | 273.2 | Good fit, but residual patterns |
| Exponential | 0.9354* | - | Better fit in log-space |
| Quadratic | 0.9610 | 231.8 | Substantial improvement (ΔAIC = -41) |
| Cubic | 0.9757 | 214.9 | Further improvement |
| Quartic | 0.9900 | 181.2 | Excellent fit, overfitting risk |

*R² on log scale

**Recommendation:** Quadratic or cubic polynomial balances fit quality and complexity. Exponential model provides good theoretical interpretation.

### 3.3 Linear Model Diagnostics

**Model:** C = 81.133 × year + 109.450

**Performance:**
- R² = 0.8846 (88.5% variance explained)
- Slope: 81.13 counts/year [95% CI: 71.81, 90.45]
- Highly significant (p = 2.09e-19)

**Residual Analysis:**
- Mean: 0 (as expected)
- SD: 29.31
- Shapiro-Wilk: p = 0.071 (approximately normal) ✓
- **Durbin-Watson: 0.195** ⚠️ **Strong positive autocorrelation**

**Key Issue:** The residual plot (see `02_temporal_analysis.png`, bottom-left) shows a clear **U-shaped pattern**, indicating systematic departure from linearity. The smoothed trend line reveals:
- Early period: model overestimates
- Middle period: model underestimates
- Late period: model overestimates again

This pattern strongly suggests **nonlinear growth** (quadratic or exponential).

### 3.4 Heteroscedasticity

**Period Comparison:**

| Period | n | Mean | Variance | SD |
|--------|---|------|----------|-----|
| Early (year < 0) | 20 | 37.20 | 157.64 | 12.56 |
| Late (year ≥ 0) | 20 | 181.70 | 4127.91 | 64.25 |

- **Variance ratio:** 26.19 (late/early)
- **F-test:** F = 26.19, p < 0.0001
- **Conclusion:** **Highly significant heteroscedasticity**

The variance structure plot (`02_temporal_analysis.png`, bottom-right) shows both mean and variance increase over time, with variance growing approximately quadratically with mean (slope ≈ 3.15 in mean-variance regression).

**Implication:** Models must account for increasing variance. Negative Binomial naturally handles this through its dispersion parameter.

---

## 4. Autocorrelation and Time Series Structure

### 4.1 Autocorrelation Function (ACF)

**See visualization:** `03_advanced_diagnostics.png`

**Key Findings:**
- **ACF(1) = 0.9886** (extremely high!)
- **ACF(2) = 0.9829**
- **ACF(3) = 0.9713**
- All lags 1-10+ exceed confidence bands

**Interpretation:** This is one of the strongest autocorrelation structures possible. Each observation is almost perfectly predicted by the previous observation. This indicates:
1. **Serial dependence dominates** the data structure
2. Standard regression assumptions are **severely violated**
3. Time series models may be more appropriate than cross-sectional GLMs

### 4.2 Partial Autocorrelation (PACF)

- **PACF(1) = 0.989** (very high)
- PACF(2-20) mostly within confidence bands
- **Pattern suggests AR(1) process**

### 4.3 Stationarity

**Augmented Dickey-Fuller (informal):**
- Original series: Clearly non-stationary (strong trend)
- First differences: Variance reduces 98% (7441 → 170)
- **Conclusion:** Series is **integrated of order 1, I(1)**

**Implication:** The series has a unit root. First differencing achieves stationarity, suggesting ARIMA(p,1,q) models may be appropriate.

### 4.4 Lag-1 Dependence

**Regression: C(t+1) = 1.011 × C(t) + 4.417**
- **R² = 0.9773** (97.7% of variance explained by previous value!)
- Correlation: 0.9886
- Slope ≈ 1 suggests **random walk with small drift**

**Visualization:** The lag-1 scatter plot (`03_advanced_diagnostics.png`, bottom-right) shows an almost perfect linear relationship between consecutive observations.

**Implication:** This near-perfect autoregression suggests that time series methods (ARIMA, state-space models) may be more natural than standard GLMs.

---

## 5. Changepoint Detection

### 5.1 CUSUM Analysis

**See visualization:** `03_advanced_diagnostics.png` (middle-right panel)

The cumulative sum (CUSUM) plot shows a clear U-shaped pattern with minimum at **year ≈ 0.30**.

**Changepoint Analysis:**
- Location: Index 23, year = 0.299, count = 95
- Before changepoint (n=24): Mean = 45.67, SD = 22.52
- After changepoint (n=16): Mean = 205.12, SD = 47.93
- **T-test:** t = -14.18, p < 0.0001 (highly significant)

**Interpretation:** There is strong statistical evidence of a regime shift around year = 0.3, where the process transitions from slow growth (mean ~46) to rapid growth (mean ~205). This represents a **4.5× jump in mean level**.

### 5.2 Alternative Explanation

The "changepoint" could also be interpreted as:
- The inflection point of a smooth exponential/logistic curve
- Natural acceleration in exponential growth
- Not necessarily a discrete structural break

**Recommendation:** Test both interpretations:
1. **Smooth model:** Single exponential or high-order polynomial
2. **Changepoint model:** Two-regime model with break at year ≈ 0.3

---

## 6. Variance Structure

### 6.1 Mean-Variance Relationship

**See visualization:** `03_advanced_diagnostics.png` (bottom-middle panel)

The mean-variance plot on log-log scale reveals:
- Data points lie **well above** the Var = Mean line (Poisson)
- Data points **above** the Var = 2×Mean line
- Approximate relationship: **Var ∝ Mean² to Mean³**

**Regression (on bins):**
- Variance = 3.151 × Mean + 83.186
- R² = 0.349 (weak due to small n of bins)

**Interpretation:** The **quadratic mean-variance relationship** is characteristic of:
1. **Negative Binomial distribution** (Var = μ + μ²/θ)
2. **Gamma-Poisson mixture**
3. Multiplicative error processes

This strongly supports using **Negative Binomial family** for modeling.

### 6.2 Squared Residuals Over Time

The squared residuals plot (`03_advanced_diagnostics.png`, middle-left) shows:
- High variance at extremes (early and late periods)
- Lower variance in middle period
- No simple linear trend in variance

**Implication:** Heteroscedasticity is present but complex. Negative Binomial's inherent variance structure may be sufficient, or consider robust standard errors.

---

## 7. Model Recommendations

### 7.1 Primary Model Classes (Ranked by Suitability)

#### **Recommendation 1: Negative Binomial GLM with Nonlinear Trend** ⭐⭐⭐

**Model Specification:**
```
C ~ NegativeBinomial(μ, θ)
log(μ) = β₀ + β₁×year + β₂×year²
```

**Rationale:**
- ✓ Addresses extreme overdispersion (θ parameter)
- ✓ Flexible mean-variance relationship (Var = μ + μ²/θ)
- ✓ Standard GLM framework (well-established methods)
- ✓ Quadratic trend captures nonlinearity
- ⚠ May need autocorrelation adjustment (GEE, robust SE)

**Variants to Consider:**
- Add year³ term for cubic trend
- Use log(μ) = β₀ + β₁×year (exponential growth)
- Quasi-Poisson as simpler alternative to NB

**Implementation Notes:**
- Estimate θ (dispersion) from data or use priors
- Check residual autocorrelation with ACF of Pearson residuals
- If autocorrelation persists, use GEE with AR(1) correlation structure

#### **Recommendation 2: Negative Binomial with Changepoint** ⭐⭐

**Model Specification:**
```
C ~ NegativeBinomial(μ, θ)
log(μ) = β₀ + β₁×year + β₂×I(year ≥ 0.3) + β₃×year×I(year ≥ 0.3)
```

**Rationale:**
- ✓ Models regime shift explicitly
- ✓ Addresses overdispersion
- ✓ Interpretable parameters (growth before/after break)
- ⚠ Assumes discrete break (may be overly simplistic)

**Alternative:** Use smooth transition function instead of hard break.

#### **Recommendation 3: Time Series Count Model (Advanced)** ⭐⭐

**Model Specification:**
```
C(t) ~ NegativeBinomial(μₜ, θ)
log(μₜ) = α + β×log(C(t-1)) + γ×year
```
or
```
ARIMA(1,1,0) on log(C) or differenced counts
```

**Rationale:**
- ✓ Explicitly models autocorrelation (ACF = 0.989)
- ✓ Natural for time series data
- ✓ Lag-1 R² = 0.977 supports AR(1)
- ⚠ More complex estimation
- ⚠ Requires careful handling of count distribution

**Implementation Notes:**
- Consider integer-valued ARMA (INARMA) models
- Or use state-space model with Poisson/NB observation equation
- May need specialized software (e.g., `tscount` in R, Stan)

#### **Recommendation 4: Generalized Additive Model (GAM)** ⭐

**Model Specification:**
```
C ~ NegativeBinomial(μ, θ)
log(μ) = s(year)  # smooth function
```

**Rationale:**
- ✓ Flexible nonparametric trend
- ✓ No need to specify polynomial degree
- ✓ Addresses overdispersion
- ⚠ Less interpretable than parametric models
- ⚠ Can overfit with small n=40

---

### 7.2 Predictor Variables and Transformations

**Primary Predictor:**
- **year** (standardized time) - Use as-is

**Nonlinear Terms to Test:**
- **year²** (quadratic) - RECOMMENDED based on polynomial comparison
- **year³** (cubic) - Consider if quadratic insufficient
- **I(year ≥ 0.3)** (changepoint indicator) - Test regime shift hypothesis

**Lagged Variables:**
- **C(t-1)** (lagged count) - Strong candidate given ACF structure
- **log(C(t-1))** - If using log-linear specification

**DO NOT:**
- ✗ Log-transform outcome (loses count structure)
- ✗ Use year⁴ or higher (overfitting risk with n=40)
- ✗ Standardize counts (interpretation issues)

---

### 7.3 Distribution Family Selection

| Family | Suitability | Rationale |
|--------|-------------|-----------|
| **Negative Binomial** | ⭐⭐⭐ Excellent | Handles overdispersion (Var/Mean = 68), flexible variance |
| **Quasi-Poisson** | ⭐⭐ Good | Simpler than NB, handles overdispersion via scale parameter |
| Poisson | ✗ Inappropriate | Var/Mean must equal 1; violated by factor of 68 |
| Zero-Inflated | ✗ Not needed | No zeros in data |
| Hurdle | ✗ Not needed | No zeros in data |
| Normal | ✗ Inappropriate | Count data, discrete values |

**Recommendation:** Start with **Negative Binomial**. If computational issues arise, fall back to **Quasi-Poisson**.

---

### 7.4 Handling Autocorrelation

The strong autocorrelation (ACF = 0.989, D-W = 0.195) requires special treatment:

**Option 1: GEE (Generalized Estimating Equations)**
- Specify NB family with AR(1) working correlation
- Provides robust standard errors
- Does not model correlation explicitly, treats as nuisance

**Option 2: Include Lagged Dependent Variable**
- Add C(t-1) as predictor: log(μₜ) = β₀ + β₁×C(t-1) + β₂×year
- Addresses autocorrelation directly
- Caution: Dynamic model, changes interpretation

**Option 3: Time Series Approach**
- Use ARIMA or state-space framework
- Most appropriate given I(1) structure
- More complex but most rigorous

**Option 4: Ignore (NOT RECOMMENDED)**
- Standard errors will be too small
- Hypothesis tests will be anti-conservative
- Confidence intervals too narrow

**Recommendation:** At minimum, use **robust/sandwich standard errors**. Ideally, use **GEE with AR(1)** or **include lagged counts**.

---

### 7.5 Model Comparison Criteria

**Goodness of Fit:**
- **AIC/BIC** for nested model comparison
- **Deviance** (for GLMs)
- **Posterior predictive checks** (if Bayesian)

**Residual Diagnostics:**
- ACF of Pearson/deviance residuals (should be white noise)
- Randomized quantile residuals (uniform if model correct)
- Residuals vs fitted (check for patterns)

**Predictive Performance:**
- **Out-of-sample prediction** (leave-last-k-out)
- **RMSE, MAE** on held-out data
- **Coverage of prediction intervals**

**Parsimony:**
- Prefer simpler models if fit is comparable
- AIC penalizes complexity (2k)
- BIC penalizes more strongly (k×log(n))

---

## 8. Critical Modeling Assumptions

### 8.1 Assumptions That WILL Be Violated:

1. **Independence** ✗ - ACF = 0.989, severe violation
2. **Constant variance** ✗ - Variance ratio = 26×
3. **Poisson equidispersion** ✗ - Var/Mean = 68

### 8.2 Assumptions That Appear Valid:

1. **Count data** ✓ - All positive integers
2. **No zeros** ✓ - Simplifies modeling
3. **No outliers** ✓ - Data quality is good
4. **Monotonic trend** ✓ - Consistent growth direction

### 8.3 Required Adjustments:

- **Must** use overdispersion-robust family (NB or quasi-Poisson)
- **Must** address autocorrelation (GEE, lagged DV, or robust SE)
- **Should** test nonlinear trend (quadratic or exponential)
- **Consider** changepoint or regime-switching model

---

## 9. Practical Considerations

### 9.1 Sample Size
- **n = 40** is adequate for:
  - Simple GLMs (2-4 parameters)
  - Linear or quadratic trends
  - Negative Binomial estimation
- **n = 40** may be insufficient for:
  - Complex polynomials (degree > 3)
  - Many changepoints
  - Complex ARIMA models (p,d,q all > 1)

**Recommendation:** Keep models relatively simple (≤ 5 parameters).

### 9.2 Extrapolation Risk
- Strong growth trend invites extrapolation
- **Caution:** Exponential models can explode beyond data range
- **Recommendation:** Flag predictions outside [-1.67, 1.67] as uncertain

### 9.3 Computational Considerations
- Negative Binomial MLE can be unstable if θ very small
- Bayesian approach with priors on θ may be more stable
- GEE requires iterative fitting, generally robust

---

## 10. Summary and Next Steps

### 10.1 Key Takeaways

1. **Data quality is excellent** - No preprocessing needed
2. **Overdispersion is extreme** - NB or quasi-Poisson mandatory
3. **Growth is strong and nonlinear** - Exponential or polynomial trend
4. **Autocorrelation is severe** - Must be addressed in modeling
5. **Variance increases over time** - Heteroscedasticity present
6. **Possible changepoint** - Worth testing explicitly
7. **Standard Poisson GLM will fail** - Guaranteed to underfit

### 10.2 Recommended Modeling Workflow

**Stage 1: Baseline Models**
1. Fit NB-GLM with linear trend (benchmark)
2. Fit NB-GLM with quadratic trend
3. Fit NB-GLM with exponential trend (log-linear)
4. Compare AIC/BIC

**Stage 2: Address Autocorrelation**
5. Compute ACF of residuals from best Stage 1 model
6. If ACF significant, refit with GEE-AR(1) or robust SE
7. Or add C(t-1) as predictor

**Stage 3: Test Changepoint**
8. Fit piecewise model with break at year = 0.3
9. Compare to smooth model via AIC/likelihood ratio test

**Stage 4: Validation**
10. Residual diagnostics (ACF, Q-Q plots, fitted vs residuals)
11. Out-of-sample prediction (leave-last-5-out)
12. Sensitivity analysis on θ (dispersion parameter)

**Stage 5: Final Model Selection**
13. Choose model balancing fit, parsimony, and residual behavior
14. Report with uncertainty quantification

### 10.3 Expected Outcomes

**Best-case scenario:**
- NB-GLM with quadratic/exponential trend
- Overdispersion parameter θ ≈ 5-20
- Residual ACF reduced to < 0.3
- R² ≈ 0.95-0.98
- Adequate prediction intervals

**Challenges to anticipate:**
- Residual autocorrelation may persist (need GEE or lagged DV)
- Changepoint may be ambiguous (smooth vs discrete)
- Small sample size limits model complexity
- Extrapolation uncertainty will be large

---

## 11. Visualizations Reference

All visualizations saved to `/workspace/eda/visualizations/`:

### `01_distribution_analysis.png` (Multi-panel)
Shows distribution properties of count variable:
- **Histogram with KDE:** Right-skewed, bimodal pattern
- **Q-Q Plot:** S-curve indicates non-normality
- **Box Plot:** No outliers, wide IQR
- **ECDF:** Quartile reference lines
- **Log-scale Histogram:** Also bimodal
- **Count Frequency:** 33 unique values, high diversity

**Key Insight:** Distribution is neither normal nor log-normal, confirming need for count models.

### `02_temporal_analysis.png` (Multi-panel)
Shows temporal trends and model comparisons:
- **Scatter with Linear Trend:** Strong positive relationship (R² = 0.885)
- **Polynomial Comparison:** Higher degrees fit better, especially quadratic
- **Log-scale Plot:** Tests exponential growth (R² = 0.935 in log-space)
- **Residual Plot:** U-shaped pattern indicates nonlinearity
- **Q-Q of Residuals:** Light-tailed but approximately normal
- **Variance Structure:** Both mean and variance increase over time

**Key Insight:** Nonlinear trend (exponential or quadratic) fits better than linear. Heteroscedasticity confirmed.

### `03_advanced_diagnostics.png` (Multi-panel)
Shows time series structure and advanced diagnostics:
- **ACF Plot:** All lags 1-10+ significant, strong autocorrelation
- **PACF Plot:** Lag-1 dominant, suggests AR(1)
- **First Differences:** Mean ≈ 5.5, more stable than levels
- **Squared Residuals:** Variance heterogeneity over time
- **Rolling Statistics:** Exponential increase in both mean and SD
- **CUSUM:** Changepoint detected at year ≈ 0.3
- **Growth Rate:** Variable but median ≈ 6%, no clear trend
- **Mean-Variance Plot:** Quadratic relationship, well above Poisson line
- **Lag-1 Scatter:** Nearly perfect linear relationship (R² = 0.977)

**Key Insight:** Time series structure dominates. Autocorrelation and changepoint must be modeled.

---

## 12. Technical Details

### 12.1 Software Used
- Python 3.x
- Libraries: pandas, numpy, scipy, matplotlib, seaborn

### 12.2 Statistical Tests Performed
- Shapiro-Wilk test (normality)
- Kolmogorov-Smirnov test (distribution fit)
- Index of Dispersion test (Poisson assumption)
- F-test (equality of variances)
- T-test (equality of means before/after changepoint)
- Durbin-Watson test (autocorrelation)
- Linear regression (multiple specifications)

### 12.3 Reproducibility
All analysis code saved to `/workspace/eda/code/`:
- `01_initial_exploration.py` - Data loading and descriptive statistics
- `02_distribution_analysis.py` - Distribution tests and overdispersion analysis
- `03_temporal_analysis.py` - Trend fitting and heteroscedasticity tests
- `04_advanced_diagnostics.py` - ACF, PACF, changepoint detection

---

## Appendix: Quick Reference

### Model Recommendation Summary

| Model | Primary Use | Pros | Cons | Priority |
|-------|-------------|------|------|----------|
| NB-GLM (quadratic) | Main model | Handles overdispersion, good fit | Autocorrelation issue | ⭐⭐⭐ |
| NB-GLM (exponential) | Interpretability | Clear growth rate, good fit | Autocorrelation issue | ⭐⭐⭐ |
| NB-GLM (changepoint) | Test regime shift | Tests hypothesis | May be overly discrete | ⭐⭐ |
| GEE-NB (AR1) | If ACF persists | Handles autocorrelation | More complex | ⭐⭐ |
| ARIMA-like | Time series focus | Natural for TS data | Complex, specialized | ⭐⭐ |

### Critical Parameters to Estimate

1. **θ (NB dispersion):** Expected range [5, 50]
2. **β₁ (linear trend):** ≈ 81 counts/year (0.85 in log-space)
3. **β₂ (quadratic term):** Test if non-zero
4. **Changepoint location:** Test if year ≈ 0.3 is significant
5. **AR(1) coefficient:** If using time series model, expect ρ ≈ 0.99

---

**Report prepared by:** EDA Specialist
**For questions on methodology, see:** `/workspace/eda/eda_log.md`
