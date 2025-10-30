# EDA Findings Report - Analyst 3
## Model Assumptions & Diagnostic Preparation

**Analyst**: EDA Analyst 3
**Focus**: Model assumption exploration, residual diagnostics, count model suitability
**Date**: 2025-10-29
**Dataset**: `/workspace/data/data_analyst_3.csv` (40 observations)

---

## Executive Summary

This analysis systematically evaluated model assumptions for count data through residual diagnostics, heteroscedasticity tests, and distributional checks. **Key finding: The data exhibits severe overdispersion (variance/mean ratio = 70.4) and strong temporal autocorrelation (r = 0.97), ruling out standard Poisson regression and necessitating more sophisticated modeling approaches.**

---

## 1. Data Quality Assessment

### Sample Characteristics
- **Sample size**: 40 observations
- **Variables**: `year` (standardized, range: [-1.67, 1.67]), `C` (count variable, range: [21, 269])
- **Missing values**: 0 (0%)
- **Data integrity**: All counts are positive integers, uniformly spaced time points
- **Duplicates**: 0

### Quality Verdict
**EXCELLENT** - No data quality issues detected. Data ready for modeling.

---

## 2. Distributional Assumptions for Count Models

### 2.1 Poisson Assumption (Equidispersion)

**Test Results:**
- **Mean (λ)**: 109.40
- **Variance**: 7704.66
- **Dispersion Ratio (Var/Mean)**: **70.43**
- **Formal test statistic**: 2746.63 (χ²₃₉)
- **p-value**: < 0.001

**Interpretation:**
The variance is **70 times larger** than the mean, indicating **severe overdispersion**. The formal chi-square test overwhelmingly rejects the Poisson assumption (p < 0.001).

**Conclusion**: ❌ **Poisson regression is NOT appropriate** for this data.

**Visual Evidence**: See `dispersion_analysis.png` - the variance-mean plot shows observed variance far above the Poisson reference line (var = mean).

### 2.2 Zero-Inflation

- **Observed zeros**: 0
- **Expected zeros (Poisson)**: 0.00
- **Conclusion**: ✓ No zero-inflation present

### 2.3 Distribution Shape

| Metric | Observed | Poisson Expected | Interpretation |
|--------|----------|------------------|----------------|
| Skewness | 0.64 | 0.10 | More right-skewed than Poisson |
| Excess Kurtosis | -1.13 | 0.01 | Flatter tails (platykurtic) |
| Coefficient of Variation | 0.80 | 0.10 | Much higher variability |

**Visual Evidence**: See `dispersion_analysis.png` (top-left panel) - observed distribution is much wider and more variable than theoretical Poisson.

---

## 3. Residual Diagnostics from Simple Models

Three models were tested:
1. **Linear**: C ~ year
2. **Log**: log(C) ~ year
3. **Sqrt**: sqrt(C) ~ year

### 3.1 Model Performance Comparison

| Model | R² | BP Test p-value | SW Test p-value | Heteroscedastic? | Residuals Normal? |
|-------|-----|-----------------|-----------------|------------------|-------------------|
| Linear | 0.881 | 0.332 | 0.112 | ✓ No | ✓ Yes |
| Log | **0.937** | **0.003** | 0.796 | ❌ **Yes** | ✓ Yes |
| Sqrt | 0.924 | 0.015 | 0.325 | ❌ **Yes** | ✓ Yes |

**Key Findings:**

1. **Linear Model** (C ~ year):
   - Best residual properties (homoscedastic, normal)
   - Lower R² (88.1%)
   - Slope: 82.4 (count increases ~82 per standard deviation of year)
   - **Visual Evidence**: `residual_diagnostics_all_models.png` (top row) - residuals evenly scattered around zero

2. **Log Model** (log(C) ~ year):
   - **Highest R²** (93.7%)
   - **Significant heteroscedasticity** (BP p-value = 0.003)
   - Residuals normal but variance pattern problematic
   - **Visual Evidence**: `residual_diagnostics_all_models.png` (middle row) - residuals vs fitted shows funnel pattern

3. **Sqrt Model** (sqrt(C) ~ year):
   - Good R² (92.4%)
   - **Mild heteroscedasticity** (BP p-value = 0.015)
   - Compromise between linear and log
   - **Visual Evidence**: `residual_diagnostics_all_models.png` (bottom row)

### 3.2 Heteroscedasticity Patterns

The **Breusch-Pagan test** reveals:
- Linear model: No heteroscedasticity detected (p = 0.332)
- Log model: **Strong heteroscedasticity** (p = 0.003)
- Sqrt model: **Moderate heteroscedasticity** (p = 0.015)

**Interpretation**: While transformations improve fit (R²), they introduce variance instability. This suggests the data generation process has mean-dependent variance, typical of count processes.

**Visual Evidence**: See `variance_mean_relationship.png`:
- Left panel: Variance increases much faster than mean (not Poisson-like)
- Right panel: Squared residuals increase with fitted values (scale-location plot)

---

## 4. Variance-Mean Relationship

### 4.1 Sliding Window Analysis

Using a sliding window of 10 observations:
- Variance consistently exceeds mean across all levels
- Relationship appears roughly quadratic (var ∝ mean²)
- Not consistent with Poisson (var = mean) or even simple overdispersion

**Visual Evidence**: `variance_mean_relationship.png` (left panel) - points lie far above the red Poisson reference line.

### 4.2 Temporal Variance Patterns

Quartile-wise analysis reveals **heterogeneous variance structure**:

| Quartile | Mean | Variance | Var/Mean Ratio | Time Period |
|----------|------|----------|----------------|-------------|
| Q1 | 27.5 | 16.1 | 0.58 | Early (low counts) |
| Q2 | 45.7 | 206.2 | 4.51 | Medium counts |
| Q3 | 126.6 | 1500.3 | **11.85** | High counts |
| Q4 | 237.8 | 1055.7 | 4.44 | Late (highest counts) |

- **Levene's test**: p = 0.010 → **Significant variance heterogeneity**
- Variance is **not stable over time** or across count levels
- Q3 shows particularly high variability

**Visual Evidence**: `dispersion_analysis.png` (bottom-left panel) - boxplots show increasing spread with annotations of mean and variance.

---

## 5. Temporal Autocorrelation

**Critical Finding:**
- **Lag-1 autocorrelation**: **0.971** (very strong positive)
- **Durbin-Watson statistic**: 0.47 (indicates strong positive autocorrelation)

**Interpretation**: Consecutive observations are highly correlated. This has major implications:
1. Standard errors will be underestimated if autocorrelation ignored
2. Simple regression assumptions violated (independence of errors)
3. Time series methods or correlated error structures needed

**Visual Evidence**: `dispersion_analysis.png` (bottom-right panel) - smooth trend with observations tightly following fitted line, characteristic of autocorrelated data.

---

## 6. Transformation Effects

### 6.1 Comparative Analysis

**Visual Evidence**: `transformation_comparison.png`

- **Original scale**: Clear exponential growth pattern
- **Log transformation**: Linearizes relationship (highest R²) but creates heteroscedasticity
- **Square root transformation**: Partial linearization, compromise solution

### 6.2 Implications for Modeling

1. **If using GLM**: Use log link function (not log transformation)
2. **If using linear model**: Consider robust standard errors due to heteroscedasticity
3. **If using transformation**: Square root is better than log for variance stability

**Metrics comparison bar chart** (`transformation_comparison.png`, bottom-right) shows:
- Log model: Highest R² but fails heteroscedasticity test (BP p-value < 0.05)
- Linear model: Passes all tests but lowest R²
- Sqrt model: Middle ground

---

## 7. Influential Observations

### Cook's Distance Analysis

**Visual Evidence**: `influence_diagnostics.png`

- No observations exceed the Cook's distance threshold (4/n = 0.1)
- Maximum Cook's D ≈ 0.06 (observation at high end of time series)
- **Conclusion**: ✓ No highly influential outliers

### Leverage Analysis

- All observations have leverage < 2p/n = 0.1 (threshold for concern)
- Extreme year values have slightly higher leverage (expected)
- **Conclusion**: ✓ No high-leverage outliers distorting the fit

---

## 8. Model Recommendations

Based on comprehensive diagnostics, **ranked by priority**:

### 8.1 RECOMMENDED: Negative Binomial Regression ⭐⭐⭐

**Why:**
- Naturally handles overdispersion (var = μ + αμ²)
- Dispersion parameter (α) accounts for extra-Poisson variation
- Well-suited for count data with high variance
- Can incorporate log link: log(μ) = β₀ + β₁×year

**Evidence:**
- Dispersion ratio = 70.4 >> 1
- Variance increases faster than mean
- Formal overdispersion test: p < 0.001

**Implementation note**: Consider including autocorrelation structure (AR(1)) given lag-1 correlation of 0.97.

### 8.2 ALTERNATIVE: Quasi-Poisson Regression ⭐⭐

**Why:**
- Adjusts standard errors for overdispersion
- Simpler than Negative Binomial (no distributional assumptions)
- Good for prediction

**Limitations:**
- No likelihood-based inference (can't use AIC/BIC)
- Doesn't model dispersion mechanism explicitly

### 8.3 ALTERNATIVE: GLM with Robust Standard Errors ⭐

**Why:**
- Flexible family choice (Gaussian with log link)
- Robust/sandwich errors handle heteroscedasticity
- Can incorporate correlated errors

**Limitations:**
- Doesn't respect count nature of data
- Predictions can be non-integer

### 8.4 NOT RECOMMENDED: Standard Poisson Regression ❌

**Why not:**
- Equidispersion assumption grossly violated (ratio = 70.4)
- Would produce severely underestimated standard errors
- Confidence intervals would be misleadingly narrow
- Hypothesis tests would have inflated Type I error

### 8.5 NOT RECOMMENDED: Simple Linear Regression ❌

**Why not:**
- Residuals are well-behaved BUT...
- Predictions can be negative (count data should be ≥ 0)
- Doesn't respect count data generation process
- Constant variance assumption questionable at extremes

---

## 9. Additional Modeling Considerations

### 9.1 Autocorrelation Structure

Given **lag-1 correlation of 0.97**, consider:
- Generalized Estimating Equations (GEE) with AR(1) correlation
- Generalized Linear Mixed Models (GLMM) with random effects
- Time series count models (ARMA-count processes)

### 9.2 Link Function Choice

For GLM/Negative Binomial:
- **Log link**: log(μ) = β₀ + β₁×year
  - Ensures positive predictions
  - Interpretable as multiplicative effects (exp(β₁) = rate ratio)
  - Natural for count data

### 9.3 Variance Function

The data suggest variance grows faster than linear in mean:
- **Poisson**: var = μ (not appropriate)
- **Negative Binomial**: var = μ + αμ² (good fit)
- **Quasi-Poisson**: var = φμ (acceptable)

---

## 10. Key Statistical Properties Summary

| Property | Value | Interpretation |
|----------|-------|----------------|
| Sample size | 40 | Adequate for simple models |
| Mean count | 109.4 | Moderate count level |
| Variance | 7704.7 | Very high variability |
| Dispersion ratio | **70.4** | **Severe overdispersion** |
| Autocorrelation (lag-1) | **0.97** | **Very strong temporal dependence** |
| R² (linear model) | 0.88 | Strong trend |
| R² (log model) | 0.94 | Excellent fit on log scale |
| Heteroscedasticity (log) | p = 0.003 | Significant |
| Levene's test | p = 0.010 | Heterogeneous variances |

---

## 11. Conclusions

### What We Learned

1. **Count Model Selection**: Standard Poisson is inappropriate; Negative Binomial is strongly preferred
2. **Variance Structure**: Heteroscedastic, mean-dependent, and time-varying
3. **Temporal Dependence**: Extremely strong autocorrelation requires special handling
4. **Transformation Trade-offs**: Log transformation improves fit but worsens variance stability
5. **Data Quality**: Excellent - no missing values, outliers, or data integrity issues

### Robust Findings (high confidence)

✓ Overdispersion is present and severe (multiple independent tests)
✓ Temporal autocorrelation is strong (lag-1 = 0.97)
✓ Exponential growth pattern in counts over time
✓ No influential outliers or data quality issues
✓ Linear trend on log scale is excellent (R² = 0.937)

### Tentative Findings (lower confidence)

⚠ Exact form of variance-mean relationship (need model-based estimates)
⚠ Stationarity assumptions (only 40 observations)
⚠ Presence of change points or regime shifts (would need more detailed analysis)

---

## 12. Next Steps for Modeling

### Immediate Actions

1. **Fit Negative Binomial model** with log link: log(μ) = β₀ + β₁×year
2. **Check dispersion parameter** (α) estimate and significance
3. **Examine model residuals** using randomized quantile residuals
4. **Compare AIC/BIC** to alternative models (Quasi-Poisson, Zero-Inflated if relevant)

### Advanced Considerations

1. **Model autocorrelation**:
   - GEE with AR(1) working correlation
   - GLMM with temporal random effects
2. **Test for structural breaks**:
   - Check if growth rate changes over time
   - Consider piecewise regression
3. **Prediction intervals**:
   - Account for both parameter uncertainty AND overdispersion
   - Bootstrap methods recommended

### Model Validation

1. **Cross-validation**: Leave-one-out or k-fold (given small sample)
2. **Residual diagnostics**: Randomized quantile residuals for count models
3. **Goodness-of-fit**: Deviance test, Pearson chi-square
4. **Sensitivity analysis**: Impact of removing extreme observations

---

## 13. Files Generated

### Visualizations (all 300 dpi)

1. **`residual_diagnostics_all_models.png`**: 3×3 panel comparing residual plots for linear, log, and sqrt models
   - Shows residuals vs fitted, residuals vs year, and Q-Q plots for each transformation
   - Key insight: Linear model has best residual behavior; log model shows heteroscedasticity

2. **`variance_mean_relationship.png`**: Variance-mean relationship and scale-location plot
   - Left: Sliding window variance vs mean with Poisson reference line
   - Right: Squared residuals vs fitted values with smoothed trend
   - Key insight: Variance far exceeds mean; not Poisson-distributed

3. **`dispersion_analysis.png`**: 2×2 panel on distribution and dispersion
   - Observed distribution, comparison to Poisson, temporal boxplots, time series with bands
   - Key insight: Distribution much wider than Poisson; variance changes over time

4. **`transformation_comparison.png`**: 2×2 panel comparing transformations
   - Original, log, and sqrt scales with fitted lines; model metrics comparison
   - Key insight: Log transformation gives best fit but worst heteroscedasticity

5. **`influence_diagnostics.png`**: Cook's distance and leverage plots
   - Identifies potential influential observations
   - Key insight: No problematic outliers detected

### Code Scripts

1. **`01_initial_exploration.py`**: Data loading, quality checks, basic statistics
2. **`02_residual_diagnostics.py`**: Three model fits, residual tests, heteroscedasticity tests
3. **`03_visualizations.py`**: All diagnostic plots generation
4. **`04_count_model_tests.py`**: Detailed Poisson assumption testing, dispersion checks

### Documentation

1. **`eda_log.md`**: Detailed exploration process with intermediate findings
2. **`findings.md`**: This comprehensive report

---

## Appendix: Statistical Test Summary

| Test | Purpose | Result | p-value | Conclusion |
|------|---------|--------|---------|------------|
| Equidispersion (χ²) | Poisson suitability | 2746.63 | < 0.001 | Reject Poisson |
| Breusch-Pagan (Linear) | Heteroscedasticity | 0.94 | 0.332 | Homoscedastic ✓ |
| Breusch-Pagan (Log) | Heteroscedasticity | 8.91 | 0.003 | Heteroscedastic ❌ |
| Breusch-Pagan (Sqrt) | Heteroscedasticity | 5.94 | 0.015 | Heteroscedastic ❌ |
| Shapiro-Wilk (Linear) | Normality | 0.955 | 0.112 | Normal ✓ |
| Shapiro-Wilk (Log) | Normality | 0.983 | 0.796 | Normal ✓ |
| Shapiro-Wilk (Sqrt) | Normality | 0.969 | 0.325 | Normal ✓ |
| Levene's Test | Variance equality | 4.39 | 0.010 | Heterogeneous ❌ |
| Durbin-Watson | Autocorrelation | 0.47 | - | Strong positive AC |

---

**Report prepared by**: EDA Analyst 3
**All outputs located in**: `/workspace/eda/analyst_3/`
