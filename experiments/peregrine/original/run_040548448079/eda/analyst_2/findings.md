# Distributional Analysis and Variance Structure
**EDA Analyst 2 Report**

---

## Executive Summary

This analysis examined the distributional properties and variance structure of 40 time-ordered count observations. The primary finding is **extreme overdispersion** (variance/mean ratio = 68) that definitively rules out the Poisson distribution. A **Negative Binomial distribution** provides dramatically better fit (ΔAIC = -2417), with a dispersion parameter of r ≈ 1.6. Additionally, significant **heteroscedasticity** was detected (p < 0.0001), with dispersion varying 6-fold across time periods. No data quality issues or outliers were identified.

---

## 1. Distributional Family Recommendation

### PRIMARY RECOMMENDATION: Negative Binomial Distribution

#### Evidence:

**Quantitative Metrics:**
- **Variance/Mean Ratio**: 67.99 (Poisson expectation: 1.0)
- **Log-likelihood**: -225.76 (NB) vs -1435.07 (Poisson)
- **AIC**: 455.51 (NB) vs 2872.13 (Poisson) → **ΔAIC = -2416.62**
- **BIC**: 458.89 (NB) vs 2873.82 (Poisson) → **ΔBIC = -2414.93**

**Interpretation:**
- Negative Binomial improves log-likelihood by **1209 units**
- AIC difference of -2417 represents **overwhelming evidence** for NB
  - Rule of thumb: ΔAIC > 10 is "essentially no support" for worse model
  - Our ΔAIC is >200× this threshold
- Conclusion is robust across all model selection criteria

**Parameter Estimates (Method of Moments):**
- **r (dispersion parameter)**: 1.634
- **p (probability parameter)**: 0.015
- **Alternative parameterization**: α = 1/r = 0.612
  - Smaller r indicates MORE overdispersion
  - r < 5 is considered substantial overdispersion

**Visual Evidence:**
- `distribution_fitting.png` (Figure 1): Shows Q-Q plots
  - Poisson Q-Q plot: Systematic S-shaped deviation
  - NB Q-Q plot: Points closely follow diagonal line
- `distribution_overview.png`: Histogram shape consistent with NB

### Why Poisson Fails:

The Poisson distribution assumes **variance = mean**, which is profoundly violated:
- Observed variance: 7441.74
- Poisson prediction: 109.45
- Actual variance is **68× larger** than Poisson predicts

This is not a borderline case. The overdispersion is so extreme that Poisson model:
- Severely underestimates uncertainty
- Produces invalid confidence intervals
- Yields biased hypothesis tests
- Is fundamentally misspecified

### Alternative Distributions Considered:

**Zero-Inflated Models**: NOT appropriate
- Zero counts: 0 (0.0% of data)
- Minimum count: 19
- No evidence of zero-inflation

**Quasi-Poisson**: Could work but suboptimal
- Pros: Simple, robust SEs
- Cons: No likelihood, less efficient than NB
- Recommendation: Use NB instead

**Generalized Poisson**: Potential alternative
- Can handle overdispersion
- Less common, harder to interpret
- Recommendation: NB is more standard

---

## 2. Variance Structure Analysis

### 2.1 Variance-Mean Relationship

**Key Finding**: Variance scales as **Mean^1.67** (not linearly)

**Power Law Model:**
```
Variance = a × Mean^b
```

**Fitted Parameters:**
- **Exponent (b)**: 1.667 (95% CI excludes 1.0)
- **R-squared**: 0.814
- **p-value**: 3.99 × 10^-12

**Interpretation:**
- b = 1 would indicate Poisson (variance ∝ mean)
- b = 2 would indicate variance ∝ mean²
- Observed b = 1.67 suggests intermediate scaling
- Variance grows **faster than mean**

**Implications:**
1. Variance structure is more complex than simple Poisson
2. NB2 parameterization may be more appropriate than NB1
   - NB1: Var = μ + α×μ (linear in mean)
   - NB2: Var = μ + α×μ² (quadratic in mean)
3. Dispersion parameter may vary with mean level

**Visual Evidence:**
- `variance_mean_analysis.png` (Figure 2):
  - Panel 1: Scatter of variance vs mean with theoretical lines
  - Panel 2: Log-log plot clearly shows slope ≠ 1
  - Both demonstrate non-linear relationship

### 2.2 Temporal Heteroscedasticity

**Key Finding**: Variance structure **changes significantly over time**

**Breusch-Pagan Test:**
- **Test statistic**: 28.28
- **p-value**: <0.0001
- **Conclusion**: Strong evidence of heteroscedasticity

**Period-Specific Dispersion:**

| Period | Time Range | Mean | Variance | Var/Mean | CV |
|--------|-----------|------|----------|----------|-----|
| 1 | Early (-1.67 to -1.07) | 27.0 | 49.7 | 1.84 | 0.261 |
| 2 | Early-Mid (-0.98 to -0.38) | 38.6 | 49.1 | **1.27** | 0.181 |
| 3 | Middle (-0.30 to 0.30) | 71.4 | 357.4 | **5.01** | 0.265 |
| 4 | Mid-Late (0.38 to 0.98) | 167.1 | 1256.1 | **7.52** | 0.212 |
| 5 | Late (1.07 to 1.67) | 243.1 | 366.7 | 1.51 | 0.079 |

**Observations:**
1. Dispersion varies **6-fold** (1.27 to 7.52)
2. Pattern is **non-monotonic**:
   - Low dispersion early and late
   - High dispersion in middle periods
   - Suggests regime changes
3. Coefficient of Variation more stable (0.08 to 0.27)
   - Relative variability decreases as mean increases
4. All periods show overdispersion (all Var/Mean > 1)

**Implications:**
1. **Time-varying dispersion** should be considered
2. Single dispersion parameter may be inadequate
3. May indicate:
   - Changing data generation process
   - Unmeasured time-varying factors
   - Different regimes in underlying process

**Visual Evidence:**
- `temporal_periods_comparison.png` (Figure 3): 6-panel comparison
  - Shows clear variation in dispersion across periods
  - Mean-variance scatter shows non-constant relationship
- `temporal_dispersion_rolling.png` (Figure 4): Rolling window analysis
  - Confirms time-varying dispersion at multiple scales

### 2.3 Implications for Modeling

**Standard NB Model:**
```
C ~ NegBinomial(μ, r)
log(μ) = β₀ + β₁ × year
```
- Assumes constant dispersion parameter r
- May underfit given heteroscedasticity

**Time-Varying Dispersion Model:**
```
C ~ NegBinomial(μ, r(t))
log(μ) = β₀ + β₁ × year
log(r) = γ₀ + γ₁ × year  (or f(year))
```
- Allows dispersion to change over time
- More flexible but more complex
- May improve fit in middle periods

**Recommendation:**
1. **Start with**: Standard NB (constant r)
2. **Assess fit**: Check if residuals show patterns
3. **If needed**: Consider time-varying r
4. **Trade-off**: Balance fit improvement vs complexity

---

## 3. Overdispersion Quantification

### 3.1 Index of Dispersion

**Index of Dispersion (ID)** = Variance / Mean = **67.99**

**Interpretation:**
- ID = 1: Poisson distribution (equidispersion)
- ID > 1: Overdispersion
- ID < 1: Underdispersion

**Our ID = 68** represents:
- **Extreme overdispersion**
- Among the highest seen in typical count data
- Suggests strong unobserved heterogeneity

### 3.2 Dispersion Parameter (α)

**α = 1/r = 0.612**

**Interpretation:**
- α = 0: Reduces to Poisson
- α > 0: Overdispersion increases with α
- Larger α = more extra-Poisson variation

**Benchmarks:**
- α < 0.1: Mild overdispersion
- α = 0.1-0.5: Moderate overdispersion
- α > 0.5: **Strong overdispersion** ← Our case
- α > 1: Extreme overdispersion

**Our α = 0.61** indicates:
- Strong but not extreme overdispersion
- Substantial unmodeled variation
- NB is clearly superior to Poisson

### 3.3 Coefficient of Variation

**Overall CV** = SD / Mean = 86.27 / 109.45 = **0.788**

**Interpretation:**
- Poisson: CV = 1/√mean = 1/√109.45 = 0.096
- Observed: CV = 0.788
- Ratio: 0.788 / 0.096 = **8.2× higher than Poisson**

**Temporal Pattern:**
- CV ranges from 0.079 to 0.265 across periods
- Decreases over time (as mean increases)
- More stable than raw variance

---

## 4. Outlier Analysis

### 4.1 Multiple Detection Methods

**Methods Applied:**
1. **Z-score** (|z| > 2): 0 outliers
2. **IQR method** (1.5×IQR rule): 0 outliers
3. **MAD-based** (robust, |mod z| > 3.5): 0 outliers
4. **Trend-adjusted** (|std resid| > 2): 0 outliers
5. **Cook's Distance** (D > 4/n): 3 influential points

**Influential Points** (high leverage, not aberrant):
- **Obs 1**: C=29, Cook's D=0.187 (early timepoint)
- **Obs 2**: C=36, Cook's D=0.172 (early timepoint)
- **Obs 36**: C=272, Cook's D=0.133 (peak value)

### 4.2 Interpretation

**Key Conclusions:**
1. **No statistical outliers** detected
2. Influential points are at **temporal extremes**
   - High leverage due to position (start/peak)
   - NOT aberrant values
3. All observations appear **legitimate**
4. Overdispersion is **NOT driven by outliers**
   - Genuine feature of data
   - Not artifact of extreme values

**Implications:**
- Can use **full dataset** (n=40)
- No need for outlier removal
- No need for robust methods specifically for outliers
- Overdispersion reflects true underlying process

**Visual Evidence:**
- `outlier_analysis.png` (Figure 5): 6-panel diagnostic
  - Box plot: No points beyond whiskers
  - Z-scores: All well within ±2 SD
  - Cook's D: Only 3 points exceed threshold, all reasonable

### 4.3 Data Quality Assessment

**Strengths:**
- ✓ No missing values (0/40)
- ✓ No outliers (0/40)
- ✓ No apparent data entry errors
- ✓ Smooth temporal progression
- ✓ Sensible value range (19-272)

**Limitations:**
- Small sample size (n=40) limits complex models
- No replicate observations at same time
- Single count variable (no covariates in this dataset)

**Overall Quality**: **EXCELLENT** - ready for modeling

---

## 5. Distribution Shape

### 5.1 Summary Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Mean** | 109.45 | Central tendency |
| **Median** | 74.50 | Slightly below mean (right skew) |
| **Skewness** | 0.602 | Moderate right skew |
| **Kurtosis** | -1.233 | Platykurtic (flatter than normal) |
| **Range** | [19, 272] | 253-unit span |
| **IQR** | 160.75 | Substantial spread |

### 5.2 Shape Characteristics

**Right-Skewed Distribution:**
- Mean > Median (109.45 > 74.50)
- Positive skewness (0.602)
- Long right tail
- Consistent with count data pattern

**Platykurtic:**
- Negative excess kurtosis (-1.233)
- Flatter than normal distribution
- Fewer extreme values in tails than normal
- Broader, more dispersed

**Bimodality?**
- No evidence in histogram
- Single-peaked distribution
- Smooth increase over time

### 5.3 Percentile Distribution

| Percentile | Value | Note |
|-----------|-------|------|
| 1% | 19.0 | Minimum value |
| 25% (Q1) | 34.8 | Lower quartile |
| 50% (Median) | 74.5 | Middle value |
| 75% (Q3) | 195.5 | Upper quartile |
| 99% | 268.5 | Near maximum |

**Observations:**
- Median (74.5) much closer to Q1 (34.8) than Q3 (195.5)
  - Distance to Q1: 39.7
  - Distance to Q3: 121.0
  - Ratio: 3.0× (confirms right skew)
- Upper quartile (195.5) is 5.6× lower quartile (34.8)
  - Large multiplicative range

---

## 6. Modeling Recommendations

### 6.1 Primary Model: Negative Binomial Regression

**Model Specification:**
```
Y_t ~ NegBinomial(μ_t, r)
log(μ_t) = β₀ + β₁ × year_t
```

**Why this model:**
- ✓ Handles overdispersion naturally
- ✓ Dramatically better fit than Poisson (ΔAIC = -2417)
- ✓ Well-established, widely available
- ✓ Interpretable parameters

**Parameter Estimates to Expect:**
- **r (dispersion)**: ~1.6 (method of moments)
  - Could use ML for more precise estimate
  - Consider fixing vs estimating
- **β₁ (slope)**: Positive, reflecting upward trend
  - Approximate from data: log(243/27) / (1.67-(-1.67)) ≈ 0.66

**Implementation Notes:**
- Most software has built-in NB regression (R, Python, Stata)
- Use log link (canonical for counts)
- Report both r and α=1/r for clarity

### 6.2 Alternative Model 1: NB with Time-Varying Dispersion

**Model Specification:**
```
Y_t ~ NegBinomial(μ_t, r_t)
log(μ_t) = β₀ + β₁ × year_t
log(r_t) = γ₀ + γ₁ × year_t
```

**When to use:**
- If residual diagnostics show heteroscedasticity
- If model validation suggests poor fit in middle periods
- If scientific interest in dispersion dynamics

**Pros:**
- Captures time-varying dispersion (observed in data)
- May improve fit significantly
- More realistic

**Cons:**
- More parameters (may overfit with n=40)
- More complex to implement
- Harder to interpret

**Recommendation:** Try after fitting standard NB; assess if improvement justifies complexity

### 6.3 Alternative Model 2: Generalized Additive Model (GAM)

**Model Specification:**
```
Y_t ~ NegBinomial(μ_t, r)
log(μ_t) = s(year_t)
```
where s() is smooth function (e.g., spline)

**When to use:**
- If linear trend inadequate
- If residuals show non-linear pattern
- If scientific theory suggests smooth function

**Pros:**
- Flexible functional form
- Can capture non-linearities
- Visual interpretation via smooth curves

**Cons:**
- Risk of overfitting (especially with n=40)
- More parameters
- Less parsimony

**Recommendation:** Exploratory; compare to linear via cross-validation

### 6.4 NOT Recommended: Poisson Models

**Poisson Regression:**
- ❌ Fundamentally misspecified
- ❌ Variance assumption violated by 68-fold
- ❌ Invalid inference
- ❌ Severe underdispersion of SEs

**Quasi-Poisson:**
- ⚠ Better than Poisson (adjusts SEs)
- ⚠ But NB is superior (full likelihood)
- ⚠ Only use if NB convergence issues

**Zero-Inflated Poisson:**
- ❌ Not applicable (0 zeros in data)

---

## 7. Prior Recommendations (Bayesian Framework)

If using Bayesian estimation, suggested priors:

### 7.1 Dispersion Parameter (r or α)

**Prior on r:**
```
r ~ Gamma(2, 1)
```
- Mean = 2, SD = 2
- Puts mass near observed r ≈ 1.6
- Allows flexibility
- Weakly informative

**Alternative: Prior on α = 1/r:**
```
α ~ Exponential(1)
```
- Mean = 1
- Supports observed α ≈ 0.6
- Prevents α → ∞ (Poisson limit)

### 7.2 Regression Coefficients (β)

**Prior on intercept:**
```
β₀ ~ Normal(log(100), 2)
```
- Centers on log(100) ≈ 4.6 (near observed mean)
- SD = 2 allows wide range on exp scale
- Weakly informative

**Prior on slope:**
```
β₁ ~ Normal(0, 2)
```
- Allows positive or negative trend
- SD = 2 → exp(2) ≈ 7-fold change per SD of year
- Weakly informative

### 7.3 Time-Varying Dispersion

**If modeling r_t:**
```
r_t ~ LogNormal(μ_r, σ_r)
μ_r ~ Normal(0, 1)
σ_r ~ HalfNormal(1)
```
- Ensures r_t > 0
- Allows smoothing across time
- Could use AR(1) or random walk

---

## 8. Variance Structure Implications

### 8.1 For Standard Errors

**Poisson SEs would be:**
```
SE_poisson ∝ sqrt(mean) = sqrt(109.45) ≈ 10.46
```

**Actual SEs should be:**
```
SE_actual ∝ sqrt(variance) = sqrt(7441.74) ≈ 86.27
```

**Ratio:** 86.27 / 10.46 ≈ **8.2×**

**Implication:**
- Poisson SEs are **8× too small**
- Confidence intervals would be **severely too narrow**
- Hypothesis tests would be **severely anti-conservative**
- Type I error rate inflated

### 8.2 For Prediction Intervals

**Poisson prediction variance:**
```
Var_pred,Poisson = μ
```

**NB prediction variance:**
```
Var_pred,NB = μ + α × μ² = μ(1 + α × μ)
```

For mean of 109.45:
- Poisson: 109.45
- NB: 109.45 × (1 + 0.612 × 109.45) ≈ **7437**

**68-fold increase** in prediction variance

**Implication:**
- Poisson intervals would severely underestimate uncertainty
- NB intervals appropriately wide

### 8.3 For Model Comparison

When comparing models with different covariates:
- Must use **NB family** as baseline
- Poisson comparisons invalid
- Use likelihood ratio tests with correct null distribution
- Account for overdispersion in test statistics

---

## 9. Practical Implications

### 9.1 For Forecasting

**Prediction uncertainty is large:**
- SD ≈ 86 when mean ≈ 109
- 95% prediction interval ≈ mean ± 170
- Very wide intervals reflect high variability

**Overdispersion source matters:**
- If from unmeasured covariates → could reduce with more data
- If from inherent process variability → irreducible
- Investigate causes of overdispersion

### 9.2 For Experimental Design

**Sample size considerations:**
- Large overdispersion → need larger n for power
- SE proportional to sqrt(μ(1 + αμ)), not sqrt(μ)
- Plan for ~68× more variability than Poisson

**Replication needs:**
- Single observations have high uncertainty
- Consider multiple replicates per time point
- Or denser temporal sampling

### 9.3 For Causal Inference

**If this is treatment effect study:**
- Overdispersion suggests heterogeneous effects
- May indicate subgroups with different responses
- Consider effect modification analysis

**Control for confounding:**
- Unmeasured variables likely contributing to overdispersion
- Include relevant covariates to reduce α
- Check if dispersion decreases with added predictors

---

## 10. Diagnostic Checklist for Final Model

After fitting recommended NB model, check:

### 10.1 Residual Diagnostics
- [ ] Pearson residuals ~N(0,1)?
- [ ] Deviance residuals ~N(0,1)?
- [ ] No patterns vs fitted values?
- [ ] No patterns vs time?
- [ ] Q-Q plot approximately linear?

### 10.2 Overdispersion Tests
- [ ] Dispersion parameter significantly > 0?
- [ ] Likelihood ratio test: NB vs Poisson?
- [ ] Residual deviance / df ≈ 1?

### 10.3 Model Fit
- [ ] AIC comparison to alternatives?
- [ ] BIC comparison to alternatives?
- [ ] Cross-validation RMSE?
- [ ] Calibration plot (observed vs predicted)?

### 10.4 Assumptions
- [ ] Log-linear mean appropriate?
- [ ] Constant dispersion reasonable?
- [ ] No influential outliers?
- [ ] Independence assumption met (or modeled)?

### 10.5 Temporal Structure
- [ ] Check for autocorrelation in residuals
- [ ] ACF/PACF plots
- [ ] Durbin-Watson test
- [ ] Consider AR errors if needed

---

## 11. Key Figures Reference

### Figure 1: `distribution_fitting.png`
**Purpose**: Compare theoretical distributions
**Key insight**: NB fits well, Poisson does not
**Support for**: Distribution family recommendation

### Figure 2: `variance_mean_analysis.png`
**Purpose**: Variance-mean relationship
**Key insight**: Power law with exponent 1.67
**Support for**: Non-linear variance structure

### Figure 3: `temporal_periods_comparison.png`
**Purpose**: Compare dispersion across time periods
**Key insight**: 6-fold variation in dispersion
**Support for**: Heteroscedasticity, time-varying dispersion

### Figure 4: `temporal_dispersion_rolling.png`
**Purpose**: Rolling window dispersion metrics
**Key insight**: Dispersion varies smoothly over time
**Support for**: Time-varying dispersion model

### Figure 5: `outlier_analysis.png`
**Purpose**: Outlier detection
**Key insight**: No outliers, 3 influential points at extremes
**Support for**: Data quality, use full dataset

### Figure 6: `distribution_overview.png`
**Purpose**: Overall distribution characteristics
**Key insight**: Right-skewed, unimodal, platykurtic
**Support for**: Distribution shape assessment

### Figure 7: `count_histogram.png`
**Purpose**: Detailed frequency distribution
**Key insight**: Clear departure from Poisson shape
**Support for**: Need for overdispersed model

---

## 12. Summary: Key Numbers

**Distributional:**
- Variance/Mean ratio: **67.99**
- Dispersion parameter (r): **1.634**
- Overdispersion parameter (α): **0.612**
- Skewness: **0.602**
- Kurtosis (excess): **-1.233**

**Model Selection:**
- ΔAIC (NB - Poisson): **-2416.62**
- ΔBIC (NB - Poisson): **-2414.93**
- ΔLog-likelihood: **+1209.31**

**Variance Structure:**
- Power law exponent: **1.667** (R² = 0.814)
- Heteroscedasticity test: **p < 0.0001**
- Dispersion range across periods: **1.27 to 7.52**

**Data Quality:**
- Missing values: **0**
- Outliers: **0**
- Influential points: **3** (legitimate)

---

## 13. Recommendations Summary

### MUST DO:
1. ✓ Use **Negative Binomial** distribution (not Poisson)
2. ✓ Report dispersion parameter (r and/or α)
3. ✓ Use full dataset (no outliers to remove)

### SHOULD CONSIDER:
4. ⚠ Check for autocorrelation (time series data)
5. ⚠ Assess time-varying dispersion model
6. ⚠ Validate with cross-validation or held-out data

### COULD EXPLORE:
7. ⭐ Investigate sources of overdispersion
8. ⭐ GAM for non-linear trends
9. ⭐ Bayesian framework for uncertainty quantification

### DO NOT:
10. ❌ Use Poisson regression (fundamentally wrong)
11. ❌ Remove any observations (no outliers)
12. ❌ Assume constant variance (heteroscedasticity present)

---

## Appendix: Technical Details

### A.1 Negative Binomial Parameterizations

**NB1** (linear variance):
```
E[Y] = μ
Var[Y] = μ + α×μ
```

**NB2** (quadratic variance):
```
E[Y] = μ
Var[Y] = μ + α×μ²
```

Our data suggests **NB2** more appropriate (power law exponent ≈ 2).

### A.2 Software Implementation

**R:**
```r
library(MASS)
model <- glm.nb(C ~ year, data = data)
```

**Python (statsmodels):**
```python
import statsmodels.api as sm
model = sm.GLM(C, X, family=sm.families.NegativeBinomial())
```

**Python (scikit-learn):** Not directly available; use statsmodels

### A.3 Interpretation of Dispersion Parameter

**r (size parameter):**
- Larger r → approaches Poisson (r → ∞)
- Smaller r → more overdispersion
- r ≈ 1.6 is moderately small

**α (overdispersion parameter):**
- α = Var(Y) - E[Y] / E[Y]²
- α = 0 → Poisson
- α = 0.6 → substantial extra variation

---

**Report prepared by**: EDA Analyst 2 (Distributional Properties Specialist)
**Date**: 2025-10-29
**Dataset**: data/data_analyst_2.csv (n=40)
**Output directory**: /workspace/eda/analyst_2/
