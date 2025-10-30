# EDA Log - Analyst 3: Model Assumptions & Diagnostics

## Objective
Systematically evaluate model assumptions for count data, focusing on:
- Poisson distributional assumptions
- Variance patterns (heteroscedasticity)
- Residual diagnostics from simple models
- Transformation effects

---

## Round 1: Initial Data Exploration

### Data Quality Assessment

- **Sample size**: 40 observations
- **Variables**: year, C
- **Missing values**: 0 (0.00%)
- **Duplicates**: 0
- **Data integrity**: All counts are non-negative integers: True

### Initial Count Distribution Characteristics

- **Mean**: 109.4000
- **Variance**: 7704.6564
- **Variance-to-Mean Ratio**: 70.4265
  - **Interpretation**: Ratio >> 1 indicates **overdispersion** (variance exceeds mean)
  - **Implication**: Simple Poisson model likely inappropriate; consider Negative Binomial
- **Skewness**: 0.6415
- **Kurtosis**: -1.1290

**Initial Hypothesis**: The data shows exponential growth with increasing variance over time. This suggests:
1. Count model needed (non-negative integers)
2. Overdispersion is severe (ratio = 70 >> 1)
3. Transformation or GLM with appropriate variance function required

---

## Round 2: Residual Diagnostics from Simple Models

Fitted three baseline models to understand residual behavior:

### Model 1: Linear Regression (C ~ year)

**Results**:
- Intercept: 109.4000
- Slope: 82.3990
- R²: 0.8812
- Residual SD: 29.87

**Diagnostics**:
- Breusch-Pagan test p-value: 0.3324 → **Homoscedastic** ✓
- Shapiro-Wilk test p-value: 0.1119 → **Normal residuals** ✓

**Interpretation**:
Linear model has well-behaved residuals despite being theoretically inappropriate for count data. This is somewhat surprising but suggests the strong trend dominates. However, predictions could be negative, which is problematic.

### Model 2: Log Transformation (log(C) ~ year)

**Results**:
- Intercept: 4.3337
- Slope: 0.8624
- R²: 0.9367 (highest!)
- Residual SD: 0.2214

**Diagnostics**:
- Breusch-Pagan test p-value: 0.0028 → **Heteroscedastic** ❌
- Shapiro-Wilk test p-value: 0.7961 → **Normal residuals** ✓

**Interpretation**:
Log transformation improves fit substantially (R² = 0.937) and linearizes the relationship. However, it creates heteroscedasticity - variance of residuals is not constant. This is a classic trade-off: better linearity but worse variance structure.

### Model 3: Square Root Transformation (sqrt(C) ~ year)

**Results**:
- Intercept: 9.6057
- Slope: 4.0292
- R²: 0.9240
- Residual SD: 1.1406

**Diagnostics**:
- Breusch-Pagan test p-value: 0.0148 → **Heteroscedastic** ❌
- Shapiro-Wilk test p-value: 0.3250 → **Normal residuals** ✓

**Interpretation**:
Square root transformation is a middle ground - better fit than linear (R² = 0.924) but less heteroscedasticity than log (p = 0.015 vs 0.003). For count data with large values, this is often a reasonable compromise.

**Key Finding**: All three models show normal residuals, but transformations introduce heteroscedasticity. This suggests using GLM with log link instead of transforming the response.

---

## Round 3: Detailed Count Model Diagnostics

### Equidispersion Test (Poisson Assumption)

- **Mean**: 109.4000
- **Variance**: 7704.6564
- **Dispersion Ratio**: 70.4265
- **Formal test statistic**: 2746.6325
- **Critical value** (χ²₃₉, α=0.05): 54.5722
- **p-value**: < 0.001
- **Decision**: **REJECT Poisson**

**Conclusion**: SEVERE OVERDISPERSION - Poisson inappropriate

### Distribution Characteristics

- **Skewness**: 0.6415 (Poisson expected: 0.0956)
- **Excess Kurtosis**: -1.1290 (Poisson expected: 0.0091)
- **Coefficient of Variation**: 0.8023 (Poisson expected: 0.0956)

**Interpretation**: All distribution shape metrics deviate substantially from Poisson expectations, confirming the need for alternative models.

### Zero-Inflation Check

- **Observed zeros**: 0
- **Expected zeros (Poisson)**: 0.00
- **Conclusion**: No zero-inflation

**Interpretation**: Zero-Inflated models not needed. Standard Negative Binomial should suffice.

### Variance Stability Over Time

| Quartile | Time Period | Mean | Variance | Var/Mean Ratio |
|----------|-------------|------|----------|----------------|
| Q1 | Early | 27.50 | 16.06 | 0.58 |
| Q2 | Early-Mid | 45.70 | 206.23 | 4.51 |
| Q3 | Mid-Late | 126.60 | 1500.27 | **11.85** |
| Q4 | Late | 237.80 | 1055.73 | 4.44 |

- **Levene's Test statistic**: 4.39
- **Levene's Test p-value**: 0.0099
- **Interpretation**: **Heterogeneous variances** across time periods

**Key Finding**: Variance is not stable over time. Q3 shows particularly high variability (ratio = 11.85). This confirms heteroscedasticity is not just a transformation artifact but an inherent feature of the data.

### Autocorrelation Analysis

- **Lag-1 Autocorrelation**: 0.9710 (very high!)
- **Durbin-Watson Statistic**: 0.4724 (indicates strong positive autocorrelation)

**Interpretation**: **CRITICAL FINDING** - consecutive observations are extremely correlated. This violates independence assumption of standard regression. Implications:
1. Standard errors will be underestimated
2. Confidence intervals too narrow
3. Hypothesis tests anti-conservative (Type I error inflation)

**Recommendation**: Use GEE with AR(1) correlation structure or GLMM with temporal random effects.

---

## Round 4: Variance-Mean Relationship Analysis

### Sliding Window Analysis (window size = 10)

The variance-mean plot shows:
- Observed points lie far above Poisson reference line (var = mean)
- Relationship appears roughly quadratic: var ∝ mean²
- This is consistent with Negative Binomial: var = μ + αμ²

**Interpretation**: The dispersion parameter α in Negative Binomial captures the quadratic relationship. This model is theoretically well-suited.

### Scale-Location Plot (Linear Model)

Squared residuals vs fitted values shows:
- Slight upward trend (variance increasing with mean)
- Not dramatic, suggesting linear model residuals are relatively stable
- But still present, confirming mean-dependent variance

---

## Round 5: Influence and Leverage Diagnostics

### Cook's Distance

- Maximum Cook's D: ~0.06
- Threshold (4/n): 0.10
- **Result**: No observations exceed threshold

**Conclusion**: No highly influential outliers detected. All observations contribute reasonably to the fit.

### Leverage Analysis

- All observations have leverage < 2p/n = 0.10
- Extreme year values (earliest and latest) have slightly higher leverage (expected)
- **Result**: No high-leverage outliers

**Conclusion**: Model estimates are not driven by a few extreme points. Results are robust.

---

## Hypothesis Testing

### Hypothesis 1: Poisson model is appropriate
- **Test**: Equidispersion test (variance/mean ratio)
- **Result**: REJECTED (ratio = 70.4, p < 0.001)
- **Evidence**: variance_mean_relationship.png, dispersion_analysis.png

### Hypothesis 2: Log transformation stabilizes variance
- **Test**: Breusch-Pagan test on log(C) ~ year
- **Result**: REJECTED (p = 0.003, heteroscedastic)
- **Evidence**: residual_diagnostics_all_models.png (middle row)
- **Alternative**: Use GLM with log link instead of log transformation

### Hypothesis 3: Data is independent over time
- **Test**: Lag-1 autocorrelation, Durbin-Watson
- **Result**: REJECTED (r = 0.97, DW = 0.47)
- **Evidence**: Strong temporal dependence
- **Implication**: Need correlated error structure in model

---

## Alternative Explanations Considered

### Could the overdispersion be due to outliers?
**Answer**: No.
- Cook's distance shows no influential points
- Boxplots show no extreme outliers
- Pattern is consistent across the entire dataset

### Could the data be from multiple populations?
**Answer**: Unlikely.
- No obvious clustering in residuals
- Temporal trend is smooth and monotonic
- More likely: single process with time-varying rate

### Could a simpler linear model work?
**Answer**: Technically yes for in-sample fit, but:
- Predictions could be negative (unacceptable for counts)
- Doesn't respect data generation process
- Standard errors underestimated due to autocorrelation

---

## Model Recommendation Rationale

### Why Negative Binomial is Preferred

1. **Handles overdispersion naturally**: var = μ + αμ²
2. **Respects count nature**: Predictions always non-negative integers (in distribution)
3. **Log link**: Ensures positive mean, interpretable coefficients
4. **Dispersion parameter**: Explicitly models extra-Poisson variation
5. **Robust**: Works well even with strong trends

### Why Not Other Models

**Standard Poisson**:
- Assumption violated (dispersion ratio = 70)
- Would severely underestimate uncertainty

**Zero-Inflated Poisson/NB**:
- No zeros observed
- Unnecessary complexity

**Quasi-Likelihood**:
- Good for prediction but no likelihood inference
- Can't use AIC/BIC for model comparison

**Gaussian GLM**:
- Doesn't respect count constraints
- Less interpretable for count processes

---

## Robustness Assessment

### Robust Findings (very high confidence)

1. ✓ **Overdispersion is real**: Multiple tests (equidispersion, variance analysis, model comparisons) all confirm
2. ✓ **Autocorrelation is strong**: Lag-1 correlation and Durbin-Watson both indicate
3. ✓ **Exponential growth pattern**: Consistent across all visualizations
4. ✓ **No data quality issues**: Complete data, no outliers, uniform sampling

### Tentative Findings (moderate confidence)

⚠ **Exact form of variance function**: Need model-based estimates of α
⚠ **Stationarity**: Only 40 points, hard to assess regime changes
⚠ **Linearity on log scale**: Very good (R² = 0.937) but could have nonlinear components

### What We Don't Know (needs further analysis)

? **Presence of change points**: Could growth rate shift at some time?
? **Seasonal patterns**: Not applicable (annual data assumed)
? **External predictors**: Only have time trend, could other variables explain variation?

---

## Visualization Summary

1. **residual_diagnostics_all_models.png**:
   - Shows linear model has best residual properties
   - Log model has worst heteroscedasticity
   - All models have approximately normal residuals

2. **variance_mean_relationship.png**:
   - Variance far exceeds mean (not Poisson)
   - Quadratic relationship suggests Negative Binomial

3. **dispersion_analysis.png**:
   - Observed distribution wider than Poisson
   - Variance changes over time (heterogeneous)
   - Strong temporal trend visible

4. **transformation_comparison.png**:
   - Log transformation best R² but worst heteroscedasticity
   - Trade-off between fit quality and assumption validity

5. **influence_diagnostics.png**:
   - No influential outliers
   - Results are robust to individual observations

---

## Final Recommendations

### For Immediate Modeling

1. **Primary model**: Negative Binomial regression with log link
2. **Specification**: log(μₜ) = β₀ + β₁ × yearₜ
3. **Correlation structure**: Consider GEE with AR(1) working correlation
4. **Validation**: Use randomized quantile residuals for diagnostics

### For Model Refinement

1. Check for nonlinearity (polynomial or spline terms)
2. Test for structural breaks or change points
3. Compare AIC/BIC across candidate models
4. Perform sensitivity analysis (bootstrap, cross-validation)

### For Reporting

1. Report dispersion parameter estimate with confidence interval
2. Show variance-mean relationship from fitted model
3. Provide prediction intervals (not just confidence intervals)
4. Acknowledge autocorrelation and its impact on inference

---

## Conclusion

This EDA has systematically explored model assumptions for count data through:
- Three rounds of residual diagnostics across different transformations
- Formal tests for equidispersion, heteroscedasticity, and normality
- Variance-mean relationship analysis
- Autocorrelation and influence diagnostics

**The data strongly supports Negative Binomial regression over Poisson**, with careful attention to temporal autocorrelation. The analysis provides concrete, evidence-based guidance for model selection and specification.

All findings are supported by multiple independent tests and visualizations, providing high confidence in the recommendations.
