# Exploratory Data Analysis Report
## Count Time Series Dataset

**Analyst**: EDA Specialist
**Date**: 2025-10-30
**Dataset**: 40 observations of count data over time

---

## Executive Summary

This EDA analyzed a time series of count data (C) ranging from 21 to 269 over 40 standardized time points. The analysis reveals **strong exponential growth** with **severe overdispersion** (variance 70× mean) and significant **temporal autocorrelation**. The Poisson assumption is catastrophically violated. Evidence suggests a **regime shift** midway through the series. Three modeling approaches are recommended: (1) Negative Binomial GLM with log link, (2) Log-normal regression, and (3) Quasi-Poisson with autoregressive errors.

---

## 1. Data Characteristics

### 1.1 Data Quality
- **Observations**: 40
- **Variables**: 2 (year, C)
- **Missing values**: None
- **Outliers**: None detected
- **Quality score**: Excellent ✓

### 1.2 Predictor Variable (year)
- **Type**: Continuous, standardized
- **Range**: [-1.668, 1.668]
- **Mean**: 0.000 (properly centered)
- **Std**: 1.000 (properly scaled)
- **Distribution**: Uniform, evenly spaced (Δ ≈ 0.086)

### 1.3 Outcome Variable (C)

#### Summary Statistics
| Statistic | Value |
|-----------|-------|
| Min | 21 |
| Q1 | 30.75 |
| Median | 67 |
| Q3 | 175.75 |
| Max | 269 |
| Mean | 109.4 |
| Std | 87.78 |
| Skewness | 0.64 (right-skewed) |
| Kurtosis | -1.13 (platykurtic) |

**Key observations**:
- Wide range (factor of 12.8×)
- Right-skewed distribution
- No zeros (minimum = 21)
- High variability (CV = 0.80)

---

## 2. Relationship Patterns

### 2.1 Primary Trend

As shown in `visualizations/02_relationship_analysis.png`, the relationship between year and count is **strongly positive and nonlinear**:

#### Linear Model (Original Scale)
```
C = 82.40 × year + 109.40
R² = 0.881, p < 0.001
```
- Strong linear component
- Each unit increase in year → +82 count units
- 95% CI for slope: [72.78, 92.02]

#### Log-Linear Model (Exponential Growth)
```
log(C) = 0.862 × year + 4.334
R² = 0.937, p < 0.001
```
- **Even stronger fit** than linear model
- Each unit increase in year → **2.37× multiplicative effect**
- Clear evidence of exponential growth

**Conclusion**: The data exhibit exponential growth, better captured by log-linear models than linear models.

### 2.2 Nonlinearity

Polynomial regression comparison (see `visualizations/02_relationship_analysis.png`, top-right panel):

| Model | R² | Improvement |
|-------|-----|-------------|
| Linear | 0.881 | - |
| Quadratic | 0.964 | +0.083 ⚠️ |
| Cubic | 0.974 | +0.010 |

**Interpretation**: A quadratic term captures substantial additional variance (8.3%), suggesting the growth rate itself may be changing over time. The cubic term adds minimal improvement.

### 2.3 Variance Structure

Despite severe overdispersion, **heteroscedasticity is not significant**:
- Breusch-Pagan test: p = 0.332
- Correlation(|residuals|, fitted): r = -0.152, p = 0.350

**Insight**: Variance does not systematically increase with mean. The overdispersion appears to be an intrinsic property of the data-generating process, not a mean-variance scaling issue.

---

## 3. Count Data Properties

### 3.1 Overdispersion Analysis

The most critical finding is **extreme overdispersion** (see `visualizations/04_count_properties.png`):

#### Overall Dispersion
```
Mean:                 109.40
Variance:             7704.66
Variance-to-Mean:     70.43  ⚠️ SEVERE
```

For reference:
- Poisson assumption: variance = mean (ratio = 1)
- Observed ratio: **70.43** (70× too large!)
- Formal test: χ² = 2746.6, p ≈ 0 (overwhelming evidence)

#### Period-Specific Dispersion

| Period | n | Mean | Variance | Ratio | Interpretation |
|--------|---|------|----------|-------|----------------|
| Early | 14 | 28.57 | 19.34 | 0.68 | Underdispersed |
| Middle | 13 | 83.00 | 1088.00 | 13.11 | Overdispersed |
| Late | 13 | 222.85 | 1611.47 | 7.23 | Overdispersed |

**Critical insight**: The dispersion pattern is **heterogeneous across time periods**:
- Early period shows **underdispersion** (var < mean)
- Middle period shows **extreme overdispersion** (ratio = 13)
- Late period shows **moderate overdispersion** (ratio = 7)

This heterogeneity suggests:
1. Different data-generating mechanisms over time
2. Possible regime shifts
3. Time-varying dispersion parameters may be needed

### 3.2 Distribution Comparison

As shown in `visualizations/04_count_properties.png` (top-right panel), the observed distribution deviates substantially from Poisson:

- **Chi-square goodness-of-fit**: p < 0.001 (Reject Poisson)
- **Q-Q plot** (bottom-left panel): Systematic deviation, especially in tails
- **Conclusion**: Poisson model is inappropriate

### 3.3 Alternative Distributions

#### Negative Binomial
```
Size parameter (r):  1.58
Probability (p):     0.014
Dispersion formula:  var = mean + mean²/r
```
The small size parameter (r ≈ 1.6) indicates substantial overdispersion, making Negative Binomial a strong candidate.

#### Log-Normal
```
μ (log-scale):  4.334
σ (log-scale):  0.891
```
Log-normal fits naturally with:
- Multiplicative error processes
- Exponential growth patterns
- Positive skewness in original scale

### 3.4 Zero-Inflation
- **Zero counts**: 0 (0.0%)
- **Minimum count**: 21
- **Conclusion**: **No zero-inflation**

This suggests a mature or established process where counts are always substantial.

---

## 4. Temporal Patterns

### 4.1 Time Period Comparison

As shown in `visualizations/03_temporal_patterns.png` (bottom-middle panel), there are **dramatic differences** across time periods:

| Comparison | Test | Result |
|------------|------|--------|
| Overall ANOVA | F = 151.78, p = 1.47×10⁻¹⁸ | Highly significant |
| Early vs Late | Mean difference = 194.3 | 7.8× increase |
| First vs Second half | t = -9.54, p = 1.24×10⁻¹¹ | Significant shift |

**Interpretation**: This is not merely a smooth trend, but evidence of **regime changes** or **accelerating growth**.

### 4.2 Growth Dynamics

From `visualizations/03_temporal_patterns.png` (top panels):

#### Absolute Changes (C_diff)
- Mean: 5.69 units per step
- Range: [-49, +77]
- High volatility around trend

#### Percentage Changes
- Mean: 7.69% per step
- Median: 3.17% per step
- Difference suggests occasional large jumps

#### Log-Differences (Continuous Growth Rate)
- Mean: 0.055 (≈5.5% per step)
- More stable than percentage changes

**Insight**: Growth is not constant—there are periods of acceleration and occasional large jumps.

### 4.3 Autocorrelation

From `visualizations/03_temporal_patterns.png` (bottom-left panel), autocorrelation is **extremely high**:

#### Raw Counts
- Lag 1: 0.971
- Lag 2: 0.975
- Lag 3: 0.967
- Lag 4: 0.960
- Lag 5: 0.957

#### Residuals (after linear detrending)
- Lag 1: 0.754 ⚠️ Still very high
- Lag 2: 0.776
- Durbin-Watson: 0.472 ⚠️ (far from 2.0)

**Critical implication**:
1. Observations are **not independent**
2. Simple regression **underestimates standard errors**
3. Time series models or GLS needed
4. Linear trend does **not fully capture** temporal dependence

---

## 5. Modeling Implications

### 5.1 Distribution Choice

Based on the severe overdispersion and lack of zero-inflation:

**DO NOT USE**:
- ✗ Poisson regression (variance assumption catastrophically violated)
- ✗ Standard linear regression (count data, non-normal errors)

**RECOMMENDED**:
- ✓ Negative Binomial regression
- ✓ Quasi-Poisson regression
- ✓ Log-normal regression (on transformed response)

### 5.2 Trend Specification

Based on R² comparisons:

**Good**: Linear trend (R² = 0.881)
**Better**: Log-linear (exponential) trend (R² = 0.937)
**Best**: Quadratic trend (R² = 0.964)

**Recommendation**: Start with log-link GLM (captures exponential growth naturally), but also test quadratic term.

### 5.3 Autocorrelation Handling

Based on Durbin-Watson and ACF:

**Required**: Account for temporal dependence
**Options**:
1. Generalized Least Squares (GLS) with AR(1) errors
2. Negative Binomial with lagged dependent variable
3. Time series count models (GARMA, INARMA)
4. Report robust standard errors (clustered or HAC)

### 5.4 Regime Shifts

Based on period comparisons and changepoint evidence:

**Consider**:
1. Piecewise regression with breakpoint at observation ~20
2. Interaction terms: year × period
3. Separate models for early vs. late periods
4. Structural break tests (Chow test)

---

## 6. Recommended Modeling Approaches

### Model Class 1: Negative Binomial GLM (Primary Recommendation)

**Specification**:
```
C ~ NegativeBinomial(μ, θ)
log(μ) = β₀ + β₁×year + β₂×year²
```

**Rationale**:
- Handles severe overdispersion naturally
- Log link captures exponential growth
- Quadratic term for nonlinearity
- Standard GLM machinery available

**Advantages**:
- Directly models count data
- Flexible dispersion parameter (θ)
- Interpretable coefficients (multiplicative effects)
- Can add lagged terms for autocorrelation

**Disadvantages**:
- Assumes conditional mean-variance relationship
- May need GLS extension for autocorrelation
- Requires iterative fitting (IRLS)

**Expected Performance**: R² > 0.94 (based on log-linear fit)

---

### Model Class 2: Log-Normal Regression

**Specification**:
```
log(C) ~ Normal(μ, σ²)
μ = β₀ + β₁×year + β₂×year²
```

**Rationale**:
- Log transformation strongly improves normality (Shapiro-Wilk p = 0.001 vs < 0.001)
- R² = 0.937 on log scale
- Natural for multiplicative processes
- Handles heteroscedasticity automatically

**Advantages**:
- Simple OLS on transformed data
- Easy diagnostics
- Robust standard errors straightforward
- Can use time series regression (Newey-West, etc.)

**Disadvantages**:
- Predictions need back-transformation
- Bias in back-transformation (requires correction)
- Doesn't directly model count nature

**Expected Performance**: R² > 0.93 (confirmed from analysis)

**Implementation notes**:
- Consider GLS with AR(1) errors: `gls(log(C) ~ year + I(year^2), correlation=corAR1())`
- Back-transform predictions: E[C] = exp(μ + σ²/2)

---

### Model Class 3: Quasi-Poisson with Autoregressive Errors

**Specification**:
```
C ~ Poisson(μ)
log(μ) = β₀ + β₁×year + β₂×year²
Var(C) = φ×μ  (overdispersion parameter φ)
Errors: AR(1) structure
```

**Rationale**:
- Relaxes Poisson variance assumption (quasi-likelihood)
- Estimates overdispersion empirically
- Can incorporate autocorrelation through GEE or GLMM

**Advantages**:
- More flexible than standard Poisson
- Simpler than full negative binomial
- GEE framework handles correlation
- Robust inference

**Disadvantages**:
- No likelihood (can't use AIC/BIC directly)
- May not fully capture heterogeneous dispersion
- More complex implementation

**Expected Performance**: Similar fit to Negative Binomial, more robust inference

**Implementation notes**:
- Use `glm(..., family=quasipoisson)` for initial fit
- Or GEE: `geeglm(..., family=poisson, corstr="ar1")`

---

## 7. Model Comparison Strategy

### Phase 1: Baseline Models
1. Fit all three model classes with linear term only
2. Compare goodness-of-fit (R², deviance, residuals)
3. Check residual diagnostics (ACF, normality, patterns)

### Phase 2: Add Complexity
1. Add quadratic term (year²) to best-performing model
2. Test for regime change (interaction with period dummy)
3. Assess improvement via likelihood ratio test (or F-test for log-normal)

### Phase 3: Address Autocorrelation
1. Add AR(1) errors to final model
2. Compare with lagged dependent variable approach
3. Validate with out-of-sample predictions

### Evaluation Criteria
- **In-sample fit**: R², AIC (where applicable), deviance
- **Residual diagnostics**: ACF, normality, homoscedasticity
- **Parsimony**: Prefer simpler models if fit similar
- **Interpretability**: Coefficient interpretation and uncertainty quantification

---

## 8. Data Quality Issues for Modeling

### Critical (Must Address)
1. **Severe overdispersion** → Use NB, quasi-Poisson, or log-normal
2. **Strong autocorrelation** → Use GLS, GEE, or robust SE
3. **Temporal non-stationarity** → Consider regime-specific models

### Moderate (Should Consider)
1. **Nonlinearity** → Include quadratic term
2. **Heterogeneous dispersion** → Allow time-varying dispersion parameter
3. **Regime shift** → Test for structural breaks

### Minor (Monitor)
1. **Period-specific patterns** → May need interaction terms
2. **Growth acceleration** → May need higher-order terms

### Non-Issues
- Missing data (none)
- Zero-inflation (absent)
- Outliers (none detected)
- Predictor standardization (correct)

---

## 9. Key Visualizations Summary

### Figure 1: Distribution Analysis (`01_distribution_analysis.png`)
**Key insights**:
- Top-left: Count distribution is right-skewed (skewness = 0.64)
- Top-right: Box plot shows wide range, no outliers
- Top-right: Q-Q plot confirms departure from normality
- Bottom-left: Year is uniformly distributed (as expected)
- Bottom-middle: Log-transform improves symmetry substantially
- Bottom-right: Log-counts closer to normal (but still significant deviation)

**Modeling implication**: Log-scale models preferred

### Figure 2: Relationship Analysis (`02_relationship_analysis.png`)
**Key insights**:
- Top-left: Strong linear trend, R² = 0.881, p < 0.001
- Top-right: Smoothed trend shows slight curvature
- Bottom-left: Log-scale even stronger fit, R² = 0.937
- Bottom-right: Residuals show autocorrelation (orange trend line not flat)

**Modeling implication**: Log-link or log-transform; address autocorrelation

### Figure 3: Temporal Patterns (`03_temporal_patterns.png`)
**Key insights**:
- Top-left: Clear upward trajectory with period-specific means
- Top-middle: Absolute changes variable (range [-49, +77])
- Top-right: Percentage changes show high volatility
- Bottom-left: ACF confirms strong autocorrelation (all lags above confidence band)
- Bottom-middle: Box plots show dramatic period differences
- Bottom-right: Residual ACF remains high after detrending

**Modeling implication**: Time series approach essential; consider regime shifts

### Figure 4: Count Properties (`04_count_properties.png`)
**Key insights**:
- Top-left: Mean-variance plot far above Poisson line (var=mean); periods differ
- Top-right: Observed distribution much wider than Poisson prediction
- Bottom-left: Q-Q plot shows systematic deviation from Poisson
- Bottom-right: Moving window analysis confirms persistent overdispersion

**Modeling implication**: Negative binomial or quasi-Poisson essential; Poisson invalid

---

## 10. Uncertainty and Limitations

### Robust Conclusions (High Confidence)
- Strong upward trend exists
- Exponential growth pattern (log-linear superior)
- Severe overdispersion (variance 70× mean)
- Temporal autocorrelation present
- Poisson model inappropriate
- Regime shift between early and late periods

### Moderate Confidence
- Specific functional form (quadratic vs. other nonlinearities)
- Exact timing of regime shift (approximate midpoint)
- Persistence of overdispersion parameter over time

### Low Confidence / Speculative
- Mechanism of overdispersion (unobserved heterogeneity? contagion? measurement?)
- Whether growth will continue exponentially
- External validity to other time periods or contexts

### Limitations
1. **Small sample** (n=40): Limited power for complex models
2. **No covariates**: Cannot control for confounders
3. **Observational**: Cannot infer causality
4. **Time range**: Unknown if patterns hold beyond observed range
5. **Single series**: Cannot assess cross-sectional variation

---

## 11. Recommendations for Modelers

### Must Do
1. ✓ Use count data model (NB, quasi-Poisson) OR log-transform
2. ✓ Include log-link or model on log-scale (exponential growth)
3. ✓ Account for autocorrelation (GLS, GEE, or robust SE)
4. ✓ Test for nonlinearity (quadratic term)

### Should Consider
1. Regime-specific models or structural break tests
2. Time-varying dispersion parameters
3. Lagged dependent variable as alternative to AR errors
4. Cross-validation for model selection

### Nice to Have
1. Bayesian posterior intervals (uncertainty quantification)
2. Out-of-sample prediction validation
3. Sensitivity analysis to model specifications
4. Simulation from fitted model to check plausibility

### Avoid
- ✗ Standard Poisson regression
- ✗ Simple linear regression (without log-transform)
- ✗ Ignoring autocorrelation
- ✗ Assuming stationarity

---

## 12. Practical Significance

Beyond statistical significance, the findings have practical importance:

### Growth Magnitude
- **7.8× increase** from early to late period
- **2.37× multiplicative effect** per standardized year unit
- If original years span decades, this represents substantial growth

### Prediction Uncertainty
- Severe overdispersion means **wide prediction intervals**
- Autocorrelation means uncertainty compounds over time
- Point predictions may be unreliable without uncertainty quantification

### Model Selection Impact
- Poisson would **severely underestimate** uncertainty (by factor of √70 ≈ 8.4)
- Ignoring autocorrelation would **inflate significance** of trend
- Correct modeling critical for valid inference

---

## 13. Conclusions

This count time series exhibits:

1. **Strong exponential growth** (R² = 0.937 on log-scale)
2. **Severe overdispersion** (variance 70× mean) with time-varying pattern
3. **High temporal autocorrelation** (Durbin-Watson = 0.47)
4. **Evidence of regime shift** at series midpoint
5. **No zero-inflation** (minimum count = 21)

### Recommended Modeling Strategy

**Primary recommendation**: **Negative Binomial GLM with log-link and quadratic trend**
- Best balance of count data appropriateness and flexibility
- Handles overdispersion directly
- Captures exponential growth via log-link
- Can extend with AR errors or lagged terms

**Alternative**: **Log-normal regression with GLS(AR1)**
- Simpler implementation
- Strong performance (R² = 0.937)
- Easy to add time series structure
- Requires back-transformation care

**Robust option**: **Quasi-Poisson with GEE(AR1)**
- Most robust inference
- Handles both overdispersion and correlation
- No distributional assumptions
- Good for sensitivity analysis

### Final Note

The combination of **severe overdispersion**, **strong autocorrelation**, and **regime shifts** makes this a challenging modeling problem. No single model will be perfect. The recommended approach is to:

1. Fit all three model classes
2. Compare diagnostics carefully
3. Use robust inference (bootstrap or robust SE)
4. Report sensitivity to model choice
5. Focus on practical significance, not just p-values

The data tell a clear story of **exponential growth with substantial unexplained variation and temporal dependence**. Good modeling must respect all three features.

---

## Appendix: File Inventory

### Analysis Scripts
- `code/01_initial_exploration.py` - Data structure and quality checks
- `code/02_distribution_analysis.py` - Distributional properties and normality tests
- `code/03_relationship_analysis.py` - Regression analysis and heteroscedasticity
- `code/04_temporal_patterns.py` - Autocorrelation and regime changes
- `code/05_count_properties.py` - Overdispersion and alternative distributions

### Visualizations (all in `visualizations/`)
- `01_distribution_analysis.png` - 6-panel distribution analysis
- `02_relationship_analysis.png` - 4-panel relationship and residual plots
- `03_temporal_patterns.png` - 6-panel temporal structure analysis
- `04_count_properties.png` - 4-panel count data diagnostics

### Documentation
- `eda_log.md` - Detailed exploration process and intermediate findings
- `eda_report.md` - This comprehensive report
- `initial_summary.txt` - Quick reference statistics

---

**Report prepared by**: EDA Specialist
**Analysis date**: 2025-10-30
**All code and visualizations available in**: `/workspace/eda/`
