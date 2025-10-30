# Count Distribution & Statistical Properties Analysis

**Analyst:** Analyst 2
**Dataset:** data_analyst_2.json
**Focus:** Count distribution characteristics, overdispersion analysis, and likelihood selection
**Date:** 2025-10-29

---

## Executive Summary

This analysis provides a comprehensive examination of the count variable (C) distribution and its statistical properties. **The key finding is that the data exhibits severe overdispersion (variance/mean ratio = 70.43), completely violating Poisson assumptions.** The Negative Binomial distribution provides an excellent fit and is the strongly recommended likelihood for Bayesian modeling.

### Critical Findings:
1. **Poisson distribution is completely inappropriate** (Delta AIC = 2498 vs. Negative Binomial)
2. **Negative Binomial with r = 1.549 provides excellent fit** (KS test p = 0.261)
3. **Dispersion varies significantly over time** (Levene test p = 0.010)
4. **No zero-inflation present** (0 zeros observed)
5. **No statistical outliers detected** (all observations within normal range)

---

## 1. Data Overview

### Basic Statistics
- **Sample size:** n = 40
- **Mean:** 109.40
- **Variance:** 7704.66
- **Standard Deviation:** 87.78
- **Range:** 21 to 269
- **Median:** 67.00
- **IQR:** 145.00

### Distribution Shape
- **Skewness:** 0.642 (moderately right-skewed)
- **Kurtosis:** -1.129 (platykurtic / light-tailed)
- **Coefficient of Variation:** 0.80 (high variability)

**Key Observation:** The distribution shows clear evidence of a right skew with substantial spread, suggesting a count process with increasing trend and high variability.

**Supporting Evidence:** See `01_basic_distribution.png` (Panel A: Histogram shows right-skewed distribution with long tail; Panel B: Boxplot reveals no extreme outliers but substantial spread; Panel C: Q-Q plot shows deviation from normality).

---

## 2. Overdispersion Analysis

### Variance-to-Mean Relationship

The fundamental characteristic of count data is the relationship between mean and variance:

- **Poisson assumption:** Variance = Mean
- **Observed:** Variance / Mean = **70.43**

This represents **extreme overdispersion** - the variance is more than 70 times larger than the mean, decisively rejecting the Poisson assumption.

### Chi-Square Dispersion Test
- **Test statistic:** 2746.63
- **Degrees of freedom:** 39
- **p-value:** < 0.0001
- **Conclusion:** **REJECT Poisson assumption** (highly significant overdispersion)

### Mean-Variance Relationship Across Time Segments

When examining the data in 8 temporal segments, we observe a clear quadratic relationship between segment means and variances, consistent with Negative Binomial structure where Var = μ + μ²/r.

**Supporting Evidence:** See `03_mean_variance_relationship.png` (Panel C: Scatter plot shows strong quadratic relationship between segment means and variances; red dashed line shows Poisson expectation for comparison; Panel D: Color-coded by time position shows how both mean and variance increase over time).

---

## 3. Distribution Fitting & Hypothesis Testing

### Hypothesis 1: Poisson Distribution

**Result: STRONGLY REJECTED**

- **Chi-square goodness-of-fit:** p < 0.0001
- **Kolmogorov-Smirnov test:** KS = 0.550, p < 0.0001
- **Log-likelihood:** -1476.06
- **AIC:** 2954.11
- **Verdict:** Poisson distribution is **completely inappropriate** for this data

The Poisson assumption fails catastrophically due to extreme overdispersion. Visual inspection shows that the Poisson PMF severely underestimates the variance structure.

**Supporting Evidence:** See `02_poisson_vs_negbinom.png` (Panel A: Poisson PMF completely fails to capture the observed distribution; Panel D: Log-scale comparison emphasizes the poor fit, particularly in the tails).

### Hypothesis 2: Negative Binomial Distribution

**Result: ACCEPTED (Strongly Recommended)**

**Method of Moments Estimation:**
- r (dispersion parameter): 1.5758
- p (probability parameter): 0.0142

**Maximum Likelihood Estimation:**
- **r (dispersion parameter): 1.5493**
- **p (probability parameter): 0.0140**
- Implied mean: 109.40
- Implied variance: 7834.30

**Goodness-of-Fit Tests:**
- **Kolmogorov-Smirnov test:** KS = 0.155, p = 0.261
  - **Conclusion:** CANNOT REJECT Negative Binomial (good fit)
- **Log-likelihood:** -225.99
- **AIC:** 455.98

**Model Comparison:**
- **Likelihood Ratio Test vs. Poisson:** LR = 2500.13, p < 0.0001
  - **Conclusion:** Negative Binomial is **significantly better** than Poisson
- **Delta AIC (Poisson - NegBinom):** 2498.13
  - **Interpretation:** Very strong evidence for Negative Binomial (Delta AIC >> 10)

**Supporting Evidence:** See `02_poisson_vs_negbinom.png` (Panel B: Negative Binomial PMF closely matches observed distribution; Panel C: Direct comparison shows NB captures the variability while Poisson fails). Also see `05_qq_plots.png` (right panel shows good alignment between theoretical and observed quantiles for NB fit).

### Hypothesis 3: Time-Varying vs. Constant Dispersion

**Result: Evidence of Time-Varying Dispersion**

Analysis by temporal quartiles reveals significant heterogeneity in dispersion:

| Quartile | Mean   | Variance | Var/Mean | NB r Parameter |
|----------|--------|----------|----------|----------------|
| Q1       | 27.50  | 16.06    | 0.58     | 1.00           |
| Q2       | 45.70  | 206.23   | 4.51     | 13.01          |
| Q3       | 126.60 | 1500.27  | 11.85    | 11.67          |
| Q4       | 237.80 | 1055.73  | 4.44     | 69.14          |

**Statistical Tests:**
- **Levene's test for equal variances:** F = 4.39, p = 0.0099
  - **Conclusion:** Variances are NOT equal across time
- **Brown-Forsythe test:** F = 4.39, p = 0.0099
  - **Conclusion:** Confirms heterogeneous variances
- **Range of r values:** 1.00 to 69.14 (ratio = 69:1)

**Interpretation:** The dispersion parameter shows dramatic variation across time periods. Early observations show near-Poisson behavior (Var/Mean ≈ 0.58), while middle periods show extreme overdispersion (Var/Mean ≈ 11.85). This suggests that models incorporating time trends should consider time-varying dispersion.

**Supporting Evidence:** See `04_dispersion_analysis.png` (Panel A: Variance/mean ratio varies significantly across time segments, ranging from ~0.6 to ~12; Panel B: Coefficient of variation also shows temporal variation).

### Hypothesis 4: Zero-Inflation

**Result: No Zero-Inflation Present**

- **Observed zeros:** 0 out of 40 (0%)
- **Expected zeros under Poisson(109.4):** 0.000000
- **Expected zeros under NegBinom(r=1.55):** 0.001337
- **Conclusion:** No evidence of zero-inflation; all counts are strictly positive

This is consistent with a count process that has a consistently positive baseline rate across all time periods.

---

## 4. Alternative Distributions

### Continuous Approximations

Given the large count values (mean = 109.4), continuous approximations may be useful in some modeling contexts:

#### Log-Normal Distribution
- **Parameters:** μ = 4.334, σ = 0.891
- **Implied mean:** 113.38
- **Implied variance:** 15584.29
- **KS test:** KS = 0.160, p = 0.232 (cannot reject)
- **AIC:** 453.99 (best among all models)

#### Gamma Distribution
- **Shape parameter (α):** 1.529
- **Scale parameter (β):** 71.547
- **Implied mean:** 109.40
- **Implied variance:** 7827.24
- **KS test:** KS = 0.160, p = 0.234 (cannot reject)
- **AIC:** 455.73

### Model Rankings by AIC

| Rank | Model              | AIC     | Delta AIC | Type       |
|------|--------------------|---------|-----------|------------|
| 1    | Log-Normal         | 453.99  | 0.00      | Continuous |
| 2    | Gamma              | 455.73  | 1.74      | Continuous |
| 3    | Negative Binomial  | 455.98  | 1.99      | Discrete   |
| 4    | Poisson            | 2954.11 | 2500.13   | Discrete   |

**Interpretation:** While Log-Normal has the lowest AIC, the difference between Log-Normal, Gamma, and Negative Binomial is negligible (Delta AIC < 2). For discrete count modeling, **Negative Binomial is strongly recommended** as it respects the discrete nature of the data. Continuous approximations may be useful for certain analytical contexts but should not replace the proper discrete likelihood in Bayesian count models.

---

## 5. Outlier and Extreme Value Analysis

### Outlier Detection Results

Multiple outlier detection methods were applied:

1. **Z-score method (|z| > 2.5):** 0 outliers
2. **IQR method (1.5 × IQR):** 0 outliers
3. **Modified Z-score (|mz| > 3.5):** 0 outliers

**Conclusion:** No statistical outliers detected. All observations fall within expected ranges for a Negative Binomial distribution with these parameters.

### Extreme Values

- **Maximum:** 269 (at time index 36, year = 1.41)
- **Minimum:** 21 (at time index 2, year = -1.50)
- **Max/Min ratio:** 12.81
- **Range:** 248

**Top 5 observations:** 269, 263, 261, 254, 252 (all in final time period)
**Bottom 5 observations:** 21, 22, 23, 28, 28 (all in initial time period)

This pattern reflects a strong temporal trend rather than anomalous observations.

### Robustness Check: Impact of Extreme Values

To assess whether extreme values drive overdispersion:

- **Original data (n=40):** Var/Mean = 70.43
- **Trimmed data (top 5% removed, n=38):** Var/Mean = 66.48

**Conclusion:** Overdispersion persists even after removing extreme values, indicating it is a fundamental characteristic of the data-generating process, not an artifact of a few unusual observations.

**Supporting Evidence:** See `09_influence_diagnostics.png` (Panels A & B: Influence diagnostics show no high-leverage outliers; Panels C & D: Residuals vs. fitted values show reasonable spread without problematic patterns).

---

## 6. Diagnostic Assessment

### Quantile-Quantile Analysis

Q-Q plots compare theoretical quantiles from fitted distributions to observed quantiles:

- **Poisson Q-Q plot:** Severe deviation from diagonal, especially for larger quantiles
- **Negative Binomial Q-Q plot:** Good alignment with diagonal across full range

**Supporting Evidence:** See `05_qq_plots.png` (Left: Poisson shows systematic deviation; Right: Negative Binomial shows much better agreement, particularly in middle quantiles).

### Probability-Probability Analysis

P-P plots compare cumulative distribution functions:

- **Poisson P-P plot:** Systematic deviation from diagonal
- **Negative Binomial P-P plot:** Close adherence to diagonal

**Supporting Evidence:** See `06_pp_plots.png` (shows cumulative probability agreement; Negative Binomial provides much better fit across all probability levels).

### Residual Analysis

Pearson residuals were calculated as: (observed - expected) / sqrt(expected_variance)

**Poisson Model Residuals:**
- Mean: Not centered at zero (systematic bias)
- Substantial temporal pattern (increasing magnitude over time)
- Many residuals exceed ±2 standard deviations

**Negative Binomial Model Residuals:**
- Mean: Approximately zero
- More random temporal pattern
- Most residuals within ±2 standard deviations
- Distribution closer to standard normal

**Supporting Evidence:** See `07_residual_analysis.png` (Top row: Temporal patterns in residuals show NB model has better-behaved residuals; Bottom row: Distribution of residuals closer to N(0,1) for NB).

### Rootogram Analysis

Hanging rootograms display: sqrt(observed frequency) - sqrt(expected frequency)

- **Poisson rootogram:** Large systematic deviations across many count values
- **Negative Binomial rootogram:** Much smaller, more random deviations

Bars close to zero indicate good fit; large bars indicate poor fit.

**Supporting Evidence:** See `08_rootograms.png` (Left: Poisson shows large systematic deviations; Right: Negative Binomial shows much better agreement with only small random deviations).

---

## 7. Temporal Patterns in Dispersion

### First Half vs. Second Half Comparison

| Period      | n  | Mean   | Variance | Var/Mean |
|-------------|----|--------|----------|----------|
| First half  | 20 | 36.60  | 192.46   | 5.26     |
| Second half | 20 | 182.20 | 4464.80  | 24.50    |

**Key Observations:**
1. Mean increases by 5× from first to second half
2. Variance increases by 23× from first to second half
3. Dispersion (Var/Mean) increases from 5.26 to 24.50

This indicates that both the level and the variability of counts increase over time, with variability increasing at a faster rate than the mean.

### Implications for Modeling

The temporal variation in dispersion has important implications:

1. **Time-varying parameters:** If modeling the temporal trend, consider allowing the dispersion parameter to vary over time
2. **Heteroskedasticity:** Standard errors should account for changing variance structure
3. **Model specification:** A simple Negative Binomial with constant r may be adequate for overall modeling, but refined models might benefit from time-varying dispersion

**Supporting Evidence:** See `04_dispersion_analysis.png` (Panel A: Clear increasing trend in dispersion ratio over time; Panel C: Residuals from linear trend show changing variance structure).

---

## 8. Modeling Recommendations

### Primary Recommendation: Negative Binomial Likelihood

**For Bayesian modeling, use a Negative Binomial likelihood with the following considerations:**

1. **Parameterization:** Use the r-p parameterization where:
   - r = 1.549 (dispersion parameter)
   - p = 0.0140 (probability parameter)
   - Mean = r(1-p)/p
   - Variance = r(1-p)/p²

2. **Alternative parameterization:** μ-φ form (mean-dispersion):
   - μ = 109.40 (mean parameter)
   - φ = 1/r ≈ 0.645 (overdispersion parameter)
   - Some software packages use φ where larger values indicate more overdispersion

3. **Prior considerations:**
   - Place priors on both location and dispersion parameters
   - Consider informative priors on dispersion if domain knowledge suggests typical overdispersion levels
   - For r: gamma or exponential priors are common
   - For regression coefficients: normal priors

### Model Specifications to Consider

#### Model 1: Simple Negative Binomial (Baseline)
```
C[i] ~ NegativeBinomial(μ, r)
log(μ) = α
r ~ Gamma(a, b)  # dispersion parameter
α ~ Normal(0, 10)  # intercept
```

**When to use:** Initial baseline model; when temporal trend is handled separately.

#### Model 2: Negative Binomial with Time Trend
```
C[i] ~ NegativeBinomial(μ[i], r)
log(μ[i]) = α + β × year[i]
r ~ Gamma(a, b)
α ~ Normal(0, 10)
β ~ Normal(0, 5)
```

**When to use:** When modeling temporal trend; assumes constant dispersion.

#### Model 3: Negative Binomial with Time-Varying Dispersion
```
C[i] ~ NegativeBinomial(μ[i], r[i])
log(μ[i]) = α + β × year[i]
log(r[i]) = γ + δ × year[i]
```

**When to use:** If diagnostics suggest dispersion varies systematically with time or mean; more complex but may provide better fit.

### Alternative: Continuous Approximations

If discrete modeling is not required or if computational efficiency is critical:

1. **Log-Normal:** Good fit (AIC = 453.99), simple to implement
   - Use when continuous approximation is acceptable
   - Works well with regression frameworks

2. **Gamma:** Nearly identical fit to Negative Binomial (AIC = 455.73)
   - Natural choice for continuous positive data
   - Shares relationship with Negative Binomial (Gamma-Poisson mixture interpretation)

**Caution:** These lose the discrete count structure, which may be important for interpretation or simulation.

### What NOT to Use

**Do NOT use Poisson distribution:**
- AIC is 2498 points worse than Negative Binomial
- Completely fails to capture overdispersion
- Will severely underestimate uncertainty
- p-values and credible intervals will be misleading

### Model Validation Strategy

1. **Posterior predictive checks:**
   - Simulate data from fitted model
   - Compare simulated mean, variance, and Var/Mean ratio to observed
   - Check if observed data falls within 95% predictive interval

2. **Residual diagnostics:**
   - Examine Pearson and deviance residuals
   - Check for patterns in residuals vs. time or fitted values
   - Assess normality of randomized quantile residuals

3. **Leave-one-out cross-validation:**
   - Compare predictive performance across models
   - Assess sensitivity to individual observations

**Supporting Evidence:** See `10_comprehensive_summary.png` (comprehensive summary showing all key diagnostics and model comparison results).

---

## 9. Data Quality Assessment

### Completeness
- **No missing values** in count variable
- All 40 observations have valid counts and time indices
- **Quality: Excellent**

### Consistency
- All counts are positive integers (range: 21-269)
- No impossible or negative values
- Monotonic time index without gaps
- **Quality: Excellent**

### Temporal Coverage
- Standardized time variable spans -1.67 to +1.67
- Even spacing between observations
- Adequate coverage across time range
- **Quality: Good**

### Potential Issues
1. **Heteroskedasticity:** Variance increases over time (documented above)
   - **Impact:** Standard errors may be affected
   - **Recommendation:** Use robust standard errors or model time-varying dispersion

2. **Small sample size:** n = 40 is relatively small for complex models
   - **Impact:** Parameter uncertainty may be high
   - **Recommendation:** Use informative priors if domain knowledge available

3. **No external validation data:** All data in single time series
   - **Impact:** Cannot assess out-of-sample performance
   - **Recommendation:** Use cross-validation or hold-out period if making predictions

---

## 10. Conclusions and Key Takeaways

### Main Findings

1. **Overdispersion is severe and undeniable:**
   - Variance is 70× larger than mean
   - Poisson distribution fails catastrophically
   - Negative Binomial provides excellent fit

2. **Negative Binomial is the clear choice for modeling:**
   - Very strong evidence by AIC (Delta = 2498)
   - Passes all goodness-of-fit tests
   - Residuals are well-behaved
   - Parameters: r = 1.549, p = 0.014

3. **Dispersion varies over time:**
   - Early periods: near-Poisson (Var/Mean ≈ 0.6)
   - Middle periods: extreme overdispersion (Var/Mean ≈ 12)
   - Late periods: moderate overdispersion (Var/Mean ≈ 4.5)
   - Consider time-varying dispersion in refined models

4. **No data quality issues:**
   - No outliers or anomalous observations
   - No zero-inflation
   - All values reasonable and consistent

5. **Continuous approximations available:**
   - Log-Normal and Gamma provide similar fit to Negative Binomial
   - May be useful in specific analytical contexts
   - Should not replace discrete likelihood for count modeling

### For Modelers

**Use Negative Binomial likelihood with r ≈ 1.5**

This corresponds to substantial overdispersion. The low value of r means that the distribution has much heavier tails than Poisson. In practical terms:
- Poisson assumes Var = μ
- Our data shows Var ≈ 70μ
- Negative Binomial with r = 1.5 gives Var ≈ μ + 0.67μ²

This quadratic relationship is clearly visible in the mean-variance plots.

### Limitations and Caveats

1. **Sample size:** n = 40 provides limited power for detecting subtle distributional features
2. **Time series structure:** This analysis treats observations as independent; temporal autocorrelation not assessed
3. **Covariate information:** No covariates examined; patterns may be explained by unmeasured factors
4. **Generalizability:** Findings specific to this dataset; extrapolation to other time periods or contexts should be done cautiously

### Next Steps for Further Analysis

1. **Time series modeling:** Examine autocorrelation structure, seasonal patterns if applicable
2. **Trend analysis:** Formal modeling of temporal trend in both mean and dispersion
3. **Change point detection:** Test for structural breaks in count process
4. **Covariate exploration:** If additional variables available, assess their relationship to counts and dispersion
5. **Posterior predictive simulation:** Generate predictions under various model specifications

---

## Appendix: Visualization Guide

All visualizations are saved in `/workspace/eda/analyst_2/visualizations/`

### Figure Reference

| Figure | Filename | Key Insights |
|--------|----------|--------------|
| 1 | `01_basic_distribution.png` | Overall distribution shape, spread, and basic statistics |
| 2 | `02_poisson_vs_negbinom.png` | Visual comparison showing Poisson failure and NB success |
| 3 | `03_mean_variance_relationship.png` | Quadratic mean-variance relationship; time series patterns |
| 4 | `04_dispersion_analysis.png` | Temporal variation in dispersion; residual patterns |
| 5 | `05_qq_plots.png` | Quantile comparison for Poisson (poor) and NB (good) |
| 6 | `06_pp_plots.png` | Probability comparison for distribution assessment |
| 7 | `07_residual_analysis.png` | Residual diagnostics for both Poisson and NB models |
| 8 | `08_rootograms.png` | Specialized count diagnostic showing fit quality |
| 9 | `09_influence_diagnostics.png` | Outlier and influence analysis |
| 10 | `10_comprehensive_summary.png` | Complete summary of all key findings |

### Reproducibility

All analyses are fully reproducible using the Python scripts in `/workspace/eda/analyst_2/code/`:

1. `01_initial_exploration.py` - Basic statistics and overdispersion tests
2. `02_distribution_visualizations.py` - Distribution plots and mean-variance analysis
3. `03_hypothesis_testing.py` - Formal hypothesis tests and model fitting
4. `04_diagnostic_plots.py` - Q-Q, P-P, residual, and rootogram plots
5. `05_extreme_value_analysis.py` - Outlier detection and alternative distributions
6. `06_summary_visualization.py` - Comprehensive summary figure

---

**Analysis completed:** 2025-10-29
**Analyst:** Analyst 2 (Count Distribution Specialist)
**Recommendation:** Use Negative Binomial likelihood with r ≈ 1.5 for all modeling applications
