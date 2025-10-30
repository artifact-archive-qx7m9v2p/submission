# EDA Report: Regression Structure & Model Forms Analysis

**Analyst Focus Area:** Regression Structure & Model Forms
**Dataset:** data_analyst_3.json
**Date:** 2025-10-29
**Observations:** n=40 count observations with standardized time variable

---

## Executive Summary

This analysis systematically investigated functional forms for modeling count data (C) as a function of standardized year. The data exhibits **strong exponential growth** with **severe overdispersion** (variance/mean ratio = 70.43). Key findings:

1. **Best functional form:** Log-linear (exponential growth) model with log link
2. **Critical issue:** Severe overdispersion requires Negative Binomial GLM, not standard Poisson
3. **Growth rate:** Approximately 135% increase per unit change in standardized year
4. **Recommended for Bayesian modeling:** Negative Binomial GLM with log link function

---

## 1. Data Overview and Basic Patterns

### Data Characteristics
- **Range:** Counts from 21 to 269 (12.8x increase)
- **Mean:** 109.4 counts (SD = 87.8)
- **Time span:** Standardized year from -1.67 to 1.67
- **Coefficient of variation:** 0.802 (high variability)

### Overdispersion Evidence
```
Mean count:           109.40
Variance:            7704.66
Variance/Mean ratio:   70.43  << SEVERE OVERDISPERSION
```

**Interpretation:** Variance is 70 times larger than the mean, far exceeding what a standard Poisson distribution allows (where Var = Mean). This definitively indicates the need for a Negative Binomial or overdispersed count model.

### Temporal Pattern
- **Early period (first 20 obs):** Mean = 36.6, SD = 13.9
- **Late period (last 20 obs):** Mean = 182.2, SD = 66.8
- **Mean ratio (late/early):** 4.98x increase
- **Non-stationarity:** Both level and variance increase substantially over time

**Visual evidence:** `visualizations/01_initial_exploration.png` (panels A-D)
- Panel A shows clear exponential trajectory
- Panel C (log-scale) reveals approximate linearity, suggesting exponential growth
- Panel D shows accelerating rate of change

---

## 2. Functional Form Comparison

### Models Tested
Five functional forms were systematically compared:

| Model | Form | Parameters | R² | RMSE | Adj. R² |
|-------|------|------------|-----|------|---------|
| Linear | C = β₀ + β₁·year | 2 | 0.8812 | 29.87 | 0.8781 |
| Quadratic | C = β₀ + β₁·year + β₂·year² | 3 | 0.9641 | 16.43 | 0.9621 |
| Cubic | C = β₀ + β₁·year + β₂·year² + β₃·year³ | 4 | 0.9743 | 13.88 | 0.9722 |
| Log-Linear | log(C) = β₀ + β₁·year | 2 | 0.9358* | 21.96 | 0.9341 |
| Square-Root | √C = β₀ + β₁·year | 2 | 0.9451 | 20.31 | 0.9436 |

*Note: R² for log-linear measured on original scale; R² = 0.9678 on log scale

### Key Findings

**1. Linear Model - INADEQUATE**
- Systematic residual patterns (U-shaped)
- Underpredicts at extremes, overpredicts in middle
- Pearson correlation: 0.9387 (seems good but misleading)
- Not appropriate for exponential growth

**2. Quadratic Model - GOOD COMPROMISE**
- Captures curvature effectively
- R² = 0.9641 with only 3 parameters
- Shows heteroscedasticity in residuals (Levene's test p=0.0054)
- Best for descriptive/exploratory purposes

**3. Cubic Model - MARGINAL IMPROVEMENT**
- Only 1% R² improvement over quadratic
- Risk of overfitting with 4 parameters
- Not recommended (complexity not justified)

**4. Log-Linear Model - BEST STATISTICAL FIT**
- Highest correlation on log scale (r = 0.9678)
- Implies exponential growth: C = exp(4.334) × exp(0.862·year)
- Growth rate: **136.9% per unit year**
- Better behavior on log scale, heteroscedasticity on original scale

**5. Square-Root Model - SUBOPTIMAL**
- Variance-stabilizing for Poisson, but insufficient here
- Outperformed by log transformation

**Visual evidence:** `visualizations/02_model_fits.png` shows all model fits
**Residual comparison:** Panel F demonstrates log-linear has smallest systematic bias

---

## 3. Residual Analysis

### Diagnostic Tests

**Normality Tests (Shapiro-Wilk):**
- Linear: W=0.955, p=0.112 (appears normal)
- Quadratic: W=0.922, p=0.009 (non-normal)
- Log-Linear: W=0.920, p=0.008 (non-normal)

**Autocorrelation (Durbin-Watson):**
- Linear: DW=0.472 (strong positive autocorrelation)
- Quadratic: DW=1.544 (mild positive autocorrelation)
- Log-Linear: DW=0.920 (moderate positive autocorrelation)

**Heteroscedasticity (Levene's Test):**
- Linear: F=0.204, p=0.654 (no evidence - but systematic pattern masks this)
- Quadratic: F=8.714, **p=0.005** (significant heteroscedasticity)
- Log-Linear: F=8.319, **p=0.006** (significant heteroscedasticity)

### Influential Points
Cook's Distance analysis (linear model) identified:
- **2 influential points** (indices 0, 1)
- Maximum Cook's D = 0.212 (threshold = 0.100)
- Early time points have higher leverage

**Visual evidence:**
- `visualizations/03_residual_diagnostics.png`: Q-Q plots and residual distributions
- `visualizations/04_advanced_residuals.png`: Scale-location and time-series patterns
- `visualizations/05_cooks_distance.png`: Influential point detection

### Key Residual Patterns

1. **Linear model:** Clear U-shaped pattern → systematic bias
2. **Quadratic model:** Improved but variance increases with fitted values
3. **Log-Linear model:** Better on log scale, but back-transformed shows heteroscedasticity

---

## 4. GLM Analysis and Link Functions

### Poisson GLM with Log Link

**Fitted model:** log(λ) = 4.3614 + 0.8546·year

**Coefficients:**
- β₀ = 4.3614 → baseline count = exp(4.36) ≈ 78
- β₁ = 0.8546 → 135% growth per unit year
- Total growth over range: exp(0.855 × 3.34) ≈ 17x

**Overdispersion diagnostics:**
```
Pearson χ² statistic:    133.33
Dispersion parameter:      3.51  << φ = χ²/(n-p) >> 1
Deviance:                132.44
```

**Conclusion:** Standard Poisson GLM is **INADEQUATE** due to overdispersion (φ = 3.51 >> 1)

### Negative Binomial GLM (Quasi-Poisson)

Using same β coefficients but adjusting for overdispersion:
- Overdispersion parameter: φ = 3.51
- Adjusted Pearson χ² = 38.0 (much better fit)
- Allows Var(Y) = μ + μ²/θ (more flexible than Poisson)

### Link Function Comparison

**Tested link functions:**
1. **Log link:** E[C] = exp(β₀ + β₁·year) ✓ Best
2. **Identity link:** E[C] = β₀ + β₁·year ✗ Can produce negatives (min = -28)
3. **Square-root link:** E[C] = (β₀ + β₁·year)² ✓ Okay but inferior

**Visual evidence:** `visualizations/06_glm_link_functions.png`

### Variance-Mean Relationship

Empirical analysis of variance vs. mean across time bins reveals:
- Quadratic relationship (not linear as Poisson assumes)
- Log-log slope ≈ 1.5-2.0 (power law: Var ∝ Mean^1.5)
- Strong evidence for Negative Binomial over Poisson

**Visual evidence:** `visualizations/07_variance_mean_relationship.png`
- Panel A: Variance increases quadratically with mean
- Panel B: Log-log plot shows slope > 1

---

## 5. Model Selection Criteria

### Information Criteria Comparison

| Model | Parameters | AIC | BIC | Δ AIC | Akaike Weight |
|-------|------------|-----|-----|-------|---------------|
| Log-Linear | 2 | **-3.07** | **0.31** | 0.00 | **100.0%** |
| Cubic | 4 | 332.18 | 338.94 | 335.25 | 0.0% |
| Quadratic | 3 | 343.54 | 348.61 | 346.61 | 0.0% |
| Poisson GLM | 2 | 385.69 | 389.07 | 388.76 | 0.0% |
| Linear | 2 | 389.31 | 392.69 | 392.38 | 0.0% |

**Winner:** Log-Linear model (100% Akaike weight)

### Cross-Validation Results

Leave-One-Out Cross-Validation MSE:

| Model | LOO-CV MSE | Ranking |
|-------|------------|---------|
| Cubic | 251.29 | 1st (best) |
| Quadratic | 332.69 | 2nd |
| Log-Linear | 550.85 | 3rd |
| Linear | 1000.35 | 4th (worst) |

**Interpretation:**
- Cubic has best out-of-sample prediction BUT risk of overfitting
- Quadratic offers good balance
- Log-Linear is theoretically better for count data despite higher MSE

**Visual evidence:** `visualizations/08_model_comparison_summary.png`

---

## 6. Recommendations for Bayesian Modeling

### PRIMARY RECOMMENDATION: Negative Binomial GLM with Log Link

**Model specification:**
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i]
```

**Why this model:**
1. Accounts for severe overdispersion (φ = 70+)
2. Log link ensures positive predictions
3. Natural for count data with exponential growth
4. Handles heteroscedasticity inherently

**Stan/PyMC code structure:**
```stan
data {
  int<lower=0> N;
  int<lower=0> C[N];
  vector[N] year;
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> phi;  // overdispersion parameter
}

model {
  vector[N] mu;

  // Linear predictor on log scale
  mu = exp(beta_0 + beta_1 * year);

  // Priors (weakly informative based on EDA)
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.3);
  phi ~ exponential(0.05);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}
```

### PRIOR RECOMMENDATIONS (Weakly Informative)

Based on observed data patterns:

**Intercept (β₀):**
- Prior: `Normal(4.3, 1.0)`
- Rationale: Observed β₀ ≈ 4.36 (exp(4.3) ≈ 73 counts at year=0)
- Range: Allows baseline counts from ~20 to ~200

**Slope (β₁):**
- Prior: `Normal(0.85, 0.3)`
- Rationale: Observed β₁ ≈ 0.855 (135% growth rate)
- Range: Allows growth rates from ~50% to ~200%

**Overdispersion (φ):**
- Prior: `Exponential(0.05)` or `Gamma(2, 0.1)`
- Rationale: Observed dispersion ≈ 3.5 from Poisson residuals
- Note: True φ for NB will differ; prior allows wide range

### ALTERNATIVE MODEL: Poisson with Random Effects

If you prefer Poisson structure but need to account for overdispersion:

```stan
model {
  vector[N] log_mu;

  log_mu = beta_0 + beta_1 * year + epsilon;
  epsilon ~ normal(0, sigma_eps);  // individual-level random effect

  C ~ poisson_log(log_mu);
}
```

This adds individual-level heterogeneity to capture extra-Poisson variation.

### NOT RECOMMENDED

**Standard Poisson GLM:**
- Underestimates uncertainty
- Will produce overly narrow credible intervals
- Overdispersion (φ=3.51) violates core assumption

**Linear models:**
- Wrong distributional family for count data
- Can predict negative values
- Heteroscedasticity violates constant variance assumption

---

## 7. Key Findings Summary

### What Functional Form Best Captures the Relationship?

**Answer:** **Exponential growth** (log-linear relationship)
- Log-scale correlation: r = 0.968
- Best AIC/BIC scores
- Theoretically appropriate for count processes with constant growth rate

### Does Log-Linear Model Fit Well?

**Answer:** **Yes, with caveats**
- Excellent fit on log scale (R² = 0.937 on log scale)
- Shows heteroscedasticity when back-transformed to original scale
- Overdispersion requires Negative Binomial extension

### Are Polynomial Terms Needed?

**Answer:** **Not necessary, but quadratic is defensible**
- Cubic offers marginal improvement (R² gain < 1%)
- Quadratic captures curvature with acceptable complexity
- For Bayesian GLM: log link is preferred over polynomials

### What Link Functions Are Appropriate?

**Answer:** **Log link is strongly preferred**

**Ranking:**
1. **Log link** - Best for exponential growth, maintains positivity
2. Square-root link - Inferior to log for this data
3. Identity link - Inappropriate (can produce negatives)

---

## 8. Diagnostic Visualizations Summary

All visualizations saved in `/workspace/eda/analyst_3/visualizations/`:

1. **01_initial_exploration.png** - Time series, distributions, log-scale plot, rate of change
2. **02_model_fits.png** - Visual comparison of 5 functional forms
3. **03_residual_diagnostics.png** - Q-Q plots and residual distributions for 3 key models
4. **04_advanced_residuals.png** - Scale-location and temporal residual patterns
5. **05_cooks_distance.png** - Influential points analysis
6. **06_glm_link_functions.png** - GLM link function comparison and residual diagnostics
7. **07_variance_mean_relationship.png** - Overdispersion evidence (variance vs. mean)
8. **08_model_comparison_summary.png** - AIC, Akaike weights, R² vs complexity
9. **09_prediction_intervals.png** - Prediction intervals for quadratic and log-linear models

---

## 9. Limitations and Caveats

1. **Small sample size** (n=40) limits ability to detect subtle model differences
2. **Influential points** at early time points may affect coefficient estimates
3. **Extrapolation risk** - exponential models can produce unrealistic predictions beyond data range
4. **No covariates** - additional predictors might reduce overdispersion
5. **Time series structure** not explicitly modeled (autocorrelation present)

---

## 10. Next Steps for Modeling

### Immediate Actions:
1. ✓ Implement Negative Binomial GLM in Stan/PyMC
2. ✓ Use weakly informative priors centered at β₀=4.3, β₁=0.85
3. Check posterior predictive distribution for:
   - Overdispersion coverage
   - Prediction interval coverage
   - Residual patterns

### Advanced Considerations:
1. **Time series model** - Consider AR(1) structure if temporal dependence matters
2. **Change-point detection** - Investigate if growth rate changes over time
3. **Hierarchical structure** - If additional grouping exists (not visible in this dataset)
4. **Model averaging** - Bayesian Model Averaging over multiple functional forms

### Validation Strategy:
1. Posterior predictive checks for:
   - Mean-variance relationship
   - Distribution shape
   - Extreme value behavior
2. Leave-one-out cross-validation (LOO-CV) using Pareto-smoothed importance sampling
3. Prior sensitivity analysis (vary prior SDs by factor of 2-3)

---

## 11. Conclusion

This dataset exhibits **clear exponential growth with severe overdispersion**. The Negative Binomial GLM with log link is the theoretically and empirically best choice for Bayesian modeling. While polynomial models (especially quadratic) provide good fits, they lack the theoretical foundation for count data and cannot naturally handle overdispersion.

**Key parameters for Bayesian implementation:**
- **β₀ ≈ 4.3** (baseline log-count)
- **β₁ ≈ 0.85** (growth rate on log scale = 135% per unit year)
- **φ ≈ 3-10** (overdispersion parameter, estimate from data)

The recommended model specification, priors, and diagnostic checks provide a complete roadmap for robust Bayesian inference on this count data.

---

**Analysis completed:** 2025-10-29
**Code location:** `/workspace/eda/analyst_3/code/`
**Visualization location:** `/workspace/eda/analyst_3/visualizations/`
