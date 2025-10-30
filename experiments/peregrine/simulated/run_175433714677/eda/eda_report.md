# Exploratory Data Analysis Report

## Dataset Overview

- **Source**: data.json
- **Observations**: n = 40
- **Variables**:
  - `C`: Count data (range: 21-269)
  - `year`: Standardized time variable (range: -1.67 to 1.67)
- **Structure**: Time series of counts showing strong growth trend

---

## Key Findings

### 1. Severe Overdispersion (CRITICAL)

All three analysts independently confirmed **severe overdispersion**:
- Variance-to-mean ratio: **~70** (should be ~1 for Poisson)
- Dispersion parameter: φ ≈ **1.5**

**Implication**: Poisson distribution is completely inappropriate. **Negative Binomial likelihood is required** for all models.

### 2. Heteroscedastic Variance Structure

Variance is **not constant** over time:
- Early period (year < 0): Var/Mean ≈ 0.58-4.5
- Middle period: Var/Mean ≈ 11.85 (peak overdispersion)
- Late period: Var/Mean ≈ 4.4

Levene's test confirms heteroscedasticity (p = 0.005-0.010 across analysts).

### 3. Functional Form: Linear vs. Quadratic

**Two viable perspectives emerged**:

**Perspective A (Exponential Growth - Analyst 3)**:
- Log-linear model: log(μ) = 4.36 + 0.85×year
- Exponential growth rate: 135% per year
- Simpler, interpretable as steady exponential growth

**Perspective B (Accelerating Growth - Analyst 1)**:
- Quadratic model: log(μ) = β₀ + β₁×year + β₂×year²
- R² = 0.96 with quadratic term
- Captures acceleration in growth rate
- Evidence of regime shift at year ≈ -0.21 (Chow test p < 0.000001)
- Growth rate accelerates 9.6x from early to late period

**Resolution**: Both should be tested. Model comparison via LOO-CV will determine which captures the data better.

### 4. No Temporal Autocorrelation

After accounting for trend, **residual autocorrelation is minimal** (r = 0.14, p = 0.37).
- Temporal dependencies are fully explained by the trend
- No need for ARIMA or autoregressive components

### 5. Data Quality

- **No outliers** detected (multiple methods agree)
- **No zero-inflation** (0 zeros observed)
- **No missing values**
- All observations consistent with Negative Binomial distribution

---

## Recommended Model Classes

Based on convergent findings from all analysts:

### Model 1: Log-Linear Negative Binomial (Baseline)
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i]
```
- **Rationale**: Simplest model, exponential growth interpretation
- **Expected parameters**: β₀ ≈ 4.3, β₁ ≈ 0.85, φ ≈ 1.5

### Model 2: Quadratic Negative Binomial
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```
- **Rationale**: Captures acceleration in growth rate (R² = 0.96)
- **Test**: Does quadratic term significantly improve fit?

### Model 3: Piecewise Negative Binomial (Optional)
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × Regime[i] + β₃ × (year × Regime)[i]
```
- **Rationale**: Explicit regime shift at year ≈ -0.21
- **Test**: If scientifically meaningful, tests 9.6x growth acceleration

### Model 4: Time-Varying Dispersion (Advanced)
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i]
log(φ[i]) = γ₀ + γ₁ × year[i]
```
- **Rationale**: Dispersion varies 20x across time periods
- **Test**: Does time-varying dispersion improve calibration?

---

## Prior Recommendations

Based on EDA parameter estimates:

```
# Log-Linear Model
β₀ ~ Normal(4.3, 1.0)    # Intercept near observed log-mean
β₁ ~ Normal(0.85, 0.3)   # Positive growth, moderate uncertainty
φ ~ Exponential(0.05)    # Weakly informative, φ̂ ≈ 1.5

# Quadratic Model
β₀ ~ Normal(4.3, 1.0)
β₁ ~ Normal(0.85, 0.3)
β₂ ~ Normal(0, 0.5)      # Centered at 0 to test necessity
φ ~ Exponential(0.05)
```

---

## Model Comparison Strategy

1. **Fit all models** using Stan/CmdStanPy with MCMC
2. **Check convergence**: R̂ < 1.01, ESS > 400
3. **Posterior predictive checks**:
   - Variance-to-mean ratio recovery
   - Calibration of prediction intervals
4. **LOO-CV comparison**: Use `az.compare()` for ΔELPD ± SE
5. **Parsimony rule**: Prefer simpler model if ΔELPD < 2×SE

---

## Critical Modeling Constraints

### Must Do:
✓ Use Negative Binomial likelihood (NOT Poisson)
✓ Use log link function
✓ Include log_likelihood in generated quantities for LOO-CV
✓ Check variance-to-mean ratio in posterior predictive checks
✓ Account for heteroscedasticity in diagnostics

### Must NOT Do:
✗ Use Poisson (will severely underestimate uncertainty)
✗ Use identity link (can produce negative predictions)
✗ Ignore overdispersion
✗ Extrapolate beyond year ∈ [-1.67, 1.67]

---

## Visual Evidence

Key visualizations supporting findings:

1. **Growth patterns**: `eda/analyst_1/visualizations/02_growth_models.png`
   - Comparison of linear, quadratic, cubic, exponential fits
   - Evidence for non-linear acceleration

2. **Overdispersion**: `eda/analyst_2/visualizations/03_mean_variance_relationship.png`
   - Clear quadratic mean-variance relationship
   - Poisson vs. Negative Binomial comparison

3. **Regime shift**: `eda/analyst_1/visualizations/04_changepoint_analysis.png`
   - Change point at year ≈ -0.21
   - 9.6x growth acceleration

4. **Distribution fit**: `eda/analyst_2/visualizations/02_poisson_vs_negbinom.png`
   - Negative Binomial KS test: p = 0.261 (good fit)
   - Poisson KS test: p < 0.001 (rejected)

---

## Summary Statistics

```
Count Variable (C):
  Mean: 120.35
  Median: 82.5
  SD: 87.29
  Min: 21
  Max: 269
  Skewness: Positive (right-skewed)

Year Variable:
  Mean: 0.0 (standardized)
  SD: 1.0 (standardized)
  Range: [-1.67, 1.67]

Overdispersion:
  Var/Mean ratio: 70.43
  Dispersion parameter φ̂: 1.549
```

---

## Conclusion

The EDA provides **strong, convergent evidence** across three independent analyses:

1. **Negative Binomial likelihood** is required (φ ≈ 1.5)
2. **Log link function** appropriate for count GLM
3. **Functional form** decision between linear (exponential growth) and quadratic (accelerating growth) requires Bayesian model comparison
4. **No temporal autocorrelation** after accounting for trend
5. **Excellent data quality** with no outliers or missing values

The next phase should test Models 1-2 (potentially 3-4) using Stan with MCMC, followed by LOO-CV comparison to select the best model.

---

## References to Detailed Reports

- **Time Series Analysis**: `eda/analyst_1/findings.md` (568 lines)
- **Distribution Analysis**: `eda/analyst_2/findings.md` (comprehensive)
- **Regression Structure**: `eda/analyst_3/findings.md` (11 sections)
- **Synthesis**: `eda/synthesis.md` (convergent/divergent findings)
