# EDA Synthesis: Convergent and Divergent Findings

## Executive Summary

Three parallel analysts explored the count data from complementary perspectives: time series patterns, count distributions, and regression structures. Their findings show **strong convergence** on key properties and modeling approaches.

---

## Convergent Findings (High Confidence)

### 1. Severe Overdispersion (All 3 Analysts Agree)
- **Analyst 1**: Variance-to-mean ratio = 68.67
- **Analyst 2**: Variance-to-mean ratio = 70.43
- **Analyst 3**: Variance-to-mean ratio = 70.43
- **Consensus**: Poisson distribution is completely inappropriate; Negative Binomial is required

### 2. Negative Binomial Likelihood (All 3 Analysts Agree)
- **Analyst 1**: "Must use Negative Binomial (NOT Poisson)"
- **Analyst 2**: "NEGATIVE BINOMIAL IS THE CLEAR CHOICE" (r ≈ 1.549, KS test p = 0.261)
- **Analyst 3**: "Negative Binomial GLM with log link" recommended
- **Consensus**: Negative Binomial with dispersion parameter φ ≈ 1.5

### 3. Log-Link Function Appropriate (All 3 Analysts Agree)
- **Analyst 1**: Recommends log(μ) = β₀ + β₁×year + β₂×year²
- **Analyst 2**: Suggests log(μ[i]) for mean function
- **Analyst 3**: "Log link is superior" with exponential growth interpretation
- **Consensus**: Log link function for GLM structure

### 4. Heteroscedasticity Present (All 3 Analysts Agree)
- **Analyst 1**: Variance increases 17x over time (Levene's test p = 0.005)
- **Analyst 2**: Dispersion NOT constant (Levene's test p = 0.010)
- **Analyst 3**: Heteroscedasticity present
- **Consensus**: Variance increases with time/mean

### 5. No Temporal Autocorrelation in Residuals (Analyst 1)
- Residual autocorrelation minimal (0.14, p = 0.37) after accounting for trend
- ARIMA models not needed

---

## Divergent Findings (Requires Reconciliation)

### Linear vs. Quadratic Relationship

**Analyst 1 (Time Series Perspective)**:
- **Strong preference for quadratic**: R² = 0.96 with quadratic term
- Identified regime shift at year ≈ -0.21 with 9.6x growth acceleration
- Recommends: log(μ) = β₀ + β₁×year + β₂×year²
- Evidence: Chow test p < 0.000001 for structural break

**Analyst 3 (Regression Structure Perspective)**:
- **Preference for log-linear (exponential)**: Log(μ) = 4.36 + 0.85×year
- Growth rate: 135% per unit year
- Simpler model with good fit
- Evidence: AIC/BIC favor parsimonious models

**Reconciliation**:
- Both perspectives are valid and should be tested
- Analyst 1's quadratic captures acceleration in growth rate
- Analyst 3's log-linear is simpler and interpretable as exponential growth
- **Resolution**: Test BOTH model classes in Bayesian framework:
  1. Log-linear (exponential growth): simpler baseline
  2. Quadratic/piecewise: more flexible, captures regime change

### Regime Shift vs. Smooth Growth

**Analyst 1**:
- Identifies significant structural break at year ≈ -0.21
- Recommends piecewise model as alternative
- Regime shift has 9.6x acceleration in growth

**Analyst 3**:
- Smooth exponential growth without explicit regime change
- Single log-linear relationship sufficient

**Reconciliation**:
- Regime shift analysis (Analyst 1) is valuable diagnostic
- Question: Is break scientifically meaningful or artifact of polynomial?
- **Resolution**: Test both continuous (polynomial/exponential) and piecewise models

---

## Distribution Properties Summary

| Property | Analyst 1 | Analyst 2 | Analyst 3 | Consensus |
|----------|-----------|-----------|-----------|-----------|
| Overdispersion | Yes (68.67) | Yes (70.43) | Yes (70.43) | ✓ **Severe** |
| Likelihood | Neg. Binomial | Neg. Binomial | Neg. Binomial | ✓ **NB** |
| Link Function | Log | Log | Log | ✓ **Log** |
| Heteroscedasticity | Yes | Yes | Yes | ✓ **Present** |
| Functional Form | Quadratic | - | Log-linear | **Test both** |
| Regime Shift | Yes (year=-0.21) | - | No | **Test both** |
| Zero-inflation | No | No | No | ✓ **None** |
| Outliers | No | No | No | ✓ **None** |

---

## Modeling Hypotheses to Test

Based on the synthesis, we should test the following model classes:

### Model Class 1: Log-Linear Negative Binomial (Baseline)
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i]
```
- **Rationale**: Simplest model, exponential growth interpretation
- **Support**: Analyst 3 primary recommendation
- **Priors**: β₀ ~ N(4.3, 1), β₁ ~ N(0.85, 0.3), φ ~ Exp(0.05)

### Model Class 2: Quadratic Negative Binomial
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```
- **Rationale**: Captures acceleration in growth rate
- **Support**: Analyst 1 primary recommendation (R² = 0.96)
- **Test**: Does β₂ differ significantly from 0?

### Model Class 3: Piecewise Negative Binomial (Regime Shift)
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × Regime[i] + β₃ × (year × Regime)[i]
where Regime[i] = 1 if year[i] > -0.21, else 0
```
- **Rationale**: Explicit modeling of 9.6x growth acceleration
- **Support**: Analyst 1 alternative recommendation (Chow test p < 0.000001)
- **Test**: Scientific interpretability of regime change

### Model Class 4: Time-Varying Dispersion
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i]
log(φ[i]) = γ₀ + γ₁ × year[i]
```
- **Rationale**: Dispersion varies over time (Q1: 0.58 → Q3: 11.85)
- **Support**: Analyst 2 detailed dispersion analysis
- **Test**: Does time-varying dispersion improve fit?

---

## Key Data Properties for Modeling

1. **Sample Size**: n = 40 (moderate, sufficient for 2-4 parameters)
2. **Range**: C ∈ [21, 269], year ∈ [-1.67, 1.67]
3. **Dispersion**: φ ≈ 1.5 (severe overdispersion)
4. **Growth**: Approximately 135% per standardized year unit
5. **No Missing Data**: Complete observations
6. **No Zero-Inflation**: 0 zeros observed
7. **No Outliers**: All points consistent with data-generating process

---

## Critical Modeling Constraints

### Must Do:
- ✓ Use Negative Binomial likelihood (not Poisson)
- ✓ Use log link function
- ✓ Test both linear and quadratic terms
- ✓ Account for heteroscedasticity in model checking
- ✓ Include log_likelihood in posterior for LOO-CV

### Must NOT Do:
- ✗ Use Poisson (will severely underestimate uncertainty)
- ✗ Use identity link (unbounded predictions)
- ✗ Ignore overdispersion
- ✗ Extrapolate beyond data range (especially polynomials)

---

## Recommended Model Testing Order

1. **Start with Log-Linear** (simplest, interpretable baseline)
2. **Test Quadratic** (if log-linear shows poor fit)
3. **Test Piecewise** (if scientific justification for regime shift exists)
4. **Test Time-Varying Dispersion** (if simpler models show poor calibration)

Use LOO-CV for model comparison. Apply parsimony principle: prefer simpler model if ΔELPD < 2×SE.

---

## Visual Evidence Summary

All three analysts generated complementary visualizations:

**Analyst 1**: Time series diagnostics, change point analysis, regime comparisons
**Analyst 2**: Distribution fits (Poisson vs NB), mean-variance relationships, Q-Q plots
**Analyst 3**: Functional form comparisons, GLM diagnostics, prediction intervals

Key plots for reference:
- `eda/analyst_1/visualizations/02_growth_models.png` - Functional form comparison
- `eda/analyst_2/visualizations/03_mean_variance_relationship.png` - Overdispersion evidence
- `eda/analyst_3/visualizations/07_variance_mean_relationship.png` - GLM diagnostics

---

## Conclusion

The three parallel analyses provide **strong, convergent evidence** for:
1. Negative Binomial likelihood with log link
2. Severe overdispersion (φ ≈ 1.5)
3. Heteroscedastic variance structure

The main modeling decision is functional form:
- **Linear** (exponential growth): simpler, interpretable
- **Quadratic**: captures acceleration
- **Piecewise**: explicit regime shift

All should be tested in Bayesian framework with LOO-CV for comparison.
