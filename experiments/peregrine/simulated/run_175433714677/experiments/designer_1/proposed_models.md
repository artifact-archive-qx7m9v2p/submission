# Bayesian Model Design: Parsimonious Approach

**Design Philosophy**: Simplicity, Interpretability, Computational Efficiency

**Designer**: Agent 1 (Parsimony Track)

**Date**: 2025-10-29

---

## Executive Summary

Based on EDA findings, I propose **2 competing model classes** with clear falsification criteria. The key tension is between:
1. **Simple exponential growth** (2 parameters + dispersion)
2. **Accelerating growth** (3 parameters + dispersion)

Both use Negative Binomial likelihood (non-negotiable given φ ≈ 1.5). I will **abandon quadratic models** if the quadratic coefficient's 95% credible interval includes zero with |β₂| < 0.1. I will **abandon linear models** if LOO-CV shows ΔELPD > 4 favoring quadratic.

---

## Problem Formulation

### Competing Hypotheses

**H1: Exponential Growth (Linear on log-scale)**
- Growth rate is constant over time
- Single exponential process drives counts
- Simplest explanation: log(μ) = β₀ + β₁·year

**H2: Accelerating Growth (Quadratic on log-scale)**
- Growth rate increases over time
- Compound process or feedback mechanism
- More complex: log(μ) = β₀ + β₁·year + β₂·year²

### What Would Make Me Reconsider Everything?

1. **Prior-posterior conflict**: If posterior means are >3 SD from EDA estimates
2. **Divergent transitions**: >1% divergences after tuning (suggests model misspecification)
3. **Poor variance recovery**: If posterior predictive Var/Mean ratio is <30 or >120 (EDA: 70)
4. **Extreme dispersion**: If φ posterior is <0.5 or >5 (EDA: 1.5)
5. **Systematic residual patterns**: If pp_check shows clear structure in residuals

**Escape routes if models fail**:
- Time-varying dispersion model (heteroscedasticity is present)
- Piecewise/changepoint model (Chow test suggests regime shift)
- Student-t observation error (if NegBin still shows outliers)

---

## Model 1: Log-Linear Negative Binomial (BASELINE)

### Mathematical Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ)    for i = 1,...,40
log(μ[i]) = β₀ + β₁ × year[i]
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)     # Intercept: log(mean count at year=0)
β₁ ~ Normal(0.85, 0.5)    # Slope: expected log-growth rate
φ  ~ Exponential(0.667)   # Dispersion: E[φ] = 1.5
```

**Stan Parameterization Note**:
Use `neg_binomial_2_log(mu, phi)` where mu is on log-scale and phi is the overdispersion parameter (larger φ = less overdispersion).

### Theoretical Justification

1. **Simplicity**: Only 3 parameters, minimal complexity
2. **Interpretability**: β₁ = instantaneous growth rate, exp(β₁) = multiplicative factor per year
3. **EDA Support**: Linear fit has R² = 0.92, residuals show no strong pattern
4. **Exponential growth**: Common in early population dynamics, technology adoption

### Expected Parameter Values (from EDA)

```
β₀: 4.3 ± 0.15   (log of mean count at standardized year = 0)
β₁: 0.85 ± 0.10  (implies 134% annual growth: exp(0.85) ≈ 2.34)
φ:  1.5 ± 0.5    (overdispersion parameter)
```

### Prior Justification

- **β₀ ~ Normal(4.3, 1.0)**: Centered at EDA estimate, SD=1.0 allows counts from exp(3.3)=27 to exp(5.3)=200 at year=0 (observed: 21-269 across all years)
- **β₁ ~ Normal(0.85, 0.5)**: Centered at EDA estimate, SD=0.5 allows growth rates from 54% to 334% annually (very wide range)
- **φ ~ Exponential(0.667)**: Mean at EDA estimate (1.5), light tail allows adaptation if actual overdispersion differs

### Falsification Criteria

**I will abandon this model if**:

1. **LOO-CV**: ΔELPD > 4 compared to quadratic model (strong evidence against)
2. **Systematic curvature**: Posterior predictive residuals show clear U-shape or inverted-U
3. **Late period failure**: Mean absolute error in last 10 observations is >2× early 10 observations
4. **Variance mismatch**: 95% of posterior predictive Var/Mean ratios fall outside [50, 90]
5. **Poor calibration**: <80% of observations fall within 90% prediction intervals

### Computational Considerations

**Expected performance**:
- **Parameters**: 3 (very fast)
- **Warmup**: ~50 iterations should suffice
- **ESS**: Expect >1000 for all parameters
- **Runtime**: <10 seconds for 4 chains × 1000 iterations
- **Divergences**: Should be 0 (simple model geometry)

**Potential issues**:
- None expected - this is a textbook GLM

---

## Model 2: Quadratic Negative Binomial (ACCELERATING GROWTH)

### Mathematical Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)      # Intercept
β₁ ~ Normal(0.85, 0.5)     # Linear coefficient
β₂ ~ Normal(0, 0.3)        # Quadratic coefficient (centered at 0)
φ  ~ Exponential(0.667)    # Dispersion
```

### Theoretical Justification

1. **EDA Evidence**: R² = 0.96 with quadratic term, Chow test shows regime shift (p < 0.000001)
2. **Growth acceleration**: Growth rate increases 9.6× from early to late period
3. **Scientific plausibility**: Positive feedback loops, network effects, compound growth
4. **Parsimony vs. fit trade-off**: Only +1 parameter, but captures curvature

### Expected Parameter Values

```
β₀: 4.3 ± 0.15
β₁: 0.85 ± 0.10
β₂: 0.3 ± 0.15   (positive = acceleration, negative = deceleration)
φ:  1.5 ± 0.5
```

### Prior Justification

- **β₀, β₁**: Same as Model 1
- **β₂ ~ Normal(0, 0.3)**:
  - Centered at 0 to avoid bias (let data determine sign/magnitude)
  - SD=0.3 allows β₂ ∈ [-0.6, 0.6] with 95% probability
  - At year²=2.8 (max), this gives log(μ) change of ±1.68, or ×0.19 to ×5.4 multiplier
- **φ**: Same as Model 1

### Falsification Criteria

**I will abandon this model if**:

1. **β₂ not significant**: 95% credible interval includes 0 AND |β₂| < 0.1
2. **No LOO improvement**: ΔELPD < 2 compared to linear model (not worth the complexity)
3. **Overfitting signs**: LOO-CV Pareto-k values > 0.7 for >10% of observations
4. **Divergences**: >0.5% divergent transitions (suggests over-parameterization)
5. **Posterior-prior overlap**: If posterior for β₂ is nearly identical to prior (data not informative)

### Computational Considerations

**Expected performance**:
- **Parameters**: 4 (still very fast)
- **Warmup**: ~100 iterations (potential correlation between β₁ and β₂)
- **ESS**: Expect >500 for all parameters (may have some correlation)
- **Runtime**: <15 seconds
- **Divergences**: Low risk, but monitor β₁-β₂ correlation

**Potential issues**:
- **Multicollinearity**: year and year² are correlated (Pearson r ≈ 0.4-0.6 for standardized data)
- **Mitigation**: Use non-centered parameterization if ESS < 400

---

## Model 3: Heteroscedastic Negative Binomial (OPTIONAL - ONLY IF MODELS 1-2 FAIL)

### Mathematical Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i]
log(φ[i]) = γ₀ + γ₁ × year[i]
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)
β₁ ~ Normal(0.85, 0.5)
γ₀ ~ Normal(0.4, 0.5)     # log(1.5) ≈ 0.4
γ₁ ~ Normal(0, 0.3)       # Time-varying dispersion
```

### When to Consider This Model

**Only implement if**:
1. Models 1-2 show poor variance calibration
2. Residual variance shows strong temporal pattern
3. Posterior predictive checks reveal systematic under/over-dispersion by period

### Theoretical Justification

1. **EDA finding**: Var/Mean ratio varies 20× across periods (0.58 early, 11.85 middle, 4.4 late)
2. **Process change**: Data generation mechanism may change over time
3. **Heteroscedasticity**: Levene's test significant (p = 0.005-0.010)

### Falsification Criteria

**I will abandon this model if**:

1. **γ₁ not significant**: 95% CI includes 0 AND |γ₁| < 0.1
2. **No improvement**: ΔELPD < 3 compared to constant-φ models
3. **Computational issues**: >1% divergences or ESS < 200 for any parameter
4. **Over-complexity**: More than 5% of LOO Pareto-k > 0.7

### Computational Considerations

**Expected performance**:
- **Parameters**: 4 (same as quadratic, different structure)
- **Warmup**: ~200 iterations (time-varying dispersion is harder)
- **ESS**: Expect >300 (dispersion parameters often have lower ESS)
- **Runtime**: ~30 seconds
- **Divergences**: Risk: 0.5-2% (dispersion modeling can be tricky)

**Potential issues**:
- **Numerical stability**: log(φ) parameterization helps but monitor for extreme values
- **Identifiability**: μ and φ can trade off in NegBin models

---

## Model Comparison Strategy

### Step 1: Fit Core Models (Sequential)

1. **Fit Model 1 first** (baseline)
   - Check convergence: R̂ < 1.01, ESS > 400
   - Run posterior predictive checks
   - Calculate LOO-CV

2. **Fit Model 2** (if Model 1 shows curvature issues)
   - Compare to Model 1
   - Check if β₂ is significant

3. **Fit Model 3** (only if Models 1-2 fail variance calibration)

### Step 2: Convergence Diagnostics

**Required checks**:
- **R̂ < 1.01** for all parameters
- **ESS_bulk > 400** and **ESS_tail > 400**
- **Divergences < 0.5%** (ideally 0)
- **Tree depth < 10** (no hitting max treedepth)
- **BFMI > 0.3** (energy diagnostic)

### Step 3: Posterior Predictive Checks

**Critical checks**:
1. **Variance-to-mean ratio**: Does posterior predictive match EDA (70 ± 20)?
2. **Calibration**: What % of observations fall in 50%, 80%, 90% prediction intervals?
3. **Residual patterns**: Plot residuals vs. fitted, vs. year - look for structure
4. **Extreme value recovery**: Can model generate values as high as 269?

### Step 4: LOO-CV Comparison

**Decision rules**:
- **ΔELPD < 2**: Models statistically equivalent, choose simpler (Model 1)
- **2 < ΔELPD < 4**: Weak preference, consider scientific justification
- **ΔELPD > 4**: Strong preference for better model
- **Pareto-k > 0.7**: Flag influential observations, investigate

### Step 5: Parameter Interpretation

**For chosen model**:
- Report posterior means, medians, 95% CIs
- Calculate derived quantities (e.g., growth rate at specific years)
- Compare to EDA estimates (should be similar)
- Assess prior sensitivity (if posteriors differ greatly from priors)

---

## Red Flags & Decision Points

### STOP and Reconsider Everything If:

1. **Prior-posterior collapse**: Posteriors unchanged from priors (model not identified)
2. **Extreme parameter values**:
   - |β₀| > 8 (implies mean counts <0.3 or >3000)
   - |β₁| > 2 (implies >600% or <14% annual growth)
   - φ < 0.3 or > 10 (overdispersion out of plausible range)
3. **Computational pathology**:
   - >5% divergences after adapt_delta=0.99
   - ESS < 100 for any parameter
   - R̂ > 1.05
4. **Prediction failure**:
   - Posterior predictive variance < 20 or > 200 (EDA: 70)
   - >20% of observations outside 90% prediction intervals

### Alternative Approaches to Consider:

If all proposed models fail:
1. **Changepoint model**: Explicit regime shift at year ≈ -0.21
2. **Mixture model**: Two latent processes generating counts
3. **Student-t errors**: Heavier tails if NegBin insufficient
4. **Nonparametric**: Gaussian Process for flexible trend
5. **Re-examine data**: Are there recording errors, batch effects, or hidden covariates?

---

## Implementation Requirements

### Stan Code Structure

All models must include:

```stan
data {
  int<lower=0> N;           // Number of observations
  array[N] int<lower=0> y;  // Count outcomes
  vector[N] year;           // Standardized year
}

parameters {
  real beta_0;              // Intercept
  real beta_1;              // Slope
  real<lower=0> phi;        // Dispersion parameter
}

model {
  // Priors
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.5);
  phi ~ exponential(0.667);

  // Likelihood
  for (i in 1:N) {
    y[i] ~ neg_binomial_2_log(beta_0 + beta_1 * year[i], phi);
  }
}

generated quantities {
  vector[N] log_lik;        // For LOO-CV
  vector[N] y_rep;          // For posterior predictive checks

  for (i in 1:N) {
    real mu_i = beta_0 + beta_1 * year[i];
    log_lik[i] = neg_binomial_2_log_lpmf(y[i] | mu_i, phi);
    y_rep[i] = neg_binomial_2_log_rng(mu_i, phi);
  }
}
```

### Sampling Configuration

**Initial run**:
- Chains: 4
- Warmup: 1000
- Sampling: 1000
- adapt_delta: 0.8 (increase to 0.95 if divergences occur)
- max_treedepth: 10

**If convergence issues**:
- Increase warmup to 2000
- Increase adapt_delta to 0.99
- Check for non-centered parameterization opportunities

---

## Stress Tests

### Test 1: Extrapolation Stability
**Goal**: Ensure model doesn't produce absurd predictions outside data range

- Generate predictions for year ∈ [-2, 2]
- Check for explosive growth (>10,000 counts)
- Flag if uncertainty doesn't increase appropriately

### Test 2: Variance Recovery
**Goal**: Confirm model can generate observed overdispersion

- Calculate Var/Mean ratio for each posterior draw
- Check if 95% interval includes 70
- Plot posterior predictive Var/Mean vs. observed

### Test 3: Tail Behavior
**Goal**: Ensure model can generate extreme values

- Check if max(y_rep) across draws includes observed max (269)
- Assess if model systematically under-predicts high values

### Test 4: Residual Randomness
**Goal**: No systematic patterns in residuals

- Plot residuals vs. fitted values (should be random cloud)
- Plot residuals vs. year (should be random)
- Run runs test for randomness (p > 0.05)

---

## Success Criteria

A model is **successful** if:

1. **Converges**: R̂ < 1.01, ESS > 400, <0.5% divergences
2. **Calibrates**: 50% PI contains ~50% obs, 90% PI contains ~90% obs
3. **Recovers variance**: Posterior predictive Var/Mean = 60-80
4. **Interpretable**: Parameters align with EDA estimates (within 2 SD)
5. **Robust**: Predictions stable for slight data perturbations

A model is **adequate** if:
- All success criteria met except LOO-CV shows improvement with added complexity
- Decision: Bayesian Occam's Razor - choose simpler if ΔELPD < 2

A model **fails** if:
- Any convergence diagnostic fails
- Posterior predictive checks show systematic bias
- Parameters are extreme or uninterpretable

---

## Stopping Rules

### When to Stop Exploring:

1. **Success**: Model meets all success criteria
2. **Diminishing returns**: ΔELPD < 1 between last two models
3. **Exhausted reasonable options**: All 3 proposed models tested
4. **Computational limits**: Runtime exceeds available resources

### When to Escalate:

If all models fail, consult with:
- Domain expert (is there a known process we're missing?)
- Other agent designers (are they seeing similar issues?)
- Literature review (has this data structure been modeled before?)

---

## Summary Table

| Model | Parameters | Complexity | Priority | When to Fit |
|-------|-----------|------------|----------|-------------|
| Model 1: Log-Linear | 3 | Low | **HIGH** | **Always** (baseline) |
| Model 2: Quadratic | 4 | Medium | **MEDIUM** | If Model 1 shows curvature |
| Model 3: Heteroscedastic | 4 | Medium-High | **LOW** | Only if 1-2 fail variance tests |

**Expected outcome**: Model 1 or Model 2 will be adequate. Model 1 is preferred unless evidence strongly favors Model 2 (ΔELPD > 4).

---

## References to EDA Findings

- **Full EDA Report**: `/workspace/eda/eda_report.md`
- **Overdispersion Evidence**: Lines 16-22 (Var/Mean = 70, φ = 1.5)
- **Functional Form Debate**: Lines 33-49 (Linear vs. Quadratic)
- **No Autocorrelation**: Lines 52-56 (r = 0.14, p = 0.37)
- **Prior Recommendations**: Lines 106-120
- **Model Comparison Strategy**: Lines 124-133

---

## Final Recommendation

**Start with Model 1 (Log-Linear Negative Binomial)**. It is:
- Maximally parsimonious (3 parameters)
- Interpretable (constant exponential growth)
- Computationally trivial
- Theoretically justified (R² = 0.92)

**Add Model 2 only if**:
- Posterior predictive checks show systematic curvature
- LOO-CV shows ΔELPD > 2
- β₂ has high probability of being non-zero

**Avoid Model 3 unless**:
- Both Models 1-2 fail variance calibration
- Clear evidence of time-varying dispersion in residuals

**Philosophy**: Occam's Razor with empirical teeth. Simple until proven inadequate.

---

**End of Model Design**
