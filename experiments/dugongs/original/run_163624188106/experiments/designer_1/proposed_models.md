# Bayesian Model Proposals: Transformation-Based and Robust Approaches

**Designer**: Model Designer 1
**Focus**: Transformation-based approaches with robustness considerations
**Date**: 2025-10-27
**Sample Size**: N = 27 observations

---

## Executive Summary

Based on EDA findings showing strong logarithmic relationship (R² = 0.90) with diminishing returns pattern, I propose **3 competing Bayesian model classes** that make fundamentally different assumptions:

1. **Log-Log Linear Model** (PRIMARY) - Assumes power law relationship with homoscedastic log-scale variance
2. **Robust Log-Log Model** - Same structure but Student-t likelihood for outlier resistance
3. **Heteroscedastic Log-Linear Model** - Original scale with variance decreasing in x

**Critical Insight**: These models will FAIL if:
- The log transformation creates spurious patterns (check via posterior predictive)
- The influential point (x=31.5) dominates inference (check via LOO Pareto-k)
- Residual patterns emerge in log-scale (indicating more complex structure)

---

## Model 1: Log-Log Linear Model (PRIMARY)

### Mathematical Specification

**Likelihood:**
```
log(Y_i) ~ Normal(mu_i, sigma)
mu_i = alpha + beta * log(x_i)
```

**Priors:**
```
alpha ~ Normal(0.6, 0.3)      # Log-scale intercept
beta ~ Normal(0.13, 0.1)       # Power law exponent (weakly informative)
sigma ~ Half-Normal(0.1)       # Log-scale residual SD
```

**Derived quantities:**
```
Power law form: Y = exp(alpha) * x^beta
Expected intercept: exp(0.6) ≈ 1.82
Expected exponent: 0.13 (strong diminishing returns)
```

### Prior Justification

**Alpha (log-scale intercept):**
- EDA shows log(Y) mean ≈ 0.6 (since Y mean ≈ 2.3)
- SD = 0.3 allows range [0, 1.2] → Y-intercept range [1.0, 3.3]
- Covers observed Y range [1.77, 2.72] with room for uncertainty

**Beta (power law exponent):**
- EDA power law analysis: exponent ≈ 0.126
- Mean = 0.13 centers on empirical estimate
- SD = 0.1 allows range [0, 0.33] with 95% probability
- CRITICAL: This weakly constrains beta > 0 (positive relationship)
- If data strongly contradicts this, posterior will shift (but flag for investigation)

**Sigma (residual SD):**
- EDA log-log model shows residual SD ≈ 0.05
- Half-Normal(0.1) is weakly informative: allows 0-0.2 range
- Heavy right tail allows model to learn if variance is higher

### Complete Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] x;
  vector<lower=0>[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
  vector[N] log_Y = log(Y);
}

parameters {
  real alpha;                    // Log-scale intercept
  real beta;                     // Power law exponent
  real<lower=0> sigma;           // Log-scale residual SD
}

model {
  // Priors
  alpha ~ normal(0.6, 0.3);
  beta ~ normal(0.13, 0.1);
  sigma ~ normal(0, 0.1);        // Half-normal via constraint

  // Likelihood
  log_Y ~ normal(alpha + beta * log_x, sigma);
}

generated quantities {
  vector[N] log_lik;              // For LOO-CV
  vector[N] y_pred;               // Posterior predictive (original scale)
  vector[N] log_y_pred;           // Posterior predictive (log scale)
  real R_squared;                 // Bayesian R²

  // Log-likelihood for LOO
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(log_Y[i] | alpha + beta * log_x[i], sigma);
  }

  // Posterior predictions (log scale)
  for (i in 1:N) {
    log_y_pred[i] = normal_rng(alpha + beta * log_x[i], sigma);
  }

  // Posterior predictions (original scale via exponential)
  y_pred = exp(log_y_pred);

  // Bayesian R² (variance explained in log scale)
  {
    real var_log_y = variance(log_Y);
    vector[N] mu = alpha + beta * log_x;
    real var_residuals = variance(log_Y - mu);
    R_squared = 1 - var_residuals / var_log_y;
  }
}
```

### Falsification Criteria (I WILL ABANDON THIS MODEL IF...)

1. **Prior-posterior conflict**: Posterior mean for beta < 0 or > 0.5
   - **Why**: Negative beta contradicts diminishing returns pattern
   - **Why**: beta > 0.5 would indicate stronger relationship than EDA suggests
   - **Action**: Reconsider functional form entirely

2. **Residual patterns in log scale**: Posterior predictive check shows systematic patterns
   - **Why**: Log transformation should linearize relationship
   - **Action**: Switch to more complex model (spline, GP)

3. **Poor LOO performance**: Pareto-k > 0.7 for multiple observations
   - **Why**: Indicates model sensitivity to individual points
   - **Action**: Switch to robust Student-t model

4. **Heteroscedasticity persists**: Residual variance changes systematically with x
   - **Why**: Log transformation should stabilize variance
   - **Action**: Switch to heteroscedastic model (Model 3)

5. **Back-transformation bias**: Mean of exp(log_y_pred) >> observed Y values
   - **Why**: Indicates lognormal assumption fails
   - **Action**: Model Y directly with different likelihood

### Expected Performance Metrics

**Success indicators:**
- R² > 0.88 (matching or exceeding EDA log-log R² = 0.903)
- LOO-ELPD within 2 SE of best model
- All Pareto-k < 0.7 (preferably < 0.5)
- Rhat < 1.01 for all parameters
- ESS > 400 for all parameters
- 95% posterior predictive intervals cover ~95% of data
- Residuals in log scale: Shapiro p > 0.05

**Known limitations:**
- Small sample (n=27): Wide posterior intervals expected
- Assumes homoscedastic variance in log scale
- Sensitive to transformation choice (log vs other)
- Assumes no change points or regime shifts

### Advantages and Limitations

**Advantages:**
- Parsimony: Only 3 parameters (appropriate for n=27)
- Interpretability: Direct power law interpretation Y = exp(alpha) * x^beta
- Best EDA fit: R² = 0.903 in log-log scale
- Strong theoretical foundation: Log transformation often stabilizes variance
- Excellent residual diagnostics in EDA (Shapiro p = 0.836)
- Computational efficiency: Simple linear model in log scale

**Limitations:**
- Transformation assumptions may not hold
- Back-transformation introduces bias (Jensen's inequality)
- Cannot capture change points or regime shifts
- Assumes homoscedastic variance in log scale (may not be true)
- Influential points still affect inference
- Extrapolation beyond x=32 highly uncertain

---

## Model 2: Robust Log-Log Model with Student-t Likelihood

### Mathematical Specification

**Likelihood:**
```
log(Y_i) ~ Student_t(nu, mu_i, sigma)
mu_i = alpha + beta * log(x_i)
```

**Priors:**
```
alpha ~ Normal(0.6, 0.3)      # Log-scale intercept (same as Model 1)
beta ~ Normal(0.13, 0.1)       # Power law exponent (same as Model 1)
sigma ~ Half-Normal(0.1)       # Log-scale residual SD (same as Model 1)
nu ~ Gamma(2, 0.1)             # Degrees of freedom (mean=20, allows 4-60 range)
```

**Rationale for Student-t:**
- EDA identified influential point at x=31.5 (Cook's D high, leverage=0.30)
- Student-t has heavier tails than Normal → downweights outliers
- If nu is large (>30), converges to Normal (data decides robustness need)
- If nu is small (4-10), indicates outliers present

### Prior Justification

**Alpha, beta, sigma:** Same as Model 1 (for direct comparison)

**Nu (degrees of freedom):**
- Gamma(2, 0.1) has mean = 2/0.1 = 20, mode ≈ 10
- Allows range approximately [4, 60] with 95% probability
- nu < 4: Too heavy-tailed, would downweight too much
- nu > 60: Essentially Normal, no robustness benefit
- Prior is weakly informative: lets data determine tail behavior

### Complete Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] x;
  vector<lower=0>[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
  vector[N] log_Y = log(Y);
}

parameters {
  real alpha;                    // Log-scale intercept
  real beta;                     // Power law exponent
  real<lower=0> sigma;           // Log-scale residual SD
  real<lower=2> nu;              // Degrees of freedom (>2 ensures finite variance)
}

model {
  // Priors
  alpha ~ normal(0.6, 0.3);
  beta ~ normal(0.13, 0.1);
  sigma ~ normal(0, 0.1);        // Half-normal via constraint
  nu ~ gamma(2, 0.1);

  // Likelihood (Student-t for robustness)
  log_Y ~ student_t(nu, alpha + beta * log_x, sigma);
}

generated quantities {
  vector[N] log_lik;              // For LOO-CV
  vector[N] y_pred;               // Posterior predictive (original scale)
  vector[N] log_y_pred;           // Posterior predictive (log scale)
  real R_squared;                 // Bayesian R²
  int<lower=0, upper=1> is_robust; // Flag: nu < 10 indicates robustness needed

  // Log-likelihood for LOO
  for (i in 1:N) {
    log_lik[i] = student_t_lpdf(log_Y[i] | nu, alpha + beta * log_x[i], sigma);
  }

  // Posterior predictions (log scale)
  for (i in 1:N) {
    log_y_pred[i] = student_t_rng(nu, alpha + beta * log_x[i], sigma);
  }

  // Posterior predictions (original scale)
  y_pred = exp(log_y_pred);

  // Bayesian R²
  {
    real var_log_y = variance(log_Y);
    vector[N] mu = alpha + beta * log_x;
    real var_residuals = variance(log_Y - mu);
    R_squared = 1 - var_residuals / var_log_y;
  }

  // Robustness indicator
  is_robust = nu < 10 ? 1 : 0;
}
```

### Falsification Criteria (I WILL ABANDON THIS MODEL IF...)

1. **Nu posterior concentrated at prior upper bound (>50)**
   - **Why**: Data prefers Normal likelihood → Model 1 is simpler
   - **Action**: Use Model 1 instead (parsimony)

2. **LOO worse than Model 1 by > 2 SE**
   - **Why**: Extra parameter doesn't improve prediction
   - **Action**: Reject in favor of simpler Model 1

3. **Extreme nu uncertainty**: 95% credible interval spans [2, 60]
   - **Why**: Model cannot learn tail behavior from n=27
   - **Action**: Use Model 1 (data insufficient for robustness estimation)

4. **Same falsification criteria as Model 1** (beta, residuals, heteroscedasticity)

### Expected Performance Metrics

**Success indicators:**
- R² similar to Model 1 (within 0.02)
- LOO-ELPD improves over Model 1 (if outliers present)
- Pareto-k < 0.7 for ALL observations (especially x=31.5)
- Nu posterior: If mean < 15, robustness needed; if > 30, Normal adequate
- Rhat < 1.01, ESS > 400 for all parameters
- Point 26 (x=31.5) should have lower influence than in Model 1

**Comparison with Model 1:**
- If nu >> 30: Models essentially identical, prefer simpler Model 1
- If nu ≈ 5-15: Robustness beneficial, prefer Model 2
- If LOO Pareto-k improves: Clear evidence for robustness

### Advantages and Limitations

**Advantages:**
- Robust to influential point at x=31.5
- Data decides robustness need (via nu posterior)
- If Normal adequate, nu→∞ and matches Model 1
- Better LOO diagnostics expected (lower Pareto-k)
- Same interpretability as Model 1 (power law)
- Automatically downweights outliers without removal

**Limitations:**
- One extra parameter (nu) increases uncertainty with n=27
- Harder to estimate with small samples
- If nu poorly identified, may not converge well
- Slightly more complex computation
- Still assumes homoscedastic variance in log scale
- Heavier tails may inflate prediction intervals unnecessarily

**When to prefer over Model 1:**
- LOO Pareto-k > 0.7 for point 26 in Model 1
- LOO-ELPD improves by > 2 SE
- Point 26 verification shows it's genuine (not measurement error)
- Sensitivity analysis shows Model 1 unstable to point 26

---

## Model 3: Heteroscedastic Log-Linear Model (Original Scale)

### Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = gamma_0 + gamma_1 * x_i
```

**Priors:**
```
beta_0 ~ Normal(1.8, 0.5)      # Intercept (original scale)
beta_1 ~ Normal(0.3, 0.2)      # Slope (change per log-unit of x)
gamma_0 ~ Normal(-2, 1)        # Log-variance intercept
gamma_1 ~ Normal(-0.02, 0.02)  # Variance decreases with x (based on EDA)
```

**Rationale for heteroscedastic variance:**
- EDA shows variance decreases 7.5× from low to high x
- Levene's test: p = 0.003 (significant heteroscedasticity)
- Log(sigma_i) ensures sigma_i > 0
- Linear model for log-variance is flexible and identifiable

### Prior Justification

**Beta_0 (intercept):**
- At x=1, log(x)=0, so mu = beta_0
- EDA shows Y at low x ≈ 1.8
- SD = 0.5 allows range [0.8, 2.8] → covers data range

**Beta_1 (slope):**
- EDA logarithmic model: Y = 1.79 + 0.30 * log(x), R² = 0.897
- Mean = 0.3 matches empirical estimate
- SD = 0.2 allows range [0, 0.7] with flexibility

**Gamma_0 (log-variance intercept):**
- Low-x variance ≈ 0.062 → log(0.062) ≈ -2.78
- Mean = -2 is slightly higher (more conservative)
- SD = 1 allows wide range [-4, 0] → variance [0.02, 1.0]

**Gamma_1 (variance slope):**
- EDA: variance drops 7.5× over x range [1, 31.5]
- Log-linear: need gamma_1 ≈ -log(7.5)/30 ≈ -0.067
- Mean = -0.02 is more conservative (less dramatic decrease)
- SD = 0.02 allows range [-0.06, 0.02]
- **CRITICAL**: Allows gamma_1 > 0 if data shows variance increasing

### Complete Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] x;
  vector<lower=0>[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
}

parameters {
  real beta_0;                   // Intercept (original scale)
  real beta_1;                   // Slope for log(x)
  real gamma_0;                  // Log-variance intercept
  real gamma_1;                  // Log-variance slope
}

transformed parameters {
  vector[N] mu;
  vector<lower=0>[N] sigma;

  // Mean structure
  mu = beta_0 + beta_1 * log_x;

  // Variance structure (heteroscedastic)
  sigma = exp(gamma_0 + gamma_1 * x);
}

model {
  // Priors
  beta_0 ~ normal(1.8, 0.5);
  beta_1 ~ normal(0.3, 0.2);
  gamma_0 ~ normal(-2, 1);
  gamma_1 ~ normal(-0.02, 0.02);

  // Likelihood (heteroscedastic)
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;              // For LOO-CV
  vector[N] y_pred;               // Posterior predictive
  real R_squared;                 // Bayesian R²
  real var_ratio;                 // Variance ratio (max/min)

  // Log-likelihood for LOO
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma[i]);
  }

  // Posterior predictions
  for (i in 1:N) {
    y_pred[i] = normal_rng(mu[i], sigma[i]);
  }

  // Bayesian R²
  {
    real var_y = variance(Y);
    real var_residuals = variance(Y - mu);
    R_squared = 1 - var_residuals / var_y;
  }

  // Variance ratio (diagnostic for heteroscedasticity)
  var_ratio = max(sigma) / min(sigma);
}
```

### Falsification Criteria (I WILL ABANDON THIS MODEL IF...)

1. **Gamma_1 posterior includes zero (95% CI overlaps zero)**
   - **Why**: No evidence for heteroscedasticity
   - **Action**: Use simpler homoscedastic model

2. **LOO worse than Model 1 by > 2 SE**
   - **Why**: Extra complexity not justified
   - **Action**: Prefer simpler Model 1

3. **Residuals still show heteroscedasticity**
   - **Why**: Linear variance model insufficient
   - **Action**: Try more flexible variance function (e.g., log(sigma) ~ gamma_0 + gamma_1 * log(x))

4. **Poor convergence for gamma parameters (Rhat > 1.05)**
   - **Why**: Variance parameters poorly identified with n=27
   - **Action**: Simplify to homoscedastic or use stronger priors

5. **Extreme variance predictions**: sigma_i > 1.0 or < 0.01
   - **Why**: Model extrapolating beyond reasonable range
   - **Action**: Revise priors or variance function

### Expected Performance Metrics

**Success indicators:**
- R² > 0.85 (close to EDA log-linear R² = 0.897)
- Gamma_1 posterior: mean < -0.01 (confirms variance decreases)
- Variance ratio (max/min): 3-10× (matching EDA 7.5×)
- LOO-ELPD competitive with Model 1
- Residuals: No heteroscedasticity in standardized residuals
- Rhat < 1.01, ESS > 400 (harder with 4 parameters and n=27)

**Diagnostic checks:**
- Plot sigma_i vs x: Should show clear decreasing trend
- Standardized residuals: Should be homoscedastic
- Variance ratio posterior: Should be > 1 with high probability

### Advantages and Limitations

**Advantages:**
- Works in original scale (no back-transformation needed)
- Explicitly models heteroscedasticity observed in EDA
- Predictions have correct uncertainty structure
- Interpretable: Y increases by beta_1 per log-unit of x
- Variance decreases with x (matches EDA pattern)

**Limitations:**
- 4 parameters for n=27 (pushing complexity limit)
- Variance parameters may be poorly identified
- Requires stronger modeling assumptions (variance function form)
- More complex than Models 1 and 2
- May have convergence issues with small sample
- Less interpretable than power law form

**When to prefer over Model 1:**
- Model 1 residuals show persistent heteroscedasticity
- Scientific interest in uncertainty quantification at different x values
- Predictions needed in original scale (not log-transformed)
- LOO diagnostics similar or better than Model 1

---

## Model Comparison and Selection Strategy

### Prioritization

**Must fit first:**
1. **Model 1** (Log-Log Linear) - PRIMARY candidate, simplest, best EDA fit
2. **Model 2** (Robust Log-Log) - If Model 1 has high Pareto-k at x=31.5

**Fit conditionally:**
3. **Model 3** (Heteroscedastic) - If Model 1 residuals show heteroscedasticity

### Decision Tree

```
START → Fit Model 1
  ├─ LOO Pareto-k > 0.7?
  │   YES → Fit Model 2, compare LOO-ELPD
  │   NO → Continue with Model 1
  │
  ├─ Residuals heteroscedastic?
  │   YES → Fit Model 3, compare LOO-ELPD
  │   NO → Continue with Model 1
  │
  └─ Select model with best LOO-ELPD (if difference > 2 SE)
      Otherwise prefer simplest (Model 1)
```

### Comparison Metrics

**Primary criterion: LOO-CV**
- Compare LOO-ELPD (expected log predictive density)
- Use SE of difference for statistical significance
- Prefer simpler model unless ΔELPD > 2×SE

**Secondary criteria:**
- Pareto-k diagnostics (all < 0.7)
- Posterior predictive checks (coverage, residual patterns)
- Parameter interpretability and uncertainty
- Computational efficiency (convergence, ESS)

**Tertiary criteria:**
- R² (in-sample fit)
- Prior-posterior consistency
- Scientific plausibility

### Stress Tests (Designed to Break the Models)

1. **Leave-one-out for point 26 (x=31.5)**
   - Refit each model excluding this point
   - Compare posteriors: If beta changes > 0.05, model unstable
   - Expected: Model 2 most stable, Model 1 most sensitive

2. **Prior sensitivity analysis**
   - Double and halve prior SDs
   - Posteriors should be similar (data dominates with n=27)
   - If posteriors shift dramatically, data insufficient

3. **Posterior predictive check: Extrapolation**
   - Generate predictions for x = 40, 50, 100
   - Check if credible intervals explode or predictions unreasonable
   - Expected: All models should show wide intervals (honest uncertainty)

4. **Residual autocorrelation check**
   - Sort residuals by x, compute autocorrelation
   - If ACF(1) > 0.3, model misses systematic pattern
   - Expected: All models should have ACF ≈ 0

5. **Variance homogeneity test (for Models 1 and 2)**
   - Split residuals into low-x and high-x groups
   - Compute variance ratio from posterior predictive
   - If ratio > 3, heteroscedasticity present → switch to Model 3

---

## Red Flags and Pivot Points

### Abandon ALL models and reconsider if:

1. **All models show Pareto-k > 0.7 for multiple points**
   - Indicates fundamental model misspecification
   - Pivot to: Gaussian process or nonparametric model

2. **Posterior predictive checks fail for all models**
   - Simulated data doesn't resemble observed data
   - Pivot to: More flexible likelihood (e.g., mixture model)

3. **Beta posterior contradicts EDA (e.g., negative or > 0.5)**
   - Data generation process different than EDA suggested
   - Pivot to: Reconsider functional form (polynomial, spline, change point)

4. **R² < 0.80 for all models**
   - Log transformation not capturing relationship
   - Pivot to: Alternative transformations (Box-Cox, square root)

5. **Extreme parameter uncertainty (95% CI spans implausible values)**
   - Sample size insufficient for proposed models
   - Pivot to: Stronger informative priors or simpler model

### Model-specific pivot points:

**Model 1:**
- Heteroscedasticity persists → Switch to Model 3
- High Pareto-k at x=31.5 → Switch to Model 2
- Residual patterns → Consider change point or spline

**Model 2:**
- Nu posterior > 40 → Revert to Model 1 (simpler)
- Poor convergence (Rhat > 1.05) → Revert to Model 1
- LOO worse than Model 1 → Reject Model 2

**Model 3:**
- Gamma_1 not significant → Revert to Model 1
- Poor convergence → Simplify variance function
- LOO worse than Model 1 → Reject Model 3

---

## Implementation Plan

### Phase 1: Fit Model 1 (Primary)
1. Implement Stan code for Model 1
2. Run prior predictive checks
3. Fit with 4 chains, 2000 iterations (1000 warmup)
4. Check convergence (Rhat, ESS)
5. Compute LOO-CV and check Pareto-k
6. Posterior predictive checks
7. Residual diagnostics

### Phase 2: Conditional Model 2 (Robustness)
**Trigger**: Model 1 Pareto-k > 0.7 for point 26
1. Implement Stan code for Model 2
2. Fit with same MCMC settings
3. Compare nu posterior to prior
4. Compare LOO-ELPD with Model 1
5. Check if Pareto-k improves

### Phase 3: Conditional Model 3 (Heteroscedasticity)
**Trigger**: Model 1 residuals show heteroscedasticity
1. Implement Stan code for Model 3
2. Fit with same MCMC settings
3. Check gamma_1 posterior vs zero
4. Compare variance predictions to EDA findings
5. Compare LOO-ELPD with Model 1

### Phase 4: Model Selection
1. Compare all fitted models via LOO-ELPD
2. Apply parsimony rule: prefer simpler if ΔELPD < 2×SE
3. Validate selected model with held-out predictions (if possible)
4. Conduct sensitivity analyses (leave-one-out, prior sensitivity)
5. Document decision and limitations

---

## Expected Outcomes

### Most Likely Scenario
- **Model 1 wins**: Simplest, best EDA fit, adequate for n=27
- **Model 2 competitive**: If point 26 influential, nu ≈ 10-15
- **Model 3 loses**: Too complex for n=27, variance parameters uncertain

### Alternative Scenarios

**Scenario A: Robustness needed**
- Model 1 Pareto-k > 0.7 at x=31.5
- Model 2 nu posterior: mean ≈ 8-12
- Model 2 LOO-ELPD improves by > 2 SE
- **Action**: Use Model 2, document outlier sensitivity

**Scenario B: Heteroscedasticity critical**
- Model 1 residuals: variance ratio > 5
- Model 3 gamma_1 < 0 with high probability
- Model 3 LOO-ELPD improves by > 2 SE
- **Action**: Use Model 3, acknowledge complexity cost

**Scenario C: All models fail**
- R² < 0.80 for all models
- Pareto-k > 0.7 for multiple points
- Posterior predictive checks fail
- **Action**: Pivot to Gaussian process or reconsider data quality

---

## Limitations and Honest Uncertainty

### What we DON'T know:
1. **True data generation process**: Models are approximations
2. **Generalization beyond x=32**: No data, pure extrapolation
3. **Whether point 26 is genuine**: Needs verification
4. **Optimal variance structure**: Limited by n=27
5. **Change point location**: If diminishing returns shift regime

### What would make me reconsider EVERYTHING:
1. Additional data shows different pattern
2. Domain expert says power law implausible
3. Point 26 confirmed as measurement error
4. Residuals show clear nonlinearity
5. External validation dataset shows poor prediction

### Stopping rules:
- If 3+ models fail falsification criteria → Consult domain expert
- If LOO Pareto-k > 0.7 for >5 points → Data quality issue
- If convergence fails after prior tuning → Model too complex
- If prediction intervals don't cover data → Likelihood misspecified

---

## Summary Table

| Model | Parameters | Complexity | Primary Advantage | Primary Risk | Expected LOO |
|-------|-----------|-----------|------------------|--------------|--------------|
| **Model 1** | 3 (alpha, beta, sigma) | Low | Simplicity, interpretability | Sensitive to outliers | Best |
| **Model 2** | 4 (+nu) | Medium | Robustness to point 26 | Extra parameter uncertainty | Similar to M1 |
| **Model 3** | 4 (beta_0, beta_1, gamma_0, gamma_1) | High | Explicit heteroscedasticity | Poor identification with n=27 | Worse than M1 |

**Recommendation**: Start with Model 1. Fit Model 2 if diagnostics require. Only fit Model 3 if heteroscedasticity severe.

---

## Stan Model Files

All three models are fully specified above with complete Stan code. For implementation:

1. **Model 1**: `log_log_linear.stan` (READY TO USE)
2. **Model 2**: `log_log_robust.stan` (READY TO USE)
3. **Model 3**: `heteroscedastic_log_linear.stan` (READY TO USE)

Each includes:
- Data block with input validation
- Transformed data block for log transformations
- Parameters block with constraints
- Model block with priors and likelihood
- Generated quantities block with log_lik for LOO-CV

**CmdStanPy usage example:**
```python
import cmdstanpy
import numpy as np

# Prepare data
data = {
    'N': len(x),
    'x': x.tolist(),
    'Y': Y.tolist()
}

# Compile and fit
model = cmdstanpy.CmdStanModel(stan_file='log_log_linear.stan')
fit = model.sample(data=data, chains=4, iter_sampling=1000, iter_warmup=1000)

# Extract LOO
log_lik = fit.stan_variable('log_lik')
```

---

**End of Model Proposals**

**Designer 1 Sign-off**: These models represent my best attempt to honor the EDA findings while maintaining scientific skepticism. I expect Model 1 to succeed, but I've designed Models 2 and 3 as principled alternatives if it fails. I'm ready to abandon any or all of these if evidence demands it.
