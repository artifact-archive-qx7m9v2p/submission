# Bayesian Model Proposals: Designer 2
## Focus: Original-Scale Approaches & Variance Modeling

**Author**: Model Designer 2
**Date**: 2025-10-27
**Complementary to**: Designer 1 (log-transformed approaches)
**Philosophy**: Work in original scale when possible, explicitly model heteroscedasticity

---

## Overview

Based on EDA findings, I propose **3 distinct Bayesian model classes** that emphasize original-scale inference and explicit variance modeling:

1. **Quadratic Heteroscedastic Model** - Polynomial with variance decreasing in x
2. **Log-Linear Heteroscedastic Model** - Logarithmic mean with modeled variance structure
3. **Robust Polynomial Model** - Student-t likelihood for influential point resistance

Each model is designed to handle the key empirical challenges:
- Strong nonlinearity (logarithmic pattern, R² = 0.90)
- Heteroscedasticity (7.5× variance decrease from low to high x)
- Small sample size (n=27, requires careful complexity management)
- One influential observation (x=31.5, requires robustness checks)

---

## Critical Modeling Principles

### Falsification Mindset
Each model below includes explicit **abandonment criteria**. I will reject a model if:
- Prior-posterior conflict indicates model fights the data
- Predictive performance degrades (LOO-RMSE > 0.15)
- Pareto k values indicate severe misspecification (k > 0.7)
- Residuals show systematic patterns
- Parameter estimates reach implausible extremes

### Decision Points
I will **stop and pivot** to alternative model classes if:
1. All 3 proposed models fail LOO cross-validation (Pareto k issues)
2. Variance modeling adds no predictive value (ΔLOO < 2)
3. Transformation-based approaches (Designer 1) dominate by ΔLOO > 10
4. Evidence emerges for change points, splines, or mixture models

### Why These Models Might FAIL

**Quadratic model**: May not capture strong diminishing returns pattern adequately. Polynomial extrapolation is dangerous beyond x=32. May overfit with small n=27.

**Log-linear heteroscedastic**: Adds 2 variance parameters - may be too complex for n=27. Heteroscedasticity modeling might not improve predictions enough to justify complexity.

**Robust polynomial**: Student-t adds df parameter. If point 26 is valid data (not error), robustness is unnecessary complexity. Heavy tails may hide real model misspecification.

---

# Model 1: Quadratic Heteroscedastic Model

## Motivation
EDA shows quadratic achieves R² = 0.874 (vs linear R² = 0.677). Variance decreases 7.5× from low to high x. This model captures curvature while explicitly modeling variance structure in original scale.

## Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * x_i + beta_2 * x_i^2
log(sigma_i) = gamma_0 + gamma_1 * x_i
```

**Priors:**
```
beta_0 ~ Normal(1.8, 0.5)         # Intercept near Y(x=1) ≈ 1.77
beta_1 ~ Normal(0.05, 0.05)       # Positive slope (truncated at 0)
beta_2 ~ Normal(-0.001, 0.002)    # Negative curvature (truncated at 0)
gamma_0 ~ Normal(-2.5, 1)         # Log-scale variance intercept
gamma_1 ~ Normal(-0.05, 0.03)     # Variance decreases with x (truncated at 0)
```

## Prior Justification (Evidence-Based)

**beta_0**: EDA shows Y_min = 1.77 at x=1. Prior centered at 1.8, allows [0.8, 2.8] range (covers observed).

**beta_1**: Initial slope must be positive. EDA shows rate of change ≈ 0.113 in low-x region. Prior allows [0, 0.15] with most mass < 0.10.

**beta_2**: Curvature must be negative (diminishing returns). Prior centered at -0.001, allows up to -0.005. This prevents excessive curvature overfitting.

**gamma_0**: Log-variance in low-x region ≈ log(0.062) = -2.78. Prior allows [-4.5, -0.5], corresponding to SD in [0.01, 0.6].

**gamma_1**: Variance ratio 7.5× over range 30 implies γ₁ ≈ -0.067. Prior allows [-0.11, 0.01], enforcing variance decrease.

## Complete Stan Implementation

```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}

transformed data {
  vector[N] x_sq = x .* x;  // Precompute x^2
}

parameters {
  real beta_0;
  real<lower=0> beta_1;           // Constrain positive slope
  real<upper=0> beta_2;           // Constrain negative curvature
  real gamma_0;
  real<upper=0> gamma_1;          // Constrain decreasing variance
}

transformed parameters {
  vector[N] mu;
  vector[N] sigma;

  // Mean function
  mu = beta_0 + beta_1 * x + beta_2 * x_sq;

  // Variance function (ensure sigma > 0)
  sigma = exp(gamma_0 + gamma_1 * x);
}

model {
  // Priors
  beta_0 ~ normal(1.8, 0.5);
  beta_1 ~ normal(0.05, 0.05);
  beta_2 ~ normal(-0.001, 0.002);
  gamma_0 ~ normal(-2.5, 1);
  gamma_1 ~ normal(-0.05, 0.03);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;  // Posterior predictive samples

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma[i]);
    y_rep[i] = normal_rng(mu[i], sigma[i]);
  }
}
```

## Falsification Criteria

**I will ABANDON this model if:**

1. **Pareto k > 0.7** for >3 observations (severe LOO diagnostic failure)
2. **LOO-RMSE > 0.15** (worse than linear model's 0.150)
3. **Posterior mean beta_2 > -0.0001** (curvature vanishes, use simpler model)
4. **gamma_1 posterior includes 0** (variance modeling unnecessary)
5. **Posterior predictive check fails**: <80% coverage of 95% intervals
6. **Residual plot shows U-shape** (quadratic insufficient for diminishing returns)
7. **sigma[high-x] / sigma[low-x] < 2** (heteroscedasticity weaker than expected)

## Expected Performance

**Success indicators:**
- LOO-RMSE: 0.10-0.13 (target < 0.15)
- R²: 0.85-0.90 (EDA baseline R² = 0.874)
- Pareto k: < 0.5 for all observations
- Effective sample size: ESS > 400 for all parameters
- R-hat: < 1.01 (convergence)
- Variance ratio: 3-10× decrease from low to high x

**What would surprise me:**
- If variance modeling provides ΔLOO > 10 improvement (suggests very strong heteroscedasticity)
- If quadratic outperforms logarithmic models (would question EDA findings)
- If beta_2 approaches 0 (would indicate linear sufficient)

## Advantages
1. **Interpretable in original scale** - coefficients have direct Y-unit meaning
2. **Explicit variance modeling** - captures observed 7.5× variance decrease
3. **No transformation bias** - predictions directly on Y scale
4. **Modest complexity** - 5 parameters reasonable for n=27

## Limitations
1. **Polynomial extrapolation dangerous** - predictions unreliable beyond x=32
2. **May underfit diminishing returns** - EDA shows R² = 0.897 for log-linear
3. **Variance model adds 2 parameters** - may be too complex for n=27
4. **No physical interpretation** - unlike power law interpretation of log-log

---

# Model 2: Log-Linear with Heteroscedastic Variance

## Motivation
EDA's primary recommendation uses log(x) transformation (R² = 0.897). This model works in original Y scale for interpretability while modeling heteroscedasticity explicitly. Best of both worlds: captures diminishing returns via log(x), models variance structure.

## Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = gamma_0 + gamma_1 * log(x_i)
```

**Priors:**
```
beta_0 ~ Normal(1.8, 0.5)         # Intercept (Y at x=1, where log(x)=0)
beta_1 ~ Normal(0.3, 0.2)         # Slope per log-unit of x (truncated at 0)
gamma_0 ~ Normal(-2.5, 1)         # Log-variance at x=1
gamma_1 ~ Normal(-0.5, 0.3)       # Variance change per log-unit x (truncated at 0)
```

## Prior Justification (Evidence-Based)

**beta_0**: At x=1 (log(x)=0), Y ≈ 1.77 (observed minimum). Prior centered at 1.8, allows [0.8, 2.8].

**beta_1**: EDA estimates β₁ ≈ 0.3-0.4 for log-linear model. Prior centered at 0.3, allows [0, 0.7]. Must be positive (diminishing returns, not diminishing).

**gamma_0**: At x=1, residual variance ≈ 0.062, so log(SD) ≈ -2.78. Prior allows [-4.5, -0.5].

**gamma_1**: Variance decreases from low to high x. If log(sigma) decreases linearly with log(x), and variance drops 7.5× over log(x) range of [0, 3.45], then γ₁ ≈ -0.58. Prior allows [-1.1, 0.1], weakly enforcing decrease.

## Complete Stan Implementation

```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}

transformed data {
  vector[N] log_x = log(x);  // Precompute log(x)
}

parameters {
  real beta_0;
  real<lower=0> beta_1;           // Positive relationship
  real gamma_0;
  real<upper=0> gamma_1;          // Decreasing variance
}

transformed parameters {
  vector[N] mu;
  vector[N] sigma;

  // Mean function (log-linear)
  mu = beta_0 + beta_1 * log_x;

  // Variance function on log-scale
  sigma = exp(gamma_0 + gamma_1 * log_x);
}

model {
  // Priors
  beta_0 ~ normal(1.8, 0.5);
  beta_1 ~ normal(0.3, 0.2);
  gamma_0 ~ normal(-2.5, 1);
  gamma_1 ~ normal(-0.5, 0.3);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  real<lower=0> R_squared;  // Bayesian R²

  // Log-likelihood for LOO
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma[i]);
    y_rep[i] = normal_rng(mu[i], sigma[i]);
  }

  // Bayesian R² (variance explained)
  {
    real var_y = variance(y);
    real var_res = mean(square(y - mu));
    R_squared = 1 - var_res / var_y;
  }
}
```

## Falsification Criteria

**I will ABANDON this model if:**

1. **ΔLOO < 2 vs simpler constant-variance version** (heteroscedasticity modeling unjustified)
2. **gamma_1 posterior 95% CI includes 0** (no evidence for variance decrease)
3. **LOO-RMSE > 0.12** (should match EDA log-linear performance)
4. **Pareto k > 0.7** for >2 observations
5. **Posterior mean beta_1 < 0.1** (relationship weaker than EDA suggests)
6. **Variance at high-x < 2× variance at low-x** (heteroscedasticity overestimated)
7. **Residuals show systematic pattern** (log(x) insufficient for mean structure)

## Expected Performance

**Success indicators:**
- LOO-RMSE: 0.09-0.11 (target: match EDA R² = 0.897)
- ΔLOO: +3 to +8 vs constant-variance version (justify complexity)
- Pareto k: < 0.5 for most observations
- Variance ratio: 4-10× decrease (match EDA 7.5×)
- R²: 0.88-0.92

**What would surprise me:**
- If ΔLOO > 10 vs constant variance (very strong heteroscedasticity effect)
- If gamma_1 is close to 0 (would question EDA variance analysis)
- If model outperforms log-log from Designer 1 (would question transformation choice)

## Advantages
1. **Captures diminishing returns** - log(x) matches EDA's best functional form
2. **Original scale inference** - predictions and intervals on Y scale directly
3. **Explicit variance structure** - models observed heteroscedasticity
4. **Interpretable parameters** - beta_1 = change in Y per doubling of x
5. **Moderate complexity** - 4 parameters appropriate for n=27

## Limitations
1. **Added complexity** - 2 extra parameters vs constant variance
2. **May not justify complexity** - if ΔLOO small, simpler model preferred
3. **Assumes variance changes with log(x)** - may be wrong functional form
4. **Original scale may inherit heteroscedasticity** - log(Y) transformation might be better

---

# Model 3: Robust Polynomial (Student-t)

## Motivation
EDA identifies point 26 (x=31.5, Y=2.57) as influential with standardized residual = -2.23. If this point is valid but unusual (not measurement error), Student-t likelihood provides robustness. Quadratic form captures curvature, heavy tails handle outliers.

## Mathematical Specification

**Likelihood:**
```
Y_i ~ Student_t(nu, mu_i, sigma)
mu_i = beta_0 + beta_1 * x_i + beta_2 * x_i^2
nu = degrees of freedom (estimated)
```

**Priors:**
```
beta_0 ~ Normal(1.8, 0.5)         # Intercept
beta_1 ~ Normal(0.05, 0.05)       # Linear term (truncated at 0)
beta_2 ~ Normal(-0.001, 0.002)    # Quadratic term (truncated at 0)
sigma ~ Half-Normal(0.3)          # Scale parameter
nu ~ Gamma(2, 0.1)                # Degrees of freedom (mean=20, allows 3-50)
```

## Prior Justification (Evidence-Based)

**Mean parameters (beta_0, beta_1, beta_2)**: Same justification as Model 1 (quadratic).

**sigma**: Half-normal prior with scale 0.3. EDA shows overall SD(Y) = 0.29. Prior allows [0, 0.9] with most mass < 0.5. Note: Student-t scale is not SD, so slightly wider prior.

**nu**: Gamma(2, 0.1) for degrees of freedom. Mean = 20, SD = 14. This allows:
- Low values (nu=3-5): Very heavy tails, strong outlier robustness
- Moderate values (nu=10-30): Mild robustness
- High values (nu>30): Approaches normal distribution

Prior weakly favors moderate robustness. If data are truly normal, posterior will push nu high.

## Complete Stan Implementation

```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}

transformed data {
  vector[N] x_sq = x .* x;
}

parameters {
  real beta_0;
  real<lower=0> beta_1;
  real<upper=0> beta_2;
  real<lower=0> sigma;
  real<lower=1> nu;  // Degrees of freedom > 1 for finite variance
}

transformed parameters {
  vector[N] mu;
  mu = beta_0 + beta_1 * x + beta_2 * x_sq;
}

model {
  // Priors
  beta_0 ~ normal(1.8, 0.5);
  beta_1 ~ normal(0.05, 0.05);
  beta_2 ~ normal(-0.001, 0.002);
  sigma ~ normal(0, 0.3);  // Half-normal via positive constraint
  nu ~ gamma(2, 0.1);

  // Likelihood (robust to outliers)
  y ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  real<lower=0> effective_outliers;  // Proportion downweighted

  for (i in 1:N) {
    log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);
    y_rep[i] = student_t_rng(nu, mu[i], sigma);
  }

  // Diagnostic: count observations with large standardized residuals
  {
    vector[N] std_res = (y - mu) / sigma;
    effective_outliers = sum(fabs(std_res) > 2.5) * 1.0 / N;
  }
}
```

## Falsification Criteria

**I will ABANDON this model if:**

1. **Posterior nu > 50** (heavy tails unnecessary, use normal likelihood)
2. **ΔLOO < 2 vs normal likelihood quadratic** (robustness unjustified)
3. **Pareto k still > 0.7 for point 26** (model still misspecified)
4. **LOO-RMSE > 0.13** (robustness doesn't improve prediction)
5. **Point 26 verified as measurement error** (should exclude, not downweight)
6. **Posterior predictive intervals unreasonably wide** (overestimating uncertainty)
7. **effective_outliers > 0.3** (too many points downweighted, model misspecified)

## Expected Performance

**Success indicators:**
- LOO-RMSE: 0.10-0.13
- Pareto k: < 0.5 for point 26 (robustness helps LOO)
- nu posterior: 5-30 (some robustness, not extreme)
- effective_outliers: 0.04-0.10 (1-3 points slightly downweighted)
- Better LOO than normal for point 26

**What would surprise me:**
- If nu < 5 (extremely heavy tails, suggests many outliers not just point 26)
- If nu > 50 (no robustness needed, why did EDA flag point 26?)
- If ΔLOO > 5 vs normal (very strong outlier effect)

## Advantages
1. **Robust to influential points** - automatically downweights outliers
2. **No manual exclusion** - lets data determine influence
3. **Nested model** - as nu→∞, converges to normal
4. **Better LOO diagnostics** - may resolve Pareto k issues
5. **Honest uncertainty** - heavier tails reflect outlier uncertainty

## Limitations
1. **Additional parameter** - nu adds complexity for n=27
2. **May mask misspecification** - heavy tails hide structural problems
3. **If point 26 is error, should exclude** - robustness wrong solution
4. **Interpretability loss** - Student-t less familiar than normal
5. **May not improve predictions** - if outliers are few and valid

---

# Model Comparison Strategy

## Primary Comparison Metrics

1. **LOO-CV (ELPD)**: Primary model selection criterion
   - Report ΔLOO with standard errors
   - Prefer simpler model if ΔLOO < 2×SE
   - Check Pareto k diagnostics (k < 0.7)

2. **LOO-RMSE**: Out-of-sample prediction accuracy
   - Target: < 0.12 (better than linear's 0.15)
   - Benchmark: log-log EDA performance = 0.093

3. **Posterior Predictive Checks**:
   - 95% intervals should cover ~95% of data
   - Residual plots should show no patterns
   - Generated data should match observed distribution

4. **Computational Diagnostics**:
   - R-hat < 1.01 for all parameters
   - ESS > 400 (effective samples)
   - No divergences (Hamiltonian Monte Carlo)

## Decision Rules

**Choose Model 1 (Quadratic Heteroscedastic) if:**
- ΔLOO > 3 vs constant-variance quadratic
- gamma_1 clearly negative (95% CI excludes 0)
- Interpretability in original scale is priority
- Heteroscedasticity modeling justifies complexity

**Choose Model 2 (Log-Linear Heteroscedastic) if:**
- Best LOO-RMSE among all models
- Captures diminishing returns better than quadratic
- ΔLOO > 2 vs constant-variance version
- Variance modeling provides clear benefit

**Choose Model 3 (Robust Polynomial) if:**
- Point 26 verified as valid but unusual
- Pareto k improved vs normal likelihood
- nu posterior suggests moderate robustness (10-30)
- But ONLY if ΔLOO > 2 vs Model 1

**Reject ALL models and pivot if:**
- All three have LOO-RMSE > 0.15
- Pareto k > 0.7 for multiple observations across all models
- Designer 1's log-log models have ΔLOO > 10 advantage
- Residuals show systematic patterns in all three

## Pairwise Comparisons

**Model 1 vs Model 2:**
- Tests: Original scale polynomial vs log-linear
- Key question: Does log(x) capture diminishing returns better?
- Expect: Model 2 slightly better LOO-RMSE (0.01-0.02)

**Model 1 vs Model 3:**
- Tests: Heteroscedasticity vs robustness
- Key question: Is variance structure or outlier handling more important?
- Expect: Similar performance unless point 26 strongly influential

**Model 2 vs Model 3:**
- Tests: Log-linear heteroscedastic vs robust polynomial
- Key question: Functional form or likelihood matters more?
- Expect: Model 2 better if point 26 not extreme

**All vs Designer 1:**
- Tests: Original scale vs log-transformed Y
- Key question: Is transformation necessary or can we model in original scale?
- Expect: Designer 1's log-log may have small advantage (ΔLOO = 2-5)

---

# Sensitivity Analyses

## Point 26 Sensitivity (Required)

**Procedure:**
1. Fit all three models with full data (N=27)
2. Fit all three models excluding point 26 (N=26)
3. Compare posterior distributions for beta parameters
4. Check if predictions at x=31.5 change substantially

**Decision criteria:**
- If posteriors shift > 20% excluding point 26: High sensitivity, consider exclusion
- If Model 3 (robust) posteriors stable but others shift: Use robust model
- If all models stable: Point 26 not problematic, prefer simplest model

## Prior Sensitivity (Standard)

**Procedure:**
1. Multiply all prior SDs by 2 (weaker priors)
2. Divide all prior SDs by 2 (stronger priors)
3. Compare posteriors to baseline

**Decision criteria:**
- If posteriors shift < 10%: Data-dominated, priors appropriate
- If posteriors shift > 30%: Data-weak, reconsider priors or model complexity

## Heteroscedasticity Test

**Procedure:**
1. Fit Models 1 & 2 with variance modeling (full specification above)
2. Fit simplified versions with constant sigma
3. Compute ΔLOO for each pair

**Decision criteria:**
- If ΔLOO > 3: Heteroscedasticity modeling justified
- If ΔLOO < 2: Use simpler constant-variance model
- Compare to EDA variance ratio (expect 4-8× decrease)

---

# Alternative Models (If Primary Models Fail)

## Backup Plan 1: If LOO diagnostics fail

**B-Spline Model:**
```
mu_i = beta_0 + sum_k beta_k * B_k(x_i)
```
- Use 4-5 basis functions (avoid overfitting n=27)
- Weakly informative priors on spline coefficients
- May capture shape better than polynomial
- Implement if Pareto k > 0.7 persists

## Backup Plan 2: If variance modeling fails

**Simplified Models:**
1. Quadratic with constant variance (3 parameters)
2. Log-linear with constant variance (2 parameters)
3. Compare via LOO to check if variance complexity unjustified

## Backup Plan 3: If polynomial insufficient

**Change Point Model:**
```
mu_i = beta_0 + beta_1 * log(x_i)  if x_i < tau
mu_i = beta_0 + beta_1 * log(tau) + beta_2 * (x_i - tau)  if x_i >= tau
```
- EDA suggests change point at x ≈ 7.4
- Useful if single curve cannot capture both regimes
- Add parameter for change point location (tau)

## Backup Plan 4: If all approaches fail

**Return to EDA:**
- Re-examine data for errors or patterns
- Consider measurement error models
- Check for interaction effects or missing predictors
- May need to collect more data (n=27 too small for complex patterns)

---

# Expected Challenges and Mitigation

## Challenge 1: Small Sample Size (n=27)

**Risk**: Overfitting, wide posteriors, unstable variance estimates

**Mitigation**:
- Use informative priors based on EDA
- Prefer simpler models (2-4 parameters)
- Mandatory LOO cross-validation
- Report posterior predictive intervals
- If ESS < 400, run longer chains

## Challenge 2: Heteroscedasticity Modeling

**Risk**: Variance parameters unstable, 2 extra parameters may overfit

**Mitigation**:
- Use ΔLOO to test if complexity justified
- Check gamma_1 posterior doesn't include 0
- Compare to constant-variance as baseline
- If ΔLOO < 2, abandon variance modeling

## Challenge 3: Influential Point 26

**Risk**: Single point dominates inference, unstable posteriors

**Mitigation**:
- Mandatory sensitivity analysis (with/without point 26)
- Try robust likelihood (Model 3)
- Check Pareto k for this observation
- Verify data if possible
- Report sensitivity in conclusions

## Challenge 4: Polynomial Extrapolation

**Risk**: Unreliable predictions beyond x=32

**Mitigation**:
- Do NOT extrapolate beyond observed range
- Report prediction intervals (will be wide at boundaries)
- Use log-linear (Model 2) if extrapolation needed
- Document extrapolation risks in model limitations

## Challenge 5: Computational Issues

**Risk**: Divergences, low ESS, non-convergence

**Mitigation**:
- Use non-centered parameterizations if needed
- Increase adapt_delta to 0.95
- Run longer chains (4000 iterations if needed)
- Check for parameter correlations
- May need to reparameterize variance model

---

# Implementation Checklist

## Before Fitting

- [ ] Verify data loaded correctly (N=27, no missing)
- [ ] Center/scale x if computational issues arise
- [ ] Prepare train/test split for out-of-sample validation
- [ ] Set up parallel MCMC chains (4 chains minimum)

## During Fitting

- [ ] Monitor convergence (R-hat, ESS)
- [ ] Check for divergences (flag if >10)
- [ ] Watch for extreme parameter values
- [ ] Verify log-likelihood computed correctly

## After Fitting

- [ ] Compute LOO-CV with Pareto k diagnostics
- [ ] Generate posterior predictive samples
- [ ] Plot residuals vs fitted, vs x
- [ ] Check 95% interval coverage
- [ ] Compare to EDA benchmarks (R² = 0.897, RMSE = 0.093)
- [ ] Sensitivity analysis for point 26
- [ ] Prior sensitivity check

## Reporting

- [ ] Report all convergence diagnostics
- [ ] Document any computational issues
- [ ] Compare models via ΔLOO table
- [ ] Show posterior distributions for key parameters
- [ ] Visualize posterior predictive fit
- [ ] State model recommendation with caveats
- [ ] Document what would change recommendation

---

# Success Criteria Summary

## Minimal Acceptable Performance

- LOO-RMSE < 0.15 (better than linear)
- Pareto k < 0.7 for all observations
- R-hat < 1.01, ESS > 400
- No systematic residual patterns
- 95% intervals cover 90-95% of data

## Target Performance

- LOO-RMSE < 0.12
- R² > 0.88
- Pareto k < 0.5 for most observations
- Variance ratio matches EDA (4-10×)
- Posteriors consistent with EDA estimates

## Excellent Performance

- LOO-RMSE < 0.10 (match log-log benchmark)
- R² > 0.90
- All Pareto k < 0.5
- Clear evidence for heteroscedasticity (ΔLOO > 3)
- Robust to point 26 exclusion

---

# Red Flags - When to Abandon and Pivot

## Abandon Individual Models If:

**Model 1 (Quadratic Heteroscedastic):**
- gamma_1 posterior includes 0 → Drop variance modeling
- beta_2 near 0 → Use linear instead
- LOO-RMSE > 0.13 → Try Model 2

**Model 2 (Log-Linear Heteroscedastic):**
- ΔLOO < 2 vs constant variance → Use simpler version
- LOO-RMSE > 0.12 → Question log transformation benefit
- Worse than Designer 1's log-log by ΔLOO > 8 → Prefer transformation

**Model 3 (Robust Polynomial):**
- nu > 50 → No robustness needed, use Model 1
- ΔLOO < 2 vs Model 1 → Robustness unjustified
- effective_outliers > 0.3 → Model misspecified, not outlier issue

## Abandon ALL Original-Scale Models If:

1. **All three have Pareto k > 0.7** for multiple points
   - Suggests original scale fundamentally wrong
   - Pivot to transformation-based approaches (Designer 1)

2. **Designer 1's models have ΔLOO > 10** advantage
   - Log-transformation essential, not optional
   - Original scale modeling adds complexity without benefit

3. **Residuals show strong systematic patterns** in all three
   - Polynomial and log-linear insufficient
   - Consider splines, change points, or mixtures

4. **All models fail posterior predictive checks**
   - Coverage < 80% or > 98%
   - Generated data looks nothing like observed
   - Return to EDA for missed patterns

## Pivot Strategies

**If original scale fails:**
- Adopt Designer 1's log-log approach
- Acknowledge transformation necessary
- Focus on robustness and diagnostics instead

**If polynomial forms fail:**
- Try B-splines with 4-5 knots
- Consider change point models
- Check for interaction effects

**If variance modeling fails:**
- Use constant variance (simpler)
- Check if log(Y) transformation stabilizes variance
- May indicate n=27 too small for heteroscedasticity modeling

---

# Why This Might All Be Wrong

## Fundamental Assumptions to Question

1. **Homogeneity assumption**: What if there are subgroups in the data?
   - Check if observations cluster by some unmeasured variable
   - Mixture models might be appropriate

2. **Independence assumption**: What if observations are sequential or clustered?
   - EDA shows Durbin-Watson = 0.775 (concerning)
   - May need autocorrelation or hierarchical structure

3. **Single predictor assumption**: What if x alone is insufficient?
   - May need interaction terms
   - Missing covariates could explain heteroscedasticity

4. **Parametric form assumption**: What if relationship is more complex?
   - Non-parametric approaches (GPs) might reveal structure
   - EDA might oversimplify by forcing polynomial/log fits

## Evidence That Would Make Me Reject Everything

1. **Consistent Pareto k > 0.7** across all models
   - Indicates fundamental misspecification
   - No parametric form captures structure

2. **Posterior predictive checks consistently fail**
   - Generated data doesn't match observed
   - Missing crucial structural element

3. **Sensitivity analysis shows extreme instability**
   - Single point changes conclusions dramatically
   - Data too sparse for reliable inference

4. **Designer 1's models dominate by ΔLOO > 15**
   - Original scale approach fundamentally wrong
   - Transformation essential, not a choice

## Alternative Hypotheses

If my models fail, consider:

1. **Gaussian Process**: Non-parametric, may capture unknown structure
2. **Mixture Model**: Multiple subpopulations with different relationships
3. **Hierarchical Model**: Observations grouped by unmeasured factor
4. **Measurement Error Model**: Uncertainty in x or Y
5. **Collect More Data**: n=27 may be insufficient for complexity observed

---

# Final Notes

## Philosophy

I've designed these models to FAIL INFORMATIVELY. Each has clear abandonment criteria. Success is not fitting all three models - success is discovering which aspects of the data are real vs artifacts.

## Expected Outcome

I expect Model 2 (Log-Linear Heteroscedastic) to perform best among my proposals:
- Captures diminishing returns via log(x)
- Models variance structure
- Works in interpretable original scale
- Moderate complexity (4 parameters)

BUT I expect Designer 1's log-log model may outperform all of mine by ΔLOO = 3-8, because:
- Transformation may stabilize variance better than modeling it
- Log(Y) scale may be more natural for this data
- Simpler (2-3 parameters) is better with n=27

If so, that's a SUCCESS - we learned original scale requires added complexity that transformation avoids.

## Commitment to Truth

I will report all results honestly:
- If all models fail diagnostics, I'll say so
- If Designer 1's models dominate, I'll recommend them
- If evidence contradicts EDA, I'll question EDA
- If n=27 is too small, I'll recommend more data

The goal is finding the best model for this data, not defending my proposals.

---

**Ready for Implementation**: All three Stan models are complete and ready to run with CmdStanPy.

**Next Steps**:
1. Load data and prepare for Stan
2. Fit all three models (4 chains, 2000 iterations each)
3. Compute LOO diagnostics
4. Run sensitivity analyses
5. Compare to Designer 1's models
6. Recommend best model with caveats

**Output Location**: `/workspace/experiments/designer_2/proposed_models.md`
