# Smooth Nonlinear Bayesian Models for Time Series Count Data
## Model Designer 2: Flexible Trend Models

**Author**: Model Designer 2 (Smooth Nonlinear Specialist)
**Date**: 2025-10-29
**Focus**: Continuous smooth functions without discrete changepoints

---

## Executive Summary

**Context**: 40 observations of count data with strong nonlinear growth (745% total), severe overdispersion (var/mean = 67.99), and high autocorrelation (ACF(1) = 0.944). EDA suggests structural break, but we explore whether smooth functions can capture this pattern without discrete regime changes.

**Philosophy**: The apparent "structural break" at observation 17 might be an artifact of forcing smooth exponential growth into discrete regimes. True underlying process may be continuously accelerating.

**Critical Question**: Can smooth nonlinear models match changepoint models? If not, why?

---

## Competing Hypotheses

### H1: Gaussian Process (Smooth + Flexible)
**Claim**: Growth is smoothly varying with strong temporal correlation. No discrete break needed.

### H2: Penalized Spline (Semi-Parametric)
**Claim**: Piecewise smooth polynomials with shrinkage can capture acceleration without pre-specifying break location.

### H3: Polynomial Trend (Parametric Baseline)
**Claim**: Quadratic or cubic polynomial sufficient. Apparent break is just acceleration region of smooth curve.

---

## Model 1: Gaussian Process Negative Binomial Regression

### Mathematical Specification

**Likelihood**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = f(year_t) + ε_t
```

**Latent Function (GP prior)**:
```
f ~ GP(m(x), k(x, x'))

Mean function:
m(x) = β_0 + β_1 × year

Covariance function (Squared Exponential):
k(x, x') = σ²_f × exp(-ρ² × (x - x')²)
```

**Observation-level noise (for extra-GP autocorrelation)**:
```
ε_t ~ Normal(0, σ_ε)
OR
ε_t ~ AR(1): ε_t = ρ × ε_{t-1} + ν_t, ν_t ~ Normal(0, σ_ν)
```

**Dispersion**:
```
φ ~ Gamma(2, 1)  [overdispersion parameter]
```

### Prior Specifications

**Intercept (log-scale mean at year=0)**:
```
β_0 ~ Normal(4.3, 0.5)
```
*Justification*: EDA shows log(C) ≈ 4.31 at year=0. Allow ±1 on log scale (factor of ~3).

**Linear trend (captures overall drift)**:
```
β_1 ~ Normal(0.5, 0.5)
```
*Justification*: Overall log-scale slope is ~0.5-0.7 across full series. GP will add nonlinear deviations.

**GP Lengthscale (controls smoothness)**:
```
ℓ = 1/√(2ρ²)
ℓ ~ InverseGamma(5, 5)
```
*Justification*: Prior mean ℓ ≈ 1 (similar to year range), but allow ℓ ∈ (0.1, 3). Shorter = more flexible, longer = smoother.

**GP Marginal SD (amplitude of deviations)**:
```
σ_f ~ Normal(0, 0.5)  [half-normal]
```
*Justification*: Expect GP to capture ±0.5 log-scale deviations from linear trend (factor of ~1.6).

**Observation noise**:
```
σ_ε ~ Normal(0, 0.2)  [half-normal]
OR
ρ_AR ~ Beta(8, 2)  [if using AR(1)]
σ_ν ~ Normal(0, 0.2)
```
*Justification*: Strong autocorrelation (ACF=0.944) suggests ρ ≈ 0.8. GP captures smooth trend, AR captures short-term persistence.

**Dispersion**:
```
φ ~ Gamma(2, 1)
```
*Justification*: EDA shows α ≈ 0.61, φ = 1/α ≈ 1.63. Gamma(2,1) has mean=2, allows range [0.5, 5].

### Why This Model Might Work

1. **Flexibility**: GP can capture arbitrary smooth functions, including S-curves and acceleration
2. **Autocorrelation**: GP covariance naturally models temporal dependency
3. **No Break Assumption**: Doesn't impose discrete regime change
4. **Principled Uncertainty**: GP quantifies uncertainty about function shape
5. **Log-scale Linearity**: EDA shows r=0.967 on log scale, GP mean includes linear term

### Why This Model Might FAIL (Falsification Criteria)

**I will abandon this model if**:

1. **Computational Failure**:
   - GP inversion fails (matrix singularity)
   - Sampling doesn't converge (R-hat > 1.05 after 4000 iterations)
   - Extreme parameter values (ℓ → 0 or ℓ → ∞)

2. **Posterior-Prior Conflict**:
   - Lengthscale posterior pushed to boundary (ℓ < 0.1 or ℓ > 3)
   - GP amplitude σ_f → 0 (model doesn't need nonlinearity)
   - φ posterior radically different from prior (dispersion misspecified)

3. **Poor Predictive Performance**:
   - LOO-ELPD worse than polynomial model by >10
   - Posterior predictive checks show systematic bias around obs 17
   - Cannot capture sharp acceleration (residual plot shows discontinuity)

4. **Evidence for Discrete Break**:
   - GP function shows near-vertical slope at specific time point
   - Lengthscale shrinks to capture discontinuity (ℓ < 0.2)
   - Model tries to force smoothness where data wants jump

5. **Autocorrelation Misspecification**:
   - ACF of residuals still shows ρ > 0.5 at lag 1
   - GP alone insufficient (need additional AR term)
   - Observation noise dominates GP signal (σ_ε > σ_f)

**Decision Rule**: If LOO-ELPD is >20 worse than changepoint model AND lengthscale shrinks below 0.2, conclude discrete break is real and GP is wrong approach.

### Implementation Considerations

**Stan Implementation**:
```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
}

transformed data {
  real delta = 1e-9;  // jitter for numerical stability
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> length_scale;
  real<lower=0> sigma_f;
  real<lower=0> phi;
  vector[N] eta;  // unit normal for non-centered GP
}

transformed parameters {
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(year, sigma_f, length_scale);
    for (n in 1:N)
      K[n, n] = K[n, n] + delta;
    L_K = cholesky_decompose(K);
    f = beta_0 + beta_1 * year + L_K * eta;
  }
}

model {
  // Priors
  beta_0 ~ normal(4.3, 0.5);
  beta_1 ~ normal(0.5, 0.5);
  length_scale ~ inv_gamma(5, 5);
  sigma_f ~ normal(0, 0.5);
  phi ~ gamma(2, 1);
  eta ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2_log(f, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_log_lpmf(C[n] | f[n], phi);
    C_rep[n] = neg_binomial_2_log_rng(f[n], phi);
  }
}
```

**PyMC Implementation**:
```python
import pymc as pm
import numpy as np

with pm.Model() as gp_negbin_model:
    # Priors
    beta_0 = pm.Normal("beta_0", mu=4.3, sigma=0.5)
    beta_1 = pm.Normal("beta_1", mu=0.5, sigma=0.5)

    length_scale = pm.InverseGamma("length_scale", alpha=5, beta=5)
    sigma_f = pm.HalfNormal("sigma_f", sigma=0.5)

    # Mean function
    mean_func = beta_0 + beta_1 * year

    # GP covariance
    cov_func = sigma_f**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)

    # GP
    gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
    f = gp.prior("f", X=year[:, None])

    # Dispersion
    phi = pm.Gamma("phi", alpha=2, beta=1)

    # Likelihood
    C_obs = pm.NegativeBinomial("C_obs", mu=pm.math.exp(f), alpha=phi, observed=C)
```

**Computational Cost**:
- **Moderate to High** (O(N³) for GP matrix operations)
- Expect 5-15 minutes on modern CPU for N=40
- May need 2000-4000 iterations for convergence
- Use non-centered parameterization (done above)

**Numerical Stability**:
- Add jitter (1e-9) to diagonal for Cholesky decomposition
- Monitor condition number of covariance matrix
- If fails, try Hilbert-space approximation (reduces to O(N×M²))

### Stress Tests

1. **Smoothness Test**: Fit to synthetic data with true discrete break. If GP fits well, it's over-flexible (bad).
2. **Extrapolation Test**: Hold out last 5 observations. GP should fail dramatically if break is real.
3. **Lengthscale Sensitivity**: Try fixed ℓ ∈ {0.5, 1.0, 2.0}. If results radically different, model unstable.

---

## Model 2: Penalized B-Spline Negative Binomial Regression

### Mathematical Specification

**Likelihood**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = Σⱼ βⱼ × Bⱼ(year_t) + ε_t
```

**Spline Basis**:
```
{Bⱼ(x)}ⱼ₌₁ᴷ = B-spline basis functions of degree d=3 (cubic)
K = number of basis functions (knots + degree + 1)
```

**Coefficients with Smoothing Prior** (P-splines):
```
β ~ MVNormal(0, Σ_β)

Second-order random walk penalty:
Δ²βⱼ = βⱼ - 2×βⱼ₋₁ + βⱼ₋₂ ~ Normal(0, σ²_β)

Equivalent precision matrix:
Σ_β⁻¹ = (1/σ²_β) × D' × D
where D is second-difference matrix
```

**Smoothing parameter**:
```
σ_β ~ Normal(0, 1)  [half-normal, controls wiggliness]
```

**Autocorrelation** (AR(1) on observation level):
```
ε_t ~ AR(1): ε_t = ρ × ε_{t-1} + ν_t
ν_t ~ Normal(0, σ_ν)
ρ ~ Beta(8, 2)
σ_ν ~ Normal(0, 0.2)
```

**Dispersion**:
```
φ ~ Gamma(2, 1)
```

### Prior Specifications

**Number of knots**:
```
K = 8-12 basis functions
```
*Justification*: Rule of thumb: K ≈ N/4 to N/3 for N=40. Gives 1 knot per ~4 observations. Too few = underfit, too many = overfit (but penalty controls this).

**Knot locations**:
```
Uniform quantiles of year variable
OR
Concentrate more knots in suspected acceleration region (year > -0.2)
```

**Spline coefficients**:
```
β ~ MVNormal(0, Σ_β)
Σ_β controlled by smoothing parameter σ_β
```
*Justification*: Penalized regression. Prior allows flexibility but penalizes roughness.

**Smoothing parameter**:
```
σ_β ~ Normal(0, 1)  [half-normal]
```
*Justification*: Cross-validation suggests moderate smoothing. σ_β ≈ 0.5-2.0 expected. Smaller = smoother.

**AR(1) autocorrelation**:
```
ρ ~ Beta(8, 2)
σ_ν ~ Normal(0, 0.2)
```
*Justification*: High ACF(1)=0.944 in data. Beta(8,2) gives E[ρ]=0.8, SD=0.12. Allows ρ ∈ (0.5, 0.95).

**Dispersion**:
```
φ ~ Gamma(2, 1)
```

### Why This Model Might Work

1. **Local Flexibility**: Splines can accelerate/decelerate locally without global discontinuity
2. **Automatic Break Detection**: If real break exists, spline will allocate knots nearby
3. **Parsimony**: Penalty prevents overfitting, shrinks toward smooth function
4. **Computational Efficiency**: O(N) operations (sparse penalty matrix)
5. **Interpretability**: Can visualize which time regions have rapid change
6. **Separate Autocorrelation**: AR(1) term handles short-term persistence independently

### Why This Model Might FAIL (Falsification Criteria)

**I will abandon this model if**:

1. **Knot Sensitivity**:
   - Results change dramatically with K ∈ {8, 10, 12}
   - Posterior mass concentrates at knot boundaries (artifact)
   - Wiggliness around observation 17 regardless of smoothing

2. **Penalty Ineffective**:
   - σ_β → ∞ (no smoothing, overfitting)
   - σ_β → 0 (over-smoothed, reduces to linear)
   - Posterior predictive shows systematic bias

3. **Computational Issues**:
   - Sampling divergences > 1%
   - R-hat > 1.05 for spline coefficients
   - High correlation between adjacent βⱼ (reparameterization needed)

4. **Autocorrelation Residual**:
   - ACF(1) of residuals > 0.5 after accounting for AR(1)
   - Need higher-order AR or different structure

5. **Discrete Break Evidence**:
   - Posterior of spline function shows near-vertical slope at specific point
   - First derivative of fitted curve discontinuous
   - Cannot achieve smooth acceleration matching data

6. **Poor Model Comparison**:
   - LOO-ELPD worse than GP or polynomial by >15
   - High Pareto-k values (>0.7) indicating influential observations

**Decision Rule**: If spline function consistently shows discontinuous derivative at observation 17 across different K and σ_β settings, conclude smooth model inadequate.

### Implementation Considerations

**Stan Implementation**:
```stan
functions {
  matrix build_b_spline(vector x, int K, int degree) {
    // Implementation of B-spline basis
    // (Use existing Stan implementations or rstan package)
  }

  matrix difference_matrix(int K, int order) {
    // Build D matrix for order-th differences
    matrix[K - order, K] D;
    // ... implementation
    return D;
  }
}

data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
  int<lower=1> K;  // number of basis functions
  int<lower=1> degree;  // spline degree (3 for cubic)
}

transformed data {
  matrix[N, K] B = build_b_spline(year, K, degree);
  matrix[K-2, K] D = difference_matrix(K, 2);
}

parameters {
  vector[K] beta;
  real<lower=0> sigma_beta;
  real<lower=0> phi;
  real<lower=0, upper=1> rho;
  real<lower=0> sigma_nu;
  vector[N] epsilon_raw;
}

transformed parameters {
  vector[N] f;
  vector[N] epsilon;

  // AR(1) process
  epsilon[1] = epsilon_raw[1] * sigma_nu / sqrt(1 - rho^2);
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + epsilon_raw[t] * sigma_nu;
  }

  // Spline + AR errors
  f = B * beta + epsilon;
}

model {
  // Smoothing prior on spline coefficients
  target += -0.5 * dot_self(D * beta) / (sigma_beta^2);

  // Priors
  sigma_beta ~ normal(0, 1);
  phi ~ gamma(2, 1);
  rho ~ beta(8, 2);
  sigma_nu ~ normal(0, 0.2);
  epsilon_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2_log(f, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  vector[N] mu = exp(f);

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_log_lpmf(C[n] | f[n], phi);
    C_rep[n] = neg_binomial_2_log_rng(f[n], phi);
  }
}
```

**PyMC Implementation**:
```python
import pymc as pm
from patsy import dmatrix

# Build B-spline basis
knots = np.quantile(year, np.linspace(0, 1, K-4))  # interior knots
B = dmatrix(
    f"bs(year, knots={list(knots)}, degree=3, include_intercept=True)",
    {"year": year},
    return_type="dataframe"
).values

# Build second-difference penalty matrix
D = np.diff(np.eye(K), n=2, axis=0)

with pm.Model() as spline_negbin_model:
    # Smoothing parameter
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)

    # Spline coefficients with penalty
    Sigma_beta = sigma_beta**2 * np.eye(K)  # simplified
    beta = pm.MvNormal("beta", mu=np.zeros(K), cov=Sigma_beta)

    # AR(1) parameters
    rho = pm.Beta("rho", alpha=8, beta=2)
    sigma_nu = pm.HalfNormal("sigma_nu", sigma=0.2)

    # AR(1) process
    epsilon = pm.AR("epsilon", rho=rho, sigma=sigma_nu, shape=N)

    # Linear predictor
    f = pm.math.dot(B, beta) + epsilon

    # Dispersion
    phi = pm.Gamma("phi", alpha=2, beta=1)

    # Likelihood
    C_obs = pm.NegativeBinomial("C_obs", mu=pm.math.exp(f), alpha=phi, observed=C)
```

**Computational Cost**:
- **Moderate** (O(N×K²) operations)
- Expect 3-10 minutes for N=40, K=10
- Faster than GP but slower than polynomial
- May have sampling issues if K too large

**Tuning Recommendations**:
- Start with K=10
- Try knot locations: uniform vs. clustered in acceleration region
- If divergences occur, increase adapt_delta to 0.95
- Use QR decomposition of B matrix for stability

### Stress Tests

1. **Knot Variation**: Fit with K ∈ {6, 8, 10, 12, 15}. LOO-ELPD should plateau.
2. **Penalty Sensitivity**: Fix σ_β ∈ {0.5, 1.0, 2.0}. If results wildly different, penalty poorly calibrated.
3. **Derivative Test**: Compute first derivative of fitted spline. Should be continuous if model correct.

---

## Model 3: Polynomial Negative Binomial Regression with AR Errors

### Mathematical Specification

**Likelihood**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β_0 + β_1×year_t + β_2×year_t² + β_3×year_t³ + ε_t
```

**Autocorrelation** (AR(1)):
```
ε_t ~ AR(1): ε_t = ρ × ε_{t-1} + ν_t
ν_t ~ Normal(0, σ_ν)
```

**Dispersion**:
```
φ ~ Gamma(2, 1)
```

### Prior Specifications

**Intercept**:
```
β_0 ~ Normal(4.3, 0.5)
```
*Justification*: log(C) ≈ 4.31 at year=0

**Linear coefficient**:
```
β_1 ~ Normal(0.5, 0.5)
```
*Justification*: Overall trend is positive with moderate slope

**Quadratic coefficient**:
```
β_2 ~ Normal(0, 0.5)
```
*Justification*: EDA shows quadratic improves ΔAIC=-41.4. Expect β_2 > 0 (acceleration).

**Cubic coefficient**:
```
β_3 ~ Normal(0, 0.3)
```
*Justification*: EDA shows cubic R²=0.976. Allow flexibility but weaker prior than quadratic.

**AR(1) parameters**:
```
ρ ~ Beta(8, 2)
σ_ν ~ Normal(0, 0.2)
```
*Justification*: Strong autocorrelation expected

**Dispersion**:
```
φ ~ Gamma(2, 1)
```

### Why This Model Might Work

1. **Simplicity**: Fewest parameters, easiest to interpret
2. **EDA Support**: Quadratic ΔAIC=-41.4, cubic R²=0.976
3. **Fast Computation**: O(N) operations, converges quickly
4. **Stable**: No matrix inversions or basis functions
5. **Log-scale Linearity**: EDA shows r=0.967, polynomial close to linear on log scale
6. **Extrapolation**: Polynomial behavior well-defined outside observed range

### Why This Model Might FAIL (Falsification Criteria)

**I will abandon this model if**:

1. **Systematic Bias**:
   - Residual plot shows clear pattern around observation 17
   - Posterior predictive checks fail (cannot capture local acceleration)
   - ACF of residuals shows structure beyond AR(1)

2. **Model Comparison**:
   - LOO-ELPD significantly worse than GP or spline (>20 difference)
   - High Pareto-k values (>0.7) for multiple observations
   - WAIC strongly prefers more flexible models

3. **Polynomial Inadequacy**:
   - Cubic not sufficient (need quartic or higher)
   - Polynomial shows inflection points where data doesn't
   - Cannot capture sharp transition without extreme coefficients

4. **Parameter Instability**:
   - Posterior of β_3 includes zero (cubic not needed)
   - High correlation between β_1, β_2, β_3 (multicollinearity)
   - Priors dominating posterior (model can't learn from data)

5. **Autocorrelation Misspecification**:
   - AR(1) insufficient (residual ACF significant at lag 2+)
   - ρ posterior pushed to boundary (ρ > 0.95)
   - Need AR(p) with p > 1

6. **Extrapolation Disaster**:
   - Polynomial goes negative or explodes beyond observed range
   - Physically implausible predictions (count model constraint violated)

**Decision Rule**: If LOO-ELPD is >20 worse than GP AND residuals show discontinuity at observation 17, conclude polynomial insufficient.

**Critical Insight**: This model serves as parametric baseline. If it performs similarly to GP/spline, suggests smooth acceleration (not discrete break). If much worse, confirms need for local flexibility.

### Implementation Considerations

**Stan Implementation**:
```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real beta_3;
  real<lower=0> phi;
  real<lower=0, upper=1> rho;
  real<lower=0> sigma_nu;
  vector[N] epsilon_raw;
}

transformed parameters {
  vector[N] epsilon;
  vector[N] f;

  // AR(1) process (non-centered)
  epsilon[1] = epsilon_raw[1] * sigma_nu / sqrt(1 - rho^2);
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + epsilon_raw[t] * sigma_nu;
  }

  // Polynomial trend
  f = beta_0 + beta_1 * year + beta_2 * square(year) + beta_3 * year .* square(year) + epsilon;
}

model {
  // Priors
  beta_0 ~ normal(4.3, 0.5);
  beta_1 ~ normal(0.5, 0.5);
  beta_2 ~ normal(0, 0.5);
  beta_3 ~ normal(0, 0.3);
  phi ~ gamma(2, 1);
  rho ~ beta(8, 2);
  sigma_nu ~ normal(0, 0.2);
  epsilon_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2_log(f, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  vector[N] mu = exp(f);

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_log_lpmf(C[n] | f[n], phi);
    C_rep[n] = neg_binomial_2_log_rng(f[n], phi);
  }
}
```

**PyMC Implementation**:
```python
import pymc as pm

with pm.Model() as poly_negbin_model:
    # Polynomial coefficients
    beta_0 = pm.Normal("beta_0", mu=4.3, sigma=0.5)
    beta_1 = pm.Normal("beta_1", mu=0.5, sigma=0.5)
    beta_2 = pm.Normal("beta_2", mu=0, sigma=0.5)
    beta_3 = pm.Normal("beta_3", mu=0, sigma=0.3)

    # AR(1) parameters
    rho = pm.Beta("rho", alpha=8, beta=2)
    sigma_nu = pm.HalfNormal("sigma_nu", sigma=0.2)

    # AR(1) process
    epsilon = pm.AR("epsilon", rho=rho, sigma=sigma_nu, shape=N)

    # Polynomial trend
    f = beta_0 + beta_1 * year + beta_2 * year**2 + beta_3 * year**3 + epsilon

    # Dispersion
    phi = pm.Gamma("phi", alpha=2, beta=1)

    # Likelihood
    C_obs = pm.NegativeBinomial("C_obs", mu=pm.math.exp(f), alpha=phi, observed=C)
```

**Computational Cost**:
- **Low** (O(N) operations)
- Expect 2-5 minutes for convergence
- Fast sampling, minimal memory
- Should converge in 1000-2000 iterations

**Numerical Considerations**:
- Use orthogonal polynomials if multicollinearity issues
- Center and scale year variable (already done in data)
- Monitor correlation matrix of parameters

### Stress Tests

1. **Degree Selection**: Compare quadratic vs cubic via LOO. If cubic not needed (β_3 ≈ 0), use simpler model.
2. **Residual Analysis**: ACF of residuals should be white noise after AR(1). If not, model inadequate.
3. **Posterior Predictive**: Generate C_rep and compare to observed. Should match growth pattern.

---

## Model Comparison Strategy

### Falsification Hierarchy

**Stage 1: Individual Model Validation**
- Each model must pass its own falsification tests
- If model fails, document why and move to next

**Stage 2: Head-to-Head Comparison**
- LOO-ELPD (primary metric, accounts for model complexity)
- WAIC (secondary metric)
- Posterior predictive checks (qualitative assessment)

**Stage 3: Decision Criteria**

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| Polynomial ≈ Spline ≈ GP | Smooth acceleration sufficient | Use simplest (polynomial) |
| GP >> Polynomial, GP ≈ Spline | Local flexibility needed | Use GP or Spline |
| All << Changepoint model | Discrete break real | ABANDON smooth models |
| GP fails computationally | N too small for GP | Use Spline or Polynomial |
| Spline unstable | Knot sensitivity | Use GP or Polynomial |

**CRITICAL**: If smooth models' LOO-ELPD is >20 worse than changepoint model (from Designer 1), conclude:
1. Discrete regime change is real phenomenon
2. Smooth functions inadequate
3. Structural break model is correct

### Red Flags (Switch Model Classes Entirely)

1. **All smooth models fail**: Discrete break confirmed, need changepoint models
2. **GP lengthscale → 0**: Trying to capture discontinuity, use discrete model
3. **Polynomial needs degree >4**: Over-parameterized, use changepoint
4. **Residual autocorrelation persists**: Need time-series state-space model
5. **Dispersion time-varying**: Need hierarchical dispersion model

### Stress Tests for All Models

1. **Leave-Future-Out Cross-Validation**:
   - Hold out last 5 observations
   - Smooth models should fail if break is recent
   - If all smooth models fail dramatically, break is real

2. **Synthetic Data Test**:
   - Generate data with known discrete break
   - Fit smooth models
   - If they fit well (bad!), they're over-flexible

3. **First Derivative Test**:
   - Compute d(log μ)/d(year) for each model
   - If discontinuous at observation 17, smooth model fails
   - If continuous but very steep, smooth model may work

4. **Autocorrelation Check**:
   - ACF of residuals should be white noise
   - If not, model misspecified (regardless of LOO-ELPD)

---

## Implementation Roadmap

### Phase 1: Baseline (Polynomial)
**Time**: 2-3 hours
1. Fit polynomial (quadratic + cubic)
2. Check convergence, residuals, LOO
3. Document performance (baseline)

### Phase 2: Flexible Models (GP or Spline)
**Time**: 4-6 hours
1. Fit GP model
   - If computational issues, try Hilbert approximation
   - If still fails, skip to Spline
2. Fit Spline model
   - Try K ∈ {8, 10, 12}
   - Select best via LOO
3. Compare GP vs Spline vs Polynomial

### Phase 3: Model Comparison
**Time**: 2-3 hours
1. LOO-ELPD comparison table
2. Posterior predictive checks
3. Residual analysis
4. Falsification assessment

### Phase 4: Decision
**Time**: 1-2 hours
1. If smooth models competitive: proceed with best
2. If smooth models fail: document why and recommend changepoint models
3. Write final report with model selection justification

**Total Time Estimate**: 10-15 hours

---

## Expected Outcomes

### Scenario A: Smooth Models Succeed
**Evidence**:
- GP or Spline LOO-ELPD within 10 of polynomial
- All within 20 of changepoint model
- Residuals show no discontinuity

**Conclusion**:
- Apparent break is smooth acceleration
- Recommend GP (most flexible) or Spline (faster)

### Scenario B: Smooth Models Fail
**Evidence**:
- All smooth models LOO-ELPD >20 worse than changepoint
- Residuals show systematic bias at observation 17
- GP lengthscale shrinks to capture discontinuity

**Conclusion**:
- Discrete regime change is real
- Smooth functions inadequate
- Recommend changepoint models from Designer 1

### Scenario C: Mixed Results
**Evidence**:
- GP works, Polynomial fails
- Or: Spline works with many knots, Polynomial fails

**Conclusion**:
- Local flexibility needed but break may not be discrete
- Recommend GP or heavily penalized Spline
- Compare carefully to changepoint model

---

## Scientific Plausibility

### Why Smooth Acceleration Might Be Real

1. **Exponential growth**: If underlying rate increases exponentially, appears like discrete break in small sample
2. **Diffusion process**: Adoption/contagion models show S-curves (smooth)
3. **Resource constraints relaxing**: Gradual removal of bottleneck appears as acceleration
4. **Measurement aggregation**: True discrete micro-events appear smooth when aggregated

### Why Discrete Break Might Be Real

1. **Policy change**: Regulatory/institutional change at specific time
2. **Technology shock**: New innovation adopted at specific point
3. **External event**: War, disaster, economic crisis
4. **Phase transition**: System crosses threshold (critical point)

**Critical Question**: Does domain knowledge suggest discrete event or gradual process?

---

## Recommendations for Implementation

### DO:
1. Fit all three models (computational cost manageable for N=40)
2. Use LOO-ELPD as primary comparison metric
3. Check residual autocorrelation for all models
4. Perform posterior predictive checks
5. Compare to changepoint models from Designer 1
6. Be ready to conclude smooth models are wrong

### DON'T:
1. Force smooth model if evidence favors discrete break
2. Ignore computational failures (often indicate misspecification)
3. Rely solely on in-sample fit (use cross-validation)
4. Extrapolate polynomials beyond observed range
5. Use Poisson likelihood (overdispersion is real)

---

## Falsification Summary

### Model 1 (GP): Abandon if
- Computational failure (matrix issues)
- Lengthscale → 0 (trying to capture discontinuity)
- LOO-ELPD >20 worse than changepoint model
- Residual ACF > 0.5

### Model 2 (Spline): Abandon if
- Knot sensitivity (results change drastically with K)
- Discontinuous derivative at observation 17
- LOO-ELPD >15 worse than GP
- Sampling divergences > 1%

### Model 3 (Polynomial): Abandon if
- LOO-ELPD >20 worse than GP
- Residuals show discontinuity
- Need degree >4 for reasonable fit
- Posterior predictive checks fail

### ALL Smooth Models: Abandon if
- Changepoint model LOO-ELPD >20 better
- Leave-future-out CV fails dramatically
- Residual patterns at observation 17 across all models
- Domain knowledge suggests discrete event

---

## Final Notes

**Philosophy**: These models assume smooth acceleration. If data truly has discrete break, smooth models will fail—and that's the correct scientific conclusion.

**Success Metric**: Not whether smooth models work, but whether we correctly identify if they should work.

**Decision Rule**: Trust the data. If LOO strongly favors discrete break, accept it.

**Next Steps**: Implement models, run comparisons, be ready to pivot to changepoint models if evidence demands.

---

**File**: `/workspace/experiments/designer_2/proposed_models.md`
**Status**: Ready for implementation and falsification testing
