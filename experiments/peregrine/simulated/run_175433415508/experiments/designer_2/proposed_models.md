# Bayesian Models with Temporal Correlation Structures

**Designer**: Model Designer 2 (Temporal Specialist)
**Focus**: Temporal dependence and autocorrelation structures
**Date**: 2025-10-29

---

## Executive Summary

This document proposes **3 Bayesian models** that explicitly model the extremely high temporal autocorrelation (ACF = 0.971) observed in this count time series. All models handle the fundamental data characteristics:
- Severe overdispersion (Var/Mean = 70.43)
- Strong exponential growth (R² = 0.937)
- **Near-perfect temporal dependence (ACF = 0.971)** ← PRIMARY FOCUS

### Critical Challenge: Separating Trend from Correlation

The extraordinarily high ACF = 0.971 creates a fundamental **identifiability problem**: is the apparent correlation due to:
1. **Smooth deterministic trend** (missing functional form)
2. **Genuine stochastic dependence** (AR process)
3. **Both** (trend + residual autocorrelation)

This is the central question these models must address.

### Model Overview

| Model | Correlation Structure | Innovation | Risk Level |
|-------|----------------------|------------|------------|
| NB-AR1 | Autoregressive latent | Standard approach | Medium |
| NB-GP | Gaussian Process | Flexible trend capture | High |
| NB-RW | Random walk | State-space framework | Medium |

---

## Critical Assumptions and Falsification Strategy

### Hypothesis 1: High ACF is Genuine Stochastic Dependence
**Prediction**: After accounting for trend, residual autocorrelation remains high (ρ > 0.5)

**REJECT IF**:
- Posterior for ρ: P(ρ < 0.3) > 0.95
- Quadratic trend reduces ACF to < 0.3
- Adding year² term makes AR(1) component unnecessary

**Evidence this would provide**: The high ACF is primarily a **trend artifact**, not genuine temporal dependence

### Hypothesis 2: Linear Trend is Sufficient
**Prediction**: Linear μ_t = β₀ + β₁×year_t with AR(1) errors fits well

**REJECT IF**:
- Posterior predictive checks show systematic deviations
- Residual ACF at high lags remains > 0.5
- Model comparison strongly favors GP (ΔELPD > 10)

**Evidence this would provide**: The growth process is **inherently non-linear** and requires flexible modeling

### Hypothesis 3: Count Distribution Matters for Correlation
**Prediction**: Modeling correlation on latent Gaussian scale vs observation scale gives different results

**REJECT IF**:
- Models with latent correlation show computational pathologies (divergences, Rhat > 1.1)
- Posterior predictions are insensitive to correlation structure
- Negative Binomial α → 0 (model wants to be Poisson)

**Evidence this would provide**: Overdispersion and temporal correlation are **confounded** in this data

---

## Model 1: Negative Binomial with AR(1) Latent Process

### Theoretical Justification

This is the **standard approach** for count time series with overdispersion and autocorrelation. The key insight is that we model correlation on a **latent Gaussian scale** that influences the count mean through the log link.

**Why this might work**:
- Separates deterministic trend (β₀ + β₁×year) from stochastic noise (AR process)
- Latent Gaussian assumption is computationally tractable
- Well-established in ecological time series (Morris & Doak 2002)

**Why this might FAIL**:
- ACF = 0.971 is **extreme** - may indicate missing trend terms, not correlation
- Identifiability issues: ρ → 1 means nearly deterministic, hard to distinguish from trend
- Latent AR(1) with NB may be **overparameterized** for n=40

### Mathematical Specification

#### Likelihood
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = η_t
η_t ~ Normal(β₀ + β₁×year_t, σ_η)  [latent process]

η_t | η_{t-1} ~ Normal(β₀ + β₁×year_t + ρ(η_{t-1} - [β₀ + β₁×year_{t-1}]), σ_η√(1-ρ²))
```

#### Reparameterization for Computation
```
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t
ν_t ~ Normal(0, σ_η)
```

Where:
- **η_t**: Latent log-mean at time t
- **ρ**: Autoregressive coefficient (correlation parameter)
- **σ_η**: Innovation standard deviation
- **α**: Negative Binomial dispersion parameter

#### Prior Specifications

```stan
// Regression parameters
β₀ ~ Normal(log(109.4), 1)      // Intercept: log of observed mean
β₁ ~ Normal(1.0, 0.5)            // Growth: positive, moderate

// Dispersion
α ~ Gamma(2, 0.1)                // E[α] = 20, favors overdispersion

// AR(1) correlation - KEY PRIOR
ρ ~ Beta(20, 2)                  // E[ρ] = 0.91, SD = 0.06
                                 // Prior mode ≈ 0.95 (informed by ACF = 0.971)
                                 // ALLOWS ρ to be lower if data supports it

// Innovation variance
σ_η ~ Exponential(10)            // Small innovations expected (trend dominates)
```

#### Prior Rationale: The ρ Prior is Critical

**Why Beta(20, 2)?**
- Observed ACF = 0.971 suggests very high correlation
- But this could be trend artifact
- Beta(20, 2) centers at 0.91 but allows full range [0.7, 0.99]
- If true ρ is lower, posterior will shrink away from prior
- If true ρ is ~0.97, posterior will match data

**Alternative priors to test**:
1. **Skeptical**: Beta(5, 5) [E = 0.5] - assumes most ACF is trend
2. **Agnostic**: Uniform(0, 1) - let data dominate
3. **Informed**: Beta(30, 2) [tighter around 0.94] - trust the ACF

### Temporal Correlation Details

#### Structure
AR(1) on latent log-scale:
```
Corr(η_t, η_{t+k}) = ρ^k
```

For ρ = 0.97:
- Lag 1: 0.970
- Lag 5: 0.859
- Lag 10: 0.737

#### Implementation Strategy
Use **Cholesky factorization** of correlation matrix for efficiency:

```stan
transformed parameters {
  vector[N] eta;  // latent log-mean

  // Construct correlated latent process
  eta[1] = beta0 + beta1 * year[1] + sigma_eta * epsilon_raw[1] / sqrt(1 - rho^2);
  for (t in 2:N) {
    eta[t] = beta0 + beta1 * year[t] +
             rho * (eta[t-1] - beta0 - beta1 * year[t-1]) +
             sigma_eta * epsilon_raw[t];
  }
}
```

Where `epsilon_raw[t] ~ Normal(0, 1)` are independent standard normals.

### Prior Predictive Distribution

**Expected autocorrelation**: 0.85 to 0.98 (95% prior interval for ρ)

**Implied variance structure**:
- Total variance = trend variance + AR(1) variance
- AR(1) component contributes σ_η² / (1 - ρ²)
- For ρ = 0.97, this amplifies variance by factor of ~17

**Prior predictive checks should show**:
- Counts ranging 10-500 (order of magnitude correct)
- Strong positive time trend
- Smooth trajectories (high ρ means little noise)

### Falsification Criteria

**REJECT this model if**:

1. **Identifiability failure**:
   - Rhat > 1.05 for ρ or σ_η
   - ESS < 100 for correlation parameters
   - Posterior correlations |r(ρ, σ_η)| > 0.95

2. **Correlation unnecessary**:
   - 95% CI for ρ includes values < 0.5
   - PSIS-LOO prefers model without AR(1)
   - Removing ρ changes ELPD by < 2

3. **Wrong correlation structure**:
   - Posterior predictive ACF shows **faster decay** than AR(1) (suggests MA or higher-order)
   - Posterior predictive ACF shows **slower decay** (suggests non-stationary process)
   - Residual ACF at lag > 5 remains > 0.3

4. **Computational pathologies**:
   - >1% divergent transitions (even after adapt_delta = 0.99)
   - Extremely slow sampling (< 10 iter/sec)
   - Posterior concentrates at boundary (ρ → 1.0)

### Stan Implementation Notes

#### Full Model Code Skeleton

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta0;
  real beta1;
  real<lower=0> alpha;         // NB dispersion
  real<lower=0, upper=1> rho;  // AR(1) correlation
  real<lower=0> sigma_eta;     // Innovation SD
  vector[N] epsilon_raw;        // Raw innovations
}

transformed parameters {
  vector[N] eta;  // Latent log-mean
  vector[N] mu;   // Expected count

  // AR(1) construction
  eta[1] = beta0 + beta1 * year[1] +
           sigma_eta * epsilon_raw[1] / sqrt(1 - rho^2);
  for (t in 2:N) {
    eta[t] = beta0 + beta1 * year[t] +
             rho * (eta[t-1] - beta0 - beta1 * year[t-1]) +
             sigma_eta * epsilon_raw[t];
  }

  mu = exp(eta);
}

model {
  // Priors
  beta0 ~ normal(log(109.4), 1);
  beta1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);
  rho ~ beta(20, 2);           // KEY PRIOR
  sigma_eta ~ exponential(10);
  epsilon_raw ~ std_normal();  // Independent innovations

  // Likelihood
  C ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;  // Posterior predictive

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], alpha);
    C_rep[t] = neg_binomial_2_rng(mu[t], alpha);
  }
}
```

#### Computational Challenges

1. **Correlation near 1**: When ρ → 1, innovation variance σ_η → 0 causes numerical issues
   - Solution: Use variance = σ_η² / (1 - ρ²) as primary parameter

2. **Initial condition**: First observation has special variance scaling
   - Solution: Multiply by 1/√(1-ρ²) to achieve stationarity

3. **Sampling efficiency**: High correlation creates dependence in posterior
   - Solution: Use `adapt_delta = 0.95`, `max_treedepth = 12`

### Computational Cost

**Expected runtime**: 2-4 minutes (4 chains × 2000 iterations)

**Sampling challenges**:
- **High**: Correlation near boundary (ρ ~ 1) may cause slow mixing
- **Medium**: Latent states (N=40 extra parameters) increase dimensionality
- **Low**: Model is not hierarchical, should be tractable

**Mitigation strategies**:
- Reparameterize ρ on logit scale if boundary issues arise
- Consider non-centered parameterization for epsilon_raw
- Monitor ESS carefully - if < 400, increase iterations

---

## Model 2: Negative Binomial with Gaussian Process Trend

### Theoretical Justification

Instead of assuming **parametric trend + AR(1) errors**, this model uses a **Gaussian Process** to flexibly capture both trend and correlation structure. This is philosophically different: we're saying "we don't know the functional form, let the data decide."

**Why this might work**:
- No need to pre-specify linear vs quadratic vs cubic
- GP naturally captures smooth trends (what high ACF might actually represent)
- Can model non-stationary correlation (correlation depends on time distance)
- Automatically does model selection over trend complexity

**Why this might FAIL**:
- **Overparameterized** for n=40 (GP has effectively ~10-20 free parameters)
- Lengthscale may be **unidentifiable** from measurement noise
- Computational cost is O(N³) for matrix operations
- May fit noise if priors are too vague
- High ACF might make GP think everything is perfectly correlated → constant function

### Mathematical Specification

#### Likelihood

```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = f(t) + ε_t
f ~ GP(m(t), K(t, t'))

where:
  m(t) = β₀ + β₁×year_t           [mean function: linear trend]
  K(t, t') = η² × exp(-||t-t'||² / (2ℓ²))  [squared exponential kernel]
  ε_t ~ Normal(0, σ_ε)             [measurement noise]
```

#### Kernel Parameterization

**Squared Exponential (RBF) Kernel**:
```
K(year_i, year_j) = η² × exp(-(year_i - year_j)² / (2ℓ²))
```

Parameters:
- **η² (amplitude)**: Maximum covariance (how much GP can deviate from mean)
- **ℓ (lengthscale)**: Correlation decay rate in year units
- **σ_ε (nugget)**: Independent observation noise

#### Connection to ACF

For a GP with SE kernel:
- Correlation at distance d: ρ(d) = exp(-d² / (2ℓ²))
- Observed ACF(1) = 0.971 → exp(-0.0855² / (2ℓ²)) ≈ 0.971
- Solving: ℓ ≈ 3.0 (in standardized year units)

This means correlation extends across **entire time series** (range = -1.67 to 1.67).

#### Prior Specifications

```stan
// Mean function (linear trend)
β₀ ~ Normal(log(109.4), 1)
β₁ ~ Normal(1.0, 0.5)

// GP hyperparameters
η ~ Exponential(1)              // Marginal SD: E[η] = 1
                                 // Allows GP to deviate ±1 on log scale
                                 // (factor of 2.7× on count scale)

ℓ ~ InverseGamma(5, 5)          // Lengthscale: E[ℓ] = 1.25
                                 // Prior mean implies ACF(1) ≈ 0.998 (!)
                                 // Wide tails allow ℓ ∈ [0.5, 5]

σ_ε ~ Exponential(5)            // Nugget: E[σ_ε] = 0.2 (small)
                                 // GP should explain most variation

// Dispersion
α ~ Gamma(2, 0.1)               // NB overdispersion
```

#### Prior Rationale: Lengthscale is Critical

The lengthscale ℓ controls how quickly correlation decays:
- **ℓ = 0.5**: Correlation decays rapidly, ACF(1) ≈ 0.9
- **ℓ = 1.0**: Medium decay, ACF(1) ≈ 0.97
- **ℓ = 5.0**: Very slow decay, nearly constant function

**Key insight**: Given ACF = 0.971, the data will push toward **large ℓ**, meaning the GP will be **nearly constant** (perfectly smooth). This might reveal that the high ACF is just a smooth trend artifact.

### Temporal Correlation Details

#### Structure: Flexible Non-Stationary

Unlike AR(1), GP correlation is **not constant** over time:
```
Corr(f(t₁), f(t₂)) = exp(-(t₁ - t₂)² / (2ℓ²))
```

This allows:
- Different correlation between early vs late periods
- Smooth interpolation between observations
- Extrapolation uncertainty (grows outside observed range)

#### Comparison to AR(1)

| Property | AR(1) | GP (SE kernel) |
|----------|-------|----------------|
| Correlation form | ρ^k | exp(-d²/2ℓ²) |
| Stationarity | Yes | Yes (in this form) |
| Parameters | 1 (ρ) | 2 (η, ℓ) + nugget |
| Flexibility | Low | High |
| Computation | O(N) | O(N³) |

### Prior Predictive Distribution

**Expected behavior**:
- Smooth functions with gentle curvature
- Lengthscale prior favors correlation range of 0.5-3 standardized years
- Amplitude prior allows ±2.7× deviations from linear trend
- Should generate realistic-looking smooth time series

**Prior predictive checks should show**:
- Diverse functional forms (some nearly linear, some curved)
- All positive counts (exp transform)
- Some trajectories wildly off (prior is intentionally vague)

### Falsification Criteria

**REJECT this model if**:

1. **Overfit to noise**:
   - Posterior lengthscale < 0.3 (too wiggly)
   - Posterior checks show irregular, non-smooth predictions
   - GP captures individual points but poor out-of-sample

2. **Underfit (constant function)**:
   - Posterior lengthscale > 10 (effectively infinite correlation)
   - Posterior reduces to linear trend (η ≈ 0, σ_ε dominates)
   - GP provides no improvement over linear model

3. **Identifiability problems**:
   - Strong posterior correlation between η and σ_ε (> 0.9)
   - Lengthscale posterior matches prior (no learning)
   - Multiple posterior modes in MCMC chains

4. **Model comparison**:
   - PSIS-LOO worse than simple NB-AR1 (ΔELPD < -5)
   - Many high Pareto-k values (> 0.7) indicating instability
   - Posterior predictions no smoother than AR(1) model

5. **Computational failure**:
   - Cholesky decomposition numerical errors
   - Sampling extremely slow (< 1 iter/sec)
   - Cannot reach Rhat < 1.05 even with long runs

### Stan Implementation Notes

#### Computational Strategy

For N=40, can use **exact GP** (not approximations). Key steps:

1. **Build covariance matrix**:
```stan
matrix[N, N] K = gp_exp_quad_cov(year, eta, length_scale);
for (n in 1:N) K[n, n] += sigma_epsilon^2;  // Add nugget
```

2. **Cholesky decomposition**:
```stan
matrix[N, N] L = cholesky_decompose(K);
```

3. **Sample latent function**:
```stan
vector[N] f = mean_function + L * epsilon_raw;  // epsilon_raw ~ std_normal()
```

#### Full Model Code Skeleton

```stan
functions {
  // Helper for squared exponential kernel
  matrix gp_se_cov(vector x, real alpha, real length_scale, real nugget) {
    int N = size(x);
    matrix[N, N] K;
    for (i in 1:N) {
      K[i, i] = alpha^2 + nugget^2;
      for (j in (i+1):N) {
        K[i, j] = alpha^2 * exp(-square(x[i] - x[j]) / (2 * square(length_scale)));
        K[j, i] = K[i, j];
      }
    }
    return K;
  }
}

data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta0;
  real beta1;
  real<lower=0> alpha;           // NB dispersion
  real<lower=0> eta;             // GP amplitude
  real<lower=0> length_scale;    // GP lengthscale
  real<lower=0> sigma_epsilon;   // Nugget
  vector[N] epsilon_raw;          // Raw GP draws
}

transformed parameters {
  vector[N] f;  // Latent GP function
  vector[N] mu; // Expected count

  {
    vector[N] mean_func = beta0 + beta1 * year;
    matrix[N, N] K = gp_se_cov(year, eta, length_scale, sigma_epsilon);
    matrix[N, N] L = cholesky_decompose(K);
    f = mean_func + L * epsilon_raw;
  }

  mu = exp(f);
}

model {
  // Priors
  beta0 ~ normal(log(109.4), 1);
  beta1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);
  eta ~ exponential(1);
  length_scale ~ inv_gamma(5, 5);
  sigma_epsilon ~ exponential(5);
  epsilon_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], alpha);
    C_rep[t] = neg_binomial_2_rng(mu[t], alpha);
  }
}
```

#### Numerical Stability

**Critical issues**:
1. **Cholesky failures**: If K is nearly singular (ℓ → ∞ or nugget → 0)
   - Solution: Add jitter (K[i,i] += 1e-9) before decomposition

2. **Memory**: N×N matrices (40×40 = 1600 elements, negligible)

3. **Gradient computation**: Stan autodiff through Cholesky is efficient

### Computational Cost

**Expected runtime**: 5-10 minutes (4 chains × 2000 iterations)

**Sampling challenges**:
- **HIGH**: Matrix operations in each iteration (slower than AR1)
- **MEDIUM**: Correlation between η and lengthscale
- **LOW**: Only 40 data points, not a "big data" problem

**Memory**: ~10 MB (N×N matrices are small)

---

## Model 3: Negative Binomial with Random Walk Prior

### Theoretical Justification

This model takes a **state-space** perspective: the true log-mean μ_t evolves as a **random walk** plus a deterministic trend. This is philosophically different from both AR(1) and GP:

- **AR(1)**: Errors revert to trend (mean-reverting)
- **GP**: Smooth deviation from trend (stationary around trend)
- **Random walk**: Innovations accumulate, non-stationary (integrated process)

**Why this might work**:
- High ACF = 0.971 could indicate **near-unit-root** process (ρ ≈ 1)
- Random walk is the **limit** of AR(1) as ρ → 1
- State-space framework is standard for count time series (Durbin & Koopman 2012)
- Naturally handles **time-varying** trend (not fixed linear)

**Why this might FAIL**:
- n=40 is **small** for estimating non-stationary process
- Random walk may **wander** far from plausible values
- Confounds deterministic trend with stochastic drift
- If true process is stationary AR(1), RW will overfit
- May not be identifiable from smooth trend

### Mathematical Specification

#### State-Space Formulation

**Observation equation**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = θ_t
```

**State equation**:
```
θ_t = β₀ + β₁×year_t + ξ_t    [deterministic trend + random walk]
ξ_t = ξ_{t-1} + ω_t
ω_t ~ Normal(0, σ_ω)           [state innovation]
ξ_0 = 0                        [initial condition]
```

#### Equivalent Representation

This is equivalent to:
```
θ_t - θ_{t-1} = β₁×Δyear_t + ω_t
```

So the **first difference** is a random walk around a linear drift.

#### Prior Specifications

```stan
// Deterministic trend
β₀ ~ Normal(log(109.4), 1)      // Intercept
β₁ ~ Normal(1.0, 0.5)            // Drift rate

// State innovation
σ_ω ~ Exponential(10)           // Small innovations (similar to AR1 σ_η)
                                 // E[σ_ω] = 0.1 on log scale
                                 // Allows ~10% deviations per step

// Dispersion
α ~ Gamma(2, 0.1)               // NB overdispersion

// No prior on ξ_t (integrated out via sequential construction)
```

#### Prior Rationale: Innovation Variance

The key parameter is **σ_ω**, which controls how much the process can wander:

- **Small σ_ω** (< 0.1): Nearly deterministic, RW ≈ linear trend
- **Large σ_ω** (> 0.5): Wild fluctuations, non-smooth
- **Exponential(10)** prior: E = 0.1, favors smooth but allows flexibility

After T steps, cumulative variance = T×σ_ω². For T=40:
- If σ_ω = 0.1, cumulative SD ≈ 0.63 (one log unit)
- If σ_ω = 0.2, cumulative SD ≈ 1.26 (factor of 3.5×)

### Temporal Correlation Details

#### Structure: Integrated Process

Random walk has **perfect memory**:
```
Var(ξ_t) = t × σ_ω²              (grows without bound!)
Corr(ξ_t, ξ_{t+k}) = √(t / (t+k)) → 1 as t → ∞
```

This creates **non-stationary** correlation:
- Early observations: Lower correlation
- Late observations: Higher correlation
- All observations: Extremely high correlation (explains ACF = 0.971)

#### Comparison to AR(1) and GP

| Property | AR(1) | GP | Random Walk |
|----------|-------|-----|-------------|
| Stationarity | Yes | Yes | **No** |
| Long-run variance | Finite | Finite | **Infinite** |
| Mean-reversion | Yes | Yes | **No** |
| ACF decay | Exponential | Gaussian | **None** |
| ρ = 1 limit | RW | - | Native |

**Key insight**: If observed ACF = 0.971 reflects near-unit-root behavior, RW is the correct model class.

### Prior Predictive Distribution

**Expected trajectories**:
- Start at β₀, drift with rate β₁
- Small random deviations accumulate
- Variance increases over time (heteroscedasticity)
- No mean reversion (if process wanders high, stays high)

**Prior predictive checks should show**:
- Diverse paths (some smooth, some wandering)
- All follow general upward trend (β₁ > 0)
- Occasional "jumps" or "plateaus" (cumulative innovations)
- Increasing uncertainty over time

### Falsification Criteria

**REJECT this model if**:

1. **Random walk inappropriate**:
   - First differences show **significant autocorrelation** (suggests AR not RW)
   - Posterior predictive trajectories too smooth (RW should have visible steps)
   - Cumulative innovations ξ_T are tiny relative to trend (RW adds nothing)

2. **Non-identifiable**:
   - Posterior for σ_ω overlaps strongly with zero
   - Cannot distinguish RW from deterministic trend
   - High posterior correlation between β₁ and σ_ω (> 0.9)

3. **Overfits noise**:
   - σ_ω posterior is large (> 0.5), suggesting fitting individual observations
   - Posterior predictive paths are not smooth
   - Poor out-of-sample performance despite good in-sample fit

4. **Model comparison**:
   - LOO strongly prefers AR(1) over RW (ΔELPD < -5)
   - RW gives unrealistic long-term predictions (wanders to impossible values)
   - Pareto-k diagnostics indicate instability

5. **Stationarity test**:
   - If we difference the data, differenced series shows no trend
   - This would indicate original series is I(1), supporting RW
   - If differenced series still shows trend, RW is wrong

### Stan Implementation Notes

#### Sequential Construction

Random walk is easiest to implement **sequentially**:

```stan
transformed parameters {
  vector[N] theta;  // State (log-mean)
  vector[N] mu;     // Expected count

  // Initial state
  theta[1] = beta0 + beta1 * year[1];

  // Random walk evolution
  for (t in 2:N) {
    theta[t] = beta0 + beta1 * year[t] +
               (theta[t-1] - beta0 - beta1 * year[t-1]) +  // Previous deviation
               sigma_omega * omega_raw[t];                  // New innovation
  }

  mu = exp(theta);
}
```

Where `omega_raw[t] ~ std_normal()` are independent.

#### Alternative: Non-Centered Parameterization

For better sampling, use cumulative sum:

```stan
transformed parameters {
  vector[N] xi;     // Cumulative innovations
  vector[N] theta;  // State

  xi = cumulative_sum(sigma_omega * omega_raw);  // Random walk component
  theta = beta0 + beta1 * year + xi;             // Total log-mean
  mu = exp(theta);
}
```

This **decorrelates** σ_ω from the state values, improving sampling.

#### Full Model Code Skeleton

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta0;
  real beta1;
  real<lower=0> alpha;         // NB dispersion
  real<lower=0> sigma_omega;   // RW innovation SD
  vector[N] omega_raw;          // Raw innovations
}

transformed parameters {
  vector[N] xi;     // Cumulative deviations (RW component)
  vector[N] theta;  // State (log-mean)
  vector[N] mu;     // Expected count

  // Construct random walk
  xi[1] = 0;  // Initial condition
  for (t in 2:N) {
    xi[t] = xi[t-1] + sigma_omega * omega_raw[t];
  }

  // Total log-mean = trend + RW
  theta = beta0 + beta1 * year + xi;
  mu = exp(theta);
}

model {
  // Priors
  beta0 ~ normal(log(109.4), 1);
  beta1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);
  sigma_omega ~ exponential(10);  // KEY PRIOR
  omega_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  // Posterior predictive
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], alpha);
    C_rep[t] = neg_binomial_2_rng(mu[t], alpha);
  }

  // Forecast (optional)
  // vector[H] theta_future;
  // for (h in 1:H) {
  //   theta_future[h] = theta[N] + beta1 * delta_year[h] +
  //                     sigma_omega * normal_rng(0, 1);
  // }
}
```

#### Computational Considerations

1. **Sequential construction**: Cannot parallelize, but N=40 is trivial
2. **Cumulative sum**: Use built-in `cumulative_sum()` for efficiency
3. **Numerical stability**: No matrix inversions, very stable

### Computational Cost

**Expected runtime**: 2-3 minutes (4 chains × 2000 iterations)

**Sampling challenges**:
- **LOW**: Model is simple, no matrix operations
- **MEDIUM**: May have posterior correlation between β₁ and σ_ω
- **LOW**: Non-centered parameterization should sample efficiently

This should be the **fastest** of the three models.

---

## Model Comparison Framework

### Quantitative Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Convergence | Rhat | < 1.01 |
| Sampling efficiency | ESS (bulk/tail) | > 400 |
| Predictive accuracy | PSIS-LOO ELPD | Higher is better |
| Model complexity | Effective parameters | Lower is better |
| Stability | Pareto-k | < 0.7 for all |

### Comparing Correlation Structures

**Posterior predictive autocorrelation**:
- Compute ACF from replicated datasets C_rep
- Compare to observed ACF = 0.971
- Model should match **both magnitude and decay pattern**

**In-sample vs out-of-sample**:
- Leave-last-10-out cross-validation
- Can model extrapolate beyond observed time?
- RW may struggle (unbounded variance), AR1/GP should be better

### Qualitative Assessment

**Smoothness of predictions**:
- AR(1): Should be smooth with small fluctuations
- GP: Should be very smooth (learns optimal smoothness)
- RW: Should show visible "steps" or innovations

**Uncertainty quantification**:
- All models: Wider intervals in regions of sparse data
- RW: Uncertainty should **grow** over time
- AR(1)/GP: Stationary uncertainty (bounded)

---

## Critical Decision Points

### Decision 1: Is temporal correlation real or artifact?

**Checkpoint**: After fitting all three models

**Evidence to evaluate**:
1. Posterior for ρ (AR1), lengthscale (GP), σ_ω (RW)
2. Fit a **quadratic trend + independent errors** baseline
3. Compare ELPD: Does correlation improve fit significantly?

**Decision rule**:
- If correlation models improve ELPD by < 2: **Trend artifact** → use simpler model
- If correlation models improve ELPD by > 5: **Genuine dependence** → keep correlation

### Decision 2: Which correlation structure?

**Checkpoint**: After model comparison

**Evidence**:
1. PSIS-LOO ELPD differences
2. Posterior predictive checks (especially ACF)
3. Computational stability

**Decision rule**:
- If ΔELPD < 2×SE: **Choose simplest** (likely AR1)
- If GP wins: Suggests **non-linear trend** more important than correlation
- If RW wins: Suggests **non-stationary** process

### Decision 3: Is this the right model class entirely?

**Checkpoint**: After posterior predictive checks

**Red flags that would trigger reconsideration**:
1. Systematic deviations in posterior predictive (model is wrong)
2. Cannot capture observed ACF pattern (wrong correlation form)
3. Predictions are unrealistic (violate domain knowledge)
4. All models have similar poor fit (missing important feature)

**Alternative model classes to consider**:
- **Hierarchical**: If there are unobserved groups/regimes
- **Structural break**: If changepoint at year = -0.21 is real
- **Different distribution**: If NB is inadequate (try Beta-Binomial, Conway-Maxwell)
- **Exogenous shocks**: If there are known intervention times

---

## Stress Tests

### Test 1: Prior Sensitivity

**Protocol**:
1. Fit each model with three prior specifications:
   - Skeptical: Weak correlation priors (ρ ~ Beta(5,5), ℓ ~ InvGamma(2,1))
   - Baseline: Priors specified above
   - Informed: Strong correlation priors (ρ ~ Beta(30,2), ℓ ~ InvGamma(10,10))

2. Compare posteriors: How much do they change?

**Rejection criterion**:
- If posterior changes dramatically with prior → **data is weak**, model is unidentified

### Test 2: Structural Break Sensitivity

**Protocol**:
1. Fit models on two subsets:
   - Early period: year < 0
   - Late period: year > 0

2. Check if correlation parameters are similar

**Rejection criterion**:
- If ρ or ℓ are vastly different across periods → **non-stationary correlation**, need more complex model

### Test 3: Forecast Performance

**Protocol**:
1. Hold out last 10 observations
2. Fit models on first 30
3. Compute predictive log-likelihood on held-out data

**Rejection criterion**:
- If all models perform poorly out-of-sample → **overfitting**, need stronger regularization

### Test 4: Residual Diagnostics

**Protocol**:
1. Compute posterior mean residuals: r_t = C_t - E[μ_t | data]
2. Check:
   - Residual ACF (should be < 0.2 at all lags)
   - Runs test (randomness)
   - Normality of Pearson residuals

**Rejection criterion**:
- If residuals show structure → **model inadequate**, need different specification

---

## Implementation Priorities

### Phase 1: Baseline (Day 1)
1. Fit **NB-AR1** (Model 1) - this is the standard approach
2. Run all diagnostics (convergence, PPCs, LOO)
3. Establish baseline performance

**Decision point**: If AR1 fails badly, proceed directly to alternatives

### Phase 2: Alternatives (Day 2)
1. Fit **NB-RW** (Model 3) - simpler than GP, tests non-stationarity
2. Compare to AR1: Is random walk better?

**Decision point**: If RW is clearly better, may skip GP (computational cost)

### Phase 3: Flexible (Day 3)
1. Fit **NB-GP** (Model 2) if time allows
2. Compare all three models
3. Sensitivity analyses

**Decision point**: Final model selection or acknowledgment that no single model is adequate

---

## Expected Outcomes and Contingencies

### Scenario A: AR(1) wins decisively
**Interpretation**: High ACF reflects genuine stochastic dependence with mean reversion

**Next steps**:
- Report posterior for ρ (likely 0.9-0.99)
- Discuss implications for uncertainty (SEs are wider than naive model)
- Consider AR(p) with p>1 if residual ACF persists

### Scenario B: GP wins decisively
**Interpretation**: High ACF is primarily a **smooth trend artifact**, not stochastic dependence

**Next steps**:
- Examine learned lengthscale (likely very large)
- Argue for flexible trend + independent errors
- Simpler than AR(1) in spirit (fewer assumptions)

### Scenario C: RW wins decisively
**Interpretation**: Process is genuinely **non-stationary**, ρ ≈ 1 (unit root)

**Next steps**:
- Acknowledge this complicates long-term forecasting
- Consider differencing or detrending
- May need **structural model** with domain knowledge

### Scenario D: All models fit poorly
**Interpretation**: Missing critical feature (changepoint, nonlinearity, external drivers)

**Next steps**:
- Revisit EDA for clues (structural break at year = -0.21?)
- Consider regime-switching models
- Acknowledge current models are inadequate

### Scenario E: Models fit well but are indistinguishable
**Interpretation**: Data is too limited (n=40) to distinguish correlation structures

**Next steps**:
- Report model-averaged predictions
- Emphasize uncertainty in correlation form
- Recommend AR(1) for parsimony but acknowledge alternatives

---

## Connection to Other Designer Proposals

This designer focuses on **temporal correlation**. Expected proposals from other designers:

- **Designer 1** (Hierarchical): May propose group-level effects, varying intercepts
- **Designer 3** (Measurement): May propose observation error models, zero-inflation checks

**Potential conflicts**:
- If Designer 1 proposes hierarchical structure, need to decide: group effects OR temporal correlation?
- If Designer 3 proposes ZIP, need to check: are zeros actually present? (EDA says no)

**Synergies**:
- Could combine: Hierarchical + AR(1) correlation (if groups are identified)
- Could test: Does temporal correlation explain what Designer 1 attributes to groups?

---

## Key References and Theoretical Background

### AR(1) for Count Time Series
- Zeger & Qaqish (1988): Markov regression models for time series
- Chan & Ledolter (1995): Monte Carlo EM for correlated Poisson/NB

### Gaussian Process Methods
- Rasmussen & Williams (2006): GP for Machine Learning (Chapter 5: Model Selection)
- Vanhatalo et al. (2013): GP regression with Poisson likelihood

### State-Space Models
- Durbin & Koopman (2012): Time Series Analysis by State Space Methods
- Held & Rue (2010): Latent Gaussian models for count data

### Model Comparison
- Vehtari et al. (2017): Practical Bayesian model evaluation using LOO-CV
- Gelman et al. (2020): Bayesian Workflow (prior/posterior predictive checks)

---

## Summary and Recommendations

### Primary Recommendation: NB-AR1 (Model 1)

**Rationale**:
- Standard approach in literature
- Balances complexity and interpretability
- Most likely to be identifiable with n=40
- Computationally tractable

**Risk**: May be inadequate if ρ ≈ 1 (should try RW) or if trend is complex (should try GP)

### Secondary Recommendation: NB-RW (Model 3)

**Rationale**:
- Simpler than GP (fewer parameters)
- Natural choice for near-unit-root (ACF = 0.971)
- Fast to fit, easy to interpret
- State-space framework is familiar

**Risk**: May overfit if true process is stationary AR(1)

### Tertiary Recommendation: NB-GP (Model 2)

**Rationale**:
- Most flexible, can adapt to data
- No parametric trend assumption
- Best if we're uncertain about functional form

**Risk**: Most complex, may be overparameterized for n=40, computationally expensive

### Critical Success Factors

1. **Identifiability**: Can we separate trend from correlation?
2. **Computational stability**: Can we sample reliably?
3. **Predictive validity**: Do models forecast well?

If ANY of these fail, we must **reconsider** the entire modeling approach.

---

**End of Document**

Files: `/workspace/experiments/designer_2/proposed_models.md`
