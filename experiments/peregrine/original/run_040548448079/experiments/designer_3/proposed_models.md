# Time-Series Bayesian Models for Count Data with Regime Change

**Designer**: Model Designer 3 (Time-Series & Dynamic Models Specialist)
**Date**: 2025-10-29
**Data Context**: 40 time-ordered count observations with extreme overdispersion (variance/mean=67.99), strong non-stationarity (ACF(1)=0.944), and confirmed structural break at observation 17

---

## Executive Summary

This document proposes three competing Bayesian model classes that treat temporal dynamics as PRIMARY features, not afterthoughts. Each model makes fundamentally different assumptions about the data generation process and includes explicit falsification criteria.

**Critical Insight**: The combination of (1) strong autocorrelation, (2) structural regime change, and (3) extreme overdispersion suggests the data may arise from a **latent dynamic process** rather than simple regression with correlated errors. The models below range from "autocorrelation as nuisance" to "dynamics as mechanism."

**Key Design Principle**: I will abandon any model that fails to capture BOTH autocorrelation AND regime change simultaneously. Models that fix one at the expense of the other are scientifically inadequate.

---

## Model 1: Dynamic Linear Model with Regime-Switching (State-Space Framework)

### Core Philosophy
Treat the log-rate as a **latent dynamic process** evolving through time, with an abrupt change in drift velocity at the structural break. Autocorrelation emerges naturally from state persistence, not as an ad-hoc error term.

### Mathematical Specification

**Observation Model** (Negative Binomial):
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = η_t
```

**State Evolution** (Dynamic Process with Regime Change):
```
η_t = η_{t-1} + δ_t + I(t > τ) × Δδ + ν_t
δ_t = φ × δ_{t-1} + ω_t

where:
  ν_t ~ Normal(0, σ_η²)    [observation-level noise]
  ω_t ~ Normal(0, σ_δ²)    [trend innovation noise]
  I(t > τ) = indicator for regime change
  Δδ = shift in drift rate post-changepoint
```

**State Interpretation**:
- `η_t`: Log-rate level at time t (latent state)
- `δ_t`: Growth velocity (AR(1) process allowing smooth acceleration/deceleration)
- `τ`: Changepoint location
- `Δδ`: Discrete jump in growth rate at changepoint
- `φ`: Velocity persistence (autocorrelation parameter, constrained to [0,1) for stationarity)

**Complete Parameter Set**:
- `η_0`: Initial log-rate level
- `δ_0`: Initial growth velocity
- `φ`: Velocity autocorrelation (0 ≤ φ < 1)
- `Δδ`: Regime change magnitude
- `τ`: Changepoint location (discrete, 10 ≤ τ ≤ 30)
- `σ_η`: Observation-level noise scale
- `σ_δ`: Trend innovation noise scale
- `α`: Negative Binomial dispersion (inverse overdispersion)

### Prior Recommendations (EDA-Informed)

```stan
// Initial conditions (at t=1, standardized year ≈ -1.67)
η_0 ~ normal(3.0, 0.5);          // log(20) ≈ 3.0 (observed starting range)
δ_0 ~ normal(0.3, 0.2);           // Initial gentle growth

// Regime change parameters
τ ~ discrete_uniform(10, 30);     // EDA suggests ~17, allow uncertainty
Δδ ~ normal(0.8, 0.3);            // Post-break acceleration (EDA: 730% increase)

// Autocorrelation (velocity persistence)
φ ~ beta(8, 2);                   // E[φ] = 0.8, strong but < 1

// Noise scales (hierarchical shrinkage)
σ_η ~ exponential(2);             // Expect small observation noise (counts follow trend)
σ_δ ~ exponential(5);             // Expect smooth velocity changes

// Dispersion
α ~ gamma(2, 1);                  // EDA: α ≈ 0.61, allow flexibility
```

**Prior Justification**:
- `η_0 ~ N(3.0, 0.5)`: Starting counts around 19-30, log(20-30) ≈ 3.0-3.4
- `δ_0 ~ N(0.3, 0.2)`: Pre-break growth moderate, not explosive
- `Δδ ~ N(0.8, 0.3)`: 730% increase in slope suggests Δδ ≈ 0.7-1.0 on log scale
- `φ ~ Beta(8,2)`: Strong autocorrelation but enforce stationarity (φ < 1)
- `τ ~ Discrete_Uniform(10,30)`: EDA says 17, but allow model to discover
- Exponential priors on noise: Weak shrinkage toward parsimony

### Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Velocity AR(1) is unnecessary**: If posterior φ concentrates near 0, it means δ_t is white noise and we don't need the velocity state. This suggests a simpler random walk would suffice.

2. **State-space adds no value over GLM**: If σ_η >> σ_δ (observation noise dominates trend innovation), the latent process is just adding complexity without capturing dynamics. Compare LOO to Model 2.

3. **Changepoint is diffuse/uncertain**: If posterior for τ spreads uniformly across [10,30], the "regime change" might be an artifact. The structural break might actually be a smooth acceleration (favor Model 3).

4. **Non-stationary velocity**: If φ posterior concentrates near 1.0, the velocity process is non-stationary (unit root), violating model assumptions. Need to reformulate as integrated process.

5. **Posterior predictive autocorrelation still strong**: If ACF of (C_t - μ_t)/sqrt(μ_t) residuals shows ACF(1) > 0.5, the model hasn't captured the temporal dependency despite its complexity.

6. **Prior-data conflict**: If prior predictive p-value < 0.01 or > 0.99 for key statistics (mean count, growth rate), priors are fighting the data.

**Stress Test**:
- Simulate from posterior predictive, compute ACF(1) and structural break test. If simulated data consistently shows weaker autocorrelation or no break, model is inadequate.

### Implementation Considerations

**Stan Implementation**:
```stan
data {
  int<lower=1> N;              // Number of observations
  array[N] int<lower=0> C;     // Count observations
  vector[N] year;              // Standardized time variable
}

parameters {
  real eta_0;                  // Initial log-rate
  real delta_0;                // Initial velocity
  vector[N] eta_raw;           // Non-centered parameterization for η_t
  vector[N] delta_raw;         // Non-centered parameterization for δ_t
  real<lower=0,upper=1> phi;   // Velocity persistence
  real Delta_delta;            // Regime shift magnitude
  int<lower=10,upper=30> tau;  // Changepoint location (discrete)
  real<lower=0> sigma_eta;     // Observation noise
  real<lower=0> sigma_delta;   // Velocity innovation noise
  real<lower=0> alpha;         // NB dispersion
}

transformed parameters {
  vector[N] eta;               // Log-rate trajectory
  vector[N] delta;             // Velocity trajectory
  vector[N] mu;                // Expected counts

  // Initialize
  eta[1] = eta_0 + eta_raw[1] * sigma_eta;
  delta[1] = delta_0 + delta_raw[1] * sigma_delta;

  // State evolution
  for (t in 2:N) {
    real regime_shift = (t > tau) ? Delta_delta : 0.0;
    delta[t] = phi * delta[t-1] + regime_shift + delta_raw[t] * sigma_delta;
    eta[t] = eta[t-1] + delta[t] + eta_raw[t] * sigma_eta;
  }

  mu = exp(eta);
}

model {
  // Priors
  eta_0 ~ normal(3.0, 0.5);
  delta_0 ~ normal(0.3, 0.2);
  phi ~ beta(8, 2);
  Delta_delta ~ normal(0.8, 0.3);
  tau ~ uniform(10, 30);  // Note: Stan requires custom discrete sampling
  sigma_eta ~ exponential(2);
  sigma_delta ~ exponential(5);
  alpha ~ gamma(2, 1);

  eta_raw ~ std_normal();
  delta_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, 1/alpha);
}
```

**Computational Challenges**:
- **Discrete changepoint**: Stan doesn't natively sample discrete parameters. Solutions:
  - Marginalize over τ (sum likelihood across plausible τ values)
  - Fix τ at EDA estimate (17) for initial runs
  - Use indicator trick with continuous relaxation
- **Non-centered parameterization**: Essential for efficient sampling (eta_raw, delta_raw)
- **State-space complexity**: Expect ~1000 divergences if priors/initialization poor

**Expected Runtime**: 15-30 minutes (4 chains, 2000 iterations)

**Diagnostics to Monitor**:
- R-hat < 1.01 for all parameters
- ESS > 400 for η, δ trajectories
- No divergences (if >50, priors too vague or model misspecified)
- Trace plots for φ, τ should show good mixing

---

## Model 2: Negative Binomial Autoregressive Model (Observation-Driven)

### Core Philosophy
Autocorrelation arises from **direct dependence** of current count on previous counts, not from latent states. This is a "conditional" or "observation-driven" time-series model. Regime change is modeled as a shift in the conditional mean structure.

### Mathematical Specification

**Observation Model**:
```
C_t ~ NegativeBinomial(μ_t, α_t)

log(μ_t) = β_0 + β_1 × year_t + β_2 × year_t² + γ × log(C_{t-1} + 1) + I(t > τ) × [β_3 × (year_t - year_τ)]

where:
  γ: Autoregressive coefficient (captures persistence)
  log(C_{t-1} + 1): Lagged log-count (+ 1 to handle zero counts if present)
  I(t > τ): Regime indicator
  β_3: Additional slope post-changepoint
```

**Time-Varying Dispersion** (EDA shows 6-fold variation):
```
log(α_t) = α_0 + α_1 × year_t

This allows dispersion to evolve with time/mean.
```

**Complete Parameter Set**:
- `β_0`: Baseline log-rate intercept
- `β_1`: Linear time trend (pre-break)
- `β_2`: Quadratic curvature
- `β_3`: Additional slope post-break
- `γ`: Autoregressive coefficient (0 < γ < 1 for stability)
- `τ`: Changepoint location
- `α_0`: Baseline log-dispersion
- `α_1`: Dispersion trend

### Prior Recommendations

```stan
// Baseline regression
β_0 ~ normal(4.0, 0.5);          // Log-count at year=0 (EDA: log(74.5) ≈ 4.3)
β_1 ~ normal(0.5, 0.3);          // Modest growth pre-break
β_2 ~ normal(0.2, 0.2);          // Slight acceleration (quadratic)
β_3 ~ normal(0.8, 0.4);          // Strong post-break acceleration

// Autoregressive component (KEY PARAMETER)
γ ~ beta(6, 2);                  // E[γ] = 0.75, strong dependence on past

// Changepoint
τ ~ discrete_uniform(10, 30);    // Or fix at 17 based on EDA

// Dispersion evolution
α_0 ~ normal(log(0.6), 0.5);     // EDA: α ≈ 0.61
α_1 ~ normal(0, 0.3);            // Mild time-variation allowed
```

**Prior Justification**:
- `γ ~ Beta(6,2)`: Strong prior for autocorrelation (ACF(1)=0.944 in data), but constrained < 1
- `β_3 ~ N(0.8, 0.4)`: Regime change is large (730% slope increase)
- Dispersion priors allow time-variation (EDA shows 6-fold change)

### Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Autoregressive term is weak**: If posterior for γ concentrates near 0, the lagged count adds no information. This means autocorrelation is spurious (driven by omitted smooth trend), not genuine temporal dependence. Switch to Model 3.

2. **Residual autocorrelation persists**: If Pearson residuals (C_t - μ_t)/sqrt(μ_t + α_t μ_t²) still show ACF(1) > 0.4, the AR(1) term hasn't captured the dependency structure. May need AR(2) or ARMA.

3. **First-observation problem**: If posterior uncertainty for C_1 prediction is vastly higher than other observations (no lagged value), the model is unstable at boundaries. This suggests the AR structure is artificial.

4. **Changepoint unnecessary**: If β_3 posterior includes 0 in 90% CI, the structural break may be smooth (favor Model 3).

5. **Time-varying dispersion is flat**: If α_1 posterior concentrates near 0, we're over-parameterizing. Use constant α.

6. **Posterior predictive checks fail**: Generate C_t^{rep} from posterior, compute growth rates. If simulated growth is smoother than observed (no regime shift), model is smoothing over the break.

**Stress Test**:
- Remove lagged term (γ = 0), refit. If ΔLOO < 2, the AR term is not justified.
- Fit on first 30 observations, predict last 10. If predictive intervals exclude observed counts, model extrapolation fails.

### Implementation Considerations

**Stan Implementation**:
```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
}

transformed data {
  vector[N] log_C_lag;
  log_C_lag[1] = log(C[1] + 1);  // Initialize lag with first observation
  for (t in 2:N) {
    log_C_lag[t] = log(C[t-1] + 1);
  }
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real beta_3;
  real<lower=0,upper=1> gamma;   // AR coefficient (constrained)
  int<lower=10,upper=30> tau;
  real alpha_0;
  real alpha_1;
}

transformed parameters {
  vector[N] log_mu;
  vector[N] alpha;
  vector[N] mu;

  for (t in 1:N) {
    real regime_term = (t > tau) ? beta_3 * (year[t] - year[tau]) : 0.0;
    log_mu[t] = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2 + gamma * log_C_lag[t] + regime_term;
    alpha[t] = exp(alpha_0 + alpha_1 * year[t]);
  }

  mu = exp(log_mu);
}

model {
  // Priors
  beta_0 ~ normal(4.0, 0.5);
  beta_1 ~ normal(0.5, 0.3);
  beta_2 ~ normal(0.2, 0.2);
  beta_3 ~ normal(0.8, 0.4);
  gamma ~ beta(6, 2);
  tau ~ uniform(10, 30);
  alpha_0 ~ normal(log(0.6), 0.5);
  alpha_1 ~ normal(0, 0.3);

  // Likelihood
  for (t in 1:N) {
    C[t] ~ neg_binomial_2(mu[t], 1/alpha[t]);
  }
}
```

**Computational Challenges**:
- **Lagged observation feedback**: Creates dependencies that slow HMC. Use step-size adaptation.
- **Boundary condition**: First observation has no lag. Sensitivity analysis needed.
- **Identifiability**: γ and β_2 may be confounded (both capture smooth persistence). Monitor correlation in posterior.

**Expected Runtime**: 10-20 minutes (4 chains, 2000 iterations)

**Diagnostics**:
- Check posterior correlation between γ and β_2 (if |corr| > 0.7, identifiability issue)
- Posterior predictive: simulate C_t^{rep} conditioning on observed C_{t-1}, check autocorrelation structure
- LOO: watch for Pareto-k > 0.7 (influential observations)

---

## Model 3: Gaussian Process with Negative Binomial Likelihood (Parameter-Driven)

### Core Philosophy
Autocorrelation and regime change are both manifestations of a **smooth underlying process** that we don't fully understand. Use a Gaussian Process to flexibly model the log-rate trajectory, capturing both gradual trends and abrupt shifts without imposing rigid structure. This is the "I don't know the mechanism" model.

### Mathematical Specification

**Observation Model**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = f(year_t) + β_0

where:
  f(·): Gaussian Process (latent smooth function)
  β_0: Global intercept
```

**Gaussian Process Prior**:
```
f ~ GP(0, K)

Kernel: K(year_i, year_j) = σ_f² × exp(-ρ × (year_i - year_j)²) + σ_n² × δ_{ij}

where:
  σ_f²: Signal variance (magnitude of deviations)
  ρ: Inverse squared length-scale (1/l², controls smoothness)
  σ_n²: Nugget (observation-level noise)
  δ_{ij}: Kronecker delta (1 if i=j, else 0)
```

**Alternative Kernel (Locally Periodic for potential cycles)**:
```
K(year_i, year_j) = σ_f² × exp(-ρ × (year_i - year_j)²) × [1 + sin²(π × (year_i - year_j) / p) / l_p²]

where:
  p: Pseudo-period (if cycles detected in residuals)
  l_p: Periodic length-scale
```

**Complete Parameter Set**:
- `β_0`: Global intercept
- `σ_f`: GP signal standard deviation
- `l`: Length-scale (smoothness parameter)
- `σ_n`: Nugget (micro-variation)
- `α`: NB dispersion
- `f = [f(year_1), ..., f(year_N)]`: Latent GP trajectory (N-dimensional)

### Prior Recommendations

```stan
// Global level
β_0 ~ normal(4.3, 0.5);          // EDA: log(C) at year≈0 is ~4.3

// GP hyperparameters
σ_f ~ exponential(1);            // Moderate signal strength (expect log-rate varies by ~1-2 units)
l ~ inv_gamma(5, 5);             // Length-scale: E[l] = 1.25 (moderate smoothness over time range)
σ_n ~ exponential(5);            // Small nugget (GP should capture most variation)

// Dispersion
α ~ gamma(2, 1);                 // EDA: α ≈ 0.61
```

**Prior Justification**:
- `σ_f ~ Exp(1)`: Log-rates range from ~3 to ~5.6 (diff ≈ 2.6), so σ_f ≈ 1-2 reasonable
- `l ~ InvGamma(5,5)`: Length-scale ~1 means correlation drops to ~0.37 at distance 1.0 (reasonable for 40 observations spanning 3.3 standardized units)
- `σ_n ~ Exp(5)`: Weak nugget, GP should explain trajectory not random noise

**Alternative: Hilbert Space GP** (for computational efficiency):
Use basis function approximation if full GP is too slow (N=40 is borderable).

### Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Length-scale collapses**: If posterior for l concentrates near 0, the GP is overfitting to noise (interpolating point-to-point). This means the smoothness assumption is wrong; data has genuine discontinuities (favor Model 1 with explicit changepoint).

2. **Length-scale explodes**: If l posterior has mass at l > 5, the GP degenerates to a linear trend. The flexibility of GP is unused; a simple polynomial would suffice (and be faster/more interpretable).

3. **Nugget dominates**: If σ_n / σ_f > 0.5 (posterior ratio), observation noise overwhelms signal. GP is just adding white noise to a mean trend; not capturing autocorrelation.

4. **Posterior predictive autocorrelation weak**: If simulated data from posterior predictive has ACF(1) < 0.7 (vs observed 0.944), the GP isn't capturing the strong persistence. This is a critical failure for a "flexible" model.

5. **No structural break in GP trajectory**: Plot posterior mean f(year) with 90% credible bands. If the trajectory is smooth without a visible acceleration around year ≈ -0.2, the model has missed the regime change. GP may be over-smoothing.

6. **Computational failure**: If sampling takes > 1 hour or has > 20% divergences, the GP geometry is pathological. Need to switch to approximate methods (Hilbert space, variational inference) or abandon GP.

**Stress Test**:
- Posterior predictive: Generate C_t^{rep}, apply Chow test at observation 17. If p-value > 0.1 in most draws, GP has smoothed away the break.
- Leave-future-out: Train on t=1:30, predict t=31:40. If 90% predictive intervals exclude observed counts, GP doesn't extrapolate well (favor parametric Models 1-2).

### Implementation Considerations

**Stan Implementation** (using built-in GP):
```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
}

parameters {
  real beta_0;
  real<lower=0> sigma_f;
  real<lower=0> length_scale;
  real<lower=0> sigma_n;
  vector[N] f_raw;               // Non-centered GP
  real<lower=0> alpha;
}

transformed parameters {
  vector[N] f;
  vector[N] log_mu;
  vector[N] mu;

  {
    matrix[N, N] K;
    matrix[N, N] L_K;

    // Compute covariance matrix
    K = gp_exp_quad_cov(year, sigma_f, length_scale);
    for (n in 1:N) {
      K[n, n] = K[n, n] + sigma_n^2;  // Add nugget
    }

    // Cholesky decomposition
    L_K = cholesky_decompose(K);

    // Non-centered parameterization
    f = L_K * f_raw;
  }

  log_mu = beta_0 + f;
  mu = exp(log_mu);
}

model {
  // Priors
  beta_0 ~ normal(4.3, 0.5);
  sigma_f ~ exponential(1);
  length_scale ~ inv_gamma(5, 5);
  sigma_n ~ exponential(5);
  alpha ~ gamma(2, 1);

  f_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, 1/alpha);
}

generated quantities {
  array[N] int C_rep;              // Posterior predictive samples
  vector[N] log_lik;               // Pointwise log-likelihood for LOO

  for (t in 1:N) {
    C_rep[t] = neg_binomial_2_rng(mu[t], 1/alpha);
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], 1/alpha);
  }
}
```

**Computational Challenges**:
- **Cholesky decomposition**: O(N³) cost, but N=40 is manageable
- **Non-centered parameterization**: Essential (f_raw ~ N(0,1), then f = L_K * f_raw)
- **Hyperparameter exploration**: Length-scale posteriors can be multimodal; use high adapt_delta (0.95+)
- **Divergences**: GP geometry is tricky. If persistent divergences, use Hilbert space approximation or increase target acceptance to 0.99

**Expected Runtime**: 20-40 minutes (4 chains, 2000 iterations) - slowest of the three models

**Diagnostics**:
- Plot posterior draws of f(year) trajectory (should show clear regime change)
- Check length-scale posterior (if bimodal, increase warmup iterations)
- Compute posterior predictive ACF(1) - should match observed ~0.94
- LOO: GP models sometimes have large Pareto-k values (watch for k > 0.7)

---

## Comparative Model Evaluation Strategy

### Phase 1: Individual Model Diagnostics (Parallel Fitting)

Fit all three models simultaneously and check:

1. **Computational Health**:
   - All R-hat < 1.01?
   - ESS > 400 for key parameters?
   - Divergences < 1% of samples?
   - Sampling time < 1 hour?

2. **Prior-Data Compatibility**:
   - Prior predictive checks: Do prior draws cover observed data range?
   - Prior predictive p-values: 0.01 < p < 0.99 for key statistics?

3. **Posterior Plausibility**:
   - Do parameter posteriors make scientific sense?
   - Are 90% credible intervals reasonable?

**Decision Point**: If any model fails Phase 1 diagnostics (divergences > 5%, R-hat > 1.05), investigate before proceeding. Model may need reparameterization or prior adjustment.

### Phase 2: Falsification Testing

For each model, apply its specific falsification criteria (listed above). Key tests:

1. **Autocorrelation Capture**:
   - Compute Pearson residuals: r_t = (C_t - μ_t) / sqrt(V(μ_t))
   - Calculate ACF(r_t) lag 1-5
   - **FAIL** if ACF(1) > 0.4 (model hasn't absorbed temporal dependency)

2. **Structural Break Detection**:
   - Posterior predictive: Generate 1000 datasets from posterior
   - Apply Chow test at t=17 to each dataset
   - **FAIL** if < 80% of datasets show significant break (p < 0.05)

3. **Overdispersion Adequacy**:
   - Compute posterior predictive variance-to-mean ratio for each t
   - **FAIL** if observed ratio (67.99) is outside 90% posterior predictive interval

4. **Out-of-Sample Prediction** (Time-Series CV):
   - Train on t=1:30, predict t=31:40 (future)
   - Train on t=1:20, predict t=21:40 (longer horizon)
   - **FAIL** if > 30% of held-out observations are outside 90% predictive interval

**Decision Point**: Any model with 2+ failures is falsified. Do not proceed with that model.

### Phase 3: Model Comparison (Among Survivors)

Use multiple metrics (no single metric is sufficient):

1. **Leave-One-Out Cross-Validation (LOO)**:
   - Compute ELPD_loo and SE for each model
   - Check Pareto-k diagnostics (if k > 0.7 for >3 observations, LOO unreliable)
   - **Prefer** model with highest ELPD_loo if difference > 2*SE

2. **WAIC** (Watanabe-Akaike Information Criterion):
   - Cross-check with LOO (should agree within 10%)
   - If LOO and WAIC strongly disagree, investigate influential observations

3. **Posterior Predictive P-Values**:
   - For 10 key statistics (mean, variance, ACF(1), max count, growth rate, etc.)
   - Compute p-value = P(T(C_rep) ≥ T(C_obs) | data)
   - **Good fit**: 0.05 < p < 0.95 for most statistics
   - **Poor fit**: p < 0.01 or p > 0.99 for multiple statistics

4. **Domain Plausibility**:
   - Which model's parameters are most interpretable?
   - Which model's assumptions align with subject-matter knowledge?
   - Which model would you trust for policy decisions?

**Decision Point**: If no clear winner (ΔLOO < 2*SE), use model stacking/averaging. If one model dominates, use it for inference.

### Phase 4: Sensitivity Analysis (Winning Model)

1. **Prior Sensitivity**:
   - Refit with wider priors (2x standard deviation)
   - Refit with informative priors (0.5x standard deviation)
   - **FAIL** if conclusions change substantially

2. **Changepoint Sensitivity** (Models 1-2):
   - Fix τ at 15, 17, 19
   - Do posterior inferences about growth rate change?
   - **FAIL** if estimated Δδ or β_3 flips sign across τ values

3. **Subsampling**:
   - Remove 10% of observations (stratified by time)
   - Does structural break still appear?
   - **FAIL** if break disappears (suggests overfitting)

---

## Red Flags Requiring Model Class Pivot

### Catastrophic Failures (Abandon All Three Models)

**STOP AND RECONSIDER EVERYTHING IF**:

1. **All models fail autocorrelation test**: If none of the three models reduce residual ACF(1) below 0.5, the temporal dependency is more complex than AR(1)/GP/DLM can capture. Possible causes:
   - Seasonality (check for periodic patterns we missed)
   - Long memory (fractional integration, ARFIMA models)
   - Regime-dependent autocorrelation (Markov-switching variance)
   - **Pivot to**: ARMA(p,q) models, Hidden Markov Models, or nonlinear state-space

2. **Changepoint is illusory**: If all three models' posterior predictive checks fail the Chow test (< 60% of simulated datasets show break), the "structural break" may be a smooth acceleration, outlier cluster, or measurement artifact. Possible causes:
   - GP is correct (smooth), DLM/AR are forcing artificial break
   - Outliers creating false impression of regime change
   - **Pivot to**: Pure GP without changepoint, or investigate data quality issues

3. **Negative Binomial is inadequate**: If all models show posterior predictive variance-to-mean ratio < 40 (vs observed 67.99), NB distribution is insufficient. Possible causes:
   - Zero-inflation (but EDA shows no zeros)
   - Fat tails beyond NB (use generalized Pareto or mixture)
   - Heterogeneous dispersion beyond time-trend
   - **Pivot to**: Conway-Maxwell-Poisson, Beta-Binomial, or mixture models

4. **Computational failure across all models**: If all three models have > 10% divergences despite tuning, the problem geometry is pathological. Possible causes:
   - Funnel geometry (reparameterize all models)
   - Multimodal posterior (data supports multiple explanations equally)
   - Unidentifiability (parameters trade off perfectly)
   - **Pivot to**: Variational inference, integrated nested Laplace approximation (INLA), or simplify to GLM baselines

5. **Out-of-sample prediction catastrophe**: If all models' 90% predictive intervals exclude > 50% of held-out observations, we're fundamentally misunderstanding the data generation process. Possible causes:
   - Exogenous shocks not in the model (covariates needed)
   - Measurement error in time variable
   - Non-exchangeable observations (spatial structure?)
   - **Pivot to**: Covariate discovery, measurement error models, or abandon time-series framework

### Warning Signs (Proceed with Caution)

1. **Posterior is prior**: If key parameter posteriors (γ, φ, length-scale) are nearly identical to priors, data is not informative. May need more data or different model.

2. **Extreme parameter values**: If α < 0.01 (extreme overdispersion) or α > 10 (underdispersion), question whether counts are real or transformed/aggregated measurements.

3. **Residual patterns**: If residual plots show clear trends, heteroscedasticity, or clusters, model is missing structure.

---

## Model Ranking (Prior to Seeing Data)

Based on EDA evidence and theoretical considerations:

### Primary Recommendation: **Model 1 (Dynamic Linear Model)**

**Rationale**:
- Structural break is definitive (4 independent tests confirm)
- Strong autocorrelation (0.944) suggests genuine dynamic process
- DLM naturally combines state persistence + regime shift
- Separates trend dynamics (δ_t) from observation noise (ν_t)
- Most aligned with "data generation process" thinking

**Risks**:
- Discrete changepoint may be too rigid (but EDA strongly supports it)
- Computational complexity (state-space is slowest to sample)
- Requires careful initialization (2 state variables)

### Secondary Recommendation: **Model 3 (Gaussian Process)**

**Rationale**:
- Maximum flexibility (no parametric assumptions on trend shape)
- Autocorrelation emerges naturally from covariance structure
- Can capture smooth regime change without imposing discrete break
- Good for exploratory analysis when mechanism is unknown

**Risks**:
- May over-smooth the structural break (falsification test critical)
- Computationally expensive (Cholesky decomposition)
- Hyperparameters (length-scale) can be difficult to tune
- Less interpretable than DLM/AR models

### Tertiary Recommendation: **Model 2 (NB-AR)**

**Rationale**:
- Simplest of the three (standard GLM + lagged term)
- Fast to fit and easy to diagnose
- Good baseline for comparison

**Risks**:
- Lagged observation feedback may not capture autocorrelation adequately (ACF induced by omitted smooth trend, not genuine AR)
- First-observation boundary condition is artificial
- May conflate quadratic trend with AR term (identifiability)
- Less theoretically motivated than DLM for this problem

---

## Implementation Roadmap

### Week 1: Baseline Fitting and Diagnostics

**Day 1-2**:
- Implement all three models in Stan
- Run prior predictive checks
- Verify models compile and sample (even if priors are rough)

**Day 3-4**:
- Fit Model 1 (DLM) with fixed τ=17 (simplify discrete parameter)
- Comprehensive diagnostics (R-hat, ESS, divergences, trace plots)
- Posterior predictive checks

**Day 5**:
- Fit Model 2 (NB-AR)
- Fit Model 3 (GP)
- Preliminary comparison

### Week 2: Falsification and Refinement

**Day 1-3**:
- Apply falsification criteria to all models
- Identify which models survive
- If multiple survivors, proceed to LOO comparison

**Day 4-5**:
- Sensitivity analysis on leading model(s)
- Refinements (priors, parameterization)
- Final posterior inference

### Week 3: Validation and Reporting

**Day 1-2**:
- Out-of-sample prediction (time-series CV)
- Posterior predictive visualizations
- Parameter interpretation

**Day 3-5**:
- Write-up: model justification, diagnostics, inferences
- Create reproducible analysis pipeline
- Document decisions and pivots

---

## Alternative Approaches (If Primary Models Fail)

### If All Three Models Fail Autocorrelation Test:

**Option A**: ARMA(p,q) with Negative Binomial
- Extend Model 2 to include MA terms
- Test orders p=1:3, q=1:2
- Use AIC/BIC for order selection

**Option B**: Generalized Additive Model (GAM) with AR errors
- Use splines for smooth trend
- Add AR(1) correlation structure
- Implement in PyMC with Bambi

**Option C**: Hidden Markov Model
- Two hidden states (pre/post regime)
- State-dependent NB distributions
- Transition matrix estimated from data

### If Structural Break is Illusory:

**Option A**: Pure GP (Model 3 without changepoint focus)
- Longer length-scale
- Focus on smooth trend

**Option B**: Polynomial regression (cubic or quartic)
- Simple NB-GLM with polynomial terms
- No explicit temporal correlation (compare LOO to AR models)

### If Negative Binomial is Inadequate:

**Option A**: Beta-Binomial (if counts are bounded)
- Check if counts have implicit upper limit
- More flexible variance function

**Option B**: Generalized Poisson or COM-Poisson
- Allows under/over-dispersion
- More complex likelihood

**Option C**: Mixture Models
- Two-component NB mixture
- Captures heterogeneity across time

---

## Success Criteria (How We Know We're Done)

A model is "successful" if it passes ALL of the following:

1. **Computational Health**: R-hat < 1.01, ESS > 400, divergences < 1%
2. **Falsification Survival**: Passes all model-specific falsification tests
3. **Autocorrelation**: Residual ACF(1) < 0.3
4. **Structural Break**: Posterior predictive Chow test > 80% significant
5. **Overdispersion**: Observed variance/mean in 90% posterior predictive interval
6. **Out-of-Sample**: < 20% of held-out observations outside 90% PI
7. **Posterior Predictive P-Values**: 0.05 < p < 0.95 for 8/10 test statistics
8. **Scientific Plausibility**: Parameters interpretable and reasonable
9. **LOO Reliable**: All Pareto-k < 0.7
10. **Prior Sensitivity**: Conclusions robust to 2x wider/narrower priors

**If no model passes all criteria**: Document failures, propose next iteration with alternative model classes.

**If multiple models pass**: Use LOO for ranking, consider model averaging if ΔLOO < 2*SE.

---

## Final Philosophical Notes

### On Falsification

The goal is not to "make a model work" but to **discover which model (if any) is consistent with the data**. Each model embodies a different scientific hypothesis about the data generation process:

- **Model 1 (DLM)**: "Counts arise from a smooth latent process with persistent velocity that abruptly changes direction"
- **Model 2 (NB-AR)**: "Today's count directly depends on yesterday's count, with a time-varying mean structure"
- **Model 3 (GP)**: "Counts follow a smooth unknown function of time, and I'm agnostic about the mechanism"

If all three fail, we learn that **none of these mechanisms explain the data**. That is scientific progress.

### On Model Complexity

Complexity is justified when it improves prediction or understanding, not for its own sake. If Model 2 (simpler) and Model 1 (complex) have equal LOO, prefer Model 2. But if Model 1 has interpretable latent states that reveal scientific insight, prefer Model 1 even if LOO is slightly worse.

### On Changepoints

The EDA provides overwhelming evidence for a structural break. However, "overwhelming evidence" can be wrong if:
- The break is an artifact of a few outliers
- The apparent break is actually smooth acceleration that looks discrete at low resolution
- The time variable is mismeasured (e.g., observations are not equally spaced)

Thus, Model 3 (GP) serves as a critical check: if the GP naturally discovers a discontinuity without being told, the break is real.

### On Autocorrelation

ACF(1) = 0.944 is extremely high for a time series of length 40. This could mean:
- Genuine AR(1) process (Model 2)
- Smooth underlying trend creating spurious correlation (Model 3)
- Dynamic latent process (Model 1)
- **Or**: The data is differenced incorrectly, measurements are correlated by design, or time variable is wrong

The falsification tests will reveal which explanation is correct.

---

## Summary Table

| Feature | Model 1 (DLM) | Model 2 (NB-AR) | Model 3 (GP) |
|---------|---------------|-----------------|--------------|
| **Core Mechanism** | Latent dynamic states | Observation-driven AR | Smooth unknown function |
| **Autocorrelation Source** | State persistence (φ) | Lagged count (γ) | Covariance structure (l) |
| **Regime Change** | Discrete shift (Δδ) | Piecewise linear (β_3) | Emergent from smoothness |
| **Parameters** | 8 + 2N states | 8 | 4 + N (GP trajectory) |
| **Complexity** | High | Medium | High |
| **Runtime** | 15-30 min | 10-20 min | 20-40 min |
| **Interpretability** | High (velocity, drift) | Medium (AR coeff) | Low (black box) |
| **Extrapolation** | Good (parametric) | Medium (depends on last obs) | Poor (uncertainty explodes) |
| **Prior Sensitivity** | Medium | Low | High (length-scale) |
| **Falsifiable?** | Yes (check φ, τ, ACF) | Yes (check γ, residuals) | Yes (check l, ACF, break) |

---

**Key Recommendation**: Start with Model 1 (DLM) as it most directly models the dual phenomena of autocorrelation + regime change. Use Model 3 (GP) as a sensitivity check on the discrete changepoint assumption. Use Model 2 (NB-AR) as a computational baseline.

**Critical Reminder**: Be prepared to abandon all three models if they fail falsification tests. Success is learning the truth, not completing the plan.

---

**File**: `/workspace/experiments/designer_3/proposed_models.md`
**Author**: Model Designer 3 (Time-Series Specialist)
**Contact**: Bayesian Modeling Strategist Team
