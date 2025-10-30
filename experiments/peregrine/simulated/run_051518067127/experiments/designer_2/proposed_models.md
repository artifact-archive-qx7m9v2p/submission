# Bayesian Model Design: Transformed Continuous Approaches with Temporal Structure

**Designer**: Model Designer 2
**Date**: 2025-10-30
**Focus**: Log-transformation approaches, temporal dependence, adaptive strategies
**Output Directory**: `/workspace/experiments/designer_2/`

---

## Executive Summary

This document proposes **three competing Bayesian model classes** for count time series data exhibiting exponential growth (2.37x/year), severe overdispersion (variance/mean = 70.43), and strong autocorrelation (ACF lag-1 = 0.971). All models leverage the excellent log-scale fit (R² = 0.937) but differ fundamentally in how they handle temporal structure.

**Critical insight**: The combination of regime shifts (7.8x increase), heterogeneous dispersion across periods, and residual autocorrelation after detrending suggests **the data generation process itself may be changing over time**. Any static model may fail.

### Model Prioritization (by ability to handle autocorrelation and regime shifts)

1. **AR(1) Log-Normal with Regime-Switching** (Most flexible for temporal structure)
2. **Gaussian Process on Log-Scale** (Discovers arbitrary temporal patterns)
3. **Hierarchical Period Model with AR(1)** (Explicitly models regime shifts)

---

## Problem Formulation: Competing Hypotheses

### Hypothesis 1: Smooth Nonstationary Process
- **Claim**: Growth rate changes smoothly over time
- **Model**: Gaussian Process with exponential growth mean function
- **Falsified by**: Abrupt changes, discontinuities in residuals, structural breaks
- **Why it might fail**: Period-specific dispersion ratios (0.68, 13.11, 7.23) suggest discrete regime changes, not smooth transitions

### Hypothesis 2: Fixed-Parameter Growth with Temporal Correlation
- **Claim**: Single exponential process with autoregressive errors
- **Model**: AR(1) Log-Normal with quadratic trend
- **Falsified by**: Non-constant variance structure over time, poor out-of-sample predictions in different regimes
- **Why it might fail**: Early period shows underdispersion (ratio=0.68), late period overdispersion (ratio=7.23)—suggests parameter heterogeneity

### Hypothesis 3: Regime-Dependent Process
- **Claim**: Different data-generating mechanisms across time periods
- **Model**: Hierarchical model with period-specific parameters
- **Falsified by**: No improvement over pooled model, continuous changepoint locations (not discrete)
- **Why it might fail**: If regime boundaries are not at tertile splits or if transitions are gradual

---

## Model 1: AR(1) Log-Normal with Regime-Switching (PRIMARY RECOMMENDATION)

### 1.1 Model Specification

```
# Likelihood
C[t] ~ LogNormal(mu[t], sigma_regime[regime[t]])

# Mean structure on log-scale
mu[t] = alpha + beta_1 * year[t] + beta_2 * year[t]^2 + phi * epsilon[t-1]

# Autoregressive error structure
epsilon[t] = log(C[t]) - (alpha + beta_1 * year[t] + beta_2 * year[t]^2)
epsilon[1] ~ Normal(0, sigma_regime[regime[1]] / sqrt(1 - phi^2))  # Stationary initialization

# Regime indicator (known from EDA: early, middle, late)
regime[1:14] = 1  (early)
regime[15:27] = 2  (middle)
regime[28:40] = 3  (late)

# Parameters
alpha ~ Normal(4.3, 0.5)  # Log-scale intercept (log(109) ≈ 4.7, but centered)
beta_1 ~ Normal(0.86, 0.2)  # Log-linear growth (EDA: 0.862)
beta_2 ~ Normal(0, 0.3)  # Quadratic term (could be positive or negative)
phi ~ Uniform(-0.95, 0.95)  # AR(1) coefficient (must be stationary)
sigma_regime[1:3] ~ HalfNormal(0, 1)  # Regime-specific residual SD
```

### 1.2 Prior Justification

**alpha (intercept)**:
- EDA: log(mean(C)) = log(109) ≈ 4.69
- But centered time means intercept is at year=0, where mean ~ 75, so log(75) ≈ 4.32
- Normal(4.3, 0.5): 95% CI = [3.3, 5.3] → exp range = [27, 200], covers data reasonably
- **Prior predictive**: Mean counts between 27-200 at center year

**beta_1 (linear growth)**:
- EDA: β₁ = 0.862 on log-scale → 2.37x per year
- Normal(0.86, 0.2): 95% CI = [0.46, 1.26] → multiplicative effects [1.58x, 3.53x]
- **Prior predictive**: Growth rates from 58% to 253% per standardized year—very wide but plausible for exponential processes

**beta_2 (quadratic)**:
- EDA: Quadratic improves R² by 0.083, suggesting acceleration/deceleration
- Normal(0, 0.3): 95% CI = [-0.59, 0.59]
- Allows for both accelerating (β₂ > 0) and decelerating (β₂ < 0) growth
- **Prior predictive**: Curvature can change growth rate by factor of 0.55x to 1.80x across year range

**phi (AR coefficient)**:
- Uniform(-0.95, 0.95): Weakly informative, enforces stationarity
- Symmetric around zero (allows negative autocorrelation, though unlikely)
- **Why not centered at 0.75?** Let the data speak—avoid anchoring to post-detrending ACF which may be model-dependent

**sigma_regime (regime-specific variability)**:
- HalfNormal(0, 1): Weakly informative, ensures positivity
- Allows data to determine relative variability across regimes
- **Prior predictive**: ~68% of observations within exp(±1σ) of predicted mean

### 1.3 Why Log-Normal Works Here

1. **No zeros**: Minimum count = 21, so log-transform always defined
2. **Excellent log-scale fit**: R² = 0.937, residuals more symmetric
3. **Multiplicative growth**: Log-normal naturally encodes exponential processes
4. **Back-transformation**: E[C|t] = exp(mu[t] + sigma²/2) handles bias correction
5. **Probabilistic predictions**: Can sample from posterior predictive directly

### 1.4 Handling Temporal Dependencies

**AR(1) structure**:
- Captures lag-1 correlation (EDA: ACF[1] = 0.971 raw, 0.754 after detrending)
- Residual ACF after detrending suggests AR(1) insufficient, but good starting point
- **If AR(1) fails**: Extend to AR(2) or AR(3) based on residual ACF

**Initialization**:
- First error term drawn from stationary distribution: N(0, σ/√(1-φ²))
- Ensures proper uncertainty propagation from t=1

**Computational note**:
- AR(1) creates sequential dependence → no longer independent observations
- Stan handles this naturally via forward recursion in likelihood

### 1.5 Falsification Criteria (I WILL ABANDON THIS IF...)

1. **Residual autocorrelation persists**: If ACF of standardized residuals still > 0.3 at lag 1
   - **Action**: Extend to higher-order AR or consider GP

2. **Regime variances are equal**: If posterior 95% CIs for all three σ_regime overlap substantially
   - **Action**: Abandon regime structure, use pooled variance

3. **AR coefficient near zero**: If posterior 95% CI for phi includes zero and centered near zero
   - **Action**: Drop AR structure, use independent errors

4. **Prior-posterior conflict**: If posterior pushes strongly against prior boundaries
   - **Action**: Revise priors or reconsider model structure

5. **Poor predictive performance**: If LOO-CV shows consistent underprediction in late regime
   - **Action**: Consider non-exponential growth (e.g., logistic, Gompertz)

6. **Non-constant AR parameter**: If residuals show time-varying autocorrelation
   - **Action**: Switch to time-varying AR(1) or GP model

### 1.6 Implementation Plan (Stan/CmdStanPy)

```stan
data {
  int<lower=1> N;
  vector[N] year;
  vector[N] log_C;
  int<lower=1, upper=3> regime[N];
}

parameters {
  real alpha;
  real beta_1;
  real beta_2;
  real<lower=-0.95, upper=0.95> phi;
  vector<lower=0>[3] sigma_regime;
}

transformed parameters {
  vector[N] mu;
  vector[N] epsilon;

  mu = alpha + beta_1 * year + beta_2 * square(year);
  epsilon = log_C - mu;
}

model {
  // Priors
  alpha ~ normal(4.3, 0.5);
  beta_1 ~ normal(0.86, 0.2);
  beta_2 ~ normal(0, 0.3);
  phi ~ uniform(-0.95, 0.95);
  sigma_regime ~ normal(0, 1);

  // Likelihood with AR(1) errors
  epsilon[1] ~ normal(0, sigma_regime[regime[1]] / sqrt(1 - phi^2));
  for (t in 2:N) {
    epsilon[t] ~ normal(phi * epsilon[t-1], sigma_regime[regime[t]]);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] C_pred;

  // Log-likelihood for LOO-CV
  for (t in 1:N) {
    if (t == 1) {
      log_lik[t] = normal_lpdf(epsilon[t] | 0, sigma_regime[regime[t]] / sqrt(1 - phi^2));
    } else {
      log_lik[t] = normal_lpdf(epsilon[t] | phi * epsilon[t-1], sigma_regime[regime[t]]);
    }

    // Posterior predictive (back-transformed)
    C_pred[t] = lognormal_rng(mu[t], sigma_regime[regime[t]]);
  }
}
```

### 1.7 Expected Challenges

1. **Computational**: AR structure creates dependencies → slower sampling
2. **Identifiability**: If phi is large, may absorb trends (confounding)
3. **Regime boundaries**: Fixed boundaries may be suboptimal (alternative: estimate changepoints)
4. **Back-transformation bias**: Need to add σ²/2 correction for mean predictions

### 1.8 Success Metrics

- **Convergence**: All Rhat < 1.01, ESS > 400
- **Residual ACF**: <0.2 at all lags up to 5
- **LOO-CV**: ELPD within 2 SE of best model
- **Posterior predictive checks**: 95% of observed counts in 95% prediction intervals
- **Regime variance ordering**: Consistent with EDA (middle > late > early)

---

## Model 2: Gaussian Process on Log-Scale (FLEXIBLE TEMPORAL STRUCTURE)

### 2.1 Model Specification

```
# Likelihood
log(C[t]) ~ Normal(f[t], sigma)

# Mean function
f[t] ~ GP(m(year[t]), k(year[t], year[t']))

# Mean function (exponential trend)
m(year[t]) = alpha + beta_1 * year[t] + beta_2 * year[t]^2

# Kernel (Matern 3/2 for smoothness without over-smoothing)
k(year[t], year[t']) = eta^2 * Matern32(year[t], year[t'], rho)

# Parameters
alpha ~ Normal(4.3, 0.5)
beta_1 ~ Normal(0.86, 0.2)
beta_2 ~ Normal(0, 0.3)
eta ~ HalfNormal(0, 0.5)  # GP marginal SD (residual variation around trend)
rho ~ InvGamma(5, 5)  # Length-scale (5, 5 → mean ≈ 1 standardized year unit)
sigma ~ HalfNormal(0, 0.3)  # Observation noise (small, since counts are "clean")
```

### 2.2 Rationale: Why GP?

**Strengths**:
1. **Discovers arbitrary temporal patterns**: No need to specify AR order
2. **Handles regime shifts**: Can capture abrupt changes via short length-scales
3. **Quantifies temporal correlation**: Length-scale rho tells us correlation decay rate
4. **Flexible smoothness**: Matern 3/2 allows for trends that are continuous but not infinitely smooth

**Why Matern 3/2 over Squared Exponential**:
- SE assumes infinitely differentiable functions (too smooth)
- Matern 3/2 allows for "kinks" and regime transitions
- More robust to model misspecification

### 2.3 Prior Justification

**eta (GP marginal standard deviation)**:
- Captures deviations from quadratic trend
- HalfNormal(0, 0.5): 95% quantile ≈ 0.8
- On log-scale, σ=0.5 → multiplicative factors of exp(±0.5) ≈ 0.6x to 1.6x
- **Prior predictive**: Temporal wiggles can cause 40% deviations from trend

**rho (length-scale)**:
- InvGamma(5, 5): Mean = 5/(5-1) = 1.25, mode = 5/6 ≈ 0.83
- Shape chosen to concentrate mass near 1 standardized year unit
- **Interpretation**: Correlation between observations ~1 year apart
- **Prior predictive**: Strong correlation within 1 year, weak beyond 2-3 years

**sigma (observation noise)**:
- HalfNormal(0, 0.3): Small, because counts are direct measurements (not noisy)
- Allows for small measurement error or Poisson-like variability on log-scale
- **Why small?** GP already captures temporal variation via f[t]

### 2.4 Computational Considerations

**Challenge**: GP requires inverting N×N covariance matrix → O(N³) operations
- For N=40, this is manageable (~64K operations)
- Stan's Cholesky decomposition is numerically stable

**Efficiency tricks**:
1. Use `gp_exp_quad_cov` or Matern functions directly
2. Add jitter (1e-9) to diagonal for numerical stability
3. Consider approximate GP (e.g., Hilbert space approximation) if slow

**Expected runtime**: 2-5 minutes for 4 chains × 2000 iterations

### 2.5 Falsification Criteria (I WILL ABANDON THIS IF...)

1. **Length-scale posterior hits boundary**: If rho → ∞ (no correlation) or rho → 0 (white noise)
   - **Action**: GP adds no value over independent errors; revert to Model 1

2. **eta near zero**: If GP variance component is negligible
   - **Action**: Data fully explained by quadratic trend; no need for GP

3. **Poor mixing**: If ESS < 100 or Rhat > 1.05 after 4000 iterations
   - **Action**: GP is over-parameterized; simplify to AR(p)

4. **Posterior predictive shows over-smoothing**: If predictions miss sharp regime transitions
   - **Action**: GP assumes continuity; switch to discrete changepoint model

5. **LOO-CV worse than AR(1) by >4 ELPD**: GP's flexibility not justified by fit
   - **Action**: Abandon GP for simpler AR structure

6. **Marginal SD scales with mean**: If residuals show heteroscedasticity despite log-transform
   - **Action**: Consider observation model on original scale (e.g., Negative Binomial)

### 2.6 Implementation Plan (Stan/CmdStanPy)

```stan
functions {
  // Matern 3/2 kernel
  matrix matern32_cov(vector x, real eta, real rho) {
    int N = rows(x);
    matrix[N, N] K;
    real dist;
    for (i in 1:N) {
      K[i, i] = eta^2 + 1e-9;  // Jitter for stability
      for (j in (i+1):N) {
        dist = fabs(x[i] - x[j]) / rho;
        K[i, j] = eta^2 * (1 + sqrt(3) * dist) * exp(-sqrt(3) * dist);
        K[j, i] = K[i, j];
      }
    }
    return K;
  }
}

data {
  int<lower=1> N;
  vector[N] year;
  vector[N] log_C;
}

parameters {
  real alpha;
  real beta_1;
  real beta_2;
  real<lower=0> eta;
  real<lower=0> rho;
  real<lower=0> sigma;
  vector[N] f_std;  // Standardized GP values
}

transformed parameters {
  vector[N] mu;
  vector[N] f;
  matrix[N, N] K;
  matrix[N, N] L_K;

  mu = alpha + beta_1 * year + beta_2 * square(year);
  K = matern32_cov(year, eta, rho);
  L_K = cholesky_decompose(K);
  f = mu + L_K * f_std;  // Non-centered parameterization
}

model {
  // Priors
  alpha ~ normal(4.3, 0.5);
  beta_1 ~ normal(0.86, 0.2);
  beta_2 ~ normal(0, 0.3);
  eta ~ normal(0, 0.5);
  rho ~ inv_gamma(5, 5);
  sigma ~ normal(0, 0.3);

  f_std ~ std_normal();  // Non-centered parameterization

  // Likelihood
  log_C ~ normal(f, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] C_pred;

  for (t in 1:N) {
    log_lik[t] = normal_lpdf(log_C[t] | f[t], sigma);
    C_pred[t] = lognormal_rng(f[t], sigma);
  }
}
```

### 2.7 Expected Challenges

1. **Computational cost**: O(N³) scales poorly if extended to larger datasets
2. **Prior sensitivity**: GP priors notoriously influence posterior (especially rho)
3. **Interpretation**: Harder to explain than AR(1) coefficients
4. **Extrapolation**: GP predictions revert to mean function outside data range

### 2.8 Success Metrics

- **Length-scale**: Posterior mode between 0.5-2.0 (reasonable temporal correlation)
- **Marginal SD**: eta > 0.2 (GP captures meaningful variation)
- **LOO-CV**: Competitive with or better than AR(1) model
- **Residuals**: No remaining autocorrelation structure
- **Visual inspection**: GP captures regime transitions smoothly

---

## Model 3: Hierarchical Period Model with AR(1) (EXPLICIT REGIME STRUCTURE)

### 3.1 Model Specification

```
# Likelihood (within regime)
log(C[t]) ~ Normal(mu_regime[regime[t]] + beta_regime[regime[t]] * (year[t] - year_center[regime[t]]),
                   sigma_regime[regime[t]])

# AR(1) structure within regime
epsilon[t] = log(C[t]) - (mu_regime[regime[t]] + beta_regime[regime[t]] * (year[t] - year_center[regime[t]]))
epsilon[t] ~ Normal(phi_regime[regime[t]] * epsilon[t-1], sigma_regime[regime[t]])

# Hierarchical priors on regime-specific parameters
mu_regime[r] ~ Normal(mu_global, tau_mu)
beta_regime[r] ~ Normal(beta_global, tau_beta)
phi_regime[r] ~ Normal(phi_global, tau_phi)
sigma_regime[r] ~ HalfNormal(0, tau_sigma)

# Hyperpriors
mu_global ~ Normal(4.3, 0.5)
beta_global ~ Normal(0.86, 0.3)
phi_global ~ Uniform(-0.8, 0.8)
tau_mu ~ HalfNormal(0, 0.5)
tau_beta ~ HalfNormal(0, 0.3)
tau_phi ~ HalfNormal(0, 0.3)
tau_sigma ~ HalfNormal(0, 0.5)
```

### 3.2 Rationale: Why Hierarchical by Period?

**Strengths**:
1. **Explicit regime modeling**: Each period has own intercept, slope, AR coefficient, variance
2. **Partial pooling**: Borrows strength across regimes (avoids overfitting with only 13-14 obs/regime)
3. **Tests regime hypothesis**: If tau parameters are small, regimes are similar
4. **Interpretable**: Can report "early period growth rate = X%, late period = Y%"

**Key design choice**: AR(1) coefficient also varies by regime
- EDA shows heterogeneous autocorrelation might exist
- If not needed, posterior will shrink phi_regime toward common phi_global

### 3.3 Prior Justification

**Hyperpriors (global parameters)**:
- Same as Model 1 for comparability
- Represent "typical" behavior across all regimes

**Tau parameters (regime heterogeneity)**:
- HalfNormal(0, 0.3-0.5): Weakly informative, allows substantial variation
- tau_mu: Inter-regime differences in baseline counts
- tau_beta: Inter-regime differences in growth rates
- tau_phi: Inter-regime differences in autocorrelation
- tau_sigma: Inter-regime differences in residual variance

**Why not tighter hyperpriors?**
- EDA shows 7.8× difference between early/late periods
- On log-scale: log(7.8) ≈ 2.05, so need tau_mu ~ 0.5-1.0 to allow this
- Let data determine pooling strength

### 3.4 Handling Regime Boundaries

**Critical issue**: AR(1) structure at regime boundaries
- Option A: Treat first observation of each regime as independent
- Option B: Allow AR(1) to carry across boundaries (continuity assumption)
- **Chosen**: Option A (conservative, assumes potential breaks)

**Implementation**:
```stan
for (t in 2:N) {
  if (regime[t] != regime[t-1]) {
    // New regime: initialize from stationary distribution
    epsilon[t] ~ normal(0, sigma_regime[regime[t]] / sqrt(1 - phi_regime[regime[t]]^2));
  } else {
    // Within regime: AR(1) dynamics
    epsilon[t] ~ normal(phi_regime[regime[t]] * epsilon[t-1], sigma_regime[regime[t]]);
  }
}
```

### 3.5 Falsification Criteria (I WILL ABANDON THIS IF...)

1. **Tau parameters near zero**: If all 95% CIs for tau include zero and centered at zero
   - **Action**: No regime heterogeneity; pool all periods into Model 1

2. **Individual regime parameters diverge from global**: If posterior for any regime has Rhat > 1.05
   - **Action**: Not enough data per regime; simplify to 2 regimes (early vs late)

3. **Phi_regime posteriors very different**: If some regimes have phi ≈ 0, others phi ≈ 0.8
   - **Action**: AR structure is regime-dependent, but this is complex; consider separate models per regime

4. **Poor fit in middle regime**: If LOO-PIT shows systematic bias for middle period
   - **Action**: Middle regime may need different functional form (not just different parameters)

5. **Regime boundaries are wrong**: If residuals show sudden jumps within supposed regimes
   - **Action**: Estimate changepoints instead of fixed tertile splits

6. **Overfitting**: If LOO-CV ELPD is much worse than simpler models despite good in-sample fit
   - **Action**: Too many parameters for N=40; reduce to 2 regimes or pool some parameters

### 3.6 Implementation Plan (Stan/CmdStanPy)

```stan
data {
  int<lower=1> N;
  vector[N] year;
  vector[N] log_C;
  int<lower=1, upper=3> regime[N];
  vector[3] year_center;  // Center of each regime (for interpretability)
}

parameters {
  // Regime-specific parameters
  vector[3] mu_regime;
  vector[3] beta_regime;
  vector<lower=-0.95, upper=0.95>[3] phi_regime;
  vector<lower=0>[3] sigma_regime;

  // Hyperparameters
  real mu_global;
  real beta_global;
  real<lower=-0.8, upper=0.8> phi_global;
  real<lower=0> tau_mu;
  real<lower=0> tau_beta;
  real<lower=0> tau_phi;
  real<lower=0> tau_sigma;
}

transformed parameters {
  vector[N] epsilon;

  for (t in 1:N) {
    int r = regime[t];
    epsilon[t] = log_C[t] - (mu_regime[r] + beta_regime[r] * (year[t] - year_center[r]));
  }
}

model {
  // Hyperpriors
  mu_global ~ normal(4.3, 0.5);
  beta_global ~ normal(0.86, 0.3);
  phi_global ~ uniform(-0.8, 0.8);
  tau_mu ~ normal(0, 0.5);
  tau_beta ~ normal(0, 0.3);
  tau_phi ~ normal(0, 0.3);
  tau_sigma ~ normal(0, 0.5);

  // Regime-specific priors (hierarchical)
  mu_regime ~ normal(mu_global, tau_mu);
  beta_regime ~ normal(beta_global, tau_beta);
  phi_regime ~ normal(phi_global, tau_phi);
  sigma_regime ~ normal(0, tau_sigma);

  // Likelihood with regime-specific AR(1)
  for (t in 1:N) {
    int r = regime[t];
    if (t == 1 || regime[t] != regime[t-1]) {
      // First observation or regime boundary
      epsilon[t] ~ normal(0, sigma_regime[r] / sqrt(1 - phi_regime[r]^2));
    } else {
      // Within regime
      epsilon[t] ~ normal(phi_regime[r] * epsilon[t-1], sigma_regime[r]);
    }
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] C_pred;

  for (t in 1:N) {
    int r = regime[t];
    real mu_t = mu_regime[r] + beta_regime[r] * (year[t] - year_center[r]);

    if (t == 1 || regime[t] != regime[t-1]) {
      log_lik[t] = normal_lpdf(epsilon[t] | 0, sigma_regime[r] / sqrt(1 - phi_regime[r]^2));
    } else {
      log_lik[t] = normal_lpdf(epsilon[t] | phi_regime[r] * epsilon[t-1], sigma_regime[r]);
    }

    C_pred[t] = lognormal_rng(mu_t, sigma_regime[r]);
  }
}
```

### 3.7 Expected Challenges

1. **Overfitting risk**: 12 parameters (4 per regime) + 7 hyperparameters = 19 total for N=40
2. **Weak identification**: Only 13-14 observations per regime may not constrain regime-specific phi well
3. **Boundary artifacts**: Abrupt changes at regime boundaries may be artificial
4. **Centering**: Need to center year within each regime for interpretability

### 3.8 Success Metrics

- **Tau posteriors away from zero**: Confirms regime heterogeneity
- **Reasonable phi_regime**: All between 0.4-0.9 (consistent autocorrelation)
- **Sigma_regime ordering**: sigma[middle] > sigma[late] > sigma[early] (per EDA)
- **LOO-CV**: Competitive despite complexity (not >5 ELPD worse than simpler models)
- **Posterior predictive**: Captures regime shifts accurately

---

## Model Comparison and Selection Strategy

### Phase 1: Fit All Three Models

**Implementation order**:
1. Model 1 (AR1 Log-Normal) - simplest, establishes baseline
2. Model 3 (Hierarchical) - tests regime hypothesis explicitly
3. Model 2 (GP) - most flexible, use if others fail

### Phase 2: Convergence and Diagnostics

**For each model, check**:
1. Rhat < 1.01 for all parameters
2. ESS > 400 (bulk and tail)
3. No divergences or max treedepth warnings
4. Posterior predictive checks (95% coverage)

**If any model fails convergence**:
- Increase iterations (4000 → 8000)
- Try different initialization
- Reparameterize if needed
- If still fails: **abandon that model** (sign of misspecification)

### Phase 3: Model Selection Criteria

**Primary: LOO-CV (ELPD)**
- Use `arviz.loo()` on log_lik samples
- Compare ELPD differences (prefer simpler if within 2 SE)
- Check Pareto k diagnostics (all < 0.7)

**Secondary: Residual diagnostics**
- ACF of standardized residuals (should be <0.2)
- QQ-plots against theoretical Normal
- Runs test for independence

**Tertiary: Scientific plausibility**
- Do parameter estimates make sense?
- Are regime differences consistent with domain knowledge?
- Do predictions extrapolate reasonably?

### Phase 4: Sensitivity Analysis

**For best model**:
1. Prior sensitivity: Refit with wider/narrower priors
2. Regime boundary sensitivity: Try different split points (±2 observations)
3. Transformation sensitivity: Try sqrt() instead of log() (if back-transformation issues)

### Phase 5: Decision Points for Major Pivots

**STOP and reconsider everything if**:

1. **All models show poor residual diagnostics**
   - **Pivot**: Consider observation-level random effects or Student-t errors (heavy tails)

2. **GP length-scale → 0 (white noise)**
   - **Pivot**: Temporal structure is artifact; use independent errors

3. **AR coefficients near 1.0 (unit root)**
   - **Pivot**: Non-stationary process; use differencing or state-space model

4. **Back-transformed predictions systematically biased**
   - **Pivot**: Log-normal assumption fails; model on original count scale (NegBin, Poisson-lognormal)

5. **Regime parameters show no heterogeneity**
   - **Pivot**: No regime shifts; use pooled model

6. **Out-of-sample predictions collapse**
   - **Pivot**: Exponential growth is not sustainable; consider bounded growth (logistic, Gompertz)

---

## Red Flags and Warning Signs

### Computational Red Flags

1. **High divergences (>1%)**: Posterior geometry is pathological
   - Often means model is too complex or misspecified
   - Try reparameterization or simpler model

2. **Low ESS despite long chains (<100)**: Poor mixing
   - Parameter is weakly identified
   - Try stronger priors or remove parameter

3. **Runtime >10 minutes**: Model is too complex for N=40
   - Consider approximations or simpler structure

### Statistical Red Flags

1. **Posterior at prior boundary**: Prior is too restrictive or model is wrong
   - Example: phi → 0.95 (model wants non-stationarity)

2. **Extreme parameter values**: Model compensating for misspecification
   - Example: sigma_regime[1] = 0.01 (underdispersion not modeled correctly)

3. **LOO Pareto k > 0.7**: Influential observations or model instability
   - Investigate those observations; may indicate regime boundary errors

### Domain Red Flags

1. **Predicted growth continues exponentially**: Unsustainable
   - Add bounds or saturation effects

2. **Early period underdispersion**: Log-normal may not fit well here
   - Consider mixture model or period-specific distributions

3. **Negative counts in prediction intervals**: Back-transformation issue
   - Report on log-scale or use truncated predictions

---

## Alternative Models (If All Proposed Models Fail)

### Escape Route 1: Student-t Errors
If residuals show heavy tails despite log-transform:
```stan
log_C[t] ~ student_t(nu, mu[t], sigma)
nu ~ gamma(2, 0.1)  // Degrees of freedom
```

### Escape Route 2: Changepoint Model
If regime boundaries are wrong:
```stan
changepoints ~ Uniform(year[1], year[N])  // Estimate split points
```

### Escape Route 3: State-Space Model
If AR(1) is insufficient:
```stan
// Local level + trend model
alpha[t] = alpha[t-1] + beta[t-1] + eps_alpha[t]
beta[t] = beta[t-1] + eps_beta[t]
log_C[t] ~ Normal(alpha[t], sigma)
```

### Escape Route 4: Non-Exponential Growth
If exponential growth is unsustainable:
```stan
// Logistic growth
mu[t] = K / (1 + exp(-r * (year[t] - t0)))
K ~ Normal(300, 50)  // Carrying capacity
r ~ Normal(1, 0.5)   // Growth rate
```

---

## Implementation Checklist

### Setup
- [ ] Install CmdStanPy and ArviZ
- [ ] Load data, create regime indicators
- [ ] Center year within regimes (for Model 3)
- [ ] Compute log(C) and store for all models

### Model 1: AR(1) Log-Normal
- [ ] Write Stan code with AR(1) structure
- [ ] Prior predictive checks (sample from priors only)
- [ ] Fit model (4 chains, 2000 iterations)
- [ ] Convergence diagnostics (Rhat, ESS, divergences)
- [ ] Posterior predictive checks
- [ ] Residual ACF analysis
- [ ] Extract log_lik for LOO-CV

### Model 2: Gaussian Process
- [ ] Write Stan code with Matern 3/2 kernel
- [ ] Non-centered parameterization for efficiency
- [ ] Prior predictive checks
- [ ] Fit model (may need longer chains)
- [ ] Check length-scale posterior
- [ ] Compare LOO-CV to Model 1

### Model 3: Hierarchical
- [ ] Write Stan code with regime-specific parameters
- [ ] Handle regime boundary initialization
- [ ] Prior predictive checks
- [ ] Fit model
- [ ] Check tau posteriors (is heterogeneity present?)
- [ ] Compare regime-specific estimates

### Comparison
- [ ] LOO-CV for all three models
- [ ] ELPD differences with SE
- [ ] Pareto k diagnostics
- [ ] Residual diagnostics for best model
- [ ] Sensitivity analysis (prior, regime boundaries)

### Documentation
- [ ] Model code (Stan files)
- [ ] Convergence summaries
- [ ] LOO-CV comparison table
- [ ] Posterior plots (parameters, predictions)
- [ ] Residual diagnostics
- [ ] Final model selection rationale

---

## Expected Outcomes and Hypotheses

### Most Likely Outcome
**Model 1 (AR1 Log-Normal with Regimes) will be best**:
- Balances flexibility and parsimony
- Explicitly models known regime shifts
- AR(1) captures most autocorrelation
- Fast computation, easy interpretation

**Expected parameter ranges**:
- beta_1: [0.7, 1.0] (exponential growth confirmed)
- beta_2: [-0.2, 0.2] (mild curvature)
- phi: [0.6, 0.8] (strong but not extreme autocorrelation)
- sigma_regime: early < late < middle (based on EDA dispersion ratios)

### Alternative Outcome 1
**Model 2 (GP) is best**:
- Suggests regime boundaries are not at tertile splits
- Or, temporal structure is more complex than AR(1)
- Length-scale will reveal true correlation scale

### Alternative Outcome 2
**Model 3 (Hierarchical) is best**:
- Confirms strong regime heterogeneity
- Tau parameters will be large
- May reveal regime-specific autocorrelation patterns

### Surprising Outcome
**All models fail residual diagnostics**:
- Log-transform assumption may be wrong
- Pivot to observation-level count models (Negative Binomial, see Designer 1)
- Or, temporal structure is fundamentally non-stationary (state-space required)

---

## Stress Tests

### Test 1: Remove Last Regime
Refit all models on observations 1-27 only:
- Can models predict regime 3 accurately?
- If not, exponential growth assumption fails at high counts

### Test 2: Shuffle Within Regimes
Randomly permute observations within each regime:
- If fit is similar, AR structure is weak (model overcomplicated)
- If fit degrades substantially, AR structure is essential

### Test 3: Split Data Randomly
Fit on 70% random sample, predict on 30%:
- Test generalization to unseen time points
- If poor, temporal structure is not well-captured

### Test 4: Prior Sensitivity
Multiply all prior SDs by 0.5 and 2.0:
- If posteriors change substantially, data are weak relative to priors
- May need more regularization or simpler models

---

## Final Thoughts: Bayesian Modeling as Discovery

This modeling plan is not a script to follow blindly. It's a **hypothesis-testing framework**:

1. We hypothesize three different temporal structures (AR, GP, hierarchical)
2. We specify clear criteria for when to abandon each
3. We build in escape routes if all fail
4. We prioritize learning over task completion

**Success is not fitting all three models**. Success is:
- Discovering which temporal structure best describes the data
- Recognizing early when a model class is failing
- Pivoting to better approaches based on evidence
- Quantifying uncertainty honestly

**Remember**: The goal is understanding the data-generating process, not minimizing ELPD. If evidence suggests exponential growth is an artifact of the time window, or that log-transformation masks important count structure, **abandon these models** and start fresh.

**The best model is the one that teaches us something true about the world, not the one that was easiest to implement.**

---

## File Outputs

All code and results will be saved to:
- `/workspace/experiments/designer_2/model_1_ar1_lognormal.stan`
- `/workspace/experiments/designer_2/model_2_gp_logscale.stan`
- `/workspace/experiments/designer_2/model_3_hierarchical_ar1.stan`
- `/workspace/experiments/designer_2/prior_predictive_checks.py`
- `/workspace/experiments/designer_2/fit_models.py`
- `/workspace/experiments/designer_2/compare_models.py`
- `/workspace/experiments/designer_2/diagnostics.py`
- `/workspace/experiments/designer_2/results/` (figures and tables)

---

**Document prepared by**: Model Designer 2
**Analysis date**: 2025-10-30
**Framework**: Bayesian inference via Stan/CmdStanPy
**Philosophy**: Falsification-first, adaptive modeling
