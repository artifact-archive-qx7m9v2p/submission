# Non-Linear and Hierarchical Bayesian Models
## Designer 3: Complexity and Structural Change Focus

**Date**: 2025-10-29
**Designer**: Model Designer 3 (Non-linear & Hierarchical Specialist)
**Dataset**: Time series, n=40, severe overdispersion (Var/Mean=70.43)

---

## Executive Summary

Based on EDA findings showing **quadratic fit (R²=0.964) superior to exponential (R²=0.937)** and a **possible structural break at year=-0.21**, I propose three Bayesian models that handle non-linearity and complexity:

1. **Quadratic Negative Binomial with AR(1)** - Polynomial growth with autocorrelation
2. **Bayesian Changepoint Model** - Explicit regime switching at unknown location
3. **Gaussian Process Negative Binomial** - Non-parametric flexible mean function

**Critical Finding**: The 2.7% improvement in R² from quadratic vs exponential (0.964 vs 0.937) may not justify the extra parameter with n=40. **I will abandon quadratic models if ΔLOO < 4 SE relative to linear baseline.**

---

## Philosophy: Falsification-First Approach

### My Commitment to Finding Truth

**I will abandon ALL proposed models if:**
1. Linear baseline performs within 2×SE via LOO (Occam's razor)
2. Prior-posterior conflict indicates model fights the data
3. Extreme computational difficulties (often signals misspecification)
4. Posterior predictive checks show systematic bias
5. Leave-one-out diagnostics show poor calibration (Pareto k > 0.7)

**Success metric**: Finding the SIMPLEST model that genuinely explains the data, even if it's none of the complex models I propose below.

### Red Flags That Trigger Full Reconsideration

- **Identifiability problems**: Quadratic + AR(1) may compete for explaining curvature
- **Overfitting**: Complexity parameters with wide posteriors overlapping zero
- **Computational warnings**: Divergences, low ESS despite reparameterization
- **Poor calibration**: 90% intervals have <80% coverage
- **Inconsistent predictions**: Different data subsets give contradictory parameter estimates

### Decision Points for Strategy Pivots

1. **After Model 1 (Quadratic)**: If β₂ credible interval contains zero → STOP, use linear model
2. **After Model 2 (Changepoint)**: If changepoint posterior is uniform → STOP, no structural break
3. **After Model 3 (GP)**: If length scale → 0 (white noise) → STOP, data too noisy for GP
4. **Overall**: If all complex models fail LOO comparison → Report linear model as best

---

## Model 1: Quadratic Negative Binomial with AR(1)

### Theoretical Justification

**Why non-linearity is needed:**
- EDA shows acceleration: growth rate increases 9.6-fold across time
- Quadratic R² = 0.964 vs exponential R² = 0.937 (2.7% improvement)
- Visual inspection suggests S-curve behavior (slow-fast-slow)
- First differences show non-constant growth rate

**Why this might FAIL:**
- With n=40, quadratic term may capture noise, not signal
- AR(1) structure might absorb curvature (confounding)
- Overdispersion may dominate, making mean function shape irrelevant
- Extrapolation beyond data range could be wildly wrong

### Mathematical Specification

**Likelihood:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
η_t = β₀ + β₁·year_t + β₂·year_t² + ε_t
ε_t ~ AR(1): ε_t = ρ·ε_{t-1} + ν_t, ν_t ~ Normal(0, σ)
```

**Parameters:**
- β₀: Intercept (log-scale mean at year=0)
- β₁: Linear growth rate
- β₂: Acceleration parameter (KEY)
- φ: Negative binomial dispersion (inverse)
- ρ: AR(1) correlation coefficient
- σ: Innovation standard deviation

**Priors:**
```stan
// Centered at EDA findings
β₀ ~ Normal(4.7, 0.5)         // log(109) ≈ 4.7, wide enough for uncertainty
β₁ ~ Normal(1.0, 0.5)          // Expected positive growth, regularizing
β₂ ~ Normal(0, 0.3)            // WEAKLY INFORMATIVE: allows negative curvature
                                // Prior SD=0.3 → exp(0.3×1.67²) ≈ 1.5× effect at extremes

// Dispersion and correlation
φ ~ Gamma(2, 0.1)               // Mode around 10, allows severe overdispersion
ρ ~ Beta(20, 2)                 // Strong prior for high correlation (EDA: 0.971)
σ ~ Normal(0, 0.5)              // Innovation noise, truncated at 0
```

### Non-linearity Details

**Form**: Quadratic polynomial on log-scale
- At year=-1.67: log(μ) ≈ β₀ - 1.67β₁ + 2.79β₂
- At year=0: log(μ) = β₀
- At year=+1.67: log(μ) ≈ β₀ + 1.67β₁ + 2.79β₂

**β₂ Interpretation:**
- β₂ > 0: Accelerating growth (exponential-like)
- β₂ = 0: Constant growth rate (linear on log scale)
- β₂ < 0: Decelerating growth (logistic-like)

**Prior Predictive Check:**
With β₂ ~ Normal(0, 0.3):
- 95% prior range: β₂ ∈ [-0.6, 0.6]
- At year=1.67, this allows μ to multiply by exp(±1.68) = [0.19, 5.37]
- Regularizes extreme curvature while allowing EDA-observed patterns

### Falsification Criteria

**I will REJECT this model if:**
1. **β₂ credible interval contains zero** → Curvature not justified, use linear model
2. **ρ posterior near 0** → AR(1) structure unnecessary, simplify
3. **LOO shows ΔELPD < 4×SE vs linear** → Not worth extra parameter (n=40 is small!)
4. **Divergent transitions despite reparameterization** → Model geometry pathological
5. **Posterior predictive checks show systematic over/under-prediction** → Wrong functional form

**Specific red flags:**
- β₁ and β₂ posterior correlation > 0.9 → Non-identifiable
- Pareto k > 0.7 for >10% of observations → Poor pointwise predictions
- Prior-posterior overlap < 10% for β₂ → Data very surprised by curvature

**Comparison benchmark:**
- Must beat linear model by ΔLOO > 4 SE (conservative due to n=40)
- Must have better posterior predictive RMSE by >10%

### Stan Implementation Notes

**Parameterization strategy:**
```stan
// Non-centered for AR(1) to improve sampling
parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> phi;
  real<lower=-1, upper=1> rho;
  real<lower=0> sigma_ar;
  vector[N] z;  // Standard normal innovations
}

transformed parameters {
  vector[N] epsilon;
  vector[N] log_mu;

  epsilon[1] = sigma_ar * z[1] / sqrt(1 - rho^2);  // Stationary start
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + sigma_ar * z[t];
  }

  log_mu = beta_0 + beta_1 * year + beta_2 * year_sq + epsilon;
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);
  beta_1 ~ normal(1.0, 0.5);
  beta_2 ~ normal(0, 0.3);
  phi ~ gamma(2, 0.1);
  rho ~ beta(20, 2);
  sigma_ar ~ normal(0, 0.5);
  z ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu[t], phi);
    C_rep[t] = neg_binomial_2_rng(exp(log_mu[t]), phi);
  }
}
```

**Computational tricks:**
- Non-centered AR(1) via standard normal innovations
- Stationary initialization: ε₁ ~ Normal(0, σ/√(1-ρ²))
- Pre-compute year_sq in transformed data block
- Use neg_binomial_2_log for numerical stability

### Computational Cost

**Expected runtime:** 2-4 minutes
- 4 chains × 2000 iterations = 8000 posterior samples
- ~40 parameters (β + φ + ρ + σ + 40 epsilons)
- AR(1) structure increases sampling difficulty slightly

**Parsimony check:**
- 1 extra parameter (β₂) vs linear baseline
- Must justify via ΔLOO > 4 SE and |β₂| > 0 conclusively
- With n=40, risk of overfitting is REAL

**Diagnostic targets:**
- R-hat < 1.01 for all parameters
- ESS_bulk > 400, ESS_tail > 400
- No divergences (if present, increase adapt_delta to 0.95)
- Max tree depth warnings acceptable if ESS adequate

---

## Model 2: Bayesian Changepoint Negative Binomial

### Theoretical Justification

**Why structural break is plausible:**
- EDA detected changepoint at year = -0.21
- Growth rate changes 9.6-fold: 13.0 → 124.7 units/year
- Visual inspection suggests distinct early/late regimes
- Substantive interpretation: system undergoes phase transition

**Why this might FAIL:**
- Changepoint detection in EDA was exploratory (single analyst)
- With n=40, estimating τ, β₁, β₂, φ, ρ simultaneously is ambitious
- Continuous quadratic might explain "break" more parsimoniously
- If τ posterior is uniform → no evidence for discrete change

### Mathematical Specification

**Likelihood:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
η_t = β₀ + β₁·year_t + β₂·I(year_t > τ)·(year_t - τ) + ε_t
ε_t ~ AR(1): ε_t = ρ·ε_{t-1} + ν_t, ν_t ~ Normal(0, σ)
```

Where:
- τ: Changepoint location (UNKNOWN, to be estimated)
- β₁: Slope before changepoint
- β₂: Change in slope after changepoint (total slope = β₁ + β₂)
- Continuous at τ (no jump discontinuity)

**Alternative parameterization (piecewise):**
```
log(μ_t) = β₀ + β₁_pre·year_t·I(year_t ≤ τ) + β₁_post·year_t·I(year_t > τ) + δ·I(year_t > τ)
```
Where δ enforces continuity at τ.

**Priors:**
```stan
// Changepoint location
τ ~ Uniform(-1.5, 1.5)          // Avoid edges, focus on interior

// Regression coefficients
β₀ ~ Normal(4.7, 0.5)
β₁ ~ Normal(0.5, 0.5)            // Pre-change slope (slower)
β₂ ~ Normal(0, 1.0)              // Slope change (can be positive or negative)

// Dispersion and correlation
φ ~ Gamma(2, 0.1)
ρ ~ Beta(20, 2)
σ ~ Normal(0, 0.5)
```

### Non-linearity Details

**Form**: Piecewise linear with unknown break location

**Regime interpretation:**
- **Before τ**: log(μ_t) = β₀ + β₁·year_t (slow growth)
- **After τ**: log(μ_t) = β₀ + (β₁ + β₂)·year_t - β₂·τ (faster growth if β₂ > 0)

**Prior predictive for τ:**
- Uniform(-1.5, 1.5) covers central 80% of data range
- Avoids spurious breaks near edges (first/last 3 observations)
- If posterior matches prior → no information about break location

**Key parameters:**
- β₁: Pre-change growth (units per standardized year on log scale)
- β₂: Growth acceleration (positive if later period faster)
- τ: Time of regime shift

### Falsification Criteria

**I will REJECT this model if:**
1. **τ posterior is uniform** → No evidence for changepoint, use continuous model
2. **β₂ credible interval contains zero** → No slope change, use linear model
3. **LOO worse than quadratic model** → Smooth curvature explains data better
4. **Pareto k diagnostics poor** → Influential points near estimated changepoint
5. **Posterior predictive shows discontinuity artifacts** → Model misspecified

**Specific evidence that would falsify:**
- Posterior P(τ < -1) or P(τ > 1) high → Break at edge = spurious
- β₁ and β₂ posteriors negatively correlated > 0.95 → Non-identifiable trade-off
- Multiple local modes in τ posterior → Model cannot decide

**Switch to quadratic model if:**
- Quadratic has better LOO by >2 SE
- τ posterior has >2 modes (suggests smooth curvature, not discrete break)

**Switch to linear model if:**
- Both β₂ ≈ 0 AND quadratic β₂ ≈ 0

### Stan Implementation Notes

**Parameterization strategy:**
```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;              // Slope change
  real<lower=-1.5, upper=1.5> tau;  // Changepoint
  real<lower=0> phi;
  real<lower=-1, upper=1> rho;
  real<lower=0> sigma_ar;
  vector[N] z;
}

transformed parameters {
  vector[N] epsilon;
  vector[N] log_mu;
  vector[N] year_after_cp;

  // AR(1) errors
  epsilon[1] = sigma_ar * z[1] / sqrt(1 - rho^2);
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + sigma_ar * z[t];
  }

  // Changepoint regression
  year_after_cp = year - tau;
  for (t in 1:N) {
    if (year[t] > tau) {
      log_mu[t] = beta_0 + beta_1 * year[t] + beta_2 * year_after_cp[t] + epsilon[t];
    } else {
      log_mu[t] = beta_0 + beta_1 * year[t] + epsilon[t];
    }
  }
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);
  beta_1 ~ normal(0.5, 0.5);
  beta_2 ~ normal(0, 1.0);
  tau ~ uniform(-1.5, 1.5);
  phi ~ gamma(2, 0.1);
  rho ~ beta(20, 2);
  sigma_ar ~ normal(0, 0.5);
  z ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  real slope_pre = beta_1;
  real slope_post = beta_1 + beta_2;
  real rate_change = slope_post / slope_pre;  // Fold-change in growth

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu[t], phi);
    C_rep[t] = neg_binomial_2_rng(exp(log_mu[t]), phi);
  }
}
```

**Computational tricks:**
- Center changepoint at 0 in transformed data if needed
- Use vectorized operations where possible
- Monitor for multimodality in τ (run multiple chains with dispersed inits)
- Consider fixing τ = -0.21 as sensitivity analysis

**Potential issues:**
- Discrete changepoint causes non-smooth posterior
- May need higher adapt_delta (0.95 or 0.99)
- ESS for τ may be lower than other parameters (acceptable if >100)

### Computational Cost

**Expected runtime:** 4-8 minutes
- Changepoint estimation is computationally demanding
- Non-smooth posterior surface for τ
- May need 3000-4000 iterations for adequate ESS

**Parsimony check:**
- 2 extra parameters (β₂, τ) vs linear baseline
- Must justify via ΔLOO > 6 SE (2 parameters × conservative multiplier)
- High bar given n=40

**Diagnostic targets:**
- R-hat < 1.01, but τ may be slightly higher (1.02 acceptable)
- ESS_bulk > 100 for τ, >400 for others
- Divergences likely; if >1%, increase adapt_delta
- Check trace plots for τ to verify mixing

---

## Model 3: Gaussian Process Negative Binomial

### Theoretical Justification

**Why non-parametric flexibility is needed:**
- Don't want to impose quadratic vs exponential vs piecewise a priori
- Let data determine functional form
- May reveal features missed by parametric models
- Provides uncertainty quantification about mean function shape

**Why this might FAIL:**
- n=40 is small for GP (typically n>100 desirable)
- Severe overdispersion + GP flexibility may overfit wildly
- Computational cost is O(n³) for dense covariance matrix
- Length scale may be unidentifiable with limited data
- GP + AR(1) is redundant (both model correlation)

**Critical assumption to test:**
This model assumes the mean function is a smooth realization from a GP. If the data has sharp discontinuities or the GP length scale collapses to near-zero, the model is inappropriate.

### Mathematical Specification

**Likelihood:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = f(year_t)
f ~ GP(m, k)
```

Where:
- f: Latent Gaussian process over year
- m: Mean function (typically constant or linear trend)
- k: Covariance kernel (squared exponential / RBF)

**Kernel specification:**
```
k(year_i, year_j) = α² · exp(-0.5 · ((year_i - year_j) / ℓ)²)
```

Parameters:
- α: Marginal standard deviation (amplitude)
- ℓ: Length scale (how quickly correlation decays)
- φ: Negative binomial dispersion

**Priors:**
```stan
// GP hyperparameters
α ~ Normal(0, 1)                 // Moderate variation on log scale
ℓ ~ InvGamma(3, 3)               // Mode around 1 (spans ~1 SD of year)
                                 // Longer length scale = smoother function

// Mean function (linear trend)
β₀ ~ Normal(4.7, 0.5)
β₁ ~ Normal(1.0, 0.5)
m(year_t) = β₀ + β₁·year_t

// Dispersion
φ ~ Gamma(2, 0.1)
```

**Full specification:**
```
log(μ_t) = β₀ + β₁·year_t + f_t
f ~ GP(0, K)   where K_ij = α² · exp(-0.5 · ((year_i - year_j) / ℓ)²)
```

### Non-linearity Details

**Form**: Non-parametric, data-driven smooth function

**Length scale interpretation:**
- ℓ → 0: Function is white noise (no smoothness)
- ℓ ≈ 0.5: Correlation decays over ~0.5 standardized years (short-range)
- ℓ ≈ 2: Correlation spans entire dataset (very smooth, near-linear)
- ℓ → ∞: Function reduces to constant + linear trend

**Prior predictive check:**
With α ~ Normal(0, 1) and ℓ ~ InvGamma(3, 3):
- Typical realizations: smooth wiggles with 1-3 inflection points
- Amplitude: log(μ) varies by ±2 units → μ varies by 7-fold
- Smoothness: no sharp discontinuities

**Key quantities to monitor:**
- Length scale ℓ: If posterior mean < 0.3 → data too noisy for GP
- Length scale ℓ: If posterior mean > 3 → GP collapses to linear trend
- Marginal SD α: If near zero → no deviation from mean function

### Falsification Criteria

**I will REJECT this model if:**
1. **ℓ posterior concentrated near 0** → White noise, GP inappropriate
2. **ℓ posterior concentrated near upper bound** → Reduces to linear, use simpler model
3. **Computational failure** → Cholesky decomposition issues, ill-conditioned covariance
4. **LOO worse than quadratic** → Flexibility doesn't help, overfitting
5. **Posterior predictive shows unreasonable extrapolations** → GP uncertainty explodes

**Specific red flags:**
- Divergent transitions that persist despite reparameterization
- Pareto k > 0.7 for many observations (GP overfits specific points)
- Posterior predictive credible intervals absurdly wide
- f(year) has >5 inflection points (overfitting noise)

**Evidence that would make me switch models:**
- If ℓ > 2 and GP ≈ linear → Use Model 1 (linear) as it's simpler
- If ℓ < 0.3 → Data too noisy, use overdispersion models without GP
- If GP finds clear "kink" → Use Model 2 (changepoint) for interpretability

### Stan Implementation Notes

**Parameterization strategy (centered):**
```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

transformed data {
  // Precompute distance matrix for efficiency
  matrix[N, N] dist;
  for (i in 1:N) {
    for (j in 1:N) {
      dist[i, j] = square(year[i] - year[j]);
    }
  }
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> alpha_gp;      // GP marginal SD
  real<lower=0> length_scale;  // GP length scale
  vector[N] eta;               // Non-centered GP innovations
  real<lower=0> phi;
}

transformed parameters {
  vector[N] f;
  vector[N] log_mu;
  matrix[N, N] L_K;            // Cholesky of covariance
  matrix[N, N] K;

  // Construct covariance matrix
  K = alpha_gp^2 * exp(-0.5 * dist / length_scale^2);
  for (n in 1:N) {
    K[n, n] = K[n, n] + 1e-9;  // Numerical stability (jitter)
  }

  L_K = cholesky_decompose(K);
  f = L_K * eta;
  log_mu = beta_0 + beta_1 * year + f;
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);
  beta_1 ~ normal(1.0, 0.5);
  alpha_gp ~ normal(0, 1);
  length_scale ~ inv_gamma(3, 3);
  eta ~ std_normal();
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu[t], phi);
    C_rep[t] = neg_binomial_2_rng(exp(log_mu[t]), phi);
  }
}
```

**Alternative: Use Stan's built-in GP functions**
```stan
// In transformed parameters
f = gp_exp_quad_cov_lpdf(year, alpha_gp, length_scale);
```

**Computational tricks:**
- Cholesky decomposition for numerical stability
- Add small jitter (1e-9) to diagonal for positive definiteness
- Consider approximate GP methods if n were larger (HSGP, Vecchia)
- Non-centered parameterization via eta ~ std_normal()

**Potential issues:**
- O(n³) complexity: 40³ = 64,000 operations per iteration
- Cholesky decomposition can fail if ℓ too small (ill-conditioned K)
- May need very high adapt_delta (0.99) to avoid divergences

### Computational Cost

**Expected runtime:** 10-20 minutes
- Dense n×n covariance matrix operations
- Cholesky decomposition at each iteration
- May need longer chains (3000 iterations) for convergence

**Parsimony check:**
- 2 extra parameters (α, ℓ) PLUS n latent f values = 42 extra parameters!
- Effective complexity much higher than Models 1-2
- Must justify via DRAMATIC improvement in LOO (ΔELPD > 10 SE)
- With n=40, this is a long shot

**Diagnostic targets:**
- R-hat < 1.01 for hyperparameters (ℓ, α)
- ESS_bulk > 400 for hyperparameters
- ESS for individual f_t can be lower (>100 acceptable)
- Divergences likely; if >5%, model may be inappropriate

**Reality check:**
- GP with n=40 is on the edge of feasibility
- If computational issues arise, ABANDON this model
- Simpler parametric models (1-2) are more appropriate for small data

---

## Model Comparison Strategy

### Sequential Testing Approach

**Stage 1: Establish baseline (Linear NB + AR(1))**
- Fit simplest defensible model
- Document baseline LOO, RMSE, coverage

**Stage 2: Test polynomial complexity (Model 1)**
- Fit quadratic model
- Compare ΔLOO vs baseline
- **Decision rule**: If ΔLOO < 4 SE → STOP, use linear
- Check β₂ credible interval excludes zero

**Stage 3: Test structural break (Model 2)**
- Only proceed if Model 1 failed OR if clear regime shift visible
- Compare to both baseline and Model 1
- **Decision rule**: If τ posterior is uniform → REJECT
- If β₂ interval contains zero → REJECT

**Stage 4: Test non-parametric (Model 3)**
- ONLY proceed if Models 1-2 show systematic posterior predictive failures
- This is a "stress test" to see if parametric forms are fundamentally wrong
- **Decision rule**: If computational issues → STOP
- Must beat best parametric model by ΔLOO > 10 SE to justify complexity

### Leave-One-Out Cross-Validation

**Primary metric:** Expected log predictive density (ELPD)
```
ΔELPD = ELPD_complex - ELPD_simple
SE_diff = Standard error of difference
```

**Interpretation:**
- ΔELPD > 0: Complex model preferred
- |ΔELPD| < 2 SE: Models equivalent, prefer simpler
- |ΔELPD| > 4 SE: Strong evidence for difference

**Diagnostic checks:**
- Pareto k < 0.5: Reliable LOO approximation (good)
- 0.5 < Pareto k < 0.7: Less reliable but usable (okay)
- Pareto k > 0.7: Poor approximation, use K-fold CV (bad)

### Posterior Predictive Checks

**Graphical:**
1. Overlay 100 posterior predictive draws on observed data
2. Plot residuals vs year (should be structureless)
3. Q-Q plot of standardized residuals
4. Rootogram: observed vs expected counts by bin

**Numerical:**
```
T_obs = test statistic on observed data
T_rep = test statistic on replicated data
p-value = P(T_rep > T_obs)
```

Test statistics:
- Mean, variance, max, min
- Temporal autocorrelation
- Variance-to-mean ratio

**Desirable:** p-values between 0.05 and 0.95 (model captures feature)

### Model Selection Criteria

**Rank models by:**
1. **Falsification**: Does model pass basic sanity checks?
2. **LOO**: Predictive performance (ΔELPD ± SE)
3. **Parsimony**: Simpler model preferred if LOO equivalent
4. **Interpretability**: Can we explain results to domain experts?
5. **Computational tractability**: Can we actually fit the model?

**Final decision:**
- If all complex models fail → Report linear baseline as best
- If one clear winner → Report that model
- If multiple models within 2 SE → Report ensemble

---

## Prior Predictive Checks (To Be Performed)

### Model 1 (Quadratic)

**Simulate 1000 datasets from prior:**
```python
β₀ ~ Normal(4.7, 0.5)
β₁ ~ Normal(1.0, 0.5)
β₂ ~ Normal(0, 0.3)
φ ~ Gamma(2, 0.1)

log_μ = β₀ + β₁·year + β₂·year²
C ~ NegativeBinomial(exp(log_μ), φ)
```

**Check:**
- Do simulated counts stay positive? (should be >95%)
- Do simulated counts span reasonable range [10, 500]? (should be >90%)
- Does curvature look plausible? (visual inspection)

**Adjust priors if:**
- >5% of simulated datasets have negative counts → tighten priors
- Curvature too extreme (>10 inflection points) → reduce SD(β₂)

### Model 2 (Changepoint)

**Simulate 1000 datasets from prior:**
```python
τ ~ Uniform(-1.5, 1.5)
β₁ ~ Normal(0.5, 0.5)
β₂ ~ Normal(0, 1.0)

log_μ = β₀ + β₁·year + β₂·I(year > τ)·(year - τ)
C ~ NegativeBinomial(exp(log_μ), φ)
```

**Check:**
- Is changepoint location uniform across simulations? (should be)
- Are both regimes plausible? (visual inspection)
- Any simulations with absurd breaks? (should be <5%)

### Model 3 (GP)

**Simulate 100 datasets from prior:**
```python
ℓ ~ InvGamma(3, 3)
α ~ Normal(0, 1)
f ~ GP(0, K) where K = α²·exp(-dist²/(2ℓ²))

log_μ = β₀ + β₁·year + f
C ~ NegativeBinomial(exp(log_μ), φ)
```

**Check:**
- Are GP realizations smooth? (should be, given kernel)
- Is amplitude reasonable? (log_μ should vary by <5 units)
- Do counts span reasonable range? (visual inspection)

**Adjust priors if:**
- GP realizations too wiggly → increase ℓ prior concentration
- Counts span [1, 10^6] → tighten α prior

---

## Implementation Roadmap

### Phase 1: Prior Predictive Checks (Day 1)
1. Write Stan programs for all 3 models
2. Simulate 1000 datasets from each prior
3. Visualize and summarize prior predictive distributions
4. Adjust priors if necessary
5. **Stop criterion**: If priors generate absurd data >10% of time → revise

### Phase 2: Fit Models (Day 1-2)
1. Fit baseline (linear NB + AR(1))
2. Fit Model 1 (quadratic)
3. **Decision point**: If Model 1 fails, SKIP Models 2-3
4. Fit Model 2 (changepoint) if justified
5. Fit Model 3 (GP) if justified

### Phase 3: Diagnostics (Day 2)
1. Check convergence (R-hat, ESS, divergences)
2. Posterior predictive checks
3. LOO cross-validation
4. **Stop criterion**: If all models fail diagnostics → reconsider approach

### Phase 4: Model Comparison (Day 2-3)
1. Compute ΔLOO with standard errors
2. Compare to linear baseline
3. Rank models by LOO + parsimony
4. **Decision**: Select best model OR report ensemble

### Phase 5: Reporting (Day 3)
1. Document parameter estimates with uncertainty
2. Visualize posterior predictive distributions
3. Explain model choice and alternatives considered
4. **Critical**: Document what would have made us choose differently

---

## Escape Routes: Alternative Models If All Fail

### If all non-linear models fail LOO comparison:
**Conclusion**: Linear model is sufficient. Report:
```
log(E[C_t]) = β₀ + β₁·year_t
C_t ~ NegativeBinomial(μ_t, φ)
```

### If AR(1) structure is problematic (non-convergence):
**Alternative**: Drop AR(1), use robust SEs
```
log(E[C_t]) = β₀ + β₁·year_t + β₂·year_t²
C_t ~ NegativeBinomial(μ_t, φ)
errors: independent
```

### If Negative Binomial overdispersion insufficient:
**Alternative**: Conway-Maxwell-Poisson or generalized Poisson
```
C_t ~ COM-Poisson(λ_t, ν)
```
Where ν controls dispersion more flexibly than NB.

### If all Bayesian models computationally intractable:
**Alternative**: Frequentist GLMM with random effects
```r
glmmTMB(C ~ year + I(year^2) + (1|time_block),
        family=nbinom2)
```

### If data suggests discrete mixture (not smooth growth):
**Alternative**: Finite mixture of Negative Binomials
```
C_t ~ π·NB(μ₁, φ₁) + (1-π)·NB(μ₂, φ₂)
```

---

## Success Metrics

### Computational Success
- ✓ R-hat < 1.01 for all parameters
- ✓ ESS > 400 for key parameters (β, φ)
- ✓ <1% divergent transitions
- ✓ Fit completes in <30 minutes

### Statistical Success
- ✓ Posterior predictive p-values ∈ [0.05, 0.95]
- ✓ 90% credible intervals have ~90% empirical coverage
- ✓ Pareto k < 0.7 for >90% of observations
- ✓ RMSE better than baseline by >10%

### Scientific Success
- ✓ Parameter estimates scientifically plausible
- ✓ Uncertainty quantification reasonable (not too wide/narrow)
- ✓ Model predictions extrapolate sensibly
- ✓ Results interpretable to domain experts

### Falsification Success
- ✓ At least one model REJECTED based on pre-specified criteria
- ✓ Documented why models failed (learning occurred)
- ✓ Final model survives stress tests

**Most important**: Finding a model that GENUINELY explains the data, even if it's simpler than proposed.

---

## Stress Tests

### Test 1: Extreme Extrapolation
- Predict at year = ±3 (well beyond data range)
- **Acceptable**: Wide uncertainty, but reasonable point estimates
- **Failure**: Absurd predictions (negative counts, 10^6 counts)

### Test 2: Leave-Out Blocks
- Remove first 10 observations, refit
- Remove last 10 observations, refit
- **Acceptable**: Parameter estimates stable within 2 SE
- **Failure**: Wildly different parameters → model unstable

### Test 3: Prior Sensitivity
- Refit with priors 2× wider and 2× narrower
- **Acceptable**: Posterior changes < 20%
- **Failure**: Posterior changes >50% → prior-driven conclusions

### Test 4: Measurement Error
- Add noise to 'year' variable: year_noisy = year + ε, ε ~ Normal(0, 0.1)
- **Acceptable**: Results qualitatively similar
- **Failure**: Conclusions reverse → model fragile

---

## Communication Plan

### For Synthesis Agent

**If models succeed:**
"Model [X] provides the best balance of fit, parsimony, and interpretability. ΔLOO = [value] ± [SE] relative to linear baseline. Key finding: [quadratic/changepoint/smooth] structure explains [Y]% of deviance. Recommendation: Proceed with Model [X]."

**If models fail:**
"All proposed non-linear models failed to justify complexity relative to linear baseline. ΔLOO values: Model 1 = [value], Model 2 = [value], Model 3 = [value], all <4 SE. Falsification criteria met: [specific reasons]. Recommendation: Use linear Negative Binomial with AR(1) as final model. This is SUCCESS—we learned the data are simpler than initially hypothesized."

**If computational failure:**
"Models 2-3 encountered computational intractability [specific issues]. This suggests the data are too noisy or n too small for complex structures. Recommendation: Restrict to Model 1 (quadratic) or linear baseline. This failure is INFORMATIVE—it reveals model misspecification."

### For Domain Experts

**Explain non-linearity in plain language:**
- Quadratic: "Growth rate itself is increasing over time"
- Changepoint: "System behavior fundamentally changed around [time]"
- GP: "We let the data determine the shape without assuming a formula"

**Explain uncertainty:**
- "We're 90% confident the parameter is between [a, b]"
- "The model predicts [Y] with 90% interval [Y_low, Y_high]"

**Explain model comparison:**
- "Model A predicts held-out data better than Model B by [ΔLOO] units"
- "The improvement does/doesn't justify the added complexity"

---

## Final Philosophical Statement

**My job is NOT to advocate for complex models.**

My job is to:
1. Propose plausible complex models based on EDA
2. Specify them rigorously with falsification criteria
3. Test them honestly against simpler alternatives
4. Report failures as successes (we learned something)
5. Recommend the simplest model that genuinely explains the data

**If the linear model wins, I have succeeded.**

This means the EDA findings of "quadratic superiority" were overfitting artifacts, or the 2.7% R² improvement doesn't translate to better out-of-sample prediction. That's valuable knowledge.

**If a complex model wins, I must justify it ruthlessly.**

It must beat the baseline by a large margin, pass all diagnostics, and survive stress tests. Extraordinary claims (non-linearity, changepoints, GPs) require extraordinary evidence.

**I am optimizing for scientific truth, not task completion.**

---

## Appendix: Mathematical Details

### Negative Binomial Parameterizations

Stan uses `neg_binomial_2(mu, phi)`:
```
Var(C) = μ + μ²/φ
```

Where:
- μ = E[C] (mean)
- φ = dispersion parameter (INVERSE of traditional α)
- As φ → ∞: Variance → μ (approaches Poisson)
- As φ → 0: Variance → ∞ (severe overdispersion)

**Connection to traditional NB:**
```
NB(r, p) where:
r = φ
p = φ/(φ + μ)
```

### AR(1) Process Stationarity

For ε_t = ρ·ε_{t-1} + ν_t with ν_t ~ Normal(0, σ):

**Stationary distribution:**
```
ε_t ~ Normal(0, σ²/(1 - ρ²))  for |ρ| < 1
```

**Initialization:**
```
ε_1 ~ Normal(0, σ/sqrt(1 - ρ²))  ensures stationarity
```

**Autocorrelation function:**
```
Cor(ε_t, ε_{t+k}) = ρ^k
```

### Gaussian Process Kernels

**Squared exponential (RBF):**
```
k(x, x') = α² · exp(-(x - x')²/(2ℓ²))
```

Properties:
- Infinitely differentiable (very smooth)
- Length scale ℓ controls correlation distance
- Marginal variance α² controls amplitude

**Alternative kernels:**
- Matérn 3/2: Less smooth, more robust
- Matérn 5/2: Intermediate smoothness
- Rational quadratic: Mixture of length scales

For this dataset, RBF is appropriate given EDA shows smooth growth.

---

## File Manifest

This document: `/workspace/experiments/designer_3/proposed_models.md`

**To be created:**
- `/workspace/experiments/designer_3/model_1_quadratic.stan`
- `/workspace/experiments/designer_3/model_2_changepoint.stan`
- `/workspace/experiments/designer_3/model_3_gp.stan`
- `/workspace/experiments/designer_3/prior_predictive_checks.py`
- `/workspace/experiments/designer_3/fit_models.py`
- `/workspace/experiments/designer_3/diagnostics.py`

**Model IDs for cross-referencing:**
- Designer 3, Model 1: `D3M1_quadratic_nb_ar1`
- Designer 3, Model 2: `D3M2_changepoint_nb_ar1`
- Designer 3, Model 3: `D3M3_gp_nb`

---

**End of Document**

**Date**: 2025-10-29
**Designer**: Model Designer 3
**Status**: Specifications complete, ready for implementation
**Critical reminder**: Success = finding truth, not defending complexity
