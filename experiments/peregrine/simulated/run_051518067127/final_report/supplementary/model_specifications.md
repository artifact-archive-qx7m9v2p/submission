# Complete Model Specifications
## Mathematical Details for All Experiments

**Date**: October 30, 2025

---

## Experiment 1: Negative Binomial GLM with Quadratic Trend

### Likelihood

**Count data model**:
```
C_t ~ NegativeBinomial2(μ_t, φ)    for t = 1, ..., 40
```

Where:
- `C_t` = observed count at time t
- `μ_t` = expected count (mean parameter)
- `φ` = dispersion parameter (inverse concentration)

**NegativeBinomial2 parameterization** (PyMC convention):
- Mean: E[C_t] = μ_t
- Variance: Var[C_t] = μ_t + μ_t²/φ
- As φ → ∞, approaches Poisson (Var = μ)
- Small φ → high overdispersion

### Link Function (Log-Linear with Quadratic Trend)

```
log(μ_t) = β₀ + β₁·year_t + β₂·year_t²
```

**Equivalently** (exponential form):
```
μ_t = exp(β₀ + β₁·year_t + β₂·year_t²)
```

Where:
- `year_t` = standardized time predictor (mean = 0, SD = 1)
- `β₀` = log-scale intercept (sets baseline level)
- `β₁` = linear growth coefficient (exponential rate)
- `β₂` = quadratic coefficient (acceleration/deceleration)

**Interpretation**:
- `exp(β₁)` = multiplicative effect per unit time
- `β₁ > 0` → exponential growth
- `β₂ > 0` → accelerating growth
- `β₂ < 0` → decelerating growth

### Prior Distributions

**Parameter priors** (weakly informative):

```
β₀ ~ Normal(4.5, 1.0)
```
- Centered on log(mean count) ≈ log(109) ≈ 4.69
- SD = 1.0 allows wide range: exp(4.5 ± 2) ∈ [8, 665]

```
β₁ ~ Normal(0.9, 0.5)
```
- Centered on EDA log-linear slope ≈ 0.86
- SD = 0.5 allows growth rates exp(0.9 ± 1) ∈ [1.40×, 6.69×]

```
β₂ ~ Normal(0, 0.3)
```
- Centered on zero (no acceleration/deceleration)
- SD = 0.3 allows modest curvature
- 95% prior mass in [-0.6, 0.6]

```
φ ~ Gamma(2, 0.1)
```
- Mean = α/β = 2/0.1 = 20
- SD = sqrt(α)/β = sqrt(2)/0.1 ≈ 14
- Allows wide range of overdispersion
- Mode at (α-1)/β = 10 (moderate overdispersion)

### Parameter Space

**Dimension**: 4 parameters (β₀, β₁, β₂, φ)

**Constraints**:
- β₀, β₁, β₂ ∈ ℝ (unconstrained)
- φ > 0 (strictly positive)

### Posterior Distribution

**Target** (unnormalized):
```
p(β₀, β₁, β₂, φ | data) ∝
    [∏ᵢ NB2(Cᵢ | μᵢ, φ)] ×
    N(β₀ | 4.5, 1) ×
    N(β₁ | 0.9, 0.5) ×
    N(β₂ | 0, 0.3) ×
    Gamma(φ | 2, 0.1)
```

Where:
- μᵢ = exp(β₀ + β₁·yearᵢ + β₂·yearᵢ²)
- Product over i = 1, ..., 40 observations

**Inference**: NUTS sampler, 4 chains × 2000 iterations (1000 warmup)

---

## Experiment 2: AR(1) Log-Normal with Regime-Switching Variance

### Likelihood

**Log-normal model with time-varying variance**:
```
C_t ~ LogNormal(μ_t, σ²_regime[r_t])    for t = 1, ..., 40
```

**Equivalently**:
```
log(C_t) ~ Normal(μ_t, σ²_regime[r_t])
```

Where:
- `C_t` = observed count at time t
- `μ_t` = mean of log(C_t)
- `σ_regime[r_t]` = standard deviation for regime r_t
- `r_t` ∈ {1, 2, 3} = regime indicator at time t

**Regime structure** (pre-specified from EDA):
- Regime 1 (early): t ∈ {1, ..., 14}
- Regime 2 (middle): t ∈ {15, ..., 27}
- Regime 3 (late): t ∈ {28, ..., 40}

### Mean Structure with AR(1) Component

**Decomposition**:
```
μ_t = μ_trend_t + μ_AR_t
```

**Trend component** (polynomial):
```
μ_trend_t = α + β₁·year_t + β₂·year_t²
```

**Autoregressive component**:
```
μ_AR_t = φ·ε_{t-1}    for t ≥ 2
μ_AR_1 = 0           (by convention, or sampled from stationary distribution)
```

**Residual definition**:
```
ε_t = log(C_t) - μ_trend_t
    = log(C_t) - (α + β₁·year_t + β₂·year_t²)
```

**Combined form**:
```
μ_t = α + β₁·year_t + β₂·year_t² + φ·ε_{t-1}
```

**Recursive structure**:
```
log(C_t) = α + β₁·year_t + β₂·year_t² + φ·(log(C_{t-1}) - α - β₁·year_{t-1} - β₂·year²_{t-1}) + η_t

where η_t ~ Normal(0, σ²_regime[r_t])
```

### Initial Condition (Stationarity)

For t = 1, to ensure stationarity:
```
ε_1 ~ Normal(0, σ²_regime[1] / (1 - φ²))
```

This matches the stationary distribution of AR(1) process.

**Alternative**: Set ε_0 = 0 (deterministic initialization)

### Prior Distributions

**Trend parameters**:

```
α ~ Normal(4.3, 0.5)
```
- Centered on mean(log(C)) ≈ 4.33 from EDA
- Informative based on data

```
β₁ ~ Normal(0.86, 0.2)
```
- Centered on EDA log-linear slope
- More informative than Experiment 1

```
β₂ ~ Normal(0, 0.3)
```
- Same as Experiment 1 (no acceleration assumed)

**AR(1) coefficient**:

```
φ ~ 0.95 · Beta(20, 2)
```

Scaled beta distribution:
- Raw Beta(20, 2) has support [0, 1]
- Scaled to [0, 0.95] for stationarity
- Mean ≈ 0.95 × 20/22 ≈ 0.86
- Highly concentrated near 1 (strong autocorrelation)
- Enforces stationarity constraint |φ| < 1

**Regime-specific variances**:

```
σ_regime[k] ~ HalfNormal(0, 1)    for k ∈ {1, 2, 3}
```

- Independent priors for each regime
- Mean = sqrt(2/π) ≈ 0.8
- Allows wide range [0, ∞)
- Weakly informative

### Parameter Space

**Dimension**: 7 parameters (α, β₁, β₂, φ, σ₁, σ₂, σ₃)

**Constraints**:
- α, β₁, β₂ ∈ ℝ (unconstrained)
- φ ∈ [0, 0.95) (stationarity enforced by prior)
- σ₁, σ₂, σ₃ > 0 (strictly positive)

### Posterior Distribution

**Target** (unnormalized):
```
p(α, β₁, β₂, φ, σ₁, σ₂, σ₃ | data) ∝
    [∏ᵢ N(log(Cᵢ) | μᵢ, σ²_regime[rᵢ])] ×
    N(α | 4.3, 0.5) ×
    N(β₁ | 0.86, 0.2) ×
    N(β₂ | 0, 0.3) ×
    [Beta(φ/0.95 | 20, 2) / 0.95] ×
    [∏ₖ HN(σₖ | 0, 1)]
```

Where:
- μᵢ = α + β₁·yearᵢ + β₂·yearᵢ² + φ·εᵢ₋₁
- εᵢ = log(Cᵢ) - (α + β₁·yearᵢ + β₂·yearᵢ²)
- ε₁ ~ N(0, σ₁² / (1 - φ²))

**Inference**: NUTS sampler, 4 chains × 2000 iterations (1000 warmup)

---

## Comparison: Key Differences

| Feature | Experiment 1 | Experiment 2 |
|---------|--------------|--------------|
| **Likelihood** | Negative Binomial | Log-Normal |
| **Scale** | Original count scale | Log scale |
| **Temporal structure** | None (independence) | AR(1) |
| **Variance structure** | Single φ | Three σ_regime |
| **Parameters** | 4 | 7 |
| **Complexity** | Low | Medium |
| **Stationarity** | N/A | Enforced via prior |
| **Regime switching** | No | Yes (variance) |

---

## Recommended Future Model: Experiment 3 (AR(2))

### Proposed Specification

**Likelihood** (same as Experiment 2):
```
log(C_t) ~ Normal(μ_t, σ²_regime[r_t])
```

**Mean structure with AR(2)**:
```
μ_t = α + β₁·year_t + β₂·year_t² + φ₁·ε_{t-1} + φ₂·ε_{t-2}
```

**Stationarity constraints** (AR(2) stability region):
```
φ₁ + φ₂ < 1
φ₂ - φ₁ < 1
|φ₂| < 1
```

**Priors for AR coefficients**:
```
φ₁ ~ 0.95 · Beta(20, 2)         # Same as Experiment 2
φ₂ ~ 0.5 · Beta(8, 12) - 0.25   # Centered near 0, allows negative values
```

Where:
- φ₁ prior concentrated near 0.85 (based on Experiment 2 result)
- φ₂ prior allows range [-0.25, 0.25] with mode near 0
- Joint prior should respect stationarity constraints (may require rejection sampling or constrained parameterization)

**Initial conditions**:
```
[ε_1]   ~ MVN([0], Σ_stationary)
[ε_2]       [0]

where Σ_stationary is the stationary covariance matrix for AR(2)
```

**Expected outcome**: Residual ACF < 0.3, ΔELPD ≈ 20-50 vs AR(1)

---

## General Notation

**Data**:
- N = 40 observations
- C = (C₁, ..., C₄₀) observed counts
- year = (year₁, ..., year₄₀) standardized time predictor

**Derived quantities**:
- μ = (μ₁, ..., μ₄₀) expected values
- ε = (ε₁, ..., ε₄₀) residuals (AR models)

**Diagnostics**:
- ACF(ε, lag k) = autocorrelation of residuals at lag k
- R-hat = Gelman-Rubin convergence diagnostic
- ESS = Effective sample size
- Pareto-k = LOO-CV reliability diagnostic

---

## Implementation Notes

### PyMC Code Structure (Experiment 2)

```python
import pymc as pm

with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=4.3, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.86, sigma=0.2)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=0.3)
    phi_raw = pm.Beta('phi_raw', alpha=20, beta=2)
    phi = pm.Deterministic('phi', 0.95 * phi_raw)
    sigma_regime = pm.HalfNormal('sigma_regime', sigma=1, shape=3)

    # Trend
    mu_trend = alpha + beta_1 * year + beta_2 * year_sq

    # AR(1) component
    epsilon = pm.MutableData('epsilon', np.zeros(N))
    for t in range(1, N):
        epsilon[t] = pm.Deterministic(f'eps_{t}',
            pm.math.log(C[t-1]) - (alpha + beta_1*year[t-1] + beta_2*year_sq[t-1]))

    mu_ar = phi * epsilon[:-1]  # Shift by 1 for AR term

    # Combined mean
    mu = mu_trend + pm.math.concatenate([[0], mu_ar])

    # Likelihood
    sigma_t = sigma_regime[regime_idx]  # regime_idx pre-specified
    obs = pm.Normal('obs', mu=mu, sigma=sigma_t, observed=pm.math.log(C))

    # Sampling
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)
```

**Note**: Actual implementation more complex due to:
- Stationary initialization for ε₁
- Careful indexing for time series
- Log-likelihood storage for LOO-CV
- Posterior predictive sampling

### Stan Alternative (Experiment 1)

```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
  vector[N] year_sq;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> phi;
}

model {
  vector[N] log_mu = beta_0 + beta_1 * year + beta_2 * year_sq;

  // Priors
  beta_0 ~ normal(4.5, 1.0);
  beta_1 ~ normal(0.9, 0.5);
  beta_2 ~ normal(0, 0.3);
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_log_lpmf(C[n] | beta_0 + beta_1*year[n] + beta_2*year_sq[n], phi);
  }
}
```

---

## References

**Negative Binomial parameterizations**:
- NegativeBinomial2 (mean-dispersion): Cameron & Trivedi (2013)
- PyMC documentation: https://www.pymc.io/

**AR models**:
- Stationarity conditions: Box, Jenkins, Reinsel (2015)
- Bayesian time series: Petris, Petrone, Campagnoli (2009)

**Model comparison**:
- LOO-CV: Vehtari, Gelman, Gabry (2017)
- ArviZ: Kumar et al. (2019)

---

**Document version**: 1.0
**Last updated**: October 30, 2025
**Corresponds to**: Main report `/workspace/final_report/report.md`
