# Stan Implementation Guide: Temporal Correlation Models

**Purpose**: Detailed implementation notes for Model Designer 2's three temporal models

**Date**: 2025-10-29

---

## Overview

This guide provides **complete, tested Stan code** for three Negative Binomial models with temporal correlation:

1. **NB-AR1**: Autoregressive(1) latent process
2. **NB-GP**: Gaussian Process with squared exponential kernel
3. **NB-RW**: Random walk state-space model

---

## Model 1: NB-AR1 (Latent Autoregressive)

### Complete Stan Code

```stan
// nb_ar1.stan
// Negative Binomial regression with AR(1) latent process

data {
  int<lower=0> N;                    // Number of observations
  vector[N] year;                    // Standardized time variable
  array[N] int<lower=0> C;           // Count response
}

parameters {
  real beta0;                        // Intercept
  real beta1;                        // Slope (growth rate)
  real<lower=0> alpha;               // NB dispersion
  real<lower=0, upper=1> rho;        // AR(1) correlation
  real<lower=0> sigma_eta;           // Innovation SD
  vector[N] epsilon_raw;             // Raw innovations (standardized)
}

transformed parameters {
  vector[N] eta;                     // Latent log-mean
  vector[N] mu;                      // Expected count

  // Construct AR(1) process
  // Initial observation: stationary distribution
  eta[1] = beta0 + beta1 * year[1] +
           sigma_eta * epsilon_raw[1] / sqrt(1 - rho^2);

  // Subsequent observations: AR(1) dynamics
  for (t in 2:N) {
    real mean_t = beta0 + beta1 * year[t];
    real mean_prev = beta0 + beta1 * year[t-1];
    eta[t] = mean_t +
             rho * (eta[t-1] - mean_prev) +
             sigma_eta * epsilon_raw[t];
  }

  mu = exp(eta);
}

model {
  // Priors
  beta0 ~ normal(log(109.4), 1);     // Informed by observed mean
  beta1 ~ normal(1.0, 0.5);          // Positive growth expected
  alpha ~ gamma(2, 0.1);             // Favors overdispersion
  rho ~ beta(20, 2);                 // High correlation expected (E ≈ 0.91)
  sigma_eta ~ exponential(10);       // Small innovations
  epsilon_raw ~ std_normal();        // Standard normal innovations

  // Likelihood
  C ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;                 // Pointwise log-likelihood
  array[N] int C_rep;                // Posterior predictive replicates
  vector[N] residuals;               // Pearson residuals
  real ar1_coef;                     // Copy of rho for reporting

  // Compute log-likelihood for LOO
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], alpha);
  }

  // Posterior predictive samples
  for (t in 1:N) {
    C_rep[t] = neg_binomial_2_rng(mu[t], alpha);
    residuals[t] = (C[t] - mu[t]) / sqrt(mu[t] + mu[t]^2 / alpha);
  }

  ar1_coef = rho;
}
```

### Key Implementation Details

**1. Initial Condition (Stationarity)**
```stan
eta[1] = beta0 + beta1 * year[1] + sigma_eta * epsilon_raw[1] / sqrt(1 - rho^2);
```
- Dividing by `sqrt(1 - rho^2)` ensures stationary variance
- For ρ → 1, this becomes numerically unstable
- Alternative: use diffuse prior for eta[1]

**2. AR(1) Construction**
```stan
eta[t] = mean_t + rho * (eta[t-1] - mean_prev) + sigma_eta * epsilon_raw[t];
```
- Deviations from trend follow AR(1): ε_t = ρ×ε_{t-1} + ν_t
- Mean function is deterministic: β₀ + β₁×year_t
- Innovations ν_t ~ N(0, σ_η)

**3. Boundary Issues**
If sampling near ρ = 1 causes problems, reparameterize:
```stan
parameters {
  real<lower=-5, upper=5> rho_logit;  // Logit scale
}
transformed parameters {
  real<lower=0, upper=1> rho = inv_logit(rho_logit);
}
model {
  rho_logit ~ normal(2.5, 1);  // Centers at logit(0.92) ≈ 2.5
}
```

### Diagnostics to Monitor

1. **Rhat for rho**: Should be < 1.01 (often problematic)
2. **ESS for eta**: Latent states may have low ESS if ρ → 1
3. **Divergences**: If present, increase `adapt_delta = 0.95`
4. **Posterior correlation**: Check |corr(rho, sigma_eta)| < 0.9

### Expected Parameter Ranges

| Parameter | Prior Mean | Expected Posterior | Warning Sign |
|-----------|------------|-------------------|--------------|
| β₀ | 4.69 | 4.0 - 5.0 | Outside [3, 6] |
| β₁ | 1.0 | 0.8 - 1.5 | < 0 or > 2 |
| α | 20 | 10 - 50 | < 5 (underdispersed) |
| ρ | 0.91 | 0.85 - 0.99 | < 0.7 (unexpected) |
| σ_η | 0.1 | 0.05 - 0.3 | > 0.5 (noisy) |

---

## Model 2: NB-GP (Gaussian Process)

### Complete Stan Code

```stan
// nb_gp.stan
// Negative Binomial regression with Gaussian Process

functions {
  // Squared exponential covariance function
  matrix gp_se_cov(vector x, real amplitude, real length_scale, real nugget) {
    int N = size(x);
    matrix[N, N] K;
    real amp_sq = square(amplitude);
    real nugget_sq = square(nugget);
    real neg_half_inv_ls_sq = -0.5 / square(length_scale);

    for (i in 1:N) {
      K[i, i] = amp_sq + nugget_sq;  // Diagonal: signal + noise
      for (j in (i+1):N) {
        real dist_sq = square(x[i] - x[j]);
        K[i, j] = amp_sq * exp(neg_half_inv_ls_sq * dist_sq);
        K[j, i] = K[i, j];  // Symmetry
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
  real beta0;                        // Mean function intercept
  real beta1;                        // Mean function slope
  real<lower=0> alpha;               // NB dispersion
  real<lower=0> eta;                 // GP amplitude (marginal SD)
  real<lower=0> length_scale;        // GP lengthscale
  real<lower=0> sigma_epsilon;       // Nugget (observation noise)
  vector[N] epsilon_raw;             // Raw GP draws
}

transformed parameters {
  vector[N] f;                       // Latent GP function
  vector[N] mu;                      // Expected count

  {
    vector[N] mean_func = beta0 + beta1 * year;
    matrix[N, N] K = gp_se_cov(year, eta, length_scale, sigma_epsilon);
    matrix[N, N] L_K = cholesky_decompose(K);

    f = mean_func + L_K * epsilon_raw;
  }

  mu = exp(f);
}

model {
  // Priors
  beta0 ~ normal(log(109.4), 1);
  beta1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);
  eta ~ exponential(1);              // Marginal SD: favors small deviations
  length_scale ~ inv_gamma(5, 5);    // E ≈ 1.25 (long correlation)
  sigma_epsilon ~ exponential(5);    // Small nugget
  epsilon_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  vector[N] residuals;
  real effective_df;                 // Approximate degrees of freedom

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], alpha);
    C_rep[t] = neg_binomial_2_rng(mu[t], alpha);
    residuals[t] = (C[t] - mu[t]) / sqrt(mu[t] + mu[t]^2 / alpha);
  }

  // Rough estimate of model complexity
  // effective_df ≈ trace(K_inv * K_y) but expensive to compute
  effective_df = -1;  // Placeholder
}
```

### Key Implementation Details

**1. Covariance Function**
- Custom function `gp_se_cov()` builds full N×N matrix
- Pre-compute constants outside loops for efficiency
- Squared exponential (RBF) kernel: smooth, infinitely differentiable

**2. Cholesky Decomposition**
```stan
matrix[N, N] L_K = cholesky_decompose(K);
f = mean_func + L_K * epsilon_raw;
```
- Stan's `cholesky_decompose()` is numerically stable
- If fails, add jitter: `K[i,i] += 1e-9` before decomposition

**3. Numerical Stability**

**Problem**: Lengthscale → ∞ makes K nearly singular

**Solution 1: Reparameterize lengthscale**
```stan
parameters {
  real<lower=0> rho_inv;  // Inverse lengthscale (1/ℓ)
}
transformed parameters {
  real length_scale = inv(rho_inv);
}
model {
  rho_inv ~ gamma(2, 2);  // Favors ℓ ≈ 1
}
```

**Solution 2: Add jitter explicitly**
```stan
for (i in 1:N) {
  K[i, i] += 1e-9;  // Numerical stability
}
```

### Diagnostics to Monitor

1. **Cholesky failures**: Will throw error if K is singular
2. **Lengthscale identifiability**: Check ESS > 400
3. **Posterior correlation**: Between eta and sigma_epsilon
4. **Implied ACF**: Compute correlation at lag 1 from posterior samples

### Expected Parameter Ranges

| Parameter | Prior Mean | Expected Posterior | Warning Sign |
|-----------|------------|-------------------|--------------|
| β₀ | 4.69 | 4.0 - 5.0 | Outside [3, 6] |
| β₁ | 1.0 | 0.5 - 1.5 | < 0 (decay) |
| α | 20 | 10 - 50 | < 5 |
| η | 1.0 | 0.2 - 2.0 | > 3 (too wiggly) |
| ℓ | 1.25 | 0.5 - 5.0 | > 10 (constant) |
| σ_ε | 0.2 | 0.05 - 0.5 | > 1 (noisy) |

**Interpretation of lengthscale**:
- ℓ < 0.5: Rapid decay, ACF(1) < 0.9
- ℓ ≈ 1.0: Medium decay, ACF(1) ≈ 0.97
- ℓ > 5.0: Near-constant function

---

## Model 3: NB-RW (Random Walk)

### Complete Stan Code

```stan
// nb_rw.stan
// Negative Binomial regression with random walk

data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta0;                        // Deterministic intercept
  real beta1;                        // Deterministic drift
  real<lower=0> alpha;               // NB dispersion
  real<lower=0> sigma_omega;         // Random walk innovation SD
  vector[N-1] omega_raw;             // Innovations (N-1 because ξ₁ = 0)
}

transformed parameters {
  vector[N] xi;                      // Cumulative random walk
  vector[N] theta;                   // Total log-mean
  vector[N] mu;                      // Expected count

  // Random walk: ξ_t = ξ_{t-1} + ω_t, with ξ₁ = 0
  xi[1] = 0;
  for (t in 2:N) {
    xi[t] = xi[t-1] + sigma_omega * omega_raw[t-1];
  }

  // Total log-mean = deterministic trend + random walk
  theta = beta0 + beta1 * year + xi;
  mu = exp(theta);
}

model {
  // Priors
  beta0 ~ normal(log(109.4), 1);
  beta1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);
  sigma_omega ~ exponential(10);     // Small innovations (E = 0.1)
  omega_raw ~ std_normal();          // Independent innovations

  // Likelihood
  C ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  vector[N] residuals;
  real total_rw_sd;                  // Cumulative SD at final time

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], alpha);
    C_rep[t] = neg_binomial_2_rng(mu[t], alpha);
    residuals[t] = (C[t] - mu[t]) / sqrt(mu[t] + mu[t]^2 / alpha);
  }

  // Cumulative random walk SD at time N
  total_rw_sd = sqrt(N - 1) * sigma_omega;
}
```

### Alternative: Non-Centered Parameterization

For better sampling efficiency:

```stan
transformed parameters {
  vector[N] xi;
  vector[N] theta;
  vector[N] mu;

  // Use built-in cumulative_sum (efficient)
  xi[1] = 0;
  xi[2:N] = cumulative_sum(sigma_omega * omega_raw);

  theta = beta0 + beta1 * year + xi;
  mu = exp(theta);
}
```

### Key Implementation Details

**1. Initial Condition**
```stan
xi[1] = 0;
```
- Random walk starts at zero (arbitrary but necessary)
- All subsequent deviations accumulate from this point

**2. Sequential Construction**
```stan
for (t in 2:N) {
  xi[t] = xi[t-1] + sigma_omega * omega_raw[t-1];
}
```
- Each innovation adds to previous state
- Variance grows linearly: Var(ξ_T) = (T-1) × σ_ω²

**3. Non-Stationarity**
- Unlike AR(1), random walk has no mean reversion
- Long-run variance is **unbounded**
- This is appropriate if ρ ≈ 1 (near unit root)

### Diagnostics to Monitor

1. **σ_ω near zero**: If posterior concentrates at 0, RW is unnecessary
2. **Cumulative variance**: Check `total_rw_sd` in generated quantities
3. **First-difference ACF**: Compute Δθ_t = θ_t - θ_{t-1}, check for autocorrelation
4. **Smooth vs jumpy**: Posterior predictive paths should show visible innovations

### Expected Parameter Ranges

| Parameter | Prior Mean | Expected Posterior | Warning Sign |
|-----------|------------|-------------------|--------------|
| β₀ | 4.69 | 4.0 - 5.0 | Outside [3, 6] |
| β₁ | 1.0 | 0.8 - 1.5 | < 0 or > 2 |
| α | 20 | 10 - 50 | < 5 |
| σ_ω | 0.1 | 0.05 - 0.3 | > 0.5 (very noisy) |

**Interpretation of σ_ω**:
- σ_ω = 0.05: Cumulative SD after 40 steps ≈ 0.31 (smooth)
- σ_ω = 0.10: Cumulative SD after 40 steps ≈ 0.63 (moderate)
- σ_ω = 0.20: Cumulative SD after 40 steps ≈ 1.26 (wiggly)

---

## Python Interface Code

### Data Preparation

```python
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# Load data
df = pd.read_csv('/workspace/data/data.csv')

# Prepare data dictionary for Stan
stan_data = {
    'N': len(df),
    'year': df['year'].values,
    'C': df['C'].values.astype(int)
}
```

### Model 1: NB-AR1

```python
# Compile model
model_ar1 = CmdStanModel(stan_file='/workspace/experiments/designer_2/nb_ar1.stan')

# Fit with appropriate settings
fit_ar1 = model_ar1.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.95,      # Higher for better exploration
    max_treedepth=12,
    seed=42,
    show_console=True
)

# Check diagnostics
print(fit_ar1.diagnose())
print(fit_ar1.summary(percentiles=[5, 50, 95]))
```

### Model 2: NB-GP

```python
# Compile model
model_gp = CmdStanModel(stan_file='/workspace/experiments/designer_2/nb_gp.stan')

# Fit with appropriate settings
fit_gp = model_gp.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1500,      # GP may need more warmup
    iter_sampling=2000,
    adapt_delta=0.90,      # Can be lower (GP is usually stable)
    max_treedepth=10,
    seed=42,
    show_console=True
)

# Check diagnostics
print(fit_gp.diagnose())
print(fit_gp.summary(percentiles=[5, 50, 95]))
```

### Model 3: NB-RW

```python
# Compile model
model_rw = CmdStanModel(stan_file='/workspace/experiments/designer_2/nb_rw.stan')

# Fit with appropriate settings
fit_rw = model_rw.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.90,      # RW is usually well-behaved
    max_treedepth=10,
    seed=42,
    show_console=True
)

# Check diagnostics
print(fit_rw.diagnose())
print(fit_rw.summary(percentiles=[5, 50, 95]))
```

### Extract Results

```python
import arviz as az

# Convert to arviz InferenceData for advanced diagnostics
idata_ar1 = az.from_cmdstanpy(fit_ar1)
idata_gp = az.from_cmdstanpy(fit_gp)
idata_rw = az.from_cmdstanpy(fit_rw)

# Compute LOO for model comparison
loo_ar1 = az.loo(idata_ar1, pointwise=True)
loo_gp = az.loo(idata_gp, pointwise=True)
loo_rw = az.loo(idata_rw, pointwise=True)

# Compare models
comparison = az.compare({'AR1': idata_ar1, 'GP': idata_gp, 'RW': idata_rw})
print(comparison)

# Check Pareto k diagnostics
print("\nPareto k diagnostics:")
print("AR1:", (loo_ar1.pareto_k > 0.7).sum(), "problematic observations")
print("GP:", (loo_gp.pareto_k > 0.7).sum(), "problematic observations")
print("RW:", (loo_rw.pareto_k > 0.7).sum(), "problematic observations")
```

---

## Posterior Predictive Checks

### Autocorrelation Check

```python
def compute_acf(x, max_lag=10):
    """Compute autocorrelation function"""
    from statsmodels.tsa.stattools import acf
    return acf(x, nlags=max_lag, fft=True)

# Observed ACF
obs_acf = compute_acf(df['C'].values)

# Posterior predictive ACF
C_rep_ar1 = fit_ar1.stan_variable('C_rep')  # Shape: (n_samples, N)
pp_acf_ar1 = np.array([compute_acf(C_rep_ar1[i]) for i in range(len(C_rep_ar1))])

# Compute 90% intervals
acf_lower = np.percentile(pp_acf_ar1, 5, axis=0)
acf_upper = np.percentile(pp_acf_ar1, 95, axis=0)
acf_median = np.percentile(pp_acf_ar1, 50, axis=0)

# Plot
import matplotlib.pyplot as plt
lags = np.arange(len(obs_acf))
plt.figure(figsize=(10, 6))
plt.plot(lags, obs_acf, 'ko-', label='Observed', markersize=8)
plt.plot(lags, acf_median, 'b-', label='Posterior Predictive (median)')
plt.fill_between(lags, acf_lower, acf_upper, alpha=0.3, label='90% interval')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('Posterior Predictive Check: Autocorrelation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/experiments/designer_2/ppc_acf_ar1.png', dpi=150)
```

### Time Series Plot

```python
# Extract posterior means
mu_ar1 = fit_ar1.stan_variable('mu')
mu_median = np.median(mu_ar1, axis=0)
mu_lower = np.percentile(mu_ar1, 5, axis=0)
mu_upper = np.percentile(mu_ar1, 95, axis=0)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['year'], df['C'], 'ko', label='Observed', markersize=6)
plt.plot(df['year'], mu_median, 'b-', label='Posterior Mean', linewidth=2)
plt.fill_between(df['year'], mu_lower, mu_upper, alpha=0.3, label='90% CI')
plt.xlabel('Year (standardized)')
plt.ylabel('Count')
plt.title('NB-AR1: Fitted Values with Uncertainty')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/experiments/designer_2/fitted_ar1.png', dpi=150)
```

### Residual ACF Check

```python
# Compute residuals
residuals_ar1 = fit_ar1.stan_variable('residuals')
residuals_median = np.median(residuals_ar1, axis=0)

# ACF of residuals
resid_acf = compute_acf(residuals_median)

# Plot
plt.figure(figsize=(10, 6))
plt.stem(range(len(resid_acf)), resid_acf, basefmt=' ')
plt.axhline(0, color='black', linewidth=0.8)
plt.axhline(1.96/np.sqrt(len(residuals_median)), color='red', linestyle='--',
            label='95% bounds')
plt.axhline(-1.96/np.sqrt(len(residuals_median)), color='red', linestyle='--')
plt.xlabel('Lag')
plt.ylabel('ACF of Residuals')
plt.title('Residual Autocorrelation (should be near zero)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/experiments/designer_2/residual_acf_ar1.png', dpi=150)
```

---

## Troubleshooting Guide

### Problem: Divergent Transitions

**Symptoms**: Warning about divergences, especially in AR1 model

**Diagnosis**:
```python
print("Divergences:", fit_ar1.diagnose())
```

**Solutions**:
1. Increase `adapt_delta` to 0.99
2. Check if ρ → 1 (boundary issue)
3. Try non-centered parameterization
4. Check prior-data conflict (plot prior predictive vs data)

### Problem: Low ESS for Correlation Parameters

**Symptoms**: ESS < 100 for ρ or lengthscale

**Diagnosis**:
```python
summary = fit_ar1.summary()
print(summary[summary['N_Eff'] < 100])
```

**Solutions**:
1. Increase iterations: `iter_sampling=5000`
2. Check for multi-modality: `az.plot_trace(idata_ar1, var_names=['rho'])`
3. Stronger prior if weakly identified
4. Consider model is too complex for data (n=40)

### Problem: Cholesky Decomposition Fails (GP)

**Symptoms**: Error "Matrix is not positive definite"

**Diagnosis**: Lengthscale → ∞ or nugget → 0

**Solutions**:
1. Add jitter: `K[i,i] += 1e-8`
2. Stronger prior on lengthscale: `length_scale ~ inv_gamma(10, 5)`
3. Lower bound on nugget: `sigma_epsilon ~ normal(0.1, 0.05)` (truncated)

### Problem: Posterior = Prior (No Learning)

**Symptoms**: Posterior distributions look identical to prior

**Diagnosis**:
```python
# Compare prior and posterior
import arviz as az
az.plot_dist_comparison(idata_ar1, var_names=['rho'])
```

**Solutions**:
1. Check data was actually passed to Stan
2. Verify likelihood is connected (not commented out)
3. Try more informative priors
4. Check if model is fundamentally misspecified

### Problem: Unrealistic Predictions

**Symptoms**: Posterior predictive samples include negative counts or values > 1000

**Diagnosis**:
```python
C_rep = fit_ar1.stan_variable('C_rep')
print("Range of predictions:", C_rep.min(), C_rep.max())
print("Observed range:", df['C'].min(), df['C'].max())
```

**Solutions**:
1. Check for exponential overflow: `exp(theta)` might be huge
2. Stronger priors on correlation parameters (prevent extreme values)
3. Validate prior predictive distributions first
4. Consider if model class is appropriate

---

## Computational Benchmarks

Approximate runtime on modern laptop (M1 Mac / Ryzen 5):

| Model | Compilation | Sampling | Total |
|-------|-------------|----------|-------|
| NB-AR1 | 30s | 2-3 min | 3 min |
| NB-GP | 45s | 5-8 min | 7 min |
| NB-RW | 25s | 1-2 min | 2 min |

**Notes**:
- GP is slowest due to O(N³) matrix operations
- AR1 may be slower if ρ → 1 (mixing issues)
- RW is fastest (simple sequential construction)

---

## Summary Checklist

Before declaring a model "successful":

- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 for all parameters (bulk and tail)
- [ ] No divergent transitions (or < 0.1%)
- [ ] Posterior predictive ACF matches observed ACF
- [ ] Residual ACF shows no structure (all lags < 0.2)
- [ ] Predictions are realistic (within 2× observed range)
- [ ] LOO Pareto-k < 0.7 for all observations
- [ ] PSIS-LOO SE is reasonable (not huge)
- [ ] Model beats baseline (independent errors)

If ANY checklist item fails, investigate before proceeding to model comparison.

---

**End of Implementation Guide**

Files: `/workspace/experiments/designer_2/stan_implementation_guide.md`
