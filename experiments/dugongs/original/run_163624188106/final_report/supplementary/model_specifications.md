# Supplementary Material A: Complete Model Specifications

**Report:** Bayesian Power Law Modeling of Y-x Relationship
**Date:** October 27, 2025

---

## Model 1: Log-Log Linear Power Law (ACCEPTED)

### Mathematical Specification

**Likelihood:**
```
log(Y_i) ~ Normal(μ_i, σ)
μ_i = α + β × log(x_i)
```

**Equivalent form:**
```
Y_i = exp(α) × x_i^β × exp(ε_i)
where ε_i ~ Normal(0, σ²)
```

**Priors:**
```
α ~ Normal(0.6, 0.3)      # Log-scale intercept
β ~ Normal(0.13, 0.1)     # Power law exponent
σ ~ HalfNormal(0.1)       # Log-scale residual standard deviation
```

### PyMC Implementation

```python
import pymc as pm
import numpy as np

# Prepare data
log_Y = np.log(Y_observed)
log_x = np.log(x_observed)

# Model specification
with pm.Model() as model1:
    # Priors
    alpha = pm.Normal('alpha', mu=0.6, sigma=0.3)
    beta = pm.Normal('beta', mu=0.13, sigma=0.1)
    sigma = pm.HalfNormal('sigma', sigma=0.1)

    # Linear predictor in log-log space
    mu = alpha + beta * log_x

    # Likelihood
    log_Y_obs = pm.Normal('log_Y_obs', mu=mu, sigma=sigma,
                         observed=log_Y)

    # Posterior predictive (for model checking)
    log_Y_pred = pm.Normal('log_Y_pred', mu=mu, sigma=sigma)

    # Sample from posterior
    idata = pm.sample(1000, tune=1000, chains=4,
                     random_seed=12345,
                     idata_kwargs={'log_likelihood': True})
```

### Stan Implementation

```stan
data {
  int<lower=1> N;                   // Number of observations
  vector[N] log_Y;                  // Log-transformed response
  vector[N] log_x;                  // Log-transformed predictor
}

parameters {
  real alpha;                       // Log-intercept
  real beta;                        // Power law exponent
  real<lower=0> sigma;              // Log-scale residual SD
}

model {
  // Priors
  alpha ~ normal(0.6, 0.3);
  beta ~ normal(0.13, 0.1);
  sigma ~ normal(0, 0.1);           // Half-normal via truncation

  // Likelihood
  log_Y ~ normal(alpha + beta * log_x, sigma);
}

generated quantities {
  // Posterior predictive samples
  vector[N] log_Y_pred;

  // Log-likelihood for LOO-CV
  vector[N] log_lik;

  for (i in 1:N) {
    log_Y_pred[i] = normal_rng(alpha + beta * log_x[i], sigma);
    log_lik[i] = normal_lpdf(log_Y[i] | alpha + beta * log_x[i], sigma);
  }
}
```

### Parameter Interpretation

**α (alpha):** Log-scale intercept
- Interpretation: When x = 1, log(Y) = α, so Y = exp(α)
- Posterior: α = 0.580 ± 0.019
- Back-transform: Y(x=1) = exp(0.580) ≈ 1.79

**β (beta):** Power law scaling exponent
- Interpretation: Y scales as x^β
- Posterior: β = 0.126 ± 0.009
- Meaning: Doubling x increases Y by 2^0.126 ≈ 1.088 (8.8%)
- Range: 95% HDI [0.111, 0.143] indicates weak positive scaling

**σ (sigma):** Residual standard deviation in log-scale
- Interpretation: Typical log-scale deviation after accounting for x
- Posterior: σ = 0.041 ± 0.006
- In original scale: Corresponds to ~4% coefficient of variation

### Prior Justification

**α ~ Normal(0.6, 0.3):**
- Centered on EDA finding: mean(log(Y)) ≈ 0.6
- SD = 0.3 allows 95% prior mass on [0.0, 1.2]
- Back-transforms to Y ∈ [1.0, 3.3], encompassing observed [1.77, 2.72]
- Weakly informative: Data dominance evident (posterior SD = 0.019 << prior SD = 0.3)

**β ~ Normal(0.13, 0.1):**
- Centered on EDA power law fit: β ≈ 0.126
- SD = 0.1 allows 95% prior mass on [0.0, 0.33]
- Constrains to positive relationship and diminishing returns (β < 1)
- Prior allows wide range while regularizing against extreme values
- Data dominance: Posterior SD = 0.009 << prior SD = 0.1

**σ ~ HalfNormal(0.1):**
- Centered on residual scale from EDA: σ ≈ 0.05
- Allows values up to ~0.2 (95th percentile at 0.165)
- Weakly regularizes to guard against overfitting
- Mode at σ = 0.07 (reasonable default)
- Data dominance: Posterior concentrates at 0.041

### Transformation Details

**Data Transformation:**
```
Y_original ∈ [1.77, 2.72]
log(Y) ∈ [0.571, 1.001]

x_original ∈ [1.0, 31.5]
log(x) ∈ [0.0, 3.450]
```

**Back-Transformation for Predictions:**

For point prediction at x_new:
```python
log_y_pred = alpha + beta * log(x_new)
y_median = exp(log_y_pred)           # Median prediction
y_mean = exp(log_y_pred + sigma^2/2) # Mean prediction (Jensen correction)
```

For predictive distribution:
```python
log_y_samples = alpha_samples + beta_samples * log(x_new)
epsilon_samples = np.random.normal(0, sigma_samples)
y_samples = np.exp(log_y_samples + epsilon_samples)
y_pred_ci = np.percentile(y_samples, [2.5, 97.5])
```

---

## Model 2: Log-Linear Heteroscedastic (REJECTED)

### Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(μ_i, σ_i)
μ_i = β₀ + β₁ × log(x_i)
log(σ_i) = γ₀ + γ₁ × x_i
```

**Priors:**
```
β₀ ~ Normal(1.8, 0.5)      # Intercept
β₁ ~ Normal(0.3, 0.2)      # Log-slope
γ₀ ~ Normal(-2, 1)         # Log-variance intercept
γ₁ ~ Normal(-0.05, 0.05)   # Log-variance slope
```

### PyMC Implementation

```python
import pymc as pm
import numpy as np
import pytensor.tensor as pt

# Prepare data
log_x = np.log(x_observed)

# Model specification
with pm.Model() as model2:
    # Priors for mean function
    beta_0 = pm.Normal('beta_0', mu=1.8, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.3, sigma=0.2)

    # Priors for variance function
    gamma_0 = pm.Normal('gamma_0', mu=-2, sigma=1)
    gamma_1 = pm.Normal('gamma_1', mu=-0.05, sigma=0.05)

    # Mean function (log-linear)
    mu = beta_0 + beta_1 * log_x

    # Variance function (exponential)
    log_sigma = gamma_0 + gamma_1 * x_observed
    sigma = pt.exp(log_sigma)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma,
                     observed=Y_observed)

    # Sample from posterior
    idata = pm.sample(1500, tune=1500, chains=4,
                     target_accept=0.97,
                     random_seed=12345,
                     idata_kwargs={'log_likelihood': True})
```

### Why Tested

EDA showed 7.5× variance decrease from low-x to high-x in original scale (Levene's test p = 0.003). Model 2 tests whether this heteroscedasticity persists in a log-linear framework.

### Why Rejected

**Critical Finding:** γ₁ = 0.003 ± 0.017, with 95% CI [-0.028, 0.039] including zero.

**Interpretation:** No evidence that variance changes with x. P(γ₁ < 0) = 43.9% indicates no directional preference.

**LOO Comparison:** ELPD = 23.56 vs Model 1's 46.99 (ΔELPD = -23.43, >5 SE worse).

**Conclusion:** Added complexity (4 vs 3 parameters) degrades predictions. Hypothesis not supported.

---

## Model Comparison Summary

| Aspect | Model 1 | Model 2 | Winner |
|--------|---------|---------|--------|
| **Parameters** | 3 | 4 | Model 1 (simpler) |
| **ELPD LOO** | 46.99 ± 3.11 | 23.56 ± 3.15 | Model 1 (+23.43) |
| **Pareto k issues** | 0/27 (0%) | 1/27 (3.7%) | Model 1 |
| **p_loo** | 2.43 | 3.41 | Model 1 |
| **R² (log scale)** | 0.902 | - | Model 1 |
| **MAPE** | 3.04% | - | Model 1 |
| **Runtime** | ~5 sec | ~110 sec | Model 1 (22× faster) |
| **Key hypothesis** | β ≠ 0 ✓ | γ₁ ≠ 0 ✗ | Model 1 (supported) |

**Decision:** Model 1 superior on all criteria. Model 2 rejected.

---

## Computational Details

### Software Versions
- **PyMC:** 5.26.1
- **ArviZ:** 0.22.0
- **NumPy:** 1.26.4
- **Python:** 3.13.0
- **Platform:** Linux 6.14.0-33-generic

### Sampling Configuration

**Model 1:**
- Chains: 4
- Warmup: 1,000 iterations per chain
- Sampling: 1,000 iterations per chain
- Total draws: 4,000
- target_accept: 0.80 (default)
- Random seed: 12345
- Divergences: 0
- Runtime: ~5 seconds

**Model 2:**
- Chains: 4
- Warmup: 1,500 iterations per chain
- Sampling: 1,500 iterations per chain
- Total draws: 6,000
- target_accept: 0.97 (conservative due to SBC warnings)
- Random seed: 12345
- Divergences: 0
- Runtime: ~110 seconds

### Hardware
- **Processor:** [System specific]
- **Memory:** [System specific]
- **Parallelization:** 4 chains run in parallel

### Reproducibility

**Random Seeds:**
- All analyses: 12345
- Prior predictive: 12345
- SBC: 12345 + simulation index
- MCMC: 12345
- Posterior predictive: 12345

**Data:**
- Location: `/workspace/data/data.csv`
- N = 27 observations
- No preprocessing beyond log transformation

**InferenceData Objects:**
- Model 1: `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Model 2: `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`
- Contains: Posterior samples, log-likelihood, sample stats

**Code:**
- Model 1: `/workspace/experiments/experiment_1/posterior_inference/code/fit_model_pymc.py`
- Model 2: `/workspace/experiments/experiment_2/posterior_inference/code/fit_model.py`

---

## Alternative Model Specifications (Not Tested)

### Model 3: Robust Log-Log (Student-t)

**Specification:**
```
log(Y_i) ~ Student_t(ν, μ_i, σ)
μ_i = α + β × log(x_i)

Priors:
  α ~ Normal(0.6, 0.3)
  β ~ Normal(0.13, 0.1)
  σ ~ HalfNormal(0.1)
  ν ~ Gamma(2, 0.1)  # Degrees of freedom
```

**Why not tested:**
- Model 1 has only 2 mild outliers (7.4% vs expected 5%)
- All Pareto k < 0.5 (no influential observations)
- Robust alternative unnecessary given excellent Model 1 diagnostics

### Model 4: Quadratic Heteroscedastic

**Specification:**
```
Y_i ~ Normal(μ_i, σ_i)
μ_i = β₀ + β₁ × x_i + β₂ × x_i²
log(σ_i) = γ₀ + γ₁ × x_i
```

**Why not tested:**
- EDA showed R² = 0.874 for quadratic (worse than log-log's 0.903)
- Model 2 already showed no heteroscedasticity
- 5 parameters too complex for n=27
- Log-log form has stronger theoretical justification

---

## References

**Probabilistic Programming:**
- Abril-Pla, O., et al. (2023). "PyMC: A Modern and Comprehensive Probabilistic Programming Framework in Python." *PeerJ Computer Science*, 9:e1516.

**Hamiltonian Monte Carlo:**
- Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *JMLR*, 15, 1593-1623.

---

**Document Status:** SUPPLEMENTARY MATERIAL A
**Version:** 1.0
**Date:** October 27, 2025
