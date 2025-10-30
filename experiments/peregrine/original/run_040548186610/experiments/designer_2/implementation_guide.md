# Implementation Guide: Designer 2 Models

**Quick start guide for implementing the three flexible Bayesian models**

---

## Pre-requisites

### Required Packages
```python
# Python environment
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from cmdstanpy import CmdStanModel
from patsy import dmatrix
import matplotlib.pyplot as plt
```

### Data Loading
```python
# Load data
data = pd.read_csv('/workspace/data/data.csv')
year = data['year'].values
C = data['C'].values
N = len(year)
```

---

## Model 1: GP-NegBin Implementation

### Stan Code Template
File: `/workspace/experiments/designer_2/model1_gp_negbin.stan`

```stan
data {
  int<lower=1> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

transformed data {
  real delta = 1e-9;
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> alpha_sq;
  real<lower=0> rho;
  real<lower=0> phi;
  vector[N] eta;
}

transformed parameters {
  vector[N] f;
  vector[N] mu;

  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(year, alpha_sq, rho);

    for (n in 1:N) {
      K[n, n] = K[n, n] + delta;
    }

    L_K = cholesky_decompose(K);
    f = beta_0 + beta_1 * year + L_K * eta;
  }

  mu = exp(f);
}

model {
  beta_0 ~ normal(log(100), 1);
  beta_1 ~ normal(0.8, 0.5);
  alpha_sq ~ normal(0, 1);
  rho ~ inv_gamma(5, 5);
  phi ~ normal(0, 10);
  eta ~ std_normal();

  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_lpmf(C[n] | mu[n], phi);
    C_rep[n] = neg_binomial_2_rng(mu[n], phi);
  }
}
```

### Python Fitting Code
```python
from cmdstanpy import CmdStanModel

# Prepare data
stan_data = {
    'N': N,
    'year': year,
    'C': C.astype(int)
}

# Compile and fit
model1 = CmdStanModel(stan_file='model1_gp_negbin.stan')
fit1 = model1.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.90,
    max_treedepth=12,
    show_progress=True
)

# Diagnostics
print(fit1.diagnose())
print(fit1.summary())

# Convert to ArviZ for analysis
idata1 = az.from_cmdstanpy(fit1)

# LOO-CV
loo1 = az.loo(idata1, pointwise=True)
print(f"Model 1 LOO-IC: {loo1.loo}")

# Posterior predictive checks
az.plot_ppc(idata1, num_pp_samples=100)
plt.savefig('model1_ppc.png', dpi=300)
```

### Key Diagnostics to Check
```python
# 1. Convergence
print("R-hat values:")
print(fit1.summary().loc[:, 'R_hat'].describe())

# 2. Effective Sample Size
print("\nESS values:")
print(fit1.summary().loc[:, 'N_Eff'].describe())

# 3. Divergences
print(f"\nDivergences: {fit1.diagnose()}")

# 4. Lengthscale interpretation
rho_samples = fit1.stan_variable('rho')
print(f"\nLengthscale posterior: mean={rho_samples.mean():.3f}, sd={rho_samples.std():.3f}")
print(f"  → Function smoothness: {'very smooth' if rho_samples.mean() > 2 else 'moderately smooth'}")

# 5. Prior-posterior comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
az.plot_dist(rho_samples, ax=axes[0,0], label='Posterior')
axes[0,0].set_title('Lengthscale (rho)')
plt.savefig('model1_diagnostics.png', dpi=300)
```

---

## Model 2: P-splines Implementation

### Step 1: Compute B-spline Basis
```python
from patsy import dmatrix

# Create B-spline basis with 8 interior knots
knots = np.quantile(year, np.linspace(0, 1, 10))  # 8 interior + 2 boundary
B_matrix = dmatrix(
    "bs(year, knots=knots[1:-1], degree=3, include_intercept=True)",
    {"year": year, "knots": knots},
    return_type="dataframe"
)

B = B_matrix.values
D = B.shape[1]  # Number of basis functions

print(f"B-spline basis: {N} observations × {D} basis functions")
```

### Step 2: Stan Code
File: `/workspace/experiments/designer_2/model2_pspline_negbin.stan`

```stan
data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] int<lower=0> C;
  matrix[N, D] B;
}

parameters {
  vector[D] theta;
  real<lower=0> tau;
  real<lower=0> phi;
}

transformed parameters {
  vector[N] eta;
  vector[N] mu;

  eta = B * theta;
  mu = exp(eta);
}

model {
  // Priors
  theta[1] ~ normal(0, 5);
  theta[2] ~ normal(0, 5);

  for (j in 3:D) {
    theta[j] - theta[j-1] ~ normal(0, tau);
  }

  tau ~ normal(0, 1);
  phi ~ normal(0, 10);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_lpmf(C[n] | mu[n], phi);
    C_rep[n] = neg_binomial_2_rng(mu[n], phi);
  }
}
```

### Step 3: Fitting Code
```python
# Prepare data
stan_data = {
    'N': N,
    'D': D,
    'C': C.astype(int),
    'B': B
}

# Compile and fit
model2 = CmdStanModel(stan_file='model2_pspline_negbin.stan')
fit2 = model2.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.85,
    show_progress=True
)

# Diagnostics
print(fit2.diagnose())
idata2 = az.from_cmdstanpy(fit2)
loo2 = az.loo(idata2, pointwise=True)
print(f"Model 2 LOO-IC: {loo2.loo}")

# Smoothness parameter
tau_samples = fit2.stan_variable('tau')
print(f"\nSmoothness tau: mean={tau_samples.mean():.3f}")
print(f"  → Interpretation: {'over-smoothed' if tau_samples.mean() < 0.2 else 'reasonable' if tau_samples.mean() < 2 else 'under-smoothed'}")
```

### Step 4: Visualization
```python
# Plot fitted curve
mu_samples = fit2.stan_variable('mu')
mu_mean = mu_samples.mean(axis=0)
mu_lower = np.percentile(mu_samples, 2.5, axis=0)
mu_upper = np.percentile(mu_samples, 97.5, axis=0)

plt.figure(figsize=(10, 6))
plt.scatter(year, C, alpha=0.6, label='Observed')
plt.plot(year, mu_mean, 'r-', linewidth=2, label='Fitted mean')
plt.fill_between(year, mu_lower, mu_upper, alpha=0.3, label='95% CI')
plt.xlabel('Year (standardized)')
plt.ylabel('Count')
plt.title('Model 2: P-spline Fit')
plt.legend()
plt.savefig('model2_fit.png', dpi=300)
```

---

## Model 3: Semi-parametric (PyMC)

### Full PyMC Implementation
File: `/workspace/experiments/designer_2/model3_semiparametric.py`

```python
import pymc as pm
import numpy as np

# Data
year_data = year
C_data = C

with pm.Model() as semi_parametric:
    # Data containers
    year_input = pm.Data("year", year_data)
    C_obs = pm.Data("C", C_data)

    # Parametric component: Logistic growth
    log_L = pm.Normal("log_L", mu=np.log(300), sigma=0.5)
    L = pm.Deterministic("L", pm.math.exp(log_L))
    k = pm.Lognormal("k", mu=0, sigma=1)
    t0 = pm.Normal("t0", mu=0, sigma=1)

    logistic = pm.Deterministic(
        "logistic",
        pm.math.log(L / (1 + pm.math.exp(-k * (year_input - t0))))
    )

    # Non-parametric component: GP deviations
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=0.5)
    ell_delta = pm.InverseGamma("ell_delta", alpha=3, beta=3)

    cov_func = sigma_delta**2 * pm.gp.cov.Matern32(1, ls=ell_delta)
    gp = pm.gp.Latent(cov_func=cov_func)
    delta = gp.prior("delta", X=year_input[:, None])

    # Combined mean
    eta = logistic + delta
    mu = pm.Deterministic("mu", pm.math.exp(eta))

    # Time-varying overdispersion
    phi_0 = pm.HalfNormal("phi_0", sigma=10)
    phi_1 = pm.Normal("phi_1", mu=0, sigma=1)
    log_phi = phi_0 + phi_1 * year_input
    phi = pm.Deterministic("phi", pm.math.exp(log_phi))

    # Likelihood (NegBinomial parameterization)
    # PyMC uses mu, alpha where alpha = mu^2 / (var - mu)
    alpha = mu**2 / (pm.math.maximum(phi, 1e-6) - mu)
    obs = pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=C_obs)

    # Sample
    trace3 = pm.sample(
        2000,
        tune=1000,
        chains=4,
        target_accept=0.95,
        return_inferencedata=True
    )

# Diagnostics
print(az.summary(trace3))
loo3 = az.loo(trace3, pointwise=True)
print(f"Model 3 LOO-IC: {loo3.loo}")

# Decomposition analysis
logistic_samples = trace3.posterior['logistic'].values
delta_samples = trace3.posterior['delta'].values

print(f"\nDecomposition:")
print(f"  Logistic component variance: {logistic_samples.var():.2f}")
print(f"  GP deviation variance: {delta_samples.var():.2f}")
print(f"  Deviation fraction: {delta_samples.var() / (logistic_samples.var() + delta_samples.var()):.2%}")
```

---

## Model Comparison

### LOO-CV Comparison
```python
import arviz as az

# Compute LOO for all models
loo1 = az.loo(idata1, pointwise=True)
loo2 = az.loo(idata2, pointwise=True)
loo3 = az.loo(trace3, pointwise=True)

# Compare
comparison = az.compare({
    'GP-NegBin': idata1,
    'P-spline': idata2,
    'Semi-parametric': trace3
})

print("\n=== MODEL COMPARISON ===")
print(comparison)
print(f"\nBest model: {comparison.index[0]}")
print(f"LOO-IC difference: {comparison.loc[comparison.index[1], 'loo'] - comparison.loc[comparison.index[0], 'loo']:.2f}")
```

### Holdout Prediction Test
```python
# Hold out last 10 observations
train_idx = np.arange(N-10)
test_idx = np.arange(N-10, N)

# Refit models on training data
# (repeat fitting code with year[train_idx], C[train_idx])

# Predict on test data
# Compare RMSE across models
rmse_scores = {}
for model_name, predictions in [('GP', pred1), ('Spline', pred2), ('Semi', pred3)]:
    rmse = np.sqrt(((predictions - C[test_idx])**2).mean())
    rmse_scores[model_name] = rmse
    print(f"{model_name} RMSE: {rmse:.2f}")

print(f"\nBest holdout performance: {min(rmse_scores, key=rmse_scores.get)}")
```

### Posterior Predictive Checks
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (idata, title) in enumerate([(idata1, 'GP-NegBin'),
                                       (idata2, 'P-spline'),
                                       (trace3, 'Semi-parametric')]):
    az.plot_ppc(idata, num_pp_samples=100, ax=axes[idx])
    axes[idx].set_title(title)

plt.tight_layout()
plt.savefig('all_models_ppc.png', dpi=300)

# Check overdispersion captured
for idata, name in [(idata1, 'GP'), (idata2, 'Spline'), (trace3, 'Semi')]:
    C_rep = idata.posterior_predictive['C_rep'].values.reshape(-1, N)
    pred_var = C_rep.var(axis=0).mean()
    pred_mean = C_rep.mean(axis=0).mean()
    print(f"{name} - Predicted Var/Mean: {pred_var/pred_mean:.2f} (Observed: 68.0)")
```

---

## Decision Rules Implementation

```python
def evaluate_models(loo1, loo2, loo3, fit1, fit2, trace3):
    """
    Apply decision rules from experiment plan
    """
    decisions = {}

    # Rule 1: Check convergence
    rhat_ok = all([
        fit1.summary()['R_hat'].max() < 1.01,
        fit2.summary()['R_hat'].max() < 1.01,
        np.max(az.rhat(trace3).to_array()) < 1.01
    ])
    decisions['convergence'] = rhat_ok

    # Rule 2: Compare LOO-IC
    loo_scores = {'GP': loo1.loo, 'Spline': loo2.loo, 'Semi': loo3.loo}
    best_model = min(loo_scores, key=loo_scores.get)
    loo_diff = sorted(loo_scores.values())[1] - sorted(loo_scores.values())[0]

    decisions['best_model'] = best_model
    decisions['loo_difference'] = loo_diff

    # Rule 3: Check if flexibility is collapsing
    rho_mean = fit1.stan_variable('rho').mean()
    tau_mean = fit2.stan_variable('tau').mean()

    decisions['gp_collapsed'] = rho_mean > 10  # Lengthscale too long
    decisions['spline_oversmoothed'] = tau_mean < 0.1
    decisions['spline_undersmoothed'] = tau_mean > 5

    # Rule 4: Recommendation
    if not rhat_ok:
        decisions['recommendation'] = "FAIL: Models did not converge"
    elif loo_diff < 5:
        decisions['recommendation'] = f"Use simplest model ({best_model}), marginal differences"
    elif loo_diff < 15:
        decisions['recommendation'] = f"Consider ensemble, moderate improvement for {best_model}"
    else:
        decisions['recommendation'] = f"Strong evidence for {best_model}"

    return decisions

# Run evaluation
results = evaluate_models(loo1, loo2, loo3, fit1, fit2, trace3)
print("\n=== DECISION RESULTS ===")
for key, value in results.items():
    print(f"{key}: {value}")
```

---

## Quick Checklist

Before accepting any model:

- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 for all parameters
- [ ] Divergences < 1% of samples
- [ ] LOO Pareto k < 0.7 for all observations
- [ ] Posterior predictive contains observed data
- [ ] Predicted Var/Mean ratio similar to observed (68)
- [ ] Parameter values scientifically plausible
- [ ] Predictions at year=±2 not absurd

If any fail → debug or simplify model

---

## File Organization

```
/workspace/experiments/designer_2/
├── proposed_models.md              # Full theoretical specifications
├── experiment_plan.md              # Strategy and decision rules
├── implementation_guide.md         # This file
├── model1_gp_negbin.stan          # Stan code (to create)
├── model2_pspline_negbin.stan     # Stan code (to create)
├── model3_semiparametric.py       # PyMC code (to create)
├── fit_all_models.py              # Orchestration script (to create)
├── results/                        # Output directory
│   ├── model1_diagnostics.png
│   ├── model2_fit.png
│   ├── all_models_ppc.png
│   └── comparison_table.csv
└── README.md                       # Quick summary
```

---

**Status**: Ready for implementation
**Estimated time**: 2-3 days for all 3 models
**Priority order**: Model 2 → Model 1 → Model 3
