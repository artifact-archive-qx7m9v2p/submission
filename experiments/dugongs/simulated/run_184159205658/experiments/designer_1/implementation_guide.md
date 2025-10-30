# Implementation Guide: Designer 1 Models
## PyMC and Stan Code Templates

**Purpose:** Practical implementation details for all three proposed models
**Date:** 2025-10-27

---

## Setup and Data Preparation

### Load Dependencies

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# For Stan alternative
# import cmdstanpy
# Or: import pystan

# Set random seed for reproducibility
np.random.seed(42)
```

### Load and Prepare Data

```python
# Load data
data = pd.read_csv('/workspace/data/data.csv')
x_obs = data['x'].values
y_obs = data['Y'].values

# Summary statistics
print(f"N = {len(x_obs)}")
print(f"x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"Y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"Y mean: {y_obs.mean():.3f}, SD: {y_obs.std():.3f}")

# Create prediction grid for visualization
x_pred = np.linspace(0.5, 40, 200)
```

---

# Model 1: Logarithmic Regression

## PyMC Implementation

```python
import pymc as pm
import arviz as az

# Model specification
with pm.Model() as log_model:
    # Data
    x = pm.Data('x', x_obs)
    y = pm.Data('y', y_obs)

    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.73, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.28, sigma=0.15)
    sigma = pm.Exponential('sigma', lam=5.0)

    # Mean structure
    mu = beta_0 + beta_1 * pm.math.log(x)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    # Prior predictive check (optional but recommended)
    prior_pred = pm.sample_prior_predictive(samples=1000)

# Sample from posterior
with log_model:
    trace_log = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True
    )

    # Posterior predictive sampling
    post_pred_log = pm.sample_posterior_predictive(
        trace_log,
        random_seed=42
    )

# Add posterior predictive to trace
trace_log.extend(post_pred_log)
```

## Convergence Diagnostics

```python
# Summary statistics
print(az.summary(trace_log, var_names=['beta_0', 'beta_1', 'sigma']))

# Check R-hat and ESS
summary = az.summary(trace_log, var_names=['beta_0', 'beta_1', 'sigma'])
print("\nConvergence checks:")
print(f"Max R-hat: {summary['r_hat'].max():.4f}")
print(f"Min ESS bulk: {summary['ess_bulk'].min():.0f}")
print(f"Min ESS tail: {summary['ess_tail'].min():.0f}")

# Trace plots
az.plot_trace(trace_log, var_names=['beta_0', 'beta_1', 'sigma'])
plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/trace_plot_log.png', dpi=300)
plt.close()

# Rank plots (sensitive to non-convergence)
az.plot_rank(trace_log, var_names=['beta_0', 'beta_1', 'sigma'])
plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/rank_plot_log.png', dpi=300)
plt.close()

# Pairplot (parameter correlations)
az.plot_pair(trace_log, var_names=['beta_0', 'beta_1', 'sigma'],
             kind='kde', marginals=True, divergences=True)
plt.savefig('/workspace/experiments/designer_1/pairplot_log.png', dpi=300)
plt.close()
```

## Posterior Predictive Checks

```python
# Extract posterior predictive samples
y_pred = trace_log.posterior_predictive['Y_obs'].values.reshape(-1, len(y_obs))

# Visual PPC
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel 1: Posterior predictive distribution
axes[0].hist(y_pred.flatten(), bins=50, alpha=0.5, density=True, label='Posterior predictive')
axes[0].axvline(y_obs.mean(), color='red', linestyle='--', label='Observed mean')
axes[0].set_xlabel('Y')
axes[0].set_ylabel('Density')
axes[0].set_title('Posterior Predictive Distribution')
axes[0].legend()

# Panel 2: Observed vs predicted
y_pred_mean = y_pred.mean(axis=0)
y_pred_lower = np.percentile(y_pred, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred, 97.5, axis=0)

axes[1].scatter(y_obs, y_pred_mean, alpha=0.6)
axes[1].plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()],
             'k--', label='Perfect prediction')
axes[1].set_xlabel('Observed Y')
axes[1].set_ylabel('Predicted Y (posterior mean)')
axes[1].set_title('Observed vs Predicted')
axes[1].legend()

plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/ppc_log.png', dpi=300)
plt.close()

# Statistical PPC tests
ppc_data = az.from_pymc3(prior=None, posterior_predictive=post_pred_log,
                          coords={'obs_id': range(len(y_obs))})

# Test statistics
def test_stat_mean(y):
    return np.mean(y, axis=-1)

def test_stat_std(y):
    return np.std(y, axis=-1)

def test_stat_min(y):
    return np.min(y, axis=-1)

def test_stat_max(y):
    return np.max(y, axis=-1)

# Compute p-values
obs_mean = y_obs.mean()
pred_means = y_pred.mean(axis=1)
p_mean = np.mean(pred_means < obs_mean)

print("\nPosterior Predictive Check Statistics:")
print(f"P(mean_pred < mean_obs) = {p_mean:.3f}")
print(f"P(std_pred < std_obs) = {np.mean(y_pred.std(axis=1) < y_obs.std()):.3f}")
print(f"P(max_pred < max_obs) = {np.mean(y_pred.max(axis=1) < y_obs.max()):.3f}")
```

## Residual Diagnostics

```python
# Posterior mean predictions
mu_post = trace_log.posterior['beta_0'].values.flatten()[:, None] + \
          trace_log.posterior['beta_1'].values.flatten()[:, None] * np.log(x_obs)
mu_mean = mu_post.mean(axis=0)

# Residuals
residuals = y_obs - mu_mean
resid_std = residuals / residuals.std()

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Residuals vs fitted
axes[0, 0].scatter(mu_mean, residuals, alpha=0.6)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Panel 2: Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Panel 3: Residuals vs x
axes[1, 0].scatter(x_obs, residuals, alpha=0.6)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residuals vs x')

# Panel 4: Residual histogram
axes[1, 1].hist(residuals, bins=15, density=True, alpha=0.7, edgecolor='black')
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, 0, residuals.std()),
                'r-', label='Normal')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/residuals_log.png', dpi=300)
plt.close()

# Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\nShapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
```

## LOO-CV Analysis

```python
# Compute LOO
loo_log = az.loo(trace_log, pointwise=True)
print("\nLOO-CV Results:")
print(loo_log)

# Extract Pareto k values
pareto_k = loo_log['pareto_k'].values
print(f"\nPareto k diagnostics:")
print(f"  k < 0.5 (good): {np.sum(pareto_k < 0.5)} ({100*np.sum(pareto_k < 0.5)/len(pareto_k):.1f}%)")
print(f"  0.5 < k < 0.7 (ok): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))} ({100*np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))/len(pareto_k):.1f}%)")
print(f"  k > 0.7 (bad): {np.sum(pareto_k >= 0.7)} ({100*np.sum(pareto_k >= 0.7)/len(pareto_k):.1f}%)")

# Plot Pareto k values
az.plot_khat(loo_log, show_bins=True)
plt.title('LOO-CV Pareto k Diagnostic')
plt.savefig('/workspace/experiments/designer_1/pareto_k_log.png', dpi=300)
plt.close()

# WAIC for comparison
waic_log = az.waic(trace_log, pointwise=True)
print("\nWAIC Results:")
print(waic_log)
```

## Predictions and Visualization

```python
# Generate predictions on fine grid
with log_model:
    # Update x data for prediction
    pm.set_data({'x': x_pred})
    post_pred_new = pm.sample_posterior_predictive(
        trace_log,
        var_names=['Y_obs'],
        predictions=True,
        random_seed=42
    )

# Extract predictions
y_pred_new = post_pred_new.predictions['Y_obs'].values
y_pred_mean = y_pred_new.mean(axis=(0,1))
y_pred_lower = np.percentile(y_pred_new, 2.5, axis=(0,1))
y_pred_upper = np.percentile(y_pred_new, 97.5, axis=(0,1))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Observed data
ax.scatter(x_obs, y_obs, s=60, alpha=0.7, color='black', label='Observed', zorder=5)

# Posterior mean
ax.plot(x_pred, y_pred_mean, 'b-', linewidth=2, label='Posterior mean', zorder=3)

# Credible intervals
ax.fill_between(x_pred, y_pred_lower, y_pred_upper, alpha=0.3,
                color='blue', label='95% credible interval', zorder=1)

# Highlight extrapolation region
ax.axvline(x_obs.max(), color='red', linestyle='--',
           label=f'Max observed x = {x_obs.max():.1f}', zorder=2)
ax.axvspan(x_obs.max(), x_pred.max(), alpha=0.1, color='red')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Model 1: Logarithmic Regression - Posterior Predictions', fontsize=14)
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/predictions_log.png', dpi=300)
plt.close()
```

## Stan Alternative (if preferred)

```python
# Stan code for Model 1
stan_code_log = """
data {
  int<lower=0> N;          // Number of observations
  vector[N] x;              // Predictor
  vector[N] y;              // Response
}

parameters {
  real beta_0;              // Intercept
  real beta_1;              // Log-slope
  real<lower=0> sigma;      // Residual SD
}

transformed parameters {
  vector[N] mu;
  mu = beta_0 + beta_1 * log(x);
}

model {
  // Priors
  beta_0 ~ normal(1.73, 0.5);
  beta_1 ~ normal(0.28, 0.15);
  sigma ~ exponential(5.0);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_pred;         // Posterior predictive
  vector[N] log_lik;        // Log-likelihood for LOO

  for (n in 1:N) {
    y_pred[n] = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
"""

# Compile and fit (using cmdstanpy)
# import cmdstanpy
#
# stan_data = {
#     'N': len(x_obs),
#     'x': x_obs,
#     'y': y_obs
# }
#
# model_stan = cmdstanpy.CmdStanModel(stan_file='log_model.stan')
# fit_stan = model_stan.sample(
#     data=stan_data,
#     chains=4,
#     iter_sampling=2000,
#     iter_warmup=1000,
#     seed=42
# )
```

---

# Model 2: Power Law Regression

## PyMC Implementation

```python
import pymc as pm
import pytensor.tensor as pt

# Model specification
with pm.Model() as power_model:
    # Data
    x = pm.Data('x', x_obs)
    y = pm.Data('y', y_obs)

    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.8, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.5, sigma=0.3)
    beta_2 = pm.Normal('beta_2', mu=0.3, sigma=0.2)
    sigma = pm.Exponential('sigma', lam=5.0)

    # Mean structure
    mu = beta_0 + beta_1 * pt.power(x, beta_2)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

# Sample from posterior (may need more tuning)
with power_model:
    trace_power = pm.sample(
        draws=2000,
        tune=2000,  # Extended tuning for nonlinear model
        chains=4,
        cores=4,
        target_accept=0.99,  # Higher for complex geometry
        random_seed=42,
        return_inferencedata=True
    )

    # Posterior predictive
    post_pred_power = pm.sample_posterior_predictive(
        trace_power,
        random_seed=42
    )

trace_power.extend(post_pred_power)

# Check for divergences
n_divergences = trace_power.sample_stats['diverging'].sum().values
print(f"Number of divergent transitions: {n_divergences}")
if n_divergences > 0:
    print("WARNING: Divergences detected. Model may be misspecified or need reparameterization.")
```

---

# Model 3: Asymptotic (Michaelis-Menten)

## PyMC Implementation

```python
import pymc as pm

# Model specification
with pm.Model() as asymptotic_model:
    # Data
    x = pm.Data('x', x_obs)
    y = pm.Data('y', y_obs)

    # Priors
    Y_min = pm.Normal('Y_min', mu=1.7, sigma=0.3)
    Y_range = pm.Normal('Y_range', mu=0.9, sigma=0.3)
    K = pm.LogNormal('K', mu=np.log(5), sigma=1.0)
    sigma = pm.Exponential('sigma', lam=5.0)

    # Derived quantity
    Y_max = pm.Deterministic('Y_max', Y_min + Y_range)

    # Mean structure (Michaelis-Menten)
    mu = Y_min + Y_range * x / (K + x)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

# Sample from posterior (most challenging model)
with asymptotic_model:
    trace_asymptotic = pm.sample(
        draws=2000,
        tune=3000,  # Extended tuning
        chains=4,
        cores=4,
        target_accept=0.99,
        init='adapt_diag',  # Better initialization
        random_seed=42,
        return_inferencedata=True
    )

    # Posterior predictive
    post_pred_asymptotic = pm.sample_posterior_predictive(
        trace_asymptotic,
        random_seed=42
    )

trace_asymptotic.extend(post_pred_asymptotic)

# Check convergence
print(az.summary(trace_asymptotic, var_names=['Y_min', 'Y_max', 'K', 'sigma']))
```

---

# Model Comparison

## LOO-CV Comparison

```python
# Compute LOO for all models
loo_log = az.loo(trace_log)
loo_power = az.loo(trace_power)
loo_asymptotic = az.loo(trace_asymptotic)

# Compare models
loo_compare = az.compare({
    'Logarithmic': trace_log,
    'Power Law': trace_power,
    'Asymptotic': trace_asymptotic
}, ic='loo')

print("\nModel Comparison (LOO-CV):")
print(loo_compare)

# Plot comparison
az.plot_compare(loo_compare)
plt.title('Model Comparison via LOO-CV')
plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/model_comparison_loo.png', dpi=300)
plt.close()
```

## WAIC Comparison

```python
# WAIC comparison
waic_compare = az.compare({
    'Logarithmic': trace_log,
    'Power Law': trace_power,
    'Asymptotic': trace_asymptotic
}, ic='waic')

print("\nModel Comparison (WAIC):")
print(waic_compare)
```

## Visual Comparison

```python
# Generate predictions for all models
# (assuming you've updated pm.Data for each model to x_pred)

# Plot all three models
fig, ax = plt.subplots(figsize=(12, 7))

# Observed data
ax.scatter(x_obs, y_obs, s=80, alpha=0.8, color='black',
           label='Observed', zorder=10)

# Model 1: Logarithmic
ax.plot(x_pred, y_pred_log_mean, 'b-', linewidth=2,
        label='Model 1: Logarithmic', zorder=3)
ax.fill_between(x_pred, y_pred_log_lower, y_pred_log_upper,
                alpha=0.2, color='blue')

# Model 2: Power Law
ax.plot(x_pred, y_pred_power_mean, 'g-', linewidth=2,
        label='Model 2: Power Law', zorder=4)
ax.fill_between(x_pred, y_pred_power_lower, y_pred_power_upper,
                alpha=0.2, color='green')

# Model 3: Asymptotic
ax.plot(x_pred, y_pred_asymptotic_mean, 'r-', linewidth=2,
        label='Model 3: Asymptotic', zorder=5)
ax.fill_between(x_pred, y_pred_asymptotic_lower, y_pred_asymptotic_upper,
                alpha=0.2, color='red')

# Mark extrapolation region
ax.axvline(x_obs.max(), color='gray', linestyle='--',
           label=f'Max observed x', zorder=2)
ax.axvspan(x_obs.max(), x_pred.max(), alpha=0.1, color='gray')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('All Models: Posterior Predictions Comparison', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/all_models_comparison.png', dpi=300)
plt.close()
```

---

# Prior Sensitivity Analysis

## Model 1 with Alternative Priors

```python
# Vague priors
with pm.Model() as log_model_vague:
    x = pm.Data('x', x_obs)
    y = pm.Data('y', y_obs)

    beta_0 = pm.Normal('beta_0', mu=0, sigma=10)  # Vague
    beta_1 = pm.Normal('beta_1', mu=0, sigma=10)  # Vague
    sigma = pm.Exponential('sigma', lam=0.1)      # Vague

    mu = beta_0 + beta_1 * pm.math.log(x)
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

with log_model_vague:
    trace_vague = pm.sample(2000, tune=1000, chains=4,
                             target_accept=0.95, random_seed=42)

# Informative priors
with pm.Model() as log_model_informative:
    x = pm.Data('x', x_obs)
    y = pm.Data('y', y_obs)

    beta_0 = pm.Normal('beta_0', mu=1.73, sigma=0.2)  # Tight
    beta_1 = pm.Normal('beta_1', mu=0.28, sigma=0.05) # Tight
    sigma = pm.Exponential('sigma', lam=10)           # Tight

    mu = beta_0 + beta_1 * pm.math.log(x)
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

with log_model_informative:
    trace_informative = pm.sample(2000, tune=1000, chains=4,
                                   target_accept=0.95, random_seed=42)

# Compare posteriors
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, param in enumerate(['beta_0', 'beta_1', 'sigma']):
    # Extract posteriors
    post_weakly = trace_log.posterior[param].values.flatten()
    post_vague = trace_vague.posterior[param].values.flatten()
    post_info = trace_informative.posterior[param].values.flatten()

    axes[i].hist(post_weakly, bins=30, alpha=0.5,
                 label='Weakly informative', density=True)
    axes[i].hist(post_vague, bins=30, alpha=0.5,
                 label='Vague', density=True)
    axes[i].hist(post_info, bins=30, alpha=0.5,
                 label='Informative', density=True)

    axes[i].set_xlabel(param)
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'{param} Posterior Sensitivity')
    axes[i].legend()

plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/prior_sensitivity.png', dpi=300)
plt.close()
```

---

# Utility Functions

## Model Summary Report

```python
def generate_model_report(trace, model_name, save_dir='/workspace/experiments/designer_1'):
    """Generate comprehensive model report"""

    report = f"""
# {model_name} - Model Report
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Convergence Diagnostics
"""

    # Summary table
    summary = az.summary(trace, var_names=[v for v in trace.posterior.data_vars])
    report += f"\n{summary.to_string()}\n"

    # Convergence checks
    report += f"\n## Convergence Checks\n"
    report += f"- Max R-hat: {summary['r_hat'].max():.4f}\n"
    report += f"- Min ESS (bulk): {summary['ess_bulk'].min():.0f}\n"
    report += f"- Min ESS (tail): {summary['ess_tail'].min():.0f}\n"

    # Divergences
    n_div = trace.sample_stats['diverging'].sum().values
    report += f"- Divergent transitions: {n_div}\n"

    # LOO
    loo = az.loo(trace)
    report += f"\n## Model Fit (LOO-CV)\n"
    report += f"- LOO-ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}\n"
    report += f"- LOO-IC: {loo.loo:.2f}\n"

    # Pareto k
    pareto_k = loo.pareto_k.values
    report += f"\n## Pareto k Diagnostics\n"
    report += f"- k < 0.5: {np.sum(pareto_k < 0.5)} ({100*np.mean(pareto_k < 0.5):.1f}%)\n"
    report += f"- 0.5 ≤ k < 0.7: {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))} ({100*np.mean((pareto_k >= 0.5) & (pareto_k < 0.7)):.1f}%)\n"
    report += f"- k ≥ 0.7: {np.sum(pareto_k >= 0.7)} ({100*np.mean(pareto_k >= 0.7):.1f}%)\n"

    # Save report
    report_path = f"{save_dir}/{model_name.lower().replace(' ', '_')}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    return report

# Usage:
# generate_model_report(trace_log, "Logarithmic Model")
```

---

# Complete Workflow Script

```python
def complete_analysis_workflow():
    """Run complete Bayesian analysis workflow for Model 1"""

    print("="*60)
    print("BAYESIAN ANALYSIS WORKFLOW - MODEL 1 (LOGARITHMIC)")
    print("="*60)

    # Step 1: Load data
    print("\n[1/8] Loading data...")
    data = pd.read_csv('/workspace/data/data.csv')
    x_obs = data['x'].values
    y_obs = data['Y'].values
    print(f"  N = {len(x_obs)}, x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")

    # Step 2: Fit model
    print("\n[2/8] Fitting Model 1 (Logarithmic)...")
    with pm.Model() as log_model:
        x = pm.Data('x', x_obs)
        y = pm.Data('y', y_obs)

        beta_0 = pm.Normal('beta_0', mu=1.73, sigma=0.5)
        beta_1 = pm.Normal('beta_1', mu=0.28, sigma=0.15)
        sigma = pm.Exponential('sigma', lam=5.0)

        mu = beta_0 + beta_1 * pm.math.log(x)
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    with log_model:
        trace = pm.sample(2000, tune=1000, chains=4, cores=4,
                         target_accept=0.95, random_seed=42)

    # Step 3: Convergence diagnostics
    print("\n[3/8] Checking convergence...")
    summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'sigma'])
    print(summary)

    # Step 4: Posterior predictive
    print("\n[4/8] Generating posterior predictive...")
    with log_model:
        post_pred = pm.sample_posterior_predictive(trace, random_seed=42)
    trace.extend(post_pred)

    # Step 5: Residual diagnostics
    print("\n[5/8] Residual diagnostics...")
    mu_post = trace.posterior['beta_0'].values.flatten()[:, None] + \
              trace.posterior['beta_1'].values.flatten()[:, None] * np.log(x_obs)
    mu_mean = mu_post.mean(axis=0)
    residuals = y_obs - mu_mean
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"  Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

    # Step 6: LOO-CV
    print("\n[6/8] Computing LOO-CV...")
    loo = az.loo(trace, pointwise=True)
    print(f"  LOO-ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

    # Step 7: Posterior predictive checks
    print("\n[7/8] Posterior predictive checks...")
    y_pred = trace.posterior_predictive['Y_obs'].values.reshape(-1, len(y_obs))
    ppc_mean = np.mean(y_pred.mean(axis=1) < y_obs.mean())
    print(f"  P(mean_pred < mean_obs) = {ppc_mean:.3f}")

    # Step 8: Generate plots
    print("\n[8/8] Generating visualizations...")
    # (call plotting functions here)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return trace, log_model

# Run workflow
# trace, model = complete_analysis_workflow()
```

---

*End of Implementation Guide*
