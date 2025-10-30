"""
Fit Log-Log Power Law Model using PyMC (fallback from Stan)

Model: log(Y) ~ Normal(α + β*log(x), σ)
Priors: α ~ Normal(0.6, 0.3), β ~ Normal(0.12, 0.05), σ ~ Half-Cauchy(0, 0.05)
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

# Setup paths
BASE_DIR = "/workspace/experiments/experiment_3/posterior_inference"
CODE_DIR = f"{BASE_DIR}/code"
DIAG_DIR = f"{BASE_DIR}/diagnostics"
PLOT_DIR = f"{BASE_DIR}/plots"

# Load data
print("Loading data...")
data_df = pd.read_csv("/workspace/data/data.csv")
print(f"Loaded {len(data_df)} observations")
print(f"x range: [{data_df['x'].min():.2f}, {data_df['x'].max():.2f}]")
print(f"Y range: [{data_df['Y'].min():.2f}, {data_df['Y'].max():.2f}]")

# Transform to log scale
log_x = np.log(data_df['x'].values)
log_Y = np.log(data_df['Y'].values)

print(f"\nLog-transformed data:")
print(f"log(x) range: [{log_x.min():.2f}, {log_x.max():.2f}]")
print(f"log(Y) range: [{log_Y.min():.2f}, {log_Y.max():.2f}]")

# Build PyMC model
print("\n" + "="*70)
print("BUILDING PYMC MODEL")
print("="*70)

with pm.Model() as model:
    # Priors (REVISED based on prior predictive checks)
    alpha = pm.Normal('alpha', mu=0.6, sigma=0.3)
    beta = pm.Normal('beta', mu=0.12, sigma=0.05)
    sigma = pm.HalfCauchy('sigma', beta=0.05)

    # Linear predictor on log scale
    mu_log = alpha + beta * log_x

    # Likelihood: log(Y) ~ Normal(α + β*log(x), σ)
    log_Y_obs = pm.Normal('log_Y_obs', mu=mu_log, sigma=sigma, observed=log_Y)

    # Posterior predictive (on original scale)
    y_rep = pm.Deterministic('y_rep', pm.math.exp(mu_log))

print("Model specification:")
print(model)

# Fit model with NUTS sampler
print("\n" + "="*70)
print("FITTING MODEL WITH NUTS SAMPLER")
print("="*70)
print("Configuration: 4 chains, 2000 iterations (1000 warmup), target_accept=0.95")

with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        return_inferencedata=True,
        random_seed=42
    )

    # Generate posterior predictive samples
    print("\nGenerating posterior predictive samples...")
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

print("\n" + "="*70)
print("SAMPLING COMPLETED")
print("="*70)

# Add constant data (observed_data already exists from PyMC)
idata.add_groups({
    'constant_data': {
        'x': (['obs_id'], data_df['x'].values),
        'log_x': (['obs_id'], log_x),
        'Y': (['obs_id'], data_df['Y'].values),
    }
})

# Add coordinates
idata.posterior = idata.posterior.assign_coords({'obs_id': np.arange(len(data_df))})
idata.posterior_predictive = idata.posterior_predictive.assign_coords({'obs_id': np.arange(len(data_df))})
idata.constant_data = idata.constant_data.assign_coords({'obs_id': np.arange(len(data_df))})

# Save InferenceData
netcdf_path = f"{DIAG_DIR}/posterior_inference.netcdf"
idata.to_netcdf(netcdf_path)
print(f"\nSaved ArviZ InferenceData to: {netcdf_path}")

# Print summary statistics
print("\n" + "="*70)
print("POSTERIOR SUMMARY")
print("="*70)
summary = az.summary(idata, var_names=['alpha', 'beta', 'sigma'])
print(summary)

# Save summary to file
summary.to_csv(f"{DIAG_DIR}/parameter_summary.csv")

# Check convergence criteria
print("\n" + "="*70)
print("CONVERGENCE CHECKS")
print("="*70)

rhat_max = summary['r_hat'].max()
ess_bulk_min = summary['ess_bulk'].min()
ess_tail_min = summary['ess_tail'].min()

# Check for divergences
divergences = idata.sample_stats['diverging'].sum().values
n_samples = idata.sample_stats['diverging'].size

print(f"Maximum R-hat: {rhat_max:.4f} (target: < 1.01)")
print(f"Minimum ESS (bulk): {ess_bulk_min:.0f} (target: > 400)")
print(f"Minimum ESS (tail): {ess_tail_min:.0f} (target: > 400)")
print(f"Divergent transitions: {divergences}/{n_samples} (target: 0)")

convergence_passed = (
    rhat_max < 1.01 and
    ess_bulk_min > 400 and
    ess_tail_min > 400 and
    divergences == 0
)

print(f"\nOverall convergence: {'PASSED ✓' if convergence_passed else 'NEEDS REVIEW'}")

# Generate diagnostic plots
print("\n" + "="*70)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*70)

# 1. Trace plots
print("Creating trace plots...")
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
az.plot_trace(idata, var_names=['alpha', 'beta', 'sigma'], axes=axes)
fig.suptitle('Trace Plots and Marginal Posteriors', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/trace_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/trace_plots.png")

# 2. Rank plots (for checking mixing)
print("Creating rank plots...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(idata, var_names=['alpha', 'beta', 'sigma'])
plt.suptitle('Rank Plots (Checking Chain Mixing)', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rank_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/rank_plots.png")

# 3. Posterior distributions
print("Creating posterior distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
az.plot_posterior(idata, var_names=['alpha', 'beta', 'sigma'], ax=axes)
plt.suptitle('Posterior Distributions with 95% HDI', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/posterior_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/posterior_distributions.png")

# 4. Pairs plot (for parameter correlations)
print("Creating pairs plot...")
az.plot_pair(
    idata,
    var_names=['alpha', 'beta', 'sigma'],
    kind='hexbin',
    divergences=True,
    figsize=(10, 10)
)
plt.suptitle('Pairs Plot (Parameter Correlations)', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/pairs_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/pairs_plot.png")

# 5. Posterior predictive check
print("Creating posterior predictive check...")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot observed data
ax.scatter(data_df['x'], data_df['Y'], color='black', s=50, alpha=0.7,
           label='Observed data', zorder=10)

# Plot posterior predictive samples (100 random draws)
log_Y_obs_samples = idata.posterior_predictive['log_Y_obs'].values
n_chains, n_samples, n_obs = log_Y_obs_samples.shape
log_Y_obs_flat = log_Y_obs_samples.reshape(-1, n_obs)

# Back-transform to original scale
y_rep_samples = np.exp(log_Y_obs_flat)

# Randomly select 100 posterior predictive samples
n_draws = min(100, y_rep_samples.shape[0])
sample_indices = np.random.choice(y_rep_samples.shape[0], n_draws, replace=False)

for idx in sample_indices:
    ax.plot(data_df['x'], y_rep_samples[idx], color='steelblue', alpha=0.05)

# Plot posterior mean prediction
y_rep_mean = y_rep_samples.mean(axis=0)
ax.plot(data_df['x'], y_rep_mean, color='red', linewidth=2,
        label='Posterior mean', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Posterior Predictive Check (100 samples)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/posterior_predictive_check.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/posterior_predictive_check.png")

# 6. Power law fit visualization
print("Creating power law fit visualization...")
fig, ax = plt.subplots(figsize=(12, 6))

# Get posterior samples for parameters
alpha_samples = idata.posterior['alpha'].values.flatten()
beta_samples = idata.posterior['beta'].values.flatten()

# Create smooth x range for predictions
x_pred = np.linspace(data_df['x'].min(), data_df['x'].max(), 100)
log_x_pred = np.log(x_pred)

# Plot posterior predictive bands
y_pred_samples = np.zeros((len(alpha_samples), len(x_pred)))
for i, (a, b) in enumerate(zip(alpha_samples, beta_samples)):
    y_pred_samples[i] = np.exp(a + b * log_x_pred)

y_pred_median = np.median(y_pred_samples, axis=0)
y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

ax.fill_between(x_pred, y_pred_lower, y_pred_upper, alpha=0.3, color='steelblue',
                label='95% Credible Interval')
ax.plot(x_pred, y_pred_median, color='blue', linewidth=2, label='Median prediction')

# Plot observed data
ax.scatter(data_df['x'], data_df['Y'], color='black', s=50, alpha=0.7,
          label='Observed data', zorder=10)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Power Law Fit: Y = exp(α) × x^β', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/power_law_fit.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/power_law_fit.png")

# 7. Residual analysis on log scale
print("Creating residual analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Get posterior mean predictions on log scale
alpha_mean = alpha_samples.mean()
beta_mean = beta_samples.mean()
log_Y_pred = alpha_mean + beta_mean * log_x
residuals_log = log_Y - log_Y_pred

# Residuals vs fitted
axes[0].scatter(log_Y_pred, residuals_log, alpha=0.6, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Fitted values (log scale)', fontsize=11)
axes[0].set_ylabel('Residuals (log scale)', fontsize=11)
axes[0].set_title('Residuals vs Fitted Values', fontsize=12)
axes[0].grid(True, alpha=0.3)

# QQ plot
from scipy import stats
stats.probplot(residuals_log, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/residual_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/residual_analysis.png")

# Calculate model fit metrics (on original scale)
print("\n" + "="*70)
print("MODEL FIT METRICS")
print("="*70)

# Posterior mean predictions
y_pred_mean = np.exp(alpha_mean + beta_mean * log_x)

# R² on original scale
y_obs = data_df['Y'].values
ss_res = np.sum((y_obs - y_pred_mean)**2)
ss_tot = np.sum((y_obs - y_obs.mean())**2)
r2 = 1 - (ss_res / ss_tot)

# RMSE on original scale
rmse = np.sqrt(np.mean((y_obs - y_pred_mean)**2))

print(f"R² (original scale): {r2:.4f}")
print(f"RMSE (original scale): {rmse:.4f}")

# Parameter estimates with back-transformation
sigma_mean = idata.posterior['sigma'].values.flatten().mean()
sigma_sd = idata.posterior['sigma'].values.flatten().std()

print(f"\nParameter estimates (posterior mean ± SD):")
print(f"  α = {alpha_mean:.3f} ± {alpha_samples.std():.3f}")
print(f"  β = {beta_mean:.3f} ± {beta_samples.std():.3f}")
print(f"  σ = {sigma_mean:.3f} ± {sigma_sd:.3f}")
print(f"\nBack-transformed:")
print(f"  exp(α) = {np.exp(alpha_mean):.3f} (scaling constant)")
print(f"  β = {beta_mean:.3f} (elasticity: 1% increase in x → {100*beta_mean:.2f}% increase in Y)")

# 95% credible intervals
alpha_hdi = az.hdi(idata, var_names=['alpha'])['alpha'].values
beta_hdi = az.hdi(idata, var_names=['beta'])['beta'].values
sigma_hdi = az.hdi(idata, var_names=['sigma'])['sigma'].values

print(f"\n95% Credible Intervals:")
print(f"  α: [{alpha_hdi[0]:.3f}, {alpha_hdi[1]:.3f}]")
print(f"  β: [{beta_hdi[0]:.3f}, {beta_hdi[1]:.3f}]")
print(f"  σ: [{sigma_hdi[0]:.3f}, {sigma_hdi[1]:.3f}]")

print("\n" + "="*70)
print("FITTING COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {BASE_DIR}/")
print(f"  - ArviZ InferenceData: {DIAG_DIR}/posterior_inference.netcdf")
print(f"  - Parameter summary: {DIAG_DIR}/parameter_summary.csv")
print(f"  - Diagnostic plots: {PLOT_DIR}/")
