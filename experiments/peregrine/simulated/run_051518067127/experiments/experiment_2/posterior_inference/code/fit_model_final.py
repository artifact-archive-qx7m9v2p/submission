"""
Fit AR(1) Log-Normal Model - Final Version with Column Name Fix
"""

import sys
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model_vectorized import build_model_vectorized

# Setup paths
BASE_DIR = Path('/workspace/experiments/experiment_2/posterior_inference')
DIAGNOSTICS_DIR = BASE_DIR / 'diagnostics'
PLOTS_DIR = BASE_DIR / 'plots'

sns.set_context('notebook')
sns.set_style('whitegrid')

print("="*80)
print("FITTING AR(1) LOG-NORMAL MODEL WITH REGIME-SWITCHING")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
data = pd.read_csv('/workspace/data/data.csv')
regime_idx = np.concatenate([
    np.zeros(14, dtype=int),
    np.ones(13, dtype=int),
    np.full(13, 2, dtype=int)
])

log_C = np.log(data['C'].values)
data_acf_lag1 = np.corrcoef(log_C[:-1], log_C[1:])[0, 1]
print(f"  Loaded {len(data)} observations")
print(f"  Data ACF lag-1: {data_acf_lag1:.4f}")

# Build model
print("\n[2/5] Building model...")
model = build_model_vectorized(data, regime_idx)
print("  Model built successfully")

# Sample
print("\n[3/5] Sampling from posterior...")
with model:
    trace = pm.sample(
        draws=2000,
        tune=1500,
        chains=4,
        cores=4,
        target_accept=0.90,
        random_seed=123,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )
print("  Sampling complete!")

# Diagnostics
print("\n[4/5] Computing diagnostics...")
summary_params = az.summary(trace, var_names=['alpha', 'beta_1', 'beta_2', 'phi', 'phi_raw',
                                               'sigma_regime', 'epsilon_0'])

# Handle different ArviZ column naming
avail_cols = list(summary_params.columns)
display_cols = []
for col in ['mean', 'sd', 'hdi_3%', 'hdi_5%', 'hdi_95%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']:
    if col in avail_cols:
        display_cols.append(col)

print("\nParameter Summary:")
print(summary_params[display_cols])

max_rhat = summary_params['r_hat'].max()
min_ess_bulk = summary_params['ess_bulk'].min()
n_divergences = trace.sample_stats.diverging.sum().values
n_samples = trace.posterior.dims['chain'] * trace.posterior.dims['draw']
divergence_rate = n_divergences / n_samples * 100

convergence_pass = (max_rhat < 1.01) and (min_ess_bulk > 400) and (divergence_rate < 2.0)

print(f"\nConvergence: {'PASS' if convergence_pass else 'FAIL'}")
print(f"  Max R-hat: {max_rhat:.4f}")
print(f"  Min ESS bulk: {min_ess_bulk:.0f}")
print(f"  Divergences: {n_divergences} ({divergence_rate:.2f}%)")

# Extract posteriors
posterior = trace.posterior
alpha_post = posterior['alpha'].values.flatten()
beta_1_post = posterior['beta_1'].values.flatten()
beta_2_post = posterior['beta_2'].values.flatten()
phi_post = posterior['phi'].values.flatten()
sigma_regime_post = posterior['sigma_regime'].values

# Residual diagnostics
mu_trend_post = posterior['mu_trend'].values
residuals = log_C[np.newaxis, np.newaxis, :] - mu_trend_post
residuals_mean = residuals.mean(axis=(0, 1))
residual_acf_lag1 = np.corrcoef(residuals_mean[:-1], residuals_mean[1:])[0, 1]

# Fit metrics
mu_full_post = posterior['mu_full'].values
C_obs = data['C'].values
C_pred_mean = np.exp(mu_full_post.mean(axis=(0, 1)))

mae = np.mean(np.abs(C_obs - C_pred_mean))
rmse = np.sqrt(np.mean((C_obs - C_pred_mean)**2))
var_fitted = np.var(C_pred_mean)
var_residual = np.var(C_obs - C_pred_mean)
r2_bayes = var_fitted / (var_fitted + var_residual)

print(f"\nKey Results:")
print(f"  phi: {phi_post.mean():.4f} ± {phi_post.std():.4f}")
print(f"  Residual ACF lag-1: {residual_acf_lag1:.4f} (Exp 1: 0.596)")
print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}")
print(f"  Bayesian R²: {r2_bayes:.4f}")

# Save outputs
print("\n[5/5] Saving outputs...")

# Save trace
trace.to_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')
print(f"  Saved: posterior_inference.netcdf")

# Save summary
with open(DIAGNOSTICS_DIR / 'convergence_summary.txt', 'w') as f:
    f.write("CONVERGENCE DIAGNOSTICS - AR(1) LOG-NORMAL MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Convergence: {'PASS' if convergence_pass else 'FAIL'}\n")
    f.write(f"  Max R-hat: {max_rhat:.4f}\n")
    f.write(f"  Min ESS bulk: {min_ess_bulk:.0f}\n")
    f.write(f"  Divergences: {n_divergences} ({divergence_rate:.2f}%)\n\n")
    f.write("Parameter Summary:\n")
    f.write(summary_params.to_string())

summary_params.to_csv(DIAGNOSTICS_DIR / 'parameter_summary.csv')

# Generate plots
print("\nGenerating plots...")

# Trace plots
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle('MCMC Trace Plots', fontsize=14, fontweight='bold')
for i, param in enumerate(['alpha', 'beta_1', 'beta_2', 'phi']):
    for chain in range(4):
        axes[i, 0].plot(trace.posterior[param].values[chain, :], alpha=0.6, lw=0.5)
        axes[i, 1].hist(trace.posterior[param].values[chain, :], bins=50, alpha=0.5, density=True)
    axes[i, 0].set_ylabel(param)
    axes[i, 0].grid(alpha=0.3)
    axes[i, 1].set_xlabel(param)
    axes[i, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# Posteriors
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle('Posterior Distributions', fontsize=14, fontweight='bold')
az.plot_posterior(trace, var_names=['alpha'], ax=axes[0, 0], hdi_prob=0.90)
az.plot_posterior(trace, var_names=['beta_1'], ax=axes[0, 1], hdi_prob=0.90)
az.plot_posterior(trace, var_names=['beta_2'], ax=axes[0, 2], hdi_prob=0.90)
az.plot_posterior(trace, var_names=['phi'], ax=axes[1, 0], hdi_prob=0.90)
for i in range(3):
    ax = axes[1, i+1] if i < 2 else axes[2, 0]
    az.plot_posterior(trace, var_names=['sigma_regime'],
                     coords={'sigma_regime_dim_0': i}, ax=ax, hdi_prob=0.90)
az.plot_posterior(trace, var_names=['epsilon_0'], ax=axes[2, 1], hdi_prob=0.90)
axes[2, 2].axis('off')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Fitted trend
fig, ax = plt.subplots(figsize=(12, 6))
year_vals = data['year'].values
C_pred_samples = np.exp(mu_full_post).reshape(-1, len(data))
C_pred_median = np.median(C_pred_samples, axis=0)
C_pred_lower = np.percentile(C_pred_samples, 5, axis=0)
C_pred_upper = np.percentile(C_pred_samples, 95, axis=0)
ax.scatter(year_vals, C_obs, color='black', s=50, alpha=0.7, label='Observed', zorder=3)
ax.plot(year_vals, C_pred_median, color='blue', lw=2, label='Posterior Median', zorder=2)
ax.fill_between(year_vals, C_pred_lower, C_pred_upper, alpha=0.3, color='blue', label='90% CI')
ax.axvline(year_vals[14], color='gray', linestyle='--', alpha=0.5)
ax.axvline(year_vals[27], color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Year (standardized)')
ax.set_ylabel('C (count)')
ax.set_title('Fitted Trend with 90% Credible Intervals', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fitted_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# Residual diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold')
axes[0, 0].scatter(year_vals, residuals_mean, color='black', alpha=0.6)
axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Residual (log scale)')
axes[0, 0].set_title('Residuals vs Time')
axes[0, 0].grid(alpha=0.3)

from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats as sp_stats
plot_acf(residuals_mean, lags=10, ax=axes[0, 1], alpha=0.05)
axes[0, 1].set_title(f'Residual ACF (lag-1 = {residual_acf_lag1:.3f})')
axes[0, 1].axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Target < 0.3')
axes[0, 1].legend(fontsize=8)

sp_stats.probplot(residuals_mean, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].grid(alpha=0.3)

fitted_mean_log = mu_full_post.mean(axis=(0, 1))
axes[1, 1].scatter(fitted_mean_log, residuals_mean, color='black', alpha=0.6)
axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Fitted (log scale)')
axes[1, 1].set_ylabel('Residual (log scale)')
axes[1, 1].set_title('Residuals vs Fitted')
axes[1, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

# Regime posteriors
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Regime-Specific Standard Deviations', fontsize=14, fontweight='bold')
sigma_data = [sigma_regime_post[:, :, i].flatten() for i in range(3)]
axes[0].violinplot(sigma_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(['Regime 1\n(Early)', 'Regime 2\n(Middle)', 'Regime 3\n(Late)'])
axes[0].set_ylabel('sigma (log scale)')
axes[0].grid(alpha=0.3, axis='y')
for i, label in enumerate(['Regime 1', 'Regime 2', 'Regime 3']):
    axes[1].hist(sigma_data[i], bins=50, alpha=0.5, density=True, label=label)
axes[1].set_xlabel('sigma')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'regime_posteriors.png', dpi=150, bbox_inches='tight')
plt.close()

# AR coefficient
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('AR(1) Coefficient Analysis', fontsize=14, fontweight='bold')
axes[0].hist(phi_post, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(phi_post.mean(), color='blue', linestyle='-', lw=2,
                label=f'Posterior mean: {phi_post.mean():.3f}')
axes[0].axvline(data_acf_lag1, color='red', linestyle='--', lw=2,
                label=f'Data ACF: {data_acf_lag1:.3f}')
axes[0].set_xlabel('phi')
axes[0].set_title('Posterior Distribution')
axes[0].legend()
axes[0].grid(alpha=0.3)

for chain in range(4):
    axes[1].plot(trace.posterior['phi'].values[chain, :], alpha=0.6, lw=0.8)
axes[1].axhline(data_acf_lag1, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('phi')
axes[1].set_title('Trace Plot')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ar_coefficient.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("FITTING COMPLETE")
print("="*80)
print(f"\nConvergence: {'PASS' if convergence_pass else 'FAIL'}")
print(f"Residual ACF lag-1: {residual_acf_lag1:.4f}")
print(f"Decision: {'PROCEED TO PPC' if (convergence_pass and abs(residual_acf_lag1) < 0.5) else 'INVESTIGATE'}")
print("\n" + "="*80)
