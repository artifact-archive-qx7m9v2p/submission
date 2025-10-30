"""
Complete the analysis after successful sampling
Load the trace and generate all diagnostics and reports
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
BASE_DIR = Path('/workspace/experiments/experiment_2/posterior_inference')
DIAGNOSTICS_DIR = BASE_DIR / 'diagnostics'
PLOTS_DIR = BASE_DIR / 'plots'

# Set plotting style
sns.set_context('notebook')
sns.set_style('whitegrid')

print("="*80)
print("COMPLETING AR(1) POSTERIOR INFERENCE ANALYSIS")
print("="*80)

# Load the trace
print("\nLoading trace from NetCDF...")
trace = az.from_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')
print("  Trace loaded successfully")

# Load data
data = pd.read_csv('/workspace/data/data.csv')
log_C = np.log(data['C'].values)
data_acf_lag1 = np.corrcoef(log_C[:-1], log_C[1:])[0, 1]

# Get summary with correct column names
print("\nComputing parameter summary...")
summary_params = az.summary(trace, var_names=['alpha', 'beta_1', 'beta_2', 'phi', 'phi_raw',
                                               'sigma_regime', 'epsilon_0'])

# Print available columns
print(f"Available columns: {list(summary_params.columns)}")

# Select columns that exist
display_cols = [col for col in ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']
                if col in summary_params.columns]

print("\n  Parameter Summary:")
print(summary_params[display_cols])

# Convergence checks
max_rhat = summary_params['r_hat'].max()
min_ess_bulk = summary_params['ess_bulk'].min()
min_ess_tail = summary_params['ess_tail'].min()

n_divergences = trace.sample_stats.diverging.sum().values
n_samples = trace.posterior.dims['chain'] * trace.posterior.dims['draw']
divergence_rate = n_divergences / n_samples * 100

print(f"\n  Convergence Metrics:")
print(f"    Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"    Min ESS bulk: {min_ess_bulk:.0f} (target: > 400)")
print(f"    Min ESS tail: {min_ess_tail:.0f} (target: > 400)")
print(f"    Divergences: {n_divergences} ({divergence_rate:.2f}%)")

# Convergence decision
convergence_pass = (max_rhat < 1.01) and (min_ess_bulk > 400) and (divergence_rate < 2.0)
print(f"\n  Convergence Status: {'PASS' if convergence_pass else 'FAIL'}")

# Extract posteriors
posterior = trace.posterior

# Key parameters
alpha_post = posterior['alpha'].values.flatten()
beta_1_post = posterior['beta_1'].values.flatten()
beta_2_post = posterior['beta_2'].values.flatten()
phi_post = posterior['phi'].values.flatten()
sigma_regime_post = posterior['sigma_regime'].values

print("\n  Posterior Summaries:")
print(f"    alpha: {alpha_post.mean():.4f} ± {alpha_post.std():.4f} [{np.percentile(alpha_post, 5):.4f}, {np.percentile(alpha_post, 95):.4f}]")
print(f"    beta_1: {beta_1_post.mean():.4f} ± {beta_1_post.std():.4f} [{np.percentile(beta_1_post, 5):.4f}, {np.percentile(beta_1_post, 95):.4f}]")
print(f"    beta_2: {beta_2_post.mean():.4f} ± {beta_2_post.std():.4f} [{np.percentile(beta_2_post, 5):.4f}, {np.percentile(beta_2_post, 95):.4f}]")
print(f"    phi (AR coef): {phi_post.mean():.4f} ± {phi_post.std():.4f} [{np.percentile(phi_post, 5):.4f}, {np.percentile(phi_post, 95):.4f}]")
print(f"    Data ACF lag-1: {data_acf_lag1:.4f}")

print(f"\n  Regime Standard Deviations:")
for i in range(3):
    sigma_i = sigma_regime_post[:, :, i].flatten()
    print(f"    sigma[{i+1}]: {sigma_i.mean():.4f} ± {sigma_i.std():.4f} [{np.percentile(sigma_i, 5):.4f}, {np.percentile(sigma_i, 95):.4f}]")

# Compute residuals
mu_full_post = posterior['mu_full'].values
mu_trend_post = posterior['mu_trend'].values

log_C_obs = np.log(data['C'].values)
residuals = log_C_obs[np.newaxis, np.newaxis, :] - mu_trend_post
residuals_mean = residuals.mean(axis=(0, 1))

residual_acf_lag1 = np.corrcoef(residuals_mean[:-1], residuals_mean[1:])[0, 1]
print(f"\n  Residual ACF lag-1: {residual_acf_lag1:.4f} (Exp 1 was 0.596)")

# Fit metrics
C_obs = data['C'].values
C_pred_mean = np.exp(mu_full_post.mean(axis=(0, 1)))

mae = np.mean(np.abs(C_obs - C_pred_mean))
rmse = np.sqrt(np.mean((C_obs - C_pred_mean)**2))

var_fitted = np.var(C_pred_mean)
var_residual = np.var(C_obs - C_pred_mean)
r2_bayes = var_fitted / (var_fitted + var_residual)

print(f"\n  Fit Metrics (original scale):")
print(f"    MAE: {mae:.2f} (Exp 1: 16.41)")
print(f"    RMSE: {rmse:.2f} (Exp 1: 26.12)")
print(f"    Bayesian R²: {r2_bayes:.4f}")

# Save convergence summary
print("\nSaving convergence summary...")
with open(DIAGNOSTICS_DIR / 'convergence_summary.txt', 'w') as f:
    f.write("CONVERGENCE DIAGNOSTICS - AR(1) LOG-NORMAL MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Sampling Configuration:\n")
    f.write(f"  Chains: {trace.posterior.dims['chain']}\n")
    f.write(f"  Draws per chain: {trace.posterior.dims['draw']}\n")
    f.write(f"  Target accept: 0.90\n\n")
    f.write(f"Convergence Metrics:\n")
    f.write(f"  Max R-hat: {max_rhat:.4f}\n")
    f.write(f"  Min ESS bulk: {min_ess_bulk:.0f}\n")
    f.write(f"  Min ESS tail: {min_ess_tail:.0f}\n")
    f.write(f"  Divergences: {n_divergences} ({divergence_rate:.2f}%)\n\n")
    f.write(f"Status: {'PASS' if convergence_pass else 'FAIL'}\n\n")
    f.write("Parameter Summary:\n")
    f.write(summary_params.to_string())

summary_params.to_csv(DIAGNOSTICS_DIR / 'parameter_summary.csv')

# Generate plots
print("\nGenerating diagnostic plots...")

# Plot 1: Trace plots
print("  Creating trace plots...")
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle('MCMC Trace Plots - Core Parameters', fontsize=14, fontweight='bold')

params_to_plot = ['alpha', 'beta_1', 'beta_2', 'phi']
for i, param in enumerate(params_to_plot):
    ax_trace = axes[i, 0]
    for chain in range(trace.posterior.dims['chain']):
        ax_trace.plot(trace.posterior[param].values[chain, :], alpha=0.6, lw=0.5)
    ax_trace.set_ylabel(param, fontsize=11)
    ax_trace.set_xlabel('Iteration')
    ax_trace.grid(alpha=0.3)

    ax_density = axes[i, 1]
    for chain in range(trace.posterior.dims['chain']):
        ax_density.hist(trace.posterior[param].values[chain, :],
                       bins=50, alpha=0.5, density=True, label=f'Chain {chain+1}')
    ax_density.set_xlabel(param)
    ax_density.set_ylabel('Density')
    ax_density.grid(alpha=0.3)
    if i == 0:
        ax_density.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Posterior distributions
print("  Creating posterior distribution plots...")
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle('Posterior Distributions - All Parameters', fontsize=14, fontweight='bold')

az.plot_posterior(trace, var_names=['alpha'], ax=axes[0, 0], hdi_prob=0.90)
az.plot_posterior(trace, var_names=['beta_1'], ax=axes[0, 1], hdi_prob=0.90)
az.plot_posterior(trace, var_names=['beta_2'], ax=axes[0, 2], hdi_prob=0.90)

az.plot_posterior(trace, var_names=['phi'], ax=axes[1, 0], hdi_prob=0.90,
                 ref_val=data_acf_lag1)
axes[1, 0].axvline(data_acf_lag1, color='red', linestyle='--', alpha=0.5,
                   label=f'Data ACF={data_acf_lag1:.3f}')
axes[1, 0].legend(fontsize=8)

for i in range(3):
    ax = axes[1, i+1] if i < 2 else axes[2, 0]
    az.plot_posterior(trace, var_names=['sigma_regime'],
                     coords={'sigma_regime_dim_0': i}, ax=ax, hdi_prob=0.90)
    ax.set_title(f'sigma_regime[{i+1}]')

az.plot_posterior(trace, var_names=['epsilon_0'], ax=axes[2, 1], hdi_prob=0.90)
axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Fitted trend
print("  Creating fitted trend plot...")
fig, ax = plt.subplots(figsize=(12, 6))

year_vals = data['year'].values
C_obs = data['C'].values

C_pred_samples = np.exp(mu_full_post)
C_pred_samples_flat = C_pred_samples.reshape(-1, len(data))

C_pred_median = np.median(C_pred_samples_flat, axis=0)
C_pred_lower = np.percentile(C_pred_samples_flat, 5, axis=0)
C_pred_upper = np.percentile(C_pred_samples_flat, 95, axis=0)

ax.scatter(year_vals, C_obs, color='black', s=50, alpha=0.7, label='Observed', zorder=3)
ax.plot(year_vals, C_pred_median, color='blue', lw=2, label='Posterior Median', zorder=2)
ax.fill_between(year_vals, C_pred_lower, C_pred_upper,
                alpha=0.3, color='blue', label='90% Credible Interval')

ax.axvline(year_vals[14], color='gray', linestyle='--', alpha=0.5)
ax.axvline(year_vals[27], color='gray', linestyle='--', alpha=0.5)
ax.text(year_vals[7], ax.get_ylim()[1]*0.95, 'Regime 1', ha='center', fontsize=10)
ax.text(year_vals[20], ax.get_ylim()[1]*0.95, 'Regime 2', ha='center', fontsize=10)
ax.text(year_vals[33], ax.get_ylim()[1]*0.95, 'Regime 3', ha='center', fontsize=10)

ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('C (count)', fontsize=12)
ax.set_title('Fitted Trend with 90% Credible Intervals', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fitted_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Residual diagnostics
print("  Creating residual diagnostics plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Residual Diagnostics (Log Scale)', fontsize=14, fontweight='bold')

axes[0, 0].scatter(year_vals, residuals_mean, color='black', alpha=0.6, s=40)
axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Year (standardized)')
axes[0, 0].set_ylabel('Residual (log scale)')
axes[0, 0].set_title('Residuals vs Time')
axes[0, 0].grid(alpha=0.3)

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals_mean, lags=10, ax=axes[0, 1], alpha=0.05)
axes[0, 1].set_title(f'Residual ACF (lag-1 = {residual_acf_lag1:.3f})')
axes[0, 1].axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Target < 0.3')
axes[0, 1].axhline(-0.3, color='orange', linestyle='--', alpha=0.5)
axes[0, 1].legend(fontsize=8)

from scipy import stats
stats.probplot(residuals_mean, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)')
axes[1, 0].grid(alpha=0.3)

fitted_mean_log = mu_full_post.mean(axis=(0, 1))
axes[1, 1].scatter(fitted_mean_log, residuals_mean, color='black', alpha=0.6, s=40)
axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Fitted values (log scale)')
axes[1, 1].set_ylabel('Residual (log scale)')
axes[1, 1].set_title('Residuals vs Fitted')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Regime posteriors
print("  Creating regime posteriors plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Regime-Specific Standard Deviations', fontsize=14, fontweight='bold')

sigma_data = []
for i in range(3):
    sigma_i = sigma_regime_post[:, :, i].flatten()
    sigma_data.append(sigma_i)

axes[0].violinplot(sigma_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(['Regime 1\n(Early)', 'Regime 2\n(Middle)', 'Regime 3\n(Late)'])
axes[0].set_ylabel('sigma (log scale)')
axes[0].set_title('Distribution Comparison')
axes[0].grid(alpha=0.3, axis='y')

for i, label in enumerate(['Regime 1', 'Regime 2', 'Regime 3']):
    axes[1].hist(sigma_data[i], bins=50, alpha=0.5, density=True, label=label)
axes[1].set_xlabel('sigma (log scale)')
axes[1].set_ylabel('Density')
axes[1].set_title('Overlapping Distributions')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'regime_posteriors.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: AR coefficient
print("  Creating AR coefficient plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('AR(1) Coefficient Analysis', fontsize=14, fontweight='bold')

axes[0].hist(phi_post, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(phi_post.mean(), color='blue', linestyle='-', lw=2,
                label=f'Posterior mean: {phi_post.mean():.3f}')
axes[0].axvline(data_acf_lag1, color='red', linestyle='--', lw=2,
                label=f'Data ACF lag-1: {data_acf_lag1:.3f}')
axes[0].set_xlabel('phi (AR coefficient)')
axes[0].set_ylabel('Density')
axes[0].set_title('Posterior Distribution of phi')
axes[0].legend()
axes[0].grid(alpha=0.3)

for chain in range(trace.posterior.dims['chain']):
    axes[1].plot(trace.posterior['phi'].values[chain, :], alpha=0.6, lw=0.8,
                 label=f'Chain {chain+1}')
axes[1].axhline(data_acf_lag1, color='red', linestyle='--', alpha=0.5,
                label=f'Data ACF={data_acf_lag1:.3f}')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('phi')
axes[1].set_title('Trace Plot for phi')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ar_coefficient.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*80)

print(f"\nConvergence: {'PASS' if convergence_pass else 'FAIL'}")
print(f"  Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"  Min ESS: {min_ess_bulk:.0f} (target: > 400)")
print(f"  Divergences: {n_divergences} ({divergence_rate:.2f}%)")

print(f"\nKey Parameters:")
print(f"  phi (AR coef): {phi_post.mean():.4f} ± {phi_post.std():.4f}")
print(f"  Data ACF lag-1: {data_acf_lag1:.4f}")

print(f"\nResidual Diagnostics:")
print(f"  Residual ACF lag-1: {residual_acf_lag1:.4f}")
print(f"  Exp 1 ACF: 0.596")
print(f"  Improvement: {abs(residual_acf_lag1) < 0.596}")

print(f"\nFit Quality:")
print(f"  MAE: {mae:.2f} (Exp 1: 16.41)")
print(f"  RMSE: {rmse:.2f} (Exp 1: 26.12)")
print(f"  Bayesian R²: {r2_bayes:.4f}")

proceed_to_ppc = convergence_pass and (abs(residual_acf_lag1) < 0.5)
print(f"\nDecision: {'PROCEED TO PPC' if proceed_to_ppc else 'INVESTIGATE FURTHER'}")

print("\n" + "="*80)
