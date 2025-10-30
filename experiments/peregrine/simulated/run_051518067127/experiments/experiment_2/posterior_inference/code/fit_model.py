"""
Fit AR(1) Log-Normal Model with Regime-Switching to Real Data

This script:
1. Loads the vectorized PyMC model to avoid compilation issues
2. Fits to real data using MCMC (NUTS sampler)
3. Performs convergence diagnostics
4. Saves posterior inference data for LOO-CV comparison
5. Generates comprehensive diagnostic plots
"""

import sys
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add current directory to path for model import
sys.path.append(str(Path(__file__).parent))
from model_vectorized import build_model_vectorized

# Setup paths
BASE_DIR = Path('/workspace/experiments/experiment_2/posterior_inference')
DIAGNOSTICS_DIR = BASE_DIR / 'diagnostics'
PLOTS_DIR = BASE_DIR / 'plots'

# Ensure directories exist
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_context('notebook')
sns.set_style('whitegrid')

print("="*80)
print("FITTING AR(1) LOG-NORMAL MODEL WITH REGIME-SWITCHING TO REAL DATA")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/7] Loading data...")
data = pd.read_csv('/workspace/data/data.csv')
print(f"  Loaded {len(data)} observations")
print(f"  Year range: [{data['year'].min():.3f}, {data['year'].max():.3f}]")
print(f"  C range: [{data['C'].min()}, {data['C'].max()}]")

# Create regime indices (0-indexed for Python)
# Regimes: 1-14 (regime 1), 15-27 (regime 2), 28-40 (regime 3)
regime_idx = np.concatenate([
    np.zeros(14, dtype=int),      # Regime 1: obs 1-14
    np.ones(13, dtype=int),        # Regime 2: obs 15-27
    np.full(13, 2, dtype=int)      # Regime 3: obs 28-40
])

print(f"  Regime structure: {len(regime_idx)} observations")
print(f"    Regime 1 (early): {(regime_idx==0).sum()} obs")
print(f"    Regime 2 (middle): {(regime_idx==1).sum()} obs")
print(f"    Regime 3 (late): {(regime_idx==2).sum()} obs")

# Compute data ACF for comparison
log_C = np.log(data['C'].values)
data_acf_lag1 = np.corrcoef(log_C[:-1], log_C[1:])[0, 1]
print(f"  Data ACF lag-1 (log-scale): {data_acf_lag1:.4f}")

# ============================================================================
# 2. BUILD MODEL
# ============================================================================

print("\n[2/7] Building PyMC model (vectorized)...")
model = build_model_vectorized(data, regime_idx)
print("  Model built successfully")
print(f"  Free parameters: {len(model.free_RVs)}")

# ============================================================================
# 3. FIT MODEL USING MCMC
# ============================================================================

print("\n[3/7] Fitting model with MCMC...")
print("  Strategy: Adaptive sampling")
print("    Phase 1: Short probe (200 iterations) to diagnose")
print("    Phase 2: Main sampling if probe succeeds")

# Phase 1: Quick probe
print("\n  Phase 1: Diagnostic probe...")
with model:
    probe_trace = pm.sample(
        draws=200,
        tune=300,
        chains=4,
        cores=4,
        target_accept=0.90,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )

# Check probe convergence
probe_summary = az.summary(probe_trace, var_names=['alpha', 'beta_1', 'beta_2', 'phi'])
max_rhat_probe = probe_summary['r_hat'].max()
min_ess_probe = probe_summary['ess_bulk'].min()

print(f"    Probe results: max R-hat = {max_rhat_probe:.4f}, min ESS = {min_ess_probe:.0f}")

if max_rhat_probe > 1.1:
    print("    WARNING: Probe shows convergence issues!")
    print("    Proceeding with caution...")
    needs_adjustment = True
else:
    print("    Probe successful - proceeding to main sampling")
    needs_adjustment = False

# Check for divergences in probe
n_divergences_probe = probe_trace.sample_stats.diverging.sum().values
print(f"    Divergences in probe: {n_divergences_probe}")

# Phase 2: Main sampling
print("\n  Phase 2: Main sampling...")
if needs_adjustment or n_divergences_probe > 5:
    print("    Using conservative settings (target_accept=0.95)")
    target_accept = 0.95
    tune_iters = 2000
else:
    print("    Using standard settings (target_accept=0.90)")
    target_accept = 0.90
    tune_iters = 1500

with model:
    trace = pm.sample(
        draws=2000,
        tune=tune_iters,
        chains=4,
        cores=4,
        target_accept=target_accept,
        random_seed=123,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )

print("  Sampling complete!")

# ============================================================================
# 4. CONVERGENCE DIAGNOSTICS
# ============================================================================

print("\n[4/7] Performing convergence diagnostics...")

# Full summary
summary = az.summary(trace)
summary_params = az.summary(trace, var_names=['alpha', 'beta_1', 'beta_2', 'phi', 'phi_raw',
                                                'sigma_regime', 'epsilon_0'])

print("\n  Parameter Summary:")
print(summary_params[['mean', 'sd', 'hdi_5%', 'hdi_95%', 'r_hat', 'ess_bulk', 'ess_tail']])

# Convergence checks
max_rhat = summary_params['r_hat'].max()
min_ess_bulk = summary_params['ess_bulk'].min()
min_ess_tail = summary_params['ess_tail'].min()

n_divergences = trace.sample_stats.diverging.sum().values
divergence_rate = n_divergences / (4 * 2000) * 100

print(f"\n  Convergence Metrics:")
print(f"    Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"    Min ESS bulk: {min_ess_bulk:.0f} (target: > 400)")
print(f"    Min ESS tail: {min_ess_tail:.0f} (target: > 400)")
print(f"    Divergences: {n_divergences} ({divergence_rate:.2f}%)")

# Convergence decision
convergence_pass = (max_rhat < 1.01) and (min_ess_bulk > 400) and (divergence_rate < 2.0)
print(f"\n  Convergence Status: {'PASS' if convergence_pass else 'FAIL'}")

if not convergence_pass:
    print("    WARNING: Convergence criteria not met!")
    if max_rhat >= 1.01:
        print(f"      - R-hat too high: {max_rhat:.4f}")
    if min_ess_bulk <= 400:
        print(f"      - ESS too low: {min_ess_bulk:.0f}")
    if divergence_rate >= 2.0:
        print(f"      - Too many divergences: {divergence_rate:.2f}%")

# Save convergence summary
with open(DIAGNOSTICS_DIR / 'convergence_summary.txt', 'w') as f:
    f.write("CONVERGENCE DIAGNOSTICS - AR(1) LOG-NORMAL MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Sampling Configuration:\n")
    f.write(f"  Chains: 4\n")
    f.write(f"  Draws per chain: 2000\n")
    f.write(f"  Tune: {tune_iters}\n")
    f.write(f"  Target accept: {target_accept}\n\n")
    f.write(f"Convergence Metrics:\n")
    f.write(f"  Max R-hat: {max_rhat:.4f}\n")
    f.write(f"  Min ESS bulk: {min_ess_bulk:.0f}\n")
    f.write(f"  Min ESS tail: {min_ess_tail:.0f}\n")
    f.write(f"  Divergences: {n_divergences} ({divergence_rate:.2f}%)\n\n")
    f.write(f"Status: {'PASS' if convergence_pass else 'FAIL'}\n\n")
    f.write("Parameter Summary:\n")
    f.write(summary_params.to_string())

# Save parameter summary
summary_params.to_csv(DIAGNOSTICS_DIR / 'parameter_summary.csv')

# ============================================================================
# 5. SAVE INFERENCE DATA FOR LOO-CV
# ============================================================================

print("\n[5/7] Saving inference data for LOO-CV...")

# Verify log_likelihood is present
if 'log_likelihood' not in trace.log_likelihood:
    print("  ERROR: log_likelihood not found in trace!")
else:
    ll_shape = trace.log_likelihood['log_likelihood'].shape
    print(f"  log_likelihood shape: {ll_shape}")
    print(f"  Expected shape: (4 chains, 2000 draws, 40 observations)")

    # Save to NetCDF
    trace.to_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')
    print(f"  Saved to: {DIAGNOSTICS_DIR / 'posterior_inference.netcdf'}")

# ============================================================================
# 6. PARAMETER INFERENCE & COMPARISON
# ============================================================================

print("\n[6/7] Parameter inference...")

# Extract posteriors
posterior = trace.posterior

# Key parameters
alpha_post = posterior['alpha'].values.flatten()
beta_1_post = posterior['beta_1'].values.flatten()
beta_2_post = posterior['beta_2'].values.flatten()
phi_post = posterior['phi'].values.flatten()
sigma_regime_post = posterior['sigma_regime'].values  # shape: (chains, draws, 3)

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

# ============================================================================
# 7. RESIDUAL DIAGNOSTICS & MODEL FIT
# ============================================================================

print("\n[7/7] Computing residual diagnostics...")

# Extract fitted values
mu_full_post = posterior['mu_full'].values  # shape: (chains, draws, n_obs)
mu_trend_post = posterior['mu_trend'].values

# Compute residuals (on log scale)
log_C_obs = np.log(data['C'].values)
residuals = log_C_obs[np.newaxis, np.newaxis, :] - mu_trend_post
residuals_flat = residuals.reshape(-1, len(data))

# Posterior mean residuals
residuals_mean = residuals.mean(axis=(0, 1))

# Compute residual ACF lag-1
residual_acf_lag1 = np.corrcoef(residuals_mean[:-1], residuals_mean[1:])[0, 1]
print(f"  Residual ACF lag-1: {residual_acf_lag1:.4f} (Exp 1 was 0.596)")

# Compute fit metrics on original scale
C_obs = data['C'].values
C_pred_mean = np.exp(mu_full_post.mean(axis=(0, 1)))

mae = np.mean(np.abs(C_obs - C_pred_mean))
rmse = np.sqrt(np.mean((C_obs - C_pred_mean)**2))

print(f"\n  Fit Metrics (original scale):")
print(f"    MAE: {mae:.2f} (Exp 1: 16.41)")
print(f"    RMSE: {rmse:.2f} (Exp 1: 26.12)")

# Bayesian R²
var_fitted = np.var(C_pred_mean)
var_residual = np.var(C_obs - C_pred_mean)
r2_bayes = var_fitted / (var_fitted + var_residual)
print(f"    Bayesian R²: {r2_bayes:.4f}")

# ============================================================================
# 8. GENERATE DIAGNOSTIC PLOTS
# ============================================================================

print("\n[8/8] Generating diagnostic plots...")

# Plot 1: Trace plots for convergence assessment
print("  Creating trace plots...")
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle('MCMC Trace Plots - Core Parameters', fontsize=14, fontweight='bold')

params_to_plot = ['alpha', 'beta_1', 'beta_2', 'phi']
for i, param in enumerate(params_to_plot):
    # Trace plot
    ax_trace = axes[i, 0]
    for chain in range(4):
        ax_trace.plot(trace.posterior[param].values[chain, :], alpha=0.6, lw=0.5)
    ax_trace.set_ylabel(param, fontsize=11)
    ax_trace.set_xlabel('Iteration')
    ax_trace.grid(alpha=0.3)

    # Density plot
    ax_density = axes[i, 1]
    for chain in range(4):
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
print(f"    Saved: {PLOTS_DIR / 'trace_plots.png'}")

# Plot 2: Posterior distributions for all parameters
print("  Creating posterior distribution plots...")
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle('Posterior Distributions - All Parameters', fontsize=14, fontweight='bold')

# Core parameters
az.plot_posterior(trace, var_names=['alpha'], ax=axes[0, 0], hdi_prob=0.90)
az.plot_posterior(trace, var_names=['beta_1'], ax=axes[0, 1], hdi_prob=0.90)
az.plot_posterior(trace, var_names=['beta_2'], ax=axes[0, 2], hdi_prob=0.90)

# AR parameter
az.plot_posterior(trace, var_names=['phi'], ax=axes[1, 0], hdi_prob=0.90,
                 ref_val=data_acf_lag1)
axes[1, 0].axvline(data_acf_lag1, color='red', linestyle='--', alpha=0.5,
                   label=f'Data ACF={data_acf_lag1:.3f}')
axes[1, 0].legend(fontsize=8)

# Regime sigmas
for i in range(3):
    ax = axes[1, i+1] if i < 2 else axes[2, 0]
    az.plot_posterior(trace, var_names=['sigma_regime'],
                     coords={'sigma_regime_dim_0': i}, ax=ax, hdi_prob=0.90)
    ax.set_title(f'sigma_regime[{i+1}]')

# epsilon_0
az.plot_posterior(trace, var_names=['epsilon_0'], ax=axes[2, 1], hdi_prob=0.90)

# Hide unused subplot
axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'posterior_distributions.png'}")

# Plot 3: Fitted trend with credible intervals
print("  Creating fitted trend plot...")
fig, ax = plt.subplots(figsize=(12, 6))

year_vals = data['year'].values
C_obs = data['C'].values

# Posterior predictive samples (on original scale)
C_pred_samples = np.exp(mu_full_post)  # shape: (chains, draws, n_obs)
C_pred_samples_flat = C_pred_samples.reshape(-1, len(data))

# Percentiles for credible intervals
C_pred_median = np.median(C_pred_samples_flat, axis=0)
C_pred_lower = np.percentile(C_pred_samples_flat, 5, axis=0)
C_pred_upper = np.percentile(C_pred_samples_flat, 95, axis=0)

# Plot
ax.scatter(year_vals, C_obs, color='black', s=50, alpha=0.7, label='Observed', zorder=3)
ax.plot(year_vals, C_pred_median, color='blue', lw=2, label='Posterior Median', zorder=2)
ax.fill_between(year_vals, C_pred_lower, C_pred_upper,
                alpha=0.3, color='blue', label='90% Credible Interval')

# Add regime boundaries
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
print(f"    Saved: {PLOTS_DIR / 'fitted_trend.png'}")

# Plot 4: Residual diagnostics (KEY PLOT)
print("  Creating residual diagnostics plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Residual Diagnostics (Log Scale)', fontsize=14, fontweight='bold')

# 4a: Residuals over time
axes[0, 0].scatter(year_vals, residuals_mean, color='black', alpha=0.6, s=40)
axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Year (standardized)')
axes[0, 0].set_ylabel('Residual (log scale)')
axes[0, 0].set_title('Residuals vs Time')
axes[0, 0].grid(alpha=0.3)

# 4b: Residual ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals_mean, lags=10, ax=axes[0, 1], alpha=0.05)
axes[0, 1].set_title(f'Residual ACF (lag-1 = {residual_acf_lag1:.3f})')
axes[0, 1].axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Target < 0.3')
axes[0, 1].axhline(-0.3, color='orange', linestyle='--', alpha=0.5)
axes[0, 1].legend(fontsize=8)

# 4c: Q-Q plot
from scipy import stats
stats.probplot(residuals_mean, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)')
axes[1, 0].grid(alpha=0.3)

# 4d: Residuals vs fitted
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
print(f"    Saved: {PLOTS_DIR / 'residual_diagnostics.png'}")

# Plot 5: Regime posteriors comparison
print("  Creating regime posteriors plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Regime-Specific Standard Deviations', fontsize=14, fontweight='bold')

# Violin plot
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

# Overlapping densities
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
print(f"    Saved: {PLOTS_DIR / 'regime_posteriors.png'}")

# Plot 6: AR coefficient focus
print("  Creating AR coefficient plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('AR(1) Coefficient Analysis', fontsize=14, fontweight='bold')

# Posterior density with reference
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

# Trace plot for phi
for chain in range(4):
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
print(f"    Saved: {PLOTS_DIR / 'ar_coefficient.png'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FITTING COMPLETE - SUMMARY")
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

print(f"\nOutputs saved to: {BASE_DIR}")
print(f"  - Convergence diagnostics: diagnostics/")
print(f"  - Posterior inference: diagnostics/posterior_inference.netcdf")
print(f"  - Diagnostic plots: plots/")

# Decision
proceed_to_ppc = convergence_pass and (abs(residual_acf_lag1) < 0.5)
print(f"\nDecision: {'PROCEED TO PPC' if proceed_to_ppc else 'INVESTIGATE FURTHER'}")

if not proceed_to_ppc:
    if not convergence_pass:
        print("  Reason: Convergence criteria not met")
    elif abs(residual_acf_lag1) >= 0.5:
        print(f"  Reason: Residual ACF too high ({residual_acf_lag1:.3f})")

print("\n" + "="*80)
