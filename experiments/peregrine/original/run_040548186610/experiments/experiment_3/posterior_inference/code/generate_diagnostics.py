#!/usr/local/bin/python3
"""
Generate diagnostic plots and LOO comparison for Experiment 3
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Paths
DATA_PATH = '/workspace/data/data.csv'
EXP3_DIR = Path('/workspace/experiments/experiment_3/posterior_inference')
EXP1_DIR = Path('/workspace/experiments/experiment_1/posterior_inference')
PLOTS_DIR = EXP3_DIR / 'plots'
DIAG_DIR = EXP3_DIR / 'diagnostics'

# Load data
df = pd.read_csv(DATA_PATH)
year = df['year'].values
C = df['C'].values

# Load inference data
print("Loading Experiment 3 InferenceData...")
idata_exp3 = az.from_netcdf(DIAG_DIR / 'posterior_inference.netcdf')

print("Loading Experiment 1 InferenceData...")
idata_exp1 = az.from_netcdf(EXP1_DIR / 'diagnostics' / 'posterior_inference.netcdf')

#=============================================================================
# 1. CONVERGENCE DIAGNOSTIC PLOTS
#=============================================================================
print("\nGenerating convergence diagnostic plots...")

# Trace plots for main parameters
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Experiment 3: Trace Plots - Main Parameters', fontsize=14, fontweight='bold')

params = ['beta_0', 'beta_1', 'beta_2', 'rho', 'sigma_eta', 'phi']
param_labels = [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$', r'$\rho$', r'$\sigma_\eta$', r'$\phi$']

for i, (param, label) in enumerate(zip(params, param_labels)):
    ax = axes.flat[i]
    for chain in range(4):
        ax.plot(idata_exp3.posterior[param].sel(chain=chain), alpha=0.6, lw=0.5)
    ax.set_title(f'{label}', fontsize=12)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots_main_params.png', dpi=150, bbox_inches='tight')
plt.close()

# Rank plots for convergence
az.plot_rank(idata_exp3, var_names=params)
plt.gcf().suptitle('Experiment 3: Rank Plots - Convergence Check', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / 'rank_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# Posterior distributions
az.plot_posterior(idata_exp3, var_names=params, hdi_prob=0.95)
plt.gcf().suptitle('Experiment 3: Posterior Distributions', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / 'posterior_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Energy plot
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_energy(idata_exp3, ax=ax)
ax.set_title('Energy Plot - NUTS Diagnostic', fontsize=12, fontweight='bold')
plt.savefig(PLOTS_DIR / 'energy_plot.png', dpi=150, bbox_inches='tight')
plt.close()

#=============================================================================
# 2. PARAMETER COMPARISON WITH EXPERIMENT 1
#=============================================================================
print("\nComparing parameters with Experiment 1...")

# Extract posterior means
exp1_summary = az.summary(idata_exp1, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])
exp3_summary = az.summary(idata_exp3, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Parameter Comparison: Experiment 1 vs Experiment 3', fontsize=14, fontweight='bold')

compare_params = ['beta_0', 'beta_1', 'beta_2', 'phi']
compare_labels = [r'$\beta_0$ (Intercept)', r'$\beta_1$ (Linear)', r'$\beta_2$ (Quadratic)', r'$\phi$ (Dispersion)']

for i, (param, label) in enumerate(zip(compare_params, compare_labels)):
    ax = axes.flat[i]

    # Experiment 1 and 3 posteriors
    post1 = idata_exp1.posterior[param].values.flatten()
    post3 = idata_exp3.posterior[param].values.flatten()

    ax.hist(post1, bins=50, alpha=0.5, color='blue', density=True, label='Exp 1')
    ax.hist(post3, bins=50, alpha=0.5, color='red', density=True, label='Exp 3')

    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_comparison_exp1_vs_exp3.png', dpi=150, bbox_inches='tight')
plt.close()

#=============================================================================
# 3. AR(1) PARAMETER POSTERIORS
#=============================================================================
print("\nPlotting AR(1) parameter posteriors...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('AR(1) Parameters - Capturing Temporal Correlation', fontsize=14, fontweight='bold')

# Rho
ax = axes[0]
az.plot_posterior(idata_exp3, var_names=['rho'], ax=ax, hdi_prob=0.95,
                 point_estimate='mean', ref_val=0.8)
ax.set_title(r'$\rho$ (AR(1) Coefficient)', fontsize=12)
ax.axvline(0.686, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Exp 1 residual ACF(1)=0.686')
ax.legend(fontsize=9)

# Sigma_eta
ax = axes[1]
az.plot_posterior(idata_exp3, var_names=['sigma_eta'], ax=ax, hdi_prob=0.95,
                 point_estimate='mean')
ax.set_title(r'$\sigma_\eta$ (Innovation SD)', fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ar1_parameters.png', dpi=150, bbox_inches='tight')
plt.close()

#=============================================================================
# 4. LOO-CV COMPARISON
#=============================================================================
print("\nComputing LOO-CV...")

# Compute LOO for both models
loo_exp1 = az.loo(idata_exp1, pointwise=True)
loo_exp3 = az.loo(idata_exp3, pointwise=True)

# Compare
loo_compare = az.compare({'Exp1_QuadTrend': idata_exp1, 'Exp3_AR1': idata_exp3})

print("\nLOO Comparison:")
print(loo_compare)

# Save comparison
loo_compare.to_csv(DIAG_DIR / 'loo_comparison.csv')

# Create comparison text file
with open(DIAG_DIR / 'loo_comparison.txt', 'w') as f:
    f.write("LOO-CV MODEL COMPARISON\n")
    f.write("="*70 + "\n\n")
    f.write("Experiment 1 (Quadratic Trend Only):\n")
    f.write(f"  LOO-ELPD: {loo_exp1.elpd_loo:.2f} ± {loo_exp1.se:.2f}\n")
    f.write(f"  p_loo: {loo_exp1.p_loo:.2f}\n\n")

    f.write("Experiment 3 (Quadratic Trend + AR(1) Errors):\n")
    f.write(f"  LOO-ELPD: {loo_exp3.elpd_loo:.2f} ± {loo_exp3.se:.2f}\n")
    f.write(f"  p_loo: {loo_exp3.p_loo:.2f}\n\n")

    delta_elpd = loo_exp3.elpd_loo - loo_exp1.elpd_loo
    delta_se = np.sqrt(loo_exp1.se**2 + loo_exp3.se**2)

    f.write("Comparison:\n")
    f.write(f"  Δ ELPD (Exp3 - Exp1): {delta_elpd:.2f} ± {delta_se:.2f}\n")

    if delta_elpd > 2 * delta_se:
        f.write("  Result: Experiment 3 is CLEARLY BETTER\n")
    elif delta_elpd > 0:
        f.write("  Result: Experiment 3 is better (weak evidence)\n")
    else:
        f.write("  Result: No improvement over Experiment 1\n")

    f.write(f"\n  Weight: Exp1={loo_compare.loc['Exp1_QuadTrend', 'weight']:.3f}, ")
    f.write(f"Exp3={loo_compare.loc['Exp3_AR1', 'weight']:.3f}\n")

# Plot LOO comparison
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(loo_compare, insample_dev=False, ax=ax)
ax.set_title('LOO-CV Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loo_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

#=============================================================================
# 5. FITTED VALUES AND RESIDUALS
#=============================================================================
print("\nPlotting fitted values...")

# Extract posterior mean predictions
mu_post = idata_exp3.posterior['mu'].values  # shape: (chains, draws, N)
mu_mean = mu_post.reshape(-1, len(year)).mean(axis=0)
mu_lower = np.percentile(mu_post.reshape(-1, len(year)), 2.5, axis=0)
mu_upper = np.percentile(mu_post.reshape(-1, len(year)), 97.5, axis=0)

# Fitted values plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(year, C, color='black', s=50, alpha=0.6, label='Observed', zorder=3)
ax.plot(year, mu_mean, color='red', linewidth=2, label='Fitted (posterior mean)', zorder=2)
ax.fill_between(year, mu_lower, mu_upper, color='red', alpha=0.2, label='95% CI', zorder=1)
ax.set_xlabel('Standardized Year', fontsize=12)
ax.set_ylabel('Case Count', fontsize=12)
ax.set_title('Experiment 3: Fitted Values with AR(1) Errors', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fitted_values.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70)
print(f"\nPlots saved to: {PLOTS_DIR}")
print(f"LOO comparison saved to: {DIAG_DIR / 'loo_comparison.txt'}")
