"""
Create diagnostic plots for convergence assessment
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

# Setup
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
PLOTS_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Load inference data
print("Loading inference data...")
trace = az.from_netcdf(BASE_DIR / "diagnostics" / "posterior_inference.netcdf")

# Load original data
data = pd.read_csv(DATA_PATH)
x = data['x'].values
y = data['Y'].values

print("Creating diagnostic plots...")

# Set style
plt.style.use('default')

# ============================================================
# PLOT 1: Convergence Overview (Trace + Rank plots)
# ============================================================
print("1. Creating convergence overview (trace + rank)...")

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
param_names = ['alpha', 'beta', 'gamma', 'sigma']

for i, param in enumerate(param_names):
    # Trace plot
    ax_trace = axes[i, 0]
    for chain in range(4):
        samples = trace.posterior[param].sel(chain=chain).values
        ax_trace.plot(samples, alpha=0.7, linewidth=0.5, label=f'Chain {chain}')
    ax_trace.set_ylabel(param, fontsize=12, fontweight='bold')
    ax_trace.set_xlabel('Iteration')
    if i == 0:
        ax_trace.set_title('Trace Plots', fontweight='bold')
    ax_trace.grid(alpha=0.3)

    # Rank plot
    ax_rank = axes[i, 1]
    az.plot_rank(trace, var_names=[param], ax=ax_rank)
    if i == 0:
        ax_rank.set_title('Rank Plots', fontweight='bold')

    # Posterior density
    ax_post = axes[i, 2]
    az.plot_posterior(trace, var_names=[param], ax=ax_post)
    if i == 0:
        ax_post.set_title('Posterior Densities', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_overview.png", dpi=300, bbox_inches='tight')
print(f"   Saved: convergence_overview.png")
plt.close()

# ============================================================
# PLOT 2: Model Fit Visualization
# ============================================================
print("2. Creating model fit visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Posterior predictive with credible intervals
ax = axes[0]
mu_samples = trace.posterior['mu'].values.reshape(-1, len(x))
mu_mean = mu_samples.mean(axis=0)
mu_lower = np.percentile(mu_samples, 2.5, axis=0)
mu_upper = np.percentile(mu_samples, 97.5, axis=0)

# Predictive intervals (from y_rep in posterior)
y_rep_samples = trace.posterior['y_rep'].values.reshape(-1, len(x))
y_rep_lower = np.percentile(y_rep_samples, 2.5, axis=0)
y_rep_upper = np.percentile(y_rep_samples, 97.5, axis=0)

# Sort for smooth plotting
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]

ax.fill_between(x_sorted, y_rep_lower[sort_idx], y_rep_upper[sort_idx],
                alpha=0.2, color='C1', label='95% Predictive Interval')
ax.fill_between(x_sorted, mu_lower[sort_idx], mu_upper[sort_idx],
                alpha=0.4, color='C0', label='95% Credible Interval')
ax.plot(x_sorted, mu_mean[sort_idx], 'C0-', linewidth=2, label='Posterior Mean')
ax.scatter(x, y, c='black', s=40, alpha=0.6, zorder=10, label='Observed Data')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Posterior Predictive Fit', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Right: Residuals
ax = axes[1]
residuals = y - mu_mean
ax.scatter(mu_mean, residuals, c='black', s=40, alpha=0.6)
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Fitted Values', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Add residual statistics
rmse = np.sqrt(np.mean(residuals**2))
ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_fit.png", dpi=300, bbox_inches='tight')
print(f"   Saved: model_fit.png")
plt.close()

# ============================================================
# PLOT 3: Posterior Distributions (detailed)
# ============================================================
print("3. Creating detailed posterior distributions...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Create axes for each parameter and pairplot
ax_alpha = fig.add_subplot(gs[0, 0])
ax_beta = fig.add_subplot(gs[0, 1])
ax_gamma = fig.add_subplot(gs[0, 2])
ax_sigma = fig.add_subplot(gs[1, 0])
ax_pair1 = fig.add_subplot(gs[1, 1:])
ax_pair2 = fig.add_subplot(gs[2, :2])
ax_pair3 = fig.add_subplot(gs[2, 2])

# Individual posteriors with HDI
for ax, param in zip([ax_alpha, ax_beta, ax_gamma, ax_sigma], param_names):
    samples = trace.posterior[param].values.flatten()
    az.plot_posterior(trace, var_names=[param], ax=ax,
                     textsize=10, ref_val=None)
    ax.set_title(f'{param}', fontweight='bold', fontsize=12)

# Joint distributions (pairplot style)
alpha_samples = trace.posterior['alpha'].values.flatten()
beta_samples = trace.posterior['beta'].values.flatten()
gamma_samples = trace.posterior['gamma'].values.flatten()

# alpha vs beta
ax_pair1.hexbin(alpha_samples, beta_samples, gridsize=30, cmap='Blues', alpha=0.7)
ax_pair1.set_xlabel('alpha', fontweight='bold')
ax_pair1.set_ylabel('beta', fontweight='bold')
ax_pair1.set_title('Joint: alpha vs beta', fontweight='bold')

# beta vs gamma
ax_pair2.hexbin(beta_samples, gamma_samples, gridsize=30, cmap='Greens', alpha=0.7)
ax_pair2.set_xlabel('beta', fontweight='bold')
ax_pair2.set_ylabel('gamma', fontweight='bold')
ax_pair2.set_title('Joint: beta vs gamma', fontweight='bold')

# alpha vs gamma
ax_pair3.hexbin(alpha_samples, gamma_samples, gridsize=30, cmap='Oranges', alpha=0.7)
ax_pair3.set_xlabel('alpha', fontweight='bold')
ax_pair3.set_ylabel('gamma', fontweight='bold')
ax_pair3.set_title('Joint: alpha vs gamma', fontweight='bold')

plt.suptitle('Posterior Parameter Distributions', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / "posterior_distributions.png", dpi=300, bbox_inches='tight')
print(f"   Saved: posterior_distributions.png")
plt.close()

# ============================================================
# PLOT 4: Posterior Predictive Checks
# ============================================================
print("4. Creating posterior predictive checks...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# PPC density overlay
ax = axes[0, 0]
y_rep_samples = trace.posterior['y_rep'].values
# Sample 100 posterior predictive draws
n_samples = min(100, y_rep_samples.shape[0] * y_rep_samples.shape[1])
y_rep_flat = y_rep_samples.reshape(-1, len(y))
sample_idx = np.random.choice(y_rep_flat.shape[0], n_samples, replace=False)

for idx in sample_idx:
    ax.hist(y_rep_flat[idx], bins=15, alpha=0.02, color='C0', density=True)
ax.hist(y, bins=15, alpha=0.7, color='red', density=True, label='Observed', edgecolor='black')
ax.set_xlabel('Y', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Predictive Density Check', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Test statistics: mean
ax = axes[0, 1]
y_rep_means = y_rep_flat.mean(axis=1)
obs_mean = y.mean()
ax.hist(y_rep_means, bins=30, alpha=0.7, color='C0', edgecolor='black')
ax.axvline(obs_mean, color='red', linewidth=2, label=f'Observed = {obs_mean:.3f}')
p_value = np.mean(y_rep_means >= obs_mean)
ax.set_xlabel('Mean(Y_rep)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Test Statistic: Mean (p={p_value:.3f})', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Test statistics: std
ax = axes[1, 0]
y_rep_stds = y_rep_flat.std(axis=1)
obs_std = y.std()
ax.hist(y_rep_stds, bins=30, alpha=0.7, color='C0', edgecolor='black')
ax.axvline(obs_std, color='red', linewidth=2, label=f'Observed = {obs_std:.3f}')
p_value = np.mean(y_rep_stds >= obs_std)
ax.set_xlabel('Std(Y_rep)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Test Statistic: Std (p={p_value:.3f})', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Test statistics: max
ax = axes[1, 1]
y_rep_maxs = y_rep_flat.max(axis=1)
obs_max = y.max()
ax.hist(y_rep_maxs, bins=30, alpha=0.7, color='C0', edgecolor='black')
ax.axvline(obs_max, color='red', linewidth=2, label=f'Observed = {obs_max:.3f}')
p_value = np.mean(y_rep_maxs >= obs_max)
ax.set_xlabel('Max(Y_rep)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Test Statistic: Max (p={p_value:.3f})', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_predictive_checks.png", dpi=300, bbox_inches='tight')
print(f"   Saved: posterior_predictive_checks.png")
plt.close()

# ============================================================
# PLOT 5: ESS and R-hat diagnostic
# ============================================================
print("5. Creating ESS and R-hat diagnostics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R-hat
ax = axes[0]
summary = az.summary(trace, var_names=['alpha', 'beta', 'gamma', 'sigma'])
params = summary.index
rhats = summary['r_hat'].values

colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhats]
ax.barh(params, rhats, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(1.01, color='green', linestyle='--', linewidth=2, label='Target (< 1.01)')
ax.axvline(1.05, color='orange', linestyle='--', linewidth=2, label='Acceptable (< 1.05)')
ax.set_xlabel('R-hat', fontsize=12, fontweight='bold')
ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
ax.set_title('R-hat Convergence Diagnostic', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# ESS
ax = axes[1]
ess_bulk = summary['ess_bulk'].values
ess_tail = summary['ess_tail'].values

x_pos = np.arange(len(params))
width = 0.35
ax.barh(x_pos - width/2, ess_bulk, width, label='ESS Bulk', alpha=0.7, color='C0', edgecolor='black')
ax.barh(x_pos + width/2, ess_tail, width, label='ESS Tail', alpha=0.7, color='C1', edgecolor='black')
ax.axvline(400, color='green', linestyle='--', linewidth=2, label='Target (> 400)')
ax.set_xlabel('Effective Sample Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
ax.set_yticks(x_pos)
ax.set_yticklabels(params)
ax.set_title('Effective Sample Size', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_metrics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: convergence_metrics.png")
plt.close()

print("\n" + "="*60)
print("DIAGNOSTIC PLOTS COMPLETE")
print("="*60)
print(f"All plots saved to: {PLOTS_DIR}")
