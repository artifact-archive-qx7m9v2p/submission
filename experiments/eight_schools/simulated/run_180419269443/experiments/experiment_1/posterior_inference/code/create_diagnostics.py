"""
Create diagnostic visualizations for posterior inference
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Setup
EXPERIMENT_DIR = Path("/workspace/experiments/experiment_1")
POSTERIOR_DIR = EXPERIMENT_DIR / "posterior_inference"
DIAGNOSTICS_DIR = POSTERIOR_DIR / "diagnostics"
PLOTS_DIR = POSTERIOR_DIR / "plots"
DATA_FILE = Path("/workspace/data/data.csv")

# Load data and results
data = pd.read_csv(DATA_FILE)
idata = az.from_netcdf(DIAGNOSTICS_DIR / "posterior_inference.netcdf")
summary = pd.read_csv(DIAGNOSTICS_DIR / "posterior_summary.csv", index_col=0)

with open(DIAGNOSTICS_DIR / "convergence_metrics.json", 'r') as f:
    metrics = json.load(f)

with open(DIAGNOSTICS_DIR / "derived_quantities.json", 'r') as f:
    derived = json.load(f)

print("Creating diagnostic visualizations...")
print("=" * 80)

# Set style
plt.style.use('default')
sns.set_palette("colorblind")

# 1. Trace plots for key parameters (mu, tau)
print("\n[1/7] Creating trace plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Mu trace
ax = axes[0, 0]
for chain in range(4):
    ax.plot(idata.posterior['mu'].values[chain, :], alpha=0.7, linewidth=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('mu')
ax.set_title(f'Trace Plot: mu (R-hat={summary.loc["mu", "r_hat"]:.4f})')
ax.axhline(metrics['key_metrics']['mu']['mean'], color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# Mu density
ax = axes[0, 1]
mu_samples = idata.posterior['mu'].values.flatten()
ax.hist(mu_samples, bins=50, density=True, alpha=0.6, edgecolor='black')
ax.axvline(metrics['key_metrics']['mu']['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
ax.axvline(metrics['key_metrics']['mu']['ci_3'], color='blue', linestyle='--', linewidth=1, label='94% HDI')
ax.axvline(metrics['key_metrics']['mu']['ci_97'], color='blue', linestyle='--', linewidth=1)
ax.set_xlabel('mu')
ax.set_ylabel('Density')
ax.set_title(f'Posterior: mu = {metrics["key_metrics"]["mu"]["mean"]:.2f} ± {metrics["key_metrics"]["mu"]["sd"]:.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Tau trace
ax = axes[1, 0]
for chain in range(4):
    ax.plot(idata.posterior['tau'].values[chain, :], alpha=0.7, linewidth=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('tau')
ax.set_title(f'Trace Plot: tau (R-hat={summary.loc["tau", "r_hat"]:.4f})')
ax.axhline(metrics['key_metrics']['tau']['mean'], color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# Tau density
ax = axes[1, 1]
tau_samples = idata.posterior['tau'].values.flatten()
ax.hist(tau_samples, bins=50, density=True, alpha=0.6, edgecolor='black')
ax.axvline(metrics['key_metrics']['tau']['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
ax.axvline(metrics['key_metrics']['tau']['ci_3'], color='blue', linestyle='--', linewidth=1, label='94% HDI')
ax.axvline(metrics['key_metrics']['tau']['ci_97'], color='blue', linestyle='--', linewidth=1)
ax.set_xlabel('tau')
ax.set_ylabel('Density')
ax.set_title(f'Posterior: tau = {metrics["key_metrics"]["tau"]["mean"]:.2f} ± {metrics["key_metrics"]["tau"]["sd"]:.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_and_posterior_key_params.png", dpi=300, bbox_inches='tight')
print(f"  Saved: trace_and_posterior_key_params.png")
plt.close()

# 2. ArviZ rank plots for convergence (compact)
print("\n[2/7] Creating rank plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

az.plot_rank(idata, var_names=['mu'], ax=axes[0])
axes[0].set_title(f'Rank Plot: mu (R-hat={summary.loc["mu", "r_hat"]:.4f})')

az.plot_rank(idata, var_names=['tau'], ax=axes[1])
axes[1].set_title(f'Rank Plot: tau (R-hat={summary.loc["tau", "r_hat"]:.4f})')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=300, bbox_inches='tight')
print(f"  Saved: rank_plots.png")
plt.close()

# 3. Forest plot for all theta_i
print("\n[3/7] Creating forest plot...")
fig, ax = plt.subplots(figsize=(10, 6))

J = len(data)
y_obs = data['y'].values
theta_means = [derived['theta_posterior'][f'study_{i+1}']['mean'] for i in range(J)]
theta_ci_low = [derived['theta_posterior'][f'study_{i+1}']['ci_2.5'] for i in range(J)]
theta_ci_high = [derived['theta_posterior'][f'study_{i+1}']['ci_97.5'] for i in range(J)]

# Plot observed y
ax.scatter(y_obs, range(J), color='red', s=100, marker='D', label='Observed y', zorder=3, alpha=0.7)

# Plot posterior theta with CIs
for i in range(J):
    ax.plot([theta_ci_low[i], theta_ci_high[i]], [i, i], color='blue', linewidth=2, alpha=0.6)
ax.scatter(theta_means, range(J), color='blue', s=80, marker='o', label='Posterior theta', zorder=3)

# Plot pooled estimate mu
mu_mean = metrics['key_metrics']['mu']['mean']
mu_ci_low = metrics['key_metrics']['mu']['ci_3']
mu_ci_high = metrics['key_metrics']['mu']['ci_97']
ax.axvline(mu_mean, color='green', linestyle='--', linewidth=2, label=f'mu = {mu_mean:.2f}', alpha=0.7)
ax.axvspan(mu_ci_low, mu_ci_high, alpha=0.1, color='green')

ax.set_yticks(range(J))
ax.set_yticklabels([f'Study {i+1}' for i in range(J)])
ax.set_xlabel('Effect Size')
ax.set_ylabel('Study')
ax.set_title('Forest Plot: Study Effects with 95% Credible Intervals')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "forest_plot.png", dpi=300, bbox_inches='tight')
print(f"  Saved: forest_plot.png")
plt.close()

# 4. Shrinkage plot
print("\n[4/7] Creating shrinkage plot...")
fig, ax = plt.subplots(figsize=(10, 6))

# Draw arrows from y_obs to theta_mean to mu_mean
for i in range(J):
    # y -> theta
    ax.arrow(y_obs[i], i - 0.1, theta_means[i] - y_obs[i], 0,
             head_width=0.15, head_length=abs(theta_means[i] - y_obs[i])*0.1 + 0.3,
             fc='blue', ec='blue', alpha=0.5, length_includes_head=True)
    # theta -> mu
    ax.arrow(theta_means[i], i + 0.1, mu_mean - theta_means[i], 0,
             head_width=0.15, head_length=abs(mu_mean - theta_means[i])*0.1 + 0.3,
             fc='green', ec='green', alpha=0.5, length_includes_head=True)

# Plot points
ax.scatter(y_obs, range(J), color='red', s=100, marker='D', label='Observed y', zorder=3)
ax.scatter(theta_means, range(J), color='blue', s=80, marker='o', label='Posterior theta', zorder=3)
ax.axvline(mu_mean, color='green', linestyle='--', linewidth=2, label=f'mu = {mu_mean:.2f}', alpha=0.7)

# Add shrinkage percentages
shrinkage = derived['shrinkage']
for i in range(J):
    shrink_val = shrinkage[f'study_{i+1}']
    ax.text(max(y_obs[i], theta_means[i]) + 1, i, f'{shrink_val:.1%}',
            fontsize=9, va='center')

ax.set_yticks(range(J))
ax.set_yticklabels([f'Study {i+1}' for i in range(J)])
ax.set_xlabel('Effect Size')
ax.set_ylabel('Study')
ax.set_title('Shrinkage Plot: y → theta → mu (percentages show shrinkage factor)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "shrinkage_plot.png", dpi=300, bbox_inches='tight')
print(f"  Saved: shrinkage_plot.png")
plt.close()

# 5. Pairs plot (mu vs tau correlation)
print("\n[5/7] Creating pairs plot...")
fig = plt.figure(figsize=(8, 8))
az.plot_pair(idata, var_names=['mu', 'tau'], kind='kde', marginals=True, figsize=(8, 8))
plt.suptitle('Joint Posterior: mu vs tau', y=1.02)
plt.savefig(PLOTS_DIR / "pairs_plot_mu_tau.png", dpi=300, bbox_inches='tight')
print(f"  Saved: pairs_plot_mu_tau.png")
plt.close()

# 6. LOO diagnostics
print("\n[6/7] Creating LOO diagnostic plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pareto k values
ax = axes[0]
pareto_k = metrics['loo']['pareto_k']
colors = ['green' if k < 0.5 else ('orange' if k < 0.7 else 'red') for k in pareto_k]
ax.bar(range(1, J+1), pareto_k, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='k=0.5 (threshold)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='k=0.7 (bad)')
ax.set_xlabel('Study')
ax.set_ylabel('Pareto k')
ax.set_title(f'LOO Pareto k Diagnostic (max k = {max(pareto_k):.3f})')
ax.set_xticks(range(1, J+1))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# ELPD contributions
ax = axes[1]
try:
    az.plot_loo_pit(idata, y='y', ecdf=True, ax=ax)
    ax.set_title('LOO-PIT: Calibration Check')
except:
    # Fallback if LOO-PIT doesn't work
    ax.text(0.5, 0.5, 'LOO-PIT plot\nnot available', ha='center', va='center',
            fontsize=14, transform=ax.transAxes)
    ax.set_title('LOO-PIT: Calibration Check')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"  Saved: loo_diagnostics.png")
plt.close()

# 7. I² posterior distribution
print("\n[7/7] Creating I² distribution plot...")
fig, ax = plt.subplots(figsize=(8, 5))

I2_mean = metrics['key_metrics']['I2']['mean']
I2_sd = metrics['key_metrics']['I2']['sd']
I2_ci_low = metrics['key_metrics']['I2']['ci_2.5']
I2_ci_high = metrics['key_metrics']['I2']['ci_97.5']

# Compute I² samples for histogram
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
mean_sigma_sq = np.mean(data['sigma'].values**2)
I2_samples = tau_samples**2 / (tau_samples**2 + mean_sigma_sq)

ax.hist(I2_samples * 100, bins=50, density=True, alpha=0.6, edgecolor='black', label='Posterior')
ax.axvline(I2_mean * 100, color='red', linestyle='--', linewidth=2, label=f'Mean = {I2_mean*100:.1f}%')
ax.axvline(I2_ci_low * 100, color='blue', linestyle='--', linewidth=1, label=f'95% CI')
ax.axvline(I2_ci_high * 100, color='blue', linestyle='--', linewidth=1)
ax.set_xlabel('I² (%)')
ax.set_ylabel('Density')
ax.set_title(f'Posterior Distribution of I² (Heterogeneity)\nI² = {I2_mean*100:.1f}% ± {I2_sd*100:.1f}%')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "I2_posterior.png", dpi=300, bbox_inches='tight')
print(f"  Saved: I2_posterior.png")
plt.close()

print("\n" + "=" * 80)
print("All diagnostic plots created successfully!")
print(f"Location: {PLOTS_DIR}")
print("=" * 80)
