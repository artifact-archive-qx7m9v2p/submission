#!/usr/bin/env python
"""Create diagnostic visualizations for hierarchical model"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
DIAG_DIR = BASE_DIR / 'experiments' / 'experiment_2' / 'posterior_inference' / 'diagnostics'
PLOT_DIR = BASE_DIR / 'experiments' / 'experiment_2' / 'posterior_inference' / 'plots'

PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data and posterior samples...")
df = pd.read_csv(DATA_PATH)
y_obs = df['y'].values
sigma_obs = df['sigma'].values
n_groups = len(y_obs)

idata = az.from_netcdf(DIAG_DIR / 'posterior_inference.netcdf')

# Extract key samples
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values

print("\n" + "="*80)
print("Creating Diagnostic Visualizations")
print("="*80)

# 1. Trace plots for key parameters (using ArviZ default)
print("\n[1/7] Trace plots...")
fig = az.plot_trace(idata, var_names=['mu', 'tau'], combined=False, figsize=(12, 6))
plt.suptitle('Trace and Density Plots - Hyperparameters', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'trace_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: trace_plots.png")

# 2. Posterior distributions for mu and tau
print("\n[2/7] Posterior distributions...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Posterior Distributions - Key Parameters', fontsize=14, fontweight='bold')

# mu
axes[0].hist(mu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
mu_hdi = az.hdi(idata.posterior['mu'])
mu_hdi_lower = float(mu_hdi['mu'].sel(hdi='lower'))
mu_hdi_upper = float(mu_hdi['mu'].sel(hdi='higher'))
axes[0].axvline(mu_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {mu_samples.mean():.2f}')
axes[0].axvline(mu_hdi_lower, color='green', linestyle=':', linewidth=1.5, label=f'95% HDI')
axes[0].axvline(mu_hdi_upper, color='green', linestyle=':', linewidth=1.5)
axes[0].set_xlabel('mu', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Population Mean (mu)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# tau
axes[1].hist(tau_samples, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
tau_hdi = az.hdi(idata.posterior['tau'])
tau_hdi_lower = float(tau_hdi['tau'].sel(hdi='lower'))
tau_hdi_upper = float(tau_hdi['tau'].sel(hdi='higher'))
axes[1].axvline(tau_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {tau_samples.mean():.2f}')
axes[1].axvline(np.median(tau_samples), color='blue', linestyle='-.', linewidth=2, label=f'Median = {np.median(tau_samples):.2f}')
axes[1].axvline(tau_hdi_lower, color='green', linestyle=':', linewidth=1.5, label=f'95% HDI')
axes[1].axvline(tau_hdi_upper, color='green', linestyle=':', linewidth=1.5)
axes[1].set_xlabel('tau', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Between-Group SD (tau)', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: posterior_distributions.png")

# 3. Forest plot - all parameters
print("\n[3/7] Forest plot...")
fig, ax = plt.subplots(figsize=(10, 8))
az.plot_forest(idata, var_names=['mu', 'tau', 'theta'], combined=True,
               hdi_prob=0.95, ax=ax, colors='steelblue')
ax.set_title('Forest Plot - All Parameters (95% HDI)', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Parameter Value', fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: forest_plot.png")

# 4. Shrinkage plot
print("\n[4/7] Shrinkage plot...")
theta_means = theta_samples.mean(axis=(0, 1))
mu_mean = mu_samples.mean()

fig, ax = plt.subplots(figsize=(10, 8))

# Plot observed data
ax.errorbar(range(n_groups), y_obs, yerr=1.96*sigma_obs, fmt='o',
            markersize=10, color='red', capsize=5, capthick=2,
            label='Observed data (y ± 1.96σ)', alpha=0.7)

# Plot posterior estimates for theta
for i in range(n_groups):
    theta_i = idata.posterior['theta'].isel(theta_dim_0=i)
    theta_hdi = az.hdi(theta_i)
    theta_hdi_lower = float(theta_hdi['theta'].sel(hdi='lower'))
    theta_hdi_upper = float(theta_hdi['theta'].sel(hdi='higher'))

    ax.errorbar(i, theta_means[i],
                yerr=[[theta_means[i]-theta_hdi_lower], [theta_hdi_upper-theta_means[i]]],
                fmt='s', markersize=8, color='blue', capsize=4, capthick=1.5,
                alpha=0.8, label='Posterior theta' if i == 0 else '')

    # Draw shrinkage arrows
    ax.annotate('', xy=(i, theta_means[i]), xytext=(i, y_obs[i]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

# Population mean
ax.axhline(mu_mean, color='green', linestyle='--', linewidth=2,
           label=f'Population mean (mu) = {mu_mean:.2f}')

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Shrinkage: Observed Data → Posterior Estimates → Population Mean',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(n_groups))
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'shrinkage_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: shrinkage_plot.png")

# 5. Funnel diagnostic (tau vs theta deviations)
print("\n[5/7] Funnel diagnostic...")
# Flatten samples
tau_flat = tau_samples
theta_raw_samples = idata.posterior['theta_raw'].values
theta_raw_flat = theta_raw_samples.reshape(-1, n_groups)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Funnel Diagnostic: tau vs theta_raw[i] (Non-Centered)',
             fontsize=14, fontweight='bold')

for i in range(n_groups):
    row = i // 4
    col = i % 4
    axes[row, col].scatter(tau_flat, theta_raw_flat[:, i], alpha=0.1, s=1)
    axes[row, col].set_xlabel('tau')
    axes[row, col].set_ylabel(f'theta_raw[{i}]')
    axes[row, col].set_title(f'Group {i}')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'funnel_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: funnel_diagnostic.png")

# 6. Rank plots (another convergence diagnostic)
print("\n[6/7] Rank plots...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(idata, var_names=['mu', 'tau'], kind='bars')
plt.suptitle('Rank Plots - Convergence Diagnostic', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'rank_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: rank_plots.png")

# 7. Comparison plot: theta posteriors with data
print("\n[7/7] Group means with data...")
fig, ax = plt.subplots(figsize=(12, 7))

# Violin plots for theta posteriors
theta_data = []
positions = []
for i in range(n_groups):
    theta_i_samples = idata.posterior['theta'].isel(theta_dim_0=i).values.flatten()
    theta_data.append(theta_i_samples)
    positions.append(i)

parts = ax.violinplot(theta_data, positions=positions, widths=0.6,
                      showmeans=True, showmedians=False)

for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_alpha(0.6)

# Overlay observed data
ax.scatter(range(n_groups), y_obs, color='red', s=100, zorder=10,
           label='Observed y', marker='o', edgecolors='black', linewidths=1.5)

# Error bars for measurement uncertainty
ax.errorbar(range(n_groups), y_obs, yerr=1.96*sigma_obs, fmt='none',
            ecolor='red', capsize=5, alpha=0.5, linewidth=1.5)

# Population mean
mu_hdi = az.hdi(idata.posterior['mu'])
mu_hdi_lower = float(mu_hdi['mu'].sel(hdi='lower'))
mu_hdi_upper = float(mu_hdi['mu'].sel(hdi='higher'))
ax.axhspan(mu_hdi_lower, mu_hdi_upper, alpha=0.2, color='green',
           label='mu 95% HDI')
ax.axhline(mu_mean, color='green', linestyle='--', linewidth=2,
           label=f'mu (mean) = {mu_mean:.2f}')

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Group-Specific Posterior Distributions (theta) with Observed Data',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(n_groups))
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'group_means.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: group_means.png")

print("\n" + "="*80)
print("All diagnostic plots created successfully!")
print("="*80)
