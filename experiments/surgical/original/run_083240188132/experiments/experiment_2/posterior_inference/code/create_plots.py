"""
Create diagnostic plots after MCMC sampling
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Paths
DATA_PATH = Path('/workspace/data/data.csv')
OUTPUT_DIR = Path('/workspace/experiments/experiment_2/posterior_inference')
PLOTS_DIR = OUTPUT_DIR / 'plots'
DIAGNOSTICS_DIR = OUTPUT_DIR / 'diagnostics'

# Load data
data = pd.read_csv(DATA_PATH)
n = data['n'].values
r = data['r'].values
n_groups = len(data)
obs_prop = r / n

# Load trace
print("Loading posterior samples...")
trace = az.from_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')

print("\nCreating diagnostic plots...")

# Plot 1: Trace plots for key parameters
print("  - Creating trace plots...")
fig, axes = plt.subplots(5, 2, figsize=(14, 14))
az.plot_trace(
    trace,
    var_names=['mu', 'tau'],
    axes=axes[0:2, :],
    compact=False
)
az.plot_trace(
    trace,
    var_names=['theta'],
    coords={'theta_dim_0': [0, 5, 11]},
    axes=axes[2:5, :],
    compact=False
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Posterior distributions for hyperparameters
print("  - Creating posterior distributions...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# mu posterior
ax = axes[0]
az.plot_posterior(trace, var_names=['mu'], ax=ax, hdi_prob=0.94)
ax.axvline(-2.51, color='red', linestyle='--', alpha=0.5, label='Prior mean')
ax.set_xlabel('μ (population log-odds)')
ax.set_title('Population Mean (μ)')
ax.legend()

# tau posterior
ax = axes[1]
az.plot_posterior(trace, var_names=['tau'], ax=ax, hdi_prob=0.94)
ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Prior mode')
ax.set_xlabel('τ (between-group SD)')
ax.set_title('Between-Group Heterogeneity (τ)')
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_hyperparameters.png', dpi=300, bbox_inches='tight')
plt.close()

# Extract posterior samples
p_samples = trace.posterior['p'].values.reshape(-1, n_groups)
p_mean = p_samples.mean(axis=0)
p_hdi = az.hdi(trace, var_names=['p'], hdi_prob=0.94)['p'].values

# Plot 3: Forest plot for group probabilities
print("  - Creating forest plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Create forest plot
y_pos = np.arange(n_groups)
ax.errorbar(
    p_mean, y_pos,
    xerr=[p_mean - p_hdi[:, 0], p_hdi[:, 1] - p_mean],
    fmt='o', capsize=5, capthick=2, markersize=8,
    label='Posterior (94% HDI)', color='steelblue'
)
ax.scatter(obs_prop, y_pos, marker='x', s=100, color='red',
           label='Observed proportion', zorder=10)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Group {i+1}' for i in range(n_groups)])
ax.set_xlabel('Probability')
ax.set_title('Group-Level Event Probabilities (p_i)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'forest_plot_probabilities.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Energy diagnostic
print("  - Creating energy plot...")
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_energy(trace, ax=ax)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'energy_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Rank plots for key parameters
print("  - Creating rank plots...")
fig = plt.figure(figsize=(14, 10))
az.plot_rank(trace, var_names=['mu', 'tau', 'theta'], coords={'theta_dim_0': [0, 5, 11]})
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rank_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Shrinkage visualization
print("  - Creating shrinkage plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Observed proportions vs posterior means
ax.scatter(obs_prop, p_mean, s=100, alpha=0.7, color='steelblue')

# Add diagonal line (no shrinkage)
lim_min = min(obs_prop.min(), p_mean.min()) - 0.01
lim_max = max(obs_prop.max(), p_mean.max()) + 0.01
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, label='No shrinkage')

# Add arrows showing shrinkage
for i in range(n_groups):
    ax.annotate('', xy=(obs_prop[i], p_mean[i]), xytext=(obs_prop[i], obs_prop[i]),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=1.5))

# Population mean
mu_samples = trace.posterior['mu'].values.reshape(-1)
pop_mean_prob = 1 / (1 + np.exp(-mu_samples.mean()))
ax.axhline(pop_mean_prob, color='green', linestyle='--', alpha=0.5,
           label=f'Population mean: {pop_mean_prob:.3f}')

ax.set_xlabel('Observed Proportion')
ax.set_ylabel('Posterior Mean Probability')
ax.set_title('Shrinkage: Observed vs Posterior Estimates')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)

# Add group labels
for i in range(n_groups):
    ax.text(obs_prop[i], p_mean[i], f' {i+1}', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'shrinkage_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ All plots saved to: {PLOTS_DIR}")
