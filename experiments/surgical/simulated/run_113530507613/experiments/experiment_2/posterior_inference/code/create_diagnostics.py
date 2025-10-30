"""
Create diagnostic visualizations for the mixture model.
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Paths
project_root = Path("/workspace")
output_dir = project_root / "experiments" / "experiment_2" / "posterior_inference"
diag_dir = output_dir / "diagnostics"
plots_dir = output_dir / "plots"
data_path = project_root / "data" / "data.csv"

# Load data
data = pd.read_csv(data_path)
idata = az.from_netcdf(diag_dir / "posterior_inference.netcdf")
assignments = pd.read_csv(diag_dir / "cluster_assignments.csv")

print("Creating diagnostic plots...")

# 1. Combined trace plots
print("  1. Trace plots...")
az.plot_trace(idata, var_names=['pi', 'mu', 'sigma'], compact=False, combined=False,
              figsize=(14, 12))
plt.suptitle('Trace Plots: All Parameters', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(plots_dir / 'trace_all_params.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Cluster parameter posteriors
print("  2. Cluster parameter posteriors...")
mu_samples = idata.posterior['mu'].values.reshape(-1, 3)
p_cluster = 1 / (1 + np.exp(-mu_samples))
pi_samples = idata.posterior['pi'].values.reshape(-1, 3)
sigma_samples = idata.posterior['sigma'].values.reshape(-1, 3)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Cluster means on probability scale
axes[0].hist([p_cluster[:, k] for k in range(3)], bins=30, alpha=0.6,
             label=[f'Cluster {k+1}' for k in range(3)], density=True)
axes[0].set_xlabel('Success Probability', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Cluster Means (Probability Scale)', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Mixing proportions
axes[1].hist([pi_samples[:, k] for k in range(3)], bins=30, alpha=0.6,
             label=[f'Cluster {k+1}' for k in range(3)], density=True)
axes[1].set_xlabel('Mixing Proportion', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Mixing Proportions', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Cluster standard deviations
axes[2].hist([sigma_samples[:, k] for k in range(3)], bins=30, alpha=0.6,
             label=[f'Cluster {k+1}' for k in range(3)], density=True)
axes[2].set_xlabel('Standard Deviation', fontsize=12)
axes[2].set_ylabel('Density', fontsize=12)
axes[2].set_title('Cluster Standard Deviations', fontsize=13, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / 'cluster_parameters.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Cluster assignment probabilities heatmap
print("  3. Cluster assignment heatmap...")
cluster_probs_mean = idata.posterior['cluster_probs'].values.mean(axis=(0, 1))

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cluster_probs_mean.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

ax.set_xticks(range(len(assignments)))
ax.set_xticklabels(assignments['group_id'], rotation=0)
ax.set_yticks(range(3))
ax.set_yticklabels([f'Cluster {k+1}' for k in range(3)])
ax.set_xlabel('Group ID', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Posterior Cluster Assignment Probabilities', fontsize=13, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Probability', fontsize=11)

# Add text annotations
for i in range(len(assignments)):
    for j in range(3):
        text = ax.text(i, j, f'{cluster_probs_mean[i, j]:.2f}',
                      ha="center", va="center", color="black" if cluster_probs_mean[i, j] < 0.5 else "white",
                      fontsize=9)

plt.tight_layout()
plt.savefig(plots_dir / 'cluster_assignments_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Cluster assignments visualization
print("  4. Cluster assignments with observed data...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Observed success rates by assigned cluster
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for k in range(3):
    mask = assignments['assigned_cluster'] == k+1
    axes[0].scatter(assignments.loc[mask, 'group_id'],
                   assignments.loc[mask, 'success_rate'],
                   c=colors[k], s=100, alpha=0.7, label=f'Cluster {k+1}',
                   edgecolors='black', linewidth=1)

# Add cluster means
for k in range(3):
    p_mean = p_cluster[:, k].mean()
    axes[0].axhline(p_mean, color=colors[k], linestyle='--', alpha=0.5, linewidth=2)

axes[0].set_xlabel('Group ID', fontsize=12)
axes[0].set_ylabel('Observed Success Rate', fontsize=12)
axes[0].set_title('Observed Data with Cluster Assignments', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Plot 2: Assignment certainty
axes[1].bar(assignments['group_id'], assignments['certainty'],
           color=['red' if c < 0.6 else 'orange' if c < 0.7 else 'green'
                  for c in assignments['certainty']])
axes[1].axhline(0.6, color='red', linestyle='--', alpha=0.5, label='Low certainty (<0.6)')
axes[1].axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='Medium certainty (<0.7)')
axes[1].set_xlabel('Group ID', fontsize=12)
axes[1].set_ylabel('Assignment Certainty', fontsize=12)
axes[1].set_title('Cluster Assignment Certainty', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / 'cluster_assignments_data.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Rank plots for convergence
print("  5. Rank plots (MCMC convergence)...")
az.plot_rank(idata, var_names=['pi', 'mu', 'sigma'], figsize=(12, 10))
plt.suptitle('Rank Plots: Assessing MCMC Convergence', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(plots_dir / 'rank_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# 6. Cluster separation visualization
print("  6. Cluster separation...")
sep_12 = mu_samples[:, 1] - mu_samples[:, 0]
sep_23 = mu_samples[:, 2] - mu_samples[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(sep_12, bins=30, alpha=0.6, label='mu[2] - mu[1]', density=True, color='blue')
ax.hist(sep_23, bins=30, alpha=0.6, label='mu[3] - mu[2]', density=True, color='green')
ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5 logits)')
ax.set_xlabel('Separation (logits)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Cluster Separation (Logit Scale)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'cluster_separation.png', dpi=150, bbox_inches='tight')
plt.close()

# 7. K_effective distribution
print("  7. Effective number of clusters...")
pi_samples = idata.posterior['pi'].values.reshape(-1, 3)
entropy = -np.sum(pi_samples * np.log(pi_samples + 1e-10), axis=1)
K_eff = np.exp(entropy)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(K_eff, bins=30, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(2, color='red', linestyle='--', linewidth=2, label='Threshold (K=2)')
ax.axvline(K_eff.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean = {K_eff.mean():.2f}')
ax.set_xlabel('Effective Number of Clusters', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of K_effective', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'k_effective.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll diagnostic plots saved to:", plots_dir)
print("\nPlots created:")
print("  - trace_all_params.png: Trace plots for all parameters")
print("  - cluster_parameters.png: Posterior distributions of cluster parameters")
print("  - cluster_assignments_heatmap.png: Assignment probability heatmap")
print("  - cluster_assignments_data.png: Assignments with observed data")
print("  - rank_plots.png: Rank plots for convergence assessment")
print("  - cluster_separation.png: Distribution of cluster separations")
print("  - k_effective.png: Effective number of clusters")
