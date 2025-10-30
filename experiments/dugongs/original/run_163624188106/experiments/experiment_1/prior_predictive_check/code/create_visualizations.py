"""
Create visualizations for prior predictive check
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load results
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check/plots"
results = np.load('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.npz')

alpha_samples = results['alpha_samples']
beta_samples = results['beta_samples']
sigma_samples = results['sigma_samples']
y_pred_samples = results['y_pred_samples']
y_eval_samples = results['y_eval_samples']
x_eval = results['x_eval']
intercept_implied = results['intercept_implied']
exponent_implied = results['exponent_implied']
x_obs = results['x_obs']
y_obs = results['y_obs']

N_PRIOR_SAMPLES = len(alpha_samples)
n_obs = len(x_obs)

# ============================================================================
# Plot 1: Prior distributions for parameters
# ============================================================================
print("Creating parameter_plausibility.png...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Prior Distributions and Parameter Plausibility', fontsize=14, fontweight='bold')

# Alpha
ax = axes[0, 0]
ax.hist(alpha_samples, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
x_range = np.linspace(alpha_samples.min(), alpha_samples.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, 0.6, 0.3), 'r-', linewidth=2, label='Prior density')
ax.axvline(np.median(alpha_samples), color='darkblue', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('alpha (log-scale intercept)', fontweight='bold')
ax.set_ylabel('Density')
ax.set_title('alpha ~ Normal(0.6, 0.3)')
ax.legend()
ax.grid(alpha=0.3)

# Beta
ax = axes[0, 1]
ax.hist(beta_samples, bins=50, density=True, alpha=0.6, color='forestgreen', edgecolor='black')
x_range = np.linspace(beta_samples.min(), beta_samples.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, 0.13, 0.1), 'r-', linewidth=2, label='Prior density')
ax.axvline(np.median(beta_samples), color='darkgreen', linestyle='--', linewidth=2, label='Median')
ax.axvline(0.13, color='orange', linestyle=':', linewidth=2, label='EDA estimate')
ax.set_xlabel('beta (power law exponent)', fontweight='bold')
ax.set_ylabel('Density')
ax.set_title('beta ~ Normal(0.13, 0.1)')
ax.legend()
ax.grid(alpha=0.3)

# Sigma
ax = axes[0, 2]
ax.hist(sigma_samples, bins=50, density=True, alpha=0.6, color='coral', edgecolor='black')
x_range = np.linspace(0, sigma_samples.max(), 100)
# Half-normal PDF
ax.plot(x_range, stats.halfnorm.pdf(x_range, scale=0.1), 'r-', linewidth=2, label='Prior density')
ax.axvline(np.median(sigma_samples), color='darkred', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('sigma (log-scale SD)', fontweight='bold')
ax.set_ylabel('Density')
ax.set_title('sigma ~ Half-Normal(0.1)')
ax.legend()
ax.grid(alpha=0.3)

# Implied intercept: A = exp(alpha)
ax = axes[1, 0]
ax.hist(intercept_implied, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(np.median(intercept_implied), color='darkblue', linestyle='--', linewidth=2, label='Median')
ax.axvline(1.82, color='orange', linestyle=':', linewidth=2, label='EDA estimate (1.82)')
ax.set_xlabel('A = exp(alpha)', fontweight='bold')
ax.set_ylabel('Density')
ax.set_title('Implied Intercept: Y = A * x^beta')
ax.legend()
ax.grid(alpha=0.3)

# Implied exponent: same as beta
ax = axes[1, 1]
ax.hist(exponent_implied, bins=50, density=True, alpha=0.6, color='forestgreen', edgecolor='black')
ax.axvline(np.median(exponent_implied), color='darkgreen', linestyle='--', linewidth=2, label='Median')
ax.axvline(0.13, color='orange', linestyle=':', linewidth=2, label='EDA estimate (0.13)')
ax.set_xlabel('B = beta', fontweight='bold')
ax.set_ylabel('Density')
ax.set_title('Implied Exponent: Y = A * x^B')
ax.legend()
ax.grid(alpha=0.3)

# Joint alpha-beta
ax = axes[1, 2]
ax.scatter(alpha_samples, beta_samples, alpha=0.3, s=10, color='purple')
ax.axhline(0.13, color='orange', linestyle=':', linewidth=2, label='EDA beta')
ax.axvline(np.log(1.82), color='orange', linestyle=':', linewidth=1.5, label='EDA alpha')
ax.set_xlabel('alpha', fontweight='bold')
ax.set_ylabel('beta', fontweight='bold')
ax.set_title('Joint Prior: alpha vs beta')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/parameter_plausibility.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/parameter_plausibility.png")
plt.close()

# ============================================================================
# Plot 2: Prior predictive coverage - trajectories on observed scatter
# ============================================================================
print("Creating prior_predictive_coverage.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prior Predictive Coverage of Observed Data', fontsize=14, fontweight='bold')

# Left panel: Sample of prior predictive trajectories
ax = axes[0]
# Plot subset of prior predictive samples
n_show = 100
for i in range(n_show):
    idx = np.random.randint(0, N_PRIOR_SAMPLES)
    ax.plot(x_obs, y_pred_samples[idx, :], alpha=0.15, color='gray', linewidth=0.5)

# Overlay observed data
ax.scatter(x_obs, y_obs, color='red', s=80, zorder=10, edgecolor='darkred', linewidth=1.5, label='Observed data')

# Add percentile bands
y_pred_median = np.median(y_pred_samples, axis=0)
y_pred_025 = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_975 = np.percentile(y_pred_samples, 97.5, axis=0)

ax.plot(x_obs, y_pred_median, color='blue', linewidth=2, label='Prior predictive median', zorder=5)
ax.fill_between(x_obs, y_pred_025, y_pred_975, alpha=0.2, color='blue', label='95% prior interval')

ax.set_xlabel('x', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_title(f'Prior Predictive Trajectories (showing {n_show}/{N_PRIOR_SAMPLES})')
ax.legend()
ax.grid(alpha=0.3)

# Right panel: Distribution of predictions at specific x values
ax = axes[1]
positions = []
labels = []
data_to_plot = []

for j, x_val in enumerate([1.0, 10.0, 30.0]):
    y_vals = y_eval_samples[:, j]
    data_to_plot.append(y_vals)
    positions.append(j)
    labels.append(f'x={x_val}')

    # Find closest observed value
    closest_idx = np.argmin(np.abs(x_obs - x_val))
    closest_y = y_obs[closest_idx]
    ax.scatter(j, closest_y, color='red', s=100, zorder=10, marker='*', edgecolor='darkred', linewidth=1.5)

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                showfliers=False, medianprops=dict(color='darkblue', linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel('Y', fontweight='bold')
ax.set_title('Prior Predictions at Key x Values\n(red star = nearest observed)')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/prior_predictive_coverage.png")
plt.close()

# ============================================================================
# Plot 3: Range and scale diagnostics
# ============================================================================
print("Creating range_scale_diagnostics.png...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Range and Scale Diagnostics', fontsize=14, fontweight='bold')

# Dataset minimums
ax = axes[0, 0]
y_pred_min = y_pred_samples.min(axis=1)
ax.hist(y_pred_min, bins=50, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(y_obs.min(), color='red', linestyle='--', linewidth=2, label=f'Observed min ({y_obs.min():.2f})')
ax.axvline(np.median(y_pred_min), color='darkblue', linestyle=':', linewidth=2, label='Prior median')
ax.set_xlabel('Dataset minimum', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of Minimum Y Values')
ax.legend()
ax.grid(alpha=0.3)

# Dataset maximums
ax = axes[0, 1]
y_pred_max = y_pred_samples.max(axis=1)
ax.hist(y_pred_max, bins=50, alpha=0.6, color='forestgreen', edgecolor='black')
ax.axvline(y_obs.max(), color='red', linestyle='--', linewidth=2, label=f'Observed max ({y_obs.max():.2f})')
ax.axvline(np.median(y_pred_max), color='darkgreen', linestyle=':', linewidth=2, label='Prior median')
ax.set_xlabel('Dataset maximum', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of Maximum Y Values')
ax.legend()
ax.grid(alpha=0.3)

# Dataset means
ax = axes[1, 0]
y_pred_mean = y_pred_samples.mean(axis=1)
ax.hist(y_pred_mean, bins=50, alpha=0.6, color='coral', edgecolor='black')
ax.axvline(y_obs.mean(), color='red', linestyle='--', linewidth=2, label=f'Observed mean ({y_obs.mean():.2f})')
ax.axvline(np.median(y_pred_mean), color='darkred', linestyle=':', linewidth=2, label='Prior median')
# Add shaded region for +/- 2 SD
ax.axvspan(y_obs.mean() - 2*y_obs.std(), y_obs.mean() + 2*y_obs.std(),
           alpha=0.2, color='yellow', label='Obs mean +/- 2 SD')
ax.set_xlabel('Dataset mean', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of Mean Y Values')
ax.legend()
ax.grid(alpha=0.3)

# Dataset ranges (max - min)
ax = axes[1, 1]
y_pred_range = y_pred_max - y_pred_min
obs_range = y_obs.max() - y_obs.min()
ax.hist(y_pred_range, bins=50, alpha=0.6, color='purple', edgecolor='black')
ax.axvline(obs_range, color='red', linestyle='--', linewidth=2, label=f'Observed range ({obs_range:.2f})')
ax.axvline(np.median(y_pred_range), color='indigo', linestyle=':', linewidth=2, label='Prior median')
ax.set_xlabel('Dataset range (max - min)', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of Y Value Ranges')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/range_scale_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/range_scale_diagnostics.png")
plt.close()

# ============================================================================
# Plot 4: Extreme value check
# ============================================================================
print("Creating extreme_value_diagnostics.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Extreme Value and Computational Diagnostics', fontsize=14, fontweight='bold')

# Left: Distribution of all predicted Y values
ax = axes[0]
y_all = y_pred_samples.flatten()

# Create histogram with log scale for better visibility
ax.hist(y_all, bins=100, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(y_obs.min(), color='red', linestyle='--', linewidth=2, label='Observed range')
ax.axvline(y_obs.max(), color='red', linestyle='--', linewidth=2)
ax.axvspan(y_obs.min(), y_obs.max(), alpha=0.2, color='red', label='Observed Y range')

# Mark regions of concern
ax.axvline(0, color='darkred', linestyle=':', linewidth=2, label='Zero')
ax.axvline(100, color='orange', linestyle=':', linewidth=2, label='Extreme (>100)')

ax.set_xlabel('Y (all prior predictions)', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of All Prior Predictive Y Values')
ax.set_xlim([-0.5, min(20, y_all.max())])  # Focus on reasonable range
ax.legend()
ax.grid(alpha=0.3)

# Right: Count of issues per dataset
ax = axes[1]
n_negative = np.sum(y_pred_samples < 0, axis=1)
n_extreme = np.sum(y_pred_samples > 100, axis=1)
n_very_small = np.sum(y_pred_samples < 0.1, axis=1)

categories = ['Y < 0\n(negative)', 'Y > 100\n(extreme)', 'Y < 0.1\n(very small)']
counts = [
    np.sum(n_negative > 0),
    np.sum(n_extreme > 0),
    np.sum(n_very_small > 0)
]
percentages = [c / N_PRIOR_SAMPLES * 100 for c in counts]

bars = ax.bar(categories, percentages, color=['darkred', 'orange', 'gold'], alpha=0.7, edgecolor='black')
ax.axhline(5, color='red', linestyle='--', linewidth=2, label='5% threshold')
ax.set_ylabel('% of datasets with issues', fontweight='bold')
ax.set_title('Datasets with Pathological Values')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, pct) in enumerate(zip(bars, percentages)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%\n({counts[i]} datasets)',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/extreme_value_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/extreme_value_diagnostics.png")
plt.close()

# ============================================================================
# Plot 5: Comparison with EDA-derived power law
# ============================================================================
print("Creating eda_comparison.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparison with EDA-Derived Power Law (Y = 1.82 * x^0.13)', fontsize=14, fontweight='bold')

# Left: Prior predictive vs EDA curve
ax = axes[0]

# Plot EDA power law
x_smooth = np.linspace(x_obs.min(), x_obs.max(), 100)
y_eda = 1.82 * x_smooth ** 0.13
ax.plot(x_smooth, y_eda, 'orange', linewidth=3, label='EDA: Y = 1.82 * x^0.13', zorder=10)

# Plot prior predictive percentiles
y_pred_median = np.median(y_pred_samples, axis=0)
y_pred_025 = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_975 = np.percentile(y_pred_samples, 97.5, axis=0)

ax.plot(x_obs, y_pred_median, color='blue', linewidth=2, label='Prior predictive median', zorder=5)
ax.fill_between(x_obs, y_pred_025, y_pred_975, alpha=0.2, color='blue', label='95% prior interval')

# Overlay observed data
ax.scatter(x_obs, y_obs, color='red', s=60, zorder=15, edgecolor='darkred', linewidth=1, label='Observed data')

ax.set_xlabel('x', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_title('Prior Predictive vs EDA Power Law')
ax.legend()
ax.grid(alpha=0.3)

# Right: Parameter comparison
ax = axes[1]

# Create violin plots
data_intercept = [intercept_implied, [1.82]]
data_exponent = [exponent_implied, [0.13]]

positions = [0, 1, 3, 4]
colors = ['lightblue', 'orange', 'lightgreen', 'orange']
labels_x = ['Prior\nIntercept', 'EDA\nIntercept', 'Prior\nExponent', 'EDA\nExponent']

parts1 = ax.violinplot([intercept_implied], positions=[0], widths=0.7, showmeans=True, showmedians=True)
for pc in parts1['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.6)

ax.scatter([1], [1.82], color='orange', s=200, marker='*', edgecolor='darkorange', linewidth=2, zorder=10)

parts2 = ax.violinplot([exponent_implied], positions=[3], widths=0.7, showmeans=True, showmedians=True)
for pc in parts2['bodies']:
    pc.set_facecolor('lightgreen')
    pc.set_alpha(0.6)

ax.scatter([4], [0.13], color='orange', s=200, marker='*', edgecolor='darkorange', linewidth=2, zorder=10)

ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(['Intercept (A)', 'Exponent (B)'])
ax.set_ylabel('Parameter Value', fontweight='bold')
ax.set_title('Prior vs EDA Parameter Estimates')
ax.axvline(2, color='gray', linestyle='--', linewidth=1)
ax.grid(alpha=0.3, axis='y')

# Add annotations
ax.text(0, np.median(intercept_implied), f'{np.median(intercept_implied):.2f}',
        ha='left', va='center', fontweight='bold', fontsize=9)
ax.text(1, 1.82, '1.82', ha='right', va='center', fontweight='bold', fontsize=9, color='darkorange')
ax.text(3, np.median(exponent_implied), f'{np.median(exponent_implied):.3f}',
        ha='left', va='center', fontweight='bold', fontsize=9)
ax.text(4, 0.13, '0.13', ha='right', va='center', fontweight='bold', fontsize=9, color='darkorange')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/eda_comparison.png")
plt.close()

print("\nAll visualizations created successfully!")
