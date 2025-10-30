"""
Visualization for Prior Predictive Check - Log-Log Power Law Model

Creates diagnostic plots to assess prior plausibility.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load results
print("Loading prior predictive check results...")
prior_samples = pd.read_csv('/workspace/experiments/experiment_3/prior_predictive_check/code/prior_samples.csv')
data = np.load('/workspace/experiments/experiment_3/prior_predictive_check/code/prior_predictions.npz')

x_pred = data['x_pred']
y_pred_samples = data['y_pred_samples']
x_obs = data['x_obs']
y_obs = data['y_obs']

N_DRAWS = len(prior_samples)
print(f"Loaded {N_DRAWS} prior draws")

# ============================================================================
# PLOT 1: Parameter Plausibility - Prior distributions
# ============================================================================
print("\nCreating Plot 1: Parameter plausibility...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Alpha
ax = axes[0]
ax.hist(prior_samples['alpha'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0.6, color='red', linestyle='--', linewidth=2, label='Prior mean')
ax.axvline(np.log(y_obs.min()), color='orange', linestyle=':', linewidth=2, label=f'log(Y_min)={np.log(y_obs.min()):.2f}')
ax.axvline(np.log(y_obs.max()), color='orange', linestyle=':', linewidth=2, label=f'log(Y_max)={np.log(y_obs.max()):.2f}')
ax.set_xlabel('α (intercept on log scale)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Prior: α ~ Normal(0.6, 0.3)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Beta
ax = axes[1]
ax.hist(prior_samples['beta'], bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
ax.axvline(0.12, color='red', linestyle='--', linewidth=2, label='Prior mean')
ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Zero (no effect)')
neg_pct = 100 * (prior_samples['beta'] < 0).mean()
ax.text(0.05, 0.95, f'{neg_pct:.1f}% negative', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('β (power law exponent)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Prior: β ~ Normal(0.12, 0.1)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Sigma (with log scale for heavy tail)
ax = axes[2]
# Plot histogram with log-scale y-axis to show tail
sigma_filtered = prior_samples['sigma'][prior_samples['sigma'] < 2.0]  # Focus on reasonable range
ax.hist(sigma_filtered, bins=50, alpha=0.7, color='darkred', edgecolor='black')
ax.axvline(0.1, color='blue', linestyle='--', linewidth=2, label='Scale=0.1')
extreme_pct = 100 * (prior_samples['sigma'] > 1.0).mean()
ax.text(0.65, 0.95, f'{extreme_pct:.1f}% > 1.0\n(heavy tail)', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('σ (residual SD on log scale)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Prior: σ ~ Half-Cauchy(0, 0.1)', fontsize=12, fontweight='bold')
ax.set_xlim([0, 2.0])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_3/prior_predictive_check/plots/parameter_plausibility.png', dpi=300)
print("  Saved: parameter_plausibility.png")
plt.close()

# ============================================================================
# PLOT 2: Prior Predictive Trajectories
# ============================================================================
print("\nCreating Plot 2: Prior predictive coverage...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plausibility bounds
PLAUSIBLE_MIN, PLAUSIBLE_MAX = 0.5, 5.0
ax.axhspan(PLAUSIBLE_MIN, PLAUSIBLE_MAX, alpha=0.15, color='green',
           label=f'Plausible range [{PLAUSIBLE_MIN}, {PLAUSIBLE_MAX}]')

# Plot a sample of trajectories (not all 2000, too cluttered)
n_show = 200
indices = np.random.choice(N_DRAWS, n_show, replace=False)

for i, idx in enumerate(indices):
    y_traj = y_pred_samples[idx, :]

    # Color by plausibility
    all_plausible = np.all((y_traj >= PLAUSIBLE_MIN) & (y_traj <= PLAUSIBLE_MAX))

    if all_plausible:
        color = 'steelblue'
        alpha = 0.15
        zorder = 1
    else:
        color = 'red'
        alpha = 0.3
        zorder = 2

    if i == 0:  # Label only once
        label = 'Plausible' if all_plausible else 'Implausible'
        ax.plot(x_pred, y_traj, color=color, alpha=alpha, linewidth=0.8, zorder=zorder, label=label)
    else:
        ax.plot(x_pred, y_traj, color=color, alpha=alpha, linewidth=0.8, zorder=zorder)

# Overlay observed data
ax.scatter(x_obs, y_obs, color='black', s=60, alpha=0.8, zorder=10,
           label=f'Observed data (N={len(x_obs)})', edgecolor='white', linewidth=1)

# Median and percentiles
median_y = np.median(y_pred_samples, axis=0)
p05 = np.percentile(y_pred_samples, 5, axis=0)
p95 = np.percentile(y_pred_samples, 95, axis=0)

ax.plot(x_pred, median_y, color='darkblue', linewidth=3, label='Median prior prediction', zorder=5)
ax.fill_between(x_pred, p05, p95, alpha=0.2, color='darkblue', label='90% prior interval', zorder=3)

ax.set_xlabel('x', fontsize=13, fontweight='bold')
ax.set_ylabel('Y', fontsize=13, fontweight='bold')
ax.set_title('Prior Predictive Coverage: Do Priors Generate Plausible Data?',
             fontsize=14, fontweight='bold')
ax.set_xlim([0, 36])
ax.set_ylim([0, 12])
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)

# Add text box with summary
n_plausible = np.sum([np.all((y_pred_samples[i, :] >= PLAUSIBLE_MIN) & (y_pred_samples[i, :] <= PLAUSIBLE_MAX))
                       for i in range(N_DRAWS)])
pct_plausible = 100 * n_plausible / N_DRAWS

summary_text = f"Trajectory Pass Rate: {pct_plausible:.1f}%\n({n_plausible}/{N_DRAWS} fully plausible)"
ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_3/prior_predictive_check/plots/prior_predictive_coverage.png', dpi=300)
print("  Saved: prior_predictive_coverage.png")
plt.close()

# ============================================================================
# PLOT 3: Behavior Diagnostics - Monotonicity and Shape
# ============================================================================
print("\nCreating Plot 3: Behavior diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Growth patterns
ax = axes[0, 0]
for idx in np.random.choice(N_DRAWS, 100, replace=False):
    y_traj = y_pred_samples[idx, :]

    # Check if monotonic
    is_monotonic = np.all(np.diff(y_traj) >= 0)

    if is_monotonic:
        ax.plot(x_pred, y_traj, color='green', alpha=0.1, linewidth=0.8)
    else:
        ax.plot(x_pred, y_traj, color='red', alpha=0.3, linewidth=0.8)

ax.scatter(x_obs, y_obs, color='black', s=40, alpha=0.8, zorder=10)
ax.set_xlabel('x', fontsize=11, fontweight='bold')
ax.set_ylabel('Y', fontsize=11, fontweight='bold')
ax.set_title('A) Monotonicity Check (Green=Increasing, Red=Not)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim([0, 8])

# Calculate monotonicity rate
monotonic_count = sum([np.all(np.diff(y_pred_samples[i, :]) >= 0) for i in range(N_DRAWS)])
mono_pct = 100 * monotonic_count / N_DRAWS
ax.text(0.05, 0.95, f'{mono_pct:.1f}% monotonic increasing', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Distribution at key x values
ax = axes[0, 1]
key_x_vals = [1.0, 10.0, 20.0, 30.0]
positions = np.arange(len(key_x_vals))
data_to_plot = []

for x_val in key_x_vals:
    idx = np.argmin(np.abs(x_pred - x_val))
    y_at_x = y_pred_samples[:, idx]
    # Filter extreme values for visualization
    y_at_x_filtered = y_at_x[(y_at_x > 0) & (y_at_x < 20)]
    data_to_plot.append(y_at_x_filtered)

parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
ax.axhspan(PLAUSIBLE_MIN, PLAUSIBLE_MAX, alpha=0.15, color='green', zorder=0)
ax.set_xticks(positions)
ax.set_xticklabels([f'x={x}' for x in key_x_vals])
ax.set_ylabel('Y', fontsize=11, fontweight='bold')
ax.set_title('B) Prior Predictions at Key x Values', fontsize=12, fontweight='bold')
ax.set_ylim([0, 10])
ax.grid(alpha=0.3, axis='y')

# Panel C: Beta vs outcome behavior
ax = axes[1, 0]
# Calculate growth factor for each trajectory
idx_x1 = np.argmin(np.abs(x_pred - 1.0))
idx_x30 = np.argmin(np.abs(x_pred - 30.0))
growth_factors = y_pred_samples[:, idx_x30] / y_pred_samples[:, idx_x1]

# Filter reasonable growth factors for visualization
valid_mask = (growth_factors > 0) & (growth_factors < 20)
ax.scatter(prior_samples['beta'][valid_mask], growth_factors[valid_mask],
           alpha=0.3, s=20, color='steelblue')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='β=0 (no effect)')
ax.axhline(1, color='orange', linestyle='--', linewidth=2, label='No growth')
ax.set_xlabel('β (power law exponent)', fontsize=11, fontweight='bold')
ax.set_ylabel('Growth factor (Y at x=30 / Y at x=1)', fontsize=11, fontweight='bold')
ax.set_title('C) Parameter β vs Growth Behavior', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel D: Sigma vs prediction spread
ax = axes[1, 1]
# Calculate spread at x=15 for each trajectory
idx_x15 = np.argmin(np.abs(x_pred - 15.0))
y_at_x15 = y_pred_samples[:, idx_x15]

# For visualization, filter sigma < 2
mask = prior_samples['sigma'] < 2.0
ax.scatter(prior_samples['sigma'][mask], y_at_x15[mask],
           alpha=0.3, s=20, color='darkred')
ax.axhspan(PLAUSIBLE_MIN, PLAUSIBLE_MAX, alpha=0.15, color='green', zorder=0)
ax.set_xlabel('σ (residual SD on log scale)', fontsize=11, fontweight='bold')
ax.set_ylabel('Y at x=15', fontsize=11, fontweight='bold')
ax.set_title('D) σ vs Prediction Variability', fontsize=12, fontweight='bold')
ax.set_ylim([0, 10])
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_3/prior_predictive_check/plots/behavior_diagnostics.png', dpi=300)
print("  Saved: behavior_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 4: Problem Identification - Heavy tails and extreme values
# ============================================================================
print("\nCreating Plot 4: Heavy tail diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Sigma heavy tail on log scale
ax = axes[0, 0]
sigma_vals = prior_samples['sigma']
ax.hist(sigma_vals, bins=100, alpha=0.7, color='darkred', edgecolor='black')
ax.set_yscale('log')
ax.axvline(0.1, color='blue', linestyle='--', linewidth=2, label='Scale=0.1')
ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='Problematic threshold')
ax.set_xlabel('σ', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency (log scale)', fontsize=11, fontweight='bold')
ax.set_title('A) Half-Cauchy Prior Has Heavy Tail', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

extreme_pct = 100 * (sigma_vals > 1.0).mean()
ax.text(0.65, 0.95, f'{extreme_pct:.1f}% > 1.0\nCauses extreme predictions',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Panel B: Effect of large sigma on predictions
ax = axes[0, 1]
# Get indices for small, medium, large sigma
small_sigma_idx = np.where(sigma_vals < 0.2)[0][:50]
large_sigma_idx = np.where(sigma_vals > 1.0)[0][:50]

for idx in small_sigma_idx:
    ax.plot(x_pred, y_pred_samples[idx, :], color='green', alpha=0.3, linewidth=0.8)

for idx in large_sigma_idx:
    y_traj = y_pred_samples[idx, :]
    # Clip for visualization
    y_traj_clip = np.clip(y_traj, 0, 50)
    ax.plot(x_pred, y_traj_clip, color='red', alpha=0.4, linewidth=1.2)

ax.scatter(x_obs, y_obs, color='black', s=40, alpha=0.8, zorder=10)
ax.axhspan(PLAUSIBLE_MIN, PLAUSIBLE_MAX, alpha=0.15, color='lightgreen', zorder=0)
ax.set_xlabel('x', fontsize=11, fontweight='bold')
ax.set_ylabel('Y (clipped at 50 for visualization)', fontsize=11, fontweight='bold')
ax.set_title('B) Green: σ<0.2, Red: σ>1.0', fontsize=12, fontweight='bold')
ax.set_ylim([0, 50])
ax.grid(alpha=0.3)

# Panel C: Negative beta problem
ax = axes[1, 0]
neg_beta_idx = np.where(prior_samples['beta'] < 0)[0][:50]
pos_beta_idx = np.where(prior_samples['beta'] > 0.05)[0][:50]

for idx in pos_beta_idx:
    ax.plot(x_pred, y_pred_samples[idx, :], color='blue', alpha=0.2, linewidth=0.8)

for idx in neg_beta_idx:
    ax.plot(x_pred, y_pred_samples[idx, :], color='red', alpha=0.4, linewidth=1.2)

ax.scatter(x_obs, y_obs, color='black', s=40, alpha=0.8, zorder=10)
ax.axhspan(PLAUSIBLE_MIN, PLAUSIBLE_MAX, alpha=0.15, color='lightgreen', zorder=0)
ax.set_xlabel('x', fontsize=11, fontweight='bold')
ax.set_ylabel('Y', fontsize=11, fontweight='bold')
ax.set_title('C) Blue: β>0 (increasing), Red: β<0 (decreasing)', fontsize=12, fontweight='bold')
ax.set_ylim([0, 8])
ax.grid(alpha=0.3)

neg_pct = 100 * (prior_samples['beta'] < 0).mean()
ax.text(0.65, 0.95, f'{neg_pct:.1f}% have β<0\n(decreasing trend)',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Panel D: Joint distribution beta vs sigma (problem space)
ax = axes[1, 1]
# Create 2D histogram
mask = (sigma_vals < 2.0)  # Focus on visualizable range
h = ax.hist2d(prior_samples['beta'][mask], sigma_vals[mask],
              bins=[40, 40], cmap='YlOrRd', cmin=1)
plt.colorbar(h[3], ax=ax, label='Count')
ax.axvline(0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='β=0')
ax.axhline(0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='σ=0.3 (high)')
ax.set_xlabel('β (power law exponent)', fontsize=11, fontweight='bold')
ax.set_ylabel('σ (residual SD)', fontsize=11, fontweight='bold')
ax.set_title('D) Joint Prior Distribution (β, σ)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_3/prior_predictive_check/plots/heavy_tail_diagnostics.png', dpi=300)
print("  Saved: heavy_tail_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 5: Pointwise plausibility across x range
# ============================================================================
print("\nCreating Plot 5: Pointwise plausibility assessment...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel A: Percentage in plausible range
ax = axes[0]
plausible_pct = np.zeros(len(x_pred))
for i in range(len(x_pred)):
    y_at_x = y_pred_samples[:, i]
    plausible_pct[i] = 100 * np.mean((y_at_x >= PLAUSIBLE_MIN) & (y_at_x <= PLAUSIBLE_MAX))

ax.fill_between(x_pred, plausible_pct, alpha=0.5, color='steelblue')
ax.plot(x_pred, plausible_pct, color='darkblue', linewidth=2)
ax.axhline(80, color='red', linestyle='--', linewidth=2, label='80% threshold (PASS criterion)')
ax.axhline(60, color='orange', linestyle='--', linewidth=2, label='60% threshold')

# Mark observed data range
ax.axvspan(x_obs.min(), x_obs.max(), alpha=0.1, color='green', label='Observed x range')

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('% Predictions in Plausible Range', fontsize=12, fontweight='bold')
ax.set_title('Pointwise Plausibility: % of Prior Predictions in [0.5, 5.0]',
             fontsize=13, fontweight='bold')
ax.set_ylim([0, 100])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel B: Median and percentiles
ax = axes[1]
median_y = np.median(y_pred_samples, axis=0)
p025 = np.percentile(y_pred_samples, 2.5, axis=0)
p975 = np.percentile(y_pred_samples, 97.5, axis=0)
p25 = np.percentile(y_pred_samples, 25, axis=0)
p75 = np.percentile(y_pred_samples, 75, axis=0)

ax.fill_between(x_pred, p025, p975, alpha=0.2, color='steelblue', label='95% interval')
ax.fill_between(x_pred, p25, p75, alpha=0.4, color='steelblue', label='50% interval')
ax.plot(x_pred, median_y, color='darkblue', linewidth=2.5, label='Median')

ax.axhspan(PLAUSIBLE_MIN, PLAUSIBLE_MAX, alpha=0.15, color='green',
           label=f'Plausible range [{PLAUSIBLE_MIN}, {PLAUSIBLE_MAX}]')
ax.scatter(x_obs, y_obs, color='black', s=60, alpha=0.8, zorder=10,
           label='Observed data', edgecolor='white', linewidth=1)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title('Prior Predictive Distribution Summary', fontsize=13, fontweight='bold')
ax.set_ylim([0, 12])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_3/prior_predictive_check/plots/pointwise_plausibility.png', dpi=300)
print("  Saved: pointwise_plausibility.png")
plt.close()

print("\n" + "="*70)
print("All visualizations completed!")
print("="*70)
