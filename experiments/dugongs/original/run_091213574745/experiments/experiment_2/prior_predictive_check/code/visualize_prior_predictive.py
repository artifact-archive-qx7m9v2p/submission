"""
Visualization for Prior Predictive Check - Experiment 2 (Student-t Model)

Creates diagnostic plots to assess:
1. Parameter plausibility (marginals and pairs for beta_0, beta_1, sigma, nu)
2. Nu-specific diagnostics (how df affects tail behavior)
3. Prior predictive coverage (curves overlay)
4. Data range diagnostics
5. Comparison to Normal (if needed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Configuration
OUTPUT_DIR = "/workspace/experiments/experiment_2/prior_predictive_check"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Load samples
print("Loading prior samples...")
data = np.load(f"{OUTPUT_DIR}/code/prior_samples.npz")
beta_0 = data['beta_0']
beta_1 = data['beta_1']
sigma = data['sigma']
nu = data['nu']
y_pred = data['y_pred']
x = data['x']
y_obs = data['y_obs']
log_x = data['log_x']

N_DRAWS = len(beta_0)
n = len(x)

print(f"Loaded {N_DRAWS} prior draws with {n} data points each")

# ============================================================================
# Plot 1: Parameter Plausibility (4 parameters + pairs)
# ============================================================================
print("\nCreating Plot 1: Parameter plausibility...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

params = [beta_0, beta_1, sigma, nu]
param_names = [r'$\beta_0$', r'$\beta_1$', r'$\sigma$', r'$\nu$']
param_labels = ['beta_0', 'beta_1', 'sigma', 'nu']

# Diagonal: marginal distributions
for i in range(4):
    ax = fig.add_subplot(gs[i, i])
    ax.hist(params[i], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(params[i].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={params[i].mean():.2f}')
    ax.set_xlabel(param_names[i], fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# Off-diagonal: scatter plots
for i in range(4):
    for j in range(4):
        if i > j:  # Lower triangle
            ax = fig.add_subplot(gs[i, j])
            ax.scatter(params[j], params[i], alpha=0.2, s=10, color='steelblue')
            ax.set_xlabel(param_names[j], fontsize=10)
            ax.set_ylabel(param_names[i], fontsize=10)
            ax.grid(alpha=0.3)

# Upper triangle: correlation + key stats
for i in range(4):
    for j in range(4):
        if i < j:  # Upper triangle
            ax = fig.add_subplot(gs[i, j])
            ax.axis('off')
            corr = np.corrcoef(params[i], params[j])[0, 1]
            text = f'Corr: {corr:.3f}\n\n'
            text += f'{param_labels[i]}:\n'
            text += f'  Mean: {params[i].mean():.3f}\n'
            text += f'  SD: {params[i].std():.3f}\n\n'
            text += f'{param_labels[j]}:\n'
            text += f'  Mean: {params[j].mean():.3f}\n'
            text += f'  SD: {params[j].std():.3f}'
            ax.text(0.1, 0.5, text, fontsize=9, family='monospace',
                   verticalalignment='center')

plt.suptitle('Prior Parameter Distributions (Student-t Model)', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f"{PLOTS_DIR}/parameter_plausibility.png", bbox_inches='tight')
plt.close()
print(f"  Saved: parameter_plausibility.png")

# ============================================================================
# Plot 2: Nu-Specific Diagnostic (tail behavior)
# ============================================================================
print("\nCreating Plot 2: Nu diagnostic (tail behavior)...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel 1: Nu marginal distribution with regime annotations
ax = axes[0, 0]
ax.hist(nu, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(nu.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={nu.mean():.1f}')
ax.axvline(np.median(nu), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(nu):.1f}')

# Add regime shading
ax.axvspan(0, 5, alpha=0.15, color='red', label='Very heavy (nu<5)')
ax.axvspan(5, 20, alpha=0.15, color='orange', label='Heavy (5-20)')
ax.axvspan(20, 30, alpha=0.15, color='yellow', label='Moderate (20-30)')
ax.axvspan(30, ax.get_xlim()[1], alpha=0.15, color='green', label='Near-Normal (>30)')

ax.set_xlabel(r'Degrees of Freedom ($\nu$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Prior Distribution of Nu', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Panel 2: Theoretical tail comparison (PDF)
ax = axes[0, 1]
x_plot = np.linspace(-5, 5, 200)
nu_examples = [3, 10, 20, 50]
colors = ['red', 'orange', 'gold', 'green']

for nu_val, color in zip(nu_examples, colors):
    pdf = stats.t.pdf(x_plot, nu_val)
    ax.plot(x_plot, pdf, linewidth=2, color=color, label=f'nu={nu_val}')

# Compare to Normal
pdf_normal = stats.norm.pdf(x_plot)
ax.plot(x_plot, pdf_normal, linewidth=2, color='blue', linestyle='--', label='Normal')

ax.set_xlabel('Standardized Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Student-t PDF: Tail Heaviness by Nu', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_yscale('log')
ax.grid(alpha=0.3)

# Panel 3: Theoretical tail comparison (survival function)
ax = axes[0, 2]
x_plot_tail = np.linspace(0, 5, 100)

for nu_val, color in zip(nu_examples, colors):
    sf = 2 * stats.t.sf(x_plot_tail, nu_val)  # Two-tailed
    ax.plot(x_plot_tail, sf, linewidth=2, color=color, label=f'nu={nu_val}')

sf_normal = 2 * stats.norm.sf(x_plot_tail)
ax.plot(x_plot_tail, sf_normal, linewidth=2, color='blue', linestyle='--', label='Normal')

ax.set_xlabel('|z| (Standardized)', fontsize=12)
ax.set_ylabel('P(|Z| > |z|)', fontsize=12)
ax.set_title('Tail Probability: Heavy vs Normal', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_yscale('log')
ax.grid(alpha=0.3)

# Panel 4: Prior predictive extremes by nu category
ax = axes[1, 0]

# Categorize draws by nu
nu_cats = []
nu_labels = []
nu_colors = []

very_heavy_idx = nu < 5
heavy_idx = (nu >= 5) & (nu < 20)
moderate_idx = (nu >= 20) & (nu < 30)
near_normal_idx = nu >= 30

if very_heavy_idx.sum() > 0:
    nu_cats.append(very_heavy_idx)
    nu_labels.append(f'Nu<5 (n={very_heavy_idx.sum()})')
    nu_colors.append('red')

if heavy_idx.sum() > 0:
    nu_cats.append(heavy_idx)
    nu_labels.append(f'Nu 5-20 (n={heavy_idx.sum()})')
    nu_colors.append('orange')

if moderate_idx.sum() > 0:
    nu_cats.append(moderate_idx)
    nu_labels.append(f'Nu 20-30 (n={moderate_idx.sum()})')
    nu_colors.append('gold')

if near_normal_idx.sum() > 0:
    nu_cats.append(near_normal_idx)
    nu_labels.append(f'Nu>30 (n={near_normal_idx.sum()})')
    nu_colors.append('green')

# Compute max absolute deviation from mean for each draw
y_pred_maxabs = np.max(np.abs(y_pred - y_pred.mean(axis=1, keepdims=True)), axis=1)

positions = []
data_to_plot = []
for idx in nu_cats:
    positions.append(len(positions) + 1)
    data_to_plot.append(y_pred_maxabs[idx])

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                patch_artist=True, labels=nu_labels)

for patch, color in zip(bp['boxes'], nu_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Max |Y - Mean(Y)| in Dataset', fontsize=12)
ax.set_title('Extremes by Nu Category', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# Panel 5: Prior predictive range by nu category
ax = axes[1, 1]

y_pred_range = y_pred.max(axis=1) - y_pred.min(axis=1)

data_to_plot = []
for idx in nu_cats:
    data_to_plot.append(y_pred_range[idx])

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                patch_artist=True, labels=nu_labels)

for patch, color in zip(bp['boxes'], nu_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add observed range
ax.axhline(y_obs.max() - y_obs.min(), color='blue', linestyle='--', linewidth=2,
          label=f'Observed={y_obs.max() - y_obs.min():.2f}')

ax.set_ylabel('Range(Y) in Dataset', fontsize=12)
ax.set_title('Data Range by Nu Category', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# Panel 6: Scatter of nu vs max extreme value
ax = axes[1, 2]
ax.scatter(nu, y_pred_maxabs, alpha=0.3, s=20, color='steelblue')
ax.set_xlabel(r'Degrees of Freedom ($\nu$)', fontsize=12)
ax.set_ylabel('Max |Y - Mean(Y)|', fontsize=12)
ax.set_title('Extremes vs Nu', fontsize=12, fontweight='bold')
ax.axhline(3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='3 SD threshold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle('Nu Diagnostic: Tail Behavior Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/nu_tail_behavior_diagnostic.png", bbox_inches='tight')
plt.close()
print(f"  Saved: nu_tail_behavior_diagnostic.png")

# ============================================================================
# Plot 3: Prior Predictive Coverage
# ============================================================================
print("\nCreating Plot 3: Prior predictive coverage...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot 100 random prior curves
n_curves = 100
curve_indices = np.random.choice(N_DRAWS, n_curves, replace=False)

for idx in curve_indices:
    mu = beta_0[idx] + beta_1[idx] * log_x
    alpha_val = 0.02 if nu[idx] < 20 else 0.01  # Highlight heavy-tailed ones
    color = 'red' if nu[idx] < 5 else ('orange' if nu[idx] < 20 else 'gray')
    ax.plot(x, mu, alpha=alpha_val, linewidth=1, color=color)

# Plot observed data
ax.scatter(x, y_obs, color='blue', s=60, zorder=10, edgecolor='black', linewidth=1,
          label=f'Observed Data (n={n})')

# Add envelope
y_pred_025 = np.percentile(y_pred, 2.5, axis=0)
y_pred_975 = np.percentile(y_pred, 97.5, axis=0)
ax.fill_between(x, y_pred_025, y_pred_975, alpha=0.15, color='green',
                label='95% Prior Predictive Interval')

ax.set_xlabel('x (Predictor)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y (Response)', fontsize=14, fontweight='bold')
ax.set_title('Prior Predictive Coverage: 100 Random Curves\n(Red: nu<5, Orange: nu 5-20, Gray: nu>20)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/prior_predictive_coverage.png", bbox_inches='tight')
plt.close()
print(f"  Saved: prior_predictive_coverage.png")

# ============================================================================
# Plot 4: Data Range Diagnostic
# ============================================================================
print("\nCreating Plot 4: Data range diagnostic...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Min values
ax = axes[0]
y_pred_min = y_pred.min(axis=1)
ax.hist(y_pred_min, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(y_obs.min(), color='red', linestyle='--', linewidth=2,
          label=f'Observed Min={y_obs.min():.2f}')
ax.axvline(y_pred_min.mean(), color='orange', linestyle='--', linewidth=2,
          label=f'Prior Mean={y_pred_min.mean():.2f}')
ax.set_xlabel('Min(Y) in Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of Minimum Values', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: Max values
ax = axes[1]
y_pred_max = y_pred.max(axis=1)
ax.hist(y_pred_max, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(y_obs.max(), color='red', linestyle='--', linewidth=2,
          label=f'Observed Max={y_obs.max():.2f}')
ax.axvline(y_pred_max.mean(), color='orange', linestyle='--', linewidth=2,
          label=f'Prior Mean={y_pred_max.mean():.2f}')
ax.set_xlabel('Max(Y) in Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of Maximum Values', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 3: Range
ax = axes[2]
y_pred_range = y_pred_max - y_pred_min
ax.hist(y_pred_range, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(y_obs.max() - y_obs.min(), color='red', linestyle='--', linewidth=2,
          label=f'Observed Range={y_obs.max() - y_obs.min():.2f}')
ax.axvline(y_pred_range.mean(), color='orange', linestyle='--', linewidth=2,
          label=f'Prior Mean={y_pred_range.mean():.2f}')
ax.set_xlabel('Range(Y) in Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Distribution of Data Range', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('Data Range Diagnostic', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/data_range_diagnostic.png", bbox_inches='tight')
plt.close()
print(f"  Saved: data_range_diagnostic.png")

# ============================================================================
# Plot 5: Slope and Scale Diagnostics
# ============================================================================
print("\nCreating Plot 5: Slope and scale diagnostics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Slope sign distribution
ax = axes[0]
ax.hist(beta_1, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='-', linewidth=2, label='Zero slope')
ax.axvline(beta_1.mean(), color='orange', linestyle='--', linewidth=2,
          label=f'Mean={beta_1.mean():.3f}')

# Shade negative region
ax.axvspan(ax.get_xlim()[0], 0, alpha=0.2, color='red',
          label=f'Negative ({100*np.mean(beta_1<0):.1f}%)')

ax.set_xlabel(r'$\beta_1$ (Slope)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Prior Distribution of Slope', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: Sigma distribution
ax = axes[1]
ax.hist(sigma, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(sigma.mean(), color='orange', linestyle='--', linewidth=2,
          label=f'Mean={sigma.mean():.3f}')
ax.axvline(np.median(sigma), color='green', linestyle='--', linewidth=2,
          label=f'Median={np.median(sigma):.3f}')

# Add reference to observed RMSE (from Model 1)
ax.axvline(0.087, color='red', linestyle='--', linewidth=2,
          label='Model 1 RMSE=0.087')

ax.set_xlabel(r'$\sigma$ (Scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Prior Distribution of Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('Slope and Scale Diagnostics', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/slope_scale_diagnostic.png", bbox_inches='tight')
plt.close()
print(f"  Saved: slope_scale_diagnostic.png")

# ============================================================================
# Plot 6: Example Prior Predictive Datasets
# ============================================================================
print("\nCreating Plot 6: Example datasets...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Select 6 diverse examples based on nu
nu_sorted_idx = np.argsort(nu)
example_indices = [
    nu_sorted_idx[int(0.01 * N_DRAWS)],  # Very heavy
    nu_sorted_idx[int(0.10 * N_DRAWS)],  # Heavy
    nu_sorted_idx[int(0.30 * N_DRAWS)],  # Moderate-heavy
    nu_sorted_idx[int(0.50 * N_DRAWS)],  # Median
    nu_sorted_idx[int(0.75 * N_DRAWS)],  # Moderate-light
    nu_sorted_idx[int(0.95 * N_DRAWS)],  # Near-Normal
]

for i, idx in enumerate(example_indices):
    ax = axes[i]

    # Compute mean curve
    mu = beta_0[idx] + beta_1[idx] * log_x

    # Plot mean curve
    ax.plot(x, mu, color='red', linewidth=2, label='Mean function')

    # Plot simulated data
    ax.scatter(x, y_pred[idx], alpha=0.6, s=40, color='steelblue', edgecolor='black',
              linewidth=0.5, label='Simulated Y')

    # Add reference to observed
    ax.scatter(x, y_obs, alpha=0.3, s=20, color='green', marker='x',
              label='Observed (ref)')

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    title = f'beta_0={beta_0[idx]:.2f}, beta_1={beta_1[idx]:.2f}\n'
    title += f'sigma={sigma[idx]:.3f}, nu={nu[idx]:.1f}'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

plt.suptitle('Example Prior Predictive Datasets (Diverse Nu Values)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/example_datasets.png", bbox_inches='tight')
plt.close()
print(f"  Saved: example_datasets.png")

# ============================================================================
# Plot 7: Comparison to Normal Likelihood (Model 1)
# ============================================================================
print("\nCreating Plot 7: Student-t vs Normal comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Simulate Normal likelihood for comparison
np.random.seed(123)  # Different seed for comparison
n_compare = 500

beta_0_normal = np.random.normal(2.3, 0.3, n_compare)  # Model 1 prior
beta_1_normal = np.random.normal(0.29, 0.15, n_compare)
sigma_normal = np.random.exponential(0.1, n_compare)

y_pred_normal = np.zeros((n_compare, n))
for i in range(n_compare):
    mu = beta_0_normal[i] + beta_1_normal[i] * log_x
    y_pred_normal[i, :] = np.random.normal(mu, sigma_normal[i])

# Panel 1: Range comparison
ax = axes[0, 0]
range_studentt = y_pred.max(axis=1) - y_pred.min(axis=1)
range_normal = y_pred_normal.max(axis=1) - y_pred_normal.min(axis=1)

ax.hist(range_normal, bins=30, alpha=0.5, color='blue', edgecolor='black',
       label=f'Normal (mean={range_normal.mean():.2f})')
ax.hist(range_studentt[:500], bins=30, alpha=0.5, color='red', edgecolor='black',
       label=f'Student-t (mean={range_studentt.mean():.2f})')
ax.axvline(y_obs.max() - y_obs.min(), color='green', linestyle='--', linewidth=2,
          label=f'Observed={y_obs.max() - y_obs.min():.2f}')

ax.set_xlabel('Range(Y)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Data Range: Student-t vs Normal', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: Maximum absolute deviation
ax = axes[0, 1]
maxabs_studentt = np.max(np.abs(y_pred - y_pred.mean(axis=1, keepdims=True)), axis=1)
maxabs_normal = np.max(np.abs(y_pred_normal - y_pred_normal.mean(axis=1, keepdims=True)), axis=1)

ax.hist(maxabs_normal, bins=30, alpha=0.5, color='blue', edgecolor='black',
       label=f'Normal (mean={maxabs_normal.mean():.2f})')
ax.hist(maxabs_studentt[:500], bins=30, alpha=0.5, color='red', edgecolor='black',
       label=f'Student-t (mean={maxabs_studentt.mean():.2f})')

ax.set_xlabel('Max |Y - Mean(Y)|', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Extremes: Student-t vs Normal', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 3: Q-Q plot of extremes
ax = axes[1, 0]
q_normal = np.percentile(maxabs_normal, np.linspace(0, 100, 100))
q_studentt = np.percentile(maxabs_studentt, np.linspace(0, 100, 100))

ax.plot(q_normal, q_studentt, 'o', alpha=0.5, color='steelblue')
ax.plot([q_normal.min(), q_normal.max()], [q_normal.min(), q_normal.max()],
       'r--', linewidth=2, label='y=x')

ax.set_xlabel('Normal Quantiles (Max Extremes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Student-t Quantiles (Max Extremes)', fontsize=12, fontweight='bold')
ax.set_title('Q-Q Plot: Heavier Tails in Student-t', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 4: Tail probability comparison
ax = axes[1, 1]
thresholds = np.linspace(0, 5, 50)
tail_prob_normal = [np.mean(maxabs_normal > t) for t in thresholds]
tail_prob_studentt = [np.mean(maxabs_studentt > t) for t in thresholds]

ax.plot(thresholds, tail_prob_normal, linewidth=2, color='blue', label='Normal')
ax.plot(thresholds, tail_prob_studentt, linewidth=2, color='red', label='Student-t')

ax.set_xlabel('Threshold (Max |Y - Mean|)', fontsize=12, fontweight='bold')
ax.set_ylabel('P(Extreme > Threshold)', fontsize=12, fontweight='bold')
ax.set_title('Tail Probability: Student-t More Heavy-Tailed', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(alpha=0.3)

plt.suptitle('Student-t vs Normal Likelihood Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/studentt_vs_normal_comparison.png", bbox_inches='tight')
plt.close()
print(f"  Saved: studentt_vs_normal_comparison.png")

print("\n" + "="*70)
print("All visualizations complete!")
print("="*70)
print(f"\nPlots saved to: {PLOTS_DIR}/")
print("\nGenerated files:")
print("  1. parameter_plausibility.png")
print("  2. nu_tail_behavior_diagnostic.png")
print("  3. prior_predictive_coverage.png")
print("  4. data_range_diagnostic.png")
print("  5. slope_scale_diagnostic.png")
print("  6. example_datasets.png")
print("  7. studentt_vs_normal_comparison.png")
