"""
Eight Schools Dataset - Comprehensive Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data.csv')

# Create output directory
import os
output_dir = '/workspace/eda/visualizations'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. Forest Plot with Error Bars (Primary Visualization)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

schools = data['school'].values
effects = data['effect'].values
sigmas = data['sigma'].values

# Sort by effect size for better visualization
sort_idx = np.argsort(effects)
schools_sorted = schools[sort_idx]
effects_sorted = effects[sort_idx]
sigmas_sorted = sigmas[sort_idx]

y_pos = np.arange(len(schools_sorted))

# Plot point estimates
ax.plot(effects_sorted, y_pos, 'o', markersize=8, color='darkblue',
        label='Observed Effect', zorder=3)

# Plot ±1 sigma error bars (68% CI)
ax.errorbar(effects_sorted, y_pos, xerr=sigmas_sorted,
            fmt='none', ecolor='steelblue', alpha=0.6,
            linewidth=2, capsize=4, label='±1 SE (68% CI)', zorder=2)

# Plot ±2 sigma error bars (95% CI)
ax.errorbar(effects_sorted, y_pos, xerr=2*sigmas_sorted,
            fmt='none', ecolor='lightblue', alpha=0.4,
            linewidth=1.5, capsize=3, label='±2 SE (95% CI)', zorder=1)

# Add reference line at 0
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No Effect')

# Add pooled estimate lines
unweighted_mean = effects.mean()
weights = 1 / (sigmas**2)
weighted_mean = np.average(effects, weights=weights)

ax.axvline(x=unweighted_mean, color='green', linestyle='-.', linewidth=1.5,
           alpha=0.7, label=f'Unweighted Mean: {unweighted_mean:.1f}')
ax.axvline(x=weighted_mean, color='orange', linestyle='-.', linewidth=1.5,
           alpha=0.7, label=f'Weighted Mean: {weighted_mean:.1f}')

ax.set_yticks(y_pos)
ax.set_yticklabels([f'School {s}' for s in schools_sorted])
ax.set_xlabel('Treatment Effect', fontsize=12, fontweight='bold')
ax.set_ylabel('School', fontsize=12, fontweight='bold')
ax.set_title('Eight Schools: Observed Effects with Uncertainty Intervals',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: 01_forest_plot.png")

# ============================================================================
# 2. Effect Distributions - Multi-panel
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Histogram with KDE
ax = axes[0, 0]
ax.hist(effects, bins=6, density=True, alpha=0.6, color='steelblue', edgecolor='black')
kde_x = np.linspace(effects.min() - 5, effects.max() + 5, 100)
kde = stats.gaussian_kde(effects)
ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
ax.axvline(effects.mean(), color='green', linestyle='--', linewidth=2,
           label=f'Mean: {effects.mean():.1f}')
ax.axvline(np.median(effects), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {np.median(effects):.1f}')
ax.set_xlabel('Effect Size', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('(A) Distribution of Observed Effects', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Q-Q plot for normality
ax = axes[0, 1]
stats.probplot(effects, dist="norm", plot=ax)
ax.set_title('(B) Q-Q Plot: Normality Check', fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel C: Box plot
ax = axes[1, 0]
bp = ax.boxplot(effects, vert=False, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('darkblue')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
ax.scatter(effects, np.ones(len(effects)), alpha=0.6, s=100,
           color='darkblue', zorder=3, label='Individual Schools')
ax.set_xlabel('Effect Size', fontweight='bold')
ax.set_title('(C) Box Plot with Individual Points', fontweight='bold')
ax.set_yticks([])
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Panel D: Empirical CDF
ax = axes[1, 1]
sorted_effects = np.sort(effects)
cdf = np.arange(1, len(sorted_effects) + 1) / len(sorted_effects)
ax.step(sorted_effects, cdf, where='post', linewidth=2, color='darkblue')
ax.scatter(sorted_effects, cdf, s=80, color='red', zorder=3)
ax.set_xlabel('Effect Size', fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontweight='bold')
ax.set_title('(D) Empirical Cumulative Distribution', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_effect_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: 02_effect_distributions.png")

# ============================================================================
# 3. Relationship between Effect and Sigma
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Scatter plot with correlation
ax = axes[0]
ax.scatter(sigmas, effects, s=150, alpha=0.6, color='darkblue', edgecolors='black', linewidth=1.5)

# Add school labels
for i, school in enumerate(schools):
    ax.annotate(f'{school}', (sigmas[i], effects[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

# Fit and plot regression line
z = np.polyfit(sigmas, effects, 1)
p = np.poly1d(z)
sigma_line = np.linspace(sigmas.min(), sigmas.max(), 100)
ax.plot(sigma_line, p(sigma_line), "r--", linewidth=2, alpha=0.7,
        label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')

corr, p_val = stats.pearsonr(sigmas, effects)
ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {p_val:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10)

ax.set_xlabel('Standard Error (sigma)', fontsize=12, fontweight='bold')
ax.set_ylabel('Observed Effect', fontsize=12, fontweight='bold')
ax.set_title('(A) Effect vs. Uncertainty', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Signal-to-noise ratio
ax = axes[1]
snr = effects / sigmas
colors = ['green' if abs(s) > 2 else 'steelblue' for s in snr]
bars = ax.bar(schools, snr, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add horizontal reference lines
ax.axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='|SNR| = 2')
ax.axhline(y=-2, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

ax.set_xlabel('School', fontsize=12, fontweight='bold')
ax.set_ylabel('Effect / Sigma (Signal-to-Noise)', fontsize=12, fontweight='bold')
ax.set_title('(B) Signal-to-Noise Ratio by School', fontsize=12, fontweight='bold')
ax.set_xticks(schools)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/03_effect_vs_sigma.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: 03_effect_vs_sigma.png")

# ============================================================================
# 4. Variance Components Visualization
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Within vs Between School Variance
ax = axes[0]
within_var = sigmas**2
between_var_empirical = (effects - effects.mean())**2

x_pos = np.arange(len(schools))
width = 0.35

bars1 = ax.bar(x_pos - width/2, within_var, width, label='Within (σ²)',
               color='lightcoral', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x_pos + width/2, between_var_empirical, width,
               label='Between (deviation²)',
               color='lightblue', alpha=0.7, edgecolor='black')

ax.set_xlabel('School', fontsize=12, fontweight='bold')
ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
ax.set_title('(A) Variance Components by School', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(schools)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add mean lines
mean_within = within_var.mean()
ax.axhline(y=mean_within, color='red', linestyle='--', linewidth=2, alpha=0.5,
           label=f'Mean Within: {mean_within:.1f}')

# Panel B: Precision weights
ax = axes[1]
precision = 1 / (sigmas**2)
bars = ax.bar(schools, precision, color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add values on bars
for i, (school, prec) in enumerate(zip(schools, precision)):
    ax.text(school, prec, f'{prec:.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('School', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision (1/σ²)', fontsize=12, fontweight='bold')
ax.set_title('(B) Precision Weights for Each School', fontsize=12, fontweight='bold')
ax.set_xticks(schools)
ax.grid(True, alpha=0.3, axis='y')

# Add mean line
mean_precision = precision.mean()
ax.axhline(y=mean_precision, color='red', linestyle='--', linewidth=2, alpha=0.5,
           label=f'Mean: {mean_precision:.4f}')
ax.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/04_variance_components.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: 04_variance_components.png")

# ============================================================================
# 5. Pooling Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

schools_labels = [f'School {s}' for s in schools]
y_pos = np.arange(len(schools))

# No pooling (individual estimates)
ax.scatter(effects, y_pos, s=150, color='darkblue', marker='o',
           label='No Pooling (Observed)', zorder=3)
ax.errorbar(effects, y_pos, xerr=sigmas, fmt='none', ecolor='steelblue',
            alpha=0.4, linewidth=2, capsize=4, zorder=2)

# Complete pooling (unweighted mean)
unweighted_mean = effects.mean()
ax.scatter([unweighted_mean]*len(schools), y_pos, s=100, color='green',
           marker='s', label='Complete Pooling (Unweighted Mean)', zorder=3, alpha=0.6)

# Weighted pooling
weighted_mean = np.average(effects, weights=1/(sigmas**2))
ax.scatter([weighted_mean]*len(schools), y_pos, s=100, color='orange',
           marker='^', label='Weighted Pooling (Inverse Variance)', zorder=3, alpha=0.6)

# Add vertical reference line at 0
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.3)

ax.set_yticks(y_pos)
ax.set_yticklabels(schools_labels)
ax.set_xlabel('Treatment Effect', fontsize=12, fontweight='bold')
ax.set_ylabel('School', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Pooling Strategies\n(Partial Pooling via Hierarchical Model would be intermediate)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/05_pooling_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: 05_pooling_comparison.png")

# ============================================================================
# 6. Comprehensive Summary Panel
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top row: Forest plot (spanning 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
sort_idx = np.argsort(effects)
schools_sorted = schools[sort_idx]
effects_sorted = effects[sort_idx]
sigmas_sorted = sigmas[sort_idx]
y_pos = np.arange(len(schools_sorted))

ax1.plot(effects_sorted, y_pos, 'o', markersize=8, color='darkblue', zorder=3)
ax1.errorbar(effects_sorted, y_pos, xerr=sigmas_sorted, fmt='none',
             ecolor='steelblue', alpha=0.6, linewidth=2, capsize=4, zorder=2)
ax1.errorbar(effects_sorted, y_pos, xerr=2*sigmas_sorted, fmt='none',
             ecolor='lightblue', alpha=0.3, linewidth=1.5, capsize=3, zorder=1)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.axvline(x=unweighted_mean, color='green', linestyle='-.', linewidth=1.5, alpha=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f'School {s}' for s in schools_sorted])
ax1.set_xlabel('Treatment Effect', fontweight='bold')
ax1.set_title('Observed Effects with ±1σ and ±2σ Intervals', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Top right: Summary statistics
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
SUMMARY STATISTICS

Effect:
  Mean: {effects.mean():.2f}
  Median: {np.median(effects):.2f}
  SD: {effects.std():.2f}
  Range: [{effects.min():.1f}, {effects.max():.1f}]

Sigma:
  Mean: {sigmas.mean():.2f}
  Median: {np.median(sigmas):.2f}
  Range: [{sigmas.min()}, {sigmas.max()}]

Variance Components:
  Between-school: {effects.var():.2f}
  Mean within-school: {(sigmas**2).mean():.2f}
  Ratio: {effects.var()/(sigmas**2).mean():.3f}

Pooling Estimates:
  Unweighted: {unweighted_mean:.2f}
  Weighted: {weighted_mean:.2f}
"""
ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
         verticalalignment='top', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

# Middle left: Distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(effects, bins=6, density=True, alpha=0.6, color='steelblue', edgecolor='black')
kde = stats.gaussian_kde(effects)
kde_x = np.linspace(effects.min() - 5, effects.max() + 5, 100)
ax3.plot(kde_x, kde(kde_x), 'r-', linewidth=2)
ax3.axvline(effects.mean(), color='green', linestyle='--', linewidth=2)
ax3.set_xlabel('Effect Size', fontweight='bold')
ax3.set_ylabel('Density', fontweight='bold')
ax3.set_title('Distribution of Effects', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Middle center: Effect vs Sigma
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(sigmas, effects, s=120, alpha=0.6, color='darkblue', edgecolors='black')
for i, school in enumerate(schools):
    ax4.annotate(f'{school}', (sigmas[i], effects[i]),
                xytext=(3, 3), textcoords='offset points', fontsize=9)
z = np.polyfit(sigmas, effects, 1)
p = np.poly1d(z)
sigma_line = np.linspace(sigmas.min(), sigmas.max(), 100)
ax4.plot(sigma_line, p(sigma_line), "r--", linewidth=2, alpha=0.7)
corr, p_val = stats.pearsonr(sigmas, effects)
ax4.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}',
        transform=ax4.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
ax4.set_xlabel('Standard Error (σ)', fontweight='bold')
ax4.set_ylabel('Observed Effect', fontweight='bold')
ax4.set_title('Effect vs. Uncertainty', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Middle right: Signal-to-noise
ax5 = fig.add_subplot(gs[1, 2])
snr = effects / sigmas
colors = ['green' if abs(s) > 2 else 'steelblue' for s in snr]
ax5.bar(schools, snr, color=colors, alpha=0.7, edgecolor='black')
ax5.axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax5.axhline(y=-2, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax5.set_xlabel('School', fontweight='bold')
ax5.set_ylabel('Effect/σ', fontweight='bold')
ax5.set_title('Signal-to-Noise Ratio', fontweight='bold')
ax5.set_xticks(schools)
ax5.grid(True, alpha=0.3, axis='y')

# Bottom left: Within variance
ax6 = fig.add_subplot(gs[2, 0])
ax6.bar(schools, sigmas**2, color='lightcoral', alpha=0.7, edgecolor='black')
ax6.axhline(y=(sigmas**2).mean(), color='red', linestyle='--', linewidth=2, alpha=0.5)
ax6.set_xlabel('School', fontweight='bold')
ax6.set_ylabel('Within Variance (σ²)', fontweight='bold')
ax6.set_title('Within-School Variance', fontweight='bold')
ax6.set_xticks(schools)
ax6.grid(True, alpha=0.3, axis='y')

# Bottom center: Between variance (deviations)
ax7 = fig.add_subplot(gs[2, 1])
between_var_empirical = (effects - effects.mean())**2
ax7.bar(schools, between_var_empirical, color='lightblue', alpha=0.7, edgecolor='black')
ax7.set_xlabel('School', fontweight='bold')
ax7.set_ylabel('Squared Deviation', fontweight='bold')
ax7.set_title('Between-School Variance', fontweight='bold')
ax7.set_xticks(schools)
ax7.grid(True, alpha=0.3, axis='y')

# Bottom right: Precision weights
ax8 = fig.add_subplot(gs[2, 2])
precision = 1 / (sigmas**2)
ax8.bar(schools, precision, color='teal', alpha=0.7, edgecolor='black')
ax8.axhline(y=precision.mean(), color='red', linestyle='--', linewidth=2, alpha=0.5)
ax8.set_xlabel('School', fontweight='bold')
ax8.set_ylabel('Precision (1/σ²)', fontweight='bold')
ax8.set_title('Precision Weights', fontweight='bold')
ax8.set_xticks(schools)
ax8.grid(True, alpha=0.3, axis='y')

fig.suptitle('Eight Schools Dataset: Comprehensive Summary',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{output_dir}/06_comprehensive_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: 06_comprehensive_summary.png")

print("\n" + "="*80)
print("All visualizations created successfully!")
print("="*80)
