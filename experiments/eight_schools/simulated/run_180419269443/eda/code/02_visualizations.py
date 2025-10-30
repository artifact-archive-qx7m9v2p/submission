"""
Visualization Suite for Meta-Analysis Dataset
==============================================
Creates comprehensive visual diagnostics
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
data = pd.read_csv('/workspace/eda/code/processed_data.csv')

# Calculate weighted mean for reference
weighted_mean = np.sum(data['y'] * data['precision']) / np.sum(data['precision'])

# ============================================================
# Figure 1: Forest Plot (Caterpillar Plot)
# ============================================================
print("Creating Figure 1: Forest Plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Sort by effect size for better visualization
data_sorted = data.sort_values('y', ascending=True)

# Plot points and confidence intervals
for i, (idx, row) in enumerate(data_sorted.iterrows()):
    # 95% CI: y ± 1.96*sigma
    ci_lower = row['y'] - 1.96 * row['sigma']
    ci_upper = row['y'] + 1.96 * row['sigma']

    # Plot CI line
    ax.plot([ci_lower, ci_upper], [i, i], 'k-', linewidth=1.5, alpha=0.6)

    # Plot point (size inversely proportional to sigma)
    point_size = 200 / row['sigma']
    ax.scatter(row['y'], i, s=point_size, c='steelblue',
               edgecolors='black', linewidth=1, zorder=3)

# Add vertical line at weighted mean
ax.axvline(weighted_mean, color='red', linestyle='--',
           linewidth=2, alpha=0.7, label=f'Weighted Mean = {weighted_mean:.2f}')

# Add vertical line at zero
ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Labels
ax.set_yticks(range(len(data_sorted)))
ax.set_yticklabels([f"Study {int(s)}" for s in data_sorted['study']])
ax.set_xlabel('Effect Size (y) with 95% CI', fontsize=12)
ax.set_ylabel('Study', fontsize=12)
ax.set_title('Forest Plot: Observed Effects with 95% Confidence Intervals',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/01_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 2: Distribution of Observed Effects
# ============================================================
print("Creating Figure 2: Distribution of Effects...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
ax = axes[0]
ax.hist(data['y'], bins=6, alpha=0.6, color='steelblue', edgecolor='black', density=True)
# KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(data['y'])
x_range = np.linspace(data['y'].min() - 5, data['y'].max() + 5, 100)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax.axvline(data['y'].mean(), color='blue', linestyle='--',
           linewidth=2, label=f'Mean = {data["y"].mean():.2f}')
ax.axvline(data['y'].median(), color='green', linestyle='--',
           linewidth=2, label=f'Median = {data["y"].median():.2f}')
ax.set_xlabel('Effect Size (y)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Observed Effects', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Q-Q plot
ax = axes[1]
stats.probplot(data['y'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Testing Normality of Effects', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/02_effect_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 3: Distribution of Standard Errors
# ============================================================
print("Creating Figure 3: Standard Error Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(data['sigma'], bins=6, alpha=0.6, color='coral', edgecolor='black')
ax.axvline(data['sigma'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean = {data["sigma"].mean():.2f}')
ax.axvline(data['sigma'].median(), color='darkred', linestyle='--',
           linewidth=2, label=f'Median = {data["sigma"].median():.2f}')
ax.set_xlabel('Standard Error (sigma)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Standard Errors', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Box plot with individual points
ax = axes[1]
ax.boxplot(data['sigma'], vert=True, widths=0.5, patch_artist=True,
           boxprops=dict(facecolor='coral', alpha=0.6))
ax.scatter(np.ones(len(data['sigma'])), data['sigma'],
           s=100, alpha=0.6, c='darkred', edgecolors='black')
ax.set_ylabel('Standard Error (sigma)', fontsize=12)
ax.set_title('Box Plot of Standard Errors', fontsize=12, fontweight='bold')
ax.set_xticks([])
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/03_sigma_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Relationship between Effect Size and Precision
# ============================================================
print("Creating Figure 4: Effect Size vs Precision...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: y vs sigma
ax = axes[0, 0]
ax.scatter(data['sigma'], data['y'], s=150, alpha=0.7, c='steelblue',
           edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax.annotate(f"{int(row['study'])}", (row['sigma'], row['y']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
# Add correlation
corr, p_val = stats.pearsonr(data['sigma'], data['y'])
ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Standard Error (sigma)', fontsize=12)
ax.set_ylabel('Effect Size (y)', fontsize=12)
ax.set_title('Effect Size vs Standard Error', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Panel 2: y vs precision
ax = axes[0, 1]
ax.scatter(data['precision'], data['y'], s=150, alpha=0.7, c='forestgreen',
           edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax.annotate(f"{int(row['study'])}", (row['precision'], row['y']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
corr, p_val = stats.pearsonr(data['precision'], data['y'])
ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Precision (1/sigma)', fontsize=12)
ax.set_ylabel('Effect Size (y)', fontsize=12)
ax.set_title('Effect Size vs Precision', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Panel 3: Funnel plot (standard)
ax = axes[1, 0]
ax.scatter(data['y'], data['sigma'], s=150, alpha=0.7, c='purple',
           edgecolors='black', linewidth=1.5)
ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Weighted Mean')
ax.invert_yaxis()
ax.set_xlabel('Effect Size (y)', fontsize=12)
ax.set_ylabel('Standard Error (sigma)', fontsize=12)
ax.set_title('Funnel Plot (inverted)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel 4: Precision-weighted effects
ax = axes[1, 1]
weights = data['precision']**2
weighted_effects = data['y'] * weights / weights.sum()
studies = data['study'].astype(int)
colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
bars = ax.bar(studies, weighted_effects, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Precision-Weighted Contribution', fontsize=12)
ax.set_title('Contribution of Each Study to Pooled Estimate', fontsize=12, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/04_effect_precision_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 5: Heterogeneity Diagnostics
# ============================================================
print("Creating Figure 5: Heterogeneity Diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Standardized effects (z-scores)
ax = axes[0, 0]
z_scores = data['y'] / data['sigma']
studies = data['study'].astype(int)
colors_pos = ['forestgreen' if z > 0 else 'coral' for z in z_scores]
ax.bar(studies, z_scores, color=colors_pos, alpha=0.7, edgecolor='black')
ax.axhline(1.96, color='red', linestyle='--', linewidth=2, alpha=0.7, label='95% CI threshold')
ax.axhline(-1.96, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Standardized Effect (y/sigma)', fontsize=12)
ax.set_title('Standardized Effects by Study', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Panel 2: Observed vs Expected variance
ax = axes[1, 0]
observed_se = data['sigma']
studies = data['study'].astype(int)
ax.scatter(studies, observed_se, s=150, alpha=0.7, c='steelblue',
           edgecolors='black', linewidth=1.5, label='Observed SE')
ax.axhline(observed_se.mean(), color='red', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Mean SE = {observed_se.mean():.2f}')
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Standard Error', fontsize=12)
ax.set_title('Standard Errors by Study', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Effect sizes with sampling uncertainty bands
ax = axes[0, 1]
studies = data['study'].astype(int)
ax.scatter(studies, data['y'], s=150, alpha=0.7, c='steelblue',
           edgecolors='black', linewidth=1.5, zorder=3)
# Add uncertainty bands (1 SE)
for idx, row in data.iterrows():
    study_id = int(row['study'])
    ax.fill_between([study_id-0.2, study_id+0.2],
                     row['y'] - row['sigma'], row['y'] + row['sigma'],
                     alpha=0.3, color='lightblue')
ax.axhline(weighted_mean, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Weighted Mean = {weighted_mean:.2f}')
ax.axhline(data['y'].mean(), color='blue', linestyle=':', linewidth=2,
           alpha=0.7, label=f'Unweighted Mean = {data["y"].mean():.2f}')
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Effect Size (y)', fontsize=12)
ax.set_title('Effect Sizes with ±1 SE Bands', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel 4: Galbraith (radial) plot
ax = axes[1, 1]
precision = data['precision']
standardized_effect = data['y'] * precision
ax.scatter(precision, standardized_effect, s=150, alpha=0.7, c='purple',
           edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax.annotate(f"{int(row['study'])}", (row['precision'], row['y']*row['precision']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
# Add line through origin with slope = pooled effect
x_line = np.array([precision.min()*0.9, precision.max()*1.1])
ax.plot(x_line, weighted_mean * x_line, 'r--', linewidth=2,
        alpha=0.7, label=f'Slope = {weighted_mean:.2f}')
ax.set_xlabel('Precision (1/sigma)', fontsize=12)
ax.set_ylabel('Precision × Effect (y/sigma)', fontsize=12)
ax.set_title('Galbraith (Radial) Plot', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/05_heterogeneity_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 6: Study-level detailed view
# ============================================================
print("Creating Figure 6: Study-level Details...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create a comprehensive study-level view
studies = data['study'].astype(int)
x_pos = np.arange(len(studies))

# Plot observed effects
ax.scatter(x_pos, data['y'], s=300, alpha=0.7, c='steelblue',
           edgecolors='black', linewidth=2, label='Observed Effect', zorder=3)

# Add error bars (±1.96*sigma for 95% CI)
ax.errorbar(x_pos, data['y'], yerr=1.96*data['sigma'],
            fmt='none', ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5, alpha=0.6)

# Add weighted mean line
ax.axhline(weighted_mean, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Weighted Mean = {weighted_mean:.2f}')

# Add confidence band around weighted mean (approximate)
pooled_se = 1 / np.sqrt(np.sum(data['precision']**2))
ax.fill_between(x_pos, weighted_mean - 1.96*pooled_se, weighted_mean + 1.96*pooled_se,
                alpha=0.2, color='red', label='95% CI of Pooled Effect')

# Add study details as text
for i, (idx, row) in enumerate(data.iterrows()):
    ax.text(i, data['y'].max() + 5,
            f"SE={row['sigma']:.0f}\nw={row['precision']**2/np.sum(data['precision']**2):.2f}",
            ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xticks(x_pos)
ax.set_xticklabels([f"Study {s}" for s in studies])
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Effect Size (y)', fontsize=12)
ax.set_title('Detailed Study-Level View with Weights and Confidence Intervals',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/06_study_level_details.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("All visualizations saved to: /workspace/eda/visualizations/")
print("="*60)
