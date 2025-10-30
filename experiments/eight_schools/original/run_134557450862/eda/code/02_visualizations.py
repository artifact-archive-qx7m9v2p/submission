"""
Visualizations for Eight Schools Dataset
========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/eda/code/data_with_diagnostics.csv')

# Color palette
colors = sns.color_palette("husl", 8)

# ============================================================================
# PLOT 1: Forest Plot with Confidence Intervals
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by effect size for better visualization
data_sorted = data.sort_values('y', ascending=True).reset_index(drop=True)

y_pos = np.arange(len(data_sorted))

# Plot confidence intervals (±2 sigma ~ 95% CI)
for i, row in data_sorted.iterrows():
    ci_lower = row['y'] - 2 * row['sigma']
    ci_upper = row['y'] + 2 * row['sigma']
    ax.plot([ci_lower, ci_upper], [i, i], 'o-', color='steelblue',
            linewidth=2, markersize=8, alpha=0.7)

# Add vertical line at weighted mean
weights = 1 / (data['sigma'] ** 2)
weighted_mean = np.sum(data['y'] * weights) / np.sum(weights)
ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Weighted Mean = {weighted_mean:.1f}', alpha=0.7)

# Add vertical line at zero
ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([f"School {int(s)}" for s in data_sorted['school']])
ax.set_xlabel('Treatment Effect (y)', fontsize=12, fontweight='bold')
ax.set_ylabel('School', fontsize=12, fontweight='bold')
ax.set_title('Forest Plot: Observed Effects with 95% Confidence Intervals',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: forest_plot.png")

# ============================================================================
# PLOT 2: Distribution Analysis (Multi-panel)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a: Histogram of effects
ax = axes[0, 0]
ax.hist(data['y'], bins=8, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(data['y'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {data["y"].mean():.1f}')
ax.axvline(data['y'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median = {data["y"].median():.1f}')
ax.set_xlabel('Observed Effect (y)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Distribution of Observed Effects', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2b: Histogram of standard errors
ax = axes[0, 1]
ax.hist(data['sigma'], bins=6, color='lightcoral', edgecolor='black', alpha=0.7)
ax.axvline(data['sigma'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {data["sigma"].mean():.1f}')
ax.axvline(data['sigma'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median = {data["sigma"].median():.1f}')
ax.set_xlabel('Standard Error (sigma)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Distribution of Measurement Uncertainty', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2c: Box plots
ax = axes[1, 0]
box_data = [data['y'], data['sigma']]
bp = ax.boxplot(box_data, labels=['Effect (y)', 'SE (sigma)'], patch_artist=True,
                widths=0.6)
bp['boxes'][0].set_facecolor('skyblue')
bp['boxes'][1].set_facecolor('lightcoral')
for patch in bp['boxes']:
    patch.set_alpha(0.7)
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Box Plots: Comparing Distributions', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 2d: Q-Q plot for effects
ax = axes[1, 1]
stats.probplot(data['y'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Testing Normality of Effects', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: distribution_analysis.png")

# ============================================================================
# PLOT 3: Effect vs Uncertainty Scatter
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot with school labels
scatter = ax.scatter(data['sigma'], data['y'], s=200, alpha=0.6,
                     c=range(len(data)), cmap='viridis', edgecolors='black', linewidth=1.5)

# Add school labels
for _, row in data.iterrows():
    ax.annotate(f"School {int(row['school'])}",
                (row['sigma'], row['y']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

# Add trend line
z = np.polyfit(data['sigma'], data['y'], 1)
p = np.poly1d(z)
x_line = np.linspace(data['sigma'].min(), data['sigma'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
        label=f'Linear fit: y = {z[0]:.2f}·sigma + {z[1]:.2f}')

# Add correlation info
corr = data['y'].corr(data['sigma'])
ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Standard Error (sigma)', fontsize=12, fontweight='bold')
ax.set_ylabel('Observed Effect (y)', fontsize=12, fontweight='bold')
ax.set_title('Relationship Between Effect Size and Measurement Uncertainty',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/effect_vs_uncertainty.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: effect_vs_uncertainty.png")

# ============================================================================
# PLOT 4: Precision-Weighted Analysis
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4a: Funnel plot
ax = axes[0]
ax.scatter(data['y'], 1/data['sigma'], s=150, alpha=0.6,
           c=range(len(data)), cmap='plasma', edgecolors='black', linewidth=1.5)

# Add weighted mean line
ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Weighted Mean = {weighted_mean:.1f}')

# Add funnel (approximate)
y_range = np.linspace(-20, 40, 100)
precision_low = 1/18  # lowest precision
ax.plot(weighted_mean + 2*18, 1/18, 'gray', alpha=0)  # dummy for symmetry

for _, row in data.iterrows():
    ax.annotate(f"{int(row['school'])}",
                (row['y'], 1/row['sigma']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

ax.set_xlabel('Observed Effect (y)', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision (1/sigma)', fontsize=11, fontweight='bold')
ax.set_title('Funnel Plot: Effect vs Precision', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4b: Standardized residuals
ax = axes[1]
z_scores = data['z_score']
schools = data['school'].astype(int)

colors_resid = ['red' if abs(z) > 2 else 'orange' if abs(z) > 1.5 else 'steelblue'
                for z in z_scores]

bars = ax.bar(schools, z_scores, color=colors_resid, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add reference lines
ax.axhline(0, color='black', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±2 SD threshold')
ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(1.5, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(-1.5, color='orange', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('School', fontsize=11, fontweight='bold')
ax.set_ylabel('Standardized Residual (z-score)', fontsize=11, fontweight='bold')
ax.set_title('Standardized Residuals from Weighted Mean', fontsize=12, fontweight='bold')
ax.set_xticks(schools)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/precision_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: precision_analysis.png")

# ============================================================================
# PLOT 5: Individual School Profiles
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Create bubble plot
sizes = 1000 / (data['sigma'] ** 2)  # Size proportional to precision

scatter = ax.scatter(data['school'], data['y'], s=sizes, alpha=0.6,
                     c=data['sigma'], cmap='RdYlGn_r',
                     edgecolors='black', linewidth=2)

# Add error bars
for _, row in data.iterrows():
    ax.errorbar(row['school'], row['y'], yerr=2*row['sigma'],
                fmt='none', color='gray', alpha=0.4, linewidth=2, capsize=5)

# Add weighted mean line
ax.axhline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Weighted Mean = {weighted_mean:.1f}', alpha=0.7)

# Add unweighted mean line
ax.axhline(data['y'].mean(), color='blue', linestyle=':', linewidth=2,
           label=f'Unweighted Mean = {data["y"].mean():.1f}', alpha=0.7)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Standard Error (sigma)', fontsize=11, fontweight='bold')

ax.set_xlabel('School', fontsize=12, fontweight='bold')
ax.set_ylabel('Observed Effect (y)', fontsize=12, fontweight='bold')
ax.set_title('School Profiles: Effect Estimates with Uncertainty\n(Bubble size = Precision)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(data['school'])
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/school_profiles.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: school_profiles.png")

# ============================================================================
# PLOT 6: Heterogeneity Diagnostic
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a: Observed effects vs expected under homogeneity
ax = axes[0, 0]
expected = np.full(len(data), weighted_mean)
ax.scatter(expected, data['y'], s=150, alpha=0.6, c='steelblue', edgecolors='black', linewidth=1.5)
ax.plot([weighted_mean-5, weighted_mean+5], [weighted_mean-5, weighted_mean+5],
        'r--', linewidth=2, label='Perfect agreement')

for _, row in data.iterrows():
    ax.annotate(f"{int(row['school'])}",
                (weighted_mean, row['y']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

ax.set_xlabel('Expected (Weighted Mean)', fontsize=11, fontweight='bold')
ax.set_ylabel('Observed Effect', fontsize=11, fontweight='bold')
ax.set_title('Observed vs Expected Under Homogeneity', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 6b: Contribution to Q statistic
ax = axes[0, 1]
Q_contributions = weights * (data['y'] - weighted_mean) ** 2
schools = data['school'].astype(int)

bars = ax.bar(schools, Q_contributions, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('School', fontsize=11, fontweight='bold')
ax.set_ylabel('Contribution to Q', fontsize=11, fontweight='bold')
ax.set_title("Each School's Contribution to Heterogeneity (Q)", fontsize=12, fontweight='bold')
ax.set_xticks(schools)
ax.grid(True, alpha=0.3, axis='y')

# Add total Q
total_Q = Q_contributions.sum()
ax.text(0.95, 0.95, f'Total Q = {total_Q:.2f}\ndf = 7\np = 0.696',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6c: Precision vs effect (another view)
ax = axes[1, 0]
precision = 1 / (data['sigma'] ** 2)
ax.scatter(precision, data['y'], s=150, alpha=0.6, c='mediumseagreen',
           edgecolors='black', linewidth=1.5)

for _, row in data.iterrows():
    ax.annotate(f"{int(row['school'])}",
                (1/row['sigma']**2, row['y']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

ax.axhline(weighted_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Precision (1/sigma²)', fontsize=11, fontweight='bold')
ax.set_ylabel('Observed Effect (y)', fontsize=11, fontweight='bold')
ax.set_title('Precision-Effect Relationship', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 6d: Leave-one-out sensitivity
ax = axes[1, 1]
loo_means = []
for idx in range(len(data)):
    loo_data = data.drop(idx)
    loo_weights = 1 / (loo_data['sigma'] ** 2)
    loo_mean = np.sum(loo_data['y'] * loo_weights) / np.sum(loo_weights)
    loo_means.append(loo_mean)

schools = data['school'].astype(int)
changes = np.array(loo_means) - weighted_mean

colors_loo = ['red' if abs(c) > 1.5 else 'orange' if abs(c) > 1 else 'steelblue'
              for c in changes]

bars = ax.bar(schools, changes, color=colors_loo, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('School Removed', fontsize=11, fontweight='bold')
ax.set_ylabel('Change in Weighted Mean', fontsize=11, fontweight='bold')
ax.set_title('Leave-One-Out Influence Analysis', fontsize=12, fontweight='bold')
ax.set_xticks(schools)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/heterogeneity_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: heterogeneity_diagnostics.png")

print("\nAll visualizations created successfully!")
