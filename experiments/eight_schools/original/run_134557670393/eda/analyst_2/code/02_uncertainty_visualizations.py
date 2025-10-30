"""
Comprehensive Uncertainty Visualizations
Focus on precision, signal-to-noise, and uncertainty patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load enhanced data
df = pd.read_csv('/workspace/eda/analyst_2/code/enhanced_data.csv')

# ============================================================
# 1. COMPREHENSIVE UNCERTAINTY OVERVIEW (Multi-panel)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Uncertainty Structure Overview', fontsize=16, fontweight='bold', y=0.995)

# Panel A: Precision vs Effect Size
ax = axes[0, 0]
ax.scatter(df['precision'], df['y'], s=100, alpha=0.6, c='steelblue', edgecolors='black', linewidth=1.5)
for idx, row in df.iterrows():
    ax.annotate(f"S{int(row['study'])}", (row['precision'], row['y']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
# Add correlation
corr, pval = stats.pearsonr(df['precision'], df['y'])
ax.text(0.05, 0.95, f"r = {corr:.3f}\np = {pval:.3f}",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Precision (1/σ)', fontsize=11)
ax.set_ylabel('Effect Size (y)', fontsize=11)
ax.set_title('A: Precision-Effect Relationship', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.grid(True, alpha=0.3)

# Panel B: Signal-to-Noise Ratio by Study
ax = axes[0, 1]
colors = ['red' if snr < 0 else 'green' if snr > 1.5 else 'orange' for snr in df['snr']]
bars = ax.bar(df['study'], df['snr'], color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.axhline(y=1.96, color='blue', linestyle='--', linewidth=2, label='Significance threshold (±1.96)')
ax.axhline(y=-1.96, color='blue', linestyle='--', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.set_xlabel('Study', fontsize=11)
ax.set_ylabel('Signal-to-Noise Ratio (y/σ)', fontsize=11)
ax.set_title('B: Signal-to-Noise Ratio by Study', fontsize=12, fontweight='bold')
ax.set_xticks(df['study'])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Uncertainty Distribution
ax = axes[1, 0]
ax.hist(df['sigma'], bins=6, alpha=0.6, color='coral', edgecolor='black', linewidth=1.5)
ax.axvline(df['sigma'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["sigma"].mean():.1f}')
ax.axvline(df['sigma'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["sigma"].median():.1f}')
ax.set_xlabel('Standard Error (σ)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('C: Distribution of Uncertainty', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Confidence Interval Widths
ax = axes[1, 1]
sorted_df = df.sort_values('ci_width')
ax.barh(range(len(sorted_df)), sorted_df['ci_width'],
        color='mediumpurple', alpha=0.6, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels([f"Study {int(s)}" for s in sorted_df['study']])
ax.set_xlabel('95% CI Width (3.92σ)', fontsize=11)
ax.set_title('D: Confidence Interval Widths', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/01_uncertainty_overview.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 01_uncertainty_overview.png")

# ============================================================
# 2. FUNNEL PLOT (Critical for bias assessment)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
ax.scatter(df['y'], df['sigma'], s=150, alpha=0.6, c='steelblue',
           edgecolors='black', linewidth=2, zorder=3)

# Annotate studies
for idx, row in df.iterrows():
    ax.annotate(f"S{int(row['study'])}", (row['y'], row['sigma']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# Add mean effect line
mean_effect = df['y'].mean()
ax.axvline(mean_effect, color='red', linestyle='--', linewidth=2, label=f'Mean effect: {mean_effect:.2f}')

# Add funnel (theoretical precision contours for zero effect)
sigma_range = np.linspace(df['sigma'].min(), df['sigma'].max(), 100)
# 95% CI bounds assuming true effect = mean
ax.plot(mean_effect + 1.96 * sigma_range, sigma_range, 'gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.plot(mean_effect - 1.96 * sigma_range, sigma_range, 'gray', linestyle='--', alpha=0.5, linewidth=1.5)

# Add zero line
ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

ax.set_xlabel('Effect Size (y)', fontsize=13, fontweight='bold')
ax.set_ylabel('Standard Error (σ)', fontsize=13, fontweight='bold')
ax.set_title('Funnel Plot: Assessing Publication Bias and Small-Study Effects',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Conventional funnel plot orientation

# Add interpretation text
textstr = 'Symmetric funnel suggests low bias\nAsymmetry suggests publication/small-study effects'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/02_funnel_plot.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 02_funnel_plot.png")

# ============================================================
# 3. FOREST PLOT WITH CONFIDENCE INTERVALS
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Sort by effect size for visual clarity
sorted_df = df.sort_values('y', ascending=True).reset_index(drop=True)

y_pos = range(len(sorted_df))

# Plot confidence intervals as horizontal lines
for i, row in sorted_df.iterrows():
    ax.plot([row['ci_lower'], row['ci_upper']], [i, i],
            'o-', linewidth=2, markersize=8, alpha=0.7,
            color='steelblue', markerfacecolor='steelblue', markeredgecolor='black', markeredgewidth=1.5)

# Plot point estimates
ax.scatter(sorted_df['y'], y_pos, s=150, c='darkblue',
           edgecolors='black', linewidth=2, zorder=3, marker='D')

# Add zero line
ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Null effect')

# Add mean effect line
ax.axvline(mean_effect, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Mean effect: {mean_effect:.2f}')

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Study {int(row['study'])} (σ={row['sigma']:.0f})"
                     for idx, row in sorted_df.iterrows()])
ax.set_xlabel('Effect Size (y) with 95% CI', fontsize=13, fontweight='bold')
ax.set_title('Forest Plot: Effect Estimates with Uncertainty',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/03_forest_plot.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 03_forest_plot.png")

# ============================================================
# 4. PRECISION-WEIGHTED ANALYSIS (Multi-panel)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Precision-Weighted Analysis', fontsize=16, fontweight='bold')

# Panel A: Precision groups comparison
ax = axes[0]
median_precision = df['precision'].median()
high_prec = df[df['precision'] >= median_precision]
low_prec = df[df['precision'] < median_precision]

bp_data = [high_prec['y'], low_prec['y']]
bp = ax.boxplot(bp_data, labels=['High Precision\n(n={})'.format(len(high_prec)),
                                   'Low Precision\n(n={})'.format(len(low_prec))],
                patch_artist=True, widths=0.6)

for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')
    patch.set_linewidth(2)

# Overlay individual points
for i, data in enumerate(bp_data, 1):
    y = data.values
    x = np.random.normal(i, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.6, s=100, edgecolors='black', linewidth=1.5, zorder=3)

ax.axhline(0, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='Null effect')
ax.set_ylabel('Effect Size (y)', fontsize=12)
ax.set_title('A: Effects by Precision Group', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
mean_high = high_prec['y'].mean()
mean_low = low_prec['y'].mean()
ax.text(0.05, 0.95, f'High precision mean: {mean_high:.2f}\nLow precision mean: {mean_low:.2f}',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Weighted vs Unweighted means
ax = axes[1]

# Calculate weighted mean (inverse variance weighting)
weights = 1 / df['variance']
weighted_mean = np.sum(df['y'] * weights) / np.sum(weights)
unweighted_mean = df['y'].mean()

bars = ax.bar(['Unweighted\nMean', 'Precision-Weighted\nMean'],
              [unweighted_mean, weighted_mean],
              color=['coral', 'lightblue'], alpha=0.6,
              edgecolor='black', linewidth=2, width=0.6)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(0, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='Null effect')
ax.set_ylabel('Effect Size', fontsize=12)
ax.set_title('B: Weighting Impact on Meta-Analytic Mean', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add difference annotation
diff = weighted_mean - unweighted_mean
ax.text(0.5, 0.05, f'Difference: {diff:.2f}',
        transform=ax.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/04_precision_weighted_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 04_precision_weighted_analysis.png")

# ============================================================
# 5. OUTLIER DETECTION WITH UNCERTAINTY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Uncertainty-Adjusted Outlier Detection', fontsize=16, fontweight='bold')

# Panel A: Standardized effects (z-scores)
ax = axes[0]
colors = ['red' if abs(z) > 1.96 else 'steelblue' for z in df['snr']]
ax.scatter(df['study'], df['snr'], s=150, c=colors, alpha=0.6,
           edgecolors='black', linewidth=2, zorder=3)

# Add reference lines
ax.axhline(1.96, color='red', linestyle='--', linewidth=2, alpha=0.5, label='p < 0.05 threshold')
ax.axhline(-1.96, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Annotate
for idx, row in df.iterrows():
    ax.annotate(f"S{int(row['study'])}", (row['study'], row['snr']),
                xytext=(0, 10), textcoords='offset points',
                fontsize=10, ha='center', fontweight='bold')

ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Standardized Effect (z-score)', fontsize=12)
ax.set_title('A: Z-scores (Effect/SE)', fontsize=12, fontweight='bold')
ax.set_xticks(df['study'])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel B: Cook's distance-like influence
ax = axes[1]

# Calculate leave-one-out influence
influences = []
full_mean = df['y'].mean()
for i in range(len(df)):
    loo_mean = df.loc[df.index != i, 'y'].mean()
    influence = abs(full_mean - loo_mean)
    influences.append(influence)

df['influence'] = influences

colors = ['red' if inf > np.median(influences) * 2 else 'steelblue' for inf in influences]
bars = ax.bar(df['study'], df['influence'], color=colors, alpha=0.6,
              edgecolor='black', linewidth=2)

ax.axhline(np.median(influences) * 2, color='red', linestyle='--',
           linewidth=2, alpha=0.5, label='2× median influence')
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Leave-One-Out Influence', fontsize=12)
ax.set_title('B: Study Influence on Meta-Mean', fontsize=12, fontweight='bold')
ax.set_xticks(df['study'])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/05_outlier_detection.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 05_outlier_detection.png")

# ============================================================
# 6. VARIANCE-EFFECT RELATIONSHIP
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(df['variance'], df['y'], s=150, alpha=0.6, c='steelblue',
           edgecolors='black', linewidth=2)

# Annotate
for idx, row in df.iterrows():
    ax.annotate(f"S{int(row['study'])}", (row['variance'], row['y']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

# Add correlation
corr, pval = stats.pearsonr(df['variance'], df['y'])
ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}\np-value = {pval:.3f}",
        transform=ax.transAxes, va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add regression line
z = np.polyfit(df['variance'], df['y'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['variance'].min(), df['variance'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')

ax.axhline(0, color='blue', linestyle='--', alpha=0.3, linewidth=1)
ax.set_xlabel('Variance (σ²)', fontsize=13, fontweight='bold')
ax.set_ylabel('Effect Size (y)', fontsize=13, fontweight='bold')
ax.set_title('Variance-Effect Relationship: Testing for Heteroscedasticity',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/06_variance_effect_relationship.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 06_variance_effect_relationship.png")

print("\n" + "="*60)
print("All visualizations created successfully!")
print("="*60)
