"""
Comprehensive Visualizations - Meta-Analysis Structure & Context
Focus: Study ordering, extreme values, data quality, structural patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data with diagnostics
data = pd.read_csv('/workspace/eda/analyst_3/code/data_with_diagnostics.csv')

# Create output directory
import os
output_dir = '/workspace/eda/analyst_3/visualizations'
os.makedirs(output_dir, exist_ok=True)

print("Generating visualizations...")

# =============================================================================
# 1. STUDY SEQUENCE PLOT - Multi-panel showing temporal/ordering patterns
# =============================================================================
print("1. Creating study sequence analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Study Sequence Analysis: Temporal/Ordering Patterns',
             fontsize=16, fontweight='bold', y=0.995)

# Panel 1: Effect sizes by study order
ax1 = axes[0, 0]
ax1.plot(data['study'], data['y'], 'o-', linewidth=2, markersize=10,
         color='steelblue', label='Effect size')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Null effect')
ax1.axhline(y=data['y'].mean(), color='green', linestyle='--', alpha=0.5,
            label=f'Mean = {data["y"].mean():.1f}')
ax1.fill_between(data['study'],
                  data['y'].mean() - data['y'].std(),
                  data['y'].mean() + data['y'].std(),
                  alpha=0.2, color='green', label='±1 SD')
ax1.set_xlabel('Study ID', fontsize=11, fontweight='bold')
ax1.set_ylabel('Effect Size (y)', fontsize=11, fontweight='bold')
ax1.set_title('Effect Sizes by Study Order', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(data['study'])

# Panel 2: Standard errors by study order
ax2 = axes[0, 1]
ax2.plot(data['study'], data['sigma'], 's-', linewidth=2, markersize=10,
         color='coral', label='Standard error')
ax2.axhline(y=data['sigma'].mean(), color='darkred', linestyle='--', alpha=0.5,
            label=f'Mean = {data["sigma"].mean():.1f}')
ax2.fill_between(data['study'],
                  data['sigma'].mean() - data['sigma'].std(),
                  data['sigma'].mean() + data['sigma'].std(),
                  alpha=0.2, color='coral', label='±1 SD')
ax2.set_xlabel('Study ID', fontsize=11, fontweight='bold')
ax2.set_ylabel('Standard Error (sigma)', fontsize=11, fontweight='bold')
ax2.set_title('Standard Errors by Study Order', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(data['study'])

# Panel 3: Precision by study order
ax3 = axes[1, 0]
ax3.bar(data['study'], data['precision'], color='mediumseagreen', alpha=0.7,
        edgecolor='darkgreen', linewidth=1.5)
ax3.axhline(y=data['precision'].mean(), color='darkgreen', linestyle='--',
            linewidth=2, label=f'Mean = {data["precision"].mean():.3f}')
ax3.set_xlabel('Study ID', fontsize=11, fontweight='bold')
ax3.set_ylabel('Precision (1/sigma)', fontsize=11, fontweight='bold')
ax3.set_title('Study Precision by Order', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(data['study'])

# Panel 4: Z-scores for effect sizes
ax4 = axes[1, 1]
colors = ['red' if abs(z) > 1.96 else 'steelblue' for z in data['y_zscore']]
bars = ax4.bar(data['study'], data['y_zscore'], color=colors, alpha=0.7,
               edgecolor='black', linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.axhline(y=1.96, color='red', linestyle='--', linewidth=1, alpha=0.5,
            label='±1.96 (p=0.05)')
ax4.axhline(y=-1.96, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Study ID', fontsize=11, fontweight='bold')
ax4.set_ylabel('Z-score', fontsize=11, fontweight='bold')
ax4.set_title('Standardized Effect Sizes (Z-scores)', fontsize=12, fontweight='bold')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks(data['study'])

plt.tight_layout()
plt.savefig(f'{output_dir}/01_study_sequence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 2. CONFIDENCE INTERVAL FOREST PLOT
# =============================================================================
print("2. Creating confidence interval forest plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Sort by effect size for better visualization
data_sorted = data.sort_values('y', ascending=False).reset_index(drop=True)

# Plot CIs
for i, row in data_sorted.iterrows():
    color = 'darkgreen' if row['ci_lower'] > 0 else ('darkred' if row['ci_upper'] < 0 else 'gray')
    ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 'o-',
            linewidth=2, markersize=8, color=color, alpha=0.7)
    # Add point estimate
    ax.plot(row['y'], i, 'D', markersize=10, color=color,
            markeredgecolor='black', markeredgewidth=1.5)

# Add vertical line at zero
ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5,
           label='Null effect')

# Add mean effect line
mean_y = data['y'].mean()
ax.axvline(x=mean_y, color='blue', linestyle='--', linewidth=2, alpha=0.5,
           label=f'Mean effect = {mean_y:.1f}')

ax.set_yticks(range(len(data_sorted)))
ax.set_yticklabels([f'Study {int(s)}' for s in data_sorted['study']], fontsize=10)
ax.set_xlabel('Effect Size (y) with 95% CI', fontsize=12, fontweight='bold')
ax.set_ylabel('Study', fontsize=12, fontweight='bold')
ax.set_title('Forest Plot: Effect Sizes with 95% Confidence Intervals\n(Sorted by Effect Size)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/02_confidence_interval_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 3. EXTREME VALUE IDENTIFICATION - Multi-panel
# =============================================================================
print("3. Creating extreme value identification plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Extreme Value Identification & Analysis',
             fontsize=16, fontweight='bold', y=0.995)

# Panel 1: Box plots for effect sizes
ax1 = axes[0, 0]
bp1 = ax1.boxplot([data['y']], labels=['Effect Size (y)'],
                   patch_artist=True, widths=0.6)
bp1['boxes'][0].set_facecolor('lightblue')
bp1['boxes'][0].set_alpha(0.7)
# Add individual points
for i, val in enumerate(data['y']):
    color = 'red' if abs(data.loc[i, 'y_zscore']) > 1.96 else 'blue'
    ax1.plot(1, val, 'o', color=color, markersize=10, alpha=0.6,
             label=f'Study {data.loc[i, "study"]}' if abs(data.loc[i, 'y_zscore']) > 1.96 else '')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_ylabel('Effect Size', fontsize=11, fontweight='bold')
ax1.set_title('Effect Size Distribution with Outliers', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Box plots for standard errors
ax2 = axes[0, 1]
bp2 = ax2.boxplot([data['sigma']], labels=['Standard Error (sigma)'],
                   patch_artist=True, widths=0.6)
bp2['boxes'][0].set_facecolor('lightcoral')
bp2['boxes'][0].set_alpha(0.7)
# Add individual points
for i, val in enumerate(data['sigma']):
    color = 'red' if abs(data.loc[i, 'sigma_zscore']) > 1.96 else 'coral'
    ax2.plot(1, val, 's', color=color, markersize=10, alpha=0.6)
ax2.set_ylabel('Standard Error', fontsize=11, fontweight='bold')
ax2.set_title('Standard Error Distribution with Outliers', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Scatter plot of y vs sigma
ax3 = axes[1, 0]
scatter = ax3.scatter(data['sigma'], data['y'], s=200, c=data['study'],
                      cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
# Add study labels
for i, row in data.iterrows():
    ax3.annotate(f"{int(row['study'])}", (row['sigma'], row['y']),
                fontsize=10, fontweight='bold', ha='center', va='center')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3.set_xlabel('Standard Error (sigma)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Effect Size (y)', fontsize=11, fontweight='bold')
ax3.set_title(f'Effect Size vs Standard Error\n(Pearson r = {data["y"].corr(data["sigma"]):.3f})',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Study ID', fontsize=10, fontweight='bold')

# Panel 4: Study weights
ax4 = axes[1, 1]
colors_weight = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
bars = ax4.barh(data['study'], data['weight'], color=colors_weight,
                alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Weight (1/sigma²)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Study ID', fontsize=11, fontweight='bold')
ax4.set_title('Study Weights in Meta-Analysis\n(Higher = More Precise)',
              fontsize=12, fontweight='bold')
ax4.set_yticks(data['study'])
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/03_extreme_value_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. DATA QUALITY SUMMARY - Multi-panel
# =============================================================================
print("4. Creating data quality summary...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Quality & Structure Assessment',
             fontsize=16, fontweight='bold', y=0.995)

# Panel 1: Histogram of effect sizes
ax1 = axes[0, 0]
ax1.hist(data['y'], bins=6, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axvline(data['y'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean = {data["y"].mean():.2f}')
ax1.axvline(data['y'].median(), color='green', linestyle='--', linewidth=2,
            label=f'Median = {data["y"].median():.2f}')
ax1.set_xlabel('Effect Size (y)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title(f'Effect Size Distribution\nSkewness = {stats.skew(data["y"]):.2f}, Kurtosis = {stats.kurtosis(data["y"]):.2f}',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Histogram of standard errors
ax2 = axes[0, 1]
ax2.hist(data['sigma'], bins=6, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axvline(data['sigma'].mean(), color='darkred', linestyle='--', linewidth=2,
            label=f'Mean = {data["sigma"].mean():.2f}')
ax2.axvline(data['sigma'].median(), color='green', linestyle='--', linewidth=2,
            label=f'Median = {data["sigma"].median():.2f}')
ax2.set_xlabel('Standard Error (sigma)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title(f'Standard Error Distribution\nCV = {data["sigma"].std()/data["sigma"].mean():.2f}',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Q-Q plot for effect sizes
ax3 = axes[1, 0]
stats.probplot(data['y'], dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot: Effect Size Normality Check', fontsize=12, fontweight='bold')
ax3.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
ax3.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel 4: Sign consistency
ax4 = axes[1, 1]
sign_counts = [sum(data['y'] > 0), sum(data['y'] < 0), sum(data['y'] == 0)]
labels = ['Positive\n(>0)', 'Negative\n(<0)', 'Zero\n(=0)']
colors_sign = ['green', 'red', 'gray']
wedges, texts, autotexts = ax4.pie(sign_counts, labels=labels, autopct='%1.1f%%',
                                     colors=colors_sign, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Effect Size Sign Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/04_data_quality_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. SINGLE COMPREHENSIVE SUMMARY PLOT
# =============================================================================
print("5. Creating comprehensive summary visualization...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Meta-Analysis Dataset: Comprehensive Structural Overview (J=8 Studies)',
             fontsize=18, fontweight='bold', y=0.98)

# Plot 1: Main effect size plot with CIs
ax1 = fig.add_subplot(gs[0, :])
for i, row in data.iterrows():
    color = 'darkgreen' if row['ci_lower'] > 0 else ('darkred' if row['ci_upper'] < 0 else 'steelblue')
    ax1.errorbar(row['study'], row['y'], yerr=1.96*row['sigma'],
                fmt='o', markersize=12, linewidth=2.5, capsize=8, capthick=2.5,
                color=color, alpha=0.7, label=f"Study {int(row['study'])}" if i < 3 else "")
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Null effect')
ax1.axhline(y=data['y'].mean(), color='blue', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Mean effect = {data["y"].mean():.1f}')
ax1.fill_between(range(0, 10),
                  data['y'].mean() - data['y'].std(),
                  data['y'].mean() + data['y'].std(),
                  alpha=0.15, color='blue', label='±1 SD')
ax1.set_xlabel('Study ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('Effect Size (y) ± 95% CI', fontsize=12, fontweight='bold')
ax1.set_title('Effect Sizes with 95% Confidence Intervals', fontsize=13, fontweight='bold')
ax1.set_xticks(data['study'])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9, ncol=2)

# Plot 2: Distribution of effect sizes
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(data['y'], bins=6, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5, density=True)
ax2.axvline(data['y'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.axvline(data['y'].median(), color='green', linestyle='--', linewidth=2, label='Median')
ax2.set_xlabel('Effect Size (y)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Density', fontsize=10, fontweight='bold')
ax2.set_title('Effect Size Distribution', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of standard errors
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(data['sigma'], bins=6, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5, density=True)
ax3.axvline(data['sigma'].mean(), color='darkred', linestyle='--', linewidth=2, label='Mean')
ax3.axvline(data['sigma'].median(), color='green', linestyle='--', linewidth=2, label='Median')
ax3.set_xlabel('Standard Error (sigma)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Density', fontsize=10, fontweight='bold')
ax3.set_title('Standard Error Distribution', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Effect vs SE scatter
ax4 = fig.add_subplot(gs[1, 2])
scatter = ax4.scatter(data['sigma'], data['y'], s=300, c=data['precision'],
                      cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2)
for i, row in data.iterrows():
    ax4.annotate(f"{int(row['study'])}", (row['sigma'], row['y']),
                fontsize=9, fontweight='bold', ha='center', va='center')
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('Standard Error (sigma)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Effect Size (y)', fontsize=10, fontweight='bold')
ax4.set_title(f'Effect vs SE (r={data["y"].corr(data["sigma"]):.2f})', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Precision', fontsize=9)

# Plot 5: Study weights
ax5 = fig.add_subplot(gs[2, 0])
colors_weight = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
bars = ax5.bar(data['study'], data['weight'], color=colors_weight,
               alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Study ID', fontsize=10, fontweight='bold')
ax5.set_ylabel('Weight (1/sigma²)', fontsize=10, fontweight='bold')
ax5.set_title('Study Weights', fontsize=11, fontweight='bold')
ax5.set_xticks(data['study'])
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: CI widths
ax6 = fig.add_subplot(gs[2, 1])
ax6.bar(data['study'], data['ci_width'], color='mediumpurple',
        alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.axhline(y=data['ci_width'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean = {data["ci_width"].mean():.1f}')
ax6.set_xlabel('Study ID', fontsize=10, fontweight='bold')
ax6.set_ylabel('95% CI Width', fontsize=10, fontweight='bold')
ax6.set_title('Confidence Interval Widths', fontsize=11, fontweight='bold')
ax6.set_xticks(data['study'])
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Summary statistics table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

summary_text = f"""
KEY STATISTICS

Sample Size:
  J = {len(data)} studies (Small meta-analysis)

Effect Size (y):
  Mean: {data['y'].mean():.2f}
  Median: {data['y'].median():.2f}
  SD: {data['y'].std():.2f}
  Range: [{data['y'].min():.0f}, {data['y'].max():.0f}]
  Skewness: {stats.skew(data['y']):.2f}

Standard Error (sigma):
  Mean: {data['sigma'].mean():.2f}
  Median: {data['sigma'].median():.2f}
  CV: {data['sigma'].std()/data['sigma'].mean():.2f}
  Range: [{data['sigma'].min():.0f}, {data['sigma'].max():.0f}]

Heterogeneity:
  Positive effects: {sum(data['y'] > 0)}/{len(data)} (75%)
  CIs including 0: {sum((data['ci_lower'] <= 0) & (data['ci_upper'] >= 0))}/{len(data)} (100%)

Correlations:
  y vs sigma: r = {data['y'].corr(data['sigma']):.3f}
  Study ID vs y: r = {data['study'].corr(data['y']):.3f}
"""

ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(f'{output_dir}/05_comprehensive_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. FUNNEL PLOT (Publication Bias Check)
# =============================================================================
print("6. Creating funnel plot for publication bias assessment...")

fig, ax = plt.subplots(figsize=(10, 8))

# Invert y-axis for funnel plot (more precise studies at top)
ax.scatter(data['y'], data['sigma'], s=200, alpha=0.7,
           c=data['study'], cmap='viridis', edgecolors='black', linewidth=2)

# Add study labels
for i, row in data.iterrows():
    ax.annotate(f"{int(row['study'])}", (row['y'], row['sigma']),
                fontsize=10, fontweight='bold', ha='center', va='center')

# Add vertical line at mean effect
mean_effect = data['y'].mean()
ax.axvline(x=mean_effect, color='red', linestyle='--', linewidth=2,
           label=f'Mean effect = {mean_effect:.1f}')

# Add funnel boundaries (pseudo 95% CI)
y_range = np.linspace(data['sigma'].min() * 0.8, data['sigma'].max() * 1.1, 100)
ax.plot(mean_effect + 1.96 * y_range, y_range, 'b--', alpha=0.5, linewidth=2,
        label='95% funnel')
ax.plot(mean_effect - 1.96 * y_range, y_range, 'b--', alpha=0.5, linewidth=2)

ax.invert_yaxis()
ax.set_xlabel('Effect Size (y)', fontsize=12, fontweight='bold')
ax.set_ylabel('Standard Error (sigma)', fontsize=12, fontweight='bold')
ax.set_title('Funnel Plot: Publication Bias Assessment\n(Small-study effects check)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add text annotation
if abs(data['y'].corr(data['sigma'])) > 0.3:
    ax.text(0.05, 0.05, 'WARNING: Possible asymmetry detected\n(correlation > 0.3)',
            transform=ax.transAxes, fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
else:
    ax.text(0.05, 0.05, 'No strong asymmetry detected',
            transform=ax.transAxes, fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}/06_funnel_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations created successfully!")
print(f"Output directory: {output_dir}")
print("\nGenerated files:")
print("  1. 01_study_sequence_analysis.png")
print("  2. 02_confidence_interval_forest_plot.png")
print("  3. 03_extreme_value_analysis.png")
print("  4. 04_data_quality_summary.png")
print("  5. 05_comprehensive_summary.png")
print("  6. 06_funnel_plot.png")
