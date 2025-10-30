"""
Comprehensive Visualizations for Distributions and Heterogeneity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/eda/analyst_1/code/processed_data_with_metrics.csv')
heterogeneity = pd.read_csv('/workspace/eda/analyst_1/code/heterogeneity_results.csv')

# Extract key values
fixed_effect = heterogeneity['fixed_effect'].values[0]
pred_lower = heterogeneity['prediction_lower'].values[0]
pred_upper = heterogeneity['prediction_upper'].values[0]

# ============================================================================
# VISUALIZATION 1: Forest Plot (Classic Meta-Analysis Visualization)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Sort by effect size for better visualization
data_sorted = data.sort_values('y', ascending=True).reset_index(drop=True)

# Plot individual studies
y_positions = np.arange(len(data_sorted))
for i, (idx, row) in enumerate(data_sorted.iterrows()):
    # Error bars for confidence intervals
    ax.errorbar(row['y'], i, xerr=[[row['y'] - row['ci_lower']],
                                      [row['ci_upper'] - row['y']]],
                fmt='o', color='steelblue', markersize=8, capsize=5, capthick=2,
                linewidth=2, alpha=0.7)

    # Add study label
    ax.text(-6, i, f"Study {int(row['study'])}", va='center', ha='right', fontsize=9)

    # Add effect estimate text
    ax.text(32, i, f"{row['y']:.1f} [{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]",
            va='center', ha='left', fontsize=8)

# Add pooled estimate
ax.axvline(fixed_effect, color='darkred', linestyle='--', linewidth=2,
           label=f'Pooled effect: {fixed_effect:.2f}', alpha=0.8)

# Add prediction interval
ax.axvspan(pred_lower, pred_upper, alpha=0.15, color='orange',
           label=f'95% Prediction interval')

# Add zero line
ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Formatting
ax.set_yticks(y_positions)
ax.set_yticklabels([''] * len(data_sorted))
ax.set_xlabel('Effect Size', fontsize=12, fontweight='bold')
ax.set_title('Forest Plot: Individual Study Estimates with 95% CIs',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(-10, 65)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: forest_plot.png")

# ============================================================================
# VISUALIZATION 2: Distribution Panel (4 subplots)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Histogram of effect sizes
ax1 = axes[0, 0]
ax1.hist(data['y'], bins=6, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(data['y'].mean(), color='darkred', linestyle='--', linewidth=2,
            label=f'Mean: {data["y"].mean():.2f}')
ax1.axvline(data['y'].median(), color='orange', linestyle='--', linewidth=2,
            label=f'Median: {data["y"].median():.2f}')
ax1.set_xlabel('Effect Size (y)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('A. Distribution of Effect Sizes', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Panel B: Histogram of standard errors
ax2 = axes[0, 1]
ax2.hist(data['sigma'], bins=6, color='seagreen', alpha=0.7, edgecolor='black')
ax2.axvline(data['sigma'].mean(), color='darkred', linestyle='--', linewidth=2,
            label=f'Mean: {data["sigma"].mean():.2f}')
ax2.axvline(data['sigma'].median(), color='orange', linestyle='--', linewidth=2,
            label=f'Median: {data["sigma"].median():.2f}')
ax2.set_xlabel('Standard Error (sigma)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('B. Distribution of Standard Errors', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Panel C: Q-Q plot for effect sizes
ax3 = axes[1, 0]
stats.probplot(data['y'], dist="norm", plot=ax3)
ax3.set_title('C. Q-Q Plot: Effect Sizes vs Normal', fontsize=12, fontweight='bold')
ax3.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
ax3.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3)

# Panel D: Box plots comparison
ax4 = axes[1, 1]
box_data = [data['y'], data['sigma']]
bp = ax4.boxplot(box_data, labels=['Effect Size (y)', 'Std Error (sigma)'],
                  patch_artist=True, notch=True, showmeans=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('seagreen')
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.5)
ax4.set_ylabel('Value', fontsize=11, fontweight='bold')
ax4.set_title('D. Boxplot Comparison', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Distribution Characteristics: Effect Sizes and Standard Errors',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/distribution_panel.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: distribution_panel.png")

# ============================================================================
# VISUALIZATION 3: Precision vs Effect Size
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot with study labels
colors = ['red' if (row['y'] < pred_lower or row['y'] > pred_upper)
          else 'steelblue' for idx, row in data.iterrows()]

for idx, row in data.iterrows():
    color = 'red' if (row['y'] < pred_lower or row['y'] > pred_upper) else 'steelblue'
    ax.scatter(row['sigma'], row['y'], s=200, c=color, alpha=0.6, edgecolors='black', linewidth=1.5)
    ax.text(row['sigma'], row['y'], f" {int(row['study'])}", fontsize=10, fontweight='bold')

# Add regression line
z = np.polyfit(data['sigma'], data['y'], 1)
p = np.poly1d(z)
x_line = np.linspace(data['sigma'].min(), data['sigma'].max(), 100)
ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label=f'Trend line (slope={z[0]:.2f})')

# Calculate correlation
corr, p_val = stats.pearsonr(data['sigma'], data['y'])
ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}\np-value = {p_val:.3f}",
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add reference lines
ax.axhline(fixed_effect, color='darkgreen', linestyle='--', linewidth=2,
           alpha=0.5, label=f'Pooled effect: {fixed_effect:.2f}')
ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

ax.set_xlabel('Standard Error (sigma) - Lower is More Precise', fontsize=12, fontweight='bold')
ax.set_ylabel('Effect Size (y)', fontsize=12, fontweight='bold')
ax.set_title('Effect Size vs Measurement Precision\n(Red = Outside Prediction Interval)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/precision_vs_effect.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: precision_vs_effect.png")

# ============================================================================
# VISUALIZATION 4: Heterogeneity Diagnostics Panel
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Contribution to Q statistic
ax1 = axes[0, 0]
data_ranked = data.sort_values('Q_contribution', ascending=True)
colors_q = ['red' if pct > 25 else 'steelblue' for pct in data_ranked['Q_contribution_pct']]
ax1.barh(range(len(data_ranked)), data_ranked['Q_contribution_pct'], color=colors_q, alpha=0.7)
ax1.set_yticks(range(len(data_ranked)))
ax1.set_yticklabels([f"Study {int(s)}" for s in data_ranked['study']])
ax1.set_xlabel('Contribution to Q (%)', fontsize=11, fontweight='bold')
ax1.set_title('A. Study Contribution to Heterogeneity', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(100/len(data), color='black', linestyle='--', linewidth=1,
            label='Equal contribution')
ax1.legend()

# Panel B: Standardized residuals
ax2 = axes[0, 1]
data_resid = data.sort_values('std_residual', ascending=True)
colors_resid = ['red' if abs(r) > 1.5 else 'steelblue' for r in data_resid['std_residual']]
ax2.barh(range(len(data_resid)), data_resid['std_residual'], color=colors_resid, alpha=0.7)
ax2.set_yticks(range(len(data_resid)))
ax2.set_yticklabels([f"Study {int(s)}" for s in data_resid['study']])
ax2.set_xlabel('Standardized Residual', fontsize=11, fontweight='bold')
ax2.set_title('B. Standardized Residuals from Pooled Effect', fontsize=12, fontweight='bold')
ax2.axvline(0, color='black', linewidth=1)
ax2.axvline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(axis='x', alpha=0.3)

# Panel C: Confidence interval widths
ax3 = axes[1, 0]
data_ci = data.sort_values('ci_width', ascending=True)
ax3.barh(range(len(data_ci)), data_ci['ci_width'], color='seagreen', alpha=0.7)
ax3.set_yticks(range(len(data_ci)))
ax3.set_yticklabels([f"Study {int(s)}" for s in data_ci['study']])
ax3.set_xlabel('95% CI Width', fontsize=11, fontweight='bold')
ax3.set_title('C. Uncertainty: Confidence Interval Widths', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.axvline(data['ci_width'].mean(), color='darkred', linestyle='--', linewidth=2,
            label=f'Mean: {data["ci_width"].mean():.1f}')
ax3.legend()

# Panel D: Weight distribution
ax4 = axes[1, 1]
data_weight = data.sort_values('weight', ascending=True)
ax4.barh(range(len(data_weight)), data_weight['weight'] / data_weight['weight'].sum() * 100,
         color='orange', alpha=0.7)
ax4.set_yticks(range(len(data_weight)))
ax4.set_yticklabels([f"Study {int(s)}" for s in data_weight['study']])
ax4.set_xlabel('Weight in Meta-Analysis (%)', fontsize=11, fontweight='bold')
ax4.set_title('D. Relative Study Weights (Inverse Variance)', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
ax4.axvline(100/len(data), color='black', linestyle='--', linewidth=1,
            label='Equal weight')
ax4.legend()

plt.suptitle('Heterogeneity Diagnostic Panel', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/heterogeneity_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: heterogeneity_diagnostics.png")

# ============================================================================
# VISUALIZATION 5: Funnel Plot (Publication Bias Check)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Invert y-axis (standard error) so most precise at top
ax.scatter(data['y'], data['sigma'], s=150, c='steelblue', alpha=0.6,
           edgecolors='black', linewidth=1.5)

# Add study labels
for idx, row in data.iterrows():
    ax.text(row['y'], row['sigma'], f" {int(row['study'])}", fontsize=9)

# Add pooled effect line
ax.axvline(fixed_effect, color='darkred', linestyle='--', linewidth=2,
           label=f'Pooled effect: {fixed_effect:.2f}')

# Add funnel (pseudo 95% confidence limits)
# The funnel should be based on the standard error
se_range = np.linspace(data['sigma'].min(), data['sigma'].max(), 100)
lower_limit = fixed_effect - 1.96 * se_range
upper_limit = fixed_effect + 1.96 * se_range

ax.plot(lower_limit, se_range, 'k--', linewidth=1, alpha=0.5, label='95% Pseudo-CI')
ax.plot(upper_limit, se_range, 'k--', linewidth=1, alpha=0.5)

# Invert y-axis
ax.invert_yaxis()

ax.set_xlabel('Effect Size (y)', fontsize=12, fontweight='bold')
ax.set_ylabel('Standard Error (sigma) - More Precise Upward', fontsize=12, fontweight='bold')
ax.set_title('Funnel Plot: Assessment of Asymmetry', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/funnel_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: funnel_plot.png")

# ============================================================================
# VISUALIZATION 6: Cumulative Meta-Analysis
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Sort by precision (inverse of standard error)
data_cumulative = data.sort_values('sigma', ascending=True).reset_index(drop=True)

cumulative_effects = []
cumulative_lower = []
cumulative_upper = []

for i in range(1, len(data_cumulative) + 1):
    subset = data_cumulative.iloc[:i]
    weights = 1 / (subset['sigma'] ** 2)
    cum_effect = sum(weights * subset['y']) / sum(weights)
    cum_se = np.sqrt(1 / sum(weights))

    cumulative_effects.append(cum_effect)
    cumulative_lower.append(cum_effect - 1.96 * cum_se)
    cumulative_upper.append(cum_effect + 1.96 * cum_se)

x_positions = np.arange(1, len(data_cumulative) + 1)

# Plot cumulative effect with CI
ax.plot(x_positions, cumulative_effects, 'o-', color='darkblue', linewidth=2,
        markersize=8, label='Cumulative effect')
ax.fill_between(x_positions, cumulative_lower, cumulative_upper, alpha=0.3,
                color='steelblue', label='95% CI')

# Add final pooled estimate
ax.axhline(fixed_effect, color='darkred', linestyle='--', linewidth=2,
           label=f'Final pooled: {fixed_effect:.2f}')
ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Add study labels on x-axis
ax.set_xticks(x_positions)
ax.set_xticklabels([f"S{int(s)}" for s in data_cumulative['study']], fontsize=9)

ax.set_xlabel('Studies Added (sorted by precision)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Effect Estimate', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Meta-Analysis: Effect Stability', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/cumulative_meta_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: cumulative_meta_analysis.png")

print("\n" + "="*70)
print("All visualizations created successfully!")
print("="*70)
