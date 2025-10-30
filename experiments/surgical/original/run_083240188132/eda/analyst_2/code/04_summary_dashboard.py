"""
Summary Dashboard - Analyst 2
Comprehensive one-page visual summary of key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

# Calculate confidence intervals
def wilson_ci(r, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = r / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
    return center - margin, center + margin

data['ci_lower'], data['ci_upper'] = zip(*data.apply(lambda row: wilson_ci(row['r'], row['n']), axis=1))

# Calculate key statistics
pooled_p = data['r'].sum() / data['n'].sum()
data['expected_r'] = data['n'] * pooled_p
data['std_residual'] = (data['r'] - data['expected_r']) / np.sqrt(data['expected_r'] * (1 - pooled_p))

# Create comprehensive summary dashboard
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Add title
fig.suptitle('EDA Summary Dashboard: Patterns, Structure, and Relationships\nBinomial Outcome Data (12 Groups, N=2,814, Events=208)',
             fontsize=16, fontweight='bold', y=0.98)

# ============================================================================
# Panel 1: Main result - Proportions with CI (TOP LEFT - larger)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])
colors = ['red' if r == 0 else '#FF6B6B' if abs(z) > 2 else '#4ECDC4'
          for r, z in zip(data['r'], data['std_residual'])]
ax1.errorbar(data['group'], data['proportion'],
            yerr=[data['proportion'] - data['ci_lower'], data['ci_upper'] - data['proportion']],
            fmt='o', markersize=10, capsize=6, linewidth=2.5, color='#2E86AB', elinewidth=2, alpha=0.7)
for i, (idx, row) in enumerate(data.iterrows()):
    ax1.scatter(row['group'], row['proportion'], s=200, color=colors[i],
               edgecolors='black', linewidth=2, zorder=3, alpha=0.8)
ax1.axhline(pooled_p, color='orange', linestyle='--', linewidth=3, alpha=0.8, label=f'Pooled: {pooled_p:.3f}')
ax1.fill_between(data['group'], pooled_p - 0.02, pooled_p + 0.02, color='orange', alpha=0.1)
ax1.set_xlabel('Group', fontsize=13, fontweight='bold')
ax1.set_ylabel('Proportion', fontsize=13, fontweight='bold')
ax1.set_title('Group-Level Proportions with 95% Confidence Intervals\n(Red: zero/outliers, Teal: normal)',
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xticks(data['group'])
ax1.set_ylim(-0.01, 0.18)

# ============================================================================
# Panel 2: Key statistics box (TOP RIGHT)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
KEY STATISTICS

Overall:
  Pooled rate: {pooled_p:.3f}
  95% CI: [{data['ci_lower'].iloc[0]:.3f}, {data['ci_upper'].iloc[0]:.3f}]
  Total N: {data['n'].sum():,}
  Total events: {data['r'].sum()}

Heterogeneity:
  Chi² test: p < 0.0001 ✓
  Overdispersion: φ = 3.51
  ICC: 0.66 (66% between-group)
  I² statistic: 71.5%

Patterns:
  Sequential trend: p = 0.20 ✗
  Sample size bias: p = 0.99 ✗

Special Cases:
  Zero-event groups: 1 (Group 1)
  Outliers (|z|>2): 3 (Groups 2, 8, 11)
  Small-n (n<100): 3 groups

RECOMMENDATION:
Hierarchical (Partial Pooling) Model
→ Beta-Binomial or Random Effects
"""
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================================
# Panel 3: Sample sizes (MIDDLE LEFT)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])
colors_n = ['#FF6B6B' if n < 100 else '#95E1D3' if n > 500 else '#4ECDC4' for n in data['n']]
bars = ax3.bar(data['group'], data['n'], color=colors_n, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axhline(data['n'].median(), color='red', linestyle='--', linewidth=2, alpha=0.7, label='Median')
ax3.set_xlabel('Group', fontsize=11, fontweight='bold')
ax3.set_ylabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax3.set_title('Sample Size Distribution\n(Red: <100, Green: >500)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y', linestyle='--')
ax3.set_xticks(data['group'])

# ============================================================================
# Panel 4: Standardized residuals (MIDDLE CENTER)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
colors_resid = ['#FF6B6B' if abs(z) > 2 else '#4ECDC4' for z in data['std_residual']]
bars = ax4.bar(data['group'], data['std_residual'], color=colors_resid, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axhline(0, color='black', linestyle='-', linewidth=2)
ax4.axhline(2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='±2 SD threshold')
ax4.axhline(-2, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax4.fill_between(data['group'], -2, 2, color='green', alpha=0.1)
ax4.set_xlabel('Group', fontsize=11, fontweight='bold')
ax4.set_ylabel('Standardized Residual (z)', fontsize=11, fontweight='bold')
ax4.set_title('Deviation from Pooled Model\n(Red: outliers |z|>2)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, axis='y', linestyle='--')
ax4.set_xticks(data['group'])

# ============================================================================
# Panel 5: n vs proportion (MIDDLE RIGHT)
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
scatter = ax5.scatter(data['n'], data['proportion'], s=data['r']*5,
                     c=data['group'], cmap='viridis', alpha=0.7,
                     edgecolors='black', linewidth=2)
ax5.axhline(pooled_p, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Pooled')
# Add labels for outliers
for idx, row in data.iterrows():
    if abs(row['std_residual']) > 2 or row['r'] == 0:
        ax5.annotate(f"G{row['group']}", (row['n'], row['proportion']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
ax5.set_xlabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Proportion', fontsize=11, fontweight='bold')
ax5.set_title('Sample Size vs Proportion\n(bubble size = events)', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3, linestyle='--')
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Group', fontsize=10)

# ============================================================================
# Panel 6: Pooling comparison (BOTTOM LEFT)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 0])
# Simple empirical Bayes shrinkage
data_nonzero = data[data['r'] > 0]
within_var = np.average(data_nonzero['proportion'] * (1 - data_nonzero['proportion']) / data_nonzero['n'],
                       weights=data_nonzero['n'])
grand_mean = np.average(data['proportion'], weights=data['n'])
between_var = max(0, np.var(data_nonzero['proportion']) - within_var/np.mean(data_nonzero['n']))

x_pos = np.arange(len(data))
width = 0.25
ax6.bar(x_pos - width, data['proportion'], width, label='No pooling',
       alpha=0.7, color='steelblue', edgecolor='black')
ax6.bar(x_pos, [pooled_p]*len(data), width, label='Complete pooling',
       alpha=0.7, color='orange', edgecolor='black')
# Calculate partial pooling estimates
partial = []
for idx, row in data.iterrows():
    shrinkage = within_var / (within_var + between_var * row['n'])
    partial.append(shrinkage * grand_mean + (1 - shrinkage) * row['proportion'])
ax6.bar(x_pos + width, partial, width, label='Partial pooling',
       alpha=0.7, color='green', edgecolor='black')
ax6.set_xlabel('Group', fontsize=11, fontweight='bold')
ax6.set_ylabel('Proportion Estimate', fontsize=11, fontweight='bold')
ax6.set_title('Pooling Strategy Comparison\n(Green recommended)', fontsize=11, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(data['group'])
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3, axis='y', linestyle='--')

# ============================================================================
# Panel 7: Variance components (BOTTOM CENTER)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 1])
variance_data = {
    'Within-group\n(expected)': [within_var],
    'Between-group\n(excess)': [between_var],
}
x_var = np.arange(len(variance_data))
colors_var = ['#4ECDC4', '#FF6B6B']
for i, (label, values) in enumerate(variance_data.items()):
    ax7.bar(i, values[0], color=colors_var[i], alpha=0.7, edgecolor='black', linewidth=2)
    ax7.text(i, values[0]/2, f'{values[0]:.6f}', ha='center', va='center',
            fontweight='bold', fontsize=11, color='white')
ax7.set_ylabel('Variance', fontsize=11, fontweight='bold')
ax7.set_title(f'Variance Decomposition\nICC = {between_var/(between_var+within_var):.3f}',
             fontsize=11, fontweight='bold')
ax7.set_xticks(x_var)
ax7.set_xticklabels(variance_data.keys(), fontsize=10)
ax7.grid(alpha=0.3, axis='y', linestyle='--')
# Add annotation
ax7.text(0.5, 0.95, '66% of variance is\nbetween groups',
        transform=ax7.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
        fontsize=10, fontweight='bold')

# ============================================================================
# Panel 8: Model recommendations (BOTTOM RIGHT)
# ============================================================================
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
model_text = """
MODEL RECOMMENDATIONS

PRIMARY (Strongly Recommended):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hierarchical/Partial Pooling Model

Options:
1. Beta-Binomial
   • Handles overdispersion (φ=3.5)
   • Closed-form inference

2. Random Effects Logit
   • logit(p_i) = μ + α_i
   • α_i ~ Normal(0, τ²)
   • Standard software (lme4)

3. Bayesian Hierarchical
   • Full uncertainty
   • Handles Group 1 naturally

ALTERNATIVE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Finite Mixture (2 components)
• Low risk: ~7% (9 groups)
• High risk: ~12-14% (3 groups)

NOT RECOMMENDED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
× Complete pooling (p<0.0001)
× No pooling (overfits small n)
"""
ax8.text(0.05, 0.95, model_text, transform=ax8.transAxes, fontsize=9.5,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.savefig('/workspace/eda/analyst_2/visualizations/00_summary_dashboard.png',
           dpi=300, bbox_inches='tight')
plt.close()

print("Summary dashboard created successfully!")
print("Saved to: /workspace/eda/analyst_2/visualizations/00_summary_dashboard.png")
