"""
Summary Visualization: Comprehensive overview of hierarchical structure evidence
This creates a single multi-panel figure summarizing all key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Setup
BASE_DIR = Path("/workspace/eda/analyst_2")
VIZ_DIR = BASE_DIR / "visualizations"
df = pd.read_csv(BASE_DIR / "code" / "clustering_analysis.csv")

pooled_rate = df['r_successes'].sum() / df['n_trials'].sum()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Create comprehensive 2x3 summary figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# Panel 1: Caterpillar plot with shrinkage
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

df_sorted = df.sort_values('success_rate')
y_positions = range(len(df_sorted))

# Observed rates with CIs
ax1.hlines(y=y_positions, xmin=df_sorted['ci_lower'], xmax=df_sorted['ci_upper'],
          color='steelblue', alpha=0.6, linewidth=2)
ax1.scatter(df_sorted['success_rate'], y_positions,
           s=df_sorted['n_trials']/5, color='darkblue', alpha=0.7, zorder=5)

# Pooled rate
ax1.axvline(pooled_rate, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Pooled: {pooled_rate:.3f}')

ax1.set_yticks(y_positions)
ax1.set_yticklabels([f"G{int(g)}" for g in df_sorted['group']], fontsize=9)
ax1.set_xlabel('Success Rate', fontsize=11)
ax1.set_ylabel('Group', fontsize=11)
ax1.set_title('(A) Group Success Rates with 95% CIs\n(sorted, sized by n)',
             fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# ============================================================================
# Panel 2: Variance decomposition visualization
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Calculate components
observed_var = df['success_rate'].var(ddof=1)
expected_var_binomial = pooled_rate * (1 - pooled_rate) / df['n_trials'].mean()
between_group_var = 0.000778  # From DL estimator

# Bar plot
components = ['Observed\nVariance', 'Expected\n(Binomial)', 'Between-Group\nVariance']
values = [observed_var, expected_var_binomial, between_group_var]
colors_bar = ['#d62728', '#ff7f0e', '#2ca02c']

bars = ax2.bar(components, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add ratio annotation
ax2.annotate(f'Ratio: {observed_var/expected_var_binomial:.1f}x',
            xy=(0.5, observed_var*0.8), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax2.set_ylabel('Variance', fontsize=11)
ax2.set_title('(B) Variance Decomposition\n(Overdispersion = 5.06x)',
             fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# ============================================================================
# Panel 3: ICC and variance partition
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

# Pie chart for variance partition
icc = 0.727
variance_components = [icc, 1-icc]
labels_pie = [f'Between-Group\n{icc*100:.1f}%', f'Within-Group\n{(1-icc)*100:.1f}%']
colors_pie = ['#2ca02c', '#ff7f0e']

wedges, texts, autotexts = ax3.pie(variance_components, labels=labels_pie, colors=colors_pie,
                                     autopct='', startangle=90, textprops={'fontsize': 11})

# Bold the labels
for text in texts:
    text.set_fontweight('bold')

ax3.set_title('(C) Variance Partition (ICC)\nICC = 0.727',
             fontsize=12, fontweight='bold')

# Add text box with interpretation
textstr = 'ICC > 0.1:\nHierarchical\nmodeling\nbeneficial'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
ax3.text(1.3, 0, textstr, fontsize=10, ha='left', va='center',
        bbox=props, fontweight='bold')

# ============================================================================
# Panel 4: Shrinkage factors by sample size
# ============================================================================
ax4 = fig.add_subplot(gs[1, 0])

ax4.scatter(df['n_trials'], df['shrinkage_factor'], s=150, color='purple', alpha=0.7,
           edgecolors='black', linewidth=1.5)

# Add labels for extreme cases
for _, row in df.iterrows():
    if row['group'] in [1, 4, 8]:  # Label smallest, largest, and extreme rate
        ax4.annotate(f"G{int(row['group'])}\n({row['shrinkage_pct']:.0f}% shrink)",
                    (row['n_trials'], row['shrinkage_factor']),
                    xytext=(10, -10), textcoords='offset points',
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax4.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax4.set_xlabel('Sample Size (n_trials)', fontsize=11)
ax4.set_ylabel('Shrinkage Factor λ', fontsize=11)
ax4.set_title('(D) Shrinkage by Sample Size\n(Mean shrinkage: 85.6%)',
             fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.set_ylim(-0.05, 1.05)

# ============================================================================
# Panel 5: Distribution normality
# ============================================================================
ax5 = fig.add_subplot(gs[1, 1])

# Q-Q plot
stats.probplot(df['success_rate'], dist="norm", plot=ax5)
ax5.get_lines()[0].set_markerfacecolor('steelblue')
ax5.get_lines()[0].set_markeredgecolor('darkblue')
ax5.get_lines()[0].set_markersize(8)
ax5.get_lines()[0].set_alpha(0.7)
ax5.get_lines()[1].set_color('red')
ax5.get_lines()[1].set_linewidth(2)

# Add test result
shapiro_stat, shapiro_p = stats.shapiro(df['success_rate'])
textstr = f'Shapiro-Wilk:\np = {shapiro_p:.3f}\n\nCannot reject\nnormality'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontweight='bold')

ax5.set_title('(E) Normal Distribution Test\n(supports normal prior)',
             fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# ============================================================================
# Panel 6: Three pooling strategies compared
# ============================================================================
ax6 = fig.add_subplot(gs[1, 2])

y_pos = range(len(df))

# No pooling (observed)
ax6.scatter(df['success_rate'], y_pos, s=100, color='darkblue',
           alpha=0.7, label='No pooling', zorder=5, marker='o')

# Complete pooling
ax6.scatter([pooled_rate]*len(df), y_pos, s=100, color='red',
           alpha=0.7, label='Complete pooling', zorder=5, marker='s')

# Partial pooling
ax6.scatter(df['partial_pooled_rate'], y_pos, s=100, color='green',
           alpha=0.7, label='Partial pooling', zorder=5, marker='^')

# Connect with lines for a few examples
for i in [0, 3, 7]:  # Groups 1, 4, 8
    ax6.plot([df.iloc[i]['success_rate'], df.iloc[i]['partial_pooled_rate'], pooled_rate],
            [i, i, i], color='gray', alpha=0.5, linestyle='--', linewidth=1)

ax6.set_yticks(y_pos)
ax6.set_yticklabels([f"G{int(g)}" for g in df['group']], fontsize=9)
ax6.set_xlabel('Success Rate', fontsize=11)
ax6.set_ylabel('Group', fontsize=11)
ax6.set_title('(F) Three Pooling Strategies\n(green = hierarchical)',
             fontsize=12, fontweight='bold')
ax6.legend(loc='lower right', fontsize=9)
ax6.grid(axis='x', alpha=0.3)

# ============================================================================
# Overall title
# ============================================================================
fig.suptitle('Hierarchical Structure Analysis: Summary of Evidence for Partial Pooling',
            fontsize=16, fontweight='bold', y=0.98)

# Add summary text box
summary_text = (
    'KEY FINDINGS:\n'
    '• Overdispersion: 5.06x (p < 0.001)\n'
    '• ICC: 72.7% (strong between-group variance)\n'
    '• Distribution: Consistent with normal\n'
    '• Recommendation: Hierarchical binomial model'
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.5, 0.01, summary_text, ha='center', fontsize=11,
        bbox=props, fontweight='bold', family='monospace')

plt.savefig(VIZ_DIR / "hierarchical_summary.png", dpi=150, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'hierarchical_summary.png'}")
plt.close()

print("\n" + "="*80)
print("SUMMARY VISUALIZATION COMPLETE")
print("="*80)
print("\nThis comprehensive figure includes:")
print("  (A) Caterpillar plot with CIs")
print("  (B) Variance decomposition showing 5x overdispersion")
print("  (C) ICC pie chart (72.7% between-group)")
print("  (D) Shrinkage factors by sample size")
print("  (E) Q-Q plot supporting normality assumption")
print("  (F) Three pooling strategies compared")
print("\nAll evidence points to hierarchical modeling as the appropriate approach.")
