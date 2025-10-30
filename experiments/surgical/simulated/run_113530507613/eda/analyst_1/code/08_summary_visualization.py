"""
Summary Visualization - Key Findings at a Glance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Calculate pooled rate and metrics
pooled_p = data['r_successes'].sum() / data['n_trials'].sum()
data['expected_se'] = np.sqrt(pooled_p * (1 - pooled_p) / data['n_trials'])
data['z_score'] = (data['success_rate'] - pooled_p) / data['expected_se']

# Create comprehensive summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('EDA Summary: Strong Evidence for Heterogeneity in Success Rates',
             fontsize=16, fontweight='bold', y=0.98)

# 1. Distribution of success rates (large)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(data['success_rate'], bins=8, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(pooled_p, color='red', linestyle='--', linewidth=2, label=f'Pooled: {pooled_p:.3f}')
ax1.axvline(data['success_rate'].mean(), color='green', linestyle='--', linewidth=2,
            label=f'Mean: {data["success_rate"].mean():.3f}')
ax1.set_xlabel('Success Rate')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Success Rates', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 2. Funnel plot
ax2 = fig.add_subplot(gs[0, 1])
n_range = np.linspace(data['n_trials'].min(), data['n_trials'].max(), 100)
ci_upper = pooled_p + 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)
ci_lower = pooled_p - 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)

ax2.fill_between(n_range, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='95% CI')
ax2.axhline(pooled_p, color='red', linestyle='--', linewidth=2)
ax2.scatter(data['n_trials'], data['success_rate'], s=80, alpha=0.7,
            edgecolors='black', linewidth=1, c=data['success_rate'], cmap='RdYlGn', vmin=0, vmax=0.15)

outliers = data[np.abs(data['z_score']) > 1.96]
for idx, row in outliers.iterrows():
    ax2.annotate(f"{row['group_id']}", (row['n_trials'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax2.set_xlabel('Number of Trials')
ax2.set_ylabel('Success Rate')
ax2.set_title('Funnel Plot: 25% Outside 95% CI', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# 3. Variance decomposition (pie chart)
ax3 = fig.add_subplot(gs[0, 2])
var_empirical = data['success_rate'].var(ddof=1)
var_expected = np.mean(pooled_p * (1 - pooled_p) / data['n_trials'])
var_between = var_empirical - var_expected

sizes = [var_between, var_expected]
labels = ['Between-Group\n(64%)', 'Within-Group\n(36%)']
colors = ['#ff6b6b', '#4ecdc4']
explode = (0.1, 0)

ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax3.set_title('Variance Decomposition\n(Variance Ratio: 2.78)', fontweight='bold')

# 4. Z-scores by group
ax4 = fig.add_subplot(gs[1, :])
data_sorted = data.sort_values('z_score', ascending=False)
x_pos = np.arange(len(data_sorted))
colors = ['red' if abs(z) > 3 else 'orange' if abs(z) > 1.96 else 'green'
          for z in data_sorted['z_score']]

bars = ax4.bar(x_pos, data_sorted['z_score'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axhline(1.96, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='95% CI (±1.96)')
ax4.axhline(-1.96, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax4.axhline(3, color='red', linestyle=':', linewidth=2, alpha=0.7, label='99.7% CI (±3.0)')
ax4.axhline(-3, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax4.axhline(0, color='black', linestyle='-', linewidth=1)

# Add group labels and sample sizes
for i, (idx, row) in enumerate(data_sorted.iterrows()):
    ax4.text(i, row['z_score'] + (0.3 if row['z_score'] > 0 else -0.3),
            f"Grp {row['group_id']}\n(n={row['n_trials']})",
            ha='center', va='bottom' if row['z_score'] > 0 else 'top',
            fontsize=8, fontweight='bold')

ax4.set_xlabel('Group (sorted by z-score)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Z-Score (Standardized Deviation)', fontsize=11, fontweight='bold')
ax4.set_title('Standardized Deviations: 2 Extreme Outliers (Groups 4 & 8)', fontsize=12, fontweight='bold')
ax4.set_xticks([])
ax4.legend(fontsize=10, loc='upper right')
ax4.grid(alpha=0.3, axis='y')
ax4.set_ylim([-4.5, 4.5])

# 5. Model comparison table (as text)
ax5 = fig.add_subplot(gs[2, 0])
ax5.axis('off')

table_data = [
    ['Model', 'AIC', 'BIC', 'Verdict'],
    ['', '', '', ''],
    ['Homogeneous', '90.63', '91.11', 'REJECTED'],
    ['(pooled rate)', '', '', 'p < 0.001'],
    ['', '', '', ''],
    ['Heterogeneous', '76.36', '82.18', 'BEST'],
    ['(group rates)', 'Δ=-14.3', 'Δ=-8.9', 'p < 0.001']
]

table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.35, 0.2, 0.2, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4ecdc4')
    table[(0, i)].set_text_props(weight='bold')

# Style model rows
table[(2, 0)].set_facecolor('#ffcccc')
table[(5, 0)].set_facecolor('#ccffcc')
table[(5, 3)].set_facecolor('#ccffcc')
table[(5, 3)].set_text_props(weight='bold')

ax5.set_title('Model Comparison\n(Δ AIC > 10 = decisive)', fontweight='bold', fontsize=11)

# 6. Key statistics (text box)
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

stats_text = f"""
KEY STATISTICS

Chi-square test:
  χ² = 39.52, p = 0.000043
  → REJECT homogeneity

Variance ratio:
  Observed / Expected = 2.78
  → Substantial overdispersion

Outliers:
  3 groups outside 95% CI (25%)
  2 groups outside 99.7% CI (17%)
  Expected: 5% and 0.3%

Sample size correlation:
  r = -0.34, p = 0.278
  → NOT explained by n
"""

ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax6.set_title('Statistical Evidence', fontweight='bold', fontsize=11)

# 7. Recommendations (text box)
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

rec_text = """
MODELING RECOMMENDATIONS

✓ Use hierarchical models:
  • Beta-binomial
  • Mixed-effects logistic
  • Bayesian hierarchical

✓ Account for overdispersion

✓ Group-specific random effects

✗ DO NOT use:
  • Simple pooled binomial
  • Fixed-effects only

Reason: 64% of variance is
between-group differences
"""

ax7.text(0.1, 0.9, rec_text, transform=ax7.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax7.set_title('Modeling Guidance', fontweight='bold', fontsize=11)

plt.savefig('/workspace/eda/analyst_1/visualizations/00_summary_dashboard.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/analyst_1/visualizations/00_summary_dashboard.png")
print("\nSummary visualization created successfully!")
