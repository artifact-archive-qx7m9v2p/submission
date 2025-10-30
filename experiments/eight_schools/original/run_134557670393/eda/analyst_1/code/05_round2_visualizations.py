"""
Round 2 Visualizations: Sensitivity and Grouping
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/eda/analyst_1/code/processed_data_with_metrics.csv')
loo_data = pd.read_csv('/workspace/eda/analyst_1/code/leave_one_out_results.csv')

# ============================================================================
# VISUALIZATION 7: Leave-One-Out Sensitivity Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Pooled effect by study removed
ax1 = axes[0, 0]
original_effect = 7.686
changes = loo_data['pooled_effect'] - original_effect
colors = ['red' if abs(c) > 1.5 else 'steelblue' for c in changes]

bars = ax1.barh(range(len(loo_data)), loo_data['pooled_effect'], color=colors, alpha=0.7)
ax1.axvline(original_effect, color='darkgreen', linestyle='--', linewidth=2,
            label=f'Original: {original_effect:.2f}')
ax1.set_yticks(range(len(loo_data)))
ax1.set_yticklabels([f"Remove S{int(s)}" for s in loo_data['study_removed']])
ax1.set_xlabel('Pooled Effect Estimate', fontsize=11, fontweight='bold')
ax1.set_title('A. Sensitivity: Pooled Effect (Leave-One-Out)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Panel B: I² by study removed
ax2 = axes[0, 1]
ax2.barh(range(len(loo_data)), loo_data['I_squared'], color='seagreen', alpha=0.7)
ax2.axvline(0, color='darkred', linestyle='--', linewidth=2, label='Original I²: 0%')
ax2.set_yticks(range(len(loo_data)))
ax2.set_yticklabels([f"Remove S{int(s)}" for s in loo_data['study_removed']])
ax2.set_xlabel('I² Statistic (%)', fontsize=11, fontweight='bold')
ax2.set_title('B. Sensitivity: Heterogeneity (Leave-One-Out)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Panel C: Change in pooled effect
ax3 = axes[1, 0]
ax3.barh(range(len(changes)), changes, color=colors, alpha=0.7)
ax3.axvline(0, color='black', linewidth=1)
ax3.set_yticks(range(len(changes)))
ax3.set_yticklabels([f"Remove S{int(s)}" for s in loo_data['study_removed']])
ax3.set_xlabel('Change in Pooled Effect', fontsize=11, fontweight='bold')
ax3.set_title('C. Influence: Change from Original Estimate', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Add annotation for most influential
most_influential_idx = np.argmax(np.abs(changes))
ax3.text(0.05, 0.95, f"Most influential:\nStudy {int(loo_data.iloc[most_influential_idx]['study_removed'])}",
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Panel D: Q-statistic p-values
ax4 = axes[1, 1]
colors_p = ['red' if p < 0.05 else 'steelblue' for p in loo_data['Q_pvalue']]
ax4.barh(range(len(loo_data)), loo_data['Q_pvalue'], color=colors_p, alpha=0.7)
ax4.axvline(0.05, color='darkred', linestyle='--', linewidth=2, label='alpha=0.05')
ax4.set_yticks(range(len(loo_data)))
ax4.set_yticklabels([f"Remove S{int(s)}" for s in loo_data['study_removed']])
ax4.set_xlabel('Q-test P-value', fontsize=11, fontweight='bold')
ax4.set_title('D. Heterogeneity: Q-test P-values', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.suptitle('Leave-One-Out Sensitivity Analysis', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/leave_one_out_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: leave_one_out_analysis.png")

# ============================================================================
# VISUALIZATION 8: SE Scaling Simulation
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Simulate different SE scalings
scales = np.linspace(0.1, 2.0, 50)
I_squared_values = []
Q_values = []

for scale in scales:
    data_temp = data.copy()
    data_temp['sigma_scaled'] = data_temp['sigma'] * scale
    weights_temp = 1 / (data_temp['sigma_scaled'] ** 2)
    pooled_temp = sum(weights_temp * data_temp['y']) / sum(weights_temp)
    Q_temp = sum(weights_temp * (data_temp['y'] - pooled_temp) ** 2)
    df_temp = len(data_temp) - 1
    I_squared_temp = max(0, 100 * (Q_temp - df_temp) / Q_temp)

    I_squared_values.append(I_squared_temp)
    Q_values.append(Q_temp)

# Panel A: I² vs SE scaling
ax1 = axes[0]
ax1.plot(scales, I_squared_values, linewidth=2.5, color='darkblue')
ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Original SEs')
ax1.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax1.axhline(50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate het. (50%)')
ax1.axhline(75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High het. (75%)')
ax1.fill_between(scales, 0, I_squared_values, alpha=0.2, color='steelblue')
ax1.set_xlabel('SE Scaling Factor', fontsize=12, fontweight='bold')
ax1.set_ylabel('I² Statistic (%)', fontsize=12, fontweight='bold')
ax1.set_title('Heterogeneity Paradox: I² Sensitivity to Precision', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# Add annotation
ax1.annotate('Current I² = 0%', xy=(1.0, I_squared_values[np.argmin(np.abs(scales - 1.0))]),
             xytext=(1.3, 20), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Q statistic vs SE scaling
ax2 = axes[1]
ax2.plot(scales, Q_values, linewidth=2.5, color='darkgreen')
ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Original SEs')
ax2.axhline(14.067, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Critical value (df=7)')
ax2.fill_between(scales, 0, Q_values, alpha=0.2, color='seagreen')
ax2.set_xlabel('SE Scaling Factor', fontsize=12, fontweight='bold')
ax2.set_ylabel('Q Statistic', fontsize=12, fontweight='bold')
ax2.set_title('Q Statistic Sensitivity to Precision', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/heterogeneity_paradox.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: heterogeneity_paradox.png")

# ============================================================================
# VISUALIZATION 9: Study Grouping Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Classify studies into groups based on median
threshold = data['y'].median()
data['group'] = ['High' if y > threshold else 'Low' for y in data['y']]
high_group = data[data['group'] == 'High']
low_group = data[data['group'] == 'Low']

# Panel A: Effects by group with error bars
ax1 = axes[0, 0]
for idx, row in high_group.iterrows():
    ax1.errorbar(row['y'], 0, xerr=[[row['y'] - row['ci_lower']]],
                 fmt='o', color='red', markersize=10, capsize=6, capthick=2,
                 linewidth=2.5, alpha=0.7, label='High' if idx == high_group.index[0] else '')
    ax1.text(row['y'], 0.05, f"S{int(row['study'])}", ha='center', fontsize=9)

for idx, row in low_group.iterrows():
    ax1.errorbar(row['y'], 0, xerr=[[row['y'] - row['ci_lower']]],
                 fmt='s', color='blue', markersize=10, capsize=6, capthick=2,
                 linewidth=2.5, alpha=0.7, label='Low' if idx == low_group.index[0] else '')
    ax1.text(row['y'], -0.05, f"S{int(row['study'])}", ha='center', fontsize=9)

ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Median: {threshold}')
ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax1.set_xlabel('Effect Size (y)', fontsize=11, fontweight='bold')
ax1.set_title('A. Study Grouping by Effect Size', fontsize=12, fontweight='bold')
ax1.set_ylim(-0.2, 0.2)
ax1.set_yticks([])
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# Panel B: Distribution by group
ax2 = axes[0, 1]
bp = ax2.boxplot([low_group['y'], high_group['y']], labels=['Low Effect', 'High Effect'],
                  patch_artist=True, notch=True, showmeans=True, widths=0.6)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('red')
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.5)

# Add individual points
for x, group_data in enumerate([low_group, high_group], 1):
    y_vals = group_data['y']
    x_vals = np.random.normal(x, 0.04, size=len(y_vals))
    ax2.scatter(x_vals, y_vals, alpha=0.6, s=50, color='black', zorder=3)

ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax2.set_ylabel('Effect Size (y)', fontsize=11, fontweight='bold')
ax2.set_title('B. Distribution Comparison', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Panel C: Pooled estimates by group
ax3 = axes[1, 0]
group_results = []
for group_name in ['Low', 'High']:
    group_data = data[data['group'] == group_name]
    weights = 1 / (group_data['sigma'] ** 2)
    pooled = sum(weights * group_data['y']) / sum(weights)
    pooled_se = np.sqrt(1 / sum(weights))
    group_results.append({
        'group': group_name,
        'pooled': pooled,
        'lower': pooled - 1.96 * pooled_se,
        'upper': pooled + 1.96 * pooled_se
    })

colors_map = {'Low': 'blue', 'High': 'red'}
for i, result in enumerate(group_results):
    color = colors_map[result['group']]
    ax3.errorbar(result['pooled'], i, xerr=[[result['pooled'] - result['lower']],
                                              [result['upper'] - result['pooled']]],
                 fmt='D', color=color, markersize=12, capsize=8, capthick=3,
                 linewidth=3, alpha=0.7, label=f"{result['group']}: {result['pooled']:.2f}")

ax3.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Low Effect', 'High Effect'])
ax3.set_xlabel('Pooled Effect Estimate', fontsize=11, fontweight='bold')
ax3.set_title('C. Pooled Estimates by Group', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(axis='x', alpha=0.3)

# Panel D: Precision by group
ax4 = axes[1, 1]
ax4.scatter(low_group['sigma'], low_group['y'], s=200, c='blue', alpha=0.6,
            edgecolors='black', linewidth=1.5, label='Low Effect', marker='s')
ax4.scatter(high_group['sigma'], high_group['y'], s=200, c='red', alpha=0.6,
            edgecolors='black', linewidth=1.5, label='High Effect', marker='o')

for idx, row in data.iterrows():
    ax4.text(row['sigma'] + 0.3, row['y'], f"{int(row['study'])}", fontsize=9)

ax4.axhline(threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax4.set_xlabel('Standard Error (sigma)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Effect Size (y)', fontsize=11, fontweight='bold')
ax4.set_title('D. Precision vs Effect by Group', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(alpha=0.3)

plt.suptitle('Study Grouping Analysis: High vs Low Effects', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/study_grouping.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: study_grouping.png")

# ============================================================================
# VISUALIZATION 10: Comprehensive Summary Panel
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel A: Key statistics summary (text-based)
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

summary_text = f"""
KEY FINDINGS SUMMARY

Distribution Characteristics:
  Effect sizes (y): Mean={data['y'].mean():.2f}, SD={data['y'].std():.2f}, Range=[{data['y'].min()}, {data['y'].max()}]
  Standard errors: Mean={data['sigma'].mean():.2f}, SD={data['sigma'].std():.2f}, Range=[{data['sigma'].min()}, {data['sigma'].max()}]

Heterogeneity Assessment:
  I² statistic: 0.0% (Low heterogeneity)
  Cochran's Q: 4.71 (p=0.696, not significant)
  Conclusion: NO significant heterogeneity detected

Pooled Estimates:
  Fixed effect: 7.69 (95% CI: [-0.30, 15.67])
  Random effect: 7.69 (identical, tau²=0)

Study Groups:
  High effects (>7.5): Studies 1, 2, 7, 8 | Mean=16.5 | Pooled=15.31 [3.50, 27.12]
  Low effects (≤7.5): Studies 3, 4, 5, 6 | Mean=1.0 | Pooled=1.28 [-9.54, 12.11]
  T-test: t=-3.19, p=0.019 (SIGNIFICANT difference between groups)

Sensitivity:
  Most influential study: Study 5 (removal changes estimate by 2.24)
  Pooled estimate range (LOO): [5.64, 9.92]
  All LOO analyses show I²=0%
"""

ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel B: Forest plot (simplified)
ax2 = fig.add_subplot(gs[1, :])
data_sorted = data.sort_values('y', ascending=True)
y_positions = np.arange(len(data_sorted))
for i, (idx, row) in enumerate(data_sorted.iterrows()):
    color = 'red' if row['group'] == 'High' else 'blue'
    ax2.errorbar(row['y'], i, xerr=[[row['y'] - row['ci_lower']],
                                      [row['ci_upper'] - row['y']]],
                fmt='o', color=color, markersize=7, capsize=4, capthick=1.5,
                linewidth=1.5, alpha=0.7)
    ax2.text(-8, i, f"S{int(row['study'])}", va='center', ha='right', fontsize=8)

ax2.axvline(7.686, color='darkred', linestyle='--', linewidth=2, alpha=0.8)
ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax2.set_yticks([])
ax2.set_xlabel('Effect Size', fontsize=10, fontweight='bold')
ax2.set_title('Forest Plot: Color-Coded by Group', fontsize=11, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Panel C: I² sensitivity
ax3 = fig.add_subplot(gs[2, 0])
scales_small = [1.0, 0.75, 0.5, 0.25]
I_small = [0, 16.3, 62.8, 90.7]
ax3.bar(range(len(scales_small)), I_small, color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
ax3.set_xticks(range(len(scales_small)))
ax3.set_xticklabels([f'{s:.2f}x' for s in scales_small])
ax3.set_xlabel('SE Scaling', fontsize=9, fontweight='bold')
ax3.set_ylabel('I² (%)', fontsize=9, fontweight='bold')
ax3.set_title('I² vs Precision', fontsize=10, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Panel D: LOO influence
ax4 = fig.add_subplot(gs[2, 1])
changes = loo_data['pooled_effect'] - 7.686
colors_inf = ['red' if abs(c) > 1.5 else 'steelblue' for c in changes]
ax4.bar(range(len(changes)), np.abs(changes), color=colors_inf, alpha=0.7)
ax4.set_xticks(range(len(changes)))
ax4.set_xticklabels([f'S{int(s)}' for s in loo_data['study_removed']], fontsize=8)
ax4.set_xlabel('Study Removed', fontsize=9, fontweight='bold')
ax4.set_ylabel('|Change|', fontsize=9, fontweight='bold')
ax4.set_title('LOO Influence', fontsize=10, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Panel E: Direction of effects
ax5 = fig.add_subplot(gs[2, 2])
positive_count = len(data[data['y'] > 0])
negative_count = len(data[data['y'] < 0])
ax5.pie([positive_count, negative_count], labels=['Positive', 'Negative'],
        colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
ax5.set_title('Effect Direction', fontsize=10, fontweight='bold')

plt.suptitle('COMPREHENSIVE EDA SUMMARY: Distributions & Heterogeneity',
             fontsize=15, fontweight='bold', y=0.98)
plt.savefig('/workspace/eda/analyst_1/visualizations/comprehensive_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created: comprehensive_summary.png")

print("\n" + "="*70)
print("All Round 2 visualizations created successfully!")
print("="*70)
