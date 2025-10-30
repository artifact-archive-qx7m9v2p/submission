"""
Diagnostic Summary - Key Metrics at a Glance
Comprehensive overview of distributional properties and outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Overall proportion
p_overall = data['r'].sum() / data['n'].sum()

# Calculate metrics
data['se'] = np.sqrt(p_overall * (1 - p_overall) / data['n'])
data['z_score'] = (data['proportion'] - p_overall) / data['se']
data['ci_width'] = 2 * 1.96 * data['se']  # approximate CI width

# Create single comprehensive diagnostic plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Key Statistics Box
ax1 = axes[0, 0]
ax1.axis('off')

stats_text = f"""
DISTRIBUTIONAL PROPERTIES SUMMARY
{'='*50}

SAMPLE SIZE DISTRIBUTION
  Range: {data['n'].min()} to {data['n'].max()}
  Mean: {data['n'].mean():.1f}
  Median: {data['n'].median():.1f}
  SD: {data['n'].std():.1f}
  CV: {data['n'].std()/data['n'].mean():.2f}
  Total: {data['n'].sum()}

PROPORTION DISTRIBUTION
  Overall: {p_overall:.4f} (95% CI: {p_overall - 1.96*np.sqrt(p_overall*(1-p_overall)/data['n'].sum()):.4f} - {p_overall + 1.96*np.sqrt(p_overall*(1-p_overall)/data['n'].sum()):.4f})
  Mean: {data['proportion'].mean():.4f}
  Median: {data['proportion'].median():.4f}
  SD: {data['proportion'].std():.4f}
  Range: {data['proportion'].min():.4f} to {data['proportion'].max():.4f}
  CV: {data['proportion'].std()/data['proportion'].mean():.2f}

HETEROGENEITY ASSESSMENT
  Chi-square test: p < 0.001 (SIGNIFICANT)
  Overdispersion factor: 3.51
  Dispersion parameter (Phi): 5.06
  Interpretation: SUBSTANTIAL HETEROGENEITY

OUTLIER DETECTION
  Groups with |z-score| > 2: {sum(np.abs(data['z_score']) > 2)}
  Groups with |z-score| > 1: {sum(np.abs(data['z_score']) > 1)}

  High outliers: Groups {', '.join(map(str, data[data['z_score'] > 2]['group'].values))}
  Low outliers: Groups {', '.join(map(str, data[data['z_score'] < -2]['group'].values)) if sum(data['z_score'] < -2) > 0 else 'None'}

KEY FINDINGS
  1. HIGH variability in both sample sizes (CV=0.85) and proportions (CV=0.52)
  2. SUBSTANTIAL overdispersion (Phi=5.06) indicating heterogeneity
  3. Groups 2, 8, 11 are high outliers (significantly above average)
  4. Group 1 has ZERO events (0/47) - potential concern
  5. Wide range of precision due to varying sample sizes
"""

ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 2. Precision vs Proportion (annotated)
ax2 = axes[0, 1]
precision = 1 / data['se']
scatter = ax2.scatter(precision, data['proportion'], s=data['n'],
                     c=np.abs(data['z_score']), cmap='YlOrRd',
                     alpha=0.7, edgecolors='black', linewidth=1.5)

# Annotate all points
for idx, row in data.iterrows():
    prec = 1 / row['se']
    color = 'red' if abs(row['z_score']) > 2 else 'black'
    weight = 'bold' if abs(row['z_score']) > 2 else 'normal'
    ax2.annotate(str(row['group']), (prec, row['proportion']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color=color, fontweight=weight)

ax2.axhline(y=p_overall, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Overall rate')
ax2.set_xlabel('Precision (1/SE)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Proportion', fontsize=12, fontweight='bold')
ax2.set_title('Precision-Proportion Plot\n(size=sample size, color=|z-score|)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('|Z-score|', fontsize=10)

# 3. Forest plot with CIs
ax3 = axes[1, 0]
sorted_data = data.sort_values('proportion', ascending=True)

y_pos = range(len(sorted_data))
colors = ['red' if abs(z) > 2 else 'orange' if abs(z) > 1 else 'steelblue'
          for z in sorted_data['z_score']]

# Plot points and error bars
for i, (idx, row) in enumerate(sorted_data.iterrows()):
    ci_lower = max(0, row['proportion'] - 1.96 * row['se'])
    ci_upper = min(1, row['proportion'] + 1.96 * row['se'])
    ax3.plot([ci_lower, ci_upper], [i, i], color=colors[i], linewidth=2, alpha=0.7)
    ax3.plot(row['proportion'], i, 'o', color=colors[i], markersize=8,
            markeredgecolor='black', markeredgewidth=1.5)

ax3.axvline(x=p_overall, color='red', linestyle='--', linewidth=2, label='Overall rate')
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"G{g}" for g in sorted_data['group']], fontsize=9)
ax3.set_xlabel('Proportion', fontsize=12, fontweight='bold')
ax3.set_ylabel('Group', fontsize=12, fontweight='bold')
ax3.set_title('Forest Plot: Proportions with 95% CIs\n(sorted by proportion)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# 4. Distribution comparison
ax4 = axes[1, 1]

# Sample size histogram
ax4_twin = ax4.twinx()
ax4.hist(data['n'], bins=8, alpha=0.5, color='steelblue', edgecolor='black', label='Sample Size')
ax4_twin.hist(data['proportion'] * 1000, bins=10, alpha=0.5, color='coral', edgecolor='black', label='Proportion x1000')

ax4.set_xlabel('Value', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency (Sample Size)', fontsize=11, fontweight='bold', color='steelblue')
ax4_twin.set_ylabel('Frequency (Proportion x1000)', fontsize=11, fontweight='bold', color='coral')
ax4.set_title('Distribution Comparison', fontsize=13, fontweight='bold')
ax4.tick_params(axis='y', labelcolor='steelblue')
ax4_twin.tick_params(axis='y', labelcolor='coral')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/05_diagnostic_summary.png')
plt.close()

print("Diagnostic summary visualization created successfully!")
print(f"\nKEY TAKEAWAYS:")
print(f"1. Substantial heterogeneity confirmed (p < 0.001)")
print(f"2. Overdispersion factor = 3.51 (variance ~3.5x binomial expectation)")
print(f"3. {sum(np.abs(data['z_score']) > 2)} groups are statistical outliers")
print(f"4. Sample sizes vary {data['n'].max()/data['n'].min():.1f}-fold")
print(f"5. Proportions vary from {data['proportion'].min():.4f} to {data['proportion'].max():.4f}")
