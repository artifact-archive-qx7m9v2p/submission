"""
Proportion Distribution Analysis and Outlier Detection
Understanding the distribution of observed proportions and identifying outliers
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

# Create comprehensive proportion visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Proportions by group with confidence intervals
ax1 = axes[0, 0]
# Calculate Wilson score confidence intervals (better for proportions near 0 or 1)
z = 1.96  # 95% CI
ci_lower = []
ci_upper = []

for idx, row in data.iterrows():
    n, r = row['n'], row['r']
    p = r / n

    if p == 0:
        # For zero proportions, use rule of 3
        ci_lower.append(0)
        ci_upper.append(3/n)
    elif p == 1:
        ci_lower.append(1 - 3/n)
        ci_upper.append(1)
    else:
        # Wilson score interval
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator

        ci_lower.append(max(0, centre - margin))
        ci_upper.append(min(1, centre + margin))

data['ci_lower'] = ci_lower
data['ci_upper'] = ci_upper

# Calculate error bars (must be positive)
yerr_lower = data['proportion'] - data['ci_lower']
yerr_upper = data['ci_upper'] - data['proportion']

ax1.errorbar(data['group'], data['proportion'],
             yerr=[yerr_lower, yerr_upper],
             fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2,
             color='steelblue', ecolor='gray', alpha=0.7)
ax1.axhline(y=p_overall, color='red', linestyle='--', linewidth=2,
            label=f'Overall = {p_overall:.4f}')
ax1.set_xlabel('Group', fontsize=11, fontweight='bold')
ax1.set_ylabel('Proportion', fontsize=11, fontweight='bold')
ax1.set_title('Proportions by Group (95% Wilson CI)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Distribution of proportions
ax2 = axes[0, 1]
ax2.hist(data['proportion'], bins=10, color='steelblue', alpha=0.7, edgecolor='black', density=True)
ax2.axvline(x=data['proportion'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean = {data["proportion"].mean():.4f}')
ax2.axvline(x=data['proportion'].median(), color='orange', linestyle='--', linewidth=2,
            label=f'Median = {data["proportion"].median():.4f}')
ax2.set_xlabel('Proportion', fontsize=11, fontweight='bold')
ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Observed Proportions', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Boxplot with individual points
ax3 = axes[0, 2]
bp = ax3.boxplot(data['proportion'], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
ax3.scatter([1]*len(data), data['proportion'], s=100, alpha=0.6,
            color='steelblue', edgecolors='black', linewidth=1, zorder=3)
# Label outlier points
for idx, row in data.iterrows():
    if row['proportion'] == data['proportion'].max() or row['proportion'] == data['proportion'].min():
        ax3.annotate(f"G{row['group']}", (1, row['proportion']),
                    xytext=(10, 0), textcoords='offset points', fontsize=9,
                    fontweight='bold')
ax3.set_ylabel('Proportion', fontsize=11, fontweight='bold')
ax3.set_title('Proportion Distribution (Boxplot)', fontsize=13, fontweight='bold')
ax3.set_xticks([])
ax3.grid(axis='y', alpha=0.3)

# 4. Q-Q plot to check normality of proportions
ax4 = axes[1, 0]
stats.probplot(data['proportion'], dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot: Proportion Distribution', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

# 5. Proportion vs Sample Size (weighted by sample size)
ax5 = axes[1, 1]
scatter = ax5.scatter(data['n'], data['proportion'], s=data['n']*2,
                     alpha=0.6, c=data['group'], cmap='tab10',
                     edgecolors='black', linewidth=1)
for idx, row in data.iterrows():
    ax5.annotate(str(row['group']), (row['n'], row['proportion']),
                ha='center', va='center', fontsize=9, fontweight='bold')
ax5.axhline(y=p_overall, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Overall = {p_overall:.4f}')
ax5.set_xlabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Proportion', fontsize=11, fontweight='bold')
ax5.set_title('Proportion vs Sample Size (bubble size = n)', fontsize=13, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Deviation from overall proportion
data['deviation'] = data['proportion'] - p_overall
data['abs_deviation'] = np.abs(data['deviation'])

ax6 = axes[1, 2]
colors = ['red' if x < 0 else 'green' for x in data['deviation']]
bars = ax6.barh(data['group'], data['deviation'], color=colors, alpha=0.7, edgecolor='black')
ax6.axvline(x=0, color='black', linewidth=2)
ax6.set_xlabel('Deviation from Overall Proportion', fontsize=11, fontweight='bold')
ax6.set_ylabel('Group', fontsize=11, fontweight='bold')
ax6.set_title('Group Deviations from Overall Rate', fontsize=13, fontweight='bold')
ax6.grid(axis='x', alpha=0.3)
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/02_proportion_distribution.png')
plt.close()

# Print key statistics
print("Proportion Analysis Complete")
print(f"\nOverall proportion: {p_overall:.4f}")
print(f"Mean of group proportions: {data['proportion'].mean():.4f}")
print(f"Median of group proportions: {data['proportion'].median():.4f}")
print(f"SD of group proportions: {data['proportion'].std():.4f}")
print(f"Range: {data['proportion'].min():.4f} to {data['proportion'].max():.4f}")
print(f"IQR: {data['proportion'].quantile(0.75) - data['proportion'].quantile(0.25):.4f}")

# Identify outliers using IQR method
Q1 = data['proportion'].quantile(0.25)
Q3 = data['proportion'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['proportion'] < lower_bound) | (data['proportion'] > upper_bound)]
print(f"\nOutliers by IQR method (1.5*IQR):")
if len(outliers) > 0:
    print(outliers[['group', 'n', 'r', 'proportion']])
else:
    print("  None detected")

# Calculate coefficient of variation
cv = data['proportion'].std() / data['proportion'].mean()
print(f"\nCoefficient of Variation: {cv:.2f}")
print("  (high CV suggests substantial heterogeneity)")
