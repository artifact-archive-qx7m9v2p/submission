"""
Individual Group Characterization
Detailed profiling of each group to identify extreme cases
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

# Calculate additional metrics
data['se'] = np.sqrt(p_overall * (1 - p_overall) / data['n'])
data['z_score'] = (data['proportion'] - p_overall) / data['se']
data['expected_r'] = data['n'] * p_overall
data['deviation_abs'] = np.abs(data['proportion'] - p_overall)

# Categorize groups
def categorize_group(row):
    if row['z_score'] > 2:
        return 'High outlier'
    elif row['z_score'] < -2:
        return 'Low outlier'
    elif abs(row['z_score']) < 1:
        return 'Typical'
    else:
        return 'Moderate deviation'

data['category'] = data.apply(categorize_group, axis=1)

# Categorize sample size
data['size_category'] = pd.cut(data['n'], bins=[0, 100, 200, 1000],
                                labels=['Small (<100)', 'Medium (100-200)', 'Large (>200)'])

print("="*80)
print("GROUP CHARACTERIZATION")
print("="*80)

print("\n1. GROUP CATEGORIES")
print("-"*40)
print(data['category'].value_counts())
print(f"\nSample size categories:")
print(data['size_category'].value_counts())

print("\n2. EXTREME GROUPS")
print("-"*40)
print("\nHighest proportions:")
top3 = data.nlargest(3, 'proportion')[['group', 'n', 'r', 'proportion', 'z_score', 'category']]
print(top3.to_string(index=False))

print("\nLowest proportions:")
bottom3 = data.nsmallest(3, 'proportion')[['group', 'n', 'r', 'proportion', 'z_score', 'category']]
print(bottom3.to_string(index=False))

print("\nLargest absolute deviations from overall rate:")
largest_dev = data.nlargest(3, 'deviation_abs')[['group', 'n', 'r', 'proportion', 'z_score', 'category']]
print(largest_dev.to_string(index=False))

print("\n3. STATISTICAL OUTLIERS (|z-score| > 2)")
print("-"*40)
outliers = data[np.abs(data['z_score']) > 2].sort_values('z_score', ascending=False)
if len(outliers) > 0:
    print(outliers[['group', 'n', 'r', 'proportion', 'z_score', 'category']].to_string(index=False))
else:
    print("No statistical outliers detected")

print("\n4. GROUP SUMMARY TABLE")
print("-"*40)
summary = data[['group', 'n', 'r', 'proportion', 'se', 'z_score', 'category']].sort_values('z_score', ascending=False)
print(summary.to_string(index=False))

# Create comprehensive group profile visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Group profiles heatmap
ax1 = fig.add_subplot(gs[0, :])
# Normalize metrics for heatmap
norm_data = data[['group', 'n', 'r', 'proportion', 'z_score']].copy()
norm_data['n_norm'] = (norm_data['n'] - norm_data['n'].min()) / (norm_data['n'].max() - norm_data['n'].min())
norm_data['r_norm'] = (norm_data['r'] - norm_data['r'].min()) / (norm_data['r'].max() - norm_data['r'].min())
norm_data['prop_norm'] = (norm_data['proportion'] - norm_data['proportion'].min()) / (norm_data['proportion'].max() - norm_data['proportion'].min())
norm_data['z_norm'] = (norm_data['z_score'] - norm_data['z_score'].min()) / (norm_data['z_score'].max() - norm_data['z_score'].min())

heatmap_data = norm_data[['n_norm', 'r_norm', 'prop_norm', 'z_norm']].T
sns.heatmap(heatmap_data, cmap='RdYlGn', center=0.5, annot=False, cbar_kws={'label': 'Normalized Value'},
            xticklabels=data['group'], yticklabels=['Sample Size', 'Events', 'Proportion', 'Z-score'],
            ax=ax1)
ax1.set_title('Group Profile Heatmap (normalized metrics)', fontsize=13, fontweight='bold')

# 2. Z-scores by group
ax2 = fig.add_subplot(gs[1, 0])
colors = ['red' if abs(x) > 2 else 'orange' if abs(x) > 1 else 'steelblue' for x in data['z_score']]
ax2.barh(data['group'], data['z_score'], color=colors, alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='black', linewidth=2)
ax2.axvline(x=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(x=-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Z-score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Group', fontsize=11, fontweight='bold')
ax2.set_title('Standardized Deviations (Z-scores)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# 3. Sample size vs proportion colored by category
ax3 = fig.add_subplot(gs[1, 1])
category_colors = {'High outlier': 'red', 'Low outlier': 'blue',
                   'Moderate deviation': 'orange', 'Typical': 'green'}
for cat, color in category_colors.items():
    mask = data['category'] == cat
    if mask.any():
        ax3.scatter(data[mask]['n'], data[mask]['proportion'], s=150,
                   label=cat, color=color, alpha=0.7, edgecolors='black', linewidth=1.5)
        for idx, row in data[mask].iterrows():
            ax3.annotate(str(row['group']), (row['n'], row['proportion']),
                        ha='center', va='center', fontsize=8, fontweight='bold')
ax3.axhline(y=p_overall, color='gray', linestyle='--', linewidth=2)
ax3.set_xlabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Proportion', fontsize=11, fontweight='bold')
ax3.set_title('Groups by Category', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 4. Distribution by category
ax4 = fig.add_subplot(gs[1, 2])
category_counts = data['category'].value_counts()
colors_pie = [category_colors[cat] for cat in category_counts.index]
ax4.pie(category_counts.values, labels=category_counts.index, autopct='%1.0f%%',
        colors=colors_pie, startangle=90)
ax4.set_title('Distribution of Group Categories', fontsize=12, fontweight='bold')

# 5. Proportion by group with size indication
ax5 = fig.add_subplot(gs[2, :])
# Sort by proportion for better visualization
sorted_data = data.sort_values('proportion')
colors_sorted = [category_colors[cat] for cat in sorted_data['category']]
bars = ax5.bar(range(len(sorted_data)), sorted_data['proportion'],
               color=colors_sorted, alpha=0.7, edgecolor='black', linewidth=1.5)

# Vary bar width by sample size
for i, (idx, row) in enumerate(sorted_data.iterrows()):
    bars[i].set_width(0.4 + 0.6 * (row['n'] / data['n'].max()))

ax5.axhline(y=p_overall, color='black', linestyle='--', linewidth=2, label='Overall rate')
ax5.set_xticks(range(len(sorted_data)))
ax5.set_xticklabels(sorted_data['group'], fontsize=10)
ax5.set_xlabel('Group (sorted by proportion)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Proportion', fontsize=11, fontweight='bold')
ax5.set_title('Proportions by Group (bar width represents sample size)', fontsize=13, fontweight='bold')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

plt.savefig('/workspace/eda/analyst_1/visualizations/04_group_characterization.png')
plt.close()

print("\n" + "="*80)
print("GROUP CHARACTERIZATION COMPLETE")
print("="*80)
