"""
Sample Size Distribution Analysis
Understanding the distribution of sample sizes and implications for inference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Create comprehensive sample size visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution of sample sizes
ax1 = axes[0, 0]
ax1.bar(data['group'], data['n'], color='steelblue', alpha=0.7, edgecolor='black')
ax1.axhline(y=data['n'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {data["n"].mean():.0f}')
ax1.axhline(y=data['n'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {data["n"].median():.0f}')
ax1.set_xlabel('Group', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax1.set_title('Sample Sizes by Group', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Histogram of sample sizes
ax2 = axes[0, 1]
ax2.hist(data['n'], bins=8, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(x=data['n'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.axvline(x=data['n'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
ax2.set_xlabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Sample Sizes', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Cumulative proportion of total sample size
sorted_data = data.sort_values('n', ascending=False).copy()
sorted_data['cumulative_n'] = sorted_data['n'].cumsum()
sorted_data['cumulative_pct'] = sorted_data['cumulative_n'] / sorted_data['n'].sum() * 100

ax3 = axes[1, 0]
ax3.plot(range(1, len(sorted_data) + 1), sorted_data['cumulative_pct'].values,
         marker='o', linewidth=2, markersize=8, color='steelblue')
ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
ax3.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80%')
ax3.set_xlabel('Number of Groups (sorted by size)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cumulative % of Total Sample', fontsize=11, fontweight='bold')
ax3.set_title('Cumulative Sample Size Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Add annotations for key milestones
groups_for_50 = (sorted_data['cumulative_pct'] <= 50).sum() + 1
groups_for_80 = (sorted_data['cumulative_pct'] <= 80).sum() + 1
ax3.text(groups_for_50, 50, f'  {groups_for_50} groups', va='bottom', fontsize=9)
ax3.text(groups_for_80, 80, f'  {groups_for_80} groups', va='bottom', fontsize=9)

# 4. Standard error by group (SE = sqrt(p*(1-p)/n))
# Use overall proportion as estimate
p_overall = data['r'].sum() / data['n'].sum()
data['se'] = np.sqrt(p_overall * (1 - p_overall) / data['n'])

ax4 = axes[1, 1]
scatter = ax4.scatter(data['n'], data['se'], s=100, c=data['proportion'],
                      cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
for idx, row in data.iterrows():
    ax4.annotate(str(row['group']), (row['n'], row['se']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax4.set_xlabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Standard Error', fontsize=11, fontweight='bold')
ax4.set_title('Standard Error vs Sample Size', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Observed Proportion', fontsize=10)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/01_sample_size_distribution.png')
plt.close()

print("Sample Size Analysis Complete")
print(f"Total sample size: {data['n'].sum()}")
print(f"Mean sample size: {data['n'].mean():.1f}")
print(f"Median sample size: {data['n'].median():.1f}")
print(f"SD of sample sizes: {data['n'].std():.1f}")
print(f"CV (SD/Mean): {data['n'].std()/data['n'].mean():.2f}")
print(f"\nLargest group contributes {sorted_data.iloc[0]['n']/data['n'].sum()*100:.1f}% of total sample")
print(f"Top 3 groups contribute {sorted_data.iloc[:3]['n'].sum()/data['n'].sum()*100:.1f}% of total sample")
print(f"\nGroups for 50% of sample: {groups_for_50}")
print(f"Groups for 80% of sample: {groups_for_80}")
