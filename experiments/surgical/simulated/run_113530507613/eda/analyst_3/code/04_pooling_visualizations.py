"""
Pooling Analysis Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')
output_dir = Path('/workspace/eda/analyst_3/visualizations')

print("Creating pooling analysis visualizations...")

# Calculate pooled rate
pooled_rate = data['r_successes'].sum() / data['n_trials'].sum()

# Calculate shrinkage estimates
weights = data['n_trials']
weighted_mean = np.average(data['success_rate'], weights=weights)
weighted_var = np.average((data['success_rate'] - weighted_mean)**2, weights=weights)
within_var = np.mean([r['success_rate'] * (1 - r['success_rate']) / r['n_trials']
                      for _, r in data.iterrows()])
between_var = max(0, weighted_var - within_var)

shrunk_rates = []
for idx, row in data.iterrows():
    n = row['n_trials']
    lambda_shrink = between_var / (between_var + pooled_rate * (1 - pooled_rate) / n)
    shrunk_rate = lambda_shrink * row['success_rate'] + (1 - lambda_shrink) * pooled_rate
    shrunk_rates.append(shrunk_rate)

data['shrunk_rate'] = shrunk_rates

# Figure 1: Pooled vs No Pooling comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Complete Pooling vs No Pooling Analysis', fontsize=14, fontweight='bold')

# Sort data by group_id for consistent ordering
data_sorted = data.sort_values('group_id')

# 1. Bar chart comparing rates
x_pos = np.arange(len(data_sorted))
axes[0, 0].bar(x_pos - 0.15, [pooled_rate] * len(data_sorted), 0.3,
               label='Complete Pooling', alpha=0.7, color='red')
axes[0, 0].bar(x_pos + 0.15, data_sorted['success_rate'], 0.3,
               label='No Pooling (MLE)', alpha=0.7, color='blue')
axes[0, 0].set_xlabel('Group')
axes[0, 0].set_ylabel('Success Rate')
axes[0, 0].set_title('Pooled vs Individual Estimates')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(data_sorted['group_id'])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Deviation from pooled rate
deviations = data_sorted['success_rate'] - pooled_rate
colors = ['green' if d < 0 else 'orange' for d in deviations]
axes[0, 1].bar(x_pos, deviations, color=colors, alpha=0.7)
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Group')
axes[0, 1].set_ylabel('Deviation from Pooled Rate')
axes[0, 1].set_title(f'How Groups Differ from Pooled Rate ({pooled_rate:.4f})')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(data_sorted['group_id'])
axes[0, 1].grid(True, alpha=0.3)

# 3. Shrinkage visualization
axes[1, 0].scatter(data_sorted['n_trials'], data_sorted['success_rate'],
                   s=100, alpha=0.6, label='No Pooling (MLE)', color='blue')
axes[1, 0].scatter(data_sorted['n_trials'], data_sorted['shrunk_rate'],
                   s=100, alpha=0.6, label='Partial Pooling (Shrunk)', color='purple', marker='s')
axes[1, 0].axhline(pooled_rate, color='red', linestyle='--', linewidth=2,
                   label=f'Complete Pooling ({pooled_rate:.4f})')

# Add arrows showing shrinkage
for idx, row in data_sorted.iterrows():
    axes[1, 0].arrow(row['n_trials'], row['success_rate'],
                     0, row['shrunk_rate'] - row['success_rate'],
                     head_width=15, head_length=0.003, fc='gray', ec='gray',
                     alpha=0.4, length_includes_head=True)

axes[1, 0].set_xlabel('Sample Size (n_trials)')
axes[1, 0].set_ylabel('Success Rate')
axes[1, 0].set_title('Shrinkage Effect: Raw Estimates Pulled Toward Pooled Mean')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Variance components
variance_data = {
    'Between-Group\n(tau^2)': between_var,
    'Within-Group\n(avg)': within_var,
    'Total': weighted_var
}
axes[1, 1].bar(variance_data.keys(), variance_data.values(), color=['red', 'blue', 'purple'], alpha=0.7)
axes[1, 1].set_ylabel('Variance')
axes[1, 1].set_title('Variance Decomposition\n(Between-group vs Within-group)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (k, v) in enumerate(variance_data.items()):
    axes[1, 1].text(i, v, f'{v:.6f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'pooling_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pooling_comparison.png")

# Figure 2: Heterogeneity test visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Evidence for Group Heterogeneity', fontsize=14, fontweight='bold')

# Expected vs observed successes
expected_successes = data['n_trials'] * pooled_rate
residuals = data['r_successes'] - expected_successes

axes[0].scatter(expected_successes, data['r_successes'], s=100, alpha=0.6, edgecolors='black')
axes[0].plot([expected_successes.min(), expected_successes.max()],
             [expected_successes.min(), expected_successes.max()],
             'r--', linewidth=2, label='Perfect fit (homogeneous)')
axes[0].set_xlabel('Expected Successes (under pooled model)')
axes[0].set_ylabel('Observed Successes')
axes[0].set_title('Observed vs Expected Successes\n(Chi-square = 39.5, p < 0.001)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Add group labels
for idx, row in data.iterrows():
    axes[0].annotate(f"{int(row['group_id'])}",
                     (expected_successes.iloc[idx], row['r_successes']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

# Standardized residuals
std_residuals = residuals / np.sqrt(expected_successes * (1 - pooled_rate))
axes[1].bar(np.arange(len(data)), std_residuals, alpha=0.7, edgecolor='black')
axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1].axhline(2, color='red', linestyle='--', linewidth=1, label='±2 SD threshold')
axes[1].axhline(-2, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Group')
axes[1].set_ylabel('Standardized Residual')
axes[1].set_title('Standardized Residuals from Pooled Model\n(Values beyond ±2 suggest heterogeneity)')
axes[1].set_xticks(np.arange(len(data)))
axes[1].set_xticklabels(data['group_id'])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'heterogeneity_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: heterogeneity_test.png")

# Figure 3: Sample size and precision
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate standard errors
se_unpooled = [np.sqrt(row['success_rate'] * (1 - row['success_rate']) / row['n_trials'])
               for _, row in data.iterrows()]

# Size of points proportional to SE (larger = less precise)
sizes = np.array(se_unpooled) * 5000

scatter = ax.scatter(data['n_trials'], data['success_rate'],
                     s=sizes, alpha=0.4, c=data['r_successes'],
                     cmap='viridis', edgecolors='black', linewidth=1)
ax.scatter(data['n_trials'], data['shrunk_rate'],
          s=50, alpha=0.8, color='red', marker='x', linewidth=2,
          label='Shrinkage estimate')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Successes', rotation=270, labelpad=20)

ax.axhline(pooled_rate, color='blue', linestyle='--', linewidth=2,
          label=f'Pooled rate ({pooled_rate:.4f})')
ax.set_xlabel('Sample Size (n_trials)')
ax.set_ylabel('Success Rate')
ax.set_title('Sample Size, Precision, and Shrinkage\n(Circle size = standard error; larger = less precise)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'sample_size_precision_shrinkage.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sample_size_precision_shrinkage.png")

print("\nPooling visualizations complete!")
