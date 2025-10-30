"""
Data Quality and Distribution Visualizations
Focus: Understanding variability and data characteristics for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')
output_dir = Path('/workspace/eda/analyst_3/visualizations')

print("Creating data quality visualizations...")

# Figure 1: Multi-panel overview of distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Data Distribution Overview', fontsize=14, fontweight='bold')

# n_trials histogram
axes[0, 0].hist(data['n_trials'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Number of Trials')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title(f'Distribution of n_trials\nMean={data["n_trials"].mean():.1f}, Median={data["n_trials"].median():.1f}')
axes[0, 0].axvline(data['n_trials'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].axvline(data['n_trials'].median(), color='orange', linestyle='--', label='Median')
axes[0, 0].legend()

# r_successes histogram
axes[0, 1].hist(data['r_successes'], bins=15, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Number of Successes')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Distribution of r_successes\nMean={data["r_successes"].mean():.1f}, Median={data["r_successes"].median():.1f}')
axes[0, 1].axvline(data['r_successes'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 1].axvline(data['r_successes'].median(), color='orange', linestyle='--', label='Median')
axes[0, 1].legend()

# success_rate histogram
axes[0, 2].hist(data['success_rate'], bins=15, edgecolor='black', alpha=0.7, color='purple')
axes[0, 2].set_xlabel('Success Rate')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title(f'Distribution of Success Rates\nMean={data["success_rate"].mean():.4f}, Median={data["success_rate"].median():.4f}')
axes[0, 2].axvline(data['success_rate'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 2].axvline(data['success_rate'].median(), color='orange', linestyle='--', label='Median')
axes[0, 2].legend()

# Box plots
axes[1, 0].boxplot([data['n_trials']], labels=['n_trials'])
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('n_trials Box Plot')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].boxplot([data['r_successes']], labels=['r_successes'])
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('r_successes Box Plot')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].boxplot([data['success_rate']], labels=['success_rate'])
axes[1, 2].set_ylabel('Value')
axes[1, 2].set_title('success_rate Box Plot')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'distributions_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: distributions_overview.png")

# Figure 2: Sample size adequacy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Sample Size Adequacy Analysis', fontsize=14, fontweight='bold')

# Scatter: n_trials vs r_successes
axes[0].scatter(data['n_trials'], data['r_successes'], s=100, alpha=0.6, edgecolors='black')
axes[0].set_xlabel('Number of Trials')
axes[0].set_ylabel('Number of Successes')
axes[0].set_title('Successes vs Trials by Group')
axes[0].grid(True, alpha=0.3)

# Add group labels
for idx, row in data.iterrows():
    axes[0].annotate(f"{row['group_id']}",
                     (row['n_trials'], row['r_successes']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

# Expected successes under pooled rate
pooled_rate = data['r_successes'].sum() / data['n_trials'].sum()
expected = data['n_trials'] * pooled_rate
axes[0].plot(data['n_trials'], expected, 'r--', label=f'Expected (pooled rate={pooled_rate:.4f})', linewidth=2)
axes[0].legend()

# Scatter: n_trials vs success_rate
axes[1].scatter(data['n_trials'], data['success_rate'], s=100, alpha=0.6, edgecolors='black', color='purple')
axes[1].set_xlabel('Number of Trials')
axes[1].set_ylabel('Success Rate')
axes[1].set_title('Success Rate vs Sample Size')
axes[1].axhline(pooled_rate, color='red', linestyle='--', label=f'Pooled rate={pooled_rate:.4f}', linewidth=2)
axes[1].grid(True, alpha=0.3)

# Add group labels
for idx, row in data.iterrows():
    axes[1].annotate(f"{row['group_id']}",
                     (row['n_trials'], row['success_rate']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'sample_size_adequacy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sample_size_adequacy.png")

# Figure 3: Ordered success rates with confidence intervals
fig, ax = plt.subplots(figsize=(12, 6))

# Sort by success rate
data_sorted = data.sort_values('success_rate').reset_index(drop=True)

# Calculate simple binomial confidence intervals (Wilson score)
from scipy import stats

def wilson_ci(r, n, alpha=0.05):
    """Wilson score confidence interval"""
    z = stats.norm.ppf(1 - alpha/2)
    phat = r / n
    denominator = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((phat * (1 - phat) / n + z**2 / (4*n**2))) / denominator
    return center - margin, center + margin

ci_lower = []
ci_upper = []
for idx, row in data_sorted.iterrows():
    lower, upper = wilson_ci(row['r_successes'], row['n_trials'])
    ci_lower.append(lower)
    ci_upper.append(upper)

data_sorted['ci_lower'] = ci_lower
data_sorted['ci_upper'] = ci_upper

x_pos = np.arange(len(data_sorted))
ax.errorbar(x_pos, data_sorted['success_rate'],
            yerr=[data_sorted['success_rate'] - data_sorted['ci_lower'],
                  data_sorted['ci_upper'] - data_sorted['success_rate']],
            fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2,
            label='Observed rates with 95% CI')

# Add pooled rate
ax.axhline(pooled_rate, color='red', linestyle='--', linewidth=2, label=f'Pooled rate={pooled_rate:.4f}')

ax.set_xlabel('Group (ordered by success rate)')
ax.set_ylabel('Success Rate')
ax.set_title('Group-Specific Success Rates with 95% Confidence Intervals\n(Wilson Score Method)')
ax.set_xticks(x_pos)
ax.set_xticklabels(data_sorted['group_id'])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'success_rates_with_ci.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: success_rates_with_ci.png")

# Figure 4: Precision vs sample size
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate CI width as measure of precision
data['ci_width'] = data.apply(lambda row: wilson_ci(row['r_successes'], row['n_trials'])[1] -
                                          wilson_ci(row['r_successes'], row['n_trials'])[0], axis=1)

ax.scatter(data['n_trials'], data['ci_width'], s=100, alpha=0.6, edgecolors='black')
ax.set_xlabel('Number of Trials (Sample Size)')
ax.set_ylabel('95% CI Width')
ax.set_title('Estimation Precision vs Sample Size\n(Smaller CI Width = More Precise)')
ax.grid(True, alpha=0.3)

# Add group labels
for idx, row in data.iterrows():
    ax.annotate(f"G{row['group_id']}\n(r={row['r_successes']})",
                (row['n_trials'], row['ci_width']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# Add reference curve
n_ref = np.linspace(data['n_trials'].min(), data['n_trials'].max(), 100)
# Approximate CI width for p=0.07 (near overall rate)
p_ref = 0.07
ci_ref = 2 * 1.96 * np.sqrt(p_ref * (1 - p_ref) / n_ref)
ax.plot(n_ref, ci_ref, 'r--', alpha=0.5, linewidth=2, label='Theoretical (p=0.07)')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'precision_vs_sample_size.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: precision_vs_sample_size.png")

print("\nData quality visualizations complete!")
