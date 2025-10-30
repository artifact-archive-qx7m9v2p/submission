"""
Sample Size Relationship Analysis
Focus: Relationship between n_trials and success_rate variability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Sample Size Effects on Success Rate Variability', fontsize=14, fontweight='bold')

# 1. Scatter plot with confidence bands
ax = axes[0]

# Calculate pooled success rate for confidence bands
pooled_p = data['r_successes'].sum() / data['n_trials'].sum()

# Create range of n values for confidence bands
n_range = np.linspace(data['n_trials'].min(), data['n_trials'].max(), 100)

# Calculate 95% confidence intervals (using normal approximation for binomial)
# CI = p +/- 1.96 * sqrt(p(1-p)/n)
ci_upper = pooled_p + 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)
ci_lower = pooled_p - 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)

# Calculate 99.7% intervals (3 sigma)
ci_upper_3sigma = pooled_p + 3 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)
ci_lower_3sigma = pooled_p - 3 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)

# Plot confidence bands
ax.fill_between(n_range, ci_lower_3sigma, ci_upper_3sigma, alpha=0.2, color='lightblue', label='99.7% CI (3σ)')
ax.fill_between(n_range, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='95% CI')

# Plot pooled estimate
ax.axhline(pooled_p, color='red', linestyle='--', linewidth=2, label=f'Pooled rate: {pooled_p:.4f}')

# Scatter plot of actual data
scatter = ax.scatter(data['n_trials'], data['success_rate'],
                     s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                     c=data['success_rate'], cmap='RdYlGn', vmin=0, vmax=0.15)

# Add group labels
for idx, row in data.iterrows():
    ax.annotate(f"{row['group_id']}",
                (row['n_trials'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('Number of Trials (n)', fontsize=11)
ax.set_ylabel('Observed Success Rate', fontsize=11)
ax.set_title('Success Rate vs Sample Size\n(with Binomial Confidence Bands)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Success Rate', fontsize=10)

# 2. Scatter plot on log scale
ax = axes[1]

# Plot on log scale
ax.fill_between(n_range, ci_lower_3sigma, ci_upper_3sigma, alpha=0.2, color='lightblue', label='99.7% CI (3σ)')
ax.fill_between(n_range, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='95% CI')
ax.axhline(pooled_p, color='red', linestyle='--', linewidth=2, label=f'Pooled rate: {pooled_p:.4f}')

scatter = ax.scatter(data['n_trials'], data['success_rate'],
                     s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                     c=data['success_rate'], cmap='RdYlGn', vmin=0, vmax=0.15)

for idx, row in data.iterrows():
    ax.annotate(f"{row['group_id']}",
                (row['n_trials'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xscale('log')
ax.set_xlabel('Number of Trials (n) - Log Scale', fontsize=11)
ax.set_ylabel('Observed Success Rate', fontsize=11)
ax.set_title('Success Rate vs Sample Size (Log Scale)\n(Easier to see small-n variation)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/02_sample_size_relationship.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/analyst_1/visualizations/02_sample_size_relationship.png")

# Analysis
print("\n" + "="*80)
print("SAMPLE SIZE RELATIONSHIP ANALYSIS")
print("="*80)

print(f"\n1. POOLED SUCCESS RATE")
print(f"Pooled estimate: {pooled_p:.6f} ({pooled_p*100:.2f}%)")
print(f"Total successes: {data['r_successes'].sum()}")
print(f"Total trials: {data['n_trials'].sum()}")

print(f"\n2. EXPECTED WIDTH OF 95% CI BY SAMPLE SIZE")
n_sizes = [50, 100, 200, 500, 810]
for n in n_sizes:
    width = 2 * 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n)
    print(f"n={n:4d}: CI width = {width:.6f} ({width*100:.2f} percentage points)")

print(f"\n3. GROUPS OUTSIDE 95% CONFIDENCE BAND")
outside_95 = []
for idx, row in data.iterrows():
    n = row['n_trials']
    p_obs = row['success_rate']
    ci_upper_val = pooled_p + 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n)
    ci_lower_val = pooled_p - 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n)

    if p_obs > ci_upper_val or p_obs < ci_lower_val:
        z_score = (p_obs - pooled_p) / np.sqrt(pooled_p * (1 - pooled_p) / n)
        outside_95.append({
            'group_id': row['group_id'],
            'n_trials': n,
            'success_rate': p_obs,
            'expected': pooled_p,
            'ci_lower': ci_lower_val,
            'ci_upper': ci_upper_val,
            'z_score': z_score
        })

if outside_95:
    print(f"Found {len(outside_95)} group(s) outside 95% CI:")
    for group in outside_95:
        print(f"  Group {group['group_id']}: rate={group['success_rate']:.4f}, "
              f"expected={group['expected']:.4f}, "
              f"CI=[{group['ci_lower']:.4f}, {group['ci_upper']:.4f}], "
              f"z-score={group['z_score']:.2f}")
else:
    print("All groups fall within 95% CI - consistent with homogeneous model")

print(f"\n4. GROUPS OUTSIDE 99.7% CONFIDENCE BAND (3 sigma)")
outside_997 = []
for idx, row in data.iterrows():
    n = row['n_trials']
    p_obs = row['success_rate']
    ci_upper_val = pooled_p + 3 * np.sqrt(pooled_p * (1 - pooled_p) / n)
    ci_lower_val = pooled_p - 3 * np.sqrt(pooled_p * (1 - pooled_p) / n)

    if p_obs > ci_upper_val or p_obs < ci_lower_val:
        z_score = (p_obs - pooled_p) / np.sqrt(pooled_p * (1 - pooled_p) / n)
        outside_997.append({
            'group_id': row['group_id'],
            'n_trials': n,
            'success_rate': p_obs,
            'z_score': z_score
        })

if outside_997:
    print(f"Found {len(outside_997)} group(s) outside 99.7% CI (EXTREME outliers):")
    for group in outside_997:
        print(f"  Group {group['group_id']}: rate={group['success_rate']:.4f}, z-score={group['z_score']:.2f}")
else:
    print("All groups fall within 99.7% CI")

print(f"\n5. CORRELATION ANALYSIS")
corr_pearson, p_pearson = stats.pearsonr(data['n_trials'], data['success_rate'])
corr_spearman, p_spearman = stats.spearmanr(data['n_trials'], data['success_rate'])
print(f"Pearson correlation: r={corr_pearson:.4f}, p-value={p_pearson:.4f}")
print(f"Spearman correlation: rho={corr_spearman:.4f}, p-value={p_spearman:.4f}")

if abs(corr_pearson) < 0.3:
    print("   -> Weak or no linear relationship")
elif abs(corr_pearson) < 0.7:
    print("   -> Moderate linear relationship")
else:
    print("   -> Strong linear relationship")

print("\n" + "="*80)
