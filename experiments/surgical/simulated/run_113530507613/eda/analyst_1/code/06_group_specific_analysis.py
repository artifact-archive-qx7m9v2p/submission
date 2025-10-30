"""
Group-Specific Analysis
Focus: Identifying and characterizing outlier groups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Calculate pooled success rate and derived metrics
pooled_p = data['r_successes'].sum() / data['n_trials'].sum()
data['expected_se'] = np.sqrt(pooled_p * (1 - pooled_p) / data['n_trials'])
data['z_score'] = (data['success_rate'] - pooled_p) / data['expected_se']
data['ci_lower_95'] = pooled_p - 1.96 * data['expected_se']
data['ci_upper_95'] = pooled_p + 1.96 * data['expected_se']
data['outside_ci'] = (data['success_rate'] < data['ci_lower_95']) | (data['success_rate'] > data['ci_upper_95'])

print("="*80)
print("GROUP-SPECIFIC ANALYSIS")
print("="*80)

print("\n1. COMPLETE GROUP SUMMARY (Sorted by absolute z-score)")
summary = data[['group_id', 'n_trials', 'r_successes', 'success_rate', 'expected_se', 'z_score', 'outside_ci']].copy()
summary = summary.sort_values('z_score', key=abs, ascending=False)
print(summary.to_string(index=False))

print("\n2. GROUP CATEGORIZATION")

# Categorize groups
extreme_outliers = data[np.abs(data['z_score']) > 3]
moderate_outliers = data[(np.abs(data['z_score']) > 1.96) & (np.abs(data['z_score']) <= 3)]
typical_groups = data[np.abs(data['z_score']) <= 1.96]

print(f"\nExtreme outliers (|z| > 3): {len(extreme_outliers)} groups")
if len(extreme_outliers) > 0:
    print(extreme_outliers[['group_id', 'n_trials', 'success_rate', 'z_score']].to_string(index=False))

print(f"\nModerate outliers (1.96 < |z| <= 3): {len(moderate_outliers)} groups")
if len(moderate_outliers) > 0:
    print(moderate_outliers[['group_id', 'n_trials', 'success_rate', 'z_score']].to_string(index=False))

print(f"\nTypical groups (|z| <= 1.96): {len(typical_groups)} groups")
if len(typical_groups) > 0:
    print(typical_groups[['group_id', 'n_trials', 'success_rate', 'z_score']].to_string(index=False))

print("\n3. CHARACTERISTICS BY CATEGORY")

categories = []
for cat_name, cat_data in [('Extreme', extreme_outliers), ('Moderate', moderate_outliers), ('Typical', typical_groups)]:
    if len(cat_data) > 0:
        categories.append({
            'Category': cat_name,
            'Count': len(cat_data),
            'Mean n_trials': cat_data['n_trials'].mean(),
            'Mean success_rate': cat_data['success_rate'].mean(),
            'SD success_rate': cat_data['success_rate'].std()
        })

cat_df = pd.DataFrame(categories)
print(cat_df.to_string(index=False))

print("\n4. SAMPLE SIZE DISTRIBUTION BY OUTLIER STATUS")
print(f"\nOutlier groups (n={len(data[data['outside_ci']])}): mean n_trials = {data[data['outside_ci']]['n_trials'].mean():.1f}")
print(f"Non-outlier groups (n={len(data[~data['outside_ci']])}): mean n_trials = {data[~data['outside_ci']]['n_trials'].mean():.1f}")

# Statistical test
if len(data[data['outside_ci']]) > 0 and len(data[~data['outside_ci']]) > 0:
    t_stat, p_val = stats.ttest_ind(data[data['outside_ci']]['n_trials'],
                                     data[~data['outside_ci']]['n_trials'])
    print(f"T-test p-value: {p_val:.4f}")

print("\n5. DETAILED OUTLIER PROFILES")
for idx, row in extreme_outliers.iterrows():
    print(f"\n--- GROUP {row['group_id']} (EXTREME OUTLIER) ---")
    print(f"Sample size: {row['n_trials']}")
    print(f"Successes: {row['r_successes']}")
    print(f"Success rate: {row['success_rate']:.6f}")
    print(f"Pooled rate: {pooled_p:.6f}")
    print(f"Deviation: {(row['success_rate'] - pooled_p):.6f} ({(row['success_rate'] - pooled_p)/pooled_p*100:+.1f}%)")
    print(f"Z-score: {row['z_score']:.4f}")
    print(f"95% CI: [{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}]")

    # Calculate probability of observing this or more extreme under null
    p_extreme = 2 * (1 - stats.norm.cdf(abs(row['z_score'])))
    print(f"P(|z| >= {abs(row['z_score']):.2f}): {p_extreme:.6f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Group-Specific Analysis: Identifying Outliers', fontsize=14, fontweight='bold')

# Plot 1: Group profiles sorted by z-score
ax = axes[0, 0]
data_sorted = data.sort_values('z_score', ascending=False)
x_pos = np.arange(len(data_sorted))

colors = ['red' if abs(z) > 3 else 'orange' if abs(z) > 1.96 else 'green'
          for z in data_sorted['z_score']]

bars = ax.bar(x_pos, data_sorted['z_score'], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(1.96, color='blue', linestyle='--', linewidth=1.5, label='95% CI boundary')
ax.axhline(-1.96, color='blue', linestyle='--', linewidth=1.5)
ax.axhline(3, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='3σ boundary')
ax.axhline(-3, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Z-Score', fontsize=11)
ax.set_title('Standardized Deviations by Group', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(data_sorted['group_id'].astype(int))
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Success rate by group
ax = axes[0, 1]
data_sorted2 = data.sort_values('success_rate', ascending=False)
x_pos2 = np.arange(len(data_sorted2))

colors2 = ['red' if outside else 'green' for outside in data_sorted2['outside_ci']]
bars = ax.bar(x_pos2, data_sorted2['success_rate'], color=colors2, alpha=0.7, edgecolor='black')
ax.axhline(pooled_p, color='blue', linestyle='--', linewidth=2, label=f'Pooled: {pooled_p:.4f}')

# Add error bars representing 95% CI
errors = 1.96 * data_sorted2['expected_se']
ax.errorbar(x_pos2, [pooled_p]*len(data_sorted2), yerr=errors,
            fmt='none', ecolor='gray', alpha=0.5, capsize=3)

ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('Success Rate by Group (with Expected 95% CI)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos2)
ax.set_xticklabels(data_sorted2['group_id'].astype(int))
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Sample size vs deviation magnitude
ax = axes[1, 0]
scatter = ax.scatter(data['n_trials'], np.abs(data['z_score']),
                     s=150, alpha=0.7, edgecolors='black', linewidth=1.5,
                     c=np.abs(data['z_score']), cmap='YlOrRd')

ax.axhline(1.96, color='blue', linestyle='--', linewidth=1.5, label='95% threshold')
ax.axhline(3, color='red', linestyle='--', linewidth=1.5, label='3σ threshold')

for idx, row in data.iterrows():
    ax.annotate(f"{row['group_id']}",
                (row['n_trials'], abs(row['z_score'])),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Number of Trials', fontsize=11)
ax.set_ylabel('|Z-Score|', fontsize=11)
ax.set_title('Sample Size vs Deviation Magnitude', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='|Z-Score|')

# Plot 4: Distribution of z-scores
ax = axes[1, 1]
ax.hist(data['z_score'], bins=10, edgecolor='black', alpha=0.7, color='steelblue', density=True)

# Overlay theoretical N(0,1) distribution
x_range = np.linspace(-4, 4, 100)
ax.plot(x_range, stats.norm.pdf(x_range), 'r-', linewidth=2, label='N(0,1)')

ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(data['z_score'].mean(), color='green', linestyle='--', linewidth=2,
           label=f'Mean: {data["z_score"].mean():.2f}')

ax.set_xlabel('Z-Score', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Z-Scores', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/05_group_specific_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: /workspace/eda/analyst_1/visualizations/05_group_specific_analysis.png")

print("\n" + "="*80)
