"""
Shrinkage and Model Comparison Visualization
=============================================
Visual comparison of different pooling strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/eda/code/processed_data.csv')

# Calculate key statistics
weighted_mean = np.sum(data['y'] * data['precision']) / np.sum(data['precision'])
weights = data['precision']**2

# Calculate tau²
Q = np.sum(weights * (data['y'] - weighted_mean)**2)
df = len(data) - 1
tau_squared = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))

# Calculate shrinkage estimates
shrinkage_factors = tau_squared / (tau_squared + data['variance'])
partial_pooling = shrinkage_factors * data['y'] + (1 - shrinkage_factors) * weighted_mean

# ============================================================
# Figure 7: Shrinkage Plot
# ============================================================
print("Creating Figure 7: Shrinkage Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Shrinkage illustration
ax = axes[0, 0]
studies = data['study'].astype(int)

for idx, row in data.iterrows():
    study_id = int(row['study'])
    # Arrow from observed to shrunken
    ax.arrow(row['y'], study_id, partial_pooling[idx] - row['y'], 0,
             head_width=0.3, head_length=0.5, fc='steelblue', ec='steelblue',
             alpha=0.6, length_includes_head=True)
    # Observed point
    ax.scatter(row['y'], study_id, s=150, c='red', marker='o',
               edgecolors='black', linewidth=1.5, label='Observed' if idx == 0 else '',
               zorder=3)
    # Shrunken point
    ax.scatter(partial_pooling[idx], study_id, s=150, c='blue', marker='s',
               edgecolors='black', linewidth=1.5, label='Shrunken' if idx == 0 else '',
               zorder=3)

# Pooled estimate line
ax.axvline(weighted_mean, color='green', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Pooled Estimate')

ax.set_yticks(studies)
ax.set_yticklabels([f"Study {s}" for s in studies])
ax.set_xlabel('Effect Size', fontsize=12)
ax.set_ylabel('Study', fontsize=12)
ax.set_title('Shrinkage: Observed → Partial Pooling', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3, axis='x')

# Panel 2: Three estimates comparison
ax = axes[0, 1]
x_pos = np.arange(len(studies))
width = 0.25

bars1 = ax.bar(x_pos - width, data['y'], width, label='No Pooling (Observed)',
               alpha=0.7, color='red', edgecolor='black')
bars2 = ax.bar(x_pos, partial_pooling, width, label='Partial Pooling',
               alpha=0.7, color='blue', edgecolor='black')
bars3 = ax.bar(x_pos + width, np.full(len(data), weighted_mean), width,
               label='Complete Pooling', alpha=0.7, color='green', edgecolor='black')

ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Effect Size', fontsize=12)
ax.set_title('Three Pooling Strategies Compared', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{s}" for s in studies])
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Panel 3: Shrinkage factor vs precision
ax = axes[1, 0]
ax.scatter(data['precision'], shrinkage_factors, s=150, alpha=0.7, c='purple',
           edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax.annotate(f"{int(row['study'])}", (row['precision'], shrinkage_factors[idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax.set_xlabel('Precision (1/sigma)', fontsize=12)
ax.set_ylabel('Shrinkage Factor (B)', fontsize=12)
ax.set_title('Shrinkage Factor vs Study Precision', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Add text annotation
ax.text(0.05, 0.95, 'Higher precision (smaller SE)\n→ Less shrinkage\n→ More trust in study',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=10)

# Panel 4: Error reduction from shrinkage
ax = axes[1, 1]
# Calculate "true" posterior variance for each estimate
# var(theta_i | y_i) = 1 / (1/sigma_i² + 1/tau²)
posterior_var = 1 / (1/data['variance'] + 1/tau_squared)
posterior_se = np.sqrt(posterior_var)
original_se = data['sigma']

reduction_pct = 100 * (1 - posterior_se / original_se)

bars = ax.bar(studies, reduction_pct, alpha=0.7, color='teal', edgecolor='black')
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('SE Reduction (%)', fontsize=12)
ax.set_title('Standard Error Reduction from Shrinkage', fontsize=12, fontweight='bold')
ax.set_xticks(studies)
ax.grid(alpha=0.3, axis='y')

# Add average line
ax.axhline(reduction_pct.mean(), color='red', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Average = {reduction_pct.mean():.1f}%')
ax.legend()

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/07_shrinkage_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 8: Model Comparison and Uncertainty
# ============================================================
print("Creating Figure 8: Model Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Point estimates comparison with CIs
ax = axes[0, 0]
studies = data['study'].astype(int)
y_pos = np.arange(len(studies)) * 3  # Space out for clarity

# No pooling (observed with original SEs)
ax.errorbar(data['y'], y_pos, xerr=1.96*data['sigma'],
            fmt='o', markersize=8, color='red', ecolor='red',
            elinewidth=2, capsize=5, alpha=0.7, label='No Pooling')

# Partial pooling (with reduced SEs)
posterior_se = np.sqrt(1 / (1/data['variance'] + 1/tau_squared))
ax.errorbar(partial_pooling, y_pos + 0.8, xerr=1.96*posterior_se,
            fmt='s', markersize=8, color='blue', ecolor='blue',
            elinewidth=2, capsize=5, alpha=0.7, label='Partial Pooling')

# Complete pooling
pooled_se = 1 / np.sqrt(np.sum(weights))
ax.errorbar([weighted_mean]*len(studies), y_pos + 1.6, xerr=1.96*pooled_se,
            fmt='^', markersize=8, color='green', ecolor='green',
            elinewidth=2, capsize=5, alpha=0.7, label='Complete Pooling')

ax.set_yticks(y_pos + 0.8)
ax.set_yticklabels([f"Study {s}" for s in studies])
ax.set_xlabel('Effect Size (95% CI)', fontsize=12)
ax.set_ylabel('Study', fontsize=12)
ax.set_title('Effect Estimates with Uncertainty: Three Models', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Panel 2: Bootstrap distributions
ax = axes[0, 1]
np.random.seed(42)
n_boot = 1000

# Bootstrap for weighted mean
boot_means = []
for _ in range(n_boot):
    boot_idx = np.random.choice(len(data), len(data), replace=True)
    boot_data = data.iloc[boot_idx]
    boot_weights = boot_data['precision']**2
    boot_mean = np.sum(boot_data['y'] * boot_weights) / np.sum(boot_weights)
    boot_means.append(boot_mean)

ax.hist(boot_means, bins=30, alpha=0.6, color='steelblue', edgecolor='black', density=True)
ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Estimate = {weighted_mean:.2f}')
ax.axvline(np.percentile(boot_means, 2.5), color='orange', linestyle=':', linewidth=2)
ax.axvline(np.percentile(boot_means, 97.5), color='orange', linestyle=':', linewidth=2,
           label='95% Bootstrap CI')
ax.set_xlabel('Pooled Effect Estimate', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Bootstrap Distribution of Pooled Estimate', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Prediction intervals
ax = axes[1, 0]

# 95% CI for pooled effect (fixed effect)
ci_fixed = 1.96 * pooled_se

# 95% Prediction interval (random effects)
pred_var = tau_squared + pooled_se**2
pred_se = np.sqrt(pred_var)
pi_random = 1.96 * pred_se

# Visualize
intervals = [ci_fixed, pi_random]
labels = ['95% CI\n(Pooled Effect)', '95% Prediction Interval\n(New Study)']
colors = ['green', 'orange']

y_positions = [1, 2]
for i, (interval, label, color) in enumerate(zip(intervals, labels, colors)):
    ax.barh(y_positions[i], 2*interval, left=weighted_mean-interval, height=0.6,
            alpha=0.6, color=color, edgecolor='black', linewidth=2)
    ax.text(weighted_mean, y_positions[i], f'{2*interval:.2f}',
            ha='center', va='center', fontsize=10, fontweight='bold')

ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_yticks(y_positions)
ax.set_yticklabels(labels)
ax.set_xlabel('Effect Size', fontsize=12)
ax.set_title('Confidence vs Prediction Intervals', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Panel 4: Variance decomposition
ax = axes[1, 1]

# Pie chart of variance components
total_var = data['y'].var()
within_var = data['variance'].mean()
between_var = tau_squared

# Approximate decomposition
total_variance_approx = within_var + between_var
within_pct = 100 * within_var / total_variance_approx
between_pct = 100 * between_var / total_variance_approx

sizes = [within_pct, between_pct]
labels_pie = [f'Within-Study\nVariance\n({within_pct:.1f}%)',
              f'Between-Study\nVariance\n({between_pct:.1f}%)']
colors_pie = ['lightblue', 'coral']
explode = (0.05, 0.05)

ax.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
       autopct='', shadow=True, startangle=90)
ax.set_title(f'Variance Decomposition\n(I² = {100*between_var/total_variance_approx:.1f}%)',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/08_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("Shrinkage visualizations saved!")
print("="*60)
