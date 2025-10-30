"""
Funnel Plot Analysis
Focus: Detecting size-dependent heterogeneity using funnel plot methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Calculate pooled success rate
pooled_p = data['r_successes'].sum() / data['n_trials'].sum()

# Create funnel plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Funnel Plot Analysis: Detecting Heterogeneity', fontsize=14, fontweight='bold')

# Plot 1: Standard Funnel Plot (success rate vs precision)
ax = axes[0]

# Calculate precision (inverse of standard error)
data['se'] = np.sqrt(pooled_p * (1 - pooled_p) / data['n_trials'])
data['precision'] = 1 / data['se']

# Calculate confidence limits
precisions = np.linspace(data['precision'].min() * 0.9, data['precision'].max() * 1.1, 100)
se_vals = 1 / precisions

ci_95_upper = pooled_p + 1.96 * se_vals
ci_95_lower = pooled_p - 1.96 * se_vals
ci_99_upper = pooled_p + 2.576 * se_vals
ci_99_lower = pooled_p - 2.576 * se_vals
ci_999_upper = pooled_p + 3.29 * se_vals
ci_999_lower = pooled_p - 3.29 * se_vals

# Plot confidence limits
ax.plot(precisions, ci_999_upper, 'b:', linewidth=1.5, alpha=0.5, label='99.9% CI')
ax.plot(precisions, ci_999_lower, 'b:', linewidth=1.5, alpha=0.5)
ax.plot(precisions, ci_99_upper, 'b--', linewidth=1.5, alpha=0.7, label='99% CI')
ax.plot(precisions, ci_99_lower, 'b--', linewidth=1.5, alpha=0.7)
ax.plot(precisions, ci_95_upper, 'b-', linewidth=2, label='95% CI')
ax.plot(precisions, ci_95_lower, 'b-', linewidth=2)

# Plot pooled estimate
ax.axhline(pooled_p, color='red', linestyle='-', linewidth=2, label=f'Pooled: {pooled_p:.4f}')

# Scatter plot
scatter = ax.scatter(data['precision'], data['success_rate'],
                     s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                     c=data['success_rate'], cmap='RdYlGn', vmin=0, vmax=0.15, zorder=5)

# Add group labels
for idx, row in data.iterrows():
    ax.annotate(f"{row['group_id']}",
                (row['precision'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Precision (1/SE)', fontsize=11)
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('Funnel Plot: Success Rate vs Precision', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Plot 2: Alternative Funnel (success rate vs sqrt(n))
ax = axes[1]

# Calculate confidence limits using sqrt(n)
sqrt_n_vals = np.linspace(np.sqrt(data['n_trials'].min()) * 0.9,
                          np.sqrt(data['n_trials'].max()) * 1.1, 100)
n_vals = sqrt_n_vals ** 2

ci_95_upper_alt = pooled_p + 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_vals)
ci_95_lower_alt = pooled_p - 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_vals)
ci_99_upper_alt = pooled_p + 2.576 * np.sqrt(pooled_p * (1 - pooled_p) / n_vals)
ci_99_lower_alt = pooled_p - 2.576 * np.sqrt(pooled_p * (1 - pooled_p) / n_vals)

# Plot confidence limits
ax.plot(sqrt_n_vals, ci_99_upper_alt, 'b--', linewidth=1.5, alpha=0.7, label='99% CI')
ax.plot(sqrt_n_vals, ci_99_lower_alt, 'b--', linewidth=1.5, alpha=0.7)
ax.plot(sqrt_n_vals, ci_95_upper_alt, 'b-', linewidth=2, label='95% CI')
ax.plot(sqrt_n_vals, ci_95_lower_alt, 'b-', linewidth=2)

# Plot pooled estimate
ax.axhline(pooled_p, color='red', linestyle='-', linewidth=2, label=f'Pooled: {pooled_p:.4f}')

# Scatter plot
data['sqrt_n'] = np.sqrt(data['n_trials'])
scatter = ax.scatter(data['sqrt_n'], data['success_rate'],
                     s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                     c=data['success_rate'], cmap='RdYlGn', vmin=0, vmax=0.15, zorder=5)

# Add group labels
for idx, row in data.iterrows():
    ax.annotate(f"{row['group_id']}",
                (row['sqrt_n'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('√(Number of Trials)', fontsize=11)
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('Alternative Funnel Plot: Success Rate vs √n', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/03_funnel_plot.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/analyst_1/visualizations/03_funnel_plot.png")

# Analysis
print("\n" + "="*80)
print("FUNNEL PLOT ANALYSIS")
print("="*80)

print(f"\n1. INTERPRETATION OF FUNNEL PLOT")
print("In a homogeneous population, we expect:")
print("  - 95% of points within 95% CI funnel")
print("  - Random scatter around pooled estimate")
print("  - Tighter clustering at higher precision (larger n)")
print("")
print("Evidence of heterogeneity:")
print("  - Systematic deviation from pooled estimate")
print("  - Points outside funnel even at high precision")
print("  - Asymmetric funnel (one-sided excess)")

print(f"\n2. OBSERVED PATTERNS")
outside_95_count = len(data[
    (data['success_rate'] > pooled_p + 1.96 * data['se']) |
    (data['success_rate'] < pooled_p - 1.96 * data['se'])
])
print(f"Groups outside 95% CI: {outside_95_count} / {len(data)} ({outside_95_count/len(data)*100:.1f}%)")
print(f"Expected under homogeneity: ~5%")

if outside_95_count / len(data) > 0.10:
    print("   -> SUBSTANTIAL EXCESS: Strong evidence of heterogeneity")
elif outside_95_count / len(data) > 0.05:
    print("   -> MODERATE EXCESS: Some evidence of heterogeneity")
else:
    print("   -> Within expected range: Consistent with homogeneity")

print(f"\n3. HIGH-PRECISION OUTLIERS")
# Consider groups with n > 150 as "high precision"
high_precision = data[data['n_trials'] > 150]
print(f"Groups with n > 150: {len(high_precision)}")

high_prec_outside = high_precision[
    (high_precision['success_rate'] > pooled_p + 1.96 * high_precision['se']) |
    (high_precision['success_rate'] < pooled_p - 1.96 * high_precision['se'])
]
print(f"High-precision groups outside 95% CI: {len(high_prec_outside)}")

if len(high_prec_outside) > 0:
    print("   -> STRONG EVIDENCE: Heterogeneity not explained by sampling variation")
    print("\nHigh-precision outliers:")
    print(high_prec_outside[['group_id', 'n_trials', 'success_rate', 'se']])
else:
    print("   -> All high-precision groups within expected range")

print(f"\n4. ASYMMETRY TEST")
above_pooled = data[data['success_rate'] > pooled_p]
below_pooled = data[data['success_rate'] < pooled_p]
print(f"Groups above pooled rate: {len(above_pooled)} (mean n={above_pooled['n_trials'].mean():.1f})")
print(f"Groups below pooled rate: {len(below_pooled)} (mean n={below_pooled['n_trials'].mean():.1f})")

# Egger's test for funnel plot asymmetry (simplified version)
# Regress standardized effect (z-score) on precision
data['z_score'] = (data['success_rate'] - pooled_p) / data['se']
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(data['precision'], data['z_score'])
print(f"\nEgger's regression:")
print(f"  Intercept: {intercept:.4f} (p={p_value:.4f})")
if abs(intercept) > 1 and p_value < 0.1:
    print("   -> Significant asymmetry detected")
else:
    print("   -> No significant asymmetry")

print("\n" + "="*80)
