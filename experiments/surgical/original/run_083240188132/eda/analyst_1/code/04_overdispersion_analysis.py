"""
Overdispersion Analysis
Testing for overdispersion beyond binomial expectation
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

print("="*80)
print("OVERDISPERSION ANALYSIS")
print("="*80)

# Calculate expected and observed variance
# For binomial: Var(p) = p*(1-p)/n
data['expected_var'] = p_overall * (1 - p_overall) / data['n']
data['observed_var'] = (data['proportion'] - p_overall)**2

print("\n1. VARIANCE ANALYSIS")
print("-"*40)
print(f"Overall proportion: {p_overall:.4f}")
print(f"\nExpected variance (if proportions were homogeneous):")
print(f"  Weighted mean: {np.average(data['expected_var'], weights=data['n']):.6f}")

# Calculate observed variance of proportions
obs_var = np.var(data['proportion'], ddof=1)
exp_var = np.average(data['expected_var'], weights=data['n'])
print(f"\nObserved variance of group proportions:")
print(f"  Variance: {obs_var:.6f}")
print(f"\nDispersion parameter (observed/expected):")
print(f"  Phi = {obs_var/exp_var:.2f}")
if obs_var/exp_var > 1.5:
    print("  --> Substantial OVERDISPERSION detected")
elif obs_var/exp_var < 0.67:
    print("  --> UNDERDISPERSION detected")
else:
    print("  --> Approximately consistent with binomial variance")

# Chi-square test for homogeneity
print("\n2. CHI-SQUARE TEST FOR HOMOGENEITY")
print("-"*40)

# Calculate chi-square statistic
chi_sq = 0
for idx, row in data.iterrows():
    n, r = row['n'], row['r']
    expected = n * p_overall
    chi_sq += (r - expected)**2 / (expected * (1 - p_overall))

df = len(data) - 1
p_value = 1 - stats.chi2.cdf(chi_sq, df)

print(f"Chi-square statistic: {chi_sq:.2f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.001:
    print("*** Highly significant heterogeneity (p < 0.001)")
elif p_value < 0.05:
    print("** Significant heterogeneity (p < 0.05)")
else:
    print("No significant heterogeneity detected")

# Overdispersion factor
overdispersion = chi_sq / df
print(f"\nOverdispersion factor (chi-sq/df): {overdispersion:.2f}")

# Pearson residuals
print("\n3. STANDARDIZED RESIDUALS")
print("-"*40)

data['expected_r'] = data['n'] * p_overall
data['pearson_residual'] = (data['r'] - data['expected_r']) / np.sqrt(data['expected_r'] * (1 - p_overall))

print("Groups with |residual| > 2 (potential outliers):")
outlier_residuals = data[np.abs(data['pearson_residual']) > 2]
if len(outlier_residuals) > 0:
    for idx, row in outlier_residuals.iterrows():
        print(f"  Group {row['group']}: residual = {row['pearson_residual']:.2f}, "
              f"observed = {row['r']}, expected = {row['expected_r']:.1f}")
else:
    print("  None detected")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Observed vs Expected variance
ax1 = axes[0, 0]
ax1.scatter(data['expected_var'], data['observed_var'], s=data['n']*2,
           alpha=0.6, edgecolors='black', linewidth=1)
for idx, row in data.iterrows():
    ax1.annotate(str(row['group']), (row['expected_var'], row['observed_var']),
                ha='center', va='center', fontsize=8, fontweight='bold')

# Add identity line
max_val = max(data['expected_var'].max(), data['observed_var'].max())
ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Expected (1:1 line)')
ax1.set_xlabel('Expected Variance (binomial)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Observed Variance (deviation²)', fontsize=11, fontweight='bold')
ax1.set_title('Variance: Observed vs Expected', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Funnel plot (proportion vs precision)
ax2 = axes[0, 1]
precision = 1 / np.sqrt(data['expected_var'])
ax2.scatter(precision, data['proportion'], s=80, alpha=0.6, edgecolors='black', linewidth=1)

# Add control limits (95% and 99.8%)
precision_range = np.linspace(precision.min(), precision.max(), 100)
ax2.axhline(y=p_overall, color='red', linestyle='-', linewidth=2, label='Overall rate')
ax2.plot(precision_range, p_overall + 1.96/precision_range, 'b--', linewidth=1.5, alpha=0.7, label='95% limits')
ax2.plot(precision_range, p_overall - 1.96/precision_range, 'b--', linewidth=1.5, alpha=0.7)
ax2.plot(precision_range, p_overall + 3.09/precision_range, 'orange', linestyle='--', linewidth=1, alpha=0.7, label='99.8% limits')
ax2.plot(precision_range, p_overall - 3.09/precision_range, 'orange', linestyle='--', linewidth=1, alpha=0.7)

# Label outliers
for idx, row in data.iterrows():
    prec = 1 / np.sqrt(row['expected_var'])
    if abs(row['proportion'] - p_overall) > 1.96/prec:
        ax2.annotate(str(row['group']), (prec, row['proportion']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.set_xlabel('Precision (1/SE)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Proportion', fontsize=11, fontweight='bold')
ax2.set_title('Funnel Plot for Overdispersion Detection', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Pearson residuals
ax3 = axes[1, 0]
colors = ['red' if abs(x) > 2 else 'steelblue' for x in data['pearson_residual']]
ax3.bar(data['group'], data['pearson_residual'], color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(y=0, color='black', linewidth=2)
ax3.axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='±2 threshold')
ax3.axhline(y=-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.set_xlabel('Group', fontsize=11, fontweight='bold')
ax3.set_ylabel('Pearson Residual', fontsize=11, fontweight='bold')
ax3.set_title('Standardized Residuals by Group', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Absolute residual vs sample size (heteroscedasticity check)
ax4 = axes[1, 1]
ax4.scatter(data['n'], np.abs(data['pearson_residual']), s=100,
           alpha=0.6, edgecolors='black', linewidth=1)
for idx, row in data.iterrows():
    if abs(row['pearson_residual']) > 2:
        ax4.annotate(str(row['group']), (row['n'], abs(row['pearson_residual'])),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

ax4.axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Critical value (|z|=2)')
ax4.set_xlabel('Sample Size (n)', fontsize=11, fontweight='bold')
ax4.set_ylabel('|Pearson Residual|', fontsize=11, fontweight='bold')
ax4.set_title('Residual Magnitude vs Sample Size', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/03_overdispersion_analysis.png')
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
