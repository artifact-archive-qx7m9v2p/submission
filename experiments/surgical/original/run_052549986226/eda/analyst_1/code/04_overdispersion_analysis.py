"""
Overdispersion Analysis: Testing for extra-binomial variation
EDA Analyst 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data_analyst_1.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'analyst_1' / 'visualizations'

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("OVERDISPERSION ANALYSIS")
print("=" * 70)

# ============================================================================
# Calculate overdispersion metrics
# ============================================================================

# Pooled proportion
pooled_p = df['r_successes'].sum() / df['n_trials'].sum()
total_n = df['n_trials'].sum()
total_r = df['r_successes'].sum()

print(f"\nPooled estimate:")
print(f"  Total trials: {total_n}")
print(f"  Total successes: {total_r}")
print(f"  Pooled proportion: {pooled_p:.4f}")

# ============================================================================
# Method 1: Chi-squared test for homogeneity
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 1: Chi-squared Test for Homogeneity")
print("-" * 70)

# Calculate expected and observed under binomial assumption
df['expected_successes'] = df['n_trials'] * pooled_p
df['expected_failures'] = df['n_trials'] * (1 - pooled_p)

# Chi-squared statistic
chi_squared = sum(((df['r_successes'] - df['expected_successes'])**2 / df['expected_successes']) +
                  ((df['n_trials'] - df['r_successes'] - df['expected_failures'])**2 / df['expected_failures']))

df_chi = len(df) - 1  # degrees of freedom
p_value_chi = 1 - stats.chi2.cdf(chi_squared, df_chi)

print(f"\nChi-squared statistic: {chi_squared:.4f}")
print(f"Degrees of freedom: {df_chi}")
print(f"P-value: {p_value_chi:.4f}")
print(f"Critical value (α=0.05): {stats.chi2.ppf(0.95, df_chi):.4f}")
print(f"\nInterpretation: {'REJECT homogeneity - Evidence of overdispersion' if p_value_chi < 0.05 else 'Cannot reject homogeneity'}")

# ============================================================================
# Method 2: Dispersion parameter (phi)
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 2: Dispersion Parameter (φ)")
print("-" * 70)

# Pearson chi-squared / df
phi_hat = chi_squared / df_chi
print(f"\nDispersion parameter (φ): {phi_hat:.4f}")
print(f"Interpretation:")
print(f"  φ = 1: Binomial variation (no overdispersion)")
print(f"  φ > 1: Overdispersion present")
print(f"  φ < 1: Underdispersion present")
print(f"\n  Result: {'OVERDISPERSION detected' if phi_hat > 1 else 'UNDERDISPERSION detected' if phi_hat < 1 else 'Binomial variation'}")
print(f"  Magnitude: {(phi_hat - 1) * 100:.1f}% {'more' if phi_hat > 1 else 'less'} variance than expected")

# ============================================================================
# Method 3: Variance components analysis
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 3: Variance Components")
print("-" * 70)

# Observed variance
observed_var = df['success_rate'].var(ddof=1)

# Expected variance under binomial (using delta method)
# Var(p_hat) = E[Var(p_hat|n)] + Var(E[p_hat|n])
# For binomial: E[Var(p_hat|n)] = E[p(1-p)/n]
expected_var_binomial = np.mean(pooled_p * (1 - pooled_p) / df['n_trials'])

# Alternative: weighted variance
weights = df['n_trials']
weighted_expected_var = np.sum(weights * pooled_p * (1 - pooled_p) / df['n_trials']) / np.sum(weights)

print(f"\nObserved variance in success rates: {observed_var:.6f}")
print(f"Expected variance (unweighted): {expected_var_binomial:.6f}")
print(f"Expected variance (weighted): {weighted_expected_var:.6f}")
print(f"\nVariance ratio (observed/expected unweighted): {observed_var / expected_var_binomial:.4f}")
print(f"Variance ratio (observed/expected weighted): {observed_var / weighted_expected_var:.4f}")

# Excess variance
excess_var = observed_var - expected_var_binomial
print(f"\nExcess variance: {excess_var:.6f}")
print(f"Proportion of total variance due to overdispersion: {excess_var / observed_var * 100:.1f}%")

# ============================================================================
# Method 4: Moment-based overdispersion test
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 4: Quasi-likelihood Dispersion")
print("-" * 70)

# Calculate Pearson residuals
df['pearson_residual'] = (df['r_successes'] - df['expected_successes']) / np.sqrt(df['expected_successes'])

# Sum of squared Pearson residuals
sum_sq_residuals = np.sum(df['pearson_residual']**2)
quasi_dispersion = sum_sq_residuals / df_chi

print(f"\nSum of squared Pearson residuals: {sum_sq_residuals:.4f}")
print(f"Quasi-likelihood dispersion: {quasi_dispersion:.4f}")
print(f"Interpretation: {'OVERDISPERSION' if quasi_dispersion > 1 else 'UNDERDISPERSION' if quasi_dispersion < 1 else 'Binomial variation'}")

# Calculate standardized residuals for plotting
df['expected_variance'] = pooled_p * (1 - pooled_p) / df['n_trials']
df['expected_se'] = np.sqrt(df['expected_variance'])
df['standardized_residual'] = (df['success_rate'] - pooled_p) / df['expected_se']

# ============================================================================
# FIGURE: Funnel Plot
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Funnel Plot: Overdispersion Assessment', fontsize=16, fontweight='bold')

# Panel A: Classic funnel plot (success rate vs sample size)
ax = axes[0]

# Create funnel lines (confidence/prediction intervals)
n_range = np.linspace(1, df['n_trials'].max() * 1.1, 500)

# 95% and 99.8% (3 sigma) control limits
ci_95_upper = pooled_p + 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)
ci_95_lower = pooled_p - 1.96 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)
ci_998_upper = pooled_p + 3 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)
ci_998_lower = pooled_p - 3 * np.sqrt(pooled_p * (1 - pooled_p) / n_range)

# Clip at [0, 1]
ci_95_lower = np.maximum(ci_95_lower, 0)
ci_95_upper = np.minimum(ci_95_upper, 1)
ci_998_lower = np.maximum(ci_998_lower, 0)
ci_998_upper = np.minimum(ci_998_upper, 1)

# Plot control limits
ax.fill_between(n_range, ci_998_lower, ci_998_upper, alpha=0.15, color='blue', label='99.8% limits (3σ)')
ax.fill_between(n_range, ci_95_lower, ci_95_upper, alpha=0.25, color='blue', label='95% limits (2σ)')

# Plot pooled proportion
ax.axhline(pooled_p, color='red', linestyle='--', linewidth=2, label=f'Pooled proportion = {pooled_p:.4f}')

# Plot observed data points
colors = ['red' if abs(z) > 1.96 else 'blue' for z in df['standardized_residual']]
ax.scatter(df['n_trials'], df['success_rate'], s=150, alpha=0.7,
           c=colors, edgecolor='black', linewidth=1.5, zorder=5)

# Add group labels
for idx, row in df.iterrows():
    color = 'red' if abs(df.loc[idx, 'standardized_residual']) > 1.96 else 'black'
    ax.annotate(f"G{row['group']}",
                (row['n_trials'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold' if color=='red' else 'normal',
                color=color)

ax.set_xlabel('Sample Size (n_trials)', fontsize=12)
ax.set_ylabel('Success Rate', fontsize=12)
ax.set_title('(A) Standard Funnel Plot\n(red points = |z| > 1.96)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, df['n_trials'].max() * 1.1)

# Panel B: Funnel plot with precision (1/SE) on x-axis
ax = axes[1]

# Calculate precision
df['precision'] = 1 / np.sqrt(pooled_p * (1 - pooled_p) / df['n_trials'])
precision_range = np.linspace(0.1, df['precision'].max() * 1.1, 500)

# Control limits
ci_95_upper_prec = pooled_p + 1.96 / precision_range
ci_95_lower_prec = pooled_p - 1.96 / precision_range
ci_998_upper_prec = pooled_p + 3 / precision_range
ci_998_lower_prec = pooled_p - 3 / precision_range

# Clip at [0, 1]
ci_95_lower_prec = np.maximum(ci_95_lower_prec, 0)
ci_95_upper_prec = np.minimum(ci_95_upper_prec, 1)
ci_998_lower_prec = np.maximum(ci_998_lower_prec, 0)
ci_998_upper_prec = np.minimum(ci_998_upper_prec, 1)

# Plot control limits
ax.fill_between(precision_range, ci_998_lower_prec, ci_998_upper_prec,
                alpha=0.15, color='blue', label='99.8% limits (3σ)')
ax.fill_between(precision_range, ci_95_lower_prec, ci_95_upper_prec,
                alpha=0.25, color='blue', label='95% limits (2σ)')

# Plot pooled proportion
ax.axhline(pooled_p, color='red', linestyle='--', linewidth=2, label=f'Pooled proportion = {pooled_p:.4f}')

# Plot observed data points
ax.scatter(df['precision'], df['success_rate'], s=150, alpha=0.7,
           c=colors, edgecolor='black', linewidth=1.5, zorder=5)

# Add group labels
for idx, row in df.iterrows():
    color = 'red' if abs(df.loc[idx, 'standardized_residual']) > 1.96 else 'black'
    ax.annotate(f"G{row['group']}",
                (df.loc[idx, 'precision'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold' if color=='red' else 'normal',
                color=color)

ax.set_xlabel('Precision (1 / SE)', fontsize=12)
ax.set_ylabel('Success Rate', fontsize=12)
ax.set_title('(B) Precision-based Funnel Plot\n(higher precision = more reliable)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, df['precision'].max() * 1.1)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'funnel_plot.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: funnel_plot.png")
plt.close()

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: OVERDISPERSION METRICS")
print("=" * 70)

print(f"\n1. Variance ratio (observed/expected): {observed_var / expected_var_binomial:.4f}")
print(f"2. Dispersion parameter (φ): {phi_hat:.4f}")
print(f"3. Quasi-likelihood dispersion: {quasi_dispersion:.4f}")
print(f"4. Chi-squared test p-value: {p_value_chi:.4f}")
print(f"5. Proportion variance from overdispersion: {excess_var / observed_var * 100:.1f}%")

print("\nConclusion:")
if phi_hat > 1.5 or observed_var / expected_var_binomial > 2:
    print("  >>> STRONG evidence of OVERDISPERSION")
    print("  >>> Groups vary more than expected under binomial model")
    print("  >>> Consider beta-binomial or mixed-effects models")
elif phi_hat > 1.2:
    print("  >>> MODERATE evidence of overdispersion")
    print("  >>> Some extra-binomial variation present")
else:
    print("  >>> Minimal overdispersion")
    print("  >>> Binomial model may be adequate")
