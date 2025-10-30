"""
Variance Analysis - Testing for Overdispersion
Focus: Empirical variance vs expected binomial variance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("="*80)
print("VARIANCE STRUCTURE ANALYSIS")
print("="*80)

# Calculate pooled success rate
pooled_p = data['r_successes'].sum() / data['n_trials'].sum()

# 1. EMPIRICAL VARIANCE OF SUCCESS RATES
empirical_var = data['success_rate'].var(ddof=1)
empirical_std = data['success_rate'].std(ddof=1)

print(f"\n1. EMPIRICAL VARIANCE OF SUCCESS RATES")
print(f"Variance: {empirical_var:.8f}")
print(f"Std Dev: {empirical_std:.6f}")
print(f"Mean: {data['success_rate'].mean():.6f}")

# 2. EXPECTED VARIANCE UNDER BINOMIAL MODEL
# For binomial, var(p_hat) = p(1-p)/n
# We need to account for different n's, so we calculate expected variance
# E[Var(p_hat)] = E[p(1-p)/n]

# Method 1: Average expected variance across groups
expected_var_avg = np.mean(pooled_p * (1 - pooled_p) / data['n_trials'])

print(f"\n2. EXPECTED VARIANCE UNDER BINOMIAL (Method 1: Average)")
print(f"Expected variance: {expected_var_avg:.8f}")
print(f"Expected std dev: {np.sqrt(expected_var_avg):.6f}")

# Method 2: Weighted by sample size
weights = data['n_trials'] / data['n_trials'].sum()
expected_var_weighted = np.sum(weights * pooled_p * (1 - pooled_p) / data['n_trials'])

print(f"\n3. EXPECTED VARIANCE (Method 2: Weighted by n)")
print(f"Expected variance: {expected_var_weighted:.8f}")
print(f"Expected std dev: {np.sqrt(expected_var_weighted):.6f}")

# 3. VARIANCE RATIO TEST (Overdispersion Test)
variance_ratio = empirical_var / expected_var_avg
print(f"\n4. VARIANCE RATIO (OVERDISPERSION TEST)")
print(f"Variance ratio (empirical/expected): {variance_ratio:.4f}")
print("")
if variance_ratio > 1.5:
    print("   -> SUBSTANTIAL OVERDISPERSION: Variance much larger than expected")
    print("      Suggests heterogeneity in true success rates")
elif variance_ratio > 1.1:
    print("   -> MODERATE OVERDISPERSION: Some excess variance")
elif variance_ratio < 0.9:
    print("   -> UNDERDISPERSION: Less variance than expected (unusual)")
else:
    print("   -> CONSISTENT WITH BINOMIAL: Variance as expected")

# 4. CHI-SQUARE TEST FOR HOMOGENEITY
# Under H0: all groups have same success rate
# Chi-square = sum((O-E)^2/E) for successes and failures
expected_successes = data['n_trials'] * pooled_p
expected_failures = data['n_trials'] * (1 - pooled_p)

chi_square = np.sum((data['r_successes'] - expected_successes)**2 / expected_successes +
                    ((data['n_trials'] - data['r_successes']) - expected_failures)**2 / expected_failures)

df = len(data) - 1
p_value_chi = 1 - stats.chi2.cdf(chi_square, df)

print(f"\n5. CHI-SQUARE TEST FOR HOMOGENEITY")
print(f"Chi-square statistic: {chi_square:.4f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value_chi:.6f}")
print(f"Critical value (α=0.05): {stats.chi2.ppf(0.95, df):.4f}")

if p_value_chi < 0.05:
    print("   -> REJECT homogeneity: Groups have different success rates")
else:
    print("   -> Cannot reject homogeneity: Consistent with same success rate")

# 5. CALCULATE BETWEEN-GROUP VARIANCE COMPONENT
# Using method of moments estimator
# Var(p_hat) = Var_between + E[Var_within]
# Var_between = Var(p_hat) - E[p(1-p)/n]

var_between = empirical_var - expected_var_avg
print(f"\n6. VARIANCE DECOMPOSITION")
print(f"Total empirical variance: {empirical_var:.8f}")
print(f"Expected within-group variance: {expected_var_avg:.8f}")
print(f"Estimated between-group variance: {var_between:.8f}")

if var_between > 0:
    pct_between = (var_between / empirical_var) * 100
    print(f"Percentage of variance from between-group differences: {pct_between:.1f}%")
    print("   -> Suggests heterogeneity in true success rates")
else:
    print("   -> No evidence of between-group variance")

# 6. COEFFICIENT OF VARIATION COMPARISON
cv_empirical = empirical_std / data['success_rate'].mean()
cv_expected = np.sqrt(expected_var_avg) / pooled_p

print(f"\n7. COEFFICIENT OF VARIATION (CV)")
print(f"Empirical CV: {cv_empirical:.4f}")
print(f"Expected CV under binomial: {cv_expected:.4f}")
print(f"CV ratio: {cv_empirical / cv_expected:.4f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Variance Analysis: Empirical vs Expected Binomial Variance', fontsize=14, fontweight='bold')

# Plot 1: Observed vs Expected Standard Errors
ax = axes[0, 0]
data['expected_se'] = np.sqrt(pooled_p * (1 - pooled_p) / data['n_trials'])
data['observed_deviation'] = data['success_rate'] - pooled_p
data['standardized_deviation'] = data['observed_deviation'] / data['expected_se']

ax.scatter(data['expected_se'], np.abs(data['observed_deviation']),
           s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

# Add 1:1 line (expected relationship)
max_se = data['expected_se'].max()
x_line = np.linspace(0, max_se * 1.2, 100)
ax.plot(x_line, x_line, 'r--', linewidth=2, label='Expected (1:1 line)')
ax.plot(x_line, 1.96 * x_line, 'b--', linewidth=1.5, alpha=0.7, label='95% CI boundary')

for idx, row in data.iterrows():
    ax.annotate(f"{row['group_id']}",
                (row['expected_se'], abs(row['observed_deviation'])),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Expected SE (under binomial)', fontsize=11)
ax.set_ylabel('Observed |Deviation from pooled|', fontsize=11)
ax.set_title('Observed Deviation vs Expected SE', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Standardized Residuals
ax = axes[0, 1]
ax.scatter(data['n_trials'], data['standardized_deviation'],
           s=100, alpha=0.7, edgecolors='black', linewidth=1.5, c=data['standardized_deviation'],
           cmap='RdBu_r', vmin=-4, vmax=4)

ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(1.96, color='red', linestyle='--', linewidth=1.5, label='95% CI')
ax.axhline(-1.96, color='red', linestyle='--', linewidth=1.5)
ax.axhline(2.576, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='99% CI')
ax.axhline(-2.576, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)

for idx, row in data.iterrows():
    ax.annotate(f"{row['group_id']}",
                (row['n_trials'], row['standardized_deviation']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Number of Trials', fontsize=11)
ax.set_ylabel('Standardized Deviation (z-score)', fontsize=11)
ax.set_title('Standardized Residuals vs Sample Size', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Variance by Group
ax = axes[1, 0]
data_sorted = data.sort_values('n_trials')
x_pos = np.arange(len(data_sorted))

bars = ax.bar(x_pos, data_sorted['expected_se']**2, alpha=0.5, label='Expected Variance', color='blue')
ax.scatter(x_pos, (data_sorted['success_rate'] - pooled_p)**2,
           color='red', s=100, zorder=5, label='Observed Squared Deviation', marker='D')

ax.set_xlabel('Groups (sorted by n_trials)', fontsize=11)
ax.set_ylabel('Variance', fontsize=11)
ax.set_title('Expected Variance vs Observed Squared Deviations', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(data_sorted['group_id'].astype(int), rotation=0)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Q-Q plot of standardized residuals
ax = axes[1, 1]
stats.probplot(data['standardized_deviation'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Standardized Residuals vs Normal', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/04_variance_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: /workspace/eda/analyst_1/visualizations/04_variance_analysis.png")

# Additional tests
print(f"\n8. STANDARDIZED RESIDUALS SUMMARY")
print(f"Mean: {data['standardized_deviation'].mean():.4f}")
print(f"Std Dev: {data['standardized_deviation'].std():.4f}")
print(f"Min: {data['standardized_deviation'].min():.4f}")
print(f"Max: {data['standardized_deviation'].max():.4f}")
print(f"\nGroups with |z| > 1.96: {len(data[np.abs(data['standardized_deviation']) > 1.96])}")
print(f"Groups with |z| > 2.576: {len(data[np.abs(data['standardized_deviation']) > 2.576])}")
print(f"Groups with |z| > 3: {len(data[np.abs(data['standardized_deviation']) > 3])}")

print("\n" + "="*80)
