"""
Variance Decomposition and Hierarchical Structure Testing
Key questions:
1. Is between-group variance greater than expected from binomial sampling alone?
2. What is the intraclass correlation?
3. How much shrinkage would hierarchical modeling provide?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Setup
BASE_DIR = Path("/workspace/eda/analyst_2")
VIZ_DIR = BASE_DIR / "visualizations"
df = pd.read_csv(BASE_DIR / "code" / "group_data_with_ci.csv")

pooled_rate = df['r_successes'].sum() / df['n_trials'].sum()

print("=" * 80)
print("VARIANCE DECOMPOSITION ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. Test for overdispersion relative to binomial
# ============================================================================
print("\n" + "=" * 80)
print("1. OVERDISPERSION TEST")
print("=" * 80)
print("\nUnder a simple binomial model with common probability p,")
print("we expect variance in observed rates to be purely due to sampling.")
print("We test if observed variance exceeds this expectation.\n")

# Expected variance under binomial (assuming common p = pooled rate)
# For each group: Var(p_hat) = p(1-p)/n
expected_variances = pooled_rate * (1 - pooled_rate) / df['n_trials']
print(f"Pooled rate (p): {pooled_rate:.6f}")

# Observed variance in success rates
observed_variance = df['success_rate'].var(ddof=1)
print(f"\nObserved variance in success rates: {observed_variance:.8f}")

# Expected variance (weighted average of expected variances)
total_trials = df['n_trials'].sum()
weights = df['n_trials'] / total_trials
expected_variance_weighted = (weights * expected_variances).sum()
print(f"Expected variance under binomial (weighted): {expected_variance_weighted:.8f}")

# Variance ratio
variance_ratio = observed_variance / expected_variance_weighted
print(f"\nVariance ratio (observed/expected): {variance_ratio:.2f}")

if variance_ratio > 1:
    print(f"  → OVERDISPERSED: Variance is {variance_ratio:.1f}x what we'd expect from sampling alone")
    print("  → This suggests genuine between-group heterogeneity")
else:
    print("  → Variance is consistent with binomial sampling variation")

# Chi-square test for homogeneity
print("\n" + "=" * 80)
print("2. CHI-SQUARE TEST FOR HOMOGENEITY")
print("=" * 80)

# Pearson chi-square statistic
observed_successes = df['r_successes'].values
observed_failures = df['n_trials'].values - df['r_successes'].values
expected_successes = df['n_trials'].values * pooled_rate
expected_failures = df['n_trials'].values * (1 - pooled_rate)

chi_square_stat = np.sum((observed_successes - expected_successes)**2 / expected_successes +
                         (observed_failures - expected_failures)**2 / expected_failures)

df_chi = len(df) - 1  # degrees of freedom
p_value = 1 - stats.chi2.cdf(chi_square_stat, df_chi)

print(f"Null hypothesis: All groups have the same underlying success probability")
print(f"Chi-square statistic: {chi_square_stat:.2f}")
print(f"Degrees of freedom: {df_chi}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print(f"  → REJECT null hypothesis (p < 0.05)")
    print(f"  → Evidence for heterogeneity across groups")
else:
    print(f"  → FAIL TO REJECT null hypothesis (p >= 0.05)")
    print(f"  → Insufficient evidence for heterogeneity")

# ============================================================================
# 3. Estimate between-group variance component
# ============================================================================
print("\n" + "=" * 80)
print("3. BETWEEN-GROUP VARIANCE COMPONENT")
print("=" * 80)
print("\nUsing method of moments to estimate variance components:")

# Total variance decomposition
# Var_total = Var_between + E[Var_within]
# Var_within for binomial = p(1-p)/n

# Sample variance of observed rates
var_between_crude = df['success_rate'].var(ddof=1)

# Expected within-group variance (average of binomial variances)
within_group_var = np.mean(df['success_rate'] * (1 - df['success_rate']) / df['n_trials'])

# Adjusted between-group variance
# Using DerSimonian-Laird estimator for tau-squared
weights_dl = 1 / (pooled_rate * (1 - pooled_rate) / df['n_trials'])
weights_sum = weights_dl.sum()
Q = np.sum(weights_dl * (df['success_rate'] - pooled_rate)**2)
C = weights_sum - (weights_dl**2).sum() / weights_sum

tau_squared = max(0, (Q - (len(df) - 1)) / C)

print(f"Crude between-group variance: {var_between_crude:.8f}")
print(f"Expected within-group variance (avg): {within_group_var:.8f}")
print(f"DerSimonian-Laird tau-squared: {tau_squared:.8f}")
print(f"DerSimonian-Laird tau (SD): {np.sqrt(tau_squared):.6f}")

# ============================================================================
# 4. Intraclass Correlation Coefficient (ICC)
# ============================================================================
print("\n" + "=" * 80)
print("4. INTRACLASS CORRELATION COEFFICIENT (ICC)")
print("=" * 80)
print("\nICC measures proportion of total variance due to between-group differences")
print("ICC = Var_between / (Var_between + Var_within)")

# For binomial data, approximate ICC
# Using the logit scale is more appropriate, but we'll compute on probability scale
var_between = tau_squared
var_within = pooled_rate * (1 - pooled_rate) / df['n_trials'].mean()

icc_approx = var_between / (var_between + var_within)
print(f"\nApproximate ICC (probability scale): {icc_approx:.4f}")
print(f"  → {icc_approx*100:.1f}% of variance is between-group")
print(f"  → {(1-icc_approx)*100:.1f}% of variance is within-group (sampling)")

if icc_approx > 0.1:
    print("\n  → ICC > 0.1: Substantial between-group variation")
    print("  → Hierarchical modeling likely beneficial")
elif icc_approx > 0.05:
    print("\n  → ICC > 0.05: Moderate between-group variation")
    print("  → Hierarchical modeling may be beneficial")
else:
    print("\n  → ICC < 0.05: Minimal between-group variation")
    print("  → Simple pooled model may suffice")

# ============================================================================
# 5. Shrinkage Factor Analysis
# ============================================================================
print("\n" + "=" * 80)
print("5. SHRINKAGE FACTOR ANALYSIS")
print("=" * 80)
print("\nShrinkage factor λ = n / (n + 1/τ²)")
print("where n = group sample size, τ² = between-group variance")
print("\nLarger λ → more weight on group data")
print("Smaller λ → more shrinkage toward overall mean\n")

if tau_squared > 0:
    shrinkage_factors = df['n_trials'] / (df['n_trials'] + 1/tau_squared)
else:
    shrinkage_factors = np.zeros(len(df))

df['shrinkage_factor'] = shrinkage_factors
df['shrinkage_pct'] = (1 - shrinkage_factors) * 100

# Partial pooling estimates (simple approximation)
df['partial_pooled_rate'] = (shrinkage_factors * df['success_rate'] +
                              (1 - shrinkage_factors) * pooled_rate)

print("Shrinkage by group:")
print(df[['group', 'n_trials', 'success_rate', 'shrinkage_factor',
          'shrinkage_pct', 'partial_pooled_rate']].to_string(index=False))

print(f"\nShrinkage factor statistics:")
print(f"  Mean: {shrinkage_factors.mean():.4f}")
print(f"  Min: {shrinkage_factors.min():.4f} (Group {df.loc[shrinkage_factors.idxmin(), 'group']})")
print(f"  Max: {shrinkage_factors.max():.4f} (Group {df.loc[shrinkage_factors.idxmax(), 'group']})")

print(f"\nAverage shrinkage: {(1-shrinkage_factors.mean())*100:.1f}% toward pooled mean")

# ============================================================================
# 6. Visual Comparison: No pooling vs Complete pooling vs Partial pooling
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left panel: Three modeling approaches
y_pos = range(len(df))

# No pooling (observed rates)
ax1.scatter(df['success_rate'], y_pos, s=100, color='darkblue',
           alpha=0.7, label='No pooling (observed)', zorder=5, marker='o')

# Complete pooling
ax1.scatter([pooled_rate]*len(df), y_pos, s=100, color='red',
           alpha=0.7, label='Complete pooling', zorder=5, marker='s')

# Partial pooling (hierarchical)
ax1.scatter(df['partial_pooled_rate'], y_pos, s=100, color='green',
           alpha=0.7, label='Partial pooling (hierarchical)', zorder=5, marker='^')

# Connect with lines
for i in range(len(df)):
    ax1.plot([df.iloc[i]['success_rate'], df.iloc[i]['partial_pooled_rate'], pooled_rate],
            [i, i, i], color='gray', alpha=0.3, linestyle='--', linewidth=1)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"Group {g}" for g in df['group']])
ax1.set_xlabel('Success Rate', fontsize=12)
ax1.set_ylabel('Group', fontsize=12)
ax1.set_title('Three Pooling Strategies Compared', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Right panel: Shrinkage as a function of sample size
ax2.scatter(df['n_trials'], df['shrinkage_factor'], s=100, color='purple', alpha=0.7)
for idx, row in df.iterrows():
    ax2.annotate(f"G{int(row['group'])}",
                (row['n_trials'], row['shrinkage_factor']),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

ax2.set_xlabel('Sample Size (n_trials)', fontsize=12)
ax2.set_ylabel('Shrinkage Factor λ', fontsize=12)
ax2.set_title('Shrinkage Factor vs Sample Size\n(larger samples → less shrinkage)',
             fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(VIZ_DIR / "pooling_comparison.png", dpi=150, bbox_inches='tight')
print(f"\n\nSaved: {VIZ_DIR / 'pooling_comparison.png'}")
plt.close()

# Save augmented data
df.to_csv(BASE_DIR / "code" / "hierarchical_analysis.csv", index=False)
print(f"Saved: {BASE_DIR / 'code' / 'hierarchical_analysis.csv'}")
