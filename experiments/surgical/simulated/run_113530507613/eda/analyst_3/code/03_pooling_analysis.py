"""
Pooling vs No-Pooling Analysis
Key question: Should we pool all groups or model separately?
Focus: Quantifying between-group vs within-group variability
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

print("="*80)
print("POOLING VS NO-POOLING ANALYSIS")
print("="*80)

# 1. COMPLETE POOLING ESTIMATE
print("\n1. COMPLETE POOLING (All groups combined)")
print("-" * 80)
total_trials = data['n_trials'].sum()
total_successes = data['r_successes'].sum()
pooled_rate = total_successes / total_trials
pooled_se = np.sqrt(pooled_rate * (1 - pooled_rate) / total_trials)

print(f"Total trials: {total_trials}")
print(f"Total successes: {total_successes}")
print(f"Pooled success rate: {pooled_rate:.6f}")
print(f"Standard error: {pooled_se:.6f}")
print(f"95% CI: [{pooled_rate - 1.96*pooled_se:.6f}, {pooled_rate + 1.96*pooled_se:.6f}]")

# 2. NO POOLING ESTIMATES
print("\n2. NO POOLING (Each group separate)")
print("-" * 80)
print("Group-specific rates:")
for idx, row in data.iterrows():
    rate = row['success_rate']
    se = np.sqrt(rate * (1 - rate) / row['n_trials'])
    print(f"  Group {int(row['group_id']):2d}: {rate:.6f} (SE={se:.6f}, n={int(row['n_trials']):3d})")

# 3. VARIABILITY MEASURES
print("\n3. BETWEEN-GROUP VARIABILITY")
print("-" * 80)

# Simple statistics
mean_rate = data['success_rate'].mean()
std_rate = data['success_rate'].std()
cv_rate = std_rate / mean_rate

print(f"Mean of group rates: {mean_rate:.6f}")
print(f"SD of group rates: {std_rate:.6f}")
print(f"Coefficient of variation: {cv_rate:.3f}")
print(f"Range: [{data['success_rate'].min():.6f}, {data['success_rate'].max():.6f}]")
print(f"IQR: {data['success_rate'].quantile(0.75) - data['success_rate'].quantile(0.25):.6f}")

# Ratio of max to min
ratio_max_min = data['success_rate'].max() / data['success_rate'].min()
print(f"Ratio (max/min): {ratio_max_min:.2f}x")

# 4. CHI-SQUARE TEST FOR HOMOGENEITY
print("\n4. STATISTICAL TEST FOR HETEROGENEITY")
print("-" * 80)

# Chi-square test for homogeneity of proportions
observed_successes = data['r_successes'].values
observed_failures = (data['n_trials'] - data['r_successes']).values
expected_successes = data['n_trials'] * pooled_rate
expected_failures = data['n_trials'] * (1 - pooled_rate)

chi2_stat = np.sum((observed_successes - expected_successes)**2 / expected_successes +
                   (observed_failures - expected_failures)**2 / expected_failures)
df = len(data) - 1
p_value = 1 - stats.chi2.cdf(chi2_stat, df)

print(f"Chi-square test for homogeneity:")
print(f"  Chi-square statistic: {chi2_stat:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  P-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"  Result: REJECT null hypothesis (groups are heterogeneous)")
else:
    print(f"  Result: FAIL TO REJECT null hypothesis (groups may be homogeneous)")

# 5. OVERDISPERSION CHECK
print("\n5. OVERDISPERSION ANALYSIS")
print("-" * 80)

# Expected variance under binomial (pooled model)
expected_var_binomial = pooled_rate * (1 - pooled_rate)
print(f"Expected variance (binomial): {expected_var_binomial:.6f}")

# Observed variance across groups
observed_var = data['success_rate'].var(ddof=1)
print(f"Observed variance (between groups): {observed_var:.6f}")

# Variance ratio (overdispersion parameter)
var_ratio = observed_var / expected_var_binomial
print(f"Variance ratio (observed/expected): {var_ratio:.3f}")

if var_ratio > 1.5:
    print(f"  Interpretation: SUBSTANTIAL overdispersion detected")
    print(f"  Implication: Groups are more variable than expected under pooling")
elif var_ratio > 1.1:
    print(f"  Interpretation: MODERATE overdispersion detected")
else:
    print(f"  Interpretation: No substantial overdispersion")

# 6. EFFECTIVE SAMPLE SIZE
print("\n6. EFFECTIVE SAMPLE SIZE UNDER PARTIAL POOLING")
print("-" * 80)

# Estimate between-group variance (tau^2)
# Using method of moments
weights = data['n_trials']
weighted_mean = np.average(data['success_rate'], weights=weights)
weighted_var = np.average((data['success_rate'] - weighted_mean)**2, weights=weights)

# Within-group variance (approximate)
within_var = np.mean([r['success_rate'] * (1 - r['success_rate']) / r['n_trials']
                      for _, r in data.iterrows()])

# Between-group variance
between_var = max(0, weighted_var - within_var)

print(f"Weighted mean rate: {weighted_mean:.6f}")
print(f"Within-group variance (avg): {within_var:.6f}")
print(f"Between-group variance (tau^2): {between_var:.6f}")
print(f"Total variance: {weighted_var:.6f}")

# Intraclass correlation (ICC)
if weighted_var > 0:
    icc = between_var / weighted_var
    print(f"\nIntraclass correlation (ICC): {icc:.3f}")
    print(f"  Interpretation: {icc*100:.1f}% of variance is between groups")

    if icc > 0.3:
        print(f"  Recommendation: HIERARCHICAL/PARTIAL POOLING strongly recommended")
    elif icc > 0.1:
        print(f"  Recommendation: PARTIAL POOLING recommended")
    else:
        print(f"  Recommendation: COMPLETE POOLING may be reasonable")
else:
    print("\nICC: Cannot calculate (zero variance)")

# 7. SHRINKAGE ESTIMATION
print("\n7. SIMPLE SHRINKAGE ESTIMATES")
print("-" * 80)

# Simple empirical Bayes shrinkage
# Shrinkage factor based on sample size and between-group variance
if between_var > 0:
    shrinkage_factors = []
    for idx, row in data.iterrows():
        n = row['n_trials']
        # Approximate shrinkage factor
        lambda_shrink = between_var / (between_var + pooled_rate * (1 - pooled_rate) / n)
        shrinkage_factors.append(lambda_shrink)

        # Shrinkage estimate
        shrunk_rate = lambda_shrink * row['success_rate'] + (1 - lambda_shrink) * pooled_rate

        print(f"  Group {int(row['group_id']):2d}: Raw={row['success_rate']:.6f}, " +
              f"Shrinkage={lambda_shrink:.3f}, Shrunk={shrunk_rate:.6f}")

    data['shrinkage_factor'] = shrinkage_factors
    print(f"\nMean shrinkage factor: {np.mean(shrinkage_factors):.3f}")
    print(f"  (0 = complete shrinkage to pooled, 1 = no shrinkage)")
else:
    print("Between-group variance near zero - complete pooling appropriate")

# 8. PREDICTIVE PERSPECTIVE
print("\n8. PREDICTIVE ACCURACY COMPARISON")
print("-" * 80)

# LOO-CV style: predict each group using pooled estimate without it
loo_errors_pooled = []
loo_errors_individual = []

for idx, row in data.iterrows():
    # True rate
    true_rate = row['success_rate']

    # Pooled prediction (excluding this group)
    other_data = data[data['group_id'] != row['group_id']]
    pooled_pred = other_data['r_successes'].sum() / other_data['n_trials'].sum()

    # Individual prediction (MLE)
    individual_pred = row['success_rate']

    # Squared errors
    loo_errors_pooled.append((true_rate - pooled_pred)**2)
    loo_errors_individual.append(0)  # By definition, MLE has zero error on itself

print(f"Leave-one-out pooled MSE: {np.mean(loo_errors_pooled):.8f}")
print(f"Individual estimates MSE (in-sample): 0.00000000 (by definition)")
print(f"\nNote: Individual estimates have zero in-sample error but may overfit.")
print(f"Pooled estimates have nonzero error but may generalize better.")

print("\n" + "="*80)
print("POOLING ANALYSIS COMPLETE")
print("="*80)
