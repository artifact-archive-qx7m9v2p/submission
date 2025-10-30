"""
Comprehensive EDA for Binomial Data - Analyst #2
Focus: Pooling assessment, hierarchical structure, and prior elicitation

Dataset: 12 groups with binomial data (n trials, r successes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import betaln
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*80)
print("BINOMIAL DATA EDA - ANALYST #2")
print("="*80)
print(f"\nDataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nData summary:")
print(data.describe())

# Calculate basic statistics for each group
data['p_hat'] = data['r'] / data['n']  # Observed success rate
data['failures'] = data['n'] - data['r']

# Wilson score interval (better than normal approximation for extreme probabilities)
def wilson_ci(r, n, alpha=0.05):
    """Calculate Wilson score confidence interval"""
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = r / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denominator
    return center - margin, center + margin

data['ci_lower'], data['ci_upper'] = zip(*data.apply(lambda x: wilson_ci(x['r'], x['n']), axis=1))
data['ci_width'] = data['ci_upper'] - data['ci_lower']

print(f"\n{'='*80}")
print("GROUP-LEVEL STATISTICS")
print("="*80)
print(data[['group', 'n', 'r', 'p_hat', 'ci_lower', 'ci_upper', 'ci_width']].to_string(index=False))

# =============================================================================
# 1. POOLING ASSESSMENT
# =============================================================================
print(f"\n{'='*80}")
print("1. POOLING ASSESSMENT")
print("="*80)

# Completely pooled estimate
total_trials = data['n'].sum()
total_successes = data['r'].sum()
pooled_rate = total_successes / total_trials

print(f"\nCompletely Pooled Model:")
print(f"  Total trials: {total_trials}")
print(f"  Total successes: {total_successes}")
print(f"  Pooled success rate: {pooled_rate:.4f}")

# Pooled confidence interval
pooled_ci_lower, pooled_ci_upper = wilson_ci(total_successes, total_trials)
print(f"  95% CI: [{pooled_ci_lower:.4f}, {pooled_ci_upper:.4f}]")

# Completely unpooled estimates
print(f"\nCompletely Unpooled Model:")
print(f"  Individual group rates range: [{data['p_hat'].min():.4f}, {data['p_hat'].max():.4f}]")
print(f"  Mean of individual rates: {data['p_hat'].mean():.4f}")
print(f"  Std of individual rates: {data['p_hat'].std():.4f}")
print(f"  Coefficient of variation: {data['p_hat'].std() / data['p_hat'].mean():.4f}")

# Calculate deviation from pooled rate
data['deviation_from_pooled'] = data['p_hat'] - pooled_rate
data['abs_deviation'] = np.abs(data['deviation_from_pooled'])
data['relative_deviation'] = data['deviation_from_pooled'] / pooled_rate

print(f"\nDeviation from Pooled Rate:")
print(f"  Mean absolute deviation: {data['abs_deviation'].mean():.4f}")
print(f"  Max deviation: {data['abs_deviation'].max():.4f} (Group {data.loc[data['abs_deviation'].idxmax(), 'group']})")
print(f"  Groups above pooled rate: {(data['p_hat'] > pooled_rate).sum()}")
print(f"  Groups below pooled rate: {(data['p_hat'] < pooled_rate).sum()}")

# Test for heterogeneity
# Chi-square test for homogeneity
observed = data[['r', 'failures']].values
chi2, p_value, dof, expected = stats.chi2_contingency(observed.T)
print(f"\nChi-square test for homogeneity:")
print(f"  Chi-square statistic: {chi2:.2f}")
print(f"  Degrees of freedom: {dof}")
print(f"  P-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"  Conclusion: Strong evidence for heterogeneity (reject null of equal rates)")
else:
    print(f"  Conclusion: Insufficient evidence for heterogeneity")

# =============================================================================
# 2. HIERARCHICAL STRUCTURE EVIDENCE
# =============================================================================
print(f"\n{'='*80}")
print("2. HIERARCHICAL STRUCTURE EVIDENCE")
print("="*80)

# Transform to logit scale for better normality
data['logit_p'] = np.log(data['p_hat'] / (1 - data['p_hat']))

# Estimate variance on logit scale (approximation for hierarchical variance)
# Standard error on logit scale
data['se_logit'] = np.sqrt(1/data['r'] + 1/(data['n'] - data['r']))

# Empirical Bayes estimate of between-group variance (tau^2)
# Using method of moments
sample_var = data['logit_p'].var()
mean_within_var = (data['se_logit']**2).mean()
tau_sq_logit = max(0, sample_var - mean_within_var)
tau_logit = np.sqrt(tau_sq_logit)

print(f"\nVariance Components (Logit Scale):")
print(f"  Total variance in logit(p): {sample_var:.4f}")
print(f"  Mean within-group variance: {mean_within_var:.4f}")
print(f"  Estimated between-group variance (tau^2): {tau_sq_logit:.4f}")
print(f"  Estimated tau: {tau_logit:.4f}")

# Intraclass correlation approximation
icc = tau_sq_logit / (tau_sq_logit + mean_within_var) if (tau_sq_logit + mean_within_var) > 0 else 0
print(f"  Approximate ICC: {icc:.4f}")

if icc > 0.1:
    print(f"  Interpretation: Moderate to high between-group variation - hierarchical model recommended")
elif icc > 0.05:
    print(f"  Interpretation: Some between-group variation - hierarchical model may be beneficial")
else:
    print(f"  Interpretation: Low between-group variation - pooling may be appropriate")

# Calculate shrinkage factors
# Shrinkage = tau^2 / (tau^2 + sigma_i^2)
data['shrinkage_factor'] = tau_sq_logit / (tau_sq_logit + data['se_logit']**2)
data['shrinkage_pct'] = (1 - data['shrinkage_factor']) * 100

print(f"\nShrinkage Analysis:")
print(f"  Mean shrinkage toward group mean: {data['shrinkage_pct'].mean():.1f}%")
print(f"  Range of shrinkage: [{data['shrinkage_pct'].min():.1f}%, {data['shrinkage_pct'].max():.1f}%]")
print(f"\n  Groups by shrinkage (highest to lowest):")
print(data[['group', 'n', 'r', 'p_hat', 'shrinkage_pct']].sort_values('shrinkage_pct', ascending=False).to_string(index=False))

# Estimate partially pooled rates (simple empirical Bayes)
grand_mean_logit = data['logit_p'].mean()
data['logit_p_pooled'] = (data['shrinkage_factor'] * data['logit_p'] +
                           (1 - data['shrinkage_factor']) * grand_mean_logit)
data['p_hat_pooled'] = 1 / (1 + np.exp(-data['logit_p_pooled']))
data['change_from_pooling'] = data['p_hat_pooled'] - data['p_hat']

print(f"\nEffect of Partial Pooling:")
print(f"  Mean absolute change in rates: {np.abs(data['change_from_pooling']).mean():.4f}")
print(f"  Max change: {np.abs(data['change_from_pooling']).max():.4f} (Group {data.loc[np.abs(data['change_from_pooling']).idxmax(), 'group']})")

# =============================================================================
# 3. PRIOR ELICITATION INSIGHTS
# =============================================================================
print(f"\n{'='*80}")
print("3. PRIOR ELICITATION INSIGHTS")
print("="*80)

print(f"\nSuccess Rate Prior (p ~ Beta(alpha, beta)):")
print(f"  Observed rate range: [{data['p_hat'].min():.4f}, {data['p_hat'].max():.4f}]")
print(f"  Median rate: {data['p_hat'].median():.4f}")
print(f"  IQR: [{data['p_hat'].quantile(0.25):.4f}, {data['p_hat'].quantile(0.75):.4f}]")

# Suggest weakly informative priors
print(f"\n  Weakly Informative Prior Suggestions:")
print(f"    Option 1 - Uniform: Beta(1, 1) [completely flat]")
print(f"    Option 2 - Jeffreys: Beta(0.5, 0.5) [non-informative]")
print(f"    Option 3 - Weak: Beta(2, 2) [mild peak at 0.5]")
print(f"    Option 4 - Data-informed: Beta(5, 50) [mean ~ {5/(5+50):.3f}]")

# Method of moments to match observed data
mean_p = data['p_hat'].mean()
var_p = data['p_hat'].var()
# Beta distribution: mean = a/(a+b), var = ab/((a+b)^2(a+b+1))
# Solving for a, b
if var_p < mean_p * (1 - mean_p):
    common_mean = mean_p * (1 - mean_p) / var_p - 1
    alpha_mom = mean_p * common_mean
    beta_mom = (1 - mean_p) * common_mean
    print(f"    Option 5 - Method of Moments: Beta({alpha_mom:.2f}, {beta_mom:.2f})")
    print(f"      [matches empirical mean={mean_p:.3f}, var={var_p:.4f}]")

print(f"\nHierarchical Variance Prior (tau ~ Half-Cauchy or Half-Normal):")
print(f"  Estimated tau (logit scale): {tau_logit:.4f}")
print(f"  Suggested Half-Cauchy scale: {max(1.0, tau_logit * 2):.2f}")
print(f"  Suggested Half-Normal scale: {max(1.0, tau_logit * 1.5):.2f}")

# =============================================================================
# 4. EXTREME GROUPS IDENTIFICATION
# =============================================================================
print(f"\n{'='*80}")
print("4. EXTREME GROUPS")
print("="*80)

# Identify extremes by multiple criteria
data['z_score_rate'] = (data['p_hat'] - data['p_hat'].mean()) / data['p_hat'].std()
data['z_score_n'] = (data['n'] - data['n'].mean()) / data['n'].std()

print(f"\nExtreme Success Rates:")
extreme_rates = data[np.abs(data['z_score_rate']) > 1.5]
if len(extreme_rates) > 0:
    print(extreme_rates[['group', 'n', 'r', 'p_hat', 'z_score_rate']].to_string(index=False))
else:
    print("  No extreme rates (|z| > 1.5)")

print(f"\nExtreme Sample Sizes:")
extreme_n = data[np.abs(data['z_score_n']) > 1.5]
if len(extreme_n) > 0:
    print(extreme_n[['group', 'n', 'r', 'p_hat', 'z_score_n']].to_string(index=False))
else:
    print("  No extreme sample sizes (|z| > 1.5)")

print(f"\nHigh Influence Groups (large n AND extreme rate):")
data['influence_score'] = np.abs(data['z_score_rate']) * (data['n'] / data['n'].sum())
high_influence = data.nlargest(3, 'influence_score')
print(high_influence[['group', 'n', 'r', 'p_hat', 'influence_score']].to_string(index=False))

print(f"\nSmall Sample Groups (potential instability):")
small_n = data[data['n'] < data['n'].quantile(0.25)]
print(small_n[['group', 'n', 'r', 'p_hat', 'ci_width']].to_string(index=False))
print(f"  Note: These groups have wide confidence intervals and may benefit most from pooling")

# =============================================================================
# 5. TEMPORAL/SPATIAL PATTERNS
# =============================================================================
print(f"\n{'='*80}")
print("5. TEMPORAL/SPATIAL PATTERNS")
print("="*80)

# Check for trends in group ordering
correlation_group_rate = stats.spearmanr(data['group'], data['p_hat'])
correlation_group_n = stats.spearmanr(data['group'], data['n'])

print(f"\nCorrelation with Group Number:")
print(f"  Success rate vs group: rho={correlation_group_rate.statistic:.3f}, p={correlation_group_rate.pvalue:.4f}")
print(f"  Sample size vs group: rho={correlation_group_n.statistic:.3f}, p={correlation_group_n.pvalue:.4f}")

if correlation_group_rate.pvalue < 0.05:
    print(f"  Interpretation: Significant trend in success rates across groups")
else:
    print(f"  Interpretation: No significant temporal/ordering trend")

# Run test for runs (randomness) - fixed to use .values
median_p = data['p_hat'].median()
above_median = (data['p_hat'] > median_p).astype(int).values
runs = 1 + sum(above_median[1:] != above_median[:-1])
n_above = sum(above_median)
n_below = len(above_median) - n_above
expected_runs = 1 + 2 * n_above * n_below / len(above_median)
var_runs = 2 * n_above * n_below * (2 * n_above * n_below - len(above_median)) / (len(above_median)**2 * (len(above_median) - 1))
z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0

print(f"\nRuns Test for Randomness:")
print(f"  Number of runs: {runs}")
print(f"  Expected runs: {expected_runs:.1f}")
print(f"  Z-score: {z_runs:.3f}")
if abs(z_runs) < 1.96:
    print(f"  Interpretation: Data appears random (no clustering pattern)")
else:
    print(f"  Interpretation: Non-random pattern detected")

# =============================================================================
# 6. DATA QUALITY CHECKS
# =============================================================================
print(f"\n{'='*80}")
print("6. DATA QUALITY CHECKS")
print("="*80)

print(f"\nMissing Values:")
print(data.isnull().sum())

print(f"\nData Validity:")
valid_trials = (data['n'] > 0).all()
valid_successes = ((data['r'] >= 0) & (data['r'] <= data['n'])).all()
print(f"  All trials > 0: {valid_trials}")
print(f"  All successes valid (0 <= r <= n): {valid_successes}")

print(f"\nPotential Issues:")
issues = []
# Zero successes or failures
zero_success = data[data['r'] == 0]
if len(zero_success) > 0:
    issues.append(f"  - {len(zero_success)} group(s) with zero successes: {zero_success['group'].tolist()}")
zero_failure = data[data['failures'] == 0]
if len(zero_failure) > 0:
    issues.append(f"  - {len(zero_failure)} group(s) with zero failures: {zero_failure['group'].tolist()}")

# Very small sample sizes
very_small = data[data['n'] < 10]
if len(very_small) > 0:
    issues.append(f"  - {len(very_small)} group(s) with n < 10: {very_small['group'].tolist()}")

# Very wide confidence intervals
wide_ci = data[data['ci_width'] > 0.3]
if len(wide_ci) > 0:
    issues.append(f"  - {len(wide_ci)} group(s) with CI width > 0.3: {wide_ci['group'].tolist()}")

if issues:
    for issue in issues:
        print(issue)
    print(f"\n  Recommendation: Consider continuity correction or informative priors for extreme groups")
else:
    print("  No major data quality issues detected")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE - Proceeding to visualizations")
print("="*80)

# Save processed data for plotting
data.to_csv('/workspace/eda/analyst_2/code/processed_data.csv', index=False)
print(f"\nProcessed data saved to: /workspace/eda/analyst_2/code/processed_data.csv")
