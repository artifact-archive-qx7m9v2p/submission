"""
Statistical Tests and Hypothesis Testing - Analyst 2
Testing competing hypotheses about data structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("=" * 80)
print("STATISTICAL HYPOTHESIS TESTING")
print("=" * 80)

# ============================================================================
# HYPOTHESIS 1: Sequential/Temporal Trend
# ============================================================================
print("\n" + "=" * 80)
print("H1: Is there a sequential/temporal trend in proportions?")
print("=" * 80)

# Test for linear trend
correlation_pearson, p_pearson = pearsonr(data['group'], data['proportion'])
correlation_spearman, p_spearman = spearmanr(data['group'], data['proportion'])

print(f"\nPearson correlation (group vs proportion): r = {correlation_pearson:.4f}, p = {p_pearson:.4f}")
print(f"Spearman correlation (group vs proportion): rho = {correlation_spearman:.4f}, p = {p_spearman:.4f}")

# Linear regression
from scipy.stats import linregress
slope, intercept, r_value, p_value_regression, std_err = linregress(data['group'], data['proportion'])
print(f"\nLinear regression: slope = {slope:.6f}, p = {p_value_regression:.4f}")
print(f"R² = {r_value**2:.4f}")

# Runs test for randomness
median_prop = data['proportion'].median()
runs = []
current_run = data['proportion'].iloc[0] > median_prop
run_length = 1
for i in range(1, len(data)):
    above = data['proportion'].iloc[i] > median_prop
    if above == current_run:
        run_length += 1
    else:
        runs.append(run_length)
        current_run = above
        run_length = 1
runs.append(run_length)
n_runs = len(runs)
print(f"\nRuns test: {n_runs} runs (random sequence would have ~{len(data)/2:.0f} runs)")

print("\nConclusion H1:", end=" ")
if p_spearman > 0.10:
    print("NO significant sequential trend detected")
else:
    print("SIGNIFICANT sequential trend detected")

# ============================================================================
# HYPOTHESIS 2: Sample Size Correlation with Proportions
# ============================================================================
print("\n" + "=" * 80)
print("H2: Is sample size correlated with observed proportions?")
print("=" * 80)

correlation_n_p_pearson, p_n_p_pearson = pearsonr(data['n'], data['proportion'])
correlation_n_p_spearman, p_n_p_spearman = spearmanr(data['n'], data['proportion'])

print(f"\nPearson correlation (n vs proportion): r = {correlation_n_p_pearson:.4f}, p = {p_n_p_pearson:.4f}")
print(f"Spearman correlation (n vs proportion): rho = {correlation_n_p_spearman:.4f}, p = {p_n_p_spearman:.4f}")

# Test if small vs large samples have different proportions
median_n = data['n'].median()
small_n = data[data['n'] <= median_n]['proportion']
large_n = data[data['n'] > median_n]['proportion']

u_stat, p_u = stats.mannwhitneyu(small_n, large_n, alternative='two-sided')
print(f"\nMann-Whitney U test (small vs large n):")
print(f"  Small n (≤{median_n:.0f}): mean proportion = {small_n.mean():.4f}")
print(f"  Large n (>{median_n:.0f}): mean proportion = {large_n.mean():.4f}")
print(f"  U-statistic = {u_stat:.2f}, p = {p_u:.4f}")

print("\nConclusion H2:", end=" ")
if p_n_p_spearman > 0.10:
    print("NO significant correlation between sample size and proportion")
else:
    print("SIGNIFICANT correlation detected")

# ============================================================================
# HYPOTHESIS 3: Homogeneity Test (Chi-square)
# ============================================================================
print("\n" + "=" * 80)
print("H3: Are groups exchangeable (homogeneous)?")
print("=" * 80)

# Chi-square test for homogeneity
observed = np.array([data['r'].values, data['failures'].values])
chi2, p_chi2, dof, expected = chi2_contingency(observed)

print(f"\nChi-square test for homogeneity:")
print(f"  Chi² = {chi2:.2f}, df = {dof}, p = {p_chi2:.4e}")
print(f"  If p < 0.05, reject homogeneity (groups differ)")

# Calculate overdispersion parameter
pooled_p = data['r'].sum() / data['n'].sum()
expected_chi2 = len(data) - 1  # df for homogeneity test
dispersion_param = chi2 / expected_chi2
print(f"\nOverdispersion parameter: φ = {dispersion_param:.4f}")
print(f"  φ = 1: consistent with binomial")
print(f"  φ > 1: overdispersed (more variation than expected)")
print(f"  φ < 1: underdispersed (less variation than expected)")

# Likelihood ratio test
from scipy.special import gammaln
def binomial_loglik(r, n, p):
    return np.sum(gammaln(n+1) - gammaln(r+1) - gammaln(n-r+1) + r*np.log(p+1e-10) + (n-r)*np.log(1-p+1e-10))

# Null model: complete pooling
ll_null = binomial_loglik(data['r'].values, data['n'].values, np.array([pooled_p]*len(data)))

# Alternative model: no pooling (saturated)
ll_alt = binomial_loglik(data['r'].values, data['n'].values, data['proportion'].values)

lr_stat = 2 * (ll_alt - ll_null)
lr_p = 1 - stats.chi2.cdf(lr_stat, df=len(data)-1)

print(f"\nLikelihood ratio test:")
print(f"  LR statistic = {lr_stat:.2f}, df = {len(data)-1}, p = {lr_p:.4e}")

print("\nConclusion H3:", end=" ")
if p_chi2 < 0.05:
    print("Groups are NOT homogeneous - hierarchical pooling recommended")
else:
    print("Groups may be homogeneous - complete pooling could be appropriate")

# ============================================================================
# HYPOTHESIS 4: Between vs Within Group Variance
# ============================================================================
print("\n" + "=" * 80)
print("H4: Is heterogeneity due to sampling variation or true differences?")
print("=" * 80)

# Calculate variance components (excluding zero group)
data_nonzero = data[data['r'] > 0]
pooled_p = data['r'].sum() / data['n'].sum()

# Within-group variance (expected under binomial)
within_var = np.average(data_nonzero['proportion'] * (1 - data_nonzero['proportion']) / data_nonzero['n'],
                        weights=data_nonzero['n'])

# Total variance (observed)
total_var = np.average((data_nonzero['proportion'] - np.average(data_nonzero['proportion'], weights=data_nonzero['n']))**2,
                       weights=data_nonzero['n'])

# Between-group variance
between_var = max(0, total_var - within_var)

print(f"\nVariance decomposition (excluding zero group):")
print(f"  Within-group variance (expected): {within_var:.6f}")
print(f"  Total variance (observed): {total_var:.6f}")
print(f"  Between-group variance: {between_var:.6f}")
print(f"  Intraclass correlation (ICC): {between_var/(between_var+within_var):.4f}")

# Calculate I² statistic (proportion of variance due to heterogeneity)
Q = chi2
I_squared = max(0, (Q - dof) / Q) * 100
print(f"\nI² statistic: {I_squared:.1f}%")
print(f"  I² < 25%: low heterogeneity")
print(f"  I² 25-75%: moderate heterogeneity")
print(f"  I² > 75%: high heterogeneity")

print("\nConclusion H4:", end=" ")
if between_var > within_var:
    print("Substantial between-group variation - partial pooling appropriate")
else:
    print("Variation mostly within-group - complete pooling may suffice")

# ============================================================================
# Additional Test: Outlier Detection
# ============================================================================
print("\n" + "=" * 80)
print("OUTLIER DETECTION")
print("=" * 80)

# Calculate standardized residuals
data['expected_r'] = data['n'] * pooled_p
data['std_residual'] = (data['r'] - data['expected_r']) / np.sqrt(data['expected_r'] * (1 - pooled_p))

print("\nStandardized residuals (|z| > 2 indicates outlier):")
for idx, row in data.iterrows():
    flag = "***" if abs(row['std_residual']) > 2 else ""
    print(f"  Group {row['group']}: z = {row['std_residual']:6.2f} {flag}")

outliers = data[abs(data['std_residual']) > 2]
print(f"\nNumber of outliers: {len(outliers)}")
if len(outliers) > 0:
    print("Outlier groups:", outliers['group'].tolist())

# ============================================================================
# SUMMARY STATISTICS FOR MODELING
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS FOR MODELING")
print("=" * 80)

# Calculate Wilson CI for pooled proportion
def wilson_ci_single(r, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = r / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
    return center - margin, center + margin

ci_low, ci_high = wilson_ci_single(data['r'].sum(), data['n'].sum())
print(f"\nPooled estimate: {pooled_p:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
print(f"Harmonic mean of n: {len(data) / np.sum(1/data['n']):.1f}")
print(f"Effective number of groups: ~{len(data)}")
print(f"Zero-event groups: {len(data[data['r'] == 0])}")
print(f"Rare-event groups (r ≤ 5): {len(data[data['r'] <= 5])}")

print("\n" + "=" * 80)
print("HYPOTHESIS TESTING COMPLETE")
print("=" * 80)
