"""
Hypothesis Testing: Competing Explanations for Observed Patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Calculate pooled success rate and derived metrics upfront
pooled_p = data['r_successes'].sum() / data['n_trials'].sum()
data['expected_se'] = np.sqrt(pooled_p * (1 - pooled_p) / data['n_trials'])
data['z_score'] = (data['success_rate'] - pooled_p) / data['expected_se']

print("="*80)
print("HYPOTHESIS TESTING: COMPETING EXPLANATIONS")
print("="*80)

print("\nWe will test three competing hypotheses:")
print("H1: Homogeneous model - All groups share the same success rate")
print("H2: Sample size dependent - Success rate varies with sample size")
print("H3: Heterogeneous model - Groups have different true success rates")

# HYPOTHESIS 1: HOMOGENEOUS MODEL
print("\n" + "="*80)
print("HYPOTHESIS 1: HOMOGENEOUS MODEL")
print("="*80)
print("All groups share the same success rate (binomial variation only)")

# Chi-square goodness of fit
expected_successes = data['n_trials'] * pooled_p
expected_failures = data['n_trials'] * (1 - pooled_p)
chi_square = np.sum((data['r_successes'] - expected_successes)**2 / expected_successes +
                    ((data['n_trials'] - data['r_successes']) - expected_failures)**2 / expected_failures)
df = len(data) - 1
p_value = 1 - stats.chi2.cdf(chi_square, df)

print(f"\nChi-square test:")
print(f"  Statistic: {chi_square:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Conclusion: {'REJECT H1' if p_value < 0.05 else 'Cannot reject H1'}")

# Calculate AIC/BIC for homogeneous model
# Log-likelihood for binomial
log_lik_h1 = 0
for idx, row in data.iterrows():
    n, r = row['n_trials'], row['r_successes']
    log_lik_h1 += stats.binom.logpmf(r, n, pooled_p)

k_h1 = 1  # 1 parameter: pooled p
aic_h1 = -2 * log_lik_h1 + 2 * k_h1
bic_h1 = -2 * log_lik_h1 + k_h1 * np.log(len(data))

print(f"\nModel fit statistics:")
print(f"  Log-likelihood: {log_lik_h1:.4f}")
print(f"  AIC: {aic_h1:.4f}")
print(f"  BIC: {bic_h1:.4f}")

# Evidence from variance ratio
var_empirical = data['success_rate'].var(ddof=1)
var_expected = np.mean(pooled_p * (1 - pooled_p) / data['n_trials'])
var_ratio = var_empirical / var_expected

print(f"\nVariance evidence:")
print(f"  Variance ratio: {var_ratio:.4f}")
print(f"  Conclusion: {'Overdispersion detected' if var_ratio > 1.5 else 'Consistent with H1'}")

# HYPOTHESIS 2: SAMPLE SIZE DEPENDENT
print("\n" + "="*80)
print("HYPOTHESIS 2: SAMPLE SIZE DEPENDENT")
print("="*80)
print("Success rate varies systematically with sample size")

# Test correlation
corr_pearson, p_pearson = stats.pearsonr(data['n_trials'], data['success_rate'])
corr_spearman, p_spearman = stats.spearmanr(data['n_trials'], data['success_rate'])

print(f"\nCorrelation tests:")
print(f"  Pearson r: {corr_pearson:.4f} (p={p_pearson:.4f})")
print(f"  Spearman rho: {corr_spearman:.4f} (p={p_spearman:.4f})")
print(f"  Conclusion: {'SUPPORT H2' if p_pearson < 0.05 else 'No support for H2'}")

# Linear regression
from scipy.stats import linregress
slope, intercept, r_value, p_value_reg, std_err = linregress(data['n_trials'], data['success_rate'])

print(f"\nLinear regression (success_rate ~ n_trials):")
print(f"  Slope: {slope:.8f} (p={p_value_reg:.4f})")
print(f"  Intercept: {intercept:.4f}")
print(f"  R-squared: {r_value**2:.4f}")

# Divide into small/large n and compare
median_n = data['n_trials'].median()
small_n = data[data['n_trials'] <= median_n]
large_n = data[data['n_trials'] > median_n]

print(f"\nComparison by sample size (median split at n={median_n}):")
print(f"  Small n groups (n <= {median_n}): mean success_rate = {small_n['success_rate'].mean():.4f}")
print(f"  Large n groups (n > {median_n}): mean success_rate = {large_n['success_rate'].mean():.4f}")

if len(small_n) > 1 and len(large_n) > 1:
    t_stat, p_val = stats.ttest_ind(small_n['success_rate'], large_n['success_rate'])
    print(f"  t-test p-value: {p_val:.4f}")
    print(f"  Conclusion: {'Significant difference' if p_val < 0.05 else 'No significant difference'}")

# HYPOTHESIS 3: HETEROGENEOUS MODEL
print("\n" + "="*80)
print("HYPOTHESIS 3: HETEROGENEOUS MODEL")
print("="*80)
print("Groups have different true success rates (not explained by sample size)")

# Calculate log-likelihood for saturated model (each group has own rate)
log_lik_h3 = 0
for idx, row in data.iterrows():
    n, r = row['n_trials'], row['r_successes']
    p_i = r / n if n > 0 else 0
    if p_i > 0 and p_i < 1:
        log_lik_h3 += stats.binom.logpmf(r, n, p_i)

k_h3 = len(data)  # One parameter per group
aic_h3 = -2 * log_lik_h3 + 2 * k_h3
bic_h3 = -2 * log_lik_h3 + k_h3 * np.log(len(data))

print(f"\nModel fit statistics (saturated model):")
print(f"  Log-likelihood: {log_lik_h3:.4f}")
print(f"  AIC: {aic_h3:.4f}")
print(f"  BIC: {bic_h3:.4f}")

print(f"\nModel comparison (vs H1):")
print(f"  Delta AIC: {aic_h3 - aic_h1:.4f}")
print(f"  Delta BIC: {bic_h3 - bic_h1:.4f}")

# Likelihood ratio test
lr_statistic = 2 * (log_lik_h3 - log_lik_h1)
lr_df = k_h3 - k_h1
lr_p = 1 - stats.chi2.cdf(lr_statistic, lr_df)

print(f"\nLikelihood ratio test:")
print(f"  LR statistic: {lr_statistic:.4f}")
print(f"  df: {lr_df}")
print(f"  p-value: {lr_p:.6f}")
print(f"  Conclusion: {'STRONG SUPPORT for H3' if lr_p < 0.001 else 'MODERATE SUPPORT for H3' if lr_p < 0.05 else 'Weak support'}")

# Test for subgroups with high precision
high_precision = data[data['n_trials'] > 150].copy()
high_prec_outliers = high_precision[np.abs(high_precision['z_score']) > 1.96]

print(f"\nHigh-precision outliers test:")
print(f"  Groups with n > 150: {len(high_precision)}")
print(f"  High-precision outliers: {len(high_prec_outliers)}")
print(f"  Conclusion: {'STRONG SUPPORT for H3' if len(high_prec_outliers) > 0 else 'No strong support'}")

# SUMMARY COMPARISON
print("\n" + "="*80)
print("SUMMARY: MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Hypothesis': ['H1: Homogeneous', 'H3: Heterogeneous'],
    'Parameters': [k_h1, k_h3],
    'Log-Likelihood': [log_lik_h1, log_lik_h3],
    'AIC': [aic_h1, aic_h3],
    'BIC': [bic_h1, bic_h3]
})
print(comparison.to_string(index=False))

print(f"\n\nBEST MODEL (by AIC): {'H1' if aic_h1 < aic_h3 else 'H3'}")
print(f"BEST MODEL (by BIC): {'H1' if bic_h1 < bic_h3 else 'H3'}")

print("\n\nOVERALL CONCLUSION:")
print("-" * 80)

evidence = []
if p_value < 0.05:
    evidence.append("Chi-square test rejects homogeneity")
if var_ratio > 1.5:
    evidence.append(f"Variance ratio ({var_ratio:.2f}) indicates overdispersion")
if lr_p < 0.05:
    evidence.append("Likelihood ratio test favors heterogeneous model")
if len(high_prec_outliers) > 0:
    evidence.append(f"{len(high_prec_outliers)} high-precision outliers detected")
if aic_h3 < aic_h1:
    evidence.append("Lower AIC for heterogeneous model")

if len(evidence) >= 3:
    print("STRONG EVIDENCE for heterogeneity (H3):")
else:
    print("MODERATE EVIDENCE for heterogeneity (H3):")

for e in evidence:
    print(f"  - {e}")

if abs(corr_pearson) > 0.3 and p_pearson < 0.10:
    print(f"\nNote: Some evidence for sample-size dependence (r={corr_pearson:.2f}, p={p_pearson:.3f})")
    print("      This suggests heterogeneity may be related to group characteristics")

print("\n" + "="*80)
