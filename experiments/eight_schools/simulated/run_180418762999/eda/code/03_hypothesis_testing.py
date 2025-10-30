"""
Hypothesis Testing and Model Selection Exploration
===================================================
Goal: Test competing hypotheses about data structure to inform modeling choices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Load data
data = pd.read_csv('/workspace/eda/code/data_with_metrics.csv')

print("="*80)
print("HYPOTHESIS TESTING FOR MODEL SELECTION")
print("="*80)

# ============================================================================
# HYPOTHESIS 1: Are all groups drawn from the same population?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 1: Complete Pooling vs Separate Groups")
print("="*80)

print("\nH0: All groups come from the same population (complete pooling appropriate)")
print("H1: Groups have different true means (partial/no pooling needed)")

# Weighted mean (accounting for measurement error)
weights = 1 / (data['sigma']**2)
weighted_mean = np.sum(data['y'] * weights) / np.sum(weights)
weighted_var = 1 / np.sum(weights)
weighted_std = np.sqrt(weighted_var)

print(f"\nSimple mean: {data['y'].mean():.4f} (SE: {data['y'].std() / np.sqrt(len(data)):.4f})")
print(f"Weighted mean: {weighted_mean:.4f} (SE: {weighted_std:.4f})")

# Chi-square test for homogeneity (accounting for measurement error)
chi_sq = np.sum(((data['y'] - weighted_mean)**2) / (data['sigma']**2))
df = len(data) - 1
p_value = 1 - stats.chi2.cdf(chi_sq, df)

print(f"\nChi-square test for homogeneity:")
print(f"  Chi-square statistic: {chi_sq:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  P-value: {p_value:.4f}")
print(f"  Expected chi-square under H0: {df} (±{np.sqrt(2*df):.2f})")

if p_value < 0.05:
    print(f"  Result: REJECT H0 (p < 0.05) - Groups appear heterogeneous")
    print(f"  Implication: Hierarchical or no-pooling model preferred")
else:
    print(f"  Result: FAIL TO REJECT H0 (p >= 0.05) - Groups may be homogeneous")
    print(f"  Implication: Complete pooling may be appropriate")

# ============================================================================
# HYPOTHESIS 2: Is there a sign pattern (positive vs negative values)?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 2: Sign Pattern Analysis")
print("="*80)

print("\nH0: True values are equally likely to be positive or negative")
print("H1: There is a systematic pattern in signs")

# Account for measurement uncertainty
# A value is "significantly" positive/negative if |y| > sigma
sig_positive = sum((data['y'] > 0) & (data['y'] > data['sigma']))
sig_negative = sum((data['y'] < 0) & (np.abs(data['y']) > data['sigma']))
uncertain = len(data) - sig_positive - sig_negative

print(f"\nSign analysis (accounting for uncertainty):")
print(f"  Significantly positive (y > sigma): {sig_positive}")
print(f"  Significantly negative (|y| > sigma): {sig_negative}")
print(f"  Uncertain (|y| <= sigma): {uncertain}")

# Binomial test (excluding uncertain)
if sig_positive + sig_negative > 0:
    binom_p = stats.binomtest(sig_positive, sig_positive + sig_negative, 0.5)
    print(f"\nBinomial test (excluding uncertain):")
    print(f"  P-value: {binom_p.pvalue:.4f}")
    if binom_p.pvalue < 0.05:
        print(f"  Result: REJECT H0 - Significant bias toward positive values")
    else:
        print(f"  Result: FAIL TO REJECT H0 - No strong sign pattern")

# Check if mean is significantly different from zero
t_stat = data['y'].mean() / (data['y'].std() / np.sqrt(len(data)))
t_p = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 1))
print(f"\nOne-sample t-test (mean vs 0):")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {t_p:.4f}")

# Weighted test
z_stat = weighted_mean / weighted_std
z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print(f"\nWeighted test (accounting for measurement error):")
print(f"  Z-statistic: {z_stat:.4f}")
print(f"  P-value: {z_p:.4f}")

# ============================================================================
# HYPOTHESIS 3: Is there clustering/grouping structure?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3: Clustering Structure")
print("="*80)

print("\nH0: Groups are randomly distributed in value space")
print("H1: Groups form clusters (e.g., high vs low groups)")

# Simple gap statistic approach
sorted_y = np.sort(data['y'].values)
gaps = np.diff(sorted_y)
max_gap = np.max(gaps)
max_gap_idx = np.argmax(gaps)
mean_gap = np.mean(gaps)

print(f"\nGap analysis:")
print(f"  Sorted y values: {sorted_y}")
print(f"  Gaps between consecutive values: {gaps}")
print(f"  Largest gap: {max_gap:.4f} (between {sorted_y[max_gap_idx]:.2f} and {sorted_y[max_gap_idx+1]:.2f})")
print(f"  Mean gap: {mean_gap:.4f}")
print(f"  Ratio (max/mean): {max_gap/mean_gap:.4f}")

if max_gap > 2.5 * mean_gap:
    print(f"  Result: Large gap detected - possible cluster structure")
    print(f"  Potential clusters: ")
    print(f"    Low cluster: {sorted_y[sorted_y <= sorted_y[max_gap_idx]]}")
    print(f"    High cluster: {sorted_y[sorted_y > sorted_y[max_gap_idx]]}")
else:
    print(f"  Result: No strong clustering pattern")

# ============================================================================
# HYPOTHESIS 4: Relationship between magnitude and uncertainty
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 4: Magnitude-Uncertainty Relationship")
print("="*80)

print("\nH0: Measurement error is independent of true magnitude")
print("H1: Measurement error depends on true magnitude")

# Test correlation
pearson_r, pearson_p = stats.pearsonr(np.abs(data['y']), data['sigma'])
spearman_r, spearman_p = stats.spearmanr(np.abs(data['y']), data['sigma'])

print(f"\nCorrelation tests:")
print(f"  Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
print(f"  Spearman correlation: rho = {spearman_r:.4f}, p = {spearman_p:.4f}")

if pearson_p < 0.05 or spearman_p < 0.05:
    print(f"  Result: Significant relationship detected")
    print(f"  Implication: Heteroscedastic measurement error model may be appropriate")
else:
    print(f"  Result: No significant relationship")
    print(f"  Implication: Constant measurement error assumption reasonable")

# ============================================================================
# HYPOTHESIS 5: Between-group variance vs within-group uncertainty
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 5: Between-Group Variability")
print("="*80)

print("\nComparing observed variance to expected measurement variance")

observed_var = np.var(data['y'], ddof=1)
mean_measurement_var = np.mean(data['sigma']**2)
between_group_var = max(0, observed_var - mean_measurement_var)

print(f"\nVariance decomposition:")
print(f"  Observed variance in y: {observed_var:.4f}")
print(f"  Mean measurement variance: {mean_measurement_var:.4f}")
print(f"  Estimated between-group variance: {between_group_var:.4f}")
print(f"  Ratio (between/within): {between_group_var/mean_measurement_var:.4f}")

intraclass_corr = between_group_var / (between_group_var + mean_measurement_var)
print(f"  Estimated intraclass correlation: {intraclass_corr:.4f}")

if between_group_var > mean_measurement_var:
    print(f"\n  Result: Between-group variance exceeds measurement variance")
    print(f"  Implication: Strong evidence for hierarchical structure")
elif between_group_var > 0.5 * mean_measurement_var:
    print(f"\n  Result: Moderate between-group variance")
    print(f"  Implication: Hierarchical model with partial pooling appropriate")
else:
    print(f"\n  Result: Between-group variance is small")
    print(f"  Implication: Complete pooling may be sufficient")

# ============================================================================
# HYPOTHESIS 6: Are any groups outliers?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 6: Outlier Detection")
print("="*80)

print("\nTesting if any group is inconsistent with the others")

# Leave-one-out analysis
print("\nLeave-one-out analysis:")
for i in range(len(data)):
    # Exclude one group
    mask = data.index != i
    loo_data = data[mask]

    # Calculate weighted mean without this group
    loo_weights = 1 / (loo_data['sigma']**2)
    loo_mean = np.sum(loo_data['y'] * loo_weights) / np.sum(loo_weights)
    loo_se = np.sqrt(1 / np.sum(loo_weights))

    # Z-score for excluded group
    z_score = (data.loc[i, 'y'] - loo_mean) / np.sqrt(data.loc[i, 'sigma']**2 + loo_se**2)
    p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))

    status = "OUTLIER" if abs(z_score) > 2.5 else "normal"
    print(f"  Group {int(data.loc[i, 'group'])}: z = {z_score:6.3f}, p = {p_val:.4f} [{status}]")

print("\n" + "="*80)
print("HYPOTHESIS TESTING COMPLETE")
print("="*80)
