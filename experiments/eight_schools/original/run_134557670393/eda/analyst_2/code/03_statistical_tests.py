"""
Statistical Tests for Uncertainty Patterns
Testing key hypotheses about precision-effect relationships
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau

# Load data
df = pd.read_csv('/workspace/eda/analyst_2/code/enhanced_data.csv')

print("="*70)
print("STATISTICAL TESTS FOR UNCERTAINTY PATTERNS")
print("="*70)

# ============================================================
# HYPOTHESIS 1: Precision-Effect Relationship (Publication Bias)
# ============================================================
print("\n" + "="*70)
print("HYPOTHESIS 1: Precision-Effect Relationship")
print("Testing for publication bias / small-study effects")
print("="*70)

# Egger's test (regression of standardized effect on precision)
# Model: z_i ~ beta_0 + beta_1 * (1/SE_i)
from scipy.stats import linregress

precision = df['precision'].values
z_scores = df['snr'].values
result = linregress(precision, z_scores)

print("\nEgger's Regression Test (Precision vs Standardized Effect):")
print(f"  Slope: {result.slope:.4f}")
print(f"  Intercept (bias indicator): {result.intercept:.4f}")
print(f"  R²: {result.rvalue**2:.4f}")
print(f"  p-value: {result.pvalue:.4f}")
print(f"  Interpretation: {'Significant asymmetry (p<0.05)' if result.pvalue < 0.05 else 'No significant asymmetry (p≥0.05)'}")

# Begg's test (rank correlation)
print("\nBegg's Rank Correlation Test:")
kendall_tau, kendall_p = kendalltau(df['variance'], np.abs(df['y']))
print(f"  Kendall's tau: {kendall_tau:.4f}")
print(f"  p-value: {kendall_p:.4f}")
print(f"  Interpretation: {'Significant rank correlation' if kendall_p < 0.05 else 'No significant rank correlation'}")

# Standard correlation tests
print("\nStandard Correlation Tests (Precision vs Effect):")
pearson_r, pearson_p = stats.pearsonr(df['precision'], df['y'])
spearman_r, spearman_p = spearmanr(df['precision'], df['y'])

print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")

# ============================================================
# HYPOTHESIS 2: High vs Low Precision Group Differences
# ============================================================
print("\n" + "="*70)
print("HYPOTHESIS 2: High vs Low Precision Group Differences")
print("Testing if precise studies show different effects")
print("="*70)

median_precision = df['precision'].median()
high_prec = df[df['precision'] >= median_precision]['y']
low_prec = df[df['precision'] < median_precision]['y']

print(f"\nMedian precision split: {median_precision:.4f}")
print(f"High precision group (n={len(high_prec)}): mean={high_prec.mean():.2f}, std={high_prec.std():.2f}")
print(f"Low precision group (n={len(low_prec)}): mean={low_prec.mean():.2f}, std={low_prec.std():.2f}")

# Mann-Whitney U test (non-parametric)
u_stat, u_pval = stats.mannwhitneyu(high_prec, low_prec, alternative='two-sided')
print(f"\nMann-Whitney U test:")
print(f"  U statistic: {u_stat:.2f}")
print(f"  p-value: {u_pval:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(high_prec)-1)*high_prec.std()**2 + (len(low_prec)-1)*low_prec.std()**2) / (len(high_prec)+len(low_prec)-2))
cohens_d = (high_prec.mean() - low_prec.mean()) / pooled_std
print(f"  Cohen's d: {cohens_d:.3f}")
print(f"  Interpretation: {abs(cohens_d):.2f} {'(small)' if abs(cohens_d) < 0.5 else '(medium)' if abs(cohens_d) < 0.8 else '(large)'}")

# ============================================================
# HYPOTHESIS 3: Heterogeneity Assessment
# ============================================================
print("\n" + "="*70)
print("HYPOTHESIS 3: Heterogeneity Assessment")
print("Testing for between-study variation beyond sampling error")
print("="*70)

# Calculate Q statistic
weights = 1 / df['variance']
weighted_mean = np.sum(df['y'] * weights) / np.sum(weights)
Q = np.sum(weights * (df['y'] - weighted_mean)**2)
df_q = len(df) - 1
Q_pval = 1 - stats.chi2.cdf(Q, df_q)

print(f"\nCochran's Q Test:")
print(f"  Q statistic: {Q:.3f}")
print(f"  Degrees of freedom: {df_q}")
print(f"  p-value: {Q_pval:.4f}")
print(f"  Interpretation: {'Significant heterogeneity (p<0.05)' if Q_pval < 0.05 else 'No significant heterogeneity (p≥0.05)'}")

# Calculate I² statistic
I2 = max(0, ((Q - df_q) / Q) * 100) if Q > 0 else 0
print(f"\nI² Statistic:")
print(f"  I² = {I2:.1f}%")
print(f"  Interpretation: ", end="")
if I2 < 25:
    print("Low heterogeneity")
elif I2 < 50:
    print("Moderate heterogeneity")
elif I2 < 75:
    print("Substantial heterogeneity")
else:
    print("Considerable heterogeneity")

# Calculate tau² (between-study variance)
C = np.sum(weights) - (np.sum(weights**2) / np.sum(weights))
tau2 = max(0, (Q - df_q) / C) if Q > df_q else 0
print(f"\nTau² (between-study variance): {tau2:.3f}")

# ============================================================
# HYPOTHESIS 4: Normality of Standardized Effects
# ============================================================
print("\n" + "="*70)
print("HYPOTHESIS 4: Distribution of Standardized Effects")
print("Testing if z-scores follow expected null distribution")
print("="*70)

# Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(df['snr'])
print(f"\nShapiro-Wilk Normality Test (on z-scores):")
print(f"  W statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
print(f"  Interpretation: {'Non-normal distribution' if shapiro_p < 0.05 else 'Cannot reject normality'}")

# One-sample t-test (are z-scores significantly different from 0?)
t_stat, t_pval = stats.ttest_1samp(df['snr'], 0)
print(f"\nOne-sample t-test (H0: mean z-score = 0):")
print(f"  t statistic: {t_stat:.4f}")
print(f"  p-value: {t_pval:.4f}")
print(f"  Mean z-score: {df['snr'].mean():.4f}")
print(f"  Interpretation: {'Reject H0' if t_pval < 0.05 else 'Cannot reject H0 (mean not significantly different from 0)'}")

# ============================================================
# HYPOTHESIS 5: Variance-Effect Relationship
# ============================================================
print("\n" + "="*70)
print("HYPOTHESIS 5: Variance-Effect Relationship")
print("Testing for heteroscedasticity patterns")
print("="*70)

var_effect_corr, var_effect_p = stats.pearsonr(df['variance'], df['y'])
print(f"\nVariance vs Effect Size Correlation:")
print(f"  Pearson r: {var_effect_corr:.4f}")
print(f"  p-value: {var_effect_p:.4f}")
print(f"  Interpretation: {'Significant relationship' if var_effect_p < 0.05 else 'No significant relationship'}")

# Test for linear trend
lr_result = linregress(df['variance'], df['y'])
print(f"\nLinear Regression (Effect ~ Variance):")
print(f"  Slope: {lr_result.slope:.4f}")
print(f"  Intercept: {lr_result.intercept:.4f}")
print(f"  R²: {lr_result.rvalue**2:.4f}")
print(f"  p-value: {lr_result.pvalue:.4f}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("SUMMARY: KEY UNCERTAINTY METRICS")
print("="*70)

print("\nCentral Tendency:")
print(f"  Unweighted mean effect: {df['y'].mean():.3f}")
print(f"  Precision-weighted mean: {weighted_mean:.3f}")
print(f"  Median effect: {df['y'].median():.3f}")

print("\nUncertainty Characteristics:")
print(f"  Mean SE: {df['sigma'].mean():.3f} (range: {df['sigma'].min():.1f}-{df['sigma'].max():.1f})")
print(f"  CV of SE: {df['sigma'].std() / df['sigma'].mean():.3f}")
print(f"  Mean SNR: {df['snr'].mean():.3f} (max: {df['snr'].max():.3f})")

print("\nStatistical Significance:")
print(f"  Studies with |z| > 1.96: {(np.abs(df['snr']) > 1.96).sum()}/8")
print(f"  Studies with |z| > 1.64: {(np.abs(df['snr']) > 1.64).sum()}/8")
print(f"  Largest |z-score|: {np.abs(df['snr']).max():.3f} (Study {df.loc[np.abs(df['snr']).idxmax(), 'study']:.0f})")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
