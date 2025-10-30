"""
Hypothesis Testing and Model Comparison
========================================
This script tests competing hypotheses about the data generating process.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/workspace/data/data.csv')
data['precision'] = 1 / data['sigma']**2

print("="*80)
print("HYPOTHESIS TESTING: COMPETING DATA GENERATION MODELS")
print("="*80)

# ============================================================================
# HYPOTHESIS 1: Common Effect Model
# All studies estimate the same underlying parameter theta
# y_i ~ N(theta, sigma_i^2)
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 1: COMMON EFFECT (FIXED EFFECT) MODEL")
print("="*80)
print("Assumption: All observations estimate the same true parameter theta")
print("Model: y_i ~ N(theta, sigma_i^2)")

# Fixed effect estimate
weights = 1 / data['sigma']**2
theta_fe = np.sum(data['y'] * weights) / np.sum(weights)
se_fe = np.sqrt(1 / np.sum(weights))

print(f"\nEstimated theta: {theta_fe:.3f} ± {se_fe:.3f}")
print(f"95% CI: [{theta_fe - 1.96*se_fe:.3f}, {theta_fe + 1.96*se_fe:.3f}]")

# Cochran's Q test for homogeneity
Q = np.sum(weights * (data['y'] - theta_fe)**2)
df = len(data) - 1
p_value_Q = 1 - stats.chi2.cdf(Q, df)
I2 = max(0, 100 * (Q - df) / Q)

print(f"\nHomogeneity test (Cochran's Q):")
print(f"  Q statistic: {Q:.3f}")
print(f"  Degrees of freedom: {df}")
print(f"  P-value: {p_value_Q:.4f}")
print(f"  I² statistic: {I2:.1f}%")
print(f"  Interpretation: {'HOMOGENEOUS (fixed effect appropriate)' if p_value_Q > 0.10 else 'HETEROGENEOUS (random effects needed)'}")

# Check individual study consistency
z_scores = (data['y'] - theta_fe) / data['sigma']
outliers = np.abs(z_scores) > 1.96
print(f"\nIndividual study z-scores:")
for i, (y, sigma, z, is_outlier) in enumerate(zip(data['y'], data['sigma'], z_scores, outliers)):
    flag = " [OUTLIER]" if is_outlier else ""
    print(f"  Obs {i+1}: y={y:6.1f}, z={z:6.2f}{flag}")
print(f"Number of outliers (|z| > 1.96): {np.sum(outliers)}/{len(data)}")

# Model fit: Log-likelihood
log_lik_fe = -0.5 * np.sum(np.log(2 * np.pi * data['sigma']**2) +
                            (data['y'] - theta_fe)**2 / data['sigma']**2)
AIC_fe = -2 * log_lik_fe + 2 * 1  # 1 parameter: theta
BIC_fe = -2 * log_lik_fe + np.log(len(data)) * 1

print(f"\nModel fit:")
print(f"  Log-likelihood: {log_lik_fe:.3f}")
print(f"  AIC: {AIC_fe:.3f}")
print(f"  BIC: {BIC_fe:.3f}")

# ============================================================================
# HYPOTHESIS 2: Random Effects Model
# Studies estimate different but related effects
# y_i ~ N(theta_i, sigma_i^2), theta_i ~ N(mu, tau^2)
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 2: RANDOM EFFECTS MODEL")
print("="*80)
print("Assumption: Studies have heterogeneous true effects")
print("Model: y_i ~ N(theta_i, sigma_i^2), theta_i ~ N(mu, tau^2)")

# DerSimonian-Laird estimator for tau^2
tau2_DL = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
print(f"\nBetween-study variance (tau²):")
print(f"  DL estimator: {tau2_DL:.3f}")

if tau2_DL > 0:
    # Random effects estimate
    re_weights = 1 / (data['sigma']**2 + tau2_DL)
    mu_re = np.sum(data['y'] * re_weights) / np.sum(re_weights)
    se_re = np.sqrt(1 / np.sum(re_weights))

    print(f"\nEstimated mu: {mu_re:.3f} ± {se_re:.3f}")
    print(f"95% CI: [{mu_re - 1.96*se_re:.3f}, {mu_re + 1.96*se_re:.3f}]")

    # Model fit
    log_lik_re = -0.5 * np.sum(np.log(2 * np.pi * (data['sigma']**2 + tau2_DL)) +
                                (data['y'] - mu_re)**2 / (data['sigma']**2 + tau2_DL))
    AIC_re = -2 * log_lik_re + 2 * 2  # 2 parameters: mu, tau²
    BIC_re = -2 * log_lik_re + np.log(len(data)) * 2

    print(f"\nModel fit:")
    print(f"  Log-likelihood: {log_lik_re:.3f}")
    print(f"  AIC: {AIC_re:.3f}")
    print(f"  BIC: {BIC_re:.3f}")
else:
    print("\nNote: tau² = 0, random effects model reduces to fixed effect model")
    mu_re = theta_fe
    se_re = se_fe
    log_lik_re = log_lik_fe
    AIC_re = AIC_fe
    BIC_re = BIC_fe

# ============================================================================
# HYPOTHESIS 3: Two-Group Model
# Are there distinct subgroups in the data?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3: TWO-GROUP MODEL")
print("="*80)
print("Question: Do observations cluster into distinct groups?")

# Simple median split
median_y = data['y'].median()
group1_mask = data['y'] <= median_y
group2_mask = data['y'] > median_y

print(f"\nMedian split on y values (median = {median_y}):")
print(f"  Group 1 (y ≤ {median_y}): {data[group1_mask]['y'].values}")
print(f"  Group 2 (y > {median_y}): {data[group2_mask]['y'].values}")
print(f"  Group 1 mean: {data[group1_mask]['y'].mean():.2f}")
print(f"  Group 2 mean: {data[group2_mask]['y'].mean():.2f}")

# Test if groups are significantly different
if group1_mask.sum() > 0 and group2_mask.sum() > 0:
    t_stat, p_val_ttest = stats.ttest_ind(data[group1_mask]['y'], data[group2_mask]['y'])
    print(f"\nTwo-sample t-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_val_ttest:.4f}")
    print(f"  Significant difference: {p_val_ttest < 0.05}")

# Check if sigma differs between groups
print(f"\nStandard errors by group:")
print(f"  Group 1 mean sigma: {data[group1_mask]['sigma'].mean():.2f}")
print(f"  Group 2 mean sigma: {data[group2_mask]['sigma'].mean():.2f}")
t_stat_sigma, p_val_sigma = stats.ttest_ind(data[group1_mask]['sigma'], data[group2_mask]['sigma'])
print(f"  t-test p-value: {p_val_sigma:.4f}")

# Look for potential grouping by extreme values
print(f"\nExtreme value analysis:")
extreme_mask = (data['y'] < data['y'].quantile(0.25)) | (data['y'] > data['y'].quantile(0.75))
print(f"  Extreme observations (Q1 or Q3): {data[extreme_mask]['y'].values}")
print(f"  Middle observations: {data[~extreme_mask]['y'].values}")

# ============================================================================
# HYPOTHESIS 4: Relationship Model
# Is y systematically related to sigma?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 4: SYSTEMATIC Y-SIGMA RELATIONSHIP")
print("="*80)
print("Question: Is there a systematic relationship between y and sigma?")

# Linear regression: y ~ sigma
from scipy.stats import linregress
slope, intercept, r_value, p_value_reg, std_err = linregress(data['sigma'], data['y'])

print(f"\nLinear regression: y = beta0 + beta1 * sigma")
print(f"  Intercept (beta0): {intercept:.3f}")
print(f"  Slope (beta1): {slope:.3f} ± {std_err:.3f}")
print(f"  R²: {r_value**2:.3f}")
print(f"  P-value for slope: {p_value_reg:.4f}")
print(f"  Significant relationship: {p_value_reg < 0.05}")

# Correlation tests
corr_pearson, p_pearson = stats.pearsonr(data['y'], data['sigma'])
corr_spearman, p_spearman = stats.spearmanr(data['y'], data['sigma'])

print(f"\nCorrelation analysis:")
print(f"  Pearson r: {corr_pearson:.3f} (p = {p_pearson:.4f})")
print(f"  Spearman ρ: {corr_spearman:.3f} (p = {p_spearman:.4f})")
print(f"  Evidence for relationship: {'YES' if p_pearson < 0.10 else 'NO'}")

# ============================================================================
# HYPOTHESIS 5: Publication Bias
# Are small studies (high sigma) systematically different?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 5: PUBLICATION BIAS")
print("="*80)
print("Question: Do small studies show systematically different effects?")

# Egger's test for funnel plot asymmetry
precision = 1 / data['sigma']
# Standard Egger test: regress standardized effect (y/sigma) on precision
slope_egger, intercept_egger, r_egger, p_egger, stderr_egger = linregress(
    precision, data['y'] / data['sigma']
)

print(f"\nEgger's regression test:")
print(f"  Regression of (y/sigma) on precision")
print(f"  Intercept: {intercept_egger:.3f}")
print(f"  Slope: {slope_egger:.3f}")
print(f"  P-value for intercept: {p_egger:.4f}")
print(f"  Publication bias detected: {abs(intercept_egger) > 2 and p_egger < 0.10}")

# Begg's rank correlation test
rank_y = stats.rankdata(data['y'])
rank_sigma = stats.rankdata(data['sigma'])
corr_begg, p_begg = stats.spearmanr(rank_y, rank_sigma)

print(f"\nBegg's rank correlation test:")
print(f"  Spearman correlation: {corr_begg:.3f}")
print(f"  P-value: {p_begg:.4f}")
print(f"  Publication bias detected: {p_begg < 0.10}")

# Additional: check for funnel asymmetry
# High sigma (small studies) should cluster around mean if no bias
high_sigma_mask = data['sigma'] > data['sigma'].median()
low_sigma_mask = data['sigma'] <= data['sigma'].median()

print(f"\nAsymmetry check:")
print(f"  High uncertainty studies (sigma > median): mean y = {data[high_sigma_mask]['y'].mean():.2f}")
print(f"  Low uncertainty studies (sigma ≤ median): mean y = {data[low_sigma_mask]['y'].mean():.2f}")
print(f"  Difference: {data[high_sigma_mask]['y'].mean() - data[low_sigma_mask]['y'].mean():.2f}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

models = {
    'Fixed Effect': {'AIC': AIC_fe, 'BIC': BIC_fe, 'LogLik': log_lik_fe, 'params': 1},
    'Random Effects': {'AIC': AIC_re, 'BIC': BIC_re, 'LogLik': log_lik_re, 'params': 2},
}

print(f"\n{'Model':<20} {'LogLik':>10} {'AIC':>10} {'BIC':>10} {'Params':>8}")
print("-" * 60)
for name, metrics in models.items():
    print(f"{name:<20} {metrics['LogLik']:>10.2f} {metrics['AIC']:>10.2f} {metrics['BIC']:>10.2f} {metrics['params']:>8}")

# Best model by AIC
best_aic = min(models.items(), key=lambda x: x[1]['AIC'])
best_bic = min(models.items(), key=lambda x: x[1]['BIC'])

print(f"\nBest model by AIC: {best_aic[0]}")
print(f"Best model by BIC: {best_bic[0]}")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS TESTING CONCLUSIONS")
print("="*80)

print("\n1. HOMOGENEITY TEST:")
if p_value_Q > 0.10:
    print("   → Studies appear HOMOGENEOUS (Q test p = {:.4f})".format(p_value_Q))
    print("   → Fixed effect model is appropriate")
    print("   → All studies likely estimate the same underlying parameter")
else:
    print("   → Studies appear HETEROGENEOUS (Q test p = {:.4f})".format(p_value_Q))
    print("   → Random effects model recommended")
    print("   → True effects may vary across studies")

print("\n2. HETEROGENEITY MAGNITUDE:")
print("   → I² = {:.1f}% ({:.1f}% of variance due to heterogeneity)".format(
    I2, I2))
if I2 < 25:
    print("   → Low heterogeneity")
elif I2 < 50:
    print("   → Moderate heterogeneity")
else:
    print("   → High heterogeneity")

print("\n3. Y-SIGMA RELATIONSHIP:")
if p_value_reg < 0.05:
    print("   → Significant relationship detected (p = {:.4f})".format(p_value_reg))
    print("   → Larger uncertainties associated with different effect sizes")
else:
    print("   → No significant relationship (p = {:.4f})".format(p_value_reg))
    print("   → Effect size independent of study precision")

print("\n4. PUBLICATION BIAS:")
if p_egger < 0.10 or p_begg < 0.10:
    print("   → Potential publication bias detected")
    print("   → Egger's test p = {:.4f}".format(p_egger))
    print("   → Begg's test p = {:.4f}".format(p_begg))
else:
    print("   → No strong evidence of publication bias")
    print("   → Egger's test p = {:.4f}".format(p_egger))
    print("   → Begg's test p = {:.4f}".format(p_begg))

print("\n5. RECOMMENDED MODEL:")
if tau2_DL == 0 and p_value_Q > 0.10:
    print("   → FIXED EFFECT MODEL")
    print("   → Estimate: {:.3f} ± {:.3f}".format(theta_fe, se_fe))
    print("   → 95% CI: [{:.3f}, {:.3f}]".format(
        theta_fe - 1.96*se_fe, theta_fe + 1.96*se_fe))
else:
    print("   → RANDOM EFFECTS MODEL")
    print("   → Estimate: {:.3f} ± {:.3f}".format(mu_re, se_re))
    print("   → 95% CI: [{:.3f}, {:.3f}]".format(
        mu_re - 1.96*se_re, mu_re + 1.96*se_re))
    print("   → Between-study SD (tau): {:.3f}".format(np.sqrt(tau2_DL)))

print("\n" + "="*80)
