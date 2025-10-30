"""
Hypothesis Testing: Competing Explanations for the Data
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'eda' / 'analyst_3' / 'code' / 'data_with_diagnostics.csv'

# Load data
df = pd.read_csv(DATA_PATH)

print("="*80)
print("HYPOTHESIS TESTING: THREE COMPETING MODELS")
print("="*80)

# ============================================================================
# HYPOTHESIS 1: Homogeneous Binomial (Pooled Model)
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 1: All groups have the same success probability")
print("Model: r_i ~ Binomial(n_i, p_pooled)")
print("="*80)

total_successes = df['r_successes'].sum()
total_trials = df['n_trials'].sum()
pooled_p = total_successes / total_trials

print(f"\nPooled estimate: p = {pooled_p:.6f}")
print(f"Log-likelihood: ", end="")

# Calculate log-likelihood
log_lik_pooled = 0
for idx, row in df.iterrows():
    n, r = row['n_trials'], row['r_successes']
    # log binom coef + r*log(p) + (n-r)*log(1-p)
    log_binom_coef = stats.binom.logpmf(r, n, pooled_p)
    log_lik_pooled += log_binom_coef

print(f"{log_lik_pooled:.2f}")

# AIC = 2k - 2*log_lik (k = 1 parameter)
aic_pooled = 2 * 1 - 2 * log_lik_pooled
bic_pooled = np.log(len(df)) * 1 - 2 * log_lik_pooled

print(f"AIC: {aic_pooled:.2f}")
print(f"BIC: {bic_pooled:.2f}")
print(f"Number of parameters: 1")

# Chi-square goodness of fit
chi_square = (df['pearson_residual']**2).sum()
p_value = 1 - stats.chi2.cdf(chi_square, df=11)
print(f"\nGoodness of fit:")
print(f"  Chi-square = {chi_square:.4f}, df = 11, p-value = {p_value:.4f}")
print(f"  Conclusion: {'REJECT - groups differ' if p_value < 0.05 else 'Cannot reject'}")

# ============================================================================
# HYPOTHESIS 2: Heterogeneous Binomial (Separate p for each group)
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 2: Each group has its own success probability")
print("Model: r_i ~ Binomial(n_i, p_i)")
print("="*80)

print("\nGroup-specific estimates:")
print(df[['group', 'n_trials', 'r_successes', 'success_rate']])

# Calculate log-likelihood
log_lik_heterogeneous = 0
for idx, row in df.iterrows():
    n, r = row['n_trials'], row['r_successes']
    p = r / n if n > 0 else 0.5
    if r == 0:
        log_lik_heterogeneous += n * np.log(1 - p + 1e-10)
    elif r == n:
        log_lik_heterogeneous += n * np.log(p + 1e-10)
    else:
        log_binom_coef = stats.binom.logpmf(r, n, p)
        log_lik_heterogeneous += log_binom_coef

print(f"\nLog-likelihood: {log_lik_heterogeneous:.2f}")

# AIC = 2k - 2*log_lik (k = 12 parameters)
aic_heterogeneous = 2 * 12 - 2 * log_lik_heterogeneous
bic_heterogeneous = np.log(len(df)) * 12 - 2 * log_lik_heterogeneous

print(f"AIC: {aic_heterogeneous:.2f}")
print(f"BIC: {bic_heterogeneous:.2f}")
print(f"Number of parameters: 12")

print(f"\nNote: This is a saturated model (perfect fit by definition)")

# ============================================================================
# HYPOTHESIS 3: Beta-Binomial Model (Hierarchical)
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3: Success probabilities drawn from Beta distribution")
print("Model: p_i ~ Beta(alpha, beta), r_i ~ Binomial(n_i, p_i)")
print("="*80)

# Method of moments estimation for Beta-Binomial
mean_p = df['success_rate'].mean()
var_p = df['success_rate'].var()

print(f"\nEmpirical moments:")
print(f"  Mean(p) = {mean_p:.6f}")
print(f"  Var(p) = {var_p:.6f}")

# Method of moments: mean = alpha/(alpha+beta), var = alpha*beta/((alpha+beta)^2*(alpha+beta+1))
# For beta distribution of p_i (not accounting for binomial sampling variance)
if var_p > 0 and mean_p * (1 - mean_p) > var_p:
    common_term = (mean_p * (1 - mean_p) / var_p) - 1
    alpha_mom = mean_p * common_term
    beta_mom = (1 - mean_p) * common_term
    print(f"\nMoment estimates (ignoring binomial variance):")
    print(f"  alpha = {alpha_mom:.4f}")
    print(f"  beta = {beta_mom:.4f}")
else:
    print("\nWARNING: Variance exceeds binomial expectation - strong heterogeneity")
    # Use alternative estimation
    alpha_mom = mean_p * 10  # placeholder
    beta_mom = (1 - mean_p) * 10

# Approximate log-likelihood (simplified - doesn't account for beta-binomial complexity)
# This is a rough approximation
log_lik_bb = 0
for idx, row in df.iterrows():
    p = row['success_rate']
    # Log-likelihood contribution from beta prior
    if p > 0 and p < 1:
        log_lik_bb += stats.beta.logpdf(p, alpha_mom, beta_mom)

print(f"\nApproximate log-likelihood (beta prior component): {log_lik_bb:.2f}")
print(f"Note: This is simplified - true beta-binomial likelihood is more complex")

# AIC = 2k - 2*log_lik (k = 2 parameters: alpha, beta)
# Note: This is approximate
aic_bb = 2 * 2 - 2 * (log_lik_pooled + log_lik_bb)
bic_bb = np.log(len(df)) * 2 - 2 * (log_lik_pooled + log_lik_bb)

print(f"Approximate AIC: {aic_bb:.2f}")
print(f"Approximate BIC: {bic_bb:.2f}")
print(f"Number of parameters: 2")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['Pooled Binomial', 'Heterogeneous Binomial', 'Beta-Binomial (approx)'],
    'Parameters': [1, 12, 2],
    'Log-Likelihood': [log_lik_pooled, log_lik_heterogeneous, log_lik_pooled + log_lik_bb],
    'AIC': [aic_pooled, aic_heterogeneous, aic_bb],
    'BIC': [bic_pooled, bic_heterogeneous, bic_bb]
})

print("\n", comparison_df.to_string(index=False))

best_aic = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']
best_bic = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']

print(f"\nBest model by AIC: {best_aic}")
print(f"Best model by BIC: {best_bic}")

# ============================================================================
# LIKELIHOOD RATIO TESTS
# ============================================================================
print("\n" + "="*80)
print("LIKELIHOOD RATIO TESTS")
print("="*80)

# Test 1: Pooled vs Heterogeneous
lr_stat_1 = 2 * (log_lik_heterogeneous - log_lik_pooled)
df_lr_1 = 12 - 1  # difference in parameters
p_value_1 = 1 - stats.chi2.cdf(lr_stat_1, df_lr_1)

print(f"\nTest 1: Pooled vs Heterogeneous")
print(f"  LR statistic: {lr_stat_1:.4f}")
print(f"  df: {df_lr_1}")
print(f"  p-value: {p_value_1:.4f}")
print(f"  Conclusion: {'Reject pooled model - groups differ significantly' if p_value_1 < 0.05 else 'Cannot reject pooled model'}")

# ============================================================================
# ADDITIONAL TESTS
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL STATISTICAL TESTS")
print("="*80)

# Cochran Q test (for related samples with binary outcomes)
# Here we're testing if success rates differ across groups
print("\n1. Test for homogeneity of proportions")

# Using chi-square test for homogeneity
observed = np.array([df['r_successes'].values, df['n_trials'].values - df['r_successes'].values]).T
chi2, p_val, dof, expected = stats.chi2_contingency(observed.T)
print(f"  Chi-square test statistic: {chi2:.4f}")
print(f"  df: {dof}")
print(f"  p-value: {p_val:.4f}")

# Bartlett's test for equal variances (transformed to logit scale, excluding zeros)
print("\n2. Bartlett's test for equal variances (logit scale, excluding extreme values)")
logit_values = []
for idx, row in df.iterrows():
    if row['success_rate'] > 0.01 and row['success_rate'] < 0.99:
        logit_values.append(np.log(row['success_rate'] / (1 - row['success_rate'])))

if len(logit_values) > 1:
    # Create groups (artificially split into two for testing)
    mid = len(logit_values) // 2
    group1 = logit_values[:mid]
    group2 = logit_values[mid:]
    if len(group1) > 1 and len(group2) > 1:
        bartlett_stat, bartlett_p = stats.bartlett(group1, group2)
        print(f"  Bartlett statistic: {bartlett_stat:.4f}")
        print(f"  p-value: {bartlett_p:.4f}")

# Shapiro-Wilk test on residuals
print("\n3. Shapiro-Wilk test on Pearson residuals (normality)")
shapiro_stat, shapiro_p = stats.shapiro(df['pearson_residual'])
print(f"  Shapiro-Wilk statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
print(f"  Conclusion: {'Residuals appear normally distributed' if shapiro_p > 0.05 else 'Residuals deviate from normality'}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_path = BASE_DIR / 'eda' / 'analyst_3' / 'code' / 'model_comparison.csv'
comparison_df.to_csv(output_path, index=False)
print(f"\n{output_path}")

print("\n" + "="*80)
print("HYPOTHESIS TESTING COMPLETE")
print("="*80)
