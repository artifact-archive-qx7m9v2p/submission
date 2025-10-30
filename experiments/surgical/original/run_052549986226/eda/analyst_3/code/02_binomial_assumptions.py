"""
Binomial Assumptions and Model Diagnostics
Focus: Testing if binomial likelihood is appropriate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data_analyst_3.csv'
OUTPUT_DIR = BASE_DIR / 'eda' / 'analyst_3'
VIZ_DIR = OUTPUT_DIR / 'visualizations'

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
df = pd.read_csv(DATA_PATH)

print("="*80)
print("BINOMIAL ASSUMPTIONS ASSESSMENT")
print("="*80)

# 1. CHECK FOR OVERDISPERSION
print("\n1. OVERDISPERSION CHECK")
print("-" * 80)

# For binomial data, variance should be n*p*(1-p)
# We'll compare observed variance to expected variance under pooled model

# Pooled estimate
total_successes = df['r_successes'].sum()
total_trials = df['n_trials'].sum()
pooled_p = total_successes / total_trials

print(f"Pooled success rate: {pooled_p:.6f}")
print(f"Total successes: {total_successes}")
print(f"Total trials: {total_trials}")

# Calculate expected variance and observed variance for each group
df['expected_variance'] = df['n_trials'] * pooled_p * (1 - pooled_p)
df['expected_std'] = np.sqrt(df['expected_variance'])

# Calculate standardized residuals (Pearson residuals)
df['expected_successes'] = df['n_trials'] * pooled_p
df['pearson_residual'] = (df['r_successes'] - df['expected_successes']) / df['expected_std']

print("\nPearson residuals (should follow N(0,1) if model is correct):")
print(df[['group', 'r_successes', 'expected_successes', 'pearson_residual']])

# Calculate chi-square goodness of fit
chi_square = (df['pearson_residual']**2).sum()
df_residual = len(df) - 1  # 12 groups - 1 parameter (pooled p)
p_value = 1 - stats.chi2.cdf(chi_square, df_residual)

print(f"\nChi-square goodness of fit test:")
print(f"  Chi-square statistic: {chi_square:.4f}")
print(f"  Degrees of freedom: {df_residual}")
print(f"  P-value: {p_value:.4f}")
print(f"  Conclusion: {'Model fits well' if p_value > 0.05 else 'Evidence of lack of fit (overdispersion or group differences)'}")

# Calculate dispersion parameter (should be ~1 for binomial)
dispersion = chi_square / df_residual
print(f"\nDispersion parameter (chi-square / df): {dispersion:.4f}")
print(f"  Interpretation: {'Slight overdispersion' if dispersion > 1.5 else 'Adequate fit' if dispersion <= 1.5 and dispersion >= 0.67 else 'Possible underdispersion'}")

# 2. CHECK FOR SAMPLE SIZE ADEQUACY
print("\n2. SAMPLE SIZE ADEQUACY FOR BINOMIAL APPROXIMATION")
print("-" * 80)

# Rule of thumb: both n*p and n*(1-p) should be >= 5 for normal approximation
df['np'] = df['n_trials'] * df['success_rate']
df['nq'] = df['n_trials'] * (1 - df['success_rate'])
df['adequate_sample'] = (df['np'] >= 5) & (df['nq'] >= 5)

print("Groups with adequate sample sizes for normal approximation (n*p >= 5 and n*q >= 5):")
print(df[['group', 'n_trials', 'np', 'nq', 'adequate_sample']])

inadequate = df[~df['adequate_sample']]
print(f"\nNumber of groups with inadequate sample sizes: {len(inadequate)}")
if len(inadequate) > 0:
    print("WARNING: These groups may require exact binomial inference:")
    print(inadequate[['group', 'n_trials', 'r_successes', 'success_rate', 'np', 'nq']])

# 3. CHECK FOR INDEPENDENCE VIOLATIONS
print("\n3. TEMPORAL/SPATIAL STRUCTURE CHECK")
print("-" * 80)

# Check if there's autocorrelation in success rates by group order
if len(df) > 2:
    # Lag-1 autocorrelation
    success_rates = df.sort_values('group')['success_rate'].values
    lag1_corr = np.corrcoef(success_rates[:-1], success_rates[1:])[0, 1]
    print(f"Lag-1 autocorrelation in success rates: {lag1_corr:.4f}")

    # Test significance (rough test)
    se = 1 / np.sqrt(len(df) - 1)
    z_score = lag1_corr / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
    print(f"  Z-score: {z_score:.4f}, P-value: {p_val:.4f}")
    print(f"  Conclusion: {'Significant autocorrelation detected' if p_val < 0.05 else 'No significant autocorrelation'}")

    # Runs test on residuals
    residual_signs = (df.sort_values('group')['pearson_residual'] > 0).astype(int)
    runs = 1 + np.sum(np.abs(np.diff(residual_signs)))
    n_pos = residual_signs.sum()
    n_neg = len(residual_signs) - n_pos

    if n_pos > 0 and n_neg > 0:
        expected_runs = 1 + (2 * n_pos * n_neg) / (n_pos + n_neg)
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / ((n_pos + n_neg)**2 * (n_pos + n_neg - 1))
        z_runs = (runs - expected_runs) / np.sqrt(var_runs)
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

        print(f"\nRuns test on Pearson residuals:")
        print(f"  Observed runs: {runs}")
        print(f"  Expected runs: {expected_runs:.2f}")
        print(f"  Z-score: {z_runs:.4f}, P-value: {p_runs:.4f}")
        print(f"  Conclusion: {'Evidence of non-randomness' if p_runs < 0.05 else 'Residuals appear random'}")

# 4. EXAMINE TRIAL SIZE PATTERNS
print("\n4. TRIAL SIZE PATTERNS")
print("-" * 80)

# Check correlation between trial size and success rate
corr_n_p = stats.spearmanr(df['n_trials'], df['success_rate'])
print(f"Spearman correlation between n_trials and success_rate:")
print(f"  rho = {corr_n_p.statistic:.4f}, p-value = {corr_n_p.pvalue:.4f}")
print(f"  Conclusion: {'Significant relationship' if corr_n_p.pvalue < 0.05 else 'No significant relationship'}")

# Check correlation between trial size and absolute residuals (heteroscedasticity check)
corr_n_resid = stats.spearmanr(df['n_trials'], np.abs(df['pearson_residual']))
print(f"\nSpearman correlation between n_trials and |Pearson residual|:")
print(f"  rho = {corr_n_resid.statistic:.4f}, p-value = {corr_n_resid.pvalue:.4f}")
print(f"  Conclusion: {'Evidence of heteroscedasticity' if corr_n_resid.pvalue < 0.05 else 'No strong evidence of heteroscedasticity'}")

# 5. ZERO-INFLATION CHECK
print("\n5. ZERO-INFLATION CHECK")
print("-" * 80)

n_zeros = (df['r_successes'] == 0).sum()
n_ones = (df['r_successes'] == 1).sum()

print(f"Groups with 0 successes: {n_zeros} ({100*n_zeros/len(df):.1f}%)")
print(f"Groups with 1 success: {n_ones} ({100*n_ones/len(df):.1f}%)")

# Expected number of zeros under pooled binomial model
expected_prob_zero = np.mean((1 - pooled_p) ** df['n_trials'])
expected_n_zeros = len(df) * expected_prob_zero
print(f"\nExpected number of groups with 0 successes (under pooled model): {expected_n_zeros:.2f}")
print(f"Observed: {n_zeros}")
print(f"Conclusion: {'Possible zero-inflation' if n_zeros > expected_n_zeros * 2 else 'No strong evidence of zero-inflation'}")

# Save augmented data
df.to_csv(OUTPUT_DIR / 'code' / 'data_with_diagnostics.csv', index=False)

print("\n" + "="*80)
print("BINOMIAL ASSUMPTIONS SUMMARY")
print("="*80)
print(f"Overdispersion parameter: {dispersion:.4f}")
print(f"Chi-square test p-value: {p_value:.4f}")
print(f"Groups with inadequate sample size: {len(inadequate)}")
print(f"Zero-inflation concern: {'Yes' if n_zeros > expected_n_zeros * 2 else 'No'}")
print(f"Autocorrelation concern: {'Yes' if abs(lag1_corr) > 2*se else 'No'}")
