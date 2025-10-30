"""
Correlation Structure Analysis
Focus: Relationships between n_trials and r_successes, correlation patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_2.csv')
df['failure_count'] = df['n_trials'] - df['r_successes']
df['logit_success_rate'] = np.log((df['r_successes'] + 0.5) / (df['failure_count'] + 0.5))

print("="*80)
print("CORRELATION STRUCTURE ANALYSIS")
print("="*80)

# 1. DETAILED CORRELATION ANALYSIS
print("\n1. CORRELATION MATRIX (Pearson)")
print("-"*80)
vars_of_interest = ['n_trials', 'r_successes', 'success_rate', 'failure_count', 'logit_success_rate']
corr_pearson = df[vars_of_interest].corr()
print(corr_pearson.round(4))

print("\n2. CORRELATION MATRIX (Spearman)")
print("-"*80)
corr_spearman = df[vars_of_interest].corr(method='spearman')
print(corr_spearman.round(4))

# 3. STATISTICAL TESTS FOR KEY CORRELATIONS
print("\n3. STATISTICAL SIGNIFICANCE OF KEY CORRELATIONS")
print("-"*80)

# n_trials vs r_successes
r_trials_success, p_trials_success = stats.pearsonr(df['n_trials'], df['r_successes'])
rho_ts, p_ts_spearman = spearmanr(df['n_trials'], df['r_successes'])
print(f"\nn_trials vs r_successes:")
print(f"  Pearson: r={r_trials_success:.4f}, p={p_trials_success:.4f}")
print(f"  Spearman: rho={rho_ts:.4f}, p={p_ts_spearman:.4f}")
print(f"  Interpretation: {'Strong positive' if abs(r_trials_success) > 0.7 else 'Moderate'} linear relationship")

# n_trials vs success_rate
r_trials_rate, p_trials_rate = stats.pearsonr(df['n_trials'], df['success_rate'])
rho_tr, p_tr_spearman = spearmanr(df['n_trials'], df['success_rate'])
print(f"\nn_trials vs success_rate:")
print(f"  Pearson: r={r_trials_rate:.4f}, p={p_trials_rate:.4f}")
print(f"  Spearman: rho={rho_tr:.4f}, p={p_tr_spearman:.4f}")
print(f"  Interpretation: Negative relationship (more trials → lower success rate)")

# r_successes vs success_rate
r_succ_rate, p_succ_rate = stats.pearsonr(df['r_successes'], df['success_rate'])
rho_sr, p_sr_spearman = spearmanr(df['r_successes'], df['success_rate'])
print(f"\nr_successes vs success_rate:")
print(f"  Pearson: r={r_succ_rate:.4f}, p={p_succ_rate:.4f}")
print(f"  Spearman: rho={rho_sr:.4f}, p={p_sr_spearman:.4f}")
print(f"  Interpretation: Weak positive (success count ≈ independent of rate)")

# 4. PARTIAL CORRELATION (controlling for confounders)
print("\n4. PARTIAL CORRELATIONS")
print("-"*80)

def partial_corr(data, x, y, z):
    """Calculate partial correlation of x and y controlling for z"""
    # Residuals of x ~ z
    slope_xz, intercept_xz = np.polyfit(data[z], data[x], 1)
    resid_x = data[x] - (slope_xz * data[z] + intercept_xz)

    # Residuals of y ~ z
    slope_yz, intercept_yz = np.polyfit(data[z], data[y], 1)
    resid_y = data[y] - (slope_yz * data[z] + intercept_yz)

    # Correlation of residuals
    return stats.pearsonr(resid_x, resid_y)

# Partial correlation: r_successes vs success_rate, controlling for n_trials
r_partial, p_partial = partial_corr(df, 'r_successes', 'success_rate', 'n_trials')
print(f"r_successes vs success_rate (controlling for n_trials):")
print(f"  Partial r={r_partial:.4f}, p={p_partial:.4f}")
print(f"  Original r={r_succ_rate:.4f}")
print(f"  Change: {r_partial - r_succ_rate:.4f}")

# 5. REGRESSION ANALYSIS: n_trials predicting r_successes
print("\n5. REGRESSION: n_trials → r_successes")
print("-"*80)

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['n_trials'], df['r_successes'])

print(f"Model: r_successes = {intercept:.4f} + {slope:.4f} * n_trials")
print(f"R-squared: {r_value**2:.4f}")
print(f"Slope p-value: {p_value:.4f}")
print(f"Standard error: {std_err:.4f}")

# Calculate residuals
df['predicted_successes'] = intercept + slope * df['n_trials']
df['residuals'] = df['r_successes'] - df['predicted_successes']

print(f"\nResidual statistics:")
print(f"  Mean: {df['residuals'].mean():.4f} (should be ~0)")
print(f"  Std: {df['residuals'].std():.4f}")
print(f"  Min: {df['residuals'].min():.4f}")
print(f"  Max: {df['residuals'].max():.4f}")

# Test for homoscedasticity (Breusch-Pagan test approximation)
# Regress squared residuals on n_trials
resid_sq = df['residuals']**2
slope_bp, intercept_bp, r_bp, p_bp, se_bp = linregress(df['n_trials'], resid_sq)
print(f"\nHomoscedasticity test (residuals² vs n_trials):")
print(f"  Slope p-value: {p_bp:.4f}")
print(f"  Interpretation: {'Heteroscedastic' if p_bp < 0.05 else 'Homoscedastic'}")

# 6. OUTLIER INFLUENCE ANALYSIS
print("\n6. OUTLIER INFLUENCE ON CORRELATIONS")
print("-"*80)

# Identify outlier (group 4 with 810 trials)
outlier_idx = df['n_trials'].idxmax()
df_no_outlier = df.drop(outlier_idx)

print(f"Outlier: Group {df.loc[outlier_idx, 'group_id']} (n_trials={df.loc[outlier_idx, 'n_trials']})")

# Recalculate correlations without outlier
r_without, p_without = stats.pearsonr(df_no_outlier['n_trials'], df_no_outlier['r_successes'])
print(f"\nn_trials vs r_successes:")
print(f"  With outlier: r={r_trials_success:.4f}")
print(f"  Without outlier: r={r_without:.4f}")
print(f"  Change: {r_without - r_trials_success:.4f}")

r_rate_without, p_rate_without = stats.pearsonr(df_no_outlier['n_trials'], df_no_outlier['success_rate'])
print(f"\nn_trials vs success_rate:")
print(f"  With outlier: r={r_trials_rate:.4f}")
print(f"  Without outlier: r={r_rate_without:.4f}")
print(f"  Change: {r_rate_without - r_trials_rate:.4f}")

# 7. NON-LINEAR RELATIONSHIP EXPLORATION
print("\n7. NON-LINEAR RELATIONSHIP TESTS")
print("-"*80)

# Test for quadratic relationship
from numpy.polynomial import polynomial as P

# Fit quadratic model
coeffs_quad = np.polyfit(df['n_trials'], df['r_successes'], 2)
poly_quad = np.poly1d(coeffs_quad)
y_pred_quad = poly_quad(df['n_trials'])
ss_res_quad = np.sum((df['r_successes'] - y_pred_quad)**2)
ss_tot = np.sum((df['r_successes'] - df['r_successes'].mean())**2)
r2_quad = 1 - (ss_res_quad / ss_tot)

print(f"Quadratic fit: R² = {r2_quad:.4f}")
print(f"Linear fit: R² = {r_value**2:.4f}")
print(f"Improvement: {r2_quad - r_value**2:.4f}")

# Test logarithmic relationship
log_trials = np.log(df['n_trials'])
r_log, p_log = stats.pearsonr(log_trials, df['r_successes'])
print(f"\nLog(n_trials) vs r_successes: r={r_log:.4f}, p={p_log:.4f}")

# 8. VARIANCE EXPLAINED
print("\n8. VARIANCE DECOMPOSITION")
print("-"*80)

# What proportion of variance in success_rate is explained by n_trials?
r_sq_trials_rate = r_trials_rate**2
print(f"Variance in success_rate explained by n_trials: {r_sq_trials_rate*100:.2f}%")
print(f"Unexplained variance: {(1-r_sq_trials_rate)*100:.2f}%")

# What about r_successes?
r_sq_trials_succ = r_trials_success**2
print(f"\nVariance in r_successes explained by n_trials: {r_sq_trials_succ*100:.2f}%")
print(f"Unexplained variance: {(1-r_sq_trials_succ)*100:.2f}%")

print("\n" + "="*80)
print("CORRELATION ANALYSIS COMPLETE")
print("="*80)

# Save processed data for clustering
df.to_csv('/workspace/eda/analyst_2/code/processed_data.csv', index=False)
print("\nProcessed data saved for clustering analysis")
