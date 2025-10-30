"""
Hypothesis Testing and Additional Analysis
===========================================
Test competing hypotheses about the data structure.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Paths
DATA_PATH = '/workspace/data/data.csv'

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("HYPOTHESIS TESTING AND ALTERNATIVE EXPLANATIONS")
print("=" * 80)

# ============================================================================
# HYPOTHESIS 1: Is the growth LINEAR or EXPONENTIAL?
# ============================================================================
print("\n1. LINEAR vs EXPONENTIAL GROWTH")
print("-" * 40)

# Linear model
X = df['year'].values
y = df['C'].values
slope_linear, intercept_linear = np.polyfit(X, y, 1)
y_pred_linear = slope_linear * X + intercept_linear
ss_res_linear = np.sum((y - y_pred_linear)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_linear = 1 - (ss_res_linear / ss_tot)
aic_linear = len(y) * np.log(ss_res_linear / len(y)) + 2 * 2  # 2 parameters

# Exponential model (via log transformation)
log_y = np.log(y)
slope_exp, intercept_exp = np.polyfit(X, log_y, 1)
y_pred_exp = np.exp(intercept_exp) * np.exp(slope_exp * X)
ss_res_exp = np.sum((y - y_pred_exp)**2)
r2_exp = 1 - (ss_res_exp / ss_tot)
aic_exp = len(y) * np.log(ss_res_exp / len(y)) + 2 * 2

# Quadratic model
z_quad = np.polyfit(X, y, 2)
p_quad = np.poly1d(z_quad)
y_pred_quad = p_quad(X)
ss_res_quad = np.sum((y - y_pred_quad)**2)
r2_quad = 1 - (ss_res_quad / ss_tot)
aic_quad = len(y) * np.log(ss_res_quad / len(y)) + 2 * 3  # 3 parameters

print(f"\nLinear Model:")
print(f"  R² = {r2_linear:.4f}")
print(f"  RMSE = {np.sqrt(ss_res_linear/len(y)):.2f}")
print(f"  AIC = {aic_linear:.2f}")
print(f"  Equation: C = {slope_linear:.2f} * year + {intercept_linear:.2f}")

print(f"\nExponential Model:")
print(f"  R² = {r2_exp:.4f}")
print(f"  RMSE = {np.sqrt(ss_res_exp/len(y)):.2f}")
print(f"  AIC = {aic_exp:.2f}")
print(f"  Equation: C = {np.exp(intercept_exp):.2f} * exp({slope_exp:.3f} * year)")
print(f"  Growth rate: {(np.exp(slope_exp) - 1) * 100:.1f}% per unit year")

print(f"\nQuadratic Model:")
print(f"  R² = {r2_quad:.4f}")
print(f"  RMSE = {np.sqrt(ss_res_quad/len(y)):.2f}")
print(f"  AIC = {aic_quad:.2f}")
print(f"  Equation: C = {z_quad[0]:.2f}*year² + {z_quad[1]:.2f}*year + {z_quad[2]:.2f}")

print(f"\nConclusion:")
print(f"  Best model by AIC: {'Linear' if aic_linear < min(aic_exp, aic_quad) else ('Exponential' if aic_exp < aic_quad else 'Quadratic')}")
print(f"  Delta AIC (Linear - Exponential): {aic_linear - aic_exp:.2f}")
print(f"  Delta AIC (Linear - Quadratic): {aic_linear - aic_quad:.2f}")

# ============================================================================
# HYPOTHESIS 2: Is there a CHANGEPOINT in the trend?
# ============================================================================
print("\n\n2. CHANGEPOINT DETECTION")
print("-" * 40)

# Test for changepoint at median year
median_idx = len(df) // 2
df['period'] = ['early' if i < median_idx else 'late' for i in range(len(df))]

early_data = df[df['period'] == 'early']['C']
late_data = df[df['period'] == 'late']['C']

# Compare means
t_stat, t_pval = stats.ttest_ind(early_data, late_data)
mann_stat, mann_pval = stats.mannwhitneyu(early_data, late_data)

print(f"\nSplit at median year (n1={len(early_data)}, n2={len(late_data)}):")
print(f"  Early period mean: {early_data.mean():.2f} (SD: {early_data.std():.2f})")
print(f"  Late period mean: {late_data.mean():.2f} (SD: {late_data.std():.2f})")
print(f"  t-test: t={t_stat:.3f}, p={t_pval:.4e}")
print(f"  Mann-Whitney U: U={mann_stat:.1f}, p={mann_pval:.4e}")

# Fit separate linear models to each period
early_X = df[df['period'] == 'early']['year'].values
early_y = df[df['period'] == 'early']['C'].values
slope_early, intercept_early = np.polyfit(early_X, early_y, 1)

late_X = df[df['period'] == 'late']['year'].values
late_y = df[df['period'] == 'late']['C'].values
slope_late, intercept_late = np.polyfit(late_X, late_y, 1)

print(f"\n  Early period slope: {slope_early:.2f} counts/year")
print(f"  Late period slope: {slope_late:.2f} counts/year")
print(f"  Slope ratio (late/early): {slope_late/slope_early:.2f}x")

# ============================================================================
# HYPOTHESIS 3: Is the VARIANCE truly HETEROSCEDASTIC?
# ============================================================================
print("\n\n3. HETEROSCEDASTICITY TESTS")
print("-" * 40)

# Breusch-Pagan test (simplified)
residuals = y - y_pred_linear
# Regress squared residuals on predictor
slope_bp, intercept_bp = np.polyfit(X, residuals**2, 1)
y_pred_bp = slope_bp * X + intercept_bp
ss_res_bp = np.sum((residuals**2 - y_pred_bp)**2)
ss_tot_bp = np.sum((residuals**2 - (residuals**2).mean())**2)
r2_bp = 1 - (ss_res_bp / ss_tot_bp)

# Correlation between fitted values and absolute residuals
corr_het, pval_het = stats.pearsonr(y_pred_linear, np.abs(residuals))

print(f"\nResidual analysis:")
print(f"  Correlation(|residuals|, fitted): r={corr_het:.3f}, p={pval_het:.4f}")
print(f"  R² for residuals² ~ year: {r2_bp:.4f}")
if pval_het < 0.05:
    print(f"  Conclusion: Significant heteroscedasticity detected")
else:
    print(f"  Conclusion: No strong evidence of heteroscedasticity")

# Split into quartiles and compare variance
df_copy = df.copy()
df_copy['quartile'] = pd.qcut(df_copy['year'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
quartile_vars = df_copy.groupby('quartile')['C'].var()

print(f"\nVariance by quartile:")
for q, var in quartile_vars.items():
    print(f"  {q}: {var:.2f}")

# Levene's test for equality of variances
q1 = df_copy[df_copy['quartile'] == 'Q1']['C']
q2 = df_copy[df_copy['quartile'] == 'Q2']['C']
q3 = df_copy[df_copy['quartile'] == 'Q3']['C']
q4 = df_copy[df_copy['quartile'] == 'Q4']['C']
levene_stat, levene_pval = stats.levene(q1, q2, q3, q4)
print(f"\n  Levene's test: stat={levene_stat:.3f}, p={levene_pval:.4f}")

# ============================================================================
# HYPOTHESIS 4: Is there AUTOCORRELATION (temporal dependence)?
# ============================================================================
print("\n\n4. TEMPORAL DEPENDENCE (AUTOCORRELATION)")
print("-" * 40)

# Durbin-Watson statistic for autocorrelation
dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"\nDurbin-Watson statistic: {dw_stat:.3f}")
print(f"  Interpretation: ", end="")
if dw_stat < 1.5:
    print("Positive autocorrelation (DW < 1.5)")
elif dw_stat > 2.5:
    print("Negative autocorrelation (DW > 2.5)")
else:
    print("No strong autocorrelation (1.5 ≤ DW ≤ 2.5)")

# Lag-1 autocorrelation
lag1_corr, lag1_pval = stats.pearsonr(df['C'][:-1], df['C'][1:])
print(f"\nLag-1 autocorrelation: r={lag1_corr:.3f}, p={lag1_pval:.4f}")

# Runs test for randomness
median_C = df['C'].median()
runs = np.diff(df['C'] > median_C).sum() + 1
expected_runs = 1 + (2 * np.sum(df['C'] > median_C) * np.sum(df['C'] <= median_C)) / len(df)
print(f"\nRuns test:")
print(f"  Observed runs: {runs}")
print(f"  Expected runs (random): {expected_runs:.1f}")

# ============================================================================
# HYPOTHESIS 5: Are there OUTLIERS or INFLUENTIAL POINTS?
# ============================================================================
print("\n\n5. OUTLIER AND INFLUENCE ANALYSIS")
print("-" * 40)

# Standardized residuals
std_residuals = (residuals - residuals.mean()) / residuals.std()
outliers_std = np.abs(std_residuals) > 2.5
n_outliers = outliers_std.sum()

print(f"\nStandardized residuals > 2.5: {n_outliers} points")
if n_outliers > 0:
    outlier_indices = np.where(outliers_std)[0]
    print(f"  Indices: {outlier_indices}")
    print(f"  Years: {df.iloc[outlier_indices]['year'].values}")
    print(f"  Counts: {df.iloc[outlier_indices]['C'].values}")
    print(f"  Std residuals: {std_residuals[outliers_std]}")

# Cook's distance (simplified approximation)
# D_i ≈ (residuals_i^2 / (p * MSE)) * (leverage_i / (1-leverage_i)^2)
# where leverage_i = 1/n + (x_i - x_mean)^2 / sum((x_j - x_mean)^2)
n = len(X)
p = 2  # number of parameters
MSE = ss_res_linear / (n - p)
leverage = 1/n + (X - X.mean())**2 / np.sum((X - X.mean())**2)
cooks_d = (residuals**2 / (p * MSE)) * (leverage / (1 - leverage)**2)

influential = cooks_d > 4/n  # Common threshold
n_influential = influential.sum()

print(f"\nCook's distance > 4/n ({4/n:.3f}): {n_influential} points")
if n_influential > 0:
    inf_indices = np.where(influential)[0]
    print(f"  Indices: {inf_indices}")
    print(f"  Cook's D: {cooks_d[influential]}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n\n6. COMPREHENSIVE SUMMARY")
print("-" * 40)
print(f"\nData characteristics:")
print(f"  Sample size: {len(df)}")
print(f"  Time span: {df['year'].min():.3f} to {df['year'].max():.3f}")
print(f"  Count range: {df['C'].min()} to {df['C'].max()}")
print(f"  Overall mean: {df['C'].mean():.2f}")
print(f"  Overall variance: {df['C'].var():.2f}")
print(f"  Variance-to-mean ratio: {df['C'].var() / df['C'].mean():.2f}")
print(f"  Coefficient of variation: {df['C'].std() / df['C'].mean():.2f}")

print(f"\nTrend characteristics:")
print(f"  Pearson correlation (year, C): {stats.pearsonr(df['year'], df['C'])[0]:.4f}")
print(f"  Spearman correlation: {stats.spearmanr(df['year'], df['C'])[0]:.4f}")
print(f"  Linear R²: {r2_linear:.4f}")
print(f"  Exponential R²: {r2_exp:.4f}")
print(f"  Quadratic R²: {r2_quad:.4f}")

print("\n" + "=" * 80)
