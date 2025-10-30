"""
Residual Analysis and Model Diagnostics
Testing simple models and examining residual patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')

print("="*80)
print("RESIDUAL DIAGNOSTICS FROM SIMPLE MODELS")
print("="*80)

# Model 1: Linear regression C ~ year using numpy
X = data['year'].values
y = data['C'].values

# Add intercept term
X_with_intercept = np.column_stack([np.ones(len(X)), X])

# Fit using least squares
beta_linear = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
y_pred_linear = X_with_intercept @ beta_linear
residuals_linear = y - y_pred_linear

# Calculate R²
ss_res = np.sum(residuals_linear ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2_linear = 1 - (ss_res / ss_tot)

print("\nModel 1: Linear Regression C ~ year")
print(f"  Intercept: {beta_linear[0]:.4f}")
print(f"  Coefficient (slope): {beta_linear[1]:.4f}")
print(f"  R²: {r2_linear:.4f}")
print(f"  Mean residual: {residuals_linear.mean():.6f}")
print(f"  Std residual: {residuals_linear.std():.4f}")

# Test for heteroscedasticity - Breusch-Pagan test
residuals_squared = residuals_linear ** 2
beta_het = np.linalg.lstsq(X_with_intercept, residuals_squared, rcond=None)[0]
y_pred_het = X_with_intercept @ beta_het
ss_explained = np.sum((y_pred_het - residuals_squared.mean()) ** 2)
ss_total = np.sum((residuals_squared - residuals_squared.mean()) ** 2)
r2_het = ss_explained / ss_total
bp_statistic = len(X) * r2_het
bp_pvalue = 1 - stats.chi2.cdf(bp_statistic, 1)

print(f"\n  Breusch-Pagan Test for Heteroscedasticity:")
print(f"    Test statistic: {bp_statistic:.4f}")
print(f"    p-value: {bp_pvalue:.4f}")
print(f"    Result: {'HETEROSCEDASTIC' if bp_pvalue < 0.05 else 'Homoscedastic'} (α=0.05)")

# Shapiro-Wilk test for normality of residuals
sw_stat, sw_pvalue = stats.shapiro(residuals_linear)
print(f"\n  Shapiro-Wilk Test for Normality of Residuals:")
print(f"    Test statistic: {sw_stat:.4f}")
print(f"    p-value: {sw_pvalue:.4f}")
print(f"    Result: {'NON-NORMAL' if sw_pvalue < 0.05 else 'Normal'} (α=0.05)")

# Model 2: Linear regression log(C) ~ year
y_log = np.log(y)
beta_log = np.linalg.lstsq(X_with_intercept, y_log, rcond=None)[0]
y_pred_log = X_with_intercept @ beta_log
residuals_log = y_log - y_pred_log

ss_res_log = np.sum(residuals_log ** 2)
ss_tot_log = np.sum((y_log - y_log.mean()) ** 2)
r2_log = 1 - (ss_res_log / ss_tot_log)

print("\n\nModel 2: Linear Regression log(C) ~ year")
print(f"  Intercept: {beta_log[0]:.4f}")
print(f"  Coefficient (slope): {beta_log[1]:.4f}")
print(f"  R²: {r2_log:.4f}")
print(f"  Mean residual: {residuals_log.mean():.6f}")
print(f"  Std residual: {residuals_log.std():.4f}")

# Test for heteroscedasticity on log model
residuals_log_squared = residuals_log ** 2
beta_het_log = np.linalg.lstsq(X_with_intercept, residuals_log_squared, rcond=None)[0]
y_pred_het_log = X_with_intercept @ beta_het_log
ss_explained_log = np.sum((y_pred_het_log - residuals_log_squared.mean()) ** 2)
ss_total_log = np.sum((residuals_log_squared - residuals_log_squared.mean()) ** 2)
r2_het_log = ss_explained_log / ss_total_log
bp_statistic_log = len(X) * r2_het_log
bp_pvalue_log = 1 - stats.chi2.cdf(bp_statistic_log, 1)

print(f"\n  Breusch-Pagan Test for Heteroscedasticity:")
print(f"    Test statistic: {bp_statistic_log:.4f}")
print(f"    p-value: {bp_pvalue_log:.4f}")
print(f"    Result: {'HETEROSCEDASTIC' if bp_pvalue_log < 0.05 else 'Homoscedastic'} (α=0.05)")

# Shapiro-Wilk test for normality of log model residuals
sw_stat_log, sw_pvalue_log = stats.shapiro(residuals_log)
print(f"\n  Shapiro-Wilk Test for Normality of Residuals:")
print(f"    Test statistic: {sw_stat_log:.4f}")
print(f"    p-value: {sw_pvalue_log:.4f}")
print(f"    Result: {'NON-NORMAL' if sw_pvalue_log < 0.05 else 'Normal'} (α=0.05)")

# Model 3: sqrt transformation
y_sqrt = np.sqrt(y)
beta_sqrt = np.linalg.lstsq(X_with_intercept, y_sqrt, rcond=None)[0]
y_pred_sqrt = X_with_intercept @ beta_sqrt
residuals_sqrt = y_sqrt - y_pred_sqrt

ss_res_sqrt = np.sum(residuals_sqrt ** 2)
ss_tot_sqrt = np.sum((y_sqrt - y_sqrt.mean()) ** 2)
r2_sqrt = 1 - (ss_res_sqrt / ss_tot_sqrt)

print("\n\nModel 3: Linear Regression sqrt(C) ~ year")
print(f"  Intercept: {beta_sqrt[0]:.4f}")
print(f"  Coefficient (slope): {beta_sqrt[1]:.4f}")
print(f"  R²: {r2_sqrt:.4f}")
print(f"  Mean residual: {residuals_sqrt.mean():.6f}")
print(f"  Std residual: {residuals_sqrt.std():.4f}")

# Test for heteroscedasticity on sqrt model
residuals_sqrt_squared = residuals_sqrt ** 2
beta_het_sqrt = np.linalg.lstsq(X_with_intercept, residuals_sqrt_squared, rcond=None)[0]
y_pred_het_sqrt = X_with_intercept @ beta_het_sqrt
ss_explained_sqrt = np.sum((y_pred_het_sqrt - residuals_sqrt_squared.mean()) ** 2)
ss_total_sqrt = np.sum((residuals_sqrt_squared - residuals_sqrt_squared.mean()) ** 2)
r2_het_sqrt = ss_explained_sqrt / ss_total_sqrt
bp_statistic_sqrt = len(X) * r2_het_sqrt
bp_pvalue_sqrt = 1 - stats.chi2.cdf(bp_statistic_sqrt, 1)

print(f"\n  Breusch-Pagan Test for Heteroscedasticity:")
print(f"    Test statistic: {bp_statistic_sqrt:.4f}")
print(f"    p-value: {bp_pvalue_sqrt:.4f}")
print(f"    Result: {'HETEROSCEDASTIC' if bp_pvalue_sqrt < 0.05 else 'Homoscedastic'} (α=0.05)")

# Shapiro-Wilk test for normality of sqrt model residuals
sw_stat_sqrt, sw_pvalue_sqrt = stats.shapiro(residuals_sqrt)
print(f"\n  Shapiro-Wilk Test for Normality of Residuals:")
print(f"    Test statistic: {sw_stat_sqrt:.4f}")
print(f"    p-value: {sw_pvalue_sqrt:.4f}")
print(f"    Result: {'NON-NORMAL' if sw_pvalue_sqrt < 0.05 else 'Normal'} (α=0.05)")

# Save results to file for later use
np.savez('/workspace/eda/analyst_3/code/model_results.npz',
         residuals_linear=residuals_linear,
         fitted_linear=y_pred_linear,
         residuals_log=residuals_log,
         fitted_log=y_pred_log,
         residuals_sqrt=residuals_sqrt,
         fitted_sqrt=y_pred_sqrt,
         year=X,
         C=y,
         r2_linear=r2_linear,
         r2_log=r2_log,
         r2_sqrt=r2_sqrt,
         bp_pvalue_linear=bp_pvalue,
         bp_pvalue_log=bp_pvalue_log,
         bp_pvalue_sqrt=bp_pvalue_sqrt,
         sw_pvalue_linear=sw_pvalue,
         sw_pvalue_log=sw_pvalue_log,
         sw_pvalue_sqrt=sw_pvalue_sqrt)

print("\n" + "="*80)
print("Residual diagnostics complete. Results saved.")
print("="*80)
