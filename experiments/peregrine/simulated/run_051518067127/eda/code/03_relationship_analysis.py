"""
Relationship Analysis: Count vs Year
====================================
Goal: Assess linearity, trend patterns, and variance structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

def lowess_smooth(x, y, frac=0.3):
    """Simple LOESS-like smoother using rolling windows"""
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    n = len(x)
    window = int(frac * n)
    if window < 3:
        window = 3

    smoothed = np.zeros(n)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        smoothed[i] = np.mean(y_sorted[start:end])

    return x_sorted, smoothed

# Load data
data = pd.read_csv('/workspace/data/data.csv')

# Create comprehensive relationship plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Relationship Analysis: Count vs Year', fontsize=16, y=0.995)

# 1. Basic scatterplot with linear fit
ax = axes[0, 0]
ax.scatter(data['year'], data['C'], alpha=0.6, s=60, color='steelblue', edgecolor='black', linewidth=0.5)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(data['year'], data['C'])
line_x = np.linspace(data['year'].min(), data['year'].max(), 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=2, label=f'Linear fit: R²={r_value**2:.3f}')

ax.set_xlabel('Standardized Year')
ax.set_ylabel('Count (C)')
ax.set_title('Scatterplot with Linear Trend')
ax.legend()
ax.grid(alpha=0.3)

# Add text with regression stats
textstr = f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\np-value: {p_value:.2e}'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Scatterplot with smoothing
ax = axes[0, 1]
ax.scatter(data['year'], data['C'], alpha=0.6, s=60, color='steelblue', edgecolor='black', linewidth=0.5)

# Simple smoother
x_smooth, y_smooth = lowess_smooth(data['year'].values, data['C'].values, frac=0.3)
ax.plot(x_smooth, y_smooth, 'g-', linewidth=2.5, label='Smoothed trend')

# Also plot linear for comparison
ax.plot(line_x, line_y, 'r--', linewidth=1.5, alpha=0.7, label='Linear fit')

ax.set_xlabel('Standardized Year')
ax.set_ylabel('Count (C)')
ax.set_title('Scatterplot with Smoothed Trend')
ax.legend()
ax.grid(alpha=0.3)

# 3. Log-scale relationship
ax = axes[1, 0]
log_C = np.log(data['C'])
ax.scatter(data['year'], log_C, alpha=0.6, s=60, color='mediumseagreen', edgecolor='black', linewidth=0.5)

# Linear fit on log scale
slope_log, intercept_log, r_value_log, p_value_log, std_err_log = stats.linregress(data['year'], log_C)
line_y_log = slope_log * line_x + intercept_log
ax.plot(line_x, line_y_log, 'r--', linewidth=2, label=f'Linear fit: R²={r_value_log**2:.3f}')

# Smoother on log scale
x_smooth_log, y_smooth_log = lowess_smooth(data['year'].values, log_C.values, frac=0.3)
ax.plot(x_smooth_log, y_smooth_log, 'orange', linewidth=2.5, label='Smoothed')

ax.set_xlabel('Standardized Year')
ax.set_ylabel('log(Count)')
ax.set_title('Log-Scale Relationship')
ax.legend()
ax.grid(alpha=0.3)

textstr = f'Slope: {slope_log:.3f}\nR²: {r_value_log**2:.3f}\np-value: {p_value_log:.2e}'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Residuals vs fitted (heteroscedasticity check)
ax = axes[1, 1]
fitted = slope * data['year'] + intercept
residuals = data['C'] - fitted
ax.scatter(fitted, residuals, alpha=0.6, s=60, color='purple', edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

# Add smoother to residuals to check for patterns
x_smooth_resid, y_smooth_resid = lowess_smooth(fitted.values, residuals.values, frac=0.5)
ax.plot(x_smooth_resid, y_smooth_resid, 'orange', linewidth=2.5, label='Smoothed trend')

ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted (Heteroscedasticity Check)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/02_relationship_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: 02_relationship_analysis.png")
plt.close()

# Statistical summary
print("\n" + "="*80)
print("RELATIONSHIP ANALYSIS SUMMARY")
print("="*80)

print("\n1. LINEAR REGRESSION (Original Scale)")
print("-" * 80)
print(f"  Equation: C = {slope:.4f} * year + {intercept:.4f}")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  Correlation: {r_value:.4f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Standard error: {std_err:.4f}")
print(f"  95% CI for slope: [{slope - 1.96*std_err:.4f}, {slope + 1.96*std_err:.4f}]")

print("\n2. LINEAR REGRESSION (Log Scale)")
print("-" * 80)
print(f"  Equation: log(C) = {slope_log:.4f} * year + {intercept_log:.4f}")
print(f"  R-squared: {r_value_log**2:.4f}")
print(f"  Correlation: {r_value_log:.4f}")
print(f"  P-value: {p_value_log:.2e}")
print(f"  Interpretation: Each unit increase in year multiplies C by {np.exp(slope_log):.4f}")

print("\n3. NONLINEARITY ASSESSMENT")
print("-" * 80)
# Polynomial fits
X = data['year'].values
y = data['C'].values

# Linear
linear_pred = slope * X + intercept
ss_res_linear = np.sum((y - linear_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2_linear = 1 - ss_res_linear / ss_tot

# Quadratic
coeffs_2 = np.polyfit(X, y, 2)
poly2_pred = np.polyval(coeffs_2, X)
ss_res_poly2 = np.sum((y - poly2_pred)**2)
r2_poly2 = 1 - ss_res_poly2 / ss_tot

# Cubic
coeffs_3 = np.polyfit(X, y, 3)
poly3_pred = np.polyval(coeffs_3, X)
ss_res_poly3 = np.sum((y - poly3_pred)**2)
r2_poly3 = 1 - ss_res_poly3 / ss_tot

print(f"  Linear R²: {r2_linear:.4f}")
print(f"  Quadratic R²: {r2_poly2:.4f} (improvement: {r2_poly2 - r2_linear:.4f})")
print(f"  Cubic R²: {r2_poly3:.4f} (improvement: {r2_poly3 - r2_poly2:.4f})")

print("\n4. HETEROSCEDASTICITY TESTS")
print("-" * 80)
# Breusch-Pagan test
from scipy.stats import chi2
residuals_squared = residuals**2
bp_slope, bp_intercept, bp_r, bp_p, bp_se = stats.linregress(fitted, residuals_squared)
n = len(data)
LM = n * bp_r**2
bp_pvalue = 1 - chi2.cdf(LM, 1)
print(f"  Breusch-Pagan test:")
print(f"    LM statistic: {LM:.4f}")
print(f"    P-value: {bp_pvalue:.4f}")
print(f"    Conclusion: {'Heteroscedasticity detected' if bp_pvalue < 0.05 else 'No significant heteroscedasticity'}")

# Correlation between absolute residuals and fitted values
abs_residuals = np.abs(residuals)
corr_abs_resid = stats.pearsonr(fitted, abs_residuals)
print(f"\n  Correlation(|residuals|, fitted):")
print(f"    Correlation: {corr_abs_resid[0]:.4f}")
print(f"    P-value: {corr_abs_resid[1]:.4f}")
print(f"    Conclusion: {'Variance increases with mean' if corr_abs_resid[0] > 0 and corr_abs_resid[1] < 0.05 else 'No clear variance pattern'}")

print("\n" + "="*80)
