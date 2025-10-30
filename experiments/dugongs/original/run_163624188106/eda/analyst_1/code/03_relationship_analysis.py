"""
Relationship Analysis - Analyst 1
==================================
Purpose: Examine the relationship between x and Y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("="*60)
print("RELATIONSHIP ANALYSIS")
print("="*60)

# Correlation analysis
print("\n1. CORRELATION MEASURES")
print("-"*60)
pearson_r, pearson_p = stats.pearsonr(data['x'], data['Y'])
spearman_r, spearman_p = stats.spearmanr(data['x'], data['Y'])
print(f"Pearson correlation: r = {pearson_r:.4f}, p-value = {pearson_p:.6f}")
print(f"Spearman correlation: rho = {spearman_r:.4f}, p-value = {spearman_p:.6f}")
print(f"\nCoefficient of determination (R²): {pearson_r**2:.4f}")

# Basic scatter plot with different fitted curves
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
ax.scatter(data['x'], data['Y'], s=80, alpha=0.6, edgecolors='black', linewidth=1,
           c=data['Y'], cmap='viridis', label='Observations')

# Linear fit
z = np.polyfit(data['x'], data['Y'], 1)
p = np.poly1d(z)
x_line = np.linspace(data['x'].min(), data['x'].max(), 100)
ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Linear: Y = {z[0]:.4f}x + {z[1]:.4f}')

# Quadratic fit
z2 = np.polyfit(data['x'], data['Y'], 2)
p2 = np.poly1d(z2)
ax.plot(x_line, p2(x_line), "g--", linewidth=2, label=f'Quadratic')

# LOWESS smooth
from scipy.signal import savgol_filter
# Sort by x for smoothing
data_sorted = data.sort_values('x')
if len(data_sorted) > 5:
    window = min(11, len(data_sorted) if len(data_sorted) % 2 == 1 else len(data_sorted) - 1)
    y_smooth = savgol_filter(data_sorted['Y'], window_length=window, polyorder=3)
    ax.plot(data_sorted['x'], y_smooth, 'purple', linewidth=2, label='LOWESS Smooth', alpha=0.7)

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Relationship between x and Y with Fitted Curves', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.colorbar(ax.scatter(data['x'], data['Y'], s=80, c=data['Y'], cmap='viridis'),
             ax=ax, label='Y value')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/03_scatter_with_fits.png', dpi=300, bbox_inches='tight')
print("\nSaved: 03_scatter_with_fits.png")
plt.close()

# Test different functional forms
print("\n2. FUNCTIONAL FORM ANALYSIS")
print("-"*60)

# Linear model
linear_coef = np.polyfit(data['x'], data['Y'], 1)
y_pred_linear = np.polyval(linear_coef, data['x'])
residuals_linear = data['Y'] - y_pred_linear
rss_linear = np.sum(residuals_linear**2)
tss = np.sum((data['Y'] - data['Y'].mean())**2)
r2_linear = 1 - (rss_linear / tss)
rmse_linear = np.sqrt(np.mean(residuals_linear**2))

print(f"\nLinear Model:")
print(f"  Equation: Y = {linear_coef[0]:.4f}x + {linear_coef[1]:.4f}")
print(f"  R² = {r2_linear:.4f}")
print(f"  RMSE = {rmse_linear:.4f}")
print(f"  RSS = {rss_linear:.4f}")

# Quadratic model
quad_coef = np.polyfit(data['x'], data['Y'], 2)
y_pred_quad = np.polyval(quad_coef, data['x'])
residuals_quad = data['Y'] - y_pred_quad
rss_quad = np.sum(residuals_quad**2)
r2_quad = 1 - (rss_quad / tss)
rmse_quad = np.sqrt(np.mean(residuals_quad**2))

print(f"\nQuadratic Model:")
print(f"  Equation: Y = {quad_coef[0]:.6f}x² + {quad_coef[1]:.4f}x + {quad_coef[2]:.4f}")
print(f"  R² = {r2_quad:.4f}")
print(f"  RMSE = {rmse_quad:.4f}")
print(f"  RSS = {rss_quad:.4f}")

# Logarithmic model (if all x > 0)
if (data['x'] > 0).all():
    log_x = np.log(data['x'])
    log_coef = np.polyfit(log_x, data['Y'], 1)
    y_pred_log = np.polyval(log_coef, log_x)
    residuals_log = data['Y'] - y_pred_log
    rss_log = np.sum(residuals_log**2)
    r2_log = 1 - (rss_log / tss)
    rmse_log = np.sqrt(np.mean(residuals_log**2))

    print(f"\nLogarithmic Model:")
    print(f"  Equation: Y = {log_coef[0]:.4f}*log(x) + {log_coef[1]:.4f}")
    print(f"  R² = {r2_log:.4f}")
    print(f"  RMSE = {rmse_log:.4f}")
    print(f"  RSS = {rss_log:.4f}")

# Power model (if all values positive)
if (data['x'] > 0).all() and (data['Y'] > 0).all():
    log_x = np.log(data['x'])
    log_y = np.log(data['Y'])
    power_coef = np.polyfit(log_x, log_y, 1)
    y_pred_power = np.exp(power_coef[1]) * data['x']**power_coef[0]
    residuals_power = data['Y'] - y_pred_power
    rss_power = np.sum(residuals_power**2)
    r2_power = 1 - (rss_power / tss)
    rmse_power = np.sqrt(np.mean(residuals_power**2))

    print(f"\nPower Model:")
    print(f"  Equation: Y = {np.exp(power_coef[1]):.4f}*x^{power_coef[0]:.4f}")
    print(f"  R² = {r2_power:.4f}")
    print(f"  RMSE = {rmse_power:.4f}")
    print(f"  RSS = {rss_power:.4f}")

# Exponential model
try:
    exp_coef = np.polyfit(data['x'], np.log(data['Y']), 1)
    y_pred_exp = np.exp(exp_coef[1]) * np.exp(exp_coef[0] * data['x'])
    residuals_exp = data['Y'] - y_pred_exp
    rss_exp = np.sum(residuals_exp**2)
    r2_exp = 1 - (rss_exp / tss)
    rmse_exp = np.sqrt(np.mean(residuals_exp**2))

    print(f"\nExponential Model:")
    print(f"  Equation: Y = {np.exp(exp_coef[1]):.4f}*exp({exp_coef[0]:.4f}*x)")
    print(f"  R² = {r2_exp:.4f}")
    print(f"  RMSE = {rmse_exp:.4f}")
    print(f"  RSS = {rss_exp:.4f}")
except:
    print("\nExponential model: Could not fit")

print("\n" + "="*60)
