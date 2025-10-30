"""
Bivariate Relationship Analysis
================================
Author: EDA Specialist Agent
Date: 2025-10-27

This script explores the relationship between x and Y through various visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'visualizations'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("BIVARIATE ANALYSIS")
print("=" * 80)

# Figure 1: Comprehensive scatter plot analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Y vs x: Relationship Exploration', fontsize=16, fontweight='bold')

# Basic scatter plot
axes[0, 0].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('Y', fontsize=12)
axes[0, 0].set_title('Basic Scatter Plot', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Scatter with linear fit
axes[0, 1].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
# Linear regression
z = np.polyfit(df['x'], df['Y'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
axes[0, 1].plot(x_line, p(x_line), "r--", linewidth=2, label=f'Linear fit: Y={z[0]:.4f}x+{z[1]:.4f}')
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel('Y', fontsize=12)
axes[0, 1].set_title('Linear Fit', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Scatter with spline smooth
axes[1, 0].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
# Sort for spline
sorted_idx = np.argsort(df['x'])
x_sorted = df['x'].values[sorted_idx]
y_sorted = df['Y'].values[sorted_idx]
spline = UnivariateSpline(x_sorted, y_sorted, s=0.1)
x_spline = np.linspace(df['x'].min(), df['x'].max(), 100)
y_spline = spline(x_spline)
axes[1, 0].plot(x_spline, y_spline, 'r-', linewidth=2, label='Spline smooth')
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('Y', fontsize=12)
axes[1, 0].set_title('Spline Smoothing (Non-parametric)', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Scatter with logarithmic fit
axes[1, 1].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
# Log fit
x_log = np.log(df['x'] + 0.1)
z_log = np.polyfit(x_log, df['Y'], 1)
x_log_line = np.linspace(df['x'].min(), df['x'].max(), 100)
y_log_line = z_log[0] * np.log(x_log_line + 0.1) + z_log[1]
axes[1, 1].plot(x_log_line, y_log_line, 'g--', linewidth=2, label=f'Log fit: Y={z_log[0]:.4f}*ln(x)+{z_log[1]:.4f}')
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel('Y', fontsize=12)
axes[1, 1].set_title('Logarithmic Fit', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'scatter_relationship.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'scatter_relationship.png'}")
plt.close()

# Figure 2: Advanced relationship patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Advanced Relationship Patterns', fontsize=16, fontweight='bold')

# Scatter with point size by index (temporal if ordered)
axes[0, 0].scatter(df['x'], df['Y'], s=df.index * 10 + 20, alpha=0.6,
                   edgecolors='black', linewidth=1, c=df.index, cmap='viridis')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('Y', fontsize=12)
axes[0, 0].set_title('Scatter (Point size by observation order)', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
cbar.set_label('Observation Index', fontsize=10)

# Residuals from linear fit
linear_pred = p(df['x'])
residuals = df['Y'] - linear_pred
axes[0, 1].scatter(df['x'], residuals, s=80, alpha=0.7, edgecolors='black', linewidth=1, color='red')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel('Residuals', fontsize=12)
axes[0, 1].set_title('Residuals vs x (from Linear Fit)', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Segmented view (low, mid, high x)
x_tertiles = df['x'].quantile([0, 0.33, 0.67, 1.0])
df['x_segment'] = pd.cut(df['x'], bins=x_tertiles, labels=['Low', 'Mid', 'High'], include_lowest=True)
colors = {'Low': 'blue', 'Mid': 'green', 'High': 'red'}
for segment, color in colors.items():
    mask = df['x_segment'] == segment
    axes[1, 0].scatter(df.loc[mask, 'x'], df.loc[mask, 'Y'],
                       s=80, alpha=0.7, label=segment, color=color, edgecolors='black', linewidth=1)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('Y', fontsize=12)
axes[1, 0].set_title('Segmented by x Range', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Joint plot style - marginal distributions
axes[1, 1].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='purple')
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel('Y', fontsize=12)
axes[1, 1].set_title('Scatter with Range Lines', fontsize=13, fontweight='bold')
axes[1, 1].axhline(df['Y'].mean(), color='red', linestyle='--', alpha=0.5, label='Y mean')
axes[1, 1].axvline(df['x'].mean(), color='blue', linestyle='--', alpha=0.5, label='x mean')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'advanced_patterns.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'advanced_patterns.png'}")
plt.close()

# Figure 3: Model comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Model Comparison: Different Functional Forms', fontsize=16, fontweight='bold')

# Prepare data
X = df['x'].values
y = df['Y'].values
x_plot = np.linspace(df['x'].min(), df['x'].max(), 100)

# Model 1: Linear
z_lin = np.polyfit(X, y, 1)
p_lin = np.poly1d(z_lin)
y_pred_linear = p_lin(X)
ss_res_lin = np.sum((y - y_pred_linear)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2_linear = 1 - (ss_res_lin / ss_tot)

axes[0, 0].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
axes[0, 0].plot(x_plot, p_lin(x_plot), 'r-', linewidth=2, label=f'Linear (R²={r2_linear:.4f})')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('Y', fontsize=12)
axes[0, 0].set_title('Linear Model', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Model 2: Polynomial (degree 2)
z_poly2 = np.polyfit(X, y, 2)
p_poly2 = np.poly1d(z_poly2)
y_pred_poly2 = p_poly2(X)
ss_res_poly2 = np.sum((y - y_pred_poly2)**2)
r2_poly2 = 1 - (ss_res_poly2 / ss_tot)

axes[0, 1].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
axes[0, 1].plot(x_plot, p_poly2(x_plot), 'g-', linewidth=2, label=f'Polynomial deg=2 (R²={r2_poly2:.4f})')
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel('Y', fontsize=12)
axes[0, 1].set_title('Quadratic Model', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Model 3: Logarithmic
X_log = np.log(X + 0.1)
z_log_m = np.polyfit(X_log, y, 1)
p_log = np.poly1d(z_log_m)
y_pred_log = p_log(X_log)
ss_res_log = np.sum((y - y_pred_log)**2)
r2_log = 1 - (ss_res_log / ss_tot)

axes[1, 0].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
axes[1, 0].plot(x_plot, p_log(np.log(x_plot + 0.1)), 'm-', linewidth=2, label=f'Logarithmic (R²={r2_log:.4f})')
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('Y', fontsize=12)
axes[1, 0].set_title('Logarithmic Model', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Model 4: Asymptotic (Michaelis-Menten style)
def asymptotic(x, a, b):
    return a * x / (b + x)

try:
    params, _ = curve_fit(asymptotic, X, y, p0=[3, 5], maxfev=5000)
    y_pred_asym = asymptotic(X, *params)
    ss_res_asym = np.sum((y - y_pred_asym)**2)
    r2_asym = 1 - (ss_res_asym / ss_tot)

    axes[1, 1].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
    axes[1, 1].plot(x_plot, asymptotic(x_plot, *params), 'orange', linewidth=2,
                    label=f'Asymptotic (R²={r2_asym:.4f})\nY={params[0]:.2f}x/({params[1]:.2f}+x)')
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('Y', fontsize=12)
    axes[1, 1].set_title('Asymptotic Model (Michaelis-Menten)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    fit_success = True
except Exception as e:
    axes[1, 1].scatter(df['x'], df['Y'], s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('Y', fontsize=12)
    axes[1, 1].set_title('Asymptotic Model (Fit Failed)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.5, 0.5, str(e), transform=axes[1, 1].transAxes,
                    ha='center', va='center', fontsize=9)
    fit_success = False

plt.tight_layout()
plt.savefig(VIZ_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'model_comparison.png'}")
plt.close()

# Print model statistics
print("\n" + "=" * 80)
print("MODEL COMPARISON STATISTICS")
print("=" * 80)
print(f"Linear Model R²: {r2_linear:.4f}")
print(f"  Coefficients: slope={z_lin[0]:.4f}, intercept={z_lin[1]:.4f}")
print(f"\nQuadratic Model R²: {r2_poly2:.4f}")
print(f"  Coefficients: a={z_poly2[0]:.4f}, b={z_poly2[1]:.4f}, c={z_poly2[2]:.4f}")
print(f"\nLogarithmic Model R²: {r2_log:.4f}")
print(f"  Form: Y = {z_log_m[0]:.4f} * ln(x) + {z_log_m[1]:.4f}")
if fit_success:
    print(f"\nAsymptotic Model R²: {r2_asym:.4f}")
    print(f"  Parameters: Ymax={params[0]:.4f}, K={params[1]:.4f}")
else:
    print("\nAsymptotic Model: Fit failed")

print("\n" + "=" * 80)
print("Bivariate analysis complete!")
print("=" * 80)
