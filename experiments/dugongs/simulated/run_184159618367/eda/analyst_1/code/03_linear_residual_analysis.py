"""
Linear Model Residual Analysis
Analyst 1 - Round 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Fit simple linear regression
coeffs = np.polyfit(data['x'], data['Y'], 1)
slope, intercept = coeffs[0], coeffs[1]
y_pred = slope * data['x'] + intercept
residuals = data['Y'] - y_pred

# Add to dataframe
data['y_pred'] = y_pred
data['residuals'] = residuals
data['abs_residuals'] = np.abs(residuals)
data['std_residuals'] = residuals / residuals.std()

print("=" * 60)
print("LINEAR REGRESSION DIAGNOSTICS")
print("=" * 60)

print(f"\nModel: Y = {intercept:.4f} + {slope:.4f} * x")
print(f"R-squared: {np.corrcoef(data['x'], data['Y'])[0,1]**2:.4f}")

# Residual statistics
print(f"\nResidual Statistics:")
print(f"Mean: {residuals.mean():.6f} (should be ~0)")
print(f"Std Dev: {residuals.std():.4f}")
print(f"Min: {residuals.min():.4f}")
print(f"Max: {residuals.max():.4f}")
print(f"Range: {residuals.max() - residuals.min():.4f}")

# Test for normality of residuals
_, p_resid = stats.shapiro(residuals)
print(f"\nShapiro-Wilk test on residuals: p = {p_resid:.4f}")
print(f"Residuals {'appear normal' if p_resid > 0.05 else 'deviate from normality'}")

# Durbin-Watson test for autocorrelation
dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"\nDurbin-Watson statistic: {dw:.4f} (2=no autocorr, <2=pos autocorr, >2=neg autocorr)")

# Breusch-Pagan test for heteroscedasticity (manual implementation)
# Regress squared residuals on x
resid_sq = residuals**2
coeffs_bp = np.polyfit(data['x'], resid_sq, 1)
resid_sq_pred = coeffs_bp[0] * data['x'] + coeffs_bp[1]
ss_total_bp = np.sum((resid_sq - resid_sq.mean())**2)
ss_resid_bp = np.sum((resid_sq - resid_sq_pred)**2)
r2_bp = 1 - ss_resid_bp / ss_total_bp
bp_stat = len(data) * r2_bp
print(f"\nBreusch-Pagan test statistic: {bp_stat:.4f}")
print(f"(Higher values suggest heteroscedasticity)")

# Variance by x range
data_sorted = data.sort_values('x')
n_third = len(data) // 3
var_low = data_sorted['residuals'].iloc[:n_third].var()
var_mid = data_sorted['residuals'].iloc[n_third:2*n_third].var()
var_high = data_sorted['residuals'].iloc[2*n_third:].var()

print(f"\nResidual variance by x range:")
print(f"  Low x third: {var_low:.6f}")
print(f"  Mid x third: {var_mid:.6f}")
print(f"  High x third: {var_high:.6f}")
print(f"  Variance ratio (high/low): {var_high/var_low:.2f}")

# Identify potential outliers (|std residual| > 2)
outliers = data[np.abs(data['std_residuals']) > 2]
print(f"\nPotential outliers (|std residual| > 2): {len(outliers)}")
if len(outliers) > 0:
    print(outliers[['x', 'Y', 'y_pred', 'residuals', 'std_residuals']])

print("\n" + "=" * 60)

# ============================================================
# FIGURE 4: Comprehensive residual diagnostics (4-panel)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Residuals vs Fitted
axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=80, color='steelblue',
                   edgecolors='black', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].axhline(y=2*residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0, 0].axhline(y=-2*residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
# Add LOWESS smoother to residuals
from scipy.signal import savgol_filter
sorted_indices = np.argsort(y_pred)
if len(data) >= 7:
    window = 7 if len(data) >= 7 else len(data) if len(data) % 2 == 1 else len(data) - 1
    smooth_resid = savgol_filter(residuals.values[sorted_indices], window, 2)
    axes[0, 0].plot(y_pred.values[sorted_indices], smooth_resid,
                    color='green', linewidth=2.5, alpha=0.8, label='Smoother')
axes[0, 0].set_xlabel('Fitted Values', fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontweight='bold')
axes[0, 0].set_title('Residuals vs Fitted Values', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Q-Q plot for residuals
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot of Residuals', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Scale-Location (sqrt of standardized residuals vs fitted)
sqrt_abs_std_resid = np.sqrt(np.abs(data['std_residuals']))
axes[1, 0].scatter(y_pred, sqrt_abs_std_resid, alpha=0.6, s=80,
                   color='steelblue', edgecolors='black', linewidth=0.5)
# Add smoother
sorted_indices = np.argsort(y_pred)
if len(data) >= 7:
    smooth_scale = savgol_filter(sqrt_abs_std_resid.values[sorted_indices], window, 2)
    axes[1, 0].plot(y_pred.values[sorted_indices], smooth_scale,
                    color='red', linewidth=2.5, alpha=0.8)
axes[1, 0].set_xlabel('Fitted Values', fontweight='bold')
axes[1, 0].set_ylabel('√|Standardized Residuals|', fontweight='bold')
axes[1, 0].set_title('Scale-Location Plot', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Residuals vs x (predictor)
axes[1, 1].scatter(data['x'], residuals, alpha=0.6, s=80, color='steelblue',
                   edgecolors='black', linewidth=0.5)
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axhline(y=2*residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
axes[1, 1].axhline(y=-2*residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
# Add smoother
x_sorted_indices = np.argsort(data['x'])
if len(data) >= 7:
    smooth_resid_x = savgol_filter(residuals.values[x_sorted_indices], window, 2)
    axes[1, 1].plot(data['x'].values[x_sorted_indices], smooth_resid_x,
                    color='green', linewidth=2.5, alpha=0.8, label='Smoother')
axes[1, 1].set_xlabel('x (predictor)', fontweight='bold')
axes[1, 1].set_ylabel('Residuals', fontweight='bold')
axes[1, 1].set_title('Residuals vs Predictor x', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/04_residual_diagnostics.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Created: 04_residual_diagnostics.png")
