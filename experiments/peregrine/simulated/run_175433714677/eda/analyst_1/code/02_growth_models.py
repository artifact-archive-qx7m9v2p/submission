"""
Growth Model Comparison: Testing Different Functional Forms
Focus: Linear, Quadratic, Exponential, and Change Point models
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('/workspace/data/data_analyst_1.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

X = df['year'].values
y = df['C'].values

print("="*60)
print("GROWTH MODEL COMPARISON")
print("="*60)

# Helper functions for R² and RMSE
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# 1. Linear model: C = a + b*year
slope_linear, intercept_linear, r_value, p_value, std_err = stats.linregress(X, y)
y_pred_linear = intercept_linear + slope_linear * X
r2_linear = r_value**2
rmse_linear = rmse(y, y_pred_linear)

print(f"\n1. LINEAR MODEL: C = a + b*year")
print(f"   Coefficients: intercept={intercept_linear:.2f}, slope={slope_linear:.2f}")
print(f"   R²={r2_linear:.4f}, RMSE={rmse_linear:.2f}")

# 2. Quadratic model: C = a + b*year + c*year²
coeffs_quad = np.polyfit(X, y, 2)
y_pred_quad = np.polyval(coeffs_quad, X)
r2_quad = r_squared(y, y_pred_quad)
rmse_quad = rmse(y, y_pred_quad)

print(f"\n2. QUADRATIC MODEL: C = a + b*year + c*year²")
print(f"   Coefficients: a={coeffs_quad[2]:.2f}, b={coeffs_quad[1]:.2f}, c={coeffs_quad[0]:.2f}")
print(f"   R²={r2_quad:.4f}, RMSE={rmse_quad:.2f}")

# 3. Cubic model: C = a + b*year + c*year² + d*year³
coeffs_cubic = np.polyfit(X, y, 3)
y_pred_cubic = np.polyval(coeffs_cubic, X)
r2_cubic = r_squared(y, y_pred_cubic)
rmse_cubic = rmse(y, y_pred_cubic)

print(f"\n3. CUBIC MODEL: C = a + b*year + c*year² + d*year³")
print(f"   R²={r2_cubic:.4f}, RMSE={rmse_cubic:.2f}")

# 4. Exponential model: C = a * exp(b*year)
y_log = np.log(y)
slope_exp, intercept_exp, _, _, _ = stats.linregress(X, y_log)
y_pred_exp = np.exp(intercept_exp + slope_exp * X)
r2_exp = r_squared(y, y_pred_exp)
rmse_exp = rmse(y, y_pred_exp)

print(f"\n4. EXPONENTIAL MODEL: log(C) = a + b*year")
print(f"   Log-space: intercept={intercept_exp:.4f}, slope={slope_exp:.4f}")
print(f"   Implies growth rate: {np.exp(slope_exp)-1:.2%} per unit year")
print(f"   R² (on original scale)={r2_exp:.4f}, RMSE={rmse_exp:.2f}")

# 5. Piecewise linear model (test for change point around year=0)
mask_early = X < 0
mask_late = X >= 0

X_early = X[mask_early]
y_early = y[mask_early]
X_late = X[mask_late]
y_late = y[mask_late]

slope_early, intercept_early, _, _, _ = stats.linregress(X_early, y_early)
slope_late, intercept_late, _, _, _ = stats.linregress(X_late, y_late)

y_pred_piece = np.zeros_like(y)
y_pred_piece[mask_early] = intercept_early + slope_early * X_early
y_pred_piece[mask_late] = intercept_late + slope_late * X_late

r2_piece = r_squared(y, y_pred_piece)
rmse_piece = rmse(y, y_pred_piece)

print(f"\n5. PIECEWISE LINEAR MODEL (change point at year=0)")
print(f"   Early period (year<0): slope={slope_early:.2f}")
print(f"   Late period (year≥0): slope={slope_late:.2f}")
print(f"   Slope ratio (late/early): {slope_late/slope_early:.2f}x")
print(f"   R²={r2_piece:.4f}, RMSE={rmse_piece:.2f}")

# Summary comparison
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
models_summary = pd.DataFrame({
    'Model': ['Linear', 'Quadratic', 'Cubic', 'Exponential', 'Piecewise Linear'],
    'R²': [r2_linear, r2_quad, r2_cubic, r2_exp, r2_piece],
    'RMSE': [rmse_linear, rmse_quad, rmse_cubic, rmse_exp, rmse_piece]
})
models_summary = models_summary.sort_values('R²', ascending=False)
print(models_summary.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Linear
ax1 = axes[0, 0]
ax1.scatter(X, y, alpha=0.6, s=50, label='Observed', color='steelblue')
ax1.plot(X, y_pred_linear, 'r-', linewidth=2, label='Linear fit')
ax1.set_xlabel('Year (standardized)', fontsize=10)
ax1.set_ylabel('Count (C)', fontsize=10)
ax1.set_title(f'A. Linear Model\nR²={r2_linear:.4f}', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Quadratic
ax2 = axes[0, 1]
ax2.scatter(X, y, alpha=0.6, s=50, label='Observed', color='steelblue')
ax2.plot(X, y_pred_quad, 'g-', linewidth=2, label='Quadratic fit')
ax2.set_xlabel('Year (standardized)', fontsize=10)
ax2.set_ylabel('Count (C)', fontsize=10)
ax2.set_title(f'B. Quadratic Model\nR²={r2_quad:.4f}', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cubic
ax3 = axes[0, 2]
ax3.scatter(X, y, alpha=0.6, s=50, label='Observed', color='steelblue')
ax3.plot(X, y_pred_cubic, color='purple', linewidth=2, label='Cubic fit')
ax3.set_xlabel('Year (standardized)', fontsize=10)
ax3.set_ylabel('Count (C)', fontsize=10)
ax3.set_title(f'C. Cubic Model\nR²={r2_cubic:.4f}', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Exponential
ax4 = axes[1, 0]
ax4.scatter(X, y, alpha=0.6, s=50, label='Observed', color='steelblue')
ax4.plot(X, y_pred_exp, 'orange', linewidth=2, label='Exponential fit')
ax4.set_xlabel('Year (standardized)', fontsize=10)
ax4.set_ylabel('Count (C)', fontsize=10)
ax4.set_title(f'D. Exponential Model\nR²={r2_exp:.4f}', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Piecewise
ax5 = axes[1, 1]
ax5.scatter(X, y, alpha=0.6, s=50, label='Observed', color='steelblue')
ax5.plot(X, y_pred_piece, 'm-', linewidth=2, label='Piecewise linear')
ax5.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Change point')
ax5.set_xlabel('Year (standardized)', fontsize=10)
ax5.set_ylabel('Count (C)', fontsize=10)
ax5.set_title(f'E. Piecewise Linear Model\nR²={r2_piece:.4f}', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Residual comparison for best models
ax6 = axes[1, 2]
residuals_quad = y - y_pred_quad
residuals_cubic = y - y_pred_cubic
ax6.scatter(X, residuals_quad, alpha=0.6, s=50, label='Quadratic', color='green')
ax6.scatter(X, residuals_cubic, alpha=0.6, s=50, label='Cubic', color='purple', marker='s')
ax6.axhline(0, color='black', linestyle='-', linewidth=1)
ax6.set_xlabel('Year (standardized)', fontsize=10)
ax6.set_ylabel('Residuals', fontsize=10)
ax6.set_title('F. Residuals: Quadratic vs Cubic', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/02_growth_models.png', dpi=300, bbox_inches='tight')
print("\nSaved: 02_growth_models.png")

plt.close()
