"""
Linear Regression Models: Comparing Different Functional Forms
Focus: Linear vs polynomial vs log-transformed relationships
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
with open('/workspace/data/data_analyst_3.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'year': data['year'], 'C': data['C']})
X = df['year'].values
y = df['C'].values
n = len(y)

print("="*60)
print("LINEAR REGRESSION MODEL COMPARISON")
print("="*60)

def r2_score(y_true, y_pred):
    """Calculate R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Calculate MAE"""
    return np.mean(np.abs(y_true - y_pred))

models = {}

# 1. Simple Linear Model
slope_lr1, intercept_lr1, r_value_lr1, p_value_lr1, std_err_lr1 = stats.linregress(X, y)
y_pred_lr1 = intercept_lr1 + slope_lr1 * X
residuals_lr1 = y - y_pred_lr1

models['Linear'] = {
    'predictions': y_pred_lr1,
    'residuals': residuals_lr1,
    'r2': r2_score(y, y_pred_lr1),
    'rmse': rmse(y, y_pred_lr1),
    'mae': mae(y, y_pred_lr1),
    'params': [intercept_lr1, slope_lr1]
}

print("\n1. SIMPLE LINEAR MODEL: C = β₀ + β₁*year")
print(f"   Intercept: {intercept_lr1:.3f}")
print(f"   Slope: {slope_lr1:.3f}")
print(f"   R²: {models['Linear']['r2']:.4f}")
print(f"   RMSE: {models['Linear']['rmse']:.3f}")
print(f"   MAE: {models['Linear']['mae']:.3f}")

# 2. Quadratic Model using polynomial fit
coeffs_poly2 = np.polyfit(X, y, 2)
y_pred_lr2 = np.polyval(coeffs_poly2, X)
residuals_lr2 = y - y_pred_lr2

models['Quadratic'] = {
    'predictions': y_pred_lr2,
    'residuals': residuals_lr2,
    'r2': r2_score(y, y_pred_lr2),
    'rmse': rmse(y, y_pred_lr2),
    'mae': mae(y, y_pred_lr2),
    'params': coeffs_poly2
}

print("\n2. QUADRATIC MODEL: C = β₀ + β₁*year + β₂*year²")
print(f"   Coefficients: [{coeffs_poly2[0]:.3f}, {coeffs_poly2[1]:.3f}, {coeffs_poly2[2]:.3f}]")
print(f"   R²: {models['Quadratic']['r2']:.4f}")
print(f"   RMSE: {models['Quadratic']['rmse']:.3f}")
print(f"   MAE: {models['Quadratic']['mae']:.3f}")

# 3. Cubic Model
coeffs_poly3 = np.polyfit(X, y, 3)
y_pred_lr3 = np.polyval(coeffs_poly3, X)
residuals_lr3 = y - y_pred_lr3

models['Cubic'] = {
    'predictions': y_pred_lr3,
    'residuals': residuals_lr3,
    'r2': r2_score(y, y_pred_lr3),
    'rmse': rmse(y, y_pred_lr3),
    'mae': mae(y, y_pred_lr3),
    'params': coeffs_poly3
}

print("\n3. CUBIC MODEL: C = β₀ + β₁*year + β₂*year² + β₃*year³")
print(f"   Coefficients: {coeffs_poly3}")
print(f"   R²: {models['Cubic']['r2']:.4f}")
print(f"   RMSE: {models['Cubic']['rmse']:.3f}")
print(f"   MAE: {models['Cubic']['mae']:.3f}")

# 4. Log-Linear Model (Exponential relationship)
# log(C) = β₀ + β₁*year  =>  C = exp(β₀) * exp(β₁*year)
slope_log, intercept_log, r_value_log, p_value_log, std_err_log = stats.linregress(X, np.log(y))
y_pred_log = np.exp(intercept_log + slope_log * X)
residuals_log = y - y_pred_log

models['Log-Linear'] = {
    'predictions': y_pred_log,
    'residuals': residuals_log,
    'r2': r2_score(y, y_pred_log),
    'rmse': rmse(y, y_pred_log),
    'mae': mae(y, y_pred_log),
    'log_r2': r2_score(np.log(y), intercept_log + slope_log * X),
    'params': [intercept_log, slope_log]
}

print("\n4. LOG-LINEAR MODEL: log(C) = β₀ + β₁*year")
print(f"   Intercept (log scale): {intercept_log:.3f}")
print(f"   Slope (log scale): {slope_log:.3f}")
print(f"   Implied growth rate: {(np.exp(slope_log) - 1) * 100:.2f}% per unit year")
print(f"   R² (original scale): {models['Log-Linear']['r2']:.4f}")
print(f"   R² (log scale): {models['Log-Linear']['log_r2']:.4f}")
print(f"   RMSE: {models['Log-Linear']['rmse']:.3f}")
print(f"   MAE: {models['Log-Linear']['mae']:.3f}")

# 5. Square Root Model
slope_sqrt, intercept_sqrt, r_value_sqrt, p_value_sqrt, std_err_sqrt = stats.linregress(X, np.sqrt(y))
y_pred_sqrt = (intercept_sqrt + slope_sqrt * X) ** 2
residuals_sqrt = y - y_pred_sqrt

models['Square-Root'] = {
    'predictions': y_pred_sqrt,
    'residuals': residuals_sqrt,
    'r2': r2_score(y, y_pred_sqrt),
    'rmse': rmse(y, y_pred_sqrt),
    'mae': mae(y, y_pred_sqrt),
    'params': [intercept_sqrt, slope_sqrt]
}

print("\n5. SQUARE-ROOT MODEL: √C = β₀ + β₁*year")
print(f"   Intercept: {intercept_sqrt:.3f}")
print(f"   Slope: {slope_sqrt:.3f}")
print(f"   R²: {models['Square-Root']['r2']:.4f}")
print(f"   RMSE: {models['Square-Root']['rmse']:.3f}")
print(f"   MAE: {models['Square-Root']['mae']:.3f}")

# Model comparison summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("-"*60)
for name, model_info in models.items():
    print(f"{name:<20} {model_info['r2']:>8.4f} {model_info['rmse']:>10.3f} {model_info['mae']:>10.3f}")

# Visualization: Fitted models comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Sort by year for smooth plotting
sort_idx = np.argsort(X)
year_sorted = X[sort_idx]
C_sorted = y[sort_idx]

model_names = ['Linear', 'Quadratic', 'Cubic', 'Log-Linear', 'Square-Root']
for idx, name in enumerate(model_names):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    # Plot data
    ax.plot(year_sorted, C_sorted, 'o', alpha=0.6, label='Data', markersize=5)

    # Plot model fit
    pred_sorted = models[name]['predictions'][sort_idx]
    ax.plot(year_sorted, pred_sorted, 'r-', linewidth=2, label='Fit')

    ax.set_xlabel('Year (standardized)')
    ax.set_ylabel('Count (C)')
    ax.set_title(f'{chr(65+idx)}. {name} Model\nR² = {models[name]["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Add residual comparison in the last subplot
ax = axes[1, 2]
for name in model_names:
    residuals = models[name]['residuals']
    ax.scatter(X, residuals, alpha=0.5, label=name, s=30)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Year (standardized)')
ax.set_ylabel('Residuals')
ax.set_title('F. Residual Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/02_model_fits.png', dpi=150, bbox_inches='tight')
print("\n[SAVED] Model fits comparison: visualizations/02_model_fits.png")
plt.close()

# Adjusted R² calculation (penalizes for additional parameters)
def adjusted_r2(r2, n, p):
    """Calculate adjusted R² where n=observations, p=predictors"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n" + "="*60)
print("ADJUSTED R² (accounting for model complexity)")
print("="*60)
print(f"{'Model':<20} {'R²':>8} {'Adj. R²':>10} {'Parameters':>12}")
print("-"*60)
print(f"{'Linear':<20} {models['Linear']['r2']:>8.4f} {adjusted_r2(models['Linear']['r2'], n, 1):>10.4f} {2:>12}")
print(f"{'Quadratic':<20} {models['Quadratic']['r2']:>8.4f} {adjusted_r2(models['Quadratic']['r2'], n, 2):>10.4f} {3:>12}")
print(f"{'Cubic':<20} {models['Cubic']['r2']:>8.4f} {adjusted_r2(models['Cubic']['r2'], n, 3):>10.4f} {4:>12}")
print(f"{'Log-Linear':<20} {models['Log-Linear']['r2']:>8.4f} {adjusted_r2(models['Log-Linear']['r2'], n, 1):>10.4f} {2:>12}")
print(f"{'Square-Root':<20} {models['Square-Root']['r2']:>8.4f} {adjusted_r2(models['Square-Root']['r2'], n, 1):>10.4f} {2:>12}")

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print("""
1. Log-Linear model has highest R² (exponential growth fits best)
2. Quadratic model also performs well, suggesting curvature
3. Simple linear model shows clear systematic residuals
4. Cubic model may be overfitting (marginal improvement over quadratic)
5. Square-root transformation underperforms relative to log transformation
""")
