"""
Trend Analysis: Testing Multiple Functional Forms
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_1.csv')
df = df.sort_values('year').reset_index(drop=True)

# Prepare data
X = df['year'].values
y = df['C'].values

# Helper functions
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

print("=" * 80)
print("TREND ANALYSIS: FUNCTIONAL FORM TESTING")
print("=" * 80)

# Define candidate models
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def exponential(x, a, b):
    return a * np.exp(b * x)

# Fit models and collect results
models = {}

# Linear
params_linear = np.polyfit(X, y, 1)
y_pred_linear = np.polyval(params_linear, X)
models['Linear'] = {
    'params': params_linear,
    'predictions': y_pred_linear,
    'r2': r2_score(y, y_pred_linear),
    'rmse': rmse(y, y_pred_linear),
    'aic': len(y) * np.log(rmse(y, y_pred_linear)**2) + 2 * 2
}

# Quadratic
params_quad = np.polyfit(X, y, 2)
y_pred_quad = np.polyval(params_quad, X)
models['Quadratic'] = {
    'params': params_quad,
    'predictions': y_pred_quad,
    'r2': r2_score(y, y_pred_quad),
    'rmse': rmse(y, y_pred_quad),
    'aic': len(y) * np.log(rmse(y, y_pred_quad)**2) + 2 * 3
}

# Cubic
params_cubic = np.polyfit(X, y, 3)
y_pred_cubic = np.polyval(params_cubic, X)
models['Cubic'] = {
    'params': params_cubic,
    'predictions': y_pred_cubic,
    'r2': r2_score(y, y_pred_cubic),
    'rmse': rmse(y, y_pred_cubic),
    'aic': len(y) * np.log(rmse(y, y_pred_cubic)**2) + 2 * 4
}

# Exponential (fit on log scale)
try:
    params_exp, _ = curve_fit(exponential, X, y, p0=[y[0], 0.1], maxfev=5000)
    y_pred_exp = exponential(X, *params_exp)
    models['Exponential'] = {
        'params': params_exp,
        'predictions': y_pred_exp,
        'r2': r2_score(y, y_pred_exp),
        'rmse': rmse(y, y_pred_exp),
        'aic': len(y) * np.log(rmse(y, y_pred_exp)**2) + 2 * 2
    }
except:
    print("Warning: Exponential fit failed")

# Log-linear (linear on log scale)
log_y = np.log(y)
params_log_linear = np.polyfit(X, log_y, 1)
y_pred_log_linear = np.exp(np.polyval(params_log_linear, X))
models['Log-Linear'] = {
    'params': params_log_linear,
    'predictions': y_pred_log_linear,
    'r2': r2_score(y, y_pred_log_linear),
    'rmse': rmse(y, y_pred_log_linear),
    'aic': len(y) * np.log(rmse(y, y_pred_log_linear)**2) + 2 * 2,
    'log_r2': r2_score(log_y, np.polyval(params_log_linear, X))
}

# Print model comparison
print("\nMODEL COMPARISON:")
print("-" * 80)
print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'AIC':<10}")
print("-" * 80)

for name, results in sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"{name:<15} {results['r2']:<10.4f} {results['rmse']:<10.2f} {results['aic']:<10.2f}")

# Find best model by R²
best_model = max(models.items(), key=lambda x: x[1]['r2'])
print(f"\nBest model by R²: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

# Find best model by AIC (lower is better)
best_aic = min(models.items(), key=lambda x: x[1]['aic'])
print(f"Best model by AIC: {best_aic[0]} (AIC = {best_aic[1]['aic']:.2f})")

# Additional diagnostics
print("\n" + "=" * 80)
print("DETAILED MODEL DIAGNOSTICS")
print("=" * 80)

for name, results in models.items():
    residuals = y - results['predictions']
    print(f"\n{name}:")
    print(f"  Parameters: {results['params']}")
    print(f"  Mean residual: {np.mean(residuals):.4f}")
    print(f"  Residual std: {np.std(residuals):.4f}")
    print(f"  Max positive residual: {np.max(residuals):.2f}")
    print(f"  Max negative residual: {np.min(residuals):.2f}")

    # Durbin-Watson statistic for autocorrelation
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"  Durbin-Watson: {dw:.4f} (2=no autocorr, <2=positive, >2=negative)")

# Save results for plotting
results_dict = {
    'models': models,
    'X': X,
    'y': y
}

import pickle
with open('/workspace/eda/analyst_1/code/trend_models.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

print("\n" + "=" * 80)
print("GROWTH RATE ANALYSIS")
print("=" * 80)

# Compare growth implications
print("\nImplied growth rate (for exponential-type models):")
if 'Exponential' in models:
    exp_rate = models['Exponential']['params'][1]
    print(f"  Exponential model: {exp_rate:.4f} (per standardized year unit)")
    print(f"    Implies ~{(np.exp(exp_rate) - 1) * 100:.1f}% growth per unit")

if 'Log-Linear' in models:
    log_rate = models['Log-Linear']['params'][0]
    print(f"  Log-Linear model: {log_rate:.4f} (per standardized year unit)")
    print(f"    Implies ~{(np.exp(log_rate) - 1) * 100:.1f}% growth per unit")

# Calculate empirical growth rates
abs_diff = np.diff(y)
pct_change = np.diff(y) / y[:-1] * 100
print(f"\nEmpirical changes:")
print(f"  Mean absolute change: {np.mean(abs_diff):.2f} (std: {np.std(abs_diff):.2f})")
print(f"  Mean percent change: {np.mean(pct_change):.2f}% (std: {np.std(pct_change):.2f}%)")
print(f"  Median percent change: {np.median(pct_change):.2f}%")

# Check if growth rate is changing over time
first_half_pct = pct_change[:19]
second_half_pct = pct_change[19:]
print(f"\nGrowth rate evolution:")
print(f"  First half mean % change: {np.mean(first_half_pct):.2f}%")
print(f"  Second half mean % change: {np.mean(second_half_pct):.2f}%")

print("\nAnalysis complete. Results saved.")
