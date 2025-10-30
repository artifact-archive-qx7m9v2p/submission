"""
Functional Form Exploration - Testing multiple hypotheses
Comparing: Linear, Polynomial (2-4 degree), Logarithmic, Square root, Asymptotic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
x = data['x'].values
y = data['Y'].values

# Define candidate functional forms
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def logarithmic(x, a, b):
    return a * np.log(x) + b

def sqrt_form(x, a, b):
    return a * np.sqrt(x) + b

def inverse(x, a, b):
    return a / x + b

def asymptotic(x, a, b, c):
    """Asymptotic form: a - b * exp(-c * x)"""
    return a - b * np.exp(-c * x)

def power_law(x, a, b):
    return a * x**b

# Fit all models and calculate metrics
models = {}
x_smooth = np.linspace(x.min(), x.max(), 200)

print("="*60)
print("FUNCTIONAL FORM COMPARISON")
print("="*60)

# Linear
try:
    popt, _ = curve_fit(linear, x, y)
    y_pred = linear(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = len(x) * np.log(ss_res / len(x)) + 2 * len(popt)
    models['Linear'] = {
        'params': popt,
        'pred': y_pred,
        'smooth': linear(x_smooth, *popt),
        'r2': r2,
        'rmse': rmse,
        'aic': aic,
        'n_params': len(popt)
    }
    print(f"\nLinear: Y = {popt[0]:.4f}*x + {popt[1]:.4f}")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
except Exception as e:
    print(f"Linear fit failed: {e}")

# Quadratic
try:
    popt, _ = curve_fit(quadratic, x, y)
    y_pred = quadratic(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = len(x) * np.log(ss_res / len(x)) + 2 * len(popt)
    models['Quadratic'] = {
        'params': popt,
        'pred': y_pred,
        'smooth': quadratic(x_smooth, *popt),
        'r2': r2,
        'rmse': rmse,
        'aic': aic,
        'n_params': len(popt)
    }
    print(f"\nQuadratic: Y = {popt[0]:.6f}*x² + {popt[1]:.4f}*x + {popt[2]:.4f}")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
except Exception as e:
    print(f"Quadratic fit failed: {e}")

# Cubic
try:
    popt, _ = curve_fit(cubic, x, y)
    y_pred = cubic(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = len(x) * np.log(ss_res / len(x)) + 2 * len(popt)
    models['Cubic'] = {
        'params': popt,
        'pred': y_pred,
        'smooth': cubic(x_smooth, *popt),
        'r2': r2,
        'rmse': rmse,
        'aic': aic,
        'n_params': len(popt)
    }
    print(f"\nCubic: Y = {popt[0]:.8f}*x³ + {popt[1]:.6f}*x² + {popt[2]:.4f}*x + {popt[3]:.4f}")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
except Exception as e:
    print(f"Cubic fit failed: {e}")

# Logarithmic
try:
    popt, _ = curve_fit(logarithmic, x, y)
    y_pred = logarithmic(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = len(x) * np.log(ss_res / len(x)) + 2 * len(popt)
    models['Logarithmic'] = {
        'params': popt,
        'pred': y_pred,
        'smooth': logarithmic(x_smooth, *popt),
        'r2': r2,
        'rmse': rmse,
        'aic': aic,
        'n_params': len(popt)
    }
    print(f"\nLogarithmic: Y = {popt[0]:.4f}*ln(x) + {popt[1]:.4f}")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
except Exception as e:
    print(f"Logarithmic fit failed: {e}")

# Square root
try:
    popt, _ = curve_fit(sqrt_form, x, y)
    y_pred = sqrt_form(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = len(x) * np.log(ss_res / len(x)) + 2 * len(popt)
    models['Square Root'] = {
        'params': popt,
        'pred': y_pred,
        'smooth': sqrt_form(x_smooth, *popt),
        'r2': r2,
        'rmse': rmse,
        'aic': aic,
        'n_params': len(popt)
    }
    print(f"\nSquare Root: Y = {popt[0]:.4f}*sqrt(x) + {popt[1]:.4f}")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
except Exception as e:
    print(f"Square root fit failed: {e}")

# Power law
try:
    popt, _ = curve_fit(power_law, x, y, p0=[1, 0.5])
    y_pred = power_law(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = len(x) * np.log(ss_res / len(x)) + 2 * len(popt)
    models['Power Law'] = {
        'params': popt,
        'pred': y_pred,
        'smooth': power_law(x_smooth, *popt),
        'r2': r2,
        'rmse': rmse,
        'aic': aic,
        'n_params': len(popt)
    }
    print(f"\nPower Law: Y = {popt[0]:.4f}*x^{popt[1]:.4f}")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
except Exception as e:
    print(f"Power law fit failed: {e}")

# Asymptotic
try:
    popt, _ = curve_fit(asymptotic, x, y, p0=[2.5, 1, 0.1], maxfev=5000)
    y_pred = asymptotic(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = len(x) * np.log(ss_res / len(x)) + 2 * len(popt)
    models['Asymptotic'] = {
        'params': popt,
        'pred': y_pred,
        'smooth': asymptotic(x_smooth, *popt),
        'r2': r2,
        'rmse': rmse,
        'aic': aic,
        'n_params': len(popt)
    }
    print(f"\nAsymptotic: Y = {popt[0]:.4f} - {popt[1]:.4f}*exp(-{popt[2]:.4f}*x)")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
except Exception as e:
    print(f"Asymptotic fit failed: {e}")

# Summary comparison
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<15} {'R²':>8} {'RMSE':>8} {'AIC':>8} {'Params':>8}")
print("-"*60)
for name, metrics in sorted(models.items(), key=lambda x: x[1]['aic']):
    print(f"{name:<15} {metrics['r2']:>8.4f} {metrics['rmse']:>8.4f} "
          f"{metrics['aic']:>8.2f} {metrics['n_params']:>8}")

print("\n" + "="*60)

# Save results for visualization
np.savez('/workspace/eda/analyst_2/code/model_fits.npz',
         x=x, y=y, x_smooth=x_smooth, **{k: v for k, v in models.items()})

print("\nSaved model fits to: model_fits.npz")
