"""
Nonlinear Pattern Analysis
Focus: Explore curvature, saturation effects, polynomial fits, and change points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Set style
sns.set_style("whitegrid")

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("=" * 80)
print("NONLINEAR PATTERN ANALYSIS")
print("=" * 80)

X = data['x'].values
y = data['Y'].values

# Sort data by x for easier visualization
sort_idx = np.argsort(X)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

# 1. Polynomial regression of various degrees
print("\n1. POLYNOMIAL REGRESSION")
print("-" * 80)

poly_results = []
for degree in range(1, 6):
    # Fit polynomial
    coeffs = np.polyfit(X, y, degree)
    poly_func = np.poly1d(coeffs)
    y_pred = poly_func(X)
    residuals = y - y_pred

    # Calculate metrics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Adjusted R²
    n = len(y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - degree - 1)

    # AIC and BIC
    mse = ss_res / n
    aic = n * np.log(mse) + 2 * (degree + 1)
    bic = n * np.log(mse) + (degree + 1) * np.log(n)

    # RMSE
    rmse = np.sqrt(mse)

    poly_results.append({
        'degree': degree,
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'aic': aic,
        'bic': bic
    })

    print(f"Degree {degree}: R²={r2:.4f}, Adj-R²={adj_r2:.4f}, RMSE={rmse:.4f}, AIC={aic:.2f}, BIC={bic:.2f}")

poly_df = pd.DataFrame(poly_results)

# 2. Test specific nonlinear models
print("\n\n2. SPECIFIC NONLINEAR MODELS")
print("-" * 80)

# Logarithmic: Y = a + b*log(x)
log_slope, log_intercept, log_r, _, _ = stats.linregress(np.log(X), y)
log_pred = log_intercept + log_slope * np.log(X)
log_r2 = log_r**2
log_rmse = np.sqrt(np.mean((y - log_pred)**2))
print(f"Logarithmic (Y = a + b*log(x)): R²={log_r2:.4f}, RMSE={log_rmse:.4f}")
print(f"  a={log_intercept:.4f}, b={log_slope:.4f}")

# Power law: Y = a * x^b (via log-log regression)
log_log_slope, log_log_intercept, log_log_r, _, _ = stats.linregress(np.log(X), np.log(y))
power_a = np.exp(log_log_intercept)
power_b = log_log_slope
power_pred = power_a * X**power_b
power_r2 = np.corrcoef(y, power_pred)[0, 1]**2
power_rmse = np.sqrt(np.mean((y - power_pred)**2))
print(f"Power law (Y = a * x^b): R²={power_r2:.4f}, RMSE={power_rmse:.4f}")
print(f"  a={power_a:.4f}, b={power_b:.4f}")

# Exponential: Y = a * exp(b*x)
try:
    exp_slope, exp_intercept, exp_r, _, _ = stats.linregress(X, np.log(y))
    exp_a = np.exp(exp_intercept)
    exp_b = exp_slope
    exp_pred = exp_a * np.exp(exp_b * X)
    exp_r2 = np.corrcoef(y, exp_pred)[0, 1]**2
    exp_rmse = np.sqrt(np.mean((y - exp_pred)**2))
    print(f"Exponential (Y = a * exp(b*x)): R²={exp_r2:.4f}, RMSE={exp_rmse:.4f}")
    print(f"  a={exp_a:.4f}, b={exp_b:.4f}")
except:
    print("Exponential model failed to fit")
    exp_pred = None

# Michaelis-Menten (saturation): Y = (a*x) / (b + x)
def michaelis_menten(x, a, b):
    return (a * x) / (b + x)

try:
    mm_params, _ = curve_fit(michaelis_menten, X, y, p0=[3, 10], maxfev=10000)
    mm_pred = michaelis_menten(X, *mm_params)
    mm_r2 = np.corrcoef(y, mm_pred)[0, 1]**2
    mm_rmse = np.sqrt(np.mean((y - mm_pred)**2))
    print(f"Michaelis-Menten (Y = (a*x)/(b+x)): R²={mm_r2:.4f}, RMSE={mm_rmse:.4f}")
    print(f"  a={mm_params[0]:.4f}, b={mm_params[1]:.4f}")
except:
    print("Michaelis-Menten model failed to fit")
    mm_pred = None

# Asymptotic exponential: Y = a * (1 - exp(-b*x)) + c
def asymptotic_exp(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

try:
    ae_params, _ = curve_fit(asymptotic_exp, X, y, p0=[1, 0.1, 1.8], maxfev=10000)
    ae_pred = asymptotic_exp(X, *ae_params)
    ae_r2 = np.corrcoef(y, ae_pred)[0, 1]**2
    ae_rmse = np.sqrt(np.mean((y - ae_pred)**2))
    print(f"Asymptotic exp (Y = a*(1-exp(-b*x))+c): R²={ae_r2:.4f}, RMSE={ae_rmse:.4f}")
    print(f"  a={ae_params[0]:.4f}, b={ae_params[1]:.4f}, c={ae_params[2]:.4f}")
except:
    print("Asymptotic exponential model failed to fit")
    ae_pred = None

# 3. Check for change points / piecewise linearity
print("\n\n3. CHANGE POINT ANALYSIS")
print("-" * 80)

# Try different breakpoints
best_break_r2 = 0
best_breakpoint = None

for break_x in np.linspace(X.min(), X.max(), 20)[1:-1]:
    # Split data
    left_mask = X <= break_x
    right_mask = X > break_x

    if np.sum(left_mask) < 3 or np.sum(right_mask) < 3:
        continue

    # Fit two separate lines
    left_slope, left_int, _, _, _ = stats.linregress(X[left_mask], y[left_mask])
    right_slope, right_int, _, _, _ = stats.linregress(X[right_mask], y[right_mask])

    # Predict
    y_pred = np.zeros_like(y)
    y_pred[left_mask] = left_int + left_slope * X[left_mask]
    y_pred[right_mask] = right_int + right_slope * X[right_mask]

    # Calculate R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

    if r2 > best_break_r2:
        best_break_r2 = r2
        best_breakpoint = break_x

print(f"Best breakpoint: x = {best_breakpoint:.2f}")
print(f"Piecewise linear R²: {best_break_r2:.4f}")

# Fit the best piecewise model
left_mask = X <= best_breakpoint
right_mask = X > best_breakpoint
left_slope, left_int, _, _, _ = stats.linregress(X[left_mask], y[left_mask])
right_slope, right_int, _, _, _ = stats.linregress(X[right_mask], y[right_mask])

print(f"Left segment (x <= {best_breakpoint:.2f}): Y = {left_int:.4f} + {left_slope:.4f}*x")
print(f"Right segment (x > {best_breakpoint:.2f}): Y = {right_int:.4f} + {right_slope:.4f}*x")

piecewise_pred = np.zeros_like(y)
piecewise_pred[left_mask] = left_int + left_slope * X[left_mask]
piecewise_pred[right_mask] = right_int + right_slope * X[right_mask]

# 4. Curvature analysis using first differences
print("\n\n4. CURVATURE ANALYSIS (using sorted data)")
print("-" * 80)

# Calculate local slopes
local_slopes = np.diff(y_sorted) / np.diff(X_sorted)
print(f"Local slopes range: [{local_slopes.min():.4f}, {local_slopes.max():.4f}]")
print(f"Mean local slope: {local_slopes.mean():.4f}")
print(f"Std of local slopes: {local_slopes.std():.4f}")

# Calculate second differences (curvature)
second_diff = np.diff(local_slopes)
print(f"\nSecond differences (curvature indicator):")
print(f"  Mean: {second_diff.mean():.6f}")
print(f"  Std: {second_diff.std():.4f}")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Polynomial fits comparison
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(X, y, alpha=0.6, s=50, label='Data', zorder=5)

colors = ['red', 'green', 'blue', 'orange', 'purple']
for i, degree in enumerate([1, 2, 3, 4, 5]):
    coeffs = np.polyfit(X, y, degree)
    poly_func = np.poly1d(coeffs)
    x_smooth = np.linspace(X.min(), X.max(), 200)
    ax1.plot(x_smooth, poly_func(x_smooth), color=colors[i],
             label=f'Degree {degree} (R²={poly_df.iloc[i]["r2"]:.3f})', linewidth=2, alpha=0.7)

ax1.set_xlabel('x')
ax1.set_ylabel('Y')
ax1.set_title('Polynomial Fits (Degrees 1-5)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Nonlinear models comparison
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(X, y, alpha=0.6, s=50, label='Data', zorder=5)

x_smooth = np.linspace(X.min(), X.max(), 200)

# Linear
linear_pred_smooth = log_intercept + log_slope * np.log(x_smooth)
# Actually use the original linear fit
lin_slope, lin_int, _, _, _ = stats.linregress(X, y)
linear_pred_smooth = lin_int + lin_slope * x_smooth
ax2.plot(x_smooth, linear_pred_smooth, 'k--', label='Linear', linewidth=2)

# Logarithmic
log_pred_smooth = log_intercept + log_slope * np.log(x_smooth)
ax2.plot(x_smooth, log_pred_smooth, 'r-', label=f'Log (R²={log_r2:.3f})', linewidth=2)

# Power law
power_pred_smooth = power_a * x_smooth**power_b
ax2.plot(x_smooth, power_pred_smooth, 'g-', label=f'Power (R²={power_r2:.3f})', linewidth=2)

ax2.set_xlabel('x')
ax2.set_ylabel('Y')
ax2.set_title('Nonlinear Models Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Piecewise linear
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(X, y, alpha=0.6, s=50, label='Data', zorder=5)

# Plot piecewise fit
x_left = np.linspace(X.min(), best_breakpoint, 100)
x_right = np.linspace(best_breakpoint, X.max(), 100)
ax3.plot(x_left, left_int + left_slope * x_left, 'r-', linewidth=2, label='Left segment')
ax3.plot(x_right, right_int + right_slope * x_right, 'b-', linewidth=2, label='Right segment')
ax3.axvline(best_breakpoint, color='orange', linestyle='--', linewidth=2, label=f'Break at x={best_breakpoint:.1f}')

ax3.set_xlabel('x')
ax3.set_ylabel('Y')
ax3.set_title(f'Piecewise Linear (R²={best_break_r2:.3f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Saturation models (if they fit)
ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(X, y, alpha=0.6, s=50, label='Data', zorder=5)

if mm_pred is not None:
    x_smooth = np.linspace(X.min(), X.max(), 200)
    mm_smooth = michaelis_menten(x_smooth, *mm_params)
    ax4.plot(x_smooth, mm_smooth, 'r-', label=f'M-M (R²={mm_r2:.3f})', linewidth=2)

if ae_pred is not None:
    ae_smooth = asymptotic_exp(x_smooth, *ae_params)
    ax4.plot(x_smooth, ae_smooth, 'g-', label=f'Asym-Exp (R²={ae_r2:.3f})', linewidth=2)

ax4.set_xlabel('x')
ax4.set_ylabel('Y')
ax4.set_title('Saturation Models')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Local slopes
ax5 = fig.add_subplot(gs[2, 0])
mid_x = (X_sorted[:-1] + X_sorted[1:]) / 2
ax5.scatter(mid_x, local_slopes, alpha=0.7, s=40)
ax5.axhline(local_slopes.mean(), color='r', linestyle='--', linewidth=2, label='Mean slope')
ax5.set_xlabel('x')
ax5.set_ylabel('Local Slope (ΔY/Δx)')
ax5.set_title('Local Slopes vs x')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. AIC/BIC comparison
ax6 = fig.add_subplot(gs[2, 1])
degrees = poly_df['degree'].values
ax6.plot(degrees, poly_df['aic'].values, 'bo-', label='AIC', linewidth=2, markersize=8)
ax6.plot(degrees, poly_df['bic'].values, 'ro-', label='BIC', linewidth=2, markersize=8)
ax6.set_xlabel('Polynomial Degree')
ax6.set_ylabel('Information Criterion')
ax6.set_title('Model Selection (AIC/BIC)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Curvature (second differences)
ax7 = fig.add_subplot(gs[2, 2])
mid_x2 = (mid_x[:-1] + mid_x[1:]) / 2
ax7.scatter(mid_x2, second_diff, alpha=0.7, s=40)
ax7.axhline(0, color='r', linestyle='--', linewidth=2)
ax7.set_xlabel('x')
ax7.set_ylabel('Second Difference (Curvature)')
ax7.set_title('Curvature Analysis')
ax7.grid(True, alpha=0.3)

plt.savefig('/workspace/eda/analyst_2/visualizations/05_nonlinear_patterns.png', dpi=300, bbox_inches='tight')
print("\n\nSaved: /workspace/eda/analyst_2/visualizations/05_nonlinear_patterns.png")
plt.close()

print("\n" + "=" * 80)
print("NONLINEAR PATTERN ANALYSIS COMPLETE")
print("=" * 80)
