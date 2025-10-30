"""
Nonlinearity Investigation
===========================
Test various functional forms and transformations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def r2_score(y_true, y_pred):
    """Calculate R-squared"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Load data
df = pd.read_csv('/workspace/data/data.csv')

# Test multiple functional forms
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Testing Different Functional Forms', fontsize=16, fontweight='bold')

x = df['x'].values
y = df['Y'].values

# 1. Linear
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z_linear = np.polyfit(x, y, 1)
p_linear = np.poly1d(z_linear)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, p_linear(x_line), 'r-', linewidth=2)
r2_linear = r2_score(y, p_linear(x))
rmse_linear = rmse(y, p_linear(x))
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title(f'Linear: R²={r2_linear:.4f}, RMSE={rmse_linear:.4f}', fontsize=11)
ax.grid(True, alpha=0.3)

# 2. Quadratic
ax = axes[0, 1]
ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z_quad = np.polyfit(x, y, 2)
p_quad = np.poly1d(z_quad)
ax.plot(x_line, p_quad(x_line), 'g-', linewidth=2)
r2_quad = r2_score(y, p_quad(x))
rmse_quad = rmse(y, p_quad(x))
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title(f'Quadratic: R²={r2_quad:.4f}, RMSE={rmse_quad:.4f}', fontsize=11)
ax.grid(True, alpha=0.3)

# 3. Cubic
ax = axes[0, 2]
ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z_cubic = np.polyfit(x, y, 3)
p_cubic = np.poly1d(z_cubic)
ax.plot(x_line, p_cubic(x_line), 'purple', linewidth=2)
r2_cubic = r2_score(y, p_cubic(x))
rmse_cubic = rmse(y, p_cubic(x))
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title(f'Cubic: R²={r2_cubic:.4f}, RMSE={rmse_cubic:.4f}', fontsize=11)
ax.grid(True, alpha=0.3)

# 4. Logarithmic (log(x))
ax = axes[1, 0]
x_log = np.log(x)
z_log = np.polyfit(x_log, y, 1)
p_log = np.poly1d(z_log)
x_log_line = np.log(x_line)
y_log_pred = p_log(x_log)
r2_log = r2_score(y, y_log_pred)
rmse_log = rmse(y, y_log_pred)
ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.plot(x_line, p_log(x_log_line), 'orange', linewidth=2)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title(f'Logarithmic Y ~ log(x): R²={r2_log:.4f}, RMSE={rmse_log:.4f}', fontsize=11)
ax.grid(True, alpha=0.3)

# 5. Square root
ax = axes[1, 1]
x_sqrt = np.sqrt(x)
z_sqrt = np.polyfit(x_sqrt, y, 1)
p_sqrt = np.poly1d(z_sqrt)
x_sqrt_line = np.sqrt(x_line)
y_sqrt_pred = p_sqrt(x_sqrt)
r2_sqrt = r2_score(y, y_sqrt_pred)
rmse_sqrt = rmse(y, y_sqrt_pred)
ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.plot(x_line, p_sqrt(x_sqrt_line), 'brown', linewidth=2)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title(f'Square root Y ~ sqrt(x): R²={r2_sqrt:.4f}, RMSE={rmse_sqrt:.4f}', fontsize=11)
ax.grid(True, alpha=0.3)

# 6. Asymptotic/Saturation model (y = a - b*exp(-c*x))
ax = axes[1, 2]
from scipy.optimize import curve_fit
def asymptotic(x, a, b, c):
    return a - b * np.exp(-c * x)
try:
    popt, _ = curve_fit(asymptotic, x, y, p0=[2.7, 1.0, 0.1], maxfev=10000)
    y_asym_pred = asymptotic(x, *popt)
    r2_asym = r2_score(y, y_asym_pred)
    rmse_asym = rmse(y, y_asym_pred)
    ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax.plot(x_line, asymptotic(x_line, *popt), 'navy', linewidth=2)
    ax.set_title(f'Asymptotic: R²={r2_asym:.4f}, RMSE={rmse_asym:.4f}', fontsize=11)
except:
    r2_asym = np.nan
    rmse_asym = np.nan
    ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax.set_title('Asymptotic: Fit failed', fontsize=11)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/05_functional_forms.png', dpi=300, bbox_inches='tight')
plt.close()

# Print comparison
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print(f"\n{'Model':<20} {'R²':<12} {'RMSE':<12} {'Parameters'}")
print("-" * 80)
print(f"{'Linear':<20} {r2_linear:<12.6f} {rmse_linear:<12.6f} 2")
print(f"{'Quadratic':<20} {r2_quad:<12.6f} {rmse_quad:<12.6f} 3")
print(f"{'Cubic':<20} {r2_cubic:<12.6f} {rmse_cubic:<12.6f} 4")
print(f"{'Logarithmic':<20} {r2_log:<12.6f} {rmse_log:<12.6f} 2")
print(f"{'Square root':<20} {r2_sqrt:<12.6f} {rmse_sqrt:<12.6f} 2")
if not np.isnan(r2_asym):
    print(f"{'Asymptotic':<20} {r2_asym:<12.6f} {rmse_asym:<12.6f} 3")

# Transformation analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Transformation Analysis', fontsize=16, fontweight='bold')

# Log-log plot
ax = axes[0, 0]
ax.scatter(np.log(x), np.log(y), alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z_loglog = np.polyfit(np.log(x), np.log(y), 1)
p_loglog = np.poly1d(z_loglog)
ax.plot(np.log(x_line), p_loglog(np.log(x_line)), 'r-', linewidth=2)
r2_loglog = r2_score(np.log(y), p_loglog(np.log(x)))
ax.set_xlabel('log(x)', fontsize=12)
ax.set_ylabel('log(Y)', fontsize=12)
ax.set_title(f'Log-Log: R²={r2_loglog:.4f} (Power law test)', fontsize=12)
ax.grid(True, alpha=0.3)

# Semi-log (x) plot
ax = axes[0, 1]
ax.scatter(x, np.log(y), alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z_semilogx = np.polyfit(x, np.log(y), 1)
p_semilogx = np.poly1d(z_semilogx)
ax.plot(x_line, p_semilogx(x_line), 'g-', linewidth=2)
r2_semilogx = r2_score(np.log(y), p_semilogx(x))
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('log(Y)', fontsize=12)
ax.set_title(f'Semi-log (Y): R²={r2_semilogx:.4f} (Exponential test)', fontsize=12)
ax.grid(True, alpha=0.3)

# Semi-log (y) plot - already did this as logarithmic
ax = axes[1, 0]
ax.scatter(np.log(x), y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.plot(x_log_line, p_log(x_log_line), 'orange', linewidth=2)
ax.set_xlabel('log(x)', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title(f'Semi-log (x): R²={r2_log:.4f} (Logarithmic test)', fontsize=12)
ax.grid(True, alpha=0.3)

# Reciprocal transformation
ax = axes[1, 1]
x_recip = 1 / x
z_recip = np.polyfit(x_recip, y, 1)
p_recip = np.poly1d(z_recip)
x_recip_line = 1 / x_line
y_recip_pred = p_recip(x_recip)
r2_recip = r2_score(y, y_recip_pred)
ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.plot(x_line, p_recip(x_recip_line), 'purple', linewidth=2)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title(f'Reciprocal Y ~ 1/x: R²={r2_recip:.4f}', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/06_transformations.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("TRANSFORMATION ANALYSIS")
print("=" * 80)
print(f"Log-log (power law):     R² = {r2_loglog:.6f}")
print(f"Semi-log Y (exponential): R² = {r2_semilogx:.6f}")
print(f"Semi-log x (logarithmic): R² = {r2_log:.6f}")
print(f"Reciprocal (1/x):        R² = {r2_recip:.6f}")

# Segmentation analysis - look for changepoints
print("\n" + "=" * 80)
print("SEGMENTATION ANALYSIS")
print("=" * 80)

# Sort data by x
df_sorted = df.sort_values('x').reset_index(drop=True)
x_sorted = df_sorted['x'].values
y_sorted = df_sorted['Y'].values

# Try different breakpoints
best_breakpoint = None
best_improvement = -np.inf
linear_sse = np.sum((y - p_linear(x))**2)

print("\nTesting potential changepoints...")
print(f"Baseline linear model SSE: {linear_sse:.6f}")

for i in range(5, len(x_sorted) - 5):  # Ensure enough points in each segment
    breakpoint = x_sorted[i]

    # Fit two separate lines
    mask1 = x <= breakpoint
    mask2 = x > breakpoint

    if np.sum(mask1) >= 3 and np.sum(mask2) >= 3:  # Need at least 3 points per segment
        # First segment
        z1 = np.polyfit(x[mask1], y[mask1], 1)
        p1 = np.poly1d(z1)
        sse1 = np.sum((y[mask1] - p1(x[mask1]))**2)

        # Second segment
        z2 = np.polyfit(x[mask2], y[mask2], 1)
        p2 = np.poly1d(z2)
        sse2 = np.sum((y[mask2] - p2(x[mask2]))**2)

        # Total SSE
        total_sse = sse1 + sse2
        improvement = linear_sse - total_sse

        if improvement > best_improvement:
            best_improvement = improvement
            best_breakpoint = breakpoint
            best_info = {
                'breakpoint': breakpoint,
                'n1': np.sum(mask1),
                'n2': np.sum(mask2),
                'slope1': z1[0],
                'intercept1': z1[1],
                'slope2': z2[0],
                'intercept2': z2[1],
                'sse': total_sse,
                'improvement': improvement
            }

if best_breakpoint is not None:
    print(f"\nBest breakpoint: x = {best_info['breakpoint']:.2f}")
    print(f"  Segment 1 (x <= {best_info['breakpoint']:.2f}): n={best_info['n1']}, slope={best_info['slope1']:.6f}")
    print(f"  Segment 2 (x > {best_info['breakpoint']:.2f}): n={best_info['n2']}, slope={best_info['slope2']:.6f}")
    print(f"  SSE reduction: {best_info['improvement']:.6f} ({100*best_info['improvement']/linear_sse:.2f}%)")

    # F-test for model comparison
    # Linear model: p=2, Piecewise: p=4
    n = len(x)
    f_stat = ((linear_sse - best_info['sse']) / 2) / (best_info['sse'] / (n - 4))
    from scipy.stats import f as f_dist
    f_pval = 1 - f_dist.cdf(f_stat, 2, n - 4)
    print(f"  F-statistic: {f_stat:.4f}, p-value: {f_pval:.6f}")
