"""
Hypothesis Testing: Relationship Structure
Analyst 1 - Round 2
Testing competing hypotheses about the x-Y relationship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("=" * 70)
print("HYPOTHESIS TESTING: RELATIONSHIP STRUCTURE")
print("=" * 70)

# ============================================================
# HYPOTHESIS 1: Linear relationship
# ============================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 1: Linear relationship (Y = a + b*x)")
print("=" * 70)

coeffs_lin = np.polyfit(data['x'], data['Y'], 1)
y_pred_lin = coeffs_lin[0] * data['x'] + coeffs_lin[1]
ss_res_lin = np.sum((data['Y'] - y_pred_lin)**2)
ss_tot = np.sum((data['Y'] - data['Y'].mean())**2)
r2_lin = 1 - ss_res_lin / ss_tot
rmse_lin = np.sqrt(ss_res_lin / len(data))
aic_lin = len(data) * np.log(ss_res_lin / len(data)) + 2 * 2  # 2 parameters

print(f"Linear: Y = {coeffs_lin[1]:.4f} + {coeffs_lin[0]:.4f} * x")
print(f"  R-squared: {r2_lin:.4f}")
print(f"  RMSE: {rmse_lin:.4f}")
print(f"  AIC: {aic_lin:.2f}")

# ============================================================
# HYPOTHESIS 2: Quadratic relationship
# ============================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 2: Quadratic relationship (Y = a + b*x + c*x^2)")
print("=" * 70)

coeffs_quad = np.polyfit(data['x'], data['Y'], 2)
y_pred_quad = coeffs_quad[0] * data['x']**2 + coeffs_quad[1] * data['x'] + coeffs_quad[2]
ss_res_quad = np.sum((data['Y'] - y_pred_quad)**2)
r2_quad = 1 - ss_res_quad / ss_tot
rmse_quad = np.sqrt(ss_res_quad / len(data))
aic_quad = len(data) * np.log(ss_res_quad / len(data)) + 2 * 3  # 3 parameters

print(f"Quadratic: Y = {coeffs_quad[2]:.4f} + {coeffs_quad[1]:.4f}*x + {coeffs_quad[0]:.6f}*x^2")
print(f"  R-squared: {r2_quad:.4f}")
print(f"  RMSE: {rmse_quad:.4f}")
print(f"  AIC: {aic_quad:.2f}")
print(f"  Improvement over linear: ΔR² = {r2_quad - r2_lin:.4f}, ΔAIC = {aic_quad - aic_lin:.2f}")

# ============================================================
# HYPOTHESIS 3: Logarithmic relationship
# ============================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 3: Logarithmic relationship (Y = a + b*log(x))")
print("=" * 70)

coeffs_log = np.polyfit(np.log(data['x']), data['Y'], 1)
y_pred_log = coeffs_log[0] * np.log(data['x']) + coeffs_log[1]
ss_res_log = np.sum((data['Y'] - y_pred_log)**2)
r2_log = 1 - ss_res_log / ss_tot
rmse_log = np.sqrt(ss_res_log / len(data))
aic_log = len(data) * np.log(ss_res_log / len(data)) + 2 * 2

print(f"Logarithmic: Y = {coeffs_log[1]:.4f} + {coeffs_log[0]:.4f} * log(x)")
print(f"  R-squared: {r2_log:.4f}")
print(f"  RMSE: {rmse_log:.4f}")
print(f"  AIC: {aic_log:.2f}")
print(f"  Improvement over linear: ΔR² = {r2_log - r2_lin:.4f}, ΔAIC = {aic_log - aic_lin:.2f}")

# ============================================================
# HYPOTHESIS 4: Saturation/Asymptotic relationship (Michaelis-Menten type)
# ============================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 4: Saturation model (Y = Ymax * x / (K + x))")
print("=" * 70)

def michaelis_menten(x, ymax, K):
    return ymax * x / (K + x)

try:
    # Initial guesses
    ymax_init = data['Y'].max()
    K_init = data['x'].median()

    popt, pcov = curve_fit(michaelis_menten, data['x'], data['Y'],
                           p0=[ymax_init, K_init], maxfev=10000)
    ymax_fit, K_fit = popt

    y_pred_mm = michaelis_menten(data['x'], ymax_fit, K_fit)
    ss_res_mm = np.sum((data['Y'] - y_pred_mm)**2)
    r2_mm = 1 - ss_res_mm / ss_tot
    rmse_mm = np.sqrt(ss_res_mm / len(data))
    aic_mm = len(data) * np.log(ss_res_mm / len(data)) + 2 * 2

    print(f"Saturation: Y = {ymax_fit:.4f} * x / ({K_fit:.4f} + x)")
    print(f"  Ymax (asymptote): {ymax_fit:.4f}")
    print(f"  K (half-max): {K_fit:.4f}")
    print(f"  R-squared: {r2_mm:.4f}")
    print(f"  RMSE: {rmse_mm:.4f}")
    print(f"  AIC: {aic_mm:.2f}")
    print(f"  Improvement over linear: ΔR² = {r2_mm - r2_lin:.4f}, ΔAIC = {aic_mm - aic_lin:.2f}")

    fit_mm_success = True
except Exception as e:
    print(f"  Failed to fit: {e}")
    fit_mm_success = False

# ============================================================
# HYPOTHESIS 5: Broken-stick (piecewise linear)
# ============================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 5: Broken-stick model (piecewise linear)")
print("=" * 70)

# Try breakpoint at median x
breakpoint = data['x'].median()
data_low = data[data['x'] <= breakpoint]
data_high = data[data['x'] > breakpoint]

if len(data_low) >= 2 and len(data_high) >= 2:
    coeffs_low = np.polyfit(data_low['x'], data_low['Y'], 1)
    coeffs_high = np.polyfit(data_high['x'], data_high['Y'], 1)

    y_pred_bs = np.where(data['x'] <= breakpoint,
                         coeffs_low[0] * data['x'] + coeffs_low[1],
                         coeffs_high[0] * data['x'] + coeffs_high[1])

    ss_res_bs = np.sum((data['Y'] - y_pred_bs)**2)
    r2_bs = 1 - ss_res_bs / ss_tot
    rmse_bs = np.sqrt(ss_res_bs / len(data))
    aic_bs = len(data) * np.log(ss_res_bs / len(data)) + 2 * 5  # 5 parameters

    print(f"Breakpoint at x = {breakpoint:.2f}")
    print(f"  Segment 1 (x ≤ {breakpoint:.2f}): Y = {coeffs_low[1]:.4f} + {coeffs_low[0]:.4f}*x")
    print(f"  Segment 2 (x > {breakpoint:.2f}): Y = {coeffs_high[1]:.4f} + {coeffs_high[0]:.4f}*x")
    print(f"  R-squared: {r2_bs:.4f}")
    print(f"  RMSE: {rmse_bs:.4f}")
    print(f"  AIC: {aic_bs:.2f}")
    print(f"  Improvement over linear: ΔR² = {r2_bs - r2_lin:.4f}, ΔAIC = {aic_bs - aic_lin:.2f}")

    fit_bs_success = True
else:
    print("  Insufficient data in segments")
    fit_bs_success = False

# ============================================================
# MODEL COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY (lower AIC is better)")
print("=" * 70)

models = [
    ("Linear", r2_lin, rmse_lin, aic_lin),
    ("Quadratic", r2_quad, rmse_quad, aic_quad),
    ("Logarithmic", r2_log, rmse_log, aic_log),
]

if fit_mm_success:
    models.append(("Saturation", r2_mm, rmse_mm, aic_mm))
if fit_bs_success:
    models.append(("Broken-stick", r2_bs, rmse_bs, aic_bs))

models_df = pd.DataFrame(models, columns=['Model', 'R²', 'RMSE', 'AIC'])
models_df = models_df.sort_values('AIC')
models_df['ΔAIC'] = models_df['AIC'] - models_df['AIC'].min()

print(models_df.to_string(index=False))

# ============================================================
# VISUALIZATION: Model comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: All models overlaid
x_smooth = np.linspace(data['x'].min(), data['x'].max(), 200)

axes[0].scatter(data['x'], data['Y'], alpha=0.6, s=80, color='black',
                edgecolors='gray', linewidth=0.5, label='Observed data', zorder=5)

# Linear
y_smooth_lin = coeffs_lin[0] * x_smooth + coeffs_lin[1]
axes[0].plot(x_smooth, y_smooth_lin, '--', linewidth=2, alpha=0.8,
             label=f'Linear (AIC={aic_lin:.1f})', color='blue')

# Quadratic
y_smooth_quad = coeffs_quad[0] * x_smooth**2 + coeffs_quad[1] * x_smooth + coeffs_quad[2]
axes[0].plot(x_smooth, y_smooth_quad, '--', linewidth=2, alpha=0.8,
             label=f'Quadratic (AIC={aic_quad:.1f})', color='green')

# Logarithmic
y_smooth_log = coeffs_log[0] * np.log(x_smooth) + coeffs_log[1]
axes[0].plot(x_smooth, y_smooth_log, '--', linewidth=2, alpha=0.8,
             label=f'Logarithmic (AIC={aic_log:.1f})', color='red')

# Saturation if fitted
if fit_mm_success:
    y_smooth_mm = michaelis_menten(x_smooth, ymax_fit, K_fit)
    axes[0].plot(x_smooth, y_smooth_mm, '-', linewidth=2.5, alpha=0.9,
                 label=f'Saturation (AIC={aic_mm:.1f})', color='purple')

axes[0].set_xlabel('x', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Y', fontsize=12, fontweight='bold')
axes[0].set_title('Competing Model Fits', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right panel: Model metrics
models_plot = models_df.copy()
x_pos = np.arange(len(models_plot))

ax2 = axes[1]
ax2_twin = ax2.twinx()

bars1 = ax2.bar(x_pos - 0.2, models_plot['R²'], 0.4, label='R²',
                alpha=0.7, color='steelblue', edgecolor='black')
bars2 = ax2_twin.bar(x_pos + 0.2, models_plot['ΔAIC'], 0.4, label='ΔAIC',
                     alpha=0.7, color='coral', edgecolor='black')

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² (higher is better)', fontsize=11, fontweight='bold', color='steelblue')
ax2_twin.set_ylabel('ΔAIC (lower is better)', fontsize=11, fontweight='bold', color='coral')
ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_plot['Model'], rotation=45, ha='right')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax2_twin.tick_params(axis='y', labelcolor='coral')
ax2.grid(True, alpha=0.3, axis='y')

# Add legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/05_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("\nCreated: 05_model_comparison.png")
print("\n" + "=" * 70)
