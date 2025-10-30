"""
Comprehensive Summary Figure
Analyst 1 - Final Summary
Single figure summarizing all key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Create comprehensive figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================
# Panel 1: Main relationship with best fit (top left, spanning 2 columns)
# ============================================================
ax1 = fig.add_subplot(gs[0, :2])

# Scatter
ax1.scatter(data['x'], data['Y'], s=100, alpha=0.6, color='steelblue',
           edgecolors='black', linewidth=0.5, label='Observed data', zorder=3)

# Broken-stick fit
breakpoint = 9.5
data_low = data[data['x'] <= breakpoint]
data_high = data[data['x'] > breakpoint]
coeffs_low = np.polyfit(data_low['x'], data_low['Y'], 1)
coeffs_high = np.polyfit(data_high['x'], data_high['Y'], 1)

x_plot_low = np.linspace(data['x'].min(), breakpoint, 100)
x_plot_high = np.linspace(breakpoint, data['x'].max(), 100)
y_plot_low = coeffs_low[0] * x_plot_low + coeffs_low[1]
y_plot_high = coeffs_high[0] * x_plot_high + coeffs_high[1]

ax1.plot(x_plot_low, y_plot_low, 'r-', linewidth=3, alpha=0.8,
         label=f'Segment 1 (x≤{breakpoint}): slope=0.078')
ax1.plot(x_plot_high, y_plot_high, 'purple', linewidth=3, alpha=0.8,
         label=f'Segment 2 (x>{breakpoint}): slope≈0')
ax1.axvline(breakpoint, color='orange', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Breakpoint at x={breakpoint}')

ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('Y', fontsize=13, fontweight='bold')
ax1.set_title('A. Best Model: Piecewise Linear (R²=0.904)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotations
ax1.text(0.05, 0.95, 'Rapid increase phase', transform=ax1.transAxes,
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax1.text(0.65, 0.95, 'Plateau phase', transform=ax1.transAxes,
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# ============================================================
# Panel 2: Model comparison (top right)
# ============================================================
ax2 = fig.add_subplot(gs[0, 2])

models = ['Broken-stick', 'Quadratic', 'Logarithmic', 'Saturation', 'Linear']
r2_values = [0.904, 0.862, 0.829, 0.816, 0.518]
colors_bar = ['darkgreen', 'green', 'orange', 'coral', 'red']

bars = ax2.barh(models, r2_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axvline(0.7, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='R²=0.7')
ax2.set_xlabel('R² (higher is better)', fontsize=11, fontweight='bold')
ax2.set_title('B. Model Performance', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlim(0, 1)
ax2.grid(True, alpha=0.3, axis='x')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, r2_values)):
    ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold', fontsize=10)

# ============================================================
# Panel 3: Segmented means (middle left)
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])

segments = ['Low\n(x≤7)', 'Mid\n(7<x≤13)', 'High\n(x>13)']
means = [1.968, 2.483, 2.509]
stds = [0.179, 0.109, 0.089]
ns = [9, 10, 8]

colors_seg = ['green', 'orange', 'red']
bars3 = ax3.bar(segments, means, yerr=stds, color=colors_seg, alpha=0.7,
               edgecolor='black', linewidth=1.5, capsize=10, error_kw={'linewidth': 2})

ax3.set_ylabel('Y (mean ± SD)', fontsize=11, fontweight='bold')
ax3.set_title('C. Y by x Range', fontsize=13, fontweight='bold', pad=10)
ax3.set_ylim(1.5, 2.8)
ax3.grid(True, alpha=0.3, axis='y')

# Add n labels
for i, (bar, n) in enumerate(zip(bars3, ns)):
    ax3.text(i, means[i] + stds[i] + 0.05, f'n={n}', ha='center', fontweight='bold', fontsize=10)

# Add delta annotations
ax3.annotate('', xy=(0.5, 2.25), xytext=(0.5, 2.05),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax3.text(0.65, 2.15, '+0.52', fontsize=10, fontweight='bold', color='darkgreen')

ax3.annotate('', xy=(1.5, 2.55), xytext=(1.5, 2.49),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax3.text(1.65, 2.52, '+0.03', fontsize=10, fontweight='bold', color='darkred')

# ============================================================
# Panel 4: Residual pattern (middle center)
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])

# Linear model residuals
coeffs_lin = np.polyfit(data['x'], data['Y'], 1)
y_pred_lin = coeffs_lin[0] * data['x'] + coeffs_lin[1]
resid_lin = data['Y'] - y_pred_lin

ax4.scatter(data['x'], resid_lin, s=80, alpha=0.6, color='steelblue',
           edgecolors='black', linewidth=0.5)
ax4.axhline(0, color='red', linestyle='--', linewidth=2)

# Add smoother
from scipy.signal import savgol_filter
x_sorted_idx = np.argsort(data['x'])
if len(data) >= 7:
    window = 7
    smooth_resid = savgol_filter(resid_lin.values[x_sorted_idx], window, 2)
    ax4.plot(data['x'].values[x_sorted_idx], smooth_resid,
            color='green', linewidth=3, alpha=0.8, label='Smoother (shows U-shape)')

ax4.set_xlabel('x', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residuals (linear model)', fontsize=11, fontweight='bold')
ax4.set_title('D. Linear Model Lack of Fit', fontsize=13, fontweight='bold', pad=10)
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

ax4.text(0.05, 0.95, 'U-shaped pattern\n= systematic bias',
        transform=ax4.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================
# Panel 5: Variance structure (middle right)
# ============================================================
ax5 = fig.add_subplot(gs[1, 2])

# Pure error vs model error
error_types = ['Pure\nError', 'Linear\nModel\nError']
error_sds = [0.075, 0.197]
colors_err = ['green', 'red']

bars5 = ax5.bar(error_types, error_sds, color=colors_err, alpha=0.7,
               edgecolor='black', linewidth=1.5)

ax5.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
ax5.set_title('E. Error Structure', fontsize=13, fontweight='bold', pad=10)
ax5.set_ylim(0, 0.25)
ax5.grid(True, alpha=0.3, axis='y')

# Add values
for bar, val in zip(bars5, error_sds):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', fontweight='bold', fontsize=11)

# Add ratio annotation
ax5.text(0.5, 0.22, 'Ratio = 6.8×\n(lack of fit)',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

# ============================================================
# Panel 6: Distribution of x (bottom left)
# ============================================================
ax6 = fig.add_subplot(gs[2, 0])

ax6.hist(data['x'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
ax6.axvline(data['x'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={data["x"].mean():.1f}')
ax6.axvline(data['x'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median={data["x"].median():.1f}')
ax6.set_xlabel('x', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('F. Predictor Distribution', fontsize=13, fontweight='bold', pad=10)
ax6.legend(loc='best', fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# ============================================================
# Panel 7: Distribution of Y (bottom center)
# ============================================================
ax7 = fig.add_subplot(gs[2, 1])

ax7.hist(data['Y'], bins=15, edgecolor='black', alpha=0.7, color='lightcoral')
ax7.axvline(data['Y'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={data["Y"].mean():.2f}')
ax7.axvline(data['Y'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median={data["Y"].median():.2f}')
ax7.set_xlabel('Y', fontsize=11, fontweight='bold')
ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7.set_title('G. Response Distribution', fontsize=13, fontweight='bold', pad=10)
ax7.legend(loc='best', fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# ============================================================
# Panel 8: Key statistics summary (bottom right)
# ============================================================
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

summary_text = f"""
KEY FINDINGS SUMMARY

Sample Size: n = 27
Data Quality: Excellent (no issues)

Relationship: NONLINEAR SATURATION
• Rapid increase: x = 1-10
• Plateau: x > 10
• Breakpoint: x ≈ 9.5

Correlations:
• Pearson r = 0.720***
• Spearman ρ = 0.782***

Model Performance:
• Linear R² = 0.52 (poor)
• Best nonlinear R² = 0.90 (excellent)

Variance:
• Homoscedastic ✓
• Normal residuals ✓
• Pure error SD = 0.075

Influential Points:
• x = 31.5 (high leverage)
• No problematic outliers

RECOMMENDATION:
Use piecewise, quadratic, or
logarithmic model. Avoid linear.
"""

ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=15))

# Overall title
fig.suptitle('Comprehensive EDA Summary: x-Y Relationship Analysis',
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/workspace/eda/analyst_1/visualizations/00_comprehensive_summary.png',
           dpi=150, bbox_inches='tight')
plt.close()

print("Created: 00_comprehensive_summary.png")
print("\nThis single figure summarizes all key findings from the EDA:")
print("  A. Best model fit (piecewise linear)")
print("  B. Model performance comparison")
print("  C. Segmented analysis showing plateau")
print("  D. Residual pattern showing lack of fit")
print("  E. Error structure (pure vs. model)")
print("  F. Predictor distribution")
print("  G. Response distribution")
print("  H. Key statistics summary")
