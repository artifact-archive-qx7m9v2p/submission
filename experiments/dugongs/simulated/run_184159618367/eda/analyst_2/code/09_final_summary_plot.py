"""
Final Summary Visualization - Key Insights in One Figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
x = data['x'].values
y = data['Y'].values

# Define best models
def asymptotic(x, a, b, c):
    return a - b * np.exp(-c * x)

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fit models
popt_asymp, _ = curve_fit(asymptotic, x, y, p0=[2.5, 1, 0.1], maxfev=5000)
popt_cubic, _ = curve_fit(cubic, x, y)

# Create smooth predictions
x_smooth = np.linspace(x.min(), x.max(), 200)
y_asymp = asymptotic(x_smooth, *popt_asymp)
y_cubic = cubic(x_smooth, *popt_cubic)

# Log-log transformation
x_log = np.log(x)
y_log = np.log(y)
z_log = np.polyfit(x_log, y_log, 1)
x_smooth_log = np.log(x_smooth)
y_log_pred = z_log[0] * x_smooth_log + z_log[1]
y_powerlaw = np.exp(y_log_pred)

# Create comprehensive summary figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# Main title
fig.suptitle('EDA Summary: x-Y Relationship Analysis\nKey Finding: Diminishing Returns / Asymptotic Pattern',
             fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# Row 1: Data Overview and Pattern
# ============================================================================

# 1. Main scatter with best models
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(x, y, alpha=0.7, s=120, edgecolors='black', linewidths=1,
           label='Observed Data (n=27)', color='steelblue', zorder=3)
ax1.plot(x_smooth, y_asymp, 'r-', linewidth=3,
        label=f'Asymptotic: Y=2.57-1.02exp(-0.20x), R²=0.89', zorder=2)
ax1.plot(x_smooth, y_cubic, 'g--', linewidth=2.5, alpha=0.8,
        label=f'Cubic Polynomial, R²=0.90', zorder=2)

# Add asymptote line
ax1.axhline(y=popt_asymp[0], color='red', linestyle=':', alpha=0.5,
           linewidth=2, label=f'Asymptote ≈ {popt_asymp[0]:.2f}')

# Highlight regime shift
ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.6, linewidth=2,
           label='Regime Shift (x≈10)')

ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('Y', fontsize=13, fontweight='bold')
ax1.set_title('Best-Fitting Models: Asymptotic vs Cubic', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# 2. Log-Log Transformation
ax2 = fig.add_subplot(gs[0, 2:])
ax2.scatter(x_log, y_log, alpha=0.7, s=120, edgecolors='black', linewidths=1,
           color='coral', zorder=3)
ax2.plot(x_smooth_log, y_log_pred, 'b-', linewidth=3,
        label=f'Linear fit: r=0.92, R²=0.84', zorder=2)

textstr = f'Power Law Form:\nlog(Y) = {z_log[1]:.3f} + {z_log[0]:.3f}*log(x)\nY = {np.exp(z_log[1]):.3f} * x^{z_log[0]:.3f}'
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax2.set_xlabel('log(x)', fontsize=13, fontweight='bold')
ax2.set_ylabel('log(Y)', fontsize=13, fontweight='bold')
ax2.set_title('Log-Log Transformation: Near-Perfect Linearity', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# ============================================================================
# Row 2: Evidence for Non-Linearity
# ============================================================================

# 3. Segmentation by x range
ax3 = fig.add_subplot(gs[1, 0])

# Define segments
seg1 = data[data['x'] < 10]
seg2 = data[(data['x'] >= 10) & (data['x'] < 20)]
seg3 = data[data['x'] >= 20]

ax3.scatter(seg1['x'], seg1['Y'], alpha=0.8, s=100, color='green',
           edgecolors='black', linewidths=1, label=f'x<10 (n={len(seg1)})', zorder=3)
ax3.scatter(seg2['x'], seg2['Y'], alpha=0.8, s=100, color='orange',
           edgecolors='black', linewidths=1, label=f'10≤x<20 (n={len(seg2)})', zorder=3)
ax3.scatter(seg3['x'], seg3['Y'], alpha=0.8, s=100, color='red',
           edgecolors='black', linewidths=1, label=f'x≥20 (n={len(seg3)})', zorder=3)

# Fit lines to each segment
if len(seg1) >= 2:
    z1 = np.polyfit(seg1['x'], seg1['Y'], 1)
    x1_line = np.linspace(seg1['x'].min(), seg1['x'].max(), 50)
    ax3.plot(x1_line, z1[0]*x1_line + z1[1], 'g-', linewidth=2.5, alpha=0.7)
    ax3.text(5, 1.85, f'r={seg1["x"].corr(seg1["Y"]):.2f}\nslope={z1[0]:.3f}',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

if len(seg2) >= 2:
    z2 = np.polyfit(seg2['x'], seg2['Y'], 1)
    x2_line = np.linspace(seg2['x'].min(), seg2['x'].max(), 50)
    ax3.plot(x2_line, z2[0]*x2_line + z2[1], 'orange', linewidth=2.5, alpha=0.7, linestyle='--')
    ax3.text(14, 2.35, f'r={seg2["x"].corr(seg2["Y"]):.2f}\nslope≈0',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('Y', fontsize=11)
ax3.set_title('Regime Shift: Strong Early, Flat Late', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Correlation by x range
ax4 = fig.add_subplot(gs[1, 1])
ranges = [(1, 10), (10, 20), (20, 32)]
range_labels = ['[1,10)', '[10,20)', '[20,32)']
correlations = []
ns = []

for x_min, x_max in ranges:
    subset = data[(data['x'] >= x_min) & (data['x'] < x_max)]
    if len(subset) >= 3:
        corr = subset['x'].corr(subset['Y'])
        correlations.append(corr)
        ns.append(len(subset))
    else:
        correlations.append(0)
        ns.append(0)

colors_bar = ['green', 'orange', 'red']
bars = ax4.bar(range_labels, correlations, alpha=0.7, edgecolor='black',
              linewidth=2, color=colors_bar)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add n labels
for i, (bar, n) in enumerate(zip(bars, ns)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'n={n}', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

ax4.set_ylabel('Pearson Correlation', fontsize=11)
ax4.set_xlabel('X Range', fontsize=11)
ax4.set_title('Local Correlation by X Range', fontsize=12, fontweight='bold')
ax4.set_ylim([-1, 1])
ax4.grid(True, alpha=0.3, axis='y')

# 5. Model comparison
ax5 = fig.add_subplot(gs[1, 2])

models = ['Linear', 'Sqrt(x)', 'Log(x)', 'Quadratic', 'Asymptotic', 'Cubic']
r2_values = [0.518, 0.707, 0.829, 0.862, 0.889, 0.898]
colors_models = ['red' if r2 < 0.7 else 'orange' if r2 < 0.85 else 'green' for r2 in r2_values]

bars = ax5.barh(models, r2_values, alpha=0.7, edgecolor='black', linewidth=1.5,
               color=colors_models)
ax5.axvline(x=0.85, color='blue', linestyle='--', linewidth=2, alpha=0.5,
           label='R²=0.85 threshold')
ax5.set_xlabel('R² (Variance Explained)', fontsize=11)
ax5.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax5.set_xlim([0, 1])
ax5.legend(loc='lower right', fontsize=9)
ax5.grid(True, alpha=0.3, axis='x')

# 6. Residuals from best model
ax6 = fig.add_subplot(gs[1, 3])
residuals_asymp = y - asymptotic(x, *popt_asymp)
ax6.scatter(x, residuals_asymp, alpha=0.7, s=100, edgecolors='black',
           linewidths=1, color='purple')
ax6.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax6.axhline(y=np.std(residuals_asymp), color='gray', linestyle=':', linewidth=1.5)
ax6.axhline(y=-np.std(residuals_asymp), color='gray', linestyle=':', linewidth=1.5)

ax6.set_xlabel('x', fontsize=11)
ax6.set_ylabel('Residuals', fontsize=11)
ax6.set_title('Asymptotic Model Residuals', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

textstr = f'RMSE: {np.sqrt(np.mean(residuals_asymp**2)):.4f}\nStd: {np.std(residuals_asymp):.4f}'
ax6.text(0.05, 0.95, textstr, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

# ============================================================================
# Row 3: Data Quality and Recommendations
# ============================================================================

# 7. Variability at repeated x
ax7 = fig.add_subplot(gs[2, 0])
repeated_x = data['x'].value_counts()
repeated_x = repeated_x[repeated_x > 1].index

for x_val in sorted(repeated_x):
    subset = data[data['x'] == x_val]
    y_values = subset['Y'].values
    ax7.scatter([x_val] * len(y_values), y_values, s=150, alpha=0.7,
               edgecolors='black', linewidths=1.5)
    if len(y_values) > 1:
        ax7.plot([x_val, x_val], [y_values.min(), y_values.max()],
                'k-', alpha=0.5, linewidth=3)

ax7.set_xlabel('x', fontsize=11)
ax7.set_ylabel('Y', fontsize=11)
ax7.set_title('Variability at Repeated X Values', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Summary statistics table
ax8 = fig.add_subplot(gs[2, 1])
ax8.axis('off')

summary_text = f"""
KEY STATISTICS

Data:
  • n = 27 observations
  • Unique x values: 20
  • x range: [1.0, 31.5]
  • Y range: [1.71, 2.63]

Correlation:
  • Overall Pearson: 0.72
  • x<10: r = 0.94 ★
  • x≥10: r = -0.03

Best Models:
  1. Asymptotic: R²=0.89
  2. Cubic: R²=0.90
  3. Log-Log: R²=0.84

Key Finding:
  Diminishing returns
  with plateau at x≥10
"""

ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

# 9. Recommendations
ax9 = fig.add_subplot(gs[2, 2:])
ax9.axis('off')

recommendations_text = """
MODELING RECOMMENDATIONS

Recommended Approaches (in order):

1. GAUSSIAN PROCESS REGRESSION
   → Captures smooth non-linearity without assuming functional form
   → Naturally handles heteroscedasticity
   → Provides uncertainty quantification

2. NON-LINEAR LEAST SQUARES (Asymptotic Model)
   → Y = a - b*exp(-c*x)
   → Best balance of interpretability and fit (R²=0.89)
   → Clear meaning: Y approaches 2.57 as x→∞

3. POLYNOMIAL REGRESSION (Quadratic)
   → Y = β₀ + β₁*x + β₂*x²
   → Good fit (R²=0.86), easy to implement
   → Standard inference available

4. LOG-LOG TRANSFORMED LINEAR
   → log(Y) = α + β*log(x)
   → Near-perfect linearity (r=0.92)
   → Simple, robust

AVOID:
  ✗ Simple linear regression (R²=0.52 - inadequate)
  ✗ High-degree polynomials (overfitting risk)

DATA COLLECTION PRIORITIES:
  • More observations at x>20 to confirm plateau
  • Replicates across all x to characterize variability
  • Dense sampling around x=10 to pinpoint transition
"""

ax9.text(0.05, 0.95, recommendations_text, transform=ax9.transAxes,
        fontsize=10.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2, pad=1))

# Save
plt.savefig('/workspace/eda/analyst_2/visualizations/00_SUMMARY_comprehensive.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("="*60)
print("FINAL SUMMARY VISUALIZATION CREATED")
print("="*60)
print("\nSaved: 00_SUMMARY_comprehensive.png")
print("\nThis figure synthesizes all key findings:")
print("  • Best models and their fits")
print("  • Evidence for non-linearity")
print("  • Regime shift at x≈10")
print("  • Model performance comparison")
print("  • Data quality assessment")
print("  • Modeling recommendations")
print("\n" + "="*60)
