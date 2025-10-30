"""
Summary Visualization
====================
Create a single comprehensive figure summarizing key EDA findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data.csv')
x = df['x'].values
y = df['Y'].values

# Fit models
z_linear = np.polyfit(x, y, 1)
p_linear = np.poly1d(z_linear)

x_log = np.log(x)
z_log = np.polyfit(x_log, y, 1)
p_log = np.poly1d(z_log)

# Piecewise model
breakpoint = 7.0
mask1 = x <= breakpoint
mask2 = x > breakpoint
z1 = np.polyfit(x[mask1], y[mask1], 1)
p1 = np.poly1d(z1)
z2 = np.polyfit(x[mask2], y[mask2], 1)
p2 = np.poly1d(z2)

# Create 2x2 summary figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('EDA Summary: Y vs x Relationship (n=27)',
             fontsize=18, fontweight='bold', y=0.98)

# Panel 1: Data and Best Fit Models
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(x, y, alpha=0.7, s=100, edgecolors='black', linewidth=0.8,
           label='Observed data', zorder=5, color='steelblue')

x_line = np.linspace(x.min(), x.max(), 200)
x_log_line = np.log(x_line)

# Linear (inadequate)
ax1.plot(x_line, p_linear(x_line), 'r--', linewidth=2, alpha=0.5,
        label=f'Linear: R²={0.677:.3f}')

# Logarithmic (best)
ax1.plot(x_line, p_log(x_log_line), 'g-', linewidth=3,
        label=f'Logarithmic: R²={0.897:.3f} (BEST)', zorder=4)

# Mark outlier
outlier_idx = 26
ax1.scatter(x[outlier_idx], y[outlier_idx], s=200, facecolors='none',
           edgecolors='red', linewidth=3, label='Influential outlier')

ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('Y', fontsize=13, fontweight='bold')
ax1.set_title('A. Best Fit Models Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)

# Add text box with key stats
textstr = 'Key Statistics:\n'
textstr += f'n = 27\n'
textstr += f'Pearson r = 0.823\n'
textstr += f'Spearman ρ = 0.920\n'
textstr += f'Log model RMSE = 0.087'
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
        verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: Two-Regime Structure
ax2 = fig.add_subplot(gs[0, 1])

# Color by regime
ax2.scatter(x[mask1], y[mask1], alpha=0.7, s=120, edgecolors='black', linewidth=0.8,
           label=f'Regime 1 (x≤7): slope={z1[0]:.3f}', color='royalblue', zorder=5)
ax2.scatter(x[mask2], y[mask2], alpha=0.7, s=120, edgecolors='black', linewidth=0.8,
           label=f'Regime 2 (x>7): slope={z2[0]:.3f}', color='coral', zorder=5)

# Fit lines
x_line1 = x_line[x_line <= breakpoint]
x_line2 = x_line[x_line > breakpoint]
ax2.plot(x_line1, p1(x_line1), 'b-', linewidth=3, alpha=0.8, zorder=4)
ax2.plot(x_line2, p2(x_line2), 'r-', linewidth=3, alpha=0.8, zorder=4)

# Changepoint
ax2.axvline(breakpoint, color='purple', linestyle=':', linewidth=3,
           label=f'Changepoint: x={breakpoint}', zorder=3)

ax2.set_xlabel('x', fontsize=13, fontweight='bold')
ax2.set_ylabel('Y', fontsize=13, fontweight='bold')
ax2.set_title('B. Two-Regime Piecewise Model', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)

# Add text box
textstr = 'Changepoint Analysis:\n'
textstr += f'F-statistic = 22.4\n'
textstr += f'p-value < 0.0001\n'
textstr += f'SSE reduction = 66%\n'
textstr += f'Slope ratio = 6.8:1'
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
        verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Panel 3: Residual Diagnostics
ax3 = fig.add_subplot(gs[1, 0])

# Residuals from linear vs log model
residuals_linear = y - p_linear(x)
residuals_log = y - p_log(x_log)

# Side-by-side boxplots
positions = [1, 2]
bp = ax3.boxplot([residuals_linear, residuals_log],
                  positions=positions,
                  widths=0.6,
                  labels=['Linear\nModel', 'Logarithmic\nModel'],
                  patch_artist=True)

# Color boxes
colors = ['lightcoral', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Residuals', fontsize=13, fontweight='bold')
ax3.set_title('C. Residual Distribution Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add RMSE text
rmse_linear = np.sqrt(np.mean(residuals_linear**2))
rmse_log = np.sqrt(np.mean(residuals_log**2))
textstr = f'RMSE Comparison:\n'
textstr += f'Linear: {rmse_linear:.4f}\n'
textstr += f'Log: {rmse_log:.4f}\n'
textstr += f'Improvement: {100*(rmse_linear-rmse_log)/rmse_linear:.1f}%'
ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes,
        verticalalignment='top', horizontalalignment='right', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 4: Model Performance Summary
ax4 = fig.add_subplot(gs[1, 1])

# Model comparison table
models = ['Linear', 'Square\nroot', 'Logarithmic', 'Quadratic', 'Cubic', 'Asymptotic']
r2_values = [0.677, 0.826, 0.897, 0.873, 0.880, 0.889]
params = [2, 2, 2, 3, 4, 3]

# Create bar chart
bars = ax4.barh(models, r2_values, color=['red', 'orange', 'green', 'yellow', 'yellow', 'lightblue'],
               alpha=0.7, edgecolor='black', linewidth=1)

# Highlight best
bars[2].set_color('darkgreen')
bars[2].set_alpha(0.9)

ax4.axvline(0.677, color='red', linestyle='--', linewidth=2, alpha=0.5,
           label='Linear baseline')
ax4.set_xlabel('R² (Coefficient of Determination)', fontsize=13, fontweight='bold')
ax4.set_title('D. Model Performance Comparison', fontsize=14, fontweight='bold')
ax4.set_xlim(0.6, 0.92)
ax4.grid(True, alpha=0.3, axis='x')

# Add parameter counts
for i, (model, r2, param) in enumerate(zip(models, r2_values, params)):
    ax4.text(r2 + 0.005, i, f'{r2:.3f} ({param}p)',
            va='center', fontsize=9, fontweight='bold')

# Legend
ax4.legend(fontsize=10)

# Add recommendation box
textstr = 'RECOMMENDATION:\n'
textstr += 'Logarithmic model\n'
textstr += '• Best R² (0.897)\n'
textstr += '• Parsimonious (2 params)\n'
textstr += '• Theoretically sound\n'
textstr += '• Y ~ β₀ + β₁*log(x)'
ax4.text(0.02, 0.02, textstr, transform=ax4.transAxes,
        verticalalignment='bottom', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.savefig('/workspace/eda/visualizations/00_eda_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("=" * 80)
print("SUMMARY VISUALIZATION CREATED")
print("=" * 80)
print("\nSaved to: /workspace/eda/visualizations/00_eda_summary.png")
print("\nThis single figure summarizes:")
print("  A. Best fit models comparison (linear vs logarithmic)")
print("  B. Two-regime piecewise structure with changepoint")
print("  C. Residual distribution improvements")
print("  D. Performance comparison across 6 models")
print("\nKey finding: Logarithmic model (Y ~ β₀ + β₁*log(x)) provides")
print("best balance of fit quality, parsimony, and interpretability.")
print("=" * 80)
