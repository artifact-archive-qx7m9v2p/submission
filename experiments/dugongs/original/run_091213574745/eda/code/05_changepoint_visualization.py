"""
Changepoint Visualization
=========================
Visualize the piecewise linear relationship
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

# Fit linear model
z_linear = np.polyfit(x, y, 1)
p_linear = np.poly1d(z_linear)

# Fit piecewise model at x=7
breakpoint = 7.0
mask1 = x <= breakpoint
mask2 = x > breakpoint

z1 = np.polyfit(x[mask1], y[mask1], 1)
p1 = np.poly1d(z1)

z2 = np.polyfit(x[mask2], y[mask2], 1)
p2 = np.poly1d(z2)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Changepoint Analysis: Evidence for Two-Regime Model', fontsize=16, fontweight='bold')

# 1. Piecewise vs linear comparison
ax = axes[0, 0]
ax.scatter(x[mask1], y[mask1], alpha=0.7, s=100, edgecolors='black', linewidth=0.8,
           label=f'Regime 1 (x <= {breakpoint})', color='royalblue')
ax.scatter(x[mask2], y[mask2], alpha=0.7, s=100, edgecolors='black', linewidth=0.8,
           label=f'Regime 2 (x > {breakpoint})', color='coral')

x_line = np.linspace(x.min(), x.max(), 100)
x_line1 = x_line[x_line <= breakpoint]
x_line2 = x_line[x_line > breakpoint]

ax.plot(x_line1, p1(x_line1), 'b-', linewidth=3, label=f'Regime 1 fit: slope={z1[0]:.4f}')
ax.plot(x_line2, p2(x_line2), 'r-', linewidth=3, label=f'Regime 2 fit: slope={z2[0]:.4f}')
ax.plot(x_line, p_linear(x_line), 'g--', linewidth=2, alpha=0.5, label='Linear fit (for comparison)')
ax.axvline(breakpoint, color='purple', linestyle=':', linewidth=2, label=f'Changepoint at x={breakpoint}')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Piecewise Linear Model', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Residual comparison
ax = axes[0, 1]
residuals_linear = y - p_linear(x)
residuals_piecewise = np.where(mask1, y - p1(x), y - p2(x))

ax.scatter(x, residuals_linear, alpha=0.6, s=80, label='Linear residuals', color='green')
ax.scatter(x, residuals_piecewise, alpha=0.6, s=80, label='Piecewise residuals', color='purple', marker='^')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(breakpoint, color='purple', linestyle=':', linewidth=2)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Residuals', fontsize=13)
ax.set_title('Residual Comparison', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics
rmse_linear = np.sqrt(np.mean(residuals_linear**2))
rmse_piecewise = np.sqrt(np.mean(residuals_piecewise**2))
ax.text(0.05, 0.95, f'RMSE Linear: {rmse_linear:.4f}\nRMSE Piecewise: {rmse_piecewise:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

# 3. Q-Q plots comparison
ax = axes[1, 0]
stats.probplot(residuals_piecewise, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Piecewise Model Residuals', fontsize=13)
ax.grid(True, alpha=0.3)

# 4. Residual histogram comparison
ax = axes[1, 1]
ax.hist(residuals_linear, bins=12, alpha=0.5, label='Linear', color='green', edgecolor='black')
ax.hist(residuals_piecewise, bins=12, alpha=0.5, label='Piecewise', color='purple', edgecolor='black')
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Residual', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('Residual Distribution Comparison', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/07_changepoint_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional diagnostic plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Rate of Change Analysis by Region', fontsize=14, fontweight='bold')

# Sort data for sequential analysis
df_sorted = df.sort_values('x').reset_index(drop=True)

# Calculate local slopes
local_slopes = []
x_midpoints = []
colors = []
for i in range(len(df_sorted) - 1):
    dx = df_sorted.loc[i+1, 'x'] - df_sorted.loc[i, 'x']
    dy = df_sorted.loc[i+1, 'Y'] - df_sorted.loc[i, 'Y']
    if dx != 0:
        slope = dy / dx
        local_slopes.append(slope)
        midpoint = (df_sorted.loc[i, 'x'] + df_sorted.loc[i+1, 'x']) / 2
        x_midpoints.append(midpoint)
        colors.append('blue' if midpoint <= breakpoint else 'red')

# Plot local slopes with color coding
ax = axes[0]
for i, (xm, slope, color) in enumerate(zip(x_midpoints, local_slopes, colors)):
    ax.scatter(xm, slope, alpha=0.6, s=80, color=color, edgecolors='black', linewidth=0.5)

ax.axvline(breakpoint, color='purple', linestyle=':', linewidth=2, label=f'Changepoint at x={breakpoint}')
ax.axhline(z1[0], color='blue', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Regime 1 slope: {z1[0]:.4f}')
ax.axhline(z2[0], color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Regime 2 slope: {z2[0]:.4f}')
ax.set_xlabel('x (midpoint)', fontsize=12)
ax.set_ylabel('Local Slope (dY/dx)', fontsize=12)
ax.set_title('Local Rate of Change', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Moving window slope (if enough points)
ax = axes[1]
window_size = 5
if len(df_sorted) >= window_size:
    window_slopes = []
    window_x = []
    for i in range(len(df_sorted) - window_size + 1):
        window = df_sorted.iloc[i:i+window_size]
        if len(window) >= 3:
            z_window = np.polyfit(window['x'], window['Y'], 1)
            window_slopes.append(z_window[0])
            window_x.append(window['x'].mean())

    window_colors = ['blue' if wx <= breakpoint else 'red' for wx in window_x]
    for wx, ws, wc in zip(window_x, window_slopes, window_colors):
        ax.scatter(wx, ws, alpha=0.6, s=100, color=wc, edgecolors='black', linewidth=0.8)

    ax.axvline(breakpoint, color='purple', linestyle=':', linewidth=2, label=f'Changepoint at x={breakpoint}')
    ax.axhline(z1[0], color='blue', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Regime 1 slope: {z1[0]:.4f}')
    ax.axhline(z2[0], color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Regime 2 slope: {z2[0]:.4f}')
    ax.set_xlabel('x (window center)', fontsize=12)
    ax.set_ylabel(f'Moving Window Slope (window={window_size})', fontsize=12)
    ax.set_title('Smoothed Rate of Change', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/08_rate_of_change.png', dpi=300, bbox_inches='tight')
plt.close()

print("=" * 80)
print("CHANGEPOINT MODEL DETAILS")
print("=" * 80)

print(f"\nBreakpoint location: x = {breakpoint}")
print(f"\nRegime 1 (x <= {breakpoint}):")
print(f"  Number of observations: {np.sum(mask1)}")
print(f"  Equation: Y = {z1[0]:.6f} * x + {z1[1]:.6f}")
print(f"  Slope: {z1[0]:.6f}")
print(f"  x range: [{x[mask1].min():.2f}, {x[mask1].max():.2f}]")
print(f"  Y range: [{y[mask1].min():.2f}, {y[mask1].max():.2f}]")

print(f"\nRegime 2 (x > {breakpoint}):")
print(f"  Number of observations: {np.sum(mask2)}")
print(f"  Equation: Y = {z2[0]:.6f} * x + {z2[1]:.6f}")
print(f"  Slope: {z2[0]:.6f}")
print(f"  x range: [{x[mask2].min():.2f}, {x[mask2].max():.2f}]")
print(f"  Y range: [{y[mask2].min():.2f}, {y[mask2].max():.2f}]")

print(f"\nSlope ratio: {z1[0] / z2[0]:.2f}x")
print(f"Regime 1 slope is {z1[0] / z2[0]:.1f} times steeper than Regime 2")

# Value at breakpoint from each regime
y_at_break_regime1 = p1(breakpoint)
y_at_break_regime2 = p2(breakpoint)
print(f"\nPredicted Y at breakpoint:")
print(f"  From Regime 1 equation: {y_at_break_regime1:.4f}")
print(f"  From Regime 2 equation: {y_at_break_regime2:.4f}")
print(f"  Difference (jump): {abs(y_at_break_regime1 - y_at_break_regime2):.4f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("The data shows strong evidence for two distinct regimes:")
print(f"1. A steep growth phase (x <= {breakpoint}) with rapid increase")
print(f"2. A plateau/saturation phase (x > {breakpoint}) with slow growth")
print("\nThis suggests an asymptotic or saturation-type relationship.")
