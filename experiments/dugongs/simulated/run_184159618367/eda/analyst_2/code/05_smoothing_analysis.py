"""
Local vs Global Trends - LOWESS and Spline Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
x = data['x'].values
y = data['Y'].values

# Sort by x for smoothing
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_sorted = y[sort_idx]

print("="*60)
print("SMOOTHING AND LOCAL TREND ANALYSIS")
print("="*60)

# Manual LOWESS implementation (simple version)
def simple_lowess(x, y, frac=0.3):
    """Simple LOWESS implementation using weighted least squares"""
    n = len(x)
    window = int(np.ceil(frac * n))
    y_smooth = np.zeros(n)

    for i in range(n):
        # Get window around point
        left = max(0, i - window // 2)
        right = min(n, i + window // 2 + 1)

        # Get points in window
        x_window = x[left:right]
        y_window = y[left:right]

        # Calculate weights (tricube)
        dists = np.abs(x_window - x[i])
        max_dist = np.max(dists) if np.max(dists) > 0 else 1
        weights = (1 - (dists / max_dist)**3)**3

        # Fit weighted linear regression
        W = np.diag(weights)
        X = np.column_stack([np.ones(len(x_window)), x_window])
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y_window)
            y_smooth[i] = beta[0] + beta[1] * x[i]
        except:
            y_smooth[i] = np.mean(y_window)

    return y_smooth

# Apply different smoothing methods
smoothing_results = {}

# 1. LOWESS with different bandwidths
for frac in [0.2, 0.3, 0.5]:
    y_lowess = simple_lowess(x_sorted, y_sorted, frac=frac)
    smoothing_results[f'LOWESS (frac={frac})'] = y_lowess
    print(f"\nLOWESS (frac={frac})")
    print(f"  Range: [{y_lowess.min():.4f}, {y_lowess.max():.4f}]")

# 2. Spline smoothing with different smoothing factors
for s in [0.01, 0.05, 0.1]:
    try:
        spline = UnivariateSpline(x_sorted, y_sorted, s=s)
        y_spline = spline(x_sorted)
        smoothing_results[f'Spline (s={s})'] = y_spline
        print(f"\nSpline (s={s})")
        print(f"  Range: [{y_spline.min():.4f}, {y_spline.max():.4f}]")
    except Exception as e:
        print(f"\nSpline (s={s}) failed: {e}")

# 3. Moving average
for window in [3, 5, 7]:
    if window <= len(x_sorted):
        y_ma = np.convolve(y_sorted, np.ones(window)/window, mode='same')
        # Fix edges
        for i in range(window//2):
            y_ma[i] = np.mean(y_sorted[:i+window//2+1])
            y_ma[-(i+1)] = np.mean(y_sorted[-(i+window//2+1):])
        smoothing_results[f'Moving Avg (w={window})'] = y_ma
        print(f"\nMoving Average (window={window})")
        print(f"  Range: [{y_ma.min():.4f}, {y_ma.max():.4f}]")

print("\n" + "="*60)

# Visualize all smoothing methods
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Smoothing Methods Comparison - Local vs Global Trends',
             fontsize=16, fontweight='bold')

methods = list(smoothing_results.keys())
for idx, method_name in enumerate(methods[:6]):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    # Original data
    ax.scatter(x, y, alpha=0.5, s=60, edgecolors='black',
               linewidths=0.5, label='Data', color='lightgray', zorder=1)

    # Smoothed curve
    ax.plot(x_sorted, smoothing_results[method_name], 'r-',
            linewidth=3, label=method_name, zorder=2)

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title(method_name, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/07_smoothing_methods.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: 07_smoothing_methods.png")

# Compare LOWESS fractions
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(x, y, alpha=0.6, s=100, edgecolors='black',
           linewidths=1, label='Data', color='steelblue', zorder=3)

colors = ['red', 'green', 'purple']
for idx, frac in enumerate([0.2, 0.3, 0.5]):
    y_lowess = smoothing_results[f'LOWESS (frac={frac})']
    ax.plot(x_sorted, y_lowess, color=colors[idx], linewidth=2.5,
            label=f'LOWESS (frac={frac})', zorder=2, alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('LOWESS Smoothing with Different Bandwidths',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/08_lowess_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 08_lowess_comparison.png")

# Analyze curvature and rate of change
y_lowess = smoothing_results['LOWESS (frac=0.3)']

# First derivative (rate of change)
dx = np.diff(x_sorted)
dy = np.diff(y_lowess)
slopes = dy / dx
x_mid = (x_sorted[:-1] + x_sorted[1:]) / 2

# Second derivative (curvature)
d2y = np.diff(slopes)
dx2 = np.diff(x_mid)
curvature = d2y / dx2
x_mid2 = (x_mid[:-1] + x_mid[1:]) / 2

print("\n" + "="*60)
print("CURVATURE ANALYSIS (from LOWESS frac=0.3)")
print("="*60)
print(f"\nSlope (dy/dx) statistics:")
print(f"  Range: [{slopes.min():.6f}, {slopes.max():.6f}]")
print(f"  Mean: {slopes.mean():.6f}")
print(f"  Std: {slopes.std():.6f}")

print(f"\nCurvature (d2y/dx2) statistics:")
print(f"  Range: [{curvature.min():.6f}, {curvature.max():.6f}]")
print(f"  Mean: {curvature.mean():.6f}")
print(f"  Std: {curvature.std():.6f}")

# Find where slope changes significantly
slope_changes = np.abs(np.diff(slopes))
significant_changes = np.where(slope_changes > np.percentile(slope_changes, 75))[0]
print(f"\nSignificant slope changes occur at x ≈: {x_mid[significant_changes]}")

# Visualize derivatives
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Rate of Change and Curvature Analysis',
             fontsize=14, fontweight='bold')

# First derivative
axes[0].plot(x_mid, slopes, 'b-', linewidth=2, marker='o', markersize=4)
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0].set_xlabel('x', fontsize=11)
axes[0].set_ylabel('dy/dx (Slope)', fontsize=11)
axes[0].set_title('First Derivative: Rate of Change', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Add annotations for key features
max_slope_idx = np.argmax(slopes)
min_slope_idx = np.argmin(slopes)
axes[0].plot(x_mid[max_slope_idx], slopes[max_slope_idx], 'ro', markersize=10,
            label=f'Max slope: {slopes[max_slope_idx]:.4f}')
axes[0].plot(x_mid[min_slope_idx], slopes[min_slope_idx], 'go', markersize=10,
            label=f'Min slope: {slopes[min_slope_idx]:.4f}')
axes[0].legend()

# Second derivative
axes[1].plot(x_mid2, curvature, 'r-', linewidth=2, marker='o', markersize=4)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1].set_xlabel('x', fontsize=11)
axes[1].set_ylabel('d²y/dx² (Curvature)', fontsize=11)
axes[1].set_title('Second Derivative: Curvature', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/09_derivative_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: 09_derivative_analysis.png")

print("\n" + "="*60)
print("Smoothing analysis complete.")
print("="*60)
