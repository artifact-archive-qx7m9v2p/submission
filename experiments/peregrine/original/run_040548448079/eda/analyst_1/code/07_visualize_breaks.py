"""
Visualization: Structural Breaks and Regime Changes
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_1.csv')
df = df.sort_values('year').reset_index(drop=True)

X = df['year'].values
y = df['C'].values
n = len(y)

with open('/workspace/eda/analyst_1/code/structural_breaks.pkl', 'rb') as f:
    break_results = pickle.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# PLOT 7: Structural Break Visualization
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Panel 1: Optimal two-regime model
ax = axes[0, 0]
optimal_bp = break_results['optimal_break']['breakpoint']
params1 = break_results['optimal_break']['params1']
params2 = break_results['optimal_break']['params2']

# Plot data
ax.scatter(X[:optimal_bp], y[:optimal_bp], s=80, alpha=0.7, color='blue',
           label='Regime 1', edgecolors='white', linewidths=1.5, zorder=5)
ax.scatter(X[optimal_bp:], y[optimal_bp:], s=80, alpha=0.7, color='red',
           label='Regime 2', edgecolors='white', linewidths=1.5, zorder=5)

# Plot fitted lines
X1_fit = np.column_stack([np.ones(optimal_bp), X[:optimal_bp]])
X2_fit = np.column_stack([np.ones(n - optimal_bp), X[optimal_bp:]])
y1_fit = X1_fit @ params1
y2_fit = X2_fit @ params2

ax.plot(X[:optimal_bp], y1_fit, linewidth=3, color='darkblue', label=f'Regime 1 fit (slope={params1[1]:.2f})')
ax.plot(X[optimal_bp:], y2_fit, linewidth=3, color='darkred', label=f'Regime 2 fit (slope={params2[1]:.2f})')

# Mark breakpoint
ax.axvline(x=X[optimal_bp], color='green', linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'Break at obs {optimal_bp}\n(year={X[optimal_bp]:.3f})')

ax.set_xlabel('Standardized Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Count (C)', fontsize=11, fontweight='bold')
ax.set_title('Optimal Two-Regime Model\n79.9% improvement in fit', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: SSE across all breakpoints
ax = axes[0, 1]
breakpoints = break_results['optimal_break']['breakpoints']
sse_values = break_results['optimal_break']['sse']

ax.plot(breakpoints, sse_values, linewidth=2.5, color='steelblue')
ax.axvline(x=optimal_bp, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_bp}')
ax.axhline(y=min(sse_values), color='red', linestyle=':', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Potential Breakpoint (Observation #)', fontsize=11, fontweight='bold')
ax.set_ylabel('Sum of Squared Errors', fontsize=11, fontweight='bold')
ax.set_title('Breakpoint Search: SSE Minimization', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: CUSUM test
ax = axes[1, 0]
cusum = break_results['cusum']['values']
cusum_critical = break_results['cusum']['critical']
time_index = np.arange(len(cusum)) + 3  # Start from observation 3

ax.plot(time_index, cusum, linewidth=2.5, color='darkgreen', label='CUSUM')
ax.axhline(y=cusum_critical, color='red', linestyle='--', linewidth=2, label='Critical value')
ax.axhline(y=-cusum_critical, color='red', linestyle='--', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Shade violation area
ax.fill_between(time_index, cusum_critical, np.max(cusum) * 1.1,
                alpha=0.2, color='red', label='Rejection region')
ax.fill_between(time_index, -cusum_critical, np.min(cusum) * 1.1,
                alpha=0.2, color='red')

ax.set_xlabel('Observation Number', fontsize=11, fontweight='bold')
ax.set_ylabel('CUSUM Statistic', fontsize=11, fontweight='bold')
ax.set_title('CUSUM Test for Stability\n(Rejects stability - increasing trend)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: Rolling slopes
ax = axes[1, 1]
rolling_slopes = break_results['rolling']['slopes']
window = break_results['rolling']['window']
window_centers = np.arange(len(rolling_slopes)) + window // 2

ax.plot(window_centers, rolling_slopes, linewidth=2.5, color='purple', marker='o', markersize=4)
ax.axhline(y=np.mean(rolling_slopes), color='orange', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(rolling_slopes):.2f}')

# Shade different regions
first_third_end = len(rolling_slopes) // 3
last_third_start = 2 * len(rolling_slopes) // 3

ax.axvspan(0, window_centers[first_third_end], alpha=0.2, color='blue',
           label=f'First 1/3: {np.mean(rolling_slopes[:first_third_end]):.2f}')
ax.axvspan(window_centers[last_third_start], window_centers[-1], alpha=0.2, color='red',
           label=f'Last 1/3: {np.mean(rolling_slopes[last_third_start:]):.2f}')

ax.set_xlabel('Window Center (Observation #)', fontsize=11, fontweight='bold')
ax.set_ylabel(f'Slope ({window}-obs window)', fontsize=11, fontweight='bold')
ax.set_title('Rolling Window Slopes\n(Acceleration pattern evident)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Structural Break Analysis: Multiple Tests', fontsize=14, fontweight='bold', y=0.997)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/07_structural_breaks.png', dpi=300, bbox_inches='tight')
print("Saved: 07_structural_breaks.png")
plt.close()

# ============================================================================
# PLOT 8: Comparative view - Single vs Two-Regime
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Single model
ax = axes[0]
X_full = np.column_stack([np.ones(n), X])
params_single = np.linalg.lstsq(X_full, y, rcond=None)[0]
y_pred_single = X_full @ params_single

ax.scatter(X, y, s=80, alpha=0.6, color='black', edgecolors='white', linewidths=1.5, zorder=5)
ax.plot(X, y_pred_single, linewidth=3, color='blue', label=f'Single regime (slope={params_single[1]:.2f})')

residuals_single = y - y_pred_single
for i in range(n):
    ax.plot([X[i], X[i]], [y[i], y_pred_single[i]], color='red', alpha=0.3, linewidth=1)

sse_single = np.sum(residuals_single**2)
r2_single = 1 - sse_single / np.sum((y - np.mean(y))**2)

ax.set_xlabel('Standardized Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Count (C)', fontsize=11, fontweight='bold')
ax.set_title(f'Single-Regime Model\nR² = {r2_single:.4f}, SSE = {sse_single:.0f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Two-regime model
ax = axes[1]
ax.scatter(X[:optimal_bp], y[:optimal_bp], s=80, alpha=0.7, color='blue',
           label='Regime 1', edgecolors='white', linewidths=1.5, zorder=5)
ax.scatter(X[optimal_bp:], y[optimal_bp:], s=80, alpha=0.7, color='red',
           label='Regime 2', edgecolors='white', linewidths=1.5, zorder=5)

ax.plot(X[:optimal_bp], y1_fit, linewidth=3, color='darkblue')
ax.plot(X[optimal_bp:], y2_fit, linewidth=3, color='darkred')

# Residuals
residuals1 = y[:optimal_bp] - y1_fit
residuals2 = y[optimal_bp:] - y2_fit
for i in range(optimal_bp):
    ax.plot([X[i], X[i]], [y[i], y1_fit[i]], color='lightblue', alpha=0.4, linewidth=1)
for i in range(optimal_bp, n):
    ax.plot([X[i], X[i]], [y[i], y2_fit[i - optimal_bp]], color='lightcoral', alpha=0.4, linewidth=1)

ax.axvline(x=X[optimal_bp], color='green', linestyle='--', linewidth=2.5, alpha=0.7, label='Break point')

sse_two = np.sum(residuals1**2) + np.sum(residuals2**2)
r2_two = 1 - sse_two / np.sum((y - np.mean(y))**2)

ax.set_xlabel('Standardized Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Count (C)', fontsize=11, fontweight='bold')
ax.set_title(f'Two-Regime Model\nR² = {r2_two:.4f}, SSE = {sse_two:.0f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Model Comparison: Single vs Two-Regime', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/08_regime_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: 08_regime_comparison.png")
plt.close()

print("\nStructural break visualizations complete.")
