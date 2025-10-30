"""
Visualizing x-Y Relationship
Analyst 1 - Round 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================
# FIGURE 1: Comprehensive scatter plot with multiple smoothers
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Raw scatter plot
ax.scatter(data['x'], data['Y'], alpha=0.6, s=80, color='steelblue',
           edgecolors='black', linewidth=0.5, label='Observed data')

# Sort for smoothing
x_sorted = np.sort(data['x'].values)
y_sorted = data['Y'].values[np.argsort(data['x'].values)]

# Linear fit
z = np.polyfit(data['x'], data['Y'], 1)
p = np.poly1d(z)
ax.plot(data['x'].sort_values(), p(data['x'].sort_values()),
        'r--', linewidth=2, alpha=0.7, label=f'Linear fit (slope={z[0]:.4f})')

# Polynomial fit (degree 2)
z2 = np.polyfit(data['x'], data['Y'], 2)
p2 = np.poly1d(z2)
ax.plot(data['x'].sort_values(), p2(data['x'].sort_values()),
        'g--', linewidth=2, alpha=0.7, label='Quadratic fit')

# LOWESS smoother
from scipy.signal import savgol_filter
if len(data) >= 11:
    window = 11 if len(data) >= 11 else len(data) if len(data) % 2 == 1 else len(data) - 1
    smooth_y = savgol_filter(y_sorted, window, 3)
    ax.plot(x_sorted, smooth_y, 'purple', linewidth=2.5,
            label='Savitzky-Golay smoother', alpha=0.8)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title('Relationship between x and Y: Multiple Smoothing Methods',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/01_scatter_with_smoothers.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Created: 01_scatter_with_smoothers.png")

# ============================================================
# FIGURE 2: Distribution plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# X distribution - histogram
axes[0, 0].hist(data['x'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(data['x'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={data["x"].mean():.2f}')
axes[0, 0].axvline(data['x'].median(), color='green', linestyle='--',
                   linewidth=2, label=f'Median={data["x"].median():.2f}')
axes[0, 0].set_xlabel('x', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Distribution of x (predictor)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# X distribution - Q-Q plot
stats.probplot(data['x'], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot: x vs Normal Distribution', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Y distribution - histogram
axes[1, 0].hist(data['Y'], bins=15, edgecolor='black', alpha=0.7, color='lightcoral')
axes[1, 0].axvline(data['Y'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={data["Y"].mean():.3f}')
axes[1, 0].axvline(data['Y'].median(), color='green', linestyle='--',
                   linewidth=2, label=f'Median={data["Y"].median():.3f}')
axes[1, 0].set_xlabel('Y', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Distribution of Y (response)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Y distribution - Q-Q plot
stats.probplot(data['Y'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Y vs Normal Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/02_distributions.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Created: 02_distributions.png")

# ============================================================
# FIGURE 3: Relationship by segments (early, mid, late x values)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Define segments
x_33_pct = data['x'].quantile(0.33)
x_66_pct = data['x'].quantile(0.66)

segment1 = data[data['x'] <= x_33_pct]
segment2 = data[(data['x'] > x_33_pct) & (data['x'] <= x_66_pct)]
segment3 = data[data['x'] > x_66_pct]

# Plot segments with different colors
ax.scatter(segment1['x'], segment1['Y'], s=100, alpha=0.7,
           color='green', edgecolors='black', linewidth=0.5,
           label=f'Low x (n={len(segment1)}, x≤{x_33_pct:.1f})')
ax.scatter(segment2['x'], segment2['Y'], s=100, alpha=0.7,
           color='orange', edgecolors='black', linewidth=0.5,
           label=f'Mid x (n={len(segment2)}, {x_33_pct:.1f}<x≤{x_66_pct:.1f})')
ax.scatter(segment3['x'], segment3['Y'], s=100, alpha=0.7,
           color='red', edgecolors='black', linewidth=0.5,
           label=f'High x (n={len(segment3)}, x>{x_66_pct:.1f})')

# Overall linear fit
z = np.polyfit(data['x'], data['Y'], 1)
p = np.poly1d(z)
ax.plot(data['x'].sort_values(), p(data['x'].sort_values()),
        'k--', linewidth=2, alpha=0.5, label='Overall linear fit')

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title('Relationship Segmented by x Range', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Add vertical lines for segment boundaries
ax.axvline(x_33_pct, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x_66_pct, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/03_segmented_relationship.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Created: 03_segmented_relationship.png")

# Print segment statistics
print("\n" + "=" * 60)
print("SEGMENT ANALYSIS")
print("=" * 60)
print(f"\nSegment 1 (Low x): n={len(segment1)}")
print(f"  x range: [{segment1['x'].min():.2f}, {segment1['x'].max():.2f}]")
print(f"  Y mean: {segment1['Y'].mean():.3f}, std: {segment1['Y'].std():.3f}")

print(f"\nSegment 2 (Mid x): n={len(segment2)}")
print(f"  x range: [{segment2['x'].min():.2f}, {segment2['x'].max():.2f}]")
print(f"  Y mean: {segment2['Y'].mean():.3f}, std: {segment2['Y'].std():.3f}")

print(f"\nSegment 3 (High x): n={len(segment3)}")
print(f"  x range: [{segment3['x'].min():.2f}, {segment3['x'].max():.2f}]")
print(f"  Y mean: {segment3['Y'].mean():.3f}, std: {segment3['Y'].std():.3f}")

print("\n" + "=" * 60)
