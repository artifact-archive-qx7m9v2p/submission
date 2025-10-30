"""
Influence and Outlier Analysis
Analyst 1 - Round 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Fit linear model for leverage analysis
X = data['x'].values
Y = data['Y'].values
n = len(data)

# Design matrix for linear regression
X_design = np.column_stack([np.ones(n), X])

# Hat matrix
H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
leverage = np.diag(H)

# Fit model
coeffs = np.polyfit(X, Y, 1)
y_pred = coeffs[0] * X + coeffs[1]
residuals = Y - y_pred
mse = np.sum(residuals**2) / (n - 2)
std_residuals = residuals / np.sqrt(mse * (1 - leverage))

# Cook's distance
cooks_d = (residuals**2 / (2 * mse)) * (leverage / (1 - leverage)**2)

# Add to dataframe
data['leverage'] = leverage
data['std_residuals'] = std_residuals
data['cooks_d'] = cooks_d
data['y_pred'] = y_pred
data['residuals'] = residuals

print("=" * 70)
print("INFLUENCE AND OUTLIER ANALYSIS")
print("=" * 70)

# Leverage threshold
leverage_threshold = 2 * 2 / n  # 2p/n for linear regression
print(f"\nLeverage threshold (2p/n): {leverage_threshold:.4f}")
high_leverage = data[data['leverage'] > leverage_threshold]
print(f"High leverage points: {len(high_leverage)}")
if len(high_leverage) > 0:
    print("\nHigh leverage observations:")
    print(high_leverage[['x', 'Y', 'leverage', 'std_residuals', 'cooks_d']].to_string(index=False))

# Outliers (|std residual| > 2)
outliers = data[np.abs(data['std_residuals']) > 2]
print(f"\nOutliers (|std residual| > 2): {len(outliers)}")
if len(outliers) > 0:
    print("\nOutlier observations:")
    print(outliers[['x', 'Y', 'leverage', 'std_residuals', 'cooks_d']].to_string(index=False))

# Influential points (Cook's D > 4/n)
cooks_threshold = 4 / n
influential = data[data['cooks_d'] > cooks_threshold]
print(f"\nInfluential points (Cook's D > {cooks_threshold:.4f}): {len(influential)}")
if len(influential) > 0:
    print("\nInfluential observations:")
    print(influential[['x', 'Y', 'leverage', 'std_residuals', 'cooks_d']].to_string(index=False))

# Summary statistics
print("\n" + "=" * 70)
print("DIAGNOSTIC STATISTICS SUMMARY")
print("=" * 70)
print(f"Leverage: min={leverage.min():.4f}, max={leverage.max():.4f}, mean={leverage.mean():.4f}")
print(f"Std Residuals: min={std_residuals.min():.4f}, max={std_residuals.max():.4f}")
print(f"Cook's D: min={cooks_d.min():.6f}, max={cooks_d.max():.6f}, mean={cooks_d.mean():.6f}")

# Identify extreme x values
x_q1, x_q3 = data['x'].quantile([0.25, 0.75])
x_iqr = x_q3 - x_q1
x_outlier_low = data[data['x'] < x_q1 - 1.5 * x_iqr]
x_outlier_high = data[data['x'] > x_q3 + 1.5 * x_iqr]

print(f"\nExtreme x values (beyond 1.5*IQR):")
print(f"  Low: {len(x_outlier_low)} observations")
if len(x_outlier_low) > 0:
    print(f"    x values: {x_outlier_low['x'].values}")
print(f"  High: {len(x_outlier_high)} observations")
if len(x_outlier_high) > 0:
    print(f"    x values: {x_outlier_high['x'].values}")

print("\n" + "=" * 70)

# ============================================================
# VISUALIZATION: Influence diagnostics
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Leverage vs Standardized Residuals
axes[0, 0].scatter(data['leverage'], data['std_residuals'],
                   s=100, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
axes[0, 0].axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='|std resid| = 2')
axes[0, 0].axhline(y=-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 0].axvline(x=leverage_threshold, color='orange', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Leverage threshold = {leverage_threshold:.3f}')

# Annotate points of interest
for idx, row in data.iterrows():
    if row['leverage'] > leverage_threshold or np.abs(row['std_residuals']) > 2:
        axes[0, 0].annotate(f"x={row['x']:.1f}", (row['leverage'], row['std_residuals']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

axes[0, 0].set_xlabel('Leverage', fontweight='bold')
axes[0, 0].set_ylabel('Standardized Residuals', fontweight='bold')
axes[0, 0].set_title('Leverage vs Standardized Residuals', fontweight='bold', fontsize=12)
axes[0, 0].legend(loc='best', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Cook's Distance
axes[0, 1].bar(range(len(data)), data['cooks_d'], color='steelblue',
               alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0, 1].axhline(y=cooks_threshold, color='red', linestyle='--',
                   linewidth=2, label=f"Cook's D threshold = {cooks_threshold:.3f}")
axes[0, 1].set_xlabel('Observation Index', fontweight='bold')
axes[0, 1].set_ylabel("Cook's Distance", fontweight='bold')
axes[0, 1].set_title("Cook's Distance for Each Observation", fontweight='bold', fontsize=12)
axes[0, 1].legend(loc='best')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Panel 3: Residuals vs Leverage
axes[1, 0].scatter(data['leverage'], data['residuals'],
                   s=100, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
axes[1, 0].axhline(y=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
axes[1, 0].axvline(x=leverage_threshold, color='orange', linestyle='--',
                   linewidth=1.5, alpha=0.7)

# Add Cook's distance contours
lev_range = np.linspace(leverage.min(), leverage.max(), 100)
for d in [0.5, 1.0]:
    if d <= cooks_d.max() * 2:  # Only show relevant contours
        resid_pos = np.sqrt(d * 2 * mse * (1 - lev_range)**2 / lev_range)
        resid_neg = -resid_pos
        axes[1, 0].plot(lev_range, resid_pos, ':', color='gray', alpha=0.5, linewidth=1)
        axes[1, 0].plot(lev_range, resid_neg, ':', color='gray', alpha=0.5, linewidth=1)

axes[1, 0].set_xlabel('Leverage', fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontweight='bold')
axes[1, 0].set_title('Residuals vs Leverage', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Index plot of key diagnostics
ax4 = axes[1, 1]
ax4_twin = ax4.twinx()

line1 = ax4.plot(range(len(data)), np.abs(data['std_residuals']), 'o-',
                 color='blue', alpha=0.7, linewidth=1, markersize=4, label='|Std Residuals|')
ax4.axhline(y=2, color='blue', linestyle='--', linewidth=1, alpha=0.5)

line2 = ax4_twin.plot(range(len(data)), data['leverage'], 's-',
                      color='orange', alpha=0.7, linewidth=1, markersize=4, label='Leverage')
ax4_twin.axhline(y=leverage_threshold, color='orange', linestyle='--', linewidth=1, alpha=0.5)

ax4.set_xlabel('Observation Index', fontweight='bold')
ax4.set_ylabel('|Standardized Residuals|', fontweight='bold', color='blue')
ax4_twin.set_ylabel('Leverage', fontweight='bold', color='orange')
ax4.set_title('Index Plot: Diagnostics Over Observations', fontweight='bold', fontsize=12)
ax4.tick_params(axis='y', labelcolor='blue')
ax4_twin.tick_params(axis='y', labelcolor='orange')
ax4.grid(True, alpha=0.3)

# Combined legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/06_influence_diagnostics.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Created: 06_influence_diagnostics.png")
