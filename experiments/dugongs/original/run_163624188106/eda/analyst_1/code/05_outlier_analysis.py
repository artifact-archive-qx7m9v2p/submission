"""
Outlier and Influential Points Analysis - Analyst 1
====================================================
Purpose: Identify outliers and influential observations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')
data['index'] = range(len(data))

# Fit logarithmic model (best performer)
log_coef = np.polyfit(np.log(data['x']), data['Y'], 1)
y_pred_log = np.polyval(log_coef, np.log(data['x']))
residuals_log = data['Y'] - y_pred_log

# Standardized residuals
std_residuals = residuals_log / residuals_log.std()

print("="*60)
print("OUTLIER AND INFLUENTIAL POINTS ANALYSIS")
print("="*60)

# 1. Z-score based outliers
print("\n1. STANDARDIZED RESIDUALS (Z-SCORES)")
print("-"*60)
threshold = 2.5
outliers_z = np.abs(std_residuals) > threshold
print(f"Observations with |standardized residual| > {threshold}:")
if outliers_z.any():
    for idx in data[outliers_z].index:
        print(f"  Index {idx}: x={data.loc[idx, 'x']:.1f}, Y={data.loc[idx, 'Y']:.3f}, "
              f"z-score={std_residuals[idx]:.3f}")
else:
    print(f"  None found")

# 2. IQR-based outliers for residuals
print("\n2. IQR-BASED OUTLIERS (RESIDUALS)")
print("-"*60)
q1 = np.percentile(residuals_log, 25)
q3 = np.percentile(residuals_log, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_iqr = (residuals_log < lower_bound) | (residuals_log > upper_bound)
print(f"IQR = {iqr:.4f}")
print(f"Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"\nOutliers detected:")
if outliers_iqr.any():
    for idx in data[outliers_iqr].index:
        print(f"  Index {idx}: x={data.loc[idx, 'x']:.1f}, Y={data.loc[idx, 'Y']:.3f}, "
              f"residual={residuals_log[idx]:.4f}")
else:
    print(f"  None found")

# 3. Leverage analysis
print("\n3. LEVERAGE ANALYSIS")
print("-"*60)

# Design matrix for log model
X = np.column_stack([np.ones(len(data)), np.log(data['x'])])
# Hat matrix
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(H)

# High leverage threshold: 2*p/n or 3*p/n where p is number of parameters
p = X.shape[1]
n = len(data)
leverage_threshold_2 = 2 * p / n
leverage_threshold_3 = 3 * p / n

print(f"Mean leverage: {leverage.mean():.4f} (= p/n = {p}/{n})")
print(f"Leverage threshold (2p/n): {leverage_threshold_2:.4f}")
print(f"Leverage threshold (3p/n): {leverage_threshold_3:.4f}")

high_leverage = leverage > leverage_threshold_2
print(f"\nHigh leverage points (> 2p/n):")
if high_leverage.any():
    for idx in data[high_leverage].index:
        print(f"  Index {idx}: x={data.loc[idx, 'x']:.1f}, Y={data.loc[idx, 'Y']:.3f}, "
              f"leverage={leverage[idx]:.4f}")
else:
    print("  None found")

# 4. Cook's distance
print("\n4. COOK'S DISTANCE")
print("-"*60)

# Calculate Cook's distance
mse = np.sum(residuals_log**2) / (n - p)
cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage))

cooks_threshold = 4 / n
print(f"Cook's distance threshold (4/n): {cooks_threshold:.4f}")

influential = cooks_d > cooks_threshold
print(f"\nInfluential points (Cook's D > 4/n):")
if influential.any():
    for idx in data[influential].index:
        print(f"  Index {idx}: x={data.loc[idx, 'x']:.1f}, Y={data.loc[idx, 'Y']:.3f}, "
              f"Cook's D={cooks_d[idx]:.4f}")
else:
    print("  None found")

# Top 5 by Cook's distance
print(f"\nTop 5 observations by Cook's distance:")
top_cooks = np.argsort(cooks_d)[-5:][::-1]
for rank, idx in enumerate(top_cooks, 1):
    print(f"  {rank}. Index {idx}: x={data.loc[idx, 'x']:.1f}, Y={data.loc[idx, 'Y']:.3f}, "
          f"Cook's D={cooks_d[idx]:.4f}")

# 5. DFFITS
print("\n5. DFFITS")
print("-"*60)

dffits = std_residuals * np.sqrt(leverage / (1 - leverage))
dffits_threshold = 2 * np.sqrt(p / n)
print(f"DFFITS threshold: {dffits_threshold:.4f}")

high_dffits = np.abs(dffits) > dffits_threshold
print(f"\nHigh DFFITS points:")
if high_dffits.any():
    for idx in data[high_dffits].index:
        print(f"  Index {idx}: x={data.loc[idx, 'x']:.1f}, Y={data.loc[idx, 'Y']:.3f}, "
              f"DFFITS={dffits[idx]:.4f}")
else:
    print("  None found")

# Create comprehensive influence plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Outlier and Influence Diagnostics (Logarithmic Model)',
             fontsize=14, fontweight='bold')

# 1. Residuals vs Leverage
axes[0, 0].scatter(leverage, std_residuals, s=80, alpha=0.6, edgecolors='black')
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].axhline(y=2.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].axhline(y=-2.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].axvline(x=leverage_threshold_2, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Annotate high leverage or high residual points
for idx in range(len(data)):
    if leverage[idx] > leverage_threshold_2 or np.abs(std_residuals[idx]) > 2:
        axes[0, 0].annotate(f'{idx}', (leverage[idx], std_residuals[idx]),
                           fontsize=8, alpha=0.7)

axes[0, 0].set_xlabel('Leverage', fontsize=11)
axes[0, 0].set_ylabel('Standardized Residuals', fontsize=11)
axes[0, 0].set_title('Residuals vs Leverage', fontsize=12)
axes[0, 0].grid(alpha=0.3)

# 2. Cook's Distance
axes[0, 1].bar(range(len(data)), cooks_d, alpha=0.7, edgecolor='black')
axes[0, 1].axhline(y=cooks_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {cooks_threshold:.4f}')
axes[0, 1].set_xlabel('Observation Index', fontsize=11)
axes[0, 1].set_ylabel("Cook's Distance", fontsize=11)
axes[0, 1].set_title("Cook's Distance by Observation", fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Leverage plot
axes[1, 0].bar(range(len(data)), leverage, alpha=0.7, edgecolor='black', color='coral')
axes[1, 0].axhline(y=leverage_threshold_2, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold (2p/n): {leverage_threshold_2:.4f}')
axes[1, 0].axhline(y=leverage.mean(), color='green', linestyle='--', linewidth=1,
                   label=f'Mean: {leverage.mean():.4f}')
axes[1, 0].set_xlabel('Observation Index', fontsize=11)
axes[1, 0].set_ylabel('Leverage', fontsize=11)
axes[1, 0].set_title('Leverage by Observation', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. DFFITS
axes[1, 1].bar(range(len(data)), np.abs(dffits), alpha=0.7, edgecolor='black',
               color='lightgreen')
axes[1, 1].axhline(y=dffits_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {dffits_threshold:.4f}')
axes[1, 1].set_xlabel('Observation Index', fontsize=11)
axes[1, 1].set_ylabel('|DFFITS|', fontsize=11)
axes[1, 1].set_title('DFFITS by Observation', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/06_influence_diagnostics.png',
            dpi=300, bbox_inches='tight')
print("\nSaved: 06_influence_diagnostics.png")
plt.close()

# Create bubble plot showing all diagnostics together
fig, ax = plt.subplots(figsize=(12, 8))

# Size by Cook's distance, color by standardized residual
scatter = ax.scatter(data['x'], data['Y'],
                    s=cooks_d * 5000,  # Scale for visibility
                    c=std_residuals,
                    cmap='RdYlBu_r',
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=1.5,
                    vmin=-3, vmax=3)

# Add fitted line
x_line = np.linspace(data['x'].min(), data['x'].max(), 100)
y_line = log_coef[0] * np.log(x_line) + log_coef[1]
ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.7, label='Log model fit')

# Annotate influential points
for idx in data[influential].index:
    ax.annotate(f'Obs {idx}',
               (data.loc[idx, 'x'], data.loc[idx, 'Y']),
               xytext=(10, 10), textcoords='offset points',
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title("Influence Diagnostic Summary\n(Bubble size = Cook's D, Color = Std. Residual)",
            fontsize=13, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Standardized Residual', fontsize=11)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/07_influence_bubble_plot.png',
            dpi=300, bbox_inches='tight')
print("Saved: 07_influence_bubble_plot.png")
plt.close()

print("\n" + "="*60)
print("OUTLIER ANALYSIS COMPLETE")
print("="*60)
