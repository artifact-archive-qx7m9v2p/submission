"""
Outlier and Influence Analysis
==============================
Identify potential outliers and influential points
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
df = df.reset_index(drop=True)
df['obs_index'] = range(len(df))  # Add index for tracking

x = df['x'].values
y = df['Y'].values

# Fit linear model
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Calculate residuals and statistics
residuals = y - p(x)
std_residuals = residuals / np.std(residuals)

# Calculate leverage (hat values)
X = np.column_stack([np.ones(len(x)), x])
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(H)

# Calculate Cook's distance
n = len(x)
p_params = 2  # number of parameters
mse = np.sum(residuals**2) / (n - p_params)
cooks_d = (residuals**2 / (p_params * mse)) * (leverage / (1 - leverage)**2)

# Create comprehensive diagnostic plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Outlier and Influence Diagnostics', fontsize=16, fontweight='bold')

# 1. Standardized residuals vs fitted
ax = axes[0, 0]
fitted = p(x)
ax.scatter(fitted, std_residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.7, label='±2 SD')
ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(3, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±3 SD')
ax.axhline(-3, color='red', linestyle=':', linewidth=1, alpha=0.5)

# Label potential outliers
outliers_2sd = np.abs(std_residuals) > 2
for i in range(len(fitted)):
    if outliers_2sd[i]:
        ax.annotate(f'{i}', (fitted[i], std_residuals[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Fitted Values', fontsize=12)
ax.set_ylabel('Standardized Residuals', fontsize=12)
ax.set_title('Standardized Residuals vs Fitted', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Leverage plot
ax = axes[0, 1]
ax.scatter(range(len(leverage)), leverage, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
mean_leverage = p_params / n
ax.axhline(mean_leverage, color='blue', linestyle='--', linewidth=1, label=f'Mean leverage: {mean_leverage:.3f}')
ax.axhline(2 * mean_leverage, color='red', linestyle='--', linewidth=1, label=f'2× mean: {2*mean_leverage:.3f}')
ax.axhline(3 * mean_leverage, color='red', linestyle=':', linewidth=1, label=f'3× mean: {3*mean_leverage:.3f}')

# Label high leverage points
high_leverage = leverage > 2 * mean_leverage
for i in range(len(leverage)):
    if high_leverage[i]:
        ax.annotate(f'{i}\n(x={x[i]:.1f})', (i, leverage[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel('Leverage (Hat Value)', fontsize=12)
ax.set_title('Leverage Plot', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Cook's distance
ax = axes[0, 2]
ax.bar(range(len(cooks_d)), cooks_d, alpha=0.7, edgecolor='black', linewidth=0.5)
threshold = 4 / n  # Common threshold
ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')

# Label influential points
influential = cooks_d > threshold
for i in range(len(cooks_d)):
    if influential[i]:
        ax.annotate(f'{i}', (i, cooks_d[i]), fontsize=8, xytext=(0, 5), textcoords='offset points', ha='center')

ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel("Cook's Distance", fontsize=12)
ax.set_title("Cook's Distance (Influence)", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Residuals vs leverage (identifying influential outliers)
ax = axes[1, 0]
ax.scatter(leverage, std_residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(2 * mean_leverage, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Label points that are both high leverage and outliers
problematic = (leverage > 2 * mean_leverage) & (np.abs(std_residuals) > 2)
for i in range(len(leverage)):
    if problematic[i]:
        ax.annotate(f'{i}', (leverage[i], std_residuals[i]), fontsize=8, xytext=(5, 5), textcoords='offset points', color='red')

ax.set_xlabel('Leverage', fontsize=12)
ax.set_ylabel('Standardized Residuals', fontsize=12)
ax.set_title('Residuals vs Leverage', fontsize=12)
ax.grid(True, alpha=0.3)

# 5. DFFITS (another influence measure)
dffits = std_residuals * np.sqrt(leverage / (1 - leverage))
ax = axes[1, 1]
ax.bar(range(len(dffits)), np.abs(dffits), alpha=0.7, edgecolor='black', linewidth=0.5)
dffits_threshold = 2 * np.sqrt(p_params / n)
ax.axhline(dffits_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {dffits_threshold:.3f}')

influential_dffits = np.abs(dffits) > dffits_threshold
for i in range(len(dffits)):
    if influential_dffits[i]:
        ax.annotate(f'{i}', (i, np.abs(dffits[i])), fontsize=8, xytext=(0, 5), textcoords='offset points', ha='center')

ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel('|DFFITS|', fontsize=12)
ax.set_title('DFFITS (Influence on Fitted Values)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Distance from regression line
ax = axes[1, 2]
scatter = ax.scatter(x, y, alpha=0.6, s=100, c=np.abs(std_residuals), cmap='YlOrRd',
           edgecolors='black', linewidth=0.5, vmin=0, vmax=3)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, p(x_line), 'b-', linewidth=2, label='Linear fit')

# Label outliers
for i in range(len(x)):
    if np.abs(std_residuals[i]) > 2:
        ax.annotate(f'{i}', (x[i], y[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Scatter with Residual Magnitude', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('|Standardized Residual|', fontsize=10)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/09_outlier_influence.png', dpi=300, bbox_inches='tight')
plt.close()

# Print diagnostic summary
print("=" * 80)
print("OUTLIER AND INFLUENCE DIAGNOSTICS")
print("=" * 80)

print("\n1. STANDARDIZED RESIDUALS (Outliers)")
print("-" * 80)
outliers_3sd = np.abs(std_residuals) > 3
print(f"Points beyond ±2 SD: {np.sum(outliers_2sd)} ({100*np.sum(outliers_2sd)/n:.1f}%)")
print(f"Points beyond ±3 SD: {np.sum(outliers_3sd)} ({100*np.sum(outliers_3sd)/n:.1f}%)")

if np.sum(outliers_2sd) > 0:
    print("\nOutliers (|std residual| > 2):")
    outlier_indices = np.where(outliers_2sd)[0]
    for idx in outlier_indices:
        print(f"  Index {idx}: x={x[idx]:.2f}, Y={y[idx]:.2f}, std_res={std_residuals[idx]:.3f}, residual={residuals[idx]:.4f}")

print("\n2. HIGH LEVERAGE POINTS")
print("-" * 80)
print(f"Mean leverage: {mean_leverage:.4f}")
print(f"2× mean leverage threshold: {2*mean_leverage:.4f}")
print(f"3× mean leverage threshold: {3*mean_leverage:.4f}")

high_lev_2x = leverage > 2 * mean_leverage
high_lev_3x = leverage > 3 * mean_leverage
print(f"\nPoints with leverage > 2× mean: {np.sum(high_lev_2x)}")
print(f"Points with leverage > 3× mean: {np.sum(high_lev_3x)}")

if np.sum(high_lev_2x) > 0:
    print("\nHigh leverage points (> 2× mean):")
    lev_indices = np.where(high_lev_2x)[0]
    for idx in lev_indices:
        print(f"  Index {idx}: x={x[idx]:.2f}, Y={y[idx]:.2f}, leverage={leverage[idx]:.4f}")

print("\n3. INFLUENTIAL POINTS (Cook's Distance)")
print("-" * 80)
print(f"Cook's D threshold (4/n): {threshold:.4f}")
print(f"Points exceeding threshold: {np.sum(influential)}")

if np.sum(influential) > 0:
    print("\nInfluential points:")
    infl_indices = np.where(influential)[0]
    for idx in infl_indices:
        print(f"  Index {idx}: x={x[idx]:.2f}, Y={y[idx]:.2f}, Cook's D={cooks_d[idx]:.4f}, std_res={std_residuals[idx]:.3f}, leverage={leverage[idx]:.4f}")

print("\n4. INFLUENTIAL POINTS (DFFITS)")
print("-" * 80)
print(f"DFFITS threshold (2√(p/n)): {dffits_threshold:.4f}")
print(f"Points exceeding threshold: {np.sum(influential_dffits)}")

if np.sum(influential_dffits) > 0:
    print("\nInfluential points (DFFITS):")
    dffits_indices = np.where(influential_dffits)[0]
    for idx in dffits_indices:
        print(f"  Index {idx}: x={x[idx]:.2f}, Y={y[idx]:.2f}, DFFITS={dffits[idx]:.4f}")

print("\n5. PROBLEMATIC POINTS (High leverage AND outliers)")
print("-" * 80)
print(f"Points that are both high leverage (>2× mean) and outliers (|std res| > 2): {np.sum(problematic)}")

if np.sum(problematic) > 0:
    print("\nProblematic points:")
    prob_indices = np.where(problematic)[0]
    for idx in prob_indices:
        print(f"  Index {idx}: x={x[idx]:.2f}, Y={y[idx]:.2f}, std_res={std_residuals[idx]:.3f}, leverage={leverage[idx]:.4f}, Cook's D={cooks_d[idx]:.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Overall assessment:")
if np.sum(problematic) > 0:
    print("- WARNING: Some points are both high leverage and outliers")
    print("- These points may have undue influence on the model")
elif np.sum(influential) > 0:
    print("- Some influential points detected, but they may be valid")
else:
    print("- No major outliers or influential points detected")
    print("- Data appears relatively clean")

# Test for the most extreme point
max_cooks_idx = np.argmax(cooks_d)
print(f"\nMost influential point:")
print(f"  Index: {max_cooks_idx}, x={x[max_cooks_idx]:.2f}, Y={y[max_cooks_idx]:.2f}")
print(f"  Cook's D: {cooks_d[max_cooks_idx]:.4f}")
print(f"  Standardized residual: {std_residuals[max_cooks_idx]:.4f}")
print(f"  Leverage: {leverage[max_cooks_idx]:.4f}")
