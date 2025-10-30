"""
Outlier and Influential Point Analysis
Identify observations that don't fit the general pattern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup
sns.set_style("whitegrid")
output_dir = Path('/workspace/eda/analyst_2/visualizations')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
year = data['year'].values
C = data['C'].values

print("=" * 80)
print("OUTLIER ANALYSIS")
print("=" * 80)

# 1. Z-scores
z_scores = (C - C.mean()) / C.std(ddof=1)
print(f"\n1. Z-SCORE ANALYSIS")
print(f"   Observations with |z| > 2: {np.sum(np.abs(z_scores) > 2)}")
print(f"   Observations with |z| > 3: {np.sum(np.abs(z_scores) > 3)}")

outliers_z = np.where(np.abs(z_scores) > 2)[0]
if len(outliers_z) > 0:
    print(f"\n   Outliers (|z| > 2):")
    for idx in outliers_z:
        print(f"     Obs {idx+1}: C={C[idx]}, year={year[idx]:.3f}, z={z_scores[idx]:.3f}")

# 2. IQR method
q1 = np.percentile(C, 25)
q3 = np.percentile(C, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"\n2. IQR METHOD")
print(f"   Q1: {q1:.1f}")
print(f"   Q3: {q3:.1f}")
print(f"   IQR: {iqr:.1f}")
print(f"   Lower bound: {lower_bound:.1f}")
print(f"   Upper bound: {upper_bound:.1f}")

outliers_iqr = np.where((C < lower_bound) | (C > upper_bound))[0]
print(f"   Outliers: {len(outliers_iqr)}")
if len(outliers_iqr) > 0:
    print(f"\n   Outlier observations:")
    for idx in outliers_iqr:
        print(f"     Obs {idx+1}: C={C[idx]}, year={year[idx]:.3f}")

# 3. Modified Z-score (using MAD - robust to outliers)
median = np.median(C)
mad = np.median(np.abs(C - median))
modified_z_scores = 0.6745 * (C - median) / mad

print(f"\n3. MODIFIED Z-SCORE (MAD-based)")
print(f"   Median: {median:.1f}")
print(f"   MAD: {mad:.1f}")
print(f"   Observations with |modified z| > 3.5: {np.sum(np.abs(modified_z_scores) > 3.5)}")

outliers_mad = np.where(np.abs(modified_z_scores) > 3.5)[0]
if len(outliers_mad) > 0:
    print(f"\n   Outliers (|modified z| > 3.5):")
    for idx in outliers_mad:
        print(f"     Obs {idx+1}: C={C[idx]}, year={year[idx]:.3f}, mod_z={modified_z_scores[idx]:.3f}")

# 4. Fit simple trend and look at residuals
# Linear trend
from numpy.polynomial import Polynomial
p = Polynomial.fit(year, C, 1)
fitted_values = p(year)
residuals = C - fitted_values
std_residuals = residuals / residuals.std(ddof=1)

print(f"\n4. TREND-ADJUSTED OUTLIERS")
print(f"   Linear trend: C = {p.coef[0]:.2f} + {p.coef[1]:.2f} * year")
print(f"   Residual std: {residuals.std(ddof=1):.2f}")
print(f"   Observations with |std residual| > 2: {np.sum(np.abs(std_residuals) > 2)}")
print(f"   Observations with |std residual| > 3: {np.sum(np.abs(std_residuals) > 3)}")

outliers_trend = np.where(np.abs(std_residuals) > 2)[0]
if len(outliers_trend) > 0:
    print(f"\n   Outliers (|std resid| > 2):")
    for idx in outliers_trend:
        print(f"     Obs {idx+1}: C={C[idx]}, fitted={fitted_values[idx]:.1f}, " +
              f"resid={residuals[idx]:.1f}, std_resid={std_residuals[idx]:.3f}")

# 5. Cook's Distance (influence on regression)
# For simple linear regression
n = len(C)
p_params = 2  # intercept + slope
X = np.column_stack([np.ones(n), year])
H = X @ np.linalg.inv(X.T @ X) @ X.T  # Hat matrix
h_ii = np.diag(H)
mse = np.sum(residuals**2) / (n - p_params)
cooks_d = (std_residuals**2 / p_params) * (h_ii / (1 - h_ii))

print(f"\n5. COOK'S DISTANCE (Influence)")
print(f"   Threshold (4/n): {4/n:.4f}")
influential = np.where(cooks_d > 4/n)[0]
print(f"   Influential points (D > 4/n): {len(influential)}")

if len(influential) > 0:
    print(f"\n   Most influential observations:")
    # Sort by Cook's distance
    top_influential = influential[np.argsort(cooks_d[influential])[::-1]]
    for idx in top_influential[:10]:  # Show top 10
        print(f"     Obs {idx+1}: C={C[idx]}, Cook's D={cooks_d[idx]:.4f}")

# Create comprehensive outlier visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Box plot with outliers marked
ax = axes[0, 0]
bp = ax.boxplot([C], widths=0.5, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 showfliers=True,
                 flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.6))
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Box Plot with Outliers', fontsize=12, fontweight='bold')
ax.set_xticks([1])
ax.set_xticklabels(['Count Data'])
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
text = f'IQR outliers: {len(outliers_iqr)}\nLower: {lower_bound:.1f}\nUpper: {upper_bound:.1f}'
ax.text(1.3, C.max() * 0.9, text, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# 2. Z-scores over time
ax = axes[0, 1]
ax.scatter(year, z_scores, alpha=0.6, s=50, color='steelblue')
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='±2 SD')
ax.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±3 SD')
ax.axhline(-3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

# Mark outliers
if len(outliers_z) > 0:
    ax.scatter(year[outliers_z], z_scores[outliers_z], color='red', s=100,
               marker='o', edgecolors='black', linewidths=2, zorder=5, label='Outliers')

ax.set_xlabel('Standardized Year', fontsize=11)
ax.set_ylabel('Z-Score', fontsize=11)
ax.set_title('Z-Score Analysis', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Residuals from linear trend
ax = axes[0, 2]
ax.scatter(fitted_values, std_residuals, alpha=0.6, s=50, color='steelblue')
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='±2 SD')
ax.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Mark outliers
if len(outliers_trend) > 0:
    ax.scatter(fitted_values[outliers_trend], std_residuals[outliers_trend],
               color='red', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5,
               label='Outliers')

ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Standardized Residuals', fontsize=11)
ax.set_title('Residuals vs Fitted (Linear Trend)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4. Cook's Distance
ax = axes[1, 0]
ax.stem(np.arange(n), cooks_d, basefmt=' ', linefmt='steelblue', markerfmt='o')
ax.axhline(4/n, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Threshold (4/n={4/n:.3f})')

# Mark influential points
if len(influential) > 0:
    ax.scatter(influential, cooks_d[influential], color='red', s=100,
               marker='o', edgecolors='black', linewidths=2, zorder=5, label='Influential')

ax.set_xlabel('Observation Index', fontsize=11)
ax.set_ylabel("Cook's Distance", fontsize=11)
ax.set_title('Influence on Linear Regression', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 5. Leverage vs Residuals
ax = axes[1, 1]
ax.scatter(h_ii, std_residuals, alpha=0.6, s=50, color='steelblue')
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(2*p_params/n, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'High leverage (2p/n={2*p_params/n:.3f})')

# Mark high leverage and outliers
high_leverage = h_ii > 2*p_params/n
high_resid = np.abs(std_residuals) > 2
problematic = high_leverage & high_resid
if np.any(problematic):
    idx_prob = np.where(problematic)[0]
    ax.scatter(h_ii[idx_prob], std_residuals[idx_prob], color='red', s=150,
               marker='*', edgecolors='black', linewidths=2, zorder=5,
               label='High leverage + outlier')

ax.set_xlabel('Leverage (h_ii)', fontsize=11)
ax.set_ylabel('Standardized Residuals', fontsize=11)
ax.set_title('Leverage vs Residuals', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6. Time series with outliers marked
ax = axes[1, 2]
ax.plot(year, C, 'o-', color='steelblue', alpha=0.4, markersize=6)
ax.plot(year, fitted_values, 'g--', linewidth=2, alpha=0.7, label='Linear trend')

# Mark different types of outliers with different colors
if len(outliers_z) > 0:
    ax.scatter(year[outliers_z], C[outliers_z], color='red', s=120,
               marker='o', edgecolors='black', linewidths=2, zorder=5,
               label=f'Z-score outliers (n={len(outliers_z)})')

if len(influential) > 0:
    # Only mark influential that aren't already marked as z-score outliers
    influential_unique = np.setdiff1d(influential, outliers_z)
    if len(influential_unique) > 0:
        ax.scatter(year[influential_unique], C[influential_unique], color='orange', s=120,
                   marker='s', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Influential (n={len(influential_unique)})')

ax.set_xlabel('Standardized Year', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Time Series with Outliers Marked', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: outlier_analysis.png")
plt.close()

# Summary table
print("\n" + "=" * 80)
print("OUTLIER SUMMARY")
print("=" * 80)
print(f"Total observations: {n}")
print(f"\nOutlier detection methods:")
print(f"  Z-score (|z| > 2): {len(outliers_z)} observations")
print(f"  IQR method: {len(outliers_iqr)} observations")
print(f"  MAD-based (|mod z| > 3.5): {len(outliers_mad)} observations")
print(f"  Trend-adjusted (|std resid| > 2): {len(outliers_trend)} observations")
print(f"  Influential (Cook's D > 4/n): {len(influential)} observations")

# Find consensus outliers
all_outliers = np.unique(np.concatenate([outliers_z, outliers_iqr, outliers_mad,
                                          outliers_trend, influential]))
print(f"\n  Unique observations flagged: {len(all_outliers)}")
print(f"  Percentage of data: {len(all_outliers)/n*100:.1f}%")

if len(all_outliers) > 0:
    print(f"\n  Consensus outliers (flagged by multiple methods):")
    for idx in all_outliers:
        methods = []
        if idx in outliers_z:
            methods.append('Z-score')
        if idx in outliers_iqr:
            methods.append('IQR')
        if idx in outliers_mad:
            methods.append('MAD')
        if idx in outliers_trend:
            methods.append('Trend')
        if idx in influential:
            methods.append('Influential')
        if len(methods) >= 2:
            print(f"    Obs {idx+1}: C={C[idx]}, year={year[idx]:.3f}, " +
                  f"methods={', '.join(methods)}")
