"""
Variance Structure Analysis
Analyst 1 - Round 2
Examining heteroscedasticity across x range
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Fit linear model
coeffs = np.polyfit(data['x'], data['Y'], 1)
y_pred = coeffs[0] * data['x'] + coeffs[1]
residuals = data['Y'] - y_pred

print("=" * 70)
print("VARIANCE STRUCTURE ANALYSIS")
print("=" * 70)

# Divide into bins by x
n_bins = 5
data['x_bin'] = pd.qcut(data['x'], q=n_bins, duplicates='drop')

# Calculate variance in each bin
bin_stats = data.groupby('x_bin', observed=True).agg({
    'x': ['mean', 'min', 'max', 'count'],
    'Y': ['mean', 'std', 'var']
}).round(4)

print("\nVariance by x bins:")
print(bin_stats)

# Absolute residuals vs x
abs_resid = np.abs(residuals)
corr_abs_resid_x, p_corr = stats.pearsonr(data['x'], abs_resid)
print(f"\nCorrelation between |residuals| and x: r = {corr_abs_resid_x:.4f}, p = {p_corr:.4f}")
print(f"{'Evidence of heteroscedasticity' if p_corr < 0.05 else 'No clear heteroscedasticity'}")

# Levene's test for equal variances across bins
bin_groups = [group['Y'].values for name, group in data.groupby('x_bin', observed=True)]
if len(bin_groups) >= 2:
    levene_stat, levene_p = stats.levene(*bin_groups)
    print(f"\nLevene's test for equal variances: stat = {levene_stat:.4f}, p = {levene_p:.4f}")
    print(f"{'Variances differ across x range' if levene_p < 0.05 else 'Variances appear homogeneous'}")

# Check for replicates to estimate pure error
x_with_reps = data['x'].value_counts()
x_with_reps = x_with_reps[x_with_reps > 1]

if len(x_with_reps) > 0:
    print(f"\n{len(x_with_reps)} x-values with replicates (useful for pure error estimation)")
    print("\nPure error estimates at replicated x values:")

    pure_errors = []
    for x_val in x_with_reps.index:
        reps = data[data['x'] == x_val]['Y']
        if len(reps) > 1:
            pure_var = reps.var()
            pure_std = reps.std()
            pure_errors.append({
                'x': x_val,
                'n_reps': len(reps),
                'Y_mean': reps.mean(),
                'Y_std': pure_std,
                'Y_var': pure_var
            })

    pure_error_df = pd.DataFrame(pure_errors)
    print(pure_error_df.to_string(index=False))

    # Overall pure error estimate
    # Pooled variance from replicates
    total_ss_pure = 0
    total_df_pure = 0
    for x_val in x_with_reps.index:
        reps = data[data['x'] == x_val]['Y']
        if len(reps) > 1:
            total_ss_pure += np.sum((reps - reps.mean())**2)
            total_df_pure += len(reps) - 1

    if total_df_pure > 0:
        pooled_pure_var = total_ss_pure / total_df_pure
        pooled_pure_std = np.sqrt(pooled_pure_var)
        print(f"\nPooled pure error estimate:")
        print(f"  Standard deviation: {pooled_pure_std:.4f}")
        print(f"  Variance: {pooled_pure_var:.6f}")
        print(f"  (Based on {total_df_pure} degrees of freedom)")

        # Compare to model residual variance
        model_var = residuals.var()
        print(f"\nModel residual variance: {model_var:.6f}")
        print(f"Ratio (model/pure): {model_var / pooled_pure_var:.2f}")
        print(f"  (Ratio >> 1 suggests lack of fit beyond pure error)")

print("\n" + "=" * 70)

# ============================================================
# VISUALIZATION: Variance structure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Absolute residuals vs x
axes[0, 0].scatter(data['x'], abs_resid, alpha=0.6, s=80, color='steelblue',
                   edgecolors='black', linewidth=0.5)

# Fit line to absolute residuals
z_abs = np.polyfit(data['x'], abs_resid, 1)
p_abs = np.poly1d(z_abs)
x_line = np.linspace(data['x'].min(), data['x'].max(), 100)
axes[0, 0].plot(x_line, p_abs(x_line), 'r--', linewidth=2, alpha=0.7,
                label=f'Trend: slope={z_abs[0]:.4f}')

axes[0, 0].set_xlabel('x', fontweight='bold')
axes[0, 0].set_ylabel('|Residuals|', fontweight='bold')
axes[0, 0].set_title('Absolute Residuals vs x (Heteroscedasticity Check)',
                     fontweight='bold', fontsize=11)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Variance by x bins
bin_means = []
bin_stds = []
bin_centers = []
for name, group in data.groupby('x_bin', observed=True):
    bin_centers.append(group['x'].mean())
    bin_means.append(group['Y'].mean())
    bin_stds.append(group['Y'].std())

axes[0, 1].errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', markersize=10,
                    capsize=8, capthick=2, linewidth=2, color='steelblue',
                    ecolor='coral', elinewidth=2, markeredgecolor='black', markeredgewidth=0.5)

axes[0, 1].set_xlabel('x (bin center)', fontweight='bold')
axes[0, 1].set_ylabel('Y (mean ± std)', fontweight='bold')
axes[0, 1].set_title('Mean and Variability by x Bins', fontweight='bold', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Squared residuals vs x
squared_resid = residuals**2
axes[1, 0].scatter(data['x'], squared_resid, alpha=0.6, s=80, color='steelblue',
                   edgecolors='black', linewidth=0.5)

# Fit line
z_sq = np.polyfit(data['x'], squared_resid, 1)
p_sq = np.poly1d(z_sq)
axes[1, 0].plot(x_line, p_sq(x_line), 'r--', linewidth=2, alpha=0.7,
                label=f'Trend: slope={z_sq[0]:.5f}')

axes[1, 0].set_xlabel('x', fontweight='bold')
axes[1, 0].set_ylabel('Squared Residuals', fontweight='bold')
axes[1, 0].set_title('Squared Residuals vs x', fontweight='bold', fontsize=11)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Within-replicate variability
if len(x_with_reps) > 0:
    # Show replicates
    for x_val in x_with_reps.index:
        reps = data[data['x'] == x_val]['Y']
        y_vals = reps.values
        x_vals = [x_val] * len(y_vals)
        axes[1, 1].scatter(x_vals, y_vals, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        # Draw line through replicates
        axes[1, 1].plot([x_val, x_val], [y_vals.min(), y_vals.max()],
                        'k-', linewidth=2, alpha=0.4)
        # Show mean
        axes[1, 1].plot(x_val, y_vals.mean(), 'r*', markersize=15, markeredgecolor='black',
                        markeredgewidth=0.5)

    # Add overall trend
    z = np.polyfit(data['x'], data['Y'], 1)
    axes[1, 1].plot(x_line, z[0] * x_line + z[1], 'b--', linewidth=2, alpha=0.5,
                    label='Overall trend')

    axes[1, 1].set_xlabel('x', fontweight='bold')
    axes[1, 1].set_ylabel('Y', fontweight='bold')
    axes[1, 1].set_title('Replicate Values at Same x (red stars = means)',
                         fontweight='bold', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'No replicates available', ha='center', va='center',
                    transform=axes[1, 1].transAxes, fontsize=14)
    axes[1, 1].set_title('Replicate Analysis', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/07_variance_structure.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Created: 07_variance_structure.png")
