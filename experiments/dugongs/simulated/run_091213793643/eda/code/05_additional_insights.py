"""
Script 5: Additional Insights and Variance Analysis
Deeper dive into patterns, variance structure, and model implications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Set up paths
DATA_PATH = Path("/workspace/data/data.csv")
VIZ_DIR = Path("/workspace/eda/visualizations")

# Load data
df = pd.read_csv(DATA_PATH)

print("="*80)
print("ADDITIONAL INSIGHTS AND VARIANCE ANALYSIS")
print("="*80)

# ============================================================================
# VARIANCE STRUCTURE OVER X
# ============================================================================

print("\n" + "="*80)
print("VARIANCE STRUCTURE")
print("="*80)

# Compute residuals for log model (best non-quadratic)
log_x = np.log(df['x'])
z_log = np.polyfit(log_x, df['Y'], 1)
y_pred_log = z_log[1] + z_log[0] * np.log(df['x'])
residuals_log = df['Y'] - y_pred_log

# Sort by x
df_sorted = df.sort_values('x').copy()
df_sorted['residuals_log'] = df_sorted['Y'] - (z_log[1] + z_log[0] * np.log(df_sorted['x']))

# Moving window analysis
window_size = 9
rolling_var = []
rolling_x = []

for i in range(len(df_sorted) - window_size + 1):
    window = df_sorted.iloc[i:i+window_size]
    rolling_var.append(window['residuals_log'].var())
    rolling_x.append(window['x'].mean())

print(f"\nMoving window variance (window size = {window_size}):")
print(f"  Mean variance: {np.mean(rolling_var):.6f}")
print(f"  Min variance: {np.min(rolling_var):.6f}")
print(f"  Max variance: {np.max(rolling_var):.6f}")
print(f"  Ratio (max/min): {np.max(rolling_var)/np.min(rolling_var):.2f}")

# Plot variance structure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Absolute residuals vs x
axes[0].scatter(df['x'], np.abs(residuals_log), alpha=0.7, s=60, color='coral', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('x (Predictor)', fontsize=12)
axes[0].set_ylabel('|Residuals| (Log Model)', fontsize=12)
axes[0].set_title('Absolute Residuals vs x', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Rolling variance
axes[1].plot(rolling_x, rolling_var, linewidth=2.5, color='darkblue', marker='o', markersize=6)
axes[1].axhline(y=np.mean(rolling_var), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(rolling_var):.4f}')
axes[1].set_xlabel('x (Predictor)', fontsize=12)
axes[1].set_ylabel('Rolling Variance', fontsize=12)
axes[1].set_title(f'Rolling Variance (window={window_size})', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'variance_structure_analysis.png', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: variance_structure_analysis.png")

# ============================================================================
# REPLICATE ANALYSIS (Multiple Y values at same x)
# ============================================================================

print("\n" + "="*80)
print("REPLICATE ANALYSIS")
print("="*80)

# Find x values with replicates
x_counts = df['x'].value_counts()
replicates = x_counts[x_counts > 1].sort_index()

print(f"\nX values with replicates:")
print(f"  Number of x values with replicates: {len(replicates)}")
print(f"\nDetails:")

replicate_data = []
for x_val in replicates.index:
    subset = df[df['x'] == x_val]['Y']
    rep_mean = subset.mean()
    rep_std = subset.std()
    rep_var = subset.var()
    rep_range = subset.max() - subset.min()

    print(f"  x = {x_val:.1f}: n={len(subset)}, mean={rep_mean:.4f}, std={rep_std:.4f}, range={rep_range:.4f}")

    replicate_data.append({
        'x': x_val,
        'n': len(subset),
        'mean': rep_mean,
        'std': rep_std,
        'var': rep_var
    })

# Visualize replicate variability
if replicate_data:
    replicate_df = pd.DataFrame(replicate_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot replicates
    axes[0].scatter(df['x'], df['Y'], alpha=0.3, s=40, color='lightgray', label='All data')
    for x_val in replicates.index:
        subset = df[df['x'] == x_val]
        axes[0].scatter(subset['x'], subset['Y'], alpha=0.8, s=80, edgecolor='black', linewidth=1.5)
        axes[0].plot([x_val, x_val], [subset['Y'].min(), subset['Y'].max()],
                    color='red', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('x (Predictor)', fontsize=12)
    axes[0].set_ylabel('Y (Response)', fontsize=12)
    axes[0].set_title('Data Points with Replicates Highlighted', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Variance vs x for replicates
    axes[1].scatter(replicate_df['x'], replicate_df['std'], alpha=0.7, s=100, color='purple', edgecolor='black', linewidth=1)
    axes[1].set_xlabel('x (Predictor)', fontsize=12)
    axes[1].set_ylabel('Standard Deviation of Replicates', fontsize=12)
    axes[1].set_title('Replicate Variability Across x', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'replicate_analysis.png', bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved: replicate_analysis.png")

# ============================================================================
# DATA COVERAGE AND GAPS
# ============================================================================

print("\n" + "="*80)
print("DATA COVERAGE ANALYSIS")
print("="*80)

x_sorted = df['x'].sort_values().values
x_gaps = np.diff(x_sorted)
unique_x = np.sort(df['x'].unique())

print(f"\nX-value coverage:")
print(f"  Range: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
print(f"  Number of unique x values: {len(unique_x)}")
print(f"  Mean gap between consecutive unique x: {np.mean(np.diff(unique_x)):.3f}")
print(f"  Largest gap: {np.max(np.diff(unique_x)):.3f} (between {unique_x[np.argmax(np.diff(unique_x))]:.1f} and {unique_x[np.argmax(np.diff(unique_x))+1]:.1f})")

# Identify regions
low_x = df[df['x'] <= 5]
mid_x = df[(df['x'] > 5) & (df['x'] <= 15)]
high_x = df[df['x'] > 15]

print(f"\nData distribution by region:")
print(f"  Low x (x <= 5): n = {len(low_x)}, Y range = [{low_x['Y'].min():.3f}, {low_x['Y'].max():.3f}]")
print(f"  Mid x (5 < x <= 15): n = {len(mid_x)}, Y range = [{mid_x['Y'].min():.3f}, {mid_x['Y'].max():.3f}]")
print(f"  High x (x > 15): n = {len(high_x)}, Y range = [{high_x['Y'].min():.3f}, {high_x['Y'].max():.3f}]")

# ============================================================================
# LOG TRANSFORMATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("LOG TRANSFORMATION ANALYSIS")
print("="*80)

# Check if log(Y) is more normal
log_Y = np.log(df['Y'])

print(f"\nOriginal Y:")
print(f"  Mean: {df['Y'].mean():.4f}")
print(f"  Std: {df['Y'].std():.4f}")
print(f"  Skewness: {df['Y'].skew():.4f}")
print(f"  Kurtosis: {df['Y'].kurtosis():.4f}")

print(f"\nlog(Y):")
print(f"  Mean: {log_Y.mean():.4f}")
print(f"  Std: {log_Y.std():.4f}")
print(f"  Skewness: {log_Y.skew():.4f}")
print(f"  Kurtosis: {log_Y.kurtosis():.4f}")

# Test normality
shapiro_y = stats.shapiro(df['Y'])
shapiro_log_y = stats.shapiro(log_Y)

print(f"\nShapiro-Wilk normality test:")
print(f"  Y: statistic={shapiro_y.statistic:.4f}, p-value={shapiro_y.pvalue:.4f}")
print(f"  log(Y): statistic={shapiro_log_y.statistic:.4f}, p-value={shapiro_log_y.pvalue:.4f}")

# Visualize transformations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Y distribution
axes[0, 0].hist(df['Y'], bins=12, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
axes[0, 0].set_xlabel('Y', fontsize=11)
axes[0, 0].set_ylabel('Density', fontsize=11)
axes[0, 0].set_title('Distribution of Y', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# log(Y) distribution
axes[0, 1].hist(log_Y, bins=12, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].set_xlabel('log(Y)', fontsize=11)
axes[0, 1].set_ylabel('Density', fontsize=11)
axes[0, 1].set_title('Distribution of log(Y)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot Y
stats.probplot(df['Y'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Y', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot log(Y)
stats.probplot(log_Y, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: log(Y)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'transformation_analysis.png', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: transformation_analysis.png")

# ============================================================================
# SATURATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SATURATION/PLATEAU ANALYSIS")
print("="*80)

# Check if Y appears to plateau at high x
high_x_threshold = 15
low_subset = df[df['x'] <= high_x_threshold]
high_subset = df[df['x'] > high_x_threshold]

print(f"\nComparison of Y values:")
print(f"  x <= {high_x_threshold}:")
print(f"    Mean Y: {low_subset['Y'].mean():.4f}")
print(f"    Std Y: {low_subset['Y'].std():.4f}")
print(f"  x > {high_x_threshold}:")
print(f"    Mean Y: {high_subset['Y'].mean():.4f}")
print(f"    Std Y: {high_subset['Y'].std():.4f}")
print(f"\nDifference in means: {high_subset['Y'].mean() - low_subset['Y'].mean():.4f}")

# T-test for difference in means
t_stat, t_pval = stats.ttest_ind(low_subset['Y'], high_subset['Y'])
print(f"\nT-test for difference in means:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pval:.4f}")
if t_pval < 0.05:
    print(f"  -> Significant difference (p < 0.05)")
else:
    print(f"  -> No significant difference (p >= 0.05)")

# Rate of change analysis
df_sorted = df.sort_values('x')
df_sorted['Y_diff'] = df_sorted['Y'].diff()
df_sorted['x_diff'] = df_sorted['x'].diff()
df_sorted['rate_of_change'] = df_sorted['Y_diff'] / df_sorted['x_diff']

print(f"\nRate of change in Y per unit x:")
valid_rates = df_sorted['rate_of_change'].dropna()
print(f"  Mean: {valid_rates.mean():.6f}")
print(f"  Median: {valid_rates.median():.6f}")
print(f"  Min: {valid_rates.min():.6f}")
print(f"  Max: {valid_rates.max():.6f}")

# Plot rate of change
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(df_sorted['x'].iloc[1:], df_sorted['rate_of_change'].iloc[1:],
          alpha=0.7, s=60, color='orange', edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('x (Predictor)', fontsize=12)
ax.set_ylabel('Rate of Change (dY/dx)', fontsize=12)
ax.set_title('Rate of Change in Y per Unit x', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'rate_of_change_analysis.png', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: rate_of_change_analysis.png")

print("\n" + "="*80)
print("ADDITIONAL INSIGHTS COMPLETE")
print("="*80)
