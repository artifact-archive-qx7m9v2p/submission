"""
Script 3: Bivariate Analysis
Analyzes the relationship between x and Y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.interpolate import UnivariateSpline

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
print("BIVARIATE ANALYSIS: Y vs x")
print("="*80)

# ============================================================================
# SCATTER PLOT WITH VARIOUS FITS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Basic scatter plot with linear fit
axes[0, 0].scatter(df['x'], df['Y'], alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
# Linear regression
z_linear = np.polyfit(df['x'], df['Y'], 1)
p_linear = np.poly1d(z_linear)
x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
axes[0, 0].plot(x_line, p_linear(x_line), 'r--', linewidth=2, label=f'Linear: Y = {z_linear[0]:.4f}x + {z_linear[1]:.4f}')
axes[0, 0].set_xlabel('x (Predictor)', fontsize=11)
axes[0, 0].set_ylabel('Y (Response)', fontsize=11)
axes[0, 0].set_title('Scatter Plot with Linear Fit', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Scatter plot with quadratic fit
axes[0, 1].scatter(df['x'], df['Y'], alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
z_quad = np.polyfit(df['x'], df['Y'], 2)
p_quad = np.poly1d(z_quad)
axes[0, 1].plot(x_line, p_quad(x_line), 'g--', linewidth=2, label=f'Quadratic')
axes[0, 1].set_xlabel('x (Predictor)', fontsize=11)
axes[0, 1].set_ylabel('Y (Response)', fontsize=11)
axes[0, 1].set_title('Scatter Plot with Quadratic Fit', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Scatter plot with logarithmic fit
axes[1, 0].scatter(df['x'], df['Y'], alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
# Log fit: Y = a + b*log(x)
log_x = np.log(df['x'])
z_log = np.polyfit(log_x, df['Y'], 1)
x_log_line = np.linspace(df['x'].min(), df['x'].max(), 100)
y_log_pred = z_log[0] * np.log(x_log_line) + z_log[1]
axes[1, 0].plot(x_log_line, y_log_pred, 'm--', linewidth=2, label=f'Log: Y = {z_log[0]:.4f}*ln(x) + {z_log[1]:.4f}')
axes[1, 0].set_xlabel('x (Predictor)', fontsize=11)
axes[1, 0].set_ylabel('Y (Response)', fontsize=11)
axes[1, 0].set_title('Scatter Plot with Logarithmic Fit', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot with LOESS/smoothing spline
axes[1, 1].scatter(df['x'], df['Y'], alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5, label='Data')
# Sort by x for spline fitting
df_sorted = df.sort_values('x')
# Smoothing spline
spline = UnivariateSpline(df_sorted['x'], df_sorted['Y'], s=0.5, k=3)
x_smooth = np.linspace(df['x'].min(), df['x'].max(), 200)
y_smooth = spline(x_smooth)
axes[1, 1].plot(x_smooth, y_smooth, 'orange', linewidth=2, label='Smoothing Spline')
axes[1, 1].set_xlabel('x (Predictor)', fontsize=11)
axes[1, 1].set_ylabel('Y (Response)', fontsize=11)
axes[1, 1].set_title('Scatter Plot with Smoothing Spline', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'bivariate_scatter_various_fits.png', bbox_inches='tight')
plt.close()

print("\nScatter plots with various fits saved: bivariate_scatter_various_fits.png")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Pearson correlation
pearson_r, pearson_p = stats.pearsonr(df['x'], df['Y'])
print(f"\nPearson correlation:")
print(f"  r = {pearson_r:.6f}")
print(f"  p-value = {pearson_p:.6f}")
print(f"  R-squared = {pearson_r**2:.6f}")

# Spearman correlation (rank-based, robust to non-linearity)
spearman_r, spearman_p = stats.spearmanr(df['x'], df['Y'])
print(f"\nSpearman correlation:")
print(f"  rho = {spearman_r:.6f}")
print(f"  p-value = {spearman_p:.6f}")

# Kendall's tau
kendall_tau, kendall_p = stats.kendalltau(df['x'], df['Y'])
print(f"\nKendall's tau:")
print(f"  tau = {kendall_tau:.6f}")
print(f"  p-value = {kendall_p:.6f}")

# ============================================================================
# RESIDUAL ANALYSIS (LINEAR MODEL)
# ============================================================================

print("\n" + "="*80)
print("RESIDUAL ANALYSIS (Linear Model)")
print("="*80)

# Fit linear model
y_pred_linear = p_linear(df['x'])
residuals_linear = df['Y'] - y_pred_linear

print(f"\nResidual statistics:")
print(f"  Mean: {residuals_linear.mean():.6f}")
print(f"  Std: {residuals_linear.std():.6f}")
print(f"  Min: {residuals_linear.min():.6f}")
print(f"  Max: {residuals_linear.max():.6f}")

# Create residual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Residuals vs fitted values
axes[0, 0].scatter(y_pred_linear, residuals_linear, alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values', fontsize=11)
axes[0, 0].set_ylabel('Residuals', fontsize=11)
axes[0, 0].set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals vs x
axes[0, 1].scatter(df['x'], residuals_linear, alpha=0.7, s=60, color='forestgreen', edgecolors='black', linewidth=0.5)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('x (Predictor)', fontsize=11)
axes[0, 1].set_ylabel('Residuals', fontsize=11)
axes[0, 1].set_title('Residuals vs Predictor', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q plot of residuals
stats.probplot(residuals_linear, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Histogram of residuals
axes[1, 1].hist(residuals_linear, bins=10, edgecolor='black', alpha=0.7, color='coral')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Residuals', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'bivariate_residual_analysis.png', bbox_inches='tight')
plt.close()

print("\nResidual plots saved: bivariate_residual_analysis.png")

# Test for normality of residuals
shapiro_resid = stats.shapiro(residuals_linear)
print(f"\nShapiro-Wilk test on residuals:")
print(f"  statistic = {shapiro_resid.statistic:.6f}")
print(f"  p-value = {shapiro_resid.pvalue:.6f}")

# ============================================================================
# HETEROSCEDASTICITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("HETEROSCEDASTICITY ANALYSIS")
print("="*80)

# Group data by x ranges and check variance
df_sorted = df.sort_values('x').reset_index(drop=True)
n_groups = 3
group_size = len(df_sorted) // n_groups

groups = []
for i in range(n_groups):
    start_idx = i * group_size
    if i == n_groups - 1:
        end_idx = len(df_sorted)
    else:
        end_idx = (i + 1) * group_size
    groups.append(df_sorted.iloc[start_idx:end_idx])

print(f"\nVariance by x-value groups:")
for i, group in enumerate(groups):
    print(f"  Group {i+1} (x: {group['x'].min():.2f} - {group['x'].max():.2f}):")
    print(f"    n = {len(group)}")
    print(f"    Variance of Y = {group['Y'].var():.6f}")
    print(f"    Std Dev of Y = {group['Y'].std():.6f}")

# Breusch-Pagan test (manual implementation)
# Regress squared residuals on x
residuals_squared = residuals_linear ** 2
z_resid = np.polyfit(df['x'], residuals_squared, 1)
p_resid = np.poly1d(z_resid)
resid_sq_pred = p_resid(df['x'])
ss_resid = np.sum((residuals_squared - residuals_squared.mean())**2)
ss_model_resid = np.sum((resid_sq_pred - residuals_squared.mean())**2)
bp_stat = ss_model_resid / (ss_resid / len(df))
bp_pvalue = 1 - stats.chi2.cdf(bp_stat, 1)

print(f"\nBreusch-Pagan test for heteroscedasticity:")
print(f"  Test statistic = {bp_stat:.6f}")
print(f"  p-value = {bp_pvalue:.6f}")
if bp_pvalue < 0.05:
    print(f"  -> Evidence of heteroscedasticity (p < 0.05)")
else:
    print(f"  -> No strong evidence of heteroscedasticity (p >= 0.05)")

# ============================================================================
# INFLUENTIAL POINTS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("INFLUENTIAL POINTS / OUTLIERS")
print("="*80)

# Standardized residuals
std_residuals = residuals_linear / residuals_linear.std()
print(f"\nStandardized residuals:")
outlier_indices = np.where(np.abs(std_residuals) > 2)[0]
if len(outlier_indices) > 0:
    print(f"  Points with |standardized residual| > 2:")
    for idx in outlier_indices:
        print(f"    Index {idx}: x={df.loc[idx, 'x']:.2f}, Y={df.loc[idx, 'Y']:.4f}, std_resid={std_residuals[idx]:.4f}")
else:
    print(f"  No points with |standardized residual| > 2")

# Cook's distance (leverage)
# H = X(X'X)^-1X'
X = np.column_stack([np.ones(len(df)), df['x']])
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(H)
cooks_d = (std_residuals**2 / 2) * (leverage / (1 - leverage)**2)

print(f"\nCook's distance:")
high_cooks = np.where(cooks_d > 4/len(df))[0]
if len(high_cooks) > 0:
    print(f"  Points with Cook's D > {4/len(df):.4f}:")
    for idx in high_cooks:
        print(f"    Index {idx}: x={df.loc[idx, 'x']:.2f}, Y={df.loc[idx, 'Y']:.4f}, Cook's D={cooks_d[idx]:.4f}")
else:
    print(f"  No highly influential points (Cook's D > {4/len(df):.4f})")

print("\n" + "="*80)
print("BIVARIATE ANALYSIS COMPLETE")
print("="*80)
