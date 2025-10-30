"""
Bivariate Analysis
==================
Examine relationship between Y and x
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data.csv')

# Create comprehensive bivariate visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Bivariate Analysis: Y vs x', fontsize=16, fontweight='bold')

# 1. Basic scatter plot
ax = axes[0, 0]
ax.scatter(df['x'], df['Y'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Scatter Plot', fontsize=12)
ax.grid(True, alpha=0.3)

# 2. Scatter with linear regression
ax = axes[0, 1]
ax.scatter(df['x'], df['Y'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
# Fit linear model
z = np.polyfit(df['x'], df['Y'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Linear fit: Y = {z[0]:.4f}x + {z[1]:.4f}')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Linear Regression Fit', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Scatter with LOESS smoothing
ax = axes[0, 2]
ax.scatter(df['x'], df['Y'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
# Use lowess from statsmodels if available, otherwise use spline
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(df['Y'], df['x'], frac=0.4)
    ax.plot(smoothed[:, 0], smoothed[:, 1], 'g-', linewidth=2, label='LOESS smooth')
except:
    # Fallback to spline
    df_sorted = df.sort_values('x')
    spline = UnivariateSpline(df_sorted['x'], df_sorted['Y'], s=0.1)
    ax.plot(x_line, spline(x_line), 'g-', linewidth=2, label='Spline smooth')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Nonparametric Smooth', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Residual plot (from linear fit)
ax = axes[1, 0]
residuals = df['Y'] - p(df['x'])
ax.scatter(df['x'], residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residual Plot (Linear Model)', fontsize=12)
ax.grid(True, alpha=0.3)

# 5. Residuals vs fitted values
ax = axes[1, 1]
fitted = p(df['x'])
ax.scatter(fitted, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted Values', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residuals vs Fitted', fontsize=12)
ax.grid(True, alpha=0.3)

# 6. Q-Q plot of residuals
ax = axes[1, 2]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Residuals', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/03_bivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional analysis plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Variance and Rate of Change Analysis', fontsize=14, fontweight='bold')

# 7. Absolute residuals vs x (heteroscedasticity check)
ax = axes[0]
abs_residuals = np.abs(residuals)
ax.scatter(df['x'], abs_residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
# Add trend line
z_abs = np.polyfit(df['x'], abs_residuals, 1)
p_abs = np.poly1d(z_abs)
ax.plot(x_line, p_abs(x_line), "r--", linewidth=2, label='Trend')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('|Residuals|', fontsize=12)
ax.set_title('Absolute Residuals vs x (Heteroscedasticity Check)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 8. Local slope analysis (rate of change)
ax = axes[1]
# Sort by x for sequential analysis
df_sorted = df.sort_values('x').reset_index(drop=True)
# Calculate local slopes (finite differences)
local_slopes = []
x_midpoints = []
for i in range(len(df_sorted) - 1):
    dx = df_sorted.loc[i+1, 'x'] - df_sorted.loc[i, 'x']
    dy = df_sorted.loc[i+1, 'Y'] - df_sorted.loc[i, 'Y']
    if dx != 0:
        slope = dy / dx
        local_slopes.append(slope)
        x_midpoints.append((df_sorted.loc[i, 'x'] + df_sorted.loc[i+1, 'x']) / 2)

ax.scatter(x_midpoints, local_slopes, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.axhline(y=np.mean(local_slopes), color='r', linestyle='--', linewidth=2, label=f'Mean slope: {np.mean(local_slopes):.4f}')
ax.set_xlabel('x (midpoint)', fontsize=12)
ax.set_ylabel('Local Slope (dY/dx)', fontsize=12)
ax.set_title('Rate of Change Analysis', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/04_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation and statistical tests
print("=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

pearson_r, pearson_p = stats.pearsonr(df['x'], df['Y'])
spearman_r, spearman_p = stats.spearmanr(df['x'], df['Y'])

print(f"\nPearson correlation: r = {pearson_r:.4f}, p-value = {pearson_p:.6f}")
print(f"Spearman correlation: rho = {spearman_r:.4f}, p-value = {spearman_p:.6f}")
print(f"R-squared (linear): {pearson_r**2:.4f}")

print("\n" + "=" * 80)
print("LINEAR REGRESSION STATISTICS")
print("=" * 80)

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['x'], df['Y'])
print(f"\nSlope: {slope:.6f} +/- {std_err:.6f}")
print(f"Intercept: {intercept:.6f}")
print(f"R-squared: {r_value**2:.6f}")
print(f"P-value: {p_value:.6f}")
print(f"Standard error: {std_err:.6f}")

# Residual statistics
print("\n" + "=" * 80)
print("RESIDUAL ANALYSIS")
print("=" * 80)
print(f"\nResidual mean: {residuals.mean():.6f}")
print(f"Residual std: {residuals.std():.6f}")
print(f"Residual range: [{residuals.min():.6f}, {residuals.max():.6f}]")
print(f"Residual skewness: {stats.skew(residuals):.6f}")
print(f"Residual kurtosis: {stats.kurtosis(residuals):.6f}")

# Test for heteroscedasticity
print("\n" + "=" * 80)
print("HETEROSCEDASTICITY TEST")
print("=" * 80)
# Breusch-Pagan test approximation: correlate squared residuals with x
corr_het, p_het = stats.pearsonr(df['x'], residuals**2)
print(f"Correlation between x and squared residuals: {corr_het:.4f}, p-value = {p_het:.4f}")
print("Note: Positive correlation suggests increasing variance with x")

# Test for autocorrelation in residuals
print("\n" + "=" * 80)
print("AUTOCORRELATION TEST (Durbin-Watson)")
print("=" * 80)
# Sort by x for sequential analysis
df_sorted = df.sort_values('x').reset_index(drop=True)
residuals_sorted = df_sorted['Y'] - p(df_sorted['x'])
# Durbin-Watson statistic
dw = np.sum(np.diff(residuals_sorted)**2) / np.sum(residuals_sorted**2)
print(f"Durbin-Watson statistic: {dw:.4f}")
print("Note: DW ~ 2 indicates no autocorrelation, < 2 positive autocorr, > 2 negative autocorr")
