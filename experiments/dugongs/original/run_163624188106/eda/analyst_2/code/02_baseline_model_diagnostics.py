"""
Baseline Model and Residual Diagnostics
Focus: Fit simple linear regression and examine residuals thoroughly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter

# Set style
sns.set_style("whitegrid")

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("=" * 80)
print("BASELINE LINEAR MODEL DIAGNOSTICS")
print("=" * 80)

# Fit simple linear regression: Y = a + b*x
X = data['x'].values
y = data['Y'].values

# Use scipy linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

# Get predictions and residuals
y_pred = intercept + slope * X
residuals = y - y_pred

# Model statistics
print("\nModel: Y = a + b*x")
print(f"Intercept (a): {intercept:.6f}")
print(f"Slope (b): {slope:.6f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.6e}")
print(f"Standard error of slope: {std_err:.6f}")
print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.6f}")
print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.6f}")

# Residual statistics
print("\n" + "=" * 80)
print("RESIDUAL STATISTICS")
print("=" * 80)
print(f"Mean of residuals: {np.mean(residuals):.6e}")
print(f"Std of residuals: {np.std(residuals, ddof=2):.6f}")
print(f"Min residual: {np.min(residuals):.6f}")
print(f"Max residual: {np.max(residuals):.6f}")
print(f"Skewness: {stats.skew(residuals):.4f}")
print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")

# Test for normality of residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\nShapiro-Wilk test for normality:")
print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")

# Test for autocorrelation (Durbin-Watson)
# DW = sum((e[i] - e[i-1])^2) / sum(e[i]^2)
dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"\nDurbin-Watson statistic: {dw_stat:.4f}")
print(f"  (2.0 = no autocorrelation, <2 = positive, >2 = negative)")

# Identify potential outliers
std_resid = residuals / np.std(residuals, ddof=2)
outlier_idx = np.where(np.abs(std_resid) > 2)[0]
print(f"\nPotential outliers (|standardized residual| > 2): {len(outlier_idx)}")
if len(outlier_idx) > 0:
    print("Indices:", outlier_idx)
    for idx in outlier_idx:
        print(f"  Point {idx}: x={data['x'].iloc[idx]:.2f}, Y={data['Y'].iloc[idx]:.2f}, "
              f"predicted={y_pred[idx]:.2f}, residual={residuals[idx]:.4f}")

# Check for patterns in residuals vs x
# Divide x into 3 regions and check variance homogeneity
x_sorted = np.sort(data['x'].unique())
n_unique = len(x_sorted)
third = n_unique // 3

region1_x = x_sorted[:third]
region2_x = x_sorted[third:2*third]
region3_x = x_sorted[2*third:]

region1_res = residuals[data['x'].isin(region1_x)]
region2_res = residuals[data['x'].isin(region2_x)]
region3_res = residuals[data['x'].isin(region3_x)]

print("\n" + "=" * 80)
print("RESIDUAL PATTERNS BY REGION")
print("=" * 80)
print(f"Region 1 (x in [{region1_x[0]:.1f}, {region1_x[-1]:.1f}]): n={len(region1_res)}, "
      f"mean={np.mean(region1_res):.4f}, std={np.std(region1_res):.4f}")
print(f"Region 2 (x in [{region2_x[0]:.1f}, {region2_x[-1]:.1f}]): n={len(region2_res)}, "
      f"mean={np.mean(region2_res):.4f}, std={np.std(region2_res):.4f}")
print(f"Region 3 (x in [{region3_x[0]:.1f}, {region3_x[-1]:.1f}]): n={len(region3_res)}, "
      f"mean={np.mean(region3_res):.4f}, std={np.std(region3_res):.4f}")

# Levene's test for homogeneity of variance
levene_stat, levene_p = stats.levene(region1_res, region2_res, region3_res)
print(f"\nLevene's test for homogeneity of variance:")
print(f"  Statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}")

# Create comprehensive diagnostic plots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Fitted vs Actual
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(data['x'], data['Y'], alpha=0.6, label='Actual', s=50)
ax1.plot(data['x'], y_pred, 'r-', linewidth=2, label='Fitted')
ax1.set_xlabel('x')
ax1.set_ylabel('Y')
ax1.set_title('Actual vs Fitted Values')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residuals vs Fitted
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.axhline(y=2*np.std(residuals, ddof=2), color='orange', linestyle=':', linewidth=1)
ax2.axhline(y=-2*np.std(residuals, ddof=2), color='orange', linestyle=':', linewidth=1)
ax2.set_xlabel('Fitted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Fitted')
ax2.grid(True, alpha=0.3)

# 3. Residuals vs x
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(data['x'], residuals, alpha=0.6, s=50)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.axhline(y=2*np.std(residuals, ddof=2), color='orange', linestyle=':', linewidth=1)
ax3.axhline(y=-2*np.std(residuals, ddof=2), color='orange', linestyle=':', linewidth=1)
ax3.set_xlabel('x')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals vs x')
ax3.grid(True, alpha=0.3)

# 4. Q-Q plot
ax4 = fig.add_subplot(gs[1, 0])
stats.probplot(residuals, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot of Residuals')

# 5. Histogram of residuals
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(residuals, bins=12, edgecolor='black', alpha=0.7, density=True)
x_range = np.linspace(residuals.min(), residuals.max(), 100)
ax5.plot(x_range, stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals, ddof=2)),
         'r-', linewidth=2, label='Normal PDF')
ax5.set_xlabel('Residuals')
ax5.set_ylabel('Density')
ax5.set_title('Distribution of Residuals')
ax5.legend()

# 6. Standardized residuals
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(range(len(std_resid)), std_resid, alpha=0.6, s=50)
ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax6.axhline(y=2, color='orange', linestyle=':', linewidth=1)
ax6.axhline(y=-2, color='orange', linestyle=':', linewidth=1)
ax6.set_xlabel('Observation Index')
ax6.set_ylabel('Standardized Residuals')
ax6.set_title('Standardized Residuals')
ax6.grid(True, alpha=0.3)

# 7. Scale-Location plot (sqrt of absolute standardized residuals)
ax7 = fig.add_subplot(gs[2, 0])
sqrt_abs_std_resid = np.sqrt(np.abs(std_resid))
ax7.scatter(y_pred, sqrt_abs_std_resid, alpha=0.6, s=50)
# Add lowess smoothing line
sort_idx = np.argsort(y_pred)
if len(y_pred) > 5:
    smoothed = savgol_filter(sqrt_abs_std_resid[sort_idx], min(11, len(y_pred)//2*2+1), 3)
    ax7.plot(y_pred[sort_idx], smoothed, 'r-', linewidth=2)
ax7.set_xlabel('Fitted Values')
ax7.set_ylabel('√|Standardized Residuals|')
ax7.set_title('Scale-Location Plot')
ax7.grid(True, alpha=0.3)

# 8. Residuals by region
ax8 = fig.add_subplot(gs[2, 1])
bp = ax8.boxplot([region1_res, region2_res, region3_res],
                   tick_labels=['Low x', 'Mid x', 'High x'],
                   patch_artist=True)
ax8.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax8.set_ylabel('Residuals')
ax8.set_title('Residuals by x Region')
ax8.grid(True, alpha=0.3)

# 9. Autocorrelation plot
ax9 = fig.add_subplot(gs[2, 2])
# Sort by x to check for spatial autocorrelation
sorted_idx = np.argsort(data['x'].values)
sorted_residuals = residuals[sorted_idx]
lags = range(1, min(11, len(residuals)//2))
autocorr = [np.corrcoef(sorted_residuals[:-lag], sorted_residuals[lag:])[0,1] for lag in lags]
ax9.bar(lags, autocorr, alpha=0.7)
ax9.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax9.axhline(y=1.96/np.sqrt(len(residuals)), color='orange', linestyle=':', linewidth=1)
ax9.axhline(y=-1.96/np.sqrt(len(residuals)), color='orange', linestyle=':', linewidth=1)
ax9.set_xlabel('Lag')
ax9.set_ylabel('Autocorrelation')
ax9.set_title('Residual Autocorrelation (by x order)')
ax9.grid(True, alpha=0.3)

plt.savefig('/workspace/eda/analyst_2/visualizations/02_baseline_diagnostics.png', dpi=300, bbox_inches='tight')
print("\nSaved: /workspace/eda/analyst_2/visualizations/02_baseline_diagnostics.png")
plt.close()

# Save residuals for further analysis
residuals_df = pd.DataFrame({
    'x': data['x'],
    'Y': data['Y'],
    'fitted': y_pred,
    'residuals': residuals,
    'std_residuals': std_resid
})
residuals_df.to_csv('/workspace/eda/analyst_2/code/baseline_residuals.csv', index=False)
print("Saved residuals to: /workspace/eda/analyst_2/code/baseline_residuals.csv")

print("\n" + "=" * 80)
print("BASELINE MODEL DIAGNOSTICS COMPLETE")
print("=" * 80)
