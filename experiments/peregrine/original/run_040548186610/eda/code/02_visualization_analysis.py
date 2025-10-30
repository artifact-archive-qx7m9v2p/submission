"""
Visualization and Trend Analysis
=================================
Create comprehensive visualizations to understand patterns and relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
from pathlib import Path

# Paths
DATA_PATH = '/workspace/data/data.csv'
VIZ_DIR = Path('/workspace/eda/visualizations')

# Load data
df = pd.read_csv(DATA_PATH)

# Set high-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

print("Creating visualizations...")

# ============================================================================
# 1. TIME SERIES PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['year'], df['C'], 'o-', linewidth=2, markersize=6,
        alpha=0.7, label='Observed counts', color='steelblue')
ax.set_xlabel('Year (normalized)', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title('Time Series of Count Data', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(VIZ_DIR / 'timeseries_plot.png', dpi=300)
plt.close()
print("  ✓ Time series plot saved")

# ============================================================================
# 2. DISTRIBUTION OF C (Histogram with density overlay)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
counts, bins, patches = ax.hist(df['C'], bins=20, density=True,
                                  alpha=0.7, edgecolor='black', label='Histogram')
# Overlay KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(df['C'])
x_range = np.linspace(df['C'].min(), df['C'].max(), 1000)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

# Add mean and median lines
ax.axvline(df['C'].mean(), color='green', linestyle='--', linewidth=2,
           label=f'Mean = {df["C"].mean():.1f}')
ax.axvline(df['C'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median = {df["C"].median():.1f}')

ax.set_xlabel('Count (C)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Count Variable', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'count_distribution.png', dpi=300)
plt.close()
print("  ✓ Count distribution saved")

# ============================================================================
# 3. SCATTER PLOT WITH SMOOTHING
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Scatter plot
ax.scatter(df['year'], df['C'], s=80, alpha=0.6, edgecolors='black',
           linewidth=0.5, label='Observed data', color='steelblue')

# Linear regression line using numpy
X = df['year'].values
y = df['C'].values

# Manual linear regression
slope, intercept = np.polyfit(X, y, 1)
y_pred_linear = slope * X + intercept

# Calculate R-squared
ss_res = np.sum((y - y_pred_linear)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_linear = 1 - (ss_res / ss_tot)

ax.plot(df['year'], y_pred_linear, 'r--', linewidth=2,
        label=f'Linear fit (R² = {r2_linear:.3f})')

# Polynomial fit (degree 2)
z = np.polyfit(df['year'], df['C'], 2)
p = np.poly1d(z)
y_pred_poly = p(df['year'])
ss_res_poly = np.sum((y - y_pred_poly)**2)
r2_poly = 1 - (ss_res_poly / ss_tot)

ax.plot(df['year'], y_pred_poly, 'purple', linestyle=':', linewidth=2,
        label=f'Polynomial (deg=2, R² = {r2_poly:.3f})')

# Smoothed curve using Savitzky-Golay filter
y_smooth = savgol_filter(df['C'], window_length=11, polyorder=3)
ax.plot(df['year'], y_smooth, 'g-', linewidth=2.5,
        label='Smoothed (Savgol filter)', alpha=0.8)

ax.set_xlabel('Year (normalized)', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title('Relationship Between Year and Count with Fitted Curves',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'scatter_with_smoothing.png', dpi=300)
plt.close()
print("  ✓ Scatter with smoothing saved")

# Store R-squared values for later
print(f"\n  Linear R²: {r2_linear:.4f}")
print(f"  Polynomial (deg=2) R²: {r2_poly:.4f}")
print(f"  Linear equation: C = {slope:.2f} * year + {intercept:.2f}")

# Correlation
pearson_r, pearson_p = stats.pearsonr(df['year'], df['C'])
spearman_r, spearman_p = stats.spearmanr(df['year'], df['C'])
print(f"  Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4e}")
print(f"  Spearman correlation: rho = {spearman_r:.4f}, p = {spearman_p:.4e}")

# ============================================================================
# 4. RESIDUAL ANALYSIS FROM LINEAR FIT
# ============================================================================
residuals = y - y_pred_linear

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs fitted
axes[0, 0].scatter(y_pred_linear, residuals, alpha=0.6, edgecolors='black', s=60)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted values', fontsize=11)
axes[0, 0].set_ylabel('Residuals', fontsize=11)
axes[0, 0].set_title('Residuals vs Fitted Values', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Add polynomial fit to residuals to check for pattern
poly_resid = np.polyfit(y_pred_linear, residuals, 2)
p_resid = np.poly1d(poly_resid)
sorted_idx = np.argsort(y_pred_linear)
axes[0, 0].plot(y_pred_linear[sorted_idx], p_resid(y_pred_linear[sorted_idx]),
                'g-', linewidth=2, alpha=0.7, label='Quadratic trend')
axes[0, 0].legend()

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot of Residuals', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Histogram of residuals
axes[1, 0].hist(residuals, bins=15, density=True, alpha=0.7,
                edgecolor='black', color='steelblue')
# Overlay normal distribution
mu, sigma = residuals.mean(), residuals.std()
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 0].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
                'r-', linewidth=2, label='Normal fit')
axes[1, 0].set_xlabel('Residuals', fontsize=11)
axes[1, 0].set_ylabel('Density', fontsize=11)
axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Residuals vs time (to check for temporal patterns)
axes[1, 1].scatter(df['year'], residuals, alpha=0.6, edgecolors='black', s=60)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Year (normalized)', fontsize=11)
axes[1, 1].set_ylabel('Residuals', fontsize=11)
axes[1, 1].set_title('Residuals vs Year', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Add polynomial fit
poly_resid_time = np.polyfit(df['year'], residuals, 2)
p_resid_time = np.poly1d(poly_resid_time)
sorted_idx_time = np.argsort(df['year'])
axes[1, 1].plot(df['year'][sorted_idx_time], p_resid_time(df['year'][sorted_idx_time]),
                'g-', linewidth=2, alpha=0.7, label='Quadratic trend')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(VIZ_DIR / 'residual_diagnostics.png', dpi=300)
plt.close()
print("  ✓ Residual diagnostics saved")

# Statistical tests on residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\n  Shapiro-Wilk test for normality: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")

# ============================================================================
# 5. VARIANCE ANALYSIS - Check if variance increases with mean
# ============================================================================
# Split data into temporal bins and compute statistics
n_bins = 4
df['time_bin'] = pd.qcut(df['year'], q=n_bins, labels=['Q1', 'Q2', 'Q3', 'Q4'])

bin_stats = df.groupby('time_bin')['C'].agg(['mean', 'var', 'std', 'count'])
bin_stats['cv'] = bin_stats['std'] / bin_stats['mean']  # Coefficient of variation
bin_stats['var_to_mean'] = bin_stats['var'] / bin_stats['mean']

print("\n\nVariance by time period:")
print(bin_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean vs Variance
axes[0].scatter(bin_stats['mean'], bin_stats['var'], s=200, alpha=0.7,
                edgecolors='black', linewidth=2, color='steelblue')
for i, label in enumerate(bin_stats.index):
    axes[0].annotate(label, (bin_stats['mean'].iloc[i], bin_stats['var'].iloc[i]),
                     fontsize=12, ha='center', fontweight='bold')
# Add reference lines
x_range = np.linspace(bin_stats['mean'].min() * 0.9, bin_stats['mean'].max() * 1.1, 100)
axes[0].plot(x_range, x_range, 'r--', linewidth=2,
             label='Var = Mean (Poisson)', alpha=0.7)
axes[0].set_xlabel('Mean Count', fontsize=12)
axes[0].set_ylabel('Variance', fontsize=12)
axes[0].set_title('Mean-Variance Relationship by Time Period', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Variance-to-Mean ratio over time
time_labels = [f'Q{i+1}' for i in range(n_bins)]
axes[1].bar(time_labels, bin_stats['var_to_mean'], alpha=0.7,
            edgecolor='black', linewidth=2, color='coral')
axes[1].axhline(1, color='red', linestyle='--', linewidth=2,
                label='Poisson expectation (ratio = 1)', alpha=0.7)
axes[1].set_xlabel('Time Period', fontsize=12)
axes[1].set_ylabel('Variance-to-Mean Ratio', fontsize=12)
axes[1].set_title('Overdispersion by Time Period', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VIZ_DIR / 'variance_analysis.png', dpi=300)
plt.close()
print("  ✓ Variance analysis saved")

# ============================================================================
# 6. BOX PLOT BY TIME PERIOD
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='C', by='time_bin', ax=ax, patch_artist=True)
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title('Distribution of Counts by Time Period', fontweight='bold')
plt.suptitle('')  # Remove the default title
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'boxplot_by_period.png', dpi=300)
plt.close()
print("  ✓ Box plot by period saved")

# ============================================================================
# 7. SIMPLE AUTOCORRELATION PLOT (Manual)
# ============================================================================
def compute_acf(series, nlags=20):
    """Compute autocorrelation function manually."""
    acf_vals = []
    series = np.array(series)
    series_centered = series - series.mean()
    variance = np.var(series)

    for lag in range(nlags + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            c0 = series_centered[:-lag]
            c_lag = series_centered[lag:]
            acf_vals.append(np.corrcoef(c0, c_lag)[0, 1])

    return np.array(acf_vals)

lags = 15
acf_vals = compute_acf(df['C'], nlags=lags)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(lags + 1), acf_vals, width=0.3, alpha=0.7,
       edgecolor='black', color='steelblue')
ax.axhline(0, color='black', linewidth=0.8)

# Add confidence intervals (approximate)
conf_level = 1.96 / np.sqrt(len(df))
ax.axhline(conf_level, color='red', linestyle='--', linewidth=1.5,
           label='95% confidence')
ax.axhline(-conf_level, color='red', linestyle='--', linewidth=1.5)

ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.legend()
plt.tight_layout()
plt.savefig(VIZ_DIR / 'autocorrelation_plot.png', dpi=300)
plt.close()
print("  ✓ Autocorrelation plot saved")

# ============================================================================
# 8. LOG TRANSFORMATION ANALYSIS
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log-log plot to check for exponential relationship
axes[0].scatter(df['year'], np.log(df['C']), s=80, alpha=0.6,
                edgecolors='black', linewidth=0.5, color='darkgreen')

# Fit linear model to log(C) vs year
slope_log, intercept_log = np.polyfit(df['year'], np.log(df['C']), 1)
y_pred_log = slope_log * df['year'] + intercept_log
axes[0].plot(df['year'], y_pred_log, 'r--', linewidth=2,
             label=f'Linear fit: log(C) = {slope_log:.3f}*year + {intercept_log:.3f}')

axes[0].set_xlabel('Year (normalized)', fontsize=12)
axes[0].set_ylabel('log(Count)', fontsize=12)
axes[0].set_title('Log-Transformed Counts vs Year', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution of log-transformed counts
axes[1].hist(np.log(df['C']), bins=15, density=True, alpha=0.7,
             edgecolor='black', color='darkgreen')

# Overlay normal distribution
log_counts = np.log(df['C'])
mu_log, sigma_log = log_counts.mean(), log_counts.std()
x_log = np.linspace(log_counts.min(), log_counts.max(), 100)
axes[1].plot(x_log, stats.norm.pdf(x_log, mu_log, sigma_log),
             'r-', linewidth=2, label='Normal fit')

axes[1].set_xlabel('log(Count)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Distribution of Log-Transformed Counts', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'log_transformation_analysis.png', dpi=300)
plt.close()
print("  ✓ Log transformation analysis saved")

# R-squared for log model
ss_res_log = np.sum((np.log(df['C']) - y_pred_log)**2)
ss_tot_log = np.sum((np.log(df['C']) - np.log(df['C']).mean())**2)
r2_log = 1 - (ss_res_log / ss_tot_log)
print(f"\n  R² for log-linear model: {r2_log:.4f}")
print(f"  This suggests exponential growth rate: {np.exp(slope_log):.4f} per unit year")

print("\n✓ All visualizations created successfully!")
