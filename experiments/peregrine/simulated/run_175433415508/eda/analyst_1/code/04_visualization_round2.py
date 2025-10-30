"""
Visualization Round 2: Detrending and Residual Diagnostics
EDA Analyst 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')
C = data['C'].values
year = data['year'].values

# ============================================================================
# PLOT 5: Residual diagnostics (4-panel)
# ============================================================================

# Linear model
slope, intercept, r_value, p_value, std_err = stats.linregress(year, C)
predicted = slope * year + intercept
residuals = C - predicted

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Residuals vs Fitted
axes[0, 0].scatter(predicted, residuals, alpha=0.6, s=70, color='steelblue', edgecolor='black', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Fitted Values\n(Check for heteroscedasticity)', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Add lowess smooth to detect patterns
from scipy.signal import savgol_filter
sorted_idx = np.argsort(predicted)
smoothed = savgol_filter(residuals[sorted_idx], window_length=11, polyorder=2)
axes[0, 0].plot(predicted[sorted_idx], smoothed, 'orange', linewidth=2, label='Smoothed trend')
axes[0, 0].legend()

# Panel 2: Q-Q plot of residuals
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Panel 3: Scale-Location (sqrt of standardized residuals)
standardized_residuals = residuals / np.std(residuals, ddof=1)
sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))
axes[1, 0].scatter(predicted, sqrt_std_residuals, alpha=0.6, s=70, color='steelblue', edgecolor='black', linewidth=0.5)
axes[1, 0].set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Scale-Location Plot\n(Check for homoscedasticity)', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Add smoothed line
sorted_idx2 = np.argsort(predicted)
smoothed2 = savgol_filter(sqrt_std_residuals[sorted_idx2], window_length=11, polyorder=2)
axes[1, 0].plot(predicted[sorted_idx2], smoothed2, 'orange', linewidth=2, label='Smoothed trend')
axes[1, 0].legend()

# Panel 4: Histogram of residuals
axes[1, 1].hist(residuals, bins=12, edgecolor='black', alpha=0.7, color='steelblue', density=True)
# Overlay normal distribution
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std(ddof=1)),
                'r-', linewidth=2, label='Normal fit')
axes[1, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/05_residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 05_residual_diagnostics.png")

# ============================================================================
# PLOT 6: Mean-Variance relationship
# ============================================================================

# Create rolling windows
window_size = 10
n_windows = len(C) - window_size + 1

window_means = []
window_vars = []

for i in range(n_windows):
    window_C = C[i:i+window_size]
    window_means.append(window_C.mean())
    window_vars.append(window_C.var(ddof=1))

window_means = np.array(window_means)
window_vars = np.array(window_vars)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Linear scale
axes[0].scatter(window_means, window_vars, alpha=0.6, s=80, color='steelblue', edgecolor='black', linewidth=1)

# Add reference lines
max_val = max(window_means.max(), window_vars.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7, label='Var = Mean (Poisson)')
axes[0].plot([0, max_val], [0, 2*max_val], 'g--', linewidth=2, alpha=0.7, label='Var = 2×Mean')

axes[0].set_xlabel('Window Mean', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Window Variance', fontsize=12, fontweight='bold')
axes[0].set_title('Mean-Variance Relationship (Linear Scale)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Right: Log-log scale with power law fit
log_means = np.log(window_means)
log_vars = np.log(window_vars)

axes[1].scatter(log_means, log_vars, alpha=0.6, s=80, color='steelblue', edgecolor='black', linewidth=1)

# Fit power law
slope_mv, intercept_mv, r_mv, p_mv, se_mv = stats.linregress(log_means, log_vars)
fit_line = slope_mv * log_means + intercept_mv
axes[1].plot(log_means, fit_line, 'r-', linewidth=2,
             label=f'Power law: Var = {np.exp(intercept_mv):.3f} × Mean^{slope_mv:.2f}\nR² = {r_mv**2:.3f}')

# Add reference lines
axes[1].plot([log_means.min(), log_means.max()],
             [log_means.min(), log_means.max()],
             'orange', linestyle='--', linewidth=2, alpha=0.5, label='Slope = 1 (Poisson)')
axes[1].plot([log_means.min(), log_means.max()],
             [2*log_means.min(), 2*log_means.max()],
             'green', linestyle='--', linewidth=2, alpha=0.5, label='Slope = 2 (Quadratic)')

axes[1].set_xlabel('log(Window Mean)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('log(Window Variance)', fontsize=12, fontweight='bold')
axes[1].set_title('Mean-Variance Relationship (Log-Log Scale)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/06_mean_variance_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 06_mean_variance_relationship.png")

# ============================================================================
# PLOT 7: Model comparison (Linear vs Log-linear)
# ============================================================================

# Log-linear model
log_C = np.log(C)
slope_log, intercept_log, r_log, p_log, se_log = stats.linregress(year, log_C)
predicted_log = slope_log * year + intercept_log
predicted_original = np.exp(predicted_log)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Linear model fit
axes[0, 0].scatter(year, C, alpha=0.6, s=70, color='steelblue', edgecolor='black', linewidth=0.5, label='Observed')
axes[0, 0].plot(year, predicted, 'r-', linewidth=2, label=f'Linear fit (R²={r_value**2:.4f})')
axes[0, 0].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Count (C)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Linear Model Fit', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Panel 2: Log-linear model fit
axes[0, 1].scatter(year, C, alpha=0.6, s=70, color='steelblue', edgecolor='black', linewidth=0.5, label='Observed')
axes[0, 1].plot(year, predicted_original, 'g-', linewidth=2, label=f'Log-linear fit (R²={r_log**2:.4f})')
axes[0, 1].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Count (C)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Log-Linear Model Fit', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Panel 3: Residuals from linear model
residuals_linear = C - predicted
axes[1, 0].scatter(year, residuals_linear, alpha=0.6, s=70, color='steelblue', edgecolor='black', linewidth=0.5)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 0].set_title(f'Linear Model Residuals (SD={residuals_linear.std(ddof=1):.2f})', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Panel 4: Residuals from log-linear model
residuals_log_scale = C - predicted_original
axes[1, 1].scatter(year, residuals_log_scale, alpha=0.6, s=70, color='green', edgecolor='black', linewidth=0.5)
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 1].set_title(f'Log-Linear Model Residuals (SD={residuals_log_scale.std(ddof=1):.2f})', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/07_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 07_model_comparison.png")

# ============================================================================
# PLOT 8: Variance structure across time
# ============================================================================

# Split data into thirds
n_per_group = len(C) // 3
early = C[:n_per_group]
middle = C[n_per_group:2*n_per_group]
late = C[2*n_per_group:]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histograms for each period
for ax, data_subset, label, color in zip(axes,
                                          [early, middle, late],
                                          ['Early Period', 'Middle Period', 'Late Period'],
                                          ['skyblue', 'lightgreen', 'salmon']):
    ax.hist(data_subset, bins=8, edgecolor='black', alpha=0.7, color=color)
    ax.axvline(data_subset.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data_subset.mean():.1f}')
    ax.set_xlabel('Count (C)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{label}\nMean={data_subset.mean():.1f}, SD={data_subset.std(ddof=1):.1f}',
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add CV
    cv = data_subset.std(ddof=1) / data_subset.mean()
    ax.text(0.05, 0.95, f'CV: {cv:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle('Distribution Changes Over Time', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/08_temporal_distribution_changes.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 08_temporal_distribution_changes.png")

print("\n" + "="*80)
print("All Round 2 visualizations complete!")
print("="*80)
