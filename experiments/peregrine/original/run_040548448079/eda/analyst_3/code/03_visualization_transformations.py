"""
Comprehensive visualizations for transformation analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
df = pd.read_csv('/workspace/data/data_analyst_3.csv')
df_sorted = df.sort_values('year').reset_index(drop=True)
X = df_sorted['year'].values
y = df_sorted['C'].values

# Box-Cox optimal lambda
_, lambda_optimal = stats.boxcox(y)

def box_cox_transform(data, lam):
    """Apply Box-Cox transformation"""
    if lam == 0:
        return np.log(data)
    else:
        return (data**lam - 1) / lam

# ============================================================
# FIGURE 1: Comparison of transformations (2x3 grid)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparison of Transformations: C vs Year', fontsize=16, fontweight='bold')

transformations = [
    ('Original Scale', y, 'C'),
    ('Log Transform', np.log(y), 'log(C)'),
    ('Square Root Transform', np.sqrt(y), 'sqrt(C)'),
    ('Box-Cox (λ={:.3f})'.format(lambda_optimal), box_cox_transform(y, lambda_optimal), 'Box-Cox(C)'),
    ('Inverse Transform', 1/y, '1/C'),
    ('Square Transform', y**2, 'C²'),
]

for idx, (ax, (title, y_trans, ylabel)) in enumerate(zip(axes.flat, transformations)):
    # Scatter plot
    ax.scatter(X, y_trans, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

    # Fit linear model
    slope, intercept = np.polyfit(X, y_trans, 1)
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = slope * X_line + intercept
    ax.plot(X_line, y_line, 'r-', linewidth=2, label='Linear fit')

    # Calculate R²
    y_pred = slope * X + intercept
    r2 = 1 - np.sum((y_trans - y_pred)**2) / np.sum((y_trans - np.mean(y_trans))**2)

    ax.set_xlabel('Year (standardized)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/01_transformation_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 01_transformation_comparison.png")

# ============================================================
# FIGURE 2: Residual diagnostics for key transformations
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Residual Diagnostics for Different Transformations', fontsize=16, fontweight='bold')

key_transformations = [
    ('Original', y),
    ('Log', np.log(y)),
    ('Box-Cox', box_cox_transform(y, lambda_optimal)),
]

for row, (name, y_trans) in enumerate(key_transformations):
    # Fit linear model
    slope, intercept = np.polyfit(X, y_trans, 1)
    y_pred = slope * X + intercept
    residuals = y_trans - y_pred

    # Plot 1: Residuals vs fitted
    axes[row, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[row, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[row, 0].set_xlabel('Fitted values', fontsize=10)
    axes[row, 0].set_ylabel('Residuals', fontsize=10)
    axes[row, 0].set_title(f'{name}: Residuals vs Fitted', fontsize=11, fontweight='bold')
    axes[row, 0].grid(True, alpha=0.3)

    # Plot 2: Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[row, 1])
    axes[row, 1].set_title(f'{name}: Q-Q Plot', fontsize=11, fontweight='bold')
    axes[row, 1].grid(True, alpha=0.3)

    # Plot 3: Histogram of residuals
    axes[row, 2].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
    axes[row, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[row, 2].set_xlabel('Residuals', fontsize=10)
    axes[row, 2].set_ylabel('Frequency', fontsize=10)

    # Add Shapiro-Wilk test result
    _, p_value = stats.shapiro(residuals)
    axes[row, 2].set_title(f'{name}: Histogram\nShapiro-Wilk p={p_value:.4f}',
                           fontsize=11, fontweight='bold')
    axes[row, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/02_residual_diagnostics.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 02_residual_diagnostics.png")

# ============================================================
# FIGURE 3: Variance stabilization analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Variance Stabilization Analysis', fontsize=16, fontweight='bold')

transformations_var = [
    ('Original Scale', y, 'C'),
    ('Log Transform', np.log(y), 'log(C)'),
    ('Square Root', np.sqrt(y), 'sqrt(C)'),
    ('Box-Cox', box_cox_transform(y, lambda_optimal), 'Box-Cox(C)'),
]

for idx, (ax, (title, y_trans, ylabel)) in enumerate(zip(axes.flat, transformations_var)):
    # Fit linear model
    slope, intercept = np.polyfit(X, y_trans, 1)
    y_pred = slope * X + intercept
    residuals = y_trans - y_pred

    # Plot residuals vs fitted (to check heteroscedasticity)
    ax.scatter(y_pred, np.abs(residuals), alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

    # Add LOWESS smooth to show trend
    from scipy.signal import savgol_filter
    sort_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sort_idx]
    abs_resid_sorted = np.abs(residuals)[sort_idx]

    # Simple moving average for trend
    window = min(11, len(y_pred)//3)
    if window % 2 == 0:
        window += 1
    if window >= 5:
        smooth = savgol_filter(abs_resid_sorted, window, 3)
        ax.plot(y_pred_sorted, smooth, 'r-', linewidth=2, label='Trend')

    # Calculate variance in thirds
    n = len(y_trans)
    third = n // 3
    var_low = np.var(residuals[:third])
    var_mid = np.var(residuals[third:2*third])
    var_high = np.var(residuals[2*third:])
    ratio = var_high / var_low if var_low > 0 else np.inf

    ax.set_xlabel('Fitted values', fontsize=10)
    ax.set_ylabel('Absolute residuals', fontsize=10)
    ax.set_title(f'{title}\nVar ratio (high/low): {ratio:.2f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if window >= 5:
        ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/03_variance_stabilization.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 03_variance_stabilization.png")

# ============================================================
# FIGURE 4: Polynomial vs Exponential comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Polynomial vs Exponential Models', fontsize=16, fontweight='bold')

# Top left: Linear
ax = axes[0, 0]
coeffs_1 = np.polyfit(X, y, 1)
y_pred_1 = np.polyval(coeffs_1, X)
r2_1 = 1 - np.sum((y - y_pred_1)**2) / np.sum((y - np.mean(y))**2)

ax.scatter(X, y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5, label='Data')
X_line = np.linspace(X.min(), X.max(), 100)
ax.plot(X_line, np.polyval(coeffs_1, X_line), 'r-', linewidth=2, label='Linear')
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('C', fontsize=10)
ax.set_title(f'Linear Model\nR² = {r2_1:.4f}', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Top right: Quadratic
ax = axes[0, 1]
coeffs_2 = np.polyfit(X, y, 2)
y_pred_2 = np.polyval(coeffs_2, X)
r2_2 = 1 - np.sum((y - y_pred_2)**2) / np.sum((y - np.mean(y))**2)

ax.scatter(X, y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5, label='Data')
ax.plot(X_line, np.polyval(coeffs_2, X_line), 'g-', linewidth=2, label='Quadratic')
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('C', fontsize=10)
ax.set_title(f'Quadratic Model\nR² = {r2_2:.4f}', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom left: Cubic
ax = axes[1, 0]
coeffs_3 = np.polyfit(X, y, 3)
y_pred_3 = np.polyval(coeffs_3, X)
r2_3 = 1 - np.sum((y - y_pred_3)**2) / np.sum((y - np.mean(y))**2)

ax.scatter(X, y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5, label='Data')
ax.plot(X_line, np.polyval(coeffs_3, X_line), 'b-', linewidth=2, label='Cubic')
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('C', fontsize=10)
ax.set_title(f'Cubic Model\nR² = {r2_3:.4f}', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom right: Exponential
ax = axes[1, 1]
log_y = np.log(y)
slope_exp, intercept_exp = np.polyfit(X, log_y, 1)
y_pred_exp = np.exp(slope_exp * X + intercept_exp)
r2_exp = 1 - np.sum((y - y_pred_exp)**2) / np.sum((y - np.mean(y))**2)

ax.scatter(X, y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5, label='Data')
ax.plot(X_line, np.exp(slope_exp * X_line + intercept_exp), 'm-', linewidth=2, label='Exponential')
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('C', fontsize=10)
ax.set_title(f'Exponential Model\nR² = {r2_exp:.4f}', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/04_polynomial_vs_exponential.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 04_polynomial_vs_exponential.png")

# ============================================================
# FIGURE 5: Model comparison with all fits overlaid
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.scatter(X, y, alpha=0.7, s=80, edgecolors='black', linewidths=1,
           label='Data', zorder=5, color='black')

X_line = np.linspace(X.min(), X.max(), 200)

# Plot all models
ax.plot(X_line, np.polyval(coeffs_1, X_line), '--', linewidth=2,
        label=f'Linear (R²={r2_1:.3f})', alpha=0.8)
ax.plot(X_line, np.polyval(coeffs_2, X_line), '-', linewidth=2,
        label=f'Quadratic (R²={r2_2:.3f})', alpha=0.8)
ax.plot(X_line, np.polyval(coeffs_3, X_line), '-', linewidth=2,
        label=f'Cubic (R²={r2_3:.3f})', alpha=0.8)
ax.plot(X_line, np.exp(slope_exp * X_line + intercept_exp), '-', linewidth=2,
        label=f'Exponential (R²={r2_exp:.3f})', alpha=0.8)

ax.set_xlabel('Year (standardized)', fontsize=12, fontweight='bold')
ax.set_ylabel('C (count)', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Model Fits', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/05_all_models_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 05_all_models_comparison.png")

print("\nAll visualizations created successfully!")
