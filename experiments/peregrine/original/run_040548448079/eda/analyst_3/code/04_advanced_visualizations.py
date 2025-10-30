"""
Advanced visualizations for transformation optimization and feature engineering
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
n = len(y)

def box_cox_transform(data, lam):
    """Apply Box-Cox transformation"""
    if abs(lam) < 1e-10:
        return np.log(data)
    else:
        return (data**lam - 1) / lam

# ============================================================
# FIGURE 6: Box-Cox lambda optimization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Box-Cox Transformation Optimization', fontsize=16, fontweight='bold')

# Test range of lambda values
lambdas = np.linspace(-2, 2, 100)
correlations = []
variance_ratios = []
shapiro_ps = []

for lam in lambdas:
    # Transform
    y_trans = box_cox_transform(y, lam)

    # Correlation with year
    corr = np.abs(np.corrcoef(X, y_trans)[0, 1])
    correlations.append(corr)

    # Variance ratio
    third = n // 3
    var_low = np.var(y_trans[:third])
    var_high = np.var(y_trans[2*third:])
    ratio = var_high / var_low if var_low > 0 else np.inf
    variance_ratios.append(ratio)

    # Normality of residuals
    slope, intercept = np.polyfit(X, y_trans, 1)
    residuals = y_trans - (slope * X + intercept)
    _, p_value = stats.shapiro(residuals)
    shapiro_ps.append(p_value)

# Plot 1: Correlation vs lambda
axes[0].plot(lambdas, correlations, 'b-', linewidth=2)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='λ=0 (log)')
best_corr_idx = np.argmax(correlations)
axes[0].axvline(x=lambdas[best_corr_idx], color='r', linestyle='--',
                label=f'Max corr: λ={lambdas[best_corr_idx]:.3f}')
axes[0].set_xlabel('Lambda (λ)', fontsize=11)
axes[0].set_ylabel('Absolute correlation with year', fontsize=11)
axes[0].set_title('Linearity Criterion', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Plot 2: Variance ratio vs lambda
axes[1].plot(lambdas, variance_ratios, 'g-', linewidth=2)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='λ=0 (log)')
axes[1].axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Perfect stabilization')
best_var_idx = np.argmin([abs(r - 1) for r in variance_ratios])
axes[1].axvline(x=lambdas[best_var_idx], color='r', linestyle='--',
                label=f'Best stab: λ={lambdas[best_var_idx]:.3f}')
axes[1].set_xlabel('Lambda (λ)', fontsize=11)
axes[1].set_ylabel('Variance ratio (high/low)', fontsize=11)
axes[1].set_title('Variance Stabilization Criterion', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, min(10, max(variance_ratios)))

# Plot 3: Normality vs lambda
axes[2].plot(lambdas, shapiro_ps, 'm-', linewidth=2)
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='λ=0 (log)')
axes[2].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='p=0.05 threshold')
best_norm_idx = np.argmax(shapiro_ps)
axes[2].axvline(x=lambdas[best_norm_idx], color='r', linestyle='--',
                label=f'Max p: λ={lambdas[best_norm_idx]:.3f}')
axes[2].set_xlabel('Lambda (λ)', fontsize=11)
axes[2].set_ylabel('Shapiro-Wilk p-value', fontsize=11)
axes[2].set_title('Normality of Residuals Criterion', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/06_boxcox_optimization.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 06_boxcox_optimization.png")

# ============================================================
# FIGURE 7: Feature correlation matrix
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create derived features
feature_df = pd.DataFrame({
    'C': y,
    'year': X,
    'year²': X**2,
    'year³': X**3,
    'exp(0.5·year)': np.exp(0.5*X),
    'exp(year)': np.exp(X),
    'log(C)': np.log(y),
    'sqrt(C)': np.sqrt(y),
})

# Compute correlation matrix
corr_matrix = feature_df.corr()

# Create heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

ax.set_title('Correlation Matrix: Original and Derived Features',
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/07_feature_correlation_matrix.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 07_feature_correlation_matrix.png")

# ============================================================
# FIGURE 8: AIC/BIC comparison across model types
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Selection Criteria', fontsize=16, fontweight='bold')

# Polynomial models
degrees = range(1, 6)
aic_poly = []
bic_poly = []
r2_poly = []

for degree in degrees:
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    k = degree + 1
    aic = n * np.log(ss_res/n) + 2*k
    bic = n * np.log(ss_res/n) + k*np.log(n)
    r2 = 1 - (ss_res / ss_tot)

    aic_poly.append(aic)
    bic_poly.append(bic)
    r2_poly.append(r2)

# Exponential model
log_y = np.log(y)
slope_exp, intercept_exp = np.polyfit(X, log_y, 1)
y_pred_exp = np.exp(slope_exp * X + intercept_exp)
ss_res_exp = np.sum((y - y_pred_exp)**2)
ss_tot = np.sum((y - np.mean(y))**2)
k_exp = 2
aic_exp = n * np.log(ss_res_exp/n) + 2*k_exp
bic_exp = n * np.log(ss_res_exp/n) + k_exp*np.log(n)
r2_exp = 1 - (ss_res_exp / ss_tot)

# Plot AIC
ax = axes[0]
ax.plot(degrees, aic_poly, 'o-', linewidth=2, markersize=8, label='Polynomial')
ax.axhline(y=aic_exp, color='r', linestyle='--', linewidth=2, label='Exponential')
ax.set_xlabel('Polynomial Degree', fontsize=12)
ax.set_ylabel('AIC', fontsize=12)
ax.set_title('Akaike Information Criterion\n(Lower is better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(degrees)

# Plot BIC
ax = axes[1]
ax.plot(degrees, bic_poly, 'o-', linewidth=2, markersize=8, label='Polynomial')
ax.axhline(y=bic_exp, color='r', linestyle='--', linewidth=2, label='Exponential')
ax.set_xlabel('Polynomial Degree', fontsize=12)
ax.set_ylabel('BIC', fontsize=12)
ax.set_title('Bayesian Information Criterion\n(Lower is better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(degrees)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/08_model_selection_criteria.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 08_model_selection_criteria.png")

# ============================================================
# FIGURE 9: Prediction intervals for different models
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Fits with Residual Analysis', fontsize=16, fontweight='bold')

models = [
    ('Linear', 1),
    ('Quadratic', 2),
    ('Cubic', 3),
    ('Quartic', 4),
]

for idx, (ax, (name, degree)) in enumerate(zip(axes.flat, models)):
    # Fit model
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    residuals = y - y_pred

    # Calculate standard error
    se = np.std(residuals)

    # Plot
    ax.scatter(X, y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5,
               label='Data', zorder=5, color='black')

    X_line = np.linspace(X.min(), X.max(), 200)
    y_line = np.polyval(coeffs, X_line)

    ax.plot(X_line, y_line, 'r-', linewidth=2, label='Fit')
    ax.fill_between(X_line, y_line - 2*se, y_line + 2*se,
                     alpha=0.2, color='red', label='±2 SE')

    # Calculate metrics
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    rmse = np.sqrt(np.mean(residuals**2))

    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('C', fontsize=10)
    ax.set_title(f'{name} Model\nR²={r2:.4f}, RMSE={rmse:.2f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/09_model_fits_with_intervals.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 09_model_fits_with_intervals.png")

# ============================================================
# FIGURE 10: Scale-location plots for variance assessment
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Scale-Location Plots: Assessing Homoscedasticity', fontsize=16, fontweight='bold')

transformations_scale = [
    ('Original Scale', y),
    ('Log Transform', np.log(y)),
    ('Square Root', np.sqrt(y)),
    ('Box-Cox', box_cox_transform(y, stats.boxcox(y)[1])),
]

for idx, (ax, (name, y_trans)) in enumerate(zip(axes.flat, transformations_scale)):
    # Fit model
    slope, intercept = np.polyfit(X, y_trans, 1)
    y_pred = slope * X + intercept
    residuals = y_trans - y_pred

    # Standardized residuals
    std_residuals = residuals / np.std(residuals)

    # Scale-location plot: sqrt(|standardized residuals|) vs fitted
    ax.scatter(y_pred, np.sqrt(np.abs(std_residuals)),
               alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

    # Add smooth trend
    from scipy.signal import savgol_filter
    sort_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sort_idx]
    sqrt_std_resid_sorted = np.sqrt(np.abs(std_residuals))[sort_idx]

    window = min(11, len(y_pred)//3)
    if window % 2 == 0:
        window += 1
    if window >= 5:
        smooth = savgol_filter(sqrt_std_resid_sorted, window, 3)
        ax.plot(y_pred_sorted, smooth, 'r-', linewidth=2, label='Trend')

    ax.set_xlabel('Fitted values', fontsize=10)
    ax.set_ylabel('√|Standardized residuals|', fontsize=10)
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if window >= 5:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/10_scale_location_plots.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Created: 10_scale_location_plots.png")

print("\nAll advanced visualizations created successfully!")
