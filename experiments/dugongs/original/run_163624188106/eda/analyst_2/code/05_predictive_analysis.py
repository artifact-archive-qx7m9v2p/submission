"""
Predictive Implications and Small Sample Considerations
Focus: Cross-validation, overfitting risk, uncertainty quantification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("=" * 80)
print("PREDICTIVE IMPLICATIONS & SMALL SAMPLE CONSIDERATIONS")
print("=" * 80)

X = data['x'].values
y = data['Y'].values
n = len(y)

print(f"\nSample size: n = {n}")
print(f"Predictor range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Response range: [{y.min():.2f}, {y.max():.2f}]")

# 1. Leave-One-Out Cross-Validation (LOO-CV) for different models
print("\n" + "=" * 80)
print("1. LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 80)

def loo_cv_linear(x_vals, y_vals):
    """LOO-CV for linear model"""
    n = len(y_vals)
    predictions = np.zeros(n)

    for i in range(n):
        # Leave out point i
        x_train = np.delete(x_vals, i)
        y_train = np.delete(y_vals, i)

        # Fit model
        slope, intercept, _, _, _ = stats.linregress(x_train, y_train)

        # Predict left-out point
        predictions[i] = intercept + slope * x_vals[i]

    return predictions

def loo_cv_log(x_vals, y_vals):
    """LOO-CV for log model"""
    n = len(y_vals)
    predictions = np.zeros(n)

    for i in range(n):
        x_train = np.delete(x_vals, i)
        y_train = np.delete(y_vals, i)

        slope, intercept, _, _, _ = stats.linregress(np.log(x_train), y_train)
        predictions[i] = intercept + slope * np.log(x_vals[i])

    return predictions

def loo_cv_poly(x_vals, y_vals, degree):
    """LOO-CV for polynomial model"""
    n = len(y_vals)
    predictions = np.zeros(n)

    for i in range(n):
        x_train = np.delete(x_vals, i)
        y_train = np.delete(y_vals, i)

        coeffs = np.polyfit(x_train, y_train, degree)
        poly_func = np.poly1d(coeffs)
        predictions[i] = poly_func(x_vals[i])

    return predictions

# Test different models
models = {
    'Linear': loo_cv_linear(X, y),
    'Logarithmic': loo_cv_log(X, y),
    'Poly-2': loo_cv_poly(X, y, 2),
    'Poly-3': loo_cv_poly(X, y, 3),
    'Poly-4': loo_cv_poly(X, y, 4),
    'Poly-5': loo_cv_poly(X, y, 5)
}

loo_results = []
for name, predictions in models.items():
    mse = np.mean((y - predictions)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - predictions))
    r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)

    loo_results.append({
        'Model': name,
        'LOO-RMSE': rmse,
        'LOO-MAE': mae,
        'LOO-R2': r2
    })

    print(f"{name:15s}: LOO-RMSE={rmse:.4f}, LOO-MAE={mae:.4f}, LOO-R²={r2:.4f}")

loo_df = pd.DataFrame(loo_results)

# 2. Bootstrap uncertainty estimation
print("\n" + "=" * 80)
print("2. BOOTSTRAP UNCERTAINTY ESTIMATION (1000 iterations)")
print("=" * 80)

n_bootstrap = 1000
bootstrap_slopes = []
bootstrap_intercepts = []
bootstrap_r2 = []

np.random.seed(42)

for _ in range(n_bootstrap):
    # Resample with replacement
    indices = np.random.choice(n, size=n, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]

    # Fit linear model
    slope, intercept, r_value, _, _ = stats.linregress(X_boot, y_boot)

    bootstrap_slopes.append(slope)
    bootstrap_intercepts.append(intercept)
    bootstrap_r2.append(r_value**2)

bootstrap_slopes = np.array(bootstrap_slopes)
bootstrap_intercepts = np.array(bootstrap_intercepts)
bootstrap_r2 = np.array(bootstrap_r2)

print("\nLinear Model Bootstrap Statistics:")
print(f"Slope: {np.mean(bootstrap_slopes):.6f} ± {np.std(bootstrap_slopes):.6f}")
print(f"  95% CI: [{np.percentile(bootstrap_slopes, 2.5):.6f}, {np.percentile(bootstrap_slopes, 97.5):.6f}]")
print(f"Intercept: {np.mean(bootstrap_intercepts):.6f} ± {np.std(bootstrap_intercepts):.6f}")
print(f"  95% CI: [{np.percentile(bootstrap_intercepts, 2.5):.6f}, {np.percentile(bootstrap_intercepts, 97.5):.6f}]")
print(f"R²: {np.mean(bootstrap_r2):.4f} ± {np.std(bootstrap_r2):.4f}")
print(f"  95% CI: [{np.percentile(bootstrap_r2, 2.5):.4f}, {np.percentile(bootstrap_r2, 97.5):.4f}]")

# 3. Prediction intervals
print("\n" + "=" * 80)
print("3. PREDICTION INTERVALS")
print("=" * 80)

# Fit linear model
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
y_pred = intercept + slope * X
residuals = y - y_pred
residual_std = np.std(residuals, ddof=2)

# Calculate prediction interval for new observations
x_new = np.linspace(X.min(), X.max(), 100)
y_new = intercept + slope * x_new

# Standard error of prediction
# SE_pred = s * sqrt(1 + 1/n + (x - x_mean)^2 / sum((x - x_mean)^2))
x_mean = np.mean(X)
se_pred = residual_std * np.sqrt(1 + 1/n + (x_new - x_mean)**2 / np.sum((X - x_mean)**2))

# 95% prediction interval
t_val = stats.t.ppf(0.975, n - 2)
pred_interval_lower = y_new - t_val * se_pred
pred_interval_upper = y_new + t_val * se_pred

# Confidence interval for the mean
se_mean = residual_std * np.sqrt(1/n + (x_new - x_mean)**2 / np.sum((X - x_mean)**2))
conf_interval_lower = y_new - t_val * se_mean
conf_interval_upper = y_new + t_val * se_mean

print(f"95% Prediction Interval width (mean): {np.mean(pred_interval_upper - pred_interval_lower):.4f}")
print(f"95% Confidence Interval width (mean): {np.mean(conf_interval_upper - conf_interval_lower):.4f}")

# 4. Sample size adequacy analysis
print("\n" + "=" * 80)
print("4. SAMPLE SIZE ADEQUACY")
print("=" * 80)

# Rule of thumb: need at least 10-20 observations per parameter
print(f"Observations per parameter for linear model: {n/2:.1f}")
print(f"Observations per parameter for quadratic model: {n/3:.1f}")
print(f"Observations per parameter for cubic model: {n/4:.1f}")

# Check leverage points (high influence)
X_matrix = np.column_stack([np.ones(n), X])
hat_matrix = X_matrix @ np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T
leverage = np.diag(hat_matrix)
high_leverage = leverage > 2 * (2 / n)  # Threshold: 2*p/n where p=2

print(f"\nHigh leverage points: {np.sum(high_leverage)} out of {n}")
if np.sum(high_leverage) > 0:
    high_lev_idx = np.where(high_leverage)[0]
    print("Indices:", high_lev_idx)
    for idx in high_lev_idx:
        print(f"  Point {idx}: x={X[idx]:.2f}, Y={y[idx]:.2f}, leverage={leverage[idx]:.3f}")

# 5. Extrapolation risk assessment
print("\n" + "=" * 80)
print("5. EXTRAPOLATION RISK")
print("=" * 80)

x_range = X.max() - X.min()
print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Range width: {x_range:.2f}")
print(f"Data density: {n / x_range:.2f} observations per unit")

# Check spacing of x values
x_sorted_unique = np.sort(np.unique(X))
gaps = np.diff(x_sorted_unique)
print(f"\nGaps in x values:")
print(f"  Mean gap: {np.mean(gaps):.2f}")
print(f"  Max gap: {np.max(gaps):.2f} (at x={x_sorted_unique[np.argmax(gaps)]:.2f})")
print(f"  Largest gaps (>5):")
large_gaps = np.where(gaps > 5)[0]
for idx in large_gaps:
    print(f"    Between x={x_sorted_unique[idx]:.2f} and x={x_sorted_unique[idx+1]:.2f}: gap={gaps[idx]:.2f}")

# Create visualizations
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. LOO-CV comparison
ax1 = fig.add_subplot(gs[0, :])
for name, predictions in models.items():
    ax1.scatter(predictions, y, alpha=0.6, s=50, label=name)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('LOO-CV Predictions')
ax1.set_ylabel('Actual Y')
ax1.set_title('Leave-One-Out Cross-Validation: Predicted vs Actual')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Bootstrap distributions
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(bootstrap_slopes, bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(bootstrap_slopes), color='r', linestyle='--', linewidth=2, label='Mean')
ax2.axvline(np.percentile(bootstrap_slopes, 2.5), color='orange', linestyle=':', linewidth=2)
ax2.axvline(np.percentile(bootstrap_slopes, 97.5), color='orange', linestyle=':', linewidth=2, label='95% CI')
ax2.set_xlabel('Slope')
ax2.set_ylabel('Frequency')
ax2.set_title('Bootstrap Distribution of Slope')
ax2.legend()

ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(bootstrap_intercepts, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(np.mean(bootstrap_intercepts), color='r', linestyle='--', linewidth=2, label='Mean')
ax3.axvline(np.percentile(bootstrap_intercepts, 2.5), color='orange', linestyle=':', linewidth=2)
ax3.axvline(np.percentile(bootstrap_intercepts, 97.5), color='orange', linestyle=':', linewidth=2, label='95% CI')
ax3.set_xlabel('Intercept')
ax3.set_ylabel('Frequency')
ax3.set_title('Bootstrap Distribution of Intercept')
ax3.legend()

ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(bootstrap_r2, bins=30, edgecolor='black', alpha=0.7)
ax4.axvline(np.mean(bootstrap_r2), color='r', linestyle='--', linewidth=2, label='Mean')
ax4.set_xlabel('R²')
ax4.set_ylabel('Frequency')
ax4.set_title('Bootstrap Distribution of R²')
ax4.legend()

# 3. Prediction intervals
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(X, y, alpha=0.6, s=50, label='Data', zorder=5)
ax5.plot(x_new, y_new, 'r-', linewidth=2, label='Fitted line')
ax5.fill_between(x_new, conf_interval_lower, conf_interval_upper, alpha=0.3,
                  color='orange', label='95% Confidence interval')
ax5.fill_between(x_new, pred_interval_lower, pred_interval_upper, alpha=0.2,
                  color='blue', label='95% Prediction interval')
ax5.set_xlabel('x')
ax5.set_ylabel('Y')
ax5.set_title('Prediction & Confidence Intervals')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 4. Leverage plot
ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(range(n), leverage, alpha=0.6, s=50)
ax6.axhline(2 * (2 / n), color='r', linestyle='--', linewidth=2, label='Leverage threshold')
ax6.set_xlabel('Observation Index')
ax6.set_ylabel('Leverage')
ax6.set_title('Leverage Plot (Influential Points)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 5. Data coverage
ax7 = fig.add_subplot(gs[2, 2])
ax7.scatter(X, np.ones_like(X), alpha=0.6, s=100, marker='|')
ax7.set_xlabel('x')
ax7.set_yticks([])
ax7.set_title('Data Coverage in x Space')
ax7.grid(True, alpha=0.3, axis='x')

# Add rugplot for better visibility
for xi in X:
    ax7.axvline(xi, ymin=0.45, ymax=0.55, color='blue', alpha=0.3, linewidth=2)

plt.savefig('/workspace/eda/analyst_2/visualizations/06_predictive_analysis.png', dpi=300, bbox_inches='tight')
print("\n\nSaved: /workspace/eda/analyst_2/visualizations/06_predictive_analysis.png")
plt.close()

# Additional visualization: LOO-CV residuals
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, predictions) in enumerate(models.items()):
    residuals_loo = y - predictions

    axes[idx].scatter(predictions, residuals_loo, alpha=0.6, s=50)
    axes[idx].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('Predicted Y')
    axes[idx].set_ylabel('LOO-CV Residuals')
    axes[idx].set_title(f'{name}: LOO-RMSE={np.sqrt(np.mean(residuals_loo**2)):.4f}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/07_loo_cv_residuals.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/analyst_2/visualizations/07_loo_cv_residuals.png")
plt.close()

# Save summary results
summary = {
    'sample_size': n,
    'x_range': [X.min(), X.max()],
    'y_range': [y.min(), y.max()],
    'bootstrap_slope_mean': np.mean(bootstrap_slopes),
    'bootstrap_slope_std': np.std(bootstrap_slopes),
    'bootstrap_slope_ci_lower': np.percentile(bootstrap_slopes, 2.5),
    'bootstrap_slope_ci_upper': np.percentile(bootstrap_slopes, 97.5),
    'high_leverage_points': int(np.sum(high_leverage)),
    'max_gap_in_x': float(np.max(gaps))
}

import json
with open('/workspace/eda/analyst_2/code/predictive_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary to: /workspace/eda/analyst_2/code/predictive_summary.json")

print("\n" + "=" * 80)
print("PREDICTIVE ANALYSIS COMPLETE")
print("=" * 80)
