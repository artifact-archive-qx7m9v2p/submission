"""
Final Model Comparison and Summary
Focus: Comprehensive comparison across all candidate models
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
with open('/workspace/data/data_analyst_3.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'year': data['year'], 'C': data['C']})
X = df['year'].values
y = df['C'].values
n = len(y)

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
print("="*80)

# Refit all models and calculate information criteria
models = {}

# 1. Linear
slope_lr, intercept_lr, _, _, _ = stats.linregress(X, y)
y_pred_lr = intercept_lr + slope_lr * X
resid_lr = y - y_pred_lr
rss_lr = np.sum(resid_lr**2)
sigma2_lr = rss_lr / (n - 2)
loglik_lr = -0.5 * n * np.log(2 * np.pi * sigma2_lr) - 0.5 * rss_lr / sigma2_lr
aic_lr = -2 * loglik_lr + 2 * 2
bic_lr = -2 * loglik_lr + 2 * np.log(n)

models['Linear'] = {
    'params': 2, 'r2': 1 - rss_lr / np.sum((y - y.mean())**2),
    'aic': aic_lr, 'bic': bic_lr, 'predictions': y_pred_lr
}

# 2. Quadratic
coeffs_quad = np.polyfit(X, y, 2)
y_pred_quad = np.polyval(coeffs_quad, X)
resid_quad = y - y_pred_quad
rss_quad = np.sum(resid_quad**2)
sigma2_quad = rss_quad / (n - 3)
loglik_quad = -0.5 * n * np.log(2 * np.pi * sigma2_quad) - 0.5 * rss_quad / sigma2_quad
aic_quad = -2 * loglik_quad + 2 * 3
bic_quad = -2 * loglik_quad + 3 * np.log(n)

models['Quadratic'] = {
    'params': 3, 'r2': 1 - rss_quad / np.sum((y - y.mean())**2),
    'aic': aic_quad, 'bic': bic_quad, 'predictions': y_pred_quad
}

# 3. Cubic
coeffs_cubic = np.polyfit(X, y, 3)
y_pred_cubic = np.polyval(coeffs_cubic, X)
resid_cubic = y - y_pred_cubic
rss_cubic = np.sum(resid_cubic**2)
sigma2_cubic = rss_cubic / (n - 4)
loglik_cubic = -0.5 * n * np.log(2 * np.pi * sigma2_cubic) - 0.5 * rss_cubic / sigma2_cubic
aic_cubic = -2 * loglik_cubic + 2 * 4
bic_cubic = -2 * loglik_cubic + 4 * np.log(n)

models['Cubic'] = {
    'params': 4, 'r2': 1 - rss_cubic / np.sum((y - y.mean())**2),
    'aic': aic_cubic, 'bic': bic_cubic, 'predictions': y_pred_cubic
}

# 4. Log-Linear
slope_log, intercept_log, _, _, _ = stats.linregress(X, np.log(y))
log_y_pred = intercept_log + slope_log * X
y_pred_log = np.exp(log_y_pred)

# For log-linear, calculate AIC on log scale
resid_log_scale = np.log(y) - log_y_pred
rss_log_scale = np.sum(resid_log_scale**2)
sigma2_log = rss_log_scale / (n - 2)
loglik_log = -0.5 * n * np.log(2 * np.pi * sigma2_log) - 0.5 * rss_log_scale / sigma2_log
aic_log = -2 * loglik_log + 2 * 2
bic_log = -2 * loglik_log + 2 * np.log(n)

models['Log-Linear'] = {
    'params': 2, 'r2': 1 - np.sum((y - y_pred_log)**2) / np.sum((y - y.mean())**2),
    'aic': aic_log, 'bic': bic_log, 'predictions': y_pred_log
}

# 5. Poisson GLM (using approximation)
# For Poisson: log-likelihood = sum(y*log(λ) - λ - log(y!))
lambda_poisson = y_pred_log  # Use same predictions as log-linear
loglik_poisson = np.sum(y * np.log(lambda_poisson) - lambda_poisson -
                        np.array([np.sum(np.log(range(1, int(yi)+1))) if yi > 0 else 0 for yi in y]))
aic_poisson = -2 * loglik_poisson + 2 * 2
bic_poisson = -2 * loglik_poisson + 2 * np.log(n)

models['Poisson GLM'] = {
    'params': 2, 'r2': 1 - np.sum((y - lambda_poisson)**2) / np.sum((y - y.mean())**2),
    'aic': aic_poisson, 'bic': bic_poisson, 'predictions': lambda_poisson
}

# Model comparison table
print("\nMODEL COMPARISON TABLE")
print("-"*80)
print(f"{'Model':<20} {'Params':>8} {'R²':>10} {'AIC':>12} {'BIC':>12} {'Δ AIC':>10}")
print("-"*80)

# Find best AIC
best_aic = min([m['aic'] for m in models.values()])

for name, info in models.items():
    delta_aic = info['aic'] - best_aic
    print(f"{name:<20} {info['params']:>8} {info['r2']:>10.4f} {info['aic']:>12.2f} "
          f"{info['bic']:>12.2f} {delta_aic:>10.2f}")

print("-"*80)
print(f"Best model by AIC: {min(models.items(), key=lambda x: x[1]['aic'])[0]}")
print(f"Best model by BIC: {min(models.items(), key=lambda x: x[1]['bic'])[0]}")

# Calculate Akaike weights
print("\n" + "="*80)
print("AKAIKE WEIGHTS (Model Probabilities)")
print("="*80)

aic_values = np.array([m['aic'] for m in models.values()])
delta_aic = aic_values - np.min(aic_values)
akaike_weights = np.exp(-0.5 * delta_aic) / np.sum(np.exp(-0.5 * delta_aic))

for (name, info), weight in zip(models.items(), akaike_weights):
    print(f"{name:<20} {weight:>10.4f} ({weight*100:>6.2f}%)")

# Cross-validation: Leave-one-out prediction error
print("\n" + "="*80)
print("LEAVE-ONE-OUT CROSS-VALIDATION")
print("="*80)

def loo_cv_mse(X, y, degree):
    """Calculate LOO-CV MSE for polynomial regression"""
    errors = []
    for i in range(len(y)):
        X_train = np.delete(X, i)
        y_train = np.delete(y, i)
        X_test = X[i]
        y_test = y[i]

        if degree == 0:  # Log-linear
            slope, intercept, _, _, _ = stats.linregress(X_train, np.log(y_train))
            y_pred = np.exp(intercept + slope * X_test)
        else:
            coeffs = np.polyfit(X_train, y_train, degree)
            y_pred = np.polyval(coeffs, X_test)

        errors.append((y_test - y_pred)**2)

    return np.mean(errors)

loo_linear = loo_cv_mse(X, y, 1)
loo_quad = loo_cv_mse(X, y, 2)
loo_cubic = loo_cv_mse(X, y, 3)
loo_log = loo_cv_mse(X, y, 0)

print(f"{'Model':<20} {'LOO-CV MSE':>15}")
print("-"*40)
print(f"{'Linear':<20} {loo_linear:>15.2f}")
print(f"{'Quadratic':<20} {loo_quad:>15.2f}")
print(f"{'Cubic':<20} {loo_cubic:>15.2f}")
print(f"{'Log-Linear':<20} {loo_log:>15.2f}")

# Visual summary
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Model fits comparison
sort_idx = np.argsort(X)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

ax = axes[0, 0]
ax.plot(X_sorted, y_sorted, 'ko', alpha=0.4, markersize=6, label='Data')
for name in ['Linear', 'Quadratic', 'Log-Linear']:
    ax.plot(X_sorted, models[name]['predictions'][sort_idx], '-', linewidth=2, label=name, alpha=0.7)
ax.set_xlabel('Year (standardized)')
ax.set_ylabel('Count (C)')
ax.set_title('A. Model Fits Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# B. AIC comparison
ax = axes[0, 1]
model_names = list(models.keys())
aic_vals = [models[name]['aic'] for name in model_names]
colors = ['green' if aic == min(aic_vals) else 'steelblue' for aic in aic_vals]
ax.barh(model_names, aic_vals, color=colors, alpha=0.7)
ax.set_xlabel('AIC (lower is better)')
ax.set_title('B. AIC Comparison')
ax.grid(True, alpha=0.3, axis='x')

# C. Akaike weights
ax = axes[1, 0]
colors_weights = ['green' if w == max(akaike_weights) else 'steelblue' for w in akaike_weights]
ax.barh(model_names, akaike_weights, color=colors_weights, alpha=0.7)
ax.set_xlabel('Akaike Weight (model probability)')
ax.set_title('C. Akaike Weights')
ax.grid(True, alpha=0.3, axis='x')

# D. R² vs Complexity
ax = axes[1, 1]
params = [models[name]['params'] for name in model_names]
r2s = [models[name]['r2'] for name in model_names]
ax.scatter(params, r2s, s=150, alpha=0.7)
for i, name in enumerate(model_names):
    ax.annotate(name, (params[i], r2s[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel('Number of Parameters')
ax.set_ylabel('R²')
ax.set_title('D. R² vs Model Complexity')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/08_model_comparison_summary.png', dpi=150, bbox_inches='tight')
print("\n[SAVED] Model comparison summary: visualizations/08_model_comparison_summary.png")
plt.close()

# Prediction intervals visualization for best models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Quadratic with prediction intervals
ax = axes[0]
ax.plot(X_sorted, y_sorted, 'ko', alpha=0.5, markersize=6, label='Data')
ax.plot(X_sorted, models['Quadratic']['predictions'][sort_idx], 'r-', linewidth=2, label='Quadratic fit')

# Approximate 95% prediction interval
se_quad = np.sqrt(sigma2_quad)
pred_interval = 1.96 * se_quad * np.sqrt(1 + 1/n)
ax.fill_between(X_sorted,
                 models['Quadratic']['predictions'][sort_idx] - pred_interval,
                 models['Quadratic']['predictions'][sort_idx] + pred_interval,
                 alpha=0.2, color='red', label='95% Pred. Interval')

ax.set_xlabel('Year (standardized)')
ax.set_ylabel('Count (C)')
ax.set_title('A. Quadratic Model with Prediction Interval')
ax.legend()
ax.grid(True, alpha=0.3)

# Log-Linear with prediction intervals
ax = axes[1]
ax.plot(X_sorted, y_sorted, 'ko', alpha=0.5, markersize=6, label='Data')
ax.plot(X_sorted, models['Log-Linear']['predictions'][sort_idx], 'b-', linewidth=2, label='Log-Linear fit')

# Prediction interval on log scale, transformed back
se_log = np.sqrt(sigma2_log)
log_pred_lower = log_y_pred - 1.96 * se_log * np.sqrt(1 + 1/n)
log_pred_upper = log_y_pred + 1.96 * se_log * np.sqrt(1 + 1/n)
ax.fill_between(X_sorted,
                 np.exp(log_pred_lower[sort_idx]),
                 np.exp(log_pred_upper[sort_idx]),
                 alpha=0.2, color='blue', label='95% Pred. Interval')

ax.set_xlabel('Year (standardized)')
ax.set_ylabel('Count (C)')
ax.set_title('B. Log-Linear Model with Prediction Interval')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/09_prediction_intervals.png', dpi=150, bbox_inches='tight')
print("[SAVED] Prediction intervals: visualizations/09_prediction_intervals.png")
plt.close()

print("\n" + "="*80)
print("FINAL RECOMMENDATIONS")
print("="*80)
print("""
RANKING BY CRITERION:

1. Best Statistical Fit (AIC/BIC): Log-Linear / Poisson GLM
   - Highest Akaike weight
   - Best predictive performance (LOO-CV)
   - Appropriate for count data

2. Best Interpretability: Quadratic
   - Simple polynomial form
   - Good R² with only 3 parameters
   - Easy to explain curvature

3. Best for Bayesian Modeling: Negative Binomial GLM
   - Accounts for overdispersion
   - Log link maintains positivity
   - Natural for count data

FOR BAYESIAN IMPLEMENTATION:
- PRIMARY: Negative Binomial with log link
- ALTERNATIVE: Poisson with random effects
- AVOID: Linear models (wrong for count data)

NEXT STEPS:
1. Implement Negative Binomial GLM in Stan/PyMC
2. Use weakly informative priors centered on observed estimates
3. Check posterior predictive distribution for overdispersion
4. Consider hierarchical structure if additional grouping exists
""")
