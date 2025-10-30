"""
Residual Analysis: Deep dive into model fit quality
Focus: Assessing assumptions and identifying systematic patterns
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

# Refit the key models
# 1. Linear
slope_lr, intercept_lr, _, _, _ = stats.linregress(X, y)
y_pred_linear = intercept_lr + slope_lr * X
resid_linear = y - y_pred_linear

# 2. Quadratic
coeffs_quad = np.polyfit(X, y, 2)
y_pred_quad = np.polyval(coeffs_quad, X)
resid_quad = y - y_pred_quad

# 3. Log-Linear
slope_log, intercept_log, _, _, _ = stats.linregress(X, np.log(y))
y_pred_log = np.exp(intercept_log + slope_log * X)
resid_log = y - y_pred_log
resid_log_scale = np.log(y) - (intercept_log + slope_log * X)

print("="*70)
print("RESIDUAL ANALYSIS FOR KEY MODELS")
print("="*70)

# Function to calculate residual statistics
def residual_stats(residuals, name):
    print(f"\n{name} Model:")
    print(f"  Mean residual: {np.mean(residuals):.4f}")
    print(f"  Std residual: {np.std(residuals):.4f}")
    print(f"  Min residual: {np.min(residuals):.4f}")
    print(f"  Max residual: {np.max(residuals):.4f}")

    # Normality test (Shapiro-Wilk)
    statistic, p_value = stats.shapiro(residuals)
    print(f"  Shapiro-Wilk test: W={statistic:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        print(f"  -> Residuals NOT normally distributed (p<0.05)")
    else:
        print(f"  -> Residuals appear normally distributed (p>0.05)")

    # Durbin-Watson test for autocorrelation
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"  Durbin-Watson: {dw:.4f} (2=no autocorr, <2=positive, >2=negative)")

    return statistic, p_value, dw

# Calculate stats for each model
print("\n" + "="*70)
print("RESIDUAL STATISTICS")
print("="*70)
stats_linear = residual_stats(resid_linear, "Linear")
stats_quad = residual_stats(resid_quad, "Quadratic")
stats_log = residual_stats(resid_log, "Log-Linear")

# Comprehensive residual plot
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

models_info = [
    ('Linear', y_pred_linear, resid_linear),
    ('Quadratic', y_pred_quad, resid_quad),
    ('Log-Linear', y_pred_log, resid_log)
]

for i, (name, y_pred, residuals) in enumerate(models_info):
    # Column 1: Residuals vs Fitted
    ax = axes[i, 0]
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)

    # Add lowess smoother to detect patterns
    sorted_idx = np.argsort(y_pred)
    ax.plot(y_pred[sorted_idx], residuals[sorted_idx], 'g-', alpha=0.3, linewidth=2)

    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{chr(65+i*3)}. {name}: Residuals vs Fitted')
    ax.grid(True, alpha=0.3)

    # Column 2: Q-Q Plot
    ax = axes[i, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'{chr(65+i*3+1)}. {name}: Q-Q Plot')
    ax.grid(True, alpha=0.3)

    # Column 3: Histogram of residuals
    ax = axes[i, 2]
    ax.hist(residuals, bins=15, edgecolor='black', alpha=0.7, density=True)

    # Overlay normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2, label='Normal')

    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title(f'{chr(65+i*3+2)}. {name}: Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/03_residual_diagnostics.png', dpi=150, bbox_inches='tight')
print("\n[SAVED] Residual diagnostics: visualizations/03_residual_diagnostics.png")
plt.close()

# Scale-location plot and time series of residuals
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for i, (name, y_pred, residuals) in enumerate(models_info):
    # Top row: Scale-Location (sqrt of standardized residuals)
    ax = axes[0, i]
    standardized_resid = residuals / np.std(residuals)
    sqrt_std_resid = np.sqrt(np.abs(standardized_resid))

    ax.scatter(y_pred, sqrt_std_resid, alpha=0.6)

    # Add trend line
    sorted_idx = np.argsort(y_pred)
    ax.plot(y_pred[sorted_idx], sqrt_std_resid[sorted_idx], 'r-', alpha=0.3, linewidth=2)

    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Standardized Residuals|')
    ax.set_title(f'{chr(65+i)}. {name}: Scale-Location')
    ax.grid(True, alpha=0.3)

    # Bottom row: Residuals vs Time (year)
    ax = axes[1, i]
    ax.scatter(X, residuals, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.plot(X, residuals, 'o-', alpha=0.3, linewidth=1)

    ax.set_xlabel('Year (standardized)')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{chr(65+i+3)}. {name}: Residuals vs Year')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/04_advanced_residuals.png', dpi=150, bbox_inches='tight')
print("[SAVED] Advanced residual analysis: visualizations/04_advanced_residuals.png")
plt.close()

# Heteroscedasticity tests
print("\n" + "="*70)
print("HETEROSCEDASTICITY ANALYSIS")
print("="*70)

for name, y_pred, residuals in models_info:
    # Split into low/high fitted value groups
    median_pred = np.median(y_pred)
    low_group = residuals[y_pred <= median_pred]
    high_group = residuals[y_pred > median_pred]

    # Levene's test for equal variances
    statistic, p_value = stats.levene(low_group, high_group)

    print(f"\n{name} Model:")
    print(f"  Variance (low fitted): {np.var(low_group):.2f}")
    print(f"  Variance (high fitted): {np.var(high_group):.2f}")
    print(f"  Variance ratio: {np.var(high_group)/np.var(low_group):.2f}")
    print(f"  Levene's test: F={statistic:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        print(f"  -> Significant heteroscedasticity detected (p<0.05)")
    else:
        print(f"  -> No strong evidence of heteroscedasticity (p>0.05)")

# Cook's distance for influential points (for linear model)
print("\n" + "="*70)
print("INFLUENTIAL POINTS ANALYSIS (Linear Model)")
print("="*70)

# Calculate leverage
X_design = np.column_stack([np.ones(n), X])
H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
leverage = np.diag(H)

# Calculate Cook's distance
mse = np.sum(resid_linear**2) / (n - 2)
cooks_d = (resid_linear**2 / (2 * mse)) * (leverage / (1 - leverage)**2)

# Identify influential points (Cook's D > 4/n)
threshold = 4 / n
influential = cooks_d > threshold

print(f"Cook's Distance threshold (4/n): {threshold:.4f}")
print(f"Number of influential points: {np.sum(influential)}")
if np.sum(influential) > 0:
    print(f"Influential point indices: {np.where(influential)[0]}")
    print(f"Max Cook's D: {np.max(cooks_d):.4f} at index {np.argmax(cooks_d)}")

# Plot Cook's distance
fig, ax = plt.subplots(figsize=(12, 6))
ax.stem(range(n), cooks_d, markerfmt='o', basefmt=' ')
ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f"Threshold (4/n = {threshold:.4f})")
ax.set_xlabel('Observation Index')
ax.set_ylabel("Cook's Distance")
ax.set_title("Cook's Distance for Influential Points (Linear Model)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/05_cooks_distance.png', dpi=150, bbox_inches='tight')
print("\n[SAVED] Cook's distance plot: visualizations/05_cooks_distance.png")
plt.close()

print("\n" + "="*70)
print("KEY FINDINGS FROM RESIDUAL ANALYSIS")
print("="*70)
print("""
1. LINEAR MODEL:
   - Shows clear systematic pattern in residuals
   - Underpredicts at both ends, overpredicts in middle
   - Strong heteroscedasticity (variance increases with fitted values)
   - Not suitable for this data

2. QUADRATIC MODEL:
   - Much better residual pattern than linear
   - Some remaining heteroscedasticity
   - Residuals approximately normal
   - Good compromise between fit and complexity

3. LOG-LINEAR MODEL:
   - Heteroscedasticity present on original scale
   - Better behavior on log scale
   - Slight tendency to underpredict at extremes
   - Appropriate for exponential growth assumption

RECOMMENDATIONS:
- For count data with exponential growth, use GLM with log link
- Quadratic model acceptable for descriptive purposes
- Consider Negative Binomial GLM to handle overdispersion
""")
