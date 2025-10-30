"""
Variance Structure Analysis - Analyst 1
========================================
Purpose: Examine variance structure and test for heteroscedasticity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Fit different models and analyze residuals
linear_coef = np.polyfit(data['x'], data['Y'], 1)
y_pred_linear = np.polyval(linear_coef, data['x'])
residuals_linear = data['Y'] - y_pred_linear

log_coef = np.polyfit(np.log(data['x']), data['Y'], 1)
y_pred_log = np.polyval(log_coef, np.log(data['x']))
residuals_log = data['Y'] - y_pred_log

quad_coef = np.polyfit(data['x'], data['Y'], 2)
y_pred_quad = np.polyval(quad_coef, data['x'])
residuals_quad = data['Y'] - y_pred_quad

# Create comprehensive residual analysis plot
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('Residual Analysis for Different Models', fontsize=16, fontweight='bold')

models = [
    ('Linear', residuals_linear, y_pred_linear),
    ('Logarithmic', residuals_log, y_pred_log),
    ('Quadratic', residuals_quad, y_pred_quad)
]

for i, (model_name, residuals, y_pred) in enumerate(models):
    # Residuals vs Fitted
    axes[i, 0].scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black')
    axes[i, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[i, 0].axhline(y=residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[i, 0].axhline(y=-residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[i, 0].set_xlabel('Fitted Values', fontsize=10)
    axes[i, 0].set_ylabel('Residuals', fontsize=10)
    axes[i, 0].set_title(f'{model_name}: Residuals vs Fitted', fontsize=11, fontweight='bold')
    axes[i, 0].grid(alpha=0.3)

    # Residuals vs x
    axes[i, 1].scatter(data['x'], residuals, alpha=0.6, s=60, edgecolors='black')
    axes[i, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[i, 1].axhline(y=residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[i, 1].axhline(y=-residuals.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[i, 1].set_xlabel('x', fontsize=10)
    axes[i, 1].set_ylabel('Residuals', fontsize=10)
    axes[i, 1].set_title(f'{model_name}: Residuals vs x', fontsize=11, fontweight='bold')
    axes[i, 1].grid(alpha=0.3)

    # Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f'{model_name}: Q-Q Plot', fontsize=11, fontweight='bold')
    axes[i, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/04_residual_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: 04_residual_analysis.png")
plt.close()

# Test for heteroscedasticity
print("\n" + "="*60)
print("HETEROSCEDASTICITY ANALYSIS")
print("="*60)

def breusch_pagan_test(residuals, x):
    """Breusch-Pagan test for heteroscedasticity"""
    # Square the residuals
    residuals_sq = residuals**2
    # Regress squared residuals on x
    x_with_const = np.column_stack([np.ones(len(x)), x])
    coeffs = np.linalg.lstsq(x_with_const, residuals_sq, rcond=None)[0]
    fitted = x_with_const @ coeffs
    # Calculate test statistic
    ss_reg = np.sum((fitted - np.mean(residuals_sq))**2)
    ss_tot = np.sum((residuals_sq - np.mean(residuals_sq))**2)
    r2 = ss_reg / ss_tot
    n = len(residuals)
    lm = n * r2
    # Chi-square test with 1 df
    p_value = 1 - stats.chi2.cdf(lm, df=1)
    return lm, p_value

for model_name, residuals, _ in models:
    print(f"\n{model_name} Model:")
    print("-"*60)

    # Shapiro-Wilk test for normality of residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"  Shapiro-Wilk test (residuals normality):")
    print(f"    W = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")

    # Breusch-Pagan test
    bp_stat, bp_p = breusch_pagan_test(residuals, data['x'].values)
    print(f"  Breusch-Pagan test (heteroscedasticity):")
    print(f"    LM = {bp_stat:.4f}, p-value = {bp_p:.4f}")
    if bp_p < 0.05:
        print(f"    => Evidence of heteroscedasticity (p < 0.05)")
    else:
        print(f"    => No strong evidence of heteroscedasticity (p >= 0.05)")

    # Residual statistics
    print(f"  Residual statistics:")
    print(f"    Mean: {residuals.mean():.6f}")
    print(f"    Std Dev: {residuals.std():.4f}")
    print(f"    Min: {residuals.min():.4f}")
    print(f"    Max: {residuals.max():.4f}")

# Analyze variance across x ranges
print("\n" + "="*60)
print("VARIANCE ACROSS X RANGES")
print("="*60)

# Split data into thirds based on x
data_sorted = data.sort_values('x')
n = len(data_sorted)
third = n // 3

low_x = data_sorted.iloc[:third]
mid_x = data_sorted.iloc[third:2*third]
high_x = data_sorted.iloc[2*third:]

print(f"\nLow x range [${low_x['x'].min():.1f}, {low_x['x'].max():.1f}]:")
print(f"  Y variance: {low_x['Y'].var():.6f}")
print(f"  Y std dev: {low_x['Y'].std():.4f}")
print(f"  n = {len(low_x)}")

print(f"\nMid x range [{mid_x['x'].min():.1f}, {mid_x['x'].max():.1f}]:")
print(f"  Y variance: {mid_x['Y'].var():.6f}")
print(f"  Y std dev: {mid_x['Y'].std():.4f}")
print(f"  n = {len(mid_x)}")

print(f"\nHigh x range [{high_x['x'].min():.1f}, {high_x['x'].max():.1f}]:")
print(f"  Y variance: {high_x['Y'].var():.6f}")
print(f"  Y std dev: {high_x['Y'].std():.4f}")
print(f"  n = {len(high_x)}")

# Levene's test for equality of variances
levene_stat, levene_p = stats.levene(low_x['Y'], mid_x['Y'], high_x['Y'])
print(f"\nLevene's test for equal variances:")
print(f"  F = {levene_stat:.4f}, p-value = {levene_p:.4f}")
if levene_p < 0.05:
    print(f"  => Variances differ significantly across x ranges (p < 0.05)")
else:
    print(f"  => No strong evidence of variance differences (p >= 0.05)")

# Create plot showing variance across x ranges
fig, ax = plt.subplots(figsize=(10, 6))

# Use logarithmic model residuals for visualization
colors = ['red' if x < low_x['x'].max() else 'green' if x < mid_x['x'].max() else 'blue'
          for x in data['x']]

ax.scatter(data['x'], residuals_log, c=colors, s=80, alpha=0.6, edgecolors='black')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.axvline(x=low_x['x'].max(), color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(x=mid_x['x'].max(), color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# Add variance bands for each region
ax.axhspan(-low_x['Y'].std(), low_x['Y'].std(), xmin=0,
           xmax=(low_x['x'].max() - data['x'].min())/(data['x'].max() - data['x'].min()),
           alpha=0.1, color='red', label=f'Low x SD: {low_x["Y"].std():.3f}')
ax.axhspan(-mid_x['Y'].std(), mid_x['Y'].std(),
           xmin=(low_x['x'].max() - data['x'].min())/(data['x'].max() - data['x'].min()),
           xmax=(mid_x['x'].max() - data['x'].min())/(data['x'].max() - data['x'].min()),
           alpha=0.1, color='green', label=f'Mid x SD: {mid_x["Y"].std():.3f}')
ax.axhspan(-high_x['Y'].std(), high_x['Y'].std(),
           xmin=(mid_x['x'].max() - data['x'].min())/(data['x'].max() - data['x'].min()),
           xmax=1, alpha=0.1, color='blue', label=f'High x SD: {high_x["Y"].std():.3f}')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Residuals (Log Model)', fontsize=12)
ax.set_title('Residuals Across X Ranges: Variance Structure', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/05_variance_by_x_range.png', dpi=300, bbox_inches='tight')
print("\nSaved: 05_variance_by_x_range.png")
plt.close()

print("\n" + "="*60)
print("VARIANCE ANALYSIS COMPLETE")
print("="*60)
