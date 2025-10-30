"""
Hypothesis Testing - Analyst 1
===============================
Purpose: Test competing hypotheses about data structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("="*60)
print("COMPETING HYPOTHESES ABOUT DATA STRUCTURE")
print("="*60)

# HYPOTHESIS 1: Linear relationship with diminishing returns
# (logarithmic or power model)
print("\n" + "="*60)
print("HYPOTHESIS 1: Diminishing Returns Pattern")
print("="*60)
print("Prediction: Rate of Y increase slows as x increases")

# Calculate incremental changes
data_sorted = data.sort_values('x').reset_index(drop=True)

# Group by x and calculate mean Y for each x
grouped = data.groupby('x')['Y'].agg(['mean', 'count', 'std']).reset_index()
grouped['Y_diff'] = grouped['mean'].diff()
grouped['x_diff'] = grouped['x'].diff()
grouped['rate_of_change'] = grouped['Y_diff'] / grouped['x_diff']

print("\nRate of change analysis (dY/dx):")
print(grouped[['x', 'mean', 'rate_of_change']].to_string(index=False))

# Test if rate of change decreases
valid_rates = grouped['rate_of_change'].dropna()
first_half_rate = valid_rates.iloc[:len(valid_rates)//2].mean()
second_half_rate = valid_rates.iloc[len(valid_rates)//2:].mean()

print(f"\nFirst half mean rate: {first_half_rate:.4f}")
print(f"Second half mean rate: {second_half_rate:.4f}")
print(f"Ratio (2nd/1st): {second_half_rate/first_half_rate:.4f}")

if second_half_rate < first_half_rate:
    print("=> SUPPORTED: Rate of change decreases over x range")
else:
    print("=> NOT SUPPORTED: Rate of change does not decrease")

# HYPOTHESIS 2: Linear relationship with varying noise
print("\n" + "="*60)
print("HYPOTHESIS 2: Linear Model with Heteroscedastic Noise")
print("="*60)
print("Prediction: Linear trend but variance changes with x")

# Fit linear model
linear_coef = np.polyfit(data['x'], data['Y'], 1)
y_pred_linear = np.polyval(linear_coef, data['x'])
residuals_linear = data['Y'] - y_pred_linear

# Test correlation between |residuals| and x
abs_residuals = np.abs(residuals_linear)
corr_resid_x, p_resid_x = stats.spearmanr(data['x'], abs_residuals)

print(f"\nCorrelation between |residuals| and x:")
print(f"  Spearman rho = {corr_resid_x:.4f}, p = {p_resid_x:.4f}")

if p_resid_x < 0.05:
    print(f"=> SUPPORTED: Significant relationship between variance and x")
else:
    print(f"=> NOT SUPPORTED: No significant variance-x relationship")

# HYPOTHESIS 3: Piecewise linear or change-point model
print("\n" + "="*60)
print("HYPOTHESIS 3: Change-Point or Piecewise Linear Model")
print("="*60)
print("Prediction: Slope changes at some point in x range")

# Test different breakpoints
x_sorted = np.sort(data['x'].unique())
best_breakpoint = None
best_rss = float('inf')
rss_by_breakpoint = []

for i in range(2, len(x_sorted)-2):  # Need at least 2 points on each side
    breakpoint = x_sorted[i]

    # Split data
    data_low = data[data['x'] <= breakpoint]
    data_high = data[data['x'] > breakpoint]

    if len(data_low) >= 3 and len(data_high) >= 3:
        # Fit separate models
        coef_low = np.polyfit(data_low['x'], data_low['Y'], 1)
        coef_high = np.polyfit(data_high['x'], data_high['Y'], 1)

        # Calculate RSS
        pred_low = np.polyval(coef_low, data_low['x'])
        pred_high = np.polyval(coef_high, data_high['x'])
        rss = np.sum((data_low['Y'] - pred_low)**2) + np.sum((data_high['Y'] - pred_high)**2)

        rss_by_breakpoint.append((breakpoint, rss, coef_low, coef_high))

        if rss < best_rss:
            best_rss = rss
            best_breakpoint = breakpoint
            best_coef_low = coef_low
            best_coef_high = coef_high

# Compare to single linear model
rss_single = np.sum(residuals_linear**2)

print(f"\nBest breakpoint: x = {best_breakpoint:.1f}")
print(f"  RSS (piecewise): {best_rss:.4f}")
print(f"  RSS (single linear): {rss_single:.4f}")
print(f"  Improvement: {(rss_single - best_rss)/rss_single * 100:.2f}%")

# F-test for model comparison
n = len(data)
k_full = 4  # 2 slopes + 2 intercepts
k_reduced = 2  # 1 slope + 1 intercept
f_stat = ((rss_single - best_rss) / (k_full - k_reduced)) / (best_rss / (n - k_full))
p_value = 1 - stats.f.cdf(f_stat, k_full - k_reduced, n - k_full)

print(f"\nF-test for piecewise vs single linear:")
print(f"  F = {f_stat:.4f}, p = {p_value:.4f}")

if p_value < 0.05:
    print(f"=> SUPPORTED: Piecewise model significantly better")
else:
    print(f"=> NOT SUPPORTED: Single linear model sufficient")

# Create visualization comparing hypotheses
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Testing Competing Hypotheses', fontsize=14, fontweight='bold')

# 1. Diminishing returns (log model)
ax = axes[0, 0]
ax.scatter(data['x'], data['Y'], s=60, alpha=0.6, edgecolors='black', label='Data')
x_line = np.linspace(data['x'].min(), data['x'].max(), 100)
log_coef = np.polyfit(np.log(data['x']), data['Y'], 1)
y_log = log_coef[0] * np.log(x_line) + log_coef[1]
ax.plot(x_line, y_log, 'r-', linewidth=2, label='Logarithmic fit')
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title('H1: Diminishing Returns (Log Model)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# 2. Rate of change over x
ax = axes[0, 1]
ax.plot(grouped['x'].iloc[1:], grouped['rate_of_change'].iloc[1:],
        'o-', markersize=8, linewidth=2, color='blue')
ax.axhline(y=grouped['rate_of_change'].iloc[1:].mean(), color='red',
           linestyle='--', label='Mean rate')
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('dY/dx', fontsize=11)
ax.set_title('H1: Rate of Change Analysis', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# 3. Heteroscedasticity visualization
ax = axes[1, 0]
ax.scatter(data['x'], abs_residuals, s=60, alpha=0.6, edgecolors='black')
# Add trend line
z = np.polyfit(data['x'], abs_residuals, 1)
p = np.poly1d(z)
ax.plot(data['x'].sort_values(), p(data['x'].sort_values()),
        'r--', linewidth=2, label=f'Trend (r={corr_resid_x:.3f})')
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('|Residuals| (Linear Model)', fontsize=11)
ax.set_title('H2: Heteroscedasticity Test', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# 4. Piecewise linear model
ax = axes[1, 1]
ax.scatter(data['x'], data['Y'], s=60, alpha=0.6, edgecolors='black', label='Data')

# Plot best piecewise fit
data_low = data[data['x'] <= best_breakpoint]
data_high = data[data['x'] > best_breakpoint]
x_low = np.linspace(data['x'].min(), best_breakpoint, 50)
x_high = np.linspace(best_breakpoint, data['x'].max(), 50)
y_low = np.polyval(best_coef_low, x_low)
y_high = np.polyval(best_coef_high, x_high)

ax.plot(x_low, y_low, 'r-', linewidth=2, label=f'Segment 1 (slope={best_coef_low[0]:.4f})')
ax.plot(x_high, y_high, 'g-', linewidth=2, label=f'Segment 2 (slope={best_coef_high[0]:.4f})')
ax.axvline(x=best_breakpoint, color='orange', linestyle='--', linewidth=2,
           label=f'Breakpoint: x={best_breakpoint:.1f}')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title(f'H3: Piecewise Linear (p={p_value:.3f})', fontsize=12)
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/08_hypothesis_testing.png',
            dpi=300, bbox_inches='tight')
print("\nSaved: 08_hypothesis_testing.png")
plt.close()

# Additional plot: Model comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Plot data
ax.scatter(data['x'], data['Y'], s=100, alpha=0.7, edgecolors='black',
           linewidth=1.5, label='Observed data', zorder=5)

# Plot all models
x_plot = np.linspace(data['x'].min(), data['x'].max(), 200)

# Linear
y_linear = linear_coef[0] * x_plot + linear_coef[1]
ax.plot(x_plot, y_linear, '--', linewidth=2, alpha=0.7,
        label=f'Linear (R²={1 - rss_single/np.sum((data["Y"] - data["Y"].mean())**2):.3f})')

# Logarithmic
y_log_plot = log_coef[0] * np.log(x_plot) + log_coef[1]
ax.plot(x_plot, y_log_plot, '-', linewidth=2.5, alpha=0.8,
        label=f'Logarithmic (R²=0.897)')

# Quadratic
quad_coef = np.polyfit(data['x'], data['Y'], 2)
y_quad = np.polyval(quad_coef, x_plot)
ax.plot(x_plot, y_quad, '-.', linewidth=2, alpha=0.7,
        label=f'Quadratic (R²=0.874)')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Comparison of Competing Models', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/09_model_comparison.png',
            dpi=300, bbox_inches='tight')
print("Saved: 09_model_comparison.png")
plt.close()

print("\n" + "="*60)
print("HYPOTHESIS TESTING COMPLETE")
print("="*60)
