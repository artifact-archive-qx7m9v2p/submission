"""
Deep dive into correlation structure and predictive informativeness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*60)
print("CORRELATION STRUCTURE AND PREDICTIVE INFORMATIVENESS")
print("="*60)

# 1. Bootstrap confidence intervals for correlation
def correlation_stat(x, y):
    return stats.pearsonr(x, y)[0]

n_bootstrap = 1000
rng = np.random.default_rng(42)

# Bootstrap for Pearson
bootstrap_corrs = []
for _ in range(n_bootstrap):
    indices = rng.choice(len(data), size=len(data), replace=True)
    sample = data.iloc[indices]
    corr = sample['x'].corr(sample['Y'])
    bootstrap_corrs.append(corr)

bootstrap_corrs = np.array(bootstrap_corrs)
ci_lower = np.percentile(bootstrap_corrs, 2.5)
ci_upper = np.percentile(bootstrap_corrs, 97.5)

print("\n1. CORRELATION CONFIDENCE INTERVALS (Bootstrap)")
print(f"Pearson correlation: {data['x'].corr(data['Y']):.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"CI width: {ci_upper - ci_lower:.4f}")

# 2. Correlation by x range
ranges = [(1, 10), (10, 20), (20, 32)]
print("\n2. CORRELATION BY X RANGE")
for x_min, x_max in ranges:
    subset = data[(data['x'] >= x_min) & (data['x'] < x_max)]
    if len(subset) >= 3:
        corr = subset['x'].corr(subset['Y'])
        print(f"x ∈ [{x_min}, {x_max}): r = {corr:.4f}, n = {len(subset)}")

# 3. Variance explained by x
total_var = data['Y'].var()
# Simple linear model
z = np.polyfit(data['x'], data['Y'], 1)
y_pred_linear = z[0] * data['x'] + z[1]
explained_var_linear = 1 - (data['Y'] - y_pred_linear).var() / total_var

print("\n3. VARIANCE DECOMPOSITION")
print(f"Total Y variance: {total_var:.6f}")
print(f"Variance explained by linear model: {explained_var_linear:.4f} ({explained_var_linear*100:.1f}%)")
print(f"Residual variance: {1-explained_var_linear:.4f} ({(1-explained_var_linear)*100:.1f}%)")

# 4. Prediction intervals
def prediction_interval(x, y, x_pred, confidence=0.95):
    """Calculate prediction interval for new observations"""
    n = len(x)

    # Fit model
    coeffs = np.polyfit(x, y, 1)
    y_fit = coeffs[0] * x + coeffs[1]

    # Residual standard error
    residuals = y - y_fit
    rse = np.sqrt(np.sum(residuals**2) / (n - 2))

    # Standard error of prediction
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)

    se_pred = rse * np.sqrt(1 + 1/n + (x_pred - x_mean)**2 / sxx)

    # t-statistic
    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)

    # Prediction
    y_pred = coeffs[0] * x_pred + coeffs[1]

    return y_pred, y_pred - t_val * se_pred, y_pred + t_val * se_pred

x_test = np.array([5, 10, 15, 20, 25])
print("\n4. PREDICTION INTERVALS (95% confidence)")
print(f"{'x':<6} {'Predicted Y':<12} {'Lower':<10} {'Upper':<10} {'Width':<10}")
print("-" * 60)
for x_val in x_test:
    pred, lower, upper = prediction_interval(data['x'].values, data['Y'].values, x_val)
    width = upper - lower
    print(f"{x_val:<6.1f} {pred:<12.4f} {lower:<10.4f} {upper:<10.4f} {width:<10.4f}")

# 5. Influence analysis - which points are most influential?
from scipy.stats import linregress

# Calculate Cook's distance
def cooks_distance(x, y):
    """Calculate Cook's distance for each observation"""
    n = len(x)
    p = 2  # number of parameters

    # Fit full model
    result = linregress(x, y)
    y_pred = result.slope * x + result.intercept
    mse = np.sum((y - y_pred)**2) / (n - p)

    # Calculate leverage
    x_mean = np.mean(x)
    h = 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2)

    # Calculate Cook's distance
    residuals = y - y_pred
    cooks_d = (residuals**2 / (p * mse)) * (h / (1 - h)**2)

    return cooks_d

cooks_d = cooks_distance(data['x'].values, data['Y'].values)
influential = cooks_d > 4 / len(data)  # Common threshold

print("\n5. INFLUENTIAL OBSERVATIONS (Cook's Distance)")
print(f"Threshold: {4/len(data):.4f}")
print(f"Influential points: {np.sum(influential)}")
if np.sum(influential) > 0:
    influential_df = data[influential].copy()
    influential_df['CooksD'] = cooks_d[influential]
    print("\nMost influential observations:")
    print(influential_df.sort_values('CooksD', ascending=False)[['x', 'Y', 'CooksD']].to_string())

# 6. R² by sample size (learning curve)
print("\n6. PREDICTIVE POWER BY SAMPLE SIZE")
sample_sizes = [5, 10, 15, 20, 27]
r2_by_size = []

for size in sample_sizes:
    if size <= len(data):
        # Take first 'size' points sorted by x
        subset = data.sort_values('x').head(size)
        if len(subset) >= 3:
            z = np.polyfit(subset['x'], subset['Y'], 1)
            y_pred = z[0] * subset['x'] + z[1]
            ss_res = np.sum((subset['Y'] - y_pred)**2)
            ss_tot = np.sum((subset['Y'] - subset['Y'].mean())**2)
            r2 = 1 - (ss_res / ss_tot)
            r2_by_size.append((size, r2))
            print(f"n = {size:2d}: R² = {r2:.4f}")

print("\n" + "="*60)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Correlation Structure and Predictive Informativeness',
             fontsize=16, fontweight='bold')

# 1. Bootstrap distribution
ax1 = axes[0, 0]
ax1.hist(bootstrap_corrs, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
ax1.axvline(data['x'].corr(data['Y']), color='red', linestyle='--',
           linewidth=2, label=f'Observed: {data["x"].corr(data["Y"]):.4f}')
ax1.axvline(ci_lower, color='green', linestyle='--', alpha=0.7,
           label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
ax1.axvline(ci_upper, color='green', linestyle='--', alpha=0.7)
ax1.set_xlabel('Pearson Correlation', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Bootstrap Distribution of Correlation', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Prediction intervals
ax2 = axes[0, 1]
x_smooth = np.linspace(data['x'].min(), data['x'].max(), 100)
predictions = []
lowers = []
uppers = []
for x_val in x_smooth:
    pred, lower, upper = prediction_interval(data['x'].values, data['Y'].values, x_val)
    predictions.append(pred)
    lowers.append(lower)
    uppers.append(upper)

ax2.scatter(data['x'], data['Y'], alpha=0.6, s=80, edgecolors='black',
           linewidths=0.5, label='Data', zorder=3)
ax2.plot(x_smooth, predictions, 'r-', linewidth=2, label='Prediction', zorder=2)
ax2.fill_between(x_smooth, lowers, uppers, alpha=0.2, color='red',
                label='95% Prediction Interval')
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('Y', fontsize=11)
ax2.set_title('Prediction Intervals', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Cook's distance
ax3 = axes[1, 0]
ax3.bar(range(len(data)), cooks_d, alpha=0.7, edgecolor='black')
ax3.axhline(4/len(data), color='red', linestyle='--', linewidth=2,
           label=f'Threshold: {4/len(data):.4f}')
ax3.set_xlabel('Observation Index', fontsize=11)
ax3.set_ylabel("Cook's Distance", fontsize=11)
ax3.set_title('Influential Observations Analysis', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Learning curve
ax4 = axes[1, 1]
sizes, r2s = zip(*r2_by_size)
ax4.plot(sizes, r2s, 'o-', linewidth=2, markersize=10, color='steelblue')
ax4.set_xlabel('Sample Size', fontsize=11)
ax4.set_ylabel('R²', fontsize=11)
ax4.set_title('R² by Sample Size (Learning Curve)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/13_correlation_structure.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved: 13_correlation_structure.png")
print("Correlation structure analysis complete.")
