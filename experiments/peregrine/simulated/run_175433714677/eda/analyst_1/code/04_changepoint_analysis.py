"""
Change Point Detection and Regime Analysis
Focus: Identifying structural breaks and growth phases
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('/workspace/data/data_analyst_1.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

X = df['year'].values
y = df['C'].values
n = len(y)

print("="*60)
print("CHANGE POINT DETECTION")
print("="*60)

# 1. Test multiple potential change points
# We'll test every possible split point and measure fit improvement
def fit_piecewise_model(split_idx):
    """Fit piecewise linear model with split at given index"""
    X1, y1 = X[:split_idx], y[:split_idx]
    X2, y2 = X[split_idx:], y[split_idx:]

    if len(X1) < 3 or len(X2) < 3:  # Need minimum points for regression
        return np.inf, None, None

    slope1, intercept1, _, _, _ = stats.linregress(X1, y1)
    slope2, intercept2, _, _, _ = stats.linregress(X2, y2)

    y_pred = np.zeros(n)
    y_pred[:split_idx] = intercept1 + slope1 * X1
    y_pred[split_idx:] = intercept2 + slope2 * X2

    sse = np.sum((y - y_pred)**2)
    return sse, (slope1, intercept1), (slope2, intercept2)

# Test all possible split points (excluding extremes)
min_segment = 5  # Minimum points in each segment
test_range = range(min_segment, n - min_segment)

results = []
for split_idx in test_range:
    sse, params1, params2 = fit_piecewise_model(split_idx)
    results.append({
        'split_idx': split_idx,
        'split_year': X[split_idx],
        'sse': sse
    })

results_df = pd.DataFrame(results)
best_split = results_df.loc[results_df['sse'].idxmin()]

print(f"\n1. OPTIMAL CHANGE POINT")
print(f"   Best split at index: {int(best_split['split_idx'])}")
print(f"   Year value: {best_split['split_year']:.4f}")
print(f"   SSE: {best_split['sse']:.2f}")

# Fit best model
best_idx = int(best_split['split_idx'])
_, params_early, params_late = fit_piecewise_model(best_idx)

print(f"\n   Early phase (n={best_idx}):")
print(f"   - Slope: {params_early[0]:.2f} counts/year")
print(f"   - Intercept: {params_early[1]:.2f}")

print(f"\n   Late phase (n={n - best_idx}):")
print(f"   - Slope: {params_late[0]:.2f} counts/year")
print(f"   - Intercept: {params_late[1]:.2f}")
print(f"   - Acceleration factor: {params_late[0]/params_early[0]:.2f}x")

# 2. Test for structural break using Chow test
# Compare full model vs piecewise model
slope_full, intercept_full, _, _, _ = stats.linregress(X, y)
y_pred_full = intercept_full + slope_full * X
sse_full = np.sum((y - y_pred_full)**2)
sse_piecewise = best_split['sse']

# F-statistic for Chow test
k = 2  # parameters per regression
F_stat = ((sse_full - sse_piecewise) / k) / (sse_piecewise / (n - 2*k))
from scipy.stats import f as f_dist
p_value = 1 - f_dist.cdf(F_stat, k, n - 2*k)

print(f"\n2. CHOW TEST FOR STRUCTURAL BREAK")
print(f"   F-statistic: {F_stat:.4f}")
print(f"   p-value: {p_value:.6f}")
if p_value < 0.05:
    print("   -> SIGNIFICANT structural break detected")
else:
    print("   -> No significant structural break")

# 3. Calculate rolling statistics to visualize regime changes
window = 8
df['rolling_mean'] = df['C'].rolling(window=window, center=True).mean()
df['rolling_std'] = df['C'].rolling(window=window, center=True).std()
df['rolling_cv'] = df['rolling_std'] / df['rolling_mean']  # Coefficient of variation

print(f"\n3. ROLLING STATISTICS (window={window})")
print(f"   Early rolling mean: {df['rolling_mean'].iloc[window:window+5].mean():.2f}")
print(f"   Late rolling mean: {df['rolling_mean'].iloc[-5:].mean():.2f}")
print(f"   Mean coefficient of variation: {df['rolling_cv'].mean():.4f}")

# 4. Calculate growth rates
df['growth_rate'] = df['C'].pct_change()
df['abs_change'] = df['C'].diff()

print(f"\n4. GROWTH RATE ANALYSIS")
print(f"   Mean growth rate: {df['growth_rate'].mean()*100:.2f}%")
print(f"   Median growth rate: {df['growth_rate'].median()*100:.2f}%")
print(f"   Max growth rate: {df['growth_rate'].max()*100:.2f}%")
print(f"   Min growth rate: {df['growth_rate'].min()*100:.2f}%")

# Split by best change point
growth_early = df['growth_rate'].iloc[1:best_idx]
growth_late = df['growth_rate'].iloc[best_idx:]
print(f"\n   Early phase mean growth: {growth_early.mean()*100:.2f}%")
print(f"   Late phase mean growth: {growth_late.mean()*100:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SSE vs split point
ax1 = axes[0, 0]
ax1.plot(results_df['split_year'], results_df['sse'], linewidth=2, color='steelblue')
ax1.scatter([best_split['split_year']], [best_split['sse']],
           s=200, color='red', zorder=5, label=f'Optimal: {best_split["split_year"]:.3f}')
ax1.set_xlabel('Split Point (Year)', fontsize=11)
ax1.set_ylabel('Sum of Squared Errors', fontsize=11)
ax1.set_title('A. Change Point Detection\n(SSE vs Split Location)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Data with piecewise fit
ax2 = axes[0, 1]
ax2.scatter(X, y, alpha=0.6, s=50, label='Observed', color='steelblue')
# Plot piecewise fit
X1, X2 = X[:best_idx], X[best_idx:]
y_pred1 = params_early[1] + params_early[0] * X1
y_pred2 = params_late[1] + params_late[0] * X2
ax2.plot(X1, y_pred1, 'r-', linewidth=3, label='Early regime')
ax2.plot(X2, y_pred2, 'g-', linewidth=3, label='Late regime')
ax2.axvline(best_split['split_year'], color='orange', linestyle='--',
           linewidth=2, alpha=0.7, label='Change point')
ax2.set_xlabel('Year (standardized)', fontsize=11)
ax2.set_ylabel('Count (C)', fontsize=11)
ax2.set_title('B. Piecewise Linear Fit\n(Two Regime Model)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Rolling mean with bands
ax3 = axes[1, 0]
ax3.plot(X, y, 'o-', alpha=0.4, linewidth=1, markersize=4, label='Observed', color='gray')
ax3.plot(X, df['rolling_mean'], linewidth=3, label=f'Rolling mean ({window}-obs)', color='darkblue')
ax3.fill_between(X,
                  df['rolling_mean'] - df['rolling_std'],
                  df['rolling_mean'] + df['rolling_std'],
                  alpha=0.2, color='blue', label='±1 SD')
ax3.axvline(best_split['split_year'], color='red', linestyle='--',
           linewidth=2, alpha=0.7, label='Change point')
ax3.set_xlabel('Year (standardized)', fontsize=11)
ax3.set_ylabel('Count (C)', fontsize=11)
ax3.set_title('C. Rolling Statistics\n(Smoothed Trend)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Growth rates
ax4 = axes[1, 1]
ax4.bar(X[1:], df['growth_rate'].iloc[1:], alpha=0.7, color='green', width=0.08)
ax4.axhline(0, color='black', linewidth=1)
ax4.axvline(best_split['split_year'], color='red', linestyle='--',
           linewidth=2, alpha=0.7, label='Change point')
ax4.axhline(growth_early.mean(), color='blue', linestyle='--',
           linewidth=2, label=f'Early mean: {growth_early.mean()*100:.1f}%')
ax4.axhline(growth_late.mean(), color='orange', linestyle='--',
           linewidth=2, label=f'Late mean: {growth_late.mean()*100:.1f}%')
ax4.set_xlabel('Year (standardized)', fontsize=11)
ax4.set_ylabel('Growth Rate', fontsize=11)
ax4.set_title('D. Period-to-Period Growth Rates', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/04_changepoint_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: 04_changepoint_analysis.png")

plt.close()
