"""
Structural Break and Regime Change Analysis
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_1.csv')
df = df.sort_values('year').reset_index(drop=True)
df['index'] = range(len(df))

X = df['year'].values
y = df['C'].values
n = len(y)

print("=" * 80)
print("STRUCTURAL BREAK ANALYSIS")
print("=" * 80)

# ============================================================================
# Method 1: Chow Test (test at midpoint)
# ============================================================================
print("\n1. CHOW TEST (Testing for break at midpoint)")
print("-" * 80)

split_point = n // 2

# Fit full model
X_full = np.column_stack([np.ones(n), X])
params_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
y_pred_full = X_full @ params_full
sse_full = np.sum((y - y_pred_full)**2)

# Fit first half
X1 = np.column_stack([np.ones(split_point), X[:split_point]])
params1 = np.linalg.lstsq(X1, y[:split_point], rcond=None)[0]
y_pred1 = X1 @ params1
sse1 = np.sum((y[:split_point] - y_pred1)**2)

# Fit second half
X2 = np.column_stack([np.ones(n - split_point), X[split_point:]])
params2 = np.linalg.lstsq(X2, y[split_point:], rcond=None)[0]
y_pred2 = X2 @ params2
sse2 = np.sum((y[split_point:] - y_pred2)**2)

# Chow F-statistic
k = 2  # number of parameters
sse_split = sse1 + sse2
chow_f = ((sse_full - sse_split) / k) / (sse_split / (n - 2 * k))
p_value = 1 - stats.f.cdf(chow_f, k, n - 2 * k)

print(f"\nSplit point: observation {split_point} (year = {X[split_point]:.3f})")
print(f"First half slope: {params1[1]:.4f}")
print(f"Second half slope: {params2[1]:.4f}")
print(f"Chow F-statistic: {chow_f:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Conclusion: {'Significant structural break' if p_value < 0.05 else 'No significant break'}")

# ============================================================================
# Method 2: Recursive residuals and CUSUM test
# ============================================================================
print("\n" + "=" * 80)
print("2. RECURSIVE RESIDUALS & CUSUM TEST")
print("-" * 80)

# Calculate recursive residuals
recursive_resid = np.zeros(n - k)
min_obs = k + 1

for t in range(min_obs, n):
    X_t = np.column_stack([np.ones(t), X[:t]])
    y_t = y[:t]

    params_t = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
    y_pred_next = params_t[0] + params_t[1] * X[t]

    recursive_resid[t - min_obs] = y[t] - y_pred_next

# CUSUM
cusum = np.cumsum(recursive_resid) / np.std(recursive_resid)
cusum_sq = np.cumsum(recursive_resid**2) / np.sum(recursive_resid**2)

print(f"\nRecursive residuals computed for {len(recursive_resid)} observations")
print(f"CUSUM range: [{np.min(cusum):.4f}, {np.max(cusum):.4f}]")
print(f"CUSUM-SQ range: [{np.min(cusum_sq):.4f}, {np.max(cusum_sq):.4f}]")

# Approximate critical values (5% significance)
n_recursive = len(recursive_resid)
cusum_critical = 0.948 * np.sqrt(n_recursive) + 2 * 0.948
cusum_violation = np.any(np.abs(cusum) > cusum_critical)

print(f"\nCUSUM critical value (approximate): +/- {cusum_critical:.2f}")
print(f"CUSUM test: {'REJECT stability' if cusum_violation else 'Cannot reject stability'}")

# ============================================================================
# Method 3: Rolling window statistics
# ============================================================================
print("\n" + "=" * 80)
print("3. ROLLING WINDOW STATISTICS")
print("-" * 80)

window_size = 10
n_windows = n - window_size + 1

rolling_means = np.zeros(n_windows)
rolling_stds = np.zeros(n_windows)
rolling_slopes = np.zeros(n_windows)

for i in range(n_windows):
    window_y = y[i:i + window_size]
    window_x = X[i:i + window_size]

    rolling_means[i] = np.mean(window_y)
    rolling_stds[i] = np.std(window_y)

    # Linear fit in window
    params = np.polyfit(window_x, window_y, 1)
    rolling_slopes[i] = params[0]

print(f"\nWindow size: {window_size} observations")
print(f"Number of windows: {n_windows}")

print(f"\nRolling mean statistics:")
print(f"  Range: {np.min(rolling_means):.2f} to {np.max(rolling_means):.2f}")
print(f"  Change: {rolling_means[-1] - rolling_means[0]:.2f} ({(rolling_means[-1] / rolling_means[0] - 1) * 100:.1f}%)")

print(f"\nRolling slope statistics:")
print(f"  Range: {np.min(rolling_slopes):.2f} to {np.max(rolling_slopes):.2f}")
print(f"  Mean: {np.mean(rolling_slopes):.2f}")
print(f"  Std: {np.std(rolling_slopes):.2f}")

# Test if slopes change significantly
first_third_slopes = rolling_slopes[:n_windows//3]
last_third_slopes = rolling_slopes[-n_windows//3:]
t_stat, p_val = stats.ttest_ind(first_third_slopes, last_third_slopes)

print(f"\nSlope comparison (first third vs last third):")
print(f"  First third mean slope: {np.mean(first_third_slopes):.2f}")
print(f"  Last third mean slope: {np.mean(last_third_slopes):.2f}")
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
print(f"  Conclusion: {'Slopes differ significantly' if p_val < 0.05 else 'No significant difference'}")

# ============================================================================
# Method 4: Search for optimal breakpoint
# ============================================================================
print("\n" + "=" * 80)
print("4. OPTIMAL BREAKPOINT SEARCH")
print("-" * 80)

# Test all possible breakpoints (excluding first and last 5 observations)
min_segment = 5
breakpoint_sse = []
breakpoints = range(min_segment, n - min_segment)

for bp in breakpoints:
    # Fit two segments
    X1 = np.column_stack([np.ones(bp), X[:bp]])
    params1 = np.linalg.lstsq(X1, y[:bp], rcond=None)[0]
    sse1 = np.sum((y[:bp] - X1 @ params1)**2)

    X2 = np.column_stack([np.ones(n - bp), X[bp:]])
    params2 = np.linalg.lstsq(X2, y[bp:], rcond=None)[0]
    sse2 = np.sum((y[bp:] - X2 @ params2)**2)

    breakpoint_sse.append(sse1 + sse2)

optimal_idx = np.argmin(breakpoint_sse)
optimal_bp = breakpoints[optimal_idx]
min_sse = breakpoint_sse[optimal_idx]

# Calculate improvement over single model
improvement = (sse_full - min_sse) / sse_full * 100

print(f"\nOptimal breakpoint: observation {optimal_bp} (year = {X[optimal_bp]:.3f})")
print(f"SSE with break: {min_sse:.2f}")
print(f"SSE without break: {sse_full:.2f}")
print(f"Improvement: {improvement:.2f}%")

# Fit optimal two-segment model
X1_opt = np.column_stack([np.ones(optimal_bp), X[:optimal_bp]])
params1_opt = np.linalg.lstsq(X1_opt, y[:optimal_bp], rcond=None)[0]

X2_opt = np.column_stack([np.ones(n - optimal_bp), X[optimal_bp:]])
params2_opt = np.linalg.lstsq(X2_opt, y[optimal_bp:], rcond=None)[0]

print(f"\nSegment 1 (observations 0-{optimal_bp-1}):")
print(f"  Intercept: {params1_opt[0]:.2f}")
print(f"  Slope: {params1_opt[1]:.2f}")

print(f"\nSegment 2 (observations {optimal_bp}-{n-1}):")
print(f"  Intercept: {params2_opt[0]:.2f}")
print(f"  Slope: {params2_opt[1]:.2f}")

print(f"\nSlope change: {params2_opt[1] - params1_opt[1]:.2f} ({(params2_opt[1] / params1_opt[1] - 1) * 100:.1f}%)")

# Save results
results_dict = {
    'chow_test': {'f_stat': chow_f, 'p_value': p_value, 'split': split_point},
    'cusum': {'values': cusum, 'critical': cusum_critical},
    'rolling': {'means': rolling_means, 'slopes': rolling_slopes, 'window': window_size},
    'optimal_break': {'breakpoint': optimal_bp, 'sse': breakpoint_sse, 'breakpoints': list(breakpoints),
                      'params1': params1_opt, 'params2': params2_opt}
}

import pickle
with open('/workspace/eda/analyst_1/code/structural_breaks.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

print("\n" + "=" * 80)
print("Analysis complete. Results saved.")
print("=" * 80)
