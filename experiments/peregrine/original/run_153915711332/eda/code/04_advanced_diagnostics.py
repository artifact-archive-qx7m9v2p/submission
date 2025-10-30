"""
Advanced Diagnostic Analysis
=============================
Purpose: Changepoint detection, autocorrelation, and model hypothesis testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data.csv')
C = data['C'].values
year = data['year'].values

# Create advanced diagnostics figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. ACF plot
ax1 = fig.add_subplot(gs[0, 0])
max_lag = min(20, len(C) // 2)
acf_values = [1.0]  # ACF at lag 0 is always 1
for lag in range(1, max_lag + 1):
    acf = np.corrcoef(C[:-lag], C[lag:])[0, 1]
    acf_values.append(acf)

ax1.stem(range(len(acf_values)), acf_values, basefmt=' ')
ax1.axhline(0, color='black', linewidth=0.8)
# Confidence bands
conf_level = 1.96 / np.sqrt(len(C))
ax1.axhline(conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.axhline(-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_xlabel('Lag', fontsize=11)
ax1.set_ylabel('ACF', fontsize=11)
ax1.set_title('Autocorrelation Function', fontsize=12)
ax1.grid(True, alpha=0.3)

# 2. PACF plot (approximate)
ax2 = fig.add_subplot(gs[0, 1])
from scipy.linalg import toeplitz, solve
def compute_pacf(x, max_lag):
    pacf = [1.0]
    for k in range(1, max_lag + 1):
        # Compute ACF up to lag k
        r = [1.0]
        for lag in range(1, k + 1):
            r.append(np.corrcoef(x[:-lag], x[lag:])[0, 1])

        if k == 1:
            pacf.append(r[1])
        else:
            R = toeplitz(r[:-1])
            try:
                phi = solve(R, r[1:])
                pacf.append(phi[-1])
            except:
                pacf.append(0)
    return pacf

pacf_values = compute_pacf(C, max_lag)
ax2.stem(range(len(pacf_values)), pacf_values, basefmt=' ')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axhline(conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.axhline(-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_xlabel('Lag', fontsize=11)
ax2.set_ylabel('PACF', fontsize=11)
ax2.set_title('Partial Autocorrelation Function', fontsize=12)
ax2.grid(True, alpha=0.3)

# 3. Differenced series (first differences)
ax3 = fig.add_subplot(gs[0, 2])
diff_C = np.diff(C)
ax3.plot(year[1:], diff_C, marker='o', markersize=5, linewidth=1, alpha=0.7, color='steelblue')
ax3.axhline(0, color='red', linestyle='--', linewidth=1)
ax3.axhline(np.mean(diff_C), color='orange', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(diff_C):.2f}')
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('First Difference', fontsize=11)
ax3.set_title('First Differences: C(t) - C(t-1)', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Squared residuals over time (from linear model)
ax4 = fig.add_subplot(gs[1, 0])
slope, intercept, _, _, _ = stats.linregress(year, C)
residuals = C - (slope * year + intercept)
squared_res = residuals**2
ax4.scatter(year, squared_res, s=50, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
# Add smoothed trend
from scipy.signal import savgol_filter
sort_idx = np.argsort(year)
window_len = min(11, len(squared_res)//3*2+1)
if window_len % 2 == 0:
    window_len += 1
smoothed = savgol_filter(squared_res[sort_idx], window_length=window_len, polyorder=2)
ax4.plot(year[sort_idx], smoothed, 'red', linewidth=2, label='Smoothed trend')
ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Squared Residuals', fontsize=11)
ax4.set_title('Squared Residuals Over Time', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Rolling statistics
ax5 = fig.add_subplot(gs[1, 1])
window_size = 8  # Rolling window
rolling_mean = []
rolling_std = []
centers = []

for i in range(window_size, len(C) + 1):
    window_data = C[i-window_size:i]
    rolling_mean.append(np.mean(window_data))
    rolling_std.append(np.std(window_data, ddof=1))
    centers.append(year[i-1])

ax5.plot(centers, rolling_mean, 'b-', linewidth=2, label='Rolling Mean')
ax5_twin = ax5.twinx()
ax5_twin.plot(centers, rolling_std, 'r-', linewidth=2, label='Rolling Std')
ax5.set_xlabel('Year', fontsize=11)
ax5.set_ylabel('Rolling Mean', fontsize=11, color='b')
ax5_twin.set_ylabel('Rolling Std', fontsize=11, color='r')
ax5.set_title(f'Rolling Statistics (window={window_size})', fontsize=12)
ax5.tick_params(axis='y', labelcolor='b')
ax5_twin.tick_params(axis='y', labelcolor='r')
ax5.grid(True, alpha=0.3)

# 6. Cumulative sum plot (CUSUM for changepoint detection)
ax6 = fig.add_subplot(gs[1, 2])
mean_C = np.mean(C)
cusum = np.cumsum(C - mean_C)
ax6.plot(year, cusum, linewidth=2, color='steelblue')
ax6.axhline(0, color='red', linestyle='--', linewidth=1)
ax6.set_xlabel('Year', fontsize=11)
ax6.set_ylabel('Cumulative Sum', fontsize=11)
ax6.set_title('CUSUM (Changepoint Detection)', fontsize=12)
ax6.grid(True, alpha=0.3)

# Find approximate changepoint
abs_cusum = np.abs(cusum)
changepoint_idx = np.argmax(abs_cusum)
ax6.axvline(year[changepoint_idx], color='orange', linestyle='--', linewidth=2,
            label=f'Max CUSUM at year={year[changepoint_idx]:.2f}')
ax6.legend()

# 7. Growth rate over time
ax7 = fig.add_subplot(gs[2, 0])
growth_rate = np.diff(C) / C[:-1] * 100  # Percentage growth
ax7.scatter(year[1:], growth_rate, s=50, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
ax7.axhline(0, color='red', linestyle='--', linewidth=1)
ax7.axhline(np.median(growth_rate), color='orange', linestyle='--', linewidth=1.5,
            label=f'Median={np.median(growth_rate):.1f}%')
ax7.set_xlabel('Year', fontsize=11)
ax7.set_ylabel('Growth Rate (%)', fontsize=11)
ax7.set_title('Period-over-Period Growth Rate', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Log-scale mean-variance plot
ax8 = fig.add_subplot(gs[2, 1])
# Create bins
n_bins = 5
bins = np.linspace(year.min(), year.max(), n_bins + 1)
bin_means = []
bin_vars = []

for i in range(n_bins):
    mask = (year >= bins[i]) & (year < bins[i+1])
    if mask.sum() > 1:
        bin_means.append(np.mean(C[mask]))
        bin_vars.append(np.var(C[mask], ddof=1))

ax8.scatter(bin_means, bin_vars, s=200, alpha=0.7, color='steelblue',
            edgecolors='black', linewidth=2)
# Add reference lines
mean_range = np.array([min(bin_means), max(bin_means)])
ax8.plot(mean_range, mean_range, 'r--', linewidth=2, alpha=0.7, label='Var = Mean (Poisson)')
ax8.plot(mean_range, 2*mean_range, 'g--', linewidth=2, alpha=0.7, label='Var = 2*Mean')
ax8.set_xlabel('Bin Mean', fontsize=11)
ax8.set_ylabel('Bin Variance', fontsize=11)
ax8.set_title('Mean-Variance Relationship', fontsize=12)
ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Count vs lagged count (check for momentum)
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(C[:-1], C[1:], s=60, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
# Add diagonal line
min_c, max_c = C.min(), C.max()
ax9.plot([min_c, max_c], [min_c, max_c], 'r--', linewidth=2, alpha=0.7, label='C(t+1)=C(t)')
# Add regression line
slope_lag, intercept_lag, r_lag, _, _ = stats.linregress(C[:-1], C[1:])
x_line = np.array([min_c, max_c])
y_line = slope_lag * x_line + intercept_lag
ax9.plot(x_line, y_line, 'g-', linewidth=2, label=f'Linear: R²={r_lag**2:.3f}')
ax9.set_xlabel('C(t)', fontsize=11)
ax9.set_ylabel('C(t+1)', fontsize=11)
ax9.set_title('Lag-1 Scatter Plot', fontsize=12)
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.suptitle('Advanced Diagnostic Analysis', fontsize=16, y=0.995)
plt.savefig('/workspace/eda/visualizations/03_advanced_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

print("="*80)
print("ADVANCED DIAGNOSTIC ANALYSIS")
print("="*80)

# 1. Autocorrelation analysis
print("\n1. AUTOCORRELATION STRUCTURE")
print("-"*80)
print(f"ACF at lag 1: {acf_values[1]:.4f}")
print(f"ACF at lag 2: {acf_values[2]:.4f}")
print(f"ACF at lag 3: {acf_values[3]:.4f}")
print(f"\nSignificant lags (|ACF| > {conf_level:.3f}):")
sig_lags = [i for i, acf in enumerate(acf_values[1:], 1) if abs(acf) > conf_level]
print(f"  Lags: {sig_lags[:10] if len(sig_lags) > 10 else sig_lags}")
if len(sig_lags) > 5:
    print(f"  → Strong autocorrelation detected")
    print(f"  → Time series models (ARIMA) may be appropriate")

# 2. Stationarity
print("\n2. STATIONARITY ANALYSIS")
print("-"*80)
print(f"Original series:")
print(f"  Mean: {np.mean(C):.2f}")
print(f"  Variance: {np.var(C, ddof=1):.2f}")
print(f"\nFirst differences:")
print(f"  Mean: {np.mean(diff_C):.2f}")
print(f"  Variance: {np.var(diff_C, ddof=1):.2f}")
print(f"  Std Dev: {np.std(diff_C, ddof=1):.2f}")

# Check if differencing helps
if np.std(diff_C, ddof=1) < np.std(C, ddof=1):
    print(f"\n  → Differencing reduces variance")
    print(f"  → Series may be integrated (I component in ARIMA)")

# 3. Changepoint detection
print("\n3. CHANGEPOINT DETECTION")
print("-"*80)
print(f"CUSUM analysis:")
print(f"  Maximum |CUSUM| at index: {changepoint_idx}")
print(f"  Year: {year[changepoint_idx]:.3f}")
print(f"  Count at changepoint: {C[changepoint_idx]}")

# Compare periods before and after potential changepoint
before = C[:changepoint_idx+1]
after = C[changepoint_idx+1:]

if len(before) > 1 and len(after) > 1:
    print(f"\nBefore changepoint:")
    print(f"  Mean: {np.mean(before):.2f}")
    print(f"  Std: {np.std(before, ddof=1):.2f}")
    print(f"After changepoint:")
    print(f"  Mean: {np.mean(after):.2f}")
    print(f"  Std: {np.std(after, ddof=1):.2f}")

    # T-test for mean difference
    t_stat, p_val = stats.ttest_ind(before, after)
    print(f"\nT-test for equal means:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_val:.4f}")
    print(f"  Conclusion: {'Means differ' if p_val < 0.05 else 'Cannot reject equal means'}")

# 4. Growth characteristics
print("\n4. GROWTH RATE ANALYSIS")
print("-"*80)
print(f"Period-over-period growth:")
print(f"  Mean: {np.mean(growth_rate):.2f}%")
print(f"  Median: {np.median(growth_rate):.2f}%")
print(f"  Std Dev: {np.std(growth_rate, ddof=1):.2f}%")
print(f"  Min: {np.min(growth_rate):.2f}%")
print(f"  Max: {np.max(growth_rate):.2f}%")

# Check for accelerating growth
first_half_growth = growth_rate[:len(growth_rate)//2]
second_half_growth = growth_rate[len(growth_rate)//2:]
print(f"\nFirst half mean growth: {np.mean(first_half_growth):.2f}%")
print(f"Second half mean growth: {np.mean(second_half_growth):.2f}%")

if np.mean(second_half_growth) > np.mean(first_half_growth) * 1.5:
    print(f"  → Accelerating growth pattern detected")
    print(f"  → Consider exponential or polynomial models")

# 5. Lag-1 relationship
print("\n5. LAG-1 DEPENDENCE")
print("-"*80)
print(f"Correlation C(t) vs C(t+1): {r_lag:.4f}")
print(f"R-squared: {r_lag**2:.4f}")
print(f"Regression: C(t+1) = {slope_lag:.3f} * C(t) + {intercept_lag:.3f}")

if r_lag > 0.8:
    print(f"\n  → Strong positive momentum")
    print(f"  → Past values strongly predictive of future")

print("\n" + "="*80)
print("Visualization saved: /workspace/eda/visualizations/03_advanced_diagnostics.png")
print("="*80)
