"""
Sequential and Temporal Pattern Analysis
Focus: Trends across group_id, autocorrelation, time-series patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_2.csv')
df['failure_count'] = df['n_trials'] - df['r_successes']
df['logit_success_rate'] = np.log((df['r_successes'] + 0.5) / (df['failure_count'] + 0.5))

print("="*80)
print("SEQUENTIAL AND TEMPORAL PATTERN ANALYSIS")
print("="*80)

# 1. TREND ANALYSIS: Does success_rate change with group_id?
print("\n1. TREND TESTS: Success Rate vs Group ID")
print("-"*80)

# Pearson correlation (linear trend)
pearson_r, pearson_p = stats.pearsonr(df['group_id'], df['success_rate'])
print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4f}")

# Spearman correlation (monotonic trend)
spearman_r, spearman_p = spearmanr(df['group_id'], df['success_rate'])
print(f"Spearman correlation: rho={spearman_r:.4f}, p={spearman_p:.4f}")

# Kendall's tau (ordinal trend)
kendall_tau, kendall_p = kendalltau(df['group_id'], df['success_rate'])
print(f"Kendall's tau: tau={kendall_tau:.4f}, p={kendall_p:.4f}")

# Linear regression for trend
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['group_id'], df['success_rate'])
print(f"\nLinear trend: slope={slope:.6f}, p={p_value:.4f}")
print(f"R-squared: {r_value**2:.4f}")

# Mann-Kendall test for monotonic trend
def mann_kendall_test(data):
    """Mann-Kendall test for monotonic trend"""
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])

    var_s = n * (n-1) * (2*n+5) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return s, z, p_value

s_stat, z_stat, mk_p = mann_kendall_test(df['success_rate'].values)
print(f"\nMann-Kendall test: S={s_stat}, Z={z_stat:.4f}, p={mk_p:.4f}")

# Test for other variables
print("\n2. TREND TESTS: Other Variables vs Group ID")
print("-"*80)
for var in ['n_trials', 'r_successes', 'logit_success_rate']:
    r, p = spearmanr(df['group_id'], df[var])
    print(f"{var}: rho={r:.4f}, p={p:.4f}")

# 3. AUTOCORRELATION ANALYSIS
print("\n3. AUTOCORRELATION ANALYSIS")
print("-"*80)

def compute_autocorr(series, max_lag=5):
    """Compute autocorrelation for multiple lags"""
    results = {}
    n = len(series)
    mean = series.mean()
    var = series.var()

    for lag in range(1, min(max_lag + 1, n)):
        if lag < n:
            c0 = var * n
            c_lag = sum((series[:-lag] - mean) * (series[lag:] - mean))
            results[lag] = c_lag / c0
    return results

# Autocorrelation for success_rate
acf_success_rate = compute_autocorr(df['success_rate'].values, max_lag=5)
print("Autocorrelation for success_rate:")
for lag, acf in acf_success_rate.items():
    print(f"  Lag {lag}: {acf:.4f}")

# Autocorrelation for n_trials
acf_n_trials = compute_autocorr(df['n_trials'].values, max_lag=5)
print("\nAutocorrelation for n_trials:")
for lag, acf in acf_n_trials.items():
    print(f"  Lag {lag}: {acf:.4f}")

# Ljung-Box test for autocorrelation
from scipy.stats import chi2

def ljung_box_test(series, lags=3):
    """Ljung-Box test for autocorrelation"""
    n = len(series)
    acf_vals = compute_autocorr(series, max_lag=lags)

    Q = n * (n + 2) * sum([acf**2 / (n - lag) for lag, acf in acf_vals.items()])
    p_value = 1 - chi2.cdf(Q, lags)
    return Q, p_value

Q_sr, p_sr = ljung_box_test(df['success_rate'].values, lags=3)
print(f"\nLjung-Box test (success_rate, 3 lags): Q={Q_sr:.4f}, p={p_sr:.4f}")

# 4. RUNS TEST (randomness vs patterns)
print("\n4. RUNS TEST FOR RANDOMNESS")
print("-"*80)

def runs_test(series):
    """Runs test for randomness around median"""
    median = np.median(series)
    runs = [series[0] > median]

    for val in series[1:]:
        if (val > median) != runs[-1]:
            runs.append(val > median)

    n_runs = len(runs)
    n_above = sum(series > median)
    n_below = sum(series <= median)
    n = len(series)

    expected_runs = (2 * n_above * n_below) / n + 1
    var_runs = (2 * n_above * n_below * (2 * n_above * n_below - n)) / (n**2 * (n - 1))

    z = (n_runs - expected_runs) / np.sqrt(var_runs)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return n_runs, expected_runs, z, p_value

n_runs, exp_runs, z_runs, p_runs = runs_test(df['success_rate'].values)
print(f"Success rate runs test:")
print(f"  Observed runs: {n_runs}")
print(f"  Expected runs: {exp_runs:.2f}")
print(f"  Z-score: {z_runs:.4f}")
print(f"  P-value: {p_runs:.4f}")
print(f"  Interpretation: {'Random' if p_runs > 0.05 else 'Non-random pattern'}")

# 5. FIRST-HALF vs SECOND-HALF COMPARISON
print("\n5. FIRST-HALF vs SECOND-HALF COMPARISON")
print("-"*80)

first_half = df.iloc[:6]
second_half = df.iloc[6:]

print("First half (groups 1-6):")
print(f"  Mean success_rate: {first_half['success_rate'].mean():.4f}")
print(f"  Median n_trials: {first_half['n_trials'].median():.1f}")

print("\nSecond half (groups 7-12):")
print(f"  Mean success_rate: {second_half['success_rate'].mean():.4f}")
print(f"  Median n_trials: {second_half['n_trials'].median():.1f}")

# Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(first_half['success_rate'], second_half['success_rate'], alternative='two-sided')
print(f"\nMann-Whitney U test (success_rate): U={u_stat}, p={u_p:.4f}")

# 6. SEGMENTED ANALYSIS (thirds)
print("\n6. SEGMENTED ANALYSIS (by thirds)")
print("-"*80)

segment1 = df.iloc[:4]
segment2 = df.iloc[4:8]
segment3 = df.iloc[8:]

print(f"Segment 1 (groups 1-4): mean SR = {segment1['success_rate'].mean():.4f}")
print(f"Segment 2 (groups 5-8): mean SR = {segment2['success_rate'].mean():.4f}")
print(f"Segment 3 (groups 9-12): mean SR = {segment3['success_rate'].mean():.4f}")

# Kruskal-Wallis test
h_stat, h_p = stats.kruskal(segment1['success_rate'], segment2['success_rate'], segment3['success_rate'])
print(f"\nKruskal-Wallis test: H={h_stat:.4f}, p={h_p:.4f}")

print("\n" + "="*80)
print("SEQUENTIAL ANALYSIS COMPLETE")
print("="*80)
