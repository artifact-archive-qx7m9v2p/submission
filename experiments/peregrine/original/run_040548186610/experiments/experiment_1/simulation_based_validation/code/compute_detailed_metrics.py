#!/usr/bin/env python3
"""
Compute detailed SBC metrics for the recovery report
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
RESULTS_DIR = BASE_DIR / "results"

# Load results
sbc_data = {}
for param in ['beta_0', 'beta_1', 'beta_2', 'phi']:
    sbc_data[param] = pd.read_csv(RESULTS_DIR / f'sbc_results_{param}.csv')

convergence_stats = pd.read_csv(RESULTS_DIR / 'convergence_stats.csv')

with open(RESULTS_DIR / 'summary_stats.json', 'r') as f:
    summary_stats = json.load(f)

# Total posterior samples per simulation (2 chains × 500 samples = 1000)
L = 1000

# Compute detailed metrics for each parameter
detailed_metrics = {}

for param in ['beta_0', 'beta_1', 'beta_2', 'phi']:
    df = sbc_data[param]

    # Ranks
    ranks = df['rank'].values
    true_vals = df['true_value'].values
    post_means = df['posterior_mean'].values
    post_medians = df['posterior_median'].values
    q025 = df['q025'].values
    q975 = df['q975'].values

    # Parameter recovery metrics
    bias = np.mean(post_means - true_vals)
    abs_bias = np.mean(np.abs(post_means - true_vals))
    rmse = np.sqrt(np.mean((post_means - true_vals)**2))
    rel_bias = bias / np.mean(true_vals)

    # Shrinkage/slope of regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_vals, post_means)

    # Coverage
    in_interval = (true_vals >= q025) & (true_vals <= q975)
    coverage_95 = np.mean(in_interval) * 100

    # Z-scores (standardized errors)
    post_sd = (q975 - q025) / (2 * 1.96)
    z_scores = (post_means - true_vals) / post_sd
    z_mean = np.mean(z_scores)
    z_sd = np.std(z_scores)

    # Rank uniformity tests
    n_bins = 50
    counts, bin_edges = np.histogram(ranks, bins=n_bins, range=(0, L))
    expected = np.full(len(counts), len(ranks) / n_bins)

    # Chi-square test
    chi2_stat = np.sum((counts - expected)**2 / expected)
    chi2_df = len(counts) - 1
    chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, chi2_df)

    # Kolmogorov-Smirnov test
    normalized_ranks = ranks / L
    ks_stat, ks_pvalue = stats.kstest(normalized_ranks, 'uniform')

    # Rank statistics
    rank_mean = np.mean(ranks)
    rank_expected = L / 2
    rank_std = np.std(ranks)
    rank_expected_std = L / np.sqrt(12)

    detailed_metrics[param] = {
        'bias': float(bias),
        'abs_bias': float(abs_bias),
        'rmse': float(rmse),
        'rel_bias': float(rel_bias),
        'regression_slope': float(slope),
        'regression_intercept': float(intercept),
        'regression_r2': float(r_value**2),
        'coverage_95': float(coverage_95),
        'z_mean': float(z_mean),
        'z_sd': float(z_sd),
        'chi2_stat': float(chi2_stat),
        'chi2_pvalue': float(chi2_pvalue),
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'rank_mean': float(rank_mean),
        'rank_expected': float(rank_expected),
        'rank_std': float(rank_std),
        'rank_expected_std': float(rank_expected_std)
    }

# Save detailed metrics
with open(RESULTS_DIR / 'detailed_metrics.json', 'w') as f:
    json.dump(detailed_metrics, f, indent=2)

print("Detailed metrics computed and saved.")

# Print summary
print("\n" + "="*80)
print("DETAILED SBC METRICS SUMMARY")
print("="*80)

for param in ['beta_0', 'beta_1', 'beta_2', 'phi']:
    print(f"\n{param}:")
    m = detailed_metrics[param]
    print(f"  Bias: {m['bias']:.4f}")
    print(f"  RMSE: {m['rmse']:.4f}")
    print(f"  Coverage (95%): {m['coverage_95']:.1f}%")
    print(f"  Rank χ² p-value: {m['chi2_pvalue']:.4f}")
    print(f"  Rank KS p-value: {m['ks_pvalue']:.4f}")
    print(f"  Regression slope: {m['regression_slope']:.3f}")
