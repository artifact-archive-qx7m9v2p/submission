"""
Compute summary metrics from SBC results
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

# Load results
results_file = Path('/workspace/experiments/experiment_1/simulation_based_validation/code/sbc_results.json')
with open(results_file, 'r') as f:
    results = json.load(f)

PARAMS = ['alpha', 'beta', 'c', 'nu', 'sigma']

print("="*70)
print("SIMULATION-BASED CALIBRATION - SUMMARY METRICS")
print("="*70)
print()

# 1. RANK HISTOGRAM UNIFORMITY
print("1. RANK HISTOGRAM UNIFORMITY (Chi-square tests)")
print("-" * 70)
n_bins = 20
expected_per_bin = len(results['ranks'][PARAMS[0]]) / n_bins

rank_results = {}
for param in PARAMS:
    ranks = results['ranks'][param]
    counts, _ = np.histogram(ranks, bins=n_bins)
    chi2_stat, p_value = stats.chisquare(counts)
    rank_results[param] = {'chi2': chi2_stat, 'p_value': p_value}

    result = 'PASS' if p_value >= 0.05 else ('WARN' if p_value >= 0.01 else 'FAIL')
    print(f"  {param:8s}: χ² = {chi2_stat:6.2f}, p = {p_value:.4f} [{result}]")

print()

# 2. Z-SCORE ANALYSIS (Bias detection)
print("2. Z-SCORE ANALYSIS (Bias Detection)")
print("-" * 70)
z_results = {}
for param in PARAMS:
    z_scores = np.array(results['z_scores'][param])
    mean_z = np.mean(z_scores)
    std_z = np.std(z_scores)

    # Test if mean is significantly different from 0
    t_stat, t_pvalue = stats.ttest_1samp(z_scores, 0)

    z_results[param] = {
        'mean': mean_z,
        'std': std_z,
        't_stat': t_stat,
        't_pvalue': t_pvalue
    }

    if abs(mean_z) > 0.3:
        result = 'BIASED'
    elif abs(mean_z) > 0.2:
        result = 'SLIGHT BIAS'
    else:
        result = 'UNBIASED'

    print(f"  {param:8s}: Mean Z = {mean_z:7.4f}, SD = {std_z:.4f}, t-test p = {t_pvalue:.4f} [{result}]")

print()

# 3. COVERAGE CALIBRATION
print("3. COVERAGE CALIBRATION")
print("-" * 70)
coverage_results = {}
for param in PARAMS:
    cov_90 = np.mean(results['in_90_CI'][param]) * 100
    cov_95 = np.mean(results['in_95_CI'][param]) * 100

    # Check if within acceptable range
    pass_90 = 88 <= cov_90 <= 92
    pass_95 = 93 <= cov_95 <= 97

    coverage_results[param] = {
        'coverage_90': cov_90,
        'coverage_95': cov_95,
        'pass_90': pass_90,
        'pass_95': pass_95
    }

    result = 'PASS' if (pass_90 and pass_95) else 'FAIL'
    print(f"  {param:8s}: 90% CI = {cov_90:5.1f}% (nominal 90%), 95% CI = {cov_95:5.1f}% (nominal 95%) [{result}]")

print()

# 4. PARAMETER RECOVERY
print("4. PARAMETER RECOVERY (Shrinkage & Bias)")
print("-" * 70)
recovery_results = {}
for param in PARAMS:
    true_vals = np.array(results['true_values'][param])
    post_means = np.array(results['posterior_means'][param])

    # Correlation (shrinkage)
    corr = np.corrcoef(true_vals, post_means)[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((true_vals - post_means)**2))

    # Bias
    bias = np.mean(post_means - true_vals)

    # Relative bias (as % of prior SD)
    prior_sd = {
        'alpha': 0.5,
        'beta': 0.2,
        'c': np.sqrt(2 / 4),  # gamma(2, 2) has var = shape/rate^2
        'nu': np.sqrt(2 / 0.01),  # gamma(2, 0.1)
        'sigma': 0.15
    }
    rel_bias = bias / prior_sd[param]

    recovery_results[param] = {
        'correlation': corr,
        'rmse': rmse,
        'bias': bias,
        'rel_bias': rel_bias
    }

    result = 'GOOD' if (corr > 0.9 and abs(rel_bias) < 0.2) else ('OK' if corr > 0.7 else 'POOR')
    print(f"  {param:8s}: r = {corr:.4f}, RMSE = {rmse:.5f}, Bias = {bias:8.5f} (rel: {rel_bias:.2f}σ) [{result}]")

print()

# 5. CONVERGENCE DIAGNOSTICS
print("5. CONVERGENCE DIAGNOSTICS")
print("-" * 70)
accept_rates = np.array(results['convergence']['accept_rate'])
ess_values = np.array(results['convergence']['ess'])

print(f"  Acceptance Rate: Mean = {np.mean(accept_rates):.3f}, SD = {np.std(accept_rates):.3f}")
print(f"                   Range = [{np.min(accept_rates):.3f}, {np.max(accept_rates):.3f}]")
print(f"                   Optimal range: [0.20, 0.40]")

print(f"  Effective SS:    Mean = {np.mean(ess_values):.0f}, Min = {np.min(ess_values):.0f}")
print(f"                   All > 400: {np.all(ess_values > 400)}")

conv_result = {
    'mean_accept': float(np.mean(accept_rates)),
    'mean_ess': float(np.mean(ess_values)),
    'min_ess': float(np.min(ess_values))
}

print()

# 6. OVERALL ASSESSMENT
print("6. OVERALL ASSESSMENT")
print("-" * 70)

# Check all criteria
all_ranks_pass = all(rank_results[p]['p_value'] >= 0.01 for p in PARAMS)
all_unbiased = all(abs(z_results[p]['mean']) < 0.3 for p in PARAMS)
all_coverage_pass = all(coverage_results[p]['pass_90'] and coverage_results[p]['pass_95'] for p in PARAMS)
all_recovery_good = all(recovery_results[p]['correlation'] > 0.7 for p in PARAMS)
convergence_good = np.mean(accept_rates) > 0.15 and np.all(ess_values > 200)

print(f"  Rank uniformity:     {'PASS' if all_ranks_pass else 'FAIL'}")
print(f"  Absence of bias:     {'PASS' if all_unbiased else 'FAIL'}")
print(f"  Coverage calibration: {'PASS' if all_coverage_pass else 'FAIL'}")
print(f"  Parameter recovery:  {'PASS' if all_recovery_good else 'FAIL'}")
print(f"  Convergence:         {'PASS' if convergence_good else 'FAIL'}")

print()

overall_pass = (all_ranks_pass and all_unbiased and all_coverage_pass and
                all_recovery_good and convergence_good)

if overall_pass:
    print("  FINAL DECISION: PASS")
    print("  Model is well-calibrated and ready for real data.")
else:
    print("  FINAL DECISION: FAIL")
    print("  Model has calibration issues - see details above.")

print()
print("="*70)

# Save summary
summary = {
    'rank_uniformity': rank_results,
    'z_scores': z_results,
    'coverage': coverage_results,
    'recovery': recovery_results,
    'convergence': conv_result,
    'overall': {
        'rank_uniformity_pass': all_ranks_pass,
        'unbiased': all_unbiased,
        'coverage_pass': all_coverage_pass,
        'recovery_good': all_recovery_good,
        'convergence_good': convergence_good,
        'final_decision': 'PASS' if overall_pass else 'FAIL'
    }
}

summary_file = Path('/workspace/experiments/experiment_1/simulation_based_validation/code/summary_metrics.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {summary_file}")
