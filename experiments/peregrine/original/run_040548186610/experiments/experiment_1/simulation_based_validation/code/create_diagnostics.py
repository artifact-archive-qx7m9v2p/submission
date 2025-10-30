#!/usr/bin/env python3
"""
Create diagnostic plots for Simulation-Based Calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Load results
print("Loading SBC results...")

sbc_data = {}
for param in ['beta_0', 'beta_1', 'beta_2', 'phi']:
    sbc_data[param] = pd.read_csv(RESULTS_DIR / f'sbc_results_{param}.csv')

convergence_stats = pd.read_csv(RESULTS_DIR / 'convergence_stats.csv')

with open(RESULTS_DIR / 'summary_stats.json', 'r') as f:
    summary_stats = json.load(f)

n_sims = summary_stats['n_successful']
n_samples = len(sbc_data['beta_0']['rank'].iloc[0:1]) if n_sims > 0 else 0

# Approximate number of posterior samples per simulation
# Assuming 4 chains * 2000 samples = 8000 samples
L = 8000

print(f"Number of successful simulations: {n_sims}")
print(f"Expected posterior samples per simulation: {L}")

# ==============================================================================
# 1. RANK HISTOGRAMS (Primary SBC diagnostic)
# ==============================================================================

print("\nCreating rank histograms...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

param_names = {
    'beta_0': r'$\beta_0$ (Intercept)',
    'beta_1': r'$\beta_1$ (Linear)',
    'beta_2': r'$\beta_2$ (Quadratic)',
    'phi': r'$\phi$ (Dispersion)'
}

param_list = ['beta_0', 'beta_1', 'beta_2', 'phi']

for idx, param in enumerate(param_list):
    ax = axes[idx]

    ranks = sbc_data[param]['rank'].values

    # Number of bins = L + 1 (for ranks 0 to L)
    n_bins = min(L + 1, 50)  # Cap at 50 bins for visibility

    # Plot histogram
    counts, bins, patches = ax.hist(ranks, bins=n_bins, edgecolor='black',
                                      alpha=0.7, color='steelblue')

    # Expected uniform distribution
    expected_count = n_sims / n_bins
    ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
               label=f'Expected (uniform): {expected_count:.1f}')

    # Add confidence band for uniform (based on binomial distribution)
    # 95% CI for binomial: p ± 1.96 * sqrt(p(1-p)/n)
    p = 1 / n_bins
    se = np.sqrt(p * (1 - p) / n_sims) * n_sims
    ci_lower = expected_count - 1.96 * se
    ci_upper = expected_count + 1.96 * se
    ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='red',
               label='95% CI for uniform')

    # Chi-square test for uniformity
    expected = np.full(len(counts), expected_count)
    chi2_stat = np.sum((counts - expected)**2 / expected)
    df = len(counts) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{param_names[param]}\nχ² = {chi2_stat:.2f}, p = {p_value:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sbc_rank_histograms.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'sbc_rank_histograms.png'}")
plt.close()

# ==============================================================================
# 2. PARAMETER RECOVERY PLOTS (Shrinkage plots)
# ==============================================================================

print("\nCreating parameter recovery plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, param in enumerate(param_list):
    ax = axes[idx]

    true_vals = sbc_data[param]['true_value'].values
    post_means = sbc_data[param]['posterior_mean'].values

    # Scatter plot with identity line
    ax.scatter(true_vals, post_means, alpha=0.5, s=50, color='steelblue')
    ax.plot([true_vals.min(), true_vals.max()],
            [true_vals.min(), true_vals.max()],
            'r--', linewidth=2, label='Perfect recovery')

    # Add regression line
    z = np.polyfit(true_vals, post_means, 1)
    p = np.poly1d(z)
    ax.plot(true_vals, p(true_vals), 'g-', linewidth=2, alpha=0.7,
            label=f'Fitted: y = {z[0]:.3f}x + {z[1]:.3f}')

    # Calculate bias
    bias = np.mean(post_means - true_vals)
    rmse = np.sqrt(np.mean((post_means - true_vals)**2))

    ax.set_xlabel(f'True {param_names[param]}', fontsize=11)
    ax.set_ylabel(f'Posterior Mean {param_names[param]}', fontsize=11)
    ax.set_title(f'{param_names[param]}\nBias = {bias:.4f}, RMSE = {rmse:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sbc_parameter_recovery.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'sbc_parameter_recovery.png'}")
plt.close()

# ==============================================================================
# 3. COVERAGE PLOTS (Calibration of uncertainty intervals)
# ==============================================================================

print("\nCreating coverage plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, param in enumerate(param_list):
    ax = axes[idx]

    true_vals = sbc_data[param]['true_value'].values
    q025 = sbc_data[param]['q025'].values
    q975 = sbc_data[param]['q975'].values

    # Check coverage
    in_interval = (true_vals >= q025) & (true_vals <= q975)
    coverage = np.mean(in_interval) * 100

    # Sort by true value for better visualization
    sort_idx = np.argsort(true_vals)
    true_sorted = true_vals[sort_idx]
    q025_sorted = q025[sort_idx]
    q975_sorted = q975[sort_idx]
    in_interval_sorted = in_interval[sort_idx]

    # Plot intervals
    x = np.arange(len(true_vals))
    for i in x:
        color = 'green' if in_interval_sorted[i] else 'red'
        ax.plot([i, i], [q025_sorted[i], q975_sorted[i]], color=color,
                alpha=0.5, linewidth=1)

    # Plot true values
    ax.scatter(x, true_sorted, color='blue', s=20, zorder=10,
               label='True value', alpha=0.7)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Simulation (sorted by true value)', fontsize=11)
    ax.set_ylabel(f'{param_names[param]}', fontsize=11)
    ax.set_title(f'{param_names[param]}\n95% Coverage: {coverage:.1f}% (Expected: 95%)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sbc_coverage.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'sbc_coverage.png'}")
plt.close()

# ==============================================================================
# 4. Z-SCORE PLOTS (Standardized error)
# ==============================================================================

print("\nCreating z-score plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, param in enumerate(param_list):
    ax = axes[idx]

    true_vals = sbc_data[param]['true_value'].values
    post_means = sbc_data[param]['posterior_mean'].values
    post_medians = sbc_data[param]['posterior_median'].values

    # Approximate posterior SD from quantiles
    q025 = sbc_data[param]['q025'].values
    q975 = sbc_data[param]['q975'].values
    post_sd = (q975 - q025) / (2 * 1.96)  # Approximation

    # Compute z-scores
    z_scores = (post_means - true_vals) / post_sd

    # Histogram of z-scores
    ax.hist(z_scores, bins=30, edgecolor='black', alpha=0.7,
            color='steelblue', density=True)

    # Overlay standard normal
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2,
            label='N(0,1)')

    # Statistics
    mean_z = np.mean(z_scores)
    sd_z = np.std(z_scores)

    ax.set_xlabel('Z-score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{param_names[param]}\nMean = {mean_z:.3f}, SD = {sd_z:.3f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-4, 4])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sbc_zscores.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'sbc_zscores.png'}")
plt.close()

# ==============================================================================
# 5. COMPUTATIONAL DIAGNOSTICS
# ==============================================================================

print("\nCreating computational diagnostics plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# R-hat distribution
ax = axes[0, 0]
ax.hist(convergence_stats['max_rhat'], bins=30, edgecolor='black',
        alpha=0.7, color='steelblue')
ax.axvline(1.01, color='red', linestyle='--', linewidth=2,
           label='Threshold (1.01)')
ax.axvline(1.1, color='orange', linestyle='--', linewidth=2,
           label='Warning (1.1)')
ax.set_xlabel(r'Max $\hat{R}$', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Convergence Diagnostic (R-hat)\nMean: {convergence_stats["max_rhat"].mean():.4f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ESS distribution
ax = axes[0, 1]
ax.hist(convergence_stats['min_ess'], bins=30, edgecolor='black',
        alpha=0.7, color='steelblue')
ax.axvline(400, color='red', linestyle='--', linewidth=2,
           label='Good (>400)')
ax.axvline(100, color='orange', linestyle='--', linewidth=2,
           label='Minimum (>100)')
ax.set_xlabel('Min ESS', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Effective Sample Size\nMean: {convergence_stats["min_ess"].mean():.0f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Acceptance rate distribution
ax = axes[1, 0]
ax.hist(convergence_stats['mean_acceptance'], bins=30, edgecolor='black',
        alpha=0.7, color='steelblue')
ax.axvline(0.234, color='red', linestyle='--', linewidth=2,
           label='Optimal (0.234)')
ax.axvspan(0.15, 0.5, alpha=0.2, color='green',
           label='Acceptable range')
ax.set_xlabel('Mean Acceptance Rate', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'MCMC Acceptance Rate\nMean: {convergence_stats["mean_acceptance"].mean():.3f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Convergence summary
ax = axes[1, 1]
ax.axis('off')

convergence_rate = summary_stats['convergence_rate']
n_converged = int(convergence_rate * summary_stats['n_successful'] / 100)

summary_text = f"""
COMPUTATIONAL HEALTH SUMMARY
{'='*40}

Success Rate: {summary_stats['success_rate']:.1f}%
  ({summary_stats['n_successful']}/{summary_stats['n_simulations']} simulations)

Convergence Rate: {convergence_rate:.1f}%
  ({n_converged}/{summary_stats['n_successful']} converged)

Average Diagnostics:
  R̂ (max):        {summary_stats['mean_rhat']:.4f}
  ESS (min):      {summary_stats['mean_ess']:.0f}
  Acceptance:     {summary_stats['mean_acceptance']:.3f}

Status: {'PASS' if convergence_rate > 80 and summary_stats['mean_rhat'] < 1.1 else 'WARNING' if convergence_rate > 60 else 'FAIL'}
"""

ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sbc_computational_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'sbc_computational_diagnostics.png'}")
plt.close()

# ==============================================================================
# 6. COMPREHENSIVE RANK STATISTIC ANALYSIS
# ==============================================================================

print("\nCreating comprehensive rank analysis...")

# Calculate detailed statistics for each parameter
rank_statistics = {}

for param in param_list:
    ranks = sbc_data[param]['rank'].values
    n_bins = min(L + 1, 50)

    # Chi-square test
    counts, _ = np.histogram(ranks, bins=n_bins)
    expected = np.full(len(counts), n_sims / n_bins)
    chi2_stat = np.sum((counts - expected)**2 / expected)
    df = len(counts) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    # ECDF uniformity test (Kolmogorov-Smirnov)
    normalized_ranks = ranks / L  # Normalize to [0, 1]
    ks_stat, ks_pvalue = stats.kstest(normalized_ranks, 'uniform')

    rank_statistics[param] = {
        'chi2_stat': chi2_stat,
        'chi2_pvalue': p_value,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'rank_mean': np.mean(ranks),
        'rank_expected': L / 2,
        'rank_std': np.std(ranks),
        'rank_expected_std': L / np.sqrt(12)
    }

# Create summary table plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

table_data = []
headers = ['Parameter', 'χ² Stat', 'χ² p-val', 'KS Stat', 'KS p-val',
           'Rank Mean', 'Expected', 'Status']

for param in param_list:
    stats_dict = rank_statistics[param]

    # Determine status
    if stats_dict['chi2_pvalue'] > 0.05 and stats_dict['ks_pvalue'] > 0.05:
        status = 'PASS'
        color = 'lightgreen'
    elif stats_dict['chi2_pvalue'] > 0.01 or stats_dict['ks_pvalue'] > 0.01:
        status = 'WARNING'
        color = 'yellow'
    else:
        status = 'FAIL'
        color = 'lightcoral'

    row = [
        param_names[param],
        f"{stats_dict['chi2_stat']:.2f}",
        f"{stats_dict['chi2_pvalue']:.4f}",
        f"{stats_dict['ks_stat']:.4f}",
        f"{stats_dict['ks_pvalue']:.4f}",
        f"{stats_dict['rank_mean']:.0f}",
        f"{stats_dict['rank_expected']:.0f}",
        status
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=headers,
                 cellLoc='center', loc='center',
                 bbox=[0, 0.3, 1, 0.6])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the status column
for i in range(len(param_list)):
    status = table_data[i][-1]
    if status == 'PASS':
        table[(i+1, 7)].set_facecolor('lightgreen')
    elif status == 'WARNING':
        table[(i+1, 7)].set_facecolor('yellow')
    else:
        table[(i+1, 7)].set_facecolor('lightcoral')

# Add title and interpretation
title_text = "SBC Rank Statistics - Uniformity Tests"
ax.text(0.5, 0.95, title_text, ha='center', va='top',
        fontsize=14, fontweight='bold', transform=ax.transAxes)

interpretation = """
Interpretation Guide:
• χ² test: Tests if rank histogram is uniform (p > 0.05 = PASS)
• KS test: Tests if rank ECDF matches uniform distribution (p > 0.05 = PASS)
• Rank Mean: Should be close to Expected (L/2)
• PASS: Both tests p > 0.05 (well-calibrated)
• WARNING: One test p < 0.05 (mild calibration issue)
• FAIL: Both tests p < 0.05 (systematic bias or poor calibration)
"""

ax.text(0.5, 0.15, interpretation, ha='center', va='top',
        fontsize=9, family='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig(PLOTS_DIR / 'sbc_rank_statistics_table.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'sbc_rank_statistics_table.png'}")
plt.close()

# Save rank statistics to JSON
with open(RESULTS_DIR / 'rank_statistics.json', 'w') as f:
    json.dump(rank_statistics, f, indent=2)

print("\n" + "="*80)
print("All diagnostic plots created successfully!")
print("="*80)
print(f"\nPlots saved to: {PLOTS_DIR}")
print("\nFiles created:")
print("  1. sbc_rank_histograms.png - Primary SBC diagnostic")
print("  2. sbc_parameter_recovery.png - Bias and shrinkage analysis")
print("  3. sbc_coverage.png - Credible interval calibration")
print("  4. sbc_zscores.png - Standardized error distribution")
print("  5. sbc_computational_diagnostics.png - MCMC health metrics")
print("  6. sbc_rank_statistics_table.png - Statistical test summary")
