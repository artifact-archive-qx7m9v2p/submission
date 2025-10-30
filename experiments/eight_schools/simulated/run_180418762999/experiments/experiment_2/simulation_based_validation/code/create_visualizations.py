"""
Create visualizations for SBC validation results.

This script generates comprehensive diagnostic plots to assess:
1. Rank uniformity (histogram)
2. Parameter recovery quality (scatter plots)
3. Coverage analysis (stratified by tau ranges)
4. Convergence diagnostics (divergences, R-hat, ESS)
5. Funnel diagnostics (divergences vs tau_true)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("Creating SBC Validation Visualizations")
print("="*80)
print()

# ============================================================================
# LOAD RESULTS
# ============================================================================

print("Loading results...")
results_df = pd.read_csv('/workspace/experiments/experiment_2/simulation_based_validation/diagnostics/sbc_results.csv')

with open('/workspace/experiments/experiment_2/simulation_based_validation/diagnostics/summary_stats.json', 'r') as f:
    summary_stats = json.load(f)

n_sims = len(results_df)
print(f"  Loaded {n_sims} successful simulations")
print()

# ============================================================================
# VISUALIZATION 1: RANK HISTOGRAMS
# ============================================================================

print("Creating Visualization 1: Rank Histograms")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mu rank histogram
ax = axes[0]
n_bins = 20
mu_hist, mu_edges = np.histogram(results_df['mu_rank'], bins=n_bins)
mu_bin_centers = (mu_edges[:-1] + mu_edges[1:]) / 2
expected_count = n_sims / n_bins

ax.bar(mu_bin_centers, mu_hist, width=mu_edges[1]-mu_edges[0],
       alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5,
       label='Observed ranks')
ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
           label=f'Uniform expectation ({expected_count:.1f})')

# Add 95% confidence band for uniformity
# Under null (uniform), counts follow multinomial with expected count = n_sims/n_bins
# 95% CI approximately: expected ± 1.96 * sqrt(expected * (1 - 1/n_bins))
ci_width = 1.96 * np.sqrt(expected_count * (1 - 1/n_bins))
ax.axhline(expected_count + ci_width, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.axhline(expected_count - ci_width, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.fill_between([mu_edges[0], mu_edges[-1]], expected_count - ci_width, expected_count + ci_width,
                alpha=0.2, color='red', label='95% CI')

# Add chi-square test result
chi2_stat = summary_stats['rank_uniformity']['mu_chi2_stat']
chi2_pval = summary_stats['rank_uniformity']['mu_chi2_pval']
pass_status = "PASS" if chi2_pval > 0.05 else "FAIL"
color = 'green' if chi2_pval > 0.05 else 'red'

ax.text(0.98, 0.98, f'χ² = {chi2_stat:.2f}\np = {chi2_pval:.4f}\n{pass_status}',
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white',
        alpha=0.8, edgecolor=color, linewidth=2))

ax.set_xlabel('Rank statistic', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Rank Uniformity: mu (Population Mean)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Tau rank histogram
ax = axes[1]
tau_hist, tau_edges = np.histogram(results_df['tau_rank'], bins=n_bins)
tau_bin_centers = (tau_edges[:-1] + tau_edges[1:]) / 2

ax.bar(tau_bin_centers, tau_hist, width=tau_edges[1]-tau_edges[0],
       alpha=0.7, color='coral', edgecolor='black', linewidth=1.5,
       label='Observed ranks')
ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
           label=f'Uniform expectation ({expected_count:.1f})')

ax.axhline(expected_count + ci_width, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.axhline(expected_count - ci_width, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.fill_between([tau_edges[0], tau_edges[-1]], expected_count - ci_width, expected_count + ci_width,
                alpha=0.2, color='red', label='95% CI')

# Add chi-square test result
chi2_stat_tau = summary_stats['rank_uniformity']['tau_chi2_stat']
chi2_pval_tau = summary_stats['rank_uniformity']['tau_chi2_pval']
pass_status_tau = "PASS" if chi2_pval_tau > 0.05 else "FAIL"
color_tau = 'green' if chi2_pval_tau > 0.05 else 'red'

ax.text(0.98, 0.98, f'χ² = {chi2_stat_tau:.2f}\np = {chi2_pval_tau:.4f}\n{pass_status_tau}',
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white',
        alpha=0.8, edgecolor=color_tau, linewidth=2))

ax.set_xlabel('Rank statistic', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Rank Uniformity: tau (Between-Group SD)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/rank_histogram.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/rank_histogram.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: PARAMETER RECOVERY
# ============================================================================

print("Creating Visualization 2: Parameter Recovery")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Mu recovery scatter
ax = axes[0, 0]
ax.scatter(results_df['mu_true'], results_df['mu_mean'], alpha=0.6, s=50,
           color='steelblue', edgecolors='black', linewidth=0.5)

# Add identity line
min_val = min(results_df['mu_true'].min(), results_df['mu_mean'].min())
max_val = max(results_df['mu_true'].max(), results_df['mu_mean'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect recovery')

# Add regression line
z = np.polyfit(results_df['mu_true'], results_df['mu_mean'], 1)
p = np.poly1d(z)
x_line = np.linspace(min_val, max_val, 100)
ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7,
        label=f'Fit: y={z[0]:.3f}x+{z[1]:.2f}')

# Compute correlation
corr = np.corrcoef(results_df['mu_true'], results_df['mu_mean'])[0, 1]
ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('True mu', fontsize=12)
ax.set_ylabel('Posterior mean mu', fontsize=12)
ax.set_title('Parameter Recovery: mu', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Tau recovery scatter
ax = axes[0, 1]
ax.scatter(results_df['tau_true'], results_df['tau_mean'], alpha=0.6, s=50,
           color='coral', edgecolors='black', linewidth=0.5)

# Add identity line
min_val_tau = 0  # tau >= 0
max_val_tau = max(results_df['tau_true'].max(), results_df['tau_mean'].max())
ax.plot([min_val_tau, max_val_tau], [min_val_tau, max_val_tau], 'r--', linewidth=2,
        label='Perfect recovery')

# Add regression line
z_tau = np.polyfit(results_df['tau_true'], results_df['tau_mean'], 1)
p_tau = np.poly1d(z_tau)
x_line_tau = np.linspace(min_val_tau, max_val_tau, 100)
ax.plot(x_line_tau, p_tau(x_line_tau), 'g-', linewidth=2, alpha=0.7,
        label=f'Fit: y={z_tau[0]:.3f}x+{z_tau[1]:.2f}')

# Compute correlation
corr_tau = np.corrcoef(results_df['tau_true'], results_df['tau_mean'])[0, 1]
ax.text(0.05, 0.95, f'r = {corr_tau:.4f}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('True tau', fontsize=12)
ax.set_ylabel('Posterior mean tau', fontsize=12)
ax.set_title('Parameter Recovery: tau', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Panel C: Mu bias vs true value
ax = axes[1, 0]
ax.scatter(results_df['mu_true'], results_df['mu_bias'], alpha=0.6, s=50,
           color='steelblue', edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')

# Add smoothed trend
from scipy.signal import savgol_filter
sorted_idx = np.argsort(results_df['mu_true'])
if len(sorted_idx) > 10:
    window = min(11, len(sorted_idx) if len(sorted_idx) % 2 == 1 else len(sorted_idx) - 1)
    if window >= 5:
        smoothed = savgol_filter(results_df['mu_bias'].iloc[sorted_idx], window, 3)
        ax.plot(results_df['mu_true'].iloc[sorted_idx], smoothed, 'g-', linewidth=2,
                alpha=0.7, label='Trend')

mean_bias = results_df['mu_bias'].mean()
ax.axhline(mean_bias, color='orange', linestyle=':', linewidth=2,
           label=f'Mean bias ({mean_bias:.3f})')

ax.set_xlabel('True mu', fontsize=12)
ax.set_ylabel('Bias (posterior mean - true)', fontsize=12)
ax.set_title('Bias Analysis: mu', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: Tau bias vs true value
ax = axes[1, 1]
ax.scatter(results_df['tau_true'], results_df['tau_bias'], alpha=0.6, s=50,
           color='coral', edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')

# Add smoothed trend
sorted_idx_tau = np.argsort(results_df['tau_true'])
if len(sorted_idx_tau) > 10:
    window = min(11, len(sorted_idx_tau) if len(sorted_idx_tau) % 2 == 1 else len(sorted_idx_tau) - 1)
    if window >= 5:
        smoothed_tau = savgol_filter(results_df['tau_bias'].iloc[sorted_idx_tau], window, 3)
        ax.plot(results_df['tau_true'].iloc[sorted_idx_tau], smoothed_tau, 'g-', linewidth=2,
                alpha=0.7, label='Trend')

mean_bias_tau = results_df['tau_bias'].mean()
ax.axhline(mean_bias_tau, color='orange', linestyle=':', linewidth=2,
           label=f'Mean bias ({mean_bias_tau:.3f})')

ax.set_xlabel('True tau', fontsize=12)
ax.set_ylabel('Bias (posterior mean - true)', fontsize=12)
ax.set_title('Bias Analysis: tau', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/parameter_recovery.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/parameter_recovery.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: COVERAGE ANALYSIS
# ============================================================================

print("Creating Visualization 3: Coverage Analysis")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Coverage by tau bins
ax = axes[0, 0]

# Define tau bins
tau_bins = [(0, 5, 'Low (0-5)'), (5, 10, 'Medium (5-10)'), (10, 30, 'High (>10)')]
coverage_by_bin = []
n_by_bin = []
labels = []

for tau_min, tau_max, label in tau_bins:
    mask = (results_df['tau_true'] >= tau_min) & (results_df['tau_true'] < tau_max)
    n = mask.sum()
    if n > 0:
        # For simplicity, use chi-square test to estimate coverage
        # Coverage ≈ fraction with rank in middle 90%
        ranks = results_df.loc[mask, 'tau_rank']
        total_ranks = 4000  # N_DRAWS * N_CHAINS
        in_90pct = ((ranks >= 0.05 * total_ranks) & (ranks <= 0.95 * total_ranks)).sum()
        coverage = in_90pct / n
        coverage_by_bin.append(coverage)
        n_by_bin.append(n)
        labels.append(f'{label}\n(n={n})')

x_pos = np.arange(len(labels))
colors = ['red' if abs(c - 0.9) > 0.1 else 'green' for c in coverage_by_bin]
bars = ax.bar(x_pos, coverage_by_bin, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axhline(0.9, color='blue', linestyle='--', linewidth=2, label='Target (0.90)')
ax.axhline(0.85, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(0.95, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.fill_between([-0.5, len(labels)-0.5], 0.85, 0.95, alpha=0.2, color='orange',
                label='Acceptable range')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Empirical coverage (90% CI)', fontsize=12)
ax.set_title('Coverage Stratified by tau_true', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Panel B: Coverage across all parameters
ax = axes[0, 1]

params = ['mu (90%)', 'mu (95%)', 'tau (90%)', 'tau (95%)']
coverages = [
    summary_stats['coverage']['mu_90'],
    summary_stats['coverage']['mu_95'],
    summary_stats['coverage']['tau_90'],
    summary_stats['coverage']['tau_95']
]
targets = [0.90, 0.95, 0.90, 0.95]
colors_bar = []
for i, (cov, target) in enumerate(zip(coverages, targets)):
    if abs(cov - target) < 0.05:
        colors_bar.append('green')
    elif abs(cov - target) < 0.1:
        colors_bar.append('orange')
    else:
        colors_bar.append('red')

x_pos = np.arange(len(params))
ax.bar(x_pos, coverages, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add target lines
for i, target in enumerate(targets):
    ax.plot([i-0.4, i+0.4], [target, target], 'b--', linewidth=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(params, fontsize=10)
ax.set_ylabel('Empirical coverage', fontsize=12)
ax.set_title('Coverage Summary', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0.7, 1.0)

# Add text annotations
for i, (cov, target) in enumerate(zip(coverages, targets)):
    ax.text(i, cov + 0.02, f'{cov:.3f}', ha='center', va='bottom', fontsize=9,
            fontweight='bold')

# Panel C: Calibration plot (observed vs expected)
ax = axes[1, 0]

# For each credible level, compute empirical coverage
credible_levels = np.linspace(0.1, 0.99, 20)
mu_coverage_levels = []
tau_coverage_levels = []

# This would require posterior samples, so we'll create a simplified version
# using the rank statistics as proxy
for level in credible_levels:
    # Approximate coverage using rank statistics
    # For uniform ranks, fraction in (alpha/2, 1-alpha/2) should equal level
    alpha = 1 - level
    lower_rank = alpha / 2
    upper_rank = 1 - alpha / 2

    total_ranks = 4000  # N_DRAWS * N_CHAINS
    mu_in = ((results_df['mu_rank'] >= lower_rank * total_ranks) &
             (results_df['mu_rank'] <= upper_rank * total_ranks)).mean()
    tau_in = ((results_df['tau_rank'] >= lower_rank * total_ranks) &
              (results_df['tau_rank'] <= upper_rank * total_ranks)).mean()

    mu_coverage_levels.append(mu_in)
    tau_coverage_levels.append(tau_in)

ax.plot(credible_levels, credible_levels, 'k--', linewidth=2, label='Perfect calibration')
ax.plot(credible_levels, mu_coverage_levels, 'o-', linewidth=2, markersize=6,
        color='steelblue', label='mu', alpha=0.7)
ax.plot(credible_levels, tau_coverage_levels, 's-', linewidth=2, markersize=6,
        color='coral', label='tau', alpha=0.7)

ax.set_xlabel('Nominal coverage', fontsize=12)
ax.set_ylabel('Empirical coverage', fontsize=12)
ax.set_title('Calibration Plot', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Panel D: Coverage vs simulation index (check for drift)
ax = axes[1, 1]

# For simplicity, check if each simulation's rank is in middle 90%
total_ranks = 4000
mu_in_90 = ((results_df['mu_rank'] >= 0.05 * total_ranks) &
            (results_df['mu_rank'] <= 0.95 * total_ranks)).astype(int)
tau_in_90 = ((results_df['tau_rank'] >= 0.05 * total_ranks) &
             (results_df['tau_rank'] <= 0.95 * total_ranks)).astype(int)

# Compute rolling average
window = min(10, n_sims // 5)
if window >= 3:
    mu_rolling = pd.Series(mu_in_90).rolling(window, center=True).mean()
    tau_rolling = pd.Series(tau_in_90).rolling(window, center=True).mean()

    ax.plot(results_df['iteration'], mu_rolling, '-', linewidth=2,
            color='steelblue', label='mu (rolling avg)', alpha=0.7)
    ax.plot(results_df['iteration'], tau_rolling, '-', linewidth=2,
            color='coral', label='tau (rolling avg)', alpha=0.7)

ax.scatter(results_df['iteration'], mu_in_90, alpha=0.3, s=20, color='steelblue')
ax.scatter(results_df['iteration'], tau_in_90, alpha=0.3, s=20, color='coral')

ax.axhline(0.9, color='blue', linestyle='--', linewidth=2, label='Target')
ax.axhline(0.85, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axhline(0.95, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Simulation index', fontsize=12)
ax.set_ylabel('In 90% CI (rolling avg)', fontsize=12)
ax.set_title('Coverage Stability Check', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/coverage_analysis.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/coverage_analysis.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: CONVERGENCE SUMMARY
# ============================================================================

print("Creating Visualization 4: Convergence Diagnostics")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: Divergences histogram
ax = axes[0, 0]
ax.hist(results_df['divergences'], bins=30, alpha=0.7, color='red',
        edgecolor='black', linewidth=1.5)
ax.axvline(results_df['divergences'].mean(), color='blue', linestyle='--',
           linewidth=2, label=f'Mean ({results_df["divergences"].mean():.1f})')

# Mark 5% threshold
threshold_5pct = 0.05 * 4000  # 5% of total samples
ax.axvline(threshold_5pct, color='orange', linestyle=':', linewidth=2,
           label=f'5% threshold ({threshold_5pct:.0f})')

ax.set_xlabel('Number of divergences', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Divergences per Simulation', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: R-hat distributions
ax = axes[0, 1]
ax.hist(results_df['mu_rhat'], bins=30, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=1.5, label='mu')
ax.hist(results_df['tau_rhat'], bins=30, alpha=0.6, color='coral',
        edgecolor='black', linewidth=1.5, label='tau')

ax.axvline(1.01, color='red', linestyle='--', linewidth=2,
           label='Threshold (1.01)')
ax.axvline(1.00, color='green', linestyle=':', linewidth=2,
           label='Perfect (1.00)', alpha=0.7)

ax.set_xlabel('R-hat', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Convergence: R-hat Distribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel C: ESS distributions
ax = axes[0, 2]
ax.hist(results_df['mu_ess'], bins=30, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=1.5, label='mu')
ax.hist(results_df['tau_ess'], bins=30, alpha=0.6, color='coral',
        edgecolor='black', linewidth=1.5, label='tau')

ax.axvline(results_df['mu_ess'].mean(), color='steelblue', linestyle='--',
           linewidth=2, alpha=0.7)
ax.axvline(results_df['tau_ess'].mean(), color='coral', linestyle='--',
           linewidth=2, alpha=0.7)

ax.set_xlabel('Effective sample size', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Effective Sample Size', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: R-hat vs iteration
ax = axes[1, 0]
ax.scatter(results_df['iteration'], results_df['mu_rhat'], alpha=0.6, s=30,
           color='steelblue', label='mu')
ax.scatter(results_df['iteration'], results_df['tau_rhat'], alpha=0.6, s=30,
           color='coral', label='tau')

ax.axhline(1.01, color='red', linestyle='--', linewidth=2, label='Threshold')
ax.axhline(1.00, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Simulation index', fontsize=12)
ax.set_ylabel('R-hat', fontsize=12)
ax.set_title('R-hat Stability', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel E: ESS vs iteration
ax = axes[1, 1]
ax.scatter(results_df['iteration'], results_df['mu_ess'], alpha=0.6, s=30,
           color='steelblue', label='mu')
ax.scatter(results_df['iteration'], results_df['tau_ess'], alpha=0.6, s=30,
           color='coral', label='tau')

ax.set_xlabel('Simulation index', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS Stability', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel F: Convergence summary
ax = axes[1, 2]
ax.axis('off')

# Create summary text
conv_text = "CONVERGENCE SUMMARY\n" + "="*35 + "\n\n"
conv_text += f"Divergences:\n"
conv_text += f"  Mean: {results_df['divergences'].mean():.1f}\n"
conv_text += f"  Max: {results_df['divergences'].max():.0f}\n"
conv_text += f"  % with any: {(results_df['divergences'] > 0).mean()*100:.1f}%\n\n"

conv_text += f"R-hat (mu):\n"
conv_text += f"  Mean: {results_df['mu_rhat'].mean():.4f}\n"
conv_text += f"  Max: {results_df['mu_rhat'].max():.4f}\n"
conv_text += f"  % > 1.01: {(results_df['mu_rhat'] > 1.01).mean()*100:.1f}%\n\n"

conv_text += f"R-hat (tau):\n"
conv_text += f"  Mean: {results_df['tau_rhat'].mean():.4f}\n"
conv_text += f"  Max: {results_df['tau_rhat'].max():.4f}\n"
conv_text += f"  % > 1.01: {(results_df['tau_rhat'] > 1.01).mean()*100:.1f}%\n\n"

conv_text += f"ESS:\n"
conv_text += f"  mu mean: {results_df['mu_ess'].mean():.0f}\n"
conv_text += f"  tau mean: {results_df['tau_ess'].mean():.0f}\n"

ax.text(0.1, 0.9, conv_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/convergence_summary.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/convergence_summary.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: FUNNEL DIAGNOSTICS
# ============================================================================

print("Creating Visualization 5: Funnel Diagnostics")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Divergences vs tau_true
ax = axes[0, 0]
scatter = ax.scatter(results_df['tau_true'], results_df['divergences'],
                     c=results_df['divergences'], cmap='Reds', alpha=0.6,
                     s=60, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax, label='Divergences')

# Add horizontal line at 5% threshold
threshold_5pct = 0.05 * 4000
ax.axhline(threshold_5pct, color='orange', linestyle='--', linewidth=2,
           label=f'5% threshold ({threshold_5pct:.0f})')

# Add vertical lines for tau bins
ax.axvline(5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(10, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('True tau', fontsize=12)
ax.set_ylabel('Number of divergences', fontsize=12)
ax.set_title('Funnel Diagnostic: Divergences vs tau', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# Panel B: R-hat vs tau_true
ax = axes[0, 1]
ax.scatter(results_df['tau_true'], results_df['tau_rhat'], alpha=0.6, s=50,
           color='coral', edgecolors='black', linewidth=0.5, label='tau R-hat')

ax.axhline(1.01, color='red', linestyle='--', linewidth=2, label='Threshold')
ax.axhline(1.00, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

ax.set_xlabel('True tau', fontsize=12)
ax.set_ylabel('R-hat (tau)', fontsize=12)
ax.set_title('Convergence vs tau', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# Panel C: ESS vs tau_true
ax = axes[1, 0]
ax.scatter(results_df['tau_true'], results_df['tau_ess'], alpha=0.6, s=50,
           color='coral', edgecolors='black', linewidth=0.5)

# Add smoothed trend
from scipy.signal import savgol_filter
sorted_idx = np.argsort(results_df['tau_true'])
if len(sorted_idx) > 10:
    window = min(11, len(sorted_idx) if len(sorted_idx) % 2 == 1 else len(sorted_idx) - 1)
    if window >= 5:
        smoothed = savgol_filter(results_df['tau_ess'].iloc[sorted_idx], window, 3)
        ax.plot(results_df['tau_true'].iloc[sorted_idx], smoothed, 'b-', linewidth=2,
                alpha=0.7, label='Trend')

ax.set_xlabel('True tau', fontsize=12)
ax.set_ylabel('ESS (tau)', fontsize=12)
ax.set_title('Effective Sample Size vs tau', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# Panel D: Bias vs tau_true
ax = axes[1, 1]
ax.scatter(results_df['tau_true'], results_df['tau_bias'], alpha=0.6, s=50,
           color='coral', edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')

# Add smoothed trend
if len(sorted_idx) > 10 and window >= 5:
    smoothed_bias = savgol_filter(results_df['tau_bias'].iloc[sorted_idx], window, 3)
    ax.plot(results_df['tau_true'].iloc[sorted_idx], smoothed_bias, 'g-', linewidth=2,
            alpha=0.7, label='Trend')

# Shade region by tau value
ax.axvspan(0, 5, alpha=0.1, color='red', label='Low tau (funnel risk)')
ax.axvspan(5, 10, alpha=0.1, color='yellow')
ax.axvspan(10, max(results_df['tau_true'].max(), 10), alpha=0.1, color='green')

ax.set_xlabel('True tau', fontsize=12)
ax.set_ylabel('Bias (posterior mean - true)', fontsize=12)
ax.set_title('Bias vs tau', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/funnel_diagnostics.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/funnel_diagnostics.png")
plt.close()

print()
print("="*80)
print("All visualizations created successfully!")
print("="*80)
