"""
Visualization code for SBC results
Creates comprehensive diagnostic plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
RESULTS_DIR = Path("/workspace/experiments/experiment_2/simulation_based_validation/results")
PLOTS_DIR = Path("/workspace/experiments/experiment_2/simulation_based_validation/plots")

# Load results
results_df = pd.read_csv(RESULTS_DIR / "sbc_results.csv")
with open(RESULTS_DIR / "sbc_summary.json") as f:
    summary = json.load(f)

N_SAMPLES = 4000

print("Creating SBC diagnostic plots...")

# 1. SBC Rank Histograms
print("  - Rank histograms...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (param, ax) in enumerate(zip(['mu_logit', 'sigma'], axes)):
    ranks = results_df[f'{param}_rank'].values

    # Histogram
    n_bins = 20
    ax.hist(ranks, bins=n_bins, range=(0, N_SAMPLES), alpha=0.7,
            edgecolor='black', linewidth=1.2)

    # Expected uniform line
    expected_count = len(ranks) / n_bins
    ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
               label=f'Expected (uniform): {expected_count:.1f}')

    # Chi-square statistic
    hist_counts, _ = np.histogram(ranks, bins=n_bins, range=(0, N_SAMPLES))
    chisq = np.sum((hist_counts - expected_count)**2 / expected_count)
    pvalue = 1 - stats.chi2.cdf(chisq, n_bins - 1)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{param}\nχ² = {chisq:.1f}, p = {pvalue:.4f}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "sbc_rank_histograms.png", bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'sbc_rank_histograms.png'}")
plt.close()

# 2. Coverage Calibration Plots
print("  - Coverage calibration...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for row, param in enumerate(['mu_logit', 'sigma']):
    true_vals = results_df[f'{param}_true'].values
    q025 = results_df[f'{param}_q025'].values
    q975 = results_df[f'{param}_q975'].values
    means = results_df[f'{param}_mean'].values

    # Scatter: true vs estimated
    ax = axes[row, 0]
    ax.scatter(true_vals, means, alpha=0.5, s=50)

    # Perfect recovery line
    min_val = min(true_vals.min(), means.min())
    max_val = max(true_vals.max(), means.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect recovery')

    ax.set_xlabel(f'True {param}', fontsize=12)
    ax.set_ylabel(f'Estimated {param} (posterior mean)', fontsize=12)
    ax.set_title(f'{param}: Parameter Recovery', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Coverage plot
    ax = axes[row, 1]
    covered = (true_vals >= q025) & (true_vals <= q975)

    # Sort by true value for better visualization
    sort_idx = np.argsort(true_vals)
    true_sorted = true_vals[sort_idx]
    q025_sorted = q025[sort_idx]
    q975_sorted = q975[sort_idx]
    covered_sorted = covered[sort_idx]

    # Plot intervals
    for i in range(len(true_sorted)):
        color = 'green' if covered_sorted[i] else 'red'
        alpha = 0.6 if covered_sorted[i] else 0.9
        ax.plot([i, i], [q025_sorted[i], q975_sorted[i]], color=color, alpha=alpha, linewidth=2)

    # Plot true values
    ax.scatter(range(len(true_sorted)), true_sorted, c=['green' if c else 'red' for c in covered_sorted],
               s=50, zorder=5, edgecolor='black', linewidth=0.5,
               label=f'True {param}')

    coverage = covered.mean()
    ax.set_xlabel('Iteration (sorted by true value)', fontsize=12)
    ax.set_ylabel(f'{param}', fontsize=12)
    ax.set_title(f'{param}: 95% CI Coverage = {coverage:.3f} (target: 0.95)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.6, label='Truth inside CI'),
                      Patch(facecolor='red', alpha=0.9, label='Truth outside CI')]
    ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_calibration.png", bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'coverage_calibration.png'}")
plt.close()

# 3. Parameter Recovery Scatter with Bias
print("  - Parameter recovery scatter...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for row, param in enumerate(['mu_logit', 'sigma']):
    true_vals = results_df[f'{param}_true'].values
    means = results_df[f'{param}_mean'].values
    sds = results_df[f'{param}_sd'].values

    # Scatter with error bars
    ax = axes[row, 0]
    ax.errorbar(true_vals, means, yerr=1.96*sds, fmt='o', alpha=0.3,
                markersize=5, elinewidth=1, capsize=0)

    # Perfect recovery line
    min_val = min(true_vals.min(), means.min())
    max_val = max(true_vals.max(), means.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect recovery')

    # Compute bias
    bias = (means - true_vals).mean()
    rmse = np.sqrt(np.mean((means - true_vals)**2))

    ax.set_xlabel(f'True {param}', fontsize=12)
    ax.set_ylabel(f'Posterior mean ± 95% CI', fontsize=12)
    ax.set_title(f'{param}: Recovery with Uncertainty\nBias = {bias:.4f}, RMSE = {rmse:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bias plot (residuals)
    ax = axes[row, 1]
    bias_vals = means - true_vals
    ax.scatter(true_vals, bias_vals, alpha=0.5, s=50)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')
    ax.axhline(bias, color='blue', linestyle='--', linewidth=2, label=f'Mean bias = {bias:.4f}')

    ax.set_xlabel(f'True {param}', fontsize=12)
    ax.set_ylabel(f'Bias (estimated - true)', fontsize=12)
    ax.set_title(f'{param}: Bias Pattern', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery_scatter.png", bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'parameter_recovery_scatter.png'}")
plt.close()

# 4. Posterior Contraction
print("  - Posterior contraction...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (param, ax) in enumerate(zip(['mu_logit', 'sigma'], axes)):
    true_vals = results_df[f'{param}_true'].values
    sds = results_df[f'{param}_sd'].values

    ax.scatter(true_vals, sds, alpha=0.5, s=50)

    # Prior SD
    if param == 'mu_logit':
        prior_sd = 1.0
    else:
        prior_sd = 0.6028  # Half-normal(0, 1) SD

    ax.axhline(prior_sd, color='red', linestyle='--', linewidth=2,
               label=f'Prior SD = {prior_sd:.3f}')

    mean_posterior_sd = sds.mean()
    contraction = mean_posterior_sd / prior_sd

    ax.axhline(mean_posterior_sd, color='blue', linestyle='--', linewidth=2,
               label=f'Mean posterior SD = {mean_posterior_sd:.3f}')

    ax.set_xlabel(f'True {param}', fontsize=12)
    ax.set_ylabel(f'Posterior SD', fontsize=12)
    ax.set_title(f'{param}: Posterior Contraction\nRatio = {contraction:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_contraction.png", bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'posterior_contraction.png'}")
plt.close()

# 5. Z-score distribution (should be standard normal if well-calibrated)
print("  - Z-score distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (param, ax) in enumerate(zip(['mu_logit', 'sigma'], axes)):
    true_vals = results_df[f'{param}_true'].values
    means = results_df[f'{param}_mean'].values
    sds = results_df[f'{param}_sd'].values

    zscores = (means - true_vals) / sds

    # Histogram
    ax.hist(zscores, bins=30, alpha=0.7, edgecolor='black', density=True,
            label='Observed z-scores')

    # Expected standard normal
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2,
            label='Standard normal')

    # Test normality
    ks_stat, ks_pvalue = stats.kstest(zscores, 'norm')

    ax.set_xlabel('Z-score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{param}: Z-score Distribution\nKS test: p = {ks_pvalue:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "zscore_distribution.png", bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'zscore_distribution.png'}")
plt.close()

# 6. Parameter space coverage
print("  - Parameter space identifiability...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# True values explored
ax = axes[0]
ax.scatter(results_df['mu_logit_true'], results_df['sigma_true'],
           alpha=0.5, s=50, label='True parameters (from prior)')
ax.set_xlabel('True μ_logit', fontsize=12)
ax.set_ylabel('True σ', fontsize=12)
ax.set_title('Parameter Space Explored\n(True values from prior)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Posterior means
ax = axes[1]
covered = ((results_df['mu_logit_true'] >= results_df['mu_logit_q025']) &
           (results_df['mu_logit_true'] <= results_df['mu_logit_q975']) &
           (results_df['sigma_true'] >= results_df['sigma_q025']) &
           (results_df['sigma_true'] <= results_df['sigma_q975']))

ax.scatter(results_df['mu_logit_mean'], results_df['sigma_mean'],
           c=['green' if c else 'red' for c in covered],
           alpha=0.5, s=50)
ax.set_xlabel('Estimated μ_logit (posterior mean)', fontsize=12)
ax.set_ylabel('Estimated σ (posterior mean)', fontsize=12)
ax.set_title('Posterior Estimates\n(Green: both params covered, Red: at least one not covered)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_space_identifiability.png", bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'parameter_space_identifiability.png'}")
plt.close()

# 7. Comprehensive summary plot
print("  - Comprehensive summary...")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Rank histograms
for idx, param in enumerate(['mu_logit', 'sigma']):
    ax = fig.add_subplot(gs[0, idx])
    ranks = results_df[f'{param}_rank'].values
    n_bins = 20
    ax.hist(ranks, bins=n_bins, range=(0, N_SAMPLES), alpha=0.7, edgecolor='black')
    expected_count = len(ranks) / n_bins
    ax.axhline(expected_count, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{param}: Rank Histogram', fontsize=11, fontweight='bold')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.3)

# Row 2: Recovery scatter
for idx, param in enumerate(['mu_logit', 'sigma']):
    ax = fig.add_subplot(gs[1, idx])
    true_vals = results_df[f'{param}_true'].values
    means = results_df[f'{param}_mean'].values
    ax.scatter(true_vals, means, alpha=0.5, s=30)
    min_val = min(true_vals.min(), means.min())
    max_val = max(true_vals.max(), means.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_title(f'{param}: Recovery', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'True {param}')
    ax.set_ylabel(f'Estimated {param}')
    ax.grid(alpha=0.3)

# Row 2, col 3: Parameter space
ax = fig.add_subplot(gs[1, 2])
covered = ((results_df['mu_logit_true'] >= results_df['mu_logit_q025']) &
           (results_df['mu_logit_true'] <= results_df['mu_logit_q975']) &
           (results_df['sigma_true'] >= results_df['sigma_q025']) &
           (results_df['sigma_true'] <= results_df['sigma_q975']))
ax.scatter(results_df['mu_logit_true'], results_df['sigma_true'],
           c=['green' if c else 'red' for c in covered], alpha=0.5, s=30)
ax.set_title('Parameter Space (Coverage)', fontsize=11, fontweight='bold')
ax.set_xlabel('True μ_logit')
ax.set_ylabel('True σ')
ax.grid(alpha=0.3)

# Row 3: Coverage
for idx, param in enumerate(['mu_logit', 'sigma']):
    ax = fig.add_subplot(gs[2, idx])
    true_vals = results_df[f'{param}_true'].values
    bias = (results_df[f'{param}_mean'] - true_vals)
    ax.scatter(true_vals, bias, alpha=0.5, s=30)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    mean_bias = bias.mean()
    ax.axhline(mean_bias, color='blue', linestyle='--', linewidth=2)
    ax.set_title(f'{param}: Bias = {mean_bias:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'True {param}')
    ax.set_ylabel('Bias (est - true)')
    ax.grid(alpha=0.3)

# Row 3, col 3: Summary text
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
summary_text = f"""
SBC Summary Statistics

Coverage (95% CI):
  μ_logit: {summary['coverage']['mu_logit_95']:.3f} (target: 0.95)
  σ:       {summary['coverage']['sigma_95']:.3f} (target: 0.95)

Bias:
  μ_logit: {summary['bias']['mu_logit']:.4f}
  σ:       {summary['bias']['sigma']:.4f}

RMSE:
  μ_logit: {summary['rmse']['mu_logit']:.4f}
  σ:       {summary['rmse']['sigma']:.4f}

Contraction:
  μ_logit: {summary['contraction']['mu_logit']:.3f}
  σ:       {summary['contraction']['sigma']:.3f}

Success rate: {summary['success_rate']:.3f}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center')

# Overall title
fig.suptitle('Simulation-Based Calibration: Hierarchical Logit Model',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(PLOTS_DIR / "sbc_comprehensive_summary.png", bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'sbc_comprehensive_summary.png'}")
plt.close()

print("\nAll plots created successfully!")
print(f"Plots saved to: {PLOTS_DIR}")
