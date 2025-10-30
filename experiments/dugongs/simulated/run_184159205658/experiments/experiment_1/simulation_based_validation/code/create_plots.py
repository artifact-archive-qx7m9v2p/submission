"""
Create visualization plots for SBC results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Load results
df = pd.read_csv('/workspace/experiments/experiment_1/simulation_based_validation/code/sbc_results.csv')

print(f"Loaded {len(df)} simulation results")

# Prior specifications
PRIOR_BETA0_MEAN = 1.73
PRIOR_BETA0_SD = 0.5
PRIOR_BETA1_MEAN = 0.28
PRIOR_BETA1_SD = 0.15
PRIOR_SIGMA_RATE = 5.0
PRIOR_SIGMA_SD = 1/PRIOR_SIGMA_RATE

N_SIMS = len(df)
N_SAMPLES = 4000  # Number of posterior samples per simulation

# ============================================================
# 1. SBC RANK HISTOGRAMS
# ============================================================
print("Creating SBC rank histograms...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

params = ['beta0', 'beta1', 'sigma']
param_labels = [r'$\beta_0$ (Intercept)', r'$\beta_1$ (Slope)', r'$\sigma$ (Noise SD)']

for i, (param, label) in enumerate(zip(params, param_labels)):
    ax = axes[i]

    ranks = df[f'rank_{param}'].values

    # Expected uniform distribution
    n_bins = 20
    expected_per_bin = N_SIMS / n_bins

    # Create histogram
    counts, bin_edges, patches = ax.hist(ranks, bins=n_bins, edgecolor='black',
                                          alpha=0.7, color='steelblue')

    # Add expected uniform line
    ax.axhline(expected_per_bin, color='red', linestyle='--', linewidth=2,
               label=f'Expected (uniform): {expected_per_bin:.1f}')

    # Chi-squared test for uniformity
    expected = np.ones(n_bins) * expected_per_bin
    chi2_stat = np.sum((counts - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=n_bins-1)

    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{label}\n' + r'$\chi^2$' + f' p-value: {p_value:.3f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add pass/fail annotation
    if p_value > 0.05:
        status = 'PASS'
        color = 'green'
    else:
        status = 'FAIL'
        color = 'red'

    ax.text(0.98, 0.95, status, transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=color,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('SBC Rank Histograms (Should be Uniform if Calibrated)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/simulation_based_validation/plots/sbc_ranks.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: sbc_ranks.png")

# ============================================================
# 2. PARAMETER RECOVERY SCATTER PLOTS
# ============================================================
print("Creating parameter recovery plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (param, label) in enumerate(zip(params, param_labels)):
    ax = axes[i]

    true_vals = df[f'true_{param}'].values
    post_means = df[f'post_mean_{param}'].values
    post_sds = df[f'post_sd_{param}'].values

    # Scatter plot with error bars
    ax.errorbar(true_vals, post_means, yerr=1.96*post_sds,
                fmt='o', alpha=0.4, markersize=4,
                elinewidth=0.5, capsize=0, color='steelblue')

    # Add identity line
    min_val = min(true_vals.min(), post_means.min())
    max_val = max(true_vals.max(), post_means.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Perfect recovery')

    # Compute and display metrics
    bias = np.mean(post_means - true_vals)
    rmse = np.sqrt(np.mean((post_means - true_vals)**2))
    corr = np.corrcoef(true_vals, post_means)[0, 1]

    metrics_text = f'Bias: {bias:.4f}\nRMSE: {rmse:.4f}\nCorr: {corr:.3f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'True {label}', fontsize=11)
    ax.set_ylabel(f'Posterior Mean {label}', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Parameter Recovery: True vs. Estimated Values (with 95% CI)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/simulation_based_validation/plots/parameter_recovery.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: parameter_recovery.png")

# ============================================================
# 3. COVERAGE DIAGNOSTIC PLOTS
# ============================================================
print("Creating coverage diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (param, label) in enumerate(zip(params, param_labels)):
    # Top row: Coverage by true parameter value
    ax1 = axes[0, i]

    true_vals = df[f'true_{param}'].values
    q025 = df[f'q025_{param}'].values
    q975 = df[f'q975_{param}'].values
    in_interval = (true_vals >= q025) & (true_vals <= q975)

    # Sort by true value for plotting
    sort_idx = np.argsort(true_vals)
    true_sorted = true_vals[sort_idx]
    in_interval_sorted = in_interval[sort_idx]

    # Compute rolling coverage
    window = 30
    rolling_coverage = []
    rolling_true = []

    for j in range(len(true_sorted) - window + 1):
        rolling_coverage.append(np.mean(in_interval_sorted[j:j+window]) * 100)
        rolling_true.append(np.mean(true_sorted[j:j+window]))

    ax1.plot(rolling_true, rolling_coverage, linewidth=2, color='steelblue')
    ax1.axhline(95, color='red', linestyle='--', linewidth=2, label='Target: 95%')
    ax1.fill_between(rolling_true, 90, 98, alpha=0.2, color='green',
                      label='Acceptable range')

    overall_coverage = np.mean(in_interval) * 100
    ax1.set_title(f'{label}\nOverall Coverage: {overall_coverage:.1f}%',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel(f'True {label}', fontsize=10)
    ax1.set_ylabel('Coverage (%)', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([80, 100])

    # Bottom row: Credible interval widths
    ax2 = axes[1, i]

    ci_widths = q975 - q025
    post_sds = df[f'post_sd_{param}'].values

    ax2.scatter(true_vals, ci_widths, alpha=0.5, s=30, color='steelblue')
    ax2.set_xlabel(f'True {label}', fontsize=10)
    ax2.set_ylabel('95% CI Width', fontsize=10)
    ax2.set_title(f'Mean Width: {np.mean(ci_widths):.4f}', fontsize=11)
    ax2.grid(alpha=0.3)

    # Add average width line
    ax2.axhline(np.mean(ci_widths), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(ci_widths):.4f}')
    ax2.legend(fontsize=8)

plt.suptitle('Coverage Diagnostics: Credible Interval Performance',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/simulation_based_validation/plots/coverage_diagnostic.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: coverage_diagnostic.png")

# ============================================================
# 4. SHRINKAGE PLOTS
# ============================================================
print("Creating shrinkage plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

prior_sds = {
    'beta0': PRIOR_BETA0_SD,
    'beta1': PRIOR_BETA1_SD,
    'sigma': PRIOR_SIGMA_SD
}

for i, (param, label) in enumerate(zip(params, param_labels)):
    ax = axes[i]

    post_sds = df[f'post_sd_{param}'].values
    prior_sd = prior_sds[param]

    # Histogram of posterior SDs
    ax.hist(post_sds, bins=30, alpha=0.7, color='steelblue',
            edgecolor='black', label='Posterior SD')

    # Add prior SD line
    ax.axvline(prior_sd, color='red', linestyle='--', linewidth=2,
               label=f'Prior SD: {prior_sd:.4f}')

    # Compute shrinkage
    mean_post_sd = np.mean(post_sds)
    shrinkage = (1 - mean_post_sd / prior_sd) * 100

    ax.axvline(mean_post_sd, color='green', linestyle='-', linewidth=2,
               label=f'Mean Posterior SD: {mean_post_sd:.4f}')

    ax.set_xlabel(f'Posterior SD of {label}', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{label}\nShrinkage: {shrinkage:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add shrinkage status
    if shrinkage > 50:
        status = 'PASS\n(Strong Learning)'
        color = 'green'
    elif shrinkage > 20:
        status = 'PASS\n(Moderate Learning)'
        color = 'orange'
    else:
        status = 'FAIL\n(Weak Learning)'
        color = 'red'

    ax.text(0.98, 0.95, status, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=color,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Shrinkage Analysis: Learning from Data (Posterior vs Prior Uncertainty)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/simulation_based_validation/plots/shrinkage_plot.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: shrinkage_plot.png")

# ============================================================
# 5. COMPUTATIONAL DIAGNOSTICS
# ============================================================
print("Creating computational diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ESS distributions
ax1 = axes[0, 0]
ess_data = [df['ess_beta0'].values, df['ess_beta1'].values, df['ess_sigma'].values]
bp = ax1.boxplot(ess_data, labels=[r'$\beta_0$', r'$\beta_1$', r'$\sigma$'],
                  patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
ax1.axhline(100, color='red', linestyle='--', linewidth=2, label='Target ESS: 100')
ax1.set_ylabel('Effective Sample Size', fontsize=11)
ax1.set_title('ESS Distribution by Parameter', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3, axis='y')

# Acceptance rate
ax2 = axes[0, 1]
ax2.hist(df['acceptance_rate'].values, bins=30, alpha=0.7,
         color='steelblue', edgecolor='black')
ax2.axvline(df['acceptance_rate'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {df["acceptance_rate"].mean():.3f}')
ax2.axvspan(0.2, 0.5, alpha=0.2, color='green', label='Typical range')
ax2.set_xlabel('Acceptance Rate', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('MCMC Acceptance Rate', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ESS vs Acceptance Rate
ax3 = axes[1, 0]
mean_ess = (df['ess_beta0'] + df['ess_beta1'] + df['ess_sigma']) / 3
ax3.scatter(df['acceptance_rate'], mean_ess, alpha=0.5, s=30, color='steelblue')
ax3.axhline(100, color='red', linestyle='--', linewidth=2, label='Target ESS')
ax3.set_xlabel('Acceptance Rate', fontsize=11)
ax3.set_ylabel('Mean ESS', fontsize=11)
ax3.set_title('ESS vs Acceptance Rate', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Posterior SD correlations (checking identifiability)
ax4 = axes[1, 1]
ax4.scatter(df['post_sd_beta0'], df['post_sd_beta1'], alpha=0.5, s=30, color='steelblue')
corr = np.corrcoef(df['post_sd_beta0'], df['post_sd_beta1'])[0, 1]
ax4.set_xlabel(r'Posterior SD of $\beta_0$', fontsize=11)
ax4.set_ylabel(r'Posterior SD of $\beta_1$', fontsize=11)
ax4.set_title(f'Parameter Uncertainty Correlation\nCorr: {corr:.3f}',
              fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

plt.suptitle('Computational Diagnostics', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/simulation_based_validation/plots/computational_diagnostics.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: computational_diagnostics.png")

print("\nAll plots created successfully!")
