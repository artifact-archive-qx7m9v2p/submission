"""
Create diagnostic visualizations for simulation-based calibration results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

# Paths
CODE_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/code')
PLOTS_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/plots')

# Load results
print("Loading SBC results...")
sbc_data = np.load(CODE_DIR / 'sbc_results.npz', allow_pickle=True)

with open(CODE_DIR / 'validation_metrics.json', 'r') as f:
    metrics = json.load(f)

# Extract data
mu_true = sbc_data['mu_true']
tau_true = sbc_data['tau_true']
mu_posterior = sbc_data['mu_posterior']
tau_posterior = sbc_data['tau_posterior']
mu_rank = sbc_data['mu_rank']
tau_rank = sbc_data['tau_rank']
mu_in_95ci = sbc_data['mu_in_95ci']
tau_in_95ci = sbc_data['tau_in_95ci']

n_sims = len(mu_true)
n_posterior_samples = len(mu_posterior[0])

print(f"Loaded {n_sims} simulations")
print("Creating visualizations...")

# ============================================================================
# PLOT 1: PARAMETER RECOVERY SCATTER PLOTS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mu recovery
ax = axes[0]
mu_means = [mu_posterior[i].mean() for i in range(n_sims)]
mu_lower = [np.percentile(mu_posterior[i], 2.5) for i in range(n_sims)]
mu_upper = [np.percentile(mu_posterior[i], 97.5) for i in range(n_sims)]

# Color by whether truth is in CI
colors = ['green' if x else 'red' for x in mu_in_95ci]

for i in range(n_sims):
    ax.plot([mu_true[i], mu_true[i]], [mu_lower[i], mu_upper[i]],
            color=colors[i], alpha=0.5, linewidth=1)
ax.scatter(mu_true, mu_means, c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

# Add diagonal line
mu_range = [min(mu_true.min(), min(mu_means)), max(mu_true.max(), max(mu_means))]
ax.plot(mu_range, mu_range, 'k--', alpha=0.5, linewidth=1, label='Perfect recovery')

ax.set_xlabel('True μ', fontsize=11, fontweight='bold')
ax.set_ylabel('Posterior Mean μ (with 95% CI)', fontsize=11, fontweight='bold')
ax.set_title('Population Mean Effect (μ) Recovery', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Add coverage annotation
coverage_mu = np.mean(mu_in_95ci)
ax.text(0.05, 0.95, f'95% CI Coverage: {coverage_mu:.2%}\n(Target: ~95%)',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

# Tau recovery
ax = axes[1]
tau_means = [tau_posterior[i].mean() for i in range(n_sims)]
tau_lower = [np.percentile(tau_posterior[i], 2.5) for i in range(n_sims)]
tau_upper = [np.percentile(tau_posterior[i], 97.5) for i in range(n_sims)]

# Color by whether truth is in CI
colors = ['green' if x else 'red' for x in tau_in_95ci]

for i in range(n_sims):
    ax.plot([tau_true[i], tau_true[i]], [tau_lower[i], tau_upper[i]],
            color=colors[i], alpha=0.5, linewidth=1)
ax.scatter(tau_true, tau_means, c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

# Add diagonal line
tau_range = [0, max(tau_true.max(), max(tau_means))]
ax.plot(tau_range, tau_range, 'k--', alpha=0.5, linewidth=1, label='Perfect recovery')

ax.set_xlabel('True τ', fontsize=11, fontweight='bold')
ax.set_ylabel('Posterior Mean τ (with 95% CI)', fontsize=11, fontweight='bold')
ax.set_title('Between-Study SD (τ) Recovery', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Add coverage annotation
coverage_tau = np.mean(tau_in_95ci)
ax.text(0.05, 0.95, f'95% CI Coverage: {coverage_tau:.2%}\n(Target: ~95%)',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_recovery.png', dpi=300, bbox_inches='tight')
print(f"  Saved: parameter_recovery.png")
plt.close()

# ============================================================================
# PLOT 2: SBC RANK HISTOGRAMS (Calibration Check)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mu ranks
ax = axes[0]
n_bins = 20
ax.hist(mu_rank, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
ax.axhline(n_sims / n_bins, color='red', linestyle='--', linewidth=2,
           label=f'Uniform expectation ({n_sims/n_bins:.1f})')
ax.set_xlabel('Rank of True μ Among Posterior Samples', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('SBC Rank Histogram: μ', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add KS test result
ks_stat_mu = metrics['sbc_summary']['ks_test_mu']['statistic']
ks_pval_mu = metrics['sbc_summary']['ks_test_mu']['pvalue']
ax.text(0.95, 0.95, f'KS test:\nD = {ks_stat_mu:.3f}\np = {ks_pval_mu:.3f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

# Tau ranks
ax = axes[1]
ax.hist(tau_rank, bins=n_bins, edgecolor='black', alpha=0.7, color='coral')
ax.axhline(n_sims / n_bins, color='red', linestyle='--', linewidth=2,
           label=f'Uniform expectation ({n_sims/n_bins:.1f})')
ax.set_xlabel('Rank of True τ Among Posterior Samples', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('SBC Rank Histogram: τ', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add KS test result
ks_stat_tau = metrics['sbc_summary']['ks_test_tau']['statistic']
ks_pval_tau = metrics['sbc_summary']['ks_test_tau']['pvalue']
ax.text(0.95, 0.95, f'KS test:\nD = {ks_stat_tau:.3f}\np = {ks_pval_tau:.3f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

plt.suptitle('SBC Calibration Check: Rank Histograms Should Be Uniform',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sbc_rank_histograms.png', dpi=300, bbox_inches='tight')
print(f"  Saved: sbc_rank_histograms.png")
plt.close()

# ============================================================================
# PLOT 3: BIAS AND COVERAGE BY TRUE PARAMETER VALUE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mu bias by true mu
ax = axes[0, 0]
mu_bias_individual = [mu_posterior[i].mean() - mu_true[i] for i in range(n_sims)]
ax.scatter(mu_true, mu_bias_individual, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('True μ', fontsize=11, fontweight='bold')
ax.set_ylabel('Bias (Posterior Mean - True)', fontsize=11, fontweight='bold')
ax.set_title('Bias in μ Estimation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add mean bias annotation
mean_bias_mu = np.mean(mu_bias_individual)
ax.text(0.05, 0.95, f'Mean Bias: {mean_bias_mu:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

# Tau bias by true tau
ax = axes[0, 1]
tau_bias_individual = [tau_posterior[i].mean() - tau_true[i] for i in range(n_sims)]
ax.scatter(tau_true, tau_bias_individual, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('True τ', fontsize=11, fontweight='bold')
ax.set_ylabel('Bias (Posterior Mean - True)', fontsize=11, fontweight='bold')
ax.set_title('Bias in τ Estimation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add mean bias annotation
mean_bias_tau = np.mean(tau_bias_individual)
ax.text(0.05, 0.95, f'Mean Bias: {mean_bias_tau:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

# CI width for mu by true value
ax = axes[1, 0]
mu_ci_width = [mu_upper[i] - mu_lower[i] for i in range(n_sims)]
colors = ['green' if x else 'red' for x in mu_in_95ci]
ax.scatter(mu_true, mu_ci_width, c=colors, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
ax.set_xlabel('True μ', fontsize=11, fontweight='bold')
ax.set_ylabel('95% CI Width', fontsize=11, fontweight='bold')
ax.set_title('Uncertainty Width for μ (Green = Truth in CI)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# CI width for tau by true value
ax = axes[1, 1]
tau_ci_width = [tau_upper[i] - tau_lower[i] for i in range(n_sims)]
colors = ['green' if x else 'red' for x in tau_in_95ci]
ax.scatter(tau_true, tau_ci_width, c=colors, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
ax.set_xlabel('True τ', fontsize=11, fontweight='bold')
ax.set_ylabel('95% CI Width', fontsize=11, fontweight='bold')
ax.set_title('Uncertainty Width for τ (Green = Truth in CI)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'bias_and_coverage_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: bias_and_coverage_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 4: POSTERIOR COVERAGE STRATIFIED BY PARAMETER REGIME
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Coverage by tau regime (low/medium/high heterogeneity)
ax = axes[0]
tau_bins = [0, 2, 10, 50]
tau_labels = ['Low\n(τ < 2)', 'Medium\n(2 ≤ τ < 10)', 'High\n(τ ≥ 10)']
tau_regime = np.digitize(tau_true, bins=tau_bins) - 1

coverage_by_regime = []
counts_by_regime = []
for regime in range(len(tau_labels)):
    mask = tau_regime == regime
    if mask.sum() > 0:
        coverage_by_regime.append(np.mean([mu_in_95ci[i] for i in range(n_sims) if mask[i]]))
        counts_by_regime.append(mask.sum())
    else:
        coverage_by_regime.append(0)
        counts_by_regime.append(0)

bars = ax.bar(range(len(tau_labels)), coverage_by_regime, alpha=0.7, edgecolor='black')
ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
ax.set_xticks(range(len(tau_labels)))
ax.set_xticklabels(tau_labels)
ax.set_ylabel('Coverage Rate', fontsize=11, fontweight='bold')
ax.set_title('μ Coverage by Heterogeneity Regime', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add counts on bars
for i, (bar, count) in enumerate(zip(bars, counts_by_regime)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'n={count}', ha='center', va='bottom', fontsize=9)

# Coverage by |mu| magnitude
ax = axes[1]
mu_bins = [0, 10, 25, 100]
mu_labels = ['Small\n(|μ| < 10)', 'Medium\n(10 ≤ |μ| < 25)', 'Large\n(|μ| ≥ 25)']
mu_regime = np.digitize(np.abs(mu_true), bins=mu_bins) - 1

coverage_by_mu_regime = []
counts_by_mu_regime = []
for regime in range(len(mu_labels)):
    mask = mu_regime == regime
    if mask.sum() > 0:
        coverage_by_mu_regime.append(np.mean([mu_in_95ci[i] for i in range(n_sims) if mask[i]]))
        counts_by_mu_regime.append(mask.sum())
    else:
        coverage_by_mu_regime.append(0)
        counts_by_mu_regime.append(0)

bars = ax.bar(range(len(mu_labels)), coverage_by_mu_regime, alpha=0.7,
              edgecolor='black', color='coral')
ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
ax.set_xticks(range(len(mu_labels)))
ax.set_xticklabels(mu_labels)
ax.set_ylabel('Coverage Rate', fontsize=11, fontweight='bold')
ax.set_title('μ Coverage by Effect Size Magnitude', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add counts on bars
for i, (bar, count) in enumerate(zip(bars, counts_by_mu_regime)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'n={count}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_by_regime.png', dpi=300, bbox_inches='tight')
print(f"  Saved: coverage_by_regime.png")
plt.close()

# ============================================================================
# PLOT 5: JOINT RECOVERY ASSESSMENT
# ============================================================================

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Main scatter: mu vs tau recovery
ax_main = fig.add_subplot(gs[0, :])

# Color by whether both in CI
both_in_ci = [mu_in_95ci[i] and tau_in_95ci[i] for i in range(n_sims)]
colors = ['green' if x else 'orange' for x in both_in_ci]

ax_main.scatter(mu_true, tau_true, c=colors, alpha=0.6, s=60,
                edgecolors='black', linewidth=0.5, label='True parameters')

# Add legends
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.6, label='Both μ and τ in 95% CI'),
    Patch(facecolor='orange', alpha=0.6, label='At least one outside 95% CI')
]
ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)

ax_main.set_xlabel('True μ', fontsize=11, fontweight='bold')
ax_main.set_ylabel('True τ', fontsize=11, fontweight='bold')
ax_main.set_title('Joint Parameter Recovery: Parameter Space Coverage',
                   fontsize=12, fontweight='bold')
ax_main.grid(True, alpha=0.3)

# Add coverage annotation
joint_coverage = np.mean(both_in_ci)
ax_main.text(0.05, 0.95, f'Joint Coverage: {joint_coverage:.2%}\n(Both in 95% CI)',
             transform=ax_main.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# Bottom left: correlation between errors
ax_bl = fig.add_subplot(gs[1, 0])
mu_errors = [mu_posterior[i].mean() - mu_true[i] for i in range(n_sims)]
tau_errors = [tau_posterior[i].mean() - tau_true[i] for i in range(n_sims)]

ax_bl.scatter(mu_errors, tau_errors, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
ax_bl.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax_bl.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax_bl.set_xlabel('μ Error (Posterior - True)', fontsize=10, fontweight='bold')
ax_bl.set_ylabel('τ Error (Posterior - True)', fontsize=10, fontweight='bold')
ax_bl.set_title('Error Correlation', fontsize=11, fontweight='bold')
ax_bl.grid(True, alpha=0.3)

# Compute correlation
corr = np.corrcoef(mu_errors, tau_errors)[0, 1]
ax_bl.text(0.05, 0.95, f'Correlation: {corr:.3f}',
           transform=ax_bl.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=8)

# Bottom right: convergence by parameter regime
ax_br = fig.add_subplot(gs[1, 1])

converged = sbc_data['converged']
convergence_by_tau = []
tau_regime_labels_short = ['Low', 'Med', 'High']

for regime in range(len(tau_labels)):
    mask = tau_regime == regime
    if mask.sum() > 0:
        convergence_by_tau.append(np.mean([converged[i] for i in range(n_sims) if mask[i]]))
    else:
        convergence_by_tau.append(0)

bars = ax_br.bar(range(len(tau_regime_labels_short)), convergence_by_tau,
                 alpha=0.7, edgecolor='black', color='steelblue')
ax_br.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
ax_br.set_xticks(range(len(tau_regime_labels_short)))
ax_br.set_xticklabels(tau_regime_labels_short)
ax_br.set_ylabel('Convergence Rate', fontsize=10, fontweight='bold')
ax_br.set_xlabel('Heterogeneity Regime', fontsize=10, fontweight='bold')
ax_br.set_title('Convergence by τ Regime', fontsize=11, fontweight='bold')
ax_br.set_ylim([0, 1.05])
ax_br.legend(fontsize=8)
ax_br.grid(True, alpha=0.3, axis='y')

plt.savefig(PLOTS_DIR / 'joint_recovery_assessment.png', dpi=300, bbox_inches='tight')
print(f"  Saved: joint_recovery_assessment.png")
plt.close()

# ============================================================================
# PLOT 6: CRITICAL SCENARIOS (Fixed vs Random Effects)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Find simulations closest to fixed-effect scenario (tau near 0)
fixed_idx = np.argmin(tau_true)
mu_true_fixed = mu_true[fixed_idx]
tau_true_fixed = tau_true[fixed_idx]
mu_post_fixed = mu_posterior[fixed_idx]
tau_post_fixed = tau_posterior[fixed_idx]

ax = axes[0]
ax.hist(tau_post_fixed, bins=50, edgecolor='black', alpha=0.7, color='steelblue', density=True)
ax.axvline(tau_true_fixed, color='red', linestyle='--', linewidth=2, label=f'True τ = {tau_true_fixed:.2f}')
ax.axvline(tau_post_fixed.mean(), color='orange', linestyle='-', linewidth=2,
           label=f'Posterior mean = {tau_post_fixed.mean():.2f}')
ax.set_xlabel('τ', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title(f'Fixed-Effect Scenario: τ ≈ 0\n(True μ = {mu_true_fixed:.1f})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add 95% CI annotation
tau_ci_lower = np.percentile(tau_post_fixed, 2.5)
tau_ci_upper = np.percentile(tau_post_fixed, 97.5)
ax.text(0.95, 0.95, f'95% CI: [{tau_ci_lower:.2f}, {tau_ci_upper:.2f}]',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

# Find simulation with moderate tau (around 5)
target_tau = 5
random_idx = np.argmin(np.abs(tau_true - target_tau))
mu_true_random = mu_true[random_idx]
tau_true_random = tau_true[random_idx]
mu_post_random = mu_posterior[random_idx]
tau_post_random = tau_posterior[random_idx]

ax = axes[1]
ax.hist(tau_post_random, bins=50, edgecolor='black', alpha=0.7, color='coral', density=True)
ax.axvline(tau_true_random, color='red', linestyle='--', linewidth=2,
           label=f'True τ = {tau_true_random:.2f}')
ax.axvline(tau_post_random.mean(), color='darkred', linestyle='-', linewidth=2,
           label=f'Posterior mean = {tau_post_random.mean():.2f}')
ax.set_xlabel('τ', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title(f'Random-Effects Scenario: τ ≈ 5\n(True μ = {mu_true_random:.1f})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add 95% CI annotation
tau_ci_lower_r = np.percentile(tau_post_random, 2.5)
tau_ci_upper_r = np.percentile(tau_post_random, 97.5)
ax.text(0.95, 0.95, f'95% CI: [{tau_ci_lower_r:.2f}, {tau_ci_upper_r:.2f}]',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'critical_scenarios.png', dpi=300, bbox_inches='tight')
print(f"  Saved: critical_scenarios.png")
plt.close()

print("\nAll visualizations created successfully!")
print(f"Plots saved to: {PLOTS_DIR}")
