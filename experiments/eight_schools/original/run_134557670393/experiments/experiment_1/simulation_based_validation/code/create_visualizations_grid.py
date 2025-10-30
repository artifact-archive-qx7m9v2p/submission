"""
Create diagnostic visualizations for grid-based simulation validation
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
print("Loading simulation results...")
sbc_data = np.load(CODE_DIR / 'sbc_results.npz', allow_pickle=True)

with open(CODE_DIR / 'validation_metrics.json', 'r') as f:
    metrics = json.load(f)

# Extract data
mu_true = sbc_data['mu_true']
tau_true = sbc_data['tau_true']
mu_mean = sbc_data['mu_mean']
tau_mean = sbc_data['tau_mean']
mu_std = sbc_data['mu_std']
tau_std = sbc_data['tau_std']
mu_in_ci = sbc_data['mu_in_ci']
tau_in_ci = sbc_data['tau_in_ci']

n_sims = len(mu_true)

print(f"Loaded {n_sims} simulations")
print("Creating visualizations...")

# ============================================================================
# PLOT 1: PARAMETER RECOVERY
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mu recovery
ax = axes[0]
mu_lower = mu_mean - 1.96 * mu_std
mu_upper = mu_mean + 1.96 * mu_std

colors = ['green' if x else 'red' for x in mu_in_ci]

for i in range(n_sims):
    ax.plot([mu_true[i], mu_true[i]], [mu_lower[i], mu_upper[i]],
            color=colors[i], alpha=0.5, linewidth=1.5)
ax.scatter(mu_true, mu_mean, c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

# Diagonal
mu_range = [min(mu_true.min(), mu_mean.min()-5), max(mu_true.max(), mu_mean.max()+5)]
ax.plot(mu_range, mu_range, 'k--', alpha=0.5, linewidth=2, label='Perfect recovery')

ax.set_xlabel('True μ', fontsize=12, fontweight='bold')
ax.set_ylabel('Posterior Mean μ (with 95% CI)', fontsize=12, fontweight='bold')
ax.set_title('Population Mean Effect (μ) Recovery', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

coverage_mu = np.mean(mu_in_ci)
ax.text(0.05, 0.95, f'95% CI Coverage: {coverage_mu:.1%}\nTarget: ~95%',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen' if coverage_mu >= 0.90 else 'wheat',
                  alpha=0.7),
        fontsize=10)

# Tau recovery
ax = axes[1]
tau_lower = np.maximum(0, tau_mean - 1.96 * tau_std)
tau_upper = tau_mean + 1.96 * tau_std

colors = ['green' if x else 'red' for x in tau_in_ci]

for i in range(n_sims):
    ax.plot([tau_true[i], tau_true[i]], [tau_lower[i], tau_upper[i]],
            color=colors[i], alpha=0.5, linewidth=1.5)
ax.scatter(tau_true, tau_mean, c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

# Diagonal
tau_range = [0, max(tau_true.max(), tau_mean.max())+2]
ax.plot(tau_range, tau_range, 'k--', alpha=0.5, linewidth=2, label='Perfect recovery')

ax.set_xlabel('True τ', fontsize=12, fontweight='bold')
ax.set_ylabel('Posterior Mean τ (with 95% CI)', fontsize=12, fontweight='bold')
ax.set_title('Between-Study SD (τ) Recovery', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

coverage_tau = np.mean(tau_in_ci)
ax.text(0.05, 0.95, f'95% CI Coverage: {coverage_tau:.1%}\nTarget: ~95%',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen' if coverage_tau >= 0.90 else 'wheat',
                  alpha=0.7),
        fontsize=10)

plt.suptitle('Simulation-Based Calibration: Parameter Recovery Assessment',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_recovery.png', dpi=300, bbox_inches='tight')
print(f"  Saved: parameter_recovery.png")
plt.close()

# ============================================================================
# PLOT 2: BIAS AND RMSE DIAGNOSTICS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mu bias
ax = axes[0, 0]
mu_bias = mu_mean - mu_true
ax.scatter(mu_true, mu_bias, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, c='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('True μ', fontsize=11, fontweight='bold')
ax.set_ylabel('Bias (Posterior Mean - True)', fontsize=11, fontweight='bold')
ax.set_title('Bias in μ Estimation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

mean_bias_mu = np.mean(mu_bias)
ax.text(0.05, 0.95, f'Mean Bias: {mean_bias_mu:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=9)

# Tau bias
ax = axes[0, 1]
tau_bias = tau_mean - tau_true
ax.scatter(tau_true, tau_bias, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, c='coral')
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('True τ', fontsize=11, fontweight='bold')
ax.set_ylabel('Bias (Posterior Mean - True)', fontsize=11, fontweight='bold')
ax.set_title('Bias in τ Estimation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

mean_bias_tau = np.mean(tau_bias)
ax.text(0.05, 0.95, f'Mean Bias: {mean_bias_tau:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=9)

# CI width for mu
ax = axes[1, 0]
mu_ci_width = 2 * 1.96 * mu_std
colors = ['green' if x else 'red' for x in mu_in_ci]
ax.scatter(mu_true, mu_ci_width, c=colors, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
ax.set_xlabel('True μ', fontsize=11, fontweight='bold')
ax.set_ylabel('95% CI Width', fontsize=11, fontweight='bold')
ax.set_title('Uncertainty Width for μ (Green = Contains Truth)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# CI width for tau
ax = axes[1, 1]
tau_ci_width = 2 * 1.96 * tau_std
colors = ['green' if x else 'red' for x in tau_in_ci]
ax.scatter(tau_true, tau_ci_width, c=colors, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
ax.set_xlabel('True τ', fontsize=11, fontweight='bold')
ax.set_ylabel('95% CI Width', fontsize=11, fontweight='bold')
ax.set_title('Uncertainty Width for τ (Green = Contains Truth)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'bias_and_uncertainty.png', dpi=300, bbox_inches='tight')
print(f"  Saved: bias_and_uncertainty.png")
plt.close()

# ============================================================================
# PLOT 3: JOINT RECOVERY ASSESSMENT
# ============================================================================

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Main: parameter space coverage
ax_main = fig.add_subplot(gs[0, :])

both_in_ci = mu_in_ci & tau_in_ci
colors = ['darkgreen' if x else 'orange' for x in both_in_ci]

ax_main.scatter(mu_true, tau_true, c=colors, alpha=0.6, s=80,
                edgecolors='black', linewidth=0.5)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkgreen', alpha=0.6, label='Both μ and τ recovered'),
    Patch(facecolor='orange', alpha=0.6, label='At least one not recovered')
]
ax_main.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax_main.set_xlabel('True μ', fontsize=12, fontweight='bold')
ax_main.set_ylabel('True τ', fontsize=12, fontweight='bold')
ax_main.set_title('Joint Parameter Recovery: Coverage Across Parameter Space',
                   fontsize=13, fontweight='bold')
ax_main.grid(True, alpha=0.3)

joint_coverage = np.mean(both_in_ci)
ax_main.text(0.05, 0.95, f'Joint Coverage: {joint_coverage:.1%}\n(Both in 95% CI)',
             transform=ax_main.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round',
                       facecolor='lightgreen' if joint_coverage >= 0.85 else 'wheat',
                       alpha=0.7),
             fontsize=10)

# Bottom left: error correlation
ax_bl = fig.add_subplot(gs[1, 0])
mu_errors = mu_mean - mu_true
tau_errors = tau_mean - tau_true

ax_bl.scatter(mu_errors, tau_errors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5,
              c='steelblue')
ax_bl.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax_bl.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax_bl.set_xlabel('μ Error (Posterior - True)', fontsize=10, fontweight='bold')
ax_bl.set_ylabel('τ Error (Posterior - True)', fontsize=10, fontweight='bold')
ax_bl.set_title('Error Correlation', fontsize=11, fontweight='bold')
ax_bl.grid(True, alpha=0.3)

corr = np.corrcoef(mu_errors, tau_errors)[0, 1]
ax_bl.text(0.05, 0.95, f'Correlation: {corr:.3f}',
           transform=ax_bl.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           fontsize=9)

# Bottom right: coverage by heterogeneity
ax_br = fig.add_subplot(gs[1, 1])

tau_bins = [0, 2, 7, 20]
tau_labels = ['Low\n(τ<2)', 'Med\n(2≤τ<7)', 'High\n(τ≥7)']
tau_regime = np.digitize(tau_true, bins=tau_bins) - 1

coverage_by_regime = []
counts_by_regime = []
for regime in range(len(tau_labels)):
    mask = tau_regime == regime
    if mask.sum() > 0:
        coverage_by_regime.append(np.mean(mu_in_ci[mask]))
        counts_by_regime.append(mask.sum())
    else:
        coverage_by_regime.append(0)
        counts_by_regime.append(0)

bars = ax_br.bar(range(len(tau_labels)), coverage_by_regime, alpha=0.7,
                 edgecolor='black', color='steelblue')
ax_br.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
ax_br.set_xticks(range(len(tau_labels)))
ax_br.set_xticklabels(tau_labels)
ax_br.set_ylabel('μ Coverage Rate', fontsize=10, fontweight='bold')
ax_br.set_xlabel('Heterogeneity Regime', fontsize=10, fontweight='bold')
ax_br.set_title('μ Coverage by τ Regime', fontsize=11, fontweight='bold')
ax_br.set_ylim([0, 1.05])
ax_br.legend(fontsize=8)
ax_br.grid(True, alpha=0.3, axis='y')

# Add counts
for i, (bar, count) in enumerate(zip(bars, counts_by_regime)):
    if count > 0:
        height = bar.get_height()
        ax_br.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'n={count}', ha='center', va='bottom', fontsize=9)

plt.savefig(PLOTS_DIR / 'joint_recovery_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: joint_recovery_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 4: CRITICAL SCENARIOS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Fixed-effect scenario
fixed = metrics['fixed_effect_scenario']

ax = axes[0]
x = [0, 1]
true_vals = [fixed['mu_true'], fixed['tau_true']]
est_vals = [fixed['mu_mean'], fixed['tau_mean']]
err_vals = [fixed['mu_std'] * 1.96, fixed['tau_std'] * 1.96]

# Format values for display
ax.errorbar(x, est_vals, yerr=err_vals, fmt='o', markersize=10, capsize=10, capthick=2,
            color='steelblue', ecolor='steelblue', label='Posterior estimate (95% CI)')
ax.scatter(x, true_vals, marker='*', s=300, color='red', zorder=10, label='True value')

ax.set_xticks(x)
ax.set_xticklabels(['μ', 'τ'], fontsize=13)
ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
ax.set_title('Fixed-Effect Scenario (True τ ≈ 0)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add recovery status
recovery_text = f"μ recovered: {fixed['mu_recovered']}\nτ near zero: {fixed['tau_near_zero']}"
ax.text(0.05, 0.95, recovery_text,
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=9)

# Random-effect scenario
random = metrics['random_effects_scenario']

ax = axes[1]
true_vals_r = [random['mu_true'], random['tau_true']]
est_vals_r = [random['mu_mean'], random['tau_mean']]
err_vals_r = [random['mu_std'] * 1.96, random['tau_std'] * 1.96]

ax.errorbar(x, est_vals_r, yerr=err_vals_r, fmt='o', markersize=10, capsize=10, capthick=2,
            color='coral', ecolor='coral', label='Posterior estimate (95% CI)')
ax.scatter(x, true_vals_r, marker='*', s=300, color='red', zorder=10, label='True value')

ax.set_xticks(x)
ax.set_xticklabels(['μ', 'τ'], fontsize=13)
ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
ax.set_title('Random-Effects Scenario (True τ = 5)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add recovery status
recovery_text_r = (f"μ recovered: {random['mu_recovered']}\n"
                   f"τ recovered: {random['tau_recovered']}\n"
                   f"θ recovery: {random['theta_recovery_rate']:.1%}")
ax.text(0.05, 0.95, recovery_text_r,
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=9)

plt.suptitle('Critical Test Cases: Model Can Distinguish Fixed vs Random Effects',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'critical_scenarios.png', dpi=300, bbox_inches='tight')
print(f"  Saved: critical_scenarios.png")
plt.close()

# ============================================================================
# PLOT 5: CALIBRATION SUMMARY
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Coverage rates
metrics_names = ['μ\nCoverage', 'τ\nCoverage', 'θ\nCoverage\n(avg)']
coverage_vals = [
    metrics['sbc_summary']['mu_coverage'],
    metrics['sbc_summary']['tau_coverage'],
    metrics['sbc_summary']['theta_coverage_mean']
]

x_pos = np.arange(len(metrics_names))
colors_bars = ['lightgreen' if v >= 0.90 else 'wheat' for v in coverage_vals]

bars = ax.bar(x_pos, coverage_vals, alpha=0.8, edgecolor='black', linewidth=2,
              color=colors_bars)

ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Target: 95%', zorder=0)
ax.axhline(0.90, color='orange', linestyle=':', linewidth=2, label='Acceptable: 90%', zorder=0)

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold')
ax.set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.set_title('Simulation-Based Calibration: Coverage Assessment',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, coverage_vals)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add verdict
verdict = metrics['verdict']
verdict_color = 'lightgreen' if verdict == 'PASS' else 'wheat'
ax.text(0.98, 0.02, f'VERDICT: {verdict}',
        transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=verdict_color, alpha=0.9, edgecolor='black',
                  linewidth=2),
        fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'calibration_summary.png', dpi=300, bbox_inches='tight')
print(f"  Saved: calibration_summary.png")
plt.close()

print("\nAll visualizations created successfully!")
print(f"Plots saved to: {PLOTS_DIR}")
