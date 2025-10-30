"""
Create comprehensive visualizations for SBC results

Visualizations reveal:
1. Parameter recovery quality (scatter plots)
2. Coverage calibration (proportion in CI)
3. Rank uniformity (SBC histograms)
4. Shrinkage recovery (hierarchical structure)
5. Bias patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
CODE_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/code')
PLOTS_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/plots')
PLOTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(CODE_DIR / 'sbc_results.csv')
with open(CODE_DIR / 'theta_recovery_examples.json', 'r') as f:
    theta_examples = json.load(f)
rank_data = np.load(CODE_DIR / 'rank_statistics.npz')
mu_ranks = rank_data['mu_ranks']
tau_ranks = rank_data['tau_ranks']

print("Creating SBC diagnostic visualizations...")
print(f"Number of simulations: {len(df)}")

# ============================================================================
# Figure 1: Parameter Recovery - Main diagnostic plot
# ============================================================================
print("\n1. Creating parameter recovery scatter plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: mu recovery
ax = axes[0, 0]
ax.scatter(df['mu_true'], df['mu_post_mean'], alpha=0.5, s=30, color='steelblue')
# Add 1:1 line
lim = [df[['mu_true', 'mu_post_mean']].min().min(),
       df[['mu_true', 'mu_post_mean']].max().max()]
ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1, label='Perfect recovery')
# Add regression line
z = np.polyfit(df['mu_true'], df['mu_post_mean'], 1)
p = np.poly1d(z)
ax.plot(lim, p(lim), 'r-', alpha=0.5, linewidth=1.5,
        label=f'Fitted: y={z[0]:.2f}x+{z[1]:.2f}')
ax.set_xlabel('True μ', fontsize=11)
ax.set_ylabel('Posterior Mean μ', fontsize=11)
ax.set_title('(A) Population Mean Recovery', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
# Add metrics
mu_r2 = np.corrcoef(df['mu_true'], df['mu_post_mean'])[0, 1]**2
bias = (df['mu_post_mean'] - df['mu_true']).mean()
rmse = np.sqrt(((df['mu_post_mean'] - df['mu_true'])**2).mean())
ax.text(0.05, 0.95, f'R² = {mu_r2:.3f}\nBias = {bias:.2f}\nRMSE = {rmse:.2f}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        fontsize=9)

# Plot 2: tau recovery
ax = axes[0, 1]
ax.scatter(df['tau_true'], df['tau_post_mean'], alpha=0.5, s=30, color='coral')
lim = [0, max(df['tau_true'].max(), df['tau_post_mean'].max())]
ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1, label='Perfect recovery')
z = np.polyfit(df['tau_true'], df['tau_post_mean'], 1)
p = np.poly1d(z)
ax.plot(lim, p(lim), 'r-', alpha=0.5, linewidth=1.5,
        label=f'Fitted: y={z[0]:.2f}x+{z[1]:.2f}')
ax.set_xlabel('True τ', fontsize=11)
ax.set_ylabel('Posterior Mean τ', fontsize=11)
ax.set_title('(B) Between-Study SD Recovery', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
tau_r2 = np.corrcoef(df['tau_true'], df['tau_post_mean'])[0, 1]**2
bias = (df['tau_post_mean'] - df['tau_true']).mean()
rmse = np.sqrt(((df['tau_post_mean'] - df['tau_true'])**2).mean())
ax.text(0.05, 0.95, f'R² = {tau_r2:.3f}\nBias = {bias:.2f}\nRMSE = {rmse:.2f}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        fontsize=9)

# Plot 3: Credible interval widths vs true tau
ax = axes[1, 0]
ci_width_mu = df['mu_q975'] - df['mu_q025']
ax.scatter(df['tau_true'], ci_width_mu, alpha=0.5, s=30, color='steelblue')
ax.set_xlabel('True τ', fontsize=11)
ax.set_ylabel('95% CI Width for μ', fontsize=11)
ax.set_title('(C) Uncertainty Increases with τ', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
# Add trend line
z = np.polyfit(df['tau_true'], ci_width_mu, 1)
p = np.poly1d(z)
x_trend = np.linspace(df['tau_true'].min(), df['tau_true'].max(), 100)
ax.plot(x_trend, p(x_trend), 'r-', alpha=0.5, linewidth=1.5,
        label=f'Trend: {z[0]:.2f}τ + {z[1]:.2f}')
ax.legend(loc='upper left', fontsize=9)

# Plot 4: Coverage by true tau
ax = axes[1, 1]
# Bin tau values
tau_bins = pd.cut(df['tau_true'], bins=5)
coverage_by_tau = df.groupby(tau_bins).agg({
    'mu_in_ci': 'mean',
    'tau_in_ci': 'mean',
    'theta_coverage_rate': 'mean'
}).reset_index()
bin_centers = [interval.mid for interval in coverage_by_tau['tau_true']]
x_pos = np.arange(len(bin_centers))
width = 0.25
ax.bar(x_pos - width, coverage_by_tau['mu_in_ci'], width, label='μ', alpha=0.8, color='steelblue')
ax.bar(x_pos, coverage_by_tau['tau_in_ci'], width, label='τ', alpha=0.8, color='coral')
ax.bar(x_pos + width, coverage_by_tau['theta_coverage_rate'], width, label='θ (avg)', alpha=0.8, color='seagreen')
ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (95%)')
ax.set_xlabel('True τ (binned)', fontsize=11)
ax.set_ylabel('Coverage Rate', fontsize=11)
ax.set_title('(D) Coverage by True τ', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{c:.1f}' for c in bin_centers], fontsize=8)
ax.set_ylim([0.8, 1.0])
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_recovery.png', bbox_inches='tight')
print(f"   Saved: parameter_recovery.png")
plt.close()

# ============================================================================
# Figure 2: SBC Rank Histograms (Key calibration diagnostic)
# ============================================================================
print("\n2. Creating SBC rank histograms...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Expected number per bin for uniform distribution
n_sims = len(mu_ranks)
n_samples = 3000  # Total MCMC samples after warmup
n_bins = 20
expected_per_bin = n_sims / n_bins

# Plot 1: mu ranks
ax = axes[0]
counts, bins, patches = ax.hist(mu_ranks, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
ax.axhline(expected_per_bin, color='red', linestyle='--', linewidth=2, label=f'Expected ({expected_per_bin:.0f})')
# Add 95% confidence band for uniform
se = np.sqrt(expected_per_bin)
ax.axhline(expected_per_bin + 2*se, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(expected_per_bin - 2*se, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Rank Statistic', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(A) SBC Ranks for μ', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
# Add chi-square test
expected = np.full(n_bins, expected_per_bin)
chi2 = np.sum((counts - expected)**2 / expected)
p_value = 1 - np.sum([counts[i] < expected_per_bin + 2*se and counts[i] > expected_per_bin - 2*se for i in range(n_bins)]) / n_bins
ax.text(0.05, 0.95, f'χ² = {chi2:.1f}\nUniformity: {"PASS" if chi2 < 30 else "FAIL"}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen' if chi2 < 30 else 'lightcoral', alpha=0.5),
        fontsize=10)

# Plot 2: tau ranks
ax = axes[1]
counts, bins, patches = ax.hist(tau_ranks, bins=n_bins, alpha=0.7, color='coral', edgecolor='black')
ax.axhline(expected_per_bin, color='red', linestyle='--', linewidth=2, label=f'Expected ({expected_per_bin:.0f})')
ax.axhline(expected_per_bin + 2*se, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(expected_per_bin - 2*se, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Rank Statistic', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(B) SBC Ranks for τ', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
chi2 = np.sum((counts - expected)**2 / expected)
ax.text(0.05, 0.95, f'χ² = {chi2:.1f}\nUniformity: {"PASS" if chi2 < 30 else "FAIL"}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen' if chi2 < 30 else 'lightcoral', alpha=0.5),
        fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sbc_rank_histograms.png', bbox_inches='tight')
print(f"   Saved: sbc_rank_histograms.png")
plt.close()

# ============================================================================
# Figure 3: Shrinkage Recovery (Hierarchical structure check)
# ============================================================================
print("\n3. Creating shrinkage recovery plots...")

# Use first 6 examples for clean visualization
n_examples = min(6, len(theta_examples))
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx in range(n_examples):
    ax = axes[idx]
    example = theta_examples[idx]

    theta_true = np.array(example['theta_true'])
    theta_post_mean = np.array(example['theta_post_mean'])
    theta_q025 = np.array(example['theta_q025'])
    theta_q975 = np.array(example['theta_q975'])
    theta_in_ci = example['theta_in_ci']

    # Get mu_true for this simulation
    sim_id = example['sim_id']
    mu_true = df.loc[df['sim_id'] == sim_id, 'mu_true'].values[0]

    # Plot
    x = np.arange(1, 9)
    colors = ['green' if in_ci else 'red' for in_ci in theta_in_ci]

    for j in range(8):
        ax.plot([x[j], x[j]], [theta_q025[j], theta_q975[j]], color=colors[j], alpha=0.6, linewidth=2)
        ax.scatter(x[j], theta_post_mean[j], color=colors[j], s=50, zorder=5, alpha=0.8)

    ax.scatter(x, theta_true, color='black', s=80, marker='x', linewidths=2, label='True θ', zorder=6)
    ax.axhline(mu_true, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='True μ')

    coverage = np.mean(theta_in_ci)
    ax.set_title(f'Simulation {sim_id + 1}: Coverage={coverage:.1%}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Study', fontsize=9)
    ax.set_ylabel('Effect θ', fontsize=9)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'shrinkage_recovery.png', bbox_inches='tight')
print(f"   Saved: shrinkage_recovery.png")
plt.close()

# ============================================================================
# Figure 4: Bias and Error Analysis
# ============================================================================
print("\n4. Creating bias and error analysis...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Bias vs true value (mu)
ax = axes[0, 0]
mu_bias = df['mu_post_mean'] - df['mu_true']
ax.scatter(df['mu_true'], mu_bias, alpha=0.5, s=30, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No bias')
ax.axhline(mu_bias.mean(), color='orange', linestyle='-', linewidth=1.5,
           label=f'Mean bias = {mu_bias.mean():.2f}')
ax.set_xlabel('True μ', fontsize=11)
ax.set_ylabel('Bias (Posterior Mean - True)', fontsize=11)
ax.set_title('(A) μ Bias vs True Value', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Bias vs true value (tau)
ax = axes[0, 1]
tau_bias = df['tau_post_mean'] - df['tau_true']
ax.scatter(df['tau_true'], tau_bias, alpha=0.5, s=30, color='coral')
ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No bias')
ax.axhline(tau_bias.mean(), color='orange', linestyle='-', linewidth=1.5,
           label=f'Mean bias = {tau_bias.mean():.2f}')
ax.set_xlabel('True τ', fontsize=11)
ax.set_ylabel('Bias (Posterior Mean - True)', fontsize=11)
ax.set_title('(B) τ Bias vs True Value', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Coverage indicators (mu)
ax = axes[1, 0]
in_ci = df['mu_in_ci'].astype(int)
colors = ['green' if x == 1 else 'red' for x in in_ci]
ax.scatter(df['mu_true'], df['mu_post_mean'], c=colors, alpha=0.5, s=30)
lim = [df[['mu_true', 'mu_post_mean']].min().min(),
       df[['mu_true', 'mu_post_mean']].max().max()]
ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('True μ', fontsize=11)
ax.set_ylabel('Posterior Mean μ', fontsize=11)
ax.set_title(f'(C) μ Coverage: {df["mu_in_ci"].mean():.1%}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.5, label='In 95% CI'),
                   Patch(facecolor='red', alpha=0.5, label='Outside 95% CI')]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Plot 4: Coverage indicators (tau)
ax = axes[1, 1]
in_ci = df['tau_in_ci'].astype(int)
colors = ['green' if x == 1 else 'red' for x in in_ci]
ax.scatter(df['tau_true'], df['tau_post_mean'], c=colors, alpha=0.5, s=30)
lim = [0, max(df['tau_true'].max(), df['tau_post_mean'].max())]
ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('True τ', fontsize=11)
ax.set_ylabel('Posterior Mean τ', fontsize=11)
ax.set_title(f'(D) τ Coverage: {df["tau_in_ci"].mean():.1%}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'bias_and_coverage.png', bbox_inches='tight')
print(f"   Saved: bias_and_coverage.png")
plt.close()

# ============================================================================
# Figure 5: MCMC Diagnostics Summary
# ============================================================================
print("\n5. Creating MCMC diagnostics summary...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: ESS distribution
ax = axes[0, 0]
ax.hist(df['ess_mu'], bins=20, alpha=0.7, color='steelblue', edgecolor='black', label='μ')
ax.axvline(df['ess_mu'].mean(), color='blue', linestyle='--', linewidth=2,
           label=f'Mean = {df["ess_mu"].mean():.0f}')
ax.axvline(400, color='red', linestyle=':', linewidth=2, label='Target = 400')
ax.set_xlabel('Effective Sample Size', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(A) ESS Distribution for μ', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
ax.hist(df['ess_tau'], bins=20, alpha=0.7, color='coral', edgecolor='black', label='τ')
ax.axvline(df['ess_tau'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {df["ess_tau"].mean():.0f}')
ax.axvline(50, color='red', linestyle=':', linewidth=2, label='Target = 50')
ax.set_xlabel('Effective Sample Size', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(B) ESS Distribution for τ', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Acceptance rate
ax = axes[1, 0]
ax.hist(df['acceptance_rate'], bins=20, alpha=0.7, color='seagreen', edgecolor='black')
ax.axvline(df['acceptance_rate'].mean(), color='darkgreen', linestyle='--', linewidth=2,
           label=f'Mean = {df["acceptance_rate"].mean():.2%}')
ax.axvline(0.234, color='red', linestyle=':', linewidth=2, label='Optimal ≈ 23%')
ax.set_xlabel('Acceptance Rate (τ MH step)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(C) Metropolis-Hastings Acceptance Rate', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Posterior SD vs coverage
ax = axes[1, 1]
colors_mu = ['green' if x else 'red' for x in df['mu_in_ci']]
ax.scatter(df['mu_post_sd'], df['theta_coverage_rate'], c=colors_mu, alpha=0.5, s=40)
ax.set_xlabel('Posterior SD of μ', fontsize=11)
ax.set_ylabel('θ Coverage Rate', fontsize=11)
ax.set_title('(D) Uncertainty vs θ Coverage', fontsize=12, fontweight='bold')
ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (95%)')
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'mcmc_diagnostics.png', bbox_inches='tight')
print(f"   Saved: mcmc_diagnostics.png")
plt.close()

print("\n" + "="*80)
print("All visualizations created successfully!")
print("="*80)
print(f"\nPlots saved to: {PLOTS_DIR}")
print("\nKey diagnostic plots:")
print("  1. parameter_recovery.png - Main recovery quality assessment")
print("  2. sbc_rank_histograms.png - Calibration uniformity check")
print("  3. shrinkage_recovery.png - Hierarchical structure validation")
print("  4. bias_and_coverage.png - Systematic error detection")
print("  5. mcmc_diagnostics.png - Computational quality metrics")
