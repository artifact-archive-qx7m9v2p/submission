"""
Visualization of SBC Results

Creates diagnostic plots to assess:
1. Rank uniformity (SBC histograms)
2. Coverage calibration
3. Parameter recovery (bias, scatter)
4. Posterior contraction
5. Identifiability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
RESULTS_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation/results")
PLOTS_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation/plots")

# Load results
results_df = pd.read_csv(RESULTS_DIR / "sbc_results.csv")
with open(RESULTS_DIR / "sbc_summary.json") as f:
    summary = json.load(f)

# Configuration
N_CHAINS = 4
N_ITER = 1000
N_WARMUP = 500
N_SAMPLES = N_CHAINS * (N_ITER - N_WARMUP)

print("Creating SBC diagnostic plots...")
print(f"Loaded {len(results_df)} SBC iterations\n")

# ============================================================================
# 1. SBC RANK HISTOGRAMS (Primary calibration diagnostic)
# ============================================================================
print("1. Creating rank histograms...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mu ranks
n_bins = 20
axes[0].hist(results_df['mu_rank'], bins=n_bins, range=(0, N_SAMPLES),
             edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axhline(len(results_df) / n_bins, color='red', linestyle='--',
                linewidth=2, label='Expected (uniform)')
axes[0].set_xlabel('Rank of True μ in Posterior', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'SBC Rank Histogram: μ\n(χ² = {summary["rank_uniformity"]["mu_chisq"]:.1f}, p = {summary["rank_uniformity"]["mu_pvalue"]:.3f})',
                  fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Phi ranks
axes[1].hist(results_df['phi_rank'], bins=n_bins, range=(0, N_SAMPLES),
             edgecolor='black', alpha=0.7, color='darkorange')
axes[1].axhline(len(results_df) / n_bins, color='red', linestyle='--',
                linewidth=2, label='Expected (uniform)')
axes[1].set_xlabel('Rank of True φ in Posterior', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'SBC Rank Histogram: φ\n(χ² = {summary["rank_uniformity"]["phi_chisq"]:.1f}, p = {summary["rank_uniformity"]["phi_pvalue"]:.3f})',
                  fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "sbc_rank_histograms.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: sbc_rank_histograms.png")

# ============================================================================
# 2. COVERAGE CALIBRATION PLOTS
# ============================================================================
print("2. Creating coverage calibration plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Mu coverage plot
mu_in_interval = ((results_df['mu_true'] >= results_df['mu_q025']) &
                   (results_df['mu_true'] <= results_df['mu_q975']))

sorted_idx = results_df['mu_true'].argsort()
axes[0, 0].fill_between(range(len(results_df)),
                         results_df.iloc[sorted_idx]['mu_q025'],
                         results_df.iloc[sorted_idx]['mu_q975'],
                         alpha=0.3, color='steelblue', label='95% CI')
axes[0, 0].scatter(range(len(results_df)),
                   results_df.iloc[sorted_idx]['mu_true'],
                   c=mu_in_interval.iloc[sorted_idx].map({True: 'green', False: 'red'}),
                   s=20, alpha=0.6, zorder=3)
axes[0, 0].set_xlabel('SBC Iteration (sorted by true μ)', fontsize=11)
axes[0, 0].set_ylabel('μ', fontsize=11)
axes[0, 0].set_title(f'μ Coverage: {summary["coverage"]["mu"]:.3f} (target: 0.950)',
                     fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Phi coverage plot
phi_in_interval = ((results_df['phi_true'] >= results_df['phi_q025']) &
                    (results_df['phi_true'] <= results_df['phi_q975']))

sorted_idx = results_df['phi_true'].argsort()
axes[0, 1].fill_between(range(len(results_df)),
                         results_df.iloc[sorted_idx]['phi_q025'],
                         results_df.iloc[sorted_idx]['phi_q975'],
                         alpha=0.3, color='darkorange', label='95% CI')
axes[0, 1].scatter(range(len(results_df)),
                   results_df.iloc[sorted_idx]['phi_true'],
                   c=phi_in_interval.iloc[sorted_idx].map({True: 'green', False: 'red'}),
                   s=20, alpha=0.6, zorder=3)
axes[0, 1].set_xlabel('SBC Iteration (sorted by true φ)', fontsize=11)
axes[0, 1].set_ylabel('φ', fontsize=11)
axes[0, 1].set_title(f'φ Coverage: {summary["coverage"]["phi"]:.3f} (target: 0.950)',
                     fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Interval width vs true value (check for heteroscedasticity)
mu_width = results_df['mu_q975'] - results_df['mu_q025']
axes[1, 0].scatter(results_df['mu_true'], mu_width, alpha=0.5, s=30, color='steelblue')
axes[1, 0].set_xlabel('True μ', fontsize=11)
axes[1, 0].set_ylabel('95% CI Width', fontsize=11)
axes[1, 0].set_title('μ Credible Interval Width', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

phi_width = results_df['phi_q975'] - results_df['phi_q025']
axes[1, 1].scatter(results_df['phi_true'], phi_width, alpha=0.5, s=30, color='darkorange')
axes[1, 1].set_xlabel('True φ', fontsize=11)
axes[1, 1].set_ylabel('95% CI Width', fontsize=11)
axes[1, 1].set_title('φ Credible Interval Width', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_calibration.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: coverage_calibration.png")

# ============================================================================
# 3. PARAMETER RECOVERY: SCATTER PLOTS
# ============================================================================
print("3. Creating parameter recovery scatter plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Mu: posterior mean vs true
axes[0, 0].scatter(results_df['mu_true'], results_df['mu_mean'],
                   alpha=0.5, s=40, color='steelblue')
axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect recovery')
axes[0, 0].set_xlabel('True μ', fontsize=11)
axes[0, 0].set_ylabel('Posterior Mean μ', fontsize=11)
axes[0, 0].set_title(f'μ Recovery (Bias: {summary["bias"]["mu"]:.4f}, RMSE: {summary["rmse"]["mu"]:.4f})',
                     fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Phi: posterior mean vs true
axes[0, 1].scatter(results_df['phi_true'], results_df['phi_mean'],
                   alpha=0.5, s=40, color='darkorange')
max_phi = max(results_df['phi_true'].max(), results_df['phi_mean'].max())
axes[0, 1].plot([0, max_phi], [0, max_phi], 'r--', linewidth=2, label='Perfect recovery')
axes[0, 1].set_xlabel('True φ', fontsize=11)
axes[0, 1].set_ylabel('Posterior Mean φ', fontsize=11)
axes[0, 1].set_title(f'φ Recovery (Bias: {summary["bias"]["phi"]:.4f}, RMSE: {summary["rmse"]["phi"]:.4f})',
                     fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Error analysis
mu_error = results_df['mu_mean'] - results_df['mu_true']
axes[1, 0].scatter(results_df['mu_true'], mu_error, alpha=0.5, s=40, color='steelblue')
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('True μ', fontsize=11)
axes[1, 0].set_ylabel('Error (Posterior Mean - True)', fontsize=11)
axes[1, 0].set_title('μ Recovery Error', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

phi_error = results_df['phi_mean'] - results_df['phi_true']
axes[1, 1].scatter(results_df['phi_true'], phi_error, alpha=0.5, s=40, color='darkorange')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('True φ', fontsize=11)
axes[1, 1].set_ylabel('Error (Posterior Mean - True)', fontsize=11)
axes[1, 1].set_title('φ Recovery Error', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: parameter_recovery_scatter.png")

# ============================================================================
# 4. POSTERIOR CONTRACTION
# ============================================================================
print("4. Creating posterior contraction plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mu contraction
prior_mu_sd = np.sqrt(2 * 25 / ((2 + 25)**2 * (2 + 25 + 1)))
axes[0].scatter(results_df['mu_true'], results_df['mu_sd'],
                alpha=0.5, s=40, color='steelblue')
axes[0].axhline(prior_mu_sd, color='red', linestyle='--', linewidth=2,
                label=f'Prior SD = {prior_mu_sd:.3f}')
axes[0].set_xlabel('True μ', fontsize=11)
axes[0].set_ylabel('Posterior SD', fontsize=11)
axes[0].set_title(f'μ Posterior Contraction (Ratio: {summary["contraction"]["mu"]:.3f})',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Phi contraction
prior_phi_sd = np.sqrt(2 / 2**2)
axes[1].scatter(results_df['phi_true'], results_df['phi_sd'],
                alpha=0.5, s=40, color='darkorange')
axes[1].axhline(prior_phi_sd, color='red', linestyle='--', linewidth=2,
                label=f'Prior SD = {prior_phi_sd:.3f}')
axes[1].set_xlabel('True φ', fontsize=11)
axes[1].set_ylabel('Posterior SD', fontsize=11)
axes[1].set_title(f'φ Posterior Contraction (Ratio: {summary["contraction"]["phi"]:.3f})',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_contraction.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: posterior_contraction.png")

# ============================================================================
# 5. IDENTIFIABILITY: PARAMETER CORRELATIONS
# ============================================================================
print("5. Creating identifiability/correlation plot...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot true parameter space explored
scatter = ax.scatter(results_df['mu_true'], results_df['phi_true'],
                     c=np.arange(len(results_df)), cmap='viridis',
                     s=80, alpha=0.6, edgecolor='black', linewidth=0.5)
ax.set_xlabel('True μ (Mean Success Probability)', fontsize=12)
ax.set_ylabel('True φ (Concentration)', fontsize=12)
ax.set_title('Parameter Space Explored in SBC\n(Tests Identifiability Across Prior Support)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('SBC Iteration', fontsize=11)

# Add prior contours
mu_grid = np.linspace(0, 0.3, 100)
phi_grid = np.linspace(0.1, 6, 100)
MU, PHI = np.meshgrid(mu_grid, phi_grid)

# Prior densities
prior_mu = stats.beta.pdf(MU, 2, 25)
prior_phi = stats.gamma.pdf(PHI, 2, scale=0.5)
prior_joint = prior_mu * prior_phi

ax.contour(MU, PHI, prior_joint, levels=5, colors='red', alpha=0.3,
           linewidths=1.5, linestyles='--')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_space_identifiability.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: parameter_space_identifiability.png")

# ============================================================================
# 6. Z-SCORE DISTRIBUTION (Additional calibration check)
# ============================================================================
print("6. Creating z-score distribution plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Compute z-scores: (posterior_mean - true) / posterior_sd
mu_zscore = (results_df['mu_mean'] - results_df['mu_true']) / results_df['mu_sd']
phi_zscore = (results_df['phi_mean'] - results_df['phi_true']) / results_df['phi_sd']

# Mu z-scores
axes[0].hist(mu_zscore, bins=30, density=True, alpha=0.7,
             edgecolor='black', color='steelblue', label='Observed')
x = np.linspace(-4, 4, 100)
axes[0].plot(x, stats.norm.pdf(x), 'r--', linewidth=2, label='N(0,1)')
axes[0].set_xlabel('Z-score', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title(f'μ Z-scores (Mean: {mu_zscore.mean():.3f}, SD: {mu_zscore.std():.3f})',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Phi z-scores
axes[1].hist(phi_zscore, bins=30, density=True, alpha=0.7,
             edgecolor='black', color='darkorange', label='Observed')
axes[1].plot(x, stats.norm.pdf(x), 'r--', linewidth=2, label='N(0,1)')
axes[1].set_xlabel('Z-score', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title(f'φ Z-scores (Mean: {phi_zscore.mean():.3f}, SD: {phi_zscore.std():.3f})',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "zscore_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: zscore_distribution.png")

# ============================================================================
# 7. COMPREHENSIVE SUMMARY PANEL
# ============================================================================
print("7. Creating comprehensive summary panel...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Rank histograms
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(results_df['mu_rank'], bins=20, range=(0, N_SAMPLES),
         edgecolor='black', alpha=0.7, color='steelblue')
ax1.axhline(len(results_df) / 20, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Rank of True μ', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('μ Rank Histogram', fontsize=11, fontweight='bold')
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(results_df['phi_rank'], bins=20, range=(0, N_SAMPLES),
         edgecolor='black', alpha=0.7, color='darkorange')
ax2.axhline(len(results_df) / 20, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Rank of True φ', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('φ Rank Histogram', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3)

# Coverage summary text
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
summary_text = f"""
CALIBRATION SUMMARY

Coverage (95% CI):
  μ:  {summary['coverage']['mu']:.3f}
  φ:  {summary['coverage']['phi']:.3f}
  Target: 0.950

Bias:
  μ:  {summary['bias']['mu']:.4f}
  φ:  {summary['bias']['phi']:.4f}
  Target: ~0

Convergence: {summary['convergence_rate']:.3f}

Success Rate: {summary['success_rate']:.3f}
"""
ax3.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center')

# Row 2: Recovery scatter
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(results_df['mu_true'], results_df['mu_mean'],
            alpha=0.5, s=30, color='steelblue')
ax4.plot([0, 1], [0, 1], 'r--', linewidth=2)
ax4.set_xlabel('True μ', fontsize=10)
ax4.set_ylabel('Posterior Mean μ', fontsize=10)
ax4.set_title('μ Recovery', fontsize=11, fontweight='bold')
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(results_df['phi_true'], results_df['phi_mean'],
            alpha=0.5, s=30, color='darkorange')
max_phi = max(results_df['phi_true'].max(), results_df['phi_mean'].max())
ax5.plot([0, max_phi], [0, max_phi], 'r--', linewidth=2)
ax5.set_xlabel('True φ', fontsize=10)
ax5.set_ylabel('Posterior Mean φ', fontsize=10)
ax5.set_title('φ Recovery', fontsize=11, fontweight='bold')
ax5.grid(alpha=0.3)

# Parameter space
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(results_df['mu_true'], results_df['phi_true'],
            c=np.arange(len(results_df)), cmap='viridis',
            s=40, alpha=0.6)
ax6.set_xlabel('True μ', fontsize=10)
ax6.set_ylabel('True φ', fontsize=10)
ax6.set_title('Parameter Space Explored', fontsize=11, fontweight='bold')
ax6.grid(alpha=0.3)

# Row 3: Error and contraction
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(results_df['mu_true'], mu_error, alpha=0.5, s=30, color='steelblue')
ax7.axhline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('True μ', fontsize=10)
ax7.set_ylabel('Error', fontsize=10)
ax7.set_title('μ Recovery Error', fontsize=11, fontweight='bold')
ax7.grid(alpha=0.3)

ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(results_df['phi_true'], phi_error, alpha=0.5, s=30, color='darkorange')
ax8.axhline(0, color='red', linestyle='--', linewidth=2)
ax8.set_xlabel('True φ', fontsize=10)
ax8.set_ylabel('Error', fontsize=10)
ax8.set_title('φ Recovery Error', fontsize=11, fontweight='bold')
ax8.grid(alpha=0.3)

# Z-scores
ax9 = fig.add_subplot(gs[2, 2])
ax9.hist(mu_zscore, bins=20, density=True, alpha=0.5,
         color='steelblue', label='μ', edgecolor='black')
ax9.hist(phi_zscore, bins=20, density=True, alpha=0.5,
         color='darkorange', label='φ', edgecolor='black')
x = np.linspace(-4, 4, 100)
ax9.plot(x, stats.norm.pdf(x), 'r--', linewidth=2, label='N(0,1)')
ax9.set_xlabel('Z-score', fontsize=10)
ax9.set_ylabel('Density', fontsize=10)
ax9.set_title('Z-score Distribution', fontsize=11, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(alpha=0.3)

plt.suptitle('Simulation-Based Calibration: Comprehensive Summary',
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / "sbc_comprehensive_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: sbc_comprehensive_summary.png")

print("\n" + "="*70)
print("All SBC visualizations created successfully!")
print("="*70)
