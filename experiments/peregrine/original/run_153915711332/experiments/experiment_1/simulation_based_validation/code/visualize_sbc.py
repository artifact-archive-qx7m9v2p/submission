"""
Visualization of Simulation-Based Calibration Results

Creates comprehensive diagnostic plots to assess:
1. Rank uniformity (histograms with confidence bands)
2. ECDF comparison with theoretical uniform
3. Convergence diagnostics across simulations
4. Parameter recovery patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

# Create plots directory
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("SBC VISUALIZATION")
print("="*80)

# Load results
print("\n[1/5] Loading SBC results...")
df = pd.read_csv(DIAGNOSTICS_DIR / "sbc_results.csv")
with open(DIAGNOSTICS_DIR / "sbc_summary.json", 'r') as f:
    summary = json.load(f)

N_SIMS = len(df)
N_DRAWS = 1000  # As specified in run_sbc.py

print(f"      Loaded {N_SIMS} simulations")

# Parameter metadata
PARAMS = {
    'delta': {'label': r'Drift $\delta$', 'units': 'log-scale/period'},
    'sigma_eta': {'label': r'Innovation SD $\sigma_\eta$', 'units': 'log-scale'},
    'phi': {'label': r'Dispersion $\phi$', 'units': 'counts'},
    'eta_1': {'label': r'Initial state $\eta_1$', 'units': 'log-scale'}
}

# ============================================================================
# PLOT 1: RANK HISTOGRAMS WITH UNIFORMITY BANDS
# ============================================================================
print("\n[2/5] Creating rank histograms...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

n_bins = 20
confidence = 0.99  # 99% confidence band

# Compute expected counts and confidence intervals
expected = N_SIMS / n_bins
# Under null hypothesis of uniformity, each bin follows Binomial(N_SIMS, 1/n_bins)
se = np.sqrt(N_SIMS * (1/n_bins) * (1 - 1/n_bins))
z_crit = stats.norm.ppf((1 + confidence) / 2)
ci_lower = expected - z_crit * se
ci_upper = expected + z_crit * se

for idx, param in enumerate(PARAMS.keys()):
    ax = axes[idx]

    ranks = df[f'ranks_{param}'].values

    # Plot histogram
    counts, bins, patches = ax.hist(
        ranks,
        bins=n_bins,
        range=(0, N_DRAWS),
        edgecolor='black',
        alpha=0.7,
        label='Observed'
    )

    # Color bars outside confidence interval
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count < ci_lower or count > ci_upper:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)

    # Add expected value and confidence bands
    ax.axhline(expected, color='blue', linestyle='--', linewidth=2, label='Expected (uniform)')
    ax.axhline(ci_lower, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'{confidence*100:.0f}% CI')
    ax.axhline(ci_upper, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.fill_between([0, N_DRAWS], ci_lower, ci_upper, color='gray', alpha=0.2)

    # Add chi-square test result
    chi2 = summary['uniformity_tests'][param]['chi2']
    p_val = summary['uniformity_tests'][param]['p_value']
    test_pass = summary['uniformity_tests'][param]['pass']
    status = "PASS" if test_pass else "FAIL"
    color = 'green' if test_pass else 'red'

    ax.text(
        0.98, 0.97,
        f'χ² = {chi2:.2f}\np = {p_val:.4f}\n[{status}]',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor=color, linewidth=2)
    )

    # Labels
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{PARAMS[param]["label"]}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('SBC Rank Histograms: Uniformity Assessment', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

plot_file = PLOTS_DIR / "rank_histograms.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"      Saved: {plot_file}")
plt.close()

# ============================================================================
# PLOT 2: ECDF COMPARISON
# ============================================================================
print("\n[3/5] Creating ECDF comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, param in enumerate(PARAMS.keys()):
    ax = axes[idx]

    ranks = df[f'ranks_{param}'].values
    # Normalize ranks to [0, 1]
    ranks_normalized = ranks / N_DRAWS

    # Compute empirical CDF
    ranks_sorted = np.sort(ranks_normalized)
    ecdf = np.arange(1, len(ranks_sorted) + 1) / len(ranks_sorted)

    # Plot empirical CDF
    ax.plot(ranks_sorted, ecdf, linewidth=2, label='Empirical CDF', color='blue')

    # Plot theoretical uniform CDF
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Uniform CDF', alpha=0.7)

    # Add confidence bands (Kolmogorov-Smirnov)
    n = len(ranks_normalized)
    alpha = 0.05
    ks_crit = np.sqrt(-np.log(alpha/2) / (2*n))

    x_fill = np.linspace(0, 1, 100)
    ax.fill_between(
        x_fill,
        np.maximum(0, x_fill - ks_crit),
        np.minimum(1, x_fill + ks_crit),
        color='red',
        alpha=0.2,
        label=f'95% KS band'
    )

    # Compute KS statistic
    ks_stat = np.max(np.abs(ranks_sorted - np.linspace(0, 1, len(ranks_sorted))))
    ks_pval = stats.kstest(ranks_normalized, 'uniform').pvalue

    ax.text(
        0.02, 0.98,
        f'KS = {ks_stat:.4f}\np = {ks_pval:.4f}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    )

    # Labels
    ax.set_xlabel('Normalized Rank', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{PARAMS[param]["label"]}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.suptitle('SBC ECDF Comparison: Deviation from Uniformity', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

plot_file = PLOTS_DIR / "ecdf_comparison.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"      Saved: {plot_file}")
plt.close()

# ============================================================================
# PLOT 3: CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[4/5] Creating convergence diagnostic plots...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# R-hat distribution
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['rhat_max'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(1.01, color='red', linestyle='--', linewidth=2, label='Threshold (1.01)')
ax1.set_xlabel(r'Maximum $\hat{R}$', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Maximum R-hat Across Simulations', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# R-hat failures
n_rhat_fail = (df['rhat_max'] > 1.01).sum()
ax1.text(
    0.98, 0.97,
    f'Failures: {n_rhat_fail}/{N_SIMS} ({n_rhat_fail/N_SIMS*100:.1f}%)',
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
)

# ESS distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df['ess_bulk_min'], bins=30, edgecolor='black', alpha=0.7, color='darkgreen')
ax2.axvline(400, color='red', linestyle='--', linewidth=2, label='Threshold (400)')
ax2.set_xlabel('Min Bulk ESS', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Minimum ESS', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Divergences
ax3 = fig.add_subplot(gs[1, :2])
ax3.hist(df['n_divergences'], bins=np.arange(0, df['n_divergences'].max()+2)-0.5,
         edgecolor='black', alpha=0.7, color='coral')
ax3.set_xlabel('Number of Divergences', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Divergences per Simulation', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

total_divs = df['n_divergences'].sum()
sims_with_divs = (df['n_divergences'] > 0).sum()
ax3.text(
    0.98, 0.97,
    f'Total: {total_divs}\nSims affected: {sims_with_divs}/{N_SIMS}',
    transform=ax3.transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
)

# Convergence rate
ax4 = fig.add_subplot(gs[1, 2])
converged = df['converged'].sum()
failed = N_SIMS - converged
ax4.bar(['Converged', 'Failed'], [converged, failed], color=['green', 'red'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Overall Convergence', fontsize=12, fontweight='bold')
ax4.text(0, converged + 0.5, f'{converged}\n({converged/N_SIMS*100:.1f}%)',
         ha='center', fontsize=10, fontweight='bold')
if failed > 0:
    ax4.text(1, failed + 0.5, f'{failed}\n({failed/N_SIMS*100:.1f}%)',
             ha='center', fontsize=10, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# R-hat vs ESS scatter
ax5 = fig.add_subplot(gs[2, :])
scatter = ax5.scatter(df['rhat_max'], df['ess_bulk_min'],
                     c=df['n_divergences'], cmap='YlOrRd',
                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax5.axvline(1.01, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label=r'$\hat{R}$ threshold')
ax5.axhline(400, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='ESS threshold')
ax5.set_xlabel(r'Maximum $\hat{R}$', fontsize=11)
ax5.set_ylabel('Minimum Bulk ESS', fontsize=11)
ax5.set_title('Convergence Landscape: R-hat vs ESS (colored by divergences)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Divergences', fontsize=10)

plt.suptitle('Computational Diagnostics Across SBC Simulations', fontsize=14, fontweight='bold', y=0.998)

plot_file = PLOTS_DIR / "convergence_diagnostics.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"      Saved: {plot_file}")
plt.close()

# ============================================================================
# PLOT 4: PARAMETER RECOVERY SCATTER
# ============================================================================
print("\n[5/5] Creating parameter recovery scatter plots...")

# For each parameter, we need to show true values vs some measure of recovery
# Since we don't have posterior means stored individually, we'll use rank positions
# as a proxy for recovery quality

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, param in enumerate(PARAMS.keys()):
    ax = axes[idx]

    true_vals = df[param].values
    ranks = df[f'ranks_{param}'].values
    # Normalize ranks to [0, 1] - perfectly calibrated should be uniform
    ranks_norm = ranks / N_DRAWS

    # Scatter plot
    scatter = ax.scatter(
        true_vals,
        ranks_norm,
        c=df['converged'],
        cmap='RdYlGn',
        s=40,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Add reference line at 0.5 (median rank)
    ax.axhline(0.5, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Expected median')

    # Add confidence band (25th-75th percentile)
    ax.axhspan(0.25, 0.75, color='blue', alpha=0.1, label='IQR band')

    # Labels
    ax.set_xlabel(f'True {PARAMS[param]["label"]} ({PARAMS[param]["units"]})', fontsize=11)
    ax.set_ylabel('Normalized Rank', fontsize=11)
    ax.set_title(f'{PARAMS[param]["label"]} Recovery', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Color bar for convergence
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Converged', fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No', 'Yes'])

plt.suptitle('Parameter Recovery: True Value vs Rank Position', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

plot_file = PLOTS_DIR / "parameter_recovery.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"      Saved: {plot_file}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nGenerated plots:")
print(f"  1. {PLOTS_DIR}/rank_histograms.png")
print(f"     - Tests uniformity of rank statistics")
print(f"     - Red bars indicate deviations outside 99% CI")
print(f"  2. {PLOTS_DIR}/ecdf_comparison.png")
print(f"     - Empirical CDF vs theoretical uniform")
print(f"     - KS test for distributional match")
print(f"  3. {PLOTS_DIR}/convergence_diagnostics.png")
print(f"     - R-hat, ESS, divergences across simulations")
print(f"     - Overall computational health assessment")
print(f"  4. {PLOTS_DIR}/parameter_recovery.png")
print(f"     - True parameter values vs rank positions")
print(f"     - Should show uniform spread across rank space")
print("="*80)

# Print overall assessment
decision = summary.get('overall_decision', 'UNKNOWN')
print(f"\nOVERALL SBC DECISION: {decision}")

if decision == "PASS":
    print("\n✓ Model demonstrates computational faithfulness")
    print("  - Rank statistics are uniformly distributed")
    print("  - Sampler converges reliably")
    print("  - Low divergence rate")
    print("\n  RECOMMENDATION: Proceed to fit real data")
elif "FAIL" in decision:
    print("\n⚠ Model shows calibration or computational issues")
    print("  - Review rank histograms for specific parameter problems")
    print("  - Check convergence diagnostics for sampler issues")
    print("\n  RECOMMENDATION: Address issues before real data fitting")

print("="*80)
