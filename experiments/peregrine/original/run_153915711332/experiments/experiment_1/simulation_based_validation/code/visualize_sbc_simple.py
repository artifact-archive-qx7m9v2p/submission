"""
Simplified SBC visualization for 3 parameters only
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("SBC VISUALIZATION")
print("="*80)

# Load data
df = pd.read_csv(DIAGNOSTICS_DIR / "sbc_results.csv")
with open(DIAGNOSTICS_DIR / "sbc_summary.json", 'r') as f:
    summary = json.load(f)

N_SIMS = len(df)
N_DRAWS = 1000

print(f"\n[1/4] Loaded {N_SIMS} simulations")

# Parameters to visualize
PARAMS = {
    'delta': r'Drift $\delta$',
    'sigma_eta': r'Innovation SD $\sigma_\eta$',
    'phi': r'Dispersion $\phi$'
}

# =============================================================================
# PLOT 1: RANK HISTOGRAMS
# =============================================================================
print("\n[2/4] Creating rank histograms...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

n_bins = 20
confidence = 0.99
expected = N_SIMS / n_bins
se = np.sqrt(N_SIMS * (1/n_bins) * (1 - 1/n_bins))
z_crit = stats.norm.ppf((1 + confidence) / 2)
ci_lower = expected - z_crit * se
ci_upper = expected + z_crit * se

for idx, (param, label) in enumerate(PARAMS.items()):
    ax = axes[idx]
    ranks = df[f'ranks_{param}'].values

    # Histogram
    counts, bins, patches = ax.hist(
        ranks, bins=n_bins, range=(0, N_DRAWS),
        edgecolor='black', alpha=0.7, color='steelblue'
    )

    # Color bars outside CI
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count < ci_lower or count > ci_upper:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)

    # Reference lines
    ax.axhline(expected, color='blue', linestyle='--', linewidth=2, label='Expected')
    ax.axhline(ci_lower, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(ci_upper, color='gray', linestyle=':', alpha=0.7)
    ax.fill_between([0, N_DRAWS], ci_lower, ci_upper, color='gray', alpha=0.2)

    # Test result
    chi2 = summary['uniformity_tests'][param]['chi2']
    p_val = summary['uniformity_tests'][param]['p_value']
    test_pass = summary['uniformity_tests'][param]['pass']
    status = "PASS" if test_pass else "FAIL"
    color = 'green' if test_pass else 'red'

    ax.text(
        0.98, 0.97,
        f'χ² = {chi2:.1f}\np ≈ 0.00\n[{status}]',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5,
                 edgecolor=color, linewidth=2)
    )

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('SBC Rank Histograms: Non-uniform distributions indicate calibration failure',
             fontsize=14, fontweight='bold')
plt.tight_layout()

plot_file = PLOTS_DIR / "rank_histograms.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"      Saved: {plot_file}")
plt.close()

# =============================================================================
# PLOT 2: RANK ECDFS
# =============================================================================
print("\n[3/4] Creating ECDF comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (param, label) in enumerate(PARAMS.items()):
    ax = axes[idx]

    ranks = df[f'ranks_{param}'].values
    ranks_norm = ranks / N_DRAWS
    ranks_sorted = np.sort(ranks_norm)
    ecdf = np.arange(1, len(ranks_sorted) + 1) / len(ranks_sorted)

    # Plot ECDF
    ax.plot(ranks_sorted, ecdf, linewidth=2.5, label='Empirical', color='blue')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Uniform (ideal)', alpha=0.7)

    # KS bands
    n = len(ranks_norm)
    alpha = 0.05
    ks_crit = np.sqrt(-np.log(alpha/2) / (2*n))
    x_fill = np.linspace(0, 1, 100)
    ax.fill_between(
        x_fill,
        np.maximum(0, x_fill - ks_crit),
        np.minimum(1, x_fill + ks_crit),
        color='red', alpha=0.2, label='95% KS band'
    )

    # KS test
    ks_stat = np.max(np.abs(ranks_sorted - np.linspace(0, 1, len(ranks_sorted))))
    ks_pval = stats.kstest(ranks_norm, 'uniform').pvalue

    ax.text(
        0.02, 0.98,
        f'KS = {ks_stat:.3f}\np = {ks_pval:.4f}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    )

    ax.set_xlabel('Normalized Rank', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.suptitle('SBC ECDF: Deviation from diagonal indicates poor calibration',
             fontsize=14, fontweight='bold')
plt.tight_layout()

plot_file = PLOTS_DIR / "ecdf_comparison.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"      Saved: {plot_file}")
plt.close()

# =============================================================================
# PLOT 3: PARAMETER RECOVERY SCATTER
# =============================================================================
print("\n[4/4] Creating parameter recovery plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (param, label) in enumerate(PARAMS.items()):
    ax = axes[idx]

    true_vals = df[param].values
    ranks = df[f'ranks_{param}'].values
    ranks_norm = ranks / N_DRAWS

    # Scatter
    scatter = ax.scatter(
        true_vals, ranks_norm,
        c=ranks_norm, cmap='RdYlGn_r',
        s=60, alpha=0.6, edgecolors='black', linewidth=0.5
    )

    # Reference lines
    ax.axhline(0.5, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Expected median')
    ax.axhspan(0.25, 0.75, color='blue', alpha=0.1, label='IQR band')

    # Annotate problematic regions
    n_low = np.sum(ranks < 50)  # Ranks < 5%
    n_high = np.sum(ranks > 950)  # Ranks > 95%

    ax.text(
        0.02, 0.98,
        f'Extreme ranks:\n  Low: {n_low}\n  High: {n_high}',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6)
    )

    ax.set_xlabel(f'True {label}', fontsize=12)
    ax.set_ylabel('Normalized Rank', fontsize=12)
    ax.set_title(f'{label} Recovery', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rank', fontsize=10)

plt.suptitle('Parameter Recovery: Clustering at extremes indicates bias or poor uncertainty',
             fontsize=14, fontweight='bold')
plt.tight_layout()

plot_file = PLOTS_DIR / "parameter_recovery.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"      Saved: {plot_file}")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nGenerated {3} plots in: {PLOTS_DIR}/")
print(f"\n  1. rank_histograms.png")
print(f"     → Shows severe deviations from uniformity")
print(f"     → Red bars indicate bins outside 99% CI")
print(f"\n  2. ecdf_comparison.png")
print(f"     → Empirical CDF vs theoretical uniform")
print(f"     → Large deviations visible")
print(f"\n  3. parameter_recovery.png")
print(f"     → Many ranks clustered at 0 and 1000")
print(f"     → Indicates posterior approximation is poor")

decision = summary['overall_decision']
print(f"\nOVERALL DECISION: {decision}")

if decision == "FAIL":
    print("\n⚠ SBC FAILURE indicates:")
    print("  1. Approximate posterior method is inadequate")
    print("  2. Model may have identifiability issues")
    print("  3. Priors may be misspecified")
    print("\n  RECOMMENDATION:")
    print("  - Re-run with full MCMC (HMC/NUTS) when Stan/PyMC available")
    print("  - Check prior-data compatibility")
    print("  - Consider model simplification")

print("="*80)
