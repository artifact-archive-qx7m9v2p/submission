"""
Analyze SBC results and create diagnostic plots.

Generates:
1. Rank histograms (uniformity check)
2. ECDF comparison (uniform vs empirical)
3. Recovery scatter plots (true vs recovered)
4. Computational diagnostics
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
WORKSPACE = Path('/workspace')
SBC_DIR = WORKSPACE / 'experiments' / 'experiment_1' / 'simulation_based_validation'
RESULTS_DIR = SBC_DIR / 'results'
PLOTS_DIR = SBC_DIR / 'plots'

# Load results
ranks_df = pd.read_csv(RESULTS_DIR / 'ranks.csv')
recovery_df = pd.read_csv(RESULTS_DIR / 'recovery.csv')
diagnostics_df = pd.read_csv(RESULTS_DIR / 'diagnostics.csv')

with open(RESULTS_DIR / 'sbc_results.json', 'r') as f:
    all_results = json.load(f)

# Parameters to analyze
PARAMS = ['beta_0', 'beta_1', 'beta_2', 'alpha', 'rho', 'sigma_eps']
PARAM_LABELS = {
    'beta_0': r'$\beta_0$ (Intercept)',
    'beta_1': r'$\beta_1$ (Pre-break slope)',
    'beta_2': r'$\beta_2$ (Post-break change)',
    'alpha': r'$\alpha$ (Dispersion)',
    'rho': r'$\rho$ (AR1 coef)',
    'sigma_eps': r'$\sigma_\epsilon$ (Innovation SD)'
}

# Number of posterior samples (for rank calculation)
# Each simulation: 4 chains * 500 samples = 2000
N_POSTERIOR_SAMPLES = 2000


def plot_rank_histograms():
    """
    Create rank histograms for all parameters.

    Should be approximately uniform if calibrated.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    n_bins = 20
    expected_count = len(ranks_df) / n_bins

    for i, param in enumerate(PARAMS):
        ax = axes[i]

        # Get ranks
        ranks = ranks_df[param].values

        # Plot histogram
        counts, bins, patches = ax.hist(
            ranks,
            bins=n_bins,
            edgecolor='black',
            alpha=0.7,
            label='Observed'
        )

        # Add uniform reference line
        ax.axhline(
            expected_count,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Uniform expectation'
        )

        # Add confidence band (95%)
        # Under uniformity, counts ~ Binomial(n_sims, 1/n_bins)
        n_sims = len(ranks_df)
        p = 1 / n_bins
        std_count = np.sqrt(n_sims * p * (1 - p))
        ax.fill_between(
            [bins[0], bins[-1]],
            expected_count - 2*std_count,
            expected_count + 2*std_count,
            alpha=0.2,
            color='red',
            label='95% CI'
        )

        # Chi-square test for uniformity
        expected = np.full(n_bins, expected_count)
        chi2_stat, p_value = stats.chisquare(counts, expected)

        ax.set_xlabel('Rank', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{PARAM_LABELS[param]}\n' +
                     f'χ² = {chi2_stat:.2f}, p = {p_value:.3f}',
                     fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'rank_histograms.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'rank_histograms.png'}")
    plt.close()


def plot_ecdf_comparison():
    """
    Compare empirical CDF of ranks to uniform CDF.

    Should follow diagonal if calibrated.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, param in enumerate(PARAMS):
        ax = axes[i]

        # Get ranks normalized to [0,1]
        ranks = ranks_df[param].values / N_POSTERIOR_SAMPLES

        # Sort for ECDF
        ranks_sorted = np.sort(ranks)
        ecdf = np.arange(1, len(ranks_sorted) + 1) / len(ranks_sorted)

        # Plot empirical CDF
        ax.plot(ranks_sorted, ecdf, 'b-', linewidth=2, label='Empirical CDF')

        # Plot theoretical uniform CDF (diagonal)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Uniform CDF')

        # Add confidence bands using Kolmogorov-Smirnov
        n = len(ranks_sorted)
        # 95% confidence band: ±1.36/sqrt(n)
        epsilon = 1.36 / np.sqrt(n)
        ax.fill_between(
            [0, 1],
            np.maximum(0, np.array([0, 1]) - epsilon),
            np.minimum(1, np.array([0, 1]) + epsilon),
            alpha=0.2,
            color='red',
            label='95% KS band'
        )

        # KS test
        ks_stat, p_value = stats.kstest(ranks, 'uniform')

        ax.set_xlabel('Rank quantile', fontsize=10)
        ax.set_ylabel('Cumulative probability', fontsize=10)
        ax.set_title(f'{PARAM_LABELS[param]}\n' +
                     f'KS = {ks_stat:.3f}, p = {p_value:.3f}',
                     fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ecdf_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'ecdf_comparison.png'}")
    plt.close()


def plot_recovery_scatter():
    """
    Scatter plots of true vs recovered (mean) parameters.

    Should follow y=x diagonal if unbiased.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, param in enumerate(PARAMS):
        ax = axes[i]

        # Extract true and recovered values
        true_vals = recovery_df[f'{param}_true'].values
        recovered_vals = recovery_df[f'{param}_mean'].values

        # Scatter plot
        ax.scatter(true_vals, recovered_vals, alpha=0.5, s=30)

        # Add y=x line
        lims = [
            min(true_vals.min(), recovered_vals.min()),
            max(true_vals.max(), recovered_vals.max())
        ]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect recovery')

        # Compute bias and correlation
        bias = np.mean(recovered_vals - true_vals)
        corr = np.corrcoef(true_vals, recovered_vals)[0, 1]
        rmse = np.sqrt(np.mean((recovered_vals - true_vals)**2))

        ax.set_xlabel(f'True {param}', fontsize=10)
        ax.set_ylabel(f'Recovered {param}', fontsize=10)
        ax.set_title(f'{PARAM_LABELS[param]}\n' +
                     f'Bias={bias:.3f}, r={corr:.3f}, RMSE={rmse:.3f}',
                     fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'recovery_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'recovery_scatter.png'}")
    plt.close()


def plot_computational_diagnostics():
    """
    Distribution of Rhat, ESS, divergences across simulations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rhat distribution
    ax = axes[0, 0]
    ax.hist(diagnostics_df['max_rhat'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(1.01, color='orange', linestyle='--', linewidth=2, label='Warning (1.01)')
    ax.axvline(1.05, color='red', linestyle='--', linewidth=2, label='Fail (1.05)')
    ax.set_xlabel('Max Rhat', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Max Rhat', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ESS distribution
    ax = axes[0, 1]
    ax.hist(diagnostics_df['min_ess_bulk'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(400, color='orange', linestyle='--', linewidth=2, label='Good (400)')
    ax.axvline(100, color='red', linestyle='--', linewidth=2, label='Minimum (100)')
    ax.set_xlabel('Min ESS Bulk', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Min ESS Bulk', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Divergences
    ax = axes[1, 0]
    ax.hist(diagnostics_df['divergences'], bins=range(0, diagnostics_df['divergences'].max()+2),
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Divergences', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Divergences per Simulation', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Max treedepth hits
    ax = axes[1, 1]
    ax.hist(diagnostics_df['max_treedepth_hits'],
            bins=range(0, diagnostics_df['max_treedepth_hits'].max()+2),
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Max Treedepth Hits', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Max Treedepth Hits per Simulation', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'computational_diagnostics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'computational_diagnostics.png'}")
    plt.close()


def compute_summary_statistics():
    """
    Compute and print summary statistics for SBC.
    """
    print("\n" + "="*80)
    print("SBC SUMMARY STATISTICS")
    print("="*80)

    # Overall success rate
    n_success = len(ranks_df)
    n_total = len(all_results)
    print(f"\nSimulations: {n_success}/{n_total} successful ({n_success/n_total*100:.1f}%)")

    # Convergence statistics
    print(f"\nConvergence:")
    print(f"  Max Rhat: {diagnostics_df['max_rhat'].max():.4f}")
    print(f"  Mean Rhat: {diagnostics_df['max_rhat'].mean():.4f}")
    print(f"  Min ESS Bulk: {diagnostics_df['min_ess_bulk'].min():.0f}")
    print(f"  Mean ESS Bulk: {diagnostics_df['min_ess_bulk'].mean():.0f}")
    print(f"  Converged (Rhat<1.05, ESS>100): {diagnostics_df['converged'].sum()}/{n_success}")

    # Divergences
    print(f"\nDivergences:")
    print(f"  Total: {diagnostics_df['divergences'].sum()}")
    print(f"  Simulations with divergences: {(diagnostics_df['divergences'] > 0).sum()}")
    print(f"  Max per simulation: {diagnostics_df['divergences'].max()}")

    # Calibration tests
    print(f"\n" + "="*80)
    print("CALIBRATION TESTS (p-values > 0.05 indicates good calibration)")
    print("="*80)
    print(f"{'Parameter':<20} {'Chi-square':<15} {'KS test':<15}")
    print("-"*80)

    n_bins = 20
    for param in PARAMS:
        # Chi-square test
        ranks = ranks_df[param].values
        counts, _ = np.histogram(ranks, bins=n_bins)
        expected = np.full(n_bins, len(ranks_df) / n_bins)
        chi2_stat, chi2_p = stats.chisquare(counts, expected)

        # KS test
        ranks_norm = ranks / N_POSTERIOR_SAMPLES
        ks_stat, ks_p = stats.kstest(ranks_norm, 'uniform')

        print(f"{PARAM_LABELS[param]:<20} {chi2_p:>8.4f}        {ks_p:>8.4f}")

    # Recovery bias
    print(f"\n" + "="*80)
    print("RECOVERY BIAS (Mean error: Recovered - True)")
    print("="*80)
    print(f"{'Parameter':<20} {'Bias':<12} {'RMSE':<12} {'Correlation':<12}")
    print("-"*80)

    for param in PARAMS:
        true_vals = recovery_df[f'{param}_true'].values
        recovered_vals = recovery_df[f'{param}_mean'].values

        bias = np.mean(recovered_vals - true_vals)
        rmse = np.sqrt(np.mean((recovered_vals - true_vals)**2))
        corr = np.corrcoef(true_vals, recovered_vals)[0, 1]

        print(f"{PARAM_LABELS[param]:<20} {bias:>8.4f}     {rmse:>8.4f}     {corr:>8.4f}")


def generate_pass_fail_assessment():
    """
    Generate final PASS/FAIL assessment.
    """
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)

    issues = []

    # Check failure rate
    n_success = len(ranks_df)
    n_total = len(all_results)
    failure_rate = (n_total - n_success) / n_total

    if failure_rate > 0.10:
        issues.append(f"HIGH FAILURE RATE: {failure_rate*100:.1f}% (threshold: 10%)")

    # Check convergence
    n_converged = diagnostics_df['converged'].sum()
    if n_converged < n_success * 0.9:
        issues.append(f"POOR CONVERGENCE: Only {n_converged}/{n_success} converged")

    # Check calibration
    failed_calibration = []
    for param in PARAMS:
        ranks = ranks_df[param].values
        ranks_norm = ranks / N_POSTERIOR_SAMPLES
        ks_stat, ks_p = stats.kstest(ranks_norm, 'uniform')

        if ks_p < 0.01:  # Very strict threshold
            failed_calibration.append(f"{param} (p={ks_p:.4f})")

    if failed_calibration:
        issues.append(f"CALIBRATION FAILURES: {', '.join(failed_calibration)}")

    # Check for systematic bias
    biased_params = []
    for param in PARAMS:
        true_vals = recovery_df[f'{param}_true'].values
        recovered_vals = recovery_df[f'{param}_mean'].values
        bias = np.mean(recovered_vals - true_vals)
        std_true = np.std(true_vals)

        # Flag if bias > 10% of prior std
        if abs(bias) > 0.1 * std_true:
            biased_params.append(f"{param} (bias={bias:.4f})")

    if biased_params:
        issues.append(f"SYSTEMATIC BIAS: {', '.join(biased_params)}")

    # Final verdict
    if not issues:
        print("\n✓ PASS: Model shows good calibration")
        print("  - Rank histograms approximately uniform")
        print("  - No systematic bias detected")
        print("  - Convergence reliable")
        print("  - Ready to proceed to real data fitting")
        verdict = "PASS"
    else:
        print("\n✗ ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")

        if failure_rate > 0.20 or n_converged < n_success * 0.8:
            print("\n✗ FAIL: Critical issues - DO NOT proceed to real data")
            verdict = "FAIL"
        else:
            print("\n⚠ INVESTIGATE: Minor issues - proceed with caution")
            verdict = "INVESTIGATE"

    print("="*80)
    return verdict


if __name__ == '__main__':
    print("Generating SBC diagnostic plots...")

    # Create plots
    plot_rank_histograms()
    plot_ecdf_comparison()
    plot_recovery_scatter()
    plot_computational_diagnostics()

    # Print summary statistics
    compute_summary_statistics()

    # Generate assessment
    verdict = generate_pass_fail_assessment()

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print(f"\nFinal verdict: {verdict}")
