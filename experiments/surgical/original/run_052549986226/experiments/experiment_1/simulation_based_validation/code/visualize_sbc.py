"""
Visualization for Simulation-Based Calibration Results

Creates diagnostic plots to assess parameter recovery quality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
WORKSPACE = Path("/workspace")
RESULTS_DIR = WORKSPACE / "experiments" / "experiment_1" / "simulation_based_validation" / "results"
PLOTS_DIR = WORKSPACE / "experiments" / "experiment_1" / "simulation_based_validation" / "plots"

# Load results
df = pd.read_csv(RESULTS_DIR / "sbc_results.csv")
df_conv = df[df['converged']].copy()

print(f"Loaded {len(df)} simulations ({len(df_conv)} converged)")

# ============================================================================
# Plot 1: Parameter Recovery Scatter Plots
# ============================================================================

def plot_parameter_recovery():
    """
    Scatter plots: true vs posterior mean for mu, kappa, phi.
    Shows if model can recover true parameters without bias.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    params = [
        ('mu', 'mu', r'$\mu$ (Population Mean)'),
        ('kappa', 'kappa', r'$\kappa$ (Concentration)'),
        ('phi', 'phi', r'$\phi$ (Overdispersion)')
    ]

    for ax, (param, param_name, label) in zip(axes, params):
        # Extract data
        true_vals = df_conv[f'true_{param}']
        post_means = df_conv[f'post_{param}_mean']
        post_lower = df_conv[f'post_{param}_q2.5']
        post_upper = df_conv[f'post_{param}_q97.5']

        # Recovery status
        recovered = df_conv[f'{param}_recovered_95']

        # Calculate error bars (ensure positive)
        yerr_lower = np.abs(post_means - post_lower)
        yerr_upper = np.abs(post_upper - post_means)

        # Scatter plot with error bars
        colors = ['green' if r else 'red' for r in recovered]
        ax.errorbar(true_vals, post_means,
                   yerr=[yerr_lower, yerr_upper],
                   fmt='o', alpha=0.6, capsize=3, markersize=6,
                   ecolor='gray', markerfacecolor='none', markeredgecolor='gray')

        # Color by recovery status
        for i, (x, y, c) in enumerate(zip(true_vals, post_means, colors)):
            ax.plot(x, y, 'o', color=c, markersize=6, alpha=0.7)

        # Identity line (perfect recovery)
        lim_min = min(true_vals.min(), post_means.min())
        lim_max = max(true_vals.max(), post_means.max())
        margin = (lim_max - lim_min) * 0.1
        ax.plot([lim_min - margin, lim_max + margin],
               [lim_min - margin, lim_max + margin],
               'k--', lw=2, alpha=0.5, label='Perfect recovery')

        ax.set_xlabel(f'True {label}', fontsize=12)
        ax.set_ylabel(f'Posterior Mean {label}', fontsize=12)
        ax.set_title(f'{label} Recovery', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add coverage rate
        coverage = recovered.sum() / len(recovered)
        ax.text(0.05, 0.95, f'Coverage: {coverage*100:.0f}%',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = PLOTS_DIR / "parameter_recovery.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# Plot 2: Coverage Diagnostic with Error Bars
# ============================================================================

def plot_coverage_diagnostic():
    """
    Shows true parameters vs 95% credible intervals.
    Visualizes which parameters are recovered and which are not.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    params = [
        ('mu', r'$\mu$ (Population Mean)'),
        ('kappa', r'$\kappa$ (Concentration)'),
        ('phi', r'$\phi$ (Overdispersion)')
    ]

    for ax, (param, label) in zip(axes, params):
        # Extract data (sort by true value for clearer visualization)
        sorted_idx = df_conv[f'true_{param}'].argsort()
        true_vals = df_conv[f'true_{param}'].values[sorted_idx]
        post_lower = df_conv[f'post_{param}_q2.5'].values[sorted_idx]
        post_upper = df_conv[f'post_{param}_q97.5'].values[sorted_idx]
        post_means = df_conv[f'post_{param}_mean'].values[sorted_idx]
        recovered = df_conv[f'{param}_recovered_95'].values[sorted_idx]

        n = len(true_vals)
        x = np.arange(n)

        # Plot credible intervals
        for i in range(n):
            color = 'green' if recovered[i] else 'red'
            alpha = 0.7 if recovered[i] else 0.9
            ax.plot([i, i], [post_lower[i], post_upper[i]],
                   color=color, lw=2, alpha=alpha)

        # Plot posterior means
        ax.scatter(x, post_means, color='blue', s=30, alpha=0.6,
                  label='Posterior mean', zorder=3)

        # Plot true values
        ax.scatter(x, true_vals, color='black', marker='x', s=50,
                  label='True value', zorder=4)

        ax.set_xlabel('Simulation (sorted by true value)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}: 95% Credible Intervals vs True Values',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Add coverage annotation
        coverage = recovered.sum() / n
        ax.text(0.02, 0.98, f'Coverage: {coverage*100:.0f}% ({recovered.sum()}/{n})',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=10)

    plt.tight_layout()
    output_file = PLOTS_DIR / "coverage_diagnostic.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# Plot 3: Interval Calibration (CI Widths)
# ============================================================================

def plot_interval_calibration():
    """
    Shows distribution of 95% CI widths across simulations.
    Checks if intervals are reasonable (not too narrow or wide).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    params = [
        ('mu', r'$\mu$', 'Population Mean'),
        ('kappa', r'$\kappa$', 'Concentration'),
        ('phi', r'$\phi$', 'Overdispersion')
    ]

    for ax, (param, symbol, name) in zip(axes, params):
        # Calculate CI widths
        widths = df_conv[f'post_{param}_q97.5'] - df_conv[f'post_{param}_q2.5']

        # Histogram
        ax.hist(widths, bins=15, alpha=0.7, color='steelblue', edgecolor='black')

        # Add mean line
        mean_width = widths.mean()
        ax.axvline(mean_width, color='red', linestyle='--', lw=2,
                  label=f'Mean: {mean_width:.4f}')

        ax.set_xlabel(f'95% CI Width for {symbol}', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{name} Credible Interval Widths', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = PLOTS_DIR / "interval_calibration.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# Plot 4: Bias Assessment
# ============================================================================

def plot_bias_assessment():
    """
    Shows distribution of estimation bias (posterior mean - true value).
    Should be centered near zero if unbiased.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    params = [
        ('mu', r'$\mu$', 'Population Mean'),
        ('kappa', r'$\kappa$', 'Concentration'),
        ('phi', r'$\phi$', 'Overdispersion')
    ]

    for ax, (param, symbol, name) in zip(axes, params):
        # Calculate bias
        bias = df_conv[f'post_{param}_mean'] - df_conv[f'true_{param}']

        # Histogram
        ax.hist(bias, bins=15, alpha=0.7, color='steelblue', edgecolor='black')

        # Zero line (no bias)
        ax.axvline(0, color='green', linestyle='-', lw=2, label='Unbiased')

        # Mean bias
        mean_bias = bias.mean()
        ax.axvline(mean_bias, color='red', linestyle='--', lw=2,
                  label=f'Mean: {mean_bias:+.4f}')

        ax.set_xlabel(f'Bias in {symbol} (Posterior Mean - True)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{name} Estimation Bias', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add standard deviation
        std_bias = bias.std()
        ax.text(0.98, 0.98, f'SD: {std_bias:.4f}',
               transform=ax.transAxes, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = PLOTS_DIR / "bias_assessment.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# Plot 5: Comprehensive Summary
# ============================================================================

def plot_comprehensive_summary():
    """
    Multi-panel plot showing key diagnostics in one view.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Parameter recovery scatter plots (3 panels)
    params = [
        ('mu', r'$\mu$'),
        ('kappa', r'$\kappa$'),
        ('phi', r'$\phi$')
    ]

    for col, (param, symbol) in enumerate(params):
        ax = fig.add_subplot(gs[0, col])

        true_vals = df_conv[f'true_{param}']
        post_means = df_conv[f'post_{param}_mean']
        recovered = df_conv[f'{param}_recovered_95']

        # Scatter
        colors = ['green' if r else 'red' for r in recovered]
        for x, y, c in zip(true_vals, post_means, colors):
            ax.plot(x, y, 'o', color=c, markersize=8, alpha=0.7)

        # Identity line
        lim_min = min(true_vals.min(), post_means.min())
        lim_max = max(true_vals.max(), post_means.max())
        margin = (lim_max - lim_min) * 0.1
        ax.plot([lim_min - margin, lim_max + margin],
               [lim_min - margin, lim_max + margin],
               'k--', lw=2, alpha=0.5)

        ax.set_xlabel(f'True {symbol}', fontsize=10)
        ax.set_ylabel(f'Posterior {symbol}', fontsize=10)
        ax.set_title(f'{symbol} Recovery', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        coverage = recovered.sum() / len(recovered)
        ax.text(0.05, 0.95, f'{coverage*100:.0f}%',
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 2: Bias distributions (3 panels)
    for col, (param, symbol) in enumerate(params):
        ax = fig.add_subplot(gs[1, col])

        bias = df_conv[f'post_{param}_mean'] - df_conv[f'true_{param}']

        ax.hist(bias, bins=12, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='green', linestyle='-', lw=2)
        ax.axvline(bias.mean(), color='red', linestyle='--', lw=2)

        ax.set_xlabel(f'Bias in {symbol}', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{symbol} Bias: {bias.mean():+.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # Row 3: Coverage summary and key metrics
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')

    # Summary statistics
    summary_text = "SIMULATION-BASED CALIBRATION SUMMARY\n" + "="*60 + "\n\n"
    summary_text += f"Total Simulations: {len(df)}\n"
    summary_text += f"Converged: {len(df_conv)} ({len(df_conv)/len(df)*100:.0f}%)\n\n"

    summary_text += "Coverage Rates (95% CI):\n"
    for param, symbol in params:
        coverage = df_conv[f'{param}_recovered_95'].sum() / len(df_conv)
        status = "PASS" if coverage >= 0.85 else "FAIL"
        summary_text += f"  {symbol}: {coverage*100:.1f}% [{status}]\n"

    summary_text += "\nMean Bias:\n"
    for param, symbol in params:
        bias = (df_conv[f'post_{param}_mean'] - df_conv[f'true_{param}']).mean()
        summary_text += f"  {symbol}: {bias:+.6f}\n"

    summary_text += "\nMean 95% CI Width:\n"
    for param, symbol in params:
        width = (df_conv[f'post_{param}_q97.5'] - df_conv[f'post_{param}_q2.5']).mean()
        summary_text += f"  {symbol}: {width:.4f}\n"

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           va='center', ha='left')

    plt.suptitle('Simulation-Based Calibration: Beta-Binomial Model',
                fontsize=14, fontweight='bold', y=0.98)

    output_file = PLOTS_DIR / "comprehensive_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Generate all diagnostic plots."""
    print("\nGenerating SBC diagnostic plots...\n")

    plot_parameter_recovery()
    plot_coverage_diagnostic()
    plot_interval_calibration()
    plot_bias_assessment()
    plot_comprehensive_summary()

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
