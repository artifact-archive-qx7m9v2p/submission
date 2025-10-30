"""
Analyze Simulation-Based Calibration Results

This script analyzes the SBC results and creates final visualizations and metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Paths
WORKSPACE = Path("/workspace")
EXPERIMENT_DIR = WORKSPACE / "experiments" / "experiment_1"
SBC_DIR = EXPERIMENT_DIR / "simulation_based_validation"
CODE_DIR = SBC_DIR / "code"
PLOTS_DIR = SBC_DIR / "plots"

# Load results
df_results = pd.read_csv(CODE_DIR / "sbc_results.csv")

# Prior specifications (from metadata)
PRIOR_ALPHA_MEAN = 1.75
PRIOR_ALPHA_SD = 0.5
PRIOR_BETA_MEAN = 0.27
PRIOR_BETA_SD = 0.15
PRIOR_SIGMA_SCALE = 0.2

print("=" * 80)
print("ANALYZING SIMULATION-BASED CALIBRATION RESULTS")
print("=" * 80)

# Use all simulations for coverage analysis (MCMC convergence is less critical for SBC)
# The key is whether credible intervals contain the true values
print(f"\nTotal simulations: {len(df_results) // 3}")
print(f"Analyzing all simulations for coverage metrics...")

# ============================================================================
# CALCULATE RECOVERY METRICS
# ============================================================================

print("\n" + "=" * 80)
print("RECOVERY METRICS")
print("=" * 80)

metrics = {}

for param in ['alpha', 'beta', 'sigma']:
    df_param = df_results[df_results['parameter'] == param].copy()

    # Calculate bias and RMSE
    df_param['bias'] = df_param['posterior_mean'] - df_param['true_value']
    df_param['squared_error'] = df_param['bias'] ** 2

    mean_bias = df_param['bias'].mean()
    rmse = np.sqrt(df_param['squared_error'].mean())
    coverage = df_param['in_ci'].mean()

    # Calculate empirical SD of true values and compare to posterior SD
    empirical_sd = df_param['true_value'].std()
    mean_posterior_sd = df_param['posterior_sd'].mean()
    sd_ratio = mean_posterior_sd / empirical_sd if empirical_sd > 0 else np.nan

    # Get prior SD for comparison
    if param == 'alpha':
        prior_sd = PRIOR_ALPHA_SD
    elif param == 'beta':
        prior_sd = PRIOR_BETA_SD
    else:  # sigma
        prior_sd = 0.85 * PRIOR_SIGMA_SCALE

    # Check for bias relative to prior SD
    bias_threshold = 0.1 * prior_sd
    bias_flag = abs(mean_bias) > bias_threshold

    # Convergence stats
    n_converged = df_param['converged'].sum()
    mean_rhat = df_param['rhat'].mean()
    mean_ess = df_param['ess_bulk'].mean()

    metrics[param] = {
        'n_simulations': len(df_param),
        'n_converged': int(n_converged),
        'mean_rhat': mean_rhat,
        'mean_ess': mean_ess,
        'coverage': coverage,
        'mean_bias': mean_bias,
        'rmse': rmse,
        'mean_posterior_sd': mean_posterior_sd,
        'empirical_sd': empirical_sd,
        'sd_ratio': sd_ratio,
        'prior_sd': prior_sd,
        'bias_threshold': bias_threshold,
        'bias_flag': bias_flag
    }

    print(f"\n{param.upper()} (n={len(df_param)})")
    print(f"  Convergence:           {n_converged}/{len(df_param)} (Rhat<1.05, ESS>400)")
    print(f"  Mean Rhat:             {mean_rhat:.3f}")
    print(f"  Mean ESS:              {mean_ess:.0f}")
    print(f"  Coverage (95% CI):     {coverage:.3f} ({coverage*100:.1f}%)")
    print(f"  Mean Bias:             {mean_bias:+.4f} (threshold: ±{bias_threshold:.4f})")
    print(f"  RMSE:                  {rmse:.4f}")
    print(f"  Mean Posterior SD:     {mean_posterior_sd:.4f}")
    print(f"  Empirical SD:          {empirical_sd:.4f}")
    print(f"  SD Ratio (post/emp):   {sd_ratio:.3f}")

    if bias_flag:
        print(f"  ⚠ WARNING: Bias exceeds threshold!")
    if coverage < 0.85 or coverage > 0.99:
        print(f"  ⚠ WARNING: Coverage outside acceptable range [0.85, 0.99]!")

# Save metrics to JSON
metrics_dict = {k: {kk: float(vv) if not isinstance(vv, (int, bool)) else vv
                   for kk, vv in v.items()}
               for k, v in metrics.items()}
with open(CODE_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")

param_labels = {'alpha': r'$\alpha$ (Intercept)',
                'beta': r'$\beta$ (Slope)',
                'sigma': r'$\sigma$ (Noise)'}

# ---------------------------------------------------------------------------
# Plot 1: Parameter Recovery (Posterior Mean vs True Value)
# ---------------------------------------------------------------------------
print("\n1. Parameter recovery scatter plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, param in enumerate(['alpha', 'beta', 'sigma']):
    ax = axes[idx]
    df_param = df_results[df_results['parameter'] == param]

    # Scatter plot
    ax.scatter(df_param['true_value'], df_param['posterior_mean'],
              alpha=0.5, s=50, edgecolors='black', linewidths=0.5)

    # Identity line (perfect recovery)
    true_range = [df_param['true_value'].min(), df_param['true_value'].max()]
    ax.plot(true_range, true_range, 'r--', linewidth=2, label='Perfect recovery')

    # Add linear fit to assess systematic bias
    z = np.polyfit(df_param['true_value'], df_param['posterior_mean'], 1)
    p = np.poly1d(z)
    ax.plot(true_range, p(true_range), 'b-', linewidth=1.5, alpha=0.7,
           label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.set_xlabel(f'True {param_labels[param]}', fontsize=11)
    ax.set_ylabel(f'Posterior Mean {param_labels[param]}', fontsize=11)
    ax.set_title(f'{param_labels[param]}\n(n={len(df_param)})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Add RMSE to plot
    rmse = metrics[param]['rmse']
    ax.text(0.05, 0.95, f'RMSE={rmse:.4f}', transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Parameter Recovery: Posterior Mean vs True Value',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_recovery.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'parameter_recovery.png'}")

# ---------------------------------------------------------------------------
# Plot 2: Coverage Calibration (Z-scores and Histograms)
# ---------------------------------------------------------------------------
print("\n2. Coverage calibration diagnostics...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for idx, param in enumerate(['alpha', 'beta', 'sigma']):
    df_param = df_results[df_results['parameter'] == param].copy()

    # Calculate z-scores
    df_param['z_score'] = (df_param['true_value'] - df_param['posterior_mean']) / df_param['posterior_sd']

    # Top row: Histogram of z-scores (should be N(0,1))
    ax_hist = axes[0, idx]
    ax_hist.hist(df_param['z_score'], bins=20, density=True, alpha=0.7,
                edgecolor='black', label='Observed')

    # Overlay N(0,1)
    z_range = np.linspace(-3, 3, 100)
    ax_hist.plot(z_range, stats.norm.pdf(z_range), 'r-', linewidth=2,
                label='N(0,1)')

    ax_hist.set_xlabel('Z-score', fontsize=11)
    ax_hist.set_ylabel('Density', fontsize=11)
    ax_hist.set_title(f'{param_labels[param]}: Z-scores', fontsize=11, fontweight='bold')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.set_xlim(-3.5, 3.5)

    # KS test against N(0,1)
    ks_stat, ks_pval = stats.kstest(df_param['z_score'], 'norm')
    ax_hist.text(0.05, 0.95, f'KS p={ks_pval:.3f}', transform=ax_hist.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Bottom row: Rank histogram (should be uniform)
    ax_rank = axes[1, idx]

    # Calculate ranks
    df_param['rank'] = stats.norm.cdf(df_param['z_score'])

    ax_rank.hist(df_param['rank'], bins=20, density=True, alpha=0.7,
                edgecolor='black', label='Observed')
    ax_rank.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Uniform')

    ax_rank.set_xlabel('Rank (normalized)', fontsize=11)
    ax_rank.set_ylabel('Density', fontsize=11)
    ax_rank.set_title(f'{param_labels[param]}: Rank Distribution', fontsize=11, fontweight='bold')
    ax_rank.legend(fontsize=9)
    ax_rank.grid(True, alpha=0.3)
    ax_rank.set_xlim(0, 1)

    # KS test against Uniform(0,1)
    ks_stat_unif, ks_pval_unif = stats.kstest(df_param['rank'], 'uniform')
    ax_rank.text(0.05, 0.95, f'KS p={ks_pval_unif:.3f}', transform=ax_rank.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.suptitle('Coverage Calibration: Z-scores and Ranks',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'coverage_calibration.png'}")

# ---------------------------------------------------------------------------
# Plot 3: Credible Interval Coverage Visualization
# ---------------------------------------------------------------------------
print("\n3. Credible interval coverage plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, param in enumerate(['alpha', 'beta', 'sigma']):
    ax = axes[idx]
    df_param = df_results[df_results['parameter'] == param].copy()

    # Sort by true value for cleaner visualization
    df_param = df_param.sort_values('true_value')
    df_show = df_param.head(min(50, len(df_param)))

    # Plot credible intervals
    for i, (idx_val, row) in enumerate(df_show.iterrows()):
        color = 'green' if row['in_ci'] else 'red'
        alpha_val = 0.7 if row['in_ci'] else 0.9
        ax.plot([i, i], [row['ci_lower'], row['ci_upper']],
               color=color, alpha=alpha_val, linewidth=1.5)
        ax.scatter(i, row['posterior_mean'], color=color, s=20, alpha=alpha_val, zorder=3)
        ax.scatter(i, row['true_value'], color='black', marker='x', s=40,
                  linewidths=2, zorder=4)

    ax.set_xlabel('Simulation Index (sorted)', fontsize=11)
    ax.set_ylabel(f'{param_labels[param]}', fontsize=11)
    ax.set_title(f'{param_labels[param]}: 95% Credible Intervals\n'
                f'Coverage: {metrics[param]["coverage"]:.1%}',
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label='Contains true value'),
        Line2D([0], [0], color='red', linewidth=2, label='Misses true value'),
        Line2D([0], [0], marker='x', color='black', linestyle='None',
              markersize=8, markeredgewidth=2, label='True value')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='best')

plt.suptitle('95% Credible Interval Coverage (first 50 simulations)',
            fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'credible_interval_coverage.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'credible_interval_coverage.png'}")

# ---------------------------------------------------------------------------
# Plot 4: Bias and RMSE Summary
# ---------------------------------------------------------------------------
print("\n4. Bias and RMSE summary bar plot...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

params = ['alpha', 'beta', 'sigma']
param_names = [r'$\alpha$', r'$\beta$', r'$\sigma$']

# Bias plot
ax_bias = axes[0]
biases = [metrics[p]['mean_bias'] for p in params]
thresholds = [metrics[p]['bias_threshold'] for p in params]
colors = ['red' if metrics[p]['bias_flag'] else 'steelblue' for p in params]

bars = ax_bias.bar(param_names, biases, color=colors, alpha=0.7, edgecolor='black')
ax_bias.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add threshold lines
for i, (p, thresh) in enumerate(zip(params, thresholds)):
    ax_bias.plot([i-0.4, i+0.4], [thresh, thresh], 'r--', linewidth=2, alpha=0.5)
    ax_bias.plot([i-0.4, i+0.4], [-thresh, -thresh], 'r--', linewidth=2, alpha=0.5)

ax_bias.set_ylabel('Mean Bias (Posterior Mean - True)', fontsize=11)
ax_bias.set_title('Parameter Recovery Bias', fontsize=12, fontweight='bold')
ax_bias.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, bias in zip(bars, biases):
    height = bar.get_height()
    ax_bias.text(bar.get_x() + bar.get_width()/2., height,
                f'{bias:+.4f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

# RMSE plot
ax_rmse = axes[1]
rmses = [metrics[p]['rmse'] for p in params]
bars_rmse = ax_rmse.bar(param_names, rmses, color='steelblue', alpha=0.7, edgecolor='black')

ax_rmse.set_ylabel('Root Mean Squared Error', fontsize=11)
ax_rmse.set_title('Parameter Recovery Accuracy', fontsize=12, fontweight='bold')
ax_rmse.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, rmse in zip(bars_rmse, rmses):
    height = bar.get_height()
    ax_rmse.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.suptitle('Parameter Recovery: Bias and Accuracy',
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'bias_rmse_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'bias_rmse_summary.png'}")

# ---------------------------------------------------------------------------
# Plot 5: Coverage by Parameter
# ---------------------------------------------------------------------------
print("\n5. Coverage summary bar plot...")

fig, ax = plt.subplots(figsize=(8, 6))

coverages = [metrics[p]['coverage'] for p in params]
colors_cov = ['red' if (c < 0.85 or c > 0.99) else 'green' for c in coverages]

bars = ax.bar(param_names, coverages, color=colors_cov, alpha=0.7, edgecolor='black')

# Add ideal coverage line and acceptable range
ax.axhline(y=0.95, color='blue', linestyle='-', linewidth=2, label='Nominal (95%)')
ax.axhspan(0.85, 0.99, alpha=0.2, color='green', label='Acceptable range')

ax.set_ylabel('Coverage Probability', fontsize=12)
ax.set_xlabel('Parameter', fontsize=12)
ax.set_title('95% Credible Interval Coverage', fontsize=13, fontweight='bold')
ax.set_ylim(0.75, 1.0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, cov in zip(bars, coverages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{cov:.3f}\n({cov*100:.1f}%)', ha='center', va='bottom',
           fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'coverage_summary.png'}")

# ---------------------------------------------------------------------------
# Plot 6: Uncertainty Calibration
# ---------------------------------------------------------------------------
print("\n6. Uncertainty calibration plot...")

fig, ax = plt.subplots(figsize=(8, 6))

for param, marker, label in [('alpha', 'o', r'$\alpha$'),
                              ('beta', 's', r'$\beta$'),
                              ('sigma', '^', r'$\sigma$')]:
    df_param = df_results[df_results['parameter'] == param]

    ax.scatter(df_param['posterior_sd'],
              np.abs(df_param['posterior_mean'] - df_param['true_value']),
              alpha=0.5, s=50, marker=marker, label=label, edgecolors='black', linewidths=0.5)

ax.set_xlabel('Posterior Standard Deviation', fontsize=12)
ax.set_ylabel('|Posterior Mean - True Value|', fontsize=12)
ax.set_title('Uncertainty Calibration: Posterior SD vs Actual Error',
            fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add reference lines
xlim = ax.get_xlim()
x_ref = np.linspace(xlim[0], xlim[1], 100)
ax.plot(x_ref, x_ref, 'r--', linewidth=2, alpha=0.5, label='SD = |Error|')
ax.plot(x_ref, 2*x_ref, 'r:', linewidth=2, alpha=0.5, label='SD = 2×|Error|')

ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'uncertainty_calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'uncertainty_calibration.png'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nMetrics saved to: {CODE_DIR / 'metrics.json'}")
print(f"All plots saved to: {PLOTS_DIR}")
