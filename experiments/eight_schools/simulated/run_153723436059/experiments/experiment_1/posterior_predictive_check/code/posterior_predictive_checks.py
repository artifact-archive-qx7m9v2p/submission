"""
Posterior Predictive Checks for Experiment 1: Standard Hierarchical Model

This script performs comprehensive posterior predictive checks to assess whether
the fitted hierarchical model can replicate key features of the Eight Schools data.

Author: Model Validation Specialist
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(456)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
PLOT_DIR = '/workspace/experiments/experiment_1/posterior_predictive_check/plots'

print("="*80)
print("POSTERIOR PREDICTIVE CHECKS: Experiment 1")
print("="*80)

# =============================================================================
# 1. LOAD DATA AND POSTERIOR SAMPLES
# =============================================================================

print("\n[1] Loading Data and Posterior Samples...")

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
y_obs = data['effect'].values
sigma_obs = data['sigma'].values
n_schools = len(y_obs)
school_names = [f"School {i+1}" for i in range(n_schools)]

print(f"   Observed effects (y): {y_obs}")
print(f"   Standard errors (sigma): {sigma_obs}")

# Load posterior samples
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

# Extract posterior predictive samples
# Note: The model saved y_rep in posterior group and y in posterior_predictive
# We'll use the posterior_predictive.y which is the proper posterior predictive
y_rep = idata.posterior_predictive.y.values
n_chains, n_draws, n_schools_check = y_rep.shape
n_total_draws = n_chains * n_draws

# Reshape to (total_draws, schools) for easier manipulation
y_rep_flat = y_rep.reshape(-1, n_schools)

print(f"   Posterior predictive samples shape: {y_rep.shape}")
print(f"   Total posterior predictive draws: {n_total_draws}")
print(f"   Schools: {n_schools}")

# Extract posterior parameters for reference
mu_samples = idata.posterior.mu.values.flatten()
tau_samples = idata.posterior.tau.values.flatten()
theta_samples = idata.posterior.theta.values.reshape(-1, n_schools)

print(f"   mu posterior mean: {mu_samples.mean():.2f} ± {mu_samples.std():.2f}")
print(f"   tau posterior mean: {tau_samples.mean():.2f} ± {tau_samples.std():.2f}")

# =============================================================================
# 2. VISUAL POSTERIOR PREDICTIVE CHECKS
# =============================================================================

print("\n[2] Creating Visual Posterior Predictive Checks...")

# -----------------------------------------------------------------------------
# PLOT 1: Spaghetti Plot (100 random replications + observed)
# -----------------------------------------------------------------------------

print("   Creating spaghetti plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Select 100 random posterior predictive datasets
n_rep_plot = 100
indices = np.random.choice(n_total_draws, size=n_rep_plot, replace=False)

# Plot replications in light gray
for idx in indices:
    ax.plot(range(1, n_schools + 1), y_rep_flat[idx, :],
            color='gray', alpha=0.1, linewidth=1, zorder=1)

# Plot observed data in bold red
ax.plot(range(1, n_schools + 1), y_obs, 'o-', color='red',
        linewidth=3, markersize=10, label='Observed Data', zorder=3)

# Add error bars for measurement uncertainty
ax.errorbar(range(1, n_schools + 1), y_obs, yerr=sigma_obs,
            fmt='none', color='red', alpha=0.5, linewidth=2, zorder=2)

ax.set_xlabel('School', fontsize=12, fontweight='bold')
ax.set_ylabel('Effect Size', fontsize=12, fontweight='bold')
ax.set_title('Posterior Predictive Check: Spaghetti Plot\n(100 Replications + Observed Data)',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(1, n_schools + 1))
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/ppc_spaghetti.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/ppc_spaghetti.png")

# -----------------------------------------------------------------------------
# PLOT 2: Posterior Predictive Distributions by School (8-panel)
# -----------------------------------------------------------------------------

print("   Creating school-specific PPC distributions...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(n_schools):
    ax = axes[i]

    # Histogram of posterior predictive samples for this school
    ax.hist(y_rep_flat[:, i], bins=40, density=True, alpha=0.6,
            color='skyblue', edgecolor='black', linewidth=0.5, label='Posterior Predictive')

    # Mark observed value with vertical line
    ax.axvline(y_obs[i], color='red', linewidth=3,
               label=f'Observed: {y_obs[i]:.1f}', zorder=5)

    # Add 50% and 90% credible intervals
    q05, q25, q75, q95 = np.percentile(y_rep_flat[:, i], [5, 25, 75, 95])

    ax.axvspan(q05, q95, alpha=0.15, color='blue', label='90% CI')
    ax.axvspan(q25, q75, alpha=0.25, color='blue', label='50% CI')

    # Calculate p-value (proportion of replications more extreme than observed)
    p_value = np.mean(y_rep_flat[:, i] >= y_obs[i])

    ax.set_title(f'{school_names[i]}\n(p={p_value:.3f}, sigma={sigma_obs[i]:.0f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Effect Size', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Check: School-Specific Distributions',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/ppc_by_school.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/ppc_by_school.png")

# -----------------------------------------------------------------------------
# PLOT 3: PPC Density Overlay (All Data Pooled)
# -----------------------------------------------------------------------------

print("   Creating density overlay plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Pool all posterior predictive data
y_rep_pooled = y_rep_flat.flatten()

# Kernel density estimate of posterior predictive
from scipy.stats import gaussian_kde
kde_rep = gaussian_kde(y_rep_pooled)
x_grid = np.linspace(y_rep_pooled.min(), y_rep_pooled.max(), 1000)
density_rep = kde_rep(x_grid)

ax.fill_between(x_grid, density_rep, alpha=0.4, color='skyblue',
                label=f'Posterior Predictive (n={len(y_rep_pooled)})')
ax.plot(x_grid, density_rep, color='blue', linewidth=2)

# Plot observed data as rug plot
ax.scatter(y_obs, np.zeros(n_schools), color='red', s=200,
           marker='|', linewidth=4, label='Observed Data (n=8)', zorder=5)

# Add vertical lines for observed values
for i, y in enumerate(y_obs):
    ax.axvline(y, color='red', alpha=0.3, linewidth=1, linestyle='--')

ax.set_xlabel('Effect Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('Posterior Predictive Check: Overall Distribution\n(Observed vs Predicted)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axvline(y_obs.mean(), color='red', linewidth=2, linestyle='-',
           label=f'Obs Mean: {y_obs.mean():.1f}')
ax.axvline(y_rep_pooled.mean(), color='blue', linewidth=2, linestyle='-',
           label=f'Pred Mean: {y_rep_pooled.mean():.1f}')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/ppc_density_overlay.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/ppc_density_overlay.png")

# -----------------------------------------------------------------------------
# PLOT 4: Q-Q Plot (Quantile-Quantile)
# -----------------------------------------------------------------------------

print("   Creating Q-Q plot...")

fig, ax = plt.subplots(figsize=(8, 8))

# Compute quantiles
quantiles = np.linspace(0.01, 0.99, 100)
q_obs = np.percentile(y_obs, quantiles * 100)
q_rep = np.percentile(y_rep_pooled, quantiles * 100)

# Q-Q plot
ax.scatter(q_rep, q_obs, alpha=0.7, s=50, color='blue', edgecolor='black', linewidth=0.5)

# Add diagonal reference line
lim_min = min(q_rep.min(), q_obs.min())
lim_max = max(q_rep.max(), q_obs.max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2,
        label='Perfect Calibration')

# Add 95% confidence band (approximate)
# For well-calibrated model, points should fall within this band
n_obs = len(y_obs)
se = (q_rep.std() / np.sqrt(n_obs)) * 1.96
ax.fill_between([lim_min, lim_max],
                [lim_min - se, lim_max - se],
                [lim_min + se, lim_max + se],
                alpha=0.2, color='gray', label='95% Confidence Band')

ax.set_xlabel('Posterior Predictive Quantiles', fontsize=12, fontweight='bold')
ax.set_ylabel('Observed Data Quantiles', fontsize=12, fontweight='bold')
ax.set_title('Posterior Predictive Check: Q-Q Plot\n(Model Calibration)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/ppc_qq_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/ppc_qq_plot.png")

# -----------------------------------------------------------------------------
# ArviZ Built-in PPC Plot
# -----------------------------------------------------------------------------

print("   Creating ArviZ PPC plot...")

fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title('ArviZ Posterior Predictive Check\n(100 Replications)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/ppc_arviz.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/ppc_arviz.png")

# =============================================================================
# 3. QUANTITATIVE TEST STATISTICS
# =============================================================================

print("\n[3] Computing Test Statistics and Bayesian p-values...")

def compute_test_stat(y, stat_func):
    """Apply test statistic function to data."""
    return stat_func(y)

def bayesian_p_value(y_obs, y_rep, stat_func):
    """
    Compute Bayesian p-value: P(T(y_rep) >= T(y_obs))

    p-value near 0.5 indicates good fit
    p-value < 0.05 or > 0.95 indicates potential misfit
    """
    t_obs = stat_func(y_obs)
    t_rep = np.array([stat_func(y_rep[i, :]) for i in range(len(y_rep))])
    p_value = np.mean(t_rep >= t_obs)
    return t_obs, t_rep, p_value

# Define test statistics
test_stats = {
    # Location
    'Mean': lambda y: np.mean(y),
    'Median': lambda y: np.median(y),

    # Spread
    'SD': lambda y: np.std(y, ddof=1),
    'Range': lambda y: np.max(y) - np.min(y),
    'IQR': lambda y: np.percentile(y, 75) - np.percentile(y, 25),

    # Shape
    'Skewness': lambda y: stats.skew(y),
    'Kurtosis': lambda y: stats.kurtosis(y),

    # Extremes
    'Min': lambda y: np.min(y),
    'Max': lambda y: np.max(y),
    'Q5': lambda y: np.percentile(y, 5),
    'Q95': lambda y: np.percentile(y, 95),
}

# Compute test statistics
results = []
for stat_name, stat_func in test_stats.items():
    t_obs, t_rep, p_value = bayesian_p_value(y_obs, y_rep_flat, stat_func)

    results.append({
        'Statistic': stat_name,
        'Observed': t_obs,
        'Predicted Mean': np.mean(t_rep),
        'Predicted SD': np.std(t_rep),
        'Bayesian p-value': p_value,
        'Status': 'PASS' if 0.05 <= p_value <= 0.95 else 'FLAG'
    })

    print(f"   {stat_name:12s}: T_obs={t_obs:7.2f}, T_rep={np.mean(t_rep):7.2f} ± {np.std(t_rep):5.2f}, p={p_value:.3f} [{results[-1]['Status']}]")

# Create DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/test_statistics.csv',
                  index=False)
print(f"\n   Saved: test_statistics.csv")

# -----------------------------------------------------------------------------
# PLOT: Test Statistics with Bayesian p-values
# -----------------------------------------------------------------------------

print("   Creating test statistics visualization...")

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, (stat_name, stat_func) in enumerate(test_stats.items()):
    ax = axes[i]

    t_obs, t_rep, p_value = bayesian_p_value(y_obs, y_rep_flat, stat_func)

    # Histogram of T(y_rep)
    ax.hist(t_rep, bins=50, density=True, alpha=0.6,
            color='skyblue', edgecolor='black', linewidth=0.5)

    # Mark T(y_obs) with vertical line
    ax.axvline(t_obs, color='red', linewidth=3,
               label=f'Observed: {t_obs:.2f}')

    # Add p-value annotation
    status_color = 'green' if 0.05 <= p_value <= 0.95 else 'red'
    ax.text(0.05, 0.95, f'p = {p_value:.3f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor=status_color, alpha=0.3))

    ax.set_title(f'{stat_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Statistic Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

# Remove extra subplots
for i in range(len(test_stats), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('Test Statistics: Observed vs Posterior Predictive Distribution\n(Green = PASS, Red = FLAG)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/test_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/test_statistics.png")

# =============================================================================
# 4. SCHOOL-SPECIFIC P-VALUES
# =============================================================================

print("\n[4] Computing School-Specific p-values...")

school_pvalues = []
for i in range(n_schools):
    # P(y_rep[i] >= y_obs[i])
    p_value = np.mean(y_rep_flat[:, i] >= y_obs[i])

    school_pvalues.append({
        'School': school_names[i],
        'Observed': y_obs[i],
        'Predicted Mean': np.mean(y_rep_flat[:, i]),
        'Predicted SD': np.std(y_rep_flat[:, i]),
        'Measurement SE': sigma_obs[i],
        'p-value': p_value,
        'Status': 'OUTLIER' if (p_value < 0.05 or p_value > 0.95) else 'OK'
    })

    print(f"   {school_names[i]:9s}: y_obs={y_obs[i]:6.1f}, y_rep={np.mean(y_rep_flat[:, i]):6.1f} ± {np.std(y_rep_flat[:, i]):5.1f}, p={p_value:.3f} [{school_pvalues[-1]['Status']}]")

school_pvalues_df = pd.DataFrame(school_pvalues)
school_pvalues_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/school_pvalues.csv',
                          index=False)
print(f"\n   Saved: school_pvalues.csv")

# =============================================================================
# 5. COVERAGE ANALYSIS
# =============================================================================

print("\n[5] Computing Coverage Analysis...")

coverage_levels = [0.50, 0.80, 0.90, 0.95]
coverage_results = []

for level in coverage_levels:
    alpha = 1 - level
    lower_q = (alpha / 2) * 100
    upper_q = (1 - alpha / 2) * 100

    # Compute credible intervals for each school
    lower_bounds = np.percentile(y_rep_flat, lower_q, axis=0)
    upper_bounds = np.percentile(y_rep_flat, upper_q, axis=0)

    # Check coverage
    covered = (y_obs >= lower_bounds) & (y_obs <= upper_bounds)
    actual_coverage = np.mean(covered)

    coverage_results.append({
        'Nominal Coverage': f'{level*100:.0f}%',
        'Actual Coverage': f'{actual_coverage*100:.1f}%',
        'Schools Covered': f'{np.sum(covered)}/{n_schools}',
        'Difference': f'{(actual_coverage - level)*100:+.1f}%',
        'Status': 'PASS' if abs(actual_coverage - level) < 0.15 else 'FLAG'
    })

    print(f"   {level*100:3.0f}% Interval: {actual_coverage*100:5.1f}% actual ({np.sum(covered)}/{n_schools} schools) [{coverage_results[-1]['Status']}]")

coverage_df = pd.DataFrame(coverage_results)
coverage_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/coverage_analysis.csv',
                    index=False)
print(f"\n   Saved: coverage_analysis.csv")

# -----------------------------------------------------------------------------
# PLOT: Coverage Analysis
# -----------------------------------------------------------------------------

print("   Creating coverage analysis plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# For each school, plot observed value and credible intervals
for i in range(n_schools):
    # Compute intervals at multiple levels
    intervals = {}
    for level in [0.50, 0.90, 0.95]:
        alpha = 1 - level
        lower_q = (alpha / 2) * 100
        upper_q = (1 - alpha / 2) * 100
        intervals[level] = (
            np.percentile(y_rep_flat[:, i], lower_q),
            np.percentile(y_rep_flat[:, i], upper_q)
        )

    # Plot intervals
    y_pos = i

    # 95% interval (lightest)
    ax.plot([intervals[0.95][0], intervals[0.95][1]], [y_pos, y_pos],
            linewidth=2, color='lightblue', alpha=0.5)

    # 90% interval (medium)
    ax.plot([intervals[0.90][0], intervals[0.90][1]], [y_pos, y_pos],
            linewidth=4, color='skyblue', alpha=0.7)

    # 50% interval (darkest)
    ax.plot([intervals[0.50][0], intervals[0.50][1]], [y_pos, y_pos],
            linewidth=6, color='steelblue', alpha=0.9)

    # Posterior mean
    mean_pred = np.mean(y_rep_flat[:, i])
    ax.plot(mean_pred, y_pos, 'o', color='blue', markersize=8)

    # Observed value
    ax.plot(y_obs[i], y_pos, 'D', color='red', markersize=10,
            markeredgecolor='black', markeredgewidth=1)

    # Add measurement error bar
    ax.errorbar(y_obs[i], y_pos, xerr=sigma_obs[i],
                fmt='none', color='red', alpha=0.4, linewidth=2)

# Formatting
ax.set_yticks(range(n_schools))
ax.set_yticklabels(school_names)
ax.set_xlabel('Effect Size', fontsize=12, fontweight='bold')
ax.set_ylabel('School', fontsize=12, fontweight='bold')
ax.set_title('Coverage Analysis: Posterior Predictive Intervals vs Observed\n(50%, 90%, 95% Credible Intervals)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='steelblue', linewidth=6, label='50% CI'),
    Line2D([0], [0], color='skyblue', linewidth=4, label='90% CI'),
    Line2D([0], [0], color='lightblue', linewidth=2, label='95% CI'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=8, label='Predicted Mean'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
           markersize=10, markeredgecolor='black', label='Observed'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/coverage_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/coverage_analysis.png")

# =============================================================================
# 6. COMPREHENSIVE SUMMARY PLOT
# =============================================================================

print("\n[6] Creating Comprehensive Summary Plot...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Overall PPC (top left)
ax1 = fig.add_subplot(gs[0, 0])
az.plot_ppc(idata, num_pp_samples=50, ax=ax1)
ax1.set_title('(A) Overall PPC', fontsize=12, fontweight='bold')

# Panel 2: Test Statistic - Mean (top middle)
ax2 = fig.add_subplot(gs[0, 1])
stat_func = test_stats['Mean']
t_obs, t_rep, p_value = bayesian_p_value(y_obs, y_rep_flat, stat_func)
ax2.hist(t_rep, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax2.axvline(t_obs, color='red', linewidth=3, label=f'Observed: {t_obs:.1f}')
ax2.set_title(f'(B) Mean (p={p_value:.3f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Mean Effect')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Test Statistic - SD (top right)
ax3 = fig.add_subplot(gs[0, 2])
stat_func = test_stats['SD']
t_obs, t_rep, p_value = bayesian_p_value(y_obs, y_rep_flat, stat_func)
ax3.hist(t_rep, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax3.axvline(t_obs, color='red', linewidth=3, label=f'Observed: {t_obs:.1f}')
ax3.set_title(f'(C) SD (p={p_value:.3f})', fontsize=12, fontweight='bold')
ax3.set_xlabel('Standard Deviation')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Test Statistic - Max (middle left)
ax4 = fig.add_subplot(gs[1, 0])
stat_func = test_stats['Max']
t_obs, t_rep, p_value = bayesian_p_value(y_obs, y_rep_flat, stat_func)
ax4.hist(t_rep, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax4.axvline(t_obs, color='red', linewidth=3, label=f'Observed: {t_obs:.1f}')
ax4.set_title(f'(D) Maximum (p={p_value:.3f})', fontsize=12, fontweight='bold')
ax4.set_xlabel('Maximum Effect')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Panel 5: Test Statistic - Min (middle middle)
ax5 = fig.add_subplot(gs[1, 1])
stat_func = test_stats['Min']
t_obs, t_rep, p_value = bayesian_p_value(y_obs, y_rep_flat, stat_func)
ax5.hist(t_rep, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax5.axvline(t_obs, color='red', linewidth=3, label=f'Observed: {t_obs:.1f}')
ax5.set_title(f'(E) Minimum (p={p_value:.3f})', fontsize=12, fontweight='bold')
ax5.set_xlabel('Minimum Effect')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Panel 6: Test Statistic - Range (middle right)
ax6 = fig.add_subplot(gs[1, 2])
stat_func = test_stats['Range']
t_obs, t_rep, p_value = bayesian_p_value(y_obs, y_rep_flat, stat_func)
ax6.hist(t_rep, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax6.axvline(t_obs, color='red', linewidth=3, label=f'Observed: {t_obs:.1f}')
ax6.set_title(f'(F) Range (p={p_value:.3f})', fontsize=12, fontweight='bold')
ax6.set_xlabel('Range (Max - Min)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Panel 7: School-specific p-values (bottom left)
ax7 = fig.add_subplot(gs[2, 0])
p_vals = [np.mean(y_rep_flat[:, i] >= y_obs[i]) for i in range(n_schools)]
colors = ['red' if (p < 0.05 or p > 0.95) else 'green' for p in p_vals]
ax7.bar(range(n_schools), p_vals, color=colors, alpha=0.6, edgecolor='black')
ax7.axhline(0.5, color='blue', linestyle='--', linewidth=2, label='Expected (p=0.5)')
ax7.axhline(0.05, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax7.axhline(0.95, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax7.set_xticks(range(n_schools))
ax7.set_xticklabels([f'S{i+1}' for i in range(n_schools)])
ax7.set_title('(G) School-Specific p-values', fontsize=12, fontweight='bold')
ax7.set_xlabel('School')
ax7.set_ylabel('p-value')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Panel 8: Coverage rates (bottom middle)
ax8 = fig.add_subplot(gs[2, 1])
nominal = [0.50, 0.80, 0.90, 0.95]
actual = []
for level in nominal:
    alpha = 1 - level
    lower_q = (alpha / 2) * 100
    upper_q = (1 - alpha / 2) * 100
    lower_bounds = np.percentile(y_rep_flat, lower_q, axis=0)
    upper_bounds = np.percentile(y_rep_flat, upper_q, axis=0)
    covered = (y_obs >= lower_bounds) & (y_obs <= upper_bounds)
    actual.append(np.mean(covered))

ax8.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
ax8.plot(nominal, actual, 'o-', color='blue', markersize=10, linewidth=2, label='Actual Coverage')
ax8.set_xlabel('Nominal Coverage', fontsize=11)
ax8.set_ylabel('Actual Coverage', fontsize=11)
ax8.set_title('(H) Coverage Calibration', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.set_aspect('equal')

# Panel 9: Q-Q plot (bottom right)
ax9 = fig.add_subplot(gs[2, 2])
quantiles = np.linspace(0.01, 0.99, 100)
q_obs = np.percentile(y_obs, quantiles * 100)
q_rep = np.percentile(y_rep_pooled, quantiles * 100)
ax9.scatter(q_rep, q_obs, alpha=0.7, s=30, color='blue', edgecolor='black', linewidth=0.5)
lim_min = min(q_rep.min(), q_obs.min())
lim_max = max(q_rep.max(), q_obs.max())
ax9.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2)
ax9.set_xlabel('Predicted Quantiles', fontsize=11)
ax9.set_ylabel('Observed Quantiles', fontsize=11)
ax9.set_title('(I) Q-Q Plot', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.set_aspect('equal')

plt.suptitle('Posterior Predictive Check: Comprehensive Summary',
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig(f'{PLOT_DIR}/ppc_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOT_DIR}/ppc_summary.png")

# =============================================================================
# 7. DISCREPANCY ANALYSIS
# =============================================================================

print("\n[7] Discrepancy Analysis...")

print("\n   Schools with extreme p-values (potential outliers):")
outlier_found = False
for i in range(n_schools):
    p_val = np.mean(y_rep_flat[:, i] >= y_obs[i])
    if p_val < 0.05 or p_val > 0.95:
        print(f"   - {school_names[i]}: p={p_val:.3f} (Observed={y_obs[i]:.1f}, Predicted={np.mean(y_rep_flat[:, i]):.1f})")
        outlier_found = True

if not outlier_found:
    print("   - None (all schools well-calibrated)")

print("\n   Test statistics with extreme p-values:")
flag_found = False
for _, row in results_df.iterrows():
    if row['Status'] == 'FLAG':
        print(f"   - {row['Statistic']}: p={row['Bayesian p-value']:.3f}")
        flag_found = True

if not flag_found:
    print("   - None (all test statistics within expected range)")

# =============================================================================
# 8. OVERALL ASSESSMENT
# =============================================================================

print("\n[8] Overall Assessment...")

# Count passes and flags
n_test_pass = np.sum(results_df['Status'] == 'PASS')
n_test_flag = np.sum(results_df['Status'] == 'FLAG')
n_school_ok = np.sum(school_pvalues_df['Status'] == 'OK')
n_school_outlier = np.sum(school_pvalues_df['Status'] == 'OUTLIER')
n_coverage_pass = np.sum(coverage_df['Status'] == 'PASS')
n_coverage_flag = np.sum(coverage_df['Status'] == 'FLAG')

# Determine overall status
if n_test_flag == 0 and n_school_outlier <= 1 and n_coverage_flag == 0:
    overall_status = "PASS"
elif n_test_flag <= 2 and n_school_outlier <= 2 and n_coverage_flag <= 1:
    overall_status = "CONDITIONAL PASS"
else:
    overall_status = "FAIL"

print(f"\n   Test Statistics: {n_test_pass}/{len(results_df)} PASS, {n_test_flag}/{len(results_df)} FLAG")
print(f"   School-Specific: {n_school_ok}/{n_schools} OK, {n_school_outlier}/{n_schools} OUTLIER")
print(f"   Coverage: {n_coverage_pass}/{len(coverage_df)} PASS, {n_coverage_flag}/{len(coverage_df)} FLAG")
print(f"\n   OVERALL STATUS: {overall_status}")

# =============================================================================
# 9. SAVE SUMMARY
# =============================================================================

summary = {
    'Overall Status': overall_status,
    'Test Statistics': f'{n_test_pass}/{len(results_df)} PASS',
    'School-Specific': f'{n_school_ok}/{n_schools} OK',
    'Coverage': f'{n_coverage_pass}/{len(coverage_df)} PASS',
    'Date': '2025-10-29',
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/ppc_summary.csv',
                   index=False)

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("="*80)
print(f"\nOutputs saved to: /workspace/experiments/experiment_1/posterior_predictive_check/")
print(f"- Plots: {PLOT_DIR}/")
print(f"- Data: test_statistics.csv, school_pvalues.csv, coverage_analysis.csv, ppc_summary.csv")
