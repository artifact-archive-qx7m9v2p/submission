"""
Visualizations for Prior Predictive Checks

Creates diagnostic plots to assess:
1. Parameter plausibility (prior draws)
2. Prior predictive coverage (does observed data fall within simulated range?)
3. Summary statistics comparison (observed vs simulated)
4. Structural patterns (growth, autocorrelation, overdispersion)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# Load Data
# ============================================================================
print("Loading data...")

# Observed data
observed_data = pd.read_csv('/workspace/data/data.csv')
years = observed_data['year'].values
observed_counts = observed_data['C'].values

# Prior samples
prior_samples = pd.read_csv('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.csv')

# Summary statistics
summary_stats = pd.read_csv('/workspace/experiments/experiment_1/prior_predictive_check/code/summary_statistics.csv')

# Simulated datasets
simulated_datasets = np.load('/workspace/experiments/experiment_1/prior_predictive_check/code/simulated_datasets_full.npy')

print(f"Prior samples: {prior_samples.shape}")
print(f"Summary statistics: {summary_stats.shape}")
print(f"Simulated datasets: {simulated_datasets.shape}")

# ============================================================================
# Plot 1: Parameter Plausibility - Prior Distributions
# ============================================================================
print("\nCreating Plot 1: Parameter plausibility...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Prior Predictive Check: Parameter Plausibility', fontsize=16, fontweight='bold')

param_info = [
    ('beta_0', r'$\beta_0$ (Intercept)', 'Expected value at year=0 (log scale)'),
    ('beta_1', r'$\beta_1$ (Pre-break slope)', 'Growth rate before changepoint'),
    ('beta_2', r'$\beta_2$ (Post-break increase)', 'Additional growth after changepoint'),
    ('alpha', r'$\alpha$ (NB dispersion)', 'Overdispersion parameter'),
    ('rho', r'$\rho$ (AR coefficient)', 'Temporal autocorrelation'),
    ('sigma_eps', r'$\sigma_\epsilon$ (AR noise)', 'Innovation std deviation')
]

for idx, (param, title, subtitle) in enumerate(param_info):
    ax = axes[idx // 3, idx % 3]
    values = prior_samples[param].values

    # Histogram with KDE
    ax.hist(values, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')

    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(values)
    x_range = np.linspace(values.min(), values.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Add vertical lines for percentiles
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    ax.axvline(p50, color='darkred', linestyle='--', linewidth=2, label=f'Median: {p50:.2f}')
    ax.axvline(p25, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(p75, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(subtitle, fontsize=9, style='italic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {np.mean(values):.2f}\nStd: {np.std(values):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/parameter_plausibility.png',
            bbox_inches='tight')
plt.close()
print("  Saved: parameter_plausibility.png")

# ============================================================================
# Plot 2: Prior Predictive Coverage - Simulated Data Envelope
# ============================================================================
print("\nCreating Plot 2: Prior predictive coverage...")

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# Plot random sample of simulated trajectories
n_show = 50
rng = np.random.RandomState(42)
sample_indices = rng.choice(simulated_datasets.shape[0], size=n_show, replace=False)

for idx in sample_indices:
    ax.plot(years, simulated_datasets[idx, :], color='lightblue', alpha=0.3, linewidth=0.5)

# Add percentile bands
p5 = np.percentile(simulated_datasets, 5, axis=0)
p25 = np.percentile(simulated_datasets, 25, axis=0)
p50 = np.percentile(simulated_datasets, 50, axis=0)
p75 = np.percentile(simulated_datasets, 75, axis=0)
p95 = np.percentile(simulated_datasets, 95, axis=0)

ax.fill_between(years, p5, p95, color='blue', alpha=0.15, label='90% Prior Predictive Interval')
ax.fill_between(years, p25, p75, color='blue', alpha=0.25, label='50% Prior Predictive Interval')
ax.plot(years, p50, 'b-', linewidth=2, label='Prior Predictive Median')

# Plot observed data
ax.plot(years, observed_counts, 'ro-', linewidth=2.5, markersize=6,
        label='Observed Data', zorder=10)

# Add changepoint line
changepoint_year = years[16]  # 0-indexed
ax.axvline(changepoint_year, color='green', linestyle='--', linewidth=2,
           label=f'Changepoint (obs 17)', alpha=0.7)

ax.set_xlabel('Standardized Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Prior Predictive Check: Coverage of Observed Data\n(50 random prior trajectories shown)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Add text annotation
coverage_text = (
    f'Observed range: [{observed_counts.min()}, {observed_counts.max()}]\n'
    f'Prior predictive 90% range: [{p5.min():.0f}, {p95.max():.0f}]\n'
    f'Prior predictive median range: [{p50.min():.0f}, {p50.max():.0f}]'
)
ax.text(0.98, 0.02, coverage_text, transform=ax.transAxes,
        verticalalignment='bottom', horizontalalignment='right',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/prior_predictive_coverage.png',
            bbox_inches='tight')
plt.close()
print("  Saved: prior_predictive_coverage.png")

# ============================================================================
# Plot 3: Summary Statistics Comparison
# ============================================================================
print("\nCreating Plot 3: Summary statistics comparison...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Prior Predictive Check: Summary Statistics vs Observed Data',
             fontsize=16, fontweight='bold')

# Compute observed statistics
obs_min = observed_counts.min()
obs_max = observed_counts.max()
obs_mean = observed_counts.mean()
obs_median = np.median(observed_counts)
obs_variance = observed_counts.var()
obs_var_mean_ratio = obs_variance / obs_mean
obs_acf1 = np.corrcoef(observed_counts[:-1], observed_counts[1:])[0, 1]
obs_growth = observed_counts[-5:].mean() / observed_counts[:5].mean()

observed_stats = {
    'min_count': obs_min,
    'max_count': obs_max,
    'mean_count': obs_mean,
    'variance_mean_ratio': obs_var_mean_ratio,
    'acf_lag1': obs_acf1,
    'growth_factor': obs_growth
}

plot_configs = [
    ('min_count', 'Minimum Count', 'Lowest observed count', [0, 300]),
    ('max_count', 'Maximum Count', 'Highest observed count', [0, 2000]),
    ('mean_count', 'Mean Count', 'Average count across time', [0, 500]),
    ('variance_mean_ratio', 'Variance/Mean Ratio', 'Overdispersion measure', [0, 200]),
    ('acf_lag1', 'ACF(1)', 'Lag-1 autocorrelation', [-0.5, 1.0]),
    ('growth_factor', 'Growth Factor', 'Final/Initial ratio', [0, 50])
]

for idx, (stat_name, title, subtitle, xlim) in enumerate(plot_configs):
    ax = axes[idx // 3, idx % 3]

    sim_values = summary_stats[stat_name].values
    obs_value = observed_stats[stat_name]

    # Histogram of simulated values
    ax.hist(sim_values, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', label='Prior Predictive')

    # KDE
    if sim_values.std() > 0:
        kde = gaussian_kde(sim_values)
        x_range = np.linspace(max(xlim[0], sim_values.min()),
                               min(xlim[1], sim_values.max()), 200)
        ax.plot(x_range, kde(x_range), 'b-', linewidth=2)

    # Observed value
    ax.axvline(obs_value, color='red', linestyle='--', linewidth=3,
               label=f'Observed: {obs_value:.2f}')

    # Percentile of observed in prior predictive
    percentile = stats.percentileofscore(sim_values, obs_value)

    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(subtitle, fontsize=9, style='italic')
    ax.set_xlim(xlim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add percentile text
    percentile_text = f'Observed at {percentile:.1f}th percentile'
    color = 'green' if 5 <= percentile <= 95 else 'orange' if 1 <= percentile <= 99 else 'red'
    ax.text(0.98, 0.98, percentile_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=8, bbox=dict(boxstyle='round', facecolor=color, alpha=0.6))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/summary_statistics_comparison.png',
            bbox_inches='tight')
plt.close()
print("  Saved: summary_statistics_comparison.png")

# ============================================================================
# Plot 4: Structural Pattern Diagnostics
# ============================================================================
print("\nCreating Plot 4: Structural pattern diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prior Predictive Check: Structural Patterns', fontsize=16, fontweight='bold')

# (a) Pre-break vs Post-break slopes
ax = axes[0, 0]
pre_slopes = summary_stats['pre_break_slope'].values
post_slopes = summary_stats['post_break_slope'].values

# Filter out extreme outliers for visualization - apply to both arrays simultaneously
valid_mask = (pre_slopes > -5) & (pre_slopes < 5) & (post_slopes > -5) & (post_slopes < 5)
pre_slopes_clean = pre_slopes[valid_mask]
post_slopes_clean = post_slopes[valid_mask]

ax.scatter(pre_slopes_clean, post_slopes_clean, alpha=0.3, s=20, color='steelblue')

# Add observed values (compute from data)
log_obs = np.log(observed_counts + 1)
obs_pre_slope = np.polyfit(years[:16], log_obs[:16], 1)[0]
obs_post_slope = np.polyfit(years[16:], log_obs[16:], 1)[0]

ax.scatter(obs_pre_slope, obs_post_slope, color='red', s=200, marker='*',
           edgecolors='black', linewidth=2, label='Observed', zorder=10)

# Add diagonal line (no change in slope)
ax.plot([-5, 5], [-5, 5], 'k--', alpha=0.5, label='No slope change')

ax.set_xlabel('Pre-break Slope (log scale)', fontsize=11, fontweight='bold')
ax.set_ylabel('Post-break Slope (log scale)', fontsize=11, fontweight='bold')
ax.set_title('Structural Break Pattern: Slope Changes', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-2, 3])
ax.set_ylim([-2, 3])

# (b) Autocorrelation distribution
ax = axes[0, 1]
acf_values = summary_stats['acf_lag1'].values

ax.hist(acf_values, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
if acf_values.std() > 0:
    kde = gaussian_kde(acf_values)
    x_range = np.linspace(-0.5, 1.0, 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2)

ax.axvline(obs_acf1, color='red', linestyle='--', linewidth=3,
           label=f'Observed: {obs_acf1:.3f}')

# Expected range from prior
expected_rho_mean = 0.8
ax.axvline(expected_rho_mean, color='green', linestyle=':', linewidth=2,
           label=f'Prior E[ρ]: {expected_rho_mean:.2f}', alpha=0.7)

ax.set_xlabel('ACF(1)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Temporal Autocorrelation Pattern', fontsize=10)
ax.set_xlim([-0.5, 1.0])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Overdispersion (variance/mean ratio)
ax = axes[1, 0]
var_mean_ratios = summary_stats['variance_mean_ratio'].values

# Filter extreme outliers
var_mean_clean = var_mean_ratios[var_mean_ratios < 200]

ax.hist(var_mean_clean, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
if var_mean_clean.std() > 0:
    kde = gaussian_kde(var_mean_clean)
    x_range = np.linspace(0, 200, 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2)

ax.axvline(obs_var_mean_ratio, color='red', linestyle='--', linewidth=3,
           label=f'Observed: {obs_var_mean_ratio:.1f}')

# Poisson expectation (ratio = 1)
ax.axvline(1, color='orange', linestyle=':', linewidth=2,
           label='Poisson (ratio=1)', alpha=0.7)

ax.set_xlabel('Variance/Mean Ratio', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Overdispersion Pattern', fontsize=10)
ax.set_xlim([0, 200])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Growth pattern
ax = axes[1, 1]
growth_factors = summary_stats['growth_factor'].values

# Filter extreme outliers
growth_clean = growth_factors[growth_factors < 50]

ax.hist(growth_clean, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
if growth_clean.std() > 0:
    kde = gaussian_kde(growth_clean)
    x_range = np.linspace(0, 50, 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2)

ax.axvline(obs_growth, color='red', linestyle='--', linewidth=3,
           label=f'Observed: {obs_growth:.1f}')

# No growth line
ax.axvline(1, color='orange', linestyle=':', linewidth=2,
           label='No growth (ratio=1)', alpha=0.7)

ax.set_xlabel('Growth Factor (Final/Initial)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Overall Growth Pattern', fontsize=10)
ax.set_xlim([0, 50])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/structural_patterns.png',
            bbox_inches='tight')
plt.close()
print("  Saved: structural_patterns.png")

# ============================================================================
# Plot 5: Range Check - Detailed View
# ============================================================================
print("\nCreating Plot 5: Range check diagnostic...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prior Predictive Check: Range Diagnostics', fontsize=16, fontweight='bold')

# (a) Distribution of min and max counts
ax = axes[0]

min_counts = summary_stats['min_count'].values
max_counts = summary_stats['max_count'].values

# Create box plots
positions = [1, 2]
bp = ax.boxplot([min_counts, max_counts], positions=positions, widths=0.5,
                 patch_artist=True, showfliers=True)

for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)

# Add observed values
ax.scatter([1], [obs_min], color='red', s=200, marker='*',
           edgecolors='black', linewidth=2, zorder=10, label='Observed')
ax.scatter([2], [obs_max], color='red', s=200, marker='*',
           edgecolors='black', linewidth=2, zorder=10)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Minimum', 'Maximum'])
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Range Bounds: Min and Max Counts', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add reference lines
ax.axhline(10, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Plausible range')
ax.axhline(400, color='green', linestyle=':', linewidth=1, alpha=0.5)

# (b) Joint distribution of range
ax = axes[1]

# Scatter plot of min vs max
ax.scatter(min_counts, max_counts, alpha=0.2, s=20, color='steelblue')

# Add observed
ax.scatter(obs_min, obs_max, color='red', s=200, marker='*',
           edgecolors='black', linewidth=2, zorder=10, label='Observed')

# Add plausible region box
plausible_box = plt.Rectangle((10, 10), 390, 390, fill=False,
                               edgecolor='green', linewidth=2,
                               linestyle='--', label='Plausible region [10, 400]')
ax.add_patch(plausible_box)

ax.set_xlabel('Minimum Count', fontsize=11, fontweight='bold')
ax.set_ylabel('Maximum Count', fontsize=11, fontweight='bold')
ax.set_title('Joint Range Distribution', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 300])
ax.set_ylim([0, 2000])

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/range_diagnostic.png',
            bbox_inches='tight')
plt.close()
print("  Saved: range_diagnostic.png")

# ============================================================================
# Print Diagnostic Summary
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

# Critical checks
print("\nCRITICAL CHECKS:")

# 1. Range check
range_pass = np.mean((min_counts <= 400) & (max_counts >= 10))
print(f"1. Range [10, 400] coverage: {range_pass*100:.1f}% of draws (PASS if >50%)")

# 2. Growth pattern
growth_positive = np.mean(growth_factors > 1.0)
print(f"2. Positive growth: {growth_positive*100:.1f}% of draws (PASS if >30%)")

# 3. Autocorrelation
acf_in_range = np.mean((acf_values >= 0.6) & (acf_values <= 0.99))
print(f"3. ACF(1) in [0.6, 0.99]: {acf_in_range*100:.1f}% of draws (PASS if >50%)")

# 4. Structural break variety
slope_change = post_slopes - pre_slopes
has_break = np.mean(slope_change > 0.1)
print(f"4. Shows slope increase: {has_break*100:.1f}% of draws (PASS if >30%)")

# 5. Overdispersion
overdispersed = np.mean(var_mean_ratios > 1.0)
print(f"5. Overdispersed (var/mean > 1): {overdispersed*100:.1f}% of draws (PASS if >80%)")

# Observed data within prior predictive envelope
print("\nOBSERVED DATA POSITION:")
for stat_name, stat_value in observed_stats.items():
    sim_values = summary_stats[stat_name].values
    percentile = stats.percentileofscore(sim_values, stat_value)
    status = "GOOD" if 5 <= percentile <= 95 else "EDGE" if 1 <= percentile <= 99 else "EXTREME"
    print(f"  {stat_name:20s}: {percentile:5.1f}th percentile ({status})")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
