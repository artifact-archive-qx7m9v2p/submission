"""
Create comprehensive posterior predictive check visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
print("Loading data...")
C_obs = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/C_obs.npy')
C_rep = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/C_rep.npy')
year = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/year.npy')
tau = int(np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/tau.npy'))
test_stats = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/test_stats.npy',
                      allow_pickle=True).item()

n_pp_samples = C_rep.shape[0]
N = len(C_obs)
t = np.arange(1, N + 1)

print(f"  PP samples: {n_pp_samples}")
print(f"  Observations: {N}")
print(f"  Changepoint: {tau}")

# Plot directory
plot_dir = '/workspace/experiments/experiment_1/posterior_predictive_check/plots'

# ============================================================================
# Plot 1: PP Overlay - Time series with replicated data
# ============================================================================
print("\n[1/7] Creating PP overlay plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot subset of PP samples (50 thin lines)
n_plot = min(50, n_pp_samples)
idx_plot = np.random.choice(n_pp_samples, size=n_plot, replace=False)
for i in idx_plot:
    ax.plot(t, C_rep[i, :], color='skyblue', alpha=0.15, linewidth=0.5)

# PP mean and 90% HDI
pp_mean = C_rep.mean(axis=0)
pp_lower = np.percentile(C_rep, 5, axis=0)
pp_upper = np.percentile(C_rep, 95, axis=0)

ax.fill_between(t, pp_lower, pp_upper, color='steelblue', alpha=0.3,
                label='90% PP HDI')
ax.plot(t, pp_mean, color='navy', linewidth=2, label='PP Mean')

# Observed data
ax.plot(t, C_obs, 'o-', color='darkred', linewidth=2, markersize=5,
        label='Observed', zorder=10)

# Changepoint
ax.axvline(tau, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Changepoint (t={tau})')

ax.set_xlabel('Observation (t)', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title('Posterior Predictive Check: Observed vs Replicated Data', fontsize=14,
             fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.savefig(f'{plot_dir}/pp_overlay.png')
plt.close()
print("  Saved: pp_overlay.png")

# ============================================================================
# Plot 2: Test Statistics Comparison (6-panel)
# ============================================================================
print("\n[2/7] Creating test statistics comparison...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

stats_to_plot = [
    ('mean', 'Mean'),
    ('variance', 'Variance'),
    ('var_mean_ratio', 'Var/Mean Ratio'),
    ('min', 'Minimum'),
    ('max', 'Maximum'),
    ('acf1', 'ACF(1)')
]

for idx, (stat_key, stat_label) in enumerate(stats_to_plot):
    ax = axes[idx]
    stat = test_stats[stat_key]

    # Histogram of PP samples
    ax.hist(stat['pp_samples'], bins=30, color='steelblue', alpha=0.6,
            edgecolor='black', density=True)

    # Observed value
    ax.axvline(stat['observed'], color='darkred', linewidth=2.5,
               label='Observed', zorder=10)

    # PP mean
    ax.axvline(stat['pp_mean'], color='navy', linewidth=2, linestyle='--',
               label='PP Mean', alpha=0.8)

    # Add p-value text
    p_val = stat['p_value']
    status = 'EXTREME' if (p_val < 0.05 or p_val > 0.95) else 'OK'
    color = 'red' if status == 'EXTREME' else 'green'

    ax.text(0.05, 0.95, f"p = {p_val:.3f}\n{status}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax.set_xlabel(stat_label, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{stat_label} Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Test Statistics: Observed vs Posterior Predictive', fontsize=14,
             fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{plot_dir}/test_statistics.png')
plt.close()
print("  Saved: test_statistics.png")

# ============================================================================
# Plot 3: Pre/Post Break Comparison
# ============================================================================
print("\n[3/7] Creating regime comparison plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pre-break
ax = axes[0]
C_obs_pre = C_obs[:tau]
C_rep_pre = C_rep[:, :tau]

# Compute mean time series for pre-break
pp_mean_pre = C_rep_pre.mean(axis=0)
pp_lower_pre = np.percentile(C_rep_pre, 5, axis=0)
pp_upper_pre = np.percentile(C_rep_pre, 95, axis=0)
t_pre = np.arange(1, tau + 1)

# Plot subset of replicates
n_plot = 30
idx_plot = np.random.choice(n_pp_samples, size=n_plot, replace=False)
for i in idx_plot:
    ax.plot(t_pre, C_rep_pre[i, :], color='skyblue', alpha=0.2, linewidth=0.5)

ax.fill_between(t_pre, pp_lower_pre, pp_upper_pre, color='steelblue', alpha=0.3,
                label='90% PP HDI')
ax.plot(t_pre, pp_mean_pre, color='navy', linewidth=2, label='PP Mean')
ax.plot(t_pre, C_obs_pre, 'o-', color='darkred', linewidth=2, markersize=5,
        label='Observed', zorder=10)

ax.set_xlabel('Observation (t)', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title(f'Pre-break Regime (t ≤ {tau})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Post-break
ax = axes[1]
C_obs_post = C_obs[tau:]
C_rep_post = C_rep[:, tau:]

pp_mean_post = C_rep_post.mean(axis=0)
pp_lower_post = np.percentile(C_rep_post, 5, axis=0)
pp_upper_post = np.percentile(C_rep_post, 95, axis=0)
t_post = np.arange(tau + 1, N + 1)

for i in idx_plot:
    ax.plot(t_post, C_rep_post[i, :], color='skyblue', alpha=0.2, linewidth=0.5)

ax.fill_between(t_post, pp_lower_post, pp_upper_post, color='steelblue', alpha=0.3,
                label='90% PP HDI')
ax.plot(t_post, pp_mean_post, color='navy', linewidth=2, label='PP Mean')
ax.plot(t_post, C_obs_post, 'o-', color='darkred', linewidth=2, markersize=5,
        label='Observed', zorder=10)

ax.set_xlabel('Observation (t)', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title(f'Post-break Regime (t > {tau})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Regime-Specific Model Fit', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{plot_dir}/regime_comparison.png')
plt.close()
print("  Saved: regime_comparison.png")

# ============================================================================
# Plot 4: Quantile-Quantile Plot
# ============================================================================
print("\n[4/7] Creating Q-Q plot...")

fig, ax = plt.subplots(figsize=(8, 8))

# Compute empirical quantiles
quantile_levels = np.linspace(0.01, 0.99, 50)
obs_quantiles = np.quantile(C_obs, quantile_levels)

# Compute PP quantiles (average across samples)
pp_quantiles_all = np.array([np.quantile(C_rep[i, :], quantile_levels)
                              for i in range(n_pp_samples)])
pp_quantiles_mean = pp_quantiles_all.mean(axis=0)
pp_quantiles_lower = np.percentile(pp_quantiles_all, 5, axis=0)
pp_quantiles_upper = np.percentile(pp_quantiles_all, 95, axis=0)

# Plot
ax.fill_between(pp_quantiles_mean, pp_quantiles_lower, pp_quantiles_upper,
                color='steelblue', alpha=0.3, label='90% PP uncertainty')
ax.plot(pp_quantiles_mean, obs_quantiles, 'o', color='darkred', markersize=6,
        label='Empirical quantiles')

# Diagonal line (perfect fit)
q_min = min(pp_quantiles_mean.min(), obs_quantiles.min())
q_max = max(pp_quantiles_mean.max(), obs_quantiles.max())
ax.plot([q_min, q_max], [q_min, q_max], 'k--', linewidth=1.5,
        label='Perfect fit', alpha=0.7)

ax.set_xlabel('PP Quantiles (mean)', fontsize=12)
ax.set_ylabel('Observed Quantiles', fontsize=12)
ax.set_title('Quantile-Quantile Plot', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/qq_plot.png')
plt.close()
print("  Saved: qq_plot.png")

# ============================================================================
# Plot 5: Autocorrelation Comparison
# ============================================================================
print("\n[5/7] Creating autocorrelation comparison...")

def compute_acf(x, max_lag=10):
    """Compute autocorrelation function."""
    acf_vals = np.zeros(max_lag + 1)
    x_centered = x - x.mean()
    c0 = np.dot(x_centered, x_centered) / len(x)

    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals[lag] = 1.0
        else:
            c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / len(x)
            acf_vals[lag] = c_lag / c0 if c0 > 0 else 0
    return acf_vals

max_lag = 10
obs_acf = compute_acf(C_obs, max_lag)

# Compute ACF for each PP sample
pp_acf_all = np.array([compute_acf(C_rep[i, :], max_lag) for i in range(n_pp_samples)])
pp_acf_mean = pp_acf_all.mean(axis=0)
pp_acf_lower = np.percentile(pp_acf_all, 5, axis=0)
pp_acf_upper = np.percentile(pp_acf_all, 95, axis=0)

fig, ax = plt.subplots(figsize=(10, 6))

lags = np.arange(max_lag + 1)

# PP ACF uncertainty
ax.fill_between(lags, pp_acf_lower, pp_acf_upper, color='steelblue', alpha=0.3,
                label='90% PP HDI', step='mid')
ax.plot(lags, pp_acf_mean, 'o-', color='navy', linewidth=2, markersize=6,
        label='PP Mean ACF')

# Observed ACF
ax.plot(lags, obs_acf, 's-', color='darkred', linewidth=2, markersize=6,
        label='Observed ACF', zorder=10)

# Highlight lag 1 (the problematic one)
ax.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(1.1, 0.95, f'Lag 1:\nObs={obs_acf[1]:.3f}\nPP={pp_acf_mean[1]:.3f}',
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.set_title('Autocorrelation Function: Model Cannot Reproduce Observed ACF',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.2, 1.1)

plt.tight_layout()
plt.savefig(f'{plot_dir}/acf_comparison.png')
plt.close()
print("  Saved: acf_comparison.png")

# ============================================================================
# Plot 6: Marginal Distribution Comparison
# ============================================================================
print("\n[6/7] Creating marginal distribution comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Histogram comparison
ax = axes[0]

# PP histogram (pooled across samples)
C_rep_pooled = C_rep.flatten()
bins = np.linspace(0, max(C_obs.max(), C_rep_pooled.max()), 40)

ax.hist(C_rep_pooled, bins=bins, color='steelblue', alpha=0.5,
        label='PP Distribution', density=True, edgecolor='black')
ax.hist(C_obs, bins=bins, color='darkred', alpha=0.6,
        label='Observed', density=True, edgecolor='black')

ax.set_xlabel('Count (C)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Marginal Distribution Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: Empirical CDF
ax = axes[1]

# Observed ECDF
obs_sorted = np.sort(C_obs)
obs_ecdf = np.arange(1, N + 1) / N
ax.step(obs_sorted, obs_ecdf, color='darkred', linewidth=2,
        label='Observed ECDF', where='post')

# PP ECDF (median and HDI)
C_rep_sorted = np.sort(C_rep, axis=1)
pp_ecdf = np.arange(1, N + 1) / N
pp_ecdf_samples = []

for i in range(n_pp_samples):
    c_sorted = np.sort(C_rep[i, :])
    pp_ecdf_samples.append(c_sorted)

pp_ecdf_samples = np.array(pp_ecdf_samples)
pp_ecdf_median = np.median(pp_ecdf_samples, axis=0)
pp_ecdf_lower = np.percentile(pp_ecdf_samples, 5, axis=0)
pp_ecdf_upper = np.percentile(pp_ecdf_samples, 95, axis=0)

# Plot PP ECDF
ax.fill_betweenx(pp_ecdf, pp_ecdf_lower, pp_ecdf_upper, color='steelblue',
                 alpha=0.3, label='90% PP HDI', step='post')
ax.step(pp_ecdf_median, pp_ecdf, color='navy', linewidth=2,
        label='PP Median ECDF', where='post')

ax.set_xlabel('Count (C)', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('Empirical Cumulative Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/marginal_distribution.png')
plt.close()
print("  Saved: marginal_distribution.png")

# ============================================================================
# Plot 7: Coverage Assessment
# ============================================================================
print("\n[7/7] Creating coverage assessment plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Compute 90% HDI for each observation
pp_lower = np.percentile(C_rep, 5, axis=0)
pp_upper = np.percentile(C_rep, 95, axis=0)
pp_mean = C_rep.mean(axis=0)

# Coverage indicator
within_hdi = (C_obs >= pp_lower) & (C_obs <= pp_upper)

# Plot HDI bands
ax.fill_between(t, pp_lower, pp_upper, color='steelblue', alpha=0.3,
                label='90% PP HDI')

# Plot observed points, colored by coverage
colors = ['green' if w else 'red' for w in within_hdi]
ax.scatter(t, C_obs, c=colors, s=50, zorder=10,
           label=f'Observed (Coverage: {within_hdi.sum()}/{N})')

# PP mean
ax.plot(t, pp_mean, color='navy', linewidth=1.5, label='PP Mean', alpha=0.7)

# Changepoint
ax.axvline(tau, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Changepoint (t={tau})')

ax.set_xlabel('Observation (t)', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title(f'90% HDI Coverage: {within_hdi.sum()}/{N} points ({100*within_hdi.mean():.1f}%)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/coverage_assessment.png')
plt.close()
print("  Saved: coverage_assessment.png")

print("\n" + "="*80)
print("All plots created successfully!")
print(f"Location: {plot_dir}")
print("="*80)
