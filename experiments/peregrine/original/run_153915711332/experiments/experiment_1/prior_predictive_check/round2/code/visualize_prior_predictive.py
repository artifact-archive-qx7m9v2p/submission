"""
Visualization script for Round 2 Prior Predictive Check
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
data = pd.read_csv('/workspace/data/data.csv')
obs_counts = data['C'].values
N_TIME = len(obs_counts)

# Load prior samples
samples = np.load('/workspace/experiments/experiment_1/prior_predictive_check/round2/code/prior_samples.npz')
delta = samples['delta']
sigma_eta = samples['sigma_eta']
phi = samples['phi']
eta = samples['eta']
C_prior = samples['C_prior']
prior_mean_counts = samples['prior_mean_counts']
prior_max_counts = samples['prior_max_counts']
prior_growth_factors = samples['prior_growth_factors']
total_log_changes = samples['total_log_changes']

N_SAMPLES = len(delta)

print("Creating visualizations for Round 2...")
print("=" * 80)

# ============================================================================
# PLOT 1: Parameter Prior Marginals
# ============================================================================
print("Creating Plot 1: Parameter Prior Marginals...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Delta
ax = axes[0]
ax.hist(delta, bins=50, density=True, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(delta), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(delta):.4f}')
ax.axvline(0.05, color='blue', linestyle=':', linewidth=2, label='Prior mean: 0.05')
ax.set_xlabel('Delta (drift parameter)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Delta ~ Normal(0.05, 0.02)\n[KEPT from Round 1]', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Sigma_eta
ax = axes[1]
ax.hist(sigma_eta, bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(np.mean(sigma_eta), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sigma_eta):.4f}')
ax.axvline(np.median(sigma_eta), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sigma_eta):.4f}')
ax.axvline(1/20, color='blue', linestyle=':', linewidth=2, label='Prior mean: 0.05')
ax.set_xlabel('Sigma_eta (innovation SD)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Sigma_eta ~ Exponential(20)\n[ADJUSTED: was Exp(10)]', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Phi
ax = axes[2]
ax.hist(phi, bins=50, density=True, alpha=0.7, edgecolor='black', color='green')
ax.axvline(np.mean(phi), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(phi):.1f}')
ax.axvline(np.median(phi), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(phi):.1f}')
ax.axvline(20, color='blue', linestyle=':', linewidth=2, label='Prior mean: 20')
ax.set_xlabel('Phi (dispersion parameter)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Phi ~ Exponential(0.05)\n[ADJUSTED: was Exp(0.1)]', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/parameter_prior_marginals.png', dpi=300, bbox_inches='tight')
print("  Saved: parameter_prior_marginals.png")
plt.close()

# ============================================================================
# PLOT 2: Prior Predictive Trajectories
# ============================================================================
print("Creating Plot 2: Prior Predictive Trajectories...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel A: Count space
ax = axes[0]
n_show = 50
for i in range(n_show):
    ax.plot(range(1, N_TIME + 1), C_prior[i, :], alpha=0.3, linewidth=0.8, color='gray')

# Add percentiles
percentiles = [2.5, 25, 50, 75, 97.5]
colors_p = ['red', 'orange', 'blue', 'orange', 'red']
styles = ['--', '-.', '-', '-.', '--']
for p, c, s in zip(percentiles, colors_p, styles):
    perc_vals = np.percentile(C_prior, p, axis=0)
    ax.plot(range(1, N_TIME + 1), perc_vals, color=c, linestyle=s, linewidth=2,
            label=f'{p}th percentile', alpha=0.8)

# Observed data
ax.plot(range(1, N_TIME + 1), obs_counts, 'ko-', linewidth=3, markersize=5,
        label='Observed data', zorder=10)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prior Predictive Trajectories (Count Space) - Round 2\n50 random draws + percentiles vs observed',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Panel B: Log space
ax = axes[1]
for i in range(n_show):
    ax.plot(range(1, N_TIME + 1), np.log(C_prior[i, :] + 1), alpha=0.3, linewidth=0.8, color='gray')

# Add percentiles in log space
for p, c, s in zip(percentiles, colors_p, styles):
    perc_vals = np.log(np.percentile(C_prior, p, axis=0) + 1)
    ax.plot(range(1, N_TIME + 1), perc_vals, color=c, linestyle=s, linewidth=2,
            label=f'{p}th percentile', alpha=0.8)

ax.plot(range(1, N_TIME + 1), np.log(obs_counts + 1), 'ko-', linewidth=3, markersize=5,
        label='Observed data', zorder=10)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Log(Count + 1)', fontsize=12)
ax.set_title('Prior Predictive Trajectories (Log Space)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/prior_predictive_trajectories.png', dpi=300, bbox_inches='tight')
print("  Saved: prior_predictive_trajectories.png")
plt.close()

# ============================================================================
# PLOT 3: Coverage Diagnostics
# ============================================================================
print("Creating Plot 3: Coverage Diagnostics...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel A: Mean counts
ax = fig.add_subplot(gs[0, 0])
ax.hist(prior_mean_counts, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(obs_counts), color='red', linestyle='--', linewidth=3, label=f'Observed: {np.mean(obs_counts):.1f}')
ax.axvline(np.median(prior_mean_counts), color='blue', linestyle='--', linewidth=2, label=f'Prior median: {np.median(prior_mean_counts):.1f}')

# Add percentile info
obs_pct = 100 * np.sum(prior_mean_counts < np.mean(obs_counts)) / N_SAMPLES
ax.text(0.95, 0.95, f'Observed at {obs_pct:.1f}th percentile',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Mean Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive: Mean Counts', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Maximum counts
ax = fig.add_subplot(gs[0, 1])
ax.hist(prior_max_counts, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(np.max(obs_counts), color='red', linestyle='--', linewidth=3, label=f'Observed: {np.max(obs_counts)}')
ax.axvline(np.median(prior_max_counts), color='blue', linestyle='--', linewidth=2, label=f'Prior median: {np.median(prior_max_counts):.1f}')

obs_pct = 100 * np.sum(prior_max_counts < np.max(obs_counts)) / N_SAMPLES
ax.text(0.95, 0.95, f'Observed at {obs_pct:.1f}th percentile',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Maximum Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive: Maximum Counts', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel C: Growth factors
ax = fig.add_subplot(gs[1, 0])
ax.hist(prior_growth_factors, bins=50, edgecolor='black', alpha=0.7, color='green')
obs_growth = obs_counts[-1] / obs_counts[0]
ax.axvline(obs_growth, color='red', linestyle='--', linewidth=3, label=f'Observed: {obs_growth:.2f}x')
ax.axvline(np.median(prior_growth_factors), color='blue', linestyle='--', linewidth=2,
           label=f'Prior median: {np.median(prior_growth_factors):.2f}x')

obs_pct = 100 * np.sum(prior_growth_factors < obs_growth) / N_SAMPLES
ax.text(0.95, 0.95, f'Observed at {obs_pct:.1f}th percentile',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Growth Factor (C_40 / C_1)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive: Growth Factors', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# Panel D: Total log change
ax = fig.add_subplot(gs[1, 1])
ax.hist(total_log_changes, bins=50, edgecolor='black', alpha=0.7, color='purple')
obs_log_change = np.log(obs_counts[-1]) - np.log(obs_counts[0])
ax.axvline(obs_log_change, color='red', linestyle='--', linewidth=3, label=f'Observed: {obs_log_change:.2f}')
ax.axvline(np.median(total_log_changes), color='blue', linestyle='--', linewidth=2,
           label=f'Prior median: {np.median(total_log_changes):.2f}')

obs_pct = 100 * np.sum(total_log_changes < obs_log_change) / N_SAMPLES
ax.text(0.95, 0.95, f'Observed at {obs_pct:.1f}th percentile',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Total Log Change (eta_40 - eta_1)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive: Total Log Change', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel E: Time-specific coverage (violin plot)
ax = fig.add_subplot(gs[2, :])
time_indices = [0, 9, 19, 29, 39]  # t=1, 10, 20, 30, 40
time_labels = [f't={i+1}' for i in time_indices]
data_for_violin = [C_prior[:, i] for i in time_indices]

parts = ax.violinplot(data_for_violin, positions=range(len(time_indices)),
                      showmeans=True, showmedians=True)

# Overlay observed values
for idx, t in enumerate(time_indices):
    ax.scatter(idx, obs_counts[t], color='red', s=100, zorder=10, marker='D',
               edgecolors='black', linewidth=1.5)

ax.scatter([], [], color='red', s=100, marker='D', edgecolors='black', linewidth=1.5,
           label='Observed data')
ax.set_xticks(range(len(time_indices)))
ax.set_xticklabels(time_labels)
ax.set_xlabel('Time Point', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Time-Specific Prior Predictive Coverage (Violin Plots)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(bottom=0)

plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
print("  Saved: prior_predictive_coverage.png")
plt.close()

# ============================================================================
# PLOT 4: Computational Red Flags
# ============================================================================
print("Creating Plot 4: Computational Red Flags...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Distribution of all prior predictive counts
ax = axes[0, 0]
all_counts = C_prior.flatten()
ax.hist(all_counts, bins=100, edgecolor='black', alpha=0.7)
ax.axvline(1000, color='orange', linestyle='--', linewidth=2, label='Threshold: 1,000')
ax.axvline(10000, color='red', linestyle='--', linewidth=2, label='Threshold: 10,000')
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of All Prior Predictive Counts', fontsize=12, fontweight='bold')
ax.set_xlim([0, 2000])  # Zoom to relevant range
ax.legend()
ax.grid(True, alpha=0.3)

n_extreme = np.sum(all_counts > 1000)
pct_extreme = 100 * n_extreme / len(all_counts)
ax.text(0.95, 0.95, f'{n_extreme:,} counts > 1,000\n({pct_extreme:.2f}%)',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Panel B: Log scale version
ax = axes[0, 1]
ax.hist(np.log10(all_counts + 1), bins=100, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(np.log10(1000), color='orange', linestyle='--', linewidth=2, label='1,000')
ax.axvline(np.log10(10000), color='red', linestyle='--', linewidth=2, label='10,000')
ax.set_xlabel('Log10(Count + 1)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prior Predictive Counts (Log Scale)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

n_extreme_10k = np.sum(all_counts > 10000)
pct_extreme_10k = 100 * n_extreme_10k / len(all_counts)
ax.text(0.95, 0.95, f'{n_extreme_10k:,} counts > 10,000\n({pct_extreme_10k:.4f}%)',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Panel C: Growth factors distribution
ax = axes[1, 0]
ax.hist(prior_growth_factors, bins=100, edgecolor='black', alpha=0.7, color='green')
ax.axvline(50, color='orange', linestyle='--', linewidth=2, label='Threshold: 50x')
ax.axvline(100, color='red', linestyle='--', linewidth=2, label='Threshold: 100x')
ax.set_xlabel('Growth Factor', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Growth Factors', fontsize=12, fontweight='bold')
ax.set_xlim([0, 100])  # Zoom to relevant range
ax.legend()
ax.grid(True, alpha=0.3)

n_growth_50 = np.sum(prior_growth_factors > 50)
pct_growth_50 = 100 * n_growth_50 / len(prior_growth_factors)
ax.text(0.95, 0.95, f'{n_growth_50} trajectories > 50x\n({pct_growth_50:.2f}%)',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Panel D: Parameter space of extreme counts
ax = axes[1, 1]
max_counts_per_sample = np.max(C_prior, axis=1)
extreme_mask = max_counts_per_sample > 1000

ax.scatter(sigma_eta[~extreme_mask], phi[~extreme_mask], alpha=0.3, s=20,
           c='blue', label='Max count < 1,000')
ax.scatter(sigma_eta[extreme_mask], phi[extreme_mask], alpha=0.7, s=50,
           c='red', marker='x', label='Max count > 1,000')

ax.set_xlabel('Sigma_eta', fontsize=12)
ax.set_ylabel('Phi', fontsize=12)
ax.set_title('Parameter Space: Which Values Generate Extremes?', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

n_extreme_samples = np.sum(extreme_mask)
pct_extreme_samples = 100 * n_extreme_samples / len(max_counts_per_sample)
ax.text(0.95, 0.95, f'{n_extreme_samples} samples with\nmax > 1,000\n({pct_extreme_samples:.1f}%)',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/computational_red_flags.png', dpi=300, bbox_inches='tight')
print("  Saved: computational_red_flags.png")
plt.close()

# ============================================================================
# PLOT 5: Latent State Prior
# ============================================================================
print("Creating Plot 5: Latent State Prior...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Latent state trajectories
ax = axes[0, 0]
n_show = 50
for i in range(n_show):
    ax.plot(range(1, N_TIME + 1), eta[i, :], alpha=0.3, linewidth=0.8, color='gray')

# Add percentiles
percentiles = [2.5, 50, 97.5]
colors_p = ['red', 'blue', 'red']
styles = ['--', '-', '--']
for p, c, s in zip(percentiles, colors_p, styles):
    perc_vals = np.percentile(eta, p, axis=0)
    ax.plot(range(1, N_TIME + 1), perc_vals, color=c, linestyle=s, linewidth=2,
            label=f'{p}th percentile', alpha=0.8)

# Observed latent trajectory (log of observed counts as proxy)
ax.plot(range(1, N_TIME + 1), np.log(obs_counts), 'ko-', linewidth=2, markersize=4,
        label='Log(observed counts)', zorder=10, alpha=0.7)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Eta (log-scale latent state)', fontsize=12)
ax.set_title('Prior Predictive: Latent State Trajectories', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Initial state distribution
ax = axes[0, 1]
ax.hist(eta[:, 0], bins=50, density=True, edgecolor='black', alpha=0.7)
ax.axvline(np.log(50), color='blue', linestyle='--', linewidth=2, label='Prior mean: log(50)=3.91')
ax.axvline(np.log(obs_counts[0]), color='red', linestyle='--', linewidth=2,
           label=f'Observed: log({obs_counts[0]})={np.log(obs_counts[0]):.2f}')
ax.set_xlabel('Eta_1 (initial state)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution: Initial State', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel C: Final state distribution
ax = axes[1, 0]
ax.hist(eta[:, -1], bins=50, density=True, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(np.mean(eta[:, -1]), color='blue', linestyle='--', linewidth=2,
           label=f'Prior mean: {np.mean(eta[:, -1]):.2f}')
ax.axvline(np.log(obs_counts[-1]), color='red', linestyle='--', linewidth=2,
           label=f'Observed: log({obs_counts[-1]})={np.log(obs_counts[-1]):.2f}')
ax.set_xlabel('Eta_40 (final state)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution: Final State', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: State evolution (CI width over time)
ax = axes[1, 1]
ci_lower = np.percentile(eta, 2.5, axis=0)
ci_upper = np.percentile(eta, 97.5, axis=0)
ci_width = ci_upper - ci_lower

ax.plot(range(1, N_TIME + 1), ci_width, 'b-', linewidth=2, label='95% CI Width')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('95% CI Width (log scale)', fontsize=12)
ax.set_title('Prior Uncertainty Growth Over Time', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotation
mean_ci_width = np.mean(ci_width)
ax.text(0.95, 0.95, f'Mean CI width: {mean_ci_width:.2f}\nFinal CI width: {ci_width[-1]:.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/latent_state_prior.png', dpi=300, bbox_inches='tight')
print("  Saved: latent_state_prior.png")
plt.close()

# ============================================================================
# PLOT 6: Joint Prior Diagnostics
# ============================================================================
print("Creating Plot 6: Joint Prior Diagnostics...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel A: Delta vs Sigma_eta
ax = fig.add_subplot(gs[0, 0])
ax.scatter(delta, sigma_eta, alpha=0.3, s=20)
ax.set_xlabel('Delta', fontsize=12)
ax.set_ylabel('Sigma_eta', fontsize=12)
ax.set_title('Joint Prior: Delta vs Sigma_eta', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
corr = np.corrcoef(delta, sigma_eta)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
        va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Sigma_eta vs Phi
ax = fig.add_subplot(gs[0, 1])
ax.scatter(sigma_eta, phi, alpha=0.3, s=20, color='orange')
ax.set_xlabel('Sigma_eta', fontsize=12)
ax.set_ylabel('Phi', fontsize=12)
ax.set_title('Joint Prior: Sigma_eta vs Phi', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
corr = np.corrcoef(sigma_eta, phi)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
        va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel C: Delta vs Phi
ax = fig.add_subplot(gs[0, 2])
ax.scatter(delta, phi, alpha=0.3, s=20, color='green')
ax.set_xlabel('Delta', fontsize=12)
ax.set_ylabel('Phi', fontsize=12)
ax.set_title('Joint Prior: Delta vs Phi', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
corr = np.corrcoef(delta, phi)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
        va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel D: Sigma_eta vs Mean Count
ax = fig.add_subplot(gs[1, 0])
ax.scatter(sigma_eta, prior_mean_counts, alpha=0.3, s=20, color='purple')
ax.axhline(np.mean(obs_counts), color='red', linestyle='--', linewidth=2, label='Observed mean')
ax.set_xlabel('Sigma_eta', fontsize=12)
ax.set_ylabel('Prior Predictive Mean Count', fontsize=12)
ax.set_title('Parameter Impact: Sigma_eta vs Mean Count', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 2000])

# Panel E: Sigma_eta vs Growth Factor
ax = fig.add_subplot(gs[1, 1])
ax.scatter(sigma_eta, prior_growth_factors, alpha=0.3, s=20, color='brown')
obs_growth = obs_counts[-1] / obs_counts[0]
ax.axhline(obs_growth, color='red', linestyle='--', linewidth=2, label='Observed growth')
ax.set_xlabel('Sigma_eta', fontsize=12)
ax.set_ylabel('Growth Factor', fontsize=12)
ax.set_title('Parameter Impact: Sigma_eta vs Growth', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

# Panel F: Prior Predictive Space (Mean vs Max)
ax = fig.add_subplot(gs[1, 2])
ax.scatter(prior_mean_counts, prior_max_counts, alpha=0.3, s=20, color='teal')
ax.scatter(np.mean(obs_counts), np.max(obs_counts), color='red', s=200, marker='*',
           edgecolors='black', linewidth=2, label='Observed data', zorder=10)
ax.set_xlabel('Mean Count', fontsize=12)
ax.set_ylabel('Max Count', fontsize=12)
ax.set_title('Prior Predictive Space: Mean vs Max', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1500])
ax.set_ylim([0, 5000])

plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/joint_prior_diagnostics.png', dpi=300, bbox_inches='tight')
print("  Saved: joint_prior_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 7: Round 1 vs Round 2 Comparison
# ============================================================================
print("Creating Plot 7: Round 1 vs Round 2 Comparison...")

# Load Round 1 samples
try:
    samples_r1 = np.load('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.npz')
    sigma_eta_r1 = samples_r1['sigma_eta']
    phi_r1 = samples_r1['phi']
    C_prior_r1 = samples_r1['C_prior']
    prior_mean_counts_r1 = samples_r1['prior_mean_counts']
    prior_max_counts_r1 = samples_r1['prior_max_counts']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel A: Sigma_eta comparison
    ax = axes[0, 0]
    ax.hist(sigma_eta_r1, bins=50, alpha=0.5, label='Round 1 (Exp(10))', density=True, edgecolor='black')
    ax.hist(sigma_eta, bins=50, alpha=0.5, label='Round 2 (Exp(20))', density=True, edgecolor='black')
    ax.axvline(np.median(sigma_eta_r1), color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(np.median(sigma_eta), color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Sigma_eta', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Sigma_eta Prior: Round 1 vs Round 2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Phi comparison
    ax = axes[0, 1]
    ax.hist(phi_r1, bins=50, alpha=0.5, label='Round 1 (Exp(0.1))', density=True, edgecolor='black')
    ax.hist(phi, bins=50, alpha=0.5, label='Round 2 (Exp(0.05))', density=True, edgecolor='black')
    ax.axvline(np.median(phi_r1), color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(np.median(phi), color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Phi', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Phi Prior: Round 1 vs Round 2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Mean counts comparison
    ax = axes[0, 2]
    ax.hist(prior_mean_counts_r1, bins=50, alpha=0.5, label='Round 1', edgecolor='black', range=(0, 2000))
    ax.hist(prior_mean_counts, bins=50, alpha=0.5, label='Round 2', edgecolor='black', range=(0, 2000))
    ax.axvline(np.mean(obs_counts), color='red', linestyle='--', linewidth=3, label='Observed')
    ax.set_xlabel('Mean Count', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prior Predictive Mean: Round 1 vs Round 2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: Max counts comparison (log scale)
    ax = axes[1, 0]
    ax.hist(np.log10(prior_max_counts_r1 + 1), bins=50, alpha=0.5, label='Round 1', edgecolor='black')
    ax.hist(np.log10(prior_max_counts + 1), bins=50, alpha=0.5, label='Round 2', edgecolor='black')
    ax.axvline(np.log10(np.max(obs_counts) + 1), color='red', linestyle='--', linewidth=3, label='Observed')
    ax.set_xlabel('Log10(Max Count + 1)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prior Predictive Max: Round 1 vs Round 2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel E: Extreme counts comparison
    ax = axes[1, 1]
    all_counts_r1 = C_prior_r1.flatten()
    all_counts_r2 = C_prior.flatten()

    thresholds = [100, 500, 1000, 5000, 10000]
    pct_r1 = [100 * np.sum(all_counts_r1 > t) / len(all_counts_r1) for t in thresholds]
    pct_r2 = [100 * np.sum(all_counts_r2 > t) / len(all_counts_r2) for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.35
    ax.bar(x - width/2, pct_r1, width, label='Round 1', alpha=0.7)
    ax.bar(x + width/2, pct_r2, width, label='Round 2', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'>{t}' for t in thresholds], rotation=45)
    ax.set_xlabel('Count Threshold', fontsize=12)
    ax.set_ylabel('Percentage of All Counts', fontsize=12)
    ax.set_title('Extreme Count Frequency: Round 1 vs Round 2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Panel F: Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')

    table_data = [
        ['Metric', 'Round 1', 'Round 2', 'Change'],
        ['', '', '', ''],
        ['Sigma_eta median', f'{np.median(sigma_eta_r1):.3f}', f'{np.median(sigma_eta):.3f}',
         f'{((np.median(sigma_eta) - np.median(sigma_eta_r1)) / np.median(sigma_eta_r1) * 100):.1f}%'],
        ['Phi median', f'{np.median(phi_r1):.1f}', f'{np.median(phi):.1f}',
         f'{((np.median(phi) - np.median(phi_r1)) / np.median(phi_r1) * 100):.1f}%'],
        ['', '', '', ''],
        ['Mean of means', f'{np.mean(prior_mean_counts_r1):.1f}', f'{np.mean(prior_mean_counts):.1f}',
         f'{((np.mean(prior_mean_counts) - np.mean(prior_mean_counts_r1)) / np.mean(prior_mean_counts_r1) * 100):.1f}%'],
        ['Max 95% CI upper', f'{np.percentile(prior_max_counts_r1, 97.5):.0f}',
         f'{np.percentile(prior_max_counts, 97.5):.0f}',
         f'{((np.percentile(prior_max_counts, 97.5) - np.percentile(prior_max_counts_r1, 97.5)) / np.percentile(prior_max_counts_r1, 97.5) * 100):.1f}%'],
        ['', '', '', ''],
        ['Counts > 10,000', f'{100 * np.sum(all_counts_r1 > 10000) / len(all_counts_r1):.3f}%',
         f'{100 * np.sum(all_counts_r2 > 10000) / len(all_counts_r2):.3f}%',
         f'{((100 * np.sum(all_counts_r2 > 10000) / len(all_counts_r2)) - (100 * np.sum(all_counts_r1 > 10000) / len(all_counts_r1))):.3f}pp']
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Comparison', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/round1_vs_round2_comparison.png',
                dpi=300, bbox_inches='tight')
    print("  Saved: round1_vs_round2_comparison.png")
    plt.close()

except Exception as e:
    print(f"  WARNING: Could not create Round 1 vs Round 2 comparison: {e}")

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS COMPLETE!")
print("=" * 80)
print("\nPlots saved to: /workspace/experiments/experiment_1/prior_predictive_check/round2/plots/")
print("\nGenerated plots:")
print("  1. parameter_prior_marginals.png")
print("  2. prior_predictive_trajectories.png")
print("  3. prior_predictive_coverage.png")
print("  4. computational_red_flags.png")
print("  5. latent_state_prior.png")
print("  6. joint_prior_diagnostics.png")
print("  7. round1_vs_round2_comparison.png")
