"""
Visualization for Prior Predictive Check
Creates diagnostic plots to assess prior plausibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
CODE_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/code")
PLOTS_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/plots")

# Load samples
data = np.load(CODE_DIR / "prior_samples.npz")
delta_samples = data['delta']
sigma_eta_samples = data['sigma_eta']
phi_samples = data['phi']
eta_samples = data['eta']
C_prior_samples = data['C_prior']
prior_growth_factor = data['prior_growth_factor']
observed_counts = data['observed_counts']
time_index = data['time_index']

N = len(observed_counts)
n_draws = len(delta_samples)

print(f"Loaded {n_draws} prior predictive draws for N={N} time points")

# ============================================================================
# PLOT 1: Parameter Prior Marginals
# ============================================================================

print("Creating parameter prior marginals plot...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Delta (drift)
axes[0].hist(delta_samples, bins=50, alpha=0.7, edgecolor='black', density=True)
axes[0].axvline(delta_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {delta_samples.mean():.3f}')
axes[0].axvline(0.05, color='green', linestyle='--', linewidth=2, label='Target: 0.05')
axes[0].set_xlabel('Drift (delta)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Prior: Delta ~ Normal(0.05, 0.02)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Sigma_eta (innovation SD)
axes[1].hist(sigma_eta_samples, bins=50, alpha=0.7, edgecolor='black', density=True, color='orange')
axes[1].axvline(np.median(sigma_eta_samples), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(sigma_eta_samples):.3f}')
axes[1].set_xlabel('Innovation SD (sigma_eta)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('Prior: Sigma_eta ~ Exp(10)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, np.percentile(sigma_eta_samples, 99))
axes[1].grid(alpha=0.3)

# Phi (dispersion)
axes[2].hist(phi_samples, bins=50, alpha=0.7, edgecolor='black', density=True, color='green')
axes[2].axvline(np.median(phi_samples), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(phi_samples):.1f}')
axes[2].set_xlabel('Dispersion (phi)', fontsize=11)
axes[2].set_ylabel('Density', fontsize=11)
axes[2].set_title('Prior: Phi ~ Exp(0.1)', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9)
axes[2].set_xlim(0, np.percentile(phi_samples, 99))
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_prior_marginals.png", dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: parameter_prior_marginals.png")

# ============================================================================
# PLOT 2: Prior Predictive Trajectories
# ============================================================================

print("Creating prior predictive trajectories plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot 50 random prior predictive trajectories
n_trajectories = 50
trajectory_indices = np.random.choice(n_draws, n_trajectories, replace=False)

for idx in trajectory_indices:
    ax.plot(time_index, C_prior_samples[idx, :], alpha=0.15, color='blue', linewidth=0.8)

# Overlay observed data
ax.plot(time_index, observed_counts, color='red', linewidth=3, marker='o',
        markersize=5, label='Observed Data', zorder=10)

# Add percentiles
prior_q025 = np.percentile(C_prior_samples, 2.5, axis=0)
prior_q975 = np.percentile(C_prior_samples, 97.5, axis=0)
prior_median = np.median(C_prior_samples, axis=0)

ax.plot(time_index, prior_median, color='darkblue', linewidth=2,
        linestyle='--', label='Prior Median', zorder=5)
ax.fill_between(time_index, prior_q025, prior_q975, alpha=0.2, color='blue',
                label='Prior 95% CI', zorder=1)

ax.set_xlabel('Time Index', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prior Predictive Trajectories (50 random draws)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_trajectories.png", dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: prior_predictive_trajectories.png")

# ============================================================================
# PLOT 3: Prior Predictive Coverage Diagnostics
# ============================================================================

print("Creating prior predictive coverage diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Distribution of prior predictive means
axes[0, 0].hist(C_prior_samples.mean(axis=1), bins=50, alpha=0.7, edgecolor='black', color='purple')
axes[0, 0].axvline(observed_counts.mean(), color='red', linewidth=3,
                   linestyle='--', label=f'Observed: {observed_counts.mean():.1f}')
axes[0, 0].axvline(C_prior_samples.mean(), color='blue', linewidth=2,
                   label=f'Prior Mean: {C_prior_samples.mean():.1f}')
axes[0, 0].set_xlabel('Mean Count Over Time Series', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('A) Distribution of Prior Predictive Means', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Panel B: Distribution of prior predictive maximums
axes[0, 1].hist(C_prior_samples.max(axis=1), bins=50, alpha=0.7, edgecolor='black', color='green')
axes[0, 1].axvline(observed_counts.max(), color='red', linewidth=3,
                   linestyle='--', label=f'Observed: {observed_counts.max()}')
axes[0, 1].axvline(np.median(C_prior_samples.max(axis=1)), color='blue', linewidth=2,
                   label=f'Prior Median: {np.median(C_prior_samples.max(axis=1)):.1f}')
axes[0, 1].set_xlabel('Maximum Count in Time Series', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('B) Distribution of Prior Predictive Max', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Panel C: Distribution of growth factors
axes[1, 0].hist(prior_growth_factor, bins=50, alpha=0.7, edgecolor='black', color='orange')
obs_growth = observed_counts[-1] / observed_counts[0]
axes[1, 0].axvline(obs_growth, color='red', linewidth=3,
                   linestyle='--', label=f'Observed: {obs_growth:.2f}x')
axes[1, 0].axvline(np.median(prior_growth_factor), color='blue', linewidth=2,
                   label=f'Prior Median: {np.median(prior_growth_factor):.2f}x')
axes[1, 0].set_xlabel('Growth Factor (C_40 / C_1)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('C) Distribution of Prior Growth Factors', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)
# Zoom in on reasonable range
axes[1, 0].set_xlim(0, np.percentile(prior_growth_factor, 99))

# Panel D: Time-specific coverage (violin plot)
# Sample a few time points
time_points_to_show = [0, 9, 19, 29, 39]  # First, then every 10th
violin_data = []
violin_labels = []

for t in time_points_to_show:
    violin_data.append(C_prior_samples[:, t])
    violin_labels.append(f't={t}')

parts = axes[1, 1].violinplot(violin_data, positions=range(len(time_points_to_show)),
                               showmeans=True, showmedians=True)
axes[1, 1].plot(range(len(time_points_to_show)),
                [observed_counts[t] for t in time_points_to_show],
                'ro-', linewidth=2, markersize=8, label='Observed')
axes[1, 1].set_xticks(range(len(time_points_to_show)))
axes[1, 1].set_xticklabels(violin_labels)
axes[1, 1].set_xlabel('Time Point', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('D) Prior Coverage at Selected Time Points', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_coverage.png", dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: prior_predictive_coverage.png")

# ============================================================================
# PLOT 4: Red Flags and Extreme Values
# ============================================================================

print("Creating red flags diagnostic plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Histogram of all prior predictive counts (log scale)
all_counts = C_prior_samples.flatten()
axes[0, 0].hist(all_counts, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
axes[0, 0].axvline(10000, color='red', linewidth=2, linestyle='--', label='Red flag: 10,000')
axes[0, 0].axvline(observed_counts.max(), color='green', linewidth=2,
                   linestyle='--', label=f'Observed max: {observed_counts.max()}')
axes[0, 0].set_xlabel('Count Value', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('A) Distribution of All Prior Predictive Counts', fontsize=12, fontweight='bold')
axes[0, 0].set_xlim(0, np.percentile(all_counts, 99.5))
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Panel B: Log scale version
axes[0, 1].hist(np.log1p(all_counts), bins=100, alpha=0.7, edgecolor='black', color='coral')
axes[0, 1].axvline(np.log1p(10000), color='red', linewidth=2, linestyle='--', label='log(10,000)')
axes[0, 1].axvline(np.log1p(observed_counts.max()), color='green', linewidth=2,
                   linestyle='--', label=f'log(obs max)')
axes[0, 1].set_xlabel('log(Count + 1)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('B) Log-Scale Distribution', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Panel C: Growth factor histogram with red flags
axes[1, 0].hist(prior_growth_factor, bins=100, alpha=0.7, edgecolor='black', color='mediumseagreen')
axes[1, 0].axvline(100, color='red', linewidth=2, linestyle='--', label='Red flag: 100x')
axes[1, 0].axvline(obs_growth, color='blue', linewidth=2,
                   linestyle='--', label=f'Observed: {obs_growth:.1f}x')
axes[1, 0].set_xlabel('Growth Factor', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('C) Growth Factor Distribution (with red flags)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlim(0, min(np.percentile(prior_growth_factor, 99.5), 150))
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

# Panel D: Parameter relationships (phi vs sigma_eta colored by extreme counts)
# Flag which draws produced extreme counts
has_extreme = (C_prior_samples > 10000).any(axis=1)
axes[1, 1].scatter(sigma_eta_samples[~has_extreme], phi_samples[~has_extreme],
                   alpha=0.3, s=20, c='blue', label='Normal')
axes[1, 1].scatter(sigma_eta_samples[has_extreme], phi_samples[has_extreme],
                   alpha=0.5, s=30, c='red', label=f'Extreme counts (n={has_extreme.sum()})')
axes[1, 1].set_xlabel('Innovation SD (sigma_eta)', fontsize=11)
axes[1, 1].set_ylabel('Dispersion (phi)', fontsize=11)
axes[1, 1].set_title('D) Parameter Space: Extreme Count Regions', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(0, np.percentile(sigma_eta_samples, 99))
axes[1, 1].set_ylim(0, np.percentile(phi_samples, 99))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "computational_red_flags.png", dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: computational_red_flags.png")

# ============================================================================
# PLOT 5: Latent State Trajectories (eta)
# ============================================================================

print("Creating latent state trajectories plot...")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel A: Prior predictive eta trajectories
n_trajectories = 50
trajectory_indices = np.random.choice(n_draws, n_trajectories, replace=False)

for idx in trajectory_indices:
    axes[0].plot(time_index, eta_samples[idx, :], alpha=0.15, color='purple', linewidth=0.8)

# Add percentiles
eta_q025 = np.percentile(eta_samples, 2.5, axis=0)
eta_q975 = np.percentile(eta_samples, 97.5, axis=0)
eta_median = np.median(eta_samples, axis=0)

axes[0].plot(time_index, eta_median, color='darkviolet', linewidth=2,
             linestyle='--', label='Prior Median eta', zorder=5)
axes[0].fill_between(time_index, eta_q025, eta_q975, alpha=0.2, color='purple',
                     label='Prior 95% CI', zorder=1)

# Add observed log-counts for comparison
obs_log_counts = np.log(observed_counts)
axes[0].plot(time_index, obs_log_counts, color='red', linewidth=2, marker='o',
             markersize=4, label='log(Observed Counts)', zorder=10)

axes[0].set_xlabel('Time Index', fontsize=11)
axes[0].set_ylabel('Log-Scale State (eta)', fontsize=11)
axes[0].set_title('A) Prior Predictive Latent State Trajectories', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Panel B: Distribution of initial states
axes[1].hist(eta_samples[:, 0], bins=50, alpha=0.7, edgecolor='black', color='teal')
axes[1].axvline(np.log(50), color='red', linewidth=2, linestyle='--',
                label=f'Prior mean: log(50) = {np.log(50):.2f}')
axes[1].axvline(obs_log_counts[0], color='green', linewidth=2,
                label=f'Observed: log({observed_counts[0]}) = {obs_log_counts[0]:.2f}')
axes[1].set_xlabel('Initial State (eta_1)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('B) Prior Distribution of Initial State', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "latent_state_prior.png", dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: latent_state_prior.png")

# ============================================================================
# PLOT 6: Joint Prior Behavior (Multi-panel parameter relationships)
# ============================================================================

print("Creating joint prior behavior plot...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Panel A: delta vs sigma_eta
axes[0, 0].scatter(delta_samples, sigma_eta_samples, alpha=0.3, s=10, c='steelblue')
axes[0, 0].set_xlabel('Drift (delta)', fontsize=10)
axes[0, 0].set_ylabel('Innovation SD (sigma_eta)', fontsize=10)
axes[0, 0].set_title('A) Drift vs Innovation SD', fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_ylim(0, np.percentile(sigma_eta_samples, 99))

# Panel B: delta vs phi
axes[0, 1].scatter(delta_samples, phi_samples, alpha=0.3, s=10, c='coral')
axes[0, 1].set_xlabel('Drift (delta)', fontsize=10)
axes[0, 1].set_ylabel('Dispersion (phi)', fontsize=10)
axes[0, 1].set_title('B) Drift vs Dispersion', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_ylim(0, np.percentile(phi_samples, 99))

# Panel C: sigma_eta vs phi
axes[0, 2].scatter(sigma_eta_samples, phi_samples, alpha=0.3, s=10, c='mediumseagreen')
axes[0, 2].set_xlabel('Innovation SD (sigma_eta)', fontsize=10)
axes[0, 2].set_ylabel('Dispersion (phi)', fontsize=10)
axes[0, 2].set_title('C) Innovation SD vs Dispersion', fontsize=11, fontweight='bold')
axes[0, 2].grid(alpha=0.3)
axes[0, 2].set_xlim(0, np.percentile(sigma_eta_samples, 99))
axes[0, 2].set_ylim(0, np.percentile(phi_samples, 99))

# Panel D: delta vs growth factor (colored by plausibility)
plausible_growth = (prior_growth_factor > 1) & (prior_growth_factor < 50)
axes[1, 0].scatter(delta_samples[plausible_growth], prior_growth_factor[plausible_growth],
                   alpha=0.3, s=10, c='green', label='Plausible')
axes[1, 0].scatter(delta_samples[~plausible_growth], prior_growth_factor[~plausible_growth],
                   alpha=0.3, s=10, c='red', label='Extreme')
axes[1, 0].axhline(obs_growth, color='blue', linewidth=2, linestyle='--', label='Observed')
axes[1, 0].set_xlabel('Drift (delta)', fontsize=10)
axes[1, 0].set_ylabel('Growth Factor', fontsize=10)
axes[1, 0].set_title('D) Drift vs Realized Growth', fontsize=11, fontweight='bold')
axes[1, 0].set_ylim(0, min(np.percentile(prior_growth_factor, 99), 60))
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Panel E: sigma_eta vs growth factor
axes[1, 1].scatter(sigma_eta_samples[plausible_growth], prior_growth_factor[plausible_growth],
                   alpha=0.3, s=10, c='green')
axes[1, 1].scatter(sigma_eta_samples[~plausible_growth], prior_growth_factor[~plausible_growth],
                   alpha=0.3, s=10, c='red')
axes[1, 1].axhline(obs_growth, color='blue', linewidth=2, linestyle='--')
axes[1, 1].set_xlabel('Innovation SD (sigma_eta)', fontsize=10)
axes[1, 1].set_ylabel('Growth Factor', fontsize=10)
axes[1, 1].set_title('E) Innovation SD vs Growth', fontsize=11, fontweight='bold')
axes[1, 1].set_xlim(0, np.percentile(sigma_eta_samples, 99))
axes[1, 1].set_ylim(0, min(np.percentile(prior_growth_factor, 99), 60))
axes[1, 1].grid(alpha=0.3)

# Panel F: Mean count vs max count (predictive space)
mean_counts = C_prior_samples.mean(axis=1)
max_counts = C_prior_samples.max(axis=1)
axes[1, 2].scatter(mean_counts, max_counts, alpha=0.3, s=10, c='purple')
axes[1, 2].scatter(observed_counts.mean(), observed_counts.max(),
                   s=200, c='red', marker='*', edgecolor='black', linewidth=2,
                   label='Observed', zorder=10)
axes[1, 2].set_xlabel('Mean Count', fontsize=10)
axes[1, 2].set_ylabel('Max Count', fontsize=10)
axes[1, 2].set_title('F) Prior Predictive Space', fontsize=11, fontweight='bold')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3)
axes[1, 2].set_xlim(0, np.percentile(mean_counts, 99.5))
axes[1, 2].set_ylim(0, np.percentile(max_counts, 99.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "joint_prior_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: joint_prior_diagnostics.png")

print("\n" + "="*70)
print("All visualizations complete!")
print("="*70)
print(f"\nPlots saved to: {PLOTS_DIR}")
print("\nGenerated files:")
print("  1. parameter_prior_marginals.png")
print("  2. prior_predictive_trajectories.png")
print("  3. prior_predictive_coverage.png")
print("  4. computational_red_flags.png")
print("  5. latent_state_prior.png")
print("  6. joint_prior_diagnostics.png")
