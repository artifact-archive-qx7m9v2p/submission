"""
Prior Predictive Check for Negative Binomial Quadratic Model (PyMC Implementation)
Purpose: Validate that priors generate scientifically plausible data before fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = "/workspace/data/data.csv"
CODE_DIR = "/workspace/experiments/experiment_1/prior_predictive_check/code"
PLOT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check/plots"

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"Data shape: {data.shape}")
print(f"Year range: [{data['year'].min():.2f}, {data['year'].max():.2f}]")
print(f"Count range: [{data['C'].min()}, {data['C'].max()}]")
print(f"Count mean: {data['C'].mean():.1f}, variance: {data['C'].var():.1f}")

# Build PyMC model
print("\nBuilding PyMC model...")
with pm.Model() as model:
    # Data
    year = pm.Data('year', data['year'].values)

    # Priors
    beta_0 = pm.Normal('beta_0', mu=4.7, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.8, sigma=0.3)
    beta_2 = pm.Normal('beta_2', mu=0.3, sigma=0.2)
    phi = pm.Gamma('phi', alpha=2, beta=0.5)

    # Expected value
    log_mu = beta_0 + beta_1 * year + beta_2 * year**2
    mu = pm.Deterministic('mu', pm.math.exp(log_mu))

    # Likelihood (not used in prior predictive, but included for completeness)
    # y = pm.NegativeBinomial('y', mu=mu, alpha=phi, observed=data['C'].values)

print("Sampling from prior predictive distribution...")
print("Generating 1000 parameter sets and simulated datasets...")

with model:
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=12345)

print("\nPrior predictive sampling complete!")

# Extract samples
print("Extracting samples...")
prior = prior_pred.prior

beta_0_samples = prior['beta_0'].values.flatten()
beta_1_samples = prior['beta_1'].values.flatten()
beta_2_samples = prior['beta_2'].values.flatten()
phi_samples = prior['phi'].values.flatten()
mu_samples = prior['mu'].values  # Shape: (chains, draws, observations)

# Flatten mu_samples to (n_samples, n_obs)
if len(mu_samples.shape) == 3:
    mu_samples = mu_samples.reshape(-1, mu_samples.shape[-1])

# Generate y_sim from negative binomial using mu and phi
print("Generating simulated counts from negative binomial distribution...")
n_samples = len(beta_0_samples)
n_obs = len(data)
y_sim = np.zeros((n_samples, n_obs))

for i in range(n_samples):
    for j in range(n_obs):
        # NegativeBinomial parameterization: mu and alpha (phi)
        # In PyMC/scipy, we need n and p where:
        # mu = n * (1-p) / p  =>  p = n / (n + mu)
        # variance = mu + mu^2/alpha
        # For NegativeBinomial(mu, alpha), we use:
        n_param = phi_samples[i]
        p_param = n_param / (n_param + mu_samples[i, j])
        y_sim[i, j] = np.random.negative_binomial(n_param, p_param)

print(f"Prior samples shape: {y_sim.shape}")
print(f"Parameter samples: beta_0={len(beta_0_samples)}, beta_1={len(beta_1_samples)}")

# ============================================================================
# DIAGNOSTIC COMPUTATIONS
# ============================================================================

print("\n" + "="*70)
print("PRIOR PREDICTIVE DIAGNOSTICS")
print("="*70)

# 1. Parameter summaries
print("\nParameter Prior Summaries:")
print(f"beta_0: mean={beta_0_samples.mean():.3f}, std={beta_0_samples.std():.3f}, "
      f"range=[{beta_0_samples.min():.3f}, {beta_0_samples.max():.3f}]")
print(f"beta_1: mean={beta_1_samples.mean():.3f}, std={beta_1_samples.std():.3f}, "
      f"range=[{beta_1_samples.min():.3f}, {beta_1_samples.max():.3f}]")
print(f"beta_2: mean={beta_2_samples.mean():.3f}, std={beta_2_samples.std():.3f}, "
      f"range=[{beta_2_samples.min():.3f}, {beta_2_samples.max():.3f}]")
print(f"phi: mean={phi_samples.mean():.3f}, std={phi_samples.std():.3f}, "
      f"range=[{phi_samples.min():.3f}, {phi_samples.max():.3f}]")

# 2. Simulated data summaries
sim_means = y_sim.mean(axis=1)
sim_maxs = y_sim.max(axis=1)
sim_mins = y_sim.min(axis=1)
sim_ranges = sim_maxs - sim_mins

print("\nSimulated Data Summaries (across all simulations):")
print(f"Mean count per simulation: mean={sim_means.mean():.1f}, "
      f"range=[{sim_means.min():.1f}, {sim_means.max():.1f}]")
print(f"Max count per simulation: mean={sim_maxs.mean():.1f}, "
      f"range=[{sim_maxs.min():.1f}, {sim_maxs.max():.1f}]")
print(f"Min count per simulation: mean={sim_mins.mean():.1f}, "
      f"range=[{sim_mins.min():.1f}, {sim_mins.max():.1f}]")
print(f"Range per simulation: mean={sim_ranges.mean():.1f}, "
      f"range=[{sim_ranges.min():.1f}, {sim_ranges.max():.1f}]")

# 3. Expected values at key time points
year_vals = data['year'].values
early_idx = 0  # First observation
mid_idx = len(year_vals) // 2  # Middle
late_idx = len(year_vals) - 1  # Last observation

print("\nExpected counts (mu) at key time points:")
print(f"Early (year={year_vals[early_idx]:.2f}): "
      f"median={np.median(mu_samples[:, early_idx]):.1f}, "
      f"90% CI=[{np.percentile(mu_samples[:, early_idx], 5):.1f}, "
      f"{np.percentile(mu_samples[:, early_idx], 95):.1f}]")
print(f"Mid (year={year_vals[mid_idx]:.2f}): "
      f"median={np.median(mu_samples[:, mid_idx]):.1f}, "
      f"90% CI=[{np.percentile(mu_samples[:, mid_idx], 5):.1f}, "
      f"{np.percentile(mu_samples[:, mid_idx], 95):.1f}]")
print(f"Late (year={year_vals[late_idx]:.2f}): "
      f"median={np.median(mu_samples[:, late_idx]):.1f}, "
      f"90% CI=[{np.percentile(mu_samples[:, late_idx], 5):.1f}, "
      f"{np.percentile(mu_samples[:, late_idx], 95):.1f}]")

# 4. Check for domain violations
print("\nDomain Violation Checks:")
negative_counts = (y_sim < 0).sum()
print(f"Negative counts: {negative_counts} (should be 0)")

# 5. Check for extreme values
extreme_high = (y_sim > 10000).sum()
extreme_low_mean = (sim_means < 1).sum()
print(f"Counts > 10,000: {extreme_high}")
print(f"Simulations with mean < 1: {extreme_low_mean}")

# 6. Coverage of observed data
obs_min, obs_max = data['C'].min(), data['C'].max()
covers_obs_min = (sim_mins <= obs_min).mean()
covers_obs_max = (sim_maxs >= obs_max).mean()
print(f"\nCoverage of observed range [{obs_min}, {obs_max}]:")
print(f"Simulations covering observed min: {covers_obs_min:.1%}")
print(f"Simulations covering observed max: {covers_obs_max:.1%}")

# 7. Plausibility check (counts in 10-500 range)
in_plausible_range = ((sim_means >= 10) & (sim_means <= 500)).mean()
print(f"Simulations with mean in [10, 500]: {in_plausible_range:.1%}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# ----------------------------------------------------------------------------
# Plot 1: Parameter Prior Distributions
# ----------------------------------------------------------------------------
print("\n1. Creating parameter plausibility visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Beta_0
axes[0, 0].hist(beta_0_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(4.7, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[0, 0].set_xlabel('β₀ (Intercept)', fontsize=12)
axes[0, 0].set_ylabel('Density', fontsize=12)
axes[0, 0].set_title('β₀ ~ Normal(4.7, 0.5)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Beta_1
axes[0, 1].hist(beta_1_samples, bins=50, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
axes[0, 1].axvline(0.8, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[0, 1].set_xlabel('β₁ (Linear term)', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].set_title('β₁ ~ Normal(0.8, 0.3)', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Beta_2
axes[1, 0].hist(beta_2_samples, bins=50, density=True, alpha=0.7, color='darkorange', edgecolor='black')
axes[1, 0].axvline(0.3, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[1, 0].set_xlabel('β₂ (Quadratic term)', fontsize=12)
axes[1, 0].set_ylabel('Density', fontsize=12)
axes[1, 0].set_title('β₂ ~ Normal(0.3, 0.2)', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Phi
axes[1, 1].hist(phi_samples, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(phi_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={phi_samples.mean():.2f}')
axes[1, 1].set_xlabel('φ (Overdispersion)', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].set_title('φ ~ Gamma(2, 0.5)', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'parameter_plausibility.png'), bbox_inches='tight')
plt.close()
print("   Saved: parameter_plausibility.png")

# ----------------------------------------------------------------------------
# Plot 2: Prior Predictive Trajectories (Spaghetti Plot)
# ----------------------------------------------------------------------------
print("2. Creating prior predictive trajectories...")

fig, ax = plt.subplots(figsize=(14, 8))

# Plot subset of simulations (100 random draws for visibility)
np.random.seed(42)
n_plot = 100
plot_indices = np.random.choice(y_sim.shape[0], size=n_plot, replace=False)

for idx in plot_indices:
    ax.plot(data['year'], y_sim[idx, :], alpha=0.15, color='steelblue', linewidth=0.8)

# Overlay observed data
ax.scatter(data['year'], data['C'], color='red', s=50, zorder=10,
           label='Observed data', edgecolors='darkred', linewidths=1.5)

# Add reference lines
ax.axhline(y=10, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Plausible range [10, 500]')
ax.axhline(y=500, color='green', linestyle=':', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Year (centered)', fontsize=13, fontweight='bold')
ax.set_ylabel('Count', fontsize=13, fontweight='bold')
ax.set_title('Prior Predictive Trajectories (100 random samples from 1000 total)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'prior_predictive_trajectories.png'), bbox_inches='tight')
plt.close()
print("   Saved: prior_predictive_trajectories.png")

# ----------------------------------------------------------------------------
# Plot 3: Prior Predictive Coverage Diagnostic
# ----------------------------------------------------------------------------
print("3. Creating prior predictive coverage diagnostic...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a: Simulated count distributions at early, mid, late time points
for idx, (time_idx, label, color) in enumerate([
    (early_idx, f'Early (year={year_vals[early_idx]:.2f})', 'steelblue'),
    (mid_idx, f'Mid (year={year_vals[mid_idx]:.2f})', 'forestgreen'),
    (late_idx, f'Late (year={year_vals[late_idx]:.2f})', 'darkorange')
]):
    axes[0, 0].hist(y_sim[:, time_idx], bins=50, density=True, alpha=0.5,
                     color=color, label=label, edgecolor='black', linewidth=0.5)
    axes[0, 0].axvline(data.iloc[time_idx]['C'], color=color, linestyle='--',
                       linewidth=2, alpha=0.8)

axes[0, 0].set_xlabel('Count', fontsize=12)
axes[0, 0].set_ylabel('Density', fontsize=12)
axes[0, 0].set_title('Simulated Counts at Key Time Points\n(dashed = observed)',
                      fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)

# 3b: Distribution of simulated ranges
axes[0, 1].hist(sim_ranges, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(obs_max - obs_min, color='red', linestyle='--', linewidth=2,
                   label=f'Observed range = {obs_max - obs_min}')
axes[0, 1].set_xlabel('Range (max - min)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Distribution of Simulated Ranges', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# 3c: Distribution of simulated means
axes[1, 0].hist(sim_means, bins=50, color='teal', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(data['C'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Observed mean = {data["C"].mean():.1f}')
axes[1, 0].axvline(10, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
axes[1, 0].axvline(500, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label='Plausible range [10, 500]')
axes[1, 0].set_xlabel('Mean count per simulation', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Simulated Means', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# 3d: Max counts per simulation
axes[1, 1].hist(sim_maxs, bins=50, color='crimson', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(obs_max, color='red', linestyle='--', linewidth=2,
                   label=f'Observed max = {obs_max}')
axes[1, 1].set_xlabel('Maximum count per simulation', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distribution of Simulated Maxima', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'prior_predictive_coverage.png'), bbox_inches='tight')
plt.close()
print("   Saved: prior_predictive_coverage.png")

# ----------------------------------------------------------------------------
# Plot 4: Expected Values (mu) Trajectories
# ----------------------------------------------------------------------------
print("4. Creating expected value trajectories...")

fig, ax = plt.subplots(figsize=(14, 8))

# Plot expected values (mu) for subset of simulations
for idx in plot_indices:
    ax.plot(data['year'], mu_samples[idx, :], alpha=0.15, color='navy', linewidth=0.8)

# Compute and plot median and credible intervals
mu_median = np.median(mu_samples, axis=0)
mu_05 = np.percentile(mu_samples, 5, axis=0)
mu_95 = np.percentile(mu_samples, 95, axis=0)

ax.plot(data['year'], mu_median, color='blue', linewidth=3, label='Median μ', zorder=5)
ax.fill_between(data['year'], mu_05, mu_95, color='blue', alpha=0.2, label='90% CI', zorder=4)

# Overlay observed data
ax.scatter(data['year'], data['C'], color='red', s=50, zorder=10,
           label='Observed data', edgecolors='darkred', linewidths=1.5)

ax.set_xlabel('Year (centered)', fontsize=13, fontweight='bold')
ax.set_ylabel('Expected Count (μ)', fontsize=13, fontweight='bold')
ax.set_title('Prior Distribution of Expected Values', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'expected_value_trajectories.png'), bbox_inches='tight')
plt.close()
print("   Saved: expected_value_trajectories.png")

# ----------------------------------------------------------------------------
# Plot 5: Parameter Space Coverage (Pairwise)
# ----------------------------------------------------------------------------
print("5. Creating parameter space coverage visualization...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Sample subset for plotting (2000 points)
n_sample = min(2000, len(beta_0_samples))
sample_idx = np.random.choice(len(beta_0_samples), size=n_sample, replace=False)

# Beta_0 vs Beta_1
axes[0, 0].scatter(beta_0_samples[sample_idx], beta_1_samples[sample_idx],
                   alpha=0.3, s=10, color='steelblue')
axes[0, 0].set_xlabel('β₀', fontsize=11)
axes[0, 0].set_ylabel('β₁', fontsize=11)
axes[0, 0].set_title('Intercept vs Linear Term', fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Beta_0 vs Beta_2
axes[0, 1].scatter(beta_0_samples[sample_idx], beta_2_samples[sample_idx],
                   alpha=0.3, s=10, color='forestgreen')
axes[0, 1].set_xlabel('β₀', fontsize=11)
axes[0, 1].set_ylabel('β₂', fontsize=11)
axes[0, 1].set_title('Intercept vs Quadratic Term', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Beta_1 vs Beta_2
axes[0, 2].scatter(beta_1_samples[sample_idx], beta_2_samples[sample_idx],
                   alpha=0.3, s=10, color='darkorange')
axes[0, 2].set_xlabel('β₁', fontsize=11)
axes[0, 2].set_ylabel('β₂', fontsize=11)
axes[0, 2].set_title('Linear vs Quadratic Term', fontsize=11, fontweight='bold')
axes[0, 2].grid(alpha=0.3)

# Beta_0 vs Phi
axes[1, 0].scatter(beta_0_samples[sample_idx], phi_samples[sample_idx],
                   alpha=0.3, s=10, color='purple')
axes[1, 0].set_xlabel('β₀', fontsize=11)
axes[1, 0].set_ylabel('φ', fontsize=11)
axes[1, 0].set_title('Intercept vs Overdispersion', fontsize=11, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Beta_1 vs Phi
axes[1, 1].scatter(beta_1_samples[sample_idx], phi_samples[sample_idx],
                   alpha=0.3, s=10, color='crimson')
axes[1, 1].set_xlabel('β₁', fontsize=11)
axes[1, 1].set_ylabel('φ', fontsize=11)
axes[1, 1].set_title('Linear Term vs Overdispersion', fontsize=11, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# Beta_2 vs Phi
axes[1, 2].scatter(beta_2_samples[sample_idx], phi_samples[sample_idx],
                   alpha=0.3, s=10, color='teal')
axes[1, 2].set_xlabel('β₂', fontsize=11)
axes[1, 2].set_ylabel('φ', fontsize=11)
axes[1, 2].set_title('Quadratic Term vs Overdispersion', fontsize=11, fontweight='bold')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'parameter_space_coverage.png'), bbox_inches='tight')
plt.close()
print("   Saved: parameter_space_coverage.png")

# ----------------------------------------------------------------------------
# Plot 6: Growth Pattern Diversity Check
# ----------------------------------------------------------------------------
print("6. Creating growth pattern diversity check...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 6a: Classify trajectories by curvature
# Compute curvature measure: if beta_2 > threshold, it's "strongly curved"
curvature_strength = np.abs(beta_2_samples)
linear_trajectories = curvature_strength < 0.1
moderate_curve = (curvature_strength >= 0.1) & (curvature_strength < 0.4)
strong_curve = curvature_strength >= 0.4

print(f"\nGrowth Pattern Diversity:")
print(f"  Near-linear (|β₂| < 0.1): {linear_trajectories.sum()} ({linear_trajectories.mean():.1%})")
print(f"  Moderate curve (0.1 ≤ |β₂| < 0.4): {moderate_curve.sum()} ({moderate_curve.mean():.1%})")
print(f"  Strong curve (|β₂| ≥ 0.4): {strong_curve.sum()} ({strong_curve.mean():.1%})")

# Plot examples of each type
n_examples = 30
colors = {'linear': 'green', 'moderate': 'orange', 'strong': 'red'}

for traj_type, mask, color, label in [
    ('linear', linear_trajectories, colors['linear'], 'Near-linear'),
    ('moderate', moderate_curve, colors['moderate'], 'Moderate curve'),
    ('strong', strong_curve, colors['strong'], 'Strong curve')
]:
    type_indices = np.where(mask)[0]
    if len(type_indices) > 0:
        sample_indices = np.random.choice(type_indices, size=min(n_examples, len(type_indices)), replace=False)
        for idx in sample_indices:
            axes[0].plot(data['year'], mu_samples[idx, :], alpha=0.3, color=color, linewidth=1)

# Add legend handles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=colors['linear'], linewidth=2, label=f"Near-linear ({linear_trajectories.mean():.0%})"),
    Line2D([0], [0], color=colors['moderate'], linewidth=2, label=f"Moderate ({moderate_curve.mean():.0%})"),
    Line2D([0], [0], color=colors['strong'], linewidth=2, label=f"Strong ({strong_curve.mean():.0%})")
]

axes[0].scatter(data['year'], data['C'], color='black', s=40, zorder=10, label='Observed', edgecolors='white', linewidths=1)
axes[0].set_xlabel('Year (centered)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Expected Count (μ)', fontsize=12, fontweight='bold')
axes[0].set_title('Growth Pattern Diversity by Curvature', fontsize=13, fontweight='bold')
axes[0].legend(handles=legend_elements + [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Observed')],
              fontsize=10)
axes[0].grid(alpha=0.3)

# 6b: Direction of curvature
positive_beta2 = beta_2_samples > 0
negative_beta2 = beta_2_samples < 0

print(f"  Positive β₂ (upward curve): {positive_beta2.sum()} ({positive_beta2.mean():.1%})")
print(f"  Negative β₂ (downward curve): {negative_beta2.sum()} ({negative_beta2.mean():.1%})")

# Plot histogram of beta_2
axes[1].hist(beta_2_samples, bins=60, color='steelblue', alpha=0.7, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='β₂ = 0 (linear)')
axes[1].axvline(beta_2_samples.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean β₂ = {beta_2_samples.mean():.3f}')
axes[1].set_xlabel('β₂ (Quadratic Coefficient)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution of Curvature Direction', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'growth_pattern_diversity.png'), bbox_inches='tight')
plt.close()
print("   Saved: growth_pattern_diversity.png")

print("\n" + "="*70)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("="*70)
print(f"\nAll visualizations saved to: {PLOT_DIR}")
print(f"Total plots created: 6")
print("\nNext step: Review findings.md for assessment and decision.")
