"""
Prior Predictive Check for Experiment 2: Hierarchical Partial Pooling Model

Model:
    mu ~ Normal(10, 20)                    # Population mean
    tau ~ Half-Normal(0, 10)               # Between-group SD
    theta_i ~ Normal(mu, tau)              # Group means (via non-centered: theta = mu + tau * theta_raw)
    y_i ~ Normal(theta_i, sigma_i)         # Observations [sigma_i known from data]

This script performs comprehensive prior predictive checks to validate:
1. Hyperpriors generate reasonable parameter values (mu, tau)
2. Group-level priors (theta_i) are scientifically plausible
3. Prior predictive distribution covers observed data
4. Hierarchical structure allows appropriate range of pooling
5. No computational issues (extreme values, numerical instability)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("PRIOR PREDICTIVE CHECK: Experiment 2 - Hierarchical Partial Pooling Model")
print("="*80)
print()

# Load observed data
data_path = "/workspace/data/data.csv"
df = pd.read_csv(data_path)
y_obs = df['y'].values
sigma_obs = df['sigma'].values
n_obs = len(y_obs)

print("Observed Data Summary:")
print(f"  Number of observations: {n_obs}")
print(f"  y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"  y mean: {y_obs.mean():.2f}")
print(f"  y median: {np.median(y_obs):.2f}")
print(f"  sigma range: [{sigma_obs.min():.2f}, {sigma_obs.max():.2f}]")
print()

# ============================================================================
# PRIOR SPECIFICATION
# ============================================================================

prior_mu_mean = 10
prior_mu_sd = 20
prior_tau_sd = 10  # Half-Normal(0, 10)

print("Prior Specification:")
print(f"  mu ~ Normal({prior_mu_mean}, {prior_mu_sd})")
print(f"  tau ~ Half-Normal(0, {prior_tau_sd})")
print(f"  theta_i ~ Normal(mu, tau)  [via non-centered parameterization]")
print()

# Half-Normal distribution properties
tau_median = prior_tau_sd * np.sqrt(2 * np.log(2))  # Median of Half-Normal
tau_mean = prior_tau_sd * np.sqrt(2/np.pi)
print(f"Half-Normal(0, {prior_tau_sd}) properties:")
print(f"  Median: {tau_median:.2f}")
print(f"  Mean: {tau_mean:.2f}")
print(f"  95% quantile: {stats.halfnorm.ppf(0.95, scale=prior_tau_sd):.2f}")
print()

# ============================================================================
# 1. SAMPLE FROM HYPERPRIOR DISTRIBUTION
# ============================================================================

print("Step 1: Sampling from Hyperprior Distribution")
print("-"*80)

n_prior_samples = 5000

# Sample hyperpriors
mu_prior_samples = np.random.normal(prior_mu_mean, prior_mu_sd, n_prior_samples)
tau_prior_samples = np.abs(np.random.normal(0, prior_tau_sd, n_prior_samples))  # Half-Normal

print(f"  Sampled {n_prior_samples} values from hyperpriors")
print()
print("mu (population mean) samples:")
print(f"  Mean: {mu_prior_samples.mean():.2f}")
print(f"  Std: {mu_prior_samples.std():.2f}")
print(f"  95% interval: [{np.percentile(mu_prior_samples, 2.5):.2f}, {np.percentile(mu_prior_samples, 97.5):.2f}]")
print(f"  Range: [{mu_prior_samples.min():.2f}, {mu_prior_samples.max():.2f}]")
print()
print("tau (between-group SD) samples:")
print(f"  Mean: {tau_prior_samples.mean():.2f}")
print(f"  Median: {np.median(tau_prior_samples):.2f}")
print(f"  Std: {tau_prior_samples.std():.2f}")
print(f"  95% interval: [{np.percentile(tau_prior_samples, 2.5):.2f}, {np.percentile(tau_prior_samples, 97.5):.2f}]")
print(f"  Range: [{tau_prior_samples.min():.2f}, {tau_prior_samples.max():.2f}]")
print()

# Check for extreme values
extreme_threshold_mu = 100
extreme_threshold_tau = 50
n_extreme_mu = np.sum(np.abs(mu_prior_samples) > extreme_threshold_mu)
n_extreme_tau = np.sum(tau_prior_samples > extreme_threshold_tau)
print(f"  Extreme values check:")
print(f"    |mu| > {extreme_threshold_mu}: {n_extreme_mu} samples ({100*n_extreme_mu/n_prior_samples:.2f}%)")
print(f"    tau > {extreme_threshold_tau}: {n_extreme_tau} samples ({100*n_extreme_tau/n_prior_samples:.2f}%)")
print()

# ============================================================================
# 2. SAMPLE GROUP-LEVEL PARAMETERS (THETA)
# ============================================================================

print("Step 2: Sampling Group-Level Parameters (theta_i)")
print("-"*80)

# For each hyperprior sample, generate theta values for all 8 groups
theta_prior_samples = np.zeros((n_prior_samples, n_obs))

for i in range(n_prior_samples):
    mu_i = mu_prior_samples[i]
    tau_i = tau_prior_samples[i]
    # theta ~ N(mu, tau)
    theta_prior_samples[i, :] = np.random.normal(mu_i, tau_i, n_obs)

print(f"  Generated {n_prior_samples} samples of theta (each with {n_obs} group means)")
print()
print("theta (group means) summary:")
print(f"  Mean: {theta_prior_samples.mean():.2f}")
print(f"  Std: {theta_prior_samples.std():.2f}")
print(f"  95% interval: [{np.percentile(theta_prior_samples, 2.5):.2f}, {np.percentile(theta_prior_samples, 97.5):.2f}]")
print(f"  Range: [{theta_prior_samples.min():.2f}, {theta_prior_samples.max():.2f}]")
print()

# Within-sample variation (captures effect of tau)
within_sample_sds = theta_prior_samples.std(axis=1)  # SD across groups for each prior sample
print("Between-group variation (within each prior sample):")
print(f"  Mean SD across groups: {within_sample_sds.mean():.2f}")
print(f"  Median SD across groups: {np.median(within_sample_sds):.2f}")
print(f"  95% interval: [{np.percentile(within_sample_sds, 2.5):.2f}, {np.percentile(within_sample_sds, 97.5):.2f}]")
print()

# ============================================================================
# 3. GENERATE PRIOR PREDICTIVE SAMPLES
# ============================================================================

print("Step 3: Generating Prior Predictive Samples")
print("-"*80)

# For each prior sample (mu, tau, theta), generate predicted y values
y_prior_pred = np.zeros((n_prior_samples, n_obs))

for i in range(n_prior_samples):
    theta_i = theta_prior_samples[i, :]
    # Generate y_pred ~ N(theta_j, sigma_j) for each observation j
    for j in range(n_obs):
        y_prior_pred[i, j] = np.random.normal(theta_i[j], sigma_obs[j])

print(f"  Generated {n_prior_samples} prior predictive datasets")
print(f"  Each dataset has {n_obs} observations")
print()

# Summary statistics for prior predictive samples
print("Prior Predictive Summary (across all observations):")
y_prior_pred_flat = y_prior_pred.flatten()
print(f"  Mean: {y_prior_pred_flat.mean():.2f}")
print(f"  Std: {y_prior_pred_flat.std():.2f}")
print(f"  95% interval: [{np.percentile(y_prior_pred_flat, 2.5):.2f}, {np.percentile(y_prior_pred_flat, 97.5):.2f}]")
print(f"  Range: [{y_prior_pred_flat.min():.2f}, {y_prior_pred_flat.max():.2f}]")
print()

# ============================================================================
# 4. PRIOR-DATA COMPATIBILITY CHECKS
# ============================================================================

print("Step 4: Prior-Data Compatibility Analysis")
print("-"*80)

# For each observation, compute percentile rank in its prior predictive distribution
percentile_ranks = np.zeros(n_obs)
for j in range(n_obs):
    # Prior predictive samples for observation j
    y_pred_j = y_prior_pred[:, j]
    # Percentile rank of observed value
    percentile_ranks[j] = stats.percentileofscore(y_pred_j, y_obs[j])

print("Percentile Ranks of Observed Data in Prior Predictive:")
for j in range(n_obs):
    status = "OK" if 5 <= percentile_ranks[j] <= 95 else "WARNING"
    print(f"  Obs {j} (y={y_obs[j]:6.2f}, sigma={sigma_obs[j]:2.0f}): {percentile_ranks[j]:5.1f}% [{status}]")

print()

# Count observations in extreme tails
n_extreme_low = np.sum(percentile_ranks < 5)
n_extreme_high = np.sum(percentile_ranks > 95)
print(f"  Observations in extreme tails:")
print(f"    < 5th percentile: {n_extreme_low}")
print(f"    > 95th percentile: {n_extreme_high}")
print()

# ============================================================================
# 5. COMPUTATIONAL DIAGNOSTICS
# ============================================================================

print("Step 5: Computational Diagnostics")
print("-"*80)

# Check for numerical issues
n_nan = np.sum(np.isnan(y_prior_pred))
n_inf = np.sum(np.isinf(y_prior_pred))
max_abs_value = np.max(np.abs(y_prior_pred[np.isfinite(y_prior_pred)]))

print(f"  NaN values: {n_nan}")
print(f"  Inf values: {n_inf}")
print(f"  Max absolute value: {max_abs_value:.2f}")

if max_abs_value > 1000:
    print(f"  WARNING: Some prior predictive values exceed 1000 (max={max_abs_value:.2f})")
    print(f"           This may cause computational issues during inference")
else:
    print(f"  Computational check: PASSED (all values in reasonable range)")

print()

# ============================================================================
# VISUALIZATION 1: HYPERPRIOR DISTRIBUTIONS
# ============================================================================

print("Creating Visualization 1: Hyperprior Distributions")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: mu prior distribution
ax = axes[0, 0]
ax.hist(mu_prior_samples, bins=60, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', label='Prior samples')

# Overlay theoretical prior density
x_range = np.linspace(mu_prior_samples.min(), mu_prior_samples.max(), 300)
prior_density = stats.norm.pdf(x_range, prior_mu_mean, prior_mu_sd)
ax.plot(x_range, prior_density, 'r-', linewidth=2, label='Prior N(10, 20)')

# Add observed data mean for reference
ax.axvline(y_obs.mean(), color='green', linestyle='--', linewidth=2,
           label=f'Observed mean ({y_obs.mean():.1f})')

# Add 95% prior interval
prior_lower = prior_mu_mean - 1.96 * prior_mu_sd
prior_upper = prior_mu_mean + 1.96 * prior_mu_sd
ax.axvline(prior_lower, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
ax.axvline(prior_upper, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
ax.fill_betweenx([0, ax.get_ylim()[1]], prior_lower, prior_upper,
                  alpha=0.1, color='red', label='95% prior interval')

ax.set_xlabel('mu (population mean)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Hyperprior: mu ~ N(10, 20)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Panel B: tau prior distribution
ax = axes[0, 1]
ax.hist(tau_prior_samples, bins=60, density=True, alpha=0.6, color='coral',
        edgecolor='black', label='Prior samples')

# Overlay theoretical prior density (Half-Normal)
x_range_tau = np.linspace(0, tau_prior_samples.max(), 300)
prior_tau_density = stats.halfnorm.pdf(x_range_tau, scale=prior_tau_sd)
ax.plot(x_range_tau, prior_tau_density, 'r-', linewidth=2, label='Half-Normal(0, 10)')

# Add median and 95th percentile
tau_median_theoretical = prior_tau_sd * np.sqrt(2 * np.log(2))
tau_95 = stats.halfnorm.ppf(0.95, scale=prior_tau_sd)
ax.axvline(tau_median_theoretical, color='purple', linestyle='--', linewidth=2,
           label=f'Median ({tau_median_theoretical:.1f})')
ax.axvline(tau_95, color='orange', linestyle=':', linewidth=2,
           label=f'95% ({tau_95:.1f})')

ax.set_xlabel('tau (between-group SD)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Hyperprior: tau ~ Half-Normal(0, 10)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, None)

# Panel C: Joint distribution (mu vs tau)
ax = axes[1, 0]
h = ax.hexbin(mu_prior_samples, tau_prior_samples, gridsize=50, cmap='Blues', mincnt=1)
ax.axvline(prior_mu_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(tau_median_theoretical, color='purple', linestyle='--', linewidth=2, alpha=0.7)
plt.colorbar(h, ax=ax, label='Count')

ax.set_xlabel('mu (population mean)', fontsize=12)
ax.set_ylabel('tau (between-group SD)', fontsize=12)
ax.set_title('Joint Hyperprior: mu vs tau', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel D: Tau quantiles
ax = axes[1, 1]
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
quantiles_tau = np.percentile(tau_prior_samples, percentiles)

ax.plot(percentiles, quantiles_tau, 'o-', linewidth=2, markersize=8, color='coral')
ax.axhline(tau_median_theoretical, color='purple', linestyle='--', linewidth=2,
           label=f'Theoretical median ({tau_median_theoretical:.1f})')
ax.axhline(5, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
ax.axhline(15, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)

# Shade regularizing range
ax.fill_between([0, 100], 0, 10, alpha=0.1, color='green', label='Strong regularization')
ax.fill_between([0, 100], 10, 20, alpha=0.1, color='orange', label='Moderate variation')

ax.set_xlabel('Percentile', fontsize=12)
ax.set_ylabel('tau value', fontsize=12)
ax.set_title('Prior Quantiles for tau', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/prior_predictive_check/plots/hyperprior_distributions.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/hyperprior_distributions.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: GROUP-LEVEL PARAMETERS (THETA)
# ============================================================================

print("Creating Visualization 2: Group-Level Parameters (theta_i)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Distribution of all theta samples
ax = axes[0, 0]
ax.hist(theta_prior_samples.flatten(), bins=80, density=True, alpha=0.6,
        color='mediumseagreen', edgecolor='black', label='All theta samples')

# Add observed y values for reference
for y in y_obs:
    ax.axvline(y, color='red', alpha=0.3, linewidth=1.5)
ax.axvline(y_obs[0], color='red', alpha=0.3, linewidth=1.5, label='Observed y values')

# Add observed mean
ax.axvline(y_obs.mean(), color='darkred', linestyle='--', linewidth=2,
           label=f'Observed mean ({y_obs.mean():.1f})')

ax.set_xlabel('theta (group mean)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution: All Group Means (theta_i)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Panel B: Within-sample variation (effect of tau)
ax = axes[1, 0]
ax.hist(within_sample_sds, bins=50, density=True, alpha=0.6, color='coral',
        edgecolor='black', label='Within-sample SD')

# Overlay tau distribution for comparison
ax.hist(tau_prior_samples, bins=50, density=True, alpha=0.4, color='blue',
        edgecolor='black', label='tau (prior)')

ax.axvline(within_sample_sds.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean within-SD ({within_sample_sds.mean():.1f})')

ax.set_xlabel('Standard deviation', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Between-Group Variation: Within-Sample SD vs tau', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Panel C: Individual group theta distributions
ax = axes[0, 1]
# Box plot for each group
bp = ax.boxplot([theta_prior_samples[:, j] for j in range(n_obs)],
                 labels=[f'{i}' for i in range(n_obs)],
                 patch_artist=True, showfliers=False)

for patch in bp['boxes']:
    patch.set_facecolor('mediumseagreen')
    patch.set_alpha(0.6)

# Overlay observed values
ax.scatter(range(1, n_obs+1), y_obs, color='red', s=100, marker='x',
           linewidths=3, label='Observed y', zorder=5)

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('theta value', fontsize=12)
ax.set_title('Prior Distribution per Group', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

# Panel D: Sample trajectories of theta
ax = axes[1, 1]
# Plot a random sample of prior parameter sets
n_trajectories = 200
idx_sample = np.random.choice(n_prior_samples, n_trajectories, replace=False)
for idx in idx_sample:
    ax.plot(range(n_obs), theta_prior_samples[idx, :], 'o-', alpha=0.05,
            color='blue', markersize=3)

# Overlay observed data
ax.plot(range(n_obs), y_obs, 'ro-', linewidth=3, markersize=10,
        label='Observed', zorder=10)

# Add median and intervals
theta_median = np.median(theta_prior_samples, axis=0)
theta_lower_50 = np.percentile(theta_prior_samples, 25, axis=0)
theta_upper_50 = np.percentile(theta_prior_samples, 75, axis=0)
theta_lower_95 = np.percentile(theta_prior_samples, 2.5, axis=0)
theta_upper_95 = np.percentile(theta_prior_samples, 97.5, axis=0)

ax.plot(range(n_obs), theta_median, 'k-', linewidth=2, label='Median', zorder=9)
ax.fill_between(range(n_obs), theta_lower_50, theta_upper_50, alpha=0.3,
                color='green', label='50% interval', zorder=8)
ax.fill_between(range(n_obs), theta_lower_95, theta_upper_95, alpha=0.1,
                color='green', label='95% interval', zorder=7)

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('theta value', fontsize=12)
ax.set_title('Prior Sample Trajectories: Group Means', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_obs))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/prior_predictive_check/plots/group_level_parameters.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/group_level_parameters.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: PRIOR PREDICTIVE OVERLAY
# ============================================================================

print("Creating Visualization 3: Prior Predictive Coverage")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(n_obs):
    ax = axes[j]

    # Prior predictive samples for this observation
    y_pred_j = y_prior_pred[:, j]

    # Histogram of prior predictive
    ax.hist(y_pred_j, bins=50, density=True, alpha=0.5, color='lightblue',
            edgecolor='black', linewidth=0.5, label='Prior predictive')

    # Note: No simple theoretical density for hierarchical model
    # (it's a mixture over mu and tau)

    # Mark observed value
    ax.axvline(y_obs[j], color='red', linestyle='--', linewidth=2.5,
               label=f'Observed ({y_obs[j]:.1f})')

    # Mark 95% prior predictive interval
    pp_lower = np.percentile(y_pred_j, 2.5)
    pp_upper = np.percentile(y_pred_j, 97.5)
    ax.axvline(pp_lower, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(pp_upper, color='green', linestyle=':', alpha=0.5, linewidth=1.5)

    # Percentile rank
    pct = percentile_ranks[j]
    color = 'black' if 5 <= pct <= 95 else 'red'

    ax.set_xlabel('y', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Group {j} (sigma={sigma_obs[j]:.0f}, pct={pct:.0f}%)',
                 fontsize=11, fontweight='bold', color=color)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Prior Predictive Distributions vs Observed Data',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_predictive_coverage.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/prior_predictive_coverage.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: HIERARCHICAL STRUCTURE CHECK
# ============================================================================

print("Creating Visualization 4: Hierarchical Structure Analysis")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Effect of tau on predictions
ax = axes[0, 0]

# Select specific tau values to illustrate pooling behavior
tau_values = [0, 5, 10, 20]
colors_tau = ['red', 'green', 'blue', 'purple']

for tau_val, color in zip(tau_values, colors_tau):
    # Generate predictions for this specific tau value
    mu_fixed = 10  # Use prior mean
    theta_fixed = np.random.normal(mu_fixed, tau_val, n_obs)
    y_fixed = np.array([np.random.normal(theta_fixed[j], sigma_obs[j])
                        for j in range(n_obs)])

    ax.plot(range(n_obs), theta_fixed, 'o-', color=color, alpha=0.6,
            linewidth=2, markersize=8, label=f'tau={tau_val}')

# Overlay observed data
ax.plot(range(n_obs), y_obs, 'ko', markersize=12, label='Observed',
        markeredgewidth=2, markerfacecolor='yellow', zorder=10)

ax.axhline(10, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
           label='mu=10 (complete pooling)')

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('theta value', fontsize=12)
ax.set_title('Effect of tau on Group Means', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_obs))

# Panel B: Prior predictive trajectories
ax = axes[0, 1]
# Plot a random sample of prior predictive datasets
n_trajectories = 100
idx_sample = np.random.choice(n_prior_samples, n_trajectories, replace=False)
for idx in idx_sample:
    ax.plot(range(n_obs), y_prior_pred[idx, :], 'o-', alpha=0.1, color='blue', markersize=3)

# Overlay observed data
ax.plot(range(n_obs), y_obs, 'ro-', linewidth=3, markersize=10,
        label='Observed', zorder=10)

# Add 50% and 95% prior predictive intervals
pp_median = np.median(y_prior_pred, axis=0)
pp_lower_50 = np.percentile(y_prior_pred, 25, axis=0)
pp_upper_50 = np.percentile(y_prior_pred, 75, axis=0)
pp_lower_95 = np.percentile(y_prior_pred, 2.5, axis=0)
pp_upper_95 = np.percentile(y_prior_pred, 97.5, axis=0)

ax.plot(range(n_obs), pp_median, 'k-', linewidth=2, label='Median', zorder=9)
ax.fill_between(range(n_obs), pp_lower_50, pp_upper_50, alpha=0.3,
                color='green', label='50% interval', zorder=8)
ax.fill_between(range(n_obs), pp_lower_95, pp_upper_95, alpha=0.1,
                color='green', label='95% interval', zorder=7)

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('y value', fontsize=12)
ax.set_title('Prior Predictive Trajectories', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_obs))

# Panel C: Variance decomposition
ax = axes[1, 0]
# For each observation, decompose total variance into components
# Total variance = tau^2 + sigma^2 (approximately, ignoring mu uncertainty)

# Average variance components
mean_mu_var = prior_mu_sd**2
mean_tau_sq = tau_prior_samples**2
mean_between_var = mean_tau_sq.mean()

# For each observation
variance_components = np.zeros((n_obs, 3))
for j in range(n_obs):
    # Mu uncertainty
    variance_components[j, 0] = mean_mu_var
    # Between-group (tau)
    variance_components[j, 1] = mean_between_var
    # Measurement error
    variance_components[j, 2] = sigma_obs[j]**2

# Calculate percentages
total_var = variance_components.sum(axis=1)
pct_mu = variance_components[:, 0] / total_var * 100
pct_tau = variance_components[:, 1] / total_var * 100
pct_sigma = variance_components[:, 2] / total_var * 100

x = range(n_obs)
ax.bar(x, pct_mu, label='Hyperprior mu variance', alpha=0.7, color='steelblue',
       edgecolor='black')
ax.bar(x, pct_tau, bottom=pct_mu, label='Between-group tau variance',
       alpha=0.7, color='coral', edgecolor='black')
ax.bar(x, pct_sigma, bottom=pct_mu+pct_tau, label='Measurement error variance',
       alpha=0.7, color='green', edgecolor='black')

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Variance Contribution (%)', fontsize=12)
ax.set_title('Prior Variance Decomposition', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(n_obs))
ax.set_xticklabels([f'{i}\n(σ={sigma_obs[i]:.0f})' for i in range(n_obs)], fontsize=9)
ax.set_ylim(0, 100)

# Panel D: Rank statistics
ax = axes[1, 1]
colors = ['red' if (pct < 5 or pct > 95) else 'green' for pct in percentile_ranks]
ax.barh(range(n_obs), percentile_ranks, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='5th/95th percentile')
ax.axvline(95, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(50, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Median')
ax.set_xlabel('Percentile Rank', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Percentile Rank per Group', fontsize=13, fontweight='bold')
ax.set_yticks(range(n_obs))
ax.set_yticklabels([f'Group {i}' for i in range(n_obs)])
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/prior_predictive_check/plots/hierarchical_structure.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/hierarchical_structure.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: PRIOR-DATA COMPATIBILITY
# ============================================================================

print("Creating Visualization 5: Prior-Data Compatibility")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Rank histogram
ax = axes[0, 0]
ax.hist(percentile_ranks, bins=20, range=(0, 100), density=True,
        alpha=0.6, color='steelblue', edgecolor='black', linewidth=1.5)
ax.axhline(1/100, color='red', linestyle='--', linewidth=2,
           label='Uniform expectation')
ax.axvline(5, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax.axvline(95, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax.fill_betweenx([0, ax.get_ylim()[1]], 0, 5, alpha=0.2, color='red')
ax.fill_betweenx([0, ax.get_ylim()[1]], 95, 100, alpha=0.2, color='red')
ax.set_xlabel('Percentile Rank', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Rank Distribution (Should be Uniform)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Observed vs Prior Predictive Mean
ax = axes[0, 1]
prior_pred_means = y_prior_pred.mean(axis=0)
prior_pred_std = y_prior_pred.std(axis=0)

ax.errorbar(range(n_obs), prior_pred_means, yerr=1.96*prior_pred_std,
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='steelblue', label='Prior predictive (95% CI)')
ax.scatter(range(n_obs), y_obs, color='red', s=100, marker='x',
           linewidths=3, label='Observed', zorder=5)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('y value', fontsize=12)
ax.set_title('Observed vs Prior Predictive Means', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_obs))

# Panel C: Q-Q plot
ax = axes[1, 0]
# For each observation, compute its z-score under prior predictive
z_scores = np.zeros(n_obs)
for j in range(n_obs):
    y_pred_j = y_prior_pred[:, j]
    z_scores[j] = (y_obs[j] - y_pred_j.mean()) / y_pred_j.std()

# Q-Q plot
stats.probplot(z_scores, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Standardized Residuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel D: Boxplots
ax = axes[1, 1]
# Box plot of prior predictive samples for each observation
bp = ax.boxplot([y_prior_pred[:, j] for j in range(n_obs)],
                 labels=[f'{i}' for i in range(n_obs)],
                 patch_artist=True, showfliers=False)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.6)

# Overlay observed values
ax.scatter(range(1, n_obs+1), y_obs, color='red', s=100, marker='x',
           linewidths=3, label='Observed', zorder=5)

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('y value', fontsize=12)
ax.set_title('Prior Predictive Distribution (Boxplots)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_data_compatibility.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/prior_data_compatibility.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS AND DECISION
# ============================================================================

print()
print("="*80)
print("PRIOR PREDICTIVE CHECK SUMMARY")
print("="*80)
print()

# Decision logic
issues = []

# Check 1: Hyperpriors reasonable?
prior_mu_95_width = np.percentile(mu_prior_samples, 97.5) - np.percentile(mu_prior_samples, 2.5)
if prior_mu_95_width < 20:
    issues.append("Prior for mu may be too narrow (95% width < 20)")
if prior_mu_95_width > 200:
    issues.append("Prior for mu may be too wide (95% width > 200)")

# Check 2: Tau prior appropriate?
tau_95 = np.percentile(tau_prior_samples, 95)
if tau_95 < 5:
    issues.append("Prior for tau may be too restrictive (95% < 5)")
if tau_95 > 30:
    issues.append("Prior for tau may be too wide (95% > 30)")

# Check 3: Observations in extreme tails?
if n_extreme_low > 0 or n_extreme_high > 0:
    issues.append(f"Some observations in extreme tails (< 5%: {n_extreme_low}, > 95%: {n_extreme_high})")

# Check 4: Computational issues?
if n_nan > 0 or n_inf > 0:
    issues.append(f"Numerical issues detected (NaN: {n_nan}, Inf: {n_inf})")

if max_abs_value > 1000:
    issues.append(f"Extreme values detected (max: {max_abs_value:.2f})")

# Decision
if len(issues) == 0:
    decision = "PASS"
    print("Decision: PASS")
    print()
    print("The hierarchical prior specification is appropriate:")
    print("  - Hyperpriors generate plausible parameter values")
    print("  - tau prior is regularizing but allows sufficient variation")
    print("  - Prior predictive distribution covers observed data")
    print("  - No observations in extreme tails")
    print("  - No computational issues")
    print()
    print("Recommendation: Proceed to Simulation-Based Calibration (SBC)")
else:
    decision = "REVIEW"
    print("Decision: REVIEW")
    print()
    print("Issues detected:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print()
    print("Recommendation: Review prior specification")

print()
print("="*80)

# Save summary statistics to file
summary_stats = {
    'decision': decision,
    'n_observations': n_obs,
    'n_prior_samples': n_prior_samples,
    'hyperpriors': {
        'mu_mean': prior_mu_mean,
        'mu_sd': prior_mu_sd,
        'tau_sd': prior_tau_sd,
        'mu_95_interval': [float(np.percentile(mu_prior_samples, 2.5)),
                           float(np.percentile(mu_prior_samples, 97.5))],
        'tau_median': float(np.median(tau_prior_samples)),
        'tau_95_interval': [float(np.percentile(tau_prior_samples, 2.5)),
                            float(np.percentile(tau_prior_samples, 97.5))]
    },
    'observed_data': {
        'mean': float(y_obs.mean()),
        'range': [float(y_obs.min()), float(y_obs.max())]
    },
    'group_parameters': {
        'theta_mean': float(theta_prior_samples.mean()),
        'theta_95_interval': [float(np.percentile(theta_prior_samples, 2.5)),
                              float(np.percentile(theta_prior_samples, 97.5))],
        'within_sample_sd_mean': float(within_sample_sds.mean()),
        'within_sample_sd_median': float(np.median(within_sample_sds))
    },
    'prior_predictive': {
        'percentile_ranks': percentile_ranks.tolist(),
        'n_extreme_low': int(n_extreme_low),
        'n_extreme_high': int(n_extreme_high),
        'mean': float(y_prior_pred.mean()),
        'std': float(y_prior_pred.std())
    },
    'computational': {
        'n_nan': int(n_nan),
        'n_inf': int(n_inf),
        'max_abs_value': float(max_abs_value)
    },
    'issues': issues
}

with open('/workspace/experiments/experiment_2/prior_predictive_check/diagnostics/summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("Saved summary statistics to: diagnostics/summary_stats.json")
print()
print("Analysis complete!")
