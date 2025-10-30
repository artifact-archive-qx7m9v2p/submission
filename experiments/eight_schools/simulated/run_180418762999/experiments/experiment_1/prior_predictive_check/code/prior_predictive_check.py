"""
Prior Predictive Check for Experiment 1: Complete Pooling Model

Model:
    mu ~ Normal(10, 20)
    y_i ~ Normal(mu, sigma_i)  [sigma_i known from data]

This script performs comprehensive prior predictive checks to validate:
1. Prior generates reasonable parameter values
2. Prior predictive distribution is scientifically plausible
3. Observed data is compatible with prior predictions
4. No computational issues (extreme values, numerical instability)
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
print("PRIOR PREDICTIVE CHECK: Experiment 1 - Complete Pooling Model")
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

print("Prior Specification:")
print(f"  mu ~ Normal({prior_mu_mean}, {prior_mu_sd})")
print()

# ============================================================================
# 1. SAMPLE FROM PRIOR DISTRIBUTION
# ============================================================================

print("Step 1: Sampling from Prior Distribution")
print("-"*80)

n_prior_samples = 5000
mu_prior_samples = np.random.normal(prior_mu_mean, prior_mu_sd, n_prior_samples)

print(f"  Sampled {n_prior_samples} values from prior")
print(f"  Prior samples - mean: {mu_prior_samples.mean():.2f}")
print(f"  Prior samples - std: {mu_prior_samples.std():.2f}")
print(f"  Prior samples - 95% interval: [{np.percentile(mu_prior_samples, 2.5):.2f}, {np.percentile(mu_prior_samples, 97.5):.2f}]")
print(f"  Prior samples - range: [{mu_prior_samples.min():.2f}, {mu_prior_samples.max():.2f}]")
print()

# Check for extreme values
extreme_threshold = 100
n_extreme = np.sum(np.abs(mu_prior_samples) > extreme_threshold)
print(f"  Extreme values check (|mu| > {extreme_threshold}): {n_extreme} samples ({100*n_extreme/n_prior_samples:.2f}%)")
print()

# ============================================================================
# 2. GENERATE PRIOR PREDICTIVE SAMPLES
# ============================================================================

print("Step 2: Generating Prior Predictive Samples")
print("-"*80)

# For each prior sample of mu, generate predicted y values for all 8 observations
y_prior_pred = np.zeros((n_prior_samples, n_obs))

for i in range(n_prior_samples):
    mu_i = mu_prior_samples[i]
    # Generate y_pred ~ N(mu_i, sigma_j) for each observation j
    for j in range(n_obs):
        y_prior_pred[i, j] = np.random.normal(mu_i, sigma_obs[j])

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
# 3. PRIOR-DATA COMPATIBILITY CHECKS
# ============================================================================

print("Step 3: Prior-Data Compatibility Analysis")
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
# 4. COMPUTATIONAL DIAGNOSTICS
# ============================================================================

print("Step 4: Computational Diagnostics")
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
# VISUALIZATION 1: PRIOR DISTRIBUTION FOR MU
# ============================================================================

print("Creating Visualization 1: Prior Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Prior distribution histogram
ax = axes[0]
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
ax.set_title('Prior Distribution: mu ~ N(10, 20)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Panel B: Prior quantiles
ax = axes[1]
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
quantiles = np.percentile(mu_prior_samples, percentiles)

ax.plot(percentiles, quantiles, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.axhline(y_obs.mean(), color='green', linestyle='--', linewidth=2,
           label=f'Observed mean ({y_obs.mean():.1f})')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

# Shade reasonable range
ax.fill_between([0, 100], -50, 50, alpha=0.1, color='green', label='Plausible range')

ax.set_xlabel('Percentile', fontsize=12)
ax.set_ylabel('mu value', fontsize=12)
ax.set_title('Prior Quantiles for mu', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/parameter_plausibility.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/parameter_plausibility.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: PRIOR PREDICTIVE OVERLAY
# ============================================================================

print("Creating Visualization 2: Prior Predictive Coverage")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(n_obs):
    ax = axes[j]

    # Prior predictive samples for this observation
    y_pred_j = y_prior_pred[:, j]

    # Histogram of prior predictive
    ax.hist(y_pred_j, bins=50, density=True, alpha=0.5, color='lightblue',
            edgecolor='black', linewidth=0.5, label='Prior predictive')

    # Overlay theoretical density: y_j ~ N(10, sqrt(20^2 + sigma_j^2))
    # Prior predictive: integrate over mu ~ N(10, 20)
    # y_j | mu ~ N(mu, sigma_j), so y_j ~ N(10, sqrt(20^2 + sigma_j^2))
    prior_pred_mean = prior_mu_mean
    prior_pred_sd = np.sqrt(prior_mu_sd**2 + sigma_obs[j]**2)
    x_range_j = np.linspace(y_pred_j.min(), y_pred_j.max(), 300)
    prior_pred_density = stats.norm.pdf(x_range_j, prior_pred_mean, prior_pred_sd)
    ax.plot(x_range_j, prior_pred_density, 'b-', linewidth=2, alpha=0.7,
            label='Theoretical')

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
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/prior_predictive_coverage.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/prior_predictive_coverage.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: PRIOR PREDICTIVE RANK STATISTICS
# ============================================================================

print("Creating Visualization 3: Prior Predictive Ranks")

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

# Panel B: Individual ranks
ax = axes[0, 1]
colors = ['red' if (pct < 5 or pct > 95) else 'green' for pct in percentile_ranks]
ax.barh(range(n_obs), percentile_ranks, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='5th/95th percentile')
ax.axvline(95, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(50, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Median')
ax.set_xlabel('Percentile Rank', fontsize=12)
ax.set_ylabel('Observation', fontsize=12)
ax.set_title('Percentile Rank per Observation', fontsize=13, fontweight='bold')
ax.set_yticks(range(n_obs))
ax.set_yticklabels([f'Obs {i}' for i in range(n_obs)])
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, 100)

# Panel C: Observed vs Prior Predictive Mean
ax = axes[1, 0]
prior_pred_means = y_prior_pred.mean(axis=0)
prior_pred_std = y_prior_pred.std(axis=0)

ax.errorbar(range(n_obs), prior_pred_means, yerr=1.96*prior_pred_std,
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='steelblue', label='Prior predictive (95% CI)')
ax.scatter(range(n_obs), y_obs, color='red', s=100, marker='x',
           linewidths=3, label='Observed', zorder=5)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Observation', fontsize=12)
ax.set_ylabel('y value', fontsize=12)
ax.set_title('Observed vs Prior Predictive Means', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_obs))

# Panel D: Q-Q plot
ax = axes[1, 1]
# For each observation, compute its z-score under prior predictive
z_scores = np.zeros(n_obs)
for j in range(n_obs):
    y_pred_j = y_prior_pred[:, j]
    z_scores[j] = (y_obs[j] - y_pred_j.mean()) / y_pred_j.std()

# Q-Q plot
stats.probplot(z_scores, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Standardized Residuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/prior_data_compatibility.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/prior_data_compatibility.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: JOINT PRIOR PREDICTIVE BEHAVIOR
# ============================================================================

print("Creating Visualization 4: Joint Prior Predictive Analysis")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Prior predictive sample trajectories
ax = axes[0, 0]
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

ax.set_xlabel('Observation', fontsize=12)
ax.set_ylabel('y value', fontsize=12)
ax.set_title('Prior Predictive Trajectories', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_obs))

# Panel B: Prior vs Prior Predictive spread
ax = axes[0, 1]
# Compare prior spread (mu) to prior predictive spread (y)
prior_pred_widths = []
for j in range(n_obs):
    width = np.percentile(y_prior_pred[:, j], 97.5) - np.percentile(y_prior_pred[:, j], 2.5)
    prior_pred_widths.append(width)

# Prior width for mu
prior_mu_width = np.percentile(mu_prior_samples, 97.5) - np.percentile(mu_prior_samples, 2.5)

x = range(n_obs)
ax.bar(x, prior_pred_widths, alpha=0.6, color='steelblue', edgecolor='black',
       label='Prior predictive (95% width)')
ax.axhline(prior_mu_width, color='red', linestyle='--', linewidth=2,
           label=f'Prior mu width ({prior_mu_width:.1f})')

# Add measurement error contribution
for j in range(n_obs):
    # Prior predictive variance = prior_mu_var + sigma_j^2
    # Width approximately 3.92 * sqrt(prior_mu_var + sigma_j^2)
    theoretical_width = 3.92 * np.sqrt(prior_mu_sd**2 + sigma_obs[j]**2)
    ax.scatter(j, theoretical_width, color='orange', s=100, marker='x',
               linewidths=2, zorder=5)

ax.scatter([], [], color='orange', s=100, marker='x', linewidths=2,
           label='Theoretical width')

ax.set_xlabel('Observation', fontsize=12)
ax.set_ylabel('95% Interval Width', fontsize=12)
ax.set_title('Prior vs Prior Predictive Spread', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_obs))

# Panel C: Measurement error contribution
ax = axes[1, 0]
# Decompose prior predictive variance into prior and likelihood components
prior_var = prior_mu_sd**2
likelihood_vars = sigma_obs**2
total_vars = prior_var + likelihood_vars

# Stacked bar chart
prior_contribution = prior_var / total_vars * 100
likelihood_contribution = likelihood_vars / total_vars * 100

x = range(n_obs)
ax.bar(x, prior_contribution, label='Prior variance', alpha=0.7, color='red',
       edgecolor='black')
ax.bar(x, likelihood_contribution, bottom=prior_contribution,
       label='Measurement error variance', alpha=0.7, color='steelblue',
       edgecolor='black')

ax.set_xlabel('Observation', fontsize=12)
ax.set_ylabel('Variance Contribution (%)', fontsize=12)
ax.set_title('Prior vs Likelihood Variance Contribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(n_obs))
ax.set_ylim(0, 100)

# Add sigma values on x-axis
ax.set_xticklabels([f'{i}\n(σ={sigma_obs[i]:.0f})' for i in range(n_obs)], fontsize=9)

# Panel D: Extreme values check
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

ax.set_xlabel('Observation', fontsize=12)
ax.set_ylabel('y value', fontsize=12)
ax.set_title('Prior Predictive Distribution (Boxplots)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/joint_prior_behavior.png',
            dpi=300, bbox_inches='tight')
print("  Saved: plots/joint_prior_behavior.png")
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

# Check 1: Prior too narrow?
prior_95_width = prior_mu_width
if prior_95_width < 20:
    issues.append("Prior may be too narrow (95% width < 20)")

# Check 2: Prior too wide?
if prior_95_width > 200:
    issues.append("Prior may be too wide (95% width > 200)")

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
    print("The prior specification is appropriate:")
    print("  - Prior generates plausible parameter values")
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
    'prior_mu_mean': prior_mu_mean,
    'prior_mu_sd': prior_mu_sd,
    'prior_95_interval': [float(np.percentile(mu_prior_samples, 2.5)),
                          float(np.percentile(mu_prior_samples, 97.5))],
    'observed_mean': float(y_obs.mean()),
    'observed_range': [float(y_obs.min()), float(y_obs.max())],
    'percentile_ranks': percentile_ranks.tolist(),
    'n_extreme_low': int(n_extreme_low),
    'n_extreme_high': int(n_extreme_high),
    'n_nan': int(n_nan),
    'n_inf': int(n_inf),
    'max_abs_value': float(max_abs_value),
    'issues': issues
}

with open('/workspace/experiments/experiment_1/prior_predictive_check/diagnostics/summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("Saved summary statistics to: diagnostics/summary_stats.json")
print()
print("Analysis complete!")
