"""
Prior Predictive Check for Fixed-Effect Meta-Analysis Model

This script performs comprehensive prior predictive checks to validate
the model specification before fitting:

1. Prior parameter distribution analysis
2. Prior predictive distributions for each observation
3. Prior-data conflict assessment
4. Prior sensitivity analysis
5. Scientific plausibility evaluation

Model:
    y_i | θ, σ_i ~ Normal(θ, σ_i²)
    θ ~ Normal(0, 20²)
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data
y_obs = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
n_obs = len(y_obs)

# EDA pooled estimate for reference
theta_mle = 7.686
theta_se = 4.072

print("="*70)
print("PRIOR PREDICTIVE CHECK: Fixed-Effect Meta-Analysis")
print("="*70)
print(f"\nObserved data: {y_obs}")
print(f"Known σ: {sigma}")
print(f"EDA pooled estimate: θ = {theta_mle:.3f} ± {theta_se:.3f}")
print(f"Number of observations: {n_obs}")

# ============================================================================
# 1. PRIOR PARAMETER DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("1. PRIOR PARAMETER DISTRIBUTION ANALYSIS")
print("="*70)

# Prior specification
prior_mean = 0
prior_sd = 20

# Sample from prior
n_prior_samples = 10000
theta_prior = np.random.normal(prior_mean, prior_sd, n_prior_samples)

# Prior statistics
print(f"\nPrior: θ ~ Normal({prior_mean}, {prior_sd}²)")
print(f"Prior 95% interval: [{prior_mean - 1.96*prior_sd:.1f}, {prior_mean + 1.96*prior_sd:.1f}]")
print(f"Prior mean: {np.mean(theta_prior):.3f}")
print(f"Prior SD: {np.std(theta_prior):.3f}")

# Check if observed estimate is in prior support
prior_percentile = stats.norm.cdf(theta_mle, prior_mean, prior_sd) * 100
print(f"\nObserved θ_MLE = {theta_mle:.3f} is at {prior_percentile:.1f}th percentile of prior")
print(f"Prior density at θ_MLE: {stats.norm.pdf(theta_mle, prior_mean, prior_sd):.6f}")

# Prior coverage check
within_1sd = np.abs(theta_prior) <= prior_sd
within_2sd = np.abs(theta_prior) <= 2*prior_sd
print(f"\nPrior samples within 1 SD: {np.mean(within_1sd)*100:.1f}%")
print(f"Prior samples within 2 SD: {np.mean(within_2sd)*100:.1f}%")

# ============================================================================
# 2. PRIOR PREDICTIVE DISTRIBUTION
# ============================================================================

print("\n" + "="*70)
print("2. PRIOR PREDICTIVE DISTRIBUTION")
print("="*70)

# Generate prior predictive samples using PyMC
with pm.Model() as model_prior_pred:
    # Prior
    theta = pm.Normal("theta", mu=prior_mean, sigma=prior_sd)

    # Likelihood (prior predictive)
    y = pm.Normal("y", mu=theta, sigma=sigma, shape=n_obs)

    # Sample from prior predictive
    prior_predictive = pm.sample_prior_predictive(samples=n_prior_samples, random_seed=42)

# Extract prior predictive samples
theta_samples = prior_predictive.prior["theta"].values.flatten()
y_pred_samples = prior_predictive.prior_predictive["y"].values.reshape(n_prior_samples, n_obs)

print(f"\nGenerated {n_prior_samples} prior predictive samples")
print(f"Theta samples shape: {theta_samples.shape}")
print(f"Y predictive samples shape: {y_pred_samples.shape}")

# Prior predictive statistics for each observation
print("\nPrior predictive statistics by observation:")
print(f"{'Obs':<5} {'y_obs':<8} {'σ':<6} {'y_pred mean':<12} {'y_pred SD':<12} {'Z-score':<10}")
print("-"*70)

z_scores = []
prior_pred_pvals = []

for i in range(n_obs):
    y_pred_i = y_pred_samples[:, i]
    pred_mean = np.mean(y_pred_i)
    pred_sd = np.std(y_pred_i)

    # Z-score: how many SDs is observed value from prior predictive mean?
    z_score = (y_obs[i] - pred_mean) / pred_sd
    z_scores.append(z_score)

    # Prior predictive p-value (two-tailed)
    p_val = 2 * min(np.mean(y_pred_i <= y_obs[i]), np.mean(y_pred_i >= y_obs[i]))
    prior_pred_pvals.append(p_val)

    print(f"{i+1:<5} {y_obs[i]:<8.1f} {sigma[i]:<6.1f} {pred_mean:<12.2f} {pred_sd:<12.2f} {z_score:<10.3f}")

print(f"\nZ-score summary:")
print(f"  Mean |Z|: {np.mean(np.abs(z_scores)):.3f}")
print(f"  Max |Z|: {np.max(np.abs(z_scores)):.3f}")
print(f"  N with |Z| > 2: {np.sum(np.abs(z_scores) > 2)}/{n_obs}")
print(f"  N with |Z| > 3: {np.sum(np.abs(z_scores) > 3)}/{n_obs}")

# ============================================================================
# 3. PRIOR-DATA CONFLICT CHECK
# ============================================================================

print("\n" + "="*70)
print("3. PRIOR-DATA CONFLICT ASSESSMENT")
print("="*70)

print("\nPrior predictive p-values (two-tailed):")
for i, p_val in enumerate(prior_pred_pvals):
    flag = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "")
    print(f"  Observation {i+1}: p = {p_val:.4f} {flag}")

min_pval = np.min(prior_pred_pvals)
print(f"\nMinimum p-value: {min_pval:.4f}")
print(f"N with p < 0.05: {np.sum(np.array(prior_pred_pvals) < 0.05)}/{n_obs}")
print(f"N with p < 0.01: {np.sum(np.array(prior_pred_pvals) < 0.01)}/{n_obs}")

# Check if observed data is within reasonable prior predictive range
# Use 95% prior predictive interval
print("\nPrior predictive coverage:")
for i in range(n_obs):
    y_pred_i = y_pred_samples[:, i]
    lower = np.percentile(y_pred_i, 2.5)
    upper = np.percentile(y_pred_i, 97.5)
    within = lower <= y_obs[i] <= upper
    status = "✓" if within else "✗"
    print(f"  Obs {i+1}: [{lower:6.1f}, {upper:6.1f}] - y_obs = {y_obs[i]:6.1f} {status}")

coverage = np.mean([
    np.percentile(y_pred_samples[:, i], 2.5) <= y_obs[i] <= np.percentile(y_pred_samples[:, i], 97.5)
    for i in range(n_obs)
])
print(f"\n95% prior predictive coverage: {coverage*100:.1f}%")

# ============================================================================
# 4. PRIOR SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("4. PRIOR SENSITIVITY ANALYSIS")
print("="*70)

# Test alternative prior scales
prior_scales = {
    "Tight": 10,
    "Default": 20,
    "Vague": 50
}

sensitivity_results = {}

for prior_name, prior_scale in prior_scales.items():
    print(f"\n--- Prior: θ ~ Normal(0, {prior_scale}²) ---")

    # Sample from prior predictive
    theta_sens = np.random.normal(0, prior_scale, n_prior_samples)
    y_pred_sens = np.array([
        np.random.normal(theta_sens, sigma[i])
        for i in range(n_obs)
    ]).T

    # Compute coverage
    coverage_sens = np.mean([
        np.percentile(y_pred_sens[:, i], 2.5) <= y_obs[i] <= np.percentile(y_pred_sens[:, i], 97.5)
        for i in range(n_obs)
    ])

    # Compute z-scores
    z_scores_sens = [
        (y_obs[i] - np.mean(y_pred_sens[:, i])) / np.std(y_pred_sens[:, i])
        for i in range(n_obs)
    ]
    mean_abs_z = np.mean(np.abs(z_scores_sens))

    # Prior predictive range
    pp_range = [np.percentile(y_pred_sens, 2.5), np.percentile(y_pred_sens, 97.5)]

    sensitivity_results[prior_name] = {
        "scale": prior_scale,
        "coverage": coverage_sens,
        "mean_abs_z": mean_abs_z,
        "pp_range": pp_range,
        "theta_samples": theta_sens,
        "y_pred_samples": y_pred_sens
    }

    print(f"  95% PP coverage: {coverage_sens*100:.1f}%")
    print(f"  Mean |Z|: {mean_abs_z:.3f}")
    print(f"  95% PP range: [{pp_range[0]:.1f}, {pp_range[1]:.1f}]")
    print(f"  Data range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")

# ============================================================================
# 5. SCIENTIFIC PLAUSIBILITY CHECK
# ============================================================================

print("\n" + "="*70)
print("5. SCIENTIFIC PLAUSIBILITY ASSESSMENT")
print("="*70)

print(f"\nPrior allows (95% probability):")
print(f"  θ ∈ [{prior_mean - 1.96*prior_sd:.1f}, {prior_mean + 1.96*prior_sd:.1f}]")
print(f"  θ ∈ [-39.2, 39.2]")

print(f"\nExtreme scenarios under prior:")
extreme_percentiles = [1, 5, 95, 99]
for p in extreme_percentiles:
    theta_extreme = np.percentile(theta_prior, p)
    print(f"  {p}th percentile: θ = {theta_extreme:.1f}")

print(f"\nPrior probability of extreme values:")
extreme_values = [-50, -30, 30, 50]
for val in extreme_values:
    prob = stats.norm.cdf(val, prior_mean, prior_sd)
    if val < 0:
        prob_extreme = prob
        print(f"  P(θ < {val}) = {prob_extreme:.4f}")
    else:
        prob_extreme = 1 - prob
        print(f"  P(θ > {val}) = {prob_extreme:.4f}")

# Prior predictive range
y_pred_all = y_pred_samples.flatten()
print(f"\nPrior predictive distribution (all observations):")
print(f"  95% interval: [{np.percentile(y_pred_all, 2.5):.1f}, {np.percentile(y_pred_all, 97.5):.1f}]")
print(f"  99% interval: [{np.percentile(y_pred_all, 0.5):.1f}, {np.percentile(y_pred_all, 99.5):.1f}]")
print(f"  Observed range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Plot 1: Prior parameter distribution with observed estimate
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(theta_prior, bins=50, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', label='Prior samples')

# Overlay theoretical prior density
x = np.linspace(-80, 80, 1000)
ax.plot(x, stats.norm.pdf(x, prior_mean, prior_sd), 'k-', lw=2,
        label=f'Prior: N(0, 20²)')

# Add observed estimate
ax.axvline(theta_mle, color='red', linestyle='--', lw=2,
           label=f'EDA estimate: θ = {theta_mle:.2f}')
ax.axvspan(theta_mle - 1.96*theta_se, theta_mle + 1.96*theta_se,
           alpha=0.2, color='red', label='EDA 95% CI')

# Add reference lines
ax.axvline(0, color='gray', linestyle=':', lw=1, alpha=0.5)
ax.axvspan(-1.96*prior_sd, 1.96*prior_sd, alpha=0.1, color='steelblue')

ax.set_xlabel('θ (effect parameter)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution for θ vs. Observed Estimate', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-80, 80)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_plausibility.png", dpi=300, bbox_inches='tight')
print(f"Saved: parameter_plausibility.png")
plt.close()

# Plot 2: Prior predictive distributions for each observation
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(n_obs):
    ax = axes[i]
    y_pred_i = y_pred_samples[:, i]

    # Histogram of prior predictive
    ax.hist(y_pred_i, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', label='Prior predictive')

    # Overlay theoretical density (convolution of prior and likelihood)
    # p(y_i) = N(0, sqrt(20² + σ_i²))
    pp_sd = np.sqrt(prior_sd**2 + sigma[i]**2)
    x = np.linspace(y_pred_i.min(), y_pred_i.max(), 200)
    ax.plot(x, stats.norm.pdf(x, 0, pp_sd), 'k-', lw=2, alpha=0.8)

    # Add observed value
    ax.axvline(y_obs[i], color='red', linestyle='--', lw=2, label=f'y_obs = {y_obs[i]}')

    # Add 95% prior predictive interval
    lower = np.percentile(y_pred_i, 2.5)
    upper = np.percentile(y_pred_i, 97.5)
    ax.axvspan(lower, upper, alpha=0.15, color='steelblue')

    # Z-score annotation
    z = z_scores[i]
    color = 'red' if np.abs(z) > 2 else 'green'
    ax.text(0.05, 0.95, f'Z = {z:.2f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax.set_xlabel(f'y_{i+1}', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Obs {i+1}: σ = {sigma[i]}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if i == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.suptitle('Prior Predictive Distributions by Observation',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_by_observation.png", dpi=300, bbox_inches='tight')
print(f"Saved: prior_predictive_by_observation.png")
plt.close()

# Plot 3: Prior-data conflict diagnostic
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Z-scores
ax = axes[0]
x_pos = np.arange(1, n_obs + 1)
colors = ['red' if np.abs(z) > 2 else 'steelblue' for z in z_scores]
bars = ax.bar(x_pos, z_scores, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', lw=1)
ax.axhline(2, color='red', linestyle='--', lw=1, alpha=0.5, label='|Z| = 2')
ax.axhline(-2, color='red', linestyle='--', lw=1, alpha=0.5)
ax.axhspan(-2, 2, alpha=0.1, color='green', label='Expected range')
ax.set_xlabel('Observation', fontsize=11)
ax.set_ylabel('Z-score', fontsize=11)
ax.set_title('Prior Predictive Z-scores', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper right', fontsize=9)

# Panel B: Prior predictive p-values
ax = axes[1]
bars = ax.bar(x_pos, prior_pred_pvals, color='steelblue', alpha=0.7, edgecolor='black')
ax.axhline(0.05, color='orange', linestyle='--', lw=1.5, label='p = 0.05')
ax.axhline(0.01, color='red', linestyle='--', lw=1.5, label='p = 0.01')
ax.set_xlabel('Observation', fontsize=11)
ax.set_ylabel('Prior predictive p-value', fontsize=11)
ax.set_title('Prior-Data Conflict Test', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper right', fontsize=9)

# Panel C: Coverage diagnostic
ax = axes[2]
for i in range(n_obs):
    y_pred_i = y_pred_samples[:, i]
    lower = np.percentile(y_pred_i, 2.5)
    upper = np.percentile(y_pred_i, 97.5)
    median = np.percentile(y_pred_i, 50)

    # Draw interval
    ax.plot([i+1, i+1], [lower, upper], 'o-', color='steelblue',
            markersize=4, lw=2, alpha=0.7)
    ax.plot(i+1, median, 'o', color='darkblue', markersize=6)

    # Draw observed
    color = 'green' if lower <= y_obs[i] <= upper else 'red'
    ax.plot(i+1, y_obs[i], 'D', color=color, markersize=8,
            markeredgecolor='black', markeredgewidth=1)

ax.axhline(0, color='gray', linestyle=':', lw=1, alpha=0.5)
ax.set_xlabel('Observation', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('95% Prior Predictive Intervals vs. Data', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.grid(True, alpha=0.3)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='steelblue', lw=2, label='95% PP interval',
           markersize=6, markerfacecolor='steelblue'),
    Line2D([0], [0], marker='D', color='green', lw=0, label='y_obs (covered)',
           markersize=8, markerfacecolor='green', markeredgecolor='black'),
    Line2D([0], [0], marker='D', color='red', lw=0, label='y_obs (not covered)',
           markersize=8, markerfacecolor='red', markeredgecolor='black')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_data_conflict_diagnostic.png", dpi=300, bbox_inches='tight')
print(f"Saved: prior_data_conflict_diagnostic.png")
plt.close()

# Plot 4: Prior sensitivity analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Prior distributions
for idx, (prior_name, prior_scale) in enumerate(prior_scales.items()):
    ax = axes[0, idx]
    theta_sens = sensitivity_results[prior_name]["theta_samples"]

    ax.hist(theta_sens, bins=50, density=True, alpha=0.6,
            color='steelblue', edgecolor='black')

    x = np.linspace(-150, 150, 1000)
    ax.plot(x, stats.norm.pdf(x, 0, prior_scale), 'k-', lw=2)

    ax.axvline(theta_mle, color='red', linestyle='--', lw=2,
               label=f'θ_MLE = {theta_mle:.2f}')
    ax.axvspan(-1.96*prior_scale, 1.96*prior_scale, alpha=0.1, color='steelblue')

    ax.set_xlabel('θ', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{prior_name}: N(0, {prior_scale}²)', fontsize=11, fontweight='bold')
    ax.set_xlim(-150, 150)
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.legend(loc='upper right', fontsize=9)

# Row 2: Prior predictive distributions
for idx, (prior_name, prior_scale) in enumerate(prior_scales.items()):
    ax = axes[1, idx]
    y_pred_sens = sensitivity_results[prior_name]["y_pred_samples"].flatten()

    ax.hist(y_pred_sens, bins=100, density=True, alpha=0.6,
            color='steelblue', edgecolor='black', label='Prior predictive')

    # Add observed data points
    for y in y_obs:
        ax.axvline(y, color='red', linestyle='-', lw=1, alpha=0.3)

    # Highlight range
    pp_range = sensitivity_results[prior_name]["pp_range"]
    ax.axvspan(pp_range[0], pp_range[1], alpha=0.1, color='steelblue',
               label='95% PP interval')

    coverage = sensitivity_results[prior_name]["coverage"]
    mean_z = sensitivity_results[prior_name]["mean_abs_z"]

    ax.text(0.05, 0.95, f'Coverage: {coverage*100:.0f}%\nMean |Z|: {mean_z:.2f}',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('y (all observations)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Prior Predictive: {prior_name}', fontsize=11, fontweight='bold')
    ax.set_xlim(-150, 150)
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Prior Sensitivity Analysis', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
print(f"Saved: prior_sensitivity_analysis.png")
plt.close()

# Plot 5: Scientific plausibility - comprehensive overview
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Prior distribution with percentiles
ax1 = fig.add_subplot(gs[0, :])
ax1.hist(theta_prior, bins=100, density=True, alpha=0.6,
         color='steelblue', edgecolor='black', label='Prior samples')

x = np.linspace(-100, 100, 1000)
ax1.plot(x, stats.norm.pdf(x, prior_mean, prior_sd), 'k-', lw=2,
         label='Prior: N(0, 20²)')

# Add percentile markers
percentiles = [1, 5, 25, 50, 75, 95, 99]
for p in percentiles:
    val = np.percentile(theta_prior, p)
    ax1.axvline(val, color='orange', linestyle=':', lw=1, alpha=0.6)
    ax1.text(val, ax1.get_ylim()[1]*0.95, f'{p}%',
             ha='center', fontsize=8, color='orange')

ax1.axvline(theta_mle, color='red', linestyle='--', lw=2,
            label=f'EDA: θ = {theta_mle:.2f}')
ax1.axvspan(-39.2, 39.2, alpha=0.1, color='steelblue', label='95% prior mass')

ax1.set_xlabel('θ (effect parameter)', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Prior Distribution: Scientific Plausibility Range',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-100, 100)

# Panel 2: Relationship between θ and measurement uncertainty
ax2 = fig.add_subplot(gs[1, 0])
for i in range(n_obs):
    # Sample some prior predictive draws
    n_draws = 200
    theta_draws = np.random.choice(theta_prior, n_draws)
    y_draws = theta_draws + np.random.normal(0, sigma[i], n_draws)

    ax2.scatter(theta_draws, y_draws, alpha=0.1, s=10, label=f'σ={sigma[i]}' if i < 3 else '')

ax2.axhline(0, color='gray', linestyle=':', lw=1, alpha=0.5)
ax2.axvline(0, color='gray', linestyle=':', lw=1, alpha=0.5)
ax2.axvline(theta_mle, color='red', linestyle='--', lw=2, alpha=0.7)

# Add observed data
ax2.scatter(np.full(n_obs, theta_mle), y_obs, color='red', s=100,
            marker='D', edgecolor='black', linewidth=1.5,
            label='Observed data', zorder=5)

ax2.set_xlabel('θ (true effect)', fontsize=10)
ax2.set_ylabel('y (observed value)', fontsize=10)
ax2.set_title('Prior Predictive: θ vs y', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-100, 100)
ax2.set_ylim(-100, 100)

# Panel 3: Distribution of measurement uncertainties
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(range(1, n_obs+1), sigma, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axhline(np.mean(sigma), color='red', linestyle='--', lw=2,
            label=f'Mean σ = {np.mean(sigma):.1f}')
ax3.set_xlabel('Observation', fontsize=10)
ax3.set_ylabel('σ (measurement SD)', fontsize=10)
ax3.set_title('Known Measurement Uncertainties', fontsize=11, fontweight='bold')
ax3.set_xticks(range(1, n_obs+1))
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Prior predictive SD by observation
ax4 = fig.add_subplot(gs[1, 2])
pp_sds = [np.sqrt(prior_sd**2 + sigma[i]**2) for i in range(n_obs)]
ax4.bar(range(1, n_obs+1), pp_sds, color='purple', alpha=0.7, edgecolor='black')
ax4.axhline(prior_sd, color='blue', linestyle='--', lw=2,
            label=f'Prior SD = {prior_sd}')
ax4.axhline(np.mean(sigma), color='orange', linestyle='--', lw=2,
            label=f'Mean meas. SD = {np.mean(sigma):.1f}')
ax4.set_xlabel('Observation', fontsize=10)
ax4.set_ylabel('Prior predictive SD', fontsize=10)
ax4.set_title('Prior Predictive Uncertainty', fontsize=11, fontweight='bold')
ax4.set_xticks(range(1, n_obs+1))
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Panel 5: Overall prior predictive distribution
ax5 = fig.add_subplot(gs[2, :2])
y_pred_all = y_pred_samples.flatten()
ax5.hist(y_pred_all, bins=100, density=True, alpha=0.6,
         color='steelblue', edgecolor='black', label='Prior predictive (all obs)')

# Add observed data points
for i, y in enumerate(y_obs):
    ax5.axvline(y, color='red', linestyle='-', lw=2, alpha=0.6)
    ax5.text(y, ax5.get_ylim()[1]*0.98, f'{i+1}', ha='center',
             fontsize=9, color='red', fontweight='bold')

# Add percentiles
pp_percentiles = [2.5, 25, 50, 75, 97.5]
for p in pp_percentiles:
    val = np.percentile(y_pred_all, p)
    ax5.axvline(val, color='green', linestyle=':', lw=1, alpha=0.4)

ax5.axvspan(np.percentile(y_pred_all, 2.5), np.percentile(y_pred_all, 97.5),
            alpha=0.1, color='steelblue', label='95% PP interval')

ax5.set_xlabel('y (observed value)', fontsize=11)
ax5.set_ylabel('Density', fontsize=11)
ax5.set_title('Overall Prior Predictive Distribution vs. Observed Data',
              fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(-150, 150)

# Panel 6: Summary statistics table
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

summary_text = f"""
PRIOR SPECIFICATION SUMMARY

Prior for θ:
  N(0, 20²)
  95% interval: [-39.2, 39.2]

Prior Predictive (pooled):
  95% interval: [{np.percentile(y_pred_all, 2.5):.1f}, {np.percentile(y_pred_all, 97.5):.1f}]
  Mean: {np.mean(y_pred_all):.2f}
  SD: {np.std(y_pred_all):.2f}

Observed Data:
  Range: [{y_obs.min()}, {y_obs.max()}]
  Mean: {np.mean(y_obs):.2f}
  EDA estimate: {theta_mle:.2f} ± {theta_se:.2f}

Coverage:
  95% PP coverage: {coverage*100:.0f}%
  Mean |Z-score|: {np.mean(np.abs(z_scores)):.2f}
  Max |Z-score|: {np.max(np.abs(z_scores)):.2f}

Prior-Data Conflict:
  Min p-value: {min_pval:.4f}
  N with p < 0.05: {np.sum(np.array(prior_pred_pvals) < 0.05)}/{n_obs}
"""

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
         verticalalignment='top', fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Scientific Plausibility Assessment', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / "scientific_plausibility_overview.png", dpi=300, bbox_inches='tight')
print(f"Saved: scientific_plausibility_overview.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS FOR REPORT
# ============================================================================

print("\n" + "="*70)
print("SUMMARY FOR REPORT")
print("="*70)

summary_stats = {
    "prior_mean": prior_mean,
    "prior_sd": prior_sd,
    "prior_95_interval": [prior_mean - 1.96*prior_sd, prior_mean + 1.96*prior_sd],
    "theta_mle": theta_mle,
    "theta_se": theta_se,
    "theta_mle_prior_percentile": prior_percentile,
    "n_prior_samples": n_prior_samples,
    "pp_coverage_95": coverage * 100,
    "mean_abs_z_score": np.mean(np.abs(z_scores)),
    "max_abs_z_score": np.max(np.abs(z_scores)),
    "n_extreme_z": np.sum(np.abs(z_scores) > 2),
    "min_pp_pvalue": min_pval,
    "n_pp_pval_lt_05": np.sum(np.array(prior_pred_pvals) < 0.05),
    "n_pp_pval_lt_01": np.sum(np.array(prior_pred_pvals) < 0.01),
    "pp_range_95": [np.percentile(y_pred_all, 2.5), np.percentile(y_pred_all, 97.5)],
    "data_range": [y_obs.min(), y_obs.max()],
    "sensitivity": {
        prior_name: {
            "coverage": res["coverage"] * 100,
            "mean_abs_z": res["mean_abs_z"],
            "pp_range": res["pp_range"]
        }
        for prior_name, res in sensitivity_results.items()
    }
}

# Save summary statistics
import json
with open(OUTPUT_DIR / "code" / "summary_statistics.json", "w") as f:
    json.dump(summary_stats, f, indent=2)

print("\nKey metrics saved to summary_statistics.json")

print("\n" + "="*70)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("="*70)
print(f"\nGenerated {len(list(PLOTS_DIR.glob('*.png')))} diagnostic plots")
print(f"All outputs saved to: {OUTPUT_DIR}")
