"""
Prior Predictive Visualizations for Hierarchical Logit Model

Creates diagnostic plots to assess whether priors generate plausible data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
data = pd.read_csv('/workspace/data/data.csv')
n_obs = data['n'].values
r_obs = data['r'].values
total_successes_obs = r_obs.sum()
total_n_obs = n_obs.sum()

# Load prior predictive samples
samples_path = Path('/workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_samples.npz')
prior_pred = np.load(samples_path)

mu_logit = prior_pred['mu_logit']
sigma = prior_pred['sigma']
theta = prior_pred['theta']
r = prior_pred['r']
total_r = prior_pred['total_r']
proportions = prior_pred['proportions']

# Output directory
plots_dir = Path('/workspace/experiments/experiment_2/prior_predictive_check/plots')

print("=" * 70)
print("CREATING PRIOR PREDICTIVE VISUALIZATIONS")
print("=" * 70)


# ============================================================================
# PLOT 1: Parameter Plausibility (mu_logit and sigma distributions)
# ============================================================================
print("\n1. Creating parameter plausibility plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# mu_logit on logit scale
ax = axes[0, 0]
ax.hist(mu_logit, bins=50, density=True, alpha=0.7, edgecolor='black', color='steelblue')
x = np.linspace(mu_logit.min(), mu_logit.max(), 200)
ax.plot(x, stats.norm.pdf(x, -2.53, 1), 'r-', lw=2, label='Prior: N(-2.53, 1)')
ax.axvline(-2.53, color='red', linestyle='--', lw=2, label='Prior mean (logit(0.074))')
ax.axvline(np.log(0.0739/(1-0.0739)), color='orange', linestyle='--', lw=2,
           label=f'Observed pooled (logit(0.074))')
ax.set_xlabel('μ_logit (log-odds scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Population Mean: Logit Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# mu on probability scale
ax = axes[0, 1]
mu_prob = expit(mu_logit)
ax.hist(mu_prob, bins=50, density=True, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(0.074, color='red', linestyle='--', lw=2, label='Prior center (0.074)')
ax.axvline(0.0739, color='orange', linestyle='--', lw=2, label='Observed pooled')
ax.set_xlabel('μ_prob (probability scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Population Mean: Probability Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
# Add quantiles
q = np.quantile(mu_prob, [0.025, 0.975])
ax.axvspan(q[0], q[1], alpha=0.2, color='blue', label=f'95% Prior interval')

# sigma distribution
ax = axes[1, 0]
ax.hist(sigma, bins=50, density=True, alpha=0.7, edgecolor='black', color='seagreen')
x = np.linspace(0, sigma.max(), 200)
ax.plot(x, stats.halfnorm.pdf(x, 0, 1), 'r-', lw=2, label='Prior: HalfNormal(0, 1)')
ax.axvline(sigma.mean(), color='blue', linestyle='--', lw=2,
           label=f'Realized mean: {sigma.mean():.2f}')
ax.set_xlabel('σ (scale of heterogeneity)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Scale Parameter Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Interpretation guide for sigma
ax = axes[1, 1]
sigma_examples = [0.5, 1.0, 1.5, 2.0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_examples)))
for sig, col in zip(sigma_examples, colors):
    # For mu_logit = -2.53, show range of theta implied by different sigma
    eta_range = np.linspace(-2, 2, 100)
    theta_range = expit(-2.53 + sig * eta_range)
    ax.plot(eta_range, theta_range, lw=2, label=f'σ = {sig}', color=col)

ax.axhline(0.074, color='red', linestyle='--', alpha=0.5, label='μ_prob = 0.074')
ax.fill_between([-2, 2], 0, 0.3, alpha=0.1, color='blue', label='Plausible range')
ax.set_xlabel('η (standardized trial effect)', fontsize=11)
ax.set_ylabel('θ (trial probability)', fontsize=11)
ax.set_title('Heterogeneity Interpretation (μ_logit = -2.53)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig(plots_dir / 'parameter_plausibility.png')
print(f"   Saved: {plots_dir / 'parameter_plausibility.png'}")
plt.close()


# ============================================================================
# PLOT 2: Prior Predictive Coverage of Observed Data
# ============================================================================
print("\n2. Creating prior predictive coverage plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total successes
ax = axes[0, 0]
ax.hist(total_r, bins=60, density=True, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(total_successes_obs, color='red', linestyle='--', lw=3,
           label=f'Observed: {total_successes_obs}')
ax.axvline(total_r.mean(), color='blue', linestyle='--', lw=2,
           label=f'Prior mean: {total_r.mean():.0f}')
q = np.quantile(total_r, [0.025, 0.975])
ax.axvspan(q[0], q[1], alpha=0.2, color='blue', label='95% interval')
ax.set_xlabel('Total successes (across all trials)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive: Total Successes', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
percentile = stats.percentileofscore(total_r, total_successes_obs)
ax.text(0.05, 0.95, f'Observed at {percentile:.1f}th percentile',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Distribution of trial proportions
ax = axes[0, 1]
# Plot prior predictive proportions
for i in range(12):
    ax.hist(proportions[:, i], bins=30, alpha=0.3, density=True, color='gray')
# Overall distribution
ax.hist(proportions.flatten(), bins=50, alpha=0.7, density=True,
        edgecolor='black', color='steelblue', label='Prior predictive')
# Overlay observed
for i, (trial_id, prop) in enumerate(zip(data['trial_id'], data['proportion'])):
    marker = 'o' if i not in [0, 7] else 's'  # Square for extreme trials
    color = 'red' if i in [0, 7] else 'orange'
    size = 100 if i in [0, 7] else 60
    ax.scatter(prop, 0, marker=marker, s=size, color=color,
              edgecolor='black', linewidth=1.5, zorder=10)

ax.scatter([], [], marker='s', s=100, color='red', edgecolor='black',
          linewidth=1.5, label='Extreme trials (1, 8)')
ax.scatter([], [], marker='o', s=60, color='orange', edgecolor='black',
          linewidth=1.5, label='Other observed trials')
ax.set_xlabel('Success proportion (r/n)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive: Trial Proportions', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 0.5)

# Q-Q plot: prior predictive proportions vs observed
ax = axes[1, 0]
# For each trial, compare observed to prior predictive quantile
trial_percentiles = []
for i in range(12):
    # Use actual count for better comparison
    percentile = stats.percentileofscore(r[:, i], r_obs[i])
    trial_percentiles.append(percentile)

trial_percentiles = np.array(trial_percentiles) / 100  # Convert to [0, 1]
expected_quantiles = np.linspace(0, 1, 12)

ax.scatter(expected_quantiles, sorted(trial_percentiles), s=100, alpha=0.7,
          edgecolor='black', linewidth=1.5, color='steelblue')
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.2, color='red')
ax.set_xlabel('Expected quantile (uniform)', fontsize=11)
ax.set_ylabel('Observed quantile (prior predictive)', fontsize=11)
ax.set_title('Calibration: Observed vs Prior Predictive', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.text(0.05, 0.95,
        'Points near line indicate\ngood prior calibration',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Coverage by trial
ax = axes[1, 1]
coverage_in_50 = []
coverage_in_95 = []
for i in range(12):
    q50 = np.quantile(r[:, i], [0.25, 0.75])
    q95 = np.quantile(r[:, i], [0.025, 0.975])
    coverage_in_50.append(q50[0] <= r_obs[i] <= q50[1])
    coverage_in_95.append(q95[0] <= r_obs[i] <= q95[1])

x = np.arange(12) + 1
width = 0.35
ax.bar(x - width/2, coverage_in_50, width, label='50% interval',
       color='steelblue', alpha=0.7, edgecolor='black')
ax.bar(x + width/2, coverage_in_95, width, label='95% interval',
       color='seagreen', alpha=0.7, edgecolor='black')
ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5, label='Expected (50%)')
ax.axhline(0.95, color='green', linestyle='--', alpha=0.5, label='Expected (95%)')
ax.set_xlabel('Trial', fontsize=11)
ax.set_ylabel('Coverage (1 = covered, 0 = not)', fontsize=11)
ax.set_title('Coverage of Observed by Prior Intervals', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(-0.1, 1.1)
ax.text(0.05, 0.05,
        f'50% coverage: {sum(coverage_in_50)}/12 ({100*sum(coverage_in_50)/12:.0f}%)\n'
        f'95% coverage: {sum(coverage_in_95)}/12 ({100*sum(coverage_in_95)/12:.0f}%)',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(plots_dir / 'prior_predictive_coverage.png')
print(f"   Saved: {plots_dir / 'prior_predictive_coverage.png'}")
plt.close()


# ============================================================================
# PLOT 3: Extreme Values Diagnostic
# ============================================================================
print("\n3. Creating extreme values diagnostic plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Trial 1: 0/47
ax = axes[0, 0]
trial_1_counts = r[:, 0]
bins = np.arange(0, trial_1_counts.max() + 2) - 0.5
ax.hist(trial_1_counts, bins=bins, density=False, alpha=0.7,
        edgecolor='black', color='steelblue')
ax.axvline(r_obs[0], color='red', linestyle='--', lw=3,
          label=f'Observed: {r_obs[0]}')
ax.set_xlabel('Successes in Trial 1 (n=47)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Trial 1: Extreme Low (0/47)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
pct_zero = 100 * (trial_1_counts == 0).mean()
ax.text(0.95, 0.95,
        f'Prior P(r=0) = {pct_zero:.1f}%\n'
        f'Prior mean: {trial_1_counts.mean():.1f}\n'
        f'Prior SD: {trial_1_counts.std():.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Trial 1: theta distribution
ax = axes[0, 1]
theta_1 = theta[:, 0]
ax.hist(theta_1, bins=50, density=True, alpha=0.7,
        edgecolor='black', color='steelblue')
ax.axvline(0, color='red', linestyle='--', lw=2,
          label='Observed: 0/47 → θ≈0')
ax.axvline(theta_1.mean(), color='blue', linestyle='--', lw=2,
          label=f'Prior mean: {theta_1.mean():.3f}')
ax.set_xlabel('θ₁ (probability for Trial 1)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Trial 1: Probability Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
pct_low = 100 * (theta_1 < 0.01).mean()
ax.text(0.95, 0.95,
        f'Prior P(θ < 0.01) = {pct_low:.1f}%\n'
        f'Range: [{theta_1.min():.4f}, {theta_1.max():.3f}]',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Trial 8: 31/215
ax = axes[1, 0]
trial_8_counts = r[:, 7]
ax.hist(trial_8_counts, bins=50, density=False, alpha=0.7,
        edgecolor='black', color='seagreen')
ax.axvline(r_obs[7], color='red', linestyle='--', lw=3,
          label=f'Observed: {r_obs[7]}')
ax.axvline(trial_8_counts.mean(), color='blue', linestyle='--', lw=2,
          label=f'Prior mean: {trial_8_counts.mean():.1f}')
q = np.quantile(trial_8_counts, [0.025, 0.975])
ax.axvspan(q[0], q[1], alpha=0.2, color='blue', label='95% interval')
ax.set_xlabel('Successes in Trial 8 (n=215)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Trial 8: Extreme High (31/215)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
percentile = stats.percentileofscore(trial_8_counts, r_obs[7])
ax.text(0.95, 0.95,
        f'Observed at {percentile:.1f}th percentile\n'
        f'Prior SD: {trial_8_counts.std():.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Trial 8: theta distribution
ax = axes[1, 1]
theta_8 = theta[:, 7]
ax.hist(theta_8, bins=50, density=True, alpha=0.7,
        edgecolor='black', color='seagreen')
ax.axvline(31/215, color='red', linestyle='--', lw=2,
          label=f'Observed: {31/215:.3f}')
ax.axvline(theta_8.mean(), color='blue', linestyle='--', lw=2,
          label=f'Prior mean: {theta_8.mean():.3f}')
q = np.quantile(theta_8, [0.025, 0.975])
ax.axvspan(q[0], q[1], alpha=0.2, color='blue', label='95% interval')
ax.set_xlabel('θ₈ (probability for Trial 8)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Trial 8: Probability Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
percentile = stats.percentileofscore(theta_8, 31/215)
ax.text(0.95, 0.95,
        f'Observed at {percentile:.1f}th percentile\n'
        f'Range: [{theta_8.min():.4f}, {theta_8.max():.3f}]',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(plots_dir / 'extreme_values_diagnostic.png')
print(f"   Saved: {plots_dir / 'extreme_values_diagnostic.png'}")
plt.close()


# ============================================================================
# PLOT 4: Heterogeneity Assessment (sigma implications)
# ============================================================================
print("\n4. Creating heterogeneity assessment plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sigma vs range of theta
ax = axes[0, 0]
theta_range = theta.max(axis=1) - theta.min(axis=1)
ax.scatter(sigma, theta_range, alpha=0.3, s=20, color='steelblue')
ax.set_xlabel('σ (scale parameter)', fontsize=11)
ax.set_ylabel('Range of θ across trials (max - min)', fontsize=11)
ax.set_title('Heterogeneity: σ vs Probability Range', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
# Add observed range
obs_range = (r_obs / n_obs).max() - (r_obs / n_obs).min()
ax.axhline(obs_range, color='red', linestyle='--', lw=2,
          label=f'Observed range: {obs_range:.3f}')
ax.legend(fontsize=9)

# Sigma vs SD of theta
ax = axes[0, 1]
theta_sd = theta.std(axis=1)
ax.scatter(sigma, theta_sd, alpha=0.3, s=20, color='seagreen')
ax.set_xlabel('σ (scale parameter)', fontsize=11)
ax.set_ylabel('SD of θ across trials', fontsize=11)
ax.set_title('Heterogeneity: σ vs Probability SD', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
# Add observed SD
obs_sd = (r_obs / n_obs).std()
ax.axhline(obs_sd, color='red', linestyle='--', lw=2,
          label=f'Observed SD: {obs_sd:.3f}')
ax.legend(fontsize=9)

# Distribution of theta by sigma quartile
ax = axes[1, 0]
sigma_quartiles = np.quantile(sigma, [0, 0.25, 0.5, 0.75, 1.0])
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))
labels = ['Q1 (low σ)', 'Q2', 'Q3', 'Q4 (high σ)']

for i in range(4):
    mask = (sigma >= sigma_quartiles[i]) & (sigma < sigma_quartiles[i+1])
    theta_subset = theta[mask].flatten()
    ax.hist(theta_subset, bins=50, alpha=0.5, density=True,
           color=colors[i], label=labels[i])

ax.set_xlabel('θ (trial probability)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Probability Distribution by σ Quartile', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Example prior predictive datasets for different sigma
ax = axes[1, 1]
# Select 4 example datasets with different sigma values
sigma_indices = []
for q in [0.1, 0.3, 0.7, 0.9]:
    idx = np.argmin(np.abs(sigma - np.quantile(sigma, q)))
    sigma_indices.append(idx)

for i, idx in enumerate(sigma_indices):
    prop_sim = r[idx] / n_obs
    x = np.arange(12) + 1 + (i - 1.5) * 0.05
    ax.scatter(x, prop_sim, s=60, alpha=0.7, color=colors[i],
              label=f'σ = {sigma[idx]:.2f}')

# Add observed
ax.scatter(np.arange(12) + 1, r_obs / n_obs, s=100, marker='s',
          color='red', edgecolor='black', linewidth=2,
          label='Observed', zorder=10)

ax.set_xlabel('Trial', fontsize=11)
ax.set_ylabel('Success proportion', fontsize=11)
ax.set_title('Example Datasets by σ Value', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(-0.02, 0.35)

plt.tight_layout()
plt.savefig(plots_dir / 'heterogeneity_diagnostic.png')
print(f"   Saved: {plots_dir / 'heterogeneity_diagnostic.png'}")
plt.close()


# ============================================================================
# PLOT 5: Trial-by-Trial Comparison
# ============================================================================
print("\n5. Creating trial-by-trial comparison plot...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(12):
    ax = axes[i]

    # Histogram of simulated counts
    bins = np.arange(0, r[:, i].max() + 2) - 0.5
    ax.hist(r[:, i], bins=min(30, len(bins)), density=False, alpha=0.7,
           edgecolor='black', color='steelblue')

    # Observed value
    ax.axvline(r_obs[i], color='red', linestyle='--', lw=2.5,
              label=f'Obs: {r_obs[i]}')

    # Prior mean
    ax.axvline(r[:, i].mean(), color='blue', linestyle='--', lw=1.5,
              label=f'Mean: {r[:, i].mean():.1f}')

    # Title and labels
    ax.set_title(f'Trial {i+1}: {r_obs[i]}/{n_obs[i]} ({100*r_obs[i]/n_obs[i]:.1f}%)',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Successes', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

    # Add percentile
    percentile = stats.percentileofscore(r[:, i], r_obs[i])
    ax.text(0.95, 0.95, f'{percentile:.0f}th %ile',
           transform=ax.transAxes, fontsize=8, verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Prior Predictive Check: All Trials',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(plots_dir / 'trial_by_trial_comparison.png')
print(f"   Saved: {plots_dir / 'trial_by_trial_comparison.png'}")
plt.close()


# ============================================================================
# PLOT 6: Logit Scale Behavior
# ============================================================================
print("\n6. Creating logit scale behavior plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution of logit(theta)
ax = axes[0, 0]
logit_theta = np.log(theta / (1 - theta + 1e-10))  # Add small epsilon for stability
logit_theta_valid = logit_theta[np.isfinite(logit_theta)]
ax.hist(logit_theta_valid, bins=60, density=True, alpha=0.7,
       edgecolor='black', color='steelblue')
ax.set_xlabel('logit(θ)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Trial Log-Odds', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.axvline(logit_theta_valid.mean(), color='blue', linestyle='--', lw=2,
          label=f'Mean: {logit_theta_valid.mean():.2f}')
ax.legend(fontsize=9)

# mu_logit + sigma*eta vs theta
ax = axes[0, 1]
# Sample 500 random (mu, sigma, eta, theta) tuples for visualization
sample_idx = np.random.choice(len(mu_logit), 500, replace=False)
for i in sample_idx:
    logit_vals = mu_logit[i] + sigma[i] * np.linspace(-3, 3, 100)
    theta_vals = expit(logit_vals)
    ax.plot(logit_vals, theta_vals, alpha=0.02, color='blue')

# Add the mean trajectory
mean_mu = mu_logit.mean()
mean_sigma = sigma.mean()
logit_vals = mean_mu + mean_sigma * np.linspace(-3, 3, 100)
theta_vals = expit(logit_vals)
ax.plot(logit_vals, theta_vals, color='red', lw=3,
       label=f'Mean: μ={mean_mu:.2f}, σ={mean_sigma:.2f}')

ax.set_xlabel('μ_logit + σ·η', fontsize=11)
ax.set_ylabel('θ = logistic(μ_logit + σ·η)', fontsize=11)
ax.set_title('Logistic Transformation', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.set_xlim(-8, 2)
ax.set_ylim(0, 1)

# Joint distribution: mu_logit vs sigma
ax = axes[1, 0]
h = ax.hist2d(mu_logit, sigma, bins=50, cmap='Blues', cmin=1)
plt.colorbar(h[3], ax=ax, label='Count')
ax.set_xlabel('μ_logit', fontsize=11)
ax.set_ylabel('σ', fontsize=11)
ax.set_title('Joint Prior: μ_logit vs σ', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
# Add correlation
corr = np.corrcoef(mu_logit, sigma)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
       transform=ax.transAxes, fontsize=10, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Extreme theta values
ax = axes[1, 1]
pct_very_low = 100 * (theta < 0.01).mean(axis=0)
pct_low = 100 * ((theta >= 0.01) & (theta < 0.05)).mean(axis=0)
pct_mid = 100 * ((theta >= 0.05) & (theta < 0.15)).mean(axis=0)
pct_high = 100 * (theta >= 0.15).mean(axis=0)

x = np.arange(12) + 1
width = 0.6
bottom1 = pct_very_low
bottom2 = bottom1 + pct_low
bottom3 = bottom2 + pct_mid

ax.bar(x, pct_very_low, width, label='θ < 0.01', color='darkred', alpha=0.8)
ax.bar(x, pct_low, width, bottom=bottom1, label='0.01 ≤ θ < 0.05',
      color='orange', alpha=0.8)
ax.bar(x, pct_mid, width, bottom=bottom2, label='0.05 ≤ θ < 0.15',
      color='steelblue', alpha=0.8)
ax.bar(x, pct_high, width, bottom=bottom3, label='θ ≥ 0.15',
      color='seagreen', alpha=0.8)

ax.set_xlabel('Trial', fontsize=11)
ax.set_ylabel('Percentage of prior draws', fontsize=11)
ax.set_title('Distribution of θ Ranges by Trial', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(plots_dir / 'logit_scale_behavior.png')
print(f"   Saved: {plots_dir / 'logit_scale_behavior.png'}")
plt.close()

print("\n" + "=" * 70)
print("All visualizations created successfully!")
print("=" * 70)
