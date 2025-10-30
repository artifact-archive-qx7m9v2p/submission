"""
Prior Predictive Check for Experiment 1: Negative Binomial GLM with Quadratic Trend

This script validates that priors generate scientifically plausible data before model fitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_DRAWS = 1000
OUTPUT_DIR = Path('/workspace/experiments/experiment_1/prior_predictive_check')
PLOTS_DIR = OUTPUT_DIR / 'plots'
DATA_PATH = Path('/workspace/data/data.csv')

# Load observed data
data = pd.read_csv(DATA_PATH)
year_values = data['year'].values
observed_counts = data['C'].values
n_obs = len(year_values)

print("="*80)
print("PRIOR PREDICTIVE CHECK: Experiment 1")
print("="*80)
print(f"\nData summary:")
print(f"  N observations: {n_obs}")
print(f"  Year range: [{year_values.min():.2f}, {year_values.max():.2f}]")
print(f"  Count range: [{observed_counts.min()}, {observed_counts.max()}]")
print(f"  Mean count: {observed_counts.mean():.1f}")
print(f"  Variance: {observed_counts.var():.1f}")
print(f"  Variance/Mean ratio: {observed_counts.var()/observed_counts.mean():.1f}")

# ==============================================================================
# 1. Sample from priors
# ==============================================================================
print(f"\n{'='*80}")
print("STEP 1: Sampling from priors")
print(f"{'='*80}")
print(f"Drawing {N_PRIOR_DRAWS} samples from prior distributions...")

# Prior specifications:
# beta_0 ~ Normal(4.5, 1.0)
# beta_1 ~ Normal(0.9, 0.5)
# beta_2 ~ Normal(0, 0.3)
# phi ~ Gamma(2, 0.1)

beta_0_samples = np.random.normal(4.5, 1.0, N_PRIOR_DRAWS)
beta_1_samples = np.random.normal(0.9, 0.5, N_PRIOR_DRAWS)
beta_2_samples = np.random.normal(0, 0.3, N_PRIOR_DRAWS)
phi_samples = np.random.gamma(2, 1/0.1, N_PRIOR_DRAWS)  # scipy uses scale = 1/rate

print(f"\nPrior samples summary:")
print(f"  beta_0: mean={beta_0_samples.mean():.2f}, sd={beta_0_samples.std():.2f}, range=[{beta_0_samples.min():.2f}, {beta_0_samples.max():.2f}]")
print(f"  beta_1: mean={beta_1_samples.mean():.2f}, sd={beta_1_samples.std():.2f}, range=[{beta_1_samples.min():.2f}, {beta_1_samples.max():.2f}]")
print(f"  beta_2: mean={beta_2_samples.mean():.2f}, sd={beta_2_samples.std():.2f}, range=[{beta_2_samples.min():.2f}, {beta_2_samples.max():.2f}]")
print(f"  phi: mean={phi_samples.mean():.2f}, sd={phi_samples.std():.2f}, range=[{phi_samples.min():.2f}, {phi_samples.max():.2f}]")

# ==============================================================================
# 2. Generate prior predictive data
# ==============================================================================
print(f"\n{'='*80}")
print("STEP 2: Generating prior predictive data")
print(f"{'='*80}")

# For each prior draw, generate predicted counts across all year values
prior_predictive = np.zeros((N_PRIOR_DRAWS, n_obs))

for i in range(N_PRIOR_DRAWS):
    # Compute log(mu) using quadratic model
    log_mu = beta_0_samples[i] + beta_1_samples[i] * year_values + beta_2_samples[i] * year_values**2
    mu = np.exp(log_mu)

    # Sample from Negative Binomial
    # NegativeBinomial2 parameterization: n = phi, p = phi/(phi + mu)
    # This gives mean = mu, variance = mu + mu^2/phi
    for j in range(n_obs):
        # Convert to scipy parameterization: n, p where mean = n*(1-p)/p
        # Our parameterization: mean = mu, var = mu + mu^2/phi
        # scipy NB: n=phi, p=phi/(phi+mu) gives mean=mu, var=mu(1+mu/phi)
        p = phi_samples[i] / (phi_samples[i] + mu[j])
        prior_predictive[i, j] = np.random.negative_binomial(phi_samples[i], p)

print(f"Prior predictive data generated: shape {prior_predictive.shape}")

# ==============================================================================
# 3. Compute diagnostic statistics
# ==============================================================================
print(f"\n{'='*80}")
print("STEP 3: Computing diagnostic statistics")
print(f"{'='*80}")

# Check for domain violations (negative counts - should be impossible)
negative_counts = np.sum(prior_predictive < 0)
print(f"\nDomain violations:")
print(f"  Negative counts: {negative_counts} (should be 0)")

# Check for extreme values
extreme_low = np.sum(prior_predictive < 10) / prior_predictive.size * 100
extreme_high = np.sum(prior_predictive > 1000) / prior_predictive.size * 100
very_extreme_high = np.sum(prior_predictive > 10000) / prior_predictive.size * 100

print(f"\nExtreme value analysis:")
print(f"  % predictions < 10: {extreme_low:.1f}%")
print(f"  % predictions > 1000: {extreme_high:.1f}%")
print(f"  % predictions > 10000: {very_extreme_high:.1f}%")

# Check plausibility range [10, 500]
plausible_range = (prior_predictive >= 10) & (prior_predictive <= 500)
pct_plausible = np.sum(plausible_range) / prior_predictive.size * 100
print(f"\nPlausibility assessment:")
print(f"  % predictions in [10, 500]: {pct_plausible:.1f}%")

# Check by prior draw (at least one plausible prediction per draw)
plausible_draws = np.sum(plausible_range, axis=1) > 0
pct_plausible_draws = np.sum(plausible_draws) / N_PRIOR_DRAWS * 100
print(f"  % prior draws with at least one plausible prediction: {pct_plausible_draws:.1f}%")

# Check if observed data is covered
observed_min, observed_max = observed_counts.min(), observed_counts.max()
predictive_min = np.percentile(prior_predictive, 0.5)
predictive_max = np.percentile(prior_predictive, 99.5)
print(f"\nObserved data coverage:")
print(f"  Observed range: [{observed_min}, {observed_max}]")
print(f"  Prior predictive 99% range: [{predictive_min:.0f}, {predictive_max:.0f}]")

# Check quantiles at each time point
coverage_stats = []
for j in range(n_obs):
    q05 = np.percentile(prior_predictive[:, j], 5)
    q95 = np.percentile(prior_predictive[:, j], 95)
    covered = (observed_counts[j] >= q05) and (observed_counts[j] <= q95)
    coverage_stats.append(covered)

pct_covered = np.sum(coverage_stats) / n_obs * 100
print(f"  % time points where observed in 90% prior predictive interval: {pct_covered:.1f}%")

# Compute mu at each timepoint for each prior draw to analyze growth patterns
mu_samples = np.zeros((N_PRIOR_DRAWS, n_obs))
for i in range(N_PRIOR_DRAWS):
    log_mu = beta_0_samples[i] + beta_1_samples[i] * year_values + beta_2_samples[i] * year_values**2
    mu_samples[i, :] = np.exp(log_mu)

# Check if priors favor exponential growth
growth_ratio = mu_samples[:, -1] / mu_samples[:, 0]  # ratio of final to initial mean
pct_growth = np.sum(growth_ratio > 1) / N_PRIOR_DRAWS * 100
median_growth = np.median(growth_ratio)
print(f"\nGrowth pattern analysis:")
print(f"  % prior draws with positive growth: {pct_growth:.1f}%")
print(f"  Median growth ratio (final/initial mu): {median_growth:.1f}")
print(f"  Observed growth ratio: {observed_counts[-1]/observed_counts[0]:.1f}")

# ==============================================================================
# 4. Create visualizations
# ==============================================================================
print(f"\n{'='*80}")
print("STEP 4: Creating visualizations")
print(f"{'='*80}")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# -------------------------
# Plot 1: Parameter plausibility - marginal prior distributions
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# beta_0
axes[0, 0].hist(beta_0_samples, bins=50, alpha=0.7, edgecolor='black', density=True)
axes[0, 0].axvline(4.5, color='red', linestyle='--', label='Prior mean')
axes[0, 0].set_xlabel('beta_0 (log-scale intercept)', fontsize=11)
axes[0, 0].set_ylabel('Density', fontsize=11)
axes[0, 0].set_title('Prior: beta_0 ~ Normal(4.5, 1.0)', fontsize=12, fontweight='bold')
axes[0, 0].legend()

# beta_1
axes[0, 1].hist(beta_1_samples, bins=50, alpha=0.7, edgecolor='black', density=True, color='orange')
axes[0, 1].axvline(0.9, color='red', linestyle='--', label='Prior mean')
axes[0, 1].axvline(0, color='black', linestyle=':', alpha=0.5, label='No linear trend')
axes[0, 1].set_xlabel('beta_1 (linear growth rate)', fontsize=11)
axes[0, 1].set_ylabel('Density', fontsize=11)
axes[0, 1].set_title('Prior: beta_1 ~ Normal(0.9, 0.5)', fontsize=12, fontweight='bold')
axes[0, 1].legend()

# beta_2
axes[1, 0].hist(beta_2_samples, bins=50, alpha=0.7, edgecolor='black', density=True, color='green')
axes[1, 0].axvline(0, color='red', linestyle='--', label='Prior mean (no curvature)')
axes[1, 0].set_xlabel('beta_2 (quadratic term)', fontsize=11)
axes[1, 0].set_ylabel('Density', fontsize=11)
axes[1, 0].set_title('Prior: beta_2 ~ Normal(0, 0.3)', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# phi
axes[1, 1].hist(phi_samples, bins=50, alpha=0.7, edgecolor='black', density=True, color='purple')
axes[1, 1].axvline(phi_samples.mean(), color='red', linestyle='--', label=f'Mean={phi_samples.mean():.1f}')
axes[1, 1].set_xlabel('phi (dispersion parameter)', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].set_title('Prior: phi ~ Gamma(2, 0.1)', fontsize=12, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_plausibility.png', dpi=300, bbox_inches='tight')
print(f"  Saved: parameter_plausibility.png")
plt.close()

# -------------------------
# Plot 2: Prior predictive coverage - main diagnostic
# -------------------------
fig, ax = plt.subplots(figsize=(14, 8))

# Compute credible intervals
q025 = np.percentile(prior_predictive, 2.5, axis=0)
q05 = np.percentile(prior_predictive, 5, axis=0)
q25 = np.percentile(prior_predictive, 25, axis=0)
q50 = np.percentile(prior_predictive, 50, axis=0)
q75 = np.percentile(prior_predictive, 75, axis=0)
q95 = np.percentile(prior_predictive, 95, axis=0)
q975 = np.percentile(prior_predictive, 97.5, axis=0)

# Plot intervals
ax.fill_between(year_values, q025, q975, alpha=0.2, color='blue', label='95% prior predictive interval')
ax.fill_between(year_values, q05, q95, alpha=0.3, color='blue', label='90% prior predictive interval')
ax.fill_between(year_values, q25, q75, alpha=0.4, color='blue', label='50% prior predictive interval')
ax.plot(year_values, q50, color='blue', linewidth=2, label='Prior predictive median')

# Overlay observed data
ax.scatter(year_values, observed_counts, color='red', s=50, zorder=5, alpha=0.8,
           edgecolors='darkred', linewidths=1.5, label='Observed data')

# Reference lines
ax.axhline(10, color='gray', linestyle=':', alpha=0.5, label='Plausibility bounds [10, 500]')
ax.axhline(500, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Year (standardized)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Prior Predictive Coverage: Do Priors Generate Plausible Data?',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Add text box with key statistics
textstr = f'N prior draws: {N_PRIOR_DRAWS}\n'
textstr += f'Obs in 90% interval: {pct_covered:.0f}%\n'
textstr += f'Predictions in [10,500]: {pct_plausible:.0f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
print(f"  Saved: prior_predictive_coverage.png")
plt.close()

# -------------------------
# Plot 3: Computational diagnostics - extreme values and scale issues
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Distribution of all prior predictive values (log scale)
axes[0, 0].hist(prior_predictive.flatten(), bins=100, alpha=0.7, edgecolor='black', color='steelblue')
axes[0, 0].axvline(observed_counts.min(), color='red', linestyle='--', linewidth=2, label='Obs min')
axes[0, 0].axvline(observed_counts.max(), color='red', linestyle='--', linewidth=2, label='Obs max')
axes[0, 0].axvline(10, color='gray', linestyle=':', alpha=0.7)
axes[0, 0].axvline(500, color='gray', linestyle=':', alpha=0.7)
axes[0, 0].set_xlabel('Prior Predictive Counts', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of All Prior Predictions', fontsize=12, fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Top right: Extreme value check
extreme_categories = ['< 10', '10-100', '100-500', '500-1000', '> 1000']
extreme_counts = [
    np.sum(prior_predictive < 10),
    np.sum((prior_predictive >= 10) & (prior_predictive < 100)),
    np.sum((prior_predictive >= 100) & (prior_predictive <= 500)),
    np.sum((prior_predictive > 500) & (prior_predictive <= 1000)),
    np.sum(prior_predictive > 1000)
]
extreme_pcts = [c / prior_predictive.size * 100 for c in extreme_counts]
colors = ['red', 'orange', 'green', 'orange', 'red']
axes[0, 1].bar(extreme_categories, extreme_pcts, color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('% of Prior Predictions', fontsize=11)
axes[0, 1].set_title('Prior Prediction Range Distribution', fontsize=12, fontweight='bold')
axes[0, 1].axhline(50, color='black', linestyle='--', alpha=0.5, label='50% threshold')
axes[0, 1].legend()
axes[0, 1].grid(True, axis='y', alpha=0.3)

# Bottom left: Growth trajectories (sample 100 prior draws)
n_sample_trajectories = 100
sample_idx = np.random.choice(N_PRIOR_DRAWS, n_sample_trajectories, replace=False)
for idx in sample_idx:
    log_mu = beta_0_samples[idx] + beta_1_samples[idx] * year_values + beta_2_samples[idx] * year_values**2
    mu = np.exp(log_mu)
    axes[1, 0].plot(year_values, mu, alpha=0.1, color='blue', linewidth=0.5)

axes[1, 0].plot(year_values, observed_counts, color='red', linewidth=2,
                marker='o', markersize=4, label='Observed', zorder=5)
axes[1, 0].set_xlabel('Year (standardized)', fontsize=11)
axes[1, 0].set_ylabel('Expected Count (mu)', fontsize=11)
axes[1, 0].set_title(f'Prior Mean Functions (n={n_sample_trajectories} samples)',
                      fontsize=12, fontweight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Bottom right: Dispersion vs mean relationship
# For each prior draw, compute mean and variance of predictions
mean_predictions = np.mean(prior_predictive, axis=1)
var_predictions = np.var(prior_predictive, axis=1)
axes[1, 1].scatter(mean_predictions, var_predictions, alpha=0.3, s=20, color='purple')
# Add theoretical line: var = mean + mean^2/phi for different phi values
mean_range = np.linspace(mean_predictions.min(), mean_predictions.max(), 100)
for phi_val in [5, 10, 20, 50]:
    var_theoretical = mean_range + mean_range**2 / phi_val
    axes[1, 1].plot(mean_range, var_theoretical, linestyle='--', alpha=0.5,
                    label=f'phi={phi_val}')
# Add observed point
axes[1, 1].scatter(observed_counts.mean(), observed_counts.var(),
                   color='red', s=200, marker='*', zorder=5,
                   edgecolors='darkred', linewidths=2, label='Observed')
axes[1, 1].set_xlabel('Mean of Prior Predictions', fontsize=11)
axes[1, 1].set_ylabel('Variance of Prior Predictions', fontsize=11)
axes[1, 1].set_title('Variance-Mean Relationship', fontsize=12, fontweight='bold')
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'computational_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: computational_diagnostics.png")
plt.close()

# -------------------------
# Plot 4: Growth pattern diagnostic
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Distribution of growth ratios
axes[0].hist(growth_ratio, bins=50, alpha=0.7, edgecolor='black', color='teal', density=True)
axes[0].axvline(observed_counts[-1]/observed_counts[0], color='red', linestyle='--',
                linewidth=2, label=f'Observed: {observed_counts[-1]/observed_counts[0]:.1f}x')
axes[0].axvline(1, color='black', linestyle=':', alpha=0.5, label='No growth')
axes[0].set_xlabel('Growth Ratio (final/initial mean)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Prior Distribution of Growth Patterns', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: Curvature analysis (beta_2 vs growth ratio)
axes[1].scatter(beta_2_samples, growth_ratio, alpha=0.2, s=20, color='teal')
axes[1].axhline(observed_counts[-1]/observed_counts[0], color='red', linestyle='--',
                linewidth=2, alpha=0.7, label='Observed growth')
axes[1].axvline(0, color='black', linestyle=':', alpha=0.5)
axes[1].axhline(1, color='black', linestyle=':', alpha=0.5)
axes[1].set_xlabel('beta_2 (quadratic term)', fontsize=11)
axes[1].set_ylabel('Growth Ratio', fontsize=11)
axes[1].set_title('Curvature vs Growth Relationship', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'growth_pattern_diagnostic.png', dpi=300, bbox_inches='tight')
print(f"  Saved: growth_pattern_diagnostic.png")
plt.close()

# ==============================================================================
# 5. Final assessment
# ==============================================================================
print(f"\n{'='*80}")
print("STEP 5: FINAL ASSESSMENT")
print(f"{'='*80}")

# Decision criteria: FAIL if <50% of prior draws produce plausible counts (10-500 range)
decision_threshold = 50.0
decision = "PASS" if pct_plausible >= decision_threshold else "FAIL"

print(f"\nDecision Rule: FAIL if <{decision_threshold}% of predictions in [10, 500] range")
print(f"Result: {pct_plausible:.1f}% of predictions in plausible range")
print(f"\n{'*'*80}")
print(f"DECISION: {decision}")
print(f"{'*'*80}")

if decision == "PASS":
    print("\nPriors are reasonable and generate scientifically plausible data.")
    print("Recommendation: Proceed to simulation validation and model fitting.")
else:
    print("\nPriors generate implausible data. Recommendations:")
    print("  1. Consider tightening priors if predictions too diffuse")
    print("  2. Consider relaxing priors if predictions too constrained")
    print("  3. Review model structure for potential misspecification")

# Save summary statistics to file
summary = {
    'n_prior_draws': N_PRIOR_DRAWS,
    'pct_plausible': pct_plausible,
    'pct_covered': pct_covered,
    'pct_growth': pct_growth,
    'median_growth': median_growth,
    'observed_growth': observed_counts[-1]/observed_counts[0],
    'decision': decision
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)
print(f"\n  Saved summary statistics to: summary_statistics.csv")

print(f"\n{'='*80}")
print("Prior predictive check complete!")
print(f"{'='*80}")
