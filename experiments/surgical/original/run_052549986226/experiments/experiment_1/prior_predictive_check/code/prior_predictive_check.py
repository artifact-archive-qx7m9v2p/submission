"""
Prior Predictive Check for Beta-Binomial (Reparameterized) Model

This script validates that the priors generate scientifically plausible data
before fitting the model to real data.

Model:
    r_i ~ Binomial(n_i, p_i)
    p_i ~ Beta(a, b) where a = mu*kappa, b = (1-mu)*kappa

Priors:
    mu ~ Beta(2, 18)      # Population mean success probability
    kappa ~ Gamma(2, 0.1)  # Concentration parameter

Generated:
    phi = 1 + 1/kappa     # Overdispersion parameter
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
n_groups = len(data)
n_trials = data['n_trials'].values
r_observed = data['r_successes'].values

# Calculate observed statistics
pooled_rate_observed = r_observed.sum() / n_trials.sum()
group_rates_observed = r_observed / n_trials

# Calculate observed overdispersion (Method of Moments estimate)
# phi = 1 + variance_excess / (mean * (1 - mean))
mean_obs = pooled_rate_observed
variance_obs = np.var(group_rates_observed, ddof=1)
variance_binom = mean_obs * (1 - mean_obs) / n_trials.mean()
phi_observed = 1 + (variance_obs - variance_binom) / (mean_obs * (1 - mean_obs))

print(f"\nObserved Statistics:")
print(f"  Pooled success rate: {pooled_rate_observed:.4f} ({pooled_rate_observed*100:.2f}%)")
print(f"  Group rates range: [{group_rates_observed.min():.4f}, {group_rates_observed.max():.4f}]")
print(f"  Estimated overdispersion (phi): {phi_observed:.2f}")
print(f"  Groups with zero successes: {np.sum(r_observed == 0)}")

# =============================================================================
# STEP 1: Sample from Prior Distributions
# =============================================================================
print("\n" + "="*80)
print("STEP 1: Sampling from Prior Distributions")
print("="*80)

n_prior_samples = 10000
print(f"Drawing {n_prior_samples} samples from priors...")

# Prior: mu ~ Beta(2, 18)
mu_prior = stats.beta(2, 18).rvs(n_prior_samples)

# Prior: kappa ~ Gamma(2, 0.1)
kappa_prior = stats.gamma(2, scale=1/0.1).rvs(n_prior_samples)

# Derived: phi = 1 + 1/kappa
phi_prior = 1 + 1/kappa_prior

# Derived: alpha and beta parameters
alpha_prior = mu_prior * kappa_prior
beta_param_prior = (1 - mu_prior) * kappa_prior

print("\nPrior Parameter Summaries:")
print("\nmu (population mean):")
mu_quantiles = np.quantile(mu_prior, [0.025, 0.25, 0.50, 0.75, 0.975])
print(f"  2.5%: {mu_quantiles[0]:.4f}")
print(f"  25%:  {mu_quantiles[1]:.4f}")
print(f"  50%:  {mu_quantiles[2]:.4f}")
print(f"  75%:  {mu_quantiles[3]:.4f}")
print(f"  97.5%: {mu_quantiles[4]:.4f}")

print("\nkappa (concentration):")
kappa_quantiles = np.quantile(kappa_prior, [0.025, 0.25, 0.50, 0.75, 0.975])
print(f"  2.5%: {kappa_quantiles[0]:.4f}")
print(f"  25%:  {kappa_quantiles[1]:.4f}")
print(f"  50%:  {kappa_quantiles[2]:.4f}")
print(f"  75%:  {kappa_quantiles[3]:.4f}")
print(f"  97.5%: {kappa_quantiles[4]:.4f}")

print("\nphi (overdispersion = 1 + 1/kappa):")
phi_quantiles = np.quantile(phi_prior, [0.025, 0.25, 0.50, 0.75, 0.975])
print(f"  2.5%: {phi_quantiles[0]:.4f}")
print(f"  25%:  {phi_quantiles[1]:.4f}")
print(f"  50%:  {phi_quantiles[2]:.4f}")
print(f"  75%:  {phi_quantiles[3]:.4f}")
print(f"  97.5%: {phi_quantiles[4]:.4f}")

# =============================================================================
# STEP 2: Prior Predictive Data Generation
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Generating Prior Predictive Data")
print("="*80)

n_simulations = 1000
print(f"Generating {n_simulations} prior predictive datasets...")

# Storage for prior predictive statistics
pooled_rates_prior_pred = np.zeros(n_simulations)
phi_prior_pred = np.zeros(n_simulations)
min_group_rates = np.zeros(n_simulations)
max_group_rates = np.zeros(n_simulations)
range_group_rates = np.zeros(n_simulations)
n_zeros = np.zeros(n_simulations, dtype=int)
n_impossible = np.zeros(n_simulations, dtype=int)

# Storage for example group rates (for visualization)
example_group_rates = []

for sim in range(n_simulations):
    # Sample hyperparameters from prior
    mu_sim = stats.beta(2, 18).rvs()
    kappa_sim = stats.gamma(2, scale=1/0.1).rvs()

    # Convert to alpha, beta for Beta distribution
    alpha_sim = mu_sim * kappa_sim
    beta_sim = (1 - mu_sim) * kappa_sim

    # Sample group-level success probabilities
    p_groups = stats.beta(alpha_sim, beta_sim).rvs(n_groups)

    # Generate observed counts
    y_rep = stats.binom.rvs(n_trials, p_groups)

    # Calculate statistics
    pooled_rates_prior_pred[sim] = y_rep.sum() / n_trials.sum()

    group_rates_sim = y_rep / n_trials
    min_group_rates[sim] = group_rates_sim.min()
    max_group_rates[sim] = group_rates_sim.max()
    range_group_rates[sim] = group_rates_sim.max() - group_rates_sim.min()

    # Count zeros
    n_zeros[sim] = np.sum(y_rep == 0)

    # Check for impossible values (y > n)
    n_impossible[sim] = np.sum(y_rep > n_trials)

    # Calculate overdispersion for this simulation
    var_sim = np.var(group_rates_sim, ddof=1)
    mean_sim = pooled_rates_prior_pred[sim]
    if mean_sim > 0 and mean_sim < 1:
        var_binom_sim = mean_sim * (1 - mean_sim) / n_trials.mean()
        phi_prior_pred[sim] = 1 + (var_sim - var_binom_sim) / (mean_sim * (1 - mean_sim))
    else:
        phi_prior_pred[sim] = np.nan

    # Store first 100 examples for visualization
    if sim < 100:
        example_group_rates.append(group_rates_sim)

example_group_rates = np.array(example_group_rates)

print("\nPrior Predictive Summaries:")
print("\nPooled success rate:")
pooled_quantiles = np.quantile(pooled_rates_prior_pred, [0.025, 0.25, 0.50, 0.75, 0.975])
print(f"  2.5%: {pooled_quantiles[0]:.4f}")
print(f"  25%:  {pooled_quantiles[1]:.4f}")
print(f"  50%:  {pooled_quantiles[2]:.4f}")
print(f"  75%:  {pooled_quantiles[3]:.4f}")
print(f"  97.5%: {pooled_quantiles[4]:.4f}")
print(f"  Observed: {pooled_rate_observed:.4f}")

print("\nOverdispersion (phi):")
phi_pred_quantiles = np.quantile(phi_prior_pred[~np.isnan(phi_prior_pred)],
                                  [0.025, 0.25, 0.50, 0.75, 0.975])
print(f"  2.5%: {phi_pred_quantiles[0]:.4f}")
print(f"  25%:  {phi_pred_quantiles[1]:.4f}")
print(f"  50%:  {phi_pred_quantiles[2]:.4f}")
print(f"  75%:  {phi_pred_quantiles[3]:.4f}")
print(f"  97.5%: {phi_pred_quantiles[4]:.4f}")
print(f"  Observed: {phi_observed:.2f}")

print("\nMaximum group success rate:")
max_rate_quantiles = np.quantile(max_group_rates, [0.025, 0.25, 0.50, 0.75, 0.975])
print(f"  2.5%: {max_rate_quantiles[0]:.4f}")
print(f"  25%:  {max_rate_quantiles[1]:.4f}")
print(f"  50%:  {max_rate_quantiles[2]:.4f}")
print(f"  75%:  {max_rate_quantiles[3]:.4f}")
print(f"  97.5%: {max_rate_quantiles[4]:.4f}")
print(f"  Observed: {group_rates_observed.max():.4f}")

print("\nNumber of groups with zero successes:")
print(f"  Mean: {n_zeros.mean():.2f}")
print(f"  Proportion of sims with >=1 zero: {np.mean(n_zeros >= 1):.4f}")
print(f"  Observed: {np.sum(r_observed == 0)}")

# =============================================================================
# STEP 3: Critical Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 3: Critical Checks")
print("="*80)

checks = {}

# Check 1: Validity - no impossible values
pct_impossible = (n_impossible > 0).mean() * 100
checks['validity'] = {
    'pass': pct_impossible == 0,
    'value': pct_impossible,
    'criterion': '0%'
}
print(f"\n1. VALIDITY: Proportion with impossible values (y > n): {pct_impossible:.2f}%")
print(f"   Status: {'PASS' if checks['validity']['pass'] else 'FAIL'}")

# Check 2: Coverage of observed mean
in_50_interval = (pooled_rate_observed >= pooled_quantiles[1] and
                  pooled_rate_observed <= pooled_quantiles[3])
in_95_interval = (pooled_rate_observed >= pooled_quantiles[0] and
                  pooled_rate_observed <= pooled_quantiles[4])
checks['mean_coverage'] = {
    'pass': in_95_interval,
    'in_50': in_50_interval,
    'in_95': in_95_interval,
    'observed': pooled_rate_observed,
    'interval_95': (pooled_quantiles[0], pooled_quantiles[4])
}
print(f"\n2. MEAN COVERAGE: Observed pooled rate = {pooled_rate_observed:.4f}")
print(f"   In 50% interval: {in_50_interval}")
print(f"   In 95% interval: {in_95_interval}")
print(f"   Status: {'PASS' if checks['mean_coverage']['pass'] else 'FAIL'}")

# Check 3: Coverage of overdispersion
phi_valid = phi_prior_pred[~np.isnan(phi_prior_pred)]
phi_10_quantile = np.quantile(phi_valid, 0.10)
phi_90_quantile = np.quantile(phi_valid, 0.90)
in_80_interval_phi = (phi_observed >= phi_10_quantile and
                       phi_observed <= phi_90_quantile)
checks['phi_coverage'] = {
    'pass': in_80_interval_phi,
    'observed': phi_observed,
    'interval_80': (phi_10_quantile, phi_90_quantile)
}
print(f"\n3. OVERDISPERSION COVERAGE: Observed phi = {phi_observed:.2f}")
print(f"   80% interval: [{phi_10_quantile:.2f}, {phi_90_quantile:.2f}]")
print(f"   Status: {'PASS' if checks['phi_coverage']['pass'] else 'FAIL'}")

# Check 4: Zero count plausibility
pct_with_zeros = (n_zeros >= 1).mean() * 100
checks['zero_plausibility'] = {
    'pass': pct_with_zeros >= 1 and pct_with_zeros <= 50,
    'value': pct_with_zeros,
    'criterion': '1-50%'
}
print(f"\n4. ZERO PLAUSIBILITY: {pct_with_zeros:.1f}% of simulations have >=1 zero")
print(f"   Status: {'PASS' if checks['zero_plausibility']['pass'] else 'FAIL'}")

# Check 5: Phi range spans reasonable values
phi_range_check = (phi_quantiles[0] < 2.0 and phi_quantiles[4] > 10.0)
checks['phi_range'] = {
    'pass': phi_range_check,
    'range': (phi_quantiles[0], phi_quantiles[4]),
    'criterion': 'spans [1.5, 10]'
}
print(f"\n5. PHI RANGE: Prior phi 95% interval = [{phi_quantiles[0]:.2f}, {phi_quantiles[4]:.2f}]")
print(f"   Spans [1.5, 10]: {phi_range_check}")
print(f"   Status: {'PASS' if checks['phi_range']['pass'] else 'FAIL'}")

# Overall pass/fail
all_pass = all(check['pass'] for check in checks.values())
print("\n" + "="*80)
print(f"OVERALL VERDICT: {'PASS' if all_pass else 'FAIL'}")
print("="*80)

# =============================================================================
# STEP 4: Visualizations
# =============================================================================
print("\n" + "="*80)
print("STEP 4: Creating Visualizations")
print("="*80)

# Plot 1: Prior Parameter Distributions
print("\nCreating parameter_plausibility.png...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# mu prior
ax = axes[0, 0]
ax.hist(mu_prior, bins=50, density=True, alpha=0.7, edgecolor='black', label='Prior samples')
x_mu = np.linspace(0, 1, 1000)
ax.plot(x_mu, stats.beta(2, 18).pdf(x_mu), 'r-', lw=2, label='Beta(2, 18) density')
ax.axvline(pooled_rate_observed, color='green', linestyle='--', lw=2, label=f'Observed rate: {pooled_rate_observed:.3f}')
ax.axvline(mu_quantiles[2], color='blue', linestyle=':', lw=2, label=f'Prior median: {mu_quantiles[2]:.3f}')
ax.set_xlabel('mu (population mean success probability)')
ax.set_ylabel('Density')
ax.set_title('Prior Distribution of mu')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# kappa prior
ax = axes[0, 1]
ax.hist(kappa_prior, bins=50, density=True, alpha=0.7, edgecolor='black', label='Prior samples')
x_kappa = np.linspace(0, np.percentile(kappa_prior, 99), 1000)
ax.plot(x_kappa, stats.gamma(2, scale=1/0.1).pdf(x_kappa), 'r-', lw=2, label='Gamma(2, 0.1) density')
ax.axvline(kappa_quantiles[2], color='blue', linestyle=':', lw=2, label=f'Prior median: {kappa_quantiles[2]:.1f}')
ax.set_xlabel('kappa (concentration parameter)')
ax.set_ylabel('Density')
ax.set_title('Prior Distribution of kappa')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# phi prior
ax = axes[1, 0]
ax.hist(phi_prior, bins=50, density=True, alpha=0.7, edgecolor='black', label='Prior samples')
ax.axvline(phi_observed, color='green', linestyle='--', lw=2, label=f'Observed phi: {phi_observed:.2f}')
ax.axvline(phi_quantiles[2], color='blue', linestyle=':', lw=2, label=f'Prior median: {phi_quantiles[2]:.2f}')
ax.set_xlabel('phi (overdispersion = 1 + 1/kappa)')
ax.set_ylabel('Density')
ax.set_title('Implied Prior Distribution of phi')
ax.set_xlim(1, 20)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Joint: mu vs kappa (colored by phi)
ax = axes[1, 1]
scatter = ax.scatter(mu_prior[:1000], kappa_prior[:1000], c=phi_prior[:1000],
                     s=10, alpha=0.5, cmap='viridis', vmin=1, vmax=10)
ax.axvline(pooled_rate_observed, color='red', linestyle='--', lw=1, alpha=0.5, label='Observed rate')
ax.set_xlabel('mu (population mean)')
ax.set_ylabel('kappa (concentration)')
ax.set_title('Joint Prior: mu vs kappa (colored by phi)')
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('phi (overdispersion)')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_plausibility.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: parameter_plausibility.png")

# Plot 2: Prior Predictive Distributions - Pooled Rate and Phi
print("\nCreating prior_predictive_coverage.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pooled rate
ax = axes[0]
ax.hist(pooled_rates_prior_pred, bins=50, density=True, alpha=0.7, edgecolor='black',
        label='Prior predictive')
ax.axvline(pooled_rate_observed, color='red', linestyle='--', lw=2,
          label=f'Observed: {pooled_rate_observed:.3f}')
ax.axvline(pooled_quantiles[1], color='blue', linestyle=':', lw=1.5, alpha=0.7,
          label=f'50% interval: [{pooled_quantiles[1]:.3f}, {pooled_quantiles[3]:.3f}]')
ax.axvline(pooled_quantiles[3], color='blue', linestyle=':', lw=1.5, alpha=0.7)
ax.axvspan(pooled_quantiles[0], pooled_quantiles[4], alpha=0.2, color='blue',
          label=f'95% interval: [{pooled_quantiles[0]:.3f}, {pooled_quantiles[4]:.3f}]')
ax.set_xlabel('Pooled success rate')
ax.set_ylabel('Density')
ax.set_title('Prior Predictive: Pooled Success Rate')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Overdispersion
ax = axes[1]
phi_valid = phi_prior_pred[~np.isnan(phi_prior_pred)]
ax.hist(phi_valid, bins=50, density=True, alpha=0.7, edgecolor='black',
        label='Prior predictive')
ax.axvline(phi_observed, color='red', linestyle='--', lw=2,
          label=f'Observed: {phi_observed:.2f}')
ax.axvline(phi_10_quantile, color='blue', linestyle=':', lw=1.5, alpha=0.7,
          label=f'80% interval: [{phi_10_quantile:.2f}, {phi_90_quantile:.2f}]')
ax.axvline(phi_90_quantile, color='blue', linestyle=':', lw=1.5, alpha=0.7)
ax.set_xlabel('Overdispersion (phi)')
ax.set_ylabel('Density')
ax.set_title('Prior Predictive: Overdispersion')
ax.set_xlim(-5, 15)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_coverage.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: prior_predictive_coverage.png")

# Plot 3: Group-level Success Rates
print("\nCreating group_rate_examples.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Example trajectories
ax = axes[0]
for i in range(min(50, len(example_group_rates))):
    ax.plot(range(1, n_groups+1), example_group_rates[i], 'o-', alpha=0.3, markersize=3)
ax.plot(range(1, n_groups+1), group_rates_observed, 'ro-', lw=2, markersize=8,
        label='Observed', zorder=100)
ax.set_xlabel('Group ID')
ax.set_ylabel('Success rate')
ax.set_title('Prior Predictive: Example Group Success Rates (50 simulations)')
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution of maximum group rate
ax = axes[1]
ax.hist(max_group_rates, bins=50, density=True, alpha=0.7, edgecolor='black',
        label='Prior predictive')
ax.axvline(group_rates_observed.max(), color='red', linestyle='--', lw=2,
          label=f'Observed max: {group_rates_observed.max():.3f}')
ax.axvspan(max_rate_quantiles[0], max_rate_quantiles[4], alpha=0.2, color='blue',
          label=f'95% interval: [{max_rate_quantiles[0]:.3f}, {max_rate_quantiles[4]:.3f}]')
ax.set_xlabel('Maximum group success rate')
ax.set_ylabel('Density')
ax.set_title('Prior Predictive: Maximum Group Success Rate')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "group_rate_examples.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: group_rate_examples.png")

# Plot 4: Zero Count Diagnostic
print("\nCreating zero_inflation_diagnostic.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of number of zeros
ax = axes[0]
zero_counts = np.bincount(n_zeros, minlength=n_groups+1)
ax.bar(range(len(zero_counts)), zero_counts, alpha=0.7, edgecolor='black')
ax.axvline(np.sum(r_observed == 0), color='red', linestyle='--', lw=2,
          label=f'Observed: {np.sum(r_observed == 0)} groups with 0')
ax.set_xlabel('Number of groups with zero successes')
ax.set_ylabel('Frequency')
ax.set_title('Prior Predictive: Distribution of Zero Counts')
ax.legend()
ax.grid(True, alpha=0.3)

# Minimum group rate distribution
ax = axes[1]
ax.hist(min_group_rates, bins=50, density=True, alpha=0.7, edgecolor='black',
        label='Prior predictive')
ax.axvline(group_rates_observed.min(), color='red', linestyle='--', lw=2,
          label=f'Observed min: {group_rates_observed.min():.3f}')
ax.set_xlabel('Minimum group success rate')
ax.set_ylabel('Density')
ax.set_title('Prior Predictive: Minimum Group Success Rate')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "zero_inflation_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: zero_inflation_diagnostic.png")

# Plot 5: Comprehensive Prior-Data Comparison
print("\nCreating comprehensive_comparison.png...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Prior parameters
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(mu_prior, bins=50, density=True, alpha=0.7, edgecolor='black')
ax1.axvline(pooled_rate_observed, color='red', linestyle='--', lw=2, label='Observed')
ax1.set_xlabel('mu')
ax1.set_title('Prior: Population Mean')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(kappa_prior, bins=50, density=True, alpha=0.7, edgecolor='black')
ax2.set_xlabel('kappa')
ax2.set_title('Prior: Concentration')
ax2.set_xlim(0, 100)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(phi_prior, bins=50, density=True, alpha=0.7, edgecolor='black')
ax3.axvline(phi_observed, color='red', linestyle='--', lw=2, label='Observed')
ax3.set_xlabel('phi')
ax3.set_title('Prior: Overdispersion')
ax3.set_xlim(1, 20)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Row 2: Prior predictive summaries
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(pooled_rates_prior_pred, bins=50, density=True, alpha=0.7, edgecolor='black')
ax4.axvline(pooled_rate_observed, color='red', linestyle='--', lw=2, label='Observed')
ax4.axvspan(pooled_quantiles[0], pooled_quantiles[4], alpha=0.2, color='blue')
ax4.set_xlabel('Pooled rate')
ax4.set_title('Prior Predictive: Pooled Rate')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(phi_valid, bins=50, density=True, alpha=0.7, edgecolor='black')
ax5.axvline(phi_observed, color='red', linestyle='--', lw=2, label='Observed')
ax5.axvspan(phi_10_quantile, phi_90_quantile, alpha=0.2, color='blue')
ax5.set_xlabel('phi')
ax5.set_title('Prior Predictive: Overdispersion')
ax5.set_xlim(-5, 15)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(max_group_rates, bins=50, density=True, alpha=0.7, edgecolor='black')
ax6.axvline(group_rates_observed.max(), color='red', linestyle='--', lw=2, label='Observed')
ax6.set_xlabel('Max group rate')
ax6.set_title('Prior Predictive: Max Rate')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Row 3: Detailed diagnostics
ax7 = fig.add_subplot(gs[2, 0])
ax7.bar(range(len(zero_counts)), zero_counts/n_simulations, alpha=0.7, edgecolor='black')
ax7.axvline(np.sum(r_observed == 0), color='red', linestyle='--', lw=2, label='Observed')
ax7.set_xlabel('# groups with 0 successes')
ax7.set_ylabel('Proportion')
ax7.set_title('Prior Predictive: Zero Counts')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(range_group_rates, bins=50, density=True, alpha=0.7, edgecolor='black')
observed_range = group_rates_observed.max() - group_rates_observed.min()
ax8.axvline(observed_range, color='red', linestyle='--', lw=2, label=f'Observed: {observed_range:.3f}')
ax8.set_xlabel('Range of group rates')
ax8.set_title('Prior Predictive: Rate Range')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

ax9 = fig.add_subplot(gs[2, 2])
# Check text summary
check_text = "CRITICAL CHECKS:\n\n"
check_text += f"1. Validity: {'PASS' if checks['validity']['pass'] else 'FAIL'}\n"
check_text += f"   {checks['validity']['value']:.2f}% impossible\n\n"
check_text += f"2. Mean coverage: {'PASS' if checks['mean_coverage']['pass'] else 'FAIL'}\n"
check_text += f"   In 95%: {checks['mean_coverage']['in_95']}\n\n"
check_text += f"3. Phi coverage: {'PASS' if checks['phi_coverage']['pass'] else 'FAIL'}\n"
check_text += f"   In 80%: {in_80_interval_phi}\n\n"
check_text += f"4. Zero plausibility: {'PASS' if checks['zero_plausibility']['pass'] else 'FAIL'}\n"
check_text += f"   {checks['zero_plausibility']['value']:.1f}% with zeros\n\n"
check_text += f"5. Phi range: {'PASS' if checks['phi_range']['pass'] else 'FAIL'}\n"
check_text += f"   95%: [{checks['phi_range']['range'][0]:.2f}, {checks['phi_range']['range'][1]:.2f}]\n\n"
check_text += f"OVERALL: {'PASS' if all_pass else 'FAIL'}"
ax9.text(0.05, 0.95, check_text, transform=ax9.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax9.axis('off')
ax9.set_title('Check Summary')

plt.savefig(PLOTS_DIR / "comprehensive_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: comprehensive_comparison.png")

print("\n" + "="*80)
print("Prior predictive check complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("="*80)

# Save summary statistics to file
summary_file = OUTPUT_DIR / "summary_statistics.txt"
with open(summary_file, 'w') as f:
    f.write("PRIOR PREDICTIVE CHECK SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("OBSERVED DATA:\n")
    f.write(f"  Pooled success rate: {pooled_rate_observed:.4f}\n")
    f.write(f"  Overdispersion (phi): {phi_observed:.2f}\n")
    f.write(f"  Groups with zero successes: {np.sum(r_observed == 0)}\n")
    f.write(f"  Max group rate: {group_rates_observed.max():.4f}\n\n")

    f.write("PRIOR PARAMETER QUANTILES:\n")
    f.write(f"  mu: {mu_quantiles}\n")
    f.write(f"  kappa: {kappa_quantiles}\n")
    f.write(f"  phi: {phi_quantiles}\n\n")

    f.write("PRIOR PREDICTIVE QUANTILES:\n")
    f.write(f"  Pooled rate: {pooled_quantiles}\n")
    f.write(f"  Phi: {phi_pred_quantiles}\n")
    f.write(f"  Max rate: {max_rate_quantiles}\n\n")

    f.write("CRITICAL CHECKS:\n")
    for i, (name, check) in enumerate(checks.items(), 1):
        f.write(f"  {i}. {name}: {'PASS' if check['pass'] else 'FAIL'}\n")

    f.write(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}\n")

print(f"\nSummary statistics saved to: {summary_file}")
