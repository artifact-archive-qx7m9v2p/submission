"""
Prior Predictive Check for Experiment 2: AR(1) Log-Normal with Regime-Switching

This script implements prior predictive simulation for a complex model with:
1. AR(1) error structure requiring sequential generation
2. Regime-specific variances
3. Log-normal likelihood

Key Challenge: Cannot vectorize due to AR dependency structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_DRAWS = 1000
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Load data
data = pd.read_csv(DATA_PATH)
year = data['year'].values
C_obs = data['C'].values
n_obs = len(year)

# Define regimes (from metadata)
regime = np.concatenate([
    np.ones(14, dtype=int),      # Early: 1-14
    np.ones(13, dtype=int) * 2,  # Middle: 15-27
    np.ones(13, dtype=int) * 3   # Late: 28-40
])

print(f"Data loaded: {n_obs} observations")
print(f"Count range: [{C_obs.min()}, {C_obs.max()}]")
print(f"Year range: [{year.min():.2f}, {year.max():.2f}]")
print(f"Regime structure: {np.unique(regime, return_counts=True)}")

# ============================================================================
# Prior Specification
# ============================================================================

def sample_priors(n_draws):
    """Sample from prior distributions."""
    priors = {
        'alpha': np.random.normal(4.3, 0.5, n_draws),
        'beta_1': np.random.normal(0.86, 0.2, n_draws),
        'beta_2': np.random.normal(0, 0.3, n_draws),
        'phi': np.random.uniform(-0.95, 0.95, n_draws),
        'sigma_1': np.abs(np.random.normal(0, 1, n_draws)),  # HalfNormal
        'sigma_2': np.abs(np.random.normal(0, 1, n_draws)),
        'sigma_3': np.abs(np.random.normal(0, 1, n_draws))
    }
    return priors

# ============================================================================
# Prior Predictive Simulation (Sequential AR Generation)
# ============================================================================

def generate_ar_trajectory(alpha, beta_1, beta_2, phi, sigma_regime, year, regime):
    """
    Generate a single AR(1) log-normal trajectory.

    This MUST be done sequentially because epsilon[t] depends on epsilon[t-1].

    Parameters:
    -----------
    alpha, beta_1, beta_2, phi : float
        Model parameters
    sigma_regime : array of shape (3,)
        Regime-specific standard deviations
    year : array of shape (n,)
        Time covariate
    regime : array of shape (n,)
        Regime indicator (1, 2, or 3)

    Returns:
    --------
    C : array of shape (n,)
        Generated count data
    log_C : array of shape (n,)
        Log-scale data
    epsilon : array of shape (n,)
        AR errors
    """
    n = len(year)
    log_C = np.zeros(n)
    epsilon = np.zeros(n)

    # Initialize epsilon[0] from stationary distribution
    # Variance of stationary AR(1): sigma^2 / (1 - phi^2)
    if np.abs(phi) < 1:
        init_sd = sigma_regime[0] / np.sqrt(1 - phi**2)
    else:
        # Edge case: if phi is exactly ±1, use sigma directly
        init_sd = sigma_regime[0]

    epsilon[0] = np.random.normal(0, init_sd)

    # Generate first observation
    mu_0 = alpha + beta_1 * year[0] + beta_2 * year[0]**2 + phi * epsilon[0]
    log_C[0] = np.random.normal(mu_0, sigma_regime[regime[0] - 1])

    # Note: epsilon[0] already initialized above, but we need to update it
    # to be consistent with the actual generated data
    epsilon[0] = log_C[0] - (alpha + beta_1 * year[0] + beta_2 * year[0]**2)

    # Sequential generation for t = 1, 2, ..., n-1
    for t in range(1, n):
        # Mean structure includes AR component
        mu_t = alpha + beta_1 * year[t] + beta_2 * year[t]**2 + phi * epsilon[t-1]

        # Generate log(C[t]) from normal distribution
        log_C[t] = np.random.normal(mu_t, sigma_regime[regime[t] - 1])

        # Compute error for next iteration
        epsilon[t] = log_C[t] - (alpha + beta_1 * year[t] + beta_2 * year[t]**2)

    # Transform to count scale
    C = np.exp(log_C)

    return C, log_C, epsilon

def prior_predictive_simulation(n_draws):
    """
    Generate prior predictive datasets.

    Returns:
    --------
    dict with:
        'C_samples': array of shape (n_draws, n_obs) - count scale
        'log_C_samples': array of shape (n_draws, n_obs) - log scale
        'epsilon_samples': array of shape (n_draws, n_obs) - AR errors
        'priors': dict of prior draws
    """
    priors = sample_priors(n_draws)

    C_samples = np.zeros((n_draws, n_obs))
    log_C_samples = np.zeros((n_draws, n_obs))
    epsilon_samples = np.zeros((n_draws, n_obs))

    print(f"\nGenerating {n_draws} prior predictive trajectories...")
    print("This may take a minute due to sequential AR generation...")

    for i in range(n_draws):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_draws}")

        sigma_regime = np.array([
            priors['sigma_1'][i],
            priors['sigma_2'][i],
            priors['sigma_3'][i]
        ])

        C, log_C, epsilon = generate_ar_trajectory(
            alpha=priors['alpha'][i],
            beta_1=priors['beta_1'][i],
            beta_2=priors['beta_2'][i],
            phi=priors['phi'][i],
            sigma_regime=sigma_regime,
            year=year,
            regime=regime
        )

        C_samples[i] = C
        log_C_samples[i] = log_C
        epsilon_samples[i] = epsilon

    print("Prior predictive simulation complete!")

    return {
        'C_samples': C_samples,
        'log_C_samples': log_C_samples,
        'epsilon_samples': epsilon_samples,
        'priors': priors
    }

# ============================================================================
# Diagnostic Calculations
# ============================================================================

def compute_autocorrelation(x, max_lag=5):
    """Compute autocorrelation function up to max_lag."""
    acf = np.zeros(max_lag + 1)
    x_centered = x - np.mean(x)
    var = np.var(x)

    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.mean(x_centered[:-lag] * x_centered[lag:]) / var

    return acf

def compute_prior_acf_distribution(epsilon_samples, lag=1):
    """Compute ACF at given lag for all prior draws."""
    n_draws = epsilon_samples.shape[0]
    acf_values = np.zeros(n_draws)

    for i in range(n_draws):
        acf = compute_autocorrelation(epsilon_samples[i], max_lag=lag)
        acf_values[i] = acf[lag]

    return acf_values

def check_domain_violations(C_samples):
    """Check for negative or extreme values."""
    violations = {
        'negative': np.sum(C_samples < 0),
        'extreme_high': np.sum(C_samples > 1000),
        'extreme_low': np.sum(C_samples < 1),
        'nan_or_inf': np.sum(~np.isfinite(C_samples))
    }
    return violations

def check_plausibility_ranges(C_samples, C_obs):
    """Check if prior predictions are in plausible ranges."""
    n_draws, n_obs = C_samples.shape

    # Observed range
    obs_min, obs_max = C_obs.min(), C_obs.max()

    # Extended plausible range (broader than observed)
    plausible_min, plausible_max = 10, 500

    # Check each draw
    in_obs_range = np.zeros(n_draws, dtype=bool)
    in_plausible_range = np.zeros(n_draws, dtype=bool)

    for i in range(n_draws):
        in_obs_range[i] = np.all((C_samples[i] >= obs_min) & (C_samples[i] <= obs_max))
        in_plausible_range[i] = np.all((C_samples[i] >= plausible_min) & (C_samples[i] <= plausible_max))

    return {
        'pct_in_obs_range': 100 * np.mean(in_obs_range),
        'pct_in_plausible_range': 100 * np.mean(in_plausible_range),
        'obs_range': (obs_min, obs_max),
        'plausible_range': (plausible_min, plausible_max)
    }

# ============================================================================
# Run Prior Predictive Check
# ============================================================================

print("\n" + "="*80)
print("PRIOR PREDICTIVE CHECK - EXPERIMENT 2")
print("="*80)

results = prior_predictive_simulation(N_PRIOR_DRAWS)

# ============================================================================
# Diagnostic Analysis
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC CHECKS")
print("="*80)

# 1. Domain violations
print("\n1. DOMAIN VIOLATIONS")
violations = check_domain_violations(results['C_samples'])
print(f"   Negative counts: {violations['negative']} ({violations['negative']/(N_PRIOR_DRAWS*n_obs)*100:.2f}%)")
print(f"   Extreme high (>1000): {violations['extreme_high']} ({violations['extreme_high']/(N_PRIOR_DRAWS*n_obs)*100:.2f}%)")
print(f"   Extreme low (<1): {violations['extreme_low']} ({violations['extreme_low']/(N_PRIOR_DRAWS*n_obs)*100:.2f}%)")
print(f"   NaN/Inf values: {violations['nan_or_inf']} ({violations['nan_or_inf']/(N_PRIOR_DRAWS*n_obs)*100:.2f}%)")

# 2. Plausibility ranges
print("\n2. PLAUSIBILITY RANGES")
plausibility = check_plausibility_ranges(results['C_samples'], C_obs)
print(f"   Observed range: {plausibility['obs_range']}")
print(f"   Plausible range: {plausibility['plausible_range']}")
print(f"   Draws fully in observed range: {plausibility['pct_in_obs_range']:.1f}%")
print(f"   Draws fully in plausible range: {plausibility['pct_in_plausible_range']:.1f}%")

# 3. Prior autocorrelation distribution
print("\n3. AUTOCORRELATION STRUCTURE")
acf_lag1 = compute_prior_acf_distribution(results['epsilon_samples'], lag=1)
obs_acf = compute_autocorrelation(np.log(C_obs), max_lag=1)[1]
print(f"   Observed log(C) ACF lag-1: {obs_acf:.3f}")
print(f"   Prior ACF lag-1 (median): {np.median(acf_lag1):.3f}")
print(f"   Prior ACF lag-1 (90% CI): [{np.percentile(acf_lag1, 5):.3f}, {np.percentile(acf_lag1, 95):.3f}]")
print(f"   Prior ACF covers observed: {np.percentile(acf_lag1, 5) <= obs_acf <= np.percentile(acf_lag1, 95)}")

# 4. Prior parameter summaries
print("\n4. PRIOR PARAMETER DISTRIBUTIONS")
for param in ['alpha', 'beta_1', 'beta_2', 'phi', 'sigma_1', 'sigma_2', 'sigma_3']:
    values = results['priors'][param]
    print(f"   {param}: median={np.median(values):.3f}, 90% CI=[{np.percentile(values, 5):.3f}, {np.percentile(values, 95):.3f}]")

# 5. Prior predictive summaries
print("\n5. PRIOR PREDICTIVE SUMMARIES (Count Scale)")
C_median = np.median(results['C_samples'], axis=0)
C_mean = np.mean(results['C_samples'], axis=0)
print(f"   Overall median prediction: {np.median(C_median):.1f}")
print(f"   Overall mean prediction: {np.mean(C_mean):.1f}")
print(f"   Min prediction (across all draws): {np.min(results['C_samples']):.1f}")
print(f"   Max prediction (across all draws): {np.max(results['C_samples']):.1f}")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ---------------------------------------------------------------------------
# Plot 1: Prior Parameter Distributions
# ---------------------------------------------------------------------------
print("\n1. Creating parameter_plausibility.png...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

params_to_plot = [
    ('alpha', 'Intercept (log scale)', None),
    ('beta_1', 'Linear trend', None),
    ('beta_2', 'Quadratic trend', None),
    ('phi', 'AR(1) coefficient', (-1, 1)),
    ('sigma_1', 'Sigma (Early regime)', (0, 3)),
    ('sigma_2', 'Sigma (Middle regime)', (0, 3)),
    ('sigma_3', 'Sigma (Late regime)', (0, 3)),
]

for idx, (param, label, xlim) in enumerate(params_to_plot):
    ax = axes[idx]
    values = results['priors'][param]

    ax.hist(values, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(np.median(values), color='red', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{param} Prior Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    if xlim:
        ax.set_xlim(xlim)

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_plausibility.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved!")

# ---------------------------------------------------------------------------
# Plot 2: Prior Predictive Coverage
# ---------------------------------------------------------------------------
print("\n2. Creating prior_predictive_coverage.png...")

fig, ax = plt.subplots(figsize=(14, 7))

# Compute quantiles
percentiles = [2.5, 5, 25, 50, 75, 95, 97.5]
quantiles = np.percentile(results['C_samples'], percentiles, axis=0)

# Plot shaded regions
ax.fill_between(year, quantiles[0], quantiles[-1], alpha=0.2, color='blue', label='95% Prior Predictive')
ax.fill_between(year, quantiles[1], quantiles[-2], alpha=0.3, color='blue', label='90% Prior Predictive')
ax.fill_between(year, quantiles[2], quantiles[-3], alpha=0.4, color='blue', label='50% Prior Predictive')

# Plot median
ax.plot(year, quantiles[3], color='blue', linewidth=2, label='Prior Predictive Median')

# Plot observed data
ax.scatter(year, C_obs, color='red', s=50, zorder=10, label='Observed Data', alpha=0.8)

# Formatting
ax.set_xlabel('Year (standardized)', fontsize=13)
ax.set_ylabel('Count (C)', fontsize=13)
ax.set_title('Prior Predictive Coverage: AR(1) Log-Normal Model', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_coverage.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved!")

# ---------------------------------------------------------------------------
# Plot 3: Sample Prior Trajectories (AR Structure Visualization)
# ---------------------------------------------------------------------------
print("\n3. Creating prior_trajectories.png...")

fig, ax = plt.subplots(figsize=(14, 7))

# Plot 50 random trajectories
n_trajectories = 50
indices = np.random.choice(N_PRIOR_DRAWS, n_trajectories, replace=False)

for i in indices:
    ax.plot(year, results['C_samples'][i], alpha=0.3, linewidth=1, color='blue')

# Plot observed data
ax.scatter(year, C_obs, color='red', s=50, zorder=10, label='Observed Data', alpha=0.8)

# Formatting
ax.set_xlabel('Year (standardized)', fontsize=13)
ax.set_ylabel('Count (C)', fontsize=13)
ax.set_title(f'Prior Predictive Trajectories (n={n_trajectories}): Visualizing AR Temporal Smoothness',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(500, C_obs.max() * 1.2))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_trajectories.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved!")

# ---------------------------------------------------------------------------
# Plot 4: Prior Autocorrelation Distribution
# ---------------------------------------------------------------------------
print("\n4. Creating prior_autocorrelation_diagnostic.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: ACF distribution
ax = axes[0]
ax.hist(acf_lag1, bins=50, alpha=0.7, edgecolor='black', color='blue', density=True)
ax.axvline(np.median(acf_lag1), color='darkblue', linestyle='--', linewidth=2, label='Prior Median')
ax.axvline(obs_acf, color='red', linestyle='-', linewidth=2, label='Observed ACF')
ax.set_xlabel('ACF Lag-1', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution of Autocorrelation (ACF Lag-1)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Right panel: phi vs ACF
ax = axes[1]
ax.scatter(results['priors']['phi'], acf_lag1, alpha=0.3, s=10, color='blue')
ax.axhline(obs_acf, color='red', linestyle='-', linewidth=2, label='Observed ACF')
ax.set_xlabel('Prior phi (AR coefficient)', fontsize=12)
ax.set_ylabel('Implied ACF Lag-1', fontsize=12)
ax.set_title('Relationship: phi vs Implied Autocorrelation', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_autocorrelation_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved!")

# ---------------------------------------------------------------------------
# Plot 5: Regime Variance Structure
# ---------------------------------------------------------------------------
print("\n5. Creating regime_variance_diagnostic.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Sigma prior distributions
ax = axes[0, 0]
ax.hist(results['priors']['sigma_1'], bins=50, alpha=0.5, label='Sigma 1 (Early)', color='green', density=True)
ax.hist(results['priors']['sigma_2'], bins=50, alpha=0.5, label='Sigma 2 (Middle)', color='orange', density=True)
ax.hist(results['priors']['sigma_3'], bins=50, alpha=0.5, label='Sigma 3 (Late)', color='purple', density=True)
ax.set_xlabel('Sigma Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distributions: Regime-Specific Sigma', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Sigma ordering (how often is each regime largest?)
ax = axes[0, 1]
sigma_matrix = np.column_stack([
    results['priors']['sigma_1'],
    results['priors']['sigma_2'],
    results['priors']['sigma_3']
])
largest_regime = np.argmax(sigma_matrix, axis=1) + 1
unique, counts = np.unique(largest_regime, return_counts=True)
ax.bar(['Early (1)', 'Middle (2)', 'Late (3)'],
       [counts[unique == i][0] if i in unique else 0 for i in [1, 2, 3]],
       color=['green', 'orange', 'purple'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Which Regime Has Largest Sigma? (Prior)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Prior predictive variance by regime
ax = axes[1, 0]
regime_labels = ['Early\n(1-14)', 'Middle\n(15-27)', 'Late\n(28-40)']
regime_bounds = [(0, 14), (14, 27), (27, 40)]

variances_by_regime = []
for start, end in regime_bounds:
    regime_var = np.var(results['C_samples'][:, start:end], axis=1)
    variances_by_regime.append(regime_var)

bp = ax.boxplot(variances_by_regime, labels=regime_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['green', 'orange', 'purple']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Variance (Count Scale)', fontsize=11)
ax.set_title('Prior Predictive Variance by Regime', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Example trajectory with regime boundaries
ax = axes[1, 1]
example_idx = np.random.randint(N_PRIOR_DRAWS)
ax.plot(year, results['C_samples'][example_idx], linewidth=2, color='blue', label='Example Trajectory')
ax.axvline(year[13], color='green', linestyle='--', linewidth=2, alpha=0.7, label='Early/Middle boundary')
ax.axvline(year[26], color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Middle/Late boundary')
ax.scatter(year, C_obs, color='red', s=30, zorder=10, label='Observed', alpha=0.5)
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Example: Regime Boundaries', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "regime_variance_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved!")

# ---------------------------------------------------------------------------
# Plot 6: Log-Scale Prior Predictive (Original vs Log Scale Comparison)
# ---------------------------------------------------------------------------
print("\n6. Creating log_scale_diagnostic.png...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Count scale (same as prior predictive coverage but focused)
ax = axes[0]
percentiles = [2.5, 25, 50, 75, 97.5]
quantiles = np.percentile(results['C_samples'], percentiles, axis=0)
ax.fill_between(year, quantiles[0], quantiles[-1], alpha=0.3, color='blue', label='95% Interval')
ax.fill_between(year, quantiles[1], quantiles[-2], alpha=0.4, color='blue', label='50% Interval')
ax.plot(year, quantiles[2], color='blue', linewidth=2, label='Median')
ax.scatter(year, C_obs, color='red', s=50, zorder=10, label='Observed', alpha=0.8)
ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Count (C)', fontsize=12)
ax.set_title('Prior Predictive: Count Scale', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: Log scale
ax = axes[1]
quantiles_log = np.percentile(results['log_C_samples'], percentiles, axis=0)
ax.fill_between(year, quantiles_log[0], quantiles_log[-1], alpha=0.3, color='blue', label='95% Interval')
ax.fill_between(year, quantiles_log[1], quantiles_log[-2], alpha=0.4, color='blue', label='50% Interval')
ax.plot(year, quantiles_log[2], color='blue', linewidth=2, label='Median')
ax.scatter(year, np.log(C_obs), color='red', s=50, zorder=10, label='Observed', alpha=0.8)
ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('log(C)', fontsize=12)
ax.set_title('Prior Predictive: Log Scale', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "log_scale_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved!")

print("\n" + "="*80)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"Plots saved to: {PLOTS_DIR}")
