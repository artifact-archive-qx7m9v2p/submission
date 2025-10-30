"""
Prior Predictive Check for Experiment 2: Log-Linear Heteroscedastic Model

Model:
  Y_i ~ Normal(mu_i, sigma_i)
  mu_i = beta_0 + beta_1 * log(x_i)
  log(sigma_i) = gamma_0 + gamma_1 * x_i

Priors:
  beta_0 ~ Normal(1.8, 0.5)
  beta_1 ~ Normal(0.3, 0.2)
  gamma_0 ~ Normal(-2, 1)
  gamma_1 ~ Normal(-0.05, 0.05)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_SAMPLES = 1000
OUTPUT_DIR = "/workspace/experiments/experiment_2/prior_predictive_check"

# Load observed data
print("Loading observed data...")
data = pd.read_csv("/workspace/data/data.csv")
print(f"Loaded {len(data)} observations")
print(f"Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")
print(f"x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")

# Calculate observed variance by bins to check heteroscedasticity
x_bins = pd.qcut(data['x'], q=3, duplicates='drop')
variance_by_bin = data.groupby(x_bins)['Y'].var()
print(f"\nObserved variance by x terciles:")
print(variance_by_bin)
print(f"Variance ratio (high x / low x): {variance_by_bin.iloc[0] / variance_by_bin.iloc[-1]:.2f}x")

# Define prior distributions
print(f"\n{'='*60}")
print("SAMPLING FROM PRIORS")
print('='*60)

# Sample from priors
beta_0_prior = np.random.normal(1.8, 0.5, N_PRIOR_SAMPLES)
beta_1_prior = np.random.normal(0.3, 0.2, N_PRIOR_SAMPLES)
gamma_0_prior = np.random.normal(-2, 1, N_PRIOR_SAMPLES)
gamma_1_prior = np.random.normal(-0.05, 0.05, N_PRIOR_SAMPLES)

print(f"Generated {N_PRIOR_SAMPLES} prior samples for each parameter")
print(f"\nbeta_0: mean={beta_0_prior.mean():.3f}, std={beta_0_prior.std():.3f}, range=[{beta_0_prior.min():.3f}, {beta_0_prior.max():.3f}]")
print(f"beta_1: mean={beta_1_prior.mean():.3f}, std={beta_1_prior.std():.3f}, range=[{beta_1_prior.min():.3f}, {beta_1_prior.max():.3f}]")
print(f"gamma_0: mean={gamma_0_prior.mean():.3f}, std={gamma_0_prior.std():.3f}, range=[{gamma_0_prior.min():.3f}, {gamma_0_prior.max():.3f}]")
print(f"gamma_1: mean={gamma_1_prior.mean():.3f}, std={gamma_1_prior.std():.3f}, range=[{gamma_1_prior.min():.3f}, {gamma_1_prior.max():.3f}]")

# Check for pathological parameter values
print(f"\n{'='*60}")
print("CHECKING PARAMETER PLAUSIBILITY")
print('='*60)

n_positive_gamma1 = np.sum(gamma_1_prior > 0)
print(f"gamma_1 > 0 (variance increases with x): {n_positive_gamma1}/{N_PRIOR_SAMPLES} ({100*n_positive_gamma1/N_PRIOR_SAMPLES:.1f}%)")
print(f"  -> Should be ~50% if centered at -0.05 with sd=0.05")

# Generate prior predictive data
print(f"\n{'='*60}")
print("GENERATING PRIOR PREDICTIVE DATA")
print('='*60)

# Use observed x values for prediction
x_obs = data['x'].values
log_x_obs = np.log(x_obs)

# Storage for generated data
y_simulated = np.zeros((N_PRIOR_SAMPLES, len(x_obs)))
mu_simulated = np.zeros((N_PRIOR_SAMPLES, len(x_obs)))
sigma_simulated = np.zeros((N_PRIOR_SAMPLES, len(x_obs)))

# Track problematic samples
n_negative_sigma = 0
n_extreme_sigma = 0
n_negative_y = 0

for i in range(N_PRIOR_SAMPLES):
    # Generate mean structure
    mu_i = beta_0_prior[i] + beta_1_prior[i] * log_x_obs

    # Generate variance structure
    log_sigma_i = gamma_0_prior[i] + gamma_1_prior[i] * x_obs
    sigma_i = np.exp(log_sigma_i)

    # Check for problematic values
    if np.any(sigma_i <= 0):
        n_negative_sigma += 1
    if np.any(sigma_i > 10):
        n_extreme_sigma += 1

    # Generate observations
    y_i = np.random.normal(mu_i, sigma_i)

    if np.any(y_i < 0):
        n_negative_y += 1

    # Store
    y_simulated[i, :] = y_i
    mu_simulated[i, :] = mu_i
    sigma_simulated[i, :] = sigma_i

print(f"Generated {N_PRIOR_SAMPLES} synthetic datasets")
print(f"\nProblematic samples:")
print(f"  Negative sigma: {n_negative_sigma}/{N_PRIOR_SAMPLES} ({100*n_negative_sigma/N_PRIOR_SAMPLES:.1f}%)")
print(f"  Extreme sigma (>10): {n_extreme_sigma}/{N_PRIOR_SAMPLES} ({100*n_extreme_sigma/N_PRIOR_SAMPLES:.1f}%)")
print(f"  Negative Y: {n_negative_y}/{N_PRIOR_SAMPLES} ({100*n_negative_y/N_PRIOR_SAMPLES:.1f}%)")

# Analyze generated data
print(f"\n{'='*60}")
print("PRIOR PREDICTIVE SUMMARY STATISTICS")
print('='*60)

y_min = y_simulated.min(axis=1)
y_max = y_simulated.max(axis=1)
y_range = y_max - y_min

print(f"\nGenerated Y ranges:")
print(f"  Mean range: [{y_min.mean():.2f}, {y_max.mean():.2f}]")
print(f"  5th percentile: [{np.percentile(y_min, 5):.2f}, {np.percentile(y_max, 5):.2f}]")
print(f"  95th percentile: [{np.percentile(y_min, 95):.2f}, {np.percentile(y_max, 95):.2f}]")
print(f"  Overall range: [{y_simulated.min():.2f}, {y_simulated.max():.2f}]")
print(f"\nObserved Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")
print(f"Target range: [0.5, 5.0]")

# Check coverage
y_obs = data['Y'].values
n_cover_obs_min = np.sum(y_min <= data['Y'].min())
n_cover_obs_max = np.sum(y_max >= data['Y'].max())
n_cover_both = np.sum((y_min <= data['Y'].min()) & (y_max >= data['Y'].max()))

print(f"\nObserved data coverage:")
print(f"  Covers observed minimum: {n_cover_obs_min}/{N_PRIOR_SAMPLES} ({100*n_cover_obs_min/N_PRIOR_SAMPLES:.1f}%)")
print(f"  Covers observed maximum: {n_cover_obs_max}/{N_PRIOR_SAMPLES} ({100*n_cover_obs_max/N_PRIOR_SAMPLES:.1f}%)")
print(f"  Covers both: {n_cover_both}/{N_PRIOR_SAMPLES} ({100*n_cover_both/N_PRIOR_SAMPLES:.1f}%)")

# Check heteroscedasticity structure
print(f"\n{'='*60}")
print("CHECKING HETEROSCEDASTIC STRUCTURE")
print('='*60)

# Compare sigma at min and max x
sigma_at_min_x = sigma_simulated[:, 0]  # x = 1.0
sigma_at_max_x = sigma_simulated[:, -1]  # x = 31.5

variance_ratio = (sigma_at_min_x**2) / (sigma_at_max_x**2)
print(f"\nVariance ratio (low x / high x):")
print(f"  Mean: {variance_ratio.mean():.2f}x")
print(f"  Median: {np.median(variance_ratio):.2f}x")
print(f"  5th-95th percentile: [{np.percentile(variance_ratio, 5):.2f}, {np.percentile(variance_ratio, 95):.2f}]")
print(f"\nObserved variance ratio: ~7.5x")

n_decreasing = np.sum(variance_ratio > 1)
print(f"\nSamples with decreasing variance: {n_decreasing}/{N_PRIOR_SAMPLES} ({100*n_decreasing/N_PRIOR_SAMPLES:.1f}%)")

# Check mean structure
print(f"\n{'='*60}")
print("CHECKING MEAN STRUCTURE")
print('='*60)

mu_at_min_x = mu_simulated[:, 0]
mu_at_max_x = mu_simulated[:, -1]
mu_change = mu_at_max_x - mu_at_min_x

print(f"\nMean change from x=1 to x=31.5:")
print(f"  Mean: {mu_change.mean():.2f}")
print(f"  5th-95th percentile: [{np.percentile(mu_change, 5):.2f}, {np.percentile(mu_change, 95):.2f}]")

obs_change = data['Y'].iloc[-1] - data['Y'].iloc[0]
print(f"\nObserved Y change (endpoints): {obs_change:.2f}")

# Calculate similarity metric
print(f"\n{'='*60}")
print("CHECKING SIMILARITY TO OBSERVED DATA")
print('='*60)

# For each prior sample, check if generated data is "similar" to observed
# Criteria: range overlaps substantially and variance structure is consistent
similar_samples = 0
for i in range(N_PRIOR_SAMPLES):
    y_i = y_simulated[i, :]

    # Check 1: Range overlap (generated range includes 80% of observed range)
    obs_range = data['Y'].max() - data['Y'].min()
    obs_80_lower = data['Y'].min() + 0.1 * obs_range
    obs_80_upper = data['Y'].max() - 0.1 * obs_range

    range_ok = (y_i.min() <= obs_80_lower) and (y_i.max() >= obs_80_upper)

    # Check 2: Variance decreases with x
    variance_decreasing = variance_ratio[i] > 1.5

    # Check 3: No extreme values
    no_extremes = (y_i.min() > -1) and (y_i.max() < 10)

    if range_ok and variance_decreasing and no_extremes:
        similar_samples += 1

print(f"Samples generating data similar to observed: {similar_samples}/{N_PRIOR_SAMPLES} ({100*similar_samples/N_PRIOR_SAMPLES:.1f}%)")
print(f"Target: >20%")

# Save results for visualization
print(f"\n{'='*60}")
print("SAVING RESULTS")
print('='*60)

results = {
    'beta_0': beta_0_prior,
    'beta_1': beta_1_prior,
    'gamma_0': gamma_0_prior,
    'gamma_1': gamma_1_prior,
    'y_simulated': y_simulated,
    'mu_simulated': mu_simulated,
    'sigma_simulated': sigma_simulated,
    'x_obs': x_obs,
    'y_obs': y_obs,
    'variance_ratio': variance_ratio,
    'mu_change': mu_change,
    'similar_samples': similar_samples,
    'n_samples': N_PRIOR_SAMPLES
}

np.savez(f"{OUTPUT_DIR}/code/prior_predictive_samples.npz", **results)
print(f"Saved prior predictive samples to prior_predictive_samples.npz")

# Summary
print(f"\n{'='*60}")
print("DIAGNOSTIC SUMMARY")
print('='*60)
print("\nKEY FINDINGS:")
print(f"1. Parameter plausibility: gamma_1 < 0 in {100*(1-n_positive_gamma1/N_PRIOR_SAMPLES):.1f}% of samples")
print(f"2. Computational issues: {n_negative_sigma} negative sigmas, {n_extreme_sigma} extreme sigmas")
print(f"3. Coverage: {100*n_cover_both/N_PRIOR_SAMPLES:.1f}% of samples cover observed range")
print(f"4. Variance structure: {100*n_decreasing/N_PRIOR_SAMPLES:.1f}% show decreasing variance")
print(f"5. Similarity: {100*similar_samples/N_PRIOR_SAMPLES:.1f}% generate plausible data")

print("\n" + "="*60)
print("Prior predictive check complete!")
print("="*60)
