"""
Prior Predictive Check for Logarithmic Regression Model
Experiment 1: Validating prior distributions before model fitting

Model:
  Y_i ~ Normal(μ_i, σ)
  μ_i = β₀ + β₁ · log(x_i)

Priors:
  β₀ ~ Normal(1.73, 0.5)
  β₁ ~ Normal(0.28, 0.15)
  σ ~ Exponential(5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_SAMPLES = 1000
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Load observed data
data = pd.read_csv("/workspace/data/data.csv")
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(x_obs)

print("=" * 80)
print("PRIOR PREDICTIVE CHECK: Logarithmic Regression")
print("=" * 80)
print(f"\nObserved Data Summary:")
print(f"  N = {n_obs}")
print(f"  x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"  Y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"  Y mean: {y_obs.mean():.2f}, SD: {y_obs.std():.2f}")

# ============================================================================
# 1. SAMPLE FROM PRIORS
# ============================================================================
print(f"\n{'-'*80}")
print("1. SAMPLING FROM PRIOR DISTRIBUTIONS")
print(f"{'-'*80}")

# Prior parameters
beta0_mean, beta0_sd = 1.73, 0.5
beta1_mean, beta1_sd = 0.28, 0.15
sigma_rate = 5.0

# Draw prior samples
beta0_prior = np.random.normal(beta0_mean, beta0_sd, N_PRIOR_SAMPLES)
beta1_prior = np.random.normal(beta1_mean, beta1_sd, N_PRIOR_SAMPLES)
sigma_prior = np.random.exponential(1/sigma_rate, N_PRIOR_SAMPLES)

print(f"Generated {N_PRIOR_SAMPLES} prior samples")
print(f"\nPrior Sample Statistics:")
print(f"  β₀: mean={beta0_prior.mean():.3f}, sd={beta0_prior.std():.3f}, "
      f"range=[{beta0_prior.min():.3f}, {beta0_prior.max():.3f}]")
print(f"  β₁: mean={beta1_prior.mean():.3f}, sd={beta1_prior.std():.3f}, "
      f"range=[{beta1_prior.min():.3f}, {beta1_prior.max():.3f}]")
print(f"  σ:  mean={sigma_prior.mean():.3f}, sd={sigma_prior.std():.3f}, "
      f"range=[{sigma_prior.min():.3f}, {sigma_prior.max():.3f}]")

# ============================================================================
# 2. GENERATE PRIOR PREDICTIVE DISTRIBUTIONS
# ============================================================================
print(f"\n{'-'*80}")
print("2. GENERATING PRIOR PREDICTIVE DATA")
print(f"{'-'*80}")

# Create grid for prediction (denser than observed data)
x_grid = np.linspace(x_obs.min(), x_obs.max(), 100)
log_x_grid = np.log(x_grid)
log_x_obs = np.log(x_obs)

# Generate prior predictive samples at observed x points
y_prior_pred_obs = np.zeros((N_PRIOR_SAMPLES, n_obs))
for i in range(N_PRIOR_SAMPLES):
    mu_i = beta0_prior[i] + beta1_prior[i] * log_x_obs
    y_prior_pred_obs[i, :] = np.random.normal(mu_i, sigma_prior[i])

# Generate prior predictive samples at grid points (for smooth curves)
mu_prior_pred_grid = np.zeros((N_PRIOR_SAMPLES, len(x_grid)))
for i in range(N_PRIOR_SAMPLES):
    mu_prior_pred_grid[i, :] = beta0_prior[i] + beta1_prior[i] * log_x_grid

print(f"Generated prior predictive distributions")
print(f"  At {n_obs} observed x points (with noise)")
print(f"  At {len(x_grid)} grid points (mean function)")

# ============================================================================
# 3. PRIOR PLAUSIBILITY CHECKS
# ============================================================================
print(f"\n{'-'*80}")
print("3. PRIOR PLAUSIBILITY ASSESSMENT")
print(f"{'-'*80}")

# Check 1: Coverage of observed data
y_obs_min, y_obs_max = y_obs.min(), y_obs.max()
y_prior_min = y_prior_pred_obs.min(axis=1)
y_prior_max = y_prior_pred_obs.max(axis=1)

covers_min = np.mean(y_prior_min <= y_obs_min)
covers_max = np.mean(y_prior_max >= y_obs_max)
covers_both = np.mean((y_prior_min <= y_obs_min) & (y_prior_max >= y_obs_max))

print(f"\nCoverage Analysis:")
print(f"  Proportion covering observed minimum: {covers_min:.1%}")
print(f"  Proportion covering observed maximum: {covers_max:.1%}")
print(f"  Proportion covering full observed range: {covers_both:.1%}")

# Check 2: Parameter plausibility
beta1_negative = np.mean(beta1_prior < 0)
beta1_positive = np.mean(beta1_prior > 0)

print(f"\nParameter Direction Analysis (β₁):")
print(f"  Proportion β₁ < 0 (negative relationship): {beta1_negative:.1%}")
print(f"  Proportion β₁ > 0 (positive relationship): {beta1_positive:.1%}")

# Check 3: Extreme predictions
# Define "reasonable" bounds based on observed data with buffer
y_lower_bound = y_obs_min - 2 * y_obs.std()
y_upper_bound = y_obs_max + 2 * y_obs.std()

extreme_low = np.mean(y_prior_pred_obs < y_lower_bound, axis=1)
extreme_high = np.mean(y_prior_pred_obs > y_upper_bound, axis=1)
any_extreme = np.mean((extreme_low > 0.1) | (extreme_high > 0.1))

print(f"\nExtreme Prediction Analysis:")
print(f"  Reasonable bounds: [{y_lower_bound:.2f}, {y_upper_bound:.2f}]")
print(f"  Proportion of prior draws with >10% extreme predictions: {any_extreme:.1%}")

# Check 4: Negative predictions (if Y represents positive quantity)
any_negative = np.mean(np.any(y_prior_pred_obs < 0, axis=1))
print(f"  Proportion of prior draws producing negative Y: {any_negative:.1%}")

# Check 5: Sigma plausibility
sigma_too_small = np.mean(sigma_prior < 0.01)
sigma_too_large = np.mean(sigma_prior > 1.0)

print(f"\nNoise Parameter (σ) Analysis:")
print(f"  Proportion σ < 0.01 (too small): {sigma_too_small:.1%}")
print(f"  Proportion σ > 1.0 (too large): {sigma_too_large:.1%}")
print(f"  Observed Y SD: {y_obs.std():.3f} (for reference)")

# ============================================================================
# 4. COMPUTE PRIOR PREDICTIVE INTERVALS
# ============================================================================
print(f"\n{'-'*80}")
print("4. COMPUTING PRIOR PREDICTIVE INTERVALS")
print(f"{'-'*80}")

# Percentiles for mean function
percentiles = [2.5, 5, 25, 50, 75, 95, 97.5]
mu_percentiles = np.percentile(mu_prior_pred_grid, percentiles, axis=0)

# Percentiles for predictions (with noise)
y_pred_grid = np.zeros((N_PRIOR_SAMPLES, len(x_grid)))
for i in range(N_PRIOR_SAMPLES):
    mu_i = beta0_prior[i] + beta1_prior[i] * log_x_grid
    y_pred_grid[i, :] = np.random.normal(mu_i, sigma_prior[i])

y_percentiles = np.percentile(y_pred_grid, percentiles, axis=0)

print(f"Computed prior predictive intervals at {len(percentiles)} percentiles")

# ============================================================================
# 5. SENSITIVITY METRICS
# ============================================================================
print(f"\n{'-'*80}")
print("5. SENSITIVITY ANALYSIS")
print(f"{'-'*80}")

# Prior informativeness: compare prior SD to posterior SD estimate
# (posterior SD will be roughly sd/sqrt(n) for linear models)
beta0_prior_sd = beta0_prior.std()
beta1_prior_sd = beta1_prior.std()

# Estimate effective sample size equivalent
# For normal prior with sd=s, equivalent to n observations with se=s
print(f"\nPrior Informativeness:")
print(f"  β₀ prior SD: {beta0_prior_sd:.3f}")
print(f"  β₁ prior SD: {beta1_prior_sd:.3f}")
print(f"  With {n_obs} observations, data should dominate if priors are weakly informative")

# Check correlation between parameters in prior
prior_corr = np.corrcoef(beta0_prior, beta1_prior)[0, 1]
print(f"\nPrior Parameter Correlation:")
print(f"  corr(β₀, β₁): {prior_corr:.3f} (should be ~0 for independent priors)")

# Save results to file
results = {
    'beta0_prior': beta0_prior,
    'beta1_prior': beta1_prior,
    'sigma_prior': sigma_prior,
    'y_prior_pred_obs': y_prior_pred_obs,
    'mu_prior_pred_grid': mu_prior_pred_grid,
    'x_grid': x_grid,
    'x_obs': x_obs,
    'y_obs': y_obs,
    'mu_percentiles': mu_percentiles,
    'y_percentiles': y_percentiles,
    'percentiles': percentiles
}

np.savez(OUTPUT_DIR / 'code' / 'prior_samples.npz', **results)
print(f"\n{'-'*80}")
print(f"Saved prior samples to: {OUTPUT_DIR / 'code' / 'prior_samples.npz'}")
print(f"{'='*80}")
