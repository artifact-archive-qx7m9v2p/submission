"""
Prior Predictive Check for Experiment 1: Log-Log Linear Model

Model:
  log(Y_i) ~ Normal(mu_i, sigma)
  mu_i = alpha + beta * log(x_i)

Priors:
  alpha ~ Normal(0.6, 0.3)
  beta ~ Normal(0.13, 0.1)
  sigma ~ Half-Normal(0.1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_SAMPLES = 1000
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check/plots"
DATA_PATH = "/workspace/data/data.csv"

# Load observed data
data = pd.read_csv(DATA_PATH)
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(x_obs)

print("=" * 60)
print("PRIOR PREDICTIVE CHECK: Log-Log Linear Model")
print("=" * 60)
print(f"\nObserved data:")
print(f"  N = {n_obs}")
print(f"  x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"  Y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"  Y mean: {y_obs.mean():.2f}, SD: {y_obs.std():.2f}")

# ============================================================================
# 1. Sample from priors
# ============================================================================
print("\n" + "=" * 60)
print("1. SAMPLING FROM PRIORS")
print("=" * 60)

# Prior specifications
alpha_prior = {'mean': 0.6, 'sd': 0.3}
beta_prior = {'mean': 0.13, 'sd': 0.1}
sigma_prior = {'scale': 0.1}  # Half-Normal scale parameter

# Sample from priors
alpha_samples = np.random.normal(alpha_prior['mean'], alpha_prior['sd'], N_PRIOR_SAMPLES)
beta_samples = np.random.normal(beta_prior['mean'], beta_prior['sd'], N_PRIOR_SAMPLES)
# Half-Normal: sample from |N(0, scale)|
sigma_samples = np.abs(np.random.normal(0, sigma_prior['scale'], N_PRIOR_SAMPLES))

print(f"\nPrior samples (N={N_PRIOR_SAMPLES}):")
print(f"\nalpha ~ Normal({alpha_prior['mean']}, {alpha_prior['sd']}):")
print(f"  Range: [{alpha_samples.min():.3f}, {alpha_samples.max():.3f}]")
print(f"  Mean: {alpha_samples.mean():.3f}, SD: {alpha_samples.std():.3f}")

print(f"\nbeta ~ Normal({beta_prior['mean']}, {beta_prior['sd']}):")
print(f"  Range: [{beta_samples.min():.3f}, {beta_samples.max():.3f}]")
print(f"  Mean: {beta_samples.mean():.3f}, SD: {beta_samples.std():.3f}")

print(f"\nsigma ~ Half-Normal({sigma_prior['scale']}):")
print(f"  Range: [{sigma_samples.min():.3f}, {sigma_samples.max():.3f}]")
print(f"  Mean: {sigma_samples.mean():.3f}, SD: {sigma_samples.std():.3f}")

# ============================================================================
# 2. Generate prior predictive datasets
# ============================================================================
print("\n" + "=" * 60)
print("2. GENERATING PRIOR PREDICTIVE DATA")
print("=" * 60)

# Store all prior predictive samples
y_pred_samples = np.zeros((N_PRIOR_SAMPLES, n_obs))

# For each prior draw, generate data
for i in range(N_PRIOR_SAMPLES):
    # Linear predictor on log scale
    log_x = np.log(x_obs)
    mu_log = alpha_samples[i] + beta_samples[i] * log_x

    # Sample from Normal on log scale
    log_y = np.random.normal(mu_log, sigma_samples[i])

    # Transform back to original scale
    y_pred_samples[i, :] = np.exp(log_y)

# Calculate summary statistics
y_pred_min = y_pred_samples.min(axis=1)
y_pred_max = y_pred_samples.max(axis=1)
y_pred_mean = y_pred_samples.mean(axis=1)
y_pred_median = np.median(y_pred_samples, axis=1)

print(f"\nPrior predictive statistics across {N_PRIOR_SAMPLES} datasets:")
print(f"\nDataset minimums:")
print(f"  Range: [{y_pred_min.min():.3f}, {y_pred_min.max():.3f}]")
print(f"  Median: {np.median(y_pred_min):.3f}")

print(f"\nDataset maximums:")
print(f"  Range: [{y_pred_max.min():.3f}, {y_pred_max.max():.3f}]")
print(f"  Median: {np.median(y_pred_max):.3f}")

print(f"\nDataset means:")
print(f"  Range: [{y_pred_mean.min():.3f}, {y_pred_mean.max():.3f}]")
print(f"  Median: {np.median(y_pred_mean):.3f}")
print(f"  Observed mean: {y_obs.mean():.3f}")

# ============================================================================
# 3. Check for domain violations and computational issues
# ============================================================================
print("\n" + "=" * 60)
print("3. DOMAIN VIOLATIONS AND COMPUTATIONAL ISSUES")
print("=" * 60)

# Check for extreme values
n_negative = np.sum(y_pred_samples < 0)
n_extreme = np.sum(y_pred_samples > 100)
n_very_small = np.sum(y_pred_samples < 0.1)
n_pathological = n_negative + n_extreme

print(f"\nPathological values:")
print(f"  Negative Y: {n_negative} / {N_PRIOR_SAMPLES * n_obs} ({100*n_negative/(N_PRIOR_SAMPLES*n_obs):.2f}%)")
print(f"  Y > 100: {n_extreme} / {N_PRIOR_SAMPLES * n_obs} ({100*n_extreme/(N_PRIOR_SAMPLES*n_obs):.2f}%)")
print(f"  Y < 0.1: {n_very_small} / {N_PRIOR_SAMPLES * n_obs} ({100*n_very_small/(N_PRIOR_SAMPLES*n_obs):.2f}%)")

# Datasets with any pathological values
datasets_with_issues = np.sum((y_pred_samples < 0) | (y_pred_samples > 100), axis=1) > 0
n_bad_datasets = np.sum(datasets_with_issues)
print(f"\nDatasets with pathological values: {n_bad_datasets} / {N_PRIOR_SAMPLES} ({100*n_bad_datasets/N_PRIOR_SAMPLES:.1f}%)")

# ============================================================================
# 4. Check coverage of observed data
# ============================================================================
print("\n" + "=" * 60)
print("4. COVERAGE OF OBSERVED DATA")
print("=" * 60)

# Does prior predictive range cover observed data?
covers_min = np.sum(y_pred_min <= y_obs.min())
covers_max = np.sum(y_pred_max >= y_obs.max())
covers_both = np.sum((y_pred_min <= y_obs.min()) & (y_pred_max >= y_obs.max()))

print(f"\nRange coverage:")
print(f"  Datasets covering min({y_obs.min():.2f}): {covers_min} / {N_PRIOR_SAMPLES} ({100*covers_min/N_PRIOR_SAMPLES:.1f}%)")
print(f"  Datasets covering max({y_obs.max():.2f}): {covers_max} / {N_PRIOR_SAMPLES} ({100*covers_max/N_PRIOR_SAMPLES:.1f}%)")
print(f"  Datasets covering full range: {covers_both} / {N_PRIOR_SAMPLES} ({100*covers_both/N_PRIOR_SAMPLES:.1f}%)")

# Check if observed mean is within prior predictive distribution
obs_mean = y_obs.mean()
obs_sd = y_obs.std()
mean_within_1sd = np.sum(np.abs(y_pred_mean - obs_mean) <= obs_sd)
mean_within_2sd = np.sum(np.abs(y_pred_mean - obs_mean) <= 2*obs_sd)

print(f"\nMean similarity:")
print(f"  Observed mean: {obs_mean:.3f} (SD: {obs_sd:.3f})")
print(f"  Prior datasets with mean within 1 SD: {mean_within_1sd} / {N_PRIOR_SAMPLES} ({100*mean_within_1sd/N_PRIOR_SAMPLES:.1f}%)")
print(f"  Prior datasets with mean within 2 SD: {mean_within_2sd} / {N_PRIOR_SAMPLES} ({100*mean_within_2sd/N_PRIOR_SAMPLES:.1f}%)")

# ============================================================================
# 5. Calculate derived quantities at key x values
# ============================================================================
print("\n" + "=" * 60)
print("5. DERIVED QUANTITIES AT KEY X VALUES")
print("=" * 60)

# Evaluate predictions at x = 1, 10, 30
x_eval = np.array([1.0, 10.0, 30.0])
y_eval_samples = np.zeros((N_PRIOR_SAMPLES, len(x_eval)))

for i in range(N_PRIOR_SAMPLES):
    log_x_eval = np.log(x_eval)
    mu_log_eval = alpha_samples[i] + beta_samples[i] * log_x_eval
    log_y_eval = np.random.normal(mu_log_eval, sigma_samples[i])
    y_eval_samples[i, :] = np.exp(log_y_eval)

for j, x_val in enumerate(x_eval):
    y_vals = y_eval_samples[:, j]
    print(f"\nY at x={x_val}:")
    print(f"  Median: {np.median(y_vals):.3f}")
    print(f"  95% interval: [{np.percentile(y_vals, 2.5):.3f}, {np.percentile(y_vals, 97.5):.3f}]")
    print(f"  Range: [{y_vals.min():.3f}, {y_vals.max():.3f}]")

    # Find closest observed x
    closest_idx = np.argmin(np.abs(x_obs - x_val))
    closest_x = x_obs[closest_idx]
    closest_y = y_obs[closest_idx]
    print(f"  Closest observed: x={closest_x:.1f}, Y={closest_y:.2f}")

# ============================================================================
# 6. Calculate implied power law parameters
# ============================================================================
print("\n" + "=" * 60)
print("6. IMPLIED POWER LAW PARAMETERS")
print("=" * 60)

# Model: Y = exp(alpha) * x^beta
intercept_implied = np.exp(alpha_samples)
exponent_implied = beta_samples

print(f"\nImplied: Y = A * x^B")
print(f"\nIntercept (A = exp(alpha)):")
print(f"  Median: {np.median(intercept_implied):.3f}")
print(f"  95% interval: [{np.percentile(intercept_implied, 2.5):.3f}, {np.percentile(intercept_implied, 97.5):.3f}]")
print(f"  EDA estimate: 1.82")

print(f"\nExponent (B = beta):")
print(f"  Median: {np.median(exponent_implied):.3f}")
print(f"  95% interval: [{np.percentile(exponent_implied, 2.5):.3f}, {np.percentile(exponent_implied, 97.5):.3f}]")
print(f"  EDA estimate: 0.13")

# Save results for plotting
results = {
    'alpha_samples': alpha_samples,
    'beta_samples': beta_samples,
    'sigma_samples': sigma_samples,
    'y_pred_samples': y_pred_samples,
    'y_eval_samples': y_eval_samples,
    'x_eval': x_eval,
    'intercept_implied': intercept_implied,
    'exponent_implied': exponent_implied,
    'x_obs': x_obs,
    'y_obs': y_obs
}

print("\n" + "=" * 60)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("=" * 60)
print(f"\nResults saved. Proceeding to visualization...")

# Save results to disk for use in plotting
np.savez('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.npz', **results)
