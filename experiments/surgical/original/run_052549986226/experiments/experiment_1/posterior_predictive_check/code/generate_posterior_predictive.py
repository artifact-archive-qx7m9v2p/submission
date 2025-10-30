"""
Generate Posterior Predictive Samples from Fitted Model
Experiment 1: Beta-Binomial Model

Since the original InferenceData doesn't have posterior_predictive saved,
we regenerate them from the posterior samples of mu and kappa.

Author: Bayesian Model Validation Specialist
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
import arviz as az
from scipy import stats

# Set random seed
np.random.seed(42)

print("=" * 80)
print("GENERATING POSTERIOR PREDICTIVE SAMPLES")
print("=" * 80)

# Load data
print("\n[1] Loading observed data...")
data = pd.read_csv('/workspace/data/data.csv')
n_trials = data['n_trials'].values
r_obs = data['r_successes'].values
n_groups = len(data)

print(f"  - {n_groups} groups")
print(f"  - Total trials: {n_trials.sum()}")
print(f"  - Total successes: {r_obs.sum()}")

# Load posterior samples
print("\n[2] Loading posterior samples...")
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

# Extract mu and kappa samples
mu_samples = idata.posterior['mu'].values  # shape: (chains, draws)
kappa_samples = idata.posterior['kappa'].values  # shape: (chains, draws)

# Flatten to get all samples
mu_flat = mu_samples.flatten()
kappa_flat = kappa_samples.flatten()

n_samples = len(mu_flat)
print(f"  - Loaded {n_samples} posterior samples")

# Generate posterior predictive samples
print("\n[3] Generating posterior predictive samples...")
print("  - Using Beta-Binomial distribution")
print("  - For each MCMC sample, generate y_rep[i] for all groups")

y_rep = np.zeros((n_samples, n_groups), dtype=int)

for s in range(n_samples):
    if s % 1000 == 0:
        print(f"  - Progress: {s}/{n_samples} samples...")

    mu_s = mu_flat[s]
    kappa_s = kappa_flat[s]

    # Convert to alpha, beta parameterization
    alpha_s = mu_s * kappa_s
    beta_s = (1 - mu_s) * kappa_s

    # For each group, generate posterior predictive sample
    for i in range(n_groups):
        # Beta-Binomial: first draw p ~ Beta(alpha, beta), then y ~ Binomial(n, p)
        p_i = stats.beta.rvs(alpha_s, beta_s)
        y_rep[s, i] = stats.binom.rvs(n_trials[i], p_i)

print(f"  - Generated {n_samples} x {n_groups} posterior predictive samples")
print(f"  - Shape: {y_rep.shape}")

# Save posterior predictive samples
print("\n[4] Saving posterior predictive samples...")

# Save as CSV for easy access
y_rep_df = pd.DataFrame(
    y_rep,
    columns=[f'group_{i}' for i in range(1, n_groups + 1)]
)
y_rep_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/results/posterior_predictive_samples.csv',
                index=False)

print(f"  - Saved to posterior_predictive_samples.csv")

# Add to InferenceData and save updated version
print("\n[5] Adding to InferenceData...")

# Reshape y_rep back to (chains, draws, groups)
n_chains = mu_samples.shape[0]
n_draws = mu_samples.shape[1]
y_rep_reshaped = y_rep.reshape(n_chains, n_draws, n_groups)

# Create posterior_predictive group
import xarray as xr

posterior_predictive = xr.Dataset({
    'y_rep': xr.DataArray(
        y_rep_reshaped,
        dims=['chain', 'draw', 'group_dim'],
        coords={
            'chain': idata.posterior.chain,
            'draw': idata.posterior.draw,
            'group_dim': range(n_groups)
        }
    )
})

# Add to idata
idata.add_groups(posterior_predictive=posterior_predictive)

# Save updated InferenceData
idata.to_netcdf('/workspace/experiments/experiment_1/posterior_predictive_check/results/posterior_inference_with_ppc.netcdf')

print(f"  - Updated InferenceData saved")
print(f"  - Groups now: {list(idata.groups())}")

print("\n[COMPLETE] Posterior predictive samples generated successfully!")
print(f"  - Samples: {n_samples} x {n_groups}")
print(f"  - CSV: posterior_predictive_samples.csv")
print(f"  - NetCDF: posterior_inference_with_ppc.netcdf")
