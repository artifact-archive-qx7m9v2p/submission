"""Generate posterior predictive samples for both models."""

import numpy as np
import arviz as az
import xarray as xr

# Paths
MODEL1_PATH = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf'
MODEL2_PATH = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf'

# Load models
idata1 = az.from_netcdf(MODEL1_PATH)
idata2 = az.from_netcdf(MODEL2_PATH)

# Get data
y_obs = idata1.observed_data['y_obs'].values
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])  # From data
n_studies = len(y_obs)

print("Generating posterior predictive samples...")

# Model 1: Fixed-effect
posterior1 = idata1.posterior
theta = posterior1['theta'].values  # (chains, draws)
n_chains, n_draws = theta.shape

y_pred1 = np.zeros((n_chains, n_draws, n_studies))
for i in range(n_studies):
    y_pred1[:, :, i] = np.random.normal(theta, sigma[i])

# Add to idata1
y_pred1_da = xr.DataArray(
    y_pred1,
    dims=['chain', 'draw', 'y_obs_dim_0'],
    coords={'chain': posterior1.chain, 'draw': posterior1.draw, 'y_obs_dim_0': range(n_studies)}
)
idata1.add_groups({'posterior_predictive': xr.Dataset({'y_obs': y_pred1_da})})

# Model 2: Random-effects
posterior2 = idata2.posterior
theta_i = posterior2['theta'].values  # (chains, draws, n_studies)

y_pred2 = np.zeros((n_chains, n_draws, n_studies))
for i in range(n_studies):
    y_pred2[:, :, i] = np.random.normal(theta_i[:, :, i], sigma[i])

# Add to idata2
y_pred2_da = xr.DataArray(
    y_pred2,
    dims=['chain', 'draw', 'y_obs_dim_0'],
    coords={'chain': posterior2.chain, 'draw': posterior2.draw, 'y_obs_dim_0': range(n_studies)}
)
idata2.add_groups({'posterior_predictive': xr.Dataset({'y_obs': y_pred2_da})})

# Save updated models
idata1.to_netcdf('/workspace/experiments/model_comparison/idata1_with_predictions.netcdf')
idata2.to_netcdf('/workspace/experiments/model_comparison/idata2_with_predictions.netcdf')

print("Posterior predictive samples generated and saved.")
print(f"Model 1: {y_pred1.shape}")
print(f"Model 2: {y_pred2.shape}")
