"""
Load from pickle and save properly to NetCDF
"""
import pickle
import arviz as az
import xarray as xr
import numpy as np

print("Loading from pickle...")
with open('/workspace/experiments/experiment_2/posterior_inference/diagnostics/idata.pkl', 'rb') as f:
    idata = pickle.load(f)

print(f"Groups: {idata.groups()}")
print(f"Posterior vars: {list(idata.posterior.data_vars.keys())}")

# Get log_lik from posterior
log_lik = idata.posterior['log_lik']
print(f"\nlog_lik shape: {log_lik.shape}")
print(f"log_lik dims: {log_lik.dims}")

# Create new InferenceData with log_likelihood group
# We need to copy to separate log_likelihood group

# Make a copy of log_lik with proper dimension name
log_lik_renamed = log_lik.rename({'log_lik_dim_0': 'y_dim_0'})

# Create log_likelihood dataset
log_likelihood = xr.Dataset({'y': log_lik_renamed})

# Create new InferenceData
idata_new = az.InferenceData(
    posterior=idata.posterior,
    sample_stats=idata.sample_stats,
    observed_data=idata.observed_data,
    log_likelihood=log_likelihood
)

print(f"\nNew InferenceData groups: {idata_new.groups()}")

# Save
output_path = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf'
print(f"\nSaving to {output_path}...")
idata_new.to_netcdf(output_path)

print("Verifying...")
idata_check = az.from_netcdf(output_path)
print(f"Loaded groups: {idata_check.groups()}")
if 'log_likelihood' in idata_check.groups():
    print("SUCCESS!")
else:
    print("FAILED!")
