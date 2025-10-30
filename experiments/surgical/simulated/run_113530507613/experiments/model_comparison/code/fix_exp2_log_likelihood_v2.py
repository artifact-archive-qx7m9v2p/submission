"""
Fix Experiment 2 InferenceData by creating a new file with log_likelihood group
"""
import arviz as az
import xarray as xr
import numpy as np

print("Loading Experiment 2 InferenceData...")
idata2_path = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf'
idata2_fixed_path = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference_fixed.netcdf'

idata2 = az.from_netcdf(idata2_path)

print(f"Current groups: {idata2.groups()}")

# Load all data into memory to avoid lazy loading issues
print("\nLoading all data into memory...")
idata2.posterior.load()
idata2.sample_stats.load()
idata2.observed_data.load()

# Extract log_lik and create log_likelihood group
log_lik = idata2.posterior['log_lik'].copy()

# Create a new dataset for log_likelihood group
log_likelihood_ds = xr.Dataset({'y': log_lik})

# Add the log_likelihood group
idata2.add_groups(log_likelihood=log_likelihood_ds)

print(f"Updated groups: {idata2.groups()}")

# Save to new file
print(f"\nSaving to: {idata2_fixed_path}")
idata2.to_netcdf(idata2_fixed_path)

print("SUCCESS! Created fixed file.")

# Replace the original
print(f"\nReplacing original file...")
import shutil
shutil.move(idata2_fixed_path, idata2_path)
print("DONE!")
