"""
Fix Experiment 2 InferenceData by moving log_lik to log_likelihood group
"""
import arviz as az
import xarray as xr
import numpy as np

print("Loading Experiment 2 InferenceData...")
idata2_path = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf'
idata2 = az.from_netcdf(idata2_path)

print(f"Current groups: {idata2.groups()}")
print(f"Posterior variables: {list(idata2.posterior.data_vars.keys())}")

# Check if log_lik exists in posterior
if 'log_lik' in idata2.posterior.data_vars:
    print("\nFound 'log_lik' in posterior group")
    print(f"Shape: {idata2.posterior['log_lik'].shape}")
    print(f"Dims: {idata2.posterior['log_lik'].dims}")

    # Extract log_lik and create log_likelihood group
    log_lik = idata2.posterior['log_lik']

    # Create a new dataset for log_likelihood group
    log_likelihood_ds = xr.Dataset({'y': log_lik})

    # Add the log_likelihood group to idata2
    idata2.add_groups(log_likelihood=log_likelihood_ds)

    print(f"\nAdded log_likelihood group")
    print(f"Updated groups: {idata2.groups()}")

    # Verify it worked
    if 'log_likelihood' in idata2.groups():
        print(f"Log likelihood vars: {list(idata2.log_likelihood.data_vars)}")
        print("SUCCESS: log_likelihood group created")

        # Save the updated InferenceData
        print(f"\nSaving updated InferenceData to: {idata2_path}")
        idata2.to_netcdf(idata2_path)
        print("DONE!")
    else:
        print("ERROR: Failed to add log_likelihood group")
else:
    print("ERROR: 'log_lik' not found in posterior group")
