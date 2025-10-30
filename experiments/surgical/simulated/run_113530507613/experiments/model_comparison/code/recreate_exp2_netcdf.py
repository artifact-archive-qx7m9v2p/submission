"""
Recreate Experiment 2 netcdf from pickle file with proper log_likelihood group
"""
import pickle
import arviz as az
import xarray as xr

print("Loading Experiment 2 InferenceData from pickle...")
idata2_pkl_path = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/idata.pkl'
idata2_netcdf_path = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf'

with open(idata2_pkl_path, 'rb') as f:
    idata2 = pickle.load(f)

print(f"Loaded groups: {idata2.groups()}")
print(f"Posterior variables: {list(idata2.posterior.data_vars.keys())}")

# Check if log_lik exists in posterior
if 'log_lik' in idata2.posterior.data_vars:
    print("\nFound 'log_lik' in posterior group")
    print(f"Shape: {idata2.posterior['log_lik'].shape}")

    # Extract log_lik and create log_likelihood group
    log_lik = idata2.posterior['log_lik'].values  # Get actual values

    # Create a new dataset for log_likelihood group
    log_likelihood_ds = xr.Dataset(
        {'y': (['chain', 'draw', 'y_dim_0'], log_lik)},
        coords={
            'chain': idata2.posterior.coords['chain'],
            'draw': idata2.posterior.coords['draw'],
            'y_dim_0': range(log_lik.shape[2])
        }
    )

    # Add the log_likelihood group
    idata2.add_groups(log_likelihood=log_likelihood_ds)

    print(f"\nAdded log_likelihood group")
    print(f"Updated groups: {idata2.groups()}")

    # Save to netcdf
    print(f"\nSaving to: {idata2_netcdf_path}")
    idata2.to_netcdf(idata2_netcdf_path)

    print("SUCCESS!")

    # Verify
    print("\nVerifying the saved file...")
    idata2_check = az.from_netcdf(idata2_netcdf_path)
    print(f"Groups: {idata2_check.groups()}")
    if 'log_likelihood' in idata2_check.groups():
        print("✓ log_likelihood group present")
        print(f"  Variables: {list(idata2_check.log_likelihood.data_vars)}")
    else:
        print("✗ log_likelihood group missing!")

else:
    print("ERROR: 'log_lik' not found in posterior group")
