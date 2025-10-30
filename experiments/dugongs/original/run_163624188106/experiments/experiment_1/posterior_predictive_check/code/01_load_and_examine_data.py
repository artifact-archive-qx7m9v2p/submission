"""
Load and examine posterior inference data and observed data
"""
import arviz as az
import pandas as pd
import numpy as np

# Load posterior inference data
print("Loading InferenceData from netcdf...")
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print("\nInferenceData structure:")
print(idata)

print("\nPosterior variables:")
print(idata.posterior.data_vars)

print("\nPosterior predictive variables:")
if hasattr(idata, 'posterior_predictive'):
    print(idata.posterior_predictive.data_vars)
else:
    print("No posterior_predictive group found")

print("\nObserved data variables:")
if hasattr(idata, 'observed_data'):
    print(idata.observed_data.data_vars)
else:
    print("No observed_data group found")

# Load observed data from CSV
print("\n\nLoading observed data from CSV...")
data = pd.read_csv('/workspace/data/data.csv')
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"\nFirst few rows:")
print(data.head(10))
print(f"\nSummary statistics:")
print(data.describe())

# Check dimensions
print("\n\nDimensions check:")
print(f"Observed data points: {len(data)}")
if hasattr(idata, 'posterior_predictive') and 'y_pred' in idata.posterior_predictive:
    y_pred_shape = idata.posterior_predictive.y_pred.shape
    print(f"Posterior predictive shape: {y_pred_shape}")
    print(f"Number of posterior draws: {y_pred_shape[0] * y_pred_shape[1]}")
