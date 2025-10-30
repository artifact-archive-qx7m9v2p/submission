"""
Investigate the posterior predictive samples to understand the scale
"""
import arviz as az
import pandas as pd
import numpy as np

# Load data
print("Loading data...")
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
data = pd.read_csv('/workspace/data/data.csv')

# Extract data
y_pred = idata.posterior_predictive.y_obs.values
y_obs_from_idata = idata.observed_data.y_obs.values
y_obs_Y = idata.observed_data.Y.values
y_obs_csv = data.Y.values

print("\nPosterior predictive samples (y_obs):")
print(f"  Shape: {y_pred.shape}")
print(f"  Mean: {np.mean(y_pred):.4f}")
print(f"  Std: {np.std(y_pred):.4f}")
print(f"  Min: {np.min(y_pred):.4f}")
print(f"  Max: {np.max(y_pred):.4f}")
print(f"  First few values (first draw): {y_pred[0, 0, :5]}")

print("\nObserved data from InferenceData (y_obs):")
print(f"  Shape: {y_obs_from_idata.shape}")
print(f"  Values: {y_obs_from_idata[:5]}")

print("\nObserved data from InferenceData (Y):")
print(f"  Shape: {y_obs_Y.shape}")
print(f"  Values: {y_obs_Y[:5]}")

print("\nObserved data from CSV:")
print(f"  Shape: {y_obs_csv.shape}")
print(f"  Values: {y_obs_csv[:5]}")

print("\nChecking if y_pred is in log scale:")
print(f"  exp(mean(y_pred)): {np.exp(np.mean(y_pred)):.4f}")
print(f"  mean(exp(y_pred)): {np.mean(np.exp(y_pred)):.4f}")
print(f"  mean(Y): {np.mean(y_obs_csv):.4f}")

print("\nIt appears y_pred is in LOG SCALE!")
print("We need to transform back to original scale for comparison.")
