import arviz as az
import numpy as np
from pathlib import Path

# Load InferenceData
idata = az.from_netcdf(Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"))

# Check what's in posterior_predictive
print("Posterior predictive variables:", list(idata.posterior_predictive.data_vars))
print("Shape:", idata.posterior_predictive['y_obs'].shape)

# Sample values
y_pred_sample = idata.posterior_predictive['y_obs'].values[0, 0, :5]
print("Sample values:", y_pred_sample)

# Check observed data
print("\nObserved data variables:", list(idata.observed_data.data_vars))
print("Y shape:", idata.observed_data['Y'].shape)
print("Y sample values:", idata.observed_data['Y'].values[:5])
print("y_obs shape:", idata.observed_data['y_obs'].shape)
print("y_obs sample values:", idata.observed_data['y_obs'].values[:5])
