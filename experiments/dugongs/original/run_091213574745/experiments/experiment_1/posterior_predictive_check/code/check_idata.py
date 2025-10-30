import arviz as az
import numpy as np

# Load InferenceData
idata = az.from_netcdf("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")

print("InferenceData groups:", list(idata.groups()))
print("\nPosterior variables:", list(idata.posterior.data_vars))
print("\nPosterior_predictive variables:", list(idata.posterior_predictive.data_vars))
print("\nLog_likelihood variables:", list(idata.log_likelihood.data_vars))
print("\nObserved_data variables:", list(idata.observed_data.data_vars))

print("\n\nPosterior_predictive shape:")
for var in idata.posterior_predictive.data_vars:
    print(f"  {var}: {idata.posterior_predictive[var].shape}")
