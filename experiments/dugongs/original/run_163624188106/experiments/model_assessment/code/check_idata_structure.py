import arviz as az
from pathlib import Path

# Load InferenceData
idata = az.from_netcdf(Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"))

print("Groups in InferenceData:")
print(idata.groups())

if hasattr(idata, 'observed_data'):
    print("\nObserved data variables:")
    print(list(idata.observed_data.data_vars))

if hasattr(idata, 'posterior_predictive'):
    print("\nPosterior predictive variables:")
    print(list(idata.posterior_predictive.data_vars))

if hasattr(idata, 'log_likelihood'):
    print("\nLog likelihood variables:")
    print(list(idata.log_likelihood.data_vars))
