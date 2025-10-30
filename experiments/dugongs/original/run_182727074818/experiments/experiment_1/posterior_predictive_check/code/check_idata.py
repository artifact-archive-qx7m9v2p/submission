import arviz as az

idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print("Groups:", list(idata.groups()))
print("\nPosterior variables:", list(idata.posterior.data_vars))
print("\nPosterior_predictive variables:", list(idata.posterior_predictive.data_vars))
print("\nObserved_data variables:", list(idata.observed_data.data_vars))

if 'posterior_predictive' in idata.groups():
    print("\nPosterior_predictive dataset:")
    print(idata.posterior_predictive)
