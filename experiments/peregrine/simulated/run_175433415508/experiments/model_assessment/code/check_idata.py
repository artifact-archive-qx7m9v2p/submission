import arviz as az

idata = az.from_netcdf("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
print("Groups:", list(idata.groups()))
print("\nPosterior variables:", list(idata.posterior.data_vars))
print("\nLog likelihood variables:", list(idata.log_likelihood.data_vars) if 'log_likelihood' in idata.groups() else "No log_likelihood group")
