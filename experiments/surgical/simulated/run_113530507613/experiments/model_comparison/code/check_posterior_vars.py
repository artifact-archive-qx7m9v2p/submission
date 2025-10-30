"""Check posterior variable names"""
import arviz as az

print("Experiment 1 posterior vars:")
idata1 = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
print(list(idata1.posterior.data_vars.keys()))

print("\nExperiment 2 posterior vars:")
idata2 = az.from_netcdf('/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf')
print(list(idata2.posterior.data_vars.keys()))
