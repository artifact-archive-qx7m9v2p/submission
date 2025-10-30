"""Check observed data variable names"""
import arviz as az

print("Experiment 1:")
idata1 = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
print(f"Observed data vars: {list(idata1.observed_data.data_vars)}")

print("\nExperiment 2:")
idata2 = az.from_netcdf('/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf')
print(f"Observed data vars: {list(idata2.observed_data.data_vars)}")
