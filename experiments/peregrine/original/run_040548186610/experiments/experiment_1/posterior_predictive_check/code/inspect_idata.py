"""Quick script to inspect InferenceData structure"""
import arviz as az

idata = az.from_netcdf("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")

print("InferenceData groups:", list(idata.groups()))
print("\n" + "="*80)

if 'posterior_predictive' in idata.groups():
    print("Posterior Predictive variables:")
    print(idata.posterior_predictive)
    print("\nVariable names:", list(idata.posterior_predictive.data_vars))
else:
    print("No posterior_predictive group found")

print("\n" + "="*80)
print("Posterior variables:")
print(idata.posterior)
print("\nVariable names:", list(idata.posterior.data_vars))

print("\n" + "="*80)
if 'observed_data' in idata.groups():
    print("Observed data:")
    print(idata.observed_data)
