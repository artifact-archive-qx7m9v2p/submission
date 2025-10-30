"""Check InferenceData structure"""
import arviz as az

idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print("InferenceData structure:")
print(idata)

print("\n\nGroups available:")
for group in idata.groups():
    print(f"\n{group}:")
    print(getattr(idata, group))
