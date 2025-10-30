"""Check what's in both InferenceData objects"""
import arviz as az

print("Checking Experiment 1:")
idata1 = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
print(f"Groups: {idata1.groups()}")
if 'log_likelihood' in idata1.groups():
    print(f"Log likelihood vars: {list(idata1.log_likelihood.data_vars)}")

print("\n" + "="*80 + "\n")

print("Checking Experiment 2:")
idata2 = az.from_netcdf('/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf')
print(f"Groups: {idata2.groups()}")
if 'log_likelihood' in idata2.groups():
    print(f"Log likelihood vars: {list(idata2.log_likelihood.data_vars)}")
else:
    print("WARNING: No log_likelihood group found!")
    print("\nAttempting to check if likelihood data exists elsewhere...")
    print(f"Available groups: {idata2.groups()}")

    # Check posterior for likelihood-related variables
    if 'posterior' in idata2.groups():
        print(f"\nPosterior variables: {list(idata2.posterior.data_vars.keys())}")
