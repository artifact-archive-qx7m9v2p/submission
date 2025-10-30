"""Check for constant data"""
import arviz as az
import pandas as pd

idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print("Checking for constant_data group:")
if hasattr(idata, 'constant_data'):
    print(idata.constant_data)
else:
    print("No constant_data group found")

# Let's just load from CSV
data = pd.read_csv('/workspace/data/data.csv')
print("\nData from CSV:")
print(data)
