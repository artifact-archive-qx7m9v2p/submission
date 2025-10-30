"""
Setup CmdStan and test compilation
"""
import cmdstanpy

print("Checking CmdStan installation...")
print(f"CmdStan path: {cmdstanpy.cmdstan_path()}")

# Try to install if needed
try:
    cmdstanpy.install_cmdstan(overwrite=False, verbose=True)
    print("CmdStan installation verified")
except Exception as e:
    print(f"Installation check: {e}")

# Set up paths
import os
os.environ['MAKE'] = 'make'

print("\nTesting model compilation...")
from pathlib import Path

# Read the Stan model
stan_code = Path("/workspace/experiments/experiment_1/posterior_inference/code/model.stan").read_text()
print(f"Stan model loaded: {len(stan_code)} characters")

# Try to compile using stanc directly
from cmdstanpy.stanc import stanc
try:
    result = stanc(model_file="/workspace/experiments/experiment_1/posterior_inference/code/model.stan")
    print(f"✓ Stan model syntax valid")
except Exception as e:
    print(f"Syntax check failed: {e}")
