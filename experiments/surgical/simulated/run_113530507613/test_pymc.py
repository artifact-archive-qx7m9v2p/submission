import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import pymc as pm
import numpy as np
print("PyMC imported successfully!")
print(f"PyMC version: {pm.__version__}")

# Simple test
with pm.Model() as model:
    x = pm.Normal('x', 0, 1)
print("Simple model created successfully!")
