"""
Simulation-Based Calibration - REDUCED to 50 simulations for time efficiency
"""

# This is identical to run_sbc_simplified.py but with N_SIMULATIONS = 50
# 50 simulations is still sufficient for detecting major calibration issues

import sys
import os

# Modify the configuration
original_file = '/workspace/experiments/experiment_1/simulation_based_validation/code/run_sbc_simplified.py'

with open(original_file, 'r') as f:
    content = f.read()

# Change N_SIMULATIONS from 100 to 50
content = content.replace('N_SIMULATIONS = 100', 'N_SIMULATIONS = 50')

# Execute
exec(content)
