#!/usr/bin/env python
import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

# Now run the actual SBC script
exec(open('/workspace/experiments/experiment_1/simulation_based_validation/code/run_sbc_pymc.py').read())
