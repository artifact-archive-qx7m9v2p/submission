#!/bin/bash
# Run SBC with correct Python path

export PYTHONPATH="/tmp/agent-home/.local/lib/python3.13/site-packages:$PYTHONPATH"
python /workspace/experiments/experiment_2/simulation_based_validation/code/sbc_validation.py
