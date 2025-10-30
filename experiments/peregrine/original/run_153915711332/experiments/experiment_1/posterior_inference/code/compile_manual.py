"""
Manually compile and run Stan model using CmdStan binaries
"""
import subprocess
import os
from pathlib import Path

# Paths
cmdstan_home = Path("/tmp/agent-home/.cmdstan/cmdstan-2.37.0")
model_dir = Path("/workspace/experiments/experiment_1/posterior_inference/code")
model_stan = model_dir / "model.stan"
model_hpp = model_dir / "model.hpp"
model_exe = model_dir / "model"

print("Attempting manual compilation...")
print(f"CmdStan home: {cmdstan_home}")
print(f"Model file: {model_stan}")

# Step 1: Use stanc to translate .stan to .hpp
stanc_path = cmdstan_home / "bin" / "linux-stanc"
print(f"\nStep 1: Running stanc...")
print(f"stanc path: {stanc_path}, exists: {stanc_path.exists()}")

# Make stanc executable
try:
    os.chmod(stanc_path, 0o755)
    print(f"Made stanc executable")
except Exception as e:
    print(f"chmod failed: {e}")

try:
    result = subprocess.run(
        [str(stanc_path), str(model_stan), "--o", str(model_hpp)],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"✓ stanc succeeded")
    if result.stdout:
        print(f"  Output: {result.stdout[:200]}")
except subprocess.CalledProcessError as e:
    print(f"✗ stanc failed: {e}")
    print(f"  stderr: {e.stderr}")
    print(f"  stdout: {e.stdout}")
except Exception as e:
    print(f"✗ stanc error: {e}")

# Check if .hpp was created
if model_hpp.exists():
    print(f"✓ Generated {model_hpp}")
    print(f"  Size: {model_hpp.stat().st_size} bytes")
else:
    print(f"✗ {model_hpp} was not created")
