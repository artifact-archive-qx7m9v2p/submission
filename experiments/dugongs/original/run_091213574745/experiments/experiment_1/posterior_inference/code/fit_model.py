"""
Fit Logarithmic Model with Normal Likelihood using MCMC (CmdStanPy)

Model: Y_i ~ Normal(β₀ + β₁*log(x_i), σ)
Priors:
  β₀ ~ Normal(2.3, 0.3)
  β₁ ~ Normal(0.29, 0.15)
  σ ~ Exponential(10)
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
STAN_FILE = Path("/workspace/experiments/experiment_1/posterior_inference/code/logarithmic_model.stan")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LOGARITHMIC MODEL - POSTERIOR INFERENCE")
print("=" * 80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"   - Observations: n = {len(data)}")
print(f"   - x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"   - Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Prepare data for Stan
stan_data = {
    'N': len(data),
    'x': data['x'].values,
    'Y': data['Y'].values
}

# Compile Stan model
print("\n2. Compiling Stan model...")
model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
print(f"   - Model compiled successfully")

# Fit model with MCMC
print("\n3. Running MCMC sampling...")
print("   - Chains: 4")
print("   - Iterations per chain: 2000 (1000 warmup + 1000 sampling)")
print("   - Adapt delta: 0.95")
print("   - Starting sampling...")

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    seed=42,
    show_progress=True,
    show_console=True
)

print("\n   - Sampling completed!")

# Check for divergences
divergences = fit.method_variables()['divergent__'].sum()
total_iterations = 4 * 1000  # 4 chains * 1000 sampling iterations
div_pct = 100 * divergences / total_iterations

print(f"\n4. Quick Diagnostics:")
print(f"   - Divergent transitions: {divergences} ({div_pct:.2f}%)")

# Save CmdStanPy summary
print("\n5. Saving CmdStanPy diagnostics...")
summary_df = fit.summary()
summary_df.to_csv(DIAG_DIR / "cmdstan_summary.csv")
print(f"   - Saved to: {DIAG_DIR / 'cmdstan_summary.csv'}")

# Create ArviZ InferenceData
print("\n6. Converting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    posterior_predictive=['y_pred', 'y_rep'],
    log_likelihood='log_lik',
    observed_data={'Y': stan_data['Y']},
    coords={'obs': np.arange(stan_data['N'])},
    dims={
        'log_lik': ['obs'],
        'y_pred': ['obs'],
        'y_rep': ['obs']
    }
)

# Save InferenceData
idata_path = DIAG_DIR / "posterior_inference.netcdf"
idata.to_netcdf(idata_path)
print(f"   - Saved to: {idata_path}")

# Compute ArviZ summary
print("\n7. Computing ArviZ summary statistics...")
az_summary = az.summary(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma'],
    kind='stats'
)
az_summary.to_csv(DIAG_DIR / "arviz_summary.csv")
print(f"   - Saved to: {DIAG_DIR / 'arviz_summary.csv'}")

# Display summary
print("\n" + "=" * 80)
print("PARAMETER ESTIMATES")
print("=" * 80)
print(az_summary.to_string())

# Check convergence criteria
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

rhat_max = az_summary['r_hat'].max()
ess_bulk_min = az_summary['ess_bulk'].min()
ess_tail_min = az_summary['ess_tail'].min()

print(f"\nR-hat (max):        {rhat_max:.4f}  [Target: < 1.01]")
print(f"ESS bulk (min):     {ess_bulk_min:.0f}    [Target: > 400]")
print(f"ESS tail (min):     {ess_tail_min:.0f}    [Target: > 400]")
print(f"Divergences:        {divergences}       [Target: 0, Acceptable: < {0.01 * total_iterations:.0f}]")

# Determine convergence status
convergence_pass = True
warnings = []
failures = []

if rhat_max > 1.01:
    failures.append(f"R-hat > 1.01 (max = {rhat_max:.4f})")
    convergence_pass = False

if ess_bulk_min < 400:
    failures.append(f"ESS bulk < 400 (min = {ess_bulk_min:.0f})")
    convergence_pass = False

if ess_tail_min < 400:
    failures.append(f"ESS tail < 400 (min = {ess_tail_min:.0f})")
    convergence_pass = False

if div_pct > 5:
    failures.append(f"Divergences > 5% ({div_pct:.2f}%)")
    convergence_pass = False
elif div_pct > 1:
    warnings.append(f"Divergences 1-5% ({div_pct:.2f}%)")

# Check E-BFMI
ebfmi = fit.method_variables()['energy__']
ebfmi_values = []
for chain in range(4):
    chain_energy = ebfmi[chain * 1000:(chain + 1) * 1000]
    ebfmi_val = np.var(np.diff(chain_energy)) / np.var(chain_energy)
    ebfmi_values.append(ebfmi_val)

ebfmi_min = np.min(ebfmi_values)
print(f"E-BFMI (min):       {ebfmi_min:.3f}    [Target: > 0.3]")

if ebfmi_min < 0.3:
    warnings.append(f"E-BFMI < 0.3 (min = {ebfmi_min:.3f})")

print("\n" + "-" * 80)
if convergence_pass and not warnings:
    print("STATUS: ALL CONVERGENCE CHECKS PASSED")
elif convergence_pass and warnings:
    print("STATUS: CONVERGENCE PASSED WITH WARNINGS")
    for w in warnings:
        print(f"  - WARNING: {w}")
else:
    print("STATUS: CONVERGENCE FAILED")
    for f in failures:
        print(f"  - FAILURE: {f}")
    for w in warnings:
        print(f"  - WARNING: {w}")

print("=" * 80)

# Save convergence report
convergence_data = {
    'rhat_max': float(rhat_max),
    'ess_bulk_min': float(ess_bulk_min),
    'ess_tail_min': float(ess_tail_min),
    'divergences': int(divergences),
    'divergence_pct': float(div_pct),
    'ebfmi_min': float(ebfmi_min),
    'convergence_pass': convergence_pass,
    'warnings': warnings,
    'failures': failures
}

with open(DIAG_DIR / "convergence_metrics.json", 'w') as f:
    json.dump(convergence_data, f, indent=2)

print(f"\n8. Convergence metrics saved to: {DIAG_DIR / 'convergence_metrics.json'}")
print("\nFitting complete!")
