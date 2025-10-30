"""
Fit Asymptotic Exponential Model using Stan (CmdStanPy)
Adaptive sampling strategy: start conservative, diagnose, then adjust
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Setup paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DATA_PATH = Path("/workspace/data/data.csv")
STAN_FILE = BASE_DIR / "code" / "asymptotic_exponential.stan"
OUTPUT_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
N = len(data)
x = data['x'].values
y = data['Y'].values

print(f"Data: {N} observations")
print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# Prepare data for Stan
stan_data = {
    'N': N,
    'x': x,
    'y': y
}

# Compile model
print("\nCompiling Stan model...")
model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
print("Model compiled successfully!")

# PHASE 1: Initial probe (short chains to diagnose)
print("\n" + "="*60)
print("PHASE 1: Initial diagnostic probe")
print("="*60)
print("Running 4 chains x 200 iterations (100 warmup)")
print("Purpose: Quick assessment of model behavior")

try:
    fit_probe = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=100,
        iter_sampling=100,
        adapt_delta=0.95,
        seed=12345,
        show_console=True
    )

    # Check diagnostics
    print("\n--- Probe Diagnostics ---")
    diagnose = fit_probe.diagnose()
    print(diagnose)

    # Quick summary
    summary_probe = fit_probe.summary()
    print("\n--- Probe Parameter Summary ---")
    print(summary_probe[['Mean', 'StdDev', 'R_hat', 'ESS_bulk', 'ESS_tail']])

    # Check for issues
    max_rhat = summary_probe['R_hat'].max()
    min_ess_bulk = summary_probe['ESS_bulk'].min()
    num_divergences = fit_probe.num_divergences()

    print(f"\nProbe results:")
    print(f"  Max R-hat: {max_rhat:.4f}")
    print(f"  Min ESS (bulk): {min_ess_bulk:.1f}")
    print(f"  Divergences: {num_divergences}")

    probe_success = (max_rhat < 1.05) and (num_divergences < 10)

    if probe_success:
        print("\n✓ Probe successful - proceeding to main sampling")
    else:
        print("\n⚠ Probe shows issues - will try adjusted settings for main run")

except Exception as e:
    print(f"\n✗ Probe failed with error: {e}")
    print("This may indicate model initialization problems")
    raise

# PHASE 2: Main sampling (based on probe results)
print("\n" + "="*60)
print("PHASE 2: Main sampling")
print("="*60)

# Adjust settings based on probe
if num_divergences > 0:
    adapt_delta = 0.98
    print(f"Increasing adapt_delta to {adapt_delta} due to divergences")
else:
    adapt_delta = 0.95
    print(f"Using adapt_delta = {adapt_delta}")

# Run main chains
print(f"Running 4 chains x 2000 iterations (1000 warmup)")

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=adapt_delta,
    seed=12345,
    show_console=True
)

# Full diagnostics
print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

diagnose_output = fit.diagnose()
print(diagnose_output)

# Save diagnostics to file
with open(OUTPUT_DIR / "stan_diagnose.txt", "w") as f:
    f.write(diagnose_output)

# Parameter summary
summary = fit.summary()
print("\n--- Parameter Summary ---")
param_summary = summary.loc[['alpha', 'beta', 'gamma', 'sigma']]
print(param_summary[['Mean', 'StdDev', '5%', '50%', '95%', 'R_hat', 'ESS_bulk', 'ESS_tail']])

# Convergence metrics
max_rhat = param_summary['R_hat'].max()
min_ess_bulk = param_summary['ESS_bulk'].min()
min_ess_tail = param_summary['ESS_tail'].min()
num_divergences = fit.num_divergences()

print(f"\n--- Convergence Assessment ---")
print(f"Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"Min ESS (bulk): {min_ess_bulk:.1f} (target: > 400)")
print(f"Min ESS (tail): {min_ess_tail:.1f} (target: > 400)")
print(f"Total divergences: {num_divergences} (target: 0)")

# Overall assessment
converged = (max_rhat < 1.01) and (min_ess_bulk > 400) and (num_divergences == 0)

if converged:
    print("\n✓ CONVERGENCE ACHIEVED")
elif max_rhat < 1.05 and min_ess_bulk > 200:
    print("\n⚠ PARTIAL CONVERGENCE - usable but not ideal")
else:
    print("\n✗ CONVERGENCE FAILED")

# Save convergence metrics
convergence_metrics = {
    'max_rhat': float(max_rhat),
    'min_ess_bulk': float(min_ess_bulk),
    'min_ess_tail': float(min_ess_tail),
    'num_divergences': num_divergences,
    'converged': converged,
    'adapt_delta': adapt_delta
}

with open(OUTPUT_DIR / "convergence_metrics.json", "w") as f:
    json.dump(convergence_metrics, f, indent=2)

# Convert to ArviZ InferenceData with log_likelihood
print("\n--- Creating ArviZ InferenceData ---")
idata = az.from_cmdstanpy(
    fit,
    posterior_predictive=['y_rep'],
    log_likelihood='log_lik',
    observed_data={'y': y},
    coords={'obs_id': np.arange(N)},
    dims={
        'y': ['obs_id'],
        'y_rep': ['obs_id'],
        'log_lik': ['obs_id']
    }
)

# Add x as constant data for plotting
idata.add_groups({'constant_data': {'x': x}})

# Save InferenceData
print(f"Saving to {OUTPUT_DIR / 'posterior_inference.netcdf'}")
idata.to_netcdf(OUTPUT_DIR / "posterior_inference.netcdf")

# Save parameter summary
param_summary.to_csv(OUTPUT_DIR / "parameter_summary.csv")

print("\n--- Model Fit Metrics ---")
# Calculate R² and RMSE
y_pred_samples = idata.posterior['mu'].values.reshape(-1, N)
y_pred_mean = y_pred_samples.mean(axis=0)

ss_res = np.sum((y - y_pred_mean)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean((y - y_pred_mean)**2))

print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save fit metrics
fit_metrics = {
    'r2': float(r2),
    'rmse': float(rmse),
    'n_obs': N
}

with open(OUTPUT_DIR / "fit_metrics.json", "w") as f:
    json.dump(fit_metrics, f, indent=2)

print("\n" + "="*60)
print("MODEL FITTING COMPLETE")
print("="*60)
print(f"ArviZ file: {OUTPUT_DIR / 'posterior_inference.netcdf'}")
print(f"Diagnostics: {OUTPUT_DIR}")
