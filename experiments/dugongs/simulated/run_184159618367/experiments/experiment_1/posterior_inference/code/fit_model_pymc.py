"""
Fit Asymptotic Exponential Model using PyMC
Fallback from Stan due to compilation issues
Adaptive sampling strategy: start conservative, diagnose, then adjust
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Setup paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DATA_PATH = Path("/workspace/data/data.csv")
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

# Build PyMC model
print("\nBuilding PyMC model...")
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=2.55, sigma=0.1)
    beta = pm.Normal('beta', mu=0.9, sigma=0.2)
    gamma = pm.Gamma('gamma', alpha=4, beta=20)  # E[gamma] = 4/20 = 0.2
    sigma = pm.HalfCauchy('sigma', beta=0.15)

    # Mean function
    mu = pm.Deterministic('mu', alpha - beta * pm.math.exp(-gamma * x))

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Posterior predictive for checks
    y_rep = pm.Normal('y_rep', mu=mu, sigma=sigma, shape=N)

print("Model built successfully!")

# PHASE 1: Initial probe (short chains to diagnose)
print("\n" + "="*60)
print("PHASE 1: Initial diagnostic probe")
print("="*60)
print("Running 4 chains x 200 iterations (100 tune)")
print("Purpose: Quick assessment of model behavior")

with model:
    try:
        trace_probe = pm.sample(
            draws=100,
            tune=100,
            chains=4,
            cores=4,
            target_accept=0.95,
            random_seed=12345,
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True}
        )

        # Check diagnostics
        print("\n--- Probe Diagnostics ---")
        summary_probe = az.summary(trace_probe, var_names=['alpha', 'beta', 'gamma', 'sigma'])
        print(summary_probe[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']])

        # Check for issues
        max_rhat = summary_probe['r_hat'].max()
        min_ess_bulk = summary_probe['ess_bulk'].min()
        num_divergences = trace_probe.sample_stats['diverging'].sum().item()

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
        num_divergences = 999  # Flag for adjustment
        probe_success = False

# PHASE 2: Main sampling (based on probe results)
print("\n" + "="*60)
print("PHASE 2: Main sampling")
print("="*60)

# Adjust settings based on probe
if num_divergences > 0:
    target_accept = 0.98
    print(f"Increasing target_accept to {target_accept} due to divergences")
else:
    target_accept = 0.95
    print(f"Using target_accept = {target_accept}")

# Run main chains
print(f"Running 4 chains x 2000 iterations (1000 tune)")

with model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=target_accept,
        random_seed=12345,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )

# Full diagnostics
print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

# Parameter summary
summary = az.summary(trace, var_names=['alpha', 'beta', 'gamma', 'sigma'])
print("\n--- Parameter Summary ---")
print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']])

# Convergence metrics
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()
num_divergences = trace.sample_stats['diverging'].sum().item()

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
    'num_divergences': int(num_divergences),
    'converged': bool(converged),
    'target_accept': target_accept
}

with open(OUTPUT_DIR / "convergence_metrics.json", "w") as f:
    json.dump(convergence_metrics, f, indent=2)

# Add x as constant data for plotting
trace.add_groups({'constant_data': {'x': x}})

# Save InferenceData
print(f"\nSaving to {OUTPUT_DIR / 'posterior_inference.netcdf'}")
trace.to_netcdf(OUTPUT_DIR / "posterior_inference.netcdf")

# Save parameter summary
summary.to_csv(OUTPUT_DIR / "parameter_summary.csv")

print("\n--- Model Fit Metrics ---")
# Calculate R² and RMSE
y_pred_samples = trace.posterior['mu'].values.reshape(-1, N)
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

# Calculate LOO for model comparison
print("\n--- LOO-CV ---")
try:
    loo = az.loo(trace, var_name='y_obs')
    print(f"LOO: {loo.loo:.2f}")
    print(f"LOO SE: {loo.se:.2f}")

    # Save LOO results
    with open(OUTPUT_DIR / "loo_results.txt", "w") as f:
        f.write(str(loo))
except Exception as e:
    print(f"LOO calculation failed: {e}")

print("\n" + "="*60)
print("MODEL FITTING COMPLETE")
print("="*60)
print(f"ArviZ file: {OUTPUT_DIR / 'posterior_inference.netcdf'}")
print(f"Diagnostics: {OUTPUT_DIR}")
