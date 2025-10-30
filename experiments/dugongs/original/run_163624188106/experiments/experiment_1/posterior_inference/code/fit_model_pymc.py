"""
Fit Bayesian Log-Log Linear Model to real data using HMC (PyMC)

Model: log(Y) ~ Normal(alpha + beta * log(x), sigma)
Priors: alpha ~ N(0.6, 0.3), beta ~ N(0.13, 0.1), sigma ~ Half-N(0.1)

Note: Using PyMC as fallback because CmdStanPy requires compilation tools not available.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(f"Y range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]")
print(f"x range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")

# Transform to log scale
log_Y = np.log(df['Y'].values)
log_x = np.log(df['x'].values)

print(f"\nlog(Y) range: [{log_Y.min():.3f}, {log_Y.max():.3f}]")
print(f"log(x) range: [{log_x.min():.3f}, {log_x.max():.3f}]")

# Build PyMC model
print("\n" + "="*70)
print("Building PyMC model")
print("="*70)

with pm.Model() as model:
    # Priors
    alpha = pm.Normal("alpha", mu=0.6, sigma=0.3)
    beta = pm.Normal("beta", mu=0.13, sigma=0.1)
    sigma = pm.HalfNormal("sigma", sigma=0.1)

    # Linear model in log-log space
    mu = alpha + beta * log_x

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=log_Y)

print(model)

# Initial probe: short run to check for issues
print("\n" + "="*70)
print("PHASE 1: Initial probe (200 iterations)")
print("="*70)

with model:
    probe_trace = pm.sample(
        draws=100,
        tune=100,
        chains=4,
        target_accept=0.8,
        random_seed=12345,
        progressbar=True,
        return_inferencedata=True
    )

# Check probe diagnostics
probe_summary = az.summary(probe_trace, var_names=["alpha", "beta", "sigma"])
print("\nProbe diagnostics:")
print(probe_summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])

probe_divergences = probe_trace.sample_stats.diverging.sum().values
print(f"\nProbe divergences: {probe_divergences}")

# Decide on main sampling strategy
if probe_divergences > 0:
    print("\nDivergences detected in probe. Increasing target_accept to 0.95")
    target_accept = 0.95
else:
    print("\nNo divergences in probe. Using target_accept = 0.8")
    target_accept = 0.8

# Main sampling
print("\n" + "="*70)
print("PHASE 2: Main sampling (2000 iterations)")
print("="*70)

with model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=target_accept,
        random_seed=12345,
        progressbar=True,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

    # Generate posterior predictive samples
    print("\nGenerating posterior predictive samples...")
    pm.sample_posterior_predictive(trace, model=model, extend_inferencedata=True, random_seed=12345)

# Add observed data to InferenceData
trace.observed_data["Y"] = ("obs_id", df['Y'].values)

# Save diagnostics
print("\n" + "="*70)
print("DIAGNOSTICS")
print("="*70)

# Summary statistics
summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])
print("\nParameter summary:")
print(summary)

# Save summary to file
with open(DIAGNOSTICS_DIR / "convergence_summary.txt", 'w') as f:
    f.write("CONVERGENCE DIAGNOSTICS\n")
    f.write("="*70 + "\n\n")

    # Divergences
    divergences = trace.sample_stats.diverging.sum().values
    f.write(f"Divergent transitions: {divergences}\n")

    # Max treedepth
    max_td = (trace.sample_stats.tree_depth == trace.sample_stats.tree_depth.max()).sum().values
    f.write(f"Max treedepth hits: {max_td}\n\n")

    # Parameter summaries
    f.write("PARAMETER SUMMARY\n")
    f.write("-"*70 + "\n")
    f.write(summary.to_string())
    f.write("\n\n")

    # Convergence criteria check
    f.write("CONVERGENCE CRITERIA\n")
    f.write("-"*70 + "\n")

    max_rhat = summary['r_hat'].max()
    min_ess_bulk = summary['ess_bulk'].min()
    min_ess_tail = summary['ess_tail'].min()

    f.write(f"Max R_hat: {max_rhat:.4f} (target: < 1.01)\n")
    f.write(f"Min ESS_bulk: {min_ess_bulk:.1f} (target: > 400)\n")
    f.write(f"Min ESS_tail: {min_ess_tail:.1f} (target: > 400)\n")
    f.write(f"Divergences: {divergences} (target: < 10)\n")

    # Pass/fail
    convergence_pass = (
        max_rhat < 1.01 and
        min_ess_bulk > 400 and
        min_ess_tail > 400 and
        divergences < 10
    )

    f.write(f"\nCONVERGENCE STATUS: {'PASS' if convergence_pass else 'FAIL'}\n")

print(f"\nDivergent transitions: {divergences}")
print(f"Max R_hat: {max_rhat:.4f}")
print(f"Min ESS_bulk: {min_ess_bulk:.1f}")
print(f"Min ESS_tail: {min_ess_tail:.1f}")

# Save InferenceData
print(f"\nSaving InferenceData to {DIAGNOSTICS_DIR / 'posterior_inference.netcdf'}")
trace.to_netcdf(DIAGNOSTICS_DIR / "posterior_inference.netcdf")

# Verify log_likelihood group exists
print("\nInferenceData groups:", list(trace.groups()))
if 'log_likelihood' in trace.groups():
    print("✓ log_likelihood group present")
    print(f"  Shape: {trace.log_likelihood.y_obs.shape}")
else:
    print("✗ WARNING: log_likelihood group missing!")

# Compute LOO
print("\n" + "="*70)
print("LOO CROSS-VALIDATION")
print("="*70)

loo = az.loo(trace, pointwise=True)
print(loo)

# Extract Pareto k diagnostics
pareto_k = loo.pareto_k.values
pareto_good = np.sum(pareto_k < 0.5)
pareto_ok = np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))
pareto_bad = np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))
pareto_very_bad = np.sum(pareto_k >= 1.0)

print(f"\nPareto k diagnostics:")
print(f"  k < 0.5 (good): {pareto_good}/{len(pareto_k)} ({100*pareto_good/len(pareto_k):.1f}%)")
print(f"  0.5 ≤ k < 0.7 (ok): {pareto_ok}/{len(pareto_k)} ({100*pareto_ok/len(pareto_k):.1f}%)")
print(f"  0.7 ≤ k < 1.0 (bad): {pareto_bad}/{len(pareto_k)} ({100*pareto_bad/len(pareto_k):.1f}%)")
print(f"  k ≥ 1.0 (very bad): {pareto_very_bad}/{len(pareto_k)} ({100*pareto_very_bad/len(pareto_k):.1f}%)")

# Save LOO results - Fixed to handle ELPDData object correctly
loo_results = {
    "elpd_loo": float(loo.elpd_loo),
    "se": float(loo.se),
    "p_loo": float(loo.p_loo),
    "pareto_k_stats": {
        "good_k_lt_0.5": int(pareto_good),
        "ok_k_0.5_to_0.7": int(pareto_ok),
        "bad_k_0.7_to_1.0": int(pareto_bad),
        "very_bad_k_gte_1.0": int(pareto_very_bad),
        "percent_good": float(100 * pareto_good / len(pareto_k)),
        "max_k": float(pareto_k.max()),
        "mean_k": float(pareto_k.mean())
    }
}

with open(DIAGNOSTICS_DIR / "loo_results.json", 'w') as f:
    json.dump(loo_results, f, indent=2)

print(f"\nLOO results saved to {DIAGNOSTICS_DIR / 'loo_results.json'}")

# Compute R-squared manually in log scale
print("\nVerifying posterior_predictive group...")
if 'posterior_predictive' in trace.groups():
    y_obs_pred = trace.posterior_predictive.y_obs.mean(dim=["chain", "draw"]).values
    residuals = log_Y - y_obs_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_Y - log_Y.mean())**2)
    r_squared = 1 - ss_res / ss_tot
else:
    # Compute manually from posterior parameters
    alpha_mean = trace.posterior.alpha.mean().values
    beta_mean = trace.posterior.beta.mean().values
    y_pred_mean = alpha_mean + beta_mean * log_x
    residuals = log_Y - y_pred_mean
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_Y - log_Y.mean())**2)
    r_squared = 1 - ss_res / ss_tot

print("\n" + "="*70)
print("MODEL FIT")
print("="*70)
print(f"R² (log scale): {r_squared:.3f}")

# Final assessment
print("\n" + "="*70)
print("FINAL ASSESSMENT")
print("="*70)

convergence_ok = max_rhat < 1.01 and min_ess_bulk > 400 and divergences < 10
pareto_ok = (pareto_good + pareto_ok) / len(pareto_k) > 0.9
r_squared_ok = r_squared > 0.85

print(f"Convergence: {'PASS' if convergence_ok else 'FAIL'}")
print(f"LOO Pareto k: {'PASS' if pareto_ok else 'FAIL'}")
print(f"R²: {'PASS' if r_squared_ok else 'FAIL'}")
print(f"\nOVERALL: {'PASS' if (convergence_ok and pareto_ok and r_squared_ok) else 'PASS/CONDITIONAL' if convergence_ok else 'FAIL'}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
