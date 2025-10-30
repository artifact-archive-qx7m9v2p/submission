"""
Fit Bayesian Log-Log Linear Model to real data using HMC (CmdStanPy)

Model: log(Y) ~ Normal(alpha + beta * log(x), sigma)
Priors: alpha ~ N(0.6, 0.3), beta ~ N(0.13, 0.1), sigma ~ Half-N(0.1)
"""

import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import arviz as az
import json
from pathlib import Path

# Paths
DATA_PATH = "/workspace/data/data.csv"
MODEL_PATH = "/workspace/experiments/experiment_1/simulation_based_validation/code/model.stan"
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
CODE_DIR = OUTPUT_DIR / "code"

# Create directories
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(f"Y range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]")
print(f"x range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")

# Prepare Stan data
stan_data = {
    'N': len(df),
    'x': df['x'].values.tolist(),
    'Y': df['Y'].values.tolist()
}

# Compile model
print("\nCompiling Stan model...")
model = CmdStanModel(stan_file=MODEL_PATH)

# Initial probe: short run to check for issues
print("\n" + "="*70)
print("PHASE 1: Initial probe (200 iterations)")
print("="*70)

probe_fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=100,
    iter_sampling=100,
    adapt_delta=0.8,
    seed=12345,
    show_console=True
)

# Check probe diagnostics
probe_summary = probe_fit.summary()
print("\nProbe diagnostics:")
print(probe_summary[['Mean', 'StdDev', 'R_hat', 'ESS_bulk', 'ESS_tail']])

probe_divergences = probe_fit.num_unconstrained_divergences()
print(f"\nProbe divergences: {probe_divergences}")

# Decide on main sampling strategy
if probe_divergences > 0:
    print("\nDivergences detected in probe. Increasing adapt_delta to 0.95")
    adapt_delta = 0.95
else:
    print("\nNo divergences in probe. Using adapt_delta = 0.8")
    adapt_delta = 0.8

# Main sampling
print("\n" + "="*70)
print("PHASE 2: Main sampling (2000 iterations)")
print("="*70)

fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=adapt_delta,
    seed=12345,
    show_console=True
)

# Save diagnostics
print("\n" + "="*70)
print("DIAGNOSTICS")
print("="*70)

# Summary statistics
summary = fit.summary()
print("\nParameter summary:")
print(summary[['Mean', 'StdDev', 'R_hat', 'ESS_bulk', 'ESS_tail']])

# Save summary to file
with open(DIAGNOSTICS_DIR / "convergence_summary.txt", 'w') as f:
    f.write("CONVERGENCE DIAGNOSTICS\n")
    f.write("="*70 + "\n\n")

    # Divergences
    divergences = fit.num_unconstrained_divergences()
    f.write(f"Divergent transitions: {divergences}\n")

    # Max treedepth
    max_td = fit.num_max_treedepth()
    f.write(f"Max treedepth warnings: {max_td}\n\n")

    # Parameter summaries
    f.write("PARAMETER SUMMARY\n")
    f.write("-"*70 + "\n")
    f.write(summary.to_string())
    f.write("\n\n")

    # Convergence criteria check
    f.write("CONVERGENCE CRITERIA\n")
    f.write("-"*70 + "\n")

    max_rhat = summary['R_hat'].max()
    min_ess_bulk = summary['ESS_bulk'].min()
    min_ess_tail = summary['ESS_tail'].min()

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
print(f"Max treedepth warnings: {max_td}")
print(f"Max R_hat: {max_rhat:.4f}")
print(f"Min ESS_bulk: {min_ess_bulk:.1f}")
print(f"Min ESS_tail: {min_ess_tail:.1f}")

# Convert to ArviZ InferenceData with log_likelihood
print("\nConverting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    posterior_predictive=["y_pred", "log_y_pred"],
    log_likelihood="log_lik",
    observed_data={"Y": stan_data['Y']},
    coords={"obs_id": np.arange(stan_data['N'])},
    dims={"log_lik": ["obs_id"], "y_pred": ["obs_id"], "log_y_pred": ["obs_id"]}
)

# Save InferenceData
print(f"\nSaving InferenceData to {DIAGNOSTICS_DIR / 'posterior_inference.netcdf'}")
idata.to_netcdf(DIAGNOSTICS_DIR / "posterior_inference.netcdf")

# Verify log_likelihood group exists
print("\nInferenceData groups:", list(idata.groups()))
if 'log_likelihood' in idata.groups():
    print("✓ log_likelihood group present")
else:
    print("✗ WARNING: log_likelihood group missing!")

# Compute LOO
print("\n" + "="*70)
print("LOO CROSS-VALIDATION")
print("="*70)

loo = az.loo(idata, pointwise=True)
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

# Save LOO results
loo_results = {
    "loo": float(loo.loo),
    "loo_se": float(loo.loo_se),
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

# Extract R-squared
r_squared_samples = fit.stan_variable("R_squared")
r_squared_mean = r_squared_samples.mean()
r_squared_std = r_squared_samples.std()
r_squared_q = np.percentile(r_squared_samples, [2.5, 50, 97.5])

print("\n" + "="*70)
print("MODEL FIT")
print("="*70)
print(f"Bayesian R²: {r_squared_mean:.3f} ± {r_squared_std:.3f}")
print(f"95% CI: [{r_squared_q[0]:.3f}, {r_squared_q[2]:.3f}]")

# Final assessment
print("\n" + "="*70)
print("FINAL ASSESSMENT")
print("="*70)

convergence_ok = max_rhat < 1.01 and min_ess_bulk > 400 and divergences < 10
pareto_ok = (pareto_good + pareto_ok) / len(pareto_k) > 0.9
r_squared_ok = r_squared_mean > 0.85

print(f"Convergence: {'PASS' if convergence_ok else 'FAIL'}")
print(f"LOO Pareto k: {'PASS' if pareto_ok else 'FAIL'}")
print(f"R²: {'PASS' if r_squared_ok else 'FAIL'}")
print(f"\nOVERALL: {'PASS' if (convergence_ok and pareto_ok and r_squared_ok) else 'FAIL'}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
