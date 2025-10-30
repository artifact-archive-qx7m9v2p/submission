"""
Fit Logarithmic Model with Student-t Likelihood using MCMC

This script:
1. Loads the real data
2. Fits the Stan model with proper ν truncation
3. Runs convergence diagnostics
4. Compares to Model 1 (Normal likelihood)
5. Saves InferenceData with log_likelihood for LOO-CV
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set random seed
np.random.seed(42)

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
MODEL_PATH = Path("/workspace/experiments/experiment_2/posterior_inference/code/student_t_log_model.stan")
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/posterior_inference")
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"Data shape: {data.shape}")
print(f"x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Prepare data for Stan
stan_data = {
    'N': len(data),
    'x': data['x'].values,
    'Y': data['Y'].values
}

# Compile model
print("\nCompiling Stan model...")
model = cmdstanpy.CmdStanModel(stan_file=str(MODEL_PATH))
print("Model compiled successfully")

# Initial probe: Short chains to assess behavior
print("\n" + "="*60)
print("PHASE 1: Initial Probe (200 iterations)")
print("="*60)

probe_fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=100,
    iter_sampling=100,
    adapt_delta=0.95,
    max_treedepth=10,
    show_console=True,
    seed=42
)

# Quick diagnostics on probe
print("\n--- Probe Diagnostics ---")
probe_summary = probe_fit.summary()
print("\nParameter estimates (probe):")
print(probe_summary[['Mean', 'StdDev', 'N_Eff', 'R_hat']].round(4))

# Check for major issues
probe_divergences = probe_fit.divergences
probe_max_treedepth = probe_fit.max_treedepths
print(f"\nDivergences: {probe_divergences}")
print(f"Max treedepth hits: {probe_max_treedepth}")

# Get ν estimate from probe
nu_probe = probe_summary.loc[probe_summary.index.str.contains('nu'), 'Mean'].values[0]
print(f"\nProbe ν estimate: {nu_probe:.2f}")

# Decide on main sampling strategy
if probe_divergences > 20:
    print("\n⚠ WARNING: Many divergences in probe. Increasing adapt_delta to 0.99")
    adapt_delta_main = 0.99
    max_treedepth_main = 12
else:
    print("\n✓ Probe looks reasonable. Proceeding with adapt_delta=0.95")
    adapt_delta_main = 0.95
    max_treedepth_main = 10

# Main sampling
print("\n" + "="*60)
print("PHASE 2: Main Sampling (2000 iterations)")
print("="*60)

main_fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=adapt_delta_main,
    max_treedepth=max_treedepth_main,
    show_console=True,
    seed=42
)

# Save raw fit
print("\nSaving Stan output...")
main_fit.save_csvfiles(dir=str(DIAGNOSTICS_DIR))

# Comprehensive diagnostics
print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

summary_df = main_fit.summary()
print("\nParameter Summary:")
print(summary_df[['Mean', 'StdDev', '5%', '50%', '95%', 'N_Eff', 'R_hat']].round(4))

# Check convergence criteria
rhat_max = summary_df['R_hat'].max()
ess_min = summary_df['N_Eff'].min()
divergences = main_fit.divergences
max_td = main_fit.max_treedepths

print(f"\n--- Convergence Checks ---")
print(f"Max R̂: {rhat_max:.4f} (target: < 1.01)")
print(f"Min ESS: {ess_min:.0f} (target: > 400)")
print(f"Divergences: {divergences} (target: < 1% = {40})")
print(f"Max treedepth hits: {max_td}")

convergence_pass = (rhat_max < 1.01) and (ess_min > 400) and (divergences < 40)
print(f"\nOverall convergence: {'✓ PASS' if convergence_pass else '✗ FAIL'}")

# Extract key parameter: ν
nu_summary = summary_df.loc[summary_df.index.str.contains('nu')]
nu_mean = nu_summary['Mean'].values[0]
nu_median = nu_summary['50%'].values[0]
nu_ci_low = nu_summary['5%'].values[0]
nu_ci_high = nu_summary['95%'].values[0]

print("\n" + "="*60)
print("KEY PARAMETER: ν (Degrees of Freedom)")
print("="*60)
print(f"Mean: {nu_mean:.2f}")
print(f"Median: {nu_median:.2f}")
print(f"95% CI: [{nu_ci_low:.2f}, {nu_ci_high:.2f}]")

if nu_mean > 30:
    print("\n⚠ INTERPRETATION: ν > 30 suggests Normal likelihood is adequate")
    print("   Recommendation: Prefer Model 1 (simpler)")
elif nu_mean < 20:
    print("\n✓ INTERPRETATION: ν < 20 suggests heavy tails are justified")
    print("   Recommendation: Student-t robustness may be valuable")
else:
    print("\n○ INTERPRETATION: ν ∈ [20, 30] is borderline")
    print("   Recommendation: Check LOO comparison to decide")

# Create ArviZ InferenceData with log_likelihood
print("\n" + "="*60)
print("Creating ArviZ InferenceData")
print("="*60)

idata = az.from_cmdstanpy(
    main_fit,
    posterior_predictive=['y_pred', 'y_rep'],
    log_likelihood='log_lik',
    observed_data={'Y': stan_data['Y']},
    coords={'obs_id': np.arange(stan_data['N'])},
    dims={'Y': ['obs_id'], 'y_pred': ['obs_id'], 'y_rep': ['obs_id'], 'log_lik': ['obs_id']}
)

# Save InferenceData
idata_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
az.to_netcdf(idata, idata_path)
print(f"InferenceData saved to: {idata_path}")

# Compute LOO-CV
print("\n" + "="*60)
print("LOO-CV Computation")
print("="*60)

loo_result = az.loo(idata, pointwise=True)
print(loo_result)

# Save LOO result
loo_dict = {
    'elpd_loo': float(loo_result.elpd_loo),
    'se': float(loo_result.se),
    'p_loo': float(loo_result.p_loo),
    'n_samples': int(loo_result.n_samples),
    'n_data_points': int(loo_result.n_data_points),
    'warning': bool(loo_result.warning),
    'pareto_k_threshold': 0.7
}

with open(DIAGNOSTICS_DIR / "loo_result.json", 'w') as f:
    json.dump(loo_dict, f, indent=2)

# Check Pareto k values
pareto_k = loo_result.pareto_k
print(f"\nPareto k diagnostics:")
print(f"  Max k: {pareto_k.max():.3f}")
print(f"  k > 0.5: {(pareto_k > 0.5).sum()}/{len(pareto_k)}")
print(f"  k > 0.7: {(pareto_k > 0.7).sum()}/{len(pareto_k)}")

# Compare to Model 1
print("\n" + "="*60)
print("COMPARISON TO MODEL 1")
print("="*60)

model1_idata_path = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
if model1_idata_path.exists():
    print("Loading Model 1 InferenceData...")
    idata_model1 = az.from_netcdf(model1_idata_path)

    # Compute LOO for Model 1
    loo_model1 = az.loo(idata_model1)

    # Compare
    loo_compare = az.compare({'Model_1_Normal': idata_model1, 'Model_2_StudentT': idata})
    print("\nLOO Comparison:")
    print(loo_compare)

    # Save comparison
    loo_compare.to_csv(DIAGNOSTICS_DIR / "loo_comparison.csv")

    # Interpret ΔLOO
    delta_loo = loo_result.elpd_loo - loo_model1.elpd_loo
    delta_loo_se = np.sqrt(loo_result.se**2 + loo_model1.se**2)

    print(f"\nΔLOO = {delta_loo:.2f} ± {delta_loo_se:.2f}")

    if delta_loo > 2:
        print("✓ Model 2 (Student-t) substantially better")
    elif delta_loo < -2:
        print("✗ Model 2 worse (overfitting?)")
    else:
        print("○ Models equivalent (|ΔLOO| < 2)")
        print("  Recommendation: Prefer simpler Model 1")
else:
    print("⚠ Model 1 results not found. Skipping comparison.")

# Summary statistics
print("\n" + "="*60)
print("POSTERIOR SUMMARY")
print("="*60)

az_summary = az.summary(idata, var_names=['beta_0', 'beta_1', 'sigma', 'nu'], round_to=4)
print(az_summary)

# Save summary
az_summary.to_csv(DIAGNOSTICS_DIR / "posterior_summary.csv")

print("\n" + "="*60)
print("Fitting Complete!")
print("="*60)
print(f"\nKey results:")
print(f"  ν = {nu_mean:.2f} [{nu_ci_low:.2f}, {nu_ci_high:.2f}]")
print(f"  LOO-ELPD = {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
print(f"  Convergence: {'✓ PASS' if convergence_pass else '✗ FAIL'}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
