"""
Fit Logarithmic Model with Student-t Likelihood using PyMC

This script:
1. Loads the real data
2. Fits the model with proper ν truncation using PyMC
3. Runs convergence diagnostics
4. Compares to Model 1 (Normal likelihood)
5. Saves InferenceData with log_likelihood for LOO-CV
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set random seed
np.random.seed(42)

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/posterior_inference")
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"Data shape: {data.shape}")
print(f"x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Prepare data
x_obs = data['x'].values
Y_obs = data['Y'].values
log_x_obs = np.log(x_obs)
N = len(data)

print(f"\nN = {N}")

# Build PyMC model
print("\n" + "="*60)
print("Building PyMC Model")
print("="*60)

with pm.Model() as student_t_model:
    # Data containers
    x = pm.Data('x', log_x_obs)
    Y = pm.Data('Y', Y_obs)

    # Priors
    beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.29, sigma=0.15)
    sigma = pm.Exponential('sigma', lam=10)

    # Truncated nu: Gamma(2, 0.1) with lower bound at 3
    # PyMC uses rate parameterization for Gamma: Gamma(alpha, beta) where mean = alpha/beta
    # Our prior: alpha=2, rate=0.1 -> mean = 2/0.1 = 20
    nu_raw = pm.Gamma('nu', alpha=2, beta=0.1)
    nu = pm.Deterministic('nu_truncated', pm.math.maximum(nu_raw, 3.0))

    # Mean function
    mu = pm.Deterministic('mu', beta_0 + beta_1 * x)

    # Student-t likelihood
    Y_obs_rv = pm.StudentT('Y_obs', nu=nu, mu=mu, sigma=sigma, observed=Y)

    # Posterior predictive
    y_pred = pm.Deterministic('y_pred', mu)

print("\nModel structure:")
print(student_t_model)

# Initial probe: Short chains to assess behavior
print("\n" + "="*60)
print("PHASE 1: Initial Probe (200 iterations)")
print("="*60)

with student_t_model:
    probe_trace = pm.sample(
        draws=100,
        tune=100,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )

# Quick diagnostics on probe
print("\n--- Probe Diagnostics ---")
probe_summary = az.summary(probe_trace, var_names=['beta_0', 'beta_1', 'sigma', 'nu_truncated'])
print(probe_summary)

nu_probe_mean = probe_summary.loc['nu_truncated', 'mean']
print(f"\nProbe ν estimate: {nu_probe_mean:.2f}")

# Check for divergences
probe_divergences = probe_trace.sample_stats.diverging.sum().values
print(f"Divergences in probe: {probe_divergences}")

# Main sampling
print("\n" + "="*60)
print("PHASE 2: Main Sampling (2000 iterations)")
print("="*60)

with student_t_model:
    main_trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )

# Add posterior predictive samples
print("\nGenerating posterior predictive samples...")
with student_t_model:
    pm.sample_posterior_predictive(
        main_trace,
        var_names=['Y_obs'],
        extend_inferencedata=True,
        random_seed=42
    )

# Comprehensive diagnostics
print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

summary_df = az.summary(main_trace, var_names=['beta_0', 'beta_1', 'sigma', 'nu_truncated'])
print("\nParameter Summary:")
print(summary_df.round(4))

# Check convergence criteria
rhat_max = summary_df['r_hat'].max()
ess_bulk_min = summary_df['ess_bulk'].min()
ess_tail_min = summary_df['ess_tail'].min()
divergences = main_trace.sample_stats.diverging.sum().values

print(f"\n--- Convergence Checks ---")
print(f"Max R̂: {rhat_max:.4f} (target: < 1.01)")
print(f"Min ESS (bulk): {ess_bulk_min:.0f} (target: > 400)")
print(f"Min ESS (tail): {ess_tail_min:.0f} (target: > 400)")
print(f"Divergences: {divergences} (target: < 1% = {40})")

convergence_pass = (rhat_max < 1.01) and (ess_bulk_min > 400) and (divergences < 40)
print(f"\nOverall convergence: {'✓ PASS' if convergence_pass else '✗ FAIL'}")

# Extract key parameter: ν
nu_mean = summary_df.loc['nu_truncated', 'mean']
nu_median = summary_df.loc['nu_truncated', '50%']
nu_ci_low = summary_df.loc['nu_truncated', 'hdi_3%']
nu_ci_high = summary_df.loc['nu_truncated', 'hdi_97%']

print("\n" + "="*60)
print("KEY PARAMETER: ν (Degrees of Freedom)")
print("="*60)
print(f"Mean: {nu_mean:.2f}")
print(f"Median: {nu_median:.2f}")
print(f"94% HDI: [{nu_ci_low:.2f}, {nu_ci_high:.2f}]")

if nu_mean > 30:
    print("\n⚠ INTERPRETATION: ν > 30 suggests Normal likelihood is adequate")
    print("   Recommendation: Prefer Model 1 (simpler)")
elif nu_mean < 20:
    print("\n✓ INTERPRETATION: ν < 20 suggests heavy tails are justified")
    print("   Recommendation: Student-t robustness may be valuable")
else:
    print("\n○ INTERPRETATION: ν ∈ [20, 30] is borderline")
    print("   Recommendation: Check LOO comparison to decide")

# Save InferenceData
print("\n" + "="*60)
print("Saving ArviZ InferenceData")
print("="*60)

idata_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
az.to_netcdf(main_trace, idata_path)
print(f"InferenceData saved to: {idata_path}")

# Compute LOO-CV
print("\n" + "="*60)
print("LOO-CV Computation")
print("="*60)

loo_result = az.loo(main_trace, pointwise=True)
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
    loo_compare = az.compare({'Model_1_Normal': idata_model1, 'Model_2_StudentT': main_trace})
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
        recommendation = "ACCEPT Model 2"
    elif delta_loo < -2:
        print("✗ Model 2 worse (overfitting?)")
        recommendation = "REJECT Model 2, prefer Model 1"
    else:
        print("○ Models equivalent (|ΔLOO| < 2)")
        print("  Recommendation: Prefer simpler Model 1")
        recommendation = "Prefer Model 1 (parsimony)"

    # Combined decision
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)

    if nu_mean < 20 and delta_loo > 2:
        print("✓ ACCEPT Model 2: Heavy tails justified AND LOO improvement")
        final_rec = "ACCEPT Model 2"
    elif nu_mean > 30 or abs(delta_loo) < 2:
        print("✓ PREFER Model 1: Simpler model adequate")
        final_rec = "Prefer Model 1"
    else:
        print("○ BORDERLINE: Check detailed diagnostics")
        final_rec = recommendation

    # Save recommendation
    with open(DIAGNOSTICS_DIR / "model_recommendation.txt", 'w') as f:
        f.write(f"Model Recommendation: {final_rec}\n")
        f.write(f"ν = {nu_mean:.2f} [{nu_ci_low:.2f}, {nu_ci_high:.2f}]\n")
        f.write(f"ΔLOO = {delta_loo:.2f} ± {delta_loo_se:.2f}\n")
        f.write(f"Convergence: {'PASS' if convergence_pass else 'FAIL'}\n")

else:
    print("⚠ Model 1 results not found. Skipping comparison.")

# Save summary
summary_df.to_csv(DIAGNOSTICS_DIR / "posterior_summary.csv")

print("\n" + "="*60)
print("Fitting Complete!")
print("="*60)
print(f"\nKey results:")
print(f"  ν = {nu_mean:.2f} [{nu_ci_low:.2f}, {nu_ci_high:.2f}]")
print(f"  LOO-ELPD = {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
print(f"  Convergence: {'✓ PASS' if convergence_pass else '✗ FAIL'}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
