"""
Fit Bayesian Fixed-Effect Meta-Analysis Model to Real Data
============================================================

Model:
    y_i | θ, σ_i ~ Normal(θ, σ_i²)   for i = 1, ..., 8
    θ ~ Normal(0, 20²)

This script:
1. Loads real data from CSV
2. Fits the model using PyMC with MCMC
3. Computes analytical posterior for validation
4. Saves InferenceData with log_likelihood for LOO
5. Performs comprehensive convergence diagnostics
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DATA_PATH = Path("/workspace/data/data.csv")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

# Create directories
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BAYESIAN FIXED-EFFECT META-ANALYSIS: POSTERIOR INFERENCE")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("1. Loading data from CSV...")
data = pd.read_csv(DATA_PATH)
y = data['y'].values
sigma = data['sigma'].values
n_studies = len(y)

print(f"   Number of studies: {n_studies}")
print(f"   Observed effects: {y}")
print(f"   Standard errors: {sigma}")
print()

# ============================================================================
# 2. COMPUTE ANALYTICAL POSTERIOR (for validation)
# ============================================================================
print("2. Computing analytical posterior (conjugate normal-normal)...")

# Precision-weighted estimation
prior_mean = 0.0
prior_sigma = 20.0
prior_precision = 1 / (prior_sigma ** 2)

# Data precisions
data_precisions = 1 / (sigma ** 2)

# Posterior precision and mean
posterior_precision = prior_precision + np.sum(data_precisions)
posterior_variance = 1 / posterior_precision
posterior_sigma = np.sqrt(posterior_variance)

# Weighted mean
posterior_mean = (prior_precision * prior_mean + np.sum(data_precisions * y)) / posterior_precision

print(f"   Analytical posterior: θ ~ N({posterior_mean:.4f}, {posterior_sigma:.4f}²)")
print(f"   Posterior mean: {posterior_mean:.4f}")
print(f"   Posterior SD: {posterior_sigma:.4f}")
print(f"   95% CI: [{posterior_mean - 1.96*posterior_sigma:.4f}, {posterior_mean + 1.96*posterior_sigma:.4f}]")
print()

# ============================================================================
# 3. FIT MODEL WITH PyMC
# ============================================================================
print("3. Fitting model with PyMC (MCMC sampling)...")

with pm.Model() as model:
    # Prior
    theta = pm.Normal('theta', mu=0, sigma=20)

    # Likelihood - each observation has its own known variance
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Sample from posterior
    # Using adaptive strategy: start with conservative settings
    print("   Sampling with 4 chains, 1000 warmup + 2000 iterations...")
    print("   Target accept: 0.95 (conservative for initial fit)")

    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        return_inferencedata=True,
        target_accept=0.95,
        random_seed=42
    )

    # CRITICAL: Compute log-likelihood for LOO-CV
    print("\n   Computing log-likelihood for model comparison...")
    pm.compute_log_likelihood(idata)

print("\n   Sampling complete!")
print()

# ============================================================================
# 4. BASIC CONVERGENCE DIAGNOSTICS
# ============================================================================
print("4. Running convergence diagnostics...")

# Get summary statistics
summary = az.summary(idata, var_names=['theta'])
print("\n" + "=" * 80)
print("CONVERGENCE SUMMARY")
print("=" * 80)
print(summary)
print()

# Extract key diagnostics
rhat = summary.loc['theta', 'r_hat']
ess_bulk = summary.loc['theta', 'ess_bulk']
ess_tail = summary.loc['theta', 'ess_tail']
mcse_mean = summary.loc['theta', 'mcse_mean']
posterior_sd = summary.loc['theta', 'sd']

# Check for divergences
divergences = idata.sample_stats.diverging.sum().item()
max_treedepth = (idata.sample_stats.tree_depth == 10).sum().item()

print("CONVERGENCE CHECKS:")
print(f"  R-hat: {rhat:.6f} {'✓ PASS' if rhat < 1.01 else '✗ FAIL'} (target < 1.01)")
print(f"  ESS (bulk): {ess_bulk:.0f} {'✓ PASS' if ess_bulk > 400 else '✗ WARNING'} (target > 400)")
print(f"  ESS (tail): {ess_tail:.0f} {'✓ PASS' if ess_tail > 400 else '✗ WARNING'} (target > 400)")
print(f"  MCSE/SD: {mcse_mean/posterior_sd:.4f} {'✓ PASS' if mcse_mean/posterior_sd < 0.05 else '✗ WARNING'} (target < 0.05)")
print(f"  Divergences: {divergences} {'✓ PASS' if divergences == 0 else '✗ FAIL'}")
print(f"  Max tree depth hits: {max_treedepth} {'✓ PASS' if max_treedepth == 0 else '✗ WARNING'}")
print()

# ============================================================================
# 5. VALIDATE MCMC AGAINST ANALYTICAL POSTERIOR
# ============================================================================
print("5. Validating MCMC against analytical posterior...")

mcmc_mean = summary.loc['theta', 'mean']
mcmc_sd = summary.loc['theta', 'sd']

mean_error = abs(mcmc_mean - posterior_mean)
sd_error = abs(mcmc_sd - posterior_sigma)

print(f"\n  Analytical posterior mean: {posterior_mean:.4f}")
print(f"  MCMC posterior mean:       {mcmc_mean:.4f}")
print(f"  Absolute error:            {mean_error:.4f} {'✓ PASS' if mean_error < 0.1 else '✗ FAIL'} (tolerance < 0.1)")
print()
print(f"  Analytical posterior SD:   {posterior_sigma:.4f}")
print(f"  MCMC posterior SD:         {mcmc_sd:.4f}")
print(f"  Absolute error:            {sd_error:.4f} {'✓ PASS' if sd_error < 0.1 else '✗ INFO'}")
print()

# Overall validation
validation_pass = (mean_error < 0.1) and (rhat < 1.01) and (ess_bulk > 400) and (divergences == 0)
print(f"VALIDATION STATUS: {'✓ PASS' if validation_pass else '✗ NEEDS INVESTIGATION'}")
print()

# ============================================================================
# 6. SAVE INFERENCE DATA WITH LOG-LIKELIHOOD
# ============================================================================
print("6. Saving InferenceData with log-likelihood...")

output_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
idata.to_netcdf(str(output_path))
print(f"   Saved to: {output_path}")

# Verify log_likelihood group exists
if 'log_likelihood' in idata.groups():
    print("   ✓ log_likelihood group confirmed in InferenceData")
    print(f"   Shape: {idata.log_likelihood.y_obs.shape}")
else:
    print("   ✗ WARNING: log_likelihood group not found!")
print()

# ============================================================================
# 7. COMPUTE QUANTILES AND TAIL PROBABILITIES
# ============================================================================
# Get posterior samples
theta_samples = idata.posterior.theta.values.flatten()

# Compute quantiles
median = np.median(theta_samples)
q05 = np.percentile(theta_samples, 5)
q95 = np.percentile(theta_samples, 95)
q025 = np.percentile(theta_samples, 2.5)
q975 = np.percentile(theta_samples, 97.5)

# Tail probabilities
p_positive = (theta_samples > 0).mean()
p_gt_10 = (theta_samples > 10).mean()
p_gt_5 = (theta_samples > 5).mean()

# ============================================================================
# 8. SAVE SUMMARY STATISTICS
# ============================================================================
print("7. Saving summary statistics...")

# Save ArviZ summary
summary_path = DIAGNOSTICS_DIR / "arviz_summary.csv"
summary.to_csv(str(summary_path))
print(f"   ArviZ summary saved to: {summary_path}")

# Save detailed diagnostics as JSON
diagnostics = {
    "convergence": {
        "r_hat": float(rhat),
        "ess_bulk": float(ess_bulk),
        "ess_tail": float(ess_tail),
        "mcse_mean": float(mcse_mean),
        "divergences": int(divergences),
        "max_treedepth_hits": int(max_treedepth)
    },
    "posterior_summary": {
        "mean": float(mcmc_mean),
        "median": float(median),
        "sd": float(mcmc_sd),
        "q05": float(q05),
        "q95": float(q95),
        "q025": float(q025),
        "q975": float(q975),
        "hdi_95_lower": float(summary.loc['theta', 'hdi_3%']),
        "hdi_95_upper": float(summary.loc['theta', 'hdi_97%'])
    },
    "tail_probabilities": {
        "p_theta_gt_0": float(p_positive),
        "p_theta_gt_5": float(p_gt_5),
        "p_theta_gt_10": float(p_gt_10)
    },
    "analytical_validation": {
        "analytical_mean": float(posterior_mean),
        "analytical_sd": float(posterior_sigma),
        "mcmc_mean": float(mcmc_mean),
        "mcmc_sd": float(mcmc_sd),
        "mean_error": float(mean_error),
        "sd_error": float(sd_error),
        "validation_pass": bool(validation_pass)
    },
    "sampling_config": {
        "draws": 2000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.95
    }
}

diag_path = DIAGNOSTICS_DIR / "diagnostics.json"
with open(diag_path, 'w') as f:
    json.dump(diagnostics, f, indent=2)
print(f"   Diagnostics JSON saved to: {diag_path}")
print()

# ============================================================================
# 9. POSTERIOR INFERENCE SUMMARY
# ============================================================================
print("=" * 80)
print("POSTERIOR INFERENCE SUMMARY")
print("=" * 80)
print()
print(f"Posterior for θ (fixed effect):")
print(f"  Mean:      {mcmc_mean:.4f}")
print(f"  Median:    {median:.4f}")
print(f"  SD:        {mcmc_sd:.4f}")
print(f"  95% HDI:   [{summary.loc['theta', 'hdi_3%']:.4f}, {summary.loc['theta', 'hdi_97%']:.4f}]")
print(f"  95% CI:    [{q025:.4f}, {q975:.4f}]")
print(f"  90% CI:    [{q05:.4f}, {q95:.4f}]")
print()

print("Tail probabilities:")
print(f"  P(θ > 0):  {p_positive:.4f}")
print(f"  P(θ > 5):  {p_gt_5:.4f}")
print(f"  P(θ > 10): {p_gt_10:.4f}")
print()

print("=" * 80)
print("FITTING COMPLETE - Proceeding to visualization...")
print("=" * 80)
