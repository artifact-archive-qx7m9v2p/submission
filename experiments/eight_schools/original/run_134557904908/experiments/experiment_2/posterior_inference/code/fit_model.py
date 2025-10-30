"""
Fit Random-Effects Hierarchical Meta-Analysis Model to Real Data

Model:
    y_i | θ_i, σ_i ~ Normal(θ_i, σ_i²)   for i = 1, ..., 8
    θ_i | μ, τ ~ Normal(μ, τ²)
    μ ~ Normal(0, 20²)
    τ ~ Half-Normal(0, 5²)

Uses non-centered parameterization:
    θ_raw ~ Normal(0, 1)
    θ = μ + τ * θ_raw

This script:
1. Loads real data from CSV
2. Fits hierarchical model using PyMC with MCMC
3. Computes log_likelihood for LOO comparison
4. Saves InferenceData with complete posterior
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
BASE_DIR = Path("/workspace/experiments/experiment_2/posterior_inference")
DATA_PATH = Path("/workspace/data/data.csv")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

print("="*70)
print("FITTING RANDOM-EFFECTS HIERARCHICAL META-ANALYSIS MODEL")
print("="*70)

# Load data
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma_obs = data['sigma'].values
J = len(y_obs)

print(f"\nData loaded:")
print(f"  Studies (J): {J}")
print(f"  Observed effects (y): {y_obs}")
print(f"  Standard errors (σ): {sigma_obs}")

# Build model
print(f"\n{'='*70}")
print("Building PyMC model...")
print(f"{'='*70}")

with pm.Model() as model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)

    # Non-centered parameterization for θ
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y = pm.Normal('y', mu=theta, sigma=sigma_obs, observed=y_obs)

print("\nModel specification:")
print(f"  Hyperparameters: μ ~ Normal(0, 20²), τ ~ Half-Normal(0, 5²)")
print(f"  Parameterization: Non-centered (θ = μ + τ * θ_raw)")
print(f"  Study-specific effects: θ_i for i=1,...,{J}")
print(f"  Likelihood: y_i ~ Normal(θ_i, σ_i²)")

# Sampling parameters
n_chains = 4
n_tune = 1000
n_draws = 2000
target_accept = 0.95  # High target for hierarchical model

print(f"\n{'='*70}")
print("Sampling from posterior...")
print(f"{'='*70}")
print(f"  Chains: {n_chains}")
print(f"  Warmup iterations: {n_tune}")
print(f"  Post-warmup draws: {n_draws}")
print(f"  Target acceptance: {target_accept}")
print(f"  Total posterior samples: {n_chains * n_draws}")

# Sample from posterior
with model:
    idata = pm.sample(
        draws=n_draws,
        tune=n_tune,
        chains=n_chains,
        cores=4,
        target_accept=target_accept,
        return_inferencedata=True,
        random_seed=42
    )

print("\n✓ Sampling complete!")

# Check for divergences
n_divergences = idata.sample_stats['diverging'].sum().item()
print(f"\nDivergences: {n_divergences}")
if n_divergences > 0:
    print(f"  WARNING: {n_divergences} divergent transitions detected")
    print(f"  This may indicate sampling issues or model misspecification")

# Compute log-likelihood for LOO
print(f"\n{'='*70}")
print("Computing log-likelihood for LOO...")
print(f"{'='*70}")

# Use pm.compute_log_likelihood to add pointwise log-likelihood
with model:
    pm.compute_log_likelihood(idata)

print("✓ Log-likelihood computed")

# Verify log_likelihood is present
if hasattr(idata, 'log_likelihood') and 'y' in idata.log_likelihood:
    log_lik_shape = idata.log_likelihood['y'].shape
    print(f"  Shape: {log_lik_shape} (chains={n_chains}, draws={n_draws}, observations={J})")
else:
    print("  WARNING: log_likelihood not found in expected location")

# Save InferenceData
output_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
print(f"\n{'='*70}")
print(f"Saving InferenceData...")
print(f"{'='*70}")
print(f"  Path: {output_path}")

idata.to_netcdf(output_path)
print("✓ Saved!")

# Basic convergence diagnostics
print(f"\n{'='*70}")
print("CONVERGENCE DIAGNOSTICS")
print(f"{'='*70}")

# Summary statistics for key parameters
summary = az.summary(
    idata,
    var_names=['mu', 'tau', 'theta'],
    
)

print("\nPosterior Summary (Hyperparameters):")
print(summary.loc[['mu', 'tau'], ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk', 'ess_tail', 'r_hat']])

print("\nPosterior Summary (Study-Specific Effects θ):")
print(summary.loc[['theta[0]', 'theta[1]', 'theta[2]', 'theta[3]',
                    'theta[4]', 'theta[5]', 'theta[6]', 'theta[7]'],
                   ['mean', 'sd', 'ess_bulk', 'r_hat']])

# Check convergence criteria
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()

print(f"\n{'='*70}")
print("Convergence Criteria:")
print(f"{'='*70}")
print(f"  Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"  Min ESS bulk: {min_ess_bulk:.1f} (target: > 400)")
print(f"  Min ESS tail: {min_ess_tail:.1f} (target: > 400)")
print(f"  Divergences: {n_divergences} (target: 0)")

converged = (max_rhat < 1.01) and (min_ess_bulk > 400) and (min_ess_tail > 400) and (n_divergences == 0)

if converged:
    print("\n✓ CONVERGENCE: EXCELLENT")
elif max_rhat < 1.01 and min_ess_bulk > 100:
    print("\n⚠ CONVERGENCE: ACCEPTABLE (but below targets)")
else:
    print("\n✗ CONVERGENCE: ISSUES DETECTED")

# Compute I² statistic
print(f"\n{'='*70}")
print("Heterogeneity Statistics")
print(f"{'='*70}")

# Extract posterior samples
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()

# I² = 100 * τ² / (τ² + mean(σ²))
sigma_sq_mean = np.mean(sigma_obs**2)
I_sq_samples = 100 * tau_samples**2 / (tau_samples**2 + sigma_sq_mean)

I_sq_mean = I_sq_samples.mean()
I_sq_median = np.median(I_sq_samples)
I_sq_hdi = az.hdi(I_sq_samples, hdi_prob=0.95)

print(f"\nI² (proportion of variance due to heterogeneity):")
print(f"  Mean: {I_sq_mean:.1f}%")
print(f"  Median: {I_sq_median:.1f}%")
print(f"  95% HDI: [{I_sq_hdi[0]:.1f}%, {I_sq_hdi[1]:.1f}%]")

# Probability that τ is small
print(f"\nτ (between-study standard deviation):")
print(f"  Mean: {tau_samples.mean():.2f}")
print(f"  Median: {np.median(tau_samples):.2f}")
print(f"  95% HDI: [{az.hdi(tau_samples, hdi_prob=0.95)[0]:.2f}, {az.hdi(tau_samples, hdi_prob=0.95)[1]:.2f}]")

p_tau_lt_1 = (tau_samples < 1).mean()
p_tau_lt_5 = (tau_samples < 5).mean()
p_I_sq_lt_25 = (I_sq_samples < 25).mean()

print(f"\nProbabilities:")
print(f"  P(τ < 1): {p_tau_lt_1:.3f}")
print(f"  P(τ < 5): {p_tau_lt_5:.3f}")
print(f"  P(I² < 25%): {p_I_sq_lt_25:.3f}")

# Scientific interpretation
print(f"\n{'='*70}")
print("Scientific Interpretation")
print(f"{'='*70}")

print(f"\nμ (population mean effect):")
print(f"  Posterior mean: {mu_samples.mean():.2f} ± {mu_samples.std():.2f}")
print(f"  95% HDI: [{az.hdi(mu_samples, hdi_prob=0.95)[0]:.2f}, {az.hdi(mu_samples, hdi_prob=0.95)[1]:.2f}]")

if I_sq_mean < 25:
    interpretation = "Low heterogeneity detected (I² < 25%)"
    print(f"\n{interpretation}")
    print(f"  → Between-study variance is small")
    print(f"  → Model 1 (fixed-effect) likely adequate")
    print(f"  → Pooling is strong (θ_i ≈ μ)")
elif I_sq_mean < 50:
    interpretation = "Moderate heterogeneity (25% < I² < 50%)"
    print(f"\n{interpretation}")
    print(f"  → Some between-study variance")
    print(f"  → Hierarchical model appropriate")
    print(f"  → Partial pooling of effects")
else:
    interpretation = "High heterogeneity (I² > 50%)"
    print(f"\n{interpretation}")
    print(f"  → Substantial between-study variance")
    print(f"  → Hierarchical model necessary")
    print(f"  → Limited pooling of effects")

# Save summary
summary_path = DIAGNOSTICS_DIR / "convergence_summary.csv"
summary.to_csv(summary_path)
print(f"\n✓ Summary saved to: {summary_path}")

# Save key results
results = {
    'converged': bool(converged),
    'max_rhat': float(max_rhat),
    'min_ess_bulk': float(min_ess_bulk),
    'min_ess_tail': float(min_ess_tail),
    'n_divergences': int(n_divergences),
    'mu_mean': float(mu_samples.mean()),
    'mu_sd': float(mu_samples.std()),
    'tau_mean': float(tau_samples.mean()),
    'tau_median': float(np.median(tau_samples)),
    'I_sq_mean': float(I_sq_mean),
    'I_sq_median': float(I_sq_median),
    'p_tau_lt_1': float(p_tau_lt_1),
    'p_tau_lt_5': float(p_tau_lt_5),
    'p_I_sq_lt_25': float(p_I_sq_lt_25),
    'interpretation': interpretation
}

results_path = DIAGNOSTICS_DIR / "posterior_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {results_path}")

print(f"\n{'='*70}")
print("Posterior inference complete!")
print(f"{'='*70}")
print(f"\nKey files created:")
print(f"  1. {output_path}")
print(f"  2. {summary_path}")
print(f"  3. {results_path}")
print(f"\nNext steps:")
print(f"  - Create diagnostic visualizations")
print(f"  - Perform posterior predictive checks")
print(f"  - Compare with Model 1 using LOO")
