"""
Prior Predictive Check for Negative Binomial State-Space Model (PyMC version)
Validates that priors generate scientifically plausible data before fitting.
"""

import numpy as np
import pandas as pd
import pymc as pm
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
CODE_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/code")
PLOTS_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/plots")
DATA_PATH = Path("/workspace/data/data.csv")

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load observed data for context
data = pd.read_csv(DATA_PATH)
N = len(data)
observed_counts = data['C'].values

print(f"Observed data: N={N}, range=[{observed_counts.min()}, {observed_counts.max()}]")
print(f"Observed mean={observed_counts.mean():.2f}, SD={observed_counts.std():.2f}")
print(f"Observed growth factor: {observed_counts[-1] / observed_counts[0]:.2f}x\n")

# Build PyMC model for prior predictive sampling
print("Building PyMC model...")

with pm.Model() as prior_model:
    # Priors
    delta = pm.Normal("delta", mu=0.05, sigma=0.02)
    sigma_eta = pm.Exponential("sigma_eta", lam=10)
    phi = pm.Exponential("phi", lam=0.1)

    # Initial state
    eta_1 = pm.Normal("eta_1", mu=np.log(50), sigma=1)

    # Non-centered parameterization for latent states
    eta_raw = pm.Normal("eta_raw", mu=0, sigma=1, shape=N-1)

    # Build latent state trajectory
    eta = pm.Deterministic("eta", pm.math.concatenate([
        [eta_1],
        eta_1 + pm.math.cumsum(delta + sigma_eta * eta_raw)
    ]))

    # Generate prior predictive counts
    C_prior = pm.NegativeBinomial("C_prior", mu=pm.math.exp(eta), alpha=phi, shape=N)

    # Summary statistics
    prior_mean_count = pm.Deterministic("prior_mean_count", pm.math.mean(C_prior))
    prior_max_count = pm.Deterministic("prior_max_count", pm.math.max(C_prior))
    prior_min_count = pm.Deterministic("prior_min_count", pm.math.min(C_prior))
    prior_growth_factor = pm.Deterministic("prior_growth_factor", pm.math.exp(eta[N-1] - eta[0]))
    total_log_change = pm.Deterministic("total_log_change", eta[N-1] - eta[0])

# Sample from prior predictive
print("Running prior predictive sampling (1000 draws)...")
with prior_model:
    prior_samples = pm.sample_prior_predictive(samples=1000, random_seed=42)

# Extract samples
delta_samples = prior_samples.prior["delta"].values.flatten()
sigma_eta_samples = prior_samples.prior["sigma_eta"].values.flatten()
phi_samples = prior_samples.prior["phi"].values.flatten()
eta_samples = prior_samples.prior["eta"].values.reshape(-1, N)
C_prior_samples = prior_samples.prior["C_prior"].values.reshape(-1, N)

# Summary statistics
prior_mean_count = prior_samples.prior["prior_mean_count"].values.flatten()
prior_max_count = prior_samples.prior["prior_max_count"].values.flatten()
prior_min_count = prior_samples.prior["prior_min_count"].values.flatten()
prior_growth_factor = prior_samples.prior["prior_growth_factor"].values.flatten()
total_log_change = prior_samples.prior["total_log_change"].values.flatten()

print(f"\nPrior Predictive Sampling Complete: {len(delta_samples)} draws\n")

# ============================================================================
# DIAGNOSTIC STATISTICS
# ============================================================================

print("="*70)
print("PRIOR PARAMETER DIAGNOSTICS")
print("="*70)

print("\nDrift (delta) - Expected ~0.05 (5% growth per period):")
print(f"  Mean: {delta_samples.mean():.4f}")
print(f"  SD: {delta_samples.std():.4f}")
print(f"  95% CI: [{np.percentile(delta_samples, 2.5):.4f}, {np.percentile(delta_samples, 97.5):.4f}]")
print(f"  Range: [{delta_samples.min():.4f}, {delta_samples.max():.4f}]")

print("\nInnovation SD (sigma_eta) - Expected small (0.05-0.10):")
print(f"  Mean: {sigma_eta_samples.mean():.4f}")
print(f"  Median: {np.median(sigma_eta_samples):.4f}")
print(f"  95% CI: [{np.percentile(sigma_eta_samples, 2.5):.4f}, {np.percentile(sigma_eta_samples, 97.5):.4f}]")
print(f"  Range: [{sigma_eta_samples.min():.4f}, {sigma_eta_samples.max():.4f}]")

print("\nDispersion (phi) - Expected moderate (10-20):")
print(f"  Mean: {phi_samples.mean():.4f}")
print(f"  Median: {np.median(phi_samples):.4f}")
print(f"  95% CI: [{np.percentile(phi_samples, 2.5):.4f}, {np.percentile(phi_samples, 97.5):.4f}]")
print(f"  Range: [{phi_samples.min():.4f}, {phi_samples.max():.4f}]")

print("\n" + "="*70)
print("PRIOR PREDICTIVE DATA DIAGNOSTICS")
print("="*70)

print("\nPrior Predictive Count Statistics:")
print(f"  Mean of means: {prior_mean_count.mean():.2f}")
print(f"  95% CI of means: [{np.percentile(prior_mean_count, 2.5):.2f}, {np.percentile(prior_mean_count, 97.5):.2f}]")

print("\nPrior Predictive Maximum Counts:")
print(f"  Median: {np.median(prior_max_count):.2f}")
print(f"  95% CI: [{np.percentile(prior_max_count, 2.5):.2f}, {np.percentile(prior_max_count, 97.5):.2f}]")
print(f"  Extreme max: {prior_max_count.max():.2f}")
print(f"  Observed max: {observed_counts.max()}")

print("\nPrior Predictive Growth Factors (C_40 / C_1):")
print(f"  Mean: {prior_growth_factor.mean():.2f}x")
print(f"  Median: {np.median(prior_growth_factor):.2f}x")
print(f"  95% CI: [{np.percentile(prior_growth_factor, 2.5):.2f}x, {np.percentile(prior_growth_factor, 97.5):.2f}x]")
print(f"  Observed: {observed_counts[-1] / observed_counts[0]:.2f}x")

print("\nTotal Log-Scale Change (eta_40 - eta_1):")
print(f"  Mean: {total_log_change.mean():.2f}")
print(f"  95% CI: [{np.percentile(total_log_change, 2.5):.2f}, {np.percentile(total_log_change, 97.5):.2f}]")
print(f"  Observed: {np.log(observed_counts[-1]) - np.log(observed_counts[0]):.2f}")

# ============================================================================
# RED FLAGS CHECK
# ============================================================================

print("\n" + "="*70)
print("COMPUTATIONAL RED FLAGS")
print("="*70)

# Check for extreme values
extreme_counts = (C_prior_samples > 10000).sum()
print(f"\nCounts > 10,000: {extreme_counts} / {C_prior_samples.size} ({100*extreme_counts/C_prior_samples.size:.2f}%)")

# Check for implausibly large growth
extreme_growth = (prior_growth_factor > 100).sum()
print(f"Growth factors > 100x: {extreme_growth} / {len(prior_growth_factor)} ({100*extreme_growth/len(prior_growth_factor):.2f}%)")

# Check for negative or zero phi
problematic_phi = (phi_samples < 0.01).sum()
print(f"Phi < 0.01 (near-zero): {problematic_phi} / {len(phi_samples)} ({100*problematic_phi/len(phi_samples):.2f}%)")

# Check for very large phi (approaching Poisson)
large_phi = (phi_samples > 1000).sum()
print(f"Phi > 1000 (near-Poisson): {large_phi} / {len(phi_samples)} ({100*large_phi/len(phi_samples):.2f}%)")

# ============================================================================
# SAVE SUMMARY STATISTICS
# ============================================================================

summary = {
    "n_draws": len(delta_samples),
    "parameters": {
        "delta": {
            "mean": float(delta_samples.mean()),
            "sd": float(delta_samples.std()),
            "q025": float(np.percentile(delta_samples, 2.5)),
            "q975": float(np.percentile(delta_samples, 97.5))
        },
        "sigma_eta": {
            "mean": float(sigma_eta_samples.mean()),
            "median": float(np.median(sigma_eta_samples)),
            "q025": float(np.percentile(sigma_eta_samples, 2.5)),
            "q975": float(np.percentile(sigma_eta_samples, 97.5))
        },
        "phi": {
            "mean": float(phi_samples.mean()),
            "median": float(np.median(phi_samples)),
            "q025": float(np.percentile(phi_samples, 2.5)),
            "q975": float(np.percentile(phi_samples, 97.5))
        }
    },
    "predictive_checks": {
        "mean_count": {
            "mean": float(prior_mean_count.mean()),
            "q025": float(np.percentile(prior_mean_count, 2.5)),
            "q975": float(np.percentile(prior_mean_count, 97.5))
        },
        "max_count": {
            "median": float(np.median(prior_max_count)),
            "q975": float(np.percentile(prior_max_count, 97.5)),
            "extreme": float(prior_max_count.max())
        },
        "growth_factor": {
            "mean": float(prior_growth_factor.mean()),
            "median": float(np.median(prior_growth_factor)),
            "q025": float(np.percentile(prior_growth_factor, 2.5)),
            "q975": float(np.percentile(prior_growth_factor, 97.5))
        }
    },
    "observed_data": {
        "mean": float(observed_counts.mean()),
        "max": int(observed_counts.max()),
        "growth_factor": float(observed_counts[-1] / observed_counts[0])
    },
    "red_flags": {
        "extreme_counts_pct": float(100 * extreme_counts / C_prior_samples.size),
        "extreme_growth_pct": float(100 * extreme_growth / len(prior_growth_factor)),
        "near_zero_phi_pct": float(100 * problematic_phi / len(phi_samples)),
        "large_phi_pct": float(100 * large_phi / len(phi_samples))
    }
}

with open(CODE_DIR / "prior_predictive_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("Summary statistics saved to prior_predictive_summary.json")
print("="*70)

# Save samples for plotting
np.savez(
    CODE_DIR / "prior_samples.npz",
    delta=delta_samples,
    sigma_eta=sigma_eta_samples,
    phi=phi_samples,
    eta=eta_samples,
    C_prior=C_prior_samples,
    prior_growth_factor=prior_growth_factor,
    observed_counts=observed_counts,
    time_index=np.arange(N)
)

print("\nPrior samples saved to prior_samples.npz")
print("Ready for visualization!")
