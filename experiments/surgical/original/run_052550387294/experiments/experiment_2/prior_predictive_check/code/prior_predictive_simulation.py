"""
Prior Predictive Simulation for Hierarchical Logit Model (Experiment 2)

This script samples from the prior distributions and generates synthetic datasets
to validate that the model specification produces scientifically plausible data.

Model:
    r_i ~ Binomial(n_i, θ_i)
    logit(θ_i) = μ_logit + σ·η_i
    η_i ~ Normal(0, 1)

Priors:
    μ_logit ~ Normal(-2.53, 1)    # logit(0.074) ≈ -2.53
    σ ~ HalfNormal(0, 1)          # Truncated at 0
    η_i ~ Normal(0, 1)            # Standard normal, i=1,...,12
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # Logistic function
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_DRAWS = 2000  # Number of prior predictive datasets
N_TRIALS = 12

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
n_obs = data['n'].values
r_obs = data['r'].values
prop_obs = r_obs / n_obs
total_successes_obs = r_obs.sum()
total_n_obs = n_obs.sum()

print("=" * 70)
print("PRIOR PREDICTIVE SIMULATION: HIERARCHICAL LOGIT MODEL")
print("=" * 70)
print(f"\nObserved Data Summary:")
print(f"  Trials: {N_TRIALS}")
print(f"  Total subjects: {total_n_obs}")
print(f"  Total successes: {total_successes_obs}")
print(f"  Overall proportion: {total_successes_obs/total_n_obs:.4f}")
print(f"  Sample sizes: {n_obs.tolist()}")
print(f"  Proportions range: [{prop_obs.min():.4f}, {prop_obs.max():.4f}]")

# Extreme cases
print(f"\n  Extreme cases:")
print(f"    Trial 1: {r_obs[0]}/{n_obs[0]} = {prop_obs[0]:.4f}")
print(f"    Trial 8: {r_obs[7]}/{n_obs[7]} = {prop_obs[7]:.4f}")


def sample_prior_predictive(n_draws=1000, seed=None):
    """
    Sample from prior predictive distribution.

    Returns:
        dict with keys:
            - mu_logit: (n_draws,) population mean on logit scale
            - sigma: (n_draws,) scale parameter
            - eta: (n_draws, N_TRIALS) standardized trial effects
            - theta: (n_draws, N_TRIALS) trial probabilities
            - r: (n_draws, N_TRIALS) simulated successes
            - total_r: (n_draws,) total successes across all trials
            - proportions: (n_draws, N_TRIALS) success proportions
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample from priors
    mu_logit = stats.norm.rvs(loc=-2.53, scale=1, size=n_draws)
    sigma = stats.halfnorm.rvs(loc=0, scale=1, size=n_draws)
    eta = stats.norm.rvs(loc=0, scale=1, size=(n_draws, N_TRIALS))

    # Transform to probability scale
    logit_theta = mu_logit[:, np.newaxis] + sigma[:, np.newaxis] * eta
    theta = expit(logit_theta)  # logistic transform

    # Generate binomial data
    r = np.zeros((n_draws, N_TRIALS), dtype=int)
    for i in range(n_draws):
        for j in range(N_TRIALS):
            r[i, j] = stats.binom.rvs(n=n_obs[j], p=theta[i, j])

    # Calculate proportions and totals
    proportions = r / n_obs[np.newaxis, :]
    total_r = r.sum(axis=1)

    return {
        'mu_logit': mu_logit,
        'sigma': sigma,
        'eta': eta,
        'theta': theta,
        'r': r,
        'total_r': total_r,
        'proportions': proportions
    }


# Generate prior predictive samples
print(f"\n" + "=" * 70)
print(f"SAMPLING FROM PRIOR PREDICTIVE DISTRIBUTION")
print(f"=" * 70)
print(f"Number of draws: {N_PRIOR_DRAWS}")

prior_pred = sample_prior_predictive(n_draws=N_PRIOR_DRAWS, seed=42)

print(f"\nPrior Predictive Summary:")
print(f"\nμ_logit (population mean on logit scale):")
print(f"  Mean: {prior_pred['mu_logit'].mean():.3f}")
print(f"  SD: {prior_pred['mu_logit'].std():.3f}")
print(f"  Range: [{prior_pred['mu_logit'].min():.3f}, {prior_pred['mu_logit'].max():.3f}]")

# Transform mu_logit to probability scale for interpretation
mu_prob = expit(prior_pred['mu_logit'])
print(f"\nμ_prob (population mean on probability scale):")
print(f"  Mean: {mu_prob.mean():.3f}")
print(f"  SD: {mu_prob.std():.3f}")
print(f"  Range: [{mu_prob.min():.3f}, {mu_prob.max():.3f}]")
print(f"  Quantiles: [0.025, 0.25, 0.50, 0.75, 0.975]")
print(f"             {np.quantile(mu_prob, [0.025, 0.25, 0.50, 0.75, 0.975])}")

print(f"\nσ (scale parameter):")
print(f"  Mean: {prior_pred['sigma'].mean():.3f}")
print(f"  SD: {prior_pred['sigma'].std():.3f}")
print(f"  Range: [{prior_pred['sigma'].min():.3f}, {prior_pred['sigma'].max():.3f}]")
print(f"  Quantiles: [0.025, 0.25, 0.50, 0.75, 0.975]")
print(f"             {np.quantile(prior_pred['sigma'], [0.025, 0.25, 0.50, 0.75, 0.975])}")

print(f"\nθ_i (trial probabilities):")
print(f"  Mean: {prior_pred['theta'].mean():.3f}")
print(f"  SD: {prior_pred['theta'].std():.3f}")
print(f"  Range: [{prior_pred['theta'].min():.6f}, {prior_pred['theta'].max():.3f}]")
print(f"  Proportion < 0.01: {(prior_pred['theta'] < 0.01).sum() / prior_pred['theta'].size:.3f}")
print(f"  Proportion > 0.30: {(prior_pred['theta'] > 0.30).sum() / prior_pred['theta'].size:.3f}")

print(f"\nTotal successes (across all trials):")
print(f"  Observed: {total_successes_obs}")
print(f"  Prior predictive mean: {prior_pred['total_r'].mean():.1f}")
print(f"  Prior predictive SD: {prior_pred['total_r'].std():.1f}")
print(f"  Prior predictive range: [{prior_pred['total_r'].min()}, {prior_pred['total_r'].max()}]")

# Calculate quantiles
quantiles = np.quantile(prior_pred['total_r'], [0.025, 0.25, 0.50, 0.75, 0.975])
print(f"  Quantiles: [0.025, 0.25, 0.50, 0.75, 0.975]")
print(f"             {quantiles.astype(int)}")

# Check if observed is extreme
percentile = stats.percentileofscore(prior_pred['total_r'], total_successes_obs)
print(f"  Observed percentile: {percentile:.1f}%")

# Check coverage of extreme trials
print(f"\nCoverage of extreme observed values:")
print(f"  Trial 1 (0/47):")
trial_1_sims = prior_pred['r'][:, 0]
print(f"    Simulated 0s: {(trial_1_sims == 0).sum()} / {N_PRIOR_DRAWS} ({100*(trial_1_sims == 0).mean():.1f}%)")
print(f"    Mean: {trial_1_sims.mean():.2f}, SD: {trial_1_sims.std():.2f}")
print(f"    Range: [{trial_1_sims.min()}, {trial_1_sims.max()}]")

print(f"  Trial 8 (31/215):")
trial_8_sims = prior_pred['r'][:, 7]
print(f"    Simulated >= 31: {(trial_8_sims >= 31).sum()} / {N_PRIOR_DRAWS} ({100*(trial_8_sims >= 31).mean():.1f}%)")
print(f"    Mean: {trial_8_sims.mean():.2f}, SD: {trial_8_sims.std():.2f}")
print(f"    Range: [{trial_8_sims.min()}, {trial_8_sims.max()}]")

# Check for numerical issues
print(f"\nNumerical Stability Checks:")
print(f"  Any NaN in theta: {np.isnan(prior_pred['theta']).any()}")
print(f"  Any Inf in theta: {np.isinf(prior_pred['theta']).any()}")
print(f"  Theta values at boundaries:")
print(f"    < 1e-10: {(prior_pred['theta'] < 1e-10).sum()} / {prior_pred['theta'].size}")
print(f"    > 1-1e-10: {(prior_pred['theta'] > 1-1e-10).sum()} / {prior_pred['theta'].size}")

# Save results
output_path = Path('/workspace/experiments/experiment_2/prior_predictive_check/code')
np.savez(
    output_path / 'prior_predictive_samples.npz',
    mu_logit=prior_pred['mu_logit'],
    sigma=prior_pred['sigma'],
    eta=prior_pred['eta'],
    theta=prior_pred['theta'],
    r=prior_pred['r'],
    total_r=prior_pred['total_r'],
    proportions=prior_pred['proportions'],
    n_obs=n_obs,
    r_obs=r_obs
)

print(f"\n" + "=" * 70)
print(f"Prior predictive samples saved to:")
print(f"  {output_path / 'prior_predictive_samples.npz'}")
print(f"=" * 70)
