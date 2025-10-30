"""
Simulation-Based Calibration for Random-Effects Hierarchical Model

Tests whether the model + inference can correctly recover known parameters.
Critical for hierarchical models to detect funnel pathology and other issues.
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Load observed data structure
data = pd.read_csv('/workspace/data/data.csv')
sigma_obs = data['sigma'].values
J = len(sigma_obs)

print("="*70)
print("SIMULATION-BASED CALIBRATION - HIERARCHICAL MODEL")
print("="*70)
print(f"\nModel: Random-Effects Meta-Analysis")
print(f"Studies: J = {J}")
print(f"Observed sigmas: {sigma_obs}")

# SBC parameters
n_sbc_sims = 200  # Balance between thoroughness and computational time
n_chains = 2  # Fewer chains for speed in SBC
n_samples = 500  # Per chain after warmup
n_warmup = 500

print(f"\nSBC settings:")
print(f"  Number of simulations: {n_sbc_sims}")
print(f"  Chains per simulation: {n_chains}")
print(f"  Samples per chain: {n_samples}")
print(f"  Warmup: {n_warmup}")

# Prior specification (baseline)
mu_prior_mean = 0
mu_prior_sd = 20
tau_prior_sd = 5

def simulate_data(J, sigma_obs):
    """
    Simulate data from the prior predictive distribution.
    Returns: mu_true, tau_true, theta_true, y_sim
    """
    # Sample hyperparameters from prior
    mu_true = np.random.normal(mu_prior_mean, mu_prior_sd)
    tau_true = np.abs(np.random.normal(0, tau_prior_sd))

    # Sample study-specific effects (non-centered)
    theta_raw = np.random.normal(0, 1, J)
    theta_true = mu_true + tau_true * theta_raw

    # Simulate observed data
    y_sim = np.random.normal(theta_true, sigma_obs)

    return mu_true, tau_true, theta_true, y_sim

def fit_model(y, sigma, n_chains, n_samples, n_warmup):
    """
    Fit the hierarchical model to data.
    Uses non-centered parameterization.
    """
    with pm.Model() as model:
        # Hyperpriors
        mu = pm.Normal('mu', mu=mu_prior_mean, sigma=mu_prior_sd)
        tau = pm.HalfNormal('tau', sigma=tau_prior_sd)

        # Non-centered parameterization
        theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
        theta = pm.Deterministic('theta', mu + tau * theta_raw)

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

        # Sample
        idata = pm.sample(
            draws=n_samples,
            tune=n_warmup,
            chains=n_chains,
            cores=1,  # Single core for stability in loop
            target_accept=0.90,
            return_inferencedata=True,
            progressbar=False
        )

    return idata

def compute_rank(true_value, posterior_samples):
    """
    Compute rank of true value in posterior samples.
    Rank should be uniform if inference is calibrated.
    """
    return np.sum(posterior_samples < true_value)

# Run SBC
print(f"\n{'='*70}")
print("Running SBC simulations...")
print(f"{'='*70}")

results = {
    'mu_rank': [],
    'tau_rank': [],
    'mu_true': [],
    'tau_true': [],
    'mu_post_mean': [],
    'tau_post_mean': [],
    'converged': [],
    'max_rhat': [],
    'min_ess': [],
    'n_divergences': []
}

failed_sims = []

for sim in tqdm(range(n_sbc_sims), desc="SBC Progress"):
    try:
        # Simulate data
        mu_true, tau_true, theta_true, y_sim = simulate_data(J, sigma_obs)

        # Fit model
        idata = fit_model(y_sim, sigma_obs, n_chains, n_samples, n_warmup)

        # Extract posterior samples
        mu_samples = idata.posterior['mu'].values.flatten()
        tau_samples = idata.posterior['tau'].values.flatten()

        # Compute ranks
        mu_rank = compute_rank(mu_true, mu_samples)
        tau_rank = compute_rank(tau_true, tau_samples)

        # Convergence diagnostics
        summary = az.summary(idata, var_names=['mu', 'tau'])
        max_rhat = summary['r_hat'].max()
        min_ess = summary['ess_bulk'].min()

        # Divergences
        n_div = idata.sample_stats['diverging'].sum().item()

        # Check convergence
        converged = (max_rhat < 1.05) and (min_ess > 50) and (n_div == 0)

        # Store results
        results['mu_rank'].append(mu_rank)
        results['tau_rank'].append(tau_rank)
        results['mu_true'].append(mu_true)
        results['tau_true'].append(tau_true)
        results['mu_post_mean'].append(mu_samples.mean())
        results['tau_post_mean'].append(tau_samples.mean())
        results['converged'].append(converged)
        results['max_rhat'].append(max_rhat)
        results['min_ess'].append(min_ess)
        results['n_divergences'].append(n_div)

    except Exception as e:
        failed_sims.append((sim, str(e)))
        # Still append None values to maintain array lengths
        for key in results.keys():
            if key in ['mu_rank', 'tau_rank']:
                results[key].append(None)
            elif key in ['mu_true', 'tau_true', 'mu_post_mean', 'tau_post_mean']:
                results[key].append(np.nan)
            elif key == 'converged':
                results[key].append(False)
            elif key in ['max_rhat', 'min_ess', 'n_divergences']:
                results[key].append(np.nan)

# Remove None ranks for analysis
mu_ranks = [r for r in results['mu_rank'] if r is not None]
tau_ranks = [r for r in results['tau_rank'] if r is not None]

print(f"\n{'='*70}")
print("SBC RESULTS")
print(f"{'='*70}")

print(f"\nSimulations completed: {len(mu_ranks)}/{n_sbc_sims}")
print(f"Failed simulations: {len(failed_sims)}")
if failed_sims:
    print(f"  First 5 failures:")
    for sim, error in failed_sims[:5]:
        print(f"    Sim {sim}: {error[:80]}")

# Convergence statistics
n_converged = sum(results['converged'])
print(f"\nConvergence:")
print(f"  Converged simulations: {n_converged}/{n_sbc_sims} ({100*n_converged/n_sbc_sims:.1f}%)")
print(f"  Mean R-hat: {np.nanmean(results['max_rhat']):.4f}")
print(f"  Mean ESS: {np.nanmean(results['min_ess']):.1f}")
print(f"  Total divergences: {np.nansum(results['n_divergences']):.0f}")

# Rank statistics (should be uniform)
n_bins = 20  # Number of bins for uniformity test
expected_per_bin = len(mu_ranks) / n_bins

print(f"\nRank Statistics (testing uniformity):")
print(f"  Total posterior samples per parameter: {n_chains * n_samples}")
print(f"  Expected samples per bin: {expected_per_bin:.1f}")

# Chi-square test for uniformity
from scipy import stats

mu_hist, _ = np.histogram(mu_ranks, bins=n_bins, range=(0, n_chains * n_samples))
mu_chi2, mu_p = stats.chisquare(mu_hist)
print(f"\n  μ ranks:")
print(f"    Chi-square statistic: {mu_chi2:.2f}")
print(f"    p-value: {mu_p:.4f}")
print(f"    Uniform: {'YES' if mu_p > 0.05 else 'NO (potential bias)'}")

tau_hist, _ = np.histogram(tau_ranks, bins=n_bins, range=(0, n_chains * n_samples))
tau_chi2, tau_p = stats.chisquare(tau_hist)
print(f"\n  τ ranks:")
print(f"    Chi-square statistic: {tau_chi2:.2f}")
print(f"    p-value: {tau_p:.4f}")
print(f"    Uniform: {'YES' if tau_p > 0.05 else 'NO (potential bias)'}")

# Coverage calibration
print(f"\n{'='*70}")
print("Coverage Calibration")
print(f"{'='*70}")

# For each credible interval level, check if coverage matches
credible_levels = [0.50, 0.68, 0.90, 0.95]
for level in credible_levels:
    alpha = 1 - level
    lower = alpha / 2
    upper = 1 - alpha / 2

    # Check μ coverage
    mu_covered = 0
    tau_covered = 0
    for i in range(len(results['mu_true'])):
        if not np.isnan(results['mu_true'][i]):
            # Would need full posterior samples here, approximate from rank
            # Rank / total gives approximate quantile
            mu_rank_quantile = results['mu_rank'][i] / (n_chains * n_samples)
            tau_rank_quantile = results['tau_rank'][i] / (n_chains * n_samples)

            if lower <= mu_rank_quantile <= upper:
                mu_covered += 1
            if lower <= tau_rank_quantile <= upper:
                tau_covered += 1

    mu_coverage = mu_covered / len(mu_ranks)
    tau_coverage = tau_covered / len(tau_ranks)

    print(f"\n{100*level:.0f}% Credible Interval:")
    print(f"  μ coverage: {100*mu_coverage:.1f}% (expected {100*level:.0f}%)")
    print(f"  τ coverage: {100*tau_coverage:.1f}% (expected {100*level:.0f}%)")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('/workspace/experiments/experiment_2/simulation_based_validation/sbc_results.csv', index=False)
print(f"\nResults saved to: /workspace/experiments/experiment_2/simulation_based_validation/sbc_results.csv")

# Visualizations
print(f"\n{'='*70}")
print("Creating visualizations...")
print(f"{'='*70}")

# Figure 1: Rank histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# μ ranks
ax = axes[0]
n_total_samples = n_chains * n_samples
bins = np.linspace(0, n_total_samples, n_bins + 1)
ax.hist(mu_ranks, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax.axhline(1/n_total_samples, color='red', linestyle='--', linewidth=2, label='Uniform expectation')
ax.set_xlabel('Rank', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'μ Rank Distribution\n(χ² p={mu_p:.3f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# τ ranks
ax = axes[1]
ax.hist(tau_ranks, bins=bins, density=True, alpha=0.7, color='coral', edgecolor='black')
ax.axhline(1/n_total_samples, color='red', linestyle='--', linewidth=2, label='Uniform expectation')
ax.set_xlabel('Rank', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'τ Rank Distribution\n(χ² p={tau_p:.3f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('SBC Rank Histograms - Hierarchical Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/sbc_ranks.png', dpi=300, bbox_inches='tight')
print("Saved: sbc_ranks.png")
plt.close()

# Figure 2: Recovery plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# μ recovery
ax = axes[0, 0]
ax.scatter(results_df['mu_true'], results_df['mu_post_mean'], alpha=0.5, s=20)
mu_range = [results_df['mu_true'].min(), results_df['mu_true'].max()]
ax.plot(mu_range, mu_range, 'r--', linewidth=2, label='Perfect recovery')
ax.set_xlabel('True μ', fontsize=12)
ax.set_ylabel('Posterior Mean μ', fontsize=12)
ax.set_title('μ Recovery', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# τ recovery (log scale due to half-normal)
ax = axes[0, 1]
ax.scatter(results_df['tau_true'], results_df['tau_post_mean'], alpha=0.5, s=20, color='coral')
tau_range = [0, results_df['tau_true'].max()]
ax.plot(tau_range, tau_range, 'r--', linewidth=2, label='Perfect recovery')
ax.set_xlabel('True τ', fontsize=12)
ax.set_ylabel('Posterior Mean τ', fontsize=12)
ax.set_title('τ Recovery', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Recovery errors
ax = axes[1, 0]
mu_errors = results_df['mu_post_mean'] - results_df['mu_true']
ax.hist(mu_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Zero error\nMean={mu_errors.mean():.2f}')
ax.set_xlabel('μ Error (Posterior - True)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('μ Recovery Error Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
tau_errors = results_df['tau_post_mean'] - results_df['tau_true']
ax.hist(tau_errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Zero error\nMean={tau_errors.mean():.2f}')
ax.set_xlabel('τ Error (Posterior - True)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('τ Recovery Error Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('SBC Parameter Recovery', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/sbc_recovery.png', dpi=300, bbox_inches='tight')
print("Saved: sbc_recovery.png")
plt.close()

# Figure 3: Convergence diagnostics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# R-hat distribution
ax = axes[0]
ax.hist(results_df['max_rhat'].dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
ax.axvline(1.01, color='red', linestyle='--', linewidth=2, label='R-hat = 1.01 threshold')
ax.set_xlabel('Max R-hat', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'R-hat Distribution\nMean={results_df["max_rhat"].mean():.4f}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ESS distribution
ax = axes[1]
ax.hist(results_df['min_ess'].dropna(), bins=30, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(100, color='red', linestyle='--', linewidth=2, label='ESS = 100 threshold')
ax.set_xlabel('Min ESS', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'ESS Distribution\nMean={results_df["min_ess"].mean():.1f}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Divergences
ax = axes[2]
ax.hist(results_df['n_divergences'].dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
ax.set_xlabel('Number of Divergences', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Divergences\nTotal={results_df["n_divergences"].sum():.0f}', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('SBC Convergence Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/sbc_convergence.png', dpi=300, bbox_inches='tight')
print("Saved: sbc_convergence.png")
plt.close()

# Assessment
print(f"\n{'='*70}")
print("ASSESSMENT")
print(f"{'='*70}")

# Decision criteria
rank_uniform = (mu_p > 0.05) and (tau_p > 0.05)
high_convergence = (n_converged / n_sbc_sims) > 0.90
low_divergences = (np.nansum(results['n_divergences']) / n_sbc_sims) < 5

print(f"\nCriteria:")
print(f"  Rank statistics uniform (p > 0.05): {'YES' if rank_uniform else 'NO'}")
print(f"  High convergence rate (> 90%): {'YES' if high_convergence else 'NO'}")
print(f"  Low divergences (< 5 per sim): {'YES' if low_divergences else 'NO'}")

if rank_uniform and high_convergence and low_divergences:
    decision = "PASS"
    reasoning = "Inference is calibrated, ranks uniform, high convergence rate"
elif rank_uniform and high_convergence:
    decision = "PASS (with warnings)"
    reasoning = "Inference calibrated but some divergences present"
elif high_convergence:
    decision = "REVIEW"
    reasoning = "Convergence adequate but rank statistics show potential bias"
else:
    decision = "FAIL"
    reasoning = "Low convergence rate indicates model or inference issues"

print(f"\n{'='*70}")
print(f"DECISION: {decision}")
print(f"REASONING: {reasoning}")
print(f"{'='*70}")

print("\nSBC complete!")
