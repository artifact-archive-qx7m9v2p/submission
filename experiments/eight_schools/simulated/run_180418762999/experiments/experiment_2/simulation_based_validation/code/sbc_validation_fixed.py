"""
Simulation-Based Calibration for Experiment 2: Hierarchical Partial Pooling Model

FIXED VERSION for PyMC 5.x
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pymc as pm
import arviz as az
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("SIMULATION-BASED CALIBRATION: Experiment 2 - Hierarchical Partial Pooling")
print("="*80)
print()

# Configuration
N_SIMULATIONS = 30
N_GROUPS = 8
SIGMA_OBS = np.array([15, 10, 16, 11, 9, 11, 10, 18])

# Prior hyperparameters
PRIOR_MU_MEAN = 10
PRIOR_MU_SD = 20
PRIOR_TAU_SD = 10

# MCMC settings
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42

print(f"Configuration:")
print(f"  N_SIMULATIONS: {N_SIMULATIONS}")
print(f"  MCMC: {N_DRAWS} draws × {N_CHAINS} chains")
print()

def compute_rank_statistic(true_value, posterior_samples):
    """Compute rank of true value within posterior samples."""
    rank = np.sum(posterior_samples < true_value)
    return rank

def fit_hierarchical_model(y_data, sigma_data):
    """Fit hierarchical model and return results."""
    with pm.Model() as model:
        # Hyperpriors
        mu = pm.Normal('mu', mu=PRIOR_MU_MEAN, sigma=PRIOR_MU_SD)
        tau = pm.HalfNormal('tau', sigma=PRIOR_TAU_SD)

        # Non-centered parameterization
        theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=N_GROUPS)
        theta = pm.Deterministic('theta', mu + tau * theta_raw)

        # Likelihood
        y = pm.Normal('y', mu=theta, sigma=sigma_data, observed=y_data)

        # Sample
        try:
            idata = pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                target_accept=TARGET_ACCEPT,
                return_inferencedata=True,
                progressbar=False,
                random_seed=RANDOM_SEED,
                discard_tuned_samples=True
            )

            # Extract samples
            mu_samples = idata.posterior['mu'].values.flatten()
            tau_samples = idata.posterior['tau'].values.flatten()

            # Diagnostics
            summary = az.summary(idata, var_names=['mu', 'tau'])
            mu_rhat = summary.loc['mu', 'r_hat']
            tau_rhat = summary.loc['tau', 'r_hat']
            mu_ess = summary.loc['mu', 'ess_bulk']
            tau_ess = summary.loc['tau', 'ess_bulk']

            # Count divergences
            divergences = int(idata.sample_stats['diverging'].sum().values)

            return {
                'mu_samples': mu_samples,
                'tau_samples': tau_samples,
                'divergences': divergences,
                'mu_rhat': float(mu_rhat),
                'tau_rhat': float(tau_rhat),
                'mu_ess': float(mu_ess),
                'tau_ess': float(tau_ess),
                'converged': mu_rhat < 1.01 and tau_rhat < 1.01
            }

        except Exception as e:
            print(f"  Error: {str(e)}")
            return None

# Run SBC
print("Starting SBC iterations...")
print("-"*80)

results = {
    'mu_true': [],
    'tau_true': [],
    'mu_rank': [],
    'tau_rank': [],
    'mu_posterior_samples': [],
    'tau_posterior_samples': [],
    'mu_mean': [],
    'tau_mean': [],
    'mu_rhat': [],
    'tau_rhat': [],
    'mu_ess': [],
    'tau_ess': [],
    'divergences': [],
    'converged': [],
    'iteration': []
}

n_failures = 0
start_time = datetime.now()

for sim_idx in range(N_SIMULATIONS):
    if (sim_idx + 1) % 10 == 0:
        print(f"  Completed {sim_idx + 1}/{N_SIMULATIONS} simulations...")

    # Sample true parameters
    mu_true = np.random.normal(PRIOR_MU_MEAN, PRIOR_MU_SD)
    tau_true = np.abs(np.random.normal(0, PRIOR_TAU_SD))
    theta_true = np.random.normal(mu_true, tau_true, N_GROUPS)

    # Generate synthetic data
    y_sim = np.array([np.random.normal(theta_true[j], SIGMA_OBS[j])
                      for j in range(N_GROUPS)])

    # Fit model
    fit_result = fit_hierarchical_model(y_sim, SIGMA_OBS)

    if fit_result is None:
        n_failures += 1
        continue

    # Compute rank statistics
    mu_rank = compute_rank_statistic(mu_true, fit_result['mu_samples'])
    tau_rank = compute_rank_statistic(tau_true, fit_result['tau_samples'])

    # Store results
    results['mu_true'].append(mu_true)
    results['tau_true'].append(tau_true)
    results['mu_rank'].append(mu_rank)
    results['tau_rank'].append(tau_rank)
    results['mu_posterior_samples'].append(fit_result['mu_samples'])
    results['tau_posterior_samples'].append(fit_result['tau_samples'])
    results['mu_mean'].append(fit_result['mu_samples'].mean())
    results['tau_mean'].append(fit_result['tau_samples'].mean())
    results['mu_rhat'].append(fit_result['mu_rhat'])
    results['tau_rhat'].append(fit_result['tau_rhat'])
    results['mu_ess'].append(fit_result['mu_ess'])
    results['tau_ess'].append(fit_result['tau_ess'])
    results['divergences'].append(fit_result['divergences'])
    results['converged'].append(fit_result['converged'])
    results['iteration'].append(sim_idx)

end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()

print()
print(f"SBC completed in {elapsed_time:.1f} seconds")
print(f"  Successful simulations: {len(results['mu_true'])}/{N_SIMULATIONS}")
print(f"  Failed simulations: {n_failures}")
print()

# Convert to arrays
n_success = len(results['mu_true'])
mu_true_arr = np.array(results['mu_true'])
tau_true_arr = np.array(results['tau_true'])
mu_rank_arr = np.array(results['mu_rank'])
tau_rank_arr = np.array(results['tau_rank'])
mu_mean_arr = np.array(results['mu_mean'])
tau_mean_arr = np.array(results['tau_mean'])
divergences_arr = np.array(results['divergences'])
mu_rhat_arr = np.array(results['mu_rhat'])
tau_rhat_arr = np.array(results['tau_rhat'])
mu_ess_arr = np.array(results['mu_ess'])
tau_ess_arr = np.array(results['tau_ess'])

# Compute summary statistics
print("Computing summary statistics...")
print("-"*80)

# Rank uniformity tests
n_bins = 10
expected_per_bin = n_success / n_bins

mu_hist, _ = np.histogram(mu_rank_arr, bins=n_bins, range=(0, N_DRAWS * N_CHAINS))
mu_chi2_stat = np.sum((mu_hist - expected_per_bin)**2 / expected_per_bin)
mu_chi2_pval = 1 - stats.chi2.cdf(mu_chi2_stat, df=n_bins-1)

tau_hist, _ = np.histogram(tau_rank_arr, bins=n_bins, range=(0, N_DRAWS * N_CHAINS))
tau_chi2_stat = np.sum((tau_hist - expected_per_bin)**2 / expected_per_bin)
tau_chi2_pval = 1 - stats.chi2.cdf(tau_chi2_stat, df=n_bins-1)

print("Rank Uniformity Tests:")
print(f"  mu:  χ² = {mu_chi2_stat:.2f}, p = {mu_chi2_pval:.4f} {'[PASS]' if mu_chi2_pval > 0.05 else '[FAIL]'}")
print(f"  tau: χ² = {tau_chi2_stat:.2f}, p = {tau_chi2_pval:.4f} {'[PASS]' if tau_chi2_pval > 0.05 else '[FAIL]'}")
print()

# Coverage analysis
def compute_coverage(true_values, posterior_samples, level=0.9):
    n = len(true_values)
    in_interval = 0
    alpha = (1 - level) / 2
    for i in range(n):
        lower = np.percentile(posterior_samples[i], alpha * 100)
        upper = np.percentile(posterior_samples[i], (1 - alpha) * 100)
        if lower <= true_values[i] <= upper:
            in_interval += 1
    return in_interval / n

mu_samples_2d = np.array(results['mu_posterior_samples'])
tau_samples_2d = np.array(results['tau_posterior_samples'])

mu_coverage_90 = compute_coverage(mu_true_arr, mu_samples_2d, 0.90)
mu_coverage_95 = compute_coverage(mu_true_arr, mu_samples_2d, 0.95)
tau_coverage_90 = compute_coverage(tau_true_arr, tau_samples_2d, 0.90)
tau_coverage_95 = compute_coverage(tau_true_arr, tau_samples_2d, 0.95)

print("Coverage Analysis:")
print(f"  mu (90%):  {mu_coverage_90:.3f} (target: 0.90)")
print(f"  mu (95%):  {mu_coverage_95:.3f} (target: 0.95)")
print(f"  tau (90%): {tau_coverage_90:.3f} (target: 0.90)")
print(f"  tau (95%): {tau_coverage_95:.3f} (target: 0.95)")
print()

# Bias analysis
mu_bias = (mu_mean_arr - mu_true_arr).mean()
tau_bias = (tau_mean_arr - tau_true_arr).mean()
mu_bias_sd = (mu_mean_arr - mu_true_arr).std()
tau_bias_sd = (tau_mean_arr - tau_true_arr).std()

print("Bias Analysis:")
print(f"  mu:  {mu_bias:.3f} ± {mu_bias_sd:.3f}")
print(f"  tau: {tau_bias:.3f} ± {tau_bias_sd:.3f}")
print()

# Convergence
print("Convergence:")
print(f"  Avg divergences: {divergences_arr.mean():.1f} ({100*divergences_arr.mean()/(N_DRAWS*N_CHAINS):.2f}%)")
print(f"  Avg R-hat (mu): {mu_rhat_arr.mean():.4f}")
print(f"  Avg R-hat (tau): {tau_rhat_arr.mean():.4f}")
print(f"  Avg ESS (mu): {mu_ess_arr.mean():.0f}")
print(f"  Avg ESS (tau): {tau_ess_arr.mean():.0f}")
print()

# Save results
print("Saving results...")

results_df = pd.DataFrame({
    'iteration': results['iteration'],
    'mu_true': results['mu_true'],
    'tau_true': results['tau_true'],
    'mu_rank': results['mu_rank'],
    'tau_rank': results['tau_rank'],
    'mu_mean': results['mu_mean'],
    'tau_mean': results['tau_mean'],
    'mu_bias': np.array(results['mu_mean']) - np.array(results['mu_true']),
    'tau_bias': np.array(results['tau_mean']) - np.array(results['tau_true']),
    'mu_rhat': results['mu_rhat'],
    'tau_rhat': results['tau_rhat'],
    'mu_ess': results['mu_ess'],
    'tau_ess': results['tau_ess'],
    'divergences': results['divergences'],
    'converged': results['converged']
})

results_df.to_csv('/workspace/experiments/experiment_2/simulation_based_validation/diagnostics/sbc_results.csv', index=False)
print("  Saved: diagnostics/sbc_results.csv")

summary_stats = {
    'n_simulations': N_SIMULATIONS,
    'n_successful': n_success,
    'n_failures': n_failures,
    'elapsed_time_seconds': elapsed_time,
    'rank_uniformity': {
        'mu_chi2_stat': float(mu_chi2_stat),
        'mu_chi2_pval': float(mu_chi2_pval),
        'mu_pass': bool(mu_chi2_pval > 0.05),
        'tau_chi2_stat': float(tau_chi2_stat),
        'tau_chi2_pval': float(tau_chi2_pval),
        'tau_pass': bool(tau_chi2_pval > 0.05)
    },
    'coverage': {
        'mu_90': float(mu_coverage_90),
        'mu_95': float(mu_coverage_95),
        'tau_90': float(tau_coverage_90),
        'tau_95': float(tau_coverage_95),
        'mu_pass': bool(0.85 <= mu_coverage_90 <= 0.95),
        'tau_pass': bool(0.80 <= tau_coverage_90 <= 0.95)
    },
    'bias': {
        'mu_mean': float(mu_bias),
        'mu_sd': float(mu_bias_sd),
        'tau_mean': float(tau_bias),
        'tau_sd': float(tau_bias_sd),
        'mu_pass': bool(abs(mu_bias) < 0.2 * PRIOR_MU_SD),
        'tau_pass': bool(abs(tau_bias) < 0.2 * PRIOR_TAU_SD)
    },
    'convergence': {
        'avg_divergences': float(divergences_arr.mean()),
        'max_divergences': float(divergences_arr.max()),
        'pct_with_divergences': float((divergences_arr > 0).sum() / n_success),
        'avg_mu_rhat': float(mu_rhat_arr.mean()),
        'max_mu_rhat': float(mu_rhat_arr.max()),
        'avg_tau_rhat': float(tau_rhat_arr.mean()),
        'max_tau_rhat': float(tau_rhat_arr.max()),
        'avg_mu_ess': float(mu_ess_arr.mean()),
        'avg_tau_ess': float(tau_ess_arr.mean()),
        'divergences_pass': bool(divergences_arr.mean() / (N_DRAWS * N_CHAINS) < 0.05),
        'rhat_pass': bool(mu_rhat_arr.max() < 1.01 and tau_rhat_arr.max() < 1.01)
    }
}

with open('/workspace/experiments/experiment_2/simulation_based_validation/diagnostics/summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)
print("  Saved: diagnostics/summary_stats.json")

print()
print("="*80)
print("SBC validation complete!")
print("="*80)
