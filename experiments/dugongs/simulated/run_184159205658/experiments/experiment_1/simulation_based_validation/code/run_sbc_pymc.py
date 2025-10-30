"""
Simulation-Based Calibration (SBC) for Logarithmic Regression Model using PyMC

This script performs SBC to validate the model's ability to recover known parameters.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIMULATIONS = 150
N_DRAWS = 1000
N_TUNE = 500
N_CHAINS = 4

# Load observed x values
data = pd.read_csv('/workspace/data/data.csv')
x_obs = data['x'].values
N = len(x_obs)

print(f"Starting SBC with {N_SIMULATIONS} simulations")
print(f"N = {N} observations")
print(f"x range: [{x_obs.min()}, {x_obs.max()}]")
print()

# Prior specifications
PRIOR_BETA0_MEAN = 1.73
PRIOR_BETA0_SD = 0.5
PRIOR_BETA1_MEAN = 0.28
PRIOR_BETA1_SD = 0.15
PRIOR_SIGMA_RATE = 5.0

# Storage for results
results = {
    'sim_id': [],
    'true_beta0': [],
    'true_beta1': [],
    'true_sigma': [],
    'post_mean_beta0': [],
    'post_mean_beta1': [],
    'post_mean_sigma': [],
    'post_sd_beta0': [],
    'post_sd_beta1': [],
    'post_sd_sigma': [],
    'q025_beta0': [],
    'q025_beta1': [],
    'q025_sigma': [],
    'q975_beta0': [],
    'q975_beta1': [],
    'q975_sigma': [],
    'rank_beta0': [],
    'rank_beta1': [],
    'rank_sigma': [],
    'max_rhat': [],
    'min_ess_bulk': [],
    'n_divergences': [],
    'converged': []
}

# Precompute log(x)
log_x_obs = np.log(x_obs)

# Run SBC simulations
n_failed = 0
n_divergent_total = 0

for sim in range(N_SIMULATIONS):
    if (sim + 1) % 10 == 0:
        print(f"Simulation {sim + 1}/{N_SIMULATIONS}")

    # 1. Draw true parameters from priors
    true_beta0 = np.random.normal(PRIOR_BETA0_MEAN, PRIOR_BETA0_SD)
    true_beta1 = np.random.normal(PRIOR_BETA1_MEAN, PRIOR_BETA1_SD)
    true_sigma = np.random.exponential(1/PRIOR_SIGMA_RATE)

    # 2. Generate synthetic data
    mu = true_beta0 + true_beta1 * log_x_obs
    y_sim = np.random.normal(mu, true_sigma)

    # 3. Fit model to synthetic data
    try:
        with pm.Model() as model:
            # Priors
            beta0 = pm.Normal('beta0', mu=PRIOR_BETA0_MEAN, sigma=PRIOR_BETA0_SD)
            beta1 = pm.Normal('beta1', mu=PRIOR_BETA1_MEAN, sigma=PRIOR_BETA1_SD)
            sigma = pm.Exponential('sigma', lam=PRIOR_SIGMA_RATE)

            # Expected value
            mu_model = beta0 + beta1 * log_x_obs

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu_model, sigma=sigma, observed=y_sim)

            # Sample
            trace = pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                cores=4,
                return_inferencedata=True,
                random_seed=42 + sim,
                progressbar=False
            )

        # Extract posterior samples
        beta0_samples = trace.posterior['beta0'].values.flatten()
        beta1_samples = trace.posterior['beta1'].values.flatten()
        sigma_samples = trace.posterior['sigma'].values.flatten()

        # Check convergence
        summary = az.summary(trace, var_names=['beta0', 'beta1', 'sigma'])
        max_rhat = summary['r_hat'].max()
        min_ess = summary['ess_bulk'].min()
        converged = max_rhat < 1.01 and min_ess > 100

        # Count divergences
        n_div = trace.sample_stats['diverging'].sum().values.item()
        if n_div > 0:
            n_divergent_total += 1

        # 4. Compute posterior summaries
        post_mean_beta0 = np.mean(beta0_samples)
        post_mean_beta1 = np.mean(beta1_samples)
        post_mean_sigma = np.mean(sigma_samples)

        post_sd_beta0 = np.std(beta0_samples)
        post_sd_beta1 = np.std(beta1_samples)
        post_sd_sigma = np.std(sigma_samples)

        q025_beta0, q975_beta0 = np.percentile(beta0_samples, [2.5, 97.5])
        q025_beta1, q975_beta1 = np.percentile(beta1_samples, [2.5, 97.5])
        q025_sigma, q975_sigma = np.percentile(sigma_samples, [2.5, 97.5])

        # 5. Compute ranks (for SBC)
        rank_beta0 = np.sum(beta0_samples < true_beta0)
        rank_beta1 = np.sum(beta1_samples < true_beta1)
        rank_sigma = np.sum(sigma_samples < true_sigma)

        # Store results
        results['sim_id'].append(sim)
        results['true_beta0'].append(true_beta0)
        results['true_beta1'].append(true_beta1)
        results['true_sigma'].append(true_sigma)
        results['post_mean_beta0'].append(post_mean_beta0)
        results['post_mean_beta1'].append(post_mean_beta1)
        results['post_mean_sigma'].append(post_mean_sigma)
        results['post_sd_beta0'].append(post_sd_beta0)
        results['post_sd_beta1'].append(post_sd_beta1)
        results['post_sd_sigma'].append(post_sd_sigma)
        results['q025_beta0'].append(q025_beta0)
        results['q025_beta1'].append(q025_beta1)
        results['q025_sigma'].append(q025_sigma)
        results['q975_beta0'].append(q975_beta0)
        results['q975_beta1'].append(q975_beta1)
        results['q975_sigma'].append(q975_sigma)
        results['rank_beta0'].append(rank_beta0)
        results['rank_beta1'].append(rank_beta1)
        results['rank_sigma'].append(rank_sigma)
        results['max_rhat'].append(max_rhat)
        results['min_ess_bulk'].append(min_ess)
        results['n_divergences'].append(n_div)
        results['converged'].append(converged)

    except Exception as e:
        print(f"  Simulation {sim} failed: {str(e)}")
        n_failed += 1
        continue

print()
print("="*60)
print("SBC SIMULATION COMPLETE")
print("="*60)
print(f"Successful simulations: {len(results['sim_id'])}/{N_SIMULATIONS}")
print(f"Failed simulations: {n_failed}")
print(f"Simulations with divergences: {n_divergent_total}")
print()

# Convert to DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv('/workspace/experiments/experiment_1/simulation_based_validation/code/sbc_results.csv', index=False)
print(f"Results saved to sbc_results.csv")
print()

# Compute summary statistics
print("="*60)
print("PARAMETER RECOVERY METRICS")
print("="*60)

for param in ['beta0', 'beta1', 'sigma']:
    print(f"\n{param.upper()}:")

    true_vals = df_results[f'true_{param}'].values
    post_means = df_results[f'post_mean_{param}'].values
    q025 = df_results[f'q025_{param}'].values
    q975 = df_results[f'q975_{param}'].values

    # Bias
    bias = np.mean(post_means - true_vals)

    # RMSE
    rmse = np.sqrt(np.mean((post_means - true_vals)**2))

    # Coverage
    in_interval = (true_vals >= q025) & (true_vals <= q975)
    coverage = np.mean(in_interval) * 100

    # Shrinkage
    if param == 'beta0':
        prior_sd = PRIOR_BETA0_SD
    elif param == 'beta1':
        prior_sd = PRIOR_BETA1_SD
    else:
        prior_sd = 1/PRIOR_SIGMA_RATE  # std of exponential

    post_sd_mean = np.mean(df_results[f'post_sd_{param}'].values)
    shrinkage = 1 - (post_sd_mean / prior_sd)

    print(f"  Bias: {bias:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Coverage (95% CI): {coverage:.1f}%")
    print(f"  Prior SD: {prior_sd:.4f}")
    print(f"  Posterior SD (mean): {post_sd_mean:.4f}")
    print(f"  Shrinkage: {shrinkage*100:.1f}%")

print()
print("="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)
print(f"Runs with Rhat < 1.01: {np.mean(df_results['max_rhat'] < 1.01)*100:.1f}%")
print(f"Runs with ESS > 100: {np.mean(df_results['min_ess_bulk'] > 100)*100:.1f}%")
print(f"Runs converged: {np.mean(df_results['converged'])*100:.1f}%")
print(f"Mean max Rhat: {df_results['max_rhat'].mean():.4f}")
print(f"Mean min ESS: {df_results['min_ess_bulk'].mean():.0f}")

print()
print("SBC analysis complete!")
