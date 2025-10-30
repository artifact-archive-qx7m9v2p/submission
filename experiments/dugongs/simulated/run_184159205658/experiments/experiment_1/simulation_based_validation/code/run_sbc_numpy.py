"""
Simulation-Based Calibration (SBC) for Logarithmic Regression Model
Using NumPy/SciPy with MCMC sampling

This script performs SBC to validate the model's ability to recover known parameters.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIMULATIONS = 150
N_MCMC_SAMPLES = 4000  # Total across all chains
N_WARMUP = 1000

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

# Precompute log(x)
log_x_obs = np.log(x_obs)

def log_prior(beta0, beta1, sigma):
    """Compute log prior density"""
    if sigma <= 0:
        return -np.inf
    lp = stats.norm.logpdf(beta0, PRIOR_BETA0_MEAN, PRIOR_BETA0_SD)
    lp += stats.norm.logpdf(beta1, PRIOR_BETA1_MEAN, PRIOR_BETA1_SD)
    lp += stats.expon.logpdf(sigma, scale=1/PRIOR_SIGMA_RATE)
    return lp

def log_likelihood(beta0, beta1, sigma, y, log_x):
    """Compute log likelihood"""
    if sigma <= 0:
        return -np.inf
    mu = beta0 + beta1 * log_x
    return np.sum(stats.norm.logpdf(y, mu, sigma))

def log_posterior(params, y, log_x):
    """Compute log posterior density"""
    beta0, beta1, log_sigma = params
    sigma = np.exp(log_sigma)  # Use log scale for sigma
    return log_prior(beta0, beta1, sigma) + log_likelihood(beta0, beta1, sigma, y, log_x)

def metropolis_hastings(y, log_x, n_samples=4000, n_warmup=1000):
    """Simple Metropolis-Hastings MCMC sampler"""

    # Find MAP estimate as starting point
    def neg_log_post(params):
        return -log_posterior(params, y, log_x)

    # Initialize at prior means
    init_params = [PRIOR_BETA0_MEAN, PRIOR_BETA1_MEAN, np.log(1/PRIOR_SIGMA_RATE)]

    # Optimize to find MAP
    res = minimize(neg_log_post, init_params, method='Nelder-Mead')
    current = res.x if res.success else init_params

    # Proposal standard deviations (tuned)
    proposal_sd = np.array([0.05, 0.02, 0.1])

    samples = []
    accepts = 0

    total_iterations = n_samples + n_warmup

    for i in range(total_iterations):
        # Propose new parameters
        proposed = current + np.random.normal(0, proposal_sd, size=3)

        # Compute acceptance ratio
        log_alpha = log_posterior(proposed, y, log_x) - log_posterior(current, y, log_x)

        # Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            current = proposed
            accepts += 1

        # Store samples after warmup
        if i >= n_warmup:
            beta0, beta1, log_sigma = current
            samples.append([beta0, beta1, np.exp(log_sigma)])

    samples = np.array(samples)
    acceptance_rate = accepts / total_iterations

    return samples, acceptance_rate

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
    'acceptance_rate': [],
    'ess_beta0': [],
    'ess_beta1': [],
    'ess_sigma': []
}

def compute_ess(samples):
    """Compute effective sample size using autocorrelation"""
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)

    if var == 0:
        return n

    # Compute autocorrelation
    max_lag = min(n // 4, 1000)
    autocorr = []

    for lag in range(1, max_lag):
        if lag >= n:
            break
        c = np.mean((samples[:-lag] - mean) * (samples[lag:] - mean))
        autocorr.append(c / var)

        # Stop when autocorrelation becomes small
        if len(autocorr) > 5 and autocorr[-1] < 0.05:
            break

    # ESS formula
    ess = n / (1 + 2 * np.sum(autocorr))
    return max(1, ess)

# Run SBC simulations
n_failed = 0

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

    # 3. Fit model to synthetic data using MCMC
    try:
        samples, accept_rate = metropolis_hastings(y_sim, log_x_obs, N_MCMC_SAMPLES, N_WARMUP)

        beta0_samples = samples[:, 0]
        beta1_samples = samples[:, 1]
        sigma_samples = samples[:, 2]

        # 4. Compute posterior summaries
        post_mean_beta0 = np.mean(beta0_samples)
        post_mean_beta1 = np.mean(beta1_samples)
        post_mean_sigma = np.mean(sigma_samples)

        post_sd_beta0 = np.std(beta0_samples, ddof=1)
        post_sd_beta1 = np.std(beta1_samples, ddof=1)
        post_sd_sigma = np.std(sigma_samples, ddof=1)

        q025_beta0, q975_beta0 = np.percentile(beta0_samples, [2.5, 97.5])
        q025_beta1, q975_beta1 = np.percentile(beta1_samples, [2.5, 97.5])
        q025_sigma, q975_sigma = np.percentile(sigma_samples, [2.5, 97.5])

        # 5. Compute ranks (for SBC)
        rank_beta0 = np.sum(beta0_samples < true_beta0)
        rank_beta1 = np.sum(beta1_samples < true_beta1)
        rank_sigma = np.sum(sigma_samples < true_sigma)

        # Compute ESS
        ess_beta0 = compute_ess(beta0_samples)
        ess_beta1 = compute_ess(beta1_samples)
        ess_sigma = compute_ess(sigma_samples)

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
        results['acceptance_rate'].append(accept_rate)
        results['ess_beta0'].append(ess_beta0)
        results['ess_beta1'].append(ess_beta1)
        results['ess_sigma'].append(ess_sigma)

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

    # Mean ESS
    mean_ess = np.mean(df_results[f'ess_{param}'].values)

    print(f"  Bias: {bias:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Coverage (95% CI): {coverage:.1f}%")
    print(f"  Prior SD: {prior_sd:.4f}")
    print(f"  Posterior SD (mean): {post_sd_mean:.4f}")
    print(f"  Shrinkage: {shrinkage*100:.1f}%")
    print(f"  Mean ESS: {mean_ess:.0f}")

print()
print("="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)
print(f"Mean acceptance rate: {df_results['acceptance_rate'].mean():.3f}")
print(f"Runs with ESS > 100 (all params): {np.mean((df_results['ess_beta0'] > 100) & (df_results['ess_beta1'] > 100) & (df_results['ess_sigma'] > 100))*100:.1f}%")
print(f"Mean ESS beta0: {df_results['ess_beta0'].mean():.0f}")
print(f"Mean ESS beta1: {df_results['ess_beta1'].mean():.0f}")
print(f"Mean ESS sigma: {df_results['ess_sigma'].mean():.0f}")

print()
print("SBC analysis complete!")
