"""
Fit with long chains for proper convergence (Metropolis-Hastings requires 10-20x HMC iterations)
"""

import numpy as np
import pandas as pd
import arviz as az
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

np.random.seed(42)

BASE_DIR = Path("/workspace/experiments/experiment_1")
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = BASE_DIR / "posterior_inference"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

print("="*80)
print("EXTENDED MCMC SAMPLING FOR CONVERGENCE")
print("="*80)
print(f"Metropolis-Hastings with 10,000 samples per chain")
print("="*80)

# Load data
data = pd.read_csv(DATA_PATH)
N = len(data)
x = data['x'].values
y = data['Y'].values
log_x = np.log(x)

print(f"\nData: N={N}, x ∈ [{x.min():.2f}, {x.max():.2f}], Y ∈ [{y.min():.3f}, {y.max():.3f}]")

# Model functions
def log_prior(params):
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)
    lp_alpha = stats.norm.logpdf(alpha, loc=1.75, scale=0.5)
    lp_beta = stats.norm.logpdf(beta, loc=0.27, scale=0.15)
    lp_sigma = stats.halfnorm.logpdf(sigma, scale=0.2)
    return lp_alpha + lp_beta + lp_sigma + log_sigma

def log_likelihood(params, x_log, y):
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)
    mu = alpha + beta * x_log
    return np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))

def log_posterior(params, x_log, y):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x_log, y)

def metropolis_hastings(log_prob_fn, initial_params, n_samples, proposal_sd,
                        warmup=2000, x_log=None, y=None):
    n_params = len(initial_params)
    samples = []
    current = initial_params.copy()
    current_log_prob = log_prob_fn(current, x_log, y)

    proposal_cov = np.diag(proposal_sd ** 2)
    accepted = 0
    total = n_samples + warmup

    pbar = tqdm(range(total), desc="MCMC Sampling")
    for i in pbar:
        proposed = np.random.multivariate_normal(current, proposal_cov)
        proposed_log_prob = log_prob_fn(proposed, x_log, y)

        log_alpha = proposed_log_prob - current_log_prob

        if np.log(np.random.rand()) < log_alpha:
            current = proposed
            current_log_prob = proposed_log_prob
            accepted += 1

        if i >= warmup:
            samples.append(current.copy())

        if i >= warmup:
            acc_rate = accepted / (i + 1)
            pbar.set_postfix({'accept_rate': f'{acc_rate:.3f}'})

        # Adaptive tuning during warmup
        if i < warmup and i > 0 and i % 200 == 0:
            acc_rate = accepted / (i + 1)
            if acc_rate < 0.20:
                proposal_cov *= 0.9
            elif acc_rate > 0.50:
                proposal_cov *= 1.1

    pbar.close()
    return np.array(samples), accepted / total

# Sampling configuration
initial_params = np.array([1.75, 0.27, np.log(0.12)])
proposal_sd = np.array([0.05, 0.02, 0.1])
n_chains = 4
n_samples = 10000
n_warmup = 2000

print(f"\nSampling: {n_chains} chains × {n_samples} samples = {n_chains * n_samples} total")
print(f"Warmup: {n_warmup} iterations per chain")

start_time = datetime.now()

all_chains = []
acceptance_rates = []

for chain_idx in range(n_chains):
    print(f"\nChain {chain_idx + 1}/{n_chains}:")
    init = initial_params + np.random.normal(0, 0.01, size=3)
    samples, acc_rate = metropolis_hastings(
        log_posterior, init, n_samples, proposal_sd,
        warmup=n_warmup, x_log=log_x, y=y
    )
    all_chains.append(samples)
    acceptance_rates.append(acc_rate)
    print(f"  Acceptance rate: {acc_rate:.3f}")

end_time = datetime.now()
sampling_time = (end_time - start_time).total_seconds()

print(f"\nSampling completed in {sampling_time:.1f} seconds")
print(f"Mean acceptance rate: {np.mean(acceptance_rates):.3f}")

# Convert to InferenceData
print("\nConverting to ArviZ InferenceData...")
chains_array = np.array(all_chains)

alpha_samples = chains_array[:, :, 0]
beta_samples = chains_array[:, :, 1]
sigma_samples = np.exp(chains_array[:, :, 2])

posterior_dict = {
    'alpha': alpha_samples,
    'beta': beta_samples,
    'sigma': sigma_samples,
}

# Compute posterior predictive and log-likelihood
print("Computing posterior predictive and log-likelihood...")
Y_pred = np.zeros((n_chains, n_samples, N))
Y_rep = np.zeros((n_chains, n_samples, N))
log_lik = np.zeros((n_chains, n_samples, N))

for chain_idx in range(n_chains):
    print(f"  Chain {chain_idx + 1}/{n_chains}...")
    for sample_idx in range(n_samples):
        alpha = alpha_samples[chain_idx, sample_idx]
        beta = beta_samples[chain_idx, sample_idx]
        sigma = sigma_samples[chain_idx, sample_idx]

        mu = alpha + beta * log_x
        Y_pred[chain_idx, sample_idx, :] = mu
        Y_rep[chain_idx, sample_idx, :] = np.random.normal(mu, sigma)
        log_lik[chain_idx, sample_idx, :] = stats.norm.logpdf(y, mu, sigma)

posterior_dict['Y_pred'] = Y_pred

idata = az.from_dict(
    posterior=posterior_dict,
    posterior_predictive={'Y': Y_rep},
    log_likelihood={'Y': log_lik},
    observed_data={'Y': y, 'x': x},
    coords={'obs_id': np.arange(N)},
    dims={'Y_pred': ['obs_id'], 'Y': ['obs_id']}
)

# Save
netcdf_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
idata.to_netcdf(netcdf_path)
print(f"\nSaved InferenceData to: {netcdf_path}")

# Convergence diagnostics
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

summary_df = az.summary(idata, var_names=['alpha', 'beta', 'sigma'])
print("\nParameter Summary:")
print(summary_df.to_string())

# Check convergence
max_rhat = summary_df['r_hat'].max()
min_ess_bulk = summary_df['ess_bulk'].min()
min_ess_tail = summary_df['ess_tail'].min()

print(f"\nKey Metrics:")
print(f"  Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"  Min ESS Bulk: {min_ess_bulk:.0f} (target: >= 400)")
print(f"  Min ESS Tail: {min_ess_tail:.0f} (target: >= 400)")

convergence_pass = (max_rhat < 1.01) and (min_ess_bulk >= 400) and (min_ess_tail >= 400)

if convergence_pass:
    print("\n✓ CONVERGENCE ACHIEVED")
else:
    print("\n✗ CONVERGENCE NOT ACHIEVED")
    if max_rhat >= 1.01:
        print(f"  - R-hat too high: {max_rhat:.4f}")
    if min_ess_bulk < 400:
        print(f"  - ESS bulk too low: {min_ess_bulk:.0f}")
    if min_ess_tail < 400:
        print(f"  - ESS tail too low: {min_ess_tail:.0f}")

# Save metrics
convergence_metrics = {
    'alpha': {
        'mean': float(summary_df.loc['alpha', 'mean']),
        'sd': float(summary_df.loc['alpha', 'sd']),
        'hdi_3%': float(summary_df.loc['alpha', 'hdi_3%']),
        'hdi_97%': float(summary_df.loc['alpha', 'hdi_97%']),
        'rhat': float(summary_df.loc['alpha', 'r_hat']),
        'ess_bulk': float(summary_df.loc['alpha', 'ess_bulk']),
        'ess_tail': float(summary_df.loc['alpha', 'ess_tail']),
        'mcse_mean': float(summary_df.loc['alpha', 'mcse_mean']),
    },
    'beta': {
        'mean': float(summary_df.loc['beta', 'mean']),
        'sd': float(summary_df.loc['beta', 'sd']),
        'hdi_3%': float(summary_df.loc['beta', 'hdi_3%']),
        'hdi_97%': float(summary_df.loc['beta', 'hdi_97%']),
        'rhat': float(summary_df.loc['beta', 'r_hat']),
        'ess_bulk': float(summary_df.loc['beta', 'ess_bulk']),
        'ess_tail': float(summary_df.loc['beta', 'ess_tail']),
        'mcse_mean': float(summary_df.loc['beta', 'mcse_mean']),
    },
    'sigma': {
        'mean': float(summary_df.loc['sigma', 'mean']),
        'sd': float(summary_df.loc['sigma', 'sd']),
        'hdi_3%': float(summary_df.loc['sigma', 'hdi_3%']),
        'hdi_97%': float(summary_df.loc['sigma', 'hdi_97%']),
        'rhat': float(summary_df.loc['sigma', 'r_hat']),
        'ess_bulk': float(summary_df.loc['sigma', 'ess_bulk']),
        'ess_tail': float(summary_df.loc['sigma', 'ess_tail']),
        'mcse_mean': float(summary_df.loc['sigma', 'mcse_mean']),
    },
    'sampling': {
        'method': 'Metropolis-Hastings',
        'chains': n_chains,
        'iter_warmup': n_warmup,
        'iter_sampling': n_samples,
        'total_samples': n_chains * n_samples,
        'sampling_time_seconds': sampling_time,
        'mean_acceptance_rate': np.mean(acceptance_rates),
    },
    'convergence_pass': convergence_pass,
}

metrics_path = DIAGNOSTICS_DIR / "convergence_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(convergence_metrics, f, indent=2)

print(f"\nMetrics saved to: {metrics_path}")
print("="*80)
