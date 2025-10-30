#!/usr/bin/env python3
"""
Simulation-Based Calibration for Negative Binomial Quadratic Model
Fast version with reduced MCMC iterations for initial testing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load real data to get year values
data = pd.read_csv("/workspace/data/data.csv")
year_values = data['year'].values

# Model specification - ADJUSTED PRIORS
PRIORS = {
    'beta_0': {'mean': 4.7, 'sd': 0.3},
    'beta_1': {'mean': 0.8, 'sd': 0.2},
    'beta_2': {'mean': 0.3, 'sd': 0.1},
    'phi': {'shape': 2, 'rate': 0.5}
}

# Reduced for speed
N_SIMS = 100
N_OBS = len(year_values)
N_WARMUP = 1000  # Reduced from 2000
N_SAMPLES = 1000  # Reduced from 2000
N_CHAINS = 4

print("="*80)
print("SIMULATION-BASED CALIBRATION (SBC) - FAST VERSION")
print("Negative Binomial Quadratic Model - Experiment 1")
print("="*80)
print(f"\nConfiguration:")
print(f"  Number of simulations: {N_SIMS}")
print(f"  MCMC: {N_CHAINS} chains × {N_SAMPLES} samples (after {N_WARMUP} warmup)")
print(f"  Observations per simulation: {N_OBS}")

def log_prior(params):
    """Log prior density"""
    beta_0, beta_1, beta_2, log_phi = params
    phi = np.exp(log_phi)

    lp = 0
    lp += stats.norm.logpdf(beta_0, PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
    lp += stats.norm.logpdf(beta_1, PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
    lp += stats.norm.logpdf(beta_2, PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd'])
    lp += stats.gamma.logpdf(phi, PRIORS['phi']['shape'], scale=1/PRIORS['phi']['rate'])
    lp += log_phi  # Jacobian

    return lp

def log_likelihood(params, y, year):
    """Log likelihood for negative binomial model"""
    beta_0, beta_1, beta_2, log_phi = params
    phi = np.exp(log_phi)

    if phi <= 0:
        return -np.inf

    log_mu = beta_0 + beta_1 * year + beta_2 * year**2
    mu = np.exp(log_mu)

    if np.any(mu <= 0) or np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
        return -np.inf

    ll = 0
    for i in range(len(y)):
        if mu[i] > 0 and phi > 0:
            p = phi / (mu[i] + phi)
            ll += stats.nbinom.logpmf(y[i], phi, p)
        else:
            return -np.inf

    return ll

def log_posterior(params, y, year):
    """Log posterior density"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, y, year)

def metropolis_hastings_chain(y, year, n_warmup, n_samples, initial_params=None,
                               proposal_sd=None, chain_id=0):
    """Single chain Metropolis-Hastings MCMC"""
    if initial_params is None:
        beta_0 = np.random.normal(PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
        beta_1 = np.random.normal(PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
        beta_2 = np.random.normal(PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd'])
        phi = np.random.gamma(PRIORS['phi']['shape'], 1/PRIORS['phi']['rate'])
        current = np.array([beta_0, beta_1, beta_2, np.log(phi)])
    else:
        current = initial_params.copy()

    if proposal_sd is None:
        proposal_sd = np.array([0.15, 0.10, 0.05, 0.2])  # Tuned proposal

    samples = []
    n_total = n_warmup + n_samples
    n_accepted = 0

    current_log_post = log_posterior(current, y, year)

    for i in range(n_total):
        proposal = current + np.random.normal(0, proposal_sd, size=4)
        proposal_log_post = log_posterior(proposal, y, year)
        log_accept_ratio = proposal_log_post - current_log_post

        if np.log(np.random.rand()) < log_accept_ratio:
            current = proposal
            current_log_post = proposal_log_post
            if i >= n_warmup:
                n_accepted += 1

        if i >= n_warmup:
            samples.append(current.copy())

        # Adaptive tuning during warmup
        if i < n_warmup and i > 0 and i % 200 == 0:
            recent_accept = n_accepted / min(i, 200)
            if recent_accept < 0.2:
                proposal_sd *= 0.9
            elif recent_accept > 0.5:
                proposal_sd *= 1.1

    acceptance_rate = n_accepted / n_samples
    samples = np.array(samples)

    return samples, acceptance_rate

def compute_rhat(chains):
    """Compute R-hat convergence diagnostic"""
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    n_params = chains[0].shape[1]

    rhats = []
    for p in range(n_params):
        chain_means = [chains[c][:, p].mean() for c in range(n_chains)]
        chain_vars = [chains[c][:, p].var(ddof=1) for c in range(n_chains)]

        B = n_samples * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

        rhat = np.sqrt(var_plus / W) if W > 0 else np.inf
        rhats.append(rhat)

    return np.array(rhats)

def compute_ess(chains):
    """Compute effective sample size (simple approximation)"""
    n_params = chains[0].shape[1]
    ess_values = []

    for p in range(n_params):
        combined = np.concatenate([chains[c][:, p] for c in range(len(chains))])
        acf = np.correlate(combined - combined.mean(), combined - combined.mean(), mode='full')
        acf = acf[len(acf)//2:] / acf.max()

        tau = 1
        for i in range(1, min(len(acf), 100)):
            if acf[i] < 0.05:
                break
            tau += 2 * acf[i]

        ess = len(combined) / tau
        ess_values.append(ess)

    return np.array(ess_values)

def draw_from_prior():
    """Draw parameters from prior distributions"""
    beta_0 = np.random.normal(PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
    beta_1 = np.random.normal(PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
    beta_2 = np.random.normal(PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd'])
    phi = np.random.gamma(PRIORS['phi']['shape'], 1/PRIORS['phi']['rate'])

    return {'beta_0': beta_0, 'beta_1': beta_1, 'beta_2': beta_2, 'phi': phi}

def generate_data(params, year):
    """Generate synthetic data from negative binomial model"""
    N = len(year)
    log_mu = params['beta_0'] + params['beta_1'] * year + params['beta_2'] * year**2
    mu = np.exp(log_mu)

    y = []
    for i in range(N):
        p = params['phi'] / (mu[i] + params['phi'])
        y.append(np.random.negative_binomial(params['phi'], p))

    return np.array(y)

def compute_rank(true_value, posterior_samples):
    """Compute rank of true value in posterior samples"""
    return np.sum(posterior_samples < true_value)

# Storage for results
sbc_results = {
    'beta_0': {'ranks': [], 'true': [], 'mean': [], 'median': [], 'q025': [], 'q975': []},
    'beta_1': {'ranks': [], 'true': [], 'mean': [], 'median': [], 'q025': [], 'q975': []},
    'beta_2': {'ranks': [], 'true': [], 'mean': [], 'median': [], 'q025': [], 'q975': []},
    'phi': {'ranks': [], 'true': [], 'mean': [], 'median': [], 'q025': [], 'q975': []}
}

convergence_stats = {
    'converged': [],
    'max_rhat': [],
    'min_ess': [],
    'mean_acceptance': []
}

failed_simulations = []

print("\nRunning SBC simulations...")
print("-"*80)

for sim in tqdm(range(N_SIMS), desc="SBC Progress"):
    try:
        # Draw true parameters from prior
        true_params = draw_from_prior()

        # Generate synthetic data
        y_sim = generate_data(true_params, year_values)

        # Run MCMC chains
        chains = []
        accept_rates = []

        for chain in range(N_CHAINS):
            samples, accept_rate = metropolis_hastings_chain(
                y_sim, year_values, N_WARMUP, N_SAMPLES, chain_id=chain
            )
            chains.append(samples)
            accept_rates.append(accept_rate)

        # Combine chains
        all_samples = np.vstack(chains)

        # Compute diagnostics
        rhats = compute_rhat(chains)
        ess = compute_ess(chains)

        max_rhat = rhats.max()
        min_ess = ess.min()
        mean_acceptance = np.mean(accept_rates)

        converged = (max_rhat < 1.1) and (min_ess > 100)

        convergence_stats['converged'].append(converged)
        convergence_stats['max_rhat'].append(max_rhat)
        convergence_stats['min_ess'].append(min_ess)
        convergence_stats['mean_acceptance'].append(mean_acceptance)

        # Extract posterior samples
        beta_0_samples = all_samples[:, 0]
        beta_1_samples = all_samples[:, 1]
        beta_2_samples = all_samples[:, 2]
        phi_samples = np.exp(all_samples[:, 3])

        # Compute ranks
        sbc_results['beta_0']['ranks'].append(compute_rank(true_params['beta_0'], beta_0_samples))
        sbc_results['beta_0']['true'].append(true_params['beta_0'])
        sbc_results['beta_0']['mean'].append(beta_0_samples.mean())
        sbc_results['beta_0']['median'].append(np.median(beta_0_samples))
        sbc_results['beta_0']['q025'].append(np.percentile(beta_0_samples, 2.5))
        sbc_results['beta_0']['q975'].append(np.percentile(beta_0_samples, 97.5))

        sbc_results['beta_1']['ranks'].append(compute_rank(true_params['beta_1'], beta_1_samples))
        sbc_results['beta_1']['true'].append(true_params['beta_1'])
        sbc_results['beta_1']['mean'].append(beta_1_samples.mean())
        sbc_results['beta_1']['median'].append(np.median(beta_1_samples))
        sbc_results['beta_1']['q025'].append(np.percentile(beta_1_samples, 2.5))
        sbc_results['beta_1']['q975'].append(np.percentile(beta_1_samples, 97.5))

        sbc_results['beta_2']['ranks'].append(compute_rank(true_params['beta_2'], beta_2_samples))
        sbc_results['beta_2']['true'].append(true_params['beta_2'])
        sbc_results['beta_2']['mean'].append(beta_2_samples.mean())
        sbc_results['beta_2']['median'].append(np.median(beta_2_samples))
        sbc_results['beta_2']['q025'].append(np.percentile(beta_2_samples, 2.5))
        sbc_results['beta_2']['q975'].append(np.percentile(beta_2_samples, 97.5))

        sbc_results['phi']['ranks'].append(compute_rank(true_params['phi'], phi_samples))
        sbc_results['phi']['true'].append(true_params['phi'])
        sbc_results['phi']['mean'].append(phi_samples.mean())
        sbc_results['phi']['median'].append(np.median(phi_samples))
        sbc_results['phi']['q025'].append(np.percentile(phi_samples, 2.5))
        sbc_results['phi']['q975'].append(np.percentile(phi_samples, 97.5))

    except Exception as e:
        failed_simulations.append((sim, str(e)))
        print(f"\nSimulation {sim} failed: {str(e)}")
        continue

print("\n" + "="*80)
print("SBC SIMULATIONS COMPLETED")
print("="*80)

# Compute success rate
n_successful = len(sbc_results['beta_0']['ranks'])
success_rate = n_successful / N_SIMS * 100
print(f"\nSuccess Rate: {n_successful}/{N_SIMS} ({success_rate:.1f}%)")

if n_successful < 0.5 * N_SIMS:
    print("\nWARNING: Less than 50% of simulations succeeded!")

# Compute convergence statistics
n_converged = sum(convergence_stats['converged'])
convergence_rate = n_converged / n_successful * 100 if n_successful > 0 else 0

print(f"\nConvergence Statistics:")
print(f"  Converged: {n_converged}/{n_successful} ({convergence_rate:.1f}%)")
print(f"  Mean R̂: {np.mean(convergence_stats['max_rhat']):.4f}")
print(f"  Mean ESS: {np.mean(convergence_stats['min_ess']):.0f}")
print(f"  Mean acceptance: {np.mean(convergence_stats['mean_acceptance']):.3f}")

# Save results
print("\nSaving results...")

for param in ['beta_0', 'beta_1', 'beta_2', 'phi']:
    df = pd.DataFrame({
        'simulation': range(n_successful),
        'true_value': sbc_results[param]['true'],
        'posterior_mean': sbc_results[param]['mean'],
        'posterior_median': sbc_results[param]['median'],
        'q025': sbc_results[param]['q025'],
        'q975': sbc_results[param]['q975'],
        'rank': sbc_results[param]['ranks']
    })
    df.to_csv(RESULTS_DIR / f'sbc_results_{param}.csv', index=False)

conv_df = pd.DataFrame(convergence_stats)
conv_df.to_csv(RESULTS_DIR / 'convergence_stats.csv', index=False)

summary_stats = {
    'n_simulations': N_SIMS,
    'n_successful': n_successful,
    'success_rate': success_rate,
    'convergence_rate': convergence_rate,
    'mean_rhat': float(np.mean(convergence_stats['max_rhat'])),
    'mean_ess': float(np.mean(convergence_stats['min_ess'])),
    'mean_acceptance': float(np.mean(convergence_stats['mean_acceptance']))
}

with open(RESULTS_DIR / 'summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"\nResults saved to: {RESULTS_DIR}")
print("\nSBC analysis complete!")
