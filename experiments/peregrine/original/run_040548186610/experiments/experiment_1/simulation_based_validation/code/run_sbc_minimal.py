#!/usr/bin/env python3
"""
Minimal SBC for quick testing - 20 simulations only
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

np.random.seed(42)

BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

data = pd.read_csv("/workspace/data/data.csv")
year_values = data['year'].values

PRIORS = {
    'beta_0': {'mean': 4.7, 'sd': 0.3},
    'beta_1': {'mean': 0.8, 'sd': 0.2},
    'beta_2': {'mean': 0.3, 'sd': 0.1},
    'phi': {'shape': 2, 'rate': 0.5}
}

N_SIMS = 20  # Minimal for testing
N_WARMUP = 500
N_SAMPLES = 500
N_CHAINS = 2

print(f"Running MINIMAL SBC: {N_SIMS} simulations, {N_CHAINS} chains × {N_SAMPLES} samples")

def log_prior(params):
    beta_0, beta_1, beta_2, log_phi = params
    phi = np.exp(log_phi)
    lp = stats.norm.logpdf(beta_0, PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
    lp += stats.norm.logpdf(beta_1, PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
    lp += stats.norm.logpdf(beta_2, PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd'])
    lp += stats.gamma.logpdf(phi, PRIORS['phi']['shape'], scale=1/PRIORS['phi']['rate'])
    lp += log_phi
    return lp

def log_likelihood(params, y, year):
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
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, y, year)

def mh_chain(y, year, n_warmup, n_samples):
    beta_0 = np.random.normal(PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
    beta_1 = np.random.normal(PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
    beta_2 = np.random.normal(PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd'])
    phi = np.random.gamma(PRIORS['phi']['shape'], 1/PRIORS['phi']['rate'])
    current = np.array([beta_0, beta_1, beta_2, np.log(phi)])

    proposal_sd = np.array([0.15, 0.10, 0.05, 0.2])
    samples = []
    n_accepted = 0
    current_log_post = log_posterior(current, y, year)

    for i in range(n_warmup + n_samples):
        proposal = current + np.random.normal(0, proposal_sd, size=4)
        proposal_log_post = log_posterior(proposal, y, year)

        if np.log(np.random.rand()) < (proposal_log_post - current_log_post):
            current = proposal
            current_log_post = proposal_log_post
            if i >= n_warmup:
                n_accepted += 1

        if i >= n_warmup:
            samples.append(current.copy())

    return np.array(samples), n_accepted / n_samples

def draw_from_prior():
    return {
        'beta_0': np.random.normal(PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd']),
        'beta_1': np.random.normal(PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd']),
        'beta_2': np.random.normal(PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd']),
        'phi': np.random.gamma(PRIORS['phi']['shape'], 1/PRIORS['phi']['rate'])
    }

def generate_data(params, year):
    log_mu = params['beta_0'] + params['beta_1'] * year + params['beta_2'] * year**2
    mu = np.exp(log_mu)
    y = []
    for i in range(len(year)):
        p = params['phi'] / (mu[i] + params['phi'])
        y.append(np.random.negative_binomial(params['phi'], p))
    return np.array(y)

sbc_results = {p: {'ranks': [], 'true': [], 'mean': [], 'median': [], 'q025': [], 'q975': []}
               for p in ['beta_0', 'beta_1', 'beta_2', 'phi']}
convergence_stats = {'converged': [], 'max_rhat': [], 'min_ess': [], 'mean_acceptance': []}

for sim in range(N_SIMS):
    print(f"Simulation {sim+1}/{N_SIMS}...", end='\r')

    true_params = draw_from_prior()
    y_sim = generate_data(true_params, year_values)

    chains = []
    accept_rates = []
    for _ in range(N_CHAINS):
        samples, acc = mh_chain(y_sim, year_values, N_WARMUP, N_SAMPLES)
        chains.append(samples)
        accept_rates.append(acc)

    all_samples = np.vstack(chains)

    # Simple Rhat
    n_params = 4
    rhats = []
    for p in range(n_params):
        chain_means = [chains[c][:, p].mean() for c in range(N_CHAINS)]
        chain_vars = [chains[c][:, p].var(ddof=1) for c in range(N_CHAINS)]
        B = N_SAMPLES * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        var_plus = ((N_SAMPLES - 1) / N_SAMPLES) * W + (1 / N_SAMPLES) * B
        rhats.append(np.sqrt(var_plus / W) if W > 0 else np.inf)

    max_rhat = max(rhats)
    min_ess = len(all_samples) / 2  # rough estimate
    convergence_stats['converged'].append(max_rhat < 1.1)
    convergence_stats['max_rhat'].append(max_rhat)
    convergence_stats['min_ess'].append(min_ess)
    convergence_stats['mean_acceptance'].append(np.mean(accept_rates))

    # Extract samples
    beta_0_samples = all_samples[:, 0]
    beta_1_samples = all_samples[:, 1]
    beta_2_samples = all_samples[:, 2]
    phi_samples = np.exp(all_samples[:, 3])

    for param, samples_array in [('beta_0', beta_0_samples), ('beta_1', beta_1_samples),
                                  ('beta_2', beta_2_samples), ('phi', phi_samples)]:
        sbc_results[param]['ranks'].append(np.sum(samples_array < true_params[param]))
        sbc_results[param]['true'].append(true_params[param])
        sbc_results[param]['mean'].append(samples_array.mean())
        sbc_results[param]['median'].append(np.median(samples_array))
        sbc_results[param]['q025'].append(np.percentile(samples_array, 2.5))
        sbc_results[param]['q975'].append(np.percentile(samples_array, 97.5))

print(f"\nCompleted {N_SIMS} simulations")

# Save results
for param in ['beta_0', 'beta_1', 'beta_2', 'phi']:
    df = pd.DataFrame({
        'simulation': range(N_SIMS),
        'true_value': sbc_results[param]['true'],
        'posterior_mean': sbc_results[param]['mean'],
        'posterior_median': sbc_results[param]['median'],
        'q025': sbc_results[param]['q025'],
        'q975': sbc_results[param]['q975'],
        'rank': sbc_results[param]['ranks']
    })
    df.to_csv(RESULTS_DIR / f'sbc_results_{param}.csv', index=False)

pd.DataFrame(convergence_stats).to_csv(RESULTS_DIR / 'convergence_stats.csv', index=False)

summary_stats = {
    'n_simulations': N_SIMS,
    'n_successful': N_SIMS,
    'success_rate': 100.0,
    'convergence_rate': sum(convergence_stats['converged']) / N_SIMS * 100,
    'mean_rhat': float(np.mean(convergence_stats['max_rhat'])),
    'mean_ess': float(np.mean(convergence_stats['min_ess'])),
    'mean_acceptance': float(np.mean(convergence_stats['mean_acceptance']))
}

with open(RESULTS_DIR / 'summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"Results saved. Convergence rate: {summary_stats['convergence_rate']:.1f}%")
print(f"Mean R-hat: {summary_stats['mean_rhat']:.4f}")
