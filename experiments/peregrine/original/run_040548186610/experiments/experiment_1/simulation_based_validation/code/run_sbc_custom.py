#!/usr/bin/env python3
"""
Simulation-Based Calibration for Negative Binomial Quadratic Model
Using custom MCMC implementation (Metropolis-Hastings)
Experiment 1: Testing parameter recovery and model calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats, optimize
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
PLOTS_DIR.mkdir(exist_ok=True)
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

N_SIMS = 100  # Number of SBC simulations
N_OBS = len(year_values)

# MCMC settings
N_WARMUP = 2000
N_SAMPLES = 2000
N_CHAINS = 4

print("="*80)
print("SIMULATION-BASED CALIBRATION (SBC)")
print("Negative Binomial Quadratic Model - Experiment 1")
print("="*80)
print(f"\nConfiguration:")
print(f"  Number of simulations: {N_SIMS}")
print(f"  Number of observations per simulation: {N_OBS}")
print(f"  Year range: [{year_values.min():.2f}, {year_values.max():.2f}]")
print(f"\nMCMC Settings:")
print(f"  Chains: {N_CHAINS}")
print(f"  Warmup: {N_WARMUP}")
print(f"  Samples per chain: {N_SAMPLES}")
print(f"\nAdjusted Priors:")
print(f"  β₀ ~ Normal({PRIORS['beta_0']['mean']}, {PRIORS['beta_0']['sd']})")
print(f"  β₁ ~ Normal({PRIORS['beta_1']['mean']}, {PRIORS['beta_1']['sd']})")
print(f"  β₂ ~ Normal({PRIORS['beta_2']['mean']}, {PRIORS['beta_2']['sd']})")
print(f"  φ ~ Gamma({PRIORS['phi']['shape']}, {PRIORS['phi']['rate']})")
print()

def log_prior(params):
    """Log prior density"""
    beta_0, beta_1, beta_2, log_phi = params
    phi = np.exp(log_phi)

    lp = 0
    lp += stats.norm.logpdf(beta_0, PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
    lp += stats.norm.logpdf(beta_1, PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
    lp += stats.norm.logpdf(beta_2, PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd'])
    lp += stats.gamma.logpdf(phi, PRIORS['phi']['shape'], scale=1/PRIORS['phi']['rate'])
    lp += log_phi  # Jacobian for log transformation

    return lp

def log_likelihood(params, y, year):
    """Log likelihood for negative binomial model"""
    beta_0, beta_1, beta_2, log_phi = params
    phi = np.exp(log_phi)

    if phi <= 0:
        return -np.inf

    # Compute mu
    log_mu = beta_0 + beta_1 * year + beta_2 * year**2
    mu = np.exp(log_mu)

    # Check for numerical issues
    if np.any(mu <= 0) or np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
        return -np.inf

    # Negative binomial log likelihood
    # Using the NB2 parameterization: p = phi/(mu + phi), n = phi
    ll = 0
    for i in range(len(y)):
        if mu[i] > 0 and phi > 0:
            # stats.nbinom uses different parameterization: n, p where p = n/(n+mu)
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
    """
    Single chain Metropolis-Hastings MCMC
    """
    if initial_params is None:
        # Initialize from prior
        beta_0 = np.random.normal(PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
        beta_1 = np.random.normal(PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
        beta_2 = np.random.normal(PRIORS['beta_2']['mean'], PRIORS['beta_2']['sd'])
        phi = np.random.gamma(PRIORS['phi']['shape'], 1/PRIORS['phi']['rate'])
        current = np.array([beta_0, beta_1, beta_2, np.log(phi)])
    else:
        current = initial_params.copy()

    if proposal_sd is None:
        # Adaptive proposal based on prior scales
        proposal_sd = np.array([
            PRIORS['beta_0']['sd'] * 0.5,
            PRIORS['beta_1']['sd'] * 0.5,
            PRIORS['beta_2']['sd'] * 0.5,
            0.2  # for log_phi
        ])

    samples = []
    n_total = n_warmup + n_samples
    n_accepted = 0

    current_log_post = log_posterior(current, y, year)

    for i in range(n_total):
        # Propose new parameters
        proposal = current + np.random.normal(0, proposal_sd, size=4)

        # Compute acceptance ratio
        proposal_log_post = log_posterior(proposal, y, year)
        log_accept_ratio = proposal_log_post - current_log_post

        # Accept or reject
        if np.log(np.random.rand()) < log_accept_ratio:
            current = proposal
            current_log_post = proposal_log_post
            if i >= n_warmup:
                n_accepted += 1

        # Store sample after warmup
        if i >= n_warmup:
            samples.append(current.copy())

        # Adaptive tuning during warmup
        if i < n_warmup and i > 0 and i % 500 == 0:
            acceptance_rate = n_accepted / (i - max(0, i - 500)) if i > 500 else n_accepted / i
            if acceptance_rate < 0.2:
                proposal_sd *= 0.9
            elif acceptance_rate > 0.5:
                proposal_sd *= 1.1

    acceptance_rate = n_accepted / n_samples
    samples = np.array(samples)

    return samples, acceptance_rate

def run_mcmc(y, year, n_chains=4, n_warmup=2000, n_samples=2000):
    """
    Run multiple MCMC chains in parallel (sequentially for simplicity)
    """
    all_chains = []
    acceptance_rates = []

    for chain in range(n_chains):
        samples, accept_rate = metropolis_hastings_chain(
            y, year, n_warmup, n_samples, chain_id=chain
        )
        all_chains.append(samples)
        acceptance_rates.append(accept_rate)

    # Combine chains
    all_samples = np.vstack(all_chains)

    return all_samples, all_chains, acceptance_rates

def compute_rhat(chains):
    """
    Compute R-hat convergence diagnostic
    chains: list of arrays, each array is (n_samples, n_params)
    """
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    n_params = chains[0].shape[1]

    rhats = []
    for p in range(n_params):
        chain_means = [chains[c][:, p].mean() for c in range(n_chains)]
        chain_vars = [chains[c][:, p].var(ddof=1) for c in range(n_chains)]

        # Between-chain variance
        B = n_samples * np.var(chain_means, ddof=1)

        # Within-chain variance
        W = np.mean(chain_vars)

        # Pooled variance
        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

        # R-hat
        rhat = np.sqrt(var_plus / W) if W > 0 else np.inf
        rhats.append(rhat)

    return np.array(rhats)

def compute_ess(chains):
    """
    Compute effective sample size (simple approximation)
    """
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    n_params = chains[0].shape[1]

    ess_values = []
    for p in range(n_params):
        # Combine all chains
        combined = np.concatenate([chains[c][:, p] for c in range(n_chains)])

        # Simple ESS based on autocorrelation
        # This is a rough approximation
        acf = np.correlate(combined - combined.mean(), combined - combined.mean(), mode='full')
        acf = acf[len(acf)//2:] / acf.max()

        # Find where ACF drops below threshold
        threshold = 0.05
        tau = 1
        for i in range(1, min(len(acf), 100)):
            if acf[i] < threshold:
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

    return {
        'beta_0': beta_0,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'phi': phi
    }

def generate_data(params, year):
    """Generate synthetic data from negative binomial model"""
    N = len(year)
    log_mu = params['beta_0'] + params['beta_1'] * year + params['beta_2'] * year**2
    mu = np.exp(log_mu)

    # Generate negative binomial data
    # Using scipy's parameterization: n=phi, p=phi/(mu+phi)
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

print("Running SBC simulations...")
print("-"*80)

for sim in tqdm(range(N_SIMS), desc="SBC Progress"):
    try:
        # Step 1: Draw true parameters from prior
        true_params = draw_from_prior()

        # Step 2: Generate synthetic data
        y_sim = generate_data(true_params, year_values)

        # Step 3: Fit model using MCMC
        all_samples, chains, accept_rates = run_mcmc(
            y_sim, year_values,
            n_chains=N_CHAINS,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES
        )

        # Step 4: Compute diagnostics
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

        # Step 5: Extract posterior samples and compute ranks
        # Convert log_phi to phi
        beta_0_samples = all_samples[:, 0]
        beta_1_samples = all_samples[:, 1]
        beta_2_samples = all_samples[:, 2]
        phi_samples = np.exp(all_samples[:, 3])

        # Compute ranks
        rank_beta_0 = compute_rank(true_params['beta_0'], beta_0_samples)
        rank_beta_1 = compute_rank(true_params['beta_1'], beta_1_samples)
        rank_beta_2 = compute_rank(true_params['beta_2'], beta_2_samples)
        rank_phi = compute_rank(true_params['phi'], phi_samples)

        # Store results
        sbc_results['beta_0']['ranks'].append(rank_beta_0)
        sbc_results['beta_0']['true'].append(true_params['beta_0'])
        sbc_results['beta_0']['mean'].append(beta_0_samples.mean())
        sbc_results['beta_0']['median'].append(np.median(beta_0_samples))
        sbc_results['beta_0']['q025'].append(np.percentile(beta_0_samples, 2.5))
        sbc_results['beta_0']['q975'].append(np.percentile(beta_0_samples, 97.5))

        sbc_results['beta_1']['ranks'].append(rank_beta_1)
        sbc_results['beta_1']['true'].append(true_params['beta_1'])
        sbc_results['beta_1']['mean'].append(beta_1_samples.mean())
        sbc_results['beta_1']['median'].append(np.median(beta_1_samples))
        sbc_results['beta_1']['q025'].append(np.percentile(beta_1_samples, 2.5))
        sbc_results['beta_1']['q975'].append(np.percentile(beta_1_samples, 97.5))

        sbc_results['beta_2']['ranks'].append(rank_beta_2)
        sbc_results['beta_2']['true'].append(true_params['beta_2'])
        sbc_results['beta_2']['mean'].append(beta_2_samples.mean())
        sbc_results['beta_2']['median'].append(np.median(beta_2_samples))
        sbc_results['beta_2']['q025'].append(np.percentile(beta_2_samples, 2.5))
        sbc_results['beta_2']['q975'].append(np.percentile(beta_2_samples, 97.5))

        sbc_results['phi']['ranks'].append(rank_phi)
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
print(f"Failed simulations: {len(failed_simulations)}")

if n_successful < 0.5 * N_SIMS:
    print("\nWARNING: Less than 50% of simulations succeeded!")
    print("This is a CRITICAL FAILURE - model may have serious computational issues")

# Compute convergence statistics
n_converged = sum(convergence_stats['converged'])
convergence_rate = n_converged / n_successful * 100 if n_successful > 0 else 0

print(f"\nConvergence Statistics:")
print(f"  Converged: {n_converged}/{n_successful} ({convergence_rate:.1f}%)")
print(f"  Mean R̂: {np.mean(convergence_stats['max_rhat']):.4f}")
print(f"  Mean ESS: {np.mean(convergence_stats['min_ess']):.0f}")
print(f"  Mean acceptance rate: {np.mean(convergence_stats['mean_acceptance']):.3f}")

# Save detailed results
print("\nSaving results...")

# Save SBC results to CSV
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

# Save convergence statistics
conv_df = pd.DataFrame(convergence_stats)
conv_df.to_csv(RESULTS_DIR / 'convergence_stats.csv', index=False)

# Save summary statistics
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
print("\nSBC analysis complete! Proceed to visualization and diagnostics.")
