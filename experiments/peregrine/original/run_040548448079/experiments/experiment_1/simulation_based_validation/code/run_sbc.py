"""
Simulation-Based Calibration for Fixed Changepoint Negative Binomial Model

This script runs SBC to validate that the model can recover known parameters
when the truth is known. Critical safety check before fitting real data.
"""

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import multiprocessing as mp
from pathlib import Path
import json
import warnings
from scipy import stats

# Suppress cmdstanpy progress bars for cleaner output
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIMULATIONS = 100  # Start with 100, can increase to 200 if needed
N_CHAINS = 4
N_ITER = 1000
N_WARMUP = 500
ADAPT_DELTA = 0.90
MAX_TREEDEPTH = 12

# Paths
WORKSPACE = Path('/workspace')
EXP_DIR = WORKSPACE / 'experiments' / 'experiment_1'
SBC_DIR = EXP_DIR / 'simulation_based_validation'
CODE_DIR = SBC_DIR / 'code'
OUTPUT_DIR = SBC_DIR / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load real data to get year values
data = pd.read_csv(WORKSPACE / 'data' / 'data.csv')
YEAR_VALUES = data['year'].values
N = len(YEAR_VALUES)
TAU = 17

# Prior specifications (REVISED)
PRIORS = {
    'beta_0': {'dist': 'normal', 'params': (4.3, 0.5)},
    'beta_1': {'dist': 'normal', 'params': (0.35, 0.3)},
    'beta_2': {'dist': 'normal', 'params': (0.85, 0.5)},
    'alpha': {'dist': 'gamma', 'params': (2, 3)},
    'rho': {'dist': 'beta', 'params': (12, 1)},
    'sigma_eps': {'dist': 'exponential', 'params': (2,)}
}


def draw_from_prior(priors):
    """Draw parameter values from prior distributions."""
    params = {}

    for param, spec in priors.items():
        if spec['dist'] == 'normal':
            loc, scale = spec['params']
            params[param] = np.random.normal(loc, scale)
        elif spec['dist'] == 'gamma':
            shape, rate = spec['params']
            params[param] = np.random.gamma(shape, 1/rate)
        elif spec['dist'] == 'beta':
            a, b = spec['params']
            params[param] = np.random.beta(a, b)
        elif spec['dist'] == 'exponential':
            rate = spec['params'][0]
            params[param] = np.random.exponential(1/rate)

    return params


def simulate_data(params, year, tau):
    """
    Generate synthetic data from the model.

    Uses AR(1) process for errors with stationary initialization.
    """
    N = len(year)
    year_tau = year[tau - 1]  # Convert to 0-indexed

    beta_0 = params['beta_0']
    beta_1 = params['beta_1']
    beta_2 = params['beta_2']
    alpha = params['alpha']
    rho = params['rho']
    sigma_eps = params['sigma_eps']

    # Generate AR(1) errors with stationary initialization
    eps = np.zeros(N)
    eps[0] = np.random.normal(0, sigma_eps / np.sqrt(1 - rho**2))

    for t in range(1, N):
        eps[t] = rho * eps[t-1] + np.random.normal(0, sigma_eps)

    # Compute log mean with changepoint
    log_mu = np.zeros(N)
    for t in range(N):
        if t < tau:
            log_mu[t] = beta_0 + beta_1 * year[t] + eps[t]
        else:
            log_mu[t] = beta_0 + beta_1 * year[t] + beta_2 * (year[t] - year_tau) + eps[t]

    mu = np.exp(log_mu)

    # Generate counts from negative binomial
    # Stan's neg_binomial_2(mu, phi) has phi = 1/alpha
    phi = 1 / alpha

    # Convert to numpy's parameterization: n=phi, p=phi/(phi+mu)
    C = np.random.negative_binomial(phi, phi / (phi + mu))

    return C


def run_single_simulation(sim_id):
    """
    Run a single SBC iteration:
    1. Draw parameters from prior
    2. Simulate data
    3. Fit model
    4. Return ranks and diagnostics
    """
    try:
        # Draw true parameters from prior
        true_params = draw_from_prior(PRIORS)

        # Simulate data
        C = simulate_data(true_params, YEAR_VALUES, TAU)

        # Prepare data for Stan
        stan_data = {
            'N': N,
            'C': C.astype(int).tolist(),
            'year': YEAR_VALUES.tolist(),
            'tau': TAU
        }

        # Compile model (cached after first compilation)
        model = CmdStanModel(stan_file=str(CODE_DIR / 'model.stan'))

        # Fit model
        fit = model.sample(
            data=stan_data,
            chains=N_CHAINS,
            iter_sampling=N_ITER - N_WARMUP,
            iter_warmup=N_WARMUP,
            adapt_delta=ADAPT_DELTA,
            max_treedepth=MAX_TREEDEPTH,
            show_progress=False,
            show_console=False
        )

        # Extract posterior samples
        posterior = fit.stan_variables()

        # Compute ranks for each parameter
        ranks = {}
        for param in ['beta_0', 'beta_1', 'beta_2', 'alpha', 'rho', 'sigma_eps']:
            samples = posterior[param].flatten()
            # Rank of true value among posterior samples
            rank = np.sum(samples < true_params[param])
            ranks[param] = rank

        # Compute diagnostics
        summary = fit.summary()

        # Check for convergence issues
        max_rhat = summary['R_hat'].max()
        min_ess_bulk = summary['N_Eff'].min() if 'N_Eff' in summary.columns else summary['ess_bulk'].min()

        # Check for divergences
        divergences = fit.num_unconstrained_divergences()
        max_treedepth_hits = fit.num_max_treedepth()

        diagnostics = {
            'sim_id': sim_id,
            'success': True,
            'max_rhat': max_rhat,
            'min_ess_bulk': min_ess_bulk,
            'divergences': divergences,
            'max_treedepth_hits': max_treedepth_hits,
            'converged': max_rhat < 1.05 and min_ess_bulk > 100
        }

        result = {
            'sim_id': sim_id,
            'true_params': true_params,
            'ranks': ranks,
            'diagnostics': diagnostics,
            'posterior_means': {param: float(posterior[param].mean())
                               for param in ['beta_0', 'beta_1', 'beta_2', 'alpha', 'rho', 'sigma_eps']}
        }

        print(f"Simulation {sim_id:3d} completed: Rhat={max_rhat:.3f}, ESS={min_ess_bulk:.0f}, "
              f"Div={divergences}, Status={'OK' if diagnostics['converged'] else 'FAILED'}")

        return result

    except Exception as e:
        print(f"Simulation {sim_id:3d} FAILED with error: {str(e)}")
        return {
            'sim_id': sim_id,
            'success': False,
            'error': str(e)
        }


def run_sbc_parallel(n_simulations):
    """Run SBC simulations in parallel."""
    print(f"Starting SBC with {n_simulations} simulations...")
    print(f"Using {N_CHAINS} chains, {N_ITER} iterations per chain")
    print(f"Adapt delta: {ADAPT_DELTA}, Max treedepth: {MAX_TREEDEPTH}")
    print("-" * 80)

    # Run simulations sequentially (parallel can cause issues with cmdstanpy)
    results = []
    for sim_id in range(n_simulations):
        result = run_single_simulation(sim_id)
        results.append(result)

    return results


def save_results(results):
    """Save SBC results to disk."""
    # Filter successful simulations
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print("\n" + "=" * 80)
    print(f"SBC COMPLETED: {len(successful)}/{len(results)} simulations successful")
    print(f"Failure rate: {len(failed)/len(results)*100:.1f}%")
    print("=" * 80)

    # Save all results
    with open(OUTPUT_DIR / 'sbc_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary dataframes
    if successful:
        # Ranks dataframe
        ranks_data = []
        for r in successful:
            row = {'sim_id': r['sim_id']}
            row.update(r['ranks'])
            ranks_data.append(row)

        ranks_df = pd.DataFrame(ranks_data)
        ranks_df.to_csv(OUTPUT_DIR / 'ranks.csv', index=False)

        # True parameters and recovered means
        recovery_data = []
        for r in successful:
            row = {'sim_id': r['sim_id']}
            for param in ['beta_0', 'beta_1', 'beta_2', 'alpha', 'rho', 'sigma_eps']:
                row[f'{param}_true'] = r['true_params'][param]
                row[f'{param}_mean'] = r['posterior_means'][param]
            recovery_data.append(row)

        recovery_df = pd.DataFrame(recovery_data)
        recovery_df.to_csv(OUTPUT_DIR / 'recovery.csv', index=False)

        # Diagnostics dataframe
        diag_data = [r['diagnostics'] for r in successful]
        diag_df = pd.DataFrame(diag_data)
        diag_df.to_csv(OUTPUT_DIR / 'diagnostics.csv', index=False)

        print(f"\nResults saved to {OUTPUT_DIR}")
        print(f"  - sbc_results.json: Full results")
        print(f"  - ranks.csv: Rank statistics")
        print(f"  - recovery.csv: True vs recovered parameters")
        print(f"  - diagnostics.csv: Convergence diagnostics")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("CONVERGENCE SUMMARY")
        print("=" * 80)
        print(f"Max Rhat: {diag_df['max_rhat'].max():.4f}")
        print(f"Min ESS bulk: {diag_df['min_ess_bulk'].min():.0f}")
        print(f"Total divergences: {diag_df['divergences'].sum()}")
        print(f"Total max treedepth hits: {diag_df['max_treedepth_hits'].sum()}")
        print(f"Converged simulations: {diag_df['converged'].sum()}/{len(diag_df)}")

    return successful, failed


if __name__ == '__main__':
    # Run SBC
    results = run_sbc_parallel(N_SIMULATIONS)

    # Save results
    successful, failed = save_results(results)

    print("\n" + "=" * 80)
    print("SBC COMPLETE - Proceed to analysis with analyze_sbc.py")
    print("=" * 80)
