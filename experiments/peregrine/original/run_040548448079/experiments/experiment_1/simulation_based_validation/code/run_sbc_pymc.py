"""
Simulation-Based Calibration for Fixed Changepoint Negative Binomial Model
Using PyMC (fallback from Stan due to compilation issues)

This script runs SBC to validate that the model can recover known parameters
when the truth is known. Critical safety check before fitting real data.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pathlib import Path
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIMULATIONS = 100  # Start with 100
N_CHAINS = 4
N_DRAWS = 500
N_TUNE = 500
TARGET_ACCEPT = 0.90

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
    # PyMC uses alpha parameterization: variance = mu + alpha * mu^2
    # So we use alpha directly
    p = 1 / (1 + alpha * mu)
    n = 1 / alpha

    C = np.random.negative_binomial(n, p)

    return C


def build_pymc_model(C, year, tau):
    """
    Build PyMC model for inference.

    Uses non-centered parameterization for AR(1) errors.
    Manually constructs AR(1) process using recursion.
    """
    year_tau = year[tau - 1]
    N = len(year)

    with pm.Model() as model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=4.3, sigma=0.5)
        beta_1 = pm.Normal('beta_1', mu=0.35, sigma=0.3)
        beta_2 = pm.Normal('beta_2', mu=0.85, sigma=0.5)
        alpha = pm.Gamma('alpha', alpha=2, beta=3)
        rho = pm.Beta('rho', alpha=12, beta=1)
        sigma_eps = pm.Exponential('sigma_eps', lam=2)

        # Non-centered AR(1) errors
        z = pm.Normal('z', mu=0, sigma=1, shape=N)

        # Manually build AR(1) process using PyTensor
        # This avoids the deprecated scan function
        def ar1_recursive(z_vec, rho_val, sigma_val):
            """Recursively build AR(1) errors."""
            eps = pt.zeros(N)
            # Stationary initialization
            eps = pt.set_subtensor(eps[0], (sigma_val / pt.sqrt(1 - rho_val**2)) * z_vec[0])

            # Build recursively
            for t in range(1, N):
                eps = pt.set_subtensor(eps[t], rho_val * eps[t-1] + sigma_val * z_vec[t])

            return eps

        eps = pm.Deterministic('eps', ar1_recursive(z, rho, sigma_eps))

        # Mean structure with changepoint
        # Create indicator for post-changepoint
        post_change = pt.arange(N) >= tau

        log_mu = pm.Deterministic('log_mu',
            beta_0 + beta_1 * year +
            post_change * beta_2 * (year - year_tau) + eps
        )

        mu = pm.Deterministic('mu', pt.exp(log_mu))

        # Likelihood: NegativeBinomial with alpha parameterization
        # PyMC: variance = mu + alpha * mu^2
        obs = pm.NegativeBinomial('obs', mu=mu, alpha=1/alpha, observed=C)

    return model


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

        # Build and fit model
        model = build_pymc_model(C, YEAR_VALUES, TAU)

        with model:
            trace = pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                target_accept=TARGET_ACCEPT,
                return_inferencedata=True,
                progressbar=False,
                random_seed=sim_id
            )

        # Extract posterior samples
        posterior = trace.posterior

        # Compute ranks for each parameter
        ranks = {}
        for param in ['beta_0', 'beta_1', 'beta_2', 'alpha', 'rho', 'sigma_eps']:
            samples = posterior[param].values.flatten()
            # Rank of true value among posterior samples
            rank = np.sum(samples < true_params[param])
            ranks[param] = int(rank)

        # Compute diagnostics
        summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'alpha', 'rho', 'sigma_eps'])

        # Check for convergence issues
        max_rhat = summary['r_hat'].max()
        min_ess_bulk = summary['ess_bulk'].min()

        # Check for divergences
        divergences = trace.sample_stats.diverging.sum().item()

        diagnostics = {
            'sim_id': sim_id,
            'success': True,
            'max_rhat': float(max_rhat),
            'min_ess_bulk': float(min_ess_bulk),
            'divergences': int(divergences),
            'converged': max_rhat < 1.05 and min_ess_bulk > 100
        }

        # Posterior means
        posterior_means = {}
        for param in ['beta_0', 'beta_1', 'beta_2', 'alpha', 'rho', 'sigma_eps']:
            posterior_means[param] = float(posterior[param].mean().item())

        result = {
            'sim_id': sim_id,
            'true_params': true_params,
            'ranks': ranks,
            'diagnostics': diagnostics,
            'posterior_means': posterior_means
        }

        print(f"Simulation {sim_id:3d} completed: Rhat={max_rhat:.3f}, ESS={min_ess_bulk:.0f}, "
              f"Div={divergences}, Status={'OK' if diagnostics['converged'] else 'FAILED'}")

        return result

    except Exception as e:
        import traceback
        print(f"Simulation {sim_id:3d} FAILED with error: {str(e)}")
        if sim_id == 0:  # Print full traceback for first error
            traceback.print_exc()
        return {
            'sim_id': sim_id,
            'success': False,
            'error': str(e)
        }


def run_sbc_sequential(n_simulations):
    """Run SBC simulations sequentially."""
    print(f"Starting SBC with {n_simulations} simulations using PyMC...")
    print(f"Using {N_CHAINS} chains, {N_DRAWS} draws per chain")
    print(f"Target accept: {TARGET_ACCEPT}")
    print("-" * 80)

    # Run simulations sequentially
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
        print(f"Converged simulations: {diag_df['converged'].sum()}/{len(diag_df)}")

    return successful, failed


if __name__ == '__main__':
    # Run SBC
    results = run_sbc_sequential(N_SIMULATIONS)

    # Save results
    successful, failed = save_results(results)

    print("\n" + "=" * 80)
    print("SBC COMPLETE - Proceed to analysis with analyze_sbc.py")
    print("=" * 80)
