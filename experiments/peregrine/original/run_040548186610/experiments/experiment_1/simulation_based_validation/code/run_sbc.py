#!/usr/bin/env python3
"""
Simulation-Based Calibration for Negative Binomial Quadratic Model
Experiment 1: Testing parameter recovery and model calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from cmdstanpy import CmdStanModel
from scipy import stats
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

N_SIMS = 100  # Number of SBC simulations (will increase if needed)
N_OBS = len(year_values)

print("="*80)
print("SIMULATION-BASED CALIBRATION (SBC)")
print("Negative Binomial Quadratic Model - Experiment 1")
print("="*80)
print(f"\nConfiguration:")
print(f"  Number of simulations: {N_SIMS}")
print(f"  Number of observations per simulation: {N_OBS}")
print(f"  Year range: [{year_values.min():.2f}, {year_values.max():.2f}]")
print(f"\nAdjusted Priors:")
print(f"  β₀ ~ Normal({PRIORS['beta_0']['mean']}, {PRIORS['beta_0']['sd']})")
print(f"  β₁ ~ Normal({PRIORS['beta_1']['mean']}, {PRIORS['beta_1']['sd']})")
print(f"  β₂ ~ Normal({PRIORS['beta_2']['mean']}, {PRIORS['beta_2']['sd']})")
print(f"  φ ~ Gamma({PRIORS['phi']['shape']}, {PRIORS['phi']['rate']})")
print()

# Compile Stan model
print("Compiling Stan model...")
model_path = CODE_DIR / "negbinom_quadratic.stan"
model = CmdStanModel(stan_file=str(model_path))
print("Model compiled successfully!\n")

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
    # Using the parameterization where variance = mu + mu^2/phi
    y = np.random.negative_binomial(params['phi'], params['phi']/(mu + params['phi']))

    return y

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
    'min_ess_bulk': [],
    'min_ess_tail': [],
    'n_divergent': [],
    'n_max_treedepth': []
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

        # Step 3: Fit model to synthetic data
        stan_data = {
            'N': N_OBS,
            'year': year_values.tolist(),
            'y': y_sim.tolist()
        }

        # Fit with reduced verbosity
        fit = model.sample(
            data=stan_data,
            chains=4,
            parallel_chains=4,
            iter_warmup=1000,
            iter_sampling=1000,
            adapt_delta=0.95,
            max_treedepth=12,
            show_console=False,
            refresh=0
        )

        # Step 4: Extract diagnostics
        summary = fit.summary()

        # Check convergence
        max_rhat = summary['R_hat'].max()
        min_ess_bulk = summary['N_Eff'].min()
        min_ess_tail = summary['N_Eff'].min()  # Approximation

        converged = (max_rhat < 1.01) and (min_ess_bulk > 400)

        # Get divergences and treedepth info
        try:
            divergent = fit.method_variables()['divergent__'].sum()
            max_td = fit.method_variables()['treedepth__'].max()
            n_max_treedepth = np.sum(fit.method_variables()['treedepth__'] >= max_td - 1)
        except:
            divergent = 0
            n_max_treedepth = 0

        convergence_stats['converged'].append(converged)
        convergence_stats['max_rhat'].append(max_rhat)
        convergence_stats['min_ess_bulk'].append(min_ess_bulk)
        convergence_stats['min_ess_tail'].append(min_ess_tail)
        convergence_stats['n_divergent'].append(divergent)
        convergence_stats['n_max_treedepth'].append(n_max_treedepth)

        # Step 5: Compute ranks and store results
        for param in ['beta_0', 'beta_1', 'beta_2', 'phi']:
            # Get posterior samples (all chains combined)
            posterior_samples = fit.stan_variable(param)

            # Compute rank
            rank = compute_rank(true_params[param], posterior_samples)

            # Store results
            sbc_results[param]['ranks'].append(rank)
            sbc_results[param]['true'].append(true_params[param])
            sbc_results[param]['mean'].append(posterior_samples.mean())
            sbc_results[param]['median'].append(np.median(posterior_samples))
            sbc_results[param]['q025'].append(np.percentile(posterior_samples, 2.5))
            sbc_results[param]['q975'].append(np.percentile(posterior_samples, 97.5))

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
print(f"  Mean ESS (bulk): {np.mean(convergence_stats['min_ess_bulk']):.0f}")
print(f"  Total divergent transitions: {sum(convergence_stats['n_divergent'])}")
print(f"  Simulations with divergences: {sum(np.array(convergence_stats['n_divergent']) > 0)}")

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
    'mean_ess_bulk': float(np.mean(convergence_stats['min_ess_bulk'])),
    'total_divergences': int(sum(convergence_stats['n_divergent'])),
    'sims_with_divergences': int(sum(np.array(convergence_stats['n_divergent']) > 0))
}

with open(RESULTS_DIR / 'summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"\nResults saved to: {RESULTS_DIR}")
print("\nSBC analysis complete! Proceed to visualization and diagnostics.")
