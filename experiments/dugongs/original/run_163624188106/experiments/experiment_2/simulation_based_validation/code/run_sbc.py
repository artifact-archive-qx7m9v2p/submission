"""
Simulation-Based Calibration for Log-Linear Heteroscedastic Model
"""

import numpy as np
import pandas as pd
import cmdstanpy
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path('/workspace/experiments/experiment_2/simulation_based_validation')
CODE_DIR = BASE_DIR / 'code'
RESULTS_DIR = CODE_DIR / 'sbc_results'
RESULTS_DIR.mkdir(exist_ok=True)

# Load x values from data
data = pd.read_csv('/workspace/data/data.csv')
x_values = data['x'].values
N = len(x_values)

print(f"Running SBC with N={N} observations")
print(f"x range: [{x_values.min()}, {x_values.max()}]")

# True parameters for simulation
TRUE_PARAMS = {
    'beta_0': 1.8,
    'beta_1': 0.3,
    'gamma_0': -2.5,
    'gamma_1': -0.067
}

# Prior specifications
PRIORS = {
    'beta_0': {'mean': 1.8, 'sd': 0.5},
    'beta_1': {'mean': 0.3, 'sd': 0.2},
    'gamma_0': {'mean': -2.0, 'sd': 1.0},
    'gamma_1': {'mean': -0.05, 'sd': 0.05}
}

# SBC settings
N_SIMS = 100
N_CHAINS = 4
N_ITER = 2000
N_WARMUP = 1000

# Compile model
print("\nCompiling Stan model...")
model = cmdstanpy.CmdStanModel(stan_file=str(CODE_DIR / 'model.stan'))
print("Model compiled successfully")

# Storage for results
sbc_results = []
convergence_issues = []

print(f"\nRunning {N_SIMS} simulations...")

for sim_idx in range(N_SIMS):
    if (sim_idx + 1) % 10 == 0:
        print(f"  Simulation {sim_idx + 1}/{N_SIMS}")

    # Draw true parameters from priors
    true_beta_0 = np.random.normal(PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd'])
    true_beta_1 = np.random.normal(PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd'])
    true_gamma_0 = np.random.normal(PRIORS['gamma_0']['mean'], PRIORS['gamma_0']['sd'])
    true_gamma_1 = np.random.normal(PRIORS['gamma_1']['mean'], PRIORS['gamma_1']['sd'])

    # Generate synthetic data
    mu = true_beta_0 + true_beta_1 * np.log(x_values)
    log_sigma = true_gamma_0 + true_gamma_1 * x_values
    sigma = np.exp(log_sigma)
    y_sim = np.random.normal(mu, sigma)

    # Prepare data for Stan
    stan_data = {
        'N': N,
        'x': x_values.tolist(),
        'y': y_sim.tolist()
    }

    # Fit model
    try:
        fit = model.sample(
            data=stan_data,
            chains=N_CHAINS,
            iter_sampling=N_ITER - N_WARMUP,
            iter_warmup=N_WARMUP,
            show_progress=False,
            show_console=False,
            refresh=0
        )

        # Extract posterior samples
        posterior_samples = fit.stan_variables()

        # Check convergence
        summary = fit.summary()
        max_rhat = summary['R_hat'].max()
        min_ess_bulk = summary['N_Eff'].min()

        converged = max_rhat < 1.01

        if not converged:
            convergence_issues.append({
                'sim': sim_idx,
                'max_rhat': max_rhat,
                'min_ess': min_ess_bulk
            })

        # Store results
        result = {
            'sim': sim_idx,
            'converged': converged,
            'max_rhat': max_rhat,
            'min_ess_bulk': min_ess_bulk,
            # True values
            'true_beta_0': true_beta_0,
            'true_beta_1': true_beta_1,
            'true_gamma_0': true_gamma_0,
            'true_gamma_1': true_gamma_1,
            # Posterior statistics
            'beta_0_mean': posterior_samples['beta_0'].mean(),
            'beta_0_sd': posterior_samples['beta_0'].std(),
            'beta_0_q05': np.percentile(posterior_samples['beta_0'], 5),
            'beta_0_q95': np.percentile(posterior_samples['beta_0'], 95),
            'beta_1_mean': posterior_samples['beta_1'].mean(),
            'beta_1_sd': posterior_samples['beta_1'].std(),
            'beta_1_q05': np.percentile(posterior_samples['beta_1'], 5),
            'beta_1_q95': np.percentile(posterior_samples['beta_1'], 95),
            'gamma_0_mean': posterior_samples['gamma_0'].mean(),
            'gamma_0_sd': posterior_samples['gamma_0'].std(),
            'gamma_0_q05': np.percentile(posterior_samples['gamma_0'], 5),
            'gamma_0_q95': np.percentile(posterior_samples['gamma_0'], 95),
            'gamma_1_mean': posterior_samples['gamma_1'].mean(),
            'gamma_1_sd': posterior_samples['gamma_1'].std(),
            'gamma_1_q05': np.percentile(posterior_samples['gamma_1'], 5),
            'gamma_1_q95': np.percentile(posterior_samples['gamma_1'], 95),
        }

        # Compute ranks for SBC (rank of true value in posterior samples)
        result['rank_beta_0'] = np.sum(posterior_samples['beta_0'] < true_beta_0)
        result['rank_beta_1'] = np.sum(posterior_samples['beta_1'] < true_beta_1)
        result['rank_gamma_0'] = np.sum(posterior_samples['gamma_0'] < true_gamma_0)
        result['rank_gamma_1'] = np.sum(posterior_samples['gamma_1'] < true_gamma_1)

        sbc_results.append(result)

    except Exception as e:
        print(f"  ERROR in simulation {sim_idx}: {str(e)}")
        convergence_issues.append({
            'sim': sim_idx,
            'error': str(e)
        })

# Convert to DataFrame
df_results = pd.DataFrame(sbc_results)

# Save results
df_results.to_csv(RESULTS_DIR / 'sbc_results.csv', index=False)

# Save convergence issues
if convergence_issues:
    pd.DataFrame(convergence_issues).to_csv(RESULTS_DIR / 'convergence_issues.csv', index=False)

print(f"\n{len(df_results)} successful simulations completed")
print(f"{len(convergence_issues)} simulations had issues")
print(f"\nResults saved to {RESULTS_DIR}")

# Print summary statistics
print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)

n_converged = df_results['converged'].sum()
print(f"\nConvergence: {n_converged}/{len(df_results)} ({100*n_converged/len(df_results):.1f}%) with R_hat < 1.01")

print("\nParameter Recovery (converged simulations only):")
df_conv = df_results[df_results['converged']]

for param in ['beta_0', 'beta_1', 'gamma_0', 'gamma_1']:
    true_col = f'true_{param}'
    mean_col = f'{param}_mean'

    bias = (df_conv[mean_col] - df_conv[true_col]).mean()
    rel_bias = 100 * bias / df_conv[true_col].mean()
    rmse = np.sqrt(((df_conv[mean_col] - df_conv[true_col])**2).mean())

    # Coverage
    q05_col = f'{param}_q05'
    q95_col = f'{param}_q95'
    coverage = ((df_conv[true_col] >= df_conv[q05_col]) &
                (df_conv[true_col] <= df_conv[q95_col])).mean()

    print(f"\n{param}:")
    print(f"  Bias: {bias:.6f} ({rel_bias:.2f}%)")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  90% Coverage: {100*coverage:.1f}%")

print("\n" + "="*60)
