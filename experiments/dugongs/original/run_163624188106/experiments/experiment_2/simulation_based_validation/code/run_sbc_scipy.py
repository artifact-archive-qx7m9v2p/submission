"""
Simulation-Based Calibration for Log-Linear Heteroscedastic Model
Using scipy optimization + approximate posterior from Hessian
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
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

# Prior specifications
PRIORS = {
    'beta_0': {'mean': 1.8, 'sd': 0.5},
    'beta_1': {'mean': 0.3, 'sd': 0.2},
    'gamma_0': {'mean': -2.0, 'sd': 1.0},
    'gamma_1': {'mean': -0.05, 'sd': 0.05}
}

# SBC settings
N_SIMS = 100
N_POSTERIOR_SAMPLES = 4000  # For rank statistics

def neg_log_posterior(params, x, y):
    """Negative log posterior for optimization"""
    beta_0, beta_1, gamma_0, gamma_1 = params

    # Prior log-probs
    log_prior = (
        stats.norm.logpdf(beta_0, PRIORS['beta_0']['mean'], PRIORS['beta_0']['sd']) +
        stats.norm.logpdf(beta_1, PRIORS['beta_1']['mean'], PRIORS['beta_1']['sd']) +
        stats.norm.logpdf(gamma_0, PRIORS['gamma_0']['mean'], PRIORS['gamma_0']['sd']) +
        stats.norm.logpdf(gamma_1, PRIORS['gamma_1']['mean'], PRIORS['gamma_1']['sd'])
    )

    # Likelihood
    mu = beta_0 + beta_1 * np.log(x)
    log_sigma = gamma_0 + gamma_1 * x
    sigma = np.exp(log_sigma)

    # Check for numerical issues
    if np.any(sigma <= 0) or np.any(~np.isfinite(sigma)):
        return 1e10

    log_lik = np.sum(stats.norm.logpdf(y, mu, sigma))

    if not np.isfinite(log_lik) or not np.isfinite(log_prior):
        return 1e10

    return -(log_lik + log_prior)

def fit_model_laplace(x, y):
    """Fit model using Laplace approximation"""
    # Initial guess from priors
    init = np.array([
        PRIORS['beta_0']['mean'],
        PRIORS['beta_1']['mean'],
        PRIORS['gamma_0']['mean'],
        PRIORS['gamma_1']['mean']
    ])

    # Try multiple starting points if initial fails
    starting_points = [
        init,
        init + np.random.normal(0, 0.1, 4),
        np.array([2.0, 0.25, -2.5, -0.05])  # Alternative reasonable values
    ]

    best_result = None
    best_nll = np.inf

    for start in starting_points:
        try:
            result = minimize(
                neg_log_posterior,
                start,
                args=(x, y),
                method='BFGS',
                options={'maxiter': 1000}
            )

            if result.success and result.fun < best_nll:
                best_result = result
                best_nll = result.fun

        except Exception:
            continue

    if best_result is None or not best_result.success:
        return None

    # Get MAP estimate
    map_estimate = best_result.x

    # Compute Hessian for covariance
    from scipy.optimize import approx_fprime

    def grad_func(params):
        return approx_fprime(params, neg_log_posterior, 1e-8, x, y)

    # Approximate Hessian using finite differences
    eps = 1e-5
    n_params = len(map_estimate)
    hessian = np.zeros((n_params, n_params))

    grad_at_map = grad_func(map_estimate)

    for i in range(n_params):
        params_plus = map_estimate.copy()
        params_plus[i] += eps
        grad_plus = grad_func(params_plus)
        hessian[:, i] = (grad_plus - grad_at_map) / eps

    # Symmetrize
    hessian = (hessian + hessian.T) / 2

    # Covariance is inverse Hessian
    try:
        cov_matrix = np.linalg.inv(hessian)

        # Check if covariance is positive definite
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if np.any(eigvals <= 0):
            # Add small diagonal to stabilize
            cov_matrix += np.eye(n_params) * 1e-6

    except np.linalg.LinAlgError:
        # Hessian is singular, use diagonal approximation
        hess_diag = np.diag(hessian)
        hess_diag[hess_diag <= 0] = 1.0  # Protect against non-positive
        cov_matrix = np.diag(1.0 / hess_diag)

    return {
        'map': map_estimate,
        'cov': cov_matrix,
        'success': True
    }

# Storage for results
sbc_results = []
failed_fits = []

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

    # Fit model
    try:
        fit_result = fit_model_laplace(x_values, y_sim)

        if fit_result is None:
            failed_fits.append({'sim': sim_idx, 'reason': 'optimization_failed'})
            continue

        map_est = fit_result['map']
        cov = fit_result['cov']

        # Sample from approximate posterior (multivariate normal)
        posterior_samples = np.random.multivariate_normal(map_est, cov, size=N_POSTERIOR_SAMPLES)

        # Store results
        result = {
            'sim': sim_idx,
            'converged': True,
            # True values
            'true_beta_0': true_beta_0,
            'true_beta_1': true_beta_1,
            'true_gamma_0': true_gamma_0,
            'true_gamma_1': true_gamma_1,
            # Posterior statistics
            'beta_0_mean': posterior_samples[:, 0].mean(),
            'beta_0_sd': posterior_samples[:, 0].std(),
            'beta_0_q05': np.percentile(posterior_samples[:, 0], 5),
            'beta_0_q95': np.percentile(posterior_samples[:, 0], 95),
            'beta_1_mean': posterior_samples[:, 1].mean(),
            'beta_1_sd': posterior_samples[:, 1].std(),
            'beta_1_q05': np.percentile(posterior_samples[:, 1], 5),
            'beta_1_q95': np.percentile(posterior_samples[:, 1], 95),
            'gamma_0_mean': posterior_samples[:, 2].mean(),
            'gamma_0_sd': posterior_samples[:, 2].std(),
            'gamma_0_q05': np.percentile(posterior_samples[:, 2], 5),
            'gamma_0_q95': np.percentile(posterior_samples[:, 2], 95),
            'gamma_1_mean': posterior_samples[:, 3].mean(),
            'gamma_1_sd': posterior_samples[:, 3].std(),
            'gamma_1_q05': np.percentile(posterior_samples[:, 3], 5),
            'gamma_1_q95': np.percentile(posterior_samples[:, 3], 95),
        }

        # Compute ranks for SBC (rank of true value in posterior samples)
        result['rank_beta_0'] = np.sum(posterior_samples[:, 0] < true_beta_0)
        result['rank_beta_1'] = np.sum(posterior_samples[:, 1] < true_beta_1)
        result['rank_gamma_0'] = np.sum(posterior_samples[:, 2] < true_gamma_0)
        result['rank_gamma_1'] = np.sum(posterior_samples[:, 3] < true_gamma_1)

        sbc_results.append(result)

    except Exception as e:
        failed_fits.append({'sim': sim_idx, 'reason': str(e)})

# Convert to DataFrame
df_results = pd.DataFrame(sbc_results)

# Save results
df_results.to_csv(RESULTS_DIR / 'sbc_results.csv', index=False)

if failed_fits:
    pd.DataFrame(failed_fits).to_csv(RESULTS_DIR / 'failed_fits.csv', index=False)

print(f"\n{len(df_results)} successful simulations completed")
print(f"{len(failed_fits)} simulations failed")
print(f"\nResults saved to {RESULTS_DIR}")

# Print summary statistics
print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)

print(f"\nSuccess rate: {len(df_results)}/{N_SIMS} ({100*len(df_results)/N_SIMS:.1f}%)")

print("\nParameter Recovery:")

for param in ['beta_0', 'beta_1', 'gamma_0', 'gamma_1']:
    true_col = f'true_{param}'
    mean_col = f'{param}_mean'

    bias = (df_results[mean_col] - df_results[true_col]).mean()
    rel_bias = 100 * bias / df_results[true_col].mean()
    rmse = np.sqrt(((df_results[mean_col] - df_results[true_col])**2).mean())

    # Coverage
    q05_col = f'{param}_q05'
    q95_col = f'{param}_q95'
    coverage = ((df_results[true_col] >= df_results[q05_col]) &
                (df_results[true_col] <= df_results[q95_col])).mean()

    print(f"\n{param}:")
    print(f"  Bias: {bias:.6f} ({rel_bias:.2f}%)")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  90% Coverage: {100*coverage:.1f}%")

print("\n" + "="*60)
