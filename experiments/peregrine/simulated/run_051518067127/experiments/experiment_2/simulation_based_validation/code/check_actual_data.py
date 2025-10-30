"""
Check the actual synthetic data that was generated earlier
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

def neg_log_likelihood(params, year, log_C, regime_idx):
    """Likelihood function"""
    alpha, beta_1, beta_2, phi, sigma_1, sigma_2, sigma_3 = params
    sigma_regime = np.array([sigma_1, sigma_2, sigma_3])

    if np.abs(phi) >= 0.95 or np.any(sigma_regime <= 0):
        return 1e10

    n_obs = len(log_C)
    mu_trend = alpha + beta_1 * year + beta_2 * year**2

    ll = 0.0

    # First observation
    sigma_init = sigma_regime[regime_idx[0]] / np.sqrt(1 - phi**2)
    epsilon_0 = log_C[0] - mu_trend[0]
    ll += norm.logpdf(epsilon_0, loc=0, scale=sigma_init)

    # Subsequent observations
    epsilon_prev = log_C[0] - mu_trend[0]
    for t in range(1, n_obs):
        mu_t = mu_trend[t] + phi * epsilon_prev
        sigma_t = sigma_regime[regime_idx[t]]
        ll += norm.logpdf(log_C[t], loc=mu_t, scale=sigma_t)
        epsilon_prev = log_C[t] - mu_trend[t]

    return -ll


# Load the actual synthetic data that was used
data = pd.read_csv('/workspace/experiments/experiment_2/simulation_based_validation/code/synthetic_data.csv')

year = data['year'].values
log_C = np.log(data['C'].values)

regime_idx = np.concatenate([
    np.zeros(14, dtype=int),
    np.ones(13, dtype=int),
    np.full(13, 2, dtype=int)
])

# True parameters that were SUPPOSED to generate the data
true_params = np.array([4.3, 0.86, 0.05, 0.85, 0.3, 0.4, 0.35])

print("Checking actual synthetic_data.csv")
print("="*80)
print()

# NLL at true parameters
nll_true = neg_log_likelihood(true_params, year, log_C, regime_idx)
print(f"NLL at TRUE parameters: {nll_true:.4f}")

# Find MLE on this actual data
result = minimize(
    neg_log_likelihood,
    true_params + np.random.normal(0, 0.05, size=7),
    args=(year, log_C, regime_idx),
    method='Nelder-Mead',
    options={'maxiter': 2000}
)

print(f"MLE: NLL = {result.fun:.4f}")
print(f"  Params: {result.x}")
print()
print(f"Δ NLL = {nll_true - result.fun:.4f}")

if (nll_true - result.fun) > 1.0:
    print("\n✗ CONFIRMED: Bug in data generation or likelihood")
    print("\nThis means the validation has detected a MODEL IMPLEMENTATION ERROR")
    print("This is exactly what simulation-based calibration is designed to catch!")
else:
    print("\n✓ Consistent")
