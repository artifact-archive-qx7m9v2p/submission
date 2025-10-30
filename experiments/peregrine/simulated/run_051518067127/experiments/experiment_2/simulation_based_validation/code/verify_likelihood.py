"""
Verify that data generation and likelihood are consistent

This will:
1. Generate data with true parameters
2. Evaluate likelihood at true parameters
3. Compare to MLE likelihood
4. Identify any bugs
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

np.random.seed(42)

def generate_data_v1(true_params, year, regime_idx):
    """
    Original data generation (from simulation_validation_simple.py)
    """
    alpha = true_params['alpha']
    beta_1 = true_params['beta_1']
    beta_2 = true_params['beta_2']
    phi = true_params['phi']
    sigma = true_params['sigma_regime']

    n_obs = len(year)
    mu_trend = alpha + beta_1 * year + beta_2 * year**2

    log_C = np.zeros(n_obs)
    epsilon = np.zeros(n_obs)

    # First observation
    sigma_init = sigma[regime_idx[0]] / np.sqrt(1 - phi**2)
    epsilon[0] = np.random.normal(0, sigma_init)
    log_C[0] = mu_trend[0] + epsilon[0]
    epsilon[0] = log_C[0] - mu_trend[0]  # Update epsilon

    # Subsequent observations
    for t in range(1, n_obs):
        mu_t = mu_trend[t] + phi * epsilon[t-1]
        log_C[t] = np.random.normal(mu_t, sigma[regime_idx[t]])
        epsilon[t] = log_C[t] - mu_trend[t]

    return log_C, epsilon


def neg_log_likelihood_v1(params, year, log_C, regime_idx):
    """
    Original likelihood (from simulation_validation_simple.py)
    """
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


def main():
    """Test likelihood consistency"""

    print("="*80)
    print("LIKELIHOOD CONSISTENCY CHECK")
    print("="*80)
    print()

    # Setup
    n_obs = 40
    year = np.linspace(-1.668, 1.668, n_obs)
    regime_idx = np.concatenate([
        np.zeros(14, dtype=int),
        np.ones(13, dtype=int),
        np.full(13, 2, dtype=int)
    ])

    true_params = {
        'alpha': 4.3,
        'beta_1': 0.86,
        'beta_2': 0.05,
        'phi': 0.85,
        'sigma_regime': np.array([0.3, 0.4, 0.35])
    }

    # Generate data
    print("Generating data with true parameters...")
    log_C, epsilon = generate_data_v1(true_params, year, regime_idx)
    print(f"  Generated {len(log_C)} observations")
    print(f"  log_C range: [{log_C.min():.2f}, {log_C.max():.2f}]")
    print()

    # Evaluate likelihood at true parameters
    true_param_vec = np.array([
        true_params['alpha'],
        true_params['beta_1'],
        true_params['beta_2'],
        true_params['phi'],
        true_params['sigma_regime'][0],
        true_params['sigma_regime'][1],
        true_params['sigma_regime'][2]
    ])

    nll_true = neg_log_likelihood_v1(true_param_vec, year, log_C, regime_idx)
    print(f"NLL at TRUE parameters: {nll_true:.4f}")
    print()

    # Find MLE
    print("Finding MLE...")
    result = minimize(
        neg_log_likelihood_v1,
        true_param_vec + np.random.normal(0, 0.05, size=7),
        args=(year, log_C, regime_idx),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'disp': False}
    )

    print(f"MLE found: NLL = {result.fun:.4f}")
    print(f"  alpha: {result.x[0]:.3f} (true: {true_params['alpha']:.3f})")
    print(f"  beta_1: {result.x[1]:.3f} (true: {true_params['beta_1']:.3f})")
    print(f"  phi: {result.x[3]:.3f} (true: {true_params['phi']:.3f})")
    print()

    # KEY DIAGNOSTIC
    print("="*80)
    print("DIAGNOSTIC")
    print("="*80)

    delta_nll = nll_true - result.fun

    print(f"\nΔ NLL = NLL(true) - NLL(MLE) = {delta_nll:.4f}")
    print()

    if delta_nll < 0.5:
        print("✓ PASS: True parameters have similar likelihood to MLE")
        print("  → Data generation and likelihood are CONSISTENT")
        print("  → Any recovery error is due to parameter identifiability or sample size")
    else:
        print("✗ FAIL: True parameters have WORSE likelihood than MLE")
        print("  → Data generation and likelihood are INCONSISTENT")
        print("  → There is a BUG in either data generation or likelihood computation")
        print()
        print("Possible issues:")
        print("  1. AR structure implemented differently in generation vs likelihood")
        print("  2. Epsilon initialization mismatch")
        print("  3. Regime indices applied incorrectly")

    print("="*80)

    # Additional diagnostic: manually trace through first few observations
    print("\nDETAILED TRACE (first 3 observations):")
    print("="*80)

    alpha = true_params['alpha']
    beta_1 = true_params['beta_1']
    beta_2 = true_params['beta_2']
    phi = true_params['phi']
    sigma = true_params['sigma_regime']

    mu_trend = alpha + beta_1 * year + beta_2 * year**2

    print(f"\nt=0:")
    print(f"  mu_trend[0] = {mu_trend[0]:.4f}")
    print(f"  log_C[0] = {log_C[0]:.4f}")
    print(f"  epsilon[0] = log_C[0] - mu_trend[0] = {log_C[0] - mu_trend[0]:.4f}")
    print(f"  sigma_init = sigma[0] / sqrt(1-phi^2) = {sigma[0] / np.sqrt(1-phi**2):.4f}")
    print(f"  Likelihood contribution: N(epsilon[0] | 0, {sigma[0] / np.sqrt(1-phi**2):.4f})")

    for t in range(1, min(3, n_obs)):
        eps_prev = log_C[t-1] - mu_trend[t-1]
        mu_t = mu_trend[t] + phi * eps_prev
        sigma_t = sigma[regime_idx[t]]

        print(f"\nt={t}:")
        print(f"  mu_trend[{t}] = {mu_trend[t]:.4f}")
        print(f"  epsilon[{t-1}] = {eps_prev:.4f}")
        print(f"  mu[{t}] = mu_trend[{t}] + phi * epsilon[{t-1}] = {mu_t:.4f}")
        print(f"  log_C[{t}] = {log_C[t]:.4f}")
        print(f"  regime[{t}] = {regime_idx[t]}, sigma = {sigma_t:.4f}")
        print(f"  Likelihood contribution: N(log_C[{t}] | {mu_t:.4f}, {sigma_t:.4f})")

    print("="*80)


if __name__ == '__main__':
    main()
