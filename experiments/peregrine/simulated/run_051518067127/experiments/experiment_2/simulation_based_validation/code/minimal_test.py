"""
Minimal test case: Generate 3 observations, check likelihood exactly

This will help identify where the mismatch is
"""

import numpy as np
from scipy.stats import norm

np.random.seed(99)

# Simple parameters
alpha = 4.0
beta_1 = 0.5
phi = 0.8
sigma = 0.3

# Three time points
year = np.array([-1.0, 0.0, 1.0])
mu_trend = alpha + beta_1 * year

print("="*80)
print("MINIMAL TEST CASE")
print("="*80)
print(f"\nParameters: alpha={alpha}, beta_1={beta_1}, phi={phi}, sigma={sigma}")
print(f"mu_trend = {mu_trend}")
print()

# DATA GENERATION
print("DATA GENERATION:")
print("-"*80)

# t=0: Draw epsilon[0] from stationary distribution
sigma_init = sigma / np.sqrt(1 - phi**2)
epsilon_0_draw = np.random.normal(0, sigma_init)
log_C_0 = mu_trend[0] + epsilon_0_draw

print(f"t=0:")
print(f"  sigma_init = {sigma_init:.4f}")
print(f"  epsilon[0] ~ N(0, {sigma_init:.4f})")
print(f"  Drew epsilon[0] = {epsilon_0_draw:.4f}")
print(f"  log_C[0] = mu_trend[0] + epsilon[0] = {mu_trend[0]:.4f} + {epsilon_0_draw:.4f} = {log_C_0:.4f}")

# Compute epsilon[0] as residual (should equal draw)
epsilon_0_residual = log_C_0 - mu_trend[0]
print(f"  epsilon[0] (as residual) = {epsilon_0_residual:.4f}")
print(f"  Match? {np.isclose(epsilon_0_draw, epsilon_0_residual)}")

# t=1: Condition on epsilon[0]
mu_1 = mu_trend[1] + phi * epsilon_0_residual  # Use residual version
log_C_1 = np.random.normal(mu_1, sigma)
epsilon_1_residual = log_C_1 - mu_trend[1]

print(f"\nt=1:")
print(f"  mu[1] = mu_trend[1] + phi * epsilon[0] = {mu_trend[1]:.4f} + {phi:.2f} * {epsilon_0_residual:.4f} = {mu_1:.4f}")
print(f"  log_C[1] ~ N({mu_1:.4f}, {sigma:.4f})")
print(f"  Drew log_C[1] = {log_C_1:.4f}")
print(f"  epsilon[1] = log_C[1] - mu_trend[1] = {epsilon_1_residual:.4f}")

# t=2
mu_2 = mu_trend[2] + phi * epsilon_1_residual
log_C_2 = np.random.normal(mu_2, sigma)
epsilon_2_residual = log_C_2 - mu_trend[2]

print(f"\nt=2:")
print(f"  mu[2] = mu_trend[2] + phi * epsilon[1] = {mu_trend[2]:.4f} + {phi:.2f} * {epsilon_1_residual:.4f} = {mu_2:.4f}")
print(f"  log_C[2] ~ N({mu_2:.4f}, {sigma:.4f})")
print(f"  Drew log_C[2] = {log_C_2:.4f}")

log_C = np.array([log_C_0, log_C_1, log_C_2])

print()
print("LIKELIHOOD EVALUATION:")
print("-"*80)

# Evaluate likelihood at TRUE parameters
def eval_likelihood(alpha, beta_1, phi, sigma, year, log_C):
    """Evaluate log-likelihood"""
    mu_trend = alpha + beta_1 * year

    # t=0: epsilon[0] ~ N(0, sigma_init)
    sigma_init = sigma / np.sqrt(1 - phi**2)
    epsilon_0 = log_C[0] - mu_trend[0]
    ll_0 = norm.logpdf(epsilon_0, 0, sigma_init)

    print(f"t=0:")
    print(f"  epsilon[0] = log_C[0] - mu_trend[0] = {log_C[0]:.4f} - {mu_trend[0]:.4f} = {epsilon_0:.4f}")
    print(f"  ll contribution = logpdf({epsilon_0:.4f} | 0, {sigma_init:.4f}) = {ll_0:.4f}")

    # t=1: log_C[1] ~ N(mu_trend[1] + phi*epsilon[0], sigma)
    epsilon_0_for_ar = log_C[0] - mu_trend[0]
    mu_1 = mu_trend[1] + phi * epsilon_0_for_ar
    ll_1 = norm.logpdf(log_C[1], mu_1, sigma)

    print(f"\nt=1:")
    print(f"  mu[1] = mu_trend[1] + phi * epsilon[0] = {mu_trend[1]:.4f} + {phi:.2f} * {epsilon_0_for_ar:.4f} = {mu_1:.4f}")
    print(f"  ll contribution = logpdf({log_C[1]:.4f} | {mu_1:.4f}, {sigma:.4f}) = {ll_1:.4f}")

    # t=2
    epsilon_1_for_ar = log_C[1] - mu_trend[1]
    mu_2 = mu_trend[2] + phi * epsilon_1_for_ar
    ll_2 = norm.logpdf(log_C[2], mu_2, sigma)

    print(f"\nt=2:")
    print(f"  mu[2] = mu_trend[2] + phi * epsilon[1] = {mu_trend[2]:.4f} + {phi:.2f} * {epsilon_1_for_ar:.4f} = {mu_2:.4f}")
    print(f"  ll contribution = logpdf({log_C[2]:.4f} | {mu_2:.4f}, {sigma:.4f}) = {ll_2:.4f}")

    total_ll = ll_0 + ll_1 + ll_2
    print(f"\nTotal log-likelihood = {total_ll:.4f}")

    return total_ll

ll_true = eval_likelihood(alpha, beta_1, phi, sigma, year, log_C)

print()
print("="*80)
print(f"RESULT: Log-likelihood at TRUE parameters = {ll_true:.4f}")
print()
print("This should be reasonably high (not pathologically low)")
print("If it's very low, there's still a bug")
print("="*80)
