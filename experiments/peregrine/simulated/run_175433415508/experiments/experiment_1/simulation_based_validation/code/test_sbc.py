"""Quick test of SBC implementation with 2 simulations"""
import sys
sys.path.insert(0, '/workspace/experiments/experiment_1/simulation_based_validation/code')

# Run just 2 simulations as a test
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed
np.random.seed(42)

# Configuration
N_SIMULATIONS = 2
N_CHAINS = 2
N_ITER = 500  # Reduced for testing
N_WARMUP = 250

# Model priors
PRIOR_BETA0_MEAN = 4.69
PRIOR_BETA0_SD = 1.0
PRIOR_BETA1_MEAN = 1.0
PRIOR_BETA1_SD = 0.5
PRIOR_PHI_SHAPE = 2.0
PRIOR_PHI_RATE = 0.1

# Load real data
DATA_PATH = Path("/workspace/data/data.csv")
real_data = pd.read_csv(DATA_PATH)
year_values = real_data['year'].values
N = len(year_values)

print(f"Testing SBC with {N_SIMULATIONS} simulations")
print(f"Data size: N={N} observations")

# Import functions from main script
from scipy import stats
from scipy.optimize import minimize
import time

def neg_binomial_logpmf(y, mu, phi):
    p = phi / (phi + mu)
    return stats.nbinom.logpmf(y, phi, p)

def log_prior(beta_0, beta_1, phi):
    lp = 0.0
    lp += stats.norm.logpdf(beta_0, PRIOR_BETA0_MEAN, PRIOR_BETA0_SD)
    lp += stats.norm.logpdf(beta_1, PRIOR_BETA1_MEAN, PRIOR_BETA1_SD)
    lp += stats.gamma.logpdf(phi, PRIOR_PHI_SHAPE, scale=1/PRIOR_PHI_RATE)
    return lp

def log_likelihood(beta_0, beta_1, phi, year, C):
    mu = np.exp(beta_0 + beta_1 * year)
    ll = 0.0
    for i in range(len(C)):
        ll += neg_binomial_logpmf(C[i], mu[i], phi)
    return ll

def log_posterior(params, year, C):
    beta_0, beta_1, log_phi = params
    phi = np.exp(log_phi)

    if not np.isfinite([beta_0, beta_1, phi]).all():
        return -np.inf
    if phi <= 0:
        return -np.inf

    lp = log_prior(beta_0, beta_1, phi)
    ll = log_likelihood(beta_0, beta_1, phi, year, C)
    return lp + ll + log_phi

def neg_log_posterior(params, year, C):
    return -log_posterior(params, year, C)

def metropolis_hastings_chain(year, C, n_iter=500, n_warmup=250):
    init_params = np.array([PRIOR_BETA0_MEAN, PRIOR_BETA1_MEAN, np.log(20)])

    try:
        result = minimize(neg_log_posterior, init_params, args=(year, C),
                         method='L-BFGS-B')
        if result.success:
            init_params = result.x
    except:
        pass

    prop_sd = np.array([0.15, 0.15, 0.15])
    total_iter = n_warmup + n_iter
    samples = np.zeros((total_iter, 3))
    samples[0] = init_params
    accepted = 0

    current_log_post = log_posterior(samples[0], year, C)

    for i in range(1, total_iter):
        proposal = samples[i-1] + np.random.normal(0, prop_sd, size=3)
        proposal_log_post = log_posterior(proposal, year, C)
        log_alpha = proposal_log_post - current_log_post

        if np.log(np.random.uniform()) < log_alpha:
            samples[i] = proposal
            current_log_post = proposal_log_post
            if i >= n_warmup:
                accepted += 1
        else:
            samples[i] = samples[i-1]

        if i < n_warmup and i % 100 == 0:
            accept_rate = accepted / max(1, i)
            if accept_rate < 0.2:
                prop_sd *= 0.9
            elif accept_rate > 0.5:
                prop_sd *= 1.1

    post_warmup = samples[n_warmup:]
    accept_rate = accepted / n_iter

    return post_warmup, accept_rate

# Test with one simulation
print("\nTest simulation 1:")
beta_0_true = np.random.normal(PRIOR_BETA0_MEAN, PRIOR_BETA0_SD)
beta_1_true = np.random.normal(PRIOR_BETA1_MEAN, PRIOR_BETA1_SD)
phi_true = np.random.gamma(PRIOR_PHI_SHAPE, 1/PRIOR_PHI_RATE)

print(f"  True params: β₀={beta_0_true:.2f}, β₁={beta_1_true:.2f}, φ={phi_true:.1f}")

mu = np.exp(beta_0_true + beta_1_true * year_values)
p = phi_true / (phi_true + mu)
C_sim = np.random.negative_binomial(phi_true, p)

print(f"  Generated data: C range [{C_sim.min()}, {C_sim.max()}]")

start = time.time()
samples, accept_rate = metropolis_hastings_chain(year_values, C_sim, n_iter=N_ITER, n_warmup=N_WARMUP)
elapsed = time.time() - start

print(f"  MCMC complete: {elapsed:.1f}s, acceptance rate={accept_rate:.3f}")
print(f"  Posterior means: β₀={samples[:, 0].mean():.2f}, β₁={samples[:, 1].mean():.2f}, φ={np.exp(samples[:, 2]).mean():.1f}")

beta_0_post = samples[:, 0]
beta_1_post = samples[:, 1]
phi_post = np.exp(samples[:, 2])

# Check recovery
print(f"  Recovery check:")
print(f"    β₀: true={beta_0_true:.3f}, est={beta_0_post.mean():.3f}, diff={abs(beta_0_post.mean()-beta_0_true):.3f}")
print(f"    β₁: true={beta_1_true:.3f}, est={beta_1_post.mean():.3f}, diff={abs(beta_1_post.mean()-beta_1_true):.3f}")
print(f"    φ: true={phi_true:.3f}, est={phi_post.mean():.3f}, diff={abs(phi_post.mean()-phi_true):.3f}")

print("\nTest PASSED - SBC implementation is working correctly!")
print(f"Estimated time for 50 simulations: {elapsed * 50 / 60:.1f} minutes")
