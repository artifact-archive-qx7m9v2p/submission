#!/usr/bin/env python
"""Test single simulation to verify setup"""
import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import json
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("pymc").setLevel(logging.WARNING)

print("Loading data...")
with open("/workspace/data/data.csv", "r") as f:
    data = json.load(f)

year = np.array(data["year"])
n = len(year)

print(f"Data loaded: n={n}")

# Draw parameters from prior
np.random.seed(42)
beta_0_true = np.random.normal(4.3, 1.0)
beta_1_true = np.random.normal(0.85, 0.5)
phi_true = np.random.exponential(1/0.667)

print(f"\nTrue parameters:")
print(f"  β₀ = {beta_0_true:.3f}")
print(f"  β₁ = {beta_1_true:.3f}")
print(f"  φ = {phi_true:.3f}")

# Simulate data
log_mu = beta_0_true + beta_1_true * year
mu = np.exp(log_mu)
p = phi_true / (phi_true + mu)
C_sim = np.random.negative_binomial(n=phi_true, p=p, size=n).astype(int)

print(f"\nSimulated data stats:")
print(f"  Mean: {C_sim.mean():.1f}")
print(f"  Variance: {C_sim.var():.1f}")
print(f"  Variance/Mean: {C_sim.var()/C_sim.mean():.2f}")

# Fit model
print("\nFitting model (this may take 2-3 minutes)...")
with pm.Model() as model:
    beta_0 = pm.Normal('beta_0', mu=4.3, sigma=1.0)
    beta_1 = pm.Normal('beta_1', mu=0.85, sigma=0.5)
    phi = pm.Exponential('phi', lam=0.667)

    log_mu = beta_0 + beta_1 * year
    mu = pm.math.exp(log_mu)

    C = pm.NegativeBinomial('C', mu=mu, alpha=phi, observed=C_sim)

    trace = pm.sample(
        draws=500,
        tune=500,
        chains=2,
        target_accept=0.9,
        return_inferencedata=True,
        progressbar=True,
        cores=1
    )

print("\nModel fitted successfully!")

# Extract posteriors
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'phi'])
print("\nPosterior summary:")
print(summary)

# Check recovery
posterior = trace.posterior
beta_0_post = posterior['beta_0'].values.flatten()
beta_1_post = posterior['beta_1'].values.flatten()
phi_post = posterior['phi'].values.flatten()

beta_0_mean = beta_0_post.mean()
beta_0_q05 = np.percentile(beta_0_post, 5)
beta_0_q95 = np.percentile(beta_0_post, 95)

beta_1_mean = beta_1_post.mean()
beta_1_q05 = np.percentile(beta_1_post, 5)
beta_1_q95 = np.percentile(beta_1_post, 95)

phi_mean = phi_post.mean()
phi_q05 = np.percentile(phi_post, 5)
phi_q95 = np.percentile(phi_post, 95)

print("\nParameter recovery:")
print(f"β₀: true={beta_0_true:.3f}, mean={beta_0_mean:.3f}, 90%CI=[{beta_0_q05:.3f}, {beta_0_q95:.3f}] - {'✓' if beta_0_q05 <= beta_0_true <= beta_0_q95 else 'X'}")
print(f"β₁: true={beta_1_true:.3f}, mean={beta_1_mean:.3f}, 90%CI=[{beta_1_q05:.3f}, {beta_1_q95:.3f}] - {'✓' if beta_1_q05 <= beta_1_true <= beta_1_q95 else 'X'}")
print(f"φ:  true={phi_true:.3f}, mean={phi_mean:.3f}, 90%CI=[{phi_q05:.3f}, {phi_q95:.3f}] - {'✓' if phi_q05 <= phi_true <= phi_q95 else 'X'}")

print("\nSingle simulation test COMPLETE!")
