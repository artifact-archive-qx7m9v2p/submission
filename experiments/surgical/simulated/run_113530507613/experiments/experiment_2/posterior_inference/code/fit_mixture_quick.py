"""
Fit 3-component finite mixture model to data using PyMC - Quick version.

Model:
- pi ~ Dirichlet([1, 1, 1])
- z[j] ~ Categorical(pi)
- mu_raw[k] ~ Normal(-2.6, 1.0)
- mu = sort(mu_raw)  # Ordered constraint
- sigma[k] ~ Half-Normal(0, 0.5)
- theta[j] ~ Normal(mu[z[j]], sigma[z[j]])
- r[j] ~ Binomial(n[j], inv_logit(theta[j]))
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pickle
from pathlib import Path

# Setup paths
project_root = Path("/workspace")
data_path = project_root / "data" / "data.csv"
output_dir = project_root / "experiments" / "experiment_2" / "posterior_inference"
code_dir = output_dir / "code"
diag_dir = output_dir / "diagnostics"
plots_dir = output_dir / "plots"

# Ensure directories exist
for d in [code_dir, diag_dir, plots_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv(data_path)

# Prepare data for PyMC
n_groups = len(data)
n_trials = data['n_trials'].values
r_successes = data['r_successes'].values

print(f"\nData summary:")
print(f"  Number of groups: {n_groups}")
print(f"  Overall success rate: {r_successes.sum() / n_trials.sum():.4f}")

# Build mixture model
print("\nBuilding 3-component mixture model...")

with pm.Model() as mixture_model:
    # Mixing proportions
    pi = pm.Dirichlet('pi', a=np.ones(3))

    # Cluster means (unordered first, then sort)
    mu_raw = pm.Normal('mu_raw', mu=-2.6, sigma=1.0, shape=3)
    mu = pm.Deterministic('mu', pt.sort(mu_raw))

    # Cluster SDs
    sigma = pm.HalfNormal('sigma', sigma=0.5, shape=3)

    # Cluster assignments (marginalized out for better mixing)
    components = []
    for k in range(3):
        components.append(pm.Normal.dist(mu=mu[k], sigma=sigma[k]))

    # Group-level parameters (mixture of normals)
    theta = pm.Mixture('theta', w=pi, comp_dists=components, shape=n_groups)

    # Likelihood
    p = pm.Deterministic('p', pm.math.invlogit(theta))
    r = pm.Binomial('r', n=n_trials, p=p, observed=r_successes)

    # Log likelihood for LOO-CV (CRITICAL)
    log_lik = pm.Deterministic('log_lik',
                               pm.logp(pm.Binomial.dist(n=n_trials, p=p), r_successes))

    # For cluster assignments, compute posterior probabilities
    log_likes_k = []
    for k in range(3):
        theta_k = pm.Normal.dist(mu=mu[k], sigma=sigma[k])
        log_p_theta_k = pm.logp(theta_k, theta)
        log_likes_k.append(log_p_theta_k)

    log_likes_matrix = pt.stack(log_likes_k, axis=1)
    log_pi = pt.log(pi)
    log_post_unnorm = log_likes_matrix + log_pi

    cluster_probs = pm.Deterministic('cluster_probs',
                                     pm.math.softmax(log_post_unnorm, axis=1))

print("Model built successfully!")

# Sample from posterior - REDUCED ITERATIONS FOR SPEED
print("\n" + "="*70)
print("FITTING MODEL WITH HMC")
print("="*70)
print("\nQuick sampling strategy (then extend if needed):")
print("  - 4 chains")
print("  - 500 warmup, 500 sampling (QUICK PROBE)")
print("  - target_accept = 0.90")

with mixture_model:
    trace = pm.sample(
        draws=500,
        tune=500,
        chains=4,
        cores=4,
        target_accept=0.90,
        init='adapt_diag',
        return_inferencedata=True,
        random_seed=42,
        progressbar=True
    )

print("\nSampling complete!")

# Check basic convergence
print("\n" + "="*70)
print("CONVERGENCE DIAGNOSTICS")
print("="*70)

summary = az.summary(trace, var_names=['pi', 'mu', 'sigma'])
print("\nParameter Summary:")
print(summary)

# Check for issues
rhat_max = summary['r_hat'].max()
ess_min = summary['ess_bulk'].min()

print(f"\nConvergence metrics:")
print(f"  Max R-hat: {rhat_max:.4f} (should be < 1.01)")
print(f"  Min ESS_bulk: {ess_min:.1f} (should be > 400 - may need more samples)")

# Check divergences
divergences = trace.sample_stats.diverging.sum().item()
total_draws = trace.posterior.dims['draw'] * trace.posterior.dims['chain']
div_pct = 100 * divergences / total_draws

print(f"  Divergences: {divergences} / {total_draws} ({div_pct:.2f}%)")

# Check mu ordering is maintained
mu_samples = trace.posterior['mu'].values
mu_flat = mu_samples.reshape(-1, 3)

ordering_violations = np.sum((mu_flat[:, 1] <= mu_flat[:, 0]) |
                              (mu_flat[:, 2] <= mu_flat[:, 1]))
print(f"  Ordering violations: {ordering_violations} / {len(mu_flat)} ({100*ordering_violations/len(mu_flat):.2f}%)")

# Save trace
print("\nSaving results...")

trace.to_netcdf(diag_dir / "posterior_inference.netcdf")
print(f"  Saved InferenceData: {diag_dir / 'posterior_inference.netcdf'}")

summary_full = az.summary(trace)
summary_full.to_csv(diag_dir / "parameter_summary.csv")
print(f"  Saved summary: {diag_dir / 'parameter_summary.csv'}")

with open(diag_dir / "idata.pkl", 'wb') as f:
    pickle.dump(trace, f)
print(f"  Saved trace: {diag_dir / 'idata.pkl'}")

# Verify log_likelihood is present
if 'log_lik' in trace.log_likelihood:
    print("\n✓ log_likelihood successfully included in InferenceData")
    print(f"  Shape: {trace.log_likelihood['log_lik'].shape}")
else:
    print("\n✗ WARNING: log_likelihood not found in InferenceData!")

print("\n" + "="*70)
print("QUICK FITTING COMPLETE")
print("="*70)
print("\nNote: This was a quick probe with 500 samples per chain.")
print("If convergence looks good, extend to 2000+ for final inference.")
