"""
Fit 3-component finite mixture model to data using PyMC.

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
warnings.filterwarnings('ignore', message='.*g\\+\\+ not detected.*')

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
print(f"Data shape: {data.shape}")
print(data)

# Prepare data for PyMC
n_groups = len(data)
n_trials = data['n_trials'].values
r_successes = data['r_successes'].values

print(f"\nData summary:")
print(f"  Number of groups: {n_groups}")
print(f"  Total trials: {n_trials.sum()}")
print(f"  Total successes: {r_successes.sum()}")
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
    # We'll use the Mixture distribution which marginalizes over z
    components = []
    for k in range(3):
        # Each component is a Normal distribution
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
    # This requires computing the likelihood under each component
    log_likes_k = []
    for k in range(3):
        theta_k = pm.Normal.dist(mu=mu[k], sigma=sigma[k])
        # Log probability of theta under component k
        log_p_theta_k = pm.logp(theta_k, theta)
        log_likes_k.append(log_p_theta_k)

    # Stack and compute posterior probabilities
    log_likes_matrix = pt.stack(log_likes_k, axis=1)  # shape: (n_groups, 3)
    log_pi = pt.log(pi)  # shape: (3,)
    log_post_unnorm = log_likes_matrix + log_pi  # Broadcasting

    # Normalize to get probabilities
    cluster_probs = pm.Deterministic('cluster_probs',
                                     pm.math.softmax(log_post_unnorm, axis=1))

print("\nModel structure:")
print(mixture_model)

# Sample from posterior
print("\n" + "="*70)
print("FITTING MODEL WITH HMC")
print("="*70)
print("\nStrategy: Start with moderate iterations for mixture model")
print("  - 4 chains")
print("  - 2000 warmup, 2000 sampling")
print("  - target_accept = 0.95 (mixture models need careful exploration)")
print("  - init = 'adapt_diag' (default)")

with mixture_model:
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.95,
        init='adapt_diag',
        return_inferencedata=True,
        random_seed=42
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
print(f"  Min ESS_bulk: {ess_min:.1f} (should be > 400)")

# Check divergences
divergences = trace.sample_stats.diverging.sum().item()
total_draws = trace.posterior.dims['draw'] * trace.posterior.dims['chain']
div_pct = 100 * divergences / total_draws

print(f"  Divergences: {divergences} / {total_draws} ({div_pct:.2f}%)")

# Check mu ordering is maintained
mu_samples = trace.posterior['mu'].values  # shape: (chains, draws, 3)
mu_flat = mu_samples.reshape(-1, 3)

ordering_violations = np.sum((mu_flat[:, 1] <= mu_flat[:, 0]) |
                              (mu_flat[:, 2] <= mu_flat[:, 1]))
print(f"  Ordering violations: {ordering_violations} / {len(mu_flat)} ({100*ordering_violations/len(mu_flat):.2f}%)")

# Save trace
print("\nSaving results...")

# Save as ArviZ InferenceData with log_likelihood
trace.to_netcdf(diag_dir / "posterior_inference.netcdf")
print(f"  Saved InferenceData: {diag_dir / 'posterior_inference.netcdf'}")

# Also save summary
summary_full = az.summary(trace)
summary_full.to_csv(diag_dir / "parameter_summary.csv")
print(f"  Saved summary: {diag_dir / 'parameter_summary.csv'}")

# Save trace object as pickle for further analysis
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
print("FITTING COMPLETE")
print("="*70)
print(f"\nResults saved to: {output_dir}")
print("\nNext steps:")
print("  1. Run diagnostic analysis script")
print("  2. Create visualizations")
print("  3. Compute cluster assignments")
