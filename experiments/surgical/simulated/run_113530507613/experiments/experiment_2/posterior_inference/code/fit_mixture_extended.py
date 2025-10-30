"""
Fit 3-component finite mixture model - Extended sampling for better convergence.
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

# Load data
print("Loading data...")
data = pd.read_csv(data_path)
n_groups = len(data)
n_trials = data['n_trials'].values
r_successes = data['r_successes'].values

print(f"Data: {n_groups} groups, overall rate = {r_successes.sum() / n_trials.sum():.4f}")

# Build mixture model
print("\nBuilding 3-component mixture model...")

with pm.Model() as mixture_model:
    # Mixing proportions
    pi = pm.Dirichlet('pi', a=np.ones(3))

    # Cluster means (ordered)
    mu_raw = pm.Normal('mu_raw', mu=-2.6, sigma=1.0, shape=3)
    mu = pm.Deterministic('mu', pt.sort(mu_raw))

    # Cluster SDs
    sigma = pm.HalfNormal('sigma', sigma=0.5, shape=3)

    # Mixture
    components = [pm.Normal.dist(mu=mu[k], sigma=sigma[k]) for k in range(3)]
    theta = pm.Mixture('theta', w=pi, comp_dists=components, shape=n_groups)

    # Likelihood
    p = pm.Deterministic('p', pm.math.invlogit(theta))
    r = pm.Binomial('r', n=n_trials, p=p, observed=r_successes)

    # Cluster assignment probabilities
    log_likes_k = [pm.logp(pm.Normal.dist(mu=mu[k], sigma=sigma[k]), theta) for k in range(3)]
    log_likes_matrix = pt.stack(log_likes_k, axis=1)
    log_post_unnorm = log_likes_matrix + pt.log(pi)
    cluster_probs = pm.Deterministic('cluster_probs',
                                     pm.math.softmax(log_post_unnorm, axis=1))

print("Model built!")

# Extended sampling for better convergence
print("\n" + "="*70)
print("EXTENDED SAMPLING FOR CONVERGENCE")
print("="*70)
print("\nStrategy:")
print("  - 4 chains")
print("  - 1000 warmup, 1500 sampling")
print("  - target_accept = 0.95 (increased from 0.90)")

with mixture_model:
    trace = pm.sample(
        draws=1500,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        init='adapt_diag',
        return_inferencedata=True,
        random_seed=43,
        progressbar=True
    )

print("\nSampling complete!")

# Convergence diagnostics
print("\n" + "="*70)
print("CONVERGENCE DIAGNOSTICS")
print("="*70)

summary = az.summary(trace, var_names=['pi', 'mu', 'sigma'])
print("\nParameter Summary:")
print(summary)

rhat_max = summary['r_hat'].max()
ess_min = summary['ess_bulk'].min()
divergences = trace.sample_stats.diverging.sum().item()
total_draws = trace.posterior.dims['draw'] * trace.posterior.dims['chain']

print(f"\nConvergence metrics:")
print(f"  Max R-hat: {rhat_max:.4f} (target: < 1.01)")
print(f"  Min ESS_bulk: {ess_min:.1f} (target: > 400)")
print(f"  Divergences: {divergences} / {total_draws} ({100*divergences/total_draws:.2f}%)")

mu_samples = trace.posterior['mu'].values.reshape(-1, 3)
ordering_ok = np.sum((mu_samples[:, 1] > mu_samples[:, 0]) &
                     (mu_samples[:, 2] > mu_samples[:, 1]))
print(f"  Ordering maintained: {ordering_ok} / {len(mu_samples)} ({100*ordering_ok/len(mu_samples):.1f}%)")

# Extract log_lik from posterior (it's a Deterministic variable)
print("\nExtracting log-likelihood for LOO...")
if 'log_lik' in trace.posterior:
    print("ERROR: log_lik not found in posterior!")
    print("Available variables:", list(trace.posterior.data_vars))
    # Compute it manually
    print("\nComputing log_lik manually...")
    p_samples = trace.posterior['p'].values  # shape: (chains, draws, n_groups)

    # Compute log likelihood for each sample
    log_lik_list = []
    for chain_idx in range(p_samples.shape[0]):
        for draw_idx in range(p_samples.shape[1]):
            p_draw = p_samples[chain_idx, draw_idx, :]
            # Binomial log-likelihood
            ll = np.sum([
                np.log(np.math.comb(n_trials[j], r_successes[j])) +
                r_successes[j] * np.log(p_draw[j] + 1e-10) +
                (n_trials[j] - r_successes[j]) * np.log(1 - p_draw[j] + 1e-10)
                for j in range(n_groups)
            ])
            log_lik_list.append(ll)

    log_lik_array = np.array(log_lik_list).reshape(p_samples.shape[0], p_samples.shape[1], 1)
    print(f"Computed log_lik shape: {log_lik_array.shape}")
else:
    # Compute pointwise log-likelihood
    print("Computing pointwise log-likelihood...")
    import scipy.stats as stats
    p_samples = trace.posterior['p'].values

    log_lik_pointwise = np.zeros((p_samples.shape[0], p_samples.shape[1], n_groups))
    for chain_idx in range(p_samples.shape[0]):
        for draw_idx in range(p_samples.shape[1]):
            for j in range(n_groups):
                log_lik_pointwise[chain_idx, draw_idx, j] = stats.binom.logpmf(
                    r_successes[j], n_trials[j], p_samples[chain_idx, draw_idx, j]
                )

    # Add to InferenceData
    import xarray as xr
    log_lik_da = xr.DataArray(
        log_lik_pointwise,
        dims=['chain', 'draw', 'obs_id'],
        coords={
            'chain': trace.posterior.coords['chain'],
            'draw': trace.posterior.coords['draw'],
            'obs_id': range(n_groups)
        }
    )

    trace.add_groups({'log_likelihood': xr.Dataset({'r': log_lik_da})})
    print(f"Added log_likelihood to InferenceData: shape {log_lik_da.shape}")

# Save results
print("\nSaving results...")
trace.to_netcdf(diag_dir / "posterior_inference.netcdf")
print(f"  Saved InferenceData: {diag_dir / 'posterior_inference.netcdf'}")

summary_full = az.summary(trace)
summary_full.to_csv(diag_dir / "parameter_summary.csv")
print(f"  Saved parameter summary")

with open(diag_dir / "idata.pkl", 'wb') as f:
    pickle.dump(trace, f)
print(f"  Saved idata pickle")

# Verify log_likelihood
if 'log_likelihood' in trace.groups():
    print("\n✓ log_likelihood successfully included in InferenceData")
    print(f"  Variables: {list(trace.log_likelihood.data_vars)}")
    print(f"  Shape: {trace.log_likelihood['r'].shape}")
else:
    print("\n✗ WARNING: log_likelihood group not found!")

print("\n" + "="*70)
print("EXTENDED FITTING COMPLETE")
print("="*70)
