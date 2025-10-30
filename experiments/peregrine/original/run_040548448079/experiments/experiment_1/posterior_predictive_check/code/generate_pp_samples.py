"""
Generate posterior predictive samples for PPC analysis.

The fitted model is simplified (no AR(1)), so we generate replicated data
by drawing from the Negative Binomial distribution using posterior parameter samples.

Model: C_t ~ NegativeBinomial(mu_t, alpha)
       log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * I(t > 17) * (year_t - year_17)
"""

import arviz as az
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
data = pd.read_csv('/workspace/data/data.csv')
C_obs = data['C'].values
year = data['year'].values
N = len(C_obs)
tau = 17  # Fixed changepoint (1-indexed, Python will use 0-indexed)
year_tau = year[tau - 1]

print(f"  N = {N} observations")
print(f"  Changepoint at observation {tau} (year = {year_tau:.3f})")
print(f"  Pre-break: observations 1-{tau}")
print(f"  Post-break: observations {tau+1}-{N}")

# Create post-break interaction term
post_break = (np.arange(N) >= tau).astype(float)
year_post = post_break * (year - year_tau)

# Load posterior
print("\nLoading posterior samples...")
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

# Extract posterior samples
print("Extracting posterior parameters...")
posterior = idata.posterior

beta_0_samples = posterior['beta_0'].values.flatten()
beta_1_samples = posterior['beta_1'].values.flatten()
beta_2_samples = posterior['beta_2'].values.flatten()
alpha_samples = posterior['alpha'].values.flatten()

n_total_samples = len(beta_0_samples)
print(f"  Total posterior samples: {n_total_samples}")

# Subsample for efficiency (500 samples)
n_pp_samples = min(500, n_total_samples)
idx = np.random.choice(n_total_samples, size=n_pp_samples, replace=False)

beta_0_subset = beta_0_samples[idx]
beta_1_subset = beta_1_samples[idx]
beta_2_subset = beta_2_samples[idx]
alpha_subset = alpha_samples[idx]

print(f"  Subsampled to: {n_pp_samples} samples")

# Generate posterior predictive samples
print("\nGenerating posterior predictive samples...")
C_rep = np.zeros((n_pp_samples, N), dtype=int)

for i in range(n_pp_samples):
    # Compute mu for this parameter draw
    log_mu = beta_0_subset[i] + beta_1_subset[i] * year + beta_2_subset[i] * year_post
    mu = np.exp(log_mu)

    # Generate counts from Negative Binomial
    # PyMC uses NegativeBinomial(mu, alpha) parameterization
    # where alpha = 1/phi (inverse dispersion)
    # variance = mu + mu^2/alpha = mu + phi*mu^2

    # NumPy uses NegativeBinomial(n, p) where mean = n*(1-p)/p
    # To convert: n = alpha, p = alpha/(alpha + mu)
    n = alpha_subset[i]
    p = n / (n + mu)

    C_rep[i, :] = np.random.negative_binomial(n=n, p=p)

    if (i + 1) % 100 == 0:
        print(f"  Generated {i+1}/{n_pp_samples} samples")

print("  Complete!")

# Save for analysis
print("\nSaving generated samples...")
np.save('/workspace/experiments/experiment_1/posterior_predictive_check/code/C_rep.npy', C_rep)
np.save('/workspace/experiments/experiment_1/posterior_predictive_check/code/C_obs.npy', C_obs)
np.save('/workspace/experiments/experiment_1/posterior_predictive_check/code/year.npy', year)
np.save('/workspace/experiments/experiment_1/posterior_predictive_check/code/tau.npy', tau)

print("  Saved:")
print("    - C_rep.npy: Posterior predictive samples (500 x 40)")
print("    - C_obs.npy: Observed data (40,)")
print("    - year.npy: Year covariate (40,)")
print("    - tau.npy: Changepoint location")

# Sanity checks
print("\n" + "="*80)
print("SANITY CHECK")
print("="*80)
print(f"\nObserved data:")
print(f"  Range: [{C_obs.min()}, {C_obs.max()}]")
print(f"  Mean: {C_obs.mean():.1f}")
print(f"  Variance: {C_obs.var():.1f}")
print(f"  Var/Mean ratio: {C_obs.var()/C_obs.mean():.2f}")

print(f"\nPosterior predictive samples:")
print(f"  Range: [{C_rep.min()}, {C_rep.max()}]")
print(f"  Mean (avg across samples): {C_rep.mean():.1f}")
print(f"  Variance (avg): {C_rep.var(axis=1).mean():.1f}")
print(f"  Var/Mean ratio (avg): {(C_rep.var(axis=1)/C_rep.mean(axis=1)).mean():.2f}")

# Check pre/post break patterns
pre_obs_mean = C_obs[:tau].mean()
post_obs_mean = C_obs[tau:].mean()
pre_pp_mean = C_rep[:, :tau].mean()
post_pp_mean = C_rep[:, tau:].mean()

print(f"\nPre-break (t ≤ {tau}):")
print(f"  Observed mean: {pre_obs_mean:.1f}")
print(f"  PP mean: {pre_pp_mean:.1f}")

print(f"\nPost-break (t > {tau}):")
print(f"  Observed mean: {post_obs_mean:.1f}")
print(f"  PP mean: {post_pp_mean:.1f}")

print(f"\nGrowth ratio:")
print(f"  Observed: {post_obs_mean/pre_obs_mean:.2f}x")
print(f"  PP: {post_pp_mean/pre_pp_mean:.2f}x")

print("\n" + "="*80)
print("Generation complete!")
print("="*80)
