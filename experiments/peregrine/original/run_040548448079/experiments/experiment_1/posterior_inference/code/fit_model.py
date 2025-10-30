"""
Fit simplified negative binomial changepoint model to real data.

Model: C_t ~ NegativeBinomial(mu_t, alpha)
       log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * I(t > 17) * (year_t - year_17)

Simplified: AR(1) terms omitted due to computational constraints.
"""

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("FITTING SIMPLIFIED NEGATIVE BINOMIAL CHANGEPOINT MODEL")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
data = pd.read_csv('/workspace/data/data.csv')
year = data['year'].values
C = data['C'].values
N = len(C)
tau = 17  # Fixed changepoint (1-indexed)
year_tau = year[tau-1]  # Year value at changepoint

print(f"  N = {N} observations")
print(f"  Changepoint: tau = {tau}")
print(f"  year_tau = {year_tau:.3f}")
print(f"  Count range: [{C.min()}, {C.max()}]")

# Create post-break indicator and interaction term
post_break = (np.arange(N) >= tau).astype(float)
year_post = post_break * (year - year_tau)

print(f"  Pre-break observations: {(1-post_break).sum():.0f}")
print(f"  Post-break observations: {post_break.sum():.0f}")

# Build PyMC model
print("\n[2/5] Building PyMC model...")
with pm.Model() as model:
    # Priors (from revised specification)
    beta_0 = pm.Normal('beta_0', mu=4.3, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.35, sigma=0.3)
    beta_2 = pm.Normal('beta_2', mu=0.85, sigma=0.5)
    alpha = pm.Gamma('alpha', alpha=2, beta=3)

    # Mean structure (log-linear with changepoint)
    log_mu = beta_0 + beta_1 * year + beta_2 * year_post
    mu = pm.math.exp(log_mu)

    # Likelihood
    # PyMC uses NegativeBinomial(mu, alpha) where:
    #   - mu is the mean
    #   - alpha is the inverse dispersion (higher alpha = less dispersion)
    # We want: var = mu + phi * mu^2, where phi is dispersion
    # PyMC parameterization: alpha = 1/phi
    obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha, observed=C)

print("  Model structure:")
print("    log(mu_t) = beta_0 + beta_1*year + beta_2*I(t>17)*(year - year_17)")
print("    C_t ~ NegativeBinomial(mu_t, alpha)")
print("\n  Priors:")
print("    beta_0 ~ Normal(4.3, 0.5)")
print("    beta_1 ~ Normal(0.35, 0.3)")
print("    beta_2 ~ Normal(0.85, 0.5)")
print("    alpha ~ Gamma(2, 3)")

# Sample from posterior
print("\n[3/5] Sampling from posterior...")
print("  Configuration:")
print("    Chains: 4")
print("    Draws: 2000 per chain")
print("    Tune: 2000")
print("    Target accept: 0.95")
print("\n  Sampling (this may take 5-10 minutes)...")

with model:
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.95,
        return_inferencedata=True,
        random_seed=42,
        cores=4
    )

    # Compute log-likelihood for LOO
    print("\n[4/5] Computing log-likelihood for LOO...")
    pm.compute_log_likelihood(trace)

print("  Sampling complete!")

# Save inference data
print("\n[5/5] Saving results...")
output_path = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf'
trace.to_netcdf(output_path)
print(f"  Saved to: {output_path}")

# Quick convergence summary
print("\n" + "="*80)
print("QUICK CONVERGENCE SUMMARY")
print("="*80)

summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'alpha'])
print(summary)

# Check for issues
rhat_max = summary['r_hat'].max()
ess_bulk_min = summary['ess_bulk'].min()
ess_tail_min = summary['ess_tail'].min()

print("\n" + "="*80)
print("CONVERGENCE STATUS")
print("="*80)
print(f"  Max R-hat: {rhat_max:.4f} (target: < 1.01)")
print(f"  Min ESS bulk: {ess_bulk_min:.0f} (target: > 400)")
print(f"  Min ESS tail: {ess_tail_min:.0f} (target: > 400)")

# Check divergences
divergences = trace.sample_stats.diverging.sum().values
n_samples = trace.posterior.dims['draw'] * trace.posterior.dims['chain']
div_pct = 100 * divergences / n_samples
print(f"  Divergences: {divergences} / {n_samples} ({div_pct:.2f}%)")

if rhat_max < 1.01 and ess_bulk_min > 400 and ess_tail_min > 400 and div_pct < 1:
    print("\n✓ ALL CONVERGENCE CRITERIA MET")
else:
    print("\n⚠ CONVERGENCE ISSUES DETECTED")
    if rhat_max >= 1.01:
        print(f"  - R-hat too high: {rhat_max:.4f}")
    if ess_bulk_min <= 400:
        print(f"  - ESS bulk too low: {ess_bulk_min:.0f}")
    if ess_tail_min <= 400:
        print(f"  - ESS tail too low: {ess_tail_min:.0f}")
    if div_pct >= 1:
        print(f"  - Too many divergences: {div_pct:.2f}%")

print("\n" + "="*80)
print("PARAMETER INFERENCE")
print("="*80)

# Extract posterior means and HDIs
beta_2_samples = trace.posterior['beta_2'].values.flatten()
beta_2_mean = beta_2_samples.mean()
beta_2_hdi = az.hdi(trace, var_names=['beta_2'], hdi_prob=0.95)['beta_2'].values
prob_beta2_positive = (beta_2_samples > 0).mean()

beta_1_samples = trace.posterior['beta_1'].values.flatten()
beta_1_mean = beta_1_samples.mean()

post_slope_samples = beta_1_samples + beta_2_samples
post_slope_mean = post_slope_samples.mean()
slope_ratio = post_slope_mean / beta_1_mean

print(f"\nRegime Change (beta_2):")
print(f"  Posterior mean: {beta_2_mean:.3f}")
print(f"  95% HDI: [{beta_2_hdi[0]:.3f}, {beta_2_hdi[1]:.3f}]")
print(f"  P(beta_2 > 0): {prob_beta2_positive:.4f}")

print(f"\nPre-break slope (beta_1): {beta_1_mean:.3f}")
print(f"Post-break slope (beta_1 + beta_2): {post_slope_mean:.3f}")
print(f"Slope ratio (post/pre): {slope_ratio:.2f}x")

alpha_mean = trace.posterior['alpha'].values.flatten().mean()
print(f"\nDispersion (alpha): {alpha_mean:.3f}")
print(f"  (For reference, EDA estimated alpha ≈ 0.61)")

print("\n" + "="*80)
print("COMPLETE - Results saved to:")
print(f"  {output_path}")
print("="*80)
