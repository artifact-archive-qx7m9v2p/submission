"""
Complete Pooling Model - Posterior Inference
=============================================

Fits the complete pooling model to real data using PyMC.

Model:
    Likelihood: y_i ~ Normal(mu, sigma_i)  [known sigma_i from data]
    Prior:      mu ~ Normal(10, 20)

Author: Bayesian Computation Specialist
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("/workspace")
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUTPUT_DIR = BASE_DIR / "experiments" / "experiment_1" / "posterior_inference"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories if they don't exist
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPLETE POOLING MODEL - POSTERIOR INFERENCE")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("1. Loading data...")
df = pd.read_csv(DATA_PATH)
y_obs = df['y'].values
sigma_obs = df['sigma'].values
N = len(y_obs)

print(f"   - Observations: {N}")
print(f"   - y values: {y_obs}")
print(f"   - sigma values: {sigma_obs}")
print()

# ============================================================================
# 2. FIT MODEL USING PYMC
# ============================================================================
print("2. Fitting model with PyMC...")
print("   - Chains: 4")
print("   - Draws per chain: 2000")
print("   - Warmup: 1000")
print("   - Target accept: 0.90")
print()

with pm.Model() as model:
    # Prior
    mu = pm.Normal('mu', mu=10, sigma=20)

    # Likelihood with known measurement error
    y = pm.Normal('y', mu=mu, sigma=sigma_obs, observed=y_obs)

    # Sample posterior
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.90,
        return_inferencedata=True,
        random_seed=RANDOM_SEED
    )

    # CRITICAL: Compute log-likelihood for LOO-CV
    print("\n   Computing log-likelihood for LOO-CV...")
    pm.compute_log_likelihood(trace)

print("\n   Sampling complete!")
print()

# Check log-likelihood structure
print("   Checking InferenceData structure...")
print(f"   Groups in trace: {list(trace.groups())}")
if hasattr(trace, 'log_likelihood'):
    print(f"   Variables in log_likelihood: {list(trace.log_likelihood.data_vars)}")

# ============================================================================
# 3. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n3. Convergence Diagnostics")
print("-" * 80)

# Get summary statistics
summary = az.summary(trace, var_names=['mu'])
print("\nParameter Summary:")
print(summary)
print()

# Extract key diagnostics
rhat = summary['r_hat'].values[0]
ess_bulk = summary['ess_bulk'].values[0]
ess_tail = summary['ess_tail'].values[0]

# Check for divergences
divergences = trace.sample_stats.diverging.sum().values
n_chains = len(trace.posterior.chain)
n_draws = len(trace.posterior.draw)
total_draws = n_chains * n_draws
divergence_rate = divergences / total_draws

print(f"Convergence Metrics:")
print(f"  - R-hat: {rhat:.6f}")
print(f"  - ESS (bulk): {ess_bulk:.0f}")
print(f"  - ESS (tail): {ess_tail:.0f}")
print(f"  - Divergences: {divergences} / {total_draws} ({divergence_rate*100:.2f}%)")
print()

# Convergence status
convergence_pass = (
    rhat < 1.01 and
    ess_bulk > 400 and
    ess_tail > 400 and
    divergence_rate < 0.01
)

print(f"Convergence Status: {'PASS' if convergence_pass else 'FAIL'}")
if convergence_pass:
    print("  All convergence criteria met!")
else:
    print("  WARNING: Convergence issues detected!")
    if rhat >= 1.01:
        print(f"    - R-hat too high: {rhat:.6f} >= 1.01")
    if ess_bulk <= 400:
        print(f"    - ESS bulk too low: {ess_bulk:.0f} <= 400")
    if ess_tail <= 400:
        print(f"    - ESS tail too low: {ess_tail:.0f} <= 400")
    if divergence_rate >= 0.01:
        print(f"    - Too many divergences: {divergence_rate*100:.2f}% >= 1%")
print()

# Save convergence summary
convergence_df = pd.DataFrame({
    'parameter': ['mu'],
    'r_hat': [rhat],
    'ess_bulk': [ess_bulk],
    'ess_tail': [ess_tail],
    'divergences': [divergences],
    'total_draws': [total_draws],
    'divergence_rate': [divergence_rate],
    'convergence_pass': [convergence_pass]
})
convergence_df.to_csv(DIAGNOSTICS_DIR / "convergence_summary.csv", index=False)
print(f"   Saved: {DIAGNOSTICS_DIR / 'convergence_summary.csv'}")

# ============================================================================
# 4. POSTERIOR SUMMARY
# ============================================================================
print("\n4. Posterior Summary")
print("-" * 80)

# Extract posterior samples
mu_samples = trace.posterior['mu'].values.flatten()

# Compute statistics
posterior_mean = mu_samples.mean()
posterior_std = mu_samples.std()
posterior_median = np.percentile(mu_samples, 50)
ci_90 = np.percentile(mu_samples, [5, 95])
ci_95 = np.percentile(mu_samples, [2.5, 97.5])

print(f"\nPosterior for mu:")
print(f"  - Mean: {posterior_mean:.3f}")
print(f"  - Std: {posterior_std:.3f}")
print(f"  - Median: {posterior_median:.3f}")
print(f"  - 90% CI: [{ci_90[0]:.3f}, {ci_90[1]:.3f}]")
print(f"  - 95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
print()

# Compare to prior
prior_mean = 10.0
prior_std = 20.0
print(f"Prior for mu: Normal({prior_mean}, {prior_std})")
print(f"Posterior contraction: {prior_std / posterior_std:.2f}x")
print()

# Save posterior summary
posterior_df = summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_bulk', 'ess_tail', 'r_hat']]
posterior_df.to_csv(DIAGNOSTICS_DIR / "posterior_summary.csv")
print(f"   Saved: {DIAGNOSTICS_DIR / 'posterior_summary.csv'}")

# ============================================================================
# 5. SAVE INFERENCE DATA (CRITICAL FOR LOO-CV)
# ============================================================================
print("\n5. Saving InferenceData with log_likelihood")
print("-" * 80)

# Verify log_likelihood is present
if hasattr(trace, 'log_likelihood') and 'y' in trace.log_likelihood:
    print("   Log-likelihood successfully computed!")
    print(f"   Shape: {trace.log_likelihood['y'].shape}")
    print(f"   Dimensions: {trace.log_likelihood["y"].dims}")
else:
    print("   WARNING: log_likelihood structure may be different than expected")
    print(f"   Available groups: {list(trace.groups())}")

# Save InferenceData
netcdf_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
trace.to_netcdf(netcdf_path)
print(f"   Saved: {netcdf_path}")
print()

# ============================================================================
# 6. DIAGNOSTIC PLOTS
# ============================================================================
print("6. Creating diagnostic plots...")
print("-" * 80)

# 6.1 Trace Plot
print("   Creating trace plot...")
fig = plt.figure(figsize=(14, 4))
az.plot_trace(trace, var_names=['mu'], figsize=(14, 4))
plt.suptitle('Trace Plot: Complete Pooling Model', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: {PLOTS_DIR / 'trace_plot.png'}")

# 6.2 Posterior vs Prior
print("   Creating posterior vs prior comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

# Prior distribution
x_range = np.linspace(-50, 70, 1000)
prior_pdf = (1 / (prior_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - prior_mean) / prior_std)**2)
ax.plot(x_range, prior_pdf, 'b--', linewidth=2, label=f'Prior: N({prior_mean}, {prior_std})', alpha=0.7)

# Posterior distribution
az.plot_dist(mu_samples, ax=ax, label=f'Posterior: N({posterior_mean:.2f}, {posterior_std:.2f})',
             color='red', plot_kwargs={'linewidth': 2})

# Add vertical lines for credible intervals
ax.axvline(ci_95[0], color='red', linestyle=':', alpha=0.5, label='95% CI')
ax.axvline(ci_95[1], color='red', linestyle=':', alpha=0.5)

ax.set_xlabel('mu', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior vs Posterior Distribution', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: {PLOTS_DIR / 'posterior_distribution.png'}")

# 6.3 Forest Plot
print("   Creating forest plot...")
fig, ax = plt.subplots(figsize=(10, 4))
az.plot_forest(trace, var_names=['mu'], combined=True, hdi_prob=0.95)
plt.title('Parameter Estimates with 95% Credible Intervals', fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: {PLOTS_DIR / 'forest_plot.png'}")

# 6.4 Autocorrelation Plot
print("   Creating autocorrelation plot...")
fig = plt.figure(figsize=(10, 6))
az.plot_autocorr(trace, var_names=['mu'], combined=True)
plt.suptitle('Autocorrelation Plot', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "autocorrelation.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: {PLOTS_DIR / 'autocorrelation.png'}")

# 6.5 Rank Plot (additional diagnostic)
print("   Creating rank plot...")
fig = plt.figure(figsize=(10, 6))
az.plot_rank(trace, var_names=['mu'])
plt.suptitle('Rank Plot: Checking Chain Mixing', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: {PLOTS_DIR / 'rank_plot.png'}")

# 6.6 Comprehensive convergence overview
print("   Creating convergence overview...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trace plot for each chain
ax = axes[0, 0]
for chain in range(n_chains):
    chain_samples = trace.posterior['mu'].sel(chain=chain).values
    ax.plot(chain_samples, alpha=0.7, label=f'Chain {chain+1}')
ax.set_xlabel('Iteration')
ax.set_ylabel('mu')
ax.set_title('Trace Plot by Chain')
ax.legend()
ax.grid(True, alpha=0.3)

# Posterior histogram
ax = axes[0, 1]
ax.hist(mu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(posterior_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {posterior_mean:.2f}')
ax.axvline(ci_95[0], color='orange', linestyle=':', linewidth=2, label='95% CI')
ax.axvline(ci_95[1], color='orange', linestyle=':', linewidth=2)
ax.set_xlabel('mu')
ax.set_ylabel('Density')
ax.set_title('Posterior Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Rank histogram
ax = axes[1, 0]
ranks = []
for chain in range(n_chains):
    chain_samples = trace.posterior['mu'].sel(chain=chain).values
    all_samples = mu_samples
    chain_ranks = np.array([np.sum(all_samples <= s) for s in chain_samples])
    ranks.append(chain_ranks)
ax.hist(np.concatenate(ranks), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('Rank Histogram (Should be Uniform)')
ax.grid(True, alpha=0.3)

# Autocorrelation
ax = axes[1, 1]
lags = np.arange(0, 50)
for chain in range(n_chains):
    chain_samples = trace.posterior['mu'].sel(chain=chain).values
    acf = [np.corrcoef(chain_samples[:-lag] if lag > 0 else chain_samples,
                       chain_samples[lag:] if lag > 0 else chain_samples)[0, 1]
           for lag in lags]
    ax.plot(lags, acf, alpha=0.7, label=f'Chain {chain+1}')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation by Chain')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Convergence Diagnostics Overview', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_overview.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: {PLOTS_DIR / 'convergence_overview.png'}")

print()
print("="*80)
print("FITTING COMPLETE")
print("="*80)
print()
print("Summary:")
print(f"  - Convergence: {'PASS' if convergence_pass else 'FAIL'}")
print(f"  - Posterior mean (mu): {posterior_mean:.3f} ± {posterior_std:.3f}")
print(f"  - 95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
print(f"  - R-hat: {rhat:.6f}")
print(f"  - ESS (bulk): {ess_bulk:.0f}")
print(f"  - Divergences: {divergences}")
print()
print("Next Steps:")
print("  1. Review convergence diagnostics")
print("  2. Proceed to Posterior Predictive Check")
print("  3. Compare with EDA expectations")
print()
print("Files saved to:")
print(f"  - {DIAGNOSTICS_DIR}")
print(f"  - {PLOTS_DIR}")
print()
