"""
Posterior Inference for Eight Schools Model
============================================

Fits the non-centered hierarchical model to Eight Schools data using PyMC.

Model:
    y_i ~ Normal(theta_i, sigma_i)     [sigma_i known]
    theta_i = mu + tau * eta_i
    eta_i ~ Normal(0, 1)
    mu ~ Normal(0, 20)
    tau ~ Half-Cauchy(0, 5)
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

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DATA_PATH = Path("/workspace/data/data.csv")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

print("="*70)
print("POSTERIOR INFERENCE: Eight Schools Model")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(data)} schools")
print(data)

# Extract data
J = len(data)
y_obs = data['y'].values
sigma_obs = data['sigma'].values

# ============================================================================
# 2. BUILD MODEL (Non-Centered Parameterization)
# ============================================================================
print("\n[2/6] Building PyMC model (non-centered parameterization)...")

with pm.Model() as model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfCauchy('tau', beta=5)

    # Non-centered parameterization
    eta = pm.Normal('eta', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * eta)

    # Likelihood
    y = pm.Normal('y', mu=theta, sigma=sigma_obs, observed=y_obs)

    # Compute log-likelihood for LOO
    log_lik = pm.logp(y, y_obs)
    pm.Deterministic('log_lik', log_lik)

print("Model built successfully")

# ============================================================================
# 3. SAMPLE FROM POSTERIOR
# ============================================================================
print("\n[3/6] Sampling from posterior...")
print("Configuration:")
print("  - 4 chains")
print("  - 2000 iterations per chain (1000 warmup)")
print("  - target_accept = 0.95")
print("  - NUTS sampler")

with model:
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=RANDOM_SEED,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

print("\nSampling complete!")

# ============================================================================
# 4. SAVE INFERENCE DATA
# ============================================================================
print("\n[4/6] Saving InferenceData...")
inference_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
trace.to_netcdf(inference_path)
print(f"Saved to: {inference_path}")

# Verify log_likelihood is present
print(f"Groups in InferenceData: {list(trace.groups())}")
if 'log_likelihood' in trace.groups():
    print("✓ log_likelihood group present for LOO-CV")
else:
    print("WARNING: log_likelihood group missing!")

# ============================================================================
# 5. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[5/6] Running convergence diagnostics...")

# Summary statistics
summary = az.summary(trace, var_names=['mu', 'tau', 'theta'])
print("\nParameter Summary:")
print(summary)

# Save diagnostics
diagnostics_text = DIAGNOSTICS_DIR / "convergence_diagnostics.txt"
with open(diagnostics_text, 'w') as f:
    f.write("="*70 + "\n")
    f.write("CONVERGENCE DIAGNOSTICS: Eight Schools Model\n")
    f.write("="*70 + "\n\n")

    # Summary table
    f.write("PARAMETER SUMMARY\n")
    f.write("-"*70 + "\n")
    f.write(summary.to_string())
    f.write("\n\n")

    # Convergence criteria
    f.write("CONVERGENCE CRITERIA\n")
    f.write("-"*70 + "\n")

    max_rhat = summary['r_hat'].max()
    min_ess_bulk = summary['ess_bulk'].min()
    min_ess_tail = summary['ess_tail'].min()

    f.write(f"Max R-hat: {max_rhat:.4f} (target: < 1.01)\n")
    f.write(f"Min ESS bulk: {min_ess_bulk:.1f} (target: > 400)\n")
    f.write(f"Min ESS tail: {min_ess_tail:.1f} (target: > 400)\n")

    # Check divergences
    divergences = trace.sample_stats.diverging.sum().item()
    total_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
    divergence_rate = divergences / total_samples

    f.write(f"\nDivergences: {divergences} / {total_samples} ({divergence_rate*100:.2f}%)\n")
    f.write(f"Target: < 1%\n")

    # Overall assessment
    f.write("\n" + "="*70 + "\n")
    f.write("OVERALL ASSESSMENT\n")
    f.write("="*70 + "\n")

    converged = (max_rhat < 1.01) and (min_ess_bulk > 400) and (divergence_rate < 0.01)

    if converged:
        f.write("✓ CONVERGENCE ACHIEVED\n")
        f.write("  - All R-hat < 1.01\n")
        f.write("  - All ESS > 400\n")
        f.write("  - Divergences < 1%\n")
    else:
        f.write("✗ CONVERGENCE ISSUES DETECTED\n")
        if max_rhat >= 1.01:
            f.write(f"  - R-hat too high: {max_rhat:.4f}\n")
        if min_ess_bulk <= 400:
            f.write(f"  - ESS too low: {min_ess_bulk:.1f}\n")
        if divergence_rate >= 0.01:
            f.write(f"  - Too many divergences: {divergence_rate*100:.2f}%\n")

print(f"\nDiagnostics saved to: {diagnostics_text}")
print(f"\nConvergence Status:")
print(f"  Max R-hat: {max_rhat:.4f} {'✓' if max_rhat < 1.01 else '✗'}")
print(f"  Min ESS: {min_ess_bulk:.1f} {'✓' if min_ess_bulk > 400 else '✗'}")
print(f"  Divergences: {divergences} ({divergence_rate*100:.2f}%) {'✓' if divergence_rate < 0.01 else '✗'}")

# ============================================================================
# 6. COMPUTE DERIVED QUANTITIES
# ============================================================================
print("\n[6/6] Computing derived quantities...")

# Extract posterior samples
mu_samples = trace.posterior['mu'].values.flatten()
tau_samples = trace.posterior['tau'].values.flatten()
theta_samples = trace.posterior['theta'].values.reshape(-1, J)

# Posterior means
mu_mean = mu_samples.mean()
tau_mean = tau_samples.mean()
theta_means = theta_samples.mean(axis=0)

# Shrinkage factors
# Shrinkage = 1 - Var(theta_i | y) / Var(y_i)
# Approximate: distance from observed y to grand mean
shrinkage = []
for j in range(J):
    # Posterior variance of theta_i
    post_var = theta_samples[:, j].var()
    # Prior variance (approximation)
    pooled_var = sigma_obs[j]**2
    # Shrinkage factor
    shrink = 1 - (post_var / (pooled_var + tau_mean**2))
    shrinkage.append(shrink)

shrinkage = np.array(shrinkage)

# LOO-CV
print("\nComputing LOO-CV...")
loo = az.loo(trace, var_name='log_lik')
print(loo)

# Save derived quantities
derived_path = DIAGNOSTICS_DIR / "derived_quantities.txt"
with open(derived_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("DERIVED QUANTITIES\n")
    f.write("="*70 + "\n\n")

    f.write("POSTERIOR MEANS\n")
    f.write("-"*70 + "\n")
    f.write(f"mu: {mu_mean:.2f}\n")
    f.write(f"tau: {tau_mean:.2f}\n")
    f.write("\ntheta (by school):\n")
    for j in range(J):
        f.write(f"  School {j+1}: {theta_means[j]:.2f}\n")

    f.write("\n\nSHRINKAGE FACTORS\n")
    f.write("-"*70 + "\n")
    for j in range(J):
        f.write(f"  School {j+1}: {shrinkage[j]:.2%}\n")
    f.write(f"\nMean shrinkage: {shrinkage.mean():.2%}\n")

    f.write("\n\nLOO-CV\n")
    f.write("-"*70 + "\n")
    f.write(f"ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}\n")
    f.write(f"p_eff: {loo.p_loo:.2f}\n")

print(f"Derived quantities saved to: {derived_path}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[7/7] Creating visualizations...")

# --- Plot 1: Trace plots ---
print("  - Trace plots...")
fig, axes = plt.subplots(5, 2, figsize=(14, 12))
fig.suptitle('Trace Plots: Eight Schools Model', fontsize=16, fontweight='bold')

# mu
az.plot_trace(trace, var_names=['mu'], axes=axes[0, :], combined=False)
axes[0, 0].set_title('mu: Posterior Distribution')
axes[0, 1].set_title('mu: Trace')

# tau
az.plot_trace(trace, var_names=['tau'], axes=axes[1, :], combined=False)
axes[1, 0].set_title('tau: Posterior Distribution')
axes[1, 1].set_title('tau: Trace')

# theta[0], theta[3], theta[7] (representative schools)
for idx, school_idx in enumerate([0, 3, 7]):
    ax_dist = axes[2+idx, 0]
    ax_trace = axes[2+idx, 1]

    # Distribution
    for chain in range(4):
        theta_chain = trace.posterior['theta'].sel(chain=chain, theta_dim_0=school_idx).values
        ax_dist.hist(theta_chain, bins=30, alpha=0.5, label=f'Chain {chain}')
    ax_dist.set_xlabel(f'theta[{school_idx}]')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title(f'theta[{school_idx}] (School {school_idx+1}): Posterior')
    ax_dist.legend()

    # Trace
    for chain in range(4):
        theta_chain = trace.posterior['theta'].sel(chain=chain, theta_dim_0=school_idx).values
        ax_trace.plot(theta_chain, alpha=0.7)
    ax_trace.set_xlabel('Iteration')
    ax_trace.set_ylabel(f'theta[{school_idx}]')
    ax_trace.set_title(f'theta[{school_idx}] (School {school_idx+1}): Trace')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'trace_plots.png'}")

# --- Plot 2: Rank plots ---
print("  - Rank plots...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(trace, var_names=['mu', 'tau', 'theta'])
plt.suptitle('Rank Plots: MCMC Mixing Diagnostics', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'rank_plots.png'}")

# --- Plot 3: Forest plot ---
print("  - Forest plot...")
fig, ax = plt.subplots(figsize=(10, 6))

# Posterior estimates
theta_hdi = az.hdi(trace, var_names=['theta'], hdi_prob=0.95)['theta'].values

# Plot posterior intervals
for j in range(J):
    ax.plot([theta_hdi[j, 0], theta_hdi[j, 1]], [j, j], 'o-', linewidth=2,
            markersize=8, label='95% HDI' if j == 0 else '', color='steelblue')
    ax.plot(theta_means[j], j, 'o', markersize=10, color='darkblue',
            label='Posterior mean' if j == 0 else '', zorder=5)

# Plot observed data
ax.errorbar(y_obs, np.arange(J), xerr=sigma_obs*1.96, fmt='s',
            markersize=8, color='coral', capsize=5, alpha=0.7,
            label='Observed ± 1.96σ', zorder=3)

# Grand mean
ax.axvline(mu_mean, color='green', linestyle='--', linewidth=2,
           label=f'Grand mean (μ={mu_mean:.1f})', alpha=0.7)

ax.set_yticks(range(J))
ax.set_yticklabels([f'School {j+1}' for j in range(J)])
ax.set_xlabel('Effect Size', fontsize=12)
ax.set_ylabel('School', fontsize=12)
ax.set_title('Forest Plot: Observed vs Posterior Estimates', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'forest_plot.png'}")

# --- Plot 4: Shrinkage plot ---
print("  - Shrinkage analysis...")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot identity line
lims = [min(y_obs.min(), theta_means.min()) - 5,
        max(y_obs.max(), theta_means.max()) + 5]
ax.plot(lims, lims, 'k--', alpha=0.3, label='No shrinkage', zorder=1)

# Plot shrinkage arrows
for j in range(J):
    ax.annotate('', xy=(theta_means[j], theta_means[j]),
                xytext=(y_obs[j], y_obs[j]),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.6))

    # Observed point
    ax.errorbar(y_obs[j], y_obs[j], xerr=sigma_obs[j]*1.96, yerr=sigma_obs[j]*1.96,
                fmt='o', markersize=10, color='coral', capsize=5, alpha=0.7, zorder=3)

    # Posterior point
    ax.plot(theta_means[j], theta_means[j], 's', markersize=12,
            color='steelblue', zorder=4)

    # Label
    ax.text(y_obs[j], y_obs[j]+2, f'S{j+1}', fontsize=9, ha='center')

# Grand mean line
ax.axvline(mu_mean, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(mu_mean, color='green', linestyle='--', linewidth=2, alpha=0.5,
           label=f'Grand mean (μ={mu_mean:.1f})')

# Dummy points for legend
ax.plot([], [], 'o', markersize=10, color='coral', label='Observed')
ax.plot([], [], 's', markersize=12, color='steelblue', label='Posterior mean')
ax.plot([], [], '->', color='gray', label='Shrinkage')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Observed Effect', fontsize=12)
ax.set_ylabel('Posterior Mean Effect', fontsize=12)
ax.set_title('Shrinkage Analysis: Hierarchical Pooling', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "shrinkage_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'shrinkage_analysis.png'}")

# --- Plot 5: Pair plot (mu, tau) ---
print("  - Pair plot (mu, tau)...")
fig = plt.figure(figsize=(10, 10))
az.plot_pair(trace, var_names=['mu', 'tau'],
             kind='kde', marginals=True,
             figsize=(10, 10))
plt.suptitle('Joint Posterior: (μ, τ)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pair_plot_mu_tau.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'pair_plot_mu_tau.png'}")

# --- Plot 6: Energy plot ---
print("  - Energy plot...")
fig = plt.figure(figsize=(10, 6))
az.plot_energy(trace)
plt.title('Energy Plot: Sampling Geometry Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "energy_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'energy_plot.png'}")

print("\n" + "="*70)
print("POSTERIOR INFERENCE COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {BASE_DIR}")
print(f"  - InferenceData: {inference_path}")
print(f"  - Diagnostics: {diagnostics_text}")
print(f"  - Plots: {PLOTS_DIR}/")
