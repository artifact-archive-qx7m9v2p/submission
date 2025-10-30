"""
Create diagnostic plots and visualizations for posterior inference
"""

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
PLOT_DIR = BASE_DIR / "plots"
DIAG_DIR = BASE_DIR / "diagnostics"
DATA_PATH = Path("/workspace/data/data.csv")

# Load data
print("Loading InferenceData and observed data...")
idata = az.from_netcdf(DIAG_DIR / "posterior_inference.netcdf")
data = pd.read_csv(DATA_PATH)

print("\n1. Creating convergence diagnostic plots...")

# Plot 1: Trace plots for main parameters
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Trace Plots: Main Parameters", fontsize=14, fontweight='bold')

params = ['delta', 'sigma_eta', 'phi']
param_names = ['Drift (δ)', 'Innovation SD (σ_η)', 'Dispersion (φ)']

for i, (param, name) in enumerate(zip(params, param_names)):
    # Trace plot
    ax = axes[i, 0]
    for chain in range(4):
        samples = idata.posterior[param].sel(chain=chain).values
        ax.plot(samples, alpha=0.7, linewidth=0.5, label=f'Chain {chain}')
    ax.set_ylabel(name)
    ax.set_xlabel('Iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Posterior distribution
    ax = axes[i, 1]
    az.plot_posterior(idata, var_names=[param], ax=ax, textsize=10)
    ax.set_title(f'{name} Posterior')

plt.tight_layout()
plt.savefig(PLOT_DIR / "convergence_trace_plots.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {PLOT_DIR / 'convergence_trace_plots.png'}")
plt.close()

# Plot 2: Rank plots (diagnostic for convergence)
print("\n2. Creating rank plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Rank Plots: Chain Mixing Diagnostics", fontsize=14, fontweight='bold')

for i, param in enumerate(params):
    az.plot_rank(idata, var_names=[param], ax=axes[i])
    axes[i].set_title(param_names[i])

plt.tight_layout()
plt.savefig(PLOT_DIR / "convergence_rank_plots.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {PLOT_DIR / 'convergence_rank_plots.png'}")
plt.close()

# Plot 3: Posterior distributions with prior overlays
print("\n3. Creating posterior vs prior comparison...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Posterior Distributions with Prior Overlays", fontsize=14, fontweight='bold')

# Delta
ax = axes[0]
posterior_delta = idata.posterior['delta'].values.flatten()
x = np.linspace(posterior_delta.min(), posterior_delta.max(), 200)
ax.hist(posterior_delta, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
ax.plot(x, stats.norm.pdf(x, 0.05, 0.02), 'r--', linewidth=2, label='Prior')
ax.axvline(posterior_delta.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean = {posterior_delta.mean():.4f}')
ax.set_xlabel('Drift (δ)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# Sigma_eta
ax = axes[1]
posterior_sigma = idata.posterior['sigma_eta'].values.flatten()
x = np.linspace(0, max(posterior_sigma.max(), 0.2), 200)
ax.hist(posterior_sigma, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
ax.plot(x, stats.expon.pdf(x, scale=1/20), 'r--', linewidth=2, label='Prior')
ax.axvline(posterior_sigma.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean = {posterior_sigma.mean():.4f}')
ax.set_xlabel('Innovation SD (σ_η)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# Phi
ax = axes[2]
posterior_phi = idata.posterior['phi'].values.flatten()
x = np.linspace(0, min(posterior_phi.max(), 100), 200)
ax.hist(posterior_phi, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
ax.plot(x, stats.expon.pdf(x, scale=1/0.05), 'r--', linewidth=2, label='Prior')
ax.axvline(posterior_phi.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean = {posterior_phi.mean():.2f}')
ax.set_xlabel('Dispersion (φ)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "posterior_vs_prior.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {PLOT_DIR / 'posterior_vs_prior.png'}")
plt.close()

# Plot 4: Latent state estimates
print("\n4. Creating latent state trajectory plot...")
fig, ax = plt.subplots(figsize=(14, 6))

# Extract eta samples
eta_samples = idata.posterior['eta'].values  # (chains, draws, time)
eta_flat = eta_samples.reshape(-1, len(data))  # (chains*draws, time)

# Compute statistics
eta_mean = eta_flat.mean(axis=0)
eta_lower = np.percentile(eta_flat, 2.5, axis=0)
eta_upper = np.percentile(eta_flat, 97.5, axis=0)

# Plot
time = data['year'].values
ax.plot(time, eta_mean, 'b-', linewidth=2, label='Posterior Mean')
ax.fill_between(time, eta_lower, eta_upper, alpha=0.3, color='blue', label='95% CI')

# Add observed log(C) for comparison
ax.scatter(time, np.log(data['C'].values + 0.5), alpha=0.5, s=30, color='red',
           label='log(Observed C)', zorder=5)

ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Latent State η (log scale)', fontsize=12)
ax.set_title('Latent State Trajectory with 95% Credible Intervals', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "latent_state_trajectory.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {PLOT_DIR / 'latent_state_trajectory.png'}")
plt.close()

# Plot 5: Pairs plot for parameter correlations
print("\n5. Creating pairs plot...")
fig = az.plot_pair(idata, var_names=['delta', 'sigma_eta', 'phi'],
                   kind='kde', marginals=True, figsize=(10, 10))
plt.suptitle('Parameter Correlations (Pairs Plot)', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOT_DIR / "parameter_pairs.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {PLOT_DIR / 'parameter_pairs.png'}")
plt.close()

# Plot 6: Posterior predictive check
print("\n6. Creating posterior predictive check...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 6a: Observed vs predicted counts
ax = axes[0]
C_pred = idata.posterior_predictive['C'].values  # (chains, draws, time)
C_pred_flat = C_pred.reshape(-1, len(data))
C_pred_mean = C_pred_flat.mean(axis=0)
C_pred_lower = np.percentile(C_pred_flat, 2.5, axis=0)
C_pred_upper = np.percentile(C_pred_flat, 97.5, axis=0)

ax.plot(time, data['C'].values, 'ko-', linewidth=1, markersize=4, label='Observed', zorder=5)
ax.plot(time, C_pred_mean, 'b-', linewidth=2, alpha=0.8, label='Predicted Mean')
ax.fill_between(time, C_pred_lower, C_pred_upper, alpha=0.2, color='blue', label='95% Predictive Interval')
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Observed vs Predicted Counts', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 6b: Residuals
ax = axes[1]
residuals = data['C'].values - C_pred_mean
ax.scatter(time, residuals, alpha=0.6, s=40)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Residual (Observed - Predicted)', fontsize=11)
ax.set_title('Residuals', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "posterior_predictive_check.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {PLOT_DIR / 'posterior_predictive_check.png'}")
plt.close()

# Plot 7: Autocorrelation plots
print("\n7. Creating autocorrelation plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Autocorrelation Plots: Check for Chain Independence", fontsize=14, fontweight='bold')

for i, param in enumerate(params):
    az.plot_autocorr(idata, var_names=[param], combined=True, ax=axes[i])
    axes[i].set_title(param_names[i])

plt.tight_layout()
plt.savefig(PLOT_DIR / "autocorrelation_plots.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {PLOT_DIR / 'autocorrelation_plots.png'}")
plt.close()

# Plot 8: Energy plot (if available)
print("\n8. Creating energy diagnostic plot...")
try:
    fig = az.plot_energy(idata, figsize=(10, 6))
    plt.suptitle('Energy Diagnostic Plot', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "energy_diagnostic.png", dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {PLOT_DIR / 'energy_diagnostic.png'}")
    plt.close()
except Exception as e:
    print(f"   ⚠ Energy plot not available: {e}")

print("\n" + "="*80)
print("ALL DIAGNOSTIC PLOTS CREATED")
print("="*80)
print(f"\nPlots saved to: {PLOT_DIR}")
print(f"  - convergence_trace_plots.png")
print(f"  - convergence_rank_plots.png")
print(f"  - posterior_vs_prior.png")
print(f"  - latent_state_trajectory.png")
print(f"  - parameter_pairs.png")
print(f"  - posterior_predictive_check.png")
print(f"  - autocorrelation_plots.png")
