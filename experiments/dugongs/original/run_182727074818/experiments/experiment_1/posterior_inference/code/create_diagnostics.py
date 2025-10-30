"""
Create comprehensive diagnostic visualizations for HMC sampling.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Set paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = BASE_DIR / "diagnostics"
PLOT_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Set plotting style
plt.style.use('default')
az.style.use('arviz-darkgrid')

print("=" * 80)
print("CREATING DIAGNOSTIC VISUALIZATIONS")
print("=" * 80)

# Load inference data
print("\n[1/5] Loading InferenceData...")
idata_path = DIAG_DIR / 'posterior_inference.netcdf'
idata = az.from_netcdf(idata_path)
print(f"  - Loaded from: {idata_path}")
print(f"  - Groups: {idata.groups()}")

# Load data
df = pd.read_csv(DATA_PATH)
x_data = df['x'].values
Y_data = df['Y'].values

# 1. Convergence Overview: Trace plots
print("\n[2/5] Creating convergence diagnostics (trace and rank plots)...")
params = ['alpha', 'beta', 'c', 'nu', 'sigma']

# Trace plots
fig = plt.figure(figsize=(14, 12))
az.plot_trace(
    idata,
    var_names=params,
    compact=False
)
plt.suptitle('Convergence Diagnostics: Trace Plots', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'trace_plots.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOT_DIR / 'trace_plots.png'}")
plt.close()

# Rank plots
fig = plt.figure(figsize=(12, 10))
az.plot_rank(
    idata,
    var_names=params
)
plt.suptitle('Rank Plots (ECDF uniformity check)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'rank_plots.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOT_DIR / 'rank_plots.png'}")
plt.close()

# 2. Posterior distributions with priors
print("\n[3/5] Creating posterior distributions with prior overlays...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Posterior Distributions (with Prior Overlays)', fontsize=14, fontweight='bold')

# Extract posterior samples
posterior_samples = {
    'alpha': idata.posterior['alpha'].values.flatten(),
    'beta': idata.posterior['beta'].values.flatten(),
    'c': idata.posterior['c'].values.flatten(),
    'nu': idata.posterior['nu'].values.flatten(),
    'sigma': idata.posterior['sigma'].values.flatten()
}

# Define priors for plotting
from scipy import stats

# alpha ~ Normal(2.0, 0.5)
ax = axes[0, 0]
x = np.linspace(0, 3.5, 200)
ax.hist(posterior_samples['alpha'], bins=40, density=True, alpha=0.6, label='Posterior', color='steelblue')
ax.plot(x, stats.norm.pdf(x, 2.0, 0.5), 'r--', lw=2, label='Prior')
ax.axvline(posterior_samples['alpha'].mean(), color='k', linestyle=':', label='Posterior mean')
ax.set_xlabel('alpha')
ax.set_ylabel('Density')
ax.set_title('Intercept (alpha)')
ax.legend()
ax.grid(alpha=0.3)

# beta ~ Normal(0.3, 0.2)
ax = axes[0, 1]
x = np.linspace(-0.2, 0.8, 200)
ax.hist(posterior_samples['beta'], bins=40, density=True, alpha=0.6, label='Posterior', color='steelblue')
ax.plot(x, stats.norm.pdf(x, 0.3, 0.2), 'r--', lw=2, label='Prior')
ax.axvline(posterior_samples['beta'].mean(), color='k', linestyle=':', label='Posterior mean')
ax.set_xlabel('beta')
ax.set_ylabel('Density')
ax.set_title('Slope (beta)')
ax.legend()
ax.grid(alpha=0.3)

# c ~ Gamma(2, 2)
ax = axes[0, 2]
x = np.linspace(0, 3, 200)
ax.hist(posterior_samples['c'], bins=40, density=True, alpha=0.6, label='Posterior', color='steelblue')
ax.plot(x, stats.gamma.pdf(x, 2, scale=1/2), 'r--', lw=2, label='Prior')
ax.axvline(posterior_samples['c'].mean(), color='k', linestyle=':', label='Posterior mean')
ax.set_xlabel('c')
ax.set_ylabel('Density')
ax.set_title('Shift parameter (c)')
ax.legend()
ax.grid(alpha=0.3)

# nu ~ Gamma(2, 0.1)
ax = axes[1, 0]
x = np.linspace(0, 100, 200)
ax.hist(posterior_samples['nu'], bins=40, density=True, alpha=0.6, label='Posterior', color='steelblue')
ax.plot(x, stats.gamma.pdf(x, 2, scale=1/0.1), 'r--', lw=2, label='Prior')
ax.axvline(posterior_samples['nu'].mean(), color='k', linestyle=':', label='Posterior mean')
ax.set_xlabel('nu')
ax.set_ylabel('Density')
ax.set_title('Degrees of freedom (nu)')
ax.legend()
ax.grid(alpha=0.3)

# sigma ~ HalfNormal(0.15)
ax = axes[1, 1]
x = np.linspace(0, 0.3, 200)
ax.hist(posterior_samples['sigma'], bins=40, density=True, alpha=0.6, label='Posterior', color='steelblue')
# HalfNormal: pdf(x) = sqrt(2/pi) * 1/sigma * exp(-x^2 / (2*sigma^2))
sigma_prior = 0.15
prior_pdf = np.sqrt(2/np.pi) / sigma_prior * np.exp(-x**2 / (2 * sigma_prior**2))
ax.plot(x, prior_pdf, 'r--', lw=2, label='Prior')
ax.axvline(posterior_samples['sigma'].mean(), color='k', linestyle=':', label='Posterior mean')
ax.set_xlabel('sigma')
ax.set_ylabel('Density')
ax.set_title('Scale parameter (sigma)')
ax.legend()
ax.grid(alpha=0.3)

# Summary statistics
ax = axes[1, 2]
ax.axis('off')
summary_text = "Posterior Summaries:\n\n"
for param in params:
    mean = posterior_samples[param].mean()
    std = posterior_samples[param].std()
    q5, q95 = np.percentile(posterior_samples[param], [5, 95])
    summary_text += f"{param}:\n"
    summary_text += f"  {mean:.4f} ± {std:.4f}\n"
    summary_text += f"  90% CI: [{q5:.4f}, {q95:.4f}]\n\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(PLOT_DIR / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOT_DIR / 'posterior_distributions.png'}")
plt.close()

# 3. Pair plot for correlated parameters
print("\n[4/5] Creating pair plot for parameter correlations...")
fig = plt.figure(figsize=(12, 10))
az.plot_pair(
    idata,
    var_names=['alpha', 'beta', 'c'],
    kind='kde',
    marginals=True,
    point_estimate='mean',
    divergences=True
)
plt.suptitle('Parameter Correlations (alpha, beta, c)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'pair_plot.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOT_DIR / 'pair_plot.png'}")
plt.close()

# 4. Energy and MCMC diagnostics
print("\n[5/5] Creating energy and MCMC diagnostic plots...")
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, figure=fig)

# Energy plot
ax1 = fig.add_subplot(gs[0, :])
az.plot_energy(idata, ax=ax1)
ax1.set_title('Energy Plot (Marginal vs Transition Energy)', fontweight='bold')

# Autocorrelation
ax2 = fig.add_subplot(gs[1, 0])
az.plot_autocorr(idata, var_names=['alpha'], combined=True, ax=ax2)
ax2.set_title('Autocorrelation: alpha', fontweight='bold')

ax3 = fig.add_subplot(gs[1, 1])
az.plot_autocorr(idata, var_names=['c'], combined=True, ax=ax3)
ax3.set_title('Autocorrelation: c', fontweight='bold')

# ESS evolution
ax4 = fig.add_subplot(gs[2, 0])
az.plot_ess(idata, var_names=params, kind='evolution', ax=ax4)
ax4.set_title('ESS Evolution', fontweight='bold')

# MCSE
ax5 = fig.add_subplot(gs[2, 1])
az.plot_mcse(idata, var_names=params, ax=ax5)
ax5.set_title('Monte Carlo Standard Error', fontweight='bold')

plt.suptitle('MCMC Diagnostic Plots', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'mcmc_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOT_DIR / 'mcmc_diagnostics.png'}")
plt.close()

# 5. Posterior predictive check visualization
print("\n[BONUS] Creating posterior predictive fit plot...")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.scatter(x_data, Y_data, s=60, alpha=0.7, color='black',
           label='Observed data', zorder=3, edgecolors='white', linewidths=0.5)

# Generate predictions
x_pred = np.linspace(0.5, 32, 200)
n_samples = 500  # Use subset of posterior samples for speed

# Get posterior samples
alpha_post = posterior_samples['alpha'][:n_samples]
beta_post = posterior_samples['beta'][:n_samples]
c_post = posterior_samples['c'][:n_samples]

# Compute predictions
y_pred_samples = np.zeros((n_samples, len(x_pred)))
for i in range(n_samples):
    y_pred_samples[i] = alpha_post[i] + beta_post[i] * np.log(x_pred + c_post[i])

# Plot credible interval
y_pred_mean = y_pred_samples.mean(axis=0)
y_pred_5 = np.percentile(y_pred_samples, 5, axis=0)
y_pred_95 = np.percentile(y_pred_samples, 95, axis=0)

ax.plot(x_pred, y_pred_mean, 'b-', lw=2, label='Posterior mean', zorder=2)
ax.fill_between(x_pred, y_pred_5, y_pred_95, alpha=0.3, color='blue',
                label='90% Credible interval', zorder=1)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Posterior Predictive Fit: Y ~ StudentT(nu, alpha + beta*log(x+c), sigma)',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'posterior_predictive_fit.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOT_DIR / 'posterior_predictive_fit.png'}")
plt.close()

print("\n" + "=" * 80)
print("ALL DIAGNOSTIC VISUALIZATIONS COMPLETED")
print("=" * 80)
