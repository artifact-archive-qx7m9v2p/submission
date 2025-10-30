"""
Create diagnostic and inference visualizations for posterior inference.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from statsmodels.tsa.stattools import acf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CREATING DIAGNOSTIC AND INFERENCE VISUALIZATIONS")
print("="*80)

# Load data
print("\n[1/7] Loading data and inference results...")
data = pd.read_csv('/workspace/data/data.csv')
year = data['year'].values
C = data['C'].values
N = len(C)
tau = 17
year_tau = year[tau-1]

trace = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
print("  Data and trace loaded")

# Create post-break indicator
post_break = (np.arange(N) >= tau).astype(float)
year_post = post_break * (year - year_tau)

# ============================================================================
# 1. CONVERGENCE OVERVIEW - Trace and rank plots
# ============================================================================
print("\n[2/7] Creating convergence overview plots...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

param_names = ['beta_0', 'beta_1', 'beta_2', 'alpha']

for i, param in enumerate(param_names):
    # Trace plot
    ax_trace = fig.add_subplot(gs[i, :2])
    samples = trace.posterior[param].values
    for chain in range(samples.shape[0]):
        ax_trace.plot(samples[chain, :], alpha=0.7, linewidth=0.5)
    ax_trace.set_ylabel(param, fontsize=12)
    ax_trace.set_xlabel('Iteration')
    if i == 0:
        ax_trace.set_title('Trace Plots (Chain Mixing)', fontsize=14, fontweight='bold')

    # Rank plot
    ax_rank = fig.add_subplot(gs[i, 2:])
    az.plot_rank(trace, var_names=[param], ax=ax_rank)
    if i == 0:
        ax_rank.set_title('Rank Plots (Uniformity Check)', fontsize=14, fontweight='bold')

fig.suptitle('Convergence Diagnostics Overview', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/workspace/experiments/experiment_1/posterior_inference/plots/convergence_overview.png',
            dpi=300, bbox_inches='tight')
print("  Saved: convergence_overview.png")
plt.close()

# ============================================================================
# 2. POSTERIOR DISTRIBUTIONS
# ============================================================================
print("\n[3/7] Creating posterior distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, param in enumerate(param_names):
    samples = trace.posterior[param].values.flatten()

    axes[i].hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Add KDE
    kde = stats.gaussian_kde(samples)
    x_range = np.linspace(samples.min(), samples.max(), 200)
    axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Add posterior mean and HDI
    mean_val = samples.mean()
    hdi = az.hdi(trace, var_names=[param], hdi_prob=0.95)[param].values

    axes[i].axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    axes[i].axvspan(hdi[0], hdi[1], alpha=0.2, color='green', label=f'95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]')

    # Add prior (for comparison)
    if param == 'beta_0':
        prior_x = np.linspace(3, 5.5, 200)
        prior_pdf = stats.norm.pdf(prior_x, 4.3, 0.5)
        axes[i].plot(prior_x, prior_pdf, 'b--', linewidth=2, alpha=0.5, label='Prior')
    elif param == 'beta_1':
        prior_x = np.linspace(-0.5, 1.5, 200)
        prior_pdf = stats.norm.pdf(prior_x, 0.35, 0.3)
        axes[i].plot(prior_x, prior_pdf, 'b--', linewidth=2, alpha=0.5, label='Prior')
    elif param == 'beta_2':
        prior_x = np.linspace(-0.5, 2, 200)
        prior_pdf = stats.norm.pdf(prior_x, 0.85, 0.5)
        axes[i].plot(prior_x, prior_pdf, 'b--', linewidth=2, alpha=0.5, label='Prior')
        # Add reference line at 0
        axes[i].axvline(0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='H0: β₂=0')
    elif param == 'alpha':
        prior_x = np.linspace(0, 12, 200)
        prior_pdf = stats.gamma.pdf(prior_x, a=2, scale=1/3)
        axes[i].plot(prior_x, prior_pdf, 'b--', linewidth=2, alpha=0.5, label='Prior')

    axes[i].set_xlabel(param, fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Posterior Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_inference/plots/posterior_distributions.png',
            dpi=300, bbox_inches='tight')
print("  Saved: posterior_distributions.png")
plt.close()

# ============================================================================
# 3. FITTED MODEL - Data + Predictions
# ============================================================================
print("\n[4/7] Creating fitted model visualization...")

# Compute posterior predictive mean and quantiles
beta_0_samples = trace.posterior['beta_0'].values.flatten()
beta_1_samples = trace.posterior['beta_1'].values.flatten()
beta_2_samples = trace.posterior['beta_2'].values.flatten()
alpha_samples = trace.posterior['alpha'].values.flatten()

# Compute mu for each posterior sample
n_samples = len(beta_0_samples)
mu_samples = np.zeros((n_samples, N))

for i in range(n_samples):
    log_mu = beta_0_samples[i] + beta_1_samples[i] * year + beta_2_samples[i] * year_post
    mu_samples[i, :] = np.exp(log_mu)

# Compute quantiles
mu_mean = mu_samples.mean(axis=0)
mu_05 = np.percentile(mu_samples, 5, axis=0)
mu_95 = np.percentile(mu_samples, 95, axis=0)

fig, ax = plt.subplots(figsize=(14, 8))

# Plot observed data
ax.scatter(np.arange(1, N+1), C, s=80, alpha=0.7, color='black',
           label='Observed data', zorder=5, edgecolor='white', linewidth=1)

# Plot posterior mean
ax.plot(np.arange(1, N+1), mu_mean, 'b-', linewidth=3, label='Posterior mean μ', zorder=4)

# Plot uncertainty band
ax.fill_between(np.arange(1, N+1), mu_05, mu_95, alpha=0.3, color='blue',
                label='90% uncertainty interval', zorder=3)

# Add vertical line at changepoint
ax.axvline(tau, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Changepoint (τ = {tau})', zorder=2)

# Shade pre/post break regions
ax.axvspan(0, tau, alpha=0.1, color='gray', label='Pre-break')
ax.axvspan(tau, N+1, alpha=0.1, color='orange', label='Post-break')

ax.set_xlabel('Observation Index', fontsize=14)
ax.set_ylabel('Count (C)', fontsize=14)
ax.set_title('Fitted Model: Observed Data and Posterior Predictions', fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N+1)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_inference/plots/fitted_model.png',
            dpi=300, bbox_inches='tight')
print("  Saved: fitted_model.png")
plt.close()

# ============================================================================
# 4. RESIDUAL ANALYSIS
# ============================================================================
print("\n[5/7] Creating residual analysis plots...")

# Compute residuals (Pearson)
residuals_pearson = (C - mu_mean) / np.sqrt(mu_mean * (1 + mu_mean / alpha_samples.mean()))

# Compute ACF
acf_vals = acf(residuals_pearson, nlags=15, fft=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs time
axes[0, 0].scatter(np.arange(1, N+1), residuals_pearson, alpha=0.7, s=60, edgecolor='black')
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].axvline(tau, color='red', linestyle=':', linewidth=2, alpha=0.5)
axes[0, 0].set_xlabel('Observation Index', fontsize=12)
axes[0, 0].set_ylabel('Pearson Residuals', fontsize=12)
axes[0, 0].set_title('Residuals vs Time', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs fitted
axes[0, 1].scatter(mu_mean, residuals_pearson, alpha=0.7, s=60, edgecolor='black')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Fitted Values (μ)', fontsize=12)
axes[0, 1].set_ylabel('Pearson Residuals', fontsize=12)
axes[0, 1].set_title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(residuals_pearson, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# ACF plot
axes[1, 1].bar(range(len(acf_vals)), acf_vals, width=0.4, alpha=0.7, edgecolor='black')
axes[1, 1].axhline(0, color='black', linewidth=1)
axes[1, 1].axhline(1.96/np.sqrt(N), color='red', linestyle='--', linewidth=2, label='95% CI')
axes[1, 1].axhline(-1.96/np.sqrt(N), color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Lag', fontsize=12)
axes[1, 1].set_ylabel('ACF', fontsize=12)
axes[1, 1].set_title(f'Autocorrelation Function (ACF[1] = {acf_vals[1]:.3f})',
                     fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

# Add text annotation about ACF
if acf_vals[1] < 0.3:
    status = "Excellent (< 0.3)"
    color = "green"
elif acf_vals[1] < 0.5:
    status = "Acceptable (< 0.5)"
    color = "orange"
else:
    status = "Concerning (> 0.5)"
    color = "red"

axes[1, 1].text(0.95, 0.95, f"ACF(1) Status: {status}",
                transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

fig.suptitle('Residual Diagnostics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_inference/plots/residual_diagnostics.png',
            dpi=300, bbox_inches='tight')
print("  Saved: residual_diagnostics.png")
plt.close()

# ============================================================================
# 5. PARAMETER CORRELATIONS - Pairs plot
# ============================================================================
print("\n[6/7] Creating parameter correlation plot...")

az.plot_pair(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'alpha'],
             kind='kde', figsize=(12, 12),
             divergences=True,
             textsize=12)
plt.suptitle('Parameter Posterior Correlations', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/workspace/experiments/experiment_1/posterior_inference/plots/parameter_correlations.png',
            dpi=300, bbox_inches='tight')
print("  Saved: parameter_correlations.png")
plt.close()

# ============================================================================
# 6. LOO DIAGNOSTICS
# ============================================================================
print("\n[7/7] Computing LOO and creating diagnostic plot...")

# Compute LOO
loo = az.loo(trace, pointwise=True)

# Save LOO results
loo_path = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/loo_results.txt'
with open(loo_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("LEAVE-ONE-OUT CROSS-VALIDATION (LOO-CV)\n")
    f.write("="*80 + "\n\n")
    f.write(str(loo) + "\n\n")

    # Check Pareto k diagnostics
    pareto_k = loo.pareto_k.values
    k_high = (pareto_k > 0.7).sum()
    k_pct = 100 * k_high / len(pareto_k)

    f.write("PARETO K DIAGNOSTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total observations: {len(pareto_k)}\n")
    f.write(f"k > 0.7 (problematic): {k_high} ({k_pct:.1f}%)\n")
    f.write(f"k > 0.5 (concerning): {(pareto_k > 0.5).sum()}\n")
    f.write(f"Max k: {pareto_k.max():.3f}\n")
    f.write(f"Mean k: {pareto_k.mean():.3f}\n\n")

    if k_pct > 10:
        f.write("⚠ WARNING: >10% of observations have k > 0.7\n")
        f.write("  This suggests the model may not fit well for some observations.\n")
    else:
        f.write("✓ Pareto k diagnostic acceptable\n")

print(f"  Saved LOO results to: {loo_path}")

# Plot Pareto k values
fig, ax = plt.subplots(figsize=(14, 6))

ax.scatter(np.arange(1, N+1), loo.pareto_k.values, s=80, alpha=0.7,
           c=loo.pareto_k.values, cmap='RdYlGn_r', edgecolor='black', linewidth=1)

# Add threshold lines
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='k = 0.5 (concerning)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.7, label='k = 0.7 (bad)')
ax.axvline(tau, color='blue', linestyle=':', linewidth=2, alpha=0.5, label=f'Changepoint (τ = {tau})')

ax.set_xlabel('Observation Index', fontsize=14)
ax.set_ylabel('Pareto k', fontsize=14)
ax.set_title('LOO Pareto k Diagnostic Values', fontsize=16, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N+1)

# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Pareto k', fontsize=12)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_inference/plots/loo_diagnostics.png',
            dpi=300, bbox_inches='tight')
print("  Saved: loo_diagnostics.png")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
print("\nSaved plots:")
print("  1. convergence_overview.png")
print("  2. posterior_distributions.png")
print("  3. fitted_model.png")
print("  4. residual_diagnostics.png")
print("  5. parameter_correlations.png")
print("  6. loo_diagnostics.png")
