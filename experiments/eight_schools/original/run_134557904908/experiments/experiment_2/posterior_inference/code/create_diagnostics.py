"""
Create Diagnostic Visualizations for Hierarchical Model

Generates comprehensive convergence and posterior diagnostic plots.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_context("notebook")
plt.style.use('seaborn-v0_8-darkgrid')

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_2/posterior_inference")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

print("="*70)
print("CREATING DIAGNOSTIC VISUALIZATIONS")
print("="*70)

# Load InferenceData
idata_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
print(f"\nLoading InferenceData from: {idata_path}")
idata = az.from_netcdf(idata_path)

print("✓ Loaded!")
print(f"  Posterior shape: {idata.posterior.dims}")

# Figure 1: Trace plots for hyperparameters
print("\nCreating trace plots...")

fig = plt.figure(figsize=(14, 8))
az.plot_trace(idata, var_names=['mu', 'tau'], compact=True, show=False)
plt.suptitle('Trace Plots - Hyperparameters', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots_hyperparameters.png', dpi=300, bbox_inches='tight')
print(f"  Saved: trace_plots_hyperparameters.png")
plt.close()

# Figure 2: Rank plots (sensitive diagnostic for convergence)
print("Creating rank plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
az.plot_rank(idata, var_names=['mu'], ax=axes[0], show=False)
axes[0].set_title('μ Rank Plot', fontsize=12, fontweight='bold')
az.plot_rank(idata, var_names=['tau'], ax=axes[1], show=False)
axes[1].set_title('τ Rank Plot', fontsize=12, fontweight='bold')
plt.suptitle('Rank Plots - MCMC Diagnostic', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rank_plots.png', dpi=300, bbox_inches='tight')
print(f"  Saved: rank_plots.png")
plt.close()

# Figure 3: Autocorrelation
print("Creating autocorrelation plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
az.plot_autocorr(idata, var_names=['mu'], ax=axes[0], show=False)
axes[0].set_title('μ Autocorrelation', fontsize=12, fontweight='bold')
az.plot_autocorr(idata, var_names=['tau'], ax=axes[1], show=False)
axes[1].set_title('τ Autocorrelation', fontsize=12, fontweight='bold')
plt.suptitle('Autocorrelation - Chain Mixing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'autocorrelation.png', dpi=300, bbox_inches='tight')
print(f"  Saved: autocorrelation.png")
plt.close()

# Figure 4: Study-specific effects trace
print("Creating theta trace plots...")

fig = plt.figure(figsize=(16, 12))
az.plot_trace(idata, var_names=['theta'], compact=True, show=False)
plt.suptitle('Trace Plots - Study-Specific Effects θ', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots_theta.png', dpi=300, bbox_inches='tight')
print(f"  Saved: trace_plots_theta.png")
plt.close()

# Figure 5: Posterior distributions with prior overlay
print("Creating posterior vs prior comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# μ posterior vs prior
ax = axes[0]
mu_samples = idata.posterior['mu'].values.flatten()
ax.hist(mu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Posterior')

# Prior
mu_prior = np.random.normal(0, 20, 10000)
ax.hist(mu_prior, bins=50, density=True, alpha=0.3, color='gray', edgecolor='black', label='Prior N(0, 20²)')
ax.axvline(mu_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Posterior mean={mu_samples.mean():.2f}')
ax.set_xlabel('μ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('μ: Posterior vs Prior', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# τ posterior vs prior
ax = axes[1]
tau_samples = idata.posterior['tau'].values.flatten()
ax.hist(tau_samples, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black', label='Posterior')

# Prior
tau_prior = np.abs(np.random.normal(0, 5, 10000))
ax.hist(tau_prior, bins=50, density=True, alpha=0.3, color='gray', edgecolor='black', label='Prior Half-N(0, 5²)')
ax.axvline(tau_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Posterior mean={tau_samples.mean():.2f}')
ax.axvline(0, color='green', linestyle=':', linewidth=2, label='τ=0 (homogeneity)')
ax.set_xlabel('τ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('τ: Posterior vs Prior', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Prior-Posterior Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_posterior_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: prior_posterior_comparison.png")
plt.close()

# Figure 6: Forest plot showing partial pooling
print("Creating forest plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
y_obs = data['y'].values
sigma_obs = data['sigma'].values

# Extract posterior means and HDI for theta
theta_summary = az.summary(idata, var_names=['theta'])
theta_means = theta_summary['mean'].values
theta_hdi_low = theta_summary['hdi_3%'].values
theta_hdi_high = theta_summary['hdi_97%'].values

# μ posterior
mu_mean = mu_samples.mean()
mu_hdi = az.hdi(mu_samples, hdi_prob=0.95)

# Plot observed data
ax.errorbar(y_obs, range(8), xerr=1.96*sigma_obs, fmt='o', color='gray',
            markersize=8, label='Observed y ± 95% CI', alpha=0.6, capsize=5)

# Plot posterior theta
ax.errorbar(theta_means, range(8), xerr=[theta_means - theta_hdi_low, theta_hdi_high - theta_means],
            fmt='s', color='steelblue', markersize=8, label='Posterior θ (95% HDI)',
            alpha=0.8, capsize=5, linewidth=2)

# Plot μ
ax.axvline(mu_mean, color='red', linestyle='--', linewidth=2, label=f'μ (population mean) = {mu_mean:.2f}')
ax.axvspan(mu_hdi[0], mu_hdi[1], alpha=0.2, color='red', label='μ 95% HDI')

ax.set_yticks(range(8))
ax.set_yticklabels([f'Study {i+1}' for i in range(8)])
ax.set_xlabel('Effect Size', fontsize=12)
ax.set_ylabel('Study', fontsize=12)
ax.set_title('Forest Plot: Partial Pooling Effects', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'forest_plot.png', dpi=300, bbox_inches='tight')
print(f"  Saved: forest_plot.png")
plt.close()

# Figure 7: Heterogeneity analysis
print("Creating heterogeneity analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# I² distribution
sigma_sq_mean = np.mean(sigma_obs**2)
I_sq_samples = 100 * tau_samples**2 / (tau_samples**2 + sigma_sq_mean)

ax = axes[0, 0]
ax.hist(I_sq_samples, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(I_sq_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={I_sq_samples.mean():.1f}%')
ax.axvline(np.median(I_sq_samples), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(I_sq_samples):.1f}%')
ax.axvline(25, color='green', linestyle=':', linewidth=2, label='Low/Moderate threshold (25%)')
ax.set_xlabel('I² (%)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('I² Distribution (Heterogeneity Statistic)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# τ vs μ (check for funnel)
ax = axes[0, 1]
ax.scatter(mu_samples, tau_samples, alpha=0.1, s=1, color='steelblue')
ax.set_xlabel('μ', fontsize=12)
ax.set_ylabel('τ', fontsize=12)
ax.set_title('Joint Posterior: τ vs μ\n(Checking for Funnel Pathology)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# τ posterior with interpretive regions
ax = axes[1, 0]
ax.hist(tau_samples, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(tau_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={tau_samples.mean():.2f}')
ax.axvline(0, color='green', linestyle=':', linewidth=2, label='τ=0 (no heterogeneity)')
tau_xlim = ax.get_xlim()
ax.axvspan(0, min(1, tau_xlim[1]), alpha=0.2, color='green', label='Very low heterogeneity')
ax.axvspan(min(1, tau_xlim[1]), min(5, tau_xlim[1]), alpha=0.2, color='yellow', label='Low-moderate heterogeneity')
if tau_xlim[1] > 5:
    ax.axvspan(5, tau_xlim[1], alpha=0.2, color='red', label='High heterogeneity')
ax.set_xlabel('τ (between-study SD)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('τ Posterior with Interpretive Regions', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Summary table
ax = axes[1, 1]
ax.axis('off')

summary_text = "Heterogeneity Summary\n" + "="*50 + "\n\n"
summary_text += f"τ (between-study SD):\n"
summary_text += f"  Mean: {tau_samples.mean():.2f}\n"
summary_text += f"  Median: {np.median(tau_samples):.2f}\n"
summary_text += f"  95% HDI: [{az.hdi(tau_samples, hdi_prob=0.95)[0]:.2f}, {az.hdi(tau_samples, hdi_prob=0.95)[1]:.2f}]\n\n"

summary_text += f"I² (% variance from heterogeneity):\n"
summary_text += f"  Mean: {I_sq_samples.mean():.1f}%\n"
summary_text += f"  Median: {np.median(I_sq_samples):.1f}%\n"
summary_text += f"  95% HDI: [{az.hdi(I_sq_samples, hdi_prob=0.95)[0]:.1f}%, {az.hdi(I_sq_samples, hdi_prob=0.95)[1]:.1f}%]\n\n"

summary_text += f"Probabilities:\n"
summary_text += f"  P(τ < 1): {(tau_samples < 1).mean():.3f}\n"
summary_text += f"  P(τ < 5): {(tau_samples < 5).mean():.3f}\n"
summary_text += f"  P(I² < 25%): {(I_sq_samples < 25).mean():.3f}\n\n"

summary_text += f"Interpretation:\n"
if I_sq_samples.mean() < 25:
    summary_text += f"  LOW heterogeneity detected\n"
    summary_text += f"  → Model 1 likely adequate\n"
    summary_text += f"  → Strong pooling toward μ"
elif I_sq_samples.mean() < 50:
    summary_text += f"  MODERATE heterogeneity\n"
    summary_text += f"  → Hierarchical model appropriate\n"
    summary_text += f"  → Partial pooling"
else:
    summary_text += f"  HIGH heterogeneity\n"
    summary_text += f"  → Hierarchical model necessary\n"
    summary_text += f"  → Limited pooling"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Heterogeneity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'heterogeneity_analysis.png', dpi=300, bbox_inches='tight')
print(f"  Saved: heterogeneity_analysis.png")
plt.close()

# Figure 8: Pairplot for key parameters
print("Creating pairplot...")

fig = plt.figure(figsize=(12, 12))
az.plot_pair(idata, var_names=['mu', 'tau'],
             kind='hexbin',
             marginals=True,
             show=False)
plt.suptitle('Joint Posterior: μ and τ', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pairplot_hyperparameters.png', dpi=300, bbox_inches='tight')
print(f"  Saved: pairplot_hyperparameters.png")
plt.close()

# Figure 9: ESS and R-hat summary plot
print("Creating convergence summary plot...")

summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ESS plot
ax = axes[0]
params = summary.index
ess_bulk = summary['ess_bulk'].values
ax.barh(range(len(params)), ess_bulk, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(400, color='red', linestyle='--', linewidth=2, label='Target ESS=400')
ax.set_yticks(range(len(params)))
ax.set_yticklabels(params, fontsize=9)
ax.set_xlabel('ESS Bulk', fontsize=12)
ax.set_title('Effective Sample Size', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# R-hat plot
ax = axes[1]
rhat = summary['r_hat'].values
ax.barh(range(len(params)), rhat, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(1.01, color='red', linestyle='--', linewidth=2, label='Target R-hat=1.01')
ax.axvline(1.0, color='green', linestyle=':', linewidth=1.5, label='Perfect convergence')
ax.set_yticks(range(len(params)))
ax.set_yticklabels(params, fontsize=9)
ax.set_xlabel('R-hat', fontsize=12)
ax.set_title('Gelman-Rubin Statistic', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([0.99, max(1.02, rhat.max() + 0.001)])

plt.suptitle('Convergence Metrics Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'convergence_summary.png', dpi=300, bbox_inches='tight')
print(f"  Saved: convergence_summary.png")
plt.close()

print("\n" + "="*70)
print("All diagnostic visualizations created!")
print("="*70)
print(f"\nPlots saved to: {PLOTS_DIR}")
print("\nVisualizations:")
print("  1. trace_plots_hyperparameters.png - μ and τ traces")
print("  2. rank_plots.png - MCMC mixing diagnostic")
print("  3. autocorrelation.png - Chain independence")
print("  4. trace_plots_theta.png - Study-specific effects")
print("  5. prior_posterior_comparison.png - Learning from data")
print("  6. forest_plot.png - Partial pooling visualization")
print("  7. heterogeneity_analysis.png - I² and τ interpretation")
print("  8. pairplot_hyperparameters.png - Joint posterior")
print("  9. convergence_summary.png - ESS and R-hat bars")
