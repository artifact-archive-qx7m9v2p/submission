"""
Create Comprehensive Diagnostic Visualizations
==============================================

This script creates convergence and posterior diagnostic plots.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('default')
az.style.use('arviz-darkgrid')

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

print("Loading InferenceData...")
idata = az.from_netcdf(DIAGNOSTICS_DIR / "posterior_inference.netcdf")

# Load diagnostics
with open(DIAGNOSTICS_DIR / "diagnostics.json", 'r') as f:
    diagnostics = json.load(f)

print("Creating diagnostic visualizations...\n")

# ============================================================================
# 1. CONVERGENCE OVERVIEW - Combined trace and rank plots
# ============================================================================
print("1. Creating convergence overview (trace + rank plots)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Convergence Diagnostics Overview', fontsize=16, fontweight='bold')

# Trace plot
ax_trace = axes[0, 0]
theta_samples = idata.posterior.theta
for chain in range(theta_samples.shape[0]):
    ax_trace.plot(theta_samples[chain, :], alpha=0.7, label=f'Chain {chain}')
ax_trace.set_xlabel('Iteration')
ax_trace.set_ylabel('θ')
ax_trace.set_title('Trace Plot (all chains)')
ax_trace.legend(loc='upper right', fontsize=8)
ax_trace.grid(True, alpha=0.3)

# Rank plot
ax_rank = axes[0, 1]
az.plot_rank(idata, var_names=['theta'], ax=ax_rank)
ax_rank.set_title('Rank Plot (uniformity check)')

# Autocorrelation plot
ax_acf = axes[1, 0]
az.plot_autocorr(idata, var_names=['theta'], combined=True, ax=ax_acf)
ax_acf.set_title('Autocorrelation Plot')

# ESS comparison
ax_ess = axes[1, 1]
ess_bulk = diagnostics['convergence']['ess_bulk']
ess_tail = diagnostics['convergence']['ess_tail']
total_samples = 4 * 2000  # 4 chains, 2000 draws each
ax_ess.bar(['Bulk ESS', 'Tail ESS', 'Total Samples'],
           [ess_bulk, ess_tail, total_samples],
           color=['#2E86AB', '#A23B72', '#F18F01'])
ax_ess.axhline(y=400, color='red', linestyle='--', label='Target (400)', linewidth=2)
ax_ess.set_ylabel('Number of Samples')
ax_ess.set_title('Effective Sample Size')
ax_ess.legend()
ax_ess.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_overview.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: convergence_overview.png")

# ============================================================================
# 2. POSTERIOR DISTRIBUTION WITH CREDIBLE INTERVALS
# ============================================================================
print("2. Creating posterior distribution plot...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Get posterior samples
theta_samples = idata.posterior.theta.values.flatten()

# Plot posterior density
az.plot_posterior(idata, var_names=['theta'],
                  hdi_prob=0.95,
                  point_estimate='mean',
                  ax=ax)

# Add analytical posterior overlay
analytical_mean = diagnostics['analytical_validation']['analytical_mean']
analytical_sd = diagnostics['analytical_validation']['analytical_sd']
x_range = np.linspace(theta_samples.min(), theta_samples.max(), 200)
analytical_pdf = stats.norm.pdf(x_range, analytical_mean, analytical_sd)

# Normalize to match KDE height approximately
kde_height = ax.get_ylim()[1]
pdf_height = analytical_pdf.max()
scaling = kde_height / pdf_height * 0.8
analytical_pdf_scaled = analytical_pdf * scaling

ax.plot(x_range, analytical_pdf_scaled, 'r--', linewidth=2,
        label=f'Analytical N({analytical_mean:.3f}, {analytical_sd:.3f}²)', alpha=0.7)
ax.legend(fontsize=10)

ax.set_title('Posterior Distribution of θ (Fixed Effect)\nwith Analytical Validation',
             fontsize=14, fontweight='bold')
ax.set_xlabel('θ (Fixed Effect Size)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: posterior_distribution.png")

# ============================================================================
# 3. PRIOR VS POSTERIOR COMPARISON
# ============================================================================
print("3. Creating prior vs posterior comparison...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Prior
prior_mean = 0.0
prior_sd = 20.0
x_prior = np.linspace(-60, 80, 500)
prior_pdf = stats.norm.pdf(x_prior, prior_mean, prior_sd)
ax.plot(x_prior, prior_pdf, 'b--', linewidth=2, label=f'Prior: N(0, 20²)', alpha=0.7)

# Posterior
x_post = np.linspace(theta_samples.min() - 5, theta_samples.max() + 5, 500)
posterior_pdf = stats.norm.pdf(x_post, analytical_mean, analytical_sd)
ax.plot(x_post, posterior_pdf, 'r-', linewidth=2,
        label=f'Posterior: N({analytical_mean:.2f}, {analytical_sd:.2f}²)', alpha=0.9)

# Add data points (scaled to fit)
data = pd.read_csv("/workspace/data/data.csv")
y_data = data['y'].values
y_height = ax.get_ylim()[1] * 0.05
ax.scatter(y_data, [y_height] * len(y_data), color='green', s=100,
           marker='o', label='Observed Data', zorder=5, alpha=0.7)

ax.set_xlabel('θ (Effect Size)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior vs Posterior Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_vs_posterior.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: prior_vs_posterior.png")

# ============================================================================
# 4. ENERGY PLOT (BFMI diagnostic)
# ============================================================================
print("4. Creating energy plot...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
az.plot_energy(idata, ax=ax)
ax.set_title('Energy Plot (BFMI Diagnostic)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "energy_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: energy_diagnostic.png")

# ============================================================================
# 5. FOREST PLOT WITH UNCERTAINTY
# ============================================================================
print("5. Creating forest plot...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Load data
data = pd.read_csv("/workspace/data/data.csv")
y_data = data['y'].values
sigma_data = data['sigma'].values

# Posterior for theta
theta_mean = diagnostics['posterior_summary']['mean']
theta_lower = diagnostics['posterior_summary']['hdi_95_lower']
theta_upper = diagnostics['posterior_summary']['hdi_95_upper']

# Plot individual study estimates
n_studies = len(y_data)
positions = np.arange(n_studies)

ax.errorbar(y_data, positions, xerr=1.96*sigma_data,
            fmt='o', color='steelblue', capsize=5, capthick=2,
            markersize=8, label='Study Estimates (95% CI)', alpha=0.7)

# Plot pooled posterior estimate
pooled_pos = n_studies + 0.5
ax.errorbar([theta_mean], [pooled_pos],
            xerr=[[theta_mean - theta_lower], [theta_upper - theta_mean]],
            fmt='D', color='red', capsize=8, capthick=3,
            markersize=12, label='Pooled Posterior (95% HDI)', alpha=0.9, linewidth=2)

# Add vertical line at zero
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Formatting
study_labels = [f'Study {i+1}' for i in range(n_studies)] + ['Pooled']
ax.set_yticks(list(positions) + [pooled_pos])
ax.set_yticklabels(study_labels)
ax.set_xlabel('Effect Size', fontsize=12)
ax.set_title('Forest Plot: Individual Studies vs Pooled Estimate',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: forest_plot.png")

# ============================================================================
# 6. POSTERIOR PREDICTIVE DISTRIBUTIONS FOR EACH OBSERVATION
# ============================================================================
print("6. Creating posterior predictive check...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# For each study, show posterior predictive distribution
for i in range(n_studies):
    ax = axes[i]

    # Posterior predictive: y_new ~ N(theta, sigma_i)
    # Sample from posterior of theta and generate predictions
    y_pred_samples = np.random.normal(theta_samples, sigma_data[i], size=len(theta_samples))

    # Plot histogram
    ax.hist(y_pred_samples, bins=50, density=True, alpha=0.6,
            color='steelblue', label='Posterior Predictive')

    # Add observed value
    ax.axvline(y_data[i], color='red', linewidth=2,
               label=f'Observed: {y_data[i]:.1f}')

    # Add theoretical posterior predictive (analytical)
    x_range = np.linspace(y_pred_samples.min(), y_pred_samples.max(), 200)
    pp_pdf = stats.norm.pdf(x_range, analytical_mean,
                            np.sqrt(analytical_sd**2 + sigma_data[i]**2))
    ax.plot(x_range, pp_pdf, 'k--', linewidth=2, alpha=0.7,
            label='Analytical')

    ax.set_title(f'Study {i+1}\n(σ={sigma_data[i]})', fontsize=10)
    ax.set_xlabel('Effect Size')
    if i % 4 == 0:
        ax.set_ylabel('Density')
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Posterior Predictive Distributions by Study',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_predictive.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: posterior_predictive.png")

# ============================================================================
# 7. QUANTILE-QUANTILE PLOT (MCMC vs Analytical)
# ============================================================================
print("7. Creating Q-Q plot (MCMC vs Analytical)...")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Get quantiles from MCMC samples
mcmc_quantiles = np.percentile(theta_samples, np.linspace(0.1, 99.9, 100))

# Get corresponding quantiles from analytical distribution
analytical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, 100),
                                      analytical_mean, analytical_sd)

# Q-Q plot
ax.scatter(analytical_quantiles, mcmc_quantiles, alpha=0.6, s=30)
ax.plot([analytical_quantiles.min(), analytical_quantiles.max()],
        [analytical_quantiles.min(), analytical_quantiles.max()],
        'r--', linewidth=2, label='Perfect Agreement')

ax.set_xlabel('Analytical Quantiles', fontsize=12)
ax.set_ylabel('MCMC Quantiles', fontsize=12)
ax.set_title('Q-Q Plot: MCMC vs Analytical Posterior', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add text with correlation
correlation = np.corrcoef(analytical_quantiles, mcmc_quantiles)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {correlation:.6f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "qq_plot_validation.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: qq_plot_validation.png")

print("\n" + "=" * 80)
print("All diagnostic visualizations created successfully!")
print("=" * 80)
