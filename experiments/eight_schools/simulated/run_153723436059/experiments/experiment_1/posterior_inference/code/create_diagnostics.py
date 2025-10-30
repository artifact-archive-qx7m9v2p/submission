"""
Create comprehensive diagnostic visualizations for hierarchical model
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
OUTPUT_DIR = Path('/workspace/experiments/experiment_1/posterior_inference')
DIAGNOSTICS_DIR = OUTPUT_DIR / 'diagnostics'
PLOTS_DIR = OUTPUT_DIR / 'plots'
DATA_PATH = Path('/workspace/data/data.csv')

# Load posterior samples
print("Loading posterior samples...")
trace = az.from_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')

# Load data
df = pd.read_csv(DATA_PATH)
y_obs = df['effect'].values
sigma_obs = df['sigma'].values
school_names = [f"School {i+1}" for i in range(len(df))]

print(f"Loaded {len(trace.posterior.chain)} chains with {len(trace.posterior.draw)} draws each")

# Set plotting style
plt.style.use('default')
az.style.use('arviz-whitegrid')

print("\n" + "="*80)
print("CREATING DIAGNOSTIC VISUALIZATIONS")
print("="*80)

# ============================================================================
# 1. CONVERGENCE OVERVIEW - Trace plots for key parameters
# ============================================================================
print("\n[1] Creating trace plots (convergence overview)...")

# Create separate trace plots for key parameters
axes = az.plot_trace(trace, var_names=['mu', 'tau'], compact=False, figsize=(14, 8))
plt.gcf().suptitle('Trace Plots: Hyperparameters (mu, tau)', fontsize=14, fontweight='bold', y=1.00)
plt.savefig(PLOTS_DIR / 'trace_hyperparameters.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'trace_hyperparameters.png'}")
plt.close()

# Trace plots for all theta
axes = az.plot_trace(trace, var_names=['theta'], compact=True, figsize=(14, 12))
plt.gcf().suptitle('Trace Plots: School Effects (theta)', fontsize=14, fontweight='bold', y=1.00)
plt.savefig(PLOTS_DIR / 'trace_school_effects.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'trace_school_effects.png'}")
plt.close()

# ============================================================================
# 2. RANK PLOTS - Check for uniformity (good mixing)
# ============================================================================
print("\n[2] Creating rank plots (chain mixing)...")

fig = plt.figure(figsize=(12, 8))
az.plot_rank(trace, var_names=['mu', 'tau', 'theta'], kind='bars')
plt.suptitle('Rank Plots: Chain Mixing Diagnostic', fontsize=14, fontweight='bold', y=1.00)
plt.savefig(PLOTS_DIR / 'rank_plots.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'rank_plots.png'}")
plt.close()

# ============================================================================
# 3. POSTERIOR DISTRIBUTIONS - Hyperparameters
# ============================================================================
print("\n[3] Creating posterior distributions plot...")

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
az.plot_posterior(trace, var_names=['mu', 'tau'], ax=ax, textsize=10)
plt.suptitle('Posterior Distributions: Hyperparameters', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_hyperparameters.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'posterior_hyperparameters.png'}")
plt.close()

# Forest plot for all theta values
fig, ax = plt.subplots(figsize=(10, 8))
az.plot_forest(trace, var_names=['theta'], combined=True,
               hdi_prob=0.95, ess=True, r_hat=True, figsize=(10, 8))
plt.title('Posterior Distributions: School Effects (theta)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'forest_school_effects.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'forest_school_effects.png'}")
plt.close()

# ============================================================================
# 4. FOREST PLOT - Compare observed data to posterior theta
# ============================================================================
print("\n[4] Creating forest plot with observations...")

fig, ax = plt.subplots(figsize=(10, 8))

# Extract posterior samples for theta
theta_samples = trace.posterior.theta.values.reshape(-1, 8)  # Flatten chains and draws

# Calculate posterior statistics
theta_mean = theta_samples.mean(axis=0)
theta_hdi = np.percentile(theta_samples, [2.5, 97.5], axis=0)

# Plot
y_pos = np.arange(8)

# Observed data with error bars
ax.errorbar(y_obs, y_pos - 0.1, xerr=1.96*sigma_obs, fmt='o', color='red',
            label='Observed (y ± 1.96σ)', markersize=8, capsize=5, alpha=0.7)

# Posterior estimates
ax.plot(theta_mean, y_pos + 0.1, 'o', color='blue', markersize=8,
        label='Posterior mean (θ)', alpha=0.7)
for i in range(8):
    ax.plot([theta_hdi[0, i], theta_hdi[1, i]], [y_pos[i] + 0.1, y_pos[i] + 0.1],
            '-', color='blue', linewidth=2, alpha=0.5)

# Add population mean
mu_mean = trace.posterior.mu.values.mean()
mu_hdi = np.percentile(trace.posterior.mu.values, [2.5, 97.5])
ax.axvline(mu_mean, color='green', linestyle='--', linewidth=2, label='Population mean (μ)', alpha=0.7)
ax.axvspan(mu_hdi[0], mu_hdi[1], color='green', alpha=0.1)

ax.set_yticks(y_pos)
ax.set_yticklabels(school_names)
ax.set_xlabel('Effect Size', fontsize=11)
ax.set_title('Forest Plot: Observed vs Posterior School Effects', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'forest_plot_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'forest_plot_comparison.png'}")
plt.close()

# ============================================================================
# 5. SHRINKAGE PLOT - Show pooling effect
# ============================================================================
print("\n[5] Creating shrinkage plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Draw lines from observed to posterior mean
for i in range(8):
    ax.plot([y_obs[i], theta_mean[i]], [i, i], 'k-', alpha=0.3, linewidth=1.5)
    if abs(theta_mean[i] - y_obs[i]) > 0.5:  # Only add arrow if shrinkage is visible
        dx = theta_mean[i] - y_obs[i]
        ax.arrow(y_obs[i], i, 0.9*dx, 0,
                 head_width=0.15, head_length=abs(0.1*dx), fc='gray', ec='gray', alpha=0.5)

# Plot points
ax.plot(y_obs, np.arange(8), 'ro', markersize=10, label='Observed effect', alpha=0.7, zorder=5)
ax.plot(theta_mean, np.arange(8), 'bo', markersize=10, label='Posterior mean', alpha=0.7, zorder=5)

# Add population mean
ax.axvline(mu_mean, color='green', linestyle='--', linewidth=2, label='Population mean (μ)')

# Formatting
ax.set_yticks(np.arange(8))
ax.set_yticklabels(school_names)
ax.set_xlabel('Effect Size', fontsize=11)
ax.set_title('Shrinkage: Observed → Posterior (Partial Pooling)', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Add text showing shrinkage amount
for i in range(8):
    shrinkage_pct = 100 * (y_obs[i] - theta_mean[i]) / (y_obs[i] - mu_mean) if abs(y_obs[i] - mu_mean) > 0.1 else 0
    if abs(shrinkage_pct) > 5:  # Only show if substantial
        mid_x = (y_obs[i] + theta_mean[i]) / 2
        ax.text(mid_x, i + 0.25, f'{shrinkage_pct:.0f}%', fontsize=8, ha='center', color='gray')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'shrinkage_plot.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'shrinkage_plot.png'}")
plt.close()

# ============================================================================
# 6. PAIRS PLOT - Check for funnel geometry
# ============================================================================
print("\n[6] Creating pairs plot (funnel check)...")

fig = az.plot_pair(
    trace,
    var_names=['mu', 'tau'],
    coords={'theta_dim_0': [0, 4]},  # Include theta[1] and theta[5] (extreme values)
    kind='hexbin',
    marginals=True,
    figsize=(12, 10)
)
plt.suptitle('Pairs Plot: Check for Funnel Geometry', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pairs_funnel_check.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'pairs_funnel_check.png'}")
plt.close()

# ============================================================================
# 7. PRIOR-POSTERIOR COMPARISON
# ============================================================================
print("\n[7] Creating prior-posterior comparison...")

# Generate prior samples
np.random.seed(42)
n_prior = 5000
mu_prior = np.random.normal(0, 50, n_prior)
# HalfCauchy implemented as absolute value of Cauchy
tau_prior = np.abs(np.random.standard_cauchy(n_prior) * 25)
tau_prior = tau_prior[tau_prior < 100][:n_prior]  # Truncate extreme values for visualization

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# mu
axes[0].hist(mu_prior, bins=50, density=True, alpha=0.5, color='gray', label='Prior')
mu_post = trace.posterior.mu.values.flatten()
axes[0].hist(mu_post, bins=50, density=True, alpha=0.7, color='blue', label='Posterior')
axes[0].axvline(mu_post.mean(), color='blue', linestyle='--', linewidth=2, label=f'Posterior mean: {mu_post.mean():.2f}')
axes[0].set_xlabel('mu (Population Mean Effect)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Prior vs Posterior: mu', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].set_xlim(-50, 50)

# tau
axes[1].hist(tau_prior, bins=50, density=True, alpha=0.5, color='gray', label='Prior')
tau_post = trace.posterior.tau.values.flatten()
axes[1].hist(tau_post, bins=50, density=True, alpha=0.7, color='blue', label='Posterior')
axes[1].axvline(tau_post.mean(), color='blue', linestyle='--', linewidth=2, label=f'Posterior mean: {tau_post.mean():.2f}')
axes[1].set_xlabel('tau (Between-School SD)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('Prior vs Posterior: tau', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].set_xlim(0, 40)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_posterior_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'prior_posterior_comparison.png'}")
plt.close()

# ============================================================================
# 8. ENERGY DIAGNOSTIC PLOT
# ============================================================================
print("\n[8] Creating energy diagnostic plot...")

fig = az.plot_energy(trace, figsize=(10, 6))
plt.suptitle('Energy Diagnostic: Marginal vs Conditional', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'energy_diagnostic.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'energy_diagnostic.png'}")
plt.close()

print("\n" + "="*80)
print("All diagnostic plots created successfully!")
print("="*80)
print(f"\nPlots saved to: {PLOTS_DIR}")
print("\nDiagnostic visualizations:")
print("  1. trace_hyperparameters.png - Trace plots for mu, tau")
print("  2. trace_school_effects.png - Trace plots for all theta")
print("  3. rank_plots.png - Chain mixing uniformity")
print("  4. posterior_hyperparameters.png - Posterior densities for mu, tau")
print("  5. forest_school_effects.png - Forest plot for all theta")
print("  6. forest_plot_comparison.png - Observed vs posterior estimates")
print("  7. shrinkage_plot.png - Partial pooling visualization")
print("  8. pairs_funnel_check.png - Funnel geometry check")
print("  9. prior_posterior_comparison.png - Learning from data")
print(" 10. energy_diagnostic.png - HMC energy transitions")
