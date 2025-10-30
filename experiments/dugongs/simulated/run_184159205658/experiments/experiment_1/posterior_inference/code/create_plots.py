"""
Create diagnostic plots for Bayesian Logarithmic Regression
Experiment 1 - Posterior Inference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pathlib import Path
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path('/workspace/experiments/experiment_1/posterior_inference')
DATA_PATH = Path('/workspace/data/data.csv')
DIAGNOSTICS_DIR = BASE_DIR / 'diagnostics'
PLOTS_DIR = BASE_DIR / 'plots'

print("="*80)
print("CREATING DIAGNOSTIC PLOTS")
print("="*80)

# Load data and results
print("\n[1] Loading data and results...")
data = pd.read_csv(DATA_PATH)
idata = az.from_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')
residuals_df = pd.read_csv(DIAGNOSTICS_DIR / 'residuals.csv')
loo_df = pd.read_csv(DIAGNOSTICS_DIR / 'loo_results.csv')

# Extract posterior samples
posterior = idata.posterior
beta0_samples = posterior['beta0'].values.flatten()
beta1_samples = posterior['beta1'].values.flatten()
sigma_samples = posterior['sigma'].values.flatten()
mu_samples = posterior['mu'].values  # (chains, draws, N)

print(f"    Data: {len(data)} observations")
print(f"    Posterior samples: {len(beta0_samples)}")

# ============================================================================
# PLOT 1: Convergence Overview (Trace + Rank Plots)
# ============================================================================
print("\n[2] Creating convergence overview plot...")

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Trace plots
ax1 = fig.add_subplot(gs[0, 0])
az.plot_trace(idata, var_names=['beta0'], axes=[[ax1, None]], legend=False)
ax1.set_title('β₀ (Intercept) - Trace Plot', fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(gs[1, 0])
az.plot_trace(idata, var_names=['beta1'], axes=[[ax2, None]], legend=False)
ax2.set_title('β₁ (Log Slope) - Trace Plot', fontsize=12, fontweight='bold')

ax3 = fig.add_subplot(gs[2, 0])
az.plot_trace(idata, var_names=['sigma'], axes=[[ax3, None]], legend=False)
ax3.set_title('σ (Error SD) - Trace Plot', fontsize=12, fontweight='bold')

# Rank plots
ax4 = fig.add_subplot(gs[0, 1])
az.plot_rank(idata, var_names=['beta0'], ax=ax4)
ax4.set_title('β₀ - Rank Plot', fontsize=12, fontweight='bold')

ax5 = fig.add_subplot(gs[1, 1])
az.plot_rank(idata, var_names=['beta1'], ax=ax5)
ax5.set_title('β₁ - Rank Plot', fontsize=12, fontweight='bold')

ax6 = fig.add_subplot(gs[2, 1])
az.plot_rank(idata, var_names=['sigma'], ax=ax6)
ax6.set_title('σ - Rank Plot', fontsize=12, fontweight='bold')

fig.suptitle('Convergence Diagnostics: Trace and Rank Plots',
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / 'convergence_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'convergence_overview.png'}")

# ============================================================================
# PLOT 2: Posterior Distributions with Prior Overlays
# ============================================================================
print("\n[3] Creating posterior distributions with priors...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Beta0
ax = axes[0]
az.plot_dist(beta0_samples, ax=ax, color='steelblue', label='Posterior',
             kind='kde', fill_kwargs={'alpha': 0.3})
# Prior overlay
x_beta0 = np.linspace(beta0_samples.min(), beta0_samples.max(), 200)
prior_beta0 = stats.norm.pdf(x_beta0, 1.73, 0.5)
ax.plot(x_beta0, prior_beta0, 'r--', linewidth=2, label='Prior: N(1.73, 0.5)')
ax.axvline(beta0_samples.mean(), color='steelblue', linestyle='--',
           linewidth=2, label=f'Post. Mean: {beta0_samples.mean():.3f}')
ax.set_xlabel('β₀ (Intercept)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('β₀ Posterior', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Beta1
ax = axes[1]
az.plot_dist(beta1_samples, ax=ax, color='darkorange', label='Posterior',
             kind='kde', fill_kwargs={'alpha': 0.3})
# Prior overlay
x_beta1 = np.linspace(max(0, beta1_samples.min()), beta1_samples.max(), 200)
prior_beta1 = stats.norm.pdf(x_beta1, 0.28, 0.15)
ax.plot(x_beta1, prior_beta1, 'r--', linewidth=2, label='Prior: N(0.28, 0.15)')
ax.axvline(beta1_samples.mean(), color='darkorange', linestyle='--',
           linewidth=2, label=f'Post. Mean: {beta1_samples.mean():.3f}')
ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('β₁ (Log Slope)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('β₁ Posterior', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Sigma
ax = axes[2]
az.plot_dist(sigma_samples, ax=ax, color='seagreen', label='Posterior',
             kind='kde', fill_kwargs={'alpha': 0.3})
# Prior overlay
x_sigma = np.linspace(0, max(sigma_samples.max(), 1), 200)
prior_sigma = stats.expon.pdf(x_sigma, scale=1/5)
ax.plot(x_sigma, prior_sigma, 'r--', linewidth=2, label='Prior: Exp(5)')
ax.axvline(sigma_samples.mean(), color='seagreen', linestyle='--',
           linewidth=2, label=f'Post. Mean: {sigma_samples.mean():.3f}')
ax.set_xlabel('σ (Error SD)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('σ Posterior', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Posterior Distributions vs Priors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'posterior_distributions.png'}")

# ============================================================================
# PLOT 3: Model Fit - Observed vs Fitted with Uncertainty
# ============================================================================
print("\n[4] Creating model fit plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create prediction grid
x_grid = np.linspace(0.5, 35, 200)
x_max_obs = data['x'].max()

# Generate predictions on grid
mu_grid = np.zeros((len(beta0_samples), len(x_grid)))
for i in range(len(beta0_samples)):
    mu_grid[i, :] = beta0_samples[i] + beta1_samples[i] * np.log(x_grid)

mu_grid_mean = mu_grid.mean(axis=0)
mu_grid_lower = np.percentile(mu_grid, 2.5, axis=0)
mu_grid_upper = np.percentile(mu_grid, 97.5, axis=0)

# Plot uncertainty bands
in_sample = x_grid <= x_max_obs
extrapolation = x_grid > x_max_obs

# In-sample region
ax.fill_between(x_grid[in_sample], mu_grid_lower[in_sample], mu_grid_upper[in_sample],
                alpha=0.3, color='steelblue', label='95% Credible Interval (In-Sample)')
ax.plot(x_grid[in_sample], mu_grid_mean[in_sample], 'b-', linewidth=2,
        label='Posterior Mean')

# Extrapolation region
ax.fill_between(x_grid[extrapolation], mu_grid_lower[extrapolation],
                mu_grid_upper[extrapolation],
                alpha=0.2, color='orange', label='95% CI (Extrapolation)')
ax.plot(x_grid[extrapolation], mu_grid_mean[extrapolation], 'orange',
        linewidth=2, linestyle='--', label='Extrapolation')

# Observed data
ax.scatter(data['x'], data['Y'], s=80, c='red', edgecolors='black',
           linewidth=1, alpha=0.7, label='Observed Data', zorder=5)

# Add vertical line at max observed x
ax.axvline(x_max_obs, color='gray', linestyle='--', linewidth=1.5,
           alpha=0.5, label=f'Max Observed x = {x_max_obs:.1f}')

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_title('Model Fit: Y = β₀ + β₁·log(x)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'model_fit.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'model_fit.png'}")

# ============================================================================
# PLOT 4: Residual Diagnostics (4-panel)
# ============================================================================
print("\n[5] Creating residual diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(residuals_df['mu_mean'], residuals_df['residual'],
           s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted Values', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 2: Residuals vs x
ax = axes[0, 1]
ax.scatter(residuals_df['x'], residuals_df['residual'],
           s=80, alpha=0.6, edgecolors='black', linewidth=0.5, color='darkorange')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residuals vs Predictor', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 3: Q-Q plot
ax = axes[1, 0]
stats.probplot(residuals_df['residual'], dist="norm", plot=ax)
ax.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 4: Histogram of residuals
ax = axes[1, 1]
ax.hist(residuals_df['residual'], bins=15, alpha=0.7, color='seagreen',
        edgecolor='black', density=True)
# Overlay normal distribution
x_norm = np.linspace(residuals_df['residual'].min(),
                     residuals_df['residual'].max(), 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, 0, residuals_df['residual'].std()),
        'r-', linewidth=2, label='Normal(0, σ)')
ax.set_xlabel('Residuals', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'residual_diagnostics.png'}")

# ============================================================================
# PLOT 5: Posterior Predictive Distribution
# ============================================================================
print("\n[6] Creating posterior predictive plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Generate posterior predictive samples for grid
y_pred_samples = np.zeros((min(500, len(beta0_samples)), len(x_grid)))
np.random.seed(42)
sample_indices = np.random.choice(len(beta0_samples),
                                  size=min(500, len(beta0_samples)),
                                  replace=False)

for idx, i in enumerate(sample_indices):
    mu_pred = beta0_samples[i] + beta1_samples[i] * np.log(x_grid)
    y_pred_samples[idx, :] = np.random.normal(mu_pred, sigma_samples[i])

y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)
y_pred_mean = y_pred_samples.mean(axis=0)

# Plot predictive uncertainty
ax.fill_between(x_grid[in_sample], y_pred_lower[in_sample], y_pred_upper[in_sample],
                alpha=0.2, color='steelblue',
                label='95% Posterior Predictive Interval (In-Sample)')
ax.fill_between(x_grid[extrapolation], y_pred_lower[extrapolation],
                y_pred_upper[extrapolation],
                alpha=0.15, color='orange',
                label='95% Predictive Interval (Extrapolation)')

# Plot mean function
ax.plot(x_grid[in_sample], mu_grid_mean[in_sample], 'b-', linewidth=2,
        label='E[Y|x] (Posterior Mean)')
ax.plot(x_grid[extrapolation], mu_grid_mean[extrapolation], 'orange',
        linewidth=2, linestyle='--')

# Observed data
ax.scatter(data['x'], data['Y'], s=80, c='red', edgecolors='black',
           linewidth=1, alpha=0.7, label='Observed Data', zorder=5)

ax.axvline(x_max_obs, color='gray', linestyle='--', linewidth=1.5,
           alpha=0.5, label=f'Max Observed x = {x_max_obs:.1f}')

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_title('Posterior Predictive Distribution', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_predictive.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'posterior_predictive.png'}")

# ============================================================================
# PLOT 6: LOO-PIT and Pareto k Diagnostics
# ============================================================================
print("\n[7] Creating LOO diagnostics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Pareto k values
ax = axes[0]
pareto_k = loo_df['pareto_k'].values
colors = ['green' if k < 0.5 else 'yellow' if k < 0.7 else 'red' for k in pareto_k]
ax.scatter(loo_df['x'], pareto_k, c=colors, s=80, alpha=0.7,
           edgecolors='black', linewidth=0.5)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='k = 0.5 (threshold)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='k = 0.7 (problematic)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Pareto k', fontsize=12)
ax.set_title('LOO-CV Pareto k Diagnostics', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: LOO-PIT plot
ax = axes[1]
az.plot_loo_pit(idata, y='Y', ax=ax, ecdf=True)
ax.set_title('LOO-PIT ECDF', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

fig.suptitle('LOO-CV Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loo_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'loo_diagnostics.png'}")

# ============================================================================
# PLOT 7: Parameter Correlations
# ============================================================================
print("\n[8] Creating parameter correlation plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Joint plot: beta0 vs beta1
ax = axes[0]
h = ax.hexbin(beta0_samples, beta1_samples, gridsize=30, cmap='Blues', mincnt=1)
ax.set_xlabel('β₀ (Intercept)', fontsize=12)
ax.set_ylabel('β₁ (Log Slope)', fontsize=12)
ax.set_title('Parameter Correlation: β₀ vs β₁', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(h, ax=ax)
cbar.set_label('Density', fontsize=10)

# Correlation with sigma
ax = axes[1]
h = ax.hexbin(beta1_samples, sigma_samples, gridsize=30, cmap='Oranges', mincnt=1)
ax.set_xlabel('β₁ (Log Slope)', fontsize=12)
ax.set_ylabel('σ (Error SD)', fontsize=12)
ax.set_title('Parameter Correlation: β₁ vs σ', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(h, ax=ax)
cbar.set_label('Density', fontsize=10)

fig.suptitle('Parameter Posterior Correlations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOTS_DIR / 'parameter_correlations.png'}")

print("\n" + "="*80)
print("ALL PLOTS CREATED SUCCESSFULLY")
print("="*80)
print(f"\nPlots saved to: {PLOTS_DIR}/")
print("\nGenerated plots:")
print("  1. convergence_overview.png - Trace and rank plots")
print("  2. posterior_distributions.png - Posteriors vs priors")
print("  3. model_fit.png - Fitted values with uncertainty")
print("  4. residual_diagnostics.png - Residual analysis (4-panel)")
print("  5. posterior_predictive.png - Predictive distribution")
print("  6. loo_diagnostics.png - LOO-CV diagnostics")
print("  7. parameter_correlations.png - Parameter correlations")
