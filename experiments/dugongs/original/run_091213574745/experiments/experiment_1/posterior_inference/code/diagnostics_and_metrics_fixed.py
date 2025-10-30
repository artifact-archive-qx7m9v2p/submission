"""
Comprehensive diagnostics, visualizations, and model fit metrics
for the Logarithmic Model with Normal Likelihood
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Setup
sns.set_style('whitegrid')
np.random.seed(42)

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Load data and inference results
print("="*80)
print("LOGARITHMIC MODEL - DIAGNOSTICS AND METRICS")
print("="*80)

print("\n1. Loading data and posterior samples...")
data = pd.read_csv(DATA_PATH)
x = data['x'].values
Y_obs = data['Y'].values
N = len(data)

idata = az.from_netcdf(DIAG_DIR / "posterior_inference.netcdf")

# Load convergence metrics
with open(DIAG_DIR / "convergence_metrics.json", 'r') as f:
    conv_metrics = json.load(f)

print(f"   - Data: n={N}")
print(f"   - Posterior samples loaded")
print(f"   - Convergence status: {conv_metrics['convergence_pass']}")

# ============================================================================
# CONVERGENCE DIAGNOSTICS VISUALIZATIONS
# ============================================================================
print("\n2. Creating convergence diagnostic plots...")

# Trace plots (compact overview)
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
az.plot_trace(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma'],
    compact=False,
    axes=axes
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Trace plots saved")

# Rank plots for detecting convergence issues
az.plot_rank(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma']
)
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Rank plots saved")

# Autocorrelation plot
az.plot_autocorr(idata, var_names=['beta_0', 'beta_1', 'sigma'], combined=True)
plt.savefig(PLOTS_DIR / "autocorrelation.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Autocorrelation plots saved")

# ============================================================================
# POSTERIOR DISTRIBUTIONS
# ============================================================================
print("\n3. Creating posterior distribution plots...")

# Posterior distributions with prior overlays
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Beta_0
post_beta0 = idata.posterior['beta_0'].values.flatten()
axes[0].hist(post_beta0, bins=50, density=True, alpha=0.6, color='steelblue', label='Posterior')
x_range = np.linspace(post_beta0.min(), post_beta0.max(), 100)
prior_beta0 = stats.norm.pdf(x_range, 2.3, 0.3)
axes[0].plot(x_range, prior_beta0, 'r--', linewidth=2, label='Prior')
axes[0].axvline(post_beta0.mean(), color='black', linestyle='-', linewidth=2, label=f'Mean: {post_beta0.mean():.3f}')
axes[0].set_xlabel('β₀ (Intercept)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_title('β₀: Posterior vs Prior')

# Beta_1
post_beta1 = idata.posterior['beta_1'].values.flatten()
axes[1].hist(post_beta1, bins=50, density=True, alpha=0.6, color='steelblue', label='Posterior')
x_range = np.linspace(post_beta1.min(), post_beta1.max(), 100)
prior_beta1 = stats.norm.pdf(x_range, 0.29, 0.15)
axes[1].plot(x_range, prior_beta1, 'r--', linewidth=2, label='Prior')
axes[1].axvline(post_beta1.mean(), color='black', linestyle='-', linewidth=2, label=f'Mean: {post_beta1.mean():.3f}')
axes[1].set_xlabel('β₁ (Log slope)')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].set_title('β₁: Posterior vs Prior')

# Sigma
post_sigma = idata.posterior['sigma'].values.flatten()
axes[2].hist(post_sigma, bins=50, density=True, alpha=0.6, color='steelblue', label='Posterior')
x_range = np.linspace(0.001, post_sigma.max(), 100)
prior_sigma = 10 * np.exp(-10 * x_range)
axes[2].plot(x_range, prior_sigma, 'r--', linewidth=2, label='Prior')
axes[2].axvline(post_sigma.mean(), color='black', linestyle='-', linewidth=2, label=f'Mean: {post_sigma.mean():.3f}')
axes[2].set_xlabel('σ (Residual SD)')
axes[2].set_ylabel('Density')
axes[2].legend()
axes[2].set_title('σ: Posterior vs Prior')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_vs_prior.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Posterior vs Prior plots saved")

# Pairs plot (parameter correlations)
az.plot_pair(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma'],
    kind='kde',
    marginals=True,
    figsize=(10, 10)
)
plt.savefig(PLOTS_DIR / "pairs_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Pairs plot saved")

# ============================================================================
# MODEL FIT VISUALIZATIONS
# ============================================================================
print("\n4. Creating model fit visualizations...")

# Get posterior predictive mean and credible intervals
y_pred = idata.posterior['y_pred'].values  # Shape: (chains, draws, obs)
y_pred_mean = y_pred.mean(axis=(0, 1))
y_pred_lower = np.percentile(y_pred, 2.5, axis=(0, 1))
y_pred_upper = np.percentile(y_pred, 97.5, axis=(0, 1))

# Sort for plotting
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
Y_sorted = Y_obs[sort_idx]
y_pred_mean_sorted = y_pred_mean[sort_idx]
y_pred_lower_sorted = y_pred_lower[sort_idx]
y_pred_upper_sorted = y_pred_upper[sort_idx]

# Fitted curve with uncertainty
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, Y_obs, color='black', s=50, alpha=0.6, label='Observed data', zorder=3)
ax.plot(x_sorted, y_pred_mean_sorted, 'b-', linewidth=2, label='Posterior mean', zorder=2)
ax.fill_between(x_sorted, y_pred_lower_sorted, y_pred_upper_sorted,
                alpha=0.3, color='blue', label='95% Credible Interval', zorder=1)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Logarithmic Model: Fitted Curve with Uncertainty', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fitted_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Fitted curve saved")

# Residuals plot
residuals = Y_obs - y_pred_mean

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs fitted
axes[0, 0].scatter(y_pred_mean, residuals, alpha=0.6, color='steelblue')
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs x
axes[0, 1].scatter(x, residuals, alpha=0.6, color='steelblue')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Predictor')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)')
axes[1, 0].grid(True, alpha=0.3)

# Histogram of residuals
axes[1, 1].hist(residuals, bins=15, density=True, alpha=0.6, color='steelblue', edgecolor='black')
x_range = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 1].plot(x_range, stats.norm.pdf(x_range, 0, residuals.std()),
                'r--', linewidth=2, label='Normal')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "residuals_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Residuals diagnostics saved")

# ============================================================================
# LOO-CV DIAGNOSTICS
# ============================================================================
print("\n5. Computing LOO-CV diagnostics...")

# Compute LOO
loo = az.loo(idata, pointwise=True)

# LOO-PIT plot
az.plot_loo_pit(idata, y='Y', figsize=(10, 6))
plt.savefig(PLOTS_DIR / "loo_pit.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - LOO-PIT plot saved")

# Pareto k diagnostic plot
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_khat(loo, ax=ax, show_bins=True)
plt.title('Pareto k Diagnostic (LOO-CV)')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pareto_k.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Pareto k plot saved")

# ============================================================================
# COMPUTE METRICS
# ============================================================================
print("\n6. Computing model fit metrics...")

# Bayesian R²
y_pred_flat = y_pred.reshape(-1, N)
var_fit = np.var(y_pred_flat.mean(axis=0))
var_res = np.mean(np.var(y_pred_flat - Y_obs[np.newaxis, :], axis=1))
R2_bayes = var_fit / (var_fit + var_res)

# RMSE (posterior mean)
RMSE = np.sqrt(np.mean(residuals**2))

# MAE
MAE = np.mean(np.abs(residuals))

# LOO metrics
loo_elpd = loo.elpd_loo
loo_se = loo.se
p_loo = loo.p_loo
loo_ic = -2 * loo_elpd

# Count problematic Pareto k values
pareto_k = loo.pareto_k.values
n_high_k = np.sum(pareto_k > 0.7)
n_moderate_k = np.sum((pareto_k > 0.5) & (pareto_k <= 0.7))

# Parameter summaries
summary = az.summary(idata, var_names=['beta_0', 'beta_1', 'sigma'], kind='all')

print("\n" + "="*80)
print("MODEL FIT METRICS")
print("="*80)
print(f"\nIn-Sample Fit:")
print(f"  Bayesian R²:      {R2_bayes:.4f}")
print(f"  RMSE:             {RMSE:.4f}")
print(f"  MAE:              {MAE:.4f}")

print(f"\nLOO Cross-Validation:")
print(f"  ELPD_loo:         {loo_elpd:.2f} ± {loo_se:.2f}")
print(f"  p_loo:            {p_loo:.2f}")
print(f"  LOO-IC:           {loo_ic:.2f}")

print(f"\nPareto k Diagnostics:")
print(f"  k > 0.7 (bad):    {n_high_k} / {N}")
print(f"  k ∈ (0.5, 0.7]:   {n_moderate_k} / {N}")
if n_high_k > 0:
    print(f"  WARNING: {n_high_k} influential observations detected")
    bad_k_idx = np.where(pareto_k > 0.7)[0]
    print(f"  Indices: {bad_k_idx}")
    print(f"  x values: {x[bad_k_idx]}")
    print(f"  Y values: {Y_obs[bad_k_idx]}")

print("\n" + "="*80)
print("PARAMETER INFERENCE")
print("="*80)
print("\nPosterior Estimates:")
print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']].to_string())

# Prior vs Posterior comparison
print("\n" + "="*80)
print("PRIOR VS POSTERIOR LEARNING")
print("="*80)

beta0_prior_mean, beta0_prior_sd = 2.3, 0.3
beta1_prior_mean, beta1_prior_sd = 0.29, 0.15

beta0_post_mean = summary.loc['beta_0', 'mean']
beta0_post_sd = summary.loc['beta_0', 'sd']
beta1_post_mean = summary.loc['beta_1', 'mean']
beta1_post_sd = summary.loc['beta_1', 'sd']

print(f"\nβ₀ (Intercept):")
print(f"  Prior:     μ = {beta0_prior_mean:.3f}, σ = {beta0_prior_sd:.3f}")
print(f"  Posterior: μ = {beta0_post_mean:.3f}, σ = {beta0_post_sd:.3f}")
print(f"  Change:    Δμ = {beta0_post_mean - beta0_prior_mean:.3f}, precision × {(beta0_prior_sd/beta0_post_sd):.2f}")

print(f"\nβ₁ (Log slope):")
print(f"  Prior:     μ = {beta1_prior_mean:.3f}, σ = {beta1_prior_sd:.3f}")
print(f"  Posterior: μ = {beta1_post_mean:.3f}, σ = {beta1_post_sd:.3f}")
print(f"  Change:    Δμ = {beta1_post_mean - beta1_prior_mean:.3f}, precision × {(beta1_prior_sd/beta1_post_sd):.2f}")

# ============================================================================
# SAVE METRICS
# ============================================================================
print("\n7. Saving metrics to file...")

metrics = {
    'convergence': conv_metrics,
    'fit': {
        'R2_bayes': float(R2_bayes),
        'RMSE': float(RMSE),
        'MAE': float(MAE)
    },
    'loo': {
        'elpd_loo': float(loo_elpd),
        'se_elpd_loo': float(loo_se),
        'p_loo': float(p_loo),
        'loo_ic': float(loo_ic),
        'n_high_pareto_k': int(n_high_k),
        'n_moderate_pareto_k': int(n_moderate_k),
        'max_pareto_k': float(pareto_k.max())
    },
    'parameters': {
        'beta_0': {
            'mean': float(beta0_post_mean),
            'sd': float(beta0_post_sd),
            'hdi_3%': float(summary.loc['beta_0', 'hdi_3%']),
            'hdi_97%': float(summary.loc['beta_0', 'hdi_97%'])
        },
        'beta_1': {
            'mean': float(beta1_post_mean),
            'sd': float(beta1_post_sd),
            'hdi_3%': float(summary.loc['beta_1', 'hdi_3%']),
            'hdi_97%': float(summary.loc['beta_1', 'hdi_97%'])
        },
        'sigma': {
            'mean': float(summary.loc['sigma', 'mean']),
            'sd': float(summary.loc['sigma', 'sd']),
            'hdi_3%': float(summary.loc['sigma', 'hdi_3%']),
            'hdi_97%': float(summary.loc['sigma', 'hdi_97%'])
        }
    }
}

with open(DIAG_DIR / "model_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"   - Saved to: {DIAG_DIR / 'model_metrics.json'}")

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Convergence: {'PASSED' if conv_metrics['convergence_pass'] else 'FAILED'}")
print(f"  Model Fit: R² = {R2_bayes:.3f}, RMSE = {RMSE:.3f}")
print(f"  LOO-ELPD: {loo_elpd:.1f} ± {loo_se:.1f}")
print(f"  Problematic obs: {n_high_k} (Pareto k > 0.7)")
print(f"\nAll plots saved to: {PLOTS_DIR}")
print(f"All diagnostics saved to: {DIAG_DIR}")
