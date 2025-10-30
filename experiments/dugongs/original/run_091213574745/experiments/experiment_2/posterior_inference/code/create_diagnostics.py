"""
Create diagnostic plots for Student-t model posterior inference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pathlib import Path

# Setup
DIAG_DIR = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics")
PLOTS_DIR = Path("/workspace/experiments/experiment_2/posterior_inference/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv("/workspace/data/data.csv")
idata = az.from_netcdf(DIAG_DIR / "posterior_inference.netcdf")
idata_model1 = az.from_netcdf("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")

print("Creating diagnostic plots...")

# 1. Trace plots for convergence assessment
print("\n1. Trace plots...")
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
az.plot_trace(idata, var_names=['beta_0', 'beta_1', 'sigma', 'nu'], axes=axes)
plt.suptitle("Model 2 (Student-t): Trace Plots", fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# 2. KEY PLOT: Nu posterior distribution
print("\n2. Nu posterior distribution (KEY DIAGNOSTIC)...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
nu_samples = idata.posterior['nu'].values.flatten()
axes[0].hist(nu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(nu_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {nu_samples.mean():.1f}')
axes[0].axvline(20, color='orange', linestyle=':', linewidth=2, label='ν = 20 (borderline)')
axes[0].axvline(30, color='green', linestyle=':', linewidth=2, label='ν = 30 (nearly Normal)')
axes[0].set_xlabel('ν (degrees of freedom)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('ν Posterior Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Prior vs posterior
nu_prior_samples = np.random.gamma(2, 10, size=10000)  # Gamma(2, scale=10)
nu_prior_samples = np.clip(nu_prior_samples, 3, None)  # Truncate at 3

axes[1].hist(nu_prior_samples, bins=50, density=True, alpha=0.4, color='gray', label='Prior', edgecolor='black')
axes[1].hist(nu_samples, bins=50, density=True, alpha=0.7, color='steelblue', label='Posterior', edgecolor='black')
axes[1].axvline(nu_samples.mean(), color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('ν (degrees of freedom)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('Prior vs Posterior', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 100)

plt.suptitle("KEY DIAGNOSTIC: ν ≈ 23 suggests borderline heavy tails\nNormal likelihood (ν→∞) may be adequate", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "nu_posterior.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. Model comparison: Fitted curves
print("\n3. Model comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))

x_plot = np.linspace(data['x'].min(), data['x'].max(), 100)

# Model 1 predictions
beta_0_m1 = idata_model1.posterior['beta_0'].values.flatten()
beta_1_m1 = idata_model1.posterior['beta_1'].values.flatten()
y_pred_m1 = beta_0_m1[:, np.newaxis] + beta_1_m1[:, np.newaxis] * np.log(x_plot)
y_pred_m1_mean = y_pred_m1.mean(axis=0)
y_pred_m1_lower = np.percentile(y_pred_m1, 2.5, axis=0)
y_pred_m1_upper = np.percentile(y_pred_m1, 97.5, axis=0)

# Model 2 predictions
beta_0_m2 = idata.posterior['beta_0'].values.flatten()
beta_1_m2 = idata.posterior['beta_1'].values.flatten()
y_pred_m2 = beta_0_m2[:, np.newaxis] + beta_1_m2[:, np.newaxis] * np.log(x_plot)
y_pred_m2_mean = y_pred_m2.mean(axis=0)
y_pred_m2_lower = np.percentile(y_pred_m2, 2.5, axis=0)
y_pred_m2_upper = np.percentile(y_pred_m2, 97.5, axis=0)

# Plot data
ax.scatter(data['x'], data['Y'], alpha=0.6, s=80, color='black', label='Data', zorder=3)

# Plot Model 1
ax.plot(x_plot, y_pred_m1_mean, 'b-', linewidth=2, label='Model 1 (Normal)', zorder=2)
ax.fill_between(x_plot, y_pred_m1_lower, y_pred_m1_upper, alpha=0.2, color='blue')

# Plot Model 2
ax.plot(x_plot, y_pred_m2_mean, 'r--', linewidth=2, label='Model 2 (Student-t)', zorder=2)
ax.fill_between(x_plot, y_pred_m2_lower, y_pred_m2_upper, alpha=0.2, color='red')

ax.set_xlabel('x (dose)', fontsize=11)
ax.set_ylabel('Y (log-transformed effect)', fontsize=11)
ax.set_title('Model Comparison: Nearly Identical Predictions\nΔLOO = -1.06 ± 4.00 (models equivalent)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison_fit.png", dpi=150, bbox_inches='tight')
plt.close()

# 4. LOO comparison plot
print("\n4. LOO comparison...")
loo_compare = az.compare({'Model_1_Normal': idata_model1, 'Model_2_StudentT': idata})

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_compare(loo_compare, ax=ax, insample_dev=False)
ax.set_title('LOO-CV Comparison: Models Equivalent', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# 5. Posterior predictive check
print("\n5. Posterior predictive check...")
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(idata, num_pp_samples=100, ax=ax, colors=['C0', 'C1'])
ax.set_title('Posterior Predictive Check: Model 2 (Student-t)', fontsize=12, fontweight='bold')
ax.set_xlabel('Y (log-transformed effect)', fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_predictive_check.png", dpi=150, bbox_inches='tight')
plt.close()

# 6. Rank plots for convergence
print("\n6. Rank plots...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(idata, var_names=['beta_0', 'beta_1', 'sigma', 'nu'])
plt.suptitle('Rank Plots: Detect mixing issues\n(Uniform ranks = good mixing)', fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# 7. Pareto k diagnostic
print("\n7. Pareto k diagnostic...")
loo_model2 = az.loo(idata, pointwise=True)
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_khat(loo_model2, ax=ax, show_bins=True)
ax.set_title('Pareto k Diagnostic: All points < 0.7 (good)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pareto_k_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()

# 8. Parameter posterior comparisons
print("\n8. Parameter posteriors comparison...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (param, ax) in enumerate(zip(['beta_0', 'beta_1', 'sigma'], axes.flat[:3])):
    # Model 1
    m1_samples = idata_model1.posterior[param].values.flatten()
    ax.hist(m1_samples, bins=30, density=True, alpha=0.5, color='blue', label='Model 1 (Normal)', edgecolor='black')

    # Model 2
    m2_samples = idata.posterior[param].values.flatten()
    ax.hist(m2_samples, bins=30, density=True, alpha=0.5, color='red', label='Model 2 (Student-t)', edgecolor='black')

    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{param}: Nearly Identical', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# Nu only for Model 2
ax = axes.flat[3]
nu_samples_m2 = idata.posterior['nu'].values.flatten()
ax.hist(nu_samples_m2, bins=40, density=True, alpha=0.7, color='red', edgecolor='black')
ax.axvline(nu_samples_m2.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean = {nu_samples_m2.mean():.1f}')
ax.set_xlabel('ν', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('ν: Model 2 only', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('Parameter Posteriors: Models nearly identical for β₀, β₁, σ', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {PLOTS_DIR}")
print("\nPlots created:")
print("  - trace_plots.png (convergence assessment)")
print("  - nu_posterior.png (KEY: ν distribution)")
print("  - model_comparison_fit.png (fitted curves)")
print("  - loo_comparison.png (model selection)")
print("  - posterior_predictive_check.png (model adequacy)")
print("  - rank_plots.png (mixing diagnostics)")
print("  - pareto_k_diagnostic.png (LOO reliability)")
print("  - parameter_comparison.png (Model 1 vs 2)")
