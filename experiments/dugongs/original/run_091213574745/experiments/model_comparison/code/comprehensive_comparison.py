"""
Comprehensive Model Assessment and Comparison
==============================================

Compares Model 1 (Normal) vs Model 2 (Student-t) using:
- LOO-CV with az.compare()
- Individual model diagnostics
- Parameter and prediction comparisons
- Calibration assessment
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
model1_path = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
model2_path = "/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf"
data_path = "/workspace/data/data.csv"
output_dir = Path("/workspace/experiments/model_comparison")
plots_dir = output_dir / "plots"

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print()

# Load data
print("Loading data...")
data = pd.read_csv(data_path)
x = data['x'].values
y = data['Y'].values
n_obs = len(y)
print(f"Data: {n_obs} observations")
print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
print()

# Load InferenceData
print("Loading InferenceData objects...")
idata1 = az.from_netcdf(model1_path)
idata2 = az.from_netcdf(model2_path)
print("Model 1 (Normal): Loaded")
print("Model 2 (Student-t): Loaded")
print()

# Verify log_likelihood exists
print("Verifying log_likelihood groups...")
assert hasattr(idata1, 'log_likelihood'), "Model 1 missing log_likelihood"
assert hasattr(idata2, 'log_likelihood'), "Model 2 missing log_likelihood"
print("Model 1: log_likelihood present")
print("Model 2: log_likelihood present")
print()

# ============================================================================
# PART 1: LOO-CV COMPARISON
# ============================================================================
print("="*80)
print("PART 1: LOO-CV MODEL COMPARISON")
print("="*80)
print()

print("Computing LOO for Model 1...")
loo1 = az.loo(idata1, pointwise=True)
print(f"Model 1 LOO-ELPD: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"Model 1 p_loo: {loo1.p_loo:.2f}")
print()

print("Computing LOO for Model 2...")
loo2 = az.loo(idata2, pointwise=True)
print(f"Model 2 LOO-ELPD: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
print(f"Model 2 p_loo: {loo2.p_loo:.2f}")
print()

print("Comparing models with az.compare()...")
compare_dict = {
    "Model 1 (Normal)": idata1,
    "Model 2 (Student-t)": idata2
}
comparison = az.compare(compare_dict, ic="loo", method="stacking")
print()
print("Comparison Table:")
print(comparison)
print()

# Save comparison table
comparison.to_csv(output_dir / "comparison_table.csv")
print(f"Saved comparison table to: {output_dir / 'comparison_table.csv'}")
print()

# Interpret results
delta_loo = comparison.loc["Model 2 (Student-t)", "elpd_diff"]
se_diff = comparison.loc["Model 2 (Student-t)", "dse"]
print("Interpretation:")
print(f"ΔLOO (Model 2 - Model 1): {delta_loo:.2f} ± {se_diff:.2f}")
if abs(delta_loo) < 2 * se_diff:
    print("→ Models are EQUIVALENT (|ΔLOO| < 2*SE)")
    print("→ Apply parsimony principle: prefer simpler Model 1")
elif delta_loo > 2 * se_diff:
    print("→ Model 2 is BETTER")
    if delta_loo > 4 * se_diff:
        print("→ Strong evidence for Model 2")
    else:
        print("→ Moderate evidence for Model 2")
else:
    print("→ Model 1 is BETTER")
print()

# Pareto k diagnostics
print("Pareto k diagnostics:")
k_threshold = 0.7
n_high_k1 = np.sum(loo1.pareto_k > k_threshold)
n_high_k2 = np.sum(loo2.pareto_k > k_threshold)
print(f"Model 1: {n_high_k1}/{n_obs} observations with k > {k_threshold}")
print(f"Model 1: max k = {loo1.pareto_k.max():.3f}, mean k = {loo1.pareto_k.mean():.3f}")
print(f"Model 2: {n_high_k2}/{n_obs} observations with k > {k_threshold}")
print(f"Model 2: max k = {loo2.pareto_k.max():.3f}, mean k = {loo2.pareto_k.mean():.3f}")
print()

# ============================================================================
# PART 2: INDIVIDUAL MODEL ASSESSMENTS
# ============================================================================
print("="*80)
print("PART 2: INDIVIDUAL MODEL ASSESSMENTS")
print("="*80)
print()

# Model 1 Assessment
print("--- Model 1 (Normal Likelihood) ---")
print()
print("Parameter Posteriors:")
beta0_1 = idata1.posterior['beta_0'].values.flatten()
beta1_1 = idata1.posterior['beta_1'].values.flatten()
sigma_1 = idata1.posterior['sigma'].values.flatten()
print(f"β₀: {beta0_1.mean():.4f} [{np.percentile(beta0_1, 2.5):.4f}, {np.percentile(beta0_1, 97.5):.4f}]")
print(f"β₁: {beta1_1.mean():.4f} [{np.percentile(beta1_1, 2.5):.4f}, {np.percentile(beta1_1, 97.5):.4f}]")
print(f"σ: {sigma_1.mean():.4f} [{np.percentile(sigma_1, 2.5):.4f}, {np.percentile(sigma_1, 97.5):.4f}]")
print()

# Convergence diagnostics
print("Convergence Diagnostics:")
summary1 = az.summary(idata1, var_names=['beta_0', 'beta_1', 'sigma'])
print(summary1[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])
print()

# Model 2 Assessment
print("--- Model 2 (Student-t Likelihood) ---")
print()
print("Parameter Posteriors:")
beta0_2 = idata2.posterior['beta_0'].values.flatten()
beta1_2 = idata2.posterior['beta_1'].values.flatten()
sigma_2 = idata2.posterior['sigma'].values.flatten()
nu_2 = idata2.posterior['nu'].values.flatten()
print(f"β₀: {beta0_2.mean():.4f} [{np.percentile(beta0_2, 2.5):.4f}, {np.percentile(beta0_2, 97.5):.4f}]")
print(f"β₁: {beta1_2.mean():.4f} [{np.percentile(beta1_2, 2.5):.4f}, {np.percentile(beta1_2, 97.5):.4f}]")
print(f"σ: {sigma_2.mean():.4f} [{np.percentile(sigma_2, 2.5):.4f}, {np.percentile(sigma_2, 97.5):.4f}]")
print(f"ν: {nu_2.mean():.2f} [{np.percentile(nu_2, 2.5):.2f}, {np.percentile(nu_2, 97.5):.2f}]")
print()
print("Interpretation of ν:")
if np.percentile(nu_2, 2.5) > 30:
    print("→ ν > 30 with high confidence: Student-t ≈ Normal, prefer simpler Model 1")
elif nu_2.mean() > 30:
    print("→ ν ≈ 30: Student-t provides minimal benefit over Normal")
else:
    print("→ ν < 30: Student-t may capture heavier tails than Normal")
print()

print("Convergence Diagnostics:")
summary2 = az.summary(idata2, var_names=['beta_0', 'beta_1', 'sigma', 'nu'])
print(summary2[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])
print()

# ============================================================================
# PART 3: PREDICTIVE METRICS
# ============================================================================
print("="*80)
print("PART 3: PREDICTIVE PERFORMANCE")
print("="*80)
print()

# Posterior predictive means
print("Computing posterior predictions...")
y_pred_1 = idata1.posterior['y_pred'].values
y_pred_mean_1 = y_pred_1.mean(axis=(0, 1))
y_pred_lower_1 = np.percentile(y_pred_1, 5, axis=(0, 1))
y_pred_upper_1 = np.percentile(y_pred_1, 95, axis=(0, 1))

y_pred_2 = idata2.posterior['y_pred'].values
y_pred_mean_2 = y_pred_2.mean(axis=(0, 1))
y_pred_lower_2 = np.percentile(y_pred_2, 5, axis=(0, 1))
y_pred_upper_2 = np.percentile(y_pred_2, 95, axis=(0, 1))

# RMSE and MAE
rmse_1 = np.sqrt(np.mean((y - y_pred_mean_1)**2))
mae_1 = np.mean(np.abs(y - y_pred_mean_1))
rmse_2 = np.sqrt(np.mean((y - y_pred_mean_2)**2))
mae_2 = np.mean(np.abs(y - y_pred_mean_2))

print("Model 1:")
print(f"  RMSE: {rmse_1:.4f}")
print(f"  MAE: {mae_1:.4f}")
print()

print("Model 2:")
print(f"  RMSE: {rmse_2:.4f}")
print(f"  MAE: {mae_2:.4f}")
print()

# R-squared
ss_tot = np.sum((y - y.mean())**2)
ss_res_1 = np.sum((y - y_pred_mean_1)**2)
ss_res_2 = np.sum((y - y_pred_mean_2)**2)
r2_1 = 1 - ss_res_1 / ss_tot
r2_2 = 1 - ss_res_2 / ss_tot

print(f"Model 1 R²: {r2_1:.4f}")
print(f"Model 2 R²: {r2_2:.4f}")
print()

# Coverage
coverage_1 = np.mean((y >= y_pred_lower_1) & (y <= y_pred_upper_1))
coverage_2 = np.mean((y >= y_pred_lower_2) & (y <= y_pred_upper_2))

print(f"Model 1 90% Interval Coverage: {coverage_1:.1%} (target: 90%)")
print(f"Model 2 90% Interval Coverage: {coverage_2:.1%} (target: 90%)")
print()

# ============================================================================
# PART 4: VISUALIZATIONS
# ============================================================================
print("="*80)
print("PART 4: CREATING VISUALIZATIONS")
print("="*80)
print()

# 1. LOO Comparison Plot
print("Creating LOO comparison plot...")
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_compare(comparison, insample_dev=False, ax=ax)
ax.set_xlabel("LOO-ELPD")
ax.set_title("Model Comparison: LOO-ELPD with Standard Errors", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / "loo_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'loo_comparison.png'}")

# 2. Pareto k Comparison
print("Creating Pareto k comparison...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
az.plot_khat(loo1, ax=ax, show_bins=True, hlines_kwargs={'colors': 'red', 'alpha': 0.5})
ax.set_title("Model 1: Pareto k Diagnostics", fontsize=11, fontweight='bold')
ax.set_ylim([-0.1, max(loo1.pareto_k.max(), 0.7) + 0.1])

ax = axes[1]
az.plot_khat(loo2, ax=ax, show_bins=True, hlines_kwargs={'colors': 'red', 'alpha': 0.5})
ax.set_title("Model 2: Pareto k Diagnostics", fontsize=11, fontweight='bold')
ax.set_ylim([-0.1, max(loo2.pareto_k.max(), 0.7) + 0.1])

plt.tight_layout()
plt.savefig(plots_dir / "pareto_k_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'pareto_k_comparison.png'}")

# 3. LOO-PIT Comparison
print("Creating LOO-PIT comparison...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
az.plot_loo_pit(idata1, y="Y", ax=ax)
ax.set_title("Model 1: LOO-PIT", fontsize=11, fontweight='bold')

ax = axes[1]
az.plot_loo_pit(idata2, y="Y", ax=ax)
ax.set_title("Model 2: LOO-PIT", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / "loo_pit_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'loo_pit_comparison.png'}")

# 4. Parameter Posteriors Comparison
print("Creating parameter comparison...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

params = [
    ('beta_0', r'$\beta_0$', beta0_1, beta0_2),
    ('beta_1', r'$\beta_1$', beta1_1, beta1_2),
    ('sigma', r'$\sigma$', sigma_1, sigma_2)
]

for idx, (name, label, vals1, vals2) in enumerate(params):
    ax = axes[idx]
    ax.hist(vals1, bins=40, alpha=0.6, density=True, label='Model 1', color='blue')
    ax.hist(vals2, bins=40, alpha=0.6, density=True, label='Model 2', color='orange')
    ax.axvline(vals1.mean(), color='blue', linestyle='--', linewidth=2)
    ax.axvline(vals2.mean(), color='orange', linestyle='--', linewidth=2)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend()
    ax.set_title(f'{label} Posterior', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'parameter_comparison.png'}")

# 5. Nu posterior (Model 2 only)
print("Creating nu posterior plot...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(nu_2, bins=50, alpha=0.7, density=True, color='purple', edgecolor='black')
ax.axvline(nu_2.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {nu_2.mean():.1f}')
ax.axvline(30, color='green', linestyle=':', linewidth=2, label='ν=30 (Normal threshold)')
ax.set_xlabel(r'$\nu$ (degrees of freedom)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Model 2: Student-t Degrees of Freedom', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(plots_dir / "nu_posterior.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'nu_posterior.png'}")

# 6. Prediction Comparison
print("Creating prediction comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by x for plotting
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_sorted = y[sort_idx]
y_pred_mean_1_sorted = y_pred_mean_1[sort_idx]
y_pred_lower_1_sorted = y_pred_lower_1[sort_idx]
y_pred_upper_1_sorted = y_pred_upper_1[sort_idx]
y_pred_mean_2_sorted = y_pred_mean_2[sort_idx]
y_pred_lower_2_sorted = y_pred_lower_2[sort_idx]
y_pred_upper_2_sorted = y_pred_upper_2[sort_idx]

# Plot data
ax.scatter(x_sorted, y_sorted, color='black', s=50, alpha=0.6, label='Observed data', zorder=5)

# Model 1
ax.plot(x_sorted, y_pred_mean_1_sorted, color='blue', linewidth=2, label='Model 1 (Normal)', zorder=3)
ax.fill_between(x_sorted, y_pred_lower_1_sorted, y_pred_upper_1_sorted,
                alpha=0.2, color='blue', label='Model 1 90% CI', zorder=1)

# Model 2
ax.plot(x_sorted, y_pred_mean_2_sorted, color='orange', linewidth=2,
        linestyle='--', label='Model 2 (Student-t)', zorder=4)
ax.fill_between(x_sorted, y_pred_lower_2_sorted, y_pred_upper_2_sorted,
                alpha=0.2, color='orange', label='Model 2 90% CI', zorder=2)

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title('Prediction Comparison: Model 1 vs Model 2', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "prediction_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'prediction_comparison.png'}")

# 7. Residual Comparison
print("Creating residual comparison...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Model 1 residuals
resid_1 = y - y_pred_mean_1
ax = axes[0, 0]
ax.scatter(y_pred_mean_1, resid_1, alpha=0.6, color='blue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted values', fontsize=10)
ax.set_ylabel('Residuals', fontsize=10)
ax.set_title('Model 1: Residual Plot', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Model 1 Q-Q
ax = axes[0, 1]
from scipy import stats
stats.probplot(resid_1, dist="norm", plot=ax)
ax.set_title('Model 1: Q-Q Plot', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Model 2 residuals
resid_2 = y - y_pred_mean_2
ax = axes[1, 0]
ax.scatter(y_pred_mean_2, resid_2, alpha=0.6, color='orange')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted values', fontsize=10)
ax.set_ylabel('Residuals', fontsize=10)
ax.set_title('Model 2: Residual Plot', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Model 2 Q-Q
ax = axes[1, 1]
stats.probplot(resid_2, dist="norm", plot=ax)
ax.set_title('Model 2: Q-Q Plot', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / "residual_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'residual_comparison.png'}")

# 8. Integrated Dashboard
print("Creating integrated comparison dashboard...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 8a. LOO comparison
ax = fig.add_subplot(gs[0, :2])
az.plot_compare(comparison, insample_dev=False, ax=ax)
ax.set_title('A. Model Comparison (LOO-ELPD)', fontsize=11, fontweight='bold')

# 8b. Pareto k summary
ax = fig.add_subplot(gs[0, 2])
k_bins = ['< 0.5', '0.5-0.7', '> 0.7']
k_counts_1 = [
    np.sum(loo1.pareto_k < 0.5),
    np.sum((loo1.pareto_k >= 0.5) & (loo1.pareto_k <= 0.7)),
    np.sum(loo1.pareto_k > 0.7)
]
k_counts_2 = [
    np.sum(loo2.pareto_k < 0.5),
    np.sum((loo2.pareto_k >= 0.5) & (loo2.pareto_k <= 0.7)),
    np.sum(loo2.pareto_k > 0.7)
]
x_pos = np.arange(len(k_bins))
width = 0.35
ax.bar(x_pos - width/2, k_counts_1, width, label='Model 1', color='blue', alpha=0.7)
ax.bar(x_pos + width/2, k_counts_2, width, label='Model 2', color='orange', alpha=0.7)
ax.set_xlabel('Pareto k', fontsize=9)
ax.set_ylabel('Count', fontsize=9)
ax.set_title('B. Pareto k Distribution', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(k_bins)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 8c. Parameter comparison (beta0)
ax = fig.add_subplot(gs[1, 0])
ax.hist(beta0_1, bins=30, alpha=0.6, density=True, label='Model 1', color='blue')
ax.hist(beta0_2, bins=30, alpha=0.6, density=True, label='Model 2', color='orange')
ax.set_xlabel(r'$\beta_0$', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title(r'C. $\beta_0$ Posterior', fontsize=11, fontweight='bold')
ax.legend()

# 8d. Parameter comparison (beta1)
ax = fig.add_subplot(gs[1, 1])
ax.hist(beta1_1, bins=30, alpha=0.6, density=True, label='Model 1', color='blue')
ax.hist(beta1_2, bins=30, alpha=0.6, density=True, label='Model 2', color='orange')
ax.set_xlabel(r'$\beta_1$', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title(r'D. $\beta_1$ Posterior', fontsize=11, fontweight='bold')
ax.legend()

# 8e. Nu posterior
ax = fig.add_subplot(gs[1, 2])
ax.hist(nu_2, bins=30, alpha=0.7, density=True, color='purple')
ax.axvline(30, color='green', linestyle='--', linewidth=2, label='ν=30')
ax.set_xlabel(r'$\nu$', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('E. ν (Model 2 only)', fontsize=11, fontweight='bold')
ax.legend()

# 8f. Predictions
ax = fig.add_subplot(gs[2, :])
ax.scatter(x_sorted, y_sorted, color='black', s=30, alpha=0.6, label='Data', zorder=5)
ax.plot(x_sorted, y_pred_mean_1_sorted, color='blue', linewidth=2, label='Model 1', zorder=3)
ax.plot(x_sorted, y_pred_mean_2_sorted, color='orange', linewidth=2,
        linestyle='--', label='Model 2', zorder=4)
ax.fill_between(x_sorted, y_pred_lower_1_sorted, y_pred_upper_1_sorted,
                alpha=0.15, color='blue', zorder=1)
ax.fill_between(x_sorted, y_pred_lower_2_sorted, y_pred_upper_2_sorted,
                alpha=0.15, color='orange', zorder=2)
ax.set_xlabel('x', fontsize=9)
ax.set_ylabel('Y', fontsize=9)
ax.set_title('F. Predictions: Model 1 vs Model 2 (with 90% CI)', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Integrated Model Comparison Dashboard', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(plots_dir / "integrated_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plots_dir / 'integrated_dashboard.png'}")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("Summary Statistics for Report:")
print("-" * 80)
print(f"Model 1 LOO-ELPD: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"Model 2 LOO-ELPD: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
print(f"ΔLOO: {delta_loo:.2f} ± {se_diff:.2f}")
print(f"Model 1 RMSE: {rmse_1:.4f}, MAE: {mae_1:.4f}, R²: {r2_1:.4f}")
print(f"Model 2 RMSE: {rmse_2:.4f}, MAE: {mae_2:.4f}, R²: {r2_2:.4f}")
print(f"Model 2 ν: {nu_2.mean():.1f} [{np.percentile(nu_2, 2.5):.1f}, {np.percentile(nu_2, 97.5):.1f}]")
print(f"Model 1 Coverage: {coverage_1:.1%}")
print(f"Model 2 Coverage: {coverage_2:.1%}")
print("-" * 80)

# Save summary statistics
summary_stats = {
    'Model': ['Model 1 (Normal)', 'Model 2 (Student-t)'],
    'LOO-ELPD': [f"{loo1.elpd_loo:.2f}", f"{loo2.elpd_loo:.2f}"],
    'SE': [f"{loo1.se:.2f}", f"{loo2.se:.2f}"],
    'p_loo': [f"{loo1.p_loo:.2f}", f"{loo2.p_loo:.2f}"],
    'RMSE': [f"{rmse_1:.4f}", f"{rmse_2:.4f}"],
    'MAE': [f"{mae_1:.4f}", f"{mae_2:.4f}"],
    'R²': [f"{r2_1:.4f}", f"{r2_2:.4f}"],
    'Coverage_90': [f"{coverage_1:.1%}", f"{coverage_2:.1%}"],
    'Max_Pareto_k': [f"{loo1.pareto_k.max():.3f}", f"{loo2.pareto_k.max():.3f}"]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
print(f"Saved summary statistics to: {output_dir / 'summary_statistics.csv'}")
