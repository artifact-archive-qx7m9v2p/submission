"""
Comprehensive Model Assessment and Comparison
Eight Schools Analysis - Hierarchical vs Complete Pooling
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
exp1_path = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
exp2_path = "/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf"
output_dir = Path("/workspace/experiments/model_comparison")
figures_dir = output_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Load models
print("Loading InferenceData objects...")
idata_hierarchical = az.from_netcdf(exp1_path)
idata_pooled = az.from_netcdf(exp2_path)

# Observed data (Eight Schools)
y_obs = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma_obs = np.array([15, 10, 16, 11, 9, 11, 10, 18])
schools = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

print("\n" + "="*80)
print("COMPREHENSIVE MODEL ASSESSMENT AND COMPARISON")
print("="*80)

# ============================================================================
# PART 1: INDIVIDUAL MODEL ASSESSMENT
# ============================================================================

print("\n" + "-"*80)
print("PART 1: INDIVIDUAL MODEL ASSESSMENT")
print("-"*80)

# Model 1: Hierarchical
print("\n### MODEL 1: HIERARCHICAL (Non-centered) ###\n")

# Check for log_likelihood
if 'log_likelihood' not in idata_hierarchical.groups():
    print("ERROR: No log_likelihood group in hierarchical model!")
    raise ValueError("Missing log_likelihood")
else:
    # Compute LOO
    print("Computing LOO-CV...")
    loo_hierarchical = az.loo(idata_hierarchical)
    print(loo_hierarchical)

    # Pareto k diagnostics
    pareto_k = loo_hierarchical.pareto_k.values
    print(f"\nPareto k diagnostics:")
    print(f"  All k < 0.5 (good): {np.sum(pareto_k < 0.5)}/{len(pareto_k)}")
    print(f"  0.5 ≤ k < 0.7 (ok): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))}/{len(pareto_k)}")
    print(f"  k ≥ 0.7 (bad): {np.sum(pareto_k >= 0.7)}/{len(pareto_k)}")
    print(f"  Max k: {np.max(pareto_k):.4f}")
    print(f"  Mean k: {np.mean(pareto_k):.4f}")

# Model 2: Complete Pooling
print("\n### MODEL 2: COMPLETE POOLING ###\n")

if 'log_likelihood' not in idata_pooled.groups():
    print("ERROR: No log_likelihood group in pooled model!")
    raise ValueError("Missing log_likelihood")
else:
    print("Computing LOO-CV...")
    loo_pooled = az.loo(idata_pooled)
    print(loo_pooled)

    # Pareto k diagnostics
    pareto_k = loo_pooled.pareto_k.values
    print(f"\nPareto k diagnostics:")
    print(f"  All k < 0.5 (good): {np.sum(pareto_k < 0.5)}/{len(pareto_k)}")
    print(f"  0.5 ≤ k < 0.7 (ok): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))}/{len(pareto_k)}")
    print(f"  k ≥ 0.7 (bad): {np.sum(pareto_k >= 0.7)}/{len(pareto_k)}")
    print(f"  Max k: {np.max(pareto_k):.4f}")
    print(f"  Mean k: {np.mean(pareto_k):.4f}")

# ============================================================================
# PART 2: MODEL COMPARISON
# ============================================================================

print("\n" + "-"*80)
print("PART 2: MODEL COMPARISON")
print("-"*80)

# Compare models
print("\nComparing models with az.compare()...\n")
model_dict = {
    "Hierarchical": idata_hierarchical,
    "Complete_Pooling": idata_pooled
}
comparison = az.compare(model_dict)
print(comparison)

# Extract key comparison metrics
delta_elpd = comparison.loc['Complete_Pooling', 'elpd_diff']
se_diff = comparison.loc['Complete_Pooling', 'dse']
weight_hier = comparison.loc['Hierarchical', 'weight']
weight_pool = comparison.loc['Complete_Pooling', 'weight']

print(f"\n### Comparison Summary ###")
print(f"ΔELPD (Pooled - Hierarchical): {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"Significance threshold (2×SE): {2*se_diff:.2f}")
print(f"Is difference significant? {abs(delta_elpd) > 2*se_diff}")
print(f"\nModel weights (Akaike):")
print(f"  Hierarchical: {weight_hier:.4f}")
print(f"  Complete Pooling: {weight_pool:.4f}")

# Parsimony assessment
p_eff_hier = comparison.loc['Hierarchical', 'p_loo']
p_eff_pool = comparison.loc['Complete_Pooling', 'p_loo']
print(f"\nEffective parameters (p_loo):")
print(f"  Hierarchical: {p_eff_hier:.2f}")
print(f"  Complete Pooling: {p_eff_pool:.2f}")
print(f"  Difference: {p_eff_hier - p_eff_pool:.2f}")

# Save comparison table
comparison.to_csv(output_dir / "loo_comparison.csv")
print(f"\nComparison table saved to: {output_dir / 'loo_comparison.csv'}")

# ============================================================================
# PART 3: GENERATE POSTERIOR PREDICTIONS FOR CALIBRATION
# ============================================================================

print("\n" + "-"*80)
print("PART 3: GENERATING POSTERIOR PREDICTIONS")
print("-"*80)

# Extract posterior samples for predictions
print("\nExtracting posterior parameters...")

# Hierarchical model: theta[j] ~ N(mu, tau)
if 'theta' in idata_hierarchical.posterior.data_vars:
    theta_hier = idata_hierarchical.posterior['theta'].values
    print(f"Hierarchical: Using theta directly, shape={theta_hier.shape}")
    # Flatten chains and draws
    n_chains_hier, n_draws_hier, n_schools = theta_hier.shape
    theta_hier_flat = theta_hier.reshape(n_chains_hier * n_draws_hier, n_schools)
else:
    print("ERROR: No theta in hierarchical model")
    theta_hier_flat = None

# Complete pooling model: all schools have same mu
if 'mu' in idata_pooled.posterior.data_vars:
    mu_pool = idata_pooled.posterior['mu'].values
    print(f"Complete pooling: Using mu, shape={mu_pool.shape}")
    # Flatten and replicate for all schools
    n_chains_pool, n_draws_pool = mu_pool.shape
    mu_pool_flat = mu_pool.reshape(n_chains_pool * n_draws_pool)
    theta_pool_flat = np.tile(mu_pool_flat[:, np.newaxis], (1, len(schools)))
else:
    print("ERROR: No mu in pooled model")
    theta_pool_flat = None

# Generate posterior predictive samples: y ~ N(theta, sigma)
print("\nGenerating posterior predictive samples...")

# Use separate sample counts for each model
n_samples_hier = theta_hier_flat.shape[0]
n_samples_pool = theta_pool_flat.shape[0]

print(f"Hierarchical: {n_samples_hier} samples")
print(f"Pooled: {n_samples_pool} samples")

y_pred_hier = np.zeros((n_samples_hier, len(schools)))
y_pred_pool = np.zeros((n_samples_pool, len(schools)))

np.random.seed(42)  # For reproducibility
for j in range(len(schools)):
    y_pred_hier[:, j] = np.random.normal(theta_hier_flat[:, j], sigma_obs[j])
    y_pred_pool[:, j] = np.random.normal(theta_pool_flat[:, j], sigma_obs[j])

print(f"Generated {n_samples_hier} hierarchical samples for each of {len(schools)} schools")
print(f"Generated {n_samples_pool} pooled samples for each of {len(schools)} schools")

# Compute posterior mean predictions
y_pred_hier_mean = theta_hier_flat.mean(axis=0)
y_pred_pool_mean = theta_pool_flat.mean(axis=0)

print(f"\nPosterior means:")
print(f"  Hierarchical: {y_pred_hier_mean}")
print(f"  Complete Pooling: {y_pred_pool_mean}")
print(f"  Observed: {y_obs}")

# ============================================================================
# PART 4: CALIBRATION ASSESSMENT (Manual LOO-PIT)
# ============================================================================

print("\n" + "-"*80)
print("PART 4: CALIBRATION ASSESSMENT")
print("-"*80)

# Note: LOO-PIT is complex to compute manually. Instead, let's use az.loo_pit
# But we need to add posterior_predictive to the InferenceData first

print("\nAdding posterior predictive samples to InferenceData for LOO-PIT...")

# Add posterior predictive to hierarchical model
idata_hierarchical.add_groups({
    'posterior_predictive': {
        'y': (['chain', 'draw', 'school'],
              y_pred_hier.reshape(n_chains_hier, n_draws_hier, len(schools)))
    }
})

# Add posterior predictive to pooled model
idata_pooled.add_groups({
    'posterior_predictive': {
        'y': (['chain', 'draw', 'school'],
              y_pred_pool.reshape(n_chains_pool, n_draws_pool, len(schools)))
    }
})

# Now compute LOO-PIT
print("\nGenerating LOO-PIT plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

az.plot_loo_pit(idata_hierarchical, y='y', legend=False, ax=axes[0])
axes[0].set_title('Hierarchical Model\nLOO-PIT', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Quantile', fontsize=10)
axes[0].set_ylabel('Density', fontsize=10)

az.plot_loo_pit(idata_pooled, y='y', legend=False, ax=axes[1])
axes[1].set_title('Complete Pooling Model\nLOO-PIT', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Quantile', fontsize=10)
axes[1].set_ylabel('Density', fontsize=10)

plt.tight_layout()
plt.savefig(figures_dir / "calibration_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'calibration_comparison.png'}")
plt.close()

# ============================================================================
# PART 5: PREDICTIVE ACCURACY METRICS
# ============================================================================

print("\n" + "-"*80)
print("PART 5: PREDICTIVE ACCURACY METRICS")
print("-"*80)

# Compute metrics
def compute_metrics(y_true, y_pred_mean, y_pred_samples):
    """Compute predictive accuracy metrics"""
    rmse = np.sqrt(np.mean((y_true - y_pred_mean)**2))
    mae = np.mean(np.abs(y_true - y_pred_mean))

    # Coverage: 90% posterior predictive interval
    lower = np.percentile(y_pred_samples, 5, axis=0)
    upper = np.percentile(y_pred_samples, 95, axis=0)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))

    # Also compute 50% and 95% coverage
    lower_50 = np.percentile(y_pred_samples, 25, axis=0)
    upper_50 = np.percentile(y_pred_samples, 75, axis=0)
    coverage_50 = np.mean((y_true >= lower_50) & (y_true <= upper_50))

    lower_95 = np.percentile(y_pred_samples, 2.5, axis=0)
    upper_95 = np.percentile(y_pred_samples, 97.5, axis=0)
    coverage_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'Coverage_50': coverage_50,
        'Coverage_90': coverage,
        'Coverage_95': coverage_95
    }

metrics_hier = compute_metrics(y_obs, y_pred_hier_mean, y_pred_hier)
print(f"\n### Hierarchical Model ###")
print(f"  RMSE: {metrics_hier['RMSE']:.2f}")
print(f"  MAE: {metrics_hier['MAE']:.2f}")
print(f"  50% Coverage: {metrics_hier['Coverage_50']:.1%} (target: 50%)")
print(f"  90% Coverage: {metrics_hier['Coverage_90']:.1%} (target: 90%)")
print(f"  95% Coverage: {metrics_hier['Coverage_95']:.1%} (target: 95%)")

metrics_pool = compute_metrics(y_obs, y_pred_pool_mean, y_pred_pool)
print(f"\n### Complete Pooling Model ###")
print(f"  RMSE: {metrics_pool['RMSE']:.2f}")
print(f"  MAE: {metrics_pool['MAE']:.2f}")
print(f"  50% Coverage: {metrics_pool['Coverage_50']:.1%} (target: 50%)")
print(f"  90% Coverage: {metrics_pool['Coverage_90']:.1%} (target: 90%)")
print(f"  95% Coverage: {metrics_pool['Coverage_95']:.1%} (target: 95%)")

# ============================================================================
# PART 6: VISUALIZATION - LOO COMPARISON
# ============================================================================

print("\n" + "-"*80)
print("PART 6: CREATING COMPARISON VISUALIZATIONS")
print("-"*80)

# Plot 1: LOO comparison plot
fig = plt.figure(figsize=(10, 6))
az.plot_compare(comparison, insample_dev=False)
plt.title('Model Comparison: LOO-CV ELPD', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('ELPD (Expected Log Pointwise Predictive Density)', fontsize=11)
plt.tight_layout()
plt.savefig(figures_dir / "loo_comparison_plot.png", dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'loo_comparison_plot.png'}")
plt.close()

# Plot 2: Pareto k comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

az.plot_khat(loo_hierarchical, ax=axes[0], show_bins=True)
axes[0].set_title('Hierarchical Model\nPareto k Diagnostics', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Data point', fontsize=10)
axes[0].set_ylabel('Pareto k', fontsize=10)
axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='k=0.5')
axes[0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='k=0.7')
axes[0].legend()

az.plot_khat(loo_pooled, ax=axes[1], show_bins=True)
axes[1].set_title('Complete Pooling Model\nPareto k Diagnostics', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Data point', fontsize=10)
axes[1].set_ylabel('Pareto k', fontsize=10)
axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='k=0.5')
axes[1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='k=0.7')
axes[1].legend()

plt.tight_layout()
plt.savefig(figures_dir / "pareto_k_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'pareto_k_comparison.png'}")
plt.close()

# Plot 3: Prediction comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Posterior means comparison
ax = axes[0, 0]
x_pos = np.arange(len(schools))
width = 0.35

ax.bar(x_pos - width/2, y_pred_hier_mean, width, label='Hierarchical', alpha=0.8, color='steelblue')
ax.bar(x_pos + width/2, y_pred_pool_mean, width, label='Complete Pooling', alpha=0.8, color='darkorange')
ax.scatter(x_pos, y_obs, color='red', s=100, zorder=5, label='Observed', marker='*')

ax.set_xlabel('School', fontsize=11)
ax.set_ylabel('Treatment Effect', fontsize=11)
ax.set_title('Posterior Mean Predictions', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(schools)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Panel 2: Prediction errors
ax = axes[0, 1]
errors_hier = y_obs - y_pred_hier_mean
errors_pool = y_obs - y_pred_pool_mean

ax.bar(x_pos - width/2, errors_hier, width, label='Hierarchical', alpha=0.8, color='steelblue')
ax.bar(x_pos + width/2, errors_pool, width, label='Complete Pooling', alpha=0.8, color='darkorange')

ax.set_xlabel('School', fontsize=11)
ax.set_ylabel('Prediction Error (Obs - Pred)', fontsize=11)
ax.set_title('Prediction Errors by School', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(schools)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Panel 3: Posterior predictive uncertainty
ax = axes[1, 0]

y_pred_hier_std = y_pred_hier.std(axis=0)
y_pred_pool_std = y_pred_pool.std(axis=0)

ax.errorbar(x_pos - 0.1, y_pred_hier_mean, yerr=1.96*y_pred_hier_std,
            fmt='o', capsize=5, label='Hierarchical (95% CI)', alpha=0.7, color='steelblue')
ax.errorbar(x_pos + 0.1, y_pred_pool_mean, yerr=1.96*y_pred_pool_std,
            fmt='s', capsize=5, label='Complete Pooling (95% CI)', alpha=0.7, color='darkorange')
ax.scatter(x_pos, y_obs, color='red', s=100, zorder=5, label='Observed', marker='*')

ax.set_xlabel('School', fontsize=11)
ax.set_ylabel('Treatment Effect', fontsize=11)
ax.set_title('Predictions with Uncertainty', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(schools)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Panel 4: Scatter plot of predictions
ax = axes[1, 1]

# Add diagonal line
min_val = min(y_pred_hier_mean.min(), y_pred_pool_mean.min()) - 1
max_val = max(y_pred_hier_mean.max(), y_pred_pool_mean.max()) + 1
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='y=x')

ax.scatter(y_pred_hier_mean, y_pred_pool_mean, s=100, alpha=0.7, color='purple')

# Label points with school names
for i, school in enumerate(schools):
    ax.annotate(school, (y_pred_hier_mean[i], y_pred_pool_mean[i]),
               xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Hierarchical Predictions', fontsize=11)
ax.set_ylabel('Complete Pooling Predictions', fontsize=11)
ax.set_title('Direct Comparison of Predictions', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Make axes equal for better comparison
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(figures_dir / "prediction_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'prediction_comparison.png'}")
plt.close()

# ============================================================================
# PART 7: POINTWISE LOO COMPARISON
# ============================================================================

print("\n" + "-"*80)
print("PART 7: POINTWISE LOO ANALYSIS")
print("-"*80)

# Extract pointwise LOO
loo_hier_pointwise = loo_hierarchical.elpd_loo.values
loo_pool_pointwise = loo_pooled.elpd_loo.values

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Pointwise ELPD
ax = axes[0]
x_pos = np.arange(len(schools))
width = 0.35

ax.bar(x_pos - width/2, loo_hier_pointwise, width, label='Hierarchical', alpha=0.8, color='steelblue')
ax.bar(x_pos + width/2, loo_pool_pointwise, width, label='Complete Pooling', alpha=0.8, color='darkorange')

ax.set_xlabel('School', fontsize=11)
ax.set_ylabel('ELPD (pointwise)', fontsize=11)
ax.set_title('Leave-One-Out ELPD by School', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(schools)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Difference in ELPD
ax = axes[1]
elpd_diff_pointwise = loo_pool_pointwise - loo_hier_pointwise

colors = ['green' if d > 0 else 'red' for d in elpd_diff_pointwise]
ax.bar(x_pos, elpd_diff_pointwise, color=colors, alpha=0.7)

ax.set_xlabel('School', fontsize=11)
ax.set_ylabel('ΔELPD (Pooled - Hierarchical)', fontsize=11)
ax.set_title('Pointwise ELPD Differences', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(schools)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

# Add text box with summary
textstr = f'Mean Δ: {elpd_diff_pointwise.mean():.2f}\nPooled better: {np.sum(elpd_diff_pointwise > 0)}/{len(schools)}'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(figures_dir / "pointwise_loo_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'pointwise_loo_comparison.png'}")
plt.close()

# School-level summary
print("\nPointwise ELPD Summary:")
for i, school in enumerate(schools):
    diff = elpd_diff_pointwise[i]
    better = "Pooled" if diff > 0 else "Hierarchical"
    print(f"  School {school}: Hierarchical={loo_hier_pointwise[i]:.2f}, "
          f"Pooled={loo_pool_pointwise[i]:.2f}, Δ={diff:.2f} (favors {better})")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary_stats = {
    'Model': ['Hierarchical', 'Complete Pooling'],
    'ELPD': [comparison.loc['Hierarchical', 'elpd_loo'],
             comparison.loc['Complete_Pooling', 'elpd_loo']],
    'SE': [comparison.loc['Hierarchical', 'se'],
           comparison.loc['Complete_Pooling', 'se']],
    'p_loo': [p_eff_hier, p_eff_pool],
    'Weight': [weight_hier, weight_pool],
    'RMSE': [metrics_hier['RMSE'], metrics_pool['RMSE']],
    'MAE': [metrics_hier['MAE'], metrics_pool['MAE']],
    'Coverage_90': [metrics_hier['Coverage_90'], metrics_pool['Coverage_90']]
}

summary_df = pd.DataFrame(summary_stats)
print("\n", summary_df.to_string(index=False))

summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
print(f"\nSummary statistics saved to: {output_dir / 'summary_statistics.csv'}")

print("\n" + "="*80)
print("ASSESSMENT COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}")
print(f"Figures saved to: {figures_dir}")
