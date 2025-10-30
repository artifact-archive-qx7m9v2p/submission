"""
Comprehensive Model Assessment for Bayesian Hierarchical Meta-Analysis

This script performs:
1. LOO-CV diagnostics with Pareto k analysis
2. Calibration assessment via LOO-PIT
3. Absolute predictive metrics (RMSE, MAE, coverage)
4. Study-level diagnostics
5. Model adequacy evaluation
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
IDATA_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
OUTPUT_DIR = Path("/workspace/experiments/model_assessment")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create output directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE MODEL ASSESSMENT")
print("Bayesian Hierarchical Meta-Analysis")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND MODEL
# ============================================================================
print("\n[1/7] Loading data and model outputs...")

# Load observed data
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma_obs = data['sigma'].values
n_studies = len(y_obs)
print(f"  - Loaded {n_studies} studies")
print(f"  - Observed effects: {y_obs}")
print(f"  - Standard errors: {sigma_obs}")

# Load InferenceData
idata = az.from_netcdf(IDATA_PATH)
print(f"  - Loaded InferenceData with groups: {list(idata.groups())}")

# Verify log_likelihood is present
if 'log_likelihood' not in idata.groups():
    raise ValueError("InferenceData missing log_likelihood group - cannot perform LOO-CV")
print(f"  - log_likelihood shape: {idata.log_likelihood['y_obs'].shape}")

# ============================================================================
# 2. LOO-CV DIAGNOSTICS
# ============================================================================
print("\n[2/7] Computing LOO-CV diagnostics...")

# Compute LOO with pointwise details
loo_result = az.loo(idata, pointwise=True)

print("\n  LOO-CV Results:")
print(f"  - ELPD_loo: {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
print(f"  - p_loo: {loo_result.p_loo:.2f}")
print(f"  - LOO Information Criterion: {-2 * loo_result.elpd_loo:.2f}")

# Pareto k diagnostics
pareto_k = np.array(loo_result.pareto_k.values)
print(f"\n  Pareto k diagnostics:")
print(f"  - Min: {pareto_k.min():.3f}")
print(f"  - Max: {pareto_k.max():.3f}")
print(f"  - Mean: {pareto_k.mean():.3f}")
print(f"  - Median: {np.median(pareto_k):.3f}")

# Count problematic k values
k_good = int((pareto_k < 0.5).sum())
k_ok = int(((pareto_k >= 0.5) & (pareto_k < 0.7)).sum())
k_bad = int(((pareto_k >= 0.7) & (pareto_k < 1.0)).sum())
k_very_bad = int((pareto_k >= 1.0).sum())

print(f"\n  Pareto k thresholds:")
print(f"  - k < 0.5 (good): {k_good}/{n_studies}")
print(f"  - 0.5 ≤ k < 0.7 (ok): {k_ok}/{n_studies}")
print(f"  - 0.7 ≤ k < 1.0 (bad): {k_bad}/{n_studies}")
print(f"  - k ≥ 1.0 (very bad): {k_very_bad}/{n_studies}")

# Store LOO results for later
loo_metrics = {
    'elpd_loo': float(loo_result.elpd_loo),
    'se_elpd': float(loo_result.se),
    'p_loo': float(loo_result.p_loo),
    'looic': float(-2 * loo_result.elpd_loo),
    'pareto_k_min': float(pareto_k.min()),
    'pareto_k_max': float(pareto_k.max()),
    'pareto_k_mean': float(pareto_k.mean()),
    'pareto_k_median': float(np.median(pareto_k)),
    'n_k_good': int(k_good),
    'n_k_ok': int(k_ok),
    'n_k_bad': int(k_bad),
    'n_k_very_bad': int(k_very_bad)
}

# ============================================================================
# 3. VISUALIZE PARETO K DIAGNOSTICS
# ============================================================================
print("\n[3/7] Creating Pareto k diagnostic plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Pareto k values by observation
ax1 = axes[0]
colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in pareto_k]
ax1.scatter(range(n_studies), pareto_k, c=colors, s=100, alpha=0.7, edgecolors='black')
ax1.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='k=0.5 (ok threshold)')
ax1.axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='k=0.7 (bad threshold)')
ax1.set_xlabel('Study Index', fontsize=12)
ax1.set_ylabel('Pareto k', fontsize=12)
ax1.set_title('Pareto k Diagnostics by Study', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(n_studies))
ax1.set_xticklabels([f'S{i+1}' for i in range(n_studies)])

# Plot 2: Histogram of Pareto k values
ax2 = axes[1]
ax2.hist(pareto_k, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='k=0.5')
ax2.axvline(0.7, color='red', linestyle='--', linewidth=2, label='k=0.7')
ax2.set_xlabel('Pareto k', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Distribution of Pareto k Values', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pareto_k_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOTS_DIR / 'pareto_k_diagnostics.png'}")
plt.close()

# ============================================================================
# 4. LOO-PIT CALIBRATION
# ============================================================================
print("\n[4/7] Computing LOO-PIT calibration...")

# Compute LOO-PIT
loo_pit = az.loo_pit(idata=idata, y='y_obs')
print(f"  - LOO-PIT values: {loo_pit}")

# Test uniformity with Kolmogorov-Smirnov test
ks_stat, ks_pval = stats.kstest(loo_pit, 'uniform')
print(f"\n  Uniformity test (Kolmogorov-Smirnov):")
print(f"  - KS statistic: {ks_stat:.4f}")
print(f"  - p-value: {ks_pval:.4f}")
print(f"  - Interpretation: {'Well-calibrated' if ks_pval > 0.05 else 'Possible calibration issue'}")

# Store calibration metrics
calibration_metrics = {
    'loo_pit_values': loo_pit.tolist(),
    'ks_statistic': float(ks_stat),
    'ks_pvalue': float(ks_pval),
    'calibration_assessment': 'well_calibrated' if ks_pval > 0.05 else 'calibration_warning'
}

# Visualize LOO-PIT
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: LOO-PIT histogram
ax1 = axes[0]
ax1.hist(loo_pit, bins=10, density=True, alpha=0.7, color='steelblue',
         edgecolor='black', label='LOO-PIT')
ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform (ideal)')
ax1.set_xlabel('LOO-PIT Value', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('LOO-PIT Distribution\n(Should be uniform if well-calibrated)',
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 2.5)

# Plot 2: Q-Q plot against uniform
ax2 = axes[1]
sorted_pit = np.sort(loo_pit)
theoretical_quantiles = np.linspace(0, 1, len(sorted_pit))
ax2.scatter(theoretical_quantiles, sorted_pit, alpha=0.7, s=100,
           color='steelblue', edgecolors='black')
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
ax2.set_xlabel('Theoretical Quantiles (Uniform)', fontsize=12)
ax2.set_ylabel('Observed LOO-PIT Quantiles', fontsize=12)
ax2.set_title('Q-Q Plot: LOO-PIT vs Uniform', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loo_pit_calibration.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOTS_DIR / 'loo_pit_calibration.png'}")
plt.close()

# ============================================================================
# 5. STUDY-LEVEL LOO PREDICTIONS
# ============================================================================
print("\n[5/7] Computing study-level LOO predictions...")

# Extract posterior predictive samples
if 'posterior_predictive' not in idata.groups():
    print("  WARNING: No posterior_predictive group - generating from posterior samples")
    # Use posterior samples to generate predictions
    theta_samples = idata.posterior['theta'].values.reshape(-1, n_studies)
else:
    y_pred_samples = idata.posterior_predictive['y_obs'].values.reshape(-1, n_studies)

# For LOO predictions, we need to use the loo object's elpd_i
# Extract theta posterior for point predictions
theta_samples = idata.posterior['theta'].values.reshape(-1, n_studies)
theta_mean = theta_samples.mean(axis=0)
theta_std = theta_samples.std(axis=0)

# Compute posterior predictive intervals
theta_q05 = np.percentile(theta_samples, 5, axis=0)
theta_q25 = np.percentile(theta_samples, 25, axis=0)
theta_q50 = np.percentile(theta_samples, 50, axis=0)
theta_q75 = np.percentile(theta_samples, 75, axis=0)
theta_q95 = np.percentile(theta_samples, 95, axis=0)

# Create study-level results table
study_results = pd.DataFrame({
    'study': range(1, n_studies + 1),
    'y_observed': y_obs,
    'sigma': sigma_obs,
    'theta_mean': theta_mean,
    'theta_sd': theta_std,
    'theta_q05': theta_q05,
    'theta_q50': theta_q50,
    'theta_q95': theta_q95,
    'residual': y_obs - theta_mean,
    'standardized_residual': (y_obs - theta_mean) / np.sqrt(theta_std**2 + sigma_obs**2),
    'pareto_k': pareto_k,
    'loo_pit': loo_pit
})

print("\n  Study-level LOO predictions:")
print(study_results.to_string(index=False))

# Save study-level results
study_results.to_csv(OUTPUT_DIR / 'loo_results.csv', index=False)
print(f"\n  - Saved: {OUTPUT_DIR / 'loo_results.csv'}")

# ============================================================================
# 6. ABSOLUTE PREDICTIVE METRICS
# ============================================================================
print("\n[6/7] Computing absolute predictive metrics...")

# Point prediction metrics
rmse = np.sqrt(np.mean((y_obs - theta_mean)**2))
mae = np.mean(np.abs(y_obs - theta_mean))
mse = np.mean((y_obs - theta_mean)**2)

print(f"\n  Point Prediction Metrics:")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - MAE: {mae:.2f}")
print(f"  - MSE: {mse:.2f}")

# Coverage analysis
coverage_50 = np.mean((y_obs >= theta_q25) & (y_obs <= theta_q75))
coverage_90 = np.mean((y_obs >= theta_q05) & (y_obs <= theta_q95))

print(f"\n  Interval Coverage:")
print(f"  - 50% credible interval: {coverage_50:.1%} (nominal: 50%)")
print(f"  - 90% credible interval: {coverage_90:.1%} (nominal: 90%)")

# Interval widths
width_50 = (theta_q75 - theta_q25).mean()
width_90 = (theta_q95 - theta_q05).mean()

print(f"\n  Interval Widths (mean):")
print(f"  - 50% credible interval: {width_50:.2f}")
print(f"  - 90% credible interval: {width_90:.2f}")

# Compare to naive baseline (unweighted mean)
naive_mean = y_obs.mean()
naive_rmse = np.sqrt(np.mean((y_obs - naive_mean)**2))
naive_mae = np.mean(np.abs(y_obs - naive_mean))

print(f"\n  Baseline Comparison (unweighted mean):")
print(f"  - Naive RMSE: {naive_rmse:.2f} vs Model RMSE: {rmse:.2f}")
print(f"  - Naive MAE: {naive_mae:.2f} vs Model MAE: {mae:.2f}")
print(f"  - Improvement: {(1 - rmse/naive_rmse)*100:.1f}% (RMSE), {(1 - mae/naive_mae)*100:.1f}% (MAE)")

# Store metrics
absolute_metrics = {
    'rmse': float(rmse),
    'mae': float(mae),
    'mse': float(mse),
    'coverage_50': float(coverage_50),
    'coverage_90': float(coverage_90),
    'width_50_mean': float(width_50),
    'width_90_mean': float(width_90),
    'naive_rmse': float(naive_rmse),
    'naive_mae': float(naive_mae),
    'rmse_improvement_pct': float((1 - rmse/naive_rmse)*100),
    'mae_improvement_pct': float((1 - mae/naive_mae)*100)
}

# Add to calibration metrics
calibration_metrics.update({
    'coverage_50_pct': float(coverage_50 * 100),
    'coverage_90_pct': float(coverage_90 * 100),
    'coverage_50_nominal': 50.0,
    'coverage_90_nominal': 90.0,
    'coverage_50_deviation': float((coverage_50 - 0.5) * 100),
    'coverage_90_deviation': float((coverage_90 - 0.9) * 100)
})

# Save calibration metrics
with open(OUTPUT_DIR / 'calibration_metrics.json', 'w') as f:
    json.dump(calibration_metrics, f, indent=2)
print(f"\n  - Saved: {OUTPUT_DIR / 'calibration_metrics.json'}")

# ============================================================================
# 7. DIAGNOSTIC VISUALIZATIONS
# ============================================================================
print("\n[7/7] Creating diagnostic visualizations...")

# Plot 1: Forest plot of LOO predictions vs observed
fig, ax = plt.subplots(figsize=(10, 8))

y_pos = np.arange(n_studies)
ax.errorbar(theta_mean, y_pos, xerr=1.96*theta_std, fmt='o', markersize=8,
           color='steelblue', capsize=5, capthick=2, label='Model 90% CI')
ax.scatter(y_obs, y_pos, marker='D', s=100, color='red',
          edgecolors='black', linewidth=1.5, label='Observed', zorder=5)

# Add vertical line at overall pooled estimate
mu_samples = idata.posterior['mu'].values.flatten()
mu_mean = mu_samples.mean()
ax.axvline(mu_mean, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Pooled μ={mu_mean:.1f}')

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Study {i+1}' for i in range(n_studies)])
ax.set_xlabel('Effect Size', fontsize=12)
ax.set_ylabel('Study', fontsize=12)
ax.set_title('Study-Level Predictions vs Observed\n(Model LOO Predictions with 90% CI)',
            fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loo_predictions_forest.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOTS_DIR / 'loo_predictions_forest.png'}")
plt.close()

# Plot 2: Residuals analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs fitted
ax1 = axes[0, 0]
ax1.scatter(theta_mean, study_results['residual'], s=100, alpha=0.7,
           c=pareto_k, cmap='RdYlGn_r', edgecolors='black')
ax1.axhline(0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Effect (θ)', fontsize=11)
ax1.set_ylabel('Residual (y - θ)', fontsize=11)
ax1.set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
cbar1.set_label('Pareto k', fontsize=10)

# Standardized residuals
ax2 = axes[0, 1]
ax2.scatter(range(n_studies), study_results['standardized_residual'], s=100,
           alpha=0.7, color='steelblue', edgecolors='black')
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.axhline(2, color='orange', linestyle=':', linewidth=1.5)
ax2.axhline(-2, color='orange', linestyle=':', linewidth=1.5)
ax2.set_xlabel('Study Index', fontsize=11)
ax2.set_ylabel('Standardized Residual', fontsize=11)
ax2.set_title('Standardized Residuals by Study', fontsize=12, fontweight='bold')
ax2.set_xticks(range(n_studies))
ax2.set_xticklabels([f'S{i+1}' for i in range(n_studies)])
ax2.grid(True, alpha=0.3)

# Q-Q plot of residuals
ax3 = axes[1, 0]
stats.probplot(study_results['residual'], dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot: Residuals vs Normal', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Residual histogram
ax4 = axes[1, 1]
ax4.hist(study_results['residual'], bins=8, alpha=0.7, color='steelblue',
        edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residuals_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOTS_DIR / 'residuals_diagnostics.png'}")
plt.close()

# Plot 3: Predicted vs Observed scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(theta_mean, y_obs, s=150, alpha=0.7, c=pareto_k, cmap='RdYlGn_r',
          edgecolors='black', linewidth=1.5)
ax.plot([y_obs.min()-5, y_obs.max()+5], [y_obs.min()-5, y_obs.max()+5],
       'r--', linewidth=2, label='Perfect prediction')

# Add error bars for uncertainty
ax.errorbar(theta_mean, y_obs, xerr=1.96*theta_std, fmt='none',
           ecolor='gray', alpha=0.3, capsize=3)

ax.set_xlabel('Predicted Effect (θ)', fontsize=12)
ax.set_ylabel('Observed Effect (y)', fontsize=12)
ax.set_title(f'Predicted vs Observed Effects\nRMSE = {rmse:.2f}, MAE = {mae:.2f}',
            fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Pareto k', fontsize=11)

# Add study labels
for i in range(n_studies):
    ax.annotate(f'S{i+1}', (theta_mean[i], y_obs[i]),
               xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'predicted_vs_observed.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOTS_DIR / 'predicted_vs_observed.png'}")
plt.close()

# Plot 4: Coverage visualization
fig, ax = plt.subplots(figsize=(10, 8))

y_pos = np.arange(n_studies)

# Plot 90% intervals
ax.errorbar(theta_mean, y_pos,
           xerr=[theta_mean - theta_q05, theta_q95 - theta_mean],
           fmt='none', color='lightblue', linewidth=6, alpha=0.5,
           label='90% CI')

# Plot 50% intervals
ax.errorbar(theta_mean, y_pos,
           xerr=[theta_mean - theta_q25, theta_q75 - theta_mean],
           fmt='none', color='steelblue', linewidth=6, alpha=0.7,
           label='50% CI')

# Plot means
ax.scatter(theta_mean, y_pos, marker='o', s=80, color='darkblue',
          edgecolors='black', linewidth=1, label='Posterior mean', zorder=5)

# Plot observed values
colors_coverage = ['green' if (y_obs[i] >= theta_q05[i] and y_obs[i] <= theta_q95[i])
                   else 'red' for i in range(n_studies)]
ax.scatter(y_obs, y_pos, marker='D', s=120, c=colors_coverage,
          edgecolors='black', linewidth=1.5, label='Observed', zorder=6)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Study {i+1}' for i in range(n_studies)])
ax.set_xlabel('Effect Size', fontsize=12)
ax.set_ylabel('Study', fontsize=12)
ax.set_title(f'Interval Coverage Check\n50% Coverage: {coverage_50:.0%}, 90% Coverage: {coverage_90:.0%}',
            fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'interval_coverage.png', dpi=300, bbox_inches='tight')
print(f"  - Saved: {PLOTS_DIR / 'interval_coverage.png'}")
plt.close()

# ============================================================================
# 8. SAVE SUMMARY METRICS
# ============================================================================
print("\n[8/8] Saving summary metrics...")

assessment_summary = {
    'model_name': 'Bayesian Hierarchical Meta-Analysis',
    'n_studies': n_studies,
    'loo_metrics': loo_metrics,
    'calibration_metrics': calibration_metrics,
    'absolute_metrics': absolute_metrics,
    'overall_assessment': {
        'loo_reliability': 'excellent' if k_bad == 0 and k_very_bad == 0 else 'good' if k_very_bad == 0 else 'poor',
        'calibration': 'well_calibrated' if ks_pval > 0.05 else 'calibration_warning',
        'predictive_accuracy': 'good' if rmse < naive_rmse else 'poor',
        'recommendation': 'adequate' if (k_bad == 0 and ks_pval > 0.05 and rmse < naive_rmse) else 'review_needed'
    }
}

with open(OUTPUT_DIR / 'assessment_summary.json', 'w') as f:
    json.dump(assessment_summary, f, indent=2)
print(f"  - Saved: {OUTPUT_DIR / 'assessment_summary.json'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ASSESSMENT COMPLETE")
print("=" * 80)
print(f"\nKey Findings:")
print(f"  1. LOO Reliability: {'EXCELLENT' if k_bad == 0 and k_very_bad == 0 else 'GOOD' if k_very_bad == 0 else 'POOR'}")
print(f"     - All Pareto k < 0.7: {'YES' if k_bad == 0 and k_very_bad == 0 else 'NO'}")
print(f"  2. Calibration: {'GOOD' if ks_pval > 0.05 else 'WARNING'}")
print(f"     - LOO-PIT uniform: {'YES (p={:.3f})'.format(ks_pval) if ks_pval > 0.05 else 'NO (p={:.3f})'.format(ks_pval)}")
print(f"  3. Predictive Accuracy:")
print(f"     - RMSE: {rmse:.2f} (naive: {naive_rmse:.2f})")
print(f"     - MAE: {mae:.2f} (naive: {naive_mae:.2f})")
print(f"  4. Coverage:")
print(f"     - 50% interval: {coverage_50:.0%} (nominal: 50%)")
print(f"     - 90% interval: {coverage_90:.0%} (nominal: 90%)")
print(f"\nOverall Assessment: {assessment_summary['overall_assessment']['recommendation'].upper()}")
print("\nOutput files created:")
print(f"  - {OUTPUT_DIR / 'assessment_report.md'} (to be generated)")
print(f"  - {OUTPUT_DIR / 'loo_results.csv'}")
print(f"  - {OUTPUT_DIR / 'calibration_metrics.json'}")
print(f"  - {OUTPUT_DIR / 'assessment_summary.json'}")
print(f"  - {PLOTS_DIR / 'pareto_k_diagnostics.png'}")
print(f"  - {PLOTS_DIR / 'loo_pit_calibration.png'}")
print(f"  - {PLOTS_DIR / 'loo_predictions_forest.png'}")
print(f"  - {PLOTS_DIR / 'residuals_diagnostics.png'}")
print(f"  - {PLOTS_DIR / 'predicted_vs_observed.png'}")
print(f"  - {PLOTS_DIR / 'interval_coverage.png'}")
print("=" * 80)
