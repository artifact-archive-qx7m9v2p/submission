"""
Comprehensive Assessment of Random Effects Logistic Regression Model
Phase 4: Model Assessment for Single Accepted Model
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit, logit
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
POSTERIOR_PATH = "/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf"
OUTPUT_DIR = "/workspace/experiments/model_assessment"

print("="*80)
print("PHASE 4: MODEL ASSESSMENT - Random Effects Logistic Regression")
print("="*80)

# Load data
print("\n1. Loading data and posterior...")
data = pd.read_csv(DATA_PATH)
print(f"   Data: {len(data)} groups, total n={data['n'].sum()}")

# Load posterior
idata = az.from_netcdf(POSTERIOR_PATH)
print(f"   Posterior loaded: {list(idata.groups())}")

# Verify log_likelihood exists
if 'log_likelihood' not in idata.groups():
    raise ValueError("log_likelihood group not found in InferenceData!")
print(f"   log_likelihood variables: {list(idata.log_likelihood.data_vars)}")
print(f"   observed_data variables: {list(idata.observed_data.data_vars)}")

# ============================================================================
# 2. LOO CROSS-VALIDATION DIAGNOSTICS
# ============================================================================
print("\n2. Computing LOO Cross-Validation...")

# Compute LOO
loo_result = az.loo(idata, pointwise=True)

print("\n   LOO Results:")
print(f"   ELPD_loo: {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
print(f"   p_loo: {loo_result.p_loo:.2f}")
print(f"   n_samples: {loo_result.n_samples}")
print(f"   n_data_points: {loo_result.n_data_points}")

# Pareto k diagnostics
pareto_k = loo_result.pareto_k.values
print(f"\n   Pareto k diagnostics:")
print(f"   All k < 0.5 (good): {np.sum(pareto_k < 0.5)} / {len(pareto_k)}")
print(f"   All k < 0.7 (ok): {np.sum(pareto_k < 0.7)} / {len(pareto_k)}")
print(f"   Any k > 0.7 (bad): {np.sum(pareto_k > 0.7)} / {len(pareto_k)}")
print(f"   Max k: {np.max(pareto_k):.3f}")
print(f"   Mean k: {np.mean(pareto_k):.3f}")

# Individual k values
print(f"\n   Pareto k by group:")
for i, k in enumerate(pareto_k):
    status = "BAD" if k > 0.7 else "OK" if k > 0.5 else "GOOD"
    print(f"   Group {i+1}: k = {k:.3f} ({status})")

# Plot Pareto k diagnostics
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_khat(loo_result, ax=ax, show_bins=True)
ax.set_title('Pareto k Diagnostic Values\n(k < 0.5: good, 0.5-0.7: ok, > 0.7: bad)', fontsize=12)
ax.set_xlabel('Data point', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/pareto_k_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"\n   Saved: {OUTPUT_DIR}/plots/pareto_k_diagnostics.png")
plt.close()

# ============================================================================
# 3. CALIBRATION ASSESSMENT - LOO-PIT
# ============================================================================
print("\n3. Computing LOO-PIT (Probability Integral Transform)...")

# Compute LOO-PIT using correct variable name
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LOO-PIT plot - ECDF
    az.plot_loo_pit(idata, y="y_obs", ecdf=True, ax=axes[0])
    axes[0].set_title('LOO-PIT ECDF\n(Should follow diagonal if well-calibrated)', fontsize=11)

    # LOO-PIT histogram
    az.plot_loo_pit(idata, y="y_obs", ecdf=False, ax=axes[1])
    axes[1].set_title('LOO-PIT Histogram\n(Should be uniform if well-calibrated)', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/loo_pit_calibration.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/plots/loo_pit_calibration.png")
    plt.close()

    loo_pit_computed = True
except Exception as e:
    print(f"   Warning: Could not compute LOO-PIT: {e}")
    loo_pit_computed = False

# ============================================================================
# 4. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================
print("\n4. Generating Posterior Predictive Samples...")

# Extract posterior samples
p = idata.posterior['p'].values  # probability parameter, shape (chains, draws, obs)
n_chains, n_draws, n_obs = p.shape
print(f"   Posterior shape: {n_chains} chains, {n_draws} draws, {n_obs} observations")

# Flatten chains
p_flat = p.reshape(-1, n_obs)  # (total_samples, n_obs)
print(f"   Flattened to: {p_flat.shape}")

# Get sample sizes
n_trials = data['n'].values

# Generate posterior predictive samples
print("   Generating posterior predictive samples...")
r_pred_samples = np.zeros_like(p_flat)
for i in range(n_obs):
    r_pred_samples[:, i] = np.random.binomial(n_trials[i], p_flat[:, i])

print(f"   Posterior predictive shape: {r_pred_samples.shape}")

# ============================================================================
# 5. ABSOLUTE PREDICTIVE METRICS
# ============================================================================
print("\n5. Computing Absolute Predictive Metrics...")

# Compute predictive statistics
r_pred_mean = r_pred_samples.mean(axis=0)
r_pred_std = r_pred_samples.std(axis=0)
r_pred_q05 = np.percentile(r_pred_samples, 5, axis=0)
r_pred_q95 = np.percentile(r_pred_samples, 95, axis=0)
r_pred_median = np.median(r_pred_samples, axis=0)

# Observed values
r_obs = idata.observed_data['y_obs'].values

# Compute metrics
residuals = r_obs - r_pred_mean
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))

# Coverage: proportion of observations within 90% interval
in_interval = (r_obs >= r_pred_q05) & (r_obs <= r_pred_q95)
coverage = np.mean(in_interval) * 100

print(f"\n   Predictive Metrics:")
print(f"   MAE: {mae:.2f} events")
print(f"   RMSE: {rmse:.2f} events")
print(f"   90% Coverage: {coverage:.1f}% ({np.sum(in_interval)}/{len(r_obs)} observations)")

# Relative metrics
mean_count = r_obs.mean()
max_count = r_obs.max()
relative_mae = (mae / mean_count) * 100 if mean_count > 0 else np.inf
relative_rmse = (rmse / mean_count) * 100 if mean_count > 0 else np.inf

print(f"   Mean observed count: {mean_count:.2f}")
print(f"   Relative MAE: {relative_mae:.1f}% of mean count")
print(f"   Relative RMSE: {relative_rmse:.1f}% of mean count")

# Prediction intervals
print(f"\n   Prediction Summary:")
for i in range(n_obs):
    in_int = "YES" if in_interval[i] else "NO"
    print(f"   Group {i+1}: obs={r_obs[i]:3.0f}, pred={r_pred_mean[i]:5.1f}, "
          f"90%CI=[{r_pred_q05[i]:5.1f}, {r_pred_q95[i]:5.1f}], in_interval={in_int}")

# ============================================================================
# 6. GROUP-LEVEL DIAGNOSTICS
# ============================================================================
print("\n6. Computing Group-Level Diagnostics...")

# Create diagnostic dataframe
diagnostics_df = pd.DataFrame({
    'group': data['group'],
    'n': data['n'],
    'r_obs': r_obs,
    'proportion_obs': data['proportion'],
    'r_pred_mean': r_pred_mean,
    'r_pred_median': r_pred_median,
    'r_pred_std': r_pred_std,
    'r_pred_q05': r_pred_q05,
    'r_pred_q95': r_pred_q95,
    'residual': residuals,
    'std_residual': residuals / r_pred_std,
    'abs_residual': np.abs(residuals),
    'in_90_interval': in_interval,
    'pareto_k': pareto_k
})

# Add LOO pointwise ELPD if available
if hasattr(loo_result, 'elpd_loo'):
    try:
        diagnostics_df['elpd_loo_i'] = loo_result.elpd_loo.values
    except:
        pass

print("\n   Group-Level Diagnostics Table:")
print(diagnostics_df[['group', 'n', 'r_obs', 'r_pred_mean', 'residual',
                      'pareto_k', 'in_90_interval']].to_string(index=False))

# Identify problematic groups
high_k_groups = diagnostics_df[diagnostics_df['pareto_k'] > 0.7]
if len(high_k_groups) > 0:
    print(f"\n   WARNING: {len(high_k_groups)} groups with high Pareto k (> 0.7):")
    print(high_k_groups[['group', 'n', 'r_obs', 'r_pred_mean', 'pareto_k']].to_string(index=False))
else:
    print("\n   All groups have acceptable Pareto k values (< 0.7)")

# Identify poorly covered groups
poorly_covered = diagnostics_df[~diagnostics_df['in_90_interval']]
if len(poorly_covered) > 0:
    print(f"\n   {len(poorly_covered)} groups outside 90% predictive interval:")
    print(poorly_covered[['group', 'n', 'r_obs', 'r_pred_mean',
                         'r_pred_q05', 'r_pred_q95']].to_string(index=False))
else:
    print("\n   All groups within 90% predictive intervals")

# Save diagnostics
diagnostics_df.to_csv(f"{OUTPUT_DIR}/group_diagnostics.csv", index=False)
print(f"\n   Saved: {OUTPUT_DIR}/group_diagnostics.csv")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n7. Creating Diagnostic Visualizations...")

# 7.1 Residual plot with influential points
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Residuals vs fitted
ax = axes[0, 0]
colors = ['red' if k > 0.7 else 'orange' if k > 0.5 else 'blue' for k in diagnostics_df['pareto_k']]
ax.scatter(diagnostics_df['r_pred_mean'], diagnostics_df['residual'],
           c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Predicted Count', fontsize=11)
ax.set_ylabel('Residual (Observed - Predicted)', fontsize=11)
ax.set_title('Residual Plot\n(Red: k>0.7, Orange: k>0.5, Blue: k<0.5)', fontsize=11)
ax.grid(True, alpha=0.3)

# Add group labels for high k
for _, row in diagnostics_df[diagnostics_df['pareto_k'] > 0.5].iterrows():
    ax.text(row['r_pred_mean'], row['residual'], f" G{row['group']}", fontsize=8)

# Standardized residuals
ax = axes[0, 1]
ax.scatter(diagnostics_df['r_pred_mean'], diagnostics_df['std_residual'],
           c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axhline(2, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(-2, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Predicted Count', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Standardized Residuals\n(Dashed lines at ±2σ)', fontsize=11)
ax.grid(True, alpha=0.3)

# Observed vs Predicted
ax = axes[1, 0]
ax.scatter(diagnostics_df['r_obs'], diagnostics_df['r_pred_mean'],
           c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
ax.plot([0, max(diagnostics_df['r_obs'])], [0, max(diagnostics_df['r_obs'])],
        'k--', linewidth=1, label='Perfect prediction')

# Add error bars for 90% intervals
ax.errorbar(diagnostics_df['r_obs'], diagnostics_df['r_pred_mean'],
            yerr=[diagnostics_df['r_pred_mean'] - diagnostics_df['r_pred_q05'],
                  diagnostics_df['r_pred_q95'] - diagnostics_df['r_pred_mean']],
            fmt='none', ecolor='gray', alpha=0.3, linewidth=1)

ax.set_xlabel('Observed Count', fontsize=11)
ax.set_ylabel('Predicted Count (mean ± 90% CI)', fontsize=11)
ax.set_title('Observed vs Predicted', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

# Add group labels for problematic predictions
for _, row in diagnostics_df[~diagnostics_df['in_90_interval']].iterrows():
    ax.text(row['r_obs'], row['r_pred_mean'], f" G{row['group']}", fontsize=8)

# Coverage assessment by group
ax = axes[1, 1]
x_pos = np.arange(len(diagnostics_df))
ax.bar(x_pos, diagnostics_df['r_obs'], alpha=0.6, label='Observed', color='steelblue')
ax.errorbar(x_pos, diagnostics_df['r_pred_mean'],
            yerr=[diagnostics_df['r_pred_mean'] - diagnostics_df['r_pred_q05'],
                  diagnostics_df['r_pred_q95'] - diagnostics_df['r_pred_mean']],
            fmt='o', color='red', alpha=0.8, linewidth=2,
            capsize=5, capthick=2, label='Predicted ± 90% CI', markersize=6)
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Coverage Assessment\n({coverage:.0f}% in interval)', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(diagnostics_df['group'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/residual_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/residual_diagnostics.png")
plt.close()

# 7.2 Pareto k vs observation characteristics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# k vs sample size
ax = axes[0]
ax.scatter(diagnostics_df['n'], diagnostics_df['pareto_k'],
           s=100, alpha=0.7, edgecolors='black', linewidth=0.5, c=colors)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='k=0.5 threshold')
ax.axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='k=0.7 threshold')
ax.set_xlabel('Sample Size (n)', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('Pareto k vs Sample Size', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

# Add labels for high k
for _, row in diagnostics_df[diagnostics_df['pareto_k'] > 0.7].iterrows():
    ax.text(row['n'], row['pareto_k'], f" G{row['group']}", fontsize=8)

# k vs observed count
ax = axes[1]
ax.scatter(diagnostics_df['r_obs'], diagnostics_df['pareto_k'],
           s=100, alpha=0.7, edgecolors='black', linewidth=0.5, c=colors)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Observed Count', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('Pareto k vs Observed Count', fontsize=11)
ax.grid(True, alpha=0.3)

# Add labels
for _, row in diagnostics_df[diagnostics_df['pareto_k'] > 0.7].iterrows():
    ax.text(row['r_obs'], row['pareto_k'], f" G{row['group']}", fontsize=8)

# k vs proportion
ax = axes[2]
ax.scatter(diagnostics_df['proportion_obs'], diagnostics_df['pareto_k'],
           s=100, alpha=0.7, edgecolors='black', linewidth=0.5, c=colors)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Observed Proportion', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('Pareto k vs Observed Proportion', fontsize=11)
ax.grid(True, alpha=0.3)

# Add labels
for _, row in diagnostics_df[diagnostics_df['pareto_k'] > 0.7].iterrows():
    ax.text(row['proportion_obs'], row['pareto_k'], f" G{row['group']}", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/pareto_k_analysis.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/pareto_k_analysis.png")
plt.close()

# 7.3 Predictive distribution for each group
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(n_obs):
    ax = axes[i]

    # Plot posterior predictive histogram
    ax.hist(r_pred_samples[:, i], bins=30, alpha=0.6, color='skyblue',
            edgecolor='black', density=True)

    # Add observed value
    ax.axvline(r_obs[i], color='red', linewidth=2, linestyle='--', label=f'Observed: {r_obs[i]}')

    # Add mean and intervals
    ax.axvline(r_pred_mean[i], color='blue', linewidth=2, label=f'Mean: {r_pred_mean[i]:.1f}')
    ax.axvline(r_pred_q05[i], color='green', linewidth=1, linestyle=':', alpha=0.7)
    ax.axvline(r_pred_q95[i], color='green', linewidth=1, linestyle=':', alpha=0.7,
               label=f'90% CI: [{r_pred_q05[i]:.1f}, {r_pred_q95[i]:.1f}]')

    # Title with Pareto k
    k_val = pareto_k[i]
    k_color = 'red' if k_val > 0.7 else 'orange' if k_val > 0.5 else 'black'
    ax.set_title(f'Group {i+1} (n={data.iloc[i]["n"]}, k={k_val:.3f})',
                 fontsize=10, color=k_color, fontweight='bold')
    ax.set_xlabel('Count', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/predictive_distributions.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/predictive_distributions.png")
plt.close()

# ============================================================================
# 8. COMPUTE WAIC AS ALTERNATIVE
# ============================================================================
print("\n8. Computing WAIC as Alternative to LOO...")

waic_result = az.waic(idata, pointwise=True)
print(f"\n   WAIC Results:")
print(f"   ELPD_waic: {waic_result.elpd_waic:.2f} ± {waic_result.se:.2f}")
print(f"   p_waic: {waic_result.p_waic:.2f}")

# Compare LOO and WAIC
print(f"\n   LOO vs WAIC:")
print(f"   ELPD difference: {loo_result.elpd_loo - waic_result.elpd_waic:.2f}")
print(f"   p difference: {loo_result.p_loo - waic_result.p_waic:.2f}")

# ============================================================================
# 9. SUMMARY METRICS
# ============================================================================
print("\n9. Saving Summary Metrics...")

metrics_summary = {
    'metric': [
        'elpd_loo', 'elpd_loo_se', 'p_loo',
        'elpd_waic', 'elpd_waic_se', 'p_waic',
        'mae', 'rmse', 'relative_mae_pct', 'relative_rmse_pct',
        'coverage_90_pct',
        'n_observations', 'total_n', 'total_r', 'mean_r',
        'pareto_k_max', 'pareto_k_mean', 'n_pareto_k_gt_0.5', 'n_pareto_k_gt_0.7',
        'n_outside_interval'
    ],
    'value': [
        loo_result.elpd_loo, loo_result.se, loo_result.p_loo,
        waic_result.elpd_waic, waic_result.se, waic_result.p_waic,
        mae, rmse, relative_mae, relative_rmse,
        coverage,
        len(data), data['n'].sum(), data['r'].sum(), mean_count,
        np.max(pareto_k), np.mean(pareto_k),
        np.sum(pareto_k > 0.5), np.sum(pareto_k > 0.7),
        len(poorly_covered)
    ]
}

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv", index=False)
print(f"   Saved: {OUTPUT_DIR}/metrics_summary.csv")
print("\n   Summary Metrics:")
print(metrics_df.to_string(index=False))

# ============================================================================
# 10. MODEL QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("MODEL QUALITY ASSESSMENT")
print("="*80)

# Criteria for quality assessment
loo_reliable = np.all(pareto_k < 0.7)
loo_mostly_reliable = np.sum(pareto_k < 0.7) >= 0.75 * len(pareto_k)  # 75% under threshold
calibration_good = coverage >= 85
predictive_accuracy_good = relative_mae < 50  # Within 50% of mean

print(f"\nCriteria:")
print(f"  LOO Reliable (all k < 0.7): {loo_reliable}")
print(f"  LOO Mostly Reliable (75%+ k < 0.7): {loo_mostly_reliable} ({np.sum(pareto_k < 0.7)}/{len(pareto_k)})")
print(f"  Calibration Good (coverage >= 85%): {calibration_good} ({coverage:.1f}%)")
print(f"  Predictive Accuracy Good (rel MAE < 50%): {predictive_accuracy_good} ({relative_mae:.1f}%)")

# Detailed analysis of high Pareto k
print(f"\nHigh Pareto k Analysis:")
print(f"  Groups with k > 0.7: {np.sum(pareto_k > 0.7)}")
print(f"  Groups with k > 0.5: {np.sum(pareto_k > 0.5)}")
print(f"  Mean k for all groups: {np.mean(pareto_k):.3f}")

# Determine overall quality
concerns = []

if not loo_reliable:
    concerns.append(f"High Pareto k values ({np.sum(pareto_k > 0.7)}/12 groups > 0.7) - LOO may be unreliable for some observations")

if not calibration_good:
    concerns.append(f"Coverage below 85% ({coverage:.1f}%) - model may not be well-calibrated")

if not predictive_accuracy_good:
    concerns.append(f"High relative MAE ({relative_mae:.1f}%) - predictions deviate substantially from observations")

# Overall assessment with nuance
if loo_reliable and calibration_good and predictive_accuracy_good:
    quality = "EXCELLENT"
    ready_phase5 = "YES"
    if not concerns:
        concerns.append("None - model shows excellent predictive performance and calibration")
elif calibration_good and predictive_accuracy_good:
    quality = "GOOD"
    ready_phase5 = "YES"
    concerns.insert(0, "Despite high Pareto k, predictive performance is good - proceed with caution")
elif loo_mostly_reliable or (calibration_good and not loo_reliable):
    quality = "ADEQUATE"
    ready_phase5 = "YES"
    concerns.insert(0, "Model has limitations but may be useful with awareness of issues")
else:
    quality = "POOR"
    ready_phase5 = "NO"
    concerns.insert(0, "Multiple serious issues - consider alternative modeling approach")

print(f"\nOVERALL ASSESSMENT:")
print(f"  Model Quality: {quality}")
print(f"  Ready for Phase 5: {ready_phase5}")
print(f"  Concerns:")
for concern in concerns:
    print(f"    - {concern}")

# Store for report
assessment_results = {
    'quality': quality,
    'ready_phase5': ready_phase5,
    'concerns': concerns,
    'loo_reliable': loo_reliable,
    'loo_mostly_reliable': loo_mostly_reliable,
    'calibration_good': calibration_good,
    'predictive_accuracy_good': predictive_accuracy_good,
    'n_high_pareto_k': int(np.sum(pareto_k > 0.7)),
    'mean_pareto_k': float(np.mean(pareto_k)),
    'coverage': float(coverage),
    'mae': float(mae),
    'rmse': float(rmse),
    'relative_mae': float(relative_mae),
    'elpd_loo': float(loo_result.elpd_loo),
    'elpd_loo_se': float(loo_result.se),
    'p_loo': float(loo_result.p_loo)
}

print("\n" + "="*80)
print("ASSESSMENT COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"  - plots/pareto_k_diagnostics.png")
print(f"  - plots/loo_pit_calibration.png" if loo_pit_computed else "  - plots/loo_pit_calibration.png (FAILED)")
print(f"  - plots/residual_diagnostics.png")
print(f"  - plots/pareto_k_analysis.png")
print(f"  - plots/predictive_distributions.png")
print(f"  - group_diagnostics.csv")
print(f"  - metrics_summary.csv")

# Save assessment results for report generation
import json
with open(f"{OUTPUT_DIR}/assessment_results.json", 'w') as f:
    json.dump(assessment_results, f, indent=2)
print(f"  - assessment_results.json")

print("\nNext: Generate comprehensive assessment_report.md")
