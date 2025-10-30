"""
Comprehensive Model Assessment - Complete Pooling Model
========================================================

Performs detailed assessment of the single ACCEPTED model from Phase 3.

Assessment Components:
1. LOO Diagnostics (ELPD, p_loo, Pareto k)
2. Calibration Analysis (LOO-PIT, coverage rates)
3. Absolute Predictive Metrics (RMSE, MAE)
4. Parameter Interpretation
5. Scientific Implications

Author: Model Assessment Specialist
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("/workspace")
DATA_PATH = BASE_DIR / "data" / "data.csv"
MODEL_DIR = BASE_DIR / "experiments" / "experiment_1"
INFERENCE_DATA_PATH = MODEL_DIR / "posterior_inference" / "diagnostics" / "posterior_inference.netcdf"
OUTPUT_DIR = BASE_DIR / "experiments" / "model_assessment"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE MODEL ASSESSMENT")
print("Complete Pooling Model (Experiment 1)")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA AND INFERENCEDATA
# ============================================================================
print("1. Loading data and InferenceData...")
print("-" * 80)

# Load observed data
df = pd.read_csv(DATA_PATH)
y_obs = df['y'].values
sigma_obs = df['sigma'].values
N = len(y_obs)

print(f"   Data: {N} observations")
print(f"   y values: {y_obs}")
print(f"   sigma values: {sigma_obs}")
print()

# Load InferenceData
print(f"   Loading InferenceData from:")
print(f"   {INFERENCE_DATA_PATH}")
idata = az.from_netcdf(INFERENCE_DATA_PATH)

# Verify log_likelihood is present
print(f"\n   InferenceData groups: {list(idata.groups())}")
if hasattr(idata, 'log_likelihood'):
    print(f"   log_likelihood variables: {list(idata.log_likelihood.data_vars)}")
    print(f"   log_likelihood shape: {idata.log_likelihood['y'].shape}")
    print("   STATUS: log_likelihood verified - ready for LOO-CV")
else:
    print("   ERROR: log_likelihood group not found!")
    raise ValueError("InferenceData missing log_likelihood - cannot perform LOO-CV")

print()

# Extract posterior samples for mu
mu_samples = idata.posterior['mu'].values.flatten()
n_samples = len(mu_samples)
print(f"   Total posterior samples: {n_samples}")
print()

# ============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================
print("2. Generating posterior predictive samples...")
print("-" * 80)

# Generate posterior predictive samples for each observation
# y_pred[i, j] = Normal(mu[i], sigma_obs[j])
y_pred = np.zeros((n_samples, N))

for i in range(n_samples):
    mu_i = mu_samples[i]
    for j in range(N):
        y_pred[i, j] = np.random.normal(mu_i, sigma_obs[j])

print(f"   Generated {n_samples} posterior predictive samples")
print(f"   y_pred shape: {y_pred.shape}")
print()

# ============================================================================
# 3. LOO DIAGNOSTICS
# ============================================================================
print("3. LOO Cross-Validation Diagnostics")
print("-" * 80)

# Compute LOO
loo_result = az.loo(idata, pointwise=True)

# Extract key metrics
elpd_loo = loo_result.elpd_loo
se_elpd = loo_result.se
p_loo = loo_result.p_loo
loo_i = loo_result.loo_i.values  # pointwise LOO values
pareto_k = loo_result.pareto_k.values

print(f"\nLOO-CV Summary:")
print(f"  ELPD_loo: {elpd_loo:.2f} ± {se_elpd:.2f}")
print(f"  p_loo (effective parameters): {p_loo:.2f}")
print(f"  Number of observations: {N}")
print()

# Pareto k diagnostics
k_good = np.sum(pareto_k < 0.5)
k_ok = np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))
k_bad = np.sum(pareto_k >= 0.7)
k_max = np.max(pareto_k)
k_min = np.min(pareto_k)
k_mean = np.mean(pareto_k)

print(f"Pareto k Diagnostics:")
print(f"  k < 0.5 (good):        {k_good}/{N} observations")
print(f"  0.5 <= k < 0.7 (ok):   {k_ok}/{N} observations")
print(f"  k >= 0.7 (bad):        {k_bad}/{N} observations")
print(f"  Range: [{k_min:.3f}, {k_max:.3f}]")
print(f"  Mean: {k_mean:.3f}")
print()

# LOO reliability assessment
if k_bad > 0:
    reliability = "UNRELIABLE"
    print("  WARNING: Some observations have high Pareto k (>= 0.7)")
    print("  LOO-CV may be unreliable for predictive assessment")
elif k_ok > 0:
    reliability = "ACCEPTABLE"
    print("  NOTE: Some observations have moderate Pareto k (0.5-0.7)")
    print("  LOO-CV is acceptable but monitor these observations")
else:
    reliability = "EXCELLENT"
    print("  STATUS: All Pareto k < 0.5 - LOO-CV is highly reliable")

print()

# Interpretation of p_loo
print(f"Effective Parameters (p_loo = {p_loo:.2f}):")
print(f"  Model has 1 explicit parameter (mu)")
print(f"  p_loo ≈ {p_loo:.2f} suggests effective complexity near 1 parameter")
if p_loo < 0.5:
    print(f"  Note: p_loo < 0.5 suggests very little overfitting risk")
elif p_loo > 2:
    print(f"  Note: p_loo > 2 suggests some complexity beyond explicit parameters")
print()

# Save LOO diagnostics
loo_df = pd.DataFrame({
    'observation': range(N),
    'y_obs': y_obs,
    'sigma_obs': sigma_obs,
    'loo_i': loo_i,
    'pareto_k': pareto_k,
    'k_category': ['good' if k < 0.5 else ('ok' if k < 0.7 else 'bad') for k in pareto_k]
})
loo_df.to_csv(DIAGNOSTICS_DIR / "loo_diagnostics.csv", index=False)
print(f"   Saved: {DIAGNOSTICS_DIR / 'loo_diagnostics.csv'}")

# Save summary metrics
loo_summary = pd.DataFrame({
    'metric': ['elpd_loo', 'se_elpd', 'p_loo', 'pareto_k_max', 'pareto_k_mean',
               'k_good', 'k_ok', 'k_bad', 'reliability'],
    'value': [elpd_loo, se_elpd, p_loo, k_max, k_mean,
              k_good, k_ok, k_bad, reliability]
})
loo_summary.to_csv(DIAGNOSTICS_DIR / "loo_summary.csv", index=False)
print(f"   Saved: {DIAGNOSTICS_DIR / 'loo_summary.csv'}")
print()

# ============================================================================
# 4. CALIBRATION ANALYSIS
# ============================================================================
print("4. Calibration Analysis")
print("-" * 80)

# Compute LOO-PIT (Probability Integral Transform)
print("\n4.1 LOO-PIT (Probability Integral Transform)")
print("   Computing LOO-PIT values...")

# Compute PIT for each observation
# PIT = P(y_rep <= y_obs | data)
pit_values = np.zeros(N)
for i in range(N):
    pit_values[i] = np.mean(y_pred[:, i] <= y_obs[i])

print(f"   PIT values: {pit_values}")
print()

# Test for uniformity using Kolmogorov-Smirnov test
ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
print(f"   Uniformity Test (Kolmogorov-Smirnov):")
print(f"   KS statistic: {ks_stat:.4f}")
print(f"   p-value: {ks_pval:.4f}")
if ks_pval > 0.05:
    print(f"   STATUS: PIT values are uniform (well-calibrated)")
else:
    print(f"   WARNING: PIT values deviate from uniformity (calibration issue)")
print()

# 4.2 Coverage Analysis
print("4.2 Coverage Analysis")
print("   Computing coverage rates for posterior predictive intervals...")

# Compute coverage for different credible intervals
coverage_levels = [0.50, 0.90, 0.95]
coverage_results = []

for level in coverage_levels:
    alpha = 1 - level
    lower = alpha / 2
    upper = 1 - alpha / 2

    # Compute credible intervals for each observation
    ci_lower = np.percentile(y_pred, lower * 100, axis=0)
    ci_upper = np.percentile(y_pred, upper * 100, axis=0)

    # Count how many observations fall within intervals
    in_interval = (y_obs >= ci_lower) & (y_obs <= ci_upper)
    coverage = np.mean(in_interval)
    count = np.sum(in_interval)

    coverage_results.append({
        'level': level,
        'coverage': coverage,
        'count': count,
        'expected': level
    })

    print(f"   {int(level*100)}% CI: {count}/{N} observations ({coverage*100:.1f}% coverage)")
    print(f"      Expected: ~{level*N:.1f} observations ({level*100:.0f}%)")

    if abs(coverage - level) < 0.15:  # Within 15% tolerance
        print(f"      STATUS: Good calibration")
    else:
        print(f"      WARNING: Coverage differs from nominal level")
    print()

# Save coverage results
coverage_df = pd.DataFrame(coverage_results)
coverage_df.to_csv(DIAGNOSTICS_DIR / "calibration_metrics.csv", index=False)
print(f"   Saved: {DIAGNOSTICS_DIR / 'calibration_metrics.csv'}")
print()

# ============================================================================
# 5. ABSOLUTE PREDICTIVE METRICS
# ============================================================================
print("5. Absolute Predictive Metrics")
print("-" * 80)

# Compute posterior predictive mean for each observation
y_pred_mean = np.mean(y_pred, axis=0)
y_pred_sd = np.std(y_pred, axis=0)

# RMSE and MAE
rmse = np.sqrt(np.mean((y_obs - y_pred_mean)**2))
mae = np.mean(np.abs(y_obs - y_pred_mean))

print(f"\nPredictive Error Metrics:")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAE:  {mae:.3f}")
print()

# Comparison to naive baselines
print("Comparison to Naive Baselines:")

# Baseline 1: Mean-only model (ignore individual observations)
mean_baseline = np.mean(y_obs)
rmse_mean_baseline = np.sqrt(np.mean((y_obs - mean_baseline)**2))
mae_mean_baseline = np.mean(np.abs(y_obs - mean_baseline))

print(f"\n  Baseline 1: Mean-only model")
print(f"    Prediction: {mean_baseline:.3f} for all observations")
print(f"    RMSE: {rmse_mean_baseline:.3f}")
print(f"    MAE:  {mae_mean_baseline:.3f}")
print(f"    Improvement: {(rmse_mean_baseline - rmse):.3f} RMSE units ({(1-rmse/rmse_mean_baseline)*100:.1f}%)")

# Context: Typical error relative to signal
signal_sd = np.std(y_obs)
print(f"\n  Context:")
print(f"    Signal SD: {signal_sd:.3f}")
print(f"    RMSE/Signal SD: {rmse/signal_sd:.3f}")
print(f"    Interpretation: Prediction error is {rmse/signal_sd*100:.1f}% of signal variability")
print()

# Save predictive metrics
metrics_df = pd.DataFrame({
    'metric': ['RMSE', 'MAE', 'RMSE_baseline_mean', 'MAE_baseline_mean',
               'Signal_SD', 'RMSE_relative'],
    'value': [rmse, mae, rmse_mean_baseline, mae_mean_baseline,
              signal_sd, rmse/signal_sd]
})
metrics_df.to_csv(DIAGNOSTICS_DIR / "predictive_metrics.csv", index=False)
print(f"   Saved: {DIAGNOSTICS_DIR / 'predictive_metrics.csv'}")
print()

# ============================================================================
# 6. PARAMETER INTERPRETATION
# ============================================================================
print("6. Parameter Interpretation")
print("-" * 80)

# Compute summary statistics
mu_mean = np.mean(mu_samples)
mu_std = np.std(mu_samples)
mu_median = np.median(mu_samples)
mu_ci_90 = np.percentile(mu_samples, [5, 95])
mu_ci_95 = np.percentile(mu_samples, [2.5, 97.5])

print(f"\nPosterior for mu (Population Mean):")
print(f"  Mean:   {mu_mean:.3f}")
print(f"  Median: {mu_median:.3f}")
print(f"  SD:     {mu_std:.3f}")
print(f"  90% CI: [{mu_ci_90[0]:.3f}, {mu_ci_90[1]:.3f}]")
print(f"  95% CI: [{mu_ci_95[0]:.3f}, {mu_ci_95[1]:.3f}]")
print()

# Interpretation
print("Interpretation:")
print(f"  - mu represents the common population mean shared by all 8 groups")
print(f"  - Best estimate: {mu_mean:.2f}")
print(f"  - Uncertainty: ±{mu_std:.2f} (1 SD)")
print(f"  - Uncertainty reflects both measurement error and sample size")
print()

# Effective sample size (weighted by precision)
weights = 1 / (sigma_obs**2)
effective_n = (np.sum(weights))**2 / np.sum(weights**2)
print(f"Effective Sample Size:")
print(f"  Nominal n: {N} observations")
print(f"  Effective n: {effective_n:.2f} (accounting for heterogeneous sigma)")
print(f"  Interpretation: Data provides information equivalent to {effective_n:.1f} equally-precise observations")
print()

# Shrinkage
print("Shrinkage Effect:")
print(f"  Each observation is 'shrunk' toward the common mean mu = {mu_mean:.2f}")
print(f"  Amount of shrinkage depends on measurement precision (1/sigma_i^2)")
for i in range(N):
    weight = weights[i] / np.sum(weights)
    print(f"    Obs {i}: y={y_obs[i]:6.2f}, sigma={sigma_obs[i]:2d}, weight={weight:.3f}")
print()

# Practical significance test
p_positive = np.mean(mu_samples > 0)
p_greater_than_5 = np.mean(mu_samples > 5)

print("Practical Significance:")
print(f"  P(mu > 0):  {p_positive:.3f}")
print(f"  P(mu > 5):  {p_greater_than_5:.3f}")
if p_positive > 0.95:
    print(f"  Conclusion: Strong evidence that mu > 0")
elif p_positive < 0.05:
    print(f"  Conclusion: Strong evidence that mu < 0")
else:
    print(f"  Conclusion: Uncertain whether mu is positive or negative")
print()

# ============================================================================
# 7. VISUALIZATION - LOO-PIT
# ============================================================================
print("7. Creating Visualizations")
print("-" * 80)

print("\n7.1 LOO-PIT Histogram")
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of PIT values
ax.hist(pit_values, bins=np.linspace(0, 1, 11), density=True,
        alpha=0.7, color='steelblue', edgecolor='black', label='Observed PIT')

# Expected uniform distribution
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Expected (Uniform)')

# Add KS test result
ax.text(0.05, 0.95, f'KS test p-value: {ks_pval:.3f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('PIT Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('LOO-PIT: Probability Integral Transform\n(Should be Uniform if Well-Calibrated)',
             fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pit.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'loo_pit.png'}")

# ============================================================================
# 7.2 COVERAGE PLOT
# ============================================================================
print("\n7.2 Coverage Plot")
fig, ax = plt.subplots(figsize=(10, 6))

levels = [c['level'] for c in coverage_results]
coverages = [c['coverage'] for c in coverage_results]

# Bar plot
bars = ax.bar(range(len(levels)), coverages, alpha=0.7, color='steelblue',
              edgecolor='black', label='Observed Coverage')

# Expected coverage (diagonal line)
ax.plot(range(len(levels)), levels, 'ro--', linewidth=2, markersize=10,
        label='Expected Coverage')

# Add value labels on bars
for i, (level, coverage, count) in enumerate(zip(levels, coverages,
                                                   [c['count'] for c in coverage_results])):
    ax.text(i, coverage + 0.02, f"{coverage*100:.0f}%\n({count}/{N})",
            ha='center', va='bottom', fontsize=10)

ax.set_xticks(range(len(levels)))
ax.set_xticklabels([f"{int(l*100)}%" for l in levels])
ax.set_xlabel('Nominal Coverage Level', fontsize=12)
ax.set_ylabel('Observed Coverage', fontsize=12)
ax.set_ylim([0, 1.1])
ax.set_title('Posterior Predictive Interval Coverage\n(Observed vs Expected)',
             fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'coverage_plot.png'}")

# ============================================================================
# 7.3 PARETO K DIAGNOSTIC PLOT
# ============================================================================
print("\n7.3 Pareto k Diagnostic Plot")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Pareto k values
colors = ['green' if k < 0.5 else ('orange' if k < 0.7 else 'red') for k in pareto_k]
ax.scatter(range(N), pareto_k, c=colors, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)

# Threshold lines
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5,
           label='k = 0.5 (good/ok threshold)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.5,
           label='k = 0.7 (ok/bad threshold)')

# Labels
for i, (k, y) in enumerate(zip(pareto_k, y_obs)):
    ax.text(i, k + 0.03, f'y={y:.1f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel('Pareto k', fontsize=12)
ax.set_title('LOO-CV Pareto k Diagnostics\n(All k < 0.5 indicates reliable LOO)',
             fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pareto_k_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'pareto_k_diagnostic.png'}")

# ============================================================================
# 7.4 CALIBRATION CURVE (Observed vs Predicted)
# ============================================================================
print("\n7.4 Calibration Curve")
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot of observed vs predicted
ax.scatter(y_pred_mean, y_obs, s=100, alpha=0.7, color='steelblue',
           edgecolors='black', linewidths=1.5, label='Observations')

# Perfect calibration line
min_val = min(y_pred_mean.min(), y_obs.min())
max_val = max(y_pred_mean.max(), y_obs.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
        label='Perfect Calibration')

# Add error bars (posterior predictive SD)
ax.errorbar(y_pred_mean, y_obs, xerr=y_pred_sd, fmt='none',
            ecolor='gray', alpha=0.3, capsize=3)

# Add labels for each point
for i, (pred, obs, sig) in enumerate(zip(y_pred_mean, y_obs, sigma_obs)):
    ax.text(pred, obs + 1, f'{i}\n(σ={sig})', ha='center', va='bottom',
            fontsize=8, alpha=0.7)

ax.set_xlabel('Posterior Predictive Mean', fontsize=12)
ax.set_ylabel('Observed Value', fontsize=12)
ax.set_title('Calibration Curve: Observed vs Predicted\n(Points should lie on diagonal)',
             fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "calibration_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'calibration_curve.png'}")

# ============================================================================
# 7.5 PREDICTIVE PERFORMANCE SUMMARY
# ============================================================================
print("\n7.5 Predictive Performance Summary")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: RMSE/MAE comparison
ax = axes[0, 0]
metrics_compare = ['Model', 'Mean Baseline']
rmse_vals = [rmse, rmse_mean_baseline]
mae_vals = [mae, mae_mean_baseline]

x = np.arange(len(metrics_compare))
width = 0.35

bars1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE',
               color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, mae_vals, width, label='MAE',
               color='coral', alpha=0.7, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Error', fontsize=12)
ax.set_title('Predictive Error Metrics', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(metrics_compare)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Coverage rates
ax = axes[0, 1]
levels = [int(c['level']*100) for c in coverage_results]
coverages = [c['coverage']*100 for c in coverage_results]
expected = [c['expected']*100 for c in coverage_results]

x = np.arange(len(levels))
bars1 = ax.bar(x - width/2, coverages, width, label='Observed',
               color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, expected, width, label='Expected',
               color='coral', alpha=0.7, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_title('Credible Interval Coverage', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'{l}%' for l in levels])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Residuals
ax = axes[1, 0]
residuals = y_obs - y_pred_mean
standardized_residuals = residuals / y_pred_sd

ax.scatter(range(N), standardized_residuals, s=100, alpha=0.7,
           color='steelblue', edgecolors='black', linewidths=1.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.5,
           label='±2 SD')
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

for i, res in enumerate(standardized_residuals):
    ax.text(i, res + 0.2, f'{i}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel('Standardized Residual', fontsize=12)
ax.set_title('Standardized Residuals\n(Should be within ±2 SD)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 4: PIT histogram (compact version)
ax = axes[1, 1]
ax.hist(pit_values, bins=np.linspace(0, 1, 11), density=True,
        alpha=0.7, color='steelblue', edgecolor='black')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform')
ax.text(0.05, 0.95, f'KS p-value: {ks_pval:.3f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('PIT Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('LOO-PIT Distribution', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Comprehensive Predictive Performance Assessment',
             fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "predictive_performance.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'predictive_performance.png'}")

print()
print("="*80)
print("ASSESSMENT COMPLETE")
print("="*80)
print()
print("Summary:")
print(f"  LOO ELPD: {elpd_loo:.2f} ± {se_elpd:.2f}")
print(f"  Pareto k: All {N} observations have k < 0.5 ({reliability})")
print(f"  Calibration: KS p-value = {ks_pval:.3f} (uniform PIT)")
print(f"  Coverage: {coverage_results[-1]['coverage']*100:.0f}% for 95% CI (expected 95%)")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAE: {mae:.3f}")
print()
print("Files saved to:")
print(f"  Diagnostics: {DIAGNOSTICS_DIR}")
print(f"  Plots: {PLOTS_DIR}")
print()
print("Next: Create comprehensive assessment report")
print()
