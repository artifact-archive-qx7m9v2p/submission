#!/usr/bin/env python3
"""
Posterior Predictive Checks for Experiment 1: Logarithmic Model with Normal Likelihood

Performs comprehensive validation by comparing observed data with posterior predictive
samples across multiple dimensions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
import json
from pathlib import Path

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
INFERENCE_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"
CODE_DIR = OUTPUT_DIR / "code"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("POSTERIOR PREDICTIVE CHECKS: LOGARITHMIC MODEL WITH NORMAL LIKELIHOOD")
print("="*80)

# ============================================================================
# 1. Load Data and Posterior Inference
# ============================================================================

print("\n1. Loading data and posterior samples...")

# Load observed data
data = pd.read_csv(DATA_PATH)
x_obs = data['x'].values
y_obs = data['Y'].values
n = len(y_obs)

print(f"   Observed data: n = {n} observations")
print(f"   x range: [{x_obs.min():.1f}, {x_obs.max():.1f}]")
print(f"   Y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")

# Load InferenceData
idata = az.from_netcdf(INFERENCE_PATH)

print(f"\n   InferenceData groups: {list(idata.groups())}")
print(f"   Posterior_predictive variables: {list(idata.posterior_predictive.data_vars)}")

# Get posterior predictive samples (variable is named 'Y')
y_rep = idata.posterior_predictive['Y'].values
print(f"   Posterior predictive shape: {y_rep.shape}")
# Reshape to (n_samples, n_obs)
n_chains, n_draws, n_obs = y_rep.shape
y_rep = y_rep.reshape(-1, n_obs)
n_samples = y_rep.shape[0]
print(f"   Total posterior predictive samples: {n_samples}")

# ============================================================================
# 2. Test Statistics: Calculate Posterior Predictive P-values
# ============================================================================

print("\n" + "="*80)
print("2. Computing Test Statistics and P-values")
print("="*80)

def compute_test_statistics(y_obs, y_rep):
    """
    Compute test statistics for observed and replicated data.

    Returns:
    --------
    stats_dict : dict
        Dictionary with test statistic name as key and tuple (T_obs, T_rep_array, p_value) as value
    """
    stats_dict = {}

    # Define test statistics
    test_funcs = {
        'Mean': np.mean,
        'Std Dev': np.std,
        'Minimum': np.min,
        'Maximum': np.max,
        'Range': lambda x: np.max(x) - np.min(x),
        'Skewness': lambda x: stats.skew(x),
        '10th Percentile': lambda x: np.percentile(x, 10),
        '90th Percentile': lambda x: np.percentile(x, 90),
        'Median': np.median,
        'IQR': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    }

    for name, func in test_funcs.items():
        # Observed statistic
        T_obs = func(y_obs)

        # Replicated statistics
        T_rep = np.array([func(y_rep[i, :]) for i in range(y_rep.shape[0])])

        # Posterior predictive p-value
        p_value = np.mean(T_rep >= T_obs)

        stats_dict[name] = (T_obs, T_rep, p_value)

    return stats_dict

# Compute statistics
stats_dict = compute_test_statistics(y_obs, y_rep)

# Create results table
print("\n   Test Statistics and Posterior Predictive P-values:")
print("   " + "-"*76)
print(f"   {'Statistic':<20} {'Observed':<12} {'Mean(Rep)':<12} {'P-value':<10} {'Flag':<10}")
print("   " + "-"*76)

results_table = []
for name, (T_obs, T_rep, p_val) in stats_dict.items():
    T_rep_mean = np.mean(T_rep)

    # Flag extreme p-values
    if p_val < 0.05 or p_val > 0.95:
        flag = "EXTREME"
    elif p_val < 0.10 or p_val > 0.90:
        flag = "Warning"
    else:
        flag = "OK"

    print(f"   {name:<20} {T_obs:<12.4f} {T_rep_mean:<12.4f} {p_val:<10.4f} {flag:<10}")

    results_table.append({
        'Statistic': name,
        'Observed': T_obs,
        'Mean_Replicated': T_rep_mean,
        'P_value': p_val,
        'Flag': flag
    })

print("   " + "-"*76)

# Save results
results_df = pd.DataFrame(results_table)
results_df.to_csv(OUTPUT_DIR / "test_statistics.csv", index=False)
print(f"\n   Test statistics saved to: {OUTPUT_DIR / 'test_statistics.csv'}")

# Summary of flags
n_extreme = sum(1 for r in results_table if r['Flag'] == 'EXTREME')
n_warning = sum(1 for r in results_table if r['Flag'] == 'Warning')
n_ok = sum(1 for r in results_table if r['Flag'] == 'OK')

print(f"\n   Summary: {n_ok} OK, {n_warning} Warnings, {n_extreme} EXTREME")

# ============================================================================
# 3. Visualization 1: PPC Density Overlay
# ============================================================================

print("\n" + "="*80)
print("3. Creating PPC Density Overlay Plot")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot 100 random posterior predictive samples
n_plot = min(100, y_rep.shape[0])
indices = np.random.choice(y_rep.shape[0], n_plot, replace=False)

for idx in indices:
    ax.hist(y_rep[idx, :], bins=20, alpha=0.02, color='skyblue', density=True, edgecolor='none')

# Plot observed data
ax.hist(y_obs, bins=20, alpha=0.7, color='darkred', density=True,
        edgecolor='black', linewidth=1.5, label='Observed')

ax.set_xlabel('Y', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Predictive Check: Density Overlay\n(100 replicated datasets vs observed)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_density_overlay.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOTS_DIR / 'ppc_density_overlay.png'}")

# ============================================================================
# 4. Visualization 2: Test Statistic Distributions
# ============================================================================

print("\n" + "="*80)
print("4. Creating Test Statistic Distribution Plots")
print("="*80)

# Select key statistics for visualization
key_stats = ['Mean', 'Std Dev', 'Minimum', 'Maximum', 'Skewness', 'Range']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, stat_name in enumerate(key_stats):
    ax = axes[idx]
    T_obs, T_rep, p_val = stats_dict[stat_name]

    # Histogram of replicated statistics
    ax.hist(T_rep, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

    # Mark observed statistic
    ax.axvline(T_obs, color='darkred', linewidth=3, label=f'Observed (p={p_val:.3f})')

    # Add shaded regions for extreme p-values
    percentile_5 = np.percentile(T_rep, 5)
    percentile_95 = np.percentile(T_rep, 95)

    y_lim = ax.get_ylim()
    if T_obs < percentile_5:
        ax.axvspan(ax.get_xlim()[0], percentile_5, alpha=0.2, color='red')
    if T_obs > percentile_95:
        ax.axvspan(percentile_95, ax.get_xlim()[1], alpha=0.2, color='red')

    ax.set_xlabel(stat_name, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{stat_name}: T(obs) = {T_obs:.3f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Test Statistic Distributions\n(Red line = observed, shaded = extreme regions)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "test_statistic_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOTS_DIR / 'test_statistic_distributions.png'}")

# ============================================================================
# 5. Visualization 3: Residual Patterns
# ============================================================================

print("\n" + "="*80)
print("5. Creating Residual Pattern Plots")
print("="*80)

# Get fitted values and residuals
# Note: Variable names are beta_0, beta_1, sigma (with underscores)
beta0_samples = idata.posterior['beta_0'].values.reshape(-1)
beta1_samples = idata.posterior['beta_1'].values.reshape(-1)
sigma_samples = idata.posterior['sigma'].values.reshape(-1)

# Posterior mean parameters
beta0_mean = np.mean(beta0_samples)
beta1_mean = np.mean(beta1_samples)
sigma_mean = np.mean(sigma_samples)

# Fitted values (posterior mean)
y_fit = beta0_mean + beta1_mean * np.log(x_obs)

# Residuals
residuals = y_obs - y_fit

# Standardized residuals
std_residuals = residuals / sigma_mean

print(f"   Mean fitted value: {y_fit.mean():.3f}")
print(f"   Mean residual: {residuals.mean():.6f}")
print(f"   Std residual: {residuals.std():.3f}")

# Create residual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(y_fit, residuals, alpha=0.7, s=80, edgecolors='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 2. Residuals vs X
ax = axes[0, 1]
ax.scatter(x_obs, residuals, alpha=0.7, s=80, edgecolors='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('X (predictor)', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('Residuals vs X', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. Scale-Location Plot
ax = axes[1, 0]
ax.scatter(y_fit, np.sqrt(np.abs(std_residuals)), alpha=0.7, s=80, edgecolors='black')
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('√|Standardized Residuals|', fontsize=11)
ax.set_title('Scale-Location Plot\n(Check Homoscedasticity)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. Q-Q Plot
ax = axes[1, 1]
stats.probplot(std_residuals, dist="norm", plot=ax)
ax.set_title('Normal Q-Q Plot\n(Standardized Residuals)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_patterns.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOTS_DIR / 'residual_patterns.png'}")

# ============================================================================
# 6. Visualization 4: Individual Predictions with Intervals
# ============================================================================

print("\n" + "="*80)
print("6. Creating Individual Predictions Plot")
print("="*80)

# Calculate prediction intervals for each observation
y_pred_mean = np.mean(y_rep, axis=0)
y_pred_lower = np.percentile(y_rep, 2.5, axis=0)
y_pred_upper = np.percentile(y_rep, 97.5, axis=0)

# Check coverage
in_interval = (y_obs >= y_pred_lower) & (y_obs <= y_pred_upper)
coverage = np.mean(in_interval)

print(f"   95% Predictive Interval Coverage: {coverage:.1%}")
print(f"   Observations in interval: {in_interval.sum()}/{n}")

fig, ax = plt.subplots(figsize=(12, 7))

# Sort by x for better visualization
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]
y_obs_sorted = y_obs[sort_idx]
y_pred_mean_sorted = y_pred_mean[sort_idx]
y_pred_lower_sorted = y_pred_lower[sort_idx]
y_pred_upper_sorted = y_pred_upper[sort_idx]
in_interval_sorted = in_interval[sort_idx]

# Plot prediction intervals
ax.fill_between(range(n), y_pred_lower_sorted, y_pred_upper_sorted,
                alpha=0.3, color='skyblue', label='95% Predictive Interval')

# Plot predictions
ax.plot(range(n), y_pred_mean_sorted, 'b-', linewidth=2, label='Posterior Mean Prediction')

# Plot observed data (color by coverage)
colors = ['green' if in_int else 'red' for in_int in in_interval_sorted]
ax.scatter(range(n), y_obs_sorted, c=colors, s=100, edgecolors='black',
          linewidth=1.5, zorder=5, label='Observed (green=in interval)')

ax.set_xlabel('Observation (sorted by x)', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title(f'Individual Predictions with 95% Intervals\nCoverage: {coverage:.1%}',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "individual_predictions.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOTS_DIR / 'individual_predictions.png'}")

# ============================================================================
# 7. Visualization 5: LOO-PIT Calibration
# ============================================================================

print("\n" + "="*80)
print("7. Creating LOO-PIT Calibration Plot")
print("="*80)

# Use ArviZ's built-in LOO-PIT functionality
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_loo_pit(idata, y='Y', ecdf=True, ax=ax)
ax.set_title('LOO Probability Integral Transform (PIT)\n(Should be uniform for well-calibrated model)',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pit_calibration.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOTS_DIR / 'loo_pit_calibration.png'}")

# ============================================================================
# 8. Visualization 6: Q-Q Plot (Observed vs Predicted Quantiles)
# ============================================================================

print("\n" + "="*80)
print("8. Creating Quantile-Quantile Comparison Plot")
print("="*80)

# Calculate quantiles
quantiles = np.linspace(0, 100, 50)
obs_quantiles = np.percentile(y_obs, quantiles)
pred_quantiles = np.percentile(y_rep.mean(axis=0), quantiles)

fig, ax = plt.subplots(figsize=(8, 8))

# Plot Q-Q
ax.scatter(obs_quantiles, pred_quantiles, s=80, alpha=0.7, edgecolors='black')

# Reference line
min_val = min(obs_quantiles.min(), pred_quantiles.min())
max_val = max(obs_quantiles.max(), pred_quantiles.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')

ax.set_xlabel('Observed Quantiles', fontsize=12)
ax.set_ylabel('Predicted Quantiles', fontsize=12)
ax.set_title('Quantile-Quantile Plot\n(Observed vs Predicted)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "qq_observed_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOTS_DIR / 'qq_observed_vs_predicted.png'}")

# ============================================================================
# 9. Visualization 7: Fitted Curve with Posterior Predictive Envelope
# ============================================================================

print("\n" + "="*80)
print("9. Creating Fitted Curve with Predictive Envelope")
print("="*80)

# Generate smooth x values for plotting
x_smooth = np.linspace(x_obs.min(), x_obs.max(), 200)
log_x_smooth = np.log(x_smooth)

# Calculate posterior predictive for smooth x
n_posterior_samples = min(1000, beta0_samples.shape[0])
y_smooth_samples = np.zeros((n_posterior_samples, len(x_smooth)))

for i in range(n_posterior_samples):
    mu = beta0_samples[i] + beta1_samples[i] * log_x_smooth
    y_smooth_samples[i, :] = np.random.normal(mu, sigma_samples[i])

# Calculate mean and intervals
y_smooth_mean = np.mean(y_smooth_samples, axis=0)
y_smooth_lower = np.percentile(y_smooth_samples, 2.5, axis=0)
y_smooth_upper = np.percentile(y_smooth_samples, 97.5, axis=0)

# Also calculate mean function (no noise)
mu_smooth_mean = beta0_mean + beta1_mean * log_x_smooth

fig, ax = plt.subplots(figsize=(12, 7))

# Plot predictive envelope
ax.fill_between(x_smooth, y_smooth_lower, y_smooth_upper,
               alpha=0.3, color='skyblue', label='95% Posterior Predictive')

# Plot mean function
ax.plot(x_smooth, mu_smooth_mean, 'b-', linewidth=2.5, label='Posterior Mean Function')

# Plot observed data
ax.scatter(x_obs, y_obs, s=100, alpha=0.7, edgecolors='black',
          linewidth=1.5, zorder=5, label='Observed Data', color='darkred')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Fitted Logarithmic Curve with Posterior Predictive Envelope',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "fitted_curve_with_envelope.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   Saved: {PLOTS_DIR / 'fitted_curve_with_envelope.png'}")

# ============================================================================
# 10. Summary Statistics and Assessment
# ============================================================================

print("\n" + "="*80)
print("10. Overall Model Adequacy Assessment")
print("="*80)

# Count flags
assessment = {
    'n_observations': n,
    'n_posterior_samples': n_samples,
    'test_statistics': {
        'n_extreme': n_extreme,
        'n_warning': n_warning,
        'n_ok': n_ok,
        'extreme_stats': [r['Statistic'] for r in results_table if r['Flag'] == 'EXTREME']
    },
    'predictive_interval_coverage': float(coverage),
    'expected_coverage': 0.95,
    "coverage_ok": bool(abs(coverage - 0.95) < 0.10)
}

# Overall assessment
if n_extreme == 0 and coverage > 0.85:
    assessment['overall_status'] = 'PASS'
    assessment['message'] = 'Model adequately captures key features of observed data'
elif n_extreme <= 1 and coverage > 0.80:
    assessment['overall_status'] = 'PASS_WITH_WARNINGS'
    assessment['message'] = 'Model shows minor discrepancies but generally adequate'
elif n_extreme <= 2:
    assessment['overall_status'] = 'MARGINAL'
    assessment['message'] = 'Model has notable deficiencies that should be addressed'
else:
    assessment['overall_status'] = 'FAIL'
    assessment['message'] = 'Model shows systematic misfit and may not be adequate'

# Save assessment
with open(OUTPUT_DIR / "ppc_assessment.json", 'w') as f:
    json.dump(assessment, f, indent=2)

print(f"\n   Overall Status: {assessment['overall_status']}")
print(f"   {assessment['message']}")
print(f"\n   Test Statistics: {n_ok} OK, {n_warning} Warning, {n_extreme} EXTREME")
print(f"   Predictive Coverage: {coverage:.1%} (expected 95%)")
print(f"\n   Assessment saved to: {OUTPUT_DIR / 'ppc_assessment.json'}")

# ============================================================================
# 11. Specific Model Checks
# ============================================================================

print("\n" + "="*80)
print("11. Specific Model Checks")
print("="*80)

print("\n   a) Functional Form Check:")
print("      - Logarithmic model assumes saturation pattern")
print("      - Visual inspection needed from fitted_curve_with_envelope.png")

print("\n   b) Homoscedasticity Check:")
# Check if residual variance changes with fitted values
# Divide data into thirds
sorted_fit_idx = np.argsort(y_fit)
n_third = n // 3
low_resid = residuals[sorted_fit_idx[:n_third]]
mid_resid = residuals[sorted_fit_idx[n_third:2*n_third]]
high_resid = residuals[sorted_fit_idx[2*n_third:]]

var_low = np.var(low_resid)
var_mid = np.var(mid_resid)
var_high = np.var(high_resid)

print(f"      - Variance in low fitted values: {var_low:.6f}")
print(f"      - Variance in mid fitted values: {var_mid:.6f}")
print(f"      - Variance in high fitted values: {var_high:.6f}")
print(f"      - Ratio (high/low): {var_high/var_low:.2f}")

if var_high/var_low > 2.0 or var_low/var_high > 2.0:
    print("      - WARNING: Possible heteroscedasticity detected")
else:
    print("      - OK: Residual variance relatively constant")

print("\n   c) Outlier Check:")
# Identify potential outliers (|std_residual| > 2.5)
outliers = np.abs(std_residuals) > 2.5
n_outliers = outliers.sum()
print(f"      - Observations with |standardized residual| > 2.5: {n_outliers}")
if n_outliers > 0:
    outlier_idx = np.where(outliers)[0]
    for idx in outlier_idx:
        print(f"        * Observation {idx}: x={x_obs[idx]:.1f}, Y={y_obs[idx]:.2f}, "
              f"fitted={y_fit[idx]:.2f}, std_resid={std_residuals[idx]:.2f}")

print("\n   d) Systematic Deviation Check:")
# Check for regions of consistent over/under-prediction
n_neg = (residuals < 0).sum()
n_pos = (residuals > 0).sum()
print(f"      - Negative residuals: {n_neg}/{n}")
print(f"      - Positive residuals: {n_pos}/{n}")
print(f"      - Balance: {abs(n_neg - n_pos)} difference")

if abs(n_neg - n_pos) > n * 0.3:
    print("      - WARNING: Systematic bias detected")
else:
    print("      - OK: Residuals roughly balanced")

# ============================================================================
# 12. Comparison to EDA
# ============================================================================

print("\n" + "="*80)
print("12. Comparison to EDA Findings")
print("="*80)

# EDA R² was 0.897, model R² is 0.889
print("   From inference_summary.md:")
print("   - Model R² = 0.889")
print("   - Model RMSE = 0.087")
print("\n   This will be compared to EDA R² (0.897) in the findings document")

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("="*80)
print(f"\nOutputs saved to:")
print(f"  - Plots: {PLOTS_DIR}")
print(f"  - Test statistics: {OUTPUT_DIR / 'test_statistics.csv'}")
print(f"  - Assessment: {OUTPUT_DIR / 'ppc_assessment.json'}")
print("\nNext step: Review plots and create ppc_findings.md")
print("="*80)
