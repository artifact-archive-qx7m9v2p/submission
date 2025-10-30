"""
Comprehensive Posterior Predictive Checks
Experiment 1: Negative Binomial State-Space Model

This script performs systematic validation of the fitted model by:
1. Loading posterior samples and generating predictive replications
2. Computing test statistics for key data features
3. Creating diagnostic visualizations
4. Assessing model adequacy
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = "/workspace/data/data.csv"
INFERENCE_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
PLOT_DIR = Path("/workspace/experiments/experiment_1/posterior_predictive_check/plots")

# Custom ACF function
def compute_acf(x, nlags=15):
    """Compute autocorrelation function"""
    x = np.asarray(x)
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)
    acf = np.array([1.] + [np.dot(x[:-i], x[i:]) / len(x) / c0 for i in range(1, nlags + 1)])
    return acf

# Load data
print("Loading observed data...")
data = pd.read_csv(DATA_PATH)
y_obs = data['C'].values
T = len(y_obs)
time_idx = np.arange(T)

# Load InferenceData
print("Loading InferenceData...")
idata = az.from_netcdf(INFERENCE_PATH)

# Check what's available
print("\nInferenceData groups:", idata.groups())
print("\nPosterior variables:", list(idata.posterior.data_vars))
if hasattr(idata, 'posterior_predictive'):
    print("Posterior predictive variables:", list(idata.posterior_predictive.data_vars))

# Extract posterior predictive samples
if hasattr(idata, 'posterior_predictive') and 'C' in idata.posterior_predictive:
    print("\nUsing existing posterior predictive samples...")
    y_pred = idata.posterior_predictive['C'].values  # Shape: (chains, draws, T)
    n_chains, n_draws, T_pred = y_pred.shape
    y_pred_flat = y_pred.reshape(-1, T_pred)  # Shape: (chains*draws, T)
    n_samples = y_pred_flat.shape[0]
    print(f"Posterior predictive shape: {y_pred.shape}")
    print(f"Total samples: {n_samples}")
else:
    print("\nNo posterior predictive samples found!")
    print("Available groups:", idata.groups())
    raise ValueError("Cannot proceed without posterior predictive samples")

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECK: SUMMARY STATISTICS")
print("="*80)

# ============================================================================
# A. SUMMARY STATISTICS CHECKS
# ============================================================================

def compute_test_statistics(y):
    """Compute test statistics for a single replication"""
    return {
        'mean': np.mean(y),
        'sd': np.std(y, ddof=1),
        'min': np.min(y),
        'max': np.max(y),
        'q25': np.percentile(y, 25),
        'q50': np.percentile(y, 50),
        'q75': np.percentile(y, 75),
        'var_mean_ratio': np.var(y, ddof=1) / np.mean(y) if np.mean(y) > 0 else np.nan,
        'growth_factor': y[-1] / y[0] if y[0] > 0 else np.nan,
        'acf_lag1': np.corrcoef(y[:-1], y[1:])[0, 1] if len(y) > 1 else np.nan
    }

# Compute test statistics for observed data
obs_stats = compute_test_statistics(y_obs)

print("\nObserved Data Statistics:")
print(f"  Mean: {obs_stats['mean']:.2f}")
print(f"  SD: {obs_stats['sd']:.2f}")
print(f"  Range: [{obs_stats['min']}, {obs_stats['max']}]")
print(f"  Quantiles (Q1, Q2, Q3): {obs_stats['q25']:.1f}, {obs_stats['q50']:.1f}, {obs_stats['q75']:.1f}")
print(f"  Var/Mean Ratio: {obs_stats['var_mean_ratio']:.2f}")
print(f"  Growth Factor: {obs_stats['growth_factor']:.2f}x")
print(f"  ACF(1): {obs_stats['acf_lag1']:.3f}")

# Compute test statistics for all posterior predictive replications
print("\nComputing test statistics for posterior predictive replications...")
pred_stats = {key: [] for key in obs_stats.keys()}

for i in range(n_samples):
    stats_i = compute_test_statistics(y_pred_flat[i, :])
    for key in pred_stats.keys():
        pred_stats[key].append(stats_i[key])

# Convert to arrays
for key in pred_stats.keys():
    pred_stats[key] = np.array(pred_stats[key])

print("\nPosterior Predictive Statistics (Mean ± SD):")
print(f"  Mean: {np.mean(pred_stats['mean']):.2f} ± {np.std(pred_stats['mean']):.2f}")
print(f"  SD: {np.mean(pred_stats['sd']):.2f} ± {np.std(pred_stats['sd']):.2f}")
print(f"  Max: {np.mean(pred_stats['max']):.2f} ± {np.std(pred_stats['max']):.2f}")
print(f"  Var/Mean Ratio: {np.mean(pred_stats['var_mean_ratio']):.2f} ± {np.std(pred_stats['var_mean_ratio']):.2f}")
print(f"  Growth Factor: {np.mean(pred_stats['growth_factor']):.2f}x ± {np.std(pred_stats['growth_factor']):.2f}x")
print(f"  ACF(1): {np.mean(pred_stats['acf_lag1']):.3f} ± {np.std(pred_stats['acf_lag1']):.3f}")

# Compute p-values (proportion of replications more extreme than observed)
print("\nBayesian p-values (2-tailed):")
for key in pred_stats.keys():
    obs_val = obs_stats[key]
    pred_vals = pred_stats[key]
    # Two-tailed: proportion of samples more extreme than observed
    p_value = np.mean(np.abs(pred_vals - np.mean(pred_vals)) > np.abs(obs_val - np.mean(pred_vals)))
    print(f"  {key}: p = {p_value:.3f}")

# ============================================================================
# B. VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING DIAGNOSTIC VISUALIZATIONS")
print("="*80)

# Plot 1: Test Statistics Distribution
print("\nPlot 1: Test statistics distribution...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

test_stat_names = ['mean', 'sd', 'max', 'var_mean_ratio', 'growth_factor', 'acf_lag1']
test_stat_labels = ['Mean', 'Standard Deviation', 'Maximum', 'Var/Mean Ratio', 'Growth Factor', 'ACF(1)']

for i, (name, label) in enumerate(zip(test_stat_names, test_stat_labels)):
    ax = axes[i]

    # Histogram of posterior predictive
    ax.hist(pred_stats[name], bins=50, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5, density=True)

    # Observed value
    obs_val = obs_stats[name]
    ax.axvline(obs_val, color='red', linewidth=2, label='Observed', linestyle='--')

    # Posterior predictive mean
    pred_mean = np.mean(pred_stats[name])
    ax.axvline(pred_mean, color='blue', linewidth=2, label='Pred Mean', linestyle=':')

    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.set_title(f'{label}\n(obs={obs_val:.2f}, pred={pred_mean:.2f})')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "test_statistics_distribution.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOT_DIR / 'test_statistics_distribution.png'}")
plt.close()

# Plot 2: PPC Time Series with Envelope
print("\nPlot 2: Posterior predictive time series envelope...")
fig, ax = plt.subplots(figsize=(14, 6))

# Compute predictive quantiles
pred_mean = np.mean(y_pred_flat, axis=0)
pred_median = np.median(y_pred_flat, axis=0)
pred_q05 = np.percentile(y_pred_flat, 5, axis=0)
pred_q25 = np.percentile(y_pred_flat, 25, axis=0)
pred_q75 = np.percentile(y_pred_flat, 75, axis=0)
pred_q95 = np.percentile(y_pred_flat, 95, axis=0)

# Plot predictive intervals
ax.fill_between(time_idx, pred_q05, pred_q95, alpha=0.2, color='steelblue', label='90% Pred Interval')
ax.fill_between(time_idx, pred_q25, pred_q75, alpha=0.3, color='steelblue', label='50% Pred Interval')
ax.plot(time_idx, pred_median, 'b-', linewidth=2, label='Pred Median', alpha=0.7)
ax.plot(time_idx, pred_mean, 'b--', linewidth=2, label='Pred Mean', alpha=0.7)

# Plot observed
ax.plot(time_idx, y_obs, 'ro-', linewidth=2, markersize=6, label='Observed', alpha=0.8)

ax.set_xlabel('Time Index')
ax.set_ylabel('Count')
ax.set_title('Posterior Predictive Check: Time Series with 50% and 90% Intervals')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "ppc_time_series_envelope.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOT_DIR / 'ppc_time_series_envelope.png'}")
plt.close()

# Plot 3: PPC Density Overlay
print("\nPlot 3: Posterior predictive density overlay...")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot 100 random replications
n_lines = min(100, n_samples)
sample_indices = np.random.choice(n_samples, n_lines, replace=False)

for idx in sample_indices:
    ax.hist(y_pred_flat[idx, :], bins=30, alpha=0.01, color='steelblue',
            density=True, edgecolor='none')

# Plot observed
ax.hist(y_obs, bins=30, alpha=0.5, color='red', density=True,
        edgecolor='black', linewidth=1.5, label='Observed')

# Plot mean of predictive distributions
y_pred_all = y_pred_flat.flatten()
ax.hist(y_pred_all, bins=50, alpha=0.3, color='blue', density=True,
        edgecolor='none', label='Predictive (pooled)')

ax.set_xlabel('Count')
ax.set_ylabel('Density')
ax.set_title('Posterior Predictive Check: Distribution Overlay')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "ppc_density_overlay.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOT_DIR / 'ppc_density_overlay.png'}")
plt.close()

# Plot 4: Residuals Analysis
print("\nPlot 4: Residuals analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Compute residuals (observed - predicted mean)
residuals = y_obs - pred_mean
std_residuals = residuals / np.std(y_pred_flat, axis=0)  # Standardized residuals

# A) Residuals over time
ax = axes[0, 0]
ax.plot(time_idx, residuals, 'ko-', markersize=6, linewidth=1.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.fill_between(time_idx, -2*np.std(y_pred_flat, axis=0), 2*np.std(y_pred_flat, axis=0),
                alpha=0.2, color='gray', label='±2 Pred SD')
ax.set_xlabel('Time Index')
ax.set_ylabel('Residual (Obs - Pred Mean)')
ax.set_title('Residuals Over Time')
ax.legend()
ax.grid(alpha=0.3)

# B) Standardized residuals over time
ax = axes[0, 1]
ax.plot(time_idx, std_residuals, 'ko-', markersize=6, linewidth=1.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time Index')
ax.set_ylabel('Standardized Residual')
ax.set_title('Standardized Residuals Over Time')
ax.grid(alpha=0.3)

# C) Q-Q plot
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Residuals vs Normal')
ax.grid(alpha=0.3)

# D) Residuals vs Predicted
ax = axes[1, 1]
ax.scatter(pred_mean, residuals, s=50, alpha=0.7, edgecolors='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Predicted Mean')
ax.set_ylabel('Residual')
ax.set_title('Residuals vs Predicted Values')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "residuals_analysis.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOT_DIR / 'residuals_analysis.png'}")
plt.close()

# Plot 5: Coverage Analysis
print("\nPlot 5: Coverage analysis...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Compute coverage at different levels
coverage_levels = [50, 80, 90, 95]
coverage_results = {}

for level in coverage_levels:
    lower = (100 - level) / 2
    upper = 100 - lower
    q_lower = np.percentile(y_pred_flat, lower, axis=0)
    q_upper = np.percentile(y_pred_flat, upper, axis=0)
    in_interval = (y_obs >= q_lower) & (y_obs <= q_upper)
    coverage_results[level] = {
        'lower': q_lower,
        'upper': q_upper,
        'in_interval': in_interval,
        'coverage': np.mean(in_interval)
    }

# A) Coverage over time
ax = axes[0]
for level in coverage_levels:
    q_lower = coverage_results[level]['lower']
    q_upper = coverage_results[level]['upper']
    in_interval = coverage_results[level]['in_interval']

    # Color points by whether they're in interval
    colors = ['green' if x else 'red' for x in in_interval]
    if level == 95:  # Only plot points for widest interval
        ax.scatter(time_idx, y_obs, c=colors, s=60, alpha=0.8,
                  edgecolors='black', linewidth=1.5, zorder=10)

    # Plot interval
    alpha = 0.1 + 0.15 * (coverage_levels.index(level))
    ax.fill_between(time_idx, q_lower, q_upper, alpha=alpha,
                    label=f'{level}% Interval')

ax.plot(time_idx, y_obs, 'k-', linewidth=1, alpha=0.3, zorder=5)
ax.set_xlabel('Time Index')
ax.set_ylabel('Count')
ax.set_title('Sequential Coverage: Predictive Intervals\n(Green = in 95% interval, Red = outside)')
ax.legend()
ax.grid(alpha=0.3)

# B) Coverage summary
ax = axes[1]
nominal_coverage = np.array(coverage_levels) / 100
actual_coverage = np.array([coverage_results[level]['coverage'] for level in coverage_levels])

ax.plot(nominal_coverage, actual_coverage, 'bo-', linewidth=2, markersize=10, label='Actual')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal (45° line)')
ax.set_xlabel('Nominal Coverage')
ax.set_ylabel('Actual Coverage')
ax.set_title('Coverage Calibration')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0.4, 1)
ax.set_ylim(0.4, 1)

# Add text with coverage rates
for level in coverage_levels:
    coverage = coverage_results[level]['coverage']
    ax.text(level/100 + 0.02, coverage - 0.02, f'{coverage:.2f}', fontsize=9)

plt.tight_layout()
plt.savefig(PLOT_DIR / "coverage_analysis.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOT_DIR / 'coverage_analysis.png'}")
plt.close()

# Plot 6: Q-Q Plot (Observed vs Predicted Quantiles)
print("\nPlot 6: Quantile-quantile plot...")
fig, ax = plt.subplots(figsize=(8, 8))

# Compute quantiles
quantiles = np.linspace(0, 1, 100)
obs_quantiles = np.percentile(y_obs, quantiles * 100)
pred_quantiles = np.percentile(y_pred_flat.flatten(), quantiles * 100)

ax.scatter(pred_quantiles, obs_quantiles, s=30, alpha=0.7, edgecolors='black')
ax.plot([0, max(pred_quantiles)], [0, max(pred_quantiles)], 'r--', linewidth=2, label='45° line')
ax.set_xlabel('Predicted Quantiles')
ax.set_ylabel('Observed Quantiles')
ax.set_title('Q-Q Plot: Observed vs Predicted Quantiles')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "qq_plot.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOT_DIR / 'qq_plot.png'}")
plt.close()

# Plot 7: ACF Comparison
print("\nPlot 7: Autocorrelation comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Compute ACF for observed
max_lag = 15
obs_acf = compute_acf(y_obs, nlags=max_lag)

# Compute ACF for each replication
pred_acf_list = []
for i in range(min(500, n_samples)):  # Sample 500 replications
    pred_acf_list.append(compute_acf(y_pred_flat[i, :], nlags=max_lag))
pred_acf_array = np.array(pred_acf_list)

# A) ACF with confidence bands
ax = axes[0]
pred_acf_mean = np.mean(pred_acf_array, axis=0)
pred_acf_q05 = np.percentile(pred_acf_array, 5, axis=0)
pred_acf_q95 = np.percentile(pred_acf_array, 95, axis=0)

lags = np.arange(max_lag + 1)
ax.fill_between(lags, pred_acf_q05, pred_acf_q95, alpha=0.3, color='steelblue', label='90% Pred Band')
ax.plot(lags, pred_acf_mean, 'b-', linewidth=2, label='Pred Mean ACF')
ax.plot(lags, obs_acf, 'ro-', linewidth=2, markersize=6, label='Observed ACF')
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation Function Comparison')
ax.legend()
ax.grid(alpha=0.3)

# B) ACF(1) distribution
ax = axes[1]
pred_acf1 = pred_acf_array[:, 1]
ax.hist(pred_acf1, bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True)
ax.axvline(obs_acf[1], color='red', linewidth=2, linestyle='--', label=f'Observed ACF(1)={obs_acf[1]:.3f}')
ax.axvline(np.mean(pred_acf1), color='blue', linewidth=2, linestyle=':', label=f'Pred Mean ACF(1)={np.mean(pred_acf1):.3f}')
ax.set_xlabel('ACF(1)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Lag-1 Autocorrelation')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "acf_comparison.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOT_DIR / 'acf_comparison.png'}")
plt.close()

# Plot 8: ArviZ PPC Plot
print("\nPlot 8: ArviZ PPC plot...")
try:
    fig = plt.figure(figsize=(12, 6))
    # Use observed_data group for comparison
    az.plot_ppc(idata, num_pp_samples=100)
    plt.suptitle('ArviZ Posterior Predictive Check', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "arviz_ppc.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR / 'arviz_ppc.png'}")
    plt.close()
except Exception as e:
    print(f"  ArviZ PPC plot failed: {e}")

# ============================================================================
# C. SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECK SUMMARY")
print("="*80)

# Count how many test statistics are well-captured
def check_coverage(obs_val, pred_vals, tolerance=0.1):
    """Check if observed is within central (1-tolerance) of predictive"""
    lower = np.percentile(pred_vals, tolerance * 100 / 2)
    upper = np.percentile(pred_vals, 100 - tolerance * 100 / 2)
    return lower <= obs_val <= upper

checks = {}
for key in test_stat_names:
    checks[key] = check_coverage(obs_stats[key], pred_stats[key], tolerance=0.1)

n_pass = sum(checks.values())
n_total = len(checks)

print(f"\nTest Statistics Coverage (within 90% predictive interval):")
for key, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {key}: {status}")

print(f"\nOverall: {n_pass}/{n_total} test statistics within 90% interval")

# Coverage assessment
print(f"\nSequential Coverage (predictive intervals contain observed):")
for level in coverage_levels:
    coverage = coverage_results[level]['coverage']
    expected = level / 100
    status = "GOOD" if abs(coverage - expected) < 0.1 else "POOR"
    print(f"  {level}% interval: {coverage:.2%} actual coverage [{status}]")

# Key findings
print("\nKey Findings:")
print(f"1. Mean: Obs={obs_stats['mean']:.1f}, Pred={np.mean(pred_stats['mean']):.1f} ± {np.std(pred_stats['mean']):.1f}")
print(f"2. Overdispersion: Obs={obs_stats['var_mean_ratio']:.1f}, Pred={np.mean(pred_stats['var_mean_ratio']):.1f} ± {np.std(pred_stats['var_mean_ratio']):.1f}")
print(f"3. Growth: Obs={obs_stats['growth_factor']:.2f}x, Pred={np.mean(pred_stats['growth_factor']):.2f}x ± {np.std(pred_stats['growth_factor']):.2f}x")
print(f"4. ACF(1): Obs={obs_stats['acf_lag1']:.3f}, Pred={np.mean(pred_stats['acf_lag1']):.3f} ± {np.std(pred_stats['acf_lag1']):.3f}")

# Overall assessment
if n_pass >= n_total * 0.8 and all(abs(coverage_results[95]['coverage'] - 0.95) < 0.15 for _ in [1]):
    verdict = "PASS"
    print("\n*** MODEL ADEQUACY: PASS ***")
    print("The model adequately captures key features of the observed data.")
elif n_pass >= n_total * 0.5:
    verdict = "MARGINAL"
    print("\n*** MODEL ADEQUACY: MARGINAL ***")
    print("The model captures some features but shows systematic discrepancies.")
else:
    verdict = "FAIL"
    print("\n*** MODEL ADEQUACY: FAIL ***")
    print("The model fails to capture key features of the observed data.")

print("\n" + "="*80)
print("Analysis complete. All plots saved to:")
print(f"  {PLOT_DIR}")
print("="*80)

# Save summary statistics to file
summary_dict = {
    'verdict': verdict,
    'n_pass': n_pass,
    'n_total': n_total,
    'coverage_95': coverage_results[95]['coverage'],
    'observed_stats': obs_stats,
    'predicted_stats_mean': {k: np.mean(v) for k, v in pred_stats.items()},
    'predicted_stats_sd': {k: np.std(v) for k, v in pred_stats.items()}
}

import json
with open(PLOT_DIR.parent / "code" / "ppc_summary.json", 'w') as f:
    # Convert numpy types to native Python types
    summary_dict_clean = {}
    for key, val in summary_dict.items():
        if isinstance(val, dict):
            summary_dict_clean[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                      for k, v in val.items()}
        elif isinstance(val, (np.floating, np.integer)):
            summary_dict_clean[key] = float(val)
        else:
            summary_dict_clean[key] = val
    json.dump(summary_dict_clean, f, indent=2)

print(f"\nSummary saved to: {PLOT_DIR.parent / 'code' / 'ppc_summary.json'}")
