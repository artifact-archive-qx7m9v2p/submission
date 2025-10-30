"""
Comprehensive Posterior Predictive Checks for Logarithmic Regression Model
============================================================================

This script performs extensive posterior predictive checks to assess whether
the fitted Bayesian logarithmic regression model can reproduce key features
of the observed data.

Model: Y ~ Normal(α + β·log(x), σ)

Checks performed:
1. Visual comparisons (density overlays, scatter plots, residuals)
2. Test statistics with Bayesian p-values
3. Coverage calibration (50%, 80%, 95% intervals)
4. LOO-PIT calibration
5. Influential point analysis
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = Path('/workspace/data/data.csv')
POSTERIOR_PATH = Path('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
OUTPUT_DIR = Path('/workspace/experiments/experiment_1/posterior_predictive_check')
PLOT_DIR = OUTPUT_DIR / 'plots'
CODE_DIR = OUTPUT_DIR / 'code'

# Ensure directories exist
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)

# Load data and posterior inference data
print("Loading data and posterior...")
data = pd.read_csv(DATA_PATH)
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(y_obs)

# Load posterior inference data
idata = az.from_netcdf(POSTERIOR_PATH)

print(f"Data: N={n_obs}, x ∈ [{x_obs.min():.1f}, {x_obs.max():.1f}], Y ∈ [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"Posterior samples: {idata.posterior.dims}")
print(f"Groups available: {list(idata.groups())}")

# Extract posterior predictive samples (variable name is 'Y')
y_rep = idata.posterior_predictive['Y'].values.reshape(-1, n_obs)
n_rep = y_rep.shape[0]
print(f"Posterior predictive: {n_rep} replications × {n_obs} observations")

# Extract observed data from idata (for consistency)
y_obs_idata = idata.observed_data['Y'].values
x_obs_idata = idata.observed_data['x'].values

# Verify consistency
assert np.allclose(y_obs, y_obs_idata), "Observed data mismatch!"
assert np.allclose(x_obs, x_obs_idata), "x data mismatch!"

# ============================================================================
# 1. VISUAL CHECKS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUAL DIAGNOSTICS")
print("="*80)

# ----------------------------------------------------------------------------
# Plot 1: Density Overlay - Observed vs Replicated Data
# ----------------------------------------------------------------------------
print("\nPlot 1: Density overlay (observed vs replicated)")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot density of replicated data (sample 100 replications)
for i in range(min(100, n_rep)):
    if i == 0:
        ax.hist(y_rep[i, :], bins=15, alpha=0.01, color='blue',
                density=True, label='Y_rep (100 samples)')
    else:
        ax.hist(y_rep[i, :], bins=15, alpha=0.01, color='blue', density=True)

# Plot observed data
ax.hist(y_obs, bins=15, alpha=0.7, color='red', density=True,
        edgecolor='black', linewidth=2, label='Y_obs')

ax.set_xlabel('Y', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Predictive Check: Observed vs Replicated Data Distribution',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'ppc_density_overlay.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 2: PPC Scatter - Y_rep vs x with Observed Data
# ----------------------------------------------------------------------------
print("Plot 2: PPC scatter with credible intervals")

# Calculate posterior predictive intervals
y_rep_mean = y_rep.mean(axis=0)
y_rep_median = np.median(y_rep, axis=0)
y_rep_025 = np.percentile(y_rep, 2.5, axis=0)
y_rep_975 = np.percentile(y_rep, 97.5, axis=0)
y_rep_10 = np.percentile(y_rep, 10, axis=0)
y_rep_90 = np.percentile(y_rep, 90, axis=0)
y_rep_25 = np.percentile(y_rep, 25, axis=0)
y_rep_75 = np.percentile(y_rep, 75, axis=0)

# Sort by x for plotting
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]
y_sorted = y_obs[sort_idx]

fig, ax = plt.subplots(figsize=(12, 7))

# Plot sample realizations (50 random draws)
sample_indices = np.random.choice(n_rep, size=min(50, n_rep), replace=False)
for idx in sample_indices:
    ax.plot(x_sorted, y_rep[idx, sort_idx], 'b-', alpha=0.05, linewidth=0.5)

# Plot credible intervals
ax.fill_between(x_sorted, y_rep_025[sort_idx], y_rep_975[sort_idx],
                alpha=0.2, color='blue', label='95% Credible Interval')
ax.fill_between(x_sorted, y_rep_25[sort_idx], y_rep_75[sort_idx],
                alpha=0.3, color='blue', label='50% Credible Interval')

# Plot median and mean
ax.plot(x_sorted, y_rep_median[sort_idx], 'b-', linewidth=2, label='Median Prediction')
ax.plot(x_sorted, y_rep_mean[sort_idx], 'g--', linewidth=2, label='Mean Prediction')

# Plot observed data
ax.scatter(x_obs, y_obs, c='red', s=80, alpha=0.8, edgecolors='black',
           linewidth=1.5, zorder=5, label='Observed Data')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Posterior Predictive Check: Model Fit with Uncertainty Bands',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'ppc_scatter_intervals.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 3: Residual Analysis
# ----------------------------------------------------------------------------
print("Plot 3: Residual analysis")

# Calculate posterior predictive residuals
residuals = y_obs - y_rep_mean
residuals_median = y_obs - y_rep_median

# Standardized residuals
residuals_std = residuals / np.std(y_rep, axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a: Residuals vs x
axes[0, 0].scatter(x_obs, residuals, c='blue', s=60, alpha=0.7, edgecolors='black')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('x', fontsize=11)
axes[0, 0].set_ylabel('Residual (Y_obs - Y_pred_mean)', fontsize=11)
axes[0, 0].set_title('Residuals vs Predictor', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# 3b: Residuals vs fitted values
axes[0, 1].scatter(y_rep_mean, residuals, c='blue', s=60, alpha=0.7, edgecolors='black')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Fitted Values (Y_pred_mean)', fontsize=11)
axes[0, 1].set_ylabel('Residual', fontsize=11)
axes[0, 1].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3c: Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Residuals vs Normal', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 3d: Histogram of standardized residuals
axes[1, 1].hist(residuals_std, bins=15, alpha=0.7, color='blue', edgecolor='black', density=True)
x_norm = np.linspace(residuals_std.min(), residuals_std.max(), 100)
axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, 0, 1), 'r-', linewidth=2, label='N(0,1)')
axes[1, 1].set_xlabel('Standardized Residual', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].set_title('Standardized Residuals Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 4: Test Statistic Distributions
# ----------------------------------------------------------------------------
print("Plot 4: Test statistic distributions")

# Compute test statistics for observed and replicated data
def compute_test_stats(y):
    """Compute suite of test statistics"""
    return {
        'mean': np.mean(y),
        'std': np.std(y, ddof=1),
        'min': np.min(y),
        'max': np.max(y),
        'median': np.median(y),
        'q25': np.percentile(y, 25),
        'q75': np.percentile(y, 75),
        'range': np.max(y) - np.min(y),
        'skewness': stats.skew(y),
        'kurtosis': stats.kurtosis(y)
    }

# Observed test statistics
test_stats_obs = compute_test_stats(y_obs)

# Replicated test statistics
test_stats_rep = {key: [] for key in test_stats_obs.keys()}
for i in range(n_rep):
    stats_i = compute_test_stats(y_rep[i, :])
    for key in test_stats_rep.keys():
        test_stats_rep[key].append(stats_i[key])

# Convert to arrays
for key in test_stats_rep.keys():
    test_stats_rep[key] = np.array(test_stats_rep[key])

# Plot distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

test_stats_to_plot = ['mean', 'std', 'min', 'max', 'median', 'range',
                      'skewness', 'kurtosis', 'q75']

for i, stat_name in enumerate(test_stats_to_plot):
    ax = axes[i]

    # Plot distribution of replicated test statistic
    ax.hist(test_stats_rep[stat_name], bins=30, alpha=0.7, color='blue',
            edgecolor='black', density=True, label='Y_rep')

    # Plot observed test statistic
    ax.axvline(test_stats_obs[stat_name], color='red', linewidth=3,
               linestyle='--', label='Y_obs')

    # Calculate Bayesian p-value (two-sided for more conservative test)
    p_val_greater = np.mean(test_stats_rep[stat_name] >= test_stats_obs[stat_name])
    p_val_less = np.mean(test_stats_rep[stat_name] <= test_stats_obs[stat_name])
    p_val = 2 * min(p_val_greater, p_val_less)

    ax.set_xlabel(stat_name.capitalize(), fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{stat_name.capitalize()} (p={p_val:.3f})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Test Statistic Distributions: Observed vs Replicated',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'test_statistic_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 5: LOO-PIT Calibration
# ----------------------------------------------------------------------------
print("Plot 5: LOO-PIT calibration")

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_loo_pit(idata, y='Y', ecdf=True, ax=ax)
    ax.set_title('LOO Probability Integral Transform (PIT) - Calibration Check',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'loo_pit_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("LOO-PIT plot saved")
except Exception as e:
    print(f"LOO-PIT plot failed: {e}")
    print("Continuing with other diagnostics...")

# ----------------------------------------------------------------------------
# Plot 6: ArviZ PPC Plot
# ----------------------------------------------------------------------------
print("Plot 6: ArviZ PPC plot")

try:
    fig, ax = plt.subplots(figsize=(12, 6))
    az.plot_ppc(idata, num_pp_samples=100, ax=ax)
    ax.set_title('ArviZ Posterior Predictive Check', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'arviz_ppc.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"ArviZ PPC plot failed: {e}")

# ============================================================================
# 2. QUANTITATIVE TEST STATISTICS WITH BAYESIAN P-VALUES
# ============================================================================

print("\n" + "="*80)
print("COMPUTING BAYESIAN P-VALUES")
print("="*80)

# Compute Bayesian p-values for all test statistics
bayesian_p_values = {}

for stat_name in test_stats_obs.keys():
    obs_val = test_stats_obs[stat_name]
    rep_vals = test_stats_rep[stat_name]

    # Two-sided p-value: proportion of reps more extreme than observed
    p_val_greater = np.mean(rep_vals >= obs_val)
    p_val_less = np.mean(rep_vals <= obs_val)
    p_val_two_sided = 2 * min(p_val_greater, p_val_less)

    bayesian_p_values[stat_name] = {
        'observed': obs_val,
        'rep_mean': np.mean(rep_vals),
        'rep_std': np.std(rep_vals),
        'rep_025': np.percentile(rep_vals, 2.5),
        'rep_975': np.percentile(rep_vals, 97.5),
        'p_value_greater': p_val_greater,
        'p_value_less': p_val_less,
        'p_value_two_sided': p_val_two_sided
    }

# Additional test statistics
print("\nComputing additional test statistics...")

# Correlation with x (should be preserved)
corr_obs = np.corrcoef(x_obs, y_obs)[0, 1]
corr_rep = np.array([np.corrcoef(x_obs, y_rep[i, :])[0, 1] for i in range(n_rep)])
p_val_greater = np.mean(corr_rep >= corr_obs)
p_val_less = np.mean(corr_rep <= corr_obs)
bayesian_p_values['correlation_xy'] = {
    'observed': corr_obs,
    'rep_mean': np.mean(corr_rep),
    'rep_std': np.std(corr_rep),
    'rep_025': np.percentile(corr_rep, 2.5),
    'rep_975': np.percentile(corr_rep, 97.5),
    'p_value_greater': p_val_greater,
    'p_value_less': p_val_less,
    'p_value_two_sided': 2 * min(p_val_greater, p_val_less)
}

# Max absolute residual
max_abs_resid_obs = np.max(np.abs(y_obs - y_rep_mean))
max_abs_resid_rep = np.array([np.max(np.abs(y_rep[i, :] - y_rep.mean(axis=0))) for i in range(n_rep)])
bayesian_p_values['max_abs_residual'] = {
    'observed': max_abs_resid_obs,
    'rep_mean': np.mean(max_abs_resid_rep),
    'rep_std': np.std(max_abs_resid_rep),
    'rep_025': np.percentile(max_abs_resid_rep, 2.5),
    'rep_975': np.percentile(max_abs_resid_rep, 97.5),
    'p_value_greater': np.mean(max_abs_resid_rep >= max_abs_resid_obs),
    'p_value_less': np.mean(max_abs_resid_rep <= max_abs_resid_obs),
    'p_value_two_sided': np.mean(max_abs_resid_rep >= max_abs_resid_obs)
}

# Print summary
print("\nBayesian P-Values Summary (two-sided):")
print("-" * 90)
print(f"{'Statistic':<20} {'Observed':>10} {'Rep Mean':>10} {'Rep 95% CI':>20} {'P-value':>10} {'Flag':>5}")
print("-" * 90)
for stat_name, vals in bayesian_p_values.items():
    flag = "**" if vals['p_value_two_sided'] < 0.05 else "  "
    ci_str = f"[{vals['rep_025']:.3f}, {vals['rep_975']:.3f}]"
    print(f"{stat_name:<20} {vals['observed']:>10.3f} {vals['rep_mean']:>10.3f} {ci_str:>20} "
          f"{vals['p_value_two_sided']:>10.3f} {flag:>5}")

# ============================================================================
# 3. COVERAGE CALIBRATION
# ============================================================================

print("\n" + "="*80)
print("COVERAGE CALIBRATION")
print("="*80)

# Calculate coverage for different credible intervals
coverage_levels = [50, 80, 90, 95, 99]
coverage_results = {}

for level in coverage_levels:
    lower = (100 - level) / 2
    upper = 100 - lower

    y_lower = np.percentile(y_rep, lower, axis=0)
    y_upper = np.percentile(y_rep, upper, axis=0)

    # Count observations within interval
    in_interval = (y_obs >= y_lower) & (y_obs <= y_upper)
    coverage = np.mean(in_interval) * 100

    coverage_results[level] = {
        'expected': level,
        'observed': coverage,
        'n_in': np.sum(in_interval),
        'n_total': n_obs
    }

    flag = ""
    if level == 95:
        if coverage < 85 or coverage > 99:
            flag = " [FAIL]"
        else:
            flag = " [PASS]"

    print(f"{level:2d}% Interval: Expected={level:2d}%, Observed={coverage:5.1f}% "
          f"({np.sum(in_interval):2d}/{n_obs} observations){flag}")

# Visual coverage check
fig, ax = plt.subplots(figsize=(10, 6))

expected = [r['expected'] for r in coverage_results.values()]
observed = [r['observed'] for r in coverage_results.values()]

ax.plot(expected, expected, 'k--', linewidth=2, label='Perfect Calibration')
ax.plot(expected, observed, 'o-', linewidth=2, markersize=10, label='Observed Coverage', color='blue')

# Add tolerance bands for 95% interval
ax.axhspan(85, 99, xmin=0.7, xmax=1.0, alpha=0.2, color='green', label='Acceptable Range (95% CI)')

ax.set_xlabel('Expected Coverage (%)', fontsize=12)
ax.set_ylabel('Observed Coverage (%)', fontsize=12)
ax.set_title('Coverage Calibration: Expected vs Observed', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([45, 102])
ax.set_ylim([45, 102])

plt.tight_layout()
plt.savefig(PLOT_DIR / 'coverage_calibration.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. INFLUENTIAL POINT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("INFLUENTIAL POINT ANALYSIS")
print("="*80)

# Compute LOO diagnostics
print("Computing LOO diagnostics...")
loo_result = az.loo(idata, pointwise=True)

pareto_k = loo_result.pareto_k.values

print(f"\nPareto k Statistics:")
print(f"  k < 0.5 (good):     {np.sum(pareto_k < 0.5):2d} / {n_obs}")
print(f"  0.5 ≤ k < 0.7 (ok): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7)):2d} / {n_obs}")
print(f"  k ≥ 0.7 (bad):      {np.sum(pareto_k >= 0.7):2d} / {n_obs}")

# Identify influential points
influential_idx = np.where(pareto_k > 0.5)[0]
if len(influential_idx) > 0:
    print(f"\nInfluential observations (k > 0.5):")
    for idx in influential_idx:
        print(f"  Obs {idx:2d}: x={x_obs[idx]:5.1f}, Y={y_obs[idx]:.3f}, k={pareto_k[idx]:.3f}")
else:
    print(f"\nNo influential observations detected (all k < 0.5)")

# Plot Pareto k values
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in pareto_k]
ax.scatter(range(n_obs), pareto_k, c=colors, s=80, alpha=0.7, edgecolors='black')

ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='k=0.5 (ok threshold)')
ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='k=0.7 (bad threshold)')

# Annotate high k points
for idx in influential_idx:
    ax.annotate(f'x={x_obs[idx]:.1f}', xy=(idx, pareto_k[idx]),
               xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel('Pareto k', fontsize=12)
ax.set_title('LOO Pareto k Diagnostic: Influential Points', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'pareto_k_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional plot: Pareto k vs x
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in pareto_k]
ax.scatter(x_obs, pareto_k, c=colors, s=80, alpha=0.7, edgecolors='black')
ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='k=0.5')
ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='k=0.7')

for idx in influential_idx:
    ax.annotate(f'{idx}', xy=(x_obs[idx], pareto_k[idx]),
               xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Pareto k', fontsize=12)
ax.set_title('Pareto k vs Predictor: Identifying Influential Regions', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'pareto_k_vs_x.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. ADDITIONAL DIAGNOSTIC: Point-wise interval coverage
# ============================================================================

print("\n" + "="*80)
print("POINT-WISE INTERVAL ANALYSIS")
print("="*80)

# For each observation, check if it's in the 95% interval
y_025 = np.percentile(y_rep, 2.5, axis=0)
y_975 = np.percentile(y_rep, 97.5, axis=0)
in_interval_95 = (y_obs >= y_025) & (y_obs <= y_975)

# Plot point-wise coverage
fig, ax = plt.subplots(figsize=(12, 7))

# Sort by x for visualization
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]

# Plot intervals
ax.fill_between(x_sorted, y_025[sort_idx], y_975[sort_idx],
                alpha=0.3, color='blue', label='95% Credible Interval')

# Plot observed points - color by whether they're in interval
colors = ['green' if in_int else 'red' for in_int in in_interval_95]
for i in range(n_obs):
    if in_interval_95[i]:
        ax.scatter(x_obs[i], y_obs[i], c='green', s=80, alpha=0.7,
                  edgecolors='black', linewidth=1.5, zorder=5)
    else:
        ax.scatter(x_obs[i], y_obs[i], c='red', s=120, alpha=0.8,
                  edgecolors='black', linewidth=2, zorder=5, marker='X')
        # Annotate points outside interval
        ax.annotate(f'{i}', xy=(x_obs[i], y_obs[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

# Add legend manually
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.3, label='95% Credible Interval'),
    plt.scatter([], [], c='green', s=80, alpha=0.7, edgecolors='black', label='In Interval'),
    plt.scatter([], [], c='red', s=120, alpha=0.8, edgecolors='black', marker='X', label='Outside Interval')
]
ax.legend(handles=legend_elements, fontsize=11)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title(f'Point-wise 95% Coverage: {np.sum(in_interval_95)}/{n_obs} observations in interval',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'pointwise_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

# Identify observations outside 95% interval
outside_idx = np.where(~in_interval_95)[0]
if len(outside_idx) > 0:
    print(f"\nObservations outside 95% credible interval:")
    for idx in outside_idx:
        print(f"  Obs {idx:2d}: x={x_obs[idx]:5.1f}, Y={y_obs[idx]:.3f}, "
              f"95% CI=[{y_025[idx]:.3f}, {y_975[idx]:.3f}]")
else:
    print(f"\nAll observations within 95% credible interval")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save numerical results
results = {
    'bayesian_p_values': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                              for kk, vv in v.items()}
                          for k, v in bayesian_p_values.items()},
    'coverage_calibration': {str(k): {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                      for kk, vv in v.items()}
                            for k, v in coverage_results.items()},
    'pareto_k_summary': {
        'k_lt_05': int(np.sum(pareto_k < 0.5)),
        'k_05_07': int(np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))),
        'k_gt_07': int(np.sum(pareto_k >= 0.7)),
        'max_k': float(np.max(pareto_k)),
        'influential_observations': [int(i) for i in influential_idx]
    },
    'pointwise_coverage': {
        'n_in_95_interval': int(np.sum(in_interval_95)),
        'n_outside_95_interval': int(np.sum(~in_interval_95)),
        'observations_outside': [int(i) for i in outside_idx]
    }
}

with open(OUTPUT_DIR / 'ppc_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {OUTPUT_DIR / 'ppc_results.json'}")
print(f"Plots saved to: {PLOT_DIR}")

# ============================================================================
# 7. SUMMARY STATISTICS FOR REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY FOR REPORT")
print("="*80)

# Count flagged test statistics
flagged_stats = [k for k, v in bayesian_p_values.items()
                 if v['p_value_two_sided'] < 0.05]

print(f"\nTest Statistics Assessment:")
print(f"  Total test statistics: {len(bayesian_p_values)}")
print(f"  Flagged (p < 0.05): {len(flagged_stats)}")
if flagged_stats:
    for stat in flagged_stats:
        print(f"    - {stat}: p={bayesian_p_values[stat]['p_value_two_sided']:.3f}")
else:
    print(f"    - None flagged (all p ≥ 0.05)")

# Coverage assessment
coverage_95 = coverage_results[95]['observed']
coverage_flag = "PASS" if 85 <= coverage_95 <= 99 else "FAIL"
print(f"\nCoverage Assessment:")
print(f"  95% Credible Interval Coverage: {coverage_95:.1f}% ({coverage_results[95]['n_in']}/{n_obs} obs)")
print(f"  Status: {coverage_flag} (acceptable range: 85-99%)")

# Pareto k assessment
pareto_flag = "PASS" if np.sum(pareto_k >= 0.7) == 0 else "FAIL" if np.sum(pareto_k >= 0.7) > 0.1 * n_obs else "MARGINAL"
print(f"\nInfluential Points Assessment:")
print(f"  Observations with k ≥ 0.7: {np.sum(pareto_k >= 0.7)}/{n_obs}")
print(f"  Observations with k ≥ 0.5: {np.sum(pareto_k >= 0.5)}/{n_obs}")
print(f"  Status: {pareto_flag} (threshold: <10% with k ≥ 0.7)")

# Overall assessment
print(f"\nOverall Model Adequacy:")
n_issues = 0
if len(flagged_stats) > 2:
    print(f"  - Multiple test statistics flagged ({len(flagged_stats)})")
    n_issues += 1
if coverage_flag == "FAIL":
    print(f"  - Coverage calibration failure")
    n_issues += 1
if pareto_flag == "FAIL":
    print(f"  - Multiple influential points")
    n_issues += 1

if n_issues == 0:
    print(f"  No major issues detected - model appears adequate")
elif n_issues == 1:
    print(f"  Minor issues detected - model may be acceptable with caveats")
else:
    print(f"  Multiple issues detected - model may need revision")

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("="*80)
print(f"\nPlots created:")
for plot_file in sorted(PLOT_DIR.glob('*.png')):
    print(f"  - {plot_file.name}")

print(f"\nNext step: Review findings and create ppc_findings.md report")
