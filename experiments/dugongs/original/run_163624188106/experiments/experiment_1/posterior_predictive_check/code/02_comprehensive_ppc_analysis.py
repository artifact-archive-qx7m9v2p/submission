"""
Comprehensive Posterior Predictive Check for Log-Log Linear Model
"""
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
data = pd.read_csv('/workspace/data/data.csv')

# Extract posterior predictive samples and observed data
y_pred = idata.posterior_predictive.y_obs.values  # shape: (chains, draws, n_obs)
y_obs = idata.observed_data.Y.values  # observed Y values
x_obs = data.x.values

# Reshape posterior predictive to (n_draws, n_obs)
n_chains, n_draws, n_obs = y_pred.shape
y_pred_flat = y_pred.reshape(-1, n_obs)  # (n_chains * n_draws, n_obs)
n_total_draws = y_pred_flat.shape[0]

print(f"\nData dimensions:")
print(f"  Observed data points: {n_obs}")
print(f"  Posterior draws: {n_total_draws}")
print(f"  Shape of y_pred_flat: {y_pred_flat.shape}")

# Create output directory
output_dir = '/workspace/experiments/experiment_1/posterior_predictive_check/plots'

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS")
print("="*80)

# ============================================================================
# 1. Summary Statistics Comparison
# ============================================================================
print("\n1. SUMMARY STATISTICS")
print("-" * 80)

# Calculate statistics for observed data
obs_stats = {
    'mean': np.mean(y_obs),
    'std': np.std(y_obs, ddof=1),
    'min': np.min(y_obs),
    'max': np.max(y_obs),
    'q25': np.percentile(y_obs, 25),
    'q50': np.percentile(y_obs, 50),
    'q75': np.percentile(y_obs, 75)
}

# Calculate statistics for each replicated dataset
rep_stats = {
    'mean': np.mean(y_pred_flat, axis=1),
    'std': np.std(y_pred_flat, axis=1, ddof=1),
    'min': np.min(y_pred_flat, axis=1),
    'max': np.max(y_pred_flat, axis=1),
    'q25': np.percentile(y_pred_flat, 25, axis=1),
    'q50': np.percentile(y_pred_flat, 50, axis=1),
    'q75': np.percentile(y_pred_flat, 75, axis=1)
}

# Calculate Bayesian p-values
bayesian_pvalues = {}
for stat_name, obs_value in obs_stats.items():
    rep_values = rep_stats[stat_name]
    # Two-sided p-value: proportion of replicated statistics more extreme than observed
    p_value = np.mean(np.abs(rep_values - np.mean(rep_values)) >= np.abs(obs_value - np.mean(rep_values)))
    bayesian_pvalues[stat_name] = p_value

    print(f"\n{stat_name.upper()}:")
    print(f"  Observed: {obs_value:.4f}")
    print(f"  Predicted mean: {np.mean(rep_values):.4f}")
    print(f"  Predicted 95% CI: [{np.percentile(rep_values, 2.5):.4f}, {np.percentile(rep_values, 97.5):.4f}]")
    print(f"  Bayesian p-value: {p_value:.4f}")
    if p_value < 0.05 or p_value > 0.95:
        print(f"  WARNING: p-value in tail!")

# ============================================================================
# 2. Visualization: Test Statistics
# ============================================================================
print("\n\n2. Creating test statistics plot...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (stat_name, obs_value) in enumerate(obs_stats.items()):
    ax = axes[idx]
    rep_values = rep_stats[stat_name]

    ax.hist(rep_values, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(obs_value, color='red', linestyle='--', linewidth=2, label='Observed')
    ax.axvline(np.mean(rep_values), color='blue', linestyle='-', linewidth=1, label='Predicted mean')

    ax.set_xlabel(f'{stat_name}')
    ax.set_ylabel('Density')
    ax.set_title(f'{stat_name.upper()}\n(p={bayesian_pvalues[stat_name]:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')  # Hide last subplot

plt.tight_layout()
plt.savefig(f'{output_dir}/test_statistics.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/test_statistics.png")
plt.close()

# ============================================================================
# 3. Overall Distribution Comparison (PPC Plot)
# ============================================================================
print("\n3. Creating overall distribution comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot posterior predictive distribution
for i in range(min(100, n_total_draws)):
    ax.plot(x_obs, y_pred_flat[i, :], 'b-', alpha=0.02, linewidth=0.5)

# Calculate percentiles
y_pred_lower = np.percentile(y_pred_flat, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred_flat, 97.5, axis=0)
y_pred_median = np.percentile(y_pred_flat, 50, axis=0)

# Sort by x for better visualization
sort_idx = np.argsort(x_obs)
ax.fill_between(x_obs[sort_idx], y_pred_lower[sort_idx], y_pred_upper[sort_idx],
                alpha=0.3, color='blue', label='95% Posterior Predictive Interval')
ax.plot(x_obs[sort_idx], y_pred_median[sort_idx], 'b-', linewidth=2, label='Predicted Median')

# Plot observed data
ax.scatter(x_obs, y_obs, color='red', s=100, zorder=10, edgecolor='black',
          linewidth=1.5, label='Observed Data', alpha=0.8)

ax.set_xlabel('x')
ax.set_ylabel('Y')
ax.set_title('Posterior Predictive Check: Observed vs Predicted')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/ppc_overall.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/ppc_overall.png")
plt.close()

# ============================================================================
# 4. Point-wise Coverage Check
# ============================================================================
print("\n4. POINT-WISE COVERAGE CHECK")
print("-" * 80)

coverage_levels = [50, 80, 95]
for level in coverage_levels:
    lower = (100 - level) / 2
    upper = 100 - lower

    y_pred_lower = np.percentile(y_pred_flat, lower, axis=0)
    y_pred_upper = np.percentile(y_pred_flat, upper, axis=0)

    in_interval = (y_obs >= y_pred_lower) & (y_obs <= y_pred_upper)
    coverage = np.mean(in_interval) * 100

    print(f"\n{level}% Credible Interval:")
    print(f"  Expected coverage: {level}%")
    print(f"  Actual coverage: {coverage:.1f}%")
    print(f"  Points outside: {np.sum(~in_interval)}/{n_obs}")

    if coverage < level - 5:
        print(f"  WARNING: Under-coverage detected! (SBC concern confirmed)")
    elif coverage > level + 5:
        print(f"  WARNING: Over-coverage detected!")

    if np.sum(~in_interval) > 0:
        print(f"  Indices of points outside: {np.where(~in_interval)[0].tolist()}")

# ============================================================================
# 5. Residual Analysis (Log Scale)
# ============================================================================
print("\n\n5. Creating residual analysis plots (log scale)...")

# Calculate residuals in log scale
log_y_obs = np.log(y_obs)
log_y_pred_median = np.log(y_pred_median)
residuals_log = log_y_obs - log_y_pred_median

# Also get residuals for all posterior draws
log_y_pred_all = np.log(y_pred_flat)
residuals_log_all = log_y_obs[np.newaxis, :] - log_y_pred_all

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a. Residuals vs fitted (log scale)
ax = axes[0, 0]
ax.scatter(log_y_pred_median, residuals_log, s=80, alpha=0.7, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
# Add uncertainty bands
residuals_std = np.std(residuals_log_all, axis=0)
ax.errorbar(log_y_pred_median, residuals_log, yerr=1.96*residuals_std,
           fmt='o', alpha=0.5, capsize=3)
ax.set_xlabel('Fitted log(Y)')
ax.set_ylabel('Residuals (log scale)')
ax.set_title('Residuals vs Fitted (Log Scale)')
ax.grid(True, alpha=0.3)

# 5b. Residuals vs x (log scale) - check for patterns
ax = axes[0, 1]
ax.scatter(x_obs, residuals_log, s=80, alpha=0.7, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.errorbar(x_obs, residuals_log, yerr=1.96*residuals_std,
           fmt='o', alpha=0.5, capsize=3)
ax.set_xlabel('x')
ax.set_ylabel('Residuals (log scale)')
ax.set_title('Residuals vs x (Log Scale)')
ax.grid(True, alpha=0.3)

# 5c. QQ plot (log scale residuals)
ax = axes[1, 0]
stats.probplot(residuals_log, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Log-Scale Residuals')
ax.grid(True, alpha=0.3)

# 5d. Histogram of residuals (log scale)
ax = axes[1, 1]
ax.hist(residuals_log, bins=15, alpha=0.7, edgecolor='black', density=True)
# Overlay normal distribution
x_range = np.linspace(residuals_log.min(), residuals_log.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, residuals_log.mean(), residuals_log.std()),
       'r-', linewidth=2, label='Normal fit')
ax.set_xlabel('Residuals (log scale)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Log-Scale Residuals')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/residuals_log_scale.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/residuals_log_scale.png")
plt.close()

# Test for normality
shapiro_stat, shapiro_p = stats.shapiro(residuals_log)
print(f"\nShapiro-Wilk test for normality (log-scale residuals):")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print(f"  WARNING: Residuals may not be normally distributed!")

# ============================================================================
# 6. Residual Analysis (Original Scale)
# ============================================================================
print("\n\n6. Creating residual analysis plots (original scale)...")

# Calculate residuals in original scale
residuals_original = y_obs - y_pred_median

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a. Residuals vs fitted (original scale)
ax = axes[0, 0]
ax.scatter(y_pred_median, residuals_original, s=80, alpha=0.7, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted Y')
ax.set_ylabel('Residuals (original scale)')
ax.set_title('Residuals vs Fitted (Original Scale)')
ax.grid(True, alpha=0.3)

# 6b. Residuals vs x (original scale)
ax = axes[0, 1]
ax.scatter(x_obs, residuals_original, s=80, alpha=0.7, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('Residuals (original scale)')
ax.set_title('Residuals vs x (Original Scale)')
ax.grid(True, alpha=0.3)

# 6c. Absolute residuals vs fitted (check homoscedasticity)
ax = axes[1, 0]
ax.scatter(y_pred_median, np.abs(residuals_original), s=80, alpha=0.7, edgecolor='black')
ax.set_xlabel('Fitted Y')
ax.set_ylabel('|Residuals| (original scale)')
ax.set_title('Scale-Location Plot (Original Scale)')
ax.grid(True, alpha=0.3)

# 6d. Standardized residuals
residuals_std_scale = residuals_original / np.std(residuals_original)
ax = axes[1, 1]
ax.scatter(y_pred_median, residuals_std_scale, s=80, alpha=0.7, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.axhline(2, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(-2, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(3, color='red', linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(-3, color='red', linestyle=':', linewidth=1, alpha=0.7)
ax.set_xlabel('Fitted Y')
ax.set_ylabel('Standardized Residuals')
ax.set_title('Standardized Residuals (Original Scale)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/residuals_original_scale.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/residuals_original_scale.png")
plt.close()

# Check for outliers
outliers = np.abs(residuals_std_scale) > 2
n_outliers = np.sum(outliers)
print(f"\nOutlier check (|standardized residual| > 2):")
print(f"  Number of outliers: {n_outliers}/{n_obs}")
if n_outliers > 0:
    print(f"  Outlier indices: {np.where(outliers)[0].tolist()}")
    print(f"  Outlier x values: {x_obs[outliers].tolist()}")
    print(f"  Outlier Y values: {y_obs[outliers].tolist()}")

# ============================================================================
# 7. LOO-PIT (Probability Integral Transform)
# ============================================================================
print("\n\n7. Creating LOO-PIT plot...")

# Use ArviZ's built-in LOO-PIT functionality
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_loo_pit(idata, y='y_obs', ax=ax)
ax.set_title('LOO-PIT: Calibration Check\n(Should be uniform for well-calibrated model)')

plt.tight_layout()
plt.savefig(f'{output_dir}/loo_pit.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/loo_pit.png")
plt.close()

# ============================================================================
# 8. Marginal Distribution Overlay
# ============================================================================
print("\n\n8. Creating marginal distribution overlay...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram of replicated data
for i in range(min(200, n_total_draws)):
    ax.hist(y_pred_flat[i, :], bins=20, alpha=0.01, color='blue', density=True)

# Plot observed data histogram
ax.hist(y_obs, bins=15, alpha=0.7, edgecolor='black', color='red',
       density=True, label='Observed', linewidth=2)

# Add KDE for observed
from scipy.stats import gaussian_kde
kde_obs = gaussian_kde(y_obs)
x_range = np.linspace(y_obs.min() - 0.2, y_obs.max() + 0.2, 200)
ax.plot(x_range, kde_obs(x_range), 'r-', linewidth=3, label='Observed KDE')

# Add envelope of predicted
y_pred_all_flat = y_pred_flat.flatten()
kde_pred = gaussian_kde(y_pred_all_flat)
ax.plot(x_range, kde_pred(x_range), 'b-', linewidth=2, label='Predicted KDE', alpha=0.7)

ax.set_xlabel('Y')
ax.set_ylabel('Density')
ax.set_title('Marginal Distribution: Observed vs Predicted')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/marginal_distribution.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/marginal_distribution.png")
plt.close()

# ============================================================================
# 9. ArviZ PPC Plot
# ============================================================================
print("\n\n9. Creating ArviZ PPC plot...")

fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(idata, data_pairs={'y_obs': 'y_obs'}, ax=ax, num_pp_samples=100)
ax.set_title('ArviZ Posterior Predictive Check')
ax.set_xlabel('Y')

plt.tight_layout()
plt.savefig(f'{output_dir}/arviz_ppc.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/arviz_ppc.png")
plt.close()

# ============================================================================
# 10. Functional Form Check (by x ranges)
# ============================================================================
print("\n\n10. Creating functional form check by x ranges...")

# Divide x into quantile-based ranges
x_quartiles = np.percentile(x_obs, [25, 50, 75])
x_ranges = [
    (x_obs <= x_quartiles[0], f'x ≤ {x_quartiles[0]:.1f}'),
    ((x_obs > x_quartiles[0]) & (x_obs <= x_quartiles[1]),
     f'{x_quartiles[0]:.1f} < x ≤ {x_quartiles[1]:.1f}'),
    ((x_obs > x_quartiles[1]) & (x_obs <= x_quartiles[2]),
     f'{x_quartiles[1]:.1f} < x ≤ {x_quartiles[2]:.1f}'),
    (x_obs > x_quartiles[2], f'x > {x_quartiles[2]:.1f}')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (mask, label) in enumerate(x_ranges):
    ax = axes[idx]

    # Get data for this range
    y_obs_range = y_obs[mask]
    y_pred_range = y_pred_flat[:, mask]

    if len(y_obs_range) > 0:
        # Flatten predictions for this range
        y_pred_range_flat = y_pred_range.flatten()

        # Plot histograms
        ax.hist(y_pred_range_flat, bins=20, alpha=0.5, color='blue',
               density=True, label='Predicted', edgecolor='black')
        ax.hist(y_obs_range, bins=min(10, len(y_obs_range)), alpha=0.7,
               color='red', density=True, label='Observed', edgecolor='black', linewidth=2)

        ax.set_xlabel('Y')
        ax.set_ylabel('Density')
        ax.set_title(f'{label} (n={len(y_obs_range)})')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/functional_form_by_x_range.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/functional_form_by_x_range.png")
plt.close()

# ============================================================================
# 11. Individual Observation Check (with highest/lowest errors)
# ============================================================================
print("\n\n11. Creating individual observation error plot...")

# Calculate prediction errors
pred_errors = np.abs(y_obs - y_pred_median)
pred_errors_pct = 100 * pred_errors / y_obs

# Find observations with largest errors
worst_indices = np.argsort(pred_errors)[-5:][::-1]

print(f"\nObservations with largest absolute errors:")
for idx in worst_indices:
    print(f"  Index {idx}: x={x_obs[idx]:.1f}, Y_obs={y_obs[idx]:.3f}, " +
          f"Y_pred={y_pred_median[idx]:.3f}, error={pred_errors[idx]:.3f} " +
          f"({pred_errors_pct[idx]:.1f}%)")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot all predictions
sort_idx = np.argsort(x_obs)
ax.scatter(range(len(x_obs)), y_obs[sort_idx], color='red', s=100,
          zorder=10, label='Observed', edgecolor='black', linewidth=1.5)
ax.scatter(range(len(x_obs)), y_pred_median[sort_idx], color='blue',
          s=80, zorder=9, label='Predicted', alpha=0.7)

# Add error bars
for i, idx in enumerate(sort_idx):
    y_lower = np.percentile(y_pred_flat[:, idx], 2.5)
    y_upper = np.percentile(y_pred_flat[:, idx], 97.5)
    ax.plot([i, i], [y_lower, y_upper], 'b-', alpha=0.3, linewidth=2)

# Highlight worst predictions
worst_positions = [np.where(sort_idx == idx)[0][0] for idx in worst_indices]
ax.scatter(worst_positions, y_obs[worst_indices], color='orange', s=200,
          zorder=11, marker='*', edgecolor='black', linewidth=2,
          label='Largest errors')

ax.set_xlabel('Observation (sorted by x)')
ax.set_ylabel('Y')
ax.set_title('Individual Observations: Observed vs Predicted (with 95% CI)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/individual_observations.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/individual_observations.png")
plt.close()

# ============================================================================
# 12. Summary Statistics Table
# ============================================================================
print("\n\n" + "="*80)
print("SUMMARY OF POSTERIOR PREDICTIVE CHECKS")
print("="*80)

print("\n1. Summary Statistics (Bayesian p-values):")
for stat_name, p_value in bayesian_pvalues.items():
    status = "OK" if 0.05 <= p_value <= 0.95 else "WARNING"
    print(f"  {stat_name:8s}: {p_value:.4f}  [{status}]")

print("\n2. Coverage Analysis:")
print(f"  50% CI: {np.mean((y_obs >= np.percentile(y_pred_flat, 25, axis=0)) & (y_obs <= np.percentile(y_pred_flat, 75, axis=0))) * 100:.1f}% (expected: 50%)")
print(f"  80% CI: {np.mean((y_obs >= np.percentile(y_pred_flat, 10, axis=0)) & (y_obs <= np.percentile(y_pred_flat, 90, axis=0))) * 100:.1f}% (expected: 80%)")
print(f"  95% CI: {np.mean((y_obs >= np.percentile(y_pred_flat, 2.5, axis=0)) & (y_obs <= np.percentile(y_pred_flat, 97.5, axis=0))) * 100:.1f}% (expected: 95%)")

print("\n3. Residual Diagnostics:")
print(f"  Normality (Shapiro-Wilk p-value): {shapiro_p:.4f}")
print(f"  Outliers (|std. residual| > 2): {n_outliers}/{n_obs}")
print(f"  Mean absolute error: {np.mean(pred_errors):.4f}")
print(f"  Mean absolute percentage error: {np.mean(pred_errors_pct):.2f}%")

print("\n" + "="*80)
print("All plots saved to: /workspace/experiments/experiment_1/posterior_predictive_check/plots/")
print("="*80)
