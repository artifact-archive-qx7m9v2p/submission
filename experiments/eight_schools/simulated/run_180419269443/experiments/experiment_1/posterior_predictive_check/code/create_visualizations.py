"""
Create comprehensive visualizations for posterior predictive checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats

# Load results
results = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_results.npz')
y_obs = results['y_obs']
sigma = results['sigma']
y_rep = results['y_rep']
theta_samples = results['theta_samples']
study_pvalues = results['study_pvalues']

n_studies = len(y_obs)
n_samples = y_rep.shape[0]

# Load idata for ArviZ plotting
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print("Creating PPC visualizations...")

# ============================================================================
# PLOT 1: Study-level overlay plots (key diagnostic)
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(n_studies):
    ax = axes[j]

    # Plot posterior predictive distribution
    ax.hist(y_rep[:, j], bins=50, alpha=0.5, color='steelblue', density=True, label='Predicted')

    # Add observed value
    ax.axvline(y_obs[j], color='red', linewidth=2, linestyle='--', label='Observed')

    # Add posterior mean
    theta_mean = np.mean(theta_samples[:, j])
    ax.axvline(theta_mean, color='green', linewidth=2, linestyle=':', label=r'$\theta$ (mean)')

    # Add 95% predictive interval
    pred_lower = np.percentile(y_rep[:, j], 2.5)
    pred_upper = np.percentile(y_rep[:, j], 97.5)
    ax.axvspan(pred_lower, pred_upper, alpha=0.1, color='steelblue')

    ax.set_xlabel('Effect size')
    ax.set_ylabel('Density')
    ax.set_title(f'Study {j+1} (p={study_pvalues[j]:.3f}, σ={sigma[j]})')
    if j == 0:
        ax.legend(fontsize=8)

plt.suptitle('Posterior Predictive Distributions vs Observed Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/study_level_ppc.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: study_level_ppc.png")

# ============================================================================
# PLOT 2: Test statistics checks
# ============================================================================

# Extract test statistics
test_stats = ['mean', 'sd', 'min', 'max', 'range', 'n_negative']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, stat in enumerate(test_stats):
    ax = axes[idx]

    T_rep = results[f'T_rep_{stat}']
    T_obs = results[f'T_obs_{stat}']
    p_value = results[f'pvalue_{stat}']

    # Histogram of replicated statistics
    ax.hist(T_rep, bins=50, alpha=0.6, color='steelblue', edgecolor='black')

    # Add observed statistic
    ax.axvline(T_obs, color='red', linewidth=2, linestyle='--', label=f'Observed (p={p_value:.3f})')

    # Add percentiles
    p5 = np.percentile(T_rep, 5)
    p95 = np.percentile(T_rep, 95)
    ax.axvline(p5, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax.axvline(p95, color='gray', linewidth=1, linestyle=':', alpha=0.5)

    ax.set_xlabel(f'{stat}')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{stat.upper()}: T_obs={T_obs:.2f}')
    ax.legend()

plt.suptitle('Test Statistics: Observed vs Posterior Predictive Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/test_statistics_checks.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: test_statistics_checks.png")

# ============================================================================
# PLOT 3: Posterior predictive intervals
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Compute predictive intervals for each study
pred_means = np.mean(y_rep, axis=0)
pred_lower = np.percentile(y_rep, 2.5, axis=0)
pred_upper = np.percentile(y_rep, 97.5, axis=0)

# Compute posterior intervals for theta
theta_means = np.mean(theta_samples, axis=0)
theta_lower = np.percentile(theta_samples, 2.5, axis=0)
theta_upper = np.percentile(theta_samples, 97.5, axis=0)

studies = np.arange(1, n_studies + 1)

# Plot predictive intervals (wider)
ax.errorbar(studies, pred_means, yerr=[pred_means - pred_lower, pred_upper - pred_means],
            fmt='o', color='steelblue', alpha=0.3, linewidth=2, capsize=5, capthick=2,
            label='95% Predictive Interval', markersize=8)

# Plot posterior intervals for theta (narrower)
ax.errorbar(studies, theta_means, yerr=[theta_means - theta_lower, theta_upper - theta_means],
            fmt='s', color='green', alpha=0.6, linewidth=1.5, capsize=4, capthick=1.5,
            label='95% Posterior Interval (θ)', markersize=6)

# Plot observed data
ax.scatter(studies, y_obs, color='red', s=100, marker='D', zorder=5, label='Observed', edgecolors='darkred')

ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Effect Size', fontsize=12)
ax.set_title('Posterior Predictive Intervals vs Observed Data', fontsize=14, fontweight='bold')
ax.set_xticks(studies)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/predictive_intervals.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: predictive_intervals.png")

# ============================================================================
# PLOT 4: Standardized residuals
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Compute standardized residuals: (y_obs - theta_mean) / sigma
residuals = (y_obs - theta_means) / sigma

# Compute replicated standardized residuals
residuals_rep = (y_rep - theta_samples) / sigma[:, np.newaxis].T

# Plot distribution of replicated residuals
for j in range(n_studies):
    ax.scatter([j+1]*100, np.random.choice(residuals_rep[:, j], 100, replace=False),
              alpha=0.1, color='steelblue', s=10)

# Plot observed residuals
ax.scatter(studies, residuals, color='red', s=150, marker='D', zorder=5,
          label='Observed', edgecolors='darkred', linewidths=2)

# Add reference lines for ±2 SD
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='±2 SD')
ax.axhline(-2, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Standardized Residual (z-score)', fontsize=12)
ax.set_title('Standardized Residuals: Observed vs Predicted', fontsize=14, fontweight='bold')
ax.set_xticks(studies)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/standardized_residuals.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: standardized_residuals.png")

# ============================================================================
# PLOT 5: Q-Q plot for calibration
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Compute empirical quantiles of observed residuals
observed_quantiles = np.sort(residuals)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))

# Plot Q-Q
ax.scatter(theoretical_quantiles, observed_quantiles, s=100, color='red', edgecolors='darkred', linewidths=2)
ax.plot([-3, 3], [-3, 3], 'k--', linewidth=2, label='Perfect fit')

# Add confidence bands (approximate)
n = len(residuals)
se = 1.36 / np.sqrt(n)  # Approximate standard error
ax.fill_between([-3, 3], [-3-2*se, 3-2*se], [-3+2*se, 3+2*se], alpha=0.2, color='gray')

ax.set_xlabel('Theoretical Quantiles (Standard Normal)', fontsize=12)
ax.set_ylabel('Observed Standardized Residuals', fontsize=12)
ax.set_title('Q-Q Plot: Residual Calibration Check', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/qq_plot_calibration.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: qq_plot_calibration.png")

# ============================================================================
# PLOT 6: Pooled effect comparison
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Mean
ax = axes[0]
T_rep_mean = results['T_rep_mean']
T_obs_mean = results['T_obs_mean']
p_mean = results['pvalue_mean']

ax.hist(T_rep_mean, bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True)
ax.axvline(T_obs_mean, color='red', linewidth=2, linestyle='--', label=f'Observed (p={p_mean:.3f})')
ax.set_xlabel('Pooled Mean Effect', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Pooled Effect: Mean', fontsize=12, fontweight='bold')
ax.legend()

# Panel B: SD
ax = axes[1]
T_rep_sd = results['T_rep_sd']
T_obs_sd = results['T_obs_sd']
p_sd = results['pvalue_sd']

ax.hist(T_rep_sd, bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True)
ax.axvline(T_obs_sd, color='red', linewidth=2, linestyle='--', label=f'Observed (p={p_sd:.3f})')
ax.set_xlabel('Between-Study SD', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Dispersion: Standard Deviation', fontsize=12, fontweight='bold')
ax.legend()

# Panel C: Range
ax = axes[2]
T_rep_range = results['T_rep_range']
T_obs_range = results['T_obs_range']
p_range = results['pvalue_range']

ax.hist(T_rep_range, bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True)
ax.axvline(T_obs_range, color='red', linewidth=2, linestyle='--', label=f'Observed (p={p_range:.3f})')
ax.set_xlabel('Range (max - min)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Extremes: Range', fontsize=12, fontweight='bold')
ax.legend()

plt.suptitle('Pooled Statistics: Central Tendency, Dispersion, and Extremes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/pooled_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: pooled_statistics.png")

# ============================================================================
# PLOT 7: Distribution comparison (observed vs replicated)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Plot a sample of replicated datasets
for i in np.random.choice(n_samples, 100, replace=False):
    ax.scatter(np.arange(1, n_studies+1), y_rep[i, :], alpha=0.05, color='steelblue', s=50)

# Plot observed data
ax.scatter(np.arange(1, n_studies+1), y_obs, color='red', s=200, marker='D', zorder=5,
          label='Observed', edgecolors='darkred', linewidths=2)

ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Effect Size', fontsize=12)
ax.set_title('Observed Data vs 100 Replicated Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(np.arange(1, n_studies+1))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/observed_vs_replicated.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: observed_vs_replicated.png")

# ============================================================================
# PLOT 8: Study-specific p-value diagnostic
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['green' if 0.1 <= p <= 0.9 else 'orange' if 0.05 <= p < 0.1 or 0.9 < p <= 0.95 else 'red'
          for p in study_pvalues]

bars = ax.bar(np.arange(1, n_studies+1), study_pvalues, color=colors, edgecolor='black', linewidth=1.5)

# Add reference lines
ax.axhline(0.5, color='gray', linestyle='-', linewidth=2, alpha=0.5, label='Perfect fit (p=0.5)')
ax.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor fit threshold')
ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(0.1, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Marginal fit threshold')
ax.axhline(0.9, color='orange', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Bayesian p-value', fontsize=12)
ax.set_title('Study-Specific Bayesian p-values', fontsize=14, fontweight='bold')
ax.set_xticks(np.arange(1, n_studies+1))
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/study_pvalues.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Created: study_pvalues.png")

print("\nAll visualizations created successfully!")
