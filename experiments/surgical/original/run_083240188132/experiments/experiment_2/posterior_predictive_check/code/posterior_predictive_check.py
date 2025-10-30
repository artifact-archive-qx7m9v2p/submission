"""
Posterior Predictive Checks for Random Effects Logistic Regression
Experiment 2: Model Fit Assessment

This script:
1. Loads posterior samples from ArviZ InferenceData
2. Generates posterior predictive distributions for all groups
3. Compares observed vs predicted on multiple dimensions
4. Creates diagnostic visualizations
5. Assesses model adequacy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DATA_PATH = "/workspace/data/data.csv"
POSTERIOR_PATH = "/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf"
OUTPUT_DIR = "/workspace/experiments/experiment_2/posterior_predictive_check"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Plotting configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("=" * 80)
print("POSTERIOR PREDICTIVE CHECKS - EXPERIMENT 2")
print("Random Effects Logistic Regression Model")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND POSTERIOR
# ============================================================================

print("\n[1] Loading observed data and posterior samples...")

# Load observed data
data = pd.read_csv(DATA_PATH)
print(f"\nObserved data:")
print(data)

# Extract key variables
n_groups = len(data)
n_obs = data['n'].values
r_obs = data['r'].values
p_obs = data['proportion'].values

print(f"\nData summary:")
print(f"  Groups: {n_groups}")
print(f"  Total trials: {n_obs.sum()}")
print(f"  Total events: {r_obs.sum()}")
print(f"  Overall proportion: {r_obs.sum() / n_obs.sum():.4f}")

# Load posterior samples
print(f"\nLoading posterior from: {POSTERIOR_PATH}")
idata = az.from_netcdf(POSTERIOR_PATH)

print("\nPosterior structure:")
print(idata)

# Extract posterior samples
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values.reshape(-1, n_groups)

n_samples = len(mu_samples)
print(f"\nPosterior samples: {n_samples}")
print(f"  mu: mean={mu_samples.mean():.3f}, sd={mu_samples.std():.3f}")
print(f"  tau: mean={tau_samples.mean():.3f}, sd={tau_samples.std():.3f}")

# ============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================

print("\n[2] Generating posterior predictive samples...")

# For each posterior draw, generate replicated data
r_pred = np.zeros((n_samples, n_groups), dtype=int)

for i in range(n_samples):
    # Get group probabilities for this draw
    p_i = expit(theta_samples[i, :])

    # Simulate binomial outcomes
    for j in range(n_groups):
        r_pred[i, j] = np.random.binomial(n_obs[j], p_i[j])

    if (i + 1) % 1000 == 0:
        print(f"  Generated {i + 1}/{n_samples} samples...")

print(f"\nPosterior predictive shape: {r_pred.shape}")

# Compute posterior predictive summaries for each group
r_pred_mean = r_pred.mean(axis=0)
r_pred_sd = r_pred.std(axis=0)
r_pred_q025 = np.percentile(r_pred, 2.5, axis=0)
r_pred_q975 = np.percentile(r_pred, 97.5, axis=0)
r_pred_q05 = np.percentile(r_pred, 5, axis=0)
r_pred_q95 = np.percentile(r_pred, 95, axis=0)

# ============================================================================
# 3. GROUP-LEVEL FIT ASSESSMENT
# ============================================================================

print("\n[3] Assessing group-level fit...")

# Check coverage: Is observed within 95% predictive interval?
coverage_95 = (r_obs >= r_pred_q025) & (r_obs <= r_pred_q975)
coverage_90 = (r_obs >= r_pred_q05) & (r_obs <= r_pred_q95)

print("\nGroup-level coverage:")
print(f"  95% interval: {coverage_95.sum()}/{n_groups} ({100*coverage_95.mean():.1f}%)")
print(f"  90% interval: {coverage_90.sum()}/{n_groups} ({100*coverage_90.mean():.1f}%)")

# Detailed group-level results
print("\nDetailed group-level fit:")
print("-" * 100)
print(f"{'Group':<6} {'n':<6} {'r_obs':<8} {'r_pred':<15} {'95% CI':<20} {'In 95%':<8} {'In 90%':<8}")
print("-" * 100)

for i in range(n_groups):
    group_id = data['group'].iloc[i]
    in_95 = "YES" if coverage_95[i] else "NO"
    in_90 = "YES" if coverage_90[i] else "NO"

    print(f"{group_id:<6} {n_obs[i]:<6} {r_obs[i]:<8} "
          f"{r_pred_mean[i]:<6.1f} ({r_pred_sd[i]:.1f}) "
          f"[{r_pred_q025[i]:.0f}, {r_pred_q975[i]:.0f}]    "
          f"{in_95:<8} {in_90:<8}")

print("-" * 100)

# Compute standardized residuals
residuals = (r_obs - r_pred_mean) / (r_pred_sd + 1e-10)
print(f"\nStandardized residuals: mean={residuals.mean():.3f}, sd={residuals.std():.3f}")
print(f"  Range: [{residuals.min():.3f}, {residuals.max():.3f}]")
print(f"  |z| > 2: {(np.abs(residuals) > 2).sum()}/{n_groups}")
print(f"  |z| > 3: {(np.abs(residuals) > 3).sum()}/{n_groups}")

# ============================================================================
# 4. TEST STATISTICS
# ============================================================================

print("\n[4] Computing test statistics...")

# Test statistic 1: Total events
T_obs = r_obs.sum()
T_pred = r_pred.sum(axis=1)

T_pvalue = np.mean(T_pred >= T_obs) if T_obs <= T_pred.mean() else np.mean(T_pred <= T_obs)
T_pvalue = 2 * min(T_pvalue, 1 - T_pvalue)  # Two-tailed

print(f"\nTotal events:")
print(f"  Observed: {T_obs}")
print(f"  Predicted: {T_pred.mean():.1f} ({T_pred.std():.1f})")
print(f"  95% CI: [{np.percentile(T_pred, 2.5):.0f}, {np.percentile(T_pred, 97.5):.0f}]")
print(f"  Percentile rank: {100*np.mean(T_pred <= T_obs):.1f}%")
print(f"  P-value: {T_pvalue:.4f}")

# Test statistic 2: Between-group variance in proportions
prop_obs = r_obs / n_obs
prop_pred = r_pred / n_obs[:, np.newaxis].T

var_obs = np.var(prop_obs)
var_pred = np.var(prop_pred, axis=1)

var_pvalue = np.mean(var_pred >= var_obs) if var_obs <= var_pred.mean() else np.mean(var_pred <= var_obs)
var_pvalue = 2 * min(var_pvalue, 1 - var_pvalue)

print(f"\nBetween-group variance in proportions:")
print(f"  Observed: {var_obs:.6f}")
print(f"  Predicted: {var_pred.mean():.6f} ({var_pred.std():.6f})")
print(f"  95% CI: [{np.percentile(var_pred, 2.5):.6f}, {np.percentile(var_pred, 97.5):.6f}]")
print(f"  Percentile rank: {100*np.mean(var_pred <= var_obs):.1f}%")
print(f"  P-value: {var_pvalue:.4f}")

# Test statistic 3: Maximum proportion
max_prop_obs = prop_obs.max()
max_prop_pred = prop_pred.max(axis=1)

max_pvalue = np.mean(max_prop_pred >= max_prop_obs) if max_prop_obs <= max_prop_pred.mean() else np.mean(max_prop_pred <= max_prop_obs)
max_pvalue = 2 * min(max_pvalue, 1 - max_pvalue)

print(f"\nMaximum proportion:")
print(f"  Observed: {max_prop_obs:.4f}")
print(f"  Predicted: {max_prop_pred.mean():.4f} ({max_prop_pred.std():.4f})")
print(f"  95% CI: [{np.percentile(max_prop_pred, 2.5):.4f}, {np.percentile(max_prop_pred, 97.5):.4f}]")
print(f"  Percentile rank: {100*np.mean(max_prop_pred <= max_prop_obs):.1f}%")
print(f"  P-value: {max_pvalue:.4f}")

# Test statistic 4: Coefficient of variation
cv_obs = np.std(prop_obs) / np.mean(prop_obs)
cv_pred = np.std(prop_pred, axis=1) / np.mean(prop_pred, axis=1)

cv_pvalue = np.mean(cv_pred >= cv_obs) if cv_obs <= cv_pred.mean() else np.mean(cv_pred <= cv_obs)
cv_pvalue = 2 * min(cv_pvalue, 1 - cv_pvalue)

print(f"\nCoefficient of variation:")
print(f"  Observed: {cv_obs:.4f}")
print(f"  Predicted: {cv_pred.mean():.4f} ({cv_pred.std():.4f})")
print(f"  95% CI: [{np.percentile(cv_pred, 2.5):.4f}, {np.percentile(cv_pred, 97.5):.4f}]")
print(f"  Percentile rank: {100*np.mean(cv_pred <= cv_obs):.1f}%")
print(f"  P-value: {cv_pvalue:.4f}")

# Test statistic 5: Number of zeros
n_zeros_obs = (r_obs == 0).sum()
n_zeros_pred = (r_pred == 0).sum(axis=1)

zeros_pvalue = np.mean(n_zeros_pred >= n_zeros_obs) if n_zeros_obs <= n_zeros_pred.mean() else np.mean(n_zeros_pred <= n_zeros_obs)
zeros_pvalue = 2 * min(zeros_pvalue, 1 - zeros_pvalue)

print(f"\nNumber of zero-event groups:")
print(f"  Observed: {n_zeros_obs}")
print(f"  Predicted: {n_zeros_pred.mean():.2f} ({n_zeros_pred.std():.2f})")
print(f"  95% CI: [{np.percentile(n_zeros_pred, 2.5):.0f}, {np.percentile(n_zeros_pred, 97.5):.0f}]")
print(f"  Percentile rank: {100*np.mean(n_zeros_pred <= n_zeros_obs):.1f}%")
print(f"  P-value: {zeros_pvalue:.4f}")

# ============================================================================
# 5. SPECIFIC GROUP ASSESSMENTS
# ============================================================================

print("\n[5] Specific group assessments...")

# Group 1: Zero events
print("\nGroup 1 (zero events):")
r1_pred = r_pred[:, 0]
p_zero = np.mean(r1_pred == 0)
print(f"  P(r_pred = 0): {p_zero:.4f}")
print(f"  Mean predicted: {r1_pred.mean():.2f}")
print(f"  95% CI: [{np.percentile(r1_pred, 2.5):.0f}, {np.percentile(r1_pred, 97.5):.0f}]")
print(f"  Distribution of r_pred: {np.bincount(r1_pred, minlength=10)[:10]}")

# High-rate groups (2, 8, 11)
print("\nHigh-rate groups:")
for idx, group_id in [(1, 2), (7, 8), (10, 11)]:
    print(f"  Group {group_id}: r_obs={r_obs[idx]}, "
          f"r_pred={r_pred_mean[idx]:.1f} [{r_pred_q025[idx]:.0f}, {r_pred_q975[idx]:.0f}], "
          f"in 95% CI: {coverage_95[idx]}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n[6] Creating visualizations...")

# ----- Plot 1: Group-level posterior predictive distributions -----
print("  Creating group-level PPC plot...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(n_groups):
    ax = axes[i]

    # Histogram of posterior predictive
    ax.hist(r_pred[:, i], bins=20, alpha=0.6, color='steelblue',
            edgecolor='black', density=True, label='Predicted')

    # Observed value
    ax.axvline(r_obs[i], color='red', linewidth=2, linestyle='--', label='Observed')

    # 95% interval
    ax.axvline(r_pred_q025[i], color='gray', linewidth=1, linestyle=':')
    ax.axvline(r_pred_q975[i], color='gray', linewidth=1, linestyle=':',
               label='95% CI')

    # Labels
    group_id = data['group'].iloc[i]
    in_interval = "YES" if coverage_95[i] else "NO"
    ax.set_title(f'Group {group_id}: n={n_obs[i]}, r_obs={r_obs[i]} (In 95% CI: {in_interval})',
                 fontsize=9)
    ax.set_xlabel('Events', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.tick_params(labelsize=7)

    if i == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/group_level_ppc.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: group_level_ppc.png")

# ----- Plot 2: Observed vs Predicted with intervals -----
print("  Creating observed vs predicted plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot predictions with error bars
ax.errorbar(range(1, n_groups + 1), r_pred_mean,
            yerr=[r_pred_mean - r_pred_q025, r_pred_q975 - r_pred_mean],
            fmt='o', color='steelblue', markersize=8, capsize=5, capthick=2,
            alpha=0.7, label='Predicted (95% CI)')

# Plot observed
ax.scatter(range(1, n_groups + 1), r_obs, color='red', s=100, marker='x',
           linewidths=3, label='Observed', zorder=5)

# Highlight groups outside 95% CI
outside_ci = ~coverage_95
if outside_ci.any():
    outside_groups = np.where(outside_ci)[0] + 1
    ax.scatter(outside_groups, r_obs[outside_ci],
               facecolors='none', edgecolors='red', s=300, linewidths=2,
               label='Outside 95% CI')

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Number of Events', fontsize=12)
ax.set_title('Observed vs Posterior Predictive by Group', fontsize=14, fontweight='bold')
ax.set_xticks(range(1, n_groups + 1))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/observed_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: observed_vs_predicted.png")

# ----- Plot 3: Calibration plot (probability integral transform) -----
print("  Creating calibration plot...")

# Compute percentile rank of observed within posterior predictive
percentile_ranks = np.zeros(n_groups)
for i in range(n_groups):
    percentile_ranks[i] = np.mean(r_pred[:, i] <= r_obs[i])

# Sort for plotting
percentile_ranks_sorted = np.sort(percentile_ranks)
expected_uniform = np.linspace(0, 1, n_groups)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Calibration curve
ax1.plot(expected_uniform, percentile_ranks_sorted, 'o-', color='steelblue',
         markersize=8, linewidth=2, label='Observed')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
ax1.fill_between([0, 1], [0.025, 0.025], [0.975, 0.975],
                  alpha=0.2, color='gray', label='95% bounds')
ax1.set_xlabel('Expected Cumulative Probability', fontsize=12)
ax1.set_ylabel('Observed Cumulative Probability', fontsize=12)
ax1.set_title('Calibration Plot (PIT)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Histogram of percentile ranks
ax2.hist(percentile_ranks, bins=12, alpha=0.7, color='steelblue',
         edgecolor='black')
ax2.axhline(n_groups/12, color='red', linestyle='--', linewidth=2,
            label='Uniform expectation')
ax2.set_xlabel('Percentile Rank', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Percentile Ranks', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/calibration_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: calibration_plot.png")

# ----- Plot 4: Test statistics distributions -----
print("  Creating test statistics plot...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Total events
ax = axes[0, 0]
ax.hist(T_pred, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(T_obs, color='red', linewidth=2, linestyle='--', label='Observed')
ax.axvline(np.percentile(T_pred, 2.5), color='gray', linestyle=':', linewidth=1)
ax.axvline(np.percentile(T_pred, 97.5), color='gray', linestyle=':', linewidth=1,
           label='95% CI')
ax.set_xlabel('Total Events', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Total Events (p={T_pvalue:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Between-group variance
ax = axes[0, 1]
ax.hist(var_pred, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(var_obs, color='red', linewidth=2, linestyle='--', label='Observed')
ax.axvline(np.percentile(var_pred, 2.5), color='gray', linestyle=':', linewidth=1)
ax.axvline(np.percentile(var_pred, 97.5), color='gray', linestyle=':', linewidth=1,
           label='95% CI')
ax.set_xlabel('Variance of Proportions', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Between-Group Variance (p={var_pvalue:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Maximum proportion
ax = axes[0, 2]
ax.hist(max_prop_pred, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(max_prop_obs, color='red', linewidth=2, linestyle='--', label='Observed')
ax.axvline(np.percentile(max_prop_pred, 2.5), color='gray', linestyle=':', linewidth=1)
ax.axvline(np.percentile(max_prop_pred, 97.5), color='gray', linestyle=':', linewidth=1,
           label='95% CI')
ax.set_xlabel('Maximum Proportion', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Maximum Proportion (p={max_pvalue:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# CV
ax = axes[1, 0]
ax.hist(cv_pred, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(cv_obs, color='red', linewidth=2, linestyle='--', label='Observed')
ax.axvline(np.percentile(cv_pred, 2.5), color='gray', linestyle=':', linewidth=1)
ax.axvline(np.percentile(cv_pred, 97.5), color='gray', linestyle=':', linewidth=1,
           label='95% CI')
ax.set_xlabel('Coefficient of Variation', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Coefficient of Variation (p={cv_pvalue:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Number of zeros
ax = axes[1, 1]
ax.hist(n_zeros_pred, bins=range(int(n_zeros_pred.max()) + 2), alpha=0.7,
        color='steelblue', edgecolor='black', align='left')
ax.axvline(n_zeros_obs, color='red', linewidth=2, linestyle='--', label='Observed')
ax.set_xlabel('Number of Zero-Event Groups', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Zero-Event Groups (p={zeros_pvalue:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Standardized residuals
ax = axes[1, 2]
ax.hist(residuals, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0, color='black', linewidth=1, linestyle='-')
ax.axvline(-2, color='red', linewidth=1, linestyle=':', label='|z| = 2')
ax.axvline(2, color='red', linewidth=1, linestyle=':')
ax.set_xlabel('Standardized Residual', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Standardized Residuals', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/test_statistics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: test_statistics.png")

# ----- Plot 5: Residual diagnostics -----
print("  Creating residual diagnostics plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Residuals vs predicted
ax = axes[0, 0]
ax.scatter(r_pred_mean, residuals, s=100, alpha=0.7, color='steelblue')
for i in range(n_groups):
    ax.annotate(data['group'].iloc[i], (r_pred_mean[i], residuals[i]),
                fontsize=8, ha='right')
ax.axhline(0, color='black', linewidth=1)
ax.axhline(-2, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax.axhline(2, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('Predicted Events', fontsize=12)
ax.set_ylabel('Standardized Residual', fontsize=12)
ax.set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Residuals vs group size
ax = axes[0, 1]
ax.scatter(n_obs, residuals, s=100, alpha=0.7, color='steelblue')
for i in range(n_groups):
    ax.annotate(data['group'].iloc[i], (n_obs[i], residuals[i]),
                fontsize=8, ha='right')
ax.axhline(0, color='black', linewidth=1)
ax.axhline(-2, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax.axhline(2, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('Group Size (n)', fontsize=12)
ax.set_ylabel('Standardized Residual', fontsize=12)
ax.set_title('Residuals vs Group Size', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Q-Q plot
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Standardized Residuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Residuals vs group index
ax = axes[1, 1]
ax.scatter(range(1, n_groups + 1), residuals, s=100, alpha=0.7, color='steelblue')
for i in range(n_groups):
    ax.annotate(data['group'].iloc[i], (i + 1, residuals[i]),
                fontsize=8, ha='right')
ax.axhline(0, color='black', linewidth=1)
ax.axhline(-2, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax.axhline(2, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('Group Index', fontsize=12)
ax.set_ylabel('Standardized Residual', fontsize=12)
ax.set_title('Residuals by Group', fontsize=13, fontweight='bold')
ax.set_xticks(range(1, n_groups + 1))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: residual_diagnostics.png")

# ----- Plot 6: Scatter with 1:1 line -----
print("  Creating observed vs predicted scatter plot...")

fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot with error bars
for i in range(n_groups):
    color = 'red' if not coverage_95[i] else 'steelblue'
    ax.errorbar(r_obs[i], r_pred_mean[i],
                yerr=[[r_pred_mean[i] - r_pred_q025[i]], [r_pred_q975[i] - r_pred_mean[i]]],
                fmt='o', color=color, markersize=10, capsize=5, capthick=2, alpha=0.7)
    ax.annotate(data['group'].iloc[i], (r_obs[i], r_pred_mean[i]),
                fontsize=9, ha='left', va='bottom')

# 1:1 line
max_val = max(r_obs.max(), r_pred_q975.max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect fit (1:1)')

# Formatting
ax.set_xlabel('Observed Events', fontsize=13)
ax.set_ylabel('Predicted Events (Mean with 95% CI)', fontsize=13)
ax.set_title('Observed vs Predicted Events', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/scatter_1to1.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: scatter_1to1.png")

# ============================================================================
# 7. OVERALL ASSESSMENT
# ============================================================================

print("\n[7] Overall model assessment...")

# Decision criteria
coverage_criterion = coverage_95.mean() >= 0.85
test_stats_criterion = all([
    0.05 <= T_pvalue <= 0.95,
    0.05 <= var_pvalue <= 0.95,
    0.05 <= max_pvalue <= 0.95,
    0.05 <= cv_pvalue <= 0.95
])

extreme_residuals = np.abs(residuals).max() <= 3

if coverage_95.mean() >= 0.90 and test_stats_criterion and extreme_residuals:
    overall_fit = "GOOD"
elif coverage_95.mean() >= 0.85 and (test_stats_criterion or extreme_residuals):
    overall_fit = "ADEQUATE"
else:
    overall_fit = "POOR"

print("\n" + "=" * 80)
print("OVERALL MODEL FIT ASSESSMENT")
print("=" * 80)
print(f"\nDECISION: {overall_fit}")
print("\nCriteria:")
print(f"  Coverage (95% CI): {100*coverage_95.mean():.1f}% (target: >=85%, ideal: >=90%)")
print(f"  Test statistics: All within 90% predictive intervals: {test_stats_criterion}")
print(f"  Maximum |residual|: {np.abs(residuals).max():.2f} (should be <=3)")
print(f"  Groups with poor fit: {n_groups - coverage_95.sum()} / {n_groups}")

print("\nKey findings:")
if coverage_95.mean() >= 0.90:
    print("  + Excellent coverage: model captures group-level variation well")
else:
    print(f"  - Coverage slightly below ideal: {coverage_95.sum()}/{n_groups} groups")

if test_stats_criterion:
    print("  + All test statistics within expected range")
else:
    print("  - Some test statistics show systematic deviations")

if extreme_residuals:
    print("  + No extreme outliers (all |z| <= 3)")
else:
    print("  - Some extreme residuals detected")

# Save summary statistics
summary_stats = pd.DataFrame({
    'Statistic': ['Total Events', 'Between-Group Variance', 'Maximum Proportion',
                  'Coefficient of Variation', 'Number of Zeros'],
    'Observed': [T_obs, var_obs, max_prop_obs, cv_obs, n_zeros_obs],
    'Predicted_Mean': [T_pred.mean(), var_pred.mean(), max_prop_pred.mean(),
                       cv_pred.mean(), n_zeros_pred.mean()],
    'Predicted_SD': [T_pred.std(), var_pred.std(), max_prop_pred.std(),
                     cv_pred.std(), n_zeros_pred.std()],
    'CI_Lower': [np.percentile(T_pred, 2.5), np.percentile(var_pred, 2.5),
                 np.percentile(max_prop_pred, 2.5), np.percentile(cv_pred, 2.5),
                 np.percentile(n_zeros_pred, 2.5)],
    'CI_Upper': [np.percentile(T_pred, 97.5), np.percentile(var_pred, 97.5),
                 np.percentile(max_prop_pred, 97.5), np.percentile(cv_pred, 97.5),
                 np.percentile(n_zeros_pred, 97.5)],
    'P_Value': [T_pvalue, var_pvalue, max_pvalue, cv_pvalue, zeros_pvalue]
})

summary_stats.to_csv(f'{OUTPUT_DIR}/test_statistics_summary.csv', index=False)
print(f"\nSaved test statistics to: {OUTPUT_DIR}/test_statistics_summary.csv")

# Save group-level results
group_results = pd.DataFrame({
    'group': data['group'],
    'n': n_obs,
    'r_obs': r_obs,
    'r_pred_mean': r_pred_mean,
    'r_pred_sd': r_pred_sd,
    'r_pred_q025': r_pred_q025,
    'r_pred_q975': r_pred_q975,
    'in_95_CI': coverage_95,
    'in_90_CI': coverage_90,
    'standardized_residual': residuals,
    'percentile_rank': percentile_ranks
})

group_results.to_csv(f'{OUTPUT_DIR}/group_level_results.csv', index=False)
print(f"Saved group results to: {OUTPUT_DIR}/group_level_results.csv")

print("\n" + "=" * 80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("=" * 80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print(f"  - Plots: {PLOTS_DIR}/")
print(f"  - Summary statistics: test_statistics_summary.csv")
print(f"  - Group-level results: group_level_results.csv")
