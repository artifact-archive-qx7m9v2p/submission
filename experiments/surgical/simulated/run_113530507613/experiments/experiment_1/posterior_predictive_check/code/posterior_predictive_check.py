"""
Posterior Predictive Check for Experiment 1: Hierarchical Logit-Normal Model

This script performs comprehensive posterior predictive checks including:
1. Group-level PPC comparisons
2. Global test statistics
3. Residual diagnostics
4. Coverage calibration
5. Special focus on outlier groups (4 and 8)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Load data
print("Loading data and InferenceData...")
data = pd.read_csv('/workspace/data/data.csv')
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

# Extract observed and predicted data
r_obs = idata.observed_data['y'].values  # Observed successes (variable named 'y' in InferenceData)
n_trials = data['n_trials'].values  # Sample sizes from CSV
p_obs = r_obs / n_trials

# Extract posterior predictive samples
y_rep = idata.posterior_predictive['y'].values  # Shape: (chains, draws, groups)
# Reshape to (n_samples, n_groups)
n_chains, n_draws, n_groups = y_rep.shape
r_rep_flat = y_rep.reshape(-1, n_groups)
p_rep_flat = r_rep_flat / n_trials[np.newaxis, :]

print(f"Observed data shape: {r_obs.shape}")
print(f"Posterior predictive shape: {r_rep_flat.shape}")
print(f"Number of posterior samples: {r_rep_flat.shape[0]}")

# =============================================================================
# 1. GROUP-LEVEL PPC PLOTS
# =============================================================================
print("\n1. Creating group-level PPC plots...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

group_pvalues = []

for j in range(n_groups):
    ax = axes[j]

    # Get replicated success rates for this group
    p_rep_j = p_rep_flat[:, j]

    # Plot histogram of replicated data
    ax.hist(p_rep_j, bins=30, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5)

    # Add observed value as vertical line
    ax.axvline(p_obs[j], color='red', linewidth=2.5, label='Observed', linestyle='--')

    # Compute posterior predictive p-value
    p_value = np.mean(p_rep_j >= p_obs[j])
    group_pvalues.append(p_value)

    # Flag extreme p-values
    flag = ""
    if p_value < 0.025 or p_value > 0.975:
        flag = " [FLAGGED]"
        ax.set_facecolor('#ffeeee')

    # Add title and labels
    ax.set_title(f'Group {j+1}: n={n_trials[j]}, r={r_obs[j]}\np-value={p_value:.3f}{flag}',
                 fontsize=10, fontweight='bold' if flag else 'normal')
    ax.set_xlabel('Success Rate', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/group_level_ppc.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Group-level PPC plot saved.")
print(f"Groups with extreme p-values: {np.sum((np.array(group_pvalues) < 0.025) | (np.array(group_pvalues) > 0.975))}")

# =============================================================================
# 2. GLOBAL TEST STATISTICS
# =============================================================================
print("\n2. Computing global test statistics...")

# Define test statistics
test_stats = {
    'Mean': (np.mean(p_obs), np.mean(p_rep_flat, axis=1)),
    'SD': (np.std(p_obs), np.std(p_rep_flat, axis=1)),
    'Min': (np.min(p_obs), np.min(p_rep_flat, axis=1)),
    'Max': (np.max(p_obs), np.max(p_rep_flat, axis=1))
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

global_pvalues = {}

for idx, (stat_name, (obs_stat, rep_stats)) in enumerate(test_stats.items()):
    ax = axes[idx]

    # Compute p-value
    if stat_name in ['Min']:
        # For minimum, we test if observed is unusually low
        p_value = np.mean(rep_stats <= obs_stat)
    else:
        # For others, two-sided: distance from median
        p_value = np.mean(rep_stats >= obs_stat)

    global_pvalues[stat_name] = p_value

    # Plot histogram
    ax.hist(rep_stats, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5)
    ax.axvline(obs_stat, color='red', linewidth=2.5, label='Observed', linestyle='--')

    # Add median of replicated data
    median_rep = np.median(rep_stats)
    ax.axvline(median_rep, color='green', linewidth=2, label='Predicted Median',
               linestyle=':', alpha=0.7)

    # Add 95% interval
    lower, upper = np.percentile(rep_stats, [2.5, 97.5])
    ax.axvspan(lower, upper, alpha=0.2, color='gray', label='95% Interval')

    # Flag if extreme
    flag = ""
    if p_value < 0.01 or p_value > 0.99:
        flag = " [FAIL]"
        ax.set_facecolor('#ffcccc')
    elif p_value < 0.05 or p_value > 0.95:
        flag = " [CONCERN]"
        ax.set_facecolor('#fff4cc')

    ax.set_title(f'{stat_name} Success Rate\np-value = {p_value:.3f}{flag}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{stat_name} Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/global_test_statistics.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Global test statistics plot saved.")
for stat_name, p_value in global_pvalues.items():
    print(f"  {stat_name}: p = {p_value:.4f}")

# =============================================================================
# 3. RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n3. Creating residual diagnostics...")

# Compute posterior predictive mean and SD for each group
p_rep_mean = np.mean(p_rep_flat, axis=0)
p_rep_sd = np.std(p_rep_flat, axis=0)

# Standardized residuals
residuals = (p_obs - p_rep_mean) / p_rep_sd

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residuals vs fitted values
ax = axes[0, 0]
ax.scatter(p_rep_mean, residuals, s=100, alpha=0.6, c=n_trials, cmap='viridis')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Predicted Success Rate (Posterior Mean)', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Sample Size', fontsize=10)

# Add group labels for extreme residuals
for j in range(n_groups):
    if np.abs(residuals[j]) > 2:
        ax.annotate(f'G{j+1}', (p_rep_mean[j], residuals[j]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 2. Residuals vs sample size
ax = axes[0, 1]
ax.scatter(n_trials, residuals, s=100, alpha=0.6, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Sample Size (n)', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Residuals vs Sample Size', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add group labels for extreme residuals
for j in range(n_groups):
    if np.abs(residuals[j]) > 2:
        ax.annotate(f'G{j+1}', (n_trials[j], residuals[j]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 3. QQ plot
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. Histogram of residuals
ax = axes[1, 1]
ax.hist(residuals, bins=15, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=0.5)
x = np.linspace(-3, 3, 100)
ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='Standard Normal')
ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Standardized Residual', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/residual_diagnostics.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Residual diagnostics plot saved.")
print(f"Residuals outside [-2, 2]: {np.sum(np.abs(residuals) > 2)}")

# =============================================================================
# 4. COVERAGE CALIBRATION
# =============================================================================
print("\n4. Computing coverage calibration...")

# For each group, check if observed value falls in various credible intervals
nominal_levels = [0.50, 0.90, 0.95, 0.99]
empirical_coverage = []

coverage_results = []

for level in nominal_levels:
    alpha = 1 - level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    in_interval = []
    for j in range(n_groups):
        lower = np.percentile(p_rep_flat[:, j], lower_percentile)
        upper = np.percentile(p_rep_flat[:, j], upper_percentile)
        in_interval.append(lower <= p_obs[j] <= upper)

    empirical = np.mean(in_interval)
    empirical_coverage.append(empirical)
    coverage_results.append({
        'Nominal': level,
        'Empirical': empirical,
        'N_covered': np.sum(in_interval),
        'N_total': n_groups
    })
    print(f"  {level*100:.0f}% interval: {empirical*100:.1f}% coverage ({np.sum(in_interval)}/{n_groups})")

# Create calibration plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Calibration curve
ax = axes[0]
ax.plot(nominal_levels, nominal_levels, 'k--', linewidth=2, label='Perfect Calibration')
ax.plot(nominal_levels, empirical_coverage, 'o-', linewidth=2, markersize=10,
        color='steelblue', label='Observed Coverage')

# Add error bars (binomial confidence intervals)
for i, (nom, emp) in enumerate(zip(nominal_levels, empirical_coverage)):
    # Wilson score interval
    n = n_groups
    p_hat = emp
    z = 1.96  # 95% CI
    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    ax.plot([nom, nom], [lower, upper], 'k-', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Nominal Coverage', fontsize=12)
ax.set_ylabel('Empirical Coverage', fontsize=12)
ax.set_title('Coverage Calibration', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.45, 1.02])
ax.set_ylim([0.45, 1.02])

# 2. Coverage by group
ax = axes[1]
coverage_by_group = []
for j in range(n_groups):
    # Check 95% interval
    lower = np.percentile(p_rep_flat[:, j], 2.5)
    upper = np.percentile(p_rep_flat[:, j], 97.5)
    in_interval = lower <= p_obs[j] <= upper
    coverage_by_group.append(in_interval)

colors = ['green' if covered else 'red' for covered in coverage_by_group]
ax.bar(range(1, n_groups+1), [1]*n_groups, color=colors, alpha=0.6, edgecolor='black')
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('In 95% Interval', fontsize=12)
ax.set_title('95% Coverage by Group', fontsize=13, fontweight='bold')
ax.set_xticks(range(1, n_groups+1))
ax.set_ylim([0, 1.2])
ax.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.6, label='Covered'),
                   Patch(facecolor='red', alpha=0.6, label='Not Covered')]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/calibration_plot.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Calibration plot saved.")

# =============================================================================
# 5. OUTLIER ANALYSIS (Groups 4 and 8)
# =============================================================================
print("\n5. Creating outlier analysis (Groups 4 and 8)...")

outlier_groups = [3, 7]  # 0-indexed (Groups 4 and 8)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, group_idx in enumerate(outlier_groups):
    group_num = group_idx + 1

    # Column 1: PPC density
    ax = axes[idx, 0]
    p_rep_j = p_rep_flat[:, group_idx]
    ax.hist(p_rep_j, bins=40, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5)
    ax.axvline(p_obs[group_idx], color='red', linewidth=2.5, label='Observed', linestyle='--')

    # Add percentiles
    percentiles = [2.5, 25, 50, 75, 97.5]
    for pct in percentiles:
        val = np.percentile(p_rep_j, pct)
        ax.axvline(val, color='gray', linewidth=1, linestyle=':', alpha=0.5)

    ax.set_xlabel('Success Rate', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Group {group_num} PPC\nn={n_trials[group_idx]}, r={r_obs[group_idx]}, p={p_obs[group_idx]:.4f}\np-value={group_pvalues[group_idx]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Column 2: Cumulative distribution
    ax = axes[idx, 1]
    sorted_rep = np.sort(p_rep_j)
    cumulative = np.arange(1, len(sorted_rep)+1) / len(sorted_rep)
    ax.plot(sorted_rep, cumulative, linewidth=2, color='steelblue')
    ax.axvline(p_obs[group_idx], color='red', linewidth=2.5, label='Observed', linestyle='--')
    ax.axhline(group_pvalues[group_idx], color='red', linewidth=1.5, linestyle=':', alpha=0.7)
    ax.set_xlabel('Success Rate', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'Group {group_num} CDF', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Column 3: Trajectory plot (sample of posterior predictive draws)
    ax = axes[idx, 2]
    n_sample_draws = 100
    sample_indices = np.random.choice(r_rep_flat.shape[0], n_sample_draws, replace=False)
    for draw_idx in sample_indices:
        ax.plot([0, 1], [p_rep_flat[draw_idx, group_idx], p_rep_flat[draw_idx, group_idx]],
                color='steelblue', alpha=0.1, linewidth=1)

    # Add summary statistics
    mean_val = p_rep_mean[group_idx]
    lower = np.percentile(p_rep_j, 2.5)
    upper = np.percentile(p_rep_j, 97.5)

    ax.plot([0, 1], [mean_val, mean_val], color='blue', linewidth=3, label='Posterior Mean')
    ax.fill_between([0, 1], [lower, lower], [upper, upper], alpha=0.3, color='blue',
                     label='95% Interval')
    ax.plot([0, 1], [p_obs[group_idx], p_obs[group_idx]], color='red', linewidth=3,
            linestyle='--', label='Observed')

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0, max(upper, p_obs[group_idx]) * 1.2])
    ax.set_xticks([])
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_title(f'Group {group_num} Posterior Summary', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/outlier_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Outlier analysis plot saved.")

# =============================================================================
# 6. ADDITIONAL DIAGNOSTIC: OVERDISPERSION CHECK
# =============================================================================
print("\n6. Checking for overdispersion...")

# Compute observed and predicted variance
variance_obs = np.var(p_obs)
variance_rep = np.var(p_rep_flat, axis=1)  # Variance across groups for each posterior draw

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Variance distribution
ax = axes[0]
ax.hist(variance_rep, bins=50, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=0.5)
ax.axvline(variance_obs, color='red', linewidth=2.5, label='Observed Variance', linestyle='--')
median_var = np.median(variance_rep)
ax.axvline(median_var, color='green', linewidth=2, label='Predicted Median',
           linestyle=':', alpha=0.7)

p_value_var = np.mean(variance_rep >= variance_obs)
ax.set_xlabel('Variance of Success Rates', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Between-Group Variance\np-value = {p_value_var:.3f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Range (max - min) distribution
ax = axes[1]
range_obs = np.max(p_obs) - np.min(p_obs)
range_rep = np.max(p_rep_flat, axis=1) - np.min(p_rep_flat, axis=1)

ax.hist(range_rep, bins=50, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=0.5)
ax.axvline(range_obs, color='red', linewidth=2.5, label='Observed Range', linestyle='--')
median_range = np.median(range_rep)
ax.axvline(median_range, color='green', linewidth=2, label='Predicted Median',
           linestyle=':', alpha=0.7)

p_value_range = np.mean(range_rep >= range_obs)
ax.set_xlabel('Range of Success Rates', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Range (Max - Min)\np-value = {p_value_range:.3f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/overdispersion_check.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"Overdispersion check plot saved.")
print(f"  Variance p-value: {p_value_var:.4f}")
print(f"  Range p-value: {p_value_range:.4f}")

# =============================================================================
# 7. SAVE RESULTS TO CSV
# =============================================================================
print("\n7. Saving results to CSV...")

# Group-level results
group_results = pd.DataFrame({
    'group_id': range(1, n_groups+1),
    'n_trials': n_trials,
    'r_observed': r_obs,
    'p_observed': p_obs,
    'p_predicted_mean': p_rep_mean,
    'p_predicted_sd': p_rep_sd,
    'residual_std': residuals,
    'p_value': group_pvalues,
    'flagged': (np.array(group_pvalues) < 0.025) | (np.array(group_pvalues) > 0.975),
    'in_95_interval': coverage_by_group
})

group_results.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/code/group_level_results.csv',
                     index=False)
print("Group-level results saved.")

# Global statistics results
global_results = pd.DataFrame({
    'statistic': list(global_pvalues.keys()),
    'p_value': list(global_pvalues.values()),
    'status': [
        'PASS' if 0.05 <= p <= 0.95 else ('CONCERN' if 0.01 <= p <= 0.99 else 'FAIL')
        for p in global_pvalues.values()
    ]
})

global_results.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/code/global_statistics_results.csv',
                      index=False)
print("Global statistics results saved.")

# Coverage results
coverage_df = pd.DataFrame(coverage_results)
coverage_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/code/coverage_results.csv',
                   index=False)
print("Coverage results saved.")

# =============================================================================
# 8. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECK SUMMARY")
print("="*80)

print("\n1. GROUP-LEVEL ASSESSMENT:")
n_flagged = np.sum((np.array(group_pvalues) < 0.025) | (np.array(group_pvalues) > 0.975))
pct_flagged = 100 * n_flagged / n_groups
print(f"   - Groups flagged (p < 0.025 or p > 0.975): {n_flagged}/{n_groups} ({pct_flagged:.1f}%)")
print(f"   - Groups with extreme residuals (|z| > 2): {np.sum(np.abs(residuals) > 2)}/{n_groups}")

print("\n2. GLOBAL TEST STATISTICS:")
for stat_name, p_value in global_pvalues.items():
    status = 'PASS' if 0.05 <= p_value <= 0.95 else ('CONCERN' if 0.01 <= p_value <= 0.99 else 'FAIL')
    print(f"   - {stat_name}: p = {p_value:.4f} [{status}]")

print("\n3. COVERAGE CALIBRATION:")
for result in coverage_results:
    print(f"   - {result['Nominal']*100:.0f}% interval: {result['Empirical']*100:.1f}% coverage ({result['N_covered']}/{result['N_total']})")

print("\n4. OUTLIER GROUPS (4 and 8):")
for group_idx in outlier_groups:
    group_num = group_idx + 1
    print(f"   - Group {group_num}: p-value = {group_pvalues[group_idx]:.4f}, residual = {residuals[group_idx]:.2f}")

print("\n5. OVERALL ASSESSMENT:")
# Determine overall status
global_fail = any(p < 0.01 or p > 0.99 for p in global_pvalues.values())
global_concern = any(p < 0.05 or p > 0.95 for p in global_pvalues.values())

if global_fail or pct_flagged > 20:
    overall = "FAIL"
elif global_concern or pct_flagged > 10:
    overall = "CONCERN"
else:
    overall = "PASS"

print(f"   Status: {overall}")
print(f"   Justification:")
print(f"   - Global statistics: {'FAIL' if global_fail else ('CONCERN' if global_concern else 'PASS')}")
print(f"   - Group-level: {pct_flagged:.1f}% flagged (threshold: 10% concern, 20% fail)")
print(f"   - Calibration: {'Good' if all(0.8 <= ec/nom <= 1.2 for ec, nom in zip(empirical_coverage, nominal_levels)) else 'Poor'}")

print("\n" + "="*80)
print("Analysis complete. All plots and results saved.")
print("="*80)
