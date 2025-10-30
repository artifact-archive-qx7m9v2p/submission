"""
Posterior Predictive Checks for Beta-Binomial Model
Experiment 1: Validate model's ability to reproduce observed data patterns

This script performs comprehensive posterior predictive checking:
1. Load posterior samples and posterior predictive samples from ArviZ InferenceData
2. Compute test statistics comparing observed vs replicated data
3. Generate diagnostic visualizations
4. Perform LOO cross-validation
5. Assess model adequacy

Author: Bayesian Model Validation Specialist
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
sns.set_palette("husl")

print("=" * 80)
print("POSTERIOR PREDICTIVE CHECKS - EXPERIMENT 1")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA AND POSTERIOR SAMPLES
# =============================================================================

print("\n[1] Loading data and posterior samples...")

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
n_groups = len(data)
r_obs = data['r_successes'].values
n_trials = data['n_trials'].values
rates_obs = r_obs / n_trials

print(f"  - Loaded {n_groups} groups")
print(f"  - Total trials: {n_trials.sum()}")
print(f"  - Total successes: {r_obs.sum()}")
print(f"  - Observed pooled rate: {r_obs.sum() / n_trials.sum():.4f}")

# Load ArviZ InferenceData
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print(f"  - Loaded InferenceData with {len(idata.posterior.chain) * len(idata.posterior.draw)} samples")
print(f"  - Available groups: {list(idata.groups())}")

# Extract posterior predictive samples
y_rep = idata.posterior_predictive['y_rep'].values  # shape: (chains, draws, groups)
n_chains, n_draws, n_groups_check = y_rep.shape
n_samples = n_chains * n_draws

# Reshape to (samples, groups)
y_rep = y_rep.reshape(n_samples, n_groups)

print(f"  - Posterior predictive samples: {n_samples} x {n_groups}")
print(f"  - Sample y_rep[0]: {y_rep[0]}")

# Extract posterior samples for parameters
mu_samples = idata.posterior['mu'].values.flatten()
kappa_samples = idata.posterior['kappa'].values.flatten()
phi_samples = idata.posterior['phi'].values.flatten()

print(f"  - Parameter samples extracted (n={len(mu_samples)})")

# =============================================================================
# 2. COMPUTE TEST STATISTICS
# =============================================================================

print("\n[2] Computing test statistics...")

# Define test statistic functions
def total_successes(y):
    """Sum of successes across all groups"""
    return np.sum(y)

def variance_rates(y, n):
    """Variance of success rates across groups"""
    rates = y / n
    return np.var(rates)

def max_rate(y, n):
    """Maximum success rate across groups"""
    rates = y / n
    return np.max(rates)

def num_zeros(y):
    """Number of groups with zero successes"""
    return np.sum(y == 0)

def range_rates(y, n):
    """Range of success rates"""
    rates = y / n
    return np.max(rates) - np.min(rates)

def chi_square_stat(y, expected):
    """Chi-square goodness of fit statistic"""
    # Avoid division by zero
    expected_safe = np.where(expected > 0, expected, 1)
    return np.sum((y - expected_safe)**2 / expected_safe)

# Compute observed test statistics
test_stats_obs = {
    'total_successes': total_successes(r_obs),
    'variance_rates': variance_rates(r_obs, n_trials),
    'max_rate': max_rate(r_obs, n_trials),
    'num_zeros': num_zeros(r_obs),
    'range_rates': range_rates(r_obs, n_trials),
}

print("\nObserved test statistics:")
for stat_name, stat_value in test_stats_obs.items():
    print(f"  - {stat_name}: {stat_value:.6f}")

# Compute replicated test statistics
test_stats_rep = {
    'total_successes': np.array([total_successes(y_rep[i]) for i in range(n_samples)]),
    'variance_rates': np.array([variance_rates(y_rep[i], n_trials) for i in range(n_samples)]),
    'max_rate': np.array([max_rate(y_rep[i], n_trials) for i in range(n_samples)]),
    'num_zeros': np.array([num_zeros(y_rep[i]) for i in range(n_samples)]),
    'range_rates': np.array([range_rates(y_rep[i], n_trials) for i in range(n_samples)]),
}

# Compute chi-square for each replicate
# Expected counts under posterior mean
mu_post_mean = mu_samples.mean()
expected_counts = n_trials * mu_post_mean
chi_square_obs = chi_square_stat(r_obs, expected_counts)
chi_square_rep = np.array([chi_square_stat(y_rep[i], expected_counts) for i in range(n_samples)])

test_stats_obs['chi_square'] = chi_square_obs
test_stats_rep['chi_square'] = chi_square_rep

print(f"\n  - Chi-square observed: {chi_square_obs:.4f}")

# Compute p-values (two-sided)
p_values = {}
for stat_name in test_stats_obs.keys():
    obs_val = test_stats_obs[stat_name]
    rep_vals = test_stats_rep[stat_name]

    # Bayesian p-value: P(T_rep >= T_obs | data)
    p_upper = np.mean(rep_vals >= obs_val)
    p_lower = np.mean(rep_vals <= obs_val)

    # Two-sided: min(2*p_lower, 2*p_upper, 1.0)
    # But for interpretation, we report the tail probability
    p_val = min(p_upper, p_lower)
    if p_val < 0.5:
        p_val = 2 * p_val

    p_values[stat_name] = p_upper  # Store upper tail for interpretation

    print(f"  - {stat_name}: p-value = {p_upper:.4f} (prop. replicated >= observed)")

# =============================================================================
# 3. CREATE TEST STATISTIC SUMMARY TABLE
# =============================================================================

print("\n[3] Creating test statistic summary table...")

summary_rows = []
for stat_name in test_stats_obs.keys():
    obs_val = test_stats_obs[stat_name]
    rep_vals = test_stats_rep[stat_name]

    row = {
        'Test Statistic': stat_name.replace('_', ' ').title(),
        'Observed': f"{obs_val:.4f}",
        'Post. Pred. Mean': f"{rep_vals.mean():.4f}",
        'Post. Pred. SD': f"{rep_vals.std():.4f}",
        '95% Pred. Interval': f"[{np.percentile(rep_vals, 2.5):.4f}, {np.percentile(rep_vals, 97.5):.4f}]",
        'p-value': f"{p_values[stat_name]:.4f}",
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/results/test_statistics_summary.csv',
                   index=False)

print("\n" + summary_df.to_string(index=False))

# =============================================================================
# 4. LOO CROSS-VALIDATION
# =============================================================================

print("\n[4] Performing LOO cross-validation...")

# Compute LOO
loo_result = az.loo(idata, pointwise=True)

print(f"  - LOO ELPD: {loo_result.elpd_loo:.2f} (SE: {loo_result.se:.2f})")
print(f"  - LOO IC: {loo_result.loo_i:.2f}")
print(f"  - p_loo (effective parameters): {loo_result.p_loo:.2f}")

# Extract Pareto k values
pareto_k = loo_result.pareto_k.values

print(f"\nPareto k diagnostics:")
print(f"  - Mean k: {pareto_k.mean():.4f}")
print(f"  - Max k: {pareto_k.max():.4f}")
print(f"  - k < 0.5 (good): {np.sum(pareto_k < 0.5)} groups")
print(f"  - 0.5 <= k < 0.7 (ok): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))} groups")
print(f"  - k >= 0.7 (problematic): {np.sum(pareto_k >= 0.7)} groups")

# Create LOO summary by group
loo_summary = pd.DataFrame({
    'group': range(1, n_groups + 1),
    'n_trials': n_trials,
    'r_successes': r_obs,
    'pareto_k': pareto_k,
    'loo_i': loo_result.loo_i.values,
})
loo_summary.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/results/loo_summary.csv',
                    index=False)

print("\nGroups with highest Pareto k:")
print(loo_summary.nlargest(5, 'pareto_k')[['group', 'r_successes', 'n_trials', 'pareto_k']])

# =============================================================================
# 5. LOO-PIT (Probability Integral Transform)
# =============================================================================

print("\n[5] Computing LOO-PIT for calibration assessment...")

# Compute PIT values
try:
    # For discrete data, we need to use the discrete PIT
    # ArviZ handles this internally

    # We'll compute it manually for better control
    # PIT = P(y_rep <= y_obs | data, leaving out observation i)

    pit_values = []

    for i in range(n_groups):
        # For each group, compute the probability that a replicated value
        # is less than or equal to the observed value
        y_obs_i = r_obs[i]
        y_rep_i = y_rep[:, i]

        # For discrete data, PIT is the CDF at the observed value
        # plus a uniform random variable times the probability mass at the observed value
        p_less = np.mean(y_rep_i < y_obs_i)
        p_equal = np.mean(y_rep_i == y_obs_i)

        # Randomized PIT for discrete data
        u = np.random.uniform(0, 1)
        pit = p_less + u * p_equal

        pit_values.append(pit)

    pit_values = np.array(pit_values)

    print(f"  - PIT computed for {n_groups} groups")
    print(f"  - PIT range: [{pit_values.min():.3f}, {pit_values.max():.3f}]")

    # Test for uniformity using Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
    print(f"  - KS test for uniformity: D={ks_stat:.4f}, p={ks_pval:.4f}")

    # Save PIT values
    pit_df = pd.DataFrame({
        'group': range(1, n_groups + 1),
        'pit_value': pit_values
    })
    pit_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/results/pit_values.csv',
                   index=False)

except Exception as e:
    print(f"  - Warning: Could not compute PIT: {e}")
    pit_values = None

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================

print("\n[6] Creating visualizations...")

# -----------------------------------------------------------------------------
# 6.1 Density Overlay Plot
# -----------------------------------------------------------------------------

print("  - Creating density overlay plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot a subset of posterior predictive samples (to avoid overplotting)
n_overlay = min(100, n_samples)
indices = np.random.choice(n_samples, n_overlay, replace=False)

for idx in indices:
    rates_rep = y_rep[idx] / n_trials
    ax.plot(range(1, n_groups + 1), rates_rep, 'o-', color='skyblue', alpha=0.1, linewidth=0.5)

# Plot observed data on top
ax.plot(range(1, n_groups + 1), rates_obs, 'o-', color='red', linewidth=2,
        markersize=8, label='Observed', zorder=10)

# Add population mean
mu_mean = mu_samples.mean()
ax.axhline(mu_mean, color='darkgreen', linestyle='--', linewidth=2,
           label=f'Population mean (μ = {mu_mean:.3f})')

ax.set_xlabel('Group')
ax.set_ylabel('Success Rate')
ax.set_title('Posterior Predictive Check: Density Overlay\n' +
             f'{n_overlay} replicated datasets vs. observed', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, n_groups + 1))

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/ppc_density_overlay.png',
            dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 6.2 Test Statistics Panel
# -----------------------------------------------------------------------------

print("  - Creating test statistics panel...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

stat_names = list(test_stats_obs.keys())

for idx, stat_name in enumerate(stat_names):
    ax = axes[idx]

    obs_val = test_stats_obs[stat_name]
    rep_vals = test_stats_rep[stat_name]
    p_val = p_values[stat_name]

    # Histogram of replicated statistics
    ax.hist(rep_vals, bins=50, alpha=0.6, color='skyblue', edgecolor='black', linewidth=0.5)

    # Mark observed value
    ax.axvline(obs_val, color='red', linewidth=2, linestyle='--', label='Observed')

    # Add percentile information
    percentile = np.mean(rep_vals <= obs_val) * 100

    ax.set_xlabel(stat_name.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.set_title(f'{stat_name.replace("_", " ").title()}\n' +
                 f'p-value = {p_val:.3f}, percentile = {percentile:.1f}%',
                 fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Check: Test Statistics',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/ppc_test_statistics.png',
            dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 6.3 Group-Specific Checks
# -----------------------------------------------------------------------------

print("  - Creating group-specific checks...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(n_groups):
    ax = axes[i]

    # Get replicated values for this group
    y_rep_i = y_rep[:, i]

    # Create histogram of replicated values
    bins = np.arange(y_rep_i.min() - 0.5, y_rep_i.max() + 1.5, 1)
    ax.hist(y_rep_i, bins=bins, alpha=0.6, color='skyblue',
            edgecolor='black', linewidth=0.5, density=True)

    # Mark observed value
    ax.axvline(r_obs[i], color='red', linewidth=2, linestyle='--', label='Observed')

    # Add title with group info
    ax.set_title(f'Group {i+1}: n={n_trials[i]}, r={r_obs[i]}\n' +
                 f'Rate: {rates_obs[i]:.3f}', fontsize=9)
    ax.set_xlabel('Successes')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Check: Group-Specific Distributions',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/ppc_group_specific.png',
            dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 6.4 LOO Diagnostics
# -----------------------------------------------------------------------------

print("  - Creating LOO diagnostics plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Pareto k by group
ax = axes[0]
colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in pareto_k]
ax.bar(range(1, n_groups + 1), pareto_k, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='k = 0.5 (threshold)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='k = 0.7 (problematic)')
ax.set_xlabel('Group')
ax.set_ylabel('Pareto k')
ax.set_title('LOO Pareto k Diagnostics', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, n_groups + 1))

# Panel 2: Pareto k vs sample size
ax = axes[1]
ax.scatter(n_trials, pareto_k, s=100, alpha=0.6, c=colors, edgecolor='black')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5)
ax.axhline(0.7, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('Pareto k')
ax.set_title('Pareto k vs Sample Size', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Annotate problematic groups
for i in range(n_groups):
    if pareto_k[i] >= 0.5:
        ax.annotate(f'Grp {i+1}', (n_trials[i], pareto_k[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/loo_diagnostics.png',
            dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 6.5 LOO-PIT Calibration
# -----------------------------------------------------------------------------

if pit_values is not None:
    print("  - Creating LOO-PIT calibration plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Histogram
    ax = axes[0]
    ax.hist(pit_values, bins=20, alpha=0.6, color='skyblue',
            edgecolor='black', linewidth=1, density=True)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
               label='Uniform (ideal)')
    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.set_title('LOO-PIT Histogram\n' +
                 f'KS test: D={ks_stat:.3f}, p={ks_pval:.3f}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Panel 2: ECDF
    ax = axes[1]
    pit_sorted = np.sort(pit_values)
    ecdf = np.arange(1, n_groups + 1) / n_groups
    ax.plot(pit_sorted, ecdf, 'o-', linewidth=2, markersize=6, label='Observed ECDF')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal (uniform)')

    # Add confidence bands (approximate)
    alpha = 0.05
    epsilon = np.sqrt(np.log(2/alpha) / (2*n_groups))
    ax.fill_between([0, 1], [max(0, -epsilon), max(0, 1-epsilon)],
                     [min(1, epsilon), min(1, 1+epsilon)],
                     alpha=0.2, color='gray', label=f'{100*(1-alpha):.0f}% confidence band')

    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('LOO-PIT Empirical CDF', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/loo_pit_calibration.png',
                dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 6.6 Comprehensive Summary Dashboard
# -----------------------------------------------------------------------------

print("  - Creating comprehensive summary dashboard...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Panel 1: Density overlay (smaller version)
ax1 = fig.add_subplot(gs[0, :2])
for idx in indices[:50]:  # Fewer lines for clarity
    rates_rep = y_rep[idx] / n_trials
    ax1.plot(range(1, n_groups + 1), rates_rep, 'o-', color='skyblue',
            alpha=0.15, linewidth=0.5)
ax1.plot(range(1, n_groups + 1), rates_obs, 'o-', color='red',
        linewidth=2, markersize=8, label='Observed', zorder=10)
ax1.axhline(mu_mean, color='darkgreen', linestyle='--', linewidth=2,
           label=f'μ = {mu_mean:.3f}')
ax1.set_xlabel('Group')
ax1.set_ylabel('Success Rate')
ax1.set_title('A. Observed vs Posterior Predictive', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, n_groups + 1))

# Panel 2: Test statistics summary
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
test_stat_text = "B. Test Statistics\n\n"
for stat_name in ['total_successes', 'variance_rates', 'max_rate']:
    p_val = p_values[stat_name]
    status = "PASS" if 0.05 < p_val < 0.95 else "BORDERLINE" if 0.01 < p_val < 0.99 else "FAIL"
    test_stat_text += f"{stat_name.replace('_', ' ').title()}:\n"
    test_stat_text += f"  p = {p_val:.3f} [{status}]\n\n"
ax2.text(0.05, 0.95, test_stat_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace')

# Panels 3-5: Selected test statistics
for idx, stat_name in enumerate(['total_successes', 'variance_rates', 'max_rate']):
    ax = fig.add_subplot(gs[1, idx])
    obs_val = test_stats_obs[stat_name]
    rep_vals = test_stats_rep[stat_name]
    p_val = p_values[stat_name]

    ax.hist(rep_vals, bins=30, alpha=0.6, color='skyblue', edgecolor='black', linewidth=0.5)
    ax.axvline(obs_val, color='red', linewidth=2, linestyle='--', label='Observed')
    ax.set_xlabel(stat_name.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.set_title(f'{chr(67+idx)}. {stat_name.replace("_", " ").title()}\np = {p_val:.3f}',
                fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Panel 6: Pareto k
ax6 = fig.add_subplot(gs[2, 0])
colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in pareto_k]
ax6.bar(range(1, n_groups + 1), pareto_k, color=colors, alpha=0.7, edgecolor='black')
ax6.axhline(0.5, color='orange', linestyle='--', linewidth=1.5)
ax6.axhline(0.7, color='red', linestyle='--', linewidth=1.5)
ax6.set_xlabel('Group')
ax6.set_ylabel('Pareto k')
ax6.set_title('F. LOO Pareto k Diagnostics', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xticks(range(1, n_groups + 1))

# Panel 7: Group 1 (zero count)
ax7 = fig.add_subplot(gs[2, 1])
y_rep_1 = y_rep[:, 0]
bins = np.arange(y_rep_1.min() - 0.5, y_rep_1.max() + 1.5, 1)
ax7.hist(y_rep_1, bins=bins, alpha=0.6, color='skyblue', edgecolor='black', linewidth=0.5)
ax7.axvline(r_obs[0], color='red', linewidth=2, linestyle='--', label='Observed')
ax7.set_xlabel('Successes')
ax7.set_ylabel('Frequency')
ax7.set_title(f'G. Group 1 (Zero Count)\nn={n_trials[0]}, r={r_obs[0]}', fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Panel 8: Group 8 (outlier)
ax8 = fig.add_subplot(gs[2, 2])
y_rep_8 = y_rep[:, 7]
bins = np.arange(y_rep_8.min() - 0.5, y_rep_8.max() + 1.5, 1)
ax8.hist(y_rep_8, bins=bins, alpha=0.6, color='skyblue', edgecolor='black', linewidth=0.5)
ax8.axvline(r_obs[7], color='red', linewidth=2, linestyle='--', label='Observed')
ax8.set_xlabel('Successes')
ax8.set_ylabel('Frequency')
ax8.set_title(f'H. Group 8 (High Rate)\nn={n_trials[7]}, r={r_obs[7]}', fontweight='bold')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Check: Comprehensive Summary Dashboard',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/ppc_summary_dashboard.png',
            dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. PASS/FAIL ASSESSMENT
# =============================================================================

print("\n[7] Assessing model adequacy...")

# Define criteria
criteria = {
    'all_pvals_reasonable': all(0.01 < p_values[s] < 0.99 for s in ['total_successes', 'variance_rates', 'max_rate']),
    'total_successes_ok': 0.05 < p_values['total_successes'] < 0.95,
    'variance_ok': 0.05 < p_values['variance_rates'] < 0.95,
    'can_generate_zeros': p_values['num_zeros'] > 0.01,
    'can_generate_extremes': p_values['max_rate'] > 0.01,
    'pareto_k_ok': np.sum(pareto_k >= 0.7) == 0,
    'pareto_k_borderline': np.sum(pareto_k >= 0.5) <= 2,
}

# Count passes
n_pass = sum(criteria.values())
n_total = len(criteria)

print(f"\nCriteria assessment ({n_pass}/{n_total} passed):")
for criterion, passed in criteria.items():
    status = "PASS" if passed else "FAIL"
    print(f"  - {criterion}: {status}")

# Overall decision
if n_pass == n_total:
    overall_decision = "PASS"
elif n_pass >= n_total - 2:
    overall_decision = "BORDERLINE"
else:
    overall_decision = "FAIL"

print(f"\n{'='*80}")
print(f"OVERALL DECISION: {overall_decision}")
print(f"{'='*80}")

# Save assessment
assessment = {
    'overall_decision': overall_decision,
    'n_criteria_passed': n_pass,
    'n_criteria_total': n_total,
    'criteria': criteria,
    'p_values': p_values,
    'pareto_k_summary': {
        'mean': float(pareto_k.mean()),
        'max': float(pareto_k.max()),
        'n_good': int(np.sum(pareto_k < 0.5)),
        'n_ok': int(np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))),
        'n_problematic': int(np.sum(pareto_k >= 0.7)),
    }
}

import json
with open('/workspace/experiments/experiment_1/posterior_predictive_check/results/assessment.json', 'w') as f:
    json.dump(assessment, f, indent=2)

print("\n[COMPLETE] Posterior predictive check completed successfully.")
print(f"  - Plots saved to: /workspace/experiments/experiment_1/posterior_predictive_check/plots/")
print(f"  - Results saved to: /workspace/experiments/experiment_1/posterior_predictive_check/results/")
