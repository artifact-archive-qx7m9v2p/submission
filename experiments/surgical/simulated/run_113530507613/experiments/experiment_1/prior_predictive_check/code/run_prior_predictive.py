"""
Prior Predictive Check for Experiment 1: Standard Hierarchical Logit-Normal Model

This script:
1. Loads the observed data
2. Samples from the prior predictive distribution
3. Generates comprehensive diagnostic visualizations
4. Computes quantitative checks
5. Saves results and diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Plotting configuration
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = '/workspace/data/data.csv'
STAN_MODEL_PATH = '/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive.stan'
PLOT_DIR = '/workspace/experiments/experiment_1/prior_predictive_check/plots/'

print("=" * 80)
print("PRIOR PREDICTIVE CHECK: Experiment 1")
print("=" * 80)

# Load observed data
print("\n[1/5] Loading observed data...")
data = pd.read_csv(DATA_PATH)
print(f"  - Loaded {len(data)} groups")
print(f"  - Success rates range: [{data['success_rate'].min():.3f}, {data['success_rate'].max():.3f}]")
print(f"  - Trial sizes range: [{data['n_trials'].min()}, {data['n_trials'].max()}]")

# Prepare data for Stan
stan_data = {
    'J': len(data),
    'n': data['n_trials'].tolist()
}

# Compile and run Stan model
print("\n[2/5] Sampling from prior predictive distribution...")
model = CmdStanModel(stan_file=STAN_MODEL_PATH)
print("  - Model compiled successfully")

# Generate prior predictive samples
# Use fixed_param algorithm since we're only using generated quantities
fit = model.sample(
    data=stan_data,
    fixed_param=True,
    iter_sampling=1000,
    chains=1,
    seed=42,
    show_progress=False
)
print("  - Generated 1000 prior predictive samples")

# Extract samples
print("\n[3/5] Extracting and processing samples...")
mu_samples = fit.stan_variable('mu')
tau_samples = fit.stan_variable('tau')
theta_samples = fit.stan_variable('theta')  # Shape: (1000, 12)
p_samples = fit.stan_variable('p')          # Shape: (1000, 12)
r_sim_samples = fit.stan_variable('r_sim')  # Shape: (1000, 12)

n_samples = len(mu_samples)
n_groups = stan_data['J']

print(f"  - mu: mean={mu_samples.mean():.3f}, sd={mu_samples.std():.3f}")
print(f"  - tau: mean={tau_samples.mean():.3f}, sd={tau_samples.std():.3f}")
print(f"  - p: mean={p_samples.mean():.3f}, sd={p_samples.std():.3f}")

# Quantitative diagnostics
print("\n[4/5] Computing quantitative diagnostics...")
print("\n  Prior Predictive Checks for Success Rates (p):")

# Check 1: All values in [0, 1]
pct_valid = 100 * np.mean((p_samples >= 0) & (p_samples <= 1))
print(f"    - % in [0, 1]: {pct_valid:.2f}% (Target: 100%)")

# Check 2: Appropriate for low success rate data
pct_low = 100 * np.mean(p_samples <= 0.5)
print(f"    - % in [0, 0.5]: {pct_low:.2f}% (Target: >50% for low-rate data)")

# Check 3: Not too many extreme values
pct_extreme = 100 * np.mean((p_samples < 0.01) | (p_samples > 0.99))
print(f"    - % extreme (<0.01 or >0.99): {pct_extreme:.2f}% (Target: <10%)")

# Check 4: Coverage of observed data
obs_min, obs_max = data['success_rate'].min(), data['success_rate'].max()
prior_min, prior_max = p_samples.min(), p_samples.max()
print(f"\n  Coverage Analysis:")
print(f"    - Observed range: [{obs_min:.3f}, {obs_max:.3f}]")
print(f"    - Prior predictive range: [{prior_min:.3f}, {prior_max:.3f}]")
print(f"    - Prior covers observed: {prior_min <= obs_min and prior_max >= obs_max}")

# Check 5: Quantiles
print(f"\n  Prior Predictive Quantiles for p:")
quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
for q in quantiles:
    val = np.quantile(p_samples, q)
    print(f"    - {int(q*100):2d}th percentile: {val:.4f}")

# Check 6: Domain knowledge violation check
obs_success_rates = data['success_rate'].values
print(f"\n  Domain Knowledge Checks:")
print(f"    - Mean prior predictive p: {p_samples.mean():.3f} vs observed mean: {obs_success_rates.mean():.3f}")
print(f"    - SD prior predictive p: {p_samples.std():.3f}")

# Visualization
print("\n[5/5] Creating visualizations...")

# ==================== PLOT 1: Parameter Plausibility ====================
print("  - Creating parameter plausibility plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# mu distribution
ax = axes[0, 0]
ax.hist(mu_samples, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
x = np.linspace(mu_samples.min(), mu_samples.max(), 200)
ax.plot(x, stats.norm.pdf(x, -2.6, 1.0), 'r-', linewidth=2, label='Prior: N(-2.6, 1.0)')
ax.axvline(mu_samples.mean(), color='darkblue', linestyle='--', linewidth=2, label=f'Sample mean: {mu_samples.mean():.2f}')
ax.set_xlabel('mu (population mean, logit scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Samples: Population Mean (mu)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# tau distribution
ax = axes[0, 1]
ax.hist(tau_samples, bins=50, density=True, alpha=0.6, color='darkorange', edgecolor='black')
x = np.linspace(0, tau_samples.max(), 200)
# Half-normal density
half_normal_density = 2 * stats.norm.pdf(x, 0, 0.5)
ax.plot(x, half_normal_density, 'r-', linewidth=2, label='Prior: Half-N(0, 0.5)')
ax.axvline(tau_samples.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Sample mean: {tau_samples.mean():.2f}')
ax.set_xlabel('tau (between-group SD, logit scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Samples: Between-Group Variability (tau)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Implied p distribution
ax = axes[1, 0]
ax.hist(p_samples.flatten(), bins=50, density=True, alpha=0.6, color='seagreen', edgecolor='black')
ax.axvline(p_samples.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {p_samples.mean():.3f}')
ax.axvline(obs_success_rates.mean(), color='red', linestyle=':', linewidth=2, label=f'Observed mean: {obs_success_rates.mean():.3f}')
ax.axvspan(obs_min, obs_max, alpha=0.2, color='red', label='Observed range')
ax.set_xlabel('p (success rate)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Implied Prior Distribution: Success Rates (p)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Joint mu-tau scatter
ax = axes[1, 1]
ax.scatter(mu_samples, tau_samples, alpha=0.3, s=20, color='purple')
ax.set_xlabel('mu (population mean, logit scale)', fontsize=11)
ax.set_ylabel('tau (between-group SD, logit scale)', fontsize=11)
ax.set_title('Prior Independence: mu vs tau', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}parameter_plausibility.png', bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOT_DIR}parameter_plausibility.png")

# ==================== PLOT 2: Prior Predictive Coverage ====================
print("  - Creating prior predictive coverage plot...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 2a: Success rates with credible intervals
ax = axes[0]
group_ids = np.arange(1, n_groups + 1)

# Calculate quantiles for each group
p_lower = np.percentile(p_samples, 2.5, axis=0)
p_median = np.percentile(p_samples, 50, axis=0)
p_upper = np.percentile(p_samples, 97.5, axis=0)

# Plot prior predictive intervals
ax.fill_between(group_ids, p_lower, p_upper, alpha=0.3, color='skyblue', label='Prior 95% CI')
ax.plot(group_ids, p_median, 'o-', color='steelblue', linewidth=2, markersize=6, label='Prior median')

# Overlay observed data
ax.plot(group_ids, obs_success_rates, 's', color='red', markersize=8, markeredgecolor='darkred',
        markeredgewidth=1.5, label='Observed', zorder=10)

ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Success Rate (p)', fontsize=11)
ax.set_title('Prior Predictive Coverage: Success Rates by Group', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks(group_ids)

# Plot 2b: Simulated vs observed success counts
ax = axes[1]

# Calculate quantiles for simulated counts
r_lower = np.percentile(r_sim_samples, 2.5, axis=0)
r_median = np.percentile(r_sim_samples, 50, axis=0)
r_upper = np.percentile(r_sim_samples, 97.5, axis=0)

# Plot prior predictive intervals
ax.fill_between(group_ids, r_lower, r_upper, alpha=0.3, color='lightcoral', label='Prior 95% CI')
ax.plot(group_ids, r_median, 'o-', color='firebrick', linewidth=2, markersize=6, label='Prior median')

# Overlay observed data
obs_counts = data['r_successes'].values
ax.plot(group_ids, obs_counts, 's', color='darkblue', markersize=8, markeredgecolor='black',
        markeredgewidth=1.5, label='Observed', zorder=10)

ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Success Count (r)', fontsize=11)
ax.set_title('Prior Predictive Coverage: Success Counts by Group', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks(group_ids)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}prior_predictive_coverage.png', bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOT_DIR}prior_predictive_coverage.png")

# ==================== PLOT 3: Distribution Diagnostics ====================
print("  - Creating distribution diagnostics plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 3a: Prior predictive p (all groups combined) - histogram
ax = axes[0, 0]
ax.hist(p_samples.flatten(), bins=60, density=True, alpha=0.6, color='teal', edgecolor='black')
ax.axvline(obs_success_rates.mean(), color='red', linestyle='--', linewidth=2, label='Observed mean')
ax.axvspan(obs_min, obs_max, alpha=0.15, color='red', label='Observed range')
ax.set_xlabel('Success Rate (p)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive: All Success Rates', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3b: Extreme value check
ax = axes[0, 1]
p_flat = p_samples.flatten()
bins_extreme = [0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0]
counts, edges = np.histogram(p_flat, bins=bins_extreme)
percentages = 100 * counts / len(p_flat)
x_labels = ['<0.01', '0.01-0.05', '0.05-0.1', '0.1-0.5', '0.5-0.9', '0.9-0.95', '0.95-0.99', '>0.99']
colors = ['red' if (i == 0 or i == len(x_labels)-1) else 'steelblue' for i in range(len(x_labels))]
ax.bar(range(len(percentages)), percentages, color=colors, edgecolor='black', alpha=0.7)
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_ylabel('Percentage (%)', fontsize=11)
ax.set_title('Extreme Value Distribution', fontsize=12, fontweight='bold')
ax.axhline(10, color='red', linestyle='--', linewidth=1.5, label='10% threshold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3c: QQ plot - check if implied prior is reasonable
ax = axes[1, 0]
stats.probplot(p_flat, dist='uniform', plot=ax)
ax.set_xlabel('Theoretical Quantiles (Uniform)', fontsize=11)
ax.set_ylabel('Sample Quantiles', fontsize=11)
ax.set_title('Q-Q Plot: Prior Predictive p vs Uniform(0,1)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3d: Coverage by sample size
ax = axes[1, 1]
n_trials = data['n_trials'].values
for i in range(n_groups):
    # Check if observed value falls within prior predictive 95% CI
    obs_in_ci = (obs_success_rates[i] >= p_lower[i]) and (obs_success_rates[i] <= p_upper[i])
    color = 'green' if obs_in_ci else 'red'
    marker = 'o' if obs_in_ci else 'x'
    ax.scatter(n_trials[i], obs_success_rates[i], color=color, marker=marker, s=100,
               edgecolors='black', linewidths=1.5, alpha=0.7, zorder=10)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
           markeredgecolor='black', label='Within prior 95% CI'),
    Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10,
           markeredgecolor='red', markeredgewidth=2, label='Outside prior 95% CI')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

ax.set_xlabel('Number of Trials (n)', fontsize=11)
ax.set_ylabel('Observed Success Rate', fontsize=11)
ax.set_title('Prior Coverage by Sample Size', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}distribution_diagnostics.png', bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOT_DIR}distribution_diagnostics.png")

# ==================== PLOT 4: Prior Predictive Draws ====================
print("  - Creating prior predictive draws visualization...")
fig, ax = plt.subplots(figsize=(14, 6))

# Plot a random subset of prior predictive draws
n_draws_to_plot = 100
draw_indices = np.random.choice(n_samples, size=n_draws_to_plot, replace=False)

for idx in draw_indices:
    ax.plot(group_ids, p_samples[idx, :], '-', color='skyblue', alpha=0.1, linewidth=1)

# Overlay observed data
ax.plot(group_ids, obs_success_rates, 's-', color='red', markersize=10,
        markeredgecolor='darkred', markeredgewidth=2, linewidth=3,
        label='Observed', zorder=10)

# Add prior median
ax.plot(group_ids, p_median, 'o-', color='darkblue', linewidth=2, markersize=6,
        label='Prior median', zorder=5, alpha=0.7)

ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Success Rate (p)', fontsize=11)
ax.set_title(f'Prior Predictive Draws (n={n_draws_to_plot}) vs Observed Data',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks(group_ids)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}prior_predictive_draws.png', bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOT_DIR}prior_predictive_draws.png")

# ==================== PLOT 5: Count-based Diagnostics ====================
print("  - Creating count-based diagnostics plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 5a: Simulated counts distribution
ax = axes[0, 0]
ax.hist(r_sim_samples.flatten(), bins=40, density=True, alpha=0.6, color='coral', edgecolor='black')
ax.axvline(obs_counts.mean(), color='darkred', linestyle='--', linewidth=2,
           label=f'Observed mean: {obs_counts.mean():.1f}')
ax.axvline(r_sim_samples.mean(), color='darkblue', linestyle='--', linewidth=2,
           label=f'Prior mean: {r_sim_samples.mean():.1f}')
ax.set_xlabel('Success Count (r)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive: All Success Counts', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5b: Observed vs predicted counts scatter
ax = axes[0, 1]
for i in range(n_groups):
    ax.scatter([obs_counts[i]], [r_median[i]], s=100, alpha=0.6, color='purple',
               edgecolors='black', linewidths=1)
    ax.plot([obs_counts[i], obs_counts[i]], [r_lower[i], r_upper[i]],
            color='gray', linewidth=2, alpha=0.5)

# Add diagonal line
max_val = max(obs_counts.max(), r_upper.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='y=x')
ax.set_xlabel('Observed Success Count', fontsize=11)
ax.set_ylabel('Prior Predictive Median Count', fontsize=11)
ax.set_title('Observed vs Prior Predicted Counts', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5c: Residuals (observed - predicted median)
ax = axes[1, 0]
residuals = obs_counts - r_median
ax.bar(group_ids, residuals, color=['red' if r < 0 else 'green' for r in residuals],
       alpha=0.6, edgecolor='black')
ax.axhline(0, color='black', linewidth=2)
ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Residual (Observed - Predicted)', fontsize=11)
ax.set_title('Residuals: Observed vs Prior Median', fontsize=12, fontweight='bold')
ax.set_xticks(group_ids)
ax.grid(alpha=0.3, axis='y')

# Plot 5d: Calibration - observed vs trial size
ax = axes[1, 1]
# Calculate success rate from simulated counts
r_sim_rates = r_sim_samples / n_trials.reshape(1, -1)
r_rates_lower = np.percentile(r_sim_rates, 2.5, axis=0)
r_rates_median = np.percentile(r_sim_rates, 50, axis=0)
r_rates_upper = np.percentile(r_sim_rates, 97.5, axis=0)

for i in range(n_groups):
    ax.scatter([n_trials[i]], [obs_success_rates[i]], s=100, alpha=0.7, color='red',
               edgecolors='darkred', linewidths=1.5, zorder=10)
    ax.plot([n_trials[i], n_trials[i]], [r_rates_lower[i], r_rates_upper[i]],
            color='steelblue', linewidth=3, alpha=0.5)

ax.set_xlabel('Number of Trials (n)', fontsize=11)
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('Prior Predictive 95% CI vs Observed by Trial Size', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}count_diagnostics.png', bbox_inches='tight')
plt.close()
print(f"    Saved: {PLOT_DIR}count_diagnostics.png")

# ==================== Summary Statistics ====================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Count how many observed values fall within prior predictive 95% CIs
n_covered = np.sum((obs_success_rates >= p_lower) & (obs_success_rates <= p_upper))
pct_covered = 100 * n_covered / n_groups

print(f"\nCoverage:")
print(f"  - Groups with observed rate in prior 95% CI: {n_covered}/{n_groups} ({pct_covered:.1f}%)")

# Check for computational issues
print(f"\nComputational Health:")
print(f"  - Any NaN in mu: {np.any(np.isnan(mu_samples))}")
print(f"  - Any NaN in tau: {np.any(np.isnan(tau_samples))}")
print(f"  - Any NaN in p: {np.any(np.isnan(p_samples))}")
print(f"  - Any inf in samples: {np.any(np.isinf(mu_samples)) or np.any(np.isinf(tau_samples)) or np.any(np.isinf(p_samples))}")

# Determine pass/fail
print("\n" + "=" * 80)
print("PASS/FAIL CRITERIA")
print("=" * 80)

checks_passed = []
checks_failed = []

# Check 1: All p values in [0, 1]
if pct_valid == 100.0:
    checks_passed.append("All success rates in [0, 1] (100%)")
else:
    checks_failed.append(f"Some success rates outside [0, 1] ({pct_valid:.2f}%)")

# Check 2: Not too many extreme values
if pct_extreme < 10.0:
    checks_passed.append(f"Extreme values < 10% ({pct_extreme:.2f}%)")
else:
    checks_failed.append(f"Too many extreme values ({pct_extreme:.2f}% >= 10%)")

# Check 3: Coverage of observed data range
if prior_min <= obs_min and prior_max >= obs_max:
    checks_passed.append(f"Prior covers observed range [{obs_min:.3f}, {obs_max:.3f}]")
else:
    checks_failed.append(f"Prior doesn't cover observed range [{obs_min:.3f}, {obs_max:.3f}]")

# Check 4: No computational issues
if not (np.any(np.isnan(p_samples)) or np.any(np.isinf(p_samples))):
    checks_passed.append("No computational issues (NaN/inf)")
else:
    checks_failed.append("Computational issues detected (NaN/inf values)")

# Check 5: Reasonable coverage percentage
if pct_covered >= 50:
    checks_passed.append(f"Good prior coverage ({pct_covered:.1f}% of groups in 95% CI)")
else:
    checks_failed.append(f"Poor prior coverage ({pct_covered:.1f}% < 50%)")

print("\nPASSED CHECKS:")
for check in checks_passed:
    print(f"  [PASS] {check}")

if checks_failed:
    print("\nFAILED CHECKS:")
    for check in checks_failed:
        print(f"  [FAIL] {check}")

# Final decision
if len(checks_failed) == 0:
    decision = "PASS"
    print(f"\n{'='*80}")
    print(f"FINAL DECISION: {decision}")
    print(f"{'='*80}")
    print("The prior predictive distribution is appropriate. Proceed to SBC.")
else:
    decision = "FAIL"
    print(f"\n{'='*80}")
    print(f"FINAL DECISION: {decision}")
    print(f"{'='*80}")
    print("The priors need adjustment before proceeding.")

print("\nPrior predictive check complete!")
print(f"Results saved to: /workspace/experiments/experiment_1/prior_predictive_check/")
