"""
Prior Predictive Check for Hierarchical Binomial Model (Logit-Normal)

This script validates that the prior distributions generate scientifically plausible
data before fitting the model to actual data.

Model: Hierarchical Binomial with Non-Centered Parameterization
Priors:
  - mu ~ Normal(-2.5, 1)
  - tau ~ Half-Cauchy(0, 1)
  - theta_raw_j ~ Normal(0, 1)
  - theta_j = mu + tau * theta_raw_j
  - r_j ~ Binomial(n_j, logit^-1(theta_j))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit  # logit^-1
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_SAMPLES = 1000
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Load observed data
data = pd.read_csv("/workspace/data/data.csv")
n_obs = data['n'].values
r_obs = data['r'].values
J = len(data)

# Observed success rates
p_obs = r_obs / n_obs
pooled_rate = r_obs.sum() / n_obs.sum()

# Observed overdispersion (proper calculation)
# Overdispersion = observed variance / expected variance under binomial
# Expected variance for binomial: Var(p) = p(1-p)/n
# For pooled estimate across groups, we use the sample variance of observed proportions
# compared to the expected variance under the binomial assumption
observed_var = np.var(p_obs, ddof=1)
expected_var_binomial = pooled_rate * (1 - pooled_rate) / np.mean(n_obs)
phi_obs = observed_var / expected_var_binomial

print("=" * 80)
print("PRIOR PREDICTIVE CHECK: Hierarchical Binomial (Logit-Normal)")
print("=" * 80)
print("\nOBSERVED DATA PROPERTIES:")
print(f"  J = {J} groups")
print(f"  Sample sizes: {n_obs.min()} to {n_obs.max()}")
print(f"  Success counts: {r_obs.min()} to {r_obs.max()}")
print(f"  Success rates: {p_obs.min():.3f} to {p_obs.max():.3f} ({p_obs.min()*100:.1f}% to {p_obs.max()*100:.1f}%)")
print(f"  Pooled rate: {pooled_rate:.4f} ({pooled_rate*100:.2f}%)")
print(f"  Observed variance of proportions: {observed_var:.6f}")
print(f"  Expected variance under binomial: {expected_var_binomial:.6f}")
print(f"  Overdispersion: phi = {phi_obs:.2f}")


# ============================================================================
# STEP 1: Sample from Prior Distributions
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Sampling from Prior Distributions")
print("=" * 80)

# Prior: mu ~ Normal(-2.5, 1)
mu_samples = np.random.normal(-2.5, 1, N_PRIOR_SAMPLES)

# Prior: tau ~ Half-Cauchy(0, 1)
# Half-Cauchy is |Cauchy(0, 1)|
tau_samples = np.abs(stats.cauchy.rvs(loc=0, scale=1, size=N_PRIOR_SAMPLES))

# Prior: theta_raw_j ~ Normal(0, 1) for each group
# Non-centered parameterization: theta_j = mu + tau * theta_raw_j
theta_raw_samples = np.random.normal(0, 1, (N_PRIOR_SAMPLES, J))

# Compute theta_j for each group
theta_samples = mu_samples[:, np.newaxis] + tau_samples[:, np.newaxis] * theta_raw_samples

# Convert to probability scale
p_samples = expit(theta_samples)  # logit^-1

print(f"\nPrior samples generated: {N_PRIOR_SAMPLES}")
print(f"\nPRIOR PARAMETER SUMMARIES:")
print(f"\nmu (population mean on logit scale):")
print(f"  Mean: {mu_samples.mean():.3f}, SD: {mu_samples.std():.3f}")
print(f"  Range: [{mu_samples.min():.3f}, {mu_samples.max():.3f}]")
print(f"  On probability scale: [{expit(mu_samples.min()):.3f}, {expit(mu_samples.max()):.3f}]")

print(f"\ntau (between-group SD on logit scale):")
print(f"  Mean: {tau_samples.mean():.3f}, SD: {tau_samples.std():.3f}")
print(f"  Range: [{tau_samples.min():.3f}, {tau_samples.max():.3f}]")
print(f"  Median: {np.median(tau_samples):.3f}, 95th percentile: {np.percentile(tau_samples, 95):.3f}")

print(f"\ntheta_j (group-level logits):")
print(f"  Mean: {theta_samples.mean():.3f}, SD: {theta_samples.std():.3f}")
print(f"  Range: [{theta_samples.min():.3f}, {theta_samples.max():.3f}]")

print(f"\np_j (group-level probabilities):")
print(f"  Mean: {p_samples.mean():.3f}, SD: {p_samples.std():.3f}")
print(f"  Range: [{p_samples.min():.6f}, {p_samples.max():.6f}]")
print(f"  Percentiles: 5%={np.percentile(p_samples, 5):.3f}, 50%={np.percentile(p_samples, 50):.3f}, 95%={np.percentile(p_samples, 95):.3f}")


# ============================================================================
# STEP 2: Generate Prior Predictive Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Generating Prior Predictive Data")
print("=" * 80)

# For each prior sample, simulate binomial data
r_prior_pred = np.zeros((N_PRIOR_SAMPLES, J))
for i in range(N_PRIOR_SAMPLES):
    for j in range(J):
        r_prior_pred[i, j] = np.random.binomial(n_obs[j], p_samples[i, j])

# Compute prior predictive success rates
p_prior_pred = r_prior_pred / n_obs

print(f"\nPrior predictive success rates:")
print(f"  Mean: {p_prior_pred.mean():.3f}, SD: {p_prior_pred.std():.3f}")
print(f"  Range: [{p_prior_pred.min():.6f}, {p_prior_pred.max():.6f}]")
print(f"  Percentiles: 5%={np.percentile(p_prior_pred, 5):.3f}, 50%={np.percentile(p_prior_pred, 50):.3f}, 95%={np.percentile(p_prior_pred, 95):.3f}")


# ============================================================================
# STEP 3: Compute Prior Predictive Overdispersion
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Computing Prior Predictive Overdispersion")
print("=" * 80)

phi_prior_pred = np.zeros(N_PRIOR_SAMPLES)
for i in range(N_PRIOR_SAMPLES):
    # Compute overdispersion for this prior predictive dataset
    observed_var_i = np.var(p_prior_pred[i, :], ddof=1)
    pooled_p = p_prior_pred[i, :].mean()
    expected_var_binomial_i = pooled_p * (1 - pooled_p) / np.mean(n_obs)

    if expected_var_binomial_i > 0:
        phi_prior_pred[i] = observed_var_i / expected_var_binomial_i
    else:
        phi_prior_pred[i] = np.nan

# Remove any NaN values
phi_prior_pred = phi_prior_pred[~np.isnan(phi_prior_pred)]

print(f"\nPrior predictive overdispersion (phi):")
print(f"  Mean: {phi_prior_pred.mean():.2f}, SD: {phi_prior_pred.std():.2f}")
print(f"  Range: [{phi_prior_pred.min():.2f}, {phi_prior_pred.max():.2f}]")
print(f"  Percentiles: 5%={np.percentile(phi_prior_pred, 5):.2f}, 50%={np.percentile(phi_prior_pred, 50):.2f}, 95%={np.percentile(phi_prior_pred, 95):.2f}")
print(f"\n  Observed phi: {phi_obs:.2f}")


# ============================================================================
# STEP 4: Diagnostic Checks
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Diagnostic Checks")
print("=" * 80)

# Check 1: Can prior generate observed range [3%, 14%]?
min_rates_prior = p_prior_pred.min(axis=1)
max_rates_prior = p_prior_pred.max(axis=1)

covers_min = (min_rates_prior <= p_obs.min()).sum() / N_PRIOR_SAMPLES
covers_max = (max_rates_prior >= p_obs.max()).sum() / N_PRIOR_SAMPLES
covers_range = ((min_rates_prior <= p_obs.min()) & (max_rates_prior >= p_obs.max())).sum() / N_PRIOR_SAMPLES

print(f"\nCHECK 1: Coverage of observed range [{p_obs.min():.3f}, {p_obs.max():.3f}]")
print(f"  Prior samples with min <= {p_obs.min():.3f}: {covers_min*100:.1f}%")
print(f"  Prior samples with max >= {p_obs.max():.3f}: {covers_max*100:.1f}%")
print(f"  Prior samples covering full range: {covers_range*100:.1f}%")

# Check 2: Can prior generate overdispersion phi >= 3?
phi_threshold = 3.0
covers_phi = (phi_prior_pred >= phi_threshold).sum() / len(phi_prior_pred)
print(f"\nCHECK 2: Overdispersion phi >= {phi_threshold}")
print(f"  Prior samples with phi >= {phi_threshold}: {covers_phi*100:.1f}%")

# Check 3: Do 95% prior predictive intervals cover observed rates?
p_prior_lower = np.percentile(p_prior_pred, 2.5, axis=0)
p_prior_upper = np.percentile(p_prior_pred, 97.5, axis=0)
coverage = ((p_obs >= p_prior_lower) & (p_obs <= p_prior_upper)).sum() / J
print(f"\nCHECK 3: Prior predictive 95% interval coverage")
print(f"  Groups covered: {coverage*100:.1f}% ({int(coverage*J)}/{J})")

# Check 4: Are there implausible extreme values?
extreme_high = (p_samples > 0.5).sum() / (N_PRIOR_SAMPLES * J)
extreme_very_high = (p_samples > 0.8).sum() / (N_PRIOR_SAMPLES * J)
print(f"\nCHECK 4: Extreme values (prior should NOT generate high success rates)")
print(f"  Prior samples with p > 0.5: {extreme_high*100:.2f}%")
print(f"  Prior samples with p > 0.8: {extreme_very_high*100:.2f}%")

# Check 5: Computational flags
extreme_tau = (tau_samples > 10).sum() / N_PRIOR_SAMPLES
extreme_theta = (np.abs(theta_samples) > 10).sum() / (N_PRIOR_SAMPLES * J)
print(f"\nCHECK 5: Computational warnings")
print(f"  tau > 10: {extreme_tau*100:.2f}%")
print(f"  |theta_j| > 10: {extreme_theta*100:.2f}%")


# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Creating Visualizations")
print("=" * 80)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# -------------------------------------------------------------------------
# PLOT 1: Prior Parameter Distributions
# -------------------------------------------------------------------------
print("\nCreating plot 1: parameter_plausibility.png")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# mu
ax = axes[0, 0]
ax.hist(mu_samples, bins=50, alpha=0.7, edgecolor='black', density=True)
ax.axvline(mu_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mu_samples.mean():.2f}')
ax.set_xlabel('mu (population mean on logit scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: mu ~ Normal(-2.5, 1)', fontsize=12, fontweight='bold')
ax.legend()

# mu on probability scale
ax = axes[0, 1]
mu_prob_samples = expit(mu_samples)
ax.hist(mu_prob_samples, bins=50, alpha=0.7, edgecolor='black', density=True, color='green')
ax.axvline(mu_prob_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mu_prob_samples.mean():.3f}')
ax.axvline(pooled_rate, color='blue', linestyle='--', linewidth=2, label=f'Observed pooled: {pooled_rate:.3f}')
ax.set_xlabel('expit(mu) (probability scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: Population Mean on Probability Scale', fontsize=12, fontweight='bold')
ax.legend()

# tau
ax = axes[1, 0]
ax.hist(tau_samples, bins=50, alpha=0.7, edgecolor='black', density=True, color='orange')
ax.axvline(tau_samples.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tau_samples.mean():.2f}')
ax.axvline(np.median(tau_samples), color='purple', linestyle='--', linewidth=2, label=f'Median: {np.median(tau_samples):.2f}')
ax.set_xlabel('tau (between-group SD on logit scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: tau ~ Half-Cauchy(0, 1)', fontsize=12, fontweight='bold')
ax.set_xlim(0, np.percentile(tau_samples, 99))
ax.legend()

# theta_j distribution (all groups combined)
ax = axes[1, 1]
theta_flat = theta_samples.flatten()
ax.hist(theta_flat, bins=60, alpha=0.7, edgecolor='black', density=True, color='purple')
ax.axvline(theta_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {theta_flat.mean():.2f}')
ax.set_xlabel('theta_j (group-level logits)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: theta_j = mu + tau * theta_raw_j', fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/parameter_plausibility.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 2: Prior Predictive Success Rates vs Observed
# -------------------------------------------------------------------------
print("Creating plot 2: prior_predictive_coverage.png")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot histogram of all prior predictive rates
p_flat = p_prior_pred.flatten()
ax.hist(p_flat, bins=100, alpha=0.6, edgecolor='black', density=True,
        label=f'Prior Predictive (n={len(p_flat)})', color='skyblue')

# Overlay observed rates
for i, p in enumerate(p_obs):
    ax.axvline(p, color='red', linestyle='-', linewidth=2, alpha=0.7)
    if i == 0:
        ax.axvline(p, color='red', linestyle='-', linewidth=2, alpha=0.7,
                  label=f'Observed Rates (n={J})')

# Mark key percentiles
ax.axvline(np.percentile(p_flat, 2.5), color='blue', linestyle='--', linewidth=2,
          label=f'Prior 95% CI: [{np.percentile(p_flat, 2.5):.3f}, {np.percentile(p_flat, 97.5):.3f}]')
ax.axvline(np.percentile(p_flat, 97.5), color='blue', linestyle='--', linewidth=2)

# Mark observed range
ax.axvspan(p_obs.min(), p_obs.max(), alpha=0.2, color='yellow',
          label=f'Observed Range: [{p_obs.min():.3f}, {p_obs.max():.3f}]')

ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Predictive Success Rates vs Observed Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-0.02, 0.5)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 3: Overdispersion Diagnostic
# -------------------------------------------------------------------------
print("Creating plot 3: overdispersion_diagnostic.png")

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(phi_prior_pred, bins=60, alpha=0.7, edgecolor='black', density=True, color='teal')
ax.axvline(phi_obs, color='red', linestyle='-', linewidth=3,
          label=f'Observed: phi = {phi_obs:.2f}')
ax.axvline(np.median(phi_prior_pred), color='blue', linestyle='--', linewidth=2,
          label=f'Prior Pred Median: {np.median(phi_prior_pred):.2f}')
ax.axvline(1, color='gray', linestyle=':', linewidth=2,
          label='No overdispersion (phi = 1)')

# Shade region where phi >= 3
if phi_prior_pred.max() >= 3:
    ax.axvspan(3, phi_prior_pred.max(), alpha=0.2, color='green',
              label=f'phi >= 3: {covers_phi*100:.1f}%')
else:
    ax.text(0.5, 0.5, f'No samples with phi >= 3\n(max phi = {phi_prior_pred.max():.2f})',
           transform=ax.transAxes, ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_xlabel('Overdispersion (phi)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Predictive Overdispersion vs Observed', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, min(20, max(10, phi_prior_pred.max() * 1.1)))

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/overdispersion_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 4: Caterpillar Plot - Group-Level Comparison
# -------------------------------------------------------------------------
print("Creating plot 4: group_level_comparison.png")

fig, ax = plt.subplots(figsize=(12, 8))

# For each group, compute prior predictive intervals
for j in range(J):
    p_prior_j = p_prior_pred[:, j]
    median_j = np.median(p_prior_j)
    lower_j = np.percentile(p_prior_j, 2.5)
    upper_j = np.percentile(p_prior_j, 97.5)

    # Plot prior predictive interval
    ax.plot([lower_j, upper_j], [j, j], 'b-', linewidth=2, alpha=0.6)
    ax.plot(median_j, j, 'bo', markersize=8, alpha=0.6, label='Prior Pred 95% CI' if j == 0 else '')

    # Plot observed rate
    ax.plot(p_obs[j], j, 'ro', markersize=10, label='Observed' if j == 0 else '')

ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Group-Level Prior Predictive Intervals vs Observed Rates', fontsize=14, fontweight='bold')
ax.set_yticks(range(J))
ax.set_yticklabels([f'Group {i+1}' for i in range(J)])
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/group_level_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 5: Range Coverage Diagnostic
# -------------------------------------------------------------------------
print("Creating plot 5: range_coverage_diagnostic.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Distribution of min/max rates
ax = axes[0]
ax.hist(min_rates_prior, bins=50, alpha=0.6, label='Min rate per simulation', color='blue', edgecolor='black')
ax.hist(max_rates_prior, bins=50, alpha=0.6, label='Max rate per simulation', color='green', edgecolor='black')
ax.axvline(p_obs.min(), color='blue', linestyle='--', linewidth=3, label=f'Observed min: {p_obs.min():.3f}')
ax.axvline(p_obs.max(), color='green', linestyle='--', linewidth=3, label=f'Observed max: {p_obs.max():.3f}')
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive Range Coverage', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

# Right: 2D density of (min, max)
ax = axes[1]
h = ax.hist2d(min_rates_prior, max_rates_prior, bins=50, cmap='Blues', cmin=1)
ax.plot(p_obs.min(), p_obs.max(), 'r*', markersize=20,
        label=f'Observed: [{p_obs.min():.3f}, {p_obs.max():.3f}]')
ax.set_xlabel('Minimum Success Rate', fontsize=12)
ax.set_ylabel('Maximum Success Rate', fontsize=12)
ax.set_title('Joint Distribution of Min/Max Rates', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.colorbar(h[3], ax=ax, label='Frequency')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/range_coverage_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 6: Extreme Values Check
# -------------------------------------------------------------------------
print("Creating plot 6: extreme_values_check.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Distribution of tau (with extreme values marked)
ax = axes[0]
ax.hist(tau_samples, bins=50, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(np.percentile(tau_samples, 95), color='red', linestyle='--', linewidth=2,
          label=f'95th percentile: {np.percentile(tau_samples, 95):.2f}')
ax.axvline(10, color='darkred', linestyle=':', linewidth=2,
          label=f'tau > 10: {extreme_tau*100:.2f}%')
ax.set_xlabel('tau (between-group SD)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior for tau: Check for Extreme Values', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, min(15, tau_samples.max()))

# Right: Probability scale extremes
ax = axes[1]
p_flat = p_samples.flatten()
ax.hist(p_flat, bins=100, alpha=0.7, edgecolor='black', color='purple')
ax.axvline(0.5, color='red', linestyle='--', linewidth=2,
          label=f'p > 0.5: {extreme_high*100:.2f}%')
ax.axvline(0.8, color='darkred', linestyle=':', linewidth=2,
          label=f'p > 0.8: {extreme_very_high*100:.2f}%')
ax.axvspan(p_obs.min(), p_obs.max(), alpha=0.2, color='yellow',
          label='Observed range')
ax.set_xlabel('Success Probability', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive Probabilities: Implausible Values Check', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/extreme_values_check.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to {PLOTS_DIR}/")


# ============================================================================
# STEP 6: Pass/Fail Decision
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: PASS/FAIL DECISION")
print("=" * 80)

# Criteria for PASS
criterion_1_pass = covers_range >= 0.50  # At least 50% cover observed range
criterion_2_pass = covers_phi >= 0.25    # At least 25% allow phi > 3
criterion_3_pass = coverage >= 0.70      # At least 70% of groups covered by 95% CI
criterion_4_pass = extreme_very_high <= 0.05  # Less than 5% with p > 0.8 (not too weak)

print("\nEVALUATING CRITERIA:")
print(f"\n1. Range Coverage [{p_obs.min():.3f}, {p_obs.max():.3f}]:")
print(f"   - Required: >= 50% of simulations cover full range")
print(f"   - Observed: {covers_range*100:.1f}% ({covers_range*N_PRIOR_SAMPLES:.0f}/{N_PRIOR_SAMPLES})")
print(f"   - Result: {'PASS' if criterion_1_pass else 'FAIL'}")

print(f"\n2. Overdispersion (phi >= 3):")
print(f"   - Required: >= 25% of simulations have phi >= 3")
print(f"   - Observed: {covers_phi*100:.1f}% ({covers_phi*len(phi_prior_pred):.0f}/{len(phi_prior_pred)})")
print(f"   - Result: {'PASS' if criterion_2_pass else 'FAIL'}")

print(f"\n3. Prior Predictive Interval Coverage:")
print(f"   - Required: >= 70% of groups covered by 95% CI")
print(f"   - Observed: {coverage*100:.1f}% ({int(coverage*J)}/{J} groups)")
print(f"   - Result: {'PASS' if criterion_3_pass else 'FAIL'}")

print(f"\n4. Implausible Values (p > 0.8):")
print(f"   - Required: <= 5% of prior samples")
print(f"   - Observed: {extreme_very_high*100:.2f}%")
print(f"   - Result: {'PASS' if criterion_4_pass else 'FAIL'}")

print("\n" + "-" * 80)

overall_pass = criterion_1_pass and criterion_2_pass and criterion_3_pass and criterion_4_pass

if overall_pass:
    print("\nFINAL VERDICT: PASS")
    print("\nThe priors generate scientifically plausible data that covers the observed")
    print("range and overdispersion without allowing implausible values. The model is")
    print("ready for fitting to actual data.")
else:
    print("\nFINAL VERDICT: FAIL")
    print("\nThe priors do not satisfy all criteria. Review the diagnostic plots and")
    print("consider adjusting the prior distributions:")

    if not criterion_1_pass:
        print("  - Range coverage is insufficient. Consider wider priors or different centering.")
    if not criterion_2_pass:
        print("  - Overdispersion coverage is insufficient. Consider increasing tau prior scale.")
    if not criterion_3_pass:
        print("  - Prior predictive intervals too narrow. Consider less informative priors.")
    if not criterion_4_pass:
        print("  - Too many implausible high values. Consider more informative priors on mu.")

print("\n" + "=" * 80)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("=" * 80)

# Save summary statistics for findings report
summary = {
    'n_prior_samples': int(N_PRIOR_SAMPLES),
    'mu_mean': float(mu_samples.mean()),
    'mu_sd': float(mu_samples.std()),
    'tau_mean': float(tau_samples.mean()),
    'tau_median': float(np.median(tau_samples)),
    'tau_95pct': float(np.percentile(tau_samples, 95)),
    'p_prior_min': float(p_samples.min()),
    'p_prior_max': float(p_samples.max()),
    'p_prior_mean': float(p_samples.mean()),
    'phi_obs': float(phi_obs),
    'phi_prior_mean': float(phi_prior_pred.mean()),
    'phi_prior_median': float(np.median(phi_prior_pred)),
    'covers_range_pct': float(covers_range * 100),
    'covers_phi_pct': float(covers_phi * 100),
    'group_coverage_pct': float(coverage * 100),
    'extreme_high_pct': float(extreme_high * 100),
    'extreme_very_high_pct': float(extreme_very_high * 100),
    'criterion_1_pass': bool(criterion_1_pass),
    'criterion_2_pass': bool(criterion_2_pass),
    'criterion_3_pass': bool(criterion_3_pass),
    'criterion_4_pass': bool(criterion_4_pass),
    'overall_pass': bool(overall_pass)
}

# Save to file
import json
with open(f'{OUTPUT_DIR}/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary statistics saved to {OUTPUT_DIR}/summary_statistics.json")
