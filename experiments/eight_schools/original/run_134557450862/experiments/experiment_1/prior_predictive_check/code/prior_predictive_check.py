"""
Prior Predictive Check for Experiment 1: Standard Non-Centered Hierarchical Model

This script validates that priors generate scientifically plausible data before model fitting.

Model:
    y_i ~ Normal(theta_i, sigma_i)     [sigma_i known]
    theta_i = mu + tau * eta_i
    eta_i ~ Normal(0, 1)
    mu ~ Normal(0, 20)
    tau ~ Half-Cauchy(0, 5)

Validation Criteria:
    - Generated data should span roughly [-50, 50]
    - Should NOT routinely generate y_i > 100 (too vague)
    - Should NOT confine y_i to [-5, 5] (too informative)
    - Observed data should not be extreme under prior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_SAMPLES = 500
N_SCHOOLS = 8
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check"

# Load observed data
data = pd.read_csv("/workspace/data/data.csv")
y_obs = data['y'].values
sigma_obs = data['sigma'].values
schools = data['school'].values

print("="*80)
print("PRIOR PREDICTIVE CHECK: Experiment 1")
print("="*80)
print("\nObserved Data Summary:")
print(f"  Schools: {N_SCHOOLS}")
print(f"  y_obs range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
print(f"  y_obs mean: {y_obs.mean():.1f}")
print(f"  y_obs std: {y_obs.std():.1f}")
print(f"  sigma range: [{sigma_obs.min():.1f}, {sigma_obs.max():.1f}]")

# ============================================================================
# STEP 1: Sample from Prior Distributions
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Sampling from Prior Distributions")
print("="*80)

# Prior: mu ~ Normal(0, 20)
mu_prior_samples = np.random.normal(0, 20, N_PRIOR_SAMPLES)

# Prior: tau ~ Half-Cauchy(0, 5)
# Half-Cauchy is absolute value of Cauchy
tau_prior_samples = np.abs(stats.cauchy.rvs(loc=0, scale=5, size=N_PRIOR_SAMPLES))

print(f"\nPrior Samples Statistics (n={N_PRIOR_SAMPLES}):")
print(f"\n  mu ~ Normal(0, 20):")
print(f"    Range: [{mu_prior_samples.min():.1f}, {mu_prior_samples.max():.1f}]")
print(f"    Mean: {mu_prior_samples.mean():.1f}")
print(f"    Std: {mu_prior_samples.std():.1f}")
print(f"    Quantiles (5%, 50%, 95%): [{np.percentile(mu_prior_samples, 5):.1f}, "
      f"{np.percentile(mu_prior_samples, 50):.1f}, {np.percentile(mu_prior_samples, 95):.1f}]")

print(f"\n  tau ~ Half-Cauchy(0, 5):")
print(f"    Range: [{tau_prior_samples.min():.1f}, {tau_prior_samples.max():.1f}]")
print(f"    Mean: {tau_prior_samples.mean():.1f}")
print(f"    Std: {tau_prior_samples.std():.1f}")
print(f"    Quantiles (5%, 50%, 95%): [{np.percentile(tau_prior_samples, 5):.1f}, "
      f"{np.percentile(tau_prior_samples, 50):.1f}, {np.percentile(tau_prior_samples, 95):.1f}]")

# Check for extreme tau values (computational red flag)
extreme_tau = np.sum(tau_prior_samples > 100)
print(f"    WARNING: {extreme_tau} samples ({100*extreme_tau/N_PRIOR_SAMPLES:.1f}%) have tau > 100")

# ============================================================================
# STEP 2: Generate Prior Predictive Data
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Generating Prior Predictive Data")
print("="*80)

# For each prior sample, generate a complete dataset
y_prior_pred = np.zeros((N_PRIOR_SAMPLES, N_SCHOOLS))
theta_prior = np.zeros((N_PRIOR_SAMPLES, N_SCHOOLS))

for i in range(N_PRIOR_SAMPLES):
    # Sample eta_i ~ Normal(0, 1) for each school
    eta = np.random.normal(0, 1, N_SCHOOLS)

    # Compute theta_i = mu + tau * eta_i (non-centered parameterization)
    theta_prior[i] = mu_prior_samples[i] + tau_prior_samples[i] * eta

    # Generate y_i ~ Normal(theta_i, sigma_i)
    y_prior_pred[i] = np.random.normal(theta_prior[i], sigma_obs)

print(f"\nPrior Predictive Statistics:")
print(f"  y_sim range: [{y_prior_pred.min():.1f}, {y_prior_pred.max():.1f}]")
print(f"  y_sim mean: {y_prior_pred.mean():.1f}")
print(f"  y_sim std: {y_prior_pred.std():.1f}")

# ============================================================================
# STEP 3: Domain Validation Checks
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Domain Validation Checks")
print("="*80)

# Check 1: Plausible range [-50, 50]
in_plausible_range = np.sum((y_prior_pred >= -50) & (y_prior_pred <= 50))
total_values = y_prior_pred.size
pct_plausible = 100 * in_plausible_range / total_values

print(f"\nCheck 1: Plausible Range [-50, 50]")
print(f"  {pct_plausible:.1f}% of simulated values in plausible range")
print(f"  EXPECTATION: Should be >70% but not >99% (weakly informative)")

# Check 2: Not too vague (absurd values)
extreme_values = np.sum(np.abs(y_prior_pred) > 100)
pct_extreme = 100 * extreme_values / total_values

print(f"\nCheck 2: Extreme Values (|y| > 100)")
print(f"  {pct_extreme:.1f}% of simulated values are extreme")
print(f"  EXPECTATION: Should be <10% (priors not too vague)")

# Check 3: Not too informative (too narrow)
very_narrow = np.sum(np.abs(y_prior_pred) < 5)
pct_narrow = 100 * very_narrow / total_values

print(f"\nCheck 3: Very Narrow Range (|y| < 5)")
print(f"  {pct_narrow:.1f}% of simulated values in narrow range")
print(f"  EXPECTATION: Should be <80% (priors not too informative)")

# Check 4: Observed data plausibility
print(f"\nCheck 4: Observed Data Plausibility")
for school_idx in range(N_SCHOOLS):
    y_school_sim = y_prior_pred[:, school_idx]
    percentile = stats.percentileofscore(y_school_sim, y_obs[school_idx])

    extreme_flag = ""
    if percentile < 2.5 or percentile > 97.5:
        extreme_flag = " [EXTREME]"

    print(f"  School {schools[school_idx]}: y_obs={y_obs[school_idx]:6.1f}, "
          f"percentile={percentile:5.1f}%{extreme_flag}")

print(f"  EXPECTATION: Observed values should not be extreme (<2.5% or >97.5%)")

# Check 5: Computational stability
print(f"\nCheck 5: Computational Stability")
n_extreme_theta = np.sum(np.abs(theta_prior) > 1000)
pct_extreme_theta = 100 * n_extreme_theta / theta_prior.size
print(f"  {pct_extreme_theta:.2f}% of theta values have |theta| > 1000")
print(f"  EXPECTATION: Should be near 0% (numerical stability)")

# ============================================================================
# STEP 4: Statistical Summaries
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Statistical Summaries")
print("="*80)

print("\nPrior Predictive Quantiles (across all schools):")
quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
y_flat = y_prior_pred.flatten()
for q in quantiles:
    print(f"  {q:2d}%: {np.percentile(y_flat, q):7.1f}")

print("\nPrior Predictive by School:")
for school_idx in range(N_SCHOOLS):
    y_school = y_prior_pred[:, school_idx]
    print(f"  School {schools[school_idx]}: "
          f"mean={y_school.mean():6.1f}, "
          f"std={y_school.std():5.1f}, "
          f"range=[{y_school.min():6.1f}, {y_school.max():6.1f}]")

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Creating Visualizations")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ------------------------------------------------------------------------
# Plot 1: Parameter Plausibility
# ------------------------------------------------------------------------
print("\n  Creating parameter_plausibility.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: mu prior samples
ax = axes[0, 0]
ax.hist(mu_prior_samples, bins=50, alpha=0.7, edgecolor='black', density=True)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Prior mean')
ax.axvline(y_obs.mean(), color='green', linestyle='--', linewidth=2, label='Observed mean')
ax.set_xlabel('mu (population mean)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: mu ~ Normal(0, 20)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: tau prior samples
ax = axes[0, 1]
# Clip tau for visualization
tau_clipped = np.clip(tau_prior_samples, 0, 50)
ax.hist(tau_clipped, bins=50, alpha=0.7, edgecolor='black', density=True)
ax.axvline(np.median(tau_prior_samples), color='red', linestyle='--',
           linewidth=2, label=f'Median: {np.median(tau_prior_samples):.1f}')
ax.set_xlabel('tau (between-school SD)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: tau ~ Half-Cauchy(0, 5) [clipped at 50 for viz]',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)

# Subplot 3: Joint mu-tau samples
ax = axes[1, 0]
tau_for_scatter = np.clip(tau_prior_samples, 0, 50)
scatter = ax.scatter(mu_prior_samples, tau_for_scatter, alpha=0.3, s=20)
ax.set_xlabel('mu (population mean)', fontsize=11)
ax.set_ylabel('tau (between-school SD)', fontsize=11)
ax.set_title('Joint Prior: mu vs tau', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 50)

# Subplot 4: theta prior samples (for each school)
ax = axes[1, 1]
theta_clipped = np.clip(theta_prior, -100, 100)
for school_idx in range(N_SCHOOLS):
    ax.hist(theta_clipped[:, school_idx], bins=30, alpha=0.4,
            label=f'School {schools[school_idx]}', density=True)
ax.set_xlabel('theta (school-specific mean)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: theta_i = mu + tau * eta_i [clipped at ±100]',
             fontsize=12, fontweight='bold')
ax.legend(ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-100, 100)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/parameter_plausibility.png", dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------------
# Plot 2: Prior Predictive Coverage
# ------------------------------------------------------------------------
print("  Creating prior_predictive_coverage.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Prior predictive distribution (all schools combined)
ax = axes[0, 0]
y_flat = y_prior_pred.flatten()
ax.hist(y_flat, bins=100, alpha=0.7, edgecolor='black', density=True,
        range=(-100, 100))
# Add vertical lines for observed data
for y_val in y_obs:
    ax.axvline(y_val, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
ax.axvline(y_obs[0], color='red', linestyle='--', alpha=0.6, linewidth=1.5,
           label='Observed values')
ax.set_xlabel('y (test score)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive Distribution (All Schools)',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-100, 100)

# Subplot 2: Prior predictive by school (overlaid)
ax = axes[0, 1]
colors = plt.cm.tab10(np.linspace(0, 1, N_SCHOOLS))
for school_idx in range(N_SCHOOLS):
    ax.hist(y_prior_pred[:, school_idx], bins=50, alpha=0.3,
            label=f'School {schools[school_idx]}', density=True,
            range=(-100, 100), color=colors[school_idx])
    ax.axvline(y_obs[school_idx], color=colors[school_idx],
               linestyle='--', linewidth=2, alpha=0.8)
ax.set_xlabel('y (test score)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive by School (dashed = observed)',
             fontsize=12, fontweight='bold')
ax.legend(ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-100, 100)

# Subplot 3: Quantile ranges
ax = axes[1, 0]
quantiles_to_plot = [5, 25, 50, 75, 95]
school_positions = np.arange(N_SCHOOLS)

for q in quantiles_to_plot:
    q_values = np.percentile(y_prior_pred, q, axis=0)
    ax.plot(school_positions, q_values, marker='o', label=f'{q}%', linewidth=2)

ax.plot(school_positions, y_obs, 'ro', markersize=10, linewidth=3,
        label='Observed', zorder=10)
ax.set_xlabel('School', fontsize=11)
ax.set_ylabel('y (test score)', fontsize=11)
ax.set_title('Prior Predictive Quantiles by School',
             fontsize=12, fontweight='bold')
ax.set_xticks(school_positions)
ax.set_xticklabels(schools)
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: Prior predictive check - percentile of observed
ax = axes[1, 1]
percentiles = []
for school_idx in range(N_SCHOOLS):
    pct = stats.percentileofscore(y_prior_pred[:, school_idx], y_obs[school_idx])
    percentiles.append(pct)

bars = ax.bar(school_positions, percentiles, alpha=0.7, edgecolor='black')
# Color bars based on extremeness
for i, (bar, pct) in enumerate(zip(bars, percentiles)):
    if pct < 2.5 or pct > 97.5:
        bar.set_color('red')
    elif pct < 10 or pct > 90:
        bar.set_color('orange')
    else:
        bar.set_color('green')

ax.axhline(50, color='blue', linestyle='--', linewidth=2, label='Median')
ax.axhline(2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='2.5%/97.5%')
ax.axhline(97.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_xlabel('School', fontsize=11)
ax.set_ylabel('Percentile of Observed Value', fontsize=11)
ax.set_title('Where Observed Data Falls in Prior Predictive',
             fontsize=12, fontweight='bold')
ax.set_xticks(school_positions)
ax.set_xticklabels(schools)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/prior_predictive_coverage.png", dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------------
# Plot 3: Range Validation Diagnostic
# ------------------------------------------------------------------------
print("  Creating range_validation.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Distribution of simulated values
ax = axes[0, 0]
y_flat = y_prior_pred.flatten()
ax.hist(y_flat, bins=100, alpha=0.7, edgecolor='black', color='skyblue')
ax.axvline(-50, color='green', linestyle='--', linewidth=2, label='Plausible range [-50, 50]')
ax.axvline(50, color='green', linestyle='--', linewidth=2)
ax.axvline(-100, color='red', linestyle='--', linewidth=2, label='Extreme threshold [±100]')
ax.axvline(100, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('y (test score)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution of All Prior Predictive Samples',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Proportion in different ranges
ax = axes[1, 0]
ranges = [
    ('[-5, 5]', -5, 5),
    ('[-10, 10]', -10, 10),
    ('[-20, 20]', -20, 20),
    ('[-50, 50]', -50, 50),
    ('[-100, 100]', -100, 100)
]
proportions = []
range_labels = []
for label, low, high in ranges:
    prop = 100 * np.sum((y_flat >= low) & (y_flat <= high)) / len(y_flat)
    proportions.append(prop)
    range_labels.append(label)

bars = ax.barh(range_labels, proportions, alpha=0.7, edgecolor='black')
# Color by informativeness
bars[0].set_color('red')  # Too informative
bars[1].set_color('orange')
bars[2].set_color('yellow')
bars[3].set_color('lightgreen')
bars[4].set_color('green')

ax.set_xlabel('Percentage of Samples in Range', fontsize=11)
ax.set_title('Coverage by Range (Green = Good)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(proportions):
    ax.text(v + 1, i, f'{v:.1f}%', va='center')

# Subplot 3: Max absolute value per simulated dataset
ax = axes[0, 1]
max_abs_per_dataset = np.max(np.abs(y_prior_pred), axis=1)
ax.hist(max_abs_per_dataset, bins=50, alpha=0.7, edgecolor='black', color='coral')
ax.axvline(50, color='green', linestyle='--', linewidth=2,
           label='Target threshold (50)')
ax.axvline(100, color='red', linestyle='--', linewidth=2,
           label='Extreme threshold (100)')
ax.set_xlabel('Max |y| in simulated dataset', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Maximum Absolute Value per Simulated Dataset',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: Standard deviation per simulated dataset
ax = axes[1, 1]
std_per_dataset = np.std(y_prior_pred, axis=1)
ax.hist(std_per_dataset, bins=50, alpha=0.7, edgecolor='black', color='plum')
ax.axvline(y_obs.std(), color='red', linestyle='--', linewidth=2,
           label=f'Observed SD ({y_obs.std():.1f})')
ax.set_xlabel('Standard deviation of y values', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Variability per Simulated Dataset', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/range_validation.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 6: Final Assessment
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Final Assessment")
print("="*80)

# Determine PASS/FAIL
pass_checks = []
fail_checks = []

# Criterion 1: Plausible range
if pct_plausible >= 70 and pct_plausible <= 99:
    pass_checks.append("Plausible range: Coverage appropriate")
else:
    fail_checks.append(f"Plausible range: {pct_plausible:.1f}% in [-50,50] (target 70-99%)")

# Criterion 2: Not too vague
if pct_extreme < 10:
    pass_checks.append("Extreme values: Priors not too vague")
else:
    fail_checks.append(f"Extreme values: {pct_extreme:.1f}% exceed |100| (target <10%)")

# Criterion 3: Not too informative
if pct_narrow < 80:
    pass_checks.append("Narrow range: Priors not too informative")
else:
    fail_checks.append(f"Narrow range: {pct_narrow:.1f}% in |y|<5 (target <80%)")

# Criterion 4: Observed data not extreme
extreme_obs = sum(1 for p in percentiles if p < 2.5 or p > 97.5)
if extreme_obs == 0:
    pass_checks.append("Observed data: All values within reasonable prior range")
elif extreme_obs <= 1:
    pass_checks.append(f"Observed data: Only {extreme_obs} value slightly extreme (acceptable)")
else:
    fail_checks.append(f"Observed data: {extreme_obs} values extreme under prior")

# Criterion 5: Computational stability
if pct_extreme_theta < 1.0:
    pass_checks.append("Computational stability: No numerical concerns")
else:
    fail_checks.append(f"Computational stability: {pct_extreme_theta:.1f}% extreme theta values")

print("\nPASS Criteria Met:")
for check in pass_checks:
    print(f"  ✓ {check}")

if fail_checks:
    print("\nFAIL Criteria:")
    for check in fail_checks:
        print(f"  ✗ {check}")

# Overall decision
if len(fail_checks) == 0:
    decision = "PASS"
    print("\n" + "="*80)
    print("FINAL DECISION: PASS")
    print("="*80)
    print("Priors are weakly informative and generate scientifically plausible data.")
    print("The model is ready for fitting.")
else:
    decision = "FAIL"
    print("\n" + "="*80)
    print("FINAL DECISION: FAIL")
    print("="*80)
    print("Priors need adjustment. See recommendations in findings.md")

# Save results
results = {
    'decision': decision,
    'pct_plausible': pct_plausible,
    'pct_extreme': pct_extreme,
    'pct_narrow': pct_narrow,
    'extreme_obs': extreme_obs,
    'pct_extreme_theta': pct_extreme_theta,
    'pass_checks': pass_checks,
    'fail_checks': fail_checks
}

np.savez(f"{OUTPUT_DIR}/code/prior_check_results.npz",
         mu_prior=mu_prior_samples,
         tau_prior=tau_prior_samples,
         theta_prior=theta_prior,
         y_prior_pred=y_prior_pred,
         y_obs=y_obs,
         sigma_obs=sigma_obs,
         percentiles=percentiles,
         results=results)

print("\nAll results saved to:")
print(f"  Code: {OUTPUT_DIR}/code/")
print(f"  Plots: {OUTPUT_DIR}/plots/")
print(f"  Results: {OUTPUT_DIR}/code/prior_check_results.npz")
print("\n" + "="*80)
