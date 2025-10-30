"""
REVISED Prior Predictive Check for Experiment 1: Robust Logarithmic Regression

This script implements the REVISED prior specification:
- sigma: Half-Cauchy(0, 0.2) -> Half-Normal(0, 0.10)  [CHANGED]
- beta: Normal(0.3, 0.3) -> Normal(0.3, 0.2)           [TIGHTENED]

Purpose:
1. Verify that revised priors resolve the 4 failed checks
2. Generate comparison visualizations showing improvement
3. Make PASS/FAIL decision on revised specification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gamma, t as student_t, halfnorm
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/revised_v2")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"

# Ensure directories exist
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Data context (from specification)
N_OBS = 27
X_MIN, X_MAX = 1.0, 31.5
Y_MIN, Y_MAX = 1.77, 2.72
Y_MEAN, Y_SD = 2.33, 0.27

# Generate x values matching the observed range
x_observed = np.linspace(X_MIN, X_MAX, N_OBS)

# Additional x values for visualization (including extrapolation)
x_extended = np.linspace(0.5, 50, 100)

print("=" * 80)
print("REVISED PRIOR PREDICTIVE CHECK: Experiment 1")
print("=" * 80)
print()
print("REVISED PRIORS:")
print("  alpha ~ Normal(2.0, 0.5)         [UNCHANGED]")
print("  beta  ~ Normal(0.3, 0.2)         [TIGHTENED: was 0.3 (v1: 0.2)]")
print("  c     ~ Gamma(2, 2)              [UNCHANGED]")
print("  nu    ~ Gamma(2, 0.1)            [UNCHANGED]")
print("  sigma ~ Half-Normal(0, 0.10)     [CHANGED: v1=Half-N(0,0.15), v2=Half-N(0,0.10)(0, 0.2)]")
print()

# ============================================================================
# 1. Sample from REVISED Prior Predictive Distribution
# ============================================================================

print("Step 1: Sampling from REVISED prior predictive distribution (1000 draws)...")

N_SAMPLES = 1000

# Sample from REVISED priors
alpha_prior = norm.rvs(loc=2.0, scale=0.5, size=N_SAMPLES)
beta_prior = norm.rvs(loc=0.3, scale=0.2, size=N_SAMPLES)  # TIGHTENED
c_prior = gamma.rvs(a=2, scale=1/2, size=N_SAMPLES)
nu_prior = gamma.rvs(a=2, scale=1/0.1, size=N_SAMPLES)
sigma_prior = halfnorm.rvs(loc=0, scale=0.10, size=N_SAMPLES)  # CHANGED to Half-Normal

print("Sampling complete.")
print()

# ============================================================================
# 2. Generate Predictions
# ============================================================================

print("Step 2: Generating prior predictive samples...")

# Initialize arrays
y_sim = np.zeros((N_SAMPLES, N_OBS))
mu = np.zeros((N_SAMPLES, N_OBS))

# Generate predictions for each prior sample
for i in range(N_SAMPLES):
    # Compute mean function
    mu[i, :] = alpha_prior[i] + beta_prior[i] * np.log(x_observed + c_prior[i])

    # Sample from Student-t likelihood
    for j in range(N_OBS):
        y_sim[i, j] = student_t.rvs(df=nu_prior[i], loc=mu[i, j], scale=sigma_prior[i])

# Compute diagnostics
min_y_sim = np.min(y_sim, axis=1)
max_y_sim = np.max(y_sim, axis=1)
mean_y_sim = np.mean(y_sim, axis=1)
sd_y_sim = np.std(y_sim, axis=1)

# Monotonicity: mu[N-1] - mu[0]
monotonic_increase = mu[:, -1] - mu[:, 0]

# Extrapolation: predictions at x=50
mu_x50 = alpha_prior + beta_prior * np.log(50 + c_prior)
y_x50 = np.array([student_t.rvs(df=nu_prior[i], loc=mu_x50[i], scale=sigma_prior[i]) for i in range(N_SAMPLES)])

print(f"Generated {N_SAMPLES} prior predictive datasets.")
print()

# ============================================================================
# 3. Compute Diagnostics
# ============================================================================

print("Step 3: Computing diagnostic statistics...")
print()

# Prior parameter summaries
print("REVISED PRIOR PARAMETER SUMMARIES:")
print("-" * 80)
params = {
    'alpha (intercept)': alpha_prior,
    'beta (slope)': beta_prior,
    'c (log shift)': c_prior,
    'nu (df)': nu_prior,
    'sigma (scale)': sigma_prior
}

for name, values in params.items():
    print(f"{name:20s}: mean={np.mean(values):7.3f}, "
          f"sd={np.std(values):6.3f}, "
          f"median={np.median(values):7.3f}, "
          f"95% CI=[{np.percentile(values, 2.5):6.3f}, {np.percentile(values, 97.5):6.3f}]")
print()

# Prior predictive summaries
print("REVISED PRIOR PREDICTIVE SUMMARIES:")
print("-" * 80)
print(f"Y range in data:           [{Y_MIN:.2f}, {Y_MAX:.2f}]")
print(f"Y mean in data:            {Y_MEAN:.2f} ± {Y_SD:.2f}")
print()
print(f"Prior pred Y range:        [{np.min(min_y_sim):.2f}, {np.max(max_y_sim):.2f}]")
print(f"Prior pred Y mean (avg):   {np.mean(mean_y_sim):.2f} ± {np.std(mean_y_sim):.2f}")
print(f"Prior pred Y SD (avg):     {np.mean(sd_y_sim):.2f} ± {np.std(sd_y_sim):.2f}")
print()

# Plausibility checks
print("PLAUSIBILITY CHECKS:")
print("-" * 80)

# Check 1: Y in plausible range [0.5, 4.5]
plausible_min, plausible_max = 0.5, 4.5
in_plausible_range = np.mean((min_y_sim >= plausible_min) & (max_y_sim <= plausible_max))
check1_pass = in_plausible_range >= 0.80
print(f"1. Predictions in [{plausible_min}, {plausible_max}]:        "
      f"{in_plausible_range*100:.1f}% (target: ≥80%) {'PASS' if check1_pass else 'FAIL'}")

# Check 2: Monotonic increase
monotonic_pct = np.mean(monotonic_increase > 0)
check2_pass = monotonic_pct >= 0.90
print(f"2. Monotonically increasing curves:     {monotonic_pct*100:.1f}% (target: ≥90%) {'PASS' if check2_pass else 'FAIL'}")

# Check 3: Observed data coverage
coverage_counts = np.zeros(N_OBS)
for i in range(N_OBS):
    y_at_x = y_sim[:, i]
    pct_2p5 = np.percentile(y_at_x, 2.5)
    pct_97p5 = np.percentile(y_at_x, 97.5)
    coverage_counts[i] = (pct_2p5 <= Y_MAX) and (pct_97p5 >= Y_MIN)

coverage_pct = np.mean(coverage_counts)
check3_pass = coverage_pct >= 0.80
print(f"3. Observed data in 95% prior interval: {coverage_pct*100:.1f}% of x values (target: ≥80%) {'PASS' if check3_pass else 'FAIL'}")

# Check 4: Extrapolation at x=50
extrapolation_reasonable = np.mean(y_x50 < 5.0)
check4_pass = extrapolation_reasonable >= 0.80
print(f"4. Predictions at x=50 reasonable (<5): {extrapolation_reasonable*100:.1f}% (target: ≥80%) {'PASS' if check4_pass else 'FAIL'}")

# Check 5: Extreme value detection
extreme_low = np.mean(min_y_sim < 0)
check5_pass = extreme_low < 0.05
print(f"5. Extreme predictions (Y<0):           {extreme_low*100:.1f}% (target: <5%) {'PASS' if check5_pass else 'FAIL'}")

extreme_high = np.mean(max_y_sim > 10)
check6_pass = extreme_high < 0.05
print(f"6. Extreme predictions (Y>10):          {extreme_high*100:.1f}% (target: <5%) {'PASS' if check6_pass else 'FAIL'}")

# Check 7: Scale alignment
y_in_2sd = np.mean((mean_y_sim >= Y_MEAN - 2*Y_SD) & (mean_y_sim <= Y_MEAN + 2*Y_SD))
check7_pass = y_in_2sd >= 0.70
print(f"7. Mean predictions within ±2 SD:       {y_in_2sd*100:.1f}% (target: ≥70%) {'PASS' if check7_pass else 'FAIL'}")

print()

# Overall assessment
all_checks_pass = all([check1_pass, check2_pass, check3_pass, check4_pass, check5_pass, check6_pass, check7_pass])
num_passed = sum([check1_pass, check2_pass, check3_pass, check4_pass, check5_pass, check6_pass, check7_pass])

print(f"SUMMARY: {num_passed}/7 checks passed")
print()

# ============================================================================
# 4. Create Visualizations
# ============================================================================

print("Step 4: Creating diagnostic visualizations...")
print()

# Load original diagnostics for comparison
ORIGINAL_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/code")
with open(ORIGINAL_DIR / 'diagnostics.json', 'r') as f:
    original_diagnostics = json.load(f)

# -------------------------------------------------------------------------
# Visualization 1: Before/After Parameter Comparison
# -------------------------------------------------------------------------
print("Creating: prior_comparison_before_after.png")

# Need to load original parameter samples for comparison
# Re-generate original samples with same seed for fair comparison
np.random.seed(42)
from scipy.stats import halfcauchy

alpha_orig = norm.rvs(loc=2.0, scale=0.5, size=N_SAMPLES)
beta_orig = norm.rvs(loc=0.3, scale=0.3, size=N_SAMPLES)
c_orig = gamma.rvs(a=2, scale=1/2, size=N_SAMPLES)
nu_orig = gamma.rvs(a=2, scale=1/0.1, size=N_SAMPLES)
sigma_orig = halfcauchy.rvs(loc=0, scale=0.2, size=N_SAMPLES)

# Reset seed for revised (already sampled above)
np.random.seed(42)
alpha_prior = norm.rvs(loc=2.0, scale=0.5, size=N_SAMPLES)
beta_prior = norm.rvs(loc=0.3, scale=0.2, size=N_SAMPLES)
c_prior = gamma.rvs(a=2, scale=1/2, size=N_SAMPLES)
nu_prior = gamma.rvs(a=2, scale=1/0.1, size=N_SAMPLES)
sigma_prior = halfnorm.rvs(loc=0, scale=0.10, size=N_SAMPLES)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Prior Comparison: Original vs Revised', fontsize=16, fontweight='bold')

# Only show parameters that changed: beta and sigma
comparisons = [
    (beta_orig, beta_prior, 'beta (slope)', 'Normal(0.3, 0.3)', 'Normal(0.3, 0.2)', [-0.5, 1.2]),
    (sigma_orig, sigma_prior, 'sigma (residual scale)', 'Half-Cauchy(0, 0.2)', 'Half-Normal(0, 0.10)', [0, 1.0])
]

for idx, (orig, revised, name, prior_orig, prior_revised, xlim) in enumerate(comparisons):
    ax = axes[0, idx]
    ax.hist(orig, bins=50, density=True, alpha=0.5, color='red', edgecolor='black', label='Original')
    ax.hist(revised, bins=50, density=True, alpha=0.5, color='green', edgecolor='black', label='Revised')
    ax.axvline(np.mean(orig), color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(np.mean(revised), color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{name}\nOriginal: {prior_orig}\nRevised: {prior_revised}', fontsize=10)
    ax.set_xlim(xlim)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove unused subplot
axes[0, 2].axis('off')

# Add comparison metrics in bottom row
# Metric 1: Beta negative probability
ax = axes[1, 0]
beta_neg_orig = np.mean(beta_orig < 0) * 100
beta_neg_revised = np.mean(beta_prior < 0) * 100
ax.bar(['Original', 'Revised'], [beta_neg_orig, beta_neg_revised], color=['red', 'green'], alpha=0.7, edgecolor='black')
ax.axhline(16, color='orange', linestyle='--', linewidth=2, label='Original: 16%')
ax.set_ylabel('% Negative beta', fontsize=11)
ax.set_title('Beta Prior: Probability of Negative Slope', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.text(0, beta_neg_orig + 1, f'{beta_neg_orig:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.text(1, beta_neg_revised + 1, f'{beta_neg_revised:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Metric 2: Sigma extreme values
ax = axes[1, 1]
sigma_high_orig = np.mean(sigma_orig > 0.5) * 100
sigma_high_revised = np.mean(sigma_prior > 0.5) * 100
ax.bar(['Original', 'Revised'], [sigma_high_orig, sigma_high_revised], color=['red', 'green'], alpha=0.7, edgecolor='black')
ax.set_ylabel('% sigma > 0.5', fontsize=11)
ax.set_title('Sigma Prior: Probability of Extreme Values', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.text(0, sigma_high_orig + 1, f'{sigma_high_orig:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.text(1, sigma_high_revised + 1, f'{sigma_high_revised:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Metric 3: Summary statistics
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
PARAMETER CHANGES:

Beta (slope):
  Original: N(0.3, 0.3)
  Revised:  N(0.3, 0.2)
  Impact:   P(β<0) = {beta_neg_orig:.1f}% → {beta_neg_revised:.1f}%

Sigma (scale):
  Original: Half-Cauchy(0, 0.2)
  Revised:  Half-Normal(0, 0.10)
  Impact:   P(σ>0.5) = {sigma_high_orig:.1f}% → {sigma_high_revised:.1f}%

Key Improvement:
  Heavy tail eliminated
  Mean: {np.mean(sigma_orig):.3f} → {np.mean(sigma_prior):.3f}
  SD:   {np.std(sigma_orig):.3f} → {np.std(sigma_prior):.3f}
"""
ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_comparison_before_after.png', dpi=300, bbox_inches='tight')
plt.close()

# Re-sample revised priors (seed reset affected them)
np.random.seed(42)
alpha_prior = norm.rvs(loc=2.0, scale=0.5, size=N_SAMPLES)
beta_prior = norm.rvs(loc=0.3, scale=0.2, size=N_SAMPLES)
c_prior = gamma.rvs(a=2, scale=1/2, size=N_SAMPLES)
nu_prior = gamma.rvs(a=2, scale=1/0.1, size=N_SAMPLES)
sigma_prior = halfnorm.rvs(loc=0, scale=0.10, size=N_SAMPLES)

# Regenerate predictions with revised priors
y_sim = np.zeros((N_SAMPLES, N_OBS))
mu = np.zeros((N_SAMPLES, N_OBS))
for i in range(N_SAMPLES):
    mu[i, :] = alpha_prior[i] + beta_prior[i] * np.log(x_observed + c_prior[i])
    for j in range(N_OBS):
        y_sim[i, j] = student_t.rvs(df=nu_prior[i], loc=mu[i, j], scale=sigma_prior[i])

min_y_sim = np.min(y_sim, axis=1)
max_y_sim = np.max(y_sim, axis=1)
mean_y_sim = np.mean(y_sim, axis=1)
sd_y_sim = np.std(y_sim, axis=1)
monotonic_increase = mu[:, -1] - mu[:, 0]
mu_x50 = alpha_prior + beta_prior * np.log(50 + c_prior)
y_x50 = np.array([student_t.rvs(df=nu_prior[i], loc=mu_x50[i], scale=sigma_prior[i]) for i in range(N_SAMPLES)])

# -------------------------------------------------------------------------
# Visualization 2: Prior Predictive Curves (Revised)
# -------------------------------------------------------------------------
print("Creating: prior_predictive_curves_revised.png")

fig, ax = plt.subplots(figsize=(10, 7))

# Plot 100 random prior predictive curves
n_curves = 100
indices = np.random.choice(N_SAMPLES, n_curves, replace=False)

for idx in indices:
    alpha_i = alpha_prior[idx]
    beta_i = beta_prior[idx]
    c_i = c_prior[idx]
    mu_i = alpha_i + beta_i * np.log(x_extended + c_i)
    ax.plot(x_extended, mu_i, alpha=0.15, color='steelblue', linewidth=0.5)

# Overlay observed data context
ax.axhspan(Y_MIN, Y_MAX, alpha=0.2, color='green', label='Observed Y range')
ax.axhline(Y_MEAN, color='green', linestyle='--', linewidth=2, label=f'Observed Y mean: {Y_MEAN:.2f}')
ax.axhline(Y_MEAN - 2*Y_SD, color='green', linestyle=':', linewidth=1, alpha=0.7, label='±2 SD')
ax.axhline(Y_MEAN + 2*Y_SD, color='green', linestyle=':', linewidth=1, alpha=0.7)

# Mark observed x range
ax.axvline(X_MIN, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(X_MAX, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvspan(X_MIN, X_MAX, alpha=0.05, color='red', label='Observed X range')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('μ(x) = α + β·log(x + c)', fontsize=12)
ax.set_title('REVISED Prior Predictive Curves (100 samples)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 50)
ax.set_ylim(0, 4.5)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_predictive_curves_revised.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 3: Coverage Comparison
# -------------------------------------------------------------------------
print("Creating: coverage_diagnostic_improvement.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prior Predictive Coverage: Improvement from Revised Priors', fontsize=14, fontweight='bold')

# Left: Revised
percentiles = [2.5, 10, 25, 50, 75, 90, 97.5]
pct_values = np.percentile(y_sim, percentiles, axis=0)

ax1.fill_between(x_observed, pct_values[0], pct_values[-1], alpha=0.2, color='green', label='95% prior interval')
ax1.fill_between(x_observed, pct_values[1], pct_values[-2], alpha=0.3, color='green', label='80% prior interval')
ax1.fill_between(x_observed, pct_values[2], pct_values[-3], alpha=0.4, color='green', label='50% prior interval')
ax1.plot(x_observed, pct_values[3], color='darkgreen', linewidth=2, label='Median prediction')
ax1.axhspan(Y_MIN, Y_MAX, alpha=0.15, color='blue', zorder=10, label='Observed Y range')
ax1.axhline(Y_MEAN, color='blue', linestyle='--', linewidth=2, zorder=11)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Y (prior predictive)', fontsize=12)
ax1.set_title('REVISED Priors', fontsize=12, fontweight='bold')
ax1.set_xlim(X_MIN - 1, X_MAX + 1)
ax1.set_ylim(0, 4.5)
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Right: Comparison metrics
ax2.axis('off')
comparison_text = f"""
COVERAGE IMPROVEMENT:

Original Priors:
  Y range: [-161,737, 4,719]
  Predictions in [0.5, 4.5]: 65.9%
  Extreme negative (Y<0): 12.1%
  Mean ±2SD coverage: 39.3%

REVISED Priors:
  Y range: [{np.min(min_y_sim):.1f}, {np.max(max_y_sim):.1f}]
  Predictions in [0.5, 4.5]: {in_plausible_range*100:.1f}%
  Extreme negative (Y<0): {extreme_low*100:.1f}%
  Mean ±2SD coverage: {y_in_2sd*100:.1f}%

KEY IMPROVEMENTS:
  ✓ Range reduced from ~166K to ~{np.max(max_y_sim) - np.min(min_y_sim):.1f}
  ✓ Plausible predictions: +{in_plausible_range*100 - 65.9:.1f}%
  ✓ Negative predictions: -{12.1 - extreme_low*100:.1f}%
  ✓ Scale alignment: +{y_in_2sd*100 - 39.3:.1f}%
"""
ax2.text(0.1, 0.5, comparison_text, fontsize=10, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_diagnostic_improvement.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 4: Check Results Comparison
# -------------------------------------------------------------------------
print("Creating: check_results_comparison.png")

fig, ax = plt.subplots(figsize=(12, 7))

checks = ['Check 1\nRange', 'Check 2\nMonotonic', 'Check 3\nCoverage',
          'Check 4\nExtrap', 'Check 5\nY<0', 'Check 6\nY>10', 'Check 7\nScale']

original_results = [65.9, 86.1, 100.0, 90.2, 12.1, 4.0, 39.3]
revised_results = [
    in_plausible_range * 100,
    monotonic_pct * 100,
    coverage_pct * 100,
    extrapolation_reasonable * 100,
    extreme_low * 100,
    extreme_high * 100,
    y_in_2sd * 100
]

# For checks 5 and 6, lower is better, so invert for visualization
targets = [80, 90, 80, 80, 5, 5, 70]
target_labels = ['≥80%', '≥90%', '≥80%', '≥80%', '<5%', '<5%', '≥70%']

x = np.arange(len(checks))
width = 0.35

bars1 = ax.bar(x - width/2, original_results, width, label='Original',
               color='red', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, revised_results, width, label='Revised',
               color='green', alpha=0.7, edgecolor='black')

# Add target lines
for i, (target, label) in enumerate(zip(targets, target_labels)):
    if i < 4 or i == 6:  # Higher is better
        ax.hlines(target, i - 0.5, i + 0.5, colors='blue', linestyles='--', linewidth=2, alpha=0.7)
    else:  # Lower is better (checks 5, 6)
        ax.hlines(target, i - 0.5, i + 0.5, colors='blue', linestyles='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Check', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Prior Predictive Check Results: Original vs Revised', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(checks, fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (orig, rev) in enumerate(zip(original_results, revised_results)):
    ax.text(i - width/2, orig + 2, f'{orig:.1f}', ha='center', fontsize=8)
    ax.text(i + width/2, rev + 2, f'{rev:.1f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'check_results_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 5: Comprehensive Revised Summary
# -------------------------------------------------------------------------
print("Creating: comprehensive_revised_summary.png")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Prior predictive curves (large)
ax1 = fig.add_subplot(gs[0:2, 0:2])
n_curves_summary = 200
indices_summary = np.random.choice(N_SAMPLES, n_curves_summary, replace=False)
for idx in indices_summary:
    alpha_i = alpha_prior[idx]
    beta_i = beta_prior[idx]
    c_i = c_prior[idx]
    mu_i = alpha_i + beta_i * np.log(x_extended + c_i)
    ax1.plot(x_extended, mu_i, alpha=0.08, color='steelblue', linewidth=0.5)
ax1.axhspan(Y_MIN, Y_MAX, alpha=0.2, color='green')
ax1.axhline(Y_MEAN, color='green', linestyle='--', linewidth=2)
ax1.axvspan(X_MIN, X_MAX, alpha=0.05, color='red')
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('μ(x)', fontsize=11)
ax1.set_title('REVISED Prior Predictive Curves (200 samples)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 4.5)
ax1.grid(True, alpha=0.3)

# Panel 2: Beta distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(beta_prior, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax2.axvline(np.mean(beta_prior), color='darkblue', linestyle='--', linewidth=2)
ax2.set_xlabel('β (slope)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title(f'REVISED: β ~ N(0.3, 0.2)\nP(β<0)={np.mean(beta_prior<0)*100:.1f}%', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel 3: Sigma distribution
ax3 = fig.add_subplot(gs[1, 2])
ax3.hist(sigma_prior, bins=40, density=True, alpha=0.7, color='orange', edgecolor='black')
ax3.axvline(np.mean(sigma_prior), color='darkorange', linestyle='--', linewidth=2)
ax3.set_xlabel('σ (scale)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title(f'REVISED: σ ~ Half-N(0, 0.15)\nMean={np.mean(sigma_prior):.3f}', fontsize=10, fontweight='bold')
ax3.set_xlim([0, 0.4])
ax3.grid(True, alpha=0.3)

# Panel 4: Coverage at x_min
ax4 = fig.add_subplot(gs[2, 0])
y_at_x_min = y_sim[:, 0]
ax4.hist(y_at_x_min, bins=40, density=True, alpha=0.7, color='coral', edgecolor='black')
ax4.axvspan(Y_MIN, Y_MAX, alpha=0.2, color='green')
ax4.set_xlabel(f'Y at x={x_observed[0]:.1f}', fontsize=10)
ax4.set_ylabel('Density', fontsize=10)
ax4.set_title('Predictions at x_min', fontsize=11, fontweight='bold')
ax4.set_xlim(0, 4.5)
ax4.grid(True, alpha=0.3)

# Panel 5: Coverage at x_max
ax5 = fig.add_subplot(gs[2, 1])
y_at_x_max = y_sim[:, -1]
ax5.hist(y_at_x_max, bins=40, density=True, alpha=0.7, color='coral', edgecolor='black')
ax5.axvspan(Y_MIN, Y_MAX, alpha=0.2, color='green')
ax5.set_xlabel(f'Y at x={x_observed[-1]:.1f}', fontsize=10)
ax5.set_ylabel('Density', fontsize=10)
ax5.set_title('Predictions at x_max', fontsize=11, fontweight='bold')
ax5.set_xlim(0, 4.5)
ax5.grid(True, alpha=0.3)

# Panel 6: Pass/Fail Summary
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
pass_fail_text = f"""
CHECK RESULTS:

✓ Check 1: {in_plausible_range*100:.1f}%
✓ Check 2: {monotonic_pct*100:.1f}%
✓ Check 3: {coverage_pct*100:.1f}%
✓ Check 4: {extrapolation_reasonable*100:.1f}%
✓ Check 5: {extreme_low*100:.1f}%
✓ Check 6: {extreme_high*100:.1f}%
✓ Check 7: {y_in_2sd*100:.1f}%

STATUS: {'PASS' if all_checks_pass else 'FAIL'}
({num_passed}/7 passed)
"""
ax6.text(0.1, 0.5, pass_fail_text, fontsize=11, family='monospace',
         verticalalignment='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if all_checks_pass else 'lightcoral', alpha=0.5))

fig.suptitle('Comprehensive REVISED Prior Predictive Check Summary', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / 'comprehensive_revised_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print()
print("All visualizations created successfully.")
print()

# ============================================================================
# 5. Save Diagnostic Results
# ============================================================================

print("Step 5: Saving diagnostic results...")

# Recalculate all checks
check1_pass = in_plausible_range >= 0.80
check2_pass = monotonic_pct >= 0.90
check3_pass = coverage_pct >= 0.80
check4_pass = extrapolation_reasonable >= 0.80
check5_pass = extreme_low < 0.05
check6_pass = extreme_high < 0.05
check7_pass = y_in_2sd >= 0.70

diagnostics = {
    'n_samples': N_SAMPLES,
    'revision_notes': {
        'beta_prior': 'TIGHTENED: Normal(0.3, 0.3) -> Normal(0.3, 0.2)',
        'sigma_prior': 'CHANGED: Half-Cauchy(0, 0.2) -> Half-Normal(0, 0.10)'
    },
    'data_context': {
        'n_obs': N_OBS,
        'x_range': [float(X_MIN), float(X_MAX)],
        'y_range': [float(Y_MIN), float(Y_MAX)],
        'y_mean': float(Y_MEAN),
        'y_sd': float(Y_SD)
    },
    'prior_parameter_summaries': {
        name: {
            'mean': float(np.mean(values)),
            'sd': float(np.std(values)),
            'median': float(np.median(values)),
            'q025': float(np.percentile(values, 2.5)),
            'q975': float(np.percentile(values, 97.5))
        } for name, values in params.items()
    },
    'prior_predictive_summaries': {
        'y_range_min': float(np.min(min_y_sim)),
        'y_range_max': float(np.max(max_y_sim)),
        'mean_y_sim_avg': float(np.mean(mean_y_sim)),
        'mean_y_sim_sd': float(np.std(mean_y_sim)),
        'sd_y_sim_avg': float(np.mean(sd_y_sim)),
        'sd_y_sim_sd': float(np.std(sd_y_sim))
    },
    'plausibility_checks': {
        'predictions_in_range_0.5_4.5_pct': float(in_plausible_range * 100),
        'monotonically_increasing_pct': float(monotonic_pct * 100),
        'observed_data_coverage_pct': float(coverage_pct * 100),
        'extrapolation_reasonable_pct': float(extrapolation_reasonable * 100),
        'extreme_low_pct': float(extreme_low * 100),
        'extreme_high_pct': float(extreme_high * 100),
        'mean_within_2sd_pct': float(y_in_2sd * 100)
    },
    'decision_criteria': {
        'check_1_pass': bool(check1_pass),
        'check_2_pass': bool(check2_pass),
        'check_3_pass': bool(check3_pass),
        'check_4_pass': bool(check4_pass),
        'check_5_pass': bool(check5_pass),
        'check_6_pass': bool(check6_pass),
        'check_7_pass': bool(check7_pass)
    },
    'improvement_over_original': {
        'check_1_improvement_pct': float(in_plausible_range * 100 - 65.9),
        'check_2_improvement_pct': float(monotonic_pct * 100 - 86.1),
        'check_5_improvement_pct': float(12.1 - extreme_low * 100),
        'check_7_improvement_pct': float(y_in_2sd * 100 - 39.3)
    }
}

# Save to JSON
with open(CODE_DIR / 'revised_diagnostics.json', 'w') as f:
    json.dump(diagnostics, f, indent=2)

print("Diagnostics saved to revised_diagnostics.json")
print()

# ============================================================================
# 6. Final Assessment
# ============================================================================

print("=" * 80)
print("FINAL ASSESSMENT - REVISED PRIORS")
print("=" * 80)
print()

all_checks_pass = all(diagnostics['decision_criteria'].values())

if all_checks_pass:
    print("STATUS: PASS ✓")
    print()
    print("All 7 plausibility checks PASSED with revised priors.")
    print()
    print("KEY IMPROVEMENTS FROM ORIGINAL:")
    print(f"  - Predictions in range: 65.9% → {in_plausible_range*100:.1f}% (+{in_plausible_range*100 - 65.9:.1f}%)")
    print(f"  - Monotonic increase: 86.1% → {monotonic_pct*100:.1f}% (+{monotonic_pct*100 - 86.1:.1f}%)")
    print(f"  - Extreme negative: 12.1% → {extreme_low*100:.1f}% (-{12.1 - extreme_low*100:.1f}%)")
    print(f"  - Scale alignment: 39.3% → {y_in_2sd*100:.1f}% (+{y_in_2sd*100 - 39.3:.1f}%)")
    print()
    print("CONCLUSION:")
    print("  The revised priors successfully resolve all issues identified in the")
    print("  original prior predictive check. The model is ready to proceed to")
    print("  simulation-based validation.")
else:
    print("STATUS: FAIL ✗")
    print()
    print("One or more plausibility checks still failed:")
    for check_name, passed in diagnostics['decision_criteria'].items():
        if not passed:
            print(f"  - {check_name}: FAILED")
    print()
    print("Further prior adjustment needed before proceeding.")

print()
print("=" * 80)
print("Revised prior predictive check complete.")
print("See findings.md for detailed analysis.")
print("=" * 80)
