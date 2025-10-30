"""
Prior Predictive Check for Experiment 1: Beta-Binomial Hierarchical Model

This script validates the prior distributions before model fitting by:
1. Sampling from prior distributions
2. Generating synthetic data from the prior
3. Comparing prior predictions with observed data
4. Assessing prior plausibility and computational viability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_SAMPLES = 1000
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check/plots/"

# Load observed data
data = pd.read_csv("/workspace/data/data.csv")
n_groups = len(data)
n_obs = data['n'].values
r_obs = data['r'].values
p_obs = data['proportion'].values

print("="*80)
print("PRIOR PREDICTIVE CHECK: BETA-BINOMIAL HIERARCHICAL MODEL")
print("="*80)
print(f"\nObserved Data Summary:")
print(f"  Number of groups: {n_groups}")
print(f"  Sample sizes (n): {n_obs.min()} to {n_obs.max()}")
print(f"  Observed counts (r): {r_obs.min()} to {r_obs.max()}")
print(f"  Observed proportions: {p_obs.min():.4f} to {p_obs.max():.4f}")
print(f"  Pooled rate: {r_obs.sum() / n_obs.sum():.4f}")

# Prior specifications
print("\n" + "="*80)
print("PRIOR SPECIFICATIONS")
print("="*80)
print("μ ~ Beta(2, 18)      # Population mean proportion")
print("κ ~ Gamma(2, 0.1)    # Concentration parameter")
print("p_i | μ, κ ~ Beta(μκ, (1-μ)κ)")
print("r_i | p_i, n_i ~ Binomial(n_i, p_i)")

# Sample from priors
print("\n" + "="*80)
print("SAMPLING FROM PRIOR PREDICTIVE DISTRIBUTION")
print("="*80)
print(f"Sampling {N_PRIOR_SAMPLES} draws from prior predictive distribution...")

# Sample hyperparameters
mu_samples = np.random.beta(2, 18, size=N_PRIOR_SAMPLES)
kappa_samples = np.random.gamma(2, scale=1/0.1, size=N_PRIOR_SAMPLES)

# Compute derived parameters
alpha_samples = mu_samples * kappa_samples
beta_samples = (1 - mu_samples) * kappa_samples
phi_samples = 1 + 1/kappa_samples

# Sample group-level proportions
p_samples = np.zeros((N_PRIOR_SAMPLES, n_groups))
for i in range(N_PRIOR_SAMPLES):
    alpha_i = alpha_samples[i]
    beta_i = beta_samples[i]
    p_samples[i, :] = np.random.beta(alpha_i, beta_i, size=n_groups)

# Sample observed counts
r_samples = np.zeros((N_PRIOR_SAMPLES, n_groups), dtype=int)
for i in range(N_PRIOR_SAMPLES):
    for j in range(n_groups):
        r_samples[i, j] = np.random.binomial(n_obs[j], p_samples[i, j])

print("Prior predictive sampling complete!")

# Compute prior predictive proportions
p_pred_samples = r_samples / n_obs[np.newaxis, :]

print("\n" + "="*80)
print("PRIOR DIAGNOSTICS")
print("="*80)

# 1. Check population mean (mu)
print("\n1. POPULATION MEAN (μ)")
print(f"   Prior: Beta(2, 18)")
print(f"   E[μ] = {2/(2+18):.4f}")
print(f"   Sampled mean: {mu_samples.mean():.4f}")
print(f"   Sampled 95% CI: [{np.percentile(mu_samples, 2.5):.4f}, {np.percentile(mu_samples, 97.5):.4f}]")
print(f"   Observed pooled rate: {r_obs.sum() / n_obs.sum():.4f}")
mu_covers = np.percentile(mu_samples, 2.5) <= r_obs.sum() / n_obs.sum() <= np.percentile(mu_samples, 97.5)
print(f"   Does prior cover observed? {mu_covers}")

# 2. Check concentration parameter (kappa)
print("\n2. CONCENTRATION PARAMETER (κ)")
print(f"   Prior: Gamma(2, 0.1)")
print(f"   E[κ] = {2/0.1:.1f}")
print(f"   Sampled mean: {kappa_samples.mean():.2f}")
print(f"   Sampled 95% CI: [{np.percentile(kappa_samples, 2.5):.2f}, {np.percentile(kappa_samples, 97.5):.2f}]")

# 3. Check overdispersion parameter (phi)
print("\n3. OVERDISPERSION PARAMETER (φ = 1 + 1/κ)")
print(f"   Expected empirical range: 2-10 (high overdispersion)")
print(f"   Sampled mean: {phi_samples.mean():.2f}")
print(f"   Sampled 95% CI: [{np.percentile(phi_samples, 2.5):.2f}, {np.percentile(phi_samples, 97.5):.2f}]")
phi_adequate = np.percentile(phi_samples, 97.5) >= 2
print(f"   Prior allows strong overdispersion? {phi_adequate}")

# 4. Check group-level proportions
print("\n4. GROUP-LEVEL PROPORTIONS (p_i)")
print(f"   Observed range: [{p_obs.min():.4f}, {p_obs.max():.4f}]")
print(f"   Prior predictive range: [{p_samples.min():.4f}, {p_samples.max():.4f}]")
print(f"   Prior predictive 95% CI: [{np.percentile(p_samples, 2.5):.4f}, {np.percentile(p_samples, 97.5):.4f}]")
p_covers = np.percentile(p_samples, 2.5) <= p_obs.min() and p_obs.max() <= np.percentile(p_samples, 97.5)
print(f"   Does prior cover observed range? {p_covers}")

# Check for implausible values
implausible_high = (p_samples > 0.5).mean()
print(f"   Proportion of samples with p_i > 0.5: {implausible_high:.4f}")
if implausible_high > 0.01:
    print(f"   WARNING: Prior generates implausibly high proportions")

# 5. Check prior predictive counts
print("\n5. PRIOR PREDICTIVE COUNTS (r_i)")
print(f"   Observed range: [{r_obs.min()}, {r_obs.max()}]")
print(f"   Prior predictive range: [{r_samples.min()}, {r_samples.max()}]")

# Check coverage for each observed data point
coverage_count = 0
for i in range(n_groups):
    lower = np.percentile(r_samples[:, i], 2.5)
    upper = np.percentile(r_samples[:, i], 97.5)
    if lower <= r_obs[i] <= upper:
        coverage_count += 1

print(f"   Observed data within prior predictive 95% CI: {coverage_count}/{n_groups} groups")

# 6. Check for computational issues
print("\n6. COMPUTATIONAL DIAGNOSTICS")
extreme_alpha = (alpha_samples > 1000).mean()
extreme_beta = (beta_samples > 1000).mean()
extreme_kappa = (kappa_samples > 1000).mean()
print(f"   Proportion with α > 1000: {extreme_alpha:.4f}")
print(f"   Proportion with β > 1000: {extreme_beta:.4f}")
print(f"   Proportion with κ > 1000: {extreme_kappa:.4f}")

if extreme_alpha > 0.01 or extreme_beta > 0.01:
    print(f"   WARNING: Prior may generate extreme Beta parameters")

# 7. Prior-data conflict check
print("\n7. PRIOR-DATA CONFLICT")
# For each observed count, compute percentile in prior predictive
percentiles = []
for i in range(n_groups):
    pct = (r_samples[:, i] <= r_obs[i]).mean() * 100
    percentiles.append(pct)

percentiles = np.array(percentiles)
extreme_percentiles = ((percentiles < 2.5) | (percentiles > 97.5)).sum()
print(f"   Observed counts in extreme tails (<2.5% or >97.5%): {extreme_percentiles}/{n_groups}")

if extreme_percentiles > n_groups * 0.2:
    print(f"   WARNING: Potential prior-data conflict")

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# VISUALIZATION 1: Prior distributions of hyperparameters
print("\nCreating: hyperparameter_priors.png")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# μ prior
ax = axes[0, 0]
ax.hist(mu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
x = np.linspace(0, 1, 1000)
ax.plot(x, stats.beta.pdf(x, 2, 18), 'r-', lw=2, label='Beta(2, 18)')
ax.axvline(r_obs.sum() / n_obs.sum(), color='green', linestyle='--', lw=2, label=f'Observed pooled: {r_obs.sum() / n_obs.sum():.3f}')
ax.axvline(mu_samples.mean(), color='blue', linestyle=':', lw=2, label=f'Prior mean: {mu_samples.mean():.3f}')
ax.set_xlabel('Population Mean (μ)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution of μ', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# κ prior
ax = axes[0, 1]
ax.hist(kappa_samples, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
x = np.linspace(0, np.percentile(kappa_samples, 99.5), 1000)
ax.plot(x, stats.gamma.pdf(x, 2, scale=1/0.1), 'r-', lw=2, label='Gamma(2, 0.1)')
ax.axvline(kappa_samples.mean(), color='blue', linestyle=':', lw=2, label=f'Prior mean: {kappa_samples.mean():.1f}')
ax.set_xlabel('Concentration (κ)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution of κ', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, np.percentile(kappa_samples, 99))

# φ = 1 + 1/κ (overdispersion)
ax = axes[1, 0]
ax.hist(phi_samples, bins=50, density=True, alpha=0.7, color='mediumpurple', edgecolor='black')
ax.axvline(phi_samples.mean(), color='blue', linestyle=':', lw=2, label=f'Prior mean: {phi_samples.mean():.2f}')
ax.axvspan(2, 10, alpha=0.2, color='green', label='Expected range: 2-10')
ax.set_xlabel('Overdispersion (φ = 1 + 1/κ)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution of Overdispersion Parameter', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(1, np.percentile(phi_samples, 99))

# α and β relationship
ax = axes[1, 1]
scatter = ax.scatter(alpha_samples, beta_samples, alpha=0.3, s=10, c=mu_samples, cmap='viridis')
ax.set_xlabel('α = μκ', fontsize=11)
ax.set_ylabel('β = (1-μ)κ', fontsize=11)
ax.set_title('Prior Relationship: α vs β', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('μ', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}hyperparameter_priors.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}hyperparameter_priors.png")

# VISUALIZATION 2: Group-level proportion priors
print("Creating: group_proportion_priors.png")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of all p_i samples
ax = axes[0, 0]
ax.hist(p_samples.flatten(), bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
for p in p_obs:
    ax.axvline(p, color='red', alpha=0.6, lw=1)
ax.axvline(np.nan, color='red', alpha=0.6, lw=2, label='Observed proportions')
ax.set_xlabel('Group Proportion (p_i)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive Distribution of Group Proportions', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 0.5)

# Box plot of p_i by group with observed overlay
ax = axes[0, 1]
positions = np.arange(n_groups)
bp = ax.boxplot([p_samples[:, i] for i in range(n_groups)],
                 positions=positions, widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='blue', linewidth=2))
ax.scatter(positions, p_obs, color='red', s=100, zorder=10, label='Observed', marker='D')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Prior Predictive vs Observed Proportions by Group', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_xticklabels(range(1, n_groups+1))

# Coverage plot: percentile of observed in prior predictive
ax = axes[1, 0]
colors = ['red' if p < 2.5 or p > 97.5 else 'green' for p in percentiles]
ax.bar(range(n_groups), percentiles, color=colors, alpha=0.7, edgecolor='black')
ax.axhspan(2.5, 97.5, alpha=0.2, color='green', label='Expected 95% range')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Percentile of Observed Count', fontsize=11)
ax.set_title('Prior Predictive Coverage Check', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_xticks(range(n_groups))
ax.set_xticklabels(range(1, n_groups+1))

# Quantile-quantile plot
ax = axes[1, 1]
# For each group, compute empirical quantiles
empirical_quantiles = []
for i in range(n_groups):
    # Compute quantile of observed in prior predictive
    q = (r_samples[:, i] <= r_obs[i]).mean()
    empirical_quantiles.append(q)

empirical_quantiles = np.array(empirical_quantiles)
theoretical_quantiles = np.linspace(0, 1, n_groups+2)[1:-1]

ax.scatter(theoretical_quantiles, np.sort(empirical_quantiles), s=100, alpha=0.7, color='steelblue')
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
ax.fill_between([0, 1], [0.025, 0.025], [0.975, 0.975], alpha=0.2, color='green', label='95% tolerance')
ax.set_xlabel('Theoretical Uniform Quantiles', fontsize=11)
ax.set_ylabel('Empirical Quantiles (Obs in Prior Pred)', fontsize=11)
ax.set_title('Calibration Check: QQ Plot', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}group_proportion_priors.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}group_proportion_priors.png")

# VISUALIZATION 3: Prior predictive counts
print("Creating: prior_predictive_counts.png")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(n_groups):
    ax = axes[i]

    # Histogram of prior predictive counts
    ax.hist(r_samples[:, i], bins=30, density=True, alpha=0.7,
            color='steelblue', edgecolor='black')

    # Mark observed value
    ax.axvline(r_obs[i], color='red', linestyle='--', lw=2,
               label=f'Observed: {r_obs[i]}')

    # Mark prior predictive median and 95% CI
    median = np.median(r_samples[:, i])
    lower = np.percentile(r_samples[:, i], 2.5)
    upper = np.percentile(r_samples[:, i], 97.5)
    ax.axvline(median, color='blue', linestyle=':', lw=1.5, alpha=0.7,
               label=f'Prior median: {median:.0f}')
    ax.axvspan(lower, upper, alpha=0.2, color='blue',
               label=f'95% CI: [{lower:.0f}, {upper:.0f}]')

    ax.set_xlabel('Count', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'Group {i+1} (n={n_obs[i]})', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}prior_predictive_counts.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}prior_predictive_counts.png")

# VISUALIZATION 4: Comprehensive diagnostic dashboard
print("Creating: diagnostic_dashboard.png")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. μ prior vs observed
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(mu_samples, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(r_obs.sum() / n_obs.sum(), color='red', linestyle='--', lw=2.5,
            label=f'Observed: {r_obs.sum() / n_obs.sum():.3f}')
ax1.axvspan(np.percentile(mu_samples, 2.5), np.percentile(mu_samples, 97.5),
            alpha=0.2, color='blue', label='Prior 95% CI')
ax1.set_xlabel('μ (Population Mean)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_title('μ Prior Coverage', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# 2. φ prior and expected range
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(phi_samples, bins=40, density=True, alpha=0.7, color='coral', edgecolor='black')
ax2.axvspan(2, 10, alpha=0.2, color='green', label='Expected φ: 2-10')
ax2.axvline(phi_samples.mean(), color='blue', linestyle=':', lw=2,
            label=f'Prior mean: {phi_samples.mean():.2f}')
ax2.set_xlabel('φ (Overdispersion)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('Overdispersion Parameter', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(1, min(50, np.percentile(phi_samples, 99)))

# 3. Sample size vs observed count
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(n_obs, r_obs, s=150, color='red', alpha=0.8, label='Observed',
            zorder=10, edgecolor='black', linewidth=1.5)
# Plot prior predictive samples (random subset)
for idx in np.random.choice(N_PRIOR_SAMPLES, size=100, replace=False):
    ax3.scatter(n_obs, r_samples[idx, :], s=20, color='steelblue', alpha=0.1)
ax3.scatter([], [], s=20, color='steelblue', alpha=0.5, label='Prior predictive (100 draws)')
ax3.set_xlabel('Sample Size (n)', fontsize=10)
ax3.set_ylabel('Count (r)', fontsize=10)
ax3.set_title('Count vs Sample Size', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 4. Prior predictive proportion distribution
ax4 = fig.add_subplot(gs[1, :])
# Violin plots for each group
parts = ax4.violinplot([p_samples[:, i] for i in range(n_groups)],
                        positions=range(n_groups), widths=0.7, showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)
# Overlay observed
ax4.scatter(range(n_groups), p_obs, color='red', s=150, zorder=10,
            label='Observed', marker='D', edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Group', fontsize=10)
ax4.set_ylabel('Proportion', fontsize=10)
ax4.set_title('Prior Predictive Proportions vs Observed (All Groups)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, axis='y')
ax4.set_xticks(range(n_groups))
ax4.set_xticklabels(range(1, n_groups+1))

# 5. Coverage summary
ax5 = fig.add_subplot(gs[2, 0])
coverage_summary = {
    'Within 95% CI': coverage_count,
    'Outside 95% CI': n_groups - coverage_count
}
colors_pie = ['green', 'red']
ax5.pie(coverage_summary.values(), labels=coverage_summary.keys(), autopct='%1.1f%%',
        colors=colors_pie, startangle=90)
ax5.set_title(f'Coverage: {coverage_count}/{n_groups} Groups', fontsize=11, fontweight='bold')

# 6. Percentile distribution
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(percentiles, bins=20, density=True, alpha=0.7, color='mediumpurple', edgecolor='black')
ax6.axvspan(0, 2.5, alpha=0.3, color='red', label='Extreme tails')
ax6.axvspan(97.5, 100, alpha=0.3, color='red')
ax6.set_xlabel('Percentile of Observed in Prior Pred', fontsize=10)
ax6.set_ylabel('Density', fontsize=10)
ax6.set_title('Distribution of Percentiles', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

# 7. Summary statistics table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
summary_text = f"""
PRIOR SUMMARY STATISTICS

μ (Population Mean):
  Prior 95% CI: [{np.percentile(mu_samples, 2.5):.3f}, {np.percentile(mu_samples, 97.5):.3f}]
  Observed: {r_obs.sum() / n_obs.sum():.3f}

κ (Concentration):
  Prior mean: {kappa_samples.mean():.1f}
  Prior 95% CI: [{np.percentile(kappa_samples, 2.5):.1f}, {np.percentile(kappa_samples, 97.5):.1f}]

φ (Overdispersion):
  Prior mean: {phi_samples.mean():.2f}
  Prior 95% CI: [{np.percentile(phi_samples, 2.5):.2f}, {np.percentile(phi_samples, 97.5):.2f}]

Coverage:
  Groups in 95% CI: {coverage_count}/{n_groups}
  Extreme tail: {extreme_percentiles}/{n_groups}
"""
ax7.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Prior Predictive Check: Diagnostic Dashboard', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(f"{OUTPUT_DIR}diagnostic_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}diagnostic_dashboard.png")

print("\nAll visualizations saved successfully!")

# FINAL ASSESSMENT
print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

# Criteria for PASS/FAIL
checks = {
    'mu_coverage': mu_covers,
    'phi_range': phi_adequate,
    'p_plausible': implausible_high < 0.05,
    'coverage_adequate': coverage_count >= n_groups * 0.7,
    'no_extreme_params': extreme_alpha < 0.05 and extreme_beta < 0.05,
    'no_severe_conflict': extreme_percentiles <= n_groups * 0.25
}

print("\nDecision Criteria:")
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}: {passed}")

all_passed = all(checks.values())
decision = "PASS" if all_passed else "FAIL"

print("\n" + "="*80)
print(f"FINAL DECISION: {decision}")
print("="*80)

if decision == "PASS":
    print("\nThe prior specifications are appropriate:")
    print("  - Priors generate scientifically plausible parameter values")
    print("  - Prior predictive distribution covers observed data adequately")
    print("  - No computational red flags detected")
    print("  - Prior allows for observed overdispersion")
    print("\nRECOMMENDATION: Proceed to model fitting (Experiment 1)")
else:
    print("\nThe prior specifications have issues:")
    failed_checks = [check for check, passed in checks.items() if not passed]
    print(f"  Failed checks: {', '.join(failed_checks)}")
    print("\nRECOMMENDATION: Revise priors before proceeding")

    # Specific recommendations
    if not checks['mu_coverage']:
        print("\n  ISSUE: μ prior doesn't cover observed pooled rate")
        print("  FIX: Adjust Beta(2, 18) to center better on observed ~7.4%")
        print("       Consider Beta(3, 37) for E[μ] = 0.075")

    if not checks['phi_range']:
        print("\n  ISSUE: κ prior too tight, doesn't allow enough overdispersion")
        print("  FIX: Use more dispersed prior like Gamma(1.5, 0.05)")

    if not checks['p_plausible']:
        print("\n  ISSUE: Prior generates implausibly high proportions")
        print("  FIX: Use more informative prior on μ to constrain p_i")

    if not checks['coverage_adequate']:
        print("\n  ISSUE: Poor prior predictive coverage of observed data")
        print("  FIX: Priors may be too informative; consider widening")

    if not checks['no_severe_conflict']:
        print("\n  ISSUE: Severe prior-data conflict detected")
        print("  FIX: Re-examine model structure and prior specifications")

print("\n" + "="*80)
print("Prior predictive check complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*80)

# Save summary to file
summary_file = "/workspace/experiments/experiment_1/prior_predictive_check/summary.txt"
with open(summary_file, 'w') as f:
    f.write("PRIOR PREDICTIVE CHECK SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Decision: {decision}\n\n")
    f.write("Checks:\n")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        f.write(f"  [{status}] {check}\n")
    f.write(f"\nCoverage: {coverage_count}/{n_groups} groups within 95% CI\n")
    f.write(f"Extreme tails: {extreme_percentiles}/{n_groups} groups\n")

print(f"\nSummary saved to: {summary_file}")
