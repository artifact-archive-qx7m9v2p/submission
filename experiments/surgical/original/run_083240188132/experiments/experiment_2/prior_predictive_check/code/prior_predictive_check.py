"""
Prior Predictive Check for Experiment 2: Random Effects Logistic Regression

Validates that priors generate scientifically plausible data before model fitting.

Model:
    r_i | θ_i, n_i ~ Binomial(n_i, logit⁻¹(θ_i))
    θ_i = μ + τ · z_i
    z_i ~ Normal(0, 1)
    μ ~ Normal(logit(0.075), 1²)
    τ ~ HalfNormal(1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import logit, expit

# Configuration
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = "/workspace/data/data.csv"
PLOT_DIR = "/workspace/experiments/experiment_2/prior_predictive_check/plots"

# Load data
print("="*80)
print("PRIOR PREDICTIVE CHECK: Random Effects Logistic Regression")
print("="*80)
print("\nLoading data...")
data = pd.read_csv(DATA_PATH)
print(f"Data shape: {data.shape}")
print(f"\nObserved data summary:")
print(data[['group', 'n', 'r', 'proportion']].to_string(index=False))

# Extract observed values
n_groups = len(data)
n_obs = data['n'].values
r_obs = data['r'].values
p_obs = data['proportion'].values

print(f"\nPooled proportion: {r_obs.sum() / n_obs.sum():.4f}")
print(f"Proportion range: [{p_obs.min():.4f}, {p_obs.max():.4f}]")
print(f"Zero counts: Group {data[data['r'] == 0]['group'].values}")

# Prior predictive simulation
print("\n" + "="*80)
print("PRIOR PREDICTIVE SIMULATION")
print("="*80)

n_prior_samples = 1000
print(f"\nGenerating {n_prior_samples} prior predictive samples...")

# Prior specifications
mu_mean = logit(0.075)  # ≈ -2.52
mu_sd = 1.0
tau_sd = 1.0

print(f"\nPrior specifications:")
print(f"  μ ~ Normal({mu_mean:.2f}, {mu_sd}²)")
print(f"  τ ~ HalfNormal({tau_sd})")
print(f"  θ_i = μ + τ·z_i, z_i ~ N(0,1)")

# Sample from priors
np.random.seed(42)
mu_samples = np.random.normal(mu_mean, mu_sd, n_prior_samples)
tau_samples = np.abs(np.random.normal(0, tau_sd, n_prior_samples))  # Half-normal

# For each prior sample, generate group-level parameters and data
theta_samples = np.zeros((n_prior_samples, n_groups))
p_samples = np.zeros((n_prior_samples, n_groups))
r_samples = np.zeros((n_prior_samples, n_groups), dtype=int)

for i in range(n_prior_samples):
    # Sample group-level offsets
    z_i = np.random.normal(0, 1, n_groups)

    # Compute group-level log-odds
    theta_samples[i] = mu_samples[i] + tau_samples[i] * z_i

    # Convert to probabilities
    p_samples[i] = expit(theta_samples[i])

    # Generate binomial counts
    for j in range(n_groups):
        r_samples[i, j] = np.random.binomial(n_obs[j], p_samples[i, j])

print(f"\nPrior samples generated successfully!")
print(f"  μ range: [{mu_samples.min():.2f}, {mu_samples.max():.2f}]")
print(f"  τ range: [{tau_samples.min():.2f}, {tau_samples.max():.2f}]")
print(f"  p range: [{p_samples.min():.4f}, {p_samples.max():.4f}]")

# Check for computational issues
n_invalid_p = np.sum((p_samples < 0) | (p_samples > 1))
print(f"\n  Invalid probabilities (p<0 or p>1): {n_invalid_p}/{n_prior_samples * n_groups}")

# Derived quantities
print("\n" + "="*80)
print("PRIOR DISTRIBUTIONS OF KEY QUANTITIES")
print("="*80)

# Convert μ back to probability scale for interpretation
p_mu = expit(mu_samples)
print(f"\nGlobal mean proportion (p = logit⁻¹(μ)):")
print(f"  Mean: {p_mu.mean():.4f}")
print(f"  Median: {np.median(p_mu):.4f}")
print(f"  95% CI: [{np.percentile(p_mu, 2.5):.4f}, {np.percentile(p_mu, 97.5):.4f}]")

print(f"\nBetween-group SD (τ):")
print(f"  Mean: {tau_samples.mean():.3f}")
print(f"  Median: {np.median(tau_samples):.3f}")
print(f"  95% CI: [{np.percentile(tau_samples, 2.5):.3f}, {np.percentile(tau_samples, 97.5):.3f}]")

# Between-group variance on probability scale
# Approximate: for logit link, var(p) ≈ τ² * p²(1-p)²
var_p_approx = tau_samples**2 * p_mu**2 * (1-p_mu)**2
print(f"\nApproximate between-group variance (probability scale):")
print(f"  Mean: {var_p_approx.mean():.6f}")
print(f"  95% CI: [{np.percentile(var_p_approx, 2.5):.6f}, {np.percentile(var_p_approx, 97.5):.6f}]")

# Prior predictive checks
print("\n" + "="*80)
print("PRIOR PREDICTIVE CHECKS")
print("="*80)

print("\n1. Group-level proportions:")
print(f"   Observed range: [{p_obs.min():.4f}, {p_obs.max():.4f}]")
print(f"   Prior pred mean: {p_samples.mean():.4f}")
print(f"   Prior pred range: [{p_samples.min():.4f}, {p_samples.max():.4f}]")
print(f"   Prior pred 95% CI: [{np.percentile(p_samples, 2.5):.4f}, {np.percentile(p_samples, 97.5):.4f}]")

print("\n2. Group 1 (n=47, r=0) - special attention:")
group1_r = r_samples[:, 0]
print(f"   Prior pred count range: [{group1_r.min()}, {group1_r.max()}]")
print(f"   P(r=0 | prior): {(group1_r == 0).mean():.3f}")
print(f"   P(r≤2 | prior): {(group1_r <= 2).mean():.3f}")

print("\n3. Coverage of observed data:")
for j in range(n_groups):
    obs_in_range = np.sum((r_samples[:, j] <= r_obs[j] + 5) &
                          (r_samples[:, j] >= r_obs[j] - 5))
    print(f"   Group {j+1} (n={n_obs[j]}, r={r_obs[j]}): "
          f"{obs_in_range}/{n_prior_samples} samples within ±5 counts")

print("\n4. Extreme values check:")
p_very_low = (p_samples < 0.001).sum()
p_very_high = (p_samples > 0.5).sum()
print(f"   Proportions < 0.1%: {p_very_low}/{n_prior_samples * n_groups}")
print(f"   Proportions > 50%: {p_very_high}/{n_prior_samples * n_groups}")

# Summary statistics for reporting
print("\n" + "="*80)
print("SUMMARY FOR DECISION")
print("="*80)

checks_passed = []
checks_failed = []

# Check 1: No invalid probabilities
if n_invalid_p == 0:
    checks_passed.append("No invalid probabilities (all p in [0,1])")
else:
    checks_failed.append(f"Found {n_invalid_p} invalid probabilities")

# Check 2: Prior predictive covers observed range
if p_samples.min() < p_obs.min() and p_samples.max() > p_obs.max():
    checks_passed.append("Prior predictive covers observed proportion range")
else:
    checks_failed.append("Prior predictive does not adequately cover observed range")

# Check 3: Group 1 can generate zero counts
if (group1_r == 0).mean() > 0.01:  # At least 1% of samples
    checks_passed.append(f"Group 1 generates r=0 with P={(group1_r == 0).mean():.3f}")
else:
    checks_failed.append(f"Group 1 rarely generates r=0 (P={(group1_r == 0).mean():.3f})")

# Check 4: τ allows sufficient heterogeneity
# Observed ICC = 0.66, which is high. τ should allow values > 0.5
if np.percentile(tau_samples, 97.5) > 1.0:
    checks_passed.append(f"τ prior allows high heterogeneity (95% upper bound: {np.percentile(tau_samples, 97.5):.2f})")
else:
    checks_failed.append(f"τ prior may be too restrictive")

# Check 5: μ prior centers near observed pooled rate
if np.percentile(p_mu, 2.5) < 0.074 < np.percentile(p_mu, 97.5):
    checks_passed.append("μ prior covers observed pooled rate")
else:
    checks_failed.append("μ prior does not adequately cover observed pooled rate")

print("\nPASSED CHECKS:")
for check in checks_passed:
    print(f"  + {check}")

print("\nFAILED CHECKS:")
if checks_failed:
    for check in checks_failed:
        print(f"  - {check}")
else:
    print("  (none)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Prior distributions for μ and τ
print("\n1. Prior distributions of parameters...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# μ on logit scale
ax = axes[0, 0]
ax.hist(mu_samples, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax.axvline(mu_mean, color='red', linestyle='--', linewidth=2, label=f'Prior mean: {mu_mean:.2f}')
ax.axvline(logit(0.074), color='green', linestyle='--', linewidth=2, label=f'Observed pooled: {logit(0.074):.2f}')
ax.set_xlabel('μ (log-odds scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution: Global Mean (μ)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# μ on probability scale
ax = axes[0, 1]
ax.hist(p_mu, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax.axvline(expit(mu_mean), color='red', linestyle='--', linewidth=2, label=f'E[p]: {expit(mu_mean):.3f}')
ax.axvline(0.074, color='green', linestyle='--', linewidth=2, label=f'Observed: 0.074')
ax.set_xlabel('p = logit⁻¹(μ)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution: Global Mean (probability scale)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# τ (between-group SD)
ax = axes[1, 0]
ax.hist(tau_samples, bins=50, alpha=0.7, color='coral', edgecolor='black', density=True)
# Overlay theoretical HalfNormal(1) density
x_tau = np.linspace(0, tau_samples.max(), 200)
y_tau = stats.halfnorm.pdf(x_tau, scale=tau_sd)
ax.plot(x_tau, y_tau, 'r-', linewidth=2, label='HalfNormal(1)')
ax.set_xlabel('τ (between-group SD, log-odds scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution: Between-Group SD (τ)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Between-group variance on probability scale
ax = axes[1, 1]
ax.hist(var_p_approx, bins=50, alpha=0.7, color='mediumpurple', edgecolor='black', density=True)
ax.set_xlabel('Var(p) ≈ τ² · p²(1-p)²', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution: Between-Group Variance (probability scale)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/parameter_plausibility.png", dpi=300, bbox_inches='tight')
print(f"   Saved: parameter_plausibility.png")
plt.close()

# Plot 2: Prior predictive distributions of group proportions
print("\n2. Prior predictive distributions of group proportions...")
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Violin plots for each group
positions = np.arange(1, n_groups + 1)
parts = ax.violinplot([p_samples[:, j] for j in range(n_groups)],
                       positions=positions,
                       showmeans=False,
                       showmedians=True,
                       widths=0.7)

for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')

# Overlay observed proportions
ax.scatter(positions, p_obs, color='red', s=100, zorder=5,
           label='Observed proportions', marker='D', edgecolors='darkred', linewidths=2)

# Add sample sizes as text
for j in range(n_groups):
    ax.text(j + 1, -0.02, f'n={n_obs[j]}', ha='center', va='top', fontsize=9, color='gray')

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_title('Prior Predictive Distribution: Group Proportions (p_i = logit⁻¹(θ_i))',
             fontsize=13, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels([f'{i}' for i in positions])
ax.legend(loc='upper left', fontsize=11)
ax.grid(alpha=0.3, axis='y')
ax.axhline(0.074, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Pooled rate')
ax.set_ylim(-0.05, max(p_samples.max(), p_obs.max()) * 1.1)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/prior_predictive_proportions.png", dpi=300, bbox_inches='tight')
print(f"   Saved: prior_predictive_proportions.png")
plt.close()

# Plot 3: Prior predictive distributions of counts (with observed overlay)
print("\n3. Prior predictive distributions of counts...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for j in range(n_groups):
    ax = axes[j]

    # Histogram of prior predictive counts
    counts, bins, patches = ax.hist(r_samples[:, j], bins=30, alpha=0.7,
                                     color='lightblue', edgecolor='black', density=True)

    # Overlay observed count
    ax.axvline(r_obs[j], color='red', linestyle='--', linewidth=2.5,
               label=f'Observed: {r_obs[j]}')

    # Add statistics
    prior_mean = r_samples[:, j].mean()
    prior_ci = np.percentile(r_samples[:, j], [2.5, 97.5])

    ax.set_xlabel('Count', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Group {j+1} (n={n_obs[j]})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add text box with statistics
    textstr = f'Prior mean: {prior_mean:.1f}\n95% CI: [{prior_ci[0]:.0f}, {prior_ci[1]:.0f}]'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Prior Predictive Distribution: Group Counts (r_i)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/prior_predictive_counts.png", dpi=300, bbox_inches='tight')
print(f"   Saved: prior_predictive_counts.png")
plt.close()

# Plot 4: Diagnostic for Group 1 (zero inflation check)
print("\n4. Group 1 diagnostic (zero count check)...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Group 1 count distribution
ax = axes[0]
unique_counts, count_freq = np.unique(group1_r, return_counts=True)
count_prob = count_freq / n_prior_samples
ax.bar(unique_counts, count_prob, alpha=0.7, color='steelblue', edgecolor='black', width=0.8)
ax.axvline(r_obs[0], color='red', linestyle='--', linewidth=3, label=f'Observed: {r_obs[0]}')
ax.set_xlabel('Count (r)', fontsize=11)
ax.set_ylabel('Prior Predictive Probability', fontsize=11)
ax.set_title('Group 1: Prior Predictive Count Distribution (n=47)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add text with key probabilities
textstr = f'P(r=0) = {(group1_r == 0).mean():.3f}\n'
textstr += f'P(r≤2) = {(group1_r <= 2).mean():.3f}\n'
textstr += f'P(r≤5) = {(group1_r <= 5).mean():.3f}'
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Group 1 proportion distribution
ax = axes[1]
ax.hist(p_samples[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax.axvline(p_obs[0], color='red', linestyle='--', linewidth=3, label=f'Observed: {p_obs[0]:.3f}')
ax.set_xlabel('Proportion (p)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Group 1: Prior Predictive Proportion Distribution',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/group1_zero_inflation_diagnostic.png", dpi=300, bbox_inches='tight')
print(f"   Saved: group1_zero_inflation_diagnostic.png")
plt.close()

# Plot 5: Prior predictive coverage assessment
print("\n5. Prior predictive coverage assessment...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# For each group, compute quantiles of prior predictive distribution
quantiles = [2.5, 25, 50, 75, 97.5]
prior_quantiles = np.percentile(r_samples, quantiles, axis=0)

# Plot prior predictive intervals
positions = np.arange(1, n_groups + 1)

# 95% interval
ax.fill_between(positions, prior_quantiles[0], prior_quantiles[4],
                alpha=0.3, color='lightblue', label='95% Prior Predictive Interval')

# 50% interval
ax.fill_between(positions, prior_quantiles[1], prior_quantiles[3],
                alpha=0.5, color='steelblue', label='50% Prior Predictive Interval')

# Median
ax.plot(positions, prior_quantiles[2], 'b-', linewidth=2, label='Prior Predictive Median')

# Observed counts
ax.scatter(positions, r_obs, color='red', s=120, zorder=5,
           label='Observed counts', marker='D', edgecolors='darkred', linewidths=2)

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prior Predictive Coverage: Observed vs. Prior Predictive Intervals',
             fontsize=13, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels([f'{i}' for i in positions])
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add sample sizes as text
for j in range(n_groups):
    ax.text(j + 1, -5, f'n={n_obs[j]}', ha='center', va='top', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/prior_predictive_coverage.png", dpi=300, bbox_inches='tight')
print(f"   Saved: prior_predictive_coverage.png")
plt.close()

print("\n" + "="*80)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("="*80)
print(f"\nAll outputs saved to:")
print(f"  Code: /workspace/experiments/experiment_2/prior_predictive_check/code/")
print(f"  Plots: /workspace/experiments/experiment_2/prior_predictive_check/plots/")
print(f"\nNext step: Review findings.md for GO/NO-GO decision")
