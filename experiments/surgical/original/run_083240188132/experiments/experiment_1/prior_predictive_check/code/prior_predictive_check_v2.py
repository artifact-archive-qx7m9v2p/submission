"""
Prior Predictive Check for Experiment 1 - REVISED PRIORS (Version 2)
=====================================================================

Model: Beta-Binomial Hierarchical Model

Revised Priors:
- μ ~ Beta(2, 18)        [E[μ] = 0.1, unchanged from v1]
- κ ~ Gamma(1.5, 0.5)    [E[κ] = 3, REVISED from Gamma(2, 0.1)]

Change rationale: Original κ prior implied φ ≈ 1.05 (too low overdispersion)
                  New κ prior allows φ ≈ 1.33-6 to cover observed φ ≈ 3.5-5.1

This script:
1. Samples from revised priors (1000 draws)
2. Generates synthetic data from prior predictive distribution
3. Compares to observed data to assess plausibility
4. Diagnoses prior-data conflicts and computational issues
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
N_PRIOR_SAMPLES = 1000
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check"

# Load observed data
data = pd.read_csv("/workspace/data/data.csv")
n_groups = len(data)
n_obs = data['n'].values
r_obs = data['r'].values
p_obs = data['proportion'].values

# Observed statistics
pooled_rate = r_obs.sum() / n_obs.sum()
observed_phi_range = (3.5, 5.1)  # From metadata
observed_icc = 0.66  # From metadata

print("="*70)
print("PRIOR PREDICTIVE CHECK - REVISED PRIORS (Version 2)")
print("="*70)
print("\nOBSERVED DATA SUMMARY:")
print(f"  Number of groups: {n_groups}")
print(f"  Sample sizes: {n_obs.min()} to {n_obs.max()}")
print(f"  Observed counts: {r_obs.min()} to {r_obs.max()}")
print(f"  Observed proportions: {p_obs.min():.1%} to {p_obs.max():.1%}")
print(f"  Pooled rate: {pooled_rate:.1%}")
print(f"  Overdispersion (φ): {observed_phi_range[0]:.1f} to {observed_phi_range[1]:.1f}")
print(f"  ICC: {observed_icc:.2f}")

print("\n" + "="*70)
print("REVISED PRIOR SPECIFICATION:")
print("="*70)
print("  μ ~ Beta(2, 18)")
print("    E[μ] = 0.1, SD[μ] = 0.065")
print("    Median[μ] ≈ 0.091")
print("\n  κ ~ Gamma(1.5, 0.5)  ← REVISED")
print("    E[κ] = 3, SD[κ] = 2.45")
print("    Allows much wider range than v1")
print("\n  Implied φ = 1 + 1/κ:")
print("    If κ=0.25: φ=5")
print("    If κ=1: φ=2")
print("    If κ=3: φ=1.33")
print("    Target range: [2, 10] to cover observed [3.5, 5.1]")

# ============================================================================
# STEP 1: Sample from priors
# ============================================================================
print("\n" + "="*70)
print("STEP 1: SAMPLING FROM REVISED PRIORS")
print("="*70)

# Prior samples
mu_prior = np.random.beta(2, 18, N_PRIOR_SAMPLES)
kappa_prior = np.random.gamma(1.5, 1/0.5, N_PRIOR_SAMPLES)  # scipy uses scale = 1/rate
phi_prior = 1 + 1/kappa_prior

print(f"\nPrior samples drawn: {N_PRIOR_SAMPLES}")
print(f"\nμ prior statistics:")
print(f"  Mean: {mu_prior.mean():.3f}")
print(f"  SD: {mu_prior.std():.3f}")
print(f"  Median: {np.median(mu_prior):.3f}")
print(f"  5th-95th percentile: [{np.percentile(mu_prior, 5):.3f}, {np.percentile(mu_prior, 95):.3f}]")

print(f"\nκ prior statistics (REVISED):")
print(f"  Mean: {kappa_prior.mean():.3f}")
print(f"  SD: {kappa_prior.std():.3f}")
print(f"  Median: {np.median(kappa_prior):.3f}")
print(f"  5th-95th percentile: [{np.percentile(kappa_prior, 5):.3f}, {np.percentile(kappa_prior, 95):.3f}]")
print(f"  Min: {kappa_prior.min():.3f}, Max: {kappa_prior.max():.3f}")

print(f"\nφ = 1 + 1/κ statistics (KEY DIAGNOSTIC):")
print(f"  Mean: {phi_prior.mean():.3f}")
print(f"  SD: {phi_prior.std():.3f}")
print(f"  Median: {np.median(phi_prior):.3f}")
print(f"  5th-95th percentile: [{np.percentile(phi_prior, 5):.3f}, {np.percentile(phi_prior, 95):.3f}]")
print(f"  Min: {phi_prior.min():.3f}, Max: {phi_prior.max():.3f}")
print(f"\n  ✓ CRITICAL CHECK: Does prior φ cover observed [{observed_phi_range[0]}, {observed_phi_range[1]}]?")
phi_coverage = (phi_prior.min() <= observed_phi_range[0]) and (phi_prior.max() >= observed_phi_range[1])
phi_percentile_coverage = np.sum((phi_prior >= observed_phi_range[0]) & (phi_prior <= observed_phi_range[1])) / N_PRIOR_SAMPLES
print(f"    Range coverage: {phi_coverage}")
print(f"    Prior mass in observed range: {phi_percentile_coverage:.1%}")

# ============================================================================
# STEP 2: Generate prior predictive data
# ============================================================================
print("\n" + "="*70)
print("STEP 2: GENERATING PRIOR PREDICTIVE DATA")
print("="*70)

# For each prior sample, generate group-level proportions and counts
p_prior_samples = np.zeros((N_PRIOR_SAMPLES, n_groups))
r_prior_samples = np.zeros((N_PRIOR_SAMPLES, n_groups), dtype=int)

for i in range(N_PRIOR_SAMPLES):
    mu = mu_prior[i]
    kappa = kappa_prior[i]

    # Beta parameters
    alpha = mu * kappa
    beta_param = (1 - mu) * kappa

    # Sample group proportions from Beta(α, β)
    p_i = np.random.beta(alpha, beta_param, n_groups)
    p_prior_samples[i, :] = p_i

    # Sample counts from Binomial(n_i, p_i)
    for j in range(n_groups):
        r_prior_samples[i, j] = np.random.binomial(n_obs[j], p_i[j])

print(f"\nGenerated {N_PRIOR_SAMPLES} prior predictive datasets")
print(f"Each dataset: {n_groups} groups with n = {n_obs}")

# ============================================================================
# STEP 3: Assess plausibility
# ============================================================================
print("\n" + "="*70)
print("STEP 3: PLAUSIBILITY ASSESSMENT")
print("="*70)

# 3.1 Check group proportions
print("\n[3.1] GROUP PROPORTIONS (p_i)")
p_min = p_prior_samples.min(axis=1)
p_max = p_prior_samples.max(axis=1)
p_mean = p_prior_samples.mean(axis=1)

print(f"\nMin proportion across groups (per simulation):")
print(f"  Mean: {p_min.mean():.3f}")
print(f"  5th-95th percentile: [{np.percentile(p_min, 5):.3f}, {np.percentile(p_min, 95):.3f}]")

print(f"\nMax proportion across groups (per simulation):")
print(f"  Mean: {p_max.mean():.3f}")
print(f"  5th-95th percentile: [{np.percentile(p_max, 5):.3f}, {np.percentile(p_max, 95):.3f}]")

print(f"\nProportion range (max - min per simulation):")
range_p = p_max - p_min
print(f"  Mean: {range_p.mean():.3f}")
print(f"  5th-95th percentile: [{np.percentile(range_p, 5):.3f}, {np.percentile(range_p, 95):.3f}]")

print(f"\n  ✓ CHECK: Observed proportions [{p_obs.min():.3f}, {p_obs.max():.3f}]")
print(f"    Observed range: {p_obs.max() - p_obs.min():.3f}")
p_range_obs = p_obs.max() - p_obs.min()
p_range_percentile = np.sum(range_p >= p_range_obs) / N_PRIOR_SAMPLES
print(f"    Prior predictive mass with range ≥ observed: {p_range_percentile:.1%}")

# 3.2 Check counts
print("\n[3.2] COUNTS (r_i)")
r_min = r_prior_samples.min(axis=1)
r_max = r_prior_samples.max(axis=1)

print(f"\nMin count across groups (per simulation):")
print(f"  Mean: {r_min.mean():.1f}")
print(f"  5th-95th percentile: [{np.percentile(r_min, 5):.1f}, {np.percentile(r_min, 95):.1f}]")

print(f"\nMax count across groups (per simulation):")
print(f"  Mean: {r_max.mean():.1f}")
print(f"  5th-95th percentile: [{np.percentile(r_max, 5):.1f}, {np.percentile(r_max, 95):.1f}]")

print(f"\n  ✓ CHECK: Observed counts [{r_obs.min()}, {r_obs.max()}]")
obs_in_range = np.sum((r_min <= r_obs.min()) & (r_max >= r_obs.max())) / N_PRIOR_SAMPLES
print(f"    Prior predictive datasets covering observed range: {obs_in_range:.1%}")

# 3.3 Check pooled rates
pooled_rates_prior = r_prior_samples.sum(axis=1) / n_obs.sum()

print("\n[3.3] POOLED RATES")
print(f"\nPrior predictive pooled rate:")
print(f"  Mean: {pooled_rates_prior.mean():.3f}")
print(f"  SD: {pooled_rates_prior.std():.3f}")
print(f"  5th-95th percentile: [{np.percentile(pooled_rates_prior, 5):.3f}, {np.percentile(pooled_rates_prior, 95):.3f}]")
print(f"\n  ✓ CHECK: Observed pooled rate = {pooled_rate:.3f}")
pooled_percentile = np.sum(pooled_rates_prior <= pooled_rate) / N_PRIOR_SAMPLES
print(f"    Percentile in prior predictive: {pooled_percentile:.1%}")

# 3.4 Check for computational issues
print("\n[3.4] COMPUTATIONAL DIAGNOSTICS")

# Check for extreme values
extreme_kappa = np.sum(kappa_prior > 100)
extreme_phi = np.sum(phi_prior > 100)
zero_p = np.sum(p_prior_samples == 0)
one_p = np.sum(p_prior_samples == 1)

print(f"\nExtreme values:")
print(f"  κ > 100: {extreme_kappa} / {N_PRIOR_SAMPLES} ({extreme_kappa/N_PRIOR_SAMPLES:.1%})")
print(f"  φ > 100: {extreme_phi} / {N_PRIOR_SAMPLES} ({extreme_phi/N_PRIOR_SAMPLES:.1%})")
print(f"  p_i = 0: {zero_p} / {N_PRIOR_SAMPLES * n_groups} ({zero_p/(N_PRIOR_SAMPLES*n_groups):.1%})")
print(f"  p_i = 1: {one_p} / {N_PRIOR_SAMPLES * n_groups} ({one_p/(N_PRIOR_SAMPLES*n_groups):.1%})")

if extreme_kappa > 0 or extreme_phi > 0 or zero_p > N_PRIOR_SAMPLES or one_p > N_PRIOR_SAMPLES:
    print("  ⚠ Some extreme values detected, but likely acceptable if rare")
else:
    print("  ✓ No extreme values detected")

# 3.5 Overdispersion check
print("\n[3.5] OVERDISPERSION ASSESSMENT")
print(f"\n  Target: φ ∈ [{observed_phi_range[0]}, {observed_phi_range[1]}]")
print(f"  Prior φ range: [{phi_prior.min():.2f}, {phi_prior.max():.2f}]")
print(f"  Prior φ 90% interval: [{np.percentile(phi_prior, 5):.2f}, {np.percentile(phi_prior, 95):.2f}]")

# What percentage of prior samples have φ in reasonable range [2, 10]?
reasonable_phi = np.sum((phi_prior >= 2) & (phi_prior <= 10)) / N_PRIOR_SAMPLES
print(f"\n  Prior mass with φ ∈ [2, 10]: {reasonable_phi:.1%}")

# What percentage have φ in observed range?
observed_phi = np.sum((phi_prior >= observed_phi_range[0]) & (phi_prior <= observed_phi_range[1])) / N_PRIOR_SAMPLES
print(f"  Prior mass with φ ∈ [{observed_phi_range[0]}, {observed_phi_range[1]}]: {observed_phi:.1%}")

# ============================================================================
# STEP 4: Create visualizations
# ============================================================================
print("\n" + "="*70)
print("STEP 4: CREATING VISUALIZATIONS")
print("="*70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Plot 1: Parameter plausibility - focus on revised κ and φ
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# μ prior (unchanged)
ax = axes[0, 0]
ax.hist(mu_prior, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax.axvline(mu_prior.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {mu_prior.mean():.3f}')
ax.axvline(pooled_rate, color='orange', linestyle='--', linewidth=2, label=f'Observed = {pooled_rate:.3f}')
ax.set_xlabel('μ (population mean)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: μ ~ Beta(2, 18) [unchanged]', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# κ prior (REVISED)
ax = axes[0, 1]
ax.hist(kappa_prior, bins=50, alpha=0.7, color='darkgreen', edgecolor='black', density=True)
ax.axvline(kappa_prior.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {kappa_prior.mean():.2f}')
ax.axvline(np.median(kappa_prior), color='purple', linestyle='--', linewidth=2, label=f'Median = {np.median(kappa_prior):.2f}')
ax.set_xlabel('κ (concentration)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior: κ ~ Gamma(1.5, 0.5) [REVISED]', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# φ prior (implied by revised κ)
ax = axes[1, 0]
ax.hist(phi_prior, bins=60, alpha=0.7, color='crimson', edgecolor='black', density=True)
ax.axvline(phi_prior.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean = {phi_prior.mean():.2f}')
ax.axvline(np.median(phi_prior), color='purple', linestyle='--', linewidth=2, label=f'Median = {np.median(phi_prior):.2f}')
# Add observed range
ax.axvspan(observed_phi_range[0], observed_phi_range[1], alpha=0.2, color='orange', label='Observed φ range')
ax.set_xlabel('φ = 1 + 1/κ (overdispersion)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Implied Prior: φ = 1 + 1/κ [KEY DIAGNOSTIC]', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, min(20, phi_prior.max()))

# Joint μ-κ
ax = axes[1, 1]
scatter = ax.scatter(mu_prior, kappa_prior, alpha=0.3, s=10, c=phi_prior, cmap='viridis')
ax.axhline(kappa_prior.mean(), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(mu_prior.mean(), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('μ', fontsize=11)
ax.set_ylabel('κ', fontsize=11)
ax.set_title('Joint Prior: μ vs κ (colored by φ)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('φ', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/v2_parameter_plausibility.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: v2_parameter_plausibility.png")
plt.close()

# Plot 2: Prior predictive coverage - comparing simulated vs observed
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 2.1 Group proportions
ax = axes[0, 0]
for i in range(min(100, N_PRIOR_SAMPLES)):  # Plot subset for clarity
    ax.plot(range(1, n_groups+1), p_prior_samples[i, :], alpha=0.05, color='gray', linewidth=0.5)
ax.plot(range(1, n_groups+1), p_obs, 'ro-', linewidth=2, markersize=8, label='Observed', zorder=100)
ax.fill_between(range(1, n_groups+1),
                 np.percentile(p_prior_samples, 5, axis=0),
                 np.percentile(p_prior_samples, 95, axis=0),
                 alpha=0.3, color='steelblue', label='90% prior pred interval')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Prior Predictive: Group Proportions', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2.2 Counts
ax = axes[0, 1]
for i in range(min(100, N_PRIOR_SAMPLES)):
    ax.plot(range(1, n_groups+1), r_prior_samples[i, :], alpha=0.05, color='gray', linewidth=0.5)
ax.plot(range(1, n_groups+1), r_obs, 'ro-', linewidth=2, markersize=8, label='Observed', zorder=100)
ax.fill_between(range(1, n_groups+1),
                 np.percentile(r_prior_samples, 5, axis=0),
                 np.percentile(r_prior_samples, 95, axis=0),
                 alpha=0.3, color='steelblue', label='90% prior pred interval')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Prior Predictive: Counts', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2.3 Pooled rate distribution
ax = axes[1, 0]
ax.hist(pooled_rates_prior, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax.axvline(pooled_rate, color='red', linestyle='--', linewidth=2, label=f'Observed = {pooled_rate:.3f}')
ax.axvline(pooled_rates_prior.mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Prior mean = {pooled_rates_prior.mean():.3f}')
ax.set_xlabel('Pooled rate', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive: Pooled Rate', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2.4 Range of proportions across groups
ax = axes[1, 1]
ax.hist(range_p, bins=50, alpha=0.7, color='darkgreen', edgecolor='black', density=True)
ax.axvline(p_range_obs, color='red', linestyle='--', linewidth=2, label=f'Observed = {p_range_obs:.3f}')
ax.axvline(range_p.mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Prior mean = {range_p.mean():.3f}')
ax.set_xlabel('Range (max - min proportion)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Predictive: Between-Group Variability', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/v2_prior_predictive_coverage.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: v2_prior_predictive_coverage.png")
plt.close()

# Plot 3: Overdispersion diagnostic (focused on φ)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3.1 κ distribution with key quantiles
ax = axes[0]
ax.hist(kappa_prior, bins=50, alpha=0.7, color='darkgreen', edgecolor='black', density=True)
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
for q in quantiles:
    val = np.percentile(kappa_prior, q*100)
    ax.axvline(val, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(val, ax.get_ylim()[1]*0.95, f'P{int(q*100)}={val:.2f}',
            rotation=90, va='top', fontsize=8)
ax.set_xlabel('κ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('κ ~ Gamma(1.5, 0.5): Quantiles', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# 3.2 φ distribution with observed range
ax = axes[1]
ax.hist(phi_prior, bins=60, alpha=0.7, color='crimson', edgecolor='black', density=True)
ax.axvspan(observed_phi_range[0], observed_phi_range[1], alpha=0.25, color='orange',
           label=f'Observed φ ∈ [{observed_phi_range[0]}, {observed_phi_range[1]}]')
ax.axvspan(2, 10, alpha=0.1, color='blue', label='Reasonable range [2, 10]')
for q in quantiles:
    val = np.percentile(phi_prior, q*100)
    if val <= 20:  # Only show if in reasonable range
        ax.axvline(val, color='blue', linestyle='--', alpha=0.6, linewidth=1)
ax.set_xlabel('φ = 1 + 1/κ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('φ Distribution vs Observed Range', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(1, min(20, np.percentile(phi_prior, 99)))

# 3.3 κ vs φ relationship
ax = axes[2]
kappa_range = np.linspace(0.1, 10, 100)
phi_theoretical = 1 + 1/kappa_range
ax.plot(kappa_range, phi_theoretical, 'b-', linewidth=2, label='φ = 1 + 1/κ')
ax.scatter(kappa_prior, phi_prior, alpha=0.3, s=10, color='steelblue')
ax.axhspan(observed_phi_range[0], observed_phi_range[1], alpha=0.2, color='orange',
           label='Observed φ range')
ax.set_xlabel('κ', fontsize=11)
ax.set_ylabel('φ', fontsize=11)
ax.set_title('κ-φ Relationship', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, min(10, np.percentile(kappa_prior, 99)))
ax.set_ylim(1, min(20, np.percentile(phi_prior, 99)))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/v2_overdispersion_diagnostic.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: v2_overdispersion_diagnostic.png")
plt.close()

# Plot 4: Comparison v1 vs v2 (show improvement)
print("\n  Creating v1 vs v2 comparison...")
# Generate v1 samples for comparison
kappa_v1 = np.random.gamma(2, 1/0.1, N_PRIOR_SAMPLES)
phi_v1 = 1 + 1/kappa_v1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compare κ distributions
ax = axes[0]
ax.hist(kappa_v1, bins=50, alpha=0.5, color='red', edgecolor='black', density=True,
        label=f'v1: Gamma(2, 0.1)\nE[κ]={kappa_v1.mean():.1f}')
ax.hist(kappa_prior, bins=50, alpha=0.5, color='green', edgecolor='black', density=True,
        label=f'v2: Gamma(1.5, 0.5)\nE[κ]={kappa_prior.mean():.1f}')
ax.set_xlabel('κ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('κ Prior Comparison: v1 vs v2', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 15)

# Compare φ distributions
ax = axes[1]
ax.hist(phi_v1, bins=60, alpha=0.5, color='red', edgecolor='black', density=True,
        label=f'v1: E[φ]={phi_v1.mean():.2f}')
ax.hist(phi_prior, bins=60, alpha=0.5, color='green', edgecolor='black', density=True,
        label=f'v2: E[φ]={phi_prior.mean():.2f}')
ax.axvspan(observed_phi_range[0], observed_phi_range[1], alpha=0.25, color='orange',
           label=f'Observed [{observed_phi_range[0]}, {observed_phi_range[1]}]')
ax.set_xlabel('φ = 1 + 1/κ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('φ Prior Comparison: v1 vs v2 [CRITICAL]', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(1, 15)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/v2_v1_vs_v2_comparison.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: v2_v1_vs_v2_comparison.png")
plt.close()

# ============================================================================
# STEP 5: Save summary statistics
# ============================================================================
print("\n" + "="*70)
print("STEP 5: SAVING RESULTS")
print("="*70)

results = {
    'prior_mu_mean': mu_prior.mean(),
    'prior_mu_sd': mu_prior.std(),
    'prior_kappa_mean': kappa_prior.mean(),
    'prior_kappa_sd': kappa_prior.std(),
    'prior_kappa_median': np.median(kappa_prior),
    'prior_kappa_5th': np.percentile(kappa_prior, 5),
    'prior_kappa_95th': np.percentile(kappa_prior, 95),
    'prior_phi_mean': phi_prior.mean(),
    'prior_phi_sd': phi_prior.std(),
    'prior_phi_median': np.median(phi_prior),
    'prior_phi_5th': np.percentile(phi_prior, 5),
    'prior_phi_95th': np.percentile(phi_prior, 95),
    'prior_phi_min': phi_prior.min(),
    'prior_phi_max': phi_prior.max(),
    'phi_covers_observed': phi_coverage,
    'phi_mass_in_observed_range': phi_percentile_coverage,
    'phi_mass_in_reasonable_range': reasonable_phi,
    'pooled_rate_obs': pooled_rate,
    'pooled_rate_prior_mean': pooled_rates_prior.mean(),
    'pooled_rate_prior_sd': pooled_rates_prior.std(),
    'p_range_obs': p_range_obs,
    'p_range_prior_mean': range_p.mean(),
}

results_df = pd.DataFrame([results])
results_df.to_csv(f"{OUTPUT_DIR}/plots/v2_summary_statistics.csv", index=False)
print(f"  ✓ Saved: v2_summary_statistics.csv")

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================
print("\n" + "="*70)
print("FINAL ASSESSMENT: REVISED PRIORS (VERSION 2)")
print("="*70)

print("\n[CRITICAL QUESTION 1] Does prior φ cover observed range?")
print(f"  Observed φ: [{observed_phi_range[0]}, {observed_phi_range[1]}]")
print(f"  Prior φ range: [{phi_prior.min():.2f}, {phi_prior.max():.2f}]")
print(f"  Coverage: {phi_coverage}")
if phi_coverage:
    print("  ✓ PASS: Prior allows sufficient overdispersion")
else:
    print("  ✗ FAIL: Prior still too restrictive")

print("\n[CRITICAL QUESTION 2] Is prior φ weakly informative?")
print(f"  Prior mass in reasonable range [2, 10]: {reasonable_phi:.1%}")
print(f"  Prior mass in observed range [{observed_phi_range[0]}, {observed_phi_range[1]}]: {observed_phi:.1%}")
if reasonable_phi > 0.6 and observed_phi > 0.1:
    print("  ✓ PASS: Prior is weakly informative, not overly restrictive")
elif reasonable_phi < 0.3:
    print("  ⚠ WARNING: Prior may be too diffuse")
else:
    print("  ✓ ACCEPTABLE: Prior has reasonable spread")

print("\n[CRITICAL QUESTION 3] Does prior predictive cover observed data?")
print(f"  Proportion range - Observed: {p_range_obs:.3f}, Prior mean: {range_p.mean():.3f}")
print(f"  Pooled rate - Observed: {pooled_rate:.3f}, Prior mean: {pooled_rates_prior.mean():.3f}")
print(f"  Prior predictive mass with range ≥ observed: {p_range_percentile:.1%}")
if p_range_percentile > 0.1:
    print("  ✓ PASS: Prior predictive generates plausible variability")
else:
    print("  ⚠ WARNING: Prior may underestimate between-group variability")

print("\n[CRITICAL QUESTION 4] Any computational issues?")
if extreme_kappa == 0 and extreme_phi == 0:
    print("  ✓ PASS: No extreme values detected")
elif extreme_kappa < 10 and extreme_phi < 10:
    print("  ✓ ACCEPTABLE: Very few extreme values")
else:
    print("  ⚠ WARNING: Some extreme values present")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

# Decision logic
pass_phi = phi_coverage and (phi_percentile_coverage > 0.1)
pass_predictive = p_range_percentile > 0.1
pass_computational = extreme_kappa < 50 and extreme_phi < 50

if pass_phi and pass_predictive and pass_computational:
    print("\n  ✓✓✓ PASS - REVISED PRIORS ARE APPROPRIATE ✓✓✓")
    print("\n  The revised κ ~ Gamma(1.5, 0.5) prior successfully addresses")
    print("  the overdispersion issue from v1. The prior φ now covers the")
    print("  observed range [3.5, 5.1] and generates plausible data.")
    print("\n  Proceed to: Simulation-based validation")
elif pass_phi and pass_predictive:
    print("\n  ⚠ CONDITIONAL PASS - Monitor computational issues ⚠")
    print("\n  Priors are scientifically appropriate but watch for numerical issues")
    print("  during MCMC sampling.")
else:
    print("\n  ✗✗✗ FAIL - FURTHER REVISION NEEDED ✗✗✗")
    if not pass_phi:
        print("\n  Issue: Prior φ still doesn't adequately cover observed range")
        print("  Recommendation: Consider κ ~ Gamma(1.2, 0.3) for even wider φ range")
    if not pass_predictive:
        print("\n  Issue: Prior predictive doesn't generate observed variability")
        print("  Recommendation: May need to reconsider hierarchical structure")

print("\n" + "="*70)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("="*70)
