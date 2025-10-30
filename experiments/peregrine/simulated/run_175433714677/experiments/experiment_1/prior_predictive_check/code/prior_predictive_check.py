"""
Prior Predictive Check for Negative Binomial Model
==================================================

Model:
  C[i] ~ NegativeBinomial(μ[i], φ)
  log(μ[i]) = β₀ + β₁ × year[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  φ ~ Exponential(0.667)  # E[φ] = 1.5

This script performs prior predictive checks WITHOUT fitting to data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_DRAWS = 500
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load observed data
with open("/workspace/data/data.csv", "r") as f:
    data = json.load(f)

observed_C = np.array(data["C"])
observed_year = np.array(data["year"])
n_obs = len(observed_C)

print("=" * 70)
print("PRIOR PREDICTIVE CHECK")
print("=" * 70)
print(f"\nObserved data summary:")
print(f"  n = {n_obs}")
print(f"  C range: [{observed_C.min()}, {observed_C.max()}]")
print(f"  C mean: {observed_C.mean():.1f}")
print(f"  C variance: {observed_C.var():.1f}")
print(f"  Var/Mean ratio: {observed_C.var() / observed_C.mean():.1f}")
print(f"  year range: [{observed_year.min():.2f}, {observed_year.max():.2f}]")

# ============================================================================
# STEP 1: Sample from prior distributions
# ============================================================================
print(f"\n{'='*70}")
print(f"STEP 1: Sampling {N_PRIOR_DRAWS} draws from priors")
print("=" * 70)

# Prior: β₀ ~ Normal(4.3, 1.0)
beta_0_prior = np.random.normal(loc=4.3, scale=1.0, size=N_PRIOR_DRAWS)

# Prior: β₁ ~ Normal(0.85, 0.5)
beta_1_prior = np.random.normal(loc=0.85, scale=0.5, size=N_PRIOR_DRAWS)

# Prior: φ ~ Exponential(0.667), where rate = 1/mean, so mean = 1.5
phi_prior = np.random.exponential(scale=1.5, size=N_PRIOR_DRAWS)

print(f"\nPrior samples summary:")
print(f"  β₀: mean={beta_0_prior.mean():.2f}, sd={beta_0_prior.std():.2f}, range=[{beta_0_prior.min():.2f}, {beta_0_prior.max():.2f}]")
print(f"  β₁: mean={beta_1_prior.mean():.2f}, sd={beta_1_prior.std():.2f}, range=[{beta_1_prior.min():.2f}, {beta_1_prior.max():.2f}]")
print(f"  φ:  mean={phi_prior.mean():.2f}, sd={phi_prior.std():.2f}, range=[{phi_prior.min():.2f}, {phi_prior.max():.2f}]")

# ============================================================================
# STEP 2: Generate prior predictive counts
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2: Generating prior predictive counts")
print("=" * 70)

# For each prior draw, generate counts for all observed year values
prior_predictive_counts = np.zeros((N_PRIOR_DRAWS, n_obs))
prior_predictive_mu = np.zeros((N_PRIOR_DRAWS, n_obs))

for i in range(N_PRIOR_DRAWS):
    # Compute mean counts: log(μ) = β₀ + β₁ × year
    log_mu = beta_0_prior[i] + beta_1_prior[i] * observed_year
    mu = np.exp(log_mu)
    prior_predictive_mu[i, :] = mu

    # Generate counts from NegativeBinomial(μ, φ)
    # NumPy parametrization: n=φ, p=φ/(φ+μ)
    # Mean = μ, Variance = μ + μ²/φ
    n = phi_prior[i]
    p = n / (n + mu)

    # Handle edge cases
    p = np.clip(p, 1e-10, 1 - 1e-10)

    prior_predictive_counts[i, :] = np.random.negative_binomial(n, p)

# Compute statistics
print(f"\nPrior predictive μ (mean parameter):")
print(f"  Range across all predictions: [{prior_predictive_mu.min():.1f}, {prior_predictive_mu.max():.1f}]")
print(f"  Median μ at earliest year: {np.median(prior_predictive_mu[:, 0]):.1f}")
print(f"  Median μ at latest year: {np.median(prior_predictive_mu[:, -1]):.1f}")

print(f"\nPrior predictive counts:")
print(f"  Range: [{prior_predictive_counts.min():.0f}, {prior_predictive_counts.max():.0f}]")
print(f"  Median: {np.median(prior_predictive_counts):.1f}")
print(f"  Mean: {prior_predictive_counts.mean():.1f}")
print(f"  95% interval: [{np.percentile(prior_predictive_counts, 2.5):.0f}, {np.percentile(prior_predictive_counts, 97.5):.0f}]")

# ============================================================================
# STEP 3: Diagnostic checks
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3: Diagnostic checks")
print("=" * 70)

# Check 1: Count range coverage
min_pred = prior_predictive_counts.min(axis=1)
max_pred = prior_predictive_counts.max(axis=1)
covers_min = (min_pred <= observed_C.min()).sum() / N_PRIOR_DRAWS
covers_max = (max_pred >= observed_C.max()).sum() / N_PRIOR_DRAWS

print(f"\n1. COUNT RANGE COVERAGE:")
print(f"   Observed range: [{observed_C.min()}, {observed_C.max()}]")
print(f"   Proportion of prior draws that cover minimum: {covers_min:.1%}")
print(f"   Proportion of prior draws that cover maximum: {covers_max:.1%}")

# Check 2: Extreme values
extreme_threshold = 10000
n_extreme = (prior_predictive_counts > extreme_threshold).any(axis=1).sum()
print(f"\n2. EXTREME VALUES:")
print(f"   Predictions exceeding {extreme_threshold}: {n_extreme}/{N_PRIOR_DRAWS} ({n_extreme/N_PRIOR_DRAWS:.1%})")

# Check 3: Zero inflation
zero_counts = (prior_predictive_counts == 0).sum(axis=1)
mean_zeros_per_draw = zero_counts.mean()
print(f"\n3. ZERO INFLATION:")
print(f"   Observed zeros: 0/{n_obs}")
print(f"   Mean zeros per prior predictive dataset: {mean_zeros_per_draw:.1f}")
print(f"   Proportion of draws with >5 zeros: {(zero_counts > 5).sum()/N_PRIOR_DRAWS:.1%}")

# Check 4: Var/Mean ratio
var_mean_ratios = np.var(prior_predictive_counts, axis=1) / np.mean(prior_predictive_counts, axis=1)
observed_var_mean = observed_C.var() / observed_C.mean()

print(f"\n4. VAR/MEAN RATIO:")
print(f"   Observed Var/Mean: {observed_var_mean:.1f}")
print(f"   Prior predictive Var/Mean:")
print(f"     Mean: {var_mean_ratios.mean():.1f}")
print(f"     Median: {np.median(var_mean_ratios):.1f}")
print(f"     95% interval: [{np.percentile(var_mean_ratios, 2.5):.1f}, {np.percentile(var_mean_ratios, 97.5):.1f}]")
print(f"   Proportion in plausible range [20, 200]: {((var_mean_ratios >= 20) & (var_mean_ratios <= 200)).sum()/N_PRIOR_DRAWS:.1%}")

# Check 5: Growth patterns
# For each prior draw, compute the ratio of mean counts (last vs first 10 obs)
growth_ratios = []
for i in range(N_PRIOR_DRAWS):
    early_mean = prior_predictive_counts[i, :10].mean()
    late_mean = prior_predictive_counts[i, -10:].mean()
    if early_mean > 0:
        growth_ratios.append(late_mean / early_mean)
    else:
        growth_ratios.append(np.nan)

growth_ratios = np.array(growth_ratios)
growth_ratios = growth_ratios[~np.isnan(growth_ratios)]

observed_growth = observed_C[-10:].mean() / observed_C[:10].mean()

print(f"\n5. GROWTH PATTERN (late/early ratio):")
print(f"   Observed growth ratio: {observed_growth:.1f}")
print(f"   Prior predictive growth ratio:")
print(f"     Median: {np.median(growth_ratios):.1f}")
print(f"     95% interval: [{np.percentile(growth_ratios, 2.5):.1f}, {np.percentile(growth_ratios, 97.5):.1f}]")

# ============================================================================
# STEP 4: Visualizations
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4: Creating visualizations")
print("=" * 70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# Plot 1: Parameter plausibility (marginal prior distributions)
# ============================================================================
print("\n  Creating: parameter_plausibility.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# β₀
axes[0].hist(beta_0_prior, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(4.3, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[0].set_xlabel('β₀ (log mean at year=0)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Prior: β₀ ~ Normal(4.3, 1.0)', fontsize=12, fontweight='bold')
axes[0].legend()

# Annotation: what β₀ means in real scale
exp_beta0_median = np.exp(np.median(beta_0_prior))
axes[0].text(0.05, 0.95, f'Median exp(β₀) = {exp_beta0_median:.0f}\n(mean count at year=0)',
             transform=axes[0].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# β₁
axes[1].hist(beta_1_prior, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
axes[1].axvline(0.85, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[1].axvline(0, color='gray', linestyle=':', linewidth=1.5, label='No growth')
axes[1].set_xlabel('β₁ (growth rate per year)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Prior: β₁ ~ Normal(0.85, 0.5)', fontsize=12, fontweight='bold')
axes[1].legend()

# Annotation: what β₁ means
median_multiplier = np.exp(np.median(beta_1_prior))
axes[1].text(0.05, 0.95, f'Median exp(β₁) = {median_multiplier:.2f}\n(annual multiplier)',
             transform=axes[1].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# φ
axes[2].hist(phi_prior, bins=50, alpha=0.7, color='darkorange', edgecolor='black')
axes[2].axvline(1.5, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[2].set_xlabel('φ (dispersion parameter)', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('Prior: φ ~ Exponential(rate=0.667)', fontsize=12, fontweight='bold')
axes[2].legend()

# Annotation
axes[2].text(0.55, 0.95, f'Smaller φ → more overdispersion\nMedian φ = {np.median(phi_prior):.2f}',
             transform=axes[2].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "parameter_plausibility.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 2: Prior predictive coverage (trajectories)
# ============================================================================
print("  Creating: prior_predictive_coverage.png")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot a sample of prior predictive trajectories (50 random draws)
sample_indices = np.random.choice(N_PRIOR_DRAWS, size=min(50, N_PRIOR_DRAWS), replace=False)
for idx in sample_indices:
    ax.plot(observed_year, prior_predictive_counts[idx, :],
            color='gray', alpha=0.15, linewidth=0.5)

# Plot prior predictive intervals
percentiles = [2.5, 25, 50, 75, 97.5]
for p in percentiles:
    percentile_values = np.percentile(prior_predictive_counts, p, axis=0)
    if p == 50:
        ax.plot(observed_year, percentile_values, color='blue', linewidth=2,
                label=f'{int(p)}th percentile (prior predictive)', linestyle='--')
    elif p in [2.5, 97.5]:
        ax.plot(observed_year, percentile_values, color='blue', linewidth=1.5,
                label=f'{p}th percentile', alpha=0.7, linestyle=':')

# Plot observed data
ax.scatter(observed_year, observed_C, color='red', s=60, alpha=0.8,
           label='Observed data', zorder=10, edgecolors='darkred', linewidths=1.5)

ax.set_xlabel('Standardized Year', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prior Predictive Coverage: Do priors generate plausible data?',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Add annotation
coverage_text = f"Observed range: [{observed_C.min()}, {observed_C.max()}]\n"
coverage_text += f"95% prior pred interval: [{np.percentile(prior_predictive_counts, 2.5):.0f}, {np.percentile(prior_predictive_counts, 97.5):.0f}]"
ax.text(0.98, 0.02, coverage_text, transform=ax.transAxes,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "prior_predictive_coverage.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 3: Variance-to-Mean diagnostic
# ============================================================================
print("  Creating: variance_mean_diagnostic.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Distribution of Var/Mean ratios
axes[0].hist(var_mean_ratios, bins=60, alpha=0.7, color='purple', edgecolor='black')
axes[0].axvline(observed_var_mean, color='red', linestyle='--', linewidth=2.5,
                label=f'Observed ({observed_var_mean:.1f})')
axes[0].axvline(np.median(var_mean_ratios), color='blue', linestyle='-', linewidth=2,
                label=f'Prior median ({np.median(var_mean_ratios):.1f})')

# Shade plausible region [20, 200]
ylim = axes[0].get_ylim()
axes[0].axvspan(20, 200, alpha=0.2, color='green', label='Plausible range [20, 200]')

axes[0].set_xlabel('Var/Mean Ratio', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prior Predictive Var/Mean Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, max(200, np.percentile(var_mean_ratios, 99)))

# Right panel: Var vs Mean scatter
means_per_draw = np.mean(prior_predictive_counts, axis=1)
vars_per_draw = np.var(prior_predictive_counts, axis=1)

axes[1].scatter(means_per_draw, vars_per_draw, alpha=0.4, s=20, color='purple')
axes[1].scatter(observed_C.mean(), observed_C.var(), color='red', s=150,
                marker='*', edgecolors='darkred', linewidths=2,
                label='Observed data', zorder=10)

# Add reference lines
mean_range = np.linspace(0, max(means_per_draw.max(), observed_C.mean() * 1.2), 100)
axes[1].plot(mean_range, mean_range, 'k--', alpha=0.5, label='Var = Mean (Poisson)')
axes[1].plot(mean_range, mean_range * 50, 'g:', alpha=0.5, label='Var = 50 × Mean')
axes[1].plot(mean_range, mean_range * 100, 'b:', alpha=0.5, label='Var = 100 × Mean')

axes[1].set_xlabel('Mean Count', fontsize=12)
axes[1].set_ylabel('Variance', fontsize=12)
axes[1].set_title('Variance vs Mean: Prior Predictive', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "variance_mean_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 4: Growth pattern diagnostic
# ============================================================================
print("  Creating: growth_pattern_diagnostic.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Distribution of growth ratios
axes[0].hist(growth_ratios, bins=60, alpha=0.7, color='teal', edgecolor='black')
axes[0].axvline(observed_growth, color='red', linestyle='--', linewidth=2.5,
                label=f'Observed ({observed_growth:.1f}x)')
axes[0].axvline(np.median(growth_ratios), color='blue', linestyle='-', linewidth=2,
                label=f'Prior median ({np.median(growth_ratios):.1f}x)')
axes[0].axvline(1, color='gray', linestyle=':', linewidth=1.5, label='No growth')

axes[0].set_xlabel('Late/Early Growth Ratio', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Growth Pattern: Ratio of Late to Early Means', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, min(np.percentile(growth_ratios, 99), 50))

# Right: Early vs Late counts
early_means = prior_predictive_counts[:, :10].mean(axis=1)
late_means = prior_predictive_counts[:, -10:].mean(axis=1)

axes[1].scatter(early_means, late_means, alpha=0.4, s=20, color='teal')
axes[1].scatter(observed_C[:10].mean(), observed_C[-10:].mean(),
                color='red', s=150, marker='*', edgecolors='darkred',
                linewidths=2, label='Observed', zorder=10)

# Add diagonal line (no growth)
max_val = max(early_means.max(), late_means.max(), observed_C[-10:].mean())
axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No growth')

axes[1].set_xlabel('Early Period Mean (first 10 obs)', fontsize=12)
axes[1].set_ylabel('Late Period Mean (last 10 obs)', fontsize=12)
axes[1].set_title('Growth: Early vs Late Period', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "growth_pattern_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 5: Extreme values and pathologies check
# ============================================================================
print("  Creating: pathology_diagnostic.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Distribution of maximum predicted values
max_vals = prior_predictive_counts.max(axis=1)
axes[0, 0].hist(max_vals, bins=50, alpha=0.7, color='crimson', edgecolor='black')
axes[0, 0].axvline(observed_C.max(), color='blue', linestyle='--', linewidth=2,
                   label=f'Observed max ({observed_C.max()})')
axes[0, 0].axvline(10000, color='red', linestyle=':', linewidth=2,
                   label='Extreme threshold (10,000)')
axes[0, 0].set_xlabel('Maximum Count per Prior Draw', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Maximum Values: Check for Extreme Predictions', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_xlim(0, min(np.percentile(max_vals, 99.5), 15000))

# Top-right: Zero counts per draw
axes[0, 1].hist(zero_counts, bins=range(0, int(zero_counts.max()) + 2),
                alpha=0.7, color='navy', edgecolor='black', align='left')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2,
                   label='Observed (0 zeros)')
axes[0, 1].set_xlabel('Number of Zeros per Prior Predictive Dataset', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Zero Inflation Check', fontsize=11, fontweight='bold')
axes[0, 1].legend(fontsize=9)

# Bottom-left: Distribution of μ values at different time points
mu_early = prior_predictive_mu[:, 0]  # earliest year
mu_mid = prior_predictive_mu[:, n_obs // 2]  # middle year
mu_late = prior_predictive_mu[:, -1]  # latest year

axes[1, 0].hist(mu_early, bins=40, alpha=0.5, color='green', label='Earliest year', edgecolor='black')
axes[1, 0].hist(mu_mid, bins=40, alpha=0.5, color='orange', label='Middle year', edgecolor='black')
axes[1, 0].hist(mu_late, bins=40, alpha=0.5, color='red', label='Latest year', edgecolor='black')

axes[1, 0].set_xlabel('μ (mean parameter)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Distribution of Mean Parameter μ Over Time', fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].set_xlim(0, np.percentile(mu_late, 99))

# Bottom-right: φ vs typical Var/Mean
axes[1, 1].scatter(phi_prior, var_mean_ratios, alpha=0.4, s=20, color='purple')
axes[1, 1].axhline(observed_var_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Observed Var/Mean ({observed_var_mean:.1f})')
axes[1, 1].set_xlabel('φ (dispersion parameter)', fontsize=11)
axes[1, 1].set_ylabel('Var/Mean Ratio', fontsize=11)
axes[1, 1].set_title('Dispersion vs Overdispersion', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

# Add annotation about relationship
axes[1, 1].text(0.05, 0.95, 'Smaller φ → higher Var/Mean\n(more overdispersion)',
                transform=axes[1, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pathology_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("PRIOR PREDICTIVE CHECK SUMMARY")
print("=" * 70)

# Evaluate pass/fail criteria
checks = {
    'count_coverage': covers_min >= 0.5 and covers_max >= 0.5,
    'no_extremes': n_extreme / N_PRIOR_DRAWS <= 0.1,
    'no_zero_inflation': mean_zeros_per_draw < 2,
    'var_mean_plausible': ((var_mean_ratios >= 20) & (var_mean_ratios <= 200)).sum() / N_PRIOR_DRAWS >= 0.5,
    'growth_plausible': np.percentile(growth_ratios, 2.5) <= observed_growth <= np.percentile(growth_ratios, 97.5)
}

print("\nCriteria evaluation:")
for criterion, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {criterion:25s}: {status}")

all_pass = all(checks.values())
final_decision = "PASS" if all_pass else "FAIL"

print(f"\n{'='*70}")
print(f"FINAL DECISION: {final_decision}")
print("=" * 70)

if all_pass:
    print("\nAll checks passed. Priors appear well-calibrated:")
    print("  - Generate data covering observed range")
    print("  - No excessive extreme values or zeros")
    print("  - Allow for observed overdispersion")
    print("  - Compatible with observed growth pattern")
    print("\nRecommendation: PROCEED to model fitting.")
else:
    print("\nSome checks failed. Issues detected:")
    failed_checks = [k for k, v in checks.items() if not v]
    for check in failed_checks:
        print(f"  - {check}")
    print("\nRecommendation: REVISE priors before fitting.")

print(f"\nPlots saved to: {OUTPUT_DIR}")
print("  - parameter_plausibility.png")
print("  - prior_predictive_coverage.png")
print("  - variance_mean_diagnostic.png")
print("  - growth_pattern_diagnostic.png")
print("  - pathology_diagnostic.png")

print("\n" + "=" * 70)
print("Prior predictive check complete!")
print("=" * 70)
