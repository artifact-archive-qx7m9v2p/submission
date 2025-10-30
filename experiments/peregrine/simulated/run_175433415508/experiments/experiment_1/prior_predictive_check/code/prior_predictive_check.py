"""
Prior Predictive Check for Experiment 1: Negative Binomial Linear Model

This script validates that the specified priors generate scientifically plausible
data BEFORE fitting the model to actual observations.

Model:
  C_t ~ NegativeBinomial(mu_t, phi)
  log(mu_t) = beta_0 + beta_1 * year_t

Priors:
  beta_0 ~ Normal(4.69, 1.0)    # log(109.4)
  beta_1 ~ Normal(1.0, 0.5)      # Positive growth
  phi ~ Gamma(2, 0.1)            # Overdispersion (mean=20)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_DRAWS = 100
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Ensure output directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 80)
print("PRIOR PREDICTIVE CHECK: Negative Binomial Linear Model")
print("=" * 80)

# Load data (to get year predictor structure, NOT the response)
data = pd.read_csv(DATA_PATH)
year = data['year'].values
C_observed = data['C'].values
N = len(year)

print(f"\nData structure:")
print(f"  N observations: {N}")
print(f"  Year range: [{year.min():.2f}, {year.max():.2f}]")
print(f"  Observed C range: [{C_observed.min()}, {C_observed.max()}]")
print(f"  Observed mean: {C_observed.mean():.1f}")
print(f"  Observed variance: {C_observed.var():.1f}")

# ============================================================================
# PRIOR PREDICTIVE SAMPLING (Pure Python Implementation)
# ============================================================================

print(f"\nSampling {N_PRIOR_DRAWS} datasets from prior predictive distribution...")

# Storage for samples
beta_0_samples = np.zeros(N_PRIOR_DRAWS)
beta_1_samples = np.zeros(N_PRIOR_DRAWS)
phi_samples = np.zeros(N_PRIOR_DRAWS)
C_sim_samples = np.zeros((N_PRIOR_DRAWS, N))

# Sample from priors and generate synthetic data
for i in range(N_PRIOR_DRAWS):
    # Sample parameters from priors
    beta_0 = np.random.normal(4.69, 1.0)
    beta_1 = np.random.normal(1.0, 0.5)
    phi = np.random.gamma(2, 1/0.1)  # Note: numpy uses shape, scale parameterization

    beta_0_samples[i] = beta_0
    beta_1_samples[i] = beta_1
    phi_samples[i] = phi

    # Generate synthetic counts
    for t in range(N):
        mu = np.exp(beta_0 + beta_1 * year[t])

        # Negative binomial parameterization: p = phi / (phi + mu)
        # This is the NB2 parameterization where var = mu + mu^2/phi
        p = phi / (phi + mu)
        C_sim_samples[i, t] = np.random.negative_binomial(phi, p)

print("Prior sampling complete!")

# ============================================================================
# DIAGNOSTIC ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC ANALYSIS")
print("=" * 80)

# 1. Prior parameter distributions
print("\n1. PRIOR PARAMETER SAMPLES:")
print(f"   beta_0: mean={beta_0_samples.mean():.2f}, sd={beta_0_samples.std():.2f}, "
      f"range=[{beta_0_samples.min():.2f}, {beta_0_samples.max():.2f}]")
print(f"   beta_1: mean={beta_1_samples.mean():.2f}, sd={beta_1_samples.std():.2f}, "
      f"range=[{beta_1_samples.min():.2f}, {beta_1_samples.max():.2f}]")
print(f"   phi: mean={phi_samples.mean():.2f}, sd={phi_samples.std():.2f}, "
      f"range=[{phi_samples.min():.2f}, {phi_samples.max():.2f}]")

# Check for negative beta_1 (should be rare given prior)
n_negative_growth = np.sum(beta_1_samples < 0)
print(f"\n   Negative growth (beta_1 < 0): {n_negative_growth}/{N_PRIOR_DRAWS} "
      f"({100*n_negative_growth/N_PRIOR_DRAWS:.1f}%)")

# 2. Prior predictive count distributions
print("\n2. PRIOR PREDICTIVE COUNT STATISTICS:")
C_means = C_sim_samples.mean(axis=1)
C_vars = C_sim_samples.var(axis=1)
C_mins = C_sim_samples.min(axis=1)
C_maxs = C_sim_samples.max(axis=1)

print(f"   Dataset means: {C_means.mean():.1f} [{C_means.min():.1f}, {C_means.max():.1f}]")
print(f"   Dataset vars: {C_vars.mean():.1f} [{C_vars.min():.1f}, {C_vars.max():.1f}]")
print(f"   Dataset mins: {C_mins.mean():.1f} [{C_mins.min():.0f}, {C_mins.max():.0f}]")
print(f"   Dataset maxs: {C_maxs.mean():.1f} [{C_maxs.min():.0f}, {C_maxs.max():.0f}]")

# 3. Range checks
all_counts = C_sim_samples.flatten()
print("\n3. RANGE CHECKS:")
print(f"   Total prior predictive counts: {len(all_counts)}")
print(f"   Counts in [0, 1000]: {np.sum((all_counts >= 0) & (all_counts <= 1000))}/{len(all_counts)} "
      f"({100*np.sum((all_counts >= 0) & (all_counts <= 1000))/len(all_counts):.1f}%)")
print(f"   Counts in [0, 5000]: {np.sum((all_counts >= 0) & (all_counts <= 5000))}/{len(all_counts)} "
      f"({100*np.sum((all_counts >= 0) & (all_counts <= 5000))/len(all_counts):.1f}%)")
print(f"   Counts > 10000: {np.sum(all_counts > 10000)}/{len(all_counts)} "
      f"({100*np.sum(all_counts > 10000)/len(all_counts):.1f}%)")

# 4. Domain violations
n_negative = np.sum(all_counts < 0)
print(f"\n4. DOMAIN VIOLATIONS:")
print(f"   Negative counts: {n_negative}/{len(all_counts)}")

# 5. Compute mu values at endpoints
mu_at_min_year = np.exp(beta_0_samples + beta_1_samples * year.min())
mu_at_max_year = np.exp(beta_0_samples + beta_1_samples * year.max())

print(f"\n5. EXPECTED COUNTS AT ENDPOINTS:")
print(f"   mu at year={year.min():.2f}: mean={mu_at_min_year.mean():.1f}, "
      f"range=[{mu_at_min_year.min():.1f}, {mu_at_max_year.min():.1f}]")
print(f"   mu at year={year.max():.2f}: mean={mu_at_max_year.mean():.1f}, "
      f"range=[{mu_at_max_year.min():.1f}, {mu_at_max_year.max():.1f}]")

# 6. Growth factor analysis
growth_factor = mu_at_max_year / mu_at_min_year
print(f"\n6. GROWTH ANALYSIS:")
print(f"   Growth factor (max/min year): mean={growth_factor.mean():.2f}, "
      f"range=[{growth_factor.min():.2f}, {growth_factor.max():.2f}]")
print(f"   This represents {growth_factor.mean():.1f}x growth over the time period")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# PLOT 1: Prior parameter distributions
print("\nPlot 1: Prior parameter distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Beta_0
axes[0].hist(beta_0_samples, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(4.69, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[0].set_xlabel('beta_0 (Intercept)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prior: beta_0 ~ Normal(4.69, 1.0)', fontsize=13, fontweight='bold')
axes[0].legend()

# Beta_1
axes[1].hist(beta_1_samples, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
axes[1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[1].axvline(0, color='black', linestyle=':', linewidth=1, label='Zero growth')
axes[1].set_xlabel('beta_1 (Growth rate)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Prior: beta_1 ~ Normal(1.0, 0.5)', fontsize=13, fontweight='bold')
axes[1].legend()

# Phi
axes[2].hist(phi_samples, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
axes[2].axvline(20, color='red', linestyle='--', linewidth=2, label='Prior mean')
axes[2].set_xlabel('phi (Overdispersion)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].set_title('Prior: phi ~ Gamma(2, 0.1)', fontsize=13, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'parameter_plausibility.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: parameter_plausibility.png")

# PLOT 2: Prior predictive growth curves
print("\nPlot 2: Prior predictive growth curves...")
fig, ax = plt.subplots(figsize=(12, 7))

# Plot a sample of prior predictive curves
n_curves_to_plot = min(50, N_PRIOR_DRAWS)
for i in range(n_curves_to_plot):
    ax.plot(year, C_sim_samples[i, :], alpha=0.2, color='steelblue', linewidth=1)

# Overlay observed data for context (but make it clear this is just for reference)
ax.scatter(year, C_observed, color='red', s=50, alpha=0.8, zorder=10,
           label='Observed data (reference)', edgecolors='darkred', linewidth=1.5)

ax.set_xlabel('Year (standardized)', fontsize=13)
ax.set_ylabel('Count (C)', fontsize=13)
ax.set_title(f'Prior Predictive Growth Curves (n={n_curves_to_plot} samples)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'prior_predictive_coverage.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: prior_predictive_coverage.png")

# PLOT 3: Prior predictive summary statistics
print("\nPlot 3: Prior predictive summary statistics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean
axes[0, 0].hist(C_means, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(C_observed.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {C_observed.mean():.1f}')
axes[0, 0].set_xlabel('Dataset Mean', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Distribution of Dataset Means', fontsize=13, fontweight='bold')
axes[0, 0].legend()

# Variance
axes[0, 1].hist(C_vars, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
axes[0, 1].axvline(C_observed.var(), color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {C_observed.var():.1f}')
axes[0, 1].set_xlabel('Dataset Variance', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Distribution of Dataset Variances', fontsize=13, fontweight='bold')
axes[0, 1].legend()

# Min
axes[1, 0].hist(C_mins, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
axes[1, 0].axvline(C_observed.min(), color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {C_observed.min()}')
axes[1, 0].set_xlabel('Dataset Minimum', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Dataset Minima', fontsize=13, fontweight='bold')
axes[1, 0].legend()

# Max
axes[1, 1].hist(C_maxs, bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(C_observed.max(), color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {C_observed.max()}')
axes[1, 1].set_xlabel('Dataset Maximum', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distribution of Dataset Maxima', fontsize=13, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'prior_summary_diagnostics.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: prior_summary_diagnostics.png")

# PLOT 4: Count distribution across all prior predictive samples
print("\nPlot 4: Prior predictive count distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full range
axes[0].hist(all_counts, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(C_observed.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Observed mean: {C_observed.mean():.1f}')
axes[0].set_xlabel('Count value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('All Prior Predictive Counts', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].set_xlim(0, np.percentile(all_counts, 99))

# Log scale for extreme values
axes[1].hist(all_counts, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[1].axvline(C_observed.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Observed mean: {C_observed.mean():.1f}')
axes[1].set_xlabel('Count value', fontsize=12)
axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
axes[1].set_yscale('log')
axes[1].set_title('All Prior Predictive Counts (log scale)', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'count_distribution_diagnostic.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: count_distribution_diagnostic.png")

# PLOT 5: Endpoint expected values
print("\nPlot 5: Expected values at endpoints...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# At minimum year
axes[0].hist(mu_at_min_year, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(mu_at_min_year.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mu_at_min_year.mean():.1f}')
axes[0].set_xlabel('Expected count (mu)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'mu at year={year.min():.2f} (earliest)', fontsize=13, fontweight='bold')
axes[0].legend()

# At maximum year
axes[1].hist(mu_at_max_year, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
axes[1].axvline(mu_at_max_year.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mu_at_max_year.mean():.1f}')
axes[1].set_xlabel('Expected count (mu)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'mu at year={year.max():.2f} (latest)', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'endpoint_diagnostics.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: endpoint_diagnostics.png")

# PLOT 6: Comprehensive diagnostic panel
print("\nPlot 6: Comprehensive diagnostic overview...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Parameter distributions
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(beta_0_samples, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(4.69, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('beta_0', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Intercept Prior', fontsize=11, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(beta_1_samples, bins=25, alpha=0.7, color='darkorange', edgecolor='black')
ax2.axvline(1.0, color='red', linestyle='--', linewidth=2)
ax2.axvline(0, color='black', linestyle=':', linewidth=1)
ax2.set_xlabel('beta_1', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Growth Rate Prior', fontsize=11, fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(phi_samples, bins=25, alpha=0.7, color='forestgreen', edgecolor='black')
ax3.axvline(20, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('phi', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Overdispersion Prior', fontsize=11, fontweight='bold')

# Row 2: Growth curves and endpoints
ax4 = fig.add_subplot(gs[1, :])
for i in range(min(30, N_PRIOR_DRAWS)):
    ax4.plot(year, C_sim_samples[i, :], alpha=0.2, color='steelblue', linewidth=1)
ax4.scatter(year, C_observed, color='red', s=30, alpha=0.8, zorder=10,
           label='Observed (reference)', edgecolors='darkred', linewidth=1.5)
ax4.set_xlabel('Year (standardized)', fontsize=11)
ax4.set_ylabel('Count (C)', fontsize=11)
ax4.set_title('Prior Predictive Growth Curves', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Row 3: Summary statistics
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(C_means, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
ax5.axvline(C_observed.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Obs: {C_observed.mean():.0f}')
ax5.set_xlabel('Mean', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Dataset Means', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)

ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(C_maxs, bins=25, alpha=0.7, color='purple', edgecolor='black')
ax6.axvline(C_observed.max(), color='red', linestyle='--', linewidth=2,
            label=f'Obs: {C_observed.max()}')
ax6.set_xlabel('Maximum', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title('Dataset Maxima', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)

ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(growth_factor, bins=25, alpha=0.7, color='coral', edgecolor='black')
ax7.axvline(growth_factor.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {growth_factor.mean():.1f}x')
ax7.set_xlabel('Growth Factor', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('Max/Min Year Ratio', fontsize=11, fontweight='bold')
ax7.legend(fontsize=9)

fig.suptitle('Prior Predictive Check: Comprehensive Diagnostic Overview',
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig(os.path.join(PLOTS_DIR, 'comprehensive_diagnostic.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: comprehensive_diagnostic.png")

# ============================================================================
# PASS/FAIL DECISION
# ============================================================================

print("\n" + "=" * 80)
print("DECISION CRITERIA")
print("=" * 80)

# Criteria
criteria = {
    'no_negative_counts': n_negative == 0,
    'counts_in_reasonable_range': np.sum((all_counts >= 0) & (all_counts <= 5000)) / len(all_counts) >= 0.95,
    'no_extreme_outliers': np.sum(all_counts > 10000) / len(all_counts) < 0.01,
    'growth_mostly_positive': n_negative_growth / N_PRIOR_DRAWS < 0.20,
    'mean_covers_observed': (C_means.min() <= C_observed.mean() <= C_means.max()),
    'max_covers_observed': (C_maxs.min() <= C_observed.max() <= C_maxs.max()),
}

print("\nCriteria evaluation:")
for criterion, passed in criteria.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {criterion}")

overall_pass = all(criteria.values())

print("\n" + "=" * 80)
if overall_pass:
    print("OVERALL DECISION: PASS")
    print("The priors generate scientifically plausible data.")
else:
    print("OVERALL DECISION: FAIL")
    print("The priors generate implausible data and need adjustment.")
print("=" * 80)

# Save results summary
results = {
    'n_prior_draws': N_PRIOR_DRAWS,
    'parameter_samples': {
        'beta_0': {
            'mean': float(beta_0_samples.mean()),
            'sd': float(beta_0_samples.std()),
            'range': [float(beta_0_samples.min()), float(beta_0_samples.max())]
        },
        'beta_1': {
            'mean': float(beta_1_samples.mean()),
            'sd': float(beta_1_samples.std()),
            'range': [float(beta_1_samples.min()), float(beta_1_samples.max())],
            'n_negative': int(n_negative_growth)
        },
        'phi': {
            'mean': float(phi_samples.mean()),
            'sd': float(phi_samples.std()),
            'range': [float(phi_samples.min()), float(phi_samples.max())]
        },
    },
    'prior_predictive_stats': {
        'count_mean': {
            'mean': float(C_means.mean()),
            'range': [float(C_means.min()), float(C_means.max())]
        },
        'count_var': {
            'mean': float(C_vars.mean()),
            'range': [float(C_vars.min()), float(C_vars.max())]
        },
        'count_min': {
            'mean': float(C_mins.mean()),
            'range': [float(C_mins.min()), float(C_mins.max())]
        },
        'count_max': {
            'mean': float(C_maxs.mean()),
            'range': [float(C_maxs.min()), float(C_maxs.max())]
        },
        'growth_factor': {
            'mean': float(growth_factor.mean()),
            'range': [float(growth_factor.min()), float(growth_factor.max())]
        },
    },
    'range_checks': {
        'pct_in_0_1000': float(100 * np.sum((all_counts >= 0) & (all_counts <= 1000)) / len(all_counts)),
        'pct_in_0_5000': float(100 * np.sum((all_counts >= 0) & (all_counts <= 5000)) / len(all_counts)),
        'pct_above_10000': float(100 * np.sum(all_counts > 10000) / len(all_counts)),
    },
    'observed_data': {
        'mean': float(C_observed.mean()),
        'variance': float(C_observed.var()),
        'min': int(C_observed.min()),
        'max': int(C_observed.max()),
    },
    'criteria': criteria,
    'overall_pass': overall_pass
}

results_file = os.path.join(OUTPUT_DIR, 'code', 'results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_file}")
print("\nPrior predictive check complete!")
