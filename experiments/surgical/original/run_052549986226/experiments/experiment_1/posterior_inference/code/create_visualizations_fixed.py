"""
Create Diagnostic Visualizations for Beta-Binomial Posterior Inference
Experiment 1: Comprehensive visual diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
DATA_PATH = Path("/workspace/data/data.csv")

# Load data
print("Loading posterior inference data...")
idata = az.from_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')
data = pd.read_csv(DATA_PATH)
group_summary = pd.read_csv(RESULTS_DIR / 'group_posterior_summary.csv')
posterior_samples = pd.read_csv(RESULTS_DIR / 'posterior_samples_scalar.csv')

print(f"Loaded {len(posterior_samples)} posterior samples")
print(f"Groups: {len(data)}")

# ============================================================================
# 1. COMPREHENSIVE CONVERGENCE OVERVIEW (combined trace, density, autocorr)
# ============================================================================
print("\n1. Creating comprehensive convergence overview...")

fig = az.plot_trace(idata, var_names=['mu', 'kappa', 'phi'], figsize=(14, 10))
plt.suptitle('Trace Plots and Posterior Distributions', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots.png', bbox_inches='tight')
print(f"  Saved: trace_plots.png")
plt.close()

# ============================================================================
# 2. POSTERIOR DISTRIBUTIONS vs PRIORS
# ============================================================================
print("\n2. Creating posterior vs prior distributions...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Mu: Beta(2, 18) prior
x_mu = np.linspace(0, 0.4, 1000)
prior_mu = stats.beta(2, 18).pdf(x_mu)
axes[0].plot(x_mu, prior_mu, 'k--', label='Prior: Beta(2, 18)', linewidth=2, alpha=0.7)
axes[0].hist(posterior_samples['mu'], bins=50, density=True, alpha=0.6, color='steelblue', label='Posterior')
axes[0].axvline(data['r_successes'].sum() / data['n_trials'].sum(), color='red', linestyle=':', linewidth=2, label='Observed pooled rate')
axes[0].set_xlabel('μ (Population Mean)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Population Mean Success Probability', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Kappa: Gamma(2, 0.1) prior
x_kappa = np.linspace(0, 120, 1000)
prior_kappa = stats.gamma(2, scale=1/0.1).pdf(x_kappa)
axes[1].plot(x_kappa, prior_kappa, 'k--', label='Prior: Gamma(2, 0.1)', linewidth=2, alpha=0.7)
axes[1].hist(posterior_samples['kappa'], bins=50, density=True, alpha=0.6, color='darkgreen', label='Posterior')
axes[1].set_xlabel('κ (Concentration)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('Concentration Parameter', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Phi: derived from kappa
x_phi = np.linspace(1, 1.6, 1000)
# Prior phi from kappa prior
kappa_samples_prior = stats.gamma(2, scale=1/0.1).rvs(10000)
phi_prior = 1 + 1/kappa_samples_prior
axes[2].hist(phi_prior, bins=100, density=True, alpha=0.3, color='gray', label='Prior (implied)', range=(1, 1.6))
axes[2].hist(posterior_samples['phi'], bins=50, density=True, alpha=0.6, color='coral', label='Posterior')
axes[2].axvline(1.02, color='red', linestyle=':', linewidth=2, label='Expected φ ≈ 1.02')
axes[2].set_xlabel('φ (Overdispersion)', fontsize=11)
axes[2].set_ylabel('Density', fontsize=11)
axes[2].set_title('Overdispersion Parameter', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].set_xlim(1, 1.2)

plt.suptitle('Posterior Distributions vs Priors', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_distributions.png', bbox_inches='tight')
print(f"  Saved: posterior_distributions.png")
plt.close()

# ============================================================================
# 3. PAIRS PLOT - Check correlations
# ============================================================================
print("\n3. Creating pairs plot...")

fig = az.plot_pair(
    idata,
    var_names=['mu', 'kappa'],
    kind='kde',
    marginals=True,
    figsize=(10, 10)
)
plt.suptitle('Joint Posterior: μ and κ Correlation', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / 'pairs_plot.png', bbox_inches='tight')
print(f"  Saved: pairs_plot.png")
plt.close()

# ============================================================================
# 4. GROUP POSTERIOR CATERPILLAR PLOT
# ============================================================================
print("\n4. Creating caterpillar plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Sort by posterior mean
group_summary_sorted = group_summary.sort_values('posterior_mean')

y_pos = np.arange(len(group_summary_sorted))

# Plot observed rates
ax.scatter(group_summary_sorted['observed_rate'], y_pos, color='red', s=80, marker='o',
           label='Observed rate', zorder=3, alpha=0.7)

# Plot posterior means with 95% CIs
ax.errorbar(
    group_summary_sorted['posterior_mean'],
    y_pos,
    xerr=[
        group_summary_sorted['posterior_mean'] - group_summary_sorted['posterior_025'],
        group_summary_sorted['posterior_975'] - group_summary_sorted['posterior_mean']
    ],
    fmt='o',
    color='steelblue',
    markersize=6,
    capsize=4,
    label='Posterior mean (95% CI)',
    zorder=2
)

# Add population mean line
mu_post_mean = posterior_samples['mu'].mean()
ax.axvline(mu_post_mean, color='green', linestyle='--', linewidth=2, label=f'Population mean μ = {mu_post_mean:.3f}')

ax.set_yticks(y_pos)
ax.set_yticklabels([f"Group {g}" for g in group_summary_sorted['group']])
ax.set_xlabel('Success Probability', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Group-Level Posterior Estimates with Shrinkage', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'caterpillar_plot.png', bbox_inches='tight')
print(f"  Saved: caterpillar_plot.png")
plt.close()

# ============================================================================
# 5. SHRINKAGE VISUALIZATION
# ============================================================================
print("\n5. Creating shrinkage plot...")

fig, ax = plt.subplots(figsize=(10, 8))

mu_post_mean = posterior_samples['mu'].mean()

for idx, row in group_summary.iterrows():
    # Arrow from observed to posterior
    ax.annotate('',
                xy=(row['posterior_mean'], idx),
                xytext=(row['observed_rate'], idx),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.6))

# Plot points
ax.scatter(group_summary['observed_rate'], range(len(group_summary)),
           color='red', s=100, marker='o', label='Observed rate', zorder=3, alpha=0.8)
ax.scatter(group_summary['posterior_mean'], range(len(group_summary)),
           color='steelblue', s=100, marker='s', label='Posterior mean', zorder=3, alpha=0.8)

# Population mean
ax.axvline(mu_post_mean, color='green', linestyle='--', linewidth=2, label=f'Population μ = {mu_post_mean:.3f}')

ax.set_yticks(range(len(group_summary)))
ax.set_yticklabels([f"Group {g} (n={n})" for g, n in zip(group_summary['group'], group_summary['n_trials'])])
ax.set_xlabel('Success Probability', fontsize=12)
ax.set_ylabel('Group (Sample Size)', fontsize=12)
ax.set_title('Shrinkage: Observed → Posterior (arrows show direction)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'shrinkage_plot.png', bbox_inches='tight')
print(f"  Saved: shrinkage_plot.png")
plt.close()

# ============================================================================
# 6. POSTERIOR VS OBSERVED SCATTER
# ============================================================================
print("\n6. Creating posterior vs observed scatter...")

fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot
ax.scatter(group_summary['observed_rate'], group_summary['posterior_mean'],
           s=100*np.sqrt(group_summary['n_trials']), alpha=0.6, color='steelblue')

# Add error bars for posterior
ax.errorbar(
    group_summary['observed_rate'],
    group_summary['posterior_mean'],
    yerr=[
        group_summary['posterior_mean'] - group_summary['posterior_025'],
        group_summary['posterior_975'] - group_summary['posterior_mean']
    ],
    fmt='none',
    color='steelblue',
    alpha=0.3,
    capsize=3
)

# Identity line
max_val = max(group_summary['observed_rate'].max(), group_summary['posterior_mean'].max())
ax.plot([0, max_val], [0, max_val], 'k--', label='Identity (no shrinkage)', alpha=0.5)

# Annotate special groups
for idx, row in group_summary.iterrows():
    if row['group'] in [1, 4, 8]:  # Group 1 (zero), 4 (large n), 8 (high rate)
        ax.annotate(f"G{row['group']}",
                   xy=(row['observed_rate'], row['posterior_mean']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)

ax.set_xlabel('Observed Rate', fontsize=12)
ax.set_ylabel('Posterior Mean', fontsize=12)
ax.set_title('Posterior vs Observed Rates\n(bubble size ∝ sample size)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Add text with correlation
from scipy.stats import pearsonr
r, _ = pearsonr(group_summary['observed_rate'], group_summary['posterior_mean'])
ax.text(0.05, 0.95, f'Correlation: {r:.3f}', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_vs_observed.png', bbox_inches='tight')
print(f"  Saved: posterior_vs_observed.png")
plt.close()

# ============================================================================
# 7. ENERGY PLOT - HMC diagnostic
# ============================================================================
print("\n7. Creating energy plot...")

fig = az.plot_energy(idata, figsize=(8, 6))
plt.suptitle('Energy Plot: HMC Geometry Diagnostic', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'energy_plot.png', bbox_inches='tight')
print(f"  Saved: energy_plot.png")
plt.close()

# ============================================================================
# 8. RANK PLOTS - Additional convergence diagnostic
# ============================================================================
print("\n8. Creating rank plots...")

fig = az.plot_rank(idata, var_names=['mu', 'kappa', 'phi'], figsize=(12, 10))
plt.suptitle('Rank Plots: Uniform → Good Mixing', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rank_plots.png', bbox_inches='tight')
print(f"  Saved: rank_plots.png")
plt.close()

# ============================================================================
# 9. SHRINKAGE ANALYSIS DETAILED
# ============================================================================
print("\n9. Creating detailed shrinkage analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Shrinkage vs sample size
axes[0, 0].scatter(group_summary['n_trials'], group_summary['shrinkage_pct'],
                   s=100, alpha=0.6, color='purple')
axes[0, 0].set_xlabel('Sample Size (n)', fontsize=11)
axes[0, 0].set_ylabel('Shrinkage (%)', fontsize=11)
axes[0, 0].set_title('Shrinkage vs Sample Size', fontweight='bold')
axes[0, 0].grid(alpha=0.3)
# Add trendline
from numpy.polynomial import Polynomial
p = Polynomial.fit(group_summary['n_trials'], group_summary['shrinkage_pct'], 1)
x_line = np.linspace(group_summary['n_trials'].min(), group_summary['n_trials'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
axes[0, 0].legend()

# Panel 2: Shrinkage vs distance from mean
distance_from_mean = np.abs(group_summary['observed_rate'] - mu_post_mean)
axes[0, 1].scatter(distance_from_mean, group_summary['shrinkage_pct'],
                   s=100, alpha=0.6, color='orange')
axes[0, 1].set_xlabel('Distance from Population Mean', fontsize=11)
axes[0, 1].set_ylabel('Shrinkage (%)', fontsize=11)
axes[0, 1].set_title('Shrinkage vs Distance from Mean', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Panel 3: Shrinkage magnitude
shrinkage_magnitude = np.abs(group_summary['observed_rate'] - group_summary['posterior_mean'])
axes[1, 0].bar(range(len(group_summary)), shrinkage_magnitude, color='teal', alpha=0.7)
axes[1, 0].set_xticks(range(len(group_summary)))
axes[1, 0].set_xticklabels([f"G{g}" for g in group_summary['group']], fontsize=9)
axes[1, 0].set_xlabel('Group', fontsize=11)
axes[1, 0].set_ylabel('Absolute Shrinkage', fontsize=11)
axes[1, 0].set_title('Magnitude of Shrinkage by Group', fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# Panel 4: Summary statistics table
summary_stats = {
    'Statistic': ['Mean shrinkage %', 'Median shrinkage %', 'Max shrinkage %',
                  'Min shrinkage %', 'Mean abs. shrinkage', 'Groups with >20% shrinkage'],
    'Value': [
        f"{group_summary['shrinkage_pct'].mean():.1f}%",
        f"{group_summary['shrinkage_pct'].median():.1f}%",
        f"{group_summary['shrinkage_pct'].max():.1f}% (Group {group_summary.loc[group_summary['shrinkage_pct'].idxmax(), 'group']:.0f})",
        f"{group_summary['shrinkage_pct'].min():.1f}% (Group {group_summary.loc[group_summary['shrinkage_pct'].idxmin(), 'group']:.0f})",
        f"{shrinkage_magnitude.mean():.4f}",
        f"{(group_summary['shrinkage_pct'] > 20).sum()}"
    ]
}
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=[[s, v] for s, v in zip(summary_stats['Statistic'], summary_stats['Value'])],
                         colLabels=['Statistic', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
axes[1, 1].set_title('Shrinkage Summary Statistics', fontweight='bold', pad=20)

plt.suptitle('Detailed Shrinkage Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'shrinkage_analysis_detailed.png', bbox_inches='tight')
print(f"  Saved: shrinkage_analysis_detailed.png")
plt.close()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\nAll plots saved to: {PLOTS_DIR}")
print("\nGenerated plots:")
print("  1. trace_plots.png - MCMC chain mixing and posteriors")
print("  2. posterior_distributions.png - Posterior vs prior comparison")
print("  3. pairs_plot.png - Joint posterior of μ and κ")
print("  4. caterpillar_plot.png - Group posteriors with CIs")
print("  5. shrinkage_plot.png - Observed → posterior shrinkage")
print("  6. posterior_vs_observed.png - Scatter plot comparison")
print("  7. energy_plot.png - HMC geometry diagnostic")
print("  8. rank_plots.png - Chain uniformity diagnostic")
print("  9. shrinkage_analysis_detailed.png - Detailed shrinkage patterns")
