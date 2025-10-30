"""
Create diagnostic visualizations for prior predictive check.

This script creates multiple visualizations designed to answer key diagnostic questions:
1. Are the priors generating plausible parameter values?
2. Do prior predictions cover observed data range?
3. Are there computational or structural issues?
4. How do different prior components interact?
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load prior predictive samples
data = np.load('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.npz')
mu_prior = data['mu_prior']
tau_prior = data['tau_prior']
theta_samples = data['theta_samples']
y_pred_samples = data['y_pred_samples']
pooled_effect_samples = data['pooled_effect_samples']
observed_y = data['observed_y']
known_sigma = data['known_sigma']
observed_pooled_effect = data['observed_pooled_effect']
observed_tau = data['observed_tau']

J = len(observed_y)
n_samples = len(mu_prior)

print("Creating visualizations...")

# ============================================================================
# Plot 1: Parameter Plausibility - Are priors generating reasonable values?
# ============================================================================
print("  1. Parameter plausibility check...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prior for mu
ax = axes[0]
ax.hist(mu_prior, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
x = np.linspace(mu_prior.min(), mu_prior.max(), 200)
ax.plot(x, stats.norm.pdf(x, 0, 25), 'r-', linewidth=2, label='Prior: N(0, 25)')
ax.axvline(observed_pooled_effect, color='darkgreen', linewidth=2, linestyle='--',
           label=f'Observed pooled effect: {observed_pooled_effect:.2f}')
ax.set_xlabel('Overall mean effect (mu)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution: mu ~ N(0, 25)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Prior for tau
ax = axes[1]
ax.hist(tau_prior, bins=50, density=True, alpha=0.6, color='coral', edgecolor='black')
x = np.linspace(0, tau_prior.max(), 200)
ax.plot(x, stats.halfnorm.pdf(x, 0, 10), 'r-', linewidth=2, label='Prior: Half-N(0, 10)')
ax.axvline(observed_tau, color='darkgreen', linewidth=2, linestyle='--',
           label=f'Observed tau: {observed_tau:.2f}')
ax.set_xlabel('Between-study SD (tau)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Prior Distribution: tau ~ Half-N(0, 10)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Prior Parameter Plausibility', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/parameter_plausibility.png',
            dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 2: Prior Predictive Coverage - Study-level comparison
# ============================================================================
print("  2. Study-level prior predictive coverage...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(J):
    ax = axes[j]

    # Plot prior predictive distribution
    ax.hist(y_pred_samples[:, j], bins=50, density=True, alpha=0.6,
            color='skyblue', edgecolor='black', label='Prior predictive')

    # Mark observed value
    ax.axvline(observed_y[j], color='red', linewidth=2.5, linestyle='--',
               label=f'Observed: {observed_y[j]:.2f}')

    # Mark prior predictive quantiles
    q05, q25, q50, q75, q95 = np.percentile(y_pred_samples[:, j], [5, 25, 50, 75, 95])
    ax.axvline(q50, color='blue', linewidth=1.5, linestyle='-', alpha=0.7, label='Median')
    ax.axvspan(q25, q75, color='blue', alpha=0.15, label='50% interval')
    ax.axvspan(q05, q95, color='blue', alpha=0.08, label='90% interval')

    # Calculate percentile of observed
    percentile = stats.percentileofscore(y_pred_samples[:, j], observed_y[j])

    ax.set_xlabel(f'y_{j+1} (sigma={known_sigma[j]})', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Study {j+1}: Obs at {percentile:.1f}%', fontsize=11, fontweight='bold')
    if j == 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Study-Level Prior Predictive Coverage', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/study_level_coverage.png',
            dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 3: Pooled Effect Diagnostic
# ============================================================================
print("  3. Pooled effect diagnostic...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(pooled_effect_samples, bins=60, density=True, alpha=0.6,
        color='mediumpurple', edgecolor='black', label='Prior predictive')

# Mark observed pooled effect
ax.axvline(observed_pooled_effect, color='red', linewidth=2.5, linestyle='--',
           label=f'Observed: {observed_pooled_effect:.2f}')

# Mark prior predictive quantiles
q05, q25, q50, q75, q95 = np.percentile(pooled_effect_samples, [5, 25, 50, 75, 95])
ax.axvline(q50, color='darkblue', linewidth=1.5, linestyle='-', alpha=0.7, label='Median')
ax.axvspan(q25, q75, color='darkblue', alpha=0.15, label='50% interval')
ax.axvspan(q05, q95, color='darkblue', alpha=0.08, label='90% interval')

# Calculate percentile
percentile = stats.percentileofscore(pooled_effect_samples, observed_pooled_effect)

ax.set_xlabel('Pooled Effect (mean of y_i)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Prior Predictive: Pooled Effect (Observed at {percentile:.1f}%)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/pooled_effect_coverage.png',
            dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 4: Hierarchical Structure Diagnostic - Joint behavior of mu and tau
# ============================================================================
print("  4. Hierarchical structure diagnostic...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot: mu vs tau from prior
ax = axes[0]
scatter = ax.scatter(mu_prior, tau_prior, alpha=0.3, s=10, c='steelblue')
ax.axvline(observed_pooled_effect, color='red', linewidth=2, linestyle='--',
           label='Observed pooled effect', alpha=0.7)
ax.axhline(observed_tau, color='darkgreen', linewidth=2, linestyle='--',
           label='Observed tau', alpha=0.7)
ax.set_xlabel('Overall mean (mu)', fontsize=11)
ax.set_ylabel('Between-study SD (tau)', fontsize=11)
ax.set_title('Joint Prior: mu vs tau', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Histogram of tau with observed marked
ax = axes[1]
ax.hist(tau_prior, bins=60, density=True, alpha=0.6, color='coral',
        edgecolor='black', label='Prior predictive')
ax.axvline(observed_tau, color='darkgreen', linewidth=2.5, linestyle='--',
           label=f'Observed: {observed_tau:.2f}')

q05, q25, q50, q75, q95 = np.percentile(tau_prior, [5, 25, 50, 75, 95])
ax.axvline(q50, color='darkred', linewidth=1.5, linestyle='-', alpha=0.7, label='Median')
ax.axvspan(q25, q75, color='darkred', alpha=0.15, label='50% interval')
ax.axvspan(q05, q95, color='darkred', alpha=0.08, label='90% interval')

tau_percentile = stats.percentileofscore(tau_prior, observed_tau)
ax.set_xlabel('Between-study SD (tau)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Prior for tau (Observed at {tau_percentile:.1f}%)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Hierarchical Structure: Joint Prior Behavior', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/hierarchical_structure_diagnostic.png',
            dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 5: Computational Safety Check - Range and extremes
# ============================================================================
print("  5. Computational safety check...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 5a: Distribution of study-specific effects (theta)
ax = axes[0, 0]
for j in range(J):
    ax.hist(theta_samples[:, j], bins=40, alpha=0.3, label=f'Study {j+1}')
ax.set_xlabel('Study-specific effect (theta_i)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Prior Predictive: Study-Specific Effects', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 5b: Range of y_pred across all studies
ax = axes[0, 1]
y_pred_flat = y_pred_samples.flatten()
ax.hist(y_pred_flat, bins=80, alpha=0.6, color='teal', edgecolor='black')
ax.axvline(y_pred_flat.min(), color='red', linestyle='--', label=f'Min: {y_pred_flat.min():.1f}')
ax.axvline(y_pred_flat.max(), color='red', linestyle='--', label=f'Max: {y_pred_flat.max():.1f}')
ax.set_xlabel('Predicted y values (all studies)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Prior Predictive: All y Values', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 5c: Extreme value check
ax = axes[1, 0]
abs_y_pred = np.abs(y_pred_flat)
ax.hist(abs_y_pred, bins=80, alpha=0.6, color='orange', edgecolor='black')
ax.axvline(100, color='red', linestyle='--', linewidth=2, label='Warning threshold: 100')
ax.axvline(1000, color='darkred', linestyle='--', linewidth=2, label='Critical threshold: 1000')
ax.set_xlabel('|y_pred| (absolute value)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Extreme Value Check', fontsize=12, fontweight='bold')
ax.set_xlim(0, min(200, abs_y_pred.max()))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 5d: Box plot comparison
ax = axes[1, 1]
positions = np.arange(1, J + 1)
bp = ax.boxplot([y_pred_samples[:, j] for j in range(J)],
                 positions=positions, widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.6),
                 medianprops=dict(color='darkblue', linewidth=2))
ax.scatter(positions, observed_y, color='red', s=100, zorder=10,
           label='Observed y', marker='D')
ax.set_xlabel('Study', fontsize=11)
ax.set_ylabel('y values', fontsize=11)
ax.set_title('Prior Predictive vs Observed (Boxplots)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Computational Safety and Range Diagnostics', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/computational_safety.png',
            dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 6: Summary Dashboard - Key findings at a glance
# ============================================================================
print("  6. Summary dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top row: Prior distributions
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(mu_prior, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax1.axvline(observed_pooled_effect, color='red', linewidth=2, linestyle='--')
ax1.set_xlabel('mu', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_title('Prior: mu ~ N(0, 25)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(tau_prior, bins=40, density=True, alpha=0.6, color='coral', edgecolor='black')
ax2.axvline(observed_tau, color='darkgreen', linewidth=2, linestyle='--')
ax2.set_xlabel('tau', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('Prior: tau ~ Half-N(0, 10)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(mu_prior, tau_prior, alpha=0.2, s=5, c='steelblue')
ax3.axvline(observed_pooled_effect, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
ax3.axhline(observed_tau, color='darkgreen', linewidth=1.5, linestyle='--', alpha=0.7)
ax3.set_xlabel('mu', fontsize=10)
ax3.set_ylabel('tau', fontsize=10)
ax3.set_title('Joint Prior', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Middle row: Prior predictive for selected studies
for idx, j in enumerate([0, 3, 4]):  # Studies 1, 4, 5
    ax = fig.add_subplot(gs[1, idx])
    ax.hist(y_pred_samples[:, j], bins=40, density=True, alpha=0.6,
            color='skyblue', edgecolor='black')
    ax.axvline(observed_y[j], color='red', linewidth=2, linestyle='--')
    q25, q75 = np.percentile(y_pred_samples[:, j], [25, 75])
    ax.axvspan(q25, q75, color='blue', alpha=0.15)
    percentile = stats.percentileofscore(y_pred_samples[:, j], observed_y[j])
    ax.set_xlabel(f'y_{j+1}', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Study {j+1}: Obs at {percentile:.1f}%', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Bottom row: Summary metrics
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(pooled_effect_samples, bins=40, density=True, alpha=0.6,
         color='mediumpurple', edgecolor='black')
ax7.axvline(observed_pooled_effect, color='red', linewidth=2, linestyle='--')
q25, q75 = np.percentile(pooled_effect_samples, [25, 75])
ax7.axvspan(q25, q75, color='darkblue', alpha=0.15)
percentile = stats.percentileofscore(pooled_effect_samples, observed_pooled_effect)
ax7.set_xlabel('Pooled effect', fontsize=10)
ax7.set_ylabel('Density', fontsize=10)
ax7.set_title(f'Pooled: Obs at {percentile:.1f}%', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 1])
positions = np.arange(1, J + 1)
bp = ax8.boxplot([y_pred_samples[:, j] for j in range(J)],
                  positions=positions, widths=0.5, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.6),
                  medianprops=dict(color='darkblue', linewidth=1.5))
ax8.scatter(positions, observed_y, color='red', s=50, zorder=10, marker='D')
ax8.set_xlabel('Study', fontsize=10)
ax8.set_ylabel('y values', fontsize=10)
ax8.set_title('All Studies: Prior vs Obs', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

ax9 = fig.add_subplot(gs[2, 2])
study_percentiles = [stats.percentileofscore(y_pred_samples[:, j], observed_y[j])
                     for j in range(J)]
colors = ['green' if 25 <= p <= 75 else 'orange' if 5 <= p <= 95 else 'red'
          for p in study_percentiles]
ax9.bar(range(1, J + 1), study_percentiles, color=colors, alpha=0.7, edgecolor='black')
ax9.axhspan(25, 75, color='green', alpha=0.1, label='Good (25-75%)')
ax9.axhspan(5, 25, color='orange', alpha=0.1, label='Marginal')
ax9.axhspan(75, 95, color='orange', alpha=0.1)
ax9.axhspan(0, 5, color='red', alpha=0.1, label='Bad (<5% or >95%)')
ax9.axhspan(95, 100, color='red', alpha=0.1)
ax9.set_xlabel('Study', fontsize=10)
ax9.set_ylabel('Percentile', fontsize=10)
ax9.set_title('Coverage Percentiles', fontsize=11, fontweight='bold')
ax9.set_ylim(0, 100)
ax9.legend(fontsize=7, loc='upper left')
ax9.grid(True, alpha=0.3, axis='y')

plt.suptitle('Prior Predictive Check: Summary Dashboard', fontsize=16, fontweight='bold')
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/summary_dashboard.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations created successfully!")
print("\nPlots saved to: /workspace/experiments/experiment_1/prior_predictive_check/plots/")
print("  1. parameter_plausibility.png")
print("  2. study_level_coverage.png")
print("  3. pooled_effect_coverage.png")
print("  4. hierarchical_structure_diagnostic.png")
print("  5. computational_safety.png")
print("  6. summary_dashboard.png")
