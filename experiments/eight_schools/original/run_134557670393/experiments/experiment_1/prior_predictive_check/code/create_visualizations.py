"""
Visualization for Prior Predictive Check
=========================================

Creates diagnostic visualizations to assess:
1. Parameter plausibility (are mu and tau reasonable?)
2. Prior predictive coverage (do predictions cover observed data?)
3. Study-level diagnostics (any problematic studies?)
4. Computational red flags (extreme values?)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'

# Load results
data = np.load('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_samples.npz')
mu_samples = data['mu_samples']
tau_samples = data['tau_samples']
theta_samples = data['theta_samples']
y_prior_pred = data['y_prior_pred']
y_obs = data['y_obs']
sigma_i = data['sigma_i']
N_PRIOR_SAMPLES = int(data['N_PRIOR_SAMPLES'])
N_STUDIES = int(data['N_STUDIES'])

output_dir = '/workspace/experiments/experiment_1/prior_predictive_check/plots'

print("Creating visualizations...")

# ============================================================================
# PLOT 1: Parameter Plausibility (Multi-panel view of joint prior behavior)
# ============================================================================
print("  1. Parameter plausibility diagnostic...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: mu prior distribution
ax = axes[0, 0]
ax.hist(mu_samples, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Prior mean')
ax.axvline(y_obs.mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Observed mean ({y_obs.mean():.1f})')
ax.set_xlabel('Overall Effect (mu)', fontsize=11)
ax.set_ylabel('Prior Sample Count', fontsize=11)
ax.set_title('A. Prior Distribution: Overall Effect (mu)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add range annotation
mu_q = np.percentile(mu_samples, [2.5, 97.5])
ax.text(0.05, 0.95, f'95% Prior Range:\n[{mu_q[0]:.1f}, {mu_q[1]:.1f}]',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: tau prior distribution
ax = axes[0, 1]
# Use log scale for tau due to heavy tail
tau_plot = tau_samples[tau_samples < 50]  # Focus on reasonable range
ax.hist(tau_plot, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
ax.axvline(np.median(tau_plot), color='red', linestyle='--', linewidth=2,
           label=f'Median ({np.median(tau_plot):.1f})')
ax.set_xlabel('Between-Study SD (tau)', fontsize=11)
ax.set_ylabel('Prior Sample Count', fontsize=11)
ax.set_title('B. Prior Distribution: Heterogeneity (tau < 50)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add interpretation guide
ax.text(0.55, 0.95, 'tau < 1: Homogeneous\n1-10: Moderate\n>10: High heterogeneity',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Panel C: Joint distribution mu vs tau (truncated for visibility)
ax = axes[1, 0]
tau_plot_mask = tau_samples < 30
ax.scatter(mu_samples[tau_plot_mask], tau_samples[tau_plot_mask],
           alpha=0.3, s=10, color='purple')
ax.set_xlabel('Overall Effect (mu)', fontsize=11)
ax.set_ylabel('Between-Study SD (tau)', fontsize=11)
ax.set_title('C. Joint Prior: mu vs tau (tau < 30)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(5, color='red', linestyle='--', alpha=0.5, label='tau = 5 (prior scale)')
ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='mu = 0 (prior mean)')
ax.legend()

# Panel D: Sample theta_i ranges for first 100 prior draws
ax = axes[1, 1]
n_show = 100
for i in range(n_show):
    theta_i = theta_samples[i, :]
    ax.plot([i, i], [theta_i.min(), theta_i.max()], 'k-', alpha=0.2, linewidth=1)
    ax.scatter([i]*N_STUDIES, theta_i, alpha=0.4, s=10, color='steelblue')

# Overlay observed data range
ax.axhline(y_obs.min(), color='red', linestyle='--', linewidth=2, label='Observed range')
ax.axhline(y_obs.max(), color='red', linestyle='--', linewidth=2)
ax.fill_between([0, n_show], y_obs.min(), y_obs.max(),
                alpha=0.2, color='red', label='Observed range')

ax.set_xlabel('Prior Sample Index', fontsize=11)
ax.set_ylabel('Study Effects (theta_i)', fontsize=11)
ax.set_title(f'D. Study Effects: First {n_show} Prior Draws', fontsize=12, fontweight='bold')
ax.set_ylim([-100, 100])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/parameter_plausibility.png', dpi=300, bbox_inches='tight')
print(f"     Saved: parameter_plausibility.png")
plt.close()

# ============================================================================
# PLOT 2: Prior Predictive Coverage (Main diagnostic for data plausibility)
# ============================================================================
print("  2. Prior predictive coverage diagnostic...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(N_STUDIES):
    ax = axes[j]

    # Plot prior predictive distribution for this study
    y_pred_study = y_prior_pred[:, j]

    # Histogram of prior predictions
    ax.hist(y_pred_study, bins=50, alpha=0.6, color='lightblue',
            edgecolor='black', density=True, label='Prior predictive')

    # Overlay normal approximation
    mu_pred = y_pred_study.mean()
    sd_pred = y_pred_study.std()
    x_range = np.linspace(y_pred_study.min(), y_pred_study.max(), 200)
    ax.plot(x_range, stats.norm.pdf(x_range, mu_pred, sd_pred),
            'b-', linewidth=2, label='Normal approx.')

    # Mark observed value
    ax.axvline(y_obs[j], color='red', linestyle='--', linewidth=3,
               label=f'Observed: {y_obs[j]}')

    # Mark 95% prior predictive interval
    ppi_lower = np.percentile(y_pred_study, 2.5)
    ppi_upper = np.percentile(y_pred_study, 97.5)
    ax.axvline(ppi_lower, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(ppi_upper, color='green', linestyle=':', linewidth=2, alpha=0.7,
               label='95% PPI')

    # Set reasonable x-axis limits
    xlim_lower = min(ppi_lower - 30, y_obs[j] - 10)
    xlim_upper = max(ppi_upper + 30, y_obs[j] + 10)
    ax.set_xlim([xlim_lower, xlim_upper])

    # Calculate percentile
    percentile = stats.percentileofscore(y_pred_study, y_obs[j])

    ax.set_xlabel(f'Study {j+1} (sigma={sigma_i[j]})', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Study {j+1}: Obs at {percentile:.1f}th percentile',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if j == 0:
        ax.legend(fontsize=8)

plt.suptitle('Prior Predictive Coverage: Study-by-Study',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{output_dir}/prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
print(f"     Saved: prior_predictive_coverage.png")
plt.close()

# ============================================================================
# PLOT 3: Overall Prior Predictive Distribution
# ============================================================================
print("  3. Overall prior predictive distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: All prior predictions overlaid
ax = axes[0]
for j in range(N_STUDIES):
    y_pred_study = y_prior_pred[:, j]
    ax.hist(y_pred_study, bins=50, alpha=0.15, color=f'C{j}',
            label=f'Study {j+1}', density=True)

# Overlay observed values
for j, y in enumerate(y_obs):
    ax.axvline(y, color=f'C{j}', linestyle='--', linewidth=2, alpha=0.8)

ax.set_xlabel('Observed Value (y_i)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('A. Prior Predictive by Study', fontsize=13, fontweight='bold')
ax.set_xlim([-150, 150])
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel B: Pooled prior predictive
ax = axes[1]
y_all = y_prior_pred.flatten()
ax.hist(y_all, bins=100, alpha=0.7, color='steelblue', edgecolor='black', density=True)

# Overlay observed values
for y in y_obs:
    ax.axvline(y, color='red', linestyle='--', linewidth=2, alpha=0.6)

# Mark observed range
ax.axvspan(y_obs.min(), y_obs.max(), alpha=0.2, color='red',
           label='Observed range')

ax.set_xlabel('All Observations', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('B. Pooled Prior Predictive Distribution', fontsize=13, fontweight='bold')
ax.set_xlim([-150, 150])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/overall_prior_predictive.png', dpi=300, bbox_inches='tight')
print(f"     Saved: overall_prior_predictive.png")
plt.close()

# ============================================================================
# PLOT 4: Computational Red Flags
# ============================================================================
print("  4. Computational red flags diagnostic...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Tau distribution with red flag zones
ax = axes[0, 0]
ax.hist(tau_samples, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvspan(0, 1, alpha=0.2, color='green', label='Homogeneous (<1)')
ax.axvspan(1, 10, alpha=0.2, color='yellow', label='Moderate (1-10)')
ax.axvspan(10, 50, alpha=0.2, color='orange', label='High (10-50)')
ax.axvspan(50, tau_samples.max(), alpha=0.3, color='red', label='Extreme (>50)')
ax.set_xlabel('Between-Study SD (tau)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Tau Distribution with Red Flag Zones', fontsize=12, fontweight='bold')
ax.set_xlim([0, min(100, tau_samples.max())])
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Extreme theta values
ax = axes[0, 1]
theta_flat = theta_samples.flatten()
extreme_mask = np.abs(theta_flat) < 200  # Focus on reasonable range
ax.hist(theta_flat[extreme_mask], bins=100, alpha=0.7, color='purple', edgecolor='black')
ax.axvspan(-50, 50, alpha=0.2, color='green', label='Plausible range')
ax.axvline(y_obs.min(), color='red', linestyle='--', linewidth=2, label='Observed range')
ax.axvline(y_obs.max(), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Study Effects (theta_i)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('B. Study Effects Distribution (|theta| < 200)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel C: Relationship between tau and theta spread
ax = axes[1, 0]
theta_ranges = np.ptp(theta_samples, axis=1)  # range per prior sample
mask = tau_samples < 50
ax.scatter(tau_samples[mask], theta_ranges[mask], alpha=0.3, s=20, color='darkgreen')
ax.set_xlabel('Between-Study SD (tau)', fontsize=11)
ax.set_ylabel('Range of theta_i in Sample', fontsize=11)
ax.set_title('C. Tau vs Theta Spread (tau < 50)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add reference line (theoretical: range ≈ 4*tau for normal)
tau_ref = np.linspace(0, 50, 100)
ax.plot(tau_ref, 4*tau_ref, 'r--', linewidth=2, label='Approx. 4*tau', alpha=0.7)
ax.legend()

# Panel D: Prior predictive ranges per dataset
ax = axes[1, 1]
y_ranges = np.ptp(y_prior_pred, axis=1)
ax.hist(y_ranges, bins=50, alpha=0.7, color='teal', edgecolor='black')
obs_range = np.ptp(y_obs)
ax.axvline(obs_range, color='red', linestyle='--', linewidth=3,
           label=f'Observed range: {obs_range}')
ax.set_xlabel('Range of y_i in Prior Predictive Dataset', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('D. Prior Predictive Data Ranges', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/computational_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"     Saved: computational_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 5: Prior Predictive Intervals Summary
# ============================================================================
print("  5. Prior predictive intervals summary...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate percentiles for each study
ppi_50 = np.percentile(y_prior_pred, [25, 75], axis=0)
ppi_95 = np.percentile(y_prior_pred, [2.5, 97.5], axis=0)
ppi_median = np.median(y_prior_pred, axis=0)

studies = np.arange(1, N_STUDIES + 1)

# Plot 95% intervals
for i in range(N_STUDIES):
    ax.plot([studies[i], studies[i]], [ppi_95[0, i], ppi_95[1, i]],
            'b-', linewidth=6, alpha=0.3, label='95% PPI' if i == 0 else '')

# Plot 50% intervals
for i in range(N_STUDIES):
    ax.plot([studies[i], studies[i]], [ppi_50[0, i], ppi_50[1, i]],
            'b-', linewidth=10, alpha=0.6, label='50% PPI' if i == 0 else '')

# Plot medians
ax.scatter(studies, ppi_median, color='blue', s=100, zorder=5,
           marker='o', label='Prior pred. median')

# Plot observed values
ax.scatter(studies, y_obs, color='red', s=150, zorder=6,
           marker='D', label='Observed', edgecolors='black', linewidth=2)

ax.set_xlabel('Study Number', fontsize=12)
ax.set_ylabel('Observed Value (y_i)', fontsize=12)
ax.set_title('Prior Predictive Intervals vs Observed Data', fontsize=13, fontweight='bold')
ax.set_xticks(studies)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/prior_predictive_intervals.png', dpi=300, bbox_inches='tight')
print(f"     Saved: prior_predictive_intervals.png")
plt.close()

print("\nAll visualizations created successfully!")
