"""
Prior Predictive Check Visualizations
Creates diagnostic plots for prior validation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load results
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"
data = np.load(OUTPUT_DIR / 'code' / 'prior_samples.npz')

# Extract data
beta0_prior = data['beta0_prior']
beta1_prior = data['beta1_prior']
sigma_prior = data['sigma_prior']
y_prior_pred_obs = data['y_prior_pred_obs']
mu_prior_pred_grid = data['mu_prior_pred_grid']
x_grid = data['x_grid']
x_obs = data['x_obs']
y_obs = data['y_obs']
mu_percentiles = data['mu_percentiles']
y_percentiles = data['y_percentiles']
percentiles = data['percentiles']

N_PRIOR_SAMPLES = len(beta0_prior)

print("=" * 80)
print("CREATING PRIOR PREDICTIVE CHECK VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# PLOT 1: Prior Predictive Coverage - Main Diagnostic
# ============================================================================
print("\nCreating Plot 1: Prior Predictive Coverage...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot a sample of prior predictive curves (semi-transparent)
n_curves_to_plot = 100
indices = np.random.choice(N_PRIOR_SAMPLES, n_curves_to_plot, replace=False)

for idx in indices:
    ax.plot(x_grid, mu_prior_pred_grid[idx, :],
            color='steelblue', alpha=0.05, linewidth=0.5)

# Plot prior predictive intervals
ax.fill_between(x_grid, y_percentiles[0, :], y_percentiles[-1, :],
                alpha=0.15, color='steelblue', label='95% Prior Predictive Interval')
ax.fill_between(x_grid, y_percentiles[1, :], y_percentiles[-2, :],
                alpha=0.15, color='steelblue', label='90% Prior Predictive Interval')
ax.fill_between(x_grid, y_percentiles[2, :], y_percentiles[-3, :],
                alpha=0.25, color='steelblue', label='50% Prior Predictive Interval')

# Plot median prior predictive
median_idx = len(percentiles) // 2
ax.plot(x_grid, mu_percentiles[median_idx, :],
        color='darkblue', linewidth=2, label='Prior Predictive Median', linestyle='--')

# Plot observed data
ax.scatter(x_obs, y_obs, color='red', s=60, alpha=0.8,
          edgecolors='darkred', linewidth=1.5, label='Observed Data', zorder=10)

# Add observed range shading
y_obs_min, y_obs_max = y_obs.min(), y_obs.max()
ax.axhline(y_obs_min, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(y_obs_max, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax.fill_between([x_obs.min(), x_obs.max()], y_obs_min, y_obs_max,
                alpha=0.05, color='red')

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title('Prior Predictive Coverage: Do Priors Generate Plausible Data?',
            fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'prior_predictive_coverage.png'}")
plt.close()

# ============================================================================
# PLOT 2: Parameter Prior Distributions
# ============================================================================
print("\nCreating Plot 2: Parameter Prior Distributions...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Beta0 prior
axes[0].hist(beta0_prior, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(beta0_prior.mean(), color='darkblue', linestyle='--', linewidth=2, label='Mean')
axes[0].axvline(np.percentile(beta0_prior, 2.5), color='gray', linestyle=':', linewidth=1.5, label='95% CI')
axes[0].axvline(np.percentile(beta0_prior, 97.5), color='gray', linestyle=':', linewidth=1.5)
axes[0].set_xlabel('β₀ (Intercept)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0].set_title(f'Prior: β₀ ~ Normal(1.73, 0.5)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Beta1 prior
axes[1].hist(beta1_prior, bins=40, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
axes[1].axvline(beta1_prior.mean(), color='darkgreen', linestyle='--', linewidth=2, label='Mean')
axes[1].axvline(np.percentile(beta1_prior, 2.5), color='gray', linestyle=':', linewidth=1.5, label='95% CI')
axes[1].axvline(np.percentile(beta1_prior, 97.5), color='gray', linestyle=':', linewidth=1.5)
axes[1].axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5, label='Zero')
axes[1].set_xlabel('β₁ (Log-x Slope)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[1].set_title(f'Prior: β₁ ~ Normal(0.28, 0.15)', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Sigma prior
axes[2].hist(sigma_prior, bins=40, density=True, alpha=0.7, color='coral', edgecolor='black')
axes[2].axvline(sigma_prior.mean(), color='darkred', linestyle='--', linewidth=2, label='Mean')
axes[2].axvline(np.percentile(sigma_prior, 95), color='gray', linestyle=':', linewidth=1.5, label='95th %ile')
axes[2].axvline(y_obs.std(), color='purple', linestyle='-', linewidth=2, label='Observed Y SD')
axes[2].set_xlabel('σ (Noise)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[2].set_title(f'Prior: σ ~ Exponential(5)', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_plausibility.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'parameter_plausibility.png'}")
plt.close()

# ============================================================================
# PLOT 3: Prior Sensitivity Analysis
# ============================================================================
print("\nCreating Plot 3: Prior Sensitivity Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Joint distribution of β₀ and β₁
axes[0, 0].hexbin(beta0_prior, beta1_prior, gridsize=30, cmap='Blues', alpha=0.8)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='β₁=0')
axes[0, 0].set_xlabel('β₀ (Intercept)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('β₁ (Slope)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('A) Joint Prior: β₀ vs β₁', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel B: Prior predictive ranges at each observation
y_prior_mins = y_prior_pred_obs.min(axis=0)
y_prior_maxs = y_prior_pred_obs.max(axis=0)
y_prior_means = y_prior_pred_obs.mean(axis=0)

axes[0, 1].fill_between(range(len(x_obs)), y_prior_mins, y_prior_maxs,
                        alpha=0.3, color='steelblue', label='Prior Predictive Range')
axes[0, 1].plot(range(len(x_obs)), y_prior_means, 'o-',
               color='darkblue', markersize=3, label='Prior Predictive Mean')
axes[0, 1].scatter(range(len(x_obs)), y_obs, color='red', s=50,
                  alpha=0.8, label='Observed Y', zorder=10)
axes[0, 1].set_xlabel('Observation Index', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Y', fontsize=11, fontweight='bold')
axes[0, 1].set_title('B) Prior Predictive Range vs Observed', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Panel C: Distribution of slope signs
slope_categories = ['Negative\n(β₁ < 0)', 'Positive\n(β₁ > 0)']
slope_counts = [np.sum(beta1_prior < 0), np.sum(beta1_prior > 0)]
colors_slope = ['salmon', 'lightgreen']

bars = axes[1, 0].bar(slope_categories, slope_counts, color=colors_slope,
                     edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1, 0].set_title('C) Prior Belief about Relationship Direction',
                    fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, (bar, count) in enumerate(zip(bars, slope_counts)):
    height = bar.get_height()
    pct = 100 * count / N_PRIOR_SAMPLES
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%\n(n={count})',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel D: Sigma vs prediction uncertainty
y_prior_sds = y_prior_pred_obs.std(axis=1)
axes[1, 1].scatter(sigma_prior, y_prior_sds, alpha=0.3, s=20, color='coral')
axes[1, 1].plot([0, sigma_prior.max()], [0, sigma_prior.max()],
               'k--', linewidth=1, alpha=0.5, label='y=x reference')
axes[1, 1].axhline(y_obs.std(), color='purple', linestyle='-',
                  linewidth=2, label='Observed Y SD', alpha=0.7)
axes[1, 1].set_xlabel('σ (Prior Sample)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('SD(Prior Predictions)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('D) Noise Parameter vs Prediction Variability',
                    fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'prior_sensitivity_analysis.png'}")
plt.close()

# ============================================================================
# PLOT 4: Extreme Cases Diagnostic
# ============================================================================
print("\nCreating Plot 4: Extreme Cases Diagnostic...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Find extreme prior draws
y_prior_ranges = y_prior_pred_obs.max(axis=1) - y_prior_pred_obs.min(axis=1)
extreme_narrow_idx = np.argmin(y_prior_ranges)
extreme_wide_idx = np.argmax(y_prior_ranges)

# Panel A: Most narrow prior prediction
axes[0].fill_between(x_obs,
                     y_prior_pred_obs[extreme_narrow_idx, :] - sigma_prior[extreme_narrow_idx],
                     y_prior_pred_obs[extreme_narrow_idx, :] + sigma_prior[extreme_narrow_idx],
                     alpha=0.3, color='steelblue', label='±1σ band')
axes[0].plot(x_obs, y_prior_pred_obs[extreme_narrow_idx, :],
            'o-', color='darkblue', markersize=5, label='Prior Prediction')
axes[0].scatter(x_obs, y_obs, color='red', s=50, alpha=0.8,
               edgecolors='darkred', label='Observed', zorder=10)
axes[0].set_xlabel('x', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Y', fontsize=11, fontweight='bold')
axes[0].set_title(f'Narrowest Prior Draw\nβ₀={beta0_prior[extreme_narrow_idx]:.2f}, '
                 f'β₁={beta1_prior[extreme_narrow_idx]:.2f}, '
                 f'σ={sigma_prior[extreme_narrow_idx]:.3f}',
                 fontsize=11, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel B: Most wide prior prediction
axes[1].fill_between(x_obs,
                     y_prior_pred_obs[extreme_wide_idx, :] - sigma_prior[extreme_wide_idx],
                     y_prior_pred_obs[extreme_wide_idx, :] + sigma_prior[extreme_wide_idx],
                     alpha=0.3, color='coral', label='±1σ band')
axes[1].plot(x_obs, y_prior_pred_obs[extreme_wide_idx, :],
            'o-', color='darkred', markersize=5, label='Prior Prediction')
axes[1].scatter(x_obs, y_obs, color='red', s=50, alpha=0.8,
               edgecolors='darkred', label='Observed', zorder=10)
axes[1].set_xlabel('x', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Y', fontsize=11, fontweight='bold')
axes[1].set_title(f'Widest Prior Draw\nβ₀={beta0_prior[extreme_wide_idx]:.2f}, '
                 f'β₁={beta1_prior[extreme_wide_idx]:.2f}, '
                 f'σ={sigma_prior[extreme_wide_idx]:.3f}',
                 fontsize=11, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'extreme_cases_diagnostic.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'extreme_cases_diagnostic.png'}")
plt.close()

# ============================================================================
# PLOT 5: Quantitative Coverage Assessment
# ============================================================================
print("\nCreating Plot 5: Quantitative Coverage Assessment...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Histogram of predicted Y values across all priors
axes[0, 0].hist(y_prior_pred_obs.flatten(), bins=60, density=True,
               alpha=0.6, color='steelblue', edgecolor='black', label='Prior Predictive')
axes[0, 0].hist(y_obs, bins=15, density=True, alpha=0.6,
               color='red', edgecolor='darkred', label='Observed')
axes[0, 0].axvline(y_obs.mean(), color='red', linestyle='--', linewidth=2, label='Obs Mean')
axes[0, 0].set_xlabel('Y', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0, 0].set_title('A) Prior Predictive vs Observed Distribution',
                    fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel B: Q-Q plot style comparison
y_obs_sorted = np.sort(y_obs)
y_prior_median = np.median(y_prior_pred_obs, axis=0)
obs_indices = np.argsort(y_obs)
y_prior_median_sorted = y_prior_median[obs_indices]

axes[0, 1].scatter(y_obs_sorted, y_prior_median_sorted, alpha=0.7, s=60, color='steelblue')
lim_min = min(y_obs.min(), y_prior_median.min()) - 0.1
lim_max = max(y_obs.max(), y_prior_median.max()) + 0.1
axes[0, 1].plot([lim_min, lim_max], [lim_min, lim_max],
               'k--', linewidth=2, alpha=0.5, label='Perfect Match')
axes[0, 1].set_xlabel('Observed Y (sorted)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Prior Predictive Median', fontsize=11, fontweight='bold')
axes[0, 1].set_title('B) Observed vs Prior Predictive Median',
                    fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([lim_min, lim_max])
axes[0, 1].set_ylim([lim_min, lim_max])

# Panel C: Coverage by percentile
coverage_levels = [50, 80, 90, 95, 99]
coverage_props = []

for level in coverage_levels:
    lower = (100 - level) / 2
    upper = 100 - lower
    y_lower = np.percentile(y_prior_pred_obs, lower, axis=0)
    y_upper = np.percentile(y_prior_pred_obs, upper, axis=0)
    in_interval = np.sum((y_obs >= y_lower) & (y_obs <= y_upper))
    coverage_props.append(100 * in_interval / len(y_obs))

axes[1, 0].plot(coverage_levels, coverage_levels, 'k--', linewidth=2,
               alpha=0.5, label='Nominal Coverage')
axes[1, 0].plot(coverage_levels, coverage_props, 'o-', color='steelblue',
               linewidth=2, markersize=8, label='Actual Coverage')
axes[1, 0].set_xlabel('Nominal Coverage Level (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Actual Coverage (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('C) Prior Predictive Interval Coverage',
                    fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim([45, 100])
axes[1, 0].set_ylim([45, 100])

# Panel D: Residuals from prior predictive median
residuals = y_obs - y_prior_median[np.argsort(x_obs)]
axes[1, 1].scatter(x_obs, residuals, alpha=0.7, s=60, color='steelblue', edgecolor='black')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 1].axhline(residuals.std(), color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
axes[1, 1].axhline(-residuals.std(), color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
axes[1, 1].set_xlabel('x', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Residual (Obs - Prior Pred Median)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('D) Residuals from Prior Predictive Median',
                    fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_assessment.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {PLOTS_DIR / 'coverage_assessment.png'}")
plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print(f"\nAll plots saved to: {PLOTS_DIR}")
print("\nGenerated plots:")
print("  1. prior_predictive_coverage.png - Main diagnostic of prior predictions vs data")
print("  2. parameter_plausibility.png - Individual parameter prior distributions")
print("  3. prior_sensitivity_analysis.png - Joint behavior and sensitivity metrics")
print("  4. extreme_cases_diagnostic.png - Most narrow and wide prior scenarios")
print("  5. coverage_assessment.png - Quantitative coverage and fit diagnostics")
