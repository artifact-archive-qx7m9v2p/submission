"""
Create visualizations for prior predictive check
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load results
OUTPUT_DIR = "/workspace/experiments/experiment_2/prior_predictive_check"
data = np.load(f"{OUTPUT_DIR}/code/prior_predictive_samples.npz")

beta_0 = data['beta_0']
beta_1 = data['beta_1']
gamma_0 = data['gamma_0']
gamma_1 = data['gamma_1']
y_simulated = data['y_simulated']
mu_simulated = data['mu_simulated']
sigma_simulated = data['sigma_simulated']
x_obs = data['x_obs']
y_obs = data['y_obs']
variance_ratio = data['variance_ratio']
mu_change = data['mu_change']

# Load observed data
obs_data = pd.read_csv("/workspace/data/data.csv")

print("Creating visualizations...")

# ============================================================
# PLOT 1: Parameter Prior Distributions
# ============================================================
print("Creating parameter_distributions.png...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Beta 0
ax = axes[0, 0]
ax.hist(beta_0, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
x_range = np.linspace(beta_0.min(), beta_0.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, 1.8, 0.5), 'r-', lw=2, label='Prior: N(1.8, 0.5)')
ax.axvline(beta_0.mean(), color='darkblue', linestyle='--', lw=2, label=f'Sample mean: {beta_0.mean():.2f}')
ax.set_xlabel('beta_0 (Intercept)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution: Intercept', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Beta 1
ax = axes[0, 1]
ax.hist(beta_1, bins=50, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
x_range = np.linspace(beta_1.min(), beta_1.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, 0.3, 0.2), 'r-', lw=2, label='Prior: N(0.3, 0.2)')
ax.axvline(beta_1.mean(), color='darkgreen', linestyle='--', lw=2, label=f'Sample mean: {beta_1.mean():.2f}')
ax.axvline(0, color='black', linestyle=':', lw=1, alpha=0.5)
ax.set_xlabel('beta_1 (Log-x slope)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution: Log-Linear Slope', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Gamma 0
ax = axes[1, 0]
ax.hist(gamma_0, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
x_range = np.linspace(gamma_0.min(), gamma_0.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, -2, 1), 'r-', lw=2, label='Prior: N(-2, 1)')
ax.axvline(gamma_0.mean(), color='darkred', linestyle='--', lw=2, label=f'Sample mean: {gamma_0.mean():.2f}')
ax.set_xlabel('gamma_0 (Log-sigma intercept)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution: Variance Intercept', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Gamma 1
ax = axes[1, 1]
ax.hist(gamma_1, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
x_range = np.linspace(gamma_1.min(), gamma_1.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, -0.05, 0.05), 'r-', lw=2, label='Prior: N(-0.05, 0.05)')
ax.axvline(gamma_1.mean(), color='indigo', linestyle='--', lw=2, label=f'Sample mean: {gamma_1.mean():.2f}')
ax.axvline(0, color='black', linestyle=':', lw=2, alpha=0.7, label='Zero (constant variance)')
n_positive = np.sum(gamma_1 > 0)
ax.text(0.05, 0.95, f'{n_positive} samples > 0 ({100*n_positive/len(gamma_1):.1f}%)',
        transform=ax.transAxes, verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('gamma_1 (Variance-x slope)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Distribution: Heteroscedasticity Parameter', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/parameter_distributions.png")
print("  Saved parameter_distributions.png")
plt.close()

# ============================================================
# PLOT 2: Prior Predictive Coverage
# ============================================================
print("Creating prior_predictive_coverage.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Top left: Sample of prior predictive trajectories
ax = axes[0, 0]
n_show = 100
for i in range(n_show):
    idx = np.random.randint(0, len(y_simulated))
    alpha = 0.1 if i < n_show - 1 else 0.1
    ax.plot(x_obs, y_simulated[idx, :], '-', color='steelblue', alpha=alpha, lw=0.5)

ax.plot(x_obs, y_obs, 'ro-', lw=2, markersize=6, label='Observed data', zorder=100)
ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title('Prior Predictive Trajectories (100 samples)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# Top right: Coverage bands
ax = axes[0, 1]
percentiles = [5, 25, 50, 75, 95]
colors_bands = ['#d6e9f7', '#a3d5ed', '#6baed6']

# Plot percentile bands
p5 = np.percentile(y_simulated, 5, axis=0)
p25 = np.percentile(y_simulated, 25, axis=0)
p50 = np.percentile(y_simulated, 50, axis=0)
p75 = np.percentile(y_simulated, 75, axis=0)
p95 = np.percentile(y_simulated, 95, axis=0)

ax.fill_between(x_obs, p5, p95, alpha=0.3, color=colors_bands[0], label='5-95th percentile')
ax.fill_between(x_obs, p25, p75, alpha=0.5, color=colors_bands[1], label='25-75th percentile')
ax.plot(x_obs, p50, 'b-', lw=2, label='Median')
ax.plot(x_obs, y_obs, 'ro-', lw=2, markersize=6, label='Observed data', zorder=100)

# Add target range
ax.axhspan(0.5, 5.0, alpha=0.1, color='green', label='Target range [0.5, 5.0]')

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title('Prior Predictive Percentile Bands', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom left: Distribution of predicted ranges
ax = axes[1, 0]
y_min = y_simulated.min(axis=1)
y_max = y_simulated.max(axis=1)

ax.scatter(y_min, y_max, alpha=0.3, s=10, color='steelblue')
ax.axvline(y_obs.min(), color='red', linestyle='--', lw=2, label=f'Obs min: {y_obs.min():.2f}')
ax.axhline(y_obs.max(), color='red', linestyle='--', lw=2, label=f'Obs max: {y_obs.max():.2f}')

# Add target box
ax.axvspan(0.5, 5.0, alpha=0.1, color='green')
ax.axhspan(0.5, 5.0, alpha=0.1, color='green')

# Count coverage
n_cover = np.sum((y_min <= y_obs.min()) & (y_max >= y_obs.max()))
ax.text(0.05, 0.95, f'{n_cover}/{len(y_min)} samples\ncover observed range\n({100*n_cover/len(y_min):.1f}%)',
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Min(Y) in sample', fontsize=12, fontweight='bold')
ax.set_ylabel('Max(Y) in sample', fontsize=12, fontweight='bold')
ax.set_title('Range of Generated Data', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom right: Histogram of y values
ax = axes[1, 1]
ax.hist(y_simulated.flatten(), bins=100, density=True, alpha=0.7, color='steelblue',
        edgecolor='black', label='Prior predictive')
ax.hist(y_obs, bins=15, density=True, alpha=0.7, color='red',
        edgecolor='black', label='Observed data')
ax.axvspan(0.5, 5.0, alpha=0.1, color='green', label='Target range')
ax.set_xlabel('Y', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Y Values', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/prior_predictive_coverage.png")
print("  Saved prior_predictive_coverage.png")
plt.close()

# ============================================================
# PLOT 3: Variance Structure Diagnostic
# ============================================================
print("Creating variance_structure_diagnostic.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Top left: Sigma trajectories
ax = axes[0, 0]
n_show = 50
for i in range(n_show):
    idx = np.random.randint(0, len(sigma_simulated))
    ax.plot(x_obs, sigma_simulated[idx, :], '-', color='purple', alpha=0.2, lw=1)

# Add percentile bands
sigma_p50 = np.percentile(sigma_simulated, 50, axis=0)
sigma_p5 = np.percentile(sigma_simulated, 5, axis=0)
sigma_p95 = np.percentile(sigma_simulated, 95, axis=0)
ax.plot(x_obs, sigma_p50, 'b-', lw=3, label='Median sigma')
ax.fill_between(x_obs, sigma_p5, sigma_p95, alpha=0.3, color='steelblue', label='5-95th percentile')

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('sigma (Standard Deviation)', fontsize=12, fontweight='bold')
ax.set_title('Prior for Standard Deviation Structure', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Top right: Variance ratio distribution
ax = axes[1, 0]
ax.hist(np.log10(variance_ratio), bins=50, density=True, alpha=0.7, color='teal', edgecolor='black')
ax.axvline(np.log10(8.83), color='red', linestyle='--', lw=2, label='Observed ratio: 8.8x')
ax.axvline(0, color='black', linestyle=':', lw=2, alpha=0.7, label='Ratio = 1 (homoscedastic)')

median_ratio = np.median(variance_ratio)
ax.axvline(np.log10(median_ratio), color='blue', linestyle='--', lw=2,
           label=f'Median: {median_ratio:.1f}x')

n_decreasing = np.sum(variance_ratio > 1)
ax.text(0.05, 0.95, f'{n_decreasing}/{len(variance_ratio)} samples\nhave decreasing variance\n({100*n_decreasing/len(variance_ratio):.1f}%)',
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('log10(Variance Ratio: low-x / high-x)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('Heteroscedasticity: Variance Ratio Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Middle left: Scatter of sigma at endpoints
ax = axes[0, 1]
sigma_at_min = sigma_simulated[:, 0]
sigma_at_max = sigma_simulated[:, -1]

ax.scatter(sigma_at_min, sigma_at_max, alpha=0.3, s=15, color='teal')
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', lw=2, alpha=0.5, label='Equal variance')

n_decreasing_scatter = np.sum(sigma_at_min > sigma_at_max)
ax.text(0.05, 0.95, f'{n_decreasing_scatter}/{len(sigma_at_min)} samples\nwith sigma(x=1) > sigma(x=31.5)',
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('sigma at x=1 (low x)', fontsize=12, fontweight='bold')
ax.set_ylabel('sigma at x=31.5 (high x)', fontsize=12, fontweight='bold')
ax.set_title('Standard Deviation at Endpoints', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom right: Relationship between gamma_1 and variance ratio
ax = axes[1, 1]
scatter = ax.scatter(gamma_1, np.log10(variance_ratio), alpha=0.3, s=15, c=gamma_0, cmap='viridis')
ax.axhline(0, color='black', linestyle=':', lw=2, alpha=0.7, label='Ratio = 1')
ax.axvline(0, color='black', linestyle=':', lw=2, alpha=0.7)
ax.axhline(np.log10(8.83), color='red', linestyle='--', lw=2, label='Observed ratio')

ax.set_xlabel('gamma_1 (variance-x slope parameter)', fontsize=12, fontweight='bold')
ax.set_ylabel('log10(Variance Ratio)', fontsize=12, fontweight='bold')
ax.set_title('Parameter vs Variance Ratio', fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, label='gamma_0')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/variance_structure_diagnostic.png")
print("  Saved variance_structure_diagnostic.png")
plt.close()

# ============================================================
# PLOT 4: Mean Structure Diagnostic
# ============================================================
print("Creating mean_structure_diagnostic.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Top left: Mean trajectories
ax = axes[0, 0]
n_show = 100
for i in range(n_show):
    idx = np.random.randint(0, len(mu_simulated))
    ax.plot(x_obs, mu_simulated[idx, :], '-', color='steelblue', alpha=0.1, lw=0.5)

ax.plot(x_obs, y_obs, 'ro-', lw=2, markersize=6, label='Observed data', zorder=100)

# Add median
mu_p50 = np.percentile(mu_simulated, 50, axis=0)
ax.plot(x_obs, mu_p50, 'b-', lw=2, label='Median prior mean', zorder=50)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean (mu)', fontsize=12, fontweight='bold')
ax.set_title('Prior for Mean Structure', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Top right: Mean vs log(x)
ax = axes[0, 1]
log_x_range = np.log(x_obs)
n_show = 50
for i in range(n_show):
    idx = np.random.randint(0, len(mu_simulated))
    ax.plot(log_x_range, mu_simulated[idx, :], '-', color='steelblue', alpha=0.2, lw=0.5)

ax.plot(log_x_range, y_obs, 'ro-', lw=2, markersize=6, label='Observed data', zorder=100)
ax.plot(log_x_range, mu_p50, 'b-', lw=2, label='Median prior mean', zorder=50)

ax.set_xlabel('log(x)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean (mu)', fontsize=12, fontweight='bold')
ax.set_title('Mean vs Log(x) - Linear Relationship', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom left: Distribution of mean change
ax = axes[1, 0]
ax.hist(mu_change, bins=50, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
obs_change = y_obs[-1] - y_obs[0]
ax.axvline(obs_change, color='red', linestyle='--', lw=2, label=f'Observed change: {obs_change:.2f}')
ax.axvline(np.median(mu_change), color='blue', linestyle='--', lw=2,
           label=f'Median: {np.median(mu_change):.2f}')

ax.set_xlabel('Mean change from x=1 to x=31.5', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Mean Change', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom right: Joint prior on beta parameters
ax = axes[1, 1]
scatter = ax.scatter(beta_0, beta_1, alpha=0.3, s=10, color='steelblue')
ax.axhline(0, color='black', linestyle=':', lw=2, alpha=0.5)
ax.axvline(1.8, color='red', linestyle='--', lw=1, alpha=0.5, label='Prior means')
ax.axhline(0.3, color='red', linestyle='--', lw=1, alpha=0.5)

# Add correlation info
corr = np.corrcoef(beta_0, beta_1)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\n(should be ~0)',
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('beta_0 (Intercept)', fontsize=12, fontweight='bold')
ax.set_ylabel('beta_1 (Log-x slope)', fontsize=12, fontweight='bold')
ax.set_title('Joint Prior: Mean Parameters', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/mean_structure_diagnostic.png")
print("  Saved mean_structure_diagnostic.png")
plt.close()

# ============================================================
# PLOT 5: Red Flags and Edge Cases
# ============================================================
print("Creating edge_cases_diagnostic.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Top left: Distribution of extreme values
ax = axes[0, 0]
y_min_per_sample = y_simulated.min(axis=1)
y_max_per_sample = y_simulated.max(axis=1)

ax.hist(y_min_per_sample, bins=50, alpha=0.6, color='blue', edgecolor='black', label='Min Y per sample')
ax.hist(y_max_per_sample, bins=50, alpha=0.6, color='red', edgecolor='black', label='Max Y per sample')
ax.axvline(0, color='black', linestyle='--', lw=2, label='Zero (physical constraint)')
ax.axvline(y_obs.min(), color='green', linestyle='--', lw=2, alpha=0.7, label='Observed range')
ax.axvline(y_obs.max(), color='green', linestyle='--', lw=2, alpha=0.7)

n_negative = np.sum(y_min_per_sample < 0)
ax.text(0.05, 0.95, f'{n_negative}/{len(y_min_per_sample)} samples\nwith negative Y\n({100*n_negative/len(y_min_per_sample):.1f}%)',
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Y value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Extreme Y Values', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Top right: Maximum sigma values
ax = axes[0, 1]
max_sigma_per_sample = sigma_simulated.max(axis=1)

ax.hist(max_sigma_per_sample, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(10, color='red', linestyle='--', lw=2, label='Extreme threshold (10)')
ax.axvline(np.median(max_sigma_per_sample), color='blue', linestyle='--', lw=2,
           label=f'Median: {np.median(max_sigma_per_sample):.2f}')

n_extreme = np.sum(max_sigma_per_sample > 10)
ax.text(0.05, 0.95, f'{n_extreme}/{len(max_sigma_per_sample)} samples\nwith sigma > 10\n({100*n_extreme/len(max_sigma_per_sample):.1f}%)',
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Maximum sigma in sample', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Maximum Standard Deviation', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, min(max_sigma_per_sample.max(), 20))

# Bottom left: Coefficient of variation over x
ax = axes[1, 0]
cv_simulated = sigma_simulated / np.abs(mu_simulated + 1e-10)  # Add small constant to avoid division by zero

cv_p50 = np.percentile(cv_simulated, 50, axis=0)
cv_p5 = np.percentile(cv_simulated, 5, axis=0)
cv_p95 = np.percentile(cv_simulated, 95, axis=0)

ax.plot(x_obs, cv_p50, 'b-', lw=2, label='Median CV')
ax.fill_between(x_obs, cv_p5, cv_p95, alpha=0.3, color='steelblue', label='5-95th percentile')

# Calculate observed CV
obs_cv_by_bin = obs_data.groupby(pd.qcut(obs_data['x'], q=5, duplicates='drop')).agg({
    'Y': ['mean', 'std'],
    'x': 'mean'
})
obs_cv = obs_cv_by_bin['Y']['std'] / obs_cv_by_bin['Y']['mean']
obs_x = obs_cv_by_bin['x']['mean']
ax.plot(obs_x, obs_cv, 'ro-', lw=2, markersize=8, label='Observed CV (binned)', zorder=100)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (sigma/mu)', fontsize=12, fontweight='bold')
ax.set_title('Relative Variability Structure', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom right: Problematic parameter combinations
ax = axes[1, 1]

# Color points by whether they produce reasonable data
reasonable = np.ones(len(gamma_0))
reasonable[y_min_per_sample < 0] = 0
reasonable[max_sigma_per_sample > 10] = 0

colors = ['red' if r == 0 else 'steelblue' for r in reasonable]
ax.scatter(gamma_0, gamma_1, alpha=0.4, s=15, c=colors)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', alpha=0.4, label='Reasonable'),
                   Patch(facecolor='red', alpha=0.4, label='Problematic')]
ax.legend(handles=legend_elements, fontsize=10)

n_problematic = np.sum(reasonable == 0)
ax.text(0.05, 0.95, f'{n_problematic}/{len(reasonable)} samples\nare problematic\n({100*n_problematic/len(reasonable):.1f}%)',
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.axhline(0, color='black', linestyle=':', lw=2, alpha=0.5)
ax.set_xlabel('gamma_0 (log-sigma intercept)', fontsize=12, fontweight='bold')
ax.set_ylabel('gamma_1 (variance-x slope)', fontsize=12, fontweight='bold')
ax.set_title('Problematic Parameter Regions', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/edge_cases_diagnostic.png")
print("  Saved edge_cases_diagnostic.png")
plt.close()

print("\nAll visualizations created successfully!")
print(f"Plots saved to: {OUTPUT_DIR}/plots/")
