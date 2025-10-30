"""
Prior Predictive Check for Experiment 1: Robust Logarithmic Regression

This script:
1. Samples from priors without conditioning on data (using pure NumPy/SciPy)
2. Generates synthetic datasets from prior predictive distribution
3. Assesses plausibility against domain knowledge
4. Creates diagnostic visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gamma, t as student_t, halfcauchy
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"

# Ensure directories exist
PLOTS_DIR.mkdir(exist_ok=True)

# Data context (from specification)
N_OBS = 27
X_MIN, X_MAX = 1.0, 31.5
Y_MIN, Y_MAX = 1.77, 2.72
Y_MEAN, Y_SD = 2.33, 0.27

# Generate x values matching the observed range
x_observed = np.linspace(X_MIN, X_MAX, N_OBS)

# Additional x values for visualization (including extrapolation)
x_extended = np.linspace(0.5, 50, 100)

print("=" * 80)
print("PRIOR PREDICTIVE CHECK: Experiment 1 - Robust Logarithmic Regression")
print("=" * 80)
print()

# ============================================================================
# 1. Sample from Prior Predictive Distribution
# ============================================================================

print("Step 1: Sampling from prior predictive distribution (1000 draws)...")

N_SAMPLES = 1000

# Sample from priors
alpha_prior = norm.rvs(loc=2.0, scale=0.5, size=N_SAMPLES)
beta_prior = norm.rvs(loc=0.3, scale=0.3, size=N_SAMPLES)
c_prior = gamma.rvs(a=2, scale=1/2, size=N_SAMPLES)  # Gamma(shape=2, rate=2) = Gamma(a=2, scale=1/rate)
nu_prior = gamma.rvs(a=2, scale=1/0.1, size=N_SAMPLES)  # Gamma(2, 0.1)
sigma_prior = halfcauchy.rvs(loc=0, scale=0.2, size=N_SAMPLES)

print("Sampling complete.")
print()

# ============================================================================
# 2. Generate Predictions
# ============================================================================

print("Step 2: Generating prior predictive samples...")

# Initialize arrays
y_sim = np.zeros((N_SAMPLES, N_OBS))
mu = np.zeros((N_SAMPLES, N_OBS))

# Generate predictions for each prior sample
for i in range(N_SAMPLES):
    # Compute mean function
    mu[i, :] = alpha_prior[i] + beta_prior[i] * np.log(x_observed + c_prior[i])

    # Sample from Student-t likelihood
    for j in range(N_OBS):
        y_sim[i, j] = student_t.rvs(df=nu_prior[i], loc=mu[i, j], scale=sigma_prior[i])

# Compute diagnostics
min_y_sim = np.min(y_sim, axis=1)
max_y_sim = np.max(y_sim, axis=1)
mean_y_sim = np.mean(y_sim, axis=1)
sd_y_sim = np.std(y_sim, axis=1)

# Monotonicity: mu[N-1] - mu[0]
monotonic_increase = mu[:, -1] - mu[:, 0]

# Extrapolation: predictions at x=50
mu_x50 = alpha_prior + beta_prior * np.log(50 + c_prior)
y_x50 = np.array([student_t.rvs(df=nu_prior[i], loc=mu_x50[i], scale=sigma_prior[i]) for i in range(N_SAMPLES)])

print(f"Generated {N_SAMPLES} prior predictive datasets.")
print()

# ============================================================================
# 3. Compute Diagnostics
# ============================================================================

print("Step 3: Computing diagnostic statistics...")
print()

# Prior parameter summaries
print("PRIOR PARAMETER SUMMARIES:")
print("-" * 80)
params = {
    'alpha (intercept)': alpha_prior,
    'beta (slope)': beta_prior,
    'c (log shift)': c_prior,
    'nu (df)': nu_prior,
    'sigma (scale)': sigma_prior
}

for name, values in params.items():
    print(f"{name:20s}: mean={np.mean(values):7.3f}, "
          f"sd={np.std(values):6.3f}, "
          f"median={np.median(values):7.3f}, "
          f"95% CI=[{np.percentile(values, 2.5):6.3f}, {np.percentile(values, 97.5):6.3f}]")
print()

# Prior predictive summaries
print("PRIOR PREDICTIVE SUMMARIES:")
print("-" * 80)
print(f"Y range in data:           [{Y_MIN:.2f}, {Y_MAX:.2f}]")
print(f"Y mean in data:            {Y_MEAN:.2f} ± {Y_SD:.2f}")
print()
print(f"Prior pred Y range:        [{np.min(min_y_sim):.2f}, {np.max(max_y_sim):.2f}]")
print(f"Prior pred Y mean (avg):   {np.mean(mean_y_sim):.2f} ± {np.std(mean_y_sim):.2f}")
print(f"Prior pred Y SD (avg):     {np.mean(sd_y_sim):.2f} ± {np.std(sd_y_sim):.2f}")
print()

# Plausibility checks
print("PLAUSIBILITY CHECKS:")
print("-" * 80)

# Check 1: Y in plausible range [0.5, 4.5]
plausible_min, plausible_max = 0.5, 4.5
in_plausible_range = np.mean((min_y_sim >= plausible_min) & (max_y_sim <= plausible_max))
print(f"1. Predictions in [{plausible_min}, {plausible_max}]:        "
      f"{in_plausible_range*100:.1f}% (target: ≥80%)")

# Check 2: Monotonic increase
monotonic_pct = np.mean(monotonic_increase > 0)
print(f"2. Monotonically increasing curves:     {monotonic_pct*100:.1f}% (target: ≥90%)")

# Check 3: Observed data coverage
# Check if observed Y range falls within prior predictive range for each x
coverage_counts = np.zeros(N_OBS)
for i in range(N_OBS):
    y_at_x = y_sim[:, i]
    # Check what percentile the observed range would fall in
    pct_2p5 = np.percentile(y_at_x, 2.5)
    pct_97p5 = np.percentile(y_at_x, 97.5)
    # Observed data should fall in central 95% interval
    coverage_counts[i] = (pct_2p5 <= Y_MAX) and (pct_97p5 >= Y_MIN)

coverage_pct = np.mean(coverage_counts)
print(f"3. Observed data in 95% prior interval: {coverage_pct*100:.1f}% of x values (target: ≥80%)")

# Check 4: Extrapolation at x=50
extrapolation_reasonable = np.mean(y_x50 < 5.0)
print(f"4. Predictions at x=50 reasonable (<5): {extrapolation_reasonable*100:.1f}% (target: ≥80%)")

# Check 5: Extreme value detection
extreme_low = np.mean(min_y_sim < 0)
extreme_high = np.mean(max_y_sim > 10)
print(f"5. Extreme predictions (Y<0):           {extreme_low*100:.1f}% (target: <5%)")
print(f"6. Extreme predictions (Y>10):          {extreme_high*100:.1f}% (target: <5%)")

# Check 6: Scale alignment
y_in_2sd = np.mean((mean_y_sim >= Y_MEAN - 2*Y_SD) & (mean_y_sim <= Y_MEAN + 2*Y_SD))
print(f"7. Mean predictions within ±2 SD:       {y_in_2sd*100:.1f}% (target: ≥70%)")

print()

# ============================================================================
# 4. Create Visualizations
# ============================================================================

print("Step 4: Creating diagnostic visualizations...")
print()

# -------------------------------------------------------------------------
# Visualization 1: Parameter Plausibility
# -------------------------------------------------------------------------
print("Creating: parameter_plausibility.png")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Prior Parameter Distributions - Assessing Plausibility', fontsize=14, fontweight='bold')

param_list = [
    (alpha_prior, 'alpha (intercept)', 'Normal(2.0, 0.5)', [0, 4]),
    (beta_prior, 'beta (slope)', 'Normal(0.3, 0.3)', [-0.5, 1.2]),
    (c_prior, 'c (log shift)', 'Gamma(2, 2)', [0, 4]),
    (nu_prior, 'nu (degrees of freedom)', 'Gamma(2, 0.1)', [0, 60]),
    (sigma_prior, 'sigma (residual scale)', 'Half-Cauchy(0, 0.2)', [0, 1.5])
]

for idx, (values, name, prior_spec, xlim) in enumerate(param_list):
    ax = axes.flatten()[idx]
    ax.hist(values, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
    ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.2f}')
    ax.set_xlabel(name)
    ax.set_ylabel('Density')
    ax.set_title(f'{prior_spec}')
    ax.set_xlim(xlim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove the extra subplot
axes.flatten()[5].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_plausibility.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 2: Prior Predictive Curves
# -------------------------------------------------------------------------
print("Creating: prior_predictive_curves.png")

fig, ax = plt.subplots(figsize=(10, 7))

# Plot 100 random prior predictive curves
n_curves = 100
indices = np.random.choice(N_SAMPLES, n_curves, replace=False)

for idx in indices:
    # Compute curve for extended x range
    alpha_i = alpha_prior[idx]
    beta_i = beta_prior[idx]
    c_i = c_prior[idx]
    mu_i = alpha_i + beta_i * np.log(x_extended + c_i)
    ax.plot(x_extended, mu_i, alpha=0.15, color='steelblue', linewidth=0.5)

# Overlay observed data context
ax.axhspan(Y_MIN, Y_MAX, alpha=0.2, color='green', label='Observed Y range')
ax.axhline(Y_MEAN, color='green', linestyle='--', linewidth=2, label=f'Observed Y mean: {Y_MEAN:.2f}')
ax.axhline(Y_MEAN - 2*Y_SD, color='green', linestyle=':', linewidth=1, alpha=0.7, label='±2 SD')
ax.axhline(Y_MEAN + 2*Y_SD, color='green', linestyle=':', linewidth=1, alpha=0.7)

# Mark observed x range
ax.axvline(X_MIN, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(X_MAX, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvspan(X_MIN, X_MAX, alpha=0.05, color='red', label='Observed X range')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('μ(x) = α + β·log(x + c)', fontsize=12)
ax.set_title('Prior Predictive Curves (100 samples from prior)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 50)
ax.set_ylim(-1, 6)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_predictive_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 3: Prior Predictive Coverage
# -------------------------------------------------------------------------
print("Creating: prior_predictive_coverage.png")

fig, ax = plt.subplots(figsize=(10, 7))

# Compute percentiles across prior predictive distribution at each x
percentiles = [2.5, 10, 25, 50, 75, 90, 97.5]
pct_values = np.percentile(y_sim, percentiles, axis=0)

# Plot bands
ax.fill_between(x_observed, pct_values[0], pct_values[-1], alpha=0.2, color='steelblue', label='95% prior interval')
ax.fill_between(x_observed, pct_values[1], pct_values[-2], alpha=0.3, color='steelblue', label='80% prior interval')
ax.fill_between(x_observed, pct_values[2], pct_values[-3], alpha=0.4, color='steelblue', label='50% prior interval')

# Median
ax.plot(x_observed, pct_values[3], color='darkblue', linewidth=2, label='Median prediction')

# Overlay observed data context
ax.axhspan(Y_MIN, Y_MAX, alpha=0.15, color='green', zorder=10, label='Observed Y range')
ax.axhline(Y_MEAN, color='green', linestyle='--', linewidth=2, zorder=11)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y (prior predictive)', fontsize=12)
ax.set_title('Prior Predictive Distribution Coverage', fontsize=14, fontweight='bold')
ax.set_xlim(X_MIN - 1, X_MAX + 1)
ax.set_ylim(-1, 6)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 4: Distribution at Key X Values
# -------------------------------------------------------------------------
print("Creating: predictions_at_key_x_values.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Prior Predictive Distributions at Key X Values', fontsize=14, fontweight='bold')

# Select three key x values: min, mid, max
x_keys = [0, N_OBS // 2, N_OBS - 1]
x_labels = [f'x={x_observed[i]:.1f}' for i in x_keys]

for idx, (x_idx, x_label) in enumerate(zip(x_keys, x_labels)):
    ax = axes[idx]
    y_at_x = y_sim[:, x_idx]

    ax.hist(y_at_x, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # Mark observed data range
    ax.axvspan(Y_MIN, Y_MAX, alpha=0.2, color='green', label='Observed Y range')
    ax.axvline(Y_MEAN, color='green', linestyle='--', linewidth=2, label=f'Observed mean: {Y_MEAN:.2f}')

    # Mark prior predictive quantiles
    pct_2p5, pct_97p5 = np.percentile(y_at_x, [2.5, 97.5])
    ax.axvline(pct_2p5, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='95% interval')
    ax.axvline(pct_97p5, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Y', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(x_label, fontsize=12, fontweight='bold')
    ax.set_xlim(-2, 8)
    if idx == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'predictions_at_key_x_values.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 5: Extrapolation Diagnostic
# -------------------------------------------------------------------------
print("Creating: extrapolation_diagnostic.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Extrapolation Behavior - Predictions Beyond Observed Data', fontsize=14, fontweight='bold')

# Left: Distribution at x=50
ax1.hist(y_x50, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
ax1.axvline(np.mean(y_x50), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y_x50):.2f}')
ax1.axvline(np.median(y_x50), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(y_x50):.2f}')
ax1.axvline(5.0, color='black', linestyle=':', linewidth=2, label='Reasonableness threshold (5.0)')
pct_below_5 = np.mean(y_x50 < 5.0) * 100
ax1.text(0.05, 0.95, f'{pct_below_5:.1f}% < 5.0', transform=ax1.transAxes,
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.set_xlabel('Y at x=50', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Predictions at x=50 (extrapolation)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Right: Curve showing extrapolation
n_curves_extrap = 50
indices_extrap = np.random.choice(N_SAMPLES, n_curves_extrap, replace=False)

for idx in indices_extrap:
    alpha_i = alpha_prior[idx]
    beta_i = beta_prior[idx]
    c_i = c_prior[idx]
    mu_i = alpha_i + beta_i * np.log(x_extended + c_i)
    ax2.plot(x_extended, mu_i, alpha=0.2, color='coral', linewidth=0.5)

# Mark observed vs extrapolated regions
ax2.axvspan(X_MIN, X_MAX, alpha=0.1, color='green', label='Observed X range')
ax2.axvspan(X_MAX, 50, alpha=0.1, color='red', label='Extrapolation region')
ax2.axvline(X_MAX, color='black', linestyle='--', linewidth=2, alpha=0.7)

ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('μ(x)', fontsize=11)
ax2.set_title('Mean function curves (50 samples)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 5)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'extrapolation_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 6: Monotonicity Check
# -------------------------------------------------------------------------
print("Creating: monotonicity_diagnostic.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Monotonicity Assessment - Do Curves Increase?', fontsize=14, fontweight='bold')

# Left: Distribution of slope (mu[N] - mu[1])
ax1.hist(monotonic_increase, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (no change)')
ax1.axvline(np.mean(monotonic_increase), color='darkviolet', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(monotonic_increase):.2f}')
pct_positive = np.mean(monotonic_increase > 0) * 100
ax1.text(0.05, 0.95, f'{pct_positive:.1f}% positive', transform=ax1.transAxes,
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.set_xlabel('μ(x_max) - μ(x_min)', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Change from x_min to x_max', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Right: Scatter of beta vs monotonic increase
ax2.scatter(beta_prior, monotonic_increase, alpha=0.3, s=20, color='purple')
ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('β (slope parameter)', fontsize=11)
ax2.set_ylabel('μ(x_max) - μ(x_min)', fontsize=11)
ax2.set_title('Relationship between β and monotonicity', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'monotonicity_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Visualization 7: Comprehensive Multi-Panel Summary
# -------------------------------------------------------------------------
print("Creating: comprehensive_summary.png")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Prior predictive curves (large)
ax1 = fig.add_subplot(gs[0:2, 0:2])
n_curves_summary = 200
indices_summary = np.random.choice(N_SAMPLES, n_curves_summary, replace=False)
for idx in indices_summary:
    alpha_i = alpha_prior[idx]
    beta_i = beta_prior[idx]
    c_i = c_prior[idx]
    mu_i = alpha_i + beta_i * np.log(x_extended + c_i)
    ax1.plot(x_extended, mu_i, alpha=0.08, color='steelblue', linewidth=0.5)
ax1.axhspan(Y_MIN, Y_MAX, alpha=0.2, color='green')
ax1.axhline(Y_MEAN, color='green', linestyle='--', linewidth=2)
ax1.axvspan(X_MIN, X_MAX, alpha=0.05, color='red')
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('μ(x)', fontsize=11)
ax1.set_title('Prior Predictive Curves (200 samples)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 50)
ax1.set_ylim(-1, 6)
ax1.grid(True, alpha=0.3)

# Panel 2: Beta distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(beta_prior, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('β (slope)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('Prior: β ~ N(0.3, 0.3)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel 3: Nu distribution
ax3 = fig.add_subplot(gs[1, 2])
ax3.hist(nu_prior, bins=40, density=True, alpha=0.7, color='orange', edgecolor='black')
ax3.set_xlabel('ν (df)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('Prior: ν ~ Gamma(2, 0.1)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel 4: Coverage at x_min
ax4 = fig.add_subplot(gs[2, 0])
y_at_x_min = y_sim[:, 0]
ax4.hist(y_at_x_min, bins=40, density=True, alpha=0.7, color='coral', edgecolor='black')
ax4.axvspan(Y_MIN, Y_MAX, alpha=0.2, color='green')
ax4.set_xlabel(f'Y at x={x_observed[0]:.1f}', fontsize=10)
ax4.set_ylabel('Density', fontsize=10)
ax4.set_title('Predictions at x_min', fontsize=11, fontweight='bold')
ax4.set_xlim(-2, 8)
ax4.grid(True, alpha=0.3)

# Panel 5: Coverage at x_max
ax5 = fig.add_subplot(gs[2, 1])
y_at_x_max = y_sim[:, -1]
ax5.hist(y_at_x_max, bins=40, density=True, alpha=0.7, color='coral', edgecolor='black')
ax5.axvspan(Y_MIN, Y_MAX, alpha=0.2, color='green')
ax5.set_xlabel(f'Y at x={x_observed[-1]:.1f}', fontsize=10)
ax5.set_ylabel('Density', fontsize=10)
ax5.set_title('Predictions at x_max', fontsize=11, fontweight='bold')
ax5.set_xlim(-2, 8)
ax5.grid(True, alpha=0.3)

# Panel 6: Extrapolation
ax6 = fig.add_subplot(gs[2, 2])
ax6.hist(y_x50, bins=40, density=True, alpha=0.7, color='purple', edgecolor='black')
ax6.axvline(5.0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Y at x=50', fontsize=10)
ax6.set_ylabel('Density', fontsize=10)
ax6.set_title('Extrapolation', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

fig.suptitle('Comprehensive Prior Predictive Check Summary', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / 'comprehensive_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print()
print("All visualizations created successfully.")
print()

# ============================================================================
# 5. Save Diagnostic Results
# ============================================================================

print("Step 5: Saving diagnostic results...")

diagnostics = {
    'n_samples': N_SAMPLES,
    'data_context': {
        'n_obs': N_OBS,
        'x_range': [float(X_MIN), float(X_MAX)],
        'y_range': [float(Y_MIN), float(Y_MAX)],
        'y_mean': float(Y_MEAN),
        'y_sd': float(Y_SD)
    },
    'prior_parameter_summaries': {
        name: {
            'mean': float(np.mean(values)),
            'sd': float(np.std(values)),
            'median': float(np.median(values)),
            'q025': float(np.percentile(values, 2.5)),
            'q975': float(np.percentile(values, 97.5))
        } for name, values in params.items()
    },
    'prior_predictive_summaries': {
        'y_range_min': float(np.min(min_y_sim)),
        'y_range_max': float(np.max(max_y_sim)),
        'mean_y_sim_avg': float(np.mean(mean_y_sim)),
        'mean_y_sim_sd': float(np.std(mean_y_sim)),
        'sd_y_sim_avg': float(np.mean(sd_y_sim)),
        'sd_y_sim_sd': float(np.std(sd_y_sim))
    },
    'plausibility_checks': {
        'predictions_in_range_0.5_4.5_pct': float(in_plausible_range * 100),
        'monotonically_increasing_pct': float(monotonic_pct * 100),
        'observed_data_coverage_pct': float(coverage_pct * 100),
        'extrapolation_reasonable_pct': float(extrapolation_reasonable * 100),
        'extreme_low_pct': float(extreme_low * 100),
        'extreme_high_pct': float(extreme_high * 100),
        'mean_within_2sd_pct': float(y_in_2sd * 100)
    },
    'decision_criteria': {
        'check_1_pass': bool(in_plausible_range >= 0.80),
        'check_2_pass': bool(monotonic_pct >= 0.90),
        'check_3_pass': bool(coverage_pct >= 0.80),
        'check_4_pass': bool(extrapolation_reasonable >= 0.80),
        'check_5_pass': bool(extreme_low < 0.05),
        'check_6_pass': bool(extreme_high < 0.05),
        'check_7_pass': bool(y_in_2sd >= 0.70)
    }
}

# Save to JSON
with open(CODE_DIR / 'diagnostics.json', 'w') as f:
    json.dump(diagnostics, f, indent=2)

print("Diagnostics saved to diagnostics.json")
print()

# ============================================================================
# 6. Final Assessment
# ============================================================================

print("=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)
print()

all_checks_pass = all(diagnostics['decision_criteria'].values())

if all_checks_pass:
    print("STATUS: PASS")
    print()
    print("All plausibility checks passed. The priors generate scientifically")
    print("plausible data that covers the observed range without systematic")
    print("violations or computational issues.")
else:
    print("STATUS: FAIL")
    print()
    print("One or more plausibility checks failed:")
    for check_name, passed in diagnostics['decision_criteria'].items():
        if not passed:
            print(f"  - {check_name}: FAILED")
    print()
    print("Prior adjustment needed before proceeding to model fitting.")

print()
print("=" * 80)
print("Prior predictive check complete. See findings.md for detailed analysis.")
print("=" * 80)
