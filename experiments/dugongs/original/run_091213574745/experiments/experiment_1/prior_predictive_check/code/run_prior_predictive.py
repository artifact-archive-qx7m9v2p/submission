"""
Prior Predictive Check for Experiment 1: Logarithmic Model

This script:
1. Samples from prior distributions (1000 draws)
2. Generates synthetic datasets from priors
3. Creates diagnostic visualizations
4. Assesses prior plausibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
import os

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
STAN_MODEL_PATH = "/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive.stan"
PLOTS_DIR = "/workspace/experiments/experiment_1/prior_predictive_check/plots"

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

print("="*70)
print("PRIOR PREDICTIVE CHECK: Experiment 1 - Logarithmic Model")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"  - Loaded {len(data)} observations")
print(f"  - x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
print(f"  - Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Prepare data for Stan
stan_data = {
    'N': len(data),
    'x': data['x'].values
}

# Compile and run Stan model
print("\n[2/5] Sampling from priors (1000 draws)...")
model = CmdStanModel(stan_file=STAN_MODEL_PATH)

# Generate prior predictive samples (use fixed_param since no parameters to sample)
prior_samples = model.sample(
    data=stan_data,
    fixed_param=True,
    iter_sampling=1000,
    chains=1,
    seed=42,
    show_progress=True,
    show_console=False
)

print("  - Sampling complete!")

# Extract samples
print("\n[3/5] Extracting prior samples...")

# Prior parameter samples
beta_0_samples = prior_samples.stan_variable('beta_0')
beta_1_samples = prior_samples.stan_variable('beta_1')
sigma_samples = prior_samples.stan_variable('sigma')

# Simulated data (shape: [n_draws, N])
y_sim = prior_samples.stan_variable('y_sim')

# Summary statistics
y_min_samples = prior_samples.stan_variable('y_min')
y_max_samples = prior_samples.stan_variable('y_max')
y_mean_samples = prior_samples.stan_variable('y_mean')
y_sd_samples = prior_samples.stan_variable('y_sd')

print(f"  - Extracted {len(beta_0_samples)} prior draws")
print(f"  - Generated {y_sim.shape[0]} synthetic datasets")

# Compute diagnostics
print("\n[4/5] Computing diagnostics...")

# Domain violations
extreme_low = np.sum(y_min_samples < -10)
extreme_high = np.sum(y_max_samples > 10)
outside_range = extreme_low + extreme_high
pct_violations = 100 * outside_range / len(y_min_samples)

# Negative slopes
negative_slopes = np.sum(beta_1_samples < 0)
pct_negative = 100 * negative_slopes / len(beta_1_samples)

# Unrealistic sigma (> 1.0 would be huge given Y range ~1)
large_sigma = np.sum(sigma_samples > 1.0)
pct_large_sigma = 100 * large_sigma / len(sigma_samples)

print(f"\n  DIAGNOSTIC SUMMARY:")
print(f"  ------------------")
print(f"  Prior parameter ranges:")
print(f"    β₀: [{beta_0_samples.min():.3f}, {beta_0_samples.max():.3f}]")
print(f"    β₁: [{beta_1_samples.min():.3f}, {beta_1_samples.max():.3f}]")
print(f"    σ:  [{sigma_samples.min():.3f}, {sigma_samples.max():.3f}]")
print(f"\n  Generated data ranges:")
print(f"    Y_min: [{y_min_samples.min():.3f}, {y_min_samples.max():.3f}]")
print(f"    Y_max: [{y_max_samples.min():.3f}, {y_max_samples.max():.3f}]")
print(f"\n  Violation checks:")
print(f"    Outside [-10, 10]: {pct_violations:.2f}% (FAIL if >10%)")
print(f"    Negative slopes:   {pct_negative:.2f}% (FAIL if >5%)")
print(f"    Large σ (>1.0):    {pct_large_sigma:.2f}% (FAIL if >10%)")

# Create visualizations
print("\n[5/5] Creating visualizations...")

# ============================================================================
# PLOT 1: Prior Parameter Distributions
# ============================================================================
print("  - Creating parameter_plausibility.png...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Prior Parameter Distributions and Joint Behavior',
             fontsize=16, fontweight='bold', y=0.995)

# β₀ marginal
ax = axes[0, 0]
ax.hist(beta_0_samples, bins=50, alpha=0.7, edgecolor='black', density=True)
ax.axvline(2.3, color='red', linestyle='--', linewidth=2, label='Prior mean')
ax.axvline(data['Y'].mean(), color='blue', linestyle='--', linewidth=2,
           label=f"Observed Y mean: {data['Y'].mean():.2f}")
ax.set_xlabel('β₀ (Intercept)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Intercept Prior: N(2.3, 0.3)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# β₁ marginal
ax = axes[0, 1]
ax.hist(beta_1_samples, bins=50, alpha=0.7, edgecolor='black', density=True)
ax.axvline(0.29, color='red', linestyle='--', linewidth=2, label='Prior mean')
ax.axvline(0, color='black', linestyle=':', linewidth=2, label='Zero (flat)')
ax.set_xlabel('β₁ (Log slope)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Slope Prior: N(0.29, 0.15)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# σ marginal
ax = axes[0, 2]
ax.hist(sigma_samples, bins=50, alpha=0.7, edgecolor='black', density=True)
ax.axvline(sigma_samples.mean(), color='red', linestyle='--', linewidth=2,
           label=f"Prior mean: {sigma_samples.mean():.3f}")
ax.axvline(1.0, color='orange', linestyle=':', linewidth=2, label='Warning threshold: 1.0')
ax.set_xlabel('σ (Scale)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Scale Prior: Exponential(10)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# β₀ vs β₁ (joint)
ax = axes[1, 0]
scatter = ax.scatter(beta_0_samples, beta_1_samples, alpha=0.3, s=10, c=sigma_samples,
                     cmap='viridis', vmin=0, vmax=sigma_samples.quantile(0.95))
ax.axvline(2.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(0.29, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(0, color='black', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xlabel('β₀ (Intercept)', fontsize=11, fontweight='bold')
ax.set_ylabel('β₁ (Slope)', fontsize=11, fontweight='bold')
ax.set_title('Joint Prior: β₀ vs β₁ (colored by σ)', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('σ', fontsize=10)
ax.grid(True, alpha=0.3)

# β₁ vs σ
ax = axes[1, 1]
ax.scatter(beta_1_samples, sigma_samples, alpha=0.3, s=10)
ax.axvline(0.29, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(sigma_samples.mean(), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(1.0, color='orange', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xlabel('β₁ (Slope)', fontsize=11, fontweight='bold')
ax.set_ylabel('σ (Scale)', fontsize=11, fontweight='bold')
ax.set_title('Joint Prior: β₁ vs σ', fontsize=12)
ax.grid(True, alpha=0.3)

# β₀ vs σ
ax = axes[1, 2]
ax.scatter(beta_0_samples, sigma_samples, alpha=0.3, s=10)
ax.axvline(2.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(sigma_samples.mean(), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(1.0, color='orange', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xlabel('β₀ (Intercept)', fontsize=11, fontweight='bold')
ax.set_ylabel('σ (Scale)', fontsize=11, fontweight='bold')
ax.set_title('Joint Prior: β₀ vs σ', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/parameter_plausibility.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 2: Prior Predictive Coverage
# ============================================================================
print("  - Creating prior_predictive_coverage.png...")

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Plot 100 random prior predictive curves
n_curves = 100
curve_indices = np.random.choice(len(beta_0_samples), n_curves, replace=False)

# Create fine grid for smooth curves
x_fine = np.linspace(data['x'].min(), data['x'].max(), 200)

for idx in curve_indices:
    mu_fine = beta_0_samples[idx] + beta_1_samples[idx] * np.log(x_fine)
    ax.plot(x_fine, mu_fine, color='gray', alpha=0.1, linewidth=1)

# Plot observed data
ax.scatter(data['x'], data['Y'], color='blue', s=80, alpha=0.8,
           edgecolors='black', linewidth=1, zorder=100, label='Observed data')

# Add reference lines
ax.axhline(data['Y'].min(), color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
ax.axhline(data['Y'].max(), color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
ax.axhline(-10, color='red', linestyle='--', alpha=0.7, linewidth=2,
           label='Plausibility bounds [-10, 10]')
ax.axhline(10, color='red', linestyle='--', alpha=0.7, linewidth=2)

ax.set_xlabel('x', fontsize=13, fontweight='bold')
ax.set_ylabel('Y', fontsize=13, fontweight='bold')
ax.set_title(f'Prior Predictive Check: 100 Random Curves from Prior\n' +
             f'Do prior predictions cover observed data range?',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-2, 5)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/prior_predictive_coverage.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 3: Generated Data Range Diagnostics
# ============================================================================
print("  - Creating data_range_diagnostic.png...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Prior Predictive Data Range Diagnostics',
             fontsize=16, fontweight='bold', y=1.00)

# Y_min distribution
ax = axes[0]
ax.hist(y_min_samples, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(data['Y'].min(), color='blue', linestyle='--', linewidth=2,
           label=f"Observed min: {data['Y'].min():.2f}")
ax.axvline(-10, color='red', linestyle='--', linewidth=2, label='Lower bound: -10')
ax.set_xlabel('min(Y_sim)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Minimum Values\n' +
             f'Extreme low (<-10): {pct_violations:.2f}%', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Y_max distribution
ax = axes[1]
ax.hist(y_max_samples, bins=50, alpha=0.7, edgecolor='black', color='coral')
ax.axvline(data['Y'].max(), color='blue', linestyle='--', linewidth=2,
           label=f"Observed max: {data['Y'].max():.2f}")
ax.axvline(10, color='red', linestyle='--', linewidth=2, label='Upper bound: 10')
ax.set_xlabel('max(Y_sim)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Maximum Values\n' +
             f'Extreme high (>10): {pct_violations:.2f}%', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Y_range (max - min) vs observed range
ax = axes[2]
y_range_samples = y_max_samples - y_min_samples
observed_range = data['Y'].max() - data['Y'].min()
ax.hist(y_range_samples, bins=50, alpha=0.7, edgecolor='black', color='seagreen')
ax.axvline(observed_range, color='blue', linestyle='--', linewidth=2,
           label=f"Observed range: {observed_range:.2f}")
ax.set_xlabel('range(Y_sim) = max - min', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Data Ranges\n' +
             f'Prior mean range: {y_range_samples.mean():.2f}', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/data_range_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 4: Residual Scale Diagnostic
# ============================================================================
print("  - Creating residual_scale_diagnostic.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Prior Scale (σ) Plausibility Check',
             fontsize=16, fontweight='bold', y=0.99)

# σ distribution with reference
ax = axes[0]
ax.hist(sigma_samples, bins=60, alpha=0.7, edgecolor='black', color='orchid')
ax.axvline(sigma_samples.mean(), color='red', linestyle='-', linewidth=2,
           label=f"Prior mean σ: {sigma_samples.mean():.3f}")
ax.axvline(sigma_samples.median(), color='orange', linestyle='--', linewidth=2,
           label=f"Prior median σ: {sigma_samples.median():.3f}")

# Estimate observed RMSE (rough approximation from EDA)
# Using log model from EDA: Y ~ β₀ + β₁*log(x)
from sklearn.linear_model import LinearRegression
X_log = np.log(data['x'].values).reshape(-1, 1)
lr = LinearRegression().fit(X_log, data['Y'].values)
y_pred = lr.predict(X_log)
observed_rmse = np.sqrt(np.mean((data['Y'].values - y_pred)**2))

ax.axvline(observed_rmse, color='blue', linestyle='--', linewidth=2,
           label=f"Observed RMSE: {observed_rmse:.3f}")
ax.axvline(1.0, color='darkred', linestyle=':', linewidth=2.5,
           label='Warning: σ > 1.0')
ax.set_xlabel('σ (Scale parameter)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title(f'Prior Distribution of σ\nExponential(10), mean = 0.1', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, min(1.5, sigma_samples.max() * 1.1))

# Prior SD of simulated data vs σ
ax = axes[1]
ax.scatter(sigma_samples, y_sd_samples, alpha=0.4, s=20, color='purple')
ax.plot([0, sigma_samples.max()], [0, sigma_samples.max()],
        'r--', linewidth=2, label='y_sd = σ (perfect match)')
ax.axhline(data['Y'].std(), color='blue', linestyle='--', linewidth=2,
           label=f"Observed Y SD: {data['Y'].std():.3f}")
ax.set_xlabel('σ (prior sample)', fontsize=12, fontweight='bold')
ax.set_ylabel('SD(Y_sim)', fontsize=12, fontweight='bold')
ax.set_title('Relationship: Prior σ vs Simulated Data SD\n' +
             'Should be approximately equal', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/residual_scale_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 5: Slope Sign Diagnostic
# ============================================================================
print("  - Creating slope_sign_diagnostic.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Slope Sign and Relationship Strength Diagnostic',
             fontsize=16, fontweight='bold', y=0.99)

# β₁ distribution focused on zero
ax = axes[0]
ax.hist(beta_1_samples, bins=60, alpha=0.7, edgecolor='black', color='teal')
ax.axvline(0, color='black', linestyle=':', linewidth=2.5, label='Zero (flat relationship)')
ax.axvline(0.29, color='red', linestyle='--', linewidth=2, label='Prior mean: 0.29')
ax.axvline(beta_1_samples.mean(), color='orange', linestyle='-', linewidth=2,
           label=f"Actual mean: {beta_1_samples.mean():.3f}")

# Shade negative region
negative_mask = beta_1_samples < 0
if np.any(negative_mask):
    ax.axvspan(beta_1_samples.min(), 0, alpha=0.2, color='red',
               label=f'Negative slopes: {pct_negative:.2f}%')

ax.set_xlabel('β₁ (log slope)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title(f'Slope Prior Distribution\nNegative slopes: {pct_negative:.2f}% (FAIL if >5%)',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Prior-implied R² approximation
# For log model: correlation between log(x) and Y_sim
ax = axes[1]
r_squared_samples = []
for i in range(min(500, len(beta_0_samples))):
    y_i = y_sim[i, :]
    # Correlation squared as rough R² proxy
    corr = np.corrcoef(np.log(data['x']), y_i)[0, 1]
    r_squared_samples.append(corr**2)

r_squared_samples = np.array(r_squared_samples)
ax.hist(r_squared_samples, bins=50, alpha=0.7, edgecolor='black', color='goldenrod')
ax.axvline(0.897, color='blue', linestyle='--', linewidth=2,
           label='Observed R² (EDA): 0.897')
ax.axvline(r_squared_samples.mean(), color='red', linestyle='-', linewidth=2,
           label=f"Prior mean R²: {r_squared_samples.mean():.3f}")
ax.set_xlabel('R² (correlation²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Prior-Implied Relationship Strength\n' +
             'Does prior allow for observed R² = 0.897?', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/slope_sign_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 6: Individual Dataset Examples
# ============================================================================
print("  - Creating example_datasets.png...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Six Random Prior Predictive Datasets\n' +
             'Assessing plausibility of individual realizations',
             fontsize=16, fontweight='bold', y=0.995)

# Select 6 random datasets
n_examples = 6
example_indices = np.random.choice(len(beta_0_samples), n_examples, replace=False)

for plot_idx, draw_idx in enumerate(example_indices):
    ax = axes[plot_idx // 3, plot_idx % 3]

    # Get parameters for this draw
    b0 = beta_0_samples[draw_idx]
    b1 = beta_1_samples[draw_idx]
    s = sigma_samples[draw_idx]

    # Get simulated data
    y_draw = y_sim[draw_idx, :]

    # Create fine grid for curve
    x_fine = np.linspace(data['x'].min(), data['x'].max(), 200)
    mu_fine = b0 + b1 * np.log(x_fine)

    # Plot curve and data
    ax.plot(x_fine, mu_fine, 'r-', linewidth=2, label='Mean function')
    ax.scatter(data['x'], y_draw, s=50, alpha=0.7, edgecolors='black',
               linewidth=0.5, label='Simulated data')

    # Plot observed data for reference
    ax.scatter(data['x'], data['Y'], s=30, alpha=0.4, color='blue',
               marker='x', linewidth=2, label='Observed (reference)')

    # Add plausibility bounds
    ax.axhline(-10, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(10, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('x', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y', fontsize=10, fontweight='bold')
    ax.set_title(f'Draw {draw_idx}: β₀={b0:.2f}, β₁={b1:.2f}, σ={s:.3f}',
                 fontsize=11)

    if plot_idx == 0:
        ax.legend(fontsize=8, loc='upper left')

    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 5)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/example_datasets.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n  All plots saved to: {PLOTS_DIR}/")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*70)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("="*70)

print("\nVALIDATION RESULTS:")
print("-" * 70)

# Check 1: Domain violations
check1_pass = pct_violations <= 10.0
print(f"\n[1] Domain Constraint Check:")
print(f"    Generated Y outside [-10, 10]: {pct_violations:.2f}%")
print(f"    Status: {'PASS' if check1_pass else 'FAIL'} (threshold: ≤10%)")

# Check 2: Negative slopes
check2_pass = pct_negative <= 5.0
print(f"\n[2] Slope Sign Check:")
print(f"    Negative slopes (β₁ < 0): {pct_negative:.2f}%")
print(f"    Status: {'PASS' if check2_pass else 'FAIL'} (threshold: ≤5%)")

# Check 3: Unrealistic sigma
check3_pass = pct_large_sigma <= 10.0
print(f"\n[3] Scale Plausibility Check:")
print(f"    Large σ (>1.0): {pct_large_sigma:.2f}%")
print(f"    Prior mean σ: {sigma_samples.mean():.3f}")
print(f"    Observed RMSE: {observed_rmse:.3f}")
print(f"    Status: {'PASS' if check3_pass else 'FAIL'} (threshold: ≤10% with σ>1)")

# Check 4: Coverage of observed data
# Check if observed range is well within prior predictive range
obs_range_coverage = (data['Y'].min() > y_min_samples.quantile(0.01) and
                      data['Y'].max() < y_max_samples.quantile(0.99))
print(f"\n[4] Coverage Check:")
print(f"    Observed Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")
print(f"    Prior 98% interval: [{y_min_samples.quantile(0.01):.2f}, {y_max_samples.quantile(0.99):.2f}]")
print(f"    Status: {'PASS' if obs_range_coverage else 'FAIL'} (observed within prior range)")

# Check 5: R² compatibility
r2_compatible = r_squared_samples.quantile(0.99) > 0.80
print(f"\n[5] Relationship Strength Check:")
print(f"    Observed R²: 0.897")
print(f"    Prior 99th percentile R²: {r_squared_samples.quantile(0.99):.3f}")
print(f"    Status: {'PASS' if r2_compatible else 'FAIL'} (prior allows strong relationship)")

# Overall decision
all_checks_pass = check1_pass and check2_pass and check3_pass and obs_range_coverage and r2_compatible

print("\n" + "="*70)
print("OVERALL DECISION:", "PASS" if all_checks_pass else "FAIL")
print("="*70)

if all_checks_pass:
    print("\nPriors are weakly informative and scientifically plausible.")
    print("RECOMMENDATION: Proceed to simulation-based validation.")
else:
    print("\nPrior issues detected. See findings.md for detailed recommendations.")

print("\n" + "="*70)
