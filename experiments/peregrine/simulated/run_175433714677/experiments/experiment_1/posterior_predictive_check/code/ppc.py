"""
Posterior Predictive Checks for Log-Linear Negative Binomial Model

This script performs comprehensive posterior predictive checks to assess
whether the fitted model can reproduce key features of the observed data.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = Path('/workspace/data/data.csv')
INFERENCE_PATH = Path('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
OUTPUT_DIR = Path('/workspace/experiments/experiment_1/posterior_predictive_check')
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("POSTERIOR PREDICTIVE CHECKS")
print("=" * 80)

# Load data
print("\n[1/6] Loading observed data...")
with open(DATA_PATH) as f:
    data = json.load(f)

C_obs = np.array(data['C'])
year = np.array(data['year'])
n = len(C_obs)

print(f"   - Observations: n = {n}")
print(f"   - Count range: [{C_obs.min()}, {C_obs.max()}]")
print(f"   - Year range: [{year.min():.2f}, {year.max():.2f}]")
print(f"   - Observed Var/Mean: {C_obs.var() / C_obs.mean():.2f}")

# Load posterior samples
print("\n[2/6] Loading posterior samples...")
idata = az.from_netcdf(INFERENCE_PATH)
print(f"   - Chains: {idata.posterior.sizes['chain']}")
print(f"   - Draws per chain: {idata.posterior.sizes['draw']}")

# Extract posterior samples (use correct variable names)
beta0 = idata.posterior['beta_0'].values.flatten()
beta1 = idata.posterior['beta_1'].values.flatten()
phi = idata.posterior['phi'].values.flatten()
n_samples = len(beta0)

print(f"   - Total posterior samples: {n_samples}")
print(f"   - β₀: {beta0.mean():.3f} ± {beta0.std():.3f}")
print(f"   - β₁: {beta1.mean():.3f} ± {beta1.std():.3f}")
print(f"   - φ: {phi.mean():.3f} ± {phi.std():.3f}")

# Generate posterior predictive samples
print("\n[3/6] Generating posterior predictive samples...")
n_ppc_samples = min(1000, n_samples)  # Use up to 1000 samples for PPC
rng = np.random.default_rng(42)

# Sample indices for PPC
ppc_indices = rng.choice(n_samples, size=n_ppc_samples, replace=False)

# Generate predictions
y_pred = np.zeros((n_ppc_samples, n))
for i, idx in enumerate(ppc_indices):
    mu = np.exp(beta0[idx] + beta1[idx] * year)
    # Negative binomial parameterization: NB(mu, phi)
    # Convert to (n, p) parameterization: n=phi, p=phi/(phi+mu)
    p = phi[idx] / (phi[idx] + mu)
    y_pred[i, :] = rng.negative_binomial(phi[idx], p)

print(f"   - Generated {n_ppc_samples} predictive samples")
print(f"   - Predicted count range: [{y_pred.min()}, {y_pred.max()}]")

# Compute summary statistics
y_pred_mean = y_pred.mean(axis=0)
y_pred_median = np.median(y_pred, axis=0)
y_pred_05 = np.percentile(y_pred, 5, axis=0)
y_pred_95 = np.percentile(y_pred, 95, axis=0)
y_pred_25 = np.percentile(y_pred, 25, axis=0)
y_pred_75 = np.percentile(y_pred, 75, axis=0)

# Residuals
residuals = C_obs - y_pred_mean

print("\n[4/6] Computing quantitative checks...")

# Check 1: Var/Mean recovery
var_mean_ppc = np.array([y_pred[i, :].var() / y_pred[i, :].mean()
                          for i in range(n_ppc_samples)])
var_mean_obs = C_obs.var() / C_obs.mean()

print(f"\n   Var/Mean Recovery:")
print(f"   - Observed: {var_mean_obs:.2f}")
print(f"   - Predicted: {var_mean_ppc.mean():.2f} ± {var_mean_ppc.std():.2f}")
print(f"   - 95% CI: [{np.percentile(var_mean_ppc, 2.5):.2f}, {np.percentile(var_mean_ppc, 97.5):.2f}]")
print(f"   - Within [50, 90]: {np.mean((var_mean_ppc >= 50) & (var_mean_ppc <= 90)) * 100:.1f}%")

# Check 2: Prediction interval coverage
in_90_interval = (C_obs >= y_pred_05) & (C_obs <= y_pred_95)
coverage_90 = in_90_interval.mean() * 100

print(f"\n   90% Prediction Interval Coverage:")
print(f"   - Coverage: {coverage_90:.1f}%")
print(f"   - Expected: 90%")
print(f"   - Observations outside: {(~in_90_interval).sum()}/{n}")

# Check 3: Systematic bias
print(f"\n   Residual Analysis:")
print(f"   - Mean residual: {residuals.mean():.2f}")
print(f"   - Median residual: {np.median(residuals):.2f}")
print(f"   - RMSE: {np.sqrt((residuals**2).mean()):.2f}")
print(f"   - MAE: {np.abs(residuals).mean():.2f}")

# Check 4: Early vs. late period fit
early_idx = np.arange(10)
late_idx = np.arange(30, 40)

mae_early = np.abs(residuals[early_idx]).mean()
mae_late = np.abs(residuals[late_idx]).mean()

print(f"\n   Early vs. Late Period Performance:")
print(f"   - MAE (first 10 obs): {mae_early:.2f}")
print(f"   - MAE (last 10 obs): {mae_late:.2f}")
print(f"   - Ratio (late/early): {mae_late/mae_early:.2f}")

# Check 5: Test for systematic curvature
# Fit quadratic to residuals
year_squared = year**2
X = np.column_stack([np.ones_like(year), year, year_squared])
quad_coef = np.linalg.lstsq(X, residuals, rcond=None)[0]
quad_curvature = quad_coef[2]

print(f"\n   Curvature Test:")
print(f"   - Quadratic coefficient: {quad_curvature:.4f}")
print(f"   - Sign: {'U-shape' if quad_curvature > 0 else 'Inverted-U' if quad_curvature < 0 else 'None'}")

# Print summary
print("\n" + "=" * 80)
print("FALSIFICATION CRITERIA CHECK")
print("=" * 80)

checks = {
    'Var/Mean in [50, 90]': np.percentile(var_mean_ppc, 2.5) >= 50 and np.percentile(var_mean_ppc, 97.5) <= 90,
    'Coverage > 80%': coverage_90 > 80,
    'Late/Early MAE < 2': mae_late / mae_early < 2.0,
    'No strong curvature': abs(quad_curvature) < 1.0,
}

for criterion, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"   [{status}] {criterion}")

overall = "PASS" if all(checks.values()) else "MIXED" if sum(checks.values()) >= 2 else "FAIL"
print(f"\n   Overall Assessment: {overall}")

print("\n[5/6] Creating diagnostic visualizations...")

# Plot 1: Time series with prediction intervals
fig, ax = plt.subplots(figsize=(12, 6))

# Prediction intervals
ax.fill_between(year, y_pred_05, y_pred_95, alpha=0.3, label='90% Prediction Interval')
ax.fill_between(year, y_pred_25, y_pred_75, alpha=0.4, label='50% Prediction Interval')

# Mean and median predictions
ax.plot(year, y_pred_mean, 'b-', linewidth=2, label='Posterior Mean', alpha=0.8)
ax.plot(year, y_pred_median, 'g--', linewidth=1.5, label='Posterior Median', alpha=0.8)

# Observed data
ax.scatter(year, C_obs, c='red', s=80, zorder=5, label='Observed', edgecolors='darkred', alpha=0.8)

# Highlight observations outside 90% interval
outside_idx = ~in_90_interval
if outside_idx.any():
    ax.scatter(year[outside_idx], C_obs[outside_idx],
              s=200, facecolors='none', edgecolors='red', linewidths=3,
              zorder=6, label=f'Outside 90% PI (n={outside_idx.sum()})')

ax.set_xlabel('Standardized Year', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Posterior Predictive Check: Observed vs. Predicted Time Series', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'timeseries_fit.png', dpi=300, bbox_inches='tight')
print(f"   - Saved: timeseries_fit.png")
plt.close()

# Plot 2: Distribution overlay
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of observed data
ax.hist(C_obs, bins=20, density=True, alpha=0.6, color='red',
        edgecolor='darkred', linewidth=1.5, label='Observed')

# Histogram of each predictive sample (light)
for i in range(min(100, n_ppc_samples)):
    ax.hist(y_pred[i, :], bins=20, density=True, alpha=0.01, color='blue')

# Mean of predictive distributions
y_pred_all = y_pred.flatten()
ax.hist(y_pred_all, bins=50, density=True, alpha=0.3, color='blue',
        edgecolor='darkblue', linewidth=1.5, label='Posterior Predictive (pooled)')

ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Predictive Distribution vs. Observed Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'distribution_overlay.png', dpi=300, bbox_inches='tight')
print(f"   - Saved: distribution_overlay.png")
plt.close()

# Plot 3: Residual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs. time
ax = axes[0, 0]
ax.scatter(year, residuals, c='black', s=60, alpha=0.7)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.plot(year, quad_coef[0] + quad_coef[1]*year + quad_coef[2]*year**2,
        'b-', linewidth=2, alpha=0.5, label='Quadratic fit')
ax.set_xlabel('Standardized Year', fontsize=11)
ax.set_ylabel('Residual (Obs - Pred)', fontsize=11)
ax.set_title('Residuals vs. Time', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals vs. predicted
ax = axes[0, 1]
ax.scatter(y_pred_mean, residuals, c='black', s=60, alpha=0.7)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Count', fontsize=11)
ax.set_ylabel('Residual (Obs - Pred)', fontsize=11)
ax.set_title('Residuals vs. Fitted Values', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Q-Q plot
ax = axes[1, 0]
standardized_residuals = residuals / np.std(residuals)
stats.probplot(standardized_residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Standardized Residuals', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Histogram of residuals
ax = axes[1, 1]
ax.hist(residuals, bins=15, density=True, alpha=0.7, color='gray', edgecolor='black')
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
        'r-', linewidth=2, label='Normal fit')
ax.axvline(0, color='blue', linestyle='--', linewidth=2, label='Zero')
ax.set_xlabel('Residual', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residuals.png', dpi=300, bbox_inches='tight')
print(f"   - Saved: residuals.png")
plt.close()

# Plot 4: Var/Mean recovery
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(var_mean_ppc, bins=50, density=True, alpha=0.7, color='skyblue',
        edgecolor='black', linewidth=1.5, label='Posterior Predictive')
ax.axvline(var_mean_obs, color='red', linestyle='--', linewidth=3,
          label=f'Observed ({var_mean_obs:.1f})')
ax.axvline(var_mean_ppc.mean(), color='blue', linestyle='-', linewidth=2,
          label=f'PP Mean ({var_mean_ppc.mean():.1f})')

# Add credible interval
ax.axvspan(np.percentile(var_mean_ppc, 2.5), np.percentile(var_mean_ppc, 97.5),
          alpha=0.2, color='blue', label='95% Credible Interval')

# Add target range
ax.axvspan(50, 90, alpha=0.1, color='green', label='Target Range [50, 90]')

ax.set_xlabel('Variance-to-Mean Ratio', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Var/Mean Ratio: Posterior Predictive vs. Observed', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'var_mean_recovery.png', dpi=300, bbox_inches='tight')
print(f"   - Saved: var_mean_recovery.png")
plt.close()

# Plot 5: Calibration plot
fig, ax = plt.subplots(figsize=(10, 6))

# Compute coverage at different levels
coverage_levels = np.arange(5, 100, 5)
observed_coverage = []
expected_coverage = []

for level in coverage_levels:
    lower = (100 - level) / 2
    upper = 100 - lower
    pred_lower = np.percentile(y_pred, lower, axis=0)
    pred_upper = np.percentile(y_pred, upper, axis=0)
    in_interval = (C_obs >= pred_lower) & (C_obs <= pred_upper)
    observed_coverage.append(in_interval.mean() * 100)
    expected_coverage.append(level)

ax.plot(expected_coverage, observed_coverage, 'o-', linewidth=2, markersize=8,
        label='Model Calibration', color='blue')
ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Calibration')

# Highlight 90% level
ax.scatter([90], [coverage_90], s=300, c='red', marker='*',
          edgecolors='darkred', linewidths=2, zorder=5,
          label=f'90% Level ({coverage_90:.1f}%)')

ax.set_xlabel('Expected Coverage (%)', fontsize=12)
ax.set_ylabel('Observed Coverage (%)', fontsize=12)
ax.set_title('Calibration Plot: Prediction Interval Coverage', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'calibration.png', dpi=300, bbox_inches='tight')
print(f"   - Saved: calibration.png")
plt.close()

# Plot 6: ArviZ PPC plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create InferenceData with posterior predictive
idata_ppc = idata.copy()
idata_ppc.add_groups({
    'posterior_predictive': {
        'C_obs': y_pred.reshape(n_ppc_samples // 4, 4, n)  # Reshape to match chains
    }
})

az.plot_ppc(idata_ppc, ax=ax, num_pp_samples=100, data_pairs={'C_obs': 'C_obs'})
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('ArviZ Posterior Predictive Check', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'arviz_ppc.png', dpi=300, bbox_inches='tight')
print(f"   - Saved: arviz_ppc.png")
plt.close()

# Plot 7: Early vs. Late period comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Early period
ax = axes[0]
ax.scatter(year[early_idx], C_obs[early_idx], c='red', s=100,
          zorder=5, label='Observed', edgecolors='darkred')
ax.fill_between(year[early_idx], y_pred_05[early_idx], y_pred_95[early_idx],
                alpha=0.3, label='90% PI')
ax.plot(year[early_idx], y_pred_mean[early_idx], 'b-', linewidth=2,
       label='Predicted Mean')
ax.set_xlabel('Standardized Year', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Early Period (MAE={mae_early:.2f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Late period
ax = axes[1]
ax.scatter(year[late_idx], C_obs[late_idx], c='red', s=100,
          zorder=5, label='Observed', edgecolors='darkred')
ax.fill_between(year[late_idx], y_pred_05[late_idx], y_pred_95[late_idx],
                alpha=0.3, label='90% PI')
ax.plot(year[late_idx], y_pred_mean[late_idx], 'b-', linewidth=2,
       label='Predicted Mean')
ax.set_xlabel('Standardized Year', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Late Period (MAE={mae_late:.2f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'early_vs_late_fit.png', dpi=300, bbox_inches='tight')
print(f"   - Saved: early_vs_late_fit.png")
plt.close()

print("\n[6/6] Creating results summary...")

# Save numerical results
results = {
    'var_mean_obs': float(var_mean_obs),
    'var_mean_pred_mean': float(var_mean_ppc.mean()),
    'var_mean_pred_std': float(var_mean_ppc.std()),
    'var_mean_pred_ci': [float(np.percentile(var_mean_ppc, 2.5)),
                         float(np.percentile(var_mean_ppc, 97.5))],
    'coverage_90': float(coverage_90),
    'mean_residual': float(residuals.mean()),
    'median_residual': float(np.median(residuals)),
    'rmse': float(np.sqrt((residuals**2).mean())),
    'mae': float(np.abs(residuals).mean()),
    'mae_early': float(mae_early),
    'mae_late': float(mae_late),
    'mae_ratio': float(mae_late / mae_early),
    'quad_curvature': float(quad_curvature),
    'checks': {k: bool(v) for k, v in checks.items()},
    'overall': overall
}

import json
with open(OUTPUT_DIR / 'ppc_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   - Saved: ppc_results.json")

print("\n" + "=" * 80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("=" * 80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"   - Code: {OUTPUT_DIR / 'code' / 'ppc.py'}")
print(f"   - Plots: {PLOTS_DIR}")
print(f"   - Results: {OUTPUT_DIR / 'ppc_results.json'}")
print(f"\nOverall Assessment: {overall}")
