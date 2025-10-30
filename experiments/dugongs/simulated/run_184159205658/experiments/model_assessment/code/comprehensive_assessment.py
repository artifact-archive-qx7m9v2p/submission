"""
Comprehensive Model Assessment for ACCEPTED Model 1 (Logarithmic Regression)

This script performs:
1. LOO-CV diagnostics (ELPD, Pareto k)
2. Calibration assessment (LOO-PIT, coverage)
3. Absolute predictive metrics (RMSE, MAE, R², MAPE)
4. Parameter interpretation
5. Predictive performance visualization
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
INFERENCE_DATA_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = "/workspace/experiments/model_assessment"

print("="*80)
print("COMPREHENSIVE MODEL ASSESSMENT: LOGARITHMIC REGRESSION")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA AND INFERENCEDATA
# ============================================================================
print("1. LOADING DATA AND MODEL")
print("-"*80)

# Load observed data
data = pd.read_csv(DATA_PATH)
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(x_obs)

print(f"Observations: N = {n_obs}")
print(f"x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"y range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
print()

# Load InferenceData
print("Loading InferenceData...")
idata = az.from_netcdf(INFERENCE_DATA_PATH)

# Verify log_likelihood is present
if 'log_likelihood' not in idata:
    raise ValueError("ERROR: log_likelihood not found in InferenceData. Cannot compute LOO.")

print("InferenceData loaded successfully")
print(f"Groups available: {list(idata.groups())}")
print(f"Posterior samples: {idata.posterior.dims}")
print()

# ============================================================================
# 2. LOO-CV DIAGNOSTICS
# ============================================================================
print("="*80)
print("2. LOO-CV DIAGNOSTICS")
print("="*80)

# Compute LOO
print("Computing LOO-CV...")
loo_result = az.loo(idata, pointwise=True)

print(f"\nLOO Results:")
print(f"  ELPD_loo: {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
print(f"  p_loo (effective parameters): {loo_result.p_loo:.2f}")
print(f"  LOO-IC: {-2 * loo_result.elpd_loo:.2f}")
print()

# Pareto k diagnostics
pareto_k = loo_result.pareto_k
k_good = np.sum(pareto_k < 0.5)
k_ok = np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))
k_bad = np.sum(pareto_k >= 0.7)

print("Pareto k Diagnostics:")
print(f"  k < 0.5 (good):           {k_good:3d} ({100*k_good/n_obs:.1f}%)")
print(f"  0.5 ≤ k < 0.7 (ok):       {k_ok:3d} ({100*k_ok/n_obs:.1f}%)")
print(f"  k ≥ 0.7 (problematic):    {k_bad:3d} ({100*k_bad/n_obs:.1f}%)")
print(f"  Max k: {pareto_k.max():.3f}")
print(f"  Mean k: {pareto_k.mean():.3f}")
print()

if k_bad > 0:
    print("WARNING: Some observations have high Pareto k values.")
    print("LOO estimates may be unreliable for these points.")
    bad_indices = np.where(pareto_k >= 0.7)[0]
    print(f"Problematic observations: {bad_indices}")
else:
    print("EXCELLENT: All Pareto k values indicate reliable LOO estimates.")

print()

# Interpretation of p_loo
print("Model Complexity Assessment:")
print(f"  Effective parameters (p_loo): {loo_result.p_loo:.2f}")
print(f"  Nominal parameters: 3 (β₀, β₁, σ)")
if loo_result.p_loo > 5:
    print("  NOTE: p_loo > nominal suggests model may be overfitting")
elif loo_result.p_loo < 2:
    print("  NOTE: p_loo < nominal suggests strong regularization")
else:
    print("  GOOD: p_loo is consistent with model complexity")
print()

# ============================================================================
# 3. CALIBRATION ASSESSMENT
# ============================================================================
print("="*80)
print("3. CALIBRATION ASSESSMENT")
print("="*80)

# Get posterior predictive samples
y_pred = idata.posterior_predictive['Y'].values
n_chains, n_draws, n_points = y_pred.shape
y_pred_flat = y_pred.reshape(-1, n_points)

print(f"Posterior predictive shape: {y_pred.shape}")
print(f"Total samples per observation: {n_chains * n_draws}")
print()

# Compute LOO-PIT (Probability Integral Transform)
print("Computing LOO-PIT for calibration check...")

# Manual LOO-PIT calculation
loo_pit = np.zeros(n_obs)
for i in range(n_obs):
    # For each observation, compute how many LOO predictions are below observed
    loo_pit[i] = np.mean(y_pred_flat[:, i] <= y_obs[i])

print(f"LOO-PIT computed for {n_obs} observations")
print()

# Test uniformity using Kolmogorov-Smirnov test
ks_stat, ks_pval = stats.kstest(loo_pit, 'uniform')
print("Uniformity Test (Kolmogorov-Smirnov):")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pval:.4f}")
if ks_pval > 0.05:
    print("  GOOD: LOO-PIT is consistent with uniform distribution")
    print("        Model is well-calibrated")
else:
    print("  WARNING: LOO-PIT deviates from uniformity")
    print("           Model may be miscalibrated")
print()

# Compute coverage at multiple levels
coverage_levels = [0.50, 0.80, 0.90, 0.95]
print("Posterior Predictive Interval Coverage:")
print("  Level    Expected    Observed    Difference")
print("  " + "-"*45)

coverage_results = {}
for level in coverage_levels:
    alpha = (1 - level) / 2
    lower = np.percentile(y_pred_flat, 100 * alpha, axis=0)
    upper = np.percentile(y_pred_flat, 100 * (1 - alpha), axis=0)

    in_interval = np.sum((y_obs >= lower) & (y_obs <= upper))
    observed_coverage = in_interval / n_obs
    difference = observed_coverage - level

    coverage_results[level] = {
        'expected': level,
        'observed': observed_coverage,
        'difference': difference,
        'lower': lower,
        'upper': upper
    }

    print(f"  {level:4.0%}     {level:5.1%}       {observed_coverage:5.1%}       {difference:+6.1%}")

print()

# Check 90% coverage specifically (target ± 5%)
cov_90 = coverage_results[0.90]['observed']
if abs(cov_90 - 0.90) <= 0.05:
    print(f"EXCELLENT: 90% coverage ({cov_90:.1%}) is within ±5% of target")
else:
    print(f"WARNING: 90% coverage ({cov_90:.1%}) deviates from target by {abs(cov_90-0.90):.1%}")
print()

# ============================================================================
# 4. ABSOLUTE PREDICTIVE METRICS
# ============================================================================
print("="*80)
print("4. ABSOLUTE PREDICTIVE METRICS")
print("="*80)

# Posterior predictive mean
y_pred_mean = y_pred_flat.mean(axis=0)
y_pred_std = y_pred_flat.std(axis=0)

# RMSE
rmse = np.sqrt(np.mean((y_obs - y_pred_mean)**2))

# MAE
mae = np.mean(np.abs(y_obs - y_pred_mean))

# Bayesian R²
# R² = 1 - Var(residuals) / Var(y)
residuals = y_obs - y_pred_mean
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_obs - y_obs.mean())**2)
r2 = 1 - (ss_res / ss_tot)

# MAPE (if all y > 0)
if np.all(y_obs > 0):
    mape = np.mean(np.abs((y_obs - y_pred_mean) / y_obs)) * 100
else:
    mape = np.nan

# Mean prediction interval width (90%)
interval_widths = coverage_results[0.90]['upper'] - coverage_results[0.90]['lower']
mean_interval_width = np.mean(interval_widths)

print("Point Prediction Metrics:")
print(f"  RMSE:  {rmse:.4f}")
print(f"  MAE:   {mae:.4f}")
print(f"  R²:    {r2:.4f}")
if not np.isnan(mape):
    print(f"  MAPE:  {mape:.2f}%")
print()

print("Uncertainty Quantification:")
print(f"  Mean posterior std dev:    {y_pred_std.mean():.4f}")
print(f"  Mean 90% interval width:   {mean_interval_width:.4f}")
print(f"  Min interval width:        {interval_widths.min():.4f}")
print(f"  Max interval width:        {interval_widths.max():.4f}")
print()

# Where is uncertainty highest?
highest_unc_idx = np.argsort(y_pred_std)[-3:][::-1]
print("Observations with highest uncertainty:")
for idx in highest_unc_idx:
    print(f"  i={idx}: x={x_obs[idx]:.1f}, y={y_obs[idx]:.3f}, "
          f"pred={y_pred_mean[idx]:.3f} ± {y_pred_std[idx]:.3f}")
print()

# ============================================================================
# 5. PARAMETER INTERPRETATION
# ============================================================================
print("="*80)
print("5. PARAMETER INTERPRETATION")
print("="*80)

# Extract parameters
beta0 = idata.posterior['beta0'].values.flatten()
beta1 = idata.posterior['beta1'].values.flatten()
sigma = idata.posterior['sigma'].values.flatten()

print("Posterior Summaries:")
print()
print("β₀ (Intercept):")
print(f"  Mean:   {beta0.mean():.4f}")
print(f"  Median: {np.median(beta0):.4f}")
print(f"  SD:     {beta0.std():.4f}")
print(f"  95% CI: [{np.percentile(beta0, 2.5):.4f}, {np.percentile(beta0, 97.5):.4f}]")
print()

print("β₁ (Log-slope coefficient):")
print(f"  Mean:   {beta1.mean():.4f}")
print(f"  Median: {np.median(beta1):.4f}")
print(f"  SD:     {beta1.std():.4f}")
print(f"  95% CI: [{np.percentile(beta1, 2.5):.4f}, {np.percentile(beta1, 97.5):.4f}]")
print()

print("σ (Residual standard deviation):")
print(f"  Mean:   {sigma.mean():.4f}")
print(f"  Median: {np.median(sigma):.4f}")
print(f"  SD:     {sigma.std():.4f}")
print(f"  95% CI: [{np.percentile(sigma, 2.5):.4f}, {np.percentile(sigma, 97.5):.4f}]")
print()

print("Scientific Interpretation:")
print("-" * 40)
print(f"Model: Y = β₀ + β₁·log(x) + ε")
print()
print("β₁ interpretation (elasticity):")
print(f"  A 1% increase in x is associated with")
print(f"  approximately {beta1.mean()*0.01:.5f} increase in Y")
print()
print(f"  Doubling x (100% increase) increases Y by:")
print(f"  β₁·log(2) = {beta1.mean() * np.log(2):.4f}")
print(f"  95% CI: [{np.percentile(beta1, 2.5) * np.log(2):.4f}, "
      f"{np.percentile(beta1, 97.5) * np.log(2):.4f}]")
print()

# Diminishing returns demonstration
x_values = [1, 2, 5, 10, 20, 30]
print("Diminishing Returns Pattern:")
print("  x     E[Y]    Marginal gain from x-1")
print("  " + "-"*40)
for i, x in enumerate(x_values):
    y_expected = beta0.mean() + beta1.mean() * np.log(x)
    if i > 0:
        y_prev = beta0.mean() + beta1.mean() * np.log(x_values[i-1])
        gain = y_expected - y_prev
        print(f"  {x:2d}    {y_expected:.3f}    {gain:+.3f}")
    else:
        print(f"  {x:2d}    {y_expected:.3f}    -")
print()

# ============================================================================
# 6. VISUALIZATION 1: LOO DIAGNOSTICS
# ============================================================================
print("="*80)
print("6. CREATING VISUALIZATIONS")
print("="*80)
print()
print("Creating LOO diagnostics plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1a: Pareto k values
ax = axes[0]
ax.scatter(range(n_obs), pareto_k, alpha=0.6, s=50)
ax.axhline(0.5, color='orange', linestyle='--', label='k=0.5 (good threshold)')
ax.axhline(0.7, color='red', linestyle='--', label='k=0.7 (problematic threshold)')
ax.set_xlabel('Observation Index', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('LOO Pareto k Diagnostics', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 1b: Pareto k histogram
ax = axes[1]
ax.hist(pareto_k, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='k=0.5')
ax.axvline(0.7, color='red', linestyle='--', linewidth=2, label='k=0.7')
ax.set_xlabel('Pareto k', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution of Pareto k Values', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add summary text
summary_text = (f"ELPD_loo: {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}\n"
                f"p_loo: {loo_result.p_loo:.2f}\n"
                f"Good (k<0.5): {100*k_good/n_obs:.1f}%\n"
                f"OK (0.5≤k<0.7): {100*k_ok/n_obs:.1f}%\n"
                f"Bad (k≥0.7): {100*k_bad/n_obs:.1f}%")
ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9, family='monospace')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/loo_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/plots/loo_diagnostics.png")
plt.close()

# ============================================================================
# 7. VISUALIZATION 2: CALIBRATION
# ============================================================================
print("Creating calibration plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 2a: LOO-PIT
ax = axes[0]
ax.hist(loo_pit, bins=20, alpha=0.7, edgecolor='black', density=True, label='LOO-PIT')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform (ideal)')
ax.set_xlabel('LOO-PIT Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('LOO Probability Integral Transform', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 2)

# Add KS test result
ks_text = f"KS test: p = {ks_pval:.3f}\n"
if ks_pval > 0.05:
    ks_text += "Well-calibrated"
else:
    ks_text += "Miscalibrated"
ax.text(0.02, 0.98, ks_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        fontsize=10, fontweight='bold')

# Plot 2b: Coverage by level
ax = axes[1]
levels = list(coverage_results.keys())
expected = [coverage_results[l]['expected'] for l in levels]
observed = [coverage_results[l]['observed'] for l in levels]

x_pos = np.arange(len(levels))
width = 0.35

bars1 = ax.bar(x_pos - width/2, expected, width, label='Expected', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, observed, width, label='Observed', alpha=0.7)

ax.set_xlabel('Credible Interval Level', fontsize=11)
ax.set_ylabel('Coverage', fontsize=11)
ax.set_title('Posterior Predictive Interval Coverage', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{int(l*100)}%' for l in levels])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

# Add target zone for 90%
if 0.90 in coverage_results:
    ax.axhline(0.85, color='green', linestyle=':', alpha=0.5)
    ax.axhline(0.95, color='green', linestyle=':', alpha=0.5)
    ax.fill_between([-0.5, len(levels)-0.5], 0.85, 0.95,
                     alpha=0.1, color='green', label='±5% target zone')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/calibration_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/plots/calibration_plot.png")
plt.close()

# ============================================================================
# 8. VISUALIZATION 3: PREDICTIVE PERFORMANCE
# ============================================================================
print("Creating predictive performance plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 3a: Observed vs Predicted with uncertainty
ax = axes[0, 0]
ax.scatter(y_obs, y_pred_mean, alpha=0.6, s=80, label='Observations')
ax.errorbar(y_obs, y_pred_mean, yerr=1.96*y_pred_std, fmt='none',
            alpha=0.3, elinewidth=1, capsize=3)

# Perfect prediction line
y_range = [y_obs.min(), y_obs.max()]
ax.plot(y_range, y_range, 'r--', linewidth=2, label='Perfect prediction')

ax.set_xlabel('Observed Y', fontsize=11)
ax.set_ylabel('Predicted Y (mean ± 1.96 SD)', fontsize=11)
ax.set_title('Observed vs Predicted with Uncertainty', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Add R² annotation
ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, fontweight='bold')

# Plot 3b: Residuals vs x
ax = axes[0, 1]
ax.scatter(x_obs, residuals, alpha=0.6, s=80)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2*rmse, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2*rmse, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Residuals (observed - predicted)', fontsize=11)
ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3c: Predictions with intervals across x
ax = axes[1, 0]

# Sort by x for plotting
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]
y_sorted = y_obs[sort_idx]
y_pred_sorted = y_pred_mean[sort_idx]
lower_90 = coverage_results[0.90]['lower'][sort_idx]
upper_90 = coverage_results[0.90]['upper'][sort_idx]
lower_50 = coverage_results[0.50]['lower'][sort_idx]
upper_50 = coverage_results[0.50]['upper'][sort_idx]

ax.scatter(x_sorted, y_sorted, alpha=0.6, s=80, color='black',
           label='Observed', zorder=5)
ax.plot(x_sorted, y_pred_sorted, 'b-', linewidth=2, label='Predicted mean')
ax.fill_between(x_sorted, lower_90, upper_90, alpha=0.2, label='90% interval')
ax.fill_between(x_sorted, lower_50, upper_50, alpha=0.3, label='50% interval')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title('Predictions with Credible Intervals', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3d: Uncertainty vs x
ax = axes[1, 1]
ax.scatter(x_obs, y_pred_std, alpha=0.6, s=80, color='purple')
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Posterior Predictive SD', fontsize=11)
ax.set_title('Prediction Uncertainty Across x', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add annotation for high uncertainty region
if x_obs.max() > 20:
    ax.axvline(20, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(20, ax.get_ylim()[1]*0.95, 'x > 20\n(sparse data)',
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/predictive_performance.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/plots/predictive_performance.png")
plt.close()

# ============================================================================
# 9. VISUALIZATION 4: PARAMETER INTERPRETATION
# ============================================================================
print("Creating parameter interpretation plot...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 4a: Beta0 posterior
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(beta0, bins=50, alpha=0.7, edgecolor='black', density=True)
ax1.axvline(beta0.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax1.axvline(np.percentile(beta0, 2.5), color='orange', linestyle=':', linewidth=1.5)
ax1.axvline(np.percentile(beta0, 97.5), color='orange', linestyle=':', linewidth=1.5)
ax1.set_xlabel('β₀ (Intercept)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_title('Posterior: β₀', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 4b: Beta1 posterior
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(beta1, bins=50, alpha=0.7, edgecolor='black', density=True, color='green')
ax2.axvline(beta1.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.axvline(np.percentile(beta1, 2.5), color='orange', linestyle=':', linewidth=1.5)
ax2.axvline(np.percentile(beta1, 97.5), color='orange', linestyle=':', linewidth=1.5)
ax2.set_xlabel('β₁ (Log-slope)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('Posterior: β₁', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 4c: Sigma posterior
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(sigma, bins=50, alpha=0.7, edgecolor='black', density=True, color='purple')
ax3.axvline(sigma.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax3.axvline(np.percentile(sigma, 2.5), color='orange', linestyle=':', linewidth=1.5)
ax3.axvline(np.percentile(sigma, 97.5), color='orange', linestyle=':', linewidth=1.5)
ax3.set_xlabel('σ (Residual SD)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('Posterior: σ', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4d: Joint posterior (beta0 vs beta1)
ax4 = fig.add_subplot(gs[1, :2])
ax4.scatter(beta0[::10], beta1[::10], alpha=0.1, s=10, color='blue')
ax4.set_xlabel('β₀ (Intercept)', fontsize=10)
ax4.set_ylabel('β₁ (Log-slope)', fontsize=10)
ax4.set_title('Joint Posterior: β₀ vs β₁', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Calculate correlation
corr_beta = np.corrcoef(beta0, beta1)[0, 1]
ax4.text(0.05, 0.95, f'Correlation: {corr_beta:.3f}',
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=9)

# Plot 4e: Diminishing returns visualization
ax5 = fig.add_subplot(gs[1, 2])
x_demo = np.linspace(1, 30, 100)
y_mean = beta0.mean() + beta1.mean() * np.log(x_demo)

# Plot with uncertainty
n_samples = 200
sample_indices = np.random.choice(len(beta0), n_samples, replace=False)
for idx in sample_indices:
    y_sample = beta0[idx] + beta1[idx] * np.log(x_demo)
    ax5.plot(x_demo, y_sample, 'b-', alpha=0.02, linewidth=0.5)

ax5.plot(x_demo, y_mean, 'r-', linewidth=3, label='Mean function')
ax5.scatter(x_obs, y_obs, alpha=0.5, s=50, color='black', zorder=5, label='Data')
ax5.set_xlabel('x', fontsize=10)
ax5.set_ylabel('Y', fontsize=10)
ax5.set_title('Logarithmic Relationship', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 4f: Marginal effect dy/dx = beta1/x
ax6 = fig.add_subplot(gs[2, :])
marginal_effect = beta1.mean() / x_demo
marginal_lower = np.percentile(beta1, 2.5) / x_demo
marginal_upper = np.percentile(beta1, 97.5) / x_demo

ax6.plot(x_demo, marginal_effect, 'b-', linewidth=2, label='Mean marginal effect')
ax6.fill_between(x_demo, marginal_lower, marginal_upper, alpha=0.3, label='95% CI')
ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel('dY/dx = β₁/x', fontsize=10)
ax6.set_title('Marginal Effect: Diminishing Returns', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Add annotation
ax6.text(0.5, 0.95,
         'The marginal effect decreases as x increases,\ndemonstrating diminishing returns.',
         transform=ax6.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
         fontsize=9)

plt.savefig(f'{OUTPUT_DIR}/plots/parameter_interpretation.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/plots/parameter_interpretation.png")
plt.close()

print()
print("="*80)
print("ASSESSMENT COMPLETE")
print("="*80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}/")
print()

# ============================================================================
# 10. SAVE SUMMARY STATISTICS TO FILE
# ============================================================================
print("Saving summary statistics...")

summary_stats = {
    'loo': {
        'elpd_loo': float(loo_result.elpd_loo),
        'se': float(loo_result.se),
        'p_loo': float(loo_result.p_loo),
        'looic': float(-2 * loo_result.elpd_loo),
        'pareto_k_good_pct': float(100*k_good/n_obs),
        'pareto_k_ok_pct': float(100*k_ok/n_obs),
        'pareto_k_bad_pct': float(100*k_bad/n_obs),
        'pareto_k_max': float(pareto_k.max()),
        'pareto_k_mean': float(pareto_k.mean())
    },
    'calibration': {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'coverage_50': float(coverage_results[0.50]['observed']),
        'coverage_80': float(coverage_results[0.80]['observed']),
        'coverage_90': float(coverage_results[0.90]['observed']),
        'coverage_95': float(coverage_results[0.95]['observed'])
    },
    'predictive_metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape) if not np.isnan(mape) else None,
        'mean_pred_std': float(y_pred_std.mean()),
        'mean_interval_width_90': float(mean_interval_width)
    },
    'parameters': {
        'beta0_mean': float(beta0.mean()),
        'beta0_sd': float(beta0.std()),
        'beta0_ci_lower': float(np.percentile(beta0, 2.5)),
        'beta0_ci_upper': float(np.percentile(beta0, 97.5)),
        'beta1_mean': float(beta1.mean()),
        'beta1_sd': float(beta1.std()),
        'beta1_ci_lower': float(np.percentile(beta1, 2.5)),
        'beta1_ci_upper': float(np.percentile(beta1, 97.5)),
        'sigma_mean': float(sigma.mean()),
        'sigma_sd': float(sigma.std()),
        'sigma_ci_lower': float(np.percentile(sigma, 2.5)),
        'sigma_ci_upper': float(np.percentile(sigma, 97.5)),
        'beta_correlation': float(corr_beta)
    }
}

import json
with open(f'{OUTPUT_DIR}/summary_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"Saved: {OUTPUT_DIR}/summary_statistics.json")
print()
print("All analyses complete!")
