"""
Posterior Predictive Checks for Hierarchical Model

Tests whether the model can generate data similar to observations.
Includes LOO-PIT, coverage analysis, and model comparison with Model 1.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_context("notebook")
plt.style.use('seaborn-v0_8-darkgrid')

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_2/posterior_predictive_check")
MODEL2_IDATA_PATH = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")
MODEL1_IDATA_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
DATA_PATH = Path("/workspace/data/data.csv")
PLOTS_DIR = BASE_DIR / "plots"

print("="*70)
print("POSTERIOR PREDICTIVE CHECKS - HIERARCHICAL MODEL")
print("="*70)

# Load data
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma_obs = data['sigma'].values
J = len(y_obs)

print(f"\nObserved data:")
print(f"  Studies: {J}")
print(f"  y: {y_obs}")
print(f"  σ: {sigma_obs}")

# Load Model 2 (Hierarchical) InferenceData
print(f"\n{'='*70}")
print("Loading Model 2 (Hierarchical) InferenceData...")
print(f"{'='*70}")
idata_model2 = az.from_netcdf(MODEL2_IDATA_PATH)
print("✓ Loaded!")

# Generate posterior predictive samples
print(f"\nGenerating posterior predictive samples...")

# Extract posterior samples
mu_samples = idata_model2.posterior['mu'].values.flatten()
tau_samples = idata_model2.posterior['tau'].values.flatten()
theta_samples = idata_model2.posterior['theta'].values.reshape(-1, J)  # shape: (n_samples, J)

n_samples = len(mu_samples)
print(f"  Posterior samples: {n_samples}")

# Generate predictive samples
np.random.seed(42)
y_pred = np.zeros((n_samples, J))
for i in range(n_samples):
    y_pred[i, :] = np.random.normal(theta_samples[i, :], sigma_obs)

print(f"  Posterior predictive shape: {y_pred.shape}")

# Summary statistics
print(f"\n{'='*70}")
print("Posterior Predictive Summary")
print(f"{'='*70}")

for j in range(J):
    y_pred_mean = y_pred[:, j].mean()
    y_pred_sd = y_pred[:, j].std()
    y_pred_hdi = np.percentile(y_pred[:, j], [2.5, 97.5])
    print(f"\nStudy {j+1}:")
    print(f"  Observed: y = {y_obs[j]:6.1f}")
    print(f"  Predicted: {y_pred_mean:6.2f} ± {y_pred_sd:5.2f}")
    print(f"  95% PI: [{y_pred_hdi[0]:6.2f}, {y_pred_hdi[1]:6.2f}]")
    contains = y_pred_hdi[0] <= y_obs[j] <= y_pred_hdi[1]
    print(f"  Contains observed: {'YES' if contains else 'NO'}")

# Compute LOO for Model 2
print(f"\n{'='*70}")
print("Computing LOO-CV...")
print(f"{'='*70}")

loo_model2 = az.loo(idata_model2, pointwise=True)
print(f"\nModel 2 (Hierarchical):")
print(f"  ELPD_LOO: {loo_model2.elpd_loo:.2f} ± {loo_model2.se:.2f}")
print(f"  p_LOO: {loo_model2.p_loo:.2f}")
print(f"  Effective parameters: {loo_model2.p_loo:.1f}")

# Check for high Pareto-k values
pareto_k = loo_model2.pareto_k
n_high_k = (pareto_k > 0.7).sum()
print(f"\n  Pareto-k diagnostics:")
print(f"    Max k: {pareto_k.max():.3f}")
print(f"    # high k (> 0.7): {n_high_k}/{J}")
if n_high_k > 0:
    print(f"    WARNING: Some observations have high k values")
else:
    print(f"    ✓ All k values acceptable")

# LOO-PIT for calibration
print(f"\n{'='*70}")
print("LOO-PIT Analysis")
print(f"{'='*70}")

# Compute PIT values
pit_values = np.zeros(J)
for j in range(J):
    # For each observation, compute probability that y_pred < y_obs
    # excluding that observation (approximated by full posterior)
    pit_values[j] = (y_pred[:, j] < y_obs[j]).mean()

print(f"\nPIT values (should be uniform [0,1]):")
for j in range(J):
    print(f"  Study {j+1}: {pit_values[j]:.3f}")

# Test uniformity with KS test
from scipy import stats
ks_stat, ks_p = stats.kstest(pit_values, 'uniform')
print(f"\nKolmogorov-Smirnov test for uniformity:")
print(f"  Statistic: {ks_stat:.3f}")
print(f"  p-value: {ks_p:.3f}")
print(f"  Uniform: {'YES' if ks_p > 0.05 else 'NO (potential miscalibration)'}")

# Coverage analysis
print(f"\n{'='*70}")
print("Coverage Analysis")
print(f"{'='*70}")

credible_levels = [0.50, 0.68, 0.90, 0.95]
for level in credible_levels:
    alpha = 1 - level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    coverage = 0
    for j in range(J):
        lower = np.quantile(y_pred[:, j], lower_q)
        upper = np.quantile(y_pred[:, j], upper_q)
        if lower <= y_obs[j] <= upper:
            coverage += 1

    empirical_coverage = coverage / J
    print(f"\n{100*level:.0f}% Credible Interval:")
    print(f"  Expected coverage: {100*level:.0f}%")
    print(f"  Empirical coverage: {100*empirical_coverage:.0f}% ({coverage}/{J} studies)")
    print(f"  Calibrated: {'YES' if abs(empirical_coverage - level) < 0.15 else 'REVIEW'}")

# Load Model 1 for comparison
print(f"\n{'='*70}")
print("Model Comparison with Model 1")
print(f"{'='*70}")

try:
    idata_model1 = az.from_netcdf(MODEL1_IDATA_PATH)
    print("✓ Model 1 loaded")

    loo_model1 = az.loo(idata_model1, pointwise=True)
    print(f"\nModel 1 (Fixed-Effect):")
    print(f"  ELPD_LOO: {loo_model1.elpd_loo:.2f} ± {loo_model1.se:.2f}")
    print(f"  p_LOO: {loo_model1.p_loo:.2f}")

    # Compare models
    loo_compare = az.compare({'Model 1 (Fixed)': idata_model1, 'Model 2 (Hierarchical)': idata_model2}, ic='loo')
    print(f"\nLOO Comparison:")
    print(loo_compare)

    # Interpret
    elpd_diff = loo_compare.loc['Model 2 (Hierarchical)', 'elpd_diff']
    se_diff = loo_compare.loc['Model 2 (Hierarchical)', 'se']

    print(f"\nInterpretation:")
    if abs(elpd_diff) < 2 * se_diff:
        print(f"  No substantial difference in predictive performance")
        print(f"  ΔELPD = {elpd_diff:.2f} ± {se_diff:.2f} (within 2 SE)")
        print(f"  Recommendation: Choose simpler Model 1")
    elif elpd_diff > 0:
        print(f"  Model 1 has better predictive performance")
        print(f"  ΔELPD = {-elpd_diff:.2f} ± {se_diff:.2f}")
        print(f"  Recommendation: Use Model 1")
    else:
        print(f"  Model 2 has better predictive performance")
        print(f"  ΔELPD = {-elpd_diff:.2f} ± {se_diff:.2f}")
        print(f"  Recommendation: Use Model 2 if difference substantial")

except Exception as e:
    print(f"Could not load Model 1: {e}")
    print("Skipping model comparison")
    loo_compare = None

# Visualizations
print(f"\n{'='*70}")
print("Creating visualizations...")
print(f"{'='*70}")

# Figure 1: Posterior predictive check
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(J):
    ax = axes[j]
    ax.hist(y_pred[:, j], bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Predictive')
    ax.axvline(y_obs[j], color='red', linestyle='--', linewidth=2, label=f'Observed={y_obs[j]:.0f}')
    ax.axvline(y_pred[:, j].mean(), color='blue', linestyle=':', linewidth=2, label=f'Mean={y_pred[:, j].mean():.1f}')
    ax.set_xlabel('y', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Study {j+1}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Distributions vs Observed Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_predictive_distributions.png', dpi=300, bbox_inches='tight')
print("  Saved: posterior_predictive_distributions.png")
plt.close()

# Figure 2: LOO-PIT
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(pit_values, bins=10, density=True, alpha=0.7, color='green', edgecolor='black', label='Empirical PIT')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform expectation')
ax.set_xlabel('PIT Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'LOO-PIT Histogram\n(KS p={ks_p:.3f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# QQ plot
ax = axes[1]
theoretical_quantiles = np.linspace(0, 1, J+2)[1:-1]
empirical_quantiles = np.sort(pit_values)
ax.scatter(theoretical_quantiles, empirical_quantiles, s=100, alpha=0.7, color='green', edgecolor='black')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
ax.set_xlabel('Theoretical Quantiles (Uniform)', fontsize=12)
ax.set_ylabel('Empirical Quantiles', fontsize=12)
ax.set_title('PIT Q-Q Plot', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('LOO-PIT Calibration Check', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loo_pit_check.png', dpi=300, bbox_inches='tight')
print("  Saved: loo_pit_check.png")
plt.close()

# Figure 3: Coverage plot
fig, ax = plt.subplots(figsize=(10, 6))

expected = [level * 100 for level in credible_levels]
empirical = []
for level in credible_levels:
    alpha = 1 - level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    coverage = 0
    for j in range(J):
        lower = np.quantile(y_pred[:, j], lower_q)
        upper = np.quantile(y_pred[:, j], upper_q)
        if lower <= y_obs[j] <= upper:
            coverage += 1
    empirical.append(coverage / J * 100)

ax.plot(expected, empirical, marker='o', markersize=10, linewidth=2, label='Empirical coverage')
ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect calibration')
ax.fill_between([0, 100], [0, 100], [0, 85], alpha=0.2, color='red', label='Under-coverage')
ax.fill_between([0, 100], [0, 100], [15, 115], alpha=0.2, color='yellow', label='Over-coverage')
ax.set_xlabel('Expected Coverage (%)', fontsize=12)
ax.set_ylabel('Empirical Coverage (%)', fontsize=12)
ax.set_title('Calibration: Expected vs Empirical Coverage', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([40, 105])
ax.set_ylim([40, 105])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_calibration.png', dpi=300, bbox_inches='tight')
print("  Saved: coverage_calibration.png")
plt.close()

# Figure 4: Residual analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Residuals (observed - predicted mean)
residuals = y_obs - y_pred.mean(axis=0)
standardized_residuals = residuals / y_pred.std(axis=0)

ax = axes[0]
ax.scatter(y_pred.mean(axis=0), residuals, s=100, alpha=0.7, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted y', fontsize=12)
ax.set_ylabel('Residual (observed - predicted)', fontsize=12)
ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Standardized residuals
ax = axes[1]
ax.scatter(range(J), standardized_residuals, s=100, alpha=0.7, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, label='±2 SD')
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5)
ax.set_xlabel('Study Index', fontsize=12)
ax.set_ylabel('Standardized Residual', fontsize=12)
ax.set_title('Standardized Residuals', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Q-Q plot of residuals
ax = axes[2]
stats.probplot(standardized_residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Standardized Residuals)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residual_analysis.png', dpi=300, bbox_inches='tight')
print("  Saved: residual_analysis.png")
plt.close()

# Figure 5: Model comparison (if available)
if loo_compare is not None:
    fig, ax = plt.subplots(figsize=(10, 6))

    models = loo_compare.index
    elpd = loo_compare['elpd_loo'].values
    se = loo_compare['se'].values

    colors = ['steelblue', 'coral']
    ax.barh(range(len(models)), elpd, xerr=se, color=colors, edgecolor='black', alpha=0.7, capsize=5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('ELPD_LOO', fontsize=12)
    ax.set_title('Model Comparison (LOO-CV)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add text annotation
    diff = loo_compare.loc[models[1], 'elpd_diff']
    se_diff = loo_compare.loc[models[1], 'se']
    ax.text(0.05, 0.95, f'ΔELPD = {abs(diff):.2f} ± {se_diff:.2f}\n(within 2 SE: {"YES" if abs(diff) < 2*se_diff else "NO"})',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'model_comparison_loo.png', dpi=300, bbox_inches='tight')
    print("  Saved: model_comparison_loo.png")
    plt.close()

# Save results
results = {
    'loo_elpd': float(loo_model2.elpd_loo),
    'loo_se': float(loo_model2.se),
    'loo_p': float(loo_model2.p_loo),
    'max_pareto_k': float(pareto_k.max()),
    'n_high_k': int(n_high_k),
    'ks_statistic': float(ks_stat),
    'ks_p_value': float(ks_p),
    'pit_uniform': bool(ks_p > 0.05),
    'coverage_50': float(empirical[0]),
    'coverage_68': float(empirical[1]),
    'coverage_90': float(empirical[2]),
    'coverage_95': float(empirical[3])
}

if loo_compare is not None:
    results['elpd_diff'] = float(loo_compare.loc['Model 2 (Hierarchical)', 'elpd_diff'])
    results['se_diff'] = float(loo_compare.loc['Model 2 (Hierarchical)', 'se'])
    results['models_similar'] = bool(abs(results['elpd_diff']) < 2 * results['se_diff'])

results_path = BASE_DIR / 'ppc_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {results_path}")

# Assessment
print(f"\n{'='*70}")
print("ASSESSMENT")
print(f"{'='*70}")

calibrated = ks_p > 0.05
good_coverage = all(abs(emp - exp) < 15 for emp, exp in zip(empirical, expected))
low_pareto_k = n_high_k == 0

print(f"\nCriteria:")
print(f"  PIT uniformity (p > 0.05): {'YES' if calibrated else 'NO'}")
print(f"  Coverage calibrated: {'YES' if good_coverage else 'REVIEW'}")
print(f"  Low Pareto-k values: {'YES' if low_pareto_k else 'REVIEW'}")

if loo_compare is not None:
    similar_to_model1 = abs(results['elpd_diff']) < 2 * results['se_diff']
    print(f"  Similar to Model 1: {'YES' if similar_to_model1 else 'NO'}")

if calibrated and good_coverage and low_pareto_k:
    decision = "GOOD FIT"
    reasoning = "Model is well-calibrated and generates plausible data"
elif calibrated and good_coverage:
    decision = "ACCEPTABLE"
    reasoning = "Model calibrated but some Pareto-k warnings"
else:
    decision = "REVIEW"
    reasoning = "Calibration issues detected"

print(f"\n{'='*70}")
print(f"DECISION: {decision}")
print(f"REASONING: {reasoning}")
print(f"{'='*70}")

print("\nPosterior predictive checks complete!")
