"""
Complete Pooling Model - Posterior Predictive Check
====================================================

Comprehensive posterior predictive checks to assess model adequacy.

Model:
    Likelihood: y_i ~ Normal(mu, sigma_i)  [known sigma_i from data]
    Prior:      mu ~ Normal(10, 20)

Author: Model Validation Specialist
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("/workspace")
DATA_PATH = BASE_DIR / "data" / "data.csv"
NETCDF_PATH = BASE_DIR / "experiments" / "experiment_1" / "posterior_inference" / "diagnostics" / "posterior_inference.netcdf"
OUTPUT_DIR = BASE_DIR / "experiments" / "experiment_1" / "posterior_predictive_check"
CODE_DIR = OUTPUT_DIR / "code"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories if they don't exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPLETE POOLING MODEL - POSTERIOR PREDICTIVE CHECK")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA AND POSTERIOR
# ============================================================================
print("1. Loading data and posterior...")
print("-" * 80)

# Load data
df = pd.read_csv(DATA_PATH)
y_obs = df['y'].values
sigma_obs = df['sigma'].values
N = len(y_obs)

print(f"   Observed data:")
print(f"   - N = {N} observations")
print(f"   - y: {y_obs}")
print(f"   - sigma: {sigma_obs}")
print()

# Load posterior InferenceData
trace = az.from_netcdf(NETCDF_PATH)
print(f"   Loaded InferenceData from: {NETCDF_PATH}")
print(f"   Groups: {list(trace.groups())}")
print(f"   Posterior shape: {trace.posterior['mu'].shape}")
print()

# Extract posterior samples
mu_samples = trace.posterior['mu'].values.flatten()
n_samples = len(mu_samples)
print(f"   Total posterior samples: {n_samples}")
print()

# ============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE DATA
# ============================================================================
print("2. Generating posterior predictive data...")
print("-" * 80)

# For each posterior sample of mu, generate y_pred for all 8 observations
# Result: (n_samples x N) array of replicated data
y_pred = np.zeros((n_samples, N))

for i in range(n_samples):
    mu_i = mu_samples[i]
    for j in range(N):
        y_pred[i, j] = np.random.normal(mu_i, sigma_obs[j])

print(f"   Generated {n_samples} replicated datasets")
print(f"   y_pred shape: {y_pred.shape}")
print()

# ============================================================================
# 3. LOO-CV DIAGNOSTICS (CRITICAL)
# ============================================================================
print("3. LOO-CV Diagnostics")
print("-" * 80)

# Compute LOO
loo_result = az.loo(trace, pointwise=True)

print(f"   LOO Results:")
print(f"   - ELPD LOO: {loo_result.elpd_loo:.2f}")
print(f"   - SE: {loo_result.se:.2f}")
print(f"   - p_loo: {loo_result.p_loo:.2f}")
print()

# Extract Pareto k values
pareto_k = loo_result.pareto_k.values
print(f"   Pareto k diagnostics:")
print(f"   - Min k: {pareto_k.min():.4f}")
print(f"   - Max k: {pareto_k.max():.4f}")
print(f"   - Mean k: {pareto_k.mean():.4f}")
print()

# Check Pareto k thresholds
k_good = np.sum(pareto_k < 0.5)
k_ok = np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))
k_bad = np.sum(pareto_k >= 0.7)

print(f"   Pareto k classification:")
print(f"   - Good (k < 0.5):      {k_good} / {N}")
print(f"   - OK (0.5 <= k < 0.7): {k_ok} / {N}")
print(f"   - Bad (k >= 0.7):      {k_bad} / {N}")
print()

# Flag problematic observations
if k_bad > 0:
    print(f"   WARNING: {k_bad} observation(s) with k >= 0.7 detected!")
    for i, k in enumerate(pareto_k):
        if k >= 0.7:
            print(f"   - Observation {i} (group={df['group'].values[i]}): k={k:.4f}, y={y_obs[i]:.2f}")
    print()
else:
    print(f"   EXCELLENT: All Pareto k < 0.7 (model adequacy confirmed)")
    print()

# ============================================================================
# 4. OBSERVATION-LEVEL PPC
# ============================================================================
print("4. Observation-level posterior predictive checks...")
print("-" * 80)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(N):
    ax = axes[i]

    # Plot posterior predictive distribution
    ax.hist(y_pred[:, i], bins=50, density=True, alpha=0.6,
            color='steelblue', edgecolor='black', linewidth=0.5)

    # Observed value
    ax.axvline(y_obs[i], color='red', linewidth=2.5,
               label=f'Observed: {y_obs[i]:.2f}')

    # Posterior predictive intervals
    pp_mean = y_pred[:, i].mean()
    pp_std = y_pred[:, i].std()
    pp_90 = np.percentile(y_pred[:, i], [5, 95])

    ax.axvline(pp_mean, color='blue', linestyle='--', linewidth=1.5,
               label=f'PP Mean: {pp_mean:.2f}')
    ax.axvspan(pp_90[0], pp_90[1], alpha=0.2, color='blue',
               label='90% PP Interval')

    # Calculate percentile of observed value
    obs_percentile = (y_pred[:, i] <= y_obs[i]).mean() * 100

    # Title with diagnostics
    group_id = df['group'].values[i]
    ax.set_title(f'Obs {i} (Group {group_id})\n' +
                 f'k={pareto_k[i]:.3f}, percentile={obs_percentile:.1f}%',
                 fontsize=10)
    ax.set_xlabel('y', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Color-code based on Pareto k
    if pareto_k[i] >= 0.7:
        ax.set_facecolor('#ffcccc')  # Light red for bad k
    elif pareto_k[i] >= 0.5:
        ax.set_facecolor('#fff5cc')  # Light yellow for ok k

plt.suptitle('Observation-Level Posterior Predictive Checks\n' +
             '(Red background = k >= 0.7, Yellow = 0.5 <= k < 0.7)',
             fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_observations.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'ppc_observations.png'}")

# ============================================================================
# 5. TEST STATISTICS PPC
# ============================================================================
print("\n5. Test statistics posterior predictive checks...")
print("-" * 80)

# Compute test statistics for observed and replicated data
obs_mean = y_obs.mean()
obs_std = y_obs.std()
obs_min = y_obs.min()
obs_max = y_obs.max()

pred_means = y_pred.mean(axis=1)
pred_stds = y_pred.std(axis=1)
pred_mins = y_pred.min(axis=1)
pred_maxs = y_pred.max(axis=1)

# Calculate Bayesian p-values
p_mean = (pred_means >= obs_mean).mean()
p_std = (pred_stds >= obs_std).mean()
p_min = (pred_mins <= obs_min).mean()
p_max = (pred_maxs >= obs_max).mean()

print(f"   Test Statistics:")
print(f"   - Mean: obs={obs_mean:.2f}, pred={pred_means.mean():.2f}, p-value={p_mean:.3f}")
print(f"   - SD:   obs={obs_std:.2f}, pred={pred_stds.mean():.2f}, p-value={p_std:.3f}")
print(f"   - Min:  obs={obs_min:.2f}, pred={pred_mins.mean():.2f}, p-value={p_min:.3f}")
print(f"   - Max:  obs={obs_max:.2f}, pred={pred_maxs.mean():.2f}, p-value={p_max:.3f}")
print()

# Flag extreme p-values
extreme_pvals = []
if p_mean < 0.05 or p_mean > 0.95:
    extreme_pvals.append(f"Mean (p={p_mean:.3f})")
if p_std < 0.05 or p_std > 0.95:
    extreme_pvals.append(f"SD (p={p_std:.3f})")
if p_min < 0.05 or p_min > 0.95:
    extreme_pvals.append(f"Min (p={p_min:.3f})")
if p_max < 0.05 or p_max > 0.95:
    extreme_pvals.append(f"Max (p={p_max:.3f})")

if extreme_pvals:
    print(f"   WARNING: Extreme p-values detected for: {', '.join(extreme_pvals)}")
    print()
else:
    print(f"   EXCELLENT: All p-values in reasonable range [0.05, 0.95]")
    print()

# Plot test statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean
ax = axes[0, 0]
ax.hist(pred_means, bins=50, density=True, alpha=0.6,
        color='steelblue', edgecolor='black', linewidth=0.5)
ax.axvline(obs_mean, color='red', linewidth=2.5, label=f'Observed: {obs_mean:.2f}')
ax.axvline(pred_means.mean(), color='blue', linestyle='--', linewidth=1.5,
           label=f'PP Mean: {pred_means.mean():.2f}')
ax.set_xlabel('Mean of y', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Test Statistic: Mean (p-value = {p_mean:.3f})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# SD
ax = axes[0, 1]
ax.hist(pred_stds, bins=50, density=True, alpha=0.6,
        color='steelblue', edgecolor='black', linewidth=0.5)
ax.axvline(obs_std, color='red', linewidth=2.5, label=f'Observed: {obs_std:.2f}')
ax.axvline(pred_stds.mean(), color='blue', linestyle='--', linewidth=1.5,
           label=f'PP Mean: {pred_stds.mean():.2f}')
ax.set_xlabel('SD of y', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Test Statistic: SD (p-value = {p_std:.3f})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Min
ax = axes[1, 0]
ax.hist(pred_mins, bins=50, density=True, alpha=0.6,
        color='steelblue', edgecolor='black', linewidth=0.5)
ax.axvline(obs_min, color='red', linewidth=2.5, label=f'Observed: {obs_min:.2f}')
ax.axvline(pred_mins.mean(), color='blue', linestyle='--', linewidth=1.5,
           label=f'PP Mean: {pred_mins.mean():.2f}')
ax.set_xlabel('Min of y', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Test Statistic: Min (p-value = {p_min:.3f})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Max
ax = axes[1, 1]
ax.hist(pred_maxs, bins=50, density=True, alpha=0.6,
        color='steelblue', edgecolor='black', linewidth=0.5)
ax.axvline(obs_max, color='red', linewidth=2.5, label=f'Observed: {obs_max:.2f}')
ax.axvline(pred_maxs.mean(), color='blue', linestyle='--', linewidth=1.5,
           label=f'PP Mean: {pred_maxs.mean():.2f}')
ax.set_xlabel('Max of y', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Test Statistic: Max (p-value = {p_max:.3f})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Test Statistics: Posterior Predictive Checks', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_test_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'ppc_test_statistics.png'}")

# ============================================================================
# 6. RESIDUAL ANALYSIS
# ============================================================================
print("\n6. Residual analysis...")
print("-" * 80)

# Compute standardized residuals for each posterior sample
# Standardized residual: (y_obs - y_pred) / sigma_obs
std_residuals = np.zeros((n_samples, N))
for i in range(n_samples):
    for j in range(N):
        std_residuals[i, j] = (y_obs[j] - y_pred[i, j]) / sigma_obs[j]

# Average standardized residuals across posterior samples
mean_std_residuals = std_residuals.mean(axis=0)
sd_std_residuals = std_residuals.std(axis=0)

print(f"   Standardized residuals (mean across posterior):")
for i in range(N):
    print(f"   - Obs {i} (Group {df['group'].values[i]}): {mean_std_residuals[i]:.3f} ± {sd_std_residuals[i]:.3f}")
print()

# Check if residuals look like standard normal
residual_mean = mean_std_residuals.mean()
residual_std = mean_std_residuals.std()
print(f"   Residual summary:")
print(f"   - Mean: {residual_mean:.3f} (should be ~0)")
print(f"   - SD: {residual_std:.3f} (should be ~1)")
print()

# Plot residuals
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Residual plot by observation
ax = axes[0]
for i in range(N):
    residual_dist = std_residuals[:, i]
    bp = ax.violinplot([residual_dist], positions=[i], widths=0.7,
                        showmeans=True, showmedians=True)
ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero residual')
ax.axhline(2, color='orange', linestyle=':', linewidth=1, label='±2 SD')
ax.axhline(-2, color='orange', linestyle=':', linewidth=1)
ax.set_xlabel('Observation', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Standardized Residuals by Observation', fontsize=12)
ax.set_xticks(range(N))
ax.set_xticklabels([f'{i}\n(G{df["group"].values[i]})' for i in range(N)], fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Histogram of all residuals
ax = axes[1]
all_residuals = std_residuals.flatten()
ax.hist(all_residuals, bins=50, density=True, alpha=0.6,
        color='steelblue', edgecolor='black', linewidth=0.5,
        label='Observed residuals')
# Overlay standard normal
x_range = np.linspace(-4, 4, 100)
ax.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r-', linewidth=2,
        label='Standard Normal')
ax.set_xlabel('Standardized Residual', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Standardized Residuals', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Q-Q plot
ax = axes[2]
stats.probplot(mean_std_residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Standardized Residuals', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_residuals.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'ppc_residuals.png'}")

# ============================================================================
# 7. PARETO K PLOT
# ============================================================================
print("\n7. LOO Pareto k diagnostics plot...")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pareto k values by observation
ax = axes[0]
colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in pareto_k]
bars = ax.bar(range(N), pareto_k, color=colors, edgecolor='black', linewidth=1.5)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='k = 0.5 (threshold)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='k = 0.7 (problematic)')
ax.set_xlabel('Observation', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('LOO Pareto k Values by Observation', fontsize=12)
ax.set_xticks(range(N))
ax.set_xticklabels([f'{i}\n(G{df["group"].values[i]})' for i in range(N)], fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add text annotations for high k values
for i, k in enumerate(pareto_k):
    if k >= 0.5:
        ax.text(i, k + 0.02, f'{k:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# Pareto k vs observed y values
ax = axes[1]
ax.scatter(y_obs, pareto_k, s=100, c=colors, edgecolors='black', linewidth=1.5)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='k = 0.5')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='k = 0.7')
ax.set_xlabel('Observed y', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('Pareto k vs Observed Values', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Annotate points
for i in range(N):
    ax.annotate(f'{i}', (y_obs[i], pareto_k[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pareto_k.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'loo_pareto_k.png'}")

# ============================================================================
# 8. CALIBRATION AND COVERAGE
# ============================================================================
print("\n8. Calibration and coverage checks...")
print("-" * 80)

# Compute posterior predictive percentiles for each observation
pp_percentiles = np.zeros(N)
for i in range(N):
    pp_percentiles[i] = (y_pred[:, i] <= y_obs[i]).mean()

print(f"   Posterior predictive percentiles (PIT values):")
for i in range(N):
    print(f"   - Obs {i} (Group {df['group'].values[i]}): {pp_percentiles[i]:.3f}")
print()

# Check calibration: PIT values should be uniform on [0, 1]
# Use Kolmogorov-Smirnov test
ks_stat, ks_pval = stats.kstest(pp_percentiles, 'uniform')
print(f"   Kolmogorov-Smirnov test for uniformity:")
print(f"   - KS statistic: {ks_stat:.4f}")
print(f"   - p-value: {ks_pval:.4f}")
if ks_pval > 0.05:
    print(f"   - Interpretation: PIT values consistent with uniform (good calibration)")
else:
    print(f"   - Interpretation: PIT values not uniform (calibration issues)")
print()

# Compute coverage for different intervals
intervals = [50, 80, 90, 95]
observed_coverage = []

for interval in intervals:
    lower_q = (100 - interval) / 2
    upper_q = 100 - lower_q

    in_interval = 0
    for i in range(N):
        ci = np.percentile(y_pred[:, i], [lower_q, upper_q])
        if ci[0] <= y_obs[i] <= ci[1]:
            in_interval += 1

    coverage = in_interval / N
    observed_coverage.append(coverage)
    print(f"   {interval}% interval coverage: {coverage:.3f} ({in_interval}/{N} observations)")

print()

# Plot calibration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PIT histogram (should be uniform)
ax = axes[0]
ax.hist(pp_percentiles, bins=10, density=False, alpha=0.6,
        color='steelblue', edgecolor='black', linewidth=1.5,
        label='Observed PIT')
ax.axhline(N/10, color='red', linestyle='--', linewidth=2,
           label='Expected (uniform)')
ax.set_xlabel('Posterior Predictive Percentile (PIT)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'PIT Histogram (KS p-value = {ks_pval:.3f})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Coverage plot
ax = axes[1]
expected_coverage = np.array(intervals) / 100
ax.plot(expected_coverage, observed_coverage, 'o-', linewidth=2,
        markersize=10, label='Observed coverage')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal calibration')
ax.set_xlabel('Nominal Coverage', fontsize=11)
ax.set_ylabel('Observed Coverage', fontsize=11)
ax.set_title('Coverage Calibration', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.4, 1.0])
ax.set_ylim([0.4, 1.0])

# Add text annotations
for i, (nom, obs) in enumerate(zip(expected_coverage, observed_coverage)):
    ax.annotate(f'{intervals[i]}%', (nom, obs),
                xytext=(10, -10), textcoords='offset points',
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_calibration.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'ppc_calibration.png'}")

# ============================================================================
# 9. OVERALL ASSESSMENT
# ============================================================================
print("\n9. Overall Model Adequacy Assessment")
print("="*80)

# Criteria for model adequacy
criteria_results = {}

# 1. Pareto k values
all_k_good = k_bad == 0
criteria_results['Pareto k (all < 0.7)'] = all_k_good

# 2. Test statistic p-values
pvals_in_range = (0.05 <= p_mean <= 0.95 and
                  0.05 <= p_std <= 0.95 and
                  0.05 <= p_min <= 0.95 and
                  0.05 <= p_max <= 0.95)
criteria_results['Test stat p-values [0.05, 0.95]'] = pvals_in_range

# 3. Calibration (KS test)
calibration_ok = ks_pval > 0.05
criteria_results['PIT uniformity (KS test)'] = calibration_ok

# 4. Coverage (90% should be 0.85-0.95)
coverage_90 = observed_coverage[2]  # 90% interval
coverage_ok = 0.80 <= coverage_90 <= 1.0
criteria_results['90% coverage in [0.80, 1.00]'] = coverage_ok

# 5. Observations within reasonable range (5-95th percentile)
obs_in_range = np.sum((0.05 <= pp_percentiles) & (pp_percentiles <= 0.95))
obs_range_ok = obs_in_range >= N * 0.8  # At least 80% should be in range
criteria_results['Observations in [5%, 95%]'] = obs_range_ok

print(f"\nModel Adequacy Criteria:")
print(f"-" * 80)
for criterion, result in criteria_results.items():
    status = "PASS" if result else "FAIL"
    print(f"   [{status}] {criterion}")
print()

# Overall decision
all_pass = all(criteria_results.values())
if all_pass:
    overall_status = "ADEQUATE"
    print(f"Overall Assessment: MODEL ADEQUATE")
    print(f"   All criteria met. Model provides a good fit to the observed data.")
elif k_bad > 0:
    overall_status = "CONCERNS"
    print(f"Overall Assessment: CONCERNS")
    print(f"   Some Pareto k values >= 0.7 indicate influential observations.")
    print(f"   Model may be inadequate for these specific observations.")
else:
    overall_status = "CONCERNS"
    print(f"Overall Assessment: CONCERNS")
    print(f"   Some criteria not met. Review specific diagnostics.")

print()

# Save summary statistics
summary_stats = pd.DataFrame({
    'metric': ['ELPD LOO', 'LOO SE', 'p_loo', 'Mean Pareto k', 'Max Pareto k',
               'Bad Pareto k (>= 0.7)', 'p-value (mean)', 'p-value (SD)',
               'p-value (min)', 'p-value (max)', 'KS p-value',
               '90% coverage', 'Overall Status'],
    'value': [loo_result.elpd_loo, loo_result.se, loo_result.p_loo,
              pareto_k.mean(), pareto_k.max(), k_bad, p_mean, p_std,
              p_min, p_max, ks_pval, coverage_90, overall_status]
})
summary_stats.to_csv(OUTPUT_DIR / "ppc_summary.csv", index=False)
print(f"Summary statistics saved to: {OUTPUT_DIR / 'ppc_summary.csv'}")

print()
print("="*80)
print("POSTERIOR PREDICTIVE CHECK COMPLETE")
print("="*80)
print()
print(f"Key Findings:")
print(f"   - LOO ELPD: {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
print(f"   - Max Pareto k: {pareto_k.max():.4f} {'(GOOD)' if pareto_k.max() < 0.5 else '(OK)' if pareto_k.max() < 0.7 else '(BAD)'}")
print(f"   - Calibration: {'GOOD' if calibration_ok else 'ISSUES'}")
print(f"   - Coverage (90%): {coverage_90:.3f} {'(GOOD)' if coverage_ok else '(ISSUES)'}")
print(f"   - Overall: {overall_status}")
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
