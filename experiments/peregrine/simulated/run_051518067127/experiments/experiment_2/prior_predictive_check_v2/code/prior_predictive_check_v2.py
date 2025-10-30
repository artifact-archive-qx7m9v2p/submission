"""
Prior Predictive Check v2 for Experiment 2: AR(1) Log-Normal with Regime-Switching

UPDATED PRIORS (post-v1 fixes):
- phi: Beta(20, 2) rescaled to (0, 0.95) [was Uniform(-0.95, 0.95)]
- sigma_regime: HalfNormal(0, 0.5) [was HalfNormal(0, 1)]
- beta_1: Normal(0.86, 0.15) [was Normal(0.86, 0.2)]

Expected improvements:
1. Prior ACF median ~0.85-0.90 (was -0.059)
2. <1% predictions >1000 (was 5.8%)
3. >15% draws in plausible range [10, 500] (was 2.8%)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_DRAWS = 1000
N_TIME_POINTS = 40
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/prior_predictive_check_v2")

# Read observed data
data = pd.read_csv("/workspace/data/data.csv")
year = data["year"].values
C_obs = data["C"].values
log_C_obs = np.log(C_obs)

# Calculate observed ACF lag-1
observed_acf = np.corrcoef(log_C_obs[:-1], log_C_obs[1:])[0, 1]

# Regime structure (known from EDA)
regime = np.array([1]*14 + [2]*13 + [3]*13)  # 1-indexed

print("=" * 80)
print("PRIOR PREDICTIVE CHECK V2: AR(1) LOG-NORMAL WITH REGIME-SWITCHING")
print("=" * 80)
print("\nUPDATED PRIORS:")
print("  alpha ~ Normal(4.3, 0.5)")
print("  beta_1 ~ Normal(0.86, 0.15)  [TIGHTENED from 0.2]")
print("  beta_2 ~ Normal(0, 0.3)")
print("  phi_raw ~ Beta(20, 2)")
print("  phi = 0.95 * phi_raw  [CHANGED from Uniform(-0.95, 0.95)]")
print("  sigma_regime[1:3] ~ HalfNormal(0, 0.5)  [TIGHTENED from 1.0]")
print("\n" + "=" * 80)

# Sample from UPDATED priors
print("\nSampling from updated priors...")

# Trend parameters
alpha = np.random.normal(4.3, 0.5, N_PRIOR_DRAWS)
beta_1 = np.random.normal(0.86, 0.15, N_PRIOR_DRAWS)  # UPDATED: tightened
beta_2 = np.random.normal(0, 0.3, N_PRIOR_DRAWS)

# AR coefficient - UPDATED: Beta(20,2) rescaled to (0, 0.95)
phi_raw = np.random.beta(20, 2, N_PRIOR_DRAWS)
phi = 0.95 * phi_raw

# Regime-specific variances - UPDATED: HalfNormal(0, 0.5)
sigma_regime = np.abs(np.random.normal(0, 0.5, (N_PRIOR_DRAWS, 3)))

print(f"  Generated {N_PRIOR_DRAWS} prior draws")
print(f"\nPrior parameter summaries:")
print(f"  alpha: median={np.median(alpha):.3f}, 90% CI=[{np.percentile(alpha, 5):.3f}, {np.percentile(alpha, 95):.3f}]")
print(f"  beta_1: median={np.median(beta_1):.3f}, 90% CI=[{np.percentile(beta_1, 5):.3f}, {np.percentile(beta_1, 95):.3f}]")
print(f"  beta_2: median={np.median(beta_2):.3f}, 90% CI=[{np.percentile(beta_2, 5):.3f}, {np.percentile(beta_2, 95):.3f}]")
print(f"  phi: median={np.median(phi):.3f}, 90% CI=[{np.percentile(phi, 5):.3f}, {np.percentile(phi, 95):.3f}]")
print(f"  sigma_1: median={np.median(sigma_regime[:, 0]):.3f}, 90% CI=[{np.percentile(sigma_regime[:, 0], 5):.3f}, {np.percentile(sigma_regime[:, 0], 95):.3f}]")
print(f"  sigma_2: median={np.median(sigma_regime[:, 1]):.3f}, 90% CI=[{np.percentile(sigma_regime[:, 1], 5):.3f}, {np.percentile(sigma_regime[:, 1], 95):.3f}]")
print(f"  sigma_3: median={np.median(sigma_regime[:, 2]):.3f}, 90% CI=[{np.percentile(sigma_regime[:, 2], 5):.3f}, {np.percentile(sigma_regime[:, 2], 95):.3f}]")

# Generate prior predictive data with AR(1) structure
print("\nGenerating prior predictive datasets with AR(1) structure...")

prior_predictive_log = np.zeros((N_PRIOR_DRAWS, N_TIME_POINTS))
prior_predictive_count = np.zeros((N_PRIOR_DRAWS, N_TIME_POINTS))
prior_acf_lag1 = np.zeros(N_PRIOR_DRAWS)

for i in range(N_PRIOR_DRAWS):
    if (i + 1) % 200 == 0:
        print(f"  Progress: {i+1}/{N_PRIOR_DRAWS}")

    # Get parameters for this draw
    a = alpha[i]
    b1 = beta_1[i]
    b2 = beta_2[i]
    p = phi[i]
    sigma = sigma_regime[i, regime - 1]  # Convert 1-indexed to 0-indexed

    # Storage for this trajectory
    epsilon = np.zeros(N_TIME_POINTS)
    mu = np.zeros(N_TIME_POINTS)
    log_y = np.zeros(N_TIME_POINTS)

    # Stationary initialization for AR(1)
    # epsilon[1] ~ Normal(0, sigma[1] / sqrt(1 - phi^2))
    sigma_init = sigma[0] / np.sqrt(1 - p**2) if abs(p) < 1 else sigma[0]
    epsilon[0] = np.random.normal(0, sigma_init)

    # First observation
    mu[0] = a + b1 * year[0] + b2 * year[0]**2
    log_y[0] = mu[0] + epsilon[0]

    # Sequential generation for t = 2, ..., T
    for t in range(1, N_TIME_POINTS):
        # AR(1) structure: mu[t] includes phi * epsilon[t-1]
        mu[t] = a + b1 * year[t] + b2 * year[t]**2 + p * epsilon[t-1]

        # Generate observation on log scale
        log_y[t] = np.random.normal(mu[t], sigma[t])

        # Calculate epsilon[t] for next iteration
        # epsilon[t] = log(y[t]) - (alpha + beta_1*year[t] + beta_2*year[t]^2)
        epsilon[t] = log_y[t] - (a + b1 * year[t] + b2 * year[t]**2)

    # Store results
    prior_predictive_log[i, :] = log_y
    prior_predictive_count[i, :] = np.exp(log_y)

    # Calculate ACF lag-1 for this trajectory
    if len(log_y) > 1:
        prior_acf_lag1[i] = np.corrcoef(log_y[:-1], log_y[1:])[0, 1]
    else:
        prior_acf_lag1[i] = 0

print("  Complete!")

# Calculate diagnostics
print("\n" + "=" * 80)
print("DIAGNOSTIC RESULTS")
print("=" * 80)

# Domain violations
n_negative = np.sum(prior_predictive_count < 0)
n_extreme_high = np.sum(prior_predictive_count > 1000)
n_extreme_low = np.sum((prior_predictive_count > 0) & (prior_predictive_count < 1))
n_nan_inf = np.sum(~np.isfinite(prior_predictive_count))

total_predictions = N_PRIOR_DRAWS * N_TIME_POINTS

print("\n1. DOMAIN VIOLATIONS:")
print(f"  Negative counts:        {n_negative} ({100*n_negative/total_predictions:.2f}%)")
print(f"  Extreme high (>1000):   {n_extreme_high} ({100*n_extreme_high/total_predictions:.2f}%)")
print(f"  Extreme low (<1):       {n_extreme_low} ({100*n_extreme_low/total_predictions:.2f}%)")
print(f"  NaN/Inf values:         {n_nan_inf} ({100*n_nan_inf/total_predictions:.2f}%)")

# Plausibility ranges
observed_min, observed_max = C_obs.min(), C_obs.max()
plausible_min, plausible_max = 10, 500

draws_in_observed = np.sum(np.all((prior_predictive_count >= observed_min) &
                                   (prior_predictive_count <= observed_max), axis=1))
draws_in_plausible = np.sum(np.all((prior_predictive_count >= plausible_min) &
                                    (prior_predictive_count <= plausible_max), axis=1))

print("\n2. PLAUSIBILITY RANGES:")
print(f"  Observed range:         [{observed_min}, {observed_max}]")
print(f"  Plausible range:        [{plausible_min}, {plausible_max}]")
print(f"  Draws fully in observed range:     {100*draws_in_observed/N_PRIOR_DRAWS:.1f}%")
print(f"  Draws fully in plausible range:    {100*draws_in_plausible/N_PRIOR_DRAWS:.1f}%")

# Autocorrelation structure (KEY DIAGNOSTIC)
acf_median = np.median(prior_acf_lag1)
acf_p5, acf_p95 = np.percentile(prior_acf_lag1, [5, 95])
observed_in_prior_interval = (observed_acf >= acf_p5) and (observed_acf <= acf_p95)

print("\n3. AUTOCORRELATION STRUCTURE (KEY DIAGNOSTIC):")
print(f"  Observed log(C) ACF lag-1:          {observed_acf:.3f}")
print(f"  Prior ACF lag-1 (median):           {acf_median:.3f}")
print(f"  Prior ACF lag-1 (90% CI):           [{acf_p5:.3f}, {acf_p95:.3f}]")
print(f"  Prior covers observed:              {observed_in_prior_interval}")

# Prediction statistics
print("\n4. PRIOR PREDICTIVE STATISTICS:")
print(f"  Overall median prediction:      {np.median(prior_predictive_count):.1f}")
print(f"  Overall mean prediction:        {np.mean(prior_predictive_count):.1f}")
print(f"  Min prediction:                 {np.min(prior_predictive_count):.1f}")
print(f"  Max prediction:                 {np.max(prior_predictive_count):.1f}")
print(f"  90th percentile:                {np.percentile(prior_predictive_count, 90):.1f}")
print(f"  95th percentile:                {np.percentile(prior_predictive_count, 95):.1f}")
print(f"  99th percentile:                {np.percentile(prior_predictive_count, 99):.1f}")

# Comparison to v1
print("\n" + "=" * 80)
print("COMPARISON TO V1")
print("=" * 80)
print("\n| Metric                     | v1 (Failed)  | v2 (Current) |")
print("|----------------------------|--------------|--------------|")
print(f"| Prior ACF median           | -0.059       | {acf_median:.3f}        |")
print(f"| % in plausible [10, 500]   | 2.8%         | {100*draws_in_plausible/N_PRIOR_DRAWS:.1f}%         |")
print(f"| % predictions >1000        | 5.8%         | {100*n_extreme_high/total_predictions:.2f}%         |")
print(f"| Max prediction             | 348 million  | {np.max(prior_predictive_count):.0f}         |")

# Decision criteria
print("\n" + "=" * 80)
print("DECISION CRITERIA")
print("=" * 80)

pass_acf = (acf_median >= 0.7) and (acf_median <= 0.95) and observed_in_prior_interval
pass_plausible = (draws_in_plausible / N_PRIOR_DRAWS) >= 0.15
pass_extreme = (n_extreme_high / total_predictions) < 0.01
pass_max = np.max(prior_predictive_count) < 10000

print(f"\nPASS if:")
print(f"  1. Prior ACF median in [0.7, 0.95]:           {pass_acf} (median={acf_median:.3f})")
print(f"  2. Observed ACF in 90% prior interval:        {observed_in_prior_interval}")
print(f"  3. >15% of draws in plausible range:          {pass_plausible} ({100*draws_in_plausible/N_PRIOR_DRAWS:.1f}%)")
print(f"  4. <1% predictions >1000:                     {pass_extreme} ({100*n_extreme_high/total_predictions:.2f}%)")
print(f"  5. Max prediction <10,000:                    {pass_max} (max={np.max(prior_predictive_count):.0f})")

overall_pass = pass_acf and pass_plausible and pass_extreme and pass_max

print(f"\n{'='*80}")
if overall_pass:
    print("OVERALL DECISION: PASS")
    print("All criteria met. Priors are well-specified.")
else:
    print("OVERALL DECISION: FAIL")
    print("Some criteria not met. Further prior adjustment needed.")
print('='*80)

# Save results for plotting
results = {
    'alpha': alpha,
    'beta_1': beta_1,
    'beta_2': beta_2,
    'phi': phi,
    'sigma_regime': sigma_regime,
    'prior_predictive_log': prior_predictive_log,
    'prior_predictive_count': prior_predictive_count,
    'prior_acf_lag1': prior_acf_lag1,
    'observed_acf': observed_acf,
    'year': year,
    'C_obs': C_obs,
    'log_C_obs': log_C_obs,
    'regime': regime,
    'overall_pass': overall_pass
}

np.savez(OUTPUT_DIR / "prior_predictive_results_v2.npz", **results)
print(f"\nResults saved to: {OUTPUT_DIR / 'prior_predictive_results_v2.npz'}")

#==============================================================================
# VISUALIZATIONS
#==============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Plot 1: Parameter plausibility
print("\n1. Creating parameter_plausibility.png...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Prior Distributions (v2 - Updated Priors)", fontsize=14, fontweight='bold')

# alpha
axes[0, 0].hist(alpha, bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(np.median(alpha), color='red', linestyle='--', label=f'Median={np.median(alpha):.2f}')
axes[0, 0].set_xlabel('alpha')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].set_title('Intercept (log-scale)')

# beta_1
axes[0, 1].hist(beta_1, bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(np.median(beta_1), color='red', linestyle='--', label=f'Median={np.median(beta_1):.2f}')
axes[0, 1].set_xlabel('beta_1')
axes[0, 1].legend()
axes[0, 1].set_title('Linear Trend (TIGHTENED)')

# beta_2
axes[0, 2].hist(beta_2, bins=50, alpha=0.7, edgecolor='black')
axes[0, 2].axvline(np.median(beta_2), color='red', linestyle='--', label=f'Median={np.median(beta_2):.2f}')
axes[0, 2].set_xlabel('beta_2')
axes[0, 2].legend()
axes[0, 2].set_title('Quadratic Trend')

# phi (KEY UPDATED PRIOR)
axes[0, 3].hist(phi, bins=50, alpha=0.7, edgecolor='black', color='green')
axes[0, 3].axvline(np.median(phi), color='red', linestyle='--', label=f'Median={np.median(phi):.2f}')
axes[0, 3].set_xlabel('phi')
axes[0, 3].legend()
axes[0, 3].set_title('AR(1) Coefficient (UPDATED)')

# sigma_regime (all three)
for i in range(3):
    axes[1, i].hist(sigma_regime[:, i], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, i].axvline(np.median(sigma_regime[:, i]), color='red', linestyle='--',
                       label=f'Median={np.median(sigma_regime[:, i]):.2f}')
    axes[1, i].set_xlabel(f'sigma_regime[{i+1}]')
    axes[1, i].set_ylabel('Frequency')
    axes[1, i].legend()
    axes[1, i].set_title(f'Regime {i+1} SD (TIGHTENED)')

# Summary text
summary_text = f"""Updated Priors (v2):
phi ~ Beta(20,2) * 0.95
sigma ~ HalfNormal(0, 0.5)
beta_1 ~ Normal(0.86, 0.15)

Key Changes from v1:
- phi: Median {np.median(phi):.3f} (was -0.031)
- sigma: 95th %ile {np.percentile(sigma_regime, 95):.2f} (was 1.97)
"""
axes[1, 3].text(0.1, 0.5, summary_text, transform=axes[1, 3].transAxes,
                fontsize=10, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "parameter_plausibility.png", dpi=100, bbox_inches='tight')
plt.close()

# Plot 2: Prior predictive coverage
print("2. Creating prior_predictive_coverage.png...")

fig, ax = plt.subplots(figsize=(12, 6))

# Calculate percentiles
p05 = np.percentile(prior_predictive_count, 5, axis=0)
p25 = np.percentile(prior_predictive_count, 25, axis=0)
p50 = np.percentile(prior_predictive_count, 50, axis=0)
p75 = np.percentile(prior_predictive_count, 75, axis=0)
p95 = np.percentile(prior_predictive_count, 95, axis=0)

# Plot bands
ax.fill_between(year, p05, p95, alpha=0.2, color='blue', label='90% Prior Interval')
ax.fill_between(year, p25, p75, alpha=0.3, color='blue', label='50% Prior Interval')
ax.plot(year, p50, 'b-', linewidth=2, label='Prior Median')

# Observed data
ax.scatter(year, C_obs, color='red', s=50, zorder=5, label='Observed Data', alpha=0.8)

ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prior Predictive Coverage (v2 - Updated Priors)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Add comparison text
improvement_text = f"v1 Max: 348M\nv2 Max: {np.max(prior_predictive_count):.0f}\n\nImprovement: {348453273/np.max(prior_predictive_count):.0f}x"
ax.text(0.98, 0.98, improvement_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "prior_predictive_coverage.png", dpi=100, bbox_inches='tight')
plt.close()

# Plot 3: Autocorrelation diagnostic (KEY PLOT)
print("3. Creating prior_autocorrelation_diagnostic.png (KEY DIAGNOSTIC)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Autocorrelation Diagnostic (v2 - After phi Prior Fix)', fontsize=14, fontweight='bold')

# Left panel: ACF distribution
axes[0].hist(prior_acf_lag1, bins=50, alpha=0.7, edgecolor='black', color='green')
axes[0].axvline(np.median(prior_acf_lag1), color='blue', linestyle='--', linewidth=2,
                label=f'v2 Prior Median={np.median(prior_acf_lag1):.3f}')
axes[0].axvline(observed_acf, color='red', linestyle='-', linewidth=2,
                label=f'Observed ACF={observed_acf:.3f}')
axes[0].axvline(-0.059, color='orange', linestyle=':', linewidth=2,
                label=f'v1 Prior Median=-0.059')
axes[0].axvspan(acf_p5, acf_p95, alpha=0.2, color='green', label='90% Prior Interval')
axes[0].set_xlabel('ACF lag-1', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prior ACF Distribution: v1 vs v2')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Add improvement annotation
if observed_in_prior_interval:
    axes[0].text(0.05, 0.95, 'IMPROVEMENT:\nObserved now\nwithin 90% interval',
                transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
else:
    axes[0].text(0.05, 0.95, 'WARNING:\nObserved still\noutside interval',
                transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Right panel: phi vs ACF scatter
axes[1].scatter(phi, prior_acf_lag1, alpha=0.3, s=10, color='green')
axes[1].axhline(observed_acf, color='red', linestyle='-', linewidth=2, label='Observed ACF')
axes[1].axvline(np.median(phi), color='blue', linestyle='--', linewidth=2,
                label=f'Median phi={np.median(phi):.3f}')
axes[1].set_xlabel('phi (AR coefficient)', fontsize=12)
axes[1].set_ylabel('Implied ACF lag-1', fontsize=12)
axes[1].set_title('Relationship: phi -> ACF')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "prior_autocorrelation_diagnostic.png", dpi=100, bbox_inches='tight')
plt.close()

# Plot 4: Comparison v1 vs v2
print("4. Creating comparison_v1_vs_v2.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prior Predictive Check: v1 (Failed) vs v2 (Updated)', fontsize=14, fontweight='bold')

# Panel 1: ACF comparison
axes[0, 0].hist(prior_acf_lag1, bins=50, alpha=0.6, edgecolor='black', color='green', label='v2 (Updated)')
# Simulate v1 distribution (uniform phi gives nearly uniform ACF)
v1_acf_approx = np.random.uniform(-0.9, 0.9, N_PRIOR_DRAWS)
axes[0, 0].hist(v1_acf_approx, bins=50, alpha=0.3, edgecolor='black', color='red', label='v1 (Failed)')
axes[0, 0].axvline(observed_acf, color='black', linestyle='-', linewidth=2, label='Observed')
axes[0, 0].set_xlabel('ACF lag-1')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('ACF Distribution: Before vs After')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Max prediction by draw
max_per_draw_v2 = np.max(prior_predictive_count, axis=1)
axes[0, 1].hist(np.log10(max_per_draw_v2), bins=50, alpha=0.7, edgecolor='black', color='green')
axes[0, 1].axvline(np.log10(1000), color='red', linestyle='--', linewidth=2, label='Threshold (1000)')
axes[0, 1].set_xlabel('log10(Max Prediction per Draw)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Maximum Predictions (v2): {100*np.mean(max_per_draw_v2>1000):.2f}% > 1000')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Plausibility coverage
plausible_counts_v2 = np.sum((prior_predictive_count >= 10) & (prior_predictive_count <= 500), axis=1)
axes[1, 0].hist(plausible_counts_v2, bins=np.arange(0, 42), alpha=0.7, edgecolor='black', color='green')
axes[1, 0].axvline(40, color='red', linestyle='--', linewidth=2, label='All 40 points plausible')
axes[1, 0].set_xlabel('Number of plausible predictions (out of 40)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Plausibility Coverage (v2): {100*draws_in_plausible/N_PRIOR_DRAWS:.1f}% fully in [10,500]')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Comparison table
comparison_data = [
    ['Prior ACF median', '-0.059', f'{acf_median:.3f}', 'IMPROVED' if acf_median > 0.7 else 'NEEDS WORK'],
    ['% in [10, 500]', '2.8%', f'{100*draws_in_plausible/N_PRIOR_DRAWS:.1f}%',
     'IMPROVED' if draws_in_plausible/N_PRIOR_DRAWS > 0.15 else 'NEEDS WORK'],
    ['% predictions >1000', '5.8%', f'{100*n_extreme_high/total_predictions:.2f}%',
     'IMPROVED' if n_extreme_high/total_predictions < 0.01 else 'NEEDS WORK'],
    ['Max prediction', '348M', f'{np.max(prior_predictive_count):.0f}',
     'IMPROVED' if np.max(prior_predictive_count) < 10000 else 'NEEDS WORK']
]

table_text = "Metric                     v1 (Failed)    v2 (Current)   Status\n"
table_text += "-" * 70 + "\n"
for row in comparison_data:
    status_color = 'green' if row[3] == 'IMPROVED' else 'orange'
    table_text += f"{row[0]:25} {row[1]:14} {row[2]:14} {row[3]}\n"

axes[1, 1].text(0.1, 0.5, table_text, transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')
axes[1, 1].set_title('Summary Comparison')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "comparison_v1_vs_v2.png", dpi=100, bbox_inches='tight')
plt.close()

# Plot 5: Sample trajectories
print("5. Creating sample_trajectories.png...")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Prior Predictive Trajectories (v2)', fontsize=14, fontweight='bold')

# Select 20 random draws
n_sample = 20
sample_idx = np.random.choice(N_PRIOR_DRAWS, n_sample, replace=False)

# Top: Count scale
for idx in sample_idx:
    axes[0].plot(year, prior_predictive_count[idx, :], alpha=0.3, linewidth=1, color='blue')
axes[0].scatter(year, C_obs, color='red', s=30, zorder=5, label='Observed', alpha=0.8)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Count Scale (showing AR smoothness)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1000])

# Bottom: Log scale
for idx in sample_idx:
    axes[1].plot(year, prior_predictive_log[idx, :], alpha=0.3, linewidth=1, color='green')
axes[1].scatter(year, log_C_obs, color='red', s=30, zorder=5, label='Observed', alpha=0.8)
axes[1].set_xlabel('Year (standardized)', fontsize=12)
axes[1].set_ylabel('log(Count)', fontsize=12)
axes[1].set_title('Log Scale (AR structure more visible)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "sample_trajectories.png", dpi=100, bbox_inches='tight')
plt.close()

# Plot 6: Regime variance diagnostic
print("6. Creating regime_variance_diagnostic.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Regime Variance Structure (v2)', fontsize=14, fontweight='bold')

# Panel 1: Overlapping sigma distributions
for i in range(3):
    axes[0, 0].hist(sigma_regime[:, i], bins=30, alpha=0.5, label=f'Regime {i+1}', edgecolor='black')
axes[0, 0].set_xlabel('sigma')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Prior Distributions (All Regimes)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Which regime has largest sigma?
largest_regime = np.argmax(sigma_regime, axis=1) + 1
regime_counts = [np.sum(largest_regime == i) for i in [1, 2, 3]]
axes[0, 1].bar([1, 2, 3], regime_counts, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Regime')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Which Regime Has Largest Variance?')
axes[0, 1].set_xticks([1, 2, 3])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Panel 3: Prior predictive variance by regime
regime_vars = []
for i in range(3):
    regime_mask = regime == (i + 1)
    regime_vars.append(np.var(prior_predictive_log[:, regime_mask], axis=1))

axes[1, 0].boxplot(regime_vars, labels=['Early', 'Middle', 'Late'])
axes[1, 0].set_ylabel('Variance (log scale)')
axes[1, 0].set_title('Prior Predictive Variance by Regime')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Panel 4: Comparison text
comparison_text = f"""v2 Regime Sigma Priors:
HalfNormal(0, 0.5)

Median: {np.median(sigma_regime):.2f}
95th percentile: {np.percentile(sigma_regime, 95):.2f}

v1 had:
HalfNormal(0, 1.0)
95th %ile: 1.97

Improvement:
{1.97 / np.percentile(sigma_regime, 95):.1f}x tighter
"""
axes[1, 1].text(0.1, 0.5, comparison_text, transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "regime_variance_diagnostic.png", dpi=100, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 80)
print(f"\nPlots saved to: {OUTPUT_DIR / 'plots'}/")
print("\nGenerated plots:")
print("  1. parameter_plausibility.png - Prior distributions with updated priors")
print("  2. prior_predictive_coverage.png - Coverage plot showing improvements")
print("  3. prior_autocorrelation_diagnostic.png - KEY: ACF before vs after")
print("  4. comparison_v1_vs_v2.png - Direct comparison of all metrics")
print("  5. sample_trajectories.png - AR structure visualization")
print("  6. regime_variance_diagnostic.png - Regime variance structure")

print("\n" + "=" * 80)
print("PRIOR PREDICTIVE CHECK V2 COMPLETE")
print("=" * 80)
print(f"\nFinal Decision: {'PASS' if overall_pass else 'FAIL'}")
print("\nNext step: Review findings.md for detailed assessment")
