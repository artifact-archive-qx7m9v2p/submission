"""
Posterior Predictive Checks for Complete Pooling Model (Experiment 2)
======================================================================

Key diagnostic: Test for under-dispersion
(Complete pooling may predict less variance than observed)
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create output directories
os.makedirs('/workspace/experiments/experiment_2/posterior_predictive_check/diagnostics', exist_ok=True)
os.makedirs('/workspace/experiments/experiment_2/posterior_predictive_check/plots', exist_ok=True)

print("="*70)
print("POSTERIOR PREDICTIVE CHECKS - COMPLETE POOLING MODEL")
print("="*70)

# Load inference data
idata = az.from_netcdf('/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf')

# Extract data
y_obs = idata.observed_data['y'].values
sigma = idata.observed_data['sigma'].values
y_rep = idata.posterior_predictive['y_rep'].values
N = len(y_obs)

print(f"\nData:")
print(f"  Studies: {N}")
print(f"  Observed y: {y_obs}")
print(f"  Known sigma: {sigma}")
print(f"\nPosterior predictive samples:")
print(f"  Shape: {y_rep.shape}")
print(f"  Total samples: {y_rep.shape[0] * y_rep.shape[1]}")

# Flatten posterior predictive samples
n_chains, n_samples, n_studies = y_rep.shape
y_rep_flat = y_rep.reshape(-1, n_studies)  # (4000, 8)

print("\n" + "="*70)
print("TEST 1: POINT-WISE POSTERIOR PREDICTIVE P-VALUES")
print("="*70)

# For each study, compute p-value: P(y_rep > y_obs)
print("\nPoint-wise checks (proportion of y_rep > y_obs):")
print("\nStudy | y_obs  | Mean(y_rep) | SD(y_rep) | p-value | Assessment")
print("-" * 70)

pointwise_pvals = []
for i in range(N):
    y_rep_i = y_rep_flat[:, i]
    y_obs_i = y_obs[i]

    mean_rep = np.mean(y_rep_i)
    sd_rep = np.std(y_rep_i)

    # Two-tailed p-value
    p_val = np.mean(y_rep_i > y_obs_i)
    p_val_two_tailed = 2 * min(p_val, 1 - p_val)

    pointwise_pvals.append(p_val_two_tailed)

    # Assessment
    if p_val_two_tailed < 0.05:
        assessment = "EXTREME"
    elif p_val_two_tailed < 0.1:
        assessment = "Unusual"
    else:
        assessment = "OK"

    print(f"{i+1:5d} | {y_obs_i:6.2f} | {mean_rep:11.2f} | {sd_rep:9.2f} | {p_val_two_tailed:7.3f} | {assessment}")

print("\n" + "="*70)
print("TEST 2: GLOBAL VARIANCE TEST (Under-dispersion)")
print("="*70)

# Key diagnostic for complete pooling: Is observed variance larger than predicted?
# Complete pooling may under-estimate between-study variation

# Test statistic: Variance of standardized effects
def compute_variance_stat(y, sigma):
    """Variance of standardized effects"""
    return np.var(y / sigma)

# Observed test statistic
T_obs = compute_variance_stat(y_obs, sigma)

# Posterior predictive test statistics
T_rep = np.array([compute_variance_stat(y_rep_flat[i, :], sigma)
                  for i in range(len(y_rep_flat))])

# Bayesian p-value
p_value_variance = np.mean(T_rep >= T_obs)

print("\nVariance of standardized effects:")
print(f"  Observed: {T_obs:.4f}")
print(f"  Mean(Replicated): {np.mean(T_rep):.4f}")
print(f"  SD(Replicated): {np.std(T_rep):.4f}")
print(f"  95% Predictive Interval: [{np.percentile(T_rep, 2.5):.4f}, {np.percentile(T_rep, 97.5):.4f}]")
print(f"\nBayesian p-value: {p_value_variance:.4f}")
print(f"  (Proportion of replications with variance >= observed)")

if p_value_variance < 0.05:
    print("\n  ** SIGNIFICANT UNDER-DISPERSION **")
    print("  Observed variance exceeds model predictions")
    print("  Complete pooling may be inadequate")
elif p_value_variance < 0.1:
    print("\n  Marginal evidence of under-dispersion")
    print("  Consider hierarchical model")
else:
    print("\n  No evidence of under-dispersion")
    print("  Complete pooling appears adequate")

print("\n" + "="*70)
print("TEST 3: EXTREME VALUE TEST")
print("="*70)

# Test if observed max/min are unusual
max_obs = np.max(y_obs / sigma)
min_obs = np.min(y_obs / sigma)

max_rep = np.array([np.max(y_rep_flat[i, :] / sigma) for i in range(len(y_rep_flat))])
min_rep = np.array([np.min(y_rep_flat[i, :] / sigma) for i in range(len(y_rep_flat))])

p_max = np.mean(max_rep >= max_obs)
p_min = np.mean(min_rep <= min_obs)

print(f"\nMaximum standardized effect:")
print(f"  Observed: {max_obs:.4f}")
print(f"  Mean(Replicated): {np.mean(max_rep):.4f}")
print(f"  Bayesian p-value: {p_max:.4f}")

print(f"\nMinimum standardized effect:")
print(f"  Observed: {min_obs:.4f}")
print(f"  Mean(Replicated): {np.mean(min_rep):.4f}")
print(f"  Bayesian p-value: {p_min:.4f}")

if p_max < 0.05 or p_min < 0.05:
    print("\n  ** EXTREME VALUES DETECTED **")
    print("  Observed extremes are unusual under complete pooling")
else:
    print("\n  Extreme values within expected range")

print("\n" + "="*70)
print("TEST 4: STUDY-LEVEL DEVIATIONS")
print("="*70)

# For each study, how often does replicate exceed observed deviation from mean?
mu_samples = idata.posterior['mu'].values.flatten()
mu_mean = np.mean(mu_samples)

print("\nStudy-level deviations from common effect:")
print("\nStudy | Obs Dev | Mean(Rep Dev) | SD(Rep Dev) | Obs/Pred Ratio")
print("-" * 70)

for i in range(N):
    obs_dev = abs(y_obs[i] - mu_mean)

    rep_devs = np.abs(y_rep_flat[:, i] - mu_samples[:, np.newaxis].flatten()[:len(y_rep_flat)])
    mean_rep_dev = np.mean(rep_devs)
    sd_rep_dev = np.std(rep_devs)

    ratio = obs_dev / mean_rep_dev if mean_rep_dev > 0 else 0

    print(f"{i+1:5d} | {obs_dev:7.2f} | {mean_rep_dev:13.2f} | {sd_rep_dev:11.2f} | {ratio:14.2f}")

print("\n" + "="*70)
print("VISUALIZATIONS")
print("="*70)

# Create comprehensive PPC plots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Plot 1: PPC overlay for each study
for i in range(N):
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])

    # Histogram of replications
    ax.hist(y_rep_flat[:, i], bins=30, alpha=0.6, color='steelblue',
            density=True, label='Replicated')

    # Observed value
    ax.axvline(y_obs[i], color='red', linewidth=2, label='Observed')

    # Expected distribution
    x_range = np.linspace(y_rep_flat[:, i].min(), y_rep_flat[:, i].max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range, mu_mean, sigma[i]),
            'k--', linewidth=1.5, alpha=0.7, label='Expected')

    ax.set_title(f'Study {i+1} (σ={sigma[i]})', fontweight='bold', fontsize=10)
    ax.set_xlabel('y', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Posterior Predictive Checks: Study-Level Comparisons',
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('/workspace/experiments/experiment_2/posterior_predictive_check/plots/ppc_studywise.png',
            dpi=150, bbox_inches='tight')
print("\nSaved: plots/ppc_studywise.png")
plt.close()

# Plot 2: Variance test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Variance comparison
axes[0].hist(T_rep, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(T_obs, color='red', linewidth=3, label=f'Observed ({T_obs:.3f})')
axes[0].axvline(np.mean(T_rep), color='blue', linestyle='--', linewidth=2,
                label=f'Mean Replicated ({np.mean(T_rep):.3f})')
axes[0].set_xlabel('Variance of Standardized Effects', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Under-Dispersion Test\n(Bayesian p-value = {:.3f})'.format(p_value_variance),
                 fontweight='bold', fontsize=13)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Add shaded region for extreme values
if p_value_variance < 0.05:
    threshold = np.percentile(T_rep, 95)
    axes[0].axvspan(threshold, axes[0].get_xlim()[1], alpha=0.2, color='red',
                   label='Extreme region (p<0.05)')

# Q-Q plot style: obs vs replicated for each study
axes[1].scatter(np.mean(y_rep_flat, axis=0), y_obs, s=100, alpha=0.7,
               c=np.arange(N), cmap='viridis', edgecolor='black', linewidth=1.5)
for i in range(N):
    axes[1].annotate(f'{i+1}', (np.mean(y_rep_flat[:, i]), y_obs[i]),
                    fontsize=9, ha='center', va='center', fontweight='bold')

# Add diagonal reference line
lim_min = min(axes[1].get_xlim()[0], axes[1].get_ylim()[0])
lim_max = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
axes[1].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=2, alpha=0.5)
axes[1].set_xlabel('Mean Replicated y', fontsize=12)
axes[1].set_ylabel('Observed y', fontsize=12)
axes[1].set_title('Observed vs Predicted', fontweight='bold', fontsize=13)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_predictive_check/plots/ppc_variance_test.png',
            dpi=150, bbox_inches='tight')
print("Saved: plots/ppc_variance_test.png")
plt.close()

# Plot 3: Distribution of test statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Max standardized effect
axes[0, 0].hist(max_rep, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(max_obs, color='red', linewidth=3, label=f'Observed ({max_obs:.2f})')
axes[0, 0].set_xlabel('Max Standardized Effect', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title(f'Maximum Test (p = {p_max:.3f})', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Min standardized effect
axes[0, 1].hist(min_rep, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].axvline(min_obs, color='red', linewidth=3, label=f'Observed ({min_obs:.2f})')
axes[0, 1].set_xlabel('Min Standardized Effect', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title(f'Minimum Test (p = {p_min:.3f})', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Study-level p-values
axes[1, 0].bar(range(1, N+1), pointwise_pvals, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].axhline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05')
axes[1, 0].axhline(0.1, color='orange', linestyle='--', linewidth=2, label='p = 0.10')
axes[1, 0].set_xlabel('Study', fontsize=11)
axes[1, 0].set_ylabel('Two-tailed p-value', fontsize=11)
axes[1, 0].set_title('Point-wise Predictive p-values', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Standardized residuals with predictive intervals
residuals = (y_obs - np.mean(y_rep_flat, axis=0)) / sigma
axes[1, 1].bar(range(1, N+1), residuals, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].axhline(0, color='black', linewidth=1)
axes[1, 1].axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='±2 SD')
axes[1, 1].axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1, 1].set_xlabel('Study', fontsize=11)
axes[1, 1].set_ylabel('Standardized Residual', fontsize=11)
axes[1, 1].set_title('Standardized Residuals', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Posterior Predictive Check Summary', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_predictive_check/plots/ppc_summary.png',
            dpi=150, bbox_inches='tight')
print("Saved: plots/ppc_summary.png")
plt.close()

print("\n" + "="*70)
print("PPC SUMMARY")
print("="*70)

# Overall assessment
print("\n**OVERALL ASSESSMENT:**")

issues = []
if p_value_variance < 0.05:
    issues.append("Significant under-dispersion detected")
if p_max < 0.05:
    issues.append("Extreme maximum value")
if p_min < 0.05:
    issues.append("Extreme minimum value")
if any(p < 0.05 for p in pointwise_pvals):
    n_extreme = sum(1 for p in pointwise_pvals if p < 0.05)
    issues.append(f"{n_extreme} studies with extreme p-values")

if len(issues) == 0:
    print("\n✓ All posterior predictive checks PASS")
    print("  Complete pooling appears adequate for this data")
    print("\nRecommendation: Consider accepting complete pooling model")
    print("                 (pending LOO comparison)")
else:
    print("\n✗ Posterior predictive checks reveal issues:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nRecommendation: Hierarchical model may be needed")
    print("                 (pending LOO comparison)")

print("\n" + "="*70)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("="*70)
print("\nFiles generated:")
print("  - plots/ppc_studywise.png - Study-level comparisons")
print("  - plots/ppc_variance_test.png - Under-dispersion test")
print("  - plots/ppc_summary.png - Summary of all tests")
print("\nNext: LOO comparison with Experiment 1")
