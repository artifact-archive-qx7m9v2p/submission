"""
Prior Predictive Check for Bayesian Hierarchical Meta-Analysis
================================================================

Model:
- y_i ~ Normal(theta_i, sigma_i) for i=1,...,8
- theta_i ~ Normal(mu, tau)
- mu ~ Normal(0, 50)
- tau ~ Half-Cauchy(0, 5)

Known:
- sigma_i = [15, 10, 16, 11, 9, 11, 10, 18]
- y_obs = [28, 8, -3, 7, -1, 1, 18, 12]
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_SAMPLES = 1000
N_STUDIES = 8

# Known data
sigma_i = np.array([15, 10, 16, 11, 9, 11, 10, 18])
y_obs = np.array([28, 8, -3, 7, -1, 1, 18, 12])

# Prior specifications
MU_PRIOR_MEAN = 0
MU_PRIOR_SD = 50
TAU_PRIOR_SCALE = 5

print("=" * 70)
print("PRIOR PREDICTIVE CHECK: Bayesian Hierarchical Meta-Analysis")
print("=" * 70)
print(f"\nGenerating {N_PRIOR_SAMPLES} prior predictive datasets...")
print(f"Number of studies: {N_STUDIES}")
print(f"\nObserved data:")
print(f"  y_obs = {y_obs}")
print(f"  sigma_i = {sigma_i}")
print(f"\nPrior specifications:")
print(f"  mu ~ Normal({MU_PRIOR_MEAN}, {MU_PRIOR_SD})")
print(f"  tau ~ Half-Cauchy(0, {TAU_PRIOR_SCALE})")

# ============================================================================
# STEP 1: Sample from priors
# ============================================================================

# Sample mu from Normal(0, 50)
mu_samples = np.random.normal(MU_PRIOR_MEAN, MU_PRIOR_SD, N_PRIOR_SAMPLES)

# Sample tau from Half-Cauchy(0, 5)
# Half-Cauchy is the absolute value of a Cauchy distribution
tau_samples = np.abs(stats.cauchy.rvs(loc=0, scale=TAU_PRIOR_SCALE, size=N_PRIOR_SAMPLES))

print(f"\nPrior samples generated:")
print(f"  mu: mean={mu_samples.mean():.2f}, sd={mu_samples.std():.2f}, "
      f"range=[{mu_samples.min():.2f}, {mu_samples.max():.2f}]")
print(f"  tau: mean={tau_samples.mean():.2f}, sd={tau_samples.std():.2f}, "
      f"range=[{tau_samples.min():.2f}, {tau_samples.max():.2f}]")

# Check for extreme tau values (computational red flag)
tau_extreme = np.sum(tau_samples > 100)
if tau_extreme > 0:
    print(f"\n  WARNING: {tau_extreme} ({100*tau_extreme/N_PRIOR_SAMPLES:.1f}%) "
          f"tau samples > 100 (potential computational issues)")

# ============================================================================
# STEP 2: Generate study-specific effects theta_i
# ============================================================================

# For each prior sample, generate theta_i for all studies
theta_samples = np.zeros((N_PRIOR_SAMPLES, N_STUDIES))

for i in range(N_PRIOR_SAMPLES):
    theta_samples[i, :] = np.random.normal(mu_samples[i], tau_samples[i], N_STUDIES)

print(f"\nStudy effects (theta_i) generated:")
print(f"  Overall range: [{theta_samples.min():.2f}, {theta_samples.max():.2f}]")
print(f"  Mean across all: {theta_samples.mean():.2f}")

# Check for extreme theta values
theta_extreme = np.sum(np.abs(theta_samples) > 200)
if theta_extreme > 0:
    print(f"  WARNING: {theta_extreme} "
          f"({100*theta_extreme/(N_PRIOR_SAMPLES*N_STUDIES):.1f}%) "
          f"theta values beyond [-200, 200]")

# ============================================================================
# STEP 3: Generate prior predictive observations y_i
# ============================================================================

# For each prior sample and each study, generate y_i ~ Normal(theta_i, sigma_i)
y_prior_pred = np.zeros((N_PRIOR_SAMPLES, N_STUDIES))

for i in range(N_PRIOR_SAMPLES):
    for j in range(N_STUDIES):
        y_prior_pred[i, j] = np.random.normal(theta_samples[i, j], sigma_i[j])

print(f"\nPrior predictive observations (y_i) generated:")
print(f"  Overall range: [{y_prior_pred.min():.2f}, {y_prior_pred.max():.2f}]")
print(f"  Mean: {y_prior_pred.mean():.2f}")
print(f"  SD: {y_prior_pred.std():.2f}")

# ============================================================================
# STEP 4: Assess Prior Plausibility
# ============================================================================

print("\n" + "=" * 70)
print("PLAUSIBILITY ASSESSMENT")
print("=" * 70)

# Check 1: Coverage of observed data
print("\n1. COVERAGE CHECK: Are observed values within prior predictive range?")
for j in range(N_STUDIES):
    y_pred_study = y_prior_pred[:, j]
    y_obs_j = y_obs[j]

    # Calculate percentile of observed value in prior predictive
    percentile = stats.percentileofscore(y_pred_study, y_obs_j)

    # Calculate 95% prior predictive interval
    lower = np.percentile(y_pred_study, 2.5)
    upper = np.percentile(y_pred_study, 97.5)

    in_interval = lower <= y_obs_j <= upper
    status = "OK" if in_interval else "OUTSIDE"

    print(f"   Study {j+1}: y_obs={y_obs_j:6.1f}, "
          f"95% PPI=[{lower:7.1f}, {upper:7.1f}], "
          f"percentile={percentile:5.1f}%, {status}")

# Check 2: Effect size plausibility
print("\n2. EFFECT SIZE PLAUSIBILITY:")
theta_in_reasonable = np.sum((theta_samples >= -50) & (theta_samples <= 50))
theta_pct = 100 * theta_in_reasonable / (N_PRIOR_SAMPLES * N_STUDIES)
print(f"   {theta_pct:.1f}% of theta_i samples in [-50, 50] range")

theta_in_tight = np.sum((theta_samples >= -20) & (theta_samples <= 50))
theta_tight_pct = 100 * theta_in_tight / (N_PRIOR_SAMPLES * N_STUDIES)
print(f"   {theta_tight_pct:.1f}% of theta_i samples in [-20, 50] range")

# Check 3: Heterogeneity coverage
print("\n3. HETEROGENEITY COVERAGE:")
tau_small = np.sum(tau_samples < 1)
tau_moderate = np.sum((tau_samples >= 1) & (tau_samples < 10))
tau_large = np.sum(tau_samples >= 10)

print(f"   tau < 1 (homogeneous): {100*tau_small/N_PRIOR_SAMPLES:.1f}%")
print(f"   1 <= tau < 10 (moderate): {100*tau_moderate/N_PRIOR_SAMPLES:.1f}%")
print(f"   tau >= 10 (high heterogeneity): {100*tau_large/N_PRIOR_SAMPLES:.1f}%")

# Check 4: Prior predictive p-values
print("\n4. PRIOR PREDICTIVE P-VALUES:")
print("   (Fraction of prior pred samples more extreme than observed)")
for j in range(N_STUDIES):
    y_pred_study = y_prior_pred[:, j]
    y_obs_j = y_obs[j]

    # Two-tailed p-value
    if y_obs_j > np.median(y_pred_study):
        pval = 2 * np.mean(y_pred_study >= y_obs_j)
    else:
        pval = 2 * np.mean(y_pred_study <= y_obs_j)

    pval = min(pval, 1.0)  # Cap at 1.0

    flag = "(!)" if pval < 0.01 or pval > 0.99 else ""
    print(f"   Study {j+1}: p = {pval:.3f} {flag}")

# Check 5: Global statistics
print("\n5. GLOBAL STATISTICS:")
print("   Observed data:")
print(f"     Range: [{y_obs.min()}, {y_obs.max()}]")
print(f"     Mean: {y_obs.mean():.2f}")
print(f"     SD: {y_obs.std():.2f}")

print("\n   Prior predictive (per dataset):")
y_range_prior = np.ptp(y_prior_pred, axis=1)  # range per sample
y_mean_prior = np.mean(y_prior_pred, axis=1)
y_sd_prior = np.std(y_prior_pred, axis=1)

print(f"     Range: median={np.median(y_range_prior):.2f}, "
      f"95% CI=[{np.percentile(y_range_prior, 2.5):.2f}, "
      f"{np.percentile(y_range_prior, 97.5):.2f}]")
print(f"     Mean: median={np.median(y_mean_prior):.2f}, "
      f"95% CI=[{np.percentile(y_mean_prior, 2.5):.2f}, "
      f"{np.percentile(y_mean_prior, 97.5):.2f}]")
print(f"     SD: median={np.median(y_sd_prior):.2f}, "
      f"95% CI=[{np.percentile(y_sd_prior, 2.5):.2f}, "
      f"{np.percentile(y_sd_prior, 97.5):.2f}]")

# ============================================================================
# STEP 6: Save results for visualization
# ============================================================================

results = {
    'mu_samples': mu_samples,
    'tau_samples': tau_samples,
    'theta_samples': theta_samples,
    'y_prior_pred': y_prior_pred,
    'y_obs': y_obs,
    'sigma_i': sigma_i,
    'N_PRIOR_SAMPLES': N_PRIOR_SAMPLES,
    'N_STUDIES': N_STUDIES
}

np.savez('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_samples.npz',
         **results)

print("\n" + "=" * 70)
print("Results saved to prior_predictive_samples.npz")
print("=" * 70)
