"""
Prior Predictive Check for Experiment 1: Hierarchical Normal Model

This script validates the model specification BEFORE fitting by:
1. Sampling from the prior distributions
2. Generating synthetic data from the hierarchical model
3. Comparing prior predictive distributions to observed data
4. Assessing scientific plausibility and computational viability
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Observed data from 8 studies
observed_y = np.array([20.02, 15.30, 26.08, 25.73, -4.88, 6.08, 3.17, 8.55])
known_sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = 8  # number of studies

# Key observed statistics for comparison
observed_pooled_effect = 11.27
observed_tau = 2.0
observed_I2 = 2.9  # percent

# Model specification
# Priors:
#   mu ~ Normal(0, 25)
#   tau ~ Half-Normal(0, 10)
# Hierarchical:
#   theta_i ~ Normal(mu, tau)
# Likelihood:
#   y_i ~ Normal(theta_i, sigma_i)

# Number of prior predictive samples
n_samples = 4000

print("="*80)
print("PRIOR PREDICTIVE CHECK: Hierarchical Normal Model")
print("="*80)
print(f"\nGenerating {n_samples} prior predictive samples...")
print(f"Number of studies: J = {J}")
print(f"\nObserved data:")
print(f"  y_obs: {observed_y}")
print(f"  sigma: {known_sigma}")
print(f"  Pooled effect: {observed_pooled_effect:.2f}")
print(f"  Between-study SD (tau): {observed_tau:.2f}")
print(f"  Heterogeneity (I²): {observed_I2:.1f}%")
print()

# Step 1: Sample from priors
print("Step 1: Sampling from priors...")
mu_prior = np.random.normal(0, 25, n_samples)
tau_prior = np.abs(np.random.normal(0, 10, n_samples))  # Half-normal

print(f"  mu prior: mean={mu_prior.mean():.2f}, std={mu_prior.std():.2f}")
print(f"  tau prior: mean={tau_prior.mean():.2f}, std={tau_prior.std():.2f}")

# Step 2: Generate theta_i from hierarchical structure
print("\nStep 2: Generating study-specific effects theta_i...")
theta_samples = np.zeros((n_samples, J))
for i in range(n_samples):
    theta_samples[i, :] = np.random.normal(mu_prior[i], tau_prior[i], J)

# Step 3: Generate y_pred from likelihood
print("Step 3: Generating prior predictive data y_pred_i...")
y_pred_samples = np.zeros((n_samples, J))
for i in range(n_samples):
    for j in range(J):
        y_pred_samples[i, j] = np.random.normal(theta_samples[i, j], known_sigma[j])

# Step 4: Calculate summary statistics for prior predictive
print("\nStep 4: Calculating prior predictive summary statistics...")

# Pooled effect (simple mean across studies for each sample)
pooled_effect_samples = y_pred_samples.mean(axis=1)

# Calculate percentiles for observed data
print("\n" + "="*80)
print("PRIOR PREDICTIVE COVERAGE ASSESSMENT")
print("="*80)

print("\nStudy-specific prior predictive percentiles for observed y:")
study_percentiles = []
for j in range(J):
    percentile = stats.percentileofscore(y_pred_samples[:, j], observed_y[j])
    study_percentiles.append(percentile)
    status = "GOOD" if 25 <= percentile <= 75 else "MARGINAL" if 5 <= percentile <= 95 else "BAD"
    print(f"  Study {j+1}: y_obs={observed_y[j]:7.2f} is at {percentile:5.1f}% [{status}]")

# Pooled effect percentile
pooled_percentile = stats.percentileofscore(pooled_effect_samples, observed_pooled_effect)
pooled_status = "GOOD" if 25 <= pooled_percentile <= 75 else "MARGINAL" if 5 <= pooled_percentile <= 95 else "BAD"
print(f"\nPooled effect: obs={observed_pooled_effect:.2f} is at {pooled_percentile:.1f}% [{pooled_status}]")

# Tau percentile
tau_percentile = stats.percentileofscore(tau_prior, observed_tau)
tau_status = "GOOD" if 25 <= tau_percentile <= 75 else "MARGINAL" if 5 <= tau_percentile <= 95 else "BAD"
print(f"Between-study SD (tau): obs={observed_tau:.2f} is at {tau_percentile:.1f}% [{tau_status}]")

# Check for extreme values and computational issues
print("\n" + "="*80)
print("COMPUTATIONAL AND SCIENTIFIC PLAUSIBILITY CHECKS")
print("="*80)

print("\nPrior predictive value ranges:")
print(f"  y_pred range: [{y_pred_samples.min():.1f}, {y_pred_samples.max():.1f}]")
print(f"  theta range: [{theta_samples.min():.1f}, {theta_samples.max():.1f}]")
print(f"  mu range: [{mu_prior.min():.1f}, {mu_prior.max():.1f}]")
print(f"  tau range: [{tau_prior.min():.1f}, {tau_prior.max():.1f}]")

# Check for domain violations
print("\nDomain constraint checks:")
print(f"  Any infinite values? {np.any(np.isinf(y_pred_samples))}")
print(f"  Any NaN values? {np.any(np.isnan(y_pred_samples))}")
print(f"  Any tau = 0? {np.any(tau_prior == 0)}")

# Check for extreme values (more than 3 orders of magnitude from observed)
y_extreme = np.abs(y_pred_samples) > 1000
print(f"  Proportion of |y_pred| > 1000: {y_extreme.mean():.4f}")

# Save samples for plotting
print("\nSaving samples to disk...")
np.savez('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.npz',
         mu_prior=mu_prior,
         tau_prior=tau_prior,
         theta_samples=theta_samples,
         y_pred_samples=y_pred_samples,
         pooled_effect_samples=pooled_effect_samples,
         observed_y=observed_y,
         known_sigma=known_sigma,
         observed_pooled_effect=observed_pooled_effect,
         observed_tau=observed_tau)

# Summary statistics for report
summary_stats = {
    'study_percentiles': study_percentiles,
    'pooled_percentile': pooled_percentile,
    'tau_percentile': tau_percentile,
    'mu_prior_mean': mu_prior.mean(),
    'mu_prior_std': mu_prior.std(),
    'tau_prior_mean': tau_prior.mean(),
    'tau_prior_std': tau_prior.std(),
    'y_pred_min': y_pred_samples.min(),
    'y_pred_max': y_pred_samples.max(),
}

print("\n" + "="*80)
print("PASS/FAIL DECISION CRITERIA")
print("="*80)

# Decision logic
n_bad_studies = sum(1 for p in study_percentiles if p < 5 or p > 95)
n_marginal_studies = sum(1 for p in study_percentiles if (5 <= p < 25) or (75 < p <= 95))
n_good_studies = sum(1 for p in study_percentiles if 25 <= p <= 75)

print(f"\nStudy-level coverage:")
print(f"  Studies in middle 50% (25-75%): {n_good_studies}/{J}")
print(f"  Studies in marginal range (5-25% or 75-95%): {n_marginal_studies}/{J}")
print(f"  Studies in extreme tails (<5% or >95%): {n_bad_studies}/{J}")

# Overall assessment
fail_conditions = []
if n_bad_studies >= 2:
    fail_conditions.append(f"{n_bad_studies} studies in extreme tails")
if pooled_percentile < 5 or pooled_percentile > 95:
    fail_conditions.append(f"Pooled effect in extreme tail ({pooled_percentile:.1f}%)")
if tau_percentile < 5 or tau_percentile > 95:
    fail_conditions.append(f"Observed tau in extreme tail ({tau_percentile:.1f}%)")
if np.any(np.isnan(y_pred_samples)) or np.any(np.isinf(y_pred_samples)):
    fail_conditions.append("Numerical instabilities detected")

if fail_conditions:
    decision = "FAIL"
    print(f"\nDECISION: {decision}")
    print("Reasons:")
    for reason in fail_conditions:
        print(f"  - {reason}")
else:
    decision = "PASS"
    print(f"\nDECISION: {decision}")
    print("All observed values fall within scientifically plausible ranges.")
    print("Prior predictive distributions adequately cover observed data.")

summary_stats['decision'] = decision
summary_stats['fail_conditions'] = fail_conditions
summary_stats['n_good_studies'] = n_good_studies
summary_stats['n_marginal_studies'] = n_marginal_studies
summary_stats['n_bad_studies'] = n_bad_studies

# Save summary
import json
with open('/workspace/experiments/experiment_1/prior_predictive_check/code/summary_stats.json', 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    summary_stats_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in summary_stats.items()}
    json.dump(summary_stats_json, f, indent=2)

print("\n" + "="*80)
print("Prior predictive check complete!")
print("="*80)
