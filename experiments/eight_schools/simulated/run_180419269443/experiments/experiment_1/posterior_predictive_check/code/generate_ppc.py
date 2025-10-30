"""
Posterior Predictive Check for Experiment 1: Hierarchical Normal Model

This script:
1. Loads posterior samples from ArviZ InferenceData (Stan/PyMC fit)
2. Generates posterior predictive samples
3. Computes test statistics and Bayesian p-values
4. Creates diagnostic visualizations
5. Assesses model fit quality
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DATA AND POSTERIOR SAMPLES
# ============================================================================

print("="*80)
print("POSTERIOR PREDICTIVE CHECK: Hierarchical Normal Model")
print("="*80)

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
y_obs = data['y'].values
sigma = data['sigma'].values
n_studies = len(y_obs)

print(f"\nObserved data:")
print(f"  Studies: {n_studies}")
print(f"  y_obs: {y_obs}")
print(f"  sigma: {sigma}")

# Load posterior samples from ArviZ InferenceData
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print(f"\nPosterior samples loaded:")
print(f"  Chains: {idata.posterior.dims['chain']}")
print(f"  Draws per chain: {idata.posterior.dims['draw']}")
print(f"  Total samples: {idata.posterior.dims['chain'] * idata.posterior.dims['draw']}")

# Check if posterior_predictive already exists
if 'posterior_predictive' in idata.groups():
    print(f"  Posterior predictive group already exists in data")
    print(f"  Variables: {list(idata.posterior_predictive.data_vars)}")

# Extract posterior samples
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values.reshape(-1, n_studies)

n_samples = len(mu_samples)
print(f"  Using {n_samples} posterior samples for PPC")

# ============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================

print("\n" + "="*80)
print("GENERATING POSTERIOR PREDICTIVE SAMPLES")
print("="*80)

# For each posterior sample, generate replicated data
# y_rep[i, j] ~ Normal(theta[i, j], sigma[j])
np.random.seed(42)
y_rep = np.zeros((n_samples, n_studies))

for i in range(n_samples):
    for j in range(n_studies):
        y_rep[i, j] = np.random.normal(theta_samples[i, j], sigma[j])

print(f"\nGenerated {n_samples} replicated datasets")
print(f"  Shape: {y_rep.shape} (samples × studies)")

# ============================================================================
# 3. COMPUTE TEST STATISTICS
# ============================================================================

print("\n" + "="*80)
print("COMPUTING TEST STATISTICS")
print("="*80)

# Define test statistics
def compute_test_statistics(y):
    """Compute test statistics for a dataset"""
    return {
        'mean': np.mean(y),
        'sd': np.std(y, ddof=1),
        'min': np.min(y),
        'max': np.max(y),
        'range': np.max(y) - np.min(y),
        'n_negative': np.sum(y < 0),
        'q25': np.percentile(y, 25),
        'q75': np.percentile(y, 75),
        'iqr': np.percentile(y, 75) - np.percentile(y, 25)
    }

# Observed test statistics
T_obs = compute_test_statistics(y_obs)

# Replicated test statistics
T_rep = {key: np.zeros(n_samples) for key in T_obs.keys()}
for i in range(n_samples):
    T_rep_i = compute_test_statistics(y_rep[i, :])
    for key in T_obs.keys():
        T_rep[key][i] = T_rep_i[key]

# Compute Bayesian p-values
print("\nTest Statistics and Bayesian p-values:")
print(f"{'Statistic':<15} {'T_obs':>10} {'T_rep mean':>12} {'T_rep SD':>10} {'p-value':>10} {'Assessment':>15}")
print("-"*80)

pvalues = {}
for key in T_obs.keys():
    p_value = np.mean(T_rep[key] >= T_obs[key])
    pvalues[key] = p_value

    # Two-tailed p-value (distance from 0.5)
    extreme_p = min(p_value, 1 - p_value)

    if extreme_p < 0.025:
        assessment = "POOR FIT"
    elif extreme_p < 0.05:
        assessment = "Marginal"
    else:
        assessment = "Good"

    print(f"{key:<15} {T_obs[key]:>10.2f} {np.mean(T_rep[key]):>12.2f} {np.std(T_rep[key]):>10.2f} {p_value:>10.3f} {assessment:>15}")

# Study-specific standardized residuals
print("\nStudy-specific diagnostics:")
print(f"{'Study':<8} {'y_obs':>10} {'theta_mean':>12} {'sigma':>10} {'z_score':>10} {'p-value':>10} {'Assessment':>15}")
print("-"*80)

study_pvalues = []
for j in range(n_studies):
    theta_mean = np.mean(theta_samples[:, j])
    z_obs = (y_obs[j] - theta_mean) / sigma[j]

    # Generate z-scores for replicated data
    z_rep = (y_rep[:, j] - theta_samples[:, j]) / sigma[j]

    # Bayesian p-value: P(|z_rep| >= |z_obs|)
    p_value = np.mean(np.abs(z_rep) >= np.abs(z_obs))
    study_pvalues.append(p_value)

    if p_value < 0.05 or p_value > 0.95:
        assessment = "POOR FIT"
    elif p_value < 0.1 or p_value > 0.9:
        assessment = "Marginal"
    else:
        assessment = "Good"

    print(f"{j+1:<8} {y_obs[j]:>10.2f} {theta_mean:>12.2f} {sigma[j]:>10.2f} {z_obs:>10.2f} {p_value:>10.3f} {assessment:>15}")

# ============================================================================
# 4. OVERALL FIT ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("OVERALL FIT ASSESSMENT")
print("="*80)

# Count test statistics by fit quality
n_poor = sum([1 for p in pvalues.values() if min(p, 1-p) < 0.025])
n_marginal = sum([1 for p in pvalues.values() if 0.025 <= min(p, 1-p) < 0.05])
n_good = sum([1 for p in pvalues.values() if min(p, 1-p) >= 0.05])

print(f"\nTest statistics summary:")
print(f"  Good fit: {n_good}/{len(pvalues)}")
print(f"  Marginal fit: {n_marginal}/{len(pvalues)}")
print(f"  Poor fit: {n_poor}/{len(pvalues)}")

# Study-specific summary
n_study_poor = sum([1 for p in study_pvalues if p < 0.05 or p > 0.95])
n_study_marginal = sum([1 for p in study_pvalues if 0.05 <= p < 0.1 or 0.9 < p <= 0.95])
n_study_good = sum([1 for p in study_pvalues if 0.1 <= p <= 0.9])

print(f"\nStudy-specific summary:")
print(f"  Good fit: {n_study_good}/{n_studies}")
print(f"  Marginal fit: {n_study_marginal}/{n_studies}")
print(f"  Poor fit: {n_study_poor}/{n_studies}")

# Overall assessment
if n_poor == 0 and n_study_poor == 0:
    if n_marginal <= 2 and n_study_marginal <= 2:
        overall_fit = "GOOD FIT"
    else:
        overall_fit = "ACCEPTABLE FIT"
elif n_poor <= 1 and n_study_poor <= 1:
    overall_fit = "ACCEPTABLE FIT"
else:
    overall_fit = "POOR FIT"

print(f"\nOVERALL ASSESSMENT: {overall_fit}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

# Save results
np.savez('/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_results.npz',
         **{f'T_rep_{k}': v for k, v in T_rep.items()},
         **{f'T_obs_{k}': v for k, v in T_obs.items()},
         **{f'pvalue_{k}': v for k, v in pvalues.items()},
         study_pvalues=np.array(study_pvalues),
         y_rep=y_rep,
         theta_samples=theta_samples,
         y_obs=y_obs,
         sigma=sigma,
         overall_fit=overall_fit)

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print(f"  Test statistics: ppc_results.npz")
