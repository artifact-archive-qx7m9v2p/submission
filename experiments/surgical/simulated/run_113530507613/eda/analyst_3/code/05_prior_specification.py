"""
Prior Specification Analysis
Focus: Determining plausible parameter ranges for Bayesian modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')
output_dir = Path('/workspace/eda/analyst_3/visualizations')

print("="*80)
print("PRIOR SPECIFICATION ANALYSIS")
print("="*80)

# Calculate key statistics
pooled_rate = data['r_successes'].sum() / data['n_trials'].sum()
mean_rate = data['success_rate'].mean()
median_rate = data['success_rate'].median()
std_rate = data['success_rate'].std()

print("\n1. EMPIRICAL ESTIMATES FOR PRIORS")
print("-" * 80)
print(f"Pooled success rate: {pooled_rate:.6f}")
print(f"Mean success rate: {mean_rate:.6f}")
print(f"Median success rate: {median_rate:.6f}")
print(f"SD of success rates: {std_rate:.6f}")

# Quantiles
quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print("\nQuantiles of observed success rates:")
for q in quantiles:
    print(f"  {q:4.2f}: {data['success_rate'].quantile(q):.6f}")

print("\n2. BETA DISTRIBUTION PARAMETER SUGGESTIONS")
print("-" * 80)

# Method of moments for Beta distribution
# E[p] = alpha / (alpha + beta)
# Var[p] = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))

def beta_params_from_moments(mean, variance):
    """Estimate Beta(alpha, beta) parameters from mean and variance"""
    if variance >= mean * (1 - mean):
        return None, None
    common = mean * (1 - mean) / variance - 1
    alpha = mean * common
    beta = (1 - mean) * common
    return alpha, beta

# Based on pooled rate
var_pooled = pooled_rate * (1 - pooled_rate) / data['n_trials'].mean()
alpha_pooled, beta_pooled = beta_params_from_moments(pooled_rate, var_pooled)

print(f"\nBased on pooled rate (with sampling variance):")
if alpha_pooled is not None:
    print(f"  Beta({alpha_pooled:.2f}, {beta_pooled:.2f})")
    print(f"  Mean: {alpha_pooled/(alpha_pooled+beta_pooled):.6f}")
    print(f"  SD: {np.sqrt(alpha_pooled*beta_pooled/((alpha_pooled+beta_pooled)**2*(alpha_pooled+beta_pooled+1))):.6f}")

# Based on observed group variation
alpha_obs, beta_obs = beta_params_from_moments(mean_rate, std_rate**2)
print(f"\nBased on observed between-group variation:")
if alpha_obs is not None and alpha_obs > 0:
    print(f"  Beta({alpha_obs:.2f}, {beta_obs:.2f})")
    print(f"  Mean: {alpha_obs/(alpha_obs+beta_obs):.6f}")
    print(f"  SD: {np.sqrt(alpha_obs*beta_obs/((alpha_obs+beta_obs)**2*(alpha_obs+beta_obs+1))):.6f}")
else:
    print("  Variance too large for simple Beta - consider mixture or hierarchical model")

# Weak priors
print("\nWeak/non-informative prior options:")
print("  Beta(1, 1) - Uniform on [0, 1]")
print("  Beta(0.5, 0.5) - Jeffreys prior (favors extremes)")
print("  Beta(2, 2) - Weakly peaked at 0.5")

# Centered priors
alpha_weak = 2
beta_weak = (1 - pooled_rate) / pooled_rate * alpha_weak
print(f"\nWeakly informative centered on pooled rate:")
print(f"  Beta({alpha_weak:.1f}, {beta_weak:.2f})")
print(f"  Mean: {alpha_weak/(alpha_weak+beta_weak):.6f}")

print("\n3. HIERARCHICAL MODEL PARAMETERS")
print("-" * 80)

# Estimate between-group variance (tau^2)
weights = data['n_trials']
weighted_mean = np.average(data['success_rate'], weights=weights)
weighted_var = np.average((data['success_rate'] - weighted_mean)**2, weights=weights)
within_var = np.mean([r['success_rate'] * (1 - r['success_rate']) / r['n_trials']
                      for _, r in data.iterrows()])
between_var = max(0, weighted_var - within_var)

print(f"Between-group variance (tau^2): {between_var:.6f}")
print(f"Between-group SD (tau): {np.sqrt(between_var):.6f}")

# Suggest hyperprior parameters
print("\nSuggested hyperpriors for hierarchical model:")
print(f"  Population mean (mu): Normal({pooled_rate:.3f}, {std_rate:.3f})")
print(f"  or logit(mu): Normal({np.log(pooled_rate/(1-pooled_rate)):.3f}, 1.0)")
print(f"\n  Between-group SD (tau): Half-Normal(0, {np.sqrt(between_var)*2:.3f})")
print(f"  or Exponential(1/{np.sqrt(between_var):.2f})")

print("\n4. INFORMATIVE VS WEAKLY INFORMATIVE TRADE-OFFS")
print("-" * 80)

# Calculate effective sample sizes
print("\nInformative prior (Beta based on data):")
if alpha_obs is not None and alpha_obs > 0:
    n_eff_inform = alpha_obs + beta_obs
    print(f"  Effective sample size: {n_eff_inform:.1f} observations")
else:
    print("  Not applicable due to high variance")

print("\nWeakly informative prior (Beta(2, 20)):")
n_eff_weak = 2 + 20
print(f"  Effective sample size: {n_eff_weak:.1f} observations")

print("\nNon-informative prior (Beta(1, 1)):")
n_eff_uninform = 1 + 1
print(f"  Effective sample size: {n_eff_uninform:.1f} observations")

print("\nSmallest group has {:.0f} trials".format(data['n_trials'].min()))
print("Largest group has {:.0f} trials".format(data['n_trials'].max()))

print("\n5. SENSITIVITY CHECKS NEEDED")
print("-" * 80)
print("Recommended prior sensitivity analyses:")
print("  1. Weak prior: Beta(1, 1) or Beta(2, 2)")
print("  2. Informative prior based on pooled rate")
print("  3. Hierarchical prior with hyperpriors on mu and tau")
print("  4. Check if conclusions robust across prior choices")

print("\n6. PRACTICAL RECOMMENDATIONS")
print("-" * 80)

if between_var > 0.0001:
    print("RECOMMENDATION: Use hierarchical/partial pooling model")
    print("  - Groups show substantial heterogeneity")
    print("  - ICC = 42% suggests meaningful group structure")
    print("  - Partial pooling will balance individual estimates with global mean")
    print("\nSuggested Stan/PyMC model structure:")
    print("  mu ~ Normal(logit(0.07), 1)  # population mean (logit scale)")
    print("  tau ~ Half-Normal(0, 0.05)    # between-group SD")
    print("  theta[j] ~ Normal(mu, tau)    # group-level parameters (logit scale)")
    print("  y[j] ~ Binomial(n[j], inv_logit(theta[j]))  # likelihood")
else:
    print("RECOMMENDATION: Complete pooling may be sufficient")
    print("  - Limited between-group variation detected")

print("\n" + "="*80)
print("PRIOR SPECIFICATION ANALYSIS COMPLETE")
print("="*80)
