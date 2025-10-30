"""
Posterior Predictive Check for Experiment 1: Robust Logarithmic Regression
=============================================================================

Comprehensive assessment of model adequacy through:
1. Graphical posterior predictive checks
2. Numerical test statistics with posterior predictive p-values
3. Residual diagnostics
4. Replicate coverage analysis
5. Systematic pattern detection

Model: Y ~ StudentT(nu, mu, sigma)
       mu = alpha + beta * log(x + c)
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import anderson_ksamp
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA AND POSTERIOR
# ============================================================================

print("="*80)
print("POSTERIOR PREDICTIVE CHECK: Robust Logarithmic Regression")
print("="*80)

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(y_obs)

print(f"\nObserved data: n = {n_obs}")
print(f"  x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"  Y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"  Y mean: {y_obs.mean():.3f}, SD: {y_obs.std():.3f}")

# Load posterior inference data
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print("\nPosterior data structure:")
print(f"  Groups: {list(idata.groups())}")
print(f"  Posterior variables: {list(idata.posterior.data_vars)}")
print(f"  Posterior_predictive variables: {list(idata.posterior_predictive.data_vars)}")

# Extract posterior predictive samples
y_rep = idata.posterior_predictive['Y_obs'].values

print(f"\nPosterior predictive shape: {y_rep.shape}")

# Reshape if needed for analysis
if y_rep.ndim == 3:  # (chains, draws, obs)
    n_chains, n_draws, n_obs_check = y_rep.shape
    y_rep = y_rep.reshape(-1, n_obs_check)
    print(f"  Reshaped from ({n_chains}, {n_draws}, {n_obs_check}) to {y_rep.shape}")

n_rep, n_obs_check = y_rep.shape
print(f"\nPosterior predictive samples: {n_rep} replicates x {n_obs_check} observations")

# ============================================================================
# POSTERIOR MEANS AND RESIDUALS
# ============================================================================

# Compute posterior mean predictions and credible intervals
y_pred_mean = y_rep.mean(axis=0)
y_pred_median = np.median(y_rep, axis=0)
y_pred_std = y_rep.std(axis=0)

# Credible intervals
y_pred_50 = np.percentile(y_rep, [25, 75], axis=0)
y_pred_90 = np.percentile(y_rep, [5, 95], axis=0)
y_pred_95 = np.percentile(y_rep, [2.5, 97.5], axis=0)

# Residuals (using posterior mean)
residuals = y_obs - y_pred_mean
standardized_residuals = residuals / y_pred_std

print("\nResidual summary:")
print(f"  Mean: {residuals.mean():.4f}")
print(f"  SD: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}, Max: {residuals.max():.4f}")
print(f"  Std. residuals - Mean: {standardized_residuals.mean():.4f}, SD: {standardized_residuals.std():.4f}")

# ============================================================================
# NUMERICAL POSTERIOR PREDICTIVE CHECKS
# ============================================================================

print("\n" + "="*80)
print("NUMERICAL POSTERIOR PREDICTIVE CHECKS")
print("="*80)

def compute_ppc_pvalue(stat_obs, stat_rep, two_sided=False):
    """
    Compute posterior predictive p-value.

    p = P(T(y_rep) >= T(y_obs) | data)

    Interpretation:
    - p in [0.05, 0.95]: GOOD
    - p < 0.05 or p > 0.95: WARNING
    - p < 0.01 or p > 0.99: FAIL
    """
    if two_sided:
        p_upper = np.mean(stat_rep >= stat_obs)
        p_lower = np.mean(stat_rep <= stat_obs)
        return min(p_upper, p_lower) * 2
    else:
        return np.mean(stat_rep >= stat_obs)

# Test statistics
test_stats = {}

# T1: Minimum (lower tail)
stat_obs = y_obs.min()
stat_rep = y_rep.min(axis=1)
p_val = compute_ppc_pvalue(stat_obs, stat_rep)
test_stats['min'] = {'obs': stat_obs, 'rep': stat_rep, 'p_value': p_val}
print(f"\nT1: min(Y)")
print(f"  Observed: {stat_obs:.4f}")
print(f"  Replicated: {stat_rep.mean():.4f} +/- {stat_rep.std():.4f}")
print(f"  P-value: {p_val:.4f} {'[GOOD]' if 0.05 <= p_val <= 0.95 else '[WARNING]' if 0.01 <= p_val <= 0.99 else '[FAIL]'}")

# T2: Maximum (upper tail)
stat_obs = y_obs.max()
stat_rep = y_rep.max(axis=1)
p_val = compute_ppc_pvalue(stat_obs, stat_rep)
test_stats['max'] = {'obs': stat_obs, 'rep': stat_rep, 'p_value': p_val}
print(f"\nT2: max(Y)")
print(f"  Observed: {stat_obs:.4f}")
print(f"  Replicated: {stat_rep.mean():.4f} +/- {stat_rep.std():.4f}")
print(f"  P-value: {p_val:.4f} {'[GOOD]' if 0.05 <= p_val <= 0.95 else '[WARNING]' if 0.01 <= p_val <= 0.99 else '[FAIL]'}")

# T3: Mean (central tendency)
stat_obs = y_obs.mean()
stat_rep = y_rep.mean(axis=1)
p_val = compute_ppc_pvalue(stat_obs, stat_rep, two_sided=True)
test_stats['mean'] = {'obs': stat_obs, 'rep': stat_rep, 'p_value': p_val}
print(f"\nT3: mean(Y)")
print(f"  Observed: {stat_obs:.4f}")
print(f"  Replicated: {stat_rep.mean():.4f} +/- {stat_rep.std():.4f}")
print(f"  P-value: {p_val:.4f} {'[GOOD]' if 0.05 <= p_val <= 0.95 else '[WARNING]' if 0.01 <= p_val <= 0.99 else '[FAIL]'}")

# T4: Standard deviation (dispersion)
stat_obs = y_obs.std()
stat_rep = y_rep.std(axis=1)
p_val = compute_ppc_pvalue(stat_obs, stat_rep, two_sided=True)
test_stats['std'] = {'obs': stat_obs, 'rep': stat_rep, 'p_value': p_val}
print(f"\nT4: SD(Y)")
print(f"  Observed: {stat_obs:.4f}")
print(f"  Replicated: {stat_rep.mean():.4f} +/- {stat_rep.std():.4f}")
print(f"  P-value: {p_val:.4f} {'[GOOD]' if 0.05 <= p_val <= 0.95 else '[WARNING]' if 0.01 <= p_val <= 0.99 else '[FAIL]'}")

# T5: Skewness (asymmetry)
stat_obs = stats.skew(y_obs)
stat_rep = np.array([stats.skew(y_rep[i, :]) for i in range(n_rep)])
p_val = compute_ppc_pvalue(stat_obs, stat_rep, two_sided=True)
test_stats['skewness'] = {'obs': stat_obs, 'rep': stat_rep, 'p_value': p_val}
print(f"\nT5: skewness(Y)")
print(f"  Observed: {stat_obs:.4f}")
print(f"  Replicated: {stat_rep.mean():.4f} +/- {stat_rep.std():.4f}")
print(f"  P-value: {p_val:.4f} {'[GOOD]' if 0.05 <= p_val <= 0.95 else '[WARNING]' if 0.01 <= p_val <= 0.99 else '[FAIL]'}")

# T6: Range (spread)
stat_obs = y_obs.max() - y_obs.min()
stat_rep = y_rep.max(axis=1) - y_rep.min(axis=1)
p_val = compute_ppc_pvalue(stat_obs, stat_rep, two_sided=True)
test_stats['range'] = {'obs': stat_obs, 'rep': stat_rep, 'p_value': p_val}
print(f"\nT6: range(Y)")
print(f"  Observed: {stat_obs:.4f}")
print(f"  Replicated: {stat_rep.mean():.4f} +/- {stat_rep.std():.4f}")
print(f"  P-value: {p_val:.4f} {'[GOOD]' if 0.05 <= p_val <= 0.95 else '[WARNING]' if 0.01 <= p_val <= 0.99 else '[FAIL]'}")

# T7: IQR (robust spread)
stat_obs = np.percentile(y_obs, 75) - np.percentile(y_obs, 25)
stat_rep = np.percentile(y_rep, 75, axis=1) - np.percentile(y_rep, 25, axis=1)
p_val = compute_ppc_pvalue(stat_obs, stat_rep, two_sided=True)
test_stats['iqr'] = {'obs': stat_obs, 'rep': stat_rep, 'p_value': p_val}
print(f"\nT7: IQR(Y)")
print(f"  Observed: {stat_obs:.4f}")
print(f"  Replicated: {stat_rep.mean():.4f} +/- {stat_rep.std():.4f}")
print(f"  P-value: {p_val:.4f} {'[GOOD]' if 0.05 <= p_val <= 0.95 else '[WARNING]' if 0.01 <= p_val <= 0.99 else '[FAIL]'}")

# Summary
p_values = [test_stats[k]['p_value'] for k in test_stats.keys()]
n_good = sum(0.05 <= p <= 0.95 for p in p_values)
n_warning = sum((0.01 <= p < 0.05) or (0.95 < p <= 0.99) for p in p_values)
n_fail = sum(p < 0.01 or p > 0.99 for p in p_values)

print("\n" + "-"*80)
print(f"SUMMARY: {n_good}/{len(p_values)} GOOD, {n_warning}/{len(p_values)} WARNING, {n_fail}/{len(p_values)} FAIL")
print("-"*80)

# ============================================================================
# REPLICATE COVERAGE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("REPLICATE COVERAGE ANALYSIS")
print("="*80)

# Overall coverage
in_50 = np.sum((y_obs >= y_pred_50[0]) & (y_obs <= y_pred_50[1]))
in_90 = np.sum((y_obs >= y_pred_90[0]) & (y_obs <= y_pred_90[1]))
in_95 = np.sum((y_obs >= y_pred_95[0]) & (y_obs <= y_pred_95[1]))

print(f"\nOverall coverage:")
print(f"  50% CI: {in_50}/{n_obs} = {100*in_50/n_obs:.1f}% (expected: 50%)")
print(f"  90% CI: {in_90}/{n_obs} = {100*in_90/n_obs:.1f}% (expected: 90%)")
print(f"  95% CI: {in_95}/{n_obs} = {100*in_95/n_obs:.1f}% (expected: 95%)")

# Coverage by x value (for replicates)
unique_x = np.unique(x_obs)
replicated_x = unique_x[np.array([sum(x_obs == ux) for ux in unique_x]) > 1]

print(f"\nReplicated x values (n > 1): {replicated_x}")
print("\nCoverage by replicated x:")

for ux in replicated_x:
    idx = x_obs == ux
    n_reps = sum(idx)
    y_subset = y_obs[idx]
    y_pred_subset = y_pred_mean[idx]
    y_pred_50_subset = y_pred_50[:, idx]

    in_ci = sum((y_subset >= y_pred_50_subset[0]) & (y_subset <= y_pred_50_subset[1]))

    print(f"  x = {ux:5.1f}: {in_ci}/{n_reps} in 50% CI, "
          f"Y = {y_subset.min():.2f}-{y_subset.max():.2f}, "
          f"pred = {y_pred_subset.mean():.2f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'test_statistics': test_stats,
    'coverage': {
        'in_50': int(in_50),
        'in_90': int(in_90),
        'in_95': int(in_95),
        'pct_50': float(100*in_50/n_obs),
        'pct_90': float(100*in_90/n_obs),
        'pct_95': float(100*in_95/n_obs)
    },
    'residuals': {
        'values': residuals.tolist(),
        'mean': float(residuals.mean()),
        'std': float(residuals.std()),
        'min': float(residuals.min()),
        'max': float(residuals.max())
    }
}

# Save for later use
np.savez('/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_results.npz',
         x_obs=x_obs,
         y_obs=y_obs,
         y_rep=y_rep,
         y_pred_mean=y_pred_mean,
         y_pred_median=y_pred_median,
         y_pred_std=y_pred_std,
         y_pred_50=y_pred_50,
         y_pred_90=y_pred_90,
         y_pred_95=y_pred_95,
         residuals=residuals,
         standardized_residuals=standardized_residuals)

print("\n" + "="*80)
print("Results saved to: ppc_results.npz")
print("="*80)
