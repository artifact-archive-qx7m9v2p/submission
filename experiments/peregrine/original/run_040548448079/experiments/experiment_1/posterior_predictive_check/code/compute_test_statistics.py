"""
Compute test statistics for posterior predictive checks.

Compare observed data with posterior predictive distribution across
multiple dimensions to assess model fit quality.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Load data
print("Loading data and posterior predictive samples...")
C_obs = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/C_obs.npy')
C_rep = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/C_rep.npy')
year = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/year.npy')
tau = int(np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/tau.npy'))

n_pp_samples = C_rep.shape[0]
N = len(C_obs)

print(f"  Observed data: {C_obs.shape}")
print(f"  PP samples: {C_rep.shape}")
print(f"  Changepoint: tau = {tau}")

# Function to compute ACF(1)
def acf_lag1(x):
    """Compute autocorrelation at lag 1."""
    x_centered = x - x.mean()
    c0 = np.dot(x_centered, x_centered) / len(x)
    c1 = np.dot(x_centered[:-1], x_centered[1:]) / len(x)
    return c1 / c0 if c0 > 0 else 0

# Define test statistics
test_stats = {}

print("\n" + "="*80)
print("COMPUTING TEST STATISTICS")
print("="*80)

# 1. Mean
print("\n[1] Mean")
obs_mean = C_obs.mean()
pp_mean = C_rep.mean(axis=1)
test_stats['mean'] = {
    'observed': obs_mean,
    'pp_samples': pp_mean,
    'pp_mean': pp_mean.mean(),
    'pp_std': pp_mean.std(),
    'p_value': (pp_mean >= obs_mean).mean()
}
print(f"  Observed: {obs_mean:.2f}")
print(f"  PP: {pp_mean.mean():.2f} ± {pp_mean.std():.2f}")
print(f"  Bayesian p-value: {test_stats['mean']['p_value']:.3f}")

# 2. Variance
print("\n[2] Variance")
obs_var = C_obs.var()
pp_var = C_rep.var(axis=1)
test_stats['variance'] = {
    'observed': obs_var,
    'pp_samples': pp_var,
    'pp_mean': pp_var.mean(),
    'pp_std': pp_var.std(),
    'p_value': (pp_var >= obs_var).mean()
}
print(f"  Observed: {obs_var:.2f}")
print(f"  PP: {pp_var.mean():.2f} ± {pp_var.std():.2f}")
print(f"  Bayesian p-value: {test_stats['variance']['p_value']:.3f}")

# 3. Variance/Mean ratio (overdispersion)
print("\n[3] Variance/Mean Ratio (Overdispersion)")
obs_vm_ratio = obs_var / obs_mean
pp_vm_ratio = pp_var / pp_mean
test_stats['var_mean_ratio'] = {
    'observed': obs_vm_ratio,
    'pp_samples': pp_vm_ratio,
    'pp_mean': pp_vm_ratio.mean(),
    'pp_std': pp_vm_ratio.std(),
    'p_value': (pp_vm_ratio >= obs_vm_ratio).mean()
}
print(f"  Observed: {obs_vm_ratio:.2f}")
print(f"  PP: {pp_vm_ratio.mean():.2f} ± {pp_vm_ratio.std():.2f}")
print(f"  Bayesian p-value: {test_stats['var_mean_ratio']['p_value']:.3f}")

# 4. Minimum
print("\n[4] Minimum")
obs_min = C_obs.min()
pp_min = C_rep.min(axis=1)
test_stats['min'] = {
    'observed': obs_min,
    'pp_samples': pp_min,
    'pp_mean': pp_min.mean(),
    'pp_std': pp_min.std(),
    'p_value': (pp_min <= obs_min).mean()
}
print(f"  Observed: {obs_min:.0f}")
print(f"  PP: {pp_min.mean():.2f} ± {pp_min.std():.2f}")
print(f"  Bayesian p-value: {test_stats['min']['p_value']:.3f}")

# 5. Maximum
print("\n[5] Maximum")
obs_max = C_obs.max()
pp_max = C_rep.max(axis=1)
test_stats['max'] = {
    'observed': obs_max,
    'pp_samples': pp_max,
    'pp_mean': pp_max.mean(),
    'pp_std': pp_max.std(),
    'p_value': (pp_max >= obs_max).mean()
}
print(f"  Observed: {obs_max:.0f}")
print(f"  PP: {pp_max.mean():.2f} ± {pp_max.std():.2f}")
print(f"  Bayesian p-value: {test_stats['max']['p_value']:.3f}")

# 6. ACF(1) - Autocorrelation at lag 1
print("\n[6] Autocorrelation at lag 1")
obs_acf1 = acf_lag1(C_obs)
pp_acf1 = np.array([acf_lag1(C_rep[i, :]) for i in range(n_pp_samples)])
test_stats['acf1'] = {
    'observed': obs_acf1,
    'pp_samples': pp_acf1,
    'pp_mean': pp_acf1.mean(),
    'pp_std': pp_acf1.std(),
    'p_value': (pp_acf1 >= obs_acf1).mean()
}
print(f"  Observed: {obs_acf1:.3f}")
print(f"  PP: {pp_acf1.mean():.3f} ± {pp_acf1.std():.3f}")
print(f"  Bayesian p-value: {test_stats['acf1']['p_value']:.3f}")

# 7. Pre-break mean
print("\n[7] Pre-break Mean (t ≤ {})".format(tau))
obs_pre_mean = C_obs[:tau].mean()
pp_pre_mean = C_rep[:, :tau].mean(axis=1)
test_stats['pre_mean'] = {
    'observed': obs_pre_mean,
    'pp_samples': pp_pre_mean,
    'pp_mean': pp_pre_mean.mean(),
    'pp_std': pp_pre_mean.std(),
    'p_value': (pp_pre_mean >= obs_pre_mean).mean()
}
print(f"  Observed: {obs_pre_mean:.2f}")
print(f"  PP: {pp_pre_mean.mean():.2f} ± {pp_pre_mean.std():.2f}")
print(f"  Bayesian p-value: {test_stats['pre_mean']['p_value']:.3f}")

# 8. Post-break mean
print("\n[8] Post-break Mean (t > {})".format(tau))
obs_post_mean = C_obs[tau:].mean()
pp_post_mean = C_rep[:, tau:].mean(axis=1)
test_stats['post_mean'] = {
    'observed': obs_post_mean,
    'pp_samples': pp_post_mean,
    'pp_mean': pp_post_mean.mean(),
    'pp_std': pp_post_mean.std(),
    'p_value': (pp_post_mean >= obs_post_mean).mean()
}
print(f"  Observed: {obs_post_mean:.2f}")
print(f"  PP: {pp_post_mean.mean():.2f} ± {pp_post_mean.std():.2f}")
print(f"  Bayesian p-value: {test_stats['post_mean']['p_value']:.3f}")

# 9. Growth ratio (post/pre)
print("\n[9] Growth Ratio (Post/Pre)")
obs_growth = obs_post_mean / obs_pre_mean
pp_growth = pp_post_mean / pp_pre_mean
test_stats['growth_ratio'] = {
    'observed': obs_growth,
    'pp_samples': pp_growth,
    'pp_mean': pp_growth.mean(),
    'pp_std': pp_growth.std(),
    'p_value': (pp_growth >= obs_growth).mean()
}
print(f"  Observed: {obs_growth:.3f}x")
print(f"  PP: {pp_growth.mean():.3f}x ± {pp_growth.std():.3f}")
print(f"  Bayesian p-value: {test_stats['growth_ratio']['p_value']:.3f}")

# 10. Quantiles
print("\n[10] Quantiles")
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
obs_quantiles = np.quantile(C_obs, quantiles)
pp_quantiles = np.array([np.quantile(C_rep[i, :], quantiles) for i in range(n_pp_samples)])
test_stats['quantiles'] = {
    'levels': quantiles,
    'observed': obs_quantiles,
    'pp_mean': pp_quantiles.mean(axis=0),
    'pp_std': pp_quantiles.std(axis=0)
}
for q, obs_q, pp_q_mean, pp_q_std in zip(quantiles, obs_quantiles,
                                           pp_quantiles.mean(axis=0),
                                           pp_quantiles.std(axis=0)):
    print(f"  Q{int(q*100)}: Obs={obs_q:.1f}, PP={pp_q_mean:.1f}±{pp_q_std:.1f}")

# 11. Coverage - what % of observed points fall within 90% PP HDI
print("\n[11] Coverage (90% HDI)")
pp_lower = np.percentile(C_rep, 5, axis=0)
pp_upper = np.percentile(C_rep, 95, axis=0)
within_hdi = ((C_obs >= pp_lower) & (C_obs <= pp_upper)).sum()
coverage_pct = 100 * within_hdi / N
test_stats['coverage'] = {
    'within_hdi': within_hdi,
    'total': N,
    'percentage': coverage_pct
}
print(f"  Points within 90% HDI: {within_hdi}/{N} ({coverage_pct:.1f}%)")
print(f"  Expected: ~90% for well-calibrated model")

# Save test statistics
np.save('/workspace/experiments/experiment_1/posterior_predictive_check/code/test_stats.npy',
        test_stats, allow_pickle=True)
print("\n" + "="*80)
print("Test statistics saved to test_stats.npy")

# Summary table
print("\n" + "="*80)
print("BAYESIAN P-VALUE SUMMARY")
print("="*80)
print("\nA p-value close to 0.5 is ideal.")
print("Extreme values (<0.05 or >0.95) indicate poor fit.\n")

summary_df = pd.DataFrame({
    'Statistic': ['Mean', 'Variance', 'Var/Mean', 'Min', 'Max', 'ACF(1)',
                  'Pre-break Mean', 'Post-break Mean', 'Growth Ratio'],
    'Observed': [test_stats['mean']['observed'], test_stats['variance']['observed'],
                 test_stats['var_mean_ratio']['observed'], test_stats['min']['observed'],
                 test_stats['max']['observed'], test_stats['acf1']['observed'],
                 test_stats['pre_mean']['observed'], test_stats['post_mean']['observed'],
                 test_stats['growth_ratio']['observed']],
    'PP Mean': [test_stats['mean']['pp_mean'], test_stats['variance']['pp_mean'],
                test_stats['var_mean_ratio']['pp_mean'], test_stats['min']['pp_mean'],
                test_stats['max']['pp_mean'], test_stats['acf1']['pp_mean'],
                test_stats['pre_mean']['pp_mean'], test_stats['post_mean']['pp_mean'],
                test_stats['growth_ratio']['pp_mean']],
    'PP SD': [test_stats['mean']['pp_std'], test_stats['variance']['pp_std'],
              test_stats['var_mean_ratio']['pp_std'], test_stats['min']['pp_std'],
              test_stats['max']['pp_std'], test_stats['acf1']['pp_std'],
              test_stats['pre_mean']['pp_std'], test_stats['post_mean']['pp_std'],
              test_stats['growth_ratio']['pp_std']],
    'p-value': [test_stats['mean']['p_value'], test_stats['variance']['p_value'],
                test_stats['var_mean_ratio']['p_value'], test_stats['min']['p_value'],
                test_stats['max']['p_value'], test_stats['acf1']['p_value'],
                test_stats['pre_mean']['p_value'], test_stats['post_mean']['p_value'],
                test_stats['growth_ratio']['p_value']],
    'Status': [''] * 9
})

# Classify status
for i, p in enumerate(summary_df['p-value']):
    if p < 0.05 or p > 0.95:
        summary_df.loc[i, 'Status'] = 'EXTREME'
    elif 0.05 <= p < 0.25 or 0.75 < p <= 0.95:
        summary_df.loc[i, 'Status'] = 'Marginal'
    else:
        summary_df.loc[i, 'Status'] = 'OK'

print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv('/workspace/experiments/experiment_1/posterior_predictive_check/code/test_stats_summary.csv',
                  index=False)
print("\n" + "="*80)
print("Summary saved to test_stats_summary.csv")
print("="*80)
