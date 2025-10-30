"""
Prior Predictive Check for Experiment 2: Student-t Logarithmic Model

Model:
    Y_i ~ StudentT(nu, mu_i, sigma)
    mu_i = beta_0 + beta_1 * log(x_i)

Priors:
    beta_0 ~ Normal(2.3, 0.5)
    beta_1 ~ Normal(0.29, 0.15)
    sigma ~ Exponential(10)  # mean = 0.1
    nu ~ Gamma(2, 0.1)       # mean = 20, allows 3-100 range

Key Questions for Student-t:
1. Does nu prior allow both robust (nu<10) and near-Normal (nu>30) behavior?
2. Do heavy tails produce reasonable outliers, not wild extremes?
3. How does prior predictive compare to Model 1 (Normal likelihood)?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_DRAWS = 1000
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = "/workspace/experiments/experiment_2/prior_predictive_check"

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
x = data['x'].values
y_obs = data['Y'].values
n = len(x)

print(f"Data: n={n}, x range=[{x.min():.1f}, {x.max():.1f}], Y range=[{y_obs.min():.2f}, {y_obs.max():.2f}]")

# Prior specifications
PRIORS = {
    'beta_0': {'dist': 'Normal', 'params': (2.3, 0.5)},
    'beta_1': {'dist': 'Normal', 'params': (0.29, 0.15)},
    'sigma': {'dist': 'Exponential', 'params': (10,)},  # rate parameter
    'nu': {'dist': 'Gamma', 'params': (2, 0.1)}  # shape, rate -> mean = 2/0.1 = 20
}

print("\nPrior Specifications:")
for param, info in PRIORS.items():
    print(f"  {param}: {info['dist']}{info['params']}")

# Sample from priors
print(f"\nSampling {N_DRAWS} draws from joint prior...")
beta_0_samples = np.random.normal(PRIORS['beta_0']['params'][0],
                                   PRIORS['beta_0']['params'][1],
                                   N_DRAWS)
beta_1_samples = np.random.normal(PRIORS['beta_1']['params'][0],
                                   PRIORS['beta_1']['params'][1],
                                   N_DRAWS)
sigma_samples = np.random.exponential(1/PRIORS['sigma']['params'][0], N_DRAWS)
nu_samples = np.random.gamma(PRIORS['nu']['params'][0],
                             1/PRIORS['nu']['params'][1],
                             N_DRAWS)

print("\nPrior sample statistics:")
print(f"  beta_0: mean={beta_0_samples.mean():.3f}, sd={beta_0_samples.std():.3f}, range=[{beta_0_samples.min():.3f}, {beta_0_samples.max():.3f}]")
print(f"  beta_1: mean={beta_1_samples.mean():.3f}, sd={beta_1_samples.std():.3f}, range=[{beta_1_samples.min():.3f}, {beta_1_samples.max():.3f}]")
print(f"  sigma: mean={sigma_samples.mean():.3f}, sd={sigma_samples.std():.3f}, range=[{sigma_samples.min():.3f}, {sigma_samples.max():.3f}]")
print(f"  nu: mean={nu_samples.mean():.2f}, sd={nu_samples.std():.2f}, range=[{nu_samples.min():.2f}, {nu_samples.max():.2f}]")

# Generate prior predictive datasets using Student-t
print("\nGenerating prior predictive datasets...")
y_pred_all = np.zeros((N_DRAWS, n))
log_x = np.log(x)

for i in range(N_DRAWS):
    beta_0 = beta_0_samples[i]
    beta_1 = beta_1_samples[i]
    sigma = sigma_samples[i]
    nu = nu_samples[i]

    # Mean function
    mu = beta_0 + beta_1 * log_x

    # Generate data from Student-t
    # StudentT(nu, mu, sigma) = mu + sigma * t(nu)
    # where t(nu) is standard Student-t with nu degrees of freedom
    y_pred_all[i, :] = mu + sigma * np.random.standard_t(nu, size=n)

print("Prior predictive datasets generated.")

# Compute summary statistics
y_pred_min = y_pred_all.min(axis=1)
y_pred_max = y_pred_all.max(axis=1)
y_pred_range = y_pred_max - y_pred_min
y_pred_sd = y_pred_all.std(axis=1)
y_pred_mean = y_pred_all.mean(axis=1)

# Validation checks
print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

# Check 1: Domain violations (extreme values)
extreme_threshold_min = -20
extreme_threshold_max = 20
n_violations_extreme = np.sum((y_pred_all < extreme_threshold_min) | (y_pred_all > extreme_threshold_max))
pct_violations_extreme = 100 * n_violations_extreme / (N_DRAWS * n)

moderate_threshold_min = -10
moderate_threshold_max = 10
n_violations_moderate = np.sum((y_pred_all < moderate_threshold_min) | (y_pred_all > moderate_threshold_max))
pct_violations_moderate = 100 * n_violations_moderate / (N_DRAWS * n)

print(f"\nCheck 1: Domain Violations")
print(f"  Extreme violations (outside [-20, 20]): {pct_violations_extreme:.2f}% (FAIL if >20%)")
print(f"  Moderate violations (outside [-10, 10]): {pct_violations_moderate:.2f}% (FAIL if >10%)")
check1_status = "PASS" if pct_violations_extreme < 20 and pct_violations_moderate < 10 else "FAIL"
print(f"  Status: {check1_status}")

# Check 2: Slope sign
n_negative_slopes = np.sum(beta_1_samples < 0)
pct_negative_slopes = 100 * n_negative_slopes / N_DRAWS
print(f"\nCheck 2: Slope Sign")
print(f"  Negative slopes (beta_1 < 0): {pct_negative_slopes:.2f}% (FAIL if >5%)")
check2_status = "PASS" if pct_negative_slopes <= 5 else "FAIL"
print(f"  Status: {check2_status}")

# Check 3: Scale parameter
n_large_sigma = np.sum(sigma_samples > 1.0)
pct_large_sigma = 100 * n_large_sigma / N_DRAWS
print(f"\nCheck 3: Scale Parameter")
print(f"  Large sigma (>1.0): {pct_large_sigma:.2f}% (FAIL if >10%)")
check3_status = "PASS" if pct_large_sigma <= 10 else "FAIL"
print(f"  Status: {check3_status}")

# Check 4: Nu distribution - key for Student-t
nu_quantiles = np.percentile(nu_samples, [5, 25, 50, 75, 95])
pct_very_heavy = 100 * np.sum(nu_samples < 5) / N_DRAWS
pct_heavy = 100 * np.sum((nu_samples >= 5) & (nu_samples < 20)) / N_DRAWS
pct_moderate = 100 * np.sum((nu_samples >= 20) & (nu_samples < 30)) / N_DRAWS
pct_near_normal = 100 * np.sum(nu_samples >= 30) / N_DRAWS

print(f"\nCheck 4: Degrees of Freedom Distribution")
print(f"  Nu quantiles: 5%={nu_quantiles[0]:.1f}, 25%={nu_quantiles[1]:.1f}, 50%={nu_quantiles[2]:.1f}, 75%={nu_quantiles[3]:.1f}, 95%={nu_quantiles[4]:.1f}")
print(f"  Very heavy tails (nu < 5): {pct_very_heavy:.1f}%")
print(f"  Heavy tails (5 <= nu < 20): {pct_heavy:.1f}%")
print(f"  Moderate (20 <= nu < 30): {pct_moderate:.1f}%")
print(f"  Near-Normal (nu >= 30): {pct_near_normal:.1f}%")
check4_status = "PASS" if (pct_very_heavy > 5 and pct_near_normal > 10) else "CAUTION"
print(f"  Status: {check4_status} (should explore both heavy and near-Normal)")

# Check 5: Coverage of observed data
y_obs_min, y_obs_max = y_obs.min(), y_obs.max()
y_pred_envelope_min = np.percentile(y_pred_min, 1)
y_pred_envelope_max = np.percentile(y_pred_max, 99)
coverage_ok = (y_obs_min >= y_pred_envelope_min) and (y_obs_max <= y_pred_envelope_max)

print(f"\nCheck 5: Coverage of Observed Data")
print(f"  Observed range: [{y_obs_min:.2f}, {y_obs_max:.2f}]")
print(f"  Prior 98% envelope: [{y_pred_envelope_min:.2f}, {y_pred_envelope_max:.2f}]")
check5_status = "PASS" if coverage_ok else "FAIL"
print(f"  Status: {check5_status}")

# Overall assessment
all_checks = [check1_status, check2_status, check3_status, check5_status]
overall_pass = all(c == "PASS" for c in all_checks)
print(f"\n{'='*70}")
print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
print(f"{'='*70}")

# Save summary statistics
summary_stats = {
    'n_draws': N_DRAWS,
    'prior_samples': {
        'beta_0': {'mean': float(beta_0_samples.mean()), 'sd': float(beta_0_samples.std()),
                   'min': float(beta_0_samples.min()), 'max': float(beta_0_samples.max())},
        'beta_1': {'mean': float(beta_1_samples.mean()), 'sd': float(beta_1_samples.std()),
                   'min': float(beta_1_samples.min()), 'max': float(beta_1_samples.max())},
        'sigma': {'mean': float(sigma_samples.mean()), 'sd': float(sigma_samples.std()),
                  'min': float(sigma_samples.min()), 'max': float(sigma_samples.max())},
        'nu': {'mean': float(nu_samples.mean()), 'sd': float(nu_samples.std()),
               'min': float(nu_samples.min()), 'max': float(nu_samples.max()),
               'quantiles': {'5': float(nu_quantiles[0]), '25': float(nu_quantiles[1]),
                           '50': float(nu_quantiles[2]), '75': float(nu_quantiles[3]),
                           '95': float(nu_quantiles[4])}}
    },
    'validation_checks': {
        'domain_violations_extreme': {'pct': float(pct_violations_extreme), 'status': check1_status},
        'domain_violations_moderate': {'pct': float(pct_violations_moderate), 'status': check1_status},
        'negative_slopes': {'pct': float(pct_negative_slopes), 'status': check2_status},
        'large_sigma': {'pct': float(pct_large_sigma), 'status': check3_status},
        'nu_distribution': {
            'very_heavy': float(pct_very_heavy),
            'heavy': float(pct_heavy),
            'moderate': float(pct_moderate),
            'near_normal': float(pct_near_normal),
            'status': check4_status
        },
        'coverage': {'status': check5_status}
    },
    'overall': 'PASS' if overall_pass else 'FAIL'
}

with open(f"{OUTPUT_DIR}/summary_stats.json", 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"\nSummary statistics saved to {OUTPUT_DIR}/summary_stats.json")

# Save samples for plotting
print("Saving samples for visualization...")
np.savez(f"{OUTPUT_DIR}/code/prior_samples.npz",
         beta_0=beta_0_samples,
         beta_1=beta_1_samples,
         sigma=sigma_samples,
         nu=nu_samples,
         y_pred=y_pred_all,
         x=x,
         y_obs=y_obs,
         log_x=log_x)

print("\nPrior predictive check complete!")
print(f"Proceed to visualization script to generate diagnostic plots.")
