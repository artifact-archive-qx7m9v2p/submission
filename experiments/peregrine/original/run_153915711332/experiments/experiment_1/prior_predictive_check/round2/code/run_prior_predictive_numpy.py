"""
Prior Predictive Check - Round 2 (ADJUSTED PRIORS)
Negative Binomial State-Space Model

UPDATED PRIORS:
- delta ~ Normal(0.05, 0.02)     [KEPT - was working well]
- sigma_eta ~ Exponential(20)    [CHANGED from Exp(10) - tighten to mean=0.05]
- phi ~ Exponential(0.05)        [CHANGED from Exp(0.1) - tighten to mean=20]
- eta_1 ~ Normal(log(50), 1)     [KEPT]
"""

import numpy as np
from scipy import stats
import json

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SAMPLES = 1000
N_TIME = 40
INITIAL_STATE_MEAN = np.log(50)  # log(50) ≈ 3.91
INITIAL_STATE_SD = 1.0

# ADJUSTED PRIORS
DELTA_MEAN = 0.05
DELTA_SD = 0.02
SIGMA_ETA_RATE = 20.0  # CHANGED: Mean = 1/20 = 0.05 (was 0.1)
PHI_RATE = 0.05        # CHANGED: Mean = 1/0.05 = 20 (was 10)

print("=" * 80)
print("PRIOR PREDICTIVE CHECK - ROUND 2 (ADJUSTED PRIORS)")
print("=" * 80)
print("\nPrior Specification:")
print(f"  delta ~ Normal({DELTA_MEAN}, {DELTA_SD})")
print(f"  sigma_eta ~ Exponential({SIGMA_ETA_RATE}) [mean = {1/SIGMA_ETA_RATE:.3f}]")
print(f"  phi ~ Exponential({PHI_RATE}) [mean = {1/PHI_RATE:.1f}]")
print(f"  eta_1 ~ Normal({INITIAL_STATE_MEAN:.2f}, {INITIAL_STATE_SD})")
print(f"\nGenerating {N_SAMPLES} prior predictive samples...")
print("-" * 80)

# Initialize storage
delta_samples = np.zeros(N_SAMPLES)
sigma_eta_samples = np.zeros(N_SAMPLES)
phi_samples = np.zeros(N_SAMPLES)
eta_samples = np.zeros((N_SAMPLES, N_TIME))
C_prior_samples = np.zeros((N_SAMPLES, N_TIME), dtype=int)

# Summary statistics
prior_mean_counts = np.zeros(N_SAMPLES)
prior_max_counts = np.zeros(N_SAMPLES)
prior_min_counts = np.zeros(N_SAMPLES)
prior_growth_factors = np.zeros(N_SAMPLES)
total_log_changes = np.zeros(N_SAMPLES)

# Generate prior predictive samples
for i in range(N_SAMPLES):
    if (i + 1) % 100 == 0:
        print(f"  Sample {i + 1}/{N_SAMPLES}...")

    # Sample parameters from ADJUSTED priors
    delta = np.random.normal(DELTA_MEAN, DELTA_SD)
    sigma_eta = np.random.exponential(1 / SIGMA_ETA_RATE)  # ADJUSTED
    phi = np.random.exponential(1 / PHI_RATE)  # ADJUSTED

    # Sample initial state
    eta_1 = np.random.normal(INITIAL_STATE_MEAN, INITIAL_STATE_SD)

    # Generate latent state trajectory (random walk with drift)
    eta = np.zeros(N_TIME)
    eta[0] = eta_1

    for t in range(1, N_TIME):
        eta[t] = np.random.normal(eta[t-1] + delta, sigma_eta)

    # Generate prior predictive counts
    # NegBin parameterization: mean = exp(eta), dispersion = phi
    # scipy uses n, p parameterization where n=phi, p=phi/(phi+mu)
    C_prior = np.zeros(N_TIME, dtype=int)
    for t in range(N_TIME):
        mu = np.exp(eta[t])
        # Convert to scipy parameterization
        n = phi
        p = phi / (phi + mu)

        # Handle numerical issues
        if p <= 0 or p >= 1 or n <= 0:
            # Fallback to Poisson if parameters are degenerate
            C_prior[t] = np.random.poisson(mu)
        else:
            C_prior[t] = np.random.negative_binomial(n, p)

    # Store samples
    delta_samples[i] = delta
    sigma_eta_samples[i] = sigma_eta
    phi_samples[i] = phi
    eta_samples[i, :] = eta
    C_prior_samples[i, :] = C_prior

    # Compute summary statistics
    prior_mean_counts[i] = np.mean(C_prior)
    prior_max_counts[i] = np.max(C_prior)
    prior_min_counts[i] = np.min(C_prior)
    prior_growth_factors[i] = np.exp(eta[-1] - eta[0])
    total_log_changes[i] = eta[-1] - eta[0]

print("\nSampling complete!")
print("=" * 80)

# Compute diagnostics
print("\nPRIOR PARAMETER SUMMARIES (Round 2):")
print("-" * 80)
print(f"Delta:")
print(f"  Mean: {np.mean(delta_samples):.4f}")
print(f"  SD: {np.std(delta_samples):.4f}")
print(f"  95% CI: [{np.percentile(delta_samples, 2.5):.4f}, {np.percentile(delta_samples, 97.5):.4f}]")

print(f"\nSigma_eta (ADJUSTED):")
print(f"  Mean: {np.mean(sigma_eta_samples):.4f}")
print(f"  Median: {np.median(sigma_eta_samples):.4f}")
print(f"  95% CI: [{np.percentile(sigma_eta_samples, 2.5):.4f}, {np.percentile(sigma_eta_samples, 97.5):.4f}]")
print(f"  Max: {np.max(sigma_eta_samples):.4f}")

print(f"\nPhi (ADJUSTED):")
print(f"  Mean: {np.mean(phi_samples):.4f}")
print(f"  Median: {np.median(phi_samples):.4f}")
print(f"  95% CI: [{np.percentile(phi_samples, 2.5):.4f}, {np.percentile(phi_samples, 97.5):.4f}]")
print(f"  Max: {np.max(phi_samples):.4f}")

print("\n" + "=" * 80)
print("PRIOR PREDICTIVE SUMMARIES (Round 2):")
print("-" * 80)

print(f"\nMean Counts:")
print(f"  Prior mean: {np.mean(prior_mean_counts):.2f}")
print(f"  Prior median: {np.median(prior_mean_counts):.2f}")
print(f"  95% CI: [{np.percentile(prior_mean_counts, 2.5):.2f}, {np.percentile(prior_mean_counts, 97.5):.2f}]")
print(f"  Observed: 109.45")

print(f"\nMaximum Counts:")
print(f"  Prior median: {np.median(prior_max_counts):.2f}")
print(f"  95% CI: [{np.percentile(prior_max_counts, 2.5):.2f}, {np.percentile(prior_max_counts, 97.5):.2f}]")
print(f"  Extreme max: {np.max(C_prior_samples):.0f}")
print(f"  Observed: 272")

print(f"\nGrowth Factors (exp(eta_40) / exp(eta_1)):")
print(f"  Prior median: {np.median(prior_growth_factors):.2f}x")
print(f"  95% CI: [{np.percentile(prior_growth_factors, 2.5):.2f}x, {np.percentile(prior_growth_factors, 97.5):.2f}x]")
print(f"  Observed: 8.45x")

print(f"\nTotal Log Change (eta_40 - eta_1):")
print(f"  Prior mean: {np.mean(total_log_changes):.2f}")
print(f"  95% CI: [{np.percentile(total_log_changes, 2.5):.2f}, {np.percentile(total_log_changes, 97.5):.2f}]")
print(f"  Observed: {np.log(245) - np.log(29):.2f}")

print("\n" + "=" * 80)
print("COMPUTATIONAL RED FLAGS (Round 2):")
print("-" * 80)

n_extreme_1000 = np.sum(C_prior_samples > 1000)
n_extreme_10000 = np.sum(C_prior_samples > 10000)
pct_extreme_1000 = 100 * n_extreme_1000 / (N_SAMPLES * N_TIME)
pct_extreme_10000 = 100 * n_extreme_10000 / (N_SAMPLES * N_TIME)

print(f"\nExtreme count frequency:")
print(f"  Counts > 1,000: {n_extreme_1000:,} / {N_SAMPLES * N_TIME:,} ({pct_extreme_1000:.3f}%)")
print(f"  Counts > 10,000: {n_extreme_10000:,} / {N_SAMPLES * N_TIME:,} ({pct_extreme_10000:.4f}%)")

n_growth_50 = np.sum(prior_growth_factors > 50)
n_growth_100 = np.sum(prior_growth_factors > 100)
pct_growth_50 = 100 * n_growth_50 / N_SAMPLES
pct_growth_100 = 100 * n_growth_100 / N_SAMPLES

print(f"\nExtreme growth frequency:")
print(f"  Growth > 50x: {n_growth_50} / {N_SAMPLES} ({pct_growth_50:.2f}%)")
print(f"  Growth > 100x: {n_growth_100} / {N_SAMPLES} ({pct_growth_100:.2f}%)")

n_low_phi = np.sum(phi_samples < 0.1)
pct_low_phi = 100 * n_low_phi / N_SAMPLES
print(f"\nNumerical issues:")
print(f"  Phi < 0.1 (near-zero): {n_low_phi} / {N_SAMPLES} ({pct_low_phi:.2f}%)")

print("\n" + "=" * 80)
print("COVERAGE DIAGNOSTICS (Round 2):")
print("-" * 80)

# Where does observed data fall in prior predictive distribution?
obs_mean = 109.45
obs_max = 272

pct_below_obs_mean = 100 * np.sum(prior_mean_counts < obs_mean) / N_SAMPLES
pct_below_obs_max = 100 * np.sum(prior_max_counts < obs_max) / N_SAMPLES

print(f"\nObserved mean (109.45) percentile: {pct_below_obs_mean:.1f}%")
print(f"Observed max (272) percentile: {pct_below_obs_max:.1f}%")

if 25 < pct_below_obs_mean < 75:
    print("  -> GOOD: Observed mean is in central region of prior predictive")
elif pct_below_obs_mean < 25:
    print("  -> WARNING: Observed mean is in lower tail (priors may be too high)")
else:
    print("  -> WARNING: Observed mean is in upper tail (priors may be too low)")

if 25 < pct_below_obs_max < 75:
    print("  -> GOOD: Observed max is in central region of prior predictive")
elif pct_below_obs_max < 25:
    print("  -> WARNING: Observed max is in lower tail (priors may be too high)")
else:
    print("  -> WARNING: Observed max is in upper tail (priors may be too low)")

print("\n" + "=" * 80)
print("COMPARISON TO ROUND 1:")
print("-" * 80)
print("\nRound 1 (FAILED - Too Diffuse):")
print("  Prior mean of means: 418.8 (observed: 109.5)")
print("  Extreme counts (>10k): 0.40%")
print("  Max value 95% CI upper: 11,610")
print("  Extreme max: 175,837")

print(f"\nRound 2 (ADJUSTED):")
print(f"  Prior mean of means: {np.mean(prior_mean_counts):.1f} (observed: 109.5)")
print(f"  Extreme counts (>10k): {pct_extreme_10000:.4f}%")
print(f"  Max value 95% CI upper: {np.percentile(prior_max_counts, 97.5):.0f}")
print(f"  Extreme max: {np.max(C_prior_samples):.0f}")

# Improvement metrics
improvement_mean = (418.8 - np.mean(prior_mean_counts)) / 418.8 * 100
improvement_extreme = (0.40 - pct_extreme_10000) / 0.40 * 100

print(f"\nImprovements:")
print(f"  Mean closer to observed: {improvement_mean:.1f}% improvement")
print(f"  Reduction in extreme counts: {improvement_extreme:.1f}% reduction")

print("\n" + "=" * 80)

# Save results
output_file = '/workspace/experiments/experiment_1/prior_predictive_check/round2/code/prior_samples.npz'
np.savez(output_file,
         delta=delta_samples,
         sigma_eta=sigma_eta_samples,
         phi=phi_samples,
         eta=eta_samples,
         C_prior=C_prior_samples,
         prior_mean_counts=prior_mean_counts,
         prior_max_counts=prior_max_counts,
         prior_min_counts=prior_min_counts,
         prior_growth_factors=prior_growth_factors,
         total_log_changes=total_log_changes)

print(f"\nSamples saved to: {output_file}")

# Save summary statistics
summary = {
    'round': 2,
    'n_samples': int(N_SAMPLES),
    'n_time': int(N_TIME),
    'priors': {
        'delta': f'Normal({DELTA_MEAN}, {DELTA_SD})',
        'sigma_eta': f'Exponential({SIGMA_ETA_RATE}) [mean={1/SIGMA_ETA_RATE:.3f}]',
        'phi': f'Exponential({PHI_RATE}) [mean={1/PHI_RATE:.1f}]',
        'eta_1': f'Normal({INITIAL_STATE_MEAN:.2f}, {INITIAL_STATE_SD})'
    },
    'parameter_summaries': {
        'delta': {
            'mean': float(np.mean(delta_samples)),
            'sd': float(np.std(delta_samples)),
            'ci_95': [float(np.percentile(delta_samples, 2.5)),
                      float(np.percentile(delta_samples, 97.5))]
        },
        'sigma_eta': {
            'mean': float(np.mean(sigma_eta_samples)),
            'median': float(np.median(sigma_eta_samples)),
            'ci_95': [float(np.percentile(sigma_eta_samples, 2.5)),
                      float(np.percentile(sigma_eta_samples, 97.5))],
            'max': float(np.max(sigma_eta_samples))
        },
        'phi': {
            'mean': float(np.mean(phi_samples)),
            'median': float(np.median(phi_samples)),
            'ci_95': [float(np.percentile(phi_samples, 2.5)),
                      float(np.percentile(phi_samples, 97.5))],
            'max': float(np.max(phi_samples))
        }
    },
    'prior_predictive_summaries': {
        'mean_counts': {
            'mean': float(np.mean(prior_mean_counts)),
            'median': float(np.median(prior_mean_counts)),
            'ci_95': [float(np.percentile(prior_mean_counts, 2.5)),
                      float(np.percentile(prior_mean_counts, 97.5))],
            'observed': 109.45
        },
        'max_counts': {
            'median': float(np.median(prior_max_counts)),
            'ci_95': [float(np.percentile(prior_max_counts, 2.5)),
                      float(np.percentile(prior_max_counts, 97.5))],
            'extreme_max': float(np.max(C_prior_samples)),
            'observed': 272
        },
        'growth_factors': {
            'median': float(np.median(prior_growth_factors)),
            'ci_95': [float(np.percentile(prior_growth_factors, 2.5)),
                      float(np.percentile(prior_growth_factors, 97.5))],
            'observed': 8.45
        }
    },
    'red_flags': {
        'counts_gt_1000': {
            'count': int(n_extreme_1000),
            'percent': float(pct_extreme_1000)
        },
        'counts_gt_10000': {
            'count': int(n_extreme_10000),
            'percent': float(pct_extreme_10000)
        },
        'growth_gt_50x': {
            'count': int(n_growth_50),
            'percent': float(pct_growth_50)
        },
        'growth_gt_100x': {
            'count': int(n_growth_100),
            'percent': float(pct_growth_100)
        },
        'phi_lt_0.1': {
            'count': int(n_low_phi),
            'percent': float(pct_low_phi)
        }
    },
    'coverage': {
        'obs_mean_percentile': float(pct_below_obs_mean),
        'obs_max_percentile': float(pct_below_obs_max)
    },
    'comparison_to_round1': {
        'round1_mean_of_means': 418.8,
        'round2_mean_of_means': float(np.mean(prior_mean_counts)),
        'improvement_pct': float(improvement_mean),
        'round1_extreme_pct': 0.40,
        'round2_extreme_pct': float(pct_extreme_10000),
        'reduction_pct': float(improvement_extreme)
    }
}

summary_file = '/workspace/experiments/experiment_1/prior_predictive_check/round2/code/prior_predictive_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary statistics saved to: {summary_file}")
print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
