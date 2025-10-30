"""
Simulation-Based Calibration (SBC) for Fixed-Effect Meta-Analysis Model

This script performs comprehensive SBC to validate that the model can recover
known parameters through simulation-based calibration.

Uses analytical posterior (conjugate Normal-Normal model) for fast, exact inference.

Model:
    y_i | θ, σ_i ~ Normal(θ, σ_i²)
    θ ~ Normal(0, 20²)

Analytical Posterior (conjugate):
    θ | y ~ Normal(μ_post, σ_post²)
    where:
        τ_prior² = 20²
        τ_post² = 1 / (1/τ_prior² + Σ(1/σ_i²))
        μ_post = τ_post² * Σ(y_i/σ_i²)

Validation Criteria:
- Rank statistics should be uniform
- Coverage should match nominal rates
- No systematic bias in recovery
- Well-calibrated uncertainty
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed
np.random.seed(42)

# Configuration
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
PLOTS_DIR = OUTPUT_DIR / "plots"
CODE_DIR = OUTPUT_DIR / "code"

# SBC parameters
N_SIMS = 500  # Number of SBC simulations
N_POSTERIOR_SAMPLES = 2000  # Posterior samples per simulation

# Model parameters
PRIOR_MEAN = 0
PRIOR_SD = 20
SIGMA = np.array([15, 10, 16, 11, 9, 11, 10, 18])
N_OBS = len(SIGMA)

print("="*80)
print("SIMULATION-BASED CALIBRATION: Fixed-Effect Meta-Analysis")
print("="*80)
print(f"\nConfiguration:")
print(f"  Number of simulations: {N_SIMS}")
print(f"  Posterior samples: {N_POSTERIOR_SAMPLES}")
print(f"  Known σ: {SIGMA}")
print(f"\nModel:")
print(f"  Prior: θ ~ N({PRIOR_MEAN}, {PRIOR_SD}²)")
print(f"  Likelihood: y_i | θ ~ N(θ, σ_i²)")
print(f"\nUsing analytical conjugate posterior (exact inference)")

# ============================================================================
# ANALYTICAL POSTERIOR FUNCTIONS
# ============================================================================

def compute_posterior(y_obs, sigma, prior_mean=0, prior_sd=20):
    """
    Compute analytical posterior for Normal-Normal conjugate model.

    Returns: (posterior_mean, posterior_sd)
    """
    # Prior precision
    prior_precision = 1 / (prior_sd ** 2)

    # Likelihood precisions
    likelihood_precisions = 1 / (sigma ** 2)

    # Posterior precision
    posterior_precision = prior_precision + np.sum(likelihood_precisions)
    posterior_sd = np.sqrt(1 / posterior_precision)

    # Posterior mean
    prior_contrib = prior_precision * prior_mean
    likelihood_contrib = np.sum(y_obs * likelihood_precisions)
    posterior_mean = (prior_contrib + likelihood_contrib) / posterior_precision

    return posterior_mean, posterior_sd

def sample_posterior(y_obs, sigma, n_samples=2000, prior_mean=0, prior_sd=20):
    """Sample from analytical posterior."""
    post_mean, post_sd = compute_posterior(y_obs, sigma, prior_mean, prior_sd)
    samples = np.random.normal(post_mean, post_sd, size=n_samples)
    return samples

def compute_rank_statistic(theta_true, posterior_samples):
    """Compute rank of true value in posterior samples."""
    rank = np.sum(posterior_samples < theta_true)
    return rank

# ============================================================================
# RUN SBC SIMULATIONS
# ============================================================================

print("\n" + "="*80)
print("RUNNING SBC SIMULATIONS")
print("="*80)

results = []

for sim_idx in tqdm(range(N_SIMS), desc="SBC Simulations"):
    # Step 1: Draw true parameter from prior
    theta_true = np.random.normal(PRIOR_MEAN, PRIOR_SD)

    # Step 2: Generate synthetic data
    y_synthetic = np.random.normal(theta_true, SIGMA)

    # Step 3: Compute analytical posterior
    post_mean, post_sd = compute_posterior(y_synthetic, SIGMA, PRIOR_MEAN, PRIOR_SD)

    # Step 4: Sample from posterior
    posterior_samples = np.random.normal(post_mean, post_sd, size=N_POSTERIOR_SAMPLES)

    # Step 5: Compute diagnostics
    rank = compute_rank_statistic(theta_true, posterior_samples)

    # Basic statistics
    theta_mean = np.mean(posterior_samples)
    theta_median = np.median(posterior_samples)
    theta_sd = np.std(posterior_samples)

    # Credible intervals
    ci_50 = np.percentile(posterior_samples, [25, 75])
    ci_90 = np.percentile(posterior_samples, [5, 95])
    ci_95 = np.percentile(posterior_samples, [2.5, 97.5])

    # Coverage
    coverage_50 = ci_50[0] <= theta_true <= ci_50[1]
    coverage_90 = ci_90[0] <= theta_true <= ci_90[1]
    coverage_95 = ci_95[0] <= theta_true <= ci_95[1]

    # Z-score
    z_score = (theta_true - theta_mean) / theta_sd

    diagnostics = {
        "sim_idx": sim_idx,
        "theta_true": theta_true,
        "theta_mean": theta_mean,
        "theta_median": theta_median,
        "theta_sd": theta_sd,
        "post_sd_analytical": post_sd,  # Should match theta_sd closely
        "bias": theta_mean - theta_true,
        "ci_50_lower": ci_50[0],
        "ci_50_upper": ci_50[1],
        "ci_90_lower": ci_90[0],
        "ci_90_upper": ci_90[1],
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "coverage_50": coverage_50,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "rank": rank,
        "z_score": z_score,
    }

    results.append(diagnostics)

print(f"\nCompleted {len(results)}/{N_SIMS} simulations successfully")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(CODE_DIR / "sbc_results.csv", index=False)
print(f"\nResults saved to: {CODE_DIR / 'sbc_results.csv'}")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SBC ANALYSIS")
print("="*80)

# 1. RANK STATISTICS
print("\n1. RANK STATISTICS")
print("-" * 80)

ranks = results_df["rank"].values
n_bins = 20  # Standard for SBC

# Expected uniform distribution
expected_count_per_bin = N_SIMS / n_bins
bin_edges = np.linspace(0, N_POSTERIOR_SAMPLES, n_bins + 1)
observed_counts, _ = np.histogram(ranks, bins=bin_edges)

# Chi-square test for uniformity
chi2_stat = np.sum((observed_counts - expected_count_per_bin)**2 / expected_count_per_bin)
chi2_pval = 1 - stats.chi2.cdf(chi2_stat, df=n_bins - 1)

print(f"Number of posterior samples: {N_POSTERIOR_SAMPLES}")
print(f"Number of bins: {n_bins}")
print(f"Expected count per bin: {expected_count_per_bin:.1f}")
print(f"Observed counts range: [{observed_counts.min()}, {observed_counts.max()}]")
print(f"\nChi-square test for uniformity:")
print(f"  χ² statistic: {chi2_stat:.2f}")
print(f"  p-value: {chi2_pval:.4f}")
print(f"  Result: {'PASS' if chi2_pval > 0.05 else 'FAIL'} (threshold: p > 0.05)")

# ECDF-based uniformity test
ks_result = stats.kstest(ranks / N_POSTERIOR_SAMPLES, 'uniform')
ks_stat = ks_result.statistic
ks_pval = ks_result.pvalue

print(f"\nKolmogorov-Smirnov test:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pval:.4f}")
print(f"  Result: {'PASS' if ks_pval > 0.05 else 'FAIL'} (threshold: p > 0.05)")

# 2. COVERAGE CALIBRATION
print("\n2. COVERAGE CALIBRATION")
print("-" * 80)

coverage_50 = results_df["coverage_50"].mean() * 100
coverage_90 = results_df["coverage_90"].mean() * 100
coverage_95 = results_df["coverage_95"].mean() * 100

print(f"50% Credible Interval Coverage: {coverage_50:.1f}% (nominal: 50%)")
print(f"90% Credible Interval Coverage: {coverage_90:.1f}% (nominal: 90%)")
print(f"95% Credible Interval Coverage: {coverage_95:.1f}% (nominal: 95%)")

coverage_50_pass = abs(coverage_50 - 50) < 5
coverage_90_pass = abs(coverage_90 - 90) < 5
coverage_95_pass = abs(coverage_95 - 95) < 5

print(f"\nCoverage assessment:")
print(f"  50% CI: {'PASS' if coverage_50_pass else 'FAIL'} (within ±5%)")
print(f"  90% CI: {'PASS' if coverage_90_pass else 'FAIL'} (within ±5%)")
print(f"  95% CI: {'PASS' if coverage_95_pass else 'FAIL'} (within ±5%)")

# Binomial confidence intervals for coverage
from scipy.stats import binom
n = N_SIMS
for level, obs_cov in [(50, coverage_50), (90, coverage_90), (95, coverage_95)]:
    n_covered = int(obs_cov * n / 100)
    ci_lower = binom.ppf(0.025, n, level/100) / n * 100
    ci_upper = binom.ppf(0.975, n, level/100) / n * 100
    print(f"  {level}% CI: 95% binomial CI for {level}% coverage: [{ci_lower:.1f}%, {ci_upper:.1f}%]")

# 3. BIAS AND SHRINKAGE
print("\n3. BIAS AND SHRINKAGE ANALYSIS")
print("-" * 80)

bias = results_df["bias"].mean()
bias_sd = results_df["bias"].std()
bias_se = bias_sd / np.sqrt(N_SIMS)

print(f"Mean bias (θ̂ - θ_true): {bias:.4f} ± {bias_se:.4f}")
print(f"Bias SD: {bias_sd:.4f}")
print(f"Bias range: [{results_df['bias'].min():.2f}, {results_df['bias'].max():.2f}]")

# Test if bias is significantly different from zero
t_stat = bias / bias_se
t_pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=N_SIMS - 1))
print(f"\nBias t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {t_pval:.4f}")
print(f"  Result: {'PASS' if abs(bias) < 0.5 else 'FAIL'} (|bias| < 0.5)")

# Correlation between true and estimated
theta_true = results_df["theta_true"].values
theta_hat = results_df["theta_mean"].values
correlation = np.corrcoef(theta_true, theta_hat)[0, 1]
r_squared = correlation**2

print(f"\nParameter recovery:")
print(f"  Correlation (θ_true, θ̂): {correlation:.4f}")
print(f"  R²: {r_squared:.4f}")
print(f"  Result: {'PASS' if r_squared > 0.95 else 'FAIL'} (R² > 0.95)")

# Linear regression to check for slope != 1 (shrinkage)
slope, intercept = np.polyfit(theta_true, theta_hat, 1)
print(f"\nLinear regression (θ̂ = a + b·θ_true):")
print(f"  Intercept: {intercept:.4f}")
print(f"  Slope: {slope:.4f}")
print(f"  Result: {'PASS' if 0.95 < slope < 1.05 else 'FAIL'} (0.95 < slope < 1.05)")

# 4. UNCERTAINTY CALIBRATION
print("\n4. UNCERTAINTY CALIBRATION")
print("-" * 80)

# Posterior SD vs empirical SD
posterior_sd = results_df["theta_sd"].mean()
empirical_sd = results_df["bias"].std()
sd_ratio = posterior_sd / empirical_sd

print(f"Mean posterior SD: {posterior_sd:.4f}")
print(f"Empirical SD of errors: {empirical_sd:.4f}")
print(f"Ratio (posterior/empirical): {sd_ratio:.4f}")
print(f"Result: {'PASS' if 0.9 < sd_ratio < 1.1 else 'FAIL'} (0.9 < ratio < 1.1)")

# Check that sampled SD matches analytical SD
analytical_sd = results_df["post_sd_analytical"].mean()
sd_difference = abs(posterior_sd - analytical_sd)
print(f"\nSampling accuracy:")
print(f"  Mean sampled SD: {posterior_sd:.4f}")
print(f"  Mean analytical SD: {analytical_sd:.4f}")
print(f"  Difference: {sd_difference:.6f}")
print(f"  Result: {'PASS' if sd_difference < 0.01 else 'FAIL'} (difference < 0.01)")

# Z-score distribution
z_scores = results_df["z_score"].values
z_mean = np.mean(z_scores)
z_sd = np.std(z_scores)

print(f"\nZ-score distribution:")
print(f"  Mean: {z_mean:.4f} (expected: 0)")
print(f"  SD: {z_sd:.4f} (expected: 1)")

# Test normality
shapiro_stat, shapiro_pval = stats.shapiro(z_scores)
print(f"\nShapiro-Wilk test for normality:")
print(f"  W statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_pval:.4f}")
print(f"  Result: {'PASS' if shapiro_pval > 0.05 else 'FAIL'} (p > 0.05)")

# Check if z-scores are approximately N(0,1)
z_mean_pass = abs(z_mean) < 0.1
z_sd_pass = 0.9 < z_sd < 1.1

print(f"\nZ-score calibration:")
print(f"  Mean check: {'PASS' if z_mean_pass else 'FAIL'} (|mean| < 0.1)")
print(f"  SD check: {'PASS' if z_sd_pass else 'FAIL'} (0.9 < SD < 1.1)")

# 5. STRATIFIED ANALYSIS
print("\n5. STRATIFIED ANALYSIS BY PARAMETER RANGE")
print("-" * 80)

# Stratify by magnitude of true parameter
results_df["magnitude_category"] = pd.cut(
    np.abs(results_df["theta_true"]),
    bins=[0, 5, 15, np.inf],
    labels=["Small (|θ| < 5)", "Medium (5 ≤ |θ| < 15)", "Large (|θ| ≥ 15)"]
)

for category in ["Small (|θ| < 5)", "Medium (5 ≤ |θ| < 15)", "Large (|θ| ≥ 15)"]:
    subset = results_df[results_df["magnitude_category"] == category]
    if len(subset) == 0:
        continue

    print(f"\n{category}: n = {len(subset)}")
    print(f"  Mean bias: {subset['bias'].mean():.4f}")
    print(f"  Coverage (50%): {subset['coverage_50'].mean() * 100:.1f}%")
    print(f"  Coverage (90%): {subset['coverage_90'].mean() * 100:.1f}%")
    print(f"  Coverage (95%): {subset['coverage_95'].mean() * 100:.1f}%")
    print(f"  Mean posterior SD: {subset['theta_sd'].mean():.4f}")

# ============================================================================
# OVERALL ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("OVERALL SBC ASSESSMENT")
print("="*80)

# Collect all pass/fail results
checks = {
    "Rank uniformity (χ²)": chi2_pval > 0.05,
    "Rank uniformity (KS)": ks_pval > 0.05,
    "Coverage (50%)": coverage_50_pass,
    "Coverage (90%)": coverage_90_pass,
    "Coverage (95%)": coverage_95_pass,
    "Low bias": abs(bias) < 0.5,
    "High correlation": r_squared > 0.95,
    "No shrinkage": 0.95 < slope < 1.05,
    "SD calibration": 0.9 < sd_ratio < 1.1,
    "Sampling accuracy": sd_difference < 0.01,
    "Z-score mean": z_mean_pass,
    "Z-score SD": z_sd_pass,
    "Z-score normality": shapiro_pval > 0.05,
}

n_passed = sum(checks.values())
n_total = len(checks)

print(f"\nValidation checks: {n_passed}/{n_total} passed")
print("\nDetailed results:")
for check_name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check_name}")

# Final decision
all_critical_passed = (
    chi2_pval > 0.05 and
    coverage_95_pass and
    abs(bias) < 0.5 and
    r_squared > 0.95 and
    0.9 < sd_ratio < 1.1
)

print("\n" + "="*80)
if all_critical_passed:
    print("OVERALL RESULT: PASS")
    print("\nThe model successfully recovers known parameters!")
    print("Inference machinery is validated and ready for real data.")
    print("\nNOTE: Using analytical conjugate posterior provides exact inference.")
else:
    print("OVERALL RESULT: FAIL")
    print("\nThe model fails to recover known parameters.")
    print("DO NOT proceed to real data fitting. Investigate issues.")

print("="*80)

# Save summary
summary = {
    "n_simulations": N_SIMS,
    "n_successful": len(results),
    "method": "analytical_conjugate_posterior",
    "rank_statistics": {
        "chi2_stat": float(chi2_stat),
        "chi2_pval": float(chi2_pval),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval)
    },
    "coverage": {
        "coverage_50": float(coverage_50),
        "coverage_90": float(coverage_90),
        "coverage_95": float(coverage_95)
    },
    "bias": {
        "mean_bias": float(bias),
        "bias_sd": float(bias_sd),
        "t_pval": float(t_pval)
    },
    "correlation": {
        "correlation": float(correlation),
        "r_squared": float(r_squared),
        "slope": float(slope),
        "intercept": float(intercept)
    },
    "uncertainty": {
        "posterior_sd": float(posterior_sd),
        "empirical_sd": float(empirical_sd),
        "analytical_sd": float(analytical_sd),
        "sd_ratio": float(sd_ratio),
        "sd_difference": float(sd_difference),
        "z_mean": float(z_mean),
        "z_sd": float(z_sd),
        "shapiro_pval": float(shapiro_pval)
    },
    "checks": checks,
    "overall_pass": all_critical_passed
}

with open(CODE_DIR / "sbc_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {CODE_DIR / 'sbc_summary.json'}")
print(f"\nNext step: Run generate_sbc_plots.py to create visualizations")
