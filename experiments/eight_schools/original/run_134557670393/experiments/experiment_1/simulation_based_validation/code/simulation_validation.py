"""
Simulation-Based Calibration for Bayesian Hierarchical Meta-Analysis

This script validates that the model can recover known parameters through:
1. Fixed-effect scenario (tau = 0)
2. Random-effects scenario (tau = 5)
3. Full SBC with multiple simulations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from cmdstanpy import CmdStanModel
from scipy import stats
import json

# Set random seed for reproducibility
np.random.seed(42)

# Paths
CODE_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/code')
PLOTS_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/plots')
PLOTS_DIR.mkdir(exist_ok=True)

# Known standard errors (from data)
SIGMA = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = len(SIGMA)

print("=" * 80)
print("SIMULATION-BASED CALIBRATION: HIERARCHICAL META-ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# PART 1: TEST FIXED-EFFECT SCENARIO (tau = 0)
# ============================================================================

print("PART 1: FIXED-EFFECT SCENARIO (tau = 0)")
print("-" * 80)
print("Testing if model can detect homogeneous effects (no between-study variation)")
print()

# True parameters
mu_true_fixed = 10.0
tau_true_fixed = 0.0  # Homogeneous effects

# Generate study-specific effects (all equal when tau = 0)
theta_true_fixed = np.repeat(mu_true_fixed, J)

# Generate observed data
np.random.seed(123)
y_fixed = theta_true_fixed + SIGMA * np.random.randn(J)

print(f"True parameters: mu = {mu_true_fixed}, tau = {tau_true_fixed}")
print(f"Generated data: {y_fixed}")
print()

# Prepare data for Stan
stan_data_fixed = {
    'J': J,
    'y': y_fixed.tolist(),
    'sigma': SIGMA.tolist()
}

# Compile and fit model (try centered first)
print("Compiling Stan model (centered parameterization)...")
model = CmdStanModel(stan_file=str(CODE_DIR / 'hierarchical_meta_analysis.stan'))

print("Fitting model to fixed-effect data...")
fit_fixed = model.sample(
    data=stan_data_fixed,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    seed=42,
    show_progress=True
)

print("\nConvergence diagnostics:")
print(fit_fixed.diagnose())

# Extract posteriors
mu_posterior_fixed = fit_fixed.stan_variable('mu')
tau_posterior_fixed = fit_fixed.stan_variable('tau')
theta_posterior_fixed = fit_fixed.stan_variable('theta')

# Summary statistics
summary_fixed = fit_fixed.summary()
print("\nParameter estimates:")
print(summary_fixed[['Mean', 'StdDev', '5%', '50%', '95%', 'R_hat', 'ESS_bulk']])

# Check recovery
mu_mean = mu_posterior_fixed.mean()
tau_mean = tau_posterior_fixed.mean()
tau_95 = np.percentile(tau_posterior_fixed, 95)

print(f"\nRecovery assessment:")
print(f"  True mu = {mu_true_fixed:.2f}, Posterior mean = {mu_mean:.2f}")
print(f"  True tau = {tau_true_fixed:.2f}, Posterior mean = {tau_mean:.2f}")
print(f"  P(tau < 2 | data) = {(tau_posterior_fixed < 2).mean():.3f}")
print(f"  95th percentile of tau = {tau_95:.2f}")

fixed_results = {
    'mu_true': mu_true_fixed,
    'tau_true': tau_true_fixed,
    'mu_mean': mu_mean,
    'tau_mean': tau_mean,
    'mu_in_95ci': (mu_true_fixed >= np.percentile(mu_posterior_fixed, 2.5)) and
                   (mu_true_fixed <= np.percentile(mu_posterior_fixed, 97.5)),
    'tau_in_95ci': (tau_true_fixed <= np.percentile(tau_posterior_fixed, 95)),
    'convergence_pass': (summary_fixed['R_hat'] < 1.01).all()
}

print(f"\nFixed-effect test results:")
print(f"  mu recovered in 95% CI: {fixed_results['mu_in_95ci']}")
print(f"  tau near zero (appropriate): {tau_mean < 2}")
print(f"  All R-hat < 1.01: {fixed_results['convergence_pass']}")
print()

# ============================================================================
# PART 2: TEST RANDOM-EFFECTS SCENARIO (tau = 5)
# ============================================================================

print("\nPART 2: RANDOM-EFFECTS SCENARIO (tau = 5)")
print("-" * 80)
print("Testing if model can detect heterogeneous effects (between-study variation)")
print()

# True parameters
mu_true_random = 10.0
tau_true_random = 5.0  # Moderate heterogeneity

# Generate study-specific effects (vary around mu)
np.random.seed(456)
theta_true_random = mu_true_random + tau_true_random * np.random.randn(J)

# Generate observed data
y_random = theta_true_random + SIGMA * np.random.randn(J)

print(f"True parameters: mu = {mu_true_random}, tau = {tau_true_random}")
print(f"True theta: {theta_true_random}")
print(f"Generated data: {y_random}")
print()

# Prepare data for Stan
stan_data_random = {
    'J': J,
    'y': y_random.tolist(),
    'sigma': SIGMA.tolist()
}

print("Fitting model to random-effects data...")
fit_random = model.sample(
    data=stan_data_random,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    seed=43,
    show_progress=True
)

print("\nConvergence diagnostics:")
print(fit_random.diagnose())

# Extract posteriors
mu_posterior_random = fit_random.stan_variable('mu')
tau_posterior_random = fit_random.stan_variable('tau')
theta_posterior_random = fit_random.stan_variable('theta')

# Summary statistics
summary_random = fit_random.summary()
print("\nParameter estimates:")
print(summary_random[['Mean', 'StdDev', '5%', '50%', '95%', 'R_hat', 'ESS_bulk']])

# Check recovery
mu_mean_r = mu_posterior_random.mean()
tau_mean_r = tau_posterior_random.mean()
theta_mean_r = theta_posterior_random.mean(axis=0)

print(f"\nRecovery assessment:")
print(f"  True mu = {mu_true_random:.2f}, Posterior mean = {mu_mean_r:.2f}")
print(f"  True tau = {tau_true_random:.2f}, Posterior mean = {tau_mean_r:.2f}")

# Check if true values in credible intervals
mu_in_ci = (mu_true_random >= np.percentile(mu_posterior_random, 2.5)) and \
           (mu_true_random <= np.percentile(mu_posterior_random, 97.5))
tau_in_ci = (tau_true_random >= np.percentile(tau_posterior_random, 2.5)) and \
            (tau_true_random <= np.percentile(tau_posterior_random, 97.5))

theta_in_ci = np.array([
    (theta_true_random[j] >= np.percentile(theta_posterior_random[:, j], 2.5)) and
    (theta_true_random[j] <= np.percentile(theta_posterior_random[:, j], 97.5))
    for j in range(J)
])

random_results = {
    'mu_true': mu_true_random,
    'tau_true': tau_true_random,
    'mu_mean': mu_mean_r,
    'tau_mean': tau_mean_r,
    'mu_in_95ci': mu_in_ci,
    'tau_in_95ci': tau_in_ci,
    'theta_recovery': theta_in_ci.sum() / J,
    'convergence_pass': (summary_random['R_hat'] < 1.01).all()
}

print(f"\nRandom-effects test results:")
print(f"  mu recovered in 95% CI: {random_results['mu_in_95ci']}")
print(f"  tau recovered in 95% CI: {random_results['tau_in_95ci']}")
print(f"  {theta_in_ci.sum()}/{J} theta_i in 95% CIs")
print(f"  All R-hat < 1.01: {random_results['convergence_pass']}")
print()

# ============================================================================
# PART 3: FULL SIMULATION-BASED CALIBRATION (SBC)
# ============================================================================

print("\nPART 3: FULL SIMULATION-BASED CALIBRATION")
print("-" * 80)
print("Running multiple simulations with parameters drawn from prior")
print()

N_SIMS = 50  # Number of simulations
print(f"Number of simulations: {N_SIMS}")
print("This will take several minutes...")
print()

# Storage for results
sbc_results = {
    'mu_true': [],
    'tau_true': [],
    'mu_posterior': [],
    'tau_posterior': [],
    'mu_rank': [],
    'tau_rank': [],
    'mu_in_95ci': [],
    'tau_in_95ci': [],
    'theta_recovery_rate': [],
    'converged': [],
    'num_divergences': [],
    'min_ess': []
}

for sim in range(N_SIMS):
    print(f"Simulation {sim + 1}/{N_SIMS}...", end=" ", flush=True)

    # Draw parameters from prior
    mu_true = np.random.normal(0, 50)

    # Sample from Half-Cauchy(0, 5) for tau
    # Use inverse CDF: tau = scale * tan(pi * (u - 0.5)) for u in (0.5, 1)
    u = np.random.uniform(0.5, 1)
    tau_true = 5 * np.tan(np.pi * (u - 0.5))

    # Truncate extreme tau values for computational stability
    tau_true = min(tau_true, 50)  # Cap at 50

    # Generate study effects
    theta_true = mu_true + tau_true * np.random.randn(J)

    # Generate data
    y_sim = theta_true + SIGMA * np.random.randn(J)

    # Prepare Stan data
    stan_data_sim = {
        'J': J,
        'y': y_sim.tolist(),
        'sigma': SIGMA.tolist()
    }

    try:
        # Fit model
        fit_sim = model.sample(
            data=stan_data_sim,
            chains=4,
            iter_warmup=1000,
            iter_sampling=1000,
            adapt_delta=0.95,
            seed=1000 + sim,
            show_progress=False,
            show_console=False
        )

        # Extract posteriors
        mu_post = fit_sim.stan_variable('mu')
        tau_post = fit_sim.stan_variable('tau')
        theta_post = fit_sim.stan_variable('theta')

        # Calculate ranks (for SBC)
        # Rank = number of posterior samples less than true value
        mu_rank = (mu_post < mu_true).sum()
        tau_rank = (tau_post < tau_true).sum()

        # Check if true values in 95% CI
        mu_in_ci = (mu_true >= np.percentile(mu_post, 2.5)) and \
                   (mu_true <= np.percentile(mu_post, 97.5))
        tau_in_ci = (tau_true >= np.percentile(tau_post, 2.5)) and \
                    (tau_true <= np.percentile(tau_post, 97.5))

        # Theta recovery rate
        theta_in_ci = sum([
            (theta_true[j] >= np.percentile(theta_post[:, j], 2.5)) and
            (theta_true[j] <= np.percentile(theta_post[:, j], 97.5))
            for j in range(J)
        ]) / J

        # Diagnostics
        summary_sim = fit_sim.summary()
        converged = (summary_sim['R_hat'] < 1.01).all()
        num_div = fit_sim.diagnose().count('divergence')
        min_ess = summary_sim['ESS_bulk'].min()

        # Store results
        sbc_results['mu_true'].append(mu_true)
        sbc_results['tau_true'].append(tau_true)
        sbc_results['mu_posterior'].append(mu_post)
        sbc_results['tau_posterior'].append(tau_post)
        sbc_results['mu_rank'].append(mu_rank)
        sbc_results['tau_rank'].append(tau_rank)
        sbc_results['mu_in_95ci'].append(mu_in_ci)
        sbc_results['tau_in_95ci'].append(tau_in_ci)
        sbc_results['theta_recovery_rate'].append(theta_in_ci)
        sbc_results['converged'].append(converged)
        sbc_results['num_divergences'].append(num_div)
        sbc_results['min_ess'].append(min_ess)

        print(f"OK (mu={mu_true:.1f}, tau={tau_true:.1f}, converged={converged})")

    except Exception as e:
        print(f"FAILED: {str(e)}")
        continue

print()
print(f"Completed {len(sbc_results['mu_true'])}/{N_SIMS} simulations successfully")
print()

# Save SBC results
np.savez(
    CODE_DIR / 'sbc_results.npz',
    **sbc_results
)

# ============================================================================
# COMPUTE SUMMARY METRICS
# ============================================================================

print("\nSBC SUMMARY METRICS")
print("-" * 80)

# Calibration metrics
mu_coverage = np.mean(sbc_results['mu_in_95ci'])
tau_coverage = np.mean(sbc_results['tau_in_95ci'])
theta_coverage_mean = np.mean(sbc_results['theta_recovery_rate'])

print(f"Calibration (95% CI should contain truth ~95% of time):")
print(f"  mu coverage: {mu_coverage:.3f} ({mu_coverage*100:.1f}%)")
print(f"  tau coverage: {tau_coverage:.3f} ({tau_coverage*100:.1f}%)")
print(f"  theta coverage (avg): {theta_coverage_mean:.3f} ({theta_coverage_mean*100:.1f}%)")
print()

# Bias assessment
mu_bias = np.mean([
    sbc_results['mu_posterior'][i].mean() - sbc_results['mu_true'][i]
    for i in range(len(sbc_results['mu_true']))
])
tau_bias = np.mean([
    sbc_results['tau_posterior'][i].mean() - sbc_results['tau_true'][i]
    for i in range(len(sbc_results['tau_true']))
])

print(f"Bias (posterior mean - true value, averaged over simulations):")
print(f"  mu bias: {mu_bias:.3f}")
print(f"  tau bias: {tau_bias:.3f}")
print()

# Convergence summary
convergence_rate = np.mean(sbc_results['converged'])
median_min_ess = np.median(sbc_results['min_ess'])

print(f"Convergence diagnostics:")
print(f"  Convergence rate (R-hat < 1.01): {convergence_rate:.3f} ({convergence_rate*100:.1f}%)")
print(f"  Median minimum ESS: {median_min_ess:.0f}")
print()

# SBC uniformity test (Kolmogorov-Smirnov test)
n_posterior_samples = len(sbc_results['mu_posterior'][0])
mu_rank_norm = np.array(sbc_results['mu_rank']) / n_posterior_samples
tau_rank_norm = np.array(sbc_results['tau_rank']) / n_posterior_samples

# Test if ranks are uniform
ks_stat_mu, ks_pval_mu = stats.kstest(mu_rank_norm, 'uniform')
ks_stat_tau, ks_pval_tau = stats.kstest(tau_rank_norm, 'uniform')

print(f"SBC Uniformity Test (ranks should be uniform):")
print(f"  mu: KS statistic = {ks_stat_mu:.3f}, p-value = {ks_pval_mu:.3f}")
print(f"  tau: KS statistic = {ks_stat_tau:.3f}, p-value = {ks_pval_tau:.3f}")
print(f"  (p > 0.05 indicates ranks are consistent with uniform)")
print()

# Overall assessment
print("=" * 80)
print("PASS/FAIL CRITERIA")
print("-" * 80)

checks = {
    'mu_coverage_ok': 0.90 <= mu_coverage <= 1.0,
    'tau_coverage_ok': 0.85 <= tau_coverage <= 1.0,  # More lenient for tau
    'theta_coverage_ok': theta_coverage_mean >= 0.85,
    'no_large_bias': abs(mu_bias) < 2 and abs(tau_bias) < 2,
    'convergence_ok': convergence_rate >= 0.95,
    'sbc_uniform_mu': ks_pval_mu > 0.05,
    'sbc_uniform_tau': ks_pval_tau > 0.05
}

print("Individual checks:")
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {check}: {status}")
print()

overall_pass = all(checks.values())
print(f"OVERALL VERDICT: {'PASS' if overall_pass else 'FAIL'}")
print("=" * 80)

# Save summary metrics
metrics = {
    'fixed_effect_scenario': fixed_results,
    'random_effects_scenario': random_results,
    'sbc_summary': {
        'n_simulations': len(sbc_results['mu_true']),
        'mu_coverage': float(mu_coverage),
        'tau_coverage': float(tau_coverage),
        'theta_coverage_mean': float(theta_coverage_mean),
        'mu_bias': float(mu_bias),
        'tau_bias': float(tau_bias),
        'convergence_rate': float(convergence_rate),
        'median_min_ess': float(median_min_ess),
        'ks_test_mu': {'statistic': float(ks_stat_mu), 'pvalue': float(ks_pval_mu)},
        'ks_test_tau': {'statistic': float(ks_stat_tau), 'pvalue': float(ks_pval_tau)}
    },
    'checks': checks,
    'overall_pass': overall_pass
}

with open(CODE_DIR / 'validation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to: {CODE_DIR / 'validation_metrics.json'}")
print(f"SBC results saved to: {CODE_DIR / 'sbc_results.npz'}")
print("\nNext: Run visualization script to create diagnostic plots")
