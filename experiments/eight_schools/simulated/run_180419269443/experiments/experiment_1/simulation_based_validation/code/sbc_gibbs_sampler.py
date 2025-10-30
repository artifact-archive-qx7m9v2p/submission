"""
Simulation-Based Calibration (SBC) for Hierarchical Normal Model
Using Gibbs Sampling (exact for this conjugate model)

The hierarchical normal model has conjugate structure, allowing exact Gibbs sampling:
- p(theta_i | mu, tau, y) is Normal (analytic)
- p(mu | theta, tau) is Normal (analytic)
- p(tau | theta, mu) uses Metropolis-Hastings step

This is more reliable than optimization for hierarchical models.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, invgamma, halfnorm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(2025)

# Configuration
N_SIMULATIONS = 100  # Number of SBC iterations
N_MCMC = 4000        # MCMC iterations
N_WARMUP = 1000      # Warmup iterations to discard

# Known sigma from the 8 schools problem
KNOWN_SIGMA = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = 8

# Output directories
CODE_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/code')
PLOTS_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation/plots')
PLOTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("SIMULATION-BASED CALIBRATION: Hierarchical Normal Model")
print("="*80)
print(f"\nConfiguration:")
print(f"  Number of simulations: {N_SIMULATIONS}")
print(f"  MCMC iterations: {N_MCMC}")
print(f"  Warmup iterations: {N_WARMUP}")
print(f"  Number of studies (J): {J}")
print(f"  Known sigma: {KNOWN_SIGMA}")
print(f"\nMethod: Gibbs Sampling (with MH for tau)")
print()

def gibbs_sampler(y, sigma, n_iter=N_MCMC, n_warmup=N_WARMUP):
    """
    Gibbs sampler for hierarchical normal model

    Model:
    y_i ~ N(theta_i, sigma_i^2)
    theta_i ~ N(mu, tau^2)
    mu ~ N(0, 25^2)
    tau ~ Half-Normal(0, 10^2)

    Returns samples from posterior
    """
    J = len(y)

    # Storage for samples
    n_samples = n_iter - n_warmup
    mu_samples = np.zeros(n_samples)
    tau_samples = np.zeros(n_samples)
    theta_samples = np.zeros((n_samples, J))

    # Initialize at reasonable values
    mu = np.mean(y)
    tau = max(np.std(y), 0.1)
    theta = y.copy()

    # Prior hyperparameters
    mu_prior_mean = 0
    mu_prior_sd = 25
    tau_prior_sd = 10

    # Metropolis-Hastings proposal for tau
    tau_proposal_sd = 2.0

    tau_accepts = 0

    for iter in range(n_iter):
        # Step 1: Sample theta_i | mu, tau, y_i (conjugate Normal)
        for j in range(J):
            # Posterior for theta_i is Normal with:
            # precision = 1/sigma_i^2 + 1/tau^2
            # mean = (y_i/sigma_i^2 + mu/tau^2) / precision
            prec = 1 / (sigma[j]**2) + 1 / (tau**2)
            post_mean = (y[j] / (sigma[j]**2) + mu / (tau**2)) / prec
            post_sd = np.sqrt(1 / prec)
            theta[j] = np.random.normal(post_mean, post_sd)

        # Step 2: Sample mu | theta, tau (conjugate Normal)
        # Posterior for mu is Normal with:
        # precision = J/tau^2 + 1/mu_prior_sd^2
        # mean = (sum(theta)/tau^2 + mu_prior_mean/mu_prior_sd^2) / precision
        prec_mu = J / (tau**2) + 1 / (mu_prior_sd**2)
        post_mean_mu = (np.sum(theta) / (tau**2) + mu_prior_mean / (mu_prior_sd**2)) / prec_mu
        post_sd_mu = np.sqrt(1 / prec_mu)
        mu = np.random.normal(post_mean_mu, post_sd_mu)

        # Step 3: Sample tau | theta, mu (Metropolis-Hastings on log scale)
        # Propose new tau
        log_tau = np.log(tau)
        log_tau_proposal = log_tau + np.random.normal(0, tau_proposal_sd)
        tau_proposal = np.exp(log_tau_proposal)

        # Log posterior for current tau
        log_post_current = (
            # Likelihood: theta_i ~ N(mu, tau^2)
            np.sum(norm.logpdf(theta, mu, tau))
            # Prior: tau ~ Half-Normal(0, tau_prior_sd)
            + norm.logpdf(tau, 0, tau_prior_sd)
            + log_tau  # Jacobian for log transform
        )

        # Log posterior for proposed tau
        log_post_proposal = (
            np.sum(norm.logpdf(theta, mu, tau_proposal))
            + norm.logpdf(tau_proposal, 0, tau_prior_sd)
            + log_tau_proposal  # Jacobian
        )

        # Accept/reject
        log_accept_ratio = log_post_proposal - log_post_current
        if np.log(np.random.rand()) < log_accept_ratio:
            tau = tau_proposal
            tau_accepts += 1

        # Store samples (after warmup)
        if iter >= n_warmup:
            idx = iter - n_warmup
            mu_samples[idx] = mu
            tau_samples[idx] = tau
            theta_samples[idx, :] = theta

    acceptance_rate = tau_accepts / n_iter

    return {
        'mu_samples': mu_samples,
        'tau_samples': tau_samples,
        'theta_samples': theta_samples,
        'acceptance_rate': acceptance_rate,
    }

# Storage for results
results = {
    'sim_id': [],
    'mu_true': [],
    'tau_true': [],
    'mu_post_mean': [],
    'mu_post_sd': [],
    'mu_q025': [],
    'mu_q975': [],
    'tau_post_mean': [],
    'tau_post_sd': [],
    'tau_q025': [],
    'tau_q975': [],
    'mu_in_ci': [],
    'tau_in_ci': [],
    'theta_coverage_rate': [],
    'acceptance_rate': [],
    'ess_mu': [],
    'ess_tau': [],
}

# Store theta recoveries for detailed analysis
theta_results = []

# Store ranks for SBC histogram
mu_ranks = []
tau_ranks = []

print("Starting SBC simulations...")
print("-"*80)

def compute_ess(samples):
    """Compute effective sample size using autocorrelation"""
    n = len(samples)
    # Demean
    x = samples - np.mean(samples)
    # Autocorrelation at lag 0
    acf_0 = np.dot(x, x) / n
    if acf_0 == 0:
        return n
    # Sum of autocorrelations (up to lag n/4)
    max_lag = min(n // 4, 100)
    rho_sum = 1.0  # lag 0
    for lag in range(1, max_lag):
        acf_lag = np.dot(x[:-lag], x[lag:]) / n / acf_0
        if acf_lag < 0.05:  # Truncate when autocorrelation becomes small
            break
        rho_sum += 2 * acf_lag
    ess = n / rho_sum
    return max(ess, 1.0)  # At least 1

successful_sims = 0

for sim in range(N_SIMULATIONS):
    try:
        # Step 1: Draw true parameters from prior
        mu_true = np.random.normal(0, 25)
        tau_true = np.abs(np.random.normal(0, 10))  # Half-normal

        # Step 2: Generate true study effects
        theta_true = np.random.normal(mu_true, tau_true, J)

        # Step 3: Generate synthetic data
        y_sim = np.random.normal(theta_true, KNOWN_SIGMA)

        # Step 4: Fit model to synthetic data using Gibbs sampling
        fit_result = gibbs_sampler(y_sim, KNOWN_SIGMA)

        # Step 5: Extract posterior samples
        mu_samples = fit_result['mu_samples']
        tau_samples = fit_result['tau_samples']
        theta_samples = fit_result['theta_samples']

        # Compute summaries
        mu_post_mean = np.mean(mu_samples)
        mu_post_sd = np.std(mu_samples)
        mu_q025 = np.percentile(mu_samples, 2.5)
        mu_q975 = np.percentile(mu_samples, 97.5)

        tau_post_mean = np.mean(tau_samples)
        tau_post_sd = np.std(tau_samples)
        tau_q025 = np.percentile(tau_samples, 2.5)
        tau_q975 = np.percentile(tau_samples, 97.5)

        # Step 6: Check coverage
        mu_in_ci = mu_q025 <= mu_true <= mu_q975
        tau_in_ci = tau_q025 <= tau_true <= tau_q975

        # Check theta coverage
        theta_in_ci = []
        for j in range(J):
            theta_j_samples = theta_samples[:, j]
            theta_q025 = np.percentile(theta_j_samples, 2.5)
            theta_q975 = np.percentile(theta_j_samples, 97.5)
            theta_in_ci.append(theta_q025 <= theta_true[j] <= theta_q975)

        theta_coverage_rate = np.mean(theta_in_ci)

        # Step 7: Compute ranks for SBC
        mu_rank = np.sum(mu_samples < mu_true)
        tau_rank = np.sum(tau_samples < tau_true)
        mu_ranks.append(mu_rank)
        tau_ranks.append(tau_rank)

        # Step 8: Compute ESS
        ess_mu = compute_ess(mu_samples)
        ess_tau = compute_ess(tau_samples)

        # Store results
        results['sim_id'].append(sim)
        results['mu_true'].append(mu_true)
        results['tau_true'].append(tau_true)
        results['mu_post_mean'].append(mu_post_mean)
        results['mu_post_sd'].append(mu_post_sd)
        results['mu_q025'].append(mu_q025)
        results['mu_q975'].append(mu_q975)
        results['tau_post_mean'].append(tau_post_mean)
        results['tau_post_sd'].append(tau_post_sd)
        results['tau_q025'].append(tau_q025)
        results['tau_q975'].append(tau_q975)
        results['mu_in_ci'].append(mu_in_ci)
        results['tau_in_ci'].append(tau_in_ci)
        results['theta_coverage_rate'].append(theta_coverage_rate)
        results['acceptance_rate'].append(fit_result['acceptance_rate'])
        results['ess_mu'].append(ess_mu)
        results['ess_tau'].append(ess_tau)

        # Store theta results for detailed analysis (only first 20)
        if sim < 20:
            theta_results.append({
                'sim_id': sim,
                'theta_true': theta_true,
                'theta_post_mean': theta_samples.mean(axis=0),
                'theta_q025': np.percentile(theta_samples, 2.5, axis=0),
                'theta_q975': np.percentile(theta_samples, 97.5, axis=0),
                'theta_in_ci': theta_in_ci,
            })

        successful_sims += 1

        if (sim + 1) % 10 == 0:
            print(f"Completed {sim + 1}/{N_SIMULATIONS} simulations "
                  f"(Success rate: {successful_sims}/{sim+1})")

    except Exception as e:
        print(f"Simulation {sim} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print("-"*80)
print(f"\nCompleted {successful_sims}/{N_SIMULATIONS} simulations successfully")
print()

# Convert results to DataFrame
df = pd.DataFrame(results)

# Save results
results_file = CODE_DIR / 'sbc_results.csv'
df.to_csv(results_file, index=False)
print(f"Results saved to: {results_file}")

# Save theta results
theta_file = CODE_DIR / 'theta_recovery_examples.json'
with open(theta_file, 'w') as f:
    theta_results_json = []
    for tr in theta_results:
        theta_results_json.append({
            'sim_id': int(tr['sim_id']),
            'theta_true': tr['theta_true'].tolist(),
            'theta_post_mean': tr['theta_post_mean'].tolist(),
            'theta_q025': tr['theta_q025'].tolist(),
            'theta_q975': tr['theta_q975'].tolist(),
            'theta_in_ci': [bool(x) for x in tr['theta_in_ci']],
        })
    json.dump(theta_results_json, f, indent=2)
print(f"Theta recovery examples saved to: {theta_file}")

# Save rank statistics
ranks_file = CODE_DIR / 'rank_statistics.npz'
np.savez(ranks_file, mu_ranks=mu_ranks, tau_ranks=tau_ranks)
print(f"Rank statistics saved to: {ranks_file}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Coverage rates (should be ~95% for 95% CIs)
mu_coverage = df['mu_in_ci'].mean()
tau_coverage = df['tau_in_ci'].mean()
theta_coverage = df['theta_coverage_rate'].mean()

print(f"\nCoverage rates (target: 95%):")
print(f"  mu coverage:    {mu_coverage:.1%} ({df['mu_in_ci'].sum()}/{len(df)})")
print(f"  tau coverage:   {tau_coverage:.1%} ({df['tau_in_ci'].sum()}/{len(df)})")
print(f"  theta coverage: {theta_coverage:.1%} (averaged across studies)")

# Bias (should be ~0)
mu_bias = (df['mu_post_mean'] - df['mu_true']).mean()
tau_bias = (df['tau_post_mean'] - df['tau_true']).mean()

mu_rel_bias_pct = 100 * mu_bias / df['mu_true'].abs().mean()
tau_rel_bias_pct = 100 * tau_bias / df['tau_true'].mean()

print(f"\nBias (posterior mean - true value):")
print(f"  mu bias:  {mu_bias:+.3f} (relative: {mu_rel_bias_pct:+.1f}%)")
print(f"  tau bias: {tau_bias:+.3f} (relative: {tau_rel_bias_pct:+.1f}%)")

# RMSE
mu_rmse = np.sqrt(((df['mu_post_mean'] - df['mu_true'])**2).mean())
tau_rmse = np.sqrt(((df['tau_post_mean'] - df['tau_true'])**2).mean())

print(f"\nRoot Mean Squared Error:")
print(f"  mu RMSE:  {mu_rmse:.3f}")
print(f"  tau RMSE: {tau_rmse:.3f}")

# MCMC diagnostics
mean_acceptance = df['acceptance_rate'].mean()
mean_ess_mu = df['ess_mu'].mean()
mean_ess_tau = df['ess_tau'].mean()

print(f"\nMCMC diagnostics:")
print(f"  Mean acceptance rate (tau): {mean_acceptance:.1%}")
print(f"  Mean ESS (mu): {mean_ess_mu:.0f}")
print(f"  Mean ESS (tau): {mean_ess_tau:.0f}")

# Distribution of true values (should span prior)
print(f"\nDistribution of true values sampled from prior:")
print(f"  mu_true: mean={df['mu_true'].mean():.2f}, std={df['mu_true'].std():.2f}")
print(f"  tau_true: mean={df['tau_true'].mean():.2f}, std={df['tau_true'].std():.2f}")

# Save summary statistics
summary_stats = {
    'n_simulations': successful_sims,
    'mu_coverage': float(mu_coverage),
    'tau_coverage': float(tau_coverage),
    'theta_coverage': float(theta_coverage),
    'mu_bias': float(mu_bias),
    'tau_bias': float(tau_bias),
    'mu_rel_bias_pct': float(mu_rel_bias_pct),
    'tau_rel_bias_pct': float(tau_rel_bias_pct),
    'mu_rmse': float(mu_rmse),
    'tau_rmse': float(tau_rmse),
    'mean_acceptance_rate': float(mean_acceptance),
    'mean_ess_mu': float(mean_ess_mu),
    'mean_ess_tau': float(mean_ess_tau),
}

summary_file = CODE_DIR / 'summary_statistics.json'
with open(summary_file, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"\nSummary statistics saved to: {summary_file}")

print("\n" + "="*80)
print("PASS/FAIL ASSESSMENT")
print("="*80)

# Decision criteria
fail_conditions = []

# Coverage should be 90-98% for 95% CIs
if mu_coverage < 0.90 or mu_coverage > 0.98:
    fail_conditions.append(f"mu coverage {mu_coverage:.1%} outside [90%, 98%]")
if tau_coverage < 0.88 or tau_coverage > 0.98:  # Slightly relaxed for tau
    fail_conditions.append(f"tau coverage {tau_coverage:.1%} outside [88%, 98%]")
if theta_coverage < 0.90 or theta_coverage > 0.98:
    fail_conditions.append(f"theta coverage {theta_coverage:.1%} outside [90%, 98%]")

# Bias should be small (< 10% relative bias)
if abs(mu_rel_bias_pct) > 10:
    fail_conditions.append(f"|mu relative bias| {abs(mu_rel_bias_pct):.1f}% > 10%")
if abs(tau_rel_bias_pct) > 15:  # Slightly relaxed for tau
    fail_conditions.append(f"|tau relative bias| {abs(tau_rel_bias_pct):.1f}% > 15%")

# ESS should be reasonable
if mean_ess_mu < 100:
    fail_conditions.append(f"mean ESS(mu) {mean_ess_mu:.0f} < 100")
if mean_ess_tau < 50:  # Lower threshold for tau (harder to sample)
    fail_conditions.append(f"mean ESS(tau) {mean_ess_tau:.0f} < 50")

if fail_conditions:
    decision = "FAIL"
    print(f"\nDECISION: {decision}")
    print("\nReasons for failure:")
    for reason in fail_conditions:
        print(f"  - {reason}")
    print("\nModel cannot reliably recover known parameters.")
    print("DO NOT proceed to real data fitting until issues are resolved.")
else:
    decision = "PASS"
    print(f"\nDECISION: {decision}")
    print("\nAll calibration criteria satisfied:")
    print(f"  - Coverage rates within acceptable ranges")
    print(f"  - Bias < 10-15% for all parameters")
    print(f"  - ESS sufficient for inference")
    print("\nModel successfully recovers known parameters.")
    print("Safe to proceed to real data fitting.")

summary_stats['decision'] = decision
summary_stats['fail_conditions'] = fail_conditions

# Re-save summary with decision
with open(summary_file, 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\n" + "="*80)
print("Simulation-based calibration complete!")
print("="*80)
