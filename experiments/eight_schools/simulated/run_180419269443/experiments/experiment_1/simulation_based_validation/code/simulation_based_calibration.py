"""
Simulation-Based Calibration (SBC) for Hierarchical Normal Model

This script validates that the model can recover known parameters by:
1. Drawing true parameters from the prior
2. Generating synthetic data with those parameters
3. Fitting the model to synthetic data
4. Checking if posteriors recover the true parameters
5. Assessing calibration of credible intervals

If the model can't recover known truth, it won't find unknown truth.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(2025)

# Configuration
N_SIMULATIONS = 100  # Number of SBC iterations
N_WARMUP = 2000
N_SAMPLES = 2000
N_CHAINS = 4

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
print(f"  MCMC warmup iterations: {N_WARMUP}")
print(f"  MCMC sampling iterations: {N_SAMPLES}")
print(f"  Number of chains: {N_CHAINS}")
print(f"  Number of studies (J): {J}")
print(f"  Known sigma: {KNOWN_SIGMA}")
print()

# Compile Stan model
print("Compiling Stan model...")
stan_file = CODE_DIR / 'hierarchical_model.stan'
model = CmdStanModel(stan_file=str(stan_file))
print("Model compiled successfully!\n")

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
    'max_rhat': [],
    'min_ess_bulk': [],
    'n_divergences': [],
    'converged': [],
}

# Store theta recoveries for detailed analysis
theta_results = []

# Store ranks for SBC histogram
mu_ranks = []
tau_ranks = []

print("Starting SBC simulations...")
print("-"*80)

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

        # Step 4: Fit model to synthetic data
        data_dict = {
            'J': J,
            'y': y_sim.tolist(),
            'sigma': KNOWN_SIGMA.tolist()
        }

        fit = model.sample(
            data=data_dict,
            chains=N_CHAINS,
            iter_warmup=N_WARMUP,
            iter_sampling=N_SAMPLES,
            show_progress=False,
            show_console=False,
            adapt_delta=0.95,  # Higher adapt_delta for better sampling
        )

        # Step 5: Extract posterior samples
        mu_samples = fit.stan_variable('mu')
        tau_samples = fit.stan_variable('tau')
        theta_samples = fit.stan_variable('theta')  # Shape: (N_SAMPLES * N_CHAINS, J)

        # Step 6: Compute summaries
        mu_post_mean = np.mean(mu_samples)
        mu_post_sd = np.std(mu_samples)
        mu_q025 = np.percentile(mu_samples, 2.5)
        mu_q975 = np.percentile(mu_samples, 97.5)

        tau_post_mean = np.mean(tau_samples)
        tau_post_sd = np.std(tau_samples)
        tau_q025 = np.percentile(tau_samples, 2.5)
        tau_q975 = np.percentile(tau_samples, 97.5)

        # Step 7: Check coverage
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

        # Step 8: Compute ranks for SBC (key diagnostic!)
        # Rank = number of posterior samples < true value
        mu_rank = np.sum(mu_samples < mu_true)
        tau_rank = np.sum(tau_samples < tau_true)
        mu_ranks.append(mu_rank)
        tau_ranks.append(tau_rank)

        # Step 9: Check convergence diagnostics
        summary = fit.summary()
        max_rhat = summary['R_hat'].max()
        min_ess_bulk = summary['N_Eff'].min()
        n_divergences = fit.divergences
        converged = (max_rhat < 1.01) and (min_ess_bulk > 400) and (n_divergences == 0)

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
        results['max_rhat'].append(max_rhat)
        results['min_ess_bulk'].append(min_ess_bulk)
        results['n_divergences'].append(n_divergences)
        results['converged'].append(converged)

        # Store theta results for detailed analysis (only store a few for memory)
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
    # Convert numpy arrays to lists for JSON serialization
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

# Convergence diagnostics
convergence_rate = df['converged'].mean()
print(f"\nConvergence diagnostics:")
print(f"  Convergence rate: {convergence_rate:.1%} ({df['converged'].sum()}/{len(df)})")
print(f"  Max R-hat (mean): {df['max_rhat'].mean():.4f}")
print(f"  Min ESS (mean): {df['min_ess_bulk'].mean():.0f}")
print(f"  Total divergences: {df['n_divergences'].sum()}")

# Distribution of true values (should span prior)
print(f"\nDistribution of true values:")
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
    'convergence_rate': float(convergence_rate),
    'mean_max_rhat': float(df['max_rhat'].mean()),
    'mean_min_ess': float(df['min_ess_bulk'].mean()),
    'total_divergences': int(df['n_divergences'].sum()),
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

# Coverage should be 90-98% for 95% CIs (exact would be 95%)
if mu_coverage < 0.90 or mu_coverage > 0.98:
    fail_conditions.append(f"mu coverage {mu_coverage:.1%} outside [90%, 98%]")
if tau_coverage < 0.90 or tau_coverage > 0.98:
    fail_conditions.append(f"tau coverage {tau_coverage:.1%} outside [90%, 98%]")
if theta_coverage < 0.90 or theta_coverage > 0.98:
    fail_conditions.append(f"theta coverage {theta_coverage:.1%} outside [90%, 98%]")

# Bias should be small (< 10% relative bias)
if abs(mu_rel_bias_pct) > 10:
    fail_conditions.append(f"|mu relative bias| {abs(mu_rel_bias_pct):.1f}% > 10%")
if abs(tau_rel_bias_pct) > 10:
    fail_conditions.append(f"|tau relative bias| {abs(tau_rel_bias_pct):.1f}% > 10%")

# Convergence rate should be high
if convergence_rate < 0.95:
    fail_conditions.append(f"convergence rate {convergence_rate:.1%} < 95%")

# Divergences should be rare
if df['n_divergences'].sum() > 0.05 * successful_sims:
    fail_conditions.append(f"{df['n_divergences'].sum()} divergences in {successful_sims} sims (>5%)")

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
    print(f"  - Coverage rates within [90%, 98%]")
    print(f"  - Bias < 10% for all parameters")
    print(f"  - Convergence rate > 95%")
    print(f"  - Minimal divergences")
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
