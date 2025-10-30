"""
Simulation-Based Calibration for Beta-Binomial (Reparameterized) Model

Tests whether the model can recover known parameters through simulation.
Critical safety check before fitting real data.
"""

import numpy as np
import pandas as pd
import cmdstanpy
from pathlib import Path
import json
from scipy import stats
import warnings
from typing import Dict, List, Tuple
import time

# Set random seed for reproducibility
np.random.seed(42)

# Paths
WORKSPACE = Path("/workspace")
DATA_PATH = WORKSPACE / "data" / "data.csv"
STAN_MODEL_PATH = WORKSPACE / "experiments" / "designer_1" / "stan_models" / "model_b_reparameterized.stan"
OUTPUT_DIR = WORKSPACE / "experiments" / "experiment_1" / "simulation_based_validation"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load actual data to get realistic sample sizes
data = pd.read_csv(DATA_PATH)
N_GROUPS = len(data)
N_TRIALS = data['n_trials'].values

print(f"Loaded data: {N_GROUPS} groups")
print(f"Sample sizes: min={N_TRIALS.min()}, max={N_TRIALS.max()}, median={np.median(N_TRIALS)}")

# SBC Configuration
N_SIMULATIONS = 25  # Balanced approach for validation
MCMC_CHAINS = 2
MCMC_ITER = 1000
MCMC_WARMUP = 500

print(f"\n{'='*70}")
print(f"SIMULATION-BASED CALIBRATION")
print(f"{'='*70}")
print(f"Simulations: {N_SIMULATIONS}")
print(f"MCMC: {MCMC_CHAINS} chains x {MCMC_ITER} iterations ({MCMC_WARMUP} warmup)")
print(f"Model: {STAN_MODEL_PATH.name}")
print(f"{'='*70}\n")


def generate_true_parameters() -> Dict[str, float]:
    """
    Generate true parameters from priors.

    Priors:
    - mu ~ Beta(2, 18)
    - kappa ~ Gamma(2, 0.1)
    """
    mu_true = stats.beta.rvs(2, 18)
    kappa_true = stats.gamma.rvs(2, scale=1/0.1)

    # Derived quantities
    alpha_true = mu_true * kappa_true
    beta_true = (1 - mu_true) * kappa_true
    phi_true = 1 + 1/kappa_true

    return {
        'mu': mu_true,
        'kappa': kappa_true,
        'alpha': alpha_true,
        'beta': beta_true,
        'phi': phi_true
    }


def simulate_data(true_params: Dict[str, float], n_trials: np.ndarray) -> np.ndarray:
    """
    Simulate data given true parameters.

    Process:
    1. For each group, draw p_i ~ Beta(alpha_true, beta_true)
    2. For each group, draw r_i ~ Binomial(n_i, p_i)
    """
    alpha_true = true_params['alpha']
    beta_true = true_params['beta']
    n_groups = len(n_trials)

    # Draw group-level success probabilities
    p_true = stats.beta.rvs(alpha_true, beta_true, size=n_groups)

    # Draw observed successes
    r_success = np.array([
        stats.binom.rvs(n_trials[i], p_true[i])
        for i in range(n_groups)
    ])

    return r_success


def fit_model(r_success: np.ndarray, n_trials: np.ndarray) -> cmdstanpy.CmdStanMCMC:
    """
    Fit Stan model to simulated data.
    """
    # Compile model (cached after first compilation)
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_MODEL_PATH))

    # Prepare data
    stan_data = {
        'N': len(n_trials),
        'n_trials': n_trials.tolist(),
        'r_success': r_success.tolist()
    }

    # Fit model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        fit = model.sample(
            data=stan_data,
            chains=MCMC_CHAINS,
            iter_sampling=MCMC_ITER - MCMC_WARMUP,
            iter_warmup=MCMC_WARMUP,
            show_progress=False,
            show_console=False,
            adapt_delta=0.95,
            seed=None  # Different seed each simulation
        )

    return fit


def extract_posterior_summary(fit: cmdstanpy.CmdStanMCMC, param_name: str) -> Dict[str, float]:
    """
    Extract posterior summary statistics for a parameter.
    """
    samples = fit.stan_variable(param_name)

    return {
        'mean': np.mean(samples),
        'median': np.median(samples),
        'sd': np.std(samples),
        'q2.5': np.percentile(samples, 2.5),
        'q97.5': np.percentile(samples, 97.5),
        'q5': np.percentile(samples, 5),
        'q95': np.percentile(samples, 95)
    }


def check_convergence(fit: cmdstanpy.CmdStanMCMC) -> Tuple[bool, Dict[str, float]]:
    """
    Check MCMC convergence diagnostics.

    Returns:
    - converged: bool indicating if chain converged
    - diagnostics: dict with Rhat and ESS values
    """
    summary = fit.summary()

    # Check key parameters
    key_params = ['mu', 'kappa', 'phi']
    diagnostics = {}

    for param in key_params:
        param_row = summary[summary['Name'] == param]
        if len(param_row) > 0:
            diagnostics[f'{param}_rhat'] = param_row['R_hat'].values[0]
            diagnostics[f'{param}_ess_bulk'] = param_row['N_Eff'].values[0]
        else:
            diagnostics[f'{param}_rhat'] = np.nan
            diagnostics[f'{param}_ess_bulk'] = np.nan

    # Convergence criteria
    max_rhat = max([v for k, v in diagnostics.items() if 'rhat' in k and not np.isnan(v)])
    min_ess = min([v for k, v in diagnostics.items() if 'ess' in k and not np.isnan(v)])

    converged = (max_rhat < 1.02) and (min_ess > 200)

    return converged, diagnostics


def compute_rank(true_value: float, posterior_samples: np.ndarray) -> int:
    """
    Compute rank of true value among posterior samples.
    Used for SBC uniformity check.
    """
    return np.sum(posterior_samples < true_value)


def run_single_simulation(sim_id: int) -> Dict:
    """
    Run one complete SBC simulation.

    Returns dict with results or None if failed.
    """
    print(f"\n--- Simulation {sim_id + 1}/{N_SIMULATIONS} ---")

    start_time = time.time()

    try:
        # Step 1: Generate true parameters
        true_params = generate_true_parameters()
        print(f"True parameters: mu={true_params['mu']:.4f}, kappa={true_params['kappa']:.2f}, phi={true_params['phi']:.4f}")

        # Step 2: Simulate data
        r_success = simulate_data(true_params, N_TRIALS)
        pooled_rate = r_success.sum() / N_TRIALS.sum()
        print(f"Simulated data: pooled rate={pooled_rate:.4f}, successes={r_success.sum()}/{N_TRIALS.sum()}")

        # Step 3: Fit model
        print("Fitting model...")
        fit = fit_model(r_success, N_TRIALS)

        # Step 4: Check convergence
        converged, diagnostics = check_convergence(fit)
        print(f"Convergence: {'PASS' if converged else 'FAIL'}")
        if not converged:
            print(f"  Max Rhat: {max([v for k, v in diagnostics.items() if 'rhat' in k]):.4f}")
            print(f"  Min ESS: {min([v for k, v in diagnostics.items() if 'ess' in k]):.0f}")

        # Step 5: Extract posteriors
        mu_post = extract_posterior_summary(fit, 'mu')
        kappa_post = extract_posterior_summary(fit, 'kappa')
        phi_post = extract_posterior_summary(fit, 'phi')

        print(f"Posterior mu: {mu_post['mean']:.4f} [{mu_post['q2.5']:.4f}, {mu_post['q97.5']:.4f}]")
        print(f"Posterior kappa: {kappa_post['mean']:.2f} [{kappa_post['q2.5']:.2f}, {kappa_post['q97.5']:.2f}]")
        print(f"Posterior phi: {phi_post['mean']:.4f} [{phi_post['q2.5']:.4f}, {phi_post['q97.5']:.4f}]")

        # Step 6: Check parameter recovery (95% CI)
        mu_recovered = (mu_post['q2.5'] <= true_params['mu'] <= mu_post['q97.5'])
        kappa_recovered = (kappa_post['q2.5'] <= true_params['kappa'] <= kappa_post['q97.5'])
        phi_recovered = (phi_post['q2.5'] <= true_params['phi'] <= phi_post['q97.5'])

        print(f"Recovery (95% CI): mu={'PASS' if mu_recovered else 'FAIL'}, "
              f"kappa={'PASS' if kappa_recovered else 'FAIL'}, "
              f"phi={'PASS' if phi_recovered else 'FAIL'}")

        # Step 7: Compute ranks for uniformity check
        mu_samples = fit.stan_variable('mu')
        kappa_samples = fit.stan_variable('kappa')
        phi_samples = fit.stan_variable('phi')

        mu_rank = compute_rank(true_params['mu'], mu_samples)
        kappa_rank = compute_rank(true_params['kappa'], kappa_samples)
        phi_rank = compute_rank(true_params['phi'], phi_samples)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s")

        # Return results
        return {
            'sim_id': sim_id,
            'converged': converged,
            'runtime_sec': elapsed,
            # True parameters
            'true_mu': true_params['mu'],
            'true_kappa': true_params['kappa'],
            'true_phi': true_params['phi'],
            # Posterior summaries
            'post_mu_mean': mu_post['mean'],
            'post_mu_median': mu_post['median'],
            'post_mu_sd': mu_post['sd'],
            'post_mu_q2.5': mu_post['q2.5'],
            'post_mu_q97.5': mu_post['q97.5'],
            'post_kappa_mean': kappa_post['mean'],
            'post_kappa_median': kappa_post['median'],
            'post_kappa_sd': kappa_post['sd'],
            'post_kappa_q2.5': kappa_post['q2.5'],
            'post_kappa_q97.5': kappa_post['q97.5'],
            'post_phi_mean': phi_post['mean'],
            'post_phi_median': phi_post['median'],
            'post_phi_sd': phi_post['sd'],
            'post_phi_q2.5': phi_post['q2.5'],
            'post_phi_q97.5': phi_post['q97.5'],
            # Recovery checks
            'mu_recovered_95': mu_recovered,
            'kappa_recovered_95': kappa_recovered,
            'phi_recovered_95': phi_recovered,
            # Ranks
            'mu_rank': mu_rank,
            'kappa_rank': kappa_rank,
            'phi_rank': phi_rank,
            # Diagnostics
            **diagnostics,
            # Data characteristics
            'simulated_pooled_rate': pooled_rate,
            'simulated_total_successes': int(r_success.sum())
        }

    except Exception as e:
        print(f"ERROR in simulation {sim_id + 1}: {str(e)}")
        elapsed = time.time() - start_time
        return {
            'sim_id': sim_id,
            'converged': False,
            'runtime_sec': elapsed,
            'error': str(e)
        }


def main():
    """
    Run full SBC validation.
    """
    print("Starting Simulation-Based Calibration...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Run all simulations
    results = []
    for i in range(N_SIMULATIONS):
        result = run_single_simulation(i)
        results.append(result)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save raw results
    output_file = RESULTS_DIR / "sbc_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Convergence rate
    n_converged = df_results['converged'].sum()
    convergence_rate = n_converged / N_SIMULATIONS
    print(f"\nConvergence Rate: {n_converged}/{N_SIMULATIONS} ({convergence_rate*100:.1f}%)")

    # Filter to converged simulations for remaining analysis
    df_conv = df_results[df_results['converged']]
    n_conv = len(df_conv)

    if n_conv > 0:
        print(f"\nAnalyzing {n_conv} converged simulations...")

        # Coverage rates (95% CI)
        mu_coverage = df_conv['mu_recovered_95'].sum() / n_conv
        kappa_coverage = df_conv['kappa_recovered_95'].sum() / n_conv
        phi_coverage = df_conv['phi_recovered_95'].sum() / n_conv

        print(f"\nCoverage Rates (95% CI):")
        print(f"  mu:    {df_conv['mu_recovered_95'].sum()}/{n_conv} ({mu_coverage*100:.1f}%)")
        print(f"  kappa: {df_conv['kappa_recovered_95'].sum()}/{n_conv} ({kappa_coverage*100:.1f}%)")
        print(f"  phi:   {df_conv['phi_recovered_95'].sum()}/{n_conv} ({phi_coverage*100:.1f}%)")

        # Bias assessment (posterior mean - true value)
        mu_bias = (df_conv['post_mu_mean'] - df_conv['true_mu']).mean()
        kappa_bias = (df_conv['post_kappa_mean'] - df_conv['true_kappa']).mean()
        phi_bias = (df_conv['post_phi_mean'] - df_conv['true_phi']).mean()

        print(f"\nMean Bias (posterior mean - true value):")
        print(f"  mu:    {mu_bias:+.6f}")
        print(f"  kappa: {kappa_bias:+.3f}")
        print(f"  phi:   {phi_bias:+.6f}")

        # Interval widths
        mu_width = (df_conv['post_mu_q97.5'] - df_conv['post_mu_q2.5']).mean()
        kappa_width = (df_conv['post_kappa_q97.5'] - df_conv['post_kappa_q2.5']).mean()
        phi_width = (df_conv['post_phi_q97.5'] - df_conv['post_phi_q2.5']).mean()

        print(f"\nMean 95% CI Width:")
        print(f"  mu:    {mu_width:.4f}")
        print(f"  kappa: {kappa_width:.2f}")
        print(f"  phi:   {phi_width:.4f}")

        # Runtime
        mean_runtime = df_results['runtime_sec'].mean()
        print(f"\nMean Runtime: {mean_runtime:.1f}s per simulation")
        print(f"Total Runtime: {df_results['runtime_sec'].sum()/60:.1f} minutes")

        # Pass/Fail decision
        print(f"\n{'='*70}")
        print("PASS/FAIL ASSESSMENT")
        print("="*70)

        checks = {
            'Convergence rate >= 90%': convergence_rate >= 0.90,
            'Coverage mu >= 85%': mu_coverage >= 0.85,
            'Coverage kappa >= 85%': kappa_coverage >= 0.85,
            'Coverage phi >= 85%': phi_coverage >= 0.85,
            'No systematic bias in mu': abs(mu_bias) < 0.01,
            'No systematic bias in kappa': abs(kappa_bias) < 2.0,
            'No systematic bias in phi': abs(phi_bias) < 0.05
        }

        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check}")

        overall_pass = all(checks.values())
        print(f"\n{'='*70}")
        if overall_pass:
            print("OVERALL VERDICT: PASS - Model ready for real data fitting")
        else:
            print("OVERALL VERDICT: FAIL - Model requires revision")
        print(f"{'='*70}\n")

        # Save summary
        summary = {
            'n_simulations': N_SIMULATIONS,
            'n_converged': int(n_converged),
            'convergence_rate': convergence_rate,
            'mu_coverage_95': mu_coverage,
            'kappa_coverage_95': kappa_coverage,
            'phi_coverage_95': phi_coverage,
            'mu_bias': mu_bias,
            'kappa_bias': kappa_bias,
            'phi_bias': phi_bias,
            'mu_ci_width': mu_width,
            'kappa_ci_width': kappa_width,
            'phi_ci_width': phi_width,
            'mean_runtime_sec': mean_runtime,
            'overall_pass': overall_pass
        }

        summary_file = RESULTS_DIR / "sbc_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")

    else:
        print("\nWARNING: No simulations converged! Model has serious issues.")
        print("Cannot assess parameter recovery.")

    return df_results


if __name__ == "__main__":
    results = main()
    print("\nSBC complete. Now run visualization script to generate plots.")
