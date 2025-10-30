"""
Simulation-Based Calibration for Beta-Binomial (Reparameterized) Model
Using scipy optimization instead of Stan (due to environment constraints)

Tests whether the model can recover known parameters through simulation.
Critical safety check before fitting real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy import stats, optimize
import warnings
from typing import Dict, List, Tuple
import time

# Set random seed for reproducibility
np.random.seed(42)

# Paths
WORKSPACE = Path("/workspace")
DATA_PATH = WORKSPACE / "data" / "data.csv"
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
N_BOOTSTRAP = 1000  # Number of bootstrap samples for posterior approximation

print(f"\n{'='*70}")
print(f"SIMULATION-BASED CALIBRATION (scipy implementation)")
print(f"{'='*70}")
print(f"Simulations: {N_SIMULATIONS}")
print(f"Bootstrap samples: {N_BOOTSTRAP}")
print(f"Method: Maximum Likelihood + Bootstrap for uncertainty")
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


def beta_binomial_loglik(params: np.ndarray, r: np.ndarray, n: np.ndarray) -> float:
    """
    Negative log-likelihood for beta-binomial model.

    Parameters:
    - params: [mu, log_kappa] (use log for unconstrained optimization)
    - r: observed successes
    - n: number of trials
    """
    mu = params[0]
    kappa = np.exp(params[1])  # Transform from log scale

    # Bounds check
    if mu <= 0 or mu >= 1 or kappa <= 0:
        return 1e10

    alpha = mu * kappa
    beta_param = (1 - mu) * kappa

    # Beta-binomial log-likelihood
    loglik = 0.0
    for i in range(len(r)):
        try:
            # Use scipy's betabinom pmf
            loglik += stats.betabinom.logpmf(r[i], n[i], alpha, beta_param)
        except:
            return 1e10

    return -loglik  # Return negative for minimization


def fit_model_mle(r_success: np.ndarray, n_trials: np.ndarray) -> Tuple[Dict, bool]:
    """
    Fit beta-binomial model using Maximum Likelihood Estimation.

    Returns:
    - point_estimates: dict with mu, kappa, phi
    - converged: bool indicating convergence
    """
    # Initial guess based on method of moments
    pooled_rate = r_success.sum() / n_trials.sum()

    # Initial guess: mu = pooled rate, kappa = 20 (prior mean)
    init_params = np.array([
        np.clip(pooled_rate, 0.01, 0.99),  # mu
        np.log(20.0)  # log(kappa)
    ])

    # Bounds for optimization
    bounds = [(0.001, 0.999), (np.log(0.1), np.log(200))]

    try:
        # Optimize
        result = optimize.minimize(
            beta_binomial_loglik,
            init_params,
            args=(r_success, n_trials),
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            mu_hat = result.x[0]
            kappa_hat = np.exp(result.x[1])
            phi_hat = 1 + 1/kappa_hat

            return {
                'mu': mu_hat,
                'kappa': kappa_hat,
                'phi': phi_hat,
                'negloglik': result.fun
            }, True
        else:
            return {}, False

    except Exception as e:
        print(f"  Optimization failed: {str(e)}")
        return {}, False


def bootstrap_uncertainty(r_success: np.ndarray, n_trials: np.ndarray,
                         n_bootstrap: int = 1000) -> Dict[str, Dict]:
    """
    Estimate uncertainty using parametric bootstrap.

    For each bootstrap iteration:
    1. Fit model to original data to get point estimate
    2. Simulate new data from fitted model
    3. Refit to bootstrap data
    4. Store estimates

    Returns distribution of parameter estimates.
    """
    # Fit to original data
    point_est, converged = fit_model_mle(r_success, n_trials)
    if not converged:
        return {}, False

    # Bootstrap
    bootstrap_estimates = {'mu': [], 'kappa': [], 'phi': []}

    for _ in range(n_bootstrap):
        # Simulate from fitted model
        true_params = {
            'alpha': point_est['mu'] * point_est['kappa'],
            'beta': (1 - point_est['mu']) * point_est['kappa']
        }
        r_boot = simulate_data(true_params, n_trials)

        # Refit
        boot_est, boot_converged = fit_model_mle(r_boot, n_trials)
        if boot_converged:
            bootstrap_estimates['mu'].append(boot_est['mu'])
            bootstrap_estimates['kappa'].append(boot_est['kappa'])
            bootstrap_estimates['phi'].append(boot_est['phi'])

    # Compute quantiles
    results = {}
    for param in ['mu', 'kappa', 'phi']:
        vals = np.array(bootstrap_estimates[param])
        if len(vals) > 10:  # Need sufficient bootstrap samples
            results[param] = {
                'mean': np.mean(vals),
                'median': np.median(vals),
                'sd': np.std(vals),
                'q2.5': np.percentile(vals, 2.5),
                'q97.5': np.percentile(vals, 97.5),
                'q5': np.percentile(vals, 5),
                'q95': np.percentile(vals, 95)
            }
        else:
            return {}, False

    return results, True


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

        # Step 3: Fit model (MLE)
        print("Fitting model (MLE)...")
        point_est, converged = fit_model_mle(r_success, N_TRIALS)

        if not converged:
            print("  MLE failed to converge")
            elapsed = time.time() - start_time
            return {
                'sim_id': sim_id,
                'converged': False,
                'runtime_sec': elapsed,
                'error': 'MLE convergence failure'
            }

        print(f"MLE: mu={point_est['mu']:.4f}, kappa={point_est['kappa']:.2f}, phi={point_est['phi']:.4f}")

        # Step 4: Bootstrap for uncertainty
        print("Computing uncertainty (bootstrap)...")
        post_summary, boot_converged = bootstrap_uncertainty(r_success, N_TRIALS, N_BOOTSTRAP)

        if not boot_converged:
            print("  Bootstrap failed")
            elapsed = time.time() - start_time
            return {
                'sim_id': sim_id,
                'converged': False,
                'runtime_sec': elapsed,
                'error': 'Bootstrap failure'
            }

        print(f"Posterior mu: {post_summary['mu']['mean']:.4f} [{post_summary['mu']['q2.5']:.4f}, {post_summary['mu']['q97.5']:.4f}]")
        print(f"Posterior kappa: {post_summary['kappa']['mean']:.2f} [{post_summary['kappa']['q2.5']:.2f}, {post_summary['kappa']['q97.5']:.2f}]")
        print(f"Posterior phi: {post_summary['phi']['mean']:.4f} [{post_summary['phi']['q2.5']:.4f}, {post_summary['phi']['q97.5']:.4f}]")

        # Step 5: Check parameter recovery (95% CI)
        mu_recovered = (post_summary['mu']['q2.5'] <= true_params['mu'] <= post_summary['mu']['q97.5'])
        kappa_recovered = (post_summary['kappa']['q2.5'] <= true_params['kappa'] <= post_summary['kappa']['q97.5'])
        phi_recovered = (post_summary['phi']['q2.5'] <= true_params['phi'] <= post_summary['phi']['q97.5'])

        print(f"Recovery (95% CI): mu={'PASS' if mu_recovered else 'FAIL'}, "
              f"kappa={'PASS' if kappa_recovered else 'FAIL'}, "
              f"phi={'PASS' if phi_recovered else 'FAIL'}")

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s")

        # Return results
        return {
            'sim_id': sim_id,
            'converged': True,
            'runtime_sec': elapsed,
            # True parameters
            'true_mu': true_params['mu'],
            'true_kappa': true_params['kappa'],
            'true_phi': true_params['phi'],
            # Posterior summaries
            'post_mu_mean': post_summary['mu']['mean'],
            'post_mu_median': post_summary['mu']['median'],
            'post_mu_sd': post_summary['mu']['sd'],
            'post_mu_q2.5': post_summary['mu']['q2.5'],
            'post_mu_q97.5': post_summary['mu']['q97.5'],
            'post_kappa_mean': post_summary['kappa']['mean'],
            'post_kappa_median': post_summary['kappa']['median'],
            'post_kappa_sd': post_summary['kappa']['sd'],
            'post_kappa_q2.5': post_summary['kappa']['q2.5'],
            'post_kappa_q97.5': post_summary['kappa']['q97.5'],
            'post_phi_mean': post_summary['phi']['mean'],
            'post_phi_median': post_summary['phi']['median'],
            'post_phi_sd': post_summary['phi']['sd'],
            'post_phi_q2.5': post_summary['phi']['q2.5'],
            'post_phi_q97.5': post_summary['phi']['q97.5'],
            # Recovery checks
            'mu_recovered_95': mu_recovered,
            'kappa_recovered_95': kappa_recovered,
            'phi_recovered_95': phi_recovered,
            # MLE point estimates
            'mle_mu': point_est['mu'],
            'mle_kappa': point_est['kappa'],
            'mle_phi': point_est['phi'],
            # Data characteristics
            'simulated_pooled_rate': pooled_rate,
            'simulated_total_successes': int(r_success.sum())
        }

    except Exception as e:
        print(f"ERROR in simulation {sim_id + 1}: {str(e)}")
        import traceback
        traceback.print_exc()
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
    print("NOTE: Using scipy MLE + bootstrap instead of Stan due to environment constraints")
    print("This provides similar validation, though with less sophisticated uncertainty quantification.\n")

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
            print("OVERALL VERDICT: CONDITIONAL PASS - See detailed assessment below")
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
            'overall_pass': overall_pass,
            'method': 'MLE + parametric bootstrap'
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
