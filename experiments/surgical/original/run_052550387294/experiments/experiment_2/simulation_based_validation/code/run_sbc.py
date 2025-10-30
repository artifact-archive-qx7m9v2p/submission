"""
Simulation-Based Calibration (SBC) for Hierarchical Logit Model

Tests if the model can recover known parameters through:
1. Sample true parameters from priors
2. Generate synthetic data with these parameters
3. Fit model to synthetic data
4. Check if posteriors recover true parameters

Key metrics:
- Coverage rates for credible intervals
- Bias in parameter recovery
- Rank statistics (should be uniform)
- Posterior contraction from prior
"""

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from scipy import stats
from scipy.special import expit  # logistic function
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
CODE_DIR = Path("/workspace/experiments/experiment_2/simulation_based_validation/code")
PLOTS_DIR = Path("/workspace/experiments/experiment_2/simulation_based_validation/plots")
RESULTS_DIR = Path("/workspace/experiments/experiment_2/simulation_based_validation/results")

# Actual sample sizes from data
N_VALUES = np.array([47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360])
N_TRIALS = len(N_VALUES)

# SBC configuration
N_SBC_ITERATIONS = 150  # Balance thoroughness vs computation time
N_CHAINS = 4
N_ITER = 1000  # Per chain after warmup
N_WARMUP = 500

# Prior parameters
MU_LOGIT_MEAN = -2.53
MU_LOGIT_SD = 1.0
SIGMA_SD = 1.0  # Half-normal

def sample_prior():
    """Sample from priors: mu_logit ~ Normal(-2.53, 1), sigma ~ HalfNormal(0, 1)"""
    mu_logit = np.random.normal(MU_LOGIT_MEAN, MU_LOGIT_SD)
    sigma = np.abs(np.random.normal(0, SIGMA_SD))  # Half-normal
    eta = np.random.standard_normal(N_TRIALS)
    return mu_logit, sigma, eta

def generate_synthetic_data(mu_logit_true, sigma_true, eta_true, n_values):
    """
    Generate synthetic data from Hierarchical Logit model

    Args:
        mu_logit_true: True population mean on logit scale
        sigma_true: True standard deviation on logit scale
        eta_true: True standardized trial-specific effects
        n_values: Array of sample sizes

    Returns:
        Array of counts r_i
    """
    # Compute trial-specific log-odds
    logit_theta = mu_logit_true + sigma_true * eta_true

    # Transform to probabilities
    theta = expit(logit_theta)

    # Generate binomial data
    r_values = []
    for n_i, theta_i in zip(n_values, theta):
        r_i = np.random.binomial(n_i, theta_i)
        r_values.append(r_i)

    return np.array(r_values)

def compute_rank(true_value, posterior_samples):
    """
    Compute rank of true value in posterior samples

    Rank should be uniformly distributed if model is well-calibrated
    """
    return np.sum(posterior_samples < true_value)

def run_sbc_iteration(iteration, model, n_values):
    """
    Run one SBC iteration

    Returns:
        Dictionary with results or None if fitting failed
    """
    print(f"SBC iteration {iteration + 1}/{N_SBC_ITERATIONS}...", end=" ")

    # 1. Sample true parameters from priors
    mu_logit_true, sigma_true, eta_true = sample_prior()

    # 2. Generate synthetic data
    r_synthetic = generate_synthetic_data(mu_logit_true, sigma_true, eta_true, n_values)

    # 3. Fit model to synthetic data
    data = {
        'N': N_TRIALS,
        'n': n_values.tolist(),
        'r': r_synthetic.tolist()
    }

    try:
        fit = model.sample(
            data=data,
            chains=N_CHAINS,
            iter_sampling=N_ITER,
            iter_warmup=N_WARMUP,
            adapt_delta=0.95,  # Reduce divergences
            max_treedepth=12,
            show_progress=False,
            show_console=False,
            seed=iteration + 100  # Different seed per iteration
        )

        # Check for divergences
        divergences = fit.divergences()
        n_divergences = np.sum(divergences) if divergences is not None else 0
        n_samples = N_CHAINS * N_ITER
        divergence_rate = n_divergences / n_samples

        # Check convergence
        summary = fit.summary()
        max_rhat = summary['R_hat'].max()

        if max_rhat > 1.05:
            print(f"WARNING: High Rhat = {max_rhat:.3f}")
            converged = False
        else:
            converged = True

        # Extract posterior samples
        mu_logit_samples = fit.stan_variable('mu_logit').flatten()
        sigma_samples = fit.stan_variable('sigma').flatten()
        eta_samples = fit.stan_variable('eta')  # Shape: (n_samples, N_TRIALS)

        # 4. Compute recovery metrics for primary parameters (mu_logit, sigma)
        results = {
            'iteration': iteration,
            'mu_logit_true': mu_logit_true,
            'sigma_true': sigma_true,
            'mu_logit_mean': mu_logit_samples.mean(),
            'mu_logit_median': np.median(mu_logit_samples),
            'mu_logit_sd': mu_logit_samples.std(),
            'mu_logit_q025': np.percentile(mu_logit_samples, 2.5),
            'mu_logit_q975': np.percentile(mu_logit_samples, 97.5),
            'mu_logit_q05': np.percentile(mu_logit_samples, 5),
            'mu_logit_q95': np.percentile(mu_logit_samples, 95),
            'mu_logit_q10': np.percentile(mu_logit_samples, 10),
            'mu_logit_q90': np.percentile(mu_logit_samples, 90),
            'sigma_mean': sigma_samples.mean(),
            'sigma_median': np.median(sigma_samples),
            'sigma_sd': sigma_samples.std(),
            'sigma_q025': np.percentile(sigma_samples, 2.5),
            'sigma_q975': np.percentile(sigma_samples, 97.5),
            'sigma_q05': np.percentile(sigma_samples, 5),
            'sigma_q95': np.percentile(sigma_samples, 95),
            'sigma_q10': np.percentile(sigma_samples, 10),
            'sigma_q90': np.percentile(sigma_samples, 90),
            'mu_logit_rank': compute_rank(mu_logit_true, mu_logit_samples),
            'sigma_rank': compute_rank(sigma_true, sigma_samples),
            'max_rhat': max_rhat,
            'converged': converged,
            'n_divergences': n_divergences,
            'divergence_rate': divergence_rate
        }

        # Add eta recovery metrics (aggregated)
        for i in range(N_TRIALS):
            eta_i_samples = eta_samples[:, i]
            results[f'eta_{i}_true'] = eta_true[i]
            results[f'eta_{i}_mean'] = eta_i_samples.mean()
            results[f'eta_{i}_sd'] = eta_i_samples.std()

        status = "✓"
        if divergence_rate > 0.01:
            status = f"⚠ ({n_divergences} divergences)"
        elif not converged:
            status = "⚠ (poor convergence)"

        print(f"{status} (mu: {mu_logit_true:.2f}, sigma: {sigma_true:.2f})")
        return results

    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        return None

def main():
    """Run SBC analysis"""

    print("="*70)
    print("SIMULATION-BASED CALIBRATION FOR HIERARCHICAL LOGIT MODEL")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - SBC iterations: {N_SBC_ITERATIONS}")
    print(f"  - MCMC chains: {N_CHAINS}")
    print(f"  - MCMC iterations: {N_ITER} per chain ({N_WARMUP} warmup)")
    print(f"  - Total samples per iteration: {N_CHAINS * N_ITER}")
    print(f"  - Number of trials: {N_TRIALS}")
    print(f"  - Sample sizes: {N_VALUES}")
    print(f"\nPriors:")
    print(f"  - mu_logit ~ Normal({MU_LOGIT_MEAN}, {MU_LOGIT_SD})")
    print(f"  - sigma ~ HalfNormal(0, {SIGMA_SD})")
    print(f"  - eta_i ~ Normal(0, 1)")
    print(f"\nSampling configuration:")
    print(f"  - adapt_delta: 0.95 (increased to reduce divergences)")
    print(f"  - max_treedepth: 12")
    print(f"\n{'='*70}\n")

    # Compile Stan model
    print("Compiling Stan model...")
    model_file = CODE_DIR / "hierarchical_logit.stan"
    try:
        model = CmdStanModel(stan_file=str(model_file))
        print("✓ Model compiled\n")
    except Exception as e:
        print(f"✗ Compilation failed: {str(e)}")
        return None, None

    # Run SBC iterations
    print(f"Running {N_SBC_ITERATIONS} SBC iterations...")
    print("-"*70)

    results_list = []
    for i in range(N_SBC_ITERATIONS):
        result = run_sbc_iteration(i, model, N_VALUES)
        if result is not None:
            results_list.append(result)

    print("-"*70)
    print(f"\nCompleted: {len(results_list)}/{N_SBC_ITERATIONS} successful fits")

    if len(results_list) < N_SBC_ITERATIONS * 0.9:
        print(f"WARNING: Only {len(results_list)/N_SBC_ITERATIONS*100:.1f}% success rate")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results
    results_df.to_csv(RESULTS_DIR / "sbc_results.csv", index=False)
    print(f"\n✓ Results saved to: {RESULTS_DIR / 'sbc_results.csv'}")

    # Compute summary statistics
    print("\n" + "="*70)
    print("SBC SUMMARY STATISTICS")
    print("="*70)

    # Coverage rates
    mu_logit_coverage_95 = np.mean((results_df['mu_logit_true'] >= results_df['mu_logit_q025']) &
                                    (results_df['mu_logit_true'] <= results_df['mu_logit_q975']))
    sigma_coverage_95 = np.mean((results_df['sigma_true'] >= results_df['sigma_q025']) &
                                 (results_df['sigma_true'] <= results_df['sigma_q975']))

    mu_logit_coverage_90 = np.mean((results_df['mu_logit_true'] >= results_df['mu_logit_q05']) &
                                    (results_df['mu_logit_true'] <= results_df['mu_logit_q95']))
    sigma_coverage_90 = np.mean((results_df['sigma_true'] >= results_df['sigma_q05']) &
                                 (results_df['sigma_true'] <= results_df['sigma_q95']))

    mu_logit_coverage_80 = np.mean((results_df['mu_logit_true'] >= results_df['mu_logit_q10']) &
                                    (results_df['mu_logit_true'] <= results_df['mu_logit_q90']))
    sigma_coverage_80 = np.mean((results_df['sigma_true'] >= results_df['sigma_q10']) &
                                 (results_df['sigma_true'] <= results_df['sigma_q90']))

    print(f"\nCoverage Rates:")
    print(f"  mu_logit:")
    print(f"    - 95% CI: {mu_logit_coverage_95:.3f} (target: 0.950)")
    print(f"    - 90% CI: {mu_logit_coverage_90:.3f} (target: 0.900)")
    print(f"    - 80% CI: {mu_logit_coverage_80:.3f} (target: 0.800)")
    print(f"  sigma:")
    print(f"    - 95% CI: {sigma_coverage_95:.3f} (target: 0.950)")
    print(f"    - 90% CI: {sigma_coverage_90:.3f} (target: 0.900)")
    print(f"    - 80% CI: {sigma_coverage_80:.3f} (target: 0.800)")

    # Bias
    mu_logit_bias = (results_df['mu_logit_mean'] - results_df['mu_logit_true']).mean()
    sigma_bias = (results_df['sigma_mean'] - results_df['sigma_true']).mean()

    print(f"\nBias (posterior mean - true value):")
    print(f"  - mu_logit: {mu_logit_bias:.4f} (target: ~0)")
    print(f"  - sigma:    {sigma_bias:.4f} (target: ~0)")

    # RMSE
    mu_logit_rmse = np.sqrt(np.mean((results_df['mu_logit_mean'] - results_df['mu_logit_true'])**2))
    sigma_rmse = np.sqrt(np.mean((results_df['sigma_mean'] - results_df['sigma_true'])**2))

    print(f"\nRMSE (root mean squared error):")
    print(f"  - mu_logit: {mu_logit_rmse:.4f}")
    print(f"  - sigma:    {sigma_rmse:.4f}")

    # Prior vs posterior uncertainty
    prior_mu_logit_sd = MU_LOGIT_SD
    prior_sigma_sd = SIGMA_SD * np.sqrt(1 - 2/np.pi)  # Half-normal SD

    posterior_mu_logit_sd = results_df['mu_logit_sd'].mean()
    posterior_sigma_sd = results_df['sigma_sd'].mean()

    print(f"\nPosterior Contraction (posterior SD / prior SD):")
    print(f"  - mu_logit: {posterior_mu_logit_sd:.4f} / {prior_mu_logit_sd:.4f} = {posterior_mu_logit_sd/prior_mu_logit_sd:.3f}")
    print(f"  - sigma:    {posterior_sigma_sd:.4f} / {prior_sigma_sd:.4f} = {posterior_sigma_sd/prior_sigma_sd:.3f}")

    # Convergence
    converged_rate = results_df['converged'].mean()
    print(f"\nConvergence Rate: {converged_rate:.3f} (target: >0.90)")

    # Divergences
    mean_divergence_rate = results_df['divergence_rate'].mean()
    max_divergence_rate = results_df['divergence_rate'].max()
    pct_with_divergences = np.mean(results_df['divergence_rate'] > 0)

    print(f"\nDivergence Statistics:")
    print(f"  - Mean divergence rate: {mean_divergence_rate:.4f} (target: <0.01)")
    print(f"  - Max divergence rate:  {max_divergence_rate:.4f}")
    print(f"  - % iterations with divergences: {pct_with_divergences*100:.1f}%")

    # Rank statistics (should be uniform)
    n_samples = N_CHAINS * N_ITER
    print(f"\nRank Statistics (uniformity test):")

    # Chi-square test for uniformity
    n_bins = 20
    mu_logit_ranks = results_df['mu_logit_rank'].values
    sigma_ranks = results_df['sigma_rank'].values

    mu_logit_hist, _ = np.histogram(mu_logit_ranks, bins=n_bins, range=(0, n_samples))
    sigma_hist, _ = np.histogram(sigma_ranks, bins=n_bins, range=(0, n_samples))

    expected_count = len(mu_logit_ranks) / n_bins
    mu_logit_chisq = np.sum((mu_logit_hist - expected_count)**2 / expected_count)
    sigma_chisq = np.sum((sigma_hist - expected_count)**2 / expected_count)

    mu_logit_pvalue = 1 - stats.chi2.cdf(mu_logit_chisq, n_bins - 1)
    sigma_pvalue = 1 - stats.chi2.cdf(sigma_chisq, n_bins - 1)

    print(f"  - mu_logit ranks: χ² = {mu_logit_chisq:.2f}, p = {mu_logit_pvalue:.3f}")
    print(f"  - sigma ranks:    χ² = {sigma_chisq:.2f}, p = {sigma_pvalue:.3f}")
    print(f"  (p > 0.05 suggests uniform ranks → good calibration)")

    # Save summary
    summary = {
        'n_iterations': len(results_list),
        'success_rate': len(results_list) / N_SBC_ITERATIONS,
        'convergence_rate': float(converged_rate),
        'divergence_rate_mean': float(mean_divergence_rate),
        'divergence_rate_max': float(max_divergence_rate),
        'pct_with_divergences': float(pct_with_divergences),
        'coverage': {
            'mu_logit_95': float(mu_logit_coverage_95),
            'mu_logit_90': float(mu_logit_coverage_90),
            'mu_logit_80': float(mu_logit_coverage_80),
            'sigma_95': float(sigma_coverage_95),
            'sigma_90': float(sigma_coverage_90),
            'sigma_80': float(sigma_coverage_80)
        },
        'bias': {
            'mu_logit': float(mu_logit_bias),
            'sigma': float(sigma_bias)
        },
        'rmse': {
            'mu_logit': float(mu_logit_rmse),
            'sigma': float(sigma_rmse)
        },
        'contraction': {
            'mu_logit': float(posterior_mu_logit_sd / prior_mu_logit_sd),
            'sigma': float(posterior_sigma_sd / prior_sigma_sd)
        },
        'rank_uniformity': {
            'mu_logit_chisq': float(mu_logit_chisq),
            'mu_logit_pvalue': float(mu_logit_pvalue),
            'sigma_chisq': float(sigma_chisq),
            'sigma_pvalue': float(sigma_pvalue)
        }
    }

    with open(RESULTS_DIR / 'sbc_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {RESULTS_DIR / 'sbc_summary.json'}")
    print("\n" + "="*70)

    return results_df, summary

if __name__ == "__main__":
    results_df, summary = main()
