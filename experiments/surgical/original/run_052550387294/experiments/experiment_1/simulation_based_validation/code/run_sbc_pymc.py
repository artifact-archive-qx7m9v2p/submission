"""
Simulation-Based Calibration (SBC) for Beta-Binomial Model using PyMC

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
import pymc as pm
import matplotlib.pyplot as plt
from scipy import stats
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
CODE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation/code")
PLOTS_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation/plots")
RESULTS_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation/results")

# Actual sample sizes from data
N_VALUES = np.array([47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360])
N_TRIALS = len(N_VALUES)

# SBC configuration
N_SBC_ITERATIONS = 150  # Balance thoroughness vs computation time
N_CHAINS = 4
N_SAMPLES = 1000
N_TUNE = 500

# Prior functions
def sample_prior():
    """Sample from priors: mu ~ Beta(2, 25), phi ~ Gamma(2, 2)"""
    mu = np.random.beta(2, 25)
    phi = np.random.gamma(2, 2)
    return mu, phi

def generate_synthetic_data(mu_true, phi_true, n_values):
    """
    Generate synthetic data from Beta-Binomial

    Args:
        mu_true: True mean probability
        phi_true: True concentration
        n_values: Array of sample sizes

    Returns:
        Array of counts r_i
    """
    alpha_true = mu_true * phi_true
    beta_true = (1 - mu_true) * phi_true

    r_values = []
    for n in n_values:
        # Beta-Binomial is compound: theta ~ Beta(alpha, beta), r ~ Binomial(n, theta)
        theta = np.random.beta(alpha_true, beta_true)
        r = np.random.binomial(n, theta)
        r_values.append(r)

    return np.array(r_values)

def compute_rank(true_value, posterior_samples):
    """
    Compute rank of true value in posterior samples

    Rank should be uniformly distributed if model is well-calibrated
    """
    return np.sum(posterior_samples < true_value)

def run_sbc_iteration(iteration, n_values):
    """
    Run one SBC iteration

    Returns:
        Dictionary with results or None if fitting failed
    """
    print(f"SBC iteration {iteration + 1}/{N_SBC_ITERATIONS}...", end=" ")

    # 1. Sample true parameters from priors
    mu_true, phi_true = sample_prior()
    alpha_true = mu_true * phi_true
    beta_true = (1 - mu_true) * phi_true

    # 2. Generate synthetic data
    r_synthetic = generate_synthetic_data(mu_true, phi_true, n_values)

    # 3. Fit model to synthetic data
    try:
        with pm.Model() as model:
            # Priors
            mu = pm.Beta('mu', alpha=2, beta=25)
            phi = pm.Gamma('phi', alpha=2, beta=2)

            # Transformed parameters
            alpha = pm.Deterministic('alpha', mu * phi)
            beta_param = pm.Deterministic('beta', (1 - mu) * phi)

            # Likelihood
            r_obs = pm.BetaBinomial('r_obs', alpha=alpha, beta=beta_param,
                                    n=n_values, observed=r_synthetic)

            # Sample
            trace = pm.sample(
                draws=N_SAMPLES,
                tune=N_TUNE,
                chains=N_CHAINS,
                return_inferencedata=True,
                progressbar=False,
                random_seed=iteration,
                cores=1,
                target_accept=0.95
            )

        # Check convergence
        rhat = pm.rhat(trace)
        max_rhat = max(float(rhat['mu'].values), float(rhat['phi'].values))

        if max_rhat > 1.05:
            print(f"WARNING: High Rhat = {max_rhat:.3f}")
            converged = False
        else:
            converged = True

        # Extract posterior samples
        mu_samples = trace.posterior['mu'].values.flatten()
        phi_samples = trace.posterior['phi'].values.flatten()
        alpha_samples = trace.posterior['alpha'].values.flatten()
        beta_samples = trace.posterior['beta'].values.flatten()

        # 4. Compute recovery metrics
        results = {
            'iteration': iteration,
            'mu_true': mu_true,
            'phi_true': phi_true,
            'alpha_true': alpha_true,
            'beta_true': beta_true,
            'mu_mean': mu_samples.mean(),
            'mu_median': np.median(mu_samples),
            'mu_sd': mu_samples.std(),
            'mu_q025': np.percentile(mu_samples, 2.5),
            'mu_q975': np.percentile(mu_samples, 97.5),
            'phi_mean': phi_samples.mean(),
            'phi_median': np.median(phi_samples),
            'phi_sd': phi_samples.std(),
            'phi_q025': np.percentile(phi_samples, 2.5),
            'phi_q975': np.percentile(phi_samples, 97.5),
            'alpha_mean': alpha_samples.mean(),
            'alpha_sd': alpha_samples.std(),
            'beta_mean': beta_samples.mean(),
            'beta_sd': beta_samples.std(),
            'mu_rank': compute_rank(mu_true, mu_samples),
            'phi_rank': compute_rank(phi_true, phi_samples),
            'max_rhat': max_rhat,
            'converged': converged
        }

        print(f"✓ (mu: {mu_true:.3f}, phi: {phi_true:.3f}, Rhat: {max_rhat:.3f})")
        return results

    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        return None

def main():
    """Run SBC analysis"""

    print("="*70)
    print("SIMULATION-BASED CALIBRATION FOR BETA-BINOMIAL MODEL")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - SBC iterations: {N_SBC_ITERATIONS}")
    print(f"  - MCMC chains: {N_CHAINS}")
    print(f"  - MCMC samples: {N_SAMPLES} (tune: {N_TUNE})")
    print(f"  - Number of trials: {N_TRIALS}")
    print(f"  - Sample sizes: {N_VALUES}")
    print(f"\nPriors:")
    print(f"  - mu ~ Beta(2, 25)")
    print(f"  - phi ~ Gamma(2, 2)")
    print(f"\n{'='*70}\n")

    # Run SBC iterations
    print(f"Running {N_SBC_ITERATIONS} SBC iterations...")
    print("-"*70)

    results_list = []
    for i in range(N_SBC_ITERATIONS):
        result = run_sbc_iteration(i, N_VALUES)
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
    mu_coverage = np.mean((results_df['mu_true'] >= results_df['mu_q025']) &
                          (results_df['mu_true'] <= results_df['mu_q975']))
    phi_coverage = np.mean((results_df['phi_true'] >= results_df['phi_q025']) &
                           (results_df['phi_true'] <= results_df['phi_q975']))

    print(f"\nCoverage Rates (95% credible intervals):")
    print(f"  - mu:  {mu_coverage:.3f} (target: 0.950)")
    print(f"  - phi: {phi_coverage:.3f} (target: 0.950)")

    # Bias
    mu_bias = (results_df['mu_mean'] - results_df['mu_true']).mean()
    phi_bias = (results_df['phi_mean'] - results_df['phi_true']).mean()

    print(f"\nBias (posterior mean - true value):")
    print(f"  - mu:  {mu_bias:.4f} (target: ~0)")
    print(f"  - phi: {phi_bias:.4f} (target: ~0)")

    # RMSE
    mu_rmse = np.sqrt(np.mean((results_df['mu_mean'] - results_df['mu_true'])**2))
    phi_rmse = np.sqrt(np.mean((results_df['phi_mean'] - results_df['phi_true'])**2))

    print(f"\nRMSE (root mean squared error):")
    print(f"  - mu:  {mu_rmse:.4f}")
    print(f"  - phi: {phi_rmse:.4f}")

    # Prior vs posterior uncertainty
    prior_mu_sd = np.sqrt(2 * 25 / ((2 + 25)**2 * (2 + 25 + 1)))  # Beta variance
    prior_phi_sd = np.sqrt(2 / 2**2)  # Gamma variance

    posterior_mu_sd = results_df['mu_sd'].mean()
    posterior_phi_sd = results_df['phi_sd'].mean()

    print(f"\nPosterior Contraction (posterior SD / prior SD):")
    print(f"  - mu:  {posterior_mu_sd:.4f} / {prior_mu_sd:.4f} = {posterior_mu_sd/prior_mu_sd:.3f}")
    print(f"  - phi: {posterior_phi_sd:.4f} / {prior_phi_sd:.4f} = {posterior_phi_sd/prior_phi_sd:.3f}")

    # Convergence
    converged_rate = results_df['converged'].mean()
    print(f"\nConvergence Rate: {converged_rate:.3f}")

    # Rank statistics (should be uniform)
    n_posterior_samples = N_CHAINS * N_SAMPLES
    print(f"\nRank Statistics (uniformity test):")

    # Chi-square test for uniformity
    n_bins = 20
    mu_ranks = results_df['mu_rank'].values
    phi_ranks = results_df['phi_rank'].values

    mu_hist, _ = np.histogram(mu_ranks, bins=n_bins, range=(0, n_posterior_samples))
    phi_hist, _ = np.histogram(phi_ranks, bins=n_bins, range=(0, n_posterior_samples))

    expected_count = len(mu_ranks) / n_bins
    mu_chisq = np.sum((mu_hist - expected_count)**2 / expected_count)
    phi_chisq = np.sum((phi_hist - expected_count)**2 / expected_count)

    mu_pvalue = 1 - stats.chi2.cdf(mu_chisq, n_bins - 1)
    phi_pvalue = 1 - stats.chi2.cdf(phi_chisq, n_bins - 1)

    print(f"  - mu ranks:  χ² = {mu_chisq:.2f}, p = {mu_pvalue:.3f}")
    print(f"  - phi ranks: χ² = {phi_chisq:.2f}, p = {phi_pvalue:.3f}")
    print(f"  (p > 0.05 suggests uniform ranks → good calibration)")

    # Save summary
    summary = {
        'n_iterations': len(results_list),
        'success_rate': len(results_list) / N_SBC_ITERATIONS,
        'convergence_rate': float(converged_rate),
        'coverage': {
            'mu': float(mu_coverage),
            'phi': float(phi_coverage)
        },
        'bias': {
            'mu': float(mu_bias),
            'phi': float(phi_bias)
        },
        'rmse': {
            'mu': float(mu_rmse),
            'phi': float(phi_rmse)
        },
        'contraction': {
            'mu': float(posterior_mu_sd / prior_mu_sd),
            'phi': float(posterior_phi_sd / prior_phi_sd)
        },
        'rank_uniformity': {
            'mu_chisq': float(mu_chisq),
            'mu_pvalue': float(mu_pvalue),
            'phi_chisq': float(phi_chisq),
            'phi_pvalue': float(phi_pvalue)
        }
    }

    with open(RESULTS_DIR / 'sbc_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {RESULTS_DIR / 'sbc_summary.json'}")
    print("\n" + "="*70)

    return results_df, summary

if __name__ == "__main__":
    results_df, summary = main()
