"""
Prior Predictive Simulation for Experiment 1: Fixed Changepoint Negative Binomial Regression

This script generates synthetic datasets from the prior distribution to assess:
1. Whether priors generate scientifically plausible data
2. Whether priors allow observed patterns
3. Computational stability of the model specification
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Model Constants (from specification)
# ============================================================================
CHANGEPOINT_INDEX = 17  # Observation index (1-indexed becomes 16 in 0-indexed)
CHANGEPOINT_YEAR = -0.213849730692075  # Standardized year at changepoint
N_OBS = 40
N_SIMULATIONS = 1000

# Load observed data to get exact year values
data = pd.read_csv('/workspace/data/data.csv')
YEARS = data['year'].values
OBSERVED_C = data['C'].values

print("=" * 80)
print("PRIOR PREDICTIVE SIMULATION")
print("=" * 80)
print(f"Number of simulations: {N_SIMULATIONS}")
print(f"Observations per simulation: {N_OBS}")
print(f"Changepoint: observation {CHANGEPOINT_INDEX} (year = {CHANGEPOINT_YEAR:.3f})")
print(f"Year range: [{YEARS.min():.3f}, {YEARS.max():.3f}]")
print(f"Observed C range: [{OBSERVED_C.min()}, {OBSERVED_C.max()}]")
print("=" * 80)

# ============================================================================
# Prior Distributions
# ============================================================================
def sample_priors(n_samples):
    """
    Sample from prior distributions.

    Returns:
        Dictionary with arrays of prior samples for each parameter
    """
    priors = {
        'beta_0': np.random.normal(4.3, 0.5, n_samples),      # Intercept
        'beta_1': np.random.normal(0.35, 0.3, n_samples),     # Pre-break slope
        'beta_2': np.random.normal(0.85, 0.5, n_samples),     # Post-break increase
        'alpha': np.random.gamma(2, 1/3, n_samples),          # NB dispersion (shape, scale)
        'rho': np.random.beta(8, 2, n_samples),               # AR(1) coefficient
        'sigma_eps': np.random.exponential(0.5, n_samples)    # AR(1) noise std
    }
    return priors

# ============================================================================
# AR(1) Error Process
# ============================================================================
def generate_ar1_errors(rho, sigma_eps, n_obs):
    """
    Generate AR(1) error process: ε_t = ρ × ε_{t-1} + η_t, η_t ~ N(0, σ_ε)

    Initial condition: ε_1 ~ N(0, σ_ε / sqrt(1 - ρ²)) for stationarity

    Args:
        rho: AR(1) coefficient
        sigma_eps: Standard deviation of innovations
        n_obs: Number of observations

    Returns:
        Array of AR(1) errors
    """
    eps = np.zeros(n_obs)

    # Stationary initial condition
    if rho < 1.0:
        eps[0] = np.random.normal(0, sigma_eps / np.sqrt(1 - rho**2))
    else:
        # If rho >= 1, use sigma_eps (non-stationary, but avoid numerical issues)
        eps[0] = np.random.normal(0, sigma_eps)

    # Generate remaining errors
    for t in range(1, n_obs):
        eps[t] = rho * eps[t-1] + np.random.normal(0, sigma_eps)

    return eps

# ============================================================================
# Model: log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > τ) × (year_t - year_τ) + ε_t
# ============================================================================
def compute_log_mean(beta_0, beta_1, beta_2, years, changepoint_idx, changepoint_year, eps):
    """
    Compute log mean for each observation according to the model.

    Args:
        beta_0: Intercept
        beta_1: Pre-break slope
        beta_2: Post-break slope increase
        years: Array of standardized year values
        changepoint_idx: Index of changepoint (0-indexed)
        changepoint_year: Year value at changepoint
        eps: AR(1) errors

    Returns:
        log_mu: Log of expected count at each time point
    """
    n_obs = len(years)
    log_mu = np.zeros(n_obs)

    for t in range(n_obs):
        # Base term
        log_mu[t] = beta_0 + beta_1 * years[t]

        # Post-changepoint term (note: changepoint_idx is 0-indexed)
        if t >= changepoint_idx:
            log_mu[t] += beta_2 * (years[t] - changepoint_year)

        # AR(1) error
        log_mu[t] += eps[t]

    return log_mu

# ============================================================================
# Negative Binomial Sampling
# ============================================================================
def sample_negative_binomial(mu, alpha):
    """
    Sample from Negative Binomial distribution.

    Parameterization: NB(μ, α) where:
    - μ = mean
    - α = dispersion parameter
    - variance = μ + α × μ²

    SciPy uses NB(n, p) where n = 1/α, p = 1/(1 + α×μ)

    Args:
        mu: Mean parameter (array)
        alpha: Dispersion parameter (scalar)

    Returns:
        Array of count samples
    """
    # Handle edge cases
    mu = np.maximum(mu, 1e-10)  # Avoid mu = 0
    alpha = np.maximum(alpha, 1e-10)  # Avoid alpha = 0

    # Convert to scipy parameterization
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)

    # Ensure p is in valid range [0, 1]
    p = np.clip(p, 1e-10, 1 - 1e-10)

    # Sample
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        counts = stats.nbinom.rvs(n, p)

    return counts

# ============================================================================
# Main Simulation Loop
# ============================================================================
def run_prior_predictive_simulation(n_sims):
    """
    Run prior predictive simulations.

    Returns:
        prior_samples: Dictionary of parameter samples
        simulated_datasets: Array of shape (n_sims, n_obs) with simulated counts
        summary_stats: Dictionary of summary statistics for each simulation
        computational_flags: Dictionary tracking computational issues
    """
    # Sample priors
    print("\n" + "=" * 80)
    print("SAMPLING FROM PRIORS")
    print("=" * 80)
    prior_samples = sample_priors(n_sims)

    # Print prior summaries
    for param, values in prior_samples.items():
        print(f"{param:12s}: mean={np.mean(values):7.3f}, std={np.std(values):7.3f}, "
              f"range=[{np.min(values):7.3f}, {np.max(values):7.3f}]")

    # Storage
    simulated_datasets = np.zeros((n_sims, N_OBS))
    summary_stats = {
        'min_count': np.zeros(n_sims),
        'max_count': np.zeros(n_sims),
        'mean_count': np.zeros(n_sims),
        'median_count': np.zeros(n_sims),
        'variance': np.zeros(n_sims),
        'variance_mean_ratio': np.zeros(n_sims),
        'acf_lag1': np.zeros(n_sims),
        'growth_factor': np.zeros(n_sims),  # final / initial
        'pre_break_slope': np.zeros(n_sims),  # empirical slope
        'post_break_slope': np.zeros(n_sims),  # empirical slope
        'total_slope': np.zeros(n_sims),  # empirical slope (log scale)
    }

    computational_flags = {
        'overflow_warnings': 0,
        'negative_mu': 0,
        'extreme_counts': 0,  # counts > 10000
        'zero_inflation': 0,  # >50% zeros
        'nan_values': 0
    }

    # Progress reporting
    print("\n" + "=" * 80)
    print("RUNNING SIMULATIONS")
    print("=" * 80)

    for i in range(n_sims):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{n_sims} simulations complete")

        # Get parameters for this simulation
        beta_0 = prior_samples['beta_0'][i]
        beta_1 = prior_samples['beta_1'][i]
        beta_2 = prior_samples['beta_2'][i]
        alpha = prior_samples['alpha'][i]
        rho = prior_samples['rho'][i]
        sigma_eps = prior_samples['sigma_eps'][i]

        # Generate AR(1) errors
        eps = generate_ar1_errors(rho, sigma_eps, N_OBS)

        # Compute log means
        log_mu = compute_log_mean(beta_0, beta_1, beta_2, YEARS,
                                   CHANGEPOINT_INDEX - 1,  # Convert to 0-indexed
                                   CHANGEPOINT_YEAR, eps)

        # Check for computational issues
        if np.any(log_mu > 10):  # exp(10) ≈ 22000
            computational_flags['overflow_warnings'] += 1

        # Convert to mu
        mu = np.exp(log_mu)

        if np.any(mu < 0):
            computational_flags['negative_mu'] += 1
            mu = np.maximum(mu, 1e-10)

        # Sample counts from Negative Binomial
        counts = sample_negative_binomial(mu, alpha)

        # Check for computational issues
        if np.any(np.isnan(counts)):
            computational_flags['nan_values'] += 1
            counts = np.nan_to_num(counts, nan=0)

        if np.any(counts > 10000):
            computational_flags['extreme_counts'] += 1

        if np.mean(counts == 0) > 0.5:
            computational_flags['zero_inflation'] += 1

        # Store simulated data
        simulated_datasets[i, :] = counts

        # Compute summary statistics
        summary_stats['min_count'][i] = np.min(counts)
        summary_stats['max_count'][i] = np.max(counts)
        summary_stats['mean_count'][i] = np.mean(counts)
        summary_stats['median_count'][i] = np.median(counts)
        summary_stats['variance'][i] = np.var(counts)
        summary_stats['variance_mean_ratio'][i] = np.var(counts) / (np.mean(counts) + 1e-10)

        # ACF(1)
        if np.std(counts) > 0:
            acf1 = np.corrcoef(counts[:-1], counts[1:])[0, 1]
            summary_stats['acf_lag1'][i] = acf1 if not np.isnan(acf1) else 0
        else:
            summary_stats['acf_lag1'][i] = 0

        # Growth factor
        initial_mean = np.mean(counts[:5])
        final_mean = np.mean(counts[-5:])
        summary_stats['growth_factor'][i] = final_mean / (initial_mean + 1e-10)

        # Empirical slopes (on log scale)
        log_counts = np.log(counts + 1)  # Add 1 to avoid log(0)

        # Pre-break slope
        pre_idx = CHANGEPOINT_INDEX - 1
        if pre_idx > 1:
            pre_slope, _ = np.polyfit(YEARS[:pre_idx], log_counts[:pre_idx], 1)
            summary_stats['pre_break_slope'][i] = pre_slope

        # Post-break slope
        post_idx = CHANGEPOINT_INDEX - 1
        if post_idx < N_OBS - 1:
            post_slope, _ = np.polyfit(YEARS[post_idx:], log_counts[post_idx:], 1)
            summary_stats['post_break_slope'][i] = post_slope

        # Total slope
        total_slope, _ = np.polyfit(YEARS, log_counts, 1)
        summary_stats['total_slope'][i] = total_slope

    print(f"\nAll {n_sims} simulations complete!")

    return prior_samples, simulated_datasets, summary_stats, computational_flags

# ============================================================================
# Execute Simulation
# ============================================================================
if __name__ == "__main__":
    prior_samples, simulated_datasets, summary_stats, comp_flags = \
        run_prior_predictive_simulation(N_SIMULATIONS)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save prior samples
    prior_df = pd.DataFrame(prior_samples)
    prior_df.to_csv('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.csv',
                     index=False)
    print("Prior samples saved to: prior_samples.csv")

    # Save simulated datasets (sample of 100 for file size)
    sim_sample_df = pd.DataFrame(simulated_datasets[:100, :],
                                  columns=[f't_{i+1}' for i in range(N_OBS)])
    sim_sample_df.to_csv('/workspace/experiments/experiment_1/prior_predictive_check/code/simulated_datasets_sample.csv',
                          index=False)
    print("Simulated datasets (sample) saved to: simulated_datasets_sample.csv")

    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('/workspace/experiments/experiment_1/prior_predictive_check/code/summary_statistics.csv',
                       index=False)
    print("Summary statistics saved to: summary_statistics.csv")

    # Save full simulated datasets as numpy array
    np.save('/workspace/experiments/experiment_1/prior_predictive_check/code/simulated_datasets_full.npy',
            simulated_datasets)
    print("Full simulated datasets saved to: simulated_datasets_full.npy")

    # Print computational flags
    print("\n" + "=" * 80)
    print("COMPUTATIONAL FLAGS")
    print("=" * 80)
    for flag, count in comp_flags.items():
        pct = 100 * count / N_SIMULATIONS
        print(f"{flag:20s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
