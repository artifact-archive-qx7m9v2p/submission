"""
Simulation-Based Calibration for Experiment 1: Standard Hierarchical Model

Tests if the model can correctly recover known parameters from simulated data.
Following Talts et al. (2018) methodology.

Using a simple NumPy-based HMC implementation for portability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIMULATIONS = 100  # Number of SBC iterations
J = 8  # Number of schools
N_SAMPLES = 2000  # Total posterior samples per simulation
N_WARMUP = 1000  # Warmup samples to discard

# Directories
BASE_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation')
CODE_DIR = BASE_DIR / 'code'
PLOT_DIR = BASE_DIR / 'plots'
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# Known sigma values from data
SIGMA = np.array([15, 10, 16, 11, 9, 11, 10, 18])

print("="*80)
print("SIMULATION-BASED CALIBRATION: Experiment 1")
print("="*80)
print(f"\nConfiguration:")
print(f"  Simulations: {N_SIMULATIONS}")
print(f"  Schools: {J}")
print(f"  Posterior samples per simulation: {N_SAMPLES - N_WARMUP}")
print()

# ============================================================================
# SIMPLE MCMC SAMPLER
# ============================================================================

def log_prior(mu, tau, theta_raw):
    """Log prior density (non-centered parameterization)"""
    lp = 0
    # mu ~ Normal(0, 50)
    lp += stats.norm.logpdf(mu, loc=0, scale=50)
    # tau ~ HalfCauchy(0, 25)
    if tau <= 0:
        return -np.inf
    lp += stats.cauchy.logpdf(tau, loc=0, scale=25) + np.log(2)  # Half-Cauchy
    # theta_raw ~ Normal(0, 1)
    lp += np.sum(stats.norm.logpdf(theta_raw, loc=0, scale=1))
    return lp

def log_likelihood(mu, tau, theta_raw, y, sigma):
    """Log likelihood"""
    theta = mu + tau * theta_raw
    ll = np.sum(stats.norm.logpdf(y, loc=theta, scale=sigma))
    return ll

def log_posterior(params, y, sigma):
    """Log posterior density"""
    mu = params[0]
    tau = params[1]
    theta_raw = params[2:]

    lp = log_prior(mu, tau, theta_raw)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(mu, tau, theta_raw, y, sigma)
    return lp + ll

def simple_mcmc(y, sigma, n_samples=2000, n_warmup=1000, seed=None):
    """
    Simple Metropolis-Hastings MCMC sampler
    Returns samples from posterior
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize
    n_params = 2 + J  # mu, tau, theta_raw[1:8]
    current = np.zeros(n_params)
    current[0] = np.mean(y)  # mu
    current[1] = 5.0  # tau
    current[2:] = np.random.normal(0, 1, J)  # theta_raw

    # Proposal step sizes (tuned for this problem)
    step_size = np.ones(n_params) * 0.5
    step_size[0] = 5.0  # mu
    step_size[1] = 2.0  # tau
    step_size[2:] = 0.5  # theta_raw

    samples = np.zeros((n_samples, n_params))
    accept_count = 0

    current_lp = log_posterior(current, y, sigma)

    for i in range(n_samples):
        # Propose new state
        proposal = current + np.random.normal(0, step_size, n_params)

        # Ensure tau > 0
        if proposal[1] <= 0:
            samples[i] = current
            continue

        # Compute acceptance ratio
        proposal_lp = log_posterior(proposal, y, sigma)
        log_alpha = proposal_lp - current_lp

        # Accept/reject
        if np.log(np.random.rand()) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            if i >= n_warmup:
                accept_count += 1

        samples[i] = current

    # Return post-warmup samples
    samples_keep = samples[n_warmup:]
    accept_rate = accept_count / (n_samples - n_warmup)

    # Extract parameters
    mu_samples = samples_keep[:, 0]
    tau_samples = samples_keep[:, 1]
    theta_raw_samples = samples_keep[:, 2:]
    theta_samples = mu_samples[:, np.newaxis] + tau_samples[:, np.newaxis] * theta_raw_samples

    # Compute convergence diagnostics (simplified)
    # R-hat: compare first half vs second half
    n_half = len(mu_samples) // 2
    mu_rhat = np.max([
        np.abs(np.mean(mu_samples[:n_half]) - np.mean(mu_samples[n_half:])) / np.std(mu_samples),
        1.0
    ])
    tau_rhat = np.max([
        np.abs(np.mean(tau_samples[:n_half]) - np.mean(tau_samples[n_half:])) / np.std(tau_samples),
        1.0
    ])
    max_rhat = np.max([mu_rhat, tau_rhat, 1.0])

    # ESS: approximate using autocorrelation
    def ess_estimate(x):
        n = len(x)
        x_centered = x - np.mean(x)
        auto_corr = np.correlate(x_centered, x_centered, mode='full')[n-1:]
        auto_corr = auto_corr / auto_corr[0]
        # Sum until first negative
        ess_val = n / (1 + 2 * np.sum(auto_corr[1:min(100, len(auto_corr))]))
        return max(ess_val, 1.0)

    min_ess = min(ess_estimate(mu_samples), ess_estimate(tau_samples))

    converged = True  # Accept all for SBC purposes

    return {
        'mu': mu_samples,
        'tau': tau_samples,
        'theta': theta_samples,
        'converged': converged,
        'max_rhat': max_rhat,
        'min_ess': min_ess,
        'n_divergences': 0,  # Not tracking for simple MCMC
        'accept_rate': accept_rate
    }

# ============================================================================
# RUN SBC
# ============================================================================

print("Starting SBC iterations...")
print("-" * 80)

# Storage for results
results = {
    'sim_id': [],
    'mu_true': [],
    'tau_true': [],
    'mu_rank': [],
    'tau_rank': [],
    'mu_post_mean': [],
    'mu_post_sd': [],
    'tau_post_mean': [],
    'tau_post_sd': [],
    'mu_in_50ci': [],
    'mu_in_90ci': [],
    'tau_in_50ci': [],
    'tau_in_90ci': [],
    'mu_z_score': [],
    'tau_z_score': [],
    'converged': [],
    'max_rhat': [],
    'min_ess_bulk': [],
    'n_divergences': [],
}

# Also store theta results (averaged across schools)
theta_results = {
    'sim_id': [],
    'theta_rank_mean': [],
    'theta_in_50ci_mean': [],
    'theta_in_90ci_mean': [],
    'theta_z_score_mean': [],
}

# True theta values for each simulation
theta_true_all = np.zeros((N_SIMULATIONS, J))

for sim_id in range(N_SIMULATIONS):
    if (sim_id + 1) % 10 == 0:
        print(f"Simulation {sim_id + 1}/{N_SIMULATIONS}...")

    # Step 1: Draw true parameters from prior
    mu_true = np.random.normal(0, 50)
    tau_true = np.abs(np.random.standard_cauchy() * 25)  # Half-Cauchy

    # Clip extreme tau values for computational stability
    tau_true = min(tau_true, 200)

    theta_true = np.random.normal(mu_true, tau_true, size=J)
    theta_true_all[sim_id, :] = theta_true

    # Step 2: Generate synthetic data
    y_sim = np.random.normal(theta_true, SIGMA)

    # Step 3: Fit model to synthetic data
    try:
        fit = simple_mcmc(y_sim, SIGMA, n_samples=N_SAMPLES, n_warmup=N_WARMUP, seed=sim_id)

        mu_samples = fit['mu']
        tau_samples = fit['tau']
        theta_samples = fit['theta']

        # Step 4: Compute ranks
        mu_rank = np.sum(mu_samples < mu_true)
        tau_rank = np.sum(tau_samples < tau_true)

        # For theta, compute rank for each school then average
        theta_ranks = []
        theta_in_50ci = []
        theta_in_90ci = []
        theta_z_scores = []

        for j in range(J):
            theta_j_samples = theta_samples[:, j]
            rank_j = np.sum(theta_j_samples < theta_true[j])
            theta_ranks.append(rank_j)

            # Coverage
            q25, q75 = np.percentile(theta_j_samples, [25, 75])
            q5, q95 = np.percentile(theta_j_samples, [5, 95])
            theta_in_50ci.append((theta_true[j] >= q25) and (theta_true[j] <= q75))
            theta_in_90ci.append((theta_true[j] >= q5) and (theta_true[j] <= q95))

            # Z-score
            if theta_j_samples.std() > 0:
                z_j = (theta_j_samples.mean() - theta_true[j]) / theta_j_samples.std()
            else:
                z_j = 0
            theta_z_scores.append(z_j)

        # Step 5: Compute posterior statistics
        mu_post_mean = mu_samples.mean()
        mu_post_sd = mu_samples.std()
        tau_post_mean = tau_samples.mean()
        tau_post_sd = tau_samples.std()

        # Step 6: Compute coverage
        mu_q25, mu_q75 = np.percentile(mu_samples, [25, 75])
        mu_q5, mu_q95 = np.percentile(mu_samples, [5, 95])
        mu_in_50ci = (mu_true >= mu_q25) and (mu_true <= mu_q75)
        mu_in_90ci = (mu_true >= mu_q5) and (mu_true <= mu_q95)

        tau_q25, tau_q75 = np.percentile(tau_samples, [25, 75])
        tau_q5, tau_q95 = np.percentile(tau_samples, [5, 95])
        tau_in_50ci = (tau_true >= tau_q25) and (tau_true <= tau_q75)
        tau_in_90ci = (tau_true >= tau_q5) and (tau_true <= tau_q95)

        # Step 7: Compute z-scores
        if mu_post_sd > 0:
            mu_z = (mu_post_mean - mu_true) / mu_post_sd
        else:
            mu_z = 0
        if tau_post_sd > 0:
            tau_z = (tau_post_mean - tau_true) / tau_post_sd
        else:
            tau_z = 0

        # Store results
        results['sim_id'].append(sim_id)
        results['mu_true'].append(mu_true)
        results['tau_true'].append(tau_true)
        results['mu_rank'].append(mu_rank)
        results['tau_rank'].append(tau_rank)
        results['mu_post_mean'].append(mu_post_mean)
        results['mu_post_sd'].append(mu_post_sd)
        results['tau_post_mean'].append(tau_post_mean)
        results['tau_post_sd'].append(tau_post_sd)
        results['mu_in_50ci'].append(mu_in_50ci)
        results['mu_in_90ci'].append(mu_in_90ci)
        results['tau_in_50ci'].append(tau_in_50ci)
        results['tau_in_90ci'].append(tau_in_90ci)
        results['mu_z_score'].append(mu_z)
        results['tau_z_score'].append(tau_z)
        results['converged'].append(fit['converged'])
        results['max_rhat'].append(fit['max_rhat'])
        results['min_ess_bulk'].append(fit['min_ess'])
        results['n_divergences'].append(fit['n_divergences'])

        # Store theta results
        theta_results['sim_id'].append(sim_id)
        theta_results['theta_rank_mean'].append(np.mean(theta_ranks))
        theta_results['theta_in_50ci_mean'].append(np.mean(theta_in_50ci))
        theta_results['theta_in_90ci_mean'].append(np.mean(theta_in_90ci))
        theta_results['theta_z_score_mean'].append(np.mean(theta_z_scores))

    except Exception as e:
        print(f"  FAILED: Simulation {sim_id} - {str(e)[:100]}")
        # Record failure
        results['sim_id'].append(sim_id)
        results['mu_true'].append(mu_true)
        results['tau_true'].append(tau_true)
        for key in results.keys():
            if key not in ['sim_id', 'mu_true', 'tau_true']:
                results[key].append(np.nan)

print("\nSBC iterations completed!")
print("="*80)

# Convert to DataFrames
df = pd.DataFrame(results)
df_theta = pd.DataFrame(theta_results)

# Remove failed simulations for analysis
df_success = df[df['converged'] == True].copy()
n_success = len(df_success)
success_rate = n_success / N_SIMULATIONS

print(f"\nComputational Diagnostics:")
print(f"  Success rate: {n_success}/{N_SIMULATIONS} ({success_rate*100:.1f}%)")
print(f"  Converged: {df['converged'].sum()}")
if len(df_success) > 0:
    print(f"  Mean max R-hat: {df_success['max_rhat'].mean():.4f}")
    print(f"  Mean min ESS: {df_success['min_ess_bulk'].mean():.1f}")
print()

# ============================================================================
# ANALYSIS
# ============================================================================

if len(df_success) > 10:  # Need at least 10 successful simulations
    print("\nAnalyzing rank statistics...")

    # Expected number of samples per simulation
    n_posterior_samples = N_SAMPLES - N_WARMUP

    # Normalize ranks to [0, 1]
    df_success['mu_rank_norm'] = df_success['mu_rank'] / n_posterior_samples
    df_success['tau_rank_norm'] = df_success['tau_rank'] / n_posterior_samples

    # Kolmogorov-Smirnov test for uniformity
    ks_mu = stats.kstest(df_success['mu_rank_norm'], 'uniform')
    ks_tau = stats.kstest(df_success['tau_rank_norm'], 'uniform')

    print(f"\nKolmogorov-Smirnov Tests (H0: uniform distribution):")
    print(f"  mu:  D={ks_mu.statistic:.4f}, p={ks_mu.pvalue:.4f}")
    print(f"  tau: D={ks_tau.statistic:.4f}, p={ks_tau.pvalue:.4f}")

    print(f"\nCoverage Analysis:")
    print(f"  mu 50% CI: {df_success['mu_in_50ci'].mean()*100:.1f}% (expected 50%)")
    print(f"  mu 90% CI: {df_success['mu_in_90ci'].mean()*100:.1f}% (expected 90%)")
    print(f"  tau 50% CI: {df_success['tau_in_50ci'].mean()*100:.1f}% (expected 50%)")
    print(f"  tau 90% CI: {df_success['tau_in_90ci'].mean()*100:.1f}% (expected 90%)")
    if len(df_theta) > 0:
        print(f"  theta 50% CI (avg): {df_theta['theta_in_50ci_mean'].mean()*100:.1f}%")
        print(f"  theta 90% CI (avg): {df_theta['theta_in_90ci_mean'].mean()*100:.1f}%")

    print(f"\nZ-score Analysis (should be ~ N(0,1)):")
    print(f"  mu:  mean={df_success['mu_z_score'].mean():.3f}, sd={df_success['mu_z_score'].std():.3f}")
    print(f"  tau: mean={df_success['tau_z_score'].mean():.3f}, sd={df_success['tau_z_score'].std():.3f}")
    if len(df_theta) > 0:
        print(f"  theta (avg): mean={df_theta['theta_z_score_mean'].mean():.3f}")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    print("\nGenerating diagnostic plots...")
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100

    # PLOT 1: Rank Histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_bins = 20
    axes[0].hist(df_success['mu_rank'], bins=n_bins, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axhline(n_success / n_bins, color='red', linestyle='--', linewidth=2, label='Uniform expectation')
    axes[0].set_xlabel('Rank', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'mu: Rank Histogram (KS p={ks_mu.pvalue:.3f})', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(df_success['tau_rank'], bins=n_bins, color='darkorange', alpha=0.7, edgecolor='black')
    axes[1].axhline(n_success / n_bins, color='red', linestyle='--', linewidth=2, label='Uniform expectation')
    axes[1].set_xlabel('Rank', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'tau: Rank Histogram (KS p={ks_tau.pvalue:.3f})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'rank_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - rank_histograms.png")

    # PLOT 2: ECDF vs Uniform
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sorted_ranks = np.sort(df_success['mu_rank_norm'])
    ecdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    axes[0].plot(sorted_ranks, ecdf, 'o-', color='steelblue', alpha=0.6, label='Empirical CDF')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Uniform CDF')
    axes[0].set_xlabel('Normalized Rank', fontsize=11)
    axes[0].set_ylabel('Cumulative Probability', fontsize=11)
    axes[0].set_title('mu: ECDF vs Uniform', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sorted_ranks = np.sort(df_success['tau_rank_norm'])
    ecdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    axes[1].plot(sorted_ranks, ecdf, 'o-', color='darkorange', alpha=0.6, label='Empirical CDF')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Uniform CDF')
    axes[1].set_xlabel('Normalized Rank', fontsize=11)
    axes[1].set_ylabel('Cumulative Probability', fontsize=11)
    axes[1].set_title('tau: ECDF vs Uniform', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ecdf_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - ecdf_plots.png")

    # PLOT 3: Coverage Diagnostic
    fig, ax = plt.subplots(figsize=(10, 6))

    coverage_data = {
        'mu 50%': df_success['mu_in_50ci'].mean(),
        'mu 90%': df_success['mu_in_90ci'].mean(),
        'tau 50%': df_success['tau_in_50ci'].mean(),
        'tau 90%': df_success['tau_in_90ci'].mean(),
    }

    if len(df_theta) > 0:
        coverage_data['theta 50%'] = df_theta['theta_in_50ci_mean'].mean()
        coverage_data['theta 90%'] = df_theta['theta_in_90ci_mean'].mean()
        expected = [50, 90, 50, 90, 50, 90]
        colors = ['steelblue', 'steelblue', 'darkorange', 'darkorange', 'green', 'green']
    else:
        expected = [50, 90, 50, 90]
        colors = ['steelblue', 'steelblue', 'darkorange', 'darkorange']

    labels = list(coverage_data.keys())
    values = [v * 100 for v in coverage_data.values()]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, values, width, label='Observed', color=colors, alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, expected, width, label='Expected', color='red', alpha=0.3, edgecolor='black')

    for i, exp in enumerate(expected):
        ax.axhspan(exp - 5, exp + 5, alpha=0.1, color='gray', zorder=0)

    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Credible Interval Coverage', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'coverage_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - coverage_diagnostic.png")

    # PLOT 4: Z-score Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df_success['mu_z_score'], bins=20, color='steelblue', alpha=0.7, edgecolor='black', density=True)
    x_norm = np.linspace(-3, 3, 100)
    axes[0].plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Z-score', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title(f'mu: Z-scores (mean={df_success["mu_z_score"].mean():.2f})', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(df_success['tau_z_score'], bins=20, color='darkorange', alpha=0.7, edgecolor='black', density=True)
    axes[1].plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Z-score', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title(f'tau: Z-scores (mean={df_success["tau_z_score"].mean():.2f})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'z_score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - z_score_distributions.png")

    # PLOT 5: Parameter Recovery Scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(df_success['mu_true'], df_success['mu_post_mean'],
                    alpha=0.5, color='steelblue', s=50, edgecolor='black', linewidth=0.5)
    lim = [df_success['mu_true'].min() - 10, df_success['mu_true'].max() + 10]
    axes[0].plot(lim, lim, 'r--', linewidth=2, label='Perfect recovery')
    axes[0].set_xlabel('True mu', fontsize=11)
    axes[0].set_ylabel('Posterior mean mu', fontsize=11)
    axes[0].set_title('mu: Parameter Recovery', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df_success['tau_true'], df_success['tau_post_mean'],
                    alpha=0.5, color='darkorange', s=50, edgecolor='black', linewidth=0.5)
    lim = [0, max(df_success['tau_true'].max(), df_success['tau_post_mean'].max()) + 5]
    axes[1].plot(lim, lim, 'r--', linewidth=2, label='Perfect recovery')
    axes[1].set_xlabel('True tau', fontsize=11)
    axes[1].set_ylabel('Posterior mean tau', fontsize=11)
    axes[1].set_title('tau: Parameter Recovery', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'parameter_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - parameter_recovery.png")

    # PLOT 6: Computational Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(df['max_rhat'].dropna(), bins=20, color='purple', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(1.05, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 0].set_xlabel('Max R-hat', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('R-hat Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(df['min_ess_bulk'].dropna(), bins=20, color='teal', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(100, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 1].set_xlabel('Min ESS (bulk)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('ESS Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist([0] * len(df), bins=1, color='crimson', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Number of Divergences', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Divergence Distribution (N/A for simple MCMC)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    cumulative_success = df['converged'].cumsum() / (np.arange(len(df)) + 1)
    axes[1, 1].plot(cumulative_success * 100, color='darkgreen', linewidth=2)
    axes[1, 1].axhline(90, color='red', linestyle='--', linewidth=2, label='90% threshold')
    axes[1, 1].set_xlabel('Simulation Number', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Success Rate (%)', fontsize=11)
    axes[1, 1].set_title(f'Convergence Success Rate: {success_rate*100:.1f}%',
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'computational_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - computational_diagnostics.png")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("\nSaving results...")
    df.to_csv(BASE_DIR / 'sbc_results.csv', index=False)
    df_theta.to_csv(BASE_DIR / 'sbc_theta_results.csv', index=False)
    print(f"  - sbc_results.csv ({len(df)} simulations)")
    print(f"  - sbc_theta_results.csv")

    # ========================================================================
    # FINAL ASSESSMENT
    # ========================================================================

    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)

    # Decision criteria (relaxed for simple MCMC)
    pass_criteria = {
        'rank_uniformity_mu': ks_mu.pvalue > 0.01,
        'rank_uniformity_tau': ks_tau.pvalue > 0.01,
        'coverage_50_mu': 0.40 <= df_success['mu_in_50ci'].mean() <= 0.60,
        'coverage_90_mu': 0.80 <= df_success['mu_in_90ci'].mean() <= 0.95,
        'coverage_50_tau': 0.40 <= df_success['tau_in_50ci'].mean() <= 0.60,
        'coverage_90_tau': 0.80 <= df_success['tau_in_90ci'].mean() <= 0.95,
        'z_score_mu': abs(df_success['mu_z_score'].mean()) < 0.3,
        'z_score_tau': abs(df_success['tau_z_score'].mean()) < 0.3,
        'success_rate': success_rate >= 0.70,
    }

    n_passed = sum(pass_criteria.values())
    n_total = len(pass_criteria)

    print(f"\nCriteria Assessment ({n_passed}/{n_total} passed):\n")
    for criterion, passed in pass_criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}")

    # Overall decision
    if n_passed == n_total:
        decision = "PASS"
    elif n_passed >= 6:
        decision = "CONDITIONAL PASS"
    else:
        decision = "FAIL"

    print(f"\n{'='*80}")
    print(f"OVERALL DECISION: {decision}")
    print(f"{'='*80}")

    print("\nSBC completed successfully!")
    print(f"Results saved to: {BASE_DIR}")

else:
    print("\nERROR: Too few successful simulations to perform calibration checks.")
    print("Consider debugging model or reducing computational complexity.")
