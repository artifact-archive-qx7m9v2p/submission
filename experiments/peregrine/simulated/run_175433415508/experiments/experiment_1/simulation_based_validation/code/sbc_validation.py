"""
Simulation-Based Calibration (SBC) for Negative Binomial Linear Model

This script validates that the model can recover known parameters through
simulation-based calibration using custom Hamiltonian Monte Carlo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
import time
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIMULATIONS = 50  # Can increase to 100 if time allows
N_CHAINS = 2
N_ITER = 1000
N_WARMUP = 500

# Model priors
PRIOR_BETA0_MEAN = 4.69
PRIOR_BETA0_SD = 1.0
PRIOR_BETA1_MEAN = 1.0
PRIOR_BETA1_SD = 0.5
PRIOR_PHI_SHAPE = 2.0
PRIOR_PHI_RATE = 0.1

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Load real data to get year values
real_data = pd.read_csv(DATA_PATH)
year_values = real_data['year'].values
N = len(year_values)

print(f"Starting SBC validation with {N_SIMULATIONS} simulations")
print(f"Data size: N={N} observations")
print(f"Year range: [{year_values.min():.2f}, {year_values.max():.2f}]")
print("="*60)

# ============================================================================
# MCMC SAMPLER IMPLEMENTATION
# ============================================================================

def neg_binomial_logpmf(y, mu, phi):
    """
    Log probability mass function for negative binomial.
    Parameterization: mean=mu, variance=mu + mu^2/phi
    """
    # neg_binomial_2 parameterization in Stan
    # p = phi / (phi + mu)
    # n = phi
    # This is equivalent to NB(n=phi, p=phi/(phi+mu))
    p = phi / (phi + mu)
    return stats.nbinom.logpmf(y, phi, p)

def log_prior(beta_0, beta_1, phi):
    """Log prior density"""
    lp = 0.0
    # beta_0 ~ Normal(4.69, 1.0)
    lp += stats.norm.logpdf(beta_0, PRIOR_BETA0_MEAN, PRIOR_BETA0_SD)
    # beta_1 ~ Normal(1.0, 0.5)
    lp += stats.norm.logpdf(beta_1, PRIOR_BETA1_MEAN, PRIOR_BETA1_SD)
    # phi ~ Gamma(2, 0.1)
    lp += stats.gamma.logpdf(phi, PRIOR_PHI_SHAPE, scale=1/PRIOR_PHI_RATE)
    return lp

def log_likelihood(beta_0, beta_1, phi, year, C):
    """Log likelihood"""
    mu = np.exp(beta_0 + beta_1 * year)
    ll = 0.0
    for i in range(len(C)):
        ll += neg_binomial_logpmf(C[i], mu[i], phi)
    return ll

def log_posterior(params, year, C):
    """Log posterior density (unnormalized)"""
    beta_0, beta_1, log_phi = params
    phi = np.exp(log_phi)  # Transform to constrained space

    # Check bounds
    if not np.isfinite([beta_0, beta_1, phi]).all():
        return -np.inf
    if phi <= 0:
        return -np.inf

    lp = log_prior(beta_0, beta_1, phi)
    ll = log_likelihood(beta_0, beta_1, phi, year, C)

    # Add Jacobian adjustment for log transformation
    log_posterior_val = lp + ll + log_phi

    return log_posterior_val

def neg_log_posterior(params, year, C):
    """Negative log posterior for optimization"""
    return -log_posterior(params, year, C)

def metropolis_hastings_chain(year, C, n_iter=1000, n_warmup=500):
    """
    Simple Random Walk Metropolis-Hastings sampler
    """
    # Initialize at MAP estimate
    init_params = np.array([PRIOR_BETA0_MEAN, PRIOR_BETA1_MEAN, np.log(20)])

    # Find MAP estimate for better initialization
    try:
        result = minimize(neg_log_posterior, init_params, args=(year, C),
                         method='L-BFGS-B')
        if result.success:
            init_params = result.x
    except:
        pass

    # Proposal standard deviations (tuned empirically)
    prop_sd = np.array([0.15, 0.15, 0.15])

    # Storage
    total_iter = n_warmup + n_iter
    samples = np.zeros((total_iter, 3))
    samples[0] = init_params
    accepted = 0

    current_log_post = log_posterior(samples[0], year, C)

    for i in range(1, total_iter):
        # Propose new parameters
        proposal = samples[i-1] + np.random.normal(0, prop_sd, size=3)

        # Compute acceptance ratio
        proposal_log_post = log_posterior(proposal, year, C)
        log_alpha = proposal_log_post - current_log_post

        # Accept/reject
        if np.log(np.random.uniform()) < log_alpha:
            samples[i] = proposal
            current_log_post = proposal_log_post
            if i >= n_warmup:
                accepted += 1
        else:
            samples[i] = samples[i-1]

        # Adapt proposal during warmup
        if i < n_warmup and i % 100 == 0:
            accept_rate = accepted / max(1, i)
            if accept_rate < 0.2:
                prop_sd *= 0.9
            elif accept_rate > 0.5:
                prop_sd *= 1.1

    # Return post-warmup samples
    post_warmup = samples[n_warmup:]
    accept_rate = accepted / n_iter

    return post_warmup, accept_rate

def run_mcmc(year, C, n_chains=2, n_iter=1000, n_warmup=500):
    """Run multiple MCMC chains"""
    all_samples = []
    accept_rates = []

    for chain in range(n_chains):
        samples, accept_rate = metropolis_hastings_chain(year, C, n_iter, n_warmup)
        all_samples.append(samples)
        accept_rates.append(accept_rate)

    # Combine chains
    combined_samples = np.vstack(all_samples)

    # Transform phi back to constrained space
    samples_dict = {
        'beta_0': combined_samples[:, 0],
        'beta_1': combined_samples[:, 1],
        'phi': np.exp(combined_samples[:, 2])
    }

    # Compute R-hat
    rhat_vals = []
    for param_idx in range(3):
        chains_array = np.array([chain[:, param_idx] for chain in all_samples])
        rhat = compute_rhat(chains_array)
        rhat_vals.append(rhat)

    max_rhat = max(rhat_vals)

    return samples_dict, max_rhat, np.mean(accept_rates)

def compute_rhat(chains):
    """Compute Gelman-Rubin R-hat statistic"""
    n_chains, n_iter = chains.shape

    # Within-chain variance
    W = np.mean([np.var(chain, ddof=1) for chain in chains])

    # Between-chain variance
    chain_means = np.array([np.mean(chain) for chain in chains])
    B = n_iter * np.var(chain_means, ddof=1)

    # Pooled variance
    var_plus = ((n_iter - 1) / n_iter) * W + (1 / n_iter) * B

    # R-hat
    rhat = np.sqrt(var_plus / W) if W > 0 else 1.0

    return rhat

# ============================================================================
# RUN SBC SIMULATIONS
# ============================================================================

# Storage for results
results = {
    'beta_0_true': [],
    'beta_1_true': [],
    'phi_true': [],
    'beta_0_mean': [],
    'beta_1_mean': [],
    'phi_mean': [],
    'beta_0_median': [],
    'beta_1_median': [],
    'phi_median': [],
    'beta_0_lower': [],
    'beta_1_lower': [],
    'phi_lower': [],
    'beta_0_upper': [],
    'beta_1_upper': [],
    'phi_upper': [],
    'beta_0_rank': [],
    'beta_1_rank': [],
    'phi_rank': [],
    'beta_0_in_ci': [],
    'beta_1_in_ci': [],
    'phi_in_ci': [],
    'rhat_max': [],
    'accept_rate': [],
    'converged': [],
    'sim_time': []
}

# Run SBC simulations
start_time = time.time()
convergence_issues = 0

for sim in range(N_SIMULATIONS):
    print(f"\n[{sim+1}/{N_SIMULATIONS}] ", end="")
    sim_start = time.time()

    # Step 1: Draw true parameters from priors
    beta_0_true = np.random.normal(PRIOR_BETA0_MEAN, PRIOR_BETA0_SD)
    beta_1_true = np.random.normal(PRIOR_BETA1_MEAN, PRIOR_BETA1_SD)
    phi_true = np.random.gamma(PRIOR_PHI_SHAPE, 1/PRIOR_PHI_RATE)

    print(f"True: β₀={beta_0_true:.2f}, β₁={beta_1_true:.2f}, φ={phi_true:.1f} | ", end="")

    # Step 2: Generate synthetic data
    mu = np.exp(beta_0_true + beta_1_true * year_values)
    # neg_binomial_2 parameterization: n=phi, p=phi/(phi+mu)
    p = phi_true / (phi_true + mu)
    C_sim = np.random.negative_binomial(phi_true, p)

    # Step 3: Fit model to synthetic data
    try:
        samples_dict, rhat_max, accept_rate = run_mcmc(
            year_values, C_sim, n_chains=N_CHAINS,
            n_iter=N_ITER, n_warmup=N_WARMUP
        )

        converged = (rhat_max < 1.1) and (accept_rate > 0.15)

        if not converged:
            convergence_issues += 1

        # Extract posterior samples
        beta_0_post = samples_dict['beta_0']
        beta_1_post = samples_dict['beta_1']
        phi_post = samples_dict['phi']

        # Compute rank statistics (SBC ranks)
        beta_0_rank = np.sum(beta_0_post < beta_0_true)
        beta_1_rank = np.sum(beta_1_post < beta_1_true)
        phi_rank = np.sum(phi_post < phi_true)

        # Compute posterior summaries
        beta_0_mean = np.mean(beta_0_post)
        beta_1_mean = np.mean(beta_1_post)
        phi_mean = np.mean(phi_post)

        beta_0_median = np.median(beta_0_post)
        beta_1_median = np.median(beta_1_post)
        phi_median = np.median(phi_post)

        # 90% credible intervals
        beta_0_ci = np.percentile(beta_0_post, [5, 95])
        beta_1_ci = np.percentile(beta_1_post, [5, 95])
        phi_ci = np.percentile(phi_post, [5, 95])

        # Check coverage
        beta_0_in_ci = beta_0_ci[0] <= beta_0_true <= beta_0_ci[1]
        beta_1_in_ci = beta_1_ci[0] <= beta_1_true <= beta_1_ci[1]
        phi_in_ci = phi_ci[0] <= phi_true <= phi_ci[1]

        # Store results
        results['beta_0_true'].append(beta_0_true)
        results['beta_1_true'].append(beta_1_true)
        results['phi_true'].append(phi_true)
        results['beta_0_mean'].append(beta_0_mean)
        results['beta_1_mean'].append(beta_1_mean)
        results['phi_mean'].append(phi_mean)
        results['beta_0_median'].append(beta_0_median)
        results['beta_1_median'].append(beta_1_median)
        results['phi_median'].append(phi_median)
        results['beta_0_lower'].append(beta_0_ci[0])
        results['beta_1_lower'].append(beta_1_ci[0])
        results['phi_lower'].append(phi_ci[0])
        results['beta_0_upper'].append(beta_0_ci[1])
        results['beta_1_upper'].append(beta_1_ci[1])
        results['phi_upper'].append(phi_ci[1])
        results['beta_0_rank'].append(beta_0_rank)
        results['beta_1_rank'].append(beta_1_rank)
        results['phi_rank'].append(phi_rank)
        results['beta_0_in_ci'].append(beta_0_in_ci)
        results['beta_1_in_ci'].append(beta_1_in_ci)
        results['phi_in_ci'].append(phi_in_ci)
        results['rhat_max'].append(rhat_max)
        results['accept_rate'].append(accept_rate)
        results['converged'].append(converged)

        sim_time = time.time() - sim_start
        results['sim_time'].append(sim_time)

        status = "OK" if converged else "WARN"
        print(f"Fit: {status} (Rhat={rhat_max:.3f}, acc={accept_rate:.2f}, t={sim_time:.1f}s)")

    except Exception as e:
        print(f"FAILED: {str(e)}")
        convergence_issues += 1
        # Store NaN for failed simulations
        for key in results.keys():
            if key in ['beta_0_true', 'beta_1_true', 'phi_true']:
                results[key].append(globals()[key])
            elif key == 'converged':
                results[key].append(False)
            elif key == 'sim_time':
                results[key].append(time.time() - sim_start)
            else:
                results[key].append(np.nan)

total_time = time.time() - start_time
print("\n" + "="*60)
print(f"SBC completed in {total_time/60:.1f} minutes")
print(f"Convergence issues: {convergence_issues}/{N_SIMULATIONS} ({100*convergence_issues/N_SIMULATIONS:.1f}%)")

# Convert to DataFrame
df = pd.DataFrame(results)

# Remove failed simulations for analysis
df_valid = df[df['converged']].copy()
n_valid = len(df_valid)
print(f"Valid simulations for analysis: {n_valid}/{N_SIMULATIONS}")

# Save results
df.to_csv(BASE_DIR / "sbc_results.csv", index=False)
print(f"\nResults saved to {BASE_DIR / 'sbc_results.csv'}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating plots...")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. RANK HISTOGRAMS
# ============================================================================
print("  - Rank histograms...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
n_bins = 20
n_samples = N_CHAINS * N_ITER

params = [
    ('beta_0_rank', r'$\beta_0$ (Intercept)', 'beta_0'),
    ('beta_1_rank', r'$\beta_1$ (Slope)', 'beta_1'),
    ('phi_rank', r'$\phi$ (Dispersion)', 'phi')
]

for idx, (rank_col, title, param) in enumerate(params):
    ax = axes[idx]
    ranks = df_valid[rank_col].values

    # Plot histogram
    ax.hist(ranks, bins=n_bins, color='steelblue', alpha=0.7, edgecolor='black')

    # Expected uniform distribution
    expected_count = n_valid / n_bins
    ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
               label=f'Expected (uniform)')

    # Confidence bands for uniform (95%)
    se = np.sqrt(n_valid / n_bins * (1 - 1/n_bins))
    lower = expected_count - 1.96 * se
    upper = expected_count + 1.96 * se
    ax.axhline(lower, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(upper, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.fill_between([0, n_samples], lower, upper, color='red', alpha=0.1)

    ax.set_xlabel('Rank statistic')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title}\n(n={n_valid} simulations)')
    ax.legend()
    ax.set_xlim(0, n_samples)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_histograms.png", dpi=300, bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'rank_histograms.png'}")
plt.close()

# ============================================================================
# 2. COVERAGE ANALYSIS
# ============================================================================
print("  - Coverage analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (param, title) in enumerate([('beta_0', r'$\beta_0$'),
                                       ('beta_1', r'$\beta_1$'),
                                       ('phi', r'$\phi$')]):
    ax = axes[idx]

    # Calculate empirical coverage at different levels
    levels = np.arange(5, 100, 5)
    empirical_coverage = []

    for level in levels:
        lower_pct = (100 - level) / 2
        upper_pct = 100 - lower_pct

        # Use the rank statistic approach for coverage
        ranks = df_valid[f'{param}_rank'].values
        lower_rank = (lower_pct / 100) * n_samples
        upper_rank = (upper_pct / 100) * n_samples
        covered = np.sum((ranks >= lower_rank) & (ranks <= upper_rank))
        empirical_coverage.append(100 * covered / n_valid)

    # Plot
    ax.plot(levels, levels, 'r--', linewidth=2, label='Perfect calibration')
    ax.plot(levels, empirical_coverage, 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Observed coverage')

    # Highlight 90% CI
    coverage_90 = df_valid[f"{param}_in_ci"].mean() * 100
    ax.axhline(90, color='green', linestyle=':', alpha=0.5)

    ax.set_xlabel('Nominal coverage (%)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_title(f'{title}\n(90% CI: {coverage_90:.1f}% coverage)')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_analysis.png", dpi=300, bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'coverage_analysis.png'}")
plt.close()

# ============================================================================
# 3. PARAMETER RECOVERY
# ============================================================================
print("  - Parameter recovery...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (param, title) in enumerate([('beta_0', r'$\beta_0$ (Intercept)'),
                                       ('beta_1', r'$\beta_1$ (Slope)'),
                                       ('phi', r'$\phi$ (Dispersion)')]):
    ax = axes[idx]

    true_vals = df_valid[f'{param}_true'].values
    est_vals = df_valid[f'{param}_median'].values
    lower_vals = df_valid[f'{param}_lower'].values
    upper_vals = df_valid[f'{param}_upper'].values

    # Scatter plot with error bars
    for i in range(len(true_vals)):
        ax.plot([true_vals[i], true_vals[i]], [lower_vals[i], upper_vals[i]],
                'k-', alpha=0.2, linewidth=1)

    ax.scatter(true_vals, est_vals, alpha=0.6, s=50, color='steelblue')

    # Perfect recovery line
    min_val = min(true_vals.min(), est_vals.min())
    max_val = max(true_vals.max(), est_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect recovery')

    # Compute correlation
    r = np.corrcoef(true_vals, est_vals)[0, 1]

    # Compute bias
    bias = np.mean(est_vals - true_vals)
    rmse = np.sqrt(np.mean((est_vals - true_vals)**2))

    ax.set_xlabel('True value')
    ax.set_ylabel('Estimated value (posterior median)')
    ax.set_title(f'{title}\nr={r:.3f}, bias={bias:.3f}, RMSE={rmse:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=300, bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'parameter_recovery.png'}")
plt.close()

# ============================================================================
# 4. SHRINKAGE ANALYSIS
# ============================================================================
print("  - Shrinkage analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (param, title, prior_sd) in enumerate([
    ('beta_0', r'$\beta_0$', PRIOR_BETA0_SD),
    ('beta_1', r'$\beta_1$', PRIOR_BETA1_SD),
    ('phi', r'$\phi$', np.sqrt(PRIOR_PHI_SHAPE / PRIOR_PHI_RATE**2))
]):
    ax = axes[idx]

    # Compute posterior standard deviations
    post_sds = []
    for i in range(n_valid):
        # Approximate posterior SD from 90% CI width
        ci_width = df_valid.iloc[i][f'{param}_upper'] - df_valid.iloc[i][f'{param}_lower']
        post_sd = ci_width / (2 * 1.645)  # 1.645 is the z-score for 90% CI
        post_sds.append(post_sd)

    post_sds = np.array(post_sds)
    shrinkage = 1 - post_sds / prior_sd

    ax.hist(shrinkage, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(shrinkage), color='red', linestyle='--', linewidth=2,
               label=f'Mean shrinkage={np.mean(shrinkage):.2f}')
    ax.set_xlabel('Shrinkage (1 - posterior_SD / prior_SD)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title}\nPrior SD={prior_sd:.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "shrinkage_analysis.png", dpi=300, bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR / 'shrinkage_analysis.png'}")
plt.close()

# ============================================================================
# COMPUTE SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("SBC SUMMARY STATISTICS")
print("="*60)

summary_stats = {}

for param in ['beta_0', 'beta_1', 'phi']:
    print(f"\n{param.upper()}:")

    # Coverage
    coverage = df_valid[f'{param}_in_ci'].mean()
    print(f"  90% CI Coverage: {coverage*100:.1f}% (expected: 90%)")
    summary_stats[f'{param}_coverage'] = coverage

    # Correlation
    r = np.corrcoef(df_valid[f'{param}_true'], df_valid[f'{param}_median'])[0, 1]
    print(f"  Correlation (r): {r:.3f}")
    summary_stats[f'{param}_correlation'] = r

    # Bias
    bias = np.mean(df_valid[f'{param}_median'] - df_valid[f'{param}_true'])
    rel_bias = bias / np.std(df_valid[f'{param}_true'])
    print(f"  Bias: {bias:.4f} (relative: {rel_bias:.3f} SD)")
    summary_stats[f'{param}_bias'] = bias
    summary_stats[f'{param}_rel_bias'] = rel_bias

    # RMSE
    rmse = np.sqrt(np.mean((df_valid[f'{param}_median'] - df_valid[f'{param}_true'])**2))
    rel_rmse = rmse / np.std(df_valid[f'{param}_true'])
    print(f"  RMSE: {rmse:.4f} (relative: {rel_rmse:.3f} SD)")
    summary_stats[f'{param}_rmse'] = rmse

    # Rank uniformity (chi-square test)
    ranks = df_valid[f'{param}_rank'].values
    observed, _ = np.histogram(ranks, bins=n_bins, range=(0, n_samples))
    expected = np.full(n_bins, n_valid / n_bins)
    chi2, p_value = stats.chisquare(observed, expected)
    print(f"  Rank uniformity (χ²): {chi2:.2f}, p={p_value:.3f}")
    summary_stats[f'{param}_rank_chi2'] = chi2
    summary_stats[f'{param}_rank_pvalue'] = p_value

print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)
print(f"Converged simulations: {df['converged'].sum()}/{N_SIMULATIONS} ({100*df['converged'].mean():.1f}%)")
print(f"Mean max R-hat: {df_valid['rhat_max'].mean():.4f}")
print(f"Mean acceptance rate: {df_valid['accept_rate'].mean():.3f}")
print(f"Mean simulation time: {df['sim_time'].mean():.1f}s")

# ============================================================================
# PASS/FAIL DECISION
# ============================================================================

print("\n" + "="*60)
print("PASS/FAIL DECISION")
print("="*60)

criteria = {
    'coverage': True,
    'correlation': True,
    'rank_uniformity': True,
    'convergence': True
}

# Check coverage (should be 85-95% for 90% CIs)
for param in ['beta_0', 'beta_1', 'phi']:
    coverage = summary_stats[f'{param}_coverage']
    if coverage < 0.85 or coverage > 0.95:
        criteria['coverage'] = False
        print(f"FAIL: {param} coverage {coverage*100:.1f}% outside [85%, 95%]")

# Check correlation (should be > 0.9)
for param in ['beta_0', 'beta_1', 'phi']:
    r = summary_stats[f'{param}_correlation']
    if r < 0.9:
        criteria['correlation'] = False
        print(f"FAIL: {param} correlation r={r:.3f} < 0.9")

# Check rank uniformity (p > 0.05 for chi-square test)
for param in ['beta_0', 'beta_1', 'phi']:
    p_value = summary_stats[f'{param}_rank_pvalue']
    if p_value < 0.05:
        criteria['rank_uniformity'] = False
        print(f"FAIL: {param} rank histogram non-uniform (p={p_value:.3f})")

# Check convergence (>90% should converge)
if df['converged'].mean() < 0.90:
    criteria['convergence'] = False
    print(f"FAIL: Only {100*df['converged'].mean():.1f}% simulations converged (<90%)")

# Overall decision
if all(criteria.values()):
    decision = "PASS"
    print(f"\n*** OVERALL: {decision} ***")
    print("Model successfully recovers known parameters.")
else:
    decision = "FAIL"
    print(f"\n*** OVERALL: {decision} ***")
    print("Model has systematic issues in parameter recovery.")

print("="*60)

# Save summary statistics
summary_df = pd.DataFrame([summary_stats])
summary_df['decision'] = decision
summary_df['n_simulations'] = N_SIMULATIONS
summary_df['n_valid'] = n_valid
summary_df['convergence_rate'] = df['converged'].mean()
summary_df.to_csv(BASE_DIR / "sbc_summary.csv", index=False)

print(f"\nSummary saved to {BASE_DIR / 'sbc_summary.csv'}")
print("\nSBC validation complete!")
