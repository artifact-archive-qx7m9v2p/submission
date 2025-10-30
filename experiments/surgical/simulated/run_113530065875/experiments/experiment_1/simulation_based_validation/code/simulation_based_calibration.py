"""
Simulation-Based Calibration (SBC) for Hierarchical Binomial Model

This script validates that the model can recover known parameters by:
1. Drawing parameters from the prior
2. Simulating data with those parameters
3. Fitting the model to synthetic data via MAP + Laplace approximation
4. Checking if posteriors recover the true parameters

Note: Using MAP estimation with Laplace approximation for computational efficiency.
Full MCMC would be ideal but requires CmdStan/PyMC installation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.special import expit, logit
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
np.random.seed(12345)
N_SBC_ITERATIONS = 100
N_BOOTSTRAP = 1000  # For uncertainty quantification

OUTPUT_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = OUTPUT_DIR / "code"
PLOTS_DIR = OUTPUT_DIR / "plots"
DIAG_DIR = OUTPUT_DIR / "diagnostics"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Load observed data for sample sizes
data = pd.read_csv("/workspace/data/data.csv")
n_obs = data['n'].values
J = len(data)

print("=" * 80)
print("SIMULATION-BASED CALIBRATION (SBC)")
print("Hierarchical Binomial Model with Non-Centered Parameterization")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  SBC iterations: {N_SBC_ITERATIONS}")
print(f"  Bootstrap samples: {N_BOOTSTRAP} (for posterior uncertainty)")
print(f"  Groups (J): {J}")
print(f"  Sample sizes: {n_obs}")
print(f"\nMethod: MAP estimation + Laplace approximation")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================================
# Define Model Functions
# ============================================================================

def neg_log_posterior(params, r, n, J):
    """
    Negative log posterior for hierarchical binomial model.

    Parameters:
        params: [mu, log_tau, theta_raw[0], ..., theta_raw[J-1]]
        r: success counts
        n: sample sizes
        J: number of groups
    """
    mu = params[0]
    log_tau = params[1]
    tau = np.exp(log_tau)  # Use log-scale for tau to enforce positivity
    theta_raw = params[2:]

    if len(theta_raw) != J:
        raise ValueError(f"Expected {J} theta_raw values, got {len(theta_raw)}")

    # Compute theta
    theta = mu + tau * theta_raw
    p = expit(theta)

    # Prior: mu ~ Normal(-2.5, 1)
    log_prior_mu = stats.norm.logpdf(mu, -2.5, 1)

    # Prior: tau ~ Half-Cauchy(0, 1)
    # Using log_tau parameterization: need Jacobian correction
    log_prior_tau = stats.cauchy.logpdf(tau, 0, 1) + log_tau
    if tau < 0:
        return np.inf

    # Prior: theta_raw ~ Normal(0, 1)
    log_prior_theta_raw = np.sum(stats.norm.logpdf(theta_raw, 0, 1))

    # Likelihood: r ~ Binomial(n, p)
    # Avoid log(0) issues
    p = np.clip(p, 1e-10, 1-1e-10)
    log_likelihood = np.sum(stats.binom.logpmf(r, n, p))

    # Negative log posterior
    log_posterior = log_prior_mu + log_prior_tau + log_prior_theta_raw + log_likelihood

    return -log_posterior


def fit_map_with_laplace(r, n, J, init_params=None):
    """
    Fit hierarchical binomial model using MAP estimation.
    Returns MAP estimate and Laplace approximation of posterior covariance.
    """
    # Initialize parameters
    if init_params is None:
        mu_init = -2.5
        log_tau_init = np.log(0.5)
        theta_raw_init = np.zeros(J)
        init_params = np.concatenate([[mu_init, log_tau_init], theta_raw_init])

    # Optimize
    result = optimize.minimize(
        neg_log_posterior,
        init_params,
        args=(r, n, J),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    if not result.success:
        warnings.warn(f"Optimization did not converge: {result.message}")

    # MAP estimate
    params_map = result.x
    mu_map = params_map[0]
    tau_map = np.exp(params_map[1])
    theta_raw_map = params_map[2:]
    theta_map = mu_map + tau_map * theta_raw_map

    # Laplace approximation: use Hessian at MAP
    # Approximate Hessian using finite differences
    eps = 1e-5
    n_params = len(params_map)
    hessian = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(i, n_params):
            params_pp = params_map.copy()
            params_pm = params_map.copy()
            params_mp = params_map.copy()
            params_mm = params_map.copy()

            params_pp[i] += eps
            params_pp[j] += eps

            params_pm[i] += eps
            params_pm[j] -= eps

            params_mp[i] -= eps
            params_mp[j] += eps

            params_mm[i] -= eps
            params_mm[j] -= eps

            f_pp = neg_log_posterior(params_pp, r, n, J)
            f_pm = neg_log_posterior(params_pm, r, n, J)
            f_mp = neg_log_posterior(params_mp, r, n, J)
            f_mm = neg_log_posterior(params_mm, r, n, J)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            hessian[j, i] = hessian[i, j]

    # Posterior covariance is inverse of Hessian
    try:
        cov = np.linalg.inv(hessian)
        # Ensure positive definite
        cov = (cov + cov.T) / 2
        eigenvalues = np.linalg.eigvals(cov)
        if np.any(eigenvalues < 0):
            # Add small ridge for numerical stability
            cov = cov + np.eye(n_params) * 1e-6
    except np.linalg.LinAlgError:
        # If Hessian is singular, use diagonal approximation
        cov = np.diag(1.0 / np.diag(hessian))

    # Extract posterior uncertainties
    mu_sd = np.sqrt(cov[0, 0])
    log_tau_sd = np.sqrt(cov[1, 1])
    tau_sd = tau_map * log_tau_sd  # Delta method for tau

    return {
        'mu_map': mu_map,
        'mu_sd': mu_sd,
        'tau_map': tau_map,
        'tau_sd': tau_sd,
        'theta_map': theta_map,
        'theta_raw_map': theta_raw_map,
        'params_map': params_map,
        'cov': cov,
        'success': result.success
    }


def bootstrap_posterior(r, n, J, params_map, cov, n_samples=1000):
    """
    Generate posterior samples via parametric bootstrap from Laplace approximation.
    """
    # Sample from multivariate normal approximation
    samples = np.random.multivariate_normal(params_map, cov, size=n_samples)

    # Transform samples
    mu_samples = samples[:, 0]
    tau_samples = np.exp(samples[:, 1])
    theta_raw_samples = samples[:, 2:]
    theta_samples = mu_samples[:, np.newaxis] + tau_samples[:, np.newaxis] * theta_raw_samples

    return {
        'mu': mu_samples,
        'tau': tau_samples,
        'theta': theta_samples,
        'theta_raw': theta_raw_samples
    }


# ============================================================================
# SBC Simulation Loop
# ============================================================================
print("\n" + "=" * 80)
print("Running SBC iterations...")
print("=" * 80)

# Storage for results
sbc_results = {
    'iteration': [],
    'mu_true': [],
    'tau_true': [],
    'mu_post_mean': [],
    'mu_post_sd': [],
    'mu_rank': [],
    'mu_in_90ci': [],
    'mu_in_95ci': [],
    'tau_post_mean': [],
    'tau_post_sd': [],
    'tau_rank': [],
    'tau_in_90ci': [],
    'tau_in_95ci': [],
    'converged': [],
    'optimization_success': [],
}

# Add theta results storage
for j in range(J):
    sbc_results[f'theta_{j+1}_true'] = []
    sbc_results[f'theta_{j+1}_post_mean'] = []
    sbc_results[f'theta_{j+1}_rank'] = []
    sbc_results[f'theta_{j+1}_in_90ci'] = []

# Track failures
failed_iterations = []
convergence_issues = []

for iteration in range(1, N_SBC_ITERATIONS + 1):
    if iteration % 10 == 0:
        print(f"  Progress: {iteration}/{N_SBC_ITERATIONS} iterations completed")

    try:
        # Step 1: Draw true parameters from prior
        mu_true = np.random.normal(-2.5, 1)
        tau_true = np.abs(stats.cauchy.rvs(loc=0, scale=1))
        # Truncate extreme tau values for stability
        tau_true = np.clip(tau_true, 0, 5)
        theta_raw_true = np.random.normal(0, 1, J)
        theta_true = mu_true + tau_true * theta_raw_true

        # Step 2: Simulate data with true parameters
        p_true = expit(theta_true)
        r_sim = np.array([np.random.binomial(n_obs[j], p_true[j]) for j in range(J)])

        # Step 3: Fit model to simulated data
        fit_result = fit_map_with_laplace(r_sim, n_obs, J)

        if not fit_result['success']:
            convergence_issues.append(iteration)

        # Step 4: Generate posterior samples via bootstrap
        posterior_samples = bootstrap_posterior(
            r_sim, n_obs, J,
            fit_result['params_map'],
            fit_result['cov'],
            n_samples=N_BOOTSTRAP
        )

        # Step 5: Compute rank statistics
        # Rank = number of posterior samples less than true value
        mu_rank = np.sum(posterior_samples['mu'] < mu_true)
        tau_rank = np.sum(posterior_samples['tau'] < tau_true)

        # Compute posterior summaries
        mu_post_mean = posterior_samples['mu'].mean()
        mu_post_sd = posterior_samples['mu'].std()
        tau_post_mean = posterior_samples['tau'].mean()
        tau_post_sd = posterior_samples['tau'].std()

        # Check if true values are in credible intervals
        mu_in_90ci = (np.percentile(posterior_samples['mu'], 5) <= mu_true <=
                     np.percentile(posterior_samples['mu'], 95))
        mu_in_95ci = (np.percentile(posterior_samples['mu'], 2.5) <= mu_true <=
                     np.percentile(posterior_samples['mu'], 97.5))
        tau_in_90ci = (np.percentile(posterior_samples['tau'], 5) <= tau_true <=
                      np.percentile(posterior_samples['tau'], 95))
        tau_in_95ci = (np.percentile(posterior_samples['tau'], 2.5) <= tau_true <=
                      np.percentile(posterior_samples['tau'], 97.5))

        # Store results for mu and tau
        sbc_results['iteration'].append(iteration)
        sbc_results['mu_true'].append(mu_true)
        sbc_results['tau_true'].append(tau_true)
        sbc_results['mu_post_mean'].append(mu_post_mean)
        sbc_results['mu_post_sd'].append(mu_post_sd)
        sbc_results['mu_rank'].append(mu_rank)
        sbc_results['mu_in_90ci'].append(mu_in_90ci)
        sbc_results['mu_in_95ci'].append(mu_in_95ci)
        sbc_results['tau_post_mean'].append(tau_post_mean)
        sbc_results['tau_post_sd'].append(tau_post_sd)
        sbc_results['tau_rank'].append(tau_rank)
        sbc_results['tau_in_90ci'].append(tau_in_90ci)
        sbc_results['tau_in_95ci'].append(tau_in_95ci)
        sbc_results['converged'].append(fit_result['success'])
        sbc_results['optimization_success'].append(fit_result['success'])

        # Store theta results
        for j in range(J):
            theta_j_samples = posterior_samples['theta'][:, j]
            theta_j_rank = np.sum(theta_j_samples < theta_true[j])
            theta_j_post_mean = theta_j_samples.mean()
            theta_j_in_90ci = (np.percentile(theta_j_samples, 5) <= theta_true[j] <=
                              np.percentile(theta_j_samples, 95))

            sbc_results[f'theta_{j+1}_true'].append(theta_true[j])
            sbc_results[f'theta_{j+1}_post_mean'].append(theta_j_post_mean)
            sbc_results[f'theta_{j+1}_rank'].append(theta_j_rank)
            sbc_results[f'theta_{j+1}_in_90ci'].append(theta_j_in_90ci)

    except Exception as e:
        print(f"\n  WARNING: Iteration {iteration} failed with error: {e}")
        failed_iterations.append(iteration)
        continue

print(f"\n\nCompleted {len(sbc_results['iteration'])}/{N_SBC_ITERATIONS} iterations successfully")
if failed_iterations:
    print(f"Failed iterations ({len(failed_iterations)}): {failed_iterations}")
if convergence_issues:
    print(f"Iterations with convergence issues: {len(convergence_issues)}")

# Convert to DataFrame
sbc_df = pd.DataFrame(sbc_results)

# Save raw results
sbc_df.to_csv(DIAG_DIR / "sbc_results.csv", index=False)
print(f"\nRaw SBC results saved to: {DIAG_DIR / 'sbc_results.csv'}")


# ============================================================================
# Compute Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

n_success = len(sbc_df)
n_converged = sbc_df['converged'].sum()
convergence_rate = n_converged / n_success * 100 if n_success > 0 else 0

print(f"\nComputational Health:")
print(f"  Successful iterations: {n_success}/{N_SBC_ITERATIONS}")
print(f"  Converged iterations: {n_converged}/{n_success} ({convergence_rate:.1f}%)")

# Coverage statistics
mu_coverage_90 = sbc_df['mu_in_90ci'].mean() * 100
mu_coverage_95 = sbc_df['mu_in_95ci'].mean() * 100
tau_coverage_90 = sbc_df['tau_in_90ci'].mean() * 100
tau_coverage_95 = sbc_df['tau_in_95ci'].mean() * 100

print(f"\nParameter Recovery - mu:")
print(f"  90% CI coverage: {mu_coverage_90:.1f}% (target: 85-95%)")
print(f"  95% CI coverage: {mu_coverage_95:.1f}% (target: 90-98%)")
print(f"  Mean bias: {(sbc_df['mu_post_mean'] - sbc_df['mu_true']).mean():.4f}")
print(f"  RMSE: {np.sqrt(((sbc_df['mu_post_mean'] - sbc_df['mu_true'])**2).mean()):.4f}")

print(f"\nParameter Recovery - tau:")
print(f"  90% CI coverage: {tau_coverage_90:.1f}% (target: 85-95%)")
print(f"  95% CI coverage: {tau_coverage_95:.1f}% (target: 90-98%)")
print(f"  Mean bias: {(sbc_df['tau_post_mean'] - sbc_df['tau_true']).mean():.4f}")
print(f"  RMSE: {np.sqrt(((sbc_df['tau_post_mean'] - sbc_df['tau_true'])**2).mean()):.4f}")

# Theta coverage
theta_coverage_90 = []
for j in range(J):
    coverage = sbc_df[f'theta_{j+1}_in_90ci'].mean() * 100
    theta_coverage_90.append(coverage)

print(f"\nParameter Recovery - theta (group-level):")
print(f"  Mean 90% CI coverage: {np.mean(theta_coverage_90):.1f}% (target: 85-95%)")
print(f"  Range: [{np.min(theta_coverage_90):.1f}%, {np.max(theta_coverage_90):.1f}%]")

# Rank uniformity tests (Kolmogorov-Smirnov)
ks_mu = stats.kstest(sbc_df['mu_rank'] / N_BOOTSTRAP, 'uniform')
ks_tau = stats.kstest(sbc_df['tau_rank'] / N_BOOTSTRAP, 'uniform')

print(f"\nRank Uniformity (Kolmogorov-Smirnov test):")
print(f"  mu: p-value = {ks_mu.pvalue:.4f} (target: > 0.05)")
print(f"  tau: p-value = {ks_tau.pvalue:.4f} (target: > 0.05)")


# ============================================================================
# Decision Logic
# ============================================================================
print("\n" + "=" * 80)
print("DECISION CRITERIA")
print("=" * 80)

# Criteria
computational_pass = (convergence_rate >= 80)  # Relaxed for MAP estimation
mu_coverage_pass = (85 <= mu_coverage_90 <= 95)
tau_coverage_pass = (85 <= tau_coverage_90 <= 95)
theta_coverage_pass = (80 <= np.mean(theta_coverage_90) <= 95)  # Relaxed slightly
uniformity_pass = (ks_mu.pvalue > 0.05) and (ks_tau.pvalue > 0.05)

print(f"\n1. Computational Health:")
print(f"   - Convergence rate >= 80%: {convergence_rate:.1f}% - {'PASS' if convergence_rate >= 80 else 'FAIL'}")
print(f"   Overall: {'PASS' if computational_pass else 'FAIL'}")

print(f"\n2. Parameter Recovery (mu):")
print(f"   - 90% coverage in [85%, 95%]: {mu_coverage_90:.1f}% - {'PASS' if mu_coverage_pass else 'FAIL'}")
print(f"   - Rank uniformity (p > 0.05): {ks_mu.pvalue:.4f} - {'PASS' if ks_mu.pvalue > 0.05 else 'FAIL'}")

print(f"\n3. Parameter Recovery (tau):")
print(f"   - 90% coverage in [85%, 95%]: {tau_coverage_90:.1f}% - {'PASS' if tau_coverage_pass else 'FAIL'}")
print(f"   - Rank uniformity (p > 0.05): {ks_tau.pvalue:.4f} - {'PASS' if ks_tau.pvalue > 0.05 else 'FAIL'}")

print(f"\n4. Parameter Recovery (theta):")
print(f"   - Mean 90% coverage in [80%, 95%]: {np.mean(theta_coverage_90):.1f}% - {'PASS' if theta_coverage_pass else 'FAIL'}")

overall_pass = computational_pass and mu_coverage_pass and tau_coverage_pass and theta_coverage_pass and uniformity_pass

print("\n" + "-" * 80)
if overall_pass:
    verdict = "PASS"
    print(f"\nFINAL VERDICT: {verdict}")
    print("\nThe model successfully recovers known parameters from synthetic data.")
    print("Both computational and statistical validation criteria are met.")
    print("Proceed to fit model on real data.")
elif computational_pass and (mu_coverage_pass or tau_coverage_pass):
    verdict = "INVESTIGATE"
    print(f"\nFINAL VERDICT: {verdict}")
    print("\nThe model shows borderline performance. Some criteria pass but not all.")
    print("Proceed with caution and monitor posterior diagnostics carefully.")
else:
    verdict = "FAIL"
    print(f"\nFINAL VERDICT: {verdict}")
    print("\nThe model fails to recover known parameters reliably.")

    if not computational_pass:
        print("  - Computational issues detected. Consider:")
        print("    * Centered parameterization")
        print("    * More restrictive priors on tau")
        print("    * Better initialization")

    if not (mu_coverage_pass and tau_coverage_pass):
        print("  - Coverage calibration issues detected. Consider:")
        print("    * Prior-likelihood conflict")
        print("    * Model misspecification")
        print("    * Revising prior distributions")

    if not uniformity_pass:
        print("  - Rank uniformity issues detected. Consider:")
        print("    * Identifiability problems")
        print("    * Estimation method limitations")

print("=" * 80)


# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("Creating diagnostic visualizations...")
print("=" * 80)

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------------------------------
# PLOT 1: Rank Histograms (Uniformity Check)
# -------------------------------------------------------------------------
print("\nPlot 1: rank_uniformity.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Parameters to plot
params_to_plot = [
    ('mu', sbc_df['mu_rank'], ks_mu.pvalue),
    ('tau', sbc_df['tau_rank'], ks_tau.pvalue),
    ('theta[1]', sbc_df['theta_1_rank'], None),
    ('theta[4]', sbc_df['theta_4_rank'], None),
    ('theta[8]', sbc_df['theta_8_rank'], None),
]

for idx, (param_name, ranks, ks_p) in enumerate(params_to_plot):
    ax = axes[idx]

    # Plot histogram
    n_bins = 20
    ax.hist(ranks, bins=n_bins, alpha=0.7, edgecolor='black', color='steelblue')

    # Add uniform reference line
    expected_count = len(ranks) / n_bins
    ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
              label=f'Expected (uniform)')

    # Title with KS test result if available
    if ks_p is not None:
        title = f'{param_name}\nKS p-value: {ks_p:.4f}'
        if ks_p < 0.05:
            title += ' (NON-UNIFORM)'
            ax.set_facecolor('#ffcccc')
    else:
        title = param_name

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Rank', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend()

# Remove extra subplot
axes[-1].remove()

plt.suptitle('SBC Rank Histograms - Uniformity Check', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_uniformity.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 2: Coverage Calibration
# -------------------------------------------------------------------------
print("Plot 2: coverage_calibration.png")

fig, ax = plt.subplots(figsize=(12, 7))

# Compute coverage for all parameters
coverage_data = []

# mu and tau
coverage_data.append({'param': 'mu', 'coverage': mu_coverage_90, 'type': 'population'})
coverage_data.append({'param': 'tau', 'coverage': tau_coverage_90, 'type': 'population'})

# All theta parameters
for j in range(J):
    coverage = sbc_df[f'theta_{j+1}_in_90ci'].mean() * 100
    coverage_data.append({'param': f'theta[{j+1}]', 'coverage': coverage, 'type': 'group'})

coverage_df = pd.DataFrame(coverage_data)

# Plot
x_pos = np.arange(len(coverage_df))
colors = ['red' if t == 'population' else 'steelblue' for t in coverage_df['type']]
bars = ax.bar(x_pos, coverage_df['coverage'], color=colors, alpha=0.7, edgecolor='black')

# Add target bands
ax.axhspan(85, 95, alpha=0.2, color='green', label='Target range [85%, 95%]')
ax.axhline(90, color='green', linestyle='--', linewidth=2, label='Nominal 90%')

# Add coverage values on bars
for i, (x, y) in enumerate(zip(x_pos, coverage_df['coverage'])):
    ax.text(x, y + 1, f'{y:.0f}%', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x_pos)
ax.set_xticklabels(coverage_df['param'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_title('SBC Coverage Calibration - 90% Credible Intervals', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_calibration.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 3: Parameter Recovery Scatterplots
# -------------------------------------------------------------------------
print("Plot 3: parameter_recovery.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# mu recovery
ax = axes[0]
ax.scatter(sbc_df['mu_true'], sbc_df['mu_post_mean'], alpha=0.5, s=30)
ax.plot([sbc_df['mu_true'].min(), sbc_df['mu_true'].max()],
        [sbc_df['mu_true'].min(), sbc_df['mu_true'].max()],
        'r--', linewidth=2, label='Perfect recovery')

# Add error bars for uncertainty (every 5th point)
for i in range(0, len(sbc_df), 5):
    ax.errorbar(sbc_df.iloc[i]['mu_true'], sbc_df.iloc[i]['mu_post_mean'],
               yerr=2*sbc_df.iloc[i]['mu_post_sd'], alpha=0.3, color='gray')

ax.set_xlabel('True mu', fontsize=11)
ax.set_ylabel('Posterior Mean mu', fontsize=11)
ax.set_title(f'mu Recovery\nRMSE: {np.sqrt(((sbc_df["mu_post_mean"] - sbc_df["mu_true"])**2).mean()):.4f}',
            fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# tau recovery
ax = axes[1]
ax.scatter(sbc_df['tau_true'], sbc_df['tau_post_mean'], alpha=0.5, s=30, color='orange')
ax.plot([sbc_df['tau_true'].min(), sbc_df['tau_true'].max()],
        [sbc_df['tau_true'].min(), sbc_df['tau_true'].max()],
        'r--', linewidth=2, label='Perfect recovery')

# Add error bars for uncertainty
for i in range(0, len(sbc_df), 5):
    ax.errorbar(sbc_df.iloc[i]['tau_true'], sbc_df.iloc[i]['tau_post_mean'],
               yerr=2*sbc_df.iloc[i]['tau_post_sd'], alpha=0.3, color='gray')

ax.set_xlabel('True tau', fontsize=11)
ax.set_ylabel('Posterior Mean tau', fontsize=11)
ax.set_title(f'tau Recovery\nRMSE: {np.sqrt(((sbc_df["tau_post_mean"] - sbc_df["tau_true"])**2).mean()):.4f}',
            fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 4: Shrinkage Calibration
# -------------------------------------------------------------------------
print("Plot 4: shrinkage_calibration.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Select representative groups for detailed visualization
representative_groups = [0, 3, 7, 11]  # Groups 1, 4, 8, 12

for plot_idx, j in enumerate(representative_groups):
    ax = axes[plot_idx // 2, plot_idx % 2]

    # Get true values and posterior means
    theta_true = sbc_df[f'theta_{j+1}_true'].values
    theta_post = sbc_df[f'theta_{j+1}_post_mean'].values
    mu_true = sbc_df['mu_true'].values

    # Plot
    scatter = ax.scatter(theta_true, theta_post, alpha=0.5, s=30, c=mu_true, cmap='viridis')
    ax.plot([theta_true.min(), theta_true.max()],
            [theta_true.min(), theta_true.max()],
            'r--', linewidth=2, label='No shrinkage')

    # Add shrinkage towards mu (every 10th point)
    for i in range(0, len(sbc_df), 10):
        ax.plot([theta_true[i], mu_true[i]],
               [theta_true[i], theta_true[i]],
               'k:', alpha=0.2, linewidth=0.5)

    ax.set_xlabel(f'True theta[{j+1}]', fontsize=10)
    ax.set_ylabel(f'Posterior Mean theta[{j+1}]', fontsize=10)
    ax.set_title(f'Group {j+1} Shrinkage Pattern', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add colorbar for mu
    if plot_idx == 0:
        plt.colorbar(scatter, ax=ax, label='True mu')

plt.suptitle('Shrinkage Calibration - Selected Groups', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shrinkage_calibration.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 5: Convergence Diagnostics
# -------------------------------------------------------------------------
print("Plot 5: convergence_diagnostics.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Convergence rate over iterations
ax = axes[0, 0]
window = 10
rolling_conv = pd.Series(sbc_df['converged'].values).rolling(window).mean() * 100
ax.plot(rolling_conv, linewidth=2, color='steelblue')
ax.axhline(80, color='red', linestyle='--', linewidth=2, label='Target (80%)')
ax.axhline(convergence_rate, color='green', linestyle='-', linewidth=2,
          label=f'Overall: {convergence_rate:.1f}%')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Convergence Rate (%)', fontsize=11)
ax.set_title(f'Rolling Convergence Rate (window={window})', fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)

# Recovery error vs iteration
ax = axes[0, 1]
mu_error = np.abs(sbc_df['mu_post_mean'] - sbc_df['mu_true'])
tau_error = np.abs(sbc_df['tau_post_mean'] - sbc_df['tau_true'])
ax.scatter(range(len(mu_error)), mu_error, alpha=0.5, s=20, label='mu', color='steelblue')
ax.scatter(range(len(tau_error)), tau_error, alpha=0.5, s=20, label='tau', color='orange')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Absolute Error', fontsize=11)
ax.set_title('Parameter Recovery Error Over Iterations', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale('log')

# Bias distribution
ax = axes[1, 0]
mu_bias = sbc_df['mu_post_mean'] - sbc_df['mu_true']
tau_bias = sbc_df['tau_post_mean'] - sbc_df['tau_true']
ax.hist(mu_bias, bins=30, alpha=0.6, label=f'mu (mean={mu_bias.mean():.4f})', edgecolor='black')
ax.hist(tau_bias, bins=30, alpha=0.6, label=f'tau (mean={tau_bias.mean():.4f})', edgecolor='black', color='orange')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('Bias (Posterior Mean - True)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Parameter Bias', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Uncertainty calibration
ax = axes[1, 1]
mu_z = mu_bias / sbc_df['mu_post_sd']
tau_z = tau_bias / sbc_df['tau_post_sd']
ax.hist(mu_z, bins=30, alpha=0.6, label=f'mu (SD={mu_z.std():.3f})', edgecolor='black', density=True)
ax.hist(tau_z, bins=30, alpha=0.6, label=f'tau (SD={tau_z.std():.3f})', edgecolor='black', color='orange', density=True)
x = np.linspace(-4, 4, 100)
ax.plot(x, stats.norm.pdf(x), 'r--', linewidth=2, label='N(0,1)')
ax.set_xlabel('Z-score (Bias / Posterior SD)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Uncertainty Calibration (Should be N(0,1))', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# PLOT 6: Bias Analysis by True Value
# -------------------------------------------------------------------------
print("Plot 6: bias_analysis.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# mu bias vs true value
ax = axes[0, 0]
mu_bias = sbc_df['mu_post_mean'] - sbc_df['mu_true']
ax.scatter(sbc_df['mu_true'], mu_bias, alpha=0.5, s=30)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.axhline(mu_bias.mean(), color='green', linestyle='-', linewidth=2,
          label=f'Mean bias: {mu_bias.mean():.4f}')
ax.set_xlabel('True mu', fontsize=11)
ax.set_ylabel('Bias (Posterior - True)', fontsize=11)
ax.set_title('mu: Bias vs True Value', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# tau bias vs true value
ax = axes[0, 1]
tau_bias = sbc_df['tau_post_mean'] - sbc_df['tau_true']
ax.scatter(sbc_df['tau_true'], tau_bias, alpha=0.5, s=30, color='orange')
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.axhline(tau_bias.mean(), color='green', linestyle='-', linewidth=2,
          label=f'Mean bias: {tau_bias.mean():.4f}')
ax.set_xlabel('True tau', fontsize=11)
ax.set_ylabel('Bias (Posterior - True)', fontsize=11)
ax.set_title('tau: Bias vs True Value', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Coverage by parameter value
ax = axes[1, 0]
# Bin mu_true and compute coverage in each bin
mu_bins = pd.cut(sbc_df['mu_true'], bins=5)
coverage_by_mu = sbc_df.groupby(mu_bins)['mu_in_90ci'].mean() * 100
bin_centers = [interval.mid for interval in coverage_by_mu.index]
ax.plot(bin_centers, coverage_by_mu.values, 'o-', linewidth=2, markersize=8, label='mu')
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target 90%')
ax.axhspan(85, 95, alpha=0.2, color='green', label='Target range')
ax.set_xlabel('True mu (binned)', fontsize=11)
ax.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Coverage Varies with True Value?', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Coverage by tau value
ax = axes[1, 1]
tau_bins = pd.cut(sbc_df['tau_true'], bins=5)
coverage_by_tau = sbc_df.groupby(tau_bins)['tau_in_90ci'].mean() * 100
bin_centers = [interval.mid for interval in coverage_by_tau.index]
ax.plot(bin_centers, coverage_by_tau.values, 'o-', linewidth=2, markersize=8, color='orange', label='tau')
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target 90%')
ax.axhspan(85, 95, alpha=0.2, color='green', label='Target range')
ax.set_xlabel('True tau (binned)', fontsize=11)
ax.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Coverage Varies with True Value?', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bias_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {PLOTS_DIR}")


# ============================================================================
# Save Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("Generating summary metrics file...")
print("=" * 80)

summary_stats = {
    'n_sbc_iterations': N_SBC_ITERATIONS,
    'n_successful': n_success,
    'n_converged': int(n_converged),
    'convergence_rate_pct': float(convergence_rate),
    'computational_pass': bool(computational_pass),
    'mu_coverage_90': float(mu_coverage_90),
    'mu_coverage_95': float(mu_coverage_95),
    'mu_coverage_pass': bool(mu_coverage_pass),
    'mu_ks_pvalue': float(ks_mu.pvalue),
    'mu_uniformity_pass': bool(ks_mu.pvalue > 0.05),
    'mu_bias_mean': float(mu_bias.mean()),
    'mu_rmse': float(np.sqrt(((sbc_df['mu_post_mean'] - sbc_df['mu_true'])**2).mean())),
    'tau_coverage_90': float(tau_coverage_90),
    'tau_coverage_95': float(tau_coverage_95),
    'tau_coverage_pass': bool(tau_coverage_pass),
    'tau_ks_pvalue': float(ks_tau.pvalue),
    'tau_uniformity_pass': bool(ks_tau.pvalue > 0.05),
    'tau_bias_mean': float(tau_bias.mean()),
    'tau_rmse': float(np.sqrt(((sbc_df['tau_post_mean'] - sbc_df['tau_true'])**2).mean())),
    'theta_mean_coverage_90': float(np.mean(theta_coverage_90)),
    'theta_coverage_pass': bool(theta_coverage_pass),
    'overall_pass': bool(overall_pass),
    'verdict': verdict,
    'method': 'MAP estimation with Laplace approximation',
    'note': 'Full MCMC would provide more robust validation but requires CmdStan/PyMC'
}

with open(DIAG_DIR / "summary_statistics.json", 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"Summary statistics saved to: {DIAG_DIR / 'summary_statistics.json'}")

print("\n" + "=" * 80)
print("SIMULATION-BASED CALIBRATION COMPLETE")
print("=" * 80)
print(f"\nFinal Verdict: {verdict}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"  - Raw results: {DIAG_DIR / 'sbc_results.csv'}")
print(f"  - Summary stats: {DIAG_DIR / 'summary_statistics.json'}")
print(f"  - Plots: {PLOTS_DIR}/")
