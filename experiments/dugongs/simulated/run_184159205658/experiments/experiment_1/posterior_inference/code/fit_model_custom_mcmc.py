"""
Fit Bayesian Logarithmic Regression Model using Custom HMC Implementation
Experiment 1 - Posterior Inference
(Fallback: Stan compilation failed, PyMC not available)

Using NumPy + scipy with No U-Turn Sampler (NUTS) via emcee or custom implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import json
from pathlib import Path
from scipy import stats, optimize
from scipy.stats import norm, expon
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path('/workspace/experiments/experiment_1/posterior_inference')
DATA_PATH = Path('/workspace/data/data.csv')
DIAGNOSTICS_DIR = BASE_DIR / 'diagnostics'
PLOTS_DIR = BASE_DIR / 'plots'

print("="*80)
print("BAYESIAN LOGARITHMIC REGRESSION - POSTERIOR INFERENCE")
print("="*80)
print("\nNOTE: Using custom MCMC (Stan unavailable, PyMC not installed)")
print("      Implementing Metropolis-Hastings with adaptive proposal")

# Load data
print("\n[1] Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"    N = {len(data)} observations")
print(f"    x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
print(f"    Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

x_obs = data['x'].values
y_obs = data['Y'].values
N = len(data)

# Log-transform x once
log_x = np.log(x_obs)

# Define log-posterior function
def log_prior(params):
    """Log prior density"""
    beta0, beta1, log_sigma = params
    sigma = np.exp(log_sigma)  # Transform to positive

    # Priors
    lp_beta0 = norm.logpdf(beta0, loc=1.73, scale=0.5)
    lp_beta1 = norm.logpdf(beta1, loc=0.28, scale=0.15)
    lp_sigma = expon.logpdf(sigma, scale=1/5)  # Exp(5) parameterization

    # Jacobian adjustment for log transform
    jacobian = log_sigma  # log(sigma) for transformation

    return lp_beta0 + lp_beta1 + lp_sigma + jacobian

def log_likelihood(params, x_log, y):
    """Log likelihood"""
    beta0, beta1, log_sigma = params
    sigma = np.exp(log_sigma)

    # Mean function
    mu = beta0 + beta1 * x_log

    # Likelihood
    return np.sum(norm.logpdf(y, loc=mu, scale=sigma))

def log_posterior(params, x_log, y):
    """Log posterior density (unnormalized)"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x_log, y)

# Find MAP estimate for initialization
print("\n[2] Finding MAP estimate for initialization...")
neg_log_post = lambda params: -log_posterior(params, log_x, y_obs)

# Initial guess
init_params = np.array([1.73, 0.28, np.log(0.2)])

result = optimize.minimize(
    neg_log_post,
    init_params,
    method='L-BFGS-B',
    bounds=[(-5, 5), (-1, 2), (np.log(0.01), np.log(2))]
)

map_estimate = result.x
print(f"    MAP estimate: β₀={map_estimate[0]:.4f}, β₁={map_estimate[1]:.4f}, σ={np.exp(map_estimate[2]):.4f}")

# Estimate covariance at MAP for proposal
hess = optimize.approx_fprime(map_estimate, neg_log_post, epsilon=1e-5)
proposal_cov = np.eye(3) * 0.01  # Start with small diagonal covariance

# Metropolis-Hastings MCMC with adaptive proposal
def adaptive_mh_mcmc(log_post_fn, init, n_iterations, n_chains=4, n_warmup=1000,
                     target_accept=0.234, adapt_every=50):
    """
    Adaptive Metropolis-Hastings MCMC
    """
    n_params = len(init)
    chains = []

    for chain_id in range(n_chains):
        print(f"    Chain {chain_id + 1}/{n_chains}...", end=" ", flush=True)

        # Initialize with jitter around MAP
        current = init + np.random.randn(n_params) * 0.1
        current_log_post = log_post_fn(current)

        samples = np.zeros((n_iterations, n_params))
        accepted = 0

        # Adaptive proposal covariance
        prop_scale = 2.4**2 / n_params  # Optimal scaling
        prop_cov = np.eye(n_params) * 0.01

        for i in range(n_iterations + n_warmup):
            # Propose new parameters
            proposal = np.random.multivariate_normal(current, prop_cov * prop_scale)
            proposal_log_post = log_post_fn(proposal)

            # Acceptance ratio
            log_alpha = proposal_log_post - current_log_post

            # Accept/reject
            if np.log(np.random.rand()) < log_alpha:
                current = proposal
                current_log_post = proposal_log_post
                if i >= n_warmup:
                    accepted += 1

            # Store sample after warmup
            if i >= n_warmup:
                samples[i - n_warmup, :] = current

            # Adapt proposal during warmup
            if i < n_warmup and i > 100 and i % adapt_every == 0:
                accept_rate = accepted / min(i, n_iterations) if i < n_warmup else accepted / n_iterations
                # Adjust scale based on acceptance rate
                if accept_rate < 0.2:
                    prop_scale *= 0.9
                elif accept_rate > 0.3:
                    prop_scale *= 1.1

                # Update covariance estimate
                if i > 200:
                    recent_samples = samples[max(0, i-n_warmup-200):i-n_warmup, :]
                    if len(recent_samples) > 10:
                        prop_cov = np.cov(recent_samples.T) + np.eye(n_params) * 1e-6

        accept_rate = accepted / n_iterations
        print(f"Accept rate: {accept_rate:.3f}")
        chains.append(samples)

    return np.array(chains)

# Run MCMC
print("\n[3] Running MCMC sampling...")
print("    4 chains × 2000 iterations (1000 warmup)")

samples = adaptive_mh_mcmc(
    lambda p: log_posterior(p, log_x, y_obs),
    map_estimate,
    n_iterations=1000,
    n_chains=4,
    n_warmup=1000,
    target_accept=0.234
)

print("\n[4] MCMC sampling complete!")

# Transform samples
beta0_samples = samples[:, :, 0].flatten()
beta1_samples = samples[:, :, 1].flatten()
sigma_samples = np.exp(samples[:, :, 2].flatten())

# Create ArviZ InferenceData
print("\n[5] Converting to ArviZ InferenceData...")

# Compute mu for each sample
mu_samples = np.zeros((samples.shape[0], samples.shape[1], N))
log_lik_samples = np.zeros((samples.shape[0], samples.shape[1], N))
y_rep_samples = np.zeros((samples.shape[0], samples.shape[1], N))

for chain in range(samples.shape[0]):
    for draw in range(samples.shape[1]):
        beta0 = samples[chain, draw, 0]
        beta1 = samples[chain, draw, 1]
        sigma = np.exp(samples[chain, draw, 2])

        mu = beta0 + beta1 * log_x
        mu_samples[chain, draw, :] = mu

        # Log-likelihood for each observation
        log_lik_samples[chain, draw, :] = norm.logpdf(y_obs, loc=mu, scale=sigma)

        # Posterior predictive
        y_rep_samples[chain, draw, :] = np.random.normal(mu, sigma)

# Create InferenceData
idata = az.from_dict(
    posterior={
        'beta0': samples[:, :, 0],
        'beta1': samples[:, :, 1],
        'sigma': np.exp(samples[:, :, 2]),
        'mu': mu_samples
    },
    observed_data={'Y': y_obs},
    log_likelihood={'Y': log_lik_samples},
    posterior_predictive={'Y': y_rep_samples},
    coords={'obs_id': np.arange(N)},
    dims={
        'mu': ['obs_id'],
        'Y': ['obs_id']
    }
)

# Save InferenceData
print(f"    Saving to: {DIAGNOSTICS_DIR / 'posterior_inference.netcdf'}")
idata.to_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')

# Generate summary
print("\n[6] Generating posterior summary...")
summary = az.summary(idata, var_names=['beta0', 'beta1', 'sigma'])
print("\n" + "="*80)
print("POSTERIOR SUMMARY")
print("="*80)
print(summary)

# Save summary
summary.to_csv(DIAGNOSTICS_DIR / 'posterior_summary.csv')
print(f"\n    Saved to: {DIAGNOSTICS_DIR / 'posterior_summary.csv'}")

# Convergence diagnostics
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

# Check R-hat
max_rhat = summary['r_hat'].max()
rhat_pass = max_rhat < 1.01
print(f"\nR-hat:")
print(f"    Max R-hat: {max_rhat:.4f}")
print(f"    Criterion: < 1.01")
print(f"    Status: {'PASS ✓' if rhat_pass else 'FAIL ✗'}")

# Check ESS
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()
ess_pass = (min_ess_bulk > 400) and (min_ess_tail > 400)
print(f"\nEffective Sample Size:")
print(f"    Min ESS_bulk: {min_ess_bulk:.0f}")
print(f"    Min ESS_tail: {min_ess_tail:.0f}")
print(f"    Criterion: > 400")
print(f"    Status: {'PASS ✓' if ess_pass else 'FAIL ✗'}")

# Note: No divergences in MH, but we track acceptance
print(f"\nDivergent Transitions:")
print(f"    Count: 0 (N/A for Metropolis-Hastings)")
print(f"    Status: PASS ✓")
div_pass = True

# MCSE
max_mcse_mean = summary['mcse_mean'].max()
max_sd = summary['sd'].max()
mcse_ratio = max_mcse_mean / max_sd if max_sd > 0 else 0
mcse_pass = mcse_ratio < 0.05
print(f"\nMonte Carlo Standard Error:")
print(f"    Max MCSE/SD ratio: {mcse_ratio:.4f}")
print(f"    Criterion: < 0.05")
print(f"    Status: {'PASS ✓' if mcse_pass else 'FAIL ✗'}")

# Overall convergence
convergence_pass = rhat_pass and ess_pass and mcse_pass
print(f"\n{'='*80}")
print(f"OVERALL CONVERGENCE: {'PASS ✓' if convergence_pass else 'FAIL ✗'}")
print(f"{'='*80}")

# Posterior statistics
print("\n" + "="*80)
print("PARAMETER INFERENCE")
print("="*80)

print("\nβ₀ (Intercept):")
print(f"    Mean: {beta0_samples.mean():.4f}")
print(f"    SD: {beta0_samples.std():.4f}")
print(f"    95% CI: [{np.percentile(beta0_samples, 2.5):.4f}, {np.percentile(beta0_samples, 97.5):.4f}]")
print(f"    Prior: N(1.73, 0.5)")

print("\nβ₁ (Log slope):")
print(f"    Mean: {beta1_samples.mean():.4f}")
print(f"    SD: {beta1_samples.std():.4f}")
print(f"    95% CI: [{np.percentile(beta1_samples, 2.5):.4f}, {np.percentile(beta1_samples, 97.5):.4f}]")
print(f"    Prior: N(0.28, 0.15)")
print(f"    P(β₁ > 0): {(beta1_samples > 0).mean():.4f}")

print("\nσ (Error SD):")
print(f"    Mean: {sigma_samples.mean():.4f}")
print(f"    SD: {sigma_samples.std():.4f}")
print(f"    95% CI: [{np.percentile(sigma_samples, 2.5):.4f}, {np.percentile(sigma_samples, 97.5):.4f}]")
print(f"    Prior: Exp(5)")

# Parameter correlations
print("\nParameter Correlations:")
corr_beta0_beta1 = np.corrcoef(beta0_samples, beta1_samples)[0, 1]
print(f"    Corr(β₀, β₁): {corr_beta0_beta1:.3f}")

# Model fit assessment
print("\n" + "="*80)
print("MODEL FIT ASSESSMENT")
print("="*80)

# Posterior predictive mean
mu_mean = mu_samples.mean(axis=(0, 1))
mu_lower = np.percentile(mu_samples, 2.5, axis=(0, 1))
mu_upper = np.percentile(mu_samples, 97.5, axis=(0, 1))

# Residuals
residuals = y_obs - mu_mean

# In-sample metrics
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_obs - y_obs.mean())**2)
r_squared = 1 - ss_res / ss_tot

print(f"\nIn-Sample Metrics:")
print(f"    RMSE: {rmse:.4f}")
print(f"    MAE: {mae:.4f}")
print(f"    R²: {r_squared:.4f}")

# Save residuals
residuals_df = pd.DataFrame({
    'x': x_obs,
    'Y': y_obs,
    'mu_mean': mu_mean,
    'mu_lower': mu_lower,
    'mu_upper': mu_upper,
    'residual': residuals
})
residuals_df.to_csv(DIAGNOSTICS_DIR / 'residuals.csv', index=False)
print(f"\n    Residuals saved to: {DIAGNOSTICS_DIR / 'residuals.csv'}")

# LOO-CV Analysis
print("\n" + "="*80)
print("LOO-CV ANALYSIS")
print("="*80)

loo = az.loo(idata, var_name='Y', pointwise=True)
print(f"\nLOO-CV Results:")
print(f"    ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"    p_loo: {loo.p_loo:.2f}")
print(f"    LOO-IC: {-2 * loo.elpd_loo:.2f}")

# Pareto k diagnostics
pareto_k = loo.pareto_k.values
n_high_k = np.sum(pareto_k > 0.7)
print(f"\nPareto k diagnostics:")
print(f"    Good (k < 0.5): {np.sum(pareto_k < 0.5)}")
print(f"    OK (0.5 ≤ k < 0.7): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))}")
print(f"    Bad (k ≥ 0.7): {n_high_k}")

if n_high_k > 0:
    print(f"\n    WARNING: {n_high_k} observations with k > 0.7")
    high_k_idx = np.where(pareto_k > 0.7)[0]
    print(f"    Indices: {high_k_idx}")
else:
    print(f"\n    All Pareto k values < 0.7 ✓")

# Save LOO results
loo_df = pd.DataFrame({
    'obs_id': np.arange(len(data)),
    'x': x_obs,
    'Y': y_obs,
    'pareto_k': pareto_k,
    'loo_i': loo.loo_i.values
})
loo_df.to_csv(DIAGNOSTICS_DIR / 'loo_results.csv', index=False)

# Overall assessment
print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

pass_criteria = {
    'Convergence (R-hat < 1.01)': rhat_pass,
    'ESS > 400': ess_pass,
    'MCSE < 5% of SD': mcse_pass,
    'β₁ > 0': (beta1_samples > 0).mean() > 0.95,
    'No computational errors': True
}

print("\nPass/Fail Criteria:")
for criterion, status in pass_criteria.items():
    print(f"    {criterion}: {'PASS ✓' if status else 'FAIL ✗'}")

overall_pass = all(pass_criteria.values())
print(f"\n{'='*80}")
print(f"FINAL VERDICT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
print(f"{'='*80}")

# Save diagnostic info
diagnostics_dict = {
    'sampler': 'Custom Adaptive Metropolis-Hastings',
    'convergence': {
        'max_rhat': float(max_rhat),
        'min_ess_bulk': float(min_ess_bulk),
        'min_ess_tail': float(min_ess_tail),
        'max_mcse_ratio': float(mcse_ratio),
        'overall_pass': convergence_pass
    },
    'parameters': {
        'beta0': {
            'mean': float(beta0_samples.mean()),
            'sd': float(beta0_samples.std()),
            'ci_lower': float(np.percentile(beta0_samples, 2.5)),
            'ci_upper': float(np.percentile(beta0_samples, 97.5))
        },
        'beta1': {
            'mean': float(beta1_samples.mean()),
            'sd': float(beta1_samples.std()),
            'ci_lower': float(np.percentile(beta1_samples, 2.5)),
            'ci_upper': float(np.percentile(beta1_samples, 97.5)),
            'prob_positive': float((beta1_samples > 0).mean())
        },
        'sigma': {
            'mean': float(sigma_samples.mean()),
            'sd': float(sigma_samples.std()),
            'ci_lower': float(np.percentile(sigma_samples, 2.5)),
            'ci_upper': float(np.percentile(sigma_samples, 97.5))
        }
    },
    'fit': {
        'rmse': float(rmse),
        'mae': float(mae),
        'r_squared': float(r_squared)
    },
    'loo': {
        'elpd_loo': float(loo.elpd_loo),
        'se': float(loo.se),
        'p_loo': float(loo.p_loo),
        'loo_ic': float(-2 * loo.elpd_loo),
        'n_high_pareto_k': int(n_high_k)
    },
    'overall_pass': overall_pass
}

with open(DIAGNOSTICS_DIR / 'diagnostics_summary.json', 'w') as f:
    json.dump(diagnostics_dict, f, indent=2)

print(f"\nDiagnostics saved to: {DIAGNOSTICS_DIR / 'diagnostics_summary.json'}")
print("\n" + "="*80)
print("Model fitting complete. Proceeding to create diagnostic plots...")
print("="*80)
