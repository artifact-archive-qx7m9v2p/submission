"""
Fit hierarchical model with skeptical priors (Model 4a) using Gibbs sampler
Skeptical of large effects, expects low heterogeneity

Since Stan compilation is not available, implementing Gibbs sampler directly.
"""

import numpy as np
import pandas as pd
import arviz as az
from scipy import stats
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Set up paths
exp_dir = Path('/workspace/experiments/experiment_4/experiment_4a_skeptical/posterior_inference')
code_dir = exp_dir / 'code'
diag_dir = exp_dir / 'diagnostics'
plot_dir = exp_dir / 'plots'

# Set random seed
np.random.seed(12345)

# Load data
data_path = Path('/workspace/data/data.csv')
df = pd.read_csv(data_path)
J = len(df)
y = df['y'].values
sigma = df['sigma'].values

print("=" * 80)
print("MODEL 4a: SKEPTICAL PRIORS (Gibbs Sampler)")
print("=" * 80)
print(f"\nData: {J} studies")
print(f"Observed effects: {y}")
print(f"Standard errors: {sigma}")
print("\nPrior specification:")
print("  mu ~ Normal(0, 10)      [Skeptical of large effects]")
print("  tau ~ Half-Normal(0, 5) [Expects low heterogeneity]")
print()

# Prior hyperparameters
mu_prior_mean = 0     # Skeptical of large effects
mu_prior_sd = 10
tau_prior_sd = 5      # Expects low heterogeneity

# Gibbs sampler for hierarchical normal model
def gibbs_hierarchical(y, sigma, mu_prior_mean, mu_prior_sd, tau_prior_sd,
                       n_chains=4, n_iter=5000, n_warmup=2500, seed=12345):
    """
    Gibbs sampler for hierarchical model:
    y_i ~ N(theta_i, sigma_i^2)
    theta_i ~ N(mu, tau^2)
    mu ~ N(mu_prior_mean, mu_prior_sd^2)
    tau ~ Half-Normal(0, tau_prior_sd^2)
    """

    J = len(y)
    n_samples = n_iter - n_warmup

    # Storage for all chains
    all_chains = {
        'mu': np.zeros((n_chains, n_samples)),
        'tau': np.zeros((n_chains, n_samples)),
        'theta': np.zeros((n_chains, n_samples, J))
    }

    print(f"Running Gibbs sampler: {n_chains} chains × {n_iter} iterations")
    print(f"Warmup: {n_warmup}, Sampling: {n_samples}")

    for chain in range(n_chains):
        print(f"\nChain {chain + 1}/{n_chains}...")

        # Set seed for reproducibility
        np.random.seed(seed + chain)

        # Initialize
        theta = y.copy()  # Initialize at observed values
        mu = np.mean(y)
        tau = np.std(y)

        # Storage for this chain
        mu_samples = np.zeros(n_iter)
        tau_samples = np.zeros(n_iter)
        theta_samples = np.zeros((n_iter, J))

        # Gibbs sampling
        for t in range(n_iter):
            # 1. Sample theta_i | mu, tau, y_i
            for j in range(J):
                # Posterior precision is sum of precisions
                prec_prior = 1 / tau**2
                prec_data = 1 / sigma[j]**2
                prec_post = prec_prior + prec_data

                # Posterior mean is precision-weighted average
                mean_post = (prec_prior * mu + prec_data * y[j]) / prec_post
                sd_post = np.sqrt(1 / prec_post)

                theta[j] = np.random.normal(mean_post, sd_post)

            # 2. Sample mu | theta, tau
            # Posterior: N(mean_post, sd_post^2)
            prec_prior = 1 / mu_prior_sd**2
            prec_data = J / tau**2
            prec_post = prec_prior + prec_data

            mean_post = (prec_prior * mu_prior_mean + prec_data * np.mean(theta)) / prec_post
            sd_post = np.sqrt(1 / prec_post)

            mu = np.random.normal(mean_post, sd_post)

            # 3. Sample tau | mu, theta
            # Using Metropolis-Hastings since no conjugate prior
            # Proposal: lognormal random walk
            tau_prop = tau * np.exp(np.random.normal(0, 0.3))

            # Log posterior for tau
            def log_post_tau(tau_val):
                if tau_val <= 0:
                    return -np.inf
                # Prior: Half-Normal(0, tau_prior_sd)
                log_prior = stats.halfnorm.logpdf(tau_val, scale=tau_prior_sd)
                # Likelihood: theta_i ~ N(mu, tau^2)
                log_lik = np.sum(stats.norm.logpdf(theta, mu, tau_val))
                return log_prior + log_lik

            log_alpha = log_post_tau(tau_prop) - log_post_tau(tau)

            if np.log(np.random.uniform()) < log_alpha:
                tau = tau_prop

            # Store samples
            mu_samples[t] = mu
            tau_samples[t] = tau
            theta_samples[t, :] = theta

            if (t + 1) % 1000 == 0:
                print(f"  Iteration {t + 1}/{n_iter}")

        # Store post-warmup samples
        all_chains['mu'][chain, :] = mu_samples[n_warmup:]
        all_chains['tau'][chain, :] = tau_samples[n_warmup:]
        all_chains['theta'][chain, :, :] = theta_samples[n_warmup:, :]

    return all_chains

# Run Gibbs sampler
print("\n" + "=" * 80)
print("PHASE 1: PROBE RUN (4 chains × 200 iterations)")
print("=" * 80)

probe_samples = gibbs_hierarchical(
    y, sigma, mu_prior_mean, mu_prior_sd, tau_prior_sd,
    n_chains=4, n_iter=200, n_warmup=100, seed=12345
)

# Quick check
probe_mu = probe_samples['mu'].flatten()
probe_rhat = az.rhat(az.convert_to_dataset({'mu': probe_samples['mu']}))['mu'].values
print(f"\nProbe R-hat for mu: {probe_rhat:.4f}")
print(f"Probe mean mu: {probe_mu.mean():.2f} ± {probe_mu.std():.2f}")

# MAIN RUN
print("\n" + "=" * 80)
print("PHASE 2: MAIN SAMPLING (4 chains × 5000 iterations)")
print("=" * 80)

samples = gibbs_hierarchical(
    y, sigma, mu_prior_mean, mu_prior_sd, tau_prior_sd,
    n_chains=4, n_iter=5000, n_warmup=2500, seed=12345
)

# Convert to ArviZ InferenceData format
print("\n" + "=" * 80)
print("Converting to ArviZ InferenceData...")
print("=" * 80)

# Reshape for ArviZ: (chain, draw, *shape)
posterior_dict = {
    'mu': samples['mu'],  # (n_chains, n_samples)
    'tau': samples['tau'],  # (n_chains, n_samples)
    'theta': samples['theta'],  # (n_chains, n_samples, J)
}

# Compute log-likelihood manually
log_lik = np.zeros((4, samples['mu'].shape[1], J))
for chain in range(4):
    for draw in range(samples['mu'].shape[1]):
        for j in range(J):
            theta_j = samples['theta'][chain, draw, j]
            log_lik[chain, draw, j] = stats.norm.logpdf(y[j], theta_j, sigma[j])

idata = az.from_dict(
    posterior=posterior_dict,
    log_likelihood={'y_obs': log_lik}
)

# Save InferenceData
netcdf_path = diag_dir / 'posterior_inference.netcdf'
idata.to_netcdf(netcdf_path)
print(f"Saved InferenceData to: {netcdf_path}")

# Compute summary statistics
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])
print(summary)

summary_path = diag_dir / 'posterior_summary.csv'
summary.to_csv(summary_path)
print(f"\nSaved summary to: {summary_path}")

# Extract key parameters
mu_samples = samples['mu'].flatten()
tau_samples = samples['tau'].flatten()

mu_mean = mu_samples.mean()
mu_sd = mu_samples.std()
mu_ci = np.percentile(mu_samples, [2.5, 97.5])

tau_mean = tau_samples.mean()
tau_sd = tau_samples.std()
tau_ci = np.percentile(tau_samples, [2.5, 97.5])

print("\n" + "=" * 80)
print("POSTERIOR ESTIMATES")
print("=" * 80)
print(f"\nmu (population mean):")
print(f"  Posterior: {mu_mean:.2f} ± {mu_sd:.2f}")
print(f"  95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]")
print(f"  Prior: 0 ± 10 (skeptical)")

print(f"\ntau (population SD):")
print(f"  Posterior: {tau_mean:.2f} ± {tau_sd:.2f}")
print(f"  95% CI: [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]")
print(f"  Prior: Half-Normal(0, 5)")

prior_shift_mu = abs(mu_mean - 0)
print(f"\nPrior-posterior shift:")
print(f"  mu shifted {prior_shift_mu:.2f} units from prior mean (0)")

# Check convergence
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()

print("\n" + "=" * 80)
print("CONVERGENCE ASSESSMENT")
print("=" * 80)
print(f"Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"Min ESS_bulk: {min_ess_bulk:.1f} (target: > 400)")
print(f"Min ESS_tail: {min_ess_tail:.1f} (target: > 400)")

convergence_ok = max_rhat < 1.01 and min_ess_bulk > 400 and min_ess_tail > 400

if convergence_ok:
    print("\nAll convergence criteria met!")
else:
    print("\nSome convergence criteria not met - see diagnostics")

# Save convergence report
convergence_report = f"""# Convergence Report: Model 4a (Skeptical Priors)

## Sampling Configuration
- Sampler: Custom Gibbs Sampler
- Chains: 4
- Warmup iterations: 2500
- Sampling iterations: 2500

## Quantitative Diagnostics

### Convergence Metrics
- Max R-hat: {max_rhat:.4f} (target: < 1.01) {'✓' if max_rhat < 1.01 else '✗'}
- Min ESS_bulk: {min_ess_bulk:.1f} (target: > 400) {'✓' if min_ess_bulk > 400 else '✗'}
- Min ESS_tail: {min_ess_tail:.1f} (target: > 400) {'✓' if min_ess_tail > 400 else '✗'}

### Overall Assessment
{('All convergence criteria met. Chains mixed well and explored the posterior efficiently.'
  if convergence_ok else
  'Some convergence criteria not met. See visual diagnostics for details.')}

## Posterior Estimates

### mu (population mean)
- Posterior: {mu_mean:.2f} ± {mu_sd:.2f}
- 95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]
- Prior: Normal(0, 10) [skeptical]
- Shift from prior: {prior_shift_mu:.2f} units

### tau (population SD)
- Posterior: {tau_mean:.2f} ± {tau_sd:.2f}
- 95% CI: [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]
- Prior: Half-Normal(0, 5) [low heterogeneity]

## Visual Diagnostics

See plots/ directory for:
- convergence_overview.png: Trace and rank plots for key parameters
- prior_posterior_overlay.png: Prior vs posterior comparison
- forest_plot.png: Study-specific effects

"""

report_path = diag_dir / 'convergence_report.md'
with open(report_path, 'w') as f:
    f.write(convergence_report)
print(f"\nSaved convergence report to: {report_path}")

# Save results as JSON
results = {
    'model': 'skeptical',
    'mu': {
        'mean': float(mu_mean),
        'sd': float(mu_sd),
        'ci_lower': float(mu_ci[0]),
        'ci_upper': float(mu_ci[1])
    },
    'tau': {
        'mean': float(tau_mean),
        'sd': float(tau_sd),
        'ci_lower': float(tau_ci[0]),
        'ci_upper': float(tau_ci[1])
    },
    'convergence': {
        'max_rhat': float(max_rhat),
        'min_ess_bulk': float(min_ess_bulk),
        'min_ess_tail': float(min_ess_tail),
        'converged': convergence_ok
    }
}

json_path = diag_dir / 'results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved results to: {json_path}")

print("\n" + "=" * 80)
print("FITTING COMPLETE - Model 4a (Skeptical)")
print("=" * 80)
