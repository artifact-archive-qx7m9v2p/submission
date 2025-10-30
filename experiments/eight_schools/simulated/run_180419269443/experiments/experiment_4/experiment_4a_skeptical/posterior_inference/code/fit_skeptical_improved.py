"""
Fit hierarchical model with skeptical priors (Model 4a)
Improved Gibbs sampler with non-centered parameterization
"""

import numpy as np
import pandas as pd
import arviz as az
from scipy import stats
from pathlib import Path
import json

# Set paths
exp_dir = Path('/workspace/experiments/experiment_4/experiment_4a_skeptical/posterior_inference')
code_dir = exp_dir / 'code'
diag_dir = exp_dir / 'diagnostics'
plot_dir = exp_dir / 'plots'

# Load data
df = pd.read_csv('/workspace/data/data.csv')
J = len(df)
y = df['y'].values
sigma = df['sigma'].values

print("=" * 80)
print("MODEL 4a: SKEPTICAL PRIORS (Improved Gibbs)")
print("=" * 80)
print(f"\nData: {J} studies")
print(f"Observed: {y}")
print(f"Priors: mu ~ N(0,10), tau ~ HalfN(0,5)")
print()

# Gibbs with non-centered parameterization
def gibbs_noncentered(y, sigma, n_iter=10000, n_warmup=5000, thin=2, seed=12345):
    """Non-centered parameterization: theta_i = mu + tau * eta_i, eta_i ~ N(0,1)"""

    np.random.seed(seed)
    J = len(y)

    # Initialize
    mu = np.mean(y)
    tau = np.std(y)
    eta = np.zeros(J)  # Standardized effects

    # Storage (after thinning)
    n_keep = (n_iter - n_warmup) // thin
    mu_samples = np.zeros(n_keep)
    tau_samples = np.zeros(n_keep)
    eta_samples = np.zeros((n_keep, J))

    accept_tau = 0
    keep_idx = 0

    print(f"Running: {n_iter} iterations, warmup={n_warmup}, thin={thin}")

    for t in range(n_iter):
        theta = mu + tau * eta  # Reconstruct theta

        # 1. Sample eta_i | mu, tau, y
        for j in range(J):
            # Posterior precision
            prec_prior = 1.0
            prec_data = tau**2 / sigma[j]**2
            prec_post = prec_prior + prec_data

            # Posterior mean
            mean_post = prec_data * (y[j] - mu) / (tau * prec_post)
            sd_post = 1.0 / np.sqrt(prec_post)

            eta[j] = np.random.normal(mean_post, sd_post)

        # 2. Sample mu | eta, tau, y
        theta = mu + tau * eta
        prec_prior = 1 / 100  # Prior: N(0, 10^2)
        prec_data = np.sum(1 / sigma**2)
        prec_post = prec_prior + prec_data

        weighted_mean = np.sum((y - tau * eta) / sigma**2)
        mean_post = weighted_mean / prec_post
        sd_post = 1 / np.sqrt(prec_post)

        mu = np.random.normal(mean_post, sd_post)

        # 3. Sample tau using Metropolis-Hastings
        tau_prop = tau * np.exp(np.random.normal(0, 0.2))

        def log_post_tau(tau_val):
            if tau_val <= 0:
                return -np.inf
            # Prior: Half-Normal(0, 5)
            log_prior = stats.halfnorm.logpdf(tau_val, scale=5)
            # Likelihood from eta (standard normal)
            log_lik_eta = np.sum(stats.norm.logpdf(eta, 0, 1))
            # Likelihood from y
            theta_temp = mu + tau_val * eta
            log_lik_y = np.sum(stats.norm.logpdf(y, theta_temp, sigma))
            return log_prior + log_lik_y

        log_alpha = log_post_tau(tau_prop) - log_post_tau(tau)

        if np.log(np.random.uniform()) < log_alpha:
            tau = tau_prop
            if t >= n_warmup:
                accept_tau += 1

        # Store samples (after warmup, with thinning)
        if t >= n_warmup and (t - n_warmup) % thin == 0:
            mu_samples[keep_idx] = mu
            tau_samples[keep_idx] = tau
            eta_samples[keep_idx, :] = eta
            keep_idx += 1

        if (t + 1) % 2000 == 0:
            print(f"  Iteration {t + 1}/{n_iter}")

    accept_rate = accept_tau / ((n_iter - n_warmup) / thin) if n_iter > n_warmup else 0
    print(f"  Tau acceptance rate: {accept_rate:.2%}")

    # Reconstruct theta
    theta_samples = mu_samples[:, np.newaxis] + tau_samples[:, np.newaxis] * eta_samples

    return {'mu': mu_samples, 'tau': tau_samples, 'theta': theta_samples}

# Run for multiple chains
print("\nRunning 4 chains...")
n_chains = 4
all_chains = {'mu': [], 'tau': [], 'theta': []}

for chain in range(n_chains):
    print(f"\nChain {chain + 1}/{n_chains}")
    samples = gibbs_noncentered(y, sigma, n_iter=10000, n_warmup=5000, thin=2, seed=12345 + chain)
    all_chains['mu'].append(samples['mu'])
    all_chains['tau'].append(samples['tau'])
    all_chains['theta'].append(samples['theta'])

# Convert to arrays
for key in all_chains:
    all_chains[key] = np.array(all_chains[key])

print("\n" + "=" * 80)
print("Converting to ArviZ...")
print("=" * 80)

# Compute log-likelihood
n_samples = all_chains['mu'].shape[1]
log_lik = np.zeros((n_chains, n_samples, J))
for chain in range(n_chains):
    for draw in range(n_samples):
        for j in range(J):
            theta_j = all_chains['theta'][chain, draw, j]
            log_lik[chain, draw, j] = stats.norm.logpdf(y[j], theta_j, sigma[j])

idata = az.from_dict(
    posterior=all_chains,
    log_likelihood={'y_obs': log_lik}
)

# Save
netcdf_path = diag_dir / 'posterior_inference.netcdf'
idata.to_netcdf(netcdf_path)
print(f"Saved to: {netcdf_path}")

# Diagnostics
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])
print(summary)

summary.to_csv(diag_dir / 'posterior_summary.csv')

# Extract results
mu_samples = all_chains['mu'].flatten()
tau_samples = all_chains['tau'].flatten()

mu_mean = mu_samples.mean()
mu_sd = mu_samples.std()
mu_ci = np.percentile(mu_samples, [2.5, 97.5])

tau_mean = tau_samples.mean()
tau_sd = tau_samples.std()
tau_ci = np.percentile(tau_samples, [2.5, 97.5])

print("\n" + "=" * 80)
print("POSTERIOR ESTIMATES")
print("=" * 80)
print(f"\nmu: {mu_mean:.2f} ± {mu_sd:.2f}, 95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]")
print(f"tau: {tau_mean:.2f} ± {tau_sd:.2f}, 95% CI: [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]")
print(f"Shift from skeptical prior (0): {mu_mean:.2f} units")

# Convergence
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()

print(f"\nMax R-hat: {max_rhat:.4f}")
print(f"Min ESS: {min_ess_bulk:.0f} (bulk), {min_ess_tail:.0f} (tail)")

convergence_ok = max_rhat < 1.01 and min_ess_bulk > 400

if convergence_ok:
    print("✓ Convergence criteria met")
else:
    print("⚠ Convergence issues detected")

# Save results
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
        'converged': bool(convergence_ok)  # Explicit bool conversion
    }
}

with open(diag_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save report
report = f"""# Convergence Report: Model 4a (Skeptical Priors)

## Sampling Configuration
- Sampler: Gibbs (non-centered parameterization)
- Chains: 4
- Iterations: 10000 (5000 warmup, thin=2)
- Total samples: {n_samples * n_chains}

## Convergence Metrics
- Max R-hat: {max_rhat:.4f} {'✓' if max_rhat < 1.01 else '✗'}
- Min ESS_bulk: {min_ess_bulk:.0f} {'✓' if min_ess_bulk > 400 else '✗'}
- Min ESS_tail: {min_ess_tail:.0f} {'✓' if min_ess_tail > 400 else '✗'}

## Posterior Estimates
**mu (population mean):**
- Posterior: {mu_mean:.2f} ± {mu_sd:.2f}
- 95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]
- Prior: N(0, 10) [skeptical]
- Shift: {mu_mean:.2f} units from prior mean

**tau (population SD):**
- Posterior: {tau_mean:.2f} ± {tau_sd:.2f}
- 95% CI: [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]
- Prior: Half-Normal(0, 5)

## Visual Diagnostics
See plots/ directory for diagnostic plots.
"""

with open(diag_dir / 'convergence_report.md', 'w') as f:
    f.write(report)

print("\n" + "=" * 80)
print("FITTING COMPLETE - Model 4a (Skeptical)")
print("=" * 80)
