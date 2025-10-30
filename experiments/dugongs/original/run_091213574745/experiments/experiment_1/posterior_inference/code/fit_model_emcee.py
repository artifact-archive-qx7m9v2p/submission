"""
Fit Logarithmic Model with Normal Likelihood using MCMC (emcee)

Model: Y_i ~ Normal(β₀ + β₁*log(x_i), σ)
Priors:
  β₀ ~ Normal(2.3, 0.3)
  β₁ ~ Normal(0.29, 0.15)
  σ ~ Exponential(10)

Uses emcee (affine-invariant ensemble sampler) as a PPL alternative
when Stan/PyMC compilation is not available.
"""

import numpy as np
import pandas as pd
import emcee
import arviz as az
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LOGARITHMIC MODEL - POSTERIOR INFERENCE (emcee)")
print("=" * 80)
print("\nNOTE: Using emcee (affine-invariant ensemble sampler)")
print("      This is a valid MCMC method when HMC/NUTS is unavailable.")

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"   - Observations: n = {len(data)}")
print(f"   - x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"   - Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Extract data
x_data = data['x'].values
Y_data = data['Y'].values

# Define log probability functions
def log_prior(theta):
    """Log prior density"""
    beta_0, beta_1, log_sigma = theta
    sigma = np.exp(log_sigma)

    lp = 0.0
    # beta_0 ~ Normal(2.3, 0.3)
    lp += -0.5 * ((beta_0 - 2.3) / 0.3)**2 - np.log(0.3 * np.sqrt(2 * np.pi))
    # beta_1 ~ Normal(0.29, 0.15)
    lp += -0.5 * ((beta_1 - 0.29) / 0.15)**2 - np.log(0.15 * np.sqrt(2 * np.pi))
    # sigma ~ Exponential(10)
    lp += np.log(10) - 10 * sigma
    # Jacobian for log transformation
    lp += log_sigma

    return lp

def log_likelihood(theta, x, Y):
    """Log likelihood"""
    beta_0, beta_1, log_sigma = theta
    sigma = np.exp(log_sigma)

    if sigma <= 0:
        return -np.inf

    # Mean function
    mu = beta_0 + beta_1 * np.log(x)

    # Log-likelihood
    ll = -0.5 * np.sum(((Y - mu) / sigma)**2) - len(Y) * np.log(sigma * np.sqrt(2 * np.pi))

    return ll

def log_probability(theta, x, Y):
    """Log posterior (prior + likelihood)"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, Y)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# Setup MCMC
print("\n2. Setting up MCMC with emcee...")

n_dim = 3
n_walkers = 32  # More walkers for better exploration
n_steps_warmup = 1000
n_steps_sampling = 1000

print(f"   - Parameters: {n_dim} (beta_0, beta_1, log(sigma))")
print(f"   - Walkers: {n_walkers}")
print(f"   - Warmup steps: {n_steps_warmup}")
print(f"   - Sampling steps: {n_steps_sampling}")

# Initial parameters (using prior means, log-transform sigma)
initial = np.array([2.3, 0.29, np.log(0.1)])

# Initialize walkers in a small ball around initial guess
pos = initial + 0.01 * np.random.randn(n_walkers, n_dim)

# Create sampler
sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, log_probability, args=(x_data, Y_data)
)

# Run warmup
print("\n3. Running warmup...")
state = sampler.run_mcmc(pos, n_steps_warmup, progress=True, skip_initial_state_check=True)
sampler.reset()

print("   - Warmup complete")

# Run sampling
print("\n4. Running sampling...")
sampler.run_mcmc(state, n_steps_sampling, progress=True, skip_initial_state_check=True)

print("   - Sampling complete")

# Extract samples and reshape for ArviZ
print("\n5. Processing samples...")
samples = sampler.get_chain(flat=False)  # Shape: (n_steps, n_walkers, n_dim)

# Reshape to (n_chains=4, n_draws_per_chain, n_dim) by grouping walkers
n_chains = 4
walkers_per_chain = n_walkers // n_chains
n_draws_per_chain = n_steps_sampling * walkers_per_chain

samples_reshaped = []
for i in range(n_chains):
    chain_samples = samples[:, i*walkers_per_chain:(i+1)*walkers_per_chain, :].reshape(-1, n_dim)
    samples_reshaped.append(chain_samples)

samples_reshaped = np.array(samples_reshaped)  # Shape: (n_chains, n_draws, n_dim)

# Transform back from log_sigma to sigma
beta_0_samples = samples_reshaped[:, :, 0]
beta_1_samples = samples_reshaped[:, :, 1]
sigma_samples = np.exp(samples_reshaped[:, :, 2])

print(f"   - Reshaped to {n_chains} chains x {n_draws_per_chain} draws")

# Compute log-likelihood and predictions
print("\n6. Computing log-likelihood and predictions...")
n_obs = len(Y_data)

log_lik_list = []
y_pred_list = []

for chain_idx in range(n_chains):
    for draw_idx in range(n_draws_per_chain):
        beta_0 = beta_0_samples[chain_idx, draw_idx]
        beta_1 = beta_1_samples[chain_idx, draw_idx]
        sigma = sigma_samples[chain_idx, draw_idx]

        mu = beta_0 + beta_1 * np.log(x_data)
        y_pred_list.append(mu)

        # Log-likelihood for each observation
        log_lik = -0.5 * ((Y_data - mu) / sigma)**2 - np.log(sigma * np.sqrt(2 * np.pi))
        log_lik_list.append(log_lik)

log_lik_array = np.array(log_lik_list).reshape((n_chains, n_draws_per_chain, n_obs))
y_pred_array = np.array(y_pred_list).reshape((n_chains, n_draws_per_chain, n_obs))

# Generate posterior predictive samples
print("\n7. Generating posterior predictive samples...")
np.random.seed(42)
y_rep_list = []

for chain_idx in range(n_chains):
    for draw_idx in range(n_draws_per_chain):
        beta_0 = beta_0_samples[chain_idx, draw_idx]
        beta_1 = beta_1_samples[chain_idx, draw_idx]
        sigma = sigma_samples[chain_idx, draw_idx]

        mu = beta_0 + beta_1 * np.log(x_data)
        y_rep = np.random.normal(mu, sigma)
        y_rep_list.append(y_rep)

y_rep_array = np.array(y_rep_list).reshape((n_chains, n_draws_per_chain, n_obs))

# Create ArviZ InferenceData
print("\n8. Creating ArviZ InferenceData...")
idata = az.from_dict(
    posterior={
        'beta_0': beta_0_samples,
        'beta_1': beta_1_samples,
        'sigma': sigma_samples,
        'y_pred': y_pred_array
    },
    log_likelihood={
        'Y': log_lik_array
    },
    posterior_predictive={
        'Y': y_rep_array
    },
    observed_data={
        'Y': Y_data
    },
    coords={
        'obs': np.arange(n_obs)
    },
    dims={
        'Y': ['obs'],
        'y_pred': ['obs']
    }
)

# Save InferenceData
idata_path = DIAG_DIR / "posterior_inference.netcdf"
idata.to_netcdf(idata_path)
print(f"   - Saved to: {idata_path}")

# Compute ArviZ summary with diagnostics
print("\n9. Computing ArviZ summary statistics...")
az_summary = az.summary(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma'],
    kind='all'  # Include diagnostics
)
az_summary.to_csv(DIAG_DIR / "arviz_summary.csv")
print(f"   - Saved to: {DIAG_DIR / 'arviz_summary.csv'}")

# Display summary
print("\n" + "=" * 80)
print("PARAMETER ESTIMATES")
print("=" * 80)
print(az_summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']].to_string())

# Check convergence criteria
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

rhat_max = az_summary['r_hat'].max()
ess_bulk_min = az_summary['ess_bulk'].min()
ess_tail_min = az_summary['ess_tail'].min()

# Acceptance rate
acceptance_rate = sampler.acceptance_fraction.mean()

print(f"\nR-hat (max):          {rhat_max:.4f}  [Target: < 1.01]")
print(f"ESS bulk (min):       {ess_bulk_min:.0f}    [Target: > 400]")
print(f"ESS tail (min):       {ess_tail_min:.0f}    [Target: > 400]")
print(f"Acceptance rate:      {acceptance_rate:.3f}  [Target: 0.2-0.5]")
print(f"Divergences:          N/A     [emcee uses different sampler]")

# Determine convergence status
convergence_pass = True
warnings = []
failures = []

if rhat_max > 1.01:
    failures.append(f"R-hat > 1.01 (max = {rhat_max:.4f})")
    convergence_pass = False

if ess_bulk_min < 400:
    failures.append(f"ESS bulk < 400 (min = {ess_bulk_min:.0f})")
    convergence_pass = False

if ess_tail_min < 400:
    failures.append(f"ESS tail < 400 (min = {ess_tail_min:.0f})")
    convergence_pass = False

if acceptance_rate < 0.15:
    warnings.append(f"Low acceptance rate: {acceptance_rate:.3f}")
elif acceptance_rate > 0.6:
    warnings.append(f"High acceptance rate (inefficient exploration): {acceptance_rate:.3f}")

print("\n" + "-" * 80)
if convergence_pass and not warnings:
    print("STATUS: ALL CONVERGENCE CHECKS PASSED")
elif convergence_pass and warnings:
    print("STATUS: CONVERGENCE PASSED WITH WARNINGS")
    for w in warnings:
        print(f"  - WARNING: {w}")
else:
    print("STATUS: CONVERGENCE FAILED")
    for f in failures:
        print(f"  - FAILURE: {f}")
    for w in warnings:
        print(f"  - WARNING: {w}")

print("=" * 80)

# Save convergence report
convergence_data = {
    'rhat_max': float(rhat_max),
    'ess_bulk_min': float(ess_bulk_min),
    'ess_tail_min': float(ess_tail_min),
    'acceptance_rate': float(acceptance_rate),
    'convergence_pass': convergence_pass,
    'warnings': warnings,
    'failures': failures,
    'sampler': 'emcee (affine-invariant ensemble sampler)',
    'note': 'Valid MCMC method, though different from HMC/NUTS'
}

with open(DIAG_DIR / "convergence_metrics.json", 'w') as f:
    json.dump(convergence_data, f, indent=2)

print(f"\n10. Convergence metrics saved to: {DIAG_DIR / 'convergence_metrics.json'}")
print("\n" + "=" * 80)
print("FITTING COMPLETE")
print("=" * 80)
