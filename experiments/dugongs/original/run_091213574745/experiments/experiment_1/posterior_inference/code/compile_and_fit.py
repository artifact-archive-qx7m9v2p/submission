"""
Compile Stan model manually and fit using CmdStanPy

This script works around the missing 'make' dependency by using stanc directly.
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
import json
import arviz as az

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
STAN_FILE = Path("/workspace/experiments/experiment_1/posterior_inference/code/logarithmic_model.stan")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"
CMDSTAN_PATH = Path("/tmp/agent-home/.cmdstan/cmdstan-2.37.0")

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LOGARITHMIC MODEL - MANUAL COMPILATION AND SAMPLING")
print("=" * 80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"   - Observations: n = {len(data)}")
print(f"   - x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"   - Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Prepare data for Stan
stan_data = {
    'N': len(data),
    'x': data['x'].tolist(),
    'Y': data['Y'].tolist()
}

# Save data as JSON for Stan
data_file = OUTPUT_DIR / "code" / "stan_data.json"
with open(data_file, 'w') as f:
    json.dump(stan_data, f)

print(f"\n2. Data saved to: {data_file}")

# Compile Stan model using stanc
print("\n3. Compiling Stan model using stanc...")
stanc_bin = CMDSTAN_PATH / "bin" / "linux-stanc"
cpp_file = STAN_FILE.with_suffix('.hpp')

try:
    result = subprocess.run(
        [str(stanc_bin), str(STAN_FILE), "--o", str(cpp_file)],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"   - Stan model compiled to: {cpp_file}")
except subprocess.CalledProcessError as e:
    print(f"   - ERROR: Compilation failed")
    print(f"   - stdout: {e.stdout}")
    print(f"   - stderr: {e.stderr}")
    sys.exit(1)

# Unfortunately, we still need make to compile the C++ to an executable
# Let's try using CmdStanPy with a pre-existing model
# Since we can't compile, let me try the bridgestan approach or just use scipy directly

print("\n   NOTE: C++ compilation requires 'make' which is not available.")
print("   Switching to alternative: Using scipy for optimization to get MAP,")
print("   then using a simpler MCMC implementation.")

# Let's use emcee (affine-invariant MCMC) instead which is pure Python
print("\n4. Installing emcee for MCMC...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "emcee"], check=True)

import emcee

print("   - emcee installed successfully")

# Define log probability function
def log_prior(theta):
    beta_0, beta_1, log_sigma = theta
    sigma = np.exp(log_sigma)

    # Priors
    lp = 0.0
    # beta_0 ~ Normal(2.3, 0.3)
    lp += -0.5 * ((beta_0 - 2.3) / 0.3)**2 - np.log(0.3 * np.sqrt(2 * np.pi))
    # beta_1 ~ Normal(0.29, 0.15)
    lp += -0.5 * ((beta_1 - 0.29) / 0.15)**2 - np.log(0.15 * np.sqrt(2 * np.pi))
    # sigma ~ Exponential(10)
    lp += np.log(10) - 10 * sigma

    # Jacobian for log_sigma
    lp += log_sigma

    return lp

def log_likelihood(theta, x, Y):
    beta_0, beta_1, log_sigma = theta
    sigma = np.exp(log_sigma)

    # Mean function
    mu = beta_0 + beta_1 * np.log(x)

    # Log-likelihood
    ll = -0.5 * np.sum(((Y - mu) / sigma)**2) - len(Y) * np.log(sigma * np.sqrt(2 * np.pi))

    return ll

def log_probability(theta, x, Y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, Y)

# Setup MCMC
print("\n5. Setting up MCMC with emcee...")
x_data = data['x'].values
Y_data = data['Y'].values

# Initial parameters (using prior means)
initial = np.array([2.3, 0.29, np.log(0.1)])

# Setup walkers
n_dim = 3
n_walkers = 32
n_steps_warmup = 1000
n_steps_sampling = 1000

# Initialize walkers around initial guess
pos = initial + 0.01 * np.random.randn(n_walkers, n_dim)

# Create sampler
sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, log_probability, args=(x_data, Y_data)
)

print(f"   - Walkers: {n_walkers}")
print(f"   - Warmup steps: {n_steps_warmup}")
print(f"   - Sampling steps: {n_steps_sampling}")

# Run warmup
print("\n6. Running warmup...")
state = sampler.run_mcmc(pos, n_steps_warmup, progress=True)
sampler.reset()

# Run sampling
print("\n7. Running sampling...")
sampler.run_mcmc(state, n_steps_sampling, progress=True)

# Extract samples
print("\n8. Extracting samples...")
samples = sampler.get_chain(flat=False)  # Shape: (n_steps, n_walkers, n_dim)
# Reshape to (n_chains=4, n_draws, n_dim) by grouping walkers
n_chains = 4
walkers_per_chain = n_walkers // n_chains

samples_reshaped = []
for i in range(n_chains):
    chain_samples = samples[:, i*walkers_per_chain:(i+1)*walkers_per_chain, :].reshape(-1, n_dim)
    samples_reshaped.append(chain_samples)

samples_reshaped = np.array(samples_reshaped)  # Shape: (n_chains, n_draws, n_dim)

# Transform log_sigma back to sigma
beta_0_samples = samples_reshaped[:, :, 0]
beta_1_samples = samples_reshaped[:, :, 1]
sigma_samples = np.exp(samples_reshaped[:, :, 2])

# Compute log-likelihood for each sample and observation
print("\n9. Computing log-likelihood...")
n_obs = len(Y_data)
n_total_draws = samples_reshaped.shape[0] * samples_reshaped.shape[1]

log_lik_all = []
y_pred_all = []

for chain_idx in range(samples_reshaped.shape[0]):
    for draw_idx in range(samples_reshaped.shape[1]):
        beta_0 = beta_0_samples[chain_idx, draw_idx]
        beta_1 = beta_1_samples[chain_idx, draw_idx]
        sigma = sigma_samples[chain_idx, draw_idx]

        mu = beta_0 + beta_1 * np.log(x_data)
        y_pred_all.append(mu)

        # Log-likelihood for each observation
        log_lik = -0.5 * ((Y_data - mu) / sigma)**2 - np.log(sigma * np.sqrt(2 * np.pi))
        log_lik_all.append(log_lik)

log_lik_array = np.array(log_lik_all).reshape((n_chains, -1, n_obs))
y_pred_array = np.array(y_pred_all).reshape((n_chains, -1, n_obs))

# Generate posterior predictive samples
print("\n10. Generating posterior predictive samples...")
y_rep_all = []
for chain_idx in range(samples_reshaped.shape[0]):
    for draw_idx in range(samples_reshaped.shape[1]):
        beta_0 = beta_0_samples[chain_idx, draw_idx]
        beta_1 = beta_1_samples[chain_idx, draw_idx]
        sigma = sigma_samples[chain_idx, draw_idx]

        mu = beta_0 + beta_1 * np.log(x_data)
        y_rep = np.random.normal(mu, sigma)
        y_rep_all.append(y_rep)

y_rep_array = np.array(y_rep_all).reshape((n_chains, -1, n_obs))

# Create ArviZ InferenceData
print("\n11. Creating ArviZ InferenceData...")
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

# Compute ArviZ summary
print("\n12. Computing ArviZ summary statistics...")
az_summary = az.summary(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma'],
    kind='stats'
)
az_summary.to_csv(DIAG_DIR / "arviz_summary.csv")
print(f"   - Saved to: {DIAG_DIR / 'arviz_summary.csv'}")

# Display summary
print("\n" + "=" * 80)
print("PARAMETER ESTIMATES")
print("=" * 80)
print(az_summary.to_string())

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
print(f"Acceptance rate:      {acceptance_rate:.3f}  [Target: 0.6-0.9]")
print(f"Divergences:          0        [emcee doesn't track divergences]")

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

if acceptance_rate < 0.2 or acceptance_rate > 0.95:
    warnings.append(f"Acceptance rate outside optimal range: {acceptance_rate:.3f}")

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
    'note': 'Fitted using emcee (affine-invariant ensemble sampler) due to Stan compilation issues'
}

with open(DIAG_DIR / "convergence_metrics.json", 'w') as f:
    json.dump(convergence_data, f, indent=2)

print(f"\n13. Convergence metrics saved to: {DIAG_DIR / 'convergence_metrics.json'}")
print("\nFitting complete!")
print("\nNOTE: This model was fitted using emcee (a pure-Python MCMC implementation)")
print("      instead of Stan/PyMC due to compilation environment limitations.")
print("      Emcee uses the affine-invariant ensemble sampler, which is a valid")
print("      MCMC method, though different from HMC/NUTS used by Stan/PyMC.")
