"""
Fit Logarithmic Model with Normal Likelihood using MCMC (NumPyro)

Model: Y_i ~ Normal(β₀ + β₁*log(x_i), σ)
Priors:
  β₀ ~ Normal(2.3, 0.3)
  β₁ ~ Normal(0.29, 0.15)
  σ ~ Exponential(10)
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)
numpyro.set_platform('cpu')

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LOGARITHMIC MODEL - POSTERIOR INFERENCE (NumPyro)")
print("=" * 80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"   - Observations: n = {len(data)}")
print(f"   - x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"   - Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Extract data
x = jnp.array(data['x'].values)
Y = jnp.array(data['Y'].values)
N = len(data)

# Define NumPyro model
def logarithmic_model(x, Y=None):
    """
    Logarithmic model with normal likelihood
    Y_i ~ Normal(β₀ + β₁*log(x_i), σ)
    """
    # Priors
    beta_0 = numpyro.sample('beta_0', dist.Normal(2.3, 0.3))
    beta_1 = numpyro.sample('beta_1', dist.Normal(0.29, 0.15))
    sigma = numpyro.sample('sigma', dist.Exponential(10))

    # Mean function
    mu = beta_0 + beta_1 * jnp.log(x)

    # Likelihood
    with numpyro.plate('obs', len(x)):
        numpyro.sample('Y', dist.Normal(mu, sigma), obs=Y)

    # Store mu for posterior predictive
    numpyro.deterministic('y_pred', mu)

print("\n2. Building NumPyro model...")
print("   - Model defined successfully")

# Setup MCMC
print("\n3. Running MCMC sampling...")
print("   - Chains: 4")
print("   - Draws per chain: 1000")
print("   - Warmup: 1000")
print("   - Target accept: 0.95")
print("   - Starting sampling...")

# Create NUTS sampler
nuts_kernel = NUTS(
    logarithmic_model,
    target_accept_prob=0.95,
    max_tree_depth=10
)

# Run MCMC
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    chain_method='parallel'
)

rng_key = jax.random.PRNGKey(42)
mcmc.run(rng_key, x, Y)

print("\n   - Sampling completed!")

# Extract samples
samples = mcmc.get_samples()

# Check for divergences
divergences = mcmc.get_extra_fields()['diverging'].sum()
total_iterations = 4 * 1000  # 4 chains * 1000 draws
div_pct = 100 * divergences / total_iterations

print(f"\n4. Quick Diagnostics:")
print(f"   - Divergent transitions: {int(divergences)} ({div_pct:.2f}%)")

# Compute log-likelihood for LOO-CV
print("\n5. Computing log-likelihood...")
log_lik = []
for i in range(N):
    # For each posterior sample, compute log-likelihood of observation i
    mu_i = samples['beta_0'] + samples['beta_1'] * jnp.log(x[i])
    ll_i = dist.Normal(mu_i, samples['sigma']).log_prob(Y[i])
    log_lik.append(ll_i)

log_lik = jnp.stack(log_lik, axis=-1)  # Shape: (num_samples, N)

# Generate posterior predictive samples (replicated data)
print("\n6. Generating posterior predictive samples...")
rng_key, rng_subkey = jax.random.split(rng_key)

# Get posterior samples reshaped for all chains
posterior_samples = mcmc.get_samples(group_by_chain=True)
n_chains, n_draws = posterior_samples['beta_0'].shape

# Generate y_rep for each chain and draw
y_rep_list = []
for chain in range(n_chains):
    for draw in range(n_draws):
        mu_draw = (posterior_samples['beta_0'][chain, draw] +
                   posterior_samples['beta_1'][chain, draw] * jnp.log(x))
        sigma_draw = posterior_samples['sigma'][chain, draw]

        rng_key, rng_subkey = jax.random.split(rng_key)
        y_rep_draw = dist.Normal(mu_draw, sigma_draw).sample(rng_subkey)
        y_rep_list.append(y_rep_draw)

y_rep = jnp.stack(y_rep_list, axis=0)  # Shape: (total_samples, N)

# Create ArviZ InferenceData
print("\n7. Converting to ArviZ InferenceData...")

# Reshape samples for ArviZ (chain, draw, ...)
log_lik_reshaped = log_lik.reshape((n_chains, n_draws, N))
y_pred_reshaped = samples['y_pred'].reshape((n_chains, n_draws, N))
y_rep_reshaped = y_rep.reshape((n_chains, n_draws, N))

idata = az.from_dict(
    posterior={
        'beta_0': np.array(posterior_samples['beta_0']),
        'beta_1': np.array(posterior_samples['beta_1']),
        'sigma': np.array(posterior_samples['sigma']),
        'y_pred': np.array(y_pred_reshaped)
    },
    log_likelihood={
        'Y': np.array(log_lik_reshaped)
    },
    posterior_predictive={
        'Y': np.array(y_rep_reshaped)
    },
    observed_data={
        'Y': np.array(Y)
    },
    coords={
        'obs': np.arange(N)
    },
    dims={
        'Y': ['obs'],
        'y_pred': ['obs']
    }
)

# Add sampling stats
diverging_reshaped = mcmc.get_extra_fields()['diverging'].reshape((n_chains, n_draws))
idata.add_groups({
    'sample_stats': {
        'diverging': np.array(diverging_reshaped)
    }
})

# Save InferenceData
idata_path = DIAG_DIR / "posterior_inference.netcdf"
idata.to_netcdf(idata_path)
print(f"   - Saved to: {idata_path}")

# Compute ArviZ summary
print("\n8. Computing ArviZ summary statistics...")
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

print(f"\nR-hat (max):        {rhat_max:.4f}  [Target: < 1.01]")
print(f"ESS bulk (min):     {ess_bulk_min:.0f}    [Target: > 400]")
print(f"ESS tail (min):     {ess_tail_min:.0f}    [Target: > 400]")
print(f"Divergences:        {int(divergences)}       [Target: 0, Acceptable: < {0.01 * total_iterations:.0f}]")

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

if div_pct > 5:
    failures.append(f"Divergences > 5% ({div_pct:.2f}%)")
    convergence_pass = False
elif div_pct > 1:
    warnings.append(f"Divergences 1-5% ({div_pct:.2f}%)")

# Check E-BFMI
try:
    ebfmi = az.bfmi(idata)
    ebfmi_min = ebfmi.min()
    print(f"E-BFMI (min):       {ebfmi_min:.3f}    [Target: > 0.3]")

    if ebfmi_min < 0.3:
        warnings.append(f"E-BFMI < 0.3 (min = {ebfmi_min:.3f})")
except:
    ebfmi_min = None
    print(f"E-BFMI:             N/A (not available)")

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
    'divergences': int(divergences),
    'divergence_pct': float(div_pct),
    'ebfmi_min': float(ebfmi_min) if ebfmi_min is not None else None,
    'convergence_pass': convergence_pass,
    'warnings': warnings,
    'failures': failures
}

with open(DIAG_DIR / "convergence_metrics.json", 'w') as f:
    json.dump(convergence_data, f, indent=2)

print(f"\n9. Convergence metrics saved to: {DIAG_DIR / 'convergence_metrics.json'}")
print("\nFitting complete!")
