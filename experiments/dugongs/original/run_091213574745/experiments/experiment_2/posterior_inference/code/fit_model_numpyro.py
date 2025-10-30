"""
Fit Logarithmic Model with Student-t Likelihood using MCMC (NumPyro)

Model: Y_i ~ StudentT(ν, β₀ + β₁*log(x_i), σ)
Priors:
  β₀ ~ Normal(2.3, 0.5)
  β₁ ~ Normal(0.29, 0.15)
  σ ~ Exponential(10)
  ν ~ Gamma(2, 0.1) truncated at ν ≥ 3
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
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STUDENT-T LOGARITHMIC MODEL - POSTERIOR INFERENCE (NumPyro)")
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
def student_t_logarithmic_model(x, Y=None):
    """
    Logarithmic model with Student-t likelihood
    Y_i ~ StudentT(ν, β₀ + β₁*log(x_i), σ)

    ν is truncated at 3 for numerical stability
    """
    # Priors
    beta_0 = numpyro.sample('beta_0', dist.Normal(2.3, 0.5))
    beta_1 = numpyro.sample('beta_1', dist.Normal(0.29, 0.15))
    sigma = numpyro.sample('sigma', dist.Exponential(10))

    # Truncated degrees of freedom
    # Gamma(2, 0.1) with rate parameterization means mean = alpha/beta = 2/0.1 = 20
    nu_raw = numpyro.sample('nu_raw', dist.Gamma(2, 0.1))
    # Truncate at 3
    nu = numpyro.deterministic('nu', jnp.maximum(nu_raw, 3.0))

    # Mean function
    mu = beta_0 + beta_1 * jnp.log(x)

    # Likelihood
    with numpyro.plate('obs', len(x)):
        numpyro.sample('Y', dist.StudentT(nu, mu, sigma), obs=Y)

    # Store mu for posterior predictive
    numpyro.deterministic('y_pred', mu)

print("\n2. Building NumPyro model...")
print("   - Model defined successfully")
print("   - ν truncated at 3 for numerical stability")

# Setup MCMC - Initial probe
print("\n3. Running INITIAL PROBE (200 iterations)...")
print("   - Chains: 4")
print("   - Draws per chain: 100")
print("   - Warmup: 100")

nuts_kernel_probe = NUTS(
    student_t_logarithmic_model,
    target_accept_prob=0.95,
    max_tree_depth=10
)

mcmc_probe = MCMC(
    nuts_kernel_probe,
    num_warmup=100,
    num_samples=100,
    num_chains=4,
    chain_method='parallel'
)

rng_key = jax.random.PRNGKey(42)
mcmc_probe.run(rng_key, x, Y)

# Probe diagnostics
probe_samples = mcmc_probe.get_samples()
nu_probe = probe_samples['nu'].mean()
probe_div = mcmc_probe.get_extra_fields()['diverging'].sum()

print(f"\n   - Probe ν estimate: {nu_probe:.2f}")
print(f"   - Probe divergences: {int(probe_div)}")

# Decide on main sampling
if probe_div > 20:
    print("   ⚠ WARNING: Many divergences in probe. Increasing target_accept to 0.99")
    target_accept_main = 0.99
else:
    print("   ✓ Probe looks reasonable. Proceeding with target_accept=0.95")
    target_accept_main = 0.95

# Main MCMC sampling
print("\n4. Running MAIN SAMPLING (2000 iterations)...")
print("   - Chains: 4")
print("   - Draws per chain: 1000")
print("   - Warmup: 1000")
print(f"   - Target accept: {target_accept_main}")
print("   - Starting sampling...")

nuts_kernel = NUTS(
    student_t_logarithmic_model,
    target_accept_prob=target_accept_main,
    max_tree_depth=10
)

mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    chain_method='parallel'
)

rng_key, rng_subkey = jax.random.split(rng_key)
mcmc.run(rng_subkey, x, Y)

print("\n   - Sampling completed!")

# Extract samples
samples = mcmc.get_samples()
posterior_samples = mcmc.get_samples(group_by_chain=True)
n_chains, n_draws = posterior_samples['beta_0'].shape

# Check for divergences
divergences = mcmc.get_extra_fields()['diverging'].sum()
total_iterations = n_chains * n_draws
div_pct = 100 * divergences / total_iterations

print(f"\n5. Quick Diagnostics:")
print(f"   - Divergent transitions: {int(divergences)} ({div_pct:.2f}%)")

# Compute log-likelihood for LOO-CV
print("\n6. Computing log-likelihood...")
log_lik = []
for i in range(N):
    # For each posterior sample, compute log-likelihood of observation i
    mu_i = samples['beta_0'] + samples['beta_1'] * jnp.log(x[i])
    ll_i = dist.StudentT(samples['nu'], mu_i, samples['sigma']).log_prob(Y[i])
    log_lik.append(ll_i)

log_lik = jnp.stack(log_lik, axis=-1)  # Shape: (num_samples, N)

# Generate posterior predictive samples (replicated data)
print("\n7. Generating posterior predictive samples...")
rng_key, rng_subkey = jax.random.split(rng_key)

# Generate y_rep for each chain and draw
y_rep_list = []
for chain in range(n_chains):
    for draw in range(n_draws):
        mu_draw = (posterior_samples['beta_0'][chain, draw] +
                   posterior_samples['beta_1'][chain, draw] * jnp.log(x))
        sigma_draw = posterior_samples['sigma'][chain, draw]
        nu_draw = posterior_samples['nu'][chain, draw]

        rng_key, rng_subkey = jax.random.split(rng_key)
        y_rep_draw = dist.StudentT(nu_draw, mu_draw, sigma_draw).sample(rng_subkey)
        y_rep_list.append(y_rep_draw)

y_rep = jnp.stack(y_rep_list, axis=0)  # Shape: (total_samples, N)

# Create ArviZ InferenceData
print("\n8. Converting to ArviZ InferenceData...")

# Reshape samples for ArviZ (chain, draw, ...)
log_lik_reshaped = log_lik.reshape((n_chains, n_draws, N))
y_pred_reshaped = samples['y_pred'].reshape((n_chains, n_draws, N))
y_rep_reshaped = y_rep.reshape((n_chains, n_draws, N))

idata = az.from_dict(
    posterior={
        'beta_0': np.array(posterior_samples['beta_0']),
        'beta_1': np.array(posterior_samples['beta_1']),
        'sigma': np.array(posterior_samples['sigma']),
        'nu': np.array(posterior_samples['nu']),
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
print("\n9. Computing ArviZ summary statistics...")
az_summary = az.summary(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma', 'nu'],
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

# KEY PARAMETER ANALYSIS: ν
print("\n" + "=" * 80)
print("KEY PARAMETER: ν (Degrees of Freedom)")
print("=" * 80)

nu_mean = az_summary.loc['nu', 'mean']
nu_median = az_summary.loc['nu', '50%']
nu_hdi_low = az_summary.loc['nu', 'hdi_3%']
nu_hdi_high = az_summary.loc['nu', 'hdi_97%']

print(f"\nMean:      {nu_mean:.2f}")
print(f"Median:    {nu_median:.2f}")
print(f"94% HDI:   [{nu_hdi_low:.2f}, {nu_hdi_high:.2f}]")

if nu_mean > 30:
    print("\n⚠ INTERPRETATION: ν > 30 suggests Normal likelihood is adequate")
    print("   Recommendation: Prefer Model 1 (simpler)")
    nu_interpretation = "Normal adequate"
elif nu_mean < 20:
    print("\n✓ INTERPRETATION: ν < 20 suggests heavy tails are justified")
    print("   Recommendation: Student-t robustness may be valuable")
    nu_interpretation = "Heavy tails justified"
else:
    print("\n○ INTERPRETATION: ν ∈ [20, 30] is borderline")
    print("   Recommendation: Check LOO comparison to decide")
    nu_interpretation = "Borderline"

# Compute LOO-CV
print("\n" + "=" * 80)
print("LOO-CV COMPUTATION")
print("=" * 80)

loo_result = az.loo(idata, pointwise=True)
print(loo_result)

# Save LOO result
loo_dict = {
    'elpd_loo': float(loo_result.elpd_loo),
    'se': float(loo_result.se),
    'p_loo': float(loo_result.p_loo),
    'n_samples': int(loo_result.n_samples),
    'n_data_points': int(loo_result.n_data_points),
    'warning': bool(loo_result.warning)
}

with open(DIAG_DIR / "loo_result.json", 'w') as f:
    json.dump(loo_dict, f, indent=2)

# Check Pareto k values
pareto_k = loo_result.pareto_k
print(f"\nPareto k diagnostics:")
print(f"  Max k: {pareto_k.max():.3f}")
print(f"  k > 0.5: {(pareto_k > 0.5).sum()}/{len(pareto_k)}")
print(f"  k > 0.7: {(pareto_k > 0.7).sum()}/{len(pareto_k)}")

# Compare to Model 1
print("\n" + "=" * 80)
print("COMPARISON TO MODEL 1")
print("=" * 80)

model1_idata_path = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
if model1_idata_path.exists():
    print("Loading Model 1 InferenceData...")
    idata_model1 = az.from_netcdf(model1_idata_path)

    # Compute LOO for Model 1
    loo_model1 = az.loo(idata_model1)

    # Compare
    loo_compare = az.compare({'Model_1_Normal': idata_model1, 'Model_2_StudentT': idata})
    print("\nLOO Comparison:")
    print(loo_compare)

    # Save comparison
    loo_compare.to_csv(DIAG_DIR / "loo_comparison.csv")

    # Interpret ΔLOO
    delta_loo = loo_result.elpd_loo - loo_model1.elpd_loo
    delta_loo_se = np.sqrt(loo_result.se**2 + loo_model1.se**2)

    print(f"\nΔLOO = {delta_loo:.2f} ± {delta_loo_se:.2f}")

    if delta_loo > 2:
        print("✓ Model 2 (Student-t) substantially better")
        loo_interpretation = "Model 2 better"
    elif delta_loo < -2:
        print("✗ Model 2 worse (overfitting?)")
        loo_interpretation = "Model 2 worse"
    else:
        print("○ Models equivalent (|ΔLOO| < 2)")
        print("  Recommendation: Prefer simpler Model 1")
        loo_interpretation = "Models equivalent"

    # Combined decision
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    if nu_mean < 20 and delta_loo > 2:
        print("✓ ACCEPT Model 2: Heavy tails justified AND LOO improvement")
        final_rec = "ACCEPT Model 2"
    elif nu_mean > 30 or abs(delta_loo) < 2:
        print("✓ PREFER Model 1: Simpler model adequate")
        final_rec = "Prefer Model 1"
    else:
        print("○ BORDERLINE: Check detailed diagnostics")
        final_rec = "Borderline - check diagnostics"

    # Save recommendation
    with open(DIAG_DIR / "model_recommendation.txt", 'w') as f:
        f.write(f"Model Recommendation: {final_rec}\n")
        f.write(f"ν = {nu_mean:.2f} [{nu_hdi_low:.2f}, {nu_hdi_high:.2f}]\n")
        f.write(f"ν interpretation: {nu_interpretation}\n")
        f.write(f"ΔLOO = {delta_loo:.2f} ± {delta_loo_se:.2f}\n")
        f.write(f"LOO interpretation: {loo_interpretation}\n")
        f.write(f"Convergence: {'PASS' if convergence_pass else 'FAIL'}\n")

    print(f"\nRecommendation saved to: {DIAG_DIR / 'model_recommendation.txt'}")

else:
    print("⚠ Model 1 results not found. Skipping comparison.")
    final_rec = "Cannot compare - Model 1 missing"

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
    'failures': failures,
    'nu_mean': float(nu_mean),
    'nu_median': float(nu_median),
    'nu_hdi_low': float(nu_hdi_low),
    'nu_hdi_high': float(nu_hdi_high),
    'nu_interpretation': nu_interpretation
}

with open(DIAG_DIR / "convergence_metrics.json", 'w') as f:
    json.dump(convergence_data, f, indent=2)

print(f"\n10. Convergence metrics saved to: {DIAG_DIR / 'convergence_metrics.json'}")
print("\n" + "=" * 80)
print("FITTING COMPLETE!")
print("=" * 80)
print(f"\nKey results:")
print(f"  ν = {nu_mean:.2f} [{nu_hdi_low:.2f}, {nu_hdi_high:.2f}]")
print(f"  LOO-ELPD = {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
print(f"  Convergence: {'✓ PASS' if convergence_pass else '✗ FAIL'}")
print(f"  Recommendation: {final_rec}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
