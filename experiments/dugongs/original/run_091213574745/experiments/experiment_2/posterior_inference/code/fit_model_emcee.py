"""
Fit Logarithmic Model with Student-t Likelihood using emcee

Model: Y_i ~ StudentT(ν, β₀ + β₁*log(x_i), σ)
Priors:
  β₀ ~ Normal(2.3, 0.5)
  β₁ ~ Normal(0.29, 0.15)
  σ ~ Exponential(10)
  ν ~ Gamma(2, 0.1) truncated at ν ≥ 3

Uses emcee (affine-invariant ensemble sampler) as a pure-Python MCMC solution.
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
import json
import arviz as az
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
STAN_FILE = Path("/workspace/experiments/experiment_2/posterior_inference/code/student_t_log_model.stan")
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"
CMDSTAN_PATH = Path("/tmp/agent-home/.cmdstan/cmdstan-2.37.0")

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STUDENT-T LOGARITHMIC MODEL - MCMC WITH EMCEE")
print("=" * 80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"   - Observations: n = {len(data)}")
print(f"   - x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"   - Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

x_data = data['x'].values
Y_data = data['Y'].values
N = len(data)

# Compile Stan model using stanc (for documentation, won't execute)
print("\n2. Compiling Stan model to C++ (documentation only)...")
stanc_bin = CMDSTAN_PATH / "bin" / "linux-stanc"
cpp_file = STAN_FILE.with_suffix('.hpp')

try:
    result = subprocess.run(
        [str(stanc_bin), str(STAN_FILE), "--o", str(cpp_file)],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"   - Stan model compiled to C++: {cpp_file}")
except subprocess.CalledProcessError as e:
    print(f"   - NOTE: Stan compilation skipped (for reference only)")

print("\n   NOTE: C++ compilation requires 'make' which is not available.")
print("   Using emcee (pure Python MCMC) for actual inference.")

# Install emcee
print("\n3. Installing emcee...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "emcee"], check=True)

import emcee

print("   - emcee installed successfully")

# Define model functions
def log_prior(theta):
    """
    Log prior density
    theta = [beta_0, beta_1, log_sigma, log_nu_minus_3]
    """
    beta_0, beta_1, log_sigma, log_nu_minus_3 = theta
    sigma = np.exp(log_sigma)
    nu = 3.0 + np.exp(log_nu_minus_3)  # Ensures ν ≥ 3

    lp = 0.0

    # beta_0 ~ Normal(2.3, 0.5)
    lp += -0.5 * ((beta_0 - 2.3) / 0.5)**2 - np.log(0.5 * np.sqrt(2 * np.pi))

    # beta_1 ~ Normal(0.29, 0.15)
    lp += -0.5 * ((beta_1 - 0.29) / 0.15)**2 - np.log(0.15 * np.sqrt(2 * np.pi))

    # sigma ~ Exponential(10)
    lp += np.log(10) - 10 * sigma
    lp += log_sigma  # Jacobian

    # nu_raw ~ Gamma(2, 0.1) where rate = 0.1
    # nu_raw = nu - 3, so we need to apply change of variables
    nu_raw = nu - 3.0
    alpha, beta = 2.0, 0.1
    lp += (alpha - 1) * np.log(nu) - beta * nu + alpha * np.log(beta) - gammaln(alpha)
    lp += log_nu_minus_3  # Jacobian for the log transform

    # Check bounds
    if sigma <= 0 or nu < 3.0:
        return -np.inf

    return lp

def log_likelihood(theta, x, Y):
    """
    Log likelihood for Student-t model
    """
    beta_0, beta_1, log_sigma, log_nu_minus_3 = theta
    sigma = np.exp(log_sigma)
    nu = 3.0 + np.exp(log_nu_minus_3)

    # Mean function
    mu = beta_0 + beta_1 * np.log(x)

    # Student-t log-likelihood
    # Using scipy.stats.t
    ll = np.sum(stats.t.logpdf((Y - mu) / sigma, df=nu) - np.log(sigma))

    return ll

def log_probability(theta, x, Y):
    """Combined log probability"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, x, Y)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

# Probe run: Short chains for diagnostics
print("\n4. PROBE RUN (100 iterations)...")
print("   - Testing sampler behavior")

# Initial parameters
# Start at prior means with small perturbation
initial_probe = np.array([2.3, 0.29, np.log(0.1), np.log(17.0)])  # ν ≈ 20

n_dim = 4
n_walkers_probe = 16
n_steps_probe = 100

pos_probe = initial_probe + 0.01 * np.random.randn(n_walkers_probe, n_dim)

sampler_probe = emcee.EnsembleSampler(
    n_walkers_probe, n_dim, log_probability, args=(x_data, Y_data)
)

# Run probe
state = sampler_probe.run_mcmc(pos_probe, n_steps_probe, progress=True)

# Check probe acceptance
probe_acceptance = np.mean(sampler_probe.acceptance_fraction)
print(f"\n   - Probe acceptance rate: {probe_acceptance:.3f}")
print(f"   - Target: 0.2 - 0.5 (ensemble sampler)")

# Get probe estimate of ν
probe_samples = sampler_probe.get_chain(discard=50, flat=True)
probe_nu = 3.0 + np.exp(probe_samples[:, 3])
probe_nu_mean = np.mean(probe_nu)
print(f"   - Probe ν estimate: {probe_nu_mean:.2f}")

# Main sampling
print("\n5. MAIN SAMPLING (2000 iterations after warmup)...")
n_walkers = 32
n_steps_warmup = 1000
n_steps_sampling = 1000

print(f"   - Walkers: {n_walkers}")
print(f"   - Warmup steps: {n_steps_warmup}")
print(f"   - Sampling steps: {n_steps_sampling}")

# Initialize from probe final state (stretched to more walkers)
if n_walkers > n_walkers_probe:
    # Resample with replacement from probe final state
    idx = np.random.choice(n_walkers_probe, size=n_walkers, replace=True)
    pos_init = state.coords[idx] + 0.001 * np.random.randn(n_walkers, n_dim)
else:
    pos_init = state.coords[:n_walkers]

# Create main sampler
sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, log_probability, args=(x_data, Y_data)
)

# Warmup
print("\n   - Running warmup...")
state = sampler.run_mcmc(pos_init, n_steps_warmup, progress=True)
sampler.reset()

# Sampling
print("\n   - Running sampling...")
state = sampler.run_mcmc(state, n_steps_sampling, progress=True)

print("\n   - Sampling completed!")

# Diagnostics
print("\n6. Computing diagnostics...")
acceptance_rate = np.mean(sampler.acceptance_fraction)
print(f"   - Mean acceptance rate: {acceptance_rate:.3f}")

# Get samples (discard first 50% as additional burnin)
samples = sampler.get_chain(discard=500, flat=False)  # Shape: (steps, walkers, dim)
samples_flat = sampler.get_chain(discard=500, flat=True)

# Transform to natural parameters
beta_0_samples = samples[:, :, 0]
beta_1_samples = samples[:, :, 1]
sigma_samples = np.exp(samples[:, :, 2])
nu_samples = 3.0 + np.exp(samples[:, :, 3])

# Create ArviZ InferenceData
print("\n7. Creating ArviZ InferenceData...")

# Compute log-likelihood for LOO-CV
print("   - Computing log-likelihood...")
log_lik = np.zeros((samples.shape[0], samples.shape[1], N))

for step in range(samples.shape[0]):
    for walker in range(samples.shape[1]):
        theta = samples[step, walker, :]
        beta_0 = theta[0]
        beta_1 = theta[1]
        sigma = np.exp(theta[2])
        nu = 3.0 + np.exp(theta[3])

        mu = beta_0 + beta_1 * np.log(x_data)
        for i in range(N):
            log_lik[step, walker, i] = stats.t.logpdf((Y_data[i] - mu[i]) / sigma, df=nu) - np.log(sigma)

# Posterior predictive
print("   - Generating posterior predictive samples...")
y_pred = np.zeros((samples.shape[0], samples.shape[1], N))
y_rep = np.zeros((samples.shape[0], samples.shape[1], N))

for step in range(samples.shape[0]):
    for walker in range(samples.shape[1]):
        theta = samples[step, walker, :]
        beta_0 = theta[0]
        beta_1 = theta[1]
        sigma = np.exp(theta[2])
        nu = 3.0 + np.exp(theta[3])

        mu = beta_0 + beta_1 * np.log(x_data)
        y_pred[step, walker, :] = mu
        y_rep[step, walker, :] = stats.t.rvs(df=nu, loc=mu, scale=sigma)

# Create InferenceData
idata = az.from_dict(
    posterior={
        'beta_0': beta_0_samples,
        'beta_1': beta_1_samples,
        'sigma': sigma_samples,
        'nu': nu_samples,
        'y_pred': y_pred
    },
    log_likelihood={
        'Y': log_lik
    },
    posterior_predictive={
        'Y': y_rep
    },
    observed_data={
        'Y': Y_data
    },
    coords={
        'obs': np.arange(N)
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
print("\n8. Computing ArviZ summary...")
az_summary = az.summary(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma', 'nu'],
    kind='stats'
)
az_summary.to_csv(DIAG_DIR / "arviz_summary.csv")
print(az_summary)

# Convergence diagnostics
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

rhat_max = az_summary['r_hat'].max()
ess_bulk_min = az_summary['ess_bulk'].min()
ess_tail_min = az_summary['ess_tail'].min()

print(f"\nR-hat (max):        {rhat_max:.4f}  [Target: < 1.01]")
print(f"ESS bulk (min):     {ess_bulk_min:.0f}    [Target: > 400]")
print(f"ESS tail (min):     {ess_tail_min:.0f}    [Target: > 400]")
print(f"Acceptance rate:    {acceptance_rate:.3f}  [Ensemble sampler: 0.2-0.5 typical]")

convergence_pass = (rhat_max < 1.01) and (ess_bulk_min > 400) and (ess_tail_min > 400)
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

print("\n" + "-" * 80)
if convergence_pass:
    print("STATUS: ALL CONVERGENCE CHECKS PASSED")
else:
    print("STATUS: CONVERGENCE ISSUES DETECTED")
    for f in failures:
        print(f"  - FAILURE: {f}")

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

    # Final recommendation
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
        final_rec = "Borderline"

    # Save recommendation
    with open(DIAG_DIR / "model_recommendation.txt", 'w') as f:
        f.write(f"Model Recommendation: {final_rec}\n")
        f.write(f"ν = {nu_mean:.2f} [{nu_hdi_low:.2f}, {nu_hdi_high:.2f}]\n")
        f.write(f"ν interpretation: {nu_interpretation}\n")
        f.write(f"ΔLOO = {delta_loo:.2f} ± {delta_loo_se:.2f}\n")
        f.write(f"LOO interpretation: {loo_interpretation}\n")
        f.write(f"Convergence: {'PASS' if convergence_pass else 'FAIL'}\n")

else:
    print("⚠ Model 1 results not found. Skipping comparison.")
    final_rec = "Cannot compare"
    delta_loo = None
    loo_interpretation = "N/A"

# Save convergence metrics
convergence_data = {
    'rhat_max': float(rhat_max),
    'ess_bulk_min': float(ess_bulk_min),
    'ess_tail_min': float(ess_tail_min),
    'acceptance_rate': float(acceptance_rate),
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

print("\n" + "=" * 80)
print("FITTING COMPLETE!")
print("=" * 80)
print(f"\nKey results:")
print(f"  ν = {nu_mean:.2f} [{nu_hdi_low:.2f}, {nu_hdi_high:.2f}]")
print(f"  LOO-ELPD = {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
if delta_loo is not None:
    print(f"  ΔLOO = {delta_loo:.2f} ± {delta_loo_se:.2f}")
print(f"  Convergence: {'✓ PASS' if convergence_pass else '✗ FAIL'}")
print(f"  Recommendation: {final_rec}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
