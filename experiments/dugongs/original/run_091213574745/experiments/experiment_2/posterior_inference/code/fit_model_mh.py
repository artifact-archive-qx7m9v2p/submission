"""
Fit Logarithmic Model with Student-t Likelihood using Metropolis-Hastings MCMC

Model: Y_i ~ StudentT(ν, β₀ + β₁*log(x_i), σ)
Priors:
  β₀ ~ Normal(2.3, 0.5)
  β₁ ~ Normal(0.29, 0.15)
  σ ~ Exponential(10)
  ν ~ Gamma(2, 0.1) truncated at ν ≥ 3

Uses custom Metropolis-Hastings implementation with adaptive proposal.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import arviz as az
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed
np.random.seed(42)

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STUDENT-T LOGARITHMIC MODEL - METROPOLIS-HASTINGS MCMC")
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
    lp += stats.norm.logpdf(beta_0, loc=2.3, scale=0.5)

    # beta_1 ~ Normal(0.29, 0.15)
    lp += stats.norm.logpdf(beta_1, loc=0.29, scale=0.15)

    # sigma ~ Exponential(10)
    lp += stats.expon.logpdf(sigma, scale=1/10)
    lp += log_sigma  # Jacobian

    # ν ~ Gamma(2, 0.1) where rate parameterization
    # In scipy, Gamma uses shape and scale, where scale = 1/rate
    # So Gamma(2, 0.1) with rate means Gamma(alpha=2, scale=1/0.1=10) in scipy
    lp += stats.gamma.logpdf(nu, a=2, scale=10)
    lp += log_nu_minus_3  # Jacobian for nu = 3 + exp(log_nu_minus_3)

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
    ll = np.sum(stats.t.logpdf(Y, df=nu, loc=mu, scale=sigma))

    return ll

def log_posterior(theta, x, Y):
    """Combined log posterior"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, x, Y)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

# Metropolis-Hastings sampler with adaptive proposals
def adaptive_mh_sampler(log_post_fn, initial_theta, n_iterations, x, Y,
                        adapt_interval=100, target_accept=0.234):
    """
    Adaptive Metropolis-Hastings sampler

    Parameters:
    -----------
    log_post_fn : callable
        Log posterior function
    initial_theta : array
        Initial parameter values
    n_iterations : int
        Number of MCMC iterations
    x, Y : arrays
        Data
    adapt_interval : int
        How often to adapt proposal scale
    target_accept : float
        Target acceptance rate (0.234 for MH)
    """
    n_dim = len(initial_theta)
    samples = np.zeros((n_iterations, n_dim))
    log_posts = np.zeros(n_iterations)

    # Proposal covariance (start with scaled identity)
    proposal_scale = 2.38 / np.sqrt(n_dim)  # Optimal scaling
    proposal_cov = proposal_scale**2 * np.eye(n_dim)

    # Initialize
    theta_current = initial_theta.copy()
    log_post_current = log_post_fn(theta_current, x, Y)

    n_accepted = 0
    accepts = []

    print(f"   - Running {n_iterations} iterations with adaptive proposals...")

    for i in tqdm(range(n_iterations)):
        # Propose new theta
        theta_proposed = np.random.multivariate_normal(theta_current, proposal_cov)
        log_post_proposed = log_post_fn(theta_proposed, x, Y)

        # Acceptance ratio
        log_alpha = log_post_proposed - log_post_current

        # Accept/reject
        if np.log(np.random.rand()) < log_alpha:
            theta_current = theta_proposed
            log_post_current = log_post_proposed
            n_accepted += 1
            accepts.append(1)
        else:
            accepts.append(0)

        # Store sample
        samples[i] = theta_current
        log_posts[i] = log_post_current

        # Adapt proposal covariance
        if (i + 1) % adapt_interval == 0 and i > 200:
            accept_rate = np.mean(accepts[-adapt_interval:])

            # Adjust scale based on acceptance rate
            if accept_rate < target_accept - 0.05:
                proposal_scale *= 0.9
            elif accept_rate > target_accept + 0.05:
                proposal_scale *= 1.1

            # Update covariance from recent samples
            if i > 500:
                recent_samples = samples[max(0, i-500):i]
                empirical_cov = np.cov(recent_samples.T)
                proposal_cov = proposal_scale**2 * (empirical_cov + 1e-6 * np.eye(n_dim))

    acceptance_rate = n_accepted / n_iterations

    return samples, log_posts, acceptance_rate

# Initial parameters (from prior knowledge)
initial_theta = np.array([2.3, 0.29, np.log(0.1), np.log(17.0)])  # ν ≈ 20

# Probe run
print("\n2. PROBE RUN (500 iterations)...")
probe_samples, probe_logpost, probe_accept = adaptive_mh_sampler(
    log_posterior, initial_theta, n_iterations=500, x=x_data, Y=Y_data
)

print(f"   - Probe acceptance rate: {probe_accept:.3f}")

# Check probe ν
probe_nu = 3.0 + np.exp(probe_samples[250:, 3])
probe_nu_mean = np.mean(probe_nu)
print(f"   - Probe ν estimate: {probe_nu_mean:.2f}")

# Main sampling (4 chains)
print("\n3. MAIN SAMPLING (4 chains × 2000 iterations)...")

n_chains = 4
n_warmup = 1000
n_samples = 1000
n_total = n_warmup + n_samples

all_samples = []
all_logposts = []
acceptance_rates = []

# Initialize chains with overdispersed starting points
chain_inits = []
for chain_id in range(n_chains):
    # Start from different regions
    init = initial_theta + 0.2 * np.random.randn(len(initial_theta))
    chain_inits.append(init)

# Run chains
for chain_id in range(n_chains):
    print(f"\n   Chain {chain_id + 1}/{n_chains}:")

    samples, logposts, accept_rate = adaptive_mh_sampler(
        log_posterior, chain_inits[chain_id], n_iterations=n_total,
        x=x_data, Y=Y_data, adapt_interval=50
    )

    # Discard warmup
    samples_post_warmup = samples[n_warmup:]
    logposts_post_warmup = logposts[n_warmup:]

    all_samples.append(samples_post_warmup)
    all_logposts.append(logposts_post_warmup)
    acceptance_rates.append(accept_rate)

    print(f"     Acceptance rate: {accept_rate:.3f}")

# Stack chains: shape (n_chains, n_samples, n_dim)
samples_array = np.stack(all_samples, axis=0)

print(f"\n4. Sampling completed!")
print(f"   - Mean acceptance rate: {np.mean(acceptance_rates):.3f}")

# Transform to natural parameters
beta_0_samples = samples_array[:, :, 0]
beta_1_samples = samples_array[:, :, 1]
sigma_samples = np.exp(samples_array[:, :, 2])
nu_samples = 3.0 + np.exp(samples_array[:, :, 3])

# Create ArviZ InferenceData
print("\n5. Creating ArviZ InferenceData...")

# Compute log-likelihood for LOO-CV
print("   - Computing log-likelihood...")
log_lik = np.zeros((n_chains, n_samples, N))

for chain in range(n_chains):
    for draw in range(n_samples):
        theta = samples_array[chain, draw, :]
        beta_0 = theta[0]
        beta_1 = theta[1]
        sigma = np.exp(theta[2])
        nu = 3.0 + np.exp(theta[3])

        mu = beta_0 + beta_1 * np.log(x_data)
        for i in range(N):
            log_lik[chain, draw, i] = stats.t.logpdf(Y_data[i], df=nu, loc=mu[i], scale=sigma)

# Posterior predictive
print("   - Generating posterior predictive samples...")
y_pred = np.zeros((n_chains, n_samples, N))
y_rep = np.zeros((n_chains, n_samples, N))

for chain in range(n_chains):
    for draw in range(n_samples):
        theta = samples_array[chain, draw, :]
        beta_0 = theta[0]
        beta_1 = theta[1]
        sigma = np.exp(theta[2])
        nu = 3.0 + np.exp(theta[3])

        mu = beta_0 + beta_1 * np.log(x_data)
        y_pred[chain, draw, :] = mu
        y_rep[chain, draw, :] = stats.t.rvs(df=nu, loc=mu, scale=sigma)

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
print("\n6. Computing ArviZ summary...")
az_summary = az.summary(
    idata,
    var_names=['beta_0', 'beta_1', 'sigma', 'nu'],
    kind='stats'
)
az_summary.to_csv(DIAG_DIR / "arviz_summary.csv")

# Display summary
print("\n" + "=" * 80)
print("PARAMETER ESTIMATES")
print("=" * 80)
print(az_summary.to_string())

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
print(f"Acceptance rate:    {np.mean(acceptance_rates):.3f}  [Target: ~0.23]")

convergence_pass = (rhat_max < 1.01) and (ess_bulk_min > 400) and (ess_tail_min > 400)
warnings = []
failures = []

if rhat_max > 1.01:
    failures.append(f"R-hat > 1.01 (max = {rhat_max:.4f})")
    convergence_pass = False

if ess_bulk_min < 400:
    warnings.append(f"ESS bulk < 400 (min = {ess_bulk_min:.0f}) - may need longer chains")
    if ess_bulk_min < 100:
        convergence_pass = False
        failures.append(f"ESS bulk < 100 (critical)")

if ess_tail_min < 400:
    warnings.append(f"ESS tail < 400 (min = {ess_tail_min:.0f}) - may need longer chains")
    if ess_tail_min < 100:
        convergence_pass = False
        failures.append(f"ESS tail < 100 (critical)")

print("\n" + "-" * 80)
if convergence_pass and not warnings:
    print("STATUS: ALL CONVERGENCE CHECKS PASSED")
elif convergence_pass:
    print("STATUS: CONVERGENCE ACCEPTABLE (with minor warnings)")
    for w in warnings:
        print(f"  - WARNING: {w}")
else:
    print("STATUS: CONVERGENCE ISSUES DETECTED")
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
        f.write(f"Convergence: {'PASS' if convergence_pass else 'ACCEPTABLE' if not failures else 'FAIL'}\n")

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
    'acceptance_rate': float(np.mean(acceptance_rates)),
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
print(f"  Convergence: {'✓ PASS' if convergence_pass else '○ ACCEPTABLE' if not failures else '✗ FAIL'}")
print(f"  Recommendation: {final_rec}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
