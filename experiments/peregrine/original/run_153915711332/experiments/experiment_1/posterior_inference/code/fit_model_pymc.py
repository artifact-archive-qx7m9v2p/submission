"""
Fit Negative Binomial State-Space Model to Real Data
Using PyMC with NUTS sampling (fallback from Stan)

Model: C_t ~ NegativeBinomial(μ_t, α)
       log(μ_t) = η_t
       η_t ~ Normal(η_{t-1} + δ, σ_η)

Priors: δ ~ Normal(0.05, 0.02)
        σ_η ~ Exponential(20)
        α ~ Exponential(0.05)  # Note: α = φ in Stan parameterization
        η_1 ~ Normal(log(50), 1)
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('default')
sns.set_palette("colorblind")

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
CODE_DIR = BASE_DIR / "code"
DIAG_DIR = BASE_DIR / "diagnostics"
PLOT_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

print("="*80)
print("FITTING NEGATIVE BINOMIAL STATE-SPACE MODEL TO REAL DATA")
print("Using PyMC (Stan compiler not available)")
print("="*80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
N = len(data)
C_obs = data['C'].values

print(f"   - Observations: {N}")
print(f"   - Count range: {C_obs.min()} to {C_obs.max()}")
print(f"   - Mean count: {C_obs.mean():.1f}")
print(f"   - Variance: {C_obs.var():.1f}")
print(f"   - Variance/Mean ratio: {C_obs.var() / C_obs.mean():.1f}")

# Build PyMC model
print("\n2. Building PyMC model...")
print("   - Using non-centered parameterization for latent states")

with pm.Model() as model:
    # Priors
    delta = pm.Normal('delta', mu=0.05, sigma=0.02)
    sigma_eta = pm.Exponential('sigma_eta', lam=20)
    alpha = pm.Exponential('alpha', lam=0.05)  # Dispersion parameter (φ in Stan)
    eta_1 = pm.Normal('eta_1', mu=np.log(50), sigma=1)

    # Non-centered parameterization for latent states
    eta_raw = pm.Normal('eta_raw', mu=0, sigma=1, shape=N-1)

    # Construct latent states
    eta = pm.Deterministic('eta', pm.math.concatenate([
        [eta_1],
        eta_1 + pm.math.cumsum(delta + sigma_eta * eta_raw)
    ]))

    # Expected counts
    mu = pm.Deterministic('mu', pm.math.exp(eta))

    # Observation likelihood
    # PyMC uses NegativeBinomial(mu, alpha) where Var = mu + mu^2/alpha
    # This matches Stan's neg_binomial_2 parameterization
    C = pm.NegativeBinomial('C', mu=mu, alpha=alpha, observed=C_obs)

    # Posterior predictive
    C_pred = pm.NegativeBinomial('C_pred', mu=mu, alpha=alpha, shape=N)

print("   ✓ Model built successfully")
print(f"   - Free parameters: {sum(v.size for v in model.free_RVs)}")

# Initial probe: Short chains to assess behavior
print("\n3. Initial probe sampling (100 iterations)...")
print("   Purpose: Quick assessment of model behavior")

try:
    with model:
        probe_trace = pm.sample(
            draws=100,
            tune=100,
            chains=4,
            cores=4,
            target_accept=0.95,
            max_treedepth=12,
            return_inferencedata=True,
            random_seed=42,
            progressbar=True
        )

    probe_summary = az.summary(probe_trace, var_names=['delta', 'sigma_eta', 'alpha'])
    probe_divergences = probe_trace.sample_stats.diverging.sum().item()

    print(f"\n   Probe Results:")
    print(f"   - Divergences: {probe_divergences}")
    print(f"   - Max R-hat: {probe_summary['r_hat'].max():.4f}")
    print(f"   - Min ESS_bulk: {probe_summary['ess_bulk'].min():.0f}")

    if probe_divergences > 10:
        print("   ⚠ Warning: Divergences detected in probe. Will increase target_accept for main sampling.")
        target_accept_main = 0.99
    else:
        print("   ✓ Probe successful, proceeding with main sampling")
        target_accept_main = 0.95

except Exception as e:
    print(f"   ⚠ Probe failed: {e}")
    print("   Proceeding with main sampling at target_accept=0.99")
    target_accept_main = 0.99

# Main sampling
print(f"\n4. Main HMC/NUTS sampling (target_accept={target_accept_main})...")
print("   - Chains: 4")
print("   - Warmup (tune): 1000")
print("   - Sampling: 2000 per chain")
print("   - Total post-warmup samples: 8000")

with model:
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=target_accept_main,
        max_treedepth=12,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True},
        random_seed=42,
        progressbar=True
    )

    # Sample posterior predictive
    print("\n5. Sampling posterior predictive...")
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

print("\n   ✓ Sampling completed")

# Extract diagnostics
print("\n6. Extracting diagnostics...")
divergences = idata.sample_stats.diverging.sum().item()
total_samples = len(idata.posterior.chain) * len(idata.posterior.draw)

# Get summary statistics
summary = az.summary(idata, var_names=['delta', 'sigma_eta', 'alpha', 'eta_1'])

print(f"\n   Convergence Diagnostics:")
print(f"   - Total divergences: {divergences} ({100*divergences/total_samples:.2f}%)")
print(f"   - Max R-hat: {summary['r_hat'].max():.4f}")
print(f"   - Min ESS_bulk: {summary['ess_bulk'].min():.0f}")
print(f"   - Min ESS_tail: {summary['ess_tail'].min():.0f}")

# Add observed data for comparison
idata.add_groups({'observed_data': {'C': C_obs}})

# Add coordinates
idata = idata.assign_coords({'time': data['year'].values})

# Save InferenceData
print(f"\n7. Saving InferenceData with log_likelihood...")
print(f"   Saving to {DIAG_DIR / 'posterior_inference.netcdf'}...")
idata.to_netcdf(DIAG_DIR / "posterior_inference.netcdf")
print("   ✓ InferenceData saved with log_likelihood group")

# Check that log_likelihood is present
if 'log_likelihood' in idata.groups():
    print(f"   ✓ log_likelihood group confirmed: shape {idata.log_likelihood['C'].shape}")
else:
    print("   ⚠ Warning: log_likelihood group not found in InferenceData")

# Save summary statistics
print("\n8. Saving summary statistics...")
summary_path = DIAG_DIR / "convergence_summary.csv"
summary.to_csv(summary_path)
print(f"   ✓ Saved to {summary_path}")

print("\n   Key Parameter Summary:")
print(summary.to_string())

# Energy diagnostics
print("\n9. Energy diagnostics...")
bfmi = az.bfmi(idata)
print(f"   - E-BFMI: {bfmi.mean():.3f} (should be > 0.2)")

# Check specific parameters for issues
params_to_check = ['delta', 'sigma_eta', 'alpha', 'eta_1']
rhat_issues = []
ess_issues = []

for param in params_to_check:
    param_summary = summary.loc[param]
    rhat = param_summary['r_hat']
    ess_bulk = param_summary['ess_bulk']
    ess_tail = param_summary['ess_tail']

    if rhat > 1.01:
        rhat_issues.append(f"{param}: R-hat={rhat:.4f}")
    if ess_bulk < 400:
        ess_issues.append(f"{param}: ESS_bulk={ess_bulk:.0f}")
    if ess_tail < 400:
        ess_issues.append(f"{param}: ESS_tail={ess_tail:.0f}")

# Overall assessment
print("\n" + "="*80)
print("CONVERGENCE ASSESSMENT")
print("="*80)

all_converged = True
issues = []

# Check R-hat
if summary['r_hat'].max() > 1.01:
    all_converged = False
    issues.append(f"R-hat criterion FAILED: Max R-hat = {summary['r_hat'].max():.4f} > 1.01")
    if rhat_issues:
        issues.append("  Problem parameters: " + ", ".join(rhat_issues))
else:
    print("✓ R-hat criterion PASSED: All R-hat < 1.01")

# Check ESS
if summary['ess_bulk'].min() < 400:
    all_converged = False
    issues.append(f"ESS_bulk criterion FAILED: Min ESS = {summary['ess_bulk'].min():.0f} < 400")
    if ess_issues:
        issues.append("  Problem parameters: " + ", ".join(ess_issues))
else:
    print("✓ ESS_bulk criterion PASSED: All ESS_bulk > 400")

if summary['ess_tail'].min() < 400:
    all_converged = False
    issues.append(f"ESS_tail criterion FAILED: Min ESS = {summary['ess_tail'].min():.0f} < 400")
else:
    print("✓ ESS_tail criterion PASSED: All ESS_tail > 400")

# Check divergences
divergence_rate = divergences / total_samples
if divergence_rate > 0.01:
    all_converged = False
    issues.append(f"Divergence criterion FAILED: {divergences} divergences ({100*divergence_rate:.2f}% > 1%)")
else:
    print(f"✓ Divergence criterion PASSED: {divergences} divergences ({100*divergence_rate:.2f}% < 1%)")

# Check BFMI
if bfmi.mean() < 0.2:
    all_converged = False
    issues.append(f"E-BFMI criterion FAILED: {bfmi.mean():.3f} < 0.2")
else:
    print(f"✓ E-BFMI criterion PASSED: {bfmi.mean():.3f} > 0.2")

# Final verdict
print("\n" + "-"*80)
if all_converged:
    print("OVERALL: ✓ CONVERGENCE ACHIEVED - ALL CRITERIA PASSED")
    verdict = "PASS"
else:
    print("OVERALL: ✗ CONVERGENCE ISSUES DETECTED")
    for issue in issues:
        print(f"  - {issue}")
    verdict = "FAIL"
print("-"*80)

# Save diagnostic report
print("\n10. Saving diagnostic report...")
report = {
    'verdict': verdict,
    'all_converged': all_converged,
    'sampler': 'PyMC',
    'diagnostics': {
        'divergences': int(divergences),
        'divergence_rate': float(divergence_rate),
        'max_rhat': float(summary['r_hat'].max()),
        'min_ess_bulk': float(summary['ess_bulk'].min()),
        'min_ess_tail': float(summary['ess_tail'].min()),
        'mean_bfmi': float(bfmi.mean())
    },
    'issues': issues,
    'parameter_summary': {
        'delta': {
            'mean': float(summary.loc['delta', 'mean']),
            'sd': float(summary.loc['delta', 'sd']),
            'hdi_3%': float(summary.loc['delta', 'hdi_3%']),
            'hdi_97%': float(summary.loc['delta', 'hdi_97%']),
            'rhat': float(summary.loc['delta', 'r_hat']),
            'ess_bulk': float(summary.loc['delta', 'ess_bulk'])
        },
        'sigma_eta': {
            'mean': float(summary.loc['sigma_eta', 'mean']),
            'sd': float(summary.loc['sigma_eta', 'sd']),
            'hdi_3%': float(summary.loc['sigma_eta', 'hdi_3%']),
            'hdi_97%': float(summary.loc['sigma_eta', 'hdi_97%']),
            'rhat': float(summary.loc['sigma_eta', 'r_hat']),
            'ess_bulk': float(summary.loc['sigma_eta', 'ess_bulk'])
        },
        'alpha': {
            'mean': float(summary.loc['alpha', 'mean']),
            'sd': float(summary.loc['alpha', 'sd']),
            'hdi_3%': float(summary.loc['alpha', 'hdi_3%']),
            'hdi_97%': float(summary.loc['alpha', 'hdi_97%']),
            'rhat': float(summary.loc['alpha', 'r_hat']),
            'ess_bulk': float(summary.loc['alpha', 'ess_bulk'])
        }
    }
}

with open(DIAG_DIR / 'diagnostic_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"   ✓ Saved to {DIAG_DIR / 'diagnostic_report.json'}")

print("\n" + "="*80)
print("FITTING COMPLETE")
print("="*80)
print(f"\nResults saved to: {BASE_DIR}")
print(f"  - InferenceData: {DIAG_DIR / 'posterior_inference.netcdf'}")
print(f"  - Diagnostics: {DIAG_DIR / 'diagnostic_report.json'}")
print(f"  - Summary: {DIAG_DIR / 'convergence_summary.csv'}")
