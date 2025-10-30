"""
Fit Negative Binomial State-Space Model to Real Data
Using CmdStanPy with HMC/NUTS sampling

Model: C_t ~ NegativeBinomial(exp(η_t), φ)
       η_t ~ Normal(η_{t-1} + δ, σ_η)

Priors: δ ~ Normal(0.05, 0.02)
        σ_η ~ Exponential(20)
        φ ~ Exponential(0.05)
        η_1 ~ Normal(log(50), 1)
"""

import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
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
print("="*80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"   - Observations: {len(data)}")
print(f"   - Count range: {data['C'].min()} to {data['C'].max()}")
print(f"   - Mean count: {data['C'].mean():.1f}")
print(f"   - Variance: {data['C'].var():.1f}")
print(f"   - Variance/Mean ratio: {data['C'].var() / data['C'].mean():.1f}")

# Prepare Stan data
stan_data = {
    'N': len(data),
    'C': data['C'].values.astype(int).tolist()
}

# Compile model
print("\n2. Compiling Stan model...")
model = CmdStanModel(stan_file=str(CODE_DIR / "model.stan"))
print("   ✓ Model compiled successfully")

# Initial probe: Short chains to assess behavior
print("\n3. Initial probe sampling (100 iterations)...")
print("   Purpose: Quick assessment of model behavior")
try:
    probe_fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=100,
        iter_sampling=100,
        adapt_delta=0.95,
        max_treedepth=12,
        show_progress=True,
        show_console=False
    )

    # Check probe diagnostics
    probe_summary = probe_fit.summary()
    probe_divergences = probe_fit.num_divergences()

    print(f"\n   Probe Results:")
    print(f"   - Divergences: {probe_divergences}")
    print(f"   - Max R-hat: {probe_summary['R_hat'].max():.4f}")
    print(f"   - Min ESS_bulk: {probe_summary['ess_bulk'].min():.0f}")

    if probe_divergences > 10:
        print("   ⚠ Warning: Divergences detected in probe. Will increase adapt_delta for main sampling.")
        adapt_delta_main = 0.99
    else:
        print("   ✓ Probe successful, proceeding with main sampling")
        adapt_delta_main = 0.95

except Exception as e:
    print(f"   ⚠ Probe failed: {e}")
    print("   Proceeding with main sampling at adapt_delta=0.99")
    adapt_delta_main = 0.99

# Main sampling
print(f"\n4. Main HMC sampling (adapt_delta={adapt_delta_main})...")
print("   - Chains: 4")
print("   - Warmup: 1000")
print("   - Sampling: 2000 per chain")
print("   - Total post-warmup samples: 8000")

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=adapt_delta_main,
    max_treedepth=12,
    show_progress=True,
    show_console=False,
    seed=42
)

print("\n   ✓ Sampling completed")

# Extract diagnostics
print("\n5. Extracting diagnostics...")
summary = fit.summary()
divergences = fit.num_divergences()
max_treedepth = fit.num_max_treedepth()

# Save CmdStan diagnostics
fit.diagnose()

print(f"\n   Convergence Diagnostics:")
print(f"   - Total divergences: {divergences} ({100*divergences/(4*2000):.2f}%)")
print(f"   - Max treedepth saturations: {max_treedepth}")
print(f"   - Max R-hat: {summary['R_hat'].max():.4f}")
print(f"   - Min ESS_bulk: {summary['ess_bulk'].min():.0f}")
print(f"   - Min ESS_tail: {summary['ess_tail'].min():.0f}")

# Convert to ArviZ InferenceData with log_likelihood
print("\n6. Converting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive=['C_pred'],
    log_likelihood='log_lik',
    observed_data={'C': data['C'].values},
    coords={'time': data['year'].values},
    dims={'log_lik': ['time'], 'C_pred': ['time'], 'eta': ['time'], 'eta_raw': ['time_minus_1']}
)

# Save InferenceData
print(f"   Saving to {DIAG_DIR / 'posterior_inference.netcdf'}...")
idata.to_netcdf(DIAG_DIR / "posterior_inference.netcdf")
print("   ✓ InferenceData saved with log_likelihood group")

# Save summary statistics
print("\n7. Saving summary statistics...")
summary_path = DIAG_DIR / "convergence_summary.csv"
summary.to_csv(summary_path)
print(f"   ✓ Saved to {summary_path}")

# ArviZ summary with focus on key parameters
az_summary = az.summary(idata, var_names=['delta', 'sigma_eta', 'phi', 'eta_1'])
print("\n   Key Parameter Summary:")
print(az_summary.to_string())

# Save detailed diagnostics
print("\n8. Generating detailed diagnostics...")

# Energy diagnostics
energy_data = fit.method_variables()
bfmi = az.bfmi(idata)
print(f"\n   Energy Diagnostics:")
print(f"   - E-BFMI: {bfmi.mean():.3f} (should be > 0.2)")

# Check specific parameters for issues
params_to_check = ['delta', 'sigma_eta', 'phi', 'eta_1']
rhat_issues = []
ess_issues = []

for param in params_to_check:
    param_summary = summary[summary.index.str.startswith(param)]
    max_rhat = param_summary['R_hat'].max()
    min_ess_bulk = param_summary['ess_bulk'].min()
    min_ess_tail = param_summary['ess_tail'].min()

    if max_rhat > 1.01:
        rhat_issues.append(f"{param}: R-hat={max_rhat:.4f}")
    if min_ess_bulk < 400:
        ess_issues.append(f"{param}: ESS_bulk={min_ess_bulk:.0f}")
    if min_ess_tail < 400:
        ess_issues.append(f"{param}: ESS_tail={min_ess_tail:.0f}")

# Overall assessment
print("\n" + "="*80)
print("CONVERGENCE ASSESSMENT")
print("="*80)

all_converged = True
issues = []

# Check R-hat
if summary['R_hat'].max() > 1.01:
    all_converged = False
    issues.append(f"R-hat criterion FAILED: Max R-hat = {summary['R_hat'].max():.4f} > 1.01")
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
divergence_rate = divergences / (4 * 2000)
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
print("\n9. Saving diagnostic report...")
report = {
    'verdict': verdict,
    'all_converged': all_converged,
    'diagnostics': {
        'divergences': int(divergences),
        'divergence_rate': float(divergence_rate),
        'max_treedepth_hits': int(max_treedepth),
        'max_rhat': float(summary['R_hat'].max()),
        'min_ess_bulk': float(summary['ess_bulk'].min()),
        'min_ess_tail': float(summary['ess_tail'].min()),
        'mean_bfmi': float(bfmi.mean())
    },
    'issues': issues,
    'parameter_summary': {
        'delta': {
            'mean': float(az_summary.loc['delta', 'mean']),
            'sd': float(az_summary.loc['delta', 'sd']),
            'hdi_3%': float(az_summary.loc['delta', 'hdi_3%']),
            'hdi_97%': float(az_summary.loc['delta', 'hdi_97%']),
            'rhat': float(az_summary.loc['delta', 'r_hat']),
            'ess_bulk': float(az_summary.loc['delta', 'ess_bulk'])
        },
        'sigma_eta': {
            'mean': float(az_summary.loc['sigma_eta', 'mean']),
            'sd': float(az_summary.loc['sigma_eta', 'sd']),
            'hdi_3%': float(az_summary.loc['sigma_eta', 'hdi_3%']),
            'hdi_97%': float(az_summary.loc['sigma_eta', 'hdi_97%']),
            'rhat': float(az_summary.loc['sigma_eta', 'r_hat']),
            'ess_bulk': float(az_summary.loc['sigma_eta', 'ess_bulk'])
        },
        'phi': {
            'mean': float(az_summary.loc['phi', 'mean']),
            'sd': float(az_summary.loc['phi', 'sd']),
            'hdi_3%': float(az_summary.loc['phi', 'hdi_3%']),
            'hdi_97%': float(az_summary.loc['phi', 'hdi_97%']),
            'rhat': float(az_summary.loc['phi', 'r_hat']),
            'ess_bulk': float(az_summary.loc['phi', 'ess_bulk'])
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
