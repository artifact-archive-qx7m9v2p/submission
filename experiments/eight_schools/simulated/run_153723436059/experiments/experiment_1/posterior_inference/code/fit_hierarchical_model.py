"""
Fit Hierarchical Eight Schools Model using PyMC with HMC/NUTS

Adaptive sampling strategy:
1. Initial probe (short chains) to assess model behavior
2. Main sampling if probe succeeds
3. Diagnostics and troubleshooting
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
DATA_PATH = Path('/workspace/data/data.csv')
OUTPUT_DIR = Path('/workspace/experiments/experiment_1/posterior_inference')
DIAGNOSTICS_DIR = OUTPUT_DIR / 'diagnostics'
PLOTS_DIR = OUTPUT_DIR / 'plots'

print("="*80)
print("HIERARCHICAL EIGHT SCHOOLS MODEL - HMC SAMPLING WITH PYMC")
print("="*80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"\nData loaded: {len(df)} schools")
print(df.to_string(index=False))

# Extract data
J = len(df)
y_obs = df['effect'].values
sigma_obs = df['sigma'].values

print(f"\nObserved effects: mean={y_obs.mean():.2f}, std={y_obs.std():.2f}, range=[{y_obs.min():.2f}, {y_obs.max():.2f}]")

# Build model
print("\n[2] Building hierarchical model with non-centered parameterization...")

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=0, sigma=50)
    tau = pm.HalfCauchy('tau', beta=25)

    # Non-centered parameterization
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y = pm.Normal('y', mu=theta, sigma=sigma_obs, observed=y_obs)

    # Posterior predictive
    y_rep = pm.Normal('y_rep', mu=theta, sigma=sigma_obs, shape=J)

print("\nModel structure:")
print(hierarchical_model)

# ADAPTIVE SAMPLING STRATEGY

print("\n" + "="*80)
print("PHASE 1: INITIAL PROBE (200 iterations)")
print("="*80)
print("\nRunning short chains to assess model behavior...")

with hierarchical_model:
    probe_trace = pm.sample(
        draws=200,
        tune=200,
        chains=4,
        cores=4,
        target_accept=0.95,
        return_inferencedata=True,
        random_seed=42
    )

# Quick diagnostics on probe
probe_summary = az.summary(probe_trace, var_names=['mu', 'tau'])
print("\nProbe summary (mu, tau):")
print(probe_summary)

probe_rhat_max = probe_summary['r_hat'].max()
probe_ess_min = probe_summary['ess_bulk'].min()
probe_divergences = probe_trace.sample_stats.diverging.sum().item()

print(f"\nProbe diagnostics:")
print(f"  Max R-hat: {probe_rhat_max:.4f}")
print(f"  Min ESS_bulk: {probe_ess_min:.1f}")
print(f"  Divergences: {probe_divergences}")

if probe_rhat_max > 1.05 or probe_divergences > 10:
    print("\n*** WARNING: Probe shows convergence issues! ***")
    print("    Proceeding cautiously with main sampling...")
else:
    print("\n*** Probe successful! Model behavior looks good. ***")

# MAIN SAMPLING

print("\n" + "="*80)
print("PHASE 2: MAIN SAMPLING")
print("="*80)
print("\nRunning full MCMC chains...")

with hierarchical_model:
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.95,
        return_inferencedata=True,
        random_seed=123,
        idata_kwargs={'log_likelihood': True}  # CRITICAL: Include log_likelihood for LOO-CV
    )

    # Add posterior predictive samples
    trace.extend(pm.sample_posterior_predictive(trace, random_seed=456))

print("\n*** Sampling complete! ***")

# CONVERGENCE DIAGNOSTICS

print("\n" + "="*80)
print("PHASE 3: CONVERGENCE DIAGNOSTICS")
print("="*80)

# Full summary statistics
summary = az.summary(trace, var_names=['mu', 'tau', 'theta'])
print("\nPosterior summary (all parameters):")
print(summary)

# Save summary
summary_path = DIAGNOSTICS_DIR / 'posterior_summary.csv'
summary.to_csv(summary_path)
print(f"\nSummary saved to: {summary_path}")

# Key diagnostics
rhat_max = summary['r_hat'].max()
rhat_issues = summary[summary['r_hat'] > 1.01]
ess_bulk_min = summary['ess_bulk'].min()
ess_tail_min = summary['ess_tail'].min()
ess_issues = summary[(summary['ess_bulk'] < 400) | (summary['ess_tail'] < 400)]

divergences = trace.sample_stats.diverging.sum().item()
total_draws = len(trace.posterior.chain) * len(trace.posterior.draw)

print("\n" + "="*80)
print("CONVERGENCE REPORT")
print("="*80)

print(f"\n[1] R-hat (convergence criterion: < 1.01)")
print(f"    Max R-hat: {rhat_max:.4f}")
if len(rhat_issues) > 0:
    print(f"    *** {len(rhat_issues)} parameters with R-hat > 1.01:")
    print(rhat_issues[['mean', 'sd', 'r_hat']])
else:
    print("    ✓ All parameters converged (R-hat < 1.01)")

print(f"\n[2] Effective Sample Size (criterion: > 400)")
print(f"    Min ESS_bulk: {ess_bulk_min:.1f}")
print(f"    Min ESS_tail: {ess_tail_min:.1f}")
if len(ess_issues) > 0:
    print(f"    *** {len(ess_issues)} parameters with ESS < 400:")
    print(ess_issues[['mean', 'sd', 'ess_bulk', 'ess_tail']])
else:
    print("    ✓ All parameters have sufficient ESS (> 400)")

print(f"\n[3] Divergent Transitions (criterion: 0)")
print(f"    Divergences: {divergences} / {total_draws} ({100*divergences/total_draws:.2f}%)")
if divergences == 0:
    print("    ✓ No divergent transitions")
elif divergences < total_draws * 0.01:
    print("    ! Minor divergences (<1% of samples)")
else:
    print("    *** Significant divergences detected")

# Energy diagnostic
energy_diff = az.bfmi(trace)
print(f"\n[4] Energy Diagnostic (E-BFMI, criterion: > 0.2)")
print(f"    E-BFMI: {energy_diff.mean():.3f}")
if energy_diff.mean() > 0.2:
    print("    ✓ No energy transition problems")
else:
    print("    *** Low E-BFMI suggests sampling difficulties")

# Overall assessment
print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

converged = (rhat_max < 1.01 and
             ess_bulk_min > 400 and
             ess_tail_min > 400 and

             divergences == 0 and
             energy_diff.mean() > 0.2)

if converged:
    print("\n✓✓✓ SAMPLING SUCCESSFUL - ALL CRITERIA MET ✓✓✓")
    print("\nThe model has converged properly and is ready for inference.")
elif rhat_max < 1.01 and divergences < total_draws * 0.01:
    print("\n✓ CONDITIONAL SUCCESS - MINOR ISSUES")
    print("\nR-hat indicates convergence, but some efficiency concerns.")
    print("Posterior inference should be reliable.")
else:
    print("\n*** CONVERGENCE ISSUES DETECTED ***")
    print("\nFurther investigation required before trusting inference.")

# Save InferenceData with log_likelihood for LOO-CV
print("\n[4] Saving posterior samples...")
idata_path = DIAGNOSTICS_DIR / 'posterior_inference.netcdf'
trace.to_netcdf(str(idata_path))
print(f"\nInferenceData saved to: {idata_path}")
print(f"  - posterior group: {list(trace.posterior.data_vars.keys())}")
print(f"  - posterior_predictive group: {list(trace.posterior_predictive.data_vars.keys())}")
if hasattr(trace, 'log_likelihood'):
    print(f"  - log_likelihood group: {list(trace.log_likelihood.data_vars.keys())}")
    print("  ✓ log_likelihood included (required for LOO-CV)")
else:
    print("  *** WARNING: log_likelihood not found!")

print("\n" + "="*80)
print("Posterior inference complete!")
print("="*80)
