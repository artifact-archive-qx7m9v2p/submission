"""
Fit hierarchical model with skeptical priors (Model 4a) using PyMC
Skeptical of large effects, expects low heterogeneity
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set up paths
exp_dir = Path('/workspace/experiments/experiment_4/experiment_4a_skeptical/posterior_inference')
code_dir = exp_dir / 'code'
diag_dir = exp_dir / 'diagnostics'
plot_dir = exp_dir / 'plots'

# Load data
data_path = Path('/workspace/data/data.csv')
df = pd.read_csv(data_path)

print("=" * 80)
print("MODEL 4a: SKEPTICAL PRIORS (PyMC)")
print("=" * 80)
print(f"\nData: {len(df)} studies")
print(f"Observed effects: {df['y'].values}")
print(f"Standard errors: {df['sigma'].values}")
print("\nPrior specification:")
print("  mu ~ Normal(0, 10)      [Skeptical of large effects]")
print("  tau ~ Half-Normal(0, 5) [Expects low heterogeneity]")
print()

# Build model
print("Building PyMC model...")
with pm.Model() as model:
    # Skeptical priors
    mu = pm.Normal('mu', mu=0, sigma=10)  # Skeptical of large effects
    tau = pm.HalfNormal('tau', sigma=5)   # Expects low heterogeneity

    # Study-specific effects
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=len(df))

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=df['sigma'].values, observed=df['y'].values)

print("Model built successfully!")
print(f"Model variables: {[v.name for v in model.unobserved_RVs]}")

# PROBE RUN: Quick check with short chains
print("\n" + "=" * 80)
print("PHASE 1: PROBE RUN (4 chains × 200 iterations)")
print("=" * 80)

with model:
    probe_trace = pm.sample(
        draws=100,
        tune=100,
        chains=4,
        cores=4,
        random_seed=12345,
        progressbar=True,
        return_inferencedata=True
    )

# Check probe diagnostics
probe_summary = az.summary(probe_trace, var_names=['mu', 'tau'])
print("\nProbe diagnostics:")
print(probe_summary)

max_rhat = probe_summary['r_hat'].max()
min_ess = probe_summary['ess_bulk'].min()

print(f"\nMax R-hat: {max_rhat:.4f}")
print(f"Min ESS_bulk: {min_ess:.1f}")

issues = []
if max_rhat > 1.01:
    issues.append(f"High R-hat ({max_rhat:.4f})")
if min_ess < 50:
    issues.append(f"Low ESS ({min_ess:.1f})")

if issues:
    print(f"\nWARNING: Probe detected issues: {', '.join(issues)}")
    print("Will increase iterations for main run")
    tune = 2000
    draws = 2500
else:
    print("\nProbe successful! Proceeding with standard sampling")
    tune = 2500
    draws = 2500

# MAIN RUN: Full sampling
print("\n" + "=" * 80)
print(f"PHASE 2: MAIN SAMPLING (4 chains × {tune + draws} iterations)")
print(f"Settings: tune={tune}, draws={draws}")
print("=" * 80)

with model:
    trace = pm.sample(
        draws=draws,
        tune=tune,
        chains=4,
        cores=4,
        random_seed=12345,
        progressbar=True,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )

# Save InferenceData
print("\nSaving InferenceData...")
netcdf_path = diag_dir / 'posterior_inference.netcdf'
trace.to_netcdf(netcdf_path)
print(f"Saved InferenceData to: {netcdf_path}")

# Compute summary statistics
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

summary = az.summary(trace, var_names=['mu', 'tau', 'theta'])
print(summary)

# Save summary
summary_path = diag_dir / 'posterior_summary.csv'
summary.to_csv(summary_path)
print(f"\nSaved summary to: {summary_path}")

# Extract key parameters
mu_samples = trace.posterior['mu'].values.flatten()
tau_samples = trace.posterior['tau'].values.flatten()

mu_mean = mu_samples.mean()
mu_sd = mu_samples.std()
mu_ci = np.percentile(mu_samples, [2.5, 97.5])

tau_mean = tau_samples.mean()
tau_sd = tau_samples.std()
tau_ci = np.percentile(tau_samples, [2.5, 97.5])

print("\n" + "=" * 80)
print("POSTERIOR ESTIMATES")
print("=" * 80)
print(f"\nmu (population mean):")
print(f"  Posterior: {mu_mean:.2f} ± {mu_sd:.2f}")
print(f"  95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]")
print(f"  Prior: 0 ± 10 (skeptical)")

print(f"\ntau (population SD):")
print(f"  Posterior: {tau_mean:.2f} ± {tau_sd:.2f}")
print(f"  95% CI: [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]")
print(f"  Prior: Half-Normal(0, 5)")

# Prior-posterior comparison
prior_shift_mu = abs(mu_mean - 0)  # Prior centered at 0
print(f"\nPrior-posterior shift:")
print(f"  mu shifted {prior_shift_mu:.2f} units from prior mean (0)")
print(f"  Data pulled estimate away from skeptical prior")

# Check convergence criteria
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()

print("\n" + "=" * 80)
print("CONVERGENCE ASSESSMENT")
print("=" * 80)
print(f"Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"Min ESS_bulk: {min_ess_bulk:.1f} (target: > 400)")
print(f"Min ESS_tail: {min_ess_tail:.1f} (target: > 400)")

convergence_ok = max_rhat < 1.01 and min_ess_bulk > 400 and min_ess_tail > 400

if convergence_ok:
    print("\nAll convergence criteria met!")
else:
    print("\nSome convergence criteria not met - see diagnostics")

# Save convergence report
convergence_report = f"""# Convergence Report: Model 4a (Skeptical Priors)

## Sampling Configuration
- Sampler: PyMC (NUTS)
- Chains: 4
- Warmup iterations: {tune}
- Sampling iterations: {draws}

## Quantitative Diagnostics

### Convergence Metrics
- Max R-hat: {max_rhat:.4f} (target: < 1.01) {'✓' if max_rhat < 1.01 else '✗'}
- Min ESS_bulk: {min_ess_bulk:.1f} (target: > 400) {'✓' if min_ess_bulk > 400 else '✗'}
- Min ESS_tail: {min_ess_tail:.1f} (target: > 400) {'✓' if min_ess_tail > 400 else '✗'}

### Overall Assessment
{('All convergence criteria met. Chains mixed well and explored the posterior efficiently.'
  if convergence_ok else
  'Some convergence criteria not met. See visual diagnostics for details.')}

## Posterior Estimates

### mu (population mean)
- Posterior: {mu_mean:.2f} ± {mu_sd:.2f}
- 95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]
- Prior: Normal(0, 10) [skeptical]
- Shift from prior: {prior_shift_mu:.2f} units

### tau (population SD)
- Posterior: {tau_mean:.2f} ± {tau_sd:.2f}
- 95% CI: [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]
- Prior: Half-Normal(0, 5) [low heterogeneity]

## Visual Diagnostics

See plots/ directory for:
- convergence_overview.png: Trace and rank plots for key parameters
- prior_posterior_overlay.png: Prior vs posterior comparison
- forest_plot.png: Study-specific effects

"""

report_path = diag_dir / 'convergence_report.md'
with open(report_path, 'w') as f:
    f.write(convergence_report)
print(f"\nSaved convergence report to: {report_path}")

# Save key results as JSON for later comparison
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
        'converged': convergence_ok
    }
}

json_path = diag_dir / 'results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved results to: {json_path}")

print("\n" + "=" * 80)
print("FITTING COMPLETE - Model 4a (Skeptical)")
print("=" * 80)
