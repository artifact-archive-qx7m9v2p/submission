"""
Fit hierarchical model with skeptical priors (Model 4a)
Skeptical of large effects, expects low heterogeneity
"""

import numpy as np
import pandas as pd
import cmdstanpy
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

# Prepare data for Stan
stan_data = {
    'J': len(df),
    'y': df['y'].values,
    'sigma': df['sigma'].values
}

print("=" * 80)
print("MODEL 4a: SKEPTICAL PRIORS")
print("=" * 80)
print(f"\nData: {stan_data['J']} studies")
print(f"Observed effects: {df['y'].values}")
print(f"Standard errors: {df['sigma'].values}")
print("\nPrior specification:")
print("  mu ~ Normal(0, 10)      [Skeptical of large effects]")
print("  tau ~ Half-Normal(0, 5) [Expects low heterogeneity]")
print()

# Compile model
print("Compiling Stan model...")
stan_file = code_dir / 'hierarchical_skeptical.stan'
model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
print("Model compiled successfully!")

# PROBE RUN: Quick check with short chains
print("\n" + "=" * 80)
print("PHASE 1: PROBE RUN (4 chains × 200 iterations)")
print("=" * 80)

probe_fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=100,
    iter_sampling=100,
    seed=12345,
    show_progress=True,
    adapt_delta=0.8
)

# Check probe diagnostics
probe_summary = probe_fit.summary()
print("\nProbe diagnostics:")
print(probe_summary[['Mean', 'StdDev', 'R_hat', 'ESS_bulk', 'ESS_tail']].head(10))

# Check for issues
max_rhat = probe_summary['R_hat'].max()
min_ess = probe_summary['ESS_bulk'].min()
divergences = probe_fit.divergences.sum() if hasattr(probe_fit, 'divergences') else 0

print(f"\nMax R-hat: {max_rhat:.4f}")
print(f"Min ESS_bulk: {min_ess:.1f}")
print(f"Divergences: {divergences}")

issues = []
if max_rhat > 1.01:
    issues.append(f"High R-hat ({max_rhat:.4f})")
if min_ess < 50:
    issues.append(f"Low ESS ({min_ess:.1f})")
if divergences > 0:
    issues.append(f"{divergences} divergences")

if issues:
    print(f"\nWARNING: Probe detected issues: {', '.join(issues)}")
    print("Will increase adapt_delta and iterations for main run")
    adapt_delta = 0.95
    iter_warmup = 2000
    iter_sampling = 2500
else:
    print("\nProbe successful! Proceeding with standard sampling")
    adapt_delta = 0.8
    iter_warmup = 2500
    iter_sampling = 2500

# MAIN RUN: Full sampling
print("\n" + "=" * 80)
print(f"PHASE 2: MAIN SAMPLING (4 chains × {iter_warmup + iter_sampling} iterations)")
print(f"Settings: adapt_delta={adapt_delta}, warmup={iter_warmup}, sampling={iter_sampling}")
print("=" * 80)

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=iter_warmup,
    iter_sampling=iter_sampling,
    seed=12345,
    show_progress=True,
    adapt_delta=adapt_delta
)

# Convert to ArviZ InferenceData with log-likelihood
print("\nConverting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    posterior_predictive=None,
    log_likelihood='log_lik'
)

# Save InferenceData
netcdf_path = diag_dir / 'posterior_inference.netcdf'
idata.to_netcdf(netcdf_path)
print(f"Saved InferenceData to: {netcdf_path}")

# Compute summary statistics
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])
print(summary)

# Save summary
summary_path = diag_dir / 'posterior_summary.csv'
summary.to_csv(summary_path)
print(f"\nSaved summary to: {summary_path}")

# Extract key parameters
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()

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
    print("\n✓ All convergence criteria met!")
else:
    print("\n⚠ Some convergence criteria not met - see diagnostics")

# Save convergence report
convergence_report = f"""# Convergence Report: Model 4a (Skeptical Priors)

## Sampling Configuration
- Chains: 4
- Warmup iterations: {iter_warmup}
- Sampling iterations: {iter_sampling}
- adapt_delta: {adapt_delta}

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
