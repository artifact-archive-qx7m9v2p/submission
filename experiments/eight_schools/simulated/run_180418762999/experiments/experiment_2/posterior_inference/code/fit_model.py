#!/usr/bin/env python
"""
Fit Hierarchical Partial Pooling Model (Experiment 2) to Real Data

Model Specification:
- Likelihood: y_i ~ Normal(theta_i, sigma_i)
- Group level: theta_i = mu + tau * theta_raw_i
- Hyperpriors: mu ~ Normal(10, 20), tau ~ Half-Normal(0, 10)
- Non-centered parameterization to avoid funnel geometry

Expected: tau ≈ 0 based on EDA (complete pooling adequate)

Using PyMC for MCMC inference
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
OUTPUT_DIR = BASE_DIR / 'experiments' / 'experiment_2' / 'posterior_inference'
CODE_DIR = OUTPUT_DIR / 'code'
DIAG_DIR = OUTPUT_DIR / 'diagnostics'
PLOT_DIR = OUTPUT_DIR / 'plots'

# Ensure directories exist
CODE_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Hierarchical Partial Pooling Model - Posterior Inference")
print("="*80)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
y_obs = df['y'].values
sigma_obs = df['sigma'].values
n_groups = len(y_obs)

print(f"  - Observations: {n_groups}")
print(f"  - y_obs range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"  - sigma_obs range: [{sigma_obs.min():.2f}, {sigma_obs.max():.2f}]")

# Build model
print("\n[2/6] Building PyMC model (Non-Centered Parameterization)...")
with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=10, sigma=20)
    tau = pm.HalfNormal('tau', sigma=10)  # Between-group SD

    # Non-centered parameterization (avoid funnel)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=n_groups)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood (known measurement error)
    y = pm.Normal('y', mu=theta, sigma=sigma_obs, observed=y_obs)

    print("  Model structure:")
    print(f"    - Hyperparameters: mu (population mean), tau (between-group SD)")
    print(f"    - Group parameters: theta[0..{n_groups-1}] (group-specific means)")
    print(f"    - Total parameters: 2 + {n_groups} + {n_groups} = {2 + 2*n_groups}")
    print(f"    - Parameterization: Non-centered (theta = mu + tau * theta_raw)")

# Initial probe sampling (quick diagnostic)
print("\n[3/6] Initial probe sampling (4 chains × 200 iterations)...")
print("  Purpose: Quick assessment of model behavior")

with hierarchical_model:
    trace_probe = pm.sample(
        draws=200,
        tune=200,
        chains=4,
        target_accept=0.90,
        return_inferencedata=True,
        random_seed=42,
        progressbar=False
    )

# Check probe diagnostics
probe_summary = az.summary(trace_probe, var_names=['mu', 'tau'])
probe_rhat = probe_summary['r_hat'].max()
probe_ess = probe_summary['ess_bulk'].min()
divergences_probe = trace_probe.sample_stats.diverging.sum().item()

print(f"\n  Probe diagnostics:")
print(f"    - Max R-hat: {probe_rhat:.4f}")
print(f"    - Min ESS: {probe_ess:.0f}")
print(f"    - Divergences: {divergences_probe}")

# Decide on main sampling strategy
if probe_rhat > 1.05 or divergences_probe > 20:
    print("\n  WARNING: Probe shows issues - increasing target_accept and tuning")
    target_accept = 0.99
    tune_samples = 3000
else:
    print("\n  Probe looks good - proceeding with standard settings")
    target_accept = 0.95
    tune_samples = 2000

# Main sampling
print(f"\n[4/6] Main sampling (4 chains × 2000 draws, target_accept={target_accept})...")
print("  This may take a few minutes for hierarchical model...")

with hierarchical_model:
    trace = pm.sample(
        draws=2000,
        tune=tune_samples,
        chains=4,
        target_accept=target_accept,
        return_inferencedata=True,
        random_seed=42,
        progressbar=True
    )

    # CRITICAL: Compute log-likelihood for LOO-CV comparison
    print("\n  Computing log-likelihood for LOO-CV...")
    pm.compute_log_likelihood(trace)

print("\n  Sampling complete!")

# Save InferenceData
print("\n[5/6] Saving posterior samples...")
netcdf_path = DIAG_DIR / 'posterior_inference.netcdf'
trace.to_netcdf(netcdf_path)
print(f"  Saved: {netcdf_path}")

# Convergence diagnostics
print("\n[6/6] Running convergence diagnostics...")

# Summary statistics
summary = az.summary(trace, var_names=['mu', 'tau', 'theta'])
summary_path = DIAG_DIR / 'posterior_summary.csv'
summary.to_csv(summary_path)
print(f"  Saved: {summary_path}")

# Key metrics
mu_rhat = summary.loc['mu', 'r_hat']
tau_rhat = summary.loc['tau', 'r_hat']
theta_rhat_max = summary.filter(like='theta[', axis=0)['r_hat'].max()
mu_ess = summary.loc['mu', 'ess_bulk']
tau_ess = summary.loc['tau', 'ess_bulk']
theta_ess_min = summary.filter(like='theta[', axis=0)['ess_bulk'].min()

# Divergences
divergences = trace.sample_stats.diverging.sum().item()
n_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
divergence_pct = 100 * divergences / n_samples

# Save convergence summary
conv_summary = pd.DataFrame({
    'parameter': ['mu', 'tau', 'theta_max'],
    'r_hat': [mu_rhat, tau_rhat, theta_rhat_max],
    'ess_bulk': [mu_ess, tau_ess, theta_ess_min],
    'divergences': [divergences, divergences, divergences],
    'divergence_pct': [divergence_pct, divergence_pct, divergence_pct]
})
conv_path = DIAG_DIR / 'convergence_summary.csv'
conv_summary.to_csv(conv_path, index=False)
print(f"  Saved: {conv_path}")

# Print diagnostics
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

print(f"\nR-hat (must be < 1.01):")
print(f"  mu:           {mu_rhat:.4f}")
print(f"  tau:          {tau_rhat:.4f}")
print(f"  theta (max):  {theta_rhat_max:.4f}")

print(f"\nESS Bulk (prefer > 400):")
print(f"  mu:           {mu_ess:.0f}")
print(f"  tau:          {tau_ess:.0f}")
print(f"  theta (min):  {theta_ess_min:.0f}")

print(f"\nDivergences:")
print(f"  Count:        {divergences}")
print(f"  Percentage:   {divergence_pct:.2f}%")

# Overall assessment
print("\n" + "-"*80)
print("CONVERGENCE ASSESSMENT:")
all_rhat_good = (mu_rhat < 1.01) and (tau_rhat < 1.01) and (theta_rhat_max < 1.01)
all_ess_good = (mu_ess > 100) and (tau_ess > 100) and (theta_ess_min > 100)
div_acceptable = divergence_pct < 5.0

if all_rhat_good and all_ess_good and div_acceptable:
    print("  ✓ PASS - All convergence criteria met")
elif all_rhat_good and all_ess_good and divergence_pct < 10:
    print("  ~ ACCEPTABLE - Convergence good, some divergences (common for hierarchical)")
else:
    print("  ✗ CONCERNS - Review diagnostics carefully")
    if not all_rhat_good:
        print("    - R-hat > 1.01 detected")
    if not all_ess_good:
        print("    - ESS < 100 detected")
    if divergence_pct >= 10:
        print(f"    - High divergence rate ({divergence_pct:.1f}%)")

print("-"*80)

# Key posterior results
print("\n" + "="*80)
print("POSTERIOR ESTIMATES (Key Parameters)")
print("="*80)

mu_samples = trace.posterior['mu'].values.flatten()
tau_samples = trace.posterior['tau'].values.flatten()
theta_samples = trace.posterior['theta'].values

mu_mean = mu_samples.mean()
mu_sd = mu_samples.std()
mu_hdi = az.hdi(trace.posterior['mu'], hdi_prob=0.95)
mu_hdi_lower = float(mu_hdi.sel(hdi='lower').values())
mu_hdi_upper = float(mu_hdi.sel(hdi='higher').values())

tau_mean = tau_samples.mean()
tau_sd = tau_samples.std()
tau_median = np.median(tau_samples)
tau_hdi = az.hdi(trace.posterior['tau'], hdi_prob=0.95)
tau_hdi_lower = float(tau_hdi.sel(hdi='lower').values())
tau_hdi_upper = float(tau_hdi.sel(hdi='higher').values())

print(f"\nmu (population mean):")
print(f"  Mean ± SD:    {mu_mean:.3f} ± {mu_sd:.3f}")
print(f"  95% HDI:      [{mu_hdi_lower:.3f}, {mu_hdi_upper:.3f}]")

print(f"\ntau (between-group SD):")
print(f"  Mean ± SD:    {tau_mean:.3f} ± {tau_sd:.3f}")
print(f"  Median:       {tau_median:.3f}")
print(f"  95% HDI:      [{tau_hdi_lower:.3f}, {tau_hdi_upper:.3f}]")

# tau interpretation
print(f"\ntau Interpretation:")
if tau_hdi_upper < 1.0:
    print(f"  → tau 95% HDI entirely below 1.0")
    print(f"  → NO evidence for between-group heterogeneity")
    print(f"  → Model effectively reduces to complete pooling (Model 1)")
    print(f"  → Expected outcome: REJECT in favor of Model 1 (parsimony)")
elif tau_hdi_lower > 3.0:
    print(f"  → tau clearly positive (HDI above 3)")
    print(f"  → Evidence FOR between-group heterogeneity")
    print(f"  → Hierarchical structure supported by data")
    print(f"  → Proceed to LOO-CV comparison with Model 1")
else:
    print(f"  → tau uncertain (HDI includes small and moderate values)")
    print(f"  → Weak evidence for heterogeneity")
    print(f"  → LOO-CV will be decisive")

# theta shrinkage
theta_means = theta_samples.mean(axis=(0, 1))
print(f"\ntheta[i] (group means):")
print(f"  Range:        [{theta_means.min():.3f}, {theta_means.max():.3f}]")
print(f"  Spread (SD):  {theta_means.std():.3f}")
print(f"  Shrinkage:    Toward mu = {mu_mean:.3f}")

# Individual group estimates
print(f"\n  Group-specific estimates:")
for i in range(n_groups):
    theta_i = trace.posterior['theta'].isel(theta_dim_0=i)
    theta_hdi = az.hdi(theta_i, hdi_prob=0.95)
    theta_hdi_lower = float(theta_hdi.sel(hdi='lower').values())
    theta_hdi_upper = float(theta_hdi.sel(hdi='higher').values())
    print(f"    theta[{i}]: {theta_means[i]:7.3f}  95% HDI: [{theta_hdi_lower:7.3f}, {theta_hdi_upper:7.3f}]  (y={y_obs[i]:7.3f})")

print("\n" + "="*80)
print("Fitting complete! Proceed to visualization and detailed diagnostics.")
print("="*80)
