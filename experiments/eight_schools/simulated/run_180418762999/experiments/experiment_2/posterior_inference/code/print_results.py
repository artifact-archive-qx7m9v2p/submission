#!/usr/bin/env python
"""Print results from already-fitted hierarchical model"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path

# Paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
DIAG_DIR = BASE_DIR / 'experiments' / 'experiment_2' / 'posterior_inference' / 'diagnostics'

# Load data
df = pd.read_csv(DATA_PATH)
y_obs = df['y'].values
n_groups = len(y_obs)

# Load fitted model
print("Loading posterior samples...")
idata = az.from_netcdf(DIAG_DIR / 'posterior_inference.netcdf')

# Extract samples
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values

# Calculate statistics
mu_mean = mu_samples.mean()
mu_sd = mu_samples.std()
mu_hdi = az.hdi(idata.posterior['mu'])
mu_hdi_lower = float(mu_hdi['mu'].sel(hdi='lower'))
mu_hdi_upper = float(mu_hdi['mu'].sel(hdi='higher'))

tau_mean = tau_samples.mean()
tau_sd = tau_samples.std()
tau_median = np.median(tau_samples)
tau_hdi = az.hdi(idata.posterior['tau'])
tau_hdi_lower = float(tau_hdi['tau'].sel(hdi='lower'))
tau_hdi_upper = float(tau_hdi['tau'].sel(hdi='higher'))

print("\n" + "="*80)
print("POSTERIOR ESTIMATES (Key Parameters)")
print("="*80)

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
    theta_i = idata.posterior['theta'].isel(theta_dim_0=i)
    theta_hdi = az.hdi(theta_i)
    theta_hdi_lower = float(theta_hdi['theta'].sel(hdi='lower'))
    theta_hdi_upper = float(theta_hdi['theta'].sel(hdi='higher'))
    print(f"    theta[{i}]: {theta_means[i]:7.3f}  95% HDI: [{theta_hdi_lower:7.3f}, {theta_hdi_upper:7.3f}]  (y={y_obs[i]:7.3f})")

print("\n" + "="*80)
