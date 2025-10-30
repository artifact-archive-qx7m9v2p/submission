"""
Posterior Predictive Checks for Eight Schools Model
===================================================

Assesses model adequacy by comparing observed data with posterior predictive distribution.

Checks:
1. Graphical PPCs: density overlays, school-by-school comparisons
2. Coverage: % of schools in 95% posterior predictive intervals
3. Calibration: LOO-PIT uniformity test
4. Test statistics: max standardized residual, SD, range
5. Specific concerns: extreme schools, precision-residual correlation

Decision Criteria:
- PASS: >85% coverage, p-values in [0.05, 0.95], LOO-PIT uniform
- CONCERN: <85% coverage, extreme p-values, systematic patterns
- FAIL: <70% coverage, p-values <0.001, clear misfit
"""

import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl"); sys.stdout.flush()"
sns.set_context("notebook", font_scale=1.1)

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_predictive_check"); sys.stdout.flush()"
INFERENCE_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"); sys.stdout.flush()"
DATA_PATH = Path("/workspace/data/data.csv"); sys.stdout.flush()"
PLOTS_DIR = BASE_DIR / "plots"

print("="*80)
print("POSTERIOR PREDICTIVE CHECKS: Eight Schools Model"); sys.stdout.flush()"
print("="*80)

# =============================================================================
# 1. LOAD DATA AND POSTERIOR
# =============================================================================
print("\n[1/6] Loading data and posterior samples..."); sys.stdout.flush()"

# Load observed data
data = pd.read_csv(DATA_PATH)
J = len(data)
y_obs = data['y'].values
sigma_obs = data['sigma'].values
school_names = [f"School {i+1}" for i in range(J)]

print(f"Loaded {J} schools"); sys.stdout.flush()"
print(data)

# Load posterior inference
print(f"\nLoading InferenceData from: {INFERENCE_PATH}"); sys.stdout.flush()"
idata = az.from_netcdf(INFERENCE_PATH)
print(f"Groups available: {list(idata.groups())}"); sys.stdout.flush()"

# Extract posterior samples
print("\nExtracting posterior samples..."); sys.stdout.flush()"
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values.reshape(-1, J)

n_samples = len(mu_samples)
print(f"Total posterior samples: {n_samples}"); sys.stdout.flush()"
print(f"  mu: mean={mu_samples.mean():.2f}, sd={mu_samples.std():.2f}"); sys.stdout.flush()"
print(f"  tau: mean={tau_samples.mean():.2f}, sd={tau_samples.std():.2f}"); sys.stdout.flush()"

# =============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# =============================================================================
print("\n[2/6] Generating posterior predictive samples..."); sys.stdout.flush()"

# Generate y_rep from posterior predictive distribution
# For each posterior sample (theta, sigma), draw y_rep ~ Normal(theta, sigma)
n_rep = n_samples  # Use all posterior samples
y_rep = np.zeros((n_rep, J))

print(f"Generating {n_rep} replicated datasets..."); sys.stdout.flush()"
for i in range(n_rep):
    for j in range(J):
        y_rep[i, j] = np.random.normal(theta_samples[i, j], sigma_obs[j])

print("Posterior predictive samples generated successfully"); sys.stdout.flush()"
print(f"Shape: {y_rep.shape} (samples x schools)"); sys.stdout.flush()"

# Add to InferenceData for ArviZ functions
print("\nAdding posterior predictive to InferenceData..."); sys.stdout.flush()"
if 'posterior_predictive' not in idata.groups():
    # Create xarray Dataset for posterior predictive
    pp_dataset = xr.Dataset({
        'y': xr.DataArray(
            y_rep.reshape(4, n_samples//4, J),
            dims=['chain', 'draw', 'y_dim_0'],
            coords={
                'chain': idata.posterior.chain,
                'draw': idata.posterior.draw,
                'y_dim_0': range(J)
            }
        )
    })
    idata.add_groups(posterior_predictive=pp_dataset)
print("Posterior predictive added to InferenceData"); sys.stdout.flush()"

