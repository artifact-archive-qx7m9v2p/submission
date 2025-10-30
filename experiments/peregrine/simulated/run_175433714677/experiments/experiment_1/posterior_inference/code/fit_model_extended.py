"""
Fit Bayesian Negative Binomial Model to Real Data (Extended Sampling)
=======================================================================

Extended version with more iterations to achieve R-hat < 1.01
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("pymc").setLevel(logging.ERROR)

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1")
DATA_PATH = Path("/workspace/data/data_analyst_1.json")
OUTPUT_DIR = BASE_DIR / "posterior_inference"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

print("=" * 80)
print("BAYESIAN MODEL FITTING: Extended Sampling (4000 iterations)")
print("=" * 80)

# Load data
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

n = data['n']
C = np.array(data['C'])
year = np.array(data['year'])

print(f"\nData: {n} observations, C range [{min(C)}, {max(C)}]")

# Build model
with pm.Model() as model:
    beta_0 = pm.Normal('beta_0', mu=4.3, sigma=1.0)
    beta_1 = pm.Normal('beta_1', mu=0.85, sigma=0.5)
    phi = pm.Exponential('phi', lam=0.667)
    log_mu = beta_0 + beta_1 * year
    mu = pm.math.exp(log_mu)
    C_obs = pm.NegativeBinomial('C_obs', mu=mu, alpha=phi, observed=C)

print("\nRunning extended sampling: 4 chains, 4000 iterations (2000 warmup + 2000 sampling)...")

with model:
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True},
        progressbar=False
    )

print("Sampling complete!")

# Check diagnostics
if hasattr(trace.sample_stats, 'diverging'):
    n_divergences = trace.sample_stats.diverging.sum().item()
    total_draws = trace.sample_stats.diverging.size
    divergence_rate = n_divergences / total_draws
else:
    n_divergences = 0
    divergence_rate = 0.0

print(f"  Divergences: {n_divergences} ({divergence_rate*100:.2f}%)")

# Get summary
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'phi'])
print("\nParameter summary:")
print(summary[['mean', 'sd', 'ess_bulk', 'ess_tail', 'r_hat']])

max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()
min_ess = min(min_ess_bulk, min_ess_tail)

print(f"\nConvergence criteria:")
print(f"  - Max R-hat: {max_rhat:.4f} (target: <1.01)")
print(f"  - Min ESS bulk: {min_ess_bulk:.0f} (target: >400)")
print(f"  - Min ESS tail: {min_ess_tail:.0f} (target: >400)")
print(f"  - Divergences: {divergence_rate*100:.2f}% (target: <0.5%)")

convergence_passed = (
    max_rhat < 1.01 and
    min_ess > 400 and
    divergence_rate < 0.005
)

if convergence_passed:
    print("\nCONVERGENCE CHECK: PASSED")
else:
    print("\nCONVERGENCE CHECK: FAILED")

# Sample posterior predictive
with model:
    trace = pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=42, progressbar=False)

# Save
idata_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
trace.to_netcdf(idata_path)
print(f"\nSaved InferenceData to: {idata_path}")

# Update summary file
summary_path = DIAGNOSTICS_DIR / "convergence_summary.txt"
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CONVERGENCE SUMMARY (Extended Sampling)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sampling Configuration:\n")
    f.write(f"  - PPL: PyMC v{pm.__version__}\n")
    f.write(f"  - Chains: 4\n")
    f.write(f"  - Iterations per chain: 4000 (2000 warmup, 2000 sampling)\n")
    f.write(f"  - target_accept: 0.95\n\n")
    f.write(f"Convergence Diagnostics:\n")
    f.write(f"  - Max R-hat: {max_rhat:.4f} (target: <1.01) {'PASS' if max_rhat < 1.01 else 'FAIL'}\n")
    f.write(f"  - Min ESS bulk: {min_ess_bulk:.0f} (target: >400) {'PASS' if min_ess_bulk > 400 else 'FAIL'}\n")
    f.write(f"  - Min ESS tail: {min_ess_tail:.0f} (target: >400) {'PASS' if min_ess_tail > 400 else 'FAIL'}\n")
    f.write(f"  - Divergences: {n_divergences} ({divergence_rate*100:.2f}%) {'PASS' if divergence_rate < 0.005 else 'FAIL'}\n")
    f.write(f"\nOverall: {'PASSED' if convergence_passed else 'FAILED'}\n\n")
    f.write("=" * 80 + "\n")
    f.write("PARAMETER ESTIMATES\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary.to_string())

print(f"Saved convergence summary to: {summary_path}")

# EDA comparison
eda_expected = {
    'beta_0': (4.3, 0.15),
    'beta_1': (0.85, 0.10),
    'phi': (1.5, 0.5)
}

print("\n" + "=" * 80)
print("POSTERIOR vs EDA EXPECTATIONS")
print("=" * 80)

for param in ['beta_0', 'beta_1', 'phi']:
    post_mean = summary.loc[param, 'mean']
    post_sd = summary.loc[param, 'sd']
    eda_mean, eda_sd = eda_expected[param]
    z_score = (post_mean - eda_mean) / eda_sd

    print(f"\n{param}:")
    print(f"  Posterior: {post_mean:.3f} ± {post_sd:.3f}")
    print(f"  EDA Expected: {eda_mean:.3f} ± {eda_sd:.3f}")
    print(f"  Deviation: {z_score:.2f} SD")
    print(f"  Status: {'REASONABLE' if abs(z_score) < 2 else 'WARNING'}")

all_reasonable = all(
    abs((summary.loc[p, 'mean'] - eda_expected[p][0]) / eda_expected[p][1]) < 2
    for p in ['beta_0', 'beta_1', 'phi']
)

success = convergence_passed and all_reasonable

print("\n" + "=" * 80)
print("FINAL STATUS")
print("=" * 80)

if success:
    print("\nRESULT: SUCCESS")
    print("  - Convergence: PASSED")
    print("  - Parameter estimates: REASONABLE")
    print("  - InferenceData saved with log_likelihood")
else:
    print("\nRESULT: FAILURE")
    if not convergence_passed:
        print("  - Convergence criteria NOT met")
    if not all_reasonable:
        print("  - Parameter estimates UNREASONABLE")

print("=" * 80)
