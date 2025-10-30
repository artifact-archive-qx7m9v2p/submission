"""
Fit Bayesian Negative Binomial Model to Real Data (PyMC)
==========================================================

Model: C[i] ~ NegativeBinomial(mu[i], phi)
       log(mu[i]) = beta_0 + beta_1 * year[i]

Priors:
  beta_0 ~ Normal(4.3, 1.0)
  beta_1 ~ Normal(0.85, 0.5)
  phi ~ Exponential(0.667)
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

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1")
DATA_PATH = Path("/workspace/data/data_analyst_1.json")
OUTPUT_DIR = BASE_DIR / "posterior_inference"
CODE_DIR = OUTPUT_DIR / "code"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

print("=" * 80)
print("BAYESIAN MODEL FITTING: Negative Binomial Regression (PyMC)")
print("=" * 80)

# ============================================================================
# STEP 1: Load Real Data
# ============================================================================
print("\n[1/6] Loading real data...")

with open(DATA_PATH, 'r') as f:
    data = json.load(f)

n = data['n']
C = np.array(data['C'])
year = np.array(data['year'])

print(f"  - Loaded {n} observations")
print(f"  - C range: [{min(C)}, {max(C)}]")
print(f"  - Year range: [{min(year):.2f}, {max(year):.2f}]")
print(f"  - Mean count: {np.mean(C):.1f}")
print(f"  - Variance/Mean ratio: {np.var(C)/np.mean(C):.2f}")

# ============================================================================
# STEP 2: Build PyMC Model
# ============================================================================
print("\n[2/6] Building PyMC model...")

with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=4.3, sigma=1.0)
    beta_1 = pm.Normal('beta_1', mu=0.85, sigma=0.5)
    phi = pm.Exponential('phi', lam=0.667)

    # Linear predictor
    log_mu = beta_0 + beta_1 * year
    mu = pm.math.exp(log_mu)

    # Likelihood (NegativeBinomial in PyMC parameterized by mu and alpha)
    # PyMC uses alpha = 1/phi (shape parameter)
    # Stan's neg_binomial_2(mu, phi) corresponds to PyMC's NegativeBinomial(mu, alpha)
    # where variance = mu + mu^2/phi = mu + mu^2*alpha
    alpha = phi  # PyMC alpha = Stan phi
    C_obs = pm.NegativeBinomial('C_obs', mu=mu, alpha=alpha, observed=C)

    # Posterior predictive
    C_rep = pm.NegativeBinomial('C_rep', mu=mu, alpha=alpha, shape=n)

print("  - Model built successfully")
print(f"  - Model structure:")
print(f"    - beta_0: Normal(4.3, 1.0)")
print(f"    - beta_1: Normal(0.85, 0.5)")
print(f"    - phi: Exponential(0.667)")

# ============================================================================
# STEP 3: Initial Probe (Quick Check)
# ============================================================================
print("\n[3/6] Running initial probe (4 chains, 200 iterations)...")

with model:
    probe_trace = pm.sample(
        draws=100,
        tune=100,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True},
        progressbar=False
    )

# Quick convergence check
probe_summary = az.summary(probe_trace, var_names=['beta_0', 'beta_1', 'phi'])
max_rhat = probe_summary['r_hat'].max()
min_ess = probe_summary['ess_bulk'].min()

print(f"  - Probe complete")
print(f"  - Max R-hat: {max_rhat:.4f}")
print(f"  - Min ESS bulk: {min_ess:.0f}")

# Check for divergences in probe
if hasattr(probe_trace.sample_stats, 'diverging'):
    n_divergences_probe = probe_trace.sample_stats.diverging.sum().item()
    total_draws_probe = probe_trace.sample_stats.diverging.size
    divergence_rate_probe = n_divergences_probe / total_draws_probe
    print(f"  - Divergences: {n_divergences_probe} ({divergence_rate_probe*100:.2f}%)")
else:
    n_divergences_probe = 0
    divergence_rate_probe = 0.0

# Adjust target_accept if needed
if divergence_rate_probe > 0.005:
    print("  WARNING: High divergence rate detected, will increase target_accept")
    target_accept = 0.99
else:
    target_accept = 0.95

# ============================================================================
# STEP 4: Main Sampling
# ============================================================================
print(f"\n[4/6] Running main sampling (4 chains, 2000 iterations, target_accept={target_accept})...")

try:
    with model:
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=target_accept,
            random_seed=42,
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True},
            progressbar=False
        )

    print("  - Sampling complete!")

    # Check diagnostics
    if hasattr(trace.sample_stats, 'diverging'):
        n_divergences = trace.sample_stats.diverging.sum().item()
        total_draws = trace.sample_stats.diverging.size
        divergence_rate = n_divergences / total_draws
    else:
        n_divergences = 0
        divergence_rate = 0.0

    print(f"\n  Sampling diagnostics:")
    print(f"  - Total draws: {total_draws}")
    print(f"  - Divergences: {n_divergences} ({divergence_rate*100:.2f}%)")

    if divergence_rate > 0.005:
        print("  WARNING: Divergence rate exceeds 0.5% threshold!")
    else:
        print("  - Divergence check: PASSED")

except Exception as e:
    print(f"  CRITICAL ERROR: {e}")
    print("  Saving error log...")
    with open(DIAGNOSTICS_DIR / "sampling_error.log", 'w') as f:
        f.write(f"Sampling failed with error:\n{str(e)}\n")
    raise

# ============================================================================
# STEP 5: Convergence Diagnostics
# ============================================================================
print("\n[5/6] Checking convergence...")

# Get summary statistics
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'phi'])
print("\nParameter summary:")
print(summary[['mean', 'sd', 'ess_bulk', 'ess_tail', 'r_hat']])

# Check convergence criteria
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()
min_ess = min(min_ess_bulk, min_ess_tail)

print(f"\n  Convergence criteria:")
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
    print("\n  CONVERGENCE CHECK: PASSED")
else:
    print("\n  CONVERGENCE CHECK: FAILED")
    if max_rhat >= 1.01:
        print(f"    - R-hat too high: {max_rhat:.4f}")
    if min_ess <= 400:
        print(f"    - ESS too low: {min_ess:.0f}")
    if divergence_rate >= 0.005:
        print(f"    - Too many divergences: {divergence_rate*100:.2f}%")

# ============================================================================
# STEP 6: Save InferenceData
# ============================================================================
print("\n[6/6] Saving InferenceData...")

# Sample posterior predictive
with model:
    trace = pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=42, progressbar=False)

# Save InferenceData
idata_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
trace.to_netcdf(idata_path)
print(f"  - Saved InferenceData to: {idata_path}")

# Save summary statistics
summary_path = DIAGNOSTICS_DIR / "convergence_summary.txt"
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CONVERGENCE SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sampling Configuration:\n")
    f.write(f"  - PPL: PyMC v{pm.__version__}\n")
    f.write(f"  - Chains: 4\n")
    f.write(f"  - Iterations per chain: 2000 (1000 warmup, 1000 sampling)\n")
    f.write(f"  - target_accept: {target_accept}\n\n")
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

print(f"  - Saved convergence summary to: {summary_path}")

# ============================================================================
# STEP 7: Generate Diagnostic Plots
# ============================================================================
print("\n[7/7] Generating diagnostic plots...")

# Plot 1: Trace plots
print("  - Creating trace plots...")
axes = az.plot_trace(trace, var_names=['beta_0', 'beta_1', 'phi'], figsize=(14, 10))
plt.suptitle("Trace Plots: Convergence Diagnostics", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Rank plots (sensitive to non-convergence)
print("  - Creating rank plots...")
az.plot_rank(trace, var_names=['beta_0', 'beta_1', 'phi'])
plt.suptitle("Rank Plots: Chain Mixing Diagnostics", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Posterior distributions
print("  - Creating posterior plots...")
az.plot_posterior(trace, var_names=['beta_0', 'beta_1', 'phi'], figsize=(15, 4))
plt.suptitle("Posterior Distributions", fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_distributions.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Pair plot (parameter correlations)
print("  - Creating pair plot...")
az.plot_pair(
    trace,
    var_names=['beta_0', 'beta_1', 'phi'],
    kind='hexbin',
    marginals=True,
    figsize=(10, 10)
)
plt.suptitle("Posterior Correlations", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pair_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: ESS plots
print("  - Creating ESS diagnostic plot...")
az.plot_ess(trace, var_names=['beta_0', 'beta_1', 'phi'], kind='evolution')
plt.suptitle("Effective Sample Size Evolution", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ess_evolution.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Energy plot (HMC-specific diagnostic)
print("  - Creating energy plot...")
az.plot_energy(trace)
plt.suptitle("Energy Diagnostic (BFMI)", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "energy_plot.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  All plots saved to: {PLOTS_DIR}/")

# ============================================================================
# STEP 8: Parameter Estimates and Comparison to EDA
# ============================================================================
print("\n" + "=" * 80)
print("POSTERIOR ESTIMATES vs EDA EXPECTATIONS")
print("=" * 80)

# EDA expected values
eda_expected = {
    'beta_0': (4.3, 0.15),  # mean, sd
    'beta_1': (0.85, 0.10),
    'phi': (1.5, 0.5)
}

print("\nParameter Comparisons:")
for param in ['beta_0', 'beta_1', 'phi']:
    post_mean = summary.loc[param, 'mean']
    post_sd = summary.loc[param, 'sd']
    post_5 = summary.loc[param, 'hdi_5%']
    post_95 = summary.loc[param, 'hdi_95%']

    eda_mean, eda_sd = eda_expected[param]

    # Calculate z-score (how many SDs from EDA mean)
    z_score = (post_mean - eda_mean) / eda_sd

    print(f"\n{param}:")
    print(f"  Posterior: {post_mean:.3f} ± {post_sd:.3f}")
    print(f"  90% HDI: [{post_5:.3f}, {post_95:.3f}]")
    print(f"  EDA Expected: {eda_mean:.3f} ± {eda_sd:.3f}")
    print(f"  Deviation: {z_score:.2f} SD from EDA")
    if abs(z_score) < 2:
        print(f"  Status: REASONABLE (within 2 SD)")
    else:
        print(f"  Status: WARNING (beyond 2 SD)")

# ============================================================================
# Final Status Report
# ============================================================================
print("\n" + "=" * 80)
print("FINAL STATUS")
print("=" * 80)

# Check if all parameters are reasonable
all_reasonable = all(
    abs((summary.loc[p, 'mean'] - eda_expected[p][0]) / eda_expected[p][1]) < 2
    for p in ['beta_0', 'beta_1', 'phi']
)

success = convergence_passed and all_reasonable

if success:
    print("\nRESULT: SUCCESS")
    print("\nAll criteria met:")
    print(f"  - Convergence: PASSED (R-hat < 1.01, ESS > 400, divergences < 0.5%)")
    print(f"  - Parameter estimates: REASONABLE (within 2 SD of EDA)")
    print(f"  - InferenceData saved with log_likelihood for LOO-CV")
else:
    print("\nRESULT: FAILURE")
    if not convergence_passed:
        print("\n  - Convergence criteria NOT met")
    if not all_reasonable:
        print("\n  - Parameter estimates UNREASONABLE (>2 SD from EDA)")

print("\n" + "=" * 80)
print("\nOutputs saved to:")
print(f"  - Code: {CODE_DIR}/fit_model.py")
print(f"  - Diagnostics: {DIAGNOSTICS_DIR}/")
print(f"  - Plots: {PLOTS_DIR}/")
print("=" * 80)
