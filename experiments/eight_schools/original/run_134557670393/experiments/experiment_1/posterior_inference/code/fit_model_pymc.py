"""
Fit Bayesian Hierarchical Meta-Analysis to Real Data using PyMC

CmdStanPy unavailable, using PyMC as fallback.
Using non-centered parameterization (expected tau near 0 from EDA).
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from pathlib import Path

# Set random seed for reproducibility
SEED = 12345
np.random.seed(SEED)

# Paths
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
CODE_DIR = OUTPUT_DIR / "code"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOT_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BAYESIAN HIERARCHICAL META-ANALYSIS - POSTERIOR INFERENCE (PyMC)")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(DATA_PATH)
J = len(df)
y = df['y'].values
sigma = df['sigma'].values

print(f"   Loaded {J} studies")
print(f"   y: {y}")
print(f"   sigma: {sigma}")

# Sampling configuration
SAMPLING_CONFIG = {
    'draws': 1000,
    'tune': 1000,
    'chains': 4,
    'target_accept': 0.95,
    'random_seed': SEED,
    'cores': 4
}

print(f"\n2. Sampling configuration:")
for key, val in SAMPLING_CONFIG.items():
    print(f"   {key}: {val}")

# Build model with non-centered parameterization
print("\n3. Building PyMC model (non-centered parameterization)...")

with pm.Model() as model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=50)
    tau = pm.HalfCauchy('tau', beta=5)

    # Non-centered parameterization
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Posterior predictive
    y_rep = pm.Normal('y_rep', mu=theta, sigma=sigma, shape=J)

    # Shrinkage (computed in post-processing)

print("   Model built successfully")
print(f"   Free parameters: mu, tau, theta_raw[{J}]")

# Fit the model
print("\n4. Fitting model with NUTS sampler...")
start_time = time.time()

with model:
    trace = pm.sample(
        draws=SAMPLING_CONFIG['draws'],
        tune=SAMPLING_CONFIG['tune'],
        chains=SAMPLING_CONFIG['chains'],
        target_accept=SAMPLING_CONFIG['target_accept'],
        random_seed=SAMPLING_CONFIG['random_seed'],
        cores=SAMPLING_CONFIG['cores'],
        return_inferencedata=True,
        idata_kwargs={
            'log_likelihood': True
        }
    )

runtime = time.time() - start_time
print(f"\n   Sampling completed in {runtime:.1f} seconds")

# Add posterior predictive
print("\n5. Generating posterior predictive samples...")
with model:
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=SEED)

# Convert to ArviZ InferenceData (already done by PyMC)
print("\n6. Verifying InferenceData structure...")
print(f"   InferenceData groups: {list(trace.groups())}")

if 'log_likelihood' in trace.groups():
    print("   SUCCESS: log_likelihood group present for LOO-CV")
else:
    print("   WARNING: log_likelihood group missing!")

# Save InferenceData
print("\n7. Saving InferenceData...")
idata_path = DIAG_DIR / "posterior_inference.netcdf"
trace.to_netcdf(idata_path)
print(f"   Saved to: {idata_path}")

# Compute convergence diagnostics
print("\n8. Computing convergence diagnostics...")
summary = az.summary(trace, var_names=['mu', 'tau', 'theta'])
print("\n" + "="*80)
print("CONVERGENCE SUMMARY")
print("="*80)
print(summary)

# Save summary
summary_path = DIAG_DIR / "convergence_summary.csv"
summary.to_csv(summary_path)
print(f"\n   Saved summary to: {summary_path}")

# Check convergence criteria
print("\n9. Checking convergence criteria...")

# Get divergences from trace
divergences = trace.sample_stats.diverging.sum().item()
total_samples = SAMPLING_CONFIG['chains'] * SAMPLING_CONFIG['draws']

convergence_checks = {
    'max_rhat': float(summary['r_hat'].max()),
    'min_ess_bulk': float(summary['ess_bulk'].min()),
    'min_ess_tail': float(summary['ess_tail'].min()),
    'all_rhat_ok': bool((summary['r_hat'] < 1.01).all()),
    'all_ess_bulk_ok': bool((summary['ess_bulk'] > 400).all()),
    'all_ess_tail_ok': bool((summary['ess_tail'] > 400).all()),
    'divergences': int(divergences),
    'divergence_rate': float(divergences / total_samples),
    'runtime_seconds': float(runtime),
    'ess_per_second_mu': float(summary.loc['mu', 'ess_bulk'] / runtime),
    'parameterization': 'non-centered',
    'backend': 'PyMC'
}

print(f"\n   Max R-hat: {convergence_checks['max_rhat']:.4f} (target: < 1.01)")
print(f"   Min ESS bulk: {convergence_checks['min_ess_bulk']:.0f} (target: > 400)")
print(f"   Min ESS tail: {convergence_checks['min_ess_tail']:.0f} (target: > 400)")
print(f"   Divergences: {convergence_checks['divergences']} ({convergence_checks['divergence_rate']*100:.2f}%)")
print(f"   ESS/sec (mu): {convergence_checks['ess_per_second_mu']:.1f}")

# Overall convergence assessment
if (convergence_checks['all_rhat_ok'] and
    convergence_checks['all_ess_bulk_ok'] and
    convergence_checks['all_ess_tail_ok'] and
    convergence_checks['divergence_rate'] < 0.001):
    print("\n   CONVERGENCE STATUS: SUCCESS")
    convergence_checks['status'] = 'SUCCESS'
else:
    print("\n   CONVERGENCE STATUS: NEEDS ATTENTION")
    convergence_checks['status'] = 'NEEDS_ATTENTION'

    # Details on what failed
    if not convergence_checks['all_rhat_ok']:
        print(f"   - Some R-hat > 1.01")
    if not convergence_checks['all_ess_bulk_ok']:
        print(f"   - Some ESS bulk < 400")
    if not convergence_checks['all_ess_tail_ok']:
        print(f"   - Some ESS tail < 400")
    if convergence_checks['divergence_rate'] >= 0.001:
        print(f"   - Divergence rate >= 0.1%")

# Save convergence checks
checks_path = DIAG_DIR / "convergence_checks.json"
with open(checks_path, 'w') as f:
    json.dump(convergence_checks, f, indent=2)
print(f"\n   Saved convergence checks to: {checks_path}")

# Compute posterior quantities of interest
print("\n10. Computing posterior quantities...")
posterior = trace.posterior

mu_samples = posterior['mu'].values.flatten()
tau_samples = posterior['tau'].values.flatten()
theta_samples = posterior['theta'].values  # shape: (chains, draws, studies)

quantities = {
    'mu_mean': float(np.mean(mu_samples)),
    'mu_median': float(np.median(mu_samples)),
    'mu_sd': float(np.std(mu_samples)),
    'mu_ci_lower': float(np.percentile(mu_samples, 2.5)),
    'mu_ci_upper': float(np.percentile(mu_samples, 97.5)),
    'tau_mean': float(np.mean(tau_samples)),
    'tau_median': float(np.median(tau_samples)),
    'tau_sd': float(np.std(tau_samples)),
    'tau_ci_lower': float(np.percentile(tau_samples, 2.5)),
    'tau_ci_upper': float(np.percentile(tau_samples, 97.5)),
    'prob_mu_positive': float(np.mean(mu_samples > 0)),
    'prob_mu_gt_5': float(np.mean(mu_samples > 5)),
    'prob_mu_gt_10': float(np.mean(mu_samples > 10)),
    'prob_tau_lt_1': float(np.mean(tau_samples < 1)),
    'prob_tau_lt_5': float(np.mean(tau_samples < 5)),
    'prob_tau_lt_10': float(np.mean(tau_samples < 10))
}

print(f"\n   mu: {quantities['mu_mean']:.2f} +/- {quantities['mu_sd']:.2f}")
print(f"   mu 95% CI: [{quantities['mu_ci_lower']:.2f}, {quantities['mu_ci_upper']:.2f}]")
print(f"   P(mu > 0): {quantities['prob_mu_positive']:.3f}")
print(f"   P(mu > 5): {quantities['prob_mu_gt_5']:.3f}")
print(f"\n   tau: {quantities['tau_mean']:.2f} +/- {quantities['tau_sd']:.2f}")
print(f"   tau 95% CI: [{quantities['tau_ci_lower']:.2f}, {quantities['tau_ci_upper']:.2f}]")
print(f"   tau median: {quantities['tau_median']:.2f}")
print(f"   P(tau < 1): {quantities['prob_tau_lt_1']:.3f}")
print(f"   P(tau < 5): {quantities['prob_tau_lt_5']:.3f}")

# Save quantities
quantities_path = DIAG_DIR / "posterior_quantities.json"
with open(quantities_path, 'w') as f:
    json.dump(quantities, f, indent=2)
print(f"\n   Saved to: {quantities_path}")

# Compute shrinkage for each study
print("\n11. Computing shrinkage statistics...")
shrinkage_stats = []
for j in range(J):
    theta_j = theta_samples[:, :, j].flatten()

    # Shrinkage: how much theta_j has moved from y[j] toward mu
    shrinkage = (y[j] - np.mean(theta_j)) / (y[j] - quantities['mu_mean']) if y[j] != quantities['mu_mean'] else 0

    shrinkage_stats.append({
        'study': j + 1,
        'y_obs': float(y[j]),
        'sigma_obs': float(sigma[j]),
        'theta_mean': float(np.mean(theta_j)),
        'theta_sd': float(np.std(theta_j)),
        'theta_ci_lower': float(np.percentile(theta_j, 2.5)),
        'theta_ci_upper': float(np.percentile(theta_j, 97.5)),
        'shrinkage': float(shrinkage)
    })

shrinkage_df = pd.DataFrame(shrinkage_stats)
print("\n" + "="*80)
print("SHRINKAGE STATISTICS")
print("="*80)
print(shrinkage_df.to_string(index=False))

shrinkage_path = DIAG_DIR / "shrinkage_stats.csv"
shrinkage_df.to_csv(shrinkage_path, index=False)
print(f"\n   Saved to: {shrinkage_path}")

# Energy diagnostics
print("\n12. Computing energy diagnostics...")
energy_stats = az.bfmi(trace)
print(f"   E-BFMI: {energy_stats}")
if isinstance(energy_stats, np.ndarray):
    energy_check = (energy_stats > 0.2).all()
    print(f"   All chains E-BFMI > 0.2: {energy_check}")
else:
    energy_check = energy_stats > 0.2
    print(f"   E-BFMI > 0.2: {energy_check}")

print("\n" + "="*80)
print("FITTING COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - InferenceData: {idata_path}")
print(f"  - Convergence summary: {summary_path}")
print(f"  - Posterior quantities: {quantities_path}")
print(f"  - Shrinkage statistics: {shrinkage_path}")
print(f"\nNext steps:")
print(f"  1. Create diagnostic plots")
print(f"  2. Create posterior visualization plots")
print(f"  3. Write inference summary report")
