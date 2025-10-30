"""
Fit Bayesian Hierarchical Meta-Analysis to Real Data

Uses CmdStanPy with adaptive sampling strategy.
Starting with non-centered parameterization (expected tau near 0 from EDA).
"""

import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from pathlib import Path

# Set random seed for reproducibility
SEED = 12345

# Paths
DATA_PATH = "/workspace/data/data.csv"
STAN_MODEL_NCP = "/workspace/experiments/experiment_1/simulation_based_validation/code/hierarchical_meta_analysis_ncp.stan"
STAN_MODEL_CP = "/workspace/experiments/experiment_1/simulation_based_validation/code/hierarchical_meta_analysis.stan"
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
CODE_DIR = OUTPUT_DIR / "code"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOT_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BAYESIAN HIERARCHICAL META-ANALYSIS - POSTERIOR INFERENCE")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded {len(df)} studies")
print(f"   y: {df['y'].values}")
print(f"   sigma: {df['sigma'].values}")

# Prepare Stan data
stan_data = {
    'J': len(df),
    'y': df['y'].values.tolist(),
    'sigma': df['sigma'].values.tolist()
}

# Initial sampling configuration (conservative start)
SAMPLING_CONFIG = {
    'chains': 4,
    'iter_warmup': 1000,
    'iter_sampling': 1000,
    'adapt_delta': 0.95,
    'max_treedepth': 12,
    'seed': SEED,
    'show_console': True
}

print(f"\n2. Sampling configuration:")
for key, val in SAMPLING_CONFIG.items():
    print(f"   {key}: {val}")

# Try non-centered parameterization first (expected tau near 0)
print("\n3. Compiling Stan model (non-centered parameterization)...")
try:
    model = CmdStanModel(stan_file=STAN_MODEL_NCP)
    print("   Model compiled successfully")
    parameterization = "non-centered"
except Exception as e:
    print(f"   Non-centered compilation failed: {e}")
    print("   Falling back to centered parameterization...")
    model = CmdStanModel(stan_file=STAN_MODEL_CP)
    parameterization = "centered"

# Fit the model
print(f"\n4. Fitting model ({parameterization} parameterization)...")
start_time = time.time()

try:
    fit = model.sample(
        data=stan_data,
        chains=SAMPLING_CONFIG['chains'],
        iter_warmup=SAMPLING_CONFIG['iter_warmup'],
        iter_sampling=SAMPLING_CONFIG['iter_sampling'],
        adapt_delta=SAMPLING_CONFIG['adapt_delta'],
        max_treedepth=SAMPLING_CONFIG['max_treedepth'],
        seed=SAMPLING_CONFIG['seed'],
        show_console=SAMPLING_CONFIG['show_console']
    )

    runtime = time.time() - start_time
    print(f"\n   Sampling completed in {runtime:.1f} seconds")

    # Check for divergences
    divergences = fit.num_unconstrained_params
    total_samples = SAMPLING_CONFIG['chains'] * SAMPLING_CONFIG['iter_sampling']
    divergence_rate = fit.divergences / total_samples if hasattr(fit, 'divergences') else 0

    print(f"\n5. Initial diagnostics:")
    print(f"   Divergences: {fit.divergences if hasattr(fit, 'divergences') else 0}")

except Exception as e:
    print(f"\n   ERROR: Sampling failed: {e}")
    print("   Attempting with increased adapt_delta=0.99...")

    fit = model.sample(
        data=stan_data,
        chains=SAMPLING_CONFIG['chains'],
        iter_warmup=SAMPLING_CONFIG['iter_warmup'],
        iter_sampling=SAMPLING_CONFIG['iter_sampling'],
        adapt_delta=0.99,
        max_treedepth=SAMPLING_CONFIG['max_treedepth'],
        seed=SAMPLING_CONFIG['seed'],
        show_console=True
    )

    runtime = time.time() - start_time
    print(f"\n   Sampling completed with adapt_delta=0.99 in {runtime:.1f} seconds")

# Convert to ArviZ InferenceData with log_likelihood
print("\n6. Converting to ArviZ InferenceData...")

# For non-centered, we need to exclude theta_raw from main posterior
if parameterization == "non-centered":
    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive=['y_rep'],
        log_likelihood='log_lik',
        coords={'study': np.arange(1, stan_data['J'] + 1)},
        dims={
            'theta': ['study'],
            'theta_raw': ['study'],
            'log_lik': ['study'],
            'y_rep': ['study'],
            'shrinkage': ['study']
        }
    )
else:
    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive=['y_rep'],
        log_likelihood='log_lik',
        coords={'study': np.arange(1, stan_data['J'] + 1)},
        dims={
            'theta': ['study'],
            'log_lik': ['study'],
            'y_rep': ['study'],
            'shrinkage': ['study']
        }
    )

# Verify log_likelihood is present
print(f"   InferenceData groups: {list(idata.groups())}")
if 'log_likelihood' in idata.groups():
    print("   SUCCESS: log_likelihood group present for LOO-CV")
else:
    print("   WARNING: log_likelihood group missing!")

# Save InferenceData
print("\n7. Saving InferenceData...")
idata_path = DIAG_DIR / "posterior_inference.netcdf"
idata.to_netcdf(idata_path)
print(f"   Saved to: {idata_path}")

# Compute convergence diagnostics
print("\n8. Computing convergence diagnostics...")
summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])
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
convergence_checks = {
    'max_rhat': summary['r_hat'].max(),
    'min_ess_bulk': summary['ess_bulk'].min(),
    'min_ess_tail': summary['ess_tail'].min(),
    'all_rhat_ok': (summary['r_hat'] < 1.01).all(),
    'all_ess_bulk_ok': (summary['ess_bulk'] > 400).all(),
    'all_ess_tail_ok': (summary['ess_tail'] > 400).all(),
    'divergences': fit.divergences if hasattr(fit, 'divergences') else 0,
    'runtime_seconds': runtime,
    'ess_per_second_mu': summary.loc['mu', 'ess_bulk'] / runtime,
    'parameterization': parameterization
}

print(f"\n   Max R-hat: {convergence_checks['max_rhat']:.4f} (target: < 1.01)")
print(f"   Min ESS bulk: {convergence_checks['min_ess_bulk']:.0f} (target: > 400)")
print(f"   Min ESS tail: {convergence_checks['min_ess_tail']:.0f} (target: > 400)")
print(f"   Divergences: {convergence_checks['divergences']}")
print(f"   ESS/sec (mu): {convergence_checks['ess_per_second_mu']:.1f}")

# Overall convergence assessment
if (convergence_checks['all_rhat_ok'] and
    convergence_checks['all_ess_bulk_ok'] and
    convergence_checks['all_ess_tail_ok'] and
    convergence_checks['divergences'] < total_samples * 0.001):
    print("\n   CONVERGENCE STATUS: SUCCESS")
    convergence_checks['status'] = 'SUCCESS'
else:
    print("\n   CONVERGENCE STATUS: ISSUES DETECTED")
    convergence_checks['status'] = 'ISSUES'

# Save convergence checks
checks_path = DIAG_DIR / "convergence_checks.json"
with open(checks_path, 'w') as f:
    json.dump(convergence_checks, f, indent=2)
print(f"\n   Saved convergence checks to: {checks_path}")

# Compute posterior quantities of interest
print("\n10. Computing posterior quantities...")
posterior = idata.posterior

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
    'prob_tau_lt_1': float(np.mean(tau_samples < 1)),
    'prob_tau_lt_5': float(np.mean(tau_samples < 5))
}

print(f"\n   mu: {quantities['mu_mean']:.2f} +/- {quantities['mu_sd']:.2f}")
print(f"   mu 95% CI: [{quantities['mu_ci_lower']:.2f}, {quantities['mu_ci_upper']:.2f}]")
print(f"   P(mu > 0): {quantities['prob_mu_positive']:.3f}")
print(f"   P(mu > 5): {quantities['prob_mu_gt_5']:.3f}")
print(f"\n   tau: {quantities['tau_mean']:.2f} +/- {quantities['tau_sd']:.2f}")
print(f"   tau 95% CI: [{quantities['tau_ci_lower']:.2f}, {quantities['tau_ci_upper']:.2f}]")
print(f"   P(tau < 1): {quantities['prob_tau_lt_1']:.3f}")

# Save quantities
quantities_path = DIAG_DIR / "posterior_quantities.json"
with open(quantities_path, 'w') as f:
    json.dump(quantities, f, indent=2)
print(f"\n   Saved to: {quantities_path}")

# Compute shrinkage for each study
print("\n11. Computing shrinkage statistics...")
shrinkage_stats = []
for j in range(stan_data['J']):
    theta_j = theta_samples[:, :, j].flatten()
    shrinkage = (df.iloc[j]['y'] - np.mean(theta_j)) / (df.iloc[j]['y'] - quantities['mu_mean']) if df.iloc[j]['y'] != quantities['mu_mean'] else 0

    shrinkage_stats.append({
        'study': j + 1,
        'y_obs': df.iloc[j]['y'],
        'sigma_obs': df.iloc[j]['sigma'],
        'theta_mean': np.mean(theta_j),
        'theta_sd': np.std(theta_j),
        'theta_ci_lower': np.percentile(theta_j, 2.5),
        'theta_ci_upper': np.percentile(theta_j, 97.5),
        'shrinkage': shrinkage
    })

shrinkage_df = pd.DataFrame(shrinkage_stats)
print(shrinkage_df)

shrinkage_path = DIAG_DIR / "shrinkage_stats.csv"
shrinkage_df.to_csv(shrinkage_path, index=False)
print(f"\n   Saved to: {shrinkage_path}")

print("\n" + "="*80)
print("FITTING COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - InferenceData: {idata_path}")
print(f"  - Convergence summary: {summary_path}")
print(f"  - Posterior quantities: {quantities_path}")
print(f"  - Shrinkage statistics: {shrinkage_path}")
