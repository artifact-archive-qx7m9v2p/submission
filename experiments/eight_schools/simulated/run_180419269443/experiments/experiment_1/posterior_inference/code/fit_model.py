"""
Fit Hierarchical Normal Model to Meta-Analysis Data
Using CmdStanPy with adaptive sampling strategy
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

# Setup paths
EXPERIMENT_DIR = Path("/workspace/experiments/experiment_1")
POSTERIOR_DIR = EXPERIMENT_DIR / "posterior_inference"
CODE_DIR = POSTERIOR_DIR / "code"
DIAGNOSTICS_DIR = POSTERIOR_DIR / "diagnostics"
PLOTS_DIR = POSTERIOR_DIR / "plots"
DATA_FILE = Path("/workspace/data/data.csv")

# Create directories
for directory in [DIAGNOSTICS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('default')
sns.set_palette("colorblind")

print("=" * 80)
print("HIERARCHICAL NORMAL MODEL - POSTERIOR INFERENCE")
print("=" * 80)

# Load data
print("\n[1/8] Loading data...")
data = pd.read_csv(DATA_FILE)
print(f"Loaded {len(data)} studies")
print(data)

# Prepare data for Stan
stan_data = {
    'J': len(data),
    'y': data['y'].values,
    'sigma': data['sigma'].values
}

print(f"\nData summary:")
print(f"  Studies (J): {stan_data['J']}")
print(f"  y range: [{stan_data['y'].min():.2f}, {stan_data['y'].max():.2f}]")
print(f"  sigma range: [{stan_data['sigma'].min():.2f}, {stan_data['sigma'].max():.2f}]")

# Compile model
print("\n[2/8] Compiling Stan model...")
model_file = CODE_DIR / "hierarchical_model_inference.stan"
model = cmdstanpy.CmdStanModel(stan_file=str(model_file))
print(f"Model compiled successfully: {model.exe_file}")

# Phase 1: Initial probe (quick diagnostic run)
print("\n[3/8] Phase 1: Initial probe (4 chains × 200 iterations)...")
print("Purpose: Quick assessment of model behavior and initialization")

probe_fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=100,
    iter_sampling=100,
    adapt_delta=0.8,
    max_treedepth=10,
    show_progress=True,
    show_console=False
)

# Check probe diagnostics
probe_summary = probe_fit.summary()
print("\nProbe diagnostics:")
print(f"  Divergences: {probe_fit.divergences}")
print(f"  Max treedepth: {probe_fit.max_treedepths}")

# Check convergence on key parameters
key_params = ['mu', 'tau']
probe_rhat = probe_summary[probe_summary.index.isin(key_params)]['R_hat']
probe_ess = probe_summary[probe_summary.index.isin(key_params)]['N_Eff']

print(f"\nKey parameter diagnostics (probe):")
for param in key_params:
    if param in probe_summary.index:
        rhat = probe_summary.loc[param, 'R_hat']
        ess = probe_summary.loc[param, 'N_Eff']
        print(f"  {param}: R_hat={rhat:.4f}, ESS={ess:.0f}")

# Decide on main sampling strategy
if probe_fit.divergences > 0:
    print("\n  WARNING: Divergences detected in probe. Will increase adapt_delta for main run.")
    main_adapt_delta = 0.95
else:
    print("\n  SUCCESS: No divergences in probe. Using standard adapt_delta.")
    main_adapt_delta = 0.8

if max(probe_rhat) > 1.05:
    print("  WARNING: High R_hat in probe. May need longer chains.")

# Phase 2: Main sampling
print("\n[4/8] Phase 2: Main sampling (4 chains × 2000 iterations)...")
print(f"Settings: adapt_delta={main_adapt_delta}, max_treedepth=10")

main_fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=main_adapt_delta,
    max_treedepth=10,
    show_progress=True,
    show_console=False,
    seed=12345
)

# Save raw samples
print("\n[5/8] Saving samples and diagnostics...")
main_fit.save_csvfiles(dir=str(DIAGNOSTICS_DIR))

# Check main diagnostics
main_summary = main_fit.summary()
print("\nMain sampling diagnostics:")
print(f"  Divergences: {main_fit.divergences}")
print(f"  Max treedepth hits: {main_fit.max_treedepths}")

# Convert to ArviZ InferenceData with log_likelihood
print("\n[6/8] Converting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    main_fit,
    posterior_predictive=['y_pred'],
    log_likelihood='log_lik',
    observed_data={'y': stan_data['y']},
    coords={'study': data['study'].values},
    dims={
        'y': ['study'],
        'theta': ['study'],
        'theta_raw': ['study'],
        'y_pred': ['study'],
        'log_lik': ['study']
    }
)

# Save InferenceData
idata_file = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
idata.to_netcdf(idata_file)
print(f"Saved InferenceData to: {idata_file}")

# Compute summary statistics
print("\n[7/8] Computing posterior summaries...")
summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])
summary_file = DIAGNOSTICS_DIR / "posterior_summary.csv"
summary.to_csv(summary_file)
print(f"Saved summary to: {summary_file}")

print("\nPosterior Summary:")
print(summary)

# Compute LOO
print("\n[8/8] Computing LOO-CV...")
loo = az.loo(idata, pointwise=True)
print("\nLOO results:")
print(loo)

# Save LOO results
loo_file = DIAGNOSTICS_DIR / "loo_results.json"
loo_dict = {
    'elpd_loo': float(loo.elpd_loo),
    'se': float(loo.se),
    'p_loo': float(loo.p_loo),
    'n_samples': int(loo.n_samples),
    'n_data_points': int(loo.n_data_points),
    'warning': loo.warning if hasattr(loo, 'warning') else False,
    'scale': loo.scale
}
with open(loo_file, 'w') as f:
    json.dump(loo_dict, f, indent=2)

# Check Pareto k values
pareto_k = loo.pareto_k
print("\nPareto k diagnostics:")
print(f"  Max k: {pareto_k.max():.3f}")
print(f"  Studies with k > 0.5: {(pareto_k > 0.5).sum()}")
print(f"  Studies with k > 0.7: {(pareto_k > 0.7).sum()}")

print("\nPareto k by study:")
for i, k in enumerate(pareto_k, 1):
    status = "GOOD" if k < 0.5 else ("OK" if k < 0.7 else "BAD")
    print(f"  Study {i}: k={k:.3f} [{status}]")

# Compute derived quantities
print("\n" + "=" * 80)
print("DERIVED QUANTITIES")
print("=" * 80)

# Extract posterior samples
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values.reshape(-1, stan_data['J'])

# I² statistic
mean_sigma_sq = np.mean(stan_data['sigma']**2)
I2_samples = tau_samples**2 / (tau_samples**2 + mean_sigma_sq)
I2_mean = I2_samples.mean()
I2_std = I2_samples.std()
I2_ci = np.percentile(I2_samples, [2.5, 97.5])

print(f"\nI² statistic (heterogeneity):")
print(f"  Mean: {I2_mean*100:.2f}%")
print(f"  SD: {I2_std*100:.2f}%")
print(f"  95% CI: [{I2_ci[0]*100:.2f}%, {I2_ci[1]*100:.2f}%]")

# Shrinkage factors
y_obs = stan_data['y']
theta_mean = theta_samples.mean(axis=0)
mu_mean = mu_samples.mean()

shrinkage = np.abs((theta_mean - y_obs) / (mu_mean - y_obs))
shrinkage = np.clip(shrinkage, 0, 1)  # Clip to [0, 1] for interpretation

print("\nShrinkage factors (|theta_i - y_i| / |mu - y_i|):")
for i in range(stan_data['J']):
    print(f"  Study {i+1}: {shrinkage[i]:.3f}")

# Convergence assessment
print("\n" + "=" * 80)
print("CONVERGENCE ASSESSMENT")
print("=" * 80)

# Extract R_hat and ESS
rhat_mu = summary.loc['mu', 'r_hat']
rhat_tau = summary.loc['tau', 'r_hat']
ess_mu = summary.loc['mu', 'ess_bulk']
ess_tau = summary.loc['tau', 'ess_bulk']

# Get theta R_hats
theta_rhats = summary[summary.index.str.startswith('theta[')]['r_hat']
max_theta_rhat = theta_rhats.max()
min_theta_ess = summary[summary.index.str.startswith('theta[')]['ess_bulk'].min()

print(f"\nConvergence metrics:")
print(f"  mu: R_hat={rhat_mu:.4f}, ESS_bulk={ess_mu:.0f}")
print(f"  tau: R_hat={rhat_tau:.4f}, ESS_bulk={ess_tau:.0f}")
print(f"  theta: max R_hat={max_theta_rhat:.4f}, min ESS={min_theta_ess:.0f}")

# Overall convergence decision
all_rhat_ok = (rhat_mu < 1.01) and (rhat_tau < 1.01) and (max_theta_rhat < 1.01)
ess_ok = (ess_mu > 400) and (ess_tau > 100) and (min_theta_ess > 100)
no_divergences = main_fit.divergences == 0
loo_ok = pareto_k.max() < 0.7

converged = all_rhat_ok and ess_ok and no_divergences

print(f"\nConvergence criteria:")
print(f"  All R_hat < 1.01: {all_rhat_ok} {'✓' if all_rhat_ok else '✗'}")
print(f"  ESS adequate: {ess_ok} {'✓' if ess_ok else '✗'}")
print(f"  No divergences: {no_divergences} {'✓' if no_divergences else '✗'}")
print(f"  LOO stable (k < 0.7): {loo_ok} {'✓' if loo_ok else '✗'}")

# Final decision
if converged and loo_ok:
    decision = "PASS"
    print(f"\n{'='*80}")
    print(f"OVERALL DECISION: {decision}")
    print(f"{'='*80}")
    print("Model has converged successfully with stable LOO diagnostics.")
elif converged and not loo_ok:
    decision = "MARGINAL"
    print(f"\n{'='*80}")
    print(f"OVERALL DECISION: {decision}")
    print(f"{'='*80}")
    print("Model converged but LOO diagnostics indicate potential issues.")
    print(f"Max Pareto k = {pareto_k.max():.3f} > 0.7")
else:
    decision = "FAIL"
    print(f"\n{'='*80}")
    print(f"OVERALL DECISION: {decision}")
    print(f"{'='*80}")
    print("Model did not meet convergence criteria.")
    if not all_rhat_ok:
        print("  - R_hat too high")
    if not ess_ok:
        print("  - ESS too low")
    if not no_divergences:
        print(f"  - {main_fit.divergences} divergences detected")

# Save decision and metrics
decision_dict = {
    'decision': decision,
    'convergence': {
        'all_rhat_ok': bool(all_rhat_ok),
        'ess_ok': bool(ess_ok),
        'no_divergences': bool(no_divergences),
        'loo_ok': bool(loo_ok),
        'divergences': int(main_fit.divergences),
        'max_treedepth_hits': int(main_fit.max_treedepths)
    },
    'key_metrics': {
        'mu': {
            'rhat': float(rhat_mu),
            'ess_bulk': float(ess_mu),
            'mean': float(summary.loc['mu', 'mean']),
            'sd': float(summary.loc['mu', 'sd']),
            'ci_2.5': float(summary.loc['mu', 'hdi_2.5%']),
            'ci_97.5': float(summary.loc['mu', 'hdi_97.5%'])
        },
        'tau': {
            'rhat': float(rhat_tau),
            'ess_bulk': float(ess_tau),
            'mean': float(summary.loc['tau', 'mean']),
            'sd': float(summary.loc['tau', 'sd']),
            'ci_2.5': float(summary.loc['tau', 'hdi_2.5%']),
            'ci_97.5': float(summary.loc['tau', 'hdi_97.5%'])
        },
        'I2': {
            'mean': float(I2_mean),
            'sd': float(I2_std),
            'ci_2.5': float(I2_ci[0]),
            'ci_97.5': float(I2_ci[1])
        }
    },
    'loo': {
        'elpd_loo': float(loo.elpd_loo),
        'se': float(loo.se),
        'p_loo': float(loo.p_loo),
        'max_pareto_k': float(pareto_k.max()),
        'n_high_pareto_k': int((pareto_k > 0.7).sum())
    }
}

decision_file = DIAGNOSTICS_DIR / "convergence_metrics.json"
with open(decision_file, 'w') as f:
    json.dump(decision_dict, f, indent=2)

print(f"\nResults saved to: {DIAGNOSTICS_DIR}")
print(f"InferenceData: {idata_file}")
print(f"Summary: {summary_file}")
print(f"Metrics: {decision_file}")

print("\n" + "=" * 80)
print("FITTING COMPLETE")
print("=" * 80)
