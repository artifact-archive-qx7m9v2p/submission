"""
Fit Beta-Binomial Model to Real Data using CmdStanPy
Experiment 1: Posterior Inference with HMC Sampling
"""

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import arviz as az
import json
from pathlib import Path

# Set random seed
np.random.seed(42)

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
STAN_MODEL_PATH = Path("/workspace/experiments/designer_1/stan_models/model_b_reparameterized.stan")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
RESULTS_DIR = OUTPUT_DIR / "results"

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(data)} groups")
print(f"Total trials: {data['n_trials'].sum()}")
print(f"Total successes: {data['r_successes'].sum()}")
print(f"Pooled rate: {data['r_successes'].sum() / data['n_trials'].sum():.4f}")

# Prepare data for Stan
stan_data = {
    'N': len(data),
    'n_trials': data['n_trials'].tolist(),
    'r_success': data['r_successes'].tolist()
}

# Save stan data for reproducibility
with open(RESULTS_DIR / 'stan_data.json', 'w') as f:
    json.dump(stan_data, f, indent=2)

print("\n" + "="*70)
print("COMPILING STAN MODEL")
print("="*70)

# Compile Stan model
try:
    model = CmdStanModel(stan_file=str(STAN_MODEL_PATH))
    print(f"Model compiled successfully")
    print(f"Model name: {model.name}")
except Exception as e:
    print(f"ERROR compiling Stan model: {e}")
    raise

print("\n" + "="*70)
print("FITTING MODEL - INITIAL PROBE (200 iterations)")
print("="*70)
print("Strategy: Short run to diagnose potential issues")
print("Chains: 4")
print("Warmup: 100")
print("Sampling: 100")
print("adapt_delta: 0.95")
print("max_treedepth: 12")
print("="*70 + "\n")

# Initial probe - quick run to check for issues
try:
    fit_probe = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=100,
        iter_sampling=100,
        adapt_delta=0.95,
        max_treedepth=12,
        seed=42,
        show_console=True
    )

    print("\n" + "="*70)
    print("PROBE RUN - SUMMARY")
    print("="*70)

    # Check probe diagnostics
    probe_summary = fit_probe.summary()
    print("\nKey parameters from probe:")
    print(probe_summary[probe_summary.index.isin(['mu', 'kappa', 'phi'])][['Mean', 'StdDev', 'N_Eff', 'R_hat']])

    # Check for divergences
    probe_divergences = fit_probe.divergences
    print(f"\nDivergent transitions: {probe_divergences}")

    # Check for max treedepth
    probe_max_td = fit_probe.num_max_treedepth
    print(f"Max treedepth hits: {probe_max_td}")

    # Check Rhat
    probe_max_rhat = probe_summary['R_hat'].max()
    print(f"Max R_hat: {probe_max_rhat:.4f}")

    # Assess probe results
    probe_issues = []
    if probe_divergences > 0:
        probe_issues.append(f"Divergences detected: {probe_divergences}")
    if probe_max_td > 4:  # >1% of 400 samples
        probe_issues.append(f"Max treedepth warnings: {probe_max_td}")
    if probe_max_rhat > 1.01:
        probe_issues.append(f"High R_hat: {probe_max_rhat:.4f}")

    if probe_issues:
        print("\n[WARNING] Probe detected issues:")
        for issue in probe_issues:
            print(f"  - {issue}")
        print("\nAdjusting sampling parameters for main run...")
        adapt_delta = 0.98
        max_treedepth = 15
        iter_warmup = 1500
        iter_sampling = 1000
    else:
        print("\n[SUCCESS] Probe run successful - no issues detected")
        print("Proceeding with standard sampling parameters...")
        adapt_delta = 0.95
        max_treedepth = 12
        iter_warmup = 1000
        iter_sampling = 1000

except Exception as e:
    print(f"\n[ERROR] Probe run failed: {e}")
    print("Attempting main run with conservative parameters...")
    adapt_delta = 0.98
    max_treedepth = 15
    iter_warmup = 1500
    iter_sampling = 1000

print("\n" + "="*70)
print("FITTING MODEL - MAIN RUN")
print("="*70)
print(f"Chains: 4")
print(f"Warmup: {iter_warmup}")
print(f"Sampling: {iter_sampling}")
print(f"adapt_delta: {adapt_delta}")
print(f"max_treedepth: {max_treedepth}")
print("="*70 + "\n")

# Main sampling run
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=iter_warmup,
    iter_sampling=iter_sampling,
    adapt_delta=adapt_delta,
    max_treedepth=max_treedepth,
    seed=42,
    show_console=True
)

print("\n" + "="*70)
print("SAMPLING COMPLETE - COMPUTING DIAGNOSTICS")
print("="*70)

# Save Stan output
print("\nSaving raw Stan output...")
fit.save_csvfiles(dir=str(DIAGNOSTICS_DIR))
print(f"Saved to {DIAGNOSTICS_DIR}")

# Get summary
summary = fit.summary()
print("\nPosterior Summary (Key Parameters):")
print(summary[summary.index.isin(['mu', 'kappa', 'phi'])][['Mean', 'StdDev', '5%', '50%', '95%', 'N_Eff', 'R_hat']])

# Save full summary
summary.to_csv(RESULTS_DIR / 'posterior_summary.csv')
print(f"\nFull summary saved to {RESULTS_DIR / 'posterior_summary.csv'}")

# Diagnostics
print("\n" + "="*70)
print("CONVERGENCE DIAGNOSTICS")
print("="*70)

divergences = fit.divergences
max_td = fit.num_max_treedepth
max_rhat = summary['R_hat'].max()
min_ess_bulk = summary['N_Eff'].min()

print(f"\nDivergent transitions: {divergences} (out of {4 * iter_sampling} total)")
print(f"Divergence rate: {100 * divergences / (4 * iter_sampling):.2f}%")
print(f"Max treedepth hits: {max_td}")
print(f"Max R_hat: {max_rhat:.4f}")
print(f"Min ESS (bulk): {min_ess_bulk:.0f}")

# Check convergence criteria
print("\n" + "="*70)
print("CONVERGENCE CRITERIA CHECK")
print("="*70)

criteria = {
    'All R_hat < 1.01': max_rhat < 1.01,
    'All ESS > 400': min_ess_bulk > 400,
    'Divergences < 1%': (divergences / (4 * iter_sampling)) < 0.01,
    'Max treedepth < 1%': (max_td / (4 * iter_sampling)) < 0.01
}

all_pass = all(criteria.values())

for criterion, passes in criteria.items():
    status = "PASS" if passes else "FAIL"
    print(f"{criterion}: {status}")

print("\n" + "="*70)
if all_pass:
    print("OVERALL: PASS - All convergence criteria met")
else:
    print("OVERALL: FAIL - Some convergence criteria not met")
print("="*70)

# Create ArviZ InferenceData
print("\n" + "="*70)
print("CREATING ARVIZ INFERENCEDATA")
print("="*70)

# Define coordinates and dimensions
coords = {
    'group': np.arange(1, len(data) + 1),
    'group_dim': np.arange(len(data))
}

dims = {
    'r_rep': ['group_dim'],
    'log_lik': ['group_dim'],
    'p_posterior_mean': ['group_dim'],
    'shrinkage_factor': ['group_dim'],
    'p_rep': ['group_dim']
}

print("Converting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive=['r_rep'],
    log_likelihood='log_lik',
    coords=coords,
    dims=dims
)

print(f"InferenceData created successfully")
print(f"Groups: {list(idata.groups())}")

# Save InferenceData
netcdf_path = DIAGNOSTICS_DIR / 'posterior_inference.netcdf'
idata.to_netcdf(netcdf_path)
print(f"\nInferenceData saved to {netcdf_path}")

# Save ArviZ summary
print("\nComputing ArviZ summary...")
az_summary = az.summary(idata, var_names=['mu', 'kappa', 'phi'])
print("\nArviZ Summary (Key Parameters):")
print(az_summary)

az_summary_full = az.summary(idata)
az_summary_full.to_csv(DIAGNOSTICS_DIR / 'arviz_summary.csv')
print(f"\nFull ArviZ summary saved to {DIAGNOSTICS_DIR / 'arviz_summary.csv'}")

# Extract posterior samples for analysis
print("\n" + "="*70)
print("EXTRACTING POSTERIOR SAMPLES")
print("="*70)

posterior = fit.stan_variables()

# Save key posterior samples
posterior_samples = pd.DataFrame({
    'mu': posterior['mu'],
    'kappa': posterior['kappa'],
    'phi': posterior['phi'],
    'alpha': posterior['alpha'],
    'beta': posterior['beta'],
    'var_p': posterior['var_p'],
    'icc': posterior['icc']
})

posterior_samples.to_csv(RESULTS_DIR / 'posterior_samples_scalar.csv', index=False)
print(f"Scalar posterior samples saved to {RESULTS_DIR / 'posterior_samples_scalar.csv'}")

# Save group-level posteriors
p_posterior_mean = posterior['p_posterior_mean']  # Shape: (n_samples, n_groups)
group_posteriors = pd.DataFrame(
    p_posterior_mean,
    columns=[f'group_{i}' for i in range(1, len(data) + 1)]
)
group_posteriors.to_csv(RESULTS_DIR / 'posterior_group_means.csv', index=False)
print(f"Group-level posteriors saved to {RESULTS_DIR / 'posterior_group_means.csv'}")

# Compute and save posterior summaries for groups
print("\n" + "="*70)
print("GROUP-LEVEL POSTERIOR SUMMARIES")
print("="*70)

group_summary = []
for i in range(len(data)):
    obs_rate = data.iloc[i]['r_successes'] / data.iloc[i]['n_trials']
    post_mean = p_posterior_mean[:, i].mean()
    post_sd = p_posterior_mean[:, i].std()
    post_q025 = np.percentile(p_posterior_mean[:, i], 2.5)
    post_q975 = np.percentile(p_posterior_mean[:, i], 97.5)

    # Compute shrinkage
    mu_post_mean = posterior['mu'].mean()
    if obs_rate != mu_post_mean:
        shrinkage_pct = 100 * (obs_rate - post_mean) / (obs_rate - mu_post_mean)
    else:
        shrinkage_pct = 0.0

    group_summary.append({
        'group': i + 1,
        'n_trials': data.iloc[i]['n_trials'],
        'r_successes': data.iloc[i]['r_successes'],
        'observed_rate': obs_rate,
        'posterior_mean': post_mean,
        'posterior_sd': post_sd,
        'posterior_025': post_q025,
        'posterior_975': post_q975,
        'shrinkage_pct': shrinkage_pct
    })

group_summary_df = pd.DataFrame(group_summary)
group_summary_df.to_csv(RESULTS_DIR / 'group_posterior_summary.csv', index=False)

print("\nGroup-level summaries:")
print(group_summary_df.to_string(index=False))
print(f"\nSaved to {RESULTS_DIR / 'group_posterior_summary.csv'}")

# Final summary
print("\n" + "="*70)
print("FITTING COMPLETE")
print("="*70)
print(f"\nPosterior Mean Estimates:")
print(f"  mu: {posterior['mu'].mean():.4f} (95% CI: [{np.percentile(posterior['mu'], 2.5):.4f}, {np.percentile(posterior['mu'], 97.5):.4f}])")
print(f"  kappa: {posterior['kappa'].mean():.2f} (95% CI: [{np.percentile(posterior['kappa'], 2.5):.2f}, {np.percentile(posterior['kappa'], 97.5):.2f}])")
print(f"  phi: {posterior['phi'].mean():.4f} (95% CI: [{np.percentile(posterior['phi'], 2.5):.4f}, {np.percentile(posterior['phi'], 97.5):.4f}])")

if all_pass:
    print("\nModel fitting SUCCESSFUL - proceed to visualization and posterior predictive checking")
else:
    print("\nModel fitting had convergence issues - review diagnostics before proceeding")

print("\nOutput files:")
print(f"  - Raw Stan output: {DIAGNOSTICS_DIR}")
print(f"  - ArviZ InferenceData: {netcdf_path}")
print(f"  - Posterior summaries: {RESULTS_DIR}")
