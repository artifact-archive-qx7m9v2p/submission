"""
Fit hierarchical model with skeptical priors (Model 4a) using CmdStanPy
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
import subprocess
import os

# Set up paths
exp_dir = Path('/workspace/experiments/experiment_4/experiment_4a_skeptical/posterior_inference')
code_dir = exp_dir / 'code'
diag_dir = exp_dir / 'diagnostics'
plot_dir = exp_dir / 'plots'

# Load data
data_path = Path('/workspace/data/data.csv')
df = pd.read_csv(data_path)

print("=" * 80)
print("MODEL 4a: SKEPTICAL PRIORS")
print("=" * 80)
print(f"\nData: {len(df)} studies")
print(f"Observed effects: {df['y'].values}")
print(f"Standard errors: {df['sigma'].values}")
print("\nPrior specification:")
print("  mu ~ Normal(0, 10)      [Skeptical of large effects]")
print("  tau ~ Half-Normal(0, 5) [Expects low heterogeneity]")
print()

# Prepare data for Stan
stan_data = {
    'J': len(df),
    'y': df['y'].values.tolist(),
    'sigma': df['sigma'].values.tolist()
}

# Compile model using stanc directly
print("Compiling Stan model manually...")
stan_file = code_dir / 'hierarchical_skeptical.stan'
cmdstan_path = cmdstanpy.cmdstan_path()
stanc_path = Path(cmdstan_path) / 'bin' / 'linux-stanc'

# Compile to C++
cpp_file = code_dir / 'hierarchical_skeptical.hpp'
compile_cmd = f"{stanc_path} {stan_file} --o={cpp_file}"
print(f"Running: {compile_cmd}")
result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Compilation failed: {result.stderr}")
    raise RuntimeError("Stan compilation failed")

print("Stan model compiled to C++")

# Now use CmdStanPy with force_compile=False since we manually compiled
# Actually, let me try a different approach - install make
print("\nAttempting to use CmdStanPy directly with model compilation...")

# Set environment to use cmdstan's make
os.environ['PATH'] = f"{cmdstan_path}/bin:{os.environ.get('PATH', '')}"

try:
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file), force_compile=True)
    print("Model compiled successfully!")
except Exception as e:
    print(f"Compilation error: {e}")
    print("\nTrying alternative: write data and use cmdstan directly")

    # Write data file
    import json
    data_file = code_dir / 'data.json'
    with open(data_file, 'w') as f:
        json.dump(stan_data, f)

    # Manually run stanc and compile
    print("Manual compilation approach...")
    raise RuntimeError("Cannot compile Stan model - 'make' not available")

# If we got here, compilation worked
print("\n" + "=" * 80)
print("PROBE RUN (4 chains × 200 iterations)")
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

max_rhat = probe_summary['R_hat'].max()
min_ess = probe_summary['ESS_bulk'].min()

print(f"\nMax R-hat: {max_rhat:.4f}")
print(f"Min ESS_bulk: {min_ess:.1f}")

if max_rhat > 1.01 or min_ess < 50:
    print("Probe detected issues - will increase sampling")
    adapt_delta = 0.95
    iter_warmup = 2000
    iter_sampling = 2500
else:
    print("Probe successful!")
    adapt_delta = 0.8
    iter_warmup = 2500
    iter_sampling = 2500

# MAIN RUN
print("\n" + "=" * 80)
print(f"MAIN SAMPLING (4 chains × {iter_warmup + iter_sampling} iterations)")
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

# Convert to ArviZ InferenceData
print("\nConverting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    posterior_predictive=None,
    log_likelihood='log_lik'
)

# Save InferenceData
netcdf_path = diag_dir / 'posterior_inference.netcdf'
idata.to_netcdf(netcdf_path)
print(f"Saved to: {netcdf_path}")

# Summary and diagnostics
summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)
print(summary)

summary.to_csv(diag_dir / 'posterior_summary.csv')

# Extract results
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
print(f"\nmu: {mu_mean:.2f} ± {mu_sd:.2f}, 95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]")
print(f"tau: {tau_mean:.2f} ± {tau_sd:.2f}, 95% CI: [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]")

max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
convergence_ok = max_rhat < 1.01 and min_ess_bulk > 400

print(f"\nMax R-hat: {max_rhat:.4f}, Min ESS: {min_ess_bulk:.1f}")
print("Converged!" if convergence_ok else "Convergence issues")

# Save results
results = {
    'model': 'skeptical',
    'mu': {'mean': float(mu_mean), 'sd': float(mu_sd),
           'ci_lower': float(mu_ci[0]), 'ci_upper': float(mu_ci[1])},
    'tau': {'mean': float(tau_mean), 'sd': float(tau_sd),
            'ci_lower': float(tau_ci[0]), 'ci_upper': float(tau_ci[1])},
    'convergence': {
        'max_rhat': float(max_rhat),
        'min_ess_bulk': float(min_ess_bulk),
        'converged': convergence_ok
    }
}

with open(diag_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nFITTING COMPLETE!")
