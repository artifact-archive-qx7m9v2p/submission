"""
Fit 3-component finite mixture model to data using Stan.

Model:
- pi ~ Dirichlet([1, 1, 1])
- z[j] ~ Categorical(pi) [marginalized]
- mu[k] ~ Normal(-2.6, 1.0) with ordered constraint
- sigma[k] ~ Half-Normal(0, 0.5)
- theta[j] ~ Normal(mu[z[j]], sigma[z[j]])
- r[j] ~ Binomial(n[j], inv_logit(theta[j]))
"""

import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import arviz as az
import pickle
from pathlib import Path

# Setup paths
project_root = Path("/workspace")
data_path = project_root / "data" / "data.csv"
output_dir = project_root / "experiments" / "experiment_2" / "posterior_inference"
code_dir = output_dir / "code"
diag_dir = output_dir / "diagnostics"
plots_dir = output_dir / "plots"

# Ensure directories exist
for d in [code_dir, diag_dir, plots_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv(data_path)
print(f"Data shape: {data.shape}")
print(data)

# Prepare data for Stan
n_groups = len(data)
n_trials = data['n_trials'].values
r_successes = data['r_successes'].values

print(f"\nData summary:")
print(f"  Number of groups: {n_groups}")
print(f"  Total trials: {n_trials.sum()}")
print(f"  Total successes: {r_successes.sum()}")
print(f"  Overall success rate: {r_successes.sum() / n_trials.sum():.4f}")

stan_data = {
    'J': n_groups,
    'n': n_trials.tolist(),
    'r': r_successes.tolist()
}

# Compile Stan model
print("\nCompiling Stan model...")
stan_file = code_dir / "mixture_model.stan"
model = CmdStanModel(stan_file=stan_file)
print("Compilation successful!")

# Sample from posterior
print("\n" + "="*70)
print("FITTING MODEL WITH HMC")
print("="*70)
print("\nStrategy: Mixture model with marginalized likelihood")
print("  - 4 chains")
print("  - 2000 warmup, 2000 sampling")
print("  - adapt_delta = 0.95 (mixture models need careful exploration)")
print("  - Ordered constraint on mu prevents label switching")

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.95,
    max_treedepth=12,
    seed=42,
    show_console=True
)

print("\nSampling complete!")

# Check basic convergence
print("\n" + "="*70)
print("CONVERGENCE DIAGNOSTICS")
print("="*70)

# Stan diagnostics
print("\nStan Diagnostics:")
print(fit.diagnose())

# Convert to ArviZ InferenceData
print("\nConverting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    posterior_predictive=['r_rep'],
    log_likelihood='log_lik',
    observed_data={'r': r_successes}
)

# Get summary
summary = az.summary(idata, var_names=['pi', 'mu', 'sigma'])
print("\nParameter Summary:")
print(summary)

# Check for issues
rhat_max = summary['r_hat'].max()
ess_min = summary['ess_bulk'].min()

print(f"\nConvergence metrics:")
print(f"  Max R-hat: {rhat_max:.4f} (should be < 1.01)")
print(f"  Min ESS_bulk: {ess_min:.1f} (should be > 400)")

# Extract diagnostics from Stan
divergences = fit.divergences
num_samples = fit.num_draws_sampling * fit.chains
div_pct = 100 * divergences / num_samples if num_samples > 0 else 0

print(f"  Divergences: {divergences} / {num_samples} ({div_pct:.2f}%)")

# Check mu ordering is maintained
mu_samples = fit.stan_variable('mu')  # shape: (draws, 3)
ordering_violations = np.sum((mu_samples[:, 1] <= mu_samples[:, 0]) |
                              (mu_samples[:, 2] <= mu_samples[:, 1]))
print(f"  Ordering violations: {ordering_violations} / {len(mu_samples)} ({100*ordering_violations/len(mu_samples):.2f}%)")

# Save results
print("\nSaving results...")

# Save as ArviZ InferenceData with log_likelihood
idata.to_netcdf(diag_dir / "posterior_inference.netcdf")
print(f"  Saved InferenceData: {diag_dir / 'posterior_inference.netcdf'}")

# Save summary
summary_full = az.summary(idata)
summary_full.to_csv(diag_dir / "parameter_summary.csv")
print(f"  Saved summary: {diag_dir / 'parameter_summary.csv'}")

# Save fit object
fit.save_csvfiles(dir=str(diag_dir))
print(f"  Saved Stan CSV files: {diag_dir}")

# Save InferenceData as pickle for further analysis
with open(diag_dir / "idata.pkl", 'wb') as f:
    pickle.dump(idata, f)
print(f"  Saved InferenceData: {diag_dir / 'idata.pkl'}")

# Verify log_likelihood is present
if 'log_lik' in idata.log_likelihood:
    print("\n✓ log_likelihood successfully included in InferenceData")
    print(f"  Shape: {idata.log_likelihood['log_lik'].shape}")
else:
    print("\n✗ WARNING: log_likelihood not found in InferenceData!")

print("\n" + "="*70)
print("FITTING COMPLETE")
print("="*70)
print(f"\nResults saved to: {output_dir}")
print("\nNext steps:")
print("  1. Run diagnostic analysis script")
print("  2. Create visualizations")
print("  3. Compute cluster assignments")
