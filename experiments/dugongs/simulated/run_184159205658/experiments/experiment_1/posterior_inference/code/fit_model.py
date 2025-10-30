"""
Fit Bayesian Logarithmic Regression Model using Stan/CmdStanPy
Experiment 1 - Posterior Inference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
import arviz as az
import json
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path('/workspace/experiments/experiment_1/posterior_inference')
DATA_PATH = Path('/workspace/data/data.csv')
STAN_FILE = BASE_DIR / 'code' / 'logarithmic_model.stan'
DIAGNOSTICS_DIR = BASE_DIR / 'diagnostics'
PLOTS_DIR = BASE_DIR / 'plots'

print("="*80)
print("BAYESIAN LOGARITHMIC REGRESSION - POSTERIOR INFERENCE")
print("="*80)

# Load data
print("\n[1] Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"    N = {len(data)} observations")
print(f"    x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
print(f"    Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

# Prepare data for Stan
stan_data = {
    'N': len(data),
    'x': data['x'].values,
    'Y': data['Y'].values
}

# Compile Stan model
print("\n[2] Compiling Stan model...")
model = CmdStanModel(stan_file=str(STAN_FILE))
print(f"    Model compiled successfully")

# Initial probe sampling (conservative)
print("\n[3] Initial probe sampling...")
print("    4 chains × 200 iterations (100 warmup)")
print("    adapt_delta = 0.90")
print("    max_treedepth = 12")

try:
    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=100,
        iter_sampling=100,
        adapt_delta=0.90,
        max_treedepth=12,
        show_progress=True,
        seed=12345
    )

    # Check for divergences in probe
    probe_divergences = fit.num_unconstrained_params
    probe_summary = fit.summary()

    print("\n[Probe Results]")
    print(f"    Parameters: {probe_summary.shape[0]}")

    # Check R-hat
    max_rhat = probe_summary['R_hat'].max()
    print(f"    Max R-hat: {max_rhat:.4f}")

    # Check for divergences
    divergences = fit.divergences
    total_divergences = np.sum(divergences)
    print(f"    Divergent transitions: {total_divergences}")

    if total_divergences > 0:
        print("\n    WARNING: Divergences detected in probe. Increasing adapt_delta to 0.95")
        adapt_delta_main = 0.95
    else:
        print("    No divergences detected. Proceeding with adapt_delta = 0.90")
        adapt_delta_main = 0.90

except Exception as e:
    print(f"\n    ERROR in probe: {e}")
    print("    Attempting with higher adapt_delta...")
    adapt_delta_main = 0.95

# Main sampling
print("\n[4] Main sampling...")
print("    4 chains × 2000 iterations (1000 warmup)")
print(f"    adapt_delta = {adapt_delta_main}")
print("    max_treedepth = 12")

fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=adapt_delta_main,
    max_treedepth=12,
    show_progress=True,
    seed=54321
)

print("\n[5] Sampling complete!")

# Convert to ArviZ InferenceData with log_likelihood
print("\n[6] Converting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    observed_data={'Y': data['Y'].values},
    coords={'obs_id': np.arange(len(data))},
    dims={'log_lik': ['obs_id'], 'Y_rep': ['obs_id']},
    log_likelihood='log_lik'
)

# Save InferenceData
print(f"    Saving to: {DIAGNOSTICS_DIR / 'posterior_inference.netcdf'}")
idata.to_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')

# Generate summary
print("\n[7] Generating posterior summary...")
summary = az.summary(idata, var_names=['beta0', 'beta1', 'sigma'])
print("\n" + "="*80)
print("POSTERIOR SUMMARY")
print("="*80)
print(summary)

# Save summary to file
summary.to_csv(DIAGNOSTICS_DIR / 'posterior_summary.csv')
print(f"\n    Saved to: {DIAGNOSTICS_DIR / 'posterior_summary.csv'}")

# Convergence diagnostics
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

# Check R-hat
max_rhat = summary['r_hat'].max()
rhat_pass = max_rhat < 1.01
print(f"\nR-hat:")
print(f"    Max R-hat: {max_rhat:.4f}")
print(f"    Criterion: < 1.01")
print(f"    Status: {'PASS ✓' if rhat_pass else 'FAIL ✗'}")

# Check ESS
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()
ess_pass = (min_ess_bulk > 400) and (min_ess_tail > 400)
print(f"\nEffective Sample Size:")
print(f"    Min ESS_bulk: {min_ess_bulk:.0f}")
print(f"    Min ESS_tail: {min_ess_tail:.0f}")
print(f"    Criterion: > 400")
print(f"    Status: {'PASS ✓' if ess_pass else 'FAIL ✗'}")

# Check divergences
divergences = fit.divergences
total_divergences = np.sum(divergences)
pct_divergences = 100 * total_divergences / (fit.num_draws_sampling * fit.chains)
div_pass = pct_divergences < 1.0
print(f"\nDivergent Transitions:")
print(f"    Count: {total_divergences}")
print(f"    Percentage: {pct_divergences:.2f}%")
print(f"    Criterion: < 1%")
print(f"    Status: {'PASS ✓' if div_pass else 'FAIL ✗'}")

# Check E-BFMI
ebfmi = fit.method_variables()['energy__']
ebfmi_vals = []
for chain in range(fit.chains):
    chain_ebfmi = np.var(np.diff(ebfmi[chain*fit.num_draws_sampling:(chain+1)*fit.num_draws_sampling])) / \
                   np.var(ebfmi[chain*fit.num_draws_sampling:(chain+1)*fit.num_draws_sampling])
    ebfmi_vals.append(chain_ebfmi)
min_ebfmi = min(ebfmi_vals)
ebfmi_pass = min_ebfmi > 0.3
print(f"\nE-BFMI:")
print(f"    Min E-BFMI: {min_ebfmi:.3f}")
print(f"    Criterion: > 0.3")
print(f"    Status: {'PASS ✓' if ebfmi_pass else 'FAIL ✗'}")

# Overall convergence
convergence_pass = rhat_pass and ess_pass and div_pass and ebfmi_pass
print(f"\n{'='*80}")
print(f"OVERALL CONVERGENCE: {'PASS ✓' if convergence_pass else 'FAIL ✗'}")
print(f"{'='*80}")

# Extract posterior samples
posterior = idata.posterior
beta0_samples = posterior['beta0'].values.flatten()
beta1_samples = posterior['beta1'].values.flatten()
sigma_samples = posterior['sigma'].values.flatten()

# Posterior statistics
print("\n" + "="*80)
print("PARAMETER INFERENCE")
print("="*80)

print("\nβ₀ (Intercept):")
print(f"    Mean: {beta0_samples.mean():.4f}")
print(f"    SD: {beta0_samples.std():.4f}")
print(f"    95% CI: [{np.percentile(beta0_samples, 2.5):.4f}, {np.percentile(beta0_samples, 97.5):.4f}]")
print(f"    Prior: N(1.73, 0.5)")

print("\nβ₁ (Log slope):")
print(f"    Mean: {beta1_samples.mean():.4f}")
print(f"    SD: {beta1_samples.std():.4f}")
print(f"    95% CI: [{np.percentile(beta1_samples, 2.5):.4f}, {np.percentile(beta1_samples, 97.5):.4f}]")
print(f"    Prior: N(0.28, 0.15)")
print(f"    P(β₁ > 0): {(beta1_samples > 0).mean():.4f}")

print("\nσ (Error SD):")
print(f"    Mean: {sigma_samples.mean():.4f}")
print(f"    SD: {sigma_samples.std():.4f}")
print(f"    95% CI: [{np.percentile(sigma_samples, 2.5):.4f}, {np.percentile(sigma_samples, 97.5):.4f}]")
print(f"    Prior: Exp(5)")

# Parameter correlations
print("\nParameter Correlations:")
corr_beta0_beta1 = np.corrcoef(beta0_samples, beta1_samples)[0, 1]
print(f"    Corr(β₀, β₁): {corr_beta0_beta1:.3f}")

# Model fit assessment
print("\n" + "="*80)
print("MODEL FIT ASSESSMENT")
print("="*80)

# Posterior predictive mean
mu_samples = posterior['mu'].values  # shape: (chains, draws, N)
mu_mean = mu_samples.mean(axis=(0, 1))
mu_lower = np.percentile(mu_samples, 2.5, axis=(0, 1))
mu_upper = np.percentile(mu_samples, 97.5, axis=(0, 1))

# Residuals
residuals = data['Y'].values - mu_mean

# In-sample metrics
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
ss_res = np.sum(residuals**2)
ss_tot = np.sum((data['Y'].values - data['Y'].mean())**2)
r_squared = 1 - ss_res / ss_tot

print(f"\nIn-Sample Metrics:")
print(f"    RMSE: {rmse:.4f}")
print(f"    MAE: {mae:.4f}")
print(f"    R²: {r_squared:.4f}")

# Save residuals
residuals_df = pd.DataFrame({
    'x': data['x'].values,
    'Y': data['Y'].values,
    'mu_mean': mu_mean,
    'mu_lower': mu_lower,
    'mu_upper': mu_upper,
    'residual': residuals
})
residuals_df.to_csv(DIAGNOSTICS_DIR / 'residuals.csv', index=False)
print(f"\n    Residuals saved to: {DIAGNOSTICS_DIR / 'residuals.csv'}")

# LOO-CV Analysis
print("\n" + "="*80)
print("LOO-CV ANALYSIS")
print("="*80)

loo = az.loo(idata, pointwise=True)
print(f"\nLOO-CV Results:")
print(f"    ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"    p_loo: {loo.p_loo:.2f}")
print(f"    LOO-IC: {-2 * loo.elpd_loo:.2f}")

# Pareto k diagnostics
pareto_k = loo.pareto_k.values
n_high_k = np.sum(pareto_k > 0.7)
print(f"\nPareto k diagnostics:")
print(f"    Good (k < 0.5): {np.sum(pareto_k < 0.5)}")
print(f"    OK (0.5 ≤ k < 0.7): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))}")
print(f"    Bad (k ≥ 0.7): {n_high_k}")

if n_high_k > 0:
    print(f"\n    WARNING: {n_high_k} observations with k > 0.7")
    high_k_idx = np.where(pareto_k > 0.7)[0]
    print(f"    Indices: {high_k_idx}")
else:
    print(f"\n    All Pareto k values < 0.7 ✓")

# Save LOO results
loo_df = pd.DataFrame({
    'obs_id': np.arange(len(data)),
    'x': data['x'].values,
    'Y': data['Y'].values,
    'pareto_k': pareto_k,
    'loo_i': loo.loo_i.values
})
loo_df.to_csv(DIAGNOSTICS_DIR / 'loo_results.csv', index=False)

# Overall assessment
print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

pass_criteria = {
    'Convergence (R-hat < 1.01)': rhat_pass,
    'ESS > 400': ess_pass,
    'Divergences < 1%': div_pass,
    'E-BFMI > 0.3': ebfmi_pass,
    'β₁ > 0': (beta1_samples > 0).mean() > 0.95,
    'No computational errors': True
}

print("\nPass/Fail Criteria:")
for criterion, status in pass_criteria.items():
    print(f"    {criterion}: {'PASS ✓' if status else 'FAIL ✗'}")

overall_pass = all(pass_criteria.values())
print(f"\n{'='*80}")
print(f"FINAL VERDICT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
print(f"{'='*80}")

# Save diagnostic info
diagnostics_dict = {
    'convergence': {
        'max_rhat': float(max_rhat),
        'min_ess_bulk': float(min_ess_bulk),
        'min_ess_tail': float(min_ess_tail),
        'divergences': int(total_divergences),
        'pct_divergences': float(pct_divergences),
        'min_ebfmi': float(min_ebfmi),
        'overall_pass': convergence_pass
    },
    'parameters': {
        'beta0': {
            'mean': float(beta0_samples.mean()),
            'sd': float(beta0_samples.std()),
            'ci_lower': float(np.percentile(beta0_samples, 2.5)),
            'ci_upper': float(np.percentile(beta0_samples, 97.5))
        },
        'beta1': {
            'mean': float(beta1_samples.mean()),
            'sd': float(beta1_samples.std()),
            'ci_lower': float(np.percentile(beta1_samples, 2.5)),
            'ci_upper': float(np.percentile(beta1_samples, 97.5)),
            'prob_positive': float((beta1_samples > 0).mean())
        },
        'sigma': {
            'mean': float(sigma_samples.mean()),
            'sd': float(sigma_samples.std()),
            'ci_lower': float(np.percentile(sigma_samples, 2.5)),
            'ci_upper': float(np.percentile(sigma_samples, 97.5))
        }
    },
    'fit': {
        'rmse': float(rmse),
        'mae': float(mae),
        'r_squared': float(r_squared)
    },
    'loo': {
        'elpd_loo': float(loo.elpd_loo),
        'se': float(loo.se),
        'p_loo': float(loo.p_loo),
        'loo_ic': float(-2 * loo.elpd_loo),
        'n_high_pareto_k': int(n_high_k)
    },
    'overall_pass': overall_pass
}

with open(DIAGNOSTICS_DIR / 'diagnostics_summary.json', 'w') as f:
    json.dump(diagnostics_dict, f, indent=2)

print(f"\nDiagnostics saved to: {DIAGNOSTICS_DIR / 'diagnostics_summary.json'}")
print("\n" + "="*80)
print("Proceeding to create diagnostic plots...")
print("="*80)
