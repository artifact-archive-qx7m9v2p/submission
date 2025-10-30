"""
Fit Negative Binomial Quadratic Model to Real Data
Experiment 1: Posterior Inference with Bayesian MCMC
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
CODE_DIR = BASE_DIR / "code"
DIAG_DIR = BASE_DIR / "diagnostics"
PLOT_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Ensure directories exist
DIAG_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

print("="*70)
print("NEGATIVE BINOMIAL QUADRATIC MODEL - POSTERIOR INFERENCE")
print("="*70)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} observations")
print(f"  Year range: [{df['year'].min():.2f}, {df['year'].max():.2f}]")
print(f"  Count range: [{df['C'].min()}, {df['C'].max()}]")
print(f"  Count mean: {df['C'].mean():.1f}, median: {df['C'].median():.1f}")

stan_data = {
    'N': len(df),
    'C': df['C'].values.astype(int).tolist(),
    'year': df['year'].values.tolist()
}

# ============================================================================
# 2. COMPILE STAN MODEL
# ============================================================================
print("\n[2/6] Compiling Stan model...")
model_path = CODE_DIR / "model.stan"
try:
    model = CmdStanModel(stan_file=str(model_path))
    print(f"  ✓ Model compiled successfully")
except Exception as e:
    print(f"  ✗ Compilation failed: {e}")
    raise

# ============================================================================
# 3. INITIAL PROBE: Quick assessment (200 iterations)
# ============================================================================
print("\n[3/6] Running initial probe (4 chains × 200 iterations)...")
try:
    probe_fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=100,
        iter_sampling=100,
        adapt_delta=0.95,
        max_treedepth=10,
        show_progress=True,
        show_console=False
    )

    probe_summary = probe_fit.summary()
    print("\n  Probe results:")
    print(f"    R̂ range: [{probe_summary['R_hat'].min():.4f}, {probe_summary['R_hat'].max():.4f}]")
    print(f"    ESS_bulk range: [{probe_summary['ess_bulk'].min():.0f}, {probe_summary['ess_bulk'].max():.0f}]")

    divergences = probe_fit.divergences
    if divergences is not None:
        div_count = np.sum(divergences)
    else:
        div_count = 0
    print(f"    Divergent transitions: {div_count}")

    if probe_summary['R_hat'].max() > 1.05 or div_count > 10:
        print("  ⚠ Warning: Probe shows convergence issues, adjusting settings...")
        adapt_delta = 0.98
        max_treedepth = 12
    else:
        print("  ✓ Probe successful, proceeding with standard settings")
        adapt_delta = 0.95
        max_treedepth = 10

except Exception as e:
    print(f"  ✗ Probe failed: {e}")
    print("  Attempting with more conservative settings...")
    adapt_delta = 0.98
    max_treedepth = 12

# ============================================================================
# 4. MAIN SAMPLING: Full posterior inference
# ============================================================================
print(f"\n[4/6] Running main sampling (4 chains × 2000 iterations)...")
print(f"  Settings: adapt_delta={adapt_delta}, max_treedepth={max_treedepth}")

try:
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_progress=True,
        show_console=False,
        save_warmup=False
    )

    print("\n  ✓ Sampling completed successfully")

    # Check for numerical issues
    if fit.divergences is not None:
        total_divergences = np.sum(fit.divergences)
    else:
        total_divergences = 0

    total_iterations = 4 * 1000  # 4 chains × 1000 samples
    div_pct = 100 * total_divergences / total_iterations

    print(f"  Divergent transitions: {total_divergences} ({div_pct:.2f}%)")

except Exception as e:
    print(f"  ✗ Sampling failed: {e}")
    # Save error log
    error_log_path = DIAG_DIR / "sampling_error.txt"
    with open(error_log_path, 'w') as f:
        f.write(f"Sampling Error:\n{str(e)}\n\n")
        f.write(f"Settings used:\n")
        f.write(f"  adapt_delta: {adapt_delta}\n")
        f.write(f"  max_treedepth: {max_treedepth}\n")
    print(f"  Error log saved to: {error_log_path}")
    raise

# ============================================================================
# 5. CONVERT TO ARVIZ AND SAVE
# ============================================================================
print("\n[5/6] Converting to ArviZ InferenceData...")

# Convert to ArviZ with log_likelihood group
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive=['C_rep'],
    log_likelihood='log_lik',
    observed_data={'C': df['C'].values},
    coords={'obs_id': np.arange(len(df))},
    dims={'log_lik': ['obs_id'], 'C_rep': ['obs_id']}
)

# Save to netcdf
idata_path = DIAG_DIR / "posterior_inference.netcdf"
idata.to_netcdf(idata_path)
print(f"  ✓ InferenceData saved to: {idata_path}")
print(f"  Groups: {list(idata.groups())}")

# Verify log_likelihood group exists
if 'log_likelihood' in idata.groups():
    print(f"  ✓ log_likelihood group present (shape: {idata.log_likelihood['log_lik'].shape})")
else:
    print(f"  ✗ WARNING: log_likelihood group missing!")

# ============================================================================
# 6. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[6/6] Running convergence diagnostics...")

# Summary statistics
summary = az.summary(idata, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])
print("\n" + "="*70)
print("PARAMETER SUMMARY")
print("="*70)
print(summary.to_string())
print("="*70)

# Save summary table
summary_path = DIAG_DIR / "summary_table.csv"
summary.to_csv(summary_path)
print(f"\n  ✓ Summary table saved to: {summary_path}")

# Detailed convergence checks
print("\n" + "="*70)
print("CONVERGENCE ASSESSMENT")
print("="*70)

convergence_checks = {
    'r_hat_max': summary['r_hat'].max(),
    'r_hat_all_below_1.01': (summary['r_hat'] < 1.01).all(),
    'ess_bulk_min': summary['ess_bulk'].min(),
    'ess_tail_min': summary['ess_tail'].min(),
    'ess_bulk_all_above_400': (summary['ess_bulk'] > 400).all(),
    'ess_tail_all_above_400': (summary['ess_tail'] > 400).all(),
    'divergent_transitions': total_divergences,
    'divergence_rate': div_pct
}

for key, value in convergence_checks.items():
    if isinstance(value, bool):
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    else:
        print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")

# Overall convergence status
convergence_pass = (
    convergence_checks['r_hat_all_below_1.01'] and
    convergence_checks['ess_bulk_all_above_400'] and
    convergence_checks['ess_tail_all_above_400'] and
    convergence_checks['divergence_rate'] < 1.0
)

print("\n" + "="*70)
if convergence_pass:
    print("CONVERGENCE STATUS: ✓ PASS")
else:
    print("CONVERGENCE STATUS: ✗ FAIL")
print("="*70)

# Save convergence metrics
convergence_path = DIAG_DIR / "convergence_metrics.json"
with open(convergence_path, 'w') as f:
    json.dump(convergence_checks, f, indent=2)
print(f"\n  ✓ Convergence metrics saved to: {convergence_path}")

# ============================================================================
# 7. DIAGNOSTIC VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*70)

# Plot 1: Trace plots (compact overview)
print("\n  [1/5] Trace plots...")
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
az.plot_trace(
    idata,
    var_names=['beta_0', 'beta_1', 'beta_2', 'phi'],
    compact=True,
    axes=axes
)
fig.suptitle("Trace Plots: Convergence Overview", fontsize=14, y=1.00)
plt.tight_layout()
trace_path = PLOT_DIR / "trace_plots.png"
plt.savefig(trace_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved to: {trace_path}")

# Plot 2: Rank plots (for detailed mixing assessment)
print("  [2/5] Rank plots...")
axes_rank = az.plot_rank(
    idata,
    var_names=['beta_0', 'beta_1', 'beta_2', 'phi'],
    figsize=(12, 8)
)
plt.suptitle("Rank Plots: Chain Mixing Uniformity", fontsize=14, y=1.00)
plt.tight_layout()
rank_path = PLOT_DIR / "rank_plots.png"
plt.savefig(rank_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved to: {rank_path}")

# Plot 3: Posterior distributions with priors
print("  [3/5] Posterior vs Prior distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
params = ['beta_0', 'beta_1', 'beta_2', 'phi']
param_labels = [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$', r'$\phi$']

# Prior distributions for reference
prior_params = {
    'beta_0': {'dist': 'normal', 'loc': 4.7, 'scale': 0.3},
    'beta_1': {'dist': 'normal', 'loc': 0.8, 'scale': 0.2},
    'beta_2': {'dist': 'normal', 'loc': 0.3, 'scale': 0.1},
    'phi': {'dist': 'gamma', 'a': 2, 'scale': 2.0}  # Gamma(2, 0.5) -> scale=1/0.5=2
}

for i, (param, label) in enumerate(zip(params, param_labels)):
    ax = axes.flat[i]

    # Plot posterior
    post_samples = idata.posterior[param].values.flatten()
    ax.hist(post_samples, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')

    # Plot prior
    x_range = np.linspace(post_samples.min(), post_samples.max(), 200)
    if prior_params[param]['dist'] == 'normal':
        from scipy.stats import norm
        prior_density = norm.pdf(x_range, prior_params[param]['loc'], prior_params[param]['scale'])
    else:  # gamma
        from scipy.stats import gamma
        prior_density = gamma.pdf(x_range, prior_params[param]['a'], scale=prior_params[param]['scale'])

    ax.plot(x_range, prior_density, 'r--', linewidth=2, label='Prior')

    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.set_title(f'{label}: Posterior vs Prior')

plt.tight_layout()
posterior_path = PLOT_DIR / "posterior_distributions.png"
plt.savefig(posterior_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved to: {posterior_path}")

# Plot 4: Pairwise correlations
print("  [4/5] Pairwise parameter correlations...")
axes_pair = az.plot_pair(
    idata,
    var_names=['beta_0', 'beta_1', 'beta_2', 'phi'],
    kind='kde',
    marginals=True,
    figsize=(12, 10),
    divergences=True
)
plt.suptitle("Pairwise Parameter Relationships", fontsize=14, y=1.00)
pair_path = PLOT_DIR / "pairwise_correlations.png"
plt.savefig(pair_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved to: {pair_path}")

# Plot 5: Energy diagnostic
print("  [5/5] Energy diagnostic...")
try:
    az.plot_energy(idata, figsize=(10, 6))
    plt.title("Energy Diagnostic: Transition vs Marginal Energy")
    energy_path = PLOT_DIR / "energy_diagnostic.png"
    plt.savefig(energy_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved to: {energy_path}")
except Exception as e:
    print(f"    ⚠ Energy plot failed: {e}")

print("\n" + "="*70)
print("POSTERIOR INFERENCE COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {BASE_DIR}")
print(f"  - Model code: {CODE_DIR}")
print(f"  - Diagnostics: {DIAG_DIR}")
print(f"  - Plots: {PLOT_DIR}")
