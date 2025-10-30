"""
Fit Negative Binomial Quadratic Model to Real Data (PyMC Implementation)
Experiment 1: Posterior Inference with Bayesian MCMC
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

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
print("NEGATIVE BINOMIAL QUADRATIC MODEL - POSTERIOR INFERENCE (PyMC)")
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

year = df['year'].values
C = df['C'].values
N = len(df)

# ============================================================================
# 2. BUILD PYMC MODEL
# ============================================================================
print("\n[2/6] Building PyMC model...")

with pm.Model() as model:
    # Priors (validated via SBC)
    beta_0 = pm.Normal('beta_0', mu=4.7, sigma=0.3)
    beta_1 = pm.Normal('beta_1', mu=0.8, sigma=0.2)
    beta_2 = pm.Normal('beta_2', mu=0.3, sigma=0.1)
    phi = pm.Gamma('phi', alpha=2, beta=0.5)

    # Expected value (quadratic on log scale)
    log_mu = beta_0 + beta_1 * year + beta_2 * year**2
    mu = pm.math.exp(log_mu)

    # Likelihood: Negative Binomial
    # PyMC uses NegativeBinomial(mu, alpha) where alpha is dispersion parameter
    # Larger alpha = more dispersion (opposite of Stan's convention)
    C_obs = pm.NegativeBinomial('C_obs', mu=mu, alpha=phi, observed=C)

print("  ✓ Model built successfully")
print(f"  Model variables: {[str(v) for v in model.free_RVs]}")

# ============================================================================
# 3. INITIAL PROBE: Quick assessment (200 iterations)
# ============================================================================
print("\n[3/6] Running initial probe (4 chains × 200 iterations)...")

with model:
    try:
        probe_trace = pm.sample(
            draws=100,
            tune=100,
            chains=4,
            cores=4,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42
        )

        probe_summary = az.summary(probe_trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])
        print("\n  Probe results:")
        print(f"    R̂ range: [{probe_summary['r_hat'].min():.4f}, {probe_summary['r_hat'].max():.4f}]")
        print(f"    ESS_bulk range: [{probe_summary['ess_bulk'].min():.0f}, {probe_summary['ess_bulk'].max():.0f}]")

        # Check divergences
        divergences = probe_trace.sample_stats.diverging.sum().values
        print(f"    Divergent transitions: {divergences}")

        if probe_summary['r_hat'].max() > 1.05 or divergences > 10:
            print("  ⚠ Warning: Probe shows convergence issues, adjusting settings...")
            target_accept = 0.98
        else:
            print("  ✓ Probe successful, proceeding with standard settings")
            target_accept = 0.95

    except Exception as e:
        print(f"  ✗ Probe failed: {e}")
        print("  Attempting with more conservative settings...")
        target_accept = 0.98

# ============================================================================
# 4. MAIN SAMPLING: Full posterior inference
# ============================================================================
print(f"\n[4/6] Running main sampling (4 chains × 2000 iterations)...")
print(f"  Settings: target_accept={target_accept}")

with model:
    try:
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True,
            random_seed=123
        )

        print("\n  ✓ Sampling completed successfully")

        # Check for divergences
        total_divergences = trace.sample_stats.diverging.sum().values
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
            f.write(f"  target_accept: {target_accept}\n")
        print(f"  Error log saved to: {error_log_path}")
        raise

# ============================================================================
# 5. ADD POSTERIOR PREDICTIVE AND LOG LIKELIHOOD
# ============================================================================
print("\n[5/6] Adding posterior predictive and log likelihood...")

with model:
    # Add posterior predictive
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=456)

    # Compute log likelihood (required for LOO-CV)
    pm.compute_log_likelihood(trace)

print("  ✓ Posterior predictive and log likelihood computed")

# Save to netcdf
idata_path = DIAG_DIR / "posterior_inference.netcdf"
trace.to_netcdf(idata_path)
print(f"  ✓ InferenceData saved to: {idata_path}")
print(f"  Groups: {list(trace.groups())}")

# Verify log_likelihood group exists
if 'log_likelihood' in trace.groups():
    print(f"  ✓ log_likelihood group present (shape: {trace.log_likelihood['C_obs'].shape})")
else:
    print(f"  ✗ WARNING: log_likelihood group missing!")

# ============================================================================
# 6. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[6/6] Running convergence diagnostics...")

# Summary statistics
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])
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
    'divergent_transitions': int(total_divergences),
    'divergence_rate': float(div_pct)
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
# Convert numpy types to native Python types for JSON serialization
convergence_checks_serializable = {
    k: (float(v) if isinstance(v, (np.floating, np.integer)) else (bool(v) if isinstance(v, np.bool_) else v))
    for k, v in convergence_checks.items()
}
with open(convergence_path, 'w') as f:
    json.dump(convergence_checks_serializable, f, indent=2)
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
    trace,
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
    trace,
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
    post_samples = trace.posterior[param].values.flatten()
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
    trace,
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
    az.plot_energy(trace, figsize=(10, 6))
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
