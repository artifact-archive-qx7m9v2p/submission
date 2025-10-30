"""
Fit Negative Binomial Linear Model to Real Data using PyMC
Experiment 1: Baseline model fitting with convergence diagnostics and LOO

Note: Using PyMC as CmdStanPy compilation requires make which is not available.
"""

import sys
import os

# Add user-installed packages to path
user_site = os.path.expanduser('~/.local/lib/python3.13/site-packages')
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1")
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = BASE_DIR / "posterior_inference"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPERIMENT 1: NEGATIVE BINOMIAL LINEAR MODEL - POSTERIOR INFERENCE")
print("=" * 80)
print("NOTE: Using PyMC (Stan compilation requires make which is not available)")

# Load data
print("\n[1/7] Loading data...")
data_df = pd.read_csv(DATA_PATH)
print(f"  - Observations: {len(data_df)}")
print(f"  - Mean count: {data_df['C'].mean():.1f}")
print(f"  - Variance: {data_df['C'].var():.1f}")
print(f"  - Var/Mean ratio: {data_df['C'].var() / data_df['C'].mean():.2f}")

# Extract data
year = data_df['year'].values
C = data_df['C'].values

# Build PyMC model
print("\n[2/7] Building PyMC model...")
with pm.Model() as model:
    # Priors (matching Stan model exactly)
    beta_0 = pm.Normal('beta_0', mu=4.69, sigma=1.0)
    beta_1 = pm.Normal('beta_1', mu=1.0, sigma=0.5)
    phi = pm.Gamma('phi', alpha=2, beta=0.1)

    # Linear predictor
    mu = pm.math.exp(beta_0 + beta_1 * year)

    # Likelihood
    # PyMC uses alpha parameterization: alpha = phi, mu = mean
    # This matches Stan's neg_binomial_2(mu, phi)
    y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=phi, observed=C)

print("  - Model built successfully")
print("  - Priors: beta_0 ~ N(4.69, 1.0), beta_1 ~ N(1.0, 0.5), phi ~ Gamma(2, 0.1)")

# Sample from posterior
print("\n[3/7] Sampling from posterior...")
print("  Configuration:")
print("    - Chains: 4")
print("    - Draws per chain: 1000")
print("    - Tune: 1000")
print("    - Target accept: 0.95")
print("    - max_treedepth: 12")

with model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        max_treedepth=12,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

print("  - Sampling completed")

# Add observed data to InferenceData

# Save InferenceData
print("\n[4/7] Saving InferenceData...")
idata_path = DIAG_DIR / "posterior_inference.netcdf"
trace.to_netcdf(idata_path)
print(f"  - Saved to: {idata_path}")
print(f"  - Groups: {list(trace.groups())}")

# Convergence diagnostics
print("\n[5/7] Computing convergence diagnostics...")
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'phi'])
print("\nPosterior Summary:")
print(summary)

# Save summary
summary_path = DIAG_DIR / "convergence_diagnostics.txt"
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CONVERGENCE DIAGNOSTICS - EXPERIMENT 1\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model: Negative Binomial Linear Model\n")
    f.write("PPL: PyMC (Stan compilation requires make which is not available)\n")
    f.write(f"Data: {len(data_df)} observations\n")
    f.write("Sampling: 4 chains x 1000 iterations (after 1000 warmup)\n\n")
    f.write("Parameter Summary:\n")
    f.write("-" * 80 + "\n")
    f.write(summary.to_string())
    f.write("\n\n")

    # Check convergence criteria
    f.write("Convergence Assessment:\n")
    f.write("-" * 80 + "\n")

    rhat_ok = (summary['r_hat'] < 1.01).all()
    ess_bulk_ok = (summary['ess_bulk'] > 400).all()
    ess_tail_ok = (summary['ess_tail'] > 400).all()

    f.write(f"R-hat < 1.01: {'PASS' if rhat_ok else 'FAIL'}\n")
    f.write(f"ESS bulk > 400: {'PASS' if ess_bulk_ok else 'FAIL'}\n")
    f.write(f"ESS tail > 400: {'PASS' if ess_tail_ok else 'FAIL'}\n")

    # Check divergences
    divergences = trace.sample_stats.diverging.sum().item()
    total_draws = len(trace.posterior.chain) * len(trace.posterior.draw)
    div_pct = 100 * divergences / total_draws
    f.write(f"Divergent transitions: {divergences}/{total_draws} ({div_pct:.2f}%)\n")
    f.write(f"Divergences < 1%: {'PASS' if div_pct < 1.0 else 'FAIL'}\n")

    # Overall assessment
    f.write("\n")
    if rhat_ok and ess_bulk_ok and ess_tail_ok and div_pct < 1.0:
        f.write("OVERALL: CONVERGENCE SUCCESS\n")
    else:
        f.write("OVERALL: CONVERGENCE ISSUES DETECTED\n")

print(f"  - Diagnostics saved to: {summary_path}")

# Check convergence criteria
rhat_ok = (summary['r_hat'] < 1.01).all()
ess_bulk_ok = (summary['ess_bulk'] > 400).all()
ess_tail_ok = (summary['ess_tail'] > 400).all()
divergences = trace.sample_stats.diverging.sum().item()
total_draws = len(trace.posterior.chain) * len(trace.posterior.draw)
div_pct = 100 * divergences / total_draws

print("\n  Convergence Criteria:")
print(f"    - R-hat < 1.01: {'PASS' if rhat_ok else 'FAIL'}")
print(f"    - ESS bulk > 400: {'PASS' if ess_bulk_ok else 'FAIL'}")
print(f"    - ESS tail > 400: {'PASS' if ess_tail_ok else 'FAIL'}")
print(f"    - Divergences: {divergences}/{total_draws} ({div_pct:.2f}%)")

if rhat_ok and ess_bulk_ok and ess_tail_ok and div_pct < 1.0:
    print("\n  CONVERGENCE: SUCCESS")
else:
    print("\n  CONVERGENCE: ISSUES DETECTED")

# Compute LOO
print("\n[6/7] Computing LOO-CV...")
loo = az.loo(trace, pointwise=True)
print(f"  - ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"  - p_loo: {loo.p_loo:.2f}")

# Check Pareto k diagnostics
pareto_k = loo.pareto_k.values
k_good = (pareto_k < 0.5).sum()
k_ok = ((pareto_k >= 0.5) & (pareto_k < 0.7)).sum()
k_bad = ((pareto_k >= 0.7) & (pareto_k < 1.0)).sum()
k_very_bad = (pareto_k >= 1.0).sum()

print(f"\n  Pareto k diagnostics:")
print(f"    - Good (k < 0.5): {k_good}/{len(pareto_k)}")
print(f"    - OK (0.5 <= k < 0.7): {k_ok}/{len(pareto_k)}")
print(f"    - Bad (0.7 <= k < 1.0): {k_bad}/{len(pareto_k)}")
print(f"    - Very bad (k >= 1.0): {k_very_bad}/{len(pareto_k)}")

# Save LOO results
loo_path = DIAG_DIR / "loo_results.txt"
with open(loo_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("LOO-CV RESULTS - EXPERIMENT 1\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}\n")
    f.write(f"p_loo: {loo.p_loo:.2f}\n\n")
    f.write("Pareto k Diagnostics:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Good (k < 0.5): {k_good}/{len(pareto_k)}\n")
    f.write(f"OK (0.5 <= k < 0.7): {k_ok}/{len(pareto_k)}\n")
    f.write(f"Bad (0.7 <= k < 1.0): {k_bad}/{len(pareto_k)}\n")
    f.write(f"Very bad (k >= 1.0): {k_very_bad}/{len(pareto_k)}\n")

print(f"  - LOO results saved to: {loo_path}")

# Save posterior samples
print("\n[7/7] Saving posterior samples...")
posterior_samples = []
for chain in range(len(trace.posterior.chain)):
    for draw in range(len(trace.posterior.draw)):
        posterior_samples.append({
            'beta_0': trace.posterior['beta_0'].values[chain, draw],
            'beta_1': trace.posterior['beta_1'].values[chain, draw],
            'phi': trace.posterior['phi'].values[chain, draw]
        })
posterior_df = pd.DataFrame(posterior_samples)
samples_path = DIAG_DIR / "posterior_samples.csv"
posterior_df.to_csv(samples_path, index=False)
print(f"  - Saved {len(posterior_df)} samples to: {samples_path}")

# Create visualizations
print("\n[8/7] Creating diagnostic plots...")

# 1. Trace plots
print("  - Creating trace plots...")
axes = az.plot_trace(
    trace,
    var_names=['beta_0', 'beta_1', 'phi'],
    compact=False,
    figsize=(14, 10)
)
# Update labels
var_labels = [r'$\beta_0$', r'$\beta_1$', r'$\phi$']
for i, label in enumerate(var_labels):
    axes[i, 0].set_ylabel(label, fontsize=12)

plt.tight_layout()
trace_path = PLOTS_DIR / "trace_plots.png"
plt.savefig(trace_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {trace_path}")

# 2. Posterior distributions
print("  - Creating posterior distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (var, label) in enumerate(zip(['beta_0', 'beta_1', 'phi'], var_labels)):
    az.plot_posterior(trace, var_names=[var], ax=axes[i], show=False)
    axes[i].set_title(f'Posterior: {label}', fontsize=12)

plt.tight_layout()
post_path = PLOTS_DIR / "posterior_distributions.png"
plt.savefig(post_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {post_path}")

# 3. Pairs plot
print("  - Creating pairs plot...")
axes = az.plot_pair(
    trace,
    var_names=['beta_0', 'beta_1', 'phi'],
    kind='kde',
    marginals=True,
    figsize=(12, 12)
)
pairs_path = PLOTS_DIR / "pairs_plot.png"
plt.savefig(pairs_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {pairs_path}")

# 4. Convergence summary (rank plots)
print("  - Creating convergence summary...")
axes = az.plot_rank(trace, var_names=['beta_0', 'beta_1', 'phi'],
                    figsize=(12, 8))
conv_path = PLOTS_DIR / "convergence_summary.png"
plt.savefig(conv_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {conv_path}")

# 5. Pareto k diagnostic plot
print("  - Creating Pareto k diagnostic plot...")
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_khat(loo, ax=ax, show_bins=True)
ax.set_title('Pareto k Diagnostic for LOO-CV', fontsize=14)
ax.set_xlabel('Data point', fontsize=12)
ax.set_ylabel('Pareto k', fontsize=12)
pareto_path = PLOTS_DIR / "pareto_k_diagnostics.png"
plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {pareto_path}")

# 6. Energy plot (PyMC specific diagnostic)
print("  - Creating energy plot...")
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_energy(trace, ax=ax)
ax.set_title('Energy Plot - MCMC Diagnostic', fontsize=14)
energy_path = PLOTS_DIR / "energy_plot.png"
plt.savefig(energy_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {energy_path}")

print("\n" + "=" * 80)
print("POSTERIOR INFERENCE COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - Diagnostics: {DIAG_DIR}")
print(f"  - Plots: {PLOTS_DIR}")
print("\nKey Results:")
print(f"  - beta_0: {summary.loc['beta_0', 'mean']:.3f} ± {summary.loc['beta_0', 'sd']:.3f}")
print(f"  - beta_1: {summary.loc['beta_1', 'mean']:.3f} ± {summary.loc['beta_1', 'sd']:.3f}")
print(f"  - phi: {summary.loc['phi', 'mean']:.3f} ± {summary.loc['phi', 'sd']:.3f}")
print(f"  - ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
