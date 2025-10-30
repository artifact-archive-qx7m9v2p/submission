"""
Fit Negative Binomial Linear Model to Real Data
Experiment 1: Baseline model fitting with convergence diagnostics and LOO
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1")
DATA_PATH = Path("/workspace/data/data.csv")
MODEL_PATH = BASE_DIR / "simulation_based_validation/code/model.stan"
OUTPUT_DIR = BASE_DIR / "posterior_inference"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPERIMENT 1: NEGATIVE BINOMIAL LINEAR MODEL - POSTERIOR INFERENCE")
print("=" * 80)

# Load data
print("\n[1/7] Loading data...")
data_df = pd.read_csv(DATA_PATH)
print(f"  - Observations: {len(data_df)}")
print(f"  - Mean count: {data_df['C'].mean():.1f}")
print(f"  - Variance: {data_df['C'].var():.1f}")
print(f"  - Var/Mean ratio: {data_df['C'].var() / data_df['C'].mean():.2f}")

# Prepare data for Stan
stan_data = {
    'N': len(data_df),
    'year': data_df['year'].values,
    'C': data_df['C'].values.astype(int)
}

# Compile model
print("\n[2/7] Compiling Stan model...")
model = cmdstanpy.CmdStanModel(stan_file=str(MODEL_PATH))
print(f"  - Model: {MODEL_PATH.name}")
print(f"  - Compiled successfully")

# Sample from posterior
print("\n[3/7] Sampling from posterior...")
print("  Configuration:")
print("    - Chains: 4")
print("    - Iterations per chain: 2000 (1000 warmup + 1000 sampling)")
print("    - adapt_delta: 0.95")
print("    - max_treedepth: 12")

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    max_treedepth=12,
    seed=42,
    show_progress=True,
    show_console=False
)

print("  - Sampling completed")

# Check for warnings
print("\n[4/7] Checking for sampling warnings...")
diagnose = fit.diagnose()
if diagnose:
    print("  WARNING: Sampling diagnostics detected issues:")
    print(diagnose)
else:
    print("  - No warnings detected")

# Convert to ArviZ InferenceData with log_likelihood
print("\n[5/7] Converting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    posterior_predictive={'C_rep': 'C'},
    log_likelihood='log_lik',
    observed_data={'C': stan_data['C']}
)
print("  - Conversion successful")
print(f"  - Groups: {list(idata.groups())}")

# Save InferenceData
idata_path = DIAG_DIR / "posterior_inference.netcdf"
idata.to_netcdf(idata_path)
print(f"  - Saved to: {idata_path}")

# Convergence diagnostics
print("\n[6/7] Computing convergence diagnostics...")
summary = az.summary(idata, var_names=['beta_0', 'beta_1', 'phi'])
print("\nPosterior Summary:")
print(summary)

# Save summary
summary_path = DIAG_DIR / "convergence_diagnostics.txt"
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CONVERGENCE DIAGNOSTICS - EXPERIMENT 1\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model: Negative Binomial Linear Model\n")
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
    divergences = fit.method_variables()['divergent__'].sum()
    total_draws = fit.draws().shape[0] * fit.draws().shape[1]
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
divergences = fit.method_variables()['divergent__'].sum()
total_draws = fit.draws().shape[0] * fit.draws().shape[1]
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
print("\n[7/7] Computing LOO-CV...")
loo = az.loo(idata, pointwise=True)
print(f"  - ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"  - p_loo: {loo.p_loo:.2f}")

# Check Pareto k diagnostics
pareto_k = loo.pareto_k
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
print("\n[8/7] Saving posterior samples...")
posterior_df = fit.draws_pd(vars=['beta_0', 'beta_1', 'phi'])
samples_path = DIAG_DIR / "posterior_samples.csv"
posterior_df.to_csv(samples_path, index=False)
print(f"  - Saved {len(posterior_df)} samples to: {samples_path}")

# Create visualizations
print("\n[9/7] Creating diagnostic plots...")

# 1. Trace plots
print("  - Creating trace plots...")
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
var_names = ['beta_0', 'beta_1', 'phi']
var_labels = [r'$\beta_0$', r'$\beta_1$', r'$\phi$']

for i, (var, label) in enumerate(zip(var_names, var_labels)):
    # Trace plot
    az.plot_trace(idata, var_names=[var], axes=axes[i:i+1, :],
                  compact=False, show=False)
    axes[i, 0].set_title(f'Trace: {label}')
    axes[i, 1].set_title(f'Posterior: {label}')

plt.tight_layout()
trace_path = PLOTS_DIR / "trace_plots.png"
plt.savefig(trace_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {trace_path}")

# 2. Posterior distributions
print("  - Creating posterior distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (var, label) in enumerate(zip(var_names, var_labels)):
    az.plot_posterior(idata, var_names=[var], ax=axes[i], show=False)
    axes[i].set_title(f'Posterior: {label}', fontsize=12)

plt.tight_layout()
post_path = PLOTS_DIR / "posterior_distributions.png"
plt.savefig(post_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {post_path}")

# 3. Pairs plot
print("  - Creating pairs plot...")
axes = az.plot_pair(
    idata,
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
axes = az.plot_rank(idata, var_names=['beta_0', 'beta_1', 'phi'],
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

print("\n" + "=" * 80)
print("POSTERIOR INFERENCE COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - Diagnostics: {DIAG_DIR}")
print(f"  - Plots: {PLOTS_DIR}")
