"""
Complete Pooling Model for Eight Schools Data

Model:
  y_i ~ Normal(mu, sigma_i)  [sigma_i known]
  mu ~ Normal(0, 25)

This is the simplest possible model - all schools share a single mean parameter mu.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
data = pd.read_csv('/workspace/data/data.csv')
print(f"Data shape: {data.shape}")
print(data)

y_obs = data['y'].values
sigma = data['sigma'].values
N = len(y_obs)

print(f"\nNumber of schools: {N}")
print(f"Observed effects: {y_obs}")
print(f"Standard errors: {sigma}")

# ============================================================
# 2. Classical Pooled Estimate (for comparison)
# ============================================================
# Weighted mean: sum(y_i/sigma_i^2) / sum(1/sigma_i^2)
weights = 1 / sigma**2
classical_mu = np.sum(y_obs * weights) / np.sum(weights)
classical_se = 1 / np.sqrt(np.sum(weights))

print(f"\nClassical pooled estimate:")
print(f"  mu = {classical_mu:.2f} ± {classical_se:.2f}")

# ============================================================
# 3. Build PyMC Model
# ============================================================
print("\nBuilding PyMC model...")
with pm.Model() as complete_pooling_model:
    # Prior for common mean
    mu = pm.Normal('mu', mu=0, sigma=25)

    # Likelihood: each school's observation comes from Normal(mu, sigma_i)
    # sigma_i is known (fixed data), not a parameter
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_obs)

    # Log-likelihood for LOO-CV (one value per observation)
    log_lik = pm.logp(y, y_obs)
    pm.Deterministic('log_lik', log_lik)

print("Model structure:")
print(complete_pooling_model)

# ============================================================
# 4. Sample from Posterior
# ============================================================
print("\nSampling from posterior...")
print("Configuration: 4 chains, 2000 iterations, 1000 warmup")

with complete_pooling_model:
    trace = pm.sample(
        draws=1000,           # Post-warmup samples
        tune=1000,            # Warmup samples
        chains=4,             # Number of chains
        cores=4,              # Parallel chains
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

print("\nSampling complete!")

# ============================================================
# 5. Save InferenceData
# ============================================================
output_file = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf'
trace.to_netcdf(output_file)
print(f"\nInferenceData saved to: {output_file}")

# ============================================================
# 6. Convergence Diagnostics
# ============================================================
print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

summary = az.summary(trace, var_names=['mu'])
print("\nPosterior Summary:")
print(summary)

# Extract diagnostics
rhat = summary['r_hat'].values[0]
ess_bulk = summary['ess_bulk'].values[0]
ess_tail = summary['ess_tail'].values[0]
mcse = summary['mcse_mean'].values[0]

print(f"\nKey Metrics:")
print(f"  R-hat: {rhat:.4f} (target: < 1.01)")
print(f"  ESS bulk: {ess_bulk:.0f} (target: > 400)")
print(f"  ESS tail: {ess_tail:.0f} (target: > 400)")
print(f"  MCSE: {mcse:.4f}")

# Check for divergences
divergences = trace.sample_stats.diverging.sum().values
print(f"  Divergent transitions: {divergences}")

# Convergence status
all_good = (rhat < 1.01) and (ess_bulk > 400) and (ess_tail > 400) and (divergences == 0)
if all_good:
    print("\nCONVERGENCE: EXCELLENT - All diagnostics passed!")
else:
    print("\nWARNING: Some diagnostics did not meet criteria")

# ============================================================
# 7. Posterior Statistics
# ============================================================
print("\n" + "="*60)
print("POSTERIOR INFERENCE")
print("="*60)

mu_samples = trace.posterior['mu'].values.flatten()
mu_mean = np.mean(mu_samples)
mu_sd = np.std(mu_samples)
mu_hdi = az.hdi(trace, var_names=['mu'], hdi_prob=0.95)['mu'].values

print(f"\nPosterior for mu:")
print(f"  Mean: {mu_mean:.2f}")
print(f"  SD: {mu_sd:.2f}")
print(f"  95% HDI: [{mu_hdi[0]:.2f}, {mu_hdi[1]:.2f}]")

print(f"\nComparison with classical estimate:")
print(f"  Bayesian: {mu_mean:.2f} ± {mu_sd:.2f}")
print(f"  Classical: {classical_mu:.2f} ± {classical_se:.2f}")
print(f"  Difference: {abs(mu_mean - classical_mu):.2f}")

# ============================================================
# 8. LOO Cross-Validation
# ============================================================
print("\n" + "="*60)
print("LOO CROSS-VALIDATION")
print("="*60)

loo = az.loo(trace)
print(f"\nLOO-CV Results:")
print(f"  ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"  p_eff: {loo.p_loo:.2f} (expected: ~1.0 for 1-parameter model)")

# Compare with Experiment 1
exp1_elpd = -30.73
exp1_se = 1.04
print(f"\nComparison with Experiment 1 (No Pooling):")
print(f"  Exp 1 ELPD: {exp1_elpd:.2f} ± {exp1_se:.2f}")
print(f"  Exp 2 ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"  Difference: {loo.elpd_loo - exp1_elpd:.2f}")

if loo.elpd_loo > exp1_elpd:
    print("  Complete pooling has BETTER predictive performance")
elif abs(loo.elpd_loo - exp1_elpd) < 2 * np.sqrt(exp1_se**2 + loo.se**2):
    print("  Models are statistically equivalent")
else:
    print("  No pooling has better predictive performance")

# ============================================================
# 9. Save Diagnostics to File
# ============================================================
diag_file = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/convergence_diagnostics.txt'
with open(diag_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("COMPLETE POOLING MODEL - CONVERGENCE DIAGNOSTICS\n")
    f.write("="*60 + "\n\n")

    f.write("Model: y_i ~ Normal(mu, sigma_i), mu ~ Normal(0, 25)\n")
    f.write(f"Data: {N} schools\n")
    f.write(f"Parameters: 1 (mu)\n\n")

    f.write("Sampling Configuration:\n")
    f.write("  Chains: 4\n")
    f.write("  Iterations: 2000 (1000 warmup + 1000 sampling)\n")
    f.write("  Total samples: 4000\n\n")

    f.write("-"*60 + "\n")
    f.write("CONVERGENCE METRICS\n")
    f.write("-"*60 + "\n\n")

    f.write(f"R-hat: {rhat:.4f} (target: < 1.01) - {'PASS' if rhat < 1.01 else 'FAIL'}\n")
    f.write(f"ESS bulk: {ess_bulk:.0f} (target: > 400) - {'PASS' if ess_bulk > 400 else 'FAIL'}\n")
    f.write(f"ESS tail: {ess_tail:.0f} (target: > 400) - {'PASS' if ess_tail > 400 else 'FAIL'}\n")
    f.write(f"MCSE: {mcse:.4f}\n")
    f.write(f"Divergent transitions: {divergences}\n\n")

    f.write("-"*60 + "\n")
    f.write("POSTERIOR INFERENCE\n")
    f.write("-"*60 + "\n\n")

    f.write(f"mu: {mu_mean:.2f} ± {mu_sd:.2f}\n")
    f.write(f"95% HDI: [{mu_hdi[0]:.2f}, {mu_hdi[1]:.2f}]\n\n")

    f.write("Classical pooled estimate:\n")
    f.write(f"mu: {classical_mu:.2f} ± {classical_se:.2f}\n\n")

    f.write("-"*60 + "\n")
    f.write("LOO CROSS-VALIDATION\n")
    f.write("-"*60 + "\n\n")

    f.write(f"ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}\n")
    f.write(f"p_eff: {loo.p_loo:.2f}\n\n")

    f.write("Comparison with Experiment 1:\n")
    f.write(f"  Exp 1 (No Pooling): {exp1_elpd:.2f} ± {exp1_se:.2f}\n")
    f.write(f"  Exp 2 (Complete Pooling): {loo.elpd_loo:.2f} ± {loo.se:.2f}\n")
    f.write(f"  Difference: {loo.elpd_loo - exp1_elpd:.2f}\n\n")

    f.write("-"*60 + "\n")
    f.write("OVERALL STATUS\n")
    f.write("-"*60 + "\n\n")

    if all_good:
        f.write("CONVERGENCE: EXCELLENT\n")
        f.write("All diagnostics passed. Model is ready for inference.\n")
    else:
        f.write("WARNING: Some diagnostics did not meet criteria\n")

print(f"\nDiagnostics saved to: {diag_file}")

# ============================================================
# 10. Visualizations
# ============================================================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

plot_dir = '/workspace/experiments/experiment_2/posterior_inference/plots'

# 10.1 Trace Plot
print("\n1. Trace plot for mu...")
fig = plt.figure(figsize=(12, 4))
az.plot_trace(trace, var_names=['mu'])
fig.suptitle('Trace Plot: mu', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{plot_dir}/trace_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {plot_dir}/trace_plot.png")

# 10.2 Posterior Density
print("2. Posterior density for mu...")
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_posterior(trace, var_names=['mu'], hdi_prob=0.95, ax=ax)
ax.axvline(classical_mu, color='red', linestyle='--', linewidth=2,
           label=f'Classical estimate: {classical_mu:.2f}')
ax.legend()
ax.set_title('Posterior Distribution: mu', fontsize=14)
plt.tight_layout()
plt.savefig(f'{plot_dir}/posterior_density.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {plot_dir}/posterior_density.png")

# 10.3 Rank Plot (for chain mixing)
print("3. Rank plot for convergence check...")
fig = plt.figure(figsize=(10, 4))
az.plot_rank(trace, var_names=['mu'])
plt.suptitle('Rank Plot: mu (uniform = good mixing)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{plot_dir}/rank_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {plot_dir}/rank_plot.png")

# 10.4 Forest Plot (comparing schools)
print("4. Forest plot showing all schools share same mu...")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot observed data points
school_labels = [f'School {i+1}' for i in range(N)]
ax.errorbar(y_obs, range(N), xerr=sigma, fmt='o', color='black',
            markersize=8, capsize=5, label='Observed data')

# Plot common mu estimate
ax.axvline(mu_mean, color='blue', linewidth=2, label=f'Pooled mu = {mu_mean:.1f}')
ax.axvspan(mu_hdi[0], mu_hdi[1], alpha=0.2, color='blue', label='95% HDI')

ax.set_yticks(range(N))
ax.set_yticklabels(school_labels)
ax.set_xlabel('Treatment Effect', fontsize=12)
ax.set_title('Complete Pooling: All Schools Share Common Mean', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plot_dir}/forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {plot_dir}/forest_plot.png")

# 10.5 Posterior Predictive Check
print("5. Posterior predictive check...")

# Generate posterior predictive samples
with complete_pooling_model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

# Extract predictions
y_pred_samples = ppc.posterior_predictive['y'].values.reshape(-1, N)

# Calculate summaries
y_pred_mean = y_pred_samples.mean(axis=0)
y_pred_std = y_pred_samples.std(axis=0)
y_pred_hdi = np.percentile(y_pred_samples, [2.5, 97.5], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Observed vs Predicted
ax = axes[0]
ax.errorbar(y_obs, y_pred_mean, xerr=sigma, yerr=y_pred_std,
            fmt='o', markersize=8, capsize=5, alpha=0.7)
for i in range(N):
    ax.text(y_obs[i], y_pred_mean[i], f' {i+1}', fontsize=9)

# Add diagonal line
lim_min = min(y_obs.min(), y_pred_mean.min()) - 5
lim_max = max(y_obs.max(), y_pred_mean.max()) + 5
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, label='Perfect prediction')

ax.set_xlabel('Observed y', fontsize=12)
ax.set_ylabel('Predicted y (posterior mean)', fontsize=12)
ax.set_title('Posterior Predictive Check: Observed vs Predicted', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Right: School-specific predictions
ax = axes[1]
ax.errorbar(range(N), y_obs, yerr=sigma, fmt='o', color='black',
            markersize=8, capsize=5, label='Observed', alpha=0.7)
ax.errorbar(range(N), y_pred_mean, yerr=y_pred_std, fmt='s', color='blue',
            markersize=6, capsize=5, label='Predicted', alpha=0.7)
ax.axhline(mu_mean, color='blue', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xticks(range(N))
ax.set_xticklabels([f'{i+1}' for i in range(N)])
ax.set_xlabel('School', fontsize=12)
ax.set_ylabel('Treatment Effect', fontsize=12)
ax.set_title('PPC: Each School vs Common Mean', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{plot_dir}/ppc_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {plot_dir}/ppc_plot.png")

# 10.6 Convergence Overview (combined diagnostics)
print("6. Convergence overview dashboard...")
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Trace
ax1 = fig.add_subplot(gs[0, 0])
for chain in range(4):
    chain_samples = trace.posterior['mu'].sel(chain=chain).values
    ax1.plot(chain_samples, alpha=0.7, label=f'Chain {chain+1}')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('mu')
ax1.set_title(f'Trace Plot (R-hat={rhat:.4f})')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Density
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(mu_samples, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax2.axvline(mu_mean, color='blue', linewidth=2, label=f'Mean={mu_mean:.2f}')
ax2.axvline(mu_hdi[0], color='blue', linestyle='--', linewidth=1)
ax2.axvline(mu_hdi[1], color='blue', linestyle='--', linewidth=1)
ax2.axvline(classical_mu, color='red', linestyle='--', linewidth=2,
            label=f'Classical={classical_mu:.2f}')
ax2.set_xlabel('mu')
ax2.set_ylabel('Density')
ax2.set_title(f'Posterior Density (SD={mu_sd:.2f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Autocorrelation
ax3 = fig.add_subplot(gs[1, 0])
az.plot_autocorr(trace, var_names=['mu'], combined=True, ax=ax3)
ax3.set_title('Autocorrelation')

# ESS comparison
ax4 = fig.add_subplot(gs[1, 1])
metrics = ['ESS Bulk', 'ESS Tail', 'Target']
values = [ess_bulk, ess_tail, 400]
colors = ['green' if v >= 400 else 'orange' for v in values[:2]] + ['gray']
bars = ax4.bar(metrics, values, color=colors, alpha=0.6, edgecolor='black')
ax4.axhline(400, color='red', linestyle='--', linewidth=2, label='Target: 400')
ax4.set_ylabel('ESS')
ax4.set_title('Effective Sample Size')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add text annotations
for i, (metric, value) in enumerate(zip(metrics[:2], values[:2])):
    ax4.text(i, value + 50, f'{value:.0f}', ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Convergence Diagnostics Overview', fontsize=16, y=0.995)
plt.savefig(f'{plot_dir}/convergence_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {plot_dir}/convergence_overview.png")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print("\nAll outputs saved to:")
print(f"  Code: /workspace/experiments/experiment_2/posterior_inference/code/")
print(f"  Diagnostics: /workspace/experiments/experiment_2/posterior_inference/diagnostics/")
print(f"  Plots: /workspace/experiments/experiment_2/posterior_inference/plots/")
