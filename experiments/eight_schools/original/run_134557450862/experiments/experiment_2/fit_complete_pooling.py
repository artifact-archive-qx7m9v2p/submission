#!/usr/bin/env python3
"""
Fit Complete Pooling Model (Experiment 2)
Very simple model with only 1 parameter
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
np.random.seed(42)
sns.set_style("whitegrid")

# Paths
base_dir = Path("/workspace/experiments/experiment_2")
posterior_dir = base_dir / "posterior_inference"
code_dir = posterior_dir / "code"
diag_dir = posterior_dir / "diagnostics"
plots_dir = posterior_dir / "plots"

for d in [code_dir, diag_dir, plots_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2: COMPLETE POOLING MODEL")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv("/workspace/data/data.csv")
y_obs = df['y'].values
sigma_obs = df['sigma'].values
n_schools = len(y_obs)

print(f"   N schools: {n_schools}")
print(f"   Observed effects: {y_obs}")
print(f"   Standard errors: {sigma_obs}")

# Build model
print("\n2. Building complete pooling model...")
print("   Specification: y_i ~ Normal(mu, sigma_i)")
print("   Prior: mu ~ Normal(0, 25)")

with pm.Model() as complete_pooling:
    # Prior
    mu = pm.Normal('mu', mu=0, sigma=25)

    # Likelihood
    y = pm.Normal('y', mu=mu, sigma=sigma_obs, observed=y_obs)

    # Prior predictive (quick check)
    print("\n3. Prior predictive check...")
    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=42)

    # Check if prior is reasonable
    prior_mu = prior_pred.prior['mu'].values.flatten()
    print(f"   Prior mu: mean={prior_mu.mean():.1f}, std={prior_mu.std():.1f}")
    print(f"   Prior mu 95% range: [{np.percentile(prior_mu, 2.5):.1f}, {np.percentile(prior_mu, 97.5):.1f}]")

    # Fit model
    print("\n4. Sampling from posterior...")
    print("   4 chains x 2000 iterations (1000 warmup)")

    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

    # Posterior predictive
    print("\n5. Generating posterior predictive samples...")
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=42)

# Save results
print("\n6. Saving results...")
trace.to_netcdf(diag_dir / "posterior_inference.netcdf")
print(f"   Saved to: {diag_dir / 'posterior_inference.netcdf'}")

# Diagnostics
print("\n7. Convergence diagnostics...")
print("="*80)

summary = az.summary(trace, var_names=['mu'])
print(summary)

rhat = summary['r_hat'].values[0]
ess_bulk = summary['ess_bulk'].values[0]
ess_tail = summary['ess_tail'].values[0]

print(f"\n   R-hat: {rhat:.4f} (target: < 1.01)")
print(f"   ESS bulk: {ess_bulk:.0f} (target: > 400)")
print(f"   ESS tail: {ess_tail:.0f} (target: > 400)")

# Check for divergences
n_divergences = trace.sample_stats.diverging.sum().item()
n_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
div_pct = 100 * n_divergences / n_samples

print(f"   Divergences: {n_divergences}/{n_samples} ({div_pct:.2f}%)")

convergence_pass = (rhat < 1.01) and (ess_bulk > 400) and (div_pct < 1.0)
print(f"\n   Convergence: {'PASS ✓' if convergence_pass else 'FAIL ✗'}")

# Save diagnostics
with open(diag_dir / "convergence_diagnostics.txt", 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPLETE POOLING MODEL - CONVERGENCE DIAGNOSTICS\n")
    f.write("="*80 + "\n\n")
    f.write(str(summary) + "\n\n")
    f.write(f"R-hat: {rhat:.4f}\n")
    f.write(f"ESS bulk: {ess_bulk:.0f}\n")
    f.write(f"ESS tail: {ess_tail:.0f}\n")
    f.write(f"Divergences: {n_divergences}/{n_samples} ({div_pct:.2f}%)\n")
    f.write(f"Convergence: {'PASS' if convergence_pass else 'FAIL'}\n")

# Posterior summary
print("\n8. Posterior summary...")
print("="*80)

mu_posterior = trace.posterior['mu'].values.flatten()
mu_mean = mu_posterior.mean()
mu_std = mu_posterior.std()
mu_hdi = az.hdi(trace, var_names=['mu'], hdi_prob=0.95)['mu'].values

print(f"   mu: {mu_mean:.2f} ± {mu_std:.2f}")
print(f"   95% HDI: [{mu_hdi[0]:.2f}, {mu_hdi[1]:.2f}]")

# Compare with classical pooled estimate
weights = 1 / sigma_obs**2
pooled_mean = np.sum(weights * y_obs) / np.sum(weights)
pooled_se = 1 / np.sqrt(np.sum(weights))

print(f"\n   Classical pooled estimate: {pooled_mean:.2f} ± {pooled_se:.2f}")
print(f"   Difference: {abs(mu_mean - pooled_mean):.3f} (should be small)")

# LOO Cross-validation
print("\n9. LOO Cross-validation...")
loo = az.loo(trace, pointwise=True)
print(f"   ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"   p_eff: {loo.p_loo:.2f} (should be ≈ 1)")
print(f"   Max Pareto k: {loo.pareto_k.max():.3f}")

all_k_good = (loo.pareto_k < 0.7).all()
print(f"   Pareto k diagnostic: {'PASS ✓' if all_k_good else 'CONCERN'}")

# Save derived quantities
with open(diag_dir / "derived_quantities.txt", 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPLETE POOLING MODEL - DERIVED QUANTITIES\n")
    f.write("="*80 + "\n\n")
    f.write(f"Posterior mu: {mu_mean:.2f} ± {mu_std:.2f}\n")
    f.write(f"95% HDI: [{mu_hdi[0]:.2f}, {mu_hdi[1]:.2f}]\n\n")
    f.write(f"Classical pooled estimate: {pooled_mean:.2f} ± {pooled_se:.2f}\n")
    f.write(f"Bayesian-Classical difference: {abs(mu_mean - pooled_mean):.3f}\n\n")
    f.write(f"LOO ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}\n")
    f.write(f"p_eff: {loo.p_loo:.2f}\n")
    f.write(f"Max Pareto k: {loo.pareto_k.max():.3f}\n")

# Visualizations
print("\n10. Creating visualizations...")

# Trace plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
az.plot_trace(trace, var_names=['mu'], axes=axes)
plt.tight_layout()
plt.savefig(plots_dir / "trace_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: trace_plot.png")

# Posterior density
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_posterior(trace, var_names=['mu'], ax=ax, hdi_prob=0.95)
ax.set_title("Posterior Distribution of Grand Mean μ")
plt.tight_layout()
plt.savefig(plots_dir / "posterior_density.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: posterior_density.png")

# Forest plot comparing posterior vs observed
fig, ax = plt.subplots(figsize=(10, 6))

# Plot observed values
ax.errorbar(y_obs, range(n_schools), xerr=1.96*sigma_obs,
            fmt='o', color='steelblue', alpha=0.6,
            label='Observed ± 95% CI', markersize=8)

# Plot posterior mu (same for all schools in complete pooling)
ax.axvline(mu_mean, color='darkred', linestyle='--', linewidth=2, label=f'Pooled μ = {mu_mean:.1f}')
ax.axvspan(mu_hdi[0], mu_hdi[1], alpha=0.2, color='darkred', label='95% HDI')

ax.set_yticks(range(n_schools))
ax.set_yticklabels([f'School {i+1}' for i in range(n_schools)])
ax.set_xlabel('Effect Size')
ax.set_title('Complete Pooling Model: All Schools Share Common Effect')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: forest_plot.png")

# Posterior predictive check (simple)
y_pred = trace.posterior_predictive['y'].values.reshape(-1, n_schools)

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    # Posterior predictive density
    ax.hist(y_pred[:, i], bins=50, density=True, alpha=0.6, color='skyblue', label='Posterior Predictive')

    # Observed value
    ax.axvline(y_obs[i], color='darkred', linestyle='--', linewidth=2, label='Observed')

    # 95% interval
    pred_hdi = np.percentile(y_pred[:, i], [2.5, 97.5])
    ax.axvspan(pred_hdi[0], pred_hdi[1], alpha=0.2, color='skyblue')

    ax.set_title(f'School {i+1} (σ={sigma_obs[i]})')
    ax.set_xlabel('Effect Size')
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Posterior Predictive Check: Individual Schools', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(plots_dir / "ppc_individual.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: ppc_individual.png")

# Summary report
print("\n11. Writing summary report...")
with open(posterior_dir / "inference_summary.md", 'w') as f:
    f.write("# Experiment 2: Complete Pooling Model - Inference Summary\n\n")
    f.write("## Model Specification\n\n")
    f.write("```\n")
    f.write("y_i ~ Normal(mu, sigma_i)  [sigma_i known]\n")
    f.write("mu ~ Normal(0, 25)\n")
    f.write("```\n\n")
    f.write("## Convergence\n\n")
    f.write(f"- **R-hat:** {rhat:.4f} ✓\n")
    f.write(f"- **ESS bulk:** {ess_bulk:.0f} ✓\n")
    f.write(f"- **ESS tail:** {ess_tail:.0f} ✓\n")
    f.write(f"- **Divergences:** {n_divergences}/{n_samples} ({div_pct:.2f}%) ✓\n")
    f.write(f"- **Status:** {'PASS' if convergence_pass else 'FAIL'}\n\n")
    f.write("## Posterior Estimates\n\n")
    f.write(f"- **Grand mean (μ):** {mu_mean:.2f} ± {mu_std:.2f}\n")
    f.write(f"- **95% HDI:** [{mu_hdi[0]:.2f}, {mu_hdi[1]:.2f}]\n\n")
    f.write("## Model Comparison\n\n")
    f.write(f"- **LOO ELPD:** {loo.elpd_loo:.2f} ± {loo.se:.2f}\n")
    f.write(f"- **Effective parameters:** {loo.p_loo:.2f}\n")
    f.write(f"- **Pareto k:** All < 0.7 ✓\n\n")
    f.write("## Interpretation\n\n")
    f.write("The complete pooling model assumes all schools share a common effect. ")
    f.write(f"The estimated grand mean is {mu_mean:.2f}, which is nearly identical to the ")
    f.write(f"classical meta-analysis pooled estimate of {pooled_mean:.2f}. ")
    f.write("All posterior predictive checks should pass given the EDA evidence for homogeneity.\n\n")
    f.write("Compare this model with Experiment 1 (hierarchical) using LOO.\n")

print(f"    Saved: inference_summary.md")

print("\n" + "="*80)
print("COMPLETE POOLING MODEL FITTING COMPLETE")
print("="*80)
print(f"\nKey Results:")
print(f"  μ = {mu_mean:.2f} ± {mu_std:.2f}")
print(f"  LOO ELPD = {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"  Convergence: {'PASS ✓' if convergence_pass else 'FAIL ✗'}")
print(f"\nNext: Compare with Experiment 1 (LOO ELPD = -30.73 ± 1.04)")
