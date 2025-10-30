#!/usr/bin/env python3
"""
Fit Beta-Binomial Model (Experiment 3) using PyMC with MCMC
============================================================

Model: Beta-Binomial marginal model (non-hierarchical)
Priors: mu_p ~ Beta(5, 50), kappa ~ Gamma(2, 0.1)
Likelihood: r ~ Beta-Binomial(n, alpha, beta)

Key differences from Experiment 1:
- No hierarchical structure (2 parameters vs 14)
- Works on probability scale (no logit transform)
- Natural overdispersion via beta-binomial
- Expected to be ~2x faster
"""

import sys
import time
import warnings
import pandas as pd
import numpy as np

# Add PyMC path
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Load Data
# ============================================================================
print("="*80)
print("EXPERIMENT 3: Beta-Binomial Model (Simple Alternative)")
print("="*80)

data = pd.read_csv('/workspace/data/data.csv')
J = len(data)
n = data['n'].values
r = data['r'].values

print(f"\nData Summary:")
print(f"  Groups: {J}")
print(f"  Total trials: {n.sum()}")
print(f"  Total successes: {r.sum()}")
print(f"  Observed pooled rate: {r.sum()/n.sum():.4f}")
print(f"  Observed overdispersion φ: {np.var(r/n) / (np.mean(r/n)*(1-np.mean(r/n))/np.mean(n)):.2f}")

# ============================================================================
# Build Model
# ============================================================================
print("\n" + "="*80)
print("Building Beta-Binomial Model")
print("="*80)

with pm.Model() as model:
    # Priors
    mu_p = pm.Beta('mu_p', alpha=5, beta=50)  # Mean success probability
    kappa = pm.Gamma('kappa', alpha=2, beta=0.1)  # Concentration parameter

    # Transformed parameters
    alpha = pm.Deterministic('alpha', mu_p * kappa)
    beta_param = pm.Deterministic('beta', (1 - mu_p) * kappa)

    # Overdispersion parameter
    phi = pm.Deterministic('phi', 1 / (kappa + 1))

    # Likelihood - Beta-Binomial
    y_obs = pm.BetaBinomial('y_obs', alpha=alpha, beta=beta_param, n=n, observed=r)

    # Log-likelihood for each observation (CRITICAL for LOO comparison)
    # Need to compute pointwise log-likelihood
    pm.Deterministic('log_lik', pm.logp(y_obs, r))

print("\nModel Structure:")
print(f"  Priors:")
print(f"    mu_p ~ Beta(5, 50)")
print(f"    kappa ~ Gamma(2, 0.1)")
print(f"  Derived:")
print(f"    alpha = mu_p × kappa")
print(f"    beta = (1 - mu_p) × kappa")
print(f"    phi = 1 / (kappa + 1)")
print(f"  Likelihood:")
print(f"    r_j ~ Beta-Binomial(n_j, alpha, beta) for j=1..{J}")

# ============================================================================
# Initial Sampling Attempt (Adaptive Strategy)
# ============================================================================
print("\n" + "="*80)
print("MCMC Sampling - Initial Attempt")
print("="*80)
print("\nConfiguration:")
print("  Chains: 4")
print("  Warmup (tune): 1000")
print("  Sampling iterations: 1000")
print("  Target accept: 0.90")
print("  Sampler: NUTS (default)")

start_time = time.time()

with model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.90,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

sampling_time = time.time() - start_time

print(f"\nSampling completed in {sampling_time:.1f} seconds")
print(f"  Expected ~30-60 sec, actual: {sampling_time:.1f} sec")
print(f"  Speedup vs Exp 1 (~90 sec): {90/sampling_time:.2f}×")

# ============================================================================
# Convergence Diagnostics
# ============================================================================
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

# Get summary statistics
summary = az.summary(trace, var_names=['mu_p', 'kappa', 'alpha', 'beta', 'phi'])
print("\nParameter Summary:")
print(summary.to_string())

# Extract convergence metrics
rhat_max = summary['r_hat'].max()
ess_bulk_min = summary['ess_bulk'].min()
ess_tail_min = summary['ess_tail'].min()

# Count divergences
divergences = trace.sample_stats.diverging.sum().item()
total_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
divergence_pct = 100 * divergences / total_samples

print("\n" + "-"*80)
print("CONVERGENCE CRITERIA CHECK")
print("-"*80)
print(f"  Worst R-hat: {rhat_max:.4f} (criterion: < 1.01) {'✓ PASS' if rhat_max < 1.01 else '✗ FAIL'}")
print(f"  Minimum ESS (bulk): {ess_bulk_min:.0f} (criterion: > 400) {'✓ PASS' if ess_bulk_min > 400 else '⚠ MARGINAL' if ess_bulk_min > 300 else '✗ FAIL'}")
print(f"  Minimum ESS (tail): {ess_tail_min:.0f} (criterion: > 400) {'✓ PASS' if ess_tail_min > 400 else '⚠ MARGINAL' if ess_tail_min > 300 else '✗ FAIL'}")
print(f"  Divergences: {divergences}/{total_samples} ({divergence_pct:.2f}%) {'✓ PASS' if divergence_pct < 1 else '⚠ INVESTIGATE' if divergence_pct < 5 else '✗ FAIL'}")

# Overall assessment
convergence_pass = (rhat_max < 1.01 and ess_bulk_min > 400 and
                   ess_tail_min > 400 and divergence_pct < 1)
convergence_marginal = (rhat_max < 1.05 and ess_bulk_min > 300 and
                       ess_tail_min > 300 and divergence_pct < 5)

if convergence_pass:
    print("\n✓ OVERALL: PASS - All convergence criteria met")
    decision = "PASS"
elif convergence_marginal:
    print("\n⚠ OVERALL: MARGINAL - Some criteria not met but acceptable")
    decision = "INVESTIGATE"
else:
    print("\n✗ OVERALL: FAIL - Convergence issues detected")
    decision = "FAIL"

# ============================================================================
# Check for Boundary Issues
# ============================================================================
print("\n" + "-"*80)
print("BOUNDARY CHECK")
print("-"*80)

mu_p_samples = trace.posterior['mu_p'].values.flatten()
mu_p_near_zero = np.sum(mu_p_samples < 0.01) / len(mu_p_samples) * 100
mu_p_near_one = np.sum(mu_p_samples > 0.99) / len(mu_p_samples) * 100

print(f"  mu_p samples near 0 (<0.01): {mu_p_near_zero:.2f}%")
print(f"  mu_p samples near 1 (>0.99): {mu_p_near_one:.2f}%")

if mu_p_near_zero > 10 or mu_p_near_one > 10:
    print("  ✗ WARNING: mu_p hitting boundaries - potential model misspecification")
else:
    print("  ✓ No boundary issues detected")

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save InferenceData with log-likelihood
idata_path = '/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf'
trace.to_netcdf(idata_path)
print(f"\n✓ Saved InferenceData to: {idata_path}")
print(f"  Contains log_likelihood group: {hasattr(trace, 'log_likelihood')}")
print(f"  Shape: {trace.posterior.dims}")

# Save summary table
summary_path = '/workspace/experiments/experiment_3/posterior_inference/diagnostics/summary_table.csv'
summary.to_csv(summary_path)
print(f"✓ Saved summary table to: {summary_path}")

# Save convergence report
report_path = '/workspace/experiments/experiment_3/posterior_inference/diagnostics/convergence_report.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXPERIMENT 3: Beta-Binomial Model - Convergence Report\n")
    f.write("="*80 + "\n\n")

    f.write(f"Sampling Date: {pd.Timestamp.now()}\n")
    f.write(f"Sampling Time: {sampling_time:.1f} seconds\n")
    f.write(f"Total Samples: {total_samples}\n\n")

    f.write("-"*80 + "\n")
    f.write("CONVERGENCE METRICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Worst R-hat: {rhat_max:.4f} (criterion: < 1.01)\n")
    f.write(f"Minimum ESS (bulk): {ess_bulk_min:.0f} (criterion: > 400)\n")
    f.write(f"Minimum ESS (tail): {ess_tail_min:.0f} (criterion: > 400)\n")
    f.write(f"Divergences: {divergences}/{total_samples} ({divergence_pct:.2f}%)\n\n")

    f.write(f"Decision: {decision}\n\n")

    f.write("-"*80 + "\n")
    f.write("PARAMETER ESTIMATES\n")
    f.write("-"*80 + "\n")
    f.write(summary.to_string())
    f.write("\n\n")

    f.write("-"*80 + "\n")
    f.write("BOUNDARY CHECK\n")
    f.write("-"*80 + "\n")
    f.write(f"mu_p near 0 (<0.01): {mu_p_near_zero:.2f}%\n")
    f.write(f"mu_p near 1 (>0.99): {mu_p_near_one:.2f}%\n\n")

    f.write("-"*80 + "\n")
    f.write("COMPARISON TO EXPERIMENT 1\n")
    f.write("-"*80 + "\n")
    f.write(f"Parameters: 2 (vs 14 in Exp 1)\n")
    f.write(f"Sampling time: {sampling_time:.1f} sec (vs ~90 sec in Exp 1)\n")
    f.write(f"Speedup: {90/sampling_time:.2f}×\n")
    f.write(f"Model type: Marginal (vs Hierarchical in Exp 1)\n")
    f.write(f"LOO: To be computed (Exp 1 failed with k > 0.7)\n\n")

print(f"✓ Saved convergence report to: {report_path}")

# ============================================================================
# Create Diagnostic Plots
# ============================================================================
print("\n" + "="*80)
print("CREATING DIAGNOSTIC PLOTS")
print("="*80)

# 1. Trace plots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Trace Plots - Beta-Binomial Model', fontsize=14, fontweight='bold')

params = ['mu_p', 'kappa', 'phi']
for i, param in enumerate(params):
    # Trace plot
    for chain in range(4):
        axes[i, 0].plot(trace.posterior[param].sel(chain=chain), alpha=0.7, label=f'Chain {chain+1}')
    axes[i, 0].set_ylabel(param, fontweight='bold')
    axes[i, 0].set_xlabel('Iteration')
    if i == 0:
        axes[i, 0].legend(loc='upper right', fontsize=8)
    axes[i, 0].grid(True, alpha=0.3)

    # Posterior distribution
    all_samples = trace.posterior[param].values.flatten()
    axes[i, 1].hist(all_samples, bins=50, alpha=0.7, edgecolor='black')
    axes[i, 1].axvline(all_samples.mean(), color='red', linestyle='--',
                       label=f'Mean: {all_samples.mean():.4f}')
    axes[i, 1].set_xlabel(param, fontweight='bold')
    axes[i, 1].set_ylabel('Frequency')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
trace_path = '/workspace/experiments/experiment_3/posterior_inference/diagnostics/trace_plots.png'
plt.savefig(trace_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved trace plots to: {trace_path}")
plt.close()

# 2. Rank plots (check mixing)
fig = plt.figure(figsize=(12, 8))
az.plot_rank(trace, var_names=['mu_p', 'kappa', 'phi'], figsize=(12, 8))
plt.suptitle('Rank Plots - Chain Mixing Check', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
rank_path = '/workspace/experiments/experiment_3/posterior_inference/plots/rank_plots.png'
plt.savefig(rank_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved rank plots to: {rank_path}")
plt.close()

# 3. Posterior predictive vs observed
print("\n" + "-"*80)
print("POSTERIOR PREDICTIVE CHECK")
print("-"*80)

# Compute posterior predictive for each group
mu_p_post = trace.posterior['mu_p'].values.flatten()
kappa_post = trace.posterior['kappa'].values.flatten()
phi_post = trace.posterior['phi'].values.flatten()

# For each group, compute predicted rate distribution
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for j in range(J):
    # Sample from beta-binomial for this group
    n_j = n[j]
    r_j = r[j]
    obs_rate = r_j / n_j

    # Sample from posterior predictive
    n_samples = len(mu_p_post)
    pred_rates = np.zeros(n_samples)

    for i in range(min(1000, n_samples)):  # Subsample for speed
        alpha_i = mu_p_post[i] * kappa_post[i]
        beta_i = (1 - mu_p_post[i]) * kappa_post[i]
        # Draw p from beta, then r from binomial
        p_i = np.random.beta(alpha_i, beta_i)
        r_pred = np.random.binomial(n_j, p_i)
        pred_rates[i] = r_pred / n_j

    # Plot
    axes[j].hist(pred_rates[:1000], bins=30, alpha=0.5, density=True,
                 label='Posterior Pred', edgecolor='black')
    axes[j].axvline(obs_rate, color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {obs_rate:.3f}')
    axes[j].set_title(f'Group {j+1} (n={n_j}, r={r_j})', fontsize=10)
    axes[j].set_xlabel('Rate')
    axes[j].set_ylabel('Density')
    if j == 0:
        axes[j].legend(fontsize=8)
    axes[j].grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Check - All Groups', fontsize=14, fontweight='bold')
plt.tight_layout()
pp_path = '/workspace/experiments/experiment_3/posterior_inference/plots/posterior_predictive.png'
plt.savefig(pp_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved posterior predictive plot to: {pp_path}")
plt.close()

# 4. Parameter distributions (publication-quality)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# mu_p
az.plot_posterior(trace, var_names=['mu_p'], ax=axes[0], point_estimate='mean')
axes[0].set_title('Mean Success Probability (mu_p)', fontweight='bold')

# kappa
az.plot_posterior(trace, var_names=['kappa'], ax=axes[1], point_estimate='mean')
axes[1].set_title('Concentration Parameter (kappa)', fontweight='bold')

# phi
az.plot_posterior(trace, var_names=['phi'], ax=axes[2], point_estimate='mean')
axes[2].axvline(0.036, color='red', linestyle='--', alpha=0.7, label='Observed φ ≈ 3.6%')
axes[2].set_title('Overdispersion Parameter (phi)', fontweight='bold')
axes[2].legend()

plt.tight_layout()
param_path = '/workspace/experiments/experiment_3/posterior_inference/plots/parameter_distributions.png'
plt.savefig(param_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved parameter distributions to: {param_path}")
plt.close()

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nModel: Beta-Binomial (Simple Alternative)")
print(f"Parameters: {len(['mu_p', 'kappa'])} free parameters")
print(f"Sampling time: {sampling_time:.1f} seconds")
print(f"Convergence: {decision}")

print(f"\nParameter Estimates:")
print(f"  mu_p: {summary.loc['mu_p', 'mean']:.4f} ± {summary.loc['mu_p', 'sd']:.4f}")
print(f"  kappa: {summary.loc['kappa', 'mean']:.1f} ± {summary.loc['kappa', 'sd']:.1f}")
print(f"  phi: {summary.loc['phi', 'mean']:.4f} ± {summary.loc['phi', 'sd']:.4f}")
print(f"    (Observed φ ≈ 0.036 or 3.6%)")

print(f"\nComparison to Experiment 1:")
print(f"  Simpler: 2 params vs 14")
print(f"  Faster: {sampling_time:.1f} sec vs ~90 sec ({90/sampling_time:.2f}× speedup)")
print(f"  LOO: To be computed (critical test)")

print(f"\nNext Steps:")
if decision == "PASS":
    print("  ✓ Proceed to posterior predictive check")
    print("  ✓ Compute LOO and compare to Experiment 1")
elif decision == "INVESTIGATE":
    print("  ⚠ Proceed with caution - document concerns")
    print("  ⚠ Compute LOO but note marginal convergence")
else:
    print("  ✗ Address convergence issues before proceeding")
    print("  ✗ Consider adaptive sampling strategy")

print("\n" + "="*80)
print("EXPERIMENT 3 COMPLETE")
print("="*80)
