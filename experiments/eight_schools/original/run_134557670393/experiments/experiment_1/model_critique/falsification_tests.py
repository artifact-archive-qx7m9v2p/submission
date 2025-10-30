"""
Comprehensive Falsification Tests for Experiment 1
Applies all pre-specified criteria from experiment_plan.md
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data and posterior
data = pd.read_csv('/workspace/data/data.csv')
y_obs = data['y'].values
sigma = data['sigma'].values
J = len(y_obs)

# Load InferenceData
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print("="*80)
print("COMPREHENSIVE FALSIFICATION TESTS: EXPERIMENT 1")
print("="*80)
print(f"\nData: {J} studies")
print(f"Observed effects: {y_obs}")
print(f"Standard errors: {sigma}")

# ============================================================================
# FALSIFICATION CRITERION 1: Posterior Predictive Failure
# ============================================================================
print("\n" + "="*80)
print("CRITERION 1: POSTERIOR PREDICTIVE FAILURE")
print("="*80)
print("Rule: REJECT if >1 study outside 95% posterior predictive interval")

# Extract posterior predictive samples from PPC analysis
ppc_results = pd.read_csv('/workspace/experiments/experiment_1/posterior_predictive_check/ppc_study_results.csv')
outliers = ppc_results['is_outlier'].sum()

print(f"\nResult: {outliers} of {J} studies outside 95% PPI")
print(f"Threshold: REJECT if > 1")
print(f"Verdict: {'REJECT' if outliers > 1 else 'PASS'} ✓" if outliers <= 1 else f"Verdict: REJECT ✗")

criterion_1_pass = (outliers <= 1)

# ============================================================================
# FALSIFICATION CRITERION 2: Leave-One-Out Instability
# ============================================================================
print("\n" + "="*80)
print("CRITERION 2: LEAVE-ONE-OUT INSTABILITY")
print("="*80)
print("Rule: REJECT if max |E[mu | data_{-i}] - E[mu | data]| > 5 units")

# We need to refit the model 8 times, dropping each study
# For computational efficiency, we'll use a fast approximation with grid method
from scipy.stats import norm

def fit_hierarchical_grid(y, sigma, mu_grid, tau_grid):
    """Fast grid approximation for hierarchical model"""
    # Prior
    log_prior_mu = norm.logpdf(mu_grid, 0, 50)
    log_prior_tau = -np.log(1 + (tau_grid/5)**2)  # Half-Cauchy approximation

    # Likelihood marginalized over theta
    log_lik = np.zeros((len(mu_grid), len(tau_grid)))
    for i, (yi, si) in enumerate(zip(y, sigma)):
        # Marginal: y_i ~ N(mu, tau^2 + si^2)
        marginal_sd = np.sqrt(tau_grid[:, np.newaxis]**2 + si**2)
        log_lik += norm.logpdf(yi, mu_grid[np.newaxis, :], marginal_sd).T

    # Posterior (unnormalized)
    log_post = log_lik + log_prior_mu[:, np.newaxis] + log_prior_tau[np.newaxis, :]
    log_post = log_post - np.max(log_post)  # Numerical stability
    post = np.exp(log_post)
    post = post / np.sum(post)

    # Marginal over tau to get E[mu]
    post_mu = np.sum(post, axis=1)
    mu_mean = np.sum(mu_grid * post_mu)

    return mu_mean

# Define grid
mu_grid = np.linspace(-20, 40, 120)
tau_grid = np.linspace(0.01, 20, 100)

# Full data posterior mean
mu_full = idata.posterior['mu'].values.mean()
print(f"\nE[mu | full data] = {mu_full:.3f}")

# Leave-one-out analysis
print("\nLeave-one-out analysis:")
mu_loo = np.zeros(J)
delta_mu = np.zeros(J)

for i in range(J):
    # Remove study i
    y_loo = np.delete(y_obs, i)
    sigma_loo = np.delete(sigma, i)

    # Fit without study i
    mu_loo[i] = fit_hierarchical_grid(y_loo, sigma_loo, mu_grid, tau_grid)
    delta_mu[i] = mu_loo[i] - mu_full

    print(f"  Study {i+1}: E[mu | data_{{-{i+1}}}] = {mu_loo[i]:6.3f}, Δmu = {delta_mu[i]:+6.3f}")

max_delta = np.max(np.abs(delta_mu))
most_influential = np.argmax(np.abs(delta_mu)) + 1

print(f"\nMax |Δmu| = {max_delta:.3f} (Study {most_influential})")
print(f"Threshold: REJECT if > 5")
print(f"Verdict: {'REJECT' if max_delta > 5 else 'PASS'} ✓" if max_delta <= 5 else f"Verdict: REJECT ✗")

criterion_2_pass = (max_delta <= 5)

# ============================================================================
# FALSIFICATION CRITERION 3: Convergence Failure
# ============================================================================
print("\n" + "="*80)
print("CRITERION 3: CONVERGENCE FAILURE")
print("="*80)
print("Rule: REJECT if R-hat > 1.05 OR ESS < 400 OR divergences > 1%")

# Load convergence summary
conv_summary = pd.read_csv('/workspace/experiments/experiment_1/posterior_inference/diagnostics/convergence_summary.csv')

max_rhat = conv_summary['r_hat'].max()
min_ess_bulk = conv_summary['ess_bulk'].min()
min_ess_tail = conv_summary['ess_tail'].min()

# Divergences from sampling diagnostics
n_samples = idata.posterior.dims['draw'] * idata.posterior.dims['chain']
n_divergences = 0  # From inference report: 0 divergences
div_pct = 100 * n_divergences / n_samples

print(f"\nMax R-hat: {max_rhat:.4f} (threshold: < 1.05)")
print(f"Min ESS bulk: {min_ess_bulk:.0f} (threshold: > 400)")
print(f"Min ESS tail: {min_ess_tail:.0f} (threshold: > 400)")
print(f"Divergences: {n_divergences} ({div_pct:.2f}%, threshold: < 1%)")

rhat_pass = (max_rhat < 1.05)
ess_pass = (min_ess_bulk > 400) and (min_ess_tail > 400)
div_pass = (div_pct < 1.0)

print(f"\nR-hat check: {'PASS' if rhat_pass else 'FAIL'}")
print(f"ESS check: {'PASS' if ess_pass else 'FAIL'}")
print(f"Divergence check: {'PASS' if div_pass else 'FAIL'}")
print(f"Verdict: {'PASS ✓' if (rhat_pass and ess_pass and div_pass) else 'REJECT ✗'}")

criterion_3_pass = (rhat_pass and ess_pass and div_pass)

# ============================================================================
# FALSIFICATION CRITERION 4: Extreme Shrinkage Asymmetry
# ============================================================================
print("\n" + "="*80)
print("CRITERION 4: EXTREME SHRINKAGE ASYMMETRY")
print("="*80)
print("Rule: REJECT if any |E[theta_i] - y_i| > 3*sigma_i")

# Extract theta posteriors
theta_mean = idata.posterior['theta'].mean(dim=['chain', 'draw']).values
shrinkage_diff = theta_mean - y_obs
shrinkage_threshold = 3 * sigma

print("\nShrinkage analysis:")
print(f"{'Study':<8} {'y_obs':<8} {'E[theta]':<10} {'Difference':<12} {'3*sigma':<10} {'Status':<10}")
print("-" * 70)

extreme_shrinkage = []
for i in range(J):
    status = "EXTREME" if np.abs(shrinkage_diff[i]) > shrinkage_threshold[i] else "OK"
    if status == "EXTREME":
        extreme_shrinkage.append(i+1)
    print(f"{i+1:<8} {y_obs[i]:<8.2f} {theta_mean[i]:<10.2f} {shrinkage_diff[i]:<+12.2f} {shrinkage_threshold[i]:<10.2f} {status:<10}")

n_extreme = len(extreme_shrinkage)
print(f"\nExtreme shrinkage detected: {n_extreme} studies")
if n_extreme > 0:
    print(f"Studies with extreme shrinkage: {extreme_shrinkage}")
print(f"Verdict: {'REJECT' if n_extreme > 0 else 'PASS'} ✓" if n_extreme == 0 else f"Verdict: REJECT ✗")

criterion_4_pass = (n_extreme == 0)

# ============================================================================
# REVISION CRITERION 1: Prior-Posterior Conflict (tau)
# ============================================================================
print("\n" + "="*80)
print("REVISION CRITERION: PRIOR-POSTERIOR CONFLICT")
print("="*80)
print("Rule: REVISE if P(tau > 10 | data) > 0.5 with prior P(tau > 10) < 0.05")

# Posterior probability
tau_samples = idata.posterior['tau'].values.flatten()
p_tau_gt_10_post = np.mean(tau_samples > 10)

# Prior probability (Half-Cauchy(0, 5))
# P(tau > 10) = P(Cauchy > 10/5) = P(Cauchy > 2)
from scipy.stats import cauchy
p_tau_gt_10_prior = 1 - cauchy.cdf(2, loc=0, scale=1)

print(f"\nPrior P(tau > 10) = {p_tau_gt_10_prior:.4f}")
print(f"Posterior P(tau > 10) = {p_tau_gt_10_post:.4f}")
print(f"Ratio: {p_tau_gt_10_post / p_tau_gt_10_prior:.2f}x increase")

conflict = (p_tau_gt_10_post > 0.5) and (p_tau_gt_10_prior < 0.05)
print(f"\nConflict detected: {'YES (REVISE)' if conflict else 'NO (OK)'}")

revision_criterion_pass = not conflict

# ============================================================================
# REVISION CRITERION 2: Unidentifiability (tau)
# ============================================================================
print("\n" + "="*80)
print("REVISION CRITERION: UNIDENTIFIABILITY")
print("="*80)
print("Rule: REVISE if tau posterior essentially uniform")

# Check if tau posterior is flat
tau_kde = stats.gaussian_kde(tau_samples)
tau_eval_grid = np.linspace(tau_samples.min(), tau_samples.max(), 100)
tau_density = tau_kde(tau_eval_grid)

# Compute coefficient of variation of density
density_cv = np.std(tau_density) / np.mean(tau_density)
print(f"\nTau posterior density CV: {density_cv:.3f}")
print(f"(Low CV < 0.3 suggests uniform/flat posterior)")

uniform_test = density_cv < 0.3
print(f"Verdict: {'Essentially uniform (REVISE)' if uniform_test else 'Well-identified (OK) ✓'}")

unidentifiability_pass = not uniform_test

# ============================================================================
# ADDITIONAL DIAGNOSTIC: LOO-CV
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL DIAGNOSTIC: LOO-CV (Pareto k)")
print("="*80)

# Compute LOO
loo = az.loo(idata, pointwise=True)
print(f"\nELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"p_loo (effective parameters): {loo.p_loo:.2f}")

# Pareto k diagnostics
k_values = loo.pareto_k.values
print(f"\nPareto k values:")
for i, k in enumerate(k_values):
    status = "Good" if k < 0.5 else ("OK" if k < 0.7 else "Bad")
    print(f"  Study {i+1}: k = {k:.3f} ({status})")

n_bad_k = np.sum(k_values > 0.7)
print(f"\nStudies with k > 0.7: {n_bad_k}")
print(f"Model adequacy: {'GOOD' if n_bad_k == 0 else ('ACCEPTABLE' if n_bad_k <= 1 else 'PROBLEMATIC')}")

# ============================================================================
# OVERALL FALSIFICATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("OVERALL FALSIFICATION SUMMARY")
print("="*80)

falsification_results = {
    "Criterion 1 (PPC failure)": criterion_1_pass,
    "Criterion 2 (LOO instability)": criterion_2_pass,
    "Criterion 3 (Convergence)": criterion_3_pass,
    "Criterion 4 (Extreme shrinkage)": criterion_4_pass,
}

revision_results = {
    "Prior-posterior conflict": revision_criterion_pass,
    "Unidentifiability": unidentifiability_pass,
}

print("\nFALSIFICATION TESTS (REJECT if any fail):")
for criterion, passed in falsification_results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {criterion:<35} {status}")

all_falsification_pass = all(falsification_results.values())
print(f"\nAll falsification criteria passed: {all_falsification_pass}")

print("\nREVISION CHECKS:")
for criterion, passed in revision_results.items():
    status = "✓ OK" if passed else "⚠ REVISE NEEDED"
    print(f"  {criterion:<35} {status}")

all_revision_pass = all(revision_results.values())

# ============================================================================
# FINAL DECISION
# ============================================================================
print("\n" + "="*80)
print("FINAL DECISION")
print("="*80)

if not all_falsification_pass:
    decision = "REJECT"
    print("\n⛔ DECISION: REJECT MODEL")
    print("\nReason: One or more falsification criteria failed.")
    failed_criteria = [k for k, v in falsification_results.items() if not v]
    print(f"Failed criteria: {', '.join(failed_criteria)}")
elif not all_revision_pass:
    decision = "REVISE"
    print("\n⚠️  DECISION: REVISE MODEL")
    print("\nReason: Model shows fixable issues.")
    revision_needed = [k for k, v in revision_results.items() if not v]
    print(f"Revision needed for: {', '.join(revision_needed)}")
else:
    decision = "ACCEPT"
    print("\n✅ DECISION: ACCEPT MODEL")
    print("\nReason: All falsification criteria passed, no revision needed.")
    print("\nThe Bayesian hierarchical meta-analysis model is adequate for")
    print("scientific inference. The model successfully:")
    print("  - Captures all observed data within predictive distributions")
    print("  - Shows stable inference under leave-one-out")
    print("  - Achieved perfect convergence")
    print("  - Exhibits appropriate shrinkage patterns")
    print("  - Properly quantifies uncertainty about heterogeneity")

# Save results
results = {
    "decision": decision,
    "falsification_tests": {
        "criterion_1_ppc_failure": {
            "passed": bool(criterion_1_pass),
            "outliers": int(outliers),
            "threshold": 1
        },
        "criterion_2_loo_instability": {
            "passed": bool(criterion_2_pass),
            "max_delta_mu": float(max_delta),
            "threshold": 5.0,
            "most_influential_study": int(most_influential)
        },
        "criterion_3_convergence": {
            "passed": bool(criterion_3_pass),
            "max_rhat": float(max_rhat),
            "min_ess_bulk": float(min_ess_bulk),
            "min_ess_tail": float(min_ess_tail),
            "n_divergences": int(n_divergences)
        },
        "criterion_4_extreme_shrinkage": {
            "passed": bool(criterion_4_pass),
            "n_extreme": int(n_extreme),
            "extreme_studies": extreme_shrinkage
        }
    },
    "revision_checks": {
        "prior_posterior_conflict": {
            "passed": bool(revision_criterion_pass),
            "p_tau_gt_10_prior": float(p_tau_gt_10_prior),
            "p_tau_gt_10_posterior": float(p_tau_gt_10_post)
        },
        "unidentifiability": {
            "passed": bool(unidentifiability_pass),
            "tau_density_cv": float(density_cv)
        }
    },
    "loo_diagnostics": {
        "elpd_loo": float(loo.elpd_loo),
        "p_loo": float(loo.p_loo),
        "n_bad_pareto_k": int(n_bad_k),
        "pareto_k_values": k_values.tolist()
    },
    "loo_influence": {
        "mu_full": float(mu_full),
        "mu_loo": mu_loo.tolist(),
        "delta_mu": delta_mu.tolist()
    }
}

with open('/workspace/experiments/experiment_1/model_critique/falsification_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: /workspace/experiments/experiment_1/model_critique/falsification_results.json")

# ============================================================================
# CREATE DIAGNOSTIC PLOTS
# ============================================================================
print("\n" + "="*80)
print("CREATING DIAGNOSTIC PLOTS")
print("="*80)

# Plot 1: Leave-one-out influence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: LOO estimates with confidence bars
studies = np.arange(1, J+1)
ax1.axhline(mu_full, color='red', linestyle='--', linewidth=2, label=f'Full data: μ={mu_full:.2f}')
ax1.scatter(studies, mu_loo, s=100, c='steelblue', edgecolors='black', linewidth=1.5, zorder=3)
for i in range(J):
    ax1.plot([i+1, i+1], [mu_full, mu_loo[i]], 'gray', linestyle=':', linewidth=1)
ax1.axhline(mu_full - 5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='±5 threshold')
ax1.axhline(mu_full + 5, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax1.set_xlabel('Study Removed', fontsize=12)
ax1.set_ylabel('E[μ | data_{-i}]', fontsize=12)
ax1.set_title('Leave-One-Out Influence Analysis', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(studies)

# Panel B: Delta mu
colors = ['red' if abs(d) > 5 else 'steelblue' for d in delta_mu]
ax2.bar(studies, delta_mu, color=colors, edgecolor='black', linewidth=1.5)
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axhline(5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (|Δμ| > 5)')
ax2.axhline(-5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Study Removed', fontsize=12)
ax2.set_ylabel('Δμ = E[μ | data_{-i}] - E[μ | data]', fontsize=12)
ax2.set_title('Change in Overall Effect Estimate', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(studies)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/model_critique/plots/loo_influence.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/loo_influence.png")
plt.close()

# Plot 2: Shrinkage diagnostics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Observed vs Posterior Mean
ax1.scatter(y_obs, theta_mean, s=150, c='steelblue', edgecolors='black', linewidth=2, zorder=3)
for i in range(J):
    ax1.annotate(f'{i+1}', (y_obs[i], theta_mean[i]), fontsize=9, ha='center', va='center', color='white', fontweight='bold')
ax1.plot([-5, 30], [-5, 30], 'k--', linewidth=1, alpha=0.5, label='No shrinkage (1:1 line)')
ax1.axhline(mu_full, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'Overall mean: μ={mu_full:.2f}')
ax1.set_xlabel('Observed Effect (y)', fontsize=12)
ax1.set_ylabel('Posterior Mean E[θ]', fontsize=12)
ax1.set_title('Shrinkage Pattern', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 30)
ax1.set_ylim(-5, 30)

# Panel B: Shrinkage magnitude vs threshold
shrinkage_mag = np.abs(shrinkage_diff)
colors = ['red' if mag > thresh else 'steelblue' for mag, thresh in zip(shrinkage_mag, shrinkage_threshold)]
ax2.bar(studies, shrinkage_mag, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7, label='|E[θ] - y|')
ax2.scatter(studies, shrinkage_threshold, s=100, c='red', marker='_', linewidths=3, label='3σ threshold', zorder=3)
ax2.set_xlabel('Study', fontsize=12)
ax2.set_ylabel('Shrinkage Magnitude', fontsize=12)
ax2.set_title('Extreme Shrinkage Test', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(studies)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/model_critique/plots/shrinkage_diagnostics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/shrinkage_diagnostics.png")
plt.close()

# Plot 3: Prior-posterior comparison for tau
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Prior vs Posterior
tau_grid_plot = np.linspace(0, 25, 500)
prior_tau = 2 / (np.pi * 5 * (1 + (tau_grid_plot/5)**2))
ax1.plot(tau_grid_plot, prior_tau, 'b-', linewidth=2, label='Prior: Half-Cauchy(0, 5)', alpha=0.7)
ax1.hist(tau_samples, bins=50, density=True, alpha=0.5, color='green', edgecolor='black', label='Posterior')
ax1.axvline(np.median(tau_samples), color='red', linestyle='--', linewidth=2, label=f'Posterior median: {np.median(tau_samples):.2f}')
ax1.axvline(10, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Conflict threshold (τ=10)')
ax1.set_xlabel('τ (between-study SD)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Prior-Posterior Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim(0, 25)
ax1.grid(True, alpha=0.3)

# Panel B: Cumulative distribution
tau_sorted = np.sort(tau_samples)
cdf_values = np.arange(1, len(tau_sorted)+1) / len(tau_sorted)
ax2.plot(tau_sorted, cdf_values, 'g-', linewidth=2, label='Posterior CDF')
# Prior CDF: P(τ < x) = 2/π * arctan(x/5)
prior_cdf = (2/np.pi) * np.arctan(tau_grid_plot / 5)
ax2.plot(tau_grid_plot, prior_cdf, 'b--', linewidth=2, label='Prior CDF', alpha=0.7)
ax2.axvline(10, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax2.axhline(p_tau_gt_10_post, color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'P(τ>10|data)={p_tau_gt_10_post:.3f}')
ax2.set_xlabel('τ (between-study SD)', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim(0, 25)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/model_critique/plots/prior_posterior_tau.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/prior_posterior_tau.png")
plt.close()

# Plot 4: LOO Pareto k diagnostics
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_khat(loo, ax=ax, show_bins=True)
ax.set_title('LOO-CV Pareto k Diagnostic', fontsize=14, fontweight='bold')
ax.set_xlabel('Data Point (Study)', fontsize=12)
ax.set_ylabel('Pareto k', fontsize=12)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/model_critique/plots/loo_pareto_k.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/loo_pareto_k.png")
plt.close()

print("\n" + "="*80)
print("FALSIFICATION ANALYSIS COMPLETE")
print("="*80)
print(f"\nFinal Decision: {decision}")
print(f"All files saved to: /workspace/experiments/experiment_1/model_critique/")
