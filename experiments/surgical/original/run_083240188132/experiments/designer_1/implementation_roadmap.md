# Implementation Roadmap
## Designer 1: Step-by-Step PyMC Implementation Guide

**Date**: 2025-10-30
**Purpose**: Detailed guide for implementing and evaluating proposed models

---

## Overview

This roadmap provides step-by-step instructions for implementing the three proposed Bayesian models. Follow this sequentially to ensure proper model validation and selection.

---

## Phase 0: Setup and Data Preparation

### Load Required Libraries
```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit, expit
from scipy import stats

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

### Load and Prepare Data
```python
# Load data
data = pd.read_csv('/workspace/data/data.csv')

# Extract arrays
groups = data['group'].values
n = data['n'].values
r = data['r'].values
n_groups = len(groups)

# Print summary
print(f"Number of groups: {n_groups}")
print(f"Sample sizes: {n.min()} to {n.max()}")
print(f"Event counts: {r.min()} to {r.max()}")
print(f"Pooled rate: {r.sum()}/{n.sum()} = {r.sum()/n.sum():.4f}")
print(f"\nGroup details:")
for i in range(n_groups):
    print(f"  Group {groups[i]}: {r[i]}/{n[i]} = {r[i]/n[i]:.4f}")
```

---

## Phase 1: Model 1 - Beta-Binomial Hierarchical

### 1.1 Prior Predictive Checks

```python
# Model 1: Beta-Binomial with mean-concentration parameterization
with pm.Model() as model_bb:
    # Hyperpriors
    mu = pm.Beta("mu", alpha=2, beta=18)  # Prior mean centered at 0.1
    kappa = pm.Gamma("kappa", alpha=2, beta=0.1)  # Concentration

    # Reparameterize to alpha, beta for Beta distribution
    alpha_param = mu * kappa
    beta_param = (1 - mu) * kappa

    # Group-level probabilities
    p = pm.Beta("p", alpha=alpha_param, beta=beta_param, shape=n_groups)

    # Likelihood
    obs = pm.Binomial("obs", n=n, p=p, observed=r)

    # Derived quantities
    sigma_sq = pm.Deterministic("sigma_sq", mu * (1 - mu) / (kappa + 1))
    phi = pm.Deterministic("phi", 1 + 1/kappa)
    icc = pm.Deterministic("icc", 1 / (kappa + 1))

# Prior predictive check
with model_bb:
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=RANDOM_SEED)

# Analyze prior predictive
prior_obs = prior_pred.prior_predictive['obs'].values[0]  # shape: (1000, 12)
prior_mu = prior_pred.prior['mu'].values[0]
prior_kappa = prior_pred.prior['kappa'].values[0]
prior_phi = prior_pred.prior['phi'].values[0]
prior_icc = prior_pred.prior['icc'].values[0]

print("\n=== PRIOR PREDICTIVE CHECKS: Model 1 ===")
print(f"Prior mu: mean={prior_mu.mean():.3f}, range=[{prior_mu.min():.3f}, {prior_mu.max():.3f}]")
print(f"Prior kappa: mean={prior_kappa.mean():.3f}, range=[{prior_kappa.min():.3f}, {prior_kappa.max():.3f}]")
print(f"Prior phi: mean={prior_phi.mean():.3f}, range=[{prior_phi.min():.3f}, {prior_phi.max():.3f}]")
print(f"Prior ICC: mean={prior_icc.mean():.3f}, range=[{prior_icc.min():.3f}, {prior_icc.max():.3f}]")

# Check if observed data within prior predictive range
prior_props = prior_obs / n[np.newaxis, :]  # shape: (1000, 12)
print(f"\nPrior predictive proportions: range=[{prior_props.min():.4f}, {prior_props.max():.4f}]")
print(f"Observed proportions: range=[{(r/n).min():.4f}, {(r/n).max():.4f}]")

# Check for zero events
prior_zeros = (prior_obs == 0).sum(axis=1)
print(f"\nPrior predictive zero-event groups per dataset: mean={prior_zeros.mean():.2f}")
print(f"Observed zero-event groups: {(r == 0).sum()}")

# Visual check
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Prior predictive distributions
axes[0, 0].hist(prior_mu, bins=30, alpha=0.7, label='Prior mu')
axes[0, 0].axvline(r.sum()/n.sum(), color='red', linestyle='--', label='Observed pooled rate')
axes[0, 0].set_xlabel('Population mean (mu)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].set_title('Prior Distribution of Mean Rate')

# Plot 2: Prior phi
axes[0, 1].hist(prior_phi, bins=30, alpha=0.7, label='Prior phi')
axes[0, 1].axvline(3.5, color='red', linestyle='--', label='Observed phi ~ 3.5-5.1')
axes[0, 1].axvline(5.1, color='red', linestyle='--')
axes[0, 1].set_xlabel('Overdispersion factor (phi)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].set_title('Prior Distribution of Overdispersion')
axes[0, 1].set_xlim(0, 20)

# Plot 3: Prior predictive rates for each group
for i in range(n_groups):
    axes[1, 0].scatter([i+1]*100, prior_props[:100, i], alpha=0.1, color='blue', s=10)
axes[1, 0].scatter(groups, r/n, color='red', s=100, marker='x', linewidths=2, label='Observed')
axes[1, 0].set_xlabel('Group')
axes[1, 0].set_ylabel('Proportion')
axes[1, 0].set_title('Prior Predictive vs Observed Rates')
axes[1, 0].legend()
axes[1, 0].set_ylim(-0.05, 0.5)

# Plot 4: Prior ICC
axes[1, 1].hist(prior_icc, bins=30, alpha=0.7, label='Prior ICC')
axes[1, 1].axvline(0.66, color='red', linestyle='--', label='Observed ICC = 0.66')
axes[1, 1].set_xlabel('Intraclass Correlation (ICC)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].set_title('Prior Distribution of ICC')

plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/prior_predictive_model1.png', dpi=300, bbox_inches='tight')
print("\nSaved prior predictive plot: /workspace/experiments/designer_1/prior_predictive_model1.png")

# DECISION POINT: Do priors look reasonable?
# - Should see wide range covering observed data
# - Should occasionally generate zeros and outliers
# - If not: adjust priors and re-run
```

### 1.2 Model Fitting

```python
print("\n=== FITTING MODEL 1: Beta-Binomial ===")

with model_bb:
    # Sample from posterior
    trace_bb = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.95,  # Higher for safety
        random_seed=RANDOM_SEED,
        return_inferencedata=True
    )

    # Posterior predictive
    post_pred_bb = pm.sample_posterior_predictive(
        trace_bb,
        random_seed=RANDOM_SEED
    )

print("\n=== CONVERGENCE DIAGNOSTICS: Model 1 ===")
# Check Rhat
rhat = az.rhat(trace_bb)
print(f"Rhat summary:")
print(f"  mu: {rhat['mu'].values:.4f}")
print(f"  kappa: {rhat['kappa'].values:.4f}")
print(f"  p: max={rhat['p'].values.max():.4f}, mean={rhat['p'].values.mean():.4f}")

# Check ESS
ess = az.ess(trace_bb)
print(f"\nEffective Sample Size:")
print(f"  mu: {ess['mu'].values:.0f}")
print(f"  kappa: {ess['kappa'].values:.0f}")
print(f"  p: min={ess['p'].values.min():.0f}, mean={ess['p'].values.mean():.0f}")

# Check divergences
divergences = trace_bb.sample_stats.diverging.sum().values
n_samples = trace_bb.posterior.dims['draw'] * trace_bb.posterior.dims['chain']
print(f"\nDivergences: {divergences} / {n_samples} ({100*divergences/n_samples:.2f}%)")

# WARNING: If Rhat > 1.01 or ESS < 400 or divergences > 1%, investigate!
if rhat['mu'].values > 1.01 or rhat['kappa'].values > 1.01:
    print("\n*** WARNING: Poor convergence detected (Rhat > 1.01) ***")
if ess['mu'].values < 400 or ess['kappa'].values < 400:
    print("\n*** WARNING: Low ESS detected (<400 per parameter) ***")
if divergences / n_samples > 0.01:
    print("\n*** WARNING: High divergence rate (>1%) ***")
```

### 1.3 Posterior Analysis

```python
print("\n=== POSTERIOR SUMMARY: Model 1 ===")

# Extract posterior samples
post_mu = trace_bb.posterior['mu'].values.flatten()
post_kappa = trace_bb.posterior['kappa'].values.flatten()
post_phi = trace_bb.posterior['phi'].values.flatten()
post_icc = trace_bb.posterior['icc'].values.flatten()
post_p = trace_bb.posterior['p'].values.reshape(-1, n_groups)

# Summary statistics
print(f"\nPosterior mu: mean={post_mu.mean():.4f}, 95% CI=[{np.percentile(post_mu, 2.5):.4f}, {np.percentile(post_mu, 97.5):.4f}]")
print(f"Posterior kappa: mean={post_kappa.mean():.4f}, 95% CI=[{np.percentile(post_kappa, 2.5):.4f}, {np.percentile(post_kappa, 97.5):.4f}]")
print(f"Posterior phi: mean={post_phi.mean():.4f}, 95% CI=[{np.percentile(post_phi, 2.5):.4f}, {np.percentile(post_phi, 97.5):.4f}]")
print(f"Posterior ICC: mean={post_icc.mean():.4f}, 95% CI=[{np.percentile(post_icc, 2.5):.4f}, {np.percentile(post_icc, 97.5):.4f}]")

print(f"\nPosterior group-level rates (p_i):")
for i in range(n_groups):
    p_mean = post_p[:, i].mean()
    p_ci = np.percentile(post_p[:, i], [2.5, 97.5])
    obs_rate = r[i]/n[i]
    print(f"  Group {groups[i]}: posterior mean={p_mean:.4f}, 95% CI=[{p_ci[0]:.4f}, {p_ci[1]:.4f}], observed={obs_rate:.4f}")

# Check if observed phi in posterior
print(f"\nObserved phi range: [3.5, 5.1]")
print(f"P(phi > 3.5) = {(post_phi > 3.5).mean():.3f}")
print(f"P(phi < 5.1) = {(post_phi < 5.1).mean():.3f}")
print(f"P(3.5 < phi < 5.1) = {((post_phi > 3.5) & (post_phi < 5.1)).mean():.3f}")
```

### 1.4 Posterior Predictive Checks

```python
print("\n=== POSTERIOR PREDICTIVE CHECKS: Model 1 ===")

# Extract posterior predictive samples
post_pred_obs = post_pred_bb.posterior_predictive['obs'].values.reshape(-1, n_groups)

# Check 1: Coverage
coverage = []
for i in range(n_groups):
    ci_lower = np.percentile(post_pred_obs[:, i], 2.5)
    ci_upper = np.percentile(post_pred_obs[:, i], 97.5)
    in_interval = (r[i] >= ci_lower) and (r[i] <= ci_upper)
    coverage.append(in_interval)
    status = "✓" if in_interval else "✗"
    print(f"  Group {groups[i]}: obs={r[i]}, 95% PI=[{ci_lower:.1f}, {ci_upper:.1f}] {status}")

coverage_pct = 100 * sum(coverage) / n_groups
print(f"\nCoverage: {sum(coverage)}/{n_groups} ({coverage_pct:.1f}%)")
if coverage_pct < 70:
    print("*** WARNING: Poor coverage (<70%) - model may be inadequate ***")
elif coverage_pct < 85:
    print("*** CAUTION: Moderate coverage (70-85%) - investigate further ***")
else:
    print("*** GOOD: Coverage >85% ***")

# Check 2: Zero-event group (Group 1)
post_pred_group1 = post_pred_obs[:, 0]
prob_zero = (post_pred_group1 == 0).mean()
print(f"\nGroup 1 (0/47) posterior predictive P(r=0) = {prob_zero:.4f}")
print(f"Expected under binomial: ~0.025-0.05")
if prob_zero < 0.001:
    print("*** WARNING: Model thinks zeros extremely rare ***")

# Check 3: Outliers (Groups 2, 8, 11)
outlier_indices = [1, 7, 10]  # 0-indexed
outlier_groups = [2, 8, 11]
print(f"\nOutlier fit:")
for idx, grp in zip(outlier_indices, outlier_groups):
    ci_lower = np.percentile(post_pred_obs[:, idx], 2.5)
    ci_upper = np.percentile(post_pred_obs[:, idx], 97.5)
    in_interval = (r[idx] >= ci_lower) and (r[idx] <= ci_upper)
    status = "✓" if in_interval else "✗"
    print(f"  Group {grp}: obs={r[idx]}, 95% PI=[{ci_lower:.1f}, {ci_upper:.1f}] {status}")

# Check 4: Overdispersion in posterior predictive
post_pred_props = post_pred_obs / n[np.newaxis, :]
post_pred_phi_vals = []
for i in range(1000):  # Sample 1000 datasets
    sample_props = post_pred_props[i*8:(i+1)*8, :].mean(axis=0)  # Average over chains
    if len(sample_props) == n_groups:
        obs_var = np.var(sample_props)
        mean_prop = sample_props.mean()
        exp_var = mean_prop * (1 - mean_prop) / n.mean()
        if exp_var > 0:
            phi_est = obs_var / exp_var
            post_pred_phi_vals.append(phi_est)

post_pred_phi_vals = np.array(post_pred_phi_vals)
print(f"\nPosterior predictive phi: mean={post_pred_phi_vals.mean():.2f}, range=[{np.percentile(post_pred_phi_vals, 5):.2f}, {np.percentile(post_pred_phi_vals, 95):.2f}]")
print(f"Observed phi: ~3.5-5.1")

# Visual PPC
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Posterior vs observed
axes[0, 0].errorbar(
    groups,
    post_p.mean(axis=0),
    yerr=1.96 * post_p.std(axis=0),
    fmt='o',
    capsize=5,
    label='Posterior (mean ± 95% CI)',
    alpha=0.7
)
axes[0, 0].scatter(groups, r/n, color='red', s=100, marker='x', linewidths=2, label='Observed')
axes[0, 0].set_xlabel('Group')
axes[0, 0].set_ylabel('Proportion')
axes[0, 0].set_title('Model 1: Posterior vs Observed Rates')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: PPC overlay
for i in range(min(100, post_pred_obs.shape[0])):
    axes[0, 1].scatter(groups, post_pred_obs[i, :]/n, alpha=0.05, color='blue', s=20)
axes[0, 1].scatter(groups, r/n, color='red', s=100, marker='x', linewidths=2, label='Observed', zorder=10)
axes[0, 1].set_xlabel('Group')
axes[0, 1].set_ylabel('Proportion')
axes[0, 1].set_title('Model 1: Posterior Predictive Draws vs Observed')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Posterior phi
axes[1, 0].hist(post_phi, bins=50, alpha=0.7, color='blue', label='Posterior phi')
axes[1, 0].axvline(3.5, color='red', linestyle='--', linewidth=2, label='Observed range')
axes[1, 0].axvline(5.1, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Overdispersion factor (phi)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Model 1: Posterior Distribution of phi')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 15)

# Plot 4: Residuals
post_pred_mean = post_pred_obs.mean(axis=0)
post_pred_sd = post_pred_obs.std(axis=0)
residuals = (r - post_pred_mean) / post_pred_sd
axes[1, 1].scatter(groups, residuals, s=100, alpha=0.7)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 1].axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Group')
axes[1, 1].set_ylabel('Standardized Residual')
axes[1, 1].set_title('Model 1: Bayesian Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/designer_1/posterior_checks_model1.png', dpi=300, bbox_inches='tight')
print("\nSaved posterior checks plot: /workspace/experiments/designer_1/posterior_checks_model1.png")
```

### 1.5 Model 1 Decision

```python
print("\n=== MODEL 1 FALSIFICATION ASSESSMENT ===")

# Apply falsification criteria
reject_reasons = []

# Criterion 1: Extreme kappa
kappa_median = np.median(post_kappa)
if kappa_median < 0.01:
    reject_reasons.append("κ → 0 (extreme overdispersion beyond beta-binomial)")
if kappa_median > 1000:
    reject_reasons.append("κ → ∞ (no overdispersion, simple binomial sufficient)")

# Criterion 2: Poor coverage
if coverage_pct < 70:
    reject_reasons.append(f"Poor posterior predictive coverage ({coverage_pct:.1f}% < 70%)")

# Criterion 3: Phi mismatch
phi_in_range = ((post_phi > 3.5) & (post_phi < 5.1)).mean()
if phi_in_range < 0.05:
    reject_reasons.append(f"Posterior phi doesn't overlap observed range (P(3.5<φ<5.1) = {phi_in_range:.3f})")

# Criterion 4: Computational failures
if divergences / n_samples > 0.01:
    reject_reasons.append(f"High divergence rate ({100*divergences/n_samples:.2f}% > 1%)")
if rhat['mu'].values > 1.05 or rhat['kappa'].values > 1.05:
    reject_reasons.append("Poor convergence (Rhat > 1.05)")

# Criterion 5: Outlier fit
outlier_covered = sum([coverage[idx] for idx in outlier_indices])
if outlier_covered < 2:
    reject_reasons.append(f"Poor outlier fit (only {outlier_covered}/3 outliers covered)")

# Decision
if len(reject_reasons) == 0:
    print("✓✓✓ MODEL 1 ACCEPTED ✓✓✓")
    print("All falsification criteria passed.")
    model1_decision = "ACCEPT"
elif len(reject_reasons) <= 2 and coverage_pct >= 70:
    print("≈ MODEL 1 PROVISIONALLY ACCEPTED ≈")
    print("Minor issues detected but model appears adequate:")
    for reason in reject_reasons:
        print(f"  - {reason}")
    model1_decision = "ACCEPT_PROVISIONAL"
else:
    print("✗✗✗ MODEL 1 REJECTED ✗✗✗")
    print("Falsification criteria failed:")
    for reason in reject_reasons:
        print(f"  - {reason}")
    model1_decision = "REJECT"

print(f"\nModel 1 Decision: {model1_decision}")
```

---

## Phase 2: Model 2 - Random Effects Logistic Regression

### 2.1 Model Definition and Prior Predictive Checks

```python
# Model 2: Random Effects Logistic (Non-Centered)
with pm.Model() as model_logistic:
    # Hyperpriors
    mu_logit = pm.Normal("mu_logit", mu=logit(0.075), sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Non-centered parameterization
    theta_raw = pm.Normal("theta_raw", mu=0, sigma=1, shape=n_groups)
    theta = pm.Deterministic("theta", mu_logit + tau * theta_raw)

    # Transform to probability scale
    p = pm.Deterministic("p", pm.math.invlogit(theta))

    # Likelihood
    obs = pm.Binomial("obs", n=n, p=p, observed=r)

    # Derived quantities
    p_mean = pm.Deterministic("p_mean", pm.math.invlogit(mu_logit))
    sigma_sq_logit = pm.Deterministic("sigma_sq_logit", tau**2)

# Prior predictive
with model_logistic:
    prior_pred_log = pm.sample_prior_predictive(samples=1000, random_seed=RANDOM_SEED)

# Analyze (similar to Model 1)
print("\n=== PRIOR PREDICTIVE CHECKS: Model 2 ===")
prior_obs_log = prior_pred_log.prior_predictive['obs'].values[0]
prior_p_mean = prior_pred_log.prior['p_mean'].values[0]
prior_tau = prior_pred_log.prior['tau'].values[0]

print(f"Prior p_mean: mean={prior_p_mean.mean():.3f}, range=[{prior_p_mean.min():.3f}, {prior_p_mean.max():.3f}]")
print(f"Prior tau: mean={prior_tau.mean():.3f}, range=[{prior_tau.min():.3f}, {prior_tau.max():.3f}]")

prior_props_log = prior_obs_log / n[np.newaxis, :]
print(f"Prior predictive proportions: range=[{prior_props_log.min():.4f}, {prior_props_log.max():.4f}]")
print(f"Observed proportions: range=[{(r/n).min():.4f}, {(r/n).max():.4f}]")

# Save plot (similar structure to Model 1)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... (plotting code similar to Model 1) ...
plt.savefig('/workspace/experiments/designer_1/prior_predictive_model2.png', dpi=300, bbox_inches='tight')
```

### 2.2 Model Fitting

```python
print("\n=== FITTING MODEL 2: Logistic GLMM ===")

with model_logistic:
    trace_logistic = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.90,  # Non-centered should sample well
        random_seed=RANDOM_SEED,
        return_inferencedata=True
    )

    post_pred_logistic = pm.sample_posterior_predictive(
        trace_logistic,
        random_seed=RANDOM_SEED
    )

# Convergence diagnostics (same structure as Model 1)
print("\n=== CONVERGENCE DIAGNOSTICS: Model 2 ===")
rhat_log = az.rhat(trace_logistic)
ess_log = az.ess(trace_logistic)
divergences_log = trace_logistic.sample_stats.diverging.sum().values

print(f"Rhat: mu_logit={rhat_log['mu_logit'].values:.4f}, tau={rhat_log['tau'].values:.4f}")
print(f"ESS: mu_logit={ess_log['mu_logit'].values:.0f}, tau={ess_log['tau'].values:.0f}")
print(f"Divergences: {divergences_log} / {n_samples} ({100*divergences_log/n_samples:.2f}%)")
```

### 2.3 Posterior Analysis and Predictive Checks

```python
# (Similar structure to Model 1 - extract posteriors, check coverage, outliers, etc.)
print("\n=== POSTERIOR SUMMARY: Model 2 ===")
# ... (analysis code) ...

print("\n=== POSTERIOR PREDICTIVE CHECKS: Model 2 ===")
# ... (PPC code) ...

print("\n=== MODEL 2 FALSIFICATION ASSESSMENT ===")
# Apply falsification criteria specific to Model 2
# ... (decision code) ...

model2_decision = "ACCEPT"  # or "REJECT" based on criteria
```

---

## Phase 3: Model Comparison (if both accepted)

```python
if model1_decision in ["ACCEPT", "ACCEPT_PROVISIONAL"] and model2_decision in ["ACCEPT", "ACCEPT_PROVISIONAL"]:
    print("\n" + "="*70)
    print("BOTH MODELS ACCEPTED - PROCEEDING TO COMPARISON")
    print("="*70)

    # LOO Cross-Validation
    print("\n=== LOO CROSS-VALIDATION ===")

    with model_bb:
        loo_bb = az.loo(trace_bb, pointwise=True)

    with model_logistic:
        loo_logistic = az.loo(trace_logistic, pointwise=True)

    print("\nModel 1 (Beta-Binomial):")
    print(f"  LOO: {loo_bb.loo:.2f}")
    print(f"  SE: {loo_bb.se:.2f}")
    print(f"  p_loo: {loo_bb.p_loo:.2f}")

    print("\nModel 2 (Logistic GLMM):")
    print(f"  LOO: {loo_logistic.loo:.2f}")
    print(f"  SE: {loo_logistic.se:.2f}")
    print(f"  p_loo: {loo_logistic.p_loo:.2f}")

    # Compare
    loo_diff = loo_bb.loo - loo_logistic.loo
    loo_se_diff = np.sqrt(loo_bb.se**2 + loo_logistic.se**2)

    print(f"\nLOO Difference (Model 1 - Model 2): {loo_diff:.2f} ± {loo_se_diff:.2f}")

    if abs(loo_diff) < 2 * loo_se_diff:
        print("  → No meaningful difference in predictive performance")
        print("  → RECOMMENDATION: Use Model 1 (more natural for binomial overdispersion)")
        final_model = "Model 1 (Beta-Binomial)"
    elif loo_diff > 2 * loo_se_diff:
        print("  → Model 1 has better predictive performance")
        final_model = "Model 1 (Beta-Binomial)"
    else:
        print("  → Model 2 has better predictive performance")
        final_model = "Model 2 (Logistic GLMM)"

    # Compare posteriors for key groups
    print("\n=== POSTERIOR COMPARISON FOR KEY GROUPS ===")

    post_p_bb = trace_bb.posterior['p'].values.reshape(-1, n_groups)
    post_p_log = trace_logistic.posterior['p'].values.reshape(-1, n_groups)

    key_groups_idx = [0, 1, 7, 10]  # Groups 1, 2, 8, 11
    key_groups_names = [1, 2, 8, 11]

    for idx, grp in zip(key_groups_idx, key_groups_names):
        bb_mean = post_p_bb[:, idx].mean()
        bb_ci = np.percentile(post_p_bb[:, idx], [2.5, 97.5])
        log_mean = post_p_log[:, idx].mean()
        log_ci = np.percentile(post_p_log[:, idx], [2.5, 97.5])
        obs_rate = r[idx] / n[idx]

        print(f"\nGroup {grp} (observed: {obs_rate:.4f}):")
        print(f"  Model 1: {bb_mean:.4f} [{bb_ci[0]:.4f}, {bb_ci[1]:.4f}]")
        print(f"  Model 2: {log_mean:.4f} [{log_ci[0]:.4f}, {log_ci[1]:.4f}]")
        print(f"  Difference: {abs(bb_mean - log_mean):.4f}")

    print(f"\n{'='*70}")
    print(f"FINAL RECOMMENDATION: {final_model}")
    print(f"{'='*70}")

else:
    print("\n=== MODEL SELECTION ===")
    if model1_decision in ["ACCEPT", "ACCEPT_PROVISIONAL"]:
        print("Model 1 accepted, Model 2 rejected → Use Model 1")
        final_model = "Model 1 (Beta-Binomial)"
    elif model2_decision in ["ACCEPT", "ACCEPT_PROVISIONAL"]:
        print("Model 2 accepted, Model 1 rejected → Use Model 2")
        final_model = "Model 2 (Logistic GLMM)"
    else:
        print("BOTH MODELS REJECTED → Proceed to Model 3 (Student-t)")
        final_model = "Need Model 3"
```

---

## Phase 4: Model 3 (Conditional - Only if Models 1-2 Fail Outlier Fit)

```python
if final_model == "Need Model 3":
    print("\n" + "="*70)
    print("IMPLEMENTING MODEL 3: ROBUST LOGISTIC WITH STUDENT-T")
    print("="*70)

    # Model 3: Student-t random effects (non-centered)
    with pm.Model() as model_robust:
        # Hyperpriors
        mu_logit = pm.Normal("mu_logit", mu=logit(0.075), sigma=1.0)
        tau = pm.HalfNormal("tau", sigma=1.0)
        nu_raw = pm.Gamma("nu_raw", alpha=2, beta=0.1)
        nu = pm.Deterministic("nu", nu_raw + 2)  # constrain > 2

        # Non-centered Student-t random effects
        theta_raw = pm.StudentT("theta_raw", nu=nu, mu=0, sigma=1, shape=n_groups)
        theta = pm.Deterministic("theta", mu_logit + tau * theta_raw)

        # Transform to probability
        p = pm.Deterministic("p", pm.math.invlogit(theta))

        # Likelihood
        obs = pm.Binomial("obs", n=n, p=p, observed=r)

        # Derived
        p_mean = pm.Deterministic("p_mean", pm.math.invlogit(mu_logit))
        sigma_sq_logit = pm.Deterministic("sigma_sq_logit", tau**2 * nu / (nu - 2))

    # Fit
    with model_robust:
        trace_robust = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,  # Higher for Student-t
            random_seed=RANDOM_SEED,
            return_inferencedata=True
        )

        post_pred_robust = pm.sample_posterior_predictive(trace_robust, random_seed=RANDOM_SEED)

    # Analyze posterior nu
    post_nu = trace_robust.posterior['nu'].values.flatten()
    print(f"\nPosterior nu: mean={post_nu.mean():.2f}, median={np.median(post_nu):.2f}")
    print(f"P(nu > 30) = {(post_nu > 30).mean():.3f}")

    if (post_nu > 30).mean() > 0.95:
        print("\n*** Model 3 posterior strongly favors Gaussian (nu > 30) ***")
        print("*** RECOMMENDATION: Use Model 2 instead (Student-t unnecessary) ***")
    else:
        print("\n*** Heavy tails justified (nu < 30) ***")
        print("*** Model 3 provides better outlier accommodation ***")

    # ... (rest of analysis similar to Models 1-2) ...
```

---

## Phase 5: Final Documentation

```python
# Save comprehensive results
print("\n=== SAVING RESULTS ===")

# Create summary report
with open('/workspace/experiments/designer_1/results_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("BAYESIAN MODEL EVALUATION RESULTS - DESIGNER 1\n")
    f.write("="*70 + "\n\n")

    f.write(f"Data: {n_groups} groups, total n={n.sum()}, r={r.sum()}\n")
    f.write(f"Pooled rate: {r.sum()/n.sum():.4f}\n")
    f.write(f"Observed overdispersion: φ ≈ 3.5-5.1\n")
    f.write(f"Observed ICC: 0.66\n\n")

    f.write("MODEL 1: Beta-Binomial Hierarchical\n")
    f.write("-" * 40 + "\n")
    f.write(f"Decision: {model1_decision}\n")
    if model1_decision in ["ACCEPT", "ACCEPT_PROVISIONAL"]:
        f.write(f"Posterior mu: {post_mu.mean():.4f} [{np.percentile(post_mu, 2.5):.4f}, {np.percentile(post_mu, 97.5):.4f}]\n")
        f.write(f"Posterior kappa: {post_kappa.mean():.4f}\n")
        f.write(f"Posterior phi: {post_phi.mean():.4f}\n")
        f.write(f"Posterior ICC: {post_icc.mean():.4f}\n")
        f.write(f"Coverage: {coverage_pct:.1f}%\n")
    f.write("\n")

    f.write("MODEL 2: Random Effects Logistic Regression\n")
    f.write("-" * 40 + "\n")
    f.write(f"Decision: {model2_decision}\n")
    # ... (similar for Model 2) ...
    f.write("\n")

    f.write("="*70 + "\n")
    f.write(f"FINAL RECOMMENDATION: {final_model}\n")
    f.write("="*70 + "\n")

print("Results saved to: /workspace/experiments/designer_1/results_summary.txt")

# Save trace objects
az.to_netcdf(trace_bb, '/workspace/experiments/designer_1/trace_model1.nc')
az.to_netcdf(trace_logistic, '/workspace/experiments/designer_1/trace_model2.nc')

print("Trace files saved:")
print("  - /workspace/experiments/designer_1/trace_model1.nc")
print("  - /workspace/experiments/designer_1/trace_model2.nc")

print("\n" + "="*70)
print("IMPLEMENTATION COMPLETE")
print("="*70)
```

---

## Troubleshooting Guide

### Issue 1: Divergences in Model 1
**Symptoms**: >1% divergent transitions, warnings during sampling
**Solutions**:
1. Increase `target_accept` to 0.95 or 0.99
2. Check if divergences concentrate near specific parameter values (plot)
3. Try tighter priors: κ ~ Gamma(2, 0.5) instead of Gamma(2, 0.1)
4. If persistent: model may be fundamentally misspecified

### Issue 2: Poor Mixing for κ (Model 1)
**Symptoms**: Low ESS for kappa, wide Rhat
**Solutions**:
1. Run longer (4000 tune, 4000 draws)
2. Accept lower ESS for κ (focus on p_i which usually estimates well)
3. Try alternative prior: κ ~ Gamma(4, 0.5) [more informative]
4. This is expected with only 12 groups - don't overinterpret κ

### Issue 3: Non-Centered Parameterization Issues (Model 2)
**Symptoms**: Still poor convergence despite non-centered
**Solutions**:
1. Verify implementation (should have theta = mu + tau * z where z ~ N(0,1))
2. Try partially centered: theta = mu + tau * z where z ~ N(0, 0.9)
3. Check if tau near 0 (funnel may persist)
4. If tau → 0 consistently, may indicate no between-group variance

### Issue 4: All Models Fail Coverage
**Symptoms**: <70% coverage for both Models 1 and 2
**Solutions**:
1. Check data quality (errors? outliers genuine?)
2. Implement Model 3 (heavy tails)
3. Consider mixture model (two subpopulations)
4. Consult domain experts about data generation process

### Issue 5: Prior Predictive Checks Fail
**Symptoms**: Priors generate impossible values or miss observed range
**Solutions**:
1. Adjust priors to be more/less informative
2. For too vague: tighten (e.g., mu ~ Beta(5, 45) instead of Beta(2, 18))
3. For too tight: relax (e.g., mu ~ Beta(1, 1) [uniform])
4. Always re-run prior predictive after adjustment

---

## Expected Timeline

- **Phase 0 (Setup)**: 5 minutes
- **Phase 1 (Model 1)**: 30 minutes
  - Prior predictive: 5 min
  - Fitting: 5 min
  - Analysis: 10 min
  - Decision: 10 min
- **Phase 2 (Model 2)**: 25 minutes
- **Phase 3 (Comparison)**: 15 minutes
- **Phase 4 (Model 3)**: 40 minutes (if needed)
- **Phase 5 (Documentation)**: 10 minutes

**Total**: ~1.5 hours (without Model 3), ~2 hours (with Model 3)

---

## Success Criteria Summary

**Model is ACCEPTED if**:
- Rhat < 1.01 for all parameters
- ESS > 400 per chain (>200 acceptable for hyperparameters)
- Divergences < 1%
- Posterior predictive coverage > 85%
- Posterior phi overlaps observed (3.5-5.1)
- Zero-event and outlier groups adequately fit

**Model is REJECTED if**:
- Coverage < 70%
- Persistent computational failures
- Systematic bias in predictions
- Posterior conflicts with strong domain knowledge

---

## Output Files

All files saved to: `/workspace/experiments/designer_1/`

- `proposed_models.md` - Full model specifications
- `model_specifications_summary.md` - Quick reference
- `implementation_roadmap.md` - This file
- `prior_predictive_model1.png` - Model 1 prior checks
- `prior_predictive_model2.png` - Model 2 prior checks
- `posterior_checks_model1.png` - Model 1 posterior analysis
- `posterior_checks_model2.png` - Model 2 posterior analysis
- `trace_model1.nc` - Model 1 MCMC samples
- `trace_model2.nc` - Model 2 MCMC samples
- `results_summary.txt` - Final decision and recommendations

---

**End of Implementation Roadmap**
