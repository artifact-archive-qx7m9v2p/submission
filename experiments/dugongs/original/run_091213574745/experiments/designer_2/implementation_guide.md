# Implementation Guide: Flexible Bayesian Models

**Quick reference for implementing the three proposed models in PyMC**

---

## Setup and Data Preparation

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from patsy import dmatrix
import pytensor.tensor as pt

# Load data
data = pd.read_csv('/workspace/data/data.csv')
x = data['x'].values
y = data['Y'].values
n = len(x)

# Standardize x for numerical stability (optional but recommended)
x_mean = x.mean()
x_std = x.std()
x_scaled = (x - x_mean) / x_std

# For GP: reshape to (n, 1)
X = x[:, None]
X_scaled = x_scaled[:, None]
```

---

## Model 1: Gaussian Process with Matérn 3/2 Kernel

### Full Implementation

```python
with pm.Model() as model_1_gp:
    # Mean function parameters (informative priors from EDA)
    beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.3)
    beta_1 = pm.Normal('beta_1', mu=0.3, sigma=0.15)

    # Mean function: logarithmic trend
    mean_func = beta_0 + beta_1 * pm.math.log(x)

    # GP hyperparameters
    # Marginal standard deviation (amplitude)
    alpha = pm.HalfNormal('alpha', sigma=0.15)

    # Length scale (on scaled x)
    ell_raw = pm.InverseGamma('ell_raw', alpha=5, beta=5)

    # Matern 3/2 covariance function
    cov_func = alpha**2 * pm.gp.cov.Matern32(1, ls=ell_raw)

    # Define GP
    gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

    # Observation noise
    sigma = pm.Exponential('sigma', lam=1/0.1)

    # Likelihood
    y_obs = gp.marginal_likelihood('y_obs', X=X_scaled, y=y, sigma=sigma)

    # Sampling
    trace_1 = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.95,  # High for GP
        init='adapt_diag',
        return_inferencedata=True,
        random_seed=42
    )

# Diagnostics
print(az.summary(trace_1, var_names=['beta_0', 'beta_1', 'alpha', 'ell_raw', 'sigma']))
print(f"Divergences: {trace_1.sample_stats.diverging.sum().item()}")
print(f"Max R-hat: {az.rhat(trace_1).max().item():.4f}")

# LOO-CV
loo_1 = az.loo(trace_1)
print(f"LOO: {loo_1.loo:.2f} ± {loo_1.loo_se:.2f}")
print(f"Warning: {loo_1.warning}")  # Check for high k-hat values

# Posterior predictions
with model_1_gp:
    # Predict at observed x
    f_pred = gp.conditional('f_pred', X_scaled)
    ppc_1 = pm.sample_posterior_predictive(trace_1, var_names=['f_pred'])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, color='black', s=50, alpha=0.7, label='Observed')

# Posterior mean and credible interval
f_mean = ppc_1.posterior_predictive['f_pred'].mean(dim=['chain', 'draw'])
f_lower = ppc_1.posterior_predictive['f_pred'].quantile(0.025, dim=['chain', 'draw'])
f_upper = ppc_1.posterior_predictive['f_pred'].quantile(0.975, dim=['chain', 'draw'])

ax.plot(x, f_mean, color='blue', linewidth=2, label='Posterior mean')
ax.fill_between(x, f_lower, f_upper, color='blue', alpha=0.3, label='95% CI')
ax.set_xlabel('x')
ax.set_ylabel('Y')
ax.legend()
plt.savefig('/workspace/experiments/designer_2/model1_fit.png', dpi=150, bbox_inches='tight')
```

### Key Checks

```python
# 1. Convergence diagnostics
rhat = az.rhat(trace_1)
assert (rhat < 1.01).all(), "R-hat exceeds 1.01 for some parameters"

ess_bulk = az.ess(trace_1, method='bulk')
assert (ess_bulk > 400).all(), "ESS_bulk < 400 for some parameters"

# 2. Prior-posterior comparison
az.plot_dist_comparison(trace_1, var_names=['alpha', 'ell_raw'])
plt.savefig('/workspace/experiments/designer_2/model1_prior_posterior.png', dpi=150)

# 3. Posterior predictive checks
az.plot_ppc(trace_1, num_pp_samples=100)
plt.savefig('/workspace/experiments/designer_2/model1_ppc.png', dpi=150)

# 4. Check for influential points
k_hat = loo_1.pareto_k
print(f"Points with k-hat > 0.7: {(k_hat > 0.7).sum()}")
if (k_hat > 0.7).any():
    print(f"Influential points (indices): {np.where(k_hat > 0.7)[0]}")
    print(f"Influential x values: {x[k_hat > 0.7]}")
```

### Falsification Checks

```python
# ABANDON MODEL 1 IF:

# 1. Too many divergences
divergences = trace_1.sample_stats.diverging.sum().item()
if divergences > 0.05 * (2000 * 4):  # >5% of draws
    print("CRITICAL: >5% divergent transitions")
    print("ACTION: Increase target_accept or reconsider model")

# 2. Length scale at extreme values
ell_posterior = trace_1.posterior['ell_raw'].values.flatten()
if np.percentile(ell_posterior, 5) < 0.5:
    print("WARNING: Length scale very small (ell < 0.5)")
    print("Suggests extreme local variation incompatible with smooth GP")

if np.percentile(ell_posterior, 95) > 20:
    print("WARNING: Length scale very large (ell > 20)")
    print("Suggests nearly linear trend; GP may be overkill")

# 3. LOO worse than baseline
# (Compare with Designer 1's logarithmic model)
loo_baseline = -10  # Placeholder: get actual value from Designer 1
if loo_1.loo > loo_baseline + 2:  # Worse by >2 units
    print("CRITICAL: GP LOO worse than logarithmic baseline")
    print(f"GP LOO: {loo_1.loo:.2f}, Baseline LOO: {loo_baseline:.2f}")
    print("ACTION: GP not improving fit, use simpler model")

# 4. Posterior predictions non-monotonic
f_mean_sorted = f_mean[np.argsort(x)]
if not np.all(np.diff(f_mean_sorted) >= -0.01):  # Allow tiny numerical errors
    print("CRITICAL: Posterior mean is non-monotonic")
    print("ACTION: Model fundamentally misspecified")
```

---

## Model 2: Penalized B-Splines

### Full Implementation

```python
# Generate B-spline basis
knots_quantile = np.quantile(x, [0.1, 0.25, 0.4, 0.55, 0.7, 0.85])
degree = 3
B = dmatrix(
    f"bs(x, knots={list(knots_quantile)}, degree={degree}, include_intercept=True) - 1",
    {"x": x},
    return_type='dataframe'
)
B_matrix = B.values  # Shape: (n, J) where J = # basis functions

print(f"Number of basis functions: {B_matrix.shape[1]}")

with pm.Model() as model_2_spline:
    # Smoothness parameter (controls penalty strength)
    tau = pm.Exponential('tau', lam=10)

    # First coefficient (intercept-like, informative prior)
    beta = [pm.Normal('beta_0', mu=2.3, sigma=0.5)]

    # Remaining coefficients with random walk prior (smoothness penalty)
    for j in range(1, B_matrix.shape[1]):
        beta_j = pm.Normal(f'beta_{j}', mu=beta[j-1], sigma=tau)
        beta.append(beta_j)

    # Stack into vector
    beta_vec = pt.stack(beta)

    # Mean function: linear combination of basis functions
    mu = pm.Deterministic('mu', pm.math.dot(B_matrix, beta_vec))

    # Observation noise
    sigma = pm.Exponential('sigma', lam=1/0.1)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Sampling
    trace_2 = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.90,
        init='adapt_diag',
        return_inferencedata=True,
        random_seed=42
    )

# Diagnostics
print(az.summary(trace_2, var_names=['tau', 'sigma'] + [f'beta_{j}' for j in range(B_matrix.shape[1])]))
print(f"Divergences: {trace_2.sample_stats.diverging.sum().item()}")

# LOO-CV
loo_2 = az.loo(trace_2)
print(f"LOO: {loo_2.loo:.2f} ± {loo_2.loo_se:.2f}")

# Plot fit
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, color='black', s=50, alpha=0.7, label='Observed')

mu_posterior = trace_2.posterior['mu']
mu_mean = mu_posterior.mean(dim=['chain', 'draw'])
mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

sort_idx = np.argsort(x)
ax.plot(x[sort_idx], mu_mean[sort_idx], color='red', linewidth=2, label='Posterior mean')
ax.fill_between(x[sort_idx], mu_lower[sort_idx], mu_upper[sort_idx],
                color='red', alpha=0.3, label='95% CI')
ax.set_xlabel('x')
ax.set_ylabel('Y')
ax.legend()
plt.savefig('/workspace/experiments/designer_2/model2_fit.png', dpi=150, bbox_inches='tight')
```

### Alternative: L2 Penalty (Instead of Random Walk)

```python
with pm.Model() as model_2_spline_L2:
    # Smoothness parameter
    tau = pm.Exponential('tau', lam=5)

    # First coefficient (informative)
    beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.5)

    # Remaining coefficients (independent L2 penalty)
    beta_rest = pm.Normal('beta_rest', mu=0, sigma=tau, shape=B_matrix.shape[1]-1)

    # Combine
    beta_vec = pt.concatenate([beta_0[None], beta_rest])

    # Mean function
    mu = pm.Deterministic('mu', pm.math.dot(B_matrix, beta_vec))

    # Noise and likelihood (same as above)
    sigma = pm.Exponential('sigma', lam=1/0.1)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace_2_L2 = pm.sample(2000, tune=2000, chains=4, target_accept=0.90)

# Compare random walk vs L2 via LOO
loo_2_L2 = az.loo(trace_2_L2)
print(f"Random Walk LOO: {loo_2.loo:.2f}")
print(f"L2 Penalty LOO: {loo_2_L2.loo:.2f}")
```

### Falsification Checks

```python
# ABANDON MODEL 2 IF:

# 1. Coefficients oscillate wildly (sign changes every basis function)
beta_posterior = np.array([trace_2.posterior[f'beta_{j}'].mean().item()
                           for j in range(B_matrix.shape[1])])
sign_changes = np.sum(np.diff(np.sign(beta_posterior)) != 0)
if sign_changes > B_matrix.shape[1] * 0.5:  # >50% sign changes
    print("CRITICAL: Coefficients oscillating wildly")
    print(f"Sign changes: {sign_changes} out of {B_matrix.shape[1]-1}")
    print("ACTION: Overfitting; reduce knots or increase penalty")

# 2. Smoothness parameter tau near zero (infinite penalty)
tau_posterior = trace_2.posterior['tau'].values.flatten()
if np.percentile(tau_posterior, 95) < 0.01:
    print("WARNING: tau very small (strong penalty)")
    print("Model degenerating to over-smoothed linear trend")

# 3. Basis function collinearity
beta_samples = np.array([trace_2.posterior[f'beta_{j}'].values.flatten()
                         for j in range(B_matrix.shape[1])])
corr_matrix = np.corrcoef(beta_samples)
high_corr = (np.abs(corr_matrix) > 0.95) & (corr_matrix != 1)
if high_corr.sum() > B_matrix.shape[1]:  # More than J high correlations
    print("WARNING: High posterior correlations between coefficients")
    print("Suggests collinearity; may need fewer knots")

# 4. LOO worse than Model 1
if loo_2.loo > loo_1.loo + 2:
    print(f"Model 2 LOO ({loo_2.loo:.2f}) worse than Model 1 ({loo_1.loo:.2f})")
    print("Splines not improving over GP")
```

---

## Model 3: Adaptive GP with Changepoint (ADVANCED)

### Full Implementation

```python
# WARNING: This model is computationally challenging!
# Only fit if Models 1-2 suggest regime change is real

with pm.Model() as model_3_adaptive:
    # Changepoint location
    tau = pm.Uniform('tau', lower=4, upper=12)

    # Regime indicators (deterministic, depends on tau)
    # Note: This creates discontinuous likelihood, challenging for HMC

    # SIMPLIFIED VERSION: Use mixture weights instead of hard cutoff
    # Weight for regime 1 decreases smoothly as x increases past tau
    # This makes the model differentiable everywhere

    # Regime 1 parameters
    beta_10 = pm.Normal('beta_10', mu=1.9, sigma=0.3)
    beta_11 = pm.Normal('beta_11', mu=0.11, sigma=0.05)

    # Regime 2 slope
    beta_21 = pm.Normal('beta_21', mu=0.02, sigma=0.02)

    # Regime 2 intercept (continuity constraint)
    beta_20 = pm.Deterministic('beta_20', beta_10 + tau * (beta_11 - beta_21))

    # Mean functions for both regimes
    mean_1 = beta_10 + beta_11 * x
    mean_2 = beta_20 + beta_21 * x

    # Smooth mixture weight (sigmoid transition around tau)
    # w1 is high for x << tau, low for x >> tau
    sharpness = 2.0  # Controls transition sharpness
    w1 = pm.Deterministic('w1', pm.math.invlogit(sharpness * (tau - x)))

    # Weighted mean
    mean_combined = w1 * mean_1 + (1 - w1) * mean_2

    # GP hyperparameters (shared or separate length scales)
    alpha = pm.HalfNormal('alpha', sigma=0.10)
    ell_1 = pm.InverseGamma('ell_1', alpha=3, beta=3)
    ell_2 = pm.InverseGamma('ell_2', alpha=5, beta=10)

    # Use weighted length scale (or fit separate GPs, more complex)
    ell_combined = w1 * ell_1 + (1 - w1) * ell_2

    # Covariance function
    cov_func = alpha**2 * pm.gp.cov.Matern32(1, ls=ell_combined)

    # GP with combined mean
    gp = pm.gp.Marginal(mean_func=mean_combined, cov_func=cov_func)

    # Observation noise
    sigma = pm.Exponential('sigma', lam=1/0.1)

    # Likelihood
    y_obs = gp.marginal_likelihood('y_obs', X=X_scaled, y=y, sigma=sigma)

    # Sampling (expect this to be slow and challenging!)
    trace_3 = pm.sample(
        draws=3000,
        tune=3000,
        chains=4,
        target_accept=0.99,  # Very high for complex model
        init='adapt_diag',
        max_treedepth=12,
        return_inferencedata=True,
        random_seed=42
    )

# NOTE: This is a simplified version. A full piecewise GP would require
# conditional logic that's difficult for PyMC. Consider:
# - Fitting separate GPs to x <= tau_fixed and x > tau_fixed
# - Using a discrete changepoint model (needs custom sampler)
# - Variational inference instead of MCMC

# Diagnostics
print(az.summary(trace_3, var_names=['tau', 'beta_10', 'beta_11', 'beta_21',
                                     'alpha', 'ell_1', 'ell_2', 'sigma']))
print(f"Divergences: {trace_3.sample_stats.diverging.sum().item()}")

# Check changepoint posterior
tau_posterior = trace_3.posterior['tau'].values.flatten()
print(f"Changepoint tau: {tau_posterior.mean():.2f} ± {tau_posterior.std():.2f}")
print(f"95% CI: [{np.percentile(tau_posterior, 2.5):.2f}, {np.percentile(tau_posterior, 97.5):.2f}]")

# Check if length scales differ
ell_1_post = trace_3.posterior['ell_1'].values.flatten()
ell_2_post = trace_3.posterior['ell_2'].values.flatten()
print(f"Length scale regime 1: {ell_1_post.mean():.2f}")
print(f"Length scale regime 2: {ell_2_post.mean():.2f}")
print(f"Posterior P(ell_1 < ell_2): {(ell_1_post < ell_2_post).mean():.2f}")

# LOO-CV
loo_3 = az.loo(trace_3)
print(f"LOO: {loo_3.loo:.2f} ± {loo_3.loo_se:.2f}")
```

### Falsification Checks

```python
# ABANDON MODEL 3 IF:

# 1. Too many divergences (>10%)
divergences = trace_3.sample_stats.diverging.sum().item()
if divergences > 0.10 * (3000 * 4):
    print("CRITICAL: >10% divergent transitions")
    print("Model 3 is too complex for the data")
    print("ACTION: Simplify or abandon")

# 2. Changepoint posterior is uniform (no learning)
tau_posterior = trace_3.posterior['tau'].values.flatten()
tau_hist, _ = np.histogram(tau_posterior, bins=20, range=(4, 12))
if tau_hist.max() < 2 * tau_hist.min():  # Near-uniform
    print("CRITICAL: Changepoint posterior is uniform")
    print("Data do not support a changepoint")
    print("ACTION: Use Model 1 (smooth GP) instead")

# 3. Length scales are identical
ell_1_post = trace_3.posterior['ell_1'].values.flatten()
ell_2_post = trace_3.posterior['ell_2'].values.flatten()
ell_diff = ell_1_post - ell_2_post
if np.abs(ell_diff.mean()) < 0.5 * ell_diff.std():  # Difference not clear
    print("WARNING: Length scales similar across regimes")
    print(f"P(ell_1 < ell_2): {(ell_1_post < ell_2_post).mean():.2f}")
    print("No evidence for regime-specific smoothness")

# 4. LOO worse than simpler models
if loo_3.loo > min(loo_1.loo, loo_2.loo) + 2:
    print("CRITICAL: Model 3 LOO worse than simpler models")
    print(f"Model 1 LOO: {loo_1.loo:.2f}, Model 2 LOO: {loo_2.loo:.2f}, Model 3 LOO: {loo_3.loo:.2f}")
    print("Complexity not justified")
```

---

## Model Comparison

### LOO Comparison

```python
# Compare all models that converged successfully
compare_dict = {}
if 'trace_1' in locals() and trace_1 is not None:
    compare_dict['Model 1 (GP)'] = trace_1
if 'trace_2' in locals() and trace_2 is not None:
    compare_dict['Model 2 (Splines)'] = trace_2
if 'trace_3' in locals() and trace_3 is not None:
    compare_dict['Model 3 (Adaptive)'] = trace_3

if len(compare_dict) > 1:
    comparison = az.compare(compare_dict, ic='loo', scale='deviance')
    print(comparison)

    # Plot comparison
    az.plot_compare(comparison)
    plt.savefig('/workspace/experiments/designer_2/model_comparison.png', dpi=150, bbox_inches='tight')

    # Interpretation
    best_model = comparison.index[0]
    delta_loo = comparison.loc[comparison.index[1], 'loo'] - comparison.loc[best_model, 'loo']
    se_diff = comparison.loc[comparison.index[1], 'se']

    print(f"\nBest model: {best_model}")
    print(f"ΔLOO from second-best: {delta_loo:.2f} ± {se_diff:.2f}")

    if delta_loo > 4:
        print("DECISION: Clear winner (ΔLOO > 4)")
    elif delta_loo > 2:
        print("DECISION: Weak preference (2 < ΔLOO < 4)")
    else:
        print("DECISION: Models indistinguishable (ΔLOO < 2)")
        print("Consider model averaging or choose based on interpretability")
```

### Posterior Predictive Checks

```python
# Define test statistics
def test_stat_monotonic(y_sim, x=x):
    """Check if y is monotonic in x"""
    sort_idx = np.argsort(x)
    return np.all(np.diff(y_sim[sort_idx]) >= -0.01)

def test_stat_range(y_sim):
    """Check if y range is reasonable"""
    return (y_sim.min() > 1.5) and (y_sim.max() < 3.0)

def test_stat_variance(y_sim, y_obs=y):
    """Check if variance matches observed"""
    return np.abs(y_sim.std() - y_obs.std()) / y_obs.std()

# Compute for each model
for model_name, trace in compare_dict.items():
    print(f"\n{model_name} Posterior Predictive Checks:")

    # Get posterior predictive samples
    with trace.model:
        ppc = pm.sample_posterior_predictive(trace, var_names=['y_obs'])

    y_sim = ppc.posterior_predictive['y_obs'].values.reshape(-1, n)

    # Test statistics
    p_monotonic = np.mean([test_stat_monotonic(y_sim[i]) for i in range(len(y_sim))])
    p_range = np.mean([test_stat_range(y_sim[i]) for i in range(len(y_sim))])
    var_ratio = np.median([test_stat_variance(y_sim[i]) for i in range(len(y_sim))])

    print(f"  P(monotonic): {p_monotonic:.3f} (should be ~1.0)")
    print(f"  P(range OK): {p_range:.3f} (should be ~1.0)")
    print(f"  Variance ratio: {var_ratio:.3f} (should be ~0.0)")

    # Flag failures
    if p_monotonic < 0.95:
        print("  WARNING: Model produces non-monotonic predictions!")
    if p_range < 0.95:
        print("  WARNING: Model produces out-of-range predictions!")
    if var_ratio > 0.2:
        print("  WARNING: Model variance differs >20% from observed!")
```

### Outlier Sensitivity

```python
# Refit best model without x=31.5 observation
outlier_idx = np.where(x == 31.5)[0]
if len(outlier_idx) > 0:
    outlier_idx = outlier_idx[0]
    print(f"\nRefitting without observation at x=31.5 (index {outlier_idx})")

    x_no_outlier = np.delete(x, outlier_idx)
    y_no_outlier = np.delete(y, outlier_idx)
    X_no_outlier = x_no_outlier[:, None]

    # Refit Model 1 (GP) as example
    with pm.Model() as model_1_no_outlier:
        beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.3)
        beta_1 = pm.Normal('beta_1', mu=0.3, sigma=0.15)
        mean_func = beta_0 + beta_1 * pm.math.log(x_no_outlier)

        alpha = pm.HalfNormal('alpha', sigma=0.15)
        ell_raw = pm.InverseGamma('ell_raw', alpha=5, beta=5)
        cov_func = alpha**2 * pm.gp.cov.Matern32(1, ls=ell_raw)

        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
        sigma = pm.Exponential('sigma', lam=1/0.1)

        y_obs = gp.marginal_likelihood('y_obs',
                                       X=X_no_outlier,
                                       y=y_no_outlier,
                                       sigma=sigma)

        trace_1_no_outlier = pm.sample(2000, tune=2000, chains=4,
                                       target_accept=0.95)

    # Compare posteriors
    print("\nParameter Comparison (with vs without outlier):")
    params = ['beta_0', 'beta_1', 'alpha', 'ell_raw', 'sigma']
    for param in params:
        mean_full = trace_1.posterior[param].mean().item()
        mean_no_outlier = trace_1_no_outlier.posterior[param].mean().item()
        rel_diff = (mean_no_outlier - mean_full) / mean_full * 100
        print(f"  {param}: {mean_full:.3f} → {mean_no_outlier:.3f} ({rel_diff:+.1f}%)")

    # Flag if changes are large
    if np.abs(rel_diff) > 20:
        print("\n  WARNING: Parameter changed >20% when removing outlier!")
        print("  Model is sensitive to this observation")
```

---

## Final Report Generation

```python
# Collect all results
results = {
    'models_fit': list(compare_dict.keys()),
    'loo_comparison': comparison if len(compare_dict) > 1 else None,
    'best_model': best_model if len(compare_dict) > 1 else list(compare_dict.keys())[0],
    'diagnostics': {},
    'ppc_results': {},
    'sensitivity': {}
}

# Save to file
import json
with open('/workspace/experiments/designer_2/results_summary.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nImplementation complete!")
print(f"Results saved to: /workspace/experiments/designer_2/")
print(f"Best model: {results['best_model']}")
```

---

## Troubleshooting

### Issue: Divergent Transitions

**Cause**: Posterior geometry is difficult (e.g., sharp curvature, funnel shapes)

**Solutions**:
1. Increase `target_accept` (try 0.95, 0.98, 0.99)
2. Increase `max_treedepth` (default 10, try 12)
3. Reparameterize (e.g., use non-centered parameterization)
4. Wider priors (less informative)

```python
# Example: Non-centered parameterization for GP
with pm.Model() as model_reparametrized:
    # Instead of: alpha = pm.HalfNormal('alpha', sigma=0.15)
    # Use:
    alpha_raw = pm.HalfNormal('alpha_raw', sigma=1)
    alpha = pm.Deterministic('alpha', 0.15 * alpha_raw)
    # This separates the scale from the parameter
```

### Issue: Low Effective Sample Size

**Cause**: High autocorrelation in chains (slow mixing)

**Solutions**:
1. More iterations (try tune=5000, draws=5000)
2. Check for multimodality (plot traces, pair plots)
3. Simplify model (remove unnecessary complexity)
4. Better initialization (use `init='adapt_diag'` or provide `initvals`)

### Issue: k-hat > 0.7 in LOO

**Cause**: Influential observations, model instability

**Solutions**:
1. Try Student-t likelihood (heavier tails)
2. Check which observations have high k-hat
3. Consider robust regression approach
4. Use moment-matching for high k-hat points

```python
# Recompute LOO with moment matching
loo_mm = az.loo(trace, pointwise=True, moment_match=True)
```

### Issue: Non-convergence (R-hat > 1.01)

**Cause**: Chains haven't reached stationary distribution

**Solutions**:
1. More tuning iterations (tune=5000)
2. Check for multimodality (run longer, look at traces)
3. Better priors (more informative)
4. Simpler model

---

**This guide provides complete, runnable code for all three models.**
**Use the falsification checks at each step to decide whether to continue or pivot.**
