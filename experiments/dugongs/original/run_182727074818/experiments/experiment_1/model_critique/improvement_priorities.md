# Improvement Priorities: Experiment 1 - Robust Logarithmic Regression

**Date:** 2025-10-27
**Status:** REVISE (Model 2 comparison required)
**Model:** Y ~ StudentT(ν, μ, σ) where μ = α + β·log(x + c)

---

## Overview

This document outlines specific improvements and additional analyses needed before Model 1 can be fully accepted. These are **prioritized actions** to complete the validation pipeline and address identified limitations.

**Note:** These are NOT fixes for model inadequacies, but rather **completion tasks** for the experimental protocol and **optional enhancements** for robustness.

---

## PRIORITY 1: CRITICAL (BLOCKING ACCEPTANCE)

### 1.1 Fit Model 2: Segmented Regression with Change Point

**Why critical:**
- Falsification criterion 3: "Reject if change-point model wins by ΔWAIC > 6"
- Minimum attempt policy: Must fit at least 2 candidate models
- EDA evidence: Strong two-regime pattern (66% RSS reduction)
- Cannot make final decision without this comparison

**Action required:**

**Step 1: Model specification**

```python
# PyMC implementation
import pymc as pm
import numpy as np

with pm.Model() as model_2_segmented:
    # Data
    x = pm.Data('x', X_obs)
    y = pm.Data('y', Y_obs)

    # Priors
    alpha = pm.Normal('alpha', mu=2.0, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.2, sigma=0.2)  # Steep initial slope
    beta_2 = pm.Normal('beta_2', mu=0.05, sigma=0.1) # Flat later slope
    tau = pm.DiscreteUniform('tau', lower=5, upper=10)  # Change point
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    sigma = pm.HalfNormal('sigma', sigma=0.15)

    # Mean function (continuous at tau)
    mu = pm.math.switch(
        x <= tau,
        alpha + beta_1 * x,
        alpha + beta_1 * tau + beta_2 * (x - tau)
    )

    # Likelihood
    y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=y)

    # Sample
    idata_2 = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.8,
        random_seed=42
    )
```

**Alternative: Continuous change point**

```python
# Use Normal distribution for tau (more flexible)
tau = pm.Normal('tau', mu=7, sigma=2)
tau_clipped = pm.math.clip(tau, 3, 15)  # Constrain to reasonable range
```

**Step 2: Convergence diagnostics**

```python
import arviz as az

# Check convergence
summary = az.summary(idata_2, var_names=['alpha', 'beta_1', 'beta_2', 'tau', 'nu', 'sigma'])
print(summary)

# Requirements:
# - All R_hat < 1.01
# - All ESS_bulk > 400
# - All ESS_tail > 400
# - Divergences < 5%
```

**Step 3: Model comparison**

```python
# Compute LOO for both models
loo_1 = az.loo(idata_model1, var_name='y_obs')
loo_2 = az.loo(idata_model2, var_name='y_obs')

# Compare models
comparison = az.compare({
    'Model 1 (Logarithmic)': idata_model1,
    'Model 2 (Segmented)': idata_model2
}, ic='loo')

print(comparison)

# Extract ΔLOO
delta_loo = comparison.loc['Model 2 (Segmented)', 'loo'] - comparison.loc['Model 1 (Logarithmic)', 'loo']
se_delta = comparison.loc['Model 2 (Segmented)', 'se']

print(f"ΔLOO = {delta_loo:.2f} ± {se_delta:.2f}")
```

**Step 4: Decision rules**

```
IF ΔLOO < -2 (Model 1 at least 2 SE better):
    → ACCEPT Model 1
    → Document that Model 2 tested and performed worse
    → Proceed with Model 1 as final

IF -2 ≤ ΔLOO ≤ 6 (Models comparable or Model 2 slightly better):
    → ACCEPT Model 1 with caveat
    → Document that alternative model exists
    → Use Model 1 for simplicity but acknowledge uncertainty

IF ΔLOO > 6 (Model 2 strongly better):
    → REJECT Model 1
    → Use Model 2 as primary model OR
    → Develop improved Model 1b OR
    → Use Bayesian model averaging
```

**Estimated time:** 3-4 hours
**Deliverables:**
- Model 2 fitted with converged MCMC
- LOO comparison table
- ΔLOO interpretation and decision
- Updated `decision.md` with final ACCEPT/REJECT

---

## PRIORITY 2: HIGH (RECOMMENDED FOR ACCEPTANCE)

### 2.1 Sensitivity Analysis: Prior Robustness

**Why important:**
- Small sample size (n=27) makes priors influential
- Need to verify conclusions robust to prior specification
- Especially important for β (main scientific parameter)

**Action required:**

**Test 1: Wider priors (double SD)**

```python
# Refit with 2× wider priors
with pm.Model() as model_1_wide_priors:
    alpha = pm.Normal('alpha', mu=2.0, sigma=1.0)    # was 0.5
    beta = pm.Normal('beta', mu=0.3, sigma=0.4)      # was 0.2
    c = pm.Gamma('c', alpha=2, beta=2)                # unchanged
    nu = pm.Gamma('nu', alpha=2, beta=0.1)            # unchanged
    sigma = pm.HalfNormal('sigma', sigma=0.3)         # was 0.15
    # ... rest of model
```

**Test 2: Narrower priors (half SD)**

```python
# Refit with 0.5× narrower priors
with pm.Model() as model_1_narrow_priors:
    alpha = pm.Normal('alpha', mu=2.0, sigma=0.25)   # was 0.5
    beta = pm.Normal('beta', mu=0.3, sigma=0.1)      # was 0.2
    c = pm.Gamma('c', alpha=2, beta=2)                # unchanged
    nu = pm.Gamma('nu', alpha=2, beta=0.1)            # unchanged
    sigma = pm.HalfNormal('sigma', sigma=0.075)       # was 0.15
    # ... rest of model
```

**Test 3: Alternative sigma prior**

```python
# Test Half-Cauchy (original) vs Half-Normal (current)
with pm.Model() as model_1_cauchy_sigma:
    # ... same priors except:
    sigma = pm.HalfCauchy('sigma', beta=0.15)  # Compare to HalfNormal
```

**Comparison metrics:**

```python
# Compare posteriors
params = ['alpha', 'beta', 'sigma']
for param in params:
    baseline = idata_baseline.posterior[param].values.flatten()
    wide = idata_wide.posterior[param].values.flatten()
    narrow = idata_narrow.posterior[param].values.flatten()

    print(f"\n{param}:")
    print(f"  Baseline: {baseline.mean():.3f} ± {baseline.std():.3f}")
    print(f"  Wide:     {wide.mean():.3f} ± {wide.std():.3f}")
    print(f"  Narrow:   {narrow.mean():.3f} ± {narrow.std():.3f}")
    print(f"  Max shift: {max(abs(wide.mean()-baseline.mean()), abs(narrow.mean()-baseline.mean())):.3f}")
```

**Acceptance criterion:**
- If β posterior mean shifts < 0.05 (15% of SD), conclusion robust
- If β 95% CI still excludes zero in all cases, positive effect robust
- If predictions change < 0.1 units at typical x values, adequate

**Estimated time:** 2 hours
**Deliverables:**
- 3 refitted models with alternative priors
- Comparison table of posteriors
- Assessment of robustness
- Documentation in sensitivity analysis report

---

### 2.2 Sensitivity Analysis: Likelihood Choice

**Why important:**
- Student-t vs Normal is modeling choice
- ν ≈ 23 is close to Normal (ν → ∞)
- Need to verify Student-t adds value

**Action required:**

**Test 1: Refit with Normal likelihood**

```python
with pm.Model() as model_1_normal:
    # Same priors as Model 1
    alpha = pm.Normal('alpha', mu=2.0, sigma=0.5)
    beta = pm.Normal('beta', mu=0.3, sigma=0.2)
    c = pm.Gamma('c', alpha=2, beta=2)
    sigma = pm.HalfNormal('sigma', sigma=0.15)
    # Note: no nu parameter

    # Mean function
    mu = alpha + beta * pm.math.log(x + c)

    # Normal likelihood (instead of Student-t)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

**Test 2: Compare models**

```python
# LOO comparison
loo_normal = az.loo(idata_normal, var_name='y_obs')
loo_student = az.loo(idata_model1, var_name='y_obs')

comparison = az.compare({
    'Normal': idata_normal,
    'Student-t': idata_model1
}, ic='loo')

print(comparison)
```

**Interpretation:**
- If ΔLOO < 2: Likelihoods equivalent, could simplify to Normal
- If ΔLOO ∈ [2, 6]: Student-t preferred but not strongly
- If ΔLOO > 6: Student-t clearly better (justifies complexity)

**Also compare:**
- Parameter estimates (should be similar if ν ≈ 23)
- Prediction intervals (Student-t should be wider)
- Pareto-k values (may differ if outliers present)

**Estimated time:** 1.5 hours
**Deliverables:**
- Normal likelihood model fitted
- LOO comparison
- Assessment of whether Student-t adds value
- Recommendation on likelihood choice

---

### 2.3 Influence Diagnostics: Remove Potential Outliers

**Why important:**
- x = 31.5 flagged in EDA as outlier
- x = 12.0 shows local prediction discrepancy
- Verify results robust to these observations

**Action required:**

**Test 1: Leave-one-out for x = 31.5**

```python
# Remove x = 31.5 and refit
idx_outlier = np.where(X_obs == 31.5)[0]
X_reduced = np.delete(X_obs, idx_outlier)
Y_reduced = np.delete(Y_obs, idx_outlier)

# Refit Model 1 on n=26 observations
# Compare posteriors to full-data Model 1
```

**Test 2: Leave-one-out for x = 12.0 replicates**

```python
# Remove both x = 12.0 observations
idx_x12 = np.where(X_obs == 12.0)[0]
X_reduced = np.delete(X_obs, idx_x12)
Y_reduced = np.delete(Y_obs, idx_x12)

# Refit and compare
```

**Comparison:**

```python
# For each parameter
for param in ['alpha', 'beta', 'sigma']:
    full = idata_full.posterior[param].values.flatten()
    reduced = idata_reduced.posterior[param].values.flatten()

    shift = abs(reduced.mean() - full.mean())
    shift_se = shift / full.std()  # In units of posterior SD

    print(f"{param}: Shift = {shift:.4f} ({shift_se:.2f} SE)")
```

**Acceptance criterion:**
- If shift < 0.5 posterior SD for all parameters, robust
- Consistent with Pareto-k < 0.5 (low influence expected)

**Estimated time:** 1 hour
**Deliverables:**
- Refitted models without outliers
- Comparison of posteriors
- Assessment of influence
- Documentation that results are robust

---

## PRIORITY 3: MEDIUM (USEFUL BUT NOT BLOCKING)

### 3.1 Fixed c Simplification Test

**Why useful:**
- c has wide uncertainty (SD = 0.43)
- Posterior mean c ≈ 0.63 is close to conventional c = 1
- Fixing c = 1 would simplify model to 4 parameters

**Action required:**

```python
# Refit with c = 1 (fixed)
with pm.Model() as model_1_fixed_c:
    alpha = pm.Normal('alpha', mu=2.0, sigma=0.5)
    beta = pm.Normal('beta', mu=0.3, sigma=0.2)
    # c = 1.0 (fixed, not a parameter)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    sigma = pm.HalfNormal('sigma', sigma=0.15)

    mu = alpha + beta * pm.math.log(x + 1.0)  # Fixed shift

    y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=y)
```

**Compare:**
- Posteriors for α and β (should adjust to compensate for fixed c)
- LOO scores (fixed c should be slightly worse but may be acceptable)
- Predictions (should be very similar)

**Decision:**
- If ΔLOO < 2 and predictions within ±0.05: Can simplify to c=1
- If ΔLOO > 2: Keep c as learned parameter

**Benefit of fixing c:**
- One fewer parameter to interpret
- Conventional transformation log(x+1)
- Easier to communicate

**Estimated time:** 1 hour

---

### 3.2 Heteroscedasticity Test

**Why useful:**
- EDA found varying replicate precision
- Constant variance is assumption, not proven
- Could improve fit if variance increases with x

**Action required:**

```python
# Model with variance function
with pm.Model() as model_1_hetero:
    alpha = pm.Normal('alpha', mu=2.0, sigma=0.5)
    beta = pm.Normal('beta', mu=0.3, sigma=0.2)
    c = pm.Gamma('c', alpha=2, beta=2)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)

    # Variance parameters
    sigma_0 = pm.HalfNormal('sigma_0', sigma=0.15)
    gamma = pm.Normal('gamma', mu=0, sigma=0.1)  # Variance trend

    # Mean and variance functions
    mu = alpha + beta * pm.math.log(x + c)
    sigma_i = sigma_0 * pm.math.exp(gamma * x)  # Exponential variance

    y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma_i, observed=y)
```

**Compare:**
- LOO scores
- Residual vs fitted plot (should be flatter if improvement)
- γ posterior (if excludes zero, heteroscedasticity present)

**Decision:**
- If ΔLOO < 2: Homoscedastic model adequate
- If ΔLOO ∈ [2, 6]: Weak evidence for heteroscedasticity
- If ΔLOO > 6: Should use variance model

**Estimated time:** 1.5 hours

---

### 3.3 Bayesian R²

**Why useful:**
- Provides standardized measure of explained variance
- Comparable to EDA's frequentist R² ≈ 0.888
- Useful for communication

**Action required:**

```python
# Compute Bayesian R²
def bayesian_r2(y_obs, y_pred):
    """
    Bayesian R² = Var(ŷ) / [Var(ŷ) + Var(residuals)]
    """
    var_pred = np.var(y_pred, axis=1)  # Variance of predictions
    var_resid = np.var(y_obs - y_pred, axis=1)  # Variance of residuals
    r2 = var_pred / (var_pred + var_resid)
    return r2

# Apply to posterior predictive samples
y_pred = idata.posterior_predictive['y_obs'].values  # Shape: (chains, draws, obs)
y_pred_flat = y_pred.reshape(-1, len(Y_obs))  # Shape: (samples, obs)

r2_samples = bayesian_r2(Y_obs, y_pred_flat)

print(f"Bayesian R²: {r2_samples.mean():.3f} ± {r2_samples.std():.3f}")
print(f"95% HDI: [{np.percentile(r2_samples, 2.5):.3f}, {np.percentile(r2_samples, 97.5):.3f}]")
```

**Compare to EDA:**
- EDA R² = 0.888 (frequentist)
- Bayesian R² should be similar (expected ~0.85-0.90)
- If substantially different, investigate discrepancy

**Estimated time:** 0.5 hours

---

### 3.4 Posterior Predictive P-values for Specific Tests

**Why useful:**
- Current PPC used 7 general test statistics
- Could add domain-specific tests
- E.g., runs test for randomness, Durbin-Watson for autocorrelation

**Action required:**

```python
from scipy import stats

# Runs test for randomness
def runs_test(residuals):
    """Test for randomness in residuals"""
    median = np.median(residuals)
    runs = np.sum(np.diff(residuals > median) != 0) + 1
    # ... compute p-value
    return runs, pval

# Apply to observed and replicated data
resid_obs = Y_obs - posterior_mean_pred
runs_obs, _ = runs_test(resid_obs)

runs_rep = []
for y_rep in posterior_predictive_samples:
    resid_rep = y_rep - posterior_mean_pred
    runs_rep.append(runs_test(resid_rep)[0])

ppc_pval = np.mean(runs_rep >= runs_obs)
print(f"Runs test PPC p-value: {ppc_pval:.3f}")
# If p ∈ [0.05, 0.95], residuals are random (good)
```

**Other tests to consider:**
- Durbin-Watson (autocorrelation)
- Breusch-Pagan (heteroscedasticity)
- Jarque-Bera (normality)
- Shapiro-Wilk (normality)

**Estimated time:** 1 hour

---

## PRIORITY 4: LOW (NICE TO HAVE)

### 4.1 Model Averaging (if Model 2 comparable)

**When needed:**
- If ΔLOO ∈ [-2, 6] (models comparable)
- Want predictions that account for model uncertainty

**Action required:**

```python
# Compute model weights from LOO
def loo_weights(loo_1, loo_2):
    """Compute Akaike-like weights from LOO"""
    loos = np.array([loo_1.elpd_loo, loo_2.elpd_loo])
    max_loo = loos.max()
    exp_loo = np.exp(loos - max_loo)
    weights = exp_loo / exp_loo.sum()
    return weights

w1, w2 = loo_weights(loo_model1, loo_model2)

# Averaged predictions
y_pred_1 = posterior_predictive_model1
y_pred_2 = posterior_predictive_model2
y_pred_avg = w1 * y_pred_1 + w2 * y_pred_2

print(f"Model 1 weight: {w1:.3f}")
print(f"Model 2 weight: {w2:.3f}")
```

**Use averaged predictions for:**
- Point estimates
- Credible intervals
- Scientific interpretation

**Estimated time:** 1 hour

---

### 4.2 Predictive Distribution at Specific x Values

**Why useful:**
- Stakeholders may care about specific predictions
- E.g., "What is Y when x = 15?"

**Action required:**

```python
# Predict at new x values
x_new = np.array([5, 10, 15, 20, 25, 30])

with model_1:
    pm.set_data({'x': x_new})
    posterior_predictive_new = pm.sample_posterior_predictive(
        idata_model1.posterior,
        var_names=['y_obs']
    )

# Summarize predictions
for i, x_val in enumerate(x_new):
    y_samples = posterior_predictive_new.posterior_predictive['y_obs'][:, :, i].values.flatten()
    print(f"x = {x_val}:")
    print(f"  Mean: {y_samples.mean():.2f}")
    print(f"  95% CI: [{np.percentile(y_samples, 2.5):.2f}, {np.percentile(y_samples, 97.5):.2f}]")
```

**Create visualization:**
- Posterior predictive distribution at each x
- Violin plots or density plots
- Annotate with mean and credible intervals

**Estimated time:** 1 hour

---

### 4.3 Extrapolation Assessment

**Why useful:**
- May need predictions beyond x = 31.5
- Should quantify increasing uncertainty

**Action required:**

```python
# Predict at high x values
x_extrap = np.array([40, 50, 75, 100])

# Same as above, compute predictions
# Compare prediction interval width to interpolation range

x_interp = np.array([10, 20, 30])  # Within data
# ... compute predictions for both

# Plot CI width vs x
ci_width_interp = []
ci_width_extrap = []
# ... compute

plt.plot(x_interp, ci_width_interp, 'o-', label='Interpolation')
plt.plot(x_extrap, ci_width_extrap, 's--', label='Extrapolation')
plt.xlabel('x')
plt.ylabel('95% CI Width')
plt.title('Prediction Uncertainty vs Distance from Data')
plt.legend()
```

**Document:**
- At what x does CI width double?
- At what x do predictions become "speculative"?
- Recommend cutoff for extrapolation (e.g., x < 50)

**Estimated time:** 1 hour

---

### 4.4 Comparison to EDA Asymptotic Model (Model 3)

**Why useful:**
- EDA tested asymptotic model (R² = 0.834)
- Addresses saturation question directly
- Could be alternative if extrapolation critical

**Action required:**

```python
# Michaelis-Menten asymptotic model
with pm.Model() as model_3_asymptotic:
    y_max = pm.Normal('y_max', mu=2.8, sigma=0.3)
    y_min = pm.Normal('y_min', mu=1.8, sigma=0.3)
    K = pm.Gamma('K', alpha=2, beta=0.2)  # Half-saturation
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    sigma = pm.HalfNormal('sigma', sigma=0.15)

    # Asymptotic mean function
    mu = y_min + (y_max - y_min) * x / (K + x)

    y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=y)
```

**Compare all three models:**

```python
comparison = az.compare({
    'Model 1 (Logarithmic)': idata_model1,
    'Model 2 (Segmented)': idata_model2,
    'Model 3 (Asymptotic)': idata_model3
}, ic='loo')

print(comparison)
```

**Use for:**
- Understanding saturation vs continued growth
- Extrapolation scenarios
- Mechanistic interpretation

**Estimated time:** 2 hours

---

## Summary Table

| Priority | Task | Estimated Time | Blocking? | Impact |
|----------|------|----------------|-----------|--------|
| **1.1** | Fit Model 2 (change-point) | 3-4 hours | ✓ YES | High |
| **2.1** | Prior sensitivity | 2 hours | No | High |
| **2.2** | Likelihood sensitivity | 1.5 hours | No | High |
| **2.3** | Influence diagnostics | 1 hour | No | Medium |
| **3.1** | Fixed c test | 1 hour | No | Medium |
| **3.2** | Heteroscedasticity test | 1.5 hours | No | Medium |
| **3.3** | Bayesian R² | 0.5 hours | No | Low |
| **3.4** | Specific PPC tests | 1 hour | No | Low |
| **4.1** | Model averaging | 1 hour | No | Low |
| **4.2** | Specific predictions | 1 hour | No | Low |
| **4.3** | Extrapolation assessment | 1 hour | No | Low |
| **4.4** | Model 3 comparison | 2 hours | No | Low |
| | **TOTAL** | **16-18 hours** | | |

---

## Recommended Workflow

### Phase 1: Critical (Required for Decision) - 3-4 hours

1. Fit Model 2 (segmented regression)
2. Compute ΔLOO and make comparison
3. Update decision.md with ACCEPT/REVISE/REJECT

**Deliverable:** Final decision on Model 1 adequacy

---

### Phase 2: Validation (If Model 1 Accepted) - 4-5 hours

4. Prior sensitivity analysis
5. Likelihood sensitivity analysis
6. Influence diagnostics (outlier removal)

**Deliverable:** Robustness report demonstrating stable conclusions

---

### Phase 3: Enhancement (Optional) - 4-5 hours

7. Fixed c simplification test
8. Heteroscedasticity test
9. Bayesian R² computation
10. Additional PPC tests

**Deliverable:** Enhanced model documentation

---

### Phase 4: Extensions (Nice to Have) - 4-5 hours

11. Model averaging (if needed)
12. Specific predictions at key x values
13. Extrapolation uncertainty assessment
14. Model 3 comparison (saturation question)

**Deliverable:** Comprehensive model comparison and applications

---

## Success Metrics

**Minimum success (Phase 1 only):**
- ✓ Model 2 fitted and compared
- ✓ Final ACCEPT/REJECT decision made
- ✓ Justification documented

**Full success (Phases 1-2):**
- ✓ All above
- ✓ Sensitivity analyses confirm robust conclusions
- ✓ Influence diagnostics show stable results
- ✓ Model 1 accepted with documented limitations

**Comprehensive success (Phases 1-3):**
- ✓ All above
- ✓ Simplified variants tested (fixed c, Normal likelihood)
- ✓ Bayesian R² computed
- ✓ Full diagnostic suite completed

**Exceptional success (Phases 1-4):**
- ✓ All above
- ✓ Model 3 compared (saturation addressed)
- ✓ Extrapolation uncertainty quantified
- ✓ Stakeholder-specific predictions provided
- ✓ Model averaging implemented (if applicable)

---

## Expected Outcomes

**Most likely outcome (90% confidence):**
- Model 2 does not substantially outperform Model 1 (ΔLOO < 6)
- Model 1 ACCEPTED after comparison
- Sensitivity analyses confirm robustness
- Minor improvements documented but not essential
- Final model: Robust logarithmic regression (Model 1)

**Alternative outcome (8% confidence):**
- Model 2 and Model 1 comparable (ΔLOO ∈ [2, 6])
- Model 1 ACCEPTED with caveat about alternative
- Model averaging recommended for critical decisions
- Final model: Logarithmic (primary) with segmented as alternative

**Unlikely outcome (2% confidence):**
- Model 2 strongly preferred (ΔLOO > 6)
- Model 1 REJECTED per falsification criteria
- Need to use Model 2 or develop hybrid
- Final model: Segmented regression or Model 1b (revised)

---

## Communication Plan

**After Phase 1 (Critical):**
- Update stakeholders on final decision
- Provide 1-page summary of comparison results
- Recommend path forward

**After Phase 2 (Validation):**
- Deliver robustness report
- Confirm conclusions stable
- Provide parameter estimates with uncertainty

**After Phase 3 (Enhancement):**
- Deliver comprehensive technical report
- Document all model variants tested
- Provide recommendations for model use

**After Phase 4 (Extensions):**
- Deliver stakeholder communication package
- Provide predictions at specific x values of interest
- Document extrapolation limits
- Create visualizations for presentations

---

## Conclusion

**The most critical task** is fitting Model 2 and performing the comparison (Priority 1.1). This is the ONLY blocking issue preventing model acceptance.

**All other tasks** are enhancements to demonstrate robustness and provide additional insights, but are not strictly necessary for a valid model.

**Recommended minimum:** Complete Phase 1 (3-4 hours) to make final decision.

**Recommended full validation:** Complete Phases 1-2 (7-9 hours) to ensure robust, well-documented model.

**Estimated timeline:** 1-2 days of focused work to complete minimum requirements and make final decision.

---

**Document Status:** FINAL
**Approval:** Model Criticism Specialist
**Date:** 2025-10-27

---
