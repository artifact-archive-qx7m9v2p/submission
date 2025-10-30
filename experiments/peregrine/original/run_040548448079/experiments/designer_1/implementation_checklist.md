# Implementation Checklist for Model Designer 1

## Quick Start Guide

### Prerequisites
- [ ] Stan (CmdStan or PyStan/RStan)
- [ ] PyMC (version ≥5.0 for Model 2 and 3)
- [ ] ArviZ (for diagnostics)
- [ ] NumPy, SciPy, Pandas

### Data Requirements
- [ ] Load 40 time-ordered count observations
- [ ] Standardized year variable (mean=0, std=1)
- [ ] Verify no missing values
- [ ] Confirm structural break at observation 17 (from EDA)

---

## Phase 1: Prior Predictive Checks (START HERE)

### Model 1 Prior Predictive
```python
# Generate 1000 datasets from prior
for i in range(1000):
    β0 = np.random.normal(4.3, 0.5)
    β1 = np.random.normal(0.35, 0.3)
    β2 = np.random.normal(1.0, 0.5)
    α = np.random.gamma(2, 1)
    ρ = np.random.beta(8, 2)

    # Simulate data
    # Check: variance/mean in [10, 200]?
    # Check: break magnitude reasonable?
```

**STOP if**: <1% of prior predictive samples cover observed statistics
**Action**: Revise priors to be less restrictive

---

## Phase 2: Simulation-Based Calibration

### Test Model 1 Recovery
```python
# Simulate 100 datasets with known parameters
true_params = {'beta2': 1.0, 'tau': 17, 'rho': 0.85, 'alpha': 0.6}

for sim in range(100):
    data = simulate_changepoint_data(true_params, N=40)
    posterior = fit_model1(data)

    # Check: true params in 95% CI?
    # Check: rank histograms uniform?
```

**STOP if**: Recovery rate <80% for any parameter
**Action**: Model is fundamentally broken; debug or redesign

---

## Phase 3: Fit Model 1

### Model 1a: Random Effects (Simplest)
```stan
// File: model1a_random_effects.stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
}
parameters {
  real beta0;
  real beta1;
  real beta2;
  real<lower=0> alpha;
  real<lower=0> sigma_obs;
  vector[N] epsilon;
}
model {
  vector[N] mu;
  vector[N] log_mu;

  beta0 ~ normal(4.3, 0.5);
  beta1 ~ normal(0.35, 0.3);
  beta2 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 1);
  sigma_obs ~ exponential(1);
  epsilon ~ normal(0, sigma_obs);

  for (t in 1:N) {
    log_mu[t] = beta0 + beta1 * year[t] +
                (t > 17 ? beta2 * (year[t] - year[17]) : 0) +
                epsilon[t];
  }
  mu = exp(log_mu);
  C ~ neg_binomial_2(mu, alpha);
}
```

**Checklist**:
- [ ] Compile Stan model
- [ ] Run 4 chains, 2000 iterations each
- [ ] Check Rhat < 1.01 for all parameters
- [ ] Check ESS > 400 for all parameters
- [ ] Check divergent transitions < 1%
- [ ] If any fail: try non-centered parameterization or stronger priors

### Model 1b: AR(1) Latent (If 1a fails autocorrelation test)
```stan
// File: model1b_ar1.stan
parameters {
  // Same as 1a, plus:
  real<lower=0,upper=1> rho;
  vector[N] eta_raw;
}
transformed parameters {
  vector[N] eta;
  eta[1] = beta0 + beta1*year[1] + sigma_eta*eta_raw[1];
  for (t in 2:N) {
    real trend = beta0 + beta1*year[t] +
                 (t > 17 ? beta2*(year[t]-year[17]) : 0);
    eta[t] = rho*eta[t-1] + (1-rho)*trend + sigma_eta*eta_raw[t];
  }
}
model {
  rho ~ beta(8, 2);
  // Rest same as 1a
}
```

**Checklist**:
- [ ] Same convergence checks as 1a
- [ ] Additionally check: ESS for rho > 400

---

## Phase 4: Model 1 Validation

### Convergence Diagnostics
```python
import arviz as az

trace = az.from_cmdstan(model1_fit)

# Check all at once
summary = az.summary(trace, var_names=['beta0', 'beta1', 'beta2', 'alpha', 'rho'])
print(summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])

# FAIL if: any r_hat > 1.01 or ess < 100
```

### Posterior Predictive Checks
```python
# Generate posterior predictive samples
C_rep = model1_fit.posterior_predictive.C_rep

# Test 1: ACF(1)
acf_obs = compute_acf(C_observed, lag=1)
acf_rep = [compute_acf(C_rep[i], lag=1) for i in range(4000)]
p_value_acf = np.mean(acf_rep > acf_obs)

# FAIL if: p_value < 0.05 or > 0.95

# Test 2: Break magnitude
growth_pre_obs = (C[17] - C[0]) / C[0]
growth_post_obs = (C[39] - C[17]) / C[17]
break_magnitude_obs = growth_post_obs / growth_pre_obs

# Compute for each posterior predictive sample
# FAIL if: p_value < 0.05 or > 0.95

# Test 3: Variance/mean ratio
# Test 4: Maximum count
# Test 5: Dispersion
```

### LOO Cross-Validation
```python
loo = az.loo(trace, pointwise=True)

# Check Pareto k values
n_high_k = np.sum(loo.pareto_k > 0.7)
print(f"Observations with k > 0.7: {n_high_k} / {N}")

# FAIL if: n_high_k / N > 0.05 (more than 5%)

# Save LOO for model comparison
loo_model1 = loo.loo
```

### Time-Series Cross-Validation
```python
# Train on 1:30, predict 31:40
train_data = C[:30]
test_data = C[30:]

fit_train = fit_model1(train_data)
predictions = fit_train.posterior_predictive.C_rep[:, 30:]

# Compute metrics
rmse_log = np.sqrt(np.mean((np.log(test_data) - np.log(predictions.mean(axis=0)))**2))
coverage = compute_coverage(test_data, predictions, level=0.95)

print(f"Out-of-sample RMSE (log): {rmse_log:.3f}")
print(f"95% Coverage: {coverage:.2%}")

# FAIL if: rmse_log > 1.5 * rmse_insample
# FAIL if: coverage < 0.85 or > 1.0
```

### Sensitivity Analysis
```python
# Vary prior SDs by 0.5x and 2x
configs = [
    {'beta0_sd': 0.25, 'beta1_sd': 0.15, 'beta2_sd': 0.25},  # Tighter
    {'beta0_sd': 0.5, 'beta1_sd': 0.3, 'beta2_sd': 0.5},     # Original
    {'beta0_sd': 1.0, 'beta1_sd': 0.6, 'beta2_sd': 1.0},     # Wider
]

results = []
for config in configs:
    fit = fit_model1_with_priors(config)
    results.append(extract_posterior_means(fit))

# Check: do posterior means change by <10% across configs?
# FAIL if: any parameter changes by >50%
```

---

## Phase 5: Decision Point (After Model 1)

### If All Tests Pass
- [ ] **STOP HERE**
- [ ] Generate final report for Model 1
- [ ] Create visualizations
- [ ] Document limitations
- [ ] **DO NOT FIT MODEL 2/3** unless requested

### If Beta2 Not Significant
- [ ] Check: 95% CI for β₂ includes zero?
- [ ] **Action**: Fit Model 2 (unknown τ)
- [ ] **Alternative**: Abandon changepoint, try GP

### If Autocorrelation Remains (ACF(1) > 0.3)
- [ ] Check: Posterior predictive ACF(1) > 0.3?
- [ ] **Action**: Try Model 1b or 1c (stronger AR structure)
- [ ] **Alternative**: State-space model

### If Computational Issues
- [ ] Check: Divergences > 1%? Rhat > 1.01?
- [ ] **Action**: Non-centered parameterization
- [ ] **Alternative**: Simplify (remove AR, use random effects)

### If Predictive Failure
- [ ] Check: LOO Pareto k > 0.7 for >5%?
- [ ] **Action**: Investigate influential points
- [ ] **Alternative**: Robust likelihood (Student-t)

---

## Phase 6: Model 2 (Unknown Changepoint) - ONLY IF NEEDED

### PyMC Implementation
```python
import pymc as pm

with pm.Model() as model2:
    # Changepoint location
    tau = pm.DiscreteUniform('tau', lower=5, upper=35)

    # Regression parameters
    beta0 = pm.Normal('beta0', mu=4.3, sigma=0.5)
    beta1 = pm.Normal('beta1', mu=0.35, sigma=0.3)
    beta2 = pm.Normal('beta2', mu=1.0, sigma=0.5)
    alpha = pm.Gamma('alpha', alpha=2, beta=1)

    # AR(1)
    rho = pm.Beta('rho', alpha=8, beta=2)
    sigma_eta = pm.Exponential('sigma_eta', lam=2)

    # Latent process (simplified - full AR(1) more complex)
    eta = pm.Normal('eta', mu=0, sigma=sigma_eta, shape=N)

    # Mean function with changepoint
    year_centered = year - year[tau]
    post_break = pm.math.switch(pm.math.ge(t_indices, tau),
                                  year_centered, 0)

    log_mu = beta0 + beta1*year + beta2*post_break + eta
    mu = pm.math.exp(log_mu)

    # Likelihood
    obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha, observed=C)

    # Sample (need custom sampler for discrete tau)
    trace = pm.sample(2000, tune=2000,
                      target_accept=0.95,
                      cores=4)
```

**Checklist**:
- [ ] Expect 10-30 minute runtime
- [ ] Monitor convergence carefully (discrete parameters challenging)
- [ ] Extract tau posterior: `tau_samples = trace.posterior.tau.values`
- [ ] Plot histogram of tau posterior
- [ ] Check: Is posterior concentrated near 17?

### Interpret Tau Posterior
```python
tau_posterior = trace.posterior.tau.values.flatten()

# Concentration test
mode_tau = scipy.stats.mode(tau_posterior).mode
prob_mode = np.mean(tau_posterior == mode_tau)

print(f"Posterior mode: tau = {mode_tau}")
print(f"Probability at mode: {prob_mode:.2%}")

# DECISION:
# If prob_mode > 0.5 and mode near 17: Validates Model 1
# If prob_mode > 0.5 but mode far from 17: Break at different location
# If prob_mode < 0.3: Diffuse posterior, no clear break
# If bimodal: Multiple breaks (try Model 3)
```

---

## Phase 7: Model Comparison

### LOO Comparison
```python
import arviz as az

# Compare all fitted models
model_dict = {
    'Model 1 (tau=17)': trace_model1,
    'Model 2 (unknown tau)': trace_model2,
}

comparison = az.compare(model_dict, ic='loo')
print(comparison)

# Interpretation:
# dloo < 2: Equivalent models
# 2 < dloo < 4: Weak preference
# dloo > 4: Strong preference
```

### Bayes Factor Approximation
```python
# BF_21 = exp(ΔLOO/2)
delta_loo = loo_model2 - loo_model1
bf_21 = np.exp(delta_loo / 2)

print(f"Approximate BF (Model 2 vs Model 1): {bf_21:.2f}")

# Interpretation:
# BF > 10: Strong evidence for Model 2
# 3 < BF < 10: Moderate evidence for Model 2
# 1/3 < BF < 3: Equivalent
# BF < 1/3: Strong evidence for Model 1
```

---

## Phase 8: Final Deliverables

### Required Outputs

1. **Model Code**:
   - [ ] `/workspace/experiments/designer_1/models/model1a.stan`
   - [ ] `/workspace/experiments/designer_1/models/model1b.stan`
   - [ ] `/workspace/experiments/designer_1/models/model2.py`

2. **Validation Scripts**:
   - [ ] `/workspace/experiments/designer_1/validation/prior_predictive.py`
   - [ ] `/workspace/experiments/designer_1/validation/sbc.py`
   - [ ] `/workspace/experiments/designer_1/validation/posterior_predictive.py`
   - [ ] `/workspace/experiments/designer_1/validation/cross_validation.py`

3. **Results**:
   - [ ] `/workspace/experiments/designer_1/results/convergence_diagnostics.txt`
   - [ ] `/workspace/experiments/designer_1/results/falsification_results.md`
   - [ ] `/workspace/experiments/designer_1/results/model_comparison.csv`

4. **Visualizations**:
   - [ ] Posterior distributions for all parameters
   - [ ] Trace plots (check mixing)
   - [ ] Posterior predictive with data overlay
   - [ ] Residual diagnostics
   - [ ] Changepoint location (if Model 2)

5. **Final Report**:
   - [ ] `/workspace/experiments/designer_1/results/final_report.md`
   - [ ] Include: Best model, diagnostics, limitations, recommendations

---

## Common Pitfalls and Solutions

### Issue: Divergent Transitions
**Symptom**: >1% divergent transitions after warmup
**Cause**: Posterior geometry is difficult (high curvature, funnels)
**Solutions**:
1. Increase adapt_delta to 0.95 or 0.99
2. Use non-centered parameterization for hierarchical terms
3. Stronger, more informative priors
4. Reparameterize (e.g., QR decomposition for regression)

### Issue: Low ESS
**Symptom**: ESS < 100 for some parameters
**Cause**: Poor mixing, high autocorrelation in chains
**Solutions**:
1. Longer chains (4000+ iterations)
2. Non-centered parameterization
3. Check for parameters on boundary (e.g., rho ≈ 1.0)
4. Consider reparameterization

### Issue: High Pareto k
**Symptom**: LOO Pareto k > 0.7 for many observations
**Cause**: Model misspecification, influential points
**Solutions**:
1. Investigate which observations have high k
2. Check for outliers or data quality issues
3. Try robust likelihood (Student-t instead of NB)
4. May indicate wrong model class entirely

### Issue: Prior-Posterior Conflict
**Symptom**: Posterior mean >2 SD from prior mean
**Cause**: Data strongly contradicts prior OR model misspecification
**Solutions**:
1. Check if data pattern is genuine (not artifact)
2. Revise prior to be more diffuse
3. May indicate structural model problem

### Issue: Model Won't Converge
**Symptom**: Rhat > 1.1, chains not mixing
**Cause**: Model is unidentified or pathologically specified
**Solutions**:
1. Simplify model (remove AR, use simpler structure)
2. Check for non-identifiability (e.g., intercept + time-varying level)
3. Try different parameterization
4. May need to abandon this model class

---

## Quick Reference: Falsification Criteria

### Model 1 - REJECT if ANY of:
- [ ] β₂ 95% CI includes zero
- [ ] Posterior predictive ACF(1) > 0.3
- [ ] Divergent transitions > 1%
- [ ] Rhat > 1.01 for any parameter
- [ ] ESS < 100 for any parameter
- [ ] LOO Pareto k > 0.7 for >5% of observations
- [ ] Out-of-sample RMSE > 1.5x in-sample
- [ ] Dispersion α < 0.1 or > 5.0
- [ ] Prior-posterior shift > 2 SD for any parameter

### Model 2 - REJECT if ANY of:
- [ ] All Model 1 criteria, PLUS:
- [ ] Posterior tau is uniform (no concentration)
- [ ] Posterior tau is multimodal (2+ modes >15% each)
- [ ] Sampling takes >30 minutes without convergence
- [ ] ΔLOO vs Model 1 < 2 (no improvement)

### Model 3 - REJECT if ANY of:
- [ ] All Model 1 criteria, PLUS:
- [ ] Posterior |τ₂ - τ₁| < 5 observations
- [ ] One break magnitude ≈0 (collapsing to k=1)
- [ ] ΔLOO(k=2, k=1) < 4 (not worth complexity)
- [ ] Out-of-sample performance worse than k=1

---

## Emergency Contacts

**If stuck on**:
- Stan compilation errors → Check Stan version, syntax
- PyMC discrete sampling → Try grid marginalization instead
- Computational issues → Simplify model, use stronger priors
- All models failing → STOP, propose alternative model classes (GP, spline)

**Remember**: Failing to fit a model is a scientific result, not a personal failure. Document what doesn't work and why.

---

## Final Checklist Before Reporting

- [ ] At least one model passed ALL falsification tests
- [ ] Convergence diagnostics clean (Rhat, ESS, divergences)
- [ ] Posterior predictive checks pass (≥80% of test statistics)
- [ ] LOO cross-validation computed and Pareto k checked
- [ ] Time-series CV shows good out-of-sample performance
- [ ] Sensitivity analysis confirms robustness
- [ ] Visualizations clearly show model fit and uncertainty
- [ ] Limitations and caveats documented
- [ ] Alternative model classes considered if changepoints fail

**If all checked**: Proceed to final report
**If any unchecked**: Continue debugging or propose alternatives

---

**END OF IMPLEMENTATION CHECKLIST**

*Quick reference for executing Model Designer 1 experiment plan*
*Always prioritize scientific integrity over task completion*
