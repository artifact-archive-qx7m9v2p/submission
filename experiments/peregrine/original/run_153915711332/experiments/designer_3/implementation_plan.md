# Implementation Plan: Designer 3 Models

## Overview

This document provides a concrete implementation roadmap for the three Bayesian model classes proposed for the time series count data.

---

## Phase 1: Setup and Preparation (30 minutes)

### 1.1 Environment Setup
```python
# Required packages
import numpy as np
import pandas as pd
import pystan  # or cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

### 1.2 Data Preparation
```python
# Load data
data = pd.read_csv('/workspace/data/data_designer_3.csv')

# Prepare for Stan
stan_data = {
    'N': len(data),
    'C': data['C'].values,
    'year': data['year'].values
}
```

### 1.3 Prior Predictive Checks
Before fitting, verify priors are reasonable:
```python
# Sample from priors only
# Generate C_sim from prior predictive
# Check: Do simulated counts cover observed range?
```

---

## Phase 2: Model 1 - Changepoint (3 hours)

### 2.1 Implementation Steps

**Step 1: Write Stan model** (45 min)
- File: `/workspace/experiments/designer_3/models/model_1_changepoint.stan`
- Include: Basic version with single changepoint
- Test compilation: `model.stanc()`

**Step 2: Initial fit with wide priors** (30 min)
```python
fit = model.sampling(
    data=stan_data,
    iter=2000,
    warmup=1000,
    chains=4,
    seed=42
)
```

**Step 3: Diagnose convergence** (30 min)
- Check Rhat for all parameters
- Check ESS (effective sample size)
- Check divergent transitions
- Trace plots for key parameters: tau, beta0_1, beta0_2, phi_1, phi_2

**Step 4: Refine priors if needed** (30 min)
- If convergence issues: tighten priors on tau
- If label switching: add ordering constraint
- Re-fit with adjusted priors

**Step 5: Posterior analysis** (45 min)
- Extract posterior samples
- Compute posterior mean and 95% CI for all parameters
- Visualize: posterior for tau, before/after growth rates
- Compute LOO-CV

### 2.2 Diagnostics Checklist
- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 for tau
- [ ] No divergent transitions
- [ ] Trace plots show good mixing
- [ ] Posterior for tau is concentrated (SD < 0.5)
- [ ] beta1_2 > beta1_1 (growth accelerates post-changepoint)

### 2.3 Falsification Tests
```python
# Test 1: Is changepoint identified?
tau_posterior_sd = fit.extract()['tau'].std()
if tau_posterior_sd > 1.0:
    print("FAIL: Changepoint not identified")

# Test 2: Are regimes different?
beta1_1 = fit.extract()['beta1_1']
beta1_2 = fit.extract()['beta1_2']
if np.quantile(beta1_2, 0.05) < np.quantile(beta1_1, 0.95):
    print("FAIL: Regimes not significantly different")

# Test 3: LOO comparison
loo1 = az.loo(fit)
if loo1.pareto_k.max() > 0.7:
    print("WARNING: High Pareto k values")
```

---

## Phase 3: Model 2 - Gaussian Process (4 hours)

### 3.1 Implementation Steps

**Step 1: Write Stan model** (60 min)
- File: `/workspace/experiments/designer_3/models/model_2_gp.stan`
- Use non-centered parameterization for GP
- Include quadratic mean function
- Use `gp_exp_quad_cov()` for covariance

**Step 2: Prior predictive checks** (30 min)
- Sample from GP prior
- Visualize: Do prior draws cover plausible functions?
- Check: Is length scale prior reasonable?

**Step 3: Initial fit** (45 min)
```python
fit2 = model2.sampling(
    data=stan_data,
    iter=2000,
    warmup=1000,
    chains=4,
    control={'adapt_delta': 0.95},  # GP often needs higher adapt_delta
    seed=42
)
```

**Step 4: Diagnose GP-specific issues** (45 min)
- Check if rho is identified (compare prior vs posterior)
- Check correlation between hyperparameters
- Visualize posterior mean function f(year)
- Check for posterior discontinuities

**Step 5: Posterior analysis** (60 min)
- Extract f(year) for each MCMC sample
- Compute pointwise credible intervals
- Compare f(year) to simple quadratic
- Compute LOO-CV

### 3.2 Diagnostics Checklist
- [ ] Rhat < 1.01 for alpha, rho, phi
- [ ] ESS > 200 for hyperparameters
- [ ] Divergences < 1%
- [ ] Length scale rho within [0.3, 3.0]
- [ ] Posterior for f is smooth (no jumps)
- [ ] GP adds structure beyond mean function

### 3.3 Falsification Tests
```python
# Test 1: Is GP meaningful?
f_samples = fit2.extract()['f']
mean_func = beta0 + beta1*year + beta2*year**2
gp_contrib = f_samples.mean(axis=0) - mean_func
if np.abs(gp_contrib).max() < 0.1:
    print("FAIL: GP is trivial")

# Test 2: Is function smooth?
f_diff = np.diff(f_samples, axis=1)
jump_threshold = 0.5  # On log scale
if np.any(np.abs(f_diff) > jump_threshold):
    print("FAIL: Posterior has discontinuities")

# Test 3: LOO comparison
loo2 = az.loo(fit2)
loo_diff = az.compare({'changepoint': loo1, 'gp': loo2})
```

---

## Phase 4: Model 3 - State-Space (5 hours)

### 4.1 Implementation Steps

**Step 1: Start with simple RW model** (60 min)
- File: `/workspace/experiments/designer_3/models/model_3_statespace.stan`
- Implement random walk: theta_t = theta_{t-1} + gamma + w_t
- Use non-centered parameterization

**Step 2: Test identification** (45 min)
```python
# Fit with different prior specifications for sigma_w
priors_to_test = [0.1, 0.2, 0.5]
for sigma_prior in priors_to_test:
    # Refit and check if posterior changes
    pass
```

**Step 3: Diagnose latent states** (60 min)
- Check Rhat for each theta_t
- Visualize: theta_t vs log(C_t)
- Check: Is theta smoother than observed?

**Step 4: Try local linear trend variant** (60 min)
- Implement variant 3b with time-varying trend
- Compare to simple RW

**Step 5: Final comparison** (75 min)
- Compute LOO for best state-space variant
- Compare to Models 1 and 2
- One-step-ahead predictions

### 4.2 Diagnostics Checklist
- [ ] Rhat < 1.01 for mu_0, gamma, sigma_w, phi
- [ ] ESS > 100 for all latent states theta_t
- [ ] Divergences < 2%
- [ ] sigma_w < 0.5 (smooth evolution)
- [ ] Latent state is smoother than observed
- [ ] No prior-posterior conflict

### 4.3 Falsification Tests
```python
# Test 1: Is latent state informative?
theta_post = fit3.extract()['theta'].mean(axis=0)
corr_with_data = np.corrcoef(theta_post, np.log(C))[0,1]
if corr_with_data > 0.99:
    print("FAIL: Latent state is trivial")

# Test 2: Is sigma_w identified?
sigma_w_prior = stats.halfnorm(0, 0.2)
sigma_w_post = fit3.extract()['sigma_w']
ks_stat = stats.ks_2samp(
    sigma_w_prior.rvs(1000),
    sigma_w_post
)
if ks_stat.pvalue > 0.1:
    print("FAIL: sigma_w not identified")

# Test 3: One-step-ahead predictions
predictions = []
for t in range(1, N):
    theta_t = fit3.extract()['theta'][:, t-1]
    mu_t = np.exp(theta_t + gamma)  # Predict t from t-1
    predictions.append(mu_t)
```

---

## Phase 5: Model Comparison and Selection (3 hours)

### 5.1 LOO-CV Comparison
```python
loo_results = az.compare({
    'changepoint': loo1,
    'gp': loo2,
    'statespace': loo3
})

print(loo_results)
```

### 5.2 Posterior Predictive Checks

For each model:
```python
C_rep = fit.extract()['C_rep']

# Test 1: Mean and variance
obs_mean, obs_var = C.mean(), C.var()
rep_mean = C_rep.mean(axis=1)
rep_var = C_rep.var(axis=1)

pval_mean = (rep_mean >= obs_mean).mean()
pval_var = (rep_var >= obs_var).mean()

# Test 2: Autocorrelation
obs_acf1 = np.corrcoef(C[:-1], C[1:])[0,1]
rep_acf1 = [np.corrcoef(C_rep[i,:-1], C_rep[i,1:])[0,1]
            for i in range(C_rep.shape[0])]
pval_acf = (np.array(rep_acf1) >= obs_acf1).mean()

# Test 3: Growth rate
obs_growth = (C[-1] - C[0]) / C[0]
rep_growth = (C_rep[:,-1] - C_rep[:,0]) / C_rep[:,0]
pval_growth = (rep_growth >= obs_growth).mean()

print(f"P-values: mean={pval_mean:.3f}, var={pval_var:.3f}, "
      f"acf={pval_acf:.3f}, growth={pval_growth:.3f}")
```

### 5.3 Cross-Validation

Leave-last-k-out for predictive validation:
```python
k = 5  # Hold out last 5 observations
train_data = {
    'N': N - k,
    'C': C[:-k],
    'year': year[:-k]
}

# Refit each model on training data
# Generate predictions for held-out data
# Compute RMSE, coverage of prediction intervals
```

### 5.4 Decision Matrix

| Criterion | Changepoint | GP | State-Space | Winner |
|-----------|-------------|-----|-------------|--------|
| LOO-ELPD | ??? | ??? | ??? | ??? |
| PPC p-values (all >0.05) | ??? | ??? | ??? | ??? |
| Convergence (Rhat<1.01) | ??? | ??? | ??? | ??? |
| Interpretability | High | Medium | Low | ??? |
| Complexity | Low | Medium | High | ??? |
| Predictive RMSE | ??? | ??? | ??? | ??? |

---

## Phase 6: Final Model Refinement (2 hours)

Once winning model is identified:

### 6.1 Sensitivity Analysis
- Vary priors systematically
- Check robustness of conclusions

### 6.2 Final Diagnostics
- Extended runs (4000 iterations) to verify stability
- Multiple random seeds
- Generate final visualizations

### 6.3 Documentation
- Write results summary
- Document all decisions made
- Include code for reproducibility

---

## Computational Resources

### Time Estimates
- Model 1: ~10 minutes per fit (simple structure)
- Model 2: ~20 minutes per fit (GP computation)
- Model 3: ~15 minutes per fit (many latent states)

### Hardware Requirements
- Minimum: 4 cores for parallel chains
- Recommended: 8 cores, 16GB RAM
- Storage: ~500MB per model (posterior samples)

---

## Contingency Plans

### If Model 1 Fails to Converge
1. Try discrete changepoint: Fix tau at grid points, marginalize
2. Use smoother transition (sigmoid)
3. Simplify to single regime with quadratic trend

### If Model 2 Has GP Issues
1. Reduce to parametric form: Drop GP, use cubic polynomial
2. Try Matern 3/2 kernel instead of SE
3. Use inducing points (sparse GP)

### If Model 3 Has Identification Issues
1. Fix gamma at known growth rate
2. Use stronger priors on sigma_w
3. Fall back to AR(1) model without latent states

### If All Models Fail
1. Use frequentist NB GLM with GEE
2. Try simpler distributions (Poisson with robust SE)
3. Consider that N=40 is too small for these approaches

---

## Deliverables

### Code
- [ ] `/workspace/experiments/designer_3/models/model_1_changepoint.stan`
- [ ] `/workspace/experiments/designer_3/models/model_2_gp.stan`
- [ ] `/workspace/experiments/designer_3/models/model_3_statespace.stan`
- [ ] `/workspace/experiments/designer_3/fit_models.py`
- [ ] `/workspace/experiments/designer_3/compare_models.py`

### Results
- [ ] `/workspace/experiments/designer_3/results/loo_comparison.csv`
- [ ] `/workspace/experiments/designer_3/results/posterior_summaries.csv`
- [ ] `/workspace/experiments/designer_3/results/ppc_results.txt`

### Visualizations
- [ ] Posterior distributions for key parameters
- [ ] Fitted values with credible intervals
- [ ] Posterior predictive checks
- [ ] LOO comparison plot

### Reports
- [ ] Final model selection document
- [ ] Sensitivity analysis summary
- [ ] Falsification test results

---

## Timeline

| Phase | Time | Cumulative |
|-------|------|------------|
| Setup | 0.5h | 0.5h |
| Model 1 | 3h | 3.5h |
| Model 2 | 4h | 7.5h |
| Model 3 | 5h | 12.5h |
| Comparison | 3h | 15.5h |
| Refinement | 2h | 17.5h |

**Total estimated time: 17-20 hours**

---

## Success Metrics

**Minimum Success:**
- At least one model converges (Rhat < 1.01)
- LOO-CV ranks models
- Clear winner or equivalence established

**Full Success:**
- All three models converge
- Clear falsification of at least one model
- Winning model passes all posterior predictive checks
- Scientific insight gained about data-generating process

**Outstanding Success:**
- Winning model predicts held-out data accurately
- Results are robust to prior choices
- Clear story emerges about why one model wins
- Recommendations for future data collection

---

**End of Implementation Plan**
