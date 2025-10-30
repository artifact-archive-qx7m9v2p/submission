# Baseline Bayesian Count Regression Models

## Designer 1: Baseline Model Proposals

**Author**: Model Designer 1
**Date**: 2025-10-29
**Focus**: Baseline GLM frameworks for overdispersed count data

---

## Executive Summary

Based on EDA findings (severe overdispersion with Var/Mean = 70.43, exponential growth with R² = 0.937, and high autocorrelation ACF = 0.971), I propose **three progressively complex baseline models**:

1. **Model 1**: Negative Binomial GLM (simple exponential growth)
2. **Model 2**: Negative Binomial GLM with quadratic trend (accelerating growth)
3. **Model 3**: Overdispersed Poisson (Gamma-Poisson mixture, computational alternative)

**Key Design Principle**: Start with standard GLM frameworks WITHOUT temporal correlation. This provides:
- Clean baselines for model comparison
- Diagnostic isolation (separate overdispersion from autocorrelation)
- Computational efficiency for rapid iteration
- Clear failure modes to guide next steps

**Falsification Philosophy**: Each model includes explicit criteria for rejection. Success means discovering WHY a model fails, not confirming it works.

---

## Model 1: Negative Binomial GLM (Log-Linear Growth)

### Theoretical Justification

**Why this model?**
- EDA shows Var ∝ Mean² (quadratic mean-variance), consistent with NegBin
- Log-linear trend R² = 0.937 suggests exponential growth
- Negative Binomial naturally handles overdispersion via dispersion parameter
- Two-parameter distribution more flexible than Poisson
- Standard baseline for count regression

**EDA Evidence**:
- Variance/Mean ratio = 70.43 (Poisson inappropriate)
- Variance = 0.057 × Mean^2.01 (power law with exponent ≈ 2)
- Exponential growth rate: 137% per standardized year
- No zero-inflation (minimum count = 21)

**Domain Plausibility**: Exponential growth with stochastic variation is common in biological/ecological systems, technology adoption, and social contagion processes.

### Mathematical Specification

**Likelihood**:
```
C_i ~ NegativeBinomial(mu_i, phi)
```

**Linear Predictor**:
```
log(mu_i) = beta_0 + beta_1 * year_i
```

**Priors**:
```
beta_0 ~ Normal(log(109.4), 1.0)     # Intercept at mean log count
beta_1 ~ Normal(1.0, 0.5)             # Expected positive growth
phi ~ Gamma(2, 0.1)                   # Overdispersion parameter
```

**Prior Justification**:

1. **beta_0 ~ Normal(log(109.4), 1.0)**:
   - Center: log(109.4) ≈ 4.69 (observed mean count)
   - SD = 1.0 allows ±63% to ±170% of mean (reasonable uncertainty)
   - 95% CI: exp(4.69 ± 2) = [40, 300] covers observed range [21, 269]

2. **beta_1 ~ Normal(1.0, 0.5)**:
   - Center: 1.0 implies exp(1) = 2.7× increase per SD of year
   - SD = 0.5 allows range [0.4, 1.6], i.e., [1.5×, 5×] per year SD
   - Weakly informative: excludes extreme growth (10×) but allows flexibility
   - Based on EDA estimate: log(137/100) ≈ 0.87 near prior mode

3. **phi ~ Gamma(2, 0.1)**:
   - Mean = 2/0.1 = 20 (moderate overdispersion)
   - SD = sqrt(2)/0.1 ≈ 14 (high uncertainty)
   - Allows range [5, 50] covering observed Var/Mean = 70
   - Weakly regularizing: prevents extreme dispersion

**Parameterization Note**: Using Stan's `neg_binomial_2(mu, phi)` parameterization where:
- Mean = mu
- Variance = mu + mu²/phi
- Larger phi = less overdispersion (phi → ∞ approaches Poisson)

### Prior Predictive Distribution

**Expected Data Range**:
- Median mu at year=0: exp(4.69) ≈ 109 (matches observed)
- At year=-1.67 (earliest): exp(4.69 - 1.67) ≈ 36 (observed min = 21) ✓
- At year=+1.67 (latest): exp(4.69 + 1.67) ≈ 320 (observed max = 269) ✓
- Variance: 109 + 109²/20 ≈ 703 (observed ≈ 7,700 higher, but reasonable)

**Prior Predictive Checks**:
1. Generate 100 datasets from prior
2. Check: 90% contain counts in [10, 500]
3. Check: Mean count in [50, 200]
4. Check: Variance/Mean ratio in [5, 100]
5. Check: No datasets with >50% zeros (EDA shows 0% zeros)

**Red Flags**:
- If prior generates mostly zeros or counts >1000, priors too diffuse
- If prior shows no overdispersion, increase phi prior uncertainty

### Falsification Criteria

**REJECT Model 1 if**:
1. **Poor predictive performance**: LOO-ELPD >> 10 worse than competitors
2. **Systematic residual patterns**: ACF of Pearson residuals >0.5 at lag 1
3. **Posterior-prior conflict**: Posterior for phi concentrates at boundary (>90% mass at phi <5 or phi >100)
4. **Underfitting**: Posterior predictive checks show observed data in tail (<5th percentile) of replications
5. **Parameter instability**: R-hat >1.05 or ESS <100 for any parameter

**WARNING SIGNS** (don't reject, but investigate):
- Quadratic trend in residuals vs year (suggests Model 2 needed)
- Heteroscedastic residuals (variance increasing with time)
- Pareto k >0.7 for >10% of observations (influential points)
- Posterior for beta_1 includes zero (no evidence for growth)

**What would make me switch model classes?**
- If phi posterior concentrates near infinity (phi >100), Poisson might suffice (unlikely given EDA)
- If residual ACF remains high (>0.8), need temporal correlation structure (Designer 2's job)
- If posterior predictive checks show extreme misfit, consider:
  - Zero-inflation (if unexpected zeros appear in residuals)
  - Mixture models (if bimodality persists)
  - Time-varying parameters (if non-stationarity detected)

### Stan Implementation

**File**: `models/model_1_negbin_linear.stan`

```stan
data {
  int<lower=0> N;                    // Number of observations
  vector[N] year;                    // Standardized year predictor
  array[N] int<lower=0> C;           // Count response
}

parameters {
  real beta_0;                       // Intercept (log scale)
  real beta_1;                       // Slope (growth rate)
  real<lower=0> phi;                 // Dispersion parameter
}

transformed parameters {
  vector[N] mu;                      // Expected counts

  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i]);
  }
}

model {
  // Priors
  beta_0 ~ normal(log(109.4), 1.0);
  beta_1 ~ normal(1.0, 0.5);
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  // Posterior predictive samples
  array[N] int C_rep;

  // Log-likelihood for LOO
  vector[N] log_lik;

  // Pearson residuals
  vector[N] residuals;

  for (i in 1:N) {
    C_rep[i] = neg_binomial_2_rng(mu[i], phi);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    residuals[i] = (C[i] - mu[i]) / sqrt(mu[i] + mu[i]^2 / phi);
  }
}
```

**Implementation Notes**:
1. Use `exp()` in transformed parameters to avoid numerical issues
2. Store log_lik for LOO-CV calculation
3. Compute Pearson residuals for diagnostics
4. Generate posterior predictive samples for PPCs

**Key Diagnostics to Check**:
- Pairs plot: (beta_0, beta_1, phi) for posterior correlations
- Trace plots: verify mixing and stationarity
- Posterior predictive: overlay C_rep vs C with uncertainty bands

### Computational Cost

**Expected Runtime**: 30-60 seconds on modern laptop
- 4 chains × 1000 warmup + 1000 sampling = 8000 total iterations
- Simple likelihood: ~0.01 sec per iteration
- No correlation structure: matrix operations minimal

**Memory**: <100 MB
- N=40 observations
- 3 parameters + N generated quantities

**Potential Issues**:
1. **None expected** - this is a well-behaved model
2. **If divergences occur**: increase adapt_delta to 0.95
3. **If phi → 0**: model struggling with extreme overdispersion (rare)
4. **If non-convergence**: check prior-data conflict

**Scaling**: Model should work for N up to 10,000 without issues

---

## Model 2: Negative Binomial GLM (Quadratic Trend)

### Theoretical Justification

**Why this model?**
- EDA shows quadratic trend R² = 0.964 > log-linear R² = 0.937
- Growth rate changes over time: 13 → 125 → 65 units/year
- Polynomial allows acceleration/deceleration (non-constant growth)
- Maintains interpretability while adding flexibility
- Tests hypothesis: Is growth rate itself changing?

**EDA Evidence**:
- Quadratic fit achieves R² = 0.964 (best among functional forms)
- Visual inspection shows curvature in log(C) vs year
- Middle period shows explosive growth (acceleration)
- Late period shows moderation (deceleration)
- Changepoint analysis suggests non-stationary growth rate

**Domain Plausibility**:
- S-shaped growth curves common in bounded processes (logistic growth)
- Quadratic on log scale approximates logistic locally
- Technology adoption, epidemic spread often show acceleration then plateau
- Resource constraints can cause growth deceleration

**Alternative Hypothesis**: Apparent quadratic trend could be artifact of:
- Measurement error at extremes
- Unmodeled structural break (single change point)
- True non-parametric smooth trend (GP would be better)

### Mathematical Specification

**Likelihood**:
```
C_i ~ NegativeBinomial(mu_i, phi)
```

**Linear Predictor**:
```
log(mu_i) = beta_0 + beta_1 * year_i + beta_2 * year_i²
```

**Priors**:
```
beta_0 ~ Normal(log(109.4), 1.0)     # Intercept
beta_1 ~ Normal(1.0, 0.5)             # Linear growth
beta_2 ~ Normal(0, 0.5)               # Quadratic term (weakly regularized)
phi ~ Gamma(2, 0.1)                   # Overdispersion
```

**Prior Justification**:

1. **beta_0 ~ Normal(log(109.4), 1.0)**: Same as Model 1

2. **beta_1 ~ Normal(1.0, 0.5)**: Same as Model 1
   - Represents growth rate at year=0 (center of data)

3. **beta_2 ~ Normal(0, 0.5)**:
   - Center at 0: no prior belief in acceleration (skeptical prior)
   - SD = 0.5: allows ±1 unit acceleration (substantial)
   - 95% CI: [-1, +1] means growth rate can change by factor of 2.7 across range
   - Weakly regularizing: prevents overfitting while allowing curvature
   - Based on: Quadratic coefficient from OLS ≈ 0.3, prior allows ±1

4. **phi ~ Gamma(2, 0.1)**: Same as Model 1

**Interpretation**:
- beta_1: instantaneous growth rate at year=0
- beta_2: acceleration (beta_2 >0) or deceleration (beta_2 <0)
- If beta_2 ≈ 0, Model 1 sufficient (parsimony wins)

### Prior Predictive Distribution

**Expected Data Range**:
- At year=0: exp(4.69) ≈ 109 (same as Model 1)
- Quadratic allows more extreme values at boundaries
- If beta_2 = +0.5: counts at year=±1.67 can be 2× Model 1 prediction
- If beta_2 = -0.5: counts at boundaries can be 0.5× Model 1 prediction

**Prior Predictive Checks**:
1. Generate 100 datasets from prior
2. Check: 90% contain counts in [5, 1000] (wider than Model 1)
3. Check: No U-shaped patterns with near-zero middle values
4. Check: Trajectories show reasonable curvature (not wild oscillations)
5. Compare to Model 1 prior: should overlap substantially

**Red Flags**:
- If prior generates extreme U-shapes (min at center), beta_2 prior too diffuse
- If prior identical to Model 1, beta_2 prior too tight

### Falsification Criteria

**REJECT Model 2 if**:
1. **No improvement over Model 1**: ΔLOO < 2 (not worth extra parameter)
2. **Overfitting evidence**: Pareto k >0.7 for >20% of observations
3. **Posterior for beta_2 includes zero**: 95% CI contains 0 → Model 1 sufficient
4. **Extreme parameter values**: |beta_2| >2 suggests misspecification
5. **Residuals still show patterns**: Quadratic trend persists in residuals

**WARNING SIGNS**:
- If beta_2 >1 (extreme acceleration), consider:
  - Structural break more appropriate (Designer 2)
  - Exponential time trend (different model class)
  - Data issues at extremes (measurement error)
- If Model 2 much worse than Model 1, quadratic may be artifact

**What would make me switch model classes?**
- If beta_2 highly uncertain (posterior ≈ prior), suggests non-parametric smoother needed (GP)
- If residuals show abrupt change, prefer changepoint model over smooth polynomial
- If cubic term improves fit substantially, polynomial framework inadequate

### Stan Implementation

**File**: `models/model_2_negbin_quadratic.stan`

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;                       // Quadratic coefficient
  real<lower=0> phi;
}

transformed parameters {
  vector[N] mu;

  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i] + beta_2 * year[i]^2);
  }
}

model {
  // Priors
  beta_0 ~ normal(log(109.4), 1.0);
  beta_1 ~ normal(1.0, 0.5);
  beta_2 ~ normal(0, 0.5);           // Skeptical of quadratic
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  array[N] int C_rep;
  vector[N] log_lik;
  vector[N] residuals;

  // Derived quantity: is quadratic needed?
  int<lower=0, upper=1> quadratic_important;
  quadratic_important = fabs(beta_2) > 0.1 ? 1 : 0;

  for (i in 1:N) {
    C_rep[i] = neg_binomial_2_rng(mu[i], phi);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    residuals[i] = (C[i] - mu[i]) / sqrt(mu[i] + mu[i]^2 / phi);
  }
}
```

**Implementation Notes**:
1. Compute year^2 inside Stan (avoid numerical precision issues)
2. Add indicator for practical significance of beta_2
3. Monitor pairs plot for (beta_1, beta_2) correlation
4. Consider centering year if numerical issues arise (already done in data)

**Key Diagnostics**:
- Marginal posterior for beta_2: does it exclude zero?
- Conditional plot: mu(year) for different beta_2 quantiles
- Residual plot: should show less pattern than Model 1

### Computational Cost

**Expected Runtime**: 45-90 seconds
- Slightly slower than Model 1 (4 parameters vs 3)
- Exponential with quadratic more expensive to evaluate
- Still very fast for N=40

**Memory**: <100 MB (similar to Model 1)

**Potential Issues**:
1. **Collinearity**: beta_1 and beta_2 may be correlated (not problematic with HMC)
2. **Numerical overflow**: If |beta_2| large, exp() may overflow
   - Mitigation: Monitor max(mu) in transformed parameters
   - If max(mu) >10^6, prior too diffuse
3. **Convergence**: May need more iterations if beta_2 poorly identified

**Scaling**: Works for N up to 10,000

---

## Model 3: Overdispersed Poisson (Gamma-Poisson)

### Theoretical Justification

**Why this model?**
- Mechanistic interpretation: Poisson rate λ varies randomly across observations
- Gamma-Poisson mixture is mathematically equivalent to Negative Binomial
- Explicit hierarchical structure makes assumptions clearer
- Alternative parameterization sometimes improves sampling
- Tests whether overdispersion is "extra-Poisson" variation

**EDA Evidence**: Same as Model 1 (both are Negative Binomial)

**Conceptual Difference**:
- **Model 1**: Direct NegBin likelihood (phenomenological)
- **Model 3**: Hierarchical Poisson with random effects (mechanistic)

**Domain Plausibility**:
- If counts are Poisson conditional on true rate, but rate varies
- Example: Disease cases are Poisson, but infection rate varies by location
- Example: Website visits are Poisson, but popularity varies by site
- Implies unobserved heterogeneity in the process

**Why might this FAIL?**
- If overdispersion not due to random variation in rate
- If count process is fundamentally not Poisson (e.g., contagion)
- Adds unnecessary hierarchy if NegBin direct parameterization works

### Mathematical Specification

**Hierarchical Structure**:
```
C_i | lambda_i ~ Poisson(lambda_i)
lambda_i ~ Gamma(alpha, beta)
```

**Marginally** (integrating out lambda_i):
```
C_i ~ NegativeBinomial(mu_i, phi)
```

**Linear Predictor**:
```
log(mu_i) = beta_0 + beta_1 * year_i
```

**Priors**:
```
beta_0 ~ Normal(log(109.4), 1.0)
beta_1 ~ Normal(1.0, 0.5)
alpha ~ Gamma(2, 0.1)              # Shape parameter
```

**Relationship to Model 1**:
- Model 1: NegBin(mu, phi) where phi is dispersion
- Model 3: NegBin via Gamma(alpha, beta) where alpha = phi, beta = phi/mu
- **Mathematically equivalent** but different computational properties

**Prior Justification**: Same as Model 1 (identical model, different implementation)

### Prior Predictive Distribution

**Identical to Model 1** (same marginal distribution)

**Why implement both?**
1. **Sampling efficiency**: Sometimes one parameterization mixes better
2. **Interpretation**: Hierarchical view clearer for some users
3. **Extensibility**: Easier to add random effects later
4. **Robustness check**: If both converge to same posterior, increases confidence

### Falsification Criteria

**REJECT Model 3 if**:
1. **Worse convergence than Model 1**: More divergences, lower ESS
2. **Longer runtime**: If >2× slower without better diagnostics
3. **Posterior differs from Model 1**: Should be nearly identical (sanity check)
4. **Numerical instability**: Gamma parameterization causes overflow

**WARNING SIGNS**:
- If alpha posterior very small (<1), extreme overdispersion may need different model
- If alpha → ∞, indicates Poisson sufficient (unlikely given EDA)

**What would make me switch model classes?**
- Same as Model 1 (equivalent models)
- If both Model 1 and Model 3 struggle with convergence, suggests NegBin inappropriate

### Stan Implementation

**File**: `models/model_3_gamma_poisson.stan`

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> alpha;               // Gamma shape parameter
  vector<lower=0>[N] lambda;         // Random Poisson rates
}

transformed parameters {
  vector[N] mu;

  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i]);
  }
}

model {
  // Priors
  beta_0 ~ normal(log(109.4), 1.0);
  beta_1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);

  // Hierarchical structure
  for (i in 1:N) {
    lambda[i] ~ gamma(alpha, alpha / mu[i]);  // E[lambda] = mu
    C[i] ~ poisson(lambda[i]);
  }
}

generated quantities {
  array[N] int C_rep;
  vector[N] log_lik;
  vector[N] residuals;

  // Convert to NegBin dispersion for comparison with Model 1
  real<lower=0> phi = alpha;

  for (i in 1:N) {
    // Generate from marginal distribution
    C_rep[i] = neg_binomial_2_rng(mu[i], phi);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    residuals[i] = (C[i] - mu[i]) / sqrt(mu[i] + mu[i]^2 / phi);
  }
}
```

**Implementation Notes**:
1. Explicitly model lambda_i as parameters (hierarchical)
2. Gamma parameterization: E[lambda_i] = mu_i
3. In generated quantities, use marginal NegBin for comparison
4. This is the "non-centered" parameterization

**Alternative: Centered Parameterization** (may be more stable):
```stan
parameters {
  vector<lower=0>[N] lambda_raw;     // Unit scale
}

transformed parameters {
  vector<lower=0>[N] lambda;
  for (i in 1:N) {
    lambda[i] = mu[i] * lambda_raw[i];  // Scale by mean
  }
}

model {
  lambda_raw ~ gamma(alpha, alpha);  // Centered at 1
  C ~ poisson(lambda);
}
```

**Key Diagnostics**:
- Compare posterior for phi in Model 3 vs Model 1 (should be identical)
- Check ESS for lambda parameters (hierarchical models can have low ESS)
- Verify mixing for alpha parameter

### Computational Cost

**Expected Runtime**: 2-5 minutes
- **Much slower than Model 1** due to N additional parameters
- N=40 lambda parameters to sample
- Hierarchical structure increases autocorrelation

**Memory**: ~200 MB (larger due to lambda storage)

**Potential Issues**:
1. **Low ESS for lambda**: Common in hierarchical models (not necessarily bad)
2. **Funnel geometry**: If alpha small, lambda posteriors may show funnel
   - Mitigation: Use non-centered parameterization
3. **Divergences**: More likely than Model 1
   - Mitigation: Increase adapt_delta to 0.95 or 0.99

**Why use this if slower?**
- **Don't** unless Model 1 has convergence issues
- Useful as robustness check
- Educational: illustrates Gamma-Poisson mixture

**Scaling**: Struggles for N >1000 (too many lambda parameters)

---

## Model Comparison Framework

### Predictive Performance

**Primary Metric**: LOO-ELPD (Leave-One-Out Expected Log Predictive Density)
- Compute via `loo` package in Python/R
- Compare models: ΔELPD ± SE
- Rule: Prefer simpler model if |ΔELPD| <2

**Pareto k Diagnostics**:
- k <0.5: Good (reliable LOO)
- 0.5 <k <0.7: OK (monitor)
- k >0.7: Bad (influential observation, refit with K-fold CV)

**Secondary Metrics**:
- RMSE: sqrt(mean((C - E[C_rep])²))
- MAE: mean(|C - E[C_rep]|)
- Coverage: % observations in 90% predictive interval

### Posterior Predictive Checks

**Graphical**:
1. Overlay plot: Observed vs predicted with uncertainty bands
2. Rootogram: Hanging rootogram for count distribution
3. Residual plot: Pearson residuals vs year (check for patterns)
4. QQ-plot: Quantiles of observed vs replicated data

**Numerical**:
1. Test statistics: Replicate T(C_rep) vs T(C_obs)
   - Mean, variance, max, min
   - Proportion of exceedances
2. Bayesian p-value: P(T(C_rep) ≥ T(C_obs) | data)
   - Good fit: 0.05 <p <0.95
   - Extreme p (<0.01 or >0.99) indicates misfit

### Convergence Diagnostics

**Essential Checks**:
1. R-hat <1.01 for all parameters (preferably <1.005)
2. ESS_bulk >400 per chain
3. ESS_tail >400 per chain
4. No divergent transitions (0 divergences)
5. Max treedepth not exceeded

**If Convergence Fails**:
1. Increase warmup iterations (1000 → 2000)
2. Increase adapt_delta (0.8 → 0.95 → 0.99)
3. Reparameterize (e.g., non-centered for Model 3)
4. Check prior-data conflict

### Decision Rules

**Model Selection Strategy**:
1. Fit all models
2. Check convergence (reject if R-hat >1.01)
3. Compute LOO for converged models
4. Compare ΔELPD ± SE
5. If |ΔELPD| <2: Choose simpler model (parsimony)
6. If |ΔELPD| >4: Choose better model
7. If 2 <|ΔELPD| <4: Uncertain, report both

**When to Abandon Baseline Framework**:
- All models show systematic residual patterns (ACF >0.5)
  → Need temporal correlation (Designer 2)
- All models show poor PPCs (p-value <0.01)
  → Need fundamentally different likelihood
- Posterior predictive intervals miss >20% of observations
  → Underfitting, need more complex model
- Pareto k >0.7 for >30% of observations
  → Model misspecification, not just outliers

---

## Implementation Plan

### Phase 1: Prior Predictive Checks (1 hour)

**Goal**: Validate priors before seeing data

**Steps**:
1. Implement `prior_predictive_check.py`
2. Sample from priors only (comment out likelihood)
3. Generate 100 datasets per model
4. Visual checks: histograms, time series plots
5. Numerical checks: mean, variance, range

**Success Criteria**:
- Priors generate plausible data (counts in [1, 1000])
- Prior predictive covers observed data range
- No degenerate cases (all zeros, all huge counts)

**If Priors Fail**: Adjust prior parameters, re-run checks

### Phase 2: Model Fitting (2 hours)

**Goal**: Obtain posterior distributions

**Steps**:
1. Implement models in Stan (3 files)
2. Fit with CmdStanPy:
   - 4 chains
   - 1000 warmup + 1000 sampling
   - adapt_delta = 0.8 (increase if divergences)
3. Extract posteriors and diagnostics
4. Save results to `experiments/designer_1/results/`

**Success Criteria**:
- All R-hat <1.01
- ESS >400
- 0 divergent transitions
- Runtime <5 minutes per model

**If Convergence Fails**: See troubleshooting section

### Phase 3: Posterior Diagnostics (1 hour)

**Goal**: Assess model fit and compare

**Steps**:
1. Compute LOO-ELPD for each model
2. Generate posterior predictive samples
3. Create diagnostic plots (trace, pairs, PPC)
4. Calculate residuals and check autocorrelation
5. Compare models via ΔELPD

**Success Criteria**:
- At least one model has Pareto k <0.7 for all observations
- Posterior predictive intervals cover 85-95% of data
- Residual ACF <0.5 at lag 1 (baseline models, don't expect perfect)

### Phase 4: Critique and Decision (30 min)

**Goal**: Decide next steps

**Questions**:
1. Do any models fit adequately?
2. Is Model 2 substantially better than Model 1?
3. Are residuals still highly autocorrelated?
4. What patterns remain unexplained?

**Outcomes**:
- **Success**: Model 1 or 2 fits well → report results
- **Partial**: Good fit but residual correlation → pass to Designer 2
- **Failure**: All models fail PPCs → need different model class

---

## Expected Failures and Contingencies

### Failure Mode 1: High Residual Autocorrelation

**Symptom**: Residual ACF(1) >0.8 despite good fit

**Interpretation**: Overdispersion handled, but temporal dependence not modeled

**Action**:
- Accept this limitation (baseline models ignore correlation)
- Pass to Designer 2 for AR(1) or GP models
- **This is NOT a failure** - we've isolated the two sources of variance

**Do NOT**: Try to "fix" by adding complexity here

### Failure Mode 2: Overfitting in Model 2

**Symptom**: Model 2 has higher LOO-ELPD than Model 1 but Pareto k >0.7

**Interpretation**: Quadratic term fitting to noise, not signal

**Action**:
- Stick with Model 1 (simpler, more robust)
- Report that quadratic term not justified
- Consider smoothing approaches (Designer 3)

**Do NOT**: Force Model 2 just because EDA showed R² improvement

### Failure Mode 3: Extreme Dispersion Parameter

**Symptom**: Posterior for phi concentrates at extreme values (<5 or >100)

**Interpretation**:
- phi <5: Even more overdispersed than NegBin handles
- phi >100: Nearly Poisson (contradicts EDA)

**Action**:
- If phi <5: Consider alternative overdispersion mechanisms
  - Conway-Maxwell Poisson
  - Zero-inflated (check for zeros)
  - Mixture models
- If phi >100: Check data preparation (maybe transformed incorrectly?)

**Do NOT**: Expand prior range to accommodate extreme values

### Failure Mode 4: Poor Posterior Predictive Coverage

**Symptom**: 90% PI contains <80% of observations

**Interpretation**: Model systematically underestimates uncertainty

**Action**:
- Check if missing important predictor
- Consider time-varying dispersion (phi(t))
- Look for structural breaks (Designer 2)

**Do NOT**: Artificially inflate uncertainty by weakening priors

### Failure Mode 5: All Models Equivalent

**Symptom**: ΔELPD <1 between all three models

**Interpretation**: Either all wrong or all equally adequate

**Action**:
- Check PPCs to distinguish
- If PPCs good: Report simplest model (Model 1)
- If PPCs bad: Need different approach entirely

---

## Connection to Other Designers

### Designer 2: Temporal Correlation Models

**What I provide**:
- Baseline overdispersion estimates (phi ≈ 20?)
- Evidence that NegBin family appropriate
- Quantification of residual autocorrelation
- Growth trend estimates (beta_1 ≈ 1.0?)

**What they add**:
- AR(1) or other correlation structures
- State-space formulations
- Dynamic dispersion parameters

**Key Question**: Does adding correlation improve LOO beyond baseline?

### Designer 3: Flexible/Nonparametric Models

**What I provide**:
- Test of polynomial vs exponential trends
- Evidence for/against quadratic term
- Baseline predictive performance

**What they add**:
- Gaussian process smooths
- Splines
- Changepoint models

**Key Question**: Is non-parametric flexibility worth computational cost?

---

## Software Requirements

### Core Dependencies

```python
import cmdstanpy
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats
```

**Versions**:
- cmdstanpy >= 1.2.0
- arviz >= 0.17.0
- numpy >= 1.24.0
- scipy >= 1.11.0

### Stan Installation

```bash
# Install CmdStan
python -m cmdstanpy.install_cmdstan --version 2.33.1

# Verify installation
python -c "import cmdstanpy; print(cmdstanpy.CmdStanModel.cmdstan_version())"
```

### File Structure

```
experiments/designer_1/
├── proposed_models.md          # This file
├── models/
│   ├── model_1_negbin_linear.stan
│   ├── model_2_negbin_quadratic.stan
│   └── model_3_gamma_poisson.stan
├── scripts/
│   ├── prior_predictive_check.py
│   ├── fit_models.py
│   ├── diagnostics.py
│   └── compare_models.py
├── results/
│   ├── model_1_fit.pkl
│   ├── model_2_fit.pkl
│   ├── model_3_fit.pkl
│   ├── loo_comparison.csv
│   └── posterior_summaries.csv
└── figures/
    ├── prior_predictive/
    ├── posterior_predictive/
    ├── diagnostics/
    └── comparison/
```

---

## Summary of Key Decisions

### What I'm Testing

1. **Is Negative Binomial appropriate?** (vs Poisson)
   - Answer: Almost certainly yes (Var/Mean = 70)
2. **Is exponential growth sufficient?** (vs quadratic)
   - Answer: Requires model comparison via LOO
3. **Which NegBin parameterization?** (direct vs hierarchical)
   - Answer: Both should work, use simpler (Model 1)

### What I'm NOT Testing

1. **Temporal correlation** - intentionally ignored for baseline
2. **Non-parametric trends** - staying with parametric GLM
3. **Structural breaks** - continuous models only
4. **Time-varying dispersion** - constant phi assumption
5. **Random effects** - no grouping structure in data

### Critical Assumptions

1. **Counts are independent conditional on trend** (FALSE per EDA)
   - Justification: Isolate overdispersion from correlation
2. **Growth is smooth** (possibly FALSE per EDA)
   - Justification: Simplest baseline, test smooth before rough
3. **Dispersion constant over time** (possibly FALSE per EDA)
   - Justification: Start simple, add complexity if needed
4. **Observations exchangeable** (FALSE per EDA)
   - Justification: Baseline models ignore temporal order

### Success Criteria for Designer 1

**Minimal Success**:
- All models converge (R-hat <1.01)
- At least one model has reasonable LOO
- Clear recommendation for Model 1 vs Model 2

**Full Success**:
- PPCs show good fit (p-value 0.05-0.95)
- Residual patterns characterized
- Clear path forward for Designer 2/3

**Failure Admitted If**:
- All models have Pareto k >0.7
- PPCs show systematic misfit (p <0.01)
- Posteriors conflict with priors (indicates wrong model class)

---

## Falsification Summary

### Global Red Flags (Abandon All Models)

1. **Computational**: None of the 3 models converge after tuning
2. **Predictive**: LOO worse than naive mean model
3. **Theoretical**: Posterior for phi >100 (implies Poisson, contradicts EDA)
4. **Diagnostic**: >50% of observations have Pareto k >0.7

### Model-Specific Reject Criteria

| Model | Reject If | Switch To |
|-------|-----------|-----------|
| Model 1 | LOO >> Model 2 (ΔELPD >4) | Model 2 |
| Model 1 | Quadratic pattern in residuals | Model 2 |
| Model 2 | beta_2 credible interval includes 0 | Model 1 |
| Model 2 | Pareto k >0.7 for >20% obs | Model 1 |
| Model 3 | ESS <100 after tuning | Model 1 |
| Model 3 | Runtime >5× Model 1 | Model 1 |

### Next Phase Triggers

| Finding | Interpretation | Action |
|---------|----------------|--------|
| Residual ACF >0.8 | Correlation not captured | → Designer 2 |
| Model 2 >> Model 1 | Non-linear but not smooth | → Designer 3 (GP) |
| All models fail PPC | Wrong likelihood family | → Revisit EDA |
| Heteroscedastic residuals | Time-varying dispersion | → Designer 2 |
| Abrupt changes in residuals | Structural break | → Designer 3 |

---

## References and Theoretical Background

### Negative Binomial Distribution

**Parameterizations**:
1. **NB1**: Var = φ × μ (linear)
2. **NB2**: Var = μ + μ²/φ (quadratic) ← We use this
3. **NBP**: Var = μ + α × μ^p (power)

**Why NB2?**
- EDA shows Var ∝ Mean² (exponent ≈ 2)
- Standard in count regression (MASS::glm.nb uses this)
- Stan's neg_binomial_2() uses this parameterization

**Generative Process** (Gamma-Poisson mixture):
```
λ_i ~ Gamma(α, β)
C_i | λ_i ~ Poisson(λ_i)
```
Marginalizing λ_i gives NegBin(μ, φ)

### Prior Elicitation Strategy

**Weakly Informative Priors**:
- Goal: Regularize without dominating likelihood
- Rule: Prior should allow for surprising (but plausible) data
- Check: Prior predictive should be diffuse but not absurd

**For Count Data**:
- Log-scale priors more natural (multiplicative effects)
- Normal priors on log-scale → log-normal on raw scale
- Dispersion: Gamma prior (positive, skewed, flexible)

### Model Comparison Philosophy

**Why LOO instead of WAIC?**
- LOO more robust to influential observations
- Pareto k diagnostics identify problematic points
- WAIC can be unstable for small N

**Why not DIC or BIC?**
- DIC: Single point estimate, ignores posterior uncertainty
- BIC: Asymptotic approximation, poor for N=40
- Both penalize complexity poorly for hierarchical models

**Parsimony vs Flexibility**:
- Occam's Razor: Prefer simpler model if fit equivalent
- But: Don't sacrifice prediction for simplicity
- ΔELPD <2: Equivalent (choose simpler)
- ΔELPD >4: Clear winner (choose better)
- 2 <ΔELPD <4: Gray zone (report both)

---

## Conclusion

I have proposed **three baseline Bayesian count regression models** that progressively test:
1. Negative Binomial with exponential growth (standard)
2. Negative Binomial with quadratic trend (flexible)
3. Gamma-Poisson hierarchical (mechanistic)

**Key Design Principles**:
- Start simple, add complexity only if justified
- Explicit falsification criteria for each model
- Computational efficiency for rapid iteration
- Clear connection to EDA findings

**Expected Outcome**:
- Model 1 or 2 will provide adequate baseline fit
- Residual autocorrelation will remain (expected)
- Foundation established for Designer 2/3 to add temporal structure

**Commitment to Falsification**:
- I will reject Model 2 if beta_2 credible interval includes zero
- I will admit baseline framework inadequate if all PPCs fail
- I will not force complexity where simplicity suffices

**Next Steps**:
1. Implement Stan models
2. Run prior predictive checks
3. Fit models to data
4. Compute LOO and diagnostics
5. Report findings with humility about limitations

---

**End of Model Proposals**

Files to be created:
- `/workspace/experiments/designer_1/models/model_1_negbin_linear.stan`
- `/workspace/experiments/designer_1/models/model_2_negbin_quadratic.stan`
- `/workspace/experiments/designer_1/models/model_3_gamma_poisson.stan`
- `/workspace/experiments/designer_1/scripts/fit_models.py`
- `/workspace/experiments/designer_1/scripts/prior_predictive_check.py`
- `/workspace/experiments/designer_1/scripts/diagnostics.py`
