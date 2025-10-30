# Changepoint and Regime-Switching Model Designs
## Model Designer 1: Structural Break Focus

**Date**: 2025-10-29
**Context**: Time series count data (N=40) with extreme structural break at observation 17

---

## Executive Summary

The EDA provides overwhelming evidence for a discrete structural break (730% growth rate increase at observation 17). However, **I am deeply skeptical of obvious patterns in small datasets**. This document proposes three fundamentally different approaches to modeling this break, each with explicit criteria for rejection.

**Core Tension**: The break looks clean in the EDA, but:
- N=40 is small - apparent patterns may be spurious
- Single break assumption may oversimplify reality
- Autocorrelation and overdispersion could create break artifacts
- What if the "break" is actually smooth acceleration?

**My Commitment**: If ANY model shows prior-data conflict, computational pathology, or poor predictive performance, I will advocate for completely different model classes (GP, splines, state-space).

---

## Competing Hypotheses

Before proposing models, I explicitly state three incompatible explanations for the observed pattern:

### H1: True Discrete Regime Change (Changepoint Models)
**Evidence FOR**: Chow test, CUSUM, grid search convergence
**Evidence AGAINST**: Small sample could create spurious breaks
**Falsifiable by**: Smooth transitions fitting better; multiple smaller breaks detected

### H2: Smooth Acceleration (Should NOT use changepoint models)
**Evidence FOR**: Quadratic model nearly as good as two-regime
**Evidence AGAINST**: 80% improvement from discrete break model
**Falsifiable by**: GP/spline models showing continuous curvature, not abrupt change

### H3: Multiple Smaller Breaks (Not just one at t=17)
**Evidence FOR**: Time-varying dispersion suggests multiple regimes
**Evidence AGAINST**: No other breaks detected in grid search
**Falsifiable by**: Multiple changepoint models selecting k=1

**I will let the data decide which hypothesis to abandon.**

---

## Model 1: Discrete Changepoint Negative Binomial (Known Break)

### Philosophical Position
Start with the strongest EDA signal: assume τ=17 is correct. This is the "null hypothesis" - if this fails, the entire changepoint framework is suspect.

### Mathematical Specification

**Likelihood**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β₀ + β₁·year_t + β₂·I(t > 17)·(year_t - year₁₇)
```

Where:
- `C_t`: Count at time t
- `μ_t`: Expected count at time t
- `α`: Dispersion parameter (variance = μ + μ²/α)
- `I(t > 17)`: Indicator for post-break regime
- `year_t`: Standardized year variable
- `year₁₇ ≈ -0.21`: Year value at observation 17

**Autocorrelation Structure** (Critical Addition):

Three sub-variants to test:

**1a. Observation-Level Random Effects** (Conservative):
```
log(μ_t) = β₀ + β₁·year_t + β₂·I(t > 17)·(year_t - year₁₇) + ε_t
ε_t ~ Normal(0, σ_obs)
```

**1b. AR(1) Latent Process** (More structured):
```
log(μ_t) = η_t
η_t ~ Normal(β₀ + β₁·year_t + β₂·I(t > 17)·(year_t - year₁₇) + ρ·η_{t-1}, σ_η)
```

**1c. Hybrid AR(1) + Overdispersion** (Most flexible):
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = η_t
η_t ~ Normal(β₀ + β₁·year_t + β₂·I(t > 17)·(year_t - year₁₇) + ρ·η_{t-1}, σ_η)
```

### Prior Recommendations

**Based on EDA but weakly informative to allow falsification**:

```stan
// Intercept (log-count at year=0, which is mid-series)
β₀ ~ Normal(4.3, 0.5)  // EDA: log(74.5) ≈ 4.31

// Pre-break slope (gentle growth)
β₁ ~ Normal(0.35, 0.3)  // EDA: ≈0.3-0.4, but allow wide range

// Post-break additional slope (steep acceleration)
β₂ ~ Normal(1.0, 0.5)  // EDA: post-pre ≈ 0.8-1.0, allow uncertainty

// Dispersion (NB parameterization: φ = 1/α)
α ~ Gamma(2, 1)  // EDA: α ≈ 0.61, but allow 2-3x variation

// Autocorrelation (if AR model)
ρ ~ Beta(8, 2)  // Strongly positive but not extreme

// Observation noise
σ_obs ~ Exponential(1)  // Weakly informative
σ_η ~ Exponential(2)    // Tighter for AR process
```

**Prior Justification**:
- Centers on EDA estimates but allows 2-3 SD range to encompass model misspecification
- Avoids overly strong priors that would mask poor fit
- Tests whether model can recover EDA patterns from quasi-uninformative priors

### Falsification Criteria (When to ABANDON this model)

**I will reject Model 1 if**:

1. **Prior-Posterior Conflict**:
   - Any parameter posterior centered >2 SD from prior mode
   - Suggests model structure fighting the data

2. **Residual Autocorrelation Remains**:
   - Posterior predictive ACF(1) > 0.3 after accounting for AR terms
   - Model failed to capture temporal dependence

3. **Dispersion Parameter Explosion**:
   - Posterior α < 0.1 or α > 5.0
   - Suggests NB distribution inadequate (need zero-inflation or alternative)

4. **Computational Pathology**:
   - Divergent transitions > 1% after tuning
   - Rhat > 1.01 for any parameter
   - ESS < 100 for any parameter
   - **These are canaries for model misspecification, not just tuning issues**

5. **Predictive Failure**:
   - LOO-CV Pareto k > 0.7 for >5% of observations
   - Posterior predictive p-value < 0.05 or > 0.95 for break magnitude
   - Out-of-sample RMSE > 1.5x in-sample RMSE (on log scale)

6. **Break Appears Spurious**:
   - 95% CI for β₂ includes zero
   - Bayes Factor BF₁₀ < 3 vs. single-trend model

7. **Heteroscedasticity Not Captured**:
   - Pearson residuals show clear trend with fitted values
   - Suggests need for time-varying dispersion

**Escape Route**: If rejected, pivot to Model 2 (unknown changepoint) or abandon changepoint framework entirely for GP/spline models.

### Implementation Details

**Stan Implementation** (Preferred):
```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
  int<lower=1,upper=N> tau;  // Fixed at 17
}

transformed data {
  vector[N] year_post = rep_vector(0, N);
  real year_tau = year[tau];
  for (t in (tau+1):N) {
    year_post[t] = year[t] - year_tau;
  }
}

parameters {
  real beta0;
  real beta1;
  real beta2;
  real<lower=0> alpha;
  real<lower=0,upper=1> rho;  // AR(1) coefficient
  real<lower=0> sigma_eta;
  vector[N] eta_raw;  // Non-centered parameterization
}

transformed parameters {
  vector[N] eta;
  vector[N] mu;

  // AR(1) process
  eta[1] = beta0 + beta1 * year[1] + sigma_eta * eta_raw[1];
  for (t in 2:N) {
    real trend = beta0 + beta1 * year[t] + beta2 * year_post[t];
    eta[t] = rho * eta[t-1] + (1-rho) * trend + sigma_eta * eta_raw[t];
  }

  mu = exp(eta);
}

model {
  // Priors
  beta0 ~ normal(4.3, 0.5);
  beta1 ~ normal(0.35, 0.3);
  beta2 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 1);
  rho ~ beta(8, 2);
  sigma_eta ~ exponential(2);
  eta_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], alpha);
    C_rep[t] = neg_binomial_2_rng(mu[t], alpha);
  }
}
```

**Computational Considerations**:
- Non-centered parameterization for AR process (critical for sampling efficiency)
- Expected sampling time: 2-5 minutes (4 chains, 2000 iterations)
- Watch for divergences in early iterations - indicates need for stronger priors or model rethinking
- Memory: <100 MB

**Expected Challenges**:
1. AR(1) + NB can have identifiability issues if autocorrelation is weak
2. Changepoint creates discontinuity that may conflict with smooth AR process
3. Small N=40 means limited information for separating AR from overdispersion

### Model Validation Strategy

1. **Prior Predictive Checks**:
   - Generate 1000 datasets from prior
   - Verify: variance/mean ratios 10-200, plausible break magnitudes

2. **Simulation-Based Calibration**:
   - Simulate 100 datasets with known τ=17, known β₂
   - Can model recover true changepoint effect?
   - If not, model is fundamentally unidentified

3. **Posterior Predictive Checks**:
   - ACF plot: posterior predictive ACF should match observed
   - Break magnitude: posterior predictive break size vs. observed 730%
   - Variance-mean relationship: should show NB pattern

4. **Sensitivity Analysis**:
   - Vary priors by 2x in either direction
   - If posteriors change substantially, model is prior-dependent (bad sign)

### Why This Model Might FAIL

**Most Likely Failure Modes**:

1. **Break is artifact of measurement**: If year 17 had a one-time event (data collection change, definition shift), the "regime" doesn't exist

2. **Multiple smaller breaks**: If there are 3-4 smaller breaks, forcing single break will create residual structure

3. **AR(1) insufficient**: ACF(1)=0.944 suggests near-unit-root behavior; AR(1) may be inadequate

4. **Dispersion heterogeneity**: EDA shows 6-fold dispersion variation; single α may fail

5. **Exponential growth assumption wrong**: Log-link assumes multiplicative growth; additive processes would break this

**What I'm Most Worried About**: The break being "too clean" in a small dataset. Overfitting to noise would create strong posterior inference but terrible predictions.

---

## Model 2: Unknown Changepoint Location (Bayesian Changepoint Detection)

### Philosophical Position
**Challenge the EDA**: What if τ=17 is a post-hoc selection artifact? This model lets the data determine break location, possibly finding NO break or MULTIPLE breaks.

### Mathematical Specification

**Likelihood** (same as Model 1):
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β₀ + β₁·year_t + β₂·I(t > τ)·(year_t - year_τ)
```

**Key Difference**: τ is now a parameter, not fixed at 17.

**Changepoint Parameter**:
```
τ ~ DiscreteUniform(5, 35)  // Exclude extreme endpoints
```

### Prior Recommendations

Same as Model 1, plus:

```stan
// Changepoint location (discrete parameter)
τ ~ DiscreteUniform(5, 35)
// Alternatively, concentrate around EDA estimate:
// τ ~ DiscreteUniform(12, 22)  // ±5 observations around 17
```

**Prior Justification**:
- Exclude t<5 and t>35 to ensure sufficient data on both sides
- Uniform prior reflects uncertainty about exact location
- Alternative: Concentrated prior tests if data alone would find τ=17

### Falsification Criteria (When to ABANDON this model)

**I will reject Model 2 if**:

1. **Posterior τ is Uniform** (Model finds no break):
   - Posterior P(τ=17) < 0.2 despite EDA evidence
   - Suggests break is artifact OR smooth process is better

2. **Posterior τ Multimodal** (Multiple breaks):
   - Posterior has 2+ peaks with probability >0.15 each
   - Suggests need for multiple changepoint model

3. **Computational Disaster**:
   - Discrete parameter sampling takes >30 minutes
   - Divergent transitions >5%
   - Suggests model is ill-posed

4. **Same Failures as Model 1**: All falsification criteria from Model 1 apply

5. **Prior-Posterior τ Mismatch**:
   - Uniform prior → Highly concentrated posterior BUT poor predictive performance
   - Suggests overfitting to noise

6. **Sensitivity to Prior Range**:
   - Changing τ prior range from [5,35] to [10,30] substantially changes posterior
   - Model is unstable

**Escape Route**: If τ posterior is diffuse or multimodal, abandon discrete changepoint framework. Move to multiple changepoint model (Model 3) or continuous alternatives (GP, splines).

### Implementation Details

**Stan Implementation** (with Marginalization):

Discrete parameters in Stan require marginalization over τ:

```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
  int<lower=1,upper=N-1> tau_min;  // Minimum changepoint
  int<lower=2,upper=N> tau_max;    // Maximum changepoint
}

parameters {
  real beta0;
  real beta1;
  real beta2;
  real<lower=0> alpha;
  real<lower=0,upper=1> rho;
  real<lower=0> sigma_eta;
  matrix[N, tau_max-tau_min+1] eta_raw;  // For each τ
}

model {
  vector[tau_max-tau_min+1] lp_tau;  // Log probability for each τ

  // Priors
  beta0 ~ normal(4.3, 0.5);
  beta1 ~ normal(0.35, 0.3);
  beta2 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 1);
  rho ~ beta(8, 2);
  sigma_eta ~ exponential(2);
  to_vector(eta_raw) ~ std_normal();

  // Marginalize over τ
  for (tau_idx in 1:(tau_max-tau_min+1)) {
    int tau = tau_min + tau_idx - 1;
    vector[N] year_post = rep_vector(0, N);
    vector[N] eta;
    vector[N] mu;
    real year_tau = year[tau];

    // Construct year_post for this τ
    for (t in (tau+1):N) {
      year_post[t] = year[t] - year_tau;
    }

    // AR(1) process for this τ
    eta[1] = beta0 + beta1 * year[1] + sigma_eta * eta_raw[1, tau_idx];
    for (t in 2:N) {
      real trend = beta0 + beta1 * year[t] + beta2 * year_post[t];
      eta[t] = rho * eta[t-1] + (1-rho) * trend + sigma_eta * eta_raw[t, tau_idx];
    }

    mu = exp(eta);

    // Log probability for this τ (uniform prior on τ)
    lp_tau[tau_idx] = sum(neg_binomial_2_lpmf(C | mu, alpha));
  }

  // Marginalize: log sum exp
  target += log_sum_exp(lp_tau);
}

generated quantities {
  // Sample τ from posterior
  vector[tau_max-tau_min+1] lp_tau;
  int<lower=tau_min,upper=tau_max> tau;

  // Recompute lp_tau (code omitted for brevity - same as model block)
  // ...

  tau = tau_min + categorical_rng(softmax(lp_tau)) - 1;
}
```

**WARNING**: This is computationally expensive!

**Alternative: PyMC Implementation** (Simpler for discrete parameters):

```python
import pymc as pm
import numpy as np

with pm.Model() as changepoint_model:
    # Changepoint location (discrete)
    tau = pm.DiscreteUniform('tau', lower=5, upper=35)

    # Regression parameters
    beta0 = pm.Normal('beta0', mu=4.3, sigma=0.5)
    beta1 = pm.Normal('beta1', mu=0.35, sigma=0.3)
    beta2 = pm.Normal('beta2', mu=1.0, sigma=0.5)
    alpha = pm.Gamma('alpha', alpha=2, beta=1)

    # Autocorrelation
    rho = pm.Beta('rho', alpha=8, beta=2)
    sigma_eta = pm.Exponential('sigma_eta', lam=2)

    # Latent AR process
    eta = pm.AR('eta', rho=[rho], sigma=sigma_eta, shape=N,
                init_dist=pm.Normal.dist(0, 1))

    # Changepoint indicator
    year_post = pm.math.switch(pm.math.ge(t_index, tau),
                                year - year[tau], 0)

    # Mean function
    mu = pm.math.exp(beta0 + beta1*year + beta2*year_post + eta)

    # Likelihood
    obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha, observed=C)

    # Sample
    trace = pm.sample(2000, tune=2000, target_accept=0.95)
```

**Computational Considerations**:
- Stan marginalization: 10-30 minutes (marginalizing over 30 possible τ values)
- PyMC discrete sampling: 5-15 minutes with careful tuning
- Memory: 200-500 MB (storing states for each τ)
- **Recommendation**: Use PyMC for this model (better discrete parameter support)

### Model Validation Strategy

Same as Model 1, plus:

1. **Changepoint Recovery Test**:
   - Simulate data with τ=10, 17, 25
   - Can model correctly identify true τ within ±2 observations?
   - If not, model lacks resolution

2. **Prior Sensitivity**:
   - Run with Uniform(5,35), Uniform(12,22), and Concentrated(17, σ=2)
   - Posteriors should be similar if data is informative

3. **Model Comparison**:
   - Compare LOO against Model 1 (known τ=17)
   - If ΔLOO < 2, unknown τ doesn't improve fit (Occam's razor favors Model 1)

### Why This Model Might FAIL

**Most Likely Failure Modes**:

1. **Computational intractability**: Marginalizing over 30 τ values with AR process may not converge

2. **Overfitting**: With 40 observations and ~30 possible changepoints, model could find spurious break

3. **Single changepoint assumption wrong**: If there are multiple breaks, model will pick strongest but fit poorly

4. **Prior dependence**: Discrete uniform prior may be too vague; posterior could be sensitive to choice

5. **Identification issues**: With AR(1) + changepoint + overdispersion, model may confuse autocorrelation with regime shifts

**What I'm Most Worried About**: Computational cost making this impractical, forcing us back to Model 1 without validating the τ=17 assumption.

---

## Model 3: Multiple Changepoint Model (Hierarchical)

### Philosophical Position
**Most radical challenge**: What if the 730% increase is itself composed of multiple smaller regime shifts? The EDA only looked for a single break. This model allows k=0,1,2,... changepoints.

### Mathematical Specification

**Reversible-Jump Framework** (Number of changepoints is unknown):

```
C_t ~ NegativeBinomial(μ_t, α)
k ~ Poisson(λ=2)  // Expected number of changepoints
τ₁, ..., τₖ ~ Ordered[1,N]  // Ordered changepoint locations
β₀⁽⁰⁾, β₁⁽⁰⁾ ~ Normal(...)  // Pre-first-break regime
β₁⁽ʲ⁾ ~ Normal(...) for j=1,...,k  // Slope in regime j
```

**Piecewise Linear on Log Scale**:

For observation t in regime j (between τⱼ and τⱼ₊₁):
```
log(μ_t) = β₀ + Σᵢ₌₁ʲ [β₁⁽ⁱ⁾ · (min(year_t, year_τᵢ₊₁) - year_τᵢ)]
```

**Simplified Alternative** (Fixed k, easier to implement):

Assume k=2 changepoints and compare via model selection:

```
log(μ_t) = β₀ + β₁·year_t + β₂·I(t>τ₁)·(year_t - year_τ₁)
           + β₃·I(t>τ₂)·(year_t - year_τ₂)
```

With constraint τ₁ < τ₂.

### Prior Recommendations

**For k=2 fixed version**:

```stan
// First changepoint (early)
τ₁ ~ DiscreteUniform(5, 25)

// Second changepoint (must be after first)
τ₂ ~ DiscreteUniform(τ₁ + 3, 35)  // At least 3 obs apart

// Baseline
β₀ ~ Normal(4.3, 0.5)

// Initial slope
β₁ ~ Normal(0.2, 0.3)

// First break magnitude
β₂ ~ Normal(0.5, 0.5)

// Second break magnitude
β₃ ~ Normal(0.5, 0.5)

// Dispersion
α ~ Gamma(2, 1)

// AR parameters (same as before)
ρ ~ Beta(8, 2)
σ_η ~ Exponential(2)
```

**Prior Justification**:
- Allow for two breaks of moderate magnitude rather than one huge break
- Enforce temporal ordering
- Weakly informative on break sizes

**For variable k version** (reversible-jump):
- k ~ Poisson(2): Expect ~2 breaks but allow 0-5
- Hierarchical prior on β⁽ʲ⁾ to share information across regimes

### Falsification Criteria (When to ABANDON this model)

**I will reject Model 3 if**:

1. **Posterior Strongly Favors k=1**:
   - P(k=1 | data) > 0.8 in reversible-jump version
   - ΔLOO < 4 comparing k=2 vs. k=1 fixed models
   - Suggests single break is sufficient; added complexity unjustified

2. **Changepoints Too Close**:
   - Posterior |τ₂ - τ₁| < 5 observations
   - Suggests model is splitting a single break artificially

3. **Break Magnitudes Inconsistent**:
   - One β large, others near zero (suggests model collapsing to k=1)
   - OR all β's similar (suggests smooth trend, not discrete breaks)

4. **Computational Failure**:
   - Reversible-jump doesn't mix (acceptance rate <5%)
   - Fixed k=2 model takes >1 hour to converge
   - ESS <50 for changepoint locations

5. **Same Core Failures**: All Model 1 and Model 2 falsification criteria apply

6. **Predictive Degradation**:
   - Out-of-sample prediction WORSE than simpler k=1 model
   - Classic overfitting signature

**Escape Route**: If k=1 is strongly favored, revert to Model 1 or 2. If computational issues arise, this suggests the changepoint framework itself is inadequate - pivot to GP or state-space models.

### Implementation Details

**Stan Implementation** (k=2 Fixed):

Similar to Model 2 but with two changepoints. Marginalization becomes much more expensive (N² possible configurations).

**Recommended Approach**: Use ordered vector trick:

```stan
parameters {
  ordered[2] tau_raw;  // Ordered changepoints on [0,1]
  // ... other parameters same as Model 1
}

transformed parameters {
  array[2] int<lower=5,upper=35> tau;
  tau[1] = 5 + floor((35-5) * tau_raw[1]);
  tau[2] = tau[1] + 3 + floor((35-tau[1]-3) * (tau_raw[2]-tau_raw[1]) / (1-tau_raw[1]));
  // Ensures tau[1] < tau[2] with sufficient spacing
}
```

This is still very tricky. **More practical: PyMC or bsts (R package)**.

**PyMC Implementation** (k=2 Fixed):

```python
with pm.Model() as two_changepoint_model:
    # Two ordered changepoints
    tau1 = pm.DiscreteUniform('tau1', lower=5, upper=25)
    tau2 = pm.DiscreteUniform('tau2', lower=tau1+3, upper=35)

    # Parameters
    beta0 = pm.Normal('beta0', mu=4.3, sigma=0.5)
    beta1 = pm.Normal('beta1', mu=0.2, sigma=0.3)
    beta2 = pm.Normal('beta2', mu=0.5, sigma=0.5)
    beta3 = pm.Normal('beta3', mu=0.5, sigma=0.5)
    alpha = pm.Gamma('alpha', alpha=2, beta=1)

    # AR process
    rho = pm.Beta('rho', alpha=8, beta=2)
    sigma_eta = pm.Exponential('sigma_eta', lam=2)
    eta = pm.AR('eta', rho=[rho], sigma=sigma_eta, shape=N)

    # Piecewise linear trend
    year_post1 = pm.math.switch(pm.math.ge(t_index, tau1),
                                 year - year[tau1], 0)
    year_post2 = pm.math.switch(pm.math.ge(t_index, tau2),
                                 year - year[tau2], 0)

    mu = pm.math.exp(beta0 + beta1*year + beta2*year_post1 + beta3*year_post2 + eta)

    # Likelihood
    obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha, observed=C)
```

**Computational Considerations**:
- Expected time: 20-60 minutes (complex discrete parameter space)
- Memory: 500 MB - 1 GB
- May require custom sampling (Metropolis steps for τ's)
- **High risk of non-convergence**

**Recommendation**: Only attempt this if Models 1 and 2 both fail falsification tests.

### Model Validation Strategy

1. **Model Selection**:
   - LOO comparison: k=0 (no breaks) vs. k=1 vs. k=2
   - If ΔLOO(k=1, k=2) < 4, prefer simpler k=1

2. **Changepoint Interpretation**:
   - If τ₁ ≈ 10-12 and τ₂ ≈ 17-20, breaks may have scientific meaning
   - If τ's are at extremes or overlapping, model is unstable

3. **Break Magnitude Tests**:
   - Do β₂ and β₃ sum to approximately the single-break β₂ from Model 1?
   - If yes, model is decomposing one break; if no, genuinely finding two processes

4. **Posterior Predictive**:
   - Does k=2 model reproduce observed pattern better than k=1?
   - Check growth rate plot with regime indicators

### Why This Model Might FAIL

**Most Likely Failure Modes**:

1. **Overfitting**: N=40 is too small to reliably detect 2+ changepoints

2. **Identifiability**: Two close changepoints confounded with autocorrelation

3. **Computational intractability**: Discrete parameter space too large to sample effectively

4. **Occam's razor**: Single changepoint is sufficient; extra complexity buys nothing

5. **Model is "too flexible"**: Can fit any pattern, including noise, leading to poor generalization

**What I'm Most Worried About**: This model is the "kitchen sink" approach. If it's needed, it suggests the data generation process is too complex for our sample size, and we should abandon parametric changepoint models entirely in favor of non-parametric alternatives (GP, random walk, state-space).

---

## Model Comparison and Selection Strategy

### Decision Tree

```
START
 │
 ├─ Fit Model 1 (known τ=17)
 │   ├─ PASS falsification tests?
 │   │   ├─ YES → Accept Model 1, report with caveats
 │   │   └─ NO → Continue
 │   │
 │   └─ Which failure mode?
 │       ├─ Computational pathology → Try simpler variants (no AR)
 │       ├─ Break not significant → Try Model 3 (multiple breaks) OR abandon changepoints
 │       └─ Residual structure → Strengthen autocorrelation model
 │
 ├─ Fit Model 2 (unknown τ)
 │   ├─ Does posterior concentrate near τ=17?
 │   │   ├─ YES → Validates Model 1; use Model 2 for inference
 │   │   └─ NO → Changepoint location uncertain; try Model 3 OR abandon
 │   │
 │   └─ Posterior shape diagnostic
 │       ├─ Unimodal → Good, proceed with inference
 │       ├─ Bimodal → Try Model 3 (two changepoints)
 │       └─ Uniform → No changepoint detected; ABANDON changepoint framework
 │
 └─ Fit Model 3 (multiple changepoints)
     ├─ Only if Models 1 and 2 suggest complexity
     ├─ Compare k=0, k=1, k=2 via LOO
     └─ If k=1 preferred → Revert to Model 2
         If k=2+ preferred → Report but FLAG overfitting risk
         If k=0 preferred → ABANDON changepoint framework
```

### Quantitative Comparison Criteria

**LOO Cross-Validation** (Accounting for temporal structure):

```
ΔLOO = LOO(model_complex) - LOO(model_simple)

Decision rules:
- ΔLOO < -4: Strong evidence for complex model
- -4 ≤ ΔLOO ≤ 4: Equivalent; prefer simpler
- ΔLOO > 4: Strong evidence for simple model
```

**Pareto k Diagnostic**:
- Good: k < 0.5 for all observations
- Moderate: k ∈ [0.5, 0.7] for <10% of observations
- Bad: k > 0.7 for >10% of observations → Model misspecified

**Predictive Performance** (Time-series CV):

Train on observations 1:30, predict 31:40:
```
RMSE_log = sqrt(mean((log(C_obs) - log(C_pred))²))
MAE_log = mean(|log(C_obs) - log(C_pred)|)
Coverage = fraction of obs within 95% predictive interval
```

**Bayes Factors** (Between model classes):

```
BF₁₀ = exp(ΔLOO/2)  // Approximate BF

Interpretation:
- BF > 10: Strong evidence for model 1
- 3 < BF < 10: Moderate evidence
- 1/3 < BF < 3: Equivalent
- BF < 1/3: Evidence against model 1
```

### Final Model Selection Rules

**I will recommend a model ONLY if**:

1. Passes ALL falsification tests
2. LOO Pareto k < 0.7 for >90% of observations
3. Predictive coverage ≈ 0.95 (well-calibrated)
4. Rhat < 1.01 and ESS > 400 for all parameters
5. No prior-posterior conflict (PPP checks pass)
6. Simpler models definitively rejected (ΔLOO > 4)

**If NO model passes**: I will advocate for completely different approaches (Gaussian Process, Bayesian structural time series, splines, or state-space models) rather than forcing a changepoint framework on unwilling data.

---

## Critical Reflection and Red Flags

### Why I Might Be Completely Wrong

**Scenario 1: The break is an artifact**
- Small N + post-hoc selection + multiple comparisons → spurious pattern
- **Test**: Permutation test on growth rates
- **Escape**: GP or polynomial models without changepoint

**Scenario 2: Smooth acceleration, not discrete break**
- Human pattern recognition sees breaks in smooth curves
- **Test**: Compare GP (smooth) vs. changepoint via LOO
- **Escape**: Abandon changepoint models entirely

**Scenario 3: Measurement heterogeneity**
- Data collection method changed at year 17 (not true regime shift)
- **Test**: Check for metadata on collection methods
- **Escape**: Cannot model this; requires domain knowledge

**Scenario 4: Autocorrelation is non-stationary**
- AR(1) with ρ ≈ 0.944 suggests near unit root
- **Test**: Random walk model vs. AR(1)
- **Escape**: State-space model with time-varying trends

**Scenario 5: Zero-inflation or alternative count distribution**
- NB may be inadequate despite passing EDA tests
- **Test**: Zero-inflated NB, COM-Poisson, or Poisson-lognormal
- **Escape**: Reparameterize likelihood family

### Red Flags That Would Make Me Abandon Everything

1. **Prior predictive generates data nothing like observed** → Model structure fundamentally wrong

2. **Posterior predictive fails to capture break** → Likelihood inadequate

3. **Parameter estimates wildly inconsistent with EDA** → Model-data conflict

4. **All models have Pareto k > 0.7 for same observations** → Those points are outliers or model class is wrong

5. **Computational issues persist across all formulations** → Model class is pathologically unidentified

6. **Out-of-sample prediction catastrophically bad** → Overfitting; need regularization or simpler models

### Alternative Model Classes (If changepoint framework fails)

**Plan B: Gaussian Process Models**
- GP on log(μ_t) with Matérn kernel
- Naturally handles autocorrelation and smooth/sharp transitions
- No changepoint assumption

**Plan C: Bayesian Structural Time Series**
- Local linear trend + seasonal (if applicable)
- Time-varying slopes naturally capture acceleration
- Handles non-stationarity better than AR(1)

**Plan D: Spline Models**
- Natural cubic splines or P-splines on year
- Flexible but regularized
- No explicit break needed

**Plan E: State-Space Models**
- Dynamic Linear Model with time-varying coefficients
- β₁(t) evolves as random walk
- Captures gradual and sudden changes

**When to pivot**: If 2+ of my proposed models fail falsification tests, especially if failures suggest systematic issues (computational, identifiability, predictive), I will immediately propose Plan B-E alternatives.

---

## Implementation Priorities

### Phase 1: Validation (Before fitting real data)

1. **Prior predictive checks** for all models
2. **Simulation-based calibration** for Model 1 (simplest)
3. **Identifiability analysis**: Can we separate AR from overdispersion?

### Phase 2: Model Fitting (Sequential, stop if any passes)

1. **Model 1** (known τ=17):
   - Start with variant 1a (random effects, simplest)
   - If fails, try 1b (AR1), then 1c (AR1 + overdispersion)

2. **Model 2** (unknown τ):
   - Only if Model 1 raises concerns about τ=17
   - Use PyMC for discrete parameter support

3. **Model 3** (multiple τ):
   - Only if Models 1 and 2 strongly suggest multiple breaks
   - Start with k=2 fixed, not reversible-jump

### Phase 3: Model Criticism (Intensive)

1. **Posterior predictive checks**: 10+ test statistics
2. **LOO cross-validation**: All models
3. **Time-series CV**: Train on 1:30, predict 31:40
4. **Sensitivity analysis**: Prior robustness
5. **Counterfactual analysis**: What if τ=10 or τ=25?

### Phase 4: Decision

- Select best model OR
- Reject all and propose alternative model classes

### Estimated Timeline

- Phase 1: 4-6 hours (coding + simulation)
- Phase 2: 6-12 hours (fitting + debugging)
- Phase 3: 8-12 hours (validation + visualization)
- Phase 4: 2-4 hours (writing + decision)

**Total**: 20-34 hours for rigorous model comparison

**Bottleneck**: Model 2 and 3 computational costs; may require simplification

---

## Expected Outcomes

### Best Case Scenario
Model 1 (known τ=17) passes all tests, showing:
- Clean convergence
- Well-calibrated predictions
- Changepoint effect strongly identified
- Autocorrelation adequately captured
- **Conclusion**: Discrete break at observation 17 is real and well-modeled

### Moderate Case Scenario
Model 2 (unknown τ) finds τ ≈ 15-19 with uncertainty:
- Validates break exists but exact location uncertain
- Suggests break happened over 3-5 observations, not instantly
- **Conclusion**: Regime shift is real but gradual, not discrete
- **Action**: Consider smooth transition model as alternative

### Pessimistic Case Scenario
All models fail falsification tests:
- Computational issues suggest fundamental identifiability problems
- Predictive performance poor despite good in-sample fit
- **Conclusion**: Changepoint framework inappropriate
- **Action**: Pivot to GP or state-space models immediately

### Disaster Scenario
Data generation process is incompatible with all proposed approaches:
- Measurement artifacts, not real phenomenon
- Non-stationary process that can't be modeled with 40 observations
- **Conclusion**: Dataset is inadequate for reliable inference
- **Action**: Request more data or acknowledge severe limitations

---

## Summary and Commitment

I propose three Bayesian changepoint models, each with explicit falsification criteria. **I commit to**:

1. **Abandoning models that fail tests**, not tweaking until they "work"
2. **Reporting uncertainty honestly**, even if it means "we don't know"
3. **Pivoting to alternative model classes** if changepoint framework is inadequate
4. **Prioritizing predictive performance** over in-sample fit
5. **Documenting all failures** as learning opportunities

**Success is not fitting a model - success is discovering whether the changepoint hypothesis is true.**

If the data tells me the break is spurious, smooth, or multiple, I will listen and change approaches accordingly. The goal is truth, not task completion.

---

## File Outputs

This analysis will generate:

1. `/workspace/experiments/designer_1/model1_implementation.stan` - Model 1 Stan code
2. `/workspace/experiments/designer_1/model2_implementation.py` - Model 2 PyMC code
3. `/workspace/experiments/designer_1/prior_predictive_checks.py` - Validation script
4. `/workspace/experiments/designer_1/falsification_tests.py` - Automated test suite
5. `/workspace/experiments/designer_1/simulation_study.py` - SBC and recovery tests
6. `/workspace/experiments/designer_1/model_comparison.md` - Results and decision

**All code will use Stan or PyMC as required.**

---

**END OF PROPOSED MODELS DOCUMENT**

*Generated by Model Designer 1 (Changepoint/Regime-Switching Focus)*
*Date: 2025-10-29*
*Philosophy: Falsification over confirmation*
