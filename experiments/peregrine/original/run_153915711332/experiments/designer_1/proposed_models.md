# Bayesian Model Design: Distributional Choices & Variance Structure
## Model Designer 1 - Time Series Count Data

**Designer:** Model Designer 1
**Focus Area:** Distributional choices and variance structure
**Data:** `/workspace/data/data_designer_1.csv`
**Date:** 2025-10-29
**Observations:** n = 40

---

## Executive Summary

I propose three competing Bayesian model classes that make fundamentally different assumptions about the data generation process. The extreme overdispersion (Var/Mean = 67.99) and heteroscedasticity (26× variance increase) are the central challenges. Each model class represents a distinct hypothesis about what causes this extreme variance:

1. **Heterogeneous Negative Binomial**: Variance is intrinsic to the count process, varying over time
2. **Gamma-Poisson Mixture (Random Effects)**: Variance comes from unobserved heterogeneity
3. **Compound Poisson**: Extreme counts arise from clustered event processes

**Critical stance:** I expect at least one of these models to fail catastrophically. The question is which failure mode reveals the true data generation process.

---

## Problem Formulation: Competing Hypotheses

### Core Mystery: Why is variance 68× larger than mean?

**Hypothesis 1: Time-varying process variance**
The count generation process itself becomes more variable over time. Early observations cluster tightly (SD=12.6), late observations scatter widely (SD=64.3). The Negative Binomial dispersion parameter θ may decrease over time.

**Hypothesis 2: Latent heterogeneity**
All observations are drawn from a Poisson process, but each has a different unobserved rate drawn from a Gamma distribution. This naturally produces a Negative Binomial marginal distribution. The heterogeneity may be structured (e.g., random walk in log-rates).

**Hypothesis 3: Compound process**
Counts result from clusters of events. E.g., each time point has λ clusters, each cluster contributes ~Poisson(μ) events. This produces a different variance structure than standard NB.

**Null hypothesis (to falsify):**
Simple overdispersed count model with constant dispersion is sufficient. The heteroscedasticity is automatically handled by NB's inherent variance structure.

### Falsification Framework

I will **abandon all proposed models** if:
- Posterior predictive checks show systematic failure in variance structure
- The data requires zero-inflation (no zeros observed, but posterior predicts many)
- Residual autocorrelation remains > 0.7 after accounting for trend
- Parameter estimates hit prior boundaries (indicates model misspecification)
- Posterior distributions are multimodal (suggests model unidentifiable)

I will **switch model classes** if:
- NB models show θ → 0 (switch to Poisson with random effects)
- NB models show θ → ∞ (switch to Poisson)
- Time-varying dispersion is not supported by data (simplify to constant θ)
- Compound Poisson shows cluster size = 1 (revert to Poisson)

---

## Model Class 1: Heterogeneous Negative Binomial (Primary Recommendation)

### Priority: 1 (Fit this first)

### Rationale
The mean-variance plot shows Var ∝ Mean² to Mean³, which is the signature of Negative Binomial. The key question is whether the dispersion parameter θ varies over time to capture the 26× variance increase.

### Full Probabilistic Specification

**Likelihood:**
```
For t = 1, ..., 40:
  C_t ~ NegativeBinomial(μ_t, θ_t)

where:
  E[C_t] = μ_t
  Var[C_t] = μ_t + μ_t² / θ_t
```

**Variant 1a: Time-varying dispersion (RECOMMENDED)**
```
# Mean structure (exponential growth)
log(μ_t) = α + β₁ × year_t + β₂ × year_t²

# Dispersion structure (decreasing over time)
log(θ_t) = γ₀ + γ₁ × year_t

# This allows variance to grow faster than quadratically with mean
```

**Priors:**
```
# Mean parameters
α ~ Normal(log(109), 1)           # Center at observed mean
β₁ ~ Normal(1, 0.5)                # Expect strong positive growth
β₂ ~ Normal(0, 0.25)               # Weakly informative for curvature

# Dispersion parameters
γ₀ ~ Normal(log(10), 1)            # θ around 10 (moderate overdispersion)
γ₁ ~ Normal(0, 0.5)                # Allow decreasing dispersion over time
```

**Variant 1b: Constant dispersion (simpler alternative)**
```
log(μ_t) = α + β₁ × year_t + β₂ × year_t²
θ ~ LogNormal(log(10), 1)          # Single dispersion parameter
```

**Variant 1c: Changepoint in dispersion**
```
log(μ_t) = α + β₁ × year_t + β₂ × year_t²

θ_t = θ₁  if year_t < τ
      θ₂  if year_t ≥ τ

τ ~ Uniform(-0.5, 0.5)             # Changepoint location
θ₁, θ₂ ~ LogNormal(log(10), 1)
```

### What This Model Tests

**Scientific hypothesis:**
The count generation process has intrinsic overdispersion that increases with the mean (or decreases in precision over time).

**Statistical hypothesis:**
H₀: θ is constant over time (Variant 1b)
H₁: θ varies with time (Variant 1a) or regime (Variant 1c)

**Testable predictions:**
- Posterior for γ₁ should be negative if dispersion decreases over time
- Variance should grow as μ + μ²/θ_t, faster than quadratic
- Early observations should have tighter prediction intervals than late observations

### Expected Computational Challenges

1. **NegativeBinomial parameterization issues:**
   Stan uses NegativeBinomial(α, β) where α = θ and β = θ/μ. Must be careful with parameterization.
   - Solution: Use `neg_binomial_2(μ, φ)` parameterization in Stan where φ = θ

2. **Numerical instability when θ is small:**
   If θ < 0.01, NB approximates Poisson but computation becomes unstable.
   - Solution: Put lower bound θ > 0.1 in model

3. **Prior-posterior conflict:**
   If θ → 0, prior on γ₀ ~ Normal(log(10), 1) may be too restrictive.
   - Solution: Use wider prior γ₀ ~ Normal(2, 2) or check prior predictive

4. **Identification issues with time-varying θ:**
   Both μ and θ increase/decrease over time, may be confounded.
   - Solution: Use strong priors on γ₁, or fix polynomial degree for μ

### Falsification Criteria

**I will abandon this model if:**

1. **θ estimates approach 0 (θ < 0.5):**
   This suggests the data is even more overdispersed than NB can handle. Would switch to:
   - Zero-inflated NB (but we have no zeros!)
   - Compound Poisson
   - Continuous mixture (but loses count structure)

2. **θ estimates are huge (θ > 100):**
   This suggests Poisson is adequate and NB is unnecessarily complex.
   Would switch to: Poisson with log-linear trend

3. **Posterior predictive variance systematically underestimates observed variance:**
   Check: For each quintile of fitted means, does posterior predictive SD match empirical SD?
   If not: Model's variance structure is wrong, need different distribution family

4. **Time-varying dispersion shows γ₁ ≈ 0 (95% CI includes 0 with large margin):**
   No evidence for time-varying dispersion. Revert to constant θ model.

5. **Mean-variance relationship in posterior predictions deviates from data:**
   Plot log(Var) vs log(Mean) for data and posterior predictive samples.
   If slopes differ by > 0.5: Wrong variance function

6. **Extreme parameter correlations (|ρ| > 0.95):**
   Suggests model is unidentifiable. Need to reparameterize or simplify.

### Model Comparison Strategy

Compare variants via:
- **LOO-CV (Leave-One-Out Cross-Validation):** Primary metric
- **WAIC:** Secondary metric
- **Posterior predictive p-values:** For variance, skewness, autocorrelation
- **Out-of-sample RMSE:** Hold out last 5 observations

**Decision rule:**
If ΔLOO < 4 between variants, choose simpler model (constant θ).
If ΔLOO > 10, complex model is strongly preferred.

### Implementation Notes

**Stan code structure:**
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  vector[N] year;
}

parameters {
  real alpha;
  real beta1;
  real beta2;
  real gamma0;
  real gamma1;
}

transformed parameters {
  vector[N] mu;
  vector[N] theta;

  for (t in 1:N) {
    mu[t] = exp(alpha + beta1 * year[t] + beta2 * year[t]^2);
    theta[t] = exp(gamma0 + gamma1 * year[t]);
  }
}

model {
  // Priors
  alpha ~ normal(log(109), 1);
  beta1 ~ normal(1, 0.5);
  beta2 ~ normal(0, 0.25);
  gamma0 ~ normal(log(10), 1);
  gamma1 ~ normal(0, 0.5);

  // Likelihood
  for (t in 1:N) {
    y[t] ~ neg_binomial_2(mu[t], theta[t]);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(y[t] | mu[t], theta[t]);
    y_rep[t] = neg_binomial_2_rng(mu[t], theta[t]);
  }
}
```

**PyMC alternative:**
```python
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=np.log(109), sigma=1)
    beta1 = pm.Normal('beta1', mu=1, sigma=0.5)
    beta2 = pm.Normal('beta2', mu=0, sigma=0.25)
    gamma0 = pm.Normal('gamma0', mu=np.log(10), sigma=1)
    gamma1 = pm.Normal('gamma1', mu=0, sigma=0.5)

    mu = pm.math.exp(alpha + beta1 * year + beta2 * year**2)
    theta = pm.math.exp(gamma0 + gamma1 * year)

    y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=theta, observed=C)

    # For LOO
    pm.Deterministic('log_likelihood',
                     pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=theta), C))
```

---

## Model Class 2: Gamma-Poisson Random Effects (Alternative)

### Priority: 2 (Fit if Model 1 shows issues)

### Rationale
Perhaps the overdispersion comes from latent heterogeneity: each time point has an unobserved "rate multiplier" that varies smoothly over time. This is a different mechanism than intrinsic count overdispersion.

### Full Probabilistic Specification

**Variant 2a: Random walk on log-rates (RECOMMENDED)**
```
# Observation model
C_t ~ Poisson(λ_t)

# Latent rate with smooth variation
log(λ_t) = log(μ_t) + η_t

# Deterministic trend
log(μ_t) = α + β₁ × year_t + β₂ × year_t²

# Latent random walk (captures extra variation)
η_1 ~ Normal(0, σ_η)
η_t ~ Normal(η_{t-1}, σ_η)  for t = 2, ..., 40

# This is a state-space model with Poisson observations
```

**Priors:**
```
α ~ Normal(log(109), 1)
β₁ ~ Normal(1, 0.5)
β₂ ~ Normal(0, 0.25)
σ_η ~ Exponential(1)              # Innovation variance
```

**Variant 2b: i.i.d. random effects**
```
C_t ~ Poisson(λ_t)
log(λ_t) = α + β₁ × year_t + β₂ × year_t² + ε_t
ε_t ~ Normal(0, σ_ε)              # i.i.d. overdispersion

# This is equivalent to log-Normal-Poisson mixture
# Marginal is NOT Negative Binomial
```

**Variant 2c: Time-varying innovation variance**
```
C_t ~ Poisson(λ_t)
log(λ_t) = log(μ_t) + η_t
log(μ_t) = α + β₁ × year_t + β₂ × year_t²

η_1 ~ Normal(0, σ₁)
η_t ~ Normal(η_{t-1}, σ_t)

log(σ_t) = γ₀ + γ₁ × year_t      # Increasing variance over time
```

### What This Model Tests

**Scientific hypothesis:**
Overdispersion is not intrinsic to the count process but arises from unobserved heterogeneity (e.g., unmeasured covariates, environmental fluctuations).

**Statistical hypothesis:**
H₀: Poisson with fixed mean is adequate
H₁: Need latent random effects to explain overdispersion

**Testable predictions:**
- σ_η should be large (≈ 0.3-0.5 on log scale) to explain overdispersion
- η_t should show smooth evolution, not white noise
- ACF of η_t should decay slowly (if random walk is appropriate)

### Expected Computational Challenges

1. **State-space model dimensionality:**
   40 latent parameters (η_1, ..., η_40) plus trend parameters.
   - Solution: Use forward filtering backward sampling (FFBS) or HMC with careful initialization

2. **Funnel geometry:**
   When σ_η is small, η_t are tightly constrained → posterior has funnel shape.
   - Solution: Non-centered parameterization: η_t = σ_η × z_t, z_t ~ Normal(0, 1)

3. **Autocorrelation in η_t:**
   Random walk produces highly correlated parameters, slow mixing.
   - Solution: Use longer chains, or consider AR(1) instead of RW

4. **Identification with trend:**
   Both polynomial trend and random walk can capture curvature.
   - Solution: Use informative priors on β₂, σ_η to separate signal from noise

### Falsification Criteria

**I will abandon this model if:**

1. **σ_η posterior concentrates near 0 (σ_η < 0.01):**
   No evidence for latent variation. Revert to simple Poisson or Negative Binomial.

2. **η_t shows no temporal structure:**
   If ACF(η_t) ≈ 0 for all lags, then i.i.d. effects are sufficient (Variant 2b).
   But this doesn't match observed ACF(C_t) = 0.989, suggesting model is wrong.

3. **Posterior predictive variance is too small:**
   Even with random effects, model underpredicts variance.
   This indicates Poisson + log-Normal mixture is insufficient.
   Would switch to: Compound Poisson or different observation model

4. **Posterior for η_t is not smooth:**
   If η_t jumps erratically, random walk prior is inappropriate.
   Would switch to: AR(1) or independent effects

5. **LOO diagnostic shows many high Pareto k values (k > 0.7):**
   Model is sensitive to individual observations, poor generalization.
   Indicates model misspecification.

6. **Computational failure:**
   If HMC shows divergences > 10% or R-hat > 1.1, model is too complex or misspecified.

### Model Comparison Strategy

Compare to Model 1 (NB) via:
- **LOO-CV:** If ΔLOO > 10, strongly prefer one model
- **Posterior predictive variance:** Which model matches empirical variance structure?
- **Parsimony:** Random effects model has more parameters (40 η_t vs 1-2 θ parameters)

**Key diagnostic:**
Plot posterior median of λ_t vs fitted μ_t from NB model. If they're nearly identical, models are equivalent despite different parameterizations.

### Implementation Notes

**Stan code (non-centered parameterization):**
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  vector[N] year;
}

parameters {
  real alpha;
  real beta1;
  real beta2;
  real<lower=0> sigma_eta;
  vector[N] z;  // Non-centered parameterization
}

transformed parameters {
  vector[N] eta;
  vector[N] lambda;
  vector[N] mu;

  // Non-centered: eta = cumsum(sigma_eta * z)
  eta[1] = sigma_eta * z[1];
  for (t in 2:N) {
    eta[t] = eta[t-1] + sigma_eta * z[t];
  }

  for (t in 1:N) {
    mu[t] = exp(alpha + beta1 * year[t] + beta2 * year[t]^2);
    lambda[t] = mu[t] * exp(eta[t]);
  }
}

model {
  alpha ~ normal(log(109), 1);
  beta1 ~ normal(1, 0.5);
  beta2 ~ normal(0, 0.25);
  sigma_eta ~ exponential(1);
  z ~ std_normal();

  y ~ poisson(lambda);
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (t in 1:N) {
    log_lik[t] = poisson_lpmf(y[t] | lambda[t]);
    y_rep[t] = poisson_rng(lambda[t]);
  }
}
```

**Warning:** This model has N + 5 parameters. With n = 40, this is complex but feasible.

---

## Model Class 3: Compound Poisson (High-risk alternative)

### Priority: 3 (Fit only if Models 1-2 fail)

### Rationale
Extreme overdispersion might arise from a compound process: each time point experiences λ_t "events", each event generates ~Poisson(ν) counts. This produces overdispersion with a different structure than NB.

### Full Probabilistic Specification

**Variant 3a: Poisson-Poisson compound**
```
# Number of clusters
N_t ~ Poisson(λ_t)

# Counts per cluster
For each cluster j = 1, ..., N_t:
  S_j ~ Poisson(ν)

# Total count
C_t = Σ_{j=1}^{N_t} S_j

# This gives: C_t ~ Poisson(λ_t × ν)
# But variance is: Var(C_t) = λ_t × ν × (1 + ν)

# Trend in cluster rate
log(λ_t) = α + β₁ × year_t + β₂ × year_t²
```

**Priors:**
```
α ~ Normal(log(50), 1)            # Fewer clusters than total counts
β₁ ~ Normal(1, 0.5)
β₂ ~ Normal(0, 0.25)
ν ~ Exponential(1)                # Mean cluster size
```

**Variant 3b: Negative Binomial-Poisson compound**
```
N_t ~ NegativeBinomial(λ_t, θ_N)  # Overdispersed number of clusters
S_j ~ Poisson(ν)                  # Cluster sizes

# This produces even more overdispersion
```

### What This Model Tests

**Scientific hypothesis:**
Counts arise from clustered events. E.g., in epidemiology, this could be outbreaks; in ecology, animal groups; in finance, transaction bursts.

**Statistical hypothesis:**
H₀: Counts are simple Poisson or NB
H₁: Counts have compound structure

**Testable predictions:**
- ν (mean cluster size) should be > 1 and < 20
- If ν ≈ 1, compound process reduces to simple Poisson
- Variance should be Var = E × (1 + ν), different from NB

### Expected Computational Challenges

1. **Latent discrete variables:**
   N_t is unobserved and discrete, difficult for HMC.
   - Solution: Marginalize out N_t analytically (compound Poisson has closed form)

2. **Identifiability:**
   Can't separately identify λ_t and ν from counts alone (only their product).
   - Solution: Put strong prior on ν, or fix it

3. **Computational cost:**
   If N_t is large, summing S_j is expensive.
   - Solution: Use compound Poisson PMF directly (available in some libraries)

### Falsification Criteria

**I will abandon this model if:**

1. **ν posterior concentrates at 1:**
   Cluster size = 1 means no clustering, revert to simple Poisson.

2. **Variance structure doesn't match compound Poisson:**
   If Var(C_t) ≠ E[C_t] × (1 + ν), model is wrong.
   Compare to NB: Var = μ + μ²/θ

3. **Worse predictive performance than NB:**
   If LOO is 20+ points worse than NB, compound process adds nothing.

4. **Computational failure:**
   If model doesn't converge after 10K iterations, too complex for data.

**Honestly:** I expect this model to fail. It's included as a stress test to see if NB's variance structure is truly adequate. If compound Poisson fits better, it would be a surprising finding suggesting event clustering.

### Implementation Notes

**Stan code (marginalized):**
```stan
// Use Poisson-Compound distribution (not standard in Stan)
// Would need custom PMF or approximation

// Alternative: Simulate N_t and sum, but slow
// Better: Use Negative Binomial as approximation to compound Poisson
```

**Practical issue:** Compound Poisson is not standard in Stan/PyMC. Would need to:
- Implement custom PMF
- Or use Negative Binomial as approximation (renders this model redundant)
- Or simulate in generated quantities only (can't fit)

**Conclusion:** This model is more theoretical than practical. I include it to demonstrate adversarial thinking, but implementation challenges make it low priority.

---

## Model Selection & Comparison Strategy

### Phase 1: Initial Fitting (Week 1)

1. **Fit Model 1b (NB with constant θ):** Simplest, establish baseline
2. **Fit Model 1a (NB with time-varying θ):** Test key hypothesis
3. **Compare via LOO-CV:** Does time-varying dispersion improve fit?

**Decision point:** If ΔLOO < 4, use constant θ. If ΔLOO > 10, use time-varying.

### Phase 2: Random Effects (Week 2)

4. **Fit Model 2a (Random walk effects):** Only if Model 1 shows systematic variance misfit
5. **Compare to Model 1 winner:** LOO, posterior predictive checks

**Decision point:** If random effects add complexity without improving variance prediction, abandon Model 2.

### Phase 3: Diagnostics & Validation (Week 3)

6. **Posterior predictive checks:**
   - Variance by time period (early vs late)
   - Mean-variance relationship
   - Autocorrelation structure (ACF of residuals)
   - Extreme value coverage (do 95% intervals cover 95%?)

7. **Cross-validation:**
   - Leave-last-5-out prediction
   - Leave-one-out (via LOO-CV)
   - Check calibration: Do 50% intervals cover 50%?

8. **Sensitivity analysis:**
   - Double prior SDs, does posterior change much?
   - Exclude first 5 observations (suspected different regime)
   - Exclude last 5 observations (highest variance)

**Final decision point:** Select model based on:
- Best LOO (primary)
- Posterior predictive variance matches data (critical)
- Parsimony (tie-breaker)

### Red Flags That Require Model Pivot

**STOP and reconsider everything if:**

1. **All models show poor posterior predictive variance:**
   Variance is systematically underestimated or overestimated.
   → Need different distribution family (e.g., generalized Poisson, Conway-Maxwell)

2. **Residual autocorrelation > 0.7 remains:**
   None of these models handle the ACF = 0.989 adequately.
   → Need explicit AR structure in mean or observation equation

3. **Parameter estimates hit prior boundaries:**
   E.g., θ → ∞ or θ → 0, σ_η → 0 or σ_η → 1
   → Model is misspecified, prior is fighting data

4. **Multimodal posteriors:**
   Multiple posterior modes indicate model is unidentifiable.
   → Simplify or add informative priors

5. **Divergent transitions > 20%:**
   HMC is struggling, model geometry is pathological.
   → Reparameterize or simplify

6. **LOO Pareto k > 0.7 for many observations:**
   Model is highly sensitive to individual points, poor generalization.
   → Add robust observation model (e.g., Student-t mixture)

### Alternative Model Classes (Escape Routes)

If all proposed models fail:

**Escape Route A: Explicit temporal dependence**
```
C_t ~ NegativeBinomial(μ_t, θ)
log(μ_t) = α + β × log(C_{t-1}) + γ × year_t
```
This directly models ACF = 0.989 via lagged dependent variable.

**Escape Route B: Changepoint model**
```
C_t ~ NegativeBinomial(μ_t, θ)
log(μ_t) = α₁ + β₁ × year_t  if year_t < τ
           α₂ + β₂ × year_t  if year_t ≥ τ
```
Two distinct regimes with different growth rates.

**Escape Route C: Robust observation model**
```
C_t ~ NegativeBinomial(μ_t, θ) with probability 1 - π
      NegativeBinomial(μ_t, θ/10) with probability π
```
Mixture model allowing occasional extreme overdispersion.

**Escape Route D: Admit defeat**
If none of the above work, the data may be too complex for n = 40. Report that multiple model classes fail to capture variance structure, recommend:
- Collecting more data
- Investigating data generation process (qualitative research)
- Using non-parametric methods (but losing probabilistic inference)

---

## Prior Elicitation & Justification

### Design Philosophy
Use **weakly informative priors** that:
- Rule out scientifically implausible values (e.g., negative means)
- Allow data to dominate (prior SD ≥ posterior SD)
- Stabilize computation (avoid flat priors)
- Encode genuine prior knowledge (e.g., growth is positive)

### Key Priors Explained

**α ~ Normal(log(109), 1):**
Center at observed mean on log scale. SD = 1 allows 95% prior mass on [40, 300], covering full data range. This is weakly informative: data will easily shift posterior if needed.

**β₁ ~ Normal(1, 0.5):**
Expect positive growth (β₁ > 0). Prior mean = 1 based on rough estimate: log(245/29) / (1.67 - (-1.67)) ≈ 0.65. Prior allows for stronger or weaker growth.

**β₂ ~ Normal(0, 0.25):**
Quadratic term could be positive (accelerating) or negative (decelerating). Prior is agnostic but weakly regularizes toward linear.

**θ ~ LogNormal(log(10), 1):**
Based on NB overdispersion formula: Var ≈ μ²/θ when overdispersion dominates.
Observed: Var/μ² ≈ 68, so θ ≈ μ²/Var ≈ 109²/7441 ≈ 1.6
But this is aggregated; within-trend θ may be larger (≈ 5-20).
Prior allows range [1, 100], data will determine exact value.

**σ_η ~ Exponential(1):**
For random walk models. Prior mean = 1 is large on log scale (means rates can double/halve between steps). This is conservative: allows large variation but data can shrink it.

### Prior Predictive Checks

Before fitting, simulate from prior:
```
1. Draw parameters from priors
2. Simulate C_1, ..., C_40
3. Check: Do simulated data span plausible range?
```

**Criterion:** Prior predictive should include:
- Counts from 1 to 1000 (wider than observed [19, 272])
- Variance/Mean ratios from 1 to 500 (covering observed 68)
- Both increasing and decreasing trends

If prior predictive is too narrow: Increase prior SDs
If prior predictive includes impossible values (e.g., C > 10^6): Tighten priors

---

## Stress Tests & Adversarial Checks

### Stress Test 1: Posterior Predictive Variance by Quintile

**Purpose:** Check if model captures heteroscedasticity

**Method:**
1. Divide data into 5 quintiles by fitted mean
2. For each quintile, compute:
   - Empirical variance
   - Posterior predictive variance (median, 95% CI)
3. Plot empirical vs predicted variance

**Failure criterion:** If empirical variance is outside 95% CI for 3+ quintiles, model's variance structure is wrong.

### Stress Test 2: Extreme Value Coverage

**Purpose:** Check if model captures tail behavior

**Method:**
1. Identify extreme observations (top/bottom 10%)
2. Compute posterior predictive 90% intervals
3. Check coverage: Do 90% intervals cover extreme values?

**Failure criterion:** If coverage < 50% for extreme values, model underestimates tail risk.

### Stress Test 3: Sequential Prediction

**Purpose:** Test if model generalizes to new data

**Method:**
1. Fit model on data[1:30]
2. Predict data[31:40]
3. Compute RMSE, coverage of 95% intervals

**Failure criterion:** If RMSE > 100 or coverage < 80%, model overfits or misspecifies trend.

### Stress Test 4: Parameter Sensitivity

**Purpose:** Check if conclusions are robust to prior choices

**Method:**
1. Fit model with prior SDs doubled
2. Fit model with prior SDs halved
3. Compare posterior medians

**Failure criterion:** If posterior medians change by > 50%, data is weak and priors are driving inference.

### Stress Test 5: Leave-Group-Out

**Purpose:** Test if early/late data are compatible

**Method:**
1. Fit model on data[year < 0]
2. Predict data[year ≥ 0]
3. Check: Are predictions wildly off?

**Failure criterion:** If RMSE for predicted half is 3× larger than fitted half, there's a regime change not captured by model.

---

## Expected Results & Interpretation

### Scenario A: Model 1a wins (time-varying dispersion)

**What we learn:**
Overdispersion is not constant but decreases over time. The process becomes more variable relative to its mean as counts grow. This suggests:
- Intrinsic stochasticity in count generation
- Measurement error may increase with scale
- Biological/physical process has time-varying noise

**Parameter expectations:**
- γ₁ < 0 (dispersion decreases over time)
- θ_early ≈ 10-20 (moderate overdispersion early)
- θ_late ≈ 2-5 (extreme overdispersion late)

**Next steps:**
Investigate scientific reason for changing dispersion. Is it:
- Measurement process?
- Environmental noise?
- Fundamental to phenomenon?

### Scenario B: Model 1b wins (constant dispersion)

**What we learn:**
NB's inherent variance structure (Var = μ + μ²/θ) is sufficient. The heteroscedasticity is automatically handled by mean growth. This suggests:
- Overdispersion is stable property of process
- No regime change in variance structure
- Simpler model is adequate

**Parameter expectations:**
- θ ≈ 5-15 (moderate to high overdispersion)
- β₂ likely non-zero (quadratic trend)

**Next steps:**
Focus on trend specification and temporal dependence (autocorrelation still unaddressed).

### Scenario C: Model 2a wins (random walk effects)

**What we learn:**
Overdispersion arises from latent heterogeneity, not intrinsic to count process. Each observation has unobserved rate multiplier that evolves smoothly. This suggests:
- Unmeasured covariates are important
- Process has memory (random walk structure)
- Measurement may be noisy

**Parameter expectations:**
- σ_η ≈ 0.2-0.5 (substantial variation)
- η_t should show smooth evolution
- ACF(η_t) should decay slowly

**Next steps:**
Try to identify what η_t represents. Correlate with auxiliary variables if available.

### Scenario D: All models fail

**What we learn:**
Data is more complex than n = 40 can resolve. Possible issues:
- Multiple unmodeled changepoints
- Non-stationary variance structure
- Extreme autocorrelation dominates everything
- Wrong distributional family entirely

**Next steps:**
- Simplify to descriptive statistics
- Collect more data
- Use non-parametric methods
- Consult domain experts on data generation

---

## Computational Plan

### Software & Tools
- **Primary:** Stan via CmdStanPy (best HMC implementation)
- **Alternative:** PyMC (more user-friendly, similar performance)
- **Validation:** Compare Stan and PyMC results, should agree

### Sampling Strategy
- **Chains:** 4 independent chains (assess convergence)
- **Warmup:** 1000 iterations (tune step size and mass matrix)
- **Sampling:** 2000 iterations per chain (total 8000 posterior samples)
- **Thinning:** No thinning (keep all samples, modern HMC is efficient)

### Convergence Diagnostics
- **R-hat < 1.01:** All parameters (indicates convergence)
- **ESS > 400:** Effective sample size (indicates good mixing)
- **Divergences = 0:** No divergent transitions (indicates correct geometry)
- **Tree depth < max:** No hitting max tree depth (indicates efficient sampling)

If convergence fails:
1. Increase warmup to 2000
2. Increase adapt_delta from 0.8 to 0.95 (smaller steps)
3. Reparameterize model (non-centered)
4. Simplify model (fewer parameters)

### Computational Resources
- **Time per model:** ~5-30 minutes (depends on complexity)
- **Memory:** < 4GB (n = 40 is small)
- **Parallelization:** Run chains in parallel (4 cores)

### Output & Diagnostics
For each model, save:
- Posterior samples (all parameters)
- Log-likelihood (for LOO-CV)
- Posterior predictive samples (y_rep)
- Convergence diagnostics (R-hat, ESS, divergences)
- Model comparison metrics (LOO, WAIC)

---

## Summary: Ranking & Recommendation

### Priority Ranking

**Rank 1: Model 1a (NB with time-varying dispersion)**
- Most directly addresses extreme overdispersion and heteroscedasticity
- Computationally feasible
- Clear falsification criteria
- Builds naturally on EDA findings
- **Start here**

**Rank 2: Model 1b (NB with constant dispersion)**
- Simpler alternative to 1a
- Baseline for comparison
- If 1a fails, this is fallback
- **Fit in parallel with 1a**

**Rank 3: Model 2a (Random walk effects)**
- Alternative mechanism for overdispersion
- More parameters, higher complexity
- Only fit if Model 1 shows systematic failures
- **Fit only if needed**

**Rank 4: Model 1c (Changepoint in dispersion)**
- Tests specific hypothesis from EDA
- More structured than time-varying
- Lower priority than continuous time variation
- **Fit if time/resources allow**

**Rank 5: Model 3 (Compound Poisson)**
- Theoretical interest only
- Implementation challenges
- Low prior probability of success
- **Skip unless other models fail completely**

### Critical Success Factors

**This modeling effort succeeds if:**
1. At least one model captures observed variance structure (passes Stress Test 1)
2. Posterior predictive checks show good calibration (coverage ≈ nominal)
3. LOO-CV gives reliable model comparison (Pareto k < 0.7)
4. Parameter estimates are scientifically interpretable
5. We learn something about the data generation process

**This modeling effort FAILS if:**
1. All models systematically underestimate variance
2. Posterior predictive intervals are uncalibrated (coverage << nominal)
3. Computational problems prevent fitting
4. Results are highly sensitive to prior choices
5. We can't distinguish between competing hypotheses

### Decision Timeline

**Week 1:**
- Fit Models 1a, 1b
- Compute LOO, posterior predictive checks
- **Decision:** Continue with winner or pivot to Model 2?

**Week 2:**
- Fit Model 2a (if needed)
- Run all stress tests
- **Decision:** Select final model or try escape routes?

**Week 3:**
- Sensitivity analysis
- Cross-validation
- **Final decision:** Report best model and limitations

---

## Conclusion: Embracing Uncertainty

I've proposed three model classes that make different assumptions about the source of extreme overdispersion. I **expect at least one to fail**, possibly all. The goal is not to complete a checklist but to discover which assumptions are wrong.

**Key uncertainties:**
1. Is dispersion constant or time-varying? (Models 1a vs 1b test this)
2. Is overdispersion intrinsic or from latent heterogeneity? (Model 1 vs Model 2)
3. Can any model handle the 26× variance increase? (All models stressed)

**Falsification is success:**
If I discover that Negative Binomial with time-varying dispersion fails Stress Test 1, that's valuable information. It tells us the variance structure is even more complex than NB can handle.

**Alternative outcomes:**
- Best case: Model 1a fits well, passes all stress tests → Done
- Good case: Model 1b sufficient, time-varying dispersion not needed → Simplify
- Acceptable case: Model 2a needed, latent heterogeneity is key → More complex
- Challenging case: All fail Stress Test 1 → Use Escape Routes A-D
- Worst case: Everything fails → Report limitations, recommend more data

**I will not force a model to work.** If the data resists all proposed models, I'll report that honestly and suggest next steps. The goal is truth, not task completion.

---

## File Structure

All model code and results will be saved to:
- `/workspace/experiments/designer_1/models/`: Stan/PyMC code
- `/workspace/experiments/designer_1/results/`: Posterior samples, diagnostics
- `/workspace/experiments/designer_1/figures/`: Posterior predictive checks, comparisons
- `/workspace/experiments/designer_1/logs/`: Convergence diagnostics, warnings

---

**Document Status:** COMPLETE
**Next Steps:** Await approval, then implement Model 1a in Stan
**Questions/Concerns:** Should I coordinate with other designers on overlapping model components (e.g., temporal autocorrelation)?
