# Bayesian Model Designs: Temporal Autocorrelation Focus
**Designer 3: Temporal Structure Specialist**

**Date:** 2025-10-29
**Focus:** Explicit modeling of temporal autocorrelation in count time series
**Data:** 40 observations, counts 19-272, lag-1 ACF = 0.989

---

## Executive Summary

This document proposes three Bayesian model classes that explicitly address the **extremely high temporal autocorrelation (0.989)** observed in the count data. The central question is: **Is the autocorrelation inherent to the data-generating process (requiring state-space or latent AR structure), or is it an artifact of smooth trending that disappears after proper detrending?**

### Critical Insight
The ACF plot shows ALL lags are highly significant (slowly decaying from 0.989 to 0.83 over 15 lags). This is the hallmark of either:
1. **Spurious autocorrelation** from deterministic trending (MOST LIKELY given the quadratic trend)
2. **True state dependence** requiring explicit temporal modeling (LESS LIKELY but must test)

### Proposed Models (Ranked by Complexity)
1. **Model T1: Negative Binomial with Latent AR(1) Process** (State-space approach)
2. **Model T2: Poisson Dynamic Linear Model** (Time-varying level & trend)
3. **Model T3: Count AR(1) with Overdispersed Innovations** (Direct temporal dependence)

---

## Critical Questions & Falsification Strategy

### The Central Hypothesis to Test
**H0:** The observed autocorrelation is spurious, induced by smooth trending. A well-specified trend model will produce independent residuals.

**H1:** The process has inherent temporal dependence beyond trending (e.g., momentum, feedback loops, contagion effects).

### Falsification Plan
1. **Fit Model T1-T3** with explicit temporal structure
2. **Compare to Designer 1's quadratic NB model** (no explicit temporal structure)
3. **If Designer 1's model has ACF(residuals) < 0.3:** H0 confirmed, temporal models are overkill
4. **If ACF(residuals) > 0.5 after detrending:** H1 confirmed, need explicit temporal models

### Decision Rules
- **I will abandon temporal modeling if:** Residuals from a good trend model are uncorrelated
- **I will escalate complexity if:** Even state-space models show residual autocorrelation
- **I will switch to changepoint models if:** Temporal structure breaks down mid-series

---

## Model T1: Negative Binomial with Latent AR(1) Process

### Rationale
The most flexible approach: combines count distribution with latent Gaussian state-space for temporal smoothness. Separates trend, temporal correlation, and overdispersion into distinct components.

### Model Specification

**Observation Model:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = α_t
```

**Latent State Process (AR(1) on log-scale):**
```
α_t = β₀ + β₁·year_t + β₂·year_t² + ε_t
ε_t = ρ·ε_{t-1} + η_t
η_t ~ Normal(0, σ_η)
ε_1 ~ Normal(0, σ_η/sqrt(1-ρ²))  # Stationary initial condition
```

**Priors:**
```
# Trend parameters
β₀ ~ Normal(4.5, 1)              # log(mean count) ≈ log(109) ≈ 4.7
β₁ ~ Normal(0, 1)                # Positive trend expected
β₂ ~ Normal(0, 0.5)              # Acceleration term

# Temporal correlation
ρ ~ Beta(12, 3)                  # Favors high correlation (mean=0.8) based on ACF
σ_η ~ HalfNormal(0, 0.5)        # Innovation SD (on log scale)

# Overdispersion
φ ~ HalfNormal(0, 10)           # Dispersion parameter
```

### Prior Justification

**β₀:** Centered on observed log-mean with weak informativeness
**β₁, β₂:** Weakly informative, allows data to determine trend shape
**ρ ~ Beta(12,3):** Strongly informative prior favoring high autocorrelation (mean=0.8, SD=0.1)
- Rationale: ACF=0.989 suggests ρ should be high IF autocorrelation is real
- This prior will be "fought" by the data if autocorrelation is spurious
- **Key diagnostic:** If posterior ρ << prior mean, autocorrelation was spurious

**σ_η:** Small innovations maintain temporal smoothness
**φ:** Vague prior on overdispersion

### Strengths
- Separates deterministic trend from stochastic temporal dependence
- Handles overdispersion via NegBin
- Maintains count data structure
- Flexible: can detect if ρ ≈ 0 (no temporal dependence needed)

### Weaknesses
- **High complexity** (7 parameters for n=40)
- **Identifiability issues:** Trend vs temporal correlation confounded
- **Computational cost:** Non-centered parameterization needed in Stan
- **May overfit** given small sample size

### Expected Failure Modes
1. **Prior-posterior conflict on ρ:** Posterior shifts to ρ ≈ 0, suggesting trend explains all autocorrelation
2. **Divergent transitions in Stan:** Indicates non-identifiability between β₂ and ρ
3. **Wide posterior credible intervals:** Insufficient data to estimate all parameters
4. **Worse LOO-IC than simpler models:** Complexity penalty not justified

### Falsification Criteria
**I will abandon this model if:**
- Posterior ρ has 95% CI including 0 (temporal correlation not needed)
- LOO-IC is 10+ points worse than Designer 1's quadratic NB model
- Divergent transitions exceed 5% despite careful tuning
- Posterior predictive checks fail (predictions outside 95% CI for >20% of data)

### Stan Implementation Hints
```stan
// Use non-centered parameterization for AR(1) errors
parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0,upper=1> rho;
  real<lower=0> sigma_eta;
  real<lower=0> phi;
  vector[N] epsilon_raw;  // Non-centered
}
transformed parameters {
  vector[N] epsilon;
  epsilon[1] = epsilon_raw[1] * sigma_eta / sqrt(1 - rho^2);
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + epsilon_raw[t] * sigma_eta;
  }
}
model {
  vector[N] log_mu;
  for (t in 1:N) {
    log_mu[t] = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2 + epsilon[t];
  }

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);

  // Priors
  beta_0 ~ normal(4.5, 1);
  beta_1 ~ normal(0, 1);
  beta_2 ~ normal(0, 0.5);
  rho ~ beta(12, 3);
  sigma_eta ~ normal(0, 0.5);
  phi ~ normal(0, 10);
  epsilon_raw ~ std_normal();
}
```

---

## Model T2: Poisson Dynamic Linear Model (DLM)

### Rationale
Classic state-space approach where the count rate evolves smoothly over time. The level and possibly trend are time-varying random walks, making this inherently temporal without imposing parametric trend forms.

### Model Specification

**Observation Model:**
```
C_t ~ Poisson(λ_t)
log(λ_t) = μ_t
```

**State Evolution (Local Linear Trend):**
```
μ_t = μ_{t-1} + δ_{t-1} + ω_t     # Level equation
δ_t = δ_{t-1} + ν_t                # Trend equation

ω_t ~ Normal(0, σ_μ)              # Level innovation
ν_t ~ Normal(0, σ_δ)              # Trend innovation
```

**Initial States:**
```
μ_1 ~ Normal(log(30), 1)          # Based on first observation
δ_1 ~ Normal(0, 0.5)              # Initial growth rate
```

**Priors:**
```
σ_μ ~ HalfNormal(0, 0.3)         # Level volatility (small = smooth)
σ_δ ~ HalfNormal(0, 0.1)         # Trend volatility (small = stable trend)
```

### Prior Justification

**σ_μ, σ_δ:** Small values enforce smooth evolution
- Rationale: Visual inspection shows smooth trajectory without sudden jumps
- σ_μ controls how much level can deviate from trend
- σ_δ controls how much growth rate changes over time

**Weakly informative:** Allows data to determine volatility but favors smoothness

### Strengths
- **No parametric trend assumption** (quadratic vs exponential doesn't matter)
- **Inherently temporal** by construction
- **Interpretable:** μ_t is log-rate at time t, δ_t is growth rate
- **Can detect changepoints** if σ_δ > 0 (evolving trend)

### Weaknesses
- **No overdispersion handling:** Poisson assumption may be too restrictive
- **Parameter proliferation:** 2 states × 40 time points = 80 latent variables + 2 hypers
- **Computational burden:** MCMC may struggle with 80+ parameters
- **Identifiability:** σ_μ and σ_δ can trade off

### Expected Failure Modes
1. **Overdispersion not captured:** Posterior predictive checks show actual variance >> predicted
2. **Extremely small σ_δ:** Suggests constant trend, reducing to simpler model
3. **Computational issues:** Poor mixing, long chains needed for convergence
4. **Smoothed states ≈ quadratic fit:** Model reduces to deterministic trend, no added value

### Falsification Criteria
**I will abandon this model if:**
- Posterior predictive variance consistently underestimates observed variance by >2×
- σ_δ posterior has 95% CI: [0, 0.01], suggesting trend is constant (not time-varying)
- LOO-IC worse than Model T1 by >5 points
- Smoothed μ_t trajectory is indistinguishable from fitted quadratic curve

### Extensions to Address Overdispersion
If Poisson fails, extend to:
```
C_t ~ NegativeBinomial(exp(μ_t), φ)
```
This adds one parameter (φ) but maintains DLM structure.

### Stan Implementation Hints
```stan
// Forward filtering for state evolution
parameters {
  real mu_1;
  real delta_1;
  real<lower=0> sigma_mu;
  real<lower=0> sigma_delta;
  vector[N] omega;      // Level innovations
  vector[N] nu;         // Trend innovations
}
transformed parameters {
  vector[N] mu;
  vector[N] delta;

  mu[1] = mu_1;
  delta[1] = delta_1;

  for (t in 2:N) {
    delta[t] = delta[t-1] + nu[t];
    mu[t] = mu[t-1] + delta[t-1] + omega[t];
  }
}
model {
  // Likelihood
  C ~ poisson_log(mu);

  // State innovations
  omega ~ normal(0, sigma_mu);
  nu ~ normal(0, sigma_delta);

  // Priors
  mu_1 ~ normal(log(30), 1);
  delta_1 ~ normal(0, 0.5);
  sigma_mu ~ normal(0, 0.3);
  sigma_delta ~ normal(0, 0.1);
}
```

**Note:** Consider using Kalman filter for faster computation if Gaussian assumptions hold on log-scale.

---

## Model T3: Count AR(1) with Overdispersed Innovations

### Rationale
Direct autoregressive modeling on the count scale (not log). Most interpretable temporal model: today's count depends directly on yesterday's count plus random innovations.

### Model Specification

**Observation Model (Detrended AR(1)):**
```
C_t ~ NegativeBinomial(μ_t, φ)
μ_t = exp(η_t)
η_t = β₀ + β₁·year_t + β₂·year_t² + ρ·(log(C_{t-1}) - [β₀ + β₁·year_{t-1} + β₂·year_{t-1}²])
```

**Interpretation:**
- The log-count follows an AR(1) process around a quadratic trend
- ρ captures correlation of deviations from trend
- If ρ ≈ 0, reduces to Designer 1's quadratic NB model

**Initial Condition:**
```
η_1 ~ Normal(β₀ + β₁·year₁ + β₂·year₁², σ_ε)
σ_ε ~ HalfNormal(0, 0.5)
```

**Priors:**
```
β₀ ~ Normal(4.5, 1)
β₁ ~ Normal(0, 1)
β₂ ~ Normal(0, 0.5)
ρ ~ Uniform(-1, 1)               # Allow negative autocorrelation too
φ ~ HalfNormal(0, 10)
```

### Prior Justification

**ρ ~ Uniform(-1,1):** Intentionally vague to let data speak
- **Critical test:** If ρ posterior is wide or centered near 0, temporal dependence is weak
- Unlike Model T1, we're NOT assuming high autocorrelation
- This is the most conservative approach to testing temporal dependence

### Strengths
- **Simple interpretation:** Direct temporal dependence on detrended counts
- **Moderate complexity:** 5 parameters (vs 7 in T1)
- **Handles overdispersion** via NegBin
- **Conservative test** of temporal dependence with flat prior on ρ

### Weaknesses
- **Assumes AR(1) sufficient:** May need AR(p) if higher-order dependencies exist
- **Mixed continuous-discrete:** Conditioning on log(C_{t-1}) is awkward for counts
- **Less flexible than state-space:** Can't capture time-varying parameters

### Expected Failure Modes
1. **ρ posterior includes 0:** No evidence for temporal dependence beyond trend
2. **Negative autocorrelation:** Would suggest model misspecification (oscillations)
3. **Residual autocorrelation:** AR(1) insufficient, need higher order
4. **Computational issues:** Conditioning on lagged log(C) may cause numerical instability

### Falsification Criteria
**I will abandon this model if:**
- Posterior ρ 95% CI: [-0.3, 0.3] (indistinguishable from 0)
- ACF of posterior predictive residuals still shows significant lag-2+ correlations
- Model fit (LOO-IC) worse than non-temporal models
- Numerical instabilities or divergent transitions >5%

### Stan Implementation Hints
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
}
parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=-1,upper=1> rho;
  real<lower=0> phi;
}
model {
  vector[N] log_mu;
  vector[N] trend;

  // Compute deterministic trend
  for (t in 1:N) {
    trend[t] = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2;
  }

  // First observation (no lag)
  log_mu[1] = trend[1];

  // AR(1) on deviations from trend
  for (t in 2:N) {
    log_mu[t] = trend[t] + rho * (log(C[t-1] + 0.5) - trend[t-1]);  // +0.5 to avoid log(0)
  }

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);

  // Priors
  beta_0 ~ normal(4.5, 1);
  beta_1 ~ normal(0, 1);
  beta_2 ~ normal(0, 0.5);
  rho ~ uniform(-1, 1);
  phi ~ normal(0, 10);
}
```

**Warning:** Using log(C_{t-1}) as a predictor creates a feedback loop. May need to use latent μ_{t-1} instead for proper Bayesian inference.

---

## Model Comparison Strategy

### Primary Metrics
1. **LOO-IC (Leave-One-Out Information Criterion):** Penalizes complexity while rewarding fit
2. **WAIC (Widely Applicable IC):** Alternative Bayesian IC
3. **Posterior Predictive Checks:** Do simulated data match observed patterns?

### Secondary Metrics
1. **Residual ACF:** Do residuals show remaining autocorrelation?
2. **Forecast accuracy:** Hold out last 5 observations, compare RMSE
3. **Variance coverage:** Does 95% posterior interval capture observed variance?

### Comparison to Designer 1's Models
**Critical comparison:** Do temporal models provide meaningful improvement over:
- Quadratic NB (no temporal structure)
- Exponential NB (no temporal structure)

**Threshold for success:** LOO-IC improvement >10 points to justify added complexity

---

## Critical Discussion: Is Temporal Structure Real or Spurious?

### Evidence AGAINST Inherent Temporal Dependence

1. **Perfect smooth trending:** Visual inspection shows no "shocks" or deviations needing temporal memory
2. **ACF pattern:** Slow linear decay typical of trending data, not true AR process
3. **Small sample (n=40):** Limited power to distinguish trend from state dependence
4. **Quadratic fit is excellent:** R²=0.961 suggests trend captures most variation

**Prediction:** Model T1-T3 will find ρ ≈ 0, σ_δ ≈ 0, indicating temporal structure is spurious.

### Evidence FOR Inherent Temporal Dependence

1. **Biological/social processes often have momentum:** If these are population counts, births depend on current population
2. **Accelerating growth suggests feedback:** Exponential growth IS an AR process (C_t ∝ C_{t-1})
3. **Variance heterogeneity:** Q3 shows extreme overdispersion, suggesting regime changes

**Alternative prediction:** Models may find time-varying dynamics (changepoints, evolving trend).

### The Adversarial Test

**Design a stress test to break all three models:**
1. Simulate data with perfect quadratic trend + i.i.d. NegBin errors
2. Fit Models T1-T3
3. Check if they incorrectly infer temporal dependence (false positive test)

**If models find ρ > 0.5 on this synthetic data:** They're overfitting, priors too strong.
**If models find ρ ≈ 0 on synthetic data:** They're appropriately conservative.

---

## Pivot Points & Alternative Models

### When to Abandon Temporal Modeling Entirely
1. **If all three models show ρ ≈ 0 or σ_δ ≈ 0:** Temporal structure not needed
2. **If Designer 1's quadratic NB has LOO-IC within 5 points:** Occam's razor favors simpler model
3. **If residuals from quadratic NB show ACF < 0.3 at all lags:** Trend explains autocorrelation

**Action:** Recommend Designer 1's model as final choice.

### When to Escalate Complexity
1. **If residual ACF remains high (>0.5) even in temporal models:** Need higher-order AR or nonlinear dynamics
2. **If posteriors show bimodality:** Suggests mixture model or regime-switching
3. **If temporal parameters vary over time:** Need time-varying coefficient models

**Alternative models to consider:**
- **AR(2) or AR(p):** Higher-order temporal dependence
- **Hidden Markov Model:** Discrete regime switches
- **Time-varying AR coefficients:** ρ_t evolves over time
- **Gaussian Process:** Non-parametric temporal correlation structure

### When to Switch to Changepoint Models
1. **If σ_δ shows sudden large jumps at specific times:** Suggests structural breaks
2. **If residuals cluster in time:** Early vs late periods behave differently
3. **If growth rate acceleration is TOO extreme:** May be two regimes, not smooth transition

**Alternative:** Bayesian changepoint regression with unknown breakpoint location.

---

## Red Flags & Warning Signs

### Computational Red Flags
- **>5% divergent transitions:** Non-identifiability or poor geometry
- **R-hat > 1.05:** Chains haven't converged
- **Effective sample size < 100:** Poor mixing, need longer chains
- **Bulk/tail ESS differ by >10×:** Tails poorly explored

**Action:** Reparameterize (non-centered), tighten priors, or abandon model.

### Inferential Red Flags
- **Prior-posterior overlap >80%:** Data not informative, model too flexible
- **Posterior ρ 95% CI: [0.01, 0.99]:** Completely uninformative, identifiability issue
- **Posterior predictive p-value < 0.05 or > 0.95:** Model systematically misses data patterns
- **95% PI coverage < 80% or > 98%:** Intervals miscalibrated

**Action:** Simplify model, add constraints, or reconsider model class.

### Scientific Red Flags
- **Extreme parameter values:** φ > 100, ρ > 0.99 suggest boundary issues
- **Inconsistent trend estimates:** β₁, β₂ differ wildly across models (should be stable)
- **Autocorrelation in "wrong" direction:** Negative ρ when positive expected

**Action:** Question scientific assumptions, check for data errors.

---

## Implementation Roadmap

### Phase 1: Baseline Temporal Tests (2-3 hours)
1. Fit Model T3 (simplest temporal model)
2. Check ACF of residuals
3. **Decision point:** If ρ ≈ 0, stop and recommend Designer 1's model

### Phase 2: Full Temporal Models (4-6 hours)
1. Fit Model T1 (latent AR1) if Phase 1 shows ρ > 0.3
2. Fit Model T2 (DLM) if Phase 1 shows ρ > 0.5
3. Compare LOO-IC across all models

### Phase 3: Stress Testing (2-3 hours)
1. Posterior predictive checks
2. Simulate from posteriors, check ACF patterns match data
3. Hold-out forecast evaluation (last 5 points)

### Phase 4: Final Recommendation
- If temporal models win: Report best model with confidence
- If non-temporal wins: Explicitly state "autocorrelation is spurious"
- If unclear: Report model uncertainty, suggest data collection

---

## Expected Outcomes & Hypotheses

### My Primary Hypothesis
**The observed autocorrelation is spurious, caused by smooth quadratic trending.**

**Evidence that would support this:**
- Model T3 finds ρ posterior centered near 0
- Residuals from quadratic NB show no significant autocorrelation
- LOO-IC favors Designer 1's simpler models

**Evidence that would refute this:**
- ρ posterior 95% CI: [0.5, 0.95] (strong temporal dependence)
- Residuals from quadratic NB still show ACF > 0.5
- Temporal models improve LOO-IC by >10 points

### Secondary Hypotheses
1. **If temporal structure is real, it's likely changing over time** (σ_δ > 0 in DLM)
2. **Overdispersion and temporal correlation may be confounded** (both increase with trend)
3. **The "true" model may be even simpler** (exponential with i.i.d. errors)

---

## Comparison with Other Designers

### Designer 1 (Parametric GLMs)
- **Overlap:** All models include trend terms (β₀, β₁, β₂)
- **Difference:** I add explicit temporal correlation (ρ, σ_δ)
- **Test:** Does temporal structure improve on their best model?

### Designer 2 (Non-parametric)
- **Overlap:** Flexibility in trend specification
- **Difference:** I use parametric forms with temporal memory vs their splines/GPs
- **Test:** Can temporal structure match their flexibility with fewer parameters?

### Expected Result
**Most likely:** Designer 1's quadratic NB wins (simplest sufficient model).
**Possible:** My Model T1 wins if true temporal dependence exists.
**Unlikely:** Designer 2's complex non-parametric wins (small n=40).

---

## Final Notes: Philosophy of Temporal Modeling

### When to Model Temporal Structure
- **Domain knowledge** suggests memory/persistence (populations, contagion)
- **Residual diagnostics** show autocorrelation after detrending
- **Forecast task** requires sequential predictions

### When NOT to Model Temporal Structure
- **Autocorrelation vanishes** with proper trending
- **Sample size too small** to estimate both trend and temporal parameters
- **Deterministic trend sufficient** for scientific goals

### The Bayesian Advantage
- Can explicitly test ρ = 0 via posterior inference
- Uncertainty quantification for temporal parameters
- Prior-posterior comparison reveals data informativeness

### Success Criterion
**Not fitting the most complex model, but correctly identifying the data-generating process.**

If the answer is "no temporal structure needed," that's a successful scientific finding, not a failure.

---

## Summary Table of Proposed Models

| Model | Temporal Mechanism | Overdispersion | Parameters | Complexity | Expected Outcome |
|-------|-------------------|----------------|------------|------------|------------------|
| **T1** | Latent AR(1) on log-scale | NegBin φ | 7 + N latents | High | ρ → 0 (spurious AC) |
| **T2** | Dynamic Linear Model | Poisson (may fail) | 2 + 2N states | Very High | σ_δ → 0 (constant trend) |
| **T3** | AR(1) on detrended counts | NegBin φ | 5 | Moderate | ρ → 0 (no evidence for AR) |

**Prediction:** Model T3 will find no evidence for temporal dependence (ρ ≈ 0), validating Designer 1's simpler approach.

**Alternative:** If ρ > 0.5 in T3, escalate to T1 for full state-space treatment.

---

## Files & Code References

**This document:** `/workspace/experiments/designer_3/proposed_models.md`

**Stan code templates:** See inline code blocks above

**Next steps:**
1. Implement models in `/workspace/experiments/designer_3/models/`
2. Create fitting scripts in `/workspace/experiments/designer_3/fit_models.py`
3. Generate diagnostics in `/workspace/experiments/designer_3/diagnostics/`

**Data location:** `/workspace/data/data.csv`

**EDA reference:** `/workspace/eda/eda_report.md`

---

**Prepared by:** Designer 3 (Temporal Structure Specialist)
**Date:** 2025-10-29
**Status:** Ready for implementation and falsification testing
