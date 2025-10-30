# Experiment 1: Negative Binomial State-Space Model

**Model Class:** Dynamic state-space with random walk drift
**Priority:** HIGHEST (all 3 designers recommended)
**Status:** In validation pipeline
**Date Started:** 2025-10-29

---

## Model Specification

### Probabilistic Model

**Observation Model:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
```

**State Evolution (Random Walk with Drift):**
```
η_t ~ Normal(η_{t-1} + δ, σ_η)
η_1 ~ Normal(log(50), 1)
```

**Priors (UPDATED after prior predictive check):**
```
δ ~ Normal(0.05, 0.02)      # Expected ~5% growth per period (KEPT - working well)
σ_η ~ Exponential(20)       # Tight innovations (CHANGED from 10 - was too diffuse)
φ ~ Exponential(0.05)       # Moderate overdispersion (CHANGED from 0.1 - was too diffuse)
```

---

## Scientific Hypotheses

1. **H1 (Main):** Most "overdispersion" is actually temporal correlation - decomposes into smooth latent growth + observation noise
2. **H2:** Growth rate (drift δ) is approximately constant over time
3. **H3:** Innovation variance σ_η is small relative to observation variance (ACF=0.989 supports this)

---

## Rationale

- **EDA Finding:** ACF(1) = 0.989, lag-1 R² = 0.977 indicates C_t ≈ C_{t-1}
- **Designer Consensus:** All 3 parallel designers independently proposed this model class
- **Mechanism:** Separates systematic temporal evolution from count-specific overdispersion
- **Addresses:** Both autocorrelation AND overdispersion simultaneously

---

## Expected Parameters

Based on EDA and designer predictions:
- **Drift (δ):** ≈ 0.06 per period (6% growth → 134% over full range)
- **Innovation SD (σ_η):** ≈ 0.05-0.10 (small fluctuations around smooth trend)
- **Dispersion (φ):** ≈ 10-20 (much less than naive 68 from treating as IID)
- **Interpretation:** Growth is a smooth latent process with small random fluctuations

---

## Falsification Criteria

**Abandon this model if:**
1. σ_η → 0 (degenerate model, state-space adds nothing)
2. σ_η ~ observation SD (no benefit from state decomposition)
3. Residual ACF > 0.5 after accounting for latent state (autocorrelation not captured)
4. One-step-ahead coverage < 75% (poor predictive performance)
5. Divergences > 20% of samples (pathological geometry)

**Switch to alternative if:**
- Innovation variance shows regime-specific patterns → Try changepoint model
- Residuals show clear non-random patterns → Try GP
- State estimates are just smoothed observations → Simplify to polynomial

---

## Computational Strategy

**Implementation:** Stan (primary), PyMC (fallback)

**Parameterization:** Non-centered for latent states
```stan
// Non-centered parameterization
parameters {
  real delta;
  real<lower=0> sigma_eta;
  real<lower=0> phi;
  real eta_1;
  vector[N-1] eta_raw;  // Non-centered
}
transformed parameters {
  vector[N] eta;
  eta[1] = eta_1;
  for (t in 2:N) {
    eta[t] = eta[t-1] + delta + sigma_eta * eta_raw[t-1];
  }
}
```

**Generated Quantities (REQUIRED for LOO):**
```stan
generated quantities {
  vector[N] log_lik;
  vector[N] y_pred;
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | eta[t], phi);
    y_pred[t] = neg_binomial_2_log_rng(eta[t], phi);
  }
}
```

**Sampling Settings:**
- Chains: 4
- Warmup: 1000
- Samples: 2000 per chain
- adapt_delta: 0.95 (increase if divergences)
- max_treedepth: 12

---

## Validation Pipeline Status

- [ ] Prior Predictive Check
- [ ] Simulation-Based Validation
- [ ] Model Fitting
- [ ] Posterior Predictive Check
- [ ] Model Critique

---

## Notes

(To be updated during validation)
