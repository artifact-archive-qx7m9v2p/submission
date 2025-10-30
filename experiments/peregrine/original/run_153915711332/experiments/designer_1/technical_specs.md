# Technical Specifications - Bayesian Models for Time Series Count Data

## Model Designer 1 Focus: Distributional Choices & Variance Structure

---

## Model 1a: Negative Binomial with Time-Varying Dispersion

### Mathematical Specification

**Likelihood:**
```
C_t ~ NegativeBinomial(μ_t, θ_t)    for t = 1, ..., 40

E[C_t] = μ_t
Var[C_t] = μ_t + μ_t² / θ_t
```

**Mean Function (Exponential Growth with Curvature):**
```
log(μ_t) = α + β₁ · year_t + β₂ · year_t²
```

**Dispersion Function (Time-Varying Precision):**
```
log(θ_t) = γ₀ + γ₁ · year_t
```

**Priors:**
```
α ~ Normal(log(109), 1)          # Intercept centered at observed mean
β₁ ~ Normal(1, 0.5)              # Positive growth expected
β₂ ~ Normal(0, 0.25)             # Agnostic on curvature direction
γ₀ ~ Normal(log(10), 1)          # Moderate baseline dispersion
γ₁ ~ Normal(0, 0.5)              # Allow time-varying dispersion
```

**Posterior Quantities:**
```
For each t:
  - Fitted mean: μ_t
  - Dispersion: θ_t
  - Variance: σ²_t = μ_t + μ_t² / θ_t
  - Coefficient of variation: CV_t = σ_t / μ_t
  - Log-likelihood: log p(C_t | μ_t, θ_t)
  - Posterior predictive: C_rep_t ~ NegativeBinomial(μ_t, θ_t)
```

### Stan Implementation Notes

**Parameterization:** Use `neg_binomial_2(mu, phi)` where:
- `mu` = mean (μ_t)
- `phi` = dispersion (θ_t)
- Variance = mu + mu^2 / phi

**Constraints:**
- θ_t > 0 (enforced by log transformation)
- μ_t > 0 (enforced by log transformation)

**Generated Quantities (for model comparison):**
```stan
generated quantities {
  vector[N] log_lik;           // For LOO-CV
  array[N] int y_rep;          // For posterior predictive checks
  vector[N] variance;          // For variance structure checks

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(y[t] | mu[t], theta[t]);
    y_rep[t] = neg_binomial_2_rng(mu[t], theta[t]);
    variance[t] = mu[t] + square(mu[t]) / theta[t];
  }
}
```

---

## Model 1b: Negative Binomial with Constant Dispersion

### Mathematical Specification

**Likelihood:**
```
C_t ~ NegativeBinomial(μ_t, θ)     for t = 1, ..., 40

E[C_t] = μ_t
Var[C_t] = μ_t + μ_t² / θ
```

**Mean Function:**
```
log(μ_t) = α + β₁ · year_t + β₂ · year_t²
```

**Dispersion (Constant):**
```
θ ~ LogNormal(log(10), 1)
```

**Priors:**
```
α ~ Normal(log(109), 1)
β₁ ~ Normal(1, 0.5)
β₂ ~ Normal(0, 0.25)
θ ~ LogNormal(log(10), 1)        # Single dispersion parameter
```

### Comparison with Model 1a

**Hypothesis Test:**
- H₀: γ₁ = 0 (constant dispersion sufficient)
- H₁: γ₁ ≠ 0 (time-varying dispersion needed)

**Decision Rule:**
- If ΔLOO < 4: Choose Model 1b (parsimony)
- If ΔLOO > 10: Choose Model 1a (complexity justified)
- If 4 ≤ ΔLOO ≤ 10: Examine posterior predictive variance structure

---

## Model 2a: Gamma-Poisson Random Effects (Random Walk)

### Mathematical Specification

**Observation Model:**
```
C_t ~ Poisson(λ_t)               for t = 1, ..., 40
```

**Latent Intensity:**
```
log(λ_t) = log(μ_t) + η_t
```

**Deterministic Trend:**
```
log(μ_t) = α + β₁ · year_t + β₂ · year_t²
```

**Random Walk on Latent Deviations:**
```
η_1 ~ Normal(0, σ_η)
η_t ~ Normal(η_{t-1}, σ_η)       for t = 2, ..., 40
```

**Priors:**
```
α ~ Normal(log(109), 1)
β₁ ~ Normal(1, 0.5)
β₂ ~ Normal(0, 0.25)
σ_η ~ Exponential(1)             # Innovation standard deviation
```

### Non-Centered Parameterization (for computational stability)

**Reparameterization:**
```
z_t ~ Normal(0, 1)               for t = 1, ..., 40

η_1 = σ_η · z_1
η_t = η_{t-1} + σ_η · z_t        for t = 2, ..., 40

This avoids funnel geometry when σ_η is small.
```

### Posterior Quantities

```
For each t:
  - Deterministic mean: μ_t
  - Latent deviation: η_t
  - Total intensity: λ_t = μ_t · exp(η_t)
  - Implied variance: Var[C_t] ≈ λ_t (Poisson) + extra from η_t variation
  - Log-likelihood: log p(C_t | λ_t)
  - Posterior predictive: C_rep_t ~ Poisson(λ_t)
```

### Comparison with Model 1

**Conceptual Difference:**
- Model 1: Overdispersion intrinsic to count process (Negative Binomial)
- Model 2: Overdispersion from latent heterogeneity (Poisson + random effects)

**Empirical Test:**
- Both should produce similar fitted values μ_t
- Model 2 produces η_t that should show:
  - Smooth evolution (random walk)
  - ACF(η_t) decaying slowly
  - σ_η large enough to explain Var/Mean = 68

**Decision Rule:**
- If σ_η < 0.1: No evidence for latent variation, use Model 1
- If Model 2 LOO >> Model 1 LOO: Latent heterogeneity is important
- If similar LOO: Use simpler Model 1 (fewer parameters)

---

## Variance Structure Specifications

### Model 1: Negative Binomial Variance

**Theoretical:**
```
Var[C_t] = μ_t + μ_t² / θ_t
         = μ_t · (1 + μ_t / θ_t)
```

**Properties:**
- Overdispersion when θ < ∞
- Reduces to Poisson when θ → ∞
- Variance grows quadratically with mean when μ >> θ

**Mean-Variance Relationship:**
```
On log-log scale:
log(Var) ≈ log(μ) + log(1 + μ/θ)
         ≈ 2·log(μ) - log(θ)     when μ >> θ

Slope ≈ 2 for large μ/θ
```

### Model 2: Poisson + Random Effects Variance

**Theoretical:**
```
Var[C_t] = E[Var[C_t | η_t]] + Var[E[C_t | η_t]]
         = E[λ_t] + Var[λ_t]
         ≈ μ_t + μ_t² · (exp(σ_η²) - 1)     (if η_t small)
```

**Properties:**
- Overdispersion from variation in log-rates
- Variance can exceed quadratic if σ_η is large
- Random walk structure induces temporal correlation

**Mean-Variance Relationship:**
```
log(Var) ≈ log(μ) + log(1 + μ · k)
where k = exp(σ_η²) - 1

Similar to NB but k here depends on σ_η, not θ
```

### Empirical Comparison

**Data shows:**
```
Var/Mean = 67.99 (overall)
Var/Mean² ≈ 0.62 (suggests quadratic mean-variance)
```

**Model 1 (NB) predicts:**
```
If θ ≈ 10:
  Var/Mean = 1 + Mean/θ = 1 + 109/10 ≈ 12 (too small!)

Need θ ≈ 1.6 to match overall Var/Mean = 68
But this is very small, suggesting extreme overdispersion.
```

**Model 2 (Random Effects) predicts:**
```
If σ_η ≈ 0.5:
  k = exp(0.25) - 1 ≈ 0.28
  Var/Mean ≈ 1 + Mean · k = 1 + 109 · 0.28 ≈ 31 (closer but still low)

Need σ_η ≈ 0.8 to match Var/Mean = 68
```

**Implication:**
Both models require extreme parameter values to match data. This is a **red flag** that either:
1. Simple models are insufficient
2. Time-varying dispersion/variance is critical
3. Some other mechanism is at play

---

## Prior Predictive Distributions

### Model 1a Prior Predictive

**Simulate from priors:**
```
1. α ~ Normal(log(109), 1)
2. β₁ ~ Normal(1, 0.5)
3. β₂ ~ Normal(0, 0.25)
4. γ₀ ~ Normal(log(10), 1)
5. γ₁ ~ Normal(0, 0.5)

For each t:
6. Compute μ_t = exp(α + β₁·year_t + β₂·year_t²)
7. Compute θ_t = exp(γ₀ + γ₁·year_t)
8. Draw C_t ~ NegativeBinomial(μ_t, θ_t)
```

**Expected range (95% prior predictive):**
- Counts: [1, 10000] (very wide, weakly informative)
- Mean at t=0: [40, 300]
- Dispersion: [1, 100]
- Variance/Mean: [2, 200]

**Check:** Does prior include observed data?
- Observed counts [19, 272]: YES
- Observed Var/Mean = 68: YES
- Observed trend direction (increasing): YES (β₁ > 0 with high prior prob)

### Model 2a Prior Predictive

**Simulate from priors:**
```
1. α ~ Normal(log(109), 1)
2. β₁ ~ Normal(1, 0.5)
3. β₂ ~ Normal(0, 0.25)
4. σ_η ~ Exponential(1)

5. η_1 ~ Normal(0, σ_η)
6. For t = 2, ..., 40:
     η_t ~ Normal(η_{t-1}, σ_η)

For each t:
7. Compute μ_t = exp(α + β₁·year_t + β₂·year_t²)
8. Compute λ_t = μ_t · exp(η_t)
9. Draw C_t ~ Poisson(λ_t)
```

**Expected range (95% prior predictive):**
- Counts: [1, 50000] (wider due to random walk can drift far)
- Mean: [10, 1000]
- Random walk SD: [0.1, 5] (prior on σ_η is quite wide)

**Check:** Does prior include observed data?
- YES, but also includes extreme scenarios (λ_t → 0 or λ_t → ∞)
- This is acceptable for weakly informative priors

---

## Posterior Predictive Checks

### Check 1: Marginal Distribution

**Statistic:** Variance/Mean ratio
```
T(y) = Var(y) / Mean(y)

For observed data: T(y_obs) = 67.99

For each posterior sample s:
  1. Simulate y_rep^(s) from posterior predictive
  2. Compute T(y_rep^(s))

Posterior predictive p-value:
  p = P(T(y_rep) ≥ T(y_obs))

Criterion: 0.05 < p < 0.95 (model captures variance)
```

### Check 2: Variance by Time Period

**Statistic:** Variance in early vs late periods
```
T_early = Var(y[year < 0])
T_late = Var(y[year ≥ 0])

Observed: T_early = 157.64, T_late = 4127.91
Ratio: T_late / T_early = 26.19

For each posterior sample:
  1. Simulate y_rep
  2. Compute T_early(y_rep), T_late(y_rep)
  3. Compute ratio

Check: Does 95% posterior predictive interval include 26.19?
```

### Check 3: Extreme Values

**Statistic:** Maximum and minimum counts
```
T_max = max(y)
T_min = min(y)

Observed: T_max = 272, T_min = 19

For each posterior sample:
  1. Simulate y_rep
  2. Compute T_max(y_rep), T_min(y_rep)

Check:
  - P(T_max(y_rep) ≥ 272) ∈ [0.05, 0.95]
  - P(T_min(y_rep) ≤ 19) ∈ [0.05, 0.95]
```

### Check 4: Mean-Variance Relationship

**Method:**
```
1. Bin data by fitted mean into 5 quintiles
2. For each bin, compute empirical variance
3. For each posterior sample:
     a. Simulate y_rep
     b. Bin by fitted mean (same bins as data)
     c. Compute variance in each bin
4. Plot: Empirical variance ± posterior predictive 95% CI

Criterion: Empirical variance should be within 95% CI for 4+ bins
```

### Check 5: Autocorrelation

**Statistic:** Lag-1 correlation
```
T_acf = cor(y[2:40], y[1:39])

Observed: T_acf = 0.9886

For each posterior sample:
  1. Simulate y_rep
  2. Compute T_acf(y_rep)

Check: Does posterior predictive capture extreme autocorrelation?

Note: Models 1 and 2 do NOT include AR structure, so we expect:
  T_acf(y_rep) << T_acf(y_obs)

This is EXPECTED FAILURE - justifies adding AR structure later.
```

---

## Model Comparison Metrics

### LOO-CV (Primary Metric)

**Computation:**
```
For each observation i:
  1. Compute log p(y_i | y_{-i}) using importance sampling
  2. Sum over all i: LOO = Σ log p(y_i | y_{-i})

Higher LOO = better predictive performance
```

**Interpretation:**
```
ΔLOO = LOO_A - LOO_B

|ΔLOO| < 4:   Models are equivalent
4 < |ΔLOO| < 10: Weak preference
|ΔLOO| > 10:    Strong preference
```

**Diagnostics:**
```
Pareto k values:
  k < 0.5: Good
  0.5 < k < 0.7: OK
  k > 0.7: Bad (observation is influential, LOO unreliable)

If many k > 0.7: Model misspecification or outliers
```

### WAIC (Secondary Metric)

**Computation:**
```
WAIC = -2 · (lppd - p_WAIC)

where:
  lppd = Σ_i log(mean_s(p(y_i | θ^(s))))
  p_WAIC = Σ_i Var_s(log p(y_i | θ^(s)))

Lower WAIC = better
```

**Comparison with LOO:**
- LOO is more reliable (Pareto k diagnostics)
- WAIC is simpler to compute
- Usually agree, if not, trust LOO

### Bayes Factor (Supplementary)

**Via bridge sampling:**
```
BF_12 = p(y | M_1) / p(y | M_2)

Interpretation:
  BF > 10: Strong evidence for M_1
  BF < 0.1: Strong evidence for M_2
  0.1 < BF < 10: Weak evidence
```

**Caution:** Sensitive to priors, use only if priors are well-calibrated.

---

## Computational Requirements

### Stan Sampling Settings

**Default:**
```
chains = 4
iter_warmup = 1000
iter_sampling = 2000
adapt_delta = 0.8
max_treedepth = 10
```

**If divergences occur:**
```
adapt_delta = 0.95   # Smaller step size
iter_warmup = 2000   # More tuning
```

**If R-hat > 1.05:**
```
chains = 8           # More chains
iter_sampling = 4000 # Longer sampling
```

### Expected Runtime

**Model 1a (Time-varying NB):**
- Parameters: 5
- Time per iteration: ~0.1 sec
- Total time: ~5 minutes

**Model 1b (Constant NB):**
- Parameters: 4
- Time per iteration: ~0.05 sec
- Total time: ~2 minutes

**Model 2a (Random walk):**
- Parameters: 44 (4 + 40 latent states)
- Time per iteration: ~0.5 sec
- Total time: ~20 minutes

### Memory Requirements

All models: < 2 GB RAM (n = 40 is small)

### Parallelization

Run 4 chains in parallel using 4 CPU cores.
Total runtime = runtime per chain.

---

## Falsification Decision Tree

```
START: Fit Model 1a (Time-varying NB)

├─ Check convergence:
│  ├─ R-hat > 1.05? → Refit with more iterations
│  ├─ Divergences > 5%? → Increase adapt_delta
│  └─ OK? → Continue
│
├─ Check parameters:
│  ├─ θ_min < 0.5? → FAIL: Overdispersion too extreme for NB
│  │                  → Try Model 3 (Compound Poisson) or Escape Route C
│  ├─ θ_min > 100? → FAIL: NB unnecessary, use Poisson
│  ├─ γ₁ ≈ 0 (95% CI includes 0)? → Switch to Model 1b (constant θ)
│  └─ OK? → Continue
│
├─ Posterior predictive checks:
│  ├─ Variance Check fails? → FAIL: Wrong variance structure
│  │                           → Try Model 2 or Escape Route
│  ├─ Extreme value Check fails? → FAIL: Wrong tail behavior
│  └─ OK? → Continue
│
├─ LOO diagnostics:
│  ├─ Many Pareto k > 0.7? → FAIL: Model misspecification
│  │                          → Add robust observation model
│  └─ OK? → Continue
│
└─ Compare to Model 1b:
   ├─ ΔLOO < 4? → Use Model 1b (simpler)
   ├─ ΔLOO > 10? → Use Model 1a (complexity justified)
   └─ 4 ≤ ΔLOO ≤ 10? → Examine variance structure, choose based on domain knowledge

If Model 1 passes all checks:
  → SUCCESS: Report Model 1 as final

If Model 1 fails any check:
  → Fit Model 2 or use Escape Route
  → Repeat decision tree
```

---

## Summary Table

| Model | Parameters | Likelihood | Key Feature | Expected θ or σ | Complexity |
|-------|-----------|------------|-------------|----------------|-----------|
| 1a | 5 | NB(μ_t, θ_t) | Time-varying dispersion | θ: 2-20, varying | Medium |
| 1b | 4 | NB(μ_t, θ) | Constant dispersion | θ: 5-15 | Low |
| 2a | 44 | Poisson(λ_t) | Random walk effects | σ_η: 0.3-0.8 | High |

---

**Document Status:** COMPLETE - Ready for implementation
**Next Step:** Create Stan code files for Models 1a, 1b, 2a
