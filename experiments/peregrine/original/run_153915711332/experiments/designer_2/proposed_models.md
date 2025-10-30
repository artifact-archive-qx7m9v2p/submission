# Bayesian Model Design: Temporal Structure & Trend Specification
## Designer 2 - Time Series Count Models

**Date:** 2025-10-29
**Data:** `/workspace/data/data_designer_2.csv` (40 observations)
**Focus Area:** Temporal structure, trend specification, autocorrelation modeling

---

## Executive Summary

This document proposes three competing Bayesian model classes for time series count data exhibiting extreme overdispersion (Var/Mean = 67.99), massive autocorrelation (ACF(1) = 0.989), and strong exponential growth (8.45x increase). The models differ fundamentally in how they conceptualize the data generation process:

1. **Dynamic Negative Binomial State-Space Model** - Treats data as a latent growth process with observation error
2. **Changepoint Negative Binomial Model** - Hypothesizes discrete regime shift at t ≈ 0.3
3. **Gaussian Process Count Model** - Assumes smooth, flexible nonparametric trend

**Critical Insight:** The near-perfect lag-1 autocorrelation (R² = 0.977) suggests the data may be better understood as a **stochastic process evolving over time** rather than independent observations with a fixed trend. This fundamentally challenges standard GLM approaches.

**Falsification Strategy:** All models will be evaluated against the same out-of-sample prediction task and posterior predictive checks. We will abandon model classes that show:
- Prior-posterior conflict (θ parameter hitting boundaries)
- Poor one-step-ahead prediction (coverage < 80%)
- Residual autocorrelation persisting (ACF > 0.5 after accounting for temporal structure)

---

## Problem Formulation: Competing Hypotheses

### The Core Question
**What generated this data?** Three fundamentally different mechanisms are plausible:

### Hypothesis 1: Latent Growth Process (State-Space)
**Claim:** There is an underlying "true" growth process that evolves smoothly over time, but we observe it with error. The massive autocorrelation reflects the persistence of the latent state.

**Prediction:** A state-space model will show that much of the "overdispersion" is actually temporal correlation. The innovation variance will be much smaller than the observation variance.

**I will abandon this if:**
- The latent state variance is similar to observation variance (no benefit from state decomposition)
- Innovations show strong autocorrelation (state doesn't capture dynamics)
- Model converges to degenerate parameters (σ² → 0 or σ² → ∞)

### Hypothesis 2: Discrete Regime Shift (Changepoint)
**Claim:** The system underwent a structural change around year ≈ 0.3. Before and after are fundamentally different processes with different growth rates and variance structures.

**Prediction:** A changepoint model will reveal distinct parameter regimes with statistically significant differences in growth rates. The changepoint location will be strongly identified (narrow posterior).

**I will abandon this if:**
- Changepoint posterior is uniform (no preferred location)
- Pre/post parameters are not meaningfully different (credible intervals overlap heavily)
- Smooth models (GP, splines) fit significantly better than any changepoint model

### Hypothesis 3: Smooth Nonparametric Growth (Gaussian Process)
**Claim:** The apparent "changepoint" is an artifact - growth is smooth but accelerating. The process is better described by a flexible nonparametric function than any parametric form.

**Prediction:** A GP with appropriate kernel will show that acceleration is gradual, not sudden. The function will be smoother than discrete changepoints suggest.

**I will abandon this if:**
- GP shows discontinuities or near-discontinuities (suggests true changepoint)
- Parametric models (exponential, polynomial) fit equivalently (Occam's razor favors simpler)
- Computational issues prevent convergence (n=40 should be tractable)

---

## Model Class 1: Dynamic Negative Binomial State-Space Model

### Priority: HIGHEST (⭐⭐⭐)

### Rationale
The ACF(1) = 0.989 and lag-1 R² = 0.977 strongly suggest an autoregressive process. The slope ≈ 1.011 in C(t+1) ~ C(t) regression indicates a **near-random-walk** with small positive drift. This is classic time series behavior, not cross-sectional. A state-space model decomposes variance into:
- **Systematic evolution:** latent state drift/growth
- **Observation noise:** count-specific variation (overdispersion)

This addresses both autocorrelation and overdispersion simultaneously.

### Full Probabilistic Specification

**Observation Model:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
```

**State Evolution (Latent Process):**
```
η_t = η_{t-1} + δ + β * year_t + ε_t
ε_t ~ Normal(0, σ²_η)
```

Where:
- `η_t` = latent log-intensity at time t
- `δ` = drift parameter (average log-growth per period)
- `β` = effect of external time trend (optional, tests if drift is constant or time-varying)
- `σ²_η` = innovation variance (how much latent state fluctuates)
- `φ` = NegBin dispersion (overdispersion after accounting for temporal structure)

**Initial Condition:**
```
η_1 ~ Normal(log(30), 1)  # Based on observed first count ≈ 29
```

**Priors:**
```
δ ~ Normal(0.05, 0.1)           # Weak prior on 5% per-period growth in log-space
β ~ Normal(0, 0.5)              # Flexible time effect
σ²_η ~ HalfNormal(0.5)          # Expect modest innovations if drift captures trend
φ ~ Gamma(2, 0.1)               # Overdispersion; mean = 20, allows wide range
```

**Log-Likelihood (for LOO):**
```
log p(C_t | η_t, φ) = log NegBin(C_t; exp(η_t), φ)
```

### Key Predictions
1. **σ²_η should be small** (< 0.1): Most variation is systematic drift, not random innovation
2. **φ should be moderate** (5-30): Less overdispersion than unconditional variance suggests
3. **δ should be positive** (0.03-0.08): Consistent positive drift
4. **One-step-ahead predictions should be excellent** (< 10% MAPE)

### Expected Computational Challenges
- **High autocorrelation in MCMC:** The near-random-walk structure creates strong posterior correlations between consecutive η_t values. This slows mixing.
  - **Solution:** Use non-centered parameterization: `η_t = η_{t-1} + δ + β*year_t + σ_η * z_t` where `z_t ~ Normal(0,1)`

- **Identifiability:** δ and β may trade off if year is linear
  - **Solution:** Consider fixing β = 0 in variant models, or using informative priors

- **Divergences in Stan:** State-space models can have difficult geometry
  - **Solution:** Increase adapt_delta to 0.95, use more warmup iterations

### Falsification Criteria: I will abandon this model if...

1. **σ²_η posterior is diffuse or hits prior boundaries** → State decomposition isn't helping
2. **Residuals (observed - E[μ_t | η_t]) still show ACF > 0.5** → Temporal structure not captured
3. **φ → ∞ or φ → 0** → NegBin distribution inappropriate
4. **One-step-ahead prediction coverage < 75%** → Model doesn't predict well despite good fit
5. **Prior-posterior overlap > 80%** → Data not informative, model not learning

### Model Variants to Test

**Variant 1a: No external time trend (β = 0)**
```
η_t = η_{t-1} + δ + ε_t
```
Tests whether drift alone explains growth, or if acceleration (year effect) is needed.

**Variant 1b: Stochastic drift**
```
δ_t = δ_{t-1} + ν_t,  ν_t ~ Normal(0, σ²_δ)
η_t = η_{t-1} + δ_t + ε_t
```
Allows growth rate itself to evolve (local linear trend model). More flexible but more parameters.

**Variant 1c: AR(1) on log-scale (simpler alternative)**
```
log(C_t) = α + ρ * log(C_{t-1}) + γ * year_t + ε_t
C_t ~ NegBin(exp(log(C_t)), φ)
```
Traditional autoregressive specification. Less principled for counts but easier to estimate.

### Expected Outcomes
- **Best case:** σ²_η ≈ 0.05, δ ≈ 0.06, φ ≈ 15, residual ACF < 0.2, LOO-ELPD ≈ -150
- **Warning signs:**
  - If σ²_η > 0.3, state model isn't helping
  - If φ < 3, overdispersion inadequately modeled
  - If posterior correlations among η_t values remain > 0.99, reparameterization needed

---

## Model Class 2: Changepoint Negative Binomial Model

### Priority: MEDIUM (⭐⭐)

### Rationale
The CUSUM analysis detected a clear U-shaped pattern with minimum at year ≈ 0.3, where mean jumps from 45.67 to 205.12 (4.5x increase). The t-test is highly significant (p < 0.0001). This could represent:
- Exogenous shock to the system
- Policy change or intervention
- Crossing a critical threshold (phase transition)

However, this could also be an illusion created by smooth exponential acceleration. We need to test this explicitly.

### Full Probabilistic Specification

**Observation Model:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β_0 + β_1 * year_t + β_2 * I(year_t ≥ τ) + β_3 * (year_t - τ) * I(year_t ≥ τ)
```

Where:
- `τ` = changepoint location (in standardized year scale)
- `β_0` = intercept (baseline log-count at year = 0)
- `β_1` = pre-changepoint slope
- `β_2` = level shift at changepoint (discontinuity)
- `β_3` = slope change after changepoint (interaction)
- `φ` = NegBin dispersion parameter

**Alternative parameterization (piecewise linear):**
```
If year_t < τ:
    log(μ_t) = α_1 + γ_1 * year_t
If year_t ≥ τ:
    log(μ_t) = α_2 + γ_2 * year_t
```
With continuity constraint: α_1 + γ_1 * τ = α_2 + γ_2 * τ

**Priors:**
```
β_0 ~ Normal(log(100), 1)          # Intercept at median count
β_1 ~ Normal(0.5, 0.5)             # Positive growth expected
β_2 ~ Normal(0, 1)                 # Level shift (could be positive or negative)
β_3 ~ Normal(0, 0.5)               # Slope change (test if growth accelerates)
τ ~ Uniform(-1.5, 1.5)             # Changepoint anywhere in observed range
φ ~ Gamma(2, 0.1)                  # Same as Model 1
```

**Alternative: Informative prior on τ based on EDA:**
```
τ ~ Normal(0.3, 0.2)               # Centered on CUSUM minimum, allows uncertainty
```

**Log-Likelihood:**
```
log p(C_t | θ) = log NegBin(C_t; μ_t, φ)
```

### Key Predictions
1. **τ posterior should be narrow** (SD < 0.3) if changepoint is real
2. **β_2 should be significantly positive** (95% CI excludes 0) for level shift
3. **β_3 should differ from 0** if post-change growth differs
4. **Model should fit better than smooth polynomial** (ΔLOO > 10)

### Expected Computational Challenges
- **Changepoint identification:** τ has discrete support in practice (only 40 time points). Can create multimodal posterior.
  - **Solution:** Use discrete grid search or adaptive MCMC proposals

- **Label switching:** If β_2 ≈ 0 and β_3 ≈ 0, pre/post regimes are indistinguishable
  - **Solution:** Add weak ordering constraint or use informative priors

- **Discontinuity:** The indicator function I(year ≥ τ) creates discontinuity in parameter space
  - **Solution:** Stan handles this, but may need tighter adapt_delta (0.95)

### Falsification Criteria: I will abandon this model if...

1. **τ posterior is uniform or bimodal across entire range** → No preferred changepoint location
2. **95% CI for β_2 and β_3 both include 0** → No evidence of regime change
3. **Smooth models (polynomial, GP) have ΔLOO > 10 in their favor** → Occam's razor
4. **Residual ACF remains > 0.7** → Changepoint doesn't explain temporal structure
5. **Posterior predictive checks show systematic deviations** → Model structure wrong

### Model Variants to Test

**Variant 2a: Continuous transition (smooth changepoint)**
```
log(μ_t) = β_0 + β_1 * year_t + β_2 * logistic((year_t - τ) / λ) + β_3 * (year_t - τ) * logistic((year_t - τ) / λ)
```
Where `λ` controls transition smoothness. Tests whether change is abrupt or gradual.

**Variant 2b: Variance changepoint**
```
Same mean structure, but:
φ_t = φ_1 * I(year_t < τ) + φ_2 * I(year_t ≥ τ)
```
Tests if overdispersion also changed at the break.

**Variant 2c: Multiple changepoints**
```
log(μ_t) = β_0 + Σ_k β_k * I(year_t ≥ τ_k)
```
Exploratory model for complex regime structure (use only if single changepoint clearly inadequate).

### Expected Outcomes
- **Best case:** τ ≈ 0.25-0.35, β_2 ≈ 0.5-1.0, β_3 ≈ 0.3-0.6, φ ≈ 10-25, LOO-ELPD ≈ -155
- **Warning signs:**
  - If τ posterior has SD > 0.5, changepoint location uncertain
  - If β_2 and β_3 near zero, changepoint not meaningful
  - If smooth model (Variant 2a with small λ) fits much better, suggests gradual transition

---

## Model Class 3: Gaussian Process Negative Binomial Model

### Priority: MEDIUM-LOW (⭐⭐)

### Rationale
Both Models 1 and 2 make strong parametric assumptions (random walk, linear changepoint). A GP provides a **falsification test**: if the truth is neither of these, the GP should fit much better. GPs are also scientifically honest - they admit "we don't know the functional form" and let data speak.

The EDA shows:
- Polynomial models fit increasingly well up to degree 4 (R² = 0.99)
- Exponential model fits well (R² = 0.935 on log-scale)
- No single parametric form is obviously correct

A GP with appropriate kernel can represent ANY smooth function, making it the ultimate flexible alternative.

### Full Probabilistic Specification

**Observation Model:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = f(year_t)
```

**Gaussian Process Prior:**
```
f ~ GP(m(year), k(year, year'))
```

**Mean Function (optional, can be 0):**
```
m(year) = β_0 + β_1 * year  # Linear trend as prior mean
```

**Kernel Function (Squared Exponential / RBF):**
```
k(year_i, year_j) = α² * exp(- (year_i - year_j)² / (2 * ℓ²) )
```

Where:
- `α²` = marginal variance (how much f varies around mean)
- `ℓ` = lengthscale (how quickly correlation decays)
- Small `ℓ` → wiggly function, large `ℓ` → smooth function

**Priors:**
```
β_0 ~ Normal(log(100), 1)
β_1 ~ Normal(0.5, 0.5)
α ~ HalfNormal(1)                  # Moderate deviations from linear trend
ℓ ~ InvGamma(5, 5)                 # Mean ≈ 1, encourages smooth functions
φ ~ Gamma(2, 0.1)
```

**Log-Likelihood:**
```
log p(C | f, φ) = Σ_t log NegBin(C_t; exp(f(year_t)), φ)
```

### Alternative Kernel: Matern 5/2 (More Flexible)
```
k(year_i, year_j) = α² * (1 + sqrt(5)*r/ℓ + 5*r²/(3*ℓ²)) * exp(-sqrt(5)*r/ℓ)
where r = |year_i - year_j|
```
Less smooth than RBF, allows sharper transitions (compromise between RBF and changepoint).

### Key Predictions
1. **Lengthscale ℓ should be moderate** (0.3-1.0): Not too wiggly, not too smooth
2. **If ℓ < 0.2, data is forcing rapid changes** → Suggests changepoint structure
3. **If ℓ > 2, function is nearly linear** → Parametric model sufficient
4. **Posterior of f should show smooth acceleration** → Neither linear nor discrete jump

### Expected Computational Challenges
- **Matrix inversion:** GP requires inverting n×n covariance matrix
  - **For n=40:** Not a problem, < 1 second per iteration
  - **Solution:** Use Cholesky decomposition, standard in Stan

- **Posterior geometry:** High correlation between nearby function values
  - **Solution:** Non-centered parameterization: `f = m + L*z` where `L = cholesky(K)`, `z ~ N(0,I)`

- **Hyperparameter identifiability:** α and ℓ can trade off
  - **Solution:** Informative priors, or fix one parameter in sensitivity analysis

### Falsification Criteria: I will abandon this model if...

1. **GP posterior is effectively linear** (f ≈ β_0 + β_1*year everywhere) → Overparameterized
2. **GP shows discontinuities** (posterior variance spikes at specific points) → Changepoint more appropriate
3. **Parametric model has ΔLOO > 15 in its favor** → Occam's razor
4. **Lengthscale posterior hits prior boundaries** (ℓ → 0 or ℓ → ∞) → Kernel misspecified
5. **Computational issues prevent convergence** (shouldn't happen with n=40)

### Model Variants to Test

**Variant 3a: Zero mean GP**
```
m(year) = 0
```
Fully nonparametric - no prior linear trend assumption.

**Variant 3b: Additive model (Parametric + GP)**
```
log(μ_t) = β_0 + β_1 * year_t + β_2 * year_t² + f(year_t)
f ~ GP(0, k)
```
Captures systematic trend parametrically, uses GP for deviations. More interpretable.

**Variant 3c: GP with periodic kernel (exploratory)**
```
k_per(year_i, year_j) = α² * exp(-2 * sin²(π|year_i - year_j|/P) / ℓ²)
```
Tests for cyclical patterns (unlikely given EDA, but worth checking).

### Expected Outcomes
- **Best case:** ℓ ≈ 0.5-0.8, α ≈ 0.3-0.7, f shows smooth S-curve, φ ≈ 8-20, LOO-ELPD ≈ -152
- **Warning signs:**
  - If ℓ < 0.15, function is too wiggly (overfitting)
  - If posterior of f has high uncertainty (wide credible bands), n=40 insufficient for GP
  - If parametric models fit equivalently, GP is overkill

---

## Cross-Model Comparisons and Decision Rules

### Model Selection Criteria

**Primary Metric: LOO-ELPD (Leave-One-Out Expected Log Pointwise Predictive Density)**
- Bayesian cross-validation, gold standard for predictive performance
- **Decision rule:** If ΔLOO < 4, models are equivalent (within SE)
- **Decision rule:** If ΔLOO > 10, strong evidence for better model

**Secondary Metrics:**
1. **One-step-ahead predictive accuracy:**
   - For each t > 1, predict C_t using data up to t-1
   - Compute coverage of 80% and 95% prediction intervals
   - **Threshold:** Coverage should be within 5% of nominal (e.g., 75-85% for 80% interval)

2. **Residual diagnostics:**
   - ACF of Pearson residuals: `r_t = (C_t - E[μ_t]) / sqrt(Var[μ_t])`
   - **Threshold:** ACF(1) should be < 0.3, all lags < 0.5

3. **Posterior predictive checks:**
   - Replicate datasets from posterior: `C_rep ~ p(C | θ_post)`
   - Check: max(C_rep), min(C_rep), mean(C_rep), variance(C_rep), ACF(C_rep)
   - **Threshold:** Observed data should be within central 80% of replicated distribution

### Falsification Decision Tree

```
START: Fit all three model classes

├─ If Model 1 (State-Space) has LOO-ELPD > others by 10+:
│  ├─ Check: Is σ²_η << Var[log(C)]? [YES → Accept Model 1] [NO → Investigate why]
│  ├─ Check: Residual ACF < 0.3? [YES → Accept] [NO → Model incomplete]
│  └─ If accepted: Report δ and φ as primary inferential targets
│
├─ If Model 2 (Changepoint) has LOO-ELPD > others by 10+:
│  ├─ Check: Is τ posterior narrow (SD < 0.3)? [YES → Continue] [NO → Changepoint uncertain]
│  ├─ Check: Are β_2 and β_3 significantly non-zero? [YES → Accept] [NO → No regime shift]
│  ├─ Compare to Variant 2a (smooth): If smooth much better → Reject discrete changepoint
│  └─ If accepted: Report τ, pre/post growth rates
│
├─ If Model 3 (GP) has LOO-ELPD > others by 10+:
│  ├─ Check: Is ℓ in reasonable range (0.2-2.0)? [YES → Continue] [NO → Kernel issue]
│  ├─ Check: Is f clearly nonlinear? [YES → Accept] [NO → Simpler model sufficient]
│  ├─ Compare to polynomial: If polynomial within LOO-ELPD 5 → Prefer polynomial (Occam)
│  └─ If accepted: Report f(year) function, highlight key features
│
├─ If all models within LOO-ELPD 4 of each other:
│  ├─ Report: "Data insufficient to discriminate model classes"
│  ├─ Use Bayesian Model Averaging (BMA) for predictions
│  └─ Emphasize: Predictive performance, not mechanism
│
└─ If all models show poor fit (coverage < 70%, ACF > 0.6):
   ├─ RED FLAG: None of our model classes are appropriate
   ├─ Investigate:
   │  - Is count distribution wrong? (Try Zero-Inflated, Conway-Maxwell-Poisson)
   │  - Is there measurement error in year variable?
   │  - Are there unobserved covariates creating spurious patterns?
   └─ ESCALATE: Return to EDA, consider entirely different model classes
```

### What Would Make Me Reconsider Everything?

**Scenario 1: All models fail residual checks**
- If ACF(1) > 0.7 in all models → We haven't captured temporal structure at all
- **Pivot:** Consider GARCH-type models (volatility clustering), long-memory processes (ARFIMA), or admit temporal structure is more complex than any of these models

**Scenario 2: Overdispersion parameter φ → extreme values in all models**
- If φ → 0 (approaching Poisson): Overdispersion is spurious, actually from model misspecification
- If φ → ∞ (variance exploding): Count distribution is wrong, try Conway-Maxwell-Poisson or hurdle models
- **Pivot:** Reconsider likelihood family entirely

**Scenario 3: Posterior predictive checks fail systematically**
- If replicated max(C) consistently << observed max(C): Right tail too light
- If replicated ACF systematically << observed ACF: Temporal structure underestimated
- **Pivot:** Hybrid models (e.g., Changepoint + GP), or admit model class inadequacy

**Scenario 4: Prior-posterior overlap > 80% for all key parameters**
- Data are not informative → n=40 insufficient for these complex models
- **Pivot:** Use simpler models (basic NB-GLM), acknowledge limitations, emphasize predictive intervals over mechanism

---

## Stress Tests: Designed to Break the Models

### Stress Test 1: Extreme Extrapolation
**Setup:** Fit models on years < 1.0, predict years ≥ 1.0 (last 8 observations)

**Expected failures:**
- **Model 2 (Changepoint):** Will extrapolate linearly, underpredict if growth is accelerating
- **Model 1 (State-Space):** If drift is constant, will also underpredict
- **Model 3 (GP):** Should extrapolate most conservatively (wide intervals)

**Success criteria:**
- At least one model achieves > 70% coverage on held-out data
- If all fail: Growth is inherently unpredictable, emphasize this in reporting

### Stress Test 2: Jackknife Around Changepoint
**Setup:** Remove observations near year ≈ 0.3 (indices 22-26), refit Model 2

**Expected failure modes:**
- τ posterior becomes bimodal or shifts dramatically
- β_2 and β_3 posteriors widen substantially

**Success criteria:**
- If Model 2 is robust, τ should remain near 0.3 (± 0.2)
- If not robust → Changepoint is artifact of those specific observations

### Stress Test 3: Simulate Under Alternative Mechanisms
**Setup:** Generate synthetic data from:
1. True exponential growth with i.i.d. NB errors (no autocorrelation)
2. True AR(1) with constant mean (no trend)
3. True changepoint at known location

**Expected failures:**
- Models should misidentify mechanisms in cross-scenarios
- E.g., State-Space on data (3) should show poor fit or misplace changepoint

**Success criteria:**
- If a model correctly identifies the simulated mechanism, it validates our inferential approach
- If not → We lack power to distinguish these mechanisms with n=40

### Stress Test 4: Prior Sensitivity
**Setup:** Refit all models with:
- Very vague priors (SD × 10)
- Strongly informative priors (SD / 10)

**Expected failure modes:**
- With vague priors: Posterior may be diffuse, computational issues
- With strong priors: Posterior may not move, prior-data conflict

**Success criteria:**
- Posteriors should be similar across prior specifications (data dominate)
- If highly sensitive → Data are weak, results driven by assumptions

---

## Implementation Plan: Stan/PyMC Specifications

All models will be implemented in **Stan** (primary) with **PyMC** as backup if needed.

### Stan Implementation Notes

**Model 1 (State-Space):**
- File: `model_1_state_space.stan`
- Use `generated quantities` block for one-step-ahead predictions
- Non-centered parameterization critical for η_t

**Model 2 (Changepoint):**
- File: `model_2_changepoint.stan`
- Discrete τ: Use marginalization or grid search in `transformed parameters`
- Continuous τ: Use adaptive HMC (standard)

**Model 3 (GP):**
- File: `model_3_gaussian_process.stan`
- Use `gp_exp_quad_cov` built-in function
- Add `+ 1e-9` to diagonal for numerical stability

**All models must include:**
```stan
generated quantities {
  vector[N] log_lik;      // For LOO-CV
  vector[N] C_rep;        // Posterior predictive replicates

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], phi);
    C_rep[t] = neg_binomial_2_rng(mu[t], phi);
  }
}
```

### PyMC Alternative (if Stan fails)
- Use `pm.NegativeBinomial` for observation model
- Use `pm.GaussianRandomWalk` for state-space
- Use `pm.gp.Latent` for GP models
- Compute LOO via `arviz.loo()`

### Computational Budget
- **Chains:** 4
- **Iterations:** 2000 warmup + 2000 sampling = 4000 per chain
- **Total samples:** 8000 posterior draws
- **Time estimate:** 5-15 minutes per model on standard laptop
- **Convergence diagnostics:** R-hat < 1.01, ESS > 400

---

## Expected Discoveries and Pivots

### Most Likely Outcome
**My prediction:** Model 1 (State-Space) will perform best, revealing that:
- Most "overdispersion" is actually temporal correlation
- Innovations σ²_η are small (≈ 0.05-0.1)
- Drift δ is positive and consistent (≈ 0.05-0.07)
- Residual φ is moderate (10-20), much less than naive φ ≈ 70

**This would suggest:** The data generation process is a smooth random walk with positive drift, not a cross-sectional GLM with fixed trend.

### Alternative Outcome 1: Changepoint Wins
**If Model 2 has ΔLOO > 10 over Model 1:**
- True structural break occurred (policy change? threshold effect?)
- Scientific interpretation: System underwent phase transition
- Implication: Future predictions depend critically on whether another break is possible

**Pivot:** Investigate what happened at year ≈ 0.3 (if real data context is known)

### Alternative Outcome 2: GP Wins
**If Model 3 has ΔLOO > 10 over Models 1-2:**
- Truth is more complex than either random walk or changepoint
- Growth is smooth but highly nonlinear (maybe logistic, Gompertz, etc.)
- **Action:** Inspect GP function carefully, try to fit parametric form post-hoc

**Pivot:** Propose specific parametric families based on GP shape (e.g., if S-curve → fit logistic model)

### Alternative Outcome 3: All Models Fail
**If all show ACF > 0.6, coverage < 70%, LOO-ELPD > -180:**
- **Fundamental problem:** Our model classes don't match data generation process
- **Possible causes:**
  1. Count distribution is wrong (not NegBin)
  2. Temporal structure is more complex (need ARMA, not just AR)
  3. Unobserved covariates are driving dynamics
  4. Measurement error in outcome or time variable

**Pivot:** Return to EDA, consider:
- Conway-Maxwell-Poisson (flexible dispersion)
- Bivariate models (if C has latent components)
- Survival/renewal process models
- Admit defeat: "Data appear to be generated by a process not well-represented by standard time series models"

---

## Domain Constraints and Scientific Plausibility

### Assumptions About Data Context
Since domain context is not specified, I assume:
- **Time is meaningful:** year variable represents actual temporal sequence
- **Counts are real:** Not just discretized continuous measurements
- **Process is stationary in some sense:** No infinite growth (bounded by some carrying capacity)
- **No external interventions:** Besides possible changepoint, no known exogenous shocks

### Plausibility Checks
**All models will be rejected if they predict:**
1. **Negative counts** (impossible by construction with NegBin)
2. **Infinite growth** (exponential models can explode)
   - **Safeguard:** Flag predictions > 500 as "extrapolation uncertain"
3. **Negative growth** (inconsistent with observed monotonic trend)
   - **Safeguard:** Constrain δ > 0 in Model 1, or use informative priors

### Scientific Reporting
Regardless of which model wins, the report will include:
1. **Mechanistic interpretation:** What does this model say about the process?
2. **Uncertainty quantification:** Credible intervals, prediction intervals
3. **Limitations:** What can this model NOT tell us?
4. **Robustness:** How sensitive are conclusions to modeling choices?

---

## Red Flags and Stopping Rules

### Red Flags That Would Halt Analysis

**Computational Red Flags:**
1. **R-hat > 1.05 after 10,000 iterations:** Model fundamentally misspecified or data pathological
2. **Divergent transitions > 5% even with adapt_delta=0.99:** Posterior geometry incompatible with HMC
3. **ESS < 100 for key parameters:** Cannot trust posterior inference

**Statistical Red Flags:**
1. **All models have LOO-ELPD < -200:** Models are barely better than random guessing
2. **Posterior predictive p-values < 0.01 or > 0.99 for multiple test statistics:** Systematic model failure
3. **Residual ACF actually increases compared to raw data:** Model making things worse

**Substantive Red Flags:**
1. **φ posterior concentrated near 0:** Overdispersion was illusory, Poisson would suffice (contradicts EDA)
2. **All trend parameters include 0 in 95% CI:** No evidence of growth (contradicts EDA)
3. **Predictions wildly inconsistent with observed data range:** Model not capturing basic structure

### Stopping Rules

**Rule 1: If any red flag occurs, STOP and diagnose**
- Do not proceed to model comparison
- Investigate root cause
- Consider whether data or models are at fault

**Rule 2: If all models fail basic diagnostics, STOP model development**
- Return to EDA with specific questions
- Consider alternative data representations (differences, log-differences, etc.)
- Consult domain experts if available

**Rule 3: If models succeed but disagree strongly (ΔLOO < 2), STOP and report uncertainty**
- Do not force a "winner"
- Use Bayesian Model Averaging
- Emphasize that mechanism is unidentified

**Rule 4: Maximum 3 iterations of model refinement**
- If after 3 rounds of variant testing we still have red flags → Admit the problem is too hard
- Document what doesn't work as valuable negative result

---

## Summary: Model Comparison Table

| Model Class | Key Mechanism | Best For | Fails If | Priority | Expected LOO-ELPD |
|-------------|---------------|----------|----------|----------|-------------------|
| **1. State-Space** | Latent random walk + drift | Strong autocorrelation | σ²_η ≈ σ²_obs, ACF persists | ⭐⭐⭐ | -148 to -155 |
| **2. Changepoint** | Discrete regime shift | Structural break hypothesis | τ diffuse, β_2≈0, smooth fits better | ⭐⭐ | -152 to -160 |
| **3. GP** | Nonparametric smooth trend | Unknown functional form | ℓ extreme, equivalent to parametric | ⭐⭐ | -150 to -158 |

**Decision Hierarchy:**
1. Fit all three, compute LOO-ELPD
2. If clear winner (ΔLOO > 10): Validate with diagnostics, report winner
3. If ambiguous (ΔLOO < 5): Use BMA, report uncertainty
4. If all fail: Pivot to alternative model classes or report negative result

**Success Definition:**
- Success is NOT "Model 1 wins" or "Model 2 wins"
- Success IS "We correctly identify which mechanism generated the data"
- If all models fail, SUCCESS is recognizing and documenting that failure

---

## Final Thoughts: Embracing Uncertainty

This analysis is designed to **falsify hypotheses, not confirm them**. I expect:
- At least one model class to be clearly inadequate
- Surprising findings that challenge my initial assumptions
- Possibly, that none of these models are quite right

**I will consider this a successful analysis if:**
1. We discover which model class is wrong (even if we don't find the right one)
2. We quantify prediction uncertainty honestly (even if very wide)
3. We identify what additional data or information would resolve ambiguity
4. We document failure modes clearly for future analysts

**I will consider this a failed analysis if:**
- We report a "best" model that fails basic diagnostics
- We ignore red flags to complete the task
- We overstate certainty in model selection or parameter estimates

The goal is truth, not task completion.

---

## Appendix: Stan Code Skeletons

### Model 1: State-Space (Skeleton)
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
}
parameters {
  real delta;                    // drift
  real beta;                     // time effect
  real<lower=0> sigma_eta;       // innovation SD
  real<lower=0> phi;             // NegBin dispersion
  vector[N] eta_raw;             // non-centered latent state
}
transformed parameters {
  vector[N] eta;
  eta[1] = log(30) + eta_raw[1] * sigma_eta;
  for (t in 2:N) {
    eta[t] = eta[t-1] + delta + beta * year[t] + sigma_eta * eta_raw[t];
  }
}
model {
  // Priors
  delta ~ normal(0.05, 0.1);
  beta ~ normal(0, 0.5);
  sigma_eta ~ normal(0, 0.5);
  phi ~ gamma(2, 0.1);
  eta_raw ~ std_normal();

  // Likelihood
  for (t in 1:N) {
    C[t] ~ neg_binomial_2_log(eta[t], phi);
  }
}
generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | eta[t], phi);
    C_rep[t] = neg_binomial_2_log_rng(eta[t], phi);
  }
}
```

### Model 2: Changepoint (Skeleton)
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
}
parameters {
  real beta_0;
  real beta_1;
  real beta_2;               // level shift
  real beta_3;               // slope change
  real tau;                  // changepoint
  real<lower=0> phi;
}
transformed parameters {
  vector[N] log_mu;
  for (t in 1:N) {
    log_mu[t] = beta_0 + beta_1 * year[t]
                + beta_2 * (year[t] >= tau ? 1 : 0)
                + beta_3 * (year[t] >= tau ? year[t] - tau : 0);
  }
}
model {
  beta_0 ~ normal(log(100), 1);
  beta_1 ~ normal(0.5, 0.5);
  beta_2 ~ normal(0, 1);
  beta_3 ~ normal(0, 0.5);
  tau ~ uniform(-1.5, 1.5);
  phi ~ gamma(2, 0.1);

  C ~ neg_binomial_2_log(log_mu, phi);
}
generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu[t], phi);
    C_rep[t] = neg_binomial_2_log_rng(log_mu[t], phi);
  }
}
```

### Model 3: Gaussian Process (Skeleton)
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> alpha;       // GP marginal SD
  real<lower=0> rho;         // GP lengthscale
  real<lower=0> phi;
  vector[N] f_raw;           // non-centered GP
}
transformed parameters {
  vector[N] f;
  {
    matrix[N,N] L_K;
    matrix[N,N] K = gp_exp_quad_cov(year, alpha, rho);
    for (n in 1:N) K[n,n] += 1e-9;
    L_K = cholesky_decompose(K);
    f = beta_0 + beta_1 * year + L_K * f_raw;
  }
}
model {
  beta_0 ~ normal(log(100), 1);
  beta_1 ~ normal(0.5, 0.5);
  alpha ~ normal(0, 1);
  rho ~ inv_gamma(5, 5);
  phi ~ gamma(2, 0.1);
  f_raw ~ std_normal();

  C ~ neg_binomial_2_log(f, phi);
}
generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | f[t], phi);
    C_rep[t] = neg_binomial_2_log_rng(f[t], phi);
  }
}
```

---

**Document Prepared By:** Model Designer 2 (Temporal Structure Specialist)
**Date:** 2025-10-29
**Status:** Ready for Implementation
**Next Step:** Implement models in Stan, fit to data, execute falsification tests
