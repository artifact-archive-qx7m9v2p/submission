# Model Comparison Matrix - Designer 2
## Side-by-Side Comparison of Proposed Models

---

## Model Specifications

| Feature | Model 1: State-Space | Model 2: Changepoint | Model 3: Gaussian Process |
|---------|---------------------|---------------------|---------------------------|
| **Core Mechanism** | Latent random walk + drift | Piecewise linear with break | Nonparametric smooth function |
| **Likelihood** | C_t ~ NegBin(exp(η_t), φ) | C_t ~ NegBin(μ_t, φ) | C_t ~ NegBin(exp(f_t), φ) |
| **Mean Structure** | η_t = η_{t-1} + δ + β·year_t + ε_t | log(μ_t) = β_0 + β_1·year + β_2·I(year≥τ) + β_3·(year-τ)·I(year≥τ) | log(μ_t) = f(year_t), f ~ GP(m, k) |
| **Key Parameters** | δ (drift), σ_η (innovation SD), φ | β_0, β_1, β_2, β_3, τ, φ | α (GP variance), ℓ (lengthscale), φ |
| **# Parameters** | 4 + 40 latent states | 6 | 4 + N GP values |
| **Computational Cost** | Medium (MCMC correlations) | Low-Medium (indicator functions) | Medium (matrix inversions) |

---

## Theoretical Motivation

| Aspect | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| **EDA Evidence** | ACF(1)=0.989, R²(lag-1)=0.977, slope≈1.011 (near random walk) | CUSUM minimum at year=0.3, 4.5× mean jump, t-test p<0.0001 | Polynomial R² increases with degree (0.885→0.990), no single form is best |
| **Data Generation Story** | Process evolves gradually with small random shocks around systematic drift | System underwent structural break, fundamentally different before/after | Growth is smooth but complex, neither linear nor discrete jump |
| **Temporal Structure** | Explicit AR(1) on latent state | Implicit via trend, not modeled directly | Implicit via GP correlation structure |
| **Growth Pattern** | Exponential (constant drift in log-space) | Piecewise linear in log-space | Flexible, data-driven |

---

## Prior Specifications

| Parameter | Model 1 | Model 2 | Model 3 |
|-----------|---------|---------|---------|
| **Intercept** | η_1 ~ N(log(30), 1) | β_0 ~ N(log(100), 1) | β_0 ~ N(log(100), 1) |
| **Trend** | β ~ N(0, 0.5) | β_1 ~ N(0.5, 0.5) | β_1 ~ N(0.5, 0.5) |
| **Drift/Slope** | δ ~ N(0.05, 0.1) | β_2 ~ N(0, 1), β_3 ~ N(0, 0.5) | - |
| **Changepoint** | - | τ ~ U(-1.5, 1.5) | - |
| **Innovation SD** | σ_η ~ HalfNormal(0.5) | - | - |
| **GP Hyperparams** | - | - | α ~ HalfNormal(1), ℓ ~ InvGamma(5,5) |
| **Dispersion** | φ ~ Gamma(2, 0.1) | φ ~ Gamma(2, 0.1) | φ ~ Gamma(2, 0.1) |

**Prior Philosophy:**
- **Model 1:** Informative on drift (expect ~5% growth), weakly informative elsewhere
- **Model 2:** Weakly informative, let data identify changepoint location
- **Model 3:** Weakly informative, encourage smooth functions (large ℓ)

---

## Expected Posterior Values

| Parameter | Model 1 Expected | Model 2 Expected | Model 3 Expected |
|-----------|-----------------|-----------------|-----------------|
| **Growth Rate** | δ = 0.06 [0.04, 0.08] | β_1 = 0.4 [0.2, 0.6] pre-change | Implicit in f, ~0.5-0.8 average |
| **Dispersion** | φ = 15 [10, 22] | φ = 18 [12, 28] | φ = 12 [8, 20] |
| **Temporal Param** | σ_η = 0.08 [0.05, 0.12] | τ = 0.30 [0.15, 0.45] | ℓ = 0.6 [0.3, 1.0] |
| **Effect Size** | Small innovations relative to drift | β_2 = 0.8 [0.4, 1.2], β_3 = 0.5 [0.2, 0.8] | α = 0.5 [0.2, 0.8] |

**Key Insight:**
- **Model 1:** Small σ_η means state evolution is smooth, predictable
- **Model 2:** Large β_2, β_3 means regime shift is substantial
- **Model 3:** Moderate ℓ means neither too wiggly nor too smooth

---

## Falsification Criteria

| Scenario | Model 1 Fails If... | Model 2 Fails If... | Model 3 Fails If... |
|----------|-------------------|-------------------|-------------------|
| **Parameter Space** | σ²_η ≈ Var[log(C)] (no benefit from latent state) | τ posterior uniform (no preferred location) | ℓ → ∞ (equivalent to linear) |
| **Residual Checks** | ACF(1) > 0.5 (temporal structure not captured) | ACF(1) > 0.7 (changepoint doesn't explain dynamics) | ACF(1) > 0.5 (GP doesn't capture correlation) |
| **Predictive** | One-step-ahead coverage < 75% | LOO-ELPD 10+ worse than smooth alternative | Parametric model within ΔLOO < 5 (Occam's razor) |
| **Computational** | Divergences > 5% despite adapt_delta=0.99 | β_2 and β_3 both include 0 (no regime shift) | Posterior shows discontinuities (changepoint better) |
| **Prior-Posterior** | Overlap > 80% for δ or σ_η | Overlap > 80% for τ | Overlap > 80% for ℓ |

---

## Prediction Strategy

| Aspect | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| **In-Sample** | One-step: η_{t+1} ~ η_t + δ + β·year + ε | Direct: μ_{t+1} from trend formula | Direct: f(year_{t+1}) from GP posterior |
| **Short-term (1-5 steps)** | Excellent (leverages AR structure) | Good if no new changepoint | Good (smooth interpolation) |
| **Long-term (>5 steps)** | Moderate (uncertainty accumulates) | Good within regime, poor if change likely | Moderate (GP extrapolates conservatively) |
| **Extrapolation** | Linear in log-space (exponential in counts) | Linear in log-space per regime | Regresses to prior mean (wide intervals) |
| **Uncertainty** | Grows with sqrt(t) from innovations | Constant within regime | Large outside training range |

**Best for:**
- **Model 1:** Short-term forecasting with autocorrelation
- **Model 2:** Long-term if regime is stable, interpretable scenarios
- **Model 3:** Uncertainty quantification, robust to functional form

---

## Computational Considerations

| Challenge | Model 1 | Model 2 | Model 3 |
|-----------|---------|---------|---------|
| **MCMC Mixing** | Slow (high posterior correlations among η_t) | Fast (6 parameters) | Medium (N×N matrix operations) |
| **Divergences** | Possible (non-centered helps) | Possible at changepoint discontinuity | Rare (smooth geometry) |
| **Memory** | O(N) for latent states | O(1) | O(N²) for covariance matrix |
| **Runtime** | 5-10 min | 2-5 min | 3-8 min |
| **Convergence** | May need 10k iterations | Usually 4k sufficient | Usually 4k sufficient |

**Solutions:**
- **Model 1:** Non-centered η_t, adapt_delta=0.95
- **Model 2:** Informative prior on τ, adapt_delta=0.95
- **Model 3:** Add jitter (1e-9) to diagonal, Cholesky decomposition

---

## Model Selection Decision Tree

```
Step 1: Compute LOO-ELPD for all three models
│
├─ If max(ΔLOO) > 10: Clear winner
│  ├─ Check: All diagnostics pass? (R-hat, ESS, ACF, coverage)
│  │  ├─ YES → Accept winner, report parameters
│  │  └─ NO → Investigate diagnostic failures, try variants
│  │
├─ If max(ΔLOO) < 4: Models equivalent
│  ├─ Use Bayesian Model Averaging
│  ├─ Report: "Multiple mechanisms consistent with data"
│  └─ Emphasize prediction over mechanism
│  │
└─ If all LOO-ELPD < -180 or all diagnostics fail:
   ├─ RED FLAG: None of these models are appropriate
   ├─ Try model variants (see proposed_models.md)
   ├─ If still fail: PIVOT to new model classes
   └─ Possible causes: Wrong distribution, wrong temporal structure, unobserved covariates
```

---

## Stress Test Matrix

| Test | Model 1 Expected | Model 2 Expected | Model 3 Expected |
|------|-----------------|-----------------|-----------------|
| **Extrapolation (year > 1.0)** | Good short-term, deteriorates long-term | Good if no new changepoint, else poor | Conservative (wide intervals) |
| **Jackknife (remove changepoint region)** | Robust (doesn't rely on specific points) | Unstable (τ posterior shifts) | Robust (smooth across gap) |
| **Simulate from exponential+i.i.d.** | May overfit temporal structure | Good fit | Good fit |
| **Simulate from AR(1) no trend** | Good fit | Poor (forces trend) | Poor (forces smooth function) |
| **Simulate from discrete changepoint** | Poor (smooth over break) | Good fit | Moderate (smooth transition) |
| **Prior sensitivity** | Moderate (δ weakly identified) | High for τ, low for slopes | High for ℓ (lengthscale crucial) |

---

## Interpretability Comparison

| Question | Model 1 Answer | Model 2 Answer | Model 3 Answer |
|----------|---------------|---------------|---------------|
| **"What is the growth rate?"** | δ = 6% per period in log-space | β_1 pre-change, β_1+β_3 post-change | Not directly estimable (need to compute from f) |
| **"When did the system change?"** | Gradual evolution, no discrete change | τ ≈ 0.3 (95% CI: [0.15, 0.45]) | No discrete change, smooth acceleration |
| **"Is growth accelerating?"** | Yes if β > 0, but drift is constant | Yes if β_3 > 0 (slope change) | Yes if f is convex (visual inspection) |
| **"What happens in 5 years?"** | exp(η_T + 5·δ) with innovation uncertainty | Depends on regime, extrapolate linearly | Wide intervals, regresses to mean |
| **"Why is variance so high?"** | Temporal correlation (ε_t innovations are small) | Overdispersion φ + regime heterogeneity | Overdispersion φ + functional uncertainty |

**Most Interpretable:** Model 2 (clear regime parameters)
**Most Predictive:** Model 1 (leverages autocorrelation)
**Most Honest:** Model 3 (admits functional uncertainty)

---

## Recommendations by Goal

### If Primary Goal is Prediction
**Rank:** Model 1 > Model 3 > Model 2
- State-Space exploits AR structure for short-term forecasting
- GP provides robust uncertainty for medium-term
- Changepoint risky if another break possible

### If Primary Goal is Mechanism Understanding
**Rank:** Model 2 > Model 1 > Model 3
- Changepoint tests explicit regime shift hypothesis
- State-Space quantifies drift and innovation variance
- GP is black box (less interpretable)

### If Primary Goal is Robustness
**Rank:** Model 3 > Model 1 > Model 2
- GP makes fewest assumptions
- State-Space robust to functional form
- Changepoint assumes discrete break (may be wrong)

### If Computational Budget is Limited
**Rank:** Model 2 > Model 3 > Model 1
- Changepoint is fastest (6 parameters)
- GP is tractable (n=40 is small)
- State-Space has mixing issues

---

## What Each Model Will Teach Us

### Model 1 Success Teaches Us:
- Data are primarily a **time series**, not cross-sectional with trend
- Autocorrelation is the dominant signal
- Most "overdispersion" is temporal correlation
- Short-term predictions are possible with high accuracy

### Model 2 Success Teaches Us:
- A **discrete event** occurred around year ≈ 0.3
- System has two distinct regimes with different dynamics
- Future depends on whether another break is possible
- Need to investigate what caused the change

### Model 3 Success Teaches Us:
- Truth is **more complex** than simple parametric forms
- Growth is smooth but nonlinear (maybe logistic, Gompertz, etc.)
- Should inspect GP function and try to match to known curves
- Functional form uncertainty is substantial

### All Models Fail Teaches Us:
- Data generation process is **stranger than we thought**
- Our model classes are inadequate
- May need: different distributions, more complex temporal structure, or unobserved covariates
- Negative result is still scientifically valuable

---

## Summary: Which Model Will Win?

### My Prediction (60% confidence): Model 1 (State-Space)
**Reason:** The ACF(1) = 0.989 is the strongest signal in the data. This screams "autoregressive process." The lag-1 R² of 0.977 means C_t is almost perfectly predicted by C_{t-1}. This is classic time series behavior.

**Expected findings:**
- δ ≈ 0.06 (6% growth per period in log-space)
- σ_η ≈ 0.08 (small innovations, smooth evolution)
- φ ≈ 15 (moderate overdispersion after accounting for temporal correlation)
- One-step-ahead MAPE < 8%

### Alternative Scenario (25% confidence): Models Equivalent
**Reason:** With n=40, we may lack power to distinguish these mechanisms. All three might fit adequately but not definitively better than others.

**Expected findings:**
- All LOO-ELPD within 4 of each other
- All pass basic diagnostics (ACF < 0.3, coverage 75-85%)
- Use BMA for predictions, report mechanistic uncertainty

### Surprise Scenario (15% confidence): Model 2 or 3 Wins
**If Model 2 wins:** The changepoint is real and substantial. Need to investigate domain context.
**If Model 3 wins:** Truth is more complex than both random walk and changepoint. Inspect GP function carefully.

---

**Prepared by:** Designer 2 (Temporal Structure Specialist)
**Date:** 2025-10-29
**Full Details:** See `/workspace/experiments/designer_2/proposed_models.md`
