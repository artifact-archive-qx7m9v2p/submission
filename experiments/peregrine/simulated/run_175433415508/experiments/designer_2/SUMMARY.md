# Designer 2 Summary: Three Temporal Models

## Visual Overview

```
                    HIGH AUTOCORRELATION (ACF = 0.971)
                                  |
                                  |
              Is this REAL or TREND ARTIFACT?
                                  |
                    +-------------+-------------+
                    |                           |
                  REAL                     ARTIFACT
            (needs correlation)        (just smooth trend)
                    |                           |
                    |                           |
         +----------+----------+                |
         |          |          |                |
       AR(1)       GP        RW          Quadratic + iid
         |          |          |                |
         |          |          |                |
    Stationary  Flexible  Non-stat        Simple model
    ρ ∈ [0,1]   smooth   ρ = 1.0        (baseline)
```

---

## Three Models at a Glance

### Model 1: NB-AR1 (Autoregressive)

**Formula**:
```
C_t ~ NegBin(μ_t, α)
log(μ_t) = η_t
η_t = β₀ + β₁·year_t + ε_t
ε_t = ρ·ε_{t-1} + ν_t
```

**Key Parameters**:
- ρ: Autoregressive coefficient (0 to 1)
- σ_η: Innovation standard deviation

**Prior for ρ**: Beta(20, 2) → E[ρ] = 0.91

**Assumes**:
- Stationary process (mean-reverting)
- Exponential ACF decay: ρ^k

**Reject if**: ρ → 1 (boundary), residual ACF high, divergences

**Best for**: Medium-high correlation (0.7-0.95), stationary dynamics

---

### Model 2: NB-GP (Gaussian Process)

**Formula**:
```
C_t ~ NegBin(μ_t, α)
log(μ_t) = f(t)
f ~ GP(β₀ + β₁·year, K)
K(t,t') = η²·exp(-(t-t')²/2ℓ²)
```

**Key Parameters**:
- ℓ: Lengthscale (correlation range)
- η: Amplitude (deviation from trend)

**Prior for ℓ**: InverseGamma(5, 5) → E[ℓ] = 1.25

**Assumes**:
- Smooth function (infinitely differentiable)
- Non-parametric trend
- Gaussian correlation structure

**Reject if**: ℓ → ∞ (constant), Cholesky fails, no better than polynomial

**Best for**: Unknown functional form, smooth non-linear trends

---

### Model 3: NB-RW (Random Walk)

**Formula**:
```
C_t ~ NegBin(μ_t, α)
log(μ_t) = β₀ + β₁·year_t + ξ_t
ξ_t = ξ_{t-1} + ω_t
ω_t ~ N(0, σ_ω)
```

**Key Parameters**:
- σ_ω: Innovation standard deviation
- Cumulative variance = t·σ_ω²

**Prior for σ_ω**: Exponential(10) → E[σ_ω] = 0.1

**Assumes**:
- Non-stationary (no mean reversion)
- Perfect memory (ρ = 1)
- Variance grows without bound

**Reject if**: σ_ω → 0 (deterministic), first-differences show ACF

**Best for**: Near unit-root (ρ ≈ 1), trending non-stationary data

---

## Key Differences

| Property | AR(1) | GP | RW |
|----------|-------|----|----|
| **Stationarity** | Yes | Yes | **No** |
| **Memory** | Exponential decay | Gaussian decay | **Infinite** |
| **Complexity** | Low (1 param) | High (2-3 params) | Low (1 param) |
| **ρ assumption** | ρ < 1 | - | **ρ = 1** |
| **Computation** | O(N) | **O(N³)** | O(N) |
| **Identifiability** | Good | Medium | Good |
| **Trend flexibility** | Fixed (linear) | **Free** | Fixed (linear) |

---

## Decision Flowchart

```
1. Fit all 3 models + baseline (independent errors)
   |
   v
2. Check convergence (Rhat, ESS, divergences)
   |
   +-- FAIL → Debug, reparameterize, try again
   |
   +-- PASS
       |
       v
3. Compare ELPD: Temporal vs Baseline
   |
   +-- ΔELPD < 2 → HIGH ACF IS ARTIFACT
   |               Use simple model
   |
   +-- ΔELPD > 5 → GENUINE CORRELATION
       |
       v
4. Compare temporal models
   |
   +-- All within ΔELPD < 2 → INDISTINGUISHABLE
   |                          Use model averaging
   |
   +-- Clear winner (ΔELPD > 5) → USE BEST MODEL
       |
       v
5. Posterior predictive checks
   |
   +-- ACF captured? → YES: Success
   |
   +-- ACF missed? → NO: Try alternatives
       |                (higher-order, structural break)
       |
       v
6. Report selected model with uncertainty
```

---

## Critical Parameters and Interpretation

### AR(1) Correlation (ρ)

| Posterior | Interpretation | Action |
|-----------|----------------|--------|
| 0.70-0.85 | Moderate correlation | AR(1) is appropriate |
| 0.85-0.95 | High correlation | AR(1) works, check residuals |
| 0.95-0.98 | Very high correlation | Consider RW (near unit root) |
| > 0.98 | At boundary | **Red flag**: Switch to RW |

### GP Lengthscale (ℓ)

| Posterior | Interpretation | Action |
|-----------|----------------|--------|
| 0.3-0.8 | Short correlation | Wiggly function, genuine GP |
| 0.8-2.0 | Medium correlation | Smooth trend, reasonable |
| 2.0-5.0 | Long correlation | Nearly constant, check if needed |
| > 5.0 | Very long | **Red flag**: Just a smooth trend |

### RW Innovation (σ_ω)

| Posterior | Interpretation | Action |
|-----------|----------------|--------|
| 0.05-0.15 | Small steps | Smooth random walk |
| 0.15-0.30 | Medium steps | Moderate volatility |
| 0.30-0.50 | Large steps | High volatility, check if realistic |
| > 0.50 | Very large | **Red flag**: Fitting noise? |

---

## Posterior Predictive Checks

### Must-Do Checks

1. **Time series overlay**
   - Plot: Observed vs 90% posterior predictive interval
   - Should: Most data within interval, smooth fits

2. **ACF comparison**
   - Plot: Observed ACF vs posterior predictive ACF distribution
   - Should: Observed within 90% interval for lags 1-5

3. **Residual ACF**
   - Plot: ACF of Pearson residuals
   - Should: All lags < 0.2 (white noise)

4. **Rootogram**
   - Plot: Observed vs predicted count distribution
   - Should: Bars near zero (good fit)

---

## Prior Sensitivity Tests

### Standard Priors (Baseline)

```
ρ ~ Beta(20, 2)           # E = 0.91, informed by data
ℓ ~ InvGamma(5, 5)        # E = 1.25, long correlation
σ_ω ~ Exponential(10)     # E = 0.1, small innovations
```

### Skeptical Priors (Weak Correlation)

```
ρ ~ Beta(5, 5)            # E = 0.5, agnostic
ℓ ~ InvGamma(2, 1)        # E = 1.0, shorter correlation
σ_ω ~ Exponential(20)     # E = 0.05, very small innovations
```

### Informed Priors (Strong Correlation)

```
ρ ~ Beta(30, 2)           # E = 0.94, strong prior
ℓ ~ InvGamma(10, 10)      # E = 1.11, tight around ACF=0.97
σ_ω ~ Exponential(5)      # E = 0.2, larger innovations
```

**Test**: If posterior changes dramatically → data is weak, identifiability issue

---

## Computational Notes

### Expected Runtime (4 chains, 2000 iterations)

- **AR(1)**: 2-3 minutes
- **GP**: 5-8 minutes (slowest, O(N³))
- **RW**: 1-2 minutes (fastest)

### Sampling Settings

**Conservative** (use initially):
```python
adapt_delta=0.95
max_treedepth=12
iter_warmup=1000
iter_sampling=2000
```

**Aggressive** (if no divergences):
```python
adapt_delta=0.90
max_treedepth=10
iter_warmup=800
iter_sampling=1500
```

### Common Issues

| Issue | Model | Solution |
|-------|-------|----------|
| Divergences | AR(1) | Increase adapt_delta, check ρ → 1 |
| Cholesky fail | GP | Add jitter, stronger ℓ prior |
| Low ESS | AR(1), GP | Non-centered parameterization |
| Rhat > 1.05 | All | Longer warmup, check identifiability |

---

## Success Criteria

### Convergence (REQUIRED)
- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 (bulk and tail)
- [ ] < 0.1% divergent transitions
- [ ] Trace plots show good mixing

### Fit Quality (REQUIRED)
- [ ] Posterior predictive ACF captures observed ACF
- [ ] Residual ACF < 0.2 at all lags
- [ ] 90% of observations within posterior predictive interval
- [ ] Pareto-k < 0.7 for > 80% of observations

### Model Comparison (REQUIRED)
- [ ] PSIS-LOO computed successfully
- [ ] Standard errors reasonable (< 30% of ELPD)
- [ ] Clear ranking OR model averaging justified
- [ ] Decision documented with evidence

---

## Expected Outcomes

### Likely Scenarios (ordered by probability)

1. **AR(1) wins** (40% probability)
   - Most standard approach
   - Should work if ρ ∈ [0.85, 0.95]
   - Report posterior ρ with interpretation

2. **Models indistinguishable** (30% probability)
   - n=40 too small to distinguish
   - Use model averaging (stacking weights)
   - Emphasize structural uncertainty

3. **Temporal structure unnecessary** (20% probability)
   - High ACF is trend artifact
   - Use baseline (independent errors)
   - Explain why ACF was misleading

4. **RW wins** (7% probability)
   - ρ ≈ 1 (near unit root)
   - Non-stationary process
   - Discuss implications for forecasting

5. **GP wins** (3% probability)
   - Unusual for count data
   - Likely means non-linear trend
   - Check if quadratic trend sufficient

---

## Key Insights for Implementer

### 1. The Identifiability Challenge

With ACF = 0.971, it's **very hard** to separate:
- Smooth deterministic trend (β₀ + β₁·t + β₂·t²)
- High stochastic correlation (ρ ≈ 0.97)

**These are nearly equivalent for predictions!**

### 2. The n=40 Constraint

- **Sufficient** for AR(1) (only 1 correlation parameter)
- **Marginal** for GP (2-3 hyperparameters)
- **Marginal** for RW (cumulative variance grows)

**Don't expect to distinguish subtle differences**

### 3. The Baseline is Critical

**Always compare to**: NB with independent errors

If temporal models don't improve ELPD > 5:
→ **Correlation is an artifact of smooth trend**

This is a scientifically valid conclusion!

### 4. Computational Red Flags

| Warning Sign | Meaning | Action |
|--------------|---------|--------|
| ρ → 1.0 | Boundary issue | Try RW instead |
| ℓ → ∞ | GP too smooth | Try polynomial |
| σ_ω → 0 | RW unnecessary | Try simpler model |
| ESS < 100 | Unidentified | Stronger priors |

---

## Final Checklist

Before submitting results:

- [ ] All models converged (Rhat, ESS)
- [ ] Posterior predictive checks performed
- [ ] PSIS-LOO comparison done
- [ ] Decision documented with evidence
- [ ] Sensitivity to priors checked
- [ ] Limitations acknowledged (n=40, structural uncertainty)
- [ ] Model code and data archived
- [ ] Figures saved (time series, ACF, diagnostics)

---

## File Reference

- **Theory**: `proposed_models.md` (35 KB)
- **Implementation**: `stan_implementation_guide.md` (21 KB)
- **Decision**: `model_selection_framework.md` (16 KB)
- **Navigation**: `README.md` (13 KB)
- **Summary**: `SUMMARY.md` (this file)

**Start with**: README.md → proposed_models.md → implementation

---

## Contact

**Designer**: Model Designer 2 (Temporal Correlation Specialist)

**Strength**: Explicit temporal dependence modeling

**Limitation**: n=40 limits complexity we can reliably estimate

**Status**: Ready for implementation

**Date**: 2025-10-29

---

**Remember**: The goal is **finding truth**, not forcing a complex model. If models are indistinguishable or correlation is unnecessary, **say so** - that's a valid scientific conclusion.
