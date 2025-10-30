# Visual Model Comparison Summary

**Three Time-Series Models for Count Data with Regime Change**

---

## At a Glance: Which Model for What?

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA CHARACTERISTICS (from EDA)                                │
├─────────────────────────────────────────────────────────────────┤
│  • Strong autocorrelation: ACF(1) = 0.944                       │
│  • Structural break at t=17: 730% slope increase                │
│  • Extreme overdispersion: variance/mean = 67.99                │
│  • Growth: 745% total over 40 observations                      │
└─────────────────────────────────────────────────────────────────┘

         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │   Model 1    │  │   Model 2    │  │   Model 3    │
         │     DLM      │  │    NB-AR     │  │      GP      │
         │  (Dynamic)   │  │ (Obs-Driven) │  │  (Flexible)  │
         └──────────────┘  └──────────────┘  └──────────────┘
```

---

## Conceptual Differences

### Model 1: Dynamic Linear Model (State-Space)
**Mental Model**: "There's a hidden process (velocity) that smoothly evolves, but it gets shocked at the regime change"

```
Time:      t-1          t           t+1

Velocity:  δ_{t-1} --> δ_t -----> δ_{t+1}
                        |            |
                    (AR process) (+ regime shift)
                        |            |
Log-rate:  η_{t-1} --> η_t -----> η_{t+1}
                        |            |
                    (integrate velocity)
                        |            |
Observed:  C_{t-1}     C_t         C_{t+1}
                    (NB noise)
```

**Key Feature**: Two-layer hierarchy (velocity -> rate -> count)
**Autocorrelation Source**: Velocity persists via AR(1) process (φ parameter)
**Regime Change**: Discrete jump in velocity drift (Δδ parameter)

---

### Model 2: Negative Binomial AR (Observation-Driven)
**Mental Model**: "Today's count depends directly on yesterday's count, plus a time trend that changes slope"

```
Time:      t-1          t           t+1

                    ┌───────┐
Observed:  C_{t-1} ─┤       ├──> μ_t ────> C_t ──────> C_{t+1}
                    │ AR(1) │     ↓              ↓
                    └───────┘   (NB)           (NB)
                        ↑
                   (γ × log(C_{t-1}))
                        +
             (polynomial trend + regime shift)
```

**Key Feature**: Direct feedback from observed counts
**Autocorrelation Source**: Explicit lagged term (γ parameter)
**Regime Change**: Piecewise linear trend (β_3 parameter)

---

### Model 3: Gaussian Process (Parameter-Driven)
**Mental Model**: "The log-rate is some smooth unknown function of time, and the GP learns its shape"

```
Time:      t-1          t           t+1

GP latent: f(t-1) ---- f(t) ----- f(t+1)
             ↓          ↓            ↓
           (correlated via kernel)
             ↓          ↓            ↓
Log-rate:  η_{t-1}    η_t        η_{t+1}
             ↓          ↓            ↓
Observed:  C_{t-1}    C_t        C_{t+1}
                    (NB noise)
```

**Key Feature**: Non-parametric smooth function
**Autocorrelation Source**: Exponential covariance kernel (length-scale parameter)
**Regime Change**: Emergent from data (not explicitly modeled)

---

## Mathematical Equations (Side-by-Side)

| Component | Model 1 (DLM) | Model 2 (NB-AR) | Model 3 (GP) |
|-----------|---------------|-----------------|--------------|
| **Observation** | C_t ~ NB(μ_t, α) | C_t ~ NB(μ_t, α_t) | C_t ~ NB(μ_t, α) |
| **Link** | log(μ_t) = η_t | log(μ_t) = β_0 + β_1·year_t + β_2·year_t² + γ·log(C_{t-1}+1) + regime_term | log(μ_t) = β_0 + f(year_t) |
| **Dynamics** | η_t = η_{t-1} + δ_t + ν_t<br>δ_t = φ·δ_{t-1} + I(t>τ)·Δδ + ω_t | regime_term = I(t>τ)·β_3·(year_t - year_τ) | f ~ GP(0, K)<br>K(s,t) = σ_f²·exp(-ρ·(s-t)²) |
| **Parameters** | η_0, δ_0, φ, Δδ, τ, σ_η, σ_δ, α | β_0, β_1, β_2, β_3, γ, τ, α_0, α_1 | β_0, σ_f, l, σ_n, α |
| **Latent Dimension** | 2N (η_t, δ_t for all t) | 0 (no latent states) | N (f_t for all t) |

---

## Strengths and Weaknesses

### Model 1: Dynamic Linear Model

**Strengths**:
- Most theoretically aligned with EDA findings
- Separates trend dynamics from observation noise
- Natural interpretation (velocity = growth rate)
- Excellent for prediction (parametric extrapolation)
- Handles regime change and autocorrelation in unified framework

**Weaknesses**:
- Most complex to implement (state-space requires care)
- Computational challenges (non-centered parameterization essential)
- Discrete changepoint parameter difficult in Stan
- High-dimensional (87 parameters for N=40)
- Assumes specific dynamic structure (may be too rigid)

**Best for**: Scientific understanding of dynamic process

---

### Model 2: Negative Binomial AR

**Strengths**:
- Simplest implementation (standard GLM + lagged term)
- Fast to fit (10-20 min)
- Familiar to practitioners (extension of GLM)
- Easy to interpret AR coefficient
- Time-varying dispersion is flexible

**Weaknesses**:
- Lagged observation may not capture true autocorrelation
- First-observation boundary condition is artificial
- γ and β_2 may be confounded (identifiability issue)
- Observation-driven dynamics less theoretically motivated
- Extrapolation depends on last observed count (unstable)

**Best for**: Quick baseline, computational efficiency

---

### Model 3: Gaussian Process

**Strengths**:
- Maximum flexibility (no parametric assumptions)
- Discovers pattern from data (including regime change)
- Principled uncertainty quantification
- Excellent for checking if discrete changepoint is necessary
- Natural autocorrelation structure

**Weaknesses**:
- Computationally expensive (O(N³) for Cholesky)
- Black box (hard to interpret what GP learned)
- May over-smooth structural break
- Poor extrapolation (uncertainty explodes outside data range)
- Hyperparameter posteriors can be multimodal

**Best for**: Model checking, flexible exploratory analysis

---

## Falsification Criteria (What Would Make Me Reject Each Model?)

### Model 1 (DLM)

| Test | Threshold | Interpretation |
|------|-----------|----------------|
| φ posterior near 0 | Mean(φ) < 0.2 | Velocity AR is unnecessary, simplify to random walk |
| φ posterior near 1 | P(φ > 0.98) > 0.5 | Non-stationarity, model misspecified |
| Noise ratio | P(σ_η > σ_δ) > 0.9 | Observation noise dominates, state-space not needed |
| Changepoint diffuse | SD(τ) > 5 | No clear regime change, use smooth model |
| Residual ACF | ACF(1) > 0.5 | Haven't captured autocorrelation |

**Verdict**: If 2+ tests fail, abandon Model 1

---

### Model 2 (NB-AR)

| Test | Threshold | Interpretation |
|------|-----------|----------------|
| γ posterior near 0 | Mean(γ) < 0.1 | AR term unnecessary, autocorrelation is spurious |
| Identifiability | |corr(γ, β_2)| > 0.7 | Can't separate AR from quadratic trend |
| Residual ACF | ACF(1) > 0.5 | AR(1) insufficient, need ARMA |
| Regime shift | 0 ∈ 90% CI for β_3 | Changepoint not needed |
| First-obs uncertainty | SD(μ_1) / SD(μ_20) > 3 | Boundary condition unstable |

**Verdict**: If 2+ tests fail, abandon Model 2

---

### Model 3 (GP)

| Test | Threshold | Interpretation |
|------|-----------|----------------|
| Length-scale collapse | Mean(l) < 0.3 | Overfitting to noise, not smooth |
| Length-scale explosion | Mean(l) > 5 | GP degenerates to linear trend |
| Nugget dominance | Mean(σ_n / σ_f) > 0.5 | Not capturing structure |
| Residual ACF | ACF(1) > 0.5 | Covariance structure inadequate |
| No visible break | Visual inspection | GP over-smoothed regime change |

**Verdict**: If 2+ tests fail, abandon Model 3

---

## Expected Parameter Values (Predictions)

Based on EDA, here's what I expect posterior distributions to look like:

### Model 1 (DLM)

```
Parameter      Prior              Expected Posterior        Interpretation
─────────────────────────────────────────────────────────────────────────
η_0            N(3.0, 0.5)        N(3.1, 0.2)              Starting log-rate ≈ log(22)
δ_0            N(0.3, 0.2)        N(0.25, 0.15)            Initial gentle velocity
φ              Beta(8, 2)         Beta-like, mean ≈ 0.82   Strong velocity persistence
Δδ             N(0.8, 0.3)        N(0.75, 0.2)             Large regime shift
τ              Uniform(10,30)     Concentrated at 17       Changepoint confirmed
σ_η            Exp(2)             Exp-like, mean ≈ 0.12    Small observation noise
σ_δ            Exp(5)             Exp-like, mean ≈ 0.08    Smooth velocity changes
α              Gamma(2, 1)        Gamma-like, mean ≈ 0.58  Match EDA estimate
```

### Model 2 (NB-AR)

```
Parameter      Prior              Expected Posterior        Interpretation
─────────────────────────────────────────────────────────────────────────
β_0            N(4.0, 0.5)        N(3.9, 0.3)              Baseline log-rate
β_1            N(0.5, 0.3)        N(0.45, 0.2)             Pre-break slope
β_2            N(0.2, 0.2)        N(0.15, 0.1)             Quadratic curvature
β_3            N(0.8, 0.4)        N(0.7, 0.3)              Post-break acceleration
γ              Beta(6, 2)         Beta-like, mean ≈ 0.65   Moderate AR dependence
τ              Uniform(10,30)     Concentrated at 17       Changepoint confirmed
α_0            N(log(0.6), 0.5)   N(-0.5, 0.3)             Baseline dispersion
α_1            N(0, 0.3)          N(0.1, 0.2)              Mild dispersion trend
```

### Model 3 (GP)

```
Parameter      Prior              Expected Posterior        Interpretation
─────────────────────────────────────────────────────────────────────────
β_0            N(4.3, 0.5)        N(4.2, 0.3)              Global intercept
σ_f            Exp(1)             Exp-like, mean ≈ 1.2     Signal strength
l              InvGamma(5,5)      IG-like, mean ≈ 1.5      Moderate smoothness
σ_n            Exp(5)             Exp-like, mean ≈ 0.15    Small nugget
α              Gamma(2, 1)        Gamma-like, mean ≈ 0.6   Match EDA estimate
```

**Note**: These are PREDICTIONS. If posteriors differ substantially, it means the model has learned something unexpected from the data.

---

## LOO Comparison (Predicted Ranking)

Based on EDA evidence and model alignment:

```
Rank  Model              Expected ELPD_loo     SE      Reason
────────────────────────────────────────────────────────────────
1     Model 1 (DLM)      -165                  8       Best captures both regime change and dynamics
2     Model 3 (GP)       -168                  9       Flexible but may over-smooth break
3     Model 2 (NB-AR)    -172                  10      AR term may not fully explain ACF(1)=0.944

ΔLOO (Model 1 vs 3):  3 ± 5    --> Inconclusive, may need model averaging
ΔLOO (Model 1 vs 2):  7 ± 6    --> Model 1 preferred
```

**If predictions are wrong**: That's valuable information! It means one of my assumptions about the data generation process is incorrect.

---

## Computational Comparison

| Aspect | Model 1 (DLM) | Model 2 (NB-AR) | Model 3 (GP) |
|--------|---------------|-----------------|--------------|
| **Parameters** | 87 (8 + 2N) | 8 | 44 (4 + N) |
| **Runtime** | 15-30 min | 10-20 min | 20-40 min |
| **Divergences (expected)** | Medium (5-10%) | Low (1-3%) | High (10-20%) |
| **Memory** | Medium | Low | High |
| **Compilation** | Fast | Fast | Slow |
| **Ease of implementation** | Hard | Easy | Medium |
| **Parallelizable** | No (sequential state evolution) | No (lagged dependence) | Yes (GP can use multi-core) |

**Recommendation**: Start with Model 2 to verify basic structure, then move to Model 1 (primary) and Model 3 (validation).

---

## Posterior Predictive Checks (What to Look For)

### Visual Checks

1. **Trajectory Plot**:
   - Overlay 100 posterior draws of μ_t on observed counts
   - Check if draws capture both smooth trend and abrupt change
   - Look for underfitting (draws too smooth) or overfitting (draws too variable)

2. **Residual ACF**:
   - Plot ACF of Pearson residuals for lags 1-10
   - All models should reduce ACF(1) from 0.944 to < 0.4
   - If residual ACF shows pattern, model is incomplete

3. **Growth Rate Distribution**:
   - Compute ΔC_t / C_{t-1} for observed data
   - Generate same statistic from posterior predictive
   - Should see bimodal distribution (pre/post regime change)

4. **Variance-Mean Relationship**:
   - Bin observations by μ_t (predicted mean)
   - Plot observed variance vs mean in each bin
   - Overlay posterior predictive variance-mean curve
   - Should match observed power law (var ∝ mean^1.67)

### Numerical Checks

Compute these test statistics on observed data and posterior predictive samples:

```
Statistic               Observed    Model 1    Model 2    Model 3
──────────────────────────────────────────────────────────────────
Mean count              109.45      ?          ?          ?
Variance                7431        ?          ?          ?
Var/Mean ratio          67.99       ?          ?          ?
ACF(1)                  0.944       ?          ?          ?
Max count               272         ?          ?          ?
Growth rate (total)     745%        ?          ?          ?
Chow test p-value       <0.001      ?          ?          ?
```

**Good fit**: Observed value falls within [5th, 95th] percentile of posterior predictive distribution for ALL statistics.

**Poor fit**: Observed value outside [1st, 99th] percentile for ANY statistic.

---

## When to Pivot (Model Class Changes)

### Scenario 1: All Models Fail Autocorrelation Test
**Symptom**: All three models show residual ACF(1) > 0.5

**Possible Causes**:
- Autocorrelation is more complex (ARMA(p,q), long memory)
- Regime-dependent autocorrelation (pre/post break have different φ)
- Latent confounders (missing covariates)

**Next Steps**:
1. Fit ARMA(2,1) variant of Model 2
2. Try Hidden Markov Model (2 states, each with different AR structure)
3. Check for external covariates

---

### Scenario 2: Structural Break is Illusory
**Symptom**: All models' posterior predictive Chow tests fail (< 60% significant)

**Possible Causes**:
- Break is actually smooth acceleration (GP is correct)
- Outliers creating false impression
- Break is in variance, not mean

**Next Steps**:
1. Trust Model 3 (GP)
2. Fit cubic polynomial without explicit changepoint
3. Check for variance regime change (two-regime dispersion model)

---

### Scenario 3: Negative Binomial is Inadequate
**Symptom**: All models underestimate variance (pp variance/mean < 40 vs observed 67.99)

**Possible Causes**:
- Heavier tails needed (generalized distributions)
- Zero-inflation (but EDA shows no zeros)
- Measurement error

**Next Steps**:
1. Generalized Poisson regression
2. Beta-Binomial (if counts have implicit upper bound)
3. Mixture models (two-component NB)

---

## Summary: Model Selection Flowchart

```
                          START HERE
                               |
                               v
                    Fit all 3 models in parallel
                               |
                               v
                    Check computational health
                    (R-hat, ESS, divergences)
                               |
                ┌──────────────┴──────────────┐
                v                             v
         All healthy?                  Any failures?
                |                             |
                v                             v
        Run falsification tests      Debug/reparameterize
                |                             |
     ┌──────────┼──────────┐                  |
     v          v          v                  |
  Model 1   Model 2   Model 3                 |
  passes?   passes?   passes?                 |
     |          |          |                  |
     └──────────┴──────────┘                  |
                |                             |
                v                             |
        How many survivors?                   |
                |                             |
      ┌─────────┼─────────┐                   |
      v         v         v                   |
   None       1-2      All 3                  |
      |         |         |                   |
      v         v         v                   |
  PIVOT    Use LOO   Model                    |
  (see     to rank  averaging                 |
  above)      |         |                     |
              v         v                     |
         Select best model                    |
              |                               |
              v                               |
      Sensitivity analysis                    |
      (priors, changepoint)                   |
              |                               |
              v                               |
         FINAL MODEL                          |
              |                               |
              v                               |
    Report results & decisions                |
                                              |
              <───────────────────────────────┘
```

---

## Key Takeaways

1. **Model 1 (DLM) is primary**: Most theoretically sound, best alignment with EDA
2. **Model 3 (GP) is validator**: Checks if discrete changepoint is real or artifact
3. **Model 2 (NB-AR) is baseline**: Fast, simple, but may be insufficient
4. **Falsification is critical**: Each model has explicit failure criteria
5. **Be ready to pivot**: If all fail, it's a learning opportunity, not a failure
6. **Combine if unclear**: If ΔLOO < 2*SE, use model stacking/averaging
7. **Trust the data**: If posteriors contradict EDA, investigate (don't ignore)

---

**Next Step**: Implement Model 1 Stan code and begin fitting.

**Critical Reminder**: The goal is scientific truth, not model completion. A "failed" model that teaches us something is more valuable than a "successful" model that we don't believe.

---

**Files**:
- Main document: `/workspace/experiments/designer_3/proposed_models.md`
- Implementation guide: `/workspace/experiments/designer_3/implementation_priority.md`
- This summary: `/workspace/experiments/designer_3/model_comparison_summary.md`
