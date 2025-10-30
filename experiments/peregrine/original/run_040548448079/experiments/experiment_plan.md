# Bayesian Modeling Experiment Plan

## Problem Statement

Model the relationship between time (`year`, standardized) and count observations (`C`) in a 40-observation time series exhibiting:
- Extreme overdispersion (variance/mean = 67.99)
- Strong autocorrelation (ACF(1) = 0.944)
- Structural break at observation 17 (730% growth rate increase)
- Exponential growth pattern (log-linear relationship)

## Synthesized Model Portfolio

Three design teams independently proposed 9 model classes. After synthesis and deduplication, we prioritize models by theoretical alignment with EDA findings.

### Core Requirements (All Models)
1. **Negative Binomial likelihood** (not Poisson, ΔAIC = 2417)
2. **Log link function** (count data + exponential growth)
3. **Autocorrelation structure** (ACF(1) = 0.944 must be addressed)
4. **Bayesian implementation** (Stan or PyMC with MCMC/VI)

## Experiment Queue (Prioritized)

### Experiment 1: Fixed Changepoint Negative Binomial (PRIMARY)
**Source**: Designer 1, Model 1A
**Rationale**: Most aligned with EDA (discrete break at t=17, strong ACF)
**Priority**: HIGH (start here)

**Model Specification**:
```
Observation model:
  C_t ~ NegativeBinomial(μ_t, α)
  log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > 17) × (year_t - year_17)

Autocorrelation (AR(1) on observation-level random effects):
  ε_t ~ Normal(ρ × ε_{t-1}, σ_ε)
  log(μ_t) = ... + ε_t
```

**Priors** (from EDA):
- β_0 ~ Normal(4.3, 0.5)           # log(median(C)) ≈ 4.31
- β_1 ~ Normal(0.35, 0.3)          # pre-break slope from EDA
- β_2 ~ Normal(0.85, 0.5)          # post-break increase (1.2 - 0.35)
- α ~ Gamma(2, 3)                   # E[α] ≈ 0.67, informed by EDA α ≈ 0.61
- ρ ~ Beta(8, 2)                    # E[ρ] ≈ 0.8, informed by ACF(1) = 0.944
- σ_ε ~ Exponential(2)

**Falsification Criteria** (abandon if):
1. β_2 95% CI includes 0 (no regime change)
2. Residual ACF(1) > 0.5 (autocorrelation not captured)
3. LOO Pareto k > 0.7 for >10% observations
4. Rhat > 1.01 or ESS_bulk < 400 (convergence failure)
5. Posterior predictive checks show systematic misfit at structural break

**Expected Runtime**: 2-5 minutes (Stan)
**Implementation**: Stan (non-centered parameterization)

---

### Experiment 2: Gaussian Process Negative Binomial (VALIDATOR)
**Source**: Designers 2 & 3 convergence (Model 2B, 3C)
**Rationale**: Test if discrete break is necessary; flexible smooth alternative
**Priority**: HIGH (comparison to Exp 1)

**Model Specification**:
```
Observation model:
  C_t ~ NegativeBinomial(μ_t, α)
  log(μ_t) = f(year_t)

GP specification:
  f ~ GP(0, K)
  K(year_i, year_j) = σ_f² × exp(-||year_i - year_j||² / (2ℓ²))

AR component (nugget + AR(1)):
  K_obs = K + σ_nugget² I + AR(1) structure
```

**Priors**:
- α ~ Gamma(2, 3)                   # as in Exp 1
- σ_f ~ Normal(2, 1)                # marginal SD on log scale
- ℓ ~ InvGamma(5, 5)                # length-scale: E[ℓ] ≈ 1, allows ℓ ∈ [0.5, 3]
- σ_nugget ~ Exponential(2)
- ρ_ar ~ Beta(8, 2)                 # AR(1) coefficient

**Falsification Criteria**:
1. Length-scale ℓ collapses (<0.2) or explodes (>5)
2. Nugget dominates: σ_nugget > 2×σ_f (GP not capturing trend)
3. GP trajectory shows no acceleration at t=17 (should see inflection)
4. LOO ΔELPD < -20 vs Experiment 1 (discrete break is real)
5. Residual ACF(1) > 0.5

**Expected Runtime**: 30-60 minutes (Stan with Hilbert space approximation)
**Implementation**: Stan (sparse GP methods)

---

### Experiment 3: Dynamic Linear Model (STATE-SPACE ALTERNATIVE)
**Source**: Designer 3, Model 3A
**Rationale**: Autocorrelation as primary; latent dynamics; regime-shift in drift
**Priority**: MEDIUM (if Exps 1-2 show issues)

**Model Specification**:
```
Observation model:
  C_t ~ NegativeBinomial(exp(η_t), α)

State evolution:
  η_t = η_{t-1} + v_{t-1}           # level
  v_t = v_{t-1} + δ_t + ε_t         # velocity with regime shift

  δ_t = δ_1 + I(t > τ) × Δδ        # discrete drift change
  ε_t ~ Normal(0, σ_v)              # velocity innovations

Regime shift:
  τ = 17 (fixed from EDA)
  Δδ ~ Normal(0.8, 0.4)             # expected large positive shift
```

**Priors**:
- η_0 ~ Normal(4.3, 0.5)
- v_0 ~ Normal(0.3, 0.2)
- δ_1 ~ Normal(0.05, 0.1)          # pre-break drift
- Δδ ~ Normal(0.8, 0.4)            # post-break increase
- σ_v ~ Exponential(2)
- α ~ Gamma(2, 3)

**Falsification Criteria**:
1. σ_v very large (state evolution too noisy, not informative)
2. Δδ 95% CI includes 0 (no regime change in velocity)
3. State trajectory η_t deviates systematically from data
4. Residual checks fail (should have minimal structure)
5. LOO worse than simpler models

**Expected Runtime**: 10-30 minutes (Stan)
**Implementation**: Stan (non-centered for v_t)

---

### Experiment 4: Polynomial Negative Binomial (BASELINE)
**Source**: Designer 2, Model 2A
**Rationale**: Simplest smooth alternative; useful baseline
**Priority**: LOW (only if smooth models competitive)

**Model Specification**:
```
Observation model:
  C_t ~ NegativeBinomial(μ_t, α)
  log(μ_t) = β_0 + β_1×year + β_2×year² + ε_t

  ε_t ~ Normal(ρ × ε_{t-1}, σ_ε)    # AR(1) errors
```

**Priors**:
- β_0 ~ Normal(4.3, 0.5)
- β_1 ~ Normal(0.8, 0.5)
- β_2 ~ Normal(0.5, 0.3)            # positive curvature expected
- α ~ Gamma(2, 3)
- ρ ~ Beta(8, 2)
- σ_ε ~ Exponential(2)

**Falsification Criteria**:
1. Residual plot shows clear pattern around t=17 (missing structural break)
2. Residual ACF(1) > 0.5
3. LOO ΔELPD < -15 vs changepoint model
4. Posterior predictive checks show systematic deviation in second half

**Expected Runtime**: 2-5 minutes (Stan)

---

### Experiment 5: Unknown Changepoint (ROBUSTNESS CHECK)
**Source**: Designer 1, Model 1B
**Rationale**: Test if τ=17 is correct or if data prefer different breakpoint
**Priority**: CONDITIONAL (only if Exp 1 shows τ=17 is questionable)

**Model Specification**:
Same as Experiment 1, but:
```
  τ ~ DiscreteUniform(5, 35)       # restrict to reasonable range
```

**Priors**:
- Same as Experiment 1 for β, α, ρ
- τ prior uniform (10 observations padding from edges)

**Falsification Criteria**:
1. Posterior p(τ) is diffuse or multimodal (no clear break)
2. τ posterior mode far from 17 AND Exp 1 residuals are fine (EDA misleading)
3. Computational failure (discrete parameter mixing issues)
4. LOO shows no improvement over fixed τ=17

**Expected Runtime**: 20-60 minutes (PyMC, discrete parameter challenging)
**Implementation**: PyMC (better discrete parameter support)

---

## Model Comparison Strategy

### Stage 1: Individual Validation
For each experiment:
1. **Prior predictive check**: Ensure priors allow observed patterns
2. **Simulation-based calibration**: Can model recover parameters?
3. **Posterior inference**: Fit to real data with diagnostics
4. **Posterior predictive check**: Does model capture key patterns?
5. **Model critique**: Apply falsification criteria

### Stage 2: Model Comparison (survivors only)
Among models that pass Stage 1:
1. **LOO cross-validation**: Compute ELPD_loo ± SE
2. **Parsimony rule**: Prefer simpler model if |ΔELPD| < 2×SE
3. **Predictive checks**: Which model best captures structural features?
4. **Scientific interpretability**: Which model tells clearest story?

### Stage 3: Sensitivity Analysis
For accepted model(s):
1. **Prior sensitivity**: Vary priors within reasonable bounds
2. **Structural sensitivity**: Test model assumptions
3. **Temporal CV**: Out-of-sample prediction performance

## Decision Tree

```
START: Fit Experiment 1 (Fixed Changepoint)
   ↓
   PASS all falsification tests?
   ├─ YES → FIT Experiment 2 (GP) for comparison
   │         ├─ GP ΔLOO < -20 → ACCEPT Exp 1 (discrete break confirmed)
   │         └─ GP ΔLOO ≥ -20 → Compare both, check PPC
   │
   └─ NO → Diagnose failure mode:
           ├─ β_2 ≈ 0 → Try Exp 2 (GP), maybe no discrete break
           ├─ ACF(1) still high → Try Exp 3 (DLM with richer dynamics)
           ├─ Break location wrong → Try Exp 5 (unknown τ)
           └─ Computational issues → Simplify to Exp 4 (polynomial)

IF Exp 1 & 2 both fail:
   → Fit Experiment 3 (DLM)
   → Consider spline models (Designer 2, Model 2C)
   → Re-examine EDA assumptions

IF all models fail falsification:
   → STOP
   → Report limitations
   → Consider alternative paradigms (Hidden Markov, BART, etc.)
```

## Minimum Attempt Policy

Per workflow guidelines:
- **Must attempt**: Experiments 1 and 2 (unless Exp 1 fails pre-fit validation)
- **Conditional**: Experiments 3, 4, 5 based on results
- **Document**: All decisions in `log.md` and `experiments/iteration_log.md`

## Expected Timeline

| Experiment | Prior/SBC | Fit | PPC/Critique | Total |
|------------|-----------|-----|--------------|-------|
| Exp 1      | 1 hour    | 5m  | 1 hour       | 2-3h  |
| Exp 2      | 2 hours   | 1h  | 2 hours      | 5-6h  |
| Exp 3      | 1.5 hours | 30m | 1.5 hours    | 3-4h  |
| Exp 4      | 1 hour    | 5m  | 1 hour       | 2-3h  |
| Exp 5      | 1 hour    | 1h  | 1 hour       | 3-4h  |
| Comparison | -         | -   | 2-3 hours    | 2-3h  |
| **TOTAL**  | 6.5h      | 3h  | 8.5h         | 18-23h |

## Success Criteria

**Adequate model** must satisfy:
1. ✅ All convergence diagnostics (Rhat ≤ 1.01, ESS_bulk ≥ 400)
2. ✅ Pass all falsification tests
3. ✅ Capture autocorrelation (residual ACF(1) < 0.4)
4. ✅ Capture structural break (if present in data)
5. ✅ LOO Pareto k < 0.7 for 90%+ observations
6. ✅ Posterior predictive checks show no systematic misfit
7. ✅ Scientific interpretability (parameters make sense)

**Model comparison** (if multiple pass):
- Use LOO ΔELPD ± SE
- Apply parsimony rule (prefer simpler if within 2 SE)
- Consider interpretability and scientific utility

## Falsification Philosophy

Each model is designed to **fail if assumptions are wrong**:
- **Changepoint models fail** if break is smooth or nonexistent
- **GP models fail** if break is discrete
- **DLM fails** if dynamics are simpler than state-space structure
- **Polynomial fails** if discrete regime change present

**Goal**: Find the model that best represents the data-generating process, not force any model to "work."

## Deliverables

For each experiment `experiment_N/`:
- `metadata.md` - Model specification
- `prior_predictive_check/` - Prior validation
- `simulation_based_validation/` - SBC results
- `posterior_inference/` - Fit results and diagnostics
- `posterior_predictive_check/` - Model validation
- `model_critique/` - Assessment and decision

Final:
- `experiments/model_comparison/` - Comparison of accepted models
- `experiments/adequacy_assessment.md` - Final determination
- `final_report/report.md` - Complete analysis

## Implementation Notes

### Stan vs PyMC
- **Stan**: Experiments 1, 2, 3, 4 (faster, better for continuous parameters)
- **PyMC**: Experiment 5 (better discrete parameter support for unknown τ)

### Key Computational Considerations
1. Use **non-centered parameterization** for hierarchical structures
2. Use **sparse GP approximations** (Hilbert space) for Experiment 2
3. **Initialize carefully**: Use EDA estimates for starting values
4. **Sampling parameters**: 4 chains, 2000 iterations (1000 warmup)

### Autocorrelation Implementation
All models include AR(1) structure. Three approaches:
1. **Observation-level random effects** with AR(1) (Exps 1, 4)
2. **GP with temporal kernel** + AR nugget (Exp 2)
3. **State-space latent dynamics** (Exp 3)

## References

- EDA Report: `/workspace/eda/eda_report.md`
- Designer 1 Proposals: `/workspace/experiments/designer_1/proposed_models.md`
- Designer 2 Proposals: `/workspace/experiments/designer_2/proposed_models.md`
- Designer 3 Proposals: `/workspace/experiments/designer_3/proposed_models.md`

---

**Status**: READY FOR IMPLEMENTATION
**Next**: Begin Experiment 1 with prior predictive checks
