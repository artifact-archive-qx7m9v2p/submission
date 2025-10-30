# Bayesian Modeling Experiment Plan
## Time Series Count Data Analysis

**Date:** 2025-10-29
**Data:** 40 observations of counts (C) over standardized time (year)
**Synthesis:** Combined proposals from 3 parallel model designers

---

## Executive Summary

This plan synthesizes proposals from three independent model designers into a prioritized, coherent modeling strategy. The designers identified 8 distinct model classes, which we've consolidated into **5 core experiments** that test fundamentally different hypotheses about the data generation process.

**Key Challenge:** Extreme overdispersion (Var/Mean = 67.99) combined with massive autocorrelation (ACF(1) = 0.989) and probable changepoint at year ≈ 0.3.

**Strategy:** Start with the most theoretically justified models, implement rigorous falsification tests, and iterate based on diagnostic failures.

---

## Consolidated Model Classes

After reviewing all three designers' proposals, we identified significant overlap and complementary emphases:

### Designer 1: Variance Structure Focus
- Negative Binomial with time-varying dispersion
- Negative Binomial with constant dispersion
- Random effects state-space model

### Designer 2: Temporal Structure Focus
- Dynamic state-space model (with drift)
- Changepoint model
- Gaussian process model

### Designer 3: Structural Hypotheses Focus
- Hierarchical changepoint model
- Gaussian process model
- Latent state-space model

### Synthesis: 5 Core Model Classes

After removing duplicates and merging similar proposals:

1. **Negative Binomial State-Space Model** (Designers 1, 2, 3 all proposed variants)
2. **Changepoint Negative Binomial Model** (Designers 2, 3)
3. **Negative Binomial with Polynomial Trend** (Designer 1 + EDA baseline)
4. **Gaussian Process Count Model** (Designers 2, 3)
5. **Time-Varying Dispersion Model** (Designer 1 unique contribution)

---

## Prioritized Experiment Plan

### Experiment 1: Negative Binomial State-Space Model
**Priority:** HIGHEST ⭐⭐⭐
**Designer Consensus:** All 3 designers independently proposed this
**Expected Time:** 5-6 hours

**Rationale:**
- ACF(1) = 0.989 indicates temporal dependence is the dominant signal
- Lag-1 R² = 0.977 means C_t ≈ C_{t-1}, classic autoregressive pattern
- Decomposes variance into: (1) latent state evolution, (2) observation noise
- Addresses autocorrelation AND overdispersion simultaneously

**Model Specification:**
```
# Observation model
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t

# State evolution (random walk with drift)
η_t ~ Normal(η_{t-1} + δ, σ_η)
η_1 ~ Normal(log(50), 1)

# Priors
δ ~ Normal(0.05, 0.02)      # Expected ~5% growth per period
σ_η ~ Exponential(10)       # Tight innovations (ACF high)
φ ~ Exponential(0.1)        # Moderate overdispersion
```

**Hypotheses Tested:**
- H1: Most "overdispersion" is actually temporal correlation
- H2: Growth rate (drift δ) is approximately constant
- H3: Innovation variance σ_η is small relative to observation variance

**Falsification Criteria:**
- Abandon if: σ_η → 0 (degenerate model) or σ_η ~ observation SD (no structure)
- Abandon if: Residual ACF > 0.5 (autocorrelation not captured)
- Abandon if: One-step-ahead coverage < 75% (poor prediction)

**Expected Outcome:**
- Drift δ ≈ 0.06 (6% growth per period = 134% over full range)
- Innovation SD σ_η ≈ 0.05-0.10 (small fluctuations)
- Dispersion φ ≈ 10-20 (moderate, much less than naive 68)
- **Interpretation:** "Growth is a smooth latent process with small random fluctuations"

**Implementation Notes:**
- Use non-centered parameterization to avoid funnel
- Generated quantities must include: log_lik, one-step-ahead predictions, η_t estimates
- Posterior predictive checks: ACF of residuals, variance structure, sequential coverage

---

### Experiment 2: Changepoint Negative Binomial Model
**Priority:** HIGH ⭐⭐
**Designer Consensus:** Designers 2 and 3 both proposed
**Expected Time:** 4-5 hours

**Rationale:**
- EDA detected probable changepoint at year ≈ 0.3 (CUSUM, t-test both significant)
- Mean increases 4.5× (45.67 → 205.12) between regimes
- Variance ratio increases 26×
- Tests discrete intervention/threshold hypothesis vs smooth acceleration

**Model Specification:**
```
# Observation model
C_t ~ NegativeBinomial(μ_t, φ)

# Regime-specific means
log(μ_t) = α_r[t] + β_r[t] × year_t

where r[t] = 1 if year_t < τ else 2

# Changepoint prior
τ ~ Normal(0.3, 0.2)  # Centered on EDA estimate

# Regime-specific parameters
α_1 ~ Normal(log(40), 0.5)   # Early regime baseline
α_2 ~ Normal(log(150), 0.5)  # Late regime baseline
β_1 ~ Normal(0.3, 0.3)       # Early growth (slower)
β_2 ~ Normal(1.0, 0.3)       # Late growth (faster)
φ ~ Exponential(0.1)
```

**Hypotheses Tested:**
- H1: Structural break exists (τ posterior is concentrated)
- H2: Early and late regimes have different growth rates (β₁ ≠ β₂)
- H3: Discrete shift better explains data than smooth acceleration

**Falsification Criteria:**
- Abandon if: τ posterior is uniform/flat (changepoint not identified)
- Abandon if: β₁ and β₂ credible intervals overlap heavily (no regime difference)
- Abandon if: ΔLOO > 10 favoring smooth models (changepoint unnecessary)

**Expected Outcome (if successful):**
- Changepoint τ ≈ 0.2-0.4 with narrow CI
- Early growth β₁ ≈ 0.3 (slow), late growth β₂ ≈ 1.0-1.2 (fast)
- **Interpretation:** "System underwent regime shift around year 0.3"

**Implementation Notes:**
- Use mixture representation or step function for regime assignment
- Consider marginalizing over τ if poorly identified
- Posterior predictive: Does model generate realistic discontinuities?
- Sensitivity analysis: Results robust to different τ priors?

---

### Experiment 3: Negative Binomial Polynomial Trend
**Priority:** MEDIUM ⭐⭐
**Designer Consensus:** Designer 1 + EDA baseline recommendation
**Expected Time:** 3-4 hours

**Rationale:**
- Serves as baseline/null model for comparison
- EDA showed exponential R² = 0.935, quadratic might be sufficient
- Simpler than state-space, tests if temporal correlation is just spurious trend
- If this wins, we learn that simple GLM is adequate

**Model Specification:**
```
# Observation model
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = α + β₁ × year_t + β₂ × year_t²

# Priors
α ~ Normal(log(109), 1)      # Center at observed mean
β₁ ~ Normal(1.0, 0.5)        # Expect positive growth
β₂ ~ Normal(0, 0.25)         # Agnostic on curvature
φ ~ Exponential(0.1)         # Moderate dispersion
```

**Hypotheses Tested:**
- H1: Quadratic trend captures all systematic variation
- H2: No temporal correlation beyond trend (residuals IID)
- H3: Constant dispersion adequate (heteroscedasticity from mean-variance)

**Falsification Criteria:**
- Abandon if: Residual ACF > 0.7 (trend doesn't explain correlation)
- Abandon if: Posterior predictive variance checks fail systematically
- Abandon if: ΔLOO > 10 compared to state-space (temporal structure needed)

**Expected Outcome:**
- This model likely to fail falsification (residual ACF issue)
- Useful as baseline to quantify benefit of temporal structure
- **Interpretation (if fails):** "Simple GLM insufficient, need time series model"

**Implementation Notes:**
- Standard Stan neg_binomial_2_log implementation
- Check residual ACF explicitly
- Compare LOO to Experiments 1-2 to quantify temporal structure benefit

---

### Experiment 4: Gaussian Process Count Model
**Priority:** MEDIUM ⭐
**Designer Consensus:** Designers 2 and 3 both proposed
**Expected Time:** 5-6 hours

**Rationale:**
- Tests smooth nonparametric alternative to changepoint
- If neither parametric form (exponential, quadratic, changepoint) is correct, GP should win
- Serves as model adequacy check for parametric assumptions
- n=40 is tractable for GP

**Model Specification:**
```
# Observation model
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = f(year_t)

# Gaussian process for latent function
f ~ GP(m(x), k(x, x'))

# Mean function
m(x) = α

# Kernel (Squared Exponential + Linear)
k(x, x') = σ²_f × exp(-ρ²(x - x')²) + σ²_linear × x × x'

# Priors
α ~ Normal(log(109), 1)
σ²_f ~ Exponential(1)         # GP variance
ρ ~ InvGamma(5, 5)            # Length scale
σ²_linear ~ Exponential(1)    # Linear trend component
φ ~ Exponential(0.1)
```

**Hypotheses Tested:**
- H1: Smooth acceleration better than discrete changepoint
- H2: Parametric forms (exponential, polynomial) mis-specified
- H3: Function complexity justified by data

**Falsification Criteria:**
- Abandon if: GP simplifies to linear/quadratic (suggests overfit)
- Abandon if: Computational issues (divergences, poor ESS)
- Abandon if: Parametric models equivalent fit with much simpler structure

**Expected Outcome:**
- GP likely to fit well but may not improve LOO over simpler models
- Useful for identifying residual structure missed by parametrics
- **Interpretation:** Diagnostic tool for model adequacy

**Implementation Notes:**
- Use Hilbert space approximation if computational issues arise
- Compare GP posterior mean to parametric fits visually
- Check if GP shows discontinuities (supports changepoint) or smooth (supports state-space)

---

### Experiment 5: Time-Varying Dispersion Model
**Priority:** LOW ⭐
**Designer Consensus:** Designer 1 unique proposal
**Expected Time:** 4 hours

**Rationale:**
- Tests if dispersion parameter φ decreases over time (variance ratio 26×)
- Unique hypothesis: overdispersion structure itself changes
- Only fit if Experiments 1-3 show constant dispersion inadequate

**Model Specification:**
```
# Observation model
C_t ~ NegativeBinomial(μ_t, φ_t)
log(μ_t) = α + β₁ × year_t + β₂ × year_t²
log(φ_t) = γ₀ + γ₁ × year_t

# Priors
α ~ Normal(log(109), 1)
β₁ ~ Normal(1.0, 0.5)
β₂ ~ Normal(0, 0.25)
γ₀ ~ Normal(log(10), 1)      # Baseline dispersion
γ₁ ~ Normal(0, 0.5)          # Allow time-varying
```

**Hypotheses Tested:**
- H1: Dispersion changes systematically over time
- H2: Heteroscedasticity not fully captured by mean-variance relationship
- H3: Time-varying φ improves predictive performance

**Falsification Criteria:**
- Abandon if: γ₁ ≈ 0 (constant dispersion sufficient)
- Abandon if: φ_t → 0 or φ_t → ∞ at extremes (model failing)
- Abandon if: ΔLOO < 4 vs constant dispersion (unnecessary complexity)

**Expected Outcome:**
- Conditional on prior models failing dispersion checks
- **Interpretation:** "Variance structure is non-stationary"

**Implementation Notes:**
- Only fit if justified by previous experiments
- Check if time-varying dispersion is confounded with trend specification

---

## Implementation Strategy

### Phase 1: Core Models (Required)
**Experiments 1-2 (State-Space + Changepoint)** must be attempted per Minimum Attempt Policy.

**Parallel execution recommended:**
- Experiment 1 and 2 can run simultaneously (different model classes)
- Both address temporal structure but via different mechanisms
- Expected time: 5-6 hours in parallel

### Phase 2: Baseline Comparison (Conditional)
**Experiment 3 (Polynomial)** serves as baseline.

**Fit if:**
- Either Experiment 1 or 2 succeeds
- Need to quantify benefit of temporal structure

### Phase 3: Model Adequacy Checks (Conditional)
**Experiment 4 (GP)** tests parametric assumptions.

**Fit if:**
- Experiments 1-2 show concerning residual patterns
- Need to assess if functional form is fundamentally wrong
- Model critique suggests parametric misspecification

### Phase 4: Refinement (Conditional)
**Experiment 5 (Time-varying dispersion)** addresses specific failure mode.

**Fit if:**
- Experiments 1-3 show systematic dispersion structure violations
- Posterior predictive checks fail on variance by time period
- Model refinement suggests this specific extension

---

## Cross-Model Comparison Framework

### Primary Metric: LOO-ELPD
All models will be compared using Leave-One-Out Cross-Validation via ArviZ.

**Decision rules:**
- ΔLOO > 10: Strong preference for better model
- 4 < ΔLOO < 10: Moderate preference
- ΔLOO < 4: Models equivalent, use parsimony

**Diagnostics:**
- Check Pareto k values (all k < 0.7 for valid LOO)
- Compute LOO standard errors for uncertainty
- Use `az.compare()` for comprehensive comparison

### Secondary Metrics

**Posterior Predictive Checks:**
1. **Variance structure:** Var(C) by quintile vs posterior predictive
2. **Autocorrelation:** ACF of residuals (should be < 0.3)
3. **Sequential prediction:** One-step-ahead coverage (target 75-85%)
4. **Extremes:** Coverage of min/max values
5. **Growth rate:** Period-over-period changes match data

**Computational Diagnostics:**
- All Rhat < 1.01
- ESS_bulk > 400, ESS_tail > 400
- Divergences < 1% of samples
- Energy diagnostic passes

**Scientific Interpretability:**
- Parameters have meaningful interpretation
- Posterior estimates align with EDA findings
- Model tells coherent story about data generation

### Comparison Table Template

| Model | LOO-ELPD | SE | ΔLOO | Weight | Converged | Var Check | ACF Check | Coverage |
|-------|----------|----|----|--------|-----------|-----------|-----------|----------|
| State-Space | ? | ? | 0 | ? | ? | ? | ? | ? |
| Changepoint | ? | ? | ? | ? | ? | ? | ? | ? |
| Polynomial | ? | ? | ? | ? | ? | ? | ? | ? |
| GP | ? | ? | ? | ? | ? | ? | ? | ? |
| Time-Var Disp | ? | ? | ? | ? | ? | ? | ? | ? |

---

## Falsification Decision Tree

```
START → Fit Experiment 1 (State-Space)
        ├─ PASS validation → Continue to comparison
        │
        └─ FAIL → Document failure mode
                  ├─ σ_η → 0 → State-space degenerates → Try Exp 2
                  ├─ Residual ACF > 0.5 → AR(1) insufficient → Add AR(2) or changepoint
                  ├─ Coverage < 75% → Prediction poor → Try Exp 4 (GP)
                  └─ Divergences → Reparameterize or switch to Exp 3

Fit Experiment 2 (Changepoint)
        ├─ PASS validation → Continue to comparison
        │
        └─ FAIL → Document failure mode
                  ├─ τ posterior flat → Changepoint not identified → Smooth models
                  ├─ β₁ ≈ β₂ → No regime difference → Use Exp 3
                  └─ Poor fit → Try Exp 4 (GP)

Compare Experiments 1 & 2
        ├─ ΔLOO > 10 → Clear winner → Proceed to Phase 4 (Assessment)
        ├─ ΔLOO < 4 → Equivalent → Use model averaging
        └─ Both fail validation → Fit Exp 3 (baseline) and Exp 4 (GP)

If all fail → Escape routes
        ├─ Route A: Add lagged dependent variable
        ├─ Route B: Hierarchical time series
        ├─ Route C: Admit data too complex
        └─ Route D: Consult with domain expert
```

---

## Expected Outcomes & Predictions

### Most Likely Scenario (60% confidence)
**Experiment 1 (State-Space) wins clearly**

- ΔLOO > 10 compared to all other models
- Drift δ ≈ 0.06, innovation σ_η ≈ 0.08
- Residual ACF < 0.3 (temporal structure captured)
- **Interpretation:** "Data is fundamentally a time series with latent growth process. Most overdispersion is temporal correlation."

### Alternative Scenario 1 (25% confidence)
**Experiment 2 (Changepoint) wins**

- Changepoint τ ≈ 0.3 with narrow CI
- Clear regime differences in growth rates
- ΔLOO > 6 compared to state-space
- **Interpretation:** "Discrete intervention/threshold occurred around year 0.3"

### Alternative Scenario 2 (10% confidence)
**Experiments 1 and 2 equivalent (ΔLOO < 4)**

- Use Bayesian model averaging
- Both models capture different aspects of temporal structure
- **Interpretation:** "Data supports both smooth evolution and regime shift"

### Failure Scenario (5% confidence)
**All models fail validation**

- Extreme overdispersion not captured by any NB model
- ACF persists in all residuals
- Need fundamentally different approach (hierarchical, mixture, etc.)

---

## Success Criteria

### Minimum Success (Phase 3 entry)
- At least one model converges (Rhat < 1.01, no divergences)
- LOO computation succeeds (all Pareto k < 0.7)
- Model passes at least 3/5 posterior predictive checks

### Full Success (Phase 4 entry)
- At least two models converge for comparison
- Clear ranking via LOO (ΔLOO > 4)
- Winner passes all posterior predictive checks
- Residual ACF < 0.3

### Outstanding Success (Phase 6 entry)
- Multiple models converge
- Winner clearly identified (ΔLOO > 10)
- Scientific interpretation is coherent and actionable
- One-step-ahead predictions are accurate (coverage 80-90%)
- Model adequacy confirmed via GP comparison

---

## Timeline Estimate

**Optimistic (everything works):** 12-15 hours
- Experiment 1: 5 hours
- Experiment 2: 4 hours (parallel with 1)
- Experiment 3: 3 hours
- Comparison: 2 hours
- Refinement: 2 hours

**Realistic (some issues):** 20-25 hours
- Initial fits: 10 hours
- Debugging/reparameterization: 5 hours
- Additional experiments: 5 hours
- Comparison and refinement: 5 hours

**Pessimistic (major issues):** 35-40 hours
- Multiple failed attempts: 15 hours
- Escape route exploration: 10 hours
- Alternative model classes: 10 hours
- Final comparison: 5 hours

---

## Key Design Principles

1. **Falsification over confirmation:** Success = discovering what's wrong
2. **Parallel hypotheses:** Multiple competing models, not sequential refinement
3. **Explicit failure criteria:** "I will abandon if..." for each model
4. **Computational pragmatism:** If Stan fails, document and try PyMC
5. **Scientific interpretability:** Parameters must tell a coherent story

---

## Summary: Prioritized Execution Order

1. ⭐⭐⭐ **Experiment 1: State-Space** (required, highest priority)
2. ⭐⭐⭐ **Experiment 2: Changepoint** (required, parallel with 1)
3. ⭐⭐ **Experiment 3: Polynomial** (baseline, fit if 1-2 succeed)
4. ⭐ **Experiment 4: GP** (model adequacy, conditional)
5. ⭐ **Experiment 5: Time-varying dispersion** (refinement, conditional)

**Next step:** Begin Experiment 1 implementation with prior predictive checks.
