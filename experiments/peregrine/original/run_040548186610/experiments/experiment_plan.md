# Bayesian Modeling Experiment Plan
**Date:** 2025-10-29
**Project:** Time Series Count Data Analysis
**Synthesized from:** 3 parallel model designers

---

## Executive Summary

Based on comprehensive EDA and input from three parallel model designers (parametric, non-parametric, temporal), this plan prioritizes **6 Bayesian models** for sequential implementation. The strategy balances theoretical rigor, computational feasibility, and scientific learning.

**Key Strategic Decision:** Start with parametric models (highest probability of success given n=40), validate whether temporal autocorrelation is real or spurious, then escalate to flexible models only if simpler approaches fail.

---

## EDA Summary: Core Modeling Challenges

**Data:** 40 observations, year ∈ [-1.67, 1.67], counts ∈ [19, 272]

**Critical Findings:**
1. **Extreme overdispersion:** Var/Mean = 68 (Poisson assumption violated)
2. **Non-linear growth:** Quadratic R²=0.961 >> Linear R²=0.885
3. **Accelerating trend:** Growth rate increases 6× (early→late)
4. **High apparent autocorrelation:** Lag-1 ACF = 0.989
5. **Heteroscedastic variance:** Different dispersion across time periods

**Central Scientific Question:**
Is the high autocorrelation (0.989) **real** (requiring state-space models) or **spurious** (artifact of smooth trending)?

---

## Model Prioritization Strategy

### Phase 1: Baseline Parametric Models (PRIORITY)
**Rationale:** Simple models with n=40, test if complexity is needed

**Experiment 1: Negative Binomial Quadratic** (Designer 1, Model 1)
- **Probability of success:** 85%
- **Expected outcome:** Good fit, but may show residual ACF 0.3-0.6
- **What we learn:** Baseline performance, whether overdispersion + trend are sufficient

**Experiment 2: Negative Binomial Exponential** (Designer 1, Model 2)
- **Probability of success:** 75%
- **Expected outcome:** Slightly worse than quadratic (less flexible)
- **What we learn:** Whether exponential growth is adequate, model parsimony test

**Decision Point 1:**
- If both models show residual ACF < 0.3 → STOP, accept simpler model, skip temporal models
- If residual ACF > 0.5 → Escalate to Phase 2 (temporal models)
- If intermediate → Proceed cautiously to Phase 3 (flexible models)

### Phase 2: Temporal Structure Models (CONDITIONAL)
**Rationale:** Only if Phase 1 shows residual autocorrelation > 0.5

**Experiment 3: AR(1) on Detrended Counts** (Designer 3, Model T3)
- **Probability of success:** 60%
- **Expected outcome:** Will reveal if ρ ≈ 0 (spurious ACF) or ρ > 0.5 (real)
- **What we learn:** Whether temporal dependence exists after detrending

**Experiment 4: Latent AR(1) State-Space** (Designer 3, Model T1)
- **Probability of success:** 50%
- **Expected outcome:** May have convergence issues, identifiability problems
- **What we learn:** Full separation of trend vs temporal correlation

**Decision Point 2:**
- If ρ posterior includes 0 → Temporal structure not needed, return to Phase 1 winner
- If ρ > 0.5 → Temporal correlation is real, select best temporal model
- If convergence fails → Pivot to Phase 3 (flexible models)

### Phase 3: Flexible Non-Parametric Models (CONDITIONAL)
**Rationale:** Only if parametric forms show systematic misfit

**Experiment 5: P-splines GLM** (Designer 2, Model 2)
- **Probability of success:** 70%
- **Expected outcome:** Good fit, automatic complexity tuning via τ
- **What we learn:** Whether flexibility improves predictions vs parametric

**Experiment 6: Gaussian Process NegBin** (Designer 2, Model 1)
- **Probability of success:** 40%
- **Expected outcome:** May overfit with n=40, computational challenges
- **What we learn:** Upper bound on flexible model performance

**Decision Point 3:**
- If LOO-IC not better than parametric baseline → Accept simpler model
- If clearly better → Accept flexible model but document cost-benefit

---

## Detailed Experiment Specifications

### Experiment 1: Negative Binomial Quadratic (BASELINE)

**Model Equations:**
```
C_i ~ NegativeBinomial(μ_i, φ)
log(μ_i) = β₀ + β₁·year_i + β₂·year_i²
```

**Priors:**
```
β₀ ~ Normal(4.7, 0.5)    # log(109) ≈ 4.7
β₁ ~ Normal(0.8, 0.3)    # Positive growth
β₂ ~ Normal(0.3, 0.2)    # Acceleration
φ ~ Gamma(2, 0.5)        # Overdispersion
```

**Implementation:** Stan (CmdStanPy primary, PyMC fallback)

**Success Criteria:**
- Convergence: R̂ < 1.01, ESS > 400
- Posterior predictive coverage: 85-98%
- No systematic residual patterns

**Failure Criteria (triggers escalation):**
- Residual ACF(1) > 0.5
- Systematic U-shaped residual pattern
- Coverage < 75%

**Falsification Tests:**
- Posterior predictive checks: scatter plots, coverage
- Residual diagnostics: ACF, QQ plots
- LOO-CV: ELPD, Pareto-k diagnostics

**Files:**
```
experiments/experiment_1/
├── metadata.md                    # This specification
├── prior_predictive_check/
├── simulation_based_validation/
├── posterior_inference/
│   └── diagnostics/
│       └── posterior_inference.netcdf  # With log_likelihood
├── posterior_predictive_check/
└── model_critique/
```

---

### Experiment 2: Negative Binomial Exponential

**Model Equations:**
```
C_i ~ NegativeBinomial(μ_i, φ)
log(μ_i) = β₀ + β₁·year_i
```

**Priors:**
```
β₀ ~ Normal(4.7, 0.5)
β₁ ~ Normal(0.8, 0.3)
φ ~ Gamma(2, 0.5)
```

**Why test this:** Parsimony—exponential growth may be adequate, avoiding quadratic term

**Success Criteria:** Same as Experiment 1

**Comparison:** If ΔLOO-IC < 2×SE vs Experiment 1, prefer simpler (exponential)

---

### Experiment 3: Count AR(1) on Detrended Data

**Model Equations:**
```
# Detrend first
C_detrend_i = C_i - exp(β₀ + β₁·year_i + β₂·year_i²)

# AR(1) on log-detrended counts
log(C_detrend_i) = ρ·log(C_detrend_{i-1}) + ε_i
ε_i ~ Normal(0, σ)
```

**Priors:**
```
ρ ~ Uniform(-1, 1)      # Flat prior—let data speak
σ ~ HalfNormal(0, 1)
```

**Critical Test:** Does ρ posterior include 0?
- If yes → ACF was spurious, temporal models unnecessary
- If no → ACF is real, need explicit temporal structure

---

### Experiment 4: Latent AR(1) State-Space

**Model Equations:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = α_t
α_t = β₀ + β₁·year_t + β₂·year_t² + ε_t
ε_t = ρ·ε_{t-1} + η_t, η_t ~ Normal(0, σ_η)
```

**Priors:**
```
β₀ ~ Normal(4.5, 1)
β₁ ~ Normal(0, 1)
β₂ ~ Normal(0, 0.5)
ρ ~ Beta(12, 3)          # Informative: favors high ACF
σ_η ~ HalfNormal(0, 0.5)
φ ~ HalfNormal(0, 10)
```

**Computational Strategy:**
- Use non-centered parameterization: `ε_t = ρ·ε_{t-1} + σ_η·η_raw_t`
- Monitor divergent transitions (< 5% threshold)

**Failure Modes:**
- Identifiability issues (β₂ vs ρ confounded)
- Prior-posterior conflict on ρ
- High divergences (> 10%)

---

### Experiment 5: Bayesian P-splines GLM

**Model Equations:**
```
C_i ~ NegativeBinomial(μ_i, φ)
log(μ_i) = Σ_k β_k · B_k(year_i)
β_k ~ Normal(β_{k-1}, τ⁻¹)   # Random walk smoothness prior
```

**Priors:**
```
β₁ ~ Normal(4.5, 1)      # First basis coefficient
τ ~ Gamma(1, 0.1)        # Smoothness penalty (auto-tuned)
φ ~ Gamma(2, 0.5)
```

**Why this works:** τ automatically tunes complexity—data decides smoothness

**Knots:** 10 evenly-spaced knots (rule: n/4 for n=40)

**Success:** τ posterior well away from 0 (not over-smooth) and ∞ (not under-smooth)

---

### Experiment 6: Gaussian Process Negative Binomial

**Model Equations:**
```
C_i ~ NegativeBinomial(μ_i, φ)
log(μ_i) = f(year_i)
f ~ GP(m(x), k(x,x'))
m(x) = β₀ + β₁·x
k(x,x') = α²·exp(-ρ²·(x-x')²)  # Squared exponential kernel
```

**Priors:**
```
β₀ ~ Normal(4.6, 1)
β₁ ~ Normal(0.8, 0.5)
α² ~ HalfNormal(0, 1)     # GP signal variance
ρ ~ InverseGamma(5, 5)    # Lengthscale
φ ~ HalfNormal(0, 10)
```

**Computational:** Use Cholesky decomposition with jitter (1e-9) for stability

**Failure Criteria:**
- Lengthscale ρ → ∞ (GP reduces to linear, wasted flexibility)
- Divergences > 10%
- LOO-IC worse than parametric baseline

---

## Minimum Attempt Policy

**Required:** Must attempt **at least Experiments 1 and 2** (both baseline parametric models)

**Exception:** Can skip Experiment 2 if Experiment 1 fails pre-fit validation (prior predictive or SBC)

**Rationale:** Need to validate baseline before exploring complexity

---

## Model Comparison Framework

**After each experiment completes:**
1. Document convergence diagnostics
2. Run posterior predictive checks
3. Compute LOO-CV (ELPD, SE, Pareto-k)
4. Check residual ACF
5. Record in `experiments/iteration_log.md`

**When multiple models ACCEPT:**
1. Compare via `az.compare()` on LOO-IC
2. Apply parsimony rule: if |ΔELPD| < 2×SE, prefer simpler
3. Check calibration (LOO-PIT plots)
4. Assess practical differences in predictions

**Final selection criteria:**
- **Statistical:** Best LOO-IC (or within 2×SE of best)
- **Scientific:** Interpretability, mechanistic insight
- **Practical:** Computational cost, robustness

---

## Expected Outcome & Contingencies

### Most Likely Scenario (70% probability)
1. **Experiment 1** (NB Quadratic) succeeds with good fit
2. Shows residual ACF = 0.3-0.5 (moderate)
3. **Experiment 2** (NB Exponential) slightly worse
4. Test **Experiment 3** (AR on detrended) → finds ρ ≈ 0.2 (weak)
5. **Conclusion:** Accept Experiment 1, temporal structure not needed

### Alternative Scenario 1 (20% probability)
1. **Experiments 1-2** show high residual ACF > 0.6
2. **Experiment 3** finds strong ρ > 0.7
3. **Experiment 4** (state-space) converges and improves fit
4. **Conclusion:** Accept temporal model, autocorrelation is real

### Alternative Scenario 2 (10% probability)
1. **Experiments 1-2** show systematic residual patterns (e.g., S-curve)
2. **Experiment 5** (P-splines) captures complex shape
3. **Conclusion:** Accept flexible model, parametric forms inadequate

---

## Iteration Strategy

### When to Refine a Model
- Fixable convergence issues (increase adapt_delta, iterations)
- Prior-data conflict → adjust priors, re-run
- Missing predictor identified in diagnostics

### When to Switch Model Classes
- Fundamental misspecification (systematic residual bias)
- Multiple refinements fail
- Computational barriers (divergences persist)

### When to Stop
- Adequate model found (passes all diagnostics, good predictions)
- Diminishing returns (improvements < 2×SE)
- All planned experiments exhausted

---

## Success Metrics Summary

**Phase 1 (Baseline) Success:**
- At least one model: R̂ < 1.01, ESS > 400, coverage 85-98%
- Interpretable parameters (β₁ > 0, β₂ reasonable)
- LOO-ELPD with SE quantified

**Phase 2 (Temporal) Success:**
- Convergence achieved
- Clear evidence for/against temporal dependence (ρ CI excludes/includes 0)
- Improvement over baseline if temporal structure included

**Phase 3 (Flexible) Success:**
- Convergence without overfitting
- LOO-IC better than baseline
- Smooth posterior predictions (not erratic)

**Overall Project Success:**
- At least one ACCEPT model
- Understanding of which complexity levels are justified
- Clear recommendation with quantified uncertainty

---

## File Organization

```
experiments/
├── experiment_plan.md              # This document
├── iteration_log.md                # Track refinements and decisions
├── adequacy_assessment.md          # Final determination (Phase 5)
├── model_comparison/               # If 2+ models accepted
│   └── comparison_report.md
├── experiment_1/                   # NB Quadratic
│   ├── metadata.md
│   ├── prior_predictive_check/
│   ├── simulation_based_validation/
│   ├── posterior_inference/
│   │   └── diagnostics/
│   │       └── posterior_inference.netcdf
│   ├── posterior_predictive_check/
│   └── model_critique/
│       ├── critique_summary.md
│       ├── decision.md             # ACCEPT/REVISE/REJECT
│       └── improvement_priorities.md
├── experiment_2/                   # NB Exponential
├── experiment_3/                   # AR(1) Detrended
├── experiment_4/                   # Latent AR(1)
├── experiment_5/                   # P-splines
└── experiment_6/                   # GP-NegBin
```

---

## Synthesis from Designers

**Designer 1 (Parametric GLMs):**
- Contributed: Experiments 1-2
- Key insight: Negative Binomial essential, quadratic vs exponential choice
- Expected: Good but not perfect (residual ACF 0.3-0.6)

**Designer 2 (Non-parametric/Flexible):**
- Contributed: Experiments 5-6
- Key insight: Flexibility may be unjustified with n=40, test rigorously
- Expected: GP may overfit, P-splines better balance

**Designer 3 (Temporal Structure):**
- Contributed: Experiments 3-4
- Key insight: High ACF likely spurious from trending
- Expected: Temporal models will show ρ ≈ 0, confirming suspicion

**Convergent Recommendations:**
1. All agree: Must use Negative Binomial (not Poisson)
2. All agree: Start simple, escalate only with evidence
3. All agree: Autocorrelation likely artifact, test explicitly
4. All agree: Define clear falsification criteria before fitting

---

## Timeline Estimate

- **Experiment 1:** 4-6 hours (prior pred check, SBC, fit, diagnostics, PPC, critique)
- **Experiment 2:** 3-4 hours (similar workflow, faster with template)
- **Decision Point 1:** 1 hour (compare, decide on Phase 2/3)
- **Experiment 3:** 3-4 hours (if needed)
- **Experiment 4:** 5-7 hours (if needed, complex)
- **Experiment 5:** 4-5 hours (if needed)
- **Experiment 6:** 5-8 hours (if needed, GP challenging)
- **Model Comparison:** 2-3 hours
- **Adequacy Assessment:** 1-2 hours
- **Final Report:** 3-4 hours

**Total:** 26-44 hours (highly dependent on which branches taken)

**Most likely path (Experiments 1-2 only):** 10-15 hours

---

## Document Complete

This experiment plan synthesizes all three designers' perspectives into a coherent, prioritized workflow with clear decision criteria at each stage. Implementation begins with Experiment 1.
