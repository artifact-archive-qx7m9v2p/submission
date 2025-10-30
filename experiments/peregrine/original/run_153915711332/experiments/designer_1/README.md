# Model Designer 1: Distributional Choices & Variance Structure

## Overview

This directory contains Bayesian model designs focused on addressing the extreme overdispersion (Var/Mean = 67.99) and heteroscedasticity (26× variance increase) observed in the time series count data.

## Quick Start

**Read these files in order:**

1. **`model_summary.md`** - Quick reference (2 min read)
2. **`proposed_models.md`** - Full design document with rationale (20 min read)
3. **`technical_specs.md`** - Mathematical specifications (15 min read)

## Key Challenge

The data shows:
- 40 time series observations of counts [19, 272]
- Extreme overdispersion: Variance is 68× larger than mean
- Increasing variance: Late observations have 26× more variance than early
- Strong growth: 8.45× increase over time period
- Massive autocorrelation: ACF(1) = 0.989

**Standard Poisson models are completely inadequate.**

## Proposed Solutions

### Model 1a: Negative Binomial with Time-Varying Dispersion (PRIMARY)

**Hypothesis:** The count generation process becomes more variable over time.

**Key equation:**
```
C_t ~ NegativeBinomial(μ_t, θ_t)
log(μ_t) = α + β₁·year_t + β₂·year_t²
log(θ_t) = γ₀ + γ₁·year_t
```

**What we're testing:** Does dispersion decrease over time (γ₁ < 0)?

**Priority:** FIT THIS FIRST

### Model 1b: Negative Binomial with Constant Dispersion (BASELINE)

**Hypothesis:** NB's inherent variance structure is sufficient without time-varying dispersion.

**Key equation:**
```
C_t ~ NegativeBinomial(μ_t, θ)
log(μ_t) = α + β₁·year_t + β₂·year_t²
```

**What we're testing:** Is constant θ adequate?

**Priority:** Fit in parallel with 1a, compare via LOO-CV

### Model 2a: Random Walk State-Space (FALLBACK)

**Hypothesis:** Overdispersion comes from latent heterogeneity, not intrinsic to count process.

**Key equation:**
```
C_t ~ Poisson(λ_t)
log(λ_t) = log(μ_t) + η_t
η_t ~ Normal(η_{t-1}, σ_η)
```

**What we're testing:** Does latent variation explain overdispersion?

**Priority:** Only fit if Models 1a/1b show systematic failures

## Critical Success Criteria

A model SUCCEEDS if it:
1. Captures the variance structure (passes Stress Test 1)
2. Has good posterior predictive calibration (coverage ≈ nominal)
3. LOO-CV is reliable (Pareto k < 0.7)
4. Parameters are interpretable

A model FAILS if:
1. Posterior predictive variance systematically wrong
2. θ → 0 (too overdispersed for NB) or θ → ∞ (Poisson sufficient)
3. Computational problems (divergences > 20%, R-hat > 1.05)
4. LOO Pareto k > 0.7 for many observations

## Decision Flow

```
START
  ↓
Fit Model 1a + 1b in parallel
  ↓
Check convergence (R-hat, divergences)
  ↓
Compare via LOO-CV
  ├─ ΔLOO < 4 → Use Model 1b (simpler)
  ├─ ΔLOO > 10 → Use Model 1a (time-varying needed)
  └─ 4-10 → Check variance structure
      ↓
Posterior Predictive Checks
  ├─ PASS → DONE, report final model
  └─ FAIL → Fit Model 2a
      ↓
  Compare Model 1 vs 2
      ├─ Model 2 better → Use random effects
      └─ Still failing → Use Escape Routes
```

## Escape Routes (if all models fail)

**Route A:** Add explicit temporal dependence via lagged dependent variable
**Route B:** Changepoint model with discrete regime shift
**Route C:** Robust mixture model for extreme observations
**Route D:** Admit defeat, recommend more data or different approach

## Files in This Directory

- **`proposed_models.md`** - Main deliverable: Full model design with rationale, falsification criteria, and implementation details
- **`technical_specs.md`** - Mathematical specifications, Stan code structure, posterior quantities
- **`model_summary.md`** - Quick reference with rankings and key parameters
- **`README.md`** - This file

## Data Location

**Your data copy:** `/workspace/data/data_designer_1.csv`
**EDA Report:** `/workspace/eda/eda_report.md`

## Implementation Notes

All models must:
- Be fully Bayesian (specify priors)
- Use Stan or PyMC for inference
- Include log-likelihood for LOO comparison
- Include posterior predictive samples for validation

## Key Parameters to Monitor

### Model 1a:
- **θ_early vs θ_late:** Expect 2-5× difference if time-varying is real
- **γ₁:** Should be < 0 if dispersion decreases
- **β₂:** Should be ≠ 0 (nonlinear trend)

### Model 1b:
- **θ:** Expect 5-15 (moderate overdispersion)
- **β₁, β₂:** Same as Model 1a

### Model 2a:
- **σ_η:** Should be 0.3-0.8 to explain overdispersion
- **η_t:** Should show smooth evolution (random walk structure)

## Red Flags

**STOP and pivot if:**
- Posterior predictive variance is systematically wrong
- Parameters hit prior boundaries
- Divergent transitions > 20%
- LOO Pareto k > 0.7 for many points
- R-hat > 1.05 for any parameter

## Expected Timeline

**Day 1-2:** Fit Models 1a, 1b → Select winner via LOO
**Day 3-4:** Posterior predictive checks, stress tests
**Day 5:** Fit Model 2a if needed
**Day 6-7:** Sensitivity analysis, final validation

## Contact & Collaboration

**Designer:** Model Designer 1
**Focus:** Distributional choices and variance structure
**Collaboration note:** Coordinate with other designers on temporal autocorrelation (ACF = 0.989 not addressed by these models)

## Philosophy

This modeling effort succeeds by **discovering what's wrong**, not by completing a checklist. If all proposed models fail, that's valuable information - it tells us the data generation process is more complex than current models can handle.

**Goal: Truth, not task completion.**

---

**Status:** Design complete, ready for implementation
**Next Step:** Create Stan code files and fitting scripts
