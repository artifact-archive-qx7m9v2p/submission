# Experiment Plan: Eight Schools Bayesian Modeling
## Synthesis of Three Independent Model Designs

**Date**: 2025-10-29
**Data**: Eight Schools dataset (n=8, known sigma_i)
**Synthesized from**: Designer 1, Designer 2, Designer 3 proposals

---

## Overview

Three independent designers have proposed a total of 9 model concepts. After removing duplicates and consolidating similar approaches, I identify **5 distinct model classes** to test, ordered by theoretical priority based on EDA findings.

### Key EDA Context
- **Very low heterogeneity**: I² = 1.6% (typically < 25% suggests pooling)
- **Variance paradox**: Observed variance (124) < Expected (166), ratio = 0.75
- **High uncertainty**: Only 1/8 schools nominally significant
- **School 5 outlier**: Only negative effect (-4.88, z = -1.56)
- **Homogeneity test**: χ² = 7.12, p = 0.417 (fail to reject H₀: all effects equal)

---

## Prioritized Model Classes

### **Experiment 1: Standard Hierarchical Model (Partial Pooling)** ⭐ PRIMARY
**Proposed by**: All three designers (consensus model)
**Priority**: HIGHEST - Fit this first

**Mathematical Specification**:
```
Likelihood:    y_i ~ Normal(theta_i, sigma_i)   [sigma_i known]
School level:  theta_i ~ Normal(mu, tau)
Hyperpriors:   mu ~ Normal(0, 50)
               tau ~ HalfCauchy(0, 25)          [Gelman's recommendation]
```

**Implementation**: Non-centered parameterization (theta = mu + tau * theta_raw)

**Rationale**:
- Canonical model for hierarchical data with known measurement error
- Data-adaptive: if tau → 0, converges to complete pooling naturally
- Standard reference point for Eight Schools problem
- Allows data to determine optimal shrinkage strength

**Expected Outcome**:
- tau posterior likely small (3-8) given I² = 1.6%
- Strong shrinkage of extreme schools (especially School 5)
- mu ≈ 10-12 (near observed mean)

**Falsification Criteria**:
- Posterior tau > 15 (conflicts with EDA homogeneity finding)
- Poor posterior predictive checks (systematic misfit)
- Extreme prior sensitivity (posterior flips with minor prior changes)
- Computational issues (divergences, R-hat > 1.01)

**Success = Most likely adequate based on EDA**

---

### **Experiment 2: Near-Complete Pooling Model** ⭐ SECONDARY
**Proposed by**: Designer 1 (skeptical of heterogeneity), Designer 3 (informative prior)
**Priority**: HIGH - Alternative interpretation of I² = 1.6%

**Mathematical Specification**:
```
Likelihood:    y_i ~ Normal(theta_i, sigma_i)
School level:  theta_i ~ Normal(mu, tau)
Hyperpriors:   mu ~ Normal(0, 50)
               tau ~ HalfNormal(0, 5)           [INFORMATIVE - expects small tau]
```

**Key Difference from Exp 1**: Informative prior on tau based on EDA evidence
- HalfNormal(0, 5): median ≈ 3.4, 95% mass < 10
- vs HalfCauchy(0, 25): median ≈ 18, heavy tails

**Rationale**:
- Takes I² = 1.6% seriously as evidence of true homogeneity
- Variance ratio = 0.75 suggests tau should be very small
- Embeds EDA findings in prior while still allowing data to override
- Tests whether regularization toward homogeneity improves predictions

**Expected Outcome**:
- Even stronger shrinkage than Exp 1
- tau posterior concentrated near 0-5
- Should outperform Exp 1 on LOO-CV if homogeneity is real

**Falsification Criteria**:
- Posterior-prior conflict (tau posterior >> 5, fighting informative prior)
- Worse LOO-CV than Exp 1 (informative prior hurts)
- Poor calibration (over-confident predictions)

**When to fit**: If Exp 1 shows tau < 5, fit this to test if informative prior helps

---

### **Experiment 3: Horseshoe Hierarchical Model (Sparse Heterogeneity)**
**Proposed by**: Designer 3 (flexible shrinkage)
**Priority**: MEDIUM - Tests sparse outlier hypothesis

**Mathematical Specification**:
```
Likelihood:     y_i ~ Normal(theta_i, sigma_i)
School level:   theta_i ~ Normal(mu, tau * lambda_i)   [school-specific shrinkage]
Hyperpriors:    mu ~ Normal(0, 50)
                tau ~ HalfCauchy(0, 25)
                lambda_i ~ HalfCauchy(0, 1)             [local shrinkage parameters]
```

**Rationale**:
- Most schools similar (lambda_i ≈ 0) but 1-2 true outliers allowed (lambda_i ≈ 1)
- Addresses concern: "What if School 5 is genuinely different?"
- Horseshoe prior provides adaptive shrinkage per school
- Can collapse to standard hierarchical if all lambda_i similar

**Expected Outcome**:
- If sparse heterogeneity exists: 1-2 schools with lambda_i >> others
- If not: lambda_i all similar, reduces to Exp 1
- Likely: homogeneity means all lambda_i ≈ 0.5-1.5 (no strong signal)

**Falsification Criteria**:
- All lambda_i converge to similar values (no sparsity)
- No improvement in LOO-CV vs Exp 1
- Overparameterization warnings (8 schools, 8 lambda_i = too flexible)

**When to fit**: If Exp 1 shows residual misfit for specific schools (e.g., School 5)

---

### **Experiment 4: Two-Component Mixture Model**
**Proposed by**: Designer 1 (latent subgroups), Designer 2 (mixture)
**Priority**: MEDIUM - Tests hidden cluster hypothesis

**Mathematical Specification**:
```
Component assignment:  z_i ~ Categorical(pi)        [pi = mixture weight]
Component likelihood:  y_i ~ Normal(theta_i, sigma_i)
School level:          theta_i ~ Normal(mu_{z_i}, tau_{z_i})
Hyperpriors:           pi ~ Dirichlet(2, 2)         [K=2 components]
                       mu_k ~ Normal(0, 50)         for k=1,2
                       tau_k ~ HalfCauchy(0, 25)    for k=1,2
```

**Rationale**:
- Tests hypothesis: apparent homogeneity masks two distinct groups
- Schools 1-4 (high effects) vs Schools 5-8 (lower effects)?
- Variance paradox could result from offsetting components
- Can detect structure missed by pooling models

**Expected Outcome**:
- Most likely: single component dominates (pi ≈ 0.9) → mixture not needed
- If real: clear separation with Schools 5,6,7 in one group, others in another
- Model comparison via LOO should penalize complexity if not supported

**Falsification Criteria**:
- One component has pi > 0.85 (essentially single component)
- Label switching in MCMC (components not identifiable)
- LOO-CV worse than Exp 1 by ΔLOO > 4

**When to fit**: If Exp 1 shows bimodal residuals or persistent misfit

---

### **Experiment 5: Measurement Error Model (Sigma Misspecification)**
**Proposed by**: Designer 2, Designer 3
**Priority**: LOW - Questions data quality assumption

**Mathematical Specification**:
```
True sigma:    sigma_true_i = sigma_i * psi_i      [multiplicative correction]
Likelihood:    y_i ~ Normal(theta_i, sigma_true_i)
School level:  theta_i ~ Normal(mu, tau)
Hyperpriors:   mu ~ Normal(0, 50)
               tau ~ HalfCauchy(0, 25)
               psi_i ~ Lognormal(0, omega)         [correction factors]
               omega ~ HalfNormal(0, 0.2)          [misspecification scale]
```

**Rationale**:
- Variance paradox could indicate overestimated reported sigma_i
- If true sigma_i smaller than reported, less shrinkage needed
- Tests assumption that sigma_i are truly "known"
- Allows data to correct measurement errors

**Expected Outcome**:
- If sigma_i accurate: omega ≈ 0, psi_i ≈ 1, reduces to Exp 1
- If overestimated: psi_i < 1, omega > 0.1
- Should show improved PPC fit if measurement issue exists

**Falsification Criteria**:
- omega posterior concentrated near 0 (no correction needed)
- All psi_i ≈ 1 (sigma_i were correct)
- Worse LOO-CV than Exp 1 (overparameterized)
- Implausible corrections (psi_i < 0.5 or > 2)

**When to fit**: Only if Exp 1 shows persistent PPC failures despite good R-hat/ESS

---

## Implementation Strategy

### Phase 1: Core Models (REQUIRED - fit both)
1. **Experiment 1** (Standard Hierarchical) - Always fit first
2. **Experiment 2** (Near-Complete Pooling) - Fit if Exp 1 shows tau < 8

**Stopping rule**: If both pass all checks and differ by |ΔLOO| < 2, STOP. Report both.

### Phase 2: Diagnostic Models (CONDITIONAL - fit if issues arise)
3. **Experiment 3** (Horseshoe) - Only if specific schools show persistent misfit
4. **Experiment 4** (Mixture) - Only if residuals show clustering or bimodality
5. **Experiment 5** (Measurement Error) - Only if PPC fails systematically

**Minimum attempt policy**: Must complete Experiments 1-2 unless Exp 1 fails prior-predictive or SBC.

---

## Validation Pipeline (same for all models)

For each experiment:
```
1. Prior Predictive Check     → Does prior generate reasonable data?
2. Simulation-Based Calibration → Can model recover known parameters?
3. Fit to Real Data            → MCMC with diagnostics (R-hat, ESS, divergences)
4. Posterior Predictive Check  → Does posterior generate data like observed?
5. Model Critique              → ACCEPT / REVISE / REJECT decision
```

**Success criteria**:
- R-hat < 1.01 for all parameters
- ESS > 400 (bulk and tail)
- Zero divergent transitions
- LOO Pareto-k < 0.7 for all observations
- PPC p-values ∈ [0.05, 0.95] for test statistics

---

## Model Comparison Strategy

**If 2+ models pass validation**:
1. Compute LOO-CV for all accepted models
2. Use `az.compare()` to rank by ELPD
3. Apply parsimony rule: if |ΔLOO| < 2×SE, prefer simpler model
4. Report top model + any within 2 SE

**Comparison metrics**:
- ΔELPD (expected log predictive density difference)
- Pareto-k diagnostics (model misspecification)
- Posterior predictive coverage (calibration)
- Shrinkage patterns (substantive interpretation)

---

## Expected Timeline

**Experiment 1**: 30-45 min (prior predictive, SBC, fit, PPC, critique)
**Experiment 2**: 20-30 min (same pipeline)
**Experiments 3-5**: 30-40 min each (if needed)

**Total**: 1.5-3 hours depending on how many models needed

---

## Synthesis Notes

### Convergent Findings Across Designers
✅ All three designers prioritized hierarchical model as starting point
✅ All recognized variance paradox as key puzzle
✅ All proposed non-centered parameterization for computational stability
✅ All suggested mu ~ Normal(0, 50) and tau ~ HalfCauchy(0, 25) as baseline

### Divergent Proposals (strength of parallel design)
🔀 Designer 1: Emphasized mixture models for latent structure
🔀 Designer 2: Focused on robustness to assumption violations
🔀 Designer 3: Proposed informative priors based on EDA evidence

### What Parallel Design Caught
- **Near-complete pooling model**: Designer 3's informative prior approach complements Designer 1's "skeptical of heterogeneity" model
- **Measurement error model**: Both Designers 2 & 3 independently identified sigma misspecification as possible explanation for variance paradox
- **Multiple falsification criteria**: Each designer proposed different red flags, creating comprehensive checklist

### Models NOT Proposed (deliberate omissions)
❌ **Robust t-likelihood**: EDA confirmed normality (all tests p > 0.67)
❌ **Spatial/network models**: No school adjacency or structure information
❌ **Covariate models**: No school-level predictors available
❌ **Time series**: Cross-sectional data, not longitudinal

---

## Decision Points

### When to stop iterating
- ✅ Model passes all validation checks
- ✅ Posterior predictive checks show good fit
- ✅ LOO-CV comparable to or better than alternatives
- ✅ Substantive interpretation makes sense

### When to try next model
- ❌ Persistent misfit in PPC despite good diagnostics
- ❌ Extreme posterior values (tau > 20, implies model misspecification)
- ❌ Specific schools consistently poorly predicted
- ❌ Prior-posterior conflict

### When to stop entirely
- ✅ Two models within ΔLOO < 2 SE, both adequate
- ❌ All models fail validation (data quality issues)
- ❌ Computational limits reached (all models show divergences)
- ⏱️ Diminishing returns (new models no better than existing)

---

## Success Definition

**Success = Discovering which model(s) the data actually support**

NOT success = Confirming our prior beliefs about hierarchical models

If Experiment 1 fails and Experiment 5 succeeds, that's a SUCCESS because we learned the sigma_i were misreported.

If all models fail, that's also valuable: it means we need to question data quality or collect more schools.

**Commitment to falsification**: Each experiment has explicit "I will abandon this if..." criteria defined BEFORE seeing results.

---

## Files Generated by This Plan

```
experiments/
├── experiment_plan.md          [this file]
├── iteration_log.md            [track refinements]
├── experiment_1/               [Standard Hierarchical]
│   ├── metadata.md
│   ├── prior_predictive_check/
│   ├── simulation_based_validation/
│   ├── posterior_inference/
│   ├── posterior_predictive_check/
│   └── model_critique/
├── experiment_2/               [Near-Complete Pooling]
│   └── [same structure]
├── experiment_3/               [Horseshoe - if needed]
├── experiment_4/               [Mixture - if needed]
├── experiment_5/               [Measurement Error - if needed]
└── model_comparison/           [if multiple models accepted]
```

---

## Next Steps

1. ✅ **COMPLETED**: EDA analysis (variance paradox identified, homogeneity suggested)
2. ✅ **COMPLETED**: Model design by 3 independent designers (9 proposals synthesized to 5)
3. ⏭️ **NEXT**: Begin Experiment 1 validation pipeline
   - Prior predictive check
   - Simulation-based calibration
   - Fit to real data
   - Posterior predictive check
   - Model critique

**Ready to proceed with implementation.**
