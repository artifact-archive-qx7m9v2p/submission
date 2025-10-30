# Designer 1: Bayesian Model Design for Eight Schools

**Designer ID**: 1
**Date**: 2025-10-29
**Status**: Design Phase Complete - Awaiting Implementation

---

## Overview

This directory contains the complete model design strategy for the Eight Schools hierarchical modeling problem. Three distinct model classes are proposed, each representing a different hypothesis about the data generation process.

---

## File Structure

### Core Documents

1. **`proposed_models.md`** (33 KB) - MAIN DOCUMENT
   - Comprehensive model proposals with full theoretical justification
   - Three competing model classes with falsification criteria
   - Implementation plan and stress tests
   - Red flags and stopping rules

2. **`model_comparison_quick_reference.md`** (2.8 KB)
   - One-page summary of key distinctions
   - Decision cascade flowchart
   - Quick falsification matrix

3. **`mathematical_specifications.md`** (11 KB)
   - Complete mathematical notation and formulas
   - Implementation-ready specifications for Stan/PyMC
   - Prior predictive and posterior predictive check details
   - Computational guidelines

4. **`README.md`** (this file)
   - Navigation guide
   - Quick summary of approach

---

## Three Proposed Model Classes

### Model 1: Standard Hierarchical (Partial Pooling)
- **Hypothesis**: Random exchangeable effects with unknown between-school variation
- **Tau prior**: HalfCauchy(0, 25) - diffuse, lets data decide
- **Expected**: Moderate shrinkage, tau posterior mode around 3-8
- **Falsified if**: tau → 0 (switch to Model 2) or poor posterior predictive checks

### Model 2: Near-Complete Pooling (Homogeneity)
- **Hypothesis**: Effects are essentially identical (I² = 1.6% is real signal)
- **Tau prior**: Exponential(0.5) - strong regularization toward 0
- **Expected**: Strong shrinkage, tau < 2, all effects near common mean
- **Falsified if**: Posterior of tau fights prior (mode > 10)

### Model 3: Mixture Model (Latent Subgroups)
- **Hypothesis**: Two hidden subpopulations with different mean effects
- **Structure**: K=2 components with separate mu_k, tau_k, mixing proportions pi
- **Expected**: School 5 (negative effect) in different component than others
- **Falsified if**: Posterior mixing proportions collapse to single component (max pi > 0.85)

---

## Key Design Principles

### Truth-Seeking, Not Task Completion
- Success = discovering which model the data actually support
- Failure = defending a chosen model despite contradictory evidence
- Explicit falsification criteria for each model

### Adversarial Thinking
- "Why will this model FAIL?" mindset
- Stress tests designed to break assumptions
- Alternative approaches documented if initial models fail

### EDA-Informed Design
**Critical finding**: Variance paradox (observed < expected)
- I² = 1.6% (extremely low heterogeneity)
- Variance ratio = 0.75 (observed 75% of expected)
- Chi-square homogeneity test p = 0.417
- School 5 is only negative effect (possible outlier)

This suggests near-complete pooling may be justified, challenging standard hierarchical modeling assumptions.

---

## Implementation Roadmap

### Phase 1: Model Fitting
1. Translate specifications to Stan code (3 .stan files)
2. Run MCMC with convergence diagnostics
3. Handle computational issues (divergences, ESS, R-hat)

### Phase 2: Posterior Analysis
1. Extract key parameter posteriors (mu, tau, theta_i)
2. Compute shrinkage factors
3. Generate posterior predictive distributions

### Phase 3: Model Checking
1. Posterior predictive checks (visual + test statistics)
2. WAIC and LOO-CV model comparison
3. Pareto-k diagnostics for influential observations

### Phase 4: Sensitivity Analysis
1. Vary priors (especially tau prior in Model 1)
2. Exclude School 5 and refit
3. Compare to complete pooling baseline

### Phase 5: Decision & Reporting
1. Determine which model(s) are supported by data
2. Document falsification results
3. Provide substantive interpretation with uncertainty

---

## Falsification Decision Tree

```
Fit all 3 models → Convergence OK?
                       |
                      Yes
                       |
                       v
              PPC all pass?
              /           \
            Yes            No
            /               \
           v                 v
    Model comparison    Question normality
    (WAIC/LOO)          Consider robust models
        |
        v
    /    |    \
Model 1  |  Model 3
  OR     |  wins
Model 2  |    |
wins     |    v
  |      |  Investigate
  v      |  hidden structure
Report   |  (surprising!)
results  v
      Model 1 & 2
      similar
         |
         v
   Near-complete
   pooling justified
```

---

## Expected Outcome (My Prediction)

**Most likely**: Model 1 and Model 2 give similar results
- Posterior of tau will be small (< 5)
- Substantial shrinkage toward common mean
- Model 2 slightly simpler, Model 1 more conservative
- **Either outcome is scientifically interesting**

**Less likely**: Model 3 discovers meaningful mixture structure
- Would require clear component separation in posterior
- EDA shows no evidence of bimodality
- But worth testing explicitly to rule out

**My commitment**: If my prediction is wrong, I will report that as **success**, not failure.

---

## Key Questions This Design Answers

1. **Should we pool effects?**
   - Model comparison will answer definitively
   - All three models allow pooling, differ in strength

2. **How much heterogeneity exists?**
   - Posterior of tau quantifies between-school variation
   - Model 1: data-driven, Model 2: skeptical, Model 3: within-group

3. **Is School 5 genuinely different?**
   - Shrinkage analysis in Models 1-2
   - Component assignment in Model 3
   - LOO influence diagnostics

4. **What should we expect in a new school?**
   - Posterior predictive: theta_new ~ N(mu, tau)
   - Provides actionable prediction with uncertainty

---

## Red Flags to Monitor

### Across All Models
- **Prior-posterior conflict**: Model fighting the data
- **Computational pathologies**: Divergences, low ESS, high R-hat
- **Poor posterior predictive performance**: Despite good in-sample fit
- **Extreme parameters**: tau > 50, mu outside [-50, 50]

### Model-Specific
- **Model 1**: tau concentrates at 0 or ∞
- **Model 2**: tau posterior mode >> 10 (prior inadequate)
- **Model 3**: Single component dominates (mixture unnecessary)

### Data-Specific
- **School 5 always influential**: High Pareto-k across models
- **Wide intervals despite pooling**: Suggests data too noisy
- **All models fail PPC**: Normality assumption wrong

---

## Alternative Approaches (If Initial Models Fail)

**Backup Plan A**: Robust hierarchical model with Student-t errors
**Backup Plan B**: Non-parametric Dirichlet Process mixture (may not work with n=8)
**Backup Plan C**: Empirical Bayes (faster, less principled)
**Backup Plan D**: Simple inverse-variance weighted average (if hierarchical approaches overfit)

---

## Success Criteria

### This Design Succeeds If:
1. We learn which model class the data support (even if not Model 1)
2. We quantify uncertainty honestly (including model uncertainty)
3. We identify falsification criteria before seeing results
4. We provide actionable recommendations
5. We document what we learned about the phenomenon

### This Design Fails If:
1. We report a "best model" that fails PPC
2. We ignore evidence against our chosen model
3. We claim certainty with weak evidence
4. We fit complex models without checking simpler baselines
5. We treat model selection as the goal rather than understanding

---

## Technical Details

**Data**: `/workspace/data/data.csv` (8 schools, observed effects, known SEs)
**EDA Report**: `/workspace/eda/eda_report.md`
**Software**: Stan or PyMC for MCMC inference
**Diagnostics**: arviz for WAIC, LOO, convergence checks

**Estimated Compute Time**:
- Model 1: ~30 seconds
- Model 2: ~30 seconds
- Model 3: ~2-5 minutes (mixture slower)
- Total with sensitivity: ~30 minutes

---

## Contact & Questions

For details on:
- **Why these three models?** → See `proposed_models.md` Section 1-2
- **Mathematical specifications?** → See `mathematical_specifications.md`
- **Quick comparison?** → See `model_comparison_quick_reference.md`
- **Falsification criteria?** → See `proposed_models.md` Section 3-5 (each model)
- **Implementation plan?** → See `proposed_models.md` Section 8

---

## Next Steps

**For implementer**:
1. Read `mathematical_specifications.md` for exact model specs
2. Implement three Stan models
3. Follow implementation roadmap above
4. Check falsification criteria after each phase
5. Be ready to pivot if evidence demands it

**For reviewer**:
1. Read `proposed_models.md` executive summary
2. Review competing hypotheses in Section 1
3. Check if falsification criteria are clear and testable
4. Verify prior choices are justified
5. Confirm stress tests are adequate

---

**Designer**: 1
**Philosophy**: Truth-seeking through falsification
**Commitment**: Report what the data say, not what we expect
**Status**: Ready for implementation

---

*"The goal is not to be right. The goal is to find out what's true."*
