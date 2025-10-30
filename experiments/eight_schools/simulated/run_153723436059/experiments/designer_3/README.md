# Designer 3 - Bayesian Model Proposals Summary

**Designer**: Designer 3 (Independent Proposal)
**Date**: 2025-10-29
**Output Directory**: `/workspace/experiments/designer_3/`

## Quick Overview

This directory contains Designer 3's independent proposals for Bayesian models to analyze the Eight Schools dataset.

## Main Document

**`proposed_models.md`** - Complete model specifications with:
- 3 distinct Bayesian model classes
- Mathematical specifications and Stan code
- Prior justifications
- Falsification criteria
- Decision frameworks

## Three Proposed Models

### Model 1: Near-Complete Pooling Hierarchical (BASELINE)
- **Core idea**: EDA suggests schools are highly similar (I² = 1.6%)
- **Key feature**: Informative HalfNormal(0, 5) prior on tau (expects small heterogeneity)
- **When it fails**: If posterior tau > 10, or posterior predictive checks fail

### Model 2: Flexible Horseshoe Hierarchical (SPARSE OUTLIERS)
- **Core idea**: Most schools similar, but 1-2 might be genuine outliers
- **Key feature**: School-specific shrinkage via horseshoe prior
- **When it fails**: If all schools shrink similarly, or no improvement in LOO-CV

### Model 3: Measurement Error Robust (SIGMA MISSPECIFICATION)
- **Core idea**: Variance paradox might indicate reported sigmas are wrong
- **Key feature**: Infers true sigmas via multiplicative correction factors
- **When it fails**: If posterior omega near zero, or corrections implausible

## Design Philosophy

**Adversarial mindset**: Each model is designed to be falsifiable. Success means finding which model the data reject, not confirming a hypothesis.

**Decision-driven**: Clear criteria for when to abandon each model, when to switch models, and when to reconsider everything.

**Three competing explanations**:
1. True homogeneity (variance paradox is real)
2. Sparse heterogeneity (variance paradox hides outliers)
3. Measurement error (variance paradox is artifact)

## Model Comparison Strategy

**Primary metric**: LOO-CV (Leave-One-Out Cross-Validation)
- Compare ELPD (Expected Log Predictive Density)
- Prefer model with ELPD > 2 SE higher
- Check Pareto-k diagnostics for influential observations

**Secondary checks**:
- Posterior predictive p-values
- Prior sensitivity analyses
- Leave-one-out school analyses
- Computational diagnostics

## Expected Outcome

**Most likely** (based on EDA): Model 1 wins
- Low heterogeneity is real
- Strong pooling appropriate
- Simple model sufficient

**But**: Analysis is designed to discover if this is wrong.

## Critical Decision Points

1. **After Model 1**: Is tau small and PPCs good? → STOP
2. **If tau large**: Fit Model 2 to check for sparse outliers
3. **If variance paradox unresolved**: Fit Model 3 to check sigma issues
4. **Final comparison**: LOO-CV across all fitted models

## Red Flags

- Prior-posterior conflict
- Computational failures
- Extreme parameter values
- All models fit equally well (data insufficient)
- All models fit equally poorly (reconsider everything)

## Success Criteria

Success = Finding a well-fitting, scientifically plausible model with reliable inferences

Success ≠ Finding complexity, rejecting nulls, or completing all models

**The goal is truth, not task completion.**

## Files in This Directory

- `proposed_models.md` - Full model specifications (22,000+ words)
- `README.md` - This summary

## Next Steps

1. Implement Model 1 (baseline) in Stan/PyMC
2. Fit and diagnose
3. Decide on Stage 2 (Models 2 and/or 3)
4. Compare and select
5. Stress test selected model

## Contact

These models were designed independently as part of a parallel design exercise. Other designers (1, 2) are proposing alternative approaches.
