# Parametric Regression Model Design - Designer #1

## Quick Reference

**Designer:** Model Designer #1 (Parametric Regression Focus)
**Date:** 2025-10-27
**Status:** Design Complete, Ready for Implementation

## Files in This Directory

1. **`proposed_models.md`** (32 KB) - Main design document
   - Complete mathematical specifications for 3 model classes
   - Prior justifications with specific distributions
   - Falsification criteria for each model
   - Computational considerations and implementation notes
   - Backup plans if initial models fail
   - Stan code skeletons

2. **`model_summary.txt`** (5.5 KB) - Quick reference summary
   - Model comparison matrix
   - Decision tree for model selection
   - Falsification checklist
   - Success criteria

3. **`README.md`** (this file) - Navigation guide

## Three Proposed Model Classes

### Model 1: Logarithmic (PRIMARY)
- **Form:** Y ~ Normal(α + β·log(x+c), σ²)
- **Why:** Best empirical fit (R²=0.888), simple, interpretable
- **Falsification:** Residual patterns, change point real (ΔWAIC>6)

### Model 2: Rational Function (SECONDARY)
- **Form:** Y ~ Normal(Y_min + (Y_max-Y_min)·x^h/(K^h+x^h), σ²)
- **Why:** Mechanistic saturation, bounded predictions
- **Falsification:** Y_max non-identified, computational failure

### Model 3: Piecewise Linear (TERTIARY)
- **Form:** Two linear segments with continuous join at τ
- **Why:** Tests EDA's 66% RSS improvement from change point
- **Falsification:** tau posterior flat, slopes not different

## Critical Design Features

### Falsification-First Approach
Every model includes specific criteria for **when to abandon it**. This is not optional - it's the core of the scientific method.

### Decision Checkpoints
1. After fitting: Compare WAIC/LOO
2. After diagnostics: Apply falsification tests
3. After sensitivity: Assess robustness

### Escape Routes
- Plan B: Non-parametric (GP, splines, BART)
- Plan C: Different likelihood (Student-t, Beta, Mixture)
- Plan D: Different error structure (heteroscedastic, AR)

## Implementation Priority

1. **Week 1:** Code all 3 models in Stan, fit to data
2. **Week 2:** Diagnostics, WAIC/LOO, falsification tests
3. **Week 3:** Sensitivity analysis (prior, outlier, subset)
4. **Week 4:** Reporting and refinement

**STOPPING RULE:** If Week 2 shows all models fail falsification → PIVOT to non-parametric immediately

## Key Insights from Design Process

### What I'm Confident About
- Non-linear relationship exists (EDA is clear)
- Logarithmic form is strong baseline
- Change point at x≈7 deserves testing
- n=27 limits model complexity

### What I'm Uncertain About
- Whether smooth parametric forms are adequate
- If the change point is real or artifact
- Whether saturation truly occurs
- If homoscedasticity holds under scrutiny

### What Would Make Me Abandon All Models
1. All show systematic residual patterns
2. All fail to predict replicates
3. Prior-data conflict in all models
4. Computational pathologies persist
5. Different data subsets give wildly different fits

## Philosophy

**Goal:** Find truth, not complete tasks
**Strategy:** Design models to be falsified, not confirmed
**Success:** Honest uncertainty quantification, clear limitations
**Failure:** Overstating confidence, ignoring warning signs

**Core Principle:** "We tried these models and they all failed because X" is SUCCESS if true, not failure.

## Next Steps

For the implementation team:

1. Review `proposed_models.md` sections 1-3 for model specifications
2. Use Stan code in Appendix A as starting point
3. Follow falsification checklist in Appendix B strictly
4. Report back at each checkpoint (don't wait until end)
5. If models fail, use escape routes - don't force a bad model to work

## Contact/Questions

This is a standalone design. Implementation team should:
- Follow specifications exactly initially
- Deviate only if data strongly suggests (document why)
- Report failures quickly (they're learning opportunities)
- Don't hesitate to pivot if models are inadequate

---

**Remember:** A model that fails quickly and clearly is more valuable than one that appears to work but is actually misspecified. Be ruthlessly honest in diagnostics.
