# Designer 3: Robust & Alternative Bayesian Models
## Focus Area: Heavy-tailed Likelihoods, Mixture Models, Hierarchical Structures

---

## Overview

This design addresses the Y vs x relationship (n=27) with emphasis on:
- **Robust handling of the outlier** at x=31.5 (Cook's D = 1.51)
- **Alternative data generation hypotheses** beyond smooth curves
- **Mixture models** that test two-regime structure
- **Hierarchical variance** that captures location-specific uncertainty

---

## Key Design Philosophy

**Falsification over confirmation**: Each model has explicit criteria for abandonment
**Robustness over perfection**: Heavy-tailed likelihoods prevent outlier distortion
**Transparency about uncertainty**: Probabilistic regime assignment, hierarchical pooling
**Scientific plausibility**: Models must make domain sense, not just fit well

---

## Deliverables

### 1. **experiment_plan.md** (START HERE)
Concise overview of:
- Three competing hypotheses
- Model specifications (equations + priors)
- Decision rules and pivot points
- Success/failure criteria
- Timeline (3 weeks)

**Use this for**: Quick reference, decision-making, progress tracking

---

### 2. **proposed_models.md** (DETAILED REFERENCE)
Comprehensive 100+ page design document with:
- Full mathematical specifications
- Theoretical justifications
- Prior rationale with domain reasoning
- Expected strengths/weaknesses
- Falsification criteria per model
- Model comparison strategy
- Alternative approaches if primary models fail

**Use this for**: Implementation details, understanding rationale, troubleshooting

---

## Three Proposed Models

### Model 1: Student-t Logarithmic (HIGHEST PRIORITY)
**Core idea**: Logarithmic mean + heavy-tailed likelihood
**Handles outlier via**: Automatic downweighting when nu is estimated small
**Falsify if**: nu > 30 (Normal sufficient), residuals show patterns
**Expected outcome**: MOST LIKELY TO WIN

### Model 2: Mixture of Two Regimes (MEDIUM PRIORITY)
**Core idea**: Growth regime (x<7, steep) vs Plateau regime (x>7, flat)
**Handles outlier via**: Probabilistic regime assignment (might be in "wrong" regime)
**Falsify if**: Regime probabilities diffuse, slopes similar
**Expected outcome**: Win if regime structure is real heterogeneity

### Model 3: Hierarchical Variance (LOWER PRIORITY)
**Core idea**: Variance increases with x (sparse high-x region)
**Handles outlier via**: High location-specific variance at x=31.5
**Falsify if**: zeta ≈ 0 (no trend), tau ≈ 0 (no heterogeneity)
**Expected outcome**: Likely unnecessary but tests assumptions

---

## Quick Start Guide

### For Implementers:

1. **Read**: `experiment_plan.md` (15 min)
2. **Code**: Model 1 first (Student-t Log, highest priority)
3. **Diagnose**: Check R-hat, ESS, divergences
4. **Compare**: Fit all three, run LOO-CV
5. **Decide**: Use decision tree in experiment_plan.md

### For Reviewers:

1. **Skim**: This README (5 min)
2. **Focus**: "Falsification Criteria" sections in proposed_models.md
3. **Check**: Are stopping rules clear? Are alternative plans ready?
4. **Verify**: Prior choices justified scientifically?

---

## Model Comparison Strategy

### Phase 1: Individual Diagnostics
- R-hat < 1.01, ESS > 400, divergences < 1%
- Posterior predictive checks
- Residual analysis

### Phase 2: Cross-Model Comparison
- LOO-CV (primary criterion)
- WAIC (secondary)
- Pareto-k diagnostics

### Phase 3: Interpretation
- Scientific plausibility
- Uncertainty quantification
- Sensitivity to outlier

**Decision rule**: If LOO-CV ELPD differs by >2 SE, clear winner. Otherwise, use model averaging.

---

## Red Flags & Pivot Plans

### Red Flag 1: All models fail diagnostics
**Action**: Move to Gaussian Process (non-parametric)

### Red Flag 2: Residual patterns persist
**Action**: Try polynomial or splines

### Red Flag 3: x=31.5 has high Pareto-k everywhere
**Action**: Sensitivity analysis excluding it; check measurement

### Red Flag 4: Prior-posterior conflict
**Action**: Revise priors OR change model class

---

## Expected Timeline

- **Week 1**: Model 1 (Student-t Log) - baseline
- **Week 2**: Models 2 & 3 - alternatives
- **Week 3**: Comparison, sensitivity, reporting

**Total**: ~3 weeks for thorough Bayesian workflow

---

## What Makes This Design Unique

### Compared to Standard Approaches:
- **Robustness first**: Student-t likelihood, not just Normal
- **Mixture thinking**: Tests regime structure probabilistically
- **Hierarchical variance**: Questions constant-variance assumption
- **Explicit falsification**: Each model has abandonment criteria

### Compared to Other Designers:
- **Designer 1** (likely): Standard parametric models (log, asymptotic)
- **Designer 2** (likely): Changepoint, splines, advanced structures
- **Designer 3** (this): Robust & alternative data generation processes

---

## Key Questions Addressed

1. **Is outlier measurement error?**
   - Model 1 answer: If nu < 10, yes (heavy tails needed)
   - Model 3 answer: If sigma_31.5 >> others, uncertain region

2. **Is two-regime structure real?**
   - Model 2 directly tests this
   - If wins: Yes, distinct populations/mechanisms
   - If loses: Smooth curve adequate

3. **How uncertain are high-x predictions?**
   - All models: Wider intervals at x>20 (sparse data)
   - Critical for extrapolation decisions

---

## Success Criteria

**Minimum requirements**:
- All MCMC diagnostics pass (R-hat, ESS, divergences)
- Posterior predictive checks show good fit
- LOO-CV ranks models clearly

**Ideal outcome**:
- One model clearly best (LOO-CV ELPD > 2 SE better)
- Scientific interpretation straightforward
- Outlier handled without manual removal
- Predictions have appropriate uncertainty

---

## Contact & Questions

**For implementation questions**: See detailed Stan/PyMC code examples in proposed_models.md
**For prior justification**: See "Prior Rationale" sections per model
**For model comparison**: See "Model Comparison Strategy" section
**For pivots/alternatives**: See "Alternative Models If Primary Models Fail" section

---

## File Locations

All outputs in: `/workspace/experiments/designer_3/`

- **This file**: `README.md`
- **Quick reference**: `experiment_plan.md`
- **Full details**: `proposed_models.md`
- **Data**: `/workspace/data/data.csv`
- **EDA report**: `/workspace/eda/eda_report.md`

---

**Ready for implementation - good luck with the Bayesian inference!**
