# Model Designer 1: Classical Hierarchical Modeling Strategy

**Designer:** Model Designer 1
**Focus:** Classical hierarchical meta-analysis models with rigorous falsification criteria
**Date:** 2025-10-28
**Status:** Design Complete, Ready for Implementation

---

## Quick Start

**If you have 8 hours:** Read `model_summary.md` and implement Models 1-2 with LOO comparison

**If you have 15+ hours:** Follow full plan in `proposed_models.md` including all sensitivity analyses

**For implementation:** Use `stan_templates.md` for ready-to-run Stan code

---

## Overview

This design proposes **three fundamentally different Bayesian model classes** for meta-analysis of J=8 studies with low heterogeneity (I²=2.9%):

### Model 1: Complete Pooling (Common Effect)
- **Assumption:** All studies measure identical true effect
- **Philosophy:** Parsimony - simplest explanation
- **Parameters:** 1 (mu only)
- **Expected:** mu ≈ 11.27, no heterogeneity

### Model 2: Partial Pooling (Hierarchical Random Effects)
- **Assumption:** Studies differ but share information via shrinkage
- **Philosophy:** Conservatism - allow uncertainty about heterogeneity
- **Parameters:** 2 + J (mu, tau, theta_1...theta_J)
- **Expected:** mu ≈ 11.27, tau ≈ 2, strong shrinkage (>95%)

### Model 3: Skeptical Prior (Hierarchical with Tight Priors)
- **Assumption:** Large effects are rare, make data prove itself
- **Philosophy:** Skepticism - guard against overconfidence
- **Parameters:** Same as Model 2, but stronger priors
- **Expected:** mu ≈ 9-10 (shrunk toward zero), tests robustness

---

## Key Design Principles

### 1. Falsification Mindset
Each model has **explicit criteria for rejection**:
- Model 1: Abandon if Study 5 residual |z| > 2.5 or ΔLOO > 4
- Model 2: Abandon if tau hits boundary or Study 4 dominates
- Model 3: Abandon if prior conflicts with data severely

### 2. Red Flags for Major Pivot
**STOP everything and reconsider if:**
- Tau > 10 (heterogeneity much higher than expected)
- Pareto-k > 0.7 for multiple studies (outliers present)
- Study 4 removal changes mu by >8 units (fragile to single study)
- Divergent transitions persist after tuning (geometry issues)
- Posterior mu negative (contradicts EDA)

### 3. Success = Learning, Not Confirming
**Success is NOT:**
- All models converging smoothly
- Results matching EDA exactly
- Narrow credible intervals

**Success IS:**
- Discovering which models fail and why
- Quantifying uncertainty honestly
- Finding evidence that forces model revision
- Understanding data generation process

---

## File Guide

### 1. `proposed_models.md` (33 KB, 911 lines)
**Complete design specification with:**
- Full mathematical specifications for all 3 models
- Theoretical justification from meta-analysis literature
- Expected behavior given EDA findings
- Detailed falsification criteria for each model
- Computational considerations (Stan/PyMC)
- Stress tests and sensitivity analyses
- Model comparison strategy
- Expected challenges and pitfalls
- 15+ academic references

**Read this for:** Rigorous theoretical foundation and complete design rationale

### 2. `model_summary.md` (4.2 KB, 92 lines)
**Quick reference with:**
- One-paragraph description of each model
- Table comparing key differences
- Critical red flags checklist
- Implementation phase breakdown
- Stopping rules
- Falsifiable predictions

**Read this for:** Quick orientation and practical checklist

### 3. `stan_templates.md` (16 KB, 585 lines)
**Ready-to-run code with:**
- Complete Stan code for all 3 models
- Python interfaces (CmdStanPy)
- LOO cross-validation code
- Posterior predictive checks
- Influence analysis (Study 4 removal)
- Visualization pipeline
- Diagnostic checklist
- Troubleshooting guide

**Read this for:** Actual implementation and analysis pipeline

### 4. `README.md` (this file)
**Navigation guide and executive summary**

---

## Expected Results (Falsifiable)

### Most Likely Outcome (70% confidence)
- Models 1 and 2 are equivalent (ΔLOO < 4)
- Tau posterior concentrates near zero
- Complete pooling is adequate
- Study 4 is influential but not problematic

### Alternative Outcome (25% confidence)
- All models agree despite different priors
- Data overwhelms all reasonable priors
- Robust positive effect around 10-12

### Surprising Outcome (5% confidence)
- LOO strongly favors Model 2 over Model 1
- Tau well away from zero
- Heterogeneity is real but EDA underestimated it

### Would Shock Me
- Negative posterior mu
- Tau > 10
- Multiple Pareto-k > 0.7
- Prior-data conflict in all models

---

## Implementation Timeline

### Phase 1: Core Models (6-8 hours)
**Must complete:**
1. Model 1: Complete pooling
2. Model 2: Partial pooling (non-centered)
3. LOO comparison
4. Basic diagnostics

**Deliverable:** Understand which pooling strategy is appropriate

### Phase 2: Sensitivity (4-6 hours)
**Should complete:**
5. Model 3: Skeptical prior
6. Study 4 removal analysis
7. Prior sensitivity grid

**Deliverable:** Robustness to prior and data perturbations

### Phase 3: Advanced (4-6 hours)
**Nice to have:**
8. Full posterior predictive checks
9. Residual analysis
10. Shrinkage visualization
11. Prior-posterior comparisons

**Deliverable:** Comprehensive diagnostic report

**Total: 15-20 hours for complete analysis**

---

## Connection to Other Designers

I am Designer 1, focused on **classical hierarchical models**. Other designers may propose:

- **Designer 2:** May explore robust likelihoods (Student-t), non-parametric approaches
- **Designer 3:** May propose meta-regression, mixture models, or alternative model classes

**My contribution:**
- Baseline classical models that should be fit regardless
- Standard meta-analysis approaches from literature
- Rigorous falsification framework
- Foundation for comparison with more complex models

**If other models fit better:** That's success! It means we learned the data requires something beyond classical approaches.

---

## Critical Warnings

### 1. Funnel Geometry
With tau likely near zero, **use non-centered parameterization** from the start. Centered will have divergences.

### 2. Small Sample (J=8)
- Heterogeneity parameters hard to estimate precisely
- Results will be sensitive to individual studies
- Report uncertainty prominently
- Don't overinterpret small differences

### 3. Study 4 Influence
EDA shows 33% influence. **Always report with/without Study 4** to demonstrate robustness (or lack thereof).

### 4. Tau at Boundary
If tau → 0, this is **not a problem** - it means complete pooling is appropriate. Only worry if tau hits **upper** boundary.

### 5. Prior Sensitivity in Model 3
If skeptical prior dominates data, either:
- Data is too weak (likely with J=8)
- Or: Prior is inappropriately strong

Either way, report openly.

---

## Diagnostic Checklist

Before declaring success, verify:

- [ ] All models: Rhat < 1.01 for all parameters
- [ ] All models: ESS > 400 for mu (and tau if applicable)
- [ ] Model 2: No divergences or <1% with adapt_delta=0.99
- [ ] Model 2: Non-centered parameterization used
- [ ] LOO: All Pareto-k < 0.7 (or investigate outliers)
- [ ] PPC: Test statistics have p-values in [0.05, 0.95]
- [ ] Residuals: No systematic patterns
- [ ] Study 4 removal: Results don't change by >50%
- [ ] Prior sensitivity: Results robust to reasonable prior choices
- [ ] Model comparison: Clear winner or equivalence documented

---

## References

Key sources informing this design:

1. **DerSimonian & Laird (1986)** - Random effects meta-analysis
2. **Gelman & Hill (2007)** - Hierarchical modeling and partial pooling
3. **Betancourt (2018)** - Non-centered parameterization and HMC geometry
4. **Vehtari et al. (2017)** - LOO cross-validation for Bayesian models
5. **Higgins & Thompson (2002)** - I² statistic and heterogeneity quantification

Full references in `proposed_models.md`.

---

## Next Steps

1. **Review this README** for orientation
2. **Read `model_summary.md`** for quick reference
3. **Study `proposed_models.md`** for theoretical foundation (optional but recommended)
4. **Use `stan_templates.md`** for implementation
5. **Follow diagnostic checklist** before declaring completion
6. **Document failures** - they're the most valuable findings!

---

## Contact / Questions

This is an independent design. For questions about:
- **Classical meta-analysis theory:** See DerSimonian & Laird (1986), Hedges & Olkin (1985)
- **Bayesian hierarchical models:** See Gelman & Hill (2007), Betancourt (2018)
- **Stan implementation:** See Stan User's Guide, Chapter 9 (Hierarchical Models)
- **Computational issues:** See Betancourt (2017) on HMC geometry

---

**Design Philosophy:** Find truth by trying to break our models, not by confirming our expectations.

**Success Metric:** Discovering model failures early and pivoting quickly, not fitting models smoothly.

**Remember:** Switching model classes is success, not failure. It means you're learning.

---

**Files in this directory:**
- `README.md` (this file) - Navigation and overview
- `proposed_models.md` - Complete theoretical specification
- `model_summary.md` - Quick reference
- `stan_templates.md` - Implementation code

**All files are in:** `/workspace/experiments/designer_1/`
