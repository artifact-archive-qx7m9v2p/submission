# Designer 1 Files Index

**Designer:** Parametric Bayesian GLM Focus
**Date:** 2025-10-29
**Status:** Complete, ready for implementation
**Total:** 5 files, 2,576 lines, 92 KB

---

## File Navigation Guide

### Start Here
1. **SUMMARY.txt** (14 KB, 389 lines) - Executive summary, read this FIRST for quick overview
2. **README.md** (11 KB, 377 lines) - Comprehensive overview with navigation guide

### For Implementation
3. **implementation_guide.md** (8.8 KB, 342 lines) - Practical guide with checklists and decision trees
4. **stan_templates.txt** (13 KB, 483 lines) - Complete Stan code + Python/R wrappers, COPY-PASTE READY

### For Theory
5. **proposed_models.md** (32 KB, 985 lines) - Complete mathematical specifications, priors, falsification criteria

---

## Quick Access by Purpose

**Need quick overview?**
→ Read `SUMMARY.txt` (5 minutes)

**Want to understand approach?**
→ Read `README.md` (15 minutes)

**Ready to implement?**
→ Use `implementation_guide.md` + `stan_templates.txt` (copy code, run)

**Need theoretical details?**
→ Read `proposed_models.md` (full specifications)

**Want everything?**
→ Read in order: SUMMARY.txt → README.md → proposed_models.md → implementation_guide.md

---

## Three Proposed Models

### Model 1: Negative Binomial Quadratic (BASELINE - START HERE)
- **File:** Lines 15-224 in `proposed_models.md`, Lines 93-154 in `stan_templates.txt`
- **Why:** Handles overdispersion + acceleration
- **Priority:** HIGHEST - Fit this first
- **Expected:** Works reasonably, shows residual ACF 0.6-0.8

### Model 2: Negative Binomial Exponential (SIMPLER ALTERNATIVE)
- **File:** Lines 226-356 in `proposed_models.md`, Lines 156-214 in `stan_templates.txt`
- **Why:** Tests exponential growth hypothesis
- **Priority:** MEDIUM - Fit second, compare to Model 1
- **Expected:** Slightly worse than Model 1

### Model 3: Quasi-Poisson with Random Effects (FLEXIBLE)
- **File:** Lines 358-490 in `proposed_models.md`, Lines 216-313 in `stan_templates.txt`
- **Why:** Allows time-varying dispersion
- **Priority:** LOW - Only if Models 1/2 show issues
- **Expected:** May overfit with n=40

---

## Key Sections by File

### SUMMARY.txt
- Lines 1-80: Three models overview
- Lines 82-110: EDA findings driving design
- Lines 112-150: Success/failure criteria
- Lines 152-180: Implementation roadmap
- Lines 182-210: Most likely outcome
- Lines 330-360: Quick start instructions

### README.md
- Lines 1-50: Quick summary
- Lines 52-80: File navigation
- Lines 82-110: Key challenges
- Lines 112-160: Model specifications
- Lines 162-200: Implementation roadmap
- Lines 202-250: Expected outcomes
- Lines 252-300: Integration with other designers

### proposed_models.md
- Lines 1-50: Executive summary
- Lines 52-300: Model 1 complete specification
- Lines 302-490: Model 2 complete specification
- Lines 492-680: Model 3 complete specification
- Lines 682-780: Model comparison strategy
- Lines 782-850: Prior predictive checks
- Lines 852-900: Implementation roadmap
- Lines 902-985: Falsification summary

### implementation_guide.md
- Lines 1-50: Three models at a glance
- Lines 52-100: Critical diagnostics checklist
- Lines 102-150: Decision tree
- Lines 152-180: Expected results table (to fill)
- Lines 182-210: Priors quick reference
- Lines 212-240: Red flags
- Lines 242-280: Code snippets
- Lines 282-320: Integration with designers

### stan_templates.txt
- Lines 1-92: Instructions
- Lines 93-154: Model 1 Stan code
- Lines 156-214: Model 2 Stan code
- Lines 216-313: Model 3 Stan code (2 versions)
- Lines 315-380: Python wrapper example
- Lines 382-430: R wrapper example
- Lines 432-465: Prior predictive check code
- Lines 467-483: Model comparison code

---

## Absolute File Paths

All files located in:
```
/workspace/experiments/designer_1/
```

Complete paths:
- `/workspace/experiments/designer_1/SUMMARY.txt`
- `/workspace/experiments/designer_1/README.md`
- `/workspace/experiments/designer_1/proposed_models.md`
- `/workspace/experiments/designer_1/implementation_guide.md`
- `/workspace/experiments/designer_1/stan_templates.txt`
- `/workspace/experiments/designer_1/INDEX.md` (this file)

---

## Key Findings Summary

**From EDA that drive model design:**
1. Extreme overdispersion: Var/Mean = 68 → Need negative binomial
2. Strong non-linearity: Quadratic R² = 0.961 → Need polynomial/exponential terms
3. Accelerating growth: 6× rate increase → Quadratic likely better
4. High temporal autocorrelation: lag-1 = 0.989 → BIGGEST CHALLENGE
5. Time-varying variance: Q3 extreme → Model 3 addresses this

**Expected outcome:**
- Model 1 fits reasonably but shows residual ACF 0.6-0.8
- Temporal correlation requires AR structure or state-space models
- Parametric trend good, but temporal dynamics need Designer 2

---

## Success Criteria

**Succeed if:**
- Coverage 85-98%
- Residual ACF(1) < 0.6
- Clear LOO-IC winner
- No systematic patterns

**Adequate if:**
- Coverage 75-85%
- Residual ACF(1) 0.6-0.8
- Can extend with AR errors

**Failed if (any 2 of):**
- All models: ACF > 0.80
- All models: Coverage < 75%
- All models: Systematic patterns
- All models: LOO-IC within 3 points
- All models: Out-sample RMSE > 50

---

## Implementation Time Budget

- Phase 1 (Fitting): 15 minutes
- Phase 2 (Diagnostics): 75 minutes
- Phase 3 (Validation): 25 minutes
- Phase 4 (Comparison): 45 minutes
- **Total: ~2.5 hours**

---

## Integration Points

**To Designer 2 (Non-Parametric/State-Space):**
- Residual ACF values
- Evidence for temporal correlation
- Baseline performance

**To Designer 3 (Hierarchical/Temporal):**
- Dispersion by period
- Evidence for changepoints
- Time-varying variance

---

## Most Important Takeaways

1. **Start with Model 1** (NB Quadratic) - highest priority
2. **Check residual ACF** - will likely be 0.6-0.8 (temporal correlation issue)
3. **Parametric GLMs good for trend** - but may need AR structure
4. **Success = learning** - even if models "fail", we learn what doesn't work
5. **Know when to pivot** - clear criteria for switching to other approaches

---

## Quick Decision Flow

```
Fit Model 1
    ↓
Check MCMC converged?
    ↓ YES
Check coverage > 85%?
    ↓ YES
Check residual ACF(1) < 0.7?
    ↓
    YES → SUCCESS, use Model 1
    NO → Temporal correlation issue
         → Try AR extensions or Designer 2
```

---

## File Relationships

```
SUMMARY.txt ──→ Quick overview
    ↓
README.md ──→ Detailed overview
    ↓
proposed_models.md ──→ Full theory
    ↓
implementation_guide.md ──→ Practical guide
    ↓
stan_templates.txt ──→ Actual code
```

**Reading path for understanding:** SUMMARY → README → proposed_models
**Reading path for implementation:** implementation_guide → stan_templates

---

## Version Information

- **Created:** 2025-10-29
- **Designer:** Designer 1 (Parametric GLM Focus)
- **Data:** `/workspace/data/data.csv` (40 observations)
- **EDA Report:** `/workspace/eda/eda_report.md`
- **Stan/PyMC:** Both supported (templates provided)

---

## Next Steps

1. **Read SUMMARY.txt** for quick orientation (5 min)
2. **Read implementation_guide.md** for practical guidance (15 min)
3. **Copy Stan code** from stan_templates.txt
4. **Fit Model 1** and run diagnostics
5. **Make decision** based on criteria
6. **Report results** to main experiment coordinator

---

**Document prepared by:** Designer 1
**Purpose:** Navigation and quick reference
**Status:** Complete
**All files ready for implementation**
