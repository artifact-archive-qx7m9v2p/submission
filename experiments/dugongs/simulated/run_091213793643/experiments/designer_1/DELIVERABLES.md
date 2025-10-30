# Designer 1 Deliverables: Parametric Modeling Proposal

## Status: COMPLETE

All deliverables for Model Designer 1 (Parametric Modeling Perspective) have been completed and are ready for review and implementation.

---

## Deliverable Checklist

- [x] **Comprehensive proposal document** with scientific justification
- [x] **Three distinct parametric model classes** proposed
- [x] **Complete prior specifications** informed by EDA
- [x] **Explicit falsification criteria** for each model
- [x] **Stan model implementations** ready to run
- [x] **Model comparison strategy** defined
- [x] **Sensitivity analyses** planned
- [x] **Red flags and stopping rules** documented
- [x] **Quick reference guides** for rapid understanding

---

## File Structure

```
/workspace/experiments/designer_1/
├── README.md                          # Overview and quick start guide
├── DELIVERABLES.md                    # This file
├── proposed_models.md                 # MAIN DOCUMENT (15,000+ words)
├── model_summary.md                   # Quick reference (1 page)
├── model_equations.md                 # Mathematical specifications
└── stan_models/
    ├── logarithmic_model.stan         # Model 1: Y = α + β·log(x)
    ├── michaelis_menten_model.stan    # Model 2: Asymptotic saturation
    └── quadratic_model.stan           # Model 3: Polynomial fit
```

---

## Document Purposes

### 1. proposed_models.md (PRIMARY)
**Length**: ~15,000 words
**Purpose**: Complete modeling strategy
**Contents**:
- Executive summary
- Three model classes with full specifications
- Prior elicitation with justifications
- Falsification criteria for each model
- Computational implementation strategy
- Model comparison approach
- Sensitivity analyses
- Red flags and stopping rules
- Expected outcomes and decision trees

**Audience**: Implementers, reviewers, domain experts

### 2. model_summary.md
**Length**: ~1 page
**Purpose**: Quick reference for decision-making
**Contents**:
- One-sentence description of each model
- Key falsification criteria table
- Expected outcome summary
- Red flags checklist
- Priority order

**Audience**: Quick consultation during implementation

### 3. model_equations.md
**Length**: ~3 pages
**Purpose**: Mathematical reference
**Contents**:
- Functional forms for all three models
- Prior specifications
- Limiting behavior analysis
- Derivatives and curvature
- Predicted values at key x locations
- Falsification test statistics

**Audience**: Mathematical/statistical reviewers

### 4. README.md
**Length**: ~2 pages
**Purpose**: Navigation and overview
**Contents**:
- Directory structure
- Core philosophy
- File descriptions
- Quick start guide
- Contact information

**Audience**: New readers, navigators

### 5. Stan Models (3 files)
**Purpose**: Direct implementation
**Contents**:
- Executable Stan code
- Comments explaining all components
- Generated quantities for diagnostics
- Prior specifications matching proposal

**Audience**: MCMC implementers

---

## Three Proposed Models

### Model 1: Logarithmic (PRIMARY)
**Form**: `Y = α + β·log(x) + ε`
**Priority**: 1 (fit first)
**Rationale**: Best balance of fit, parsimony, and plausibility
**Key assumption**: Unbounded slow growth (Weber-Fechner law)
**Abandon if**: LOO-ELPD worse than MM by >4

**Expected outcome**: Most likely winner

### Model 2: Michaelis-Menten (ALTERNATIVE)
**Form**: `Y = Y_max - (Y_max - Y_min)·K/(K + x) + ε`
**Priority**: 2 (fit second)
**Rationale**: Tests fundamental hypothesis (asymptote exists?)
**Key assumption**: Finite upper limit Y_max
**Abandon if**: Y_max posterior unbounded or K > 25

**Expected outcome**: Competitive but Y_max weakly identified

### Model 3: Quadratic (BASELINE)
**Form**: `Y = α + β₁·x + β₂·x² + ε`
**Priority**: 3 (fit third, maybe skip)
**Rationale**: Flexible empirical fit, comparison baseline
**Key assumption**: Polynomial approximation (local only)
**Abandon if**: Vertex at x < 31.5 (predicts downturn)

**Expected outcome**: Best ELPD but extrapolation concerns

---

## Key Innovations in This Proposal

### 1. Competing Hypotheses Framework
Rather than sequential model building, proposed **three fundamentally different hypotheses** about the data-generating process:
- H1: Unbounded logarithmic growth
- H2: Asymptotic saturation
- H3: Polynomial trajectory

These represent **mutually exclusive limiting behaviors**.

### 2. Explicit Falsification Criteria
Each model has **specific, measurable conditions** that would lead to its rejection:
- Not vague ("if it doesn't fit well")
- But precise ("if LOO-ELPD difference > 4")
- Includes both statistical and scientific criteria

### 3. Prior Predictive Planning
All priors are:
- **Weakly informative** (not flat, not too strong)
- **Informed by EDA** (centered on observed values)
- **Scientifically constrained** (e.g., β > 0 for monotonicity)
- **Numerically specified** (no hand-waving)

### 4. Computational Realism
Anticipated challenges for each model:
- **Log**: None (simple)
- **MM**: Possible correlation between Y_max and K, divergences
- **Quad**: None (simple)

Included backup plans (reparameterization, adapt_delta adjustments).

### 5. Multi-Level Comparison Strategy
Not just "which fits best?" but:
- **Stage 1**: Within-model diagnostics (MCMC, PPC)
- **Stage 2**: Between-model comparison (LOO, WAIC)
- **Stage 3**: Scientific plausibility (extrapolation, interpretation)
- **Stage 4**: Sensitivity analysis (influence, priors)

### 6. Planned Pivots
Explicit conditions for **abandoning the entire parametric approach**:
- All models fail diagnostics → Robust likelihood needed
- High Pareto k across models → Influential points dominate
- Gap region deviation → Piecewise models needed
- Complex residual patterns → Non-parametric approach

### 7. Embracing Uncertainty
If models are close (ΔLOO < 2), the recommendation is:
- **Use Bayesian model averaging**
- **Acknowledge structural uncertainty**
- **Report: "Data insufficient to distinguish mechanisms"**

This is presented as **success** (learned the limits of data), not failure.

---

## Alignment with Requirements

### Requirement: 2-3 distinct model classes
**Delivered**: 3 model classes (log, MM, quadratic)
**Status**: ✓ Complete

### Requirement: Mathematical specifications
**Delivered**: Full likelihood + priors for each model
**Status**: ✓ Complete

### Requirement: Theoretical justification
**Delivered**: Scientific context for each (psychophysics, kinetics, empirical)
**Status**: ✓ Complete

### Requirement: Falsification criteria
**Delivered**: Explicit conditions for rejecting each model
**Status**: ✓ Complete

### Requirement: Computational considerations
**Delivered**: Stan vs PyMC, expected challenges, diagnostics
**Status**: ✓ Complete

### Requirement: Stan/PyMC implementations
**Delivered**: 3 Stan models with complete specifications
**Status**: ✓ Complete

### Requirement: Prior elicitation from EDA
**Delivered**: All priors informed by EDA statistics (means, SDs, ranges)
**Status**: ✓ Complete

### Requirement: Weakly informative priors
**Delivered**: All priors centered at reasonable values, wide enough for data to dominate
**Status**: ✓ Complete

### Requirement: Success/failure criteria
**Delivered**: Multi-stage evaluation with specific thresholds
**Status**: ✓ Complete

---

## Expected Timeline

### Phase 1: Prior Predictive Checks
**Time**: ~30 minutes
**Activities**: Sample from priors, verify reasonable ranges

### Phase 2: MCMC Fitting
**Time**: ~1-5 minutes total (all three models)
**Activities**: Compile and run Stan models

### Phase 3: Diagnostics
**Time**: ~30 minutes
**Activities**: Check Rhat, ESS, divergences, pairs plots

### Phase 4: Posterior Predictive Checks
**Time**: ~45 minutes
**Activities**: Generate y_rep, compare to y_obs, residual analysis

### Phase 5: Model Comparison
**Time**: ~15 minutes
**Activities**: Compute LOO, compare ELPD, assess Pareto k

### Phase 6: Sensitivity Analysis
**Time**: ~30 minutes
**Activities**: Refit without x=31.5, check prior sensitivity

**Total**: ~2.5-3 hours for complete analysis

---

## Success Metrics

### Quantitative
- [ ] All models converge (Rhat < 1.01)
- [ ] Effective sample sizes > 400
- [ ] Divergent transitions < 1% (for MM model)
- [ ] LOO computed successfully (Pareto k < 0.7 for most points)
- [ ] Clear ranking of models by LOO-ELPD

### Qualitative
- [ ] Best model passes posterior predictive checks
- [ ] Parameter estimates scientifically plausible
- [ ] Predictions in gap region reasonable
- [ ] Extrapolation behavior makes sense
- [ ] Can tell coherent scientific story

### Meta
- [ ] Understand why rejected models failed
- [ ] Documented decision process transparently
- [ ] Acknowledged limitations and uncertainties
- [ ] Identified areas where more data needed

---

## Risk Assessment

### Low Risk
- **MCMC convergence for log and quadratic**: Simple models, should be clean
- **Prior-data agreement**: Priors informed by EDA, conflict unlikely
- **Computational time**: All models should run in seconds

### Medium Risk
- **MM model identifiability**: Y_max may have wide posterior with limited high-x data
- **MM divergent transitions**: Possible due to parameter correlations
- **Gap region predictions**: Large uncertainty at x∈[23, 29]

### High Risk
- **Model distinguishability**: Only 3% R² difference from EDA; may be ΔLOO < 2
- **Influential point sensitivity**: x=31.5 may drive conclusions
- **Extrapolation disagreement**: Models may diverge strongly beyond data range

**Mitigation**: All risks explicitly planned for with sensitivity analyses and alternative strategies.

---

## Recommendations for Implementation

### Do's
1. **Start with log model** - Most likely to succeed
2. **Check diagnostics immediately** - Don't wait for all models
3. **Run prior predictive first** - Catch problems early
4. **Save all outputs** - Traces, diagnostics, plots
5. **Document decisions** - Why each choice was made

### Don'ts
1. **Don't skip prior predictive checks** - May reveal prior-data conflicts
2. **Don't ignore divergences** - Even a few indicate problems
3. **Don't compare only ELPD** - Also check scientific plausibility
4. **Don't force a winner** - Model averaging if close
5. **Don't extrapolate quadratic** - Use only for interpolation

---

## Questions This Proposal Answers

1. **What models to fit?** → Log, MM, quadratic
2. **In what order?** → Log first, MM second, quadratic third
3. **What priors?** → Specific numerical priors for each parameter
4. **How to compare?** → LOO-CV, then PPC, then scientific plausibility
5. **When to stop?** → When one model clearly superior OR all fail
6. **What if models are close?** → Use Bayesian model averaging
7. **What if all fail?** → Escalate to non-parametric approaches
8. **How to handle influential point?** → Sensitivity analysis without x=31.5
9. **How to assess extrapolation?** → Predict at x=50, 100; check plausibility
10. **What's the expected outcome?** → Log model likely wins

---

## Integration with Other Designers

This proposal is **independent** but complementary:

- **Designer 2** (likely non-parametric): Will propose GP or splines
  - Comparison: Parametric vs non-parametric
  - If parametric adequate → Prefer for interpretability
  - If parametric fails → GP provides flexible alternative

- **Designer 3** (if exists): May propose hierarchical or robust models
  - Comparison: Normal vs heavy-tailed errors
  - Standard vs robust approach

**Synthesis**: Best model may combine insights from multiple designers

---

## Contact and Attribution

**Designer**: Model Designer 1
**Perspective**: Parametric regression with explicit functional forms
**Philosophy**: Falsificationist - plan for failure, embrace uncertainty
**Date**: 2025-10-28

**Key Principle**: *Finding truth > completing tasks*

---

## Final Checklist Before Implementation

- [ ] Read `README.md` for overview
- [ ] Read `model_summary.md` for quick reference
- [ ] Read `proposed_models.md` sections 1-5 (models + comparison)
- [ ] Examine Stan models in `stan_models/`
- [ ] Understand falsification criteria for each model
- [ ] Review prior specifications and justifications
- [ ] Note red flags and stopping rules
- [ ] Prepare to pivot if evidence suggests it

**Status**: READY FOR IMPLEMENTATION

---

**All deliverables complete. Awaiting approval to proceed with MCMC fitting.**
