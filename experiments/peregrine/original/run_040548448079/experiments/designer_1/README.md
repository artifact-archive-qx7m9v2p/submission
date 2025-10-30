# Model Designer 1: Changepoint and Regime-Switching Models

**Focus**: Discrete structural break modeling for time series count data
**Philosophy**: Falsification over confirmation; truth over task completion
**Date**: 2025-10-29

---

## Overview

This directory contains a complete experimental design for Bayesian changepoint and regime-switching models applied to time series count data (N=40) with a suspected structural break at observation 17.

**Key Question**: Is the observed 730% growth rate increase a true discrete regime change, or an artifact of smooth acceleration, autocorrelation, or measurement heterogeneity?

---

## Document Guide

### For Planning and Understanding

1. **experiment_plan.md** (PRIMARY DOCUMENT)
   - Complete experimental strategy
   - Problem formulation with competing hypotheses
   - Three model classes with falsification criteria
   - Decision tree for model selection
   - Validation strategy and timeline
   - **Read this first for overall strategy**

2. **proposed_models.md** (DETAILED SPECIFICATIONS)
   - Full mathematical specifications for 3 model classes
   - Prior recommendations with EDA justification
   - Extensive falsification criteria
   - Implementation considerations
   - Why each model might fail
   - Alternative model classes if changepoints fail
   - **Read this for technical details**

3. **model_summary.md** (QUICK REFERENCE)
   - One-page summary of three models
   - Decision rules and comparison criteria
   - Estimated runtimes
   - **Read this for quick lookup during implementation**

### For Implementation

4. **implementation_checklist.md** (EXECUTION GUIDE)
   - Step-by-step checklist for model fitting
   - Code templates (Stan and PyMC)
   - Validation procedures
   - Diagnostic thresholds
   - Common pitfalls and solutions
   - **Use this during actual model fitting**

---

## Three Model Classes

### Model 1: Fixed Changepoint (τ=17)
- **Assumption**: EDA is correct
- **Likelihood**: Negative Binomial with discrete break at t=17
- **Autocorrelation**: AR(1) latent process
- **Priority**: HIGH (start here)
- **Runtime**: 2-5 minutes
- **Tool**: Stan

### Model 2: Unknown Changepoint (τ ∈ [5,35])
- **Assumption**: Challenge EDA; data determines τ
- **Likelihood**: Same as Model 1 but τ is parameter
- **Priority**: MEDIUM (only if Model 1 raises concerns)
- **Runtime**: 10-30 minutes
- **Tool**: PyMC (better discrete parameter support)

### Model 3: Multiple Changepoints (k=2)
- **Assumption**: Single break is oversimplification
- **Likelihood**: Two ordered changepoints τ₁ < τ₂
- **Priority**: LOW (only if Models 1-2 suggest complexity)
- **Runtime**: 20-60 minutes
- **Tool**: PyMC

---

## Decision Flow

```
START
  │
  ├─ Fit Model 1 (τ=17 fixed)
  │   ├─ Passes all tests? → STOP, use Model 1
  │   └─ Fails on:
  │       ├─ β₂ not significant → Try Model 2 or abandon changepoints
  │       ├─ Autocorrelation → Strengthen AR structure
  │       └─ Computational issues → Simplify or abandon
  │
  ├─ Fit Model 2 (τ unknown)
  │   ├─ Posterior concentrated near τ=17? → Validates Model 1
  │   ├─ Posterior at different τ? → Investigate alternative location
  │   ├─ Posterior diffuse? → No clear break; try GP/splines
  │   └─ Posterior bimodal? → Try Model 3
  │
  └─ Fit Model 3 (multiple τ)
      ├─ k=1 preferred? → Revert to Model 2
      ├─ k=2 preferred? → Report with overfitting caveats
      └─ k=0 preferred? → Abandon changepoint framework
```

---

## Critical Falsification Criteria

**All models MUST pass**:
- Rhat < 1.01 for all parameters
- ESS > 100 (preferably >400) for all parameters
- Divergent transitions < 1%
- LOO Pareto k < 0.7 for ≥95% of observations
- Posterior predictive ACF(1) < 0.3
- Out-of-sample RMSE < 1.5x in-sample

**If ANY model fails these**: Document failure and either simplify or abandon approach.

---

## Implementation Requirements

### Software
- Stan (CmdStan, PyStan, or RStan) for Model 1
- PyMC ≥5.0 for Models 2 and 3
- ArviZ for diagnostics
- NumPy, SciPy, Pandas

### Data Format
```python
# Expected data structure
data = {
    'N': 40,                          # Number of observations
    'C': np.array([...]),            # Count observations
    'year': np.array([...]),         # Standardized year (-1.67 to 1.67)
}
```

### Key EDA Findings to Use
- Variance/mean ratio: 67.99 (strongly overdispersed)
- Structural break at observation 17 (year ≈ -0.21)
- Pre-break slope: ~0.35 on log scale
- Post-break slope: ~1.2 on log scale
- Strong autocorrelation: ACF(1) = 0.944
- Dispersion parameter α ≈ 0.6

---

## Expected Deliverables

### Code
- Stan files for Model 1 variants (1a, 1b, 1c)
- PyMC scripts for Models 2 and 3
- Validation scripts (prior predictive, SBC, posterior predictive, CV)

### Results
- Convergence diagnostics report
- Falsification test results (pass/fail table)
- Model comparison table (LOO, Bayes factors)
- Final recommendation document

### Visualizations
- Posterior distributions
- Trace plots
- Posterior predictive checks with data overlay
- Residual diagnostics
- Changepoint location posterior (if Model 2)

---

## Alternative Model Classes (If Changepoints Fail)

If all changepoint models fail falsification tests, immediately consider:

1. **Gaussian Process** (Matérn kernel)
   - Flexibly captures smooth and sharp transitions
   - No discrete changepoint assumption
   - Naturally handles autocorrelation

2. **Bayesian Structural Time Series**
   - Local linear trend with time-varying slopes
   - State-space formulation
   - Handles non-stationarity better than AR(1)

3. **Spline Models** (Natural cubic or P-splines)
   - Smooth but regularized
   - No explicit break needed
   - Interpretable basis functions

4. **State-Space Models**
   - Dynamic Linear Model with time-varying coefficients
   - Captures gradual and sudden changes
   - Natural framework for forecasting

**When to pivot**: If 2+ changepoint models fail core tests (especially computational or predictive), immediately propose these alternatives rather than forcing changepoint framework.

---

## Philosophy and Commitments

### Core Principles

1. **Falsification over Confirmation**
   - Actively try to break models
   - Report all failures, not just successes
   - Abandoning a model is scientific progress

2. **Truth over Task Completion**
   - Goal is understanding reality, not fitting a model
   - "We don't know" is a valid conclusion
   - Negative results are valuable

3. **Adaptive Strategy**
   - Follow decision tree rigorously
   - Pivot when evidence demands it
   - Don't force failing models to "work"

4. **Honest Uncertainty**
   - Quantify what we don't know
   - Acknowledge limitations prominently
   - Avoid overconfident claims

### Success Criteria

**Success is NOT**:
- Getting a model to converge
- Producing nice plots
- Confirming hypotheses

**Success IS**:
- Determining whether changepoint hypothesis is supported
- Quantifying uncertainty honestly
- Discovering when model classes are inadequate
- Providing clear recommendations (even if "this approach doesn't work")

---

## Timeline Estimate

| Phase | Task | Hours |
|-------|------|-------|
| 1 | Prior predictive checks | 2-3 |
| 2 | Simulation-based calibration | 2-3 |
| 3 | Fit Model 1 (all variants) | 2-4 |
| 4 | Validate Model 1 | 3-5 |
| 5 | Fit Model 2 (if needed) | 4-8 |
| 6 | Validate Model 2 (if needed) | 3-5 |
| 7 | Fit Model 3 (if needed) | 6-12 |
| 8 | Model comparison | 2-4 |
| 9 | Reporting and visualization | 2-4 |

**Total**: 20-40 hours depending on which models are attempted

**Bottlenecks**:
- Models 2 and 3 computational costs
- Debugging convergence issues (can add 5-10 hours)
- Comprehensive validation suite (intentionally thorough)

---

## File Structure

```
/workspace/experiments/designer_1/
├── README.md                          # This file
├── experiment_plan.md                 # Complete strategy (READ FIRST)
├── proposed_models.md                 # Detailed specifications
├── model_summary.md                   # Quick reference
├── implementation_checklist.md        # Step-by-step execution guide
│
├── models/                            # (To be created during implementation)
│   ├── model1a_random_effects.stan
│   ├── model1b_ar1.stan
│   ├── model1c_ar1_timevarying.stan
│   ├── model2_unknown_tau.py
│   └── model3_multiple_tau.py
│
├── validation/                        # (To be created)
│   ├── prior_predictive.py
│   ├── sbc.py
│   ├── posterior_predictive.py
│   └── cross_validation.py
│
├── results/                           # (To be created)
│   ├── convergence_diagnostics.txt
│   ├── falsification_results.md
│   ├── model_comparison.csv
│   ├── final_report.md
│   └── figures/
│       ├── posteriors.png
│       ├── traces.png
│       ├── posterior_predictive.png
│       └── residuals.png
│
└── data/                              # (Link to main data directory)
    └── processed_data.csv
```

---

## Getting Started

### Quick Start (First Time)

1. Read `/workspace/experiments/designer_1/experiment_plan.md` (15-20 minutes)
2. Read `/workspace/experiments/designer_1/proposed_models.md` (30-40 minutes)
3. Review `/workspace/experiments/designer_1/implementation_checklist.md`
4. Verify software installation (Stan, PyMC, ArviZ)
5. Begin Phase 1: Prior predictive checks

### Resume Work

1. Open `/workspace/experiments/designer_1/implementation_checklist.md`
2. Find last completed checkbox
3. Continue from next step
4. Update checklist as you progress

---

## Key Contacts and Collaboration

**Model Designer 1**: Changepoint/Regime-Switching Specialist
- Specialty: Discrete structural breaks, Bayesian changepoint detection
- Philosophy: Falsification mindset

**Coordination with Other Designers**:
- Will provide LOO scores for inter-designer comparison
- Open to model averaging if multiple classes viable
- Committed to ensemble if no single model dominates

**When to Escalate**:
- All models failing despite simplification attempts
- Computational issues that can't be resolved
- Domain knowledge suggests approach is fundamentally wrong
- Need for completely different model classes (GP, splines, etc.)

---

## Citation and Acknowledgment

This model design follows principles from:
- Gelman et al. "Bayesian Workflow" (2020)
- Vehtari et al. "Practical Bayesian model evaluation using LOO-CV" (2017)
- Taleb & Goldstein "The problem is beyond psychology: The real world is more random than regression analyses" (2020)
- McElreath "Statistical Rethinking" (2nd ed, 2020)

**Philosophy**: Models are tools for understanding, not truth. Our goal is to discover when they break, not to prove they work.

---

## Version History

- **v1.0** (2025-10-29): Initial model design
  - Three changepoint model classes proposed
  - Comprehensive falsification criteria defined
  - Decision tree and validation strategy specified
  - Implementation checklist created

---

## Questions or Issues?

**Before proceeding, ensure**:
- [ ] You understand the competing hypotheses (H1, H2, H3)
- [ ] You've reviewed all falsification criteria
- [ ] You have Stan and PyMC properly installed
- [ ] You've read the EDA report at `/workspace/eda/eda_report.md`
- [ ] You're prepared to abandon models that fail tests

**If uncertain about**:
- Model specifications → See `proposed_models.md`
- Implementation steps → See `implementation_checklist.md`
- Decision points → See `experiment_plan.md` decision tree
- Validation procedures → See `experiment_plan.md` validation section

**Remember**: The goal is finding truth, not completing tasks. If the changepoint framework is wrong, discovering that is success, not failure.

---

**END OF README**

*Model Designer 1 (Changepoint and Regime-Switching Focus)*
*Generated: 2025-10-29*
*Status: Ready for implementation*
