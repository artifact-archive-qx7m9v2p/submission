# Designer 1: Parametric Bayesian GLM Models

**Focus:** Parametric GLM approach with explicit trend functions and standard distributions
**Date:** 2025-10-29
**Status:** Design complete, ready for implementation

---

## Quick Summary

I propose **3 parametric Bayesian GLM models** to address the key challenges in this time series count data:

1. **Negative Binomial with Quadratic Trend** (BASELINE)
   - Handles overdispersion (Var/Mean = 68) + acceleration
   - Most likely to succeed
   - Start here

2. **Negative Binomial with Exponential Trend** (SIMPLER ALTERNATIVE)
   - Tests exponential growth hypothesis
   - More parsimonious (3 vs 4 parameters)
   - Compare to Model 1

3. **Quasi-Poisson with Observation-Level Random Effects** (FLEXIBLE)
   - Allows time-varying dispersion
   - More complex, use if Models 1/2 show inadequacy
   - May overfit with n=40

---

## Files in This Directory

### Core Documents
- **`proposed_models.md`** (32 KB, 985 lines)
  - Complete model specifications with math, priors, justifications
  - Falsification criteria for each model
  - Expected strengths and weaknesses
  - **READ THIS FIRST for full details**

- **`implementation_guide.md`** (8.8 KB, 342 lines)
  - Quick reference for implementation
  - Diagnostic checklist
  - Decision tree
  - Success/failure criteria
  - **READ THIS for quick start**

- **`stan_templates.txt`** (13 KB, 483 lines)
  - Complete Stan code for all 3 models
  - Python and R wrapper examples
  - Ready to copy-paste and run
  - **USE THIS for actual implementation**

- **`README.md`** (this file)
  - Overview and navigation guide

---

## Key Data Challenges (from EDA)

1. **Extreme overdispersion:** Variance/Mean = 68 (Poisson expects 1.0)
2. **Strong non-linearity:** Quadratic R² = 0.961 vs Linear R² = 0.885
3. **Accelerating growth:** 6× rate increase from early to late period
4. **High autocorrelation:** lag-1 r = 0.989 (major concern!)
5. **Time-varying variance:** Q3 shows Var/Mean = 13.5 vs others 1.6-2.4

---

## What My Models Address

### Strengths (What Parametric GLMs Do Well)
- Handle overdispersion via negative binomial distribution
- Capture non-linear trends (quadratic, exponential)
- Interpretable parameters (growth rate, acceleration, dispersion)
- Computationally efficient
- Clear hypothesis testing

### Weaknesses (What Parametric GLMs Cannot Do)
- **Cannot handle high temporal correlation** (lag-1 = 0.989 is huge!)
  - Will underestimate uncertainty
  - Residuals will show autocorrelation
  - This is my biggest limitation
- Assume constant dispersion (Model 3 addresses this)
- Assume fixed functional form
- No temporal dynamics or state evolution

---

## Critical Decision Points

### I will declare parametric GLMs SUCCESSFUL if:
- Posterior predictive coverage: 85-98%
- Residual ACF(1): < 0.6
- Clear winner in LOO-IC comparison
- No systematic residual patterns

### I will declare parametric GLMs INADEQUATE if:
**Any 2 of the following:**
1. All models show residual ACF(1) > 0.80
2. All models have coverage < 75%
3. All models show systematic residual patterns
4. LOO-IC differences < 3 (models equally bad)
5. Out-of-sample RMSE > 50

**Action if inadequate:** Recommend Designer 2's state-space/AR models or Designer 3's hierarchical temporal models

---

## Model Specifications Summary

### Model 1: Negative Binomial Quadratic
```
C ~ NegBinomial(mu, phi)
log(mu) = β₀ + β₁·year + β₂·year²

Priors:
  β₀ ~ Normal(4.7, 0.5)    # log(109) ≈ 4.7
  β₁ ~ Normal(0.8, 0.3)    # Growth rate
  β₂ ~ Normal(0.3, 0.2)    # Acceleration
  phi ~ Gamma(2, 0.5)      # Dispersion
```

**Reject if:** Residual ACF > 0.9 OR coverage < 50%

### Model 2: Negative Binomial Exponential
```
C ~ NegBinomial(mu, phi)
log(mu) = β₀ + β₁·year

Priors:
  β₀ ~ Normal(4.7, 0.5)
  β₁ ~ Normal(0.85, 0.2)   # EDA exponential fit
  phi ~ Gamma(2, 0.5)
```

**Reject if:** LOO-IC worse than Model 1 by > 10 points

### Model 3: Quasi-Poisson with Random Effects
```
C ~ Poisson(mu · exp(ε))
log(mu) = β₀ + β₁·year + β₂·year²
ε ~ Normal(0, σ)

Priors:
  β₀, β₁, β₂: Same as Model 1
  σ ~ Exponential(1)
```

**Reject if:** Divergent transitions OR ε shows strong autocorrelation

---

## Implementation Roadmap

### Phase 1: Fitting (15 minutes)
1. Fit Model 1 (NB Quadratic) - PRIORITY
2. Fit Model 2 (NB Exponential)
3. Fit Model 3 (Quasi-Poisson) - only if needed

### Phase 2: Diagnostics (75 minutes)
For each model:
- Check MCMC health (Rhat, ESS, divergences)
- Posterior predictive checks
- Residual analysis (ACF, plots)
- LOO-CV computation

### Phase 3: Validation (25 minutes)
- Fit on 80% of data
- Predict last 20%
- Compute RMSE, coverage

### Phase 4: Comparison (45 minutes)
- Create comparison table
- Make decision
- Write summary

**Total estimated time: ~2.5 hours**

---

## Expected Outcomes

### Most Likely Scenario
- Model 1 (NB Quadratic) fits reasonably well
- Captures acceleration and overdispersion
- **But:** Shows residual autocorrelation 0.6-0.8
- **Conclusion:** Parametric trend is good, but needs AR structure
- **Recommendation:** Hybrid approach (parametric trend + AR errors) or handoff to Designer 2

### Best Case Scenario
- One model has coverage > 90%, residual ACF < 0.5
- Clear winner in LOO-IC
- Interpretable parameters
- **Conclusion:** Parametric GLM sufficient!
- **Recommendation:** Use as final model

### Worst Case Scenario
- All models show residual ACF > 0.8
- All models have coverage < 75%
- No clear winner (LOO-IC within 3 points)
- **Conclusion:** Temporal correlation dominates, parametric GLMs insufficient
- **Recommendation:** Switch to Designer 2's state-space models entirely

---

## Integration with Other Designers

### To Designer 2 (Non-Parametric / State-Space)
**I will provide:**
- Residual ACF values from all models
- Whether parametric trends capture signal
- Evidence for time-varying parameters
- Baseline performance to beat

**Questions for Designer 2:**
- Can AR models reduce residual autocorrelation?
- Is GP regression better than parametric trends?
- Are state-space models necessary?

### To Designer 3 (Hierarchical / Temporal)
**I will provide:**
- Dispersion estimates by time period
- Evidence for changepoints
- Time-varying variance patterns

**Questions for Designer 3:**
- Should periods be modeled separately?
- Are changepoint models warranted?
- Is hierarchical structure by period needed?

---

## Key Insights from Design Process

### What I Learned from EDA
1. **Overdispersion is extreme (68x):** Negative binomial is mandatory
2. **Quadratic outperforms exponential (AIC: 232 vs 254):** Acceleration matters
3. **Temporal correlation is huge (0.989):** This will be my biggest challenge
4. **Variance changes over time:** Q3 is very different from other periods

### Design Principles I Applied
1. **Falsification mindset:** Explicit criteria for when to reject each model
2. **Multiple hypotheses:** Polynomial vs exponential vs flexible dispersion
3. **Practical constraints:** Simple enough for n=40, complex enough to capture structure
4. **Escape routes:** Clear decision rules for when to pivot to other approaches

### Critical Decisions I Made
1. **Start with negative binomial:** Poisson cannot handle overdispersion
2. **Test quadratic vs exponential:** Both plausible from EDA
3. **Include flexible dispersion option:** Q3 period suggests time-varying variance
4. **Use weakly informative priors:** Centered on EDA but allowing flexibility
5. **Plan for temporal correlation:** Know it's an issue, but test parametric first

---

## Red Flags to Watch For

### During Fitting
- Divergent transitions (especially Model 3)
- Rhat > 1.01 or ESS < 400
- Extreme parameter values (beta_2 < -1 or phi near 0)

### During Diagnostics
- Residual ACF(1) > 0.8 → Temporal correlation dominates
- Coverage < 70% → Distributional misspecification
- Systematic residual patterns → Wrong functional form
- LOO Pareto k > 0.7 for many points → Overfitting

### During Comparison
- All models similar LOO-IC (<3 difference) → Equally inadequate
- Model 3 much worse → Overfitting to noise
- All models fail in same way → Wrong model class

---

## Success Metrics

### Good Parametric GLM
- Coverage: 85-98%
- Residual ACF(1): < 0.6
- Clear best model (LOO-IC difference > 5)
- Out-of-sample RMSE < 35
- No systematic residual patterns

### Adequate Parametric GLM
- Coverage: 75-85%
- Residual ACF(1): 0.6-0.8
- Can be improved with extensions (AR errors, time-varying dispersion)

### Failed Parametric GLMs
- Coverage < 75%
- Residual ACF(1) > 0.8
- No clear winner
- Systematic patterns remain

---

## Files to Generate During Implementation

### Required
1. `model1_fit.pkl` (or .rds) - Fitted Stan object
2. `model2_fit.pkl`
3. `diagnostics_report.md` - Full diagnostic results
4. `model_comparison_table.csv` - LOO-IC, coverage, etc.
5. `posterior_predictive_plots.png` - Visual checks
6. `residual_plots.png` - ACF and residual diagnostics

### Recommended
7. `prior_predictive_check.png` - Before fitting
8. `loo_comparison.csv` - Detailed LOO results
9. `trace_plots.png` - MCMC diagnostics
10. `parameter_summaries.csv` - Posterior summaries

---

## Contact Points with Main Experiment

**My parametric GLMs feed into:**
- Model comparison with other designers
- Baseline performance metrics
- Evidence for/against parametric approaches
- Parameter estimates for interpretation

**I need from main experiment:**
- Decision on which models to prioritize
- Computational resources for fitting
- Feedback on whether to pursue extensions

**I provide to main experiment:**
- Clear baseline with interpretable parameters
- Evidence for temporal correlation importance
- Recommendations for next steps

---

## Quick Start Guide

**To get started immediately:**

1. Read `implementation_guide.md` (10 minutes)
2. Copy Stan code from `stan_templates.txt`
3. Fit Model 1 first (highest priority)
4. Run diagnostics checklist from implementation guide
5. Compare to Model 2
6. Make decision based on decision tree

**Critical files:**
- Start: `implementation_guide.md`
- Details: `proposed_models.md`
- Code: `stan_templates.txt`

---

## Final Thoughts

**My role as Designer 1:**
- Provide interpretable baseline with parametric GLMs
- Test specific hypotheses (polynomial vs exponential growth)
- Quantify how much temporal correlation matters
- Know when to hand off to more flexible approaches

**My prediction:**
- Models will capture trend and overdispersion reasonably well
- But will show residual autocorrelation 0.6-0.8 (due to lag-1 = 0.989)
- This will point toward hybrid approach or state-space models
- Success = finding this out efficiently and providing clear evidence

**Success is not "my models win":**
- Success is learning what works and what doesn't
- Success is providing clear evidence for next steps
- Success is knowing when parametric GLMs are insufficient
- Success is efficient exploration of model space

---

**Prepared by:** Designer 1 (Parametric GLM Focus)
**Date:** 2025-10-29
**Status:** Ready for implementation
**Next step:** Fit models and run diagnostics
