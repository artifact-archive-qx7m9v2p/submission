# Experiment Plan: Non-parametric/Flexible Bayesian Models

**Designer:** 2 (Non-parametric Specialist)
**Date:** 2025-10-29
**Status:** Design complete, ready for implementation

---

## Problem Formulation

### Multiple Competing Hypotheses

**Hypothesis 1: Smooth but Complex Non-linearity**
- Data follows an irregular growth curve not captured by simple polynomials
- Requires fully flexible functional form (GP or splines)
- Evidence: EDA shows strong curvature, but exact form uncertain

**Hypothesis 2: Structured Growth with Random Deviations**
- Underlying process is mechanistic (logistic, Gompertz) with stochastic shocks
- Requires hybrid parametric + flexible components
- Evidence: Accelerating growth suggests S-curve, but deviations visible

**Hypothesis 3: Data is Actually Simple (null hypothesis)**
- Apparent complexity is sampling noise with n=40
- Parametric models (Designer 1) are sufficient
- Evidence: Would be revealed by flexible models collapsing to simple forms

### Research Question
**Can flexible non-parametric models improve predictive performance over parametric GLMs for this time series count data, or is the apparent complexity an artifact of small sample size?**

---

## Model Classes to Explore

### Model Class 1: Gaussian Process Models
**Implementation**: GP-NegBin (GP prior on latent function + negative binomial likelihood)

**Rationale**:
- Makes minimal assumptions about functional form
- Learns smoothness and complexity from data
- Naturally handles temporal correlation through covariance structure

**Falsification criteria** - I will abandon GP models if:
- Posterior lengthscale ρ → ∞ (collapsing to constant/linear)
- Computational divergences > 10% despite extensive tuning
- LOO-CV worse than parametric baseline by >10 units
- Prior-posterior conflict on hyperparameters
- Wild extrapolation behavior (predictions outside [0, 500])

### Model Class 2: Spline-based GLMs
**Implementation**: P-spline with negative binomial likelihood

**Rationale**:
- Good middle ground: flexible but not too complex
- Automatic smoothness penalty prevents overfitting
- Computationally efficient for n=40
- Established literature on count data splines

**Falsification criteria** - I will abandon splines if:
- Smoothness parameter τ → 0 (over-smoothed, nearly linear)
- Smoothness parameter τ → ∞ (under-smoothed, fitting noise)
- High sensitivity to knot placement (results change >20% with different knots)
- Boundary artifacts (unrealistic predictions at year extremes)
- Cannot beat simpler parametric models on LOO-CV

### Model Class 3: Semi-parametric Models
**Implementation**: Logistic growth + GP deviations + time-varying overdispersion

**Rationale**:
- Combines mechanistic knowledge with data-driven flexibility
- Can decompose "expected" from "surprising" components
- Tests whether variance structure is constant or evolving

**Falsification criteria** - I will abandon semi-parametric if:
- GP deviations dominate (σ² > 50% of total variation) → parametric form wrong
- Logistic parameters non-identifiable (posterior spans full prior) → no S-curve
- Time-varying dispersion φ₁ credible interval includes zero → unnecessary complexity
- Computational failure (won't converge in <2 hours)
- Not competitive with pure parametric or pure GP models

---

## Specific Model Variants

### Variant 1.1: GP with Squared Exponential Kernel
- Standard choice, assumes infinite differentiability
- Hyperparameters: amplitude α², lengthscale ρ

### Variant 1.2: GP with Matern 3/2 Kernel
- Less smooth than SE, more realistic for real processes
- Often better behaved computationally
- Use if SE kernel has convergence issues

### Variant 2.1: P-splines with 8 Knots
- Baseline configuration
- K=8 interior knots, cubic B-splines (degree=3)
- Random walk smoothness prior

### Variant 2.2: P-splines with Adaptive Knot Selection
- Use if fixed knots show artifacts
- Place knots at changepoints detected in EDA
- May be overfitting risk with n=40

### Variant 3.1: Logistic Growth + GP
- Full model as specified
- Test mechanistic growth hypothesis

### Variant 3.2: Exponential Growth + GP
- If logistic shows no inflection
- Simpler baseline for structured component

---

## Red Flags That Would Trigger Model Class Changes

### From Flexible to Parametric (Abandon Flexibility)

**Red Flag 1: All flexible models collapse**
- Evidence: GP lengthscale >10, spline τ→0, semi-parametric δ→0
- Interpretation: Data is simpler than we thought
- Action: Adopt Designer 1's quadratic NegBin, stop wasting complexity

**Red Flag 2: Worse out-of-sample prediction than simple models**
- Evidence: Holdout RMSE for flexible models >1.5× parametric models
- Interpretation: Overfitting to training data
- Action: Use simpler model, add regularization, or ensemble with parametric

**Red Flag 3: Posterior predictive checks fail for all models**
- Evidence: Observed data outside 95% predictive intervals in systematic way
- Interpretation: Missing crucial structure (not just about flexibility)
- Action: Look for discrete events, external shocks, regime changes

### From Flexible to Temporal (Change Model Class)

**Red Flag 4: Residuals show strong autocorrelation**
- Evidence: Even best flexible model has residual ACF(1) > 0.7
- Interpretation: Temporal dynamics more important than mean function flexibility
- Action: Pivot to Designer 3's state-space or AR models

**Red Flag 5: Variance structure not captured**
- Evidence: Observed variance-to-mean ratio not matched by predictions
- Interpretation: Need more sophisticated variance modeling
- Action: Try Designer 3's dynamic variance models

### Computational Red Flags

**Red Flag 6: Models won't converge across implementations**
- Evidence: Divergences, R̂>1.1, ESS<100 after extensive tuning
- Interpretation: Bayesian inference intractable for this problem size/structure
- Action: Try variational inference, or abandon Bayesian for frequentist methods

---

## Decision Points for Major Strategy Pivots

### Decision Point 1: After Fitting GP and P-splines (Week 1)

**Question**: Are flexible models improving over parametric baseline?

**Metrics**:
- LOO-CV: Flexible vs Designer 1's best model
- Holdout RMSE: Last 10 observations
- Posterior predictive checks: Capture overdispersion?

**Decision Rules**:
- If LOO-IC improvement <5: Flexibility not justified → use parametric
- If LOO-IC improvement 5-15: Marginal benefit → ensemble models
- If LOO-IC improvement >15: Flexibility warranted → continue

### Decision Point 2: After Semi-parametric Fitting (Week 1)

**Question**: Is decomposition into structure+noise useful?

**Metrics**:
- Deviation magnitude: σ² relative to total variation
- Logistic parameters: Do they make scientific sense?
- Time-varying dispersion: Is φ₁ significantly nonzero?

**Decision Rules**:
- If deviations tiny (<10% of variance): Pure parametric sufficient
- If deviations huge (>50% of variance): Parametric form wrong, use pure GP/spline
- If intermediate: Semi-parametric adds value, investigate further

### Decision Point 3: Cross-Designer Comparison (Week 2)

**Question**: Which design perspective yielded the best model?

**Comparison Matrix**:
```
                    LOO-IC  Holdout-RMSE  Interpretability  Computation
Designer 1 (Param)    ?          ?              High          Fast
Designer 2 (Flex)     ?          ?              Medium        Medium
Designer 3 (Temp)     ?          ?              Medium        Slow
```

**Decision Rules**:
- If Designer 1 wins: Data is simpler than we thought
- If Designer 2 wins: Non-linearity requires flexibility
- If Designer 3 wins: Temporal correlation dominates
- If close: Ensemble or use simplest for deployment

---

## Alternative Approaches If Initial Models Fail

### Backup Plan 1: Simplified GP
**When**: If GP-NegBin has computational issues
**Approach**:
- Log-normal likelihood instead of negative binomial (easier)
- Sparse GP with M=15 inducing points
- Fix lengthscale via cross-validation instead of learning
**Expected outcome**: Faster convergence, possibly less accurate

### Backup Plan 2: Linear Splines
**When**: If cubic P-splines overfit
**Approach**:
- Degree 1 (piecewise linear) instead of degree 3
- Fewer knots (K=4)
- Stronger smoothness penalty
**Expected outcome**: More bias, less variance

### Backup Plan 3: Piecewise Parametric
**When**: If non-parametric fails but simple parametric inadequate
**Approach**:
- Divide time into 2-3 segments
- Fit separate quadratic/exponential per segment
- Hierarchical structure on parameters
**Expected outcome**: Captures different regimes without full flexibility

### Backup Plan 4: Ensemble Approach
**When**: If multiple models perform similarly
**Approach**:
- Bayesian model averaging across top 3 models
- Weights based on LOO-CV or stacking
- Predictions averaged across ensemble
**Expected outcome**: Robust to model misspecification

### Backup Plan 5: Admit Defeat
**When**: If everything fails (divergences, poor predictions, nonsense parameters)
**Approach**:
- Use Designer 1's simplest working model
- Focus on getting uncertainty quantification right
- Accept that n=40 limits what we can learn
**Expected outcome**: Honest assessment of what data can support

---

## Stopping Rules

### When to Stop Exploring Model Classes

**Rule 1: Diminishing Returns**
- If LOO-IC improvements <2 units for additional complexity
- Action: Stop, use simpler model

**Rule 2: Computational Wall**
- If fitting time >4 hours per model with no convergence
- Action: Abandon that model class, try simpler

**Rule 3: Time Constraint**
- If Week 1 ends without 3 converged models
- Action: Focus on best 1-2 models, drop others

**Rule 4: Evidence of Simplicity**
- If all flexible components collapse to linear/quadratic
- Action: Stop, adopt Designer 1's parametric approach

**Rule 5: Exhausted Reasonable Options**
- If tried all variants, backups, and nothing works
- Action: Report findings, recommend frequentist or simpler Bayesian

---

## What Evidence Would Make Me Switch Model Classes Entirely

### Switch to Parametric (Designer 1)
**Evidence**:
1. All my models have worse LOO-CV than Designer 1's quadratic NegBin
2. Flexible components collapse to simple polynomials
3. Computational issues insurmountable
4. Overfitting evident on holdout data

**Interpretation**: Flexibility is unjustified luxury for n=40, Occam's razor favors simple

### Switch to Temporal (Designer 3)
**Evidence**:
1. Even best flexible model has high residual autocorrelation (>0.7)
2. Designer 3's AR/state-space models substantially better LOO-CV
3. Temporal structure more important than mean function shape

**Interpretation**: Dynamics matter more than flexibility in mean

### Abandon Bayesian Entirely
**Evidence**:
1. No model converges despite extensive tuning
2. Prior-data conflicts across all reasonable priors
3. Computational time unacceptable for inference

**Interpretation**: Problem structure poorly suited to MCMC, use MLE with robust SEs

---

## Domain Constraints and Scientific Plausibility

### Constraints

**Hard constraints** (must satisfy):
1. Predictions must be non-negative (counts)
2. Growth cannot be unbounded forever (need saturation for extrapolation)
3. Variance must increase with mean (overdispersion pattern)
4. Temporal ordering matters (not exchangeable)

**Soft constraints** (prefer to satisfy):
5. Smooth transitions (no discontinuous jumps without explanation)
6. Monotonic growth (unless evidence of decline)
7. Bounded uncertainty (credible intervals shouldn't be absurdly wide)
8. Computational tractability (<1 hour per model)

### Plausibility Checks

**Before accepting any model, verify**:
1. Parameter values in reasonable range
   - GP lengthscale: 0.1 to 5.0 (smooth but not flat)
   - Overdispersion φ: 1 to 50 (allows current Var/Mean=68)
   - Growth rate: Positive (accelerating, not declining)

2. Posterior predictive distribution contains observed data
   - All 40 observations within 95% prediction intervals
   - Mean, variance, max of predictions match observed

3. Extrapolation behavior reasonable
   - Predictions at year=±2 (outside training) not absurd
   - Uncertainty increases appropriately outside data range

4. Scientific interpretation makes sense
   - Can tell coherent story about what process generated data
   - Parameters map to real-world quantities

---

## Expected Insights from Each Model

### From GP-NegBin (Model 1)
**Will learn**:
- Exact shape of non-linearity without assumptions
- How much flexibility is justified by data
- Appropriate smoothness scale (lengthscale)
- Whether temporal correlation captured by covariance function

**Decision**: If GP ≈ quadratic → parametric sufficient. If GP ≈ irregular → flexibility needed.

### From P-splines (Model 2)
**Will learn**:
- Optimal smoothness penalty (τ parameter)
- Local curvature patterns (where does growth accelerate?)
- Robustness to basis configuration
- Computational efficiency for this problem size

**Decision**: If τ→0 → over-smoothing, try GP. If τ→∞ → under-smoothing, use parametric.

### From Semi-parametric (Model 3)
**Will learn**:
- Is there mechanistic structure (logistic/Gompertz)?
- Magnitude of deviations from mechanism
- Time-varying variance structure
- Decomposition of predictable vs unpredictable

**Decision**: If large deviations → mechanism wrong. If small → mechanism + noise sufficient.

### Cross-Model Insights
**Will learn**:
- Model agreement = confidence (all predict similar → robust)
- Model disagreement = uncertainty (predictions vary → model-dependent)
- Complexity benefits = LOO-CV comparison
- Computational tradeoffs = time vs accuracy

---

## Summary and Implementation Priority

### Priority 1: P-splines (Model 2)
**Why**: Most likely to succeed for n=40, good flexibility/simplicity tradeoff
**Timeline**: Fit first, ~1 day
**Deliverable**: Working model with diagnostics, LOO-CV, posterior predictive checks

### Priority 2: GP-NegBin (Model 1)
**Why**: Full flexibility benchmark, scientifically interesting
**Timeline**: Fit second, ~1 day (may need tuning)
**Deliverable**: Comparison to P-splines, lengthscale interpretation

### Priority 3: Semi-parametric (Model 3)
**Why**: Most complex, may not converge, but novel insights if works
**Timeline**: Fit if time allows, ~2 days (debugging likely)
**Deliverable**: Decomposition analysis, time-varying variance investigation

### Expected Outcome
**Best guess**: P-splines will win among my models, competitive with Designer 1's quadratic NegBin

**Success defined as**:
- At least 1 model converges with good diagnostics
- LOO-CV competitive with or better than parametric baseline
- Posterior predictive checks pass
- Scientific interpretation plausible
- Honest assessment of when flexibility is/isn't needed

---

## Files and Next Steps

**Completed Documents**:
- `/workspace/experiments/designer_2/proposed_models.md` - Full specifications
- `/workspace/experiments/designer_2/model_comparison_matrix.md` - Quick reference
- `/workspace/experiments/designer_2/experiment_plan.md` - This document
- `/workspace/experiments/designer_2/README.md` - Summary

**Next Steps** (Implementation Phase):
1. Pre-process data (compute B-spline basis)
2. Write Stan code for Models 1-2
3. Write PyMC code for Model 3
4. Fit models with proper diagnostics
5. Compute LOO-CV and compare
6. Generate posterior predictive plots
7. Write results report with recommendations

**Cross-Designer Coordination**:
- Compare with Designer 1's parametric models (baseline)
- Consider hybrid with Designer 3's temporal models if needed
- Final ensemble if multiple models perform well

---

**Status**: Design phase complete, ready for implementation
**Designer**: 2 (Non-parametric/Flexible Specialist)
**Date**: 2025-10-29
**Mindset**: Truth over tasks, ready to pivot when evidence demands
