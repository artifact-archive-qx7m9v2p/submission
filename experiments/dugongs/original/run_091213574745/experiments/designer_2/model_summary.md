# Quick Reference: Flexible Bayesian Models Summary

**Designer**: Flexible/Nonparametric Specialist
**Date**: 2025-10-28
**Full Proposal**: `/workspace/experiments/designer_2/proposed_models.md`

---

## Three Competing Model Classes

### Model 1: Gaussian Process with Matérn 3/2 Kernel (RECOMMENDED START)
- **Hypothesis**: Smooth monotonic saturation (no true regime change)
- **Core**: GP with informative logarithmic mean function
- **Key Parameters**: Length scale ℓ, marginal SD α
- **Abandonment Criteria**: Divergences >5%, wild oscillations, LOO worse than baseline
- **Expected Runtime**: 5-10 minutes
- **Priority**: FIRST (most principled)

### Model 2: Penalized B-Splines with Fixed Knots
- **Hypothesis**: Locally complex with sharp transitions possible
- **Core**: 10 cubic B-spline basis functions with random walk penalty
- **Key Parameters**: 6 interior knots, smoothness τ
- **Abandonment Criteria**: Wild coefficient oscillations, poor LOO, extreme extrapolation
- **Expected Runtime**: 2-3 minutes
- **Priority**: SECOND (alternative approach)

### Model 3: Adaptive GP with Regime-Specific Smoothness
- **Hypothesis**: True regime change at unknown τ with different smoothness per regime
- **Core**: Piecewise GP with separate length scales, unknown changepoint
- **Key Parameters**: τ ∈ [4,12], ℓ₁, ℓ₂ (regime-specific)
- **Abandonment Criteria**: τ posterior uniform, divergences >10%, no difference in ℓ₁ vs ℓ₂
- **Expected Runtime**: 15-30 minutes
- **Priority**: THIRD (only if Models 1-2 suggest regime change)

---

## Critical Decision Points

### Checkpoint 1: Convergence (Day 1)
**Question**: Do models converge cleanly?
- **Pass**: R̂ < 1.01, ESS > 400, divergences < 1%
- **Fail**: Simplify model or abandon approach

### Checkpoint 2: LOO Comparison (Day 1-2)
**Question**: Do flexible models beat parametric baselines?
- **Baseline**: Logarithmic model (LOO ≈ -10)
- **Success**: ΔLOO > 2 favoring flexible models
- **Failure**: Logarithmic wins → Accept simple model

### Checkpoint 3: Posterior Predictive Checks (Day 2)
**Question**: Do models capture essential features?
- **Must pass**: Monotonic, asymptotic, correct variance
- **Red flag**: Systematic residual patterns, poor predictions

### Checkpoint 4: Scientific Story (Day 2)
**Question**: Can we interpret results coherently?
- **Good**: Clear winner, interpretable parameters
- **Bad**: Models disagree on fundamentals
- **Action**: Model averaging if indistinguishable

---

## Falsification Mindset

### I Will Abandon Flexible Approaches If...
1. All models have LOO worse than simple logarithmic (ΔLOO < -2)
2. All models show signs of overfitting (wild oscillations, poor out-of-sample)
3. Computational issues persist despite tuning
4. Posterior uncertainty is so large that conclusions are meaningless

### I Will Declare Success If...
1. One model clearly beats baselines (ΔLOO > 4)
2. Regime change hypothesis definitively confirmed or rejected
3. Outlier sensitivity analysis shows robustness
4. Can provide actionable predictions with UQ

### I Expect to Find...
- Model 1 (GP) beats baselines by ΔLOO ≈ 2-3
- Length scale ℓ ≈ 5-10
- Model 3 (adaptive) identifies τ ∈ [6, 8] if regime change is real
- All models struggle with x=31.5 outlier

---

## Implementation Notes

### Software Stack
- **Primary**: PyMC 5.x (Bayesian PPL)
- **Basis functions**: patsy (splines), scipy (B-spline evaluation)
- **Diagnostics**: arviz (plots, LOO-CV, R̂, ESS)

### Sampling Strategy
- **Warmup**: 2000 iterations minimum
- **Sampling**: 2000 iterations (4 chains = 8000 total)
- **Target accept**: 0.90-0.95 (0.99 for complex models)
- **Adaptation**: adapt_diag initialization

### Expected Challenges
1. **GP sampling**: May need higher target_accept, more iterations
2. **P-spline collinearity**: Random walk prior should help
3. **Adaptive GP changepoint**: Discrete parameter, slow mixing
4. **Outlier x=31.5**: May inflate GP variance locally

---

## Escape Routes

If initial models fail, try:
1. **Student-t likelihood** (robust to outliers)
2. **Heteroscedastic noise** (variance increases with x)
3. **Fixed changepoint** at τ=7 (remove estimation)
4. **Transform Y** (log or sqrt transformation)
5. **Abandon flexibility** (use Designer 1's parametric models)

---

## Success Metrics

### Computational (Must Pass)
- R̂ < 1.01 for all parameters
- ESS_bulk > 400, ESS_tail > 400
- Divergences < 1% (ideally 0%)
- E-BFMI > 0.3

### Predictive (Primary)
- **LOO-CV**: Lower is better, ΔLOO > 2 is meaningful
- **WAIC**: Should agree with LOO
- **k̂ diagnostics**: All k̂ < 0.7 (influential point check)

### Scientific (Validation)
- Posterior predictive checks pass
- Monotonic increasing relationship
- Outlier sensitivity shows robustness
- Interpretable parameter estimates

---

## Key Files

- **Full proposal**: `/workspace/experiments/designer_2/proposed_models.md`
- **This summary**: `/workspace/experiments/designer_2/model_summary.md`
- **Data**: `/workspace/data/data.csv` (n=27)
- **EDA report**: `/workspace/eda/eda_report.md`

---

## Timeline

- **Day 1 Morning**: Fit Models 1 & 2, check convergence
- **Day 1 Afternoon**: LOO comparison, decide if Model 3 needed
- **Day 2 Morning**: Fit Model 3 (if warranted), posterior predictive checks
- **Day 2 Afternoon**: Sensitivity analyses, final comparison
- **Day 3**: Write results report with honest assessment

**Total compute time**: ~2-3 hours
**Total analyst time**: ~2-3 days

---

## Philosophy

This proposal embodies:
- **Skepticism of EDA**: Simple patterns may hide complexity
- **Falsification focus**: Each model has clear failure criteria
- **Computational honesty**: Divergences indicate real problems
- **Scientific humility**: Simple models may be right
- **Rigorous comparison**: Let data choose, not preferences

**Bottom line**: These models are designed to fail informatively. If they succeed, great. If they fail, we learn why flexibility doesn't help with n=27.

---

**For implementation, start with Model 1 (GP) as the primary flexible approach.**
