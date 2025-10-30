# Recommendations: Next Steps After Dual Model Rejection

**Date**: 2025-10-29
**Context**: Both polynomial models REJECTED
**Status**: Requires fundamentally different modeling approach

---

## Executive Summary

Given that BOTH polynomial Negative Binomial models failed despite perfect computational properties, we must explore fundamentally different model classes. This document provides evidence-based recommendations for alternative approaches, prioritized by likelihood of success and feasibility.

### Key Constraint

The data exhibit:
1. **Non-polynomial curvature** (adding quadratic term made residuals WORSE)
2. **Late-period systematic deviation** (4× degradation in predictive accuracy)
3. **Regime-change signature** (early vs late period behave differently)
4. **Overdispersion** (variance exceeds mean, Negative Binomial appropriate)

### Recommended Priority Order

1. **HIGH PRIORITY**: Changepoint models (strong evidence for regime shift)
2. **MEDIUM PRIORITY**: Gaussian processes (flexible nonparametric alternative)
3. **MEDIUM PRIORITY**: Time-varying coefficient models (growth rate changes)
4. **LOW PRIORITY**: Higher-order polynomials (likely to fail for same reasons)
5. **EXPLORATORY**: Missing covariates investigation

---

## Recommendation 1: Changepoint Models (HIGH PRIORITY)

### Rationale

**Strong Evidence for Regime Change**:
1. Early period MAE = 6.3 (good fit)
2. Late period MAE = 26.5 (poor fit)
3. Ratio of 4.17× suggests different processes
4. Residual pattern shows systematic shift, not gradual curvature
5. LOO diagnostics are excellent (no influential points), so this isn't outlier-driven

### Proposed Model Class

**Two-Regime Changepoint Model**:

```python
# Model 3a: Single unknown changepoint
C[i] ~ NegativeBinomial(μ[i], φ)

if year[i] < τ:
    log(μ[i]) = β₀₁ + β₁₁ × year[i]  # Regime 1
else:
    log(μ[i]) = β₀₂ + β₁₂ × year[i]  # Regime 2

Priors:
  τ ~ Uniform(year_min, year_max)  # Changepoint location
  β₀₁, β₀₂ ~ Normal(4.3, 1.0)      # Regime-specific intercepts
  β₁₁, β₁₂ ~ Normal(0.85, 0.5)     # Regime-specific slopes
  φ ~ Exponential(0.667)
```

**Parameters**: 6 (τ, β₀₁, β₁₁, β₀₂, β₁₂, φ)

### Expected Benefits

1. **Captures regime shift**: Different growth rates before/after changepoint
2. **Addresses late-period failure**: Second regime can have different dynamics
3. **Interpretable**: Changepoint has scientific meaning (what changed?)
4. **Testable**: Compare ELPD to polynomial models

### Implementation Considerations

**Computational Challenges**:
- Changepoint parameter (τ) is discrete → challenging for HMC
- May need special sampling scheme (e.g., marginalization)
- Consider continuous approximation (sigmoid transition)

**Solution**: Use continuous transition function:

```python
# Model 3b: Smooth changepoint (sigmoid transition)
weight[i] = 1 / (1 + exp(-k × (year[i] - τ)))

log(μ[i]) = weight[i] × (β₀₁ + β₁₁ × year[i]) +
            (1 - weight[i]) × (β₀₂ + β₁₂ × year[i])

k ~ Exponential(1)  # Controls transition sharpness
```

### Falsification Criteria

**REJECT this model if**:
1. τ posterior is uniform (no clear changepoint)
2. β₁₁ ≈ β₁₂ (regimes not different)
3. ΔELPD < 4 vs. Model 1 (no improvement)
4. Residual curvature persists within each regime
5. Late-period MAE still >2× early-period MAE

**ACCEPT this model if**:
1. τ posterior is concentrated (clear changepoint identified)
2. β₁₁ ≠ β₁₂ (different growth rates)
3. ΔELPD > 10 vs. Model 1 (strong improvement)
4. Residuals show no systematic patterns within regimes
5. Early/late MAE ratio < 1.5

### Expected ELPD Improvement

**Predicted**: ΔELPD = 15-25 (strong evidence)

**Justification**: If early/late periods truly have different dynamics, changepoint model should:
- Reduce late-period errors by ~50% (26.5 → 13)
- Maintain early-period fit (MAE ≈ 6.3)
- Overall predictive accuracy should improve substantially

### Interpretation Questions

If changepoint model succeeds, investigate:
1. **What happened at τ?** (scientific/historical event)
2. **Why did growth rate change?** (mechanism)
3. **Is the shift permanent?** (implications for forecasting)
4. **Are there external covariates aligned with τ?** (confounders)

---

## Recommendation 2: Gaussian Process Models (MEDIUM PRIORITY)

### Rationale

**Flexible Nonparametric Trend**:
1. Makes no assumptions about functional form
2. Can capture complex, non-polynomial patterns
3. Provides uncertainty quantification
4. Data-driven smoothness

### Proposed Model Class

**GP with Negative Binomial Likelihood**:

```python
# Model 4: Gaussian Process trend
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = f(year[i])

f ~ GP(mean=m(year), cov=k(year, year'))

Mean function options:
  - m(year) = β₀ + β₁ × year  # Linear trend
  - m(year) = 0                # Zero mean

Covariance function options:
  - k = Matern(ν=3/2, ℓ, σ)   # Flexible, once-differentiable
  - k = RBF(ℓ, σ)              # Smooth
  - k = RatQuad(ℓ, σ, α)      # Mix of length scales

Priors:
  β₀, β₁ ~ Normal(...)  # If using linear mean
  ℓ ~ InverseGamma(...)  # Length scale
  σ ~ HalfNormal(...)    # GP variance
  φ ~ Exponential(0.667)
```

**Parameters**: Depends on mean/cov choice, typically 4-6

### Expected Benefits

1. **No functional form assumption**: Learns shape from data
2. **Smooth but flexible**: Can capture complex patterns
3. **Uncertainty quantification**: Wider in regions with less data
4. **Interpolation strength**: Excellent for filling gaps

### Challenges

1. **Computational cost**: O(n³) for n observations (manageable for n=40)
2. **Hyperparameter sensitivity**: Length scale ℓ controls smoothness
3. **Interpretability**: Less intuitive than parametric models
4. **Extrapolation risk**: GP uncertainty grows rapidly beyond data range

### Falsification Criteria

**REJECT this model if**:
1. ΔELPD < 4 vs. Model 1 (no improvement)
2. Residual patterns remain systematic
3. Posterior predictive intervals unrealistically wide
4. Length scale ℓ → 0 or ℓ → ∞ (degenerate)

**ACCEPT this model if**:
1. ΔELPD > 10 vs. Model 1
2. Residuals show no structure
3. Learned function f(year) reveals interpretable pattern
4. Late/early MAE ratio < 2.0

### Expected ELPD Improvement

**Predicted**: ΔELPD = 10-20 (moderate to strong evidence)

**Justification**: GP's flexibility should capture non-polynomial curvature, but may overfit without sufficient data structure.

### Implementation Notes

**Computational Tips**:
- Use approximate GP (e.g., HSGP, sparse variational)
- Start with Matern ν=3/2 (good default, not too smooth)
- Place informative prior on length scale to avoid overfitting

**Priors**:
```python
ℓ ~ InverseGamma(5, 5)  # Concentrates around 1 (moderate smoothness)
σ ~ HalfNormal(1)       # GP variance comparable to data scale
```

---

## Recommendation 3: Time-Varying Coefficient Models (MEDIUM PRIORITY)

### Rationale

**Growth Rate Changes Over Time**:
1. Residual curvature suggests β₁ is not constant
2. Late-period deviation implies accelerating growth
3. More parsimonious than GP, more flexible than polynomial

### Proposed Model Class

**Random Walk in Growth Rate**:

```python
# Model 5: Time-varying growth rate
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁[i] × year[i]

β₁[1] ~ Normal(0.85, 0.5)           # Initial growth rate
β₁[i] ~ Normal(β₁[i-1], σ_β)        # Random walk
  for i = 2, ..., n

Priors:
  β₀ ~ Normal(4.3, 1.0)
  σ_β ~ HalfNormal(0.1)  # Controls rate of change
  φ ~ Exponential(0.667)
```

**Parameters**: n+3 (β₀, β₁[1:n], σ_β, φ) - HIGH dimensional

### Expected Benefits

1. **Captures acceleration**: β₁ can increase over time
2. **Data-driven changes**: No assumption about when/how β₁ changes
3. **Smooth transitions**: Random walk ensures continuity

### Challenges

1. **High dimensionality**: n+3 parameters for n=40 observations (43 total)
2. **Overfitting risk**: Many parameters, may fit noise
3. **Identifiability**: Difficulty separating trend from random walk
4. **Computational cost**: More parameters → longer sampling

### Alternative: Parametric Time-Varying

**More parsimonious**:

```python
# Model 5b: Linear time-varying growth rate
log(μ[i]) = β₀ + (β₁₀ + β₁₁ × year[i]) × year[i]
          = β₀ + β₁₀ × year[i] + β₁₁ × year²[i]
```

**Issue**: This reduces to Model 2 (quadratic), which already failed!

### Falsification Criteria

**REJECT this model if**:
1. σ_β → 0 (β₁ is constant, reduces to Model 1)
2. σ_β very large (overfitting, β₁ varies randomly)
3. ΔELPD < 4 vs. Model 1
4. p_loo >> n (effective parameters exceed observations)

**ACCEPT this model if**:
1. σ_β is moderate (0.01 < σ_β < 0.1)
2. β₁[i] shows smooth, interpretable trend
3. ΔELPD > 10 vs. Model 1
4. p_loo ≈ 5-10 (reasonable effective dimensionality)

### Expected ELPD Improvement

**Predicted**: ΔELPD = 5-15 (moderate evidence)

**Caveat**: High risk of overfitting given n=40 observations and 43 parameters

---

## Recommendation 4: Higher-Order Polynomials (LOW PRIORITY)

### Rationale

**Complete the polynomial exploration**:
- Model 1: Linear (failed)
- Model 2: Quadratic (failed)
- Model 3?: Cubic or higher

### Why LOW Priority?

**Strong Evidence Against This Approach**:
1. **Quadratic made things WORSE** (residual curvature increased)
2. **No theoretical justification** for cubic growth
3. **Overfitting risk** increases with polynomial order
4. **Extrapolation danger** (polynomials oscillate)

### Only Consider If

You need to definitively rule out polynomial growth for publication purposes:

```python
# Model 6: Cubic (NOT RECOMMENDED)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i] + β₃ × year³[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  β₂ ~ Normal(0, 0.5)
  β₃ ~ Normal(0, 0.25)  # Smaller SD for higher-order
  φ ~ Exponential(0.667)
```

### Falsification Criteria

**REJECT this model if** (likely):
1. β₃ non-significant (CI includes 0)
2. ΔELPD < 2 vs. Model 2
3. Residual curvature persists or worsens
4. Model fit oscillates (polynomial wiggling)

### Expected Outcome

**Predicted**: Model 6 will ALSO fail for the same reasons as Model 2

**Justification**: The data structure is fundamentally non-polynomial. Adding higher-order terms will:
- Fit noise, not signal
- Increase variance without reducing bias
- Make extrapolation even more unreliable

**Recommendation**: SKIP this unless required for completeness.

---

## Recommendation 5: Missing Covariate Investigation (EXPLORATORY)

### Rationale

**The curvature may be a confounder**:
1. If an omitted variable correlates with time AND affects counts, residuals will show systematic patterns
2. Late-period deviation could be driven by changing covariate
3. Polynomial models fail because the true relationship is with X, not time

### Investigation Steps

**1. Domain Expert Consultation**:
- What changed between early and late period?
- Were there policy changes, measurement protocol changes, population shifts?
- Are there known confounders in this domain?

**2. Data Availability Check**:
- Do auxiliary datasets exist?
- Can we obtain:
  - Population counts over time?
  - Economic indicators?
  - Policy implementation dates?
  - Measurement quality metrics?

**3. Exploratory Data Analysis**:
- Plot known covariates vs. time
- Look for correlations with residuals
- Identify candidate predictors

### Proposed Model Class

**If covariate X is identified**:

```python
# Model 7: Linear trend + covariate
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × X[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  β₂ ~ Normal(0, 1.0)  # Covariate effect
  φ ~ Exponential(0.667)
```

**Expected**: If X explains the curvature:
- β₁ will be smaller (less time trend after adjusting for X)
- Residual curvature will reduce
- ELPD improvement proportional to X's explanatory power

### Falsification Criteria

**REJECT this approach if**:
1. No plausible covariates can be identified
2. Available covariates don't correlate with residuals
3. Adding X doesn't improve ELPD (ΔELPD < 4)

**ACCEPT this approach if**:
1. X is scientifically justified
2. X correlates with time AND with residuals
3. ΔELPD > 10 after adding X
4. Residual curvature disappears

### Expected Outcome

**Uncertain**—depends entirely on whether relevant covariates exist and are measurable.

---

## Recommendation 6: Alternative Likelihood Families (EXPLORATORY)

### Rationale

**All models assume Negative Binomial likelihood**. What if the issue is the likelihood, not the trend?

### Alternatives to Consider

**1. Zero-Inflated Negative Binomial (ZINB)**:
```python
# Model 8: ZINB with excess zeros
C[i] ~ ZINB(μ[i], φ, p_zero)

With probability p_zero: C[i] = 0
With probability 1 - p_zero: C[i] ~ NegBin(μ[i], φ)
```

**Issue**: Observed data has ZERO zeros (minimum count = 21)
**Verdict**: NOT APPLICABLE

**2. Conway-Maxwell-Poisson (CMP)**:
```python
# Model 9: CMP (flexible dispersion)
C[i] ~ CMP(λ[i], ν)

ν controls dispersion:
  ν > 1: Under-dispersed
  ν = 1: Poisson
  ν < 1: Over-dispersed
```

**Issue**: More complex than NegBin, unlikely to fix trend misspecification
**Verdict**: LOW PRIORITY

**3. Hurdle Model**:
```python
# Model 10: Hurdle (separate zero and positive counts)
Binary process: P(C > 0) = logit^(-1)(α)
Count process: C | C > 0 ~ NegBin(μ, φ)
```

**Issue**: Again, no zeros in data
**Verdict**: NOT APPLICABLE

### When to Consider

Only if:
1. Trend model is fixed (changepoint, GP, etc. works)
2. Residuals STILL show patterns
3. Diagnostic suggests likelihood issue (e.g., variance structure)

**Current Assessment**: Trend misspecification is the problem, not likelihood family.

---

## Implementation Roadmap

### Phase 1: Changepoint Model (IMMEDIATE)

**Priority**: HIGH
**Effort**: Moderate (4-6 hours)
**Expected Success**: 70%

**Steps**:
1. Implement smooth changepoint model (sigmoid transition)
2. Use same prior predictive → SBC → inference → PPC pipeline
3. Compare LOO-CV to Models 1 & 2
4. If successful, interpret τ (changepoint location)

**Decision Point**: If ΔELPD > 10, ACCEPT and publish. Otherwise, continue.

### Phase 2: Gaussian Process (IF Phase 1 Fails)

**Priority**: MEDIUM
**Effort**: High (8-10 hours)
**Expected Success**: 50%

**Steps**:
1. Implement GP with Matern covariance
2. Use approximate method (HSGP) for efficiency
3. Tune length scale prior carefully
4. Compare LOO-CV to all previous models

**Decision Point**: If ΔELPD > 10, ACCEPT. Otherwise, reconsider problem.

### Phase 3: Missing Covariates (PARALLEL)

**Priority**: EXPLORATORY
**Effort**: Variable (depends on data availability)
**Expected Success**: Unknown

**Steps**:
1. Consult domain experts
2. Identify candidate covariates
3. Obtain auxiliary data if available
4. Add to best-performing model from Phases 1-2

**Decision Point**: If covariate available and improves fit, incorporate.

### Phase 4: Time-Varying Coefficients (IF All Else Fails)

**Priority**: LOW
**Effort**: High (overfitting risk, many diagnostics)
**Expected Success**: 30%

**Only pursue if**:
- Changepoint model fails
- GP model fails
- No covariates available
- Must produce a working model

---

## Decision Criteria

### When to ACCEPT a New Model

**Minimum Requirements**:
1. **Strong LOO-CV improvement**: ΔELPD > 10 vs. Model 1
2. **Statistical significance**: ΔELPD > 4 × SE
3. **Posterior predictive checks pass**: ≥3 of 4 criteria
4. **Residuals show no systematic patterns**: |quadratic coef| < 1.0
5. **Late/early MAE ratio < 2.0**: Fit doesn't degrade

**Ideal Requirements**:
1. ΔELPD > 20 (very strong evidence)
2. All posterior predictive checks pass
3. Residuals approximately normal
4. Late/early MAE ratio < 1.5
5. Parameters scientifically interpretable

### When to REJECT and Move On

**Abandon model if**:
1. ΔELPD < 4 vs. previous models
2. Computational issues (divergences, poor mixing)
3. Posterior predictive checks still fail
4. Parameters not identifiable
5. Results not interpretable

### When to Reconsider the Problem

**If all recommended models fail**:
1. **Re-examine the data**: Are there errors, outliers, measurement issues?
2. **Reconsider the goal**: Maybe prediction isn't feasible; focus on description?
3. **Consult domain experts**: What do they think is driving the pattern?
4. **Consider ensemble methods**: Model averaging, stacking
5. **Accept limitations**: Document what DOESN'T work as scientific contribution

---

## Resources Required

### Computational

**Changepoint Model**:
- Runtime: ~5 minutes (similar to Models 1-2)
- Memory: Minimal
- Software: PyMC (already available)

**Gaussian Process**:
- Runtime: ~15-30 minutes (GP is O(n³))
- Memory: Moderate (covariance matrices)
- Software: PyMC with GP module

**Time-Varying Coefficients**:
- Runtime: ~30-60 minutes (high-dimensional)
- Memory: High (many parameters)
- Software: PyMC

### Human

**Analyst Time**:
- Changepoint: 4-6 hours (implementation + validation)
- GP: 8-10 hours (tuning + validation)
- Covariates: Variable (data acquisition)
- Time-varying: 10-12 hours (overfitting diagnostics)

**Domain Expert**:
- 1-2 hours for covariate identification
- 1 hour for changepoint interpretation

### Data

**Current**:
- 40 observations, no covariates
- Sufficient for changepoint, GP
- Marginal for time-varying (overfitting risk)

**Ideal**:
- Additional covariates if available
- More observations would reduce uncertainty
- Historical context for changepoint interpretation

---

## Risk Assessment

### Risk 1: All Alternative Models Also Fail

**Likelihood**: Moderate (30%)
**Impact**: High (no publishable model)

**Mitigation**:
- Pursue multiple approaches in parallel
- Document failures as scientific contribution
- Consider ensemble/stacking methods
- Reconsider research question

### Risk 2: Overfitting in Complex Models

**Likelihood**: High for GP and time-varying (60%)
**Impact**: Moderate (spurious patterns)

**Mitigation**:
- Strong emphasis on LOO-CV
- Cross-validation beyond LOO
- Conservative priors on complexity parameters
- Skeptical posterior predictive checks

### Risk 3: Computational Challenges

**Likelihood**: Moderate for GP (40%)
**Impact**: Low (delays, not failure)

**Mitigation**:
- Use approximate methods (HSGP, sparse variational)
- Increase warmup iterations
- Consider reparameterization
- Profile and optimize code

### Risk 4: Interpretation Difficulty

**Likelihood**: High for GP (70%)
**Impact**: Moderate (reduced scientific value)

**Mitigation**:
- Focus on predicted function f(year), not parameters
- Visualize learned trends extensively
- Compare to domain knowledge
- Supplement with simpler models for communication

---

## Success Metrics

### Technical Success

**Minimum**:
- ΔELPD > 10 vs. Model 1
- ≥3 of 4 posterior predictive checks pass
- Pareto k < 0.7 for all observations

**Target**:
- ΔELPD > 20 vs. Model 1
- All posterior predictive checks pass
- Late/early MAE ratio < 1.5

### Scientific Success

**Minimum**:
- Model captures key data features
- Residuals show no systematic bias
- Parameters/function interpretable

**Target**:
- Model reveals mechanism (e.g., changepoint explains process shift)
- Predictions actionable for decision-making
- Results publishable

### Practical Success

**Minimum**:
- Model converges reliably
- Inference completes in <30 minutes
- Results reproducible

**Target**:
- Inference completes in <10 minutes
- Model easy to explain to stakeholders
- Predictions have tight intervals (informative)

---

## Conclusion

### Immediate Action

**Implement Changepoint Model** (Model 3b with sigmoid transition)

**Rationale**:
1. Strongest evidence (4× MAE degradation early to late)
2. Highest expected ELPD improvement
3. Most interpretable if successful
4. Moderate implementation effort

### Fallback Plan

If changepoint fails: **Gaussian Process** (Model 4)

### Long-Term Strategy

This dual-rejection experience demonstrates the value of:
1. **Falsification frameworks** (catching inadequate models)
2. **Comprehensive validation** (PPC + LOO-CV)
3. **Model comparison** (preventing premature acceptance)
4. **Iterative refinement** (learning from failures)

The next model will benefit from:
- Lessons learned about data structure
- Clear criteria for success/failure
- Established validation pipeline
- Comparative baseline (Models 1 & 2)

### Final Thought

**Failure is progress**. We now know:
- What DOESN'T work (polynomial growth)
- What the data require (regime change or flexible trend)
- How to evaluate alternatives (LOO-CV + PPC)

The path forward is clearer for having tried—and properly rejected—these inadequate models.

---

**Recommendations finalized**: 2025-10-29
**Priority**: Changepoint model → Gaussian process → Covariates
**Next step**: Implement Model 3b and reassess
**Expected timeline**: 1-2 weeks for full exploration
