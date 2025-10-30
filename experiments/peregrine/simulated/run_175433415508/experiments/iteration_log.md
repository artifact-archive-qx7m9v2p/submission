# Model Iteration Log

**Purpose**: Track model refinements, changes, and rationale across experiment iterations.

**Project**: Negative Binomial Time Series Modeling

---

## Iteration History

### Iteration 0: Baseline Model (Experiment 1)

**Date**: 2025-10-29
**Experiment**: experiment_1
**Model**: Negative Binomial Linear Regression

**Specification**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁×year_t

Priors:
  β₀ ~ Normal(4.69, 1.0)
  β₁ ~ Normal(1.0, 0.5)
  φ ~ Gamma(2, 0.1)
```

**Status**: SUCCESS
- Prior predictive check: PASS
- Model fitting: Converged (R-hat=1.00, ESS>2500)
- Posterior predictive check: Issues with residual autocorrelation
- Model critique: Identified temporal correlation (residual ACF(1)=0.511)

**Key Results**:
- β₀ = 4.35 ± 0.04
- β₁ = 0.87 ± 0.04
- φ = 35.6 ± 10.8
- ELPD_loo = -170.05 ± 5.17

**Limitation**: Residual autocorrelation suggests missing temporal structure

**Next Step**: Add AR(1) component to model temporal correlation

---

### Iteration 1a: AR(1) Model - Original Priors (Experiment 2)

**Date**: 2025-10-29
**Experiment**: experiment_2
**Model**: Negative Binomial with AR(1) Temporal Correlation

**Specification**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t, ν_t ~ Normal(0, σ)

Priors:
  β₀ ~ Normal(4.69, 1.0)
  β₁ ~ Normal(1.0, 0.5)
  φ ~ Gamma(2, 0.1)
  ρ ~ Beta(20, 2)
  σ ~ Exponential(2)
```

**Status**: FAILED - Prior Predictive Check
- Prior predictive check: FAIL (3 critical failures)
- Model fitting: Not attempted
- Issue: Extreme tail behavior

**Failure Metrics**:
- 3.22% of counts > 10,000 (threshold: <1%)
- Maximum count: 674,970,346 (observed max: 269)
- 99th percentile: 143,745 (vs observed: 269)
- Mean maximum per series: 2,038,561

**Root Cause Analysis**:
Wide priors + exponential link → multiplicative explosion:
- β₁ ~ Normal(1.0, 0.5) allows extreme growth (β₁ > 2.0)
- φ ~ Gamma(2, 0.1) allows small values (φ < 5) → high variance
- σ ~ Exponential(2) allows large innovations (σ > 1.5)
- Rare joint extremes create counts in millions through exp(η)

**Decision**: Refine priors to constrain tail behavior

---

### Iteration 1b: AR(1) Model - Refined Priors (Experiment 2 Refined)

**Date**: 2025-10-29
**Experiment**: experiment_2_refined
**Model**: Negative Binomial with AR(1) Temporal Correlation (Refined Priors)

**Specification**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t, ν_t ~ Normal(0, σ)

REFINED Priors:
  β₀ ~ Normal(4.69, 1.0)                      [UNCHANGED]
  β₁ ~ TruncatedNormal(1.0, 0.5, -0.5, 2.0)  [REFINED: constrain growth]
  φ ~ Normal(35, 15), φ > 0                   [REFINED: inform from Exp1]
  ρ ~ Beta(20, 2)                             [UNCHANGED]
  σ ~ Exponential(5)                          [REFINED: tighter scale]
```

**Changes from Iteration 1a**:

1. **β₁**: Truncated to [-0.5, 2.0]
   - **Rationale**: Prevent extreme growth rates (>25× over study period)
   - **Scientific justification**: Observed growth is 13×, bound at 25× is conservative
   - **Experiment 1 evidence**: β₁=0.87±0.04, well within bounds

2. **φ**: Centered at Experiment 1 posterior (35 vs 20)
   - **Rationale**: Stabilize variance structure, prevent small φ
   - **Scientific justification**: φ is data property, unlikely to change with AR(1)
   - **Information transfer**: Valid use of Exp1 posterior mean

3. **σ**: Tighter prior (E[σ]=0.2 vs 0.5)
   - **Rationale**: Constrain AR process innovations
   - **Scientific justification**: Innovations should be small deviations, not shocks
   - **Stationary SD**: E[SD(ε)] ≈ 0.48 vs 1.2 original

**Expected Improvements**:
- Counts > 10,000: 3.22% → <1% (>90% reduction)
- Maximum count: 674M → <100,000 (>99.99% reduction)
- 99th percentile: 143,745 → <5,000 (>96% reduction)
- Median behavior: Preserved (~110 counts)

**Status**: Prior Predictive Check Pending
- Script ready: `/workspace/experiments/experiment_2_refined/prior_predictive_check/code/prior_predictive_check.py`
- Next step: Execute and evaluate

**Success Criteria**:
- [ ] <1% counts > 10,000
- [ ] <5% counts > 5,000
- [ ] Maximum count < 100,000
- [ ] Mean ACF(1) in [0.3, 0.99]
- [ ] Median ~100-200 (covers observed)

**If Successful**: Proceed to model fitting with PyMC
**If Failed**: Consider further refinement or model simplification

---

## Refinement Strategy Summary

### Principles Applied

1. **Targeted Constraints**: Only modified parameters driving failures (β₁, φ, σ)
2. **Minimal Changes**: Kept β₀ and ρ unchanged (working well)
3. **Scientific Validity**: All changes justified by data or theory
4. **Information Reuse**: Leveraged Experiment 1 for φ (valid transfer)
5. **Falsifiability**: Clear success criteria and fallback plans

### Trade-offs

**Increased Informativeness**:
- More constrained priors (less "objective")
- BUT: Prevents numerical instabilities
- AND: Still substantial uncertainty (wide credible intervals expected)

**Computational Stability**:
- Truncation may cause MCMC boundary issues
- BUT: Non-centered parameterization can help
- AND: Prevents overflow in exp(η)

**Model Flexibility**:
- Less flexible for extreme dynamics
- BUT: Extreme scenarios scientifically implausible
- AND: Data can still inform within reasonable ranges

### Stopping Criteria

**Continue iterating if**:
- Prior predictive checks identify fixable issues
- Changes target root causes (not symptoms)
- Model structure remains scientifically plausible
- Complexity justified by diagnostics

**Stop iterating if**:
- Multiple refinements fail to fix core issues
- Hitting complexity ceiling (uninterpretable)
- Fundamental model class inadequacy
- Computational problems persist despite reparameterization
- Maximum iterations reached (typically 5-7)

---

## Lessons Learned

### From Iteration 1a Failure

1. **Exponential link amplifies tail behavior**: Wide priors on log-scale create extreme values on count scale
2. **Joint extremes are multiplicative**: Small probabilities multiply (0.025 × 0.10 × 0.08 ≈ 0.0002)
3. **Prior predictive checks catch this**: Would have wasted computation on fitting
4. **Median ≠ tail**: Median can be reasonable while tail is catastrophic

### From Refinement Process

1. **Use available information**: Experiment 1 φ posterior is valid information
2. **Truncation is scientifically valid**: Constraining to plausible range ≠ "cheating"
3. **Multiple constraints compound**: Three refinements → multiplicative improvement
4. **Preserve what works**: Don't change everything, target the problem

### Prior Predictive Workflow Value

**What it prevented**:
- Wasting 30-60 minutes on MCMC fitting
- Diagnosing posterior issues that were prior issues
- Circular debugging (fit → diagnose → refit → ...)

**What it enabled**:
- Targeted refinement before any computation
- Clear comparison (original vs refined)
- Documented rationale for changes

---

## Next Steps

### Immediate (Iteration 1b)

1. **Execute refined prior predictive check**
   ```bash
   python /workspace/experiments/experiment_2_refined/prior_predictive_check/code/prior_predictive_check.py
   ```

2. **Evaluate against success criteria**
   - All 7 diagnostic checks
   - Comparison to original metrics
   - Decision: PASS/CONDITIONAL/FAIL

3. **Document results**
   - Update this log with PPC outcome
   - Create findings.md in experiment_2_refined/

### If Iteration 1b Passes

4. **Implement model fitting**
   - PyMC with refined priors
   - 4 chains × 2000 iterations
   - target_accept=0.95 (high for AR stability)

5. **Convergence diagnostics**
   - R-hat < 1.01
   - ESS > 400
   - <1% divergences
   - Posterior vs prior separation

6. **Posterior predictive checks**
   - Residual ACF (should be <0.3 vs Exp1's 0.511)
   - Count distribution fit
   - Coverage of observed data

7. **Model comparison**
   - LOO-CV vs Experiment 1
   - Expected: ELPD improvement if AR(1) helps

### If Iteration 1b Fails

**Diagnose failure mode**:
- Still >1% extreme outliers → Structural problem
- Numerical instabilities → Reparameterization needed
- AR validation poor → N=40 insufficient for AR(1)

**Alternative paths**:
- **Path A**: Further constrain (e.g., β₁ ∈ [-0.3, 1.5], σ ~ Exp(10))
- **Path B**: Simplify temporal structure (random walk, simpler correlation)
- **Path C**: Alternative likelihood (log-normal, zero-inflated)
- **Path D**: Accept Experiment 1 as final (temporal correlation not addressable)

---

## Model Complexity Evolution

| Iteration | Model | Parameters | Complexity | Status |
|-----------|-------|------------|------------|--------|
| 0 (Exp1) | NB Linear | 3 (β₀, β₁, φ) | Simple | SUCCESS |
| 1a (Exp2) | NB AR(1) | 5 (+ρ, σ) | Moderate | FAILED PPC |
| 1b (Exp2 Refined) | NB AR(1) | 5 (refined priors) | Moderate | PENDING |

**Complexity justification**: Adding 2 parameters (ρ, σ) to address residual ACF(1)=0.511 from Experiment 1

---

**Last Updated**: 2025-10-29
**Current Experiment**: experiment_2_refined (Iteration 1b)
**Status**: Awaiting prior predictive check results
