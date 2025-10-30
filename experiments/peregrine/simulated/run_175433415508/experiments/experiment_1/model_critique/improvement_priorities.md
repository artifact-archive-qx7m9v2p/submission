# Improvement Priorities for Future Models

**Baseline Model**: Negative Binomial Linear (Experiment 1)
**Status**: ACCEPTED - But extensions recommended
**Date**: 2025-10-29

---

## Executive Summary

The baseline model successfully captures trend and overdispersion but reveals clear temporal correlation structure that should be addressed. This document prioritizes improvements for subsequent experiments, ordered by evidence strength, expected impact, and feasibility.

**Priority 1 (MANDATORY)**: Address temporal correlation via AR(1) structure
**Priority 2 (CONDITIONAL)**: Investigate non-linear growth if evidence emerges
**Priority 3 (EXPLORATORY)**: Consider structural breaks only if strong evidence persists

---

## Priority 1: Temporal Correlation Structure (MANDATORY)

### Evidence for This Improvement

**Strength**: VERY STRONG (Multiple independent sources)

1. **Residual Autocorrelation**: ACF(1) = 0.511
   - Far exceeds 95% confidence bands (±0.310)
   - Highly significant (p < 0.001)
   - Persists through lag 3
   - PACF(1) = 0.511, cuts off after lag 1 (suggests AR(1))

2. **Visual Patterns**: Clear wave patterns in residual time series
   - Positive residuals cluster together
   - Negative residuals cluster together
   - Systematic oscillations visible

3. **EDA Finding**: Original ACF = 0.971 in raw data
   - Baseline model reduces to 0.511 (detrending helps)
   - But still 51% of consecutive variation unexplained

4. **Predictive Implications**: Model treats observations as independent
   - Loses information about short-term momentum
   - Prediction intervals don't account for persistence
   - One-step forecasts sub-optimal

### Proposed Solution: AR(1) Extension (Experiment 2)

**Specification**:
```
C_t ~ NegativeBinomial(exp(η_t), φ)
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t
ν_t ~ Normal(0, σ)

New parameters:
  ρ ~ Beta(20, 2)           # AR(1) coefficient (E[ρ] ≈ 0.91)
  σ ~ Exponential(2)        # Innovation SD
```

**Expected Impact**:
- **Primary**: Reduce residual ACF(1) from 0.511 to <0.1
- **Secondary**: Improve one-step-ahead predictions
- **Tertiary**: More efficient parameter estimates (use all information)
- **Uncertainty**: Tighter, more realistic prediction intervals

**Success Criteria**:
1. **ρ posterior clearly > 0.5** (narrow credible interval)
2. **Residual ACF < 0.3** (most correlation captured)
3. **ΔLOO > 5** (clear improvement over baseline)
4. **Convergence**: R-hat < 1.01, ESS > 400

**Decision Threshold**:
- **ACCEPT AR(1)** if: ΔLOO > 5 AND residual ACF < 0.3
- **REJECT AR(1)** if: ρ ≈ 0 (correlation is trend artifact) OR ΔLOO < 2×SE
- **INVESTIGATE FURTHER** if: ρ → 1 (suggests non-stationary process)

**Expected Outcome** (70% confidence):
- ρ ≈ 0.7-0.9 (substantial positive correlation)
- Residual ACF ≈ 0.2-0.5 (reduced but may not reach <0.1)
- ΔLOO ≈ 5-15 (moderate to strong improvement)
- **Likely result**: AR(1) becomes new baseline

**Alternative Scenarios**:
1. **ρ ≈ 0** (20% probability): Correlation was trend artifact
   - Baseline model adequate
   - Consider quadratic trend (Priority 2)
2. **ρ → 1** (10% probability): Non-stationary process
   - Try random walk model (Experiment 7)
   - Current model inadequate

**Implementation Considerations**:
- Use **non-centered parameterization**: ε_t = ρ×ε_{t-1} + σ×ν_t_raw where ν_t_raw ~ Normal(0,1)
- Careful initialization (avoid ρ=1 boundary)
- Monitor divergences (may indicate identifiability issues)
- Runtime: 2-5 minutes (more expensive than baseline)
- Risk: MEDIUM (convergence issues possible near boundaries)

**Why This Is Priority 1**:
- Strongest evidence (visual, statistical, theoretical)
- Addresses largest model inadequacy (51% unexplained correlation)
- Standard approach for time series
- Clear, measurable success criteria
- Modest complexity increase (2 parameters)
- High probability of success (70%)

---

## Priority 2: Non-Linear Growth (CONDITIONAL)

### Evidence for This Improvement

**Strength**: MODERATE (Suggestive but needs validation)

1. **EDA Finding**: Quadratic model R² = 0.964 vs Linear R² = 0.937
   - Improvement: 2.7% variance explained
   - Modest gain, may reflect overfitting

2. **Growth Rate Changes**: EDA identified 9.6-fold increase in growth rate
   - Early period: ~13 units/year
   - Middle period: ~125 units/year
   - Late period: ~65 units/year
   - Pattern suggests acceleration then deceleration

3. **Visual Inspection**: Some curvature in scatter plots
   - Not overwhelming
   - Could be noise with n=40

4. **Skepticism**:
   - Small sample (n=40) increases overfitting risk
   - EDA comparison was in-sample (no cross-validation)
   - Linear model fits well (no obvious systematic curvature)
   - Temporal correlation might explain apparent curvature

### When to Pursue This

**Pursue IF**:
1. **Experiment 2 (AR1) completed** and shows systematic quadratic residual pattern
2. **Residual plots** show clear U-shape or inverted-U after accounting for AR(1)
3. **Domain knowledge** suggests non-constant growth rate

**Skip IF**:
1. **AR(1) extension** eliminates apparent curvature (correlation explains it)
2. **Residuals** show no systematic pattern after AR(1)
3. **Parsimony argument**: Linear sufficient, quadratic adds noise

**Most Likely Scenario** (60% probability):
- AR(1) extension captures apparent curvature
- Quadratic term not needed
- Linear trend sufficient

### Proposed Solution: Quadratic Term (Experiment 3)

**Specification**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁×year_t + β₂×year_t²

New parameter:
  β₂ ~ Normal(0, 0.3)       # Skeptical prior (centered at zero)
```

**Expected Impact**:
- **Primary**: Capture acceleration/deceleration in growth
- **Risk**: May overfit with n=40
- **Trade-off**: +1 parameter for potentially small gain

**Success Criteria**:
1. **β₂ credible interval excludes zero** (95% CI)
2. **ΔLOO(Quad - Linear) > 4×SE** (strict threshold for 1 parameter)
3. **Pareto k < 0.7** for >80% observations (no overfitting)
4. **Residual diagnostics better** than linear

**Decision Threshold**:
- **ACCEPT quadratic** if: β₂ CI excludes 0 AND ΔLOO > 4×SE
- **REJECT quadratic** if: β₂ ≈ 0 OR ΔLOO < 2×SE
- **Parsimony bias**: Prefer linear unless quadratic clearly better

**Combined Model** (Experiment 4):
- If both AR(1) and quadratic accepted individually
- Test NB-Quad-AR1 with both features
- Requires ΔLOO > 8 compared to simpler models
- High risk of identifiability issues (curvature vs autocorrelation confounding)
- **Expected outcome** (60% confidence): One feature absorbs the other, simpler model wins

**Why This Is Priority 2 (Not Priority 1)**:
- Weaker evidence than temporal correlation
- May be artifact of unmodeled correlation
- Higher overfitting risk
- Less certain to improve fit
- Should wait for AR(1) results before pursuing

---

## Priority 3: Structural Breaks (EXPLORATORY)

### Evidence for This Improvement

**Strength**: WEAK (Single analyst finding, needs validation)

1. **EDA Finding**: Changepoint detected at year = -0.21
   - From Designer 2 only (not confirmed by others)
   - Identified via changepoint detection algorithm
   - Not visually obvious in plots

2. **Growth Rate Pattern**: Appears to change across periods
   - But could be gradual acceleration (quadratic)
   - Or noise in growth estimates
   - Or artifact of temporal correlation

3. **Skepticism**:
   - n=40 provides limited power for changepoint detection
   - Changepoint algorithms can find spurious breaks
   - Linear and quadratic models fit well without explicit break
   - No domain knowledge suggesting regime shift

### When to Pursue This

**Pursue ONLY IF**:
1. **Both AR(1) and Quadratic fail** to explain residual patterns
2. **Clear evidence** of distinct regimes in residual plots
3. **Domain experts** suggest plausible mechanism for regime shift
4. **Residual patterns** show abrupt change at specific time point

**Most Likely Scenario** (80% probability):
- No structural break present
- Apparent regime change explained by smooth acceleration or correlation
- Changepoint model not needed

### Proposed Solution: Changepoint Model (Experiment 5)

**Specification** (if pursued):
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁×year_t + β₂×I(year_t > τ)×year_t

New parameters:
  β₂ ~ Normal(0, 1.0)       # Change in slope
  τ ~ Normal(-0.21, 0.3)     # EDA-suggested location
```

**Expected Challenges**:
- **Identifiability**: Hard to distinguish changepoint from smooth curve
- **Multiple modes**: Posterior may have multiple peaks (uncertainty in τ location)
- **Computational cost**: 10-20 minutes runtime
- **High complexity**: 6-7 parameters (2 more than AR1-Quad)

**Success Criteria** (very strict):
1. **τ posterior concentrated** (not uniform or multimodal)
2. **β₂ clearly non-zero** (slope change is real)
3. **ΔLOO > 6×SE** compared to best simpler model
4. **τ estimate robust** to prior specification

**Why This Is Priority 3 (Not Higher)**:
- Weakest evidence (single analyst)
- High complexity and computational cost
- Low probability of success
- Likely explained by simpler models
- Should be last resort, not early choice

---

## Priority 4: Other Considerations (LOW PRIORITY)

### 4.1 Time-Varying Dispersion

**Evidence**: WEAK
- EDA showed variance increases over time
- But mostly explained by mean-variance relationship
- φ estimate (35.6) adequate for whole period

**When to pursue**: Only if residual variance shows strong pattern after AR(1)

**Expected outcome**: Not needed (constant φ sufficient)

### 4.2 Zero-Inflation

**Evidence**: NONE
- Zero observations: 0/40
- Minimum count: 21
- No zero-inflation concern

**When to pursue**: Never (for this dataset)

### 4.3 Gaussian Process

**Evidence**: NONE needed (stress test)
- Purpose: Verify parametric forms not missing structure
- Only if all parametric models fail systematically

**When to pursue**: After Experiments 2-4 complete and inadequate

**Expected outcome** (90% confidence): Parametric models sufficient, GP overparameterized

### 4.4 Random Walk (Experiment 7)

**Evidence**: CONDITIONAL
- Depends on Experiment 2 showing ρ → 1

**When to pursue**: Only if AR(1) hits unit root boundary

**Expected outcome**: Needed only if process is non-stationary (low probability)

---

## Improvement Strategy Flowchart

```
START: Baseline Model (Experiment 1) ACCEPTED
  |
  v
STEP 1: Experiment 2 (AR1) - MANDATORY
  |
  |-- IF ρ ≈ 0 (correlation is artifact)
  |     |
  |     v
  |   Consider Experiment 3 (Quadratic)
  |   STOP if linear adequate
  |
  |-- IF ρ ∈ [0.5, 0.95] (clear correlation)
  |     |
  |     v
  |   AR(1) becomes new baseline
  |   Check residuals for curvature
  |     |
  |     |-- IF curvature present
  |     |     |
  |     |     v
  |     |   Consider Experiment 4 (Quad-AR1)
  |     |   Strict threshold: ΔLOO > 8
  |     |
  |     |-- IF curvature absent
  |           |
  |           v
  |         STOP - AR(1) is final model
  |
  |-- IF ρ → 1 (non-stationary)
        |
        v
      Try Experiment 7 (Random Walk)

THROUGHOUT: Monitor for structural break evidence
  IF clear regime shift AND simpler models inadequate
    THEN consider Experiment 5 (Changepoint)
  ELSE skip

LAST RESORT: Experiment 6 (Gaussian Process)
  Only if all parametric forms fail
```

---

## Decision Rules for Each Priority

### Priority 1 (AR1): Clear Decision Criteria

**ACCEPT if**:
- ΔLOO > 5 (clear improvement)
- ρ posterior > 0.5 with narrow CI
- Residual ACF < 0.3

**REJECT if**:
- ΔLOO < 2×SE (no improvement)
- ρ ≈ 0 (no correlation)

**INVESTIGATE if**:
- ρ → 1 (try random walk)

### Priority 2 (Quadratic): Strict Criteria

**ACCEPT if**:
- β₂ CI excludes zero
- ΔLOO > 4×SE (strict)
- No overfitting (Pareto k good)

**REJECT if**:
- β₂ ≈ 0
- ΔLOO < 2×SE
- Parsimony favors linear

### Priority 3 (Changepoint): Very Strict

**ACCEPT if**:
- τ well-identified
- β₂ clearly non-zero
- ΔLOO > 6×SE
- Robust to priors

**REJECT if**:
- Any of above fails
- (Expected outcome)

---

## Timeline and Resource Allocation

### Immediate (Today)
- **Experiment 2 (AR1)**: Start immediately after critique complete
- **Estimated time**: 3-4 hours (prior check, SBC, fitting, PPC, critique)
- **Priority**: MANDATORY
- **Resources**: Full computational resources

### Near-Term (If needed)
- **Experiment 3 (Quadratic)**: Only if AR(1) residuals show curvature
- **Estimated time**: 2-3 hours
- **Priority**: CONDITIONAL
- **Decision point**: After Experiment 2 critique

### Later (If needed)
- **Experiment 4 (Quad-AR1)**: Only if both 2 and 3 succeed individually
- **Estimated time**: 4-5 hours (complexity)
- **Priority**: LOW
- **Decision point**: After Experiments 2 and 3 complete

### Exploratory (Unlikely)
- **Experiment 5 (Changepoint)**: Only if strong evidence persists
- **Estimated time**: 6-8 hours (computational)
- **Priority**: VERY LOW
- **Decision point**: After all others attempted

---

## Expected Final Model

### Most Likely (70% confidence)
**NB-AR(1)** from Experiment 2
- Captures trend + overdispersion + temporal correlation
- Modest complexity (5 parameters)
- Addresses all major patterns
- Baseline + AR structure sufficient

### Alternative Scenarios

**Baseline Sufficient** (15% confidence)
- AR(1) shows ρ ≈ 0 (correlation was trend artifact)
- Current model is final model
- Simplest adequate approach wins

**More Complex Model** (10% confidence)
- Both AR(1) and Quadratic needed
- NB-Quad-AR1 from Experiment 4
- Higher complexity justified by data

**Non-Stationary** (5% confidence)
- Random Walk model needed
- Current approach inadequate
- Fundamental rethinking required

---

## Success Metrics for Model Comparison

### Primary Metrics
1. **LOO-ELPD**: Higher is better (compare via ΔLOO and SE)
2. **Residual ACF**: Lower is better (target <0.1)
3. **Convergence**: Must achieve R-hat < 1.01, ESS > 400
4. **Pareto k**: >80% observations with k < 0.5

### Secondary Metrics
1. **Predictive calibration**: 90% intervals should cover ~90%
2. **Posterior predictive checks**: Bayesian p-values in [0.05, 0.95]
3. **Parameter interpretability**: Clear scientific meaning
4. **Computational efficiency**: Runtime < 30 minutes

### Trade-offs
- **Parsimony vs Fit**: Favor simpler if ΔLOO < 2×SE per parameter
- **Interpretation vs Accuracy**: Favor interpretable if ΔLOO close
- **Complexity vs Robustness**: Favor robust if convergence issues

---

## Monitoring and Adaptation

### After Each Experiment

**Document**:
1. Decision (ACCEPT/REVISE/REJECT)
2. Key metrics (ELPD, ACF, convergence)
3. Residual patterns remaining
4. Evidence for next priority

**Adapt Strategy IF**:
1. Unexpected patterns emerge
2. Multiple models fail
3. Computational barriers encountered
4. Scientific questions change

### Early Stopping Conditions

**STOP if**:
1. Adequate model found (all criteria met)
2. Diminishing returns (ΔLOO < 2×SE repeatedly)
3. Computational limits reached
4. Data quality issues discovered

**Current Status**: Baseline established, clear path forward

---

## Summary Table

| Priority | Improvement | Evidence | Expected Impact | Complexity | Probability Success | When to Pursue |
|----------|-------------|----------|-----------------|------------|---------------------|----------------|
| **1** | AR(1) correlation | Very Strong | High (ΔAC F≈0.4) | Low (+2 params) | 70% | IMMEDIATE |
| **2** | Quadratic trend | Moderate | Moderate | Low (+1 param) | 40% | After AR(1) |
| **3** | Structural break | Weak | Unknown | High (+3-4 params) | 20% | Last resort |
| 4a | Time-varying φ | Weak | Low | Medium | 10% | Unlikely |
| 4b | Zero-inflation | None | N/A | N/A | 0% | Never |
| 4c | Gaussian Process | None (test) | Low | Very High | 10% | Stress test |
| 4d | Random Walk | Conditional | High (if needed) | Low (+1 param) | 5% | If ρ→1 |

---

## Conclusion

The clear and unambiguous next step is **Experiment 2: NB-AR(1) Model**. This has the strongest evidence, highest expected impact, lowest risk, and clearest success criteria.

All other improvements are conditional on Experiment 2 results and should be pursued only if evidence justifies the added complexity.

The baseline model has successfully established:
1. What can be achieved without temporal correlation
2. How much improvement is available (ACF=0.511)
3. Clear metrics for evaluating extensions

**Next Action**: Begin Experiment 2 immediately.

---

**Document Status**: COMPLETE
**Next Review**: After Experiment 2 critique
**Priority Updates**: Based on AR(1) residual patterns
