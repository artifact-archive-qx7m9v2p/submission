# Model Decision: Experiment 1

**Date:** 2025-10-29
**Model:** Negative Binomial Quadratic Regression
**Analyst:** Model Criticism Specialist

---

## DECISION: REJECT - PROCEED TO PHASE 2

---

## Summary

The Negative Binomial Quadratic model is **REJECTED for final inference** due to fundamental violation of the temporal independence assumption. The model exhibits excellent computational performance and successfully captures trend and overdispersion, but systematically fails to model temporal correlation structure.

**Residual ACF(1) = 0.686 > 0.5 TRIGGERS PHASE 2 (Temporal Models)**

This decision follows pre-specified criteria in the experiment plan and is supported by multiple independent diagnostic tests.

---

## Justification

### Pre-Specified Trigger Met

From experiment metadata (`metadata.md`):
- **Trigger criterion:** Residual ACF(1) > 0.5 → Proceed to Phase 2
- **Observed:** Residual ACF(1) = 0.686
- **Conclusion:** Threshold substantially exceeded (not borderline)

This trigger was established a priori as evidence that temporal correlation structure is necessary for adequate model fit.

### Multiple Convergent Diagnostics

The decision is not based on a single test but synthesis of seven independent failures:

1. **Residual ACF(1) = 0.686** (threshold: 0.5) - PRIMARY TRIGGER
2. **Observed data ACF(1) = 0.944** at 100th percentile (p = 0.000)
3. **Maximum value** at 99.4th percentile (p = 0.994)
4. **Skewness** at 0.1th percentile (p = 0.999)
5. **Kurtosis** at 0.0th percentile (p = 1.000)
6. **Coverage = 100%** (excessive, target: 90-98%)
7. **Temporal wave pattern** in residuals vs time plot

All diagnostics point to the same root cause: temporal dependencies not captured by model.

### Mechanistic Understanding

The data exhibits **ACF(1) = 0.944**, meaning:
- 89% of variance at time t is predictable from time t-1
- Observations are highly non-independent
- Process has strong memory/persistence
- Independence assumption is fundamentally violated

This is not a numerical artifact but a structural feature of the data-generating process.

### Visual Evidence

Key plots showing systematic failures:
- **`residual_diagnostics.png` Panel B:** Clear sinusoidal wave pattern over time
- **`residual_diagnostics.png` Panel C:** ACF(1) far above Phase 2 threshold (orange line at 0.5)
- **`ppc_dashboard.png` Panel C:** Observed trajectory smooth, replications choppy
- **`test_statistics.png` ACF(1) panel:** Observed far right of posterior predictive distribution

These patterns are diagnostic signatures of unmodeled temporal correlation.

---

## What This Decision Does NOT Mean

### This is NOT a Failed Experiment

The model:
- ✓ Converged perfectly (R̂ = 1.000, ESS > 2100, 0 divergences)
- ✓ Captures overall trend (R² = 0.883)
- ✓ Handles overdispersion (φ = 16.6)
- ✓ Can recover known parameters (SBC validation)
- ✓ Serves as excellent parametric baseline

**The model works correctly for i.i.d. data.** The issue is data structure, not model implementation.

### This is NOT a Computational Problem

Perfect convergence diagnostics demonstrate:
- Model is well-specified for independent data
- Priors are well-calibrated (after adjustment)
- Posterior is well-defined and thoroughly sampled
- No numerical instabilities or pathologies

**The computational workflow succeeded.** The model is simply inappropriate for temporal data.

### This is NOT Unexpected

From EDA report warnings:
- "Temporal autocorrelation (r = 0.989): Violates independence assumption"
- "Need time series structure or correlated errors"
- Expected to trigger Phase 2 based on ACF analysis

**This outcome was predicted and is part of the planned model progression.**

---

## Why NOT ACCEPT?

**ACCEPT criteria from metadata:**
- ✗ Coverage 85-98% (observed: 100%)
- ✗ Residual ACF(1) < 0.3 (observed: 0.686)
- ✗ No systematic patterns (observed: temporal waves)
- ✗ Valid for scientific conclusions (independence assumption violated)

**Specific violations:**
1. ACF(1) exceeds borderline threshold (0.6) and Phase 2 threshold (0.5)
2. Seven test statistics with extreme Bayesian p-values (< 0.05 or > 0.95)
3. Clear systematic patterns in residuals vs time
4. Credible intervals not trustworthy (independence assumption false)

**Cannot make valid scientific inference with violated fundamental assumptions.**

---

## Why NOT REVISE?

**REVISE is appropriate when:**
- Issues are fixable within current model class
- Tweaking priors, predictors, or functional form would help
- Core structure is sound, just needs refinement

**Why this doesn't apply here:**

1. **Cannot fix temporal correlation within i.i.d. framework**
   - Adjusting priors won't add temporal structure
   - Adding polynomial terms won't model autocorrelation
   - Changing marginal distribution won't capture dependencies

2. **No available fixes within model class**
   - No covariates to add (only time available)
   - Polynomial already flexible (quadratic captures curvature)
   - Dispersion already well-modeled (NegBin appropriate)

3. **Core independence assumption is wrong**
   - Not a parameterization issue
   - Not a prior specification issue
   - Fundamental structural mismatch with data

**Need different model class, not model refinement.**

---

## Why REJECT? (Proceed to Phase 2)

**REJECT is appropriate when:**
- ✓ Fundamental misspecification evident
- ✓ Pre-specified trigger met (ACF(1) > 0.5)
- ✓ Need different model class
- ✓ Clear path forward exists

**This is the correct decision because:**

1. **Decision rule clearly triggered:**
   - Residual ACF(1) = 0.686 > 0.5 threshold
   - Not a close call (37% above threshold)
   - Multiple supporting diagnostics

2. **Mechanistic understanding:**
   - Data has temporal memory
   - Each observation depends on previous
   - Independence assumption fundamentally violated

3. **Clear alternatives exist:**
   - AR models can capture lag-1 correlation
   - State-space models allow smooth evolution
   - Temporal extensions preserve successful elements

4. **Expected outcome:**
   - Temporal models should reduce ACF(1) to <0.3
   - Coverage should normalize to 90-98%
   - Substantial ELPD improvement expected

**The decision framework leads unambiguously to REJECT + Phase 2.**

---

## Next Steps: Phase 2 Temporal Models

### Primary Recommendation: Experiment 3 or 4

**Experiment 3: AR(1) Negative Binomial**
- Add autoregressive random effect to current model
- Structure: `μ[t] = exp(β₀ + β₁·year + β₂·year² + α[t])`
- Where: `α[t] ~ Normal(ρ·α[t-1], σ_α)`
- Expected: ρ ≈ 0.7 based on residual ACF(1) = 0.686

**Experiment 4: State-Space Model**
- If AR(1) insufficient (ACF(2) = 0.423 also substantial)
- Allows smooth latent state evolution
- More flexible but more complex

### Elements to Preserve

**Keep from Experiment 1:**
- ✓ Negative Binomial distribution (overdispersion well-handled)
- ✓ Log link function (positive predictions, multiplicative effects)
- ✓ Quadratic time trend (captures acceleration)
- ✓ Adjusted prior scales (prevent extreme predictions)
- ✓ PyMC + NUTS sampler (efficient, converges well)

### Elements to Add

**New in Phase 2:**
- ✓ Temporal correlation structure (AR, state-space, or ARMA)
- ✓ Correlation parameter(s) (ρ for AR, process variance for state-space)
- ✓ Initial state priors (for t=1)

### Success Criteria for Phase 2

A temporal model will be considered successful if:

1. **Residual ACF(1) < 0.3** (substantial reduction from 0.686)
2. **Coverage 90-98%** (not excessive 100%)
3. **ACF(1) test statistic p-value ∈ [0.1, 0.9]** (not extreme)
4. **ELPD improvement > 10** (substantial predictive gain)
5. **Convergence maintained** (R̂ < 1.01, ESS > 400)
6. **No new systematic patterns** in residuals

### What NOT to Try

**Don't:**
- ❌ Experiment 2 only (exponential trend) - will still show temporal issues
- ❌ Add more polynomial terms - won't fix autocorrelation
- ❌ Switch to different marginal distribution - temporal problem remains
- ❌ Fit overly complex models first - start simple (AR(1))

**Do:**
- ✓ Start with AR(1) extension of current model
- ✓ Validate residual ACF drops substantially
- ✓ Compare to Experiment 1 baseline using LOO-CV
- ✓ Consider mechanistic interpretation of correlation parameter

---

## Role of Experiment 1 Going Forward

### As Parametric Baseline

**This model remains valuable for:**

1. **Comparison:** Quantifies improvement from adding temporal structure
2. **Interpretation:** Simpler to explain (4 parameters vs 6+)
3. **Trend estimates:** β₁ = 0.84, β₂ = 0.10 are informative about growth
4. **Overdispersion:** φ = 16.6 quantifies variance structure
5. **Diagnostics:** Demonstrates what residual autocorrelation looks like

**Report as:** "Baseline parametric model - inadequate for inference but useful for comparison"

### For Model Selection

**Use in LOO-CV comparison:**
- Calculate ELPD difference vs temporal models
- Quantify predictive improvement
- Test whether temporal complexity justified
- Assess trade-off between simplicity and adequacy

**Expected:** Temporal models should show ELPD improvement of 10-20+ points, confirming Phase 2 was necessary.

---

## Confidence in Decision

### Very High Confidence (95%+)

**Reasons for high confidence:**

1. **Clear threshold exceedance:** 0.686 vs 0.5 (not borderline)
2. **Multiple independent tests:** 7 diagnostics point to same issue
3. **Visual and quantitative agreement:** All evidence aligned
4. **Pre-specified decision rule:** Not post-hoc rationalization
5. **Mechanistic understanding:** Clear explanation of failure mode
6. **EDA prediction confirmed:** Expected this outcome

**No reasonable ambiguity.** All evidence points unambiguously to temporal structure being necessary.

### What Could Change This Decision?

**Nothing, unless:**
- Data collection error discovered (very unlikely)
- ACF calculation error (checked multiple ways, consistent)
- Different scientific objectives (not changing mid-analysis)

**The decision is robust to:**
- Prior sensitivity (posterior not prior-dominated)
- Computational choices (perfect convergence)
- Diagnostic method (multiple approaches agree)

---

## Summary Table

| Aspect | Status | Evidence |
|--------|--------|----------|
| **Convergence** | ✓ PERFECT | R̂=1.000, ESS>2100, 0 divergences |
| **Trend Capture** | ✓ GOOD | R²=0.883, mean p=0.668 |
| **Overdispersion** | ✓ GOOD | φ=16.6, variance p=0.910 |
| **Temporal Independence** | ✗ VIOLATED | ACF(1)=0.686>0.5 threshold |
| **Coverage Calibration** | ✗ EXCESSIVE | 100% vs target 90-98% |
| **Test Statistics** | ✗ POOR | 7 extreme p-values |
| **Residual Patterns** | ✗ SYSTEMATIC | Clear temporal waves |
| **Overall Adequacy** | ✗ INADEQUATE | Cannot proceed to inference |

---

## Final Decision Statement

**The Negative Binomial Quadratic model (Experiment 1) is REJECTED for final scientific inference due to fundamental violation of temporal independence assumptions (residual ACF(1) = 0.686 > 0.5 threshold).**

**Recommendation: Proceed immediately to Phase 2 temporal modeling (Experiment 3: AR Negative Binomial or Experiment 4: State-Space Model).**

**The model serves as a well-calibrated parametric baseline and has successfully revealed the importance of temporal structure in this data.**

---

**Decision Date:** 2025-10-29
**Analyst:** Model Criticism Specialist
**Status:** Final - No appeal necessary (clear-cut case)
**Next Action:** Fit Experiment 3 (AR model)
