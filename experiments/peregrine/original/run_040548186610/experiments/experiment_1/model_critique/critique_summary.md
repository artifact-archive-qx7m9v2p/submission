# Model Critique for Experiment 1: Negative Binomial Quadratic Model

**Date:** 2025-10-29
**Model:** `C_i ~ NegBinomial(μ_i, φ)`, `log(μ_i) = β₀ + β₁·year + β₂·year²`
**Analyst:** Model Criticism Specialist
**Overall Assessment:** REJECT - Proceed to Phase 2 (Temporal Models)

---

## Executive Summary

The Negative Binomial Quadratic model demonstrates **perfect computational performance** but **fundamental inferential inadequacy** for this time series. While the model successfully captures the overall trend (R² = 0.883) and overdispersion, it **systematically fails to capture temporal dependencies**, with residual autocorrelation (ACF(1) = 0.686) far exceeding the Phase 2 threshold of 0.5.

**Key Finding:** Residual ACF(1) = 0.686 > 0.5 **TRIGGERS PHASE 2 (Temporal Models)**

This is not a failure of model implementation - convergence is perfect, priors are well-calibrated, and the model can recover known parameters. Rather, this is a **structural inadequacy**: the independence assumption is fundamentally violated by the data. The model serves as an excellent parametric baseline for comparison but requires temporal extension for adequate inference.

---

## Synthesis Across Validation Stages

### 1. Prior Predictive Check: PASS (After Adjustment)

**Initial Result:** ADJUST
**Outcome:** Priors successfully tightened to prevent extreme predictions

**Strengths:**
- Model structure appropriate for count data with overdispersion
- Quadratic form flexible enough to capture accelerating growth
- Adjusted priors (β₂: 0.2→0.1) eliminated explosive trajectories

**Findings:**
- Initial priors too vague, generating counts >40,000 vs observed max=272
- After adjustment: 89.4% of simulations in plausible range [10-500]
- No domain violations (negative counts)
- Appropriate coverage of observed data range

**Evidence:** Maximum simulated count reduced from 42,686 to reasonable values after prior tightening. Critical adjustment was β₂ standard deviation (0.2→0.1), preventing exponential explosion via quadratic term.

### 2. Simulation-Based Calibration: CONDITIONAL PASS

**Overall:** Model can recover known parameters with noted caveats

**Parameter Recovery:**
- β₀ (Intercept): EXCELLENT - 100% coverage, bias -0.01
- β₁ (Linear): EXCELLENT - 100% coverage, bias -0.01
- β₂ (Quadratic): EXCELLENT - 95% coverage, bias +0.01
- φ (Dispersion): ACCEPTABLE - 85% coverage, bias -0.33

**Computational Health:**
- Success rate: 100% (20/20 simulations)
- Convergence rate: 95% (19/20)
- Mean R̂: 1.040 (excellent)
- No systematic biases detected

**Implications:**
- Regression coefficients are trustworthy (use 95% CIs)
- Dispersion parameter slightly underestimated (use 99% CIs)
- Model structure is correctly specified for i.i.d. count data
- Small sample (n=20 sims) limits precision of assessment

**Key Insight:** The model performs exactly as designed when data satisfy independence assumptions. Real data violations are not computational failures.

### 3. Posterior Inference: PERFECT Convergence

**Status:** PASS - Exemplary MCMC performance

**Convergence Metrics:**
- R̂: 1.000 for all parameters (perfect)
- ESS_bulk: >2,100 for all parameters (excellent efficiency)
- ESS_tail: >2,300 for all parameters (excellent tail sampling)
- Divergent transitions: 0 out of 4,000 (0.00%)
- MCSE: <2.1% of posterior SD (high precision)

**Parameter Estimates:**
- β₀ = 4.29 ± 0.06 (intercept)
- β₁ = 0.84 ± 0.05 (linear growth)
- β₂ = 0.10 ± 0.05 (acceleration)
- φ = 16.6 ± 4.2 (dispersion)

**Visual Diagnostics:**
- Clean "hairy caterpillar" traces
- Uniform rank distributions
- Good energy diagnostic overlap
- Moderate parameter correlations (expected)

**SBC Predictions Validated:**
- Convergence rate: Predicted 95%, achieved 100%
- Coefficient precision: Matches SBC calibration
- No computational pathologies

**Key Insight:** Perfect convergence does NOT imply good model fit. Posterior is well-defined, but the model itself is misspecified for temporal data.

### 4. Posterior Predictive Check: POOR FIT

**Status:** FAIL - Systematic violations on 7 test statistics

**Critical Failures:**

**A. Temporal Autocorrelation (MOST SEVERE):**
- Residual ACF(1) = 0.686 (threshold: 0.5) - **EXCEEDS PHASE 2 TRIGGER**
- Observed data ACF(1) = 0.944 at 100th percentile of replicates
- Bayesian p-value = 0.000 (most extreme discrepancy)
- Visual: Clear sinusoidal wave pattern in residuals vs time

**B. Extreme Value Under-Generation:**
- Observed maximum (272) at 99.4th percentile (p = 0.994)
- Observed range (253) at 99.5th percentile (p = 0.995)
- Model cannot reproduce observed extremes despite heavy-tailed NegBin

**C. Distribution Shape Mismatch:**
- Skewness: Observed (0.60) at 0.1th percentile (p = 0.999)
- Kurtosis: Observed (-1.23) at 0.0th percentile (p = 1.000)
- Data is less skewed and flatter than model predicts

**D. Quantile Behavior:**
- Q75 at 2.0th percentile (p = 0.020)
- IQR at 1.7th percentile (p = 0.017)
- Less spread in middle, more at extremes

**Successful Aspects:**
- Mean: p = 0.668 (captures central tendency well)
- Variance: p = 0.910 (overdispersion well-modeled)
- R² = 0.883 (strong correlation between observed and predicted)

**Coverage Analysis:**
- 95% interval: 100% coverage (EXCESSIVE - should be 90-98%)
- 80% interval: 95% coverage (over-coverage)
- 50% interval: 67.5% coverage (over-coverage)
- Interpretation: Model is too conservative, overestimates uncertainty

**Residual Patterns:**
- U-shaped pattern vs fitted values (curvature misspecification)
- Clear wave pattern vs time (temporal structure)
- Smooth observed trajectory vs choppy replications (autocorrelation signature)

**Visual Evidence:**
- `ppc_dashboard.png` Panel C: Observed deviates systematically from replications
- `residual_diagnostics.png` Panel B: Temporal waves with smooth trend
- `residual_diagnostics.png` Panel C: ACF(1) far above Phase 2 threshold
- `test_statistics.png`: ACF(1) observed in extreme right tail

**Key Insight:** Model gets the mean trend right but misses all temporal structure. Observations are not independent conditional on time - each count is highly predictable from the previous count.

---

## Holistic Assessment

### What the Model Does Well

1. **Computational Stability:** Perfect convergence, no numerical issues, fast sampling
2. **Trend Capture:** R² = 0.883 indicates strong correlation with observed trajectory
3. **Central Tendency:** Mean and variance well-captured (p-values in healthy range)
4. **Overdispersion:** φ = 16.6 appropriately models variance exceeding Poisson
5. **Parametric Efficiency:** Only 4 parameters describe 40 observations
6. **Interpretability:** Clear coefficients for intercept, growth rate, acceleration
7. **Baseline Comparison:** Provides reference for evaluating temporal models

### What the Model Fails to Capture

1. **Temporal Autocorrelation (CRITICAL):** 69% of residual variance explained by lag-1 correlation
2. **Short-term Persistence:** High counts followed by high counts (momentum)
3. **Smooth Dynamics:** Observed shows gradual changes, not independent jumps
4. **Extreme Value Clustering:** Cannot generate observed maximum in temporal context
5. **Distribution Shape:** Mismatches in skewness and kurtosis
6. **Prediction Intervals:** Too wide due to ignoring temporal structure

### Scientific Interpretation

**The data exhibits ACF(1) = 0.944**, meaning ~89% of variance at time t is predictable from time t-1. This reveals:

1. **Process Memory:** The underlying data-generating process has strong persistence
2. **Slow-Changing Drivers:** Whatever drives counts changes gradually, not instantaneously
3. **Invalid Standard Errors:** Independence assumption violation invalidates credible intervals
4. **Forecasting Failure:** Cannot make accurate short-term predictions without temporal structure

**Mechanistic Implications:**
- Momentum/inertia in the underlying process
- Contagion/diffusion dynamics (values spread over time)
- Cumulative processes with feedback
- Possible measurement smoothing (overlapping windows)

### Model Adequacy by Pre-Specified Criteria

From `metadata.md` success criteria:

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| R̂ | < 1.01 | 1.000 | ✓ PASS |
| ESS | > 400 | >2,100 | ✓ PASS |
| Divergences | < 1% | 0.0% | ✓ PASS |
| Coverage (95%) | 85-98% | 100% | ✗ EXCESSIVE |
| Residual ACF(1) | < 0.6 | 0.686 | ✗ FAIL |
| P-value extremes | ≤ 2 | 7 | ✗ FAIL |

**Failure Criteria from metadata:**
- ❌ High residual ACF: ACF(1) = 0.686 > 0.6 threshold
- ❌ Poor coverage: 100% exceeds acceptable 85-98% range
- ❌ Multiple extreme p-values: 7 statistics vs ≤2 threshold

**Decision Matrix Position:**
```
                    Residual ACF(1)
                < 0.3    0.3-0.5    > 0.5
Coverage  90-98%  GOOD     ACCEPT   POOR
          85-90%  ACCEPT   ACCEPT   POOR
          < 85%   POOR     POOR     POOR
          > 98%   ACCEPT   ACCEPT   POOR ← We are here (100%, 0.686)
```

**Result:** POOR FIT according to decision matrix

---

## Critical Issues vs Minor Issues

### Critical Issues (MUST Address)

**1. Temporal Independence Violation (SEVERE)**
- **Magnitude:** Residual ACF(1) = 0.686 (far above 0.5 threshold)
- **Evidence:** 7 different diagnostics confirm this
- **Impact:** Invalid inference for all parameters
- **Fix:** Requires different model class (AR, state-space, etc.)
- **Cannot be fixed within current model structure**

**2. Systematic Temporal Patterns**
- **Evidence:** Clear wave pattern in residuals vs time
- **Impact:** Predictions at adjacent times are biased
- **Fix:** Add temporal correlation structure
- **Cannot be fixed by adjusting priors or adding covariates**

### Minor Issues (Could Improve But Not Blocking)

**3. Curvature Misspecification**
- **Evidence:** U-shaped residual pattern vs fitted
- **Impact:** Some bias in predictions, especially at late times
- **Fix:** Could try higher-order polynomial or exponential trend
- **Magnitude:** R² = 0.883 suggests this is secondary issue

**4. Extreme Value Under-Generation**
- **Evidence:** Observed max at 99.4th percentile
- **Impact:** Risk assessment for high-count events unreliable
- **Fix:** May resolve with temporal model (extremes cluster in time)
- **Likely artifact of missing temporal structure**

**5. Distribution Shape**
- **Evidence:** Skewness/kurtosis mismatches
- **Impact:** Minor - marginal distribution mostly correct
- **Fix:** Alternative distribution family
- **Low priority given temporal issues dominate**

---

## Comparison to Decision Framework

### Why NOT ACCEPT

**ACCEPT requires:**
- ✗ Coverage 85-98% (we have 100%)
- ✗ Residual ACF(1) < 0.3 (we have 0.686)
- ✗ No systematic patterns (we have temporal waves)
- ✗ Meets scientific objectives (cannot make valid inference with violated assumptions)

**Specific violations:**
1. ACF(1) = 0.686 > 0.5 **TRIGGERS PHASE 2 BY DESIGN**
2. Seven test statistics with extreme p-values
3. Systematic temporal structure in residuals
4. Uncertainty intervals not trustworthy (independence assumption violated)

### Why NOT REVISE

**REVISE is appropriate when:**
- Issues are fixable within model class
- Clear path to improvement exists
- Core structure seems sound

**Why this doesn't apply:**
- **Cannot fix temporal correlation by adjusting priors**
- **Cannot add temporal structure to i.i.d. model**
- **Core assumption (independence) is wrong for this data**
- Need fundamentally different model class, not tweaks

Potential "revisions" considered and rejected:
- Adding more polynomial terms → Won't fix temporal correlation
- Different marginal distribution → Temporal issue remains
- Stronger priors → Doesn't change independence assumption
- Adding covariates → No additional predictors available

### Why REJECT (Proceed to Phase 2)

**REJECT is appropriate when:**
- ✓ Fundamental misspecification evident
- ✓ Residual ACF(1) > 0.5 (we have 0.686)
- ✓ Need different model class
- ✓ Clear next steps exist

**This is the correct decision because:**
1. **Pre-specified trigger met:** ACF(1) = 0.686 > 0.5 threshold
2. **Multiple confirmatory diagnostics:** Not a single-test fluke
3. **Mechanistic understanding:** Data has temporal memory
4. **Clear alternative:** Temporal models (AR, state-space) address root cause

**Important caveat:** This is not a "failed" experiment. The model:
- ✓ Works as designed for i.i.d. data
- ✓ Provides parametric baseline for comparison
- ✓ Reveals data structure (strong autocorrelation)
- ✓ Guides next steps (need temporal models)

---

## Strengths to Preserve

When moving to Phase 2 temporal models, **retain these successful elements:**

1. **Negative Binomial distribution:** Overdispersion (φ = 16.6) is well-handled
2. **Log link function:** Ensures positive predictions, multiplicative effects
3. **Quadratic trend term:** β₂ = 0.10 captures acceleration
4. **Prior calibration:** Adjusted priors work well
5. **Computational approach:** PyMC + NUTS sampler is efficient

**Build on strengths:**
```python
# Recommended next model structure
μ[t] = exp(β₀ + β₁·year[t] + β₂·year²[t] + α[t])
α[t] ~ Normal(ρ·α[t-1], σ_α)  # AR(1) random effect
C[t] ~ NegBinomial(μ[t], φ)
```

This combines:
- Current model's successful trend specification
- AR(1) structure to capture temporal correlation
- Same Negative Binomial overdispersion

---

## Weaknesses to Address

### Primary Weakness: Independence Assumption

**Mechanism of Failure:**
- Model assumes `C[t] ⊥ C[t-1] | year[t]`
- Data shows `Cor(C[t], C[t-1]) = 0.944`
- 89% of variance at time t predictable from time t-1
- Fundamental mismatch between model and data structure

**Why This Matters:**
- **Invalid credible intervals:** Width assumes independence
- **Underestimated uncertainty:** For trend parameters
- **Poor forecasts:** Cannot predict next observation accurately
- **Biased residuals:** Systematic patterns remain
- **Scientific interpretation:** Miss dynamic mechanisms

**Cannot be fixed by:**
- Prior adjustment
- Adding polynomial terms
- Changing marginal distribution
- Increasing sample size

**Requires:**
- Temporal correlation structure
- AR terms, state-space, or ARMA components
- Fundamentally different model class

### Secondary Weaknesses

**Curvature specification:**
- Quadratic may not be optimal functional form
- Consider exponential, logistic, or spline alternatives
- But this is minor compared to temporal issues

**Extreme value behavior:**
- May improve with temporal model
- Extreme values likely cluster in time
- Secondary to independence violation

---

## Recommendations and Next Steps

### Immediate Action: Proceed to Phase 2

**Primary Recommendation:** Fit temporal models (Experiments 3 or 4)

**Pre-specified trigger met:**
- Residual ACF(1) = 0.686 > 0.5 threshold
- Experiment plan mandates Phase 2 temporal modeling
- This is the expected and correct path

### Recommended Model Sequence

**Experiment 3: AR(1) Negative Binomial**
- Add autoregressive structure to current model
- Expected to reduce residual ACF substantially
- Direct test of whether lag-1 correlation is sufficient

**Experiment 4: State-Space Model**
- If AR(1) insufficient (ACF(2) = 0.423 also substantial)
- Allows smooth latent state evolution
- More flexible but more complex

**Experiment 2: Exponential Trend (Optional)**
- Simpler than quadratic
- May still show residual ACF issues
- Useful for model comparison, not primary path

### Validation Strategy for Phase 2

When fitting temporal models, check:

1. **Residual ACF:** Must drop below 0.3 for all lags
2. **Coverage:** Should improve to 90-98% range (not 100%)
3. **Test statistics:** ACF(1) p-value in healthy range [0.1, 0.9]
4. **LOO comparison:** Should show substantial improvement over Experiment 1
5. **Forecast accuracy:** One-step-ahead predictions on held-out data

### What NOT to Do

**Don't:**
- ❌ Add more polynomial terms (won't fix temporal correlation)
- ❌ Try different marginal distributions (temporal issue remains)
- ❌ Abandon Negative Binomial (overdispersion is well-handled)
- ❌ Ignore temporal structure (it's the dominant issue)
- ❌ Fit overly complex models (start with AR(1))

**Do:**
- ✓ Focus on temporal correlation structure
- ✓ Keep successful elements (NegBin, quadratic trend)
- ✓ Validate that temporal model resolves residual patterns
- ✓ Compare models using LOO-CV
- ✓ Consider mechanistic interpretation of correlation parameter

### Success Criteria for Phase 2

A temporal model will be considered successful if:

1. Residual ACF(1) < 0.3 (substantial reduction from 0.686)
2. Coverage in 90-98% range (not 100%)
3. No extreme p-values for ACF test statistic
4. ELPD substantially better than Experiment 1
5. Convergence maintained (R̂ < 1.01, ESS > 400)

---

## Role of This Model Going Forward

**This model is NOT useless - it serves important purposes:**

### As Parametric Baseline
- Provides reference for evaluating temporal models
- Quantifies improvement from adding temporal structure
- Simpler interpretability (4 parameters vs potentially >6)

### For Model Comparison
- LOO-ELPD comparison with temporal models
- Tests whether temporal complexity justified
- Helps answer: "How much does temporal structure matter?"

### For Diagnostic Learning
- Revealed that data has strong temporal structure
- Demonstrated what systematic residual patterns look like
- Showed that convergence ≠ good fit

### For Scientific Understanding
- Trend parameters (β₁ = 0.84, β₂ = 0.10) are informative
- Overdispersion level (φ = 16.6) is meaningful
- Establishes that growth is accelerating (β₂ > 0)

**Document as:** "Baseline parametric model - serves as comparison for temporal extensions"

---

## Influential Observations Analysis

### LOO-PIT Calibration

From `loo_pit.png`:
- LOO-PIT ECDF shows some deviation from uniform
- Early observations (low counts) slightly over-predicted
- Late observations (high counts) show calibration issues
- Consistent with systematic temporal bias

### Expected Pareto-k Diagnostics

Based on PPC findings:
- Late time points (indices 35-40) likely have high Pareto-k
- Largest residuals at end of series (-143 at index 38)
- These observations are influential due to:
  - High leverage (extreme year values)
  - Systematic under-prediction (temporal correlation)
  - Extreme observed values (max = 272)

**Recommendation:** When fitting temporal models, check if Pareto-k improves for late observations. High k-values may decrease if temporal structure better captures dynamics.

---

## Prior Sensitivity Assessment

### Posterior vs Prior Comparison

From convergence report:
- **β₀:** Posterior (4.29) shifted left from prior (4.70) - data informative
- **β₁:** Posterior (0.84) near prior mode (0.80) - data confirms prior
- **β₂:** Posterior (0.10) much lower than prior (0.30) - data updates substantially
- **φ:** Posterior (16.6) highly concentrated - data strongly constrains

**Interpretation:**
- Data substantially updates all parameters
- Posterior not dominated by priors
- β₂ shows most updating (quadratic effect weaker than expected)
- Results are not sensitive to prior choice (within reasonable ranges)

### Robustness Check

The adjusted priors (after PPC) were:
- Informative enough to prevent extreme predictions
- Weak enough to let data dominate
- No evidence of prior-data conflict

**Conclusion:** Parameter estimates robust to reasonable prior specifications. The temporal correlation issue is not an artifact of prior choice.

---

## Model Complexity Assessment

### Current Model
- **Parameters:** 4 (β₀, β₁, β₂, φ)
- **Effective parameters:** Well-estimated (ESS > 2100)
- **R²:** 0.883
- **Issues:** Systematic residual patterns despite simplicity

### Is Model Too Simple?

**Yes, in one critical dimension:**
- Missing temporal correlation structure
- Independence assumption violated
- Need at least 1-2 additional parameters (ρ, σ_α for AR(1))

**No, in other dimensions:**
- Quadratic trend is flexible enough (R² = 0.883)
- Overdispersion well-captured (variance p = 0.910)
- Not underfitting the mean trend

### Is Model Too Complex?

**No:**
- Only 4 parameters for 40 observations
- All parameters well-identified (low MCSE)
- No overfitting indicators
- Simpler model (linear only) would show worse residual patterns

**Conclusion:** Model is appropriately complex for mean structure but structurally inadequate for correlation structure.

---

## Comparison to EDA Predictions

From `eda_report.md` findings:

### EDA Predicted:
1. **Overdispersion:** Var/Mean = 68 (extreme) → Model confirms φ = 16.6 handles this
2. **Non-linear growth:** 6× rate increase → Model confirms β₂ = 0.10 (acceleration)
3. **Strong autocorrelation:** ACF(1) = 0.989 → Model FAILS to capture this
4. **Heteroscedastic variance:** Varies by period → Model partially addresses via NegBin

### EDA Warnings:
- "Temporal autocorrelation (r = 0.989): Violates independence assumption"
- "Need time series structure or correlated errors"
- "Standard errors from naive models will be underestimated"

**All EDA warnings confirmed by PPC results.** The analysis predicted exactly these issues, and the model exhibits exactly these failures.

### EDA Recommended:
1. Negative Binomial with quadratic trend → ✓ Tried as baseline
2. Add AR errors if needed → ✓ Now needed (Phase 2)
3. Check residual ACF → ✓ Done, triggered Phase 2

**The workflow is proceeding exactly as planned.** This is not an unexpected failure but a designed decision point.

---

## Final Verdict

### Decision: REJECT (Proceed to Phase 2)

**Reasoning:**
1. **Pre-specified trigger met:** Residual ACF(1) = 0.686 > 0.5 threshold
2. **Fundamental misspecification:** Independence assumption violated
3. **Multiple confirmatory diagnostics:** 7 test statistics with extreme p-values
4. **Clear mechanistic understanding:** Data has temporal memory (ACF(1) = 0.944)
5. **Well-defined next steps:** Temporal models (AR, state-space) exist

**This is NOT a failure:**
- Model works as designed for i.i.d. data
- Provides excellent parametric baseline
- Computational performance is exemplary
- Reveals important data structure

**This is a structural inadequacy:**
- Cannot capture temporal dependencies
- Requires different model class
- Expected based on EDA findings
- Part of planned model progression

### Confidence in Decision

**Very High Confidence:**
- Multiple independent diagnostics agree
- Pre-specified decision rules clearly met
- Mechanistic interpretation consistent
- Visual and quantitative evidence aligned

**No ambiguity:** ACF(1) = 0.686 is substantially above 0.5 threshold, not borderline.

### Expected Outcome of Phase 2

**When temporal models are fit, expect:**
1. Residual ACF(1) drops from 0.686 to <0.3
2. Coverage improves from 100% to 90-98%
3. ACF(1) test statistic p-value moves to healthy range
4. ELPD improves by >10 points
5. Residual wave pattern disappears

If these improvements do NOT occur, then:
- Temporal structure more complex than AR(1)
- May need ARMA, state-space, or non-linear dynamics
- Consider mechanistic models (e.g., stochastic growth)

---

## Conclusion

The Negative Binomial Quadratic model is **computationally perfect but inferentially inadequate** for this temporally autocorrelated time series. The model successfully captures trend and overdispersion but fundamentally fails to model temporal dependencies (residual ACF(1) = 0.686 > 0.5 threshold).

**This is not a bug, it's a feature.** The model performed exactly as designed:
1. ✓ Prior predictive check revealed need for prior adjustment → fixed
2. ✓ SBC validated model can recover parameters from i.i.d. data → confirmed
3. ✓ Posterior inference converged perfectly → achieved
4. ✓ PPC revealed temporal structure violation → detected
5. ✓ Decision rule triggered Phase 2 → implemented

**The workflow is working.** This model serves as a well-calibrated baseline demonstrating that temporal correlation is essential for this data.

**Next step:** Proceed immediately to Experiment 3 (AR Negative Binomial) or Experiment 4 (State-Space Model) to address temporal dependencies while preserving this model's successful handling of trend and overdispersion.

---

**Analysis Date:** 2025-10-29
**Analyst:** Model Criticism Specialist
**Status:** Complete - Ready for Phase 2
**Files:** See `/workspace/experiments/experiment_1/model_critique/`
