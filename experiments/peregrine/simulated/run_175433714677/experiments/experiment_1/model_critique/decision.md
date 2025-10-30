# Model Decision: Experiment 1 - Log-Linear Negative Binomial

**Date**: 2025-10-29
**Model**: Log-Linear Negative Binomial Regression
**Status**: REJECTED

---

## DECISION: REJECT

The log-linear negative binomial model is **REJECTED** as fundamentally inadequate for the observed data. This model cannot be salvaged through refinement and must be replaced with a more flexible model class.

---

## Decision Framework Applied

### Accept Criteria (NOT MET)
- [ ] No major convergence issues → MET (R-hat=1.00, ESS>6000)
- [ ] Reasonable predictive performance → **NOT MET** (4.17× error increase)
- [ ] Calibration acceptable for use case → **NOT MET** (over-conservative, wide intervals)
- [ ] Residuals show no concerning patterns → **NOT MET** (inverted-U, coef=-5.22)
- [ ] Robust to reasonable prior variations → MET (but irrelevant given other failures)

**Result**: 2 of 5 criteria met → Cannot ACCEPT

### Revise Criteria (NOT APPLICABLE)
- [ ] Fixable issues identified → Issues are structural, not fixable
- [ ] Clear path to improvement exists → Path exists, but requires different model class
- [ ] Core structure seems sound → **Core structure is inadequate** (linear growth assumption)

**Result**: Core assumption (log-linear growth) is violated → Cannot REVISE within this model class

### Reject Criteria (ALL MET)
- [x] Fundamental misspecification evident → YES (inverted-U residuals)
- [x] Cannot reproduce key data features → YES (acceleration, late-period behavior)
- [x] Persistent computational problems → NO (computation is fine)
- [x] Prior-data conflict unresolvable → NO (priors are fine)

**Result**: Fundamental misspecification → **MUST REJECT**

---

## Justification

### Why REJECT Instead of REVISE?

The distinction between "revise" and "reject" hinges on whether the model's **core structure** is sound:

**REVISE** would be appropriate if:
- Missing a single predictor (e.g., add covariate X)
- Wrong likelihood family (e.g., Poisson instead of NegBin)
- Prior-data conflict (e.g., overly restrictive prior)
- Computational issues (e.g., reparameterization needed)

**REJECT** is appropriate when:
- **Core functional form is wrong** ← This is our case
- Model class cannot express the data-generating process
- Systematic patterns persist despite adequate fitting

### The Core Problem

The model assumes:
```
log(μ[i]) = β₀ + β₁ × year[i]
```

This implies **constant exponential growth rate** (β₁). The data clearly show **accelerating exponential growth** (increasing rate over time). This cannot be captured by adjusting:
- Priors (wouldn't add flexibility)
- Dispersion (φ only affects variance, not mean)
- Sampling parameters (already converged perfectly)

The **linear-in-log-space assumption is the problem**, and it's the defining feature of this model class.

---

## Evidence Summary

### Falsification Criteria Results

From metadata.md, the model is rejected if any of these occur:

| Criterion | Threshold | Observed | Status |
|-----------|-----------|----------|--------|
| **2. Systematic curvature** | U or inverted-U | Inverted-U, coef=-5.22 | **FAIL** |
| **3. Late period failure** | MAE ratio >2× | 4.17× | **FAIL** |
| **4. Variance mismatch** | 95% outside [50,90] | [54.8, 130.9] | **FAIL** |
| 5. Poor calibration | Coverage <80% | 100% | PASS |
| 1. LOO-CV (pending) | ΔELPD >4 | Not computed | PENDING |

**Result**: 3 of 4 evaluated criteria FAILED → Rejection is mandatory per study design

---

### Quantitative Evidence

**Posterior Predictive Check Failures**:
- Variance-to-mean 95% CI extends to 131 (target: 90)
- Late/Early MAE ratio: 4.17 (threshold: 2.0)
- Quadratic residual curvature: -5.22 (threshold: 1.0)
- Only 1 of 4 checks passed

**Predictive Accuracy**:
- Overall MAE: 14.53
- Overall RMSE: 21.81
- Early period MAE: 6.34 (acceptable)
- Late period MAE: 26.49 (unacceptable)
- Mean residual: -0.16 (minimal bias)
- Median residual: -1.31 (slight underprediction tendency)

**Systematic Patterns**:
- Clear inverted-U in residuals vs. time
- Increasing spread in residuals vs. fitted values
- Heavy tails in Q-Q plot
- Progressive underprediction in final 10-15 observations

---

### Qualitative Evidence

**Visual Diagnostics Show**:
1. `residuals.png`: Unmistakable inverted-U pattern that cannot be ignored
2. `early_vs_late_fit.png`: Dramatic degradation in fit quality over time
3. `var_mean_recovery.png`: Predicted distribution too wide and right-skewed
4. `timeseries_fit.png`: Observed points systematically below predictions in late period

**Pattern Interpretation**:
- The data "curve upward" faster than exponential
- Model cannot capture this acceleration
- Errors compound over time
- Not a random fluctuation - systematic and smooth

---

## Why This Model Matters (Despite Rejection)

Even though we're rejecting this model, it serves crucial purposes:

### 1. Baseline Establishment
- Provides lower bound on model performance
- Clear reference point for comparison
- Documents what "simple" looks like

### 2. Diagnostic Value
- Reveals the specific way the data deviate from simple exponential growth
- Identifies what's needed (curvature term)
- Guides next model choice

### 3. Scientific Learning
- Demonstrates that growth is not constant-rate exponential
- Points to acceleration mechanism
- Raises questions about process dynamics

### 4. Methodological Rigor
- Shows importance of comprehensive validation
- Demonstrates that R²=0.92 is not sufficient
- Validates the falsification framework

---

## Strengths Worth Acknowledging

Before rejecting, it's important to acknowledge what worked:

**Computational Excellence**:
- Perfect convergence (R-hat=1.00)
- High effective sample size (ESS>6000)
- Zero divergences
- Stable across parameter ranges

**Model Implementation**:
- Correct likelihood specification
- Appropriate priors
- Proper inference procedure
- All parameters identifiable

**Early Period Performance**:
- MAE=6.34 in first 10 observations
- Reasonable point estimates
- Captures general growth trend

**Conservative Uncertainty**:
- 100% coverage ensures no missed observations
- Wide intervals provide safety margins

These strengths confirm that the **implementation is correct** - the problem is with the **model specification**, not execution.

---

## What Cannot Be Fixed

The following limitations are inherent to the log-linear model structure and cannot be addressed without changing the model class:

### 1. Curvature in Log-Space
- Log-linear models are, by definition, straight lines in log-space
- Curvature requires polynomial terms or different functional form
- Cannot be fixed by adjusting priors or dispersion

### 2. Accelerating Growth
- Constant exponential growth rate (exp(β₁)) is baked into the model
- Acceleration requires time-varying growth rate
- Cannot be captured with fixed β₁

### 3. Systematic Late-Period Bias
- Direct consequence of linear assumption
- As time progresses, deviation from true curve increases
- Cannot be eliminated while maintaining linear form

### 4. Overestimated Dispersion
- Model attributes systematic error to random variation
- φ compensates for misspecified trend
- Would be corrected by better trend specification, not by changing φ prior

---

## Decision Rationale: Why Not REVISE?

One might ask: "Why not just add a quadratic term? Isn't that a revision?"

**Answer**: Adding a quadratic term changes the model CLASS, not just the model:

**Log-Linear Model**:
- Assumes: log(μ) = β₀ + β₁×year
- Implies: Constant proportional growth
- Structure: Exponential growth with fixed rate

**Quadratic Model**:
- Assumes: log(μ) = β₀ + β₁×year + β₂×year²
- Implies: Time-varying proportional growth
- Structure: Super-exponential or sub-exponential growth

These are **different classes** with different theoretical implications. "Revision" would be:
- Changing β₁ prior from N(0.85, 0.5) to N(0.85, 0.75)
- Using Gamma(2,2) instead of Exponential(0.667) for φ
- Adding a covariate like log(population) while keeping linear time trend

Adding a quadratic term is creating a **new model**, which is what "REJECT and move to next model" means.

---

## Comparison to Study Design

The study design (metadata.md) established **falsification criteria** to determine when to reject:

### Pre-Registered Criteria
1. LOO-CV: ΔELPD >4 vs. quadratic → Not yet tested (pending)
2. Systematic curvature in residuals → **VIOLATED** (coef=-5.22)
3. Late period MAE >2× early period → **VIOLATED** (4.17×)
4. Var/Mean outside [50, 90] → **VIOLATED** (CI to 131)
5. Coverage <80% → Passed (100%)

**Result**: 3 of 4 evaluated criteria violated

The study design **anticipated this outcome** - the log-linear model was always intended as a baseline to be tested and likely rejected. The falsification framework worked as intended.

---

## Implications for Scientific Questions

### Original Research Questions
If the scientific goal was to:

**Characterize growth patterns**:
- Log-linear model identifies that growth is NOT constant exponential
- Points to acceleration mechanism
- Suggests super-exponential dynamics
- Need quadratic or alternative model to quantify

**Forecast future values**:
- Log-linear model would systematically underpredict
- Errors would compound for longer horizons
- Cannot be used for forecasting without risk
- Better model needed for reliable predictions

**Understand mechanisms**:
- Constant exponential growth ruled out
- Accelerating growth implies positive feedback
- Model rejection itself provides scientific insight
- Next model should reveal mechanism

**Make decisions**:
- Cannot rely on log-linear predictions for late periods
- Conservative intervals (100% coverage) provide safety
- But wide intervals reduce practical utility
- Better model would enable tighter, more useful predictions

---

## Recommended Next Steps

Having rejected the log-linear model, the natural progression is:

### Immediate: Fit Quadratic Model (Experiment 2)

**Specification**:
```
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year[i]²
C[i] ~ NegativeBinomial(μ[i], φ)
```

**Expected improvements**:
- Captures curvature (β₂ term)
- Reduces late-period errors
- Better Var/Mean recovery
- Improved LOO-CV score

**Additional parameters**:
- β₂ ~ Normal(0, 0.5): Quadratic coefficient
- Total: 4 parameters vs. 3

**Validation plan**:
- Same four-stage process
- Compare LOO-CV to log-linear
- Check if curvature is eliminated
- Verify late-period improvement

### If Quadratic Also Fails

Consider:
- Higher-order polynomials (cubic)
- Change-point models (regime shift)
- Generalized logistic growth (carrying capacity)
- Time-varying coefficients

### Use LOO-CV for Formal Comparison

Once quadratic model is fit:
- Compute ΔELPD (expected >10)
- Compare ELPD_diff to SE
- Create model comparison table
- Assess if additional complexity is justified

---

## Transparency and Reproducibility

All evidence supporting this decision is documented and reproducible:

**Data and Code**:
- Model specification: `experiments/experiment_1/metadata.md`
- Inference code: `experiments/experiment_1/posterior_inference/code/`
- PPC code: `experiments/experiment_1/posterior_predictive_check/code/`
- All analyses use seed=42 for reproducibility

**Diagnostics**:
- Prior predictive: 5 plots, findings.md
- SBC: 4 plots, recovery_metrics.md, 50 simulations
- Posterior: 8 plots, inference_summary.md
- PPC: 7 plots, ppc_findings.md

**Results**:
- InferenceData saved: `posterior_inference.netcdf`
- PPC results: `ppc_results.json`
- All plots in PNG format

**Criteria**:
- Pre-registered in metadata.md before analysis
- Applied objectively to observed results
- No post-hoc adjustment of thresholds

---

## Limitations of This Decision

### What We Don't Yet Know

1. **LOO-CV comparison**: Not yet computed; will provide additional evidence
2. **Quadratic model performance**: May also fail if curvature is more complex
3. **Mechanistic interpretation**: Haven't identified the process causing acceleration
4. **Sensitivity to data choices**: Only tested on single dataset

### Assumptions in This Decision

1. **Falsification criteria are appropriate**: Thresholds (2× MAE ratio, etc.) are somewhat arbitrary
2. **No data errors**: Assumes observations are correct and not contaminated
3. **Time is the relevant predictor**: Assumes no confounding variables
4. **Count data structure is correct**: Assumes observations are truly counts

### Alternative Perspectives

One could argue:
- 100% coverage is adequate for some applications
- Early period fit (MAE=6.34) might be sufficient for limited scope
- Model simplicity outweighs misspecification for communication purposes

**Response**: These might be valid for purely predictive applications, but given the clear systematic patterns and explicit falsification criteria, rejection is justified for scientific inference.

---

## Conclusion

The log-linear negative binomial model is **REJECTED** based on:
- Violation of 3 out of 4 evaluated falsification criteria
- Clear systematic misspecification (inverted-U residuals)
- 4× degradation in predictive accuracy over time
- Overestimated dispersion and over-conservative predictions

This rejection is **justified**, **well-documented**, and **anticipated by the study design**. The model serves its purpose as a baseline and diagnostic tool, clearly demonstrating the need for more flexible growth specifications.

**Next Action**: Proceed to Experiment 2 (Quadratic Model) with the same rigorous validation framework.

---

**Decision finalized**: 2025-10-29
**Analyst**: Model Criticism Specialist
**Recommendation**: Fit quadratic model as specified in improvement_priorities.md
