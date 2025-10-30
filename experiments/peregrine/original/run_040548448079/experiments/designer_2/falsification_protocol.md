# Falsification Protocol: Smooth vs Discrete Models

## Core Philosophy

**Goal**: Determine whether the apparent structural break at observation 17 is:
- **A**: A smooth acceleration region (continuous process)
- **B**: A true discrete regime change (discontinuous process)

**Principle**: Design tests that can FALSIFY smooth models, not just confirm them.

---

## Test Battery

### Test 1: Leave-Future-Out Cross-Validation

**Hypothesis**: If discrete break is recent (obs 17/40), smooth models will fail to predict future.

**Procedure**:
1. Fit all models on observations 1-35
2. Predict observations 36-40
3. Compute prediction error (RMSE, MAE on log scale)

**Decision Rule**:
- If smooth models RMSE > 2× changepoint model: **FAIL** (discrete break real)
- If smooth models RMSE ≈ changepoint model: **PASS** (smooth sufficient)

**Why this works**: Smooth models extrapolate continuous trend. Discrete break model captures regime change explicitly.

---

### Test 2: Posterior Predictive Check for Discontinuity

**Hypothesis**: Smooth models should predict smooth first derivative.

**Procedure**:
1. For each model, compute posterior samples of log(μ_t)
2. Calculate first derivative: d(log μ)/dt
3. Plot derivative over time with 95% credible intervals

**Decision Rule**:
- If derivative has discontinuity at obs 17: **FAIL** (smooth model forcing discontinuity)
- If derivative continuous but very steep: **BORDERLINE** (check magnitude)
- If derivative smooth and moderate: **PASS**

**Thresholds**:
- GP lengthscale < 0.2: Trying to capture discontinuity
- Spline derivative jump > 1.0 log-units/year: Discontinuous
- Polynomial inflection point at obs 17 with extreme curvature: Artifact

---

### Test 3: Synthetic Data Test (Model Over-Flexibility)

**Hypothesis**: Good models should FAIL on wrong data-generating process.

**Procedure**:
1. Generate synthetic data with TRUE discrete break:
   - Pre-break: log(μ) = 4.0 + 0.3×year, φ = 1.5
   - Post-break: log(μ) = 4.0 + 1.2×year, φ = 1.5
   - Break at year = -0.21 (same as observed)
2. Fit smooth models (GP, Spline, Polynomial)
3. Check if they fit well (BAD) or poorly (GOOD)

**Decision Rule**:
- If smooth models LOO-ELPD on synthetic data within 10 of true model: **OVER-FLEXIBLE** (not trustworthy)
- If smooth models LOO-ELPD >> 20 worse than true model: **CORRECTLY FAILS** (trustworthy)

**Interpretation**: Models that fit everything fit nothing. Good models should fail on wrong DGP.

---

### Test 4: Residual Autocorrelation Structure

**Hypothesis**: All models must remove temporal dependency. Failure indicates misspecification.

**Procedure**:
1. Extract residuals: r_t = C_t - E[C_t | model]
2. Compute ACF of residuals (lags 1-10)
3. Check for significant autocorrelation

**Decision Rule**:
- If ACF(1) > 0.5: **FAIL** (autocorrelation not captured)
- If ACF(1) < 0.3: **PASS** (reasonable)
- If ACF shows periodic pattern: **FAIL** (missing structure)

**Why this matters**: Even if LOO-ELPD is good, residual structure means model is wrong.

---

### Test 5: Parameter Interpretation Test

**Hypothesis**: Parameters should be scientifically plausible, not artifacts.

**Procedure**:
For each model, check posterior distributions:

**GP Model**:
- Lengthscale ℓ: Should be 0.5-2.0 (similar to year range)
  - If ℓ < 0.2: **RED FLAG** (trying to capture discontinuity)
  - If ℓ > 3.0: **RED FLAG** (over-smoothing, missing pattern)
- GP amplitude σ_f: Should be 0.2-0.8
  - If σ_f < 0.1: **RED FLAG** (GP not needed)
  - If σ_f > 1.0: **RED FLAG** (extreme deviations)

**Spline Model**:
- Smoothing parameter σ_β: Should be 0.5-2.0
  - If σ_β > 5.0: **RED FLAG** (no smoothing, overfitting)
  - If σ_β < 0.1: **RED FLAG** (over-smoothed, linear)
- Knot sensitivity: Refit with K ∈ {8, 10, 12}
  - If LOO-ELPD changes > 10: **RED FLAG** (knot-dependent)

**Polynomial Model**:
- Cubic coefficient β_3: Should be small
  - If 95% CI excludes zero AND |β_3| > 0.5: **RED FLAG** (extreme curvature)
  - If 95% CI includes zero: **SIMPLIFY** (use quadratic)

**All Models**:
- Dispersion φ: Should be 1.0-3.0 (based on EDA α ≈ 0.61)
  - If φ < 0.5 or φ > 5.0: **RED FLAG** (dispersion misspecified)
- AR(1) parameter ρ: Should be 0.6-0.95 (based on ACF = 0.944)
  - If ρ < 0.5: **RED FLAG** (underfitting autocorrelation)
  - If ρ > 0.95: **RED FLAG** (near non-stationarity)

---

### Test 6: LOO-ELPD Comparison with Changepoint Models

**Hypothesis**: If smooth models substantially worse, discrete break is real.

**Procedure**:
1. Compute LOO-ELPD for all smooth models
2. Compare to changepoint models from Designer 1
3. Calculate ΔLOO = LOO_smooth - LOO_changepoint

**Decision Rule**:

| ΔLOO | SE(ΔLOO) | Interpretation | Action |
|------|----------|----------------|--------|
| > -10 | Any | Smooth ≈ Changepoint | Use simpler (smooth) |
| -10 to -20 | < 5 | Weak evidence for changepoint | Check residuals |
| -10 to -20 | > 5 | Borderline | Use changepoint |
| < -20 | Any | Strong evidence for changepoint | **ABANDON SMOOTH** |

**Critical Threshold**: ΔLOO < -20 is decisive evidence against smooth models.

---

### Test 7: Prior-Posterior Conflict

**Hypothesis**: Posterior fighting prior indicates model misspecification.

**Procedure**:
1. Compare prior and posterior distributions for key parameters
2. Check if posterior is pushed to prior boundaries
3. Assess prior-data conflict

**Red Flags**:
- GP lengthscale posterior mode at boundary (ℓ ≈ 0.1 or ℓ ≈ 3.0)
- Spline coefficients extreme values (|β_j| > 5)
- Polynomial coefficients require wider priors than expected
- Dispersion parameter radically different from EDA estimate

**Decision Rule**:
- If 2+ parameters show prior-posterior conflict: **FAIL** (model fighting data)

---

## Falsification Flowchart

```
START
  |
  v
Fit Polynomial (baseline)
  |
  v
LOO-ELPD acceptable? ──NO──> FAIL (too simple)
  |                            |
  YES                          v
  |                         Fit GP/Spline
  v                            |
Residual ACF < 0.3? ──NO──> Add AR(p) or FAIL
  |                            |
  YES                          v
  |                         LOO-ELPD improved?
  v                            |
Fit GP/Spline              YES / NO
  |                            |
  v                            v
Both converge? ──NO──> Use working model or FAIL
  |                            |
  YES                          v
  |                         Parameter checks
  v                            |
LOO: Smooth vs Changepoint     v
  |                      Red flags? ──YES──> FAIL
  |                            |
  v                            NO
ΔLOO < -20? ──YES──> ABANDON SMOOTH MODELS
  |                            |
  NO                           v
  |                      Posterior predictive checks
  v                            |
Residual checks                v
  |                      Discontinuity? ──YES──> FAIL
  |                            |
  v                            NO
First derivative               |
  |                            v
Continuous? ──NO──> FAIL    SUCCESS: Smooth model valid
  |
  YES
  |
  v
Leave-future-out CV
  |
  v
RMSE competitive? ──NO──> FAIL
  |
  YES
  |
  v
SUCCESS: Smooth model validated
  |
  v
Select best: Polynomial < Spline < GP
         (simplicity vs flexibility)
```

---

## Stopping Rules

### When to ABANDON Smooth Models Entirely

1. **All smooth models** LOO-ELPD > 20 worse than changepoint
2. **All smooth models** fail leave-future-out CV (RMSE > 2× changepoint)
3. **All smooth models** show derivative discontinuity at obs 17
4. **Synthetic data test** shows over-flexibility
5. **Domain knowledge** confirms discrete event at obs 17

### When to PROCEED with Smooth Models

1. **Any smooth model** LOO-ELPD within 10 of changepoint
2. **Residuals** show no discontinuity
3. **First derivative** continuous
4. **Leave-future-out CV** competitive
5. **Parameters** scientifically plausible

---

## Expected Outcome (Honest Prediction)

**Most Likely Result**: Smooth models FAIL

**Reasoning**:
1. EDA shows 730% growth rate increase at obs 17
2. Four independent tests confirm structural break
3. Discrete break model will likely dominate LOO-ELPD
4. GP lengthscale will shrink trying to capture discontinuity
5. Spline derivative will show jump at obs 17

**Probability Estimates**:
- Smooth models fail: 75%
- Smooth models borderline: 20%
- Smooth models succeed: 5%

**Why test anyway?**:
- Confirms discrete break is ROBUST finding
- Rules out smooth alternatives (falsification)
- Builds confidence in changepoint model
- Documents what DOESN'T work (scientific value)

---

## Documentation Requirements

### If Smooth Models FAIL

Document:
1. Which tests failed (with evidence)
2. Why failure indicates discrete break
3. Comparison to changepoint models
4. Recommendation: Use Designer 1's models
5. Scientific interpretation of discrete break

### If Smooth Models SUCCEED

Document:
1. Which tests passed (with evidence)
2. Why smooth acceleration is plausible
3. Best smooth model (Polynomial vs Spline vs GP)
4. Parameter interpretations
5. Scientific interpretation of smooth growth

### If Results MIXED

Document:
1. Conflicting evidence
2. Model uncertainty
3. Recommend ensemble or sensitivity analysis
4. Need for additional data or domain knowledge

---

## Final Checklist

Before concluding analysis, verify:

- [ ] All three smooth models fitted
- [ ] Convergence diagnostics (R-hat < 1.05)
- [ ] LOO-ELPD computed for all models
- [ ] Residual ACF checked
- [ ] First derivative computed
- [ ] Leave-future-out CV performed
- [ ] Posterior predictive checks completed
- [ ] Comparison to changepoint models
- [ ] Parameter plausibility assessed
- [ ] Synthetic data test run
- [ ] Falsification criteria evaluated
- [ ] Decision rule applied
- [ ] Results documented

---

**File**: `/workspace/experiments/designer_2/falsification_protocol.md`
**Purpose**: Systematic testing to determine if smooth models are adequate
**Key Insight**: Good science requires designing tests that can prove you WRONG
