# Falsification Predictions

## Purpose

Before fitting any models, we document our predictions for what will happen. This prevents post-hoc rationalization and ensures honest scientific inference.

---

## Primary Prediction

**HYPOTHESIS**: Smooth nonlinear models will FAIL to match changepoint models.

**CONFIDENCE**: 75%

**REASONING**:
1. EDA shows 730% growth rate increase at observation 17
2. Four independent structural break tests converge on same location
3. Discrete regime change is most parsimonious explanation
4. Smooth acceleration would require extreme polynomial coefficients or very short GP lengthscale

---

## Model-Specific Predictions

### Polynomial Model

**Will likely show**:
- Systematic underfit in early observations (counts too high)
- Systematic underfit in late observations (counts too low)
- Residuals show clear pattern around observation 17
- Need cubic term (β₃ significantly different from zero)
- Posterior predictive checks fail around transition region

**Parameter predictions**:
- β₂ > 0 (positive acceleration)
- β₃ may be near zero (quadratic sufficient for smooth curve)
- ρ ≈ 0.85 (high autocorrelation to compensate for trend misspecification)
- φ ≈ 1.5-2.5 (from EDA)

**LOO-ELPD prediction**: -115 to -125 (significantly worse than changepoint)

**Falsification criterion**: ΔLOO < -20 compared to changepoint → FAIL

---

### Gaussian Process Model

**Will likely show**:
- Lengthscale shrinks to capture sharp transition (ℓ < 0.3)
- GP amplitude increases (σ_f > 0.7) to allow large deviations
- First derivative shows near-discontinuity at observation 17
- Computational issues (matrix conditioning problems)

**Parameter predictions**:
- ℓ ≈ 0.2-0.4 (shorter than prior mean, trying to be flexible)
- σ_f ≈ 0.5-0.8 (larger than prior mean)
- β₁ ≈ 0.4-0.6 (mean trend slope)
- ρ ≈ 0.7-0.8 (lower than polynomial, GP captures some autocorr)

**LOO-ELPD prediction**: -105 to -115 (better than polynomial, worse than changepoint)

**Falsification criterion**: ℓ < 0.2 AND ΔLOO < -15 → FAIL (trying to force smoothness on discontinuity)

**Diagnostic prediction**:
- 95% CI of derivative will widen dramatically at observation 17
- Posterior of lengthscale will push toward lower boundary
- May see divergences or high R-hat values

---

### B-Spline Model

**Will likely show**:
- Knot sensitivity (results change with K)
- Effective degrees of freedom concentrate around observation 17
- First derivative shows discontinuity despite smoothing penalty
- Smoothing parameter pushed to boundary (τ → ∞, no penalty)

**Parameter predictions**:
- τ > 2 (weak penalty, need flexibility)
- β coefficients show abrupt change between adjacent knots near obs 17
- ρ ≈ 0.75-0.85
- Knot at or near observation 17 will have large influence

**LOO-ELPD prediction**: -110 to -120 (between polynomial and GP)

**Falsification criterion**:
- First derivative discontinuous OR
- ΔLOO changes by >5 with different K OR
- ΔLOO < -15 compared to changepoint → FAIL

---

## Comparative Predictions

### LOO-ELPD Ranking (most likely)

1. Changepoint model (Designer 1): -95 to -105 [BEST]
2. Gaussian Process: -105 to -115 [~10 worse]
3. B-Spline: -110 to -120 [~15 worse]
4. Polynomial: -115 to -125 [~20 worse]

**Interpretation**: All smooth models fail threshold (ΔLOO < -20 for polynomial, < -10 for others)

### Alternative Scenarios

#### Scenario A: Smooth Models Competitive (25% probability)

**Evidence would include**:
- GP LOO-ELPD within 5 of changepoint
- Lengthscale ≈ 0.8-1.2 (reasonable)
- First derivative continuous
- Residual ACF < 0.3

**Conclusion**: Smooth acceleration real, apparent break is artifact of forcing exponential growth into linear-log space

**Action**: Use GP or Spline, document why changepoint seemed real in EDA

#### Scenario B: Mixed Results (15% probability)

**Evidence**:
- GP ≈ Changepoint, but Polynomial fails
- Or: GP works with very short lengthscale (ℓ ≈ 0.2)

**Conclusion**: Need local flexibility, but unclear if discrete break

**Action**:
- Sensitivity analysis
- Check domain knowledge for discrete event
- Consider ensemble or model averaging

#### Scenario C: All Models Fail (60% probability) [BASE CASE]

**Evidence**:
- All smooth models ΔLOO < -15
- GP lengthscale < 0.3
- Polynomial residuals show pattern
- Spline derivative discontinuous

**Conclusion**: Discrete regime change confirmed

**Action**: Use changepoint model from Designer 1

---

## Residual Autocorrelation Predictions

### Polynomial Model
- Raw residuals ACF(1) ≈ 0.6-0.7 (high)
- After AR(1): ACF(1) ≈ 0.2-0.4
- May need AR(2) for adequate fit

### GP Model
- Raw residuals ACF(1) ≈ 0.5-0.6
- After AR(1): ACF(1) ≈ 0.1-0.3
- GP captures some temporal structure, less reliance on AR

### Spline Model
- Raw residuals ACF(1) ≈ 0.5-0.6
- After AR(1): ACF(1) ≈ 0.2-0.3
- Similar to GP

**All models**: If ACF(1) > 0.5 after AR(1), indicates fundamental misspecification

---

## Leave-Future-Out Cross-Validation Predictions

**Setup**: Train on observations 1-35, test on 36-40

### Polynomial Model
- RMSE (log scale): 0.8-1.2
- Will underpredict if trend continues to accelerate
- Extrapolation relies on cubic term

### GP Model
- RMSE (log scale): 0.6-1.0
- Uncertainty will be very high (GP unsure beyond training range)
- May do better if acceleration continues smoothly

### Spline Model
- RMSE (log scale): 0.7-1.1
- Extrapolation difficult (splines not designed for it)
- Prediction will be linear beyond last knot

### Changepoint Model
- RMSE (log scale): 0.4-0.7
- If break already happened, just extends post-break regime
- Should be best

**Decision**: If smooth models RMSE > 2× changepoint RMSE, strong evidence for discrete break

---

## Parameter Boundary Issues (Prior-Posterior Conflict)

### Expected Conflicts

**GP Lengthscale**:
- Prior: InverseGamma(5, 5), mode ≈ 0.8
- Predicted posterior: mode ≈ 0.2-0.3, 95% CI pushed toward 0.1
- **Conflict**: Posterior fighting prior to be more flexible
- **Interpretation**: Model trying to capture discontinuity

**Spline Smoothing**:
- Prior: HalfNormal(0, 1), mode = 0
- Predicted posterior: mode ≈ 2-5, 95% CI wide
- **Conflict**: Posterior wants no smoothing
- **Interpretation**: Penalty inadequate for capturing sharp change

**Polynomial Cubic Term**:
- Prior: Normal(0, 0.3), 95% CI = [-0.6, 0.6]
- Predicted posterior: mode ≈ 0.1-0.3, may include zero
- **Less conflict**: Cubic helps but not essential

---

## Computational Issues Predictions

### GP Model

**Likely issues**:
1. **Matrix conditioning**: Correlation matrix near-singular if ℓ too small
2. **Sampling efficiency**: Low ESS for lengthscale (difficult geometry)
3. **Divergences**: 1-5% of iterations may diverge
4. **Time**: 15-25 minutes total (longest of three models)

**Mitigation**:
- Increase adapt_delta to 0.99
- Add jitter (1e-9) to diagonal
- Use non-centered parameterization (done)
- Consider Hilbert-space approximation if fails

### Spline Model

**Likely issues**:
1. **Coefficient correlation**: High correlation between adjacent β if too many knots
2. **Penalty degeneracy**: If τ → 0, matrix D'D singular
3. **Knot placement**: Results sensitive to knot locations

**Mitigation**:
- Use QR decomposition of basis matrix
- Add small ridge penalty (1e-6 × I)
- Try multiple K values

### Polynomial Model

**Expected**: Clean convergence, no issues (simplest model)

---

## Posterior Predictive Check Predictions

For all models, will generate C_rep and compare to observed C:

### Expected Patterns

**Polynomial**:
- Over-predicts early observations (1-16)
- Under-predicts late observations (18-40)
- Transition region (15-20) shows systematic bias

**GP**:
- Better match overall
- But 95% predictive intervals very wide around observation 17
- May show bimodal predictions (model uncertain about transition)

**Spline**:
- Similar to GP
- May show discontinuity in median prediction

**All smooth models**:
- Will struggle to capture sharp acceleration
- Posterior predictive p-values < 0.05 for "maximum derivative" test
- Changepoint model will pass this test

---

## Decision Tree Predictions

```
Fit all models
    │
    ▼
Convergence?
    │
    ├─> Polynomial: YES (99% confidence)
    ├─> GP: LIKELY (80% confidence, may need adapt_delta=0.99)
    └─> Spline: LIKELY (85% confidence)
    │
    ▼
LOO Comparison
    │
    └─> Changepoint > GP > Spline > Polynomial (80% confidence)
        ΔLOO ~ 10-25 between best smooth and changepoint
    │
    ▼
First Derivative Test
    │
    ├─> Polynomial: DISCONTINUOUS jump at obs 17 (90% confidence)
    ├─> GP: NEAR-DISCONTINUOUS (very steep slope) (70% confidence)
    └─> Spline: DISCONTINUOUS derivative (75% confidence)
    │
    ▼
Residual ACF
    │
    └─> All models: ACF(1) = 0.2-0.4 after AR(1) (acceptable but not great)
    │
    ▼
Leave-Future-Out CV
    │
    └─> Changepoint RMSE < Smooth RMSE (90% confidence)
    │
    ▼
FINAL DECISION (80% confidence)
    │
    └─> REJECT smooth models
        ACCEPT changepoint model (Designer 1)
        Discrete regime change at observation 17 confirmed
```

---

## Falsification Scorecard (Predicted)

| Test | Polynomial | GP | Spline | Changepoint |
|------|-----------|----|----|------------|
| Convergence | PASS | PASS | PASS | PASS |
| LOO-ELPD | FAIL (-20) | FAIL (-10) | FAIL (-15) | PASS |
| Residual ACF | PASS | PASS | PASS | PASS |
| 1st Derivative | FAIL | BORDERLINE | FAIL | PASS |
| Parameter Plausibility | PASS | BORDERLINE | BORDERLINE | PASS |
| LFO-CV | FAIL | FAIL | FAIL | PASS |
| **Overall** | **FAIL** | **FAIL** | **FAIL** | **PASS** |

**Predicted Winner**: Changepoint model (Designer 1)

---

## What Would Change Our Mind?

### Evidence that would SUPPORT smooth models:

1. **GP LOO-ELPD within 5 of changepoint** (currently predict -10)
2. **GP lengthscale ≈ 1.0** (currently predict 0.2-0.3)
3. **First derivative continuous for all models** (currently predict discontinuous)
4. **Synthetic data test shows GP rejects true discrete break** (currently predict GP fits everything)
5. **Domain knowledge confirms no discrete event** (need to check)

If 3+ of above true, would revise to "smooth models adequate"

### Evidence that would REJECT smooth models even more strongly:

1. **All smooth models ΔLOO < -30**
2. **GP fails to converge** (matrix singularity)
3. **All derivatives show discontinuity with >95% probability**
4. **Domain knowledge confirms discrete event** (policy change, etc.)

If 2+ of above true, would recommend not even reporting smooth models (changepoint overwhelmingly better)

---

## Meta-Prediction: Why Am I Probably Wrong?

**Overconfidence Check**: What if smooth models actually work?

**Possible errors in reasoning**:

1. **Forcing discrete thinking**: Maybe I'm pattern-matching "structural break" too quickly
2. **EDA misleading**: Four tests agree, but all assume discrete break framework
3. **Small sample illusion**: N=40 may make smooth curve look discrete
4. **Measurement aggregation**: True micro-level discrete events may aggregate smoothly
5. **Exponential growth illusion**: Rapid acceleration can look like regime shift

**Bayesian update**: If GP LOO is competitive AND lengthscale reasonable, will update to 50-50 smooth vs discrete

---

## Post-Analysis Checklist

After fitting models, verify predictions:

- [ ] LOO-ELPD rankings match predictions?
- [ ] GP lengthscale value as predicted?
- [ ] First derivative continuity as predicted?
- [ ] Residual ACF as predicted?
- [ ] Computational issues as predicted?
- [ ] Parameter conflicts as predicted?
- [ ] LFO-CV results as predicted?

If >5 predictions wrong, STOP and reconsider entire framework.

If predictions mostly correct, proceed with decision.

---

## Honest Assessment

**What I believe will happen**: Smooth models fail, discrete break confirmed.

**What I hope will happen**: Either outcome, but with decisive evidence (not borderline).

**What would be scientifically most interesting**: GP works well despite apparent break, forcing reconsideration of what "structural break" means in continuous time.

---

**File**: `/workspace/experiments/designer_2/predictions.md`
**Purpose**: Document falsifiable predictions before seeing results
**Date**: 2025-10-29 (before any models fitted)
**Will not be modified after model fitting to prevent rationalization**
