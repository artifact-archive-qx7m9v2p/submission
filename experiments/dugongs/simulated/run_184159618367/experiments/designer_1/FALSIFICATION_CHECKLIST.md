# Falsification Checklist for Designer 1 Models
## Quick Reference for Model Validation

**Purpose**: This checklist operationalizes the falsification criteria for each model. Use during model validation to determine if a model should be abandoned.

---

## Model 1: Asymptotic Exponential

### Convergence & Computation
- [ ] R-hat < 1.01 for all parameters (check after 4000 iterations)
- [ ] Effective sample size > 100 for all parameters
- [ ] No divergent transitions (target: 0, acceptable: <5)
- [ ] Chains mix well (visual inspection of trace plots)

**FAIL IF**: R-hat > 1.01 after good initialization OR ESS < 100

### Prior-Posterior Consistency
- [ ] α posterior mean within 3 SD of prior mean (2.55)
- [ ] β posterior mean within 3 SD of prior mean (0.9)
- [ ] γ posterior 95% CI overlaps prior support region
- [ ] No extreme parameter values (α < 3.0, β < 2.0, γ < 2.0)

**FAIL IF**: Any parameter shifts >3 SD from prior OR posterior mass at boundaries

### Residual Diagnostics
- [ ] No U-shape in residuals vs x
- [ ] No systematic pattern in residuals vs fitted values
- [ ] Residuals approximately normal (QQ plot)
- [ ] No heteroscedasticity (variance constant across x)
- [ ] Autocorrelation < 0.3 at replicated x-values

**FAIL IF**: Clear systematic patterns remain after accounting for saturation

### Predictive Performance
- [ ] LOO-R² > 0.80 (benchmark from EDA)
- [ ] 90-95% of observations in 95% posterior predictive intervals
- [ ] All Pareto-k < 0.7 (or at most 1 observation with k ∈ [0.7, 1.0])
- [ ] RMSE < 0.12
- [ ] Visual fit captures saturation pattern

**FAIL IF**: LOO-R² < 0.80 OR >10% observations outside 95% intervals OR multiple high Pareto-k

### Parameter Plausibility
- [ ] α posterior: 2.4 < α < 2.8 (should be near observed plateau)
- [ ] α posterior: <10% mass above max(Y) + 0.3 = 2.93
- [ ] β posterior: 0.5 < β < 1.5 (amplitude should match data range)
- [ ] γ posterior: 0.05 < γ < 1.0 (rate should be moderate)
- [ ] γ posterior: mode not at extreme boundaries

**FAIL IF**: α has >10% mass above 2.93 OR γ < 0.05 (too slow) OR γ > 1.0 (too fast)

### Derived Quantities
- [ ] Half-saturation x = ln(2)/γ in reasonable range [2, 15]
- [ ] Predicted Y at x=0: Y₀ = α - β should be in [1.3, 2.0]
- [ ] Initial slope: β*γ should be positive and < 0.3

**FAIL IF**: Half-saturation outside observed x range OR Y₀ is implausible

---

## Model 2: Piecewise Linear

### Convergence & Computation
- [ ] R-hat < 1.01 for all parameters (may need 5000 iterations)
- [ ] Effective sample size > 100 for all parameters (esp. τ)
- [ ] τ chain mixes well (no stickiness at boundaries)
- [ ] No divergent transitions

**FAIL IF**: R-hat > 1.01 OR ESS(τ) < 100 OR severe mixing problems

### Breakpoint Identification
- [ ] τ posterior 95% CI width < 10 x-units (preferably < 5)
- [ ] τ posterior mode in reasonable range [5, 20]
- [ ] τ posterior not bimodal (single clear breakpoint)
- [ ] τ 95% CI: [6, 13] overlaps with visual estimate ≈ 9.5

**FAIL IF**: τ 95% CI width > 10 units (uncertain breakpoint) OR mode at data boundary (τ<3 or τ>25)

### Regime Differentiation
- [ ] β₁ > β₂ (first slope larger than second)
- [ ] P(β₁ > β₂) > 0.95 (strong evidence for slope change)
- [ ] |β₁ - β₂| > 0.03 (meaningful difference, not trivial)
- [ ] β₁ posterior: 0.04 < β₁ < 0.12 (should match low-x slope)
- [ ] β₂ posterior: -0.02 < β₂ < 0.03 (should be near-zero)

**FAIL IF**: P(β₁ > β₂) < 0.95 OR slopes not meaningfully different

### Residual Diagnostics
- [ ] No discontinuity artifact at breakpoint (smooth residuals near τ)
- [ ] Observations near τ not systematically under/over-predicted
- [ ] No heteroscedasticity within regimes
- [ ] Residuals approximately normal

**FAIL IF**: Clear jump in residuals at τ OR systematic bias near breakpoint

### Predictive Performance
- [ ] LOO-R² > 0.85 (should match EDA benchmark of 0.90)
- [ ] 92-96% of observations in 95% posterior predictive intervals
- [ ] Pareto-k for observations near τ: ideally < 0.7, acceptable < 1.0
- [ ] RMSE < 0.10 (should be best of all models)

**FAIL IF**: LOO-R² < 0.85 (despite being best in EDA) OR poor coverage

### Parameter Plausibility
- [ ] β₀ posterior: 1.4 < β₀ < 1.9 (intercept reasonable)
- [ ] Y at τ: β₀ + β₁*τ should be in [2.3, 2.6] (transition at plateau)
- [ ] Predictions monotonically increasing (no segments crossing)

**FAIL IF**: Predictions become non-monotonic OR Y at τ is implausible

### Prior Sensitivity
- [ ] Refit with τ ~ Normal(9.5, 1.0) and τ ~ Normal(9.5, 2.5)
- [ ] τ posterior mode shifts < 2 x-units across prior choices
- [ ] Conclusions robust to prior specification

**FAIL IF**: τ posterior highly sensitive to prior (weak identification)

---

## Model 3: Power Law

### Convergence & Computation
- [ ] R-hat < 1.01 for all parameters (should be easy)
- [ ] Effective sample size > 200 for all parameters
- [ ] Fast convergence (<2000 iterations sufficient)
- [ ] No sampling pathologies

**FAIL IF**: Convergence problems (unlikely for linear model in log-log space)

### Log-Log Linearity
- [ ] Residuals on log-log scale show no curvature
- [ ] No U-shape or systematic pattern in log(Y) vs log(x) residuals
- [ ] Log-log correlation ≈ 0.90-0.95 (matching EDA)

**FAIL IF**: Clear curvature in log-log residuals (power law inadequate)

### High-x Fit Quality
- [ ] Observations with x > 20: mean residual near 0
- [ ] Model doesn't systematically under-predict for x > 20
- [ ] Predictions capture plateau region (not continuously increasing)

**FAIL IF**: Systematic under-prediction for x > 20 (saturation too slow)

### Exponent Plausibility
- [ ] β posterior: 0.05 < β < 0.50 (diminishing returns)
- [ ] β posterior mode near 0.12 (matching EDA)
- [ ] β 95% CI doesn't include 0 (evidence for relationship)
- [ ] β 95% CI doesn't include 1 (evidence for saturation)

**FAIL IF**: β < 0.05 (essentially flat) OR β > 0.5 (faster than expected)

### Predictive Performance
- [ ] LOO-R² > 0.75 (minimum acceptable)
- [ ] Ideally LOO-R² > 0.80 to be competitive
- [ ] 88-93% of observations in 95% posterior predictive intervals
- [ ] Pareto-k for x=31.5: monitor if > 0.7 (high leverage)

**FAIL IF**: LOO-R² < 0.75 OR >10 point gap from Models 1-2

### Lognormal Assumption
- [ ] Log(Y) residuals approximately normal
- [ ] No heavy tails or skewness in log-scale residuals
- [ ] Posterior predictive coverage adequate on original scale

**FAIL IF**: Lognormal likelihood clearly violated

### Elasticity Consistency
- [ ] Constant elasticity across x range (no regime differences)
- [ ] Segmented log-log regressions (x<10 vs x≥10) yield similar β
- [ ] Difference in β estimates < 0.10 across segments

**FAIL IF**: Different elasticity in low-x vs high-x (scale-free fails)

### Extrapolation Check
- [ ] Document that model predicts unbounded growth for x >> 31.5
- [ ] Compare to Models 1-2 predictions at x=50, 100
- [ ] Acknowledge this is a limitation (not a failure per se)

**NOTE**: Unbounded extrapolation is expected but limits model utility

---

## Global Checks (Apply to All Models)

### Cross-Model Validation
- [ ] At least one model passes all falsification criteria
- [ ] Models that pass have LOO-R² > 0.80
- [ ] Posterior predictive coverage > 90% for at least one model
- [ ] Parameters scientifically interpretable for passing models

**GLOBAL FAILURE IF**: All three models fail validation

### Influential Observations
- [ ] Count observations with Pareto-k > 0.7 across all models
- [ ] If same observation problematic across models: investigate
- [ ] x=31.5 may have elevated k (monitor but don't automatically fail)

**GLOBAL FAILURE IF**: >5 observations with k > 0.7 across any model

### Replicate Consistency
- [ ] For 6 replicated x-values: posterior predictive SD ≈ observed SD
- [ ] Calibration check: observed variance in 95% posterior predictive interval
- [ ] No systematic over/under-estimation of within-group variance

**GLOBAL FAILURE IF**: Consistent mis-calibration across all models

### Prior-Posterior Conflict
- [ ] Posterior distributions not fighting priors across all models
- [ ] At least one model shows reasonable prior-posterior evolution
- [ ] Data is informative (posteriors narrower than priors)

**GLOBAL FAILURE IF**: Data contradicts all reasonable priors

---

## Decision Tree

```
START
  │
  ├─ Fit all 3 models
  │
  ├─ Check convergence for each
  │   └─ FAIL → Fix or abandon model
  │
  ├─ Check falsification criteria for each
  │   ├─ Model 1: 0+ criteria met?
  │   ├─ Model 2: 0+ criteria met?
  │   └─ Model 3: 0+ criteria met?
  │
  ├─ Count models that PASS (criteria = 0)
  │
  ├─ IF 0 models pass:
  │   └─ GLOBAL FAILURE → Abandon approach, pivot to alternative models
  │
  ├─ IF 1 model passes:
  │   └─ Report that model as best (document why others failed)
  │
  ├─ IF 2+ models pass:
  │   ├─ Compare via LOO-CV
  │   ├─ IF |ΔELPD| > 2×SE: Select highest ELPD
  │   └─ IF |ΔELPD| < 2×SE: Select most interpretable/simple
  │
  └─ DONE
```

---

## Expected Falsification Outcomes

### Model 1 (Most Likely to Pass)
**Strengths**: Theoretically motivated, good fit, interpretable
**Potential Failures**:
- α-β correlation (identifiability)
- Extrapolation to x→0 uncertain
- May not capture sharp transition if it exists

**Expected**: PASS with strong performance

### Model 2 (May Pass or Fail on Breakpoint)
**Strengths**: Best empirical fit, captures regime shift
**Potential Failures**:
- Breakpoint uncertainty too large (→ FAIL)
- Mixing issues with τ parameter
- Overfitting to specific observations near breakpoint

**Expected**: PASS if breakpoint well-identified, FAIL if transition is smooth

### Model 3 (Most Likely to Fail)
**Strengths**: Simple, fast, well-identified
**Potential Failures**:
- Poor high-x fit (likely → FAIL)
- Log-log curvature (likely → FAIL)
- Doesn't saturate fast enough
- Exponent too low (β→0)

**Expected**: FAIL on high-x fit or log-log linearity

---

## Stress Test Expectations

### Test 1: High-x Extrapolation (x=50, 100)
- Model 1: Should plateau at α ≈ 2.55 ✓
- Model 2: Should remain flat ✓
- Model 3: Will predict Y > 3.0 (fails extrapolation) ✗

### Test 2: Low-x Interpolation (x=0.1, 0.5)
- Model 1: Predicts Y ≈ 1.55 (reasonable) ✓
- Model 2: May extrapolate to Y < 1.5 (uncertain) ?
- Model 3: Power law explodes as x→0 ✗

### Test 3: Replicate Consistency
- All models should pass if well-specified
- Check σ posterior vs empirical SD at replicates

### Test 4: Leave-One-Out by Regime
- Models 1-2 should be stable
- Model 3 may be sensitive to high-x removal

---

## Implementation Notes

### Automation
- Write script to check all criteria automatically
- Flag failures in red, passes in green
- Generate HTML report with checklist

### Logging
- Log all checks with timestamps
- Save diagnostic plots for failed criteria
- Document reasons for abandonment

### Iterative Refinement
- If model fails on minor issue (e.g., prior too tight), adjust and refit
- If model fails on major issue (e.g., systematic residuals), abandon
- Maximum 2 refinement iterations per model

---

**Designer**: Bayesian Model Designer 1
**Date**: 2025-10-27
**Purpose**: Operationalize falsification criteria for rigorous model validation
**Usage**: Check each box during validation; abandon model if any FAIL condition met
