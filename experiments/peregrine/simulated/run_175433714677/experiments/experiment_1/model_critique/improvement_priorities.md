# Model Improvement Priorities: Experiment 1

**Current Model**: Log-Linear Negative Binomial (REJECTED)
**Date**: 2025-10-29
**Status**: Prioritized refinement plan for next model

---

## Executive Summary

Since the log-linear model is REJECTED as fundamentally misspecified, this document outlines the **next model class** to explore rather than incremental refinements to the current model.

The data exhibit clear evidence of accelerating exponential growth that cannot be captured by the current linear-in-log-space specification. The recommended next step is to fit a **quadratic model** that adds a year² term to capture the observed curvature.

---

## Priority 1: FIT QUADRATIC MODEL (HIGH PRIORITY)

### Rationale

The posterior predictive check revealed:
- **Strong inverted-U curvature** in residuals (quadratic coefficient = -5.22)
- **Systematic underprediction** in late period
- **4.17× increase** in errors from early to late period

These symptoms all point to a **missing quadratic term** in the log-linear predictor. Adding year² allows the model to capture acceleration or deceleration in the exponential growth rate.

### Specification

**Model**: Quadratic Negative Binomial Regression

```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year[i]²
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)     # Intercept (same as before)
β₁ ~ Normal(0.85, 0.5)    # Linear term (same as before)
β₂ ~ Normal(0, 0.5)       # Quadratic term (NEW, centered at 0)
φ  ~ Exponential(0.667)   # Dispersion (same as before)
```

### Prior Justification for β₂

**β₂ ~ Normal(0, 0.5)** encodes:
- **Mean = 0**: No prior belief in curvature direction (let data decide)
- **SD = 0.5**: Allows substantial curvature but isn't dogmatic
- **95% interval**: [-1.0, 1.0] in log-space

**Interpretation of β₂**:
- β₂ >0: Accelerating exponential growth (super-exponential)
- β₂ <0: Decelerating exponential growth (sub-exponential)
- β₂ =0: Reduces to log-linear model (null hypothesis)

**Expected value**: Based on residual analysis, we expect β₂ >0 (positive curvature in log-space creates inverted-U pattern in residuals when fitted with linear model).

### Expected Improvements

| Diagnostic | Current (Log-Linear) | Expected (Quadratic) | Target |
|------------|---------------------|---------------------|--------|
| Residual curvature | -5.22 | <1.0 | <1.0 |
| Late/Early MAE ratio | 4.17 | <2.0 | <2.0 |
| Var/Mean recovery | CI to 131 | CI to ~90 | [50, 90] |
| Late period MAE | 26.49 | <15 | Reduce by >40% |

### Implementation Steps

1. **Create Experiment 2 directory structure**:
   ```
   experiments/experiment_2/
   ├── metadata.md
   ├── prior_predictive_check/
   ├── simulation_based_validation/
   ├── posterior_inference/
   ├── posterior_predictive_check/
   └── model_critique/
   ```

2. **Specify model in metadata.md**:
   - Include quadratic term in likelihood
   - Document β₂ prior choice
   - Update falsification criteria if needed

3. **Run four-stage validation pipeline**:
   - Prior predictive check (verify β₂ prior is reasonable)
   - SBC (verify parameter recovery with quadratic term)
   - Posterior inference (fit to real data)
   - Posterior predictive check (verify improvements)

4. **Compare to log-linear model**:
   - Compute LOO-CV for both models
   - Calculate ΔELPD (expected >10)
   - Assess if additional parameter is justified

### Success Criteria

The quadratic model should be ACCEPTED if:
- [x] Residual curvature |coef| <1.0
- [x] Late/Early MAE ratio <2.0
- [x] Var/Mean 95% CI mostly within [50, 90]
- [x] Coverage >80% (should maintain near 90%)
- [x] LOO-CV shows ΔELPD >4 compared to log-linear

### Potential Issues to Monitor

1. **Overfitting**: With 40 observations and 4 parameters, still safe (10:1 ratio)
2. **Collinearity**: year and year² are correlated, but manageable with weak priors
3. **Extrapolation**: Quadratic can diverge rapidly outside data range (use caution)
4. **Interpretation**: β₂ is harder to interpret than β₁ alone

---

## Priority 2: MODEL COMPARISON VIA LOO-CV (HIGH PRIORITY)

### Rationale

Even though PPC clearly shows log-linear model inadequacy, formal model comparison via LOO-CV is essential for:
- Quantifying improvement magnitude
- Justifying additional parameter
- Comparing multiple candidate models
- Publication and reproducibility

### Implementation

**For Log-Linear Model (already fitted)**:
```python
import arviz as az
idata = az.from_netcdf("posterior_inference.netcdf")
loo_linear = az.loo(idata, pointwise=True)
```

**For Quadratic Model (after fitting)**:
```python
loo_quadratic = az.loo(idata_quad, pointwise=True)
```

**Comparison**:
```python
comparison = az.compare({"linear": idata, "quadratic": idata_quad})
```

### Expected Results

| Model | ELPD | ΔELPD | SE(ΔELPD) | Weight |
|-------|------|-------|-----------|--------|
| Quadratic | -140 | 0.0 | 0.0 | ~1.0 |
| Log-Linear | -155 | 15 | 5 | ~0.0 |

Expected ΔELPD ≈ 10-20 (strong evidence for quadratic model)

### Interpretation

- **ΔELPD >10**: Decisive evidence for quadratic model
- **ΔELPD = 4-10**: Strong evidence for quadratic model
- **ΔELPD <4**: Weak evidence; may not justify additional parameter

If ΔELPD <4, it would suggest the curvature, while statistically significant, may not improve out-of-sample prediction substantially.

### Influential Points

Use `az.plot_khat()` to identify observations with high Pareto k values:
```python
az.plot_khat(loo_quadratic, show_bins=True)
```

- k <0.5: Good
- k 0.5-0.7: OK
- k 0.7-1.0: Concerning
- k >1.0: Bad (LOO unreliable)

If many k >0.7, consider:
- Different likelihood (e.g., Student-t for robustness)
- Outlier investigation
- Model respecification

---

## Priority 3: INVESTIGATE TIME-VARYING DISPERSION (MEDIUM PRIORITY)

### Rationale

The PPC showed:
- Increasing residual spread over time (`residuals.png`, top-right panel)
- Var/Mean overestimation (predicted 84.5 vs. observed 68.7)
- Possible heteroscedasticity

While the quadratic term will address much of this (by improving the mean function), residual variance may still vary systematically with time or mean.

### Specification Option 1: Mean-Dependent Dispersion

```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year[i]²
log(φ[i]) = α₀ + α₁ × log(μ[i])
```

This allows dispersion to scale with the mean, which is common in count data.

### Specification Option 2: Time-Varying Dispersion

```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year[i]²
log(φ[i]) = α₀ + α₁ × year[i]
```

This allows dispersion to change linearly over time.

### When to Consider

**Fit time-varying dispersion if**:
- Quadratic model still shows heteroscedasticity in residuals vs. fitted plot
- Var/Mean recovery still poor after quadratic term added
- LOO-CV shows improvement with time-varying φ

**Skip if**:
- Quadratic model resolves residual patterns
- Constant φ is adequate
- Additional complexity not justified by LOO-CV

### Trade-offs

**Pros**:
- More flexible variance structure
- May improve coverage calibration
- Can handle heteroscedasticity

**Cons**:
- 2 additional parameters (6 total)
- Harder to interpret
- Risk of overfitting with n=40
- Extrapolation becomes more uncertain

**Recommendation**: Only pursue if quadratic model is inadequate.

---

## Priority 4: ALTERNATIVE GROWTH FUNCTIONS (LOW-MEDIUM PRIORITY)

### Rationale

If the quadratic model also fails (unlikely but possible), consider alternative functional forms that allow for acceleration.

### Option 1: Cubic Polynomial

```
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year[i]² + β₃ × year[i]³
```

**When**: If quadratic residuals still show curvature
**Pros**: Maximum flexibility within polynomial family
**Cons**: 5 parameters, risk of overfitting, difficult extrapolation

### Option 2: Generalized Logistic Growth

```
μ[i] = K / (1 + exp(-r × (year[i] - t₀)))^(1/ν)
```

**When**: If data approach an asymptote (carrying capacity)
**Pros**: Mechanistically interpretable, bounded
**Cons**: Nonlinear in parameters, harder to fit, more assumptions

### Option 3: Piecewise Linear with Change-Point

```
log(μ[i]) = β₀ + β₁ × year[i] + (year[i] > τ) × β₂ × (year[i] - τ)
```

**When**: If there's a discrete regime shift
**Pros**: Allows for step-change in growth rate
**Cons**: Change-point τ is difficult to infer, may not be smooth

### Option 4: Gaussian Process

```
log(μ[i]) = GP(year[i])
```

**When**: If growth pattern is very complex and irregular
**Pros**: Maximum flexibility, no parametric assumptions
**Cons**: Computational expense, harder interpretation, extrapolation issues

### Decision Rule

**Try cubic** if:
- Quadratic residuals show systematic curvature (|quad_coef| >1)
- Late/Early MAE ratio still >2
- LOO-CV suggests improvement

**Try logistic** if:
- Data show evidence of saturation (leveling off)
- Counts appear to approach maximum
- Scientific theory suggests carrying capacity

**Try change-point** if:
- Clear discontinuity in growth rate visible
- External event or intervention occurred
- Sudden shift in residual patterns

**Try GP** if:
- Growth pattern is highly irregular
- No parametric form seems adequate
- Interpretability is not critical

**Recommendation**: Start with cubic before considering non-polynomial forms.

---

## Priority 5: ROBUSTNESS CHECKS (LOW PRIORITY)

### Outlier Robustness

If LOO shows high Pareto k values (>0.7), consider:

**Student-t Likelihood**:
```
C[i] ~ StudentT(df=ν, μ[i], σ[i])
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year[i]²
ν ~ Gamma(2, 0.1)  # Degrees of freedom
```

**Pros**: Heavy tails, robust to outliers
**Cons**: Different likelihood family, less standard

### Prior Sensitivity

Test posterior robustness to prior choices:

**Wider priors**:
- β₁ ~ Normal(0.85, 1.0) instead of N(0.85, 0.5)
- β₂ ~ Normal(0, 1.0) instead of N(0, 0.5)

**Flatter priors**:
- β₁ ~ Normal(0, 2) (less informative)
- β₂ ~ Normal(0, 2)

**Compare**: If posteriors are similar across prior specifications, conclusions are robust.

### Leave-One-Out Analysis

For each observation i:
- Fit model without observation i
- Predict observation i
- Compare prediction to actual

Identifies influential observations and assesses prediction stability.

---

## Non-Priorities (What NOT to Do)

### 1. Don't Adjust Log-Linear Model Priors

**Why**: The problem is structural (missing curvature), not prior specification
**Evidence**: SBC showed good parameter recovery under model assumptions
**Conclusion**: Changing priors won't fix misspecification

### 2. Don't Add Non-Time Covariates (Yet)

**Why**: Time is clearly the primary driver; adding covariates is premature
**When**: After establishing adequate time model, check for additional predictors
**Conclusion**: Fix temporal trend first

### 3. Don't Use Data Transformations

**Why**: Log-transform is already in the model (log-link)
**Temptation**: Transform count data to make it more linear
**Problem**: Breaks count structure, harder interpretation, not principled
**Conclusion**: Keep count data as counts

### 4. Don't Ignore the Rejection

**Temptation**: "MAE=14 is acceptable for my application"
**Problem**: Systematic bias means predictions will be wrong in specific directions
**Risk**: Late-period forecasts will be systematically low
**Conclusion**: Accept rejection and fit better model

---

## Implementation Timeline

### Immediate (Next Step)

**Week 1: Fit Quadratic Model**
- Create Experiment 2 structure
- Run prior predictive check
- Run SBC (if time permits, or skip if urgent)
- Fit to real data
- Run posterior predictive check
- Compare to log-linear via LOO-CV

### Short-Term (If Quadratic Passes)

**Week 2: Model Comparison and Interpretation**
- Formal LOO-CV comparison table
- Interpret β₂ coefficient
- Generate predictions and intervals
- Create visualizations for reporting
- Write up results

### Short-Term (If Quadratic Fails)

**Week 2: Investigate Alternatives**
- Assess residual patterns from quadratic model
- Determine if cubic or alternative form needed
- Specify and fit next candidate model
- Repeat validation pipeline

### Medium-Term (If Time Permits)

**Week 3-4: Robustness and Extensions**
- Time-varying dispersion if needed
- Prior sensitivity analysis
- Influential point analysis
- Additional candidate models if justified

---

## Success Metrics

### For Quadratic Model

**Must achieve**:
- Residual curvature |coef| <1.0
- Late/Early MAE ratio <2.0
- Coverage >80%

**Should achieve**:
- Var/Mean recovery within [50, 90]
- LOO-CV ΔELPD >4 vs log-linear
- RMSE <15 (vs. 21.8 currently)

**Nice to have**:
- Coverage near 90% (not over-conservative)
- Uniform residual spread over time
- All Pareto k <0.7

### For Overall Model Selection

**Final model should**:
- Pass all 4 PPC checks
- Show no systematic residual patterns
- Achieve similar accuracy across time periods
- Have LOO-CV support over simpler alternatives
- Be interpretable and scientifically meaningful

---

## Risk Assessment

### High Risk Issues

**Quadratic model also fails**:
- Probability: 20%
- Impact: Requires cubic or alternative form
- Mitigation: Have alternative specifications ready

**Overfitting with small n**:
- Probability: 10%
- Impact: Poor out-of-sample prediction
- Mitigation: Use LOO-CV to detect, use informative priors

### Medium Risk Issues

**Collinearity between year and year²**:
- Probability: 30%
- Impact: Wide posterior SDs, high correlation
- Mitigation: Orthogonal polynomials, or accept correlation

**Extrapolation issues**:
- Probability: 50%
- Impact: Predictions unreliable beyond data range
- Mitigation: Don't extrapolate far, or use bounded growth models

### Low Risk Issues

**Prior sensitivity**:
- Probability: 10%
- Impact: Results depend on prior choice
- Mitigation: Test multiple priors, report sensitivity

---

## Consultation Points

**Consult domain expert**:
- Is accelerating growth scientifically plausible?
- What mechanisms could cause it?
- Are there external events that might explain change-points?

**Consult statistician**:
- Is 4 parameters too many for n=40?
- Should we use orthogonal polynomials?
- Are there better growth models for this domain?

**Consult stakeholder**:
- What level of accuracy is needed?
- Is extrapolation required?
- How will predictions be used?

---

## Documentation Requirements

For each new model fitted:

1. **metadata.md**: Specification, priors, falsification criteria
2. **Four validation stages**: Prior predictive, SBC, posterior, PPC
3. **Model comparison**: LOO-CV table with all candidates
4. **Interpretation**: What does β₂ mean scientifically?
5. **Limitations**: Where does model still fall short?

---

## Conclusion

The clear path forward is to:

1. **Fit quadratic model** (Priority 1, HIGH)
2. **Compare via LOO-CV** (Priority 2, HIGH)
3. **Consider time-varying dispersion** only if quadratic inadequate (Priority 3, MEDIUM)
4. **Try alternative forms** only if quadratic fails (Priority 4, LOW-MEDIUM)

The diagnostic evidence (inverted-U residuals, 4× MAE ratio, systematic curvature) all point strongly to a missing quadratic term. This should be the first and most likely successful modification.

**Recommendation**: Proceed immediately to Experiment 2 (Quadratic Model).

---

**Priorities document completed**: 2025-10-29
**Analyst**: Model Criticism Specialist
**Next action**: Create Experiment 2 directory and metadata.md
