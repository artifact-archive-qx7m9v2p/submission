# Improvement Priorities for Fixed-Effect Model

**Model**: Experiment 1 - Fixed-Effect Normal Meta-Analysis
**Date**: 2025-10-28
**Status**: Model ACCEPTED with recommendations for enhancement

---

## Overview

The fixed-effect model is **technically sound and adequate** for its purpose. However, to strengthen the analysis and increase confidence in conclusions, we recommend the following enhancements. These are **improvements to the overall workflow**, not fixes to a broken model.

**Key insight**: This model is excellent at what it does (estimate θ under homogeneity), but the scientific question may require more than this simple model can provide.

---

## Priority 1: Compare to Random-Effects Model (ESSENTIAL)

### Why This Matters

**Problem**: The fixed-effect model **assumes** τ² = 0 (no between-study heterogeneity) without testing this assumption.

**Consequence**: If moderate heterogeneity exists (τ ≈ 5) but is masked by large measurement errors (mean σ = 12.5):
- Fixed-effect CI would be too narrow
- Predictions for new studies would be overconfident
- Ignoring substantively meaningful variation

**Evidence of concern**:
- Wide observed range: y ∈ [-3, 28] (31-unit span)
- Small sample: J = 8 provides low power to detect τ ≈ 5
- Large measurement errors: Mean σ = 12.5 comparable to posterior SD = 4.0

**Current status**: Cannot distinguish between:
- **Scenario A**: True homogeneity (τ = 0), wide range due to noise
- **Scenario B**: Moderate heterogeneity (τ ≈ 5), masked by large σ

### Implementation

**Model 2 specification**:
```
Likelihood: y_i ~ Normal(θ_i, σ_i²)
Hierarchy:  θ_i ~ Normal(μ, τ²)
Priors:     μ ~ Normal(0, 20²)
            τ ~ Half-Normal(0, 5²)
```

**Use non-centered parameterization** to avoid funnel pathology:
```python
theta_raw ~ Normal(0, 1)
theta = mu + tau * theta_raw
```

### Analysis Plan

1. **Fit Model 2** using same validation pipeline:
   - Prior predictive checks
   - SBC validation
   - MCMC with convergence diagnostics
   - Posterior predictive checks

2. **Parameter comparison**:
   - Does posterior for τ concentrate near zero?
   - Calculate I² = τ² / (τ² + σ̄²) to quantify heterogeneity
   - Compare μ (Model 2) to θ (Model 1)

3. **LOO-CV comparison**:
   - Compute LOO-ELPD for both models
   - Check difference and standard error
   - If ΔELPD > 2 SE, favor better model
   - Examine Pareto k diagnostics for influential observations

4. **Posterior predictive comparison**:
   - Which model has better calibration?
   - Which produces more realistic predictions for new studies?

### Expected Outcomes

**If τ ≈ 0** (homogeneity confirmed):
- Model 2 collapses to Model 1
- μ ≈ θ ≈ 7.4, similar CI widths
- LOO-CV shows equivalent performance (or slight advantage to Model 1 for parsimony)
- **Conclusion**: Fixed-effect assumption validated, Model 1 preferred

**If τ > 0 substantially** (heterogeneity detected):
- Model 2 has wider CI for μ
- LOO-CV favors Model 2
- Study-specific estimates θ_i show meaningful variation
- **Conclusion**: Random-effects model more appropriate, revise interpretation

**Most likely scenario** (based on I² = 0% from EDA): τ will be near zero, confirming fixed-effect assumption is reasonable.

### Deliverables

1. Full validation for Model 2 (matching Model 1 rigor)
2. Model comparison report with LOO-CV results
3. Decision: Which model is preferred?
4. Sensitivity: How robust are conclusions to model choice?

**Timeline**: High priority - should be completed before finalizing analysis

---

## Priority 2: Influential Observation Analysis (RECOMMENDED)

### Why This Matters

**Problem**: With only J = 8 observations, each study has substantial influence on the pooled estimate.

**Question**: Are conclusions sensitive to inclusion/exclusion of individual studies?

**Current analysis**: Leave-one-out from EDA shows modest influence (max change ±1.13), but this is frequentist analysis without full uncertainty quantification.

### Implementation

**Using LOO diagnostics** (already computed in posterior inference):

1. **Pareto k values**: Identify observations with k > 0.7
   - k > 0.7: Observation is influential, LOO-CV may be unreliable
   - k > 1.0: Very influential, must investigate

2. **Leave-one-out posterior**:
   - For each observation i, fit model on y₋ᵢ
   - Compare posterior means: θ̂ (full) vs θ̂₋ᵢ (leave-i-out)
   - Identify observations that substantially shift estimate

3. **Visual diagnostics**:
   - Plot posterior distributions with/without each observation
   - Show how credible intervals change
   - Highlight most influential studies

### Expected Findings

**Most likely**:
- Observation 1 (y=28, σ=15): Largest effect, but also largest σ, so downweighted
- Observation 5 (y=-1, σ=9): Negative effect, most precise, so upweighted
- No single observation should dominate (precision-weighting protects against this)

**If found**: Any observation with Pareto k > 0.7:
- Investigate why it's influential
- Check for data entry errors
- Consider sensitivity analysis excluding that observation
- Robust model (Student-t) might be appropriate

### Deliverables

1. LOO diagnostics table with Pareto k for each observation
2. Leave-one-out sensitivity plot showing θ̂₋ᵢ for each i
3. Identification of any influential observations
4. Recommendations if high-influence studies found

**Timeline**: Medium priority - can be done post-hoc if needed

---

## Priority 3: Enhanced Prior Sensitivity Analysis (OPTIONAL)

### Why This Matters

**Current status**: Tested σ ∈ {10, 20, 50}, found results robust.

**Extension**: More systematic exploration of prior space to document robustness comprehensively.

### Implementation

**1. Grid of prior specifications**:
```
Prior location: μ₀ ∈ {-5, 0, 5}
Prior scale: σ₀ ∈ {5, 10, 15, 20, 30, 50}
Total: 18 combinations
```

**2. For each prior**:
- Compute analytical posterior (fast)
- Extract posterior mean, SD, 95% CI
- Calculate P(θ > 0), P(θ > 5), P(θ > 10)

**3. Visualization**:
- Heatmap showing posterior mean as function of (μ₀, σ₀)
- Contour plot of P(θ > 0) across prior space
- Identify regions where conclusions change

**4. Reporting**:
- "Results are robust to priors with σ₀ ∈ [10, 50] regardless of location"
- "Conclusions sensitive only to very informative priors (σ₀ < 5)"
- Provides ammunition against reviewer concerns about prior choice

### Expected Findings

Given weak prior influence observed:
- Posterior mean varies by < 10% across reasonable prior range
- P(θ > 0) remains > 90% for all priors with σ₀ > 10
- Only very tight priors (σ₀ < 5) substantially affect inference

### Deliverables

1. Prior sensitivity grid analysis
2. Visualization of posterior quantities across prior space
3. Statement of robustness region
4. Documentation for supplementary materials

**Timeline**: Low priority - only if reviewers question prior choice

---

## Priority 4: Posterior Predictive for New Study (RECOMMENDED)

### Why This Matters

**Question**: If a 9th study were conducted, what effect would we predict?

**Current limitation**: Fixed-effect model gives predictive distribution:
```
y_new | θ ~ N(θ, σ_new²)
```

But this assumes new study estimates the same θ (fixed-effect philosophy).

### Implementation

**1. Fixed-effect prediction**:
```python
# For new study with σ_new
theta_samples = posterior['theta']
y_new = np.random.normal(theta_samples, sigma_new)
```

**2. Visualization**:
- Show predictive distribution for range of σ_new ∈ [9, 18]
- Compare to observed study effects
- Assess: Does predictive distribution cover observed range?

**3. Coverage analysis**:
- What proportion of observed studies fall in 95% predictive interval?
- If < 90%, suggests model underpredicts variation (heterogeneity?)

**4. If Random Effects fitted**:
- Compare fixed-effect prediction to random-effects prediction
- Random effects prediction: y_new ~ N(θ_new, σ²_new) where θ_new ~ N(μ, τ²)
- Random effects should have wider predictive distribution

### Expected Findings

**Fixed-effect prediction**:
- For new study with σ_new = 13 (mean):
  - 95% PI ≈ [-18, 33] (very wide)
  - Observed studies: 7/8 fall within this range (good)

**Comparison to random effects**:
- If τ ≈ 0: Predictions nearly identical
- If τ > 0: Random effects prediction wider and better calibrated

### Deliverables

1. Posterior predictive plot for new study
2. Coverage assessment: Do predictions encompass observed studies?
3. Comparison to random-effects predictions (if Model 2 fitted)
4. Guidance on expected effect in future studies

**Timeline**: Medium priority - useful for practical interpretation

---

## Priority 5: Meta-Regression Exploration (FUTURE)

### Why This Matters

**Current limitation**: Model cannot explain why effects vary (no covariates).

**Question**: Do study characteristics (dose, age, design, etc.) explain heterogeneity?

**Requirement**: Need covariate data, not available in current dataset.

### If Covariates Become Available

**Model specification**:
```
Fixed-effect meta-regression:
  y_i ~ Normal(β₀ + β₁X_i, σ_i²)

Random-effects meta-regression:
  y_i ~ Normal(θ_i, σ_i²)
  θ_i ~ Normal(β₀ + β₁X_i, τ²)
```

**Potential covariates**:
- Study year (temporal trends)
- Sample size (small-study effects)
- Study quality score
- Intervention characteristics (dose, duration)
- Population characteristics (age, baseline severity)

**Analysis**:
- Test whether β₁ ≠ 0 (covariate explains variation)
- Calculate R² for explained heterogeneity
- Assess whether τ² reduces with covariates included

### Deliverables (if implemented)

1. Covariate effect estimates with uncertainty
2. Forest plot stratified by covariate levels
3. Assessment of explained heterogeneity
4. Predictions conditional on covariate values

**Timeline**: Future work - requires additional data

---

## Priority 6: Bayesian Model Averaging (ADVANCED)

### Why This Matters

**Model uncertainty**: We're uncertain whether fixed-effect or random-effects is correct.

**Solution**: Instead of choosing one model, average across both weighted by predictive performance.

### Implementation

**Using LOO stacking weights**:

1. Fit both Model 1 (fixed) and Model 2 (random)
2. Compute LOO-CV for each
3. Calculate stacking weights: w₁, w₂ (sum to 1)
4. Average predictions: ŷ = w₁·ŷ₁ + w₂·ŷ₂

**ArviZ provides** `az.compare()` with stacking option.

### When to Use

**If LOO comparison shows**:
- Models perform similarly (ΔELPD < 2 SE)
- Both have reasonable weights (neither dominates)
- Uncertainty about which model is correct

**Benefit**: Accounts for model uncertainty in predictions.

**Example**: If w₁ = 0.6, w₂ = 0.4:
- 60% weight to fixed-effect
- 40% weight to random-effects
- Predictions blend both perspectives

### Deliverables

1. Model weights from LOO stacking
2. Model-averaged posterior for θ (or μ)
3. Model-averaged predictions
4. Comparison to single-model inference

**Timeline**: Advanced - only if model comparison is inconclusive

---

## Priority 7: Robustness to Normality (LOW PRIORITY)

### Why This Matters

**Current**: Model assumes y_i | θ ~ Normal(θ, σ_i²)

**Evidence**: Residuals pass Shapiro-Wilk test (p = 0.546), no outliers detected (all |z| < 2)

**Conclusion**: Normality appears adequate for these data.

### If Concerned About Outliers

**Model 3 (Robust Student-t)**:
```
Likelihood: y_i ~ StudentT(ν, θ, σ_i²)
Prior:      ν ~ Gamma(2, 0.1)  # Mean ≈ 20
```

**Expected**: ν > 20-30 would indicate normality is adequate.

**When to implement**:
- If outliers detected in future applications
- If reviewer raises concern about normality
- If observations with |z| > 3 appear

### Deliverables (if implemented)

1. Posterior for ν (degrees of freedom)
2. Comparison: Does θ change with heavy tails?
3. LOO-CV: Does robust model predict better?
4. Decision: Is robustness necessary?

**Timeline**: Low priority - only if normality assumption questioned

---

## Summary of Priorities

| Priority | Task | Importance | Effort | Status |
|----------|------|------------|--------|--------|
| **1** | Random-effects comparison | ESSENTIAL | High | NOT STARTED |
| **2** | Influential observations | RECOMMENDED | Low | Data available |
| **3** | Prior sensitivity grid | OPTIONAL | Medium | NOT NEEDED |
| **4** | Posterior predictive (new study) | RECOMMENDED | Low | NOT STARTED |
| **5** | Meta-regression | FUTURE | N/A | Requires covariates |
| **6** | Bayesian model averaging | ADVANCED | Medium | After Priority 1 |
| **7** | Robust model | LOW | Medium | NOT NEEDED |

---

## Recommended Immediate Actions

### Must Do (Before Finalizing Analysis)

1. **Fit Model 2 (random effects)** - Essential for validating homogeneity assumption
2. **LOO-CV comparison** - Determine which model predicts better
3. **Check Pareto k diagnostics** - Identify influential observations

### Should Do (For Complete Analysis)

4. **Posterior predictive for new study** - Practical interpretation
5. **Leave-one-out sensitivity** - Document robustness

### Nice to Have (For Publication)

6. **Enhanced prior sensitivity** - Address reviewer concerns
7. **Model averaging** - If model comparison inconclusive

### Not Needed Now

8. **Robust model** - No evidence of outliers
9. **Meta-regression** - No covariates available

---

## Implementation Guidance

### For Priority 1 (Random Effects)

**Steps**:
1. Create `/workspace/experiments/experiment_2/` directory structure
2. Copy validation pipeline from Experiment 1
3. Modify model specification for hierarchical structure
4. Run full validation (prior → SBC → fit → PPC)
5. Compare to Experiment 1 using LOO-CV
6. Write comparison report

**Expected timeline**: Same as Experiment 1 (~30-45 minutes)

**Outcome**: Clear decision on whether heterogeneity exists and which model is preferred

### For Priority 2 (Influential Observations)

**Steps**:
1. Load InferenceData from `/workspace/experiments/experiment_1/posterior_inference/diagnostics/`
2. Run `az.loo()` with return diagnostics
3. Extract Pareto k values
4. Plot LOO diagnostics using `az.plot_khat()`
5. If k > 0.7 for any observation, investigate further

**Expected timeline**: 10-15 minutes

**Code example**:
```python
import arviz as az
idata = az.from_netcdf('posterior_inference.netcdf')
loo = az.loo(idata, pointwise=True)
print(loo)
az.plot_khat(loo)
```

### For Priority 4 (Posterior Predictive New Study)

**Steps**:
1. Load posterior samples for θ
2. For range of σ_new ∈ [9, 18]:
   - Sample y_new ~ N(θ, σ²_new)
   - Compute 95% predictive interval
3. Plot predictive distributions
4. Check coverage: Do observed studies fall within predictions?

**Expected timeline**: 15-20 minutes

---

## What NOT to Change

The following aspects of Model 1 are **working correctly** and should not be modified:

1. **Prior specification**: θ ~ N(0, 20²) is appropriate and validated
2. **Likelihood**: Normal with known σ_i is standard and supported by data
3. **MCMC implementation**: Perfect convergence, validated against analytical solution
4. **Posterior predictive checks**: Comprehensive and all passed
5. **Model structure**: Fixed-effect is correct for its intended purpose

**Key point**: Model 1 is not broken. Improvements are about **extending** the analysis, not fixing errors.

---

## Final Recommendation

**Minimum viable analysis**:
- Model 1 (current) + Model 2 (random effects) + LOO comparison

**Complete analysis**:
- Above + influential observations + posterior predictive for new study

**Publication-ready analysis**:
- Above + prior sensitivity documentation + model averaging if needed

**Current status**: Model 1 is complete and validated. Next step is **Priority 1: Fit Model 2**.

---

**Prepared by**: Model Criticism Specialist
**Date**: 2025-10-28
**Next action**: Begin Experiment 2 (Random Effects Model)
