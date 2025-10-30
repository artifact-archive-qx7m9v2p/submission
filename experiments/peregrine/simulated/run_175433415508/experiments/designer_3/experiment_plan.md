# Experiment Plan: Non-Linear and Hierarchical Bayesian Models
## Model Designer 3

**Date**: 2025-10-29
**Focus**: Quadratic growth, structural breaks, non-parametric flexibility
**Philosophy**: Complexity must justify itself or be abandoned

---

## Executive Summary

I propose **three Bayesian models** that handle non-linearity based on EDA findings showing:
- Quadratic fit (R²=0.964) marginally superior to exponential (R²=0.937)
- Possible structural break at year=-0.21 with 9.6-fold growth rate change
- High temporal autocorrelation (ACF=0.971)

**Critical stance**: The 2.7% R² improvement may be overfitting with n=40. **All models must beat linear baseline by ΔLOO > 4 SE or be rejected.**

---

## Proposed Models

### Model 1: Quadratic Negative Binomial with AR(1)
**ID**: D3M1_quadratic_nb_ar1
**File**: `/workspace/experiments/designer_3/model_1_quadratic.stan`

**Specification:**
```
log(μ_t) = β₀ + β₁·year_t + β₂·year_t² + ε_t
ε_t ~ AR(1), C_t ~ NegativeBinomial(μ_t, φ)
```

**Key Innovation**: Polynomial term captures acceleration in growth

**Priors:**
- β₂ ~ Normal(0, 0.3) - weakly informative, regularizes curvature
- ρ ~ Beta(20, 2) - strong prior for high autocorrelation from EDA

**Falsification Criteria:**
- REJECT if β₂ credible interval contains zero
- REJECT if ΔLOO < 4 SE vs linear baseline
- REJECT if β₁ and β₂ correlation > 0.9 (non-identifiable)

**Expected Runtime**: 2-4 minutes

---

### Model 2: Bayesian Changepoint with AR(1)
**ID**: D3M2_changepoint_nb_ar1
**File**: `/workspace/experiments/designer_3/model_2_changepoint.stan`

**Specification:**
```
log(μ_t) = β₀ + β₁·year_t + β₂·I(year_t > τ)·(year_t - τ) + ε_t
τ ~ Uniform(-1.5, 1.5), C_t ~ NegativeBinomial(μ_t, φ)
```

**Key Innovation**: Explicit regime switching at unknown location τ

**Priors:**
- τ ~ Uniform(-1.5, 1.5) - avoids edges, EDA suggested -0.21
- β₂ ~ Normal(0, 1.0) - slope change parameter

**Falsification Criteria:**
- REJECT if τ posterior is uniform (no information from data)
- REJECT if β₂ credible interval contains zero (no slope change)
- REJECT if ΔLOO < 6 SE vs linear (2 extra parameters need justification)

**Expected Runtime**: 4-8 minutes (changepoint estimation is hard)

---

### Model 3: Gaussian Process Negative Binomial
**ID**: D3M3_gp_nb
**File**: `/workspace/experiments/designer_3/model_3_gp.stan`

**Specification:**
```
log(μ_t) = β₀ + β₁·year_t + f(year_t)
f ~ GP(0, K), K = α²·exp(-dist²/(2ℓ²))
C_t ~ NegativeBinomial(μ_t, φ)
```

**Key Innovation**: Non-parametric mean function, no assumed form

**Priors:**
- ℓ ~ InvGamma(3, 3) - length scale, controls smoothness
- α ~ Normal(0, 1) - amplitude of deviations

**Falsification Criteria:**
- REJECT if ℓ → 0 (white noise, GP inappropriate)
- REJECT if ℓ → ∞ (reduces to linear, use simpler model)
- REJECT if computational failure (Cholesky issues)
- REJECT if ΔLOO < 10 SE vs best parametric model (high bar for n+2 parameters!)

**Expected Runtime**: 10-20 minutes (dense covariance matrix O(n³))

**Reality Check**: GP with n=40 is ambitious. May be computationally intractable.

---

## Sequential Testing Strategy

### Stage 1: Establish Linear Baseline
Fit linear Negative Binomial with AR(1):
```
log(μ_t) = β₀ + β₁·year_t + ε_t
```

Document baseline LOO, RMSE, posterior predictive checks.

### Stage 2: Test Quadratic (Model 1)
**Decision Rule**:
- If ΔLOO < 4 SE → **STOP**, use linear model
- If β₂ interval contains zero → **STOP**, use linear model
- If passes → Proceed to Stage 3

### Stage 3: Test Changepoint (Model 2)
**Only proceed if**:
- Model 1 failed OR
- Clear visual evidence of regime shift

**Decision Rule**:
- If τ posterior is uniform → **REJECT**
- If β₂ interval contains zero → **REJECT**
- Compare to both baseline and Model 1

### Stage 4: Test GP (Model 3)
**Only proceed if**:
- Models 1-2 show systematic posterior predictive failures
- This is a "stress test" for parametric forms

**Decision Rule**:
- If computational issues → **STOP**
- Must beat best parametric by ΔLOO > 10 SE

---

## Model Comparison Framework

### Primary Metric: LOO Cross-Validation
```
ΔELPD = ELPD_complex - ELPD_baseline
SE_diff = Standard error of difference
```

**Interpretation**:
- |ΔELPD| < 2 SE: Models equivalent, **prefer simpler**
- ΔELPD > 4 SE: Complex model justified (for 1 extra parameter)
- ΔELPD > 6 SE: Justified for 2 extra parameters
- ΔELPD > 10 SE: Justified for n extra parameters (GP case)

**Diagnostics**:
- Pareto k < 0.7 for >90% of observations
- If Pareto k poor → Use K-fold CV instead

### Secondary Metrics
1. **Posterior Predictive Checks**: p-values ∈ [0.05, 0.95]
2. **Coverage**: 90% intervals contain ~90% of data
3. **RMSE**: Out-of-sample prediction error
4. **Interpretability**: Can we explain results?

---

## Falsification Philosophy

### My Commitment

**I will abandon ALL proposed models if**:
1. Linear baseline performs equivalently (within 2 SE)
2. Prior-posterior conflict indicates model fights data
3. Computational intractability (signals misspecification)
4. Poor calibration (Pareto k > 0.7 for many points)

**Success is finding truth, not defending complexity.**

### Red Flags Triggering Full Reconsideration

**Model 1 (Quadratic)**:
- β₁, β₂ correlation > 0.9 → Non-identifiable
- Divergent transitions despite reparameterization
- Posterior predictive shows systematic bias

**Model 2 (Changepoint)**:
- τ posterior has >2 modes → Smooth curvature, not discrete break
- ESS for τ < 50 → Cannot estimate reliably
- Posterior P(τ at edge) > 0.3 → Spurious

**Model 3 (GP)**:
- Cholesky decomposition fails → Ill-conditioned covariance
- Length scale ℓ < 0.3 → Data too noisy
- >5% divergences → Model geometry pathological

### Decision Points for Strategy Pivots

**Checkpoint 1 (After Model 1)**:
- If β₂ ≈ 0 → **STOP all complex models**, report linear baseline
- If β₂ significant but poor LOO → **SKIP Model 3**, focus on Model 2

**Checkpoint 2 (After Model 2)**:
- If both polynomial and changepoint fail → **DO NOT fit Model 3**
- If changepoint succeeds but computational issues → **STOP**

**Checkpoint 3 (After Model 3)**:
- If GP fails → **Report best parametric model**
- If all fail → **Report linear baseline as success**

---

## Prior Predictive Checks (Pre-Flight)

### Model 1: Quadratic
Simulate 1000 datasets from prior:
```python
β₀ ~ Normal(4.7, 0.5)
β₁ ~ Normal(1.0, 0.5)
β₂ ~ Normal(0, 0.3)
φ ~ Gamma(2, 0.1)
```

**Check**:
- Do >95% of simulations have positive counts?
- Does curvature look plausible (not >10 inflections)?
- Do counts span reasonable range [10, 500]?

**Adjust priors if**: >5% of simulations are absurd

### Model 2: Changepoint
Simulate 1000 datasets from prior with τ ~ Uniform(-1.5, 1.5)

**Check**:
- Is τ location uniform across simulations? (should be)
- Are both regimes plausible?
- Any sharp discontinuities? (should be continuous at τ)

### Model 3: GP
Simulate 100 datasets from prior (expensive!)

**Check**:
- Are GP realizations smooth given RBF kernel?
- Does amplitude span reasonable range (log_μ varies by <5 units)?
- Any numerical issues (Cholesky failures)?

**If >10% fail**: Revise priors or abandon GP model

---

## Convergence Diagnostics

### Targets for All Models
- R-hat < 1.01 for all parameters
- ESS_bulk > 400 for key parameters (β, φ)
- ESS_tail > 400 for distributional checks
- Divergences < 1%

### Model-Specific Tolerances

**Model 1**: Standard HMC, expect clean diagnostics

**Model 2**:
- τ may have R-hat up to 1.02 (acceptable)
- ESS for τ can be lower (>100 acceptable)
- May need adapt_delta = 0.95

**Model 3**:
- Individual f_t can have lower ESS (>100 each)
- Hyperparameters (ℓ, α) must have ESS > 400
- May need adapt_delta = 0.99

### Remediation Strategy
1. Increase adapt_delta (0.9 → 0.95 → 0.99)
2. Try non-centered parameterization (already done)
3. Increase iterations (2000 → 4000)
4. If still failing → **ABANDON model** (it's telling us something!)

---

## Stress Tests

### Test 1: Extreme Extrapolation
Predict at year = ±3 (beyond data range)

**Pass**: Wide uncertainty, reasonable point estimates
**Fail**: Negative counts or >10^6 counts → Model fragile

### Test 2: Leave-Out Blocks
Remove first/last 10 observations, refit

**Pass**: Parameters stable within 2 SE
**Fail**: Wildly different parameters → Model unstable

### Test 3: Prior Sensitivity
Refit with priors 2× wider and 2× narrower

**Pass**: Posterior changes <20%
**Fail**: Posterior changes >50% → Prior-driven, not data-driven

### Test 4: Measurement Error
Add noise to 'year': year_noisy ~ year + Normal(0, 0.1)

**Pass**: Results qualitatively similar
**Fail**: Conclusions reverse → Model fragile

---

## Escape Routes

### If All Non-Linear Models Fail LOO
**Conclusion**: Linear model is sufficient

Report:
```
log(E[C_t]) = β₀ + β₁·year_t
C_t ~ NegativeBinomial(μ_t, φ)
ε_t ~ AR(1)
```

**This is success** - we learned the data are simpler than hypothesized.

### If AR(1) Causes Non-Convergence
**Alternative**: Drop temporal correlation, use robust SEs

### If Negative Binomial Insufficient
**Alternative**: Conway-Maxwell-Poisson for more flexible dispersion

### If All Bayesian Models Intractable
**Alternative**: Frequentist GLMM with glmmTMB

### If Discrete Mixture Suspected
**Alternative**: Finite mixture of Negative Binomials

---

## Implementation Timeline

### Day 1 Morning: Prior Predictive Checks
1. Write Stan programs (DONE)
2. Simulate from priors
3. Visualize prior predictive distributions
4. Adjust priors if needed

**Stop Criterion**: If priors generate absurd data >10% → Revise

### Day 1 Afternoon: Fit Baseline + Model 1
1. Fit linear baseline
2. Fit quadratic model
3. **Decision**: If Model 1 fails → STOP

### Day 2 Morning: Fit Model 2 (if justified)
1. Fit changepoint model
2. Compare to baseline and Model 1
3. **Decision**: If Model 2 fails → Likely skip Model 3

### Day 2 Afternoon: Fit Model 3 (if justified)
1. Only proceed if parametric models systematically fail
2. Fit GP model
3. **Expect**: Computational challenges

### Day 3: Model Comparison and Reporting
1. Compute LOO for all successful models
2. Posterior predictive checks
3. Stress tests
4. Write final report

---

## Communication to Synthesis Agent

### If Models Succeed
"Model [X] provides best balance of fit, parsimony, interpretability. ΔLOO = [value] ± [SE] vs linear baseline. Key finding: [quadratic/changepoint/smooth] structure explains data. Recommendation: Proceed with Model [X]."

### If Models Fail
"All proposed non-linear models failed to justify complexity vs linear baseline. ΔLOO values: M1=[value], M2=[value], M3=[value], all <4 SE. Falsification criteria met: [reasons]. **Recommendation: Use linear Negative Binomial with AR(1) as final model.** This is **SUCCESS**—we learned data are simpler than hypothesized."

### If Computational Failure
"Models 2-3 encountered computational intractability [specific issues]. This suggests data too noisy or n too small for complexity. Recommendation: Restrict to Model 1 or linear baseline. **This failure is INFORMATIVE**—reveals model misspecification."

---

## Key Files

**Model Specifications**:
- Main document: `/workspace/experiments/designer_3/proposed_models.md`
- This plan: `/workspace/experiments/designer_3/experiment_plan.md`

**Stan Programs**:
- Model 1 (Quadratic): `/workspace/experiments/designer_3/model_1_quadratic.stan`
- Model 2 (Changepoint): `/workspace/experiments/designer_3/model_2_changepoint.stan`
- Model 3 (GP): `/workspace/experiments/designer_3/model_3_gp.stan`

**To Be Created**:
- `/workspace/experiments/designer_3/prior_predictive_checks.py`
- `/workspace/experiments/designer_3/fit_models.py`
- `/workspace/experiments/designer_3/diagnostics.py`
- `/workspace/experiments/designer_3/results_summary.md`

---

## Final Statement

**My job is NOT to advocate for complex models.**

My job is to:
1. Propose plausible complex models based on EDA evidence
2. Specify them rigorously with falsification criteria
3. Test them honestly against simpler alternatives
4. Report failures as learning opportunities
5. Recommend the SIMPLEST model that genuinely explains the data

**If the linear model wins, I have succeeded.**

The EDA finding of "quadratic superiority" (R²=0.964 vs 0.937) is a 2.7% improvement that may not survive cross-validation with n=40. That's valuable to know.

**If a complex model wins, I must justify it ruthlessly.**

It must beat the baseline by a large margin, pass all diagnostics, survive stress tests. Extraordinary claims require extraordinary evidence.

**I am optimizing for scientific truth, not task completion.**

---

**Status**: Specifications complete, ready for implementation
**Designer**: Model Designer 3 (Non-linear & Hierarchical Specialist)
**Date**: 2025-10-29

