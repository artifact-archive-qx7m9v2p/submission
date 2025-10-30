# Experiment Plan: Changepoint and Regime-Switching Models
## Designer 1 - Structural Break Focus

**Date**: 2025-10-29
**Designer**: Model Designer 1 (Changepoint/Regime-Switching Specialist)
**Data**: Time series counts (N=40) with suspected structural break at t=17

---

## Problem Formulation

### Core Question
**Is the observed 730% growth rate increase a true discrete regime change, or an artifact of:**
- Post-hoc selection bias (we looked for breaks and found one)
- Smooth acceleration misinterpreted as discrete shift
- Autocorrelation creating spurious structure
- Multiple smaller breaks aggregated into apparent single break
- Measurement/definition changes at observation 17

### Competing Hypotheses

#### H1: True Discrete Regime Change at t≈17
**Predicts**:
- Single changepoint model fits well
- Changepoint location concentrated around t=17
- Growth rates clearly differ pre/post break
- Model generalizes to held-out data

**Would be falsified by**:
- Unknown changepoint model finding τ far from 17
- Smooth GP model fitting substantially better
- Break parameter β₂ not significantly different from zero
- Poor out-of-sample predictions despite good in-sample fit

#### H2: Smooth Exponential Acceleration (No True Break)
**Predicts**:
- Quadratic/cubic polynomial fits as well as changepoint
- Gaussian Process model superior to discrete changepoint
- Unknown changepoint posterior is diffuse/unstable
- Break appears at different locations under cross-validation

**Would be falsified by**:
- Changepoint model strongly preferred (ΔLOO > 10)
- Sharp discontinuity in growth rates at single time point
- Posterior predictive checks showing discrete jump, not smooth curve

#### H3: Multiple Smaller Regime Shifts (Not Single Break)
**Predicts**:
- Multiple changepoint model preferred over single changepoint
- Two or more distinct break locations with similar magnitudes
- Single changepoint model shows residual temporal structure
- Time-varying dispersion suggests regime heterogeneity

**Would be falsified by**:
- Single changepoint sufficient (k=1 strongly preferred)
- Additional changepoints not improving fit
- Second/third breaks have negligible effect sizes

### Why This Matters
**Scientific**: Understanding whether real-world processes exhibit discrete regime changes vs. continuous evolution has implications for:
- Prediction and early warning systems
- Causal mechanism identification
- Policy intervention timing

**Statistical**: Distinguishing discrete breaks from smooth transitions is:
- Challenging with small samples (N=40)
- Sensitive to model assumptions
- Often confounded with autocorrelation

**This experiment explicitly tests whether our apparent pattern is real or illusory.**

---

## Model Classes to Explore

### Model Class 1: Fixed Changepoint Model
**Core Idea**: Assume EDA is correct; τ=17 is the true break point

**Mathematical Form**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = η_t
η_t ~ Normal(β₀ + β₁·year_t + β₂·I(t>17)·(year_t - year₁₇) + ρ·η_{t-1}, σ_η)
```

**Variants**:
1. **1a**: Observation-level random effects (no AR)
2. **1b**: AR(1) latent process (structured autocorrelation)
3. **1c**: AR(1) + time-varying dispersion α(t)

**Priors**:
- β₀ ~ Normal(4.3, 0.5) [EDA: log-count at center]
- β₁ ~ Normal(0.35, 0.3) [Pre-break slope]
- β₂ ~ Normal(1.0, 0.5) [Break magnitude]
- α ~ Gamma(2, 1) [Dispersion]
- ρ ~ Beta(8, 2) [Strong positive autocorrelation expected]

**Falsification Criteria** - I will ABANDON this model if:
1. β₂ credible interval includes zero (break not significant)
2. Posterior predictive ACF(1) > 0.3 (autocorrelation not captured)
3. Divergent transitions > 1% after tuning (computational pathology)
4. LOO Pareto k > 0.7 for >5% of observations
5. Out-of-sample RMSE > 1.5x in-sample on log scale
6. Dispersion parameter α posterior <0.1 or >5.0 (distribution inadequate)
7. Prior-posterior shift >2 SD for any parameter (model-data conflict)

**Why It Might Fail**:
- τ=17 may be wrong location (post-hoc selection bias)
- Single break may be oversimplification (multiple regimes exist)
- AR(1) may be inadequate for ρ≈0.944 (near unit root)
- Fixed dispersion may miss heteroscedasticity

**Implementation**: Stan (preferred for speed and stability)
**Expected Runtime**: 2-5 minutes per variant
**Priority**: HIGH (start here)

---

### Model Class 2: Unknown Changepoint Model
**Core Idea**: Challenge EDA; let data determine break location

**Mathematical Form**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β₀ + β₁·year_t + β₂·I(t>τ)·(year_t - year_τ) + AR(1)
τ ~ DiscreteUniform(5, 35)
```

**Key Difference**: τ is now a parameter, not fixed at 17

**Variants**:
1. **2a**: Uniform prior on τ (completely agnostic)
2. **2b**: Concentrated prior around τ≈17 (weak EDA guidance)
3. **2c**: Two-stage: fit multiple fixed-τ models, then compare

**Priors**:
- Same as Model 1, plus:
- τ ~ DiscreteUniform(5, 35) [Wide range, exclude endpoints]
- Alternative: τ ~ DiscreteUniform(12, 22) [Concentrated near EDA estimate]

**Falsification Criteria** - I will ABANDON this model if:
1. **Posterior τ is diffuse** (no clear break detected)
   - Entropy(p(τ|data)) > 0.8 × Entropy(prior)
2. **Posterior τ is multimodal** with 2+ peaks >0.15 probability each
   - Suggests multiple breaks or no dominant break
3. **Computational failure**: Sampling takes >30 minutes or won't converge
4. **Sensitivity to prior**: Changing τ prior range by ±5 substantially changes inference
5. **All Model 1 falsification criteria also apply**

**Additional Test**:
- Compare LOO against Model 1 (τ=17 fixed)
- If ΔLOO < 2: unknown τ doesn't improve fit → Occam's razor favors Model 1
- If ΔLOO > 4: Fixed τ=17 is inadequate → Proceed with Model 2

**Why It Might Fail**:
- Computational cost prohibitive (marginalizing over 30 τ values)
- Discrete parameter space poorly explored by HMC
- Small N=40 means limited resolution (can't distinguish τ=16 vs. 17)
- May overfit: finding spurious break to minimize residuals

**Implementation**: PyMC (better discrete parameter support than Stan)
**Expected Runtime**: 10-30 minutes
**Priority**: MEDIUM (only if Model 1 raises concerns)

---

### Model Class 3: Multiple Changepoint Model
**Core Idea**: Single break is oversimplification; test for k=2 changepoints

**Mathematical Form**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β₀ + β₁·year_t + β₂·I(t>τ₁)·(year_t - year_τ₁)
           + β₃·I(t>τ₂)·(year_t - year_τ₂) + AR(1)
τ₁ < τ₂ (ordered)
```

**Variants**:
1. **3a**: Fixed k=2 changepoints (simplest)
2. **3b**: Variable k with model selection k∈{0,1,2,3}
3. **3c**: Reversible-jump MCMC (k is random; most complex)

**Priors**:
- τ₁ ~ DiscreteUniform(5, 25)
- τ₂ ~ DiscreteUniform(τ₁+3, 35) [At least 3 observations between breaks]
- β₀ ~ Normal(4.3, 0.5)
- β₁ ~ Normal(0.2, 0.3) [Initial slope]
- β₂ ~ Normal(0.5, 0.5) [First break]
- β₃ ~ Normal(0.5, 0.5) [Second break]
- Other parameters same as Model 1

**Falsification Criteria** - I will ABANDON this model if:
1. **Posterior strongly favors k=1**: P(k=1|data) > 0.8 in variable-k version
2. **Changepoints too close**: Posterior |τ₂ - τ₁| < 5 observations
   - Model artificially splitting single break
3. **Break magnitudes inconsistent**:
   - One β large, others ≈0 (collapsing to k=1)
   - All β's similar (suggests smooth trend, not breaks)
4. **Predictive degradation**: Out-of-sample performance WORSE than k=1 model
   - Classic overfitting signature
5. **Computational failure**: Won't converge or takes >1 hour
6. **Model comparison**: ΔLOO(k=2, k=1) < 4 (not worth added complexity)

**Why It Might Fail**:
- N=40 too small to reliably detect 2+ changepoints (overfitting)
- Computational intractability (discrete parameter space explodes)
- Identifiability issues: close changepoints confounded with autocorrelation
- Occam's razor: simpler k=1 model is sufficient

**Implementation**: PyMC (only practical option for multiple discrete parameters)
**Expected Runtime**: 20-60 minutes
**Priority**: LOW (only if Models 1 and 2 suggest multiple breaks)

---

## Red Flags and Decision Points

### Checkpoint 1: After Model 1 Fitting
**Decision Points**:

1. **If Model 1 passes all falsification tests**:
   - STOP HERE
   - Report Model 1 as best
   - Document limitations (assumes τ=17, assumes single break)
   - Recommend sensitivity analyses but no further models

2. **If Model 1 fails on β₂ significance**:
   - Break may not exist
   - → Try Model 2 (unknown τ) to see if different location works
   - → OR abandon changepoint framework (try GP/spline)

3. **If Model 1 fails on computational issues**:
   - Model structure is pathological
   - → Try simpler variant (1a: no AR)
   - → If still fails, model class is fundamentally wrong

4. **If Model 1 fails on residual autocorrelation**:
   - AR(1) insufficient
   - → Try AR(2) or alternative autocorrelation structure
   - → Consider state-space model instead

5. **If Model 1 fails on predictive performance**:
   - Overfitting despite good in-sample fit
   - → Need more regularization OR simpler model
   - → Consider informative priors or ridge penalty

### Checkpoint 2: After Model 2 Fitting (if reached)
**Decision Points**:

1. **If Model 2 finds τ ≈ 17 (posterior concentrated)**:
   - Validates Model 1
   - Use Model 2 for inference (properly accounts for τ uncertainty)
   - Report that EDA changepoint identification was correct

2. **If Model 2 finds τ far from 17 (e.g., τ ≈ 10 or τ ≈ 25)**:
   - EDA was wrong or τ=17 was spurious
   - Investigate: why did EDA suggest τ=17?
   - Check data at new τ location for scientific plausibility

3. **If Model 2 posterior is bimodal** (e.g., modes at τ=12 and τ=22):
   - Multiple breaks may exist
   - → Try Model 3 (k=2)
   - → OR model may be unidentified

4. **If Model 2 posterior is uniform** (no concentration):
   - No clear changepoint exists
   - → ABANDON changepoint framework
   - → Pivot to GP or polynomial models

5. **If Model 2 has computational disaster**:
   - Discrete parameter sampling is impractical
   - → Revert to Model 1 with sensitivity analysis on τ
   - → Try grid search: fit Model 1 with τ ∈ {10, 12, 14, 16, 18, 20, 22} and compare LOO

### Checkpoint 3: After Model 3 Fitting (if reached)
**Decision Points**:

1. **If Model 3 prefers k=1** (single changepoint):
   - Confirms single break hypothesis
   - Revert to Model 2 results
   - Extra complexity of k=2 not justified

2. **If Model 3 prefers k=2 with well-separated τ's**:
   - Multiple regimes may be real
   - Check scientific plausibility: what changed at both times?
   - CAUTIOUSLY report k=2 with large uncertainty flags

3. **If Model 3 finds τ₁ ≈ τ₂** (within 5 observations):
   - Model is splitting a single break
   - Collapse to k=1 model

4. **If Model 3 has worse predictive performance than k=1**:
   - Overfitting confirmed
   - Reject k=2 model despite good in-sample fit

### Global Red Flags (Across All Models)

**ABANDON CHANGEPOINT FRAMEWORK ENTIRELY if**:

1. **All models show computational pathology**:
   - Divergences, non-convergence, extreme runtimes
   - → Model class is fundamentally unidentified or pathological

2. **All models have similar predictive performance**:
   - Can't distinguish between k=0, k=1, k=2
   - → Data insufficient to support changepoint inference

3. **Gaussian Process or spline model vastly superior**:
   - ΔLOO > 10 favoring smooth model
   - → "Break" is actually smooth acceleration

4. **Posterior predictive checks fail systematically**:
   - Models can't reproduce key data features
   - → Likelihood family (NB) or link function (log) is wrong

5. **Simulation studies show non-recovery**:
   - Models can't recover known changepoints from synthetic data
   - → Models are broken; don't trust real data results

---

## Alternative Approaches (Escape Routes)

If changepoint models fail, I will immediately propose:

### Plan B: Gaussian Process Model
**Why**: Flexibly captures smooth and sharp transitions without changepoint assumption

```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = f(year_t)
f ~ GP(0, k_Matérn(year, ρ, ℓ))
```

**Advantages**:
- No discrete changepoint assumption
- Naturally handles autocorrelation
- Can capture smooth or sharp transitions

**When to try**: If Model 2 posterior is diffuse OR smooth alternative hypothesized

### Plan C: Bayesian Structural Time Series
**Why**: Time-varying slopes naturally capture acceleration without explicit breaks

```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = level_t + trend_t
level_t ~ Normal(level_{t-1} + trend_{t-1}, σ_level)
trend_t ~ Normal(trend_{t-1}, σ_trend)
```

**Advantages**:
- Non-stationary trends allowed
- Smoother than changepoint but more structured than GP
- Natural state-space interpretation

**When to try**: If autocorrelation issues persist OR need forecasting

### Plan D: Polynomial + Spline Hybrid
**Why**: Quadratic base with spline refinement balances flexibility and parsimony

```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β₀ + β₁·year + β₂·year² + Σᵢ βᵢ·B_i(year) + AR(1)
```

**Advantages**:
- Simpler than GP
- More interpretable than full spline
- Can still capture non-linearity

**When to try**: If GP is too flexible OR interpretability is paramount

### Plan E: State-Space with Interventions
**Why**: Explicitly model known or unknown intervention at t≈17

```
log(μ_t) = α_t + β·intervention_t
α_t ~ Normal(α_{t-1}, σ_α)
```

**Advantages**:
- Separates smooth evolution from discrete intervention
- Allows partial changepoint (level shift without slope change)
- Better for causal interpretation

**When to try**: If scientific context suggests external intervention

---

## Validation and Stress Testing

### Prior Predictive Checks (Before Fitting)
**Goal**: Ensure priors don't preclude observed patterns

**Tests**:
1. Generate 1000 datasets from prior
2. Check coverage of observed statistics:
   - Variance/mean ratio ∈ [10, 200]? (Observed: 67.99)
   - Growth rate increase ∈ [200%, 1500%]? (Observed: 730%)
   - Maximum count ∈ [100, 500]? (Observed: 272)
3. If <5% of prior predictive samples cover observed: priors too restrictive

**Rejection Rule**: If prior predictive coverage <1% for key statistics, revise priors before fitting.

### Simulation-Based Calibration (Model Validation)
**Goal**: Can model recover known parameters from synthetic data?

**Protocol**:
1. Simulate 100 datasets with known parameters:
   - τ=17, β₂=1.0, ρ=0.85, α=0.6
2. Fit model to each dataset
3. Check: Are true parameters within 95% credible intervals ~95% of time?
4. Check: Are posteriors well-calibrated (rank histograms uniform)?

**Rejection Rule**: If calibration fails (true parameters outside CIs >20% of time), model is fundamentally broken.

### Posterior Predictive Checks (After Fitting)
**Goal**: Does model reproduce key data features?

**Test Statistics**:
1. **Autocorrelation**: ACF(1) from posterior predictive vs. observed
2. **Break magnitude**: Growth rate change at t=17 from posterior predictive vs. observed 730%
3. **Dispersion**: Variance/mean ratio from posterior predictive vs. observed 67.99
4. **Extremes**: Maximum count from posterior predictive vs. observed 272
5. **Trend**: Overall growth rate posterior predictive vs. observed 745% over 40 periods
6. **Regime means**: Pre-break vs. post-break mean from posterior predictive

**Rejection Rule**: If p-value <0.05 or >0.95 for 2+ test statistics, model fails to capture data structure.

### Cross-Validation (Predictive Performance)
**Goal**: Does model generalize to unseen data?

**Time-Series CV Protocol**:
1. Train on observations 1:25, predict 26:30
2. Train on observations 1:30, predict 31:35
3. Train on observations 1:35, predict 36:40

**Metrics**:
- Log-score: sum(log p(C_t | data_{1:train}))
- RMSE on log scale
- Absolute error on count scale
- Coverage: fraction of observations in 95% predictive interval

**Rejection Rule**: If out-of-sample RMSE > 1.5x in-sample RMSE, model is overfitting.

### LOO Cross-Validation (Model Comparison)
**Goal**: Which model predicts best?

**Protocol**:
1. Compute LOO for each model using PSIS-LOO
2. Check Pareto k diagnostics:
   - Good: k < 0.5 for all observations
   - Concerning: k ∈ [0.5, 0.7] for <10%
   - Bad: k > 0.7 for ≥10% (model misspecified)
3. Compare ΔLOO between models:
   - |ΔLOO| < 4: Models equivalent; prefer simpler
   - |ΔLOO| > 4: Strong preference for better model

**Rejection Rule**: If Pareto k >0.7 for >10% of observations, model is misspecified.

### Sensitivity Analysis (Robustness)
**Goal**: Are conclusions robust to prior choices?

**Protocol**:
1. **Prior width**: Scale all prior SDs by 0.5x and 2x
2. **Prior location**: Shift prior means by ±1 SD
3. **Prior family**: Try Normal vs. Student-t for regression parameters
4. **Refit and compare**:
   - Do posteriors change by <10%?
   - Do model rankings change?

**Rejection Rule**: If posterior mean changes by >50% or model ranking reverses, inference is prior-dependent (concerning).

---

## Expected Outcomes and Interpretation

### Scenario 1: Model 1 (Fixed τ=17) Passes All Tests
**Interpretation**:
- Discrete changepoint at observation 17 is well-supported
- EDA changepoint identification was correct
- Growth regime shift is real phenomenon

**Caveats**:
- Assumes τ=17 is exact (may be ±2 observations)
- Assumes single break (multiple breaks not ruled out)
- Predictive performance good but not perfect

**Recommendation**:
- Report Model 1 as primary result
- Conduct sensitivity analysis: try τ ∈ {15, 16, 18, 19}
- Acknowledge assumption that break location is known

### Scenario 2: Model 2 (Unknown τ) Validates τ≈17
**Interpretation**:
- Changepoint exists and data independently identifies it near t=17
- Stronger evidence than Scenario 1 (not assuming τ)
- Properly accounts for changepoint location uncertainty

**Caveats**:
- Computational cost higher than Model 1
- Posterior may be wide (e.g., τ ∈ [15, 20])
- Small sample means limited resolution

**Recommendation**:
- Report Model 2 as primary result
- Quantify uncertainty in τ location
- Model 1 can be used for faster inference if τ posterior is tight

### Scenario 3: Model 2 Finds τ Far From 17
**Interpretation**:
- EDA changepoint identification was spurious or suboptimal
- True break at different location (e.g., t=10 or t=25)
- Investigate scientific plausibility of alternative τ

**Caveats**:
- Why did EDA suggest τ=17? (Investigate methodologically)
- Alternative τ may be data-driven spurious finding
- Need external validation (domain knowledge, additional data)

**Recommendation**:
- Report uncertainty prominently
- Compare models with τ=17 vs. τ=new via LOO
- Seek external evidence for break at alternative location

### Scenario 4: Model 2 Posterior is Diffuse (No Clear Break)
**Interpretation**:
- No strong evidence for discrete changepoint
- Observed pattern may be smooth acceleration, not regime shift
- Small sample size may limit detection power

**Caveats**:
- Doesn't prove no break exists (just not detectable)
- May need more data or different model class

**Recommendation**:
- Abandon changepoint framework
- Try Gaussian Process or polynomial models
- Report that discrete break hypothesis is not strongly supported

### Scenario 5: Model 3 Finds Multiple Breaks
**Interpretation**:
- Single changepoint is oversimplification
- Multiple smaller regime shifts aggregate to apparent large break
- Time-varying growth process with 2+ inflection points

**Caveats**:
- High overfitting risk with N=40 and k=2+ changepoints
- May not generalize to future data
- Scientifically: need explanation for multiple breaks

**Recommendation**:
- Report with large uncertainty caveats
- Investigate what changed at each identified break point
- Prefer simpler model unless strong evidence (ΔLOO > 10)

### Scenario 6: All Models Fail
**Interpretation**:
- Changepoint framework inappropriate for this data
- Possible issues:
  - Measurement artifacts, not real process change
  - Process is smooth, not discrete
  - Sample size too small for reliable inference
  - Likelihood family (NB) or structure (AR) inadequate

**Caveats**:
- Doesn't mean data is bad (just model class is wrong)
- May need completely different approach

**Recommendation**:
- Immediately pivot to Plan B (Gaussian Process)
- Try alternative likelihood families (Zero-inflated NB, Poisson-lognormal)
- Consider requesting more data or different analysis approach
- Acknowledge severe modeling limitations

---

## Implementation Timeline

### Phase 1: Pre-Fitting Validation (4-6 hours)
- [ ] Write Stan code for Model 1 variants (1a, 1b, 1c)
- [ ] Write PyMC code for Model 2
- [ ] Implement prior predictive check script
- [ ] Run prior predictive checks for all models
- [ ] Implement simulation-based calibration for Model 1
- [ ] Run SBC: simulate 100 datasets, check recovery
- [ ] **DECISION POINT**: If SBC fails, debug or abandon model

### Phase 2: Model 1 Fitting (2-4 hours)
- [ ] Fit Model 1a (random effects, simplest)
- [ ] Check convergence diagnostics (Rhat, ESS, divergences)
- [ ] If issues: try non-centered parameterization or stronger priors
- [ ] If 1a passes: use as baseline
- [ ] If 1a fails: try 1b (AR1), then 1c (AR1 + time-varying α)
- [ ] **DECISION POINT**: If Model 1 passes falsification tests, STOP

### Phase 3: Model 1 Validation (3-5 hours)
- [ ] Posterior predictive checks (6+ test statistics)
- [ ] LOO cross-validation with Pareto k diagnostics
- [ ] Time-series CV: train on 1:30, predict 31:40
- [ ] Sensitivity analysis: vary priors ±2x
- [ ] Visualize: posteriors, predictive intervals, residuals
- [ ] **DECISION POINT**: If all pass, accept Model 1; if fail, proceed to Model 2

### Phase 4: Model 2 Fitting (If Needed, 4-8 hours)
- [ ] Fit Model 2 with PyMC (unknown τ)
- [ ] Monitor MCMC diagnostics carefully (discrete parameters tricky)
- [ ] If convergence issues: try tighter prior or grid marginalization
- [ ] Extract posterior for τ: plot histogram
- [ ] Compare τ posterior to EDA suggestion (τ=17)
- [ ] **DECISION POINT**: Based on τ posterior shape (concentrated, bimodal, uniform)

### Phase 5: Model 2 Validation (If Needed, 3-5 hours)
- [ ] Same validation as Phase 3
- [ ] Additional: sensitivity to τ prior (uniform vs. concentrated)
- [ ] Compare LOO against Model 1 (fixed τ)
- [ ] **DECISION POINT**: Accept Model 2 or proceed to Model 3

### Phase 6: Model 3 Fitting (If Needed, 6-12 hours)
- [ ] Only attempt if Models 1 and 2 suggest multiple breaks
- [ ] Fit Model 3 (k=2 fixed) with PyMC
- [ ] Expect computational challenges (long runtime)
- [ ] Extract posteriors for τ₁ and τ₂
- [ ] Check: are τ₁ and τ₂ well-separated (>5 obs)?
- [ ] **DECISION POINT**: Compare k=0, k=1, k=2 via LOO

### Phase 7: Model Comparison and Selection (2-4 hours)
- [ ] Create LOO comparison table across all fitted models
- [ ] Weight models by stacking or pseudo-BMA (if multiple pass)
- [ ] Generate ensemble predictions
- [ ] Create comparison visualizations
- [ ] **FINAL DECISION**: Select best model or recommend alternative

### Phase 8: Reporting (2-4 hours)
- [ ] Write results document with all diagnostics
- [ ] Create publication-quality figures
- [ ] Document all falsification test results (pass/fail table)
- [ ] Acknowledge limitations and uncertainties
- [ ] Provide recommendations for future work

**Total Estimated Time**: 20-40 hours depending on which models are fit

**Bottlenecks**:
- Model 2 and 3 computational costs (may require simplification or parallelization)
- Debugging convergence issues (can add 5-10 hours if pathological)
- Validation suite is comprehensive (intentionally thorough)

---

## Success Criteria

### What Success Looks Like
**Success is NOT**:
- Fitting a model that "works"
- Getting pretty plots and nice posteriors
- Publishing a positive result

**Success IS**:
- **Discovering truth**: Determining whether changepoint hypothesis is supported
- **Honest uncertainty**: Quantifying what we don't know
- **Falsification**: Actively trying to break our models and reporting results
- **Adaptive**: Changing approaches when evidence demands it

### Metrics for Success
1. **At least one model passes all falsification tests**
2. **Predictive performance validated on held-out data**
3. **Robust to prior perturbations (±2x doesn't change conclusions)**
4. **Simulation studies confirm model can recover truth**
5. **Posterior predictive checks pass for 90%+ test statistics**
6. **Clear answer to core question**: Does discrete changepoint exist?

### What Would Be FAILURE
1. **Forcing a model to "work" by weakening falsification criteria**
2. **Ignoring computational warnings or poor diagnostics**
3. **Cherry-picking results that support preferred hypothesis**
4. **Not attempting alternative model classes when changepoints fail**
5. **Reporting point estimates without uncertainty or caveats**

---

## Commitment to Scientific Integrity

I commit to:

1. **Reporting all results**, including negative findings and failures
2. **Following decision tree rigorously**, not skipping checkpoints
3. **Abandoning models that fail falsification**, even if it means more work
4. **Recommending alternative approaches** when changepoint framework is inadequate
5. **Acknowledging uncertainty** prominently in all communications
6. **Prioritizing predictive performance** over in-sample fit
7. **Seeking truth, not task completion**

If the data tells me there is no changepoint, I will report that clearly and confidently. If the data tells me the changepoint framework is wrong, I will pivot immediately to alternatives.

**The goal is understanding reality, not confirming hypotheses.**

---

## Deliverables

1. **Stan/PyMC Code**: All models implemented and documented
2. **Validation Scripts**: Prior predictive, SBC, posterior predictive, CV
3. **Results Report**: Model comparison with all diagnostics
4. **Visualizations**: Posteriors, predictions, diagnostics, comparisons
5. **Falsification Results**: Pass/fail table for all criteria
6. **Recommendations**: Best model(s) or alternative approaches needed
7. **Limitations Document**: What we learned we cannot determine

**All code will use Stan or PyMC for probabilistic inference (no frequentist baselines).**

---

## Contact and Collaboration

**Model Designer 1** (Changepoint/Regime-Switching Focus)
**Specialty**: Discrete structural breaks, time-series models, Bayesian changepoint detection
**Philosophy**: Falsification over confirmation; truth over convenience

**Coordination with Other Designers**:
- Open to model comparison against smooth alternatives (GP, spline)
- Will provide LOO scores for inter-designer model comparison
- Committed to ensemble modeling if multiple model classes viable

**Questions/Concerns**: Flag immediately if:
- Computational issues persist across multiple reformulations
- Domain knowledge suggests changepoint hypothesis is implausible
- Alternative model classes are clearly superior (ΔLOO > 10)

---

**END OF EXPERIMENT PLAN**

*Generated: 2025-10-29*
*Designer: Model Designer 1*
*Status: Ready for implementation*
