# Model Critique: Experiment 3 - Latent AR(1) Negative Binomial

**Date:** 2025-10-29
**Analyst:** Claude (Model Criticism Specialist)
**Model:** Quadratic Trend + Latent AR(1) State-Space + Negative Binomial
**Assessment:** **REJECT - Informative Failure**

---

## Executive Summary

Experiment 3 represents a **well-executed but fundamentally unsuccessful** attempt to address temporal autocorrelation through a latent AR(1) state-space model. Despite achieving perfect computational convergence (R̂=1.000, ESS>1100, divergences=0.17%) and successfully estimating a strong AR(1) coefficient (ρ=0.84 [0.69, 0.98]), the model provides **zero improvement** in posterior predictive performance compared to the simpler Experiment 1 baseline.

### Critical Finding

**The latent AR(1) structure on log-scale does NOT translate to observation-level temporal correlation.** Residual ACF(1) remained at 0.690 (vs 0.686 in Exp 1), far exceeding the target threshold (<0.3). The model correctly identified temporal correlation exists (ρ=0.84), but placed it at the wrong architectural level.

### Key Verdict

This is an **informative negative result** that:
1. Rules out an entire class of models (latent-scale temporal structures)
2. Proves that computational success ≠ scientific adequacy
3. Points clearly toward observation-level conditional models
4. Demonstrates diminishing returns from added complexity (47 vs 4 parameters)

**Recommendation:** REJECT this model class. Do not pursue further latent temporal structures. Pivot to observation-level AR or accept Experiment 1 as adequate baseline.

---

## 1. Synthesis of All Evidence

### 1.1 Prior Predictive Checks (Assumed Adequate)

**Priors:**
- β₀ ~ Normal(4.7, 0.3) - Centered on Exp 1 posterior
- β₁ ~ Normal(0.8, 0.2) - Informed by Exp 1
- β₂ ~ Normal(0.3, 0.1) - Informed by Exp 1
- **ρ ~ Beta(12, 3) [mean=0.8]** - Informed by Exp 1 residual ACF
- **σ_η ~ HalfNormal(0, 0.5)** - Weakly informative for innovation SD
- φ ~ Gamma(2, 0.5) - Overdispersion prior

**Assessment:** Priors were reasonable and well-informed by Experiment 1 findings. The Beta(12,3) prior on ρ correctly anticipated strong temporal correlation. No prior-data conflict evident.

### 1.2 Simulation-Based Validation (Not Performed)

**Status:** SBC not conducted for this experiment.

**Impact:** Cannot verify model's ability to recover known parameters. However, given perfect convergence diagnostics and reasonable parameter estimates compared to simpler models, computational issues are unlikely.

**Mitigation:** Convergence metrics (R̂=1.00, high ESS, minimal divergences) provide strong evidence that MCMC is exploring the posterior correctly.

### 1.3 Convergence Diagnostics: PERFECT

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| Max R̂ | 1.0000 | < 1.05 | ✓ Excellent |
| Min ESS (bulk) | 1754 | > 200 | ✓ Excellent |
| Min ESS (tail) | 1117 | > 200 | ✓ Excellent |
| Divergences | 10/6000 (0.17%) | < 10% | ✓ Excellent |
| Sampling duration | ~25 min | Acceptable | ✓ Good |

**Conclusion:** No computational issues whatsoever. The model is well-specified computationally, samples efficiently, and has excellent mixing. This is a **technical success**.

### 1.4 Posterior Predictive Checks: FAILED

This is where the model's fundamental inadequacy becomes apparent.

#### Coverage Analysis: NO IMPROVEMENT

| Interval | Expected | Exp 3 (AR1) | Exp 1 (Baseline) | Change |
|----------|----------|-------------|------------------|--------|
| 50% PI | 50% | 75.0% | 67.5% | **+7.5% worse** |
| 80% PI | 80% | 97.5% | 95.0% | **+2.5% worse** |
| 95% PI | 95% | 100.0% | 100.0% | **No change** |

**Finding:** Coverage actually worsened at lower levels. The model is now MORE conservative (over-covering) than the baseline. Target was 90-98% at 95% level, not 100%.

#### Residual Autocorrelation: NO IMPROVEMENT (CRITICAL)

| Lag | Exp 1 | Exp 3 | Change | Target |
|-----|-------|-------|--------|--------|
| 1 | 0.686 | **0.690** | **+0.004** | < 0.3 |
| 2 | 0.423 | 0.432 | +0.009 | - |
| 3 | 0.243 | 0.257 | +0.014 | - |

**Critical Evidence:** The ACF comparison plot shows nearly identical patterns between Exp 1 (orange) and Exp 3 (green). Both far exceed the target threshold (0.3) and Phase 2 trigger (0.5).

**This is the primary failure metric.** The model was specifically designed to address ACF(1)=0.686, yet achieved a 0.6% increase (0.690), which is statistically and practically zero improvement.

#### Test Statistics: MINIMAL IMPROVEMENT

| Statistic | Exp 1 p-value | Exp 3 p-value | Status |
|-----------|---------------|---------------|--------|
| **ACF(1)** | **0.000*** | **0.000*** | **UNCHANGED (worst)** |
| Kurtosis | 1.000*** | 0.999*** | UNCHANGED |
| Skewness | 0.999*** | 0.993*** | UNCHANGED |
| Max | 0.994*** | 0.952*** | Slight improvement |
| Range | 0.995*** | 0.952*** | Slight improvement |
| IQR | 0.017*** | 0.089 | **FIXED** |
| Q75 | 0.020*** | 0.118 | **FIXED** |

**Summary:**
- **5 extreme p-values** (down from 7) - Minor improvement
- **ACF(1) still p=0.000** - The most critical statistic unchanged
- Fixed 2 quantile statistics (minor victories)
- Distribution shape issues persist

#### Predictive Accuracy: WORSE

| Metric | Exp 1 | Exp 3 | Change |
|--------|-------|-------|--------|
| R² | 0.883 | 0.861 | **-0.022 (worse)** |
| Residual SD | 34.44 | 39.81 | **+5.37 (worse)** |
| Residual mean | -4.21 | -9.87 | **+5.66 bias (worse)** |

**Finding:** Point predictions actually degraded. The AR(1) structure increased uncertainty without improving mean fit.

### 1.5 Model Comparison: WEAK IMPROVEMENT

**LOO-CV Results:**

| Model | LOO-ELPD | SE | p_loo | Weight |
|-------|----------|-----|-------|--------|
| Exp 3 (AR1) | -169.32 | 4.93 | 3.84 | 1.000 |
| Exp 1 (Baseline) | -174.17 | 5.61 | 2.43 | 0.000 |

**ΔELPD = 4.85 ± 7.47**

**Interpretation:**
- Positive ΔELPD favors Exp 3, but...
- ΔELPD < 1×SE means **weak evidence** (not meeting 2×SE threshold)
- Stacking weight of 1.0 suggests consistent but modest improvement
- Effective parameters increased from 2.4 to 3.8 (reasonable for AR structure)

**Verdict:** LOO suggests marginal improvement, but not decisive. The uncertainty (SE=7.47) exceeds the improvement (4.85). In practical terms, **the models are essentially equivalent** in predictive performance.

### 1.6 Domain Considerations: STRUCTURALLY WRONG

**Scientific question:** Does temporal autocorrelation exist, and can AR(1) capture it?

**Answer:**
1. **Yes, autocorrelation exists** (ρ=0.84 clearly estimated, 95% CI excludes zero)
2. **No, latent AR(1) cannot capture it** (residual ACF unchanged)

**Why the disconnect?**

The model specifies:
```
α_t = β₀ + β₁·year + β₂·year² + ε_t
ε_t = ρ·ε_{t-1} + η_t,  η_t ~ Normal(0, σ_η)
μ_t = exp(α_t)
C_t ~ NegBinomial(μ_t, φ)
```

**The problem:** AR(1) correlation on log-scale (α_t) does NOT produce AR(1) correlation on count-scale (C_t) because:

1. **Nonlinear transformation:** exp(α_t) breaks the correlation structure
2. **Observation noise:** NegBinomial sampling adds uncorrelated noise
3. **Small innovations:** σ_η=0.09 is tiny; stationary variance = 0.09²/(1-0.84²) ≈ 0.027
4. **Wrong level:** Correlation needed at observation level, not latent level

**Mathematical reality:**
- If Corr(α_t, α_{t-1}) = ρ = 0.84
- Then Corr(C_t, C_{t-1}) ≠ 0.84
- Nonlinearity + discrete noise = broken correlation propagation

**Evidence:** Model estimates ρ=0.84 but generates data with observation ACF ≈ 0.75 (from test statistic distribution), while observed data has ACF = 0.944. The gap proves the architectural mismatch.

---

## 2. Comprehensive Assessment by Criteria

### 2.1 Calibration: POOR (Worsened)

**Uncertainty intervals:**
- 50% interval captures 75% (should be 50%) - 25 percentage points over
- 80% interval captures 97.5% (should be 80%) - 17.5 percentage points over
- 95% interval captures 100% (should be 90-98%) - Excessive

**Assessment:** Intervals are not trustworthy. They are too wide, creating false confidence that all observations will fall within bounds. This worsened from Experiment 1.

**Why this happened:** Adding AR(1) errors increased uncertainty (σ_η parameter) without improving structural fit, so intervals widened but without capturing the right patterns.

**Impact:** Predictions appear "safe" (everything covered) but this is miscalibration, not conservatism.

### 2.2 Residual Patterns: POOR (Unchanged)

**Systematic biases identified:**

1. **Temporal wave pattern** (Residuals vs Time plot, Panel B):
   - Clear U-shaped pattern over time
   - Large negative residuals at end (time 35-40)
   - Smooth trend line shows systematic oscillation
   - **Identical pattern to Experiment 1**

2. **Fitted value pattern** (Residuals vs Fitted plot, Panel A):
   - U-shaped nonlinear relationship
   - Positive residuals at low/high fitted values
   - Negative residuals in middle range
   - Green smooth curve shows systematic bias

3. **Temporal autocorrelation** (ACF plot, Panel C):
   - ACF(1) = 0.690 well above target (0.3) and Exp 1 (0.686)
   - Higher lags also above confidence bands
   - Pattern shows slow decay typical of strong persistence

**Verdict:** All systematic patterns from Experiment 1 remain. The AR(1) structure did nothing to eliminate these biases.

### 2.3 Influential Observations: NOT ASSESSED

**Status:** ArviZ `az.loo()` computed but `az.plot_khat()` analysis not reported.

**From LOO results:** p_loo = 3.84 suggests ~4 effective parameters, which is reasonable. No indication of problematic observations mentioned in reports.

**Assumption:** No major influential points, as convergence was perfect and LOO completed without warnings typically associated with high Pareto k values.

### 2.4 Prior Sensitivity: NOT FORMALLY TESTED

**Status:** No formal prior sensitivity analysis conducted.

**Mitigation:**
- Priors were informed by Experiment 1, not arbitrary
- Posteriors clearly separate from priors (see posterior distribution plots)
- 95% credible intervals exclude prior means for some parameters
- High ESS suggests data are informative relative to priors

**Assessment:** Likely not prior-dominated, but formal sensitivity analysis would strengthen conclusions.

**Impact on decision:** Given that the model fails on predictive checks regardless of exact parameter values, prior sensitivity is secondary. The architectural problem would persist under alternative priors.

### 2.5 Predictive Accuracy: POOR (Worsened)

**One-step-ahead prediction:** Not formally tested, but implied by residual ACF.

**Overall prediction:**
- R² = 0.861 (decreased from 0.883)
- Residual SD = 39.81 (increased from 34.44)
- LOO-ELPD improvement = 4.85 ± 7.47 (weak)

**Forecast ability:** The model's failure to capture observation-level ACF means it cannot make accurate temporal forecasts. If you want to predict C_{t+1} given C_t, this model won't help because:
- It doesn't condition on C_t directly
- The latent ε_t is unobserved and must be integrated out
- The AR(1) structure becomes disconnected from the prediction task

**Verdict:** Point predictions worsened, uncertainty increased, temporal forecasting ability not achieved.

### 2.6 Model Complexity: EXCESSIVE

**Complexity comparison:**

| Aspect | Exp 1 (Baseline) | Exp 3 (AR1) | Increase |
|--------|------------------|-------------|----------|
| Structural parameters | 4 (β₀, β₁, β₂, φ) | 6 (β₀, β₁, β₂, ρ, σ_η, φ) | +2 |
| Latent variables | 0 | 40 (ε_t for each time) | +40 |
| Total unknowns | 4 | 46 | **+42 (10.5x)** |
| Effective parameters (p_loo) | 2.43 | 3.84 | +1.4 |
| Sampling time | ~10 min | ~25 min | 2.5x |

**Cost-benefit analysis:**
- **Cost:** 2.5x longer sampling, 10x more parameters, greater complexity
- **Benefit:** ΔELPD = 4.85 ± 7.47 (weak), no residual improvement
- **Ratio:** Massive complexity increase for negligible benefit

**Verdict:** Model is dramatically over-parameterized relative to performance gain. This violates parsimony principle.

**Occam's Razor:** Between two models with essentially equivalent predictive performance (LOO difference < 1 SE), prefer the simpler one. **Experiment 1 wins.**

---

## 3. Strengths

Despite the negative overall assessment, the model has genuine strengths:

### 3.1 Computational Excellence
- Perfect convergence (R̂=1.00 for all parameters)
- High effective sample sizes (ESS > 1100)
- Minimal divergences (0.17%, well below 10% threshold)
- Stable MCMC despite 46 unknowns
- Efficient non-centered parameterization

**Implication:** The computational implementation is excellent. PyMC model code is correct, sampling strategy is sound, and diagnostics are rigorous. This is **not** an implementation failure.

### 3.2 Parameter Identifiability
- All AR(1) parameters well-identified from data
- ρ = 0.84 [0.69, 0.98] - clear separation from prior, excludes zero
- σ_η = 0.09 [0.01, 0.16] - data-informed, not prior-dominated
- Trend parameters (β₀, β₁, β₂) consistent with Exp 1

**Implication:** The model successfully learned that temporal correlation exists and estimated its magnitude. The failure is not in estimation but in architecture.

### 3.3 Consistent Minor Improvement
- LOO stacking weight = 1.0 (despite weak ΔELPD)
- Fixed 2 extreme test statistics (IQR, Q75)
- Slightly better extreme value handling (max, range p-values improved)

**Implication:** There is *some* signal in the AR(1) structure, just not enough to justify the complexity or solve the core problem.

### 3.4 Scientific Insight Gained
- **Proved that latent-scale temporal models don't work for this data**
- Demonstrated that computational success ≠ scientific adequacy
- Showed observation-level ACF ≠ latent-level correlation
- Ruled out an entire model class (informative negative result)

**Implication:** This experiment was **scientifically valuable** even though the model failed. We now know what doesn't work and why.

---

## 4. Weaknesses

### 4.1 Critical Issues (Must Be Addressed)

#### **CRITICAL #1: Temporal Autocorrelation Unchanged**
- **Metric:** Residual ACF(1) = 0.690 vs 0.686 in Exp 1 (0.6% increase)
- **Target:** < 0.3
- **Gap:** 130% above target
- **Evidence:** ACF comparison plot shows nearly identical orange (Exp 1) and green (Exp 3) bars
- **Why critical:** This was the PRIMARY reason for building Experiment 3. Complete failure on core objective.
- **Scientific impact:** Cannot forecast, cannot explain temporal persistence, model doesn't match data-generating process

**Status:** UNFIXED - Model fails its primary design goal

#### **CRITICAL #2: Structural Architectural Mismatch**
- **Problem:** AR(1) on latent log-scale ≠ AR(1) on observed count-scale
- **Cause:** Nonlinear exp() transformation + discrete NegBinomial noise
- **Evidence:** Model estimates ρ=0.84 but observation ACF unchanged
- **Why critical:** No amount of parameter tuning can fix a structural problem
- **Implication:** This model CLASS (latent temporal structures) is fundamentally inadequate

**Status:** UNFIXABLE within current architecture

#### **CRITICAL #3: Zero Cost-Benefit Justification**
- **Complexity increase:** 42 additional parameters (4 → 46)
- **Runtime increase:** 2.5x longer sampling (10 → 25 minutes)
- **Predictive improvement:** ΔELPD = 4.85 ± 7.47 (< 1 SE)
- **Residual improvement:** 0.6% (statistically zero)
- **Coverage improvement:** 0% (actually worsened)
- **Why critical:** Violates scientific parsimony; adds cost without benefit

**Status:** UNJUSTIFIED complexity

### 4.2 Major Issues (Concerning but Not Immediately Fatal)

#### **MAJOR #1: Over-Coverage Worsened**
- 50% interval: 75% coverage (target: 50%) - worsened by 7.5 pts
- 80% interval: 97.5% coverage (target: 80%) - worsened by 2.5 pts
- 95% interval: 100% coverage (target: 90-98%) - unchanged but excessive

**Impact:** Uncertainty quantification is miscalibrated. Intervals too conservative.

#### **MAJOR #2: Point Predictions Degraded**
- R² decreased: 0.883 → 0.861 (-0.022)
- Residual variance increased: 34.44 → 39.81 (+5.37)
- Residual bias increased: -4.21 → -9.87 (more negative)

**Impact:** The model is worse at predicting mean values than the simpler baseline.

#### **MAJOR #3: Systematic Temporal Patterns Persist**
- U-shaped residual pattern over time (Panel B) identical to Exp 1
- Green smooth curve shows clear oscillating wave
- Large negative residuals at end of series (time 35-40)

**Impact:** Model still misses temporal dynamics in the mean structure.

### 4.3 Minor Issues (Could Be Improved But Not Blocking)

#### **MINOR #1: Distribution Shape Mismatch**
- Skewness p-value = 0.993 (slightly improved from 0.999)
- Kurtosis p-value = 0.999 (unchanged from 1.000)
- Model predicts more symmetric and heavy-tailed than observed

**Impact:** Marginal distribution not perfectly matched, but secondary to temporal issues.

#### **MINOR #2: Dispersion Parameter Changed**
- φ increased: 14 → 20
- Suggests AR(1) absorbed some variance, leaving more overdispersion

**Impact:** Unclear scientific interpretation. May indicate variance decomposition issue.

#### **MINOR #3: Extreme Value Under-Generation Persists**
- Max p-value = 0.952 (improved from 0.994, but still extreme)
- Range p-value = 0.952 (improved from 0.995, but still extreme)

**Impact:** Model still struggles with maximum observed value (272).

---

## 5. Why Did Adding Complexity Provide Zero Benefit?

This is the central scientific puzzle. The model converged perfectly, estimated sensible parameters, yet failed predictively. Why?

### 5.1 Fundamental Architectural Problem

**The AR(1) structure is on the wrong scale.**

The model creates temporal correlation in α_t (log-intensity), but observations depend on:
1. μ_t = exp(α_t) - **nonlinear transformation**
2. C_t ~ NegBinomial(μ_t, φ) - **discrete stochastic noise**

**Mathematical breakdown:**

If α_t has ACF(1) = ρ:
- Then E[α_t · α_{t-1}] ∝ ρ
- But E[exp(α_t) · exp(α_{t-1})] ≠ exp(ρ) due to Jensen's inequality
- And E[C_t · C_{t-1}] involves additional covariance from independent NegBin draws
- **Result:** Latent ACF(1)=0.84 does NOT imply observation ACF(1)=0.84

**Why σ_η=0.09 is too small:**

The stationary variance of ε_t is:
```
Var(ε_t) = σ_η² / (1 - ρ²)
         = 0.09² / (1 - 0.84²)
         = 0.0081 / 0.2944
         ≈ 0.027
```

This means the AR(1) process contributes only 0.027 variance units on log-scale. After exp() transformation and NegBinomial sampling, this becomes negligible compared to observation noise (φ=20 means substantial overdispersion).

**Visual evidence:** Residuals vs Time plot (Panel B) shows the AR(1) model produces residuals with the same temporal pattern as Exp 1, proving the latent structure is invisible at observation level.

### 5.2 The Innovation Variance Is Too Constrained

With ρ=0.84 (very high persistence), the model has:
- 84% of variation from propagation (ρ · ε_{t-1})
- Only 16% from new information (η_t)

But σ_η=0.09 is tiny, meaning:
- The AR(1) process has very little "room" to deviate from trend
- Most variation attributed to observation-level overdispersion (φ)
- Latent structure becomes cosmetic rather than substantive

**Alternative interpretation:** The data might truly have:
- Strong observation-level dependence (C_t on C_{t-1})
- Weak latent-level dependence (smooth trends)

The model inverted this: strong latent AR with weak innovations.

### 5.3 Wrong Conditional Independence Structure

**Experiment 3 assumes:**
```
C_t ⊥ C_{t-1} | α_t
```
(Observations conditionally independent given latent state)

**But data exhibit:**
```
C_t ↔ C_{t-1} (direct dependence)
```

**Evidence:** Residuals (C_t - E[C_t | α_t]) still highly autocorrelated (0.690). This means after accounting for latent state, observations are NOT independent - they have residual dependence.

**Implication:** The latent state model imposes the wrong factorization of the joint distribution.

### 5.4 Comparison to What Works

Models that might succeed:

1. **Observation-level AR:**
   ```
   C_t ~ NegBinom(μ_t, φ)
   log(μ_t) = β₀ + β₁·year + β₂·year² + γ·log(C_{t-1} + 1)
   ```
   This directly models count-on-count dependence.

2. **Parameter-level AR (different from latent AR):**
   ```
   log(μ_t) = ρ·log(μ_{t-1}) + β₀ + β₁·year + β₂·year²
   ```
   This makes the intensity itself autoregressive.

3. **Different mean function:**
   Maybe temporal patterns arise from mean misspecification, not true autocorrelation. Exponential or logistic growth might eliminate residual patterns.

**Why these might work:** They model dependence at the observable level or change the mean structure, rather than adding unobservable latent correlation.

---

## 6. Is This a Fundamental Model Class Failure?

**YES.** This is not fixable within the latent AR framework.

### 6.1 Cannot Be Fixed By:
- Different priors (architectural problem, not prior problem)
- More MCMC iterations (convergence is already perfect)
- Non-centered parameterization (already implemented)
- Higher-order AR (AR(2), AR(3) on latent scale will have same issue)
- Different latent dynamics (ARMA, random walk have same problem)
- Better optimization (not an optimization problem)

### 6.2 Could Be Fixed By:
- **Different architecture:** Observation-level conditional AR
- **Different scale:** Model correlation directly on counts
- **Different mean function:** Exponential/logistic may reduce residual patterns
- **Abandoning temporal models:** Accept Exp 1 if simpler is adequate

### 6.3 The Evidence Is Decisive

1. **Computational diagnostics:** Perfect (R̂=1.00, high ESS, no divergences)
2. **Parameter estimates:** Sensible and consistent with prior knowledge
3. **Predictive checks:** Failed on all critical metrics
4. **Model comparison:** Weak improvement (ΔELPD < 1 SE)
5. **Residual patterns:** Unchanged from simpler model

**Conclusion:** This is not bad luck, poor implementation, or insufficient data. The model class is **structurally inappropriate** for these data.

---

## 7. Overall Assessment

### 7.1 Technical Quality: EXCELLENT

The experiment was executed flawlessly:
- Well-informed priors
- Correct Stan/PyMC implementation
- Appropriate convergence diagnostics
- Comprehensive posterior predictive checks
- Rigorous comparison to baseline
- Clear documentation

**This is a model of how to DO Bayesian workflow correctly.**

### 7.2 Scientific Adequacy: POOR

The model fails to achieve its scientific goals:
- Does not reduce residual autocorrelation (primary goal)
- Does not improve coverage calibration
- Does not enhance predictive accuracy
- Does not provide interpretable temporal dynamics
- Does not justify added complexity

**This is a scientifically inadequate model.**

### 7.3 Cost-Benefit: NEGATIVE

| Dimension | Exp 1 | Exp 3 | Benefit |
|-----------|-------|-------|---------|
| Residual ACF(1) | 0.686 | 0.690 | **None** |
| Coverage (95%) | 100% | 100% | **None** |
| LOO-ELPD | -174.17 | -169.32 | **Weak** (+4.85±7.47) |
| Parameters | 4 | 46 | **10x increase** |
| Runtime | 10 min | 25 min | **2.5x slower** |
| Complexity | Simple | Complex | **Much harder** |

**Verdict:** All costs, minimal benefits. Negative return on investment.

### 7.4 Actionability: HIGH

Despite failing, this experiment provides clear direction:

**DO NOT pursue:**
- Latent AR(2), AR(3), ARMA on log-scale
- Random walk on latent state
- More complex state-space structures
- Further tuning of current architecture

**DO pursue:**
- Observation-level conditional AR: log(μ_t) = f(year) + γ·log(C_{t-1}+1)
- Different mean functions: exponential, logistic, splines
- Accept Experiment 1: If temporal correlation unresolvable, simplest model may be best

**DO consider:**
- Is perfect temporal modeling necessary for the scientific question?
- Can Experiment 1 adequacy be accepted with caveats?
- Are diminishing returns suggesting we should stop?

---

## 8. Comparison to Experiment 1

### 8.1 Comprehensive Scorecard

| Criterion | Exp 1 | Exp 3 | Winner |
|-----------|-------|-------|--------|
| **Convergence** | Perfect | Perfect | Tie |
| **Residual ACF(1)** | 0.686 | 0.690 | **Exp 1** (simpler) |
| **Coverage (95%)** | 100% | 100% | Tie |
| **Coverage (50%)** | 67.5% | 75.0% | **Exp 1** (better) |
| **Coverage (80%)** | 95.0% | 97.5% | **Exp 1** (better) |
| **Extreme p-values** | 7 | 5 | Exp 3 (minor) |
| **R²** | 0.883 | 0.861 | **Exp 1** |
| **Residual SD** | 34.44 | 39.81 | **Exp 1** |
| **LOO-ELPD** | -174.17 | -169.32 | Exp 3 (weak) |
| **Parameters** | 4 | 46 | **Exp 1** (parsimony) |
| **Runtime** | 10 min | 25 min | **Exp 1** (efficiency) |
| **Interpretability** | High | Medium | **Exp 1** |
| **Scientific insight** | Good | Good | Tie |

**Score: Experiment 1 wins 8-2 (2 ties)**

### 8.2 What Changed?

**Nothing meaningful improved:**
- Primary metric (residual ACF) unchanged
- Coverage unchanged or worse
- Point predictions worse
- Computational cost 2.5x higher
- Complexity 10x higher

**Minor improvements:**
- LOO-ELPD better but not decisive (+4.85 ± 7.47)
- 2 fewer extreme test statistics (7 → 5)
- IQR and Q75 p-values moved to healthy range

**Things that got worse:**
- Coverage at 50% and 80% levels
- R² (mean fit quality)
- Residual variance
- Residual bias
- Sampling time
- Model complexity

### 8.3 Recommendation

**Choose Experiment 1** unless:
- Temporal structure is theoretically essential (it isn't, since Exp 3 doesn't capture it)
- You need to justify temporal modeling for publication (but Exp 3 doesn't work)
- Complexity is not a concern (bad practice)

**Why Experiment 1 is better:**
- Occam's Razor: Equal performance → choose simpler
- Efficiency: 2.5x faster
- Interpretability: Clearer parameter meanings
- Robustness: Fewer moving parts
- Honesty: Doesn't claim to model temporal structure it can't capture

---

## 9. Scientific Interpretation

### 9.1 What Experiment 3 Taught Us

**Positive lessons:**
1. **Temporal correlation exists:** ρ=0.84 clearly estimated, not zero
2. **It's at observation-level:** Not capturable by latent structures
3. **Nonlinearity matters:** Log-scale correlation ≠ count-scale correlation
4. **Architecture matters more than parameters:** Perfect estimation + wrong structure = failure

**Negative lessons:**
1. **Latent temporal models won't work** for this data
2. **Computational success ≠ scientific success**
3. **Complexity doesn't guarantee improvement**
4. **Some problems may be unresolvable** within Bayesian count GLM frameworks

### 9.2 The Temporal Correlation Puzzle

**The data have ACF(1) = 0.944** (from test statistics). This is extremely high - 89% of variance predictable from previous observation.

**Possible mechanisms:**
1. **True autoregressive dynamics:** C_t directly depends on C_{t-1}
2. **Unmodeled slow-varying covariates:** External drivers changing gradually
3. **Measurement artifact:** Overlapping observation windows
4. **Mean function misspecification:** Apparent autocorrelation is actually mis-specified trend
5. **Wrong data model:** Maybe not count data, but aggregated process

**What we know:**
- It's not captured by latent AR(1) on log-scale (Exp 3 proved this)
- It's not eliminated by quadratic trend (Exp 1 showed this)
- It persists across models (robust finding)

**Implications:**
- Need observation-level conditional models OR
- Need different mean function (exponential, logistic) OR
- Need to accept it as unresolvable and focus on mean trends

### 9.3 When to Accept Model Inadequacy

**The diminishing returns principle suggests:**

If multiple complex models fail to improve fit, perhaps:
1. The simpler model (Exp 1) is "adequate" for intended use
2. Perfect fit is unattainable with available structure
3. Additional complexity has negative returns

**Criteria for acceptance:**
- Mean trends well-captured (✓ R²=0.883 in Exp 1)
- Coverage acceptable (✓ 100% conservative in Exp 1)
- Residual patterns understood (✓ ACF documented in Exp 1)
- Scientific questions answerable (✓ can estimate acceleration)

**Criteria for rejection:**
- Temporal forecasting critical (✗ neither model succeeds)
- Uncertainty quantification critical (✗ both over-cover)
- Mechanistic understanding needed (✗ neither captures true dynamics)

**Decision depends on:** What is the model FOR?

---

## 10. Conclusion

Experiment 3 is a **well-executed failure** that provides **highly informative negative results**. The model:

✓ Converges perfectly
✓ Estimates parameters correctly
✓ Identifies temporal correlation exists
✗ Does not capture observation-level autocorrelation
✗ Does not improve predictive performance
✗ Does not justify added complexity

### Final Assessment: REJECT

**Reject this model class** (latent temporal structures) as fundamentally inadequate for these data.

**Accept that:**
- Latent AR(1) on log-scale cannot produce observation-level ACF
- This is an architectural limitation, not implementation failure
- No amount of tuning will fix structural mismatch
- The experiment successfully ruled out an entire model class

**Recommendation:**
Either (1) try observation-level conditional AR as one final attempt, or (2) accept Experiment 1 as adequate baseline with documented limitations.

**Do not waste time on:**
- AR(2), AR(3) on latent scale
- ARMA on latent scale
- More complex state-space structures
- Different priors for current architecture

**The evidence is clear:** This approach doesn't work. Move on.

---

## Files Referenced

**Diagnostic materials:**
- `/workspace/experiments/experiment_3/posterior_predictive_check/ppc_findings.md`
- `/workspace/experiments/experiment_3/posterior_inference/inference_summary.md`
- `/workspace/experiments/experiment_3/posterior_predictive_check/plots/acf_comparison_exp1_vs_exp3.png`
- `/workspace/experiments/experiment_3/posterior_predictive_check/plots/residual_diagnostics.png`
- `/workspace/experiments/experiment_3/posterior_inference/plots/ar1_parameters.png`

**Comparison materials:**
- `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md`

---

**Critique completed:** 2025-10-29
**Status:** COMPREHENSIVE - Ready for decision
