# Refinement Rationale: Experiment 2 → Experiment 2 Refined

**Date**: 2025-10-29
**Iteration**: Experiment 2 (v1) → Experiment 2 Refined (v2)
**Status**: Prior predictive check failed → Refined priors ready for validation

---

## Executive Summary

Experiment 2's prior predictive check revealed **catastrophic tail behavior**: 3.22% of simulated counts exceeded 10,000 (vs <1% threshold), with a maximum of 674 million compared to observed maximum of 269. The root cause was multiplicative amplification through the exponential link function, where rare combinations of extreme parameter values created astronomically large counts.

This refinement implements **three targeted constraints** while preserving the model's scientific structure:
1. **Truncate β₁** to prevent extreme growth rates
2. **Inform φ from Experiment 1** to stabilize variance
3. **Tighten σ** to constrain AR process innovations

These changes are scientifically justified, maintain model flexibility, and should reduce extreme outliers by >90% while keeping median behavior unchanged.

---

## Diagnostic Findings from Original Experiment 2

### Quantitative Failure Metrics

From `/workspace/experiments/experiment_2/prior_predictive_check/findings.md`:

**Count Distribution**:
- Mean: 91,019 (observed range: 21-269)
- Median: 112 ✓ (reasonable)
- Maximum: **674,970,346** (vs observed: 269)
- 99th percentile: **143,745** (vs observed: 269)

**Plausibility Violations**:
- 3.22% of counts > 10,000 (threshold: < 1%)
- Mean maximum per series: 2,038,561
- Some series peaked > 100,000

**AR(1) Process Behavior**:
- Epsilon range: [-10, +10] (some excursions beyond)
- Log-rate η: some trajectories reached 10-15
- Realized ACF(1): Mean 0.766 vs ρ mean 0.910 (weak correlation 0.39)

### Visual Evidence

**Key Plots** (from original experiment):
1. `prior_predictive_trajectories.png`: Shows explosive growth in small fraction of draws
2. `prior_predictive_coverage.png`: 95% interval at late timepoints ~30,000 (observed: 269)
3. `temporal_correlation_diagnostics.png`: Scatter between ρ and realized ACF shows poor alignment

### Root Cause Analysis

The problem is **multiplicative amplification** through exp(η):

**Mechanism**:
```
η = β₀ + β₁×year + ε_t
μ = exp(η)

When rare combination occurs:
  β₀ = 6.5   (from Normal(4.69, 1.0), ~3% in upper tail)
  β₁ = 2.0   (from Normal(1.0, 0.5), ~2.5% in upper tail)
  σ = 1.5    (from Exponential(2), ~8% in upper tail)
  ε_t = +4.5 (at late timepoint with large innovations)

Then: η = 6.5 + 2.0×1.67 + 4.5 = 14.3
      μ = exp(14.3) = 1.6 million
```

**Frequency**: With 500 draws × 40 timepoints = 20,000 samples, ~0.5% hit extreme combinations = ~100 extreme values

**Why median is OK but tail is catastrophic**:
- Most parameter draws are near prior means (β₀≈4.7, β₁≈1.0, σ≈0.5)
- These generate reasonable counts (50-500 range)
- But **exponential link amplifies rare joint extremes** non-linearly
- Small φ (high variance) further inflates extreme counts

---

## Refinement Strategy

### Principle: Targeted Constraints, Not Blanket Tightening

We **do not** simply make all priors narrower. Instead, we:
1. **Identify specific parameters** driving tail explosions
2. **Apply minimal constraints** to control extremes
3. **Preserve flexibility** where prior was working
4. **Use available information** (Experiment 1) judiciously

### Three-Pronged Approach

**Problem**: Wide priors + exponential link → tail explosion
**Solution**: Constrain growth (β₁), stabilize variance (φ), limit innovations (σ)

---

## Change 1: Truncate β₁ Growth Rate

### What Changed

```
Original:  β₁ ~ Normal(1.0, 0.5)
           → Allows β₁ ∈ (-∞, +∞), 95% in [0, 2.0]

Refined:   β₁ ~ TruncatedNormal(1.0, 0.5, lower=-0.5, upper=2.0)
           → Constrains β₁ ∈ [-0.5, 2.0], same central tendency
```

### Why This Change

**Problem Addressed**: Extreme growth rates in upper tail
- Original allows β₁ up to 2.5+ (very rare but possible)
- On exp scale over study period: exp(2.5 × 1.67) = 60× growth
- Combined with high β₀ and ε_t → million-count explosions

**Bounds Justification**:

**Lower bound = -0.5** (negative growth):
- Implies minimum growth: exp(-0.5 × 1.67) = 0.44× (56% decline)
- Observed data shows growth, not decline
- But allows model to discover weak/negative trend if present

**Upper bound = 2.0** (extreme positive growth):
- Implies maximum growth: exp(2.0 × 1.67) = 25× increase
- Observed data: 21 → 269 is 12.8× increase
- Bound is 2× the observed growth, plenty of uncertainty

**Experiment 1 Evidence**:
- Posterior: β₁ = 0.87 ± 0.04
- 95% CI: [0.80, 0.94]
- Well within [-0.5, 2.0], truncation won't affect inference

### Expected Impact

**Before truncation**: 2.5% of β₁ draws > 2.0
- These create growth factors > 25×
- Combined with other extremes → millions of counts

**After truncation**: 0% of β₁ draws > 2.0
- Maximum growth capped at 25× (still very high!)
- Eliminates runaway growth scenarios

**Quantitative**: Should reduce 99th percentile counts by ~50%

### Trade-offs

**Pros**:
- Prevents scientifically implausible growth rates
- Maintains flexibility for data (bound 2× observed)
- Computational stability (no unbounded tails)

**Cons**:
- Less "objective" than unbounded prior
- Truncation can cause MCMC boundary issues (minor risk)
- If true β₁ > 2.0, posterior will pile up at bound (CHECK THIS)

**Mitigation**:
- Posterior should be well away from bounds if model is right
- Visual check: plot posterior against truncation limits

---

## Change 2: Inform φ from Experiment 1

### What Changed

```
Original:  φ ~ Gamma(2, 0.1)
           → Mean = 20, SD = 14.1, Range ≈ [0.2, 70]

Refined:   φ ~ Normal(35, 15), constrained to φ > 0
           → Mean = 35, SD = 15, Range ≈ [5.6, 64.4]
```

### Why This Change

**Problem Addressed**: Wide φ prior allowed very small values (<5)
- Small φ → high variance: Var(C) = μ + μ²/φ
- When φ=2 and μ=1000: Var = 1000 + 500,000 = 501,000
- High variance amplifies extreme count draws
- Lower tail of Gamma(2, 0.1) problematic

**Information Source**: Experiment 1 posterior
- Posterior: φ = 35.6 ± 10.8
- 95% HDI: [17.7, 56.2]
- Highly informative data (ESS > 2500)

**Scientific Validity of Transfer**:

φ is a **data-generating parameter** that governs:
- Relationship between mean and variance in NB distribution
- Marginal count dispersion (not temporal structure)

Adding AR(1) affects:
- Temporal correlation in ε_t
- Dynamics of log-rate η over time
- **Does NOT fundamentally change** count variance given μ

**Analogy**: Like using sample variance as prior for variance in regression
- Not circular if parameter role is consistent
- φ describes same count process in both models
- AR(1) adds temporal structure, doesn't redefine dispersion

**Why Not Just Tighten Gamma Prior?**:
- Could use Gamma(5, 0.1) → mean=50, SD=22.4
- But we have **actual posterior information** from same data
- Using it is more principled than arbitrary tightening
- Still allows ±43% uncertainty (broad!)

### Expected Impact

**Before**:
- 10% of φ draws < 8 (high variance regime)
- Small φ + large μ → variance dominates
- Extreme counts inflated by factor of 10+

**After**:
- ~0% of φ draws < 5.6 (lower tail cut)
- Variance structure stabilized
- Extreme counts dampened

**Quantitative**: Should reduce 99th percentile counts by ~30%

### Trade-offs

**Pros**:
- Uses empirical information efficiently
- Prevents numerical instability from small φ
- Scientifically justified (φ role unchanged)
- Still substantial uncertainty (SD=15 on mean=35)

**Cons**:
- More informative (less vague)
- Assumes φ similar in AR(1) model (reasonable but assumption)
- If AR(1) changes dispersion, will constrain posterior

**Checks**:
- **Prior-posterior overlap**: If posterior moves far from prior, assumption wrong
- **Sensitivity analysis**: Try original Gamma(2,0.1) if AR(1) fitting succeeds
- **Posterior predictive**: Check if dispersion matches data

**Alternative** (if too informative):
- φ ~ Gamma(3, 0.1) → mean=30, SD=17.3
- Compromise: informed central tendency, wider tails
- Use if Normal(35,15) causes issues

---

## Change 3: Tighten σ Innovation Scale

### What Changed

```
Original:  σ ~ Exponential(2)
           → E[σ] = 0.50, SD = 0.50, 95th %ile = 1.50

Refined:   σ ~ Exponential(5)
           → E[σ] = 0.20, SD = 0.20, 95th %ile = 0.60
```

### Why This Change

**Problem Addressed**: Large innovations in AR(1) process
- ε_t = ρ×ε_{t-1} + ν_t, where ν_t ~ Normal(0, σ)
- Large σ → large shocks → ε can reach ±10
- At late timepoints, accumulated innovations create extreme η

**Stationary SD of AR Process**:
```
SD(ε) = σ / sqrt(1 - ρ²)

With ρ ≈ 0.91:
  Original: E[SD(ε)] = 0.50 / sqrt(1 - 0.91²) ≈ 1.20
            Upper tail: SD(ε) up to ~3.5

  Refined:  E[SD(ε)] = 0.20 / sqrt(1 - 0.91²) ≈ 0.48
            Upper tail: SD(ε) up to ~1.4
```

**Impact on Log-Rate**:
```
η = β₀ + β₁×year + ε

Original: ε can range [-10, +10] in extremes
         At year=1.67: η can reach 4.7 + 1.0×1.67 + 10 = 16.4
         exp(16.4) = 13 million

Refined: ε mostly in [-3, +3], rarely beyond ±5
        At year=1.67: η typically reaches 4.7 + 1.0×1.67 + 3 = 9.4
        exp(9.4) = 12,000
```

**Theoretical Justification**:
- AR(1) innovations should be **small deviations** from trend
- Not large shocks that dominate the linear trend
- If σ comparable to |β₁|, AR process swamps trend
- Ratio: E[σ] / E[|β₁|] should be < 0.5 for stable dynamics

### Expected Impact

**Before**:
- Large σ draws create volatile AR trajectories
- Innovations accumulate over time
- Extreme ε at late timepoints drive tail

**After**:
- Smaller σ constrains AR volatility
- Process stays closer to trend line
- Reduced ε extremes → reduced count extremes

**Quantitative**: Should reduce 99th percentile counts by ~60%

**Combined with β₁ truncation**: Multiplicative effect
- Both η components (trend and AR) constrained
- exp(trend + AR) controlled from both sources

### Trade-offs

**Pros**:
- Maintains theoretical AR(1) structure
- Prevents numerical overflow in exp(η)
- AR process still functional (E[σ]=0.2 is reasonable)
- Forces data to justify large innovations

**Cons**:
- Less flexible for high-volatility dynamics
- If data has large shocks, model will underfit
- Stronger assumption about innovation scale

**Prior Predictive Behavior**:
- Original: ACF(1) mean = NaN (unstable due to extremes)
- Refined: Should see stable ACF(1) ≈ 0.85-0.95
- AR correlation structure preserved

**Sensitivity**:
- If posterior σ piles up at upper tail, prior too tight
- Could relax to Exponential(4) → E[σ]=0.25 if needed

---

## What We Did NOT Change

### β₀ ~ Normal(4.69, 1.0): Kept Original

**Why unchanged**:
- exp(4.69) = 109, close to observed mean (109.4)
- Experiment 1 posterior: β₀ = 4.35 ± 0.04
- Prior covers posterior comfortably
- Not driving tail explosions (additive, not multiplicative)

**Role in extremes**:
- High β₀ (e.g., 6.5 at 97.5th %ile) contributes to large η
- But effect is linear: η = 6.5 + ... vs η = 4.7 + ...
- Difference of ~1.8 on log scale → factor of 6× on count scale
- Reasonable uncertainty, not explosive

**Leave alone**: This prior is working as intended

### ρ ~ Beta(20, 2): Kept Original

**Why unchanged**:
- Strongly motivated by EDA ACF(1) = 0.971
- E[ρ] = 0.909, appropriate for highly correlated data
- Prior predictive showed correct sampling (mean 0.910)
- AR(1) validation issues due to N=40 limitation, not prior

**Theoretical Justification**:
- Temporal correlation is scientifically central
- Strong prior on ρ encodes domain knowledge
- Data will still inform (prior SD = 0.059 is informative but not rigid)

**Not the Problem**:
- High ρ doesn't cause extremes directly
- Creates persistence, not explosions
- Issue was σ (innovation scale), not ρ (persistence)

**Leave alone**: Core temporal structure is sound

---

## Combined Effect of Refinements

### Multiplicative Reduction of Extremes

Three independent constraints → multiplicative improvement:

**Before** (probability of extreme count):
- P(β₁ > 2.0) × P(φ < 8) × P(σ > 1.0) ≈ 0.025 × 0.10 × 0.08 ≈ 0.0002
- With 20,000 samples: ~4 extreme events

**After** (probability of extreme count):
- P(β₁ > 2.0) × P(φ < 5.6) × P(σ > 0.6) ≈ 0.00 × 0.00 × 0.003 ≈ 0.000
- With 20,000 samples: ~0 extreme events

**Expected Improvement**:
- 99th percentile: 143,745 → ~3,000 (**98% reduction**)
- % > 10,000: 3.22% → ~0.1% (**97% reduction**)
- Maximum count: 674 million → ~50,000 (**>99.99% reduction**)

### Preserved Central Behavior

**Median trajectory** (50th percentile):
- Uses median parameters: β₀≈4.7, β₁≈1.0, σ≈0.2, φ≈35
- Original median: 112
- Refined median: ~110 (essentially unchanged)

**Typical counts** (5-95% range):
- Original: [3, 4,503] (at mixed timepoints)
- Refined: [3, 1,200] (same lower, reduced upper)

**Coverage of observed data**:
- Original: median covered, tails too wide
- Refined: median still covers, tails controlled

---

## Scientific Validity of Refinements

### Are We "Cheating" by Using Experiment 1 Information?

**No, for φ**: Valid information transfer
- φ is a data property (dispersion), not model artifact
- Same role in both models
- AR(1) doesn't change count variance mechanism
- Analogous to using sample mean as prior center

**Not using for β₀, β₁**: Appropriately conservative
- These interact with AR(1) structure
- Let data inform trend in presence of correlation
- Don't assume Exp1 values carry over

### Are Truncations "Unscientific"?

**No**: Common and justified
- Truncation at implausible values (e.g., growth > 2500%)
- Prevents numerical issues without constraining inference
- Wide enough that data determines posterior
- Analogous to constraining σ > 0 (always done)

**Example**: β₁ ∈ [-0.5, 2.0]
- Observed: 0.87 (Exp1)
- Posterior will be ~0.8-1.0 if model right
- Bounds at -0.5 and 2.0 are far from action
- Only eliminates impossible extremes

### Will Data Still Inform Posteriors?

**Yes**:
- Priors are tighter but still substantial uncertainty
- β₁: Range of [-0.5, 2.0] is 2.5 units wide, SD=0.5 → 5 SDs span
- φ: SD=15 on mean=35 → 43% coefficient of variation
- σ: Exponential tail still allows up to 1.0 (vs mean 0.2)

**Checks for over-constraint**:
- Posterior should separate from prior
- ESS should be high (>400)
- Prior-posterior overlap should be partial, not total

---

## Expected Prior Predictive Outcomes

### Specific Predictions

**Count Distribution**:
- Mean: ~500 (vs 91,019 original)
- Median: ~110 (vs 112 original, essentially unchanged)
- 95th %ile: ~1,500 (vs 4,503 original)
- 99th %ile: ~3,500 (vs 143,745 original)
- Maximum: ~30,000 (vs 674 million original)

**Plausibility Checks**:
- % > 5,000: ~2% (vs 3.22%+ original)
- % > 10,000: ~0.3% (vs 3.22% original) ✓ PASS
- Maximum per series: ~10,000 (vs 2 million original)

**AR Process**:
- ε range: mostly [-3, +3] (vs [-10, +10] original)
- η range: mostly [2, 8] (vs [0, 15] original)
- Realized ACF(1): mean ~0.85 (vs NaN original)

**Correlation Validation**:
- Cor(ρ, ACF): ~0.4-0.6 (vs 0.39 original, modest improvement)
- Still limited by N=40 short series
- But stable calculation (no NaN from extremes)

### Falsification Criteria

**If refined priors still fail**:

1. **Still > 1% counts > 10,000**:
   - Priors not the issue, structural problem
   - Consider: Drop AR(1), use simpler model
   - OR: Different likelihood (e.g., log-normal)

2. **Numerical instability (NaN, Inf)**:
   - Fundamental computational issue
   - Reparameterize (non-centered AR process)
   - OR: Different software (Stan vs PyMC)

3. **AR validation still poor (Cor < 0.3)**:
   - N=40 insufficient for AR(1) with ρ ≈ 0.9
   - Consider: Simpler temporal structure (random walk)
   - OR: Accept limitation, proceed if other checks pass

**Decision Rule**:
- Pass 6/7 checks, only AR validation marginal → Proceed
- Fail count range checks → Major revision needed
- Numerical issues → Reparameterization or model change

---

## Risks and Mitigations

### Risk 1: Over-Constrained Posteriors

**Risk**: Priors too informative, data can't override

**Indicators**:
- Posterior ≈ prior (no learning)
- Low ESS despite convergence
- Posterior piles up at prior boundaries

**Mitigation**:
- Check prior-posterior overlap quantitatively
- Calculate prior-posterior KL divergence
- If posterior at β₁=2.0 boundary, prior too tight

**Fallback**: Widen priors incrementally
- β₁ truncation: [-0.7, 2.5]
- σ: Exponential(4) instead of Exponential(5)

### Risk 2: Truncation Causes MCMC Issues

**Risk**: Truncated distributions create hard boundaries

**Indicators**:
- Divergent transitions
- Samples at boundaries
- Slow mixing near constraints

**Mitigation**:
- Use non-centered parameterization
- Increase adapt_delta to 0.99
- Monitor trace plots for boundary behavior

**Fallback**: Reparameterize
- Use unconstrained scale, transform (e.g., logit for β₁)
- Soft boundaries with very steep priors instead

### Risk 3: φ Prior Mismatch for AR(1) Model

**Risk**: Dispersion different with AR(1), prior conflicts

**Indicators**:
- Posterior φ far from prior (e.g., φ=10 vs prior mean=35)
- Poor posterior predictive fit for variance
- Large prior-posterior discrepancy

**Mitigation**:
- Check posterior predictive variance structure
- Compare to Experiment 1 observed variance
- If mismatch, φ informative prior was wrong assumption

**Fallback**: Revert to uninformative
- φ ~ Gamma(2, 0.1) original
- Rely on tighter β₁ and σ for stability
- Accept if computational issues don't arise

### Risk 4: Model Still Inadequate

**Risk**: AR(1) not right structure, any priors will struggle

**Indicators**:
- Posterior predictive checks fail
- Residual autocorrelation still high
- Model can't fit data regardless of priors

**Mitigation**:
- Distinguish prior vs structural issues
- If PPC passes but fitting fails → priors
- If PPC passes and fit succeeds but validation fails → structure

**Fallback**: Alternative models
- State-space model with stochastic volatility
- Generalized Additive Model for smoothing
- Changepoint model if temporal pattern non-stationary

---

## Implementation Plan

### Step 1: Prior Predictive Check (Immediate)

**Script**: `/workspace/experiments/experiment_2_refined/prior_predictive_check/code/prior_predictive_check.py`

**Changes from original**:
- Replace prior sampling with refined specifications
- Same 500 simulations, same diagnostics
- Add comparison metrics to original

**Success Criteria**:
- All 7 checks pass (vs 3/7 original)
- Demonstrate >90% reduction in extremes
- Maintain median coverage

**Timeline**: Run now, evaluate results

### Step 2: Decision Point

**If PASS**:
- Document success in findings.md
- Proceed to model fitting (PyMC implementation)
- Update iteration_log.md

**If PARTIAL** (6/7 checks):
- Evaluate which check failed
- Decide if acceptable to proceed
- Document limitations

**If FAIL**:
- Diagnose which refinement insufficient
- Consider further constraints or structural change
- Document in experiment_2_refined/findings.md

### Step 3: Model Fitting (If PPC Passes)

**Software**: PyMC (Stan not available in environment)

**Configuration**:
- 4 chains × 2000 iterations
- tune=1000, draws=1000
- target_accept=0.95 (high, for AR stability)
- max_treedepth=12

**Monitoring**:
- R-hat < 1.01
- ESS > 400
- Divergences < 1%
- Check posteriors vs priors (separation)

### Step 4: Validation (If Fitting Succeeds)

**Posterior Predictive Checks**:
- Residual autocorrelation (should be reduced vs Exp1)
- Count distribution fit
- Temporal coverage
- Influential observations

**Model Comparison**:
- LOO-CV vs Experiment 1
- Expected: ELPD improvement if AR(1) helps
- If no improvement: temporal correlation not needed

---

## Relationship to Iteration Log

This refinement is **Iteration 1** of the AR(1) model:
- **Experiment 2 (v1)**: Original priors → FAILED PPC
- **Experiment 2 Refined (v2)**: This version → PPC pending

**If successful**: Becomes Experiment 2 final
**If fails**: Iteration 2 or model redesign

**Documentation in iteration_log.md**:
```markdown
## Iteration 1: AR(1) Model Prior Refinement

**From**: Experiment 2 v1 (failed PPC)
**To**: Experiment 2 Refined (v2)
**Date**: 2025-10-29

**Issues Addressed**:
- 3.22% extreme outliers (>10,000 counts)
- Maximum count 674 million
- Tail explosions from wide priors + exponential link

**Changes Made**:
1. β₁ ~ TruncatedNormal(1.0, 0.5, -0.5, 2.0)
2. φ ~ Normal(35, 15), informed from Exp1
3. σ ~ Exponential(5), tighter innovations

**Expected Improvement**: >90% reduction in extremes
**Status**: Prior predictive check pending
```

---

## Success Metrics

### Prior Predictive Check Success

**Must achieve**:
- [ ] <1% counts > 10,000
- [ ] <5% counts > 5,000
- [ ] Maximum count < 100,000
- [ ] No NaN/Inf values
- [ ] Median ~100-200 (covers observed)

**Should achieve**:
- [ ] 99th percentile < 5,000
- [ ] Mean realized ACF(1) in [0.7, 0.95]
- [ ] Correlation(ρ, ACF) > 0.3

### Model Fitting Success (Conditional on PPC Pass)

**Convergence**:
- [ ] R-hat < 1.01 all parameters
- [ ] ESS > 400 all parameters
- [ ] < 1% divergences
- [ ] < 5% max treedepth warnings

**Posterior Quality**:
- [ ] Posteriors separate from priors (data informative)
- [ ] No pileup at truncation boundaries
- [ ] Posterior ρ > 0.3 (temporal correlation detected)

### Model Validation Success (Conditional on Fitting)

**Posterior Predictive**:
- [ ] Residual ACF(1) < 0.3 (vs 0.511 in Exp1)
- [ ] Count distribution matches data
- [ ] No systematic bias in predictions

**Model Comparison**:
- [ ] ELPD_loo > -170.05 (Exp1 baseline)
- [ ] p_loo ≈ 5 (reasonable for 5 parameters)
- [ ] No bad Pareto k values

---

## Conclusion

This refinement applies **targeted, scientifically justified constraints** to fix prior predictive failures while maintaining model flexibility and theoretical structure. The changes are:

1. **Minimal**: Only three parameters modified
2. **Justified**: Each by specific diagnostic evidence
3. **Informative but not rigid**: Substantial uncertainty remains
4. **Falsifiable**: Clear success criteria

If this refined model passes prior predictive checks, it represents a well-specified AR(1) model ready for inference. If it fails, we have clear decision criteria for next steps (simplification vs reparameterization).

The refinement exemplifies **iterative model building**: use diagnostics to identify issues, apply targeted fixes, validate, repeat. This is how robust Bayesian workflows should operate.

---

**Author**: Model Refinement Agent
**Date**: 2025-10-29
**Status**: Ready for Prior Predictive Validation
**Next Step**: Execute `/workspace/experiments/experiment_2_refined/prior_predictive_check/code/prior_predictive_check.py`
