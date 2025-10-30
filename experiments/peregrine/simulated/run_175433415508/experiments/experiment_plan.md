# Bayesian Model Experiment Plan

## Overview

This document synthesizes proposals from three independent model designers into a unified, prioritized experiment plan. The plan balances theoretical justification, computational feasibility, and falsification rigor.

**Dataset**: n=40, time series with count response C (range 21-269) and standardized year predictor

**EDA Key Findings**:
- Severe overdispersion: Var/Mean = 70.43
- Strong growth: R² = 0.937 (exponential), 0.964 (quadratic)
- High autocorrelation: ACF(1) = 0.971
- Possible structural break at year = -0.21

---

## Design Philosophy

### Falsification-First Approach
- **Every model includes explicit rejection criteria**
- **Simpler models are default** - complexity must justify itself via LOO with ΔELPD > 2×SE per parameter
- **"Model failed" is a valid and informative outcome**
- **Linear baseline wins if no complex model beats it decisively**

### Sequential Testing Strategy
1. Start with simplest baseline (independent errors)
2. Add temporal correlation only if needed
3. Add non-linearity only if justified
4. Stop early if simpler model adequate

### Minimum Attempt Policy
- **Must attempt at least first TWO models** unless Model 1 fails pre-fit validation
- Document reason in log.md if fewer than two attempted
- May proceed to Phase 4 after minimum attempts complete

---

## Synthesized Model Catalog

### Summary Table

| Model | Source | Parameters | Key Innovation | Risk | Priority |
|-------|--------|------------|----------------|------|----------|
| **M1: NB-Linear** | D1 | 3 | Baseline (no correlation) | Low | **1** ⭐ |
| **M2: NB-AR1** | D2 | 5 | Temporal correlation | Med | **2** ⭐ |
| **M3: NB-Quad** | D1, D3 | 4 | Quadratic growth | Med | 3 |
| **M4: NB-Quad-AR1** | D3 | 6 | Quad + correlation | High | 4 |
| **M5: NB-Changepoint** | D3 | 6-7 | Regime switching | High | 5 |
| **M6: NB-GP** | D2, D3 | N+3 | Non-parametric | Very High | 6 |
| **M7: NB-RW** | D2 | 4 | Random walk | Med | 7 |

*D1=Designer 1 (Baseline), D2=Designer 2 (Temporal), D3=Designer 3 (Non-linear)*

---

## Prioritized Experiment Sequence

### **Experiment 1: Negative Binomial Linear (NB-Linear)** ⭐⭐⭐
**Status**: MANDATORY BASELINE

#### Specification
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁×year_t

Priors:
  β₀ ~ Normal(4.69, 1.0)    # log(109.4) with wide uncertainty
  β₁ ~ Normal(1.0, 0.5)      # Positive growth
  φ ~ Gamma(2, 0.1)          # Overdispersion (mean=20)
```

#### Theoretical Justification
- **Overdispersion**: Var/Mean = 70.43 requires NB distribution
- **Exponential growth**: Log-linear R² = 0.937 from EDA
- **Simplest adequate model**: Occam's razor starting point
- **Diagnostic baseline**: Isolates what temporal correlation must explain

#### Prior Predictive Range
- At year=0: median μ ≈ 109 (matches EDA mean)
- At year=-1.67: μ ≈ 36 (observed min=21) ✓
- At year=+1.67: μ ≈ 320 (observed max=269) ✓

#### Falsification Criteria
**REJECT if**:
- Convergence failure (R-hat > 1.01, ESS < 400)
- φ posterior > 100 (contradicts overdispersion finding)
- Posterior predictive checks fail systematically (Bayesian p < 0.01)
- LOO worse than naive mean model

**ACCEPT if**:
- Convergence successful
- Captures mean trend and dispersion
- **Residual ACF > 0.8 is EXPECTED and OK** (justifies Experiment 2)

#### Expected Outcome (80% confidence)
- β₀ ≈ 4.5 to 4.9
- β₁ ≈ 0.7 to 1.2
- φ ≈ 10 to 40
- Residual ACF ≈ 0.7-0.9 (high, as expected without correlation structure)
- **Decision**: ACCEPT as baseline, proceed to Experiment 2

#### Computational Cost
- Runtime: 30-60 seconds (4 chains × 2000 iterations)
- Risk: LOW - standard model, well-tested

#### Implementation Notes
- Use Stan's `neg_binomial_2(mu, phi)` parameterization
- Include `log_lik` in generated quantities for LOO
- Standard centered parameterization (no technical issues expected)

---

### **Experiment 2: Negative Binomial AR(1) (NB-AR1)** ⭐⭐⭐
**Status**: MANDATORY - Tests temporal correlation hypothesis

#### Specification
```
C_t ~ NegativeBinomial(exp(η_t), φ)
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t
ν_t ~ Normal(0, σ)

Priors:
  β₀ ~ Normal(4.69, 1.0)
  β₁ ~ Normal(1.0, 0.5)
  φ ~ Gamma(2, 0.1)
  ρ ~ Beta(20, 2)           # E[ρ] = 0.91 (based on ACF=0.971)
  σ ~ Exponential(2)
```

#### Theoretical Justification
- **High ACF = 0.971**: Strong evidence for temporal dependence
- **Separates trend from correlation**: Linear trend + AR(1) errors
- **Standard approach**: Well-established for count time series
- **Critical test**: Is correlation genuine or trend artifact?

#### Prior Predictive Range
- Similar to Experiment 1 for mean structure
- Autocorrelation: E[ρ] = 0.91, allowing range [0.7, 0.99]
- Innovation σ: Allows modest deviations from trend

#### Falsification Criteria
**REJECT if**:
- Convergence issues (divergences, R-hat > 1.01)
- ρ posterior: P(ρ < 0.3) > 0.95 (correlation is trend artifact)
- Residual ACF still > 0.7 (model inadequate)
- ΔLOO(M2 - M1) < 2×SE (correlation doesn't improve fit)

**WARNING if**:
- ρ → 1.0 (boundary, suggests random walk or missing trend)
- Wide posterior for ρ (data insufficient to estimate)

**ACCEPT if**:
- ρ posterior clearly > 0.5 with narrow interval
- Residual ACF < 0.3 (correlation captured)
- ΔLOO > 5 (substantial improvement over M1)

#### Expected Outcome (70% confidence)
- ρ ≈ 0.7-0.95 (high but < observed ACF)
- Residual ACF ≈ 0.2-0.5 (reduced but may not be zero)
- ΔLOO ≈ 5-15 (moderate to strong improvement)
- **Decision**: If ACCEPT, use as new baseline for comparisons

#### Alternative Outcomes
1. **ρ ≈ 0** (20% probability): Correlation is trend artifact → M1 wins, consider M3 (quadratic)
2. **ρ → 1** (10% probability): Non-stationary → Try M7 (random walk)

#### Computational Cost
- Runtime: 2-5 minutes (AR(1) more expensive)
- Risk: MEDIUM - potential convergence issues near ρ=1

#### Implementation Notes
- Use **non-centered parameterization**: ε_t = ρ×ε_{t-1} + σ×ν_t, ν_t ~ Normal(0,1)
- Careful initialization: avoid ρ=1 boundary
- Monitor divergences (may indicate identifiability issues)

---

### **Experiment 3: Negative Binomial Quadratic (NB-Quad)**
**Status**: CONDITIONAL - Only if M1 shows quadratic residual pattern

#### Specification
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁×year_t + β₂×year_t²

Priors:
  β₀ ~ Normal(4.69, 1.0)
  β₁ ~ Normal(1.0, 0.5)
  β₂ ~ Normal(0, 0.3)       # Skeptical prior (centered at 0)
  φ ~ Gamma(2, 0.1)
```

#### Theoretical Justification
- **EDA finding**: Quadratic R² = 0.964 vs exponential 0.937 (2.7% improvement)
- **Growth rate change**: Increases 9.6-fold across time period
- **Visual evidence**: Apparent acceleration in scatter plot
- **Skepticism**: Small n=40, may be overfitting

#### Falsification Criteria
**REJECT if**:
- β₂ credible interval includes zero (95% CI)
- ΔLOO(M3 - M1) < 4×SE (not worth extra parameter)
- Pareto k > 0.7 for > 20% observations (overfitting)

**ACCEPT if**:
- β₂ posterior clearly excludes zero
- ΔLOO > 6 (strong improvement)
- Residual diagnostics better than M1

#### Expected Outcome (50% confidence in REJECT)
- **Most likely**: β₂ ≈ 0 with wide CI → REJECT, M1 sufficient
- **If ACCEPT**: β₂ ≈ -0.2 to 0.2 (modest curvature)

#### When to Run
- **Run if**: M1 residuals show systematic quadratic pattern
- **Skip if**: M2 (AR1) explains curvature via correlation
- **Priority**: After M1 and M2 complete

#### Computational Cost
- Runtime: 45-90 seconds
- Risk: LOW

---

### **Experiment 4: Negative Binomial Quadratic + AR(1) (NB-Quad-AR1)**
**Status**: CONDITIONAL - Only if both M2 and M3 accepted

#### Specification
```
C_t ~ NegativeBinomial(exp(η_t), φ)
η_t = β₀ + β₁×year_t + β₂×year_t² + ε_t
ε_t = ρ×ε_{t-1} + ν_t

Priors: Combine M2 and M3 priors
```

#### Theoretical Justification
- **Combining innovations**: Both curvature and correlation may be real
- **Comprehensive model**: Most flexible of parametric options
- **Risk**: Overparameterization with n=40

#### Falsification Criteria
**REJECT if**:
- Either β₂≈0 or ρ≈0 (one feature unnecessary)
- ΔLOO(M4 - M2) < 4×SE (quadratic not needed given AR1)
- ΔLOO(M4 - M3) < 4×SE (AR1 not needed given quadratic)
- Convergence issues (identifiability problems)

**ACCEPT if**:
- Both β₂ and ρ clearly non-zero
- ΔLOO > 8 compared to both M2 and M3
- No identifiability issues

#### Expected Outcome (60% confidence in REJECT)
- **Most likely**: One parameter (β₂ or ρ) absorbs the other → Simpler model wins
- **Identifiability concern**: Curvature vs autocorrelation confounding

#### When to Run
- **Only if**: Both M2 and M3 individually accepted
- **Priority**: Low (parsimony argues against)

#### Computational Cost
- Runtime: 5-10 minutes
- Risk: HIGH (identifiability, convergence)

---

### **Experiment 5: Bayesian Changepoint Model (NB-Changepoint)**
**Status**: EXPLORATORY - Only if strong evidence for regime shift

#### Specification
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁×year_t + β₂×I(year_t > τ)×year_t

Priors:
  β₀ ~ Normal(4.69, 1.0)
  β₁ ~ Normal(0.5, 0.5)     # Early period growth
  β₂ ~ Normal(0, 1.0)        # Change in slope
  τ ~ Normal(-0.21, 0.3)     # EDA suggested location
  φ ~ Gamma(2, 0.1)
```

#### Theoretical Justification
- **EDA finding**: Changepoint detected at year = -0.21 (Designer 2)
- **Growth change**: 9.6-fold increase in growth rate
- **Mechanistic plausibility**: Regime shift in underlying process
- **Skepticism**: Single analyst finding, needs validation

#### Falsification Criteria
**REJECT if**:
- τ posterior is uniform (no identifiable changepoint)
- β₂ credible interval includes zero (no slope change)
- ΔLOO(M5 - best simpler model) < 6×SE
- τ estimate highly sensitive to prior

**WARNING if**:
- Multiple local modes in posterior (uncertainty in τ location)
- Wide posterior for τ (data can't pinpoint break)

#### When to Run
- **Only if**: Residual diagnostics show clear regime change
- **Priority**: Low (exploratory, high complexity)

#### Computational Cost
- Runtime: 10-20 minutes (changepoint estimation expensive)
- Risk: VERY HIGH (identifiability, multiple modes)

#### Implementation Notes
- May need marginal likelihood approach or RJMCMC
- Consider fixed τ = -0.21 first (simpler)
- Non-centered parameterization essential

---

### **Experiment 6: Gaussian Process Model (NB-GP)**
**Status**: STRESS TEST - Only if parametric forms systematically fail

#### Specification
```
C_t ~ NegativeBinomial(exp(f(year_t)), φ)
f ~ GP(0, K)
K(year_i, year_j) = α² × exp(-(year_i - year_j)² / (2ℓ²))

Priors:
  α ~ Normal(1, 1)           # GP amplitude
  ℓ ~ InvGamma(5, 5)         # Length scale (E[ℓ] ≈ 1.25)
  φ ~ Gamma(2, 0.1)
```

#### Theoretical Justification
- **Non-parametric flexibility**: No assumed functional form
- **Smooth interpolation**: Captures any smooth trend
- **Stress test**: If GP doesn't beat parametric, we didn't miss anything

#### Falsification Criteria
**REJECT if** (EXPECT to reject):
- ΔLOO(M6 - simpler models) < 10×SE (not worth N extra parameters)
- ℓ → 0 (white noise, no structure)
- ℓ → ∞ (constant, no variation)
- Computational failure (Cholesky decomposition issues)

**ACCEPT if** (UNLIKELY with n=40):
- Dramatic improvement (ΔLOO > 20)
- Posterior predictive checks substantially better
- Reveals clear missing structure in parametric models

#### Expected Outcome (90% confidence in REJECT)
- **Most likely**: Parametric models sufficient, GP overparameterized
- **Alternative**: Computational failure due to n=40

#### When to Run
- **Only if**: All parametric models show systematic misfit
- **Priority**: Very Low (stress test only)

#### Computational Cost
- Runtime: 5-15 minutes (if converges)
- Risk: VERY HIGH (computational, overparameterization)

---

### **Experiment 7: Random Walk Model (NB-RW)**
**Status**: DIAGNOSTIC - Only if ρ → 1 in Experiment 2

#### Specification
```
C_t ~ NegativeBinomial(exp(η_t), φ)
η_t = η_{t-1} + β₁ + ν_t
ν_t ~ Normal(0, σ)

Priors:
  η_1 ~ Normal(log(30), 1)   # Initial value
  β₁ ~ Normal(0.05, 0.05)     # Drift term
  σ ~ Exponential(10)         # Innovation SD
  φ ~ Gamma(2, 0.1)
```

#### Theoretical Justification
- **Diagnostic purpose**: If ρ → 1, process may be non-stationary
- **Random walk + drift**: Allows persistent shocks
- **Alternative parameterization**: May fit better than AR(1) if unit root

#### Falsification Criteria
**REJECT if**:
- σ → 0 (deterministic trend, not stochastic)
- ΔLOO(M7 - M2) < 2×SE (AR1 sufficient)
- First-differenced residuals still show strong ACF

**ACCEPT if**:
- M2 showed ρ posterior concentrated near 1
- First-differenced residuals uncorrelated
- ΔLOO > 4

#### When to Run
- **Only if**: M2 diagnostic shows ρ → 1
- **Priority**: Conditional (diagnostic fallback)

---

## Model Comparison Strategy

### LOO-Based Decision Framework

#### Comparing Two Models
- **Decisive (> 4×SE)**: Choose better model with confidence
- **Strong (2-4×SE)**: Prefer better model unless parsimony argues otherwise
- **Weak (< 2×SE)**: Models indistinguishable, prefer simpler (Occam's razor)

#### Multiple Models
1. Compute LOO-ELPD for all converged models
2. Use `az.compare()` for ranking and SE estimates
3. Apply parsimony rule: Δ parameters → require Δ (4×SE per parameter)

#### Tie-Breaking Rules
If LOO indecisive:
1. **Posterior predictive checks**: Better qualitative fit wins
2. **Residual diagnostics**: Lower residual ACF wins
3. **Interpretability**: Simpler parameter interpretation wins
4. **Robustness**: Narrower posterior intervals win
5. **Parsimony**: Fewer parameters win (default)

---

## Validation Pipeline (For Each Model)

### Phase 3a: Prior Predictive Check
**Goal**: Validate priors generate plausible data

1. Sample 100 datasets from prior only
2. Check plausible range: counts in [1, 1000]?
3. Check growth patterns: reasonable trend shapes?
4. **FAIL if**: Extreme outliers, negative counts, absurd patterns
5. **PASS to next phase** if priors reasonable

### Phase 3b: Simulation-Based Calibration
**Goal**: Can model recover known parameters?

1. Generate data from known parameters
2. Fit model, check coverage of credible intervals
3. Repeat 100 simulations
4. **FAIL if**: < 80% coverage or systematic bias
5. **PASS to next phase** if inference valid

### Phase 3c: Model Fitting
**Goal**: Obtain posterior samples

1. Compile Stan model
2. Run 4 chains × 2000 iterations (1000 warmup)
3. Check convergence: R-hat < 1.01, ESS > 400
4. Check divergences: < 1% divergent transitions
5. **FAIL if**: Convergence issues persist after reparameterization
6. **PASS to next phase** if converged

### Phase 3d: Posterior Predictive Check
**Goal**: Does model fit observed data?

1. Generate replicated datasets from posterior
2. Compare to observed: counts, mean, variance, ACF
3. Compute Bayesian p-values
4. Visual checks: rootograms, residual plots
5. **WARNING if**: Systematic misfit (p < 0.05 or > 0.95)
6. **Continue regardless** (document fit quality)

### Phase 3e: Model Critique
**Goal**: Accept, revise, or reject?

1. Apply falsification criteria
2. Check LOO diagnostics (Pareto k)
3. Assess practical adequacy
4. **Decision**: ACCEPT, REVISE, or REJECT
5. Document decision rationale

---

## Expected Timeline & Resource Allocation

### Phase 1: Prior Predictive Checks (Day 1, 2-3 hours)
- M1: 30 min
- M2: 45 min
- Conditional models: 1-2 hours

### Phase 2: Simulation-Based Calibration (Day 1-2, 4-6 hours)
- M1: 1 hour
- M2: 2 hours
- Conditional models: 2-3 hours

### Phase 3: Model Fitting (Day 2-3, 4-8 hours)
- M1: 1 hour
- M2: 2-3 hours
- Conditional models: 2-4 hours

### Phase 4: Posterior Checks & Critique (Day 3-4, 3-5 hours)
- Per model: 1-1.5 hours

### Phase 5: Model Comparison (Day 4, 2-3 hours)
- LOO computation and comparison
- Final model selection
- Documentation

**Total estimated time**: 15-25 hours over 4 days

---

## Iteration Strategy

### When to Refine vs Switch vs Stop

#### REFINE a model if:
- Convergence issues but fixable (reparameterization)
- Prior-posterior conflict with clear resolution
- Missing predictor identified (e.g., year²)
- Slight misspecification with obvious fix

#### SWITCH model class if:
- Fundamental misspecification (e.g., ρ → 1 suggests RW not AR1)
- Multiple refinements fail
- Prior-data conflict unresolvable
- Computational issues persist

#### STOP iteration if:
- Adequate model found (passes all checks, reasonable LOO)
- Diminishing returns (ΔLOO < 2×SE after multiple attempts)
- Data quality issues discovered (unfixable)
- Computational limits reached (complex models infeasible with n=40)

---

## Success Criteria & Adequacy Assessment

### Minimal Adequate Model Must Have:
1. ✓ R-hat < 1.01 for all parameters
2. ✓ ESS > 400 (bulk and tail)
3. ✓ < 5% divergent transitions
4. ✓ Pareto k < 0.7 for > 80% observations
5. ✓ Posterior predictive checks reasonable (p ∈ [0.05, 0.95])
6. ✓ Residual ACF < 0.5 (if claiming to model correlation)
7. ✓ 90% posterior intervals have ~90% coverage (calibration)

### Phase 5: Adequacy Assessment Outcomes

#### ADEQUATE → Proceed to Final Report
- Best model meets all criteria
- Answers scientific questions
- Uncertainty quantified

#### CONTINUE → Refine Models
- Fixable issues identified
- Clear improvement path
- Resources available

#### STOP → Report Limitations
- No model adequate despite attempts
- Fundamental data limitations (n=40)
- Simpler approach recommended
- **Remain within Bayesian paradigm**

---

## Risk Assessment & Mitigation

### High Risk Areas

1. **Temporal correlation identifiability** (M2, M4)
   - Risk: ρ vs trend confounding
   - Mitigation: Compare with and without correlation, test on differenced data

2. **Quadratic overfitting** (M3, M4)
   - Risk: Capturing noise with n=40
   - Mitigation: Skeptical priors, strict LOO threshold (ΔLOO > 4×SE)

3. **Changepoint estimation** (M5)
   - Risk: Multiple modes, non-identifiability
   - Mitigation: Try fixed τ first, use informative prior from EDA

4. **Computational expense** (M6, M4)
   - Risk: GP and complex models may not converge in reasonable time
   - Mitigation: Timeout after 30 min, skip if infeasible

### Contingency Plans

**If all models fail convergence:**
- Simplify to Quasi-Poisson (non-Bayesian baseline for comparison)
- Document computational barriers
- Recommend hierarchical structure or different prior family

**If LOO unstable (high Pareto k):**
- Use K-fold cross-validation (K=5 or K=10)
- Report uncertainty in model comparison
- Consider model averaging

**If < 2 models attempted:**
- Document reason in log.md (per Minimum Attempt Policy)
- Ensure M1 failed for valid reason (not just skipped)
- Proceed to Phase 4 with available models

---

## Summary: Experiment Priority Ranking

### MUST RUN (Minimum 2)
1. ⭐⭐⭐ **Experiment 1 (NB-Linear)**: Baseline, always run first
2. ⭐⭐⭐ **Experiment 2 (NB-AR1)**: Test main hypothesis (ACF=0.971)

### SHOULD RUN (High Value)
3. ⭐⭐ **Experiment 3 (NB-Quad)**: If residuals show curvature
4. ⭐ **Experiment 7 (NB-RW)**: If ρ → 1 in Experiment 2

### MAY RUN (Conditional)
5. **Experiment 4 (NB-Quad-AR1)**: Only if both M2 and M3 accepted
6. **Experiment 5 (NB-Changepoint)**: Only if strong regime shift evidence

### RARELY RUN (Stress Test)
7. **Experiment 6 (NB-GP)**: Only if all parametric models fail

---

## Final Notes

### Philosophy Reminder
- **Failure is information**: A rejected model teaches us about the data
- **Simplicity is virtue**: Prefer simpler model when LOO indecisive
- **Transparency is essential**: Document all decisions, including failures
- **Bayesian constraint**: Final model MUST be Bayesian (no pivoting to frequentist)

### Expected Outcome (Prediction)
**Most likely scenario** (60% confidence):
- M1 (NB-Linear) captures trend and dispersion
- M2 (NB-AR1) adds temporal correlation, improves LOO by 5-10
- M2 accepted as final model
- Quadratic and other complex models rejected (not worth complexity)

**Alternative scenarios**:
- M1 sufficient (ACF was trend artifact): 20%
- M4 needed (both curvature and correlation): 15%
- M7 or other (non-stationary process): 5%

---

**Document Version**: 1.0
**Date**: 2025
**Status**: Ready for implementation
**Next Step**: Begin Experiment 1 - Prior Predictive Check
