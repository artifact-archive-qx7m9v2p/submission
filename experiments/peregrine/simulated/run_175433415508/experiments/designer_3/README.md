# Designer 3: Non-Linear and Hierarchical Bayesian Models

**Designer Role**: Complexity and structural change specialist
**Focus**: Quadratic growth, changepoints, non-parametric flexibility
**Philosophy**: Extraordinary claims require extraordinary evidence

---

## Quick Start

**Main Documents**:
1. `experiment_plan.md` - Executive summary for synthesis agent (START HERE)
2. `proposed_models.md` - Detailed mathematical specifications (COMPREHENSIVE)

**Stan Programs**:
- `model_1_quadratic.stan` - Polynomial growth with AR(1)
- `model_2_changepoint.stan` - Regime switching with unknown break location
- `model_3_gp.stan` - Gaussian process for non-parametric flexibility

---

## Models Overview

### Model 1: Quadratic NB + AR(1)
- **Rationale**: EDA shows R²=0.964 for quadratic vs 0.937 for exponential
- **Key Parameter**: β₂ (curvature)
- **Falsification**: If β₂ ≈ 0 OR ΔLOO < 4 SE → Use linear model
- **Runtime**: 2-4 minutes
- **Risk**: Overfitting with n=40

### Model 2: Bayesian Changepoint + AR(1)
- **Rationale**: EDA detected break at year=-0.21, 9.6× growth rate change
- **Key Parameters**: τ (changepoint location), β₂ (slope change)
- **Falsification**: If τ posterior uniform OR β₂ ≈ 0 → No structural break
- **Runtime**: 4-8 minutes
- **Risk**: Spurious break detection, computational difficulty

### Model 3: Gaussian Process NB
- **Rationale**: Let data determine functional form without assumptions
- **Key Parameters**: ℓ (length scale), α (amplitude)
- **Falsification**: If ℓ→0 (noise) OR ℓ→∞ (linear) OR Cholesky fails
- **Runtime**: 10-20 minutes
- **Risk**: n=40 is small for GP, may be intractable

---

## Critical Decision Logic

```
START → Fit Linear Baseline
         ↓
      Fit Model 1 (Quadratic)
         ↓
      β₂ ≈ 0? → YES → STOP, use linear
         ↓ NO
      ΔLOO > 4 SE? → NO → STOP, use linear
         ↓ YES
      Fit Model 2 (Changepoint)
         ↓
      τ uniform? → YES → STOP, no break
         ↓ NO
      β₂ ≈ 0? → YES → STOP, no slope change
         ↓ NO
      ΔLOO > 6 SE vs linear? → NO → Use best so far
         ↓ YES
      Systematic PPC failures? → NO → DONE, report best model
         ↓ YES
      Fit Model 3 (GP) [stress test]
         ↓
      Computational success? → NO → STOP
         ↓ YES
      ΔLOO > 10 SE vs best? → NO → Use best parametric
         ↓ YES
      Report GP model
```

---

## Falsification Criteria Summary

### Model 1 Rejection Conditions
1. β₂ credible interval contains zero
2. ΔLOO < 4 SE vs linear baseline
3. β₁, β₂ correlation > 0.9 (non-identifiable)
4. Divergent transitions despite reparameterization
5. Posterior predictive checks show systematic bias

### Model 2 Rejection Conditions
1. τ posterior is uniform (no information)
2. β₂ credible interval contains zero
3. ΔLOO < 6 SE vs linear
4. Posterior P(τ at edge) > 0.3
5. Multiple modes in τ posterior

### Model 3 Rejection Conditions
1. Length scale ℓ < 0.3 (data too noisy)
2. Length scale ℓ > 3 (reduces to linear)
3. Cholesky decomposition failures
4. ΔLOO < 10 SE vs best parametric
5. >5% divergences

---

## Expected Outcomes

### Scenario 1: Linear Model Wins (Most Likely)
**Probability**: 60%

The 2.7% R² improvement (0.964 vs 0.937) doesn't survive cross-validation with n=40. All complex models fail LOO comparison.

**Conclusion**: Report linear Negative Binomial with AR(1) as best model.
**Status**: SUCCESS - we learned data are simpler than EDA suggested.

### Scenario 2: Quadratic Model Wins
**Probability**: 30%

β₂ significantly different from zero AND ΔLOO > 4 SE. Acceleration in growth is genuine, not overfitting.

**Conclusion**: Report quadratic model.
**Status**: SUCCESS - non-linearity justified.

### Scenario 3: Changepoint Model Wins
**Probability**: 8%

τ posterior concentrated around specific value, β₂ clearly non-zero, ΔLOO > 6 SE.

**Conclusion**: Report regime shift model.
**Status**: SUCCESS - structural break validated.

### Scenario 4: GP Model Wins
**Probability**: 2%

Parametric forms systematically fail, GP provides dramatically better fit despite complexity.

**Conclusion**: Report GP model with caution about n=40.
**Status**: SUCCESS but surprising - suggests complex dynamics.

---

## Integration with Other Designers

### Designer 1 (Baseline Models)
- **Overlap**: Both propose linear Negative Binomial baseline
- **Differentiation**: Designer 3 adds polynomial/changepoint/GP extensions
- **Comparison**: Designer 3 models must beat Designer 1 baseline by LOO

### Designer 2 (Time Series Specialist)
- **Overlap**: Both use AR(1) temporal structure
- **Differentiation**: Designer 3 focuses on non-linear mean function
- **Comparison**: May propose hierarchical variance models vs Designer 2's time-varying dispersion

### Synthesis Strategy
- Use Designer 1's linear model as baseline for all
- Compare Designer 2's temporal vs Designer 3's non-linear approaches
- Ensemble if multiple models within 2 SE on LOO

---

## Model Selection Priorities

**Rank by**:
1. **Falsification**: Does model pass basic sanity checks?
2. **LOO**: Out-of-sample predictive performance
3. **Parsimony**: Prefer simpler if LOO equivalent (within 2 SE)
4. **Interpretability**: Can we explain to domain experts?
5. **Computational cost**: Practical feasibility

**Philosophy**:
- Simpler models are preferred by default
- Complex models bear burden of proof
- ΔLOO must exceed conservative thresholds:
  - 1 extra parameter: >4 SE
  - 2 extra parameters: >6 SE
  - Many parameters (GP): >10 SE

---

## Computational Specifications

### Hardware Requirements
- **Memory**: 4GB sufficient for Models 1-2, 8GB for Model 3
- **CPU**: 4 cores recommended (parallel chains)
- **Time**: Total 20-30 minutes for all models

### Software Stack
- Stan 2.33+ or CmdStanPy 1.2+
- Python 3.9+ with numpy, pandas, matplotlib
- ArviZ for diagnostics
- LOO package for cross-validation

### Sampling Configuration
```python
# Standard configuration
chains = 4
iter_warmup = 1000
iter_sampling = 1000
adapt_delta = 0.9  # increase to 0.95-0.99 if divergences
max_treedepth = 10
```

---

## Quality Assurance Checklist

### Pre-Flight (Before Fitting)
- [ ] Prior predictive checks show reasonable simulations
- [ ] Stan programs compile without errors
- [ ] Data loaded correctly (n=40, no NaNs)

### During Fitting
- [ ] Monitor R-hat convergence in real-time
- [ ] Check for divergences (expect some in Models 2-3)
- [ ] Verify ESS > 400 for key parameters

### Post-Fitting
- [ ] All R-hat < 1.01 (or 1.02 for changepoint)
- [ ] No post-warmup divergences (or <1%)
- [ ] Posterior predictive checks: p-values ∈ [0.05, 0.95]
- [ ] LOO Pareto k < 0.7 for >90% of observations

### Model Comparison
- [ ] Compute ΔLOO with standard errors
- [ ] Apply parsimony rules (2 SE threshold)
- [ ] Check for prior-posterior conflict
- [ ] Verify stress tests passed

---

## Common Issues and Solutions

### Issue 1: Model 1 - Divergences
**Cause**: β₁, β₂ correlation high (identifiability issue)
**Solution**: Increase adapt_delta to 0.95, check correlation
**Abandon if**: Persists at adapt_delta=0.99

### Issue 2: Model 2 - Poor ESS for τ
**Cause**: Discrete changepoint creates rugged posterior
**Solution**: Increase iterations to 3000, ensure multiple modes ruled out
**Abandon if**: ESS < 50 even with 5000 iterations

### Issue 3: Model 3 - Cholesky Fails
**Cause**: Ill-conditioned covariance (length scale too small)
**Solution**: Check ℓ prior, add larger jitter (1e-6)
**Abandon if**: Failures persist (signals GP inappropriate)

### Issue 4: All Models - Poor LOO
**Cause**: Data genuinely simple, linear model sufficient
**Solution**: Report linear baseline as best model
**Status**: SUCCESS - this is valuable information!

---

## Reporting Template

### If Complex Model Succeeds
```
Model [ID] selected as best.

Evidence:
- ΔLOO = [value] ± [SE] vs linear baseline
- Key parameters: [estimates with 90% CI]
- Posterior predictive checks: [p-values]
- Interpretation: [plain language]

Falsification:
- Tested against [N] alternative models
- Survived [N] stress tests
- No red flags detected

Limitations:
- [specific caveats]
- [extrapolation cautions]
```

### If Linear Model Wins
```
Linear Negative Binomial with AR(1) selected as best.

Evidence:
- All complex models: ΔLOO < 4 SE vs baseline
- Quadratic β₂: [estimate] with CI containing zero
- Changepoint τ: posterior uniform distribution
- GP: [computational failure / ℓ→∞]

Interpretation:
EDA findings of non-linearity were artifacts of small sample size (n=40).
Occam's razor favors linear model.

This is SUCCESS: We avoided overfitting and found the truth.
```

---

## References

**EDA Report**: `/workspace/eda/eda_report.md`
- Key Finding 1: Quadratic R²=0.964 vs exponential R²=0.937
- Key Finding 2: Possible changepoint at year=-0.21
- Key Finding 3: Severe overdispersion (Var/Mean=70.43)
- Key Finding 4: High autocorrelation (ACF=0.971)

**Related Work**:
- Designer 1: Linear and baseline models
- Designer 2: Time series and autocorrelation focus

---

## Contact and Questions

**For Synthesis Agent**:
- Start with `experiment_plan.md` for executive summary
- Consult `proposed_models.md` for mathematical details
- Use Stan programs as-is or modify for specific needs

**Key Philosophy**:
> "The goal is finding truth, not completing tasks. If all complex models fail, that's success—we learned the data are simpler than we thought. Abandon complexity unless data demand it."

---

**Designer**: Model Designer 3
**Status**: Complete and ready for implementation
**Last Updated**: 2025-10-29
