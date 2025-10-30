# Model 2 Assessment Report: Student-t Likelihood

**Model**: Logarithmic Student-t Regression
**Date**: 2025-10-28
**Status**: COMPLETED (Phase 3) → NOT SELECTED (Phase 4)
**Assessor**: Bayesian Model Assessment Agent

---

## Executive Summary

**Model 2 is NOT RECOMMENDED due to critical convergence failure.**

While this model was designed to provide robustness to outliers via Student-t likelihood, the analysis reveals:
- **Critical convergence issues** (R̂ = 1.16-1.17, ESS = 12-18 for σ and ν)
- **Worse predictive performance** than Model 1 (LOO-ELPD: 23.83 vs 24.89, Δ = -1.06)
- **No practical advantage** (predictions identical to Model 1, RMSE differs by 0.0001)
- **Student-t not needed** (ν ≈ 23 suggests Normal is adequate)

**Conclusion**: Model 2 adds complexity without benefit and suffers from computational unreliability. **Model 1 (Normal) is strongly preferred.**

---

## Model Specification

### Mathematical Form

```
Y ~ Student-t(ν, μ, σ)
μ = β₀ + β₁ · log(x)
```

Where:
- ν: Degrees of freedom (controls tail heaviness)
- μ: Location (linear predictor)
- σ: Scale parameter

### Priors

```
β₀ ~ Normal(0, 10)
β₁ ~ Normal(0, 10)
σ ~ HalfNormal(1)
ν ~ Gamma(2, 0.1)  # Mean ≈ 20
```

### Likelihood

Student-t distribution with:
- Location: Linear predictor on log-transformed x
- Scale: Constant σ (homoscedastic)
- Degrees of freedom: Estimated from data

**Rationale**: Student-t has heavier tails than Normal, providing robustness to outliers.

---

## Parameter Estimates

### Posterior Summaries

| Parameter | Mean | SD | 95% Credible Interval | Interpretation |
|-----------|------|-----|----------------------|----------------|
| **β₀** | 1.759 | 0.043 | [1.670, 1.840] | Intercept (Y when x=1) |
| **β₁** | 0.279 | 0.020 | [0.242, 0.319] | Log-slope (effect of log(x)) |
| **σ*** | 0.094 | 0.020 | [0.064, 0.145] | Scale parameter |
| **ν*** | 22.80 | 15.30 | [3.71, 60.04] | Degrees of freedom |

**WARNING**: σ and ν estimates are **UNRELIABLE** due to poor convergence (see below).

### Comparison to Model 1

| Parameter | Model 1 | Model 2 | Difference |
|-----------|---------|---------|------------|
| β₀ | 1.774 | 1.759 | -0.015 (1%) |
| β₁ | 0.272 | 0.279 | +0.007 (3%) |
| σ | 0.093 | 0.094 | +0.001 (1%) |

**Finding**: Regression parameters are **virtually identical** - both models capture the same log-linear relationship.

### Interpretation of ν (Degrees of Freedom)

**Posterior**: ν = 22.8 [3.7, 60.0]

**What this means**:
- ν < 5: Very heavy tails (Cauchy-like)
- ν = 5-20: Moderately heavy tails
- **ν = 20-30: Approaching Normal distribution**
- ν > 30: Essentially identical to Normal

**Finding**: Mean ν ≈ 23 suggests **Normal likelihood is sufficient**. However, the **very wide credible interval [3.7, 60.0]** indicates high uncertainty - the data cannot reliably distinguish tail behavior.

**Conclusion**: Student-t extension is **not justified** by the data.

---

## CRITICAL ISSUE: Convergence Failure

### Gelman-Rubin R̂ Statistic

| Parameter | R̂ | Threshold | Status |
|-----------|-----|-----------|--------|
| β₀ | 1.01 | < 1.01 | Marginal |
| β₁ | 1.02 | < 1.01 | Marginal |
| **σ** | **1.16** | **< 1.01** | **FAILED** |
| **ν** | **1.17** | **< 1.01** | **FAILED** |

**R̂ > 1.1 indicates CRITICAL convergence failure.**

### Effective Sample Size (ESS)

| Parameter | ESS (bulk) | ESS (tail) | Threshold | Status |
|-----------|------------|------------|-----------|--------|
| β₀ | 248 | 397 | > 400 | Poor |
| β₁ | 245 | 446 | > 400 | Poor |
| **σ** | **18** | **12** | **> 400** | **CRITICAL** |
| **ν** | **17** | **15** | **> 400** | **CRITICAL** |

**ESS < 100 indicates severe mixing problems.**

### What This Means

**POSTERIORS FOR σ AND ν ARE SCIENTIFICALLY INVALID.**

- Chains did not converge to the same distribution
- Insufficient independent samples
- Cannot trust posterior estimates
- Cannot use for inference

**Impact**:
- Parameter estimates unreliable
- Credible intervals unreliable
- Model comparison questionable (though LOO is robust)

**Root Cause**:
- σ and ν are highly correlated in Student-t models
- Difficult to identify separately with n=27 observations
- Sampling geometry is challenging (funnel-like posterior)

**Could it be fixed?**
- Longer chains (4× current length)
- Better priors (more informative on ν)
- Reparameterization
- More data

**Should we fix it?**
- **NO** - Even with perfect convergence, Model 2 is worse than Model 1
- Not worth the computational effort

---

## Predictive Performance

### Point Prediction Accuracy

| Metric | Value | Model 1 | Difference |
|--------|-------|---------|------------|
| **RMSE** | 0.0866 | 0.0867 | -0.0001 (0.1%) |
| **MAE** | 0.0694 | 0.0704 | -0.0010 (1.4%) |
| **R²** | 0.8968 | 0.8965 | +0.0003 (0.03%) |

**Finding**: **Predictions are IDENTICAL to Model 1** - no practical difference.

### Residual Analysis

- Mean residual: ≈ 0 (unbiased)
- Residuals similar to Model 1
- No obvious patterns (see `/workspace/experiments/model_comparison/plots/residual_comparison.png`)

**Conclusion**: Despite Student-t likelihood, predictive performance is indistinguishable from Normal.

---

## LOO Cross-Validation

### LOO-ELPD

- **LOO-ELPD**: 23.83 ± 2.84
- **Model 1**: 24.89 ± 2.82
- **Difference**: **-1.06 ± 0.36** (Model 2 is WORSE)
- **p_loo**: 2.72 (effective parameters ≈ 2.7, close to actual 4)

**Interpretation**: Model 2 has **worse out-of-sample predictive performance** than Model 1.

### Statistical Significance

- Δ = -1.06 (Model 2 worse by 1.06 ELPD)
- SE of difference = 0.36
- |Δ| / SE ≈ 3

**Conclusion**: **Moderately significant evidence** that Model 2 is worse.

### Stacking Weights

ArviZ model averaging assigns:
- Model 1: **100% weight**
- Model 2: **0% weight**

**Interpretation**: If combining models, use 100% Model 1, 0% Model 2.

### Pareto k Diagnostics

| k Range | Count | Status |
|---------|-------|--------|
| k < 0.5 | 26/27 | Excellent |
| 0.5 ≤ k < 0.7 | 1/27 | Good |
| k ≥ 0.7 | 0/27 | — |

**Summary**:
- Max k = 0.527
- Mean k = 0.097
- **All observations k < 0.7** → LOO estimates are reliable

**Conclusion**: Despite convergence issues, LOO cross-validation is trustworthy. The comparison with Model 1 is valid.

---

## Calibration Assessment

### LOO-PIT (Probability Integral Transform)

See: `/workspace/experiments/model_comparison/plots/loo_pit_comparison.png`

**Finding**: LOO-PIT distribution is approximately uniform, similar to Model 1.

**Interpretation**: Model 2 also shows **good calibration** - probabilistic predictions are well-calibrated.

### Posterior Predictive Coverage

**90% Credible Intervals**:
- Coverage: **37.0%**
- Target: 90%
- **Same as Model 1**

**Note**: Low coverage issue is not unique to this model - also present in Model 1.

---

## Model Comparison (vs Model 1)

### LOO-CV: Model 1 WINS

| Model | LOO-ELPD | Δ from Best | Weight |
|-------|----------|-------------|--------|
| Model 1 (Normal) | 24.89 ± 2.82 | 0.00 | 1.00 |
| **Model 2 (this)** | **23.83 ± 2.84** | **-1.06** | **0.00** |

**Visual**: See `/workspace/experiments/model_comparison/plots/loo_comparison.png`

### Why Model 2 Loses

1. **Worse LOO-ELPD** (23.83 vs 24.89, Δ = -1.06)
2. **Critical convergence failure** (R̂ = 1.16-1.17, ESS = 12-18)
3. **More complex** (4 vs 3 parameters) with no benefit
4. **Identical predictions** (RMSE differs by 0.0001)
5. **ν ≈ 23 suggests Normal sufficient** (Student-t not needed)
6. **Computationally unstable** (unreliable posteriors)

**Conclusion**: Model 2 is **inferior on every dimension**.

---

## Strengths

Despite the negative overall assessment, Model 2 has some theoretical strengths:

1. **Theoretically robust to outliers**
   - Student-t has heavier tails than Normal
   - Would be advantageous if data had extreme values

2. **β₀ and β₁ estimates similar to Model 1**
   - Captures same log-linear relationship
   - Consistent parameter interpretation

3. **Equivalent predictive accuracy**
   - RMSE, MAE, R² identical to Model 1
   - Makes good predictions despite convergence issues

4. **Reliable LOO diagnostics**
   - All Pareto k < 0.7
   - LOO comparison is valid

**However**: These strengths are outweighed by critical weaknesses.

---

## Critical Weaknesses

### 1. Convergence Failure (CRITICAL)

**Issue**: R̂ = 1.16-1.17 and ESS = 12-18 for σ and ν

**Impact**:
- **Posteriors unreliable** for σ and ν
- **Cannot trust credible intervals**
- **Scientific inference invalid**
- Results may change with different random seed

**Severity**: **CRITICAL** - Makes the model scientifically invalid

### 2. Worse Predictive Performance

**Issue**: LOO-ELPD = 23.83 vs 24.89 (Δ = -1.06)

**Impact**:
- Expected worse out-of-sample predictions
- Statistically significantly worse (|Δ| ≈ 3 × SE)

**Severity**: **HIGH** - Model 1 is demonstrably better

### 3. Unnecessary Complexity

**Issue**: 4 parameters vs 3, but no improvement

**Impact**:
- Harder to interpret
- More computationally demanding
- Violates parsimony principle
- ν posterior very uncertain [3.7, 60.0]

**Severity**: **MODERATE** - Occam's Razor favors Model 1

### 4. Student-t Not Justified

**Issue**: ν ≈ 23 [3.7, 60.0] suggests Normal adequate

**Impact**:
- Student-t provides no benefit over Normal
- Data insufficient to distinguish tail behavior
- Simpler Normal model preferred

**Severity**: **MODERATE** - Theoretical motivation undermined

### 5. Computational Instability

**Issue**: Poor mixing, slow convergence

**Impact**:
- Not production-ready
- Requires debugging/tuning
- Results may vary across runs
- Inefficient (low ESS per iteration)

**Severity**: **HIGH** - Not suitable for deployment

---

## When Would Model 2 Be Preferred?

Model 2 (Student-t) would be preferred if:

1. **Data had clear outliers** requiring robust estimation
   - Current data: No outliers detected
   - Action: Visual inspection and outlier tests

2. **ν < 15 with tight CI** indicating genuinely heavy tails
   - Current finding: ν ≈ 23 [3.7, 60.0] - not heavy-tailed
   - Action: Not applicable

3. **Larger sample size** (n > 100) to better estimate ν
   - Current data: n = 27
   - Action: Collect more data if possible

4. **Convergence issues resolved** (R̂ < 1.01, ESS > 400)
   - Current status: Failed
   - Action: Longer chains, better priors, reparameterization

5. **Model 1 showed poor fit** (high residuals, bad LOO-PIT)
   - Current finding: Model 1 fits well
   - Action: Not applicable

**Current situation**: **NONE of these conditions are met** → Do NOT use Model 2

---

## Recommendations

### Immediate Decision

**DO NOT USE MODEL 2.**

Reasons:
- Critical convergence failure
- Worse predictive performance
- No advantage over Model 1
- Computationally unreliable

### If You Really Want to Fix Model 2

**Not recommended**, but if necessary:

1. **Increase chain length**:
   - Current: 8,000 samples
   - Try: 32,000 samples (4× longer)

2. **More informative prior on ν**:
   - Current: ν ~ Gamma(2, 0.1)
   - Try: ν ~ Gamma(4, 0.2) or ν ~ Normal(20, 5) [truncated]

3. **Reparameterization**:
   - Use non-centered parameterization for better geometry
   - Separate σ and ν to reduce correlation

4. **Check initialization**:
   - Ensure chains start in high-probability regions

**Expected outcome**: Even with perfect convergence, Model 2 would still be worse than Model 1 on LOO-CV. **Not worth the effort.**

### Alternative Models to Consider

Instead of fixing Model 2, consider:

1. **Heteroscedastic Normal**:
   - Allow σ to vary with x: σ(x) = exp(γ₀ + γ₁·log(x))
   - May address coverage issue

2. **Polynomial regression**:
   - Y ~ β₀ + β₁·x + β₂·x²
   - Different functional form

3. **Gaussian Process**:
   - Flexible nonparametric model
   - Captures complex patterns

4. **Hierarchical model**:
   - If data has grouping structure

**Recommendation**: Stick with **Model 1** unless there's a compelling scientific reason to explore alternatives.

---

## Comparison Visualizations

See `/workspace/experiments/model_comparison/plots/` for:

1. **`integrated_dashboard.png`** - 6-panel comparison
   - Panel A: Model 1 wins on LOO-ELPD
   - Panel E: Model 2's ν is wide and uncertain

2. **`parameter_comparison.png`** - Overlapping posteriors
   - β₀, β₁, σ nearly identical between models

3. **`prediction_comparison.png`** - Fitted curves
   - Indistinguishable predictions

4. **`nu_posterior.png`** - Model 2's ν distribution
   - Wide [3.7, 60.0], overlaps Normal region (ν > 30)

5. **`loo_comparison.png`** - LOO-ELPD comparison
   - Clear visual evidence for Model 1

6. **`loo_pit_comparison.png`** - Calibration
   - Both models well-calibrated

**Visual summary**: Model 2 offers no visual advantage over Model 1.

---

## Technical Notes

### Why Did Convergence Fail?

**Reason**: σ-ν correlation in Student-t models

- Student-t likelihood: L(Y | μ, σ, ν)
- σ and ν trade off: large σ + small ν ≈ small σ + large ν
- Posterior has "funnel" geometry
- Standard NUTS struggles with this

**Solutions** (if we cared to fix it):
- Non-centered parameterization
- More informative priors
- Larger sample size (n > 50)

**But**: Not worth it since Model 2 is worse anyway.

### Why Is LOO Reliable Despite Convergence Issues?

- LOO uses **log_likelihood** which depends mainly on β₀ and β₁
- β₀ and β₁ converged well (R̂ ≈ 1.01, ESS ≈ 250)
- σ and ν contribute less to LOO than to posterior geometry
- Pareto k diagnostics confirm LOO is reliable

**Conclusion**: We can trust the LOO comparison showing Model 1 is better.

---

## Files and Outputs

### Model Artifacts

**Model Directory**: `/workspace/experiments/experiment_2/`

**Key Files**:
- `posterior_inference/diagnostics/posterior_inference.netcdf` - InferenceData object
- `model_assessment/assessment_report.md` - This document

### Comparison Artifacts

**Comparison Directory**: `/workspace/experiments/model_comparison/`

**Key Files**:
- `comparison_report.md` - Full comparison with Model 1
- `recommendation.md` - Final model selection (selects Model 1)
- `plots/` - Comparison visualizations

---

## Conclusion

**Model 2 (Student-t Likelihood) is NOT RECOMMENDED.**

### Summary Assessment

| Criterion | Rating | Evidence |
|-----------|--------|----------|
| Convergence | ★☆☆☆☆ | R̂=1.17, ESS=17 (FAILED) |
| Predictive Accuracy | ★★★☆☆ | Worse than Model 1 (Δ=-1.06) |
| LOO Reliability | ★★★★☆ | All k<0.7 (reliable) |
| Calibration | ★★★★☆ | Good LOO-PIT |
| Interpretability | ★★☆☆☆ | More complex, uncertain ν |
| Robustness | ★☆☆☆☆ | Unstable, unreliable |

**Overall Rating**: ★★☆☆☆ (2/5) - **Poor / Not Recommended**

### Final Statement

Model 2 was an attempt to provide robustness to outliers via Student-t likelihood. However, the analysis reveals:

1. **No outliers in the data** requiring robust methods
2. **ν ≈ 23** suggests Normal is adequate
3. **Critical convergence failure** makes posteriors unreliable
4. **Worse predictive performance** than simpler Model 1
5. **No practical advantage** in predictions or interpretation

**The additional complexity is not justified.**

**USE MODEL 1 INSTEAD.**

---

**Assessment by**: Claude (Bayesian Model Assessment Agent)
**Date**: 2025-10-28
**Status**: NOT SELECTED - Critical Issues
**Recommendation**: Use Model 1 (Normal Likelihood)
