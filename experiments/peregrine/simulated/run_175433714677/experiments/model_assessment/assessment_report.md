# Comprehensive Model Assessment: Two Rejected Models

**Date**: 2025-10-29
**Analyst**: Model Assessment Specialist
**Status**: BOTH MODELS REJECTED

---

## Executive Summary

This assessment evaluates two Negative Binomial regression models, both of which were REJECTED despite excellent computational properties. This is an atypical assessment—rather than selecting the "best" model, we document **why both failed** and what this reveals about the data structure.

### Key Findings

1. **Both models converged perfectly** (R = 1.00, ESS >6000, zero divergences)
2. **Both models are statistically inadequate** (fail posterior predictive checks)
3. **Model 2 provides NO improvement** over Model 1 (ΔELPD = 0.45 ± 0.93)
4. **Common failure mode**: Polynomial functional form is inappropriate for this data
5. **LOO diagnostics are excellent** (all Pareto k < 0.5), so comparisons are reliable

### Bottom Line

**Neither model is acceptable for inference or prediction**. The systematic failures reveal that the data structure requires a fundamentally different modeling approach—likely changepoint models, Gaussian processes, or alternative likelihood structures rather than polynomial growth curves.

---

## Model Specifications

### Model 1: Log-Linear Negative Binomial (REJECTED)

```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  φ ~ Exponential(0.667)
```

**Parameters**: 3
**Decision**: REJECTED
**Rejection Reason**: Systematic residual curvature (quadratic coef = -5.22), 4.17× MAE degradation from early to late period

### Model 2: Quadratic Negative Binomial (REJECTED)

```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  β₂ ~ Normal(0, 0.5)
  φ ~ Exponential(0.667)
```

**Parameters**: 4
**Decision**: REJECTED
**Rejection Reason**: β₂ non-significant (95% CI: [-0.051, 0.173]), no improvement over Model 1, residual curvature WORSENED (coef = -11.99)

---

## LOO-CV Comparison Results

### Quantitative Metrics

| Model | ELPD | SE | p_loo | Pareto k >0.7 | Pareto k >0.5 | Decision |
|-------|------|-----|-------|---------------|---------------|----------|
| **Model 1 (Log-Linear)** | -174.61 | 4.80 | 1.51 | 0/40 (0.0%) | 0/40 (0.0%) | REJECTED |
| **Model 2 (Quadratic)** | -175.06 | 5.21 | 2.12 | 0/40 (0.0%) | 0/40 (0.0%) | REJECTED |

**ΔELPD (Model 2 - Model 1)**: 0.45 ± 0.93

### Interpretation

**Models are EQUIVALENT** (|ΔELPD| < 2 × SE)

- Model 1 is slightly favored (higher ELPD by 0.45)
- The difference is well within sampling uncertainty (SE = 0.93)
- Model 2's additional parameter (β₂) increases model complexity (p_loo: 2.12 vs 1.51) without improving fit
- **Conclusion**: Adding the quadratic term provides NO predictive advantage

### LOO Reliability

**Excellent**: All Pareto k values < 0.5 for both models

- Model 1: max k = 0.35, mean k = 0.13
- Model 2: max k = 0.42, mean k = 0.17
- No problematic observations (k > 0.7)
- LOO estimates are highly reliable—this comparison is trustworthy

**Visual Evidence**: See `plots/loo_comparison.png` and `plots/pareto_k_diagnostics.png`

---

## Computational Diagnostics

### Model 1: Log-Linear

**Convergence**: EXCELLENT

- R max: 1.0000 (target: < 1.01)
- ESS min: 6,603 (target: > 400)
- Divergent transitions: 0
- MCSE/SD: < 0.3%

**Parameter Estimates**:

```
         mean     sd   hdi_3%  hdi_97%
beta_0  4.355  0.049    4.28     4.44
beta_1  0.863  0.050    0.78     0.94
phi    13.835  3.449    8.32    19.77
```

**Assessment**: Model fitting succeeded perfectly. The computational machinery worked flawlessly.

### Model 2: Quadratic

**Convergence**: EXCELLENT

- R max: 1.0000 (target: < 1.01)
- ESS min: 7,783 (target: > 400)
- Divergent transitions: 0
- MCSE/SD: < 0.3%

**Parameter Estimates**:

```
         mean     sd   hdi_3%  hdi_97%
beta0   4.300  0.073    4.16     4.43
beta1   0.862  0.050    0.76     0.95
beta2   0.059  0.057   -0.05     0.17    ← INCLUDES 0 (non-significant)
phi    13.579  3.368    7.72    20.06
```

**Assessment**: Model fitting succeeded perfectly. The issue is NOT computational—it's that β₂ is genuinely not supported by the data.

**Visual Evidence**: See `plots/convergence_summary.png` and `plots/parameter_comparison.png`

---

## Critical Finding: β₂ is Non-Significant

### The Quadratic Term Does Not Help

**Posterior**: β₂ = 0.059 ± 0.057
**95% Credible Interval**: [-0.051, 0.173]
**Includes Zero**: YES
**Meaningful Effect**: NO (|β₂| < 0.1 threshold)

### What This Means

1. **The data do NOT support a quadratic trend** in log-space
2. The posterior for β₂ is essentially unchanged from the prior N(0, 0.5)
3. EDA predicted ΔELPD > 10, but Bayesian inference found ΔELPD ≈ 0
4. This discrepancy reveals that:
   - Point estimates (OLS/MLE) can be misleading
   - Proper uncertainty quantification changes conclusions
   - The apparent curvature in EDA may be noise or a different pattern

### Paradox: Residual Curvature WORSENED

Despite adding a quadratic term, residual analysis shows:

- **Model 1**: Quadratic coefficient in residuals = -5.22
- **Model 2**: Quadratic coefficient in residuals = -11.99 (WORSE!)

**Interpretation**: The quadratic form (β₀ + β₁×year + β₂×year²) is NOT the right functional form to capture the data's nonlinearity. The curvature exists, but it's not well-described by a simple polynomial.

---

## Posterior Predictive Check Summary

### Model 1: Log-Linear

| Criterion | Target | Result | Status |
|-----------|--------|---------|---------|
| Residual curvature | \|coef\| < 1.0 | -5.22 | FAIL |
| Late/Early MAE ratio | < 2.0 | 4.17 | FAIL |
| Var/Mean recovery | 95% in [50, 90] | 67% | FAIL |
| 90% Coverage | > 80% | 100% | PASS (over-conservative) |

**Verdict**: 1 of 4 criteria passed → FAIL

**Key Failures**:
- Strong inverted-U residual pattern (systematic underprediction in late period)
- Predictive accuracy degrades 4× from early to late period
- Overestimates dispersion (prediction intervals too wide)

### Model 2: Quadratic

| Criterion | Target | Result | Status |
|-----------|--------|---------|---------|
| β₂ significance | CI excludes 0, \|β₂\| > 0.1 | [-0.05, 0.17] | FAIL |
| LOO improvement | ΔELPD > 4 | 0.4 | FAIL |
| Residual curvature | \|coef\| < 1.0 | -11.99 | FAIL (WORSE!) |
| Late/Early MAE ratio | < 2.0 | 4.56 | FAIL (WORSE!) |
| Var/Mean recovery | 95% in [50, 90] | Partially | PASS |
| Coverage | 80-95% | 100% | FAIL (too conservative) |

**Verdict**: 1 of 6 criteria passed → FAIL

**Key Failures**:
- β₂ not significantly different from zero
- NO improvement in predictive accuracy (ΔELPD ≈ 0)
- Residual curvature DOUBLED compared to Model 1
- Late-period fit DEGRADED (MAE ratio 4.56 vs 4.17)

---

## Parameter Comparison

### Shared Parameters

**β₀ (Intercept)**:
- Model 1: 4.355 ± 0.049
- Model 2: 4.300 ± 0.073
- **Difference**: Minimal, within posterior uncertainty
- Model 2 has slightly wider posterior (more uncertainty due to added parameter)

**β₁ (Linear Term)**:
- Model 1: 0.863 ± 0.050
- Model 2: 0.862 ± 0.050
- **Difference**: Essentially identical
- The quadratic term didn't change the linear term estimate

**φ (Dispersion)**:
- Model 1: 13.835 ± 3.449
- Model 2: 13.579 ± 3.368
- **Difference**: Minimal
- Both models estimate similar overdispersion

**Interpretation**: Model 2 essentially recovers Model 1 with an extra parameter (β₂) that's indistinguishable from zero.

### Model 2 Unique Parameter

**β₂ (Quadratic Term)**:
- Posterior: 0.059 ± 0.057
- 95% CI: [-0.051, 0.173]
- Includes zero: YES
- **Interpretation**: No evidence for quadratic curvature in log-space

**Visual Evidence**: See `plots/parameter_comparison.png` - the β₂ posterior clearly overlaps zero

---

## Common Failure Mode Analysis

### What We've Learned

Both models fail in similar ways despite different specifications:

1. **Late-period fit degradation** (4.17× and 4.56× MAE ratios)
2. **Strong residual curvature** (-5.22 and -11.99)
3. **Overestimated dispersion** (φ ≈ 14 captures systematic error as random variation)
4. **Over-conservative predictions** (100% coverage at 90% nominal level)

### Root Cause: Wrong Functional Form Class

**The Problem is NOT**:
- Computational issues (both converged perfectly)
- Prior specification (posteriors are data-driven)
- Likelihood family (Negative Binomial is appropriate for count data)
- Parameter identification (all parameters are well-estimated)

**The Problem IS**:
- **Polynomial growth in log-space is fundamentally inadequate**
- The data exhibit a pattern that cannot be captured by β₀ + β₁×year + β₂×year²
- The curvature is real (evident in residuals) but not polynomial in nature

### Evidence for Non-Polynomial Structure

1. **Adding polynomial terms made things WORSE** (residual curvature doubled)
2. **β₂ is non-significant** despite clear visual curvature in EDA
3. **LOO shows no improvement** (ΔELPD ≈ 0)
4. **Late-period errors compound** (not a random fluctuation, but systematic divergence)

### Possible Data Structures

The systematic failures suggest the data may have:

1. **Changepoint structure**: Different growth regimes (early vs late period)
2. **Exponential growth in growth rate**: Not polynomial but super-exponential
3. **Missing covariates**: Confounders driving the apparent curvature
4. **Measurement changes**: Systematic shifts in data collection
5. **Non-stationary process**: Time-varying parameters

---

## Why Did EDA Mislead Us?

### The Discrepancy

**EDA Prediction** (based on OLS R² comparison):
- Linear R² = 0.92
- Quadratic R² = 0.96
- Expected improvement: ΔELPD > 10

**Bayesian Reality**:
- ΔELPD = 0.45 ± 0.93 (models equivalent)
- β₂ non-significant
- No predictive improvement

### Lessons Learned

1. **R² is a poor guide for model selection**
   - R² = 0.96 suggests good fit, but misses systematic patterns
   - High R² doesn't imply well-calibrated uncertainty
   - Residual diagnostics are essential

2. **Point estimates (OLS/MLE) differ from Bayesian posteriors**
   - OLS can find a "best fit" quadratic even when the effect is not robust
   - Bayesian approach properly accounts for uncertainty
   - Credible intervals reveal that β₂ is not reliably different from zero

3. **In-sample fit ≠ out-of-sample prediction**
   - LOO-CV shows no predictive advantage despite better in-sample R²
   - Overfitting: quadratic term fits noise, not signal

4. **Visual inspection can be misleading**
   - Apparent curvature in scatter plots may be:
     - Noise (sampling variation)
     - A different functional form (not quadratic)
     - Driven by influential observations
   - Proper model comparison is essential

---

## Diagnostic Plots Summary

All plots are saved in `/workspace/experiments/model_assessment/plots/`

### 1. LOO Comparison (`loo_comparison.png`)

**Shows**: Model comparison using LOO-CV ELPD with standard errors

**Finding**: Models are indistinguishable (error bars overlap completely)

**Implication**: Model 2's added complexity buys nothing

### 2. Pareto k Diagnostics (`pareto_k_diagnostics.png`)

**Shows**: Pareto k values for each observation (both models)

**Finding**: All k values < 0.5 (excellent LOO reliability)

**Implication**: The comparison is trustworthy—no problematic observations

### 3. Parameter Comparison (`parameter_comparison.png`)

**Shows**: Posterior distributions for all parameters (side-by-side)

**Finding**:
- β₀, β₁, φ are nearly identical between models
- β₂ clearly overlaps zero

**Implication**: Model 2 recovers Model 1 + noise

### 4. Convergence Summary (`convergence_summary.png`)

**Shows**: R, ESS, and decision status for both models

**Finding**: Both models CONVERGED perfectly but REJECTED for statistical reasons

**Implication**: Computational success ≠ statistical adequacy

---

## Model Complexity vs. Performance

### Effective Parameters (p_loo)

- **Model 1**: p_loo = 1.51 (nominal: 3 parameters)
- **Model 2**: p_loo = 2.12 (nominal: 4 parameters)

**Interpretation**:
- Model 1 is effectively using fewer parameters than it has (1.51 vs 3)
  - Suggests priors are constraining or data are sparse
- Model 2 is using more effective parameters (2.12 vs 1.51)
  - The extra parameter adds complexity without improving fit
  - Classic overfitting signature

### Parsimony Principle

When models have equivalent predictive performance (ΔELPD < 2 SE), prefer the simpler model:

- **Model 1**: 3 parameters, simpler interpretation
- **Model 2**: 4 parameters, added complexity

**Decision**: Even if we HAD to choose one, Model 1 is preferred (parsimony + slightly better ELPD)

But **neither should be used** for inference or prediction given the systematic failures.

---

## Critical Analysis: Why Did Both Models Fail?

### Question 1: Are Both Models Computationally Sound?

**YES**—Perfect convergence:
- R = 1.0000 for all parameters
- ESS > 6,000 (far exceeding threshold of 400)
- Zero divergent transitions
- MCSE < 0.3% of posterior SD

### Question 2: Are Both Models Statistically Inadequate?

**YES**—Both fail posterior predictive checks:
- Model 1: 3 of 4 criteria failed
- Model 2: 5 of 6 criteria failed
- Systematic residual patterns persist
- Late-period predictive accuracy degrades 4×

### Question 3: What is the Common Failure Mode?

**Polynomial functional form is inappropriate**:
- Linear (Model 1): Assumes constant exponential growth
- Quadratic (Model 2): Assumes polynomial acceleration
- Data: Exhibit a pattern not captured by either

Evidence:
- Residual curvature persists (and worsens) despite adding β₂
- Late-period errors compound systematically
- β₂ is non-significant yet residuals still curved

### Question 4: What Alternatives Should Be Explored?

See **Recommendations** section below.

---

## Sensitivity and Robustness

### Prior Sensitivity

Both models used weakly informative priors:
- β₀, β₁: Normal with SD = 0.5-1.0
- β₂: Normal(0, 0.5)
- φ: Exponential(0.667)

**Assessment**: The failures would persist with different priors because:
1. Posteriors are data-driven (high ESS, priors are overwhelmed)
2. Structural misspecification is not fixable by prior tuning
3. β₂ would remain non-significant with flatter or narrower priors

### Outlier Influence

**No single observation drives the failures**:
- All Pareto k < 0.5 (no influential points in LOO)
- Residual patterns are smooth and systematic (not driven by outliers)
- All 40 observations contribute to the curvature signal

**Conclusion**: These are robust, distributed failures—not artifacts of data quirks.

---

## What Use Cases Are Ruled Out?

Given both models are rejected, the following applications are **NOT VALID**:

### Forecasting

- Both models systematically underpredict in late period
- Errors compound over time (4× degradation)
- Extrapolation beyond observed data would be highly unreliable
- Wide prediction intervals (100% coverage) reduce practical utility

### Scientific Inference

- Cannot identify true growth mechanism
- Model 1: Assumes constant exponential growth (violated)
- Model 2: Assumes polynomial acceleration (not supported)
- True process remains uncharacterized

### Resource Planning

- Late-period MAE of 26-28 represents substantial uncertainty
- Prediction intervals too wide for precise allocation
- Systematic bias would lead to under-preparation

### Causal Inference

- Structural misspecification precludes causal interpretation
- Cannot separate trend from other effects
- Missing the true functional form

---

## What These Models DID Accomplish

Despite rejection, both models served important purposes:

### 1. Established Baseline Performance

- Lower bound on model complexity
- Reference point for future comparisons
- Clear evidence that "simple" models are inadequate

### 2. Diagnostic Value

- Revealed the SPECIFIC way data deviate from polynomial growth
- Identified late-period degradation as key failure mode
- Demonstrated that adding polynomial terms doesn't help

### 3. Methodological Rigor

- Validated the falsification framework
- Demonstrated importance of posterior predictive checks
- Showed that R² and convergence are not sufficient

### 4. Scientific Learning

- Growth is NOT constant-rate exponential (Model 1 fails)
- Growth is NOT polynomial super-exponential (Model 2 fails)
- Points to more complex structure (changepoint, GP, etc.)

---

## Comparison to Study Design Expectations

### Pre-Registered Criteria

From `experiments/experiment_1/metadata.md` and `experiments/experiment_2/metadata.md`:

**Model 1 Rejection Criteria** (from metadata):
1. LOO-CV: ΔELPD >4 vs. quadratic → **NOT MET** (ΔELPD ≈ 0, models equivalent)
2. Systematic curvature in residuals → **MET** (coef = -5.22)
3. Late period MAE >2× early period → **MET** (4.17×)
4. Var/Mean outside [50, 90] → **MET** (CI to 131)
5. Coverage <80% → Not met (100%)

**Result**: 3 of 4 evaluated criteria violated → REJECTED

**Model 2 Rejection Criteria** (from metadata):
1. β₂ not significant (CI includes 0, |β₂| < 0.1) → **MET**
2. No improvement over Model 1 (ΔELPD < 2) → **MET**
3. Residual curvature persists (|coef| > 1.0) → **MET** (coef = -11.99)
4. Late period still poor (MAE ratio > 2.0) → **MET** (4.56)

**Result**: 4 of 4 criteria violated → REJECTED

### Study Design Worked as Intended

The falsification framework successfully:
1. Identified Model 1 as inadequate (baseline rejection)
2. Tested Model 2 as natural extension
3. Discovered Model 2 also inadequate (added complexity doesn't help)
4. Guided next steps (need different model class)

This is **scientific progress**—we've learned what DOESN'T work and why.

---

## Conclusions

### Summary of Findings

1. **Both models are computationally excellent** (perfect convergence, reliable LOO estimates)
2. **Both models are statistically inadequate** (fail posterior predictive checks)
3. **Model 2 provides no improvement** over Model 1 (ΔELPD ≈ 0, β₂ non-significant)
4. **The common failure mode** is polynomial functional form inadequacy
5. **Neither model is suitable** for inference, prediction, or decision-making

### Key Insight

The most important finding is **HOW they failed**:
- Polynomial growth models cannot capture this data structure
- The curvature is real but not polynomial
- Late-period systematic degradation suggests regime change or missing structure
- Simple functional forms are insufficient

### What This Tells Us About the Data

The systematic failures across two polynomial models reveal:

1. **Growth is more complex than polynomial**: Neither linear nor quadratic works
2. **There may be regimes or changepoints**: Early vs late period behave differently
3. **The process may be non-stationary**: Parameters change over time
4. **Missing covariates are possible**: Confounders drive apparent patterns
5. **Measurement or structural changes**: Something fundamental shifted

### Methodological Lessons

1. **Computational success ≠ Statistical adequacy**: R = 1.00 doesn't mean the model is right
2. **R² is insufficient**: 0.96 can hide systematic failures
3. **EDA can mislead**: Point estimates suggested quadratic would help, but Bayesian analysis disagreed
4. **Posterior predictive checks are essential**: Only comprehensive validation reveals inadequacy
5. **Model comparison prevents overfitting**: LOO-CV catches complexity that doesn't help

---

## Next Steps

This assessment confirms that both polynomial models are inadequate. The rejection of BOTH models guides our next modeling choices:

### See Recommendations Document

Detailed recommendations for alternative model classes are provided in:

`/workspace/experiments/model_assessment/recommendations.md`

Including:
- Changepoint models (different growth regimes)
- Gaussian processes (flexible nonparametric trend)
- Alternative likelihoods (different data structures)
- Time-varying coefficient models
- Missing covariate investigation

---

## Files and Reproducibility

All evidence supporting this assessment is documented and reproducible:

### Code

- `/workspace/experiments/model_assessment/code/comprehensive_assessment.py`
- All analyses use seed=42 for reproducibility

### Data

- Model 1 InferenceData: `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Model 2 InferenceData: `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`
- Both contain log_likelihood for LOO-CV

### Plots

All diagnostic visualizations in `/workspace/experiments/model_assessment/plots/`:
1. `loo_comparison.png` - LOO-CV ELPD comparison
2. `pareto_k_diagnostics.png` - Pareto k reliability diagnostics
3. `parameter_comparison.png` - Posterior distributions side-by-side
4. `convergence_summary.png` - Convergence metrics and decision status

### Metrics

- CSV summary: `/workspace/experiments/model_assessment/loo_comparison.csv`
- Detailed JSON: `/workspace/experiments/model_assessment/comparison_results.json`

### Reports

- This document: `/workspace/experiments/model_assessment/assessment_report.md`
- Recommendations: `/workspace/experiments/model_assessment/recommendations.md`
- Model 1 critique: `/workspace/experiments/experiment_1/model_critique/critique_summary.md`
- Model 2 summary: `/workspace/experiments/experiment_2/posterior_inference/inference_summary.md`

---

**Assessment completed**: 2025-10-29
**Analyst**: Model Assessment Specialist
**Status**: BOTH MODELS REJECTED - Polynomial functional form inadequate
**Recommendation**: Explore changepoint models, GPs, or alternative structures (see recommendations.md)
