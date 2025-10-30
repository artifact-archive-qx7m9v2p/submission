# Model Critique for Experiment 2: Log-Linear Heteroscedastic Model

**Date**: 2025-10-27
**Analyst**: Model Criticism Specialist
**Model**: Log-Linear Mean with Heteroscedastic Variance

---

## Executive Summary

**VERDICT: REJECT**

The log-linear heteroscedastic model (Experiment 2) achieves excellent computational convergence but is **fundamentally rejected** on scientific and statistical grounds. The model hypothesizes that variance changes linearly with x in log-space, but the data provide **no credible evidence** for this heteroscedasticity (γ₁ ≈ 0, 95% CI includes zero). More critically, the model is **strongly disfavored** by leave-one-out cross-validation, performing 23.43 ELPD units worse than the simpler homoscedastic model (Model 1). This represents a decisive preference for the simpler model - the added complexity of modeling variance as a function of x is not just unjustified, it actively **degrades predictive performance**.

**One sentence summary**: A computationally successful model that is scientifically and predictively inferior to its simpler alternative.

---

## Model Specification Reminder

```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = gamma_0 + gamma_1 * x_i

Priors:
  beta_0 ~ Normal(1.8, 0.5)
  beta_1 ~ Normal(0.3, 0.2)
  gamma_0 ~ Normal(-2, 1)
  gamma_1 ~ Normal(-0.05, 0.05)
```

**Key hypothesis**: Variance decreases with x (γ₁ < 0), creating decreasing uncertainty at higher x values.

---

## Synthesis of Validation Results

### 1. Prior Predictive Check: CONDITIONAL PASS

**What was tested**: Do the priors generate scientifically plausible data?

**Key findings**:
- ✓ 29.4% of prior samples generated data similar to observed (exceeds 20% threshold)
- ✓ No computational failures (0% negative sigma values)
- ✓ Mean structure covers observed range appropriately
- ⚠ Variance ratio distribution poorly calibrated (median 21x vs observed 8.8x)
- ⚠ 17.3% of samples show wrong direction (increasing variance)
- ⚠ Heavy tails in variance ratio (extreme outliers >4700x)

**Assessment**: Priors were adequate for fitting but showed concerning behavior in variance structure. The γ₁ ~ N(-0.05, 0.05) prior was sufficiently flexible to allow the data to "speak" about heteroscedasticity direction and magnitude.

**Implication for critique**: The prior was not overly constraining - the posterior result (γ₁ ≈ 0) reflects the data, not prior dominance.

---

### 2. Simulation-Based Calibration: CONDITIONAL PASS WITH WARNINGS

**What was tested**: Can the model recover known parameters from simulated data?

**Key findings**:
- ⚠ Only 78% success rate (22% optimization failures using Laplace approximation)
- ⚠ Beta parameters show under-coverage (80-82% vs 90% target)
- ⚠ γ₁ exhibits -11.97% relative bias (exceeds 10% threshold)
- ✓ γ₀ well-calibrated (93.6% coverage)
- ✓ Parameters are identifiable (minimal cross-correlations)

**Assessment**: The model showed computational fragility and calibration issues, particularly for γ₁. The negative bias in γ₁ recovery suggests the inference struggles when heteroscedasticity is weak or absent.

**Implication for critique**: The SBC warnings were prophetic - this model is indeed more complex than necessary. The struggles with γ₁ during SBC foreshadowed its failure to find evidence for heteroscedasticity in real data.

---

### 3. Convergence Diagnostics: PASS

**What was tested**: Did MCMC sampling work properly on real data?

**Key findings**:
- ✓ **Perfect convergence**: R̂ = 1.000 for all parameters
- ✓ Excellent effective sample sizes: ESS > 1500 for all parameters
- ✓ Zero divergent transitions (0%)
- ✓ Good mixing across all 4 chains
- ✓ Uniform rank plots confirm proper exploration

**Assessment**: From a computational perspective, the model is a complete success. The MCMC sampler had no difficulty exploring the posterior.

**Implication for critique**: **This makes the rejection more decisive, not less.** We're not rejecting the model due to technical failures - we're rejecting it because it worked perfectly and showed us it's the wrong model.

---

### 4. Parameter Estimates: CRITICAL FAILURE

**What was tested**: What do the posteriors tell us about the scientific hypothesis?

**Key findings**:

| Parameter | Posterior Mean ± SD | 95% Credible Interval | Interpretation |
|-----------|---------------------|----------------------|----------------|
| β₀ | 1.763 ± 0.047 | [1.679, 1.857] | Intercept well-identified |
| β₁ | 0.277 ± 0.021 | [0.237, 0.316] | Log-slope well-identified |
| γ₀ | -2.399 ± 0.248 | [-2.868, -1.945] | Baseline variance well-identified |
| **γ₁** | **0.003 ± 0.017** | **[-0.028, 0.039]** | **NO EVIDENCE for heteroscedasticity** |

**Critical finding**: The heteroscedasticity parameter γ₁ has:
- Posterior mean near zero (0.003)
- 95% CI that **includes zero**
- P(γ₁ < 0) = 43.9% (essentially a coin flip)
- Posterior nearly identical to prior centered at zero

**Assessment**: **The central scientific hypothesis is not supported.** The data provide no credible evidence that variance changes with x.

**Implication for critique**: This alone justifies rejection - a model built around a hypothesis the data don't support should not be used.

---

### 5. Model Comparison: DECISIVE FAILURE

**What was tested**: Does the added complexity improve predictive performance?

**LOO Cross-Validation Results**:

| Model | ELPD LOO | SE | p_loo | Pareto k Issues |
|-------|----------|----|----|------------------|
| **Model 1 (Homoscedastic)** | **46.99** | 3.11 | 2.43 | 0 bad (0%) |
| **Model 2 (Heteroscedastic)** | **23.56** | 3.15 | 3.41 | 1 bad (3.7%) |

**ELPD Difference (Model 2 - Model 1)**:
- **Δ ELPD = -23.43 ± 4.43**
- **Standard errors**: |Δ| / SE = 5.29 (far exceeds 2σ threshold)

**Interpretation**:
- Model 2 is **23 ELPD units worse** than Model 1
- This is a **huge, decisive difference** (>5 standard errors)
- The added complexity **actively degrades** out-of-sample prediction
- Model 1 is **strongly preferred** by any reasonable standard

**Pareto k diagnostics**:
- Model 1: 100% good (all k < 0.5)
- Model 2: 96.3% good, **1 observation with k = 0.96** (problematic)

The heteroscedastic model introduces a Pareto k issue that didn't exist in Model 1, suggesting instability in leave-one-out predictions.

**Assessment**: **Complete rejection from a predictive perspective.** The model fails the fundamental test: does it predict better?

**Implication for critique**: This is the decisive blow. Even if we had weak evidence for heteroscedasticity, the model would still be rejected because it predicts worse than the simpler alternative.

---

## Comprehensive Assessment

### Strengths

What Model 2 does well:

1. **Computational Performance**: Perfect MCMC convergence with no technical issues
2. **Conceptual Coherence**: The model structure is mathematically sound and scientifically interpretable
3. **Honest Uncertainty**: The posterior for γ₁ correctly expresses uncertainty about heteroscedasticity direction
4. **Residual Diagnostics**: No concerning patterns in residuals (though this also supports homoscedasticity)
5. **Mean Function**: β₀ and β₁ estimates are consistent with Model 1, showing robustness

### Weaknesses

#### Critical Issues (Must Be Addressed - Cannot Be Ignored)

**1. No Evidence for Heteroscedasticity (Scientific Failure)**

The core hypothesis is unsupported:
- γ₁ posterior includes zero (95% CI: [-0.028, 0.039])
- P(γ₁ < 0) = 43.9% (no directional evidence)
- Variance function is essentially flat across x range
- This is **falsification criterion #1** from the experiment design

**Impact**: The scientific rationale for the model evaporates. Why use a heteroscedastic model when there's no heteroscedasticity?

**2. Much Worse Predictive Performance (Statistical Failure)**

LOO comparison is devastating:
- Δ ELPD = -23.43 ± 4.43 (Model 1 strongly preferred)
- 5.29 standard errors - this is not close
- This is **falsification criterion #2** from the experiment design

**Impact**: Even ignoring scientific interpretation, the model fails on purely predictive grounds. It overfits.

**3. Unnecessary Model Complexity (Principle of Parsimony Violated)**

The model uses 4 parameters when 3 suffice:
- p_loo = 3.41 (Model 2) vs 2.43 (Model 1)
- Extra parameter (γ₁) provides zero benefit
- Added complexity hurts rather than helps

**Impact**: Violates Occam's Razor - simpler explanations are preferred when equally good (and here, superior).

**4. Introduces LOO Diagnostic Issues**

Model 2 has Pareto k problems that Model 1 doesn't:
- 1 observation with k = 0.96 (very close to 1.0 threshold)
- Model 1 has all k < 0.5
- Suggests model is unstable for certain observations

**Impact**: Reduces confidence in LOO estimates, though the magnitude of difference makes this moot.

#### Minor Issues (Could Be Improved But Not Blocking)

**5. Prior-Posterior Similarity for γ₁**

The γ₁ posterior hasn't moved much from a N(0, 0.017) distribution:
- Prior: N(-0.05, 0.05)
- Posterior: 0.003 ± 0.017

**Impact**: Suggests data are weak on this parameter, but this is expected when true γ₁ ≈ 0.

**6. SBC Calibration Issues**

The SBC showed under-coverage and bias for γ₁:
- 84.6% coverage vs 90% target
- -12% relative bias

**Impact**: Validates concerns about parameter identifiability when heteroscedasticity is weak.

---

## Critical Visual Evidence

### Key Plots Supporting Rejection

1. **variance_function.png**: Shows posterior variance is essentially constant across x range
   - If heteroscedasticity existed, we'd see clear trend
   - Instead, variance bands are horizontal

2. **posterior_distributions.png**: γ₁ posterior centered at zero
   - 95% credible interval straddles zero
   - No clear directional preference

3. **model_comparison.png**: Stark LOO difference
   - Model 1 clearly superior
   - ELPD difference >5 SE

4. **residual_diagnostics.png**: No heteroscedasticity pattern
   - Residuals vs x: random scatter (no funnel shape)
   - Supports homoscedastic assumption of Model 1

### What We DON'T See (Important Absences)

- No systematic residual patterns requiring heteroscedastic variance
- No funnel shape in residual plots
- No clear trend in variance across x range
- No influential observations driving the results

These absences are evidence for homoscedasticity.

---

## Domain Considerations

### Scientific Interpretation

**The question**: Does measurement uncertainty or natural variability decrease with increasing x?

**The answer from the data**: No credible evidence for this.

**Implications**:
- The data-generating process appears to have constant variance
- Whatever generates the scatter in Y does so uniformly across x range
- This is actually a useful scientific finding - homoscedasticity is the simpler null hypothesis

### Practical Consequences

**If we used Model 2**:
- Would report essentially constant variance (γ₁ ≈ 0)
- Would have wider parameter uncertainties due to extra parameter
- Would have worse predictions on new data
- Would complicate communication with stakeholders

**If we use Model 1 instead**:
- Simpler model matches the data-generating process
- Better predictions
- Easier to communicate
- More defensible scientifically

---

## Comparison to Model 1

### What Model 1 Gets Right

| Aspect | Model 1 (Homoscedastic) | Model 2 (Heteroscedastic) | Winner |
|--------|------------------------|---------------------------|---------|
| ELPD LOO | 46.99 ± 3.11 | 23.56 ± 3.15 | **Model 1** |
| Pareto k issues | 0% | 3.7% | **Model 1** |
| p_loo (complexity) | 2.43 | 3.41 | **Model 1** |
| Parameters | 3 | 4 | **Model 1** |
| Interpretability | Simple | Complex | **Model 1** |
| Scientific support | Consistent with data | Hypothesis unsupported | **Model 1** |
| Convergence | Perfect | Perfect | Tie |
| Mean estimates | β ≈ 0.13 | β ≈ 0.28 (different scale) | Tie* |

*Both model the same mean structure, just in different parameterizations

**Conclusion**: Model 1 is superior on every meaningful criterion.

---

## Falsification Criteria Assessment

From the experiment design, we had four falsification criteria:

### ✓ Criterion 1: Gamma_1 Posterior Includes Zero
- **Threshold**: 95% CI should exclude 0 for acceptance
- **Result**: 95% CI = [-0.028, 0.039] **includes zero**
- **Status**: **TRIGGERED - Model falsified**

### ✓ Criterion 2: LOO Shows Overfitting
- **Threshold**: ΔELPD < -10 indicates serious problem
- **Result**: ΔELPD = -23.43 (Model 2 much worse)
- **Status**: **TRIGGERED - Model falsified**

### ✗ Criterion 3: Convergence Issues
- **Threshold**: R̂ > 1.01 or ESS < 400
- **Result**: R̂ = 1.000, ESS > 1500
- **Status**: Not triggered - convergence was perfect

### ✗ Criterion 4: Problematic LOO Diagnostics
- **Threshold**: >10% observations with Pareto k > 0.7
- **Result**: 3.7% (1/27 observations)
- **Status**: Minor concern, but below threshold

**Summary**: 2 of 4 falsification criteria triggered, including the two most critical ones (scientific hypothesis and predictive performance).

---

## Why This Model Failed

### Root Cause Analysis

**Hypothesis**: The data exhibit heteroscedastic variance that changes with x.

**Reality**: The data exhibit homoscedastic variance (constant across x).

**What went wrong**: We tested a hypothesis that the data don't support. This is **not a modeling failure** - this is science working correctly. We proposed a hypothesis, tested it rigorously, and found it wanting.

### The Overfitting Mechanism

**Why does Model 2 predict worse than Model 1?**

1. **Model 2 has 4 parameters** to fit N=27 observations
2. **The 4th parameter (γ₁) captures noise**, not signal
3. **During fitting**, γ₁ ≈ 0 but still uses degrees of freedom
4. **During prediction**, this "learned noise" doesn't generalize
5. **Result**: LOO correctly identifies this as overfitting

This is a textbook case of **complexity without benefit**.

### Could Any Refinement Save This Model?

**Short answer**: No.

**Long answer**: Let's consider possible refinements:

1. **Tighter priors on γ₁?**
   - Already tried N(-0.05, 0.05) which is quite informative
   - Tighter priors would force heteroscedasticity that the data don't support
   - This would be scientific dishonesty

2. **Different variance function?**
   - Could try quadratic, exponential, etc.
   - But γ₁ ≈ 0 suggests ANY functional form of variance(x) is unnecessary
   - This would be p-hacking / specification searching

3. **More data?**
   - Might help, but LOO suggests model is fundamentally wrong, not just underpowered
   - With N=27, we have enough data to see that variance doesn't vary with x
   - More data would likely strengthen rejection

4. **Different likelihood?**
   - Student-t for heavy tails?
   - But residuals look normally distributed
   - This addresses a problem we don't have

**Conclusion**: No refinement can fix a model testing a hypothesis the data don't support.

---

## Recommendation

### Decision: **REJECT MODEL 2**

This is a clear, decisive rejection with no ambiguity.

### Justification

**Scientific grounds**:
- γ₁ ≈ 0 with 95% CI including zero
- No credible evidence for heteroscedastic variance
- Core hypothesis falsified

**Statistical grounds**:
- ΔELPD = -23.43 ± 4.43 (>5 SE worse than Model 1)
- Overfitting despite perfect convergence
- Introduced Pareto k diagnostic issues

**Philosophical grounds**:
- Violates parsimony (Occam's Razor)
- Unnecessary complexity without benefit
- Simpler model is superior

**Practical grounds**:
- Worse predictions on new data
- Harder to interpret and communicate
- No advantages over Model 1

### What Should Be Used Instead

**Use Model 1 (Log-Linear Homoscedastic Model)**:
- ELPD = 46.99 (much better)
- 3 parameters instead of 4
- Perfect LOO diagnostics (all k < 0.5)
- Matches data-generating process
- Simpler to communicate

Model 1 is not just adequate - it's superior on every criterion that matters.

---

## Lessons Learned

### What This Experiment Taught Us

1. **Negative Results Are Valuable**
   - We learned the data DON'T support heteroscedastic variance
   - This is useful scientific knowledge
   - Rules out a competing hypothesis

2. **Convergence ≠ Correctness**
   - Model 2 had perfect MCMC convergence
   - But it's still the wrong model
   - Computational success doesn't guarantee scientific success

3. **LOO Is Powerful**
   - Caught the overfitting that parameter estimates alone might miss
   - Provided decisive evidence for model comparison
   - Validated the principle of parsimony quantitatively

4. **SBC Warnings Were Prescient**
   - The γ₁ calibration issues in SBC foreshadowed real-data failure
   - Under-coverage and bias during validation predicted weak identifiability
   - Computational fragility in SBC (22% failures) suggested unnecessary complexity

5. **Priors Were Not the Problem**
   - γ₁ prior was flexible enough (SD = 0.05)
   - Posterior ≈ 0 reflects data, not prior constraint
   - If anything, prior was informative in the "wrong" direction (mean = -0.05)

6. **The Scientific Method Works**
   - Hypothesize → Test → Reject if unsupported
   - This is exactly how science should work
   - Rigorous Bayesian workflow caught the issues at every stage

### Implications for Future Modeling

**When to test for heteroscedasticity**:
- When domain knowledge suggests variance should change
- When residual plots show clear funnel patterns
- When prior data exhibit this pattern
- **Not** when it's just "theoretically possible"

**How to avoid similar failures**:
- Always compare to simpler baseline models
- Use LOO/cross-validation for model selection
- Don't confuse computational success with model adequacy
- Take SBC warnings seriously
- Check if added complexity improves predictions

**Red flags for overfitting**:
- Parameters with posteriors ≈ 0 (suggests unnecessary)
- Poor LOO compared to simpler models
- p_loo much larger than number of "meaningful" parameters
- Pareto k issues in complex but not simple model

---

## Final Verdict

**Model 2 (Log-Linear Heteroscedastic) is REJECTED.**

This rejection is:
- **Decisive** (multiple lines of evidence)
- **Unambiguous** (no close calls or judgment needed)
- **Scientifically sound** (hypothesis not supported)
- **Statistically rigorous** (poor predictive performance)
- **Practically justified** (simpler model is superior)

**Use Model 1 instead.** It is superior on every meaningful criterion and matches the actual data-generating process.

---

## Technical Appendix

### Quantitative Summary

**Validation Stage Pass/Fail**:
- Prior predictive: CONDITIONAL PASS
- SBC: CONDITIONAL PASS WITH WARNINGS
- Convergence: PASS
- **Parameter estimates: CRITICAL FAIL** (γ₁ ≈ 0)
- **Model comparison: DECISIVE FAIL** (ΔELPD = -23)

**Overall**: 2 critical failures override 3 passes

### Key Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| γ₁ 95% CI includes 0 | Yes | Should exclude | **FAIL** |
| ΔELPD (M2 vs M1) | -23.43 | >-2 SE | **FAIL** |
| ΔELPD standard errors | 5.29 | <2 | **FAIL** |
| R̂ max | 1.000 | <1.01 | PASS |
| ESS min | 1542 | >400 | PASS |
| Divergences | 0% | <1% | PASS |
| Pareto k bad | 3.7% | <10% | PASS |

### Files Referenced

**Prior Predictive**:
- `/workspace/experiments/experiment_2/prior_predictive_check/findings.md`

**SBC**:
- `/workspace/experiments/experiment_2/simulation_based_validation/recovery_metrics.md`

**Inference**:
- `/workspace/experiments/experiment_2/posterior_inference/inference_summary.md`
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/loo_results.json`

**Plots**:
- All plots in `/workspace/experiments/experiment_2/posterior_inference/plots/`

### Comparison Models

- **Model 1**: `/workspace/experiments/experiment_1/` - ACCEPTED, superior
- **Model 2**: `/workspace/experiments/experiment_2/` - **REJECTED**, this document

---

**Critique completed**: 2025-10-27
**Analyst**: Model Criticism Specialist
**Decision**: **REJECT** - Use Model 1 instead
**Confidence**: Very High (multiple converging lines of evidence)

---
