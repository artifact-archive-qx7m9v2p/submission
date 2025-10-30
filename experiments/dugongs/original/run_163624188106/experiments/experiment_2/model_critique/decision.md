# Model Decision: Experiment 2 - Log-Linear Heteroscedastic Model

**Date**: 2025-10-27
**Analyst**: Model Criticism Specialist

---

## DECISION: REJECT

**The Log-Linear Heteroscedastic Model (Experiment 2) is REJECTED and should NOT be used for inference or prediction.**

---

## Summary Statement

Despite achieving perfect computational convergence (R̂ = 1.000, ESS > 1500, 0 divergences), the heteroscedastic variance model fails on two critical grounds:

1. **Scientific Hypothesis Not Supported**: The heteroscedasticity parameter γ₁ = 0.003 ± 0.017 has a 95% credible interval [-0.028, 0.039] that includes zero, providing no evidence that variance changes with x.

2. **Predictive Performance Much Worse**: Leave-one-out cross-validation shows Model 2 is 23.43 ELPD units worse than the simpler Model 1, with this difference exceeding 5 standard errors (ΔELPD = -23.43 ± 4.43).

**Conclusion**: The model tests a hypothesis the data don't support, adds unnecessary complexity, and predicts worse than its simpler alternative. This is a decisive rejection.

---

## Recommendation

**USE MODEL 1 (Log-Linear Homoscedastic Model) INSTEAD**

Model 1 is superior on every meaningful criterion:
- ELPD LOO: 46.99 vs 23.56 (Model 1 is +23.43 units better)
- Pareto k: 0% problematic vs 3.7% problematic
- Parameters: 3 vs 4 (simpler)
- Scientific support: Matches data-generating process
- Interpretability: Constant variance is easier to communicate

---

## Decision Framework Applied

### ACCEPT Criteria (None Met)
- ✗ Strong evidence for hypothesized effect (γ₁ ≠ 0)
- ✗ Better or equal predictive performance vs baselines
- ✗ Justification for added complexity

### REVISE Criteria (Not Applicable)
- No fixable issues - the model answered the question correctly (γ₁ ≈ 0)
- No structural problems - the model worked as designed
- No computational issues - convergence was perfect

### REJECT Criteria (Multiple Met)
- ✓ **Core hypothesis falsified** (γ₁ ≈ 0, no heteroscedasticity)
- ✓ **Much worse predictive performance** (ΔELPD = -23.43)
- ✓ **Overfitting evident** (p_loo = 3.41 vs 2.43)
- ✓ **Principle of parsimony violated** (unnecessary complexity)

**Result**: REJECT is the only appropriate decision.

---

## Falsification Criteria Assessment

The experiment design specified four falsification criteria. Status of each:

### Criterion 1: Gamma_1 Posterior Includes Zero ✓ TRIGGERED

**Threshold**: For model acceptance, γ₁ 95% CI should exclude zero.

**Result**:
- γ₁ = 0.003 ± 0.017
- 95% CI = [-0.028, 0.039] **includes zero**
- P(γ₁ < 0) = 43.9% (essentially 50/50)

**Interpretation**: The data provide NO evidence for heteroscedastic variance. The posterior is consistent with γ₁ = 0 (homoscedastic case).

**Implication**: **Primary scientific hypothesis FALSIFIED.**

---

### Criterion 2: LOO Shows Overfitting ✓ TRIGGERED

**Threshold**: ΔELPD < -10 indicates serious overfitting.

**Result**:
- ΔELPD (Model 2 vs Model 1) = **-23.43 ± 4.43**
- This is **5.29 standard errors** below zero
- Model 2 is decisively worse

**Interpretation**: The added complexity of modeling variance as f(x) doesn't improve predictions - it actively degrades them. This is textbook overfitting: the model fits noise in the training data that doesn't generalize.

**Implication**: **Model fails predictive performance test.**

---

### Criterion 3: Convergence Issues ✗ NOT TRIGGERED

**Threshold**: R̂ > 1.01 or ESS < 400

**Result**:
- All R̂ = 1.000
- All ESS > 1500
- 0 divergent transitions

**Interpretation**: The model has no computational problems. MCMC sampling worked perfectly.

**Implication**: We're NOT rejecting due to technical failure - we're rejecting because the model worked perfectly and showed us it's wrong.

---

### Criterion 4: LOO Diagnostics Problems ✗ MINOR CONCERN

**Threshold**: >10% observations with Pareto k > 0.7

**Result**:
- 1/27 observations (3.7%) has k = 0.96
- Below the 10% threshold but concerning

**Interpretation**: One observation has unreliable LOO estimate in Model 2, but none in Model 1. This suggests Model 2 is less stable.

**Implication**: Minor concern, but doesn't change decision (already rejected on other grounds).

---

## Evidence Synthesis

### Converging Lines of Evidence for Rejection

**1. Parameter Estimates (γ₁ ≈ 0)**
- Point estimate: 0.003 (essentially zero)
- Posterior SD: 0.017 (not small relative to estimate)
- No directional preference: P(γ₁ < 0) = 43.9%

**2. Credible Intervals**
- 95% CI: [-0.028, 0.039] includes zero
- 90% CI: [-0.025, 0.031] includes zero
- 80% CI: [-0.019, 0.025] includes zero
- Even narrow intervals include homoscedastic case

**3. Visual Evidence**
- Variance function plot: essentially flat across x
- Residual plots: no funnel pattern
- Posterior distribution for γ₁: centered at zero

**4. Model Comparison**
- LOO heavily favors Model 1
- Difference of 23 ELPD units is huge (>5 SE)
- No reasonable interpretation supports Model 2

**5. SBC Foreshadowing**
- γ₁ showed -12% bias in recovery
- Under-coverage suggested identifiability issues
- 22% optimization failures indicated fragility
- Real data confirmed these warnings

**6. Principle of Parsimony**
- Model 2 uses 4 parameters, Model 1 uses 3
- Extra parameter provides zero benefit
- Occam's Razor strongly favors Model 1

### No Counterevidence

**What would need to be true for acceptance:**
- γ₁ 95% CI excludes zero: **FALSE**
- ΔELPD > 0 (or at least > -2 SE): **FALSE**
- Residuals show heteroscedasticity: **FALSE**
- Scientific theory requires heteroscedasticity: **FALSE**

**Conclusion**: All evidence points in the same direction - REJECT.

---

## Why Revision Would Not Help

Could we revise Model 2 to make it acceptable? **No, for fundamental reasons:**

### Revision Option 1: Tighter Priors on γ₁

**Proposal**: Use γ₁ ~ N(-0.10, 0.02) to force decreasing variance.

**Problem**: This would be scientific dishonesty. We'd be forcing a conclusion the data don't support. The current prior (SD = 0.05) is already fairly informative, and the posterior still centered at 0 means the data override the prior's preference for γ₁ < 0.

**Verdict**: Unacceptable.

---

### Revision Option 2: Different Variance Function

**Proposal**: Try quadratic, exponential, or other functional forms.

**Problem**: Specification searching / p-hacking. If linear log-variance doesn't work, why would other forms? The data show no pattern in variance across x. Trying multiple forms until one "works" is data dredging.

**Verdict**: Unacceptable.

---

### Revision Option 3: Reparameterization

**Proposal**: Use centered or non-centered parameterization.

**Problem**: Convergence is already perfect (R̂ = 1.000). Reparameterization might help computational issues, but we don't have computational issues. The problem is conceptual, not computational.

**Verdict**: Won't help.

---

### Revision Option 4: More Data

**Proposal**: Collect more observations to increase power.

**Problem**:
1. We already have N=27, which is adequate to detect the lack of heteroscedasticity
2. LOO suggests fundamental model misspecification, not just low power
3. More data would likely strengthen evidence against heteroscedasticity
4. Not a revision of the model anyway

**Verdict**: Won't change conclusion.

---

### Revision Option 5: Different Likelihood

**Proposal**: Use Student-t instead of Normal for robustness.

**Problem**:
1. Residuals look normally distributed (Q-Q plot is fine)
2. Doesn't address the heteroscedasticity issue
3. Model 1 with Normal likelihood is already adequate
4. Adding degrees of freedom parameter would increase complexity further

**Verdict**: Addresses wrong problem.

---

### Why Revision Is Inappropriate Here

**The fundamental issue**: Model 2 tests a hypothesis (heteroscedastic variance) that the data don't support. This is NOT a modeling error - this is the correct scientific conclusion from a properly fitted model.

**What revision assumes**: There's a "true" heteroscedastic model that we just haven't found yet.

**What the data tell us**: Variance is constant (or at least, any change with x is too small to detect and model).

**Conclusion**: Revising Model 2 to force heteroscedasticity would be fitting our desires, not the data. The appropriate action is to REJECT Model 2 and USE Model 1, which correctly captures the homoscedastic data-generating process.

---

## Comparison: Model 1 vs Model 2

### Quantitative Comparison

| Criterion | Model 1 (Homoscedastic) | Model 2 (Heteroscedastic) | Better Model |
|-----------|------------------------|---------------------------|--------------|
| **ELPD LOO** | **46.99 ± 3.11** | 23.56 ± 3.15 | **Model 1** by 23.43 |
| **Pareto k issues** | 0% | 3.7% | **Model 1** |
| **p_loo (complexity)** | 2.43 | 3.41 | **Model 1** (simpler) |
| **Parameters** | 3 | 4 | **Model 1** (simpler) |
| **R̂ max** | 1.000 | 1.000 | Tie |
| **ESS min** | >1600 | >1500 | Tie |
| **Divergences** | 0 | 0 | Tie |
| **Interpretability** | Simple | Complex | **Model 1** |
| **Scientific support** | Data match model | Hypothesis unsupported | **Model 1** |

**Score**: Model 1 wins on 7/9 criteria, ties on 2/9, loses on 0/9.

### Qualitative Comparison

**Model 1 advantages**:
- Matches data-generating process (constant variance)
- Better out-of-sample predictions
- Simpler to communicate to stakeholders
- More defensible scientifically
- More parsimonious (Occam's Razor)

**Model 2 advantages**:
- None. It has no advantages over Model 1.

**Model 2 only makes sense if**:
- Heteroscedastic variance existed: **It doesn't**
- It predicted better: **It doesn't**
- It was required by theory: **It isn't**
- Diagnostic issues required it: **They don't**

---

## Implications for Inference and Prediction

### If We Incorrectly Used Model 2

**What we would report**:
- γ₁ = 0.003 ± 0.017 (which we'd correctly interpret as "no evidence for heteroscedasticity")
- Wider credible intervals for predictions (due to extra parameter uncertainty)
- More complex model structure to defend

**Problems**:
- Predictions would be worse (ΔELPD = -23.43)
- Would need to explain why we used a more complex model that the data don't support
- Reviewers would (correctly) ask: "Why not use the simpler model?"
- Would appear to be overfitting or p-hacking

### If We Correctly Use Model 1

**What we would report**:
- Simple log-linear relationship: Y = β₀ + β₁ log(x) + ε, ε ~ N(0, σ²)
- Constant variance σ² ≈ exp(-2.4) ≈ 0.09
- Better predictions for new data
- Clean, defensible model selection process

**Advantages**:
- Simpler communication
- Better science (following evidence)
- More accurate predictions
- Easier to defend and replicate

---

## Decision Rationale

### Why REJECT Is Appropriate

**1. Scientific Falsification**
- Primary hypothesis (γ₁ ≠ 0) is not supported
- P(γ₁ < 0) = 43.9% provides no directional evidence
- Posterior consistent with null hypothesis (γ₁ = 0)

**2. Statistical Inadequacy**
- Predictive performance much worse than baseline
- ΔELPD = -23.43 is decisive (>5 standard errors)
- Overfitting evident despite perfect convergence

**3. Philosophical Inconsistency**
- Violates principle of parsimony
- Adds complexity without benefit
- Simpler explanation (constant variance) is superior

**4. Practical Considerations**
- Model 1 is easier to use and communicate
- No scientific or statistical reason to prefer Model 2
- Model 2 would be difficult to defend in peer review

### Why ACCEPT Is Inappropriate

To accept Model 2, we would need:
- Strong evidence for γ₁ ≠ 0: **We have the opposite**
- Better or equal predictions: **23 ELPD units worse**
- Compelling scientific reason: **None exists**
- Diagnostic issues requiring it: **Model 1 has better diagnostics**

None of these conditions are met.

### Why REVISE Is Inappropriate

Revision implies fixable problems, but:
- There are no computational issues to fix
- The model answered the scientific question correctly (γ₁ ≈ 0)
- Forcing heteroscedasticity would be unscientific
- The simpler alternative (Model 1) already exists and is superior

Revision would be misguided - we should use Model 1, not modify Model 2.

---

## Lessons for Future Model Development

### What This Decision Teaches Us

**1. Convergence ≠ Correctness**
- Model 2 had perfect MCMC convergence
- But it's still the wrong model for these data
- Computational success doesn't imply scientific adequacy

**2. Test Hypotheses, Don't Assume Them**
- We hypothesized heteroscedastic variance
- We tested it rigorously
- We found it unsupported
- We rejected the hypothesis
- **This is good science**

**3. Simpler Is Often Better**
- Model 1 with 3 parameters beats Model 2 with 4
- Parsimony is not just aesthetically pleasing - it's predictively superior
- Occam's Razor is validated quantitatively by LOO

**4. LOO Is A Powerful Reality Check**
- Caught overfitting that parameter estimates alone might miss
- Provided decisive evidence for model comparison
- Validated our concerns about unnecessary complexity

**5. Listen to SBC Warnings**
- The γ₁ calibration issues in SBC foreshadowed real-data problems
- 22% optimization failures suggested model fragility
- Under-coverage indicated weak identifiability
- These warnings were prescient

### Guidelines for Future Modeling

**When considering complex models**:
1. Have strong prior reason to believe added complexity is needed
2. Compare to simpler baselines using LOO/cross-validation
3. Check if parameters of interest exclude null values
4. Be willing to accept simpler models when data support them
5. Remember: negative results (hypothesis not supported) are valuable

**Red flags for overfitting**:
- Parameters with posteriors centered near zero
- Much worse LOO than simpler alternatives
- p_loo much larger than expected
- Pareto k issues in complex but not simple model

**When to reject a converged model**:
- Core hypothesis not supported by data (this case)
- Predictive performance worse than baselines (this case)
- Unnecessary complexity without benefit (this case)
- Prior-data conflict that can't be resolved

---

## Final Decision Summary

**MODEL 2 (LOG-LINEAR HETEROSCEDASTIC) IS REJECTED**

### Decision Type
- [x] **REJECT** - Model is inadequate and should not be used
- [ ] REVISE - Model has fixable issues
- [ ] ACCEPT - Model is adequate for inference

### Confidence Level
**Very High** - Multiple converging lines of evidence, decisive statistical tests

### Primary Reasons
1. Scientific hypothesis falsified (γ₁ ≈ 0)
2. Predictive performance much worse (ΔELPD = -23.43)
3. Unnecessary complexity (4 parameters vs 3)
4. Simpler alternative (Model 1) is superior

### Recommended Alternative
**USE MODEL 1 (Log-Linear Homoscedastic Model)**
- Location: `/workspace/experiments/experiment_1/`
- Status: ACCEPTED
- Advantages: Better predictions, simpler, matches data

### This Decision Is
- **Decisive**: Not a close call or judgment call
- **Rigorous**: Based on multiple validation stages
- **Scientific**: Follows evidence, not preferences
- **Practical**: Recommends superior alternative
- **Defensible**: Would withstand peer review

---

## Action Items

### Immediate Actions

1. **Do NOT use Model 2** for any inference, prediction, or reporting
2. **Use Model 1 instead** for all analyses
3. **Archive Model 2 results** for documentation purposes
4. **Report the negative finding**: No evidence for heteroscedastic variance

### Documentation

1. Include Model 2 in methods as "tested but rejected"
2. Report LOO comparison showing Model 1 superiority
3. Report γ₁ ≈ 0 as scientific finding (homoscedastic variance)
4. Cite this critique document for detailed justification

### Scientific Communication

**How to frame this**:
"We tested whether variance changes with x by fitting a heteroscedastic model (Model 2). The heteroscedasticity parameter γ₁ = 0.003 ± 0.017 has a 95% credible interval [-0.028, 0.039] that includes zero, providing no evidence for heteroscedastic variance. Moreover, leave-one-out cross-validation strongly favors the simpler homoscedastic model (ΔELPD = 23.43 ± 4.43 in favor of Model 1). We therefore conclude that variance is constant across the range of x and use the simpler homoscedastic model for inference."

This frames the rejection as a **scientific finding** (homoscedastic variance) rather than a **modeling failure**.

---

## Approval

**Decision**: REJECT Model 2, use Model 1 instead

**Justification**: Scientific hypothesis not supported, predictive performance much worse, unnecessary complexity

**Confidence**: Very High

**Date**: 2025-10-27

**Analyst**: Model Criticism Specialist

---

## Appendix: Decision Criteria Checklist

### REJECT Criteria (✓ = Met, Primary Reasons)

- [x] ✓ **Core hypothesis falsified** - γ₁ ≈ 0, no heteroscedasticity
- [x] ✓ **Much worse predictive performance** - ΔELPD = -23.43
- [x] ✓ **Overfitting evident** - p_loo increase without benefit
- [x] ✓ **Principle of parsimony violated** - unnecessary complexity
- [x] ✓ **Superior alternative exists** - Model 1 is better
- [ ] Fundamental computational issues - N/A, convergence perfect
- [ ] Unresolvable prior-data conflict - N/A
- [ ] Cannot reproduce key data features - N/A, fit is adequate

### REVISE Criteria (All Unmet)

- [ ] Fixable computational issues - Convergence is perfect
- [ ] Minor specification problems - No specification issues
- [ ] Prior needs adjustment - Prior is appropriate
- [ ] Likelihood needs change - Likelihood is appropriate
- [ ] Clear path to improvement - No improvement possible

### ACCEPT Criteria (All Unmet)

- [ ] Strong evidence for hypotheses - γ₁ ≈ 0 contradicts hypothesis
- [ ] Good predictive performance - 23 ELPD units worse
- [ ] Justified complexity - Complexity not justified
- [ ] No concerning diagnostics - Pareto k issue introduced
- [ ] Scientific coherence - Hypothesis not supported

**Conclusion**: REJECT is the only appropriate decision. No criteria for ACCEPT or REVISE are met.

---

**Document version**: 1.0
**Status**: FINAL
**Decision**: **REJECT**

---
