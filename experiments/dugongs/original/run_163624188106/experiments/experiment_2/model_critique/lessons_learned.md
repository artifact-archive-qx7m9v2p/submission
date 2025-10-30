# Lessons Learned from Experiment 2: Log-Linear Heteroscedastic Model

**Date**: 2025-10-27
**Analyst**: Model Criticism Specialist
**Outcome**: Model REJECTED - Hypothesis not supported by data

---

## Executive Summary

Experiment 2 tested whether variance changes with x (heteroscedastic variance) vs. remains constant (homoscedastic variance). Despite perfect computational convergence, the model is decisively rejected: the heteroscedasticity parameter γ₁ ≈ 0 (95% CI includes zero), and the model performs 23 ELPD units worse than the simpler homoscedastic model in cross-validation.

**Key lesson**: This is not a modeling failure - this is science working correctly. We proposed a hypothesis, tested it rigorously, and found it unsupported. The negative result is valuable scientific knowledge.

---

## What We Learned About the Data

### Finding 1: Variance Is Constant Across x

**Hypothesis tested**: log(σᵢ) = γ₀ + γ₁ × xᵢ (variance changes with x)

**Result**: γ₁ = 0.003 ± 0.017, with 95% CI [-0.028, 0.039] including zero

**Interpretation**: The data provide **no credible evidence** that variance changes with x. The scatter in Y appears to be constant across the range of x values.

**Scientific meaning**: Whatever generates measurement uncertainty or natural variability in this system does so uniformly, not as a function of x. This is actually useful knowledge - it rules out mechanisms that would produce heteroscedastic variance.

**Practical implication**: We can use a simpler model (constant variance) without sacrificing accuracy.

---

### Finding 2: Simple Model Predicts Better

**Comparison**: Model 1 (3 parameters, constant variance) vs. Model 2 (4 parameters, variance as f(x))

**Result**: ΔELPD = 23.43 ± 4.43 in favor of Model 1 (>5 standard errors)

**Interpretation**: The extra parameter in Model 2 captures noise, not signal. This noise doesn't generalize to new data, resulting in worse predictions.

**Scientific meaning**: The data-generating process is simpler than we hypothesized. Adding complexity doesn't help - it hurts.

**Practical implication**: Occam's Razor is not just philosophical - it's empirically validated by cross-validation.

---

### Finding 3: Small Sample Size Was Adequate

**Concern**: N=27 might be too small to detect heteroscedasticity

**Result**: The 95% CI for γ₁ is tight enough (±0.017) to exclude meaningful effect sizes

**Interpretation**: We have sufficient data to detect that variance doesn't change substantially with x. The uncertainty is not "we don't have enough data" but rather "the effect, if present, is negligible."

**Scientific meaning**: 27 observations can provide decisive evidence when the effect is absent or very small.

**Practical implication**: Not all negative results require more data - sometimes the current data are sufficient to rule out an effect.

---

## What We Learned About Bayesian Workflow

### Success 1: Multi-Stage Validation Caught Issues Early

**Workflow stages**:
1. Prior predictive check: CONDITIONAL PASS (warned about variance ratio calibration)
2. SBC: CONDITIONAL PASS (warned about γ₁ bias and under-coverage)
3. Convergence: PASS (perfect MCMC)
4. Parameter estimates: FAIL (γ₁ ≈ 0)
5. LOO comparison: FAIL (much worse than Model 1)

**Pattern**: Warnings in early stages (prior predictive, SBC) foreshadowed rejection in later stages.

**Lesson**: The workflow is designed to catch problems before you commit to a model. **The warnings were prescient** - take them seriously.

**Best practice**: Don't ignore CONDITIONAL PASS results. They're telling you something about model fragility or complexity.

---

### Success 2: SBC Warnings Were Validated

**SBC findings**:
- γ₁ showed -11.97% bias
- Under-coverage (84.6% vs 90% target)
- 22% optimization failures
- Struggles when heteroscedasticity is weak

**Real data findings**:
- γ₁ ≈ 0 (heteroscedasticity IS weak/absent)
- Model overfits despite perfect convergence
- LOO shows model is too complex

**Connection**: The SBC struggles with γ₁ predicted that real data would not support strong heteroscedasticity. The computational fragility (22% failures) indicated unnecessary model complexity.

**Lesson**: SBC issues with a particular parameter often mean that parameter is either poorly identified or unnecessary. In this case, γ₁ was unnecessary (true value ≈ 0).

**Best practice**: When SBC shows bias or under-coverage for a specific parameter, ask: "Do I really need this parameter?" Consider simpler models without it.

---

### Success 3: LOO Provided Decisive Evidence

**Model comparison**:
- ΔELPD = 23.43 (huge difference)
- Standard error = 4.43 (tight, confident estimate)
- Ratio: 23.43 / 4.43 = 5.29 standard errors

**What this means**: This is not a close call requiring judgment. The evidence is overwhelming.

**Lesson**: LOO cross-validation can provide decisive evidence for model selection, not just vague guidance. When |ΔELPD| > 5 SE, the decision is clear.

**Best practice**: Always compare complex models to simpler baselines using LOO. The extra complexity must earn its keep by improving predictions.

---

### Success 4: Perfect Convergence Doesn't Mean Correct Model

**Convergence diagnostics**:
- All R̂ = 1.000 (perfect)
- All ESS > 1500 (excellent)
- 0 divergent transitions (clean)

**But the model was still rejected** because:
- Hypothesis not supported (γ₁ ≈ 0)
- Predictions worse than baseline
- Unnecessary complexity

**Lesson**: **Convergence ≠ Correctness**. A well-sampled posterior can still be the wrong posterior to sample from. Computational success is necessary but not sufficient.

**Best practice**: Don't stop at convergence diagnostics. Always check:
1. Do parameter estimates support your hypothesis?
2. Do predictions improve over simpler models?
3. Do residuals show expected patterns?

---

## What We Learned About Model Comparison

### Insight 1: Parsimony Is Predictively Superior

**Occam's Razor** (philosophical): Prefer simpler explanations when equally good.

**Our finding**: The simpler model isn't just "equally good" - it's **much better** (23 ELPD units).

**Mechanism**:
1. Model 2 has 4 parameters for N=27 observations
2. γ₁ ≈ 0 but still uses degrees of freedom
3. This "fits noise" that doesn't generalize
4. LOO correctly penalizes this overfitting

**Lesson**: Parsimony is not just aesthetic - it's statistically justified. Unnecessary parameters hurt predictive performance even when they "converge" to near-zero values.

**Best practice**: When adding parameters, verify they improve predictions, not just fit. Use LOO to check whether complexity is justified.

---

### Insight 2: p_loo Reveals Effective Complexity

**p_loo (effective number of parameters)**:
- Model 1: 2.43 (close to 3 actual parameters)
- Model 2: 3.41 (close to 4 actual parameters)

**Interpretation**: Both models use approximately all their parameters. But Model 2's 4th parameter (γ₁) doesn't help predictions.

**Lesson**: p_loo can reveal whether parameters are earning their keep. Large p_loo relative to number of parameters suggests overfitting.

**Best practice**: Check p_loo in addition to ΔELPD. If p_loo increases substantially without improving ELPD, you're overfitting.

---

### Insight 3: Pareto k Diagnostics Matter

**Pareto k values**:
- Model 1: 0/27 problematic (0%)
- Model 2: 1/27 problematic (3.7%, with k = 0.96)

**Interpretation**: The complex model introduces instability that the simple model doesn't have. One observation has unreliable LOO estimate in Model 2.

**Lesson**: Complex models can be more sensitive to individual observations. Pareto k issues often indicate model misspecification or overfitting.

**Best practice**: Always check Pareto k diagnostics. If complex model has more issues than simple model, that's a red flag.

---

## What We Learned About Scientific Inference

### Principle 1: Negative Results Are Valuable

**Common misconception**: A "failed" model is wasted effort.

**Reality**: We learned that variance is constant, ruling out a competing hypothesis.

**Value**:
1. Rules out heteroscedastic mechanisms
2. Justifies simpler model for future work
3. Informs future experimental design
4. Prevents others from testing same hypothesis

**Lesson**: **This is not a failed experiment - this is successful hypothesis testing.** We tested whether γ₁ ≠ 0 and found strong evidence against it.

**Best practice**: Frame rejections as scientific findings, not failures. Report: "We found no evidence for heteroscedastic variance" rather than "the complex model didn't work."

---

### Principle 2: Let Data Override Theory

**Prior belief**: Variance might decrease with x (γ₁ < 0)
- Prior mean: -0.05
- Prior SD: 0.05
- Prior P(γ₁ < 0) ≈ 84%

**Posterior finding**: Variance is constant (γ₁ ≈ 0)
- Posterior mean: 0.003
- Posterior SD: 0.017
- Posterior P(γ₁ < 0) = 43.9%

**Lesson**: The data "pushed back" against the prior, moving it from negative values toward zero. This is Bayesian inference working correctly - the likelihood overwhelmed the prior.

**Best practice**: Use priors that encode domain knowledge but remain flexible enough for data to contradict them. If posteriors are very different from priors, investigate why.

---

### Principle 3: Hypothesis Testing via Model Comparison

**Traditional approach**: Test H₀: γ₁ = 0 vs. H₁: γ₁ ≠ 0

**Our approach**: Compare Model 1 (γ₁ = 0 by construction) vs. Model 2 (γ₁ estimated)

**Advantage**: Model comparison via LOO incorporates both parameter uncertainty AND predictive performance, not just statistical significance.

**Result**: Model 1 is strongly preferred (ΔELPD = 23.43), providing evidence for the null hypothesis of constant variance.

**Lesson**: Bayesian model comparison via cross-validation is more informative than traditional hypothesis tests. It tells you not just "is the effect zero?" but "does modeling the effect improve predictions?"

**Best practice**: Frame scientific questions as model comparisons with LOO, not just parameter significance tests.

---

## What We Learned About Model Complexity

### When Complex Models Fail

**Three ways complex models can fail**:

1. **Computational failure**: Model doesn't converge
   - **Model 2 status**: ✓ No issues (R̂ = 1.000)

2. **Scientific failure**: Hypothesis not supported by data
   - **Model 2 status**: ✗ γ₁ ≈ 0, no evidence for heteroscedasticity

3. **Predictive failure**: Worse predictions than simpler models
   - **Model 2 status**: ✗ ΔELPD = -23.43

**Model 2 failed on 2 of 3 criteria**, even though it passed the computational test.

**Lesson**: **Computational success is necessary but not sufficient.** You must also check scientific coherence and predictive performance.

**Best practice**: Evaluate models on three axes:
1. Does it converge? (computational)
2. Do results make sense? (scientific)
3. Does it predict well? (statistical)

---

### When to Add Complexity

**Model 2 added complexity** by modeling variance as f(x).

**When this makes sense**:
- Theory predicts heteroscedastic variance
- Residuals show clear funnel patterns
- Prior data exhibit this behavior
- Cross-validation shows improvement

**Our case**: ✗ None of these were true

**Lesson**: Don't add complexity "just in case" or because it's "theoretically possible." Add complexity when:
1. Theory or prior data suggest it's needed
2. Diagnostics reveal problems in simpler models
3. Cross-validation validates the improvement

**Best practice**: Start simple, add complexity only when justified. Use LOO to verify each addition helps.

---

### The Overfitting Mechanism

**How Model 2 overfits despite γ₁ ≈ 0**:

1. **Training phase**: γ₁ fits to random noise in training data
2. **Parameter estimate**: γ₁ ≈ 0 on average, but varies by fold in cross-validation
3. **Prediction phase**: The "learned noise" doesn't transfer to test data
4. **Result**: Worse predictions than assuming γ₁ = 0 always (Model 1)

**Lesson**: Even parameters that "converge to zero" can overfit. They consume degrees of freedom fitting noise that doesn't generalize.

**Best practice**: Parameters with posteriors centered near zero are candidates for removal. Use LOO to check if they're helping or hurting.

---

## What We Learned About Communication

### Framing Rejections Positively

**Poor framing**: "The heteroscedastic model failed."

**Better framing**: "We tested for heteroscedastic variance and found no evidence, supporting the simpler constant-variance model."

**Best framing**: "Cross-validation strongly favors the simpler model (ΔELPD = 23.43 ± 4.43), and the heteroscedasticity parameter γ₁ = 0.003 ± 0.017 is consistent with zero, indicating constant variance. We therefore use the parsimonious constant-variance model for inference."

**Lesson**: Frame rejections as scientific findings, not failures. You learned something about the data-generating process.

**Best practice**: In publications, include rejected models in methods as "tested but not supported" rather than hiding them. This demonstrates scientific rigor.

---

### Explaining Model Comparison

**For technical audiences**:
"Leave-one-out cross-validation (LOO-CV) using the expected log pointwise predictive density (ELPD) metric shows Model 1 is 23.43 units better (SE = 4.43), a decisive preference (>5 SE)."

**For general audiences**:
"We compared models by testing how well each predicts new data. The simpler model makes substantially better predictions, indicating the added complexity doesn't help."

**For stakeholders**:
"The more complex model doesn't improve accuracy and is harder to use, so we recommend the simpler model."

**Lesson**: Tailor explanation to audience, but always emphasize predictive performance, not just fit.

**Best practice**: Use phrases like "predicts better" rather than "has lower WAIC" - focus on what matters to the audience.

---

## What We Learned About Priors

### Were the Priors the Problem?

**Prior for γ₁**: N(-0.05, 0.05)

**Could tighter priors have "saved" the model?**

**Option 1**: γ₁ ~ N(-0.10, 0.02) (force decreasing variance)

**Problem**: This is forcing a conclusion the data don't support. Scientific dishonesty.

**Option 2**: γ₁ ~ N(-0.05, 0.10) (more flexible)

**Problem**: Current prior is already flexible (posterior moved to zero despite prior preference for negative). More flexibility wouldn't change conclusion.

**Lesson**: The prior wasn't the problem - the data genuinely don't support heteroscedastic variance. Changing priors to force a conclusion would be unethical.

**Best practice**: When posteriors contradict hypotheses, don't blame the prior (if it was reasonable). Accept what the data tell you.

---

### Prior Predictive Check Warnings

**Variance ratio distribution**:
- Median: 21x (reasonable)
- Mean: 1322x (heavy right tail)
- Observed: 8.8x

**Warning**: Heavy tails indicate potential numerical issues.

**What happened**: No numerical issues in MCMC, but model still rejected on other grounds.

**Lesson**: Prior predictive warnings about variance calibration were valid but not fatal. The model worked computationally but failed scientifically.

**Best practice**: CONDITIONAL PASS means "proceed with caution." Monitor carefully, but don't necessarily stop.

---

## What We Learned About Workflow Design

### Multi-Stage Validation Works

**Experiment 2 workflow**:
1. Prior predictive → CONDITIONAL PASS
2. SBC → CONDITIONAL PASS WITH WARNINGS
3. Fitting → PASS
4. Parameter estimates → FAIL
5. LOO comparison → FAIL

**Pattern**: Each stage provides independent evidence. When multiple stages raise concerns, take them seriously.

**Lesson**: The workflow is designed with multiple "checkpoints" precisely for cases like this. A model can pass early stages but still fail later.

**Best practice**: Don't skip stages. Each provides complementary information:
- Prior predictive: Are priors reasonable?
- SBC: Can model recover truth?
- Fitting: Does MCMC work?
- Parameters: Are hypotheses supported?
- LOO: Does complexity help predictions?

---

### When to Stop Early vs. Continue

**Should we have stopped after SBC warnings?**

**Arguments for stopping**:
- 22% failure rate suggested fragility
- γ₁ bias indicated identifiability issues
- Under-coverage warned of calibration problems

**Arguments for continuing** (what we did):
- Need to see if real data support hypothesis
- SBC warnings might not apply to real data
- Computational issues might not occur with full MCMC

**Result**: We continued and found decisive rejection. This was the right call.

**Lesson**: CONDITIONAL PASS means "proceed but monitor carefully," not "stop." The point of validation is to learn, and we learned by proceeding.

**Best practice**: Continue through workflow even with warnings, but:
- Use conservative sampler settings (we used target_accept=0.97)
- Monitor diagnostics carefully
- Compare to simpler baselines
- Be prepared to reject if issues persist

---

## Practical Guidelines for Future Work

### Checklist for Testing Complex Models

Before adding complexity (like heteroscedastic variance):

- [ ] **Theory predicts it**: Do we have scientific reason to expect this?
- [ ] **Data suggest it**: Do residual plots show patterns requiring it?
- [ ] **Prior data show it**: Has heteroscedasticity been observed before?
- [ ] **Simpler model inadequate**: Does the baseline model have clear problems?

**Model 2 status**: ✗ None of these were true

If you check these boxes:

- [ ] **Run SBC**: Can the model recover parameters?
- [ ] **Check convergence**: Does MCMC work properly?
- [ ] **Test hypothesis**: Are key parameters away from zero?
- [ ] **Compare to baseline**: Does LOO favor the complex model?

**Model 2 status**: ✓✓✗✗ (passed first two, failed second two)

**Decision rule**:
- If ≥3 of 4 comparison checks pass: Consider complex model
- If ≤2 of 4 comparison checks pass: **Use simple model** (our case)

---

### Red Flags for Overparameterization

**Signs your model is too complex**:

1. **Parameters near zero**: Posterior ≈ 0 suggests unnecessary
   - Model 2: γ₁ = 0.003 ± 0.017 ✓ Red flag

2. **Worse LOO**: ΔELPD < -5 is decisive
   - Model 2: ΔELPD = -23.43 ✓ Red flag

3. **Higher p_loo**: Extra parameters without benefit
   - Model 2: p_loo = 3.41 vs 2.43 ✓ Red flag

4. **More Pareto k issues**: Complex model less stable
   - Model 2: 3.7% vs 0% ✓ Red flag

5. **SBC calibration issues**: Struggles to recover parameters
   - Model 2: γ₁ under-coverage, bias ✓ Red flag

**Model 2 status**: 5 of 5 red flags

**Action**: If ≥3 red flags, strongly consider simpler model.

---

### When Negative Results Should Be Published

**Our case**: γ₁ ≈ 0, no evidence for heteroscedastic variance

**Publishable because**:
1. Tests a clear hypothesis (heteroscedastic variance)
2. Uses rigorous Bayesian workflow
3. Provides decisive evidence (95% CI includes zero)
4. Rules out competing explanation
5. Has practical implications (use simpler model)

**How to frame**:
- "We tested whether variance depends on x using a heteroscedastic model..."
- "The heteroscedasticity parameter γ₁ = 0.003 ± 0.017 is consistent with zero..."
- "Cross-validation strongly favors the simpler constant-variance model..."
- "We conclude that variance is constant and use the parsimonious model."

**Lesson**: Negative results from rigorous workflows are valuable and should be reported.

**Best practice**: Include model comparison in publications to demonstrate you considered alternatives.

---

## Key Takeaways

### Top 10 Lessons from Experiment 2

1. **Convergence ≠ Correctness**: Perfect MCMC doesn't mean correct model
2. **Negative results are valuable**: We learned variance is constant
3. **LOO is decisive**: ΔELPD = 23.43 (>5 SE) provides clear evidence
4. **Parsimony is predictive**: Simpler model predicts better, not just easier
5. **SBC warnings were prescient**: Early concerns validated by later failures
6. **Parameters ≈ 0 are red flags**: γ₁ ≈ 0 suggests unnecessary complexity
7. **Multi-stage validation works**: Each stage provided complementary evidence
8. **Model comparison > parameter significance**: LOO more informative than CI
9. **Let data override theory**: Posterior shifted from prior toward zero
10. **Frame rejections positively**: This is successful hypothesis testing

---

### What to Do Differently Next Time

**Before fitting complex models**:
- Check if residuals from simple model show patterns requiring complexity
- Review prior data/literature for evidence supporting complexity
- Set explicit falsification criteria (like we did: γ₁ ≠ 0, ΔELPD > 0)

**During fitting**:
- Always fit simpler baseline for comparison
- Use LOO routinely, not just when problems appear
- Monitor not just convergence but also parameter magnitudes

**After fitting**:
- Compare to baselines even if focal model looks good
- Check if parameters are far from null values
- Be willing to accept simpler models when data support them

**When reporting**:
- Include model comparison in methods
- Frame negative results as scientific findings
- Report LOO differences with standard errors
- Emphasize predictive performance, not just fit

---

### Success Criteria for Future Experiments

**A complex model is justified when**:

1. **Hypothesis supported**: Key parameters exclude null values
   - Model 2: ✗ γ₁ 95% CI includes zero

2. **Predictions improve**: ΔELPD > 2 SE vs. baseline
   - Model 2: ✗ ΔELPD = -23.43

3. **Diagnostics clean**: No issues absent in simpler model
   - Model 2: ✗ Pareto k issue introduced

4. **Complexity warranted**: p_loo increase justified by ELPD gain
   - Model 2: ✗ p_loo higher, ELPD worse

5. **Scientific coherence**: Results make theoretical sense
   - Model 2: ✗ Homoscedasticity simpler/more plausible

**Model 2 failed all five criteria** → Clear rejection

**Lesson**: Use this as a template for future model decisions.

---

## Broader Implications

### For the Field

**Workflow validation works**: This case study demonstrates that multi-stage Bayesian workflow catches inadequate models before they're used for inference.

**Negative results matter**: Reporting that heteroscedastic variance is NOT supported helps future researchers avoid testing the same hypothesis.

**LOO is underused**: Model comparison via cross-validation should be standard practice, not optional.

---

### For Scientific Practice

**Hypothesis testing via modeling**: Model comparison (Model 1 vs Model 2) is rigorous hypothesis testing (constant vs changing variance).

**Parsimony is empirical**: Occam's Razor is not just philosophical - it's validated by predictive performance.

**Computational ≠ Statistical success**: A well-sampled posterior can still be from the wrong model. Check science, not just computation.

---

### For Future Research

**Open questions**:
1. Under what sample sizes does LOO reliably detect overfitting?
2. How do we set thresholds for "meaningful" ΔELPD differences?
3. Can we develop better priors that are both flexible and informative?
4. How do we communicate model comparison to non-technical audiences?

**Methodological contributions**:
1. Demonstrated multi-stage validation catching issues
2. Showed SBC warnings predicting real-data problems
3. Illustrated decisive model comparison via LOO
4. Provided template for reporting negative results

---

## Conclusion

Experiment 2 was **not a failure** - it was **successful hypothesis testing**. We rigorously tested whether variance changes with x and found decisive evidence that it doesn't. This negative result is valuable scientific knowledge that:

1. Rules out heteroscedastic variance in this system
2. Justifies simpler model for future work
3. Demonstrates the power of Bayesian workflow
4. Validates parsimony as predictively superior
5. Shows that multi-stage validation catches issues

**The workflow worked exactly as designed**: early warnings (SBC) were validated by later failures (γ₁ ≈ 0, poor LOO), leading to the correct scientific conclusion (reject complex model, use simple one).

**The lesson**: Follow the evidence, not your preferences. When data don't support a hypothesis, accept it and report the negative finding. That's good science.

---

**Document date**: 2025-10-27
**Analyst**: Model Criticism Specialist
**Status**: Final
**Key lesson**: Negative results from rigorous workflows are valuable scientific contributions

---

## Appendix: Quick Reference

### Model 2 at a Glance

| Aspect | Result | Interpretation |
|--------|--------|----------------|
| **γ₁ estimate** | 0.003 ± 0.017 | Essentially zero |
| **γ₁ 95% CI** | [-0.028, 0.039] | Includes zero |
| **P(γ₁ < 0)** | 43.9% | No directional evidence |
| **ΔELPD vs M1** | -23.43 ± 4.43 | Much worse (>5 SE) |
| **Convergence** | R̂ = 1.000 | Perfect |
| **Decision** | REJECT | Use Model 1 |

### Files and Locations

**Critique documents**:
- `/workspace/experiments/experiment_2/model_critique/critique_summary.md` - Comprehensive assessment
- `/workspace/experiments/experiment_2/model_critique/decision.md` - REJECT justification
- `/workspace/experiments/experiment_2/model_critique/lessons_learned.md` - This document

**Validation results**:
- Prior predictive: `/workspace/experiments/experiment_2/prior_predictive_check/findings.md`
- SBC: `/workspace/experiments/experiment_2/simulation_based_validation/recovery_metrics.md`
- Inference: `/workspace/experiments/experiment_2/posterior_inference/inference_summary.md`
- LOO: `/workspace/experiments/experiment_2/posterior_inference/diagnostics/loo_results.json`

**Comparison model**:
- Model 1: `/workspace/experiments/experiment_1/` - **ACCEPTED, use this**

---
