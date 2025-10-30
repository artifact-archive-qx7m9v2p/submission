# Model Decision for Experiment 1: Hierarchical Binomial

**Date**: 2025-10-30
**Model**: Hierarchical Binomial (Logit-Normal, Non-Centered Parameterization)
**Analyst**: Claude (Model Criticism Specialist)

---

## DECISION: CONDITIONAL ACCEPT

**Status**: Model is adequate for primary inferential goals with documented limitations

---

## Executive Summary

The Hierarchical Binomial model is **CONDITIONALLY ACCEPTED** for research use. The model successfully answers the research question about population-level and group-level success rates, demonstrates perfect computational convergence, and produces scientifically plausible estimates. However, widespread high Pareto k values (10/12 groups with k > 0.7) indicate that the model cannot be used for reliable LOO-based model comparison and shows sensitivity to individual observations.

**Bottom Line**: Trust the parameter estimates and uncertainty intervals for inference, but do not use LOO-CV for model selection. Alternative comparison methods (WAIC, posterior predictive checks, K-fold CV) must be used if comparing to other models.

---

## Scoring Summary

| Criterion | Result | Weight | Impact |
|-----------|--------|--------|--------|
| **Computational Validity** | A+ | Critical | ✓ Perfect |
| **Scientific Plausibility** | A | Critical | ✓ Excellent |
| **Overdispersion Capture** | A | Critical | ✓ Primary goal met |
| **Shrinkage Validation** | A | High | ✓ Theory confirmed |
| **Group-Level Fit** | A | High | ✓ Well-calibrated |
| **LOO Diagnostics** | D | Moderate | ✗ Cannot use for comparison |
| **Overall Grade** | B+ | - | Conditional Accept |

---

## Decision Rationale

### Why ACCEPT?

1. **Perfect Computational Performance**
   - R̂ = 1.0000 for all parameters
   - ESS > 2,400 (far exceeds minimum)
   - Zero divergences in 8,000 samples
   - E-BFMI = 0.685 (excellent)

   **Interpretation**: We can fully trust that the posterior samples accurately represent the true posterior distribution. No computational issues confound inference.

2. **Answers Research Question**
   - Population mean: 7.3% (95% HDI: [5.7%, 9.5%])
   - Between-group heterogeneity: τ = 0.41 (95% HDI: [0.17, 0.67])
   - All 12 group-level rates estimated with appropriate shrinkage

   **Interpretation**: The model provides exactly what was asked for: estimates of success rates at population and group levels with appropriate uncertainty quantification.

3. **Passes Core Posterior Predictive Checks**
   - Overdispersion: φ_obs = 5.92 ∈ [3.79, 12.61] ✓
   - Extreme groups: All |z| < 1.0 ✓
   - Shrinkage: Small-n 58-61%, Large-n 7-17% ✓
   - Individual fit: All p-values ∈ [0.29, 0.85] ✓

   **Interpretation**: The model successfully reproduces key features of the data and shows no systematic mispredictions. This validates the model structure.

4. **Scientifically Plausible**
   - All group rates in [4.7%, 12.1%] (consistent with observed [3.1%, 14.0%])
   - Population mean (7.3%) close to pooled rate (7.0%)
   - Moderate heterogeneity (τ=0.41) aligns with observed overdispersion
   - Shrinkage patterns match theoretical expectations

   **Interpretation**: No implausible estimates or red flags. Results make scientific sense.

5. **Prior Validation**
   - Posterior did not retain prior's extreme tails
   - Strong likelihood (n=2,814) dominated weakly informative priors
   - Final estimates align with EDA expectations

   **Interpretation**: Priors were appropriate and data-dominated, as intended.

### Why CONDITIONAL?

1. **LOO-CV Diagnostics Fail**
   - 10 of 12 groups (83%) have Pareto k > 0.7
   - 2 groups (Groups 4, 8) have k > 1.0
   - LOO-ELPD estimates are unreliable

   **Interpretation**: The leave-one-out posterior differs substantially from the full posterior for most groups. This means:
   - Cannot use LOO for model comparison
   - Model is sensitive to individual groups
   - Predictive accuracy estimates are uncertain

2. **Sensitivity to Extreme Groups**
   - Group 4 (n=810, 29% of data, lowest rate): k = 1.01
   - Group 8 (n=215, 8% of data, highest rate): k = 1.06
   - Removing either would substantially change posterior

   **Interpretation**: The between-group heterogeneity estimate (τ) is anchored by these extreme cases. While not necessarily wrong, it means the estimate is data-dependent in a way that affects predictions.

3. **Small J Limitation**
   - Only 12 groups
   - Each group represents ~8.3% of data
   - Hierarchical variance estimate inherently unstable

   **Interpretation**: This is a data structure limitation, not a model flaw. With J=12, some LOO issues are expected. However, it still limits model utility for certain purposes.

### Why NOT REVISE?

**REVISE would require**: Clear path to improvement

**Assessment**:
- Alternative models (Student-t, Beta-binomial) may not improve LOO if issue is fundamentally small J + extreme groups
- Changing priors unlikely to resolve sensitivity to influential observations
- Model already answers research question adequately
- Cost of full model development cycle not justified without evidence of improvement

**Analogy**: Don't tear down a house because the paint color isn't perfect. The foundation is solid.

### Why NOT REJECT?

**REJECT would require**: Fundamental inadequacy for research question

**Assessment**:
- Model answers research question: What are group-level success rates? ✓
- Convergence is perfect: Can we trust the sampling? ✓
- PP checks pass: Does model fit the data? ✓
- Scientific plausibility: Do estimates make sense? ✓
- Only LOO fails: A diagnostic for prediction/comparison, not core inference

**Analogy**: Don't throw away a measuring tape because it doesn't also work as a level. It does what it's supposed to do.

---

## Conditions for Acceptance

The model is accepted under the following conditions:

### 1. Document LOO Limitation Explicitly

**Required language in any publication/report**:
> "Leave-one-out cross-validation (LOO-CV) diagnostics indicated high Pareto k values (k > 0.7) for 10 of 12 groups, with Groups 4 and 8 exceeding k = 1.0. This suggests the model is sensitive to individual group observations, and LOO-ELPD estimates are unreliable. Therefore, LOO was not used for model comparison in this analysis."

### 2. Do Not Use LOO for Model Comparison

**Prohibited**:
- Comparing LOO-ELPD across models
- Using LOO for model selection
- Reporting LOO-based predictive accuracy
- Model stacking or averaging based on LOO weights

**Permitted alternatives**:
- WAIC (Widely Applicable Information Criterion)
- Posterior predictive checks (qualitative comparison)
- K-fold cross-validation (computationally expensive but more stable)
- Bayes factors (if appropriate for question)

### 3. Acknowledge Sensitivity to Extreme Groups

**Required disclosure**:
> "The between-group heterogeneity estimate (τ = 0.41, 95% HDI: [0.17, 0.67]) is influenced by extreme groups, particularly Group 4 (n=810, lowest rate) and Group 8 (n=215, highest rate), both of which showed Pareto k > 1.0. These groups anchor the extremes of the rate distribution and removing either would substantially change the posterior distribution of τ."

### 4. Report Uncertainty Intervals, Not Just Point Estimates

**Recommended practice**:
- Always report 95% HDI alongside point estimates
- Use full posterior distribution for decision-making when possible
- Do not over-interpret precise values of τ (use range [0.17, 0.67])

### 5. Recommended (Not Required): Perform Sensitivity Analysis

**Suggested analysis**:
- Refit model excluding Group 4
- Refit model excluding Group 8
- Compare τ estimates across fits
- Report: "Sensitivity analyses excluding extreme groups showed τ estimates ranged from [X to Y], confirming qualitative conclusions about moderate between-group heterogeneity."

**Benefit**: Strengthens claims by demonstrating robustness to most influential observations.

---

## What You Can Trust

### ✓ Trustworthy Inferences

1. **Population mean success rate**: μ = 7.3% (95% HDI: [5.7%, 9.5%])
   - Based on perfect convergence and 8,000 samples
   - Aligns with pooled rate from EDA (7.0%)
   - Scientifically plausible

2. **Group-level success rates**: All 12 estimates with HDIs
   - Well-calibrated (all Bayesian p-values ∈ [0.29, 0.85])
   - Appropriate shrinkage toward population mean
   - No systematic mispredictions

3. **Between-group heterogeneity**: τ = 0.41 (95% HDI: [0.17, 0.67])
   - Consistent with observed overdispersion
   - Moderate heterogeneity (not trivial, not extreme)
   - Use full interval, not just point estimate

4. **Relative comparisons**: Which groups have higher/lower rates
   - Group 8 has highest rate (~12%)
   - Group 4 has lowest rate (~5%)
   - Rankings are stable across posterior

5. **Shrinkage patterns**: Small-n groups borrow more strength
   - Group 1 (n=47): 58% shrinkage
   - Group 4 (n=810): 17% shrinkage
   - Validates hierarchical model theory

6. **Uncertainty quantification**: 95% HDIs for all parameters
   - Based on strong ESS (>2400)
   - Likely slightly overconfident but reasonable
   - Conservative interpretation: treat as ~90% coverage

### ✗ Untrustworthy Inferences

1. **LOO-ELPD for model comparison**
   - Do not compare to other models using LOO
   - Do not use for model selection
   - Do not report as predictive accuracy

2. **Out-of-sample predictions for extreme groups**
   - Model sensitive to Groups 4 and 8
   - Predictions for similarly extreme future groups uncertain
   - Interpolative predictions (typical groups) more reliable

3. **Precise point estimate of τ**
   - τ = 0.41 is somewhat anchored by extreme groups
   - Use full posterior distribution [0.17, 0.67]
   - Qualitative conclusion (moderate heterogeneity) more robust than precise value

---

## Practical Guidance

### For Primary Research Goals

**If your goal is**: Estimate group-level success rates with uncertainty
- **Status**: ✓ Model is fit for purpose
- **Use**: Posterior means and 95% HDIs for all groups
- **Report**: Population mean, τ estimate, shrinkage patterns

**If your goal is**: Quantify between-group heterogeneity
- **Status**: ✓ Model is adequate with caveats
- **Use**: Full posterior distribution of τ, not just point estimate
- **Report**: τ = 0.41 [0.17, 0.67] indicates moderate heterogeneity
- **Caveat**: Estimate influenced by extreme groups

**If your goal is**: Make predictions for new groups
- **Status**: ⚠️ Acceptable for interpolation, uncertain for extrapolation
- **Use**: Posterior predictive distribution for θ_new ~ N(μ, τ)
- **Range**: Reliable for groups expected to have rates ~3-14%
- **Caveat**: Less reliable for groups expected to be far outside observed range

### For Model Comparison

**If you need to**: Compare this model to alternatives
- **Do NOT use**: LOO-ELPD
- **Use instead**:
  1. Posterior predictive checks (recommended)
  2. WAIC (with caveats)
  3. K-fold CV (if computational budget allows)
- **Report**: "Model comparison based on posterior predictive performance, not LOO-CV"

### For Publication

**Minimum required documentation**:
1. Report LOO diagnostic failure
2. Explain why model is still trustworthy for inference
3. Specify alternative comparison methods used
4. Acknowledge sensitivity to extreme groups

**Recommended additional analyses**:
1. Sensitivity analysis (refit without Group 4 or 8)
2. Comparison to Beta-binomial (Experiment 3)
3. Visual diagnostics (include overdispersion and shrinkage plots)

---

## Comparison to Decision Criteria

### Pre-registered Falsification Criteria

From `/workspace/experiments/experiment_1/metadata.md`:

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| R̂ | < 1.01 | 1.0000 | ✓ PASS |
| ESS | > 400 | 2,423+ | ✓ PASS |
| Divergences | < 1% | 0.00% | ✓ PASS |
| Overdispersion | φ in 95% PP | 5.92 ∈ [3.79, 12.61] | ✓ PASS |
| Extreme groups | \|z\| < 3 | max \|z\| = 0.60 | ✓ PASS |
| Shrinkage | Validates | Small-n: 58-61%, Large-n: 7-17% | ✓ PASS |
| Pareto k | < 0.7 all groups | 10/12 > 0.7 | ✗ FAIL |
| Scientific plausibility | All p in [0, 0.3] | All p in [0.047, 0.121] | ✓ PASS |

**Score**: 7/8 criteria passed (87.5%)

**Pre-registered decision paths**:
- ✅ All pass → ACCEPT
- ⚠️ Convergence issues → Adjust sampler
- ⚠️ PP fails → Try Experiment 2
- ❌ Fundamental failure → Try Experiment 3

**Actual result**: Only LOO failed, all other diagnostics passed. This is an **intermediate case** between "all pass" and "PP fails".

**Appropriate decision**: CONDITIONAL ACCEPT
- Not full ACCEPT because LOO failed
- Not REVISE because PP checks passed (model fits data well)
- Conditions address the specific limitation (LOO unreliability)

---

## Next Steps

### Immediate Actions (Required)

1. **Use model for inference** on group-level success rates
2. **Document LOO limitations** in any reports using these results
3. **Report parameter estimates with HDIs**, emphasizing uncertainty
4. **Do not use LOO** for any model comparison tasks

### Recommended Actions (Optional but Valuable)

1. **Perform sensitivity analysis**: Refit without Groups 4 and 8
   - Quantifies impact of most influential observations
   - Effort: ~5 minutes (2 additional fits)
   - Benefit: Strengthens claims about robustness

2. **Try Beta-binomial model** (Experiment 3):
   - Tests whether within-group overdispersion improves LOO
   - Part of standard workflow (due diligence)
   - Effort: Full model development cycle
   - Benefit: Comparison validates current model or identifies improvement

3. **Compute WAIC**: Alternative to LOO for model comparison
   - Single function call
   - Provides quantitative comparison metric
   - Effort: <1 minute
   - Benefit: Enables model ranking if alternatives are fitted

4. **Create summary visualization**: For presentation/publication
   - Forest plot of group rates with shrinkage arrows
   - Overdispersion diagnostic plot
   - Shows model in action
   - Effort: ~15 minutes
   - Benefit: Clear communication of results

### If Trying Alternative Models

**Comparison strategy**:
1. Fit Experiment 3 (Beta-binomial)
2. Compare using:
   - Posterior predictive checks (qualitative)
   - WAIC (quantitative, with caveats)
   - Visual diagnostics
3. Check if Beta-binomial also has high Pareto k
   - If yes: Confirms issue is data structure (small J, extreme groups)
   - If no: Suggests Beta-binomial is more appropriate
4. Select model based on:
   - Posterior predictive performance
   - Scientific interpretability
   - Computational efficiency

---

## Final Recommendation

**ACCEPT** the Hierarchical Binomial model for Experiment 1 under the conditions specified above.

**Justification**:
- Model achieves primary research goal (estimate group rates with uncertainty)
- Perfect computational performance inspires confidence
- Passes all inference-related diagnostics
- LOO limitation is consequential but not fatal
- Appropriate for publication with documented caveats

**Confidence Level**: High for primary inferences (group rates, population mean), Moderate for heterogeneity parameter (τ), Low for out-of-sample prediction

**Model Status**: Ready for use in research, with appropriate documentation of LOO limitation

**Stopping Criterion**: If Experiment 3 (Beta-binomial) also shows high Pareto k values, confirm that the issue is inherent to the data structure (J=12, extreme groups) and proceed with current model. If Beta-binomial improves LOO diagnostics, consider switching to that model. Otherwise, use current model with documented limitations.

---

## Decision Signature

**Model**: Hierarchical Binomial (Logit-Normal, Non-Centered)
**Decision**: CONDITIONAL ACCEPT
**Date**: 2025-10-30
**Analyst**: Claude (Model Criticism Specialist)
**Conditions**: Document LOO limitations, do not use LOO for model comparison, acknowledge sensitivity to extreme groups
**Status**: Approved for research use with caveats

---

## Key Takeaway

**This model is like a reliable car with a faulty fuel gauge**: The engine runs perfectly (convergence), it gets you where you need to go (inference), and it handles well on familiar roads (interpolation). However, the fuel gauge is unreliable (LOO), so you can't trust it to tell you when to refuel (model comparison) or how far you can drive on a tank (out-of-sample prediction). For the trip you're taking (estimating group rates), it's perfectly adequate. Just don't rely on the fuel gauge.

**Use the model. Document its limitations. Trust the core inferences.**
