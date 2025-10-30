# Model Selection Recommendation

**Date**: 2025-10-30
**Models Compared**: Experiment 1 (Hierarchical Binomial) vs Experiment 3 (Beta-Binomial)
**Status**: FINAL RECOMMENDATION

---

## Executive Recommendation

### **CHOOSE EXPERIMENT 3 (Beta-Binomial Model)**

Both models adequately fit the data, but Experiment 3 is the clear winner based on:

1. **Equivalent Predictive Performance**: ΔELPD = -1.5 ± 3.7 (0.4×SE) → statistically indistinguishable
2. **Dramatically Superior LOO Reliability**: 0/12 bad Pareto k (vs 10/12 for Exp1)
3. **7× Greater Parsimony**: 2 parameters vs 14
4. **15× Faster Computation**: 6 seconds vs 90 seconds
5. **Simpler Interpretation**: Probability scale vs logit scale
6. **Perfect Diagnostics**: Passes all 5 posterior predictive checks (vs 4/5 for Exp1)

---

## The Decisive Factor: LOO Reliability

**Visual Evidence**: See `plots/pareto_k_detailed_comparison.png` and `plots/comprehensive_comparison.png` Panel B

### Experiment 1 (Hierarchical):
- **10 out of 12 groups** (83%) have Pareto k > 0.7 ("bad")
- **2 groups** (Groups 4 and 8) have Pareto k > 1.0 ("very bad")
- **LOO estimates are UNRELIABLE** and cannot be trusted for model comparison

### Experiment 3 (Beta-Binomial):
- **0 out of 12 groups** have Pareto k ≥ 0.7
- All groups have k < 0.5 ("good")
- **LOO estimates are RELIABLE** and trustworthy

**Implication**: While Exp1 shows nominally better predictive performance (ELPD = -38.8 vs -40.3), we cannot trust this estimate due to unreliable diagnostics. Exp3's estimate is trustworthy and shows the models are statistically equivalent.

---

## Key Visual Evidence

### 1. Pareto k Comparison (`pareto_k_detailed_comparison.png`)
Shows group-by-group LOO reliability with color coding:
- **Exp1**: 10 red/dark red bars (bad/very bad k)
- **Exp3**: 12 green bars (all good k)

**Message**: Exp3's LOO is reliable across all groups; Exp1's is not.

### 2. Comprehensive Comparison (`comprehensive_comparison.png`)
Four-panel dashboard revealing:
- **Panel A**: Models equivalent in ELPD (bars overlap within error bars)
- **Panel B**: Exp3 dominates in Pareto k (all bars below threshold)
- **Panel C**: Exp3 is 7× simpler and 15× faster
- **Panel D**: Exp1's problematic groups (red points) vs Exp3's stable estimates

**Message**: Exp3 wins on reliability, parsimony, and speed while maintaining equivalent predictive accuracy.

### 3. Spider Plot (`model_trade_offs_spider.png`)
Multi-criteria radar plot showing:
- **Exp3**: Dominates in LOO reliability (10/10), simplicity (10/10), speed (10/10), parsimony (10/10)
- **Exp1**: Only competitive in predictive accuracy (~5/10, tied with Exp3)

**Message**: Exp3 is superior across 4 of 5 assessment dimensions.

---

## When to Use Each Model

### Use Exp3 (Beta-Binomial) When:

**Recommended for most applications**, especially if:

✓ Research question is **population-level**
  - "What is the overall success rate?"
  - "Is there evidence of overdispersion?"

✓ Need **reliable model comparison**
  - LOO-CV must be trustworthy
  - Planning model selection or averaging

✓ Want **simple, fast, interpretable** analysis
  - Easier to communicate to non-statisticians
  - Faster iteration for model checking
  - Probability scale (no transformations)

✓ Need **robust predictions**
  - Less sensitive to individual groups
  - More stable out-of-sample performance

✓ **Publication robustness** is priority
  - No caveats about unreliable diagnostics
  - Cleaner methods section

### Use Exp1 (Hierarchical) Only When:

**Use ONLY if group-specific inference is essential**, specifically when:

✓ Research question is **group-specific**
  - "What is Group 4's success rate?"
  - "Which groups differ significantly?"
  - "How much heterogeneity exists between groups?"

✓ Need **explicit heterogeneity quantification**
  - Estimate between-group variance (τ)
  - Visualize shrinkage patterns
  - Understand partial pooling

✓ Want to **predict for new groups**
  - Use hierarchical structure for new group forecasting
  - Interested in group-level generalization

✓ **Willing to accept LOO limitations**
  - Can document unreliable diagnostics
  - Will use alternative comparison methods (WAIC, K-fold CV)
  - Understand sensitivity to extreme groups (Groups 4, 8)

---

## Decision Framework

```
START: What is your research question?
  │
  ├─ Population-level summary? ────────────────→ USE EXP3
  │
  ├─ Group-specific estimates needed? ─────────→ USE EXP1
  │                                              (Accept LOO limitations)
  │
  ├─ Model comparison essential? ──────────────→ USE EXP3
  │                                              (Reliable LOO required)
  │
  ├─ Prediction/cross-validation focus? ───────→ USE EXP3
  │                                              (Trustworthy diagnostics)
  │
  └─ Uncertain / Exploratory? ─────────────────→ USE BOTH
                                                 (Compare perspectives)
```

---

## Quantitative Comparison Summary

| Criterion | Exp1 (Hierarchical) | Exp3 (Beta-Binomial) | Winner |
|-----------|---------------------|----------------------|--------|
| **Parameters** | 14 | 2 | **Exp3** (7× simpler) |
| **Effective Params (p_loo)** | 8.27 | 0.61 | **Exp3** (13× more parsimonious) |
| **Sampling Time** | 90 sec | 6 sec | **Exp3** (15× faster) |
| **ELPD LOO** | -38.76 ± 2.94 | -40.28 ± 2.19 | Exp1 (but unreliable) |
| **ΔELPD** | — | -1.51 ± 3.67 | **EQUIVALENT** (0.4×SE) |
| **Pareto k > 0.7** | 10/12 (83%) | 0/12 (0%) | **Exp3** (DRAMATICALLY) |
| **Pareto k > 1.0** | 2/12 | 0/12 | **Exp3** |
| **LOO Reliability** | UNRELIABLE | RELIABLE | **Exp3** |
| **PPC Tests Passed** | 4/5 | 5/5 | **Exp3** |
| **WAIC Warning** | Yes | No | **Exp3** |
| **Convergence (R̂)** | 1.0000 | 1.0000 | Tie (both perfect) |
| **ESS min** | 2,423 | 2,371 | Tie (both adequate) |
| **Divergences** | 0 | 0 | Tie (both perfect) |
| **Interpretation** | Logit scale | Probability scale | **Exp3** (simpler) |
| **Group Estimates** | Yes | No | Exp1 (if needed) |

**Overall Winner**: **Experiment 3** (wins 9 criteria, ties 3, loses 1)

---

## What the Models Tell Us

### Exp3 (Beta-Binomial) Results:

**Population-level success rate**: μ_p = 8.4% (95% CI: [6.8%, 10.3%])

**Overdispersion**: κ = 14.6 (95% CI: [7.3, 27.9])
- Lower κ indicates more variation between groups
- Implies groups vary according to Beta(μ_p × κ, (1-μ_p) × κ)

**Interpretation**:
> "The overall success rate is approximately 8.4% [6.8%, 10.3%]. There is substantial between-group variation (κ = 14.6), indicating groups differ in their success rates beyond binomial sampling variability."

**Key Insight**: Captures essential feature (overdispersion) with minimal complexity.

### Exp1 (Hierarchical) Results:

**Population-level mean**: μ = -2.62 (logit scale) → 7.3% (95% CI: [5.7%, 9.5%])

**Between-group heterogeneity**: τ = 0.41 (95% CI: [0.17, 0.67]) on logit scale

**Group-specific rates** (examples):
- Group 4: 4.7% [3.8%, 5.8%] (largest group, lowest rate)
- Group 8: 12.1% [9.5%, 15.3%] (highest rate)
- Group 10: 5.8% [2.9%, 10.9%] (small sample, shrinks toward μ)

**Interpretation**:
> "The population mean success rate is 7.3% [5.7%, 9.5%]. Between-group standard deviation is τ = 0.41 [0.17, 0.67] on the logit scale, indicating moderate heterogeneity. Individual group rates range from 4.7% to 12.1% after accounting for partial pooling."

**Key Insight**: Provides richer inferential structure but at cost of reliability and complexity.

---

## The Core Trade-off

**Exp1 trades reliability for detail**:
- Gain: Group-specific estimates, explicit heterogeneity (τ), shrinkage patterns
- Lose: LOO reliability, computational speed, simplicity, robustness

**Exp3 trades detail for reliability**:
- Gain: Perfect LOO, parsimony, speed, robustness, ease of interpretation
- Lose: Group-specific inference, explicit between-group variance decomposition

**For most research questions, reliability and simplicity outweigh the loss of group-specific detail.**

---

## Scientific Interpretation

Both models agree on the fundamental conclusion:
1. **Population success rate is approximately 7-8%**
   - Exp1: 7.3% [5.7%, 9.5%]
   - Exp3: 8.4% [6.8%, 10.3%]
   - Difference is within uncertainty

2. **Strong evidence of overdispersion**
   - Exp1: τ = 0.41 [0.17, 0.67] (moderate between-group variation)
   - Exp3: κ = 14.6 [7.3, 27.9] (low concentration = high variation)
   - Both reject pooled model (χ² = 39.47, p < 0.0001 from EDA)

3. **Groups vary substantially**
   - Observed range: 3.1% to 14.0% (4.5-fold)
   - Both models capture this variation
   - Exp1: Via group-specific θ_j
   - Exp3: Via Beta distribution over groups

**Substantive conclusion is robust to modeling choice.**

---

## If You Must Choose One Sentence

**"Choose Experiment 3 (Beta-Binomial) for its equivalent predictive performance, dramatically superior LOO reliability (0/12 vs 10/12 bad Pareto k), 7× greater parsimony, and 15× faster computation—use Experiment 1 only if group-specific rate estimates are essential."**

---

## Implementation Guidance

### Reporting Exp3 (Recommended)

**Methods**:
> "We fit a Beta-Binomial model to the data using Bayesian inference via PyMC. The model estimates a population-level success rate (μ_p) and concentration parameter (κ), allowing for overdispersion beyond binomial variance. Priors were weakly informative: μ_p ~ Beta(2, 18), κ ~ Gamma(0.01, 0.01). We used 4 chains of 1,000 draws each, with convergence assessed via R̂ < 1.01 and ESS > 400. Model adequacy was evaluated via posterior predictive checks and leave-one-out cross-validation (LOO-CV)."

**Results**:
> "The estimated population success rate was 8.4% (95% credible interval: [6.8%, 10.3%]). The concentration parameter (κ = 14.6 [7.3, 27.9]) indicated substantial between-group variation. LOO-CV diagnostics confirmed model reliability (all 12 groups had Pareto k < 0.5). Posterior predictive checks showed the model adequately captured observed overdispersion (p = 0.74) and individual group outcomes (all p-values > 0.30)."

**No caveats required.**

### Reporting Exp1 (If Used)

**Methods**:
> "We fit a Bayesian hierarchical binomial model with group-specific success rates (θ_j) drawn from a common logit-normal distribution with population mean (μ) and between-group standard deviation (τ). The model used non-centered parameterization for computational efficiency. Priors were weakly informative: μ ~ Normal(-2.5, 1), τ ~ Half-Cauchy(0, 1). We used 4 chains of 2,000 draws each, with excellent convergence (R̂ = 1.00, ESS > 2,400, zero divergences)."

**Results**:
> "The estimated population mean success rate was 7.3% (95% credible interval: [5.7%, 9.5%]), with between-group standard deviation τ = 0.41 [0.17, 0.67] on the logit scale. Group-specific rates ranged from 4.7% to 12.1% after accounting for hierarchical shrinkage. The model passed posterior predictive checks for overdispersion (p = 0.73) and individual group fit (all p-values ∈ [0.29, 0.85])."

**Required caveat**:
> "Leave-one-out cross-validation (LOO-CV) diagnostics indicated high Pareto k values (k > 0.7) for 10 of 12 groups, including 2 groups with k > 1.0 (Groups 4 and 8). This suggests the model is sensitive to these influential observations, and LOO estimates should be interpreted with caution. Therefore, LOO-CV was not used for model comparison; adequacy was assessed via posterior predictive checks instead."

---

## Final Recommendation

**PRIMARY**: Use **Experiment 3 (Beta-Binomial)** for:
- Population-level inference ✓
- Reliable model comparison ✓
- Robust predictions ✓
- Simple, fast analysis ✓
- Publication without caveats ✓

**ALTERNATIVE**: Use **Experiment 1 (Hierarchical)** only if:
- Group-specific estimates essential ✓
- Heterogeneity quantification (τ) needed ✓
- Willing to document LOO limitations ✓

**BEST PRACTICE**: Report Exp3 as primary, mention Exp1 in sensitivity analysis:
> "Primary analysis used a Beta-Binomial model for population-level inference. As a sensitivity check, we also fit a hierarchical binomial model with group-specific parameters, which yielded qualitatively consistent conclusions (population rate 7.3% vs 8.4%, within overlapping credible intervals)."

---

## Confidence in Recommendation

**HIGH CONFIDENCE** (90%+) that Exp3 is the correct choice for:
- Researchers wanting population-level summaries
- Applications requiring model comparison
- Practitioners needing robust, interpretable results

**MODERATE CONFIDENCE** (70%) that Exp1 should be avoided for:
- Model comparison (unreliable LOO)
- Applications where simplicity matters
- Audiences unfamiliar with hierarchical models

**LOW CONFIDENCE** (40%) in recommending Exp1 even when group-specific estimates are needed:
- Could use Exp3 plus post-hoc group-level analysis
- Could report both models for complementary insights
- Trade-off between detail and reliability is context-dependent

**Overall**: Exp3 is the safer, more defensible choice for the vast majority of research applications.

---

## Questions to Guide Your Choice

Ask yourself:

1. **"Do I need to report individual group success rates?"**
   - Yes → Exp1
   - No → Exp3

2. **"Will I use LOO for model comparison?"**
   - Yes → Exp3 (only reliable option)
   - No → Either (but Exp3 still simpler)

3. **"Is computational speed important?"**
   - Yes → Exp3 (15× faster)
   - No → Either

4. **"Will non-statisticians read my results?"**
   - Yes → Exp3 (easier to explain)
   - No → Either

5. **"Do I want the simplest adequate model?"**
   - Yes → Exp3
   - No → Exp1 (but justify added complexity)

**If you answered "Exp3" to 3+ questions**, use Exp3.

---

## Analyst's Bottom Line

After comprehensive comparison across 13 dimensions (LOO, WAIC, Pareto k, parsimony, speed, interpretability, PPC, convergence, etc.), the evidence overwhelmingly supports **Experiment 3** as the superior choice for these data.

The only compelling reason to use Experiment 1 is the explicit need for group-specific parameter estimates—and even then, the unreliable LOO diagnostics should give pause.

**Trust the simpler model.** It works, it's reliable, and it's easier to defend.

---

**Recommendation Status**: FINAL
**Confidence Level**: HIGH
**Date**: 2025-10-30
**Analyst**: Model Assessment Specialist (Claude Agent SDK)

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════╗
║                  MODEL SELECTION DECISION                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  RECOMMEND: EXPERIMENT 3 (Beta-Binomial)                     ║
║                                                               ║
║  Rationale:                                                   ║
║    • Equivalent predictive performance (ΔELPD within 2×SE)   ║
║    • Perfect LOO reliability (0/12 bad k vs 10/12)           ║
║    • 7× simpler (2 vs 14 parameters)                         ║
║    • 15× faster (6 vs 90 seconds)                            ║
║    • Easier interpretation (probability scale)               ║
║                                                               ║
║  Use Exp1 ONLY IF:                                           ║
║    • Need group-specific rate estimates (essential)          ║
║    • Can document and accept LOO limitations                 ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
```
