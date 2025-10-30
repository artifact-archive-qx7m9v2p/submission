# Comprehensive Experiment Plan
## Bayesian Models for Eight Schools Meta-Analysis

**Generated:** 2025-10-28
**Based on:** Synthesis of three parallel independent model designers
**EDA Summary:** Strong evidence for homogeneity (Q p=0.696, I²=0%, tau²=0), pooled effect 7.69±4.07

---

## Synthesis Overview

Three independent designers proposed complementary model classes:
- **Designer 1 (Hierarchical Specialist):** Focus on parameterization and prior philosophy for hierarchical models
- **Designer 2 (Pooling Strategist):** Focus on pooling spectrum from complete to none
- **Designer 3 (Robustness Tester):** Focus on distributional robustness and sensitivity

**Key Convergence:** All three designers agree:
1. EDA evidence for homogeneity is strong
2. Hierarchical models will likely show tau ≈ 0
3. Complete pooling is well-supported
4. With n=8, power to detect moderate heterogeneity is limited

**Key Divergence:** Designers differ on:
- Which prior for tau (Designer 1: Half-Cauchy(0,10), Designer 2: multiple variants, Designer 3: sensitivity grid)
- Whether robustness is needed (Designer 3 skeptical, others didn't emphasize)
- Parameterization details (Designer 1 proposed adaptive approach)

---

## Prioritized Model Classes

Following the guidelines, I synthesize the proposals into a prioritized list of distinct model classes,removing duplicates and ordering by theoretical justification:

### **Model 1: Standard Non-Centered Hierarchical Model (PRIORITY 1)**
**Source:** Designer 1, Model 1; Designer 2, Variant 3a
**Rationale:** Gold standard for hierarchical meta-analysis, widely validated

```
Likelihood: y_i ~ Normal(theta_i, sigma_i)  [sigma_i known]
Hierarchy:  theta_i = mu + tau * eta_i
            eta_i ~ Normal(0, 1)
Priors:     mu ~ Normal(0, 20)
            tau ~ Half-Cauchy(0, 5)
```

**Why prioritize:**
- Standard approach with extensive literature support
- Non-centered parameterization handles tau ≈ 0 well
- Half-Cauchy(0,5) is Gelman et al. (2006) recommendation
- Expected to work well given EDA

**Falsification criteria:**
- Computational failure: divergences >1%, R-hat >1.01, ESS(tau) <100
- tau posterior median >15 (exceeds observed SD=10.4)
- <90% of y_i in 95% posterior predictive intervals
- Shrinkage inconsistencies (high-precision schools shrink more than low-precision)

**Expected outcome:** mu ≈ N(7.7, 4), tau with mode near 0-2 and median 2-4, strong shrinkage (70-90%)

---

### **Model 2: Complete Pooling Model (PRIORITY 2)**
**Source:** Designer 2, Model 1; Designer 1 as alternative
**Rationale:** Maximum parsimony, EDA strongly supports this

```
Likelihood: y_i ~ Normal(mu, sigma_i)  [sigma_i known]
Prior:      mu ~ Normal(0, 25)
```

**Why prioritize:**
- Simplest possible model (1 parameter)
- Directly supported by EDA (tau²=0, I²=0%)
- If hierarchical model reduces to this, use this instead
- Serves as baseline for all comparisons

**Falsification criteria:**
- Posterior predictive p-value <0.05 for max(|z_i|)
- >2 schools fall outside 95% posterior predictive intervals
- LOO-PIT shows U-shape (underdispersed)
- Systematic residual pattern by precision

**Expected outcome:** mu ≈ N(7.7, 4.1), all schools covered within prediction intervals

---

### **Model 3: Skeptical Hierarchical Model (PRIORITY 3)**
**Source:** Designer 1, Model 2; Designer 2, Variant 3b
**Rationale:** Prior reflects EDA evidence for homogeneity

```
Likelihood: y_i ~ Normal(theta_i, sigma_i)
Hierarchy:  theta_i = mu + tau * eta_i
            eta_i ~ Normal(0, 1)
Priors:     mu ~ Normal(8, 5)       [Informed by EDA]
            tau ~ Half-Normal(0, 3)  [Skeptical of heterogeneity]
```

**Why prioritize:**
- Tests whether skeptical prior (informed by EDA) is appropriate
- More concentrated tau prior should yield tighter posteriors
- Natural sensitivity check vs. Model 1

**Falsification criteria:**
- Prior dominates data: KL(posterior || prior) <0.1 for mu
- tau posterior truncated at upper bound of prior
- School 1 or 3 outside 95% posterior predictive interval
- LOO substantially worse than Model 1 (diff >5)

**Expected outcome:** mu ≈ N(7.7, 3-4), tau more concentrated near 0 (median 1-3), stronger shrinkage (80-95%)

---

### **Model 4: No Pooling Model (PRIORITY 4)**
**Source:** Designer 2, Model 2
**Rationale:** Lower bound on performance, tests independence assumption

```
Likelihood: y_i ~ Normal(theta_i, sigma_i)
Priors:     theta_i ~ Normal(0, 25)  [independent]
```

**Why include:**
- Establishes worst-case baseline (no information sharing)
- Expected to lose badly, confirming pooling is beneficial
- Useful for computing shrinkage relative to no pooling

**Falsification criteria:**
- LOO worse than complete pooling (expected)
- Posterior predictive for new school excessively wide
- theta_i posteriors barely differ from y_i (no learning)
- LOO effective parameter count ≈8

**Expected outcome:** theta_i ≈ N(y_i, 0.96*sigma_i), very wide intervals, poor prediction

---

### **Model 5: Student-t Robust Hierarchical (PRIORITY 5)**
**Source:** Designer 3, Model 1A
**Rationale:** Test robustness to tail assumptions

```
Likelihood: y_i ~ StudentT(nu, theta_i, sigma_i)
Hierarchy:  theta_i = mu + tau * eta_i
            eta_i ~ Normal(0, 1)
Priors:     mu ~ Normal(0, 20)
            tau ~ Half-Cauchy(0, 5)
            nu ~ Gamma(2, 0.1)       [Mean=20, allows 4-40+]
```

**Why include:**
- Tests whether normal likelihood is appropriate
- If nu posterior >30, validates normality assumption
- Provides robustness check given School 1's apparent extremeness

**Falsification criteria:**
- nu posterior >30 with 95% CI >20 (abandons Student-t, use normal)
- No LOO improvement over Model 1
- Computational difficulties
- High posterior correlation between nu and tau

**Expected outcome:** nu ≈ 20-40, mu and tau similar to Model 1, validates normal assumption

---

### Additional Models (Lower Priority, Conditional)

**Model 6: Prior Sensitivity Analysis (Designer 3, Model 3)**
- Not a single model but systematic grid search
- Fit Models 1-3 with multiple prior variants
- Compute sensitivity metrics (Relative_Sensitivity)
- **Implement if:** Models 1-3 disagree substantially OR to quantify prior sensitivity
- **Skip if:** Models 1-3 agree and computational resources limited

**Model 7: Mixture/Outlier Models (Designer 3, Model 2)**
- Outlier indicator or latent class models
- **Implement only if:** Student-t (Model 5) shows evidence of issues OR Models 1-4 fail PPCs
- **Expected:** These will find K=1 cluster (no mixture needed)

**Model 8: Adaptive Non-Centered (Designer 1, Model 3)**
- Adaptive parameterization for boundary regime
- **Implement only if:** Model 1 has computational issues OR for methodological comparison
- **Expected:** Similar results to Model 1 but potentially better ESS

---

## Minimum Attempt Policy Compliance

Per guidelines, we must attempt at least the first TWO models unless Model 1 fails pre-fit validation:

**Required:** Models 1 and 2 (Standard Hierarchical + Complete Pooling)
**Strongly Recommended:** Model 3 (Skeptical Hierarchical) for sensitivity
**Conditional:** Models 4-5 based on results of 1-3
**Optional:** Models 6-8 based on time/resources/findings

---

## Implementation Order and Decision Points

### Phase A: Core Models (Required - ~3 hours)

**Step 1:** Fit Model 1 (Standard Hierarchical)
- Prior predictive checks
- Posterior inference with convergence diagnostics
- Posterior predictive checks
- Compute LOO-CV
- **Decision Point:** If computational failure, try Model 8 (adaptive). If Model 8 fails, document and proceed to Model 2.

**Step 2:** Fit Model 2 (Complete Pooling)
- Prior predictive checks
- Posterior inference
- Posterior predictive checks
- Compute LOO-CV
- **Decision Point:** Compare LOO with Model 1. If Model 2 wins by >2 ELPD, complete pooling is preferred.

**Step 3:** Fit Model 3 (Skeptical Hierarchical)
- Same workflow as Model 1
- Compare tau posteriors between Models 1 and 3
- **Decision Point:** If posteriors similar (medians within 50%), low prior sensitivity. If differ >2x, prior choice matters.

**Checkpoint A:** After Models 1-3, assess:
- Do all three succeed computationally? (expected: yes)
- Do Models 1 and 3 agree on tau < 5? (expected: yes)
- Does any model fail PPC? (expected: no)
- Is Model 2 LOO competitive with Models 1/3? (expected: yes, within 2-3 ELPD)

---

### Phase B: Baselines and Robustness (Conditional - ~2 hours)

**Step 4:** Fit Model 4 (No Pooling)
- **If:** Models 1-3 all agree, this is mostly confirmatory
- **Purpose:** Quantify benefit of pooling
- **Expected:** LOO substantially worse, confirming pooling value

**Step 5:** Fit Model 5 (Student-t)
- **If:** Models 1-3 all pass PPCs, this is mostly validation
- **Purpose:** Confirm normal likelihood appropriate
- **Expected:** nu posterior >30, no improvement in LOO

**Checkpoint B:** After Models 4-5, assess:
- Does no pooling lose as expected? (confirms exchangeability)
- Does Student-t validate normality? (nu >30)
- Any surprises that require further investigation?

---

### Phase C: Extensions (Optional - ~2-4 hours)

**Step 6:** Prior Sensitivity Grid (Model 6)
- **If:** Models 1 and 3 disagreed substantially
- **Or:** Want to quantify prior robustness formally
- **Purpose:** Compute Relative_Sensitivity metrics

**Step 7:** Mixture Models (Model 7)
- **Only if:** Student-t showed issues (nu <10) OR PPCs failed
- **Purpose:** Test for latent clusters
- **Expected:** These won't be needed

**Step 8:** Adaptive Parameterization (Model 8)
- **Only if:** Model 1 had computational issues OR methodological interest
- **Purpose:** Improve sampling efficiency
- **Expected:** Not needed

---

## Falsification Criteria Summary

### Success Criteria for Hierarchical Approach (Models 1, 3, 5, 8)
- Convergence: R-hat <1.01, ESS >400 for all parameters, <1% divergences
- Plausibility: tau posterior median <10, mu posterior near 7.7
- Prediction: >90% of y_i in 95% posterior predictive intervals
- Shrinkage: High-precision schools shrink less than low-precision schools

### Success Criteria for Complete Pooling (Model 2)
- Posterior predictive p-value >0.05 for test statistics
- <2 schools outside 95% posterior predictive intervals
- Residuals show no systematic pattern by precision
- LOO competitive with hierarchical models

### Pivot Triggers (Abandon Hierarchical Models)
- All hierarchical models show tau firmly at boundary (tau <0.1) → use complete pooling instead
- Computational failure across models despite reparameterization → consider alternative framework
- Posterior predictive failure across all models → investigate data quality or model structure
- Between-model variance exceeds within-model variance → report uncertainty, consider model averaging

### Red Flags (Reconsider Everything)
- Prior-posterior conflict across all model classes
- Extreme parameter values (tau >20, nu <3)
- Computational breakdown in simple models
- Strong outliers emerge in residual analysis despite EDA showing none

---

## Expected Outcomes and Predictions

### Most Likely Scenario (80% confidence):
- **Models 1, 2, 3:** All converge successfully
- **tau posteriors:** Model 1 median≈2-4, Model 3 median≈1-3, both concentrated near 0
- **mu posteriors:** All three very similar, ≈N(7.7, 4)
- **LOO comparison:** Model 2 slightly favored or ties with Model 1/3 (differences <3 ELPD)
- **Shrinkage:** Strong (70-90%) in Models 1/3
- **Model 5 (Student-t):** nu posterior >30, validates normality
- **Recommendation:** Use Model 2 (complete pooling) for parsimony, report Model 1 as sensitivity showing minimal heterogeneity

### Alternative Scenario A (15% confidence): Hierarchical Preferred
- **Models 1, 3:** tau posteriors concentrated in [2, 6]
- **LOO:** Models 1 or 3 win by >2 ELPD over Model 2
- **Interpretation:** Modest heterogeneity detected
- **Recommendation:** Use Model 1 or 3 (hierarchical), report that heterogeneity is small but detectable

### Alternative Scenario B (5% confidence): Strong Prior Sensitivity
- **Models 1, 3:** tau posteriors differ substantially (Model 1 median≈5, Model 3 median≈1)
- **LOO:** Models 1, 2, 3 all similar (within 2 ELPD)
- **Interpretation:** n=8 insufficient to resolve pooling degree
- **Recommendation:** Report range of estimates, acknowledge fundamental uncertainty, may need Model 6 (sensitivity grid)

### Surprise Scenario C (<1% confidence): EDA Was Misleading
- **Model 5:** nu posterior <10 (heavy tails detected)
- **Or:** Mixture models find K=2 clusters
- **Or:** All models fail posterior predictive checks
- **Interpretation:** Fundamental model misspecification or data issue
- **Action:** Investigate data quality, consult domain experts, pivot to different model class

---

## Reporting Structure

### Primary Report Will Include:
1. **Model 1 (Standard Hierarchical):** Full specification, diagnostics, posteriors, PPCs, LOO
2. **Model 2 (Complete Pooling):** Full specification, diagnostics, posteriors, PPCs, LOO
3. **Model comparison:** LOO table, effective parameter counts, ΔELPD with SE
4. **Recommendation:** Which model for inference, with justification
5. **Shrinkage analysis:** Plot of posterior theta_i vs observed y_i
6. **Posterior predictive checks:** Visual and quantitative

### Sensitivity Analyses (Supplementary):
1. **Model 3:** Compare tau posteriors with Model 1, assess prior sensitivity
2. **Model 5:** Report nu posterior, validate normality assumption
3. **Model 4:** Quantify benefit of pooling (shrinkage relative to no pooling)

### If Needed:
1. **Model 6:** Sensitivity metrics, prior robustness plots
2. **Models 7, 8:** Only if invoked by decision points

---

## Computational Budget

| Phase | Models | Estimated Time | Cumulative |
|-------|--------|----------------|------------|
| Phase A | Models 1-3 | 3 hours | 3 hours |
| Phase B | Models 4-5 | 2 hours | 5 hours |
| Phase C | Models 6-8 (conditional) | 2-4 hours | 7-9 hours |
| Synthesis & Reporting | - | 2 hours | 9-11 hours |

**Target:** Complete Phases A-B (core + baselines) within 5-6 hours
**Minimum:** Complete Phase A (Models 1-3) within 3 hours per guidelines

---

## File Organization

```
experiments/
├── experiment_plan.md                 [This document]
├── iteration_log.md                   [To be created during implementation]
├── experiment_1/                      [Model 1: Standard Hierarchical]
│   ├── metadata.md
│   ├── prior_predictive_check/
│   ├── posterior_inference/
│   ├── posterior_predictive_check/
│   └── model_critique/
├── experiment_2/                      [Model 2: Complete Pooling]
│   └── [same structure]
├── experiment_3/                      [Model 3: Skeptical Hierarchical]
│   └── [same structure]
├── experiment_4/                      [Model 4: No Pooling - if implemented]
│   └── [same structure]
├── experiment_5/                      [Model 5: Student-t - if implemented]
│   └── [same structure]
└── model_comparison/                  [Final comparison across models]
    ├── comparison_report.md
    ├── loo_comparison.csv
    └── figures/
```

---

## Success Metrics

**Minimum success (meets guidelines):**
- Models 1 and 2 fitted with convergence diagnostics
- Prior predictive checks for both
- Posterior predictive checks for both
- LOO comparison
- Clear recommendation with justification

**Good success:**
- Models 1-3 fitted and compared
- Model 4 or 5 added for robustness
- Comprehensive PPC and shrinkage analysis
- Thorough sensitivity assessment

**Excellent success:**
- All core models (1-5) fitted
- Prior sensitivity analysis (Model 6) if warranted
- Clear falsification criteria tested
- Transparent reporting of limitations
- Convergence or divergence with EDA explained

---

## Key Insights from Designer Synthesis

1. **Unanimity on EDA strength:** All designers acknowledge EDA strongly favors homogeneity
2. **Power limitations:** With n=8, detecting tau<5 is difficult (important caveat)
3. **Falsification mindset:** All designers defined abandonment criteria (good practice)
4. **Computational awareness:** Non-centered parameterization unanimously recommended
5. **Prior caution:** Designers differ on optimal tau prior, suggesting sensitivity analysis valuable
6. **Robustness skepticism:** Designer 3 correctly notes robustness may be unnecessary here
7. **Parsimony emphasis:** All note complete pooling is well-supported and should be considered seriously

---

## References from Designer Proposals

- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. Bayesian Analysis, 1(3), 515-534.
- Juárez, M. A., & Steel, M. F. (2010). Model-based clustering of non-Gaussian panel data based on skew-t distributions. Journal of Business & Economic Statistics, 28(1), 52-66.

---

## Next Steps

1. ✅ EDA complete
2. ✅ Model design complete (3 parallel designers synthesized)
3. ⏭️ **Begin Phase A:** Implement Models 1-3 following validation pipeline
   - Prior predictive checks
   - Simulation-based validation
   - Posterior inference
   - Posterior predictive checks
   - Model critique
4. ⏭️ Make decision to continue to Phase B based on Phase A results
5. ⏭️ Model comparison and assessment (Phase 4)
6. ⏭️ Adequacy assessment (Phase 5)
7. ⏭️ Final report (Phase 6)

**Status:** Ready for implementation. Awaiting signal to begin validation pipeline.

---

**Document created:** 2025-10-28
**Created by:** Main agent synthesis of Designers 1, 2, and 3
**Approval:** Ready for implementation
