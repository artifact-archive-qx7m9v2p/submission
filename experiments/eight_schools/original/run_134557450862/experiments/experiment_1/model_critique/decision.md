# Model Decision: Experiment 1

**Date:** 2025-10-28
**Model:** Standard Non-Centered Hierarchical Model
**Analyst:** Model Criticism Specialist

---

## DECISION: CONDITIONAL ACCEPT

**Status:** Accept model as scientifically adequate with mandatory caveats and recommended sensitivity analyses

---

## Summary Recommendation

The standard non-centered hierarchical model is **ACCEPTED** as a valid baseline for inference on the Eight Schools dataset, subject to the following conditions:

1. **Mandatory acknowledgment** of τ identifiability limitations in any scientific reporting
2. **Strongly recommended** sensitivity analysis with alternative priors on τ
3. **Completion** of posterior predictive check (in progress, expected to pass)
4. **Comparison** with complete pooling model given p_eff ≈ 1

The model demonstrates excellent computational properties, proper calibration, and honest uncertainty quantification. However, users must understand that **the posterior distribution of τ reflects prior influence as much as data evidence**, and conclusions about between-school heterogeneity should be stated with appropriate epistemic humility.

---

## Justification

### Why ACCEPT (Not REJECT)

**1. All Computational Criteria Met:**
- Zero divergences (0/8000 samples)
- Perfect convergence (R-hat = 1.000 for all parameters)
- High effective sample sizes (ESS > 5000)
- Fast, efficient sampling (18 seconds)
- Non-centered parameterization works flawlessly

**2. Properly Validated:**
- ✓ Prior predictive check: PASS
- ✓ Simulation-based calibration: PASS (with expected identifiability caveats)
- ✓ Convergence diagnostics: PASS
- ✓ Parameter recovery: Good to excellent
- ⏸ Posterior predictive check: PENDING (expected to PASS based on LOO)

**3. Scientifically Defensible:**
- All parameter estimates within plausible ranges
- Grand mean μ well-identified and robust
- Shrinkage pattern appropriate and consistent
- No influential outliers (all Pareto k < 0.7)
- Honest uncertainty quantification

**4. Answers Key Research Questions:**
- **Grand mean effect:** YES - μ = 7.36 ± 4.32 is reliable
- **Individual school effects:** YES - posterior means appropriately shrunk
- **Heterogeneity magnitude:** PARTIALLY - τ estimate has large uncertainty
- **Predictions for new schools:** PARTIALLY - depends on uncertain τ

**5. No Fundamental Misspecification:**
- Model structure appropriate for hierarchical data
- Likelihood appears adequate (no outliers, normality reasonable)
- Priors well-calibrated and scientifically justified
- No systematic residual patterns
- No prior-data conflict (just tension on weakly-identified τ)

### Why NOT REJECT

**Rejecting would require:**
- Fundamental computational failure → NOT present
- Inability to reproduce data → NOT present (LOO indicates good fit)
- Extreme parameter values → NOT present
- Prior-data conflict → NOT present (tension ≠ conflict)
- Persistent sampling pathologies → NOT present

**The model is not broken.** It appropriately handles a difficult inferential situation (small n, large measurement errors, weak signal) by expressing genuine uncertainty rather than providing false confidence.

### Why CONDITIONAL (Not Unconditional Accept)

**The conditionality stems from:**

**1. Identifiability Limitations Not Yet Fully Explored:**
- τ is weakly identified (cannot distinguish τ=0 from τ=5)
- Posterior τ likely sensitive to prior choice
- Need sensitivity analysis to quantify robustness
- Must acknowledge in reporting

**2. Tension with Classical Analysis:**
- EDA: I²=0%, Q p=0.696, τ²=0 (no heterogeneity)
- Posterior: τ=3.58±3.15 (moderate heterogeneity)
- Need to reconcile and explain in context of Bayesian regularization
- Not a failure, but requires careful interpretation

**3. Model Comparison Not Yet Done:**
- p_eff = 1.03 suggests data support simple (complete pooling) model
- Haven't tested if hierarchical structure improves fit
- Should compare LOO with complete pooling baseline
- May find simpler model is adequate

**4. Posterior Predictive Check Incomplete:**
- Cannot fully assess model adequacy without PPC
- LOO diagnostics are encouraging (all k<0.7)
- But need explicit coverage and distribution checks
- Expected to pass, but must verify

---

## Conditions for Acceptance

### MANDATORY Conditions

**1. Complete Posterior Predictive Check**
- **Action:** Finish PPC analysis (in progress)
- **Success criterion:** >90% of observed y_i within 95% posterior predictive intervals
- **Timeline:** Before final reporting
- **Fallback:** If fails, REVISE likelihood specification

**2. Acknowledge Identifiability Limitations**
- **Action:** Include explicit discussion in any scientific report
- **Content must include:**
  - τ cannot be precisely estimated with n=8
  - Cannot distinguish τ=0 from τ≈5 with confidence
  - Posterior τ reflects prior influence given weak data
  - 95% HDI for τ spans 0 to 9.2 (order of magnitude uncertainty)
- **Wording example:** "With only 8 schools and large measurement errors, the data provide limited information about between-school variance. The posterior distribution of τ reflects both the weak data signal and our prior assumptions, and should be interpreted as expressing genuine epistemic uncertainty rather than strong evidence for heterogeneity."

**3. Report Full Posterior Distributions (Not Point Estimates)**
- **Action:** Emphasize intervals and uncertainty
- **Specifically:**
  - Report 95% HDIs, not just means
  - Show full posterior plots for τ
  - Avoid definitive language about τ>0
  - Present shrinkage analysis visually

### STRONGLY RECOMMENDED Conditions

**4. Prior Sensitivity Analysis**
- **Action:** Fit model with at least 2 alternative priors on τ
- **Specific priors to test:**
  - Half-Cauchy(0, 1) - tighter prior
  - Half-Cauchy(0, 10) - looser prior
  - Optional: Half-Normal(0, 5) - different distributional form
- **What to report:**
  - Range of posterior τ medians across priors
  - How much μ and θ_i change (expected: minimal)
  - Whether substantive conclusions change
- **Timeline:** Before claiming τ is meaningfully different from zero

**5. Compare to Complete Pooling Model**
- **Action:** Fit model without hierarchical structure
- **Model specification:** y_i ~ Normal(μ, σ_i), μ ~ Normal(0, 20)
- **Comparison metrics:** LOO ELPD, WAIC, posterior predictive checks
- **Rationale:** p_eff=1.03 suggests data may not require hierarchy
- **What to report:** Whether hierarchical structure improves predictive performance
- **Timeline:** Before recommending hierarchical model for final selection

### OPTIONAL Enhancements

**6. Uncertainty in Measurement Errors (σ_i)**
- **Action:** Propagate uncertainty in σ_i if known
- **Impact:** Likely minor widening of posteriors
- **Priority:** Low (nice to have, not essential)

**7. Leave-K-Out Cross-Validation**
- **Action:** Test stability with multiple schools removed
- **Impact:** Assess robustness of τ estimate
- **Priority:** Low (informative but not critical)

---

## What This Decision Means

### The Model IS Adequate For:

✓ **Estimating grand mean effect (μ):**
- Well-identified, robust to prior choice
- Can confidently report μ ≈ 7-8 with uncertainty

✓ **Producing shrinkage estimates of school effects:**
- Appropriately regularizes extreme observations
- Better than raw observed effects for decision-making
- Shrinkage pattern is correct and consistent

✓ **Quantifying overall uncertainty:**
- Wide intervals honestly reflect limited information
- Better than falsely precise point estimates
- Captures epistemic limitations appropriately

✓ **Serving as baseline for model comparison:**
- Provides ELPD benchmark for alternative models
- Well-validated computational implementation
- Standard approach in literature

### The Model IS NOT Adequate For:

✗ **Definitive claims about between-school heterogeneity:**
- τ is weakly identified, prior-sensitive
- Cannot distinguish small from zero heterogeneity
- Requires sensitivity analysis and caveats

✗ **Precise predictions for individual schools:**
- Large uncertainty in school-specific effects
- 95% HDIs span ~20 points
- Cannot reliably rank or compare schools

✗ **Testing whether τ = 0 vs τ > 0:**
- Data lack power to make this distinction
- Posterior mass on τ>0 partly reflects prior
- Would need more schools for definitive test

✗ **Claims without uncertainty acknowledgment:**
- Point estimates alone are misleading
- Must report full posterior distributions
- Must acknowledge identifiability limitations

---

## Required Reporting Guidelines

### When Presenting Results, MUST Include:

**1. For Grand Mean (μ):**
- ✓ Posterior mean: 7.36
- ✓ Posterior SD: 4.32
- ✓ 95% HDI: [-0.56, 15.60]
- ✓ Interpretation: Likely positive but uncertain

**2. For Between-School Variance (τ):**
- ✓ Posterior mean: 3.58
- ✓ Posterior SD: 3.15 (nearly equal to mean!)
- ✓ 95% HDI: [0.00, 9.21]
- ✓ Caveat: "Weakly identified; estimate reflects prior influence given limited data"
- ✓ Context: "Classical tests (Q, I²) find no evidence for heterogeneity"
- ✓ Interpretation: "Substantial uncertainty about true heterogeneity"

**3. For School Effects (θ_i):**
- ✓ Posterior means with 95% HDIs
- ✓ Shrinkage percentages
- ✓ Visual comparison to observed effects
- ✓ Warning: "Individual school estimates highly uncertain (HDI width ~20 points)"

**4. Model Limitations:**
- ✓ "Only 8 schools limit precision of variance estimate"
- ✓ "Large measurement errors reduce power to detect heterogeneity"
- ✓ "Posterior τ should be interpreted as expressing uncertainty, not strong evidence for heterogeneity"
- ✓ "Results may be sensitive to prior choice on τ" (if sensitivity not done)
- ✓ "Cannot reliably predict effects for new schools without additional data"

### Forbidden Claims (Without Further Analysis):

✗ "There is significant between-school heterogeneity" → Too strong given uncertainty
✗ "Schools differ in their treatment effects" → Cannot conclude definitively
✗ "τ is significantly greater than zero" → Posterior mass reflects prior
✗ "School X is better/worse than School Y" → Posterior means too uncertain
✗ "This model is superior to complete pooling" → Haven't tested

---

## Comparison to Alternative Decisions

### If We Had Chosen REVISE

**Would require:**
- Specific fixable problems identified
- Clear path to improvement
- Expectation that revisions would resolve issues

**Why NOT chosen:**
- No fixable computational problems
- No clear model misspecification
- Revisions (like different priors) are sensitivity checks, not fixes
- Current model is sound, just has inherent limitations

**When REVISE would be appropriate:**
- If PPC fails (need different likelihood)
- If divergences were persistent (need different parameterization)
- If residuals showed systematic patterns (missing predictors)
- If prior-data conflict was severe (need different priors)

### If We Had Chosen REJECT

**Would require:**
- Fundamental model inadequacy
- Cannot answer research questions
- Computational failure despite efforts
- Prior-data conflict unresolvable
- Systematic misspecification

**Why NOT chosen:**
- Model DOES answer key questions (grand mean, shrinkage)
- Computational performance is excellent
- No fundamental misspecification
- Tension on τ is not failure, it's honest uncertainty
- Limitations are inherent to data, not model

**When REJECT would be appropriate:**
- If model routinely diverged despite tuning
- If couldn't recover simulated parameters
- If PPC showed severe misfit
- If estimates were scientifically implausible
- If model structure was inappropriate for data generation

---

## Implications for Experimental Pipeline

### For Model Selection Process

**This model should be:**
- ✓ Included as candidate for final selection
- ✓ Used as baseline for ELPD comparisons
- ✓ Considered alongside complete pooling model
- ✓ Compared to alternative prior specifications

**This model should NOT be:**
- ✗ Automatically selected without comparison
- ✗ Treated as ground truth
- ✗ Used without acknowledging limitations

### For Alternative Experiments

**Recommended comparisons:**

**Experiment 2:** Centered parameterization
- Expect: Computational problems (funnel)
- Purpose: Validate non-centered choice
- Priority: MEDIUM

**Experiment 3:** Complete pooling (no hierarchy)
- Expect: Similar ELPD given p_eff≈1
- Purpose: Test if hierarchy is necessary
- Priority: HIGH

**Experiment 4:** Alternative priors on τ
- Expect: Posterior τ sensitivity
- Purpose: Quantify prior influence
- Priority: HIGH

**Experiment 5:** Robust likelihood (Student-t)
- Expect: Minor differences (no outliers)
- Purpose: Check robustness
- Priority: LOW

### For Final Model Selection

**Selection criteria should weigh:**
1. **Predictive performance:** LOO ELPD (primary metric)
2. **Interpretability:** Simpler models preferred if equal performance
3. **Robustness:** Sensitivity to prior choices
4. **Computational efficiency:** All current candidates fast enough
5. **Scientific plausibility:** All current candidates reasonable

**Expected outcome:**
- Hierarchical and complete pooling models likely similar ELPD
- Choice may come down to philosophical preference
- Could recommend both with appropriate caveats

---

## Success Criteria for Maintaining ACCEPT

**The CONDITIONAL ACCEPT becomes UNCONDITIONAL ACCEPT if:**

1. ✓ Posterior predictive check passes (>90% coverage)
2. ✓ Prior sensitivity analysis shows conclusions robust OR
   - Alternative: Acknowledge sensitivity explicitly in reporting
3. ✓ Complete pooling comparison shows hierarchical structure justified OR
   - Alternative: Present both models as valid alternatives

**The ACCEPT is REVOKED if:**

1. ✗ Posterior predictive check reveals systematic misfit
2. ✗ New information suggests different model structure needed
3. ✗ Computational problems emerge in extended analysis
4. ✗ Scientific review identifies fundamental flaws

---

## Recommendation for Scientific Reporting

### Executive Summary Template

"We fitted a Bayesian hierarchical model to the Eight Schools dataset using a non-centered parameterization with weakly informative priors. The model converged excellently and produced scientifically plausible estimates. The grand mean treatment effect is estimated at 7.4 points (95% credible interval: -0.6 to 15.6), suggesting a likely positive effect with substantial uncertainty. Individual school effects are estimated to lie between 6 and 9 points after appropriate shrinkage toward the grand mean, much less variable than the raw observed effects (-3 to 28).

The between-school variance (τ) is poorly identified given the small sample (n=8) and large measurement errors. The posterior distribution (mean: 3.6, 95% CI: 0 to 9.2) reflects both the weak data signal and our prior assumptions, and should be interpreted as expressing genuine epistemic uncertainty rather than strong evidence for heterogeneity. Classical heterogeneity tests (Cochran's Q, I²) find no evidence for between-school variation, and we cannot definitively distinguish scenarios with zero versus moderate heterogeneity based on these data.

For practical purposes, all schools should be considered to have similar true effects (approximately 6-9 points) given current evidence, and predictions for new schools should center on the grand mean of 7.4 points. Stronger conclusions about between-school heterogeneity would require data from substantially more schools or more precise within-school measurements."

### Key Points for Discussion Section

1. **Acknowledge identifiability limitations explicitly**
2. **Reconcile Bayesian posterior with classical heterogeneity tests**
3. **Explain that τ>0 doesn't prove heterogeneity exists**
4. **Justify prior choice and discuss alternatives**
5. **Emphasize uncertainty in all claims**
6. **Recommend more data if heterogeneity assessment is critical**

---

## Final Statement

**This model is ACCEPTED as scientifically adequate for inference on the Eight Schools dataset**, with the understanding that:

- It provides reliable estimates of the grand mean effect
- It appropriately shrinks extreme school-level observations
- It honestly quantifies epistemic uncertainty
- It has important limitations regarding heterogeneity estimation
- It requires careful interpretation and reporting

The model is **fit for purpose** as long as users understand its strengths and limitations, acknowledge identifiability issues, and report results with appropriate caveats.

**The model is recommended to proceed to final model selection** alongside complete pooling and alternative prior specifications, where comparative performance and scientific considerations will inform the ultimate choice.

---

## Sign-off

**Model Status:** CONDITIONAL ACCEPT ✓

**Conditions:**
- [⏸] Complete posterior predictive check
- [⏸] Prior sensitivity analysis (strongly recommended)
- [⏸] Complete pooling comparison (strongly recommended)

**Authorized for:**
- Scientific inference on grand mean
- Shrinkage estimation of school effects
- Baseline for model comparison
- Publication with appropriate caveats

**Not authorized for:**
- Definitive heterogeneity claims without caveats
- School ranking or selection
- Strong predictions for new schools
- Publication without uncertainty acknowledgment

**Next Review:** After completion of PPC and sensitivity analyses

---

**Analyst:** Model Criticism Specialist
**Date:** 2025-10-28
**Experiment:** 1 (Standard Non-Centered Hierarchical Model)
