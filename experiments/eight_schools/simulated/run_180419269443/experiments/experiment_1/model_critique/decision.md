# Model Decision: Experiment 1 - Hierarchical Normal Model

**Date:** 2025-10-28
**Model:** Hierarchical Normal Model with known within-study variance
**Analyst:** Model Criticism Specialist (Claude)

---

## DECISION: ACCEPT

**Status:** ACCEPTED FOR INFERENCE (with conditions)

---

## Executive Summary

The Hierarchical Normal Model (Experiment 1) is **ACCEPTED** as adequate for baseline meta-analytic inference. The model has passed all validation stages with strong performance across computational, statistical, and scientific dimensions. While not perfect (no model is), it demonstrates no critical flaws that would require rejection or fundamental revision.

**Key Evidence Supporting Acceptance:**
- All falsification criteria passed (tau < 15, all Pareto k < 0.7, PPC passed, Study 4 not overly influential)
- Excellent simulation-based calibration (94-95% coverage)
- Strong posterior predictive performance (9/9 test statistics pass)
- Adequate convergence (R-hat at boundary but supported by other diagnostics)
- Scientific plausibility (effect size reasonable, handles outliers appropriately)

**Conditions for Final Acceptance:**
1. Must compare to Experiment 2 (complete pooling) - MANDATORY per plan
2. Must assess prior sensitivity (Experiment 4) - MANDATORY for J=8
3. Phase 4 synthesis must confirm no superior alternatives

This is a **provisional acceptance** pending comparison with alternative models.

---

## Justification (Detailed)

### 1. Why ACCEPT?

**Computational Reliability (STRONG):**
- Convergence achieved (R-hat = 1.01 at boundary, but all other diagnostics excellent)
- ESS adequate: mu = 440, tau = 166, all theta > 400
- No divergences (Gibbs sampler property)
- Validated via SBC: 94-95% coverage confirms correct implementation
- MCSE < 5% of posterior SD (sampling uncertainty negligible)

**Statistical Adequacy (STRONG):**
- Well-calibrated posteriors (SBC validation)
- Posterior predictive checks: All 9 test statistics pass (p-values in [0.29, 0.85])
- Study-level fit: 7/8 studies good fit, 1/8 marginal (within expectation)
- No systematic residual patterns detected
- LOO diagnostics excellent: All Pareto k < 0.7 (max 0.647)
- Q-Q plot confirms normality assumption adequate

**Scientific Validity (STRONG):**
- Effect size plausible: mu = 9.87 ± 4.89, consistent with EDA (11.27)
- Heterogeneity reasonable: I² = 17.6% (low to moderate), though uncertain
- Handles potential outlier (Study 5) appropriately via hierarchical shrinkage
- Shrinkage patterns sensible: 70-88%, higher for noisy/extreme studies
- Uncertainty quantification honest: Wide CIs reflect limited data (J=8)

**Falsification Criteria (ALL PASSED):**
- ✅ Posterior tau = 5.55 < 15 (not severely underestimated)
- ✅ Max Pareto k = 0.647 < 0.7 (no problematic outliers)
- ✅ Posterior predictive checks: All p ∈ [0.29, 0.85] (no failures)
- ✅ Study 4 influence: k = 0.398 < 1.0 (not excessive)
- ✅ Prior predictive check: Passed (from prior_predictive_check stage)

**No Critical Limitations:**
- No evidence of model misspecification
- No computational pathologies
- No prior-data conflicts
- No influential points requiring removal
- No need for robust alternatives (yet)

### 2. Why NOT REVISE?

**"REVISE" implies fixable issues requiring model modification.**

**Assessment:** No such issues identified.

**Potential issues considered and dismissed:**

1. **R-hat = 1.01 (at boundary)?**
   - **Not actionable:** All other diagnostics (ESS, visual, LOO) are excellent
   - **Not a revision issue:** More iterations would not change conclusions (ESS already adequate)
   - **Decision:** Note but accept

2. **Tau poorly estimated (SD = 4.21, nearly as large as mean)?**
   - **Not a model issue:** This is a data limitation (J=8 insufficient for precise tau)
   - **Not fixable by revision:** Different priors (Experiment 4) or likelihood (Experiment 3) won't dramatically improve precision
   - **Decision:** Acknowledge limitation, not grounds for revision

3. **Study 5 potential outlier (k = 0.647)?**
   - **Not problematic:** k < 0.7 threshold, model accommodates adequately
   - **Not requiring revision:** PPC shows good fit (p = 0.234), no need for robust model
   - **Decision:** Monitor in Experiment 3, but current model adequate

4. **Known sigma assumption?**
   - **Not fixable:** Standard meta-analytic assumption, requires IPD to address
   - **Not critical:** Likely minor impact on credible intervals
   - **Decision:** Acknowledge limitation, not grounds for revision

5. **Small sample (J=8)?**
   - **Not a model issue:** Data constraint, not model specification problem
   - **Not fixable by revision:** Need more studies, not different model
   - **Decision:** Quantify uncertainty honestly (done), not grounds for revision

**Conclusion:** No **fixable issues** that would improve model adequacy. Moving to "REVISE" would be unfocused revisions without clear path to improvement.

### 3. Why NOT REJECT?

**"REJECT" implies fundamental misspecification or critical failures.**

**Assessment:** No evidence of fundamental problems.

**Rejection criteria considered and dismissed:**

1. **Convergence failure?**
   - **Reality:** Converged (R-hat at boundary but supported, ESS adequate)
   - **Decision:** NOT REJECTED

2. **Systematic misfit?**
   - **Reality:** All posterior predictive checks pass, no residual patterns
   - **Decision:** NOT REJECTED

3. **Extreme outliers?**
   - **Reality:** All Pareto k < 0.7, no problematic influential points
   - **Decision:** NOT REJECTED

4. **Prior-data conflict?**
   - **Reality:** Prior predictive checks passed, posteriors reasonable
   - **Decision:** NOT REJECTED

5. **Fundamental misspecification?**
   - **Reality:** Normal likelihood adequate (PPC, Q-Q plot confirm)
   - **Reality:** Hierarchical structure appropriate (shrinkage patterns sensible)
   - **Decision:** NOT REJECTED

6. **Results scientifically implausible?**
   - **Reality:** Effect size reasonable, consistent with EDA
   - **Decision:** NOT REJECTED

**Conclusion:** Model is **not fundamentally flawed**. All validation stages passed. No evidence requiring rejection.

### 4. Why CONDITIONS?

**Conditions do not indicate inadequacy - they reflect:**

**A. Experiment Plan Requirements:**
- **Minimum attempt policy:** Must fit Experiments 1-2
- **Prior sensitivity mandatory:** For J=8, must assess (Experiment 4)
- **Model comparison essential:** Cannot accept single model without comparison

**B. Scientific Rigor:**
- **Baseline model:** Experiment 1 is baseline, not final answer
- **Alternative plausibility:** Complete pooling (Exp 2) is defensible given I² CI includes zero
- **Prior sensitivity:** With J=8 and tau uncertain, prior choice matters

**C. Bayesian Workflow Best Practices:**
- **Model comparison:** LOO cross-validation will rank models
- **Sensitivity testing:** Multiple models provide robustness
- **Ensemble possibility:** If models similar, may combine via stacking

**Conditions are procedural, not signs of inadequacy.**

---

## What ACCEPT Means

**ACCEPT does NOT mean:**
- ❌ Model is perfect
- ❌ No uncertainty remains
- ❌ No alternative models needed
- ❌ Results are definitive
- ❌ Skip Experiments 2-4

**ACCEPT DOES mean:**
- ✅ Model is fit for its intended purpose (baseline hierarchical meta-analysis)
- ✅ No critical flaws detected
- ✅ Inference from this model is scientifically defensible
- ✅ Results can be reported (with appropriate uncertainty)
- ✅ Model is valid baseline for comparison with alternatives
- ✅ Proceed to Experiments 2-4 for comparison and sensitivity testing

**Analogy:**
- ACCEPT is like passing a peer review: Model is publication-worthy, but reviewers may request additional analyses (Experiments 2-4) for completeness.

---

## Supported Conclusions

**With Experiment 1 ACCEPTED, the following conclusions are scientifically supported:**

### Strongly Supported (High Confidence):

1. **Treatment effect is likely positive**
   - Evidence: mu = 9.87, 97% posterior mass > 0
   - Caveat: Magnitude uncertain (95% CI [0.28, 18.71])

2. **Study-specific effects are uncertain and overlapping**
   - Evidence: All theta_i CIs overlap substantially
   - Caveat: Cannot confidently rank studies

3. **Hierarchical shrinkage is appropriate**
   - Evidence: 70-88% shrinkage, validated via SBC
   - Caveat: Reflects limited data (J=8) and high within-study variance

4. **Study 5 is most discrepant but not dismissible**
   - Evidence: Pareto k = 0.647 (highest), only negative effect
   - Caveat: k < 0.7 indicates model accommodates adequately

5. **Model is well-calibrated**
   - Evidence: SBC 94-95% coverage, PPC passes
   - Caveat: Calibration validated on simulated data, assumes model correct

### Moderately Supported (Moderate Confidence):

6. **Between-study heterogeneity is low to moderate**
   - Evidence: I² = 17.6%, tau = 5.55
   - Caveat: Huge uncertainty (95% CI for I² is [0.01%, 59.9%])

7. **Heterogeneity is greater than zero**
   - Evidence: Posterior mode for tau > 0, I² posterior mean = 17.6%
   - Caveat: 95% CI for tau starts at 0.03 (nearly zero)

8. **More studies are needed for precise inference**
   - Evidence: Wide CIs, tau poorly estimated
   - Caveat: This is a recommendation, not a finding

### Weakly Supported (Low Confidence, Requires Confirmation):

9. **Complete pooling may be adequate**
   - Evidence: I² CI includes near-zero, EDA I² = 2.9%
   - Caveat: Requires Experiment 2 comparison

10. **Normal likelihood is adequate**
    - Evidence: PPC passes, all Pareto k < 0.7
    - Caveat: Requires Experiment 3 (robust model) for sensitivity test

11. **Prior choice for tau moderately influences posterior**
    - Evidence: 58% prior-to-posterior SD reduction
    - Caveat: Requires Experiment 4 for formal quantification

---

## Conclusions NOT Supported

**The following conclusions are NOT supported by Experiment 1 alone:**

1. ❌ **"Heterogeneity is definitely low"** - CI too wide
2. ❌ **"Study 4 is the best/most effective"** - CIs overlap substantially
3. ❌ **"Study 5 should be excluded"** - No statistical justification (k < 0.7, PPC passes)
4. ❌ **"Effect size is definitely large (>15)"** - Only 12% posterior mass > 15
5. ❌ **"Effect size is definitely small (<5)"** - Only 18% posterior mass < 5
6. ❌ **"Complete pooling is preferred"** - Requires Experiment 2 comparison
7. ❌ **"Hierarchical model is definitely preferred"** - Requires Experiment 2 comparison
8. ❌ **"Results are insensitive to prior choice"** - Requires Experiment 4
9. ❌ **"Normal likelihood is optimal"** - Requires Experiment 3
10. ❌ **"Intervention is causally effective"** - Requires causal study designs

---

## Required Next Steps

**Before final conclusions, MUST complete:**

### 1. Experiment 2: Complete Pooling (MANDATORY)

**Purpose:** Test whether heterogeneity matters
- If ΔLOO < 2: Models equivalent, prefer simpler (complete pooling)
- If ΔLOO > 4: Hierarchical clearly better
- If 2 < ΔLOO < 4: Marginal preference for hierarchical

**Expected outcome:** Similar mu, narrower CI in Exp 2
**Decision:** Model comparison will decide whether to use Exp 1 or Exp 2

### 2. Experiment 4: Prior Sensitivity (MANDATORY)

**Purpose:** Quantify sensitivity to prior choice (critical for J=8)
- Skeptical prior: tau ~ Half-N(0, 5)
- Enthusiastic prior: tau ~ Half-Cauchy(0, 10)
- Assess |mu_diff| and |tau_diff|

**Expected outcome:** Low mu sensitivity, moderate tau sensitivity (2-4 units)
**Decision:** If extreme sensitivity (|tau_diff| > 5), use ensemble

### 3. Experiment 3: Robust Hierarchical (CONDITIONAL)

**Purpose:** Validate normal likelihood assumption
- Student-t likelihood
- If nu > 50: Normal adequate (validate Exp 1)
- If nu < 20: Heavy tails matter (prefer Exp 3)

**Expected outcome:** nu ≈ 30-50 (validates Exp 1)
**Decision:** If time permits, provides robustness check

### 4. Phase 4: Model Assessment (ALWAYS)

**Purpose:** Holistic comparison across all fitted models
- LOO stacking weights
- Ensemble if models similar
- Final recommendations

**Required:** Always runs after Phase 3
**Decision:** Final acceptance contingent on Phase 4 synthesis

---

## Conditional Acceptance Criteria

**Experiment 1 will remain ACCEPTED if:**

1. **Experiment 2 comparison:**
   - If Exp 2 strongly preferred (ΔLOO > 4): May switch to Exp 2, but Exp 1 still valid
   - If Exp 1 preferred or equivalent: Exp 1 confirmed
   - **Action:** Report both, use LOO to decide or ensemble

2. **Experiment 4 prior sensitivity:**
   - If low sensitivity (|tau_diff| < 3): Exp 1 results robust
   - If moderate sensitivity (3 < |tau_diff| < 5): Report range, acknowledge uncertainty
   - If high sensitivity (|tau_diff| > 5): Use ensemble, data insufficient
   - **Action:** Quantify sensitivity, adjust conclusions accordingly

3. **Experiment 3 robustness (if performed):**
   - If nu > 30: Normal likelihood validated, Exp 1 confirmed
   - If nu < 20: Prefer Exp 3 (Student-t), but Exp 1 still adequate for comparison
   - **Action:** Use Exp 3 if substantially better, otherwise Exp 1

**Exp 1 would be DOWNGRADED if:**
- Experiment 2 shows ΔLOO > 10 (extremely strong preference, unlikely)
- Experiment 3 shows nu < 10 (severe outliers, extremely unlikely)
- Experiment 4 shows |mu_diff| > 10 (extreme prior sensitivity, very unlikely)

**Given current diagnostics, downgrading is UNLIKELY.**

---

## Summary

**DECISION: ACCEPT**

The Hierarchical Normal Model (Experiment 1) is adequate for baseline meta-analytic inference. All validation stages passed, no critical flaws detected. Model provides scientifically defensible estimates of population-level effects with appropriate uncertainty quantification.

**Key Strengths:**
- Excellent validation (SBC, PPC, LOO all pass)
- Adequate convergence (R-hat at boundary but supported)
- Scientific plausibility (effect size reasonable)
- Honest uncertainty (wide CIs reflect limited data)

**Acknowledged Limitations:**
- Small sample (J=8) limits precision
- Tau poorly estimated (data limitation, not model flaw)
- Prior sensitivity for tau to be tested (Experiment 4)

**Conditions:**
- Must compare to Experiment 2 (complete pooling)
- Must assess prior sensitivity (Experiment 4)
- Phase 4 synthesis for final confirmation

**Recommendation:**
Proceed to Experiments 2 and 4 as planned. Use Experiment 1 results for baseline inference, with formal comparison and sensitivity testing to follow.

**Model is FIT FOR PURPOSE.**

---

**Approval:** Model Criticism Specialist (Claude)
**Date:** 2025-10-28
**Status:** ACCEPTED (conditional on Experiments 2 and 4)
**Next:** Proceed to Experiment 2
