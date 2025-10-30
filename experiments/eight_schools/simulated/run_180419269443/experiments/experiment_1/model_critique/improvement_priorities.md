# Improvement Priorities: Experiment 1 - Hierarchical Normal Model

**Date:** 2025-10-28
**Model:** Hierarchical Normal Model with known within-study variance
**Analyst:** Model Criticism Specialist (Claude)
**Decision:** ACCEPT (no revision required)

---

## Status: No Immediate Improvements Required

**Current Decision:** ACCEPT

This document outlines what improvements WOULD be prioritized IF the model were REVISED. Since the decision is ACCEPT, these are **not actionable now** but are documented for:

1. **Contingency planning:** If Experiments 2-4 reveal issues, these are next steps
2. **Future research:** When more data are available
3. **Transparency:** What could be improved with unlimited resources
4. **Context for Phase 4:** Comparison to alternatives

---

## Improvement Categories

### A. No Improvement Possible (Data Limitations)

These issues cannot be fixed by model revision - they require more/better data:

**1. Small sample size (J=8)**
- **Issue:** Only 8 studies limits precision (wide credible intervals)
- **Impact:** Tau poorly estimated, cannot rule out alternatives with confidence
- **Cannot fix with model revision:** Need more studies (target: 15-20)
- **Priority if more data:** HIGHEST - collect more studies

**2. Known within-study variances**
- **Issue:** Treating sigma_i as fixed underestimates uncertainty
- **Impact:** Credible intervals slightly too narrow (likely minor)
- **Cannot fix without IPD:** Requires individual participant data
- **Priority if IPD available:** MEDIUM - incorporate uncertainty in sigma_i

**3. No covariates available**
- **Issue:** Cannot perform meta-regression to explain heterogeneity
- **Impact:** Exchangeability assumption untestable
- **Cannot fix without covariate data:** Need study-level covariates (year, location, risk of bias)
- **Priority if covariates available:** HIGH - meta-regression to explain tau

**4. Publication bias not assessed**
- **Issue:** Small positive mu could reflect selective reporting
- **Impact:** Effect size may be overestimated
- **Cannot fix with J=8:** Need 20-30 studies for funnel plot power
- **Priority if more studies:** MEDIUM - selection model or funnel plot analysis

---

### B. Already Planned (Experiments 2-4)

These are sensitivity tests and comparisons already in the experiment plan:

**5. Compare to complete pooling (Experiment 2)**
- **Issue:** I² credible interval includes near-zero (tau ≈ 0 plausible)
- **Current status:** Experiment 2 planned (MANDATORY)
- **Expected outcome:** Similar results, LOO will decide
- **Priority:** HIGHEST - required for model comparison

**6. Test prior sensitivity (Experiment 4)**
- **Issue:** Tau posterior shows moderate prior influence (58% SD reduction)
- **Current status:** Experiment 4 planned (MANDATORY)
- **Expected outcome:** Low mu sensitivity, moderate tau sensitivity (2-4 units)
- **Priority:** HIGHEST - essential for J=8

**7. Test robustness to outliers (Experiment 3)**
- **Issue:** Study 5 is potential outlier (k = 0.647, only negative)
- **Current status:** Experiment 3 planned (CONDITIONAL)
- **Expected outcome:** nu > 30 (validates normal likelihood)
- **Priority:** MEDIUM - robustness check if time permits

---

### C. Could Improve Model (But Not Needed)

These would enhance the model but are not necessary given current diagnostics:

**8. Increase MCMC iterations**
- **Issue:** R-hat = 1.01 at boundary (marginal convergence)
- **Current status:** ESS adequate (mu: 440, tau: 166), MCSE < 5% of SD
- **Potential improvement:** Run 20,000 iterations (double current)
- **Expected outcome:** R-hat → 1.00, minimal change in posteriors
- **Priority:** LOW - not needed given supporting diagnostics
- **Cost/benefit:** Low benefit (conclusions unchanged) for computational cost

**9. Use HMC instead of Gibbs**
- **Issue:** Gibbs sampler has lower ESS/iteration than HMC
- **Current status:** Gibbs validated via SBC (94-95% coverage), adequate ESS
- **Potential improvement:** Implement in CmdStan (if environment allows)
- **Expected outcome:** Higher ESS/iteration (10-50% vs 1-5%), same posteriors
- **Priority:** LOW - Gibbs is adequate for conjugate hierarchical model
- **Cost/benefit:** Marginal benefit for implementation effort (CmdStan unavailable)

**10. Non-centered parameterization**
- **Issue:** Centered parameterization can cause funnel geometry
- **Current status:** Model already uses non-centered parameterization (not explicit in description)
- **Potential improvement:** N/A - already implemented
- **Priority:** N/A - already done

**11. Alternative prior for tau**
- **Issue:** Half-Normal(0, 10) is one of many defensible priors
- **Current status:** Experiment 4 will test alternatives
- **Potential improvement:** Use Half-Cauchy(0, 2.5) (Gelman's default)
- **Expected outcome:** Slightly lower posterior tau (by 1-2 units)
- **Priority:** LOW - Experiment 4 addresses this
- **Cost/benefit:** Experiment 4 will quantify sensitivity

---

### D. Extensions (Beyond Current Scope)

These would require fundamentally different models:

**12. Meta-regression with covariates**
- **When needed:** If covariates available to explain heterogeneity
- **Model:** theta_i ~ N(mu + beta * X_i, tau)
- **Benefit:** Explain tau, improve predictions, test moderators
- **Priority if data available:** HIGH
- **Current:** No covariates, cannot implement

**13. IPD meta-analysis**
- **When needed:** If individual participant data available
- **Model:** Hierarchical model with study-level and participant-level effects
- **Benefit:** More precise inference, participant-level heterogeneity
- **Priority if IPD available:** HIGHEST
- **Current:** Only study-level data, cannot implement

**14. Network meta-analysis**
- **When needed:** If comparing multiple interventions
- **Model:** Multivariate hierarchical with intervention contrasts
- **Benefit:** Rank interventions, indirect comparisons
- **Priority if data available:** HIGH
- **Current:** Single intervention, cannot implement

**15. Bayesian model averaging (BMA)**
- **When needed:** If multiple models have similar LOO
- **Model:** Weighted ensemble via LOO stacking
- **Benefit:** More robust inference, accounts for model uncertainty
- **Priority:** MEDIUM - Phase 4 will assess
- **Current:** Will consider in Phase 4 if Experiments 1-4 similar

**16. Dynamic/temporal meta-analysis**
- **When needed:** If studying effects over time
- **Model:** Hierarchical with temporal correlation structure
- **Benefit:** Assess time trends, predict future studies
- **Priority if temporal data:** MEDIUM
- **Current:** No temporal information, cannot implement

---

## Prioritized List (If Revision Were Needed)

**Since decision is ACCEPT, this is HYPOTHETICAL. If model were REVISED, priorities would be:**

### Tier 1: Mandatory Before Final Conclusions (ALREADY PLANNED)

1. **Experiment 2: Complete pooling comparison** - MUST DO
   - Purpose: Test if heterogeneity matters
   - Timeline: Next immediate step
   - Effort: 1-2 hours

2. **Experiment 4: Prior sensitivity** - MUST DO
   - Purpose: Quantify sensitivity to prior choice (critical for J=8)
   - Timeline: After Experiments 1-2
   - Effort: 2-3 hours

3. **Experiment 3: Robust model (conditional)** - SHOULD DO
   - Purpose: Validate normal likelihood assumption
   - Timeline: If time permits after 1-2-4
   - Effort: 2-3 hours

### Tier 2: Computational Refinements (NOT NEEDED)

4. **Increase iterations (if R-hat remains concern)** - LOW PRIORITY
   - Purpose: R-hat → 1.00
   - Benefit: Minimal (ESS already adequate)
   - Effort: 1 hour (rerun with more iterations)
   - **Assessment:** NOT NEEDED given current diagnostics

5. **Switch to HMC (if Gibbs concerns)** - LOW PRIORITY
   - Purpose: Higher ESS/iteration
   - Benefit: Marginal (Gibbs adequate for conjugate model)
   - Effort: 2-3 hours (CmdStan implementation)
   - **Assessment:** NOT NEEDED, Gibbs validated

### Tier 3: Future Research (REQUIRES MORE DATA)

6. **Collect more studies** - HIGHEST LONG-TERM PRIORITY
   - Purpose: Narrow credible intervals, improve tau estimation
   - Target: 15-20 studies
   - Benefit: HIGH - most impactful improvement
   - Effort: Months (literature search, data extraction)

7. **Collect study-level covariates** - HIGH PRIORITY
   - Purpose: Meta-regression to explain heterogeneity
   - Target: Year, location, risk of bias, intervention details
   - Benefit: HIGH - explain tau, improve predictions
   - Effort: Weeks (revisit studies, extract covariates)

8. **Investigate Study 5 characteristics** - HIGH PRIORITY
   - Purpose: Understand why only negative effect
   - Target: Population, methods, intervention differences
   - Benefit: MEDIUM - inform future studies
   - Effort: Days (review Study 5 paper)

9. **Publication bias assessment** - MEDIUM PRIORITY
   - Purpose: Test if positive mu reflects selective reporting
   - Requires: 20-30 studies (not J=8)
   - Benefit: MEDIUM - adjust for bias if present
   - Effort: Weeks (after collecting more studies)

10. **IPD meta-analysis** - LOW PRIORITY (HIGH BENEFIT IF FEASIBLE)
    - Purpose: Participant-level heterogeneity, more precision
    - Requires: Raw data from all studies (often unavailable)
    - Benefit: HIGHEST if feasible
    - Effort: Months (data access, harmonization)

### Tier 4: Extensions (BEYOND CURRENT SCOPE)

11. **Meta-regression** - Requires covariates
12. **Network meta-analysis** - Requires multiple interventions
13. **Bayesian model averaging** - Phase 4 will assess
14. **Temporal meta-analysis** - Requires time series

---

## What NOT to Prioritize

**Do NOT waste effort on:**

1. ❌ **Excluding Study 5** - No statistical justification (k < 0.7, PPC passes)
2. ❌ **Forcing R-hat < 1.00** - Already at 1.01 with excellent ESS, diminishing returns
3. ❌ **Narrowing credible intervals artificially** - Wide CIs are honest, reflect limited data
4. ❌ **Switching models without justification** - Current model adequate, alternatives for comparison not replacement
5. ❌ **Over-interpreting precise estimates** - Precision is limited by J=8, cannot fix with modeling tricks
6. ❌ **Publication bias models with J=8** - Insufficient power, wait for more studies
7. ❌ **Complex models (mixture, DP, etc.)** - Current diagnostics don't justify complexity

---

## Recommended Path Forward

**Since decision is ACCEPT, the path forward is:**

### Short Term (Next Steps):

1. **Proceed to Experiment 2** (complete pooling comparison)
   - Expected: 1-2 hours
   - Purpose: Test null hypothesis of no heterogeneity
   - Decision: LOO comparison in Phase 4

2. **Proceed to Experiment 4** (prior sensitivity)
   - Expected: 2-3 hours
   - Purpose: Quantify sensitivity to prior choice
   - Decision: Ensemble if high sensitivity, else confirm Exp 1

3. **Conditionally proceed to Experiment 3** (robust model)
   - Expected: 2-3 hours (if time permits)
   - Purpose: Validate normal likelihood
   - Decision: If nu > 30, confirm Exp 1

4. **Phase 4: Model assessment and comparison**
   - Expected: 2-3 hours
   - Purpose: Synthesize across all models
   - Decision: Final model selection or ensemble

### Medium Term (Future Research):

5. **Collect more studies** (target: 15-20)
   - Timeline: Months (systematic review)
   - Benefit: Narrow CIs, improve tau estimation
   - Priority: HIGHEST for long-term improvement

6. **Collect study-level covariates**
   - Timeline: Weeks (literature review)
   - Benefit: Explain heterogeneity via meta-regression
   - Priority: HIGH if covariates available

7. **Investigate Study 5**
   - Timeline: Days (re-read paper)
   - Benefit: Understand discrepancy
   - Priority: MEDIUM for context

### Long Term (Extensions):

8. **IPD meta-analysis** (if raw data accessible)
9. **Publication bias assessment** (when J > 20)
10. **Network meta-analysis** (if comparing interventions)

---

## Conditional Improvements

**IF Experiments 2-4 reveal issues, THEN:**

### Scenario 1: Experiment 2 strongly preferred (ΔLOO > 4)

**Interpretation:** Hierarchical model overparameterized, heterogeneity negligible

**Improvement:**
- Use Experiment 2 (complete pooling) results instead
- No refinement of Experiment 1 needed (still valid for comparison)
- **Priority:** Switch to Experiment 2 for final inference

### Scenario 2: Experiment 3 shows nu < 20

**Interpretation:** Normal likelihood inadequate, outliers matter

**Improvement:**
- Use Experiment 3 (Student-t) results instead
- No refinement of Experiment 1 needed (normal is subset of t)
- **Priority:** Switch to Experiment 3 for final inference

### Scenario 3: Experiment 4 shows extreme prior sensitivity (|tau_diff| > 5)

**Interpretation:** Data insufficient to overcome prior choice

**Improvement:**
- Use ensemble (LOO stacking) across prior choices
- Report range of estimates, acknowledge uncertainty
- Recommend collecting more studies
- **Priority:** Ensemble instead of single model

### Scenario 4: All Experiments 1-3 flag Study 5 as extreme

**Interpretation:** Study 5 may need special treatment

**Improvement:**
- Sensitivity analysis excluding Study 5
- Consider Experiment 5 (mixture model)
- Investigate Study 5 characteristics
- **Priority:** MEDIUM - assess impact, possibly exclude or model separately

---

## Summary

**Current Status:** ACCEPT (no immediate improvements needed)

**Mandatory Next Steps:**
1. Experiment 2 (complete pooling) - comparison
2. Experiment 4 (prior sensitivity) - sensitivity testing
3. Phase 4 (model assessment) - synthesis

**If Revision Were Needed, Top 3 Priorities Would Be:**
1. Experiment 2 comparison (already planned)
2. Experiment 4 prior sensitivity (already planned)
3. Collect more studies (long-term, requires new data)

**What NOT to Do:**
- Do NOT exclude Study 5 without strong justification
- Do NOT artificially narrow credible intervals
- Do NOT force R-hat < 1.00 when ESS already adequate
- Do NOT implement complex models without evidence of need

**Philosophy:**
- Model is adequate as baseline
- Comparison and sensitivity testing provide context
- Future improvements require more/better data, not model tweaking
- Honest uncertainty quantification is strength, not weakness

---

**Document Status:** COMPLETE
**Recommendation:** Proceed to Experiment 2
**No immediate model revision required**

---

**Analyst:** Model Criticism Specialist (Claude)
**Date:** 2025-10-28
**Version:** 1.0
