# Improvement Priorities for Experiment 1

**Model:** Standard Non-Centered Hierarchical Model
**Decision:** CONDITIONAL ACCEPT
**Date:** 2025-10-28

---

## Overview

While the model is accepted as scientifically adequate, several improvements and sensitivity analyses are recommended to strengthen confidence in the results and clarify the model's limitations. This document prioritizes these improvements based on their impact on scientific conclusions and feasibility.

---

## Priority Tier 1: CRITICAL (Must Complete Before Publication)

These items are essential for responsible scientific reporting and should be completed before any final claims are made.

### 1.1 Complete Posterior Predictive Check

**Status:** In progress, expected to complete soon

**Why Critical:**
- Cannot fully assess model adequacy without PPC
- Need to verify model reproduces observed data patterns
- Required for validating likelihood specification
- Essential validation step in Bayesian workflow

**What to Check:**
- Coverage: What % of observed y_i fall within 95% posterior predictive intervals?
- Distribution: Does posterior predictive distribution match observed distribution shape?
- Extremes: Does model reproduce range of observed effects?
- Residuals: Any systematic patterns in posterior predictive residuals?

**Success Criteria:**
- >90% of observed values within 95% intervals
- No systematic misfit patterns
- Distribution shapes compatible
- All Pareto k < 0.7 (already verified via LOO)

**Expected Outcome:** PASS based on LOO diagnostics

**If Fails:**
- Investigate which observations are poorly predicted
- Consider alternative likelihood (Student-t for robustness)
- Check for missing structure (unlikely with n=8)
- May need to REVISE model

**Time Required:** 1-2 hours (code already exists, just needs to run)

**Blocking:** Cannot make final publication-ready claims without this

---

### 1.2 Document τ Identifiability Limitations

**Status:** Not yet done

**Why Critical:**
- Posterior τ ≈ 3.6 conflicts with classical analysis (τ=0)
- Users may misinterpret posterior τ>0 as strong evidence for heterogeneity
- Essential for honest scientific communication
- Prevents over-interpretation of results

**What to Document:**

1. **Quantitative evidence for weak identifiability:**
   - Simulation validation: 0% coverage when true τ=0
   - Cannot distinguish τ=0 from τ≈5 with n=8
   - Posterior SD(τ) ≈ mean(τ) → high relative uncertainty
   - 95% HDI spans order of magnitude (0 to 9.2)

2. **Reconciliation with EDA:**
   - Classical tests: Q p=0.696, I²=0%, τ²=0
   - Bayesian posterior: τ=3.6±3.15
   - Explain this is NOT a contradiction:
     * Classical estimate hits boundary (τ²≥0)
     * Bayesian prior prevents collapse to boundary
     * Both expressing same epistemic state: weak evidence

3. **Implications for interpretation:**
   - Posterior τ>0 reflects prior influence + data
   - Cannot conclude heterogeneity is "real"
   - Wide posterior expresses genuine uncertainty
   - Different priors would yield different τ estimates

**Where to Document:**
- Methods section: Explain prior choice and its implications
- Results section: Present τ with full posterior, emphasize uncertainty
- Discussion section: Reconcile with classical analysis, acknowledge limitations
- Supplement: Include prior sensitivity analysis

**Success Criteria:**
- Reader understands τ is weakly identified
- No claims of "significant heterogeneity" without caveats
- Prior influence acknowledged explicitly
- Uncertainty emphasized over point estimates

**Time Required:** 2-3 hours for writing and literature review

**Blocking:** Cannot responsibly publish without this

---

### 1.3 Report Full Uncertainty (Not Just Point Estimates)

**Status:** Partially done (posteriors computed, but reporting template needed)

**Why Critical:**
- Point estimates are misleading with high uncertainty
- 95% HDIs are essential for honest inference
- Particularly important for θ_i (individual schools)
- Standard practice in Bayesian reporting

**What to Report:**

**For μ (grand mean):**
```
Mean: 7.36
SD: 4.32
95% HDI: [-0.56, 15.60]
Interpretation: Likely positive effect, substantial uncertainty
```

**For τ (between-school SD):**
```
Mean: 3.58
SD: 3.15
Median: ~3.0 (estimate from HDI)
95% HDI: [0.00, 9.21]
Interpretation: Weakly identified, high uncertainty
Context: Classical τ²=0, Q p=0.696 (no evidence for heterogeneity)
Caveat: Estimate reflects prior influence given limited data
```

**For θ_i (school effects):**
```
Present as forest plot with:
- Posterior means (points)
- 95% HDIs (thick intervals)
- 50% HDIs (thin intervals)
- Observed effects (comparison)
- Grand mean (reference line)

Table format:
School | Observed | Post. Mean | 95% HDI | Shrinkage
1      | 28       | 8.90       | [-2.0, 19.9] | 85%
...
```

**Visualization Requirements:**
- Forest plot: θ_i posteriors vs observed
- Shrinkage plot: Observed → Posterior with arrows
- Posterior density: τ with classical estimate marked
- Joint posterior: (μ, τ) relationship

**Success Criteria:**
- All reported estimates include uncertainty
- Visualizations emphasize intervals over points
- Tables include both means and HDIs
- Text interpretation reflects uncertainty

**Time Required:** 2-4 hours for figure creation and polishing

**Blocking:** Partial - can publish draft, but needs refinement

---

## Priority Tier 2: STRONGLY RECOMMENDED (Should Complete Before Final Acceptance)

These analyses are highly valuable for understanding model behavior and robustness, and should be completed before making strong claims or selecting this as the final model.

### 2.1 Prior Sensitivity Analysis on τ

**Status:** Not done

**Why Strongly Recommended:**
- τ posterior likely sensitive to prior choice
- Simulation validation suggests weak identifiability
- EDA conflicts with posterior (suggests prior driving inference)
- Essential for understanding robustness

**Specific Analyses:**

**Test 1: Tighter prior - Half-Cauchy(0, 1)**
- Expected: Posterior τ ≈ 1-2 (smaller than current)
- Purpose: See if tighter prior pulls τ toward zero
- Interpretation: If much smaller, confirms prior sensitivity

**Test 2: Looser prior - Half-Cauchy(0, 10)**
- Expected: Posterior τ ≈ 4-6 (larger than current)
- Purpose: See if data can constrain τ with vaguer prior
- Interpretation: If much larger, confirms data are weak

**Test 3: Different family - Half-Normal(0, 5)**
- Expected: Posterior τ ≈ 2-3 (more mass near zero)
- Purpose: Test sensitivity to prior shape, not just scale
- Interpretation: Half-Normal concentrates mass lower than Half-Cauchy

**Optional Test 4: Uniform(0, 20)**
- Expected: Posterior τ ≈ 4-7
- Purpose: Least informative prior (within reason)
- Interpretation: Shows what data alone suggest

**What to Report:**

Summary table:
```
Prior              | Post. τ Mean | Post. τ Median | 95% HDI
-------------------|--------------|-----------------|---------
Half-Cauchy(0,1)   | ~2.0         | ~1.5            | [0, 5]
Half-Cauchy(0,5)*  | 3.58         | ~3.0            | [0, 9.2]
Half-Cauchy(0,10)  | ~5.0         | ~4.0            | [0, 12]
Half-Normal(0,5)   | ~2.5         | ~2.0            | [0, 7]

*Current choice
```

**Also check:**
- How much does μ change? (Expected: minimal)
- How much do θ_i change? (Expected: minimal)
- Does ELPD change? (Expected: <1 difference)
- Do substantive conclusions change?

**Success Criteria:**
- Document range of plausible τ estimates
- Show whether conclusions are robust
- If highly sensitive: report range, don't claim single value
- If robust: strengthens confidence in current choice

**Expected Finding:**
- τ posterior will vary 2x to 3x across priors
- μ and θ_i will be stable (±10%)
- Substantive conclusions (strong shrinkage, uncertain heterogeneity) unchanged

**Implications:**
- If sensitive: Report "τ estimated between 2 and 6 depending on prior choice"
- If robust: "τ estimated around 3-4 across reasonable priors"
- Either way: Strengthens paper by showing due diligence

**Time Required:** 4-6 hours (refit models, create comparison plots, analyze)

**Blocking:** Not blocking for ACCEPT, but critical for strong claims about τ

---

### 2.2 Compare to Complete Pooling Model

**Status:** Not done

**Why Strongly Recommended:**
- p_eff = 1.03 suggests data support simple model
- Classical analysis (I²=0%) suggests complete pooling
- Good scientific practice to compare nested models
- May find simpler model is adequate

**Model Specification:**
```python
# Complete Pooling Model
with pm.Model() as model_pooling:
    # Prior on grand mean only
    mu = pm.Normal('mu', mu=0, sigma=20)

    # Likelihood (no θ_i, just single μ)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_obs)
```

**Comparison Metrics:**

1. **LOO ELPD:**
   - Current hierarchical: -30.73 ± 1.04
   - Complete pooling: ? (likely similar)
   - Difference: If |Δ ELPD| < 2, models equivalent
   - If pooling is equal/better: hierarchical structure not justified

2. **Parameter Comparison:**
   - μ should be nearly identical
   - Complete pooling has no τ (simpler)
   - Effective parameters: pooling = 1, hierarchical = 1.03

3. **Posterior Predictive:**
   - Both should cover observed data well
   - Hierarchical may have slightly wider intervals (accounts for τ uncertainty)

4. **Interpretation:**
   - Complete pooling: All schools have same effect
   - Hierarchical: Schools may differ, but data don't require it
   - If ELPD similar: Parsimony favors complete pooling

**What to Report:**

```
Model Comparison (LOO Cross-Validation):

Model                ELPD     SE    Δ ELPD  Δ SE   Weight
----------------------------------------------------------
Complete Pooling     -30.X    1.0   0.0     --     0.XX
Hierarchical         -30.73   1.04  -0.X    0.X    0.XX

Interpretation: [Models are equivalent / Hierarchical preferred /
                 Pooling preferred] based on predictive performance.

Given p_eff=1.03 for hierarchical model, the data provide minimal
evidence for between-school variation, and complete pooling may be
a more parsimonious choice.
```

**Success Criteria:**
- Quantitative comparison completed
- Interpretation provided
- If pooling equivalent: Present both as valid
- If hierarchical better: Justify added complexity

**Expected Outcome:**
- ELPD difference < 2 (models equivalent)
- Pooling is simpler and aligns with EDA
- Could recommend pooling or say "both are valid"
- Hierarchical is more conservative (allows heterogeneity)

**Implications:**
- If pooling preferred: Simpler model for reporting, easier interpretation
- If equivalent: Choice based on philosophy (conservative vs parsimonious)
- If hierarchical wins: Justifies current approach

**Time Required:** 3-4 hours (fit model, run LOO, compare, interpret)

**Blocking:** Not blocking for ACCEPT, but important for model selection

---

### 2.3 Enhanced Reporting Template

**Status:** Outline exists (in critique), needs implementation

**Why Strongly Recommended:**
- Ensures consistent, accurate reporting
- Prevents common misinterpretations
- Useful for paper, presentations, reports
- Creates reusable template

**Components to Create:**

**1. Results Summary Template:**
- Pre-written paragraph with blanks for values
- Includes all necessary caveats
- Appropriate uncertainty language
- Ready for Methods/Results sections

**2. Visualization Suite:**
- Forest plot (publication quality)
- Shrinkage analysis plot
- Posterior distributions for μ, τ
- Joint (μ, τ) posterior
- Posterior predictive check plots

**3. Table Templates:**
- Parameter summary (mean, SD, HDI)
- School-level estimates with shrinkage
- Model comparison (if done)
- Convergence diagnostics

**4. Discussion Points:**
- Interpretation of τ uncertainty
- Comparison to classical analysis
- Limitations and caveats
- Practical implications
- Recommendations for future work

**Success Criteria:**
- Complete reporting package ready
- Can generate publication-quality output
- Includes all necessary caveats
- Scientifically defensible

**Time Required:** 4-6 hours (plot refinement, template creation, documentation)

**Blocking:** Not blocking, but improves communication quality

---

## Priority Tier 3: OPTIONAL ENHANCEMENTS (Nice to Have)

These analyses would strengthen the paper but are not essential for the current model's acceptance.

### 3.1 Propagate Uncertainty in σ_i

**Status:** Not done

**Why Optional:**
- Current model treats σ_i as known (common assumption)
- In reality, σ_i were estimated from within-school data
- Ignoring this adds slight overconfidence
- But likely minor impact given already large uncertainties

**How to Implement:**

**Option A: Sensitivity Analysis**
- Assume σ_i have 10% estimation error
- Perturb σ_i by ±10% and refit
- See how much posteriors change

**Option B: Full Uncertainty Propagation**
- Model σ_i as estimated with uncertainty
- Requires within-school data (may not be available)
- More complex model, may not converge as well

**Expected Impact:**
- Wider posterior intervals for all parameters (by ~5-10%)
- Conclusions unlikely to change
- More honest uncertainty quantification

**Time Required:** 2-3 hours for sensitivity, 6-8 hours for full propagation

**Value:** LOW - unlikely to change conclusions, but more rigorous

---

### 3.2 Leave-K-Out Stability Analysis

**Status:** Not done

**Why Optional:**
- LOO already done (all Pareto k < 0.7)
- Leave-K-out tests robustness to multiple observations
- Useful for understanding τ sensitivity to specific schools
- But n=8 limits what can be tested (leave-3-out is 37.5% of data)

**What to Test:**

**Leave-one-out influence (extended):**
- Refit model removing each school
- See how τ posterior changes
- Identify influential schools for τ estimate

**Leave-two-out combinations:**
- Test pairs of schools (28 combinations)
- See if τ estimate is stable
- Check for high-influence pairs

**Expected Findings:**
- τ estimate will vary substantially (it's weakly identified)
- Removing high-variance schools (1, 3) may lower τ
- Removing low-variance schools (5, 7) may raise τ
- Confirms τ is data-sensitive (not just prior-sensitive)

**Time Required:** 4-6 hours (many refits, synthesis)

**Value:** MEDIUM - informative but not critical

---

### 3.3 Robust Likelihood (Student-t)

**Status:** Not done

**Why Optional:**
- Current normal likelihood appears adequate
- No outliers detected (all |z| < 2)
- Student-t would be more robust to extreme values
- But may be unnecessary complexity

**Model Specification:**
```python
# Replace Normal with Student-t
y = pm.StudentT('y', nu=4, mu=theta, sigma=sigma, observed=y_obs)
```

**When Useful:**
- If concerned about School 1 (y=28) being too influential
- If want to test robustness to distributional assumptions
- If reviewer asks "what about outliers?"

**Expected Outcome:**
- Minimal difference from normal likelihood
- Possibly slightly less shrinkage for extreme schools
- Computational complexity may increase slightly

**Time Required:** 2-3 hours

**Value:** LOW - unlikely needed, but good robustness check

---

### 3.4 Alternative Non-Centered Parameterizations

**Status:** Current non-centered is working perfectly

**Why Optional:**
- Current approach has zero divergences
- Could test "partially non-centered" for comparison
- Mostly academic interest, not practical need

**When Useful:**
- If writing methodological paper about parameterizations
- If teaching/demonstrating different approaches
- Not needed for applied analysis

**Time Required:** 4-6 hours

**Value:** VERY LOW - current approach is excellent

---

## Implementation Roadmap

### Phase 1: Critical Items (1-2 days)

**Day 1:**
1. Complete posterior predictive check (2 hours)
2. Document τ identifiability limitations (3 hours)
3. Create uncertainty reporting template (2 hours)

**Day 2:**
4. Refine visualizations for publication (3 hours)
5. Write methods/results sections (3 hours)
6. Internal review and revision (2 hours)

**Deliverable:** Publication-ready results with appropriate caveats

---

### Phase 2: Strongly Recommended (2-3 days)

**Day 3:**
1. Fit Half-Cauchy(0,1) model (1 hour)
2. Fit Half-Cauchy(0,10) model (1 hour)
3. Fit Half-Normal(0,5) model (1 hour)
4. Compare posteriors and create sensitivity plot (2 hours)
5. Document findings (2 hours)

**Day 4:**
6. Fit complete pooling model (1 hour)
7. Run LOO comparison (1 hour)
8. Create comparison visualizations (2 hours)
9. Interpret and document (2 hours)

**Day 5:**
10. Synthesize all sensitivity analyses (3 hours)
11. Update manuscript with findings (3 hours)

**Deliverable:** Robust analysis with sensitivity checks completed

---

### Phase 3: Optional Enhancements (1-2 days, as needed)

**Only if:**
- Reviewers request
- Time/resources permit
- Methodological interest

**Suggested priority order:**
1. Robust likelihood (quick win if needed)
2. Leave-K-out (if τ sensitivity is major concern)
3. σ_i uncertainty (if reviewer raises issue)

---

## Success Metrics

### Minimum for Conditional Accept → Unconditional Accept

- [x] All computational diagnostics passed
- [⏸] Posterior predictive check completed and passed
- [⏸] τ identifiability documented in manuscript
- [⏸] Full uncertainty reported (not just point estimates)

### Ideal for Strong Publication

- [ ] Prior sensitivity analysis completed
- [ ] Complete pooling comparison done
- [ ] Robust conclusions documented
- [ ] All limitations acknowledged
- [ ] Replication materials provided

### Exceptional (Beyond Requirements)

- [ ] Uncertainty in σ_i addressed
- [ ] Leave-K-out stability shown
- [ ] Robust likelihood compared
- [ ] Tutorial-quality documentation

---

## Resource Estimates

### Time Investment

- **Critical items:** 8-12 hours (1-2 days)
- **Strongly recommended:** 12-16 hours (2-3 days)
- **Optional enhancements:** 8-12 hours (1-2 days)

**Total for complete analysis:** 28-40 hours (4-7 days)

**Minimum for acceptance:** 8-12 hours (critical items only)

### Computational Resources

- **All analyses:** Can run on laptop
- **Runtime per model fit:** ~20 seconds
- **Total runtime:** <30 minutes for all refits
- **Storage:** <100 MB for all outputs

**No special infrastructure needed**

---

## Risk Assessment

### Risks of NOT Completing Critical Items

**High Risk:**
- Misinterpretation of τ>0 as strong evidence
- Over-confident claims about heterogeneity
- Criticism from reviewers about identifiability
- Retraction if errors discovered post-publication

**Mitigation:** Complete all Tier 1 items before publication

### Risks of NOT Completing Recommended Items

**Medium Risk:**
- Uncertainty about robustness of findings
- Valid criticism about prior sensitivity
- Missed opportunity to strengthen paper
- Questions about model necessity

**Mitigation:** Complete Tier 2 items before claiming strong conclusions

### Risks of NOT Completing Optional Items

**Low Risk:**
- Minor concerns about assumptions
- Slightly less comprehensive analysis
- Potential reviewer questions

**Mitigation:** Address if raised in review, not essential upfront

---

## Conclusion

The improvement priorities are structured to ensure:

1. **Critical items** protect scientific integrity
2. **Recommended items** strengthen confidence in results
3. **Optional items** provide extra robustness if needed/desired

**Minimum viable path:** Complete Tier 1 (8-12 hours)

**Recommended path:** Complete Tier 1 + Tier 2 (20-28 hours)

**Comprehensive path:** All tiers (28-40 hours)

Given the model's strong computational performance and fundamental soundness, **most effort should focus on proper interpretation and reporting rather than model modification**. The key improvements are about communication and sensitivity analysis, not fixing broken components.

---

**Next Steps:**
1. Review priorities with research team
2. Allocate resources for critical items
3. Set timeline for completion
4. Begin with PPC (already in progress)
5. Move to documentation and sensitivity analyses

**Timeline Recommendation:** Complete Tier 1 + Tier 2 within 1 week, defer Tier 3 unless specifically requested.
