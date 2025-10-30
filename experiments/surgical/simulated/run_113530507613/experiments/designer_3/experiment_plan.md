# Experiment Plan: Bayesian Regression Models with Covariates

**Designer:** Model Designer 3 (Regression/Covariate Specialist)
**Date:** 2025-10-30
**Status:** Ready for Implementation

---

## Executive Summary

Based on EDA findings showing strong heterogeneity (ICC = 0.42) but no significant linear correlation with sample size (r = -0.34, p = 0.278), I propose testing whether covariates can explain between-group variation through formal Bayesian regression. This experiment plan includes 3 competing models with explicit falsification criteria.

**Critical Insight:** The absence of significant correlation (p = 0.278) does NOT imply covariates are uninformative. With J=12, frequentist tests lack power. Bayesian model comparison via LOO-CV provides a rigorous test while quantifying uncertainty.

---

## Problem Formulation

### Research Questions

1. **Does sample size systematically predict success rates?**
   - EDA: r = -0.34, p = 0.278 (underpowered)
   - Hypothesis: Log-transform may reveal non-linear relationship
   - Implication: If true, suggests study design confounding

2. **Is there hidden structure in group ordering?**
   - EDA: No linear trend (p = 0.69), but 3 clusters identified
   - Hypothesis: Non-linear (quadratic) pattern may exist
   - Implication: If true, groups not exchangeable

3. **Does the size-response vary across groups?**
   - EDA: Large variance ratio (2.78), two extreme outliers
   - Hypothesis: Heterogeneous slopes (interactions)
   - Implication: If true, suggests subpopulations with different dynamics

### Competing Hypotheses

**Hypothesis 1 (H1): Fixed Covariate Effect**
- Sample size has constant effect across all groups
- Model 1 tests this with hierarchical regression
- **Falsification:** 95% CI for beta_1 includes zero AND R² < 0.05

**Hypothesis 2 (H2): Sequential Structure**
- Group ordering reflects underlying process (temporal/spatial)
- Model 2 tests this with quadratic regression
- **Falsification:** Both beta_1 and beta_2 include zero

**Hypothesis 3 (H3): Varying Effects**
- Size-response differs across groups (interactions)
- Model 3 tests this with random slopes
- **Falsification:** tau_gamma < 0.1 AND no LOO improvement

**Null Hypothesis (H0): No Covariate Effects**
- All heterogeneity is unexplained random variation
- Tested by comparing all models to baseline (random effects only)
- **Support for H0:** All ΔLOO < 2 with narrow standard errors

---

## Model Specifications

### Model 1: Hierarchical Logistic Regression with Sample Size Covariate

**Priority:** HIGHEST (most theoretically justified)

**Mathematical Form:**
```
r_j ~ Binomial(n_j, p_j)
logit(p_j) = alpha_j
alpha_j ~ Normal(mu_j, tau)
mu_j = beta_0 + beta_1 * log(n_j / n_mean)

Priors:
  beta_0 ~ Normal(-2.6, 1.0)
  beta_1 ~ Normal(0, 0.5)
  tau ~ Normal^+(0, 0.5)
```

**Key Parameters:**
- **beta_1:** Effect of log(sample size) on success rate (logit scale)
- **tau:** Residual between-group variance NOT explained by sample size
- **R²:** Proportion of variance explained by covariate

**Decision Criteria:**
- **Strong evidence:** |beta_1| > 0.3, R² > 0.15, ΔLOO > 4
- **Weak evidence:** 0.1 < |beta_1| < 0.3, R² = 0.05-0.15, ΔLOO = 2-4
- **No evidence:** |beta_1| < 0.1, R² < 0.05, ΔLOO < 2

**Falsification Triggers:**
1. 95% CI for beta_1 overlaps zero substantially
2. R² < 0.05 (explains < 5% of variance)
3. ΔLOO vs baseline < 2 (no improvement)
4. tau_model1 ≈ tau_baseline (no reduction in residual variance)

**If falsified:** Sample size is not a meaningful predictor → Focus on random effects only

---

### Model 2: Hierarchical Logistic Regression with Quadratic Group Effect

**Priority:** MEDIUM (exploratory, tests exchangeability)

**Mathematical Form:**
```
r_j ~ Binomial(n_j, p_j)
logit(p_j) = alpha_j
alpha_j ~ Normal(mu_j, tau)
mu_j = beta_0 + beta_1 * group_scaled_j + beta_2 * group_scaled_j^2

where: group_scaled = (group_id - 6.5) / 6.5  # Scale to [-1, 1]

Priors:
  beta_0 ~ Normal(-2.6, 1.0)
  beta_1 ~ Normal(0, 0.5)
  beta_2 ~ Normal(0, 0.5)
  tau ~ Normal^+(0, 0.5)
```

**Key Parameters:**
- **beta_1:** Linear trend in group ordering
- **beta_2:** Quadratic curvature (U-shape if positive, inverted-U if negative)
- **peak_location:** Where the parabola peaks (in scaled units)

**Decision Criteria:**
- **Strong evidence:** |beta_2| > 0.3, peak_location in [-1, 1], ΔLOO > 4
- **Weak evidence:** 0.1 < |beta_2| < 0.3, ΔLOO = 2-4
- **No evidence:** Both |beta_1| and |beta_2| < 0.1, ΔLOO < 2

**Falsification Triggers:**
1. Both beta_1 and beta_2 include zero (no polynomial effect)
2. R² < 0.05
3. Peak/trough outside observed range (extrapolation)
4. Pattern doesn't match visual inspection

**If falsified:** Group ordering is arbitrary → Groups are exchangeable

---

### Model 3: Hierarchical Logistic Regression with Random Slopes

**Priority:** LOWEST (complex, may overfit with J=12)

**Mathematical Form:**
```
r_j ~ Binomial(n_j, p_j)
logit(p_j) = alpha_j + gamma_j * log(n_j / n_mean)

Hierarchical structure:
  alpha_j ~ Normal(beta_0, tau_alpha)    # Random intercepts
  gamma_j ~ Normal(beta_1, tau_gamma)    # Random slopes
  (alpha_j, gamma_j) ~ MVN with correlation rho

Priors:
  beta_0 ~ Normal(-2.6, 1.0)
  beta_1 ~ Normal(0, 0.5)
  tau_alpha ~ Normal^+(0, 0.5)
  tau_gamma ~ Normal^+(0, 0.3)  # More conservative
  rho ~ Uniform(-1, 1)
```

**Key Parameters:**
- **tau_gamma:** Variation in slopes across groups
- **rho:** Correlation between intercepts and slopes
- **prop_var_slopes:** Proportion of variance due to varying slopes

**Decision Criteria:**
- **Strong evidence:** tau_gamma > 0.2, prop_var_slopes > 0.15, ΔLOO > 4 vs Model 1
- **Weak evidence:** tau_gamma = 0.1-0.2, ΔLOO = 2-4
- **No evidence:** tau_gamma < 0.1, ΔLOO < 2

**Falsification Triggers:**
1. tau_gamma < 0.1 (slopes don't meaningfully vary)
2. prop_var_slopes < 0.05
3. ΔLOO < 0 compared to Model 1 (simpler fixed slopes)
4. Persistent computational issues (divergences, poor mixing)
5. rho at boundaries (±0.99) suggesting misspecification

**If falsified:** Slopes are homogeneous → Use Model 1 (simpler)

---

## Implementation Plan

### Phase 1: Setup (Day 1, ~30 minutes)

**Tasks:**
1. Validate data structure (binomial constraints)
2. Create derived covariates:
   - `log_n_centered = log(n_trials) - mean(log(n_trials))`
   - `group_scaled = (group_id - 6.5) / 6.5`
3. Compile Stan models (one-time cost)
4. Run prior predictive checks (verify priors are sensible)

**Success Criteria:**
- All data constraints satisfied
- Stan models compile without errors
- Prior predictive samples in reasonable range (p ∈ [0, 0.3])

---

### Phase 2: Model Fitting (Day 1-2, ~2 hours)

**Sampling Strategy:**
- 4 chains × 2000 iterations (1000 warmup + 1000 sampling)
- adapt_delta = 0.95 (conservative for robustness)
- Target: Rhat < 1.01, ESS > 400, divergences < 1%

**Fitting Order:**
1. **Model 1** (sample size covariate) - ~30 seconds
2. **Model 2** (quadratic group effect) - ~30 seconds
3. **Baseline** (random effects only) - ~20 seconds
4. **Model 3** (random slopes) - ~5 minutes (optional, if Models 1-2 show promise)

**Success Criteria:**
- All models converge (Rhat < 1.01)
- ESS > 400 for all parameters
- < 1% divergent transitions
- Reasonable parameter estimates (no extreme values)

**Checkpoints:**
- After Model 1: If divergences > 1%, increase adapt_delta to 0.99
- After Model 2: If both fail, consider data issues or model misspecification
- Before Model 3: Only proceed if Models 1-2 show ΔLOO > 2

---

### Phase 3: Model Comparison (Day 2, ~1 hour)

**LOO-CV Analysis:**
1. Compute LOO for all fitted models
2. Check Pareto-k diagnostics (should be < 0.7)
3. Compare elpd_loo with standard errors
4. Create comparison table

**Decision Rules:**

**Scenario A: One Model Dominates (ΔLOO > 4)**
→ Clear winner identified
→ Report best model's coefficients
→ Investigate scientific implications
→ Run sensitivity analysis on priors

**Scenario B: Weak Evidence (2 < ΔLOO < 4)**
→ Uncertain which model is best
→ Report top 2 models with uncertainty
→ Consider model averaging
→ Acknowledge limitations (small J)

**Scenario C: No Model Wins (all ΔLOO < 2)**
→ **Covariates are uninformative**
→ Accept null hypothesis (H0)
→ Recommend random effects only (Designer 1)
→ Focus on partial pooling, not covariate effects

**Scenario D: All Models Fail Diagnostics**
→ **Fundamental model misspecification**
→ Pivot to alternative approach:
  - Mixture models (Designer 2)
  - Robust likelihoods (Beta-binomial)
  - Non-parametric methods (Gaussian processes)

**Red Flags That Trigger Pivot:**
- Persistent divergences across all models (> 1%)
- Extreme Pareto-k values (> 0.7 for > 50% of groups)
- Posterior predictive checks fail systematically
- Parameter estimates are scientifically implausible

---

### Phase 4: Validation (Day 2-3, ~2 hours)

**Posterior Predictive Checks:**
1. Generate replicated datasets from posterior
2. Compare test statistics:
   - Total successes (should match observed)
   - Variance of success rates (capture heterogeneity?)
   - Number of outliers (capture Groups 4 and 8?)
3. Plot observed vs. predicted with uncertainty

**Target:** p-value ∈ [0.05, 0.95] for all test statistics

**Sensitivity Analysis:**
1. Refit best model with:
   - Weaker priors: N(0, 2) instead of N(0, 0.5)
   - Stronger priors: N(0, 0.25)
2. Check robustness of conclusions
3. If results change substantially, report full range

**Expected Outcome:**
- Coefficients stable within ±20%
- LOO ranking unchanged
- Scientific conclusions robust

---

### Phase 5: Interpretation (Day 3, ~2 hours)

**For Each Model:**

**Model 1 (if wins):**
- Report: beta_1, R², effect size on probability scale
- Interpret: "Doubling sample size associated with X% change in success rate"
- Discuss: Why might large studies differ? (confounding, selection)
- Visualize: Success rate vs. log(n_trials) with fitted line

**Model 2 (if wins):**
- Report: beta_1, beta_2, peak location, curvature direction
- Interpret: "Success rates follow U-shape (or inverted-U) pattern"
- Discuss: What does group_id represent? (temporal? spatial?)
- Visualize: Success rate vs. group_id with fitted curve

**Model 3 (if wins):**
- Report: tau_gamma, rho, prop_var_slopes, range of slopes
- Interpret: "Size-response varies across groups (heterogeneous slopes)"
- Discuss: Which groups have extreme slopes? Why?
- Visualize: Individual group regression lines

**Null Case (if all ΔLOO < 2):**
- Report: "No covariates explain heterogeneity"
- Interpret: "True random differences OR unmeasured covariates"
- Recommend: Focus on random effects model (Designer 1)
- Discuss: What unmeasured factors might explain variation?

---

## Falsification Strategy

### Decision Points for Major Pivots

**Decision Point 1: After Model 1 Fitting**

**If Model 1 shows:**
- No convergence issues (Rhat < 1.01)
- beta_1 credibly non-zero (95% CI excludes 0)
- ΔLOO > 4 vs baseline

→ **Action:** Proceed to validation, skip Models 2-3 (unless scientific interest)

**If Model 1 shows:**
- Convergence issues OR
- beta_1 includes zero AND R² < 0.05 AND ΔLOO < 2

→ **Action:** Proceed to Model 2, sample size likely uninformative

---

**Decision Point 2: After Models 1-2 Fitting**

**If both models:**
- Fail diagnostics (Rhat > 1.01, divergences > 1%)
- Show extreme parameters
- ΔLOO < 2 vs baseline

→ **PIVOT:** Abandon regression approach
→ **Alternatives:**
  1. Random effects only (Designer 1)
  2. Mixture models (Designer 2)
  3. Beta-binomial (for overdispersion)

**If at least one model:**
- Converges cleanly
- Shows ΔLOO > 2
- Passes posterior predictive checks

→ **Action:** Proceed to validation and interpretation

---

**Decision Point 3: Model Class Pivot**

**Abandon ALL regression models if:**

1. **Computational failures across all models:**
   - Persistent divergences despite reparameterization
   - Rhat > 1.05 for key parameters
   - ESS < 200 despite long runs

2. **Poor predictive performance:**
   - All posterior predictive checks fail (p < 0.05 or > 0.95)
   - Cannot capture outliers (Groups 4, 8)
   - Systematic bias in predictions

3. **No improvement over baseline:**
   - All ΔLOO < 2 with narrow SE (< 1.5)
   - All R² < 0.05
   - All coefficient CIs include zero

4. **Scientifically implausible results:**
   - Extreme effect sizes (e.g., doubling n → 50% rate change)
   - Posterior-prior conflict (data fights prior)
   - Contradicts EDA without explanation

→ **Pivot Strategy:**
- Document why regression failed
- Switch to Designer 1's random effects model
- Consider Designer 2's mixture model for clusters
- Recommend collecting additional covariates

---

## Expected Outcomes and Evidence Thresholds

### Outcome 1: Sample Size Explains Heterogeneity (H1 Supported)

**Evidence Required:**
- beta_1: 95% CI excludes zero, |beta_1| > 0.2
- R²: > 0.15 (explains > 15% of variance)
- ΔLOO: > 4 vs baseline (SE < ΔLOO)
- Effect size: Substantively meaningful (> 2 percentage points)

**Interpretation:**
- Larger studies systematically differ
- Likely confounding (study design, population)
- Need domain expertise to explain WHY

**Next Steps:**
- Report with caution (correlation ≠ causation)
- Investigate what sample size proxies for
- Consider adjusting for confounders in future studies

**Probability Given EDA:** ~30% (EDA found r = -0.34, but not significant)

---

### Outcome 2: Sequential Structure Detected (H2 Supported)

**Evidence Required:**
- beta_2: 95% CI excludes zero, |beta_2| > 0.2
- Pattern: Matches visual inspection
- ΔLOO: > 4 vs baseline
- Peak location: Within [-1, 1] (not extrapolation)

**Interpretation:**
- Group ordering is NOT arbitrary
- Temporal, spatial, or ordinal structure exists
- Groups are NOT exchangeable

**Next Steps:**
- Investigate what group_id represents
- If temporal: Consider time series models
- If spatial: Consider spatial models
- If ordinal: Consider ordinal regression

**Probability Given EDA:** ~20% (EDA found no linear trend, but clusters exist)

---

### Outcome 3: Heterogeneous Slopes (H3 Supported)

**Evidence Required:**
- tau_gamma: > 0.15, 95% CI excludes 0.1
- prop_var_slopes: > 0.15
- ΔLOO: > 4 vs Model 1 (fixed slopes)
- Posterior: Wide range of slopes, not all near beta_1

**Interpretation:**
- Size-response varies across groups
- Some groups sensitive to sample size, others not
- Suggests subpopulations with different dynamics

**Next Steps:**
- Identify which groups have extreme slopes
- Consider mixture model with covariates
- Investigate scientific explanation for heterogeneity

**Probability Given EDA:** ~15% (ambitious with J=12, may lack power)

---

### Outcome 4: No Covariate Effects (H0 Supported)

**Evidence Required:**
- All models: ΔLOO < 2 vs baseline (with SE < 2)
- All R²: < 0.05
- All coefficients: 95% CIs include zero
- Robustness: Consistent across prior specifications

**Interpretation:**
- Covariates (n_trials, group_id) are uninformative
- Heterogeneity is "pure noise" (or unmeasured covariates)
- Random effects model is appropriate

**Next Steps:**
- Focus on Designer 1's random effects model
- Report ICC and shrinkage
- Recommend collecting additional covariates
- Consider Designer 2's mixture model for clusters

**Probability Given EDA:** ~35% (highest prior probability, consistent with p = 0.278)

---

## Warning Signs and Stopping Rules

### Red Flags That Require Investigation

**Computational Red Flags:**
- [ ] Divergent transitions > 1% (model misspecification)
- [ ] Rhat > 1.01 for any parameter (poor convergence)
- [ ] ESS < 400 despite long runs (poor mixing)
- [ ] Compilation errors (Stan syntax issues)

**Statistical Red Flags:**
- [ ] Pareto-k > 0.7 for > 50% of groups (influential outliers)
- [ ] Posterior predictive p-value < 0.05 or > 0.95 (poor fit)
- [ ] Extreme parameter values (|beta| > 2, tau > 2)
- [ ] Prior-posterior conflict (posterior = prior)

**Scientific Red Flags:**
- [ ] Results contradict EDA without explanation
- [ ] Effect sizes implausible (e.g., 50% change)
- [ ] Predictions outside [0, 1] probability range
- [ ] Coefficients change sign across models

### Stopping Rules

**Stop and Pivot After:**

1. **3 failed model fits** (same model, different initializations)
   → Problem: Fundamental misspecification
   → Action: Try different model class

2. **All models show ΔLOO < 2** with narrow SE (< 1.5)
   → Problem: Covariates genuinely uninformative
   → Action: Accept H0, use random effects only

3. **Persistent computational issues** across all models
   → Problem: Data-model mismatch
   → Action: Check data, try simpler models

4. **Sensitivity analysis shows instability**
   → Problem: Results driven by priors, not data
   → Action: Collect more data or use stronger priors

### Success Criteria for Experiment

**Experiment succeeds if ONE of:**
1. Clear model winner identified (ΔLOO > 4)
2. Clear evidence for H0 (all ΔLOO < 2, narrow SE)
3. Computational issues prompt productive pivot

**Experiment fails if:**
1. No conclusion possible (all ΔLOO = 2-4, wide SE)
2. Results unstable across specifications
3. Cannot falsify or support any hypothesis

---

## Alternative Approaches If Regression Fails

### Pivot Plan

**If all regression models fail (Scenario D above):**

**Alternative 1: Random Effects Only (Designer 1)**
- Simplest approach
- Focus on partial pooling and shrinkage
- Report ICC, not covariate effects
- **When:** No covariates help (ΔLOO < 2)

**Alternative 2: Mixture Models (Designer 2)**
- Captures 3-cluster structure from EDA
- Discrete subpopulations
- No need for covariates
- **When:** Regression + convergence issues

**Alternative 3: Beta-Binomial Overdispersion**
- Alternative to hierarchical model
- Captures overdispersion differently
- Conjugate (faster)
- **When:** Binomial likelihood insufficient

**Alternative 4: Gaussian Process Regression**
- Flexible non-parametric function
- Can capture complex patterns
- Computationally intensive
- **When:** Quadratic insufficient, need flexibility

**Alternative 5: Robust Models (Student-t)**
- Downweight outliers
- Handle Groups 4 and 8 differently
- More robust to misspecification
- **When:** Outliers dominate inference

---

## Resource Requirements

### Computational

- **Hardware:** Standard laptop (4+ cores, 8GB RAM)
- **Software:** Stan (via CmdStanPy), Python 3.8+, ArviZ
- **Time:**
  - Model 1-2: ~30 seconds each
  - Model 3: ~5 minutes
  - Total: ~2 hours (with diagnostics and plotting)

### Personnel

- **Analyst:** 1 person, 3 days
- **Domain Expert:** Consultation for interpretation (1-2 hours)
- **Reviewer:** QA on diagnostics and code (2 hours)

### Data

- **Required:** group_id, n_trials, r_successes (already available)
- **Nice to have:** Additional covariates (study design, context)
- **Sample size:** J=12 is minimal for hierarchical models

---

## Deliverables

### Primary Outputs

1. **Technical Report:**
   - Model specifications
   - Diagnostics (Rhat, ESS, divergences)
   - LOO comparison table
   - Coefficient estimates with uncertainty

2. **Visualizations:**
   - Observed vs. predicted (all models)
   - Coefficient posteriors (forest plots)
   - Posterior predictive checks
   - LOO comparison (with SE bars)

3. **Code Archive:**
   - Stan model files (.stan)
   - Fitting script (fit_models.py)
   - Data preprocessing
   - Reproducible workflow

4. **Interpretation Document:**
   - Scientific implications
   - Limitations and caveats
   - Recommendations for future work
   - Integration with other designers

### Communication Products

**For Technical Audience:**
- Full report with math and diagnostics
- Stan code and posterior samples
- Sensitivity analysis results

**For Non-Technical Stakeholders:**
- Executive summary (1 page)
- Key visualizations (annotated)
- Plain-language interpretation
- Decision-relevant conclusions

---

## Integration with Other Designers

### Complementarity

**Designer 1 (Random Effects):**
- My baseline comparison
- If I fail (ΔLOO < 2), their approach wins
- Complementary, not competing

**Designer 2 (Mixture/Robust):**
- If Model 2 succeeds, aligns with clusters
- If Model 3 succeeds, suggests mixture components
- Could combine: mixture + covariates

### Synthesis Strategy

**After all designers complete:**

1. **Compare LOO across ALL models** (mine + theirs)
2. **Identify consensus vs. divergence**
3. **Report best model(s)** with model uncertainty
4. **Acknowledge limitations** of small J
5. **Recommend future directions**

**If my models lose:**
→ Document gracefully
→ Support winner
→ Explain what was learned

**If my models win:**
→ Report with humility
→ Acknowledge alternatives
→ Quantify model uncertainty

---

## Timeline

**Day 1 (Morning):**
- Setup and data validation (30 min)
- Compile models (30 min)
- Prior predictive checks (1 hour)

**Day 1 (Afternoon):**
- Fit Model 1 (30 min including diagnostics)
- Fit Model 2 (30 min including diagnostics)
- Fit baseline (20 min)
- Decision: Proceed to Model 3? (1 hour)

**Day 2 (Morning):**
- Fit Model 3 if justified (30 min)
- Compute LOO for all models (30 min)
- Model comparison (1 hour)

**Day 2 (Afternoon):**
- Posterior predictive checks (1 hour)
- Sensitivity analysis (1 hour)
- Generate plots (1 hour)

**Day 3:**
- Interpretation and writing (4 hours)
- QA and review (2 hours)
- Finalize deliverables (2 hours)

**Total:** 3 days, ~20 hours of work

---

## Success Metrics

**Experiment is successful if:**

1. **Technical Success:**
   - [ ] All models converge (Rhat < 1.01)
   - [ ] Clear LOO comparison (no ties)
   - [ ] Pass posterior predictive checks
   - [ ] Reproducible workflow

2. **Scientific Success:**
   - [ ] Clear answer to research questions
   - [ ] Falsification criteria applied
   - [ ] Interpretation is plausible
   - [ ] Limitations acknowledged

3. **Communication Success:**
   - [ ] Results understandable
   - [ ] Uncertainty quantified
   - [ ] Decision-relevant
   - [ ] Integration with other designers

**Red Line for Failure:**
- Cannot distinguish models (all ΔLOO = 2-4, SE > ΔLOO)
- Results unstable across specifications
- Cannot falsify or support any hypothesis
- Contradicts EDA without explanation

---

## Conclusion

This experiment plan tests three competing hypotheses about covariate effects in binomial data with strong heterogeneity. Each model has explicit falsification criteria and decision points for pivoting. The approach is designed to:

1. **Find truth, not complete tasks** - Ready to accept H0 if covariates don't help
2. **Plan for failure** - Multiple pivot points and alternative approaches
3. **Think adversarially** - Red flags and stopping rules explicitly defined
4. **Embrace uncertainty** - Model comparison, not forcing a winner

**Expected Outcome:** Most likely (35%) is that covariates are uninformative (H0), but formal testing is essential to quantify evidence and guide future work.

**Commitment to Falsification:** I will abandon each model according to stated criteria and pivot to alternatives if the entire regression approach fails.

---

**Files:**
- Complete specifications: `/workspace/experiments/designer_3/proposed_models.md`
- Stan models: `/workspace/experiments/designer_3/model*.stan`
- Implementation: `/workspace/experiments/designer_3/fit_models.py`
- Quick start: `/workspace/experiments/designer_3/QUICKSTART.md`
- Usage guide: `/workspace/experiments/designer_3/README.md`
