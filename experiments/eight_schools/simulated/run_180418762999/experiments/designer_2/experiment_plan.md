# Experiment Plan: Hierarchical Measurement Error Dataset
## Designer 2 - Adaptive Bayesian Strategy

**Date**: 2025-10-28
**Approach**: Competing hypotheses with explicit falsification criteria

---

## Problem Formulation

### Competing Hypotheses

**Hypothesis 1** (EDA conclusion): Groups are homogeneous, tau = 0
- **Supporting evidence**: Chi-square p=0.42, between-group variance = 0
- **Challenge**: With n=8 and SNR~1, we lack power to detect moderate heterogeneity
- **Model**: Complete pooling (Model A)

**Hypothesis 2** (Alternative): Groups have modest heterogeneity that EDA missed
- **Supporting evidence**: Scientific plausibility (different groups usually differ)
- **Challenge**: Small sample size may lead to overfitting
- **Model**: Weakly regularized hierarchical (Model B)

**Hypothesis 3** (Skeptical): Reported measurement errors underestimate true uncertainty
- **Supporting evidence**: Observed variance < expected measurement variance
- **Challenge**: May be unfalsifiable with current data
- **Model**: Inflated measurement error (Model C)

### Scientific Question

**Primary**: Can we detect group-level heterogeneity with n=8 observations and large measurement error?

**Secondary**: Are the reported sigma values adequate, or do they underestimate true uncertainty?

**Meta-question**: Is this dataset informative enough to distinguish these hypotheses?

---

## Model Classes to Explore

### Model A: Complete Pooling Baseline

**Specification**:
```
y_i ~ Normal(mu, sigma_i)  [known sigma_i]
mu ~ Normal(0, 30)          [skeptical prior, wider than EDA suggestion]
```

**Why explore this**:
- EDA strongly recommends it
- Simplest model consistent with data
- Establishes baseline for comparison

**I will abandon this if**:
- LOO Pareto k > 0.7 for any observation (influential outlier)
- Posterior predictive checks fail for variance (underfitting)
- Extreme parameter values (|mu| > 30)

**Stress test**: Leave-one-out analysis - posterior for mu should be stable when dropping any single observation.

**Expected outcome**: Posterior for mu ~ N(8-12, 3-5) with good convergence.

---

### Model B: Hierarchical with Weak Regularization

**Specification**:
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
mu ~ Normal(0, 30)
tau ~ Normal_plus(0, 10)    [Half-normal, not half-Cauchy]
```

**Why explore this**:
- Tests whether ANY group structure can be detected
- More scientifically realistic (groups usually differ)
- Adaptive pooling based on data

**Design choice**: Half-normal instead of half-Cauchy
- Weakly regularizing (prevents extreme tau values)
- Tests sensitivity to standard recommendations
- More appropriate with small n

**I will abandon this if**:
- Divergent transitions > 5% (model fighting the data)
- Posterior for tau: 95% mass below 1 (no meaningful heterogeneity)
- LOO-CV worse than Model A by > 1 SE (complexity penalty not justified)
- R-hat > 1.05 for any parameter (convergence failure)

**Stress test**: Compare three priors on tau: Half-Normal(0,10), Half-Cauchy(0,5), Exponential(0.2). If posterior is strongly prior-dependent, insufficient data to estimate tau.

**Escape route**: If divergences occur, switch to non-centered parameterization:
```
theta_i = mu + tau * theta_raw_i
theta_raw_i ~ Normal(0, 1)
```

**Expected outcome**: Either (1) tau near boundary with poor sampling, or (2) moderate tau (3-8) if heterogeneity exists.

---

### Model C: Measurement Error Misspecification

**Specification**:
```
y_i ~ Normal(mu, sqrt(sigma_i^2 + tau_meas^2))
mu ~ Normal(0, 30)
tau_meas ~ Normal_plus(0, 5)
```

**Why explore this**:
- Challenges assumption that sigma_i are known exactly
- Explains discrepancy: observed var < expected measurement var
- Laboratories often underestimate measurement uncertainty

**I will abandon this if**:
- Posterior for tau_meas: 95% mass below 2 (reported errors adequate)
- Correlation between mu and tau_meas > 0.7 (identifiability issue)
- LOO-CV worse than Model A by > 1 SE (not worth the complexity)
- Posterior SD for mu increases > 2x vs Model A (losing all precision)

**Stress test**: Posterior predictive check for variance. If predicted variance >> observed variance, tau_meas is too large (overfitting).

**Expected outcome**: Either (1) tau_meas near zero (reported errors OK), or (2) substantial tau_meas (5-10) indicating error underestimation.

---

## Red Flags for Model Class Changes

### Trigger 1: All Models Fail Posterior Predictive Checks
**What it means**: Fundamental misspecification (wrong likelihood family)

**Action**:
1. Try t-distributed errors instead of normal
2. Consider mixture model (two latent groups)
3. Check for data errors or outliers missed by EDA

**Evidence needed**: Systematic deviation in multiple aspects (variance, extremes, symmetry)

---

### Trigger 2: Strong Prior-Posterior Conflict
**What it means**: Data are weaker than anticipated, or model severely misspecified

**Diagnostic**:
- Prior predictive and posterior predictive distributions barely overlap
- Likelihood and prior "fighting" each other

**Action**:
1. Increase prior width (more vague)
2. Check data quality (outliers, errors)
3. May need informative priors from domain knowledge

**Evidence needed**: Prior predictive p-value < 0.01 for multiple statistics

---

### Trigger 3: Systematic Computational Failures
**What it means**: Data incompatible with model structure

**Examples**:
- Model A won't converge (should be trivial)
- All variants of Model B have divergences
- Extreme correlations between parameters

**Action**:
1. Revisit data quality (are sigma_i plausible?)
2. Try different parameterizations
3. Consider non-parametric approach
4. May need to report "data insufficient for parametric modeling"

---

### Trigger 4: Extreme Parameter Values
**What it means**: Model extrapolating beyond plausible range

**Thresholds**:
- mu > 50 or mu < -30 (implausible given observed data)
- tau > 30 (unrealistic heterogeneity)
- tau_meas > 20 (all variation attributed to measurement error)

**Action**: Check for data errors, outliers, or need for bounded priors

---

## Decision Points

### Decision Point 1: After Model A
**Timeline**: After initial fit (~2 minutes)

**Check**:
1. Convergence diagnostics (R-hat, ESS, trace plots)
2. Posterior predictive checks (variance, min/max)
3. LOO-CV Pareto k values

**Branches**:
- **All good**: Proceed to Model B for comparison
- **High Pareto k**: Influential observation, Model B may help
- **Poor predictive fit**: Try Model C (inflated errors)
- **Convergence issues**: Red flag - even simple model failing

---

### Decision Point 2: After Model B
**Timeline**: After hierarchical fit (~10 minutes)

**Key comparison**: LOO-CV difference between Model B and Model A

**Scenarios**:

| Outcome | Decision |
|---------|----------|
| Model B wins by > 2 SE AND tau > 2 | Use hierarchical, report group effects |
| Model B wins but tau < 2 | Improvement from flexibility, not real heterogeneity |
| Model A wins or tie (within 2 SE) | Use complete pooling (parsimony) |
| Model B has divergences > 5% | Try non-centered; if still fails, abandon |

---

### Decision Point 3: After Model C
**Timeline**: After measurement error model (~10 minutes)

**Key questions**:
1. Is tau_meas posterior substantially > 0?
2. Does Model C improve LOO-CV?
3. Are mu and tau_meas identifiable?

**Decision logic**:
- If tau_meas > 5 AND LOO-CV better: Measurement errors underestimated
- If tau_meas < 2 OR corr(mu, tau_meas) > 0.7: Reported errors adequate
- If identifiability issues: Report as fundamental limitation

---

### Final Decision: Model Selection

**Criteria** (in order of importance):

1. **Falsification**: Any model that triggers its abandonment criteria is eliminated
2. **Predictive performance**: LOO-CV with > 2 SE difference
3. **Computational quality**: No divergences, good R-hat, adequate ESS
4. **Parsimony**: If tied, choose simpler model

**Possible outcomes**:

| Scenario | Interpretation | Report |
|----------|---------------|---------|
| Model A wins | EDA conclusion confirmed | Use complete pooling |
| Model B wins | Hidden heterogeneity detected | Report group-specific estimates with uncertainty |
| Model C wins | Measurement errors misspecified | Wider credible intervals, re-estimate uncertainties |
| No clear winner | Data insufficient | Report model uncertainty, recommend more data |
| All models fail | Fundamental misspecification | Pivot to alternative approaches |

---

## Alternative Approaches (Escape Routes)

### If Complete Pooling and Hierarchical Both Fail

**Option 1: Robust Modeling**
```
y_i ~ Student_t(nu, mu, sigma_i)
nu ~ Gamma(2, 0.1)
```
- Use if outliers detected despite EDA
- More forgiving to extreme values

**Option 2: Mixture Model**
```
y_i ~ p * Normal(mu_1, sigma_i) + (1-p) * Normal(mu_2, sigma_i)
```
- Use if Group 4 (negative value) consistently influential
- Requires n > 15 for reliable estimation (may not be feasible)

**Option 3: Non-parametric**
- Bootstrap-based inference
- Permutation tests for group differences
- Use if parametric models all fail

---

## Model Comparison Approach

### Primary: LOO-CV (Leave-One-Out Cross-Validation)

**Why LOO-CV**:
- More robust than WAIC with small n
- Provides Pareto k diagnostic (model checking)
- Natural for small datasets (n=8)

**Interpretation**:
- Compare elpd_loo (expected log pointwise predictive density)
- Difference > 2 SE is "meaningful"
- Pareto k > 0.7 indicates influential observation (model inadequacy)

**Decision rule**:
```
if elpd_diff > 2 * SE(elpd_diff):
    prefer model with higher elpd
else:
    prefer simpler model (parsimony)
```

---

### Secondary: Posterior Predictive Checks

**Variance test**:
```
For each posterior sample:
  Simulate y_rep ~ model
  Compute var(y_rep)
Compare var(y_obs) to distribution of var(y_rep)
```
**Success**: Observed variance in 5-95% predictive interval

**Extremes test**:
```
For each posterior sample:
  Compute min(y_rep) and max(y_rep)
Check if observed min/max are plausible
```
**Success**: Observed extremes not in tail (< 5% or > 95%)

---

### Tertiary: Prior Sensitivity

**For Model A**:
- Refit with N(10, 20) vs my N(0, 30)
- Compare posteriors for mu
- If posteriors differ by > 20%, data are weak

**For Model B**:
- Refit with Half-Cauchy(0, 5) vs Half-Normal(0, 10)
- Compare posteriors for tau
- If posteriors differ by > 50%, cannot estimate tau reliably

**Stopping rule**: If prior sensitivity is high AND no model clearly wins, report "data insufficient for confident inference."

---

## Computational Strategy

### Software: Stan (primary)

**Rationale**:
- Best HMC/NUTS implementation
- Superior divergence diagnostics
- Handles boundary problems (tau near 0) well

**Configuration**:
```
chains = 4
iterations = 2000 (1000 warmup + 1000 sampling)
adapt_delta = 0.95 (start conservatively)
max_treedepth = 12
```

**Adjustment protocol**:
- If divergences > 1%: increase adapt_delta to 0.99
- If ESS < 100: double iterations
- If still issues: reparameterize (non-centered for Model B)

---

### Computational Budget

| Model | Expected Time | Difficulty |
|-------|--------------|------------|
| Model A | 1-2 min | Easy (single parameter) |
| Model B | 5-10 min | Hard (boundary, potential funnel) |
| Model C | 5-10 min | Moderate (correlation between params) |
| **Total** | ~20 min | Plus diagnostics time |

**Parallelization**: 4 chains in parallel per model

---

## Success Criteria

### Model-Specific

**Model A succeeds if**:
- R-hat < 1.01 for mu
- ESS > 1000 for mu
- Posterior predictive checks pass
- No Pareto k > 0.7

**Model B succeeds if**:
- Divergences < 2%
- R-hat < 1.05 for all parameters
- ESS > 100 for tau (harder to estimate)
- If tau > 2: meaningful heterogeneity detected

**Model C succeeds if**:
- Correlation(mu, tau_meas) < 0.7
- Posterior not at boundary (tau_meas > 1)
- Predicted variance matches observed

---

### Overall Success

**Primary goal**: Find at least one model that:
1. Passes all diagnostic checks
2. Passes posterior predictive checks
3. Makes scientific sense

**Secondary goal**: Distinguish between competing hypotheses using LOO-CV

**Acceptable outcome**: "Cannot distinguish between complete pooling and modest heterogeneity with n=8" - this is an honest finding.

**Unacceptable outcome**: Forcing a conclusion when models disagree or all fail diagnostics.

---

## Stopping Rules

### When to Stop and Report Success

**Condition 1**: Model A succeeds and is best by LOO-CV
- Report: "Complete pooling confirmed, EDA conclusion validated"
- Deliver: Posterior for mu, credible intervals, predictive distribution

**Condition 2**: Model B succeeds and is best by LOO-CV with tau > 3
- Report: "Modest heterogeneity detected despite small sample"
- Deliver: Group-specific posteriors with shrinkage estimates

**Condition 3**: Model C succeeds and is best by LOO-CV
- Report: "Measurement errors appear underestimated"
- Deliver: Adjusted uncertainty estimates, wider credible intervals

---

### When to Pivot Strategy

**Condition 1**: All three models fail posterior predictive checks
- Action: Try robust alternatives (t-distribution, mixture model)
- Timeline: Additional 1-2 hours

**Condition 2**: Strong prior-posterior conflict across all models
- Action: Investigate data quality, consider informative priors
- Timeline: Additional 30 minutes diagnostics

**Condition 3**: No clear winner after trying all alternatives
- Action: Report model uncertainty as finding
- Recommendation: Collect more data (need n > 20)

---

### When to Exhaustively Explore

**Situations requiring deeper investigation**:

1. **Model B shows divergences**: Try non-centered parameterization before abandoning
2. **Model C has identifiability issues**: Try tighter prior on tau_meas
3. **All models similar LOO-CV**: Run stacking weights to see if ensemble helps
4. **Suspicious patterns in residuals**: Try additional diagnostic plots

**Time budget**: Allow 1 hour for deep-dive diagnostics if initial fits are ambiguous

---

## Warning Signs (What Would Make Me Reconsider Everything)

### Critical Warning Signs

**Sign 1: Model A (simplest) won't converge**
- Implication: Data quality issue or fundamental misspecification
- Action: Inspect data values, check for typos or implausible values

**Sign 2: Posterior for mu has heavy mass on negative values**
- Implication: Conflicts with EDA finding (mean > 0)
- Action: Check prior specification, investigate Group 4 (only negative obs)

**Sign 3: All models predict much wider variance than observed**
- Implication: Model assumes too much noise
- Action: Reconsider measurement error model, check if sigma_i are overestimates

**Sign 4: LOO-CV identifies multiple influential observations (Pareto k > 0.7)**
- Implication: Model inadequate for this data structure
- Action: Try robust or mixture models

---

### Moderate Warning Signs (Proceed with Caution)

**Sign 1: Prior sensitivity > 50% for key parameters**
- Interpretation: Data are weak, conclusions tentative
- Action: Report full prior sensitivity analysis, emphasize uncertainty

**Sign 2: Model B posterior for tau concentrated at boundary (< 0.5)**
- Interpretation: Complete pooling is correct, hierarchical adds nothing
- Action: Use Model A, but report that hierarchical was tried

**Sign 3: Wide credible intervals (e.g., mu: [-10, 30])**
- Interpretation: Insufficient data for precise inference
- Action: Expected given n=8 and SNR~1, report honestly

---

## Documentation and Reproducibility

### Code Structure

All Stan models saved as:
- `/workspace/experiments/designer_2/model_a_complete_pooling.stan`
- `/workspace/experiments/designer_2/model_b_hierarchical.stan`
- `/workspace/experiments/designer_2/model_b_noncentered.stan` (if needed)
- `/workspace/experiments/designer_2/model_c_inflated_errors.stan`

Fitting scripts:
- `/workspace/experiments/designer_2/fit_all_models.py` (or R script)
- `/workspace/experiments/designer_2/model_comparison.py`
- `/workspace/experiments/designer_2/diagnostics.py`

---

### Outputs to Generate

**Numerical summaries**:
- `model_comparison_loo.csv` (LOO-CV results)
- `posterior_summaries.csv` (means, SDs, quantiles)
- `diagnostic_summary.txt` (R-hat, ESS, divergences)

**Visualizations**:
- `posterior_distributions.png` (all parameters)
- `trace_plots.png` (MCMC diagnostics)
- `posterior_predictive_checks.png` (model adequacy)
- `model_comparison.png` (LOO-CV differences)
- `prior_sensitivity.png` (robustness checks)

**Written report**:
- `diagnostic_report.md` (computational diagnostics)
- `model_selection_report.md` (comparison and choice)
- `final_results.md` (conclusions and recommendations)

---

## Timeline and Milestones

### Phase 1: Initial Fits (30 minutes)
- Fit Model A (5 min: code + run)
- Check diagnostics, posterior predictive checks (10 min)
- Fit Model B (5 min: code + run)
- Check diagnostics, LOO-CV comparison (10 min)

**Milestone 1 decision**: Is Model B worth exploring further, or stick with Model A?

---

### Phase 2: Extended Exploration (30 minutes)
- Fit Model C (5 min: code + run)
- Three-way model comparison with LOO-CV (10 min)
- Posterior predictive checks for all models (10 min)
- Prior sensitivity analysis (5 min)

**Milestone 2 decision**: Is there a clear winner, or need deeper investigation?

---

### Phase 3: Robustness Checks (30 minutes, if needed)
- Leave-one-out analysis at model level (15 min)
- Alternative priors for key parameters (10 min)
- Stress tests (posterior predictive variance, extremes) (5 min)

**Milestone 3 decision**: Final model selection or declare insufficient data

---

### Phase 4: Reporting (30 minutes)
- Generate all plots and tables (15 min)
- Write diagnostic and selection reports (15 min)
- Document code and results (10 min)

**Total time**: 2-2.5 hours (plus writing time)

---

## Meta-Level Reflection

### On the EDA Conclusion

The EDA strongly advocates complete pooling based on:
- Between-group variance = 0
- Homogeneity test p = 0.42
- Extensive group overlap

**My skepticism**: With n=8 and large measurement error, we have very low power to detect moderate heterogeneity (tau ~ 5-10). A tau of 8 could easily look like tau = 0 with this data.

**My test**: Fit hierarchical model and see if we can detect ANY structure. If tau posterior is entirely at boundary with no computational issues, then EDA is likely correct. If tau > 0 with good support, EDA missed something.

---

### On Model Complexity

Standard advice: "Start simple, add complexity as needed"

**My approach**: Start with three competing hypotheses:
1. Simplest (Model A): Groups identical
2. Moderate (Model B): Groups differ modestly
3. Alternative (Model C): Errors misspecified

This is NOT about complexity for its sake - it's about testing different data generation stories.

---

### On Falsification

Each model has explicit abandonment criteria. This is deliberate:
- Prevents confirmation bias (cherry-picking supportive evidence)
- Forces honesty about model failures
- Makes research process transparent

**Expectation**: At least one (possibly two) models will be abandoned. That's success - we learned what the data reject.

---

### On Reporting Uncertainty

If no model clearly wins, I will report:
1. "Data are insufficient to distinguish complete pooling from modest heterogeneity"
2. Show all three posteriors with their differences
3. Recommend n > 20 for reliable tau estimation

This is more valuable than forcing a conclusion that the data don't support.

---

## Final Checklist

Before reporting results, verify:

**Computational**:
- [ ] All models: R-hat < 1.05 for all parameters
- [ ] All models: ESS > 100 (> 400 for main parameters)
- [ ] Model B: Divergences < 5% (or successfully used non-centered)
- [ ] Trace plots reviewed and show good mixing

**Model Adequacy**:
- [ ] Posterior predictive checks pass for chosen model(s)
- [ ] LOO-CV: All Pareto k < 0.7 (or addressed if higher)
- [ ] Prior sensitivity checked (< 30% change with reasonable alternatives)
- [ ] No extreme parameter values outside plausible range

**Comparison**:
- [ ] LOO-CV computed for all models
- [ ] Differences interpreted with standard errors
- [ ] Simplest adequate model chosen (parsimony)

**Documentation**:
- [ ] All code saved and commented
- [ ] Diagnostic plots generated
- [ ] Numerical summaries exported
- [ ] Written reports complete

---

## Conclusion

This experiment plan is deliberately cautious and self-critical. With n=8 and high measurement error, we should expect:
- High model uncertainty
- Wide credible intervals
- Possibly inconclusive comparison

**Success is not confirming a preferred model** - success is finding the model(s) most consistent with the data while honestly reporting limitations.

If the data truly support complete pooling (Model A), the experiment will show:
1. Model A passes all checks
2. Model B finds tau ≈ 0 with no improvement
3. Model C finds tau_meas ≈ 0 with no improvement

If there's hidden heterogeneity, we'll detect it in Model B posteriors and LOO-CV.

If measurement errors are misspecified, Model C will reveal it.

**The plan adapts based on what we find** - that's the point of explicit decision points and escape routes.

---

**Files**:
- Model specifications: `/workspace/experiments/designer_2/proposed_models.md`
- This experiment plan: `/workspace/experiments/designer_2/experiment_plan.md`
- Stan models: To be written in implementation phase
- Results: To be generated during fitting

---

**End of Experiment Plan**
