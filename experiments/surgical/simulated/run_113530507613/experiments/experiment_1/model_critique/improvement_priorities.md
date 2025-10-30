# Improvement Priorities: Experiment 1
## Hierarchical Logit-Normal Model

**Date:** 2025-10-30
**Status:** Model ACCEPTED - This document outlines monitoring recommendations and potential extensions

---

## Purpose

While the hierarchical logit-normal model is **accepted** for scientific inference, this document provides:
1. **Monitoring recommendations** - what to watch when using this model
2. **Alternative comparisons** - what models should be compared against
3. **Potential extensions** - valuable improvements if time/resources permit
4. **Known limitations** - what users should be aware of

This is NOT a list of required fixes (model is adequate as-is), but rather guidance for **responsible use** and **potential improvement**.

---

## Priority 1: Essential Model Comparisons (REQUIRED)

### 1.1 Compare to Mixture Model (Experiment 2)

**Motivation:**
- EDA identified K=3 distinct clusters (low, very low, high success rate groups)
- Current continuous hierarchy assumes smooth variation
- 50% of groups have high Pareto k (> 0.7), suggesting discrete structure may be more efficient

**Comparison Metrics:**
- **LOO-CV:** ΔLOO and standard error
  - Expectation: Mixture may have lower Pareto k at cluster boundaries
  - Threshold: ΔLOO > 4 considered meaningful difference
- **Posterior Predictive Checks:** Both should pass, but check test statistic p-values
- **Interpretability:** Continuous hierarchy is simpler; mixture requires justification

**Decision Rule:**
- If mixture ΔLOO > 4: Prefer mixture (better fit, captures structure)
- If mixture ΔLOO < -4: Prefer continuous (simpler is better)
- If |ΔLOO| < 4: Consider model averaging or prefer simpler continuous hierarchy

**Implementation:**
```python
# LOO comparison
loo_continuous = az.loo(idata_exp1)
loo_mixture = az.loo(idata_exp2)
comparison = az.compare({'continuous': idata_exp1, 'mixture': idata_exp2})
```

### 1.2 Compare to Robust Student-t Model (Experiment 3)

**Motivation:**
- Groups 4 and 8 identified as outliers in EDA
- Both have high Pareto k (0.830, 1.015)
- Heavy-tailed priors may reduce influence of outliers

**Comparison Metrics:**
- **LOO-CV:** Focus on Pareto k for Groups 4, 8
  - Expectation: Student-t may reduce k for outlier groups
- **Effective degrees of freedom:** If ν >> 30, Normal is sufficient
- **Computational cost:** Student-t typically requires more divergence tuning

**Decision Rule:**
- If Student-t ΔLOO > 4 AND reduces Pareto k: Prefer robust model
- If ν posterior > 100: Normal approximation sufficient, prefer continuous
- Consider computational cost (divergences, runtime)

### 1.3 Use K-Fold Cross-Validation

**Motivation:**
- LOO with high Pareto k has inflated standard errors
- K-fold CV more stable for model comparison
- Provides independent validation

**Recommendation:**
```python
# 5-fold or 10-fold CV
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Refit model on each fold, compute ELPD on held-out fold
# More computationally expensive but more reliable
```

**Interpretation:**
- K-fold CV should align with LOO in ranking models
- If rankings differ substantially, high Pareto k is affecting LOO reliability
- Use K-fold as tiebreaker if LOO is ambiguous

---

## Priority 2: Monitoring When Using This Model (IMPORTANT)

### 2.1 Track Influential Observations

**What to Monitor:**
- **Which groups drive conclusions?**
  - Groups with high Pareto k (2, 3, 4, 6, 8, 12) have outsized influence
  - Population mean (mu) should be robust to individual group removal
  - Group-specific estimates (theta) will be affected

**How to Monitor:**
```python
# Leave-one-out influence analysis
for i in range(12):
    data_loo_i = data.drop(index=i)
    # Refit model without group i
    # Compare mu posterior to full-data mu
    # Substantial change (> 0.2 logits) indicates high influence
```

**Interpretation:**
- If mu changes < 0.2 logits when removing any single group: Robust
- If mu changes > 0.5 logits: Specific group is driving conclusion (report this)
- Expected: Group 4 (n=810, 29% of data) may have moderate influence

### 2.2 Prior Sensitivity Analysis

**What to Test:**
- **tau prior:** Half-Normal(0, 0.5) vs Half-Normal(0, 1.0)
- **mu prior:** Normal(-2.6, 1.0) vs Normal(-2.6, 2.0)

**Expected Outcome:**
- mu posterior should be robust (large data influence)
- tau posterior may shift slightly (moderate data influence)
- If conclusions change substantively, report sensitivity

**Implementation:**
```python
# Refit with wider tau prior
with pm.Model():
    tau = pm.HalfNormal('tau', sigma=1.0)  # vs sigma=0.5
    # ... rest of model
    # Compare tau posteriors and derived quantities
```

**Decision Rule:**
- If 95% HDI for key quantities overlap: Conclusions robust
- If qualitative conclusions differ: Report both, acknowledge sensitivity
- Prioritize weakly informative priors that are data-dominated

### 2.3 Convergence Monitoring (Ongoing)

**What to Check:**
- **Divergences:** Should remain 0% (currently 0/8000)
  - If divergences appear with new data: Increase adapt_delta to 0.95
  - Persistent divergences: Consider reparameterization
- **R-hat:** Should remain < 1.01 (currently 1.00)
  - If R-hat > 1.01: Increase warmup or run longer chains
- **ESS:** Should remain > 400 (currently > 1000)
  - Low ESS for tau expected (variance parameters harder to sample)

**Red Flags:**
- Divergences + funnel in mu-tau pairs plot: Non-centered parameterization failing
- R-hat > 1.05: Chains exploring different modes (mixture model needed?)
- ESS < 100: Severe autocorrelation (increase thinning or use different sampler)

### 2.4 Posterior Predictive Monitoring

**What to Track:**
- **Group-level fit:** Should have 0-1 groups flagged at 5% level
- **Global statistics:** Test statistic p-values should remain in [0.05, 0.95]
- **New data:** When new groups added, recompute PPC

**Warning Signs:**
- Sudden increase in flagged groups: Model no longer appropriate
- Systematic residual patterns emerging: Missing covariate or structure
- Overdispersion increasing: Between-group variance growing beyond model capacity

---

## Priority 3: Potential Extensions (OPTIONAL)

### 3.1 Add Group-Level Covariates

**Motivation:**
- EDA found no sample-size effect, but other predictors may exist
- Could explain heterogeneity, reduce residual tau
- Improve predictions for new groups

**Possible Covariates:**
- Study quality (if groups = studies)
- Time period (if groups = temporal batches)
- Setting (if groups = locations)
- Population characteristics

**Model Extension:**
```stan
theta[j] = mu + beta * X[j] + tau * theta_raw[j]
```

**Expected Impact:**
- Reduce tau (less unexplained variance)
- Improve predictive accuracy
- Better interpretability (explain WHY groups differ)

**Decision to Implement:**
- Only if covariates available and scientifically motivated
- Test whether adding X improves LOO-CV
- Beware overfitting with small J=12

### 3.2 Implement Model Averaging

**Motivation:**
- If ΔLOO between models is small (< 4)
- Incorporates model uncertainty into predictions
- More robust than selecting single "best" model

**Implementation:**
```python
# Bayesian model averaging with LOO weights
comparison = az.compare({'continuous': idata1, 'mixture': idata2, 'robust': idata3})
weights = comparison['weight']  # LOO weights

# Weighted predictions
y_pred_avg = sum(w * y_pred[model] for model, w in weights.items())
```

**When to Use:**
- Models have similar LOO (ΔLOO < 4)
- No clear theoretical preference
- Want robust predictions across model uncertainty

**When NOT to Use:**
- Clear winner by LOO (ΔLOO > 6)
- Models make contradictory scientific predictions
- Interpretation becomes too complex

### 3.3 Add Temporal/Sequential Structure

**Motivation:**
- If groups represent time periods, may have autocorrelation
- EDA found no trend, but could emerge with more data

**Model Extension:**
```stan
theta[j] = mu + rho * theta[j-1] + tau * theta_raw[j]  // AR(1)
```

**Current Evidence:**
- EDA found NO sequential dependence (p = 0.23)
- Linear trend not significant (p = 0.69)
- Autocorrelation r = 0.10 (not significant)

**Decision:** NOT recommended based on current data

### 3.4 Zero-Inflated Extension

**Motivation:**
- If success rate = 0 has special meaning (structural zeros)
- Could improve fit if some groups truly have zero baseline rate

**Model Extension:**
```stan
theta[j] ~ Bernoulli(pi) * 0 + (1 - pi) * LogitNormal(mu, tau)
```

**Current Evidence:**
- All 12 groups have r > 0 (no observed zeros)
- Lowest rate = 0.031 (Group 10), not a structural zero
- No evidence of zero-inflation

**Decision:** NOT recommended based on current data

---

## Priority 4: Known Limitations to Communicate (ESSENTIAL)

### 4.1 High Influential Observations

**Limitation:**
- 6/12 groups (50%) have Pareto k > 0.7
- Group 8 has k = 1.015 (very high influence)
- Posterior is sensitive to specific group inclusion/exclusion

**Communication:**
> "The hierarchical model provides reliable population-level inference (mu) and group-specific estimates (theta) that pass all posterior predictive checks. However, LOO cross-validation diagnostics indicate that several groups have high influence on the posterior distribution (Pareto k > 0.7 for 6 of 12 groups). While this does not invalidate the model's predictive performance (confirmed by comprehensive posterior predictive checks), it suggests that alternative formulations (e.g., mixture models, robust priors) may capture the data structure more parsimoniously. Conclusions are most robust for population-level parameters; group-specific estimates should be interpreted with awareness of influential observations."

### 4.2 Continuous Hierarchy Assumption

**Limitation:**
- Model assumes all groups drawn from single Normal(mu, tau)
- EDA suggests possible K=3 discrete clusters
- Continuous hierarchy may be averaging over discrete structure

**Communication:**
> "The model assumes groups vary continuously around a population mean, following a logit-normal distribution. Exploratory analysis identified three potential clusters (low, medium, high success rate groups), which the continuous hierarchy captures through a single heterogeneity parameter (tau). While this provides a parsimonious and interpretable summary, discrete mixture models may better represent the data-generating process if true subpopulations exist. Model comparison (Experiment 2) will formally test this hypothesis."

### 4.3 Moderate Uncertainty in Between-Group SD

**Limitation:**
- tau posterior has wide credible interval: [0.175, 0.632]
- Coefficient of variation: 32%
- Limited precision with J=12 groups

**Communication:**
> "Between-group heterogeneity (tau) is estimated with moderate uncertainty, reflecting the limited number of groups (J=12). The posterior mean tau = 0.394 indicates moderate variability (ICC ≈ 0.40), but the 94% credible interval [0.175, 0.632] spans from low to high heterogeneity. Qualitative conclusions about the presence of heterogeneity are robust, but precise quantitative claims about tau magnitude should be made cautiously. This is an inherent limitation of hierarchical models with small J, not a deficiency of this particular implementation."

### 4.4 Extrapolation Caution

**Limitation:**
- Model validated for success rates in observed range [0.03, 0.14]
- Predictions outside this range increasingly uncertain
- No data to constrain tail behavior

**Communication:**
> "Predictions are most reliable for success rates within the observed data range [0.03, 0.14]. The hierarchical distribution provides a mechanism for predicting new groups, but uncertainty increases substantially for extreme values. Extrapolation to success rates < 0.01 or > 0.20 should be made cautiously, acknowledging increased model uncertainty in these regions."

### 4.5 No Covariate Explanation

**Limitation:**
- Model captures heterogeneity (tau) but doesn't explain WHY groups differ
- No group-level predictors included
- Limits ability to make predictions for specific group types

**Communication:**
> "The model quantifies between-group heterogeneity (tau = 0.394) but does not explain the source of this variation. Incorporating group-level covariates (e.g., study characteristics, temporal trends, population features) could reduce unexplained variance and improve predictions. However, no such covariates were available in the current analysis. Interpreting group differences requires domain knowledge about what the groups represent."

---

## Priority 5: Alternative Models to Explore (FUTURE WORK)

### 5.1 Beta-Binomial Parameterization (Experiment 4)

**Rationale:**
- Conjugate structure may be more natural for binomial data
- Directly models overdispersion parameter
- Faster computation (no logit transformation)

**Trade-offs:**
- Different parameterization (not directly comparable to logit-normal)
- Harder to incorporate covariates
- Less standard in meta-analysis literature

**Expected Outcome:**
- Similar fit to logit-normal (both capture heterogeneity)
- May have computational advantages
- Interpretation differences need careful communication

### 5.2 Multilevel Regression with Poststratification (MRP)

**Rationale:**
- If groups have different covariate distributions
- Can adjust for compositional differences
- Improves generalizability

**Requirements:**
- Individual-level data (not just group summaries)
- Relevant covariates measured
- Poststratification weights available

**Current Data:**
- Only group-level summaries available
- No individual-level covariates
- NOT feasible with current data

### 5.3 Gaussian Process Prior

**Rationale:**
- If groups have natural ordering (spatial, temporal)
- Smooth transitions between adjacent groups
- Flexible correlation structure

**Requirements:**
- Groups have meaningful ordering
- Correlation decreases with distance
- Enough groups to estimate correlation structure

**Current Data:**
- No evidence of ordering (EDA sequential tests not significant)
- Groups appear exchangeable
- NOT recommended for current data

### 5.4 Nonparametric Dirichlet Process

**Rationale:**
- Let data determine number of clusters
- More flexible than fixed K mixture
- Automatic clustering

**Trade-offs:**
- Computationally expensive
- Harder to interpret (number of clusters uncertain)
- Requires strong prior on concentration parameter

**Decision:**
- Interesting exploration but not priority
- Finite mixture (K=3) more interpretable
- Consider if mixture model selection is ambiguous

---

## Recommended Action Plan

### Immediate Next Steps (Required):
1. **Fit Experiment 2 (Mixture Model):** Test K=3 discrete clusters hypothesis
2. **Fit Experiment 3 (Robust Student-t):** Test heavy-tailed alternative for outliers
3. **Compute LOO comparison:** Formal model comparison via ΔLOO
4. **Run K-fold CV:** Validate LOO rankings with more stable metric

### Short-Term Monitoring (Within Analysis):
1. **Leave-one-out influence:** Verify mu robust to individual group removal
2. **Prior sensitivity:** Test tau prior variation (Half-Normal(0, 1.0))
3. **Document limitations:** Include in final report (use templates above)

### Long-Term Extensions (If Time/Resources Permit):
1. **Model averaging:** If ΔLOO < 4 between top models
2. **Group-level covariates:** If relevant predictors become available
3. **External validation:** Test on independent dataset if available

### What NOT to Do:
1. **Don't revise current model:** No clear fixes for LOO concerns without alternative comparison
2. **Don't add complexity without justification:** J=12 limits identifiability of complex models
3. **Don't ignore limitations:** Transparent reporting is essential for scientific credibility

---

## Success Metrics

### This Model is Successful If:
- Population-level conclusions (mu) are robust to model choice (ΔLOO < 2 for mu)
- Group-specific predictions have appropriate coverage (calibrated intervals)
- Scientific interpretation remains consistent across alternative models
- Limitations are transparently communicated to users

### Alternative Model is Preferred If:
- ΔLOO > 4 in favor of alternative
- Alternative reduces Pareto k < 0.7 for most groups
- Alternative provides comparable interpretability
- Alternative maintains computational stability

### Model Averaging is Appropriate If:
- Multiple models have ΔLOO < 4
- Models make consistent scientific predictions
- Increased complexity of interpretation is acceptable

---

## Final Recommendations

**For Scientific Inference:**
- Use this model with documented limitations
- Report wide uncertainty in tau
- Acknowledge influential observations
- Compare to at least one alternative

**For Prediction:**
- Use posterior predictive distribution for existing groups
- Use hierarchical distribution for new groups
- Consider model averaging if alternatives are close
- Validate predictions on external data if available

**For Publication:**
- Present this as baseline model
- Compare to mixture and robust alternatives
- Use LOO and K-fold CV for comparison
- Discuss trade-offs (simplicity vs capturing structure)
- Include sensitivity analyses

**For Decision-Making:**
- Population mean (mu) is most robust parameter
- Group-specific estimates should acknowledge shrinkage
- Uncertainty intervals are trustworthy (slightly conservative)
- High-influence groups (4, 8) should be noted in context

---

**Document Purpose:** Guide responsible use of accepted model and plan for improvement

**Key Message:** Model is adequate as-is, but comparison to alternatives will strengthen inference

**Next Critical Step:** Proceed to Experiment 2 (Mixture Model) comparison
