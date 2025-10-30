# Bayesian Modeling Experiment Plan

**Date:** Plan Created After Parallel Model Design Phase
**Status:** Ready for Implementation

---

## Executive Summary

Three independent model designers proposed 9 distinct Bayesian models to analyze binomial data with strong heterogeneity (12 groups, ICC = 0.42). This document synthesizes their proposals into a prioritized experiment plan with clear falsification criteria and minimum attempt requirements.

**Minimum Attempt Policy:** We will attempt at least the first **two models** (Experiments 1 and 2) unless Experiment 1 fails pre-fit validation. This ensures adequate exploration before proceeding to assessment.

---

## Model Selection Rationale

### Criteria for Prioritization

1. **Theoretical justification** (alignment with EDA findings)
2. **Computational feasibility** (expected runtime < 5 min)
3. **Falsifiability** (clear rejection criteria)
4. **Baseline necessity** (standard approach first)
5. **Complementary strengths** (different model classes)

### EDA Findings Driving Model Choice

- ✓ Strong heterogeneity (ICC = 0.42, χ² p < 0.001)
- ✓ Overdispersion (variance ratio = 2.78)
- ✓ Three clusters identified (low/very low/high rates)
- ✓ Two extreme outliers (Groups 4, 8)
- ✓ No sample size effect (r = -0.34, p = 0.278)
- ✓ No sequential dependence (p > 0.23)

---

## Prioritized Experiment Queue

### **Experiment 1: Standard Hierarchical Logit-Normal** [REQUIRED]

**Source:** Designer 1, Model 1.1
**Priority:** HIGHEST (baseline, mandatory)
**Expected Runtime:** 30-60 seconds

**Why This Model First:**
- Standard approach in literature (defensible choice)
- Directly addresses heterogeneity (ICC = 0.42)
- Non-centered parameterization avoids funnel geometry
- Provides baseline for all comparisons (LOO-CV reference)

**Mathematical Specification:**
```
Likelihood:  r[j] ~ Binomial(n[j], inv_logit(theta[j]))
Group effects: theta[j] = mu + tau * theta_raw[j]  [non-centered]
Prior on raw: theta_raw[j] ~ Normal(0, 1)
Hyperprior mu: mu ~ Normal(-2.6, 1.0)
Hyperprior tau: tau ~ Normal(0, 0.5)  [half-normal via constraint]
```

**Falsification Criteria:**
- REJECT if: Rhat > 1.01, ESS < 400, divergences > 1%
- REJECT if: Posterior predictive p-value < 0.05
- REJECT if: Shrinkage estimates unreasonable (all → complete pooling)
- REJECT if: Prior-data conflict (prior constrains posteriors inappropriately)

**Expected Outcome:** ACCEPT (high confidence ~70%)
- Should converge well with non-centered parameterization
- Partial pooling should stabilize small-sample estimates
- May underfit if cluster structure is strong

**If REJECTED:** Pivot to robust alternatives (Experiment 3 or 5)

---

### **Experiment 2: Finite Mixture Model (K=3)** [REQUIRED]

**Source:** Designer 2, Model 2.1
**Priority:** HIGH (tests cluster hypothesis)
**Expected Runtime:** 20-40 seconds

**Why This Model Second:**
- EDA identified 3 distinct clusters (6 methods convergent)
- Tests whether discrete heterogeneity better than continuous
- Complementary to Experiment 1 (different model class)
- Faster than hierarchical model (simpler structure)

**Mathematical Specification:**
```
Cluster assignment: z[j] ~ Categorical(pi)  [pi = mixture weights]
Cluster means: mu_k ~ Normal(-2.6, 1.0)  [k = 1,2,3, ordered]
Cluster precision: tau_k ~ Gamma(2, 0.5)
Likelihood: r[j] ~ Binomial(n[j], inv_logit(theta[j]))
            theta[j] ~ Normal(mu[z[j]], 1/sqrt(tau[z[j]]))
```

**Falsification Criteria:**
- REJECT if: Cluster assignments ambiguous (mean probability < 0.6)
- REJECT if: Clusters collapse (K_effective < 2)
- REJECT if: Cluster separation < 2σ (overlap too large)
- REJECT if: ΔLOO vs Experiment 1 < -10 (much worse fit)

**Expected Outcome:** ACCEPT or REVISE (moderate confidence ~50%)
- May reveal whether clusters are real or artifacts
- Could show similar LOO to Experiment 1 (model uncertainty)
- Cluster assignments should align with EDA K-means results

**If REJECTED:**
- If K_effective = 1 → Clusters are artifacts, use Experiment 1
- If assignments ambiguous → Continuous heterogeneity, use Experiment 1

---

### **Experiment 3: Robust Student-t Hierarchy** [CONDITIONAL]

**Source:** Designer 1, Model 1.2
**Priority:** MEDIUM (if outliers problematic)
**Expected Runtime:** 60-120 seconds

**Why This Model Third:**
- Addresses extreme outliers (Groups 4, 8 with |z| > 3)
- Tests whether heavy tails needed
- More robust than Experiment 1 if outliers influential

**Mathematical Specification:**
```
Likelihood:  r[j] ~ Binomial(n[j], inv_logit(theta[j]))
Group effects: theta[j] ~ StudentT(nu, mu, tau)  [nu = degrees of freedom]
Hyperprior nu: nu ~ Gamma(2, 0.1)  [weakly informative]
Hyperprior mu: mu ~ Normal(-2.6, 1.0)
Hyperprior tau: tau ~ Normal(0, 0.5)
```

**Falsification Criteria:**
- REJECT if: nu > 30 (tails effectively Normal → use Experiment 1)
- REJECT if: nu < 3 (too heavy-tailed, implausible)
- REJECT if: ΔLOO vs Experiment 1 < 2 (no improvement)

**Expected Outcome:** ACCEPT if Experiment 1 shows outlier sensitivity
- Posterior nu ∈ [4, 15] would indicate moderate heavy tails needed
- Should downweight Groups 4, 8 appropriately
- LOO may show modest improvement (ΔLOO = 2-5)

**If REJECTED:**
- If nu > 30 → Normal sufficient, use Experiment 1
- If no LOO improvement → Stick with Experiment 1

**Trigger Condition:** Only run if Experiment 1 shows:
- Outliers have high influence (Cook's D > 0.5)
- Poor posterior predictive fit for Groups 4, 8
- Standardized residuals > 3 for outliers

---

### **Experiment 4: Hierarchical Beta-Binomial** [OPTIONAL]

**Source:** Designer 1, Model 1.3
**Priority:** LOW (alternative parameterization)
**Expected Runtime:** 20-40 seconds

**Why This Model:**
- Conjugate structure (simpler inference)
- Direct overdispersion parameter (kappa)
- Comparison with logit-normal hierarchy

**Mathematical Specification:**
```
Likelihood: r[j] ~ BetaBinomial(n[j], alpha[j], beta[j])
            alpha[j] = p[j] * kappa
            beta[j] = (1 - p[j]) * kappa
Group means: p[j] ~ Beta(a_0, b_0)  [hierarchical]
Overdispersion: kappa ~ Gamma(2, 0.1)
Hyperpriors: a_0, b_0 ~ Gamma(1, 0.1)
```

**Falsification Criteria:**
- REJECT if: kappa > 1000 (no overdispersion needed)
- REJECT if: ΔLOO vs Experiment 1 < -5 (worse fit)
- REJECT if: Computationally problematic (divergences, non-convergence)

**Expected Outcome:** ACCEPT as alternative (low priority)
- May have similar LOO to Experiment 1 (model uncertainty)
- Simpler interpretation (kappa = overdispersion parameter)

**Trigger Condition:** Only run if:
- Experiments 1-2 both ACCEPT → Need comparison
- Time permits (< 30 min remaining in workflow)

---

### **Experiment 5: Dirichlet Process Mixture** [EXPLORATORY]

**Source:** Designer 2, Model 2.3
**Priority:** LOWEST (robustness check)
**Expected Runtime:** 40-60 seconds

**Why This Model:**
- Non-parametric (doesn't commit to K=3)
- Tests robustness of cluster structure
- Infers number of clusters from data

**Mathematical Specification:**
```
Concentration: alpha ~ Gamma(1, 1)  [DP concentration parameter]
Stick-breaking: pi = stick_breaking(alpha, K_max=8)
Cluster means: mu_k ~ Normal(-2.6, 1.0)
Cluster precision: tau_k ~ Gamma(2, 0.5)
Likelihood: r[j] ~ Binomial(n[j], inv_logit(theta[j]))
            theta[j] ~ Normal(mu[z[j]], 1/sqrt(tau[z[j]]))
```

**Falsification Criteria:**
- REJECT if: K_effective = 1 (no clustering)
- REJECT if: K_effective > 6 (overfitting)
- REJECT if: Singletons > 5 (fragmentation)
- REJECT if: ΔLOO vs Experiment 2 < -5 (worse than FMM)

**Expected Outcome:** REVISE or REJECT (high uncertainty)
- Most likely: K_effective ≈ 3 (confirms Experiment 2)
- Possible: K_effective = 1 (clusters are artifacts)
- Low probability: K_effective ∈ {2, 4, 5}

**Trigger Condition:** Only run if:
- Experiment 2 ACCEPT (clusters seem real)
- Want to test sensitivity to K=3 assumption

---

### **Experiments 6-9: Covariate Models** [LOW PRIORITY]

**Source:** Designer 3 (Models 3.1, 3.2, 3.3)
**Priority:** LOWEST (EDA found no covariate effects)
**Status:** DEFERRED unless justified

**Rationale for Low Priority:**
- EDA found r = -0.34, p = 0.278 (no sample size effect)
- No sequential dependence (p > 0.23)
- Only 12 groups → low power for covariate detection

**Models Proposed:**
- Experiment 6: Sample size covariate (log n_trials)
- Experiment 7: Quadratic group effect
- Experiment 8: Random slopes (varying size effects)

**When to Run:**
1. If all random effects models (1-5) fail falsification
2. If domain knowledge suggests covariates important
3. If reviewer/stakeholder requests covariate exploration

**Falsification (if run):**
- REJECT if: ΔLOO vs Experiment 1 < 2 (no improvement)
- REJECT if: Covariate coefficients include zero with high probability

---

## Iteration Strategy

### Within Model Class (Refinement)

**Refine if:**
- Fixable issues (wrong prior scale, missing constraint)
- Clear improvement path (e.g., reparameterization)
- Convergence issues solvable (increase adapt_delta, warmup)

**Examples:**
- Experiment 1 divergences → Increase adapt_delta to 0.95
- Experiment 2 label switching → Add ordered constraint
- Shrinkage unreasonable → Adjust hyperprior scale

### Between Model Classes (Switching)

**Switch if:**
- Fundamental misspecification (posterior predictive p < 0.01)
- Multiple refinements fail (3 attempts with no improvement)
- Cluster structure dominates (Experiment 2 ΔLOO > 10 vs Experiment 1)

**Pivot Rules:**
- If Experiment 1 fails → Try Experiment 3 (robust)
- If Experiments 1 & 3 fail → Try Experiment 2 (mixture)
- If all fail → Reconsider model class (stay Bayesian!)

### Stopping Criteria

**Stop experimenting if:**
1. **SUCCESS:** Model ACCEPT with adequate fit (ELPD ± SE reasonable)
2. **DIMINISHING RETURNS:** Improvements < 2 SE between models
3. **COMPUTATIONAL LIMITS:** Runtime exceeds 10 min per model
4. **DATA QUALITY:** Discover fundamental data issues

---

## Model Comparison Strategy

### LOO-CV (Primary)

**All ACCEPT models compared via:**
```python
loo_dict = {
    'experiment_1': az.loo(idata_1, pointwise=True),
    'experiment_2': az.loo(idata_2, pointwise=True),
    ...
}
comparison = az.compare(loo_dict)
```

**Decision Rules:**
- ΔLOO < 2: Models equivalent → Prefer simpler (parsimony)
- 2 < ΔLOO < 4: Weak evidence → Report uncertainty
- ΔLOO > 4: Strong evidence → Clear winner

**Pareto-k Diagnostics:**
- k < 0.5: Good (all points)
- 0.5 < k < 0.7: OK (few points acceptable)
- k > 0.7: Bad (refit with importance sampling or K-fold CV)

### Posterior Predictive Checks (Secondary)

**Visual checks for all models:**
- Observed vs predicted success rates (group-level)
- Posterior predictive p-values (T_rep vs T_obs)
- Residual plots (standardized residuals vs fitted)

**Calibration check:**
- 50% of observed should fall in 50% posterior interval
- 90% of observed should fall in 90% posterior interval

### Model Stacking (If Uncertainty)

**If multiple models have similar LOO (ΔLOO < 2):**
```python
weights = az.compare(loo_dict, method='stacking')['weight']
```
Report weights and use ensemble predictions.

---

## Prior Sensitivity Analysis

### Three Prior Specifications (Per Model)

**1. Vague (Reference):**
```
mu ~ Normal(0, 10)
tau ~ Normal(0, 5)
```

**2. Weakly Informative (Primary):**
```
mu ~ Normal(-2.6, 1.0)
tau ~ Normal(0, 0.5)
```

**3. Moderately Informative (Stronger):**
```
mu ~ Normal(-2.6, 0.5)
tau ~ Normal(0, 0.2)
```

**Sensitivity Check:**
- Posteriors should be similar across all three
- If posteriors differ substantially → Data weak, report uncertainty
- If vague prior gives extreme posteriors → Data insufficient

**Required for:** All ACCEPT models before final decision

---

## Validation Pipeline (Per Experiment)

### Stage 1: Prior Predictive Check
- Agent: `prior-predictive-checker`
- Output: `experiments/experiment_N/prior_predictive_check/`
- Criteria: Priors generate plausible data (success rates ∈ [0, 1])
- FAIL → Skip to next model

### Stage 2: Simulation-Based Calibration
- Agent: `simulation-based-validator`
- Output: `experiments/experiment_N/simulation_based_validation/`
- Criteria: Model recovers known parameters from simulated data
- FAIL → Skip to next model

### Stage 3: Posterior Inference
- Agent: `model-fitter`
- Output: `experiments/experiment_N/posterior_inference/`
- Criteria: Rhat < 1.01, ESS > 400, divergences < 1%, log_lik saved
- FAIL → Try refinement or skip

### Stage 4: Posterior Predictive Check
- Agent: `posterior-predictive-checker`
- Output: `experiments/experiment_N/posterior_predictive_check/`
- Criteria: Good visual fit, p-value ∈ [0.05, 0.95]
- Document fit quality (continue regardless)

### Stage 5: Model Critique
- Agent: `model-critique` (parallel if borderline)
- Output: `experiments/experiment_N/model_critique/`
- Decision: ACCEPT / REVISE / REJECT
- ACCEPT → Add to successful models
- REVISE → `model-refiner` creates new experiment
- REJECT → Document and move to next model

---

## Minimum Attempt Policy

**REQUIRED ATTEMPTS:**
- Must attempt **Experiment 1** (baseline mandatory)
- Must attempt **Experiment 2** (complementary model class)
- Exception: If Experiment 1 fails pre-fit validation (Stages 1-2), may skip Experiment 2

**RATIONALE:**
- Experiment 1 = standard approach (defensible baseline)
- Experiment 2 = tests alternative hypothesis (cluster structure)
- Two attempts ensure adequate exploration before stopping

**DOCUMENTATION:**
- If fewer than two models attempted, document reason in `log.md`
- Explain why minimum not met (e.g., computational failure, data quality)

---

## Expected Timeline

| Stage | Duration | Agent |
|-------|----------|-------|
| Experiment 1 (all stages) | 2-3 hours | 5 agents (sequential) |
| Experiment 2 (all stages) | 2-3 hours | 5 agents (sequential) |
| Experiment 3 (if triggered) | 2-3 hours | 5 agents (sequential) |
| Model assessment | 1 hour | 1 agent |
| Model comparison (if 2+) | 1 hour | 1 agent |
| Adequacy assessment | 30 min | 1 agent |
| Final report | 1 hour | 1 agent |
| **TOTAL** | **8-12 hours** | **~15-25 agents** |

---

## Falsification Summary Table

| Experiment | Primary Falsification Criterion | Threshold | Action if Failed |
|------------|--------------------------------|-----------|-----------------|
| 1 | Posterior predictive p-value | < 0.05 | Try Experiment 3 |
| 1 | MCMC diagnostics | Rhat > 1.01 | Reparameterize, retry |
| 2 | Cluster certainty | < 0.6 | Use Experiment 1 |
| 2 | K_effective | < 2 | Clusters artifacts, use Exp 1 |
| 3 | Degrees of freedom | > 30 | Use Experiment 1 (Normal OK) |
| 3 | ΔLOO vs Experiment 1 | < 2 | Use Experiment 1 (no benefit) |
| 4 | Overdispersion parameter | > 1000 | Use Experiment 1 |
| 5 | K_effective | = 1 or > 6 | Reject, use prior results |

---

## Success Criteria

**Adequate Model Achieved If:**
1. ✓ At least one model ACCEPT (passes all validation)
2. ✓ MCMC diagnostics pass (Rhat < 1.01, ESS > 400)
3. ✓ Posterior predictive checks reasonable (p ≥ 0.05)
4. ✓ LOO-CV diagnostics good (Pareto k < 0.7)
5. ✓ Prior sensitivity analysis shows robustness
6. ✓ Model uncertainty quantified (if multiple models similar)

**Report Even If:**
- No model ACCEPT → Document failures, recommend simpler approach
- Model uncertainty high → Report stacking weights, ensemble predictions
- Limitations discovered → Honest assessment of data/model limitations

---

## Integration Notes

### Designer 1 (Hierarchical Models)
- **Primary contribution:** Experiments 1, 3, 4
- **Strengths:** Standard approaches, well-documented, computationally efficient
- **Files:** `experiments/designer_1/proposed_models.md`, Stan files

### Designer 2 (Alternative Approaches)
- **Primary contribution:** Experiments 2, 5
- **Strengths:** Tests cluster hypothesis, non-parametric robustness
- **Files:** `experiments/designer_2/proposed_models.md`, Stan files, Python pipeline

### Designer 3 (Covariate Models)
- **Primary contribution:** Experiments 6-8 (deferred)
- **Strengths:** Covariate testing, regression extensions
- **Files:** `experiments/designer_3/proposed_models.md`, Stan files
- **Status:** On hold unless justified by domain knowledge or model failures

---

## Risk Mitigation

### Computational Risks
- **Risk:** Models fail to converge
- **Mitigation:** Non-centered parameterization, increase adapt_delta, longer warmup

### Statistical Risks
- **Risk:** All models rejected
- **Mitigation:** Multiple model classes proposed, refinement strategy in place

### Time Risks
- **Risk:** Experiments take longer than expected
- **Mitigation:** Prioritized queue, minimum attempt policy, clear stopping criteria

### Interpretation Risks
- **Risk:** Model uncertainty too high to draw conclusions
- **Mitigation:** Report uncertainty honestly, use stacking, ensemble predictions

---

## Conclusion

This experiment plan prioritizes:
1. **Baseline first** (Experiment 1 = standard hierarchy)
2. **Alternative hypothesis** (Experiment 2 = clusters)
3. **Robustness checks** (Experiments 3-5 = conditional)
4. **Clear falsification** (explicit rejection criteria)
5. **Minimum exploration** (at least 2 models attempted)

The plan balances thoroughness with efficiency, ensures adequate exploration, and maintains strict Bayesian requirements (MCMC inference, posterior predictive checks, LOO-CV comparison).

**Next Step:** Begin Experiment 1 with prior predictive check.

---

**Document Status:** ✅ Complete and ready for implementation
**Last Updated:** After synthesis of three parallel designer proposals
