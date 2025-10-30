# Experiment Plan: Bayesian Models for Hierarchical Measurement Error Data
## Model Designer 1

**Date:** 2025-10-28
**Status:** Ready for Implementation

---

## Overview

This plan proposes THREE competing Bayesian model classes with explicit falsification criteria. Each model tests different assumptions about the data generation process.

**Key Strategy:** Start with Model 1 (EDA-supported complete pooling). Models 2-3 are designed to CHALLENGE this baseline and discover if hierarchical structure or outliers exist despite null EDA findings.

---

## Three Model Classes

### Model 1: Complete Pooling (Baseline)
**Hypothesis:** All groups share a single true mean

**Specification:**
```
y_i ~ Normal(mu, sigma_i)    # Known sigma_i
mu ~ Normal(10, 20)           # Weakly informative
```

**Why:** EDA supports homogeneity (p=0.42, tau^2=0)

**I will abandon this if:**
- Posterior predictive p-value < 0.05 or > 0.95
- Any standardized residual |r_i| > 3
- Any Pareto k > 0.7 in LOO-CV
- Systematic patterns in residuals

**Expected outcome:** ACCEPT - best model given EDA

---

### Model 2: Hierarchical Partial Pooling (Falsification Test)
**Hypothesis:** Groups have distinct means from common distribution

**Specification:**
```
y_i ~ Normal(theta_i, sigma_i)    # Known sigma_i
theta_i ~ Normal(mu, tau)         # Group-level
mu ~ Normal(10, 20)
tau ~ Half-Cauchy(0, 5)
```

**Why:** Standard approach; tests if EDA missed subtle structure

**I will abandon this if:**
- Posterior median tau < 1 (groups homogeneous → revert to Model 1)
- LOO improvement over Model 1 < 2 ELPD
- Divergent transitions despite non-centered parameterization
- All theta_i posteriors indistinguishable

**Expected outcome:** REJECT - tau will be ~0, revert to Model 1

---

### Model 3: Robust t-Distribution (Outlier Test)
**Hypothesis:** Group 4 is an outlier requiring robust estimation

**Specification:**
```
y_i ~ Student_t(nu, mu, sigma_i)
mu ~ Normal(10, 20)
nu ~ Gamma(2, 0.1)                # Tail heaviness
```

**Why:** Tests if negative observation (Group 4) is genuinely anomalous

**I will abandon this if:**
- Posterior median nu > 30 (data consistent with normal → revert to Model 1)
- LOO improvement over Model 1 < 2 ELPD
- |mu_robust - mu_model1| < 2 (robustness not affecting inference)

**Expected outcome:** REJECT - nu will be >30, revert to Model 1

---

## Implementation Plan

### Phase 1: Fit and Diagnose (Priority 1)
1. Implement all three models in Stan
2. Run MCMC: 4 chains, 2000 iterations, warmup=1000
3. Check convergence diagnostics:
   - R-hat < 1.01 for all parameters
   - ESS_bulk > 400 per chain
   - ESS_tail > 400 per chain
   - Zero divergent transitions

### Phase 2: Model Checking (Priority 1)
4. Posterior predictive checks:
   - Chi-square test statistic
   - Visual: observed vs replicated data
   - p-value should be in [0.05, 0.95]

5. LOO cross-validation:
   - Compute ELPD_LOO for each model
   - Check Pareto k diagnostics
   - Compare models (Delta-ELPD > 2*SE for meaningful difference)

6. Residual analysis:
   - Standardized residuals: r_i = (y_i - mu_post) / sigma_i
   - Check for patterns, outliers (|r_i| > 3)

### Phase 3: Falsification Tests (Priority 1)
7. **Model 1:**
   - If posterior predictive checks pass AND Pareto k < 0.7 → ACCEPT
   - Otherwise → Move to Model 2 or 3

8. **Model 2:**
   - If median tau < 1 → REVERT to Model 1
   - If tau > 5 → ACCEPT, investigate group structure
   - If 1 < tau < 5 → Uncertain, use conservatively

9. **Model 3:**
   - If median nu > 30 → REVERT to Model 1
   - If nu < 10 → ACCEPT, investigate outliers

### Phase 4: Sensitivity Analysis (Priority 2)
10. Prior sensitivity for winning model:
    - Vary prior width: mu ~ N(10, 10), N(10, 20), N(10, 40)
    - Check posterior robustness

11. Leave-one-out model fits:
    - Refit excluding each group
    - Check influence (max change in mu < 2 units expected)

12. Stress tests (see detailed plan in proposed_models.md)

---

## Decision Rules

### Accept Model 1 if:
- Posterior predictive checks pass (0.05 < p < 0.95)
- All Pareto k < 0.7
- No systematic residual patterns
- Model 2 estimates tau < 1
- Model 3 estimates nu > 30

### Accept Model 2 if:
- Posterior median tau > 5
- Improves LOO by > 2 ELPD over Model 1
- No computational pathologies
- theta_i are meaningfully different (95% CIs don't overlap)

### Accept Model 3 if:
- Posterior median nu < 10
- Improves LOO by > 2 ELPD over Model 1
- mu estimate differs from Model 1 by > 2 units
- Identifies specific outliers

---

## Red Flags (Trigger Major Pivot)

### Level 1: Minor Issues (Iterate within current models)
- Single parameter has R-hat = 1.05 → Run longer
- One Pareto k = 0.75 → Investigate that observation
- Posterior predictive p = 0.04 → Check specific discrepancy

### Level 2: Major Issues (Reconsider model class)
- All models fail posterior predictive checks → Investigate data generation process
- Persistent divergent transitions across all models → Model-data mismatch
- Group 4 has Pareto k > 1.0 in all models → Qualitatively different observation
- All posteriors track priors closely → Insufficient data

### Level 3: Critical Issues (Stop modeling, investigate data)
- Computational failures across all implementations
- Physically implausible parameter estimates
- Evidence of data quality issues
- Measurement model assumptions violated

---

## Alternative Approaches (If All Three Models Fail)

### Escape Route 1: Regression Structure
```
y_i ~ Normal(alpha + beta * group_i, sigma_i)
```
**When:** Groups show linear trend, not random variation

### Escape Route 2: Mixture Model
```
y_i ~ p * Normal(mu_1, sigma_i) + (1-p) * Normal(mu_2, sigma_i)
```
**When:** Evidence for two distinct subpopulations (e.g., groups 0-3 vs 4-7)

### Escape Route 3: Measurement Error Uncertainty
```
y_i ~ Normal(mu, sigma_i * kappa_i)
kappa_i ~ LogNormal(0, 0.5)
```
**When:** Residual patterns suggest sigma values are misspecified

### Escape Route 4: Correlated Errors
```
y ~ MVNormal(mu * 1, Sigma)
Sigma_ij = sigma_i^2 (diagonal) + rho * sigma_i * sigma_j (off-diagonal)
```
**When:** Evidence for spatial/temporal correlation

### Escape Route 5: Data Investigation
**Action:** Stop Bayesian modeling, return to data collection/quality assessment
**When:** Fundamental issues with data generation process

---

## Success Criteria

This experiment is SUCCESSFUL if:

1. **We identify the true data generation process**
   - Not just "fit three models"
   - But discover which one actually explains the data

2. **We can definitively reject inadequate models**
   - Clear criteria applied
   - Transparent decision-making
   - Document why models were rejected

3. **We have quantified uncertainty appropriately**
   - Credible intervals reflect true uncertainty
   - Posterior predictive distributions match data structure
   - No false precision

4. **We're ready to pivot if needed**
   - If all models fail, we know what to try next
   - If surprising results emerge, we reconsider assumptions
   - Truth over task completion

---

## Expected Timeline

**Phase 1 (Implementation):** 2-4 hours
- Write Stan code for 3 models
- Debug compilation issues
- Initial test runs

**Phase 2 (Diagnostics):** 2-3 hours
- Run production MCMC (4 chains x 3 models)
- Extract diagnostics
- Generate trace plots, pairs plots

**Phase 3 (Model Checking):** 3-4 hours
- Posterior predictive checks
- LOO cross-validation
- Residual analysis
- Apply falsification criteria

**Phase 4 (Reporting):** 2-3 hours
- Document decisions
- Create visualizations
- Write summary of findings

**Total:** 9-14 hours (for thorough analysis)

**Fast Track (Priority 1 only):** 5-7 hours
- Fit Model 1, check diagnostics
- If passes, fit Models 2-3 for comparison
- Apply falsification tests
- Report winner

---

## Deliverables

### Primary Outputs
1. **Stan code** for all three models (`model_1.stan`, `model_2.stan`, `model_3.stan`)
2. **Posterior samples** (CSV or netCDF format)
3. **Diagnostic report** (R-hat, ESS, divergences for each model)
4. **Model comparison table** (LOO-CV, ELPD, Pareto k)
5. **Decision document** (which model accepted/rejected and why)

### Visualizations
6. Trace plots (for each model)
7. Pairs plots (for Model 2, checking funnel geometry)
8. Posterior predictive distributions vs observed data
9. Residual plots (standardized residuals with error bars)
10. LOO pointwise comparison (identify influential observations)

### Documentation
11. **Final report** synthesizing results
12. **Falsification log** documenting each decision point
13. **Sensitivity analysis** results
14. **Recommendations** for future work

---

## Computational Resources

**Requirements:**
- Stan (via CmdStan, PyStan, or RStan)
- Python/R for post-processing
- 4 CPU cores (for parallel chains)
- ~1GB RAM (small dataset, simple models)

**Expected Runtime:**
- Model 1: ~1 minute per chain
- Model 2: ~2-5 minutes per chain (depending on funnel)
- Model 3: ~2 minutes per chain
- Total: ~15-30 minutes for all models

**Storage:**
- ~50 MB for posterior samples (all models)
- ~100 MB for visualizations and diagnostics

---

## Key References

1. **Prior selection:** Gelman (2006) - Prior distributions for variance parameters
2. **Model checking:** Gabry et al. (2019) - Visualization in Bayesian workflow
3. **LOO-CV:** Vehtari et al. (2017) - Practical Bayesian model evaluation
4. **Hierarchical models:** Gelman et al. (2013) - Bayesian Data Analysis, Ch. 5
5. **Robust models:** Kruschke (2013) - Bayesian estimation supersedes the t-test

---

## Contact and Questions

**Model Design Document:** `/workspace/experiments/designer_1/proposed_models.md`
- Full mathematical specifications
- Detailed falsification criteria
- Stress tests for each model
- Computational considerations

**This Experiment Plan:** `/workspace/experiments/designer_1/experiment_plan.md`
- High-level strategy
- Implementation roadmap
- Decision rules and red flags

---

## Final Note

**This is not a predetermined script.** It's a framework for discovering truth.

If Model 1 fails unexpectedly → Investigate why
If Model 2 succeeds unexpectedly → EDA was misleading, learn from it
If all models fail → Pivot to alternative approaches

**Success = Finding the right model, not completing the plan.**

---

**Ready to implement:** All specifications complete. Proceed with Stan implementation when ready.
