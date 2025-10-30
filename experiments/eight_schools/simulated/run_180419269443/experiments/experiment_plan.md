# Synthesized Bayesian Modeling Experiment Plan

**Date:** 2025-10-28
**Synthesis:** Combined proposals from three independent model designers
**Dataset:** Meta-analysis with J=8 studies, I²=2.9%, pooled effect=11.27

---

## Synthesis Overview

Three independent designers proposed 9 total model specifications, which synthesize into **5 distinct model classes** after removing duplicates and consolidating similar approaches:

### Designer Contributions:
- **Designer 1 (Classical):** Complete pooling, partial pooling, skeptical prior hierarchical
- **Designer 2 (Robustness):** Heavy-tailed (t-distribution), mixture model, Dirichlet process
- **Designer 3 (Prior sensitivity):** Weakly informative hierarchical, conflict detection, skeptical-enthusiastic ensemble

### Synthesis Result:
After removing duplicates (e.g., Designer 1 and 3 both propose standard hierarchical models) and grouping by fundamental model class:

1. **Hierarchical Model with Normal Likelihood** (baseline)
2. **Complete Pooling Model** (boundary case)
3. **Heavy-Tailed Hierarchical (t-distribution)** (robustness)
4. **Skeptical-Enthusiastic Prior Ensemble** (prior sensitivity)
5. **Mixture Model** (heterogeneity in heterogeneity)

---

## Model Priority Order

Based on theoretical justification, EDA findings, computational feasibility, and information value:

### **EXPERIMENT 1: Hierarchical Model with Normal Likelihood** [PRIORITY: HIGHEST]

**Rationale for priority:**
- Standard meta-analysis approach, well-justified by literature
- EDA supports normal assumptions (no outliers, low heterogeneity)
- Fast to fit (~1 min), establishes baseline
- Proposed by both Designer 1 and 3 independently (convergence)
- MUST fit this before others to establish reference

**Mathematical Specification:**
```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)    for i = 1,...,8

Hierarchical structure:
  theta_i ~ Normal(mu, tau)

Priors (weakly informative):
  mu ~ Normal(0, 25)
  tau ~ Half-Normal(0, 10)
```

**Implementation:** Stan with non-centered parameterization (avoid funnel geometry)

**Falsification Criteria (abandon if):**
1. Posterior tau > 15 (heterogeneity severely underestimated)
2. Multiple Pareto k > 0.7 (outliers not captured by normal)
3. Posterior predictive checks fail (systematic misfit)
4. Study 4 has >100% influence (fragility)
5. Prior predictive check fails (observed data in extreme 5% tails)

**Expected Results:** mu ≈ 11 ± 4, tau ≈ 2 ± 2, strong shrinkage (>95%)

---

### **EXPERIMENT 2: Complete Pooling (Common Effect)** [PRIORITY: HIGH]

**Rationale for priority:**
- AIC-preferred in EDA (63.85 vs 65.82 for random effects)
- I²=2.9% suggests homogeneity
- Tests null hypothesis: "no between-study heterogeneity"
- Very fast (~30 sec), provides comparison benchmark
- Proposed by Designer 1

**Mathematical Specification:**
```
Likelihood:
  y_i ~ Normal(mu, sigma_i)    for i = 1,...,8

Prior:
  mu ~ Normal(0, 50)
```

**Implementation:** Stan (simple model)

**Falsification Criteria (abandon if):**
1. LOO strongly prefers Experiment 1 (ΔELPD > 4)
2. Posterior predictive checks show systematic under-dispersion
3. Residuals show patterns (e.g., Study 5 consistently extreme)
4. Prior sensitivity analysis reveals instability

**Expected Results:** mu ≈ 11.27 ± 3.8, narrower CI than Experiment 1

**Comparison to Experiment 1:** If |mu_diff| < 2 and CIs overlap strongly, pooling is adequate

---

### **EXPERIMENT 3: Heavy-Tailed Hierarchical (t-distribution)** [PRIORITY: MEDIUM]

**Rationale for priority:**
- Tests robustness to outlier assumptions
- Study 5 (-4.88) is only negative effect
- Small sample (J=8) limits outlier detection power
- Future-proofs against new outliers
- Proposed by Designer 2

**Mathematical Specification:**
```
Likelihood:
  y_i ~ Student_t(nu, theta_i, sigma_i)    for i = 1,...,8

Hierarchical structure:
  theta_i ~ Normal(mu, tau)

Priors:
  mu ~ Normal(0, 50)
  tau ~ Half-Cauchy(0, 5)
  nu ~ Gamma(2, 0.1)    # degrees of freedom, mean=20
```

**Implementation:** Stan

**Falsification Criteria (abandon if):**
1. Posterior nu > 50 (data prefers normal, return to Experiment 1)
2. Computational issues persist (divergences despite tuning)
3. LOO shows no improvement over Experiment 1
4. Prior predictive check generates extreme outliers not in data

**Expected Results:** nu ≈ 30-50 (near-normal), similar mu/tau to Experiment 1

**Decision Rule:** If nu > 50, abandon and use Experiment 1 results

---

### **EXPERIMENT 4: Skeptical-Enthusiastic Prior Ensemble** [PRIORITY: MEDIUM]

**Rationale for priority:**
- Critical for small sample (J=8) where priors matter
- Tests robustness to prior choice
- EDA shows large positive effect (11.27), but is it robust?
- Proposed by Designer 3
- MANDATORY for credible inference with small J

**Mathematical Specification:**
```
Model 4a (Skeptical):
  y_i ~ Normal(theta_i, sigma_i)
  theta_i ~ Normal(mu, tau)
  mu ~ Normal(0, 10)           # Skeptical of large effects
  tau ~ Half-Normal(0, 5)       # Expects low heterogeneity

Model 4b (Enthusiastic):
  y_i ~ Normal(theta_i, sigma_i)
  theta_i ~ Normal(mu, tau)
  mu ~ Normal(15, 15)           # Expects large positive effect
  tau ~ Half-Cauchy(0, 10)      # Allows high heterogeneity

Ensemble: Combine via LOO stacking weights
```

**Implementation:** Stan (fit both models separately)

**Falsification Criteria (abandon if):**
1. Models converge trivially (|mu_4a - mu_4b| < 1) → Data overwhelms priors, prior choice irrelevant
2. Models diverge absurdly (|mu_4a - mu_4b| > 20) → Data insufficient
3. Computational failure in either model
4. Stacking weights extreme (w < 0.01) → One prior inappropriate

**Expected Results:** Moderate agreement (|mu_diff| ≈ 3-7), ensemble mu ≈ 11 ± 5

**Decision Rule:**
- If |mu_diff| < 5: Robust inference
- If 5 < |mu_diff| < 10: Report range, acknowledge uncertainty
- If |mu_diff| > 10: Data insufficient, report honestly

---

### **EXPERIMENT 5: Mixture Model** [PRIORITY: LOW - CONDITIONAL]

**Rationale for priority:**
- LOW priority: EDA shows low heterogeneity, mixture may be overfit
- Tests hypothesis: "Study 5 belongs to different population"
- Allows two clusters with different means/variances
- Proposed by Designer 2
- **ONLY fit if Experiments 1-3 show evidence of subpopulations**

**Mathematical Specification:**
```
Likelihood:
  y_i ~ pi * Normal(mu_1, sqrt(tau_1² + sigma_i²)) +
        (1-pi) * Normal(mu_2, sqrt(tau_2² + sigma_i²))

Priors:
  pi ~ Beta(2, 2)                     # Cluster proportion
  mu_1, mu_2 ~ Normal(0, 50)
  tau_1, tau_2 ~ Half-Normal(0, 10)
```

**Implementation:** PyMC (easier for mixture models)

**Falsification Criteria (abandon if):**
1. Mixture collapses (pi < 0.1 or pi > 0.9) → Return to Experiment 1
2. Label switching in MCMC
3. LOO shows no improvement over Experiment 1
4. Computational failure

**Expected Results (if EDA correct):** Mixture collapses, pi → 0 or 1

**Decision Rule:** SKIP unless Experiments 1-3 show:
- Pareto k > 0.7 for multiple studies, OR
- Posterior predictive checks reveal two distinct clusters, OR
- Study 5 consistently flagged as outlier

---

## Convergent Findings Across Designers

### All designers agreed on:

1. **Normal likelihood is reasonable starting point** (given EDA)
2. **Partial pooling (hierarchical) is primary approach** (baseline)
3. **Prior sensitivity testing is mandatory** (J=8 is small)
4. **Study 4 influence requires checking** (33% impact)
5. **Falsification criteria must be explicit** (abandon if criteria met)
6. **Non-centered parameterization for hierarchical models** (avoid funnel)

### Divergent perspectives:

1. **Designer 1:** Emphasized pooling strategies (complete vs partial)
2. **Designer 2:** Emphasized robustness (t-distribution, mixture)
3. **Designer 3:** Emphasized prior sensitivity (ensemble approach)

**Synthesis value:** Combined, these perspectives ensure comprehensive model space exploration

---

## Minimum Attempt Policy

**Must attempt:** Experiments 1 and 2 (unless Experiment 1 fails pre-fit validation)

**Rationale:**
- Experiment 1 is baseline hierarchical (standard approach)
- Experiment 2 tests null hypothesis of homogeneity
- Together they bracket the model space given EDA findings

**If fewer than 2 attempted:** Document reason in log.md

---

## Critical Decision Points

### After Experiment 1:

**IF convergence + LOO good + posterior predictive checks pass:**
→ Proceed to Experiment 2 (comparison)

**IF convergence but multiple Pareto k > 0.7:**
→ Proceed to Experiment 3 (robust)

**IF divergences persist despite tuning:**
→ Check prior predictive, investigate parameterization

**IF posterior tau > 15:**
→ STOP, reconsider model class (need covariates?)

### After Experiments 1-2:

**IF models agree (|mu_diff| < 2) AND LOO difference < 2:**
→ Proceed to Experiment 4 (prior sensitivity mandatory)
→ SKIP Experiments 3 and 5 (not needed)

**IF models disagree OR LOO strongly favors one:**
→ Investigate cause, may need Experiment 3

### After Experiments 1-4:

**IF all converge with similar results:**
→ Proceed to Phase 4 (model assessment)

**IF evidence of outliers or subpopulations:**
→ Consider Experiment 5 (mixture)

**IF results unstable or contradictory:**
→ Consult model-adequacy-assessor

---

## Falsification Philosophy

### RED FLAGS that require STOPPING ALL experiments:

1. **Prior predictive failures across all models** → Fundamental misspecification
2. **Study 4 has >100% influence across all models** → Data too fragile
3. **Posterior mu negative across all models** → Contradicts EDA fundamentally
4. **Computational failure across all models** → Implementation error
5. **All models converge to I² > 50%** → EDA severely misled

### What success looks like:

- **NOT:** All models converging smoothly to same answer
- **YES:** Understanding which models fail and why
- **YES:** Quantifying uncertainty honestly
- **YES:** Finding evidence that forces model revision

---

## Implementation Sequence

### Phase 3 Timeline (Estimated):

**Week 1:**
- Day 1-2: Experiment 1 (prior predictive, fit, diagnostics)
- Day 3-4: Experiment 2 (fit, compare to Exp 1)
- Day 5: Assessment, decide on Experiments 3-5

**Week 2 (if needed):**
- Day 1-2: Experiment 4 (prior sensitivity)
- Day 3-4: Experiment 3 OR 5 (if warranted)
- Day 5: Synthesis

**Phase 4:** Model assessment (always runs after Phase 3)

---

## Expected Outcome (Prediction)

**Most likely (70% confidence):**
- Experiments 1 and 2 converge to similar results
- tau ≈ 2, mu ≈ 11 ± 4
- Complete pooling adequate (parsimony rule)
- Experiments 3-5 not needed (agreement with normal model)

**Alternative (25% confidence):**
- Experiment 1 preferred over 2 (LOO favors hierarchical)
- tau > 0 but small, partial pooling warranted
- Experiment 4 shows moderate prior sensitivity

**Would be surprised (5% confidence):**
- Experiment 3 strongly preferred (nu < 20, heavy tails matter)
- Mixture model needed (genuine subpopulations)
- Prior sensitivity extreme (data insufficient)

---

## Stan vs PyMC Implementation

**Use Stan for:**
- Experiments 1, 2, 3, 4 (normal and t-distribution likelihoods)
- Better HMC performance for these model classes
- Established non-centered parameterization patterns

**Use PyMC for:**
- Experiment 5 (mixture model, if needed)
- Easier specification of complex mixtures
- Good diagnostics tools via ArviZ

**Critical:** All models must save log_likelihood for LOO comparison

---

## Success Criteria for Phase 3

**Before proceeding to Phase 4:**
- ✅ At least Experiments 1-2 attempted (unless validation failures)
- ✅ All fitted models converged (R-hat < 1.01, ESS > 400)
- ✅ Log-likelihood saved in all InferenceData objects
- ✅ Posterior predictive checks completed
- ✅ Falsification criteria evaluated
- ✅ Decision documented (ACCEPT/REVISE/REJECT) for each experiment

**Phase 4 always runs** after Phase 3 to provide assessment context

---

## Summary Statistics for Quick Reference

| Experiment | Model Class | Priority | Time | Software | Designer Source |
|------------|-------------|----------|------|----------|-----------------|
| 1 | Normal Hierarchical | HIGHEST | 1-2 hrs | Stan | 1 & 3 |
| 2 | Complete Pooling | HIGH | 30 min | Stan | 1 |
| 3 | Heavy-Tailed | MEDIUM | 2-3 hrs | Stan | 2 |
| 4 | Prior Ensemble | MEDIUM | 2-3 hrs | Stan | 3 |
| 5 | Mixture | LOW | 3-4 hrs | PyMC | 2 |

**Total estimated time:** 6-8 hours (Experiments 1-2), up to 15-20 hours (all experiments)

---

## Theoretical Foundation

Models grounded in established literature:
- DerSimonian & Laird (1986): Random effects meta-analysis
- Gelman & Hill (2007): Hierarchical modeling
- Higgins & Thompson (2002): I² statistic
- Betancourt (2018): HMC geometry
- Vehtari et al. (2017): LOO cross-validation
- Rubin (1981): Eight Schools problem (similar structure)

---

**Next Step:** Begin Phase 3 with Experiment 1 - Prior Predictive Check
