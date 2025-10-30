# Designer 3: Prior Specification and Model Adequacy
## Bayesian Meta-Analysis Modeling Strategy

**Designer:** Model Designer 3
**Focus:** Prior specification strategies, model adequacy checking, robustness analysis
**Date:** 2025-10-28
**Status:** Proposal complete, awaiting implementation approval

---

## Overview

This directory contains a comprehensive Bayesian modeling strategy for meta-analysis with **J=8 studies**. The key challenge is that with such a small sample size, **prior choices critically affect inference**. Rather than pretending priors don't matter, this strategy explicitly tests robustness to prior specification.

**Core principle:** Plan for failure, test assumptions adversarially, and be ready to abandon models when evidence warrants.

---

## Files in This Directory

### 1. Core Documentation
- **`proposed_models.md`** - Main proposal document with 3 distinct model classes
- **`README.md`** - This file, overview and quick start guide
- **`diagnostics_checklist.md`** - Systematic checklist for model validation

### 2. Model Specifications
- **`model_1_spec.stan`** - Weakly informative hierarchical model (BASELINE)
- **`model_2_spec.py`** - Prior-data conflict detection model (PyMC, DIAGNOSTIC)
- **`model_3a_spec.stan`** - Skeptical prior model (ROBUSTNESS)
- **`model_3b_spec.stan`** - Enthusiastic prior model (ROBUSTNESS)

### 3. Validation Code
- **`prior_predictive_checks.py`** - Pre-data validation of priors

---

## Three Model Classes (Not Just Parameter Variations)

### Model 1: Weakly Informative Hierarchical (BASELINE)
**Philosophy:** Standard Bayesian approach with carefully chosen priors

**Priors:**
- `mu ~ Normal(0, 25)` - Weakly informative on mean effect
- `tau ~ Half-Normal(0, 10)` - Standard meta-analysis prior

**When to use:** Primary analysis, fast, interpretable

**When it fails:** If prior-posterior conflicts arise, or heterogeneity severely underestimated

---

### Model 2: Prior-Data Conflict Detection (DIAGNOSTIC)
**Philosophy:** Explicitly model disagreement between prior assumptions and data

**Key features:**
- Mixture priors on mu (skeptical + optimistic)
- Conflict detection via Bernoulli indicators
- SE inflation for conflicted studies

**When to use:** If Model 1 shows poor LOO-CV or influential outliers

**When it fails:** If conflict mechanism is unused OR overfits noise

---

### Model 3: Skeptical-Enthusiastic Ensemble (ROBUSTNESS)
**Philosophy:** Test robustness by deliberately choosing opposing priors

**Key features:**
- Model 3a: Skeptical (null-favoring)
- Model 3b: Enthusiastic (optimistic)
- Ensemble via stacking weights

**When to use:** Mandatory sensitivity analysis for all projects

**When it fails:** If models converge trivially OR diverge absurdly

---

## Recommended Workflow

### Phase 1: Pre-Fitting Validation (30 minutes)
```bash
# Run prior predictive checks
cd /workspace/experiments/designer_3
python prior_predictive_checks.py
```

**Decision point:** Do priors include observed data?
- YES → Proceed to Phase 2
- NO → Revise priors, repeat Phase 1

---

### Phase 2: Fit Baseline Model (15 minutes)
```python
# Fit Model 1 in Stan
import cmdstanpy
model1 = cmdstanpy.CmdStanModel(stan_file='model_1_spec.stan')
fit1 = model1.sample(data={'J': 8, 'y': y_obs, 'sigma': sigma_obs})
```

**Decision point:** Do diagnostics pass? (Check `diagnostics_checklist.md`)
- YES → Proceed to Phase 3
- NO → Debug convergence issues, may need reparameterization

---

### Phase 3: Robustness Check (20 minutes)
```python
# Fit Model 3a and 3b
model3a = cmdstanpy.CmdStanModel(stan_file='model_3a_spec.stan')
fit3a = model3a.sample(data={'J': 8, 'y': y_obs, 'sigma': sigma_obs})

model3b = cmdstanpy.CmdStanModel(stan_file='model_3b_spec.stan')
fit3b = model3b.sample(data={'J': 8, 'y': y_obs, 'sigma': sigma_obs})

# Compute stacking weights
import arviz as az
weights = az.compare([fit3a, fit3b], method='stacking')
```

**Decision point:** Do models agree (|mu_skep - mu_enth| < 5)?
- YES → Inference is robust, DONE
- NO → Data insufficient to overcome priors, report high uncertainty

---

### Phase 4: Diagnostic Deep Dive (ONLY IF NEEDED, 60 minutes)
```python
# Fit Model 2 in PyMC
from model_2_spec import build_conflict_detection_model, sample_model
model2, data = build_conflict_detection_model()
trace2 = sample_model(model2)
```

**Decision point:** Are specific studies flagged as conflicts?
- YES → Refit Model 1 without flagged studies
- NO → Overall model class may be wrong, reconsider everything

---

## Falsification Criteria (When to Abandon Models)

### Abandon Model 1 if:
1. Posterior tau > 15 (EDA severely underestimated heterogeneity)
2. Pareto k > 0.7 for multiple studies (systematic outliers)
3. Posterior predictive checks fail (model can't reproduce data)
4. Study 4 removal changes estimate by >50% (too fragile)

### Abandon Model 2 if:
1. Conflict mechanism unused (all z_i = 0) - unnecessary complexity
2. Most studies flagged (pi_conflict > 0.5) - model overfitting
3. Non-convergence due to discrete parameters

### Abandon Model 3 if:
1. Models converge trivially (priors too weak to test)
2. Stacking weights unstable (change >0.3 with single study removal)
3. Agreement metric is arbitrary (threshold-dependent)

### Abandon ALL models if:
1. Prior predictive checks fail catastrophically
2. Study 4 drives everything (>100% influence)
3. Heterogeneity is actually high (I² > 50% in all models)
4. Data quality issues discovered

---

## Expected Outcomes (Based on EDA)

### Most Likely Scenario (80% probability):
- EDA is correct: I² = 2.9%, mu = 11.27, tau = 2.02
- Model 1 fits well, no diagnostics fail
- Model 3 shows convergence (skeptical and enthusiastic agree)
- Model 2 unnecessary
- **Conclusion:** Pooled effect ~11 with high confidence

### Possible Scenario (15% probability):
- Heterogeneity underestimated due to J=8
- Model 1 shows posterior tau ~ 5-10
- Model 3 shows moderate divergence
- Model 2 flags no specific studies but overall high uncertainty
- **Conclusion:** Effect estimate ~8-14, wider uncertainty than EDA suggests

### Concerning Scenario (5% probability):
- Study 4 is genuine outlier or from different population
- Model 1 shows Pareto k > 0.7 for Study 4
- Model 2 flags Study 4 with high probability
- Ensemble unstable (depends on Study 4)
- **Conclusion:** Exclude Study 4, report robust estimate without it

---

## Key Insights for Small Sample Meta-Analysis

1. **Priors matter with J=8** - Not just academic exercise, prior choice affects conclusions
2. **Heterogeneity estimates are uncertain** - I² has huge sampling variance with small J
3. **Individual studies can dominate** - Study 4 affects estimate by 33% (EDA finding)
4. **Robustness testing is essential** - Not optional, mandatory for credible inference
5. **Honesty about uncertainty** - If models disagree, data are insufficient

---

## Stress Tests (Adversarial Validation)

### Test 1: Extreme Prior Sensitivity
Refit Model 1 with mu ~ N(0, 1000) [nearly flat]
**Pass if:** Posterior changes by < 20%

### Test 2: Outlier Injection
Add synthetic outlier: y_9 = 100, sigma_9 = 10
**Pass if:** Pareto k_9 > 0.7 (detected)

### Test 3: Data Doubling
Duplicate all studies (J=16)
**Pass if:** Uncertainty decreases by ~1/sqrt(2)

### Test 4: Heterogeneity Injection
Simulate with tau=10 (not tau=2)
**Pass if:** Model recovers tau ~ 8-12

---

## Computational Requirements

| Model | Platform | Time | Cores | Memory |
|-------|----------|------|-------|--------|
| Prior checks | Python | 2 min | 1 | <1 GB |
| Model 1 | Stan | 1 min | 4 | <1 GB |
| Model 2 | PyMC | 5 min | 4 | 2 GB |
| Model 3a | Stan | 45 sec | 4 | <1 GB |
| Model 3b | Stan | 45 sec | 4 | <1 GB |

**Total time (all models):** ~10 minutes

---

## Deliverables After Fitting

1. **Comparison table** - Posterior estimates for all models
2. **Visualization suite** - Forest plots, prior-posterior, shrinkage
3. **Sensitivity analysis** - Leave-one-out, prior variations
4. **Model adequacy report** - All diagnostics with pass/fail
5. **Recommendation** - Which model to trust and why

---

## Decision Tree Summary

```
START
  |
  v
[Prior Predictive Checks]
  |
  |--FAIL--> Revise priors --> [Recheck]
  |
  |--PASS--> Fit Model 1
              |
              v
            [Diagnostics Pass?]
              |
              |--NO--> [Computational issue?]
              |         |--YES--> Reparameterize
              |         |--NO--> Check prior-posterior conflict
              |
              |--YES--> Fit Model 3 (Robustness)
                         |
                         v
                       [Models Agree?]
                         |
                         |--YES--> DONE (robust inference)
                         |
                         |--NO--> [Study-specific?]
                                   |--YES--> Fit Model 2
                                   |--NO--> Report high uncertainty
```

---

## Warning Signs (Red Flags)

1. **Prior-posterior conflict** - Overlap < 0.05 (prior was very wrong)
2. **Computational pathologies** - Divergences, low ESS, non-convergence
3. **Influential observations** - Pareto k > 0.7, >50% influence
4. **Poor predictions** - Posterior predictive checks fail systematically
5. **Instability** - Results change dramatically with small prior changes

**If multiple red flags:** STOP, reconsider entire modeling approach

---

## Next Steps

1. **Review proposal** - Is modeling strategy sound?
2. **Run prior checks** - Execute `prior_predictive_checks.py`
3. **Fit models** - Follow Phase 1-4 workflow
4. **Complete diagnostics** - Use `diagnostics_checklist.md`
5. **Report results** - Document findings with uncertainty quantification

---

## Contact / Questions

This is an independent design by Designer 3, focused on prior specification and model adequacy.

**Key philosophy:**
- Truth-seeking over task-completion
- Plan for failure, not just success
- Test assumptions adversarially
- Be ready to abandon all models if evidence warrants

**Remember:** A good Bayesian discovers their model was wrong early and pivots quickly. Success is finding truth, not confirming priors.

---

**Last updated:** 2025-10-28
**Status:** Ready for implementation
