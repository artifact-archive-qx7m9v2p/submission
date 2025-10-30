# Designer 3 Summary: Prior Specification Strategy
## Independent Bayesian Modeling Proposal

**Date:** 2025-10-28
**Focus:** Prior specification, model adequacy, robustness testing
**Status:** Complete and ready for implementation

---

## Executive Summary

I have designed **three fundamentally different Bayesian model classes** for the meta-analysis dataset (J=8 studies, I²=2.9%, pooled effect=11.27). Unlike typical "sensitivity analyses" that just vary parameter values, these models embody **different epistemological stances** toward small-sample meta-analysis.

### Key Insight
With only 8 studies, **prior choices critically affect inference**. Rather than pretending priors don't matter, my strategy explicitly tests robustness by:
1. Using weakly informative priors as baseline
2. Detecting prior-data conflicts explicitly
3. Testing opposing priors (skeptical vs enthusiastic)

### Critical Principle
**I will abandon these models if evidence suggests fundamental misspecification** - not just poor prior choice. Success is finding truth, not completing a predetermined plan.

---

## Three Model Classes Proposed

### Model 1: Weakly Informative Hierarchical (BASELINE)
**File:** `model_1_spec.stan`

**Mathematical Specification:**
```
y_i ~ Normal(theta_i, sigma_i)  for i=1,...,8
theta_i ~ Normal(mu, tau)
mu ~ Normal(0, 25)              # Weakly informative
tau ~ Half-Normal(0, 10)        # Standard meta-analysis prior
```

**When I abandon this:**
- Posterior tau > 15 (heterogeneity severely underestimated)
- Pareto k > 0.7 for multiple studies (systematic outliers)
- Posterior predictive checks fail
- Study 4 removal changes estimate by >50%

**Expected outcome:** Most likely to be primary analysis (fast, interpretable, standard)

---

### Model 2: Prior-Data Conflict Detection (DIAGNOSTIC)
**File:** `model_2_spec.py` (PyMC implementation)

**Mathematical Specification:**
```
mu ~ 0.5*N(0,25) + 0.5*N(11,8)  # Mixture: skeptical + optimistic
tau ~ 0.7*HalfNormal(0,5) + 0.3*HalfCauchy(0,10)  # Mixture
pi_conflict ~ Beta(1,1)         # Probability of conflict
z_i ~ Bernoulli(pi_conflict)    # Is study i in conflict?
sigma_adjusted = sigma * (1 + (inflation-1)*z_i)  # Inflate SE if conflict
```

**When I abandon this:**
- Conflict mechanism unused (all z_i = 0)
- Most studies flagged (pi_conflict > 0.5) - overfitting
- Non-convergence due to discrete parameters
- LOO-CV worse than Model 1 despite added complexity

**Expected outcome:** Use only if Model 1 shows influential outliers

---

### Model 3: Skeptical-Enthusiastic Ensemble (ROBUSTNESS)
**Files:** `model_3a_spec.stan` (skeptical), `model_3b_spec.stan` (enthusiastic)

**Mathematical Specification:**

**Model 3a (Skeptical):**
```
mu ~ Normal(0, 10)          # Strong prior: shrink toward null
tau ~ Half-Normal(0, 5)     # Expect low heterogeneity
```

**Model 3b (Enthusiastic):**
```
mu ~ Normal(15, 15)         # Optimistic: allow large effects
tau ~ Half-Cauchy(0, 10)    # Allow high heterogeneity
```

**Ensemble:** Combine via stacking weights (learned from cross-validation)

**When I abandon this:**
- Models converge trivially (priors too similar)
- Models diverge absurdly (|mu_skep - mu_enth| > 20)
- Stacking weights unstable (change >0.3 with single study removal)
- Agreement metric is arbitrary and uninformative

**Expected outcome:** Mandatory sensitivity check - if models agree, inference is robust

---

## Prioritization Recommendation

### Phase 1: Baseline (MUST FIT FIRST)
**Fit Model 1** - Fast (~1 minute), establishes baseline, standard approach

**Decision point:** If diagnostics pass → proceed to Phase 2

---

### Phase 2: Robustness (FIT SECOND)
**Fit Model 3 (both 3a and 3b)** - Tests prior sensitivity explicitly (~2 minutes)

**Decision point:**
- If models agree (|mu_skep - mu_enth| < 5) → DONE, inference is robust
- If models diverge → Proceed to Phase 3

---

### Phase 3: Diagnostic (FIT ONLY IF NEEDED)
**Fit Model 2** - Most complex (~5 minutes), use only if Phase 1-2 reveal issues

**Decision point:**
- If specific studies flagged → Refit Model 1 without them
- If overall high uncertainty → Report honestly, data insufficient

---

## Falsification Framework

### I Will Abandon ALL Models If:

1. **Prior predictive checks fail catastrophically**
   - Observed data in extreme 5% tails of prior predictive
   - Suggests fundamental model class is wrong

2. **Study 4 drives everything**
   - Removing Study 4 changes all posteriors by >100%
   - Pareto k > 1.0 for Study 4 in all models
   - **Pivot:** Abandon meta-analysis, report "insufficient data for pooling"

3. **Heterogeneity is genuinely large, not small**
   - All models converge to posterior I² > 50%
   - EDA's 3% was severe underestimate due to J=8
   - **Pivot:** Meta-regression to find moderators, or report "too heterogeneous"

4. **Posterior predictive checks fail for all models**
   - All three models show systematic misfit
   - Bayesian p-values < 0.01 for key statistics
   - **Pivot:** Try t-distribution (heavy tails), or non-exchangeable models

5. **Data quality issues discovered**
   - Extreme coincidences suggesting fabrication
   - Impossibly small SEs given known sample sizes
   - **Pivot:** Data quality investigation before any modeling

---

## Expected Findings (Based on EDA)

### Scenario A: EDA is Correct (80% probability)
**What happens:**
- Model 1 fits well, no diagnostics fail
- Posterior: mu ~ 11 ± 4, tau ~ 2 ± 2, I² ~ 5%
- Model 3: Skeptical and enthusiastic converge (agreement ~ 1)
- Model 2: Unnecessary

**Action:** Report Model 1 as primary, Model 3 as robustness check
**Conclusion:** Pooled effect ~11 with high confidence, strong shrinkage

---

### Scenario B: Heterogeneity Underestimated (15% probability)
**What happens:**
- Model 1: Posterior tau ~ 5-10 (wider than EDA)
- Model 3: Moderate divergence (|mu_skep - mu_enth| ~ 7-10)
- Model 2: High overall pi_conflict, but no specific studies flagged

**Action:** Report wider uncertainty than EDA suggests
**Conclusion:** Effect estimate ~8-14, emphasize uncertainty due to J=8

---

### Scenario C: Study 4 is Outlier (5% probability)
**What happens:**
- Model 1: Pareto k > 0.7 for Study 4
- Model 3: Divergence only in leave-one-out with Study 4
- Model 2: z_4 = 1 with high probability (Study 4 flagged)

**Action:** Refit all models without Study 4, investigate why
**Conclusion:** Report robust estimate excluding Study 4, investigate moderators

---

## Computational Specifications

| Model | Platform | Time | Chains | Iterations | Warmup | Divergences Expected |
|-------|----------|------|--------|------------|--------|---------------------|
| Model 1 | Stan | 1 min | 4 | 2000 | 1000 | 0 |
| Model 2 | PyMC | 5 min | 4 | 2000 | 2000 | <10 |
| Model 3a | Stan | 45 sec | 4 | 2000 | 1000 | 0 |
| Model 3b | Stan | 45 sec | 4 | 2000 | 1000 | 0 |

**Total runtime:** ~10 minutes for all models
**Memory:** <2 GB
**Parallelization:** 4 cores recommended

---

## Diagnostic Checklist (Abbreviated)

### Pre-Fitting (MUST PASS)
- [ ] Prior predictive checks: observed data in middle 50% of prior predictive
- [ ] Simulation-based calibration: SBC histogram is uniform

### Post-Fitting (MUST PASS)
- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 1000 for mu, tau
- [ ] Zero divergent transitions (or <1%)
- [ ] Posterior predictive checks: Bayesian p-values in [0.05, 0.95]
- [ ] LOO-CV: All Pareto k < 0.7

### Robustness (SHOULD PASS)
- [ ] Prior-posterior overlap in [0.1, 0.8] (learned but not conflict)
- [ ] Shrinkage matches theoretical (>95% expected)
- [ ] Influence analysis matches EDA (Study 4 ~33%, Study 5 ~23%)

**Full checklist:** See `diagnostics_checklist.md` (494 lines)

---

## Files Delivered

| File | Lines | Purpose |
|------|-------|---------|
| `proposed_models.md` | 1,021 | Main proposal with full mathematical specs |
| `diagnostics_checklist.md` | 494 | Systematic validation checklist |
| `README.md` | 318 | Quick start guide and workflow |
| `prior_predictive_checks.py` | 615 | Pre-data validation code |
| `model_1_spec.stan` | 59 | Stan code for baseline model |
| `model_2_spec.py` | 276 | PyMC code for conflict detection |
| `model_3a_spec.stan` | 62 | Stan code for skeptical model |
| `model_3b_spec.stan` | 62 | Stan code for enthusiastic model |

**Total:** 2,907 lines of documentation and code

---

## Key Contributions (Distinct from Other Designers)

### 1. Explicit Prior Sensitivity Testing
Unlike vague "sensitivity analyses", I propose **opposing priors** (skeptical vs enthusiastic) and test if they converge. If they do, inference is trustworthy. If not, we must report uncertainty honestly.

### 2. Prior-Data Conflict Detection
Rather than assuming prior and data share reality, Model 2 **explicitly tests for conflicts** via a mixture model with study-specific indicators. This is rare in meta-analysis.

### 3. Adversarial Validation
I design **stress tests** to break my own models:
- Outlier injection (should detect)
- Data doubling (uncertainty should scale correctly)
- Heterogeneity injection (should recover true tau)

### 4. Falsification Mindset
For each model, I specify **exactly when I will abandon it**. This is critical for truth-seeking (not task-completion).

### 5. Small-Sample Focus
With J=8, I emphasize:
- Heterogeneity estimates are very uncertain
- Individual studies can dominate (Study 4: 33% influence)
- Priors matter more than with J=50

---

## What Makes Me Abandon Everything?

### Condition 1: Prior Predictive Failure
If observed data is in extreme 5% tails of prior predictive for all models, the entire model class (normal hierarchical) is wrong.

**Pivot:** Try t-distributed errors, non-exchangeable models, or abandon pooling

---

### Condition 2: Study 4 Dominates Completely
If all models show >100% change when Study 4 removed AND Pareto k > 1.0, meta-analysis is inappropriate.

**Pivot:** Report individual study estimates, do not pool

---

### Condition 3: Heterogeneity is Actually High
If all models converge to I² > 50% (despite EDA showing 3%), small sample severely misled us.

**Pivot:** Meta-regression, investigate moderators, or report high heterogeneity

---

### Condition 4: Computational Pathologies Across All Models
If all models fail to converge despite reparameterization, priors, adaptation, the problem is fundamental.

**Pivot:** Simplify to common-effect model, or use empirical Bayes (REML)

---

## Final Recommendation for Implementation

### Primary Analysis
**Model 1** with extensive diagnostics from `diagnostics_checklist.md`

### Mandatory Sensitivity
**Model 3** (skeptical-enthusiastic ensemble) to test robustness

### Conditional Analysis
**Model 2** only if outliers detected in Phase 1-2

### Stress Tests
Run all 4 adversarial tests to validate model behavior

### Reporting
- Posterior mu, tau, I² with 95% credible intervals
- Prediction interval for future study
- LOO-CV diagnostics (especially Pareto k)
- Prior-posterior comparison plots
- Sensitivity to Study 4 and Study 5 removal
- Agreement metric from Model 3 ensemble

---

## Why This Strategy is Robust

1. **Tests assumptions explicitly** - Not hidden in defaults
2. **Plans for failure** - Knows when to pivot
3. **Adversarial validation** - Tries to break own models
4. **Honest uncertainty** - Reports when data are insufficient
5. **Computationally feasible** - Total runtime ~10 minutes

---

## Comparison to Alternative Approaches

### Why Not Just Use Vague Priors?
Vague priors with J=8 can lead to improper inference (posterior dominated by prior). Weakly informative priors constrain to scientifically plausible region.

### Why Not Just Use Frequentist Meta-Analysis?
Frequentist DerSimonian-Laird estimator can give negative tau² with small J. Bayesian approach handles this naturally via proper priors.

### Why Not Use Empirical Bayes?
Empirical Bayes treats tau as fixed at point estimate, ignoring uncertainty. With J=8, tau is very uncertain - full Bayes quantifies this.

### Why Three Models Instead of One?
With small sample, no single model is "correct". Testing robustness across fundamentally different models reveals whether conclusions depend on arbitrary choices.

---

## Success Criteria

### GREEN LIGHT (Primary Analysis Successful)
- All diagnostics pass for Model 1
- Model 3 shows agreement (|mu_skep - mu_enth| < 5)
- Results match EDA (mu ~ 11, tau ~ 2, I² ~ 3%)
- No influential outliers (all Pareto k < 0.7)

→ **Report Model 1 with high confidence**

---

### YELLOW LIGHT (Report with Caveats)
- Model 1 diagnostics pass but sensitivity analysis reveals fragility
- Model 3 shows moderate divergence (5 < |mu_skep - mu_enth| < 10)
- Study 4 or 5 moderately influential (Pareto k ~ 0.6-0.7)

→ **Report Model 1 but emphasize uncertainty and sensitivity**

---

### RED LIGHT (Major Issues)
- Diagnostics fail (non-convergence, poor posterior predictive)
- Model 3 shows severe divergence (|mu_skep - mu_enth| > 15)
- Multiple studies highly influential (Pareto k > 0.7)

→ **Pivot to alternative model class or report "data insufficient for pooling"**

---

## Conclusion

I have designed a **comprehensive, falsifiable, adversarially-validated** Bayesian modeling strategy that:

1. **Proposes three fundamentally different model classes** (not just parameter tweaks)
2. **Specifies exact falsification criteria** for each model
3. **Tests robustness explicitly** via opposing priors
4. **Plans for failure** with clear pivot points
5. **Emphasizes small-sample challenges** (J=8 is borderline for meta-analysis)

**Core philosophy:** Find truth, not complete tasks. Be ready to abandon all models if evidence warrants.

**Next steps:**
1. Run prior predictive checks (`prior_predictive_checks.py`)
2. Fit models in recommended order (1 → 3 → 2 if needed)
3. Complete diagnostic checklist
4. Report results with honest uncertainty quantification

---

**Deliverables Location:** `/workspace/experiments/designer_3/`

**Key Files:**
- Main proposal: `proposed_models.md`
- Quick start: `README.md`
- Diagnostics: `diagnostics_checklist.md`
- Validation: `prior_predictive_checks.py`
- Model code: `model_1_spec.stan`, `model_2_spec.py`, `model_3a_spec.stan`, `model_3b_spec.stan`

**Status:** Ready for implementation

---

**Designer 3, signing off.**
*Truth-seeking over task-completion. Always.*
