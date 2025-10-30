# Model Designer #3: Robust & Alternative Formulations
## Bayesian Meta-Analysis Model Proposals

**Designer Focus**: Robustness to outliers, distributional assumptions, and model misspecification
**Date**: 2025-10-28
**Status**: Ready for implementation

---

## Quick Overview

This designer proposes **3 distinct Bayesian model classes** that address potential issues with the standard meta-analysis approach:

1. **Student-t Robust Meta-Analysis** - Heavy-tailed likelihood for outlier robustness
2. **Finite Mixture Meta-Analysis** - Explicit modeling of subgroup structure
3. **Uncertainty-Inflated Meta-Analysis** - Accounts for potentially underestimated standard errors

Each model has:
- Full mathematical specification
- Prior justifications with references
- Explicit falsification criteria
- Implementation code (Stan)
- Expected challenges documented

---

## Files in This Directory

### Core Documents
- **`proposed_models.md`** (25K words) - Complete model specifications with mathematical details, priors, falsification criteria, implementation notes
- **`falsification_summary.md`** - Quick reference for when to abandon each model
- **`implementation_guide.md`** - Step-by-step Python/Stan code for fitting and evaluating
- **`README.md`** (this file) - Overview and navigation

### Directory Structure (to be created during implementation)
```
/workspace/experiments/designer_3/
├── proposed_models.md          # Main design document
├── falsification_summary.md    # Quick falsification reference
├── implementation_guide.md     # Coding guide
├── README.md                   # This file
├── models/                     # Stan code (to be created)
│   ├── model_0_standard.stan
│   ├── model_1_student_t.stan
│   ├── model_2_mixture.stan
│   └── model_3_inflation.stan
├── fits/                       # MCMC output (to be created)
├── diagnostics/                # Diagnostic plots (to be created)
└── results/                    # Final reports (to be created)
```

---

## Three Models at a Glance

### Model 1: Robust Student-t Meta-Analysis (RSTMA)
**Core idea**: Use Student-t likelihood instead of Normal to automatically downweight potential outliers

**Key equation**:
```
y_i ~ Student-t(nu, theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
nu ~ Gamma(2, 0.1)  # Learns tail heaviness from data
```

**When to use**: Study 1 (y=28) is influential; concerned about distributional assumptions

**Abandon if**: nu > 50 (converges to Normal, unnecessary complexity)

**Implementation priority**: HIGH (most robust, moderate complexity)

---

### Model 2: Finite Mixture Meta-Analysis (TMMA)
**Core idea**: Explicitly model two latent groups of studies with different effect sizes

**Key equation**:
```
theta_i ~ pi * Normal(mu_2, tau_2) + (1-pi) * Normal(mu_1, tau_1)
pi ~ Beta(2, 2)  # Mixing proportion
```

**When to use**: EDA showed clustering (p=0.009); suspicion of distinct subgroups

**Abandon if**: Groups collapse (pi extreme) or not separated (|mu_2-mu_1| < 5)

**Implementation priority**: MEDIUM (complex, may not identify with J=8)

---

### Model 3: Uncertainty-Inflated Meta-Analysis (UIMA)
**Core idea**: Allow reported standard errors to be systematically underestimated

**Key equation**:
```
y_i ~ Normal(theta_i, sigma_i * lambda)
lambda ~ Log-Normal(0, 0.5)  # Inflation factor (median=1)
```

**When to use**: Concerned about SE quality; want conservative inference

**Abandon if**: lambda ≈ 1 (no inflation needed, standard model sufficient)

**Implementation priority**: MEDIUM (easy to implement, may not add value)

---

## Key Findings from EDA That Motivated These Models

1. **I²=0% but wide range** (-3 to 28): "Heterogeneity paradox" - may reflect low power
2. **Study 1 highly influential**: Removing it changes pooled estimate by 2.20 points
3. **Borderline significance**: Pooled p≈0.05, sensitive to modeling choices
4. **Potential clustering**: High-effect vs low-effect groups (EDA p=0.009)
5. **Small sample**: J=8 limits power for heterogeneity detection
6. **Large measurement errors**: sigma = 9-18, may obscure true patterns

**Standard approach risks**: Assuming normality, treating sigma as known exactly, ignoring potential substructure

---

## Philosophy: Design for Falsification

**Critical principle**: These models are designed to **fail informatively**

Each model has explicit criteria that would make us abandon it:
- Student-t: If nu > 50, revert to Normal
- Mixture: If groups collapse, use single population
- Inflation: If lambda ≈ 1, fix sigma as known

**Success is not completing all analyses** - success is finding the model that genuinely explains the data, even if that means abandoning complex models for simpler ones.

---

## Implementation Roadmap

### Phase 1: Baseline (30 minutes)
- Fit standard Normal hierarchical model
- Purpose: Benchmark for comparison

### Phase 2: Student-t (1-2 hours)
- **Priority 1** - Most likely to be useful
- Robust to Study 1 influence
- Learns tail behavior from data

### Phase 3: Uncertainty Inflation (1 hour)
- **Priority 2** - Easy to implement
- May not add much but good robustness check

### Phase 4: Mixture (2-3 hours)
- **Priority 3** - Most complex
- May not identify with J=8
- Only if evidence for clustering persists

### Phase 5: Comparison & Sensitivity (2-3 hours)
- LOO-CV comparison
- Leave-one-out sensitivity (especially Study 1)
- Prior sensitivity
- Final report

**Total time**: 6-10 hours for complete analysis

**Minimum viable**: Models 0, 1, 3 + LOO + PPC = ~4 hours

---

## Model Comparison Strategy

### Use LOO-CV (Leave-One-Out Cross-Validation)
- Primary metric: `elpd` (expected log predictive density)
- Decision rule: If |elpd_diff| < 2*SE, models equivalent → choose simpler
- Watch Pareto k diagnostic: k > 0.7 indicates influential points

### Posterior Predictive Checks
- Test statistic: Coverage in 95% intervals
- Pass criterion: >80% of studies within intervals
- If all models fail PPC → need different model class

### Convergence Requirements
- R-hat < 1.01 for all parameters
- ESS_bulk > 400 for mu, tau, and model-specific params
- Divergent transitions < 10 after tuning
- If any fail → don't trust results

---

## Expected Outcomes

### Scenario A: All models agree on mu
- **Interpretation**: Effect estimate robust to modeling choices
- **Action**: Report consensus, use simplest model
- **Most likely**: Standard or Student-t with nu > 30

### Scenario B: Student-t finds nu < 30
- **Interpretation**: Heavy tails important, Study 1 genuinely anomalous
- **Action**: Use Student-t results as primary
- **Implication**: Distributional assumptions matter

### Scenario C: Mixture finds clear groups
- **Interpretation**: Subgroup structure real, I²=0% misleading
- **Action**: Report mixture results, investigate group characteristics
- **Challenge**: Without covariates, can't explain what defines groups

### Scenario D: Inflation finds lambda >> 1
- **Interpretation**: Reported SEs systematically underestimated
- **Action**: Report inflated uncertainty, raise data quality concerns
- **Implication**: Primary studies may have methodological issues

### Scenario E: No model fits well
- **Interpretation**: Structural misspecification beyond these classes
- **Action**: Pivot to alternative approach (state-space, GP, non-parametric)
- **This is okay**: Better to discover model inadequacy than report wrong results

### Scenario F: Wide posteriors, high uncertainty
- **Interpretation**: J=8 insufficient for confident inference
- **Action**: Report high uncertainty, recommend more studies
- **This is the honest answer if data is weak**

---

## Key References

**Robust meta-analysis**:
- Baker & Jackson (2008) - Outliers in meta-analysis
- Lee & Thompson (2008) - Flexible parametric models

**Mixture models**:
- Frühwirth-Schnatter (2006) - Finite Mixture and Markov Switching Models
- Beath (2012) - Application to meta-analysis

**Measurement error**:
- Turner et al. (2015) - Predictive distributions for heterogeneity
- Riley et al. (2010) - Individual participant data meta-analysis

**Bayesian meta-analysis**:
- Gelman (2006) - Prior distributions for variance parameters (THE classic reference)
- Higgins et al. (2002) - Quantifying heterogeneity
- Röver (2020) - bayesmeta R package

**Stan implementation**:
- Betancourt (2017) - Diagnosing biased inference with divergences
- Stan User's Guide v2.32

---

## Diagnostic Checklist

Before trusting any model:

**Convergence**:
- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 for key parameters
- [ ] Divergences < 10
- [ ] Trace plots show good mixing

**Validation**:
- [ ] Prior predictive check: Reasonable simulated data
- [ ] Posterior predictive check: >80% coverage in 95% intervals
- [ ] LOO Pareto k < 0.7 for >75% studies
- [ ] Parameter recovery (if tested on synthetic data)

**Falsification**:
- [ ] Check model-specific abandonment criteria
- [ ] Leave-one-out sensitivity (especially Study 1)
- [ ] Prior sensitivity (especially tau)
- [ ] Compare to EDA findings (rough consistency expected)

**Reporting**:
- [ ] Posterior summaries (median + 95% CI)
- [ ] Probability statements: P(mu > 0 | data)
- [ ] Study-specific shrinkage estimates
- [ ] Convergence diagnostics reported
- [ ] Model comparison results (LOO)
- [ ] Sensitivity analyses documented

---

## What Makes These Models "Robust"?

1. **Student-t**: Automatically downweights extreme observations
   - Adapts to tail behavior in data
   - Nests Normal model (nu→∞)
   - Robust to outliers without arbitrary exclusion

2. **Mixture**: Explicitly models heterogeneity structure
   - Separates within-group from between-group variation
   - Avoids forcing all studies into single distribution
   - Can detect publication bias mechanisms

3. **Inflation**: Conservative uncertainty quantification
   - Accounts for SE estimation error
   - Widens credible intervals appropriately
   - Robust to optimistic SE reporting

All three are more conservative than standard Normal model, providing guard rails against overconfident inference from J=8 studies.

---

## Critical Warnings

1. **Small sample (J=8)**: All models may have identification issues
   - Expect wide posteriors
   - Prior sensitivity important
   - May not distinguish complex model components

2. **Mixture with J=8**: High risk of non-identification
   - 6 parameters (mu_1, mu_2, tau_1, tau_2, pi, theta_i)
   - May see label switching despite ordering constraint
   - Be ready to abandon if not converging

3. **Borderline significance (p≈0.05)**: Results may be sensitive
   - Small changes in model can flip conclusions
   - Report uncertainty honestly
   - Probability statements better than p-values

4. **Influential Study 1**: All models sensitive to it
   - Leave-one-out analysis essential
   - Don't over-interpret if driven by single study
   - Consider reporting with/without Study 1

5. **No covariates**: Can't explain heterogeneity sources
   - If mixture finds groups, can't say what defines them
   - Limited scientific interpretation
   - Need external data to understand structure

---

## Next Steps

1. **Read documents in order**:
   - Start: `proposed_models.md` (full specifications)
   - Reference: `falsification_summary.md` (quick criteria)
   - Implement: `implementation_guide.md` (step-by-step code)

2. **Set up environment**:
   - Install cmdstanpy (or pymc)
   - Install arviz for diagnostics
   - Create directory structure

3. **Implement models**:
   - Start with Model 0 (standard) + Model 1 (Student-t)
   - Add Model 3 if time permits
   - Add Model 2 only if clear clustering evidence

4. **Evaluate rigorously**:
   - Check convergence BEFORE trusting results
   - Run falsification checks
   - Compare models via LOO-CV
   - Do sensitivity analyses

5. **Report honestly**:
   - If models don't converge: say so
   - If conclusions uncertain: say so
   - If data insufficient: say so
   - Success = truth-finding, not task-completion

---

## Contact/Questions

This is an independent proposal from Model Designer #3. It will be synthesized with proposals from other designers (if any) by the main orchestrator agent.

**Design philosophy**: Robust methods that fail informatively
**Key innovation**: Explicit falsification criteria for each model
**Primary concern**: J=8 is small; be conservative, report uncertainty honestly

**Remember**: Abandoning a model because it failed falsification checks is SUCCESS, not failure. It means we learned something about the data.

---

**Status**: Design complete, ready for implementation
**Estimated implementation time**: 6-10 hours for full analysis
**Minimum viable analysis**: 4 hours (Models 0, 1, 3 + comparison)
**Expected output**: Model comparison report with honest uncertainty quantification

Good luck finding the truth in this data!
