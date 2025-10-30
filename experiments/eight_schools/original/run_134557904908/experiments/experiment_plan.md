# Bayesian Model Experiment Plan

**Date**: 2025-10-28
**Project**: Meta-analysis with measurement error (J=8 observations)
**Synthesis**: Combined proposals from 3 independent model designers

---

## Executive Summary

Three independent designers proposed 9 model classes spanning classical meta-analysis (Designer 1), robust alternatives (Designer 2), and hierarchical structures (Designer 3). After removing duplicates and synthesizing recommendations, we have **5 distinct model classes** to implement in priority order.

**EDA Consensus**: All designers acknowledge:
- Strong homogeneity (I² = 0%, Cochran's Q p = 0.696)
- Pooled estimate: θ = 7.686 ± 4.072
- No outliers or publication bias detected
- Small sample size (J=8) limits power

**Strategic Approach**: Start with simplest model justified by EDA, then test robustness and validate homogeneity assumption through hierarchical alternatives.

---

## Proposed Model Classes (Priority Order)

### Model 1: Fixed-Effect Normal Model ⭐ BASELINE
**Designers**: 1 (primary), implicit baseline for 2 & 3
**Priority**: **HIGHEST** - must implement
**Complexity**: Simplest (1 parameter)

**Rationale**:
- Directly supported by EDA (I² = 0%, Q p = 0.696)
- Maximally efficient under homogeneity
- Analytical posterior available (conjugate)
- Baseline for all comparisons

**Specification**:
```
Likelihood: y_i ~ Normal(θ, σ_i²)  for i=1,...,8
Prior:      θ ~ Normal(0, 20²)
```

**Expected Results**:
- θ ≈ 7.7 ± 4.0
- Perfect convergence (R-hat = 1.000)
- Will serve as reference for LOO comparisons

**Falsification Criteria**:
- Systematic posterior predictive failures
- Evidence of heterogeneity in residual patterns
- Model inadequacy on LOO-PIT calibration

---

### Model 2: Bayesian Random-Effects (Hierarchical) Model ⭐ CRITICAL TEST
**Designers**: 1, 3 (both strongly recommended)
**Priority**: **HIGHEST** - must implement
**Complexity**: Medium (2 parameters + J latent effects)

**Rationale**:
- Tests whether EDA homogeneity finding holds in Bayesian framework
- Designer 3: "Small sample paradox - low power to detect moderate heterogeneity"
- Designer 1: "Hypothesis test of whether τ = 0"
- Partial pooling provides automatic regularization
- If τ → 0, collapses to Model 1 (adaptive)

**Specification**:
```
Likelihood: y_i ~ Normal(θ_i, σ_i²)
Hierarchy:  θ_i ~ Normal(μ, τ²)
Priors:     μ ~ Normal(0, 20²)
            τ ~ Half-Normal(0, 5²)  or  Half-Cauchy(0, 2.5)
```

**Implementation**: Non-centered parameterization to avoid funnel
```python
theta_raw ~ Normal(0, 1)
theta = mu + tau * theta_raw
```

**Expected Results**:
- μ ≈ 7.7 ± 4.0 (similar to θ in Model 1)
- τ ≈ 0 (confirming homogeneity)
- I² ≈ 0%
- Should perform similarly to Model 1 on LOO

**Falsification Criteria**:
- If τ substantially > 0 with high certainty → EDA missed heterogeneity
- If τ = 0 but worse LOO than Model 1 → unnecessary complexity
- If funnel pathology persists → switch to centered parameterization or abandon

---

### Model 3: Robust Fixed-Effect (Student-t) Model ⭐ ROBUSTNESS CHECK
**Designers**: 1, 2 (both recommended), 3 (variant)
**Priority**: **HIGH** - should implement
**Complexity**: Low (2 parameters)

**Rationale**:
- Designer 2: "Small sample vulnerability - single outlier could mislead"
- Designer 1: "Sensitivity analysis for outlier protection"
- Minimal complexity increase (one extra parameter: ν)
- Heavy tails automatically downweight outliers if present
- Converges to normal if ν > 30 (data decides)

**Specification**:
```
Likelihood: y_i ~ Student_t(ν, θ, σ_i²)
Priors:     θ ~ Normal(0, 20²)
            ν ~ Gamma(2, 0.1)  [mean=20, allows 5-60]
```

**Expected Results**:
- θ ≈ 7.7 ± 4.0 (similar to Model 1)
- ν > 20-30 (indicating normality adequate)
- Posterior slightly wider than Model 1 (robustness cost)
- Similar LOO to Model 1

**Falsification Criteria**:
- If ν < 5 consistently → serious non-normality, need mixture model
- If ν > 50 → Student-t unnecessary, use Model 1
- If θ substantially different from Model 1 → outlier detection working

**Designer Consensus**: "Best risk-benefit ratio for robustness"

---

### Model 4: Robust Hierarchical (Student-t Random Effects) [OPTIONAL]
**Designers**: 2, 3 (both suggested as synthesis)
**Priority**: **MEDIUM** - implement if Models 2 or 3 show issues
**Complexity**: High (3+ parameters)

**Rationale**:
- Combines hierarchical structure + heavy tails
- Only needed if BOTH heterogeneity AND non-normality detected
- Designer 2: "If mixture model suggests contamination"
- Designer 3: "Most flexible, but may be poorly identified"

**Specification**:
```
Likelihood: y_i ~ Student_t(ν, θ_i, σ_i²)
Hierarchy:  θ_i ~ Normal(μ, τ²)
Priors:     μ ~ Normal(0, 20²)
            τ ~ Half-Normal(0, 5²)
            ν ~ Gamma(2, 0.1)
```

**Implementation Strategy**:
- Only pursue if Models 2 & 3 individually inadequate
- Use non-centered parameterization
- Expect convergence challenges with J=8

**Expected Results**:
- Should collapse to simpler model (either τ→0 or ν→∞)
- If both remain active, serious model misspecification upstream

**Falsification Criteria**:
- Model overfits with worse LOO than simpler alternatives
- Computational instability (divergences, low ESS)
- Cannot identify both τ and ν simultaneously

**Stopping Rule**: If this model shows issues, reconsider problem framing

---

### Model 5: Contaminated Normal Mixture [DIAGNOSTIC TOOL]
**Designers**: 2 (priority 2)
**Priority**: **LOW** - only if outliers suspected
**Complexity**: High (4 parameters)

**Rationale**:
- Explicitly identifies problematic observations
- Designer 2: "Useful for diagnostic purposes"
- Two-component mixture: good data vs. contaminated
- More interpretable than Student-t for explaining outliers

**Specification**:
```
Likelihood: y_i ~ π·Normal(θ, σ_i²) + (1-π)·Normal(θ, λ²σ_i²)
Priors:     θ ~ Normal(0, 20²)
            π ~ Beta(9, 1)  [expect 90% good data]
            λ ~ Gamma(4, 1)  [variance inflation for bad data]
```

**Expected Results**:
- π > 0.95 (most data is good)
- If π < 0.6, majority contaminated → model misspecification

**Falsification Criteria**:
- π < 0.05 → no contamination, use simpler model
- All observations assigned similar probabilities → no clear separation
- Worse LOO than Student-t → use robust model instead

**Implementation Note**: Only pursue if:
1. Models 1-3 show systematic issues, OR
2. Posterior predictive checks reveal specific problematic observations

---

## Implementation Strategy

### Minimum Attempt Policy
**MUST implement at minimum**: Models 1 and 2
- These test the core scientific question (fixed vs. random effects)
- Model 1 is baseline, Model 2 tests homogeneity assumption
- Failure to implement both requires documentation in log.md

### Recommended Sequence
1. **Model 1** (Fixed-Effect Normal) - establish baseline
2. **Model 2** (Random-Effects) - test homogeneity hypothesis
3. **Model 3** (Robust Student-t) - robustness check

Stop and assess:
- If all three converge and agree → adequate solution
- If discrepancies → implement Model 4 or 5 as appropriate

### Decision Rules

**After Model 1**:
- ✅ If clean convergence + good PPC → proceed to Model 2
- ⚠️ If predictive failures → investigate before continuing

**After Model 2**:
- ✅ If τ → 0 → homogeneity confirmed, proceed to Model 3
- ⚠️ If τ > 0 substantially → reconsider, may need Model 4
- ❌ If computational issues → check parameterization

**After Model 3**:
- ✅ If ν > 20 and agrees with Model 1 → normality validated
- ⚠️ If ν < 10 → consider Model 5 for diagnostics
- ✅ If all three models agree → move to assessment phase

**Model 4** trigger: Both heterogeneity (τ>0) AND heavy tails (ν<10)

**Model 5** trigger: Specific observations flagged as outliers

---

## Validation Plan for Each Model

### Prior Predictive Checks
- Sample from prior: does θ span reasonable range (-40, 40)?
- Prior-data conflict: is data surprising under prior?
- Prior sensitivity: test alternative prior scales

### Simulation-Based Calibration (SBC)
- Simulate data with known θ
- Verify posterior recovers true θ within credible intervals
- Check for bias in point estimates
- Confirm correct uncertainty quantification

### Posterior Inference
- Convergence: R-hat < 1.01 for all parameters
- ESS: Bulk ESS > 400, Tail ESS > 400
- Divergences: Zero (or investigate carefully)
- Energy diagnostics: E-BFMI > 0.3

### Posterior Predictive Checks
- LOO-PIT histogram: uniform distribution?
- Posterior p-values: T(y^rep) vs T(y^obs) for multiple test statistics
- Graphical checks: overlay posterior predictive samples on data
- Residual analysis: patterns in (y - θ̂)?

### Model Comparison (if multiple models)
- LOO-CV: compare ELPD, check Pareto k diagnostics
- Stacking weights: which model gets highest weight?
- Posterior predictive performance: coverage of held-out data

---

## Falsification Framework

### Global Abandonment Criteria (exit Bayesian approach)
**NEVER ABANDON BAYESIAN FRAMEWORK** per guidelines.
If all models fail:
1. Return to EDA - is data quality issue?
2. Try simpler Bayesian model (e.g., just estimating mean with wide prior)
3. Consult with user about data generating process

### Model-Specific Abandonment (move to next model)
- Prior predictive check fails: prior-data conflict
- SBC fails: model cannot recover known parameters
- Convergence fails after multiple attempts: computational issues
- Posterior predictive failures: systematic misfit
- LOO diagnostics: Pareto k > 0.7 for many observations

### Success Criteria
A model is **ACCEPTED** if:
1. ✅ All convergence diagnostics pass
2. ✅ Posterior predictive checks show good calibration
3. ✅ LOO performance adequate (ELPD reasonable, Pareto k < 0.7)
4. ✅ Parameters have reasonable posteriors (not hitting boundaries)
5. ✅ Results interpretable and scientifically plausible

---

## Expected Outcomes & Synthesis

### Most Likely Scenario (80% probability)
Based on EDA and designer consensus:
- **Model 1** and **Model 2** converge to similar conclusions
- τ ≈ 0 in Model 2, confirming homogeneity
- **Model 3** shows ν > 20, confirming normality
- All models yield θ ≈ 7.7 ± 4.0
- LOO comparison shows Model 1 or 2 are best (within 2 SE)
- **Conclusion**: Fixed-effect model is adequate, data are homogeneous and normal

### Alternative Scenario 1: Heterogeneity Detected (15% probability)
- Model 2 shows τ > 5 with strong evidence
- Model 2 has better LOO than Model 1
- **Implication**: EDA's low power missed real heterogeneity
- **Action**: Report Model 2 as primary, investigate sources of heterogeneity

### Alternative Scenario 2: Outliers/Non-Normality (5% probability)
- Model 3 shows ν < 10
- Specific observations have poor posterior predictive fit
- **Action**: Implement Model 5 for diagnostics, report robust estimates

### Null Scenario: Model Inadequacy (< 1% probability)
- All models show systematic failures
- Large discrepancies between models
- Poor predictive performance across the board
- **Action**: Return to EDA, question data quality or problem framing

---

## Designer-Specific Insights

### Designer 1 (Classical Meta-Analysis)
- Strongest emphasis on parsimony and EDA alignment
- Recommended Models 1, 2, 3 in that order
- Provided closed-form posterior for Model 1 validation
- Emphasized Model 2 as hypothesis test, not default

### Designer 2 (Robust Models)
- Strongest emphasis on small-sample vulnerability
- Argued for Student-t as "default" over normal
- Proposed mixture model for diagnostics
- Most skeptical of EDA conclusions

### Designer 3 (Hierarchical Models)
- Strongest emphasis on "small sample paradox"
- Argued hierarchical structure even under homogeneity
- Most concerned about partial pooling benefits
- Proposed measurement error model (not prioritized here)

### Convergent Findings (High Confidence)
All three designers agreed:
- Model 1 (fixed-effect) is essential baseline
- Model 2 (random-effects) must be tested
- Model 3 (Student-t) is valuable robustness check
- Expected posterior for θ: 7-8 ± 4-5
- Expected τ ≈ 0 (but must verify)

### Divergent Emphasis
- Designer 1: Trust EDA, start simple
- Designer 2: Be skeptical, emphasize robustness
- Designer 3: Embrace hierarchy, partial pooling always beneficial

**Synthesis**: Implement Models 1-3, let data arbitrate between philosophies

---

## Model Class Summary Table

| Model | Parameters | Complexity | Priority | Expected τ | Expected ν | Expected θ | LOO Rank |
|-------|-----------|------------|----------|-----------|-----------|-----------|----------|
| 1. Fixed Normal | 1 (θ) | Simplest | ⭐ MUST | N/A | N/A | 7.7 ± 4.0 | 1-2 |
| 2. Random Effects | 2+J (μ,τ,θᵢ) | Medium | ⭐ MUST | ~0 | N/A | 7.7 ± 4.0 | 1-2 |
| 3. Robust (t) | 2 (θ,ν) | Low | ⭐ Should | N/A | 20-30 | 7.7 ± 4.0 | 1-3 |
| 4. Robust Hier. | 3+J (μ,τ,ν,θᵢ) | High | Maybe | ~0 | 20-30 | 7.7 ± 4.0 | 2-4 |
| 5. Mixture | 4 (θ,π,λ) | High | Diagnostic | N/A | N/A | 7.7 ± 4.0 | 3-5 |

---

## Next Steps

1. ✅ Create experiment directories for Models 1-3
2. → Implement Model 1: prior predictive → SBC → fit → posterior predictive → critique
3. → Implement Model 2: prior predictive → SBC → fit → posterior predictive → critique
4. → Implement Model 3: prior predictive → SBC → fit → posterior predictive → critique
5. → Model assessment: LOO comparison, calibration, final decision
6. → (Optional) Models 4-5 if needed based on assessment
7. → Final report with selected model and posterior inference

---

## Success Metrics

**Project succeeds if**:
- Minimum 2 models implemented and assessed (Models 1 & 2)
- All implemented models pass convergence diagnostics
- At least one model shows good posterior predictive performance
- Clear recommendation for best model based on LOO and scientific criteria
- Posterior for θ with quantified uncertainty
- All findings properly documented

**Project quality enhanced if**:
- All 3 priority models implemented
- Models converge on similar conclusions (validates robustness)
- Clear explanation of why preferred model is chosen
- Sensitivity analyses demonstrate stability
- Posterior captures EDA findings in probabilistic framework

---

**Document Status**: Ready for implementation
**Prepared by**: Main agent synthesis of 3 independent designers
**Approved for**: Phase 3 (Model Development Loop)
