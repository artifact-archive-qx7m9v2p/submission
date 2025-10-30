# Model Designer 3: Alternative Approaches
## Bayesian Models for Binomial Overdispersion

**Designer Focus**: Mixture models, robust specifications, assumption-challenging frameworks
**Philosophy**: Test fundamentally different data-generating processes
**Date**: 2025-10-30

---

## Overview

This directory contains proposals for **three alternative Bayesian model classes** that take different perspectives on the overdispersion problem (φ = 3.51, p < 0.001).

Unlike standard Beta-Binomial approaches, these models explicitly test:
1. **Discrete population structure** (Finite Mixture)
2. **Contamination processes** (Robust Contamination)
3. **Data-driven outlier detection** (Structured Outlier Detection)

---

## Files in This Directory

### 1. `proposed_models.md` (PRIMARY DOCUMENT)
**30 KB | Comprehensive model specifications**

Contains:
- Full mathematical specifications for all three models
- Stan/PyMC implementation code
- Theoretical justifications
- Falsification criteria for each model
- Implementation considerations
- Stress tests and robustness checks
- Decision framework for model selection
- Backup plans if all models fail

**Read this first** for complete understanding of the modeling strategy.

### 2. `falsification_checklist.md` (IMPLEMENTATION GUIDE)
**11 KB | Practical diagnostic checklist**

Contains:
- Quick reference for abandonment criteria
- Threshold tables for all metrics
- Diagnostic code snippets
- Decision tree flowchart
- Reporting templates
- Red flags for major pivots

**Use this during implementation** to systematically evaluate models.

### 3. `model_summary.txt` (QUICK REFERENCE)
**12 KB | ASCII art summary**

Contains:
- High-level model comparisons
- Unified decision framework
- Key principles and scenarios
- Implementation tools matrix

**Share this** for quick communication with collaborators.

### 4. `README.md` (THIS FILE)
**Navigation and context**

---

## Three Proposed Model Classes

### Model 1: Finite Mixture of Binomials
**Philosophy**: Trials come from K discrete populations

```
z_i ~ Categorical(π)              # K=2 or 3 latent classes
r_i | z_i ~ Binomial(n_i, p_{z_i})
p_1 < p_2 < p_3                   # Ordered for identifiability
```

**Key insight**: EDA tercile analysis suggests 3 groups (p ≈ 0.04, 0.07, 0.11)

**Abandon if**: Component separation < 0.6, overdispersion persists within clusters

**Implementation**: Stan (ordered parameters, marginal mixture likelihood)

### Model 2: Robust Contamination Model
**Philosophy**: Beta-Binomial + rare outlier process

```
δ_i ~ Bernoulli(λ)                # Contamination indicator
If δ_i=0: θ_i ~ Beta(α,β)         # Clean process
If δ_i=1: θ_i ~ Uniform(0,1)      # Outlier process
r_i ~ Binomial(n_i, θ_i)
```

**Key insight**: Trials 1 (0/47) and 8 (31/215) are suspicious

**Abandon if**: No outliers identified, or too many flagged (>30%)

**Implementation**: Stan (log_mix function for mixture)

### Model 3: Latent Beta-Binomial + Structured Outlier Detection
**Philosophy**: Learn outlier definition from residuals

```
θ_i ~ Beta(α,β)                   # Base heterogeneity
ψ_i = logit(η₀ + η₁|z_i|)         # Outlier probability (residual-based)
ω_i ~ Bernoulli(ψ_i)              # Outlier indicator
θ_eff = ω_i·θ*_i + (1-ω_i)·θ_i    # Mix base & outlier
```

**Key insight**: Data-driven outlier detection, not ad-hoc thresholds

**Abandon if**: Mechanism unnecessary (all ω < 0.2), computational instability

**Implementation**: PyMC (discrete latents, flexible modeling)

---

## Key Design Principles

### 1. Falsification Over Confirmation
- Every model has explicit "I will abandon this if..." criteria
- Focus on finding reasons to REJECT models
- Failure is success if it rules out wrong approaches

### 2. Computational Health = Model Health
- Divergences indicate misspecification
- Poor convergence means rethink the model
- Don't fight computational problems, listen to them

### 3. Simplicity When Tied
- If ΔWAIC < 2, prefer simpler model
- Don't overfit with N=12 observations
- Occam's razor applies

### 4. Honest Uncertainty
- If models disagree, report that
- Distinguish robust vs model-dependent conclusions
- Model ensemble if multiple viable

### 5. Prepared to Pivot
- Backup plans if all models fail
- Non-parametric alternatives ready
- Alternative likelihoods considered

---

## Implementation Workflow

### Phase 1: Initial Fitting
1. Implement all three models in Stan/PyMC
2. Fit Beta-Binomial baseline for comparison
3. Check computational health (R̂, ESS, divergences)
4. Generate posterior predictive datasets

### Phase 2: Falsification
1. Apply model-specific criteria (see checklist)
2. Eliminate models that fail
3. Document failure modes clearly

### Phase 3: Comparison
1. Compute WAIC/LOO for remaining models
2. Compare to Beta-Binomial baseline
3. If all ΔWAIC > -2, use baseline

### Phase 4: Stress Testing
1. Jackknife (remove trials 1, 4, 8)
2. Prior sensitivity analysis
3. Posterior predictive checks
4. Alternative likelihoods

### Phase 5: Selection
- 0 candidates → Beta-Binomial sufficient
- 1 candidate → Report with caveats
- 2+ candidates → Model ensemble or uncertainty

---

## Expected Scenarios

### Scenario A: Finite Mixture Wins
**Evidence**: Clear component separation, interpretable groups

**Interpretation**: Discrete population structure confirmed

**Next steps**: Report group-specific probabilities, investigate why groups differ

### Scenario B: Robust Contamination Wins
**Evidence**: 2-3 trials clearly flagged as outliers

**Interpretation**: Legitimate heterogeneity + some contamination

**Next steps**: Flag outlier trials, report clean population parameters

### Scenario C: All Alternatives Fail
**Evidence**: All ΔWAIC > -2, falsification criteria trigger

**Interpretation**: Beta-Binomial is sufficient

**PIVOT**: Accept simpler model, focus on prior specification

### Scenario D: Multiple Models Viable
**Evidence**: Similar WAIC, all pass tests

**Interpretation**: Fundamental DGP uncertainty

**Report**: Model ensemble or sensitivity analysis across models

---

## Falsification Metric Summary

| Model | Primary Metric | Threshold | Pass Indicates |
|-------|---------------|-----------|----------------|
| Finite Mixture | Component separation | > 0.60 | Clear discrete groups |
| Robust Contamination | Outlier coherence | Flags trials 1 or 8 | Contamination detected |
| Structured Outlier | max(ω) | > 0.20 | Mechanism necessary |
| **All Models** | ΔWAIC vs baseline | < -2 | Better than simple |

---

## Red Flags (Major Pivot Required)

1. **Data quality issues**: Measurement error, wrong units
2. **Likelihood misspecification**: Need different distribution entirely
3. **Exchangeability violated**: Temporal/spatial structure
4. **All models fail**: Need completely different approach

---

## Comparison to Other Designers

**Designer 1** (likely): Standard Beta-Binomial, hierarchical extensions

**Designer 2** (likely): Regression approaches, covariate modeling

**Designer 3** (this): **Alternative mechanisms** - mixtures, contamination, robust

**Synthesis**: Combine complementary strengths, identify robust conclusions

---

## Critical Success Metrics

This design succeeds if:

1. We **identify clear failure modes** for each model
2. We **document decisions** transparently
3. We **acknowledge uncertainty** honestly
4. We **challenge assumptions** rigorously
5. We **prioritize learning** over task completion

**Remember**: The best outcome is understanding which DGP features we CAN and CANNOT identify with 12 observations.

---

## Contact and Context

**Designer Role**: Model Designer 3 (Alternative Approaches)
**Main Agent**: Bayesian modeling strategist
**EDA Report**: `/workspace/eda/eda_report.md`
**Data**: `/workspace/data/data.csv`
**Synthesis**: Will be combined with other designers' proposals

---

## Usage Examples

### For Implementers

```bash
# Read the main document
cat /workspace/experiments/designer_3/proposed_models.md

# During implementation, use checklist
cat /workspace/experiments/designer_3/falsification_checklist.md

# Quick reference during meetings
cat /workspace/experiments/designer_3/model_summary.txt
```

### For Decision-Makers

1. Read `model_summary.txt` for high-level overview
2. Check `falsification_checklist.md` for objective criteria
3. Consult `proposed_models.md` for technical details

### For Synthesis

Compare:
- `/workspace/experiments/designer_1/` (if exists)
- `/workspace/experiments/designer_2/` (if exists)
- `/workspace/experiments/designer_3/` (this)

Look for:
- **Overlapping recommendations**: High confidence
- **Contradictory recommendations**: Dig deeper
- **Unique insights**: Test independently

---

## Version History

- **v1.0** (2025-10-30): Initial proposals
  - Three model classes specified
  - Falsification criteria defined
  - Implementation guidance provided

---

**Next Steps**: Implement models, run diagnostics, synthesize with other designers
