# Designer 1: Robust Hierarchical Models for Overdispersed Binomial Data

**Designer**: Model Designer 1 (Robust, Well-Established Approaches)
**Date**: 2025-10-30
**Status**: Model specifications complete, ready for implementation

---

## Overview

This directory contains complete specifications for three Bayesian hierarchical models designed for binomial data with strong overdispersion (φ=3.5-5.1) and heterogeneity (ICC=0.66).

**Data Context**: 12 groups, n=47-810 per group, r=0-46 events, pooled rate 7.39%

**Key Challenges**:
- Substantial overdispersion (variance 3.5-5× binomial expectation)
- High between-group heterogeneity (66% of variance between groups)
- Zero-event group (Group 1: 0/47)
- Three high-rate outliers (Groups 2, 8, 11)

---

## Proposed Models

### Model 1: Beta-Binomial Hierarchical (PRIMARY RECOMMENDATION)
- **Approach**: Direct overdispersion modeling via beta-distributed success probabilities
- **Strength**: Natural conjugate structure for binomial data, excellent zero-event handling
- **Implementation**: PyMC, ~2-5 minutes runtime
- **Priority**: **Implement first**

### Model 2: Random Effects Logistic Regression (ALTERNATIVE PRIMARY)
- **Approach**: Gaussian random effects on log-odds scale
- **Strength**: Standard approach, excellent computational properties, easy to extend
- **Implementation**: PyMC with non-centered parameterization, ~2-5 minutes runtime
- **Priority**: **Implement second for comparison**

### Model 3: Robust Logistic with Student-t Random Effects (BACKUP)
- **Approach**: Heavy-tailed random effects for outlier accommodation
- **Strength**: Adaptive shrinkage, robust to extreme groups
- **Implementation**: PyMC, ~5-10 minutes runtime
- **Priority**: **Only if Models 1-2 fail outlier accommodation**

---

## Files in This Directory

### Core Specifications
- **`proposed_models.md`** (26,000 words)
  - Complete mathematical specifications for all three models
  - Theoretical justifications
  - Prior specifications with rationale
  - Falsification criteria (what would make me reject each model)
  - Expected computational challenges
  - Full PyMC implementation code

- **`model_specifications_summary.md`** (3,500 words)
  - Quick reference for model specifications
  - Comparison table
  - Priority ordering
  - Expected outcomes

- **`implementation_roadmap.md`** (8,000 words)
  - Step-by-step implementation guide
  - Complete PyMC code for all phases
  - Prior and posterior predictive checks
  - Model comparison procedures
  - Troubleshooting guide

### This File
- **`README.md`** - Overview and navigation guide

---

## Decision Tree

```
START: Fit Models 1 and 2 in parallel
  │
  ├─→ Prior Predictive Checks
  │   ├─ PASS → Continue to fitting
  │   └─ FAIL → Adjust priors, retry
  │
  ├─→ MCMC Fitting (2000 draws × 4 chains each)
  │   ├─ Check convergence (Rhat, ESS, divergences)
  │   └─ If failures → Troubleshoot or reject
  │
  ├─→ Posterior Predictive Checks
  │   ├─ Coverage > 85%? → ACCEPT
  │   ├─ Coverage 70-85%? → PROVISIONAL ACCEPT
  │   └─ Coverage < 70%? → REJECT
  │
  ├─→ Apply Falsification Criteria
  │   ├─ Model 1: Check κ, φ, coverage, outliers
  │   └─ Model 2: Check τ, outlier fit, coverage
  │
  ├─→ DECISION POINT
  │   │
  │   ├─ BOTH ACCEPTED
  │   │   ├─→ LOO Cross-Validation
  │   │   ├─→ Compare posteriors
  │   │   └─→ FINAL: Choose Model 1 if similar
  │   │       (more natural for binomial overdispersion)
  │   │
  │   ├─ ONE ACCEPTED
  │   │   └─→ FINAL: Use the accepted model
  │   │
  │   └─ BOTH REJECTED or OUTLIER FIT POOR
  │       └─→ Implement Model 3 (Student-t)
  │           ├─→ Fit and check posterior ν
  │           ├─→ If ν > 30: Use Model 2 (heavy tails unnecessary)
  │           └─→ If ν < 30: Use Model 3 (heavy tails justified)
  │
  └─→ FINAL OUTPUT
      ├─ Selected model
      ├─ Posterior summaries for all groups
      ├─ Uncertainty quantification
      └─ Model diagnostics and validation
```

---

## Quick Start

1. **Read the full specifications**: Start with `proposed_models.md`
2. **Review the summary**: Quick reference in `model_specifications_summary.md`
3. **Follow implementation guide**: Step-by-step code in `implementation_roadmap.md`
4. **Implement Models 1 and 2**: Run in parallel
5. **Evaluate**: Apply falsification criteria
6. **Select final model**: Based on posterior predictive performance

---

## Key Design Principles

### 1. Falsification Mindset
Every model has explicit rejection criteria:
- "I will abandon Model 1 if κ → 0 or κ → ∞"
- "I will abandon Model 2 if τ > 2.0 or outliers systematically misfit"
- "I will abandon Model 3 if ν > 30 (use Model 2 instead)"

### 2. Competing Hypotheses
Three fundamentally different variance structures:
- **Beta** (Model 1): Continuous probability distribution
- **Gaussian** (Model 2): Symmetric, light-tailed
- **Student-t** (Model 3): Symmetric, heavy-tailed

### 3. Built-In Checkpoints
- Prior predictive checks before fitting
- Convergence diagnostics during fitting
- Posterior predictive checks after fitting
- Falsification criteria for rejection decisions
- Model comparison if multiple accepted

### 4. Escape Routes
- Model 1 fails → Try Model 2
- Model 2 fails → Try Model 3
- All fail → Consider mixture model, investigate data

### 5. Scientific Plausibility
- All models appropriate for overdispersed binomial data
- Weakly informative priors allowing data to dominate
- Zero-event and outlier groups handled naturally via partial pooling
- No ad-hoc adjustments or corrections

---

## Expected Outcomes

### Most Likely (90% confidence)
- **Models 1 and 2 both adequate**
- Similar substantive conclusions
- Model 1 selected (more direct for overdispersion)
- Model 3 unnecessary

### Specific Predictions
- **Model 1**: κ ≈ 0.5-1.0, φ ≈ 4.0, ICC ≈ 0.65
- **Model 2**: τ ≈ 0.7-1.0, similar p_i posteriors to Model 1
- **Zero-event group** (Group 1): posterior mean ≈ 3-5%
- **Outliers** (Groups 2, 8, 11): appropriate shrinkage toward population mean
- **Coverage**: >85% for both models

### Less Likely (10% confidence)
- One model clearly superior to other
- Model 3 needed for outlier accommodation
- All models fail (would trigger mixture model consideration)

---

## Success Metrics

### Computational Success
- Rhat < 1.01 for all parameters
- ESS > 400 per chain (>200 acceptable for hyperparameters)
- Divergences < 1% of post-warmup samples
- Runtime < 10 minutes per model

### Statistical Success
- Posterior predictive coverage > 85%
- Posterior φ overlaps observed range (3.5-5.1)
- Zero-event group (Group 1) within 95% posterior predictive interval
- Outliers (Groups 2, 8, 11) adequately fit
- No systematic bias in residuals

### Scientific Success
- Posteriors scientifically plausible (rates 0.1-30%)
- Partial pooling working appropriately
- Uncertainty properly quantified
- Model provides actionable insights

---

## Model Comparison Summary

| Feature | Beta-Binomial | Logistic GLMM | Robust Logistic |
|---------|--------------|---------------|-----------------|
| **Overdispersion** | Direct | Indirect | Indirect |
| **Scale** | Probability | Log-odds | Log-odds |
| **Outlier Handling** | Moderate | Moderate | Excellent |
| **Zero-Event Handling** | Excellent | Good | Good |
| **Computation** | Fast (2-5 min) | Fast (2-5 min) | Moderate (5-10 min) |
| **Extensibility** | Limited | High | High |
| **Interpretability** | High | High | Moderate |
| **Priority** | 1st | 2nd | 3rd (conditional) |

---

## Falsification Criteria Quick Reference

### Model 1 (Beta-Binomial)
**REJECT if**:
- κ → 0 or κ → ∞ (boundary behavior)
- Coverage < 70%
- Posterior φ far from observed (3.5-5.1)
- Persistent computational failures

### Model 2 (Logistic GLMM)
**REJECT if**:
- τ > 2.0 (extreme between-group variance)
- Outliers consistently outside 95% intervals
- Coverage < 70%
- Computational failures despite non-centered parameterization

### Model 3 (Robust Logistic)
**REJECT if**:
- ν > 30 (use Model 2 instead)
- ν → 2 (hitting constraint, too extreme)
- Still poor fit despite heavy tails
- Persistent computational failures (>2% divergences)

---

## Red Flags That Would Trigger Model Class Changes

1. **All models show poor coverage (<70%)** → Consider mixture model
2. **Overdispersion can't be matched (φ << observed)** → Reconsider likelihood structure
3. **Zero-event group extremely pathological** → Investigate data quality
4. **Outliers consistently misfit all models** → Two subpopulations? Mixture model
5. **Computational failures across all approaches** → Fundamental model misspecification
6. **Posteriors conflict with domain knowledge** → Consult experts, check data

---

## Implementation Timeline

- **Setup and data prep**: 5 minutes
- **Model 1 (full workflow)**: 30 minutes
- **Model 2 (full workflow)**: 25 minutes
- **Model comparison**: 15 minutes
- **Model 3 (if needed)**: 40 minutes
- **Documentation**: 10 minutes

**Total Expected Time**: 1.5 hours (without Model 3), 2 hours (with Model 3)

---

## Output Files (Generated During Implementation)

- `prior_predictive_model1.png` - Prior checks for Model 1
- `prior_predictive_model2.png` - Prior checks for Model 2
- `posterior_checks_model1.png` - Posterior analysis for Model 1
- `posterior_checks_model2.png` - Posterior analysis for Model 2
- `trace_model1.nc` - MCMC samples for Model 1
- `trace_model2.nc` - MCMC samples for Model 2
- `results_summary.txt` - Final decision and recommendations

---

## Contact and Support

**Designer**: Model Designer 1
**Focus**: Robust, well-established hierarchical approaches
**Philosophy**: Falsification mindset, competing hypotheses, clear escape routes

**Key Insight**: Success is discovering when models fail early and pivoting quickly, not completing a predetermined plan. These models are designed to fail gracefully and provide clear signals when alternative approaches are needed.

---

## References and Theory

### Beta-Binomial Models
- Conjugate structure for binomial data
- Canonical model for overdispersed counts
- ICC = 1/(κ+1) directly interpretable
- Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. Biometrics, 949-952.

### Random Effects Logistic Regression
- Standard GLMM approach
- Non-centered parameterization essential for efficiency
- Betancourt, M., & Girolami, M. (2015). Hamiltonian Monte Carlo for hierarchical models. Current trends in Bayesian methodology with applications, 79, 30.

### Student-t Random Effects
- Robust to outliers via heavy tails
- Adaptive shrinkage based on outlierness
- Juarez, M. A., & Steel, M. F. (2010). Model-based clustering of non-Gaussian panel data based on skew-t distributions. Journal of Business & Economic Statistics, 28(1), 52-66.

---

## Final Note

These models represent robust, well-established approaches for overdispersed binomial data. The design emphasizes:
- **Scientific plausibility** over methodological novelty
- **Falsification** over confirmation
- **Practical implementation** over theoretical elegance
- **Clear decision rules** over ambiguous criteria

All models are fully specified, implementable in PyMC, and designed to handle the specific challenges of this dataset (overdispersion, heterogeneity, zero-events, outliers).

**Expected final model**: Model 1 (Beta-Binomial Hierarchical)

Ready for implementation.
