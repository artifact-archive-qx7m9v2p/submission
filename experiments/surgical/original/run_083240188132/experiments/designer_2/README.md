# Model Designer 2: Alternative Bayesian Specifications

**Designer**: Model Designer 2 (Alternative Perspectives Track)
**Date**: 2025-10-30
**Status**: Design Complete

---

## Overview

This directory contains three alternative Bayesian model specifications that offer different structural perspectives on the binomial outcome data, moving beyond standard beta-binomial hierarchical approaches.

**Data context**:
- 12 groups, n=47-810 per group
- Strong heterogeneity (ICC=0.66)
- Substantial overdispersion (φ=3.5-5.1)
- Three high outliers (Groups 2, 8, 11)
- Possible bimodal distribution

---

## Files in This Directory

### 1. `proposed_models.md` (Main Document)
**38KB, comprehensive specifications**

Complete model specifications including:
- Mathematical formulation (likelihood, priors)
- Theoretical justification
- How each addresses key challenges (heterogeneity, overdispersion, outliers)
- Unique advantages of each approach
- Implementation plans (PyMC code structure)
- Falsification criteria (when to abandon each model)
- Comparison to standard hierarchical approaches
- Prior/posterior predictive check protocols
- Model comparison strategies
- Scientific plausibility checks

**Read this for**: Full understanding of each model's motivation, structure, and evaluation strategy.

### 2. `quick_reference.md` (Quick Summary)
**3KB, essential information**

One-page summary of:
- Core idea of each model
- Key parameters
- Falsification criteria
- When each is best
- Implementation priority
- Comparison matrix

**Read this for**: Quick lookup during model implementation and comparison.

### 3. `implementation_templates.py` (Ready-to-Use Code)
**20KB, fully functional PyMC implementations**

Ready-to-run Python code including:
- Three complete model-building functions
- Specialized fitting functions with appropriate MCMC settings
- Prior/posterior predictive check utilities
- Falsification check functions
- Example usage workflow

**Use this for**: Direct implementation of proposed models.

### 4. `README.md` (This File)
Navigation guide and overview.

---

## Three Proposed Models

### Model 1: Finite Mixture (K=2 Components)
**Structural assumption**: Discrete subpopulations

- Groups come from two distinct risk categories (low ~6%, high ~12%)
- Component membership is a latent variable
- Mixture weight determines proportion in each component
- **Best when**: Evidence of bimodality, interest in subgroup identification
- **Falsify if**: w < 0.1 or > 0.9, components too close (mu_diff < 0.03)

### Model 2: Robust Hierarchical (Student-t Random Effects)
**Structural assumption**: Heavy-tailed continuous distribution

- Single population with fat tails accommodating outliers
- Degrees of freedom (nu) controls tail weight
- Less aggressive shrinkage for extreme groups
- **Best when**: Outliers are legitimate, want robust estimates
- **Falsify if**: nu > 30 (normal sufficient) or nu < 2 (too extreme)

### Model 3: Dirichlet Process Mixture
**Structural assumption**: Nonparametric, flexible clustering

- Unknown number of latent subpopulations
- Data determines effective cluster count (K_eff)
- Maximum flexibility via Bayesian nonparametrics
- **Best when**: Genuinely uncertain about cluster structure
- **Falsify if**: K_eff = 1 consistently, computational issues

---

## Implementation Priority

**Recommended sequence**:

1. **Start with Model 2 (Robust Student-t)**:
   - Lowest complexity of the three
   - Fastest to implement and fit
   - Directly tests whether heavy tails explain outliers
   - If nu > 30: Revert to standard normal random effects
   - If nu < 10: Consider Model 1 (mixture)

2. **Then Model 1 (Finite Mixture)**:
   - Moderate complexity
   - Tests discrete subpopulation hypothesis
   - Interpretable cluster membership
   - Compare to Model 2 via LOO

3. **Finally Model 3 (DP) only if**:
   - Uncertain whether K=1, 2, or 3 clusters
   - Computational resources available (most expensive)
   - Scientific interest in flexible clustering

---

## Key Principles

### Falsification Mindset
Each model has clear falsification criteria. **Abandoning a model when evidence contradicts it is success, not failure**.

### Global Stopping Rules
Revert to standard hierarchical beta-binomial if:
- All three alternatives fail their specific criteria
- All show worse LOO than baseline
- All have computational difficulties
- Prior predictives are implausible

### Comparison to Standard Approaches
All models must be compared to standard beta-binomial hierarchical:
- Only report complex model if ΔLOO > 4 (clearly better)
- If ΔLOO < 2*SE, models are indistinguishable
- Always prefer simpler model when performance is equal (Occam's razor)

---

## Usage Workflow

### Step 1: Read EDA Findings
- Location: `/workspace/eda/eda_report.md`
- Key findings: heterogeneity, overdispersion, outliers, possible bimodality

### Step 2: Review Model Specifications
- Read: `proposed_models.md` (full details)
- Skim: `quick_reference.md` (quick lookup)

### Step 3: Prior Predictive Checks
```python
from implementation_templates import *
import pandas as pd

data = pd.read_csv('/workspace/data/data.csv')
n, r = data['n'].values, data['r'].values

# Build model
model = build_robust_hierarchical_model(n, r)

# Check prior
prior = prior_predictive_check(model, n_samples=1000)

# Visualize prior predictive (ensure it's reasonable)
```

### Step 4: Fit Model
```python
trace = fit_robust_model(model)

# Check convergence
# - Rhat < 1.01
# - ESS > 400 per chain
# - No divergences
```

### Step 5: Posterior Checks
```python
# Posterior predictive
trace = posterior_predictive_check(model, trace)

# Falsification criteria
falsification = check_robust_falsification(trace)
if falsification['falsified']:
    print(f"Model falsified: {falsification['reason']}")
```

### Step 6: Model Comparison
```python
import arviz as az

# LOO cross-validation
loo = az.loo(trace)

# Compare multiple models
az.compare({'robust': trace_robust,
            'mixture': trace_mixture})
```

---

## Falsification Summary

### Model 1 (Finite Mixture)
**Abandon if**:
- Mixture weight extreme (w < 0.1 or w > 0.9)
- Components not distinct (mu_diff < 0.03)
- LOO worse than unimodal model
- Component assignments unstable

### Model 2 (Robust Student-t)
**Abandon if**:
- Posterior nu > 30 (normal sufficient)
- Posterior nu < 2 (too extreme)
- No improvement over normal random effects
- Divergent transitions persist

### Model 3 (DP Mixture)
**Abandon if**:
- K_eff = 1 consistently (no clustering)
- Extreme computational difficulties
- Clusters not interpretable
- LOO worse than simpler models

---

## Deliverables Per Model

For each model that passes initial assessment:
1. Prior predictive check report
2. Convergence diagnostics
3. Posterior summary statistics
4. Posterior predictive check report
5. Falsification assessment
6. LOO comparison (if multiple models accepted)
7. Scientific interpretation

---

## Expected Outcomes

### Success Case 1: One Model Clearly Best
- ΔLOO > 4 favoring one model
- That model passes falsification criteria
- Posterior predictives reproduce data well
- **Action**: Report this model as final

### Success Case 2: Multiple Models Viable
- ΔLOO < 2*SE (indistinguishable)
- Multiple models pass falsification
- **Action**: Report all, explain what each reveals

### Success Case 3: All Models Fail
- All fail falsification criteria
- All worse than standard hierarchical
- **Action**: Use standard beta-binomial (this is valid scientific conclusion)

### Success Case 4: Pivot Required
- Evidence suggests different model class entirely
- **Action**: Design new models (e.g., if temporal structure discovered)

---

## Comparison to Standard Hierarchical

| Feature | Standard Beta-Binomial | Model 1: Mixture | Model 2: Robust | Model 3: DP |
|---------|----------------------|-----------------|----------------|------------|
| **Structure** | Continuous, unimodal | Discrete, K=2 | Heavy-tailed | Flexible K |
| **Parameters** | ~4 | ~8-10 | ~15 | ~3K_max+15 |
| **Complexity** | Low | Moderate | Low-Mod | High |
| **Comp. Cost** | Low | Moderate | Low | High |
| **Outliers** | Aggressive shrinkage | Separate cluster | Less shrinkage | Flexible |
| **When Best** | Unimodal continuous | Bimodal | Heavy tails | Unknown K |
| **Falsifiable** | Via posterior predictive | Via mixture weights | Via nu | Via K_eff |

---

## Technical Notes

### Computational Requirements
- **Model 1**: ~5-10 minutes, 2000 draws, 4 chains
- **Model 2**: ~3-7 minutes, 2000 draws, 4 chains
- **Model 3**: ~15-30 minutes, 3000 draws, 4 chains

### Dependencies
```python
import pymc as pm  # >= 5.0
import arviz as az  # >= 0.17
import numpy as np
import pandas as pd
```

### MCMC Settings
All models use:
- NUTS sampler (PyMC default)
- target_accept=0.95 (high for mixture models)
- 4 chains for convergence checking
- Random seed for reproducibility

### Convergence Criteria
- Rhat < 1.01 (ideally < 1.005)
- ESS > 400 per chain (ideally > 1000)
- No divergent transitions (or < 1%)
- Visual inspection of trace plots

---

## Contact & Next Steps

**This is Designer 2's output**. Next steps:
1. Coordinator reviews all designer proposals
2. Select models for implementation
3. Model developers implement and test
4. Assessors evaluate and compare
5. Final model selection or iteration

**Critical principle**: These are hypotheses to test, not a predetermined plan. If all fail, that's a valid scientific outcome indicating standard hierarchical is best.

---

## References

See `proposed_models.md` for:
- Full mathematical specifications
- Detailed prior justifications
- Complete falsification protocols
- Prior/posterior check procedures
- Model comparison strategies
- Scientific plausibility assessments

---

**End of README**
