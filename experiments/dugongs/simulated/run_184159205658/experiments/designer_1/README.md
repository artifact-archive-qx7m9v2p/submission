# Designer 1: Parsimonious & Interpretable Bayesian Models

**Designer:** Designer 1 (Parsimony Focus)
**Date:** 2025-10-27
**Philosophy:** Start simple, add complexity only when justified

---

## Overview

This directory contains a complete Bayesian modeling strategy for the Y vs x relationship based on EDA findings. The approach prioritizes:

1. **Parsimony** - Fewest parameters that adequately explain data
2. **Interpretability** - Models stakeholders can understand
3. **Falsification** - Actively trying to break models, not confirm them
4. **Scientific plausibility** - Parameters must make domain sense

---

## File Structure

```
/workspace/experiments/designer_1/
│
├── README.md                    # This file - overview and navigation
├── QUICK_REFERENCE.md           # Quick lookup guide for busy readers
├── proposed_models.md           # Full model specifications (main document)
└── implementation_guide.md      # PyMC/Stan code templates
```

---

## Recommended Reading Order

### For Quick Assessment (5 minutes)
1. Read **QUICK_REFERENCE.md** - TL;DR and decision flowchart

### For Implementation (30 minutes)
1. **proposed_models.md** - Section 1 (Model 1: Logarithmic)
2. **implementation_guide.md** - Model 1 code
3. Start coding!

### For Complete Understanding (2 hours)
1. **proposed_models.md** - Full document (all 3 models)
2. **implementation_guide.md** - All implementations
3. **QUICK_REFERENCE.md** - Reference during work

---

## Three Proposed Models

### Priority Ranking

| Rank | Model | Parameters | R² (expected) | When to Use |
|------|-------|------------|---------------|-------------|
| **1** | Logarithmic | 2 (β₀, β₁) | 0.83 | Default, EDA-supported |
| **2** | Power Law | 3 (β₀, β₁, β₂) | 0.85 | If Model 1 fails checks |
| **3** | Asymptotic | 3 (Y_min, Y_max, K) | 0.82 | If saturation expected |

### Model Comparison

```
                     Logarithmic    Power Law    Asymptotic
                     -----------    ---------    ----------
Complexity           Simple         Moderate     Complex
Parameters           2              3            3
Computational Cost   Low            Medium       High
Interpretability     High           Medium       High
Extrapolation        Unbounded      Unbounded    Bounded
EDA Support          Strong         Moderate     Moderate
Recommended Order    1st            2nd          3rd
```

---

## Key Insights from EDA

**Data characteristics:**
- N = 27 observations
- x ∈ [1.0, 31.5], Y ∈ [1.71, 2.63]
- Strong nonlinear relationship (r = 0.72)
- Diminishing returns pattern
- Normal residuals (Shapiro p = 0.334)
- Constant variance (Breusch-Pagan p = 0.546)
- Only 3 observations with x > 20 (sparse high-x region)

**Implications for modeling:**
- Normal likelihood justified
- Constant variance σ² appropriate
- Concave functional form needed
- High uncertainty for x > 20
- Modest N → favor simpler models

---

## Model 1: Logarithmic (PRIMARY)

**Specification:**
```
Y ~ Normal(μ, σ)
μ = β₀ + β₁·log(x)

Priors:
β₀ ~ Normal(1.73, 0.5)     # Intercept
β₁ ~ Normal(0.28, 0.15)    # Log-slope
σ ~ Exponential(5)         # Residual SD
```

**Why start here:**
- Only 2 parameters (lowest overfitting risk)
- Clear interpretation (elasticity)
- EDA shows R² = 0.83 (strong fit)
- Computationally efficient
- Diminishing returns built-in

**Success criteria:**
- R-hat < 1.01, ESS > 400
- Residuals normal, no patterns
- >90% observations in 95% posterior CI
- LOO-CV RMSE < 0.25

**Failure criteria (abandon if):**
- β₁ ≤ 0 (wrong direction)
- Systematic residual patterns
- Divergent transitions
- >20% obs outside 95% CI

**Expected outcome:** Adequate fit, serves as benchmark

---

## Implementation Quick Start

### Minimal PyMC Example

```python
import pymc as pm
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('/workspace/data/data.csv')
x_obs = data['x'].values
y_obs = data['Y'].values

# Model
with pm.Model() as log_model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.73, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.28, sigma=0.15)
    sigma = pm.Exponential('sigma', lam=5.0)

    # Mean structure
    mu = beta_0 + beta_1 * pm.math.log(x_obs)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_obs)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)

# Diagnostics
import arviz as az
print(az.summary(trace))
az.plot_trace(trace)
```

**See `implementation_guide.md` for complete code with diagnostics, PPC, LOO-CV, and visualization.**

---

## Decision Strategy

### Phase 1: Fit Model 1 (Days 1-2)
1. Implement logarithmic model
2. Check convergence (R-hat, ESS, trace plots)
3. Residual diagnostics
4. Posterior predictive checks
5. LOO-CV

**If Model 1 passes all checks → DONE, report results**

### Phase 2: Compare Models (Days 3-4, if needed)
6. Fit Model 2 (Power Law) if Model 1 shows residual patterns
7. Compare LOO-CV: ΔELPD > 3?
8. Choose based on fit + parsimony

### Phase 3: Explore Asymptotic (Days 5-6, only if warranted)
9. Fit Model 3 if unbounded growth implausible
10. Three-way comparison
11. Final selection based on LOO + interpretability

---

## Critical Checks

### Convergence
- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 (preferably >1000)
- [ ] Trace plots show good mixing
- [ ] No divergent transitions

### Model Adequacy
- [ ] Residuals approximately normal
- [ ] No systematic patterns in residual plot
- [ ] Homoscedastic (constant variance)
- [ ] No autocorrelation

### Predictive Performance
- [ ] >90% obs in 95% posterior CI
- [ ] LOO-CV: <20% with Pareto k > 0.7
- [ ] RMSE < 0.25 (better than marginal SD)

### Scientific Plausibility
- [ ] Parameter signs correct (β₁ > 0)
- [ ] Parameter magnitudes reasonable
- [ ] Extrapolation behavior sensible
- [ ] Interpretable to stakeholders

---

## Red Flags and Responses

| Red Flag | Meaning | Response |
|----------|---------|----------|
| Divergent transitions | Model misspecified | Check functional form, try reparameterization |
| R-hat > 1.05 | Non-convergence | Extend sampling, check initialization |
| Systematic residuals | Wrong functional form | Try Model 2 (power law) |
| β₁ ≤ 0 | Wrong direction | Model fundamentally wrong, reconsider |
| Wide Pareto k | Influential observations | Investigate outliers, robust likelihood? |
| Prior = Posterior | Data not informative | Need more data or different model |

---

## Extrapolation Caution

**Critical limitation:** Only 3 observations with x > 20

**Implications:**
- High uncertainty for predictions beyond x = 31.5
- Models 1 & 2 predict unbounded Y growth
- Model 3 predicts bounded Y (asymptote)

**Recommendation:**
- Report wide credible intervals for x > 20
- Explicitly state extrapolation uncertainty
- Collect more high-x data if predictions needed
- Consider Model 3 if domain suggests saturation

---

## Comparison to Other Designers

**If parallel designers exist:**

This design (Designer 1) emphasizes **parsimony and interpretability**. Other designers may propose:

- **Designer 2:** More flexible models (splines, Gaussian processes)
- **Designer 3:** More robust models (hierarchical, mixture)

**How to synthesize:**
1. Compare LOO-CV across all designs
2. Check for consensus on key features
3. Choose based on: LOO + interpretability + scientific plausibility
4. Simplest adequate model wins (Occam's razor)

---

## Expected Outcomes

### Best case (80% probability)
- Model 1 (Log) adequate
- Clean convergence, good residuals
- LOO-CV RMSE ≈ 0.20
- Clear interpretation, stakeholder-friendly

### Good case (15% probability)
- Model 1 inadequate, Model 2 better
- ΔELPD > 3 favors Power Law
- More complex but justified
- Still interpretable

### Challenging case (5% probability)
- All models fail checks
- Need robust likelihood (Student-t) or GP
- Heteroscedastic variance
- Data quality issues

---

## Contact and Questions

**For theoretical questions:**
- See `proposed_models.md` - full mathematical specifications

**For implementation questions:**
- See `implementation_guide.md` - complete PyMC/Stan code

**For quick lookup:**
- See `QUICK_REFERENCE.md` - condensed decision guide

---

## Philosophy: Falsification Over Confirmation

**Core principle:** We are trying to BREAK these models, not confirm them.

**Success = finding where models fail quickly**, then either:
1. Fixing the issue (better priors, reparameterization)
2. Pivoting to better model class
3. Stopping if model passes all stress tests

**A model that passes all falsification attempts is trustworthy.**

---

## Final Checklist Before Reporting

- [ ] Fitted at least Model 1 (Log)
- [ ] All convergence checks passed
- [ ] Residual diagnostics clean
- [ ] Posterior predictive checks good (>90%)
- [ ] LOO-CV computed, no severe warnings
- [ ] Parameters scientifically plausible
- [ ] Visualizations generated (trace, residual, predictions)
- [ ] Limitations documented (especially x > 20 extrapolation)
- [ ] Compared to alternative models if available
- [ ] Prior sensitivity checked
- [ ] Ready to defend model choice

---

## Summary

**Start simple (Model 1), validate thoroughly, add complexity only if justified.**

**Expected deliverable:** Logarithmic model with full diagnostics, or Power Law if data demands it.

**Timeline:** 2-4 days for Model 1 alone, up to 7 days if full comparison needed.

**Key strength:** Interpretability and parsimony (N=27 is modest).

**Key limitation:** High extrapolation uncertainty for x > 20.

---

**Good luck with the analysis! The simplest model that passes all checks is the best model.**

---

*Designer 1 - October 27, 2025*
