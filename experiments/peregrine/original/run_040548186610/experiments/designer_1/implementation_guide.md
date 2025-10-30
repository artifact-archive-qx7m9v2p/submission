# Quick Implementation Guide: Parametric Bayesian GLMs

**Designer 1 - Parametric Focus**
**Date:** 2025-10-29

---

## Quick Start: Three Models at a Glance

### Model 1: Negative Binomial Quadratic (START HERE)
```
C ~ NegBinomial(mu, phi)
log(mu) = β₀ + β₁·year + β₂·year²
```
**Why:** Handles overdispersion + acceleration
**Reject if:** Residual ACF > 0.9 OR posterior predictive coverage < 50%

### Model 2: Negative Binomial Exponential
```
C ~ NegBinomial(mu, phi)
log(mu) = β₀ + β₁·year
```
**Why:** Tests exponential growth hypothesis (simpler)
**Reject if:** LOO-IC worse than Model 1 by > 10 points

### Model 3: Quasi-Poisson with Random Effects
```
C ~ Poisson(mu · exp(ε))
log(mu) = β₀ + β₁·year + β₂·year²
ε ~ Normal(0, σ)
```
**Why:** Flexible time-varying dispersion
**Reject if:** Divergent transitions OR ε shows strong autocorrelation

---

## Critical Diagnostics Checklist

After fitting each model, CHECK:

### 1. MCMC Health
- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 for all parameters
- [ ] No divergent transitions
- [ ] No max treedepth warnings

### 2. Posterior Predictive Checks
- [ ] Plot: Overlay y_rep on observed data
- [ ] Compute: % observations in 95% credible intervals (target: 90-98%)
- [ ] Compute: Var(y_rep)/Mean(y_rep) vs observed (target: ratio ≈ 1.0)

### 3. Residual Analysis
- [ ] Plot: ACF of Pearson residuals (target: all lags < 0.3)
- [ ] Plot: Residuals vs year (target: no pattern, random scatter)
- [ ] Plot: Residuals vs fitted (target: no U-shape or trend)

### 4. Model Comparison
- [ ] Compute LOO-IC for each model
- [ ] Check for Pareto k > 0.7 (influential points)
- [ ] Compare LOO-IC differences (>4 = meaningful, >10 = substantial)

### 5. Out-of-Sample Validation
- [ ] Fit on first 32 observations
- [ ] Predict last 8 observations
- [ ] Compute RMSE and check predictive intervals

---

## Decision Tree

```
Start: Fit Model 1 (NB Quadratic)
   |
   +---> MCMC converged?
          |
          NO --> Check non-centered parameterization, increase warmup
          |
          YES --> Check posterior predictive coverage
                  |
                  +---> Coverage > 85%?
                        |
                        YES --> Check residual ACF(1)
                        |       |
                        |       +---> ACF(1) < 0.7?
                        |             |
                        |             YES --> SUCCESS!
                        |             |      Model 1 is adequate baseline
                        |             |      Compare to Model 2
                        |             |
                        |             NO --> HIGH TEMPORAL CORRELATION
                        |                    Recommend AR models (Designer 2)
                        |
                        NO --> Coverage < 85%?
                               |
                               +---> Try Model 3 (flexible dispersion)
                                     OR recommend Designer 2 models
```

---

## Expected Results Summary Table

Fill this in after fitting:

| Metric | Model 1 | Model 2 | Model 3 | Threshold |
|--------|---------|---------|---------|-----------|
| LOO-IC | ___ | ___ | ___ | Lower is better |
| Coverage (%) | ___ | ___ | ___ | 90-98% |
| Var/Mean Ratio | ___ | ___ | ___ | ≈ 1.0 |
| Residual ACF(1) | ___ | ___ | ___ | < 0.5 |
| RMSE (OOS) | ___ | ___ | ___ | < 40 |
| ESS (min) | ___ | ___ | ___ | > 400 |
| Divergences | ___ | ___ | ___ | 0 |

**Decision:**
- Best model: ___________
- Adequate for data? YES / NO
- Next steps: ___________

---

## Priors Quick Reference

### Model 1 & 2:
```python
beta_0 ~ Normal(4.7, 0.5)   # Intercept: log(109) ≈ 4.7
beta_1 ~ Normal(0.8, 0.3)   # Growth rate
beta_2 ~ Normal(0.3, 0.2)   # Acceleration (Model 1 only)
phi ~ Gamma(2, 0.5)         # Dispersion
```

### Model 3:
```python
beta_0 ~ Normal(4.7, 0.5)
beta_1 ~ Normal(0.8, 0.3)
beta_2 ~ Normal(0.3, 0.2)
sigma_obs ~ Exponential(1)  # Obs-level variance
```

---

## Red Flags: When to Abandon Parametric GLMs

### CRITICAL - Abandon immediately if:
1. **All models** show residual ACF(1) > 0.80
2. **All models** have coverage < 75%
3. **All models** show systematic residual patterns
4. **All models** have LOO-IC within 3 points (equally bad)
5. **All models** have out-of-sample RMSE > 50

**If 2+ red flags → Parametric GLMs are insufficient**
**Recommend:** State-space models (Designer 2) or hierarchical temporal (Designer 3)

---

## Success Criteria

**GOOD parametric GLM if:**
- Coverage: 85-98%
- Residual ACF(1): < 0.6
- LOO-IC: Clearly best model (delta > 5 from others)
- No systematic residual patterns
- Out-of-sample RMSE < 35

**ADEQUATE parametric GLM if:**
- Coverage: 75-85%
- Residual ACF(1): 0.6-0.8
- Can be improved with extensions (AR errors, time-varying dispersion)

**FAILED parametric GLMs if:**
- Coverage < 75%
- Residual ACF(1) > 0.8
- Systematic patterns remain
- No clear winner among models

---

## Code Snippets

### Loading Data (Python/PyMC)
```python
import pandas as pd
import pymc as pm
import numpy as np

data = pd.read_csv('/workspace/data/data.csv')
year = data['year'].values
C = data['C'].values
N = len(C)

year_sq = year ** 2  # For Model 1 & 3
```

### Loading Data (R/Stan)
```r
library(rstan)
library(loo)

data <- read.csv('/workspace/data/data.csv')

stan_data <- list(
  N = nrow(data),
  y = data$C,
  year = data$year
)
```

### Quick Posterior Predictive Check
```python
import arviz as az

# After fitting
ppc = az.from_pymc3(trace)
az.plot_ppc(ppc)

# Compute coverage
y_rep = trace.posterior_predictive['y_rep']
lower = y_rep.quantile(0.025, dim=['chain', 'draw'])
upper = y_rep.quantile(0.975, dim=['chain', 'draw'])
coverage = np.mean((C >= lower) & (C <= upper))
print(f"Coverage: {coverage:.1%}")
```

### Quick LOO Comparison
```python
# PyMC
loo1 = az.loo(trace1)
loo2 = az.loo(trace2)
az.compare({'Model 1': trace1, 'Model 2': trace2})
```

```r
# Stan
loo1 <- loo(fit1)
loo2 <- loo(fit2)
loo_compare(loo1, loo2)
```

---

## Time Budget

**Phase 1: Fitting**
- Model 1: 1-2 min (simple, should converge fast)
- Model 2: 1-2 min (even simpler)
- Model 3: 5-10 min (more parameters)
- **Total: ~15 min**

**Phase 2: Diagnostics**
- Posterior predictive checks: 10 min per model
- Residual analysis: 10 min per model
- LOO computation: 5 min per model
- **Total: ~75 min**

**Phase 3: Out-of-Sample Validation**
- Refit on 80% data: 15 min
- Generate predictions: 5 min
- Compute metrics: 5 min
- **Total: ~25 min**

**Phase 4: Comparison & Decision**
- Create comparison table: 15 min
- Write summary: 30 min
- **Total: ~45 min**

**GRAND TOTAL: ~2.5 hours**

---

## Files to Generate

### Required Outputs:
1. `/workspace/experiments/designer_1/model1_fit.pkl` (or .rds)
2. `/workspace/experiments/designer_1/model2_fit.pkl`
3. `/workspace/experiments/designer_1/model3_fit.pkl` (if needed)
4. `/workspace/experiments/designer_1/diagnostics_report.md`
5. `/workspace/experiments/designer_1/model_comparison_table.csv`
6. `/workspace/experiments/designer_1/posterior_predictive_plots.png`
7. `/workspace/experiments/designer_1/residual_plots.png`

### Optional but Recommended:
8. `/workspace/experiments/designer_1/prior_predictive_check.png`
9. `/workspace/experiments/designer_1/loo_comparison.csv`
10. `/workspace/experiments/designer_1/trace_plots.png`

---

## Communication with Other Designers

### To Designer 2 (Non-Parametric):
**Pass along:**
- Residual ACF values from all models
- Whether parametric trends are adequate
- Evidence for time-varying parameters

**Questions to answer:**
- Should Designer 2 focus on AR structure?
- Is GP regression needed for trend?
- Are state-space models necessary?

### To Designer 3 (Hierarchical):
**Pass along:**
- Dispersion estimates by time period
- Evidence for changepoints
- Time-varying variance patterns

**Questions to answer:**
- Is hierarchical by period needed?
- Are changepoint models warranted?
- Should periods be modeled separately?

---

## Key Takeaways

**What parametric GLMs are GOOD at:**
- Interpretable parameters (growth rates, acceleration)
- Computational efficiency
- Clear hypothesis testing (polynomial vs exponential)
- Handling overdispersion (negative binomial)

**What parametric GLMs are BAD at:**
- High temporal correlation (lag-1 = 0.989 is HUGE)
- Time-varying parameters
- Unknown functional forms
- Small sample sizes with complex patterns (n=40 is limiting)

**Most likely outcome:**
- One model (probably Model 1) fits reasonably well
- But shows residual autocorrelation 0.6-0.8
- Suggests hybrid: Parametric trend + AR errors
- OR: Handoff to Designer 2 for full temporal treatment

**Least likely outcome:**
- All models fail completely → Parametric GLMs wrong class
- All models succeed perfectly → Data unrealistically simple

---

**Document prepared by:** Designer 1
**Purpose:** Quick reference for implementation
**Status:** Ready for model fitting
