# Experiment 1: Logarithmic Regression

**Model Class:** Bayesian Logarithmic Regression
**Priority:** PRIMARY (Expected Success: 80%)
**Date:** 2024-10-27

## Model Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = β₀ + β₁ · log(x_i)
```

### Priors
```
β₀ ~ Normal(1.73, 0.5)     # Intercept, centered at EDA estimate
β₁ ~ Normal(0.28, 0.15)    # Slope, weakly positive
σ ~ Exponential(5)         # Residual SD, mean=0.2
```

## Theoretical Justification

- Captures diminishing returns naturally (concave function)
- Elasticity interpretation: 1% change in x → (β₁/100)% change in Y
- Strong EDA support (R²=0.83)
- Only 2 parameters (most parsimonious)
- Linear in parameters (easy MCMC sampling)

## Success Criteria

- R-hat < 1.01, ESS > 400 for all parameters
- No divergent transitions
- β₁ > 0 (positive relationship)
- Residuals show no systematic patterns
- Posterior predictive checks: >90% coverage at 95% CI
- LOO-CV ELPD better than linear baseline

## Failure Criteria (ABANDON if)

- Systematic residual patterns vs x or fitted values
- β₁ posterior includes 0 or negative values
- Posterior predictive check shows <85% coverage
- LOO-CV: >20% of observations have Pareto k > 0.7

## Validation Pipeline Status

- [ ] Prior Predictive Check
- [ ] Simulation-Based Validation
- [ ] Model Fitting
- [ ] Posterior Predictive Check
- [ ] Model Critique
