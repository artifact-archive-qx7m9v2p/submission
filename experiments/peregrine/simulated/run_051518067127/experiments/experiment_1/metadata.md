# Experiment 1: Negative Binomial GLM with Quadratic Trend

**Model Class**: Count likelihood with overdispersion
**Priority**: Tier 1 (MUST attempt)
**Date Started**: 2025-10-30

## Model Specification

### Likelihood
```
C_t ~ NegativeBinomial2(mu_t, phi)
```

### Link Function
```
log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * year_t^2
```

### Parameters and Priors
```
beta_0 ~ Normal(4.5, 1.0)     # Intercept on log scale
beta_1 ~ Normal(0.9, 0.5)     # Linear growth rate
beta_2 ~ Normal(0, 0.3)       # Quadratic term (acceleration/deceleration)
phi ~ Gamma(2, 0.1)           # Dispersion parameter (reciprocal parameterization)
```

## Rationale

This is the simplest model that addresses the two core data features:
1. **Severe overdispersion** (Var/Mean = 70.43) → Negative Binomial likelihood
2. **Nonlinear exponential growth** → Quadratic term on log scale

The model assumes:
- Homogeneous dispersion across time (phi constant)
- No temporal autocorrelation (independent observations)
- Smooth exponential growth with potential acceleration/deceleration

## Falsification Criteria

**I will abandon this model if**:
- Residual ACF lag-1 > 0.5 (temporal structure not captured)
- Posterior predictive checks show systematic bias
- R-hat > 1.01 or divergent transitions
- LOO Pareto-k > 0.7 for >10% of observations

## Expected Outcomes

**Most likely**: Adequate fit for mean trend, but residual diagnostics fail due to autocorrelation (ACF lag-1 = 0.971 in data)

**If succeeds**: Simple solution! No need for more complex models.

**If fails**: Provides baseline for comparison, pivot to Experiment 2 (AR structure)

## Implementation

**Software**: Stan via CmdStanPy
**Sampling**: 4 chains, 2000 iterations (1000 warmup)
**Expected runtime**: 30-60 seconds
