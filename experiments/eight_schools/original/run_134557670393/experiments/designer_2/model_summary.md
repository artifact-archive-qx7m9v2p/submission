# Quick Reference: Designer #2 Models

## Three Proposed Model Classes

### Model 1: Bayesian Fixed-Effect Meta-Analysis
**Core equation**: `y_i ~ Normal(mu, sigma_i)`
**Parameters**: 1 (mu)
**Key assumption**: Perfect homogeneity (theta_i = mu for all i)
**Best for**: If I²=0% reflects true homogeneity
**Will fail if**: Posterior predictive checks reject (studies outside 95% intervals)

### Model 2: Precision-Stratified Fixed-Effect
**Core equation**: `y_i ~ Normal(mu_g[i], sigma_i)` where g[i] ∈ {1, 2}
**Parameters**: 2 (mu_high_precision, mu_low_precision)
**Key assumption**: Two effect levels (by precision group), within-group homogeneity
**Best for**: If small-study effects exist
**Will fail if**: Groups not different (|mu_1 - mu_2| < 2) or LOO worse than Model 1

### Model 3: Fixed-Effect with SE Uncertainty
**Core equation**: `y_i ~ Normal(mu, sigma_i * lambda)`
**Parameters**: 2 (mu, lambda)
**Key assumption**: Homogeneity + reported SEs may be miscalibrated
**Best for**: If measurement errors systematically under/overestimated
**Will fail if**: Lambda near 1 (no adjustment needed) or LOO worse than Model 1

## Critical Falsification Criteria (Abandon ALL Fixed-Effects If)

1. **Posterior predictive failure**: >2 studies outside 95% prediction intervals
2. **LOO strongly favors random-effects**: Δelpd > 10
3. **Study 1 dominates**: Removal changes pooled effect >50%
4. **Prior-data conflict**: p < 0.01
5. **Implausible parameters**: lambda > 3, |mu| > 50

## Priors Summary

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| mu | Normal(0, 15) | Weakly informative, spans observed range |
| mu_group | Normal(0, 15) | Identical for both groups, exchangeable |
| lambda | LogNormal(0, 0.2) | Centered at 1, allows ±20% SE uncertainty |

## Expected Best Outcome

**Most likely**: Model 1 wins (simplest, I²=0% is real)
**Most interesting**: Model 2 wins (precision stratification matters)
**Most important**: All fail → adopt random-effects (fixed-effects wrong)

## Implementation Order

1. Model 1 (fastest, baseline)
2. Model 2 (most interesting alternative)
3. Model 3 (sensitivity check)
4. Compare to Designer #1's random-effects models

## Key Insight

**Taking I²=0% seriously means testing it aggressively**. If these models fail, we've learned that complete pooling is wrong despite I²=0%—a scientifically valuable finding.
