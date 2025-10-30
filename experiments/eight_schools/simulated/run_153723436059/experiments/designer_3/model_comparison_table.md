# Quick Model Comparison Table

## Designer 3 - Three Proposed Bayesian Models

| Aspect | Model 1: Near-Complete Pooling | Model 2: Horseshoe Sparse | Model 3: Sigma Misspecification |
|--------|-------------------------------|---------------------------|--------------------------------|
| **Core Hypothesis** | Schools are highly similar | Most schools similar, 1-2 outliers | Reported sigmas are unreliable |
| **EDA Motivation** | I² = 1.6%, variance ratio = 0.75 | School 5 negative, School 4 large | Variance paradox (obs < expected) |
| **Likelihood** | y_i ~ Normal(theta_i, sigma_i) | y_i ~ Normal(theta_i, sigma_i) | y_i ~ Normal(theta_i, sigma_i * psi_i) |
| **School Effects** | theta_i ~ Normal(mu, tau) | theta_i ~ Normal(mu, lambda_i * tau) | theta_i ~ Normal(mu, tau) |
| **Key Innovation** | Informative tau prior | School-specific shrinkage | Infer true sigmas |
| **Prior on tau** | HalfNormal(0, 5) | HalfCauchy(0, 25) | HalfNormal(0, 5) |
| **Extra Parameters** | None | lambda_i (J parameters) | psi_i, omega (J+1 parameters) |
| **Complexity** | Low (J+2 params) | Medium (2J+2 params) | Medium (2J+3 params) |
| **Expected tau** | Small (< 5) | Moderate (5-15) | Small, but with corrected sigmas |
| **Expected Shrinkage** | Strong, uniform | Strong for most, weak for 1-2 | Strong, accounting for sigma errors |
| **Computational Cost** | Low | Medium-High | Medium |
| **Wins If** | Low heterogeneity is real | Schools 4,5 are true outliers | Sigmas wrong by 20-50% |
| **Falsified If** | Posterior tau > 10 | All lambda_i similar | Posterior omega ≈ 0 |
| **Stress Tests** | Leave-one-out schools, tau prior | Outlier ID, lambda prior | Sigma corrections plausible |

## Parameter Definitions

- **theta_i**: True effect for school i (what we want to estimate)
- **mu**: Population mean effect (hyperparameter)
- **tau**: Between-school standard deviation (heterogeneity)
- **lambda_i**: School-specific shrinkage factor (Model 2 only)
- **psi_i**: Multiplicative correction to reported sigma_i (Model 3 only)
- **omega**: Scale of sigma misspecification (Model 3 only)

## Decision Tree

```
Fit Model 1 (Baseline)
│
├─ tau small & PPCs pass → DONE (Model 1 wins)
│
├─ tau large (> 8) → Fit Model 2
│   ├─ 1-2 schools have lambda > 1 → Model 2 wins
│   └─ All lambda similar → Model 1 wins
│
├─ Variance paradox unresolved → Fit Model 3
│   ├─ omega > 0 & corrections plausible → Model 3 wins
│   └─ omega ≈ 0 → Model 1 wins
│
└─ All fail → RECONSIDER EVERYTHING
```

## Expected Results (Prior to Seeing Data)

**Most likely**: Model 1 wins
- EDA strongly suggests homogeneity
- Simplest model, well-powered
- Variance paradox explained by true similarity

**Possible**: Model 2 wins
- If Schools 4 and 5 are genuine outliers
- Horseshoe identifies them
- Better LOO-CV for those schools

**Unlikely but possible**: Model 3 wins
- If sigma estimation was poor across studies
- Correcting sigmas resolves paradox
- Would need 20%+ misspecification

**Null result**: All equivalent
- Data too sparse (n=8)
- Report simplest (Model 1)
- Acknowledge uncertainty

## Key Metrics for Model Selection

1. **LOO-CV ELPD**: Higher is better, difference > 2 SE meaningful
2. **Pareto-k**: < 0.5 good, > 0.7 problematic
3. **Posterior predictive p-value**: 0.05 < p < 0.95 ideal
4. **R-hat**: < 1.01 required
5. **Effective sample size**: > 400 desired

## Prior Choices Rationale

### Model 1: Informed by EDA
- tau ~ HalfNormal(0, 5): Expects small heterogeneity based on I² = 1.6%
- More concentrated than standard HalfCauchy(0, 25)
- Justified by strong EDA evidence

### Model 2: Standard Horseshoe
- lambda ~ HalfCauchy(0, 1): Standard horseshoe specification
- tau ~ HalfCauchy(0, 25): Allows more heterogeneity than Model 1
- Designed for sparse heterogeneity

### Model 3: Weak Sigma Prior
- omega ~ HalfNormal(0, 0.3): Allows up to 2x sigma errors
- psi ~ LogNormal(0, omega): Unbiased multiplicative errors
- Weakly informative: doesn't assume large errors

## Implementation Notes

All models use:
- Non-centered parameterization (avoids funnel)
- Stan NUTS sampler with adapt_delta = 0.9
- 4 chains x 2000 iterations (1000 warmup)
- Posterior predictive checks
- LOO-CV for model comparison

## Files Reference

- Full details: `/workspace/experiments/designer_3/proposed_models.md`
- Summary: `/workspace/experiments/designer_3/README.md`
- This table: `/workspace/experiments/designer_3/model_comparison_table.md`
