# Prior Predictive Check - Findings

**Model**: Random-Effects Hierarchical Meta-Analysis
**Date**: 2025-10-28
**Status**: PASS (with note)

## Objective

Test whether the observed data is plausible under the joint prior distribution and assess prior sensitivity to the hyperprior on τ (between-study heterogeneity).

## Prior Specifications Tested

1. **Baseline**: τ ~ Half-Normal(0, 5²)
2. **Tight**: τ ~ Half-Normal(0, 3²)
3. **Wide**: τ ~ Half-Normal(0, 10²)

All specifications use: μ ~ Normal(0, 20²)

## Results

### Baseline Prior (τ ~ Half-Normal(0, 5²))

**Hyperparameter priors**:
- μ: mean = 0.11, SD = 19.93
- τ: mean = 4.02, median = 3.42, 95% CI = [0.15, 11.21]

**Prior predictive distribution**:
- Range: [-107.0, 110.1]
- All 8 observed data points fall within [1%, 99%] percentile range
- No extreme observations detected

**Percentile ranks of observed data**:
- Study 0 (y=28): 85.8th percentile
- Study 1 (y=8): 63.0th percentile
- Study 2 (y=-3): 45.0th percentile
- Study 3 (y=7): 61.6th percentile
- Study 4 (y=-1): 48.4th percentile
- Study 5 (y=1): 51.6th percentile
- Study 6 (y=18): 78.1th percentile
- Study 7 (y=12): 66.1th percentile

All percentile ranks are reasonable (none < 1% or > 99%).

### Prior Sensitivity Analysis

Comparing three τ hyperpriors:

| Prior Spec | τ mean | τ median | τ 97.5% | y range | Extreme obs |
|-----------|--------|----------|---------|---------|-------------|
| Tight (σ=3) | 2.39 | 1.99 | 6.73 | 196.1 | 0 |
| Baseline (σ=5) | 4.02 | 3.42 | 11.21 | 217.1 | 0 |
| Wide (σ=10) | 8.01 | 6.79 | 22.17 | 248.0 | 0 |

**Observations**:
- Prior predictive ranges are similar across specifications (196-248)
- Percentile ranks remain consistent across priors
- τ prior has moderate influence on prior predictive (sensitivity ratio = 3.30)
- **Note**: Ratio > 3 indicates some sensitivity, typical for hierarchical models with J=8

## Visualizations

1. **prior_predictive_check.png**:
   - Top row: μ prior distributions
   - Second row: τ prior distributions for three specifications
   - Third row: Prior predictive vs observed data
   - Bottom row: Percentile ranks of observed data

2. **prior_sensitivity.png**:
   - Comparison of τ priors
   - Prior predictive sensitivity
   - Percentile rank comparison
   - Summary statistics table

## Assessment

**Plausibility**: All observed data points are plausible under the prior (0/8 extreme observations).

**Prior sensitivity**: Moderate sensitivity to τ hyperprior (ratio = 3.30). This is expected for hierarchical models with small J and suggests:
- Posterior inference may show some prior dependence
- Should report sensitivity analysis in final results
- With J=8, data may not strongly identify τ

**Prior adequacy**: The baseline prior (τ ~ Half-Normal(0, 5²)) is reasonable:
- Allows τ = 0 (homogeneity) as plausible
- Permits substantial heterogeneity (τ up to ~11)
- Balances informativeness with flexibility

## Decision

**PASS (with note)**

**Reasoning**:
- All observed data is plausible under the joint prior
- Prior predictive distribution covers reasonable range
- Prior sensitivity is moderate and expected for hierarchical models
- **Note**: Report posterior sensitivity to τ hyperprior in final analysis

## Recommendations

1. Proceed with baseline prior τ ~ Half-Normal(0, 5²)
2. Conduct sensitivity analysis by refitting with tight and wide priors
3. In posterior inference, carefully examine:
   - Posterior distribution of τ (especially P(τ < 1))
   - Prior-posterior overlap for τ
   - Whether data provides information beyond prior
4. If τ posterior is very similar to prior, data is uninformative about heterogeneity

## Next Steps

- Proceed to Simulation-Based Calibration (Stage 2)
- Test whether inference can recover known parameter values
- Check for funnel pathology with non-centered parameterization
