# Experiment 1: Fixed-Effect Normal Model

**Model Class**: Classical fixed-effect meta-analysis with Gaussian measurement error
**Date Created**: 2025-10-28
**Status**: In Development

## Model Specification

**Likelihood**:
```
y_i | θ, σ_i ~ Normal(θ, σ_i²)   for i = 1, ..., 8
```

**Prior**:
```
θ ~ Normal(0, 20²)
```

**Parameters**:
- θ: Common effect parameter (single parameter of interest)

**Key Assumptions**:
1. All studies estimate the same underlying parameter (homogeneity)
2. Measurement errors are normally distributed
3. Measurement uncertainties σ_i are known and correctly specified
4. Studies are independent conditional on θ

## Rationale

This is the canonical model for meta-analysis when all studies estimate the same underlying parameter. The EDA provides strong evidence for homogeneity:
- Cochran's Q p = 0.696
- I² = 0%
- No outliers or publication bias detected

## Expected Results

- θ ≈ 7.7 ± 4.0 (matching EDA pooled estimate)
- Perfect convergence (R-hat = 1.000)
- Posterior should closely match analytical solution
- Good posterior predictive performance

## Falsification Criteria

- Systematic posterior predictive failures
- Evidence of heterogeneity in residual patterns
- Poor LOO-PIT calibration
- Residual patterns suggesting missing structure

## Implementation

- **PPL**: PyMC
- **Sampler**: NUTS with 4 chains
- **Samples**: 2000 post-warmup per chain
- **Validation**: Prior predictive → SBC → Fit → Posterior predictive → Critique
