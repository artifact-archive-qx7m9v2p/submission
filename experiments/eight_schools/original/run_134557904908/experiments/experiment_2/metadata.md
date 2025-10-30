# Experiment 2: Random-Effects Hierarchical Model

**Model Class**: Bayesian hierarchical meta-analysis with partial pooling
**Date Created**: 2025-10-28
**Status**: In Development

## Model Specification

**Likelihood**:
```
y_i | θ_i, σ_i ~ Normal(θ_i, σ_i²)   for i = 1, ..., 8
θ_i | μ, τ ~ Normal(μ, τ²)
```

**Priors**:
```
μ ~ Normal(0, 20²)
τ ~ Half-Normal(0, 5²)
```

**Parameters**:
- μ: Population mean effect (hyperparameter)
- τ: Between-study heterogeneity (hyperparameter)
- θ_i: Study-specific true effects (8 latent parameters)

**Implementation**: Non-centered parameterization for computational stability
```
θ_raw ~ Normal(0, 1)
θ = μ + τ * θ_raw
```

## Rationale

This model tests the key scientific hypothesis: **Does between-study heterogeneity exist?**

While EDA found strong homogeneity (I² = 0%, Q p = 0.696), this could reflect:
1. True homogeneity (τ = 0)
2. Low power with J=8 studies
3. Heterogeneity masked by large measurement errors

The hierarchical model allows data to determine the degree of pooling:
- If τ → 0: collapses to fixed-effect (Model 1)
- If τ > 0: partial pooling between studies

## Expected Results

Based on EDA and Model 1:
- μ ≈ 7.4 ± 4.0 (similar to θ in Model 1)
- τ ≈ 0 (confirming homogeneity)
- I² ≈ 0%
- LOO performance similar to Model 1

## Falsification Criteria

**REJECT** if:
- Cannot achieve convergence (funnel pathology, divergences)
- τ substantially > 0 contradicting EDA
- Worse predictive performance than Model 1
- Computational instability across parameterizations

**REVISE** if:
- Mild convergence issues → try centered parameterization
- Prior sensitivity extreme → adjust τ prior

## Scientific Hypotheses

**H0**: τ = 0 (homogeneity, Model 1 adequate)
**H1**: τ > 0 (heterogeneity, hierarchical model needed)

EDA supports H0, but Bayesian inference provides direct evidence.

## Implementation

- **PPL**: PyMC
- **Sampler**: NUTS with 4 chains
- **Samples**: 2000 post-warmup per chain
- **Parameterization**: Non-centered (prevents funnel)
- **Validation**: Prior predictive → SBC → Fit → Posterior predictive → Critique
