# Quick Reference: Designer 2 Models

## Three Proposed Model Classes

### 1. Gaussian Process Regression (Non-parametric)
**Formula**: f ~ GP(μ_0, Matérn3/2(α², ℓ))
**Priors**:
- μ_0 ~ Normal(2.3, 0.5)
- α² ~ HalfNormal(0.3)
- ℓ ~ InverseGamma(5, 10)
- σ ~ HalfNormal(0.15)

**Abandon if**: ℓ_post >> 30 OR LOO worse than log model by >3

### 2. Robust Regression (Student-t errors)
**Formula**: Y ~ StudentT(ν, α + β*log(x), σ)
**Priors**:
- α ~ Normal(1.75, 0.5)
- β ~ Normal(0.27, 0.15)
- ν ~ Gamma(2, 0.1)
- σ ~ HalfNormal(0.2)

**Abandon if**: ν_post > 50 OR LOO worse than Gaussian by >2

### 3. Penalized B-Splines (Flexible parametric)
**Formula**: μ = B * β, β ~ Normal(0, τ²)
**Priors**:
- β_k ~ Normal(0, τ²) for k=1,...,9
- τ ~ HalfCauchy(0, 1)
- σ ~ HalfNormal(0.15)
**Knots**: 5 interior knots at x quantiles

**Abandon if**: τ_post < 0.1 OR LOO worse than log by >3

## Decision Logic

```
Fit all three → Compute LOO → Check ΔELPD

If ΔELPD < 2:  Choose simplest (Robust-t)
If 2-5:        Choose Spline (best balance)
If > 5:        Choose GP (strong non-parametric evidence)
```

## Critical Diagnostics

1. Rhat < 1.01 for all parameters
2. ESS > 400 (10% of post-warmup)
3. Divergences < 1%
4. Pareto-k < 0.7 for LOO
5. Posterior predictive: Y_rep ∈ [1, 3]

## Sensitivity Tests Required

1. Remove x=31.5, refit all
2. Fit on x≤20, predict x>20
3. Check replicate variance
4. Prior sensitivity (2x scale)
5. Synthetic data recovery

## Files

- Full specification: `/workspace/experiments/designer_2/proposed_models.md`
- Stan models: To be implemented in separate files
- Results: Will be in `/workspace/experiments/designer_2/results/`
