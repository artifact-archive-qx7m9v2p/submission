# Quick Reference: Changepoint Model Designs

## Three Competing Models

### Model 1: Known Changepoint (τ=17 Fixed)
**Assumption**: EDA is correct; break at observation 17
**Likelihood**: NB(μ_t, α) with log(μ_t) = β₀ + β₁·year + β₂·I(t>17)·(year-year₁₇) + AR(1)
**Priors**: Centered on EDA estimates but weakly informative
**Reject if**: β₂ not significant, residual autocorrelation >0.3, computational issues

### Model 2: Unknown Changepoint (τ ∈ [5,35])
**Assumption**: Challenge EDA; let data find break location
**Likelihood**: Same as Model 1 but τ is a parameter
**Priors**: Discrete uniform on τ; same for other parameters
**Reject if**: Posterior τ is diffuse/multimodal, computational disaster, τ far from 17 without justification

### Model 3: Multiple Changepoints (k=2 Fixed)
**Assumption**: Single break is oversimplification
**Likelihood**: NB with two changepoints τ₁ < τ₂
**Priors**: Ordered changepoints, hierarchical slopes
**Reject if**: Collapses to k=1, changepoints too close (<5 obs), predictive performance worse

## Decision Rules

1. **Start with Model 1** (simplest, fastest)
2. **If Model 1 passes all tests**: STOP, use Model 1
3. **If Model 1 fails on τ location**: Try Model 2
4. **If Model 2 suggests multiple breaks**: Try Model 3
5. **If all fail**: Abandon changepoint framework → GP/spline models

## Key Falsification Criteria (All Models)

- Rhat > 1.01 or ESS < 100 → Reject
- Divergent transitions > 1% → Reject
- LOO Pareto k > 0.7 for >5% obs → Reject
- Posterior predictive ACF(1) > 0.3 → Reject
- Out-of-sample RMSE > 1.5x in-sample → Reject
- Prior-posterior conflict (>2 SD shift) → Reject

## Implementation Tools

- **Model 1**: Stan (recommended) - fast, stable
- **Model 2**: PyMC (recommended) - better discrete parameter support
- **Model 3**: PyMC (required) - Stan too complex for multiple discrete params

## Estimated Runtime

- Model 1: 2-5 minutes
- Model 2: 10-30 minutes
- Model 3: 20-60 minutes (if attempted)

## What Would Make Me Abandon Everything?

1. All models show computational pathology → Model class is unidentified
2. Predictive performance terrible for all → Overfitting/wrong likelihood
3. GP model fits much better (ΔLOO > 10) → Smooth process, not discrete break
4. Simulation studies show non-recovery → Models fundamentally broken

## Alternative Model Classes (Plan B)

If changepoint models fail:
- Gaussian Process (Matérn kernel)
- Bayesian Structural Time Series (local linear trend)
- Natural cubic splines (regularized)
- State-space with time-varying slopes

**Philosophy**: Model selection is science, not preference. Let the data decide.
