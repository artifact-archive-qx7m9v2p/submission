# Experiment 3: Pooled Hierarchical Logit (Simplified)

## Model Specification

**Model Class**: Partially pooled model with analytical posterior approximation
**Implementation**: scipy optimization with Laplace approximation (documented limitations)
**Date**: 2025-10-30
**Status**: Pragmatic fallback given computational constraints

### Context

**Why This Experiment?**
- Experiments 1 & 2 failed SBC due to inadequate inference method (MAP + Laplace approximation)
- Stan compiler unavailable, PyMC installation issues
- Need a Bayesian model that can be fit with available tools while acknowledging limitations

### Model Structure

**Hierarchical Model with Fixed Dispersion**:
```
r_i ~ Binomial(n_i, θ_i)
logit(θ_i) ~ Normal(μ_logit, σ_fixed)
μ_logit ~ Normal(-2.53, 1)
```

**Key Simplification**: Fix σ based on observed data variance, estimate only μ_logit and trial-specific θ_i.

### Alternative: Fully Bayesian with Uncertainty

Implement full Bayesian workflow using scipy + bootstrap:
1. Find MAP estimates for all parameters
2. Compute Hessian at MAP for uncertainty quantification
3. Use bootstrap resampling to validate uncertainty estimates
4. Acknowledge this is approximate Bayesian inference

### Implementation Plan

Given computational constraints, take pragmatic approach:
1. Fit model using available tools (scipy, numpy)
2. Quantify uncertainty using available methods
3. Perform posterior predictive checks
4. **Clearly document limitations and assumptions**
5. Compare to simple pooled binomial model

### Falsification Criteria

Will REJECT if:
1. Posterior predictive checks fail
2. Cannot capture observed overdispersion
3. Bootstrap/approximate inference shows model inadequacy
4. Simpler pooled model performs equally well

### Success Criteria

Will ACCEPT if:
1. Model captures key data features in posterior predictive checks
2. Uncertainty quantification is reasonable (even if approximate)
3. Better than simple pooled model
4. Scientific conclusions are defensible with stated limitations

##Status

- [ ] Model fitting with available tools
- [ ] Uncertainty quantification
- [ ] Posterior predictive checks
- [ ] Model critique
- [ ] Decision: ACCEPT/REVISE/REJECT
