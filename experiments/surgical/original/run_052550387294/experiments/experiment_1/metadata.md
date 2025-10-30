# Experiment 1: Beta-Binomial Model

## Model Specification

**Model Class**: Beta-Binomial with conjugate priors
**Implementation**: Stan (CmdStanPy)
**Date**: 2025-10-30

### Likelihood

```
r_i ~ BetaBinomial(n_i, α, β)
```

Marginalized form (numerically stable):
- Integrates out trial-specific probabilities θ_i
- Direct computation of marginal likelihood

### Parameterization

Using mean-concentration parameterization:
```
μ = α/(α + β)        # Mean success probability
φ = α + β            # Concentration parameter
```

Then:
```
α = μ·φ
β = (1-μ)·φ
```

### Priors

```
μ ~ Beta(2, 25)      # Prior mean: E[μ] = 2/27 ≈ 0.074, matches pooled proportion
φ ~ Gamma(2, 2)      # Prior mean: E[φ] = 1, allows wide range for overdispersion
```

**Prior Justification**:
- **μ ~ Beta(2, 25)**: Centers around observed pooled proportion (0.074) with moderate uncertainty
  - 95% prior credible interval: approximately [0.01, 0.20]
  - Allows data to shift mean if needed

- **φ ~ Gamma(2, 2)**: Concentration parameter controls overdispersion
  - E[φ] = 1, SD[φ] ≈ 0.7
  - Small φ → high overdispersion (what we expect)
  - Allows φ ∈ (0, ∞) but concentrates mass in [0.1, 3]

### Expected Behavior

**Overdispersion Relationship**:
- Beta-Binomial variance: Var[r_i/n_i] = μ(1-μ)/(φ+1) × [1 + (n_i-1)/(φ+n_i)]
- Small φ → high variance → explains observed overdispersion
- Expected posterior: φ ∈ [1, 5] based on observed φ_obs = 3.51

### Falsification Criteria

Will REJECT this model if:
1. Posterior predictive p-value outside [0.05, 0.95] for χ² statistic
2. Concentration φ has posterior mode < 0.5 or > 50 (extreme/degenerate)
3. Funnel plot violations persist in posterior predictive samples
4. LOO Pareto k > 0.7 for more than 2 observations
5. Cannot recover known parameters in simulation-based validation

### Success Criteria

Will ACCEPT this model if:
1. Convergence: Rhat < 1.01 for all parameters
2. Effective sample size: ESS > 400 for all parameters
3. Posterior predictive checks: Captures key data features (variance, extreme values)
4. LOO diagnostics: All or most Pareto k < 0.7
5. Scientific interpretability: Clear, defensible parameter estimates

### Computational Approach

- **Software**: CmdStanPy with Stan
- **Sampling**: 4 chains, 2000 iterations (1000 warmup, 1000 sampling)
- **Diagnostics**: ArviZ for convergence and model checking
- **Log-likelihood**: Save pointwise log_lik for LOO-CV

### Implementation Status

- [ ] Prior predictive check
- [ ] Simulation-based validation
- [ ] Model fitting
- [ ] Posterior predictive check
- [ ] Model critique
- [ ] Decision: ACCEPT/REVISE/REJECT
