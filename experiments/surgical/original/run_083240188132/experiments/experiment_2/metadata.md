# Experiment 2: Random Effects Logistic Regression

## Model Specification

**Model Class**: Random Effects Logistic Regression (GLMM with Gaussian random effects on logit scale)

**Likelihood**:
```
r_i | θ_i, n_i ~ Binomial(n_i, logit⁻¹(θ_i))  for i = 1, ..., 12 groups
```

**Hierarchical Structure** (Non-centered parameterization):
```
θ_i = μ + τ · z_i
z_i ~ Normal(0, 1)
```

**Priors**:
```
μ ~ Normal(logit(0.075), 1²)     # E[p] ≈ 0.075 (pooled rate 7.4%)
τ ~ HalfNormal(1)                # Between-group SD on logit scale
```

## Why This Model After Experiment 1 Failed

**Experiment 1 (Beta-Binomial) failure**: κ parameter had identifiability issues in high-overdispersion regime.

**Experiment 2 advantages**:
1. **Different parameterization**: τ (SD) is often better identified than κ (concentration)
2. **Logit scale**: More natural for hierarchical modeling (unbounded scale)
3. **Non-centered**: Improves MCMC geometry and convergence
4. **Well-studied**: Logistic GLMM is workhouse model, extensively validated
5. **Computational**: Expected to converge better than Beta-Binomial

## Parameters of Interest

- **μ** (scalar): Population mean log-odds
  - Expected: -2.5 to -2.3 (corresponds to p ~ 7-9%)
- **τ** (scalar): Between-group SD on logit scale
  - Expected: 0.6-1.2 based on ICC=0.66
  - Larger τ → more heterogeneity
- **θ_i** (12 values): Group-specific log-odds
- **Derived**: p_i = logit⁻¹(θ_i) (group proportions, 0-1)

## Theoretical Justification

1. **Standard practice**: Logistic mixed models are most common for grouped binomial data
2. **Symmetric heterogeneity**: Assumes log-odds vary normally (reasonable for rates 1-20%)
3. **Unbounded scale**: Logit scale avoids boundary issues (p∈[0,1] → θ∈ℝ)
4. **Non-centered**: Separates location (μ) from scale (τ), improves sampling
5. **Proven track record**: Extensively used in epidemiology, meta-analysis, ecology

## How This Model Addresses Data Challenges

1. **Overdispersion (φ=3.5-5.1)**: Induced through random effects variance τ²
2. **Heterogeneity (ICC=0.66)**: Directly modeled via τ
3. **Zero-event group (Group 1)**: Normal prior prevents θ₁ → -∞, shrinks toward μ
4. **Outliers (Groups 2,8,11)**: Gaussian allows extreme values with shrinkage

## Falsification Criteria

Will REJECT this model if:
1. **Extreme heterogeneity**: τ > 2.0 (suggests discrete subpopulations, try mixture)
2. **Outlier misfit**: Groups 2,8,11 consistently outside 95% posterior intervals
3. **Poor coverage**: < 70% of groups within posterior predictive 95% CI
4. **Computational failure**: Divergences >1%, Rhat > 1.01, ESS < 400
5. **Implausible posteriors**: Any p_i > 0.3

## Expected Outcomes

- μ posterior: -2.7 to -2.3 (median ~-2.5)
- τ posterior: 0.6-1.2 (median ~0.9)
- Similar substantive conclusions to Experiment 1 (if it had worked)
- Better convergence than Experiment 1 (target: >90% simulations converge)
- Runtime: 2-5 minutes (4 chains × 1000 samples)

## Comparison to Experiment 1

| Aspect | Exp 1 (Beta-Binomial) | Exp 2 (RE Logistic) |
|--------|----------------------|---------------------|
| Scale | Probability [0,1] | Log-odds (-∞,∞) |
| Heterogeneity param | κ (concentration) | τ (SD) |
| Parameterization | Centered | Non-centered |
| Identifiability | Poor for high OD | Generally better |
| Boundary issues | Yes (p near 0/1) | No (θ unbounded) |
| SBC result | FAILED | TBD |

## Implementation Details

**Software**: PyMC 5.x
**Sampler**: NUTS
**Chains**: 4
**Samples per chain**: 1000 (after 1000 tune)
**Total posterior samples**: 4000

## Status

- [ ] Prior predictive check
- [ ] Simulation-based validation
- [ ] Model fitting
- [ ] Posterior predictive check
- [ ] Model critique

## Notes

**Why starting with Experiment 2 instead of refining Experiment 1**:
- Experiment 1's identifiability issue is structural, not fixable with prior tuning
- Better to try alternative model class than iterate on broken model
- Experiment plan already identified this as "alternative primary"
- If Experiment 2 also fails SBC, we'll reconsider strategy

**Success metric**: Pass SBC with >80% convergence and <30% relative error in parameter recovery for scenarios matching our data (high heterogeneity).
