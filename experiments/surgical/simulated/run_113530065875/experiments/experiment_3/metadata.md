# Experiment 3: Beta-Binomial (Simple Alternative)

**Date**: 2024
**Status**: Starting

## Model Specification

### Data
- J = 12 groups
- n_j = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360]
- r_j = [6, 19, 8, 34, 12, 13, 9, 30, 16, 3, 19, 27]

### Parameters
- **mu_p**: Mean success rate (probability scale, 0-1)
- **kappa**: Concentration parameter (controls overdispersion)
- **alpha**: Shape parameter 1 = mu_p × kappa
- **beta**: Shape parameter 2 = (1 - mu_p) × kappa
- **phi**: Overdispersion parameter = 1 / (kappa + 1)

### Priors
```
mu_p ~ Beta(5, 50)               # Weakly informative, centered at ~0.09
kappa ~ Gamma(2, 0.1)            # Allows wide range of overdispersion
```

### Likelihood
```
r_j ~ Beta-Binomial(n_j, alpha, beta)
where:
  alpha = mu_p × kappa
  beta = (1 - mu_p) × kappa
```

### Model Advantages
- **Simpler**: No hierarchical structure, fewer parameters (2 vs 14)
- **Faster**: Expected 2× faster than hierarchical
- **Direct interpretation**: Works on probability scale (no logit transform)
- **Natural overdispersion**: Beta-binomial naturally handles extra-binomial variation
- **Potentially better LOO**: Fewer parameters may reduce sensitivity

### Model Disadvantages
- **No group-specific estimates**: Only population-level inference
- **Cannot assess shrinkage**: No partial pooling
- **May underfit**: If heterogeneity is complex beyond simple overdispersion

## Comparison to Experiment 1

| Aspect | Exp 1 (Hierarchical) | Exp 3 (Beta-Binomial) |
|--------|---------------------|----------------------|
| **Parameters** | 14 (mu, tau, 12×theta) | 2 (mu_p, kappa) |
| **Inference** | Group-specific rates | Population-level only |
| **Complexity** | High (hierarchical) | Low (marginal) |
| **Overdispersion** | Via between-group SD | Via beta distribution |
| **Speed** | Slower (~90 sec) | Faster (~30-45 sec expected) |
| **LOO** | Failed (k > 0.7) | Unknown (test this) |

## Expected Results (from EDA)

- **mu_p**: 0.07 ± 0.01 (7% pooled rate)
- **kappa**: 10-30 (moderate concentration)
- **phi**: 0.03-0.09 (corresponds to observed φ ≈ 3.6)
- **Sampling time**: 30-45 seconds

## Falsification Criteria

### Must Pass All:
1. **Convergence**: R̂ < 1.01, ESS > 400
2. **Posterior predictive**:
   - Observed φ = 3.59 in 95% PP interval
   - Model captures overall rate distribution
3. **LOO**: Pareto k < 0.7 (CRITICAL - reason for trying this model)
4. **Boundary check**: mu_p not at 0 or 1 (computational failure)

### Decision Paths
- ✅ **All pass** → ACCEPT, compare to Exp 1
- ⚠️ **PP marginal** → INVESTIGATE, may be too simple
- ❌ **LOO still fails** → Both models have LOO issues, choose based on research goals
- ❌ **Cannot capture overdispersion** → REJECT, model inadequate

## Rationale for Testing

1. **Minimum attempt policy**: Need to test at least 2 models
2. **LOO comparison**: Exp 1 has unreliable LOO, need alternative with good LOO
3. **Simplicity**: If adequate, simpler model preferred (parsimony)
4. **Population-level goals**: If only need overall rate, this suffices

## Model Class
Simple marginal model (Experiment 3 of 6 in plan)

## Implementation
File: `posterior_inference/code/fit_beta_binomial.py`
