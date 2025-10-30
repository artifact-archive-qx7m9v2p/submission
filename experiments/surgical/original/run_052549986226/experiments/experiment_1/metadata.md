# Experiment 1: Beta-Binomial (Reparameterized)

## Model Specification

### Likelihood
```
r_i ~ Binomial(n_i, p_i)  for i = 1, ..., 12
```

### Group-level probabilities
```
p_i ~ Beta(a, b)
```

### Reparameterization
```
a = μ * κ
b = (1 - μ) * κ

where:
  μ = population mean success probability
  κ = concentration parameter (controls heterogeneity)
```

### Priors
```
μ ~ Beta(2, 18)      # Centered on 0.1, weak prior
κ ~ Gamma(2, 0.1)    # Mean = 20, allows wide range
```

### Generated Quantities
```
φ = 1 + 1/κ          # Overdispersion parameter
log_lik[i]           # For LOO-CV
y_rep[i]             # Posterior predictive samples
p_posterior[i]        # Posterior distribution of group probabilities
```

## Rationale

### Why Beta-Binomial?
1. **Best preliminary fit:** AIC = 47.69 in EDA analysis (42 points better than pooled)
2. **Natural overdispersion:** Beta distribution explicitly models heterogeneity in success probabilities
3. **Handles zero counts:** No log-odds singularity for Group 1 (0/47)
4. **Parsimonious:** Only 2 hyperparameters vs 12+ for hierarchical models
5. **Shrinkage built-in:** Provides automatic partial pooling

### Why Reparameterization (μ, κ)?
- More interpretable than (α, β)
- μ directly represents population mean
- κ controls concentration: κ → 0 (high heterogeneity), κ → ∞ (homogeneous)
- Easier to set priors based on EDA findings

## Prior Justification

### μ ~ Beta(2, 18)
- **EDA finding:** Pooled success rate = 7.6%
- **Prior mean:** 2/(2+18) = 10%
- **Prior 95% interval:** [0.02, 0.28] or [2%, 28%]
- **Interpretation:** Weakly informative, centered near observed rate but allows wide range

### κ ~ Gamma(2, 0.1)
- **Prior mean:** 2/0.1 = 20
- **Prior SD:** √(2/0.1²) = 20 (wide uncertainty)
- **Implied φ = 1 + 1/κ:**
  - If κ = 1: φ = 2 (mild overdispersion)
  - If κ = 20: φ = 1.05 (nearly binomial)
  - If κ = 0.3: φ = 4.3 (severe overdispersion, observed level)
- **Interpretation:** Wide prior allows data to determine concentration, consistent with observed φ ≈ 3.5

## Expected Posterior

If model is correct:
- μ ≈ 0.07-0.08 (near observed 7.6%)
- κ ≈ 0.3-5 (low concentration given high ICC = 0.73)
- φ ≈ 3.0-4.0 (matching observed overdispersion)

### Group-specific predictions:
- **Group 1** (0/47): Shrink from 0% to ~2-4%
- **Group 4** (46/810): Minimal shrinkage ~5.7% (large sample)
- **Group 8** (31/215): Shrink from 14.4% to ~11-13% (moderate outlier)
- **Average shrinkage:** ~85% toward population mean (consistent with EDA estimate)

## Falsification Criteria

ABANDON this model if:

1. **Cannot reproduce overdispersion:**
   - Posterior predictive φ_rep < 2.0 (observed ≈ 3.5)
   - p-value for variance < 0.01 or > 0.99

2. **Prior-posterior conflict:**
   - Posterior κ entirely outside prior support
   - Posterior μ conflicts with data (>99% posterior density outside [0.01, 0.30])

3. **Poor predictive performance:**
   - LOO Pareto k > 0.7 for >3 groups
   - Cannot predict zero counts (Group 1)

4. **Bimodality in group rates:**
   - Posterior shows discrete clusters (contradicts EDA finding of continuous variation)

5. **Computational failure:**
   - Persistent convergence issues (Rhat > 1.01, ESS < 400)
   - >5% divergences

## Success Criteria

ACCEPT this model if:

- ✅ **Convergence:** Rhat < 1.01, ESS > 400 for μ, κ
- ✅ **Overdispersion reproduced:** Posterior φ ≈ 3.0-4.5
- ✅ **Predictions reasonable:** LOO Pareto k < 0.7 for all groups
- ✅ **Posterior predictive checks pass:** 0.05 < p < 0.95 for key statistics
- ✅ **Shrinkage patterns reasonable:** Group 1 → 1-4%, Group 8 moderate shrinkage
- ✅ **Interpretable:** Can explain findings to domain experts

## Implementation

### Software
- Stan via CmdStanPy
- beta_binomial_lpmf likelihood available in Stan

### Sampling Strategy
- 4 chains
- 2000 iterations per chain (1000 warmup)
- adapt_delta = 0.95 (control divergences)
- Expected runtime: 2-5 minutes

### Stan Model Location
`experiments/designer_1/stan_models/model_b_reparameterized.stan`

## Status

- **Stage:** Prior Predictive Check
- **Started:** 2024
- **Expected completion:** Phase 3 of modeling workflow
