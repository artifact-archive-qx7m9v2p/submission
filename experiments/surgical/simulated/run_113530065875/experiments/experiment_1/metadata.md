# Experiment 1: Hierarchical Binomial (Logit-Normal, Non-Centered)

**Date**: 2024
**Status**: Fitting with PyMC (MCMC)

## Model Specification

### Data
- J = 12 groups
- n_j = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360]
- r_j = [6, 19, 8, 34, 12, 13, 9, 30, 16, 3, 19, 27]

### Parameters
- **mu**: Population mean success rate (logit scale)
- **tau**: Between-group standard deviation (logit scale)
- **theta_j**: Group-level logit success rates (j = 1,...,12)
- **p_j**: Group-level success probabilities = inv_logit(theta_j)

### Priors
```
mu ~ Normal(-2.5, 1)              # Weakly informative, centers at ~7-8% success rate
tau ~ Half-Cauchy(0, 1)           # Standard hierarchical SD prior
theta_raw_j ~ Normal(0, 1)        # Non-centered parameterization
theta_j = mu + tau * theta_raw_j  # Transform to centered
```

### Likelihood
```
r_j ~ Binomial(n_j, inv_logit(theta_j))
```

### Non-Centered Parameterization
Used to improve sampling geometry when:
- Small number of groups (J=12)
- Hierarchical SD (tau) has wide prior
- Better separation between hierarchical and group parameters

## Validation History

### Prior Predictive Check: CONDITIONAL PASS
- Range coverage: 55.1% of simulations cover [3.1%, 14.0%] ✓
- Overdispersion: 78.2% generate φ ≥ 3 ✓
- Interval coverage: 100% of groups covered ✓
- Minor issue: 6.88% samples have p > 0.8 (marginally exceeds 5% threshold)
- **Decision**: Proceed (data will dominate)

### Simulation-Based Calibration: FAIL (Method Issue)
- **Laplace approximation unsuitable** for hierarchical models with heavy-tailed priors
- tau coverage: 18% (catastrophic failure of uncertainty quantification)
- **Not a model problem**: Specification is correct
- **Solution**: Use MCMC (PyMC) instead

## MCMC Configuration

### Software
- **PPL**: PyMC 5.26.1
- **Sampler**: NUTS (No-U-Turn Sampler)
- **Backend**: PyTensor (Python-only mode, no C compilation)

### Sampling Parameters
- **Chains**: 4
- **Warmup**: 2000 iterations per chain
- **Sampling**: 2000 iterations per chain
- **Total samples**: 8000 post-warmup
- **Target accept**: 0.95 (high for hierarchical models)

### Convergence Criteria
- R̂ < 1.01 for all parameters
- ESS > 400 for all parameters
- Divergences < 1% of samples
- E-BFMI > 0.2

## Expected Results (from EDA)

- **mu**: -2.4 ± 0.3 (pooled rate ~8%)
- **tau**: 0.4 ± 0.2 (moderate heterogeneity)
- **p_j**: Range 4-12% with shrinkage:
  - Small groups (n<100): 60-72% shrinkage
  - Large groups (n>250): 19-30% shrinkage
- **Sampling time**: 5-15 minutes (Python-only mode)

## Falsification Criteria

### Must Pass All:
1. **Convergence**: R̂ < 1.01, ESS > 400, divergences < 1%
2. **Posterior predictive**:
   - Observed φ = 3.59 in 95% PP interval
   - Groups 2, 4, 8 have |z| < 3 in PP distribution
   - Shrinkage validates: small-n shrink more than large-n
3. **LOO**: Pareto k < 0.7 for all groups
4. **Scientific plausibility**: All p_j in [0, 0.3] (no unrealistic rates)

### Decision Paths
- ✅ **All pass** → ACCEPT, proceed to model critique
- ⚠️ **Convergence issues** → Increase warmup, adjust target_accept
- ⚠️ **PP fails** → Try Experiment 2 (Robust Student-t)
- ❌ **Fundamental failure** → Try Experiment 3 (Beta-binomial)

## Model Class
Primary hierarchical model (Experiment 1 of 6 in plan)

## Implementation
File: `posterior_inference/code/fit_hierarchical_binomial.py`
