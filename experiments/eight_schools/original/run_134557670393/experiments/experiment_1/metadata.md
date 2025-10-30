# Experiment 1: Bayesian Hierarchical Meta-Analysis

**Model Class**: Bayesian Random-Effects Meta-Analysis with Adaptive Shrinkage
**Date Started**: 2025-10-28
**Priority**: HIGH (Required - Primary Model)
**Source**: Synthesized from Designer #1 (Adaptive Hierarchical) and Designer #3 (heterogeneity framework)

---

## Model Specification

### Likelihood
```
y_i | theta_i, sigma_i ~ Normal(theta_i, sigma_i^2)   for i = 1, ..., 8
```

### Hierarchical Structure
```
theta_i | mu, tau ~ Normal(mu, tau^2)
```

### Priors
```
mu ~ Normal(0, 50)           # Weakly informative on overall effect
tau ~ Half-Cauchy(0, 5)      # Standard meta-analysis prior (Gelman 2006)
```

### Parameters
- `y_i`: Observed effect size (data)
- `sigma_i`: Known standard error (data, NOT estimated)
- `theta_i`: True underlying effect for study i
- `mu`: Population mean effect (primary estimand)
- `tau`: Between-study standard deviation (heterogeneity)

---

## Rationale

**Why this model**:
1. Most flexible: nests fixed-effect (tau→0) and random-effects (tau>0) as special cases
2. Addresses "heterogeneity paradox" from EDA via data-driven shrinkage
3. Standard approach recommended by all three designers
4. Well-established priors (Gelman 2006) for meta-analysis

**What it captures**:
- Partial pooling with adaptive shrinkage
- Measurement error structure (precise studies weighted higher)
- Full uncertainty propagation
- Handles I²=0% via tau posterior concentrating near zero

---

## Falsification Criteria

### REJECT if:
1. **Posterior predictive failure**: >1 study outside 95% posterior predictive interval
2. **Leave-one-out instability**: max |Δmu| > 5 units when removing any study
3. **Convergence failure**: R-hat > 1.05 OR ESS < 400 OR divergences > 1%
4. **Extreme shrinkage asymmetry**: Any |E[theta_i] - y_i| > 3*sigma_i

### REVISE if:
- **Prior-posterior conflict**: P(tau > 10 | data) > 0.5 with prior P(tau > 10) < 0.05
- **Unidentifiability**: tau posterior essentially uniform

### ACCEPT if:
- All falsification checks pass
- Convergence achieved (R-hat < 1.01, ESS > 400, no divergences)
- Posterior predictive shows reasonable fit
- Leave-one-out stable (all Δmu < 5)

---

## Implementation Plan

1. **Prior Predictive Check** (prior-predictive-checker agent)
   - Verify priors generate plausible data
   - Check prior coverage of observed data
   - Identify prior misspecification

2. **Simulation-Based Validation** (simulation-based-validator agent)
   - Test parameter recovery with known tau=0 and tau=5
   - Verify model can distinguish fixed vs random effects
   - Check identifiability with J=8

3. **Model Fitting** (model-fitter agent)
   - Fit with Stan/CmdStanPy
   - HMC with adaptive sampling
   - Save log_likelihood in InferenceData for LOO

4. **Posterior Predictive Check** (posterior-predictive-checker agent)
   - Compare observed data to posterior predictive
   - Check for systematic misfits
   - Assess model adequacy

5. **Model Critique** (model-critique agent)
   - Apply all falsification criteria
   - Make ACCEPT/REVISE/REJECT decision
   - Document strengths and weaknesses

---

## Expected Challenges

1. **Funnel geometry**: If tau→0, may need non-centered parameterization
2. **tau weakly identified**: J=8 may lead to wide tau posterior (expected, not problematic)
3. **Study 1 influence**: y=28 may dominate, check via leave-one-out
4. **Borderline significance**: Overall effect CI may include zero

---

## Status

- [ ] Prior Predictive Check
- [ ] Simulation-Based Validation
- [ ] Model Fitting
- [ ] Posterior Predictive Check
- [ ] Model Critique
- [ ] Decision: (Pending)
