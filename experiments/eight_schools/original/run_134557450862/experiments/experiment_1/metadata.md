# Experiment 1: Standard Non-Centered Hierarchical Model

**Model Type:** Bayesian Hierarchical Meta-Analysis
**Parameterization:** Non-centered
**Probabilistic Programming Language:** PyMC
**Status:** In Progress

---

## Model Specification

### Mathematical Formulation

```
Data Model:
  y_i ~ Normal(theta_i, sigma_i)    for i = 1, ..., 8
  where sigma_i are KNOWN measurement errors

Hierarchical Structure (Non-Centered Parameterization):
  theta_i = mu + tau * eta_i
  eta_i ~ Normal(0, 1)               [standardized random effects]

Priors:
  mu ~ Normal(0, 20)                 [weakly informative grand mean]
  tau ~ Half-Cauchy(0, 5)            [Gelman 2006 recommendation]
```

### Rationale

**Non-Centered Parameterization:**
- Decorrelates mu and tau in posterior (better sampling geometry)
- Prevents funnel-shaped posterior when tau is small
- Critical for this dataset where tau expected near 0

**Half-Cauchy(0, 5) Prior on tau:**
- Heavy tails allow large tau if data support it
- Scale=5 appropriate for effect size scale
- Does NOT force tau toward zero (unlike half-normal)
- Gelman et al. (2006) recommendation for hierarchical models

**Normal(0, 20) Prior on mu:**
- Weakly informative: covers [-40, 40] at 2 SD
- Observed range [-3, 28] well within prior support
- Data are informative (pooled SE=4.07), so prior has modest influence

### Expected Behavior (Given EDA)

**Posterior predictions:**
- **mu:** Concentrated around 7.7 (pooled estimate), SD ≈ 4-5
- **tau:** Mode near 0-2, median ≈ 2-4, heavy right tail
  - 95% CI likely [0, 8-10]
  - Substantial uncertainty in small-sample regime
- **theta_i:** Strong shrinkage (70-90%) toward mu
  - School 1 (y=28): shrink from 28 → ~8-12
  - School 5 (y=-1): shrink from -1 → ~5-7
  - High-precision schools shrink less than low-precision schools

**Shrinkage formula:**
```
Shrinkage_i = 1 - [tau^2 / (tau^2 + sigma_i^2)]
```
With tau ≈ 3 and sigma_i ≈ 9-18, expect 60-90% shrinkage.

---

## Falsification Criteria

This model will be ABANDONED if:

1. **Computational failure:**
   - Divergent transitions > 1% despite tuning
   - R-hat > 1.01 for any parameter
   - ESS < 100 for tau
   - **Why:** Non-centered should be robust; failure indicates misspecification

2. **Prior-posterior conflict for tau:**
   - Posterior median tau > 15 (exceeds observed SD=10.4)
   - Posterior entirely at tau=0 (model degenerate)
   - **Why:** If tau >> observed SD, data don't support hierarchy; if tau ≡ 0, use simpler model

3. **Shrinkage inconsistencies:**
   - High-precision schools shrink MORE than low-precision schools
   - **Why:** Violates fundamental shrinkage logic

4. **Posterior predictive failure:**
   - < 90% of observed y_i fall within posterior predictive 95% intervals
   - **Why:** Model should easily capture observed data given large sigma_i

5. **Extreme parameter values:**
   - Any theta_i posterior mean outside [-10, 20]
   - **Why:** Schools should shrink toward pooled mean, not away

---

## Implementation Details

**Sampling Strategy:**
- 4 chains, 2000 iterations each, 1000 warmup
- Target accept_prob = 0.95 (conservative for potential boundary issues)
- NUTS sampler (PyMC default)

**Computational Expectations:**
- Runtime: 2-5 minutes
- Memory: < 1 GB
- Convergence: Should be straightforward for this simple model

**Diagnostics to Monitor:**
- Trace plots for mu, tau, eta (check mixing)
- Pair plots for (mu, tau) - check for funnel (shouldn't be present with non-centered)
- ESS for all parameters (target > 400)
- R-hat for all parameters (target < 1.01)
- Divergence count (target 0)
- Energy plots (check HMC geometry)

---

## Validation Pipeline

1. ✅ **Prior Predictive Check:** Validate prior generates reasonable data
2. ✅ **Simulation-Based Validation:** Test parameter recovery with known parameters
3. ⏳ **Posterior Inference:** Fit to actual data with convergence diagnostics
4. ⏳ **Posterior Predictive Check:** Validate model captures observed data
5. ⏳ **Model Critique:** Assess adequacy and make ACCEPT/REVISE/REJECT decision

---

## References

- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. Bayesian Analysis, 1(3), 515-534.
- Betancourt, M., & Girolami, M. (2015). Hamiltonian Monte Carlo for hierarchical models. Current Trends in Bayesian Methodology with Applications, 79, 30.

---

**Created:** 2025-10-28
**Last Updated:** 2025-10-28
