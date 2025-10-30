# Experiment 1: Standard Hierarchical Logit-Normal Model

**Model Class:** Hierarchical Bayesian (Random Effects)
**Parameterization:** Non-centered (to avoid funnel geometry)
**Status:** Validation in progress

---

## Model Specification

### Likelihood
```
r[j] ~ Binomial(n[j], inv_logit(theta[j]))
```
where:
- r[j] = number of successes in group j
- n[j] = number of trials in group j
- theta[j] = group-specific logit success rate

### Group Effects (Non-centered)
```
theta[j] = mu + tau * theta_raw[j]
theta_raw[j] ~ Normal(0, 1)
```

### Hyperpriors
```
mu ~ Normal(-2.6, 1.0)     # population mean (logit scale)
tau ~ Normal(0, 0.5)       # between-group SD (half-normal via constraint tau > 0)
```

---

## Prior Justification

**mu ~ Normal(-2.6, 1.0):**
- Centers on observed pooled rate: inv_logit(-2.6) ≈ 0.07
- SD = 1.0 allows substantial flexibility (95% CI: [0.008, 0.42] on probability scale)
- Weakly informative: constrains to plausible region but doesn't over-constrain

**tau ~ Half-Normal(0, 0.5):**
- Centers on observed between-group SD ≈ 0.02 on logit scale
- SD = 0.5 allows 0 to ~1.0 (wide range for between-group variability)
- Prevents extreme shrinkage (tau → 0) and no pooling (tau → ∞)

**Non-centered parameterization:**
- theta_raw ~ Normal(0, 1) instead of theta ~ Normal(mu, tau)
- Avoids funnel geometry when tau is small
- Improves sampling efficiency, reduces divergences

---

## Falsification Criteria

### Stage 1: Prior Predictive (will FAIL if)
- Generated success rates outside [0, 1] (impossible values)
- Extreme values frequent (e.g., > 50% samples with p < 0.01 or p > 0.99)
- Prior predictive distribution doesn't cover observed data range

### Stage 2: Simulation-Based Calibration (will FAIL if)
- Cannot recover known parameters from simulated data
- Rank statistics show non-uniformity (SBC p-value < 0.05)
- Computational pathologies in simulation (divergences, non-convergence)

### Stage 3: Posterior Inference (will FAIL if)
- Rhat > 1.01 for any parameter
- ESS < 400 for any parameter
- Divergences > 1% of samples
- Missing log_likelihood (required for LOO)

### Stage 4: Posterior Predictive (will CONCERN if)
- p-value < 0.05 (poor fit)
- Visual misfit evident (systematically over/underpredicts)
- Poor calibration (e.g., 90% intervals don't contain 90% of obs)

### Stage 5: Model Critique (will REJECT if)
- Posterior predictive p-value < 0.05
- Fundamental misspecification evident
- Better model class clearly needed (e.g., mixture if clusters dominate)

---

## Expected Outcomes

**Most Likely (70% probability):** ACCEPT
- Model converges well (non-centered helps)
- Partial pooling stabilizes small-sample estimates
- Adequate fit for homogeneous-ish heterogeneity
- Baseline for LOO comparison

**Possible (20% probability):** REVISE
- Minor prior misspecification (adjust scale)
- Convergence issues (increase adapt_delta, warmup)
- Moderate misfit suggests robustification needed

**Unlikely (10% probability):** REJECT
- Fundamental misspecification (cluster structure dominates)
- Cannot converge despite reparameterization
- Posterior predictive checks fail decisively

---

## Comparison Targets

This model will be compared against:
- **Experiment 2:** Mixture model (K=3) - tests if clusters better than continuous heterogeneity
- **Experiment 3:** Robust Student-t - tests if heavy tails needed for outliers
- **Experiment 4:** Beta-binomial - alternative parameterization

Comparison via LOO-CV (ΔLOO), posterior predictive checks, and model interpretation.

---

## Validation Pipeline Status

- [ ] Stage 1: Prior predictive check
- [ ] Stage 2: Simulation-based calibration
- [ ] Stage 3: Posterior inference (fit to data)
- [ ] Stage 4: Posterior predictive check
- [ ] Stage 5: Model critique

**Current Stage:** 1 (Prior Predictive Check)
