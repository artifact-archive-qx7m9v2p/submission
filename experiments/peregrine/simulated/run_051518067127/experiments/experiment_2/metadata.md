# Experiment 2: AR(1) Log-Normal with Regime-Switching

**Model Class**: Transformed continuous with autoregressive errors
**Priority**: Tier 1 (MUST attempt)
**Date Started**: 2025-10-30
**Motivation**: Experiment 1 REJECTED due to residual ACF=0.596, cannot capture temporal autocorrelation

## Model Specification

### Likelihood
```
C[t] ~ LogNormal(mu[t], sigma_regime[regime[t]])
```

### Mean Structure on Log-Scale
```
mu[t] = alpha + beta_1 * year[t] + beta_2 * year[t]^2 + phi * epsilon[t-1]
```

### Autoregressive Error Structure
```
epsilon[t] = log(C[t]) - (alpha + beta_1 * year[t] + beta_2 * year[t]^2)
epsilon[1] ~ Normal(0, sigma_regime[1] / sqrt(1 - phi^2))  # Stationary initialization
```

### Regime Structure (Known from EDA)
```
regime[1:14] = 1   # Early period
regime[15:27] = 2  # Middle period
regime[28:40] = 3  # Late period
```

### Parameters and Priors

**Updated after Prior Predictive Check (initial priors had ACF mismatch)**:
```
alpha ~ Normal(4.3, 0.5)              # Log-scale intercept
beta_1 ~ Normal(0.86, 0.15)           # Linear growth (tightened from 0.2)
beta_2 ~ Normal(0, 0.3)               # Quadratic term
phi_raw ~ Beta(20, 2)                 # Raw autocorrelation (median ~0.9)
phi = 0.95 * phi_raw                  # AR(1) coefficient, scaled to (0, 0.95)
sigma_regime[1:3] ~ HalfNormal(0, 0.5)  # Regime-specific SD (tightened from 1.0)
```

**Rationale for changes**:
- **phi**: Changed from Uniform(-0.95, 0.95) to Beta(20,2) scaled to (0, 0.95)
  - Data has ACF lag-1 = 0.961, prior should encode high positive autocorrelation
  - Beta(20,2) gives median ≈ 0.90, covers observed 0.96 in tails
- **sigma_regime**: Changed from HalfNormal(0, 1) to HalfNormal(0, 0.5)
  - Prevents extreme predictions (original had 5.8% > 1000)
  - On log-scale, sigma=0.5 is generous for residual variation
- **beta_1**: Tightened from 0.2 to 0.15
  - EDA strongly suggests beta_1 ≈ 0.86, reduce prior variance
```

## Rationale

This model addresses the critical failure of Experiment 1 by:

1. **AR(1) structure** → Captures temporal autocorrelation (ACF lag-1 = 0.971 in data)
2. **Regime-specific variance** → Handles heterogeneous dispersion across time periods
3. **Log-transformation** → Leverages excellent log-scale fit (R²=0.937 from EDA)
4. **Quadratic trend** → Allows for acceleration/deceleration

The model assumes:
- Exponential growth on original scale (multiplicative on log-scale)
- Stationary AR(1) process for errors (|phi| < 1)
- Three distinct regimes with different variance structures
- Errors are normally distributed after accounting for AR structure

## Key Differences from Experiment 1

| Feature | Experiment 1 (NB) | Experiment 2 (AR Log-Normal) |
|---------|------------------|------------------------------|
| Scale | Count scale | Log scale |
| Likelihood | Negative Binomial | Log-Normal |
| Overdispersion | Via phi parameter | Via log-transformation + regime variance |
| Temporal dependence | **None (independent)** | **AR(1) structure** |
| Variance structure | Homogeneous | Regime-specific |

## Falsification Criteria

**I will abandon this model if**:
- Residual ACF lag-1 > 0.3 after fitting (AR(1) insufficient)
- All sigma_regime posteriors overlap >80% (no regime effect)
- phi posterior centered near 0 (no autocorrelation benefit)
- Back-transformed predictions systematically biased (>20% error)
- Worse LOO-CV than Experiment 1
- Convergence failures (R-hat > 1.05)

## Expected Outcomes

**Most likely**:
- Excellent fit with residual ACF < 0.3
- phi posterior around 0.6-0.8 (high positive autocorrelation)
- sigma_regime ordering: Middle > Late > Early (based on EDA dispersion patterns)
- Better LOO-CV than Experiment 1 (ΔELPD > 10)

**If succeeds**: Likely ACCEPT (addresses main limitation of Exp 1)

**If fails**:
- If phi ≈ 0 → autocorrelation was artifact, reconsider Exp 1
- If residual ACF still high → need AR(p) with p>1, or GP
- If regime variances homogeneous → simplify to single variance

## Implementation Notes

**Software**: PyMC (CmdStan unavailable)

**Back-transformation**:
- For mean predictions: `exp(mu + sigma²/2)` (bias correction)
- For median predictions: `exp(mu)`
- For credible intervals: Transform on log scale, then exponentiate

**AR(1) Initialization**:
- First observation: epsilon[1] ~ Normal(0, sigma_regime[1] / sqrt(1 - phi^2))
- Ensures stationarity at t=1

**Computational Considerations**:
- AR structure introduces sequential dependence → may slow sampling
- Non-centered parameterization may help if needed
- Expected runtime: 2-5 minutes

**Expected Comparison to Experiment 1**:
- Similar mean trend fit (both R² ≈ 0.94)
- Much better residual diagnostics (ACF near 0)
- Better posterior predictive performance on autocorrelation tests
- Improved LOO-CV (should favor temporal structure)
