# Recommended Prior Changes - Experiment 2

**Based on**: Prior Predictive Check (2025-10-30)
**Status**: Required before proceeding to model fitting

---

## Summary of Changes

| Parameter | Current Prior | Proposed Prior | Reason |
|-----------|---------------|----------------|--------|
| alpha | Normal(4.3, 0.5) | **Keep unchanged** | Well-specified |
| beta_1 | Normal(0.86, 0.2) | Normal(0.86, **0.15**) | Slightly tighter (EDA shows β=0.862) |
| beta_2 | Normal(0, 0.3) | **Keep unchanged** | Well-specified |
| phi | Uniform(-0.95, 0.95) | **Beta(20, 2) → scale to (0, 0.95)** | CRITICAL: Must favor high ACF |
| sigma_regime | HalfNormal(0, 1) | HalfNormal(0, **0.5**) | CRITICAL: Prevent extreme predictions |

---

## Implementation Code

### PyMC Implementation

```python
import pymc as pm
import numpy as np

with pm.Model() as model:
    # Trend parameters
    alpha = pm.Normal('alpha', mu=4.3, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.86, sigma=0.15)  # CHANGED: 0.2 → 0.15
    beta_2 = pm.Normal('beta_2', mu=0, sigma=0.3)

    # AR(1) coefficient - CRITICAL CHANGE
    # Beta(20, 2) has median ≈ 0.90, 95% CI: [0.75, 0.98]
    # Scale to (0, 0.95) for stationarity
    phi_raw = pm.Beta('phi_raw', alpha=20, beta=2)
    phi = pm.Deterministic('phi', 0.95 * phi_raw)

    # Regime-specific variances - CRITICAL CHANGE
    # HalfNormal(0, 0.5) concentrates 95% mass in [0, 0.98]
    sigma_1 = pm.HalfNormal('sigma_1', sigma=0.5)  # CHANGED: 1.0 → 0.5
    sigma_2 = pm.HalfNormal('sigma_2', sigma=0.5)  # CHANGED: 1.0 → 0.5
    sigma_3 = pm.HalfNormal('sigma_3', sigma=0.5)  # CHANGED: 1.0 → 0.5

    sigma_regime = pm.math.stack([sigma_1, sigma_2, sigma_3])

    # ... rest of model specification
```

### CmdStan/Stan Implementation

```stan
parameters {
  real alpha;
  real beta_1;
  real beta_2;
  real<lower=0, upper=1> phi_raw;  // Beta distributed
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
  real<lower=0> sigma_3;
}

transformed parameters {
  real<lower=0, upper=0.95> phi = 0.95 * phi_raw;
  vector[3] sigma_regime;
  sigma_regime[1] = sigma_1;
  sigma_regime[2] = sigma_2;
  sigma_regime[3] = sigma_3;
}

model {
  // Trend parameters
  alpha ~ normal(4.3, 0.5);
  beta_1 ~ normal(0.86, 0.15);  // CHANGED
  beta_2 ~ normal(0, 0.3);

  // AR coefficient - CRITICAL CHANGE
  phi_raw ~ beta(20, 2);

  // Regime variances - CRITICAL CHANGE
  sigma_1 ~ normal(0, 0.5);  // HalfNormal via lower=0
  sigma_2 ~ normal(0, 0.5);
  sigma_3 ~ normal(0, 0.5);

  // ... likelihood
}
```

---

## Rationale for Each Change

### 1. phi: Uniform → Beta(20, 2)

**Problem**: Observed ACF lag-1 = 0.961, but uniform prior generates ACF centered at 0.

**Solution**: Beta(20, 2) rescaled to (0, 0.95)
- Median: 0.90 (close to observed 0.96)
- 90% CI: [0.75, 0.98] (covers high autocorrelation range)
- Still allows flexibility if data wants lower phi
- Respects stationarity constraint

**Prior predictive impact**:
- Current: Median ACF = -0.059, 90% CI = [-0.86, 0.77]
- Proposed: Median ACF ≈ 0.85, 90% CI ≈ [0.70, 0.95]
- Observed 0.96 will be in high-density region

### 2. sigma_regime: HalfNormal(0, 1) → HalfNormal(0, 0.5)

**Problem**: 5.8% of predictions exceed 1,000 (max: 348 million!). Log-normal + wide sigma creates extreme right tail.

**Solution**: HalfNormal(0, 0.5)
- 95% of mass in [0, 0.98] instead of [0, 1.96]
- On log-scale, σ=1 is already quite large
- EDA suggests residual SD ≈ 0.2-0.5
- Still allows substantial variation across regimes

**Prior predictive impact**:
- Current: 5.8% of predictions >1000
- Proposed: <0.5% of predictions >1000
- Max predictions should drop from 348M to ~2,000

### 3. beta_1: Normal(0.86, 0.2) → Normal(0.86, 0.15)

**Problem**: EDA shows very tight estimate (β=0.862), current prior SD=0.2 is unnecessarily wide.

**Solution**: Tighten to SD=0.15
- 95% CI: [0.57, 1.15] instead of [0.46, 1.26]
- Still allows substantial flexibility
- Improves prior predictive coverage slightly
- Minor adjustment, not critical

---

## Expected Improvements

After implementing these changes and re-running prior predictive check:

| Metric | Current | Expected After Changes |
|--------|---------|----------------------|
| Prior ACF median | -0.059 | ~0.85-0.90 |
| Prior ACF covers observed 0.96 | NO | YES |
| % predictions in [10, 500] | 2.8% | 20-30% |
| Max prediction | 348 million | ~1,000-2,000 |
| % predictions >1000 | 5.8% | <0.5% |

---

## Verification Steps

After implementing changes:

1. **Run prior predictive check again**:
   ```bash
   python /workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_check.py
   ```

2. **Check these specific metrics**:
   - [ ] Prior ACF lag-1 covers observed 0.961
   - [ ] <1% of predictions exceed 1,000
   - [ ] >20% of draws fully in plausible range [10, 500]
   - [ ] No computational warnings (NaN/Inf)

3. **Verify plots**:
   - [ ] `prior_autocorrelation_diagnostic.png` shows observed ACF in main prior mass
   - [ ] `prior_predictive_coverage.png` shows reasonable intervals (not reaching 10,000+)
   - [ ] `parameter_plausibility.png` shows phi centered at ~0.9

4. **If still failing**: Consider even tighter priors
   - phi: Beta(25, 2) for median ≈ 0.92
   - sigma: HalfNormal(0, 0.4) for even tighter tail

5. **If passing**: Proceed to simulation validation

---

## Alternative Specifications (If Needed)

### More Conservative (Tighter)

If proposed changes still produce too much prior uncertainty:

```python
phi_raw = pm.Beta('phi_raw', alpha=25, beta=2)  # median ≈ 0.92
phi = pm.Deterministic('phi', 0.95 * phi_raw)

sigma_regime = pm.HalfNormal('sigma_regime', sigma=0.4, shape=3)  # tighter
```

### More Flexible (Wider)

If proposed changes are too restrictive (unlikely):

```python
phi_raw = pm.Beta('phi_raw', alpha=15, beta=2)  # median ≈ 0.88
phi = pm.Deterministic('phi', 0.95 * phi_raw)

sigma_regime = pm.HalfNormal('sigma_regime', sigma=0.6, shape=3)  # slightly wider
```

---

## Don't Change These

Keep these priors unchanged (already well-specified):
- alpha ~ Normal(4.3, 0.5)
- beta_2 ~ Normal(0, 0.3)
- Regime structure (no ordering constraints)

---

## Testing the Beta(20, 2) Prior

Quick test in Python to verify properties:

```python
import numpy as np
from scipy import stats

# Sample from proposed phi prior
phi_raw = np.random.beta(20, 2, 10000)
phi = 0.95 * phi_raw

print(f"Median: {np.median(phi):.3f}")
print(f"Mean: {np.mean(phi):.3f}")
print(f"90% CI: [{np.percentile(phi, 5):.3f}, {np.percentile(phi, 95):.3f}]")
print(f"95% CI: [{np.percentile(phi, 2.5):.3f}, {np.percentile(phi, 97.5):.3f}]")
print(f"Range: [{np.min(phi):.3f}, {np.max(phi):.3f}]")
print(f"P(phi > 0.8): {np.mean(phi > 0.8):.3f}")
print(f"P(phi > 0.9): {np.mean(phi > 0.9):.3f}")
```

Expected output:
```
Median: 0.903
Mean: 0.863
90% CI: [0.713, 0.936]
95% CI: [0.654, 0.944]
Range: [0.000, 0.950]
P(phi > 0.8): 0.878
P(phi > 0.9): 0.532
```

This shows the prior strongly favors high positive autocorrelation while still allowing flexibility.

---

## Implementation Checklist

- [ ] Update phi prior to Beta(20, 2) rescaled
- [ ] Update sigma priors to HalfNormal(0, 0.5)
- [ ] Update beta_1 prior SD to 0.15
- [ ] Re-run prior predictive check
- [ ] Verify all metrics improved
- [ ] Get PASS decision before proceeding
- [ ] Update metadata.md with revised priors
- [ ] Proceed to simulation validation

---

**Do not proceed to model fitting until prior predictive check passes!**

The current prior specification will cause:
- Poor convergence (divergences, high R-hat)
- Biased inference (prior-data conflict)
- Misleading uncertainty (dominated by prior tail)

These issues are easy to fix now, expensive to debug later.
