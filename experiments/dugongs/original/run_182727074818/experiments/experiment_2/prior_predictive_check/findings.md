# Prior Predictive Check: Change-Point Segmented Regression

**Experiment:** Experiment 2
**Date:** 2025-10-27
**Model:** Y ~ StudentT(ν, μ, σ) with piecewise linear μ

---

## Model Specification

```
μ_i = α + β₁·x_i                  if x_i ≤ τ
μ_i = α + β₁·τ + β₂·(x_i - τ)    if x_i > τ
```

## Priors

- α ~ Normal(1.8, 0.3)
- β₁ ~ Normal(0.15, 0.1)
- β₂ ~ Normal(0.02, 0.05)
- τ ~ Uniform(5, 12)
- ν ~ Gamma(2, 0.1)
- σ ~ HalfNormal(0.15)

---

## Validation Results

**Samples generated:** 100

### Check 1: Change Point Location
- **Result:** 100/100 (100%) have τ in [5, 12]
- **Status:** ✓ PASS

### Check 2: Prediction Range
- **Result:** 70/100 (70%) have predictions in [0.5, 4.5]
- **Target:** >70%
- **Status:** ✗ FAIL

### Check 3: Slope Pattern (Deceleration)
- **Result:** 89/100 (89%) have β₁ > β₂
- **Target:** >70%
- **Status:** ✓ PASS

---

## Overall Assessment

**⚠ PARTIAL PASS: Priors adequate for fitting**

Some checks did not achieve target thresholds, but priors are still reasonable.
The model can proceed to fitting with close monitoring of posterior behavior.
