# Quick Reference: Non-Linear Mechanistic Models (Designer #2)

## Three Proposed Model Classes

### 1. Michaelis-Menten Saturation (PRIMARY - Recommended)

**Functional Form:**
```
Y ~ Normal(Y_min + Δ * x/(K + x), σ²)
```

**Parameters:** Y_min (baseline), Δ (range), K (half-saturation), σ (error)

**Why:**
- Classic saturation model from enzyme kinetics
- Explicit asymptote Y_max = Y_min + Δ
- Best balance of interpretability and parsimony
- 4 parameters feasible for n=27

**Falsify if:** Y_max unbounded, K at boundary, ΔWAIC > 10 vs log model

---

### 2. Hill Function / Sigmoid (SECONDARY - Exploratory)

**Functional Form:**
```
Y ~ Normal(Y_min + Δ * x^n/(K^n + x^n), σ²)
```

**Parameters:** Y_min, Δ, K, n (cooperativity), σ

**Why:**
- Generalizes MM with Hill coefficient n
- Can capture steeper transitions (n>1) or S-curves
- Tests for cooperative effects

**Falsify if:** n posterior includes 1 (reduces to MM), ΔWAIC < 2 vs MM

---

### 3. Broken-Stick with Change Point (EXPLORATORY)

**Functional Form:**
```
Y ~ Normal(α + β₁*x  if x≤τ, else  α + β₁*τ + β₂*(x-τ), σ²)
```

**Parameters:** α (intercept), β₁ (slope 1), β₂ (slope 2), τ (change point), σ

**Why:**
- EDA found 66% RSS improvement with breakpoint at x=7
- Tests sharp regime change hypothesis
- Interpretable if τ has mechanistic meaning

**Falsify if:** τ posterior uniform, β₁≈β₂, ΔWAIC > 5 vs smooth models

---

## Model Selection Decision Tree

```
1. Fit all three models
   ↓
2. Check computational diagnostics
   → If failures: Simplify priors, check data
   ↓
3. Compare WAIC/LOO-CV
   ↓
4. ΔWAIC < 3?
   → YES: Choose simplest (MM)
   → NO: Choose best WAIC
   ↓
5. Validate with PPCs and residuals
   ↓
6. Report selected model + sensitivity
```

---

## Key Priors (Informative for n=27)

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| Y_min | Normal(1.8, 0.2) | Data minimum ≈ 1.77 |
| Δ (or Y_max) | Normal(1.0, 0.5) | Range ≈ 0.95, Y_max ≈ 2.8 |
| K | Gamma(5, 0.5) | Mode ≈ 8, consistent with EDA x=7 |
| n (Hill) | Gamma(2, 1) | Mean=2, allows [0.5, 5] |
| τ (BS) | Uniform(5, 12) | EDA found τ=7 |
| σ | HalfCauchy(0, 0.2) | Expect small residuals |

---

## Falsification Summary

**Global Red Flags (abandon all models if):**
- All models fail diagnostics (Rhat > 1.05)
- All show systematic residual patterns
- All posteriors strongly conflict with priors
- All equivalent by WAIC (ΔWAIC < 2 for all)
- Predictions nonsensical (Y<0 or Y>10)

**Escape Routes:**
1. Simple log model (no asymptote)
2. Gaussian Process (non-parametric)
3. GAM with splines (flexible smooth)
4. Heteroscedastic model (if variance structure emerges)
5. Hierarchical model (if replicate structure matters)

---

## Expected Best Model Scenario

**Winner: Michaelis-Menten**
- Y_min ≈ 1.75-1.85
- Y_max ≈ 2.70-2.90
- K ≈ 8-12
- σ ≈ 0.15-0.25
- R² ≈ 0.85-0.88
- WAIC best among three, within 5 of log model

**Runner-up: Broken-stick**
- If change point is real (τ ≈ 6-9)
- May compete with MM if ΔWAIC > 3

**Unlikely Winner: Hill**
- Probably n ≈ 1 (reduces to MM)
- Penalized for extra parameter

---

## Implementation Priority

1. **Week 1:** Michaelis-Menten model
   - Code, simulate, fit, diagnose
   - Full posterior analysis and PPCs

2. **Week 2:** Hill and Broken-stick (if MM works)
   - Fit and compare to MM
   - Compute WAIC differences

3. **Week 3:** Sensitivity analysis
   - Prior sensitivity
   - Outlier sensitivity (x=31.5)
   - Model robustness checks

4. **Week 4:** Final report
   - Model selection justification
   - Parameter interpretations
   - Predictions with uncertainty
   - Recommendations

---

## Files Created

- `/workspace/experiments/designer_2/proposed_models.md` - Full technical specification (6000+ words)
- `/workspace/experiments/designer_2/model_summary.md` - This quick reference

**Next to create:**
- `mm_model.stan` - Stan implementation
- `hill_model.stan` - Stan implementation
- `bs_model.stan` - Stan implementation
- `fit_models.py` - Python fitting script
- Analysis outputs in `results/` subdirectory
