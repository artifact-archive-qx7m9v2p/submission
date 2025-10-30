# Model Decision Tree & Comparison Strategy
## Quick Reference for Model Selection

---

## Decision Flowchart

```
START: Fit Model 2A (Quadratic Polynomial, Constant Dispersion)
│
├─→ [Q1] R² > 0.95 AND residuals look good?
│   ├─ YES → STOP. Use polynomial. Don't overfit!
│   └─ NO → Continue to Q2
│
├─→ [Q2] Residuals show clear regime structure?
│   ├─ YES → Fit Model 1A (Piecewise Changepoint)
│   │   │
│   │   ├─→ [Q3] Posterior tau concentrated?
│   │   │   ├─ YES → Strong changepoint evidence
│   │   │   │   └─→ [Q4] LOO improves by >4 ELPD?
│   │   │   │       ├─ YES → Use changepoint model
│   │   │   │       └─ NO → Use polynomial (simpler)
│   │   │   │
│   │   │   └─ NO → Changepoint not supported
│   │   │       └─→ Try Model 3A (Two-State Hierarchical)
│   │   │
│   └─ NO → Try time-varying dispersion (Model 2B)
│       │
│       └─→ [Q5] Time-varying phi justified?
│           ├─ YES → Use polynomial + varying phi
│           └─ NO → Consider Model 3A
│
└─→ [Q6] If all simple models fail:
    ├─ Try Model 3A (Two-State Hierarchical)
    ├─ Check for data quality issues
    ├─ Consider alternative likelihood families
    └─ Consult domain experts
```

---

## Model Complexity Ladder

**Level 1: SIMPLE** (Fit first, stop if adequate)
- Model 2A: Quadratic polynomial, constant dispersion
- Parameters: 4 (beta0, beta1, beta2, phi)
- Expected runtime: 2-5 minutes

**Level 2: MODERATE** (Fit if Level 1 inadequate)
- Model 2B: Quadratic polynomial, time-varying dispersion
  - Parameters: 5 (+ gamma1 for dispersion trend)
- Model 1A: Piecewise linear, constant dispersion per regime
  - Parameters: 6 (beta0, beta1_before, beta1_after, phi_before, phi_after, tau)
- Expected runtime: 5-15 minutes each

**Level 3: COMPLEX** (Fit only if strong evidence for structure)
- Model 1B: Piecewise quadratic, varying dispersion
  - Parameters: 9
- Model 3A: Two-state hierarchical
  - Parameters: 10+ (with hyperpriors)
- Expected runtime: 20-60 minutes each

**Level 4: ADVANCED** (Last resort)
- Model 3B: Three-state hierarchical
- Model 4: Gaussian Process (not specified, escape route)
- Expected runtime: 1-4 hours

---

## Stopping Rules

### STOP and use simpler model if:
- Simple model R² > 0.95 AND passes diagnostics
- Complex model LOO within 4 ELPD of simple model
- Complex model has >10% high Pareto-k values
- Complex model has computational issues (divergences, low ESS)

### STOP entire approach if:
- All models fail posterior predictive checks
- All models have LOO Pareto-k > 0.7 for >50% observations
- Parameter estimates are extreme/implausible across all models
- Prior sensitivity shows conclusions are fragile

---

## Model Comparison Matrix

| Model | Pros | Cons | When to Use |
|-------|------|------|-------------|
| **2A: Polynomial** | Simple, fast, interpretable | May miss regime shifts | First try, baseline |
| **2B: Poly + Varying φ** | Handles heterogeneous variance | Still assumes smooth trend | If dispersion changes smoothly |
| **1A: Changepoint** | Tests discrete shift directly | Strong assumption | If visual evidence of "elbow" |
| **1B: Changepoint + Quad** | Most flexible regime model | Many parameters (n=40!) | If both shift and nonlinearity |
| **3A: Two-State** | Soft transitions, hierarchical | Complex, identifiability risk | If uncertain about discrete vs smooth |
| **3B: Three-State** | Matches 3 EDA periods | High overfitting risk | Only if two-state clearly fails |

---

## Diagnostic Checklist

### Before Interpreting Results:
- [ ] Rhat < 1.01 for all parameters?
- [ ] ESS > 400 for key parameters?
- [ ] < 1% divergent transitions?
- [ ] Trace plots show good mixing?
- [ ] Posterior differs from prior (learning occurred)?

### Model-Specific Checks:
- [ ] **Changepoint**: Posterior tau concentrated (not diffuse)?
- [ ] **Polynomial**: Coefficients not extreme (|beta3| < 1)?
- [ ] **Hierarchical**: States separable (not uniform probabilities)?
- [ ] **Time-varying phi**: Credible interval for gamma1 excludes zero?

### Comparison Checks:
- [ ] LOO computed successfully (Pareto-k < 0.7 for >90%)?
- [ ] Best model clearly better (ΔLOO > 4 ELPD)?
- [ ] Posterior predictive checks pass (p ∈ [0.05, 0.95])?
- [ ] Results robust to prior perturbations?

---

## Red Flags & Immediate Actions

| Red Flag | Interpretation | Action |
|----------|---------------|--------|
| Many divergences (>5%) | Poor geometry | Increase adapt_delta to 0.99 |
| Rhat > 1.05 | Poor convergence | Run longer chains (4000+ iter) |
| ESS < 100 | High autocorrelation | Check for funnels, reparameterize |
| Posterior at prior boundary | Prior-data conflict | Revise prior or check likelihood |
| Pareto-k > 0.7 for >25% | Poor generalization | Model overfitting or misspecified |
| Uniform state posteriors | No latent structure | Use simpler model |
| Extreme predictions | Model unstable | Check parameterization |

---

## Prior Sensitivity Protocol

For final selected model, refit with:

1. **Tighter priors** (halve all SDs)
   - If results change substantially → Not enough data

2. **Looser priors** (double all SDs)
   - If results change substantially → Prior-dominated

3. **Alternative prior families**
   - Student-t instead of Normal (heavier tails)
   - If results change → Assess which is more reasonable

**Interpretation**:
- Robust = conclusions consistent across prior choices
- Sensitive = report uncertainty, don't overinterpret

---

## Expected Outcomes

### Most Likely Scenario (70% probability):
- Polynomial (Model 2A or 2B) is adequate
- R² ≈ 0.96, clean residuals
- Time-varying dispersion may be needed
- **Conclusion**: Smooth acceleration, no strong changepoint

### Alternative Scenario 1 (20% probability):
- Clear changepoint around observation 20-22
- Posterior tau concentrated (SD ≈ 2-3 observations)
- Regime-specific parameters differ substantially
- **Conclusion**: Discrete structural break supported

### Alternative Scenario 2 (8% probability):
- Two-state hierarchical model wins
- Soft transition, not discrete
- States interpretable (low/high growth)
- **Conclusion**: Gradual regime change

### Unexpected Scenario (2% probability):
- All models fail diagnostics
- Need to reconsider data quality or likelihood family
- **Conclusion**: Pivot to alternative approach

---

## Time Budget (Estimated)

| Task | Hours | Cumulative |
|------|-------|------------|
| Implement Model 2A (polynomial) | 2 | 2 |
| Diagnostics & PPC for 2A | 2 | 4 |
| Implement Model 1A (changepoint) | 4 | 8 |
| Diagnostics & comparison | 3 | 11 |
| Implement Model 2B (varying phi) | 2 | 13 |
| Implement Model 3A (hierarchical) | 6 | 19 |
| Debugging & convergence tuning | 4 | 23 |
| LOO comparison & sensitivity | 3 | 26 |
| Final validation & write-up | 4 | 30 |
| **TOTAL** | **30** | - |

**Contingency**: +10 hours if major issues arise

---

## Communication Protocol

### After Each Model:
Report to main agent:
1. Convergence diagnostics (Rhat, ESS, divergences)
2. LOO-ELPD with SE
3. Key posterior summaries
4. Any red flags or concerns
5. **Recommendation**: Continue, stop, or pivot?

### Final Report:
Include:
1. Model comparison table (LOO, WAIC, effective parameters)
2. Best model specification with posterior intervals
3. Posterior predictive check plots
4. Uncertainty quantification (prediction intervals)
5. **Honest assessment** of limitations
6. Sensitivity analysis results

---

## Key Reminders

1. **Simplicity first**: Don't fit complex models unless needed
2. **Diagnostics always**: Never interpret poor-quality samples
3. **Uncertainty matters**: Point estimates alone are misleading
4. **Prior transparency**: Document and justify all priors
5. **Honest reporting**: Report failures and limitations
6. **Scientific humility**: We're seeking truth, not confirming beliefs

---

**This document is a living guide. Update as you learn from the data.**
