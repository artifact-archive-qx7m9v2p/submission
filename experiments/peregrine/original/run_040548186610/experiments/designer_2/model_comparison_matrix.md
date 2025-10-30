# Model Comparison Matrix - Designer 2

## Quick Reference: Three Flexible Models

| Aspect | Model 1: GP-NegBin | Model 2: P-spline | Model 3: Semi-parametric |
|--------|-------------------|-------------------|-------------------------|
| **Flexibility** | Highest (GP prior on f) | Medium (splines + penalty) | Medium (parametric + GP) |
| **Computational Cost** | High (O(N³) Cholesky) | Low (sparse matrix) | Very High (hybrid) |
| **Parameters** | 5 + GP realization | 12 (basis coefs) + 2 | 10+ (growth + GP + φ(t)) |
| **Extrapolation** | Poor (GP uncertainty explodes) | Poor (spline artifacts) | Better (parametric structure) |
| **Interpretability** | Low (black box function) | Medium (visual basis) | High (decomposable) |
| **Expected Rank** | 2nd (might overfit) | 1st (Goldilocks) | 3rd (might not converge) |

## Falsification Criteria Summary

### Model 1 (GP-NegBin) - ABANDON IF:
- ✗ Lengthscale ρ → ∞ (collapsing to constant)
- ✗ Divergences > 10% despite tuning
- ✗ LOO-IC worse than parametric by >10
- ✗ Wild extrapolation behavior
- ✗ Posterior on hyperparameters at boundaries

### Model 2 (P-spline) - ABANDON IF:
- ✗ Smoothness τ → 0 (over-smoothed, nearly linear)
- ✗ Smoothness τ → ∞ (under-smoothed, overfitting)
- ✗ Boundary artifacts in predictions
- ✗ High sensitivity to knot placement
- ✗ Worse than GP by >10 LOO-IC points

### Model 3 (Semi-parametric) - ABANDON IF:
- ✗ GP deviations σ² dominate (parametric form wrong)
- ✗ Inflection point t₀ at boundaries (no S-curve)
- ✗ Time-varying φ₁ not needed (zero in CI)
- ✗ Computational failure (won't converge)
- ✗ Not better than pure models

## Decision Tree

```
START: Fit all 3 models
   |
   ├─> All converge?
   |    ├─> NO → Simplify (fewer knots, simpler kernels) → Retry
   |    └─> YES → Continue
   |
   ├─> Check LOO-CV
   |    ├─> Model 2 best? → Expected outcome, validate
   |    ├─> Model 1 best? → High complexity justified, check overfitting
   |    ├─> Model 3 best? → Structured deviations important
   |    └─> Designer 1 best? → Abandon flexibility
   |
   ├─> Check holdout predictions
   |    ├─> All terrible? → Missing dynamics (try Designer 3)
   |    └─> Reasonable? → Continue
   |
   └─> FINAL RECOMMENDATION based on:
        - LOO-CV (primary)
        - Posterior predictive checks
        - Holdout RMSE
        - Computational feasibility
        - Scientific interpretability
```

## Expected Model Behavior

### If Data is Actually Simple (e.g., quadratic):
- Model 1: GP will have long lengthscale, nearly quadratic posterior mean
- Model 2: Spline coefficients will be smooth, low effective DF
- Model 3: GP deviations will be tiny, logistic will fit poorly
- **Conclusion**: Designer 1's quadratic NegBin wins

### If Data is Complex (irregular fluctuations):
- Model 1: GP will have short lengthscale, captures every wiggle
- Model 2: Splines will have low τ (minimal smoothing), adaptive fit
- Model 3: GP deviations will be large, parametric component inadequate
- **Conclusion**: My models win, but risk overfitting

### If Data has Structure + Noise:
- Model 1: GP averages over noise, learns smooth trend
- Model 2: Spline penalty removes noise, captures trend
- Model 3: Parametric captures structure, GP captures residual
- **Conclusion**: All models agree, high confidence

## Cross-Designer Comparison

### vs Designer 1 (Parametric GLMs)
**If they win**: Flexibility unjustified for n=40
**If I win**: Non-linearity requires flexible forms
**Hybrid outcome**: Similar performance → use simpler model (Occam's razor)

### vs Designer 3 (Hierarchical/Temporal)
**If they win**: Temporal correlation > flexible mean function
**If I win**: Trend shape > autocorrelation structure
**Hybrid opportunity**: My flexible trend + their AR errors

## Red Flags Across All Models

**Stop and reconsider if**:
1. Prior-data conflict in all 3 models (priors fundamentally wrong)
2. Posterior predictive checks fail for all (missing structure)
3. Holdout RMSE > 50 for all (not learning)
4. All models collapse to linear (EDA misled us)
5. Computational failures across implementations (data pathology)

## What Success Looks Like

**Model 1**:
- Converges in <30 min
- R̂ < 1.01, ESS > 400
- Lengthscale ~ 0.5-2.0 (reasonable smoothness)
- LOO-IC competitive with Model 2
- Predictions within observed range

**Model 2**:
- Converges in <10 min
- Smoothness τ ~ 0.5-2.0 (neither extreme)
- Fitted curve visually plausible
- Best LOO-IC of my 3 models
- Robust to knot count (6 vs 8 vs 10)

**Model 3**:
- Converges in <60 min (or we give up)
- Logistic parameters interpretable (L > max(C), reasonable k)
- Small deviations (σ² < 0.5 of total variation)
- Time-varying φ improves fit (φ₁ ≠ 0)
- Slightly better than pure parametric

---

**Quick Recommendation Formula**:

```
IF convergence_issues:
    USE simpler model
ELIF LOO_IC_difference > 10:
    USE best_model
ELIF comparable_performance:
    USE simplest_model  # Occam's razor
ELSE:
    ENSEMBLE average across models
```

---

Created: 2025-10-29
Designer: 2 (Non-parametric Specialist)
