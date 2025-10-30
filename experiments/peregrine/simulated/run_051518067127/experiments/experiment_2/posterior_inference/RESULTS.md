# Experiment 2: AR(1) Log-Normal Posterior Inference Results

**Status**: ✅ CONVERGED | ❌ RESIDUAL ACF FAILED
**Date**: 2025-10-30
**Decision**: **DO NOT PROCEED TO PPC** - AR(1) insufficient for temporal structure

---

## Quick Summary

| Aspect | Result | Status |
|--------|--------|--------|
| **Convergence** | R-hat=1.00, ESS>5000, 0 divergences | ✅ EXCELLENT |
| **Residual ACF** | 0.611 (target <0.3) | ❌ FAILED |
| **vs Experiment 1** | ACF worse (0.611 vs 0.596) | ❌ NO IMPROVEMENT |
| **Point Predictions** | MAE=13.99, RMSE=20.12 | ✅ Better than Exp 1 |
| **LOO-CV Ready** | posterior_inference.netcdf saved | ✅ YES |

**Critical Finding**: AR(1) model converged perfectly but **failed primary objective** - residual autocorrelation (0.611) is HIGHER than Experiment 1 (0.596), indicating AR(1) structure is insufficient.

---

## Parameter Estimates

### Core Parameters

```
α (intercept):     4.342 ± 0.257  [3.85, 4.83]₉₄
β₁ (linear):       0.808 ± 0.110  [0.60, 1.01]₉₄
β₂ (quadratic):    0.015 ± 0.125  [-0.21, 0.26]₉₄  ← Weakly identified
φ (AR coefficient): 0.847 ± 0.061  [0.74, 0.94]₉₄  ← Strong but insufficient
```

**Key Insight**: φ = 0.847 << data ACF (0.975), suggesting AR(1) cannot explain full autocorrelation.

### Regime Variances (Log Scale)

```
σ₁ (Early):   0.239 ± 0.053  [0.15, 0.34]₉₄  ← Highest
σ₂ (Middle):  0.207 ± 0.047  [0.13, 0.29]₉₄
σ₃ (Late):    0.169 ± 0.040  [0.10, 0.24]₉₄  ← Lowest
```

**Ordering**: σ₁ > σ₂ > σ₃ (unexpected - EDA suggested σ₂ > σ₃ > σ₁)

---

## Convergence Diagnostics

**Sampling**: 4 chains × 2000 draws, 1500 warmup, NUTS with target_accept=0.90

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max R-hat | 1.0000 | <1.01 | ✅ |
| Min ESS bulk | 5,042 | >400 | ✅ |
| Min ESS tail | 4,099 | >400 | ✅ |
| Divergences | 0 | <2% | ✅ (0.00%) |
| MCSE/SD ratio | <0.05 | <0.05 | ✅ |

**Visual Diagnostics** (`plots/trace_plots.png`): Perfect chain mixing, no issues.

---

## Residual Diagnostics (CRITICAL)

### Autocorrelation Test (PRIMARY FAILURE)

```
Residual ACF lag-1:  0.611  ← FAILS (target <0.3)
Data ACF lag-1:      0.975  ← Reference
Experiment 1 ACF:    0.596  ← Baseline
```

**Result**: AR(1) provides **NO IMPROVEMENT** - actually slightly worse than Experiment 1!

### ACF Plot Analysis (`plots/residual_diagnostics.png`)

- Lag-1 ACF far exceeds target threshold (0.3)
- Multiple higher lags also significant
- Suggests need for AR(2) or higher-order structure

### Other Residual Checks

- **Normality**: Q-Q plot shows good central fit, slight tail deviation (acceptable)
- **Homoscedasticity**: Mild funnel pattern (expected with regime variance)
- **Time trend**: No systematic pattern (good)

---

## Fit Quality

### In-Sample Performance

| Metric | Exp 2 (AR Log-Normal) | Exp 1 (NB Quadratic) | Change |
|--------|----------------------|---------------------|---------|
| MAE | **13.99** | 16.41 | -15% ✅ |
| RMSE | **20.12** | 26.12 | -23% ✅ |
| Bayesian R² | **0.952** | 0.939 | +1.4% ✅ |

**Interpretation**: Better point predictions but temporal structure still inadequate.

### Predictive Coverage

- 90% credible intervals cover ~88% of observations
- No systematic over/under-prediction visible
- See `plots/fitted_trend.png` for visual assessment

---

## Key Plots

### 1. Residual ACF (`plots/residual_diagnostics.png`)
**Shows**: Lag-1 ACF = 0.611 far exceeds target (0.3)
**Implication**: AR(1) insufficient

### 2. AR Coefficient (`plots/ar_coefficient.png`)
**Shows**: φ posterior (0.847) well below data ACF (0.975)
**Implication**: Model cannot match observed correlation with AR(1) alone

### 3. Fitted Trend (`plots/fitted_trend.png`)
**Shows**: Excellent median fit, appropriate uncertainty
**Implication**: Good central tendency despite temporal issues

### 4. Trace Plots (`plots/trace_plots.png`)
**Shows**: Perfect convergence across all parameters
**Implication**: Computational success, not model success

### 5. Regime Posteriors (`plots/regime_posteriors.png`)
**Shows**: Well-separated variance posteriors
**Implication**: Regime structure validated

---

## Comparison to Experiment 1

| Feature | Exp 1 (NB) | Exp 2 (AR Log-Normal) | Winner |
|---------|------------|---------------------|--------|
| Temporal structure | None | AR(1) | Exp 2 |
| Residual ACF | 0.596 | **0.611** | **Exp 1** ⚠️ |
| MAE | 16.41 | **13.99** | Exp 2 |
| RMSE | 26.12 | **20.12** | Exp 2 |
| Convergence | Excellent | Excellent | Tie |

**Verdict**: Experiment 2 has better fit metrics but **worse temporal diagnostics** - paradoxical!

**Possible Explanation**:
- AR structure improves point predictions by smoothing
- But AR(1) is too weak - leaves more residual correlation than baseline
- Need AR(2+) to properly capture dependencies

---

## Why AR(1) Failed

### Gap Between Model and Data

```
Data ACF lag-1:        0.975
Model AR coefficient:  0.847
Residual ACF:          0.611  ← Large remaining correlation
```

### Interpretation

1. **AR(1) is real but weak**: φ = 0.847 is significantly > 0
2. **AR(1) is insufficient**: Cannot explain ACF = 0.975
3. **Higher-order needed**: Residual ACF (0.611) suggests AR(2) or AR(3)

### Mathematical Insight

For stationary AR(1): ρ(lag-1) ≈ φ = 0.847
But data shows: ρ(lag-1) = 0.975
Gap = 0.128 → appears as residual correlation

---

## Recommendations

### DO NOT Proceed to PPC

**Reason**: Model fails primary falsification criterion (residual ACF < 0.3)

### Next Steps (Priority Order)

**1. Try AR(2) Model** (HIGHEST PRIORITY)
```
μ[t] = α + β₁·year[t] + φ₁·ε[t-1] + φ₂·ε[t-2]
```
Expected: Residual ACF < 0.3

**2. Try AR(3) or ARMA(2,1)**
If AR(2) insufficient, explore:
- AR(3): Three lagged residuals
- ARMA(2,1): AR(2) + MA(1) component

**3. Simplify Trend Structure**
- β₂ ≈ 0 suggests quadratic unnecessary
- Try linear trend + higher-order AR

**4. Alternative: State-Space Model**
If AR(p) fails:
- Time-varying parameters
- Local level + AR errors

### FOR Phase 4 (LOO-CV Comparison)

Despite failure, **DO save for comparison**:
- ✅ `posterior_inference.netcdf` ready
- ✅ log_likelihood computed correctly
- ✅ Can compare LOO-ELPD to Experiment 1

**Expected**: Exp 2 may have better LOO (lower MAE/RMSE) despite ACF failure.

---

## Artifacts

### Code
- `/workspace/experiments/experiment_2/posterior_inference/code/model_vectorized.py`
- `/workspace/experiments/experiment_2/posterior_inference/code/fit_model_final.py`

### Diagnostics
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf` (11 MB)
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/convergence_summary.txt`
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/parameter_summary.csv`

### Plots (All 1.5 MB total)
- `plots/residual_diagnostics.png` ⭐ **KEY** - Shows ACF failure
- `plots/ar_coefficient.png` - φ vs data ACF
- `plots/fitted_trend.png` - Excellent visual fit
- `plots/trace_plots.png` - Perfect convergence
- `plots/regime_posteriors.png` - Variance hierarchy
- `plots/posterior_distributions.png` - All parameters

### Documentation
- `inference_summary.md` - Detailed analysis (12 sections)
- `RESULTS.md` - This file

---

## Conclusions

### What Worked ✅
- Excellent MCMC convergence (R-hat = 1.00)
- Strong parameter identification (ESS > 5000)
- Better point predictions than Experiment 1
- Clear regime differentiation
- Computational efficiency (~2 min)

### What Failed ❌
- **Residual ACF = 0.611** (target <0.3)
- AR(1) insufficient for temporal structure
- No improvement over Experiment 1 on primary criterion
- Unexpected regime variance ordering

### Final Verdict

**REJECT AR(1) Log-Normal Model** for this application.

**Paradox**: Model has perfect convergence and better fit metrics, yet fails the fundamental test of adequately capturing temporal autocorrelation.

**Root Cause**: Data exhibits stronger autocorrelation (ACF=0.975) than AR(1) can represent (φ≈0.85), requiring higher-order temporal structure.

**Path Forward**: Fit AR(2) or AR(3) model before proceeding to posterior predictive checks.

---

**Analysis Date**: 2025-10-30
**Analyst**: Claude (Bayesian Computation Specialist)
**Runtime**: ~2 minutes (NUTS sampling)
**Sample Size**: 8,000 draws (4 chains × 2,000)
