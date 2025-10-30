# Posterior Predictive Check: Experiment 2
## AR(1) Log-Normal with Regime-Switching

**Date**: 2025-10-30
**Status**: Complete

---

## Quick Summary

**VERDICT**: MIXED (Major improvement over Exp 1, but still fails residual ACF threshold)

### Key Results

| Metric | Result | Status | vs Exp 1 |
|--------|--------|--------|----------|
| ACF lag-1 PPC | p = 0.560 | ✓ PASS | p < 0.001 (FAIL) |
| Residual ACF | 0.549 | ✗ FAIL (>0.5) | 0.596 (FAIL) |
| All test statistics | 9/9 pass | ✓ PASS | 5/9 pass |
| 90% PI coverage | 100% | ✓ PASS | 100% |
| Point predictions | MAE=13.99 | ✓ Better | MAE=16.41 |

### The Paradox

**Model CAN generate observed ACF but HAS high residual ACF**

**Explanation**: AR(1) captures lag-1 dependence but misses higher-order structure (likely needs AR(2))

---

## Files

### Code
- `code/posterior_predictive_check.py` - Main PPC analysis (1,000 replications)
- `code/create_comparison_plot.py` - Exp1 vs Exp2 comparison
- `code/test_statistics_summary.csv` - All test statistics with p-values

### Plots
- `plots/distributional_checks.png` - Marginal distribution alignment (PASS)
- `plots/temporal_checks.png` - Time series fit with predictive bands (PASS)
- `plots/test_statistics.png` - 9 test statistics, all PASS
- `plots/autocorrelation_check.png` - **CRITICAL**: ACF PPC PASS, but residuals FAIL
- `plots/residual_diagnostics.png` - Residual ACF = 0.549 > 0.5 threshold (FAIL)
- `plots/comparison_exp1_vs_exp2.png` - Side-by-side comparison showing improvement

### Documentation
- `ppc_findings.md` - **Comprehensive 15-page report with detailed analysis**

---

## Key Findings

### What Works
1. **AR(1) structure successfully reproduces observed autocorrelation** (p=0.560)
2. **All test statistics pass** (vs 56% pass rate in Exp 1)
3. **Better point predictions** (15% reduction in MAE)
4. **Perfect predictive coverage** (100% of observations in 90% PI)

### What Doesn't
1. **Residual ACF = 0.549 exceeds threshold** (0.5) → Model abandoned per falsification criteria
2. **Temporal patterns remain** in residual diagnostic plots
3. **Model misses higher-order dynamics** beyond lag-1

### Why the Paradox?
- Model uses AR(1) to predict → captures lag-1 structure → improves fit
- Remaining residuals reveal lag-2+ structure → high residual ACF
- **Not a contradiction**: Different diagnostics measure different things

---

## Recommendations

1. **Experiment 3**: Test AR(2) model
   ```
   ε[t] = φ₁·ε[t-1] + φ₂·ε[t-2] + noise
   ```

2. **Alternative**: State-space model with time-varying growth

3. **Keep**: Log-scale, quadratic trend, regime structure (all working well)

---

## Comparison to Experiment 1

### Experiment 1 (Baseline: No AR structure)
- **Cannot generate observed ACF** (p < 0.001, EXTREME FAIL)
- **4 test statistics fail** including ACF, range, maximum
- Residual ACF = 0.596 (all temporal structure missed)

### Experiment 2 (AR(1))
- **Can generate observed ACF** (p = 0.560, PASS)
- **All test statistics pass** (9/9)
- Residual ACF = 0.549 (higher-order structure missed)

### Verdict
AR(1) is a **major step forward** but **not sufficient**. The 0.549 residual ACF tells us exactly what to fix next.

---

## Usage

### Run PPC
```bash
python code/posterior_predictive_check.py
```

### View Results
```bash
# All plots
ls plots/*.png

# Test statistics
cat code/test_statistics_summary.csv

# Full report
less ppc_findings.md
```

---

## Technical Details

- **Posterior samples**: 8,000 (4 chains × 2,000 draws)
- **PPC replications**: 1,000
- **AR(1) implementation**: Stationary initialization ε[0] ~ N(0, σ/√(1-φ²))
- **Residuals**: Randomized quantile residuals
- **Test statistics**: 9 summary statistics with Bayesian p-values

---

## Next Steps

Per workflow:
1. Continue to **Model Critique** (regardless of PPC failure)
2. Document specific model deficiencies
3. Plan Experiment 3: AR(2) or state-space model

---

**Contact**: Model Validation Specialist
**Full Report**: See `ppc_findings.md` for 15-page detailed analysis
