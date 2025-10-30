# Posterior Predictive Check - Experiment 1

**Model**: Negative Binomial GLM with Quadratic Trend
**Date**: 2025-10-30
**Overall Verdict**: **FAIL** - Cannot capture temporal autocorrelation

---

## Quick Summary

- ✓ **PASS**: Marginal distribution, mean trend, overdispersion
- ✗ **FAIL**: Temporal autocorrelation (ACF lag-1: p < 0.001)
- ✗ **FAIL**: Residual ACF = 0.595 > 0.5 threshold (meets falsification criterion)
- ✗ **FAIL**: Overestimates range and maximum values (p = 0.998)

**Implication**: Model is good for mean trend estimation but unsuitable for time series forecasting. Proceed to Experiment 2 (AR model).

---

## Files

### Main Report
- **`ppc_findings.md`** - Comprehensive 10-page analysis with all findings

### Code
- **`code/posterior_predictive_check.py`** - Full implementation
- **`code/test_statistics_summary.csv`** - Quantitative results table

### Visualizations
- **`plots/distributional_checks.png`** - Marginal distribution checks (PASS)
- **`plots/temporal_checks.png`** - Time series patterns (shows missing autocorrelation)
- **`plots/test_statistics.png`** - Summary statistics with Bayesian p-values
- **`plots/autocorrelation_check.png`** - **CRITICAL** - ACF comparison (FAIL)
- **`plots/residual_diagnostics.png`** - Residual patterns (ACF = 0.595)

---

## Key Findings

### Test Statistics Results

| Statistic | Observed | Replicated | p-value | Status |
|-----------|----------|------------|---------|--------|
| **ACF lag-1** | **0.926** | **0.818 ± 0.056** | **< 0.001** | **FAIL*** |
| Variance/Mean | 68.7 | 85.2 ± 16.0 | 0.869 | PASS |
| **Range** | **248** | **378 ± 63** | **0.998** | **FAIL*** |
| Mean | 109.4 | 111.3 ± 6.7 | 0.608 | PASS |
| Variance | 7512 | 9551 ± 2273 | 0.831 | PASS |

*p < 0.05 or p > 0.95 indicates extreme discrepancy

### What This Means

**The model assumes observations are independent** (given the trend), but the data shows strong temporal dependence:
- Observed data: High values persist, creating smooth trends
- Replicated data: Random fluctuations around mean curve
- **This is the expected limitation of the baseline model**

### Falsification Criteria Met

From `metadata.md`:
- ✓ Residual ACF lag-1 = 0.595 > 0.5 threshold → **ABANDON MODEL**
- ✓ PPC shows systematic bias (autocorrelation) → **ABANDON MODEL**

---

## Recommendations

1. **Proceed to Experiment 2**: Add autoregressive (AR) structure to capture temporal dependence
2. **Keep**: Negative Binomial likelihood, quadratic trend, Bayesian framework
3. **Add**: Lag-1 dependence term (e.g., ρ·log(C_{t-1}))

---

## How to Reproduce

```bash
cd /workspace
python experiments/experiment_1/posterior_predictive_check/code/posterior_predictive_check.py
```

**Requirements**:
- ArviZ InferenceData from fitted Stan model
- 1000 posterior predictive replications generated
- Runtime: ~30-60 seconds

---

**Next Phase**: Model Critique (will make final ACCEPT/REVISE/REJECT decision)
