# Phase 2 Decision Card
## Experiment 1: Negative Binomial Quadratic Model

---

## DECISION: PHASE 2 TRIGGERED ✓

**Residual ACF(1) = 0.686 > 0.5 (threshold)**

---

## Quick Facts

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| **Residual ACF(1)** | **0.686** | **< 0.5** | **FAIL → PHASE 2** |
| Coverage (95%) | 100.0% | 85-98% | ACCEPTABLE |
| Extreme p-values | 7 | ≤ 2 | POOR |
| Overall Fit | POOR | GOOD/ACCEPT | **POOR** |

---

## The Problem in One Sentence

The model assumes observations are independent over time, but **residual ACF(1) = 0.686 reveals that 47% of residual variance is predictable from the previous observation**.

---

## Visual Evidence (Quick Look)

1. **`residual_diagnostics.png` Panel C**: ACF(1) bar far above orange "Phase 2 threshold" line at 0.5
2. **`residual_diagnostics.png` Panel B**: Clear wave pattern in residuals over time (not random scatter)
3. **`test_statistics.png` Bottom left**: Observed ACF(1) = 0.944 at extreme right tail (100th percentile)

---

## What This Means

- Observations at time t are highly correlated with time t-1
- Independence assumption violated
- Need models that explicitly handle temporal dependence
- Current model underestimates uncertainty in trends
- Cannot make reliable forecasts without temporal structure

---

## Next Model Class: TEMPORAL

Recommended approaches:
1. AR(1) random effects on top of quadratic trend
2. State-space models with evolving latent process
3. Dynamic linear models with time-varying slopes
4. Random walk with drift

**Keep:** Negative Binomial distribution (overdispersion working well)
**Keep:** Quadratic trend baseline (captures overall shape)
**Add:** Temporal correlation structure (AR component)

---

## Expected Improvement

After adding temporal structure:
- Residual ACF(1) should drop to < 0.3
- Coverage should be 90-95% (not 100%)
- ACF(1) Bayesian p-value: 0.1-0.9 (not 0.000)
- Better one-step-ahead forecasts
- More realistic uncertainty quantification

---

## Reference Documents

- **Quick Summary**: `SUMMARY.md` (2 pages)
- **Full Analysis**: `ppc_findings.md` (50 pages with all details)
- **This Card**: `DECISION.md` (decision rationale)

---

**Analysis Date:** 2025-10-29
**Status:** Complete - Ready for Phase 2
**Analyst:** Claude (Model Validation Specialist)
