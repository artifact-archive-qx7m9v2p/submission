# Model Selection Recommendation

## Decision: Select Experiment 2 (AR(1) Log-Normal with Regime-Switching)

### Bottom Line

**Experiment 2 is 177 ELPD points better than Experiment 1** (23.7 standard errors), representing overwhelming predictive superiority. Stacking assigns 100% weight to Exp2. **There is no ambiguity in this comparison.**

However, both models show residual temporal dependence (ACF ≈ 0.55), so Exp2 receives **CONDITIONAL ACCEPT** with recommendation for AR(2) exploration.

---

## Summary Statistics

| Metric | Exp1 (Neg Binomial) | Exp2 (AR1 Log-Normal) | Winner |
|--------|---------------------|------------------------|--------|
| **ELPD_LOO** | -170.96 ± 5.60 | **+6.13 ± 4.32** | Exp2 by 177 points |
| **MAE** | 16.53 | **14.53** | Exp2 (12% better) |
| **RMSE** | 26.48 | **20.87** | Exp2 (21% better) |
| **R²** | 0.907 | **0.943** | Exp2 |
| **90% Coverage** | 97.5% (over) | **90.0% (nominal)** | Exp2 |
| **Pareto-k ≥ 0.7** | 0 | 1 | Exp1 (minor) |
| **Residual ACF(1)** | 0.596 (poor) | 0.549 (less poor) | Exp2 |
| **Stacking Weight** | ≈0.000 | **1.000** | Exp2 (unanimous) |

---

## Why Experiment 2 Wins

### 1. Predictive Superiority (Decisive)

**ΔELPD = +177.09 ± 7.48**

- Statistical significance: **23.7 SE** (|ΔELPD| > 4×SE → clear winner)
- Probability that Exp2 is better: >99.999%
- Stacking: 100% weight to Exp2
- **No statistical ambiguity whatsoever**

**Visual evidence**: `loo_comparison.png` shows massive gap

### 2. Better Fit Quality (Strong)

- **12% lower MAE**: More accurate point predictions
- **21% lower RMSE**: Especially better for large errors
- **4% better R²**: Captures more variance

**Visual evidence**: `fitted_comparison.png` shows Exp2 tracking data more closely

### 3. Appropriate Uncertainty (Strong)

- Achieves **nominal 90% coverage** (Exp1 over-covers at 97.5%)
- Prediction intervals adapt to local data structure
- Better calibrated for decision-making

**Visual evidence**: `prediction_intervals.png` lower panel

### 4. Temporal Structure (Moderate)

- AR(1) reduces residual ACF from 0.596 to 0.549
- Models "memory" in the process
- Still inadequate (ACF should be <0.2) but improvement is substantial

### 5. Multi-Criteria Balance (Visualization)

**Visual evidence**: `model_trade_offs.png` spider plot shows:
- Exp2 dominates on 3 of 5 criteria
- Exp1 wins on simplicity and LOO reliability
- **Trade-off clearly favors Exp2**

---

## Why Not Experiment 1?

Exp1 has two advantages:
1. **Simpler**: 4 parameters, easier to explain
2. **Perfect LOO diagnostics**: All Pareto-k < 0.5

**However, these do not overcome**:
- ELPD 177 points worse
- Fails to capture temporal structure (ACF=0.596)
- Over-covers (too uncertain)
- **Status: REJECTED** in model checking

**When a model is REJECTED, no amount of simplicity makes it preferable.**

---

## Caveats and Limitations

### Exp2 is the Winner But Not Perfect

**Warning**: Experiment 2 receives **CONDITIONAL ACCEPT**, meaning:
- ✓ Best of models tested
- ✗ Not fully adequate for temporal structure
- → AR(2) recommended for robustness

**Specific concerns**:

1. **Residual ACF = 0.549**: Still indicates temporal dependence
   - Improved from 0.596 but target is <0.2
   - Suggests second-order autocorrelation present

2. **One problematic Pareto-k**: k=0.724 for one observation
   - Minor concern (only 2.5% of data)
   - Doesn't invalidate LOO given 177-point margin
   - May indicate an influential outlier

3. **Increased complexity**: More parameters, harder to interpret
   - Regime boundaries pre-specified (not learned)
   - AR structure requires understanding of time series models

4. **Both models struggle**: Late-period observations show misfit
   - Possible regime change or acceleration not captured
   - Visual inspection reveals room for improvement

---

## Use Case Recommendations

| Use Case | Model | Confidence | Notes |
|----------|-------|------------|-------|
| **Preliminary inference** | Exp2 | High | Adequate for hypothesis generation |
| **Point prediction** | Exp2 | High | MAE 12% better |
| **Uncertainty quantification** | Exp2 | High | Achieves nominal coverage |
| **Trend estimation** | Exp2 | Moderate | Better but temporal structure incomplete |
| **Forecasting** | Exp2 | Moderate | AR helps but ACF=0.549 is concerning |
| **Publication-quality inference** | **AR(2)** | Pending | Recommend Experiment 3 first |
| **Quick exploration** | Exp1 | Low | Faster but less accurate |

---

## Decision Tree

```
Is model selection ambiguous?
├─ No: ΔELPD = 177 ± 7.5 (23.7 SE)
│   └─ Winner: Experiment 2
│
├─ Is Exp2 adequate for final inference?
│   ├─ For preliminary work: YES (conditional)
│   └─ For publication: RECOMMEND AR(2) (Experiment 3)
│
└─ Should we trust Exp2 despite 1 bad Pareto-k?
    └─ YES: 177 ELPD margin >> 1 problematic observation
```

---

## Recommendation for Phase 5 Adequacy Decision

**Question**: Should we accept Exp2 or continue to Experiment 3 (AR(2))?

**Short answer**: **Accept conditionally** for current work, plan AR(2) for robustness

**Detailed recommendation**:

### Accept Exp2 Now If:
- Need results soon (computational budget)
- Exploratory analysis or hypothesis generation
- Willing to document limitations clearly
- Primary goal is trend estimation (not forecasting)

### Continue to AR(2) If:
- Publication-quality inference required
- Forecasting is critical application
- Residual ACF = 0.549 is unacceptable for domain
- Want to eliminate that problematic Pareto-k
- Time/resources available for another iteration

### Our Recommendation:
**Use Exp2 for current Phase 5 assessment, but flag AR(2) as future work.**

**Rationale**:
1. Exp2 is **vastly better** than Exp1 (validates experimental design)
2. Minimum 2 experiments completed (project requirements met)
3. Exp2 adequate for documenting temporal structure matters
4. Marginal improvement from AR(2) uncertain (could be small)
5. Better to document known limitation than over-promise perfection

**If continuing to AR(2)**:
- Expected improvement: ΔELPD ~ +5 to +20 (based on ACF reduction)
- May eliminate problematic Pareto-k
- Likely reduces residual ACF to <0.3
- Worth pursuing if resources allow

---

## Key Takeaways

1. **No ambiguity**: Exp2 is 23.7 SE better than Exp1
2. **Decisive evidence**: LOO-CV, multiple metrics, visual inspection all agree
3. **Conditional success**: Exp2 is best tested model but not perfect
4. **Clear path forward**: AR(2) is logical next step if time permits
5. **Documented limitations**: ACF=0.549 and 1 bad Pareto-k are known issues

**The comparison validates the experimental progression**: accounting for temporal structure provides massive improvements (177 ELPD points), justifying the added complexity. Further improvements via AR(2) are plausible but represent diminishing returns.

---

## Final Recommendation Summary

**Select**: Experiment 2 (AR(1) Log-Normal with Regime-Switching)

**Status**: CONDITIONAL ACCEPT

**Confidence**: Very high (23.7 SE superiority)

**Condition**: Document residual ACF=0.549, recommend AR(2) for robustness

**Next step**: Use Exp2 for Phase 5 adequacy assessment, plan Experiment 3 if resources allow

---

**Prepared by**: Model Assessment Specialist
**Date**: 2025-10-30
**Review status**: Ready for Phase 5 decision
