# Model Comparison: Experiments 1 vs 2

## Quick Summary

**Winner**: Experiment 2 (AR(1) Log-Normal) - CONDITIONAL ACCEPT

**Margin**: +177.1 ELPD points (23.7 standard errors)

**Confidence**: Overwhelming (weight = 1.000)

**Status**: Best available model, AR(2) recommended for robustness

---

## Key Results at a Glance

| Metric | Exp1: Neg Binomial | Exp2: AR(1) Log-Normal | Winner |
|--------|-------------------|------------------------|--------|
| ELPD_LOO | -170.96 ± 5.60 | **+6.13 ± 4.32** | Exp2 (+177) |
| MAE | 16.53 | **14.53** | Exp2 (12% better) |
| RMSE | 26.48 | **20.87** | Exp2 (21% better) |
| R² | 0.907 | **0.943** | Exp2 |
| 90% Coverage | 97.5% | **90.0%** | Exp2 (nominal) |
| Pareto-k bad | 0 | 1 | Exp1 (minor) |
| Residual ACF | 0.596 | 0.549 | Exp2 (but still high) |
| Weight | ≈0.000 | **1.000** | Exp2 (unanimous) |

**Decision**: ΔELPD = 177 ± 7.5 → 23.7 SE → **CLEAR WINNER**

---

## What This Means

### The Good News
1. **No ambiguity**: Exp2 is overwhelmingly better (23.7 SE)
2. **Validates temporal structure**: AR(1) provides massive improvement
3. **Ready for use**: Can proceed with Exp2 for preliminary inference
4. **Experimental design validated**: Progression from GLM to AR pays off

### The Caveats
1. **Not perfect**: Residual ACF = 0.549 (target: <0.2)
2. **One bad Pareto-k**: Observation with k=0.724 (minor concern)
3. **Conditional acceptance**: Recommend AR(2) for publication quality
4. **Both models struggle**: Late-period observations show misfit

### Bottom Line
**Use Exp2 now, plan AR(2) for robustness.**

---

## Files in This Directory

### Documentation
- **`comparison_report.md`**: Comprehensive 7-section analysis with full technical details
- **`recommendation.md`**: Decision-focused summary with use case guidance
- **`README.md`**: This file - quick reference

### Code
- **`run_comparison.py`**: Main analysis script (working version)
- Other Python files: Development iterations (can ignore)

### Results (CSV)
- **`summary_metrics.csv`**: All key metrics in table format
- **`loo_comparison.csv`**: ArviZ compare() output
- **`loo_summary_exp1.txt`**: Detailed LOO diagnostics for Exp1
- **`loo_summary_exp2.txt`**: Detailed LOO diagnostics for Exp2

### Visualizations (PNG, 300 DPI)
1. **`loo_comparison.png`**: ELPD comparison showing 177-point gap
2. **`pareto_k_comparison.png`**: Diagnostic reliability (1 bad k-value in Exp2)
3. **`calibration_comparison.png`**: LOO-PIT distributions and Q-Q plots
4. **`fitted_comparison.png`**: Fitted trends with 90% prediction intervals
5. **`prediction_intervals.png`**: Uncertainty quantification and coverage analysis
6. **`model_trade_offs.png`**: Multi-criteria spider plot showing Exp2 dominance

---

## Visual Evidence Summary

**Most important plots for decision**:
1. `loo_comparison.png` - Decisive 177-point gap
2. `model_trade_offs.png` - Exp2 dominates 3 of 5 criteria
3. `fitted_comparison.png` - Visual confirmation of better fit

**Supporting evidence**:
4. `prediction_intervals.png` - Exp2 achieves nominal coverage
5. `calibration_comparison.png` - Both reasonably calibrated
6. `pareto_k_comparison.png` - Documents 1 problematic observation

---

## How to Use These Results

### For Preliminary Inference (Now)
- Use **Experiment 2** with confidence
- Report ELPD = +6.13 ± 4.32
- Document: "AR(1) structure, 177 ELPD points better than GLM"
- Caveat: "Residual ACF = 0.549, AR(2) recommended"

### For Publication (Future)
- **Recommend Experiment 3 (AR(2))** first
- Expected further improvement: ΔELPD ~ +5 to +20
- Would likely reduce ACF to <0.3
- May eliminate problematic Pareto-k

### For Presentations
- **Key message**: "Temporal structure matters - 177 ELPD point improvement"
- **Key visual**: `loo_comparison.png` (massive gap)
- **Key caveat**: "Still room for improvement (AR(2))"

---

## Technical Notes

### LOO Cross-Validation
- Method: PSIS-LOO-CV (Pareto-smoothed importance sampling)
- Scale: log (higher ELPD = better)
- Comparison: ArviZ `compare()` with stacking weights
- Reliability: Pareto-k diagnostics (k < 0.7 preferred)

### Model Specifications
**Exp1**: Count ~ NegBin(μ, φ), log(μ) = β₀ + β₁·year + β₂·year²

**Exp2**: log(Count_t) ~ Normal(μ_t, σ_regime)
- μ_t = (α + β₁·year + β₂·year²) + φ·ε_{t-1}
- Three regime-specific variances
- AR(1) coefficient φ ~ 0.95 · Beta(20, 2)

### Pareto-k Thresholds
- k < 0.5: Good (reliable LOO)
- 0.5 ≤ k < 0.7: OK (acceptable)
- k ≥ 0.7: Problematic (LOO may be unreliable)

**Exp1**: 40/40 good (perfect)
**Exp2**: 36/40 good, 3/40 ok, 1/40 problematic (97.5% acceptable)

### Statistical Decision Rules
- |ΔELPD| < 2×SE: Indistinguishable → prefer simpler model
- 2×SE < |ΔELPD| < 4×SE: Moderate difference → consider practical factors
- |ΔELPD| > 4×SE: Clear winner → prefer better model

**This comparison**: 177 / 7.5 = 23.7 SE → **CLEAR WINNER**

---

## Reproducibility

**Random seed**: Set where applicable
**Software versions**: See environment
**Data**: `/workspace/data/data.csv` (40 observations)
**Posterior samples**: Stored in InferenceData format (.netcdf)

**To reproduce**:
```bash
python /workspace/experiments/model_comparison/code/run_comparison.py
```

**Runtime**: ~2-3 minutes (posterior predictive generation is main cost)

---

## Contact / Questions

For questions about:
- **Technical details**: See `comparison_report.md` Section 2 (single model assessments)
- **Decision rationale**: See `recommendation.md` Section "Why Experiment 2 Wins"
- **Visualizations**: See `comparison_report.md` Section 5 (visual evidence)
- **Future work**: See `recommendation.md` Section "Recommendation for Phase 5"

---

## Citation

If using these results:

> Model comparison via LOO-CV (Vehtari et al. 2017) revealed Experiment 2
> (AR(1) Log-Normal with regime-switching) provided 177.1 ± 7.5 ELPD points
> of improved predictive performance relative to Experiment 1 (Negative
> Binomial GLM), a 23.7 standard error difference indicating overwhelming
> superiority. However, residual autocorrelation (ACF = 0.549) suggests
> AR(2) structure may provide further improvements.

---

**Assessment completed**: 2025-10-30
**Status**: CONDITIONAL ACCEPT (Experiment 2)
**Next step**: Use Exp2 for Phase 5, consider AR(2) for robustness
