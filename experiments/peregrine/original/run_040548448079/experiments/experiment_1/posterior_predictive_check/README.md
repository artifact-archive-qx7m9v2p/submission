# Posterior Predictive Check: Experiment 1

## Quick Summary

**Verdict**: PASS WITH CONCERNS

The model successfully validates the primary hypothesis (structural break at t=17) but exhibits systematic deficiencies in temporal dependency modeling.

---

## Key Results

### Structural Break Hypothesis: VALIDATED ✓
- Pre-break mean: 33.6 (observed) vs 36.6 (PP)
- Post-break mean: 165.5 (observed) vs 173.9 (PP)
- Growth ratio: 4.93x (observed) vs 4.87x (PP)
- β₂ = 0.556, 95% HDI = [0.113, 0.981]

### Model Deficiencies

1. **Autocorrelation**: FAIL ✗
   - Cannot reproduce observed ACF(1) = 0.944
   - PP generates ACF(1) = 0.613 (p < 0.001)
   - Residual ACF(1) = 0.519 (exceeds 0.5 threshold)

2. **Overdispersion**: Overestimated ~
   - Observed var/mean = 66.3
   - PP var/mean = 129.1 (p = 0.946)

3. **Extreme values**: Maximum misfit ✗
   - Observed max = 272
   - PP max = 541.6 (p = 0.990)

---

## Files

### Plots (`plots/`)
1. `pp_overlay.png` - Overall model fit visualization
2. `test_statistics.png` - Bayesian p-value diagnostics (6 panels)
3. `regime_comparison.png` - Pre/post-break regime fit
4. `qq_plot.png` - Quantile calibration
5. `acf_comparison.png` - Temporal dependency failure
6. `marginal_distribution.png` - Count distribution comparison
7. `coverage_assessment.png` - HDI coverage (100% vs 90% expected)

### Code (`code/`)
- `generate_pp_samples.py` - Generate 500 PP replicates
- `compute_test_statistics.py` - Calculate 9 test statistics
- `create_ppc_plots.py` - Create all visualizations
- `compute_model_residual_acf.py` - Residual ACF analysis

### Results (`code/`)
- `test_stats_summary.csv` - Bayesian p-values for all statistics
- `C_rep.npy` - Posterior predictive samples (500 × 40)
- `test_stats.npy` - Full test statistics dictionary

---

## Detailed Findings

See `ppc_findings.md` for comprehensive analysis including:
- Validation of all 6 falsification criteria
- Visual evidence for each finding
- Scientific conclusions
- Recommended model improvements
- Model adequacy assessment

---

## Bottom Line

**For hypothesis testing**: Model is adequate - structural break clearly validated

**For prediction**: Model is inadequate - temporal dependencies must be added

**Recommended action**: Accept for inference, enhance with AR(1) for forecasting
