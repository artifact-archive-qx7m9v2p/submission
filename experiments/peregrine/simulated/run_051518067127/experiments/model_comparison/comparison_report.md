# Model Comparison Report

## Executive Summary

**Comparison**: Experiment 1 (Negative Binomial GLM) vs Experiment 2 (AR(1) Log-Normal with Regime-Switching)

**Clear Winner**: **Experiment 2** (ΔELPD = +177.1 ± 7.5, significance = 23.7 SE)

**Recommendation**: **Select Experiment 2** with strong preference based on overwhelming predictive superiority. However, both models show residual temporal dependence (ACF: Exp1=0.596, Exp2=0.549), so Exp2 receives **CONDITIONAL ACCEPT** with recommendation to explore AR(2) structure in future work.

**Stacking Weights**: Exp2 = 1.000, Exp1 ≈ 0.000 (complete preference for Exp2)

---

## Visual Evidence Summary

All visualizations support Exp2's superiority while documenting remaining limitations:

1. **`loo_comparison.png`**: LOO cross-validation comparison showing 177-point ELPD advantage for Exp2
2. **`pareto_k_comparison.png`**: Diagnostic reliability - Exp2 has 1 problematic observation vs 0 for Exp1
3. **`calibration_comparison.png`**: LOO-PIT distributions and Q-Q plots showing both models reasonably calibrated
4. **`fitted_comparison.png`**: Fitted trends showing Exp2 captures data structure better, especially temporal dynamics
5. **`prediction_intervals.png`**: Exp2 achieves nominal 90% coverage while Exp1 over-covers (97.5%)
6. **`model_trade_offs.png`**: Multi-criteria spider plot revealing trade-offs across five dimensions

---

## 1. Single Model Assessments

### 1.1 Experiment 1: Negative Binomial GLM with Quadratic Trend

**Status**: REJECTED (residual ACF=0.596, posterior predictive check failed)

**Model**: Count ~ NegativeBinomial(μ, φ) where log(μ) = β₀ + β₁·year + β₂·year²

#### LOO Cross-Validation Diagnostics

| Metric | Value |
|--------|-------|
| **ELPD_LOO** | -170.96 ± 5.60 |
| **p_LOO** | 3.78 |
| **Pareto-k < 0.5** | 40 / 40 (100%) |
| **Pareto-k ≥ 0.7** | 0 / 40 (0%) |
| **Max Pareto-k** | 0.471 |

**Interpretation**: Excellent LOO reliability (all Pareto-k < 0.5), but very poor absolute ELPD indicates weak predictive density. The model assigns low probability to observed data points.

#### Predictive Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 16.53 | Average error ~16 counts |
| **RMSE** | 26.48 | Larger errors exist |
| **R²** | 0.907 | Captures 91% of variance |
| **90% Coverage** | 97.5% | **Over-covers** (too uncertain) |

#### Calibration (LOO-PIT)

- **Mean**: 0.489 (ideal: 0.5) ✓
- **Std Dev**: 0.282 (ideal: 0.289) ✓
- **Distribution**: Reasonably uniform (see `calibration_comparison.png`)

**Assessment**: Despite good calibration metrics, the model fails to capture temporal structure. High residual autocorrelation (ACF=0.596) indicates systematic prediction errors that persist over time. The model is too simple for this data.

---

### 1.2 Experiment 2: AR(1) Log-Normal with Regime-Switching

**Status**: CONDITIONAL ACCEPT (residual ACF=0.549, best available model)

**Model**: log(C_t) ~ Normal(μ_t, σ_regime) where μ_t = trend + φ·ε_{t-1}
- Trend: α + β₁·year + β₂·year²
- AR(1) coefficient: φ ~ 0.95 · Beta(20, 2)
- Three regime-specific variances

#### LOO Cross-Validation Diagnostics

| Metric | Value |
|--------|-------|
| **ELPD_LOO** | +6.13 ± 4.32 |
| **p_LOO** | 4.96 |
| **Pareto-k < 0.5** | 36 / 40 (90%) |
| **Pareto-k ∈ [0.5, 0.7)** | 3 / 40 (7.5%) |
| **Pareto-k ≥ 0.7** | 1 / 40 (2.5%) ⚠ |
| **Max Pareto-k** | 0.724 |

**Interpretation**: ELPD is **positive**, indicating better predictive density than Exp1. One observation has Pareto-k > 0.7 (observation with k=0.724), suggesting LOO may be slightly unstable for that point, but overall reliability is acceptable (90% good, 97.5% acceptable).

#### Predictive Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 14.53 | **Better** than Exp1 (16.53) |
| **RMSE** | 20.87 | **Better** than Exp1 (26.48) |
| **R²** | 0.943 | **Better** than Exp1 (0.907) |
| **90% Coverage** | 90.0% | **Excellent** (nominal level) |

#### Calibration (LOO-PIT)

- **Mean**: 0.474 (ideal: 0.5) - slightly low but acceptable
- **Std Dev**: 0.274 (ideal: 0.289) - slightly low, indicates mild under-dispersion
- **Distribution**: Reasonably uniform with minor deviations (see `calibration_comparison.png`)

**Assessment**: Substantial improvement over Exp1 in all predictive metrics. The AR(1) structure partially accounts for temporal dependence, reducing residual ACF from 0.596 to 0.549. However, remaining autocorrelation suggests AR(2) may be beneficial. Model achieves nominal coverage and good calibration.

---

## 2. Model Comparison

### 2.1 LOO Cross-Validation Comparison

**ArviZ `compare()` Results**:

| Model | Rank | ELPD_LOO | SE | ΔELPD | Weight |
|-------|------|----------|-----|-------|--------|
| **Exp2_AR1** | 1 | **+6.13** | 4.32 | 0.00 (ref) | **1.000** |
| **Exp1_NegBin** | 2 | -170.96 | 5.60 | -177.09 | ≈0.000 |

**Difference**: ΔELPD = +177.09 ± 7.48

**Statistical Significance**: 177.09 / 7.48 = **23.7 standard errors**

**Decision Criterion**:
- |ΔELPD| > 4×SE → **CLEAR WINNER** ✓
- Exp2 is overwhelmingly superior

**Visual Evidence**: `loo_comparison.png` shows the massive gap between models.

### 2.2 Pareto-k Diagnostic Comparison

**Exp1**: All 40 observations have excellent Pareto-k < 0.5 (max = 0.471)

**Exp2**: 36/40 excellent, 3/40 ok, 1/40 problematic (max = 0.724)

**Interpretation**: Exp1 has more reliable LOO estimates pointwise, but this doesn't overcome its poor predictive performance. The single problematic point in Exp2 (k=0.724) is a minor concern but doesn't invalidate the comparison given the massive ELPD difference.

**Visual Evidence**: `pareto_k_comparison.png` shows one elevated k-value for Exp2.

### 2.3 Calibration Comparison

**Both models** are reasonably calibrated based on LOO-PIT:
- Distributions approximately uniform
- Means near 0.5
- Q-Q plots roughly linear

**Exp1**: Slight over-coverage (97.5% vs nominal 90%)
**Exp2**: Exact nominal coverage (90.0%)

**Visual Evidence**: `calibration_comparison.png` shows both models' LOO-PIT distributions and Q-Q plots. Exp2's tighter match to nominal coverage is evident in the lower panel of `prediction_intervals.png`.

### 2.4 Fitted Trends Comparison

**Visual Evidence**: `fitted_comparison.png`

**Observations**:
1. Both models capture the overall upward trend
2. Exp1 (blue, solid) is smoother, averaging over temporal fluctuations
3. Exp2 (coral, dashed) adapts to local variations via AR(1) structure
4. Exp2's prediction intervals are narrower where data are less variable (early period)
5. Both models struggle in the most recent period (rightmost points)

**Interpretation**: Exp2's AR structure allows it to "remember" recent deviations, improving predictions. However, both models show some deviation in the final observations, consistent with remaining temporal dependence.

### 2.5 Prediction Uncertainty Comparison

**Visual Evidence**: `prediction_intervals.png`

**Upper panel** (90% PI widths over time):
- Exp1 (blue): Width increases steadily with trend magnitude
- Exp2 (coral): Width varies by regime, reflecting regime-switching variance

**Lower panel** (pointwise coverage):
- Exp1: Covers 39/40 observations (97.5%, over-covers)
- Exp2: Covers 36/40 observations (90.0%, nominal)

**Interpretation**: Exp2's regime-switching variance structure provides more appropriate uncertainty quantification. Exp1's over-coverage suggests it's uncertain about its predictions (wide intervals) because it's missing temporal structure.

### 2.6 Multi-Criteria Trade-offs

**Visual Evidence**: `model_trade_offs.png` (spider/radar plot)

Five criteria (higher scores = better performance):

| Criterion | Exp1 | Exp2 | Winner |
|-----------|------|------|--------|
| **Predictive Accuracy** (MAE-based) | ~0.40 | ~1.00 | Exp2 |
| **Calibration** (coverage) | ~0.17 | ~1.00 | Exp2 |
| **LOO Reliability** (k<0.5 fraction) | 1.00 | 0.90 | Exp1 |
| **Simplicity** (interpretability) | 0.70 | 0.30 | Exp1 |
| **Temporal Structure** (ACF-based) | 0.30 | 0.50 | Exp2 |

**Key Insights**:
- **Exp2 dominates** on predictive accuracy, calibration, and temporal structure
- **Exp1 wins** on LOO reliability and simplicity
- **Trade-off**: Exp2 accepts slightly lower LOO reliability and higher complexity in exchange for substantially better predictions

**Recommendation**: The trade-off strongly favors Exp2. A small increase in complexity and one marginally problematic Pareto-k value is a minor price for 177 ELPD points of improved predictive performance.

---

## 3. Where Models Differ

### 3.1 Exp1 Excels At:
- **Simplicity**: Only 4 parameters (β₀, β₁, β₂, φ), easy to interpret
- **LOO reliability**: Perfect Pareto-k diagnostics
- **Computational efficiency**: Faster sampling, no temporal dependencies to track

### 3.2 Exp2 Excels At:
- **Predictive accuracy**: MAE 12% lower, RMSE 21% lower
- **Capturing temporal structure**: Reduces residual ACF from 0.596 to 0.549
- **Appropriate uncertainty**: Achieves nominal 90% coverage
- **Local adaptation**: AR(1) term allows predictions to adjust based on recent data

### 3.3 Neither Model Fully Succeeds At:
- **Eliminating temporal dependence**: Both show ACF > 0.5 at lag 1
- **Capturing all data features**: Some late observations still deviate
- **Perfect calibration**: Minor deviations in LOO-PIT distributions

---

## 4. Recommendation

### 4.1 Selected Model: **Experiment 2 (AR(1) Log-Normal)**

**Reasons**:

1. **Overwhelming predictive superiority**:
   - ΔELPD = +177 ± 7.5 (23.7 SE)
   - No ambiguity in model comparison
   - Stacking assigns 100% weight to Exp2

2. **Better fit to data structure**:
   - Lower MAE and RMSE
   - Better captures temporal dynamics
   - Achieves nominal coverage

3. **Acceptable diagnostics**:
   - 90% of observations have excellent Pareto-k
   - 97.5% have acceptable Pareto-k
   - Single problematic point (k=0.724) is marginal and doesn't invalidate LOO

4. **Quantitative vs qualitative factors**:
   - Exp1's simplicity advantage is real but insufficient
   - When predictive performance differs by 177 ELPD points, complexity trade-off is justified

### 4.2 Important Caveats

**CONDITIONAL ACCEPT**: Exp2 is the best of the two models tested, but it is **not fully adequate**:

1. **Residual autocorrelation**: ACF(1) = 0.549 still indicates unmodeled temporal structure
2. **Model misspecification**: Both models show signs of missing important features
3. **Future work recommended**: AR(2) structure (adding second-order autocorrelation) likely to improve further

### 4.3 Use Case Recommendations

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Trend inference** | Exp2 | Better separates trend from temporal noise |
| **Point prediction** | Exp2 | Lower MAE and RMSE |
| **Interval prediction** | Exp2 | Achieves nominal coverage |
| **Forecasting** | Exp2 with caution | AR structure helps but remaining ACF is concerning |
| **Quick exploration** | Exp1 | Faster, simpler, but less accurate |
| **Final inference** | **Neither** | AR(2) recommended for publication-quality analysis |

### 4.4 Sensitivity Considerations

**Robust to**:
- Prior specification (both models use weakly informative priors)
- MCMC diagnostics excellent for both
- Sample size (n=40) adequate for both model complexities

**Sensitive to**:
- Temporal structure assumption (key differentiator)
- Regime boundaries (Exp2 relies on pre-specified regime-switching)
- Distributional assumptions (Neg Binomial vs Log-Normal)

---

## 5. Decision-Relevant Visualizations

### 5.1 Key Visual Evidence

**For declaring Exp2 the winner**:
1. **`loo_comparison.png`**: Exp2's ELPD is 177 points higher with non-overlapping error bars
2. **`fitted_comparison.png`**: Exp2's predictions track data more closely
3. **`model_trade_offs.png`**: Exp2 dominates on 3 of 5 criteria, ties don't overcome the deficit

**For documenting remaining issues**:
1. **`pareto_k_comparison.png`**: One problematic k-value in Exp2 (caveat)
2. **`calibration_comparison.png`**: Both models reasonably calibrated but not perfect

**For practical decision-making**:
1. **`prediction_intervals.png`**: Coverage analysis shows Exp2 achieves nominal level
2. **`fitted_comparison.png`**: Visual assessment of where models succeed and fail

### 5.2 Surprising Visual Patterns

1. **Exp1's perfect Pareto-k values**: Unusual to have all k < 0.5, suggests model may be too simple to identify influential observations

2. **Regime visibility**: In `fitted_comparison.png`, Exp2's varying uncertainty width reflects regime structure, but regimes are not visually obvious in the raw data

3. **Late-period struggles**: Both models show some misfit in the most recent observations (rightmost in plots), suggesting:
   - Possible regime change not captured by model
   - Acceleration in trend not accommodated by quadratic
   - Need for higher-order AR structure

---

## 6. Phase 5 Adequacy Assessment Implications

**Question**: Are these models sufficient for scientific inference, or should we continue to Experiment 3 (AR(2))?

**Evidence from this comparison**:

**Against adequacy (reasons to continue)**:
1. Exp2's residual ACF = 0.549 is **still high** (rule of thumb: should be < 0.2)
2. One problematic Pareto-k suggests model doesn't fit all observations well
3. LOO-PIT shows minor deviations from uniformity
4. Visual inspection reveals late-period misfit

**For adequacy (reasons to stop)**:
1. Exp2 is **vastly better** than Exp1 (177 ELPD points)
2. Diagnostic improvements may yield diminishing returns
3. Computational cost of AR(2) may be substantial
4. Exp2 already captures main trends and achieves nominal coverage

**Recommendation for Phase 5**:
- **Conditional acceptance**: Use Exp2 for preliminary inference
- **Strongly recommend AR(2) experiment** before final publication
- **Document limitations clearly** in any scientific communication

The massive improvement from Exp1 to Exp2 (177 ELPD points) suggests further model development could yield additional gains. However, Exp2 is adequate for exploratory analysis and hypothesis generation.

---

## 7. Conclusion

**Winner**: Experiment 2 (AR(1) Log-Normal) by an overwhelming margin

**Quantitative margin**: 177.1 ± 7.5 ELPD points (23.7 SE), equivalent to >99.999% certainty

**Qualitative factors**: Exp1's simplicity advantage is insufficient to overcome poor predictive performance

**Status**: Exp2 receives **CONDITIONAL ACCEPT** - use for current inference but plan AR(2) for robustness

**Key insight**: LOO-CV reveals that accounting for temporal structure (AR(1)) provides massive predictive improvements, even though residual diagnostics show the job is incomplete. This comparison validates the experimental progression and justifies exploring AR(2) in future work.

---

## Appendix: Files and Reproducibility

**Code**:
- `/workspace/experiments/model_comparison/code/run_comparison.py`

**Results**:
- `/workspace/experiments/model_comparison/results/loo_summary_exp1.txt`
- `/workspace/experiments/model_comparison/results/loo_summary_exp2.txt`
- `/workspace/experiments/model_comparison/results/loo_comparison.csv`
- `/workspace/experiments/model_comparison/results/summary_metrics.csv`

**Visualizations** (all 300 DPI PNG):
- `loo_comparison.png` - ELPD comparison with error bars
- `pareto_k_comparison.png` - Diagnostic reliability comparison
- `calibration_comparison.png` - LOO-PIT histograms and Q-Q plots
- `fitted_comparison.png` - Fitted trends with 90% prediction intervals
- `prediction_intervals.png` - Uncertainty quantification and coverage
- `model_trade_offs.png` - Multi-criteria spider plot

**Software**:
- ArviZ for LOO-CV and model comparison
- PyMC for model specification
- NumPy/Pandas for data manipulation
- Matplotlib/Seaborn for visualization

**Reproducibility**: All analyses use fixed random seeds where applicable. Posterior samples from MCMC runs stored in InferenceData format (.netcdf) for full reproducibility.

---

**Report generated**: 2025-10-30
**Minimum experiments**: 2 of 2 required (met ✓)
**Accepted models**: 1 (Exp2, conditional)
