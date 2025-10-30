# Model Comparison Metrics Summary

**Bayesian Meta-Analysis: Model 1 (Fixed-Effect) vs Model 2 (Random-Effects)**

Date: 2025-10-28

---

## LOO-CV Comparison (Primary Metric)

### Expected Log Pointwise Predictive Density (ELPD)

| Model | ELPD LOO | SE | Rank | LOO Weight |
|-------|----------|-----|------|------------|
| **Model 1 (Fixed)** | **-30.52** | 1.14 | 1 | 1.0 |
| **Model 2 (Random)** | -30.69 | 1.05 | 2 | 0.0 |

### Difference Metrics

- **ΔELPD**: -0.17 ± 0.10 (Model 2 - Model 1)
- **|ΔELPD/SE|**: **1.62** (< 2 threshold for meaningful difference)
- **Conclusion**: **No substantial difference** - prefer simpler model

### Effective Parameters

| Model | Actual Parameters | p_LOO | Interpretation |
|-------|------------------|-------|----------------|
| Model 1 | 1 | 0.64 | Slightly regularized |
| Model 2 | 10 | 0.98 | Strong shrinkage reduces to ~1 |

**Key Insight**: Model 2's 10 parameters shrink to effectively 1, gaining no predictive advantage.

---

## LOO Reliability: Pareto k Diagnostics

### Model 1 (Fixed-Effect)

| k Range | Count | Percentage | Interpretation |
|---------|-------|------------|----------------|
| k < 0.5 | 8 | 100% | Good |
| 0.5 < k < 0.7 | 0 | 0% | OK |
| k > 0.7 | 0 | 0% | Bad |

**Status**: ✓ All observations reliable

### Model 2 (Random-Effects)

| k Range | Count | Percentage | Interpretation |
|---------|-------|------------|----------------|
| k < 0.5 | 7 | 87.5% | Good |
| 0.5 < k < 0.7 | 1 | 12.5% | OK |
| k > 0.7 | 0 | 0% | Bad |

**Status**: ✓ All observations reliable (1 moderately influential)

### Individual Pareto k Values

| Study | y_obs | Model 1 k | Model 2 k | Status |
|-------|-------|-----------|-----------|--------|
| 1 | 28 | 0.10 | 0.27 | Good |
| 2 | 8 | 0.14 | 0.47 | OK |
| 3 | -3 | 0.05 | 0.25 | Good |
| 4 | 7 | 0.13 | 0.40 | Good |
| 5 | -1 | 0.26 | 0.35 | Good |
| 6 | 1 | 0.14 | 0.21 | Good |
| 7 | 18 | 0.23 | 0.55 | OK |
| 8 | 12 | 0.02 | 0.17 | Good |

**Observation**: Model 2 has higher k values (studies 2 and 7 near 0.5), but all within acceptable range.

---

## Calibration Metrics

### Posterior Predictive Coverage

| Nominal Coverage | Model 1 Empirical | Model 2 Empirical | Target | Status |
|------------------|-------------------|-------------------|---------|--------|
| **50%** | 62.5% | 62.5% | 50% | Slightly conservative |
| **90%** | 100.0% | 100.0% | 90% | Conservative |
| **95%** | 100.0% | 100.0% | 95% | Conservative |

**Interpretation**: Both models show slightly conservative coverage (over-covering), likely due to small sample size (J=8).

### Interval Sharpness (Average Width)

| Interval | Model 1 Width | Model 2 Width | Difference | % Wider (M2) |
|----------|---------------|---------------|------------|--------------|
| 50% | 17.8 | 18.3 | +0.5 | +2.8% |
| 90% | 43.2 | 44.8 | +1.6 | +3.7% |
| **95%** | **51.5** | **53.3** | **+1.8** | **+3.4%** |

**Conclusion**: Model 2 has slightly wider intervals (3.4% at 95% level), trading sharpness for conservatism.

---

## Predictive Performance

### Error Metrics

| Metric | Model 1 | Model 2 | Difference | % Improvement |
|--------|---------|---------|------------|---------------|
| **RMSE** | 9.88 | 9.09 | -0.79 | -8.0% |
| **MAE** | 7.74 | 7.08 | -0.66 | -8.5% |
| **Standardized RMSE** | 0.77 | 0.70 | -0.07 | -9.1% |
| **Standardized MAE** | 0.62 | 0.56 | -0.06 | -9.7% |

**Interpretation**: Model 2 shows marginally better point predictions (~8% lower error), but:
- Not reflected in LOO-CV (accounting for overfitting)
- Expected from added complexity
- Not statistically meaningful

### Model Agreement

- **Prediction correlation**: -0.049 (essentially uncorrelated due to small J)
- **Mean absolute difference**: 0.68
- **Maximum difference**: 1.35 (Study 1)

### Study-by-Study Predictions

| Study | y_obs | σ | Model 1 Pred | Model 2 Pred | Difference |
|-------|-------|---|--------------|--------------|------------|
| 1 | 28 | 15 | 7.35 | 8.70 | +1.35 |
| 2 | 8 | 10 | 7.55 | 7.60 | +0.05 |
| 3 | -3 | 16 | 7.54 | 6.85 | -0.69 |
| 4 | 7 | 11 | 7.37 | 7.48 | +0.11 |
| 5 | -1 | 9 | 7.37 | 6.36 | -1.02 |
| 6 | 1 | 11 | 7.43 | 6.67 | -0.76 |
| 7 | 18 | 10 | 7.48 | 8.76 | +1.28 |
| 8 | 12 | 18 | 7.42 | 7.63 | +0.21 |

**Pattern**: Largest differences occur for extreme observations (Studies 1, 5, 7), where Model 2 moves slightly toward observed values.

---

## Parameter Estimates

### Overall Effect

| Parameter | Model 1 (θ) | Model 2 (μ) | Difference |
|-----------|-------------|-------------|------------|
| **Mean** | 7.40 | 7.43 | +0.03 (0.4%) |
| **SD** | 4.00 | 4.26 | +0.26 (6.5%) |
| **95% HDI Lower** | -0.26 | -1.43 | -1.17 |
| **95% HDI Upper** | 15.38 | 15.33 | -0.05 |

**Conclusion**: Point estimates differ by only 0.4% - scientifically identical.

### Heterogeneity (Model 2 Only)

| Parameter | Mean | SD | 95% HDI |
|-----------|------|-----|---------|
| **τ** (between-study SD) | 3.36 | 2.51 | [0.00, 8.25] |
| **I²** (heterogeneity %) | ~8.3% | - | Very low |

**Interpretation**:
- τ posterior includes zero
- I² = 8.3% indicates minimal heterogeneity
- Threshold for "moderate" is I² > 30%

### Study-Specific Effects (Model 2)

| Study | y_obs | θᵢ (Mean) | θᵢ (SD) | Shrinkage % |
|-------|-------|-----------|---------|-------------|
| 1 | 28 | 8.71 | 5.71 | 6.2% |
| 2 | 8 | 7.50 | 5.19 | 11.5% |
| 3 | -3 | 6.80 | 5.52 | 6.1% |
| 4 | 7 | 7.37 | 5.20 | 14.9% |
| 5 | -1 | 6.28 | 5.15 | 13.6% |
| 6 | 1 | 6.76 | 5.18 | 10.4% |
| 7 | 18 | 8.79 | 5.27 | 12.9% |
| 8 | 12 | 7.63 | 5.69 | 4.3% |

**Pattern**: All estimates strongly shrunk (6-15%) toward μ = 7.43, confirming low heterogeneity.

---

## Complexity-Performance Trade-off

### Parsimony Metrics

| Metric | Model 1 | Model 2 | Analysis |
|--------|---------|---------|----------|
| **Actual Parameters** | 1 | 10 | 10× difference |
| **Effective Parameters** | 0.64 | 0.98 | 1.5× difference (shrinkage) |
| **Additional Complexity** | - | +0.34 | Minimal |
| **Performance Gain** | - | **-0.17 ELPD** | **Negative** |
| **Complexity Justified?** | - | **NO** | ΔELPD < 2 SE |

### Interpretation

Model 2's 9 additional parameters:
- Are shrunk to add only 0.34 effective parameters
- Provide **no predictive advantage** (ΔELPD = -0.17)
- Yield **identical** scientific inference
- Are **not justified** by the data

---

## Sensitivity Metrics

### Influence Analysis

**Observations with moderate influence (k > 0.5)**:
- Model 1: None
- Model 2: Study 2 (k=0.47), Study 7 (k=0.55)

**Maximum Pareto k**:
- Model 1: 0.26 (Study 5)
- Model 2: 0.55 (Study 7)

**Status**: Both models have reliable LOO estimates (all k < 0.7)

### Residual Analysis

**Model 1**:
- Mean residual: 3.48
- Max residual: 20.65 (Study 1)
- Standardized residuals: All within ±2 SD

**Model 2**:
- Mean residual: 2.72
- Max residual: 19.30 (Study 1)
- Standardized residuals: All within ±2 SD

**Observation**: Both models show no systematic bias; Model 2 slightly smaller residuals (consistent with lower RMSE).

---

## Decision Metrics Summary

### Quantitative Criteria

| Criterion | Threshold | Model 1 | Model 2 | Decision |
|-----------|-----------|---------|---------|----------|
| **LOO distinguishability** | \|ΔELPD/SE\| > 2 | - | 1.62 | Models tied → parsimony |
| **Effective complexity** | Lower is better | 0.64 | 0.98 | Model 1 wins |
| **Heterogeneity** | I² > 30% needed | N/A | 8.3% | Model 1 sufficient |
| **Parameter agreement** | Similar inference | ✓ | ✓ | Both give same θ |
| **Calibration** | Coverage ≈ nominal | ✓ | ✓ | Both well-calibrated |
| **Diagnostics** | Pareto k < 0.7 | ✓ | ✓ | Both reliable |

**Final Score**: Model 1 wins on parsimony with equivalent performance.

---

## Comparison Rankings

### By Predictive Accuracy
1. Model 2 (RMSE = 9.09) - marginal
2. Model 1 (RMSE = 9.88)

### By LOO-CV
1. **Model 1** (ELPD = -30.52)
2. Model 2 (ELPD = -30.69)

### By Parsimony
1. **Model 1** (1 parameter)
2. Model 2 (10 parameters, but ~1 effective)

### By Calibration
Tie - both well-calibrated

### By Reliability
Tie - both pass diagnostics

### By Interpretability
1. **Model 1** (single pooled effect)
2. Model 2 (hierarchical structure)

---

## Numerical Evidence for Key Conclusions

### 1. "No substantial difference"
- ΔELPD = -0.17 ± 0.10
- Ratio = 1.62 < 2 threshold
- p(ΔELPD > 0) ≈ 0.05 (favors M1)

### 2. "Low heterogeneity"
- I² = 8.3% (< 25% threshold)
- τ = 3.36, 95% HDI includes 0
- Shrinkage: 6-15% (strong pooling)

### 3. "Identical inference"
- θ (M1) = 7.40 vs μ (M2) = 7.43
- Difference = 0.03 (0.4%)
- HDIs overlap by >95%

### 4. "Complexity not justified"
- Additional p_LOO = 0.34
- Performance gain = -0.17 ELPD
- Gain/complexity ratio = -0.5 (negative!)

---

## Statistical Power and Limitations

### Sample Size
- J = 8 studies
- Limited power to detect heterogeneity
- Wide credible intervals (±4)

### Implications
- I² estimate is uncertain
- τ posterior includes wide range
- May need Model 2 with more data

### Future Considerations
- Monitor I² as data accumulates
- Reconsider if J > 15 studies
- Switch to Model 2 if I² > 30%

---

## Files Referenced

**Data**:
- `/workspace/data/data.csv`

**Results**:
- `/workspace/experiments/model_comparison/comparison_results.json`
- `/workspace/experiments/model_comparison/loo_comparison_table.csv`
- `/workspace/experiments/model_comparison/predictions_comparison.csv`
- `/workspace/experiments/model_comparison/predictive_metrics.csv`
- `/workspace/experiments/model_comparison/influence_diagnostics.csv`

**Visualizations**:
- `/workspace/experiments/model_comparison/plots/*.png`

---

**Prepared by**: Claude (Model Assessment Specialist)
**Date**: 2025-10-28
**Status**: COMPLETE
