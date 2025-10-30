# Posterior Predictive Check: Experiment 2
## Hierarchical Partial Pooling Model with Known Measurement Error

**Date**: 2025-10-28
**Status**: COMPLETE
**Model**: Hierarchical Partial Pooling (mu, tau, theta[1:8])

---

## Executive Summary

**RECOMMENDATION: PREFER MODEL 1 (COMPLETE POOLING) BY PARSIMONY**

The hierarchical partial pooling model (Model 2) shows **no significant improvement** in predictive performance over the simpler complete pooling model (Model 1). LOO-CV comparison reveals:
- Δ ELPD = -0.11 ± 0.36 (Model 2 slightly worse, but not significant)
- |Δ ELPD| = 0.11 < 2×SE = 0.71 (models statistically equivalent)

**Critical Issue**: Model 2 shows one observation (Obs 5) with problematic Pareto k = 0.8690 (BAD), indicating unreliable LOO estimates for this point. Model 1 has all k < 0.5 (GOOD).

By the principle of parsimony, the simpler Model 1 (1 parameter) is preferred over Model 2 (10 parameters) when predictive performance is equivalent.

---

## Plots Generated

### Primary Diagnostic Plots

| Plot File | Purpose | Key Finding |
|-----------|---------|-------------|
| `loo_comparison.png` | Compare Model 1 vs Model 2 ELPD | Models statistically equivalent (Δ ELPD < 2×SE) |
| `loo_pareto_k.png` | Assess reliability of LOO estimates | Model 2 has one BAD k-value (0.87); Model 1 all GOOD |
| `ppc_observations.png` | Observation-level fit quality | All 8 observations well-calibrated (p > 0.05) |
| `ppc_test_statistics.png` | Summary statistics comparison | All 8 statistics within predictive distribution |
| `ppc_residuals.png` | Residual patterns and diagnostics | No systematic patterns, good normality |
| `ppc_calibration.png` | LOO-PIT calibration check | Good calibration, no systematic bias |

---

## 1. LOO-CV Model Comparison (CRITICAL)

### Overall Model Comparison

```
                          rank   elpd_loo     p_loo      se     dse  warning
Model_1_Complete_Pooling     0    -32.05      1.17    1.43    0.00    False
Model_2_Hierarchical         1    -32.16      2.11    1.09    0.36     True
```

**Key Metrics**:
- **Model 1 ELPD**: -32.05 ± 1.43
- **Model 2 ELPD**: -32.16 ± 1.09
- **Δ ELPD**: -0.11 ± 0.36 (Model 2 - Model 1)
- **Significance threshold**: 2×SE = 0.71
- **Result**: |Δ ELPD| = 0.11 < 0.71 → **Models statistically equivalent**

### Visual Evidence

`loo_comparison.png` (left panel) shows both models have overlapping error bars, confirming no significant difference. The right panel shows pointwise ELPD values are nearly identical across all 8 observations, with Model 2 performing slightly worse on observations 3, 5, and 6.

### Interpretation

The hierarchical model's added complexity (9 additional parameters: tau + 8 theta values) does **not** translate into improved out-of-sample predictive performance. This suggests:

1. **No genuine heterogeneity**: The data do not require group-specific estimates
2. **Parsimony principle applies**: Simpler Model 1 should be preferred
3. **Consistent with posterior**: tau is very uncertain (95% HDI: [0.28, 15.65]), overlapping with zero
4. **Consistent with EDA**: Variance decomposition showed tau² = 0

---

## 2. Pareto k Diagnostics

### Model 1: Complete Pooling (EXCELLENT)

```
Obs 1: k = 0.1276 [GOOD]
Obs 2: k = 0.2498 [GOOD]
Obs 3: k = 0.1285 [GOOD]
Obs 4: k = 0.3000 [GOOD]
Obs 5: k = 0.3729 [GOOD]
Obs 6: k = 0.1538 [GOOD]
Obs 7: k = 0.2073 [GOOD]
Obs 8: k = 0.0767 [GOOD]
Max k: 0.3729
```

**Assessment**: All observations have k < 0.5 → LOO estimates are **reliable**

### Model 2: Hierarchical Partial Pooling (PROBLEMATIC)

```
Obs 1: k = 0.4808 [GOOD]
Obs 2: k = 0.4516 [GOOD]
Obs 3: k = 0.5409 [OK]      ← borderline
Obs 4: k = 0.4345 [GOOD]
Obs 5: k = 0.8690 [BAD]      ← PROBLEMATIC
Obs 6: k = 0.6945 [OK]      ← borderline
Obs 7: k = 0.4056 [GOOD]
Obs 8: k = 0.3543 [GOOD]
Max k: 0.8690
```

**Assessment**:
- 1/8 observations (12.5%) have k > 0.7 (BAD)
- 2/8 observations (25%) have k > 0.5 (OK/borderline)
- LOO estimates are **unreliable** for Observation 5

### Visual Evidence

`loo_pareto_k.png` clearly shows Model 2 (right panel) has elevated k-values compared to Model 1 (left panel). Observation 5 is far above the k=0.7 threshold, indicating the hierarchical model's posterior is very sensitive to this observation.

### Why is Observation 5 Problematic?

**Observation 5**: y = -4.88, sigma = 9

This is the most extreme negative value in the dataset. The hierarchical model struggles because:
1. It must balance this extreme value against the group mean
2. The shrinkage towards mu creates high sensitivity
3. Uncertain tau amplifies this sensitivity
4. Model 1 handles this better by directly pooling all data

**Implication**: The hierarchical model is **less robust** to unusual observations than the complete pooling model for this dataset.

---

## 3. Observation-Level Posterior Predictive Checks

### Summary Statistics

- **Min p-value**: 0.333 (Observation 5)
- **Max p-value**: 0.927 (Observation 8)
- **Extreme p-values (< 0.05)**: 0/8
- **Assessment**: All observations are **well-calibrated**

### Individual Observations

| Obs | y_obs | sigma | y_pred_mean | p-value | Assessment |
|-----|-------|-------|-------------|---------|------------|
| 1   | 20.02 | 15    | 12.19       | 0.625   | Good fit   |
| 2   | 15.30 | 10    | 11.22       | 0.754   | Good fit   |
| 3   | 26.08 | 16    | 12.59       | 0.455   | Good fit   |
| 4   | 25.73 | 11    | 13.85       | 0.456   | Good fit   |
| 5   | -4.88 | 9     | 5.97        | 0.333   | Good fit   |
| 6   | 6.08  | 11    | 9.41        | 0.791   | Good fit   |
| 7   | 3.17  | 10    | 8.73        | 0.630   | Good fit   |
| 8   | 8.55  | 18    | 10.27       | 0.927   | Good fit   |

### Visual Evidence

`ppc_observations.png` shows all 8 observed values (red lines) fall well within the posterior predictive distributions (blue histograms). No observation appears in the tails of its predictive distribution.

### Key Insight

Despite the problematic Pareto k for Observation 5, the posterior predictive distribution still covers the observed value. The high k indicates sensitivity, not necessarily poor fit.

---

## 4. Test Statistics: Observed vs Replicated Data

All 8 summary statistics show good agreement between observed and posterior predictive data:

| Statistic | Observed | Pred Mean | Pred SD | p-value | Status |
|-----------|----------|-----------|---------|---------|--------|
| mean      | 12.51    | 10.57     | 6.14    | 0.744   | GOOD   |
| std       | 10.43    | 12.94     | 3.94    | 0.560   | GOOD   |
| min       | -4.88    | -9.77     | 10.21   | 0.664   | GOOD   |
| max       | 26.08    | 31.42     | 10.91   | 0.670   | GOOD   |
| range     | 30.96    | 41.20     | 13.53   | 0.470   | GOOD   |
| median    | 11.92    | 10.40     | 6.53    | 0.815   | GOOD   |
| q25       | 5.35     | 2.64      | 6.86    | 0.701   | GOOD   |
| q75       | 21.45    | 18.35     | 7.10    | 0.630   | GOOD   |

**Extreme statistics (p < 0.05)**: 0/8

### Visual Evidence

`ppc_test_statistics.png` shows all observed statistics (red lines) fall within the bulk of the replicated distributions (green histograms), with no systematic bias.

### Interpretation

The model successfully captures:
- Central tendency (mean, median)
- Dispersion (std, range, quartiles)
- Extremes (min, max)

This indicates the hierarchical model adequately represents the data-generating process, even though it doesn't improve predictions over Model 1.

---

## 5. Residual Analysis

### Summary

- **Mean absolute residual**: 7.30
- **Mean standardized residual**: 0.08 (near zero, good)
- **Extreme residuals (|z| > 2)**: 0/8

### Residual Patterns

All residual plots in `ppc_residuals.png` show:

1. **No systematic patterns**: Residuals scatter randomly around zero
2. **No outliers**: All standardized residuals |z| < 2
3. **Homoscedasticity**: Residuals do not increase with predicted values or measurement error
4. **Approximate normality**: Q-Q plot shows reasonable agreement with normal distribution
5. **No trend with observation index**: No sequential patterns

### Visual Evidence

The 6-panel residual plot (`ppc_residuals.png`) confirms:
- Random scatter in residuals vs observation number
- Random scatter in residuals vs predicted values
- Random scatter in residuals vs measurement error (sigma)
- Q-Q plot shows approximate normality
- Histogram of standardized residuals overlaps well with N(0,1)

**Conclusion**: No evidence of model misspecification in residual structure.

---

## 6. Calibration Assessment (LOO-PIT)

### LOO-PIT Results

`ppc_calibration.png` shows two views of the probability integral transform:

1. **ECDF plot**: The empirical CDF closely follows the diagonal (ideal calibration)
2. **Histogram**: The distribution is approximately uniform

### Interpretation

Good calibration indicates:
- Predictive intervals have correct coverage
- No systematic over/under-confidence
- Model uncertainty is well-calibrated

**Note**: Some irregularity expected with only n=8 observations.

---

## 7. Model Comparison: Why Prefer Model 1?

### Evidence Summary

| Criterion | Model 1 | Model 2 | Winner |
|-----------|---------|---------|--------|
| **ELPD LOO** | -32.05 ± 1.43 | -32.16 ± 1.09 | Tie (Δ < 2×SE) |
| **Parameters** | 1 (mu) | 10 (mu, tau, θ₁...θ₈) | Model 1 |
| **Max Pareto k** | 0.37 (GOOD) | 0.87 (BAD) | Model 1 |
| **Robustness** | Handles all observations well | Sensitive to Obs 5 | Model 1 |
| **Interpretability** | Simple pooling | Complex hierarchy | Model 1 |
| **Posterior certainty** | mu well-defined | tau uncertain | Model 1 |

### Parsimony Principle

When two models have equivalent predictive performance, prefer the simpler one:

**Model 1** (Complete Pooling):
```
y_i ~ Normal(mu, sigma_i)
mu ~ Normal(10, 20)
```
- 1 parameter estimated from data
- Direct interpretation: "common mean across all observations"
- Stable, reliable LOO estimates

**Model 2** (Hierarchical Partial Pooling):
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
mu ~ Normal(10, 20)
tau ~ Half-Normal(0, 10)
```
- 10 parameters estimated from data
- Complex interpretation: "group means shrunk toward common mean"
- Unstable LOO for one observation
- tau very uncertain (not clearly > 0)

### Theoretical Support

**From EDA** (`experiments/exploratory_data_analysis/eda_report.md`):
- Variance decomposition: tau² = 0
- Recommendation: "Complete pooling sufficient"

**From Posterior Inference** (`experiments/experiment_2/posterior_inference/posterior_inference_report.md`):
- tau: 5.91 ± 4.16, 95% HDI [0.28, 15.65]
- "tau is VERY UNCERTAIN and compatible with zero"
- "No clear evidence of heterogeneity"

**From This PPC**:
- No improvement in predictive accuracy
- Worse Pareto k diagnostics
- More parameters without benefit

**Conclusion**: The data do not require a hierarchical model.

---

## 8. Limitations and Caveats

### Small Sample Size

With only n=8 observations:
- LOO-CV is appropriate (leave-one-out)
- But uncertainty in comparisons is large (SE = 0.36)
- Difficult to detect small improvements in predictive performance

### One Problematic Observation

Observation 5 (y = -4.88) is:
- The most extreme value
- Causes high Pareto k in Model 2
- Yet still well-predicted by both models

This suggests the hierarchical structure creates unnecessary sensitivity without improving predictions.

### Model Adequacy

Both models:
- Capture all test statistics
- Show good calibration
- Have reasonable residuals
- Predict all observations well

The issue is not model **adequacy** (both are adequate), but model **preference** (which to use).

---

## 9. Implications for Model Critique

### Status Summary

| Validation Stage | Model 1 | Model 2 |
|------------------|---------|---------|
| Prior Predictive Check | PASS | PASS |
| Simulation-Based Calibration | PASS | PASS |
| Posterior Inference | PASS | PASS |
| **Posterior Predictive Check** | **PASS** | **PASS*** |

*Model 2 passes adequacy checks but is not preferred for use.

### Decision Matrix

**Model 2 Adequacy**: ADEQUATE
- All observation-level fits acceptable (p > 0.05)
- All test statistics captured
- Good calibration
- No systematic residual patterns

**Model 2 Preference**: NOT PREFERRED
- No improvement over Model 1 (Δ ELPD = -0.11 ± 0.36)
- More complex (10 vs 1 parameter)
- Less robust (high Pareto k for Obs 5)
- Theoretical reasons favor Model 1 (tau ≈ 0)

### Recommendation for Model Critique

**REJECT Model 2** in favor of **Model 1** with the following justification:

1. **Equivalent predictive performance**: Models are statistically indistinguishable (|Δ ELPD| < 2×SE)
2. **Parsimony**: Model 1 achieves same predictions with 90% fewer parameters
3. **Robustness**: Model 1 has better Pareto k diagnostics (all k < 0.5 vs max k = 0.87)
4. **Theoretical support**: EDA and posterior both indicate tau ≈ 0 (no heterogeneity)
5. **Interpretability**: Single pooled mean is simpler than hierarchical structure

**The added complexity of hierarchical modeling is not justified by the data.**

---

## 10. Conclusions

### Summary of Findings

1. **LOO-CV Comparison**: Model 2 does not significantly improve predictive performance over Model 1 (Δ ELPD = -0.11 ± 0.36, |Δ| < 2×SE)

2. **Pareto k Diagnostics**: Model 2 has problematic LOO estimates (max k = 0.87) while Model 1 is reliable (max k = 0.37)

3. **Model Adequacy**: Both models adequately fit the data (all p-values > 0.05, good calibration, no residual patterns)

4. **Model Preference**: Model 1 preferred by parsimony (1 vs 10 parameters, equivalent predictions)

5. **Theoretical Consistency**: Results consistent with EDA (tau² = 0) and posterior uncertainty in tau

### Answer to Key Question

**Does the hierarchical model improve predictive performance over Model 1?**

**NO**. The hierarchical partial pooling model (Model 2) provides no improvement in out-of-sample predictive accuracy compared to the simpler complete pooling model (Model 1). The added complexity is not justified by the data.

### Final Recommendation

**Use Model 1 (Complete Pooling) for inference and prediction.**

Rationale:
- Simpler (1 parameter vs 10)
- Equally accurate (Δ ELPD not significant)
- More robust (better Pareto k values)
- Theoretically justified (no evidence of heterogeneity)
- Easier to interpret and communicate

---

## Files Generated

### Code
- `/workspace/experiments/experiment_2/posterior_predictive_check/code/posterior_predictive_check.py`

### Visualizations
- `/workspace/experiments/experiment_2/posterior_predictive_check/plots/loo_comparison.png`
- `/workspace/experiments/experiment_2/posterior_predictive_check/plots/loo_pareto_k.png`
- `/workspace/experiments/experiment_2/posterior_predictive_check/plots/ppc_observations.png`
- `/workspace/experiments/experiment_2/posterior_predictive_check/plots/ppc_test_statistics.png`
- `/workspace/experiments/experiment_2/posterior_predictive_check/plots/ppc_residuals.png`
- `/workspace/experiments/experiment_2/posterior_predictive_check/plots/ppc_calibration.png`

### Report
- `/workspace/experiments/experiment_2/posterior_predictive_check/ppc_findings.md` (this document)

---

## Appendix: Technical Details

### Posterior Predictive Generation

For each of 8000 posterior samples (4 chains × 2000 draws):
```
For i = 1 to 8000:
    For j = 1 to 8:
        y_pred[i, j] ~ Normal(theta[i, j], sigma_obs[j])
```

This generates 8000 replicated datasets for comparison with observed data.

### LOO-CV Methodology

Leave-One-Out Cross-Validation using Pareto Smoothed Importance Sampling (PSIS):
- Each observation held out in turn
- Predictive density computed using remaining n-1 observations
- Importance sampling used to approximate left-out predictions
- Pareto k estimates reliability of importance sampling

**Interpretation**:
- k < 0.5: Reliable estimates (GOOD)
- k = 0.5 to 0.7: Less reliable (OK)
- k > 0.7: Unreliable, consider K-fold CV (BAD)

### Test Statistics

Eight summary statistics computed for observed and replicated datasets:
1. Mean
2. Standard deviation
3. Minimum
4. Maximum
5. Range
6. Median
7. 25th percentile
8. 75th percentile

Two-tailed p-value: P(T_rep ≥ T_obs) or P(T_rep ≤ T_obs), whichever is smaller, times 2.

### Software and Versions

- Python 3.13
- ArviZ (latest)
- NumPy, Pandas, Matplotlib, Seaborn
- CmdStanPy (for fitting, not used in PPC)

---

**Report prepared by**: Claude (Posterior Predictive Check Agent)
**Date**: 2025-10-28
**Experiment**: 2 (Hierarchical Partial Pooling Model)
**Recommendation**: PREFER MODEL 1 BY PARSIMONY
