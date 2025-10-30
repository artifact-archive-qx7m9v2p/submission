# Experiment 2: Change-Point Segmented Regression

**Date:** 2025-10-27
**Status:** COMPLETED
**Model Type:** Piecewise Linear Regression with Unknown Change Point
**Data:** `/workspace/data/data.csv` (N=27)

---

## Quick Summary

**Purpose:** Test if apparent change point at x≈7 is real feature of data

**Result:** Model 1 (Logarithmic) is preferred
- ΔELPD_LOO = -3.31 ± 3.35 (Model 2 worse than Model 1)
- Change point location highly uncertain (τ = 6.3 ± 1.2)
- Smooth log curve better explains data than abrupt change

**Decision:** REJECT Model 2, ACCEPT Model 1

---

## Model Specification

### Likelihood
```
Y_i ~ StudentT(ν, μ_i, σ)

μ_i = α + β₁·x_i                  if x_i ≤ τ
μ_i = α + β₁·τ + β₂·(x_i - τ)    if x_i > τ
```

### Priors
```
α ~ Normal(1.8, 0.3)
β₁ ~ Normal(0.15, 0.1)
β₂ ~ Normal(0.02, 0.05)
τ ~ Uniform(5, 12)
ν ~ Gamma(2, 0.1)
σ ~ HalfNormal(0.15)
```

### Parameters (6 total)
- **α:** Intercept
- **β₁:** Slope before change point
- **β₂:** Slope after change point
- **τ:** Change point location
- **ν:** Degrees of freedom (robustness)
- **σ:** Residual scale

---

## Results Summary

### Convergence
- ✓ All R-hat < 1.02
- ✓ All ESS > 555
- ✓ Zero divergent transitions
- ✓ Sampling time: 210 seconds

### Parameter Estimates

| Parameter | Mean   | SD    | 95% HDI        | Interpretation |
|-----------|--------|-------|----------------|----------------|
| α         | 1.701  | 0.069 | [1.576, 1.839] | Intercept |
| β₁        | 0.107  | 0.021 | [0.064, 0.143] | Steep initial slope |
| β₂        | 0.015  | 0.004 | [0.008, 0.022] | Flat later slope |
| τ         | 6.296  | 1.188 | [5.000, 8.692] | Change point (uncertain!) |
| ν         | 22.320 | 14.29 | [3.211, 49.03] | Moderate robustness |
| σ         | 0.099  | 0.016 | [0.071, 0.129] | Residual variation |

### Key Finding

**Change point is poorly identified:**
- Wide 95% HDI: [5.0, 8.7] spans 3.7 units
- Posterior mass at prior boundary (τ ≈ 5)
- Suggests data does not strongly support change-point hypothesis

### LOO-CV Comparison

| Model | ELPD_LOO | SE   | Δ from best | Weight |
|-------|----------|------|-------------|--------|
| Model 1 (Log) | 23.71 | 3.09 | 0.00 | 1.000 |
| Model 2 (Change) | 20.39 | 3.35 | -3.31 | 0.000 |

**Interpretation:** Model 1 has better predictive performance with simpler structure

---

## Directory Structure

```
experiment_2/
├── README.md (this file)
├── model_comparison.md (final decision document)
│
├── prior_predictive_check/
│   ├── findings.md
│   ├── code/
│   │   └── prior_predictive_check.py
│   └── plots/
│       └── prior_predictive_check.png
│
└── posterior_inference/
    ├── inference_summary.md (detailed results)
    ├── code/
    │   ├── fit_changepoint_model.py
    │   ├── loo_comparison.py
    │   ├── posterior_predictive_check.py
    │   └── model_comparison_visualization.py
    ├── diagnostics/
    │   ├── convergence_report.md
    │   ├── posterior_inference.netcdf (ArviZ InferenceData with log_lik)
    │   ├── parameter_summary.csv
    │   └── loo_comparison.csv
    └── plots/
        ├── trace_plots.png
        ├── rank_plots.png
        ├── posterior_distributions.png
        ├── model_fit.png
        ├── loo_comparison.png
        ├── pareto_k_diagnostic.png
        ├── posterior_predictive_check.png
        └── model_comparison_visual.png
```

---

## Key Documents

### 1. Model Comparison (START HERE)
**File:** `model_comparison.md`

Comprehensive comparison of Model 1 vs Model 2:
- LOO-CV results
- Why Model 1 wins
- Scientific interpretation
- Final decision and justification

### 2. Inference Summary
**File:** `posterior_inference/inference_summary.md`

Detailed results for Model 2:
- Parameter estimates
- Convergence diagnostics
- LOO-CV analysis
- Technical notes

### 3. Convergence Report
**File:** `posterior_inference/diagnostics/convergence_report.md`

In-depth convergence assessment:
- Quantitative diagnostics (R-hat, ESS, MCSE)
- Visual diagnostics (traces, ranks)
- Sampling efficiency
- Comparison to Model 1

---

## Visualizations

### Model Fit
**Files:**
- `posterior_inference/plots/model_fit.png` - Model 2 fit to data
- `posterior_inference/plots/model_comparison_visual.png` - Side-by-side comparison of Model 1 and 2

### Convergence Diagnostics
**Files:**
- `posterior_inference/plots/trace_plots.png` - MCMC traces
- `posterior_inference/plots/rank_plots.png` - Rank ECDF plots
- `posterior_inference/plots/posterior_distributions.png` - Parameter posteriors

### Model Comparison
**Files:**
- `posterior_inference/plots/loo_comparison.png` - LOO-CV comparison bar chart
- `posterior_inference/plots/pareto_k_diagnostic.png` - Pareto k values for both models

### Validation
**Files:**
- `posterior_inference/plots/posterior_predictive_check.png` - PPC with residual analysis
- `prior_predictive_check/plots/prior_predictive_check.png` - Prior validation

---

## Reproducibility

### Environment
- Python 3.13.9
- PyMC 5.26.1
- ArviZ 0.22.0
- NumPy 2.3.4
- Matplotlib 3.10.7

### Random Seed
All analyses use `random_seed=42` for reproducibility.

### Running the Analysis

```bash
# 1. Prior predictive check
python experiments/experiment_2/prior_predictive_check/code/prior_predictive_check.py

# 2. Fit model
python experiments/experiment_2/posterior_inference/code/fit_changepoint_model.py

# 3. LOO comparison
python experiments/experiment_2/posterior_inference/code/loo_comparison.py

# 4. Posterior predictive check
python experiments/experiment_2/posterior_inference/code/posterior_predictive_check.py

# 5. Visual comparison
python experiments/experiment_2/posterior_inference/code/model_comparison_visualization.py
```

---

## Scientific Interpretation

### What We Tested

**Hypothesis:** The data shows an abrupt change in slope around x ≈ 7, suggesting a change-point structure.

**Alternative:** The apparent change point is an artifact of smooth logarithmic diminishing returns.

### What We Found

1. **Model 2 fits the data well** (100% coverage in posterior predictive check)
2. **But change point location is uncertain** (wide credible interval)
3. **And Model 1 has better predictive performance** (ΔELPD = 3.31 ± 3.35)
4. **Parsimony favors simpler Model 1**

### Conclusion

The visual appearance of a change point is better explained by smooth logarithmic diminishing returns (Model 1) than by an actual structural break (Model 2). While Model 2 is not "wrong" per se, it adds complexity without sufficient improvement in predictive performance.

---

## Why Model 1 Wins

### 1. Predictive Performance
- Higher ELPD_LOO (23.71 vs 20.39)
- Better out-of-sample predictions

### 2. Parsimony
- Simpler structure (5 vs 6 parameters)
- Smooth curve vs piecewise
- Occam's razor applies

### 3. Parameter Identification
- Model 1 parameters well-identified
- Model 2 change point τ poorly identified
- Posterior at prior boundary suggests weak evidence

### 4. Theoretical Plausibility
- Logarithmic diminishing returns is well-established
- Change points rare without external intervention
- Smooth transitions more common in natural processes

---

## Minimum Attempt Policy

This experiment fulfills the minimum attempt policy requirement:
- ✓ Two distinct models fitted (Log and Change-Point)
- ✓ Both models converged successfully
- ✓ Rigorous comparison via LOO-CV
- ✓ Clear decision based on statistical criteria

---

## Next Steps (If Continuing)

1. **Sensitivity analysis:** Test robustness to prior choices
2. **Alternative change-point priors:** Try data-driven priors for τ
3. **Smoothed change-point:** Use sigmoid transition instead of abrupt switch
4. **Model averaging:** Combine predictions from both models
5. **External validation:** Test on new data if available

However, **for current purposes, Model 1 (Logarithmic) is the recommended model** for scientific inference.

---

## References

- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.

- Carlin, B. P., Gelfand, A. E., & Smith, A. F. (1992). Hierarchical Bayesian analysis of changepoint problems. *Journal of the Royal Statistical Society: Series C*, 41(2), 389-405.

---

## Contact

For questions about this analysis, see:
- Model comparison: `model_comparison.md`
- Technical details: `posterior_inference/inference_summary.md`
- Convergence issues: `posterior_inference/diagnostics/convergence_report.md`
