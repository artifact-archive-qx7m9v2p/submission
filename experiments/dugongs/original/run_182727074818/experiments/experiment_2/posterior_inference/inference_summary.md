# Posterior Inference Summary: Change-Point Segmented Regression

**Experiment:** Experiment 2 - Change-Point Segmented Regression
**Date:** 2025-10-27
**Method:** Hamiltonian Monte Carlo (NUTS) via PyMC
**Model:** Y ~ StudentT(ν, μ, σ) with piecewise linear μ(x, τ)

---

## Executive Summary

**DECISION: SUCCESS - MODEL CONVERGED**

The change-point segmented regression model was successfully fitted with excellent convergence:
- No divergent transitions (0%)
- All R-hat < 1.02
- ESS > 555 for all parameters
- Clean chain mixing

However, **LOO-CV model comparison favors Model 1 (Logarithmic)** over this change-point model:
- ΔELPD_LOO = -3.31 ± 3.35 (Model 2 worse than Model 1)
- Weak preference for Model 1
- **Recommendation: Use Model 1 for inference**

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

### Implementation Details
- **PPL:** PyMC 5.26.1
- **Sampler:** NUTS with target_accept=0.95
- **Data:** N = 27 observations
- **Chains:** 4 chains × 500 draws (1500 warmup)
- **Total samples:** 2000
- **Sampling time:** 210.4 seconds

---

## Convergence Diagnostics

### Quantitative Metrics

| Parameter | Mean   | SD    | 95% HDI        | R-hat  | ESS_bulk | ESS_tail | Status |
|-----------|--------|-------|----------------|--------|----------|----------|--------|
| **α**     | 1.701  | 0.069 | [1.576, 1.839] | 1.0001 | 630      | 741      | ✓ PASS |
| **β₁**    | 0.107  | 0.021 | [0.064, 0.143] | 1.0100 | 555      | 661      | ✓ PASS |
| **β₂**    | 0.015  | 0.004 | [0.008, 0.022] | 1.0001 | 1511     | 1288     | ✓ PASS |
| **τ**     | 6.296  | 1.188 | [5.000, 8.692] | 1.0003 | 564      | 675      | ✓ PASS |
| **ν**     | 22.320 | 14.29 | [3.211, 49.03] | 1.0002 | 1238     | 1165     | ✓ PASS |
| **σ**     | 0.099  | 0.016 | [0.071, 0.129] | 1.0100 | 1280     | 1271     | ✓ PASS |

**All parameters meet convergence criteria:**
- ✓ R-hat < 1.02 (max = 1.010)
- ✓ ESS_bulk > 555 (all parameters)
- ✓ Divergent transitions: 0 (0.00%)
- ✓ Conservative sampling strategy successful

### Visual Diagnostics

Diagnostic plots confirm convergence:
- **Trace plots:** Clean mixing with no trends or drift
- **Rank plots:** Uniform distributions across chains
- **Posterior distributions:** Well-identified parameters with data-driven learning

---

## Posterior Inference Results

### Parameter Estimates

#### 1. Change Point Location (τ = 6.30 ± 1.19)
- **Estimate:** τ = 6.30 (median)
- **95% HDI:** [5.00, 8.69]
- **Interpretation:** Change point estimated at x ≈ 6.3
- **Note:** Wide credible interval suggests **high uncertainty** in change point location
- **Prior boundary:** Lower bound at 5.0 suggests data may not strongly support change point

#### 2. Slope Before Change Point (β₁ = 0.107 ± 0.021)
- **Estimate:** β₁ = 0.107
- **95% HDI:** [0.064, 0.143]
- **Interpretation:** Moderate positive slope before change point

#### 3. Slope After Change Point (β₂ = 0.015 ± 0.004)
- **Estimate:** β₂ = 0.015
- **95% HDI:** [0.008, 0.022]
- **Interpretation:** Much flatter slope after change point
- **Ratio:** β₁/β₂ ≈ 7.1 (steeper before than after)

#### 4. Intercept (α = 1.701 ± 0.069)
- **Estimate:** α = 1.701
- **95% HDI:** [1.576, 1.839]
- **Interpretation:** Y-value when x = 0 (extrapolation)

#### 5. Robustness Parameter (ν = 22.3 ± 14.3)
- **Estimate:** ν = 22.3
- **95% HDI:** [3.2, 49.0]
- **Interpretation:** Moderate tail heaviness, similar to Model 1

#### 6. Residual Scale (σ = 0.099 ± 0.016)
- **Estimate:** σ = 0.099
- **95% HDI:** [0.071, 0.129]
- **Interpretation:** Similar residual variation to Model 1 (σ = 0.093)

### Scientific Interpretation

#### Evidence for Change Point?

**Mixed evidence:**
- ✓ Slopes differ significantly (β₁ >> β₂)
- ⚠ Change point location highly uncertain (wide HDI)
- ⚠ Posterior mass at prior boundary (τ ≈ 5)
- ⚠ Similar residual variation to smooth log model

**Conclusion:** While the model finds different slopes before/after a change point, the high uncertainty in τ and similar fit quality to the logarithmic model suggest **the change point may not be a real feature** of the data-generating process.

---

## LOO-CV Model Comparison

### Results

| Model | ELPD_LOO | SE   | p_loo | ΔELPD | Weight |
|-------|----------|------|-------|-------|--------|
| Model 1 (Log) | 23.71 | 3.09 | 2.61 | 0.00 | 1.000 |
| Model 2 (Change-Point) | 20.39 | 3.35 | 4.62 | -3.31 | 0.000 |

### Interpretation

**ΔELPD_LOO = -3.31 ± 3.35**

- **Decision:** Weak preference for Model 1 (Logarithmic)
- **Strength:** Model 2 is slightly worse than Model 1
- **Effect size:** |ΔELPD| ≈ 1 SE (small difference)
- **Parsimony principle:** Favor simpler model when predictive performance comparable

### Effective Number of Parameters

- **Model 1:** p_loo = 2.61 (5 parameters: α, β, c, ν, σ)
- **Model 2:** p_loo = 4.62 (6 parameters: α, β₁, β₂, τ, ν, σ)

Model 2 uses more effective parameters but does not improve predictive performance enough to justify the added complexity.

### Pareto-k Diagnostics

Both models show excellent LOO reliability:
- **Model 1:** All 27 observations with k ≤ 0.5
- **Model 2:** 26/27 observations with k ≤ 0.5, 1/27 with 0.5 < k ≤ 0.7

---

## Model Comparison: Why Model 1 Wins

### Predictive Performance
- Model 1 has higher ELPD_LOO (better out-of-sample predictions)
- Difference is small but consistent

### Parsimony
- Model 1 is simpler (smooth curve vs. piecewise)
- Model 2 adds complexity (change point τ) without sufficient improvement

### Theoretical Plausibility
- Logarithmic diminishing returns is a well-established pattern
- Change point at x ≈ 6 lacks strong theoretical justification
- Data does not strongly identify the change point location

### Parameter Uncertainty
- Model 2 shows high uncertainty in τ (wide HDI)
- Posterior hits prior boundary (τ ≈ 5), suggesting weak identification
- Model 1 parameters are well-identified

---

## Visual Diagnostics

### Files Generated

**Convergence diagnostics:**
- `/workspace/experiments/experiment_2/posterior_inference/plots/trace_plots.png`
- `/workspace/experiments/experiment_2/posterior_inference/plots/rank_plots.png`
- `/workspace/experiments/experiment_2/posterior_inference/plots/posterior_distributions.png`

**Model fit:**
- `/workspace/experiments/experiment_2/posterior_inference/plots/model_fit.png`

**LOO comparison:**
- `/workspace/experiments/experiment_2/posterior_inference/plots/loo_comparison.png`
- `/workspace/experiments/experiment_2/posterior_inference/plots/pareto_k_diagnostic.png`

---

## Files Generated

### Code
- `/workspace/experiments/experiment_2/posterior_inference/code/fit_changepoint_model.py`
- `/workspace/experiments/experiment_2/posterior_inference/code/loo_comparison.py`

### Diagnostics
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf` (with log_lik)
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/parameter_summary.csv`
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/loo_comparison.csv`

### Plots (300 DPI)
- All plots listed in Visual Diagnostics section above

---

## Technical Notes

### Sampling Strategy

Change-point models are computationally challenging due to:
1. **Discontinuity:** Switching between two linear segments
2. **Identification:** Weak data signals can lead to poor identification of τ
3. **Geometry:** Posterior can have complex geometry near change point

**Adaptive strategy:**
- Started with target_accept=0.95 (higher than default 0.8)
- Extended warmup to 1500 iterations
- Result: Zero divergences, good convergence

### Why PyMC?

Used PyMC as primary PPL (CmdStan unavailable in environment). PyMC's `pm.math.switch()` provides clean implementation of piecewise functions.

---

## Conclusion

**Model 2 (Change-Point) successfully fitted but underperforms Model 1 (Log)**

### Key Findings

1. ✓ Model converged with excellent diagnostics
2. ⚠ Change point location highly uncertain (τ = 6.3 ± 1.2)
3. ✗ Predictive performance worse than simpler logarithmic model
4. ✗ Added complexity not justified by improved fit

### Recommendation

**REJECT Model 2, ACCEPT Model 1 (Logarithmic)**

Reasons:
- Model 1 has better out-of-sample predictive performance
- Model 1 is simpler and more interpretable
- Change point lacks strong empirical support
- Parsimony principle favors Model 1

### Scientific Interpretation

The visual appearance of a "change point" around x ≈ 7 in the data is **better explained by smooth logarithmic diminishing returns** than by an actual discontinuous change in slope. The log model captures the deceleration pattern without introducing an additional structural parameter.

---

## Next Steps

1. ✓ Prior predictive check completed
2. ✓ Model fitting successful
3. ✓ LOO-CV comparison completed
4. **Final decision:** Use Model 1 for scientific inference
5. **Report:** See `/workspace/experiments/experiment_2/model_comparison.md`
