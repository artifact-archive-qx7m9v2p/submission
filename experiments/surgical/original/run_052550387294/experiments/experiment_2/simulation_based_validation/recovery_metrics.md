# Simulation-Based Calibration: Recovery Metrics

**Model**: Hierarchical Logit Model (Experiment 2)
**Date**: 2025-10-30
**Analyst**: Model Validation Specialist
**Inference Method**: MAP + Laplace Approximation (scipy.optimize)

---

## Visual Assessment

This report is structured around visual evidence from diagnostic plots:

1. **`sbc_rank_histograms.png`**: Primary calibration diagnostic showing rank uniformity
2. **`coverage_calibration.png`**: Interval coverage and parameter recovery visualization
3. **`parameter_recovery_scatter.png`**: Bias patterns and uncertainty calibration
4. **`posterior_contraction.png`**: Learning from data assessment
5. **`parameter_space_identifiability.png`**: Joint parameter recovery patterns
6. **`sbc_comprehensive_summary.png`**: Integrated overview of all diagnostics

---

## Executive Summary

Simulation-based calibration reveals **CATASTROPHIC FAILURE** for both parameters:
- **μ_logit**: Poor calibration (40.7% coverage vs 95% target)
- **σ**: Complete failure (2.0% coverage vs 95% target)

Both parameters show severe under-coverage and biased recovery. The Laplace approximation is grossly underestimating uncertainty.

**Status**: **CRITICAL FAILURE** - Do not proceed to real data fitting.

---

## 1. Coverage Rates (Target: 0.95 for 95% CI)

As illustrated in **`coverage_calibration.png`** (right panels):

### μ_logit (Population Mean on Logit Scale)

| Interval | Coverage | Target | Status |
|----------|----------|--------|--------|
| 95% CI | 0.407 | 0.950 | **FAILED** ❌ |
| 90% CI | 0.327 | 0.900 | **FAILED** ❌ |
| 80% CI | 0.227 | 0.800 | **FAILED** ❌ |

**Visual Evidence**: In `coverage_calibration.png` (top-right), approximately 60% of intervals are RED (truth outside CI), when only 5% should be red. The intervals are systematically **too narrow**.

### σ (Scale Parameter)

| Interval | Coverage | Target | Status |
|----------|----------|--------|--------|
| 95% CI | 0.020 | 0.950 | **CATASTROPHIC FAILURE** ❌❌❌ |
| 90% CI | 0.013 | 0.900 | **CATASTROPHIC FAILURE** ❌❌❌ |
| 80% CI | 0.007 | 0.800 | **CATASTROPHIC FAILURE** ❌❌❌ |

**Visual Evidence**: In `coverage_calibration.png` (bottom-right), **98% of intervals are RED**. The model is recovering σ with extreme over-confidence - credible intervals miss the truth 98% of the time.

**Critical Finding**: Even 80% credible intervals only capture the true σ 0.7% of the time. This indicates the posterior is orders of magnitude too confident.

---

## 2. Rank Statistics (Uniformity Test)

As shown in **`sbc_rank_histograms.png`**:

### μ_logit

- **χ² statistic**: 567.6 (expected: ~19 for 20 bins)
- **p-value**: < 0.0001 (expected: > 0.05)
- **Pattern**: Extreme bimodal distribution with spikes at ranks 0-200 and 3800-4000
- **Interpretation**: Posterior is either grossly underestimating or overestimating μ_logit

**Visual Evidence**: Left panel shows U-shaped histogram - ranks concentrate at the extremes. This indicates the posterior is frequently completely wrong about where the true value lies.

### σ

- **χ² statistic**: 2770.8 (expected: ~19)
- **p-value**: < 0.0001
- **Pattern**: Massive spike at rank 0-200 (nearly all posteriors below true value)
- **Interpretation**: Posterior systematically underestimates σ and is far too confident

**Visual Evidence**: Right panel shows catastrophic spike at rank 0. The true σ value is almost always in the upper tail of the posterior (rank near 0 means true value is smaller than almost all posterior samples). This means **the model is systematically overestimating σ while being overconfident**.

**Wait, there's a contradiction**: The rank histogram shows ranks near 0 (posterior overestimates), but the bias is positive (+1.0453), suggesting overestimation. Let me verify...

Actually, rank 0 means true_value < posterior_samples, so the posterior is sampling values HIGHER than truth. This is consistent with positive bias (+1.05).

---

## 3. Bias (Posterior Mean - True Value)

As shown in **`parameter_recovery_scatter.png`** (right panels):

### μ_logit

- **Mean bias**: -0.0162 (negligible)
- **Bias range**: [-1.0, +0.8] (highly variable)
- **Pattern**: No systematic directional bias, but high variance

**Visual Evidence**: Top-right panel shows bias centered near zero but with substantial scatter. The issue isn't systematic bias, but rather **poor uncertainty quantification**.

### σ

- **Mean bias**: +1.0453 (severe overestimation)
- **Bias range**: [+0.3, +1.9]
- **Pattern**: Consistent overestimation across all true values

**Visual Evidence**: Bottom-right panel shows ALL points above zero (positive bias). When true σ = 0.5, estimated σ ≈ 1.5. When true σ = 2.0, estimated σ ≈ 2.5.

**Critical Pattern**: The bias is not uniform - it's worst for small true σ values. The scatter plot shows the bias decreases slightly as true σ increases, suggesting a **floor effect** or optimization issue.

---

## 4. Root Mean Squared Error (RMSE)

### μ_logit
- **RMSE**: 0.3544
- **Prior SD**: 1.0
- **Ratio**: 0.35 (reasonable prediction accuracy)

### σ
- **RMSE**: 1.0875
- **Prior SD**: 0.6028 (half-normal)
- **Ratio**: 1.80 (predictions worse than prior!)

**Critical Finding**: For σ, the RMSE is 80% larger than the prior SD. The data is making predictions **worse** than just using the prior. This indicates fundamental failure in σ estimation.

---

## 5. Posterior Contraction

As illustrated in **`posterior_contraction.png`**:

### μ_logit

- **Prior SD**: 1.000
- **Mean Posterior SD**: 0.0785
- **Contraction ratio**: 0.079

**Interpretation**: Posterior is 92% narrower than prior, indicating **extremely strong learning**. This is actually TOO much contraction given the poor coverage.

**Visual Evidence**: Left panel shows posterior SD scattered around 0.08, well below the prior SD line at 1.0. The posteriors are systematically too narrow (overconfident).

### σ

- **Prior SD**: 0.603 (half-normal)
- **Mean Posterior SD**: 0.219
- **Contraction ratio**: 0.364

**Interpretation**: Posterior is 64% narrower than prior. This suggests substantial learning, but the poor coverage indicates this contraction is **excessive** - posteriors are too confident.

**Visual Evidence**: Right panel shows posterior SD scattered around 0.22, below the prior SD line. Combined with 2% coverage, this confirms extreme over-confidence.

---

## 6. Parameter Space Coverage

As illustrated in **`parameter_space_identifiability.png`**:

**Visual Evidence**: Right panel shows that nearly ALL points are RED (at least one parameter not covered). Only a tiny fraction are green (both parameters covered).

**Joint Coverage**: Approximately 1-2% of simulations have BOTH μ_logit AND σ within their 95% credible intervals simultaneously.

**Interpretation**: The parameters are either:
1. Not jointly identifiable from N=12 trials
2. Strongly correlated in ways the Laplace approximation cannot capture
3. Both suffering from over-confident posterior approximation

---

## 7. Z-Score Distribution

From **`zscore_distribution.png`**:

### μ_logit
- **KS test p-value**: ~0.000
- **Pattern**: Heavy tails and bimodal structure
- **Interpretation**: Z-scores deviate from standard normal - calibration failure

### σ
- **KS test p-value**: ~0.000
- **Pattern**: Extreme deviation from normality, highly right-skewed
- **Interpretation**: Massive overcalibration (posterior too confident)

**Expected**: If calibrated, z-scores should follow standard Normal(0,1)
**Observed**: Extreme deviations indicate miscalibration

---

## 8. Convergence and Computational Health

### Success Rate
- **Successful fits**: 150/150 (100%)
- **Status**: Excellent ✓

### Convergence Issues
- **Divergences**: 0 (N/A for Laplace approximation)
- **Optimization failures**: 0
- **Status**: Excellent ✓

**Critical Note**: The optimization converges reliably, but the **Laplace approximation is inappropriate** for this model. The posterior geometry is not approximately normal.

---

## Critical Visual Findings

### 1. Overconfident Posteriors (Multiple Plots)

**Evidence**:
- Coverage plots show overwhelming RED intervals
- Contraction plots show excessive narrowing
- Rank histograms show extreme non-uniformity

**Conclusion**: The Laplace approximation drastically underestimates posterior uncertainty.

### 2. σ Recovery Failure (All σ Panels)

**Evidence**:
- 98% coverage failure rate
- Systematic positive bias (+1.05)
- Rank histogram spike at rank 0
- RMSE worse than prior

**Conclusion**: σ is essentially **not estimable** with this approach and N=12 trials.

### 3. μ_logit Partial Failure (All μ_logit Panels)

**Evidence**:
- 60% coverage failure rate
- Reasonable bias but poor intervals
- Bimodal rank distribution
- Excessive contraction

**Conclusion**: μ_logit estimates are directionally correct but uncertainty is severely underestimated.

---

## Comparison to Beta-Binomial (Experiment 1)

| Metric | Beta-Binomial (Exp 1) | Hierarchical Logit (Exp 2) |
|--------|----------------------|---------------------------|
| Location parameter coverage | 96.6% (μ) ✓ | 40.7% (μ_logit) ❌ |
| Scale parameter coverage | 45.6% (φ) ❌ | 2.0% (σ) ❌❌ |
| Location bias | 0.013 ✓ | -0.016 ✓ |
| Scale bias | -2.185 ❌ | +1.045 ❌ |
| Success rate | 99.3% ✓ | 100% ✓ |

**Key Differences**:
1. **Hierarchical Logit is WORSE**: Even μ_logit fails where Beta-Binomial μ succeeded
2. **σ fails harder than φ**: 2% vs 45% coverage
3. **Different bias direction**: φ underestimated, σ overestimated

**Hypothesis**: The Laplace approximation may be particularly poor for hierarchical models with 12 nuisance parameters (η_i). The posterior geometry in 14-dimensional space (μ_logit, log(σ), 12 η's) is likely highly non-normal.

---

## Root Cause Analysis

### Why is Calibration Failing?

#### 1. Laplace Approximation Inadequacy (Primary Cause)

The Laplace approximation assumes the posterior is approximately normal around the MAP estimate. This fails when:

- **High dimensionality**: 14 parameters (μ_logit, σ, 12 η's)
- **Funnel geometry**: Hierarchical models create "funnel" posteriors even with non-centered parameterization
- **Boundary effects**: σ > 0 constraint creates non-normal geometry
- **Correlation structure**: Parameters correlated in complex ways

**Evidence**:
- Excessive posterior contraction (overconfident)
- Non-normal z-scores
- Bimodal rank distributions

#### 2. Weak Identifiability (Contributing Factor)

With only N=12 trials:
- Limited information about σ (overdispersion scale)
- η_i values partially confounded with σ
- Small N means high posterior correlation between μ_logit and σ

**Evidence**:
- σ coverage near 0%
- Parameter space plot shows poor joint recovery
- Similar to φ failure in Beta-Binomial

#### 3. Optimization Landscape Issues

The MAP optimization may be finding local optima or regions where the Hessian (curvature) is poorly estimated.

**Evidence**:
- Systematic bias in σ
- Floor effect (bias worse for small σ)

---

## Implications for Real Data Analysis

### If we fit this model to real data using Laplace approximation:

1. **μ_logit estimates**:
   - Point estimates may be reasonable
   - But 95% CIs will miss truth 60% of the time
   - **Over-confident about location**

2. **σ estimates**:
   - Will systematically overestimate overdispersion
   - Bias of ~+1.0 on σ scale
   - 95% CIs will miss truth 98% of the time
   - **Completely unreliable**

3. **Scientific conclusions**:
   - Will conclude higher overdispersion than reality
   - Will make over-confident claims about precision
   - **Invalid for publication**

---

## Recommendations

### Immediate Actions (Before Real Data Fitting)

1. **STOP using Laplace approximation**
   - Fundamentally inappropriate for hierarchical models
   - Would need full MCMC (Stan, PyMC, etc.)

2. **Install proper MCMC tools**
   - Fix Stan compilation (requires make/compiler)
   - Or install PyMC (recommended)
   - Or use numpyro/JAX

3. **Re-run SBC with full MCMC**
   - Test if calibration improves with proper inference
   - May still find σ weakly identifiable, but intervals will be honest

### If MCMC Still Shows Poor Calibration

4. **Consider model simplification**
   - Fixed-effects logistic regression (no hierarchy)
   - Or stronger priors on σ
   - Or partial pooling with informative prior

5. **Accept N=12 limitation**
   - May need to report that overdispersion scale is not reliably estimable
   - Focus on μ_logit which may be identifiable with MCMC

6. **Collect more data**
   - Need N >> 12 trials for reliable σ estimation
   - Likely N ≥ 50-100 trials needed

---

## Pass/Fail Assessment

### PASS Criteria (from task specification)

| Criterion | μ_logit | σ | Overall |
|-----------|---------|---|---------|
| Coverage ∈ [0.90, 0.98] | 0.407 ❌ | 0.020 ❌ | **FAIL** |
| Bias ≈ 0 | -0.016 ✓ | +1.045 ❌ | **FAIL** |
| Uniform ranks | χ²=568 ❌ | χ²=2771 ❌ | **FAIL** |
| Posterior contraction | 0.079 ✓ | 0.364 ✓ | PASS |
| Convergence >90% | 100% ✓ | 100% ✓ | PASS |

### FAIL Criteria (from task specification)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Coverage < 0.85 | **YES** ❌ | Both parameters < 0.85 |
| Systematic bias | **YES** ❌ | σ bias = +1.045 |
| Non-uniform ranks | **YES** ❌ | χ² >> expected |
| Convergence failures | NO ✓ | 100% success |
| No posterior contraction | NO ✓ | Both show contraction |

**Result**: **FAILED** on 3/5 fail criteria

---

## Final Verdict

**STATUS**: **CATASTROPHIC FAILURE** ❌❌❌

### Summary

| Parameter | Recovery | Coverage | Bias | Verdict |
|-----------|----------|----------|------|---------|
| **μ_logit** | Poor | 40.7% | -0.016 | **FAIL** |
| **σ** | Catastrophic | 2.0% | +1.045 | **FAIL** |

### Why This Happened

1. **Laplace approximation inadequate** for hierarchical model posterior geometry
2. **Overconfident uncertainty estimates** from Hessian-based covariance
3. **Weak identifiability** of σ with N=12 trials (same as Beta-Binomial φ)
4. **High-dimensional optimization** (14 parameters) with complex correlations

### What This Means

The current inference approach (MAP + Laplace) cannot reliably estimate parameters for the Hierarchical Logit Model with N=12 trials.

**The model itself may be valid**, but we cannot verify this without proper MCMC inference.

### Required Actions

1. ✓ SBC validation completed (caught the problem!)
2. ❌ **STOP** - Do not proceed to real data fitting
3. ⚠️ **CRITICAL** - Must implement full MCMC before proceeding
4. 🔄 Re-run validation with MCMC
5. 📊 Only then proceed to real data (if validation passes)

---

## Files Generated

All results saved to `/workspace/experiments/experiment_2/simulation_based_validation/`:

### Code
- `hierarchical_logit.stan`: Stan model (compilation failed)
- `run_sbc_scipy.py`: SBC implementation using MAP + Laplace approximation
- `visualize_sbc.py`: Diagnostic visualization code

### Results
- `results/sbc_results.csv`: Raw SBC results (150 iterations)
- `results/sbc_summary.json`: Quantitative metrics
- `results/sbc_log.txt`: Complete execution log

### Plots (Visual Evidence)
- `plots/sbc_rank_histograms.png`: ⚠️ Primary evidence of calibration failure
- `plots/coverage_calibration.png`: ⚠️ Shows 98% failure rate for σ
- `plots/parameter_recovery_scatter.png`: ⚠️ Reveals systematic bias
- `plots/posterior_contraction.png`: ⚠️ Shows overconfidence
- `plots/parameter_space_identifiability.png`: ⚠️ Joint coverage failure
- `plots/zscore_distribution.png`: Miscalibration evidence
- `plots/sbc_comprehensive_summary.png`: Integrated overview

### Reports
- `recovery_metrics.md`: This document
- `findings.md`: Executive summary and decision

---

**Validation Date**: 2025-10-30
**Validation Status**: FAILED
**Required Action**: Implement full MCMC and re-validate before proceeding
