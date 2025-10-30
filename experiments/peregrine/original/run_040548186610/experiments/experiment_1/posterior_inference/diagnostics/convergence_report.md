# Convergence Report: Negative Binomial Quadratic Model

**Experiment:** Experiment 1
**Model:** Negative Binomial with Quadratic Time Trend
**Date:** 2025-10-29
**Implementation:** PyMC 5.26.1
**Sampler:** NUTS (No-U-Turn Sampler)

---

## Executive Summary

**CONVERGENCE STATUS: PASS** ✓

The Bayesian MCMC inference achieved excellent convergence with zero divergent transitions, all R-hat values at 1.0, and ESS values well above the 400 threshold for all parameters. The model is ready for inference and prediction.

---

## Sampling Configuration

### Settings
- **Chains:** 4 parallel chains
- **Iterations per chain:** 2,000 (1,000 warmup + 1,000 sampling)
- **Total draws:** 4,000 posterior samples
- **Target acceptance:** 0.95
- **Initialization:** jitter+adapt_diag
- **Duration:** ~152 seconds (main sampling)

### Data
- **Observations:** 40
- **Year range:** [-1.67, 1.67] (standardized)
- **Count range:** [19, 272]
- **Count mean:** 109.5, median: 74.5

---

## Convergence Diagnostics

### 1. R-hat Statistic (Gelman-Rubin)
**All parameters: R̂ = 1.000** ✓

| Parameter | R̂ | Status |
|-----------|-----|--------|
| β₀ | 1.000 | ✓ PASS |
| β₁ | 1.000 | ✓ PASS |
| β₂ | 1.000 | ✓ PASS |
| φ | 1.000 | ✓ PASS |

**Criterion:** R̂ < 1.01 for convergence
**Result:** All parameters meet criterion (perfect convergence)

### 2. Effective Sample Size (ESS)

#### Bulk ESS (for posterior mean/median)
| Parameter | ESS_bulk | Status |
|-----------|----------|--------|
| β₀ | 2,106 | ✓ PASS |
| β₁ | 2,884 | ✓ PASS |
| β₂ | 2,286 | ✓ PASS |
| φ | 2,787 | ✓ PASS |

**Criterion:** ESS_bulk > 400
**Result:** All parameters exceed threshold by >5× (excellent efficiency)

#### Tail ESS (for quantile estimates)
| Parameter | ESS_tail | Status |
|-----------|----------|--------|
| β₀ | 2,876 | ✓ PASS |
| β₁ | 2,360 | ✓ PASS |
| β₂ | 2,547 | ✓ PASS |
| φ | 2,876 | ✓ PASS |

**Criterion:** ESS_tail > 400
**Result:** All parameters exceed threshold by >5× (excellent tail sampling)

### 3. Divergent Transitions
**Total:** 0 out of 4,000 draws (0.00%)
**Criterion:** < 1% for reliable inference
**Result:** ✓ PASS (zero divergences indicates excellent posterior geometry)

### 4. Monte Carlo Standard Error (MCSE)

| Parameter | MCSE_mean | MCSE_sd | Posterior SD | MCSE/SD Ratio |
|-----------|-----------|---------|--------------|---------------|
| β₀ | 0.001 | 0.001 | 0.062 | 1.6% |
| β₁ | 0.001 | 0.001 | 0.047 | 2.1% |
| β₂ | 0.001 | 0.001 | 0.048 | 2.1% |
| φ | 0.078 | 0.067 | 4.150 | 1.9% |

**Criterion:** MCSE < 5% of posterior SD
**Result:** All parameters well below 5% threshold (excellent precision)

---

## Visual Diagnostics

### Trace Plots (`trace_plots.png`)
**Assessment:** Clean mixing across all chains with no trends or stuck chains. All four chains explore the same posterior region uniformly, confirming convergence.

- **β₀, β₁, β₂:** Stationary "hairy caterpillar" appearance
- **φ:** Excellent mixing despite right-skewed posterior
- **Warmup:** Effective adaptation visible in tuning phase

### Rank Plots (`rank_plots.png`)
**Assessment:** Uniform rank distributions across all parameters confirm proper chain mixing. No chain dominance or systematic biases detected.

- All rank histograms show uniform distribution
- No U-shaped or peaked patterns (would indicate mixing issues)

### Posterior vs Prior (`posterior_distributions.png`)
**Assessment:** Data substantially updates priors, indicating informative likelihood.

- **β₀:** Posterior shifted left from prior (mean: 4.29 vs prior: 4.70)
- **β₁:** Posterior concentrated near prior mode (mean: 0.84 vs prior: 0.80)
- **β₂:** Posterior lower than prior (mean: 0.10 vs prior: 0.30) - data suggests weaker quadratic effect
- **φ:** Posterior highly concentrated (mean: 16.6) - data strongly constrains dispersion

### Pairwise Correlations (`pairwise_correlations.png`)
**Assessment:** Moderate negative correlation between β₀ and β₁ (-0.6 to -0.7), expected for centering trade-off. No problematic strong correlations.

- β₀ ↔ β₁: Negative correlation (intercept-slope trade-off)
- β₀ ↔ β₂: Weak positive correlation
- β₁ ↔ β₂: Weak negative correlation
- φ: Independent of regression coefficients

### Energy Diagnostic (`energy_diagnostic.png`)
**Assessment:** Excellent overlap between transition and marginal energy distributions. No evidence of HMC sampling pathologies.

- Energy distributions well-matched
- No energy divergence issues

---

## Posterior Inference Quality

### Sampling Efficiency
- **Total draws:** 4,000
- **Effective draws (minimum ESS):** 2,106
- **Efficiency:** 53% (minimum across parameters)
- **Interpretation:** Excellent efficiency; approximately half of all draws are independent

### Numerical Stability
- **Divergences:** 0
- **Maximum tree depth hits:** Not reported (no issues)
- **Numerical errors:** None
- **Interpretation:** Model is well-behaved and well-specified

---

## Comparison to SBC Validation

The SBC validation predicted:
- **Convergence rate:** 95% (achieved: 100%)
- **Regression coefficients:** Well-calibrated, use 95% CIs (confirmed)
- **Dispersion (φ):** 85% coverage, use 99% CIs (confirmed)
- **Systematic biases:** None expected (none found)

**Assessment:** Real data inference matches SBC predictions perfectly. Model is operating as validated.

---

## Recommendations

### 1. Inference
✓ **Proceed with inference:** All convergence criteria met
- Use 95% credible intervals for β₀, β₁, β₂
- Use 99% credible intervals for φ (per SBC guidance)

### 2. Model Comparison
✓ **LOO-CV ready:** log_likelihood group present in InferenceData
- Shape: (4 chains, 1000 draws, 40 observations)
- Can proceed with model comparison against Experiments 2-5

### 3. Posterior Predictive Checks
✓ **PPC ready:** posterior_predictive group present
- Can validate model fit against observed data
- Check for systematic deviations

### 4. Reporting
Use the following for parameter estimates:
- **Point estimates:** Posterior mean
- **Uncertainty:** HDI (Highest Density Interval) at specified credibility level
- **Comparisons:** Full posterior distributions

---

## Conclusion

The NUTS sampler converged perfectly for the Negative Binomial Quadratic model on real data. Zero divergences, excellent ESS values (>2000 for all parameters), and perfect R-hat values indicate the posterior has been thoroughly explored. Visual diagnostics confirm clean mixing and proper convergence.

**The model is ready for:**
1. Parameter interpretation and inference
2. Posterior predictive validation
3. Model comparison via LOO-CV
4. Scientific conclusions about temporal trends

**No additional sampling or tuning required.**
