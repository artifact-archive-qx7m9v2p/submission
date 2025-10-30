# Posterior Predictive Check Findings
## Experiment 1: Negative Binomial Quadratic Model

**Date:** 2025-10-29
**Model:** NegBinomial Quadratic (β₀ + β₁·year + β₂·year²)
**Data:** 40 observations, count range [19, 272]
**Posterior Samples:** 4,000 replications (4 chains × 1,000 draws)

---

## Executive Summary

**OVERALL FIT QUALITY: POOR**

**CRITICAL FINDING: PHASE 2 TRIGGERED**
- Residual ACF(1) = 0.686 substantially exceeds threshold of 0.5
- Strong temporal autocorrelation indicates the model fails to capture temporal dependencies
- **Recommendation: Proceed immediately to Phase 2 (Temporal Models)**

Despite perfect convergence diagnostics (R̂=1.000), the model exhibits fundamental deficiencies in capturing data structure:
1. **Strong residual autocorrelation** (ACF(1) = 0.686) reveals unmodeled temporal dependencies
2. **Seven test statistics** show extreme Bayesian p-values (< 0.05 or > 0.95)
3. **Excessive coverage** (100% in 95% interval) suggests overestimation of uncertainty
4. **Systematic residual patterns** visible in time series plots

The model captures the overall trend but misses critical temporal structure. Temporal models (AR, ARMA, state-space) are essential for adequate fit.

---

## Plots Generated

### Comprehensive Diagnostics
1. **ppc_dashboard.png** - 12-panel comprehensive overview covering:
   - Observed vs predicted scatter
   - Coverage plot with prediction intervals
   - Trajectory spaghetti plot
   - Distribution comparison
   - Residual diagnostics (vs fitted, vs time)
   - Residual ACF (CRITICAL: shows ACF(1) = 0.686)
   - Q-Q plot
   - Test statistic comparisons (mean, variance, ACF(1), maximum)

2. **coverage_detailed.png** - Detailed coverage assessment with 50%, 80%, 95% intervals

3. **residual_diagnostics.png** - 6-panel residual suite:
   - Residuals vs fitted (nonlinear pattern)
   - Residuals vs time (strong temporal pattern)
   - Residual ACF (shows high autocorrelation)
   - Residual distribution
   - Q-Q plot
   - Scale-location plot

4. **test_statistics.png** - 6 key test statistics vs posterior predictive distributions:
   - Mean, Variance, Max (location/spread)
   - ACF(1), Skewness, Kurtosis (structure/shape)

5. **arviz_ppc.png** - ArviZ standard posterior predictive check

6. **loo_pit.png** - LOO-PIT calibration assessment

---

## Visual Diagnosis Summary

| Aspect Tested | Plot File | Finding | Implication |
|---------------|-----------|---------|-------------|
| Overall prediction accuracy | ppc_dashboard.png (Panel A) | R² = 0.883, good linear fit | Captures mean trend well |
| Uncertainty calibration | coverage_detailed.png | 100% coverage (excessive) | Overestimates uncertainty |
| Temporal patterns | ppc_dashboard.png (Panel C) | Observed deviates systematically from replications | Model misses temporal structure |
| Residual independence | residual_diagnostics.png (Panel C) | ACF(1) = 0.686, far above threshold | **Strong temporal dependence** |
| Residual homoscedasticity | residual_diagnostics.png (Panel A) | Nonlinear U-shaped pattern | Variance not fully modeled |
| Time series structure | residual_diagnostics.png (Panel B) | Clear temporal wave pattern | Systematic temporal misfit |
| Extreme value behavior | test_statistics.png (Max panel) | Observed at 99.4th percentile | Cannot reproduce observed maximum |
| Distribution shape | test_statistics.png (Skewness/Kurtosis) | p-values 0.999, 1.000 | Distribution shape mismatch |
| Autocorrelation | test_statistics.png (ACF(1) panel) | Observed ACF = 0.944 far exceeds replications | **Most severe discrepancy** |

---

## Coverage Analysis

### Empirical Coverage Rates

| Prediction Interval | Expected | Observed | Assessment |
|---------------------|----------|----------|------------|
| 50% PI | 50% | 67.5% (27/40) | OVER-COVERAGE |
| 80% PI | 80% | 95.0% (38/40) | OVER-COVERAGE |
| 95% PI | 95% | 100.0% (40/40) | **EXCESSIVE** |

**Coverage Quality: ACCEPTABLE** (by threshold criteria, but excessive)

### Coverage Findings

**All observations fall within 95% prediction intervals** (visible in `coverage_detailed.png`). While this might seem positive, it indicates the model is being too conservative with uncertainty estimates. Ideal coverage is 90-98%, not 100%.

**Interpretation:**
- The Negative Binomial model's dispersion parameter (φ = 16.6 ± 4.2) may be overestimating variance
- Prediction intervals are wider than necessary
- This could mask poor mean predictions by having overly wide intervals

**Evidence:** Coverage plot in `ppc_dashboard.png` (Panel B) shows observed points comfortably within all intervals, with no excursions outside 95% bands.

---

## Residual Diagnostics

### Residual Summary Statistics

- **Mean:** -4.21 (slight negative bias)
- **Std:** 34.44
- **Range:** [-143.29, 40.94]
- **Distribution:** Approximately normal (visible in Q-Q plot, Panel H of dashboard)

### Residual Autocorrelation (CRITICAL)

**Residual ACF values** (evident in `residual_diagnostics.png` Panel C):
- **ACF(1) = 0.686** ← **EXCEEDS PHASE 2 THRESHOLD (0.5)**
- ACF(2) = 0.423
- ACF(3) = 0.243

**Decision: TRIGGERS PHASE 2 (Temporal Models)**

The residual ACF(1) of 0.686 is far above the decision threshold of 0.5, indicating strong temporal dependencies that the quadratic model cannot capture. This is the most critical finding of the PPC.

**Visual Evidence:**
1. **Residual ACF plot** (`residual_diagnostics.png` Panel C) shows first lag well above Phase 2 threshold (orange line at 0.5)
2. **Residuals vs Time** (`residual_diagnostics.png` Panel B) exhibits clear sinusoidal pattern with smooth trend line showing systematic waves
3. **Test Statistic ACF(1)** (`test_statistics.png` bottom left) shows observed ACF(1) = 0.944 in far right tail (100th percentile) of posterior predictive distribution

### Residual Patterns

**Pattern vs Fitted Values** (`residual_diagnostics.png` Panel A):
- U-shaped nonlinear pattern evident from smooth trend line
- Negative residuals in middle range, positive at extremes
- Suggests quadratic form may not fully capture curvature

**Pattern vs Time** (`residual_diagnostics.png` Panel B):
- **Strong systematic wave pattern** with smooth temporal structure
- Largest negative residuals at end of series (time index 35-40)
- Pattern inconsistent with independent errors
- Green smooth trend shows clear oscillation

**Implication:** The model treats observations as independent conditional on time, but substantial temporal correlation remains in residuals.

---

## Test Statistics and Bayesian P-Values

### Overview

Computed 12 test statistics comparing observed vs 4,000 replicated datasets. Bayesian p-value = P(T_rep ≥ T_obs), where extreme values (< 0.05 or > 0.95) indicate poor fit.

### Problematic Statistics (p < 0.05 or p > 0.95)

| Statistic | Observed | P-value | Percentile | Assessment | Plot Evidence |
|-----------|----------|---------|------------|------------|---------------|
| **ACF(1)** | **0.944** | **0.000*** | **100.0th** | **SEVERE** | `test_statistics.png` - far right of distribution |
| **Kurtosis** | -1.23 | 1.000*** | 0.0th | POOR | `test_statistics.png` - far left of distribution |
| **Skewness** | 0.60 | 0.999*** | 0.1th | POOR | `test_statistics.png` - left tail |
| **Range** | 253 | 0.995*** | 99.5th | POOR | Dashboard Panel L proxy |
| **Max** | 272 | 0.994*** | 99.4th | POOR | `test_statistics.png` - right tail |
| **IQR** | 160.75 | 0.017*** | 1.7th | POOR | Not plotted individually |
| **Q75** | 195.50 | 0.020*** | 2.0th | POOR | Not plotted individually |

### Well-Fitting Statistics (0.1 < p < 0.9)

| Statistic | Observed | P-value | Assessment |
|-----------|----------|---------|------------|
| Mean | 109.45 | 0.668 | GOOD |
| Variance | 7441.74 | 0.910 | GOOD |
| Std | 86.27 | 0.910 | GOOD |
| Min | 19.00 | 0.244 | GOOD |
| Q25 | 34.75 | 0.776 | GOOD |
| Q50 | 74.50 | 0.371 | GOOD |

### Interpretation

**The model captures central tendency and overall variation well** (mean, variance p-values in healthy range), visible in `test_statistics.png` top row showing observed values near center of distributions.

**However, the model fails on:**

1. **Temporal structure** (ACF(1) p = 0.000): The observed autocorrelation of 0.944 is far higher than any replicated dataset can produce. This is the most severe discrepancy. The posterior predictive distribution of ACF(1) is centered around 0.75, and the observed value of 0.944 is in the extreme right tail (`test_statistics.png` bottom left panel).

2. **Distribution shape** (skewness p = 0.999, kurtosis p = 1.000): The observed data has positive skewness (0.60) and negative kurtosis (-1.23, flatter than normal). Replicated datasets are more symmetric and have heavier tails. Visible in `test_statistics.png` bottom middle and right panels.

3. **Extreme values** (max p = 0.994, range p = 0.995): The observed maximum of 272 is higher than 99.4% of replicated maximums. The model struggles to generate values this extreme, even though the Negative Binomial has heavy tails. Evidence in `test_statistics.png` top right panel.

4. **Upper quantile behavior** (Q75 p = 0.020, IQR p = 0.017): The 75th percentile is lower than expected, and interquartile range is narrower, suggesting the observed data has more spread at the ends but less in the middle quartiles.

**Pattern:** Model gets the center right but misses structure in dependencies and tail behavior.

---

## Observed vs Predicted Comparison

### Point Predictions

**R² = 0.883** (visible in `ppc_dashboard.png` Panel A) indicates strong correlation between observed and posterior mean predictions. The scatter plot shows:
- Good linear relationship
- Some deviation for highest values
- Slight systematic pattern (observations trending above predictions for mid-range)

### Trajectory Analysis

**Spaghetti plot** (`ppc_dashboard.png` Panel C) shows 100 replicated trajectories (blue) vs observed (red):
- Replicated trajectories have similar overall trend
- **Observed trajectory systematically deviates** - shows smoother acceleration
- Replications more variable point-to-point
- Observed shows more persistent runs above/below trend

**Key Insight:** Individual replications look choppy while observed data looks smooth and persistent. This is exactly what autocorrelation causes - consecutive observations are similar, creating smoother patterns than independent data.

### Distribution Comparison

**Distribution overlay** (`ppc_dashboard.png` Panel D):
- Overall shapes similar
- Observed histogram (red) aligns reasonably with replicated (blue overlays)
- Minor differences in peak height
- Both show right skew

**Conclusion:** Marginal distribution is well-captured, but temporal dependencies are not.

---

## Specific Model Deficiencies

### 1. Temporal Independence Assumption Violated

**Evidence:**
- Residual ACF(1) = 0.686 (`residual_diagnostics.png` Panel C)
- Observed data ACF(1) = 0.944 vs replicated ~0.75 (`test_statistics.png`)
- Clear wave pattern in residuals vs time (`residual_diagnostics.png` Panel B)
- Smooth observed trajectory vs choppy replications (`ppc_dashboard.png` Panel C)

**Implication:** The model assumes counts at time t are independent of counts at time t-1, conditional on the time covariate. This assumption is strongly violated. Observations are highly correlated with their neighbors.

**Why this matters:**
- Underestimates uncertainty in predictions
- Invalid standard errors and credible intervals for trends
- Cannot make accurate short-term forecasts
- Misses important dynamic structure

### 2. Curvature Misspecification

**Evidence:**
- U-shaped residual pattern vs fitted (`residual_diagnostics.png` Panel A)
- Systematic under-prediction in middle range, over-prediction at start
- Largest negative residuals at high end (time indices 35-40)

**Implication:** The quadratic polynomial may not perfectly capture the acceleration in the time series. The true curve may be more complex (e.g., exponential, logistic, piecewise).

**Why this matters:**
- Extrapolation outside observed range unreliable
- Predictions at late time points biased low
- May need more flexible functional form

### 3. Extreme Value Under-Generation

**Evidence:**
- Observed maximum (272) at 99.4th percentile of replicated maximums (`test_statistics.png`)
- Observed range (253) at 99.5th percentile
- Model rarely generates values as extreme as observed

**Implication:** Even with Negative Binomial's heavy-tailed distribution, the model cannot reproduce the observed extremes. This could be due to:
- Underestimated dispersion at high means
- Missing temporal clustering (extreme values occur together)
- Incorrect mean function (under-predicts at late times)

**Why this matters:**
- Risk assessment for high-count events unreliable
- Underestimates probability of extreme observations
- Prediction intervals may not protect against surprises

### 4. Distribution Shape Mismatch

**Evidence:**
- Observed skewness (0.60) at 0.1th percentile - data less skewed than model predicts
- Observed kurtosis (-1.23) at 0.0th percentile - data flatter than model predicts
- Q-Q plot shows good overall fit but some deviation in tails

**Implication:** The Negative Binomial distribution family may not perfectly match the marginal distribution shape, though this is a minor concern compared to temporal issues.

---

## Overall Model Adequacy Assessment

### Fit Quality by Criterion

| Criterion | Quality | Value | Target | Finding |
|-----------|---------|-------|--------|---------|
| Coverage (95%) | ACCEPTABLE | 100.0% | 85-100% | Excessive but in range |
| Residual ACF(1) | **POOR** | **0.686** | **< 0.5** | **Far exceeds threshold** |
| P-values | POOR | 7 extreme | ≤ 2 | Too many discrepancies |

### Strengths

1. **Perfect convergence**: R̂ = 1.000, ESS > 2,100, no divergences
2. **Central tendency**: Mean well-captured (p = 0.668)
3. **Overall variation**: Variance well-captured (p = 0.910)
4. **Marginal distribution**: Reasonable shape match
5. **General trend**: Strong R² = 0.883
6. **Computational**: Fast, stable, interpretable

### Critical Weaknesses

1. **Temporal dependence**: Residual ACF(1) = 0.686 indicates strong unmodeled correlation
2. **Independence assumption**: Clearly violated, observations not independent given time
3. **Structural patterns**: Systematic waves in residuals reveal missing dynamics
4. **Extreme values**: Cannot generate observed maximum
5. **Distribution shape**: Skewness and kurtosis mismatches

### Decision Matrix Position

```
                    Residual ACF(1)
                < 0.3    0.3-0.5    > 0.5
Coverage  90-98%  GOOD     ACCEPT   POOR
          85-90%  ACCEPT   ACCEPT   POOR
          < 85%   POOR     POOR     POOR
          > 98%   ACCEPT   ACCEPT   POOR ← We are here (100%, 0.686)
```

**Position:** POOR FIT (100% coverage, ACF(1) = 0.686)

---

## Scientific Interpretation

### What the Model Gets Right

The Negative Binomial Quadratic model successfully captures:
- **Long-term acceleration**: The positive quadratic term (β₂ = 0.10) correctly identifies that growth is accelerating
- **Overdispersion**: The estimated dispersion (φ = 16.6) accounts for variance exceeding Poisson
- **Overall magnitude**: Mean predictions track observed means closely
- **Parametric efficiency**: Only 4 parameters (β₀, β₁, β₂, φ) describe 40 observations

### What the Model Misses

The model fails to capture:
- **Short-term persistence**: High counts followed by high counts, low by low
- **Smooth trajectories**: Observed data shows gradual changes, not independent jumps
- **Temporal clustering**: Extreme values occur in connected sequences
- **Dynamic structure**: Process has memory, not just time-varying mean

### Mechanistic Implications

**The data exhibits temporal autocorrelation of 0.944**, meaning ~89% of variance in count at time t is predictable from count at time t-1. This suggests:

1. **Momentum/inertia**: Whatever drives the counts has persistence
2. **Slow-changing factors**: Underlying drivers change gradually, not instantaneously
3. **Contagion/diffusion**: Counts may spread or propagate over time
4. **Measurement smoothing**: Counts may aggregate over overlapping windows

**From Phase 2 perspective**, this strongly suggests mechanisms like:
- Auto-regressive processes (each observation depends on previous)
- State-space models (underlying latent state evolves smoothly)
- Growth processes with stochastic innovations
- Epidemic-like dynamics with transmission

---

## Recommendations for Phase 2

### Primary Recommendation: Temporal Models Required

**Residual ACF(1) = 0.686 decisively triggers Phase 2.** Proceed immediately to temporal modeling.

### Suggested Temporal Model Classes

1. **AR(1) or AR(p) models**
   - Condition on previous observations directly
   - May capture lag-1 correlation of 0.686
   - Check if higher-order lags needed (ACF(2) = 0.423 also substantial)

2. **ARMA models**
   - Add moving average component
   - Can capture more complex autocorrelation structures
   - Good if ACF decays gradually

3. **Random Walk with trend**
   - Allow slow-varying latent state
   - Add drift term for trend
   - Natural for cumulative processes

4. **State-space models**
   - Separate latent process from observations
   - Allow smooth evolution
   - Can incorporate quadratic trend in state equation

5. **Dynamic Linear Models**
   - Time-varying regression coefficients
   - Can make slope change over time
   - Flexible for non-stationary series

### Specific Modeling Suggestions

**Build on current model's strengths:**
- Keep Negative Binomial observation distribution (overdispersion well-handled)
- Retain time/time² predictors as baseline trend
- Add temporal correlation structure on top

**Example model structure:**
```
μ[t] = β₀ + β₁·time[t] + β₂·time²[t] + α[t]
α[t] ~ Normal(ρ·α[t-1], σ²)  # AR(1) random effect
y[t] ~ NegBinomial(μ[t], φ)
```

This combines:
- Quadratic trend (already validated)
- AR(1) random effect (captures temporal correlation)
- Negative Binomial overdispersion (keeps current strength)

**Key parameters to estimate:**
- ρ: Autocorrelation coefficient (expect ~0.7 based on residual ACF)
- σ²: Innovation variance in AR process
- Retain φ for dispersion

### Validation Strategy for Phase 2

When fitting temporal models:
1. Check residual ACF → should be < 0.3 for all lags
2. Re-run PPC → coverage should be 90-98% (not 100%)
3. Test statistics → ACF(1) p-value should be 0.1-0.9
4. Cross-validation → one-step-ahead predictions
5. Compare DIC/WAIC/LOO → should substantially improve

### What Not to Do

**Don't:**
- Add more polynomial terms (won't fix temporal correlation)
- Switch to different marginal distributions (NegBin is fine)
- Ignore temporal structure (it's the dominant issue)
- Over-fit with too many AR lags (start with AR(1))

**Do:**
- Focus on temporal correlation structure
- Consider mechanistic interpretation of ρ parameter
- Validate that temporal model resolves residual patterns
- Compare forecasting performance

---

## Technical Appendix

### Computation Details

- **Software**: PyMC 5.26.1, ArviZ 0.22.0
- **Sampling**: 4 chains, 1,000 draws each (4,000 total)
- **Posterior predictive**: Generated at sampling time
- **Test statistics**: Computed for all 4,000 replications
- **Runtime**: < 5 minutes total

### Diagnostic Checks Performed

1. **Coverage analysis**: 50%, 80%, 95% prediction intervals
2. **Residual autocorrelation**: Up to lag 20
3. **Test statistics**: 12 summary statistics with Bayesian p-values
4. **Visual checks**: 6 comprehensive plots
5. **Calibration**: LOO-PIT diagnostic

### Files Generated

**Code:**
- `code/posterior_predictive_checks.py` - Main analysis script
- `code/acf_util.py` - Custom ACF implementation
- `code/inspect_idata.py` - InferenceData inspection utility
- `code/ppc_results.npz` - Numerical results archive

**Plots:**
- `plots/ppc_dashboard.png` - 12-panel overview
- `plots/coverage_detailed.png` - Detailed coverage plot
- `plots/residual_diagnostics.png` - 6-panel residual suite
- `plots/test_statistics.png` - Test statistic comparisons
- `plots/arviz_ppc.png` - ArviZ standard PPC
- `plots/loo_pit.png` - Calibration check

**Documentation:**
- `ppc_findings.md` - This document

---

## Conclusion

The Negative Binomial Quadratic model provides a reasonable first approximation to the data's trend and dispersion but **fundamentally fails to capture temporal dependencies**. The residual ACF(1) of 0.686 decisively triggers Phase 2 temporal modeling.

**Key takeaway:** Perfect convergence diagnostics do not imply good model fit. Posterior predictive checks reveal that the independence assumption is strongly violated, with 69% of residual variance explained by lag-1 correlation.

**Next steps:** Proceed to Phase 2 (Temporal Models) to incorporate AR/ARMA/state-space structures that can capture the observed persistence and generate data that looks like what was actually observed.

---

**Analysis completed:** 2025-10-29
**Analyst:** Claude (Model Validation Specialist)
**Status:** ✓ Complete - Proceed to Phase 2
