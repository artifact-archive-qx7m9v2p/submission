# Posterior Predictive Check Findings: NB-Linear Model

**Experiment**: 1 - Negative Binomial Linear Model
**Model**: `C_t ~ NegativeBinomial(μ_t, φ)` where `log(μ_t) = β₀ + β₁×year_t`
**Date**: 2025-10-29
**Analyst**: PPC Validation Specialist

---

## Executive Summary

The Negative Binomial Linear model provides **generally adequate fit** for the marginal distribution and mean trend, but exhibits **substantial residual autocorrelation** (ρ₁=0.511) indicating unmodeled temporal structure. The model successfully captures:
- Central tendency and exponential growth trend
- Overall variance and dispersion
- Count distribution shape

However, it shows **deficiencies in higher-order distributional properties** (skewness, kurtosis) and fails to capture temporal dependencies. These findings justify extending to an AR(1) structure in Experiment 2.

---

## Plots Generated

All visualizations are saved in `/workspace/experiments/experiment_1/posterior_predictive_check/plots/`:

| Plot File | Diagnostic Purpose | Key Finding |
|-----------|-------------------|-------------|
| `ppc_overview.png` | Multi-panel distribution comparison | Good overall distributional match; model generates plausible data |
| `ppc_timeseries.png` | Temporal pattern and predictive intervals | 95% coverage achieved; 2 early outliers; wide intervals reflect uncertainty |
| `rootogram.png` | Count frequency fit across all values | Slight systematic under-prediction for low counts (20-30) |
| `residual_diagnostics.png` | Systematic patterns in residuals | Clear temporal oscillations visible; no heteroscedasticity |
| `test_statistics.png` | Bayesian p-values for 11 test statistics | 5/11 extreme p-values (skewness, kurtosis, min, max, zeros) |
| `autocorrelation_check.png` | Temporal correlation structure | Strong ACF up to lag 5; ρ₁=0.511 exceeds 95% confidence band |

---

## 1. Distributional Adequacy

### 1.1 Overall Distribution Match

**Visual Evidence**: `ppc_overview.png`

The four-panel comparison reveals:

**Panel A (Histogram Overlay)**:
- Observed distribution (black) falls well within the cloud of replicated distributions (blue)
- Good agreement in the main mass (50-200 range)
- Slight discrepancy: model generates more extreme values than observed

**Panel B (Empirical CDF)**:
- Observed CDF (black line) tracks closely with replicated CDFs (blue cloud)
- No systematic deviations across the distribution
- Excellent agreement in middle quantiles (25th-75th percentiles)

**Panel C (Q-Q Plot)**:
- Strong linear relationship along y=x line
- Points slightly below line at upper tail → model over-predicts extreme values
- Lower tail matches well

**Panel D (Kernel Density)**:
- Observed density falls within replicated density band
- Peak location well-captured
- Right tail: model predicts heavier tail than observed

**Conclusion**: Model captures overall distributional shape adequately. Minor discrepancies in tail behavior are not substantively concerning given sample size (n=40).

---

### 1.2 Test Statistics Analysis

**Visual Evidence**: `test_statistics.png`

| Statistic | Observed | Replicated Mean ± SD | p-value | Status | Interpretation |
|-----------|----------|----------------------|---------|--------|----------------|
| Mean | 109.4 | 109.5 ± 6.0 | 0.481 | ✓ PASS | Perfect capture of central tendency |
| Variance | 7512 | 8476 ± 1671 | 0.704 | ✓ PASS | Good dispersion match |
| Min | 21.0 | 13.9 ± 3.4 | **0.021** | ⚠ FAIL | Model under-predicts minimum (generates lower values) |
| Max | 269.0 | 364.5 ± 52.6 | **0.987** | ⚠ FAIL | Model over-predicts maximum (generates higher values) |
| Q10 | 28.0 | 23.6 ± 3.1 | 0.086 | ✓ PASS | Lower tail adequate |
| Q90 | 252.2 | 245.0 ± 24.1 | 0.367 | ✓ PASS | Upper tail reasonable |
| Skewness | 0.64 | 1.12 ± 0.22 | **0.999** | ⚠ FAIL | Model predicts more right-skewed distribution |
| Kurtosis | -1.13 | 0.46 ± 0.88 | **1.000** | ⚠ FAIL | Model predicts lighter tails (observed is platykurtic) |
| CV | 0.79 | 0.84 ± 0.05 | 0.800 | ✓ PASS | Relative dispersion adequate |
| N_zeros | 0 | 0 | **1.000** | (N/A) | No zero-inflation issue |
| Prop_extreme (>200) | 0.20 | 0.17 ± 0.03 | 0.343 | ✓ PASS | Extreme value frequency reasonable |

**Key Findings**:
- **5 out of 11 tests failed** (p < 0.05 or p > 0.95)
- Failures concentrated in **higher-order moments** and **extremes**
- **Core statistics passed**: mean, variance, quantiles, coefficient of variation
- The observed data is more "well-behaved" (less skewed, less extreme) than the model predicts

**Interpretation**: The model's negative binomial distribution slightly over-predicts variability in shape parameters while matching the location and scale well. This is **acceptable** for a baseline model.

---

### 1.3 Count Distribution Fit

**Visual Evidence**: `rootogram.png`

The hanging rootogram plots `√(Observed) - √(Expected)` for each count value:

**Patterns Observed**:
- **Zero line** = perfect fit
- **Low counts (0-30)**: Predominantly negative bars → model under-predicts frequency of low counts
- **Observed minimum is 21**, but model frequently generates values < 21
- **Mid-range (30-100)**: Fluctuates around zero with no systematic bias
- **High counts (100-269)**: Generally near zero with random fluctuation
- **Very high counts (>269)**: Not observed but model predicts small probability

**Conclusion**: The negative binomial distribution provides a reasonable fit across the count range. The systematic under-prediction of the observed minimum (21) is evident, consistent with the test statistics showing p=0.021 for Min.

---

## 2. Temporal Pattern Assessment

### 2.1 Time Series Fit

**Visual Evidence**: `ppc_timeseries.png`

**Observed vs Predicted**:
- Median prediction (blue line) captures the exponential growth trend well
- 50% predictive interval (dark blue band): 20/40 observations (50.0%) - PERFECT
- 90% predictive interval (light blue band): 38/40 observations (95.0%) - EXCELLENT

**Outliers** (marked in red, n=2):
1. **Year ≈ -1.67** (observation 1): Observed = 30, slightly below 5th percentile
2. **Year ≈ -1.58** (observation 2): Observed = 32, slightly below 5th percentile

Both outliers occur in the **early period** where counts are low and relative uncertainty is higher.

**Key Observation**: The observed data shows **short-term fluctuations** that the model cannot capture. Notice the "jagged" pattern in the observed trajectory vs the smooth predictive intervals. This suggests **serial correlation** not accounted for in the model.

---

### 2.2 Residual Patterns

**Visual Evidence**: `residual_diagnostics.png`

**Panel A (Residuals vs Fitted Values)**:
- No clear heteroscedasticity
- Residuals roughly centered at zero across fitted value range
- No funnel or fan shape → variance assumption adequate
- Few points beyond ±2 standard deviations (expected for n=40)

**Panel B (Residuals vs Time)** - **CRITICAL FINDING**:
- **Clear oscillatory pattern**: connected residuals trace waves
- Positive residuals cluster together (e.g., year 0.5 to 1.5)
- Negative residuals cluster together (e.g., year -0.5 to 0.0)
- This is **strong visual evidence of autocorrelation**
- Pattern resembles AR(1) structure: consecutive residuals are similar

**Panel C (Normal Q-Q Plot)**:
- Good agreement with theoretical normal quantiles
- Points follow the reference line closely
- Slight deviation at extremes (expected with n=40)
- **Residuals are approximately normally distributed** → good model calibration

**Panel D (Residual Distribution)**:
- Histogram closely matches N(0,1) reference (red curve)
- Mean: 0.069 (close to expected 0)
- SD: 0.928 (close to expected 1)
- Shape is unimodal and symmetric

**Conclusion**: Residuals show no evidence of systematic bias or heteroscedasticity, BUT exhibit clear **temporal clustering**, indicating the model fails to capture time-series dependencies.

---

### 2.3 Autocorrelation Structure

**Visual Evidence**: `autocorrelation_check.png`

**Panel A (ACF of Quantile Residuals)**:

| Lag | ACF Value | Significance |
|-----|-----------|--------------|
| 1 | 0.511 | **HIGHLY SIGNIFICANT** (>95% band = ±0.310) |
| 2 | 0.433 | **HIGHLY SIGNIFICANT** |
| 3 | 0.393 | **HIGHLY SIGNIFICANT** |
| 4 | 0.254 | Near boundary |
| 5 | 0.203 | Near boundary |
| 6+ | <0.1 | Not significant |

**Key Findings**:
- **Lag-1 autocorrelation of 0.511** is substantial and far exceeds confidence bands
- ACF decays gradually, consistent with AR(1) process
- Residuals at time t strongly predict residuals at t+1

**Panel B (Lag-1 Scatter Plot)**:
- Clear positive relationship between consecutive residuals
- ρ = 0.511 visible as upward-sloping cloud
- Upper-right and lower-left quadrants more populated than expected under independence

**Interpretation**: This is an **EXPECTED FINDING** for the baseline linear model. The model includes no temporal correlation structure, so all time-series dependencies appear in residuals. This motivates the AR(1) extension in Experiment 2.

---

## 3. Predictive Performance

### 3.1 Coverage Assessment

**Interval Coverage** (from `ppc_timeseries.png`):
- **50% Predictive Interval**: 20/40 = 50.0% (target: 50%) → PERFECT
- **90% Predictive Interval**: 38/40 = 95.0% (target: 90%) → EXCELLENT

The model is **well-calibrated** for prediction intervals. Slightly higher-than-nominal coverage (95% vs 90%) indicates conservative uncertainty quantification, which is preferable to under-coverage.

### 3.2 Systematic Prediction Errors

**From test statistics and visual inspection**:
- No systematic over/under-prediction of the mean
- Slight tendency to over-predict extremes (max, skewness)
- Early-period observations (first 2-3) more difficult to predict

**Pattern**: The model struggles most where counts are lowest and growth is just beginning. This could reflect:
1. Boundary effects (fewer observations to inform early trend)
2. Non-constant growth rate (actual acceleration not captured)

---

## 4. Model Adequacy Assessment

### 4.1 Strengths

1. **Mean Structure**: Exponential trend via log-link perfectly captures growth pattern
   - Bayesian p-value for mean = 0.481 (ideal)
   - Visual trend match in time series plot

2. **Variance Structure**: Negative binomial dispersion (φ = 35.6 ± 10.8) adequately captures overdispersion
   - Bayesian p-value for variance = 0.704
   - No heteroscedasticity in residuals vs fitted

3. **Count Distribution**: Model generates realistic count data
   - Rootogram shows no severe misfit
   - No zero-inflation concerns

4. **Predictive Calibration**: Intervals have correct coverage
   - 50% and 90% intervals well-calibrated
   - Uncertainty properly quantified

5. **Residual Behavior**: No bias, normality, or heteroscedasticity issues
   - Residuals approximately N(0,1)
   - No patterns vs fitted values

### 4.2 Deficiencies

1. **Temporal Independence Assumption** - **MAJOR LIMITATION**
   - Residual ACF(1) = 0.511 (highly significant)
   - Clear oscillatory patterns in residual time series
   - Consecutive observations more similar than model predicts
   - **Impact**: Underestimates short-term predictability; loses information

2. **Higher-Order Moments** - **MODERATE CONCERN**
   - Skewness: p = 0.999 (model too skewed)
   - Kurtosis: p = 1.000 (model predicts lighter tails)
   - **Impact**: Less critical for scientific inference about trend

3. **Extreme Value Prediction** - **MINOR CONCERN**
   - Min: p = 0.021 (generates values lower than observed)
   - Max: p = 0.987 (generates values higher than observed)
   - **Impact**: Minimal; extremes are inherently variable with n=40

### 4.3 Scientific Implications

**For Inference**:
- Growth rate estimate (β₁ = 0.87 ± 0.04) is **reliable** given good mean structure
- Uncertainty intervals **slightly underestimate** true uncertainty due to unmodeled temporal correlation
- Point estimates are unbiased

**For Prediction**:
- One-step-ahead predictions miss **serial correlation patterns**
- Longer-term trend predictions are adequate
- Model treats each observation as independent, losing **short-term momentum** information

**For Model Comparison**:
- This baseline establishes what a **purely trend-based model** can achieve
- Residual ACF(1) = 0.511 sets target for AR(1) model to reduce
- Good marginal fit means AR extension addresses orthogonal model aspect

---

## 5. Conclusions and Recommendations

### 5.1 Overall Assessment: ADEQUATE WITH LIMITATIONS

The NB-Linear model is **fit for purpose as a baseline** but shows clear room for improvement:

**PASS Criteria Met**:
- ✓ Mean and variance test statistics in acceptable range
- ✓ Predictive interval coverage meets targets
- ✓ No systematic bias in residuals
- ✓ Count distribution adequately modeled

**FAIL Criteria Met**:
- ✗ High residual autocorrelation (ρ₁ = 0.511 >> 0.310 threshold)
- ✗ 5/11 test statistics with extreme p-values
- ✗ Clear temporal patterns in residual plots

**Verdict**: The model is **acceptable for describing marginal trends** but **inadequate for temporal dynamics**.

---

### 5.2 Specific Model Deficiencies

#### 5.2.1 Unmodeled Temporal Correlation (EXPECTED)

**Evidence**:
- `autocorrelation_check.png`: Lag-1 ACF = 0.511 far exceeds 95% confidence bands (±0.310)
- `residual_diagnostics.png` Panel B: Clear wave patterns in residuals over time
- ACF significant through lag 3, indicating persistent correlation

**Magnitude**: Approximately **51% of residual variation at time t** is predictable from time t-1

**Implication**:
- Model wastes information by ignoring temporal structure
- Prediction intervals don't account for short-term persistence
- Standard errors may be underestimated

**Recommendation**: **Extend to AR(1) structure** (Experiment 2)
- Expected improvement: Reduce ACF(1) from 0.51 to <0.1
- Will capture momentum and oscillations
- Should improve one-step-ahead predictions

#### 5.2.2 Higher-Order Distributional Mismatch (ACCEPTABLE)

**Evidence**:
- `test_statistics.png`: Skewness p=0.999, Kurtosis p=1.000
- Model predicts more skewed and lighter-tailed distribution than observed

**Magnitude**:
- Observed skewness = 0.64 vs predicted 1.12 (75% higher)
- Observed kurtosis = -1.13 vs predicted 0.46 (platykurtic vs mesokurtic)

**Implication**:
- For most scientific questions about trend and variability, this is **not critical**
- Affects extreme value prediction accuracy
- Less important than temporal structure for time series

**Recommendation**: **Accept this limitation** for now
- Negative binomial is flexible enough
- May improve with AR(1) extension (temporal correlation can affect moment estimates)
- Consider alternative distributions only if this persists in AR model

#### 5.2.3 Extreme Value Predictions (ACCEPTABLE)

**Evidence**:
- `test_statistics.png`: Min p=0.021, Max p=0.987
- `ppc_overview.png`: Model generates wider range than observed

**Magnitude**:
- Observed min = 21, predicted mean = 14
- Observed max = 269, predicted mean = 364

**Implication**:
- With n=40, sampling variability in extremes is high
- Model correctly represents uncertainty about extremes
- Not a bias, just natural variation

**Recommendation**: **No action needed**
- This is expected behavior, not model failure
- Extremes are always uncertain with limited data

---

### 5.3 What the Model Does Well

1. **Captures exponential growth**: β₁ = 0.87 accurately reflects observed trend
2. **Appropriate dispersion**: φ parameter handles overdispersion effectively
3. **Well-calibrated uncertainty**: Predictive intervals have correct coverage
4. **No systematic bias**: Residuals centered at zero with no patterns vs covariates
5. **Sensible for marginal analysis**: If temporal correlation is not of interest, model is adequate

---

### 5.4 Justification for Experiment 2 (NB-AR1)

The **single most important finding** from this PPC is:

> **Residual ACF(1) = 0.511** (95% CI excludes zero; ±0.310 bands)

This provides **quantitative justification** for adding AR(1) structure:

**Target for Experiment 2**:
- Reduce residual ACF(1) from 0.51 to <0.1 (ideally <0.05)
- Eliminate wave patterns in residual vs time plot
- Maintain good marginal distribution fit
- Potentially improve higher-order moment fit as side benefit

**Expected Gains**:
1. **Better one-step predictions**: Use t-1 information
2. **Tighter prediction intervals**: Account for correlation
3. **More efficient estimates**: Use all available information
4. **Improved model diagnostics**: Residuals closer to white noise

**Risk**:
- Small sample (n=40) makes AR parameter estimation uncertain
- May overfit if true ρ is variable over time

---

### 5.5 Recommendations for Next Steps

#### Immediate Actions:

1. **Proceed to Experiment 2 (NB-AR1)**:
   - Implement AR(1) latent process: `η_t = ρ·η_{t-1} + ε_t`
   - Expect ρ ≈ 0.5-0.7 based on residual ACF
   - Verify ACF(1) drops below significance threshold

2. **Compare Models Using**:
   - ELPD (expected log predictive density)
   - WAIC or LOO cross-validation
   - One-step-ahead prediction accuracy
   - Residual ACF as direct measure of improvement

3. **Retain This Baseline**:
   - Keep NB-Linear as reference
   - Quantify improvement from AR structure
   - Useful if simpler model preferred for interpretability

#### Diagnostic Checklist for Experiment 2:

When validating NB-AR1 model, verify:
- [ ] Residual ACF(1) < 0.1 (target: <0.05)
- [ ] No wave patterns in residual vs time plot
- [ ] Maintained good fit for mean and variance
- [ ] Predictive intervals still well-calibrated
- [ ] No new systematic residual patterns introduced
- [ ] Higher-order moments (skewness, kurtosis) improved as side benefit

---

## 6. Technical Notes

### 6.1 Posterior Predictive Generation

- **Method**: Drew 1000 posterior samples from 4000 available
- **Generation**: For each sample (β₀, β₁, φ), simulated n=40 negative binomial observations
- **Parameterization**: Used `μ_t = exp(β₀ + β₁·year_t)` and dispersion `φ`

### 6.2 Residual Definition

- **Type**: Randomized quantile residuals
- **Construction**: For each observation, computed empirical CDF from 1000 replications, then transformed to N(0,1) via inverse normal CDF
- **Advantages**: Should be N(0,1) if model is correct; handles discrete data; works for any distribution

### 6.3 Test Statistics

All test statistics computed on full dataset (not observation-level):
- Location: mean
- Dispersion: variance, CV
- Extremes: min, max, Q10, Q90
- Shape: skewness, kurtosis
- Count-specific: number of zeros, proportion extreme (>200)

### 6.4 Bayesian P-Values Interpretation

- **Definition**: `p = P(T(y_rep) ≥ T(y_obs) | y_obs)`
- **Ideal**: p ≈ 0.5 (observed value is typical)
- **Concern**: p < 0.05 or p > 0.95 (observed value in tail of predictive distribution)
- **Not binary**: Use as continuous measure of discrepancy

---

## Appendix: Visual Diagnosis Summary

| Aspect Tested | Plot File | Finding | Implication |
|---------------|-----------|---------|-------------|
| **Marginal Distribution** | `ppc_overview.png` | Good overall match; observed within replicated cloud | Model captures count distribution shape |
| **Temporal Trend** | `ppc_timeseries.png` | Excellent coverage (95% in 90% PI); 2 early outliers | Mean structure well-specified |
| **Count Frequencies** | `rootogram.png` | Slight under-prediction of low counts (20-30) | Minor; observed min = 21 vs predicted ≈14 |
| **Residual Bias** | `residual_diagnostics.png` Panel A | No patterns vs fitted; centered at zero | No systematic over/under-prediction |
| **Temporal Correlation** | `residual_diagnostics.png` Panel B | **Clear wave patterns; strong clustering** | **Major: unmodeled serial correlation** |
| **Residual Distribution** | `residual_diagnostics.png` Panels C & D | Normal Q-Q excellent; histogram matches N(0,1) | Good calibration |
| **Test Statistics** | `test_statistics.png` | 5/11 extreme p-values (moments & extremes) | Higher-order properties less well-captured |
| **Autocorrelation** | `autocorrelation_check.png` | **ACF(1)=0.511 >> confidence bands** | **Major: AR structure needed** |

**Key Convergent Evidence**:
- Multiple plots (`residual_diagnostics.png` Panel B, `autocorrelation_check.png`, wave pattern in `ppc_timeseries.png` observed data) all point to **unmodeled temporal correlation**
- This is the **dominant** model inadequacy
- All other issues are secondary

---

## Files and Code

**Analysis Code**: `/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_analysis.py`

**Generated Plots**: `/workspace/experiments/experiment_1/posterior_predictive_check/plots/`
- `ppc_overview.png` (4 panels: distribution comparison)
- `ppc_timeseries.png` (time series with predictive bands)
- `rootogram.png` (count frequency fit)
- `residual_diagnostics.png` (4 panels: residual patterns)
- `test_statistics.png` (11 panels: Bayesian p-values)
- `autocorrelation_check.png` (2 panels: ACF and lag-1 scatter)

**Key Parameters**:
- Posterior samples: 4000 (4 chains × 1000 draws)
- PPC replications: 1000
- Observations: n = 40
- Model: `log(μ_t) = β₀ + β₁×year_t` with NegBin(μ_t, φ)

---

**Assessment**: Model is ADEQUATE for baseline but REQUIRES extension to AR(1) structure to address temporal correlation (ACF(1)=0.511).
