# Posterior Predictive Check Findings
## Log-Linear Negative Binomial Model (Experiment 1)

**Date**: 2025-10-29
**Model**: Log-Linear Negative Binomial with parameters β₀, β₁, φ
**Data**: n=40 observations, Count range [21, 269], Observed Var/Mean = 68.7

---

## Executive Summary

**Overall Assessment: FAIL**

The log-linear negative binomial model shows **significant deficiencies** in reproducing key features of the observed data. While the model achieves excellent prediction interval coverage (100%), it fails three of four falsification criteria:

1. **FAIL**: Variance-to-mean ratio recovery (95% CI extends beyond target range)
2. **PASS**: Prediction interval coverage (100% vs. 80% threshold)
3. **FAIL**: Late period performance degradation (4.2× worse than early period)
4. **FAIL**: Strong systematic curvature in residuals (inverted-U pattern)

The model's linear assumption is **systematically violated**: residuals show clear inverted-U curvature, indicating the true growth pattern accelerates beyond what exponential growth can capture. This manifests as progressively worsening fit in later time periods.

---

## Plots Generated

All diagnostic visualizations are saved in `/workspace/experiments/experiment_1/posterior_predictive_check/plots/`:

| Plot File | Purpose | Key Finding |
|-----------|---------|-------------|
| `timeseries_fit.png` | Time series overlay with 90% prediction intervals | All observations within 90% PI; visual trend match appears good |
| `residuals.png` | Four-panel residual diagnostics | Strong inverted-U curvature (quad coef = -5.2); increasing residual magnitude over time |
| `early_vs_late_fit.png` | Side-by-side comparison of early vs. late periods | Late period MAE (26.5) is 4.2× larger than early period MAE (6.3) |
| `var_mean_recovery.png` | Variance-to-mean ratio distribution | Predicted mean (84.5) overshoots observed (68.7); 95% CI [54.8, 130.9] extends above target [50, 90] |
| `calibration.png` | Coverage calibration curve | Model is over-conservative: all coverage levels exceed expected (100% at 90% level) |
| `distribution_overlay.png` | Overall distribution comparison | Posterior predictive distribution generally overlaps observed |
| `arviz_ppc.png` | ArviZ posterior predictive check | Confirms distribution overlap |

---

## Quantitative Results

### 1. Variance-to-Mean Recovery [FAIL]

**Criterion**: 95% of posterior predictive Var/Mean ratios should fall within [50, 90]

| Metric | Value |
|--------|-------|
| Observed Var/Mean | 68.7 |
| Predicted Mean | 84.5 ± 20.1 |
| 95% Credible Interval | [54.8, 130.9] |
| Within target [50, 90] | 66.8% |

**Finding**: The model **systematically overestimates** dispersion. The 95% credible interval extends to 131, far exceeding the target upper bound of 90. Only 67% of posterior predictive samples have Var/Mean in the acceptable range, falling well short of the 95% threshold.

**Visual Evidence**: `var_mean_recovery.png` shows the posterior predictive distribution (blue) has substantial mass above the target range (green shading). The observed value (red dashed line at 68.7) falls within the predictive distribution but the distribution is too wide and right-skewed.

---

### 2. Prediction Interval Coverage [PASS]

**Criterion**: >80% of observations should fall within 90% prediction intervals

| Metric | Value |
|--------|-------|
| 90% PI Coverage | 100.0% (40/40 observations) |
| Expected | 90% |
| Threshold | 80% |

**Finding**: The model achieves **perfect coverage** - all 40 observations fall within their 90% prediction intervals. This exceeds the 80% threshold and even surpasses the nominal 90% level.

**Visual Evidence**: `timeseries_fit.png` shows all red observed points falling comfortably within the pink 90% prediction interval band. No observations are circled as outliers.

**Caveat**: Perfect coverage suggests the model may be **over-conservative** - the prediction intervals may be too wide, providing less informative predictions than necessary. The calibration plot confirms this: observed coverage exceeds expected coverage across all levels.

---

### 3. Early vs. Late Period Performance [FAIL]

**Criterion**: Late period MAE should be <2× early period MAE

| Period | MAE | Observations |
|--------|-----|--------------|
| Early (first 10 obs) | 6.34 | Years -1.67 to -0.90 |
| Late (last 10 obs) | 26.49 | Years 0.98 to 1.67 |
| Ratio (Late/Early) | **4.17** | **FAIL** (>2.0) |

**Finding**: The model's predictive accuracy **deteriorates dramatically** in the late period. Errors in the final 10 observations are more than 4 times larger than in the first 10 observations, indicating the linear trend assumption breaks down as time progresses.

**Visual Evidence**: `early_vs_late_fit.png` reveals the problem clearly:
- **Early period**: Observed values (red dots) track the predicted mean (blue line) closely, with MAE=6.34
- **Late period**: Observed values systematically deviate from predictions, with much larger scatter and MAE=26.49

This pattern suggests the true growth rate **accelerates** beyond what the log-linear model predicts - the model underestimates late-period counts because it assumes constant exponential growth.

---

### 4. Residual Curvature Test [FAIL]

**Criterion**: No strong curvature (|quadratic coefficient| < 1.0)

| Metric | Value |
|--------|-------|
| Quadratic Coefficient | **-5.22** |
| Pattern | Inverted-U (negative curvature) |
| Threshold | \|coef\| < 1.0 |

**Finding**: Residuals exhibit **strong systematic curvature** with quadratic coefficient = -5.22, far exceeding the threshold of 1.0. The inverted-U pattern indicates:
- Early period: Model slightly overpredicts (positive residuals)
- Middle period: Good fit (residuals near zero)
- Late period: Model significantly underpredicts (negative residuals)

**Visual Evidence**: The top-left panel of `residuals.png` shows residuals vs. time with a clear inverted-U curve (blue line). The quadratic fit captures this systematic pattern, which should not exist if the model were correctly specified.

**Interpretation**: This curvature pattern is the **diagnostic signature** of a model that assumes linear growth in log-space (constant exponential growth rate) when the true data exhibit accelerating growth (increasing exponential growth rate over time).

---

## Additional Diagnostic Findings

### Residual Characteristics

| Statistic | Value |
|-----------|-------|
| Mean Residual | -0.16 (near zero, good) |
| Median Residual | -1.31 |
| RMSE | 21.81 |
| MAE | 14.53 |

The mean residual near zero indicates no overall bias, but the median being negative suggests a slight tendency toward underprediction. The RMSE and MAE values are substantial relative to the mean count.

### Residual Diagnostics (from `residuals.png`)

1. **Residuals vs. Time** (top-left): Clear inverted-U curvature
2. **Residuals vs. Fitted** (top-right): Increasing spread with fitted values, suggesting heteroscedasticity
3. **Q-Q Plot** (bottom-left): Heavy tails on both ends, with several extreme outliers
4. **Residual Distribution** (bottom-right): Approximately symmetric around zero, but with wider spread than expected

---

## Visual Diagnosis Summary

| Aspect Tested | Plot File | Finding | Implication |
|---------------|-----------|---------|-------------|
| Overall time series fit | `timeseries_fit.png` | All obs in 90% PI, but intervals very wide | Over-conservative predictions |
| Systematic bias over time | `residuals.png` (top-left) | Inverted-U curvature (coef=-5.2) | Linear growth assumption violated |
| Heteroscedasticity | `residuals.png` (top-right) | Spread increases with fitted values | Variance structure may be misspecified |
| Tail behavior | `residuals.png` (bottom-left) | Heavy tails in Q-Q plot | Extreme values not well captured |
| Temporal performance | `early_vs_late_fit.png` | Late period MAE 4.2× early period | Degrading fit over time |
| Dispersion recovery | `var_mean_recovery.png` | Predicted (84.5) > observed (68.7) | Overdispersion overestimated |
| Interval calibration | `calibration.png` | All coverage exceeds expected | Over-conservative intervals |

---

## Model Deficiencies Identified

### 1. Linear Growth Assumption is Violated

**Evidence**:
- Inverted-U residual curvature (`residuals.png`, quad coef = -5.22)
- Progressive underprediction in late period (`early_vs_late_fit.png`)
- Late/early MAE ratio of 4.17

**Mechanism**: The model assumes `log(μ) = β₀ + β₁ × year`, implying constant exponential growth rate β₁. However, the residual pattern indicates the true growth rate **increases over time** - the data are "curving upward" faster than exponential.

**Consequence**: The model systematically underpredicts counts in the late period, missing the accelerating trend.

### 2. Overdispersion is Overestimated

**Evidence**:
- Posterior predictive Var/Mean = 84.5 ± 20.1 vs. observed 68.7
- 95% CI [54.8, 130.9] extends well above target range [50, 90]
- Only 67% of samples within acceptable range

**Mechanism**: The negative binomial dispersion parameter φ captures extra-Poisson variation, but the model attributes too much variation to random overdispersion when some is actually systematic (due to model misspecification).

**Consequence**: Prediction intervals are too wide, reducing the model's practical utility for precise forecasting.

### 3. Prediction Intervals are Over-Conservative

**Evidence**:
- 100% coverage at 90% nominal level
- Calibration curve consistently above perfect calibration line (`calibration.png`)

**Mechanism**: The combination of overestimated dispersion and uncertainty in parameters leads to intervals wider than necessary.

**Consequence**: While this ensures good coverage, it reduces the informativeness of predictions. A better-specified model could provide tighter, more useful intervals.

---

## Substantive Importance

### Are These Deficiencies Practically Important?

**YES** - for the following reasons:

1. **Forecasting Applications**: If this model were used to predict future counts (beyond year = 1.67), it would systematically underestimate growth. The 4× degradation in late-period accuracy suggests predictions further into the future would be even less reliable.

2. **Scientific Understanding**: The inverted-U residual pattern reveals that the true data-generating process is **not** simple exponential growth. Missing this accelerating pattern means missing a key feature of the phenomenon being studied.

3. **Resource Planning**: For applications requiring count predictions (e.g., capacity planning, resource allocation), the 26.5 average error in late periods represents substantial uncertainty that could lead to under-preparation.

4. **Model Selection**: These deficiencies provide clear guidance for model improvement - a quadratic term in the log-linear predictor would likely address the curvature issue.

### Can the Model Still Be Useful?

**Limited utility**:
- For **interpolation** within the observed range, the model provides reasonable point estimates (mean residual ≈ 0)
- For **uncertainty quantification**, the wide intervals do provide conservative coverage
- For **early-period prediction**, performance is acceptable (MAE = 6.34)

**Not recommended for**:
- **Extrapolation** beyond observed data
- **Late-period or future prediction** where systematic bias is large
- **Precise forecasting** where tight intervals are needed
- **Scientific inference** about growth mechanisms

---

## Recommended Model Improvements

Based on the identified deficiencies, the following modifications are recommended:

### 1. Add Quadratic Term (High Priority)

**Modification**: `log(μ) = β₀ + β₁ × year + β₂ × year²`

**Rationale**: Directly addresses the inverted-U curvature in residuals. A negative β₂ would allow the log-mean to curve upward, capturing the accelerating growth pattern.

**Expected Impact**:
- Reduce late-period errors
- Improve Late/Early MAE ratio from 4.17 to <2.0
- Eliminate systematic curvature in residuals

### 2. Consider Alternative Growth Functions (Medium Priority)

**Options**:
- Generalized logistic growth (if data approach a carrying capacity)
- Piecewise linear with change-point (if growth regime shifts)
- Time-varying growth rate: `β₁(t)` as a function

**Rationale**: Exponential growth with changing rate may be more appropriate than polynomial modification.

### 3. Investigate Heteroscedasticity (Low Priority)

**Observation**: Residual spread increases with fitted values (`residuals.png`, top-right)

**Potential fix**: Time-varying or mean-dependent dispersion parameter φ(t) or φ(μ)

**Note**: This is lower priority because the negative binomial already allows mean-variance relationship; further refinement may be unnecessary.

---

## Falsification Criteria: Final Verdict

| Criterion | Result | Evidence |
|-----------|--------|----------|
| Var/Mean in [50, 90] | **FAIL** | 95% CI extends to 131; only 67% of samples in range |
| Coverage > 80% | **PASS** | 100% coverage (over-conservative) |
| Late/Early MAE < 2 | **FAIL** | Ratio = 4.17 |
| No strong curvature | **FAIL** | Quadratic coefficient = -5.22 |

**Result**: 1 of 4 criteria passed → **OVERALL FAIL**

Per the falsification framework defined in `experiments/experiment_1/metadata.md`:
- Criterion 2 (systematic curvature): Clearly violated with inverted-U residual pattern
- Criterion 3 (late period failure): MAE ratio 4.17 >> 2.0 threshold
- Criterion 4 (variance mismatch): 95% CI extends outside [50, 90] range

The model must be **REJECTED** as inadequate for capturing the key features of the observed data.

---

## Conclusions

1. **The log-linear negative binomial model systematically misspecifies the growth pattern**, assuming constant exponential growth when the data exhibit accelerating growth.

2. **Late-period predictive accuracy degrades by 4×**, making the model unreliable for forecasting or extrapolation.

3. **Residual curvature (coefficient = -5.22) provides clear evidence** that a quadratic term or alternative growth function is needed.

4. **The model overestimates dispersion**, leading to over-conservative prediction intervals that are less informative than necessary.

5. **While the model achieves good coverage**, this is achieved by being overly uncertain rather than being well-calibrated.

### Next Steps

1. **Fit a quadratic model** (Experiment 2): Add `year²` term to log-linear predictor
2. **Perform model comparison**: Use LOO-CV to quantify improvement
3. **Repeat posterior predictive checks** on improved model to verify deficiencies are resolved
4. **Consider mechanistic interpretation**: What scientific process could generate accelerating exponential growth?

---

## Technical Details

### Posterior Predictive Generation

- **Method**: Sampled 1000 draws from posterior, generated negative binomial predictions for all 40 observations
- **Parameterization**: `NB(mu, phi)` where `mu = exp(β₀ + β₁ × year)`, `p = φ/(φ + μ)`
- **Posterior Summary**: β₀ = 4.355 ± 0.049, β₁ = 0.863 ± 0.050, φ = 13.835 ± 3.449

### Files Generated

- **Code**: `/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc.py`
- **Plots**: `/workspace/experiments/experiment_1/posterior_predictive_check/plots/` (7 PNG files)
- **Results**: `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_results.json`
- **Report**: `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md` (this file)

---

**Assessment Complete**: This model fails posterior predictive checks and should be rejected in favor of a more flexible growth specification.
