# EDA Log: Temporal Patterns and Trends Analysis
## Analyst 1

**Dataset:** data/data_analyst_1.csv
**Focus:** Temporal trend analysis, functional forms, autocorrelation, structural breaks
**Date:** 2025-10-29

---

## Exploration Process

### Phase 1: Initial Data Understanding

**Objective:** Understand the basic structure and characteristics of the time series

**Actions:**
- Loaded 40 time-ordered observations
- Verified data quality (no missing values, no duplicates)
- Examined variable ranges and distributions

**Key Findings:**
- Year variable is standardized (mean=0, std=1) with uniform spacing (0.0855 units)
- Count variable (C) ranges from 19 to 272
- Strong positive growth: 745% increase from first to last observation
- High coefficient of variation (0.79) suggests heteroscedasticity
- Second half mean (181.7) is 4.88x the first half mean (37.2)

**Interpretation:** The data exhibits strong non-stationary behavior with substantial growth over time. The magnitude of change suggests either exponential growth or a structural regime change.

---

### Phase 2: Functional Form Testing

**Objective:** Determine which mathematical function best describes the temporal trend

**Hypotheses Tested:**
1. H1: Linear growth (constant absolute change)
2. H2: Quadratic growth (accelerating at constant rate)
3. H3: Cubic growth (non-monotonic acceleration)
4. H4: Exponential growth (constant percentage change)
5. H5: Log-linear growth (exponential on original scale)

**Methods:**
- Fitted 5 different functional forms
- Compared using R², RMSE, and AIC
- Examined residual patterns and autocorrelation

**Results:**

| Model | R² | RMSE | AIC | Durbin-Watson |
|-------|-----|------|-----|---------------|
| Cubic | 0.9757 | 13.28 | 214.91 | 0.8400 |
| Quadratic | 0.9610 | 16.81 | 231.78 | 0.5694 |
| Exponential | 0.9366 | 21.44 | 249.22 | 0.3852 |
| Log-Linear | 0.9286 | 22.76 | 253.99 | 0.3643 |
| Linear | 0.8846 | 28.94 | 273.22 | 0.1951 |

**Interpretation:**
- Cubic model provides best fit (R² = 0.976)
- All models show strong positive residual autocorrelation (DW < 1)
- Even the best model leaves substantial temporal dependency unexplained
- The cubic form suggests initial acceleration followed by deceleration

**Visualization:** `01_trend_comparison.png`, `02_top_models_panel.png`

---

### Phase 3: Growth Rate Analysis

**Objective:** Understand how the rate of change varies over time

**Methods:**
- Calculated first differences (absolute changes)
- Calculated percentage changes (growth rates)
- Applied rolling window smoothing
- Examined log-scale to test exponential hypothesis

**Results:**
- Mean absolute change: 5.54 (std: 12.87)
- Mean percentage change: 7.72% (std: 21.37%)
- Median percentage change: 6.25%
- First half mean % change: 8.02%
- Second half mean % change: 7.43%
- Exponential model implies ~114% growth per standardized year unit

**Interpretation:**
- High variability in growth rates (std > mean) indicates non-constant growth
- Percentage changes are relatively stable between halves (~8% vs ~7%)
- However, absolute changes increase dramatically due to larger base values
- This pattern is consistent with either exponential growth or regime change

**Visualization:** `03_growth_rates.png`

---

### Phase 4: Autocorrelation Analysis

**Objective:** Examine temporal dependencies and assess stationarity

**Methods:**
- Computed ACF and PACF for raw data
- Computed ACF and PACF for first differences
- Calculated ACF for residuals from each model
- Applied Ljung-Box test for autocorrelation

**Results - Raw Data:**
- ACF(1) = 0.944 (extremely high)
- ACF decays slowly, suggesting strong persistence
- PACF(1) = 0.944, then oscillates (AR(1) pattern)
- Ljung-Box test: Q = 198.96, p < 0.001 (highly significant)

**Results - First Differences:**
- ACF(1) = -0.255 (mild negative correlation)
- ACF oscillates around zero with no clear pattern
- Ljung-Box test: Q = 13.63, p = 0.19 (not significant)
- Differencing achieves approximate stationarity

**Results - Model Residuals:**
- Linear model: ACF(1) = 0.858 (poor, high autocorrelation remains)
- Quadratic model: ACF(1) = 0.609 (moderate improvement)
- Cubic model: ACF(1) = 0.514 (best, but still significant)
- Exponential model: ACF(1) = 0.717 (worse than quadratic)
- All models show significant Ljung-Box statistics (p < 0.05)

**Interpretation:**
- Raw data is clearly non-stationary with strong trend
- First differencing successfully removes non-stationarity
- Even best polynomial models fail to capture all temporal structure
- Residual autocorrelation suggests omitted time-varying factors or regime changes
- For modeling: need to account for autocorrelation (ARIMA, state-space, or GLS)

**Visualization:** `04_acf_pacf_analysis.png`, `05_residual_diagnostics.png`, `06_residual_acf_comparison.png`

---

### Phase 5: Structural Break Analysis

**Objective:** Test whether the data represents a single continuous process or multiple distinct regimes

**Hypotheses:**
- H0: Single regime with smooth trend
- H1: Two or more distinct regimes with different dynamics

**Methods Applied:**
1. Chow test at midpoint
2. CUSUM test for parameter stability
3. Rolling window slope analysis
4. Exhaustive breakpoint search

**Method 1 - Chow Test (midpoint):**
- Split at observation 20 (year = 0.043)
- First half slope: 20.18
- Second half slope: 122.06
- F-statistic: 68.72, p < 0.001
- **Conclusion: Highly significant structural break**

**Method 2 - CUSUM Test:**
- CUSUM exceeds critical bounds by large margin
- Maximum CUSUM = 39.05 vs critical = ±7.74
- **Conclusion: Strong rejection of parameter stability**

**Method 3 - Rolling Window Analysis:**
- Window size: 10 observations
- First third mean slope: 19.29
- Last third mean slope: 133.12
- t-statistic: -9.64, p < 0.001
- Slope increases by 690% across the series
- **Conclusion: Dramatic acceleration over time**

**Method 4 - Optimal Breakpoint Search:**
- Tested all possible breakpoints (observations 5-35)
- Optimal break at observation 17 (year = -0.214)
- Segment 1 slope: 14.87
- Segment 2 slope: 123.36
- Slope increases by 730%
- SSE improvement: 79.91% over single model
- **Conclusion: Very strong evidence for regime change**

**Overall Interpretation:**
All four methods converge on the same conclusion: there is a dramatic structural break around observation 17-20. This represents a fundamental change in the data-generating process, not just smooth acceleration.

**Implications:**
- The data should NOT be modeled as a single continuous process
- Two-regime models will dramatically outperform single-trend models
- The break occurs slightly before the midpoint (year ≈ -0.21)
- Post-break dynamics are fundamentally different (8x steeper slope)

**Visualization:** `07_structural_breaks.png`, `08_regime_comparison.png`

---

## Summary of Key Insights

### What We Know with High Confidence:
1. **Strong non-stationarity:** Raw data has persistent trend and high autocorrelation
2. **Structural break exists:** Multiple tests unanimously detect regime change around obs 17-20
3. **Two distinct regimes:** Pre-break (gentle growth) and post-break (steep acceleration)
4. **First-difference stationarity:** Differencing removes trend, suggesting I(1) process
5. **Residual autocorrelation:** Even best models leave unexplained temporal structure

### What Remains Uncertain:
1. **Nature of the break:** Is it a sudden shock or gradual transition?
2. **Within-regime dynamics:** Are there additional patterns within each regime?
3. **Cause of acceleration:** What drives the 730% increase in growth rate?
4. **Variability patterns:** Does variance change with level (heteroscedasticity)?

### Competing Hypotheses for Further Testing:

**Hypothesis A: Pure Regime Change**
- Two distinct linear trends with abrupt transition
- Evidence: Chow test, breakpoint search
- Best for: Simple interpretation, regime-specific forecasting

**Hypothesis B: Non-linear Smooth Transition**
- Cubic or logistic growth with single underlying process
- Evidence: Cubic model has highest R²
- Best for: Continuous forecasting, fewer parameters

**Hypothesis C: Exponential with Regime Change**
- Exponential growth rate that shifts at breakpoint
- Evidence: Stable percentage changes, log-linearity within regimes
- Best for: Percentage-based thinking, multiplicative errors

---

## Recommendations for Modeling

### Priority 1: Address Structural Break
- **Do not use single-regime models** - they will badly misfit
- Consider: Segmented regression, switching regression, or separate models per regime
- If forced to use single model, cubic provides best compromise

### Priority 2: Handle Autocorrelation
- Standard OLS will underestimate standard errors
- Consider: ARIMA framework, GLS, or state-space models
- First differencing can induce stationarity but loses level information

### Priority 3: Model the Count Nature
- Counts are non-negative integers, not truly continuous
- Consider: Poisson regression, negative binomial, or zero-inflated models
- Log-link can naturally accommodate exponential growth

### Suggested Model Classes:

1. **Segmented Poisson Regression**
   - Poisson/negative binomial with regime-specific parameters
   - Allows for overdispersion and discrete count nature
   - Can include AR errors or random effects

2. **Structural Break State-Space Model**
   - Kalman filter with known breakpoint
   - Estimates time-varying coefficients
   - Naturally handles autocorrelation

3. **Two-Stage Approach**
   - Model each regime separately
   - Poisson or negative binomial for each segment
   - Simplest interpretation and implementation

---

## Files Generated

### Code:
- `code/01_initial_exploration.py` - Data loading and basic statistics
- `code/02_trend_analysis.py` - Functional form testing
- `code/03_visualize_trends.py` - Trend visualization
- `code/04_autocorrelation_analysis.py` - ACF/PACF analysis
- `code/05_visualize_acf_residuals.py` - Autocorrelation plots
- `code/06_structural_breaks.py` - Structural break tests
- `code/07_visualize_breaks.py` - Break point visualization

### Visualizations:
- `visualizations/01_trend_comparison.png` - All functional forms overlaid
- `visualizations/02_top_models_panel.png` - Top 3 models with confidence bands
- `visualizations/03_growth_rates.png` - Changes and growth rates over time
- `visualizations/04_acf_pacf_analysis.png` - ACF/PACF for raw and differenced data
- `visualizations/05_residual_diagnostics.png` - Residual plots for top models
- `visualizations/06_residual_acf_comparison.png` - Residual autocorrelation by model
- `visualizations/07_structural_breaks.png` - Comprehensive break analysis
- `visualizations/08_regime_comparison.png` - Single vs two-regime comparison

### Data:
- `code/trend_models.pkl` - Fitted model results
- `code/acf_data.pkl` - Autocorrelation results
- `code/structural_breaks.pkl` - Structural break test results

---

## Next Steps for Deeper Analysis

1. **Estimate transition smoothness** - Is break sharp or gradual?
2. **Test for additional breaks** - Are there more than two regimes?
3. **Examine variance structure** - Does variability scale with level?
4. **Cross-validate breakpoint** - Is observation 17 stable across resampling?
5. **Investigate causality** - What external factors might explain the break?
