# EDA Log: Distributional Properties and Variance Structure
**Analyst 2 - Focus: Count Distribution and Overdispersion**

## Dataset Overview
- **File**: data/data_analyst_2.csv
- **Observations**: 40 time-ordered data points
- **Variables**:
  - `year`: Standardized time variable (mean=0, std=1, range=[-1.67, 1.67])
  - `C`: Count observations (range=[19, 272])
- **Missing values**: None
- **Data quality**: Clean dataset, no apparent errors

---

## Round 1: Initial Exploration and Basic Statistics

### Hypothesis 1: Poisson Distribution is Appropriate for Count Data
**Test**: Calculate variance-to-mean ratio and basic distributional properties

**Key Findings**:
- **Mean**: 109.45
- **Variance**: 7441.74
- **Variance/Mean Ratio**: **67.99**
- **Standard Deviation**: 86.27
- **Coefficient of Variation**: 0.788
- **Skewness**: 0.602 (moderate right skew)
- **Kurtosis (excess)**: -1.233 (platykurtic - flatter than normal)

**Interpretation**:
- Poisson assumption requires Var/Mean ≈ 1
- Observed ratio of **68** indicates **EXTREME OVERDISPERSION**
- This is ~68× more variance than expected under Poisson
- **HYPOTHESIS REJECTED**: Poisson is clearly inappropriate

**Visual Evidence**: `distribution_overview.png`, `count_histogram.png`

### Hypothesis 2: Variance Structure is Constant Over Time
**Test**: Split data into temporal periods and calculate dispersion metrics

**Key Findings** (3 periods):
- **Early period** (n=13): Mean=29.54, Var=40.60, Var/Mean=1.38
- **Middle period** (n=13): Mean=73.38, Var=769.26, Var/Mean=10.48
- **Late period** (n=14): Mean=217.14, Var=1401.21, Var/Mean=6.45

**Interpretation**:
- Variance structure is **NOT constant** over time
- Middle period shows highest dispersion (10.48)
- Even "lowest" period (early) shows overdispersion (1.38 > 1)
- **HYPOTHESIS REJECTED**: Clear heteroscedasticity present

---

## Round 2: Variance-Mean Relationship and Model Selection

### Hypothesis 3: Variance Scales Linearly with Mean (Poisson-like)
**Test**: Fit power law model to variance-mean relationship using rolling windows

**Key Findings**:
- **Power law fit**: Variance = a × Mean^b
- **Exponent (b)**: 1.667 (95% CI would exclude 1.0)
- **R-squared**: 0.814
- **p-value**: 3.99e-12

**Interpretation**:
- b = 1 would indicate Poisson (variance proportional to mean)
- b = 2 would indicate variance proportional to mean²
- **Observed b = 1.67** suggests variance grows faster than mean
- This is characteristic of Negative Binomial distribution
- **HYPOTHESIS REJECTED**: Variance does NOT scale linearly with mean

**Visual Evidence**: `variance_mean_analysis.png` (log-log plot shows clear departure from slope=1)

### Distribution Fitting Results

#### Poisson Model:
- **Parameter**: λ = 109.45
- **Log-likelihood**: -1435.07
- **AIC**: 2872.13
- **BIC**: 2873.82
- **Chi-square GOF**: Cannot be reliably computed (too much overdispersion)

#### Negative Binomial Model:
- **Parameters**:
  - r (dispersion) = 1.634
  - p (probability) = 0.015
  - Alternative parameterization: α = 1/r = 0.612 (overdispersion parameter)
- **Log-likelihood**: -225.76
- **AIC**: 455.51
- **BIC**: 458.89

#### Model Comparison:
- **Δ Log-likelihood (NB - Poisson)**: +1209.31 (huge improvement!)
- **Δ AIC (NB - Poisson)**: -2416.62 (strongly favors NB)
- **Δ BIC (NB - Poisson)**: -2414.93 (strongly favors NB)

**Conclusion**:
- Negative Binomial provides **dramatically better fit**
- Improvement in log-likelihood of >1200 is exceptional
- Small r parameter (1.63) indicates substantial overdispersion
- **STRONG RECOMMENDATION**: Use Negative Binomial likelihood

**Visual Evidence**: `distribution_fitting.png` shows:
- Q-Q plot vs Poisson shows systematic deviation
- Q-Q plot vs NB shows much better fit
- Histogram overlay confirms NB matches observed distribution

---

## Round 3: Temporal Patterns and Dispersion Dynamics

### Hypothesis 4: Dispersion is Constant Across Time Periods
**Test**: Calculate rolling dispersion metrics and period-specific variance-mean ratios

**Key Findings** (5 equal periods):

| Period | Year Range | n | Mean | Variance | Var/Mean | CV |
|--------|-----------|---|------|----------|----------|-----|
| 1 | -1.67 to -1.07 | 8 | 27.0 | 49.7 | 1.84 | 0.261 |
| 2 | -0.98 to -0.38 | 8 | 38.6 | 49.1 | 1.27 | 0.181 |
| 3 | -0.30 to 0.30 | 8 | 71.4 | 357.4 | **5.01** | 0.265 |
| 4 | 0.38 to 0.98 | 8 | 167.1 | 1256.1 | **7.52** | 0.212 |
| 5 | 1.07 to 1.67 | 8 | 243.1 | 366.7 | 1.51 | 0.079 |

**Interpretation**:
- Dispersion varies **dramatically** from 1.27 to 7.52
- Middle periods (3 & 4) show highest overdispersion
- Pattern is **NON-MONOTONIC**: dispersion increases then decreases
- Coefficient of Variation is more stable (0.08 to 0.27)
- **HYPOTHESIS REJECTED**: Dispersion structure changes over time

**Heteroscedasticity Test**:
- **Breusch-Pagan statistic**: 28.28
- **p-value**: <0.0001
- **Conclusion**: Strong evidence of heteroscedasticity

**Visual Evidence**: `temporal_dispersion_rolling.png`, `temporal_periods_comparison.png`

---

## Round 4: Outlier Detection and Influential Points

### Hypothesis 5: Data Contains Outliers That Distort Distribution
**Test**: Multiple outlier detection methods (Z-score, IQR, MAD, Cook's Distance)

**Key Findings**:
- **Z-score method** (|z| > 2): 0 outliers detected
- **IQR method**: 0 outliers detected (bounds: [-206.4, 436.6])
- **MAD-based** (|modified z| > 3.5): 0 outliers detected
- **Trend-adjusted residuals** (|std resid| > 2): 0 outliers detected
- **Cook's Distance** (D > 4/n=0.10): **3 influential points**
  - Obs 1: C=29, Cook's D=0.187
  - Obs 2: C=36, Cook's D=0.172
  - Obs 36: C=272, Cook's D=0.133

**Interpretation**:
- **NO statistical outliers** detected by any method
- Three influential points are at temporal extremes (start and peak)
- These points have high leverage but are NOT aberrant
- Overdispersion is **NOT driven by outliers**
- All observations appear to be legitimate data
- **HYPOTHESIS REJECTED**: No outliers to remove; overdispersion is genuine

**Visual Evidence**: `outlier_analysis.png` shows:
- All points fall within reasonable bounds
- Influential points are at edges (expected for leverage)
- No evidence of data entry errors or measurement anomalies

---

## Alternative Hypothesis Testing

### Could Zero-Inflation Explain Overdispersion?
**Test**: Count zero observations

**Findings**:
- **Zero counts**: 0 (0.0% of data)
- **Minimum observed count**: 19

**Conclusion**: Zero-inflation is **NOT** a factor. Overdispersion must come from other sources (likely unmeasured heterogeneity or clustering).

### Is Overdispersion Due to Temporal Trend?
**Test**: Examine residuals after removing linear trend

**Findings**:
- Linear trend explains substantial variation
- **Residual std**: 29.31 (compared to raw std of 86.27)
- **Standardized residuals**: All |std resid| < 2
- Trend-adjusted data still shows dispersion

**Conclusion**: Temporal trend explains much variation, but overdispersion remains even after accounting for trend.

---

## Key Discoveries

### Discovery 1: Extreme Overdispersion
**Evidence**:
- Variance/Mean ratio of 68 (should be 1 for Poisson)
- Negative Binomial provides 1200+ improvement in log-likelihood
- Observed across ALL time periods (min dispersion still >1)

**Implications for Modeling**:
- **MUST use Negative Binomial** (or similar overdispersed) distribution
- Poisson model would be severely misspecified
- Need to account for extra-Poisson variation

### Discovery 2: Non-Linear Variance-Mean Relationship
**Evidence**:
- Power law exponent b = 1.67 (not 1.0)
- Variance grows faster than mean
- R² = 0.81 for power law fit

**Implications for Modeling**:
- Variance structure is complex
- May need flexible dispersion parameter
- Consider allowing dispersion to vary with predictors

### Discovery 3: Temporal Heteroscedasticity
**Evidence**:
- Dispersion varies 6-fold across periods (1.27 to 7.52)
- Breusch-Pagan test strongly significant (p < 0.0001)
- Non-monotonic pattern (increases then decreases)

**Implications for Modeling**:
- Consider time-varying dispersion parameter
- May need separate models for different periods
- Or include time interaction in dispersion model

### Discovery 4: No Data Quality Issues
**Evidence**:
- No missing values
- No outliers detected
- All observations appear legitimate
- Smooth temporal progression

**Implications for Modeling**:
- Can proceed with full dataset
- No need for outlier removal or imputation
- Overdispersion is genuine phenomenon

---

## Modeling Recommendations

### Primary Recommendation: Negative Binomial Regression

**Strong evidence**:
- AIC improvement: -2416
- BIC improvement: -2415
- Log-likelihood improvement: +1209

**Model form**:
```
C ~ NegBinomial(μ, r)
log(μ) = β₀ + β₁ × year
```

**Parameter estimates to expect**:
- Dispersion parameter r ≈ 1.6 (or α = 1/r ≈ 0.6)
- Could be estimated via ML or moment matching

### Alternative Model 1: NB with Time-Varying Dispersion

**Rationale**: Dispersion changes over time (1.27 to 7.52)

**Model form**:
```
C ~ NegBinomial(μ, r(t))
log(μ) = β₀ + β₁ × year
log(r) = γ₀ + γ₁ × year  (or more complex function)
```

**Advantages**:
- Captures heteroscedasticity
- More flexible variance structure
- May improve fit in middle periods

**Disadvantages**:
- More parameters to estimate
- Requires careful implementation

### Alternative Model 2: Quasi-Poisson

**Rationale**: Simple extension of Poisson for overdispersion

**Model form**:
```
E[C] = μ
Var[C] = φ × μ  (where φ is dispersion parameter)
log(μ) = β₀ + β₁ × year
```

**Advantages**:
- Simple to implement (most software packages)
- Robust standard errors
- No distributional assumptions beyond mean-variance

**Disadvantages**:
- No likelihood (can't use AIC/BIC)
- Less efficient than NB if NB is true model
- Doesn't capture non-linear variance-mean relationship

### Alternative Model 3: Generalized Additive Model (GAM)

**Rationale**: Non-linear patterns in dispersion suggest smooth function

**Model form**:
```
C ~ NegBinomial(μ, r)
log(μ) = s(year)  (where s is smooth function)
```

**Advantages**:
- Flexible functional form
- Can capture non-linear mean trend
- May improve fit

**Disadvantages**:
- More complex
- Harder to interpret
- May overfit with n=40

---

## Prior Recommendations (for Bayesian Models)

If using Bayesian approach:

### For NB Dispersion Parameter (r):
- **Prior**: Gamma(2, 1) or Exponential(1)
- **Rationale**:
  - Puts mass on small values (observed r ≈ 1.6)
  - Allows for substantial overdispersion
  - Weakly informative

### For Mean Parameters (β):
- **Prior**: Normal(0, 5) on log scale
- **Rationale**:
  - Weakly informative
  - Allows for range of mean values (19 to 272)
  - On log scale: exp(±5) covers very wide range

### For Time-Varying Dispersion:
- **Prior**: Random walk or AR(1) on log(r)
- **Rationale**: Smooth temporal changes in dispersion

---

## Data Quality Summary

### Strengths:
✓ Complete dataset (no missing values)
✓ No outliers or aberrant observations
✓ Smooth temporal progression
✓ Reasonable sample size (n=40)
✓ Clear signal in data

### Limitations:
⚠ Moderate sample size limits complex models
⚠ Extreme overdispersion may indicate unmeasured factors
⚠ Time-varying dispersion complicates modeling
⚠ No replicate observations at same time point

### Issues Requiring Attention:
❌ **NONE**: Dataset is clean and ready for modeling

---

## Visualization Summary

### Key Figures Created:

1. **distribution_overview.png**: 4-panel overview
   - Histogram with KDE
   - Box plot with violin
   - Empirical CDF
   - Time series context

2. **count_histogram.png**: Detailed frequency histogram
   - Shows right-skewed distribution
   - Clear departure from Poisson shape

3. **variance_mean_analysis.png**: 4-panel variance-mean analysis
   - Scatter with theoretical lines
   - Log-log plot (power law fit)
   - Dispersion over time
   - Period comparison

4. **distribution_fitting.png**: Model comparison
   - Histogram with Poisson/NB overlays
   - Q-Q plot vs Poisson (poor fit)
   - Q-Q plot vs NB (good fit)
   - Residual diagnostics

5. **outlier_analysis.png**: 6-panel outlier detection
   - Box plot
   - Z-scores over time
   - Residuals vs fitted
   - Cook's distance
   - Leverage vs residuals
   - Time series with flags

6. **temporal_dispersion_rolling.png**: 3 window sizes
   - Rolling mean ± SD
   - Rolling dispersion metrics

7. **temporal_periods_comparison.png**: 6-panel period analysis
   - Mean by period
   - Variance by period
   - Dispersion by period
   - CV by period
   - Mean-variance scatter
   - Skewness by period

---

## Final Conclusions

### Distribution Family:
**DEFINITIVE ANSWER: Negative Binomial**

Evidence strength: ★★★★★ (overwhelming)
- Variance/Mean ratio: 68
- AIC improvement: >2400
- Log-likelihood improvement: >1200
- Consistent across all periods

### Variance Structure:
**Heteroscedastic with non-linear mean-variance relationship**

Evidence strength: ★★★★☆ (strong)
- Breusch-Pagan p < 0.0001
- Power law exponent: 1.67 (not 1.0)
- 6-fold variation in dispersion across time

### Data Quality:
**Excellent - no issues detected**

Evidence strength: ★★★★★ (unanimous across methods)
- Zero outliers by all methods
- No missing values
- All observations legitimate

### Modeling Priority:
1. **Must do**: Use Negative Binomial (not Poisson)
2. **Should consider**: Time-varying dispersion
3. **Could explore**: GAM for non-linear trends

---

## Questions for Other Analysts

1. **Time series analyst**: Is there autocorrelation in residuals that could explain overdispersion?

2. **Covariate analyst**: Are there unmeasured factors that could explain the extreme overdispersion (r=1.6)?

3. **Model specialist**: Should we use NB1 (variance = μ + α×μ) vs NB2 (variance = μ + α×μ²)?

4. **Validation specialist**: How to assess whether time-varying dispersion model is worth the added complexity?

---

## Reproducibility

All analyses reproducible via scripts in `/workspace/eda/analyst_2/code/`:
- `01_initial_exploration.py`: Basic statistics
- `02_distribution_plots.py`: Distribution visualizations
- `03_variance_mean_analysis.py`: Variance-mean relationship
- `04_distribution_fitting.py`: Model fitting and comparison
- `05_outlier_analysis.py`: Outlier detection
- `06_dispersion_temporal.py`: Temporal dispersion patterns

All visualizations saved to `/workspace/eda/analyst_2/visualizations/`
