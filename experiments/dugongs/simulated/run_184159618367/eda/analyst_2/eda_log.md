# EDA Log - Analyst 2
## Detailed Exploration Process

### Dataset Overview
- **Size**: 27 observations, 2 variables (x, Y)
- **x range**: [1.0, 31.5], span = 30.5
- **Y range**: [1.71, 2.63], span = 0.92
- **Unique x values**: 20 (7 values have repeats)
- **Data quality**: No missing values, no exact duplicates

### Phase 1: Initial Assessment

#### Data Quality
- No missing values or duplicates
- One potential outlier in x: 31.5 (far from other values at 29.0)
- No outliers in Y by IQR method
- Both x and Y fail normality tests (Shapiro-Wilk p < 0.05)

#### Repeated X Values
Found 6 x values with multiple Y observations:
- x=1.5 (n=3): Y range=[1.712, 1.846], std=0.067
- x=5.0 (n=2): Y range=[2.165, 2.191], std=0.019
- x=9.5 (n=2): Y range=[2.395, 2.434], std=0.028
- x=12.0 (n=2): Y range=[2.431, 2.513], std=0.058
- x=13.0 (n=2): Y range=[2.574, 2.631], std=0.040
- x=15.5 (n=2): Y range=[2.410, 2.632], std=0.157

**Key Insight**: Variability at repeated x values ranges from very small (std=0.019) to substantial (std=0.157), suggesting heteroscedasticity or measurement noise.

#### Initial Correlation
- Pearson correlation: 0.7200
- Spearman correlation: 0.7816
- Difference: 0.0616 (suggests some non-linearity)

**Visualization**: `01_data_overview.png`, `02_residual_analysis.png`, `03_repeated_x_variability.png`

### Phase 2: Functional Form Exploration

Tested 7 functional forms using curve fitting:

| Model | R² | RMSE | AIC | Parameters |
|-------|-----|------|-----|------------|
| **Cubic** | 0.8975 | 0.0890 | -122.63 | 4 |
| **Asymptotic** | 0.8885 | 0.0928 | -122.38 | 3 |
| Quadratic | 0.8617 | 0.1033 | -116.56 | 3 |
| Logarithmic | 0.8293 | 0.1148 | -112.88 | 2 |
| Power Law | 0.8102 | 0.1211 | -110.00 | 2 |
| Square Root | 0.7066 | 0.1505 | -98.25 | 2 |
| Linear | 0.5184 | 0.1929 | -84.87 | 2 |

**Key Findings**:
1. **Cubic and Asymptotic models perform best** by AIC, with nearly identical fit quality
2. The asymptotic form: Y = 2.565 - 1.019*exp(-0.204*x) suggests **diminishing returns pattern**
3. Simple linear model is clearly inadequate (R² = 0.52)
4. Logarithmic and power law models also fit well, consistent with decelerating growth

**Hypothesis**: The relationship exhibits rapid growth at low x, then levels off - classic saturation curve.

**Visualization**: `04_all_functional_forms.png`, `05_top_models_comparison.png`, `06_residual_comparison.png`

### Phase 3: Local vs Global Trends

Applied multiple smoothing methods:
- LOWESS with different bandwidths (frac = 0.2, 0.3, 0.5)
- Moving averages (windows = 3, 5, 7)
- Univariate splines (failed due to repeated x values)

**Key Findings**:
1. All smoothing methods reveal similar overall pattern: steep initial rise, then plateau
2. LOWESS with frac=0.3 provides good balance between smoothness and local detail
3. Derivative analysis attempted but hampered by repeated x values causing numerical issues
4. No evidence of multiple distinct regimes, rather a smooth transition

**Visualization**: `07_smoothing_methods.png`, `08_lowess_comparison.png`, `09_derivative_analysis.png`

### Phase 4: Segmentation Analysis

Tested multiple segmentation strategies:

#### Quantile-Based (33rd, 67th percentiles: x=7.58, 13.0)
- **Low segment** (x ≤ 7.58): n=9, correlation=0.899, strong linear relationship
- **Medium segment** (7.58 < x ≤ 13.0): n=10, correlation=0.505, weaker relationship
- **High segment** (x > 13.0): n=8, correlation=0.088, essentially flat

#### Equal-Width Bins
- **Bin 1** (x ∈ [1, 11.17]): n=15, correlation=0.945 - **very strong**
- **Bin 2** (x ∈ [11.17, 21.33]): n=9, correlation=-0.149 - essentially no relationship
- **Bin 3** (x ∈ [21.33, 31.5]): n=3, correlation=-0.781 - negative (likely spurious)

#### Natural Breakpoints
Large gaps in x found between:
- x=5.0 and x=7.0 (gap=2.0)
- x=10.0 and x=12.0 (gap=2.0)
- x=17.0 and x=22.5 (gap=5.5) - **largest gap**
- x=22.5 and x=29.0 (gap=6.5)

#### Sliding Window Correlation
- Window size: 7
- Correlation ranges from -0.51 to 0.95
- High variability (std=0.43) indicates relationship changes across x range

#### Change Point Testing
Tested breakpoints at x = 5, 10, 15, 20:

| Breakpoint | Early Slope | Late Slope | Ratio |
|------------|-------------|------------|-------|
| x=5 | 0.0975 | 0.0059 | 0.06 |
| x=10 | 0.0796 | -0.0002 | -0.002 |
| x=15 | 0.0619 | -0.0002 | -0.003 |
| x=20 | 0.0482 | -0.0149 | -0.31 |

**Critical Finding**: The relationship shows **dramatic diminishing returns**. For x > 10, the slope is near zero or slightly negative, indicating Y has effectively plateaued.

**Visualization**: `10_segmentation_analysis.png`

### Phase 5: Transformation Analysis

Tested 6 transformations to linearize the relationship:

| Transform | Pearson r | R² | RMSE | Improvement |
|-----------|-----------|-----|------|-------------|
| **Log-Log** | **0.9162** | **0.8395** | 0.051 | +61.9% |
| Log(x) | 0.9107 | 0.8293 | 0.115 | +60.0% |
| 1/x | -0.8687 | 0.7547 | 0.138 | +45.6% |
| Sqrt(x) | 0.8406 | 0.7066 | 0.151 | +36.3% |
| Original | 0.7200 | 0.5184 | 0.193 | baseline |
| Log(Y) | 0.7161 | 0.5128 | 0.089 | -1.1% |

**Key Findings**:
1. **Log-Log transformation achieves near-perfect linearity** (r=0.92)
2. This implies a **power law relationship**: Y ∝ x^β
3. Log(x) transformation alone provides 83% R², suggesting **logarithmic or power law form**
4. The 1/x transformation also works reasonably well, consistent with hyperbolic/asymptotic behavior

**Implication**: The true relationship is likely Y = a * x^b where b < 1 (diminishing returns).

**Visualization**: `11_transformations.png`, `12_transformation_residuals.png`

### Phase 6: Correlation Structure and Predictive Power

#### Correlation Stability
- Bootstrap 95% CI for Pearson r: [0.562, 0.904]
- Wide confidence interval indicates **moderate uncertainty** given sample size
- Overall correlation (0.72) is robust but with substantial variability

#### Correlation by X Range
Critical finding about **non-stationarity**:
- x ∈ [1, 10): r = **0.937** (very strong positive)
- x ∈ [10, 20): r = **-0.258** (weak negative)
- x ∈ [20, 32): r = **-0.781** (strong negative, but only n=3)

**Interpretation**: The relationship is **fundamentally different** across x ranges. At low x, there's strong predictive power. At high x, Y is essentially constant or declining.

#### Variance Decomposition
- Total Y variance: 0.0802
- Variance explained by linear model: 51.8%
- Residual variance: 48.2%

Nearly half the variance is unexplained by a simple linear model, but non-linear models explain ~85-90%.

#### Prediction Intervals
95% prediction intervals for new observations:
- At x=5: [1.74, 2.59], width=0.85
- At x=10: [1.87, 2.72], width=0.84
- At x=25: [2.24, 3.13], width=0.89

**Insight**: Prediction uncertainty is substantial (±0.42 on average), representing ~18% of the Y range.

#### Influential Observations
Cook's distance identified 3 influential points:
1. x=31.5 (most influential) - distant outlier in x-space
2. x=29.0 - also in sparse high-x region
3. x=1.5 - extreme low-x point with multiple observations

**Visualization**: `13_correlation_structure.png`

### Competing Hypotheses Tested

#### Hypothesis 1: Linear Relationship
**REJECTED** - R²=0.52, clear patterns in residuals, poor fit at extremes

#### Hypothesis 2: Power Law / Logarithmic Growth
**STRONGLY SUPPORTED** - Log(x) and log-log transformations achieve R²>0.83, theoretical support from diminishing returns

#### Hypothesis 3: Asymptotic Plateau
**STRONGLY SUPPORTED** - Asymptotic model (Y = a - b*exp(-c*x)) fits excellently (R²=0.89), segmentation shows plateau at x>10

#### Hypothesis 4: Piecewise Linear (Change Point)
**PARTIALLY SUPPORTED** - Clear evidence of regime shift around x=10, but smooth transition more parsimonious than hard breakpoint

#### Hypothesis 5: Polynomial (Quadratic/Cubic)
**SUPPORTED** - Cubic model has best AIC (-122.63), but less interpretable than asymptotic form

### Data Generation Process Considerations

Based on the pattern observed, plausible data generating mechanisms:
1. **Learning/Saturation Process**: Early rapid gains that level off (e.g., learning curves, dose-response)
2. **Resource Limitation**: Some limiting factor prevents Y from continuing to grow
3. **Logarithmic Utility**: Diminishing marginal returns at higher x values
4. **Physical Constraint**: Y may have a theoretical maximum near 2.6-2.7

The heteroscedasticity at repeated x values suggests:
- Measurement error in Y
- Unmeasured confounders
- Stochastic variation in the underlying process

### Tentative vs Robust Findings

#### Robust Findings (High Confidence):
- Non-linear relationship with diminishing returns pattern
- Strong predictive power of x for Y in range x < 10
- Plateau behavior for x > 10
- Log transformations substantially improve linearity
- Asymptotic/power law models fit well

#### Tentative Findings (Lower Confidence):
- Exact form of optimal model (cubic vs asymptotic very close)
- Behavior at x > 30 (sparse data)
- Whether negative correlation at high x is real or artifact
- Homoscedasticity assumption (mixed evidence)

### Unanswered Questions

1. Is there truly a hard upper limit to Y, or just very slow growth at high x?
2. What causes the varying amounts of noise at different x values?
3. Are there unobserved variables that would explain residual variance?
4. Would more data at x > 20 change the plateau interpretation?
5. Is the cubic polynomial overfitting, or capturing real curvature?

### Recommendations for Further Analysis

1. Collect more data at x > 20 to confirm plateau behavior
2. Investigate causes of heteroscedasticity
3. Consider Bayesian approach to incorporate prior knowledge about saturation
4. Test formal change-point detection methods
5. Explore weighted regression to account for varying noise levels
