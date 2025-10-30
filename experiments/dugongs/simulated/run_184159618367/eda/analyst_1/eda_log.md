# EDA Exploration Log - Analyst 1

## Dataset Overview
- **File**: data/data_analyst_1.csv
- **Observations**: 27
- **Variables**: x (predictor), Y (response)
- **Date**: Analysis conducted systematically in two rounds

---

## Round 1: Initial Exploration

### Data Quality Assessment
**Script**: `code/01_initial_exploration.py`

**Findings**:
- No missing values or duplicate rows (excellent data quality)
- Both variables are continuous (float64)
- x range: [1.00, 31.50], span = 30.50
- Y range: [1.71, 2.63], span = 0.92

**Distribution Properties**:
- x is right-skewed (skewness = 0.947), non-normal (Shapiro-Wilk p = 0.031)
- Y is left-skewed (skewness = -0.830), non-normal (Shapiro-Wilk p = 0.003)
- 20 unique x values with 6 having replicates (valuable for pure error estimation)
- Replicate structure:
  - x = 1.5 (3 replicates)
  - x = 5.0, 9.5, 12.0, 13.0, 15.5 (2 replicates each)

**Correlation Analysis**:
- Pearson r = 0.720, p < 0.001 (strong positive linear correlation)
- Spearman rho = 0.782, p < 0.001 (stronger rank correlation suggests nonlinearity)
- The difference between Pearson and Spearman suggests monotonic but possibly nonlinear relationship

### Visual Exploration
**Script**: `code/02_relationship_visualizations.py`

**Key Observations from Scatter Plots** (`01_scatter_with_smoothers.png`):
- Clear positive relationship between x and Y
- Linear fit appears adequate but smoothers suggest curvature
- Possible saturation effect at higher x values
- No obvious outliers in the main relationship

**Distribution Analysis** (`02_distributions.png`):
- x shows bimodal tendency with concentration at lower and mid values
- Y shows relatively symmetric distribution around median despite statistical non-normality
- Q-Q plots confirm deviations from normality, especially in tails
- Mean and median are close for Y (2.32 vs 2.43), more separated for x (10.9 vs 9.5)

**Segmented Analysis** (`03_segmented_relationship.png`):
Divided data into thirds by x:
- **Low x segment** (n=9, x ≤ 7.0): Y mean = 1.968, std = 0.179
- **Mid x segment** (n=10, 7.0 < x ≤ 13.0): Y mean = 2.483, std = 0.109
- **High x segment** (n=8, x > 13.0): Y mean = 2.509, std = 0.089

**Critical Insight**: Y increases substantially from low to mid x (0.52 units), but barely changes from mid to high x (0.03 units). This suggests a **saturation or plateau effect**.

### Linear Model Diagnostics
**Script**: `code/03_linear_residual_analysis.py`

**Model**: Y = 2.0353 + 0.0259 * x
- R² = 0.518 (explains only 52% of variance)
- RMSE = 0.193
- Residuals appear normal (Shapiro-Wilk p = 0.334)

**Residual Diagnostics** (`04_residual_diagnostics.png`):
1. **Residuals vs Fitted**: Clear U-shaped pattern in smoother indicates systematic lack of fit
2. **Q-Q Plot**: Residuals reasonably normal (a positive finding for inference)
3. **Scale-Location**: Some evidence of non-constant variance but not severe
4. **Residuals vs x**: Similar U-shape confirms nonlinearity

**Heteroscedasticity Assessment**:
- Breusch-Pagan statistic = 0.36 (low, minimal heteroscedasticity)
- Variance ratio (high/low x): 2.16 (moderate increase in variance with x)
- Durbin-Watson = 0.66 (suggests positive autocorrelation when ordered by x)

**Interpretation**: The positive autocorrelation in residuals ordered by x is a strong indicator that a linear model misses systematic patterns - likely the saturation effect observed in segmented analysis.

---

## Round 2: Hypothesis Testing and Deep Dive

### Competing Model Hypotheses
**Script**: `code/04_hypothesis_testing.py`

Tested five different functional forms to explain x-Y relationship:

#### Model Comparison Results (`05_model_comparison.png`):

| Model | R² | RMSE | AIC | ΔAIC | Interpretation |
|-------|-----|------|-----|------|----------------|
| **Broken-stick** | 0.904 | 0.086 | -122.4 | 0.0 | **Best model** |
| Quadratic | 0.862 | 0.103 | -116.6 | 5.9 | Strong performer |
| Logarithmic | 0.829 | 0.115 | -112.9 | 9.5 | Good fit |
| Saturation (M-M) | 0.816 | 0.119 | -110.8 | 11.6 | Mechanistically appealing |
| Linear | 0.518 | 0.193 | -84.9 | 37.5 | Poor fit |

**Key Findings**:

1. **Broken-stick model (piecewise linear)** performs best:
   - Breakpoint at x = 9.5
   - Segment 1 (x ≤ 9.5): Y = 1.723 + 0.0775*x (steep slope)
   - Segment 2 (x > 9.5): Y = 2.539 - 0.0009*x (essentially flat)
   - Explains 90.4% of variance (huge improvement over linear)
   - ΔAIC = 37.5 units better than linear (decisive evidence)

2. **Quadratic model**: Y = 1.746 + 0.0862*x - 0.002*x²
   - Second best performer
   - Negative x² term confirms diminishing returns
   - R² = 0.862, more parsimonious than broken-stick

3. **Logarithmic model**: Y = 1.751 + 0.275*log(x)
   - Natural for processes with diminishing returns
   - Good fit (R² = 0.829)
   - Interpretable: constant percentage change in x leads to constant absolute change in Y

4. **Saturation (Michaelis-Menten)**: Y = 2.587 * x / (0.644 + x)
   - Mechanistically interpretable (asymptotic approach to Ymax)
   - Ymax (asymptote) = 2.59
   - K (half-maximum) = 0.64 (very low, suggests rapid saturation)
   - Slightly worse fit than log/quadratic but more interpretable

**Hypothesis Conclusion**: The relationship is **definitively nonlinear** with a clear saturation/plateau pattern. The broken-stick model's dominance suggests there may be an underlying threshold or regime change around x = 9-10.

### Influence and Outlier Analysis
**Script**: `code/05_influence_outliers.py`

**Diagnostic Thresholds**:
- Leverage threshold (2p/n): 0.148
- Cook's D threshold (4/n): 0.148

**Findings** (`06_influence_diagnostics.png`):

1. **High Leverage Points** (2 observations):
   - x = 29.0 (leverage = 0.240, std residual = -1.90)
   - x = 31.5 (leverage = 0.300, std residual = -1.95)
   - Both are extreme x values with moderate negative residuals

2. **Influential Points** (Cook's D > 0.148, 3 observations):
   - x = 1.5 (Cook's D = 0.183): Low x, low Y - defines lower bound
   - x = 29.0 (Cook's D = 0.567): High x, moderate influence
   - x = 31.5 (Cook's D = 0.812): **Most influential** - extreme x value

3. **No traditional outliers**: All |standardized residuals| < 2

**Interpretation**:
- The extreme x values (29, 31.5) are influential because they're in the sparse high-x region
- However, they're not outliers - they follow the saturation pattern
- Point at x = 1.5 is influential for setting the lower relationship bound
- **Robust conclusion**: The saturation pattern is not driven by outliers but is a genuine data feature

### Variance Structure Analysis
**Script**: `code/06_variance_structure.py`

**Heteroscedasticity Tests** (`07_variance_structure.png`):
- Correlation(|residuals|, x) = 0.059, p = 0.77 (no systematic pattern)
- Levene's test: p = 0.81 (variances homogeneous across x bins)
- **Conclusion**: Homoscedastic residuals (good for standard regression assumptions)

**Pure Error Analysis**:
Using the 6 x-values with replicates:
- Pooled pure error SD = 0.075 (n=7 df)
- Model residual SD = 0.197 (from linear model)
- **Ratio = 6.82** (model variance / pure error variance)

**Critical Interpretation**: A ratio >> 1 means the model residual variance is much larger than pure experimental noise. This confirms **substantial lack of fit** for the linear model. The structure in the residuals is real signal that should be captured by a nonlinear model.

**Within-Replicate Variability**:
- Replicates show consistent small variation (SD typically 0.02-0.07)
- Exception: x = 15.5 has SD = 0.157 (larger but still reasonable)
- Pure error is fairly constant across x range (supports homoscedasticity)

---

## Key Questions Explored

### Q1: Is the relationship linear?
**Answer**: No. Multiple lines of evidence:
- Segmented analysis shows plateau effect
- Linear model R² = 0.52 (poor fit)
- Residuals show systematic U-shaped pattern
- Nonlinear models improve R² by 0.31-0.39

### Q2: What is the nature of nonlinearity?
**Answer**: Saturation/diminishing returns pattern:
- Rapid increase in Y at low x
- Leveling off around x = 9-10
- Near-plateau at high x
- Broken-stick model suggests possible threshold

### Q3: Are there outliers or data quality issues?
**Answer**: No significant concerns:
- No missing values
- No extreme outliers (all |std residuals| < 2)
- Influential points are legitimate boundary values
- Pure error is reasonable and consistent

### Q4: Is variance constant (homoscedastic)?
**Answer**: Yes, largely homoscedastic:
- No significant correlation between |residuals| and x
- Levene's test non-significant
- Pure error estimates fairly consistent
- Slight increase in variance at high x but not problematic

---

## Modeling Recommendations

### Strongly Recommended Models:

1. **Piecewise Linear (Broken-stick)**
   - Best fit (R² = 0.904)
   - Clear interpretability: threshold at x ≈ 9.5
   - Useful if there's a mechanistic reason for regime change
   - **Use case**: If x represents a treatment dose or resource level with saturation

2. **Quadratic Polynomial**
   - Second best fit (R² = 0.862)
   - Simple, interpretable
   - Good for interpolation
   - **Caution**: May predict unrealistic decreases beyond observed x range

3. **Logarithmic Transformation**
   - Good fit (R² = 0.829)
   - Natural for diminishing returns processes
   - Stable extrapolation behavior
   - **Use case**: If Y represents a response to cumulative stimulus

### Moderately Recommended:

4. **Saturation/Michaelis-Menten Model**
   - Mechanistically interpretable (asymptotic Ymax = 2.59)
   - Appropriate for biological/chemical processes
   - Slightly lower fit but constrained to sensible behavior
   - **Use case**: If domain knowledge suggests true asymptotic plateau

### Not Recommended:

5. **Simple Linear Model**
   - R² = 0.52 (inadequate)
   - Systematic lack of fit
   - Ignores clear saturation pattern

---

## Data Characteristics for Future Reference

- **Sample size**: Small (n=27) - limits complex model fitting
- **x distribution**: Right-skewed, range [1, 31.5]
- **Y distribution**: Left-skewed, range [1.71, 2.63]
- **Replication**: 6 x-values with replicates (22% of data) - good for model validation
- **Influence structure**: Extreme x values influential but not problematic
- **Error structure**: Homoscedastic, approximately normal
- **Pure error**: Small (SD ≈ 0.075) relative to model lack of fit

---

## Tentative vs. Robust Findings

### ROBUST (high confidence):
- Nonlinear relationship with saturation
- Homoscedastic residuals
- No outlier problems
- Threshold/inflection around x = 9-10
- Nonlinear models improve fit by ~35-40 percentage points in R²

### TENTATIVE (lower confidence):
- Exact breakpoint location (limited by sample size)
- Choice between quadratic vs. saturation model (both fit well)
- Variance slightly higher at extreme x (small effect, could be sampling)
- Extrapolation beyond x = 31.5 uncertain

---

## Unexpected Findings

1. **Strong saturation effect**: Expected some nonlinearity but the near-complete plateau above x=10 was striking

2. **Cleanliness of data**: No outliers, no missing values, reasonable pure error - unusually clean dataset

3. **Broken-stick dominance**: Didn't expect piecewise model to outperform smooth curves so decisively - suggests genuinely different regimes

4. **Low K value in M-M model** (K = 0.64): Suggests very rapid saturation, reaching half-maximum by x ≈ 0.64

---

## Next Steps / Questions for Further Analysis

1. **Biological/Physical context**: What does x represent? Is there a mechanistic explanation for the threshold at x ≈ 9.5?

2. **Cross-validation**: With n=27, would be valuable to assess prediction error via LOOCV

3. **Confidence bands**: Uncertainty quantification for the nonlinear models, especially in extrapolation regions

4. **Alternative breakpoint**: Test other breakpoint locations systematically (grid search)

5. **Transformation approaches**: Consider Box-Cox transformation to find optimal power transformation

6. **Compare with other analyst**: Does independent analysis corroborate the saturation pattern?

---

## Files Generated

### Code (`eda/analyst_1/code/`):
- `01_initial_exploration.py` - Data quality and descriptive statistics
- `02_relationship_visualizations.py` - Scatter plots and distribution analysis
- `03_linear_residual_analysis.py` - Linear model diagnostics
- `04_hypothesis_testing.py` - Competing model comparison
- `05_influence_outliers.py` - Leverage and Cook's distance analysis
- `06_variance_structure.py` - Heteroscedasticity and pure error assessment

### Visualizations (`eda/analyst_1/visualizations/`):
- `01_scatter_with_smoothers.png` - Multiple smoothing methods overlay
- `02_distributions.png` - Histograms and Q-Q plots for x and Y
- `03_segmented_relationship.png` - Relationship by x range thirds
- `04_residual_diagnostics.png` - 4-panel residual diagnostic plots
- `05_model_comparison.png` - Competing models overlaid and metrics
- `06_influence_diagnostics.png` - Leverage, Cook's D, and influence plots
- `07_variance_structure.png` - Variance consistency and pure error analysis

---

**Analysis completed**: Comprehensive two-round exploration with systematic hypothesis testing
