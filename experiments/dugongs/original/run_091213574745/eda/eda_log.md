# Exploratory Data Analysis Log

## Dataset Overview
- **File**: `/workspace/data/data.csv`
- **Observations**: 27
- **Variables**: 2 (x, Y)
- **Goal**: Understand Y ~ x relationship for Bayesian modeling

---

## Round 1: Initial Exploration

### Data Quality Assessment

**Structure:**
- 27 observations, 2 numeric variables (x, Y)
- No missing values (0%)
- Data types: both float64

**Duplicates:**
- 1 exact duplicate row found (x=12.0, Y=2.32 appears twice)
- 7 replicate x values (legitimate experimental replicates at x = 1.5, 5.0, 9.5, 12.0, 13.0, 15.5)
- **Interpretation**: Replicates suggest designed experiment or measurement replication

**Data Ranges:**
- x: [1.0, 31.5], span = 30.5
- Y: [1.77, 2.72], span = 0.95
- x shows right skew (skewness = 1.004)
- Y shows left skew (skewness = -0.741)

**Key Observations:**
- Small sample size (n=27) limits some statistical power
- x has high coefficient of variation (0.719) - wide spread
- Y has low coefficient of variation (0.118) - relatively tight spread
- No missing data - clean dataset

---

### Univariate Distributions

**Variable: x (Predictor)**
- Mean: 10.94, SD: 7.87
- Range: [1.0, 31.5]
- Distribution: Right-skewed (skewness = 1.004), platykurtic (kurtosis = 1.042)
- Shapiro-Wilk test: W = 0.9157, p = 0.0311 (rejects normality)
- **Interpretation**: x is not normally distributed; shows concentration in lower values with long right tail
- Sequential plot shows x values are pre-sorted in dataset (ascending order)

**Variable: Y (Response)**
- Mean: 2.33, SD: 0.27
- Range: [1.77, 2.72]
- Distribution: Left-skewed (skewness = -0.741), platykurtic (kurtosis = -0.277)
- Shapiro-Wilk test: W = 0.9230, p = 0.0466 (marginal rejection of normality)
- **Interpretation**: Y appears approximately normal with slight left skew; Q-Q plot shows reasonable fit to normal

**Visualizations Created:**
- `01_x_distribution.png`: Multi-panel showing histogram, KDE, boxplot, Q-Q plot, ECDF, sequential plot
- `02_Y_distribution.png`: Same diagnostics for Y

---

### Bivariate Relationship Analysis

**Correlation:**
- Pearson r = 0.8229 (p < 0.000001) - strong positive correlation
- Spearman rho = 0.9200 (p < 0.000001) - very strong monotonic relationship
- **Note**: Spearman > Pearson suggests nonlinear relationship

**Linear Regression:**
- Equation: Y = 0.0287x + 2.0198
- R² = 0.677 (explains 67.7% of variance)
- Slope = 0.0287 ± 0.0040 (highly significant, p < 0.000001)
- RMSE = 0.153

**Residual Analysis:**
- Mean residual: -0.000000 (as expected)
- SD residual: 0.156
- Skewness: -0.643 (moderate left skew)
- **Heteroscedasticity test**: Correlation between x and squared residuals = 0.0796 (p = 0.693)
  - **Conclusion**: No strong evidence of heteroscedasticity
- **Durbin-Watson**: 0.7752 (< 2 suggests positive autocorrelation in residuals)
  - **Interpretation**: Systematic pattern in residuals when sorted by x

**Key Visual Findings** (`03_bivariate_analysis.png`):
1. Scatter shows clear positive relationship but with curvature
2. Linear fit systematically underestimates in middle range
3. LOESS smooth reveals nonlinear pattern: steep rise then plateau
4. Residual plot shows systematic pattern (not random scatter)
5. Q-Q plot of residuals acceptable but shows deviation in tails

**Rate of Change Analysis** (`04_variance_analysis.png`):
- Local slopes vary dramatically across x range
- Absolute residuals show no strong trend with x (supports homoscedasticity)
- Rate of change appears to decrease with increasing x

---

## Round 2: Testing Competing Hypotheses

### Hypothesis 1: Simple Linear Relationship
- **Model**: Y = β₀ + β₁x + ε
- **R²**: 0.677, **RMSE**: 0.153
- **Evidence Against**:
  - Systematic residual pattern
  - Spearman >> Pearson correlation
  - Visual curvature in scatter plot
- **Verdict**: REJECTED - Insufficient to explain data

### Hypothesis 2: Polynomial Relationship
- **Quadratic**: R² = 0.873, RMSE = 0.096 (major improvement)
- **Cubic**: R² = 0.880, RMSE = 0.093 (marginal improvement over quadratic)
- **Evidence For**:
  - Substantial improvement in fit
  - Captures curvature
- **Evidence Against**:
  - May overfit with small n
  - Cubic shows minimal improvement over quadratic
  - Doesn't match theoretical expectation of saturation
- **Verdict**: PLAUSIBLE but potentially overfitting

### Hypothesis 3: Logarithmic/Power Law Relationship
- **Logarithmic (Y ~ log(x))**: R² = 0.897, RMSE = 0.087 (BEST simple transformation)
- **Log-log (power law)**: R² = 0.903 on log scale
- **Square root**: R² = 0.826, RMSE = 0.112
- **Evidence For**:
  - Excellent fit with just 2 parameters
  - Matches pattern of diminishing returns
  - Theoretically plausible for many physical/biological processes
- **Verdict**: STRONG CANDIDATE - Best balance of fit and parsimony

### Hypothesis 4: Asymptotic/Saturation Model
- **Model**: Y = a - b*exp(-c*x)
- **R²**: 0.889, **RMSE**: 0.090
- **Evidence For**:
  - Theoretically motivated (approach to equilibrium/saturation)
  - Good fit
  - 3 parameters with clear interpretation
- **Evidence Against**:
  - More parameters than logarithmic
  - Slightly worse R² than logarithmic
- **Verdict**: STRONG CANDIDATE - Theoretically attractive

### Hypothesis 5: Piecewise Linear (Changepoint) Model
- **Optimal breakpoint**: x = 7.0
- **Regime 1 (x ≤ 7)**: slope = 0.113, n = 9 observations
- **Regime 2 (x > 7)**: slope = 0.017, n = 18 observations
- **SSE reduction**: 66% compared to linear model
- **F-test**: F = 22.38, p = 0.000004 (highly significant)
- **Slope ratio**: 6.8:1 (Regime 1 is 6.8× steeper)
- **Jump at breakpoint**: 0.132 units (models predict slightly different values)

**Evidence For**:
- Statistically highly significant improvement
- Clear biological/physical interpretation (growth → plateau phases)
- Local slopes analysis confirms regime change
- Moving window slopes support two distinct regimes

**Evidence Against**:
- Discontinuity may be artifact of sampling
- More complex (4 parameters)

**Verdict**: VERY STRONG EVIDENCE - Supported by multiple analyses

**Visualizations Created:**
- `05_functional_forms.png`: Comparison of 6 different functional forms
- `06_transformations.png`: Log-log, semi-log, reciprocal transformations
- `07_changepoint_analysis.png`: Piecewise model vs linear, residual comparison
- `08_rate_of_change.png`: Local and moving window slopes showing regime shift

---

## Round 3: Data Quality and Robustness

### Outlier Detection

**Standardized Residuals:**
- 1 point beyond ±2 SD (3.7% - within expected range)
- 0 points beyond ±3 SD
- Outlier identified: Index 26 (x=31.5, Y=2.57, std_res=-2.31)

**High Leverage Points:**
- Mean leverage: 0.074
- 2 points exceed 2× mean leverage threshold:
  - Index 25: x=29.0, leverage=0.240
  - Index 26: x=31.5, leverage=0.300
- **Interpretation**: Extreme x values have high leverage (expected)

**Influential Points (Cook's Distance):**
- Threshold (4/n): 0.148
- 2 points exceed threshold:
  - Index 3: x=1.5, Y=1.77, Cook's D=0.190 (low end)
  - Index 26: x=31.5, Y=2.57, Cook's D=1.513 (HIGHLY INFLUENTIAL)

**DFFITS:**
- Threshold: 0.544
- Same 2 points flagged as with Cook's D

**Problematic Points:**
- Index 26 is BOTH high leverage AND outlier
- This point pulls linear fit down at high x values
- **Impact**: May artificially reduce apparent relationship strength in linear model
- **Recommendation**: Investigate this point; may represent measurement error or different regime

**Visualizations Created:**
- `09_outlier_influence.png`: Comprehensive diagnostics including Cook's D, leverage, DFFITS, residual plots

---

## Key Insights Summary

### Data Structure
1. **Two-regime pattern**: Clear evidence of changepoint at x ≈ 7
2. **Steep growth phase**: x ∈ [1, 7], rapid Y increase (slope ≈ 0.11)
3. **Plateau phase**: x ∈ (7, 32], slow Y increase (slope ≈ 0.02)
4. **Saturation behavior**: Y approaches asymptote around 2.7

### Data Quality
1. **Generally clean**: Few outliers, no missing data
2. **One influential point**: x=31.5 observation is problematic
3. **Replicated measurements**: Multiple x values have replicates (good for uncertainty estimation)
4. **Small sample**: n=27 limits some analyses but sufficient for modeling

### Modeling Implications
1. **Linear model inadequate**: Systematic residual pattern, poor fit
2. **Nonlinearity evident**: Multiple lines of evidence (Spearman > Pearson, curvature, changepoint)
3. **Best candidates**:
   - Logarithmic: Y ~ log(x) - Best R², parsimonious
   - Piecewise linear: Statistically supported, interpretable
   - Asymptotic: Theoretically motivated, good fit
4. **Heteroscedasticity**: No strong evidence, likely constant variance assumption OK
5. **Outliers**: One influential point at x=31.5 requires attention

---

## Robust Findings vs. Tentative

### ROBUST (High Confidence):
- Positive relationship between Y and x
- Nonlinear relationship (strong evidence from multiple methods)
- Two-regime structure (changepoint analysis, visual inspection, local slopes)
- Saturation/plateau at high x values
- No heteroscedasticity
- Generally clean data quality

### TENTATIVE (Requires Further Investigation):
- Exact location of changepoint (could be 6-8 range)
- Whether discontinuity is real or artifact
- Impact of x=31.5 observation (influential outlier)
- Best functional form among top candidates (log vs asymptotic vs piecewise)
- Appropriate error distribution (Normal vs Student-t for robustness)

---

## Next Steps for Modeling

1. **Fit all three candidate models in Bayesian framework**:
   - Logarithmic: Y ~ Normal(β₀ + β₁*log(x), σ)
   - Asymptotic: Y ~ Normal(a - b*exp(-c*x), σ)
   - Piecewise linear: Y ~ Normal(f(x, θ_regime), σ)

2. **Consider robust likelihood** (Student-t) given influential outlier

3. **Model comparison via**:
   - WAIC/LOO-CV for out-of-sample prediction
   - Posterior predictive checks
   - Prior predictive checks for elicitation

4. **Sensitivity analysis**: Refit without x=31.5 observation

5. **Uncertainty quantification**: Use replicates to inform σ prior

---

## Files Generated

### Code Scripts (`/workspace/eda/code/`):
1. `01_initial_exploration.py` - Data quality, structure, basic stats
2. `02_univariate_analysis.py` - Distribution analysis for x and Y
3. `03_bivariate_analysis.py` - Correlation, regression, residuals
4. `04_nonlinearity_investigation.py` - Functional forms, transformations, segmentation
5. `05_changepoint_visualization.py` - Detailed piecewise model analysis
6. `06_outlier_influence_analysis.py` - Outliers, leverage, influence diagnostics

### Visualizations (`/workspace/eda/visualizations/`):
1. `01_x_distribution.png` - Univariate analysis of predictor
2. `02_Y_distribution.png` - Univariate analysis of response
3. `03_bivariate_analysis.png` - Scatter, fits, residuals
4. `04_variance_analysis.png` - Heteroscedasticity and rate of change
5. `05_functional_forms.png` - Comparison of 6 models
6. `06_transformations.png` - Log, power, reciprocal transforms
7. `07_changepoint_analysis.png` - Piecewise model details
8. `08_rate_of_change.png` - Local slopes by regime
9. `09_outlier_influence.png` - Comprehensive diagnostics

---

## Analyst Notes

This dataset exhibits classic saturation/asymptotic behavior common in:
- Dose-response relationships
- Learning curves
- Chemical equilibria
- Growth processes approaching carrying capacity
- Michaelis-Menten enzyme kinetics

The strong evidence for two regimes suggests the underlying process may have a threshold or bifurcation point around x=7. This could represent:
- Phase transition
- Saturation of binding sites
- Resource limitation threshold
- Regime change in physical system

The small sample size (n=27) is adequate for Bayesian analysis but limits power for complex model comparison. The presence of replicates is valuable for estimating residual variance.

**Recommendation**: Pursue hierarchical Bayesian model comparison of the three top candidates, with emphasis on the logarithmic and piecewise models as they provide best balance of fit, interpretability, and theoretical motivation.
