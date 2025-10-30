# EDA Analyst 2 - Complete Analysis Package

**Analyst Focus:** Residual diagnostics, transformations, nonlinear patterns, and predictive implications
**Dataset:** 27 observations of Y vs x relationship
**Analysis Date:** 2025-10-27

---

## Quick Start

**Main Report:** [`findings.md`](findings.md) - Comprehensive findings and recommendations
**Process Log:** [`eda_log.md`](eda_log.md) - Detailed exploration process and decision points

---

## Key Findings Summary

1. **Simple linear model is inadequate** (R² = 0.677, systematic residual patterns)
2. **Log-log transformation dramatically improves fit** (R² = 0.903)
3. **Relationship follows power law:** Y ≈ 1.79 * x^0.126
4. **Strong saturation/diminishing returns pattern** evident
5. **LOO-CV confirms logarithmic model generalizes best**

**Recommended Model for Bayesian Analysis:**
```
log(Y) ~ Normal(alpha + beta * log(x), sigma)
alpha ~ Normal(0.6, 0.3)
beta ~ Normal(0.13, 0.1) [constrained beta > 0]
sigma ~ Half-Normal(0.1)
```

---

## Directory Structure

```
eda/analyst_2/
├── README.md (this file)
├── findings.md (main report)
├── eda_log.md (exploration log)
├── code/
│   ├── 01_initial_exploration.py
│   ├── 02_baseline_model_diagnostics.py
│   ├── 03_transformation_exploration.py
│   ├── 04_nonlinear_patterns.py
│   ├── 05_predictive_analysis.py
│   ├── baseline_residuals.csv
│   ├── transformation_results.csv
│   └── predictive_summary.json
└── visualizations/
    ├── 01_initial_exploration.png
    ├── 02_baseline_diagnostics.png
    ├── 03_transformation_fits.png
    ├── 04_top_transformations_diagnostics.png
    ├── 05_nonlinear_patterns.png
    ├── 06_predictive_analysis.png
    └── 07_loo_cv_residuals.png
```

---

## Analysis Components

### 1. Initial Exploration
**Script:** `code/01_initial_exploration.py`
**Visualization:** `visualizations/01_initial_exploration.png`

**Key Findings:**
- Spearman correlation (0.920) > Pearson (0.823) suggests nonlinearity
- Y left-skewed, x right-skewed
- 7 replicate x values provide measurement precision estimates

### 2. Baseline Model Diagnostics
**Script:** `code/02_baseline_model_diagnostics.py`
**Visualization:** `visualizations/02_baseline_diagnostics.png` (9-panel)
**Output:** `code/baseline_residuals.csv`

**Key Findings:**
- Linear model: Y = 2.020 + 0.0287*x, R² = 0.677
- U-shaped residual pattern indicates missing nonlinear term
- 1 outlier at x=31.5 (standardized residual = -2.23)
- Durbin-Watson = 0.775 suggests autocorrelation

### 3. Transformation Exploration
**Script:** `code/03_transformation_exploration.py`
**Visualizations:**
- `visualizations/03_transformation_fits.png` (top 12 transformations)
- `visualizations/04_top_transformations_diagnostics.png` (detailed diagnostics)
**Output:** `code/transformation_results.csv` (36 combinations tested)

**Key Findings:**
- Log-log transformation best: R² = 0.903 (+33% improvement)
- Back-transformed: Y ≈ 1.79 * x^0.126 (power law)
- Excellent residual diagnostics (Shapiro-p = 0.836)
- Alternative: Y ~ log(x) also performs well (R² = 0.897)

### 4. Nonlinear Pattern Analysis
**Script:** `code/04_nonlinear_patterns.py`
**Visualization:** `visualizations/05_nonlinear_patterns.png` (7-panel)

**Key Findings:**
- Quadratic model favored by AIC/BIC (R² = 0.874)
- Logarithmic model outperforms polynomials
- Change point detected at x = 7.4 (slope changes from 0.113 to 0.017)
- Michaelis-Menten saturation model fits well (R² = 0.835)
- Multiple approaches confirm saturation/diminishing returns

### 5. Predictive Analysis
**Script:** `code/05_predictive_analysis.py`
**Visualizations:**
- `visualizations/06_predictive_analysis.png` (bootstrap, LOO-CV, intervals)
- `visualizations/07_loo_cv_residuals.png` (model comparison)
**Output:** `code/predictive_summary.json`

**Key Findings:**
- **LOO-CV winner:** Logarithmic model (RMSE = 0.093)
- Polynomial degree 5 overfits (LOO-CV R² = 0.748 vs training 0.921)
- Bootstrap: slope uncertainty ±22% (95% CI: [0.020, 0.046])
- 2 high-leverage points at x=29, 31.5 (leverage 0.24, 0.30)
- Large data gaps for x > 17 (only 5 of 27 points)

---

## Visualization Guide

### Figure 1: Initial Exploration (6-panel)
**File:** `visualizations/01_initial_exploration.png`

Panels:
1. Scatter plot Y vs x
2. Histogram of Y
3. Histogram of x
4. Q-Q plot for Y
5. Q-Q plot for x
6. Box plots

**What to look for:** Nonlinear trend, skewed distributions

### Figure 2: Baseline Diagnostics (9-panel)
**File:** `visualizations/02_baseline_diagnostics.png`

Panels:
1. Actual vs fitted values
2. Residuals vs fitted
3. Residuals vs x
4. Q-Q plot of residuals
5. Histogram of residuals
6. Standardized residuals
7. Scale-location plot
8. Residuals by x region
9. Autocorrelation plot

**What to look for:** U-shaped residual pattern, outlier at point 26

### Figure 3: Transformation Fits (12-panel)
**File:** `visualizations/03_transformation_fits.png`

Shows top 12 transformations with fit lines and R² values

**What to look for:** Log-log transformation linearizes relationship best

### Figure 4: Top Transformations Diagnostics (16-panel, 4x4)
**File:** `visualizations/04_top_transformations_diagnostics.png`

For each of top 4 transformations:
- Fitted vs actual
- Residuals vs fitted
- Q-Q plot
- Residual histogram

**What to look for:** Improved residual patterns vs baseline

### Figure 5: Nonlinear Patterns (7-panel)
**File:** `visualizations/05_nonlinear_patterns.png`

Panels:
1. Polynomial fits comparison (degrees 1-5)
2. Nonlinear models overlay
3. Piecewise linear fit
4. Saturation models
5. Local slopes vs x
6. AIC/BIC comparison
7. Curvature analysis

**What to look for:** Change point at x≈7, saturation models

### Figure 6: Predictive Analysis (6-panel)
**File:** `visualizations/06_predictive_analysis.png`

Panels:
1. LOO-CV predicted vs actual
2. Bootstrap slope distribution
3. Bootstrap intercept distribution
4. Bootstrap R² distribution
5. Prediction & confidence intervals
6. Leverage plot

**What to look for:** Bootstrap uncertainty, high leverage points

### Figure 7: LOO-CV Residuals (6-panel)
**File:** `visualizations/07_loo_cv_residuals.png`

LOO-CV residual plots for 6 model types

**What to look for:** Logarithmic model has smallest, most random residuals

---

## Key Results Tables

### Model Comparison Summary

| Model | R² | LOO-RMSE | Parameters | Recommended? |
|-------|------|----------|------------|--------------|
| Linear | 0.677 | 0.178 | 2 | ✗ Inadequate |
| Log-log | 0.903 | **0.093** | 2 | **✓ BEST** |
| Logarithmic | 0.897 | 0.093 | 2 | ✓ Good alternative |
| Quadratic | 0.874 | 0.106 | 3 | ✓ If no transform |
| Cubic | 0.880 | 0.117 | 4 | ✗ Overfitting risk |
| Poly-4 | 0.913 | 0.097 | 5 | ✗ Overfitting |
| Poly-5 | 0.921 | 0.135 | 6 | ✗ Clear overfit |

### Transformation Rankings (Top 5)

| Rank | Y Transform | x Transform | R² | RMSE | Shapiro-p |
|------|-------------|-------------|------|--------|-----------|
| 1 | log | log | 0.903 | 0.038 | 0.836 |
| 2 | reciprocal | log | 0.902 | 0.018 | 0.919 |
| 3 | sqrt | log | 0.901 | 0.029 | 0.647 |
| 4 | none | log | 0.897 | 0.087 | 0.533 |
| 5 | log1p | log | 0.902 | 0.026 | 0.735 |

---

## Critical Findings for Bayesian Modeling

### 1. Functional Form
**Strong evidence for log-log (power law) relationship:**
- Best fit (R² = 0.903)
- Best cross-validation (LOO-RMSE = 0.093)
- Interpretable (Y ∝ x^0.126)
- Good diagnostics

### 2. Parameter Estimates (from frequentist analysis)
**For log(Y) ~ log(x):**
- Intercept (alpha): 0.581 ± 0.03
- Slope (beta): 0.126 ± 0.01
- Residual SD (sigma): 0.038

**For original scale Y ~ x:**
- Intercept: 2.020 ± 0.074
- Slope: 0.029 ± 0.007
- Residual SD: 0.159

### 3. Sample Size Considerations
- n = 27 (small sample)
- Rule of thumb: 13.5 obs/parameter for linear model
- High-degree polynomials (>3) not supported
- Use informative priors given limited data

### 4. Data Quality Issues
**High leverage points:**
- x = 29.0 (leverage = 0.24)
- x = 31.5 (leverage = 0.30, also outlier)

**Data gaps:**
- Large gap between x = 22.5 and x = 29.0 (6.5 units)
- Sparse data for x > 17 (only 5 points)

**Recommendation:** Sensitivity analysis excluding high-leverage points

### 5. Uncertainty Quantification
**Bootstrap 95% CI for linear model:**
- Slope: [0.020, 0.046] (relative uncertainty: 22%)
- Intercept: [1.863, 2.153]
- R²: [0.538, 0.855]

**Prediction intervals (95%):**
- Mean width: 0.688 (linear model)
- Substantial uncertainty for new observations

---

## Recommended Bayesian Model Specification

### Model 1: Log-Log (RECOMMENDED)

```python
# Priors
alpha ~ Normal(0.6, 0.3)          # Log-scale intercept
beta ~ Normal(0.13, 0.1)           # Power law exponent
sigma ~ HalfNormal(0.1)            # Residual SD on log scale

# Constraint
beta > 0  # Enforce monotonicity (justified by data)

# Likelihood
log(Y) ~ Normal(alpha + beta * log(x), sigma)

# Back-transformation
Y_pred = exp(alpha + beta * log(x))
      = exp(alpha) * x^beta
```

**Advantages:**
- Best predictive performance
- Interpretable as power law
- Only 2 parameters (appropriate for n=27)
- Good residual diagnostics

### Model 2: Quadratic (Alternative)

```python
# Priors
beta0 ~ Normal(2.0, 0.5)
beta1 ~ Normal(0.05, 0.1)
beta2 ~ Normal(-0.002, 0.005)
sigma ~ HalfNormal(0.15)

# Likelihood
Y ~ Normal(beta0 + beta1*x + beta2*x^2, sigma)
```

**Use if:** Log transformation is undesirable for domain reasons

### Prior Justification
- **Weakly informative:** Allow data to dominate
- **Centered on frequentist estimates:** Incorporate EDA findings
- **Appropriate scale:** Match observed data ranges
- **Constraints where justified:** beta > 0 for monotonicity

---

## Reproducibility

### Requirements
- Python 3.x
- pandas, numpy, scipy, matplotlib, seaborn

### Running the Analysis

```bash
# From /workspace directory
python eda/analyst_2/code/01_initial_exploration.py
python eda/analyst_2/code/02_baseline_model_diagnostics.py
python eda/analyst_2/code/03_transformation_exploration.py
python eda/analyst_2/code/04_nonlinear_patterns.py
python eda/analyst_2/code/05_predictive_analysis.py
```

All scripts read from: `/workspace/data/data_analyst_2.csv`
All outputs write to: `/workspace/eda/analyst_2/`

### Random Seed
Bootstrap analysis uses `np.random.seed(42)` for reproducibility.

---

## Questions for Stakeholders

1. **Is the power law relationship (Y ∝ x^0.126) scientifically plausible?**
   - This implies strong diminishing returns
   - Common in biology, economics, physics

2. **Should we be concerned about point 26 (x=31.5, Y=2.57)?**
   - High leverage and outlier
   - Check for measurement error or data entry mistake

3. **Is prediction beyond x=31.5 needed?**
   - If yes, collect more data in high-x region
   - Current model has high extrapolation uncertainty

4. **Can we collect more data for x > 17?**
   - Current: 5 points (19% of data)
   - Target: 10-15 points for better characterization

5. **Is log transformation acceptable for domain?**
   - If not, use quadratic model (second-best option)
   - Trade-off: slightly worse fit, more interpretable on original scale

---

## Contact & Attribution

**Analyst:** EDA Analyst 2
**Specialization:** Residual diagnostics, transformations, nonlinear patterns
**Date:** 2025-10-27

This analysis was conducted as part of a comprehensive multi-analyst EDA project. For questions about methodology or findings, refer to the detailed documentation in `findings.md` and `eda_log.md`.

---

**End of README**
