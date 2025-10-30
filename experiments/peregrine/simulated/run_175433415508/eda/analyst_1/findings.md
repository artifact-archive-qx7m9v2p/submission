# EDA Findings Report: Distributional Properties and Count Characteristics
**Analyst 1 - Comprehensive Analysis**

---

## Executive Summary

This report presents a systematic exploratory data analysis of count variable **C** measured over **year**, focusing on distributional properties, overdispersion, and temporal patterns. The analysis was conducted in two iterative rounds, testing multiple competing hypotheses.

### Critical Findings

1. **Strong exponential growth**: Count C exhibits exponential growth with year (R² = 0.94)
2. **Severe overdispersion**: Variance is 70× the mean (Poisson completely inappropriate)
3. **Quadratic mean-variance relationship**: Var ∝ Mean² (power law with exponent ≈ 2)
4. **Trend-induced heterogeneity**: 88% of variance explained by temporal trend
5. **High data quality**: No missing values, outliers, or quality issues

### Primary Recommendation

**Negative Binomial GLM with log link and year as covariate** is the preferred modeling approach, with Quasi-Poisson GLM as a robust alternative.

---

## Data Overview

- **Sample size**: 40 observations
- **Variables**: year (continuous, standardized), C (count, integer)
- **Data quality**: Excellent - no missing values, no anomalies, no outliers
- **Count range**: 21 to 269
- **Zero-inflation**: None (minimum count = 21)

---

## Key Findings

### 1. Distribution Shape and Characteristics

**Visual Evidence**: `01_distribution_overview.png`

**Summary Statistics**:
- Mean: 109.4
- Median: 67.0
- Standard Deviation: 87.78
- Skewness: 0.64 (right-skewed)
- Kurtosis: -1.13 (platykurtic, lighter tails than normal)

**Finding**: The distribution is **right-skewed** with mean substantially higher than median (109.4 vs 67.0). The histogram shows a bimodal-like pattern with concentrations in lower values (20-70) and higher values (160-270).

**Interpretation**: This is NOT a typical count distribution. The wide spread and bimodality are driven by the strong temporal trend pooling observations from different time periods with vastly different expected values.

**Plot Reference**:
- Histogram in `01_distribution_overview.png` (top-left panel)
- Boxplot in `01_distribution_overview.png` (top-right panel) shows right skew with no outliers

---

### 2. CRITICAL: Extreme Overdispersion

**Visual Evidence**: `03_variance_mean_relationship.png`, `04_theoretical_distributions.png`

**Quantitative Evidence**:
- **Variance**: 7,704.66
- **Mean**: 109.4
- **Variance-to-Mean Ratio**: **70.43**
- Chi-square test: p < 0.000001 (highly significant)

**Finding**: The data exhibits **extreme overdispersion** - variance is more than 70 times the mean. This completely violates the Poisson assumption (Var = Mean).

**Interpretation**:
- **Poisson distribution is completely inappropriate** for this data
- The variance-mean plot (`03_variance_mean_relationship.png`) shows all data groups fall far above the Poisson identity line
- The Poisson theoretical distribution (`04_theoretical_distributions.png`, left panel) is far too narrow
- Negative Binomial provides better fit but still imperfect

**Root Cause**: Two-component overdispersion:
1. **Between-time heterogeneity** (trend-induced): 88% of variance
2. **Within-time variability** (residual): 12% of variance

**Implication**: Count models MUST account for year as a covariate and accommodate residual overdispersion.

---

### 3. Strong Temporal Trend (Exponential Growth)

**Visual Evidence**: `02_temporal_pattern.png`, `07_model_comparison.png`

**Quantitative Evidence**:
- **Pearson correlation**: r = 0.939 (p < 0.000001)
- **Spearman correlation**: ρ = 0.954 (p < 0.000001)
- **Linear model R²**: 0.881
- **Log-linear model R²**: 0.937 (superior)

**Finding**: Count C exhibits **exponential growth** over time. The log-linear model (log(C) = 4.33 + 0.86×year) fits better than the linear model (MSE: 482 vs 892).

**Interpretation**:
- The temporal trend explains **88% of total variance**
- Growth rate: 137% per standardized year unit
- This pattern is characteristic of:
  - Population growth processes
  - Compound accumulation
  - Epidemic/diffusion phenomena

**Plot Reference**:
- `02_temporal_pattern.png` shows strong upward trend with tight fit
- `07_model_comparison.png` compares linear vs log-linear fits (log-linear superior in all panels)

**Implication**: Any count model should use **log link** to match the exponential growth pattern.

---

### 4. Major Discovery: Quadratic Mean-Variance Relationship

**Visual Evidence**: `06_mean_variance_relationship.png`

**Quantitative Evidence**:
- **Power law fit**: Variance = 0.057 × Mean^2.01
- **R²**: 0.843 (excellent fit)
- **P-value**: < 0.000001 (highly significant)
- **Power exponent**: 2.01 ≈ 2 (quadratic)

**Finding**: Variance scales as the **square of the mean** (not linearly as in Poisson).

**Interpretation**: This quadratic relationship has profound modeling implications:
- **Rules out Poisson**: Poisson has Var = Mean (power = 1)
- **Consistent with Negative Binomial**: NB has Var = μ + μ²/θ
- **Suggests small dispersion parameter**: For NB, small θ makes quadratic term dominate

This relationship is characteristic of:
- Multiplicative (rather than additive) variability
- Heterogeneous populations being aggregated
- Log-normal or gamma-distributed rates in underlying Poisson process

**Plot Reference**:
- `06_mean_variance_relationship.png` (left panel) shows all points far above Poisson line
- `06_mean_variance_relationship.png` (right panel, log-log scale) shows excellent linear fit with slope ≈ 2

**Implication**: Negative Binomial or Quasi-Poisson models are natural choices.

---

### 5. Detrending Analysis: Two Sources of Variability

**Visual Evidence**: `05_residual_diagnostics.png`, `07_model_comparison.png`

**Quantitative Evidence**:
- **Original variance**: 7,704.66
- **Residual variance** (after linear detrending): 915.05
- **Variance explained by trend**: **88.12%**
- **Residual Var / Predicted Mean**: 8.36

**Finding**: Detrending dramatically reduces variance, but **substantial overdispersion remains**.

**Interpretation**:
1. The extreme apparent overdispersion (Var/Mean = 70.43) was largely driven by pooling across time periods
2. However, even after accounting for year, residuals show meaningful overdispersion (ratio ≈ 8.4)
3. This confirms **two distinct sources** of variability:
   - **Trend-induced** (between-time): 88% of variance
   - **Residual** (within-time): 12% of variance, but still substantial

**Residual Properties**:
- **Shapiro-Wilk test**: W = 0.955, p = 0.112 (cannot reject normality)
- **Q-Q plot** (`05_residual_diagnostics.png`, top-right): approximately normal with slight tail deviation
- **Residuals vs Fitted** (`05_residual_diagnostics.png`, top-left): no clear systematic pattern
- **Scale-Location plot** (`05_residual_diagnostics.png`, bottom-left): suggests non-constant variance

**Implication**: Models must include year AND accommodate residual overdispersion (can't just rely on covariate).

---

### 6. Non-Stationary Process: Distribution Changes Over Time

**Visual Evidence**: `08_temporal_distribution_changes.png`

**Quantitative Evidence** (splitting data into thirds):

| Period | Mean | SD | Coefficient of Variation |
|--------|------|----|-----------------------|
| Early (obs 1-13) | 28.3 | 4.3 | 0.151 |
| Middle (obs 14-26) | 70.7 | 22.5 | 0.318 |
| Late (obs 27-40) | 215.6 | 35.6 | 0.165 |

**Finding**: The **coefficient of variation is NOT constant** - it peaks in the middle period.

**Interpretation**:
- The data generation process is **non-stationary**
- Distribution shifts systematically: mean increases 7.6× from early to late period
- The middle period shows highest relative variability (CV = 0.318)
- Early and late periods show lower relative variability (CV ≈ 0.15-0.16)

**Plot Reference**: `08_temporal_distribution_changes.png` shows three histograms side-by-side with very different shapes and scales.

**Implication**: Simple scaling assumptions may not hold throughout. Consider time-varying dispersion or non-linear time effects.

---

### 7. Complex Variance Structure

**Visual Evidence**: `05_residual_diagnostics.png`

**Quantitative Evidence** (residual variance by quartile of predicted values):
- Q1 (lowest predictions): Var = 525.49
- Q2: Var = 89.15 (much lower!)
- Q3: Var = 372.13
- Q4 (highest predictions): Var = 514.96

**Finding**: Residual variance exhibits a **U-shaped pattern** - highest at extremes, lowest in middle range.

**Interpretation**:
- This is NOT simple heteroscedasticity (monotonic increase)
- Suggests different mechanisms at different scales
- However, formal Breusch-Pagan test shows no significant linear relationship (r = -0.15, p = 0.35)

**Plot Reference**: Scale-Location plot in `05_residual_diagnostics.png` (bottom-left) shows this pattern visually.

**Status**: TENTATIVE finding (based on small samples in each quartile, n=10)

**Implication**: Further investigation needed. May require flexible variance modeling.

---

### 8. No Data Quality Issues

**Evidence**: Multiple diagnostic checks across all analyses

**Findings**:
- **No missing values**: Complete dataset (40/40 observations)
- **No outliers**: IQR method and Z-score method both confirm (0 outliers)
- **All values are valid integers**: Count data integrity confirmed
- **No negative values**: All counts ≥ 21
- **No zero-inflation**: 0% zeros (minimum count = 21)
- **Temporal sequence intact**: Year values equally spaced

**Plot Reference**: Boxplot in `01_distribution_overview.png` shows no outlier points beyond whiskers.

**Implication**: Data is analysis-ready. No preprocessing required before modeling.

---

## Model Recommendations

Based on comprehensive analysis, the following approaches are recommended **in order of preference**:

### 1. Negative Binomial GLM (PREFERRED) ⭐

**Model Specification**:
```
Response: C (count)
Distribution: Negative Binomial
Link: log
Formula: log(E[C]) = β₀ + β₁×year
```

**Rationale**:
- Naturally accommodates overdispersion via dispersion parameter θ
- Log link matches exponential growth pattern (R² = 0.94 in log-linear model)
- Variance function Var = μ + μ²/θ can accommodate quadratic mean-variance relationship
- Can incorporate year as covariate (explains 88% of variance)
- Widely supported in statistical software (R, Python, Stata, etc.)

**Advantages**:
- Handles both trend-induced and residual overdispersion
- Full likelihood available (allows AIC/BIC model comparison)
- Allows formal inference on growth rate via β₁
- Can extend to include year² or splines if needed

**Implementation Notes**:
- Estimate dispersion parameter θ from data
- Check if single θ is adequate (dispersion diagnostic plots)
- Consider robust standard errors as sensitivity check
- Validate on held-out data if available

**Expected Performance**:
- Should achieve R² > 0.90 (based on log-linear model's 0.94)
- Residual deviance should be close to residual df if well-fitted

---

### 2. Quasi-Poisson GLM (ROBUST ALTERNATIVE)

**Model Specification**:
```
Response: C
Family: Quasi-Poisson (quasi-likelihood)
Link: log
Formula: log(E[C]) = β₀ + β₁×year
```

**Rationale**:
- Relaxes Var = Mean assumption while keeping Poisson-like structure
- Simpler than NB (fewer distributional assumptions)
- Appropriate when exact distribution is uncertain
- Accounts for overdispersion via estimated dispersion parameter φ

**Advantages**:
- Very robust to distribution misspecification
- Standard errors automatically adjusted for overdispersion
- Simpler than full NB (no additional shape parameter to estimate)
- Works well when focus is on mean structure (not distribution details)

**Disadvantages**:
- No full likelihood (can't compute AIC for model selection)
- Less efficient than NB if NB is the true model
- Cannot generate predictive distributions (only point predictions)

**Implementation Notes**:
- Dispersion parameter φ estimated from Pearson χ²
- Use for inference on coefficients with proper SEs
- Good for sensitivity analysis comparing with NB

---

### 3. Log-Linear Gaussian Regression (SIMPLE ALTERNATIVE)

**Model Specification**:
```
Response: log(C)
Distribution: Normal
Formula: log(C) = β₀ + β₁×year + ε, where ε ~ N(0, σ²)
```

**Rationale**:
- Round 2 analysis showed log(C) has approximately normal residuals (Shapiro-Wilk p = 0.11)
- Best empirical fit in direct model comparison (MSE = 482 vs 892 for linear)
- Simpler to interpret than count models
- Familiar to broad audience

**Advantages**:
- Best fit among tested models (lowest MSE)
- Residuals pass normality test
- Standard linear regression (maximum simplicity)
- Easy interpretation: β₁ is proportional growth rate

**Disadvantages**:
- Back-transformation from log scale introduces bias (need smearing estimator)
- Doesn't respect count nature of data (not a count model)
- May predict non-integers or negatives after back-transformation
- Philosophically less appropriate for count data

**Implementation Notes**:
- Use Duan's smearing estimator for back-transformation: E[C] = exp(β₀ + β₁×year) × (1/n)Σexp(residuals)
- Report R² on log scale: 0.937
- Check residual diagnostics carefully

**Recommendation**: Use as benchmark for comparison, but prefer NB for final model.

---

## Key Modeling Features to Include

Regardless of model class chosen:

1. **Year as primary covariate**: Mandatory (explains 88% of variance)
2. **Log link function**: Matches exponential growth pattern
3. **Overdispersion accommodation**: Via NB, quasi-likelihood, or robust SEs
4. **Diagnostics**:
   - Residual plots (deviance, Pearson)
   - Dispersion parameter checks
   - Rootogram (for count models)
   - Cross-validation (if data permits)

### Extensions to Consider

- **Non-linear time effects**: Try year² or natural splines
- **Time-varying dispersion**: Allow θ to vary with predictors
- **Robust standard errors**: Sandwich estimators for inference

---

## Diagnostic Checklist for Modeling Phase

After fitting models, perform these diagnostics:

### Essential
- [ ] Residual vs fitted plot (check for patterns)
- [ ] Q-Q plot of randomized quantile residuals (for count models)
- [ ] Dispersion parameter estimate and interpretation
- [ ] Compare observed vs predicted counts (rootogram for count models)
- [ ] Check for influential observations (Cook's distance, DFBETAS)

### Recommended
- [ ] Cross-validation (if sufficient data)
- [ ] Compare models via AIC/BIC (NB) or pseudo-R² (Quasi-Poisson)
- [ ] Simulate from fitted model and compare to observed data
- [ ] Check predictions at extremes (boundary of data range)

### Optional
- [ ] Bootstrap confidence intervals
- [ ] Sensitivity to outliers (though none detected)
- [ ] Test for non-linear time effects (year²)

---

## Summary Tables

### Descriptive Statistics
| Statistic | Value |
|-----------|-------|
| N | 40 |
| Mean | 109.4 |
| Median | 67.0 |
| Std Dev | 87.78 |
| Variance | 7,704.66 |
| Min | 21 |
| Max | 269 |
| Skewness | 0.64 |
| Kurtosis (excess) | -1.13 |
| Var/Mean Ratio | 70.43 |

### Model Comparison
| Model | R² | MSE | Key Feature |
|-------|------|-----|-------------|
| Linear | 0.881 | 892 | Simple but inadequate |
| Log-linear | 0.937 | 482 | Best empirical fit |
| NB GLM | ~0.94* | TBD | Best for count data* |

*Expected based on log-linear performance; actual fitting needed

### Temporal Patterns
| Metric | Value |
|--------|-------|
| Correlation (Pearson) | 0.939 |
| Correlation (Spearman) | 0.954 |
| Variance explained by year | 88.12% |
| Growth rate (log scale) | 86% per unit year |

---

## Robust vs Tentative Findings

### ROBUST (High Confidence) ✓
- Strong positive temporal trend (r = 0.94, p < 0.001)
- Severe overdispersion relative to Poisson (Var/Mean = 70.43)
- Exponential growth pattern (log-linear R² = 0.937)
- No outliers or data quality issues (multiple diagnostic checks)
- Power law mean-variance relationship (power ≈ 2, R² = 0.84)
- Trend explains 88% of variance (detrending analysis)

### TENTATIVE (Medium Confidence) ⚠
- U-shaped residual variance pattern (based on n=10 per quartile)
- Distinct regimes in early/middle/late periods (CV differences could be sampling variation)
- Specific NB parameters (r=1.56, p=0.014) based on method of moments, not MLE

### REQUIRES FURTHER INVESTIGATION ❓
- Whether non-linear time effects (year²) improve fit
- Presence of additional unmeasured covariates explaining residual variation
- Predictive performance on held-out data
- Whether dispersion parameter constant over time
- Domain context (what does C represent?)

---

## Visualizations Created

All plots saved to `/workspace/eda/analyst_1/visualizations/` at 300 dpi:

1. **01_distribution_overview.png** (4-panel):
   - Histogram with mean/median lines
   - Boxplot with summary statistics
   - Q-Q plot (normal)
   - Empirical CDF

2. **02_temporal_pattern.png**:
   - Scatter plot of C vs year with linear regression fit
   - Shows strong upward trend (R² = 0.88)

3. **03_variance_mean_relationship.png**:
   - Groups plotted on variance-mean coordinates
   - Shows extreme overdispersion (all points far above Poisson line)

4. **04_theoretical_distributions.png** (2-panel):
   - Left: Empirical vs Poisson distribution (poor fit)
   - Right: Empirical vs Negative Binomial (better but imperfect)

5. **05_residual_diagnostics.png** (4-panel):
   - Residuals vs Fitted
   - Normal Q-Q plot of residuals
   - Scale-Location plot
   - Histogram of residuals with normal overlay

6. **06_mean_variance_relationship.png** (2-panel):
   - Left: Linear scale showing quadratic relationship
   - Right: Log-log scale showing power law (slope ≈ 2)

7. **07_model_comparison.png** (4-panel):
   - Linear model fit and residuals
   - Log-linear model fit and residuals
   - Shows log-linear superiority

8. **08_temporal_distribution_changes.png** (3-panel):
   - Histograms for early, middle, late periods
   - Shows non-stationary distribution

---

## Code and Reproducibility

All analysis code saved to `/workspace/eda/analyst_1/code/`:
- `01_initial_exploration.py` - Round 1 descriptive analysis
- `02_visualization_round1.py` - Round 1 plots
- `03_round2_detrending.py` - Round 2 detrending and hypothesis testing
- `04_visualization_round2.py` - Round 2 plots

All code is fully reproducible and well-documented.

---

## Conclusions

This comprehensive EDA reveals a **high-quality dataset** with **strong temporal structure** and **substantial overdispersion**. The count variable C exhibits clear exponential growth over time (R² = 0.94) with a quadratic mean-variance relationship (Var ∝ Mean²).

**Primary recommendation**: Fit a **Negative Binomial GLM** with log link and year as covariate. This model class naturally handles the observed patterns and is well-suited for count data with overdispersion.

**No data quality issues** were identified, and the data is **ready for modeling** without preprocessing.

**Key insight**: The extreme overdispersion (Var/Mean = 70) is primarily driven by the temporal trend (88% of variance), but meaningful residual overdispersion remains (12%), necessitating appropriate modeling choices.

---

**Report prepared by EDA Analyst 1**
**Analysis date**: 2025-10-29
**Total observations analyzed**: 40
**Visualizations created**: 8 multi-panel figures (300 dpi)
**Code files**: 4 Python scripts (fully reproducible)
