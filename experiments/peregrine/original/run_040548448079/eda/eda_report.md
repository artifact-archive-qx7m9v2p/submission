# Exploratory Data Analysis Report

## Dataset Overview
- **Observations**: 40 time-ordered count measurements
- **Variables**:
  - `year`: Standardized time variable (mean=0, std=1, range=[-1.67, 1.67])
  - `C`: Count observations (range=[19, 272], mean=109.45, std=86.27)
- **Quality**: Excellent (no missing values, no outliers)

## Key Findings

### 1. Distribution Family: NEGATIVE BINOMIAL (Definitive)

**Evidence**:
- Variance/Mean ratio: **67.99** (Poisson assumes 1.0)
- Model comparison ΔAIC: **-2417** (overwhelming evidence for NB over Poisson)
- Estimated dispersion: r = 1.634, α = 0.612
- Time-varying dispersion: 6-fold variation across time (1.27 to 7.52)

**Conclusion**: Poisson is fundamentally inappropriate; Negative Binomial is required.

### 2. Temporal Structure: STRUCTURAL BREAK + STRONG AUTOCORRELATION

**Structural Break** (4 independent tests confirm):
- **Location**: Observation 17 (standardized year ≈ -0.21)
- **Magnitude**: 730% increase in growth rate
  - Pre-break slope: 14.87
  - Post-break slope: 123.36
- **Evidence**: Chow test, CUSUM, rolling statistics, grid search all converge

**Autocorrelation**:
- Raw data ACF(1) = 0.944 (strong non-stationarity)
- First differences achieve stationarity (I(1) process)
- All single-trend model residuals show temporal dependency

**Growth Dynamics**:
- Total growth: 745% over 40 observations
- Mean per-period growth: 7.72% (highly variable)
- Second half mean: 4.88× higher than first half

### 3. Functional Form: EXPONENTIAL/QUADRATIC

**Log Transformation Analysis**:
- Linearity on log scale: r = 0.967
- Variance stabilization: ratio drops from 34.7× to 0.58×
- Box-Cox optimal λ = -0.036 (confirms log, where λ=0)
- Shapiro-Wilk on log residuals: p = 0.945 (excellent normality)

**Polynomial Analysis**:
- Quadratic improvement: ΔAIC = -41.4 (highly significant)
- Cubic polynomial R² = 0.976 (single-trend)
- But two-regime model achieves 80% improvement over cubic

**Growth Pattern**:
- Exponential model R² = 0.935 (growth rate 2.34×/year)
- Consistent with log-linear relationship

### 4. Variance Structure: HETEROSCEDASTIC

**Key Findings**:
- Power law: Variance ∝ Mean^1.67 (non-linear)
- Breusch-Pagan test: p < 0.0001 (significant heteroscedasticity)
- Time-varying dispersion present
- Log transformation nearly stabilizes variance

## Modeling Implications

### Must-Have Features
1. **Negative Binomial likelihood** (not Poisson)
2. **Log link function** (count data + exponential growth)
3. **Autocorrelation structure** (AR terms, GP, or time-series model)
4. **Structural break accommodation**:
   - Option A: Two-regime model with changepoint
   - Option B: Smooth transition (splines, GP)
   - Option C: Quadratic/cubic trend (acknowledging limitations)

### Model Classes to Consider

**Tier 1 (Primary Recommendations)**:
1. **Changepoint Negative Binomial Regression**
   - Two regimes with separate slopes
   - Log link: log(μ) = β₀ + β₁×year + β₂×(year>τ)×(year-τ)
   - AR(1) errors or observation-level random effects

2. **Negative Binomial GP Regression**
   - Gaussian Process on log-rate
   - Captures smooth non-linear trends and autocorrelation
   - Negative Binomial observation model

3. **Hierarchical Negative Binomial with Splines**
   - B-splines or natural cubic splines for smooth trend
   - Time-varying dispersion
   - AR structure on observation level

**Tier 2 (Simpler Alternatives)**:
4. **Polynomial Negative Binomial Regression** (quadratic or cubic)
   - Log link: log(μ) = β₀ + β₁×year + β₂×year²
   - Easier to fit but may miss structural break
   - Should include AR terms

5. **Negative Binomial Dynamic Linear Model**
   - State-space formulation
   - Time-varying coefficients
   - Handles autocorrelation naturally

### Prior Recommendations

Based on EDA findings:

**Intercept** (log-scale mean at year=0):
- Observed: log(C) at year≈0 is log(74.5) ≈ 4.31
- Prior: Normal(4.3, 0.5) or Student-t(5, 4.3, 0.5)

**Slope(s)**:
- Pre-break: gentle (β₁ ≈ 0.3-0.4 on log scale)
- Post-break: steep (β₂ ≈ 1.0-1.2 on log scale)
- Priors: Normal(0, 1) weakly informative, or use EDA estimates

**Dispersion**:
- Observed: r ≈ 1.63, α ≈ 0.61
- Prior: Gamma(2, 1) or Exponential(1) for α

**Changepoint** (if using two-regime):
- Strong evidence for τ ≈ observation 17 (year ≈ -0.21)
- Prior: Uniform over plausible range or concentrated around -0.2

**Autocorrelation** (if AR(1)):
- Strong positive autocorrelation expected
- Prior: Beta(8, 2) giving E[ρ] ≈ 0.8

## Data Quality Assessment

✅ **Excellent Quality**:
- No missing values (0/40)
- No outliers detected (0/40 by all methods)
- 3 influential points at temporal extremes (legitimate)
- Complete time series
- Consistent measurement scale

✅ **Use full dataset** (all 40 observations)

## Critical Warnings

❌ **DO NOT**:
- Use Poisson regression (fundamentally wrong, ΔAIC = +2417)
- Ignore autocorrelation (will severely underestimate uncertainty)
- Ignore structural break (will lead to poor fit and misleading inference)
- Use standard GLM without accounting for temporal dependency
- Extrapolate beyond observed range without extreme caution

⚠️ **MUST CONSIDER**:
- Time-series cross-validation (not random splits)
- Autocorrelation in residuals
- Model comparison via LOO or WAIC (accounting for temporal structure)
- Posterior predictive checks for autocorrelation and structural break

## Visual Evidence Summary

**Key Plots**:
1. `/workspace/eda/analyst_1/visualizations/00_summary_dashboard.png` - Temporal patterns overview
2. `/workspace/eda/analyst_1/visualizations/07_structural_breaks.png` - Structural break evidence
3. `/workspace/eda/analyst_2/visualizations/KEY_RESULT_summary.png` - Distribution evidence
4. `/workspace/eda/analyst_3/visualizations/01_transformation_comparison.png` - Transformation analysis

## Synthesis Across Analysts

All three analysts converged on:
1. **Negative Binomial distribution** (not Poisson)
2. **Log link function** (exponential growth + count data)
3. **Structural break** at observation 17
4. **Strong autocorrelation** requiring temporal structure
5. **Quadratic or two-regime** predictor structure
6. **Excellent data quality** (use all observations)

Divergences (complementary perspectives):
- Analyst 1 emphasized discrete regime change
- Analyst 2 quantified overdispersion mechanism
- Analyst 3 showed smooth transformations (log) work well

## Recommended Next Steps

1. **Model Design**: Propose 3-4 model classes incorporating:
   - Negative Binomial likelihood
   - Log link
   - Structural break (changepoint or smooth)
   - Autocorrelation structure

2. **Implementation Priority**:
   - Start with changepoint NB regression (most aligned with EDA)
   - Try GP-based model (flexible, captures autocorrelation)
   - Consider polynomial NB as simpler baseline

3. **Validation Strategy**:
   - Prior predictive checks (ensure priors allow observed patterns)
   - Simulation-based calibration (can model recover parameters?)
   - Posterior predictive checks (autocorrelation, structural break)
   - LOO cross-validation (temporal structure preserved)

---

**Analysis conducted by 3 parallel EDA analysts with systematic hypothesis testing and convergent validation.**
