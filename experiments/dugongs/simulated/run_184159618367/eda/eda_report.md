# Exploratory Data Analysis: Final Report
## Relationship between Y and x

**Date**: 2025
**Dataset**: 27 observations
**Analysis Method**: Dual independent EDA with synthesis

---

## Executive Summary

Analysis of 27 observations reveals a **strong nonlinear saturation relationship** between predictor x and response Y. Key findings:

- **Pattern**: Rapid Y increase at low x (1-10), plateau at high x (>10)
- **Strength**: Moderate-to-strong correlation (r=0.72), much stronger at low x
- **Best Models**: Nonlinear models (R²=0.83-0.90) vastly superior to linear (R²=0.52)
- **Data Quality**: Excellent - no missing data, no problematic outliers, 6 replicated x-values
- **Implication**: Bayesian models must incorporate saturation mechanism

---

## 1. Data Overview

### Basic Characteristics
- **Sample Size**: N = 27
- **Variables**: 2 continuous variables
  - `x`: Predictor, range [1.0, 31.5], mean=10.94, SD=7.87
  - `Y`: Response, range [1.71, 2.63], mean=2.32, SD=0.28

### Data Quality
✓ No missing values
✓ No duplicates
✓ 6 x-values with replicates (enables pure error estimation)
✓ No extreme outliers (x=31.5 somewhat influential but not aberrant)
✓ Clean distributions

**Verdict**: Data quality is **EXCELLENT** and suitable for Bayesian modeling.

---

## 2. Primary Finding: Nonlinear Saturation Pattern

### Visual Evidence
The relationship exhibits clear **diminishing returns**:

```
x=1-10:   Y increases from ~1.8 to ~2.5 (+0.7 units, 70% of total range)
x=10-32:  Y remains ~2.5-2.6 (+0.1 units, 10% of total range)
```

**Key Visualization**: See `eda/analyst_1/visualizations/00_comprehensive_summary.png` and `eda/analyst_2/visualizations/00_SUMMARY_comprehensive.png`

### Quantitative Evidence

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pearson r | 0.720*** | Strong positive correlation |
| Spearman ρ | 0.782*** | Even stronger rank correlation |
| Spearman > Pearson | Yes | Indicates monotonic but nonlinear |
| Segmented correlation (x<10) | 0.94*** | Very strong at low x |
| Segmented correlation (x≥10) | -0.03 | No relationship at high x |

***p < 0.001

### Regime Analysis

| x Range | n | Y Mean | Y SD | Slope |
|---------|---|--------|------|-------|
| Low (x ≤ 7) | 9 | 1.968 | 0.179 | 0.080 |
| Mid (7 < x ≤ 13) | 10 | 2.483 | 0.109 | 0.024 |
| High (x > 13) | 8 | 2.509 | 0.089 | ~0.000 |

**Interpretation**: Slope decreases by ~94% from low to high x region, confirming saturation.

---

## 3. Model Comparison: Nonlinear Models Dominate

Seven functional forms tested independently by both analysts:

| Model Class | R² | RMSE | AIC | Parameters | Recommendation |
|-------------|-----|------|-----|------------|----------------|
| **Piecewise Linear** | **0.904** | 0.086 | -122.4 | 4 | **Best fit** |
| **Asymptotic Exponential** | **0.889** | 0.093 | -122.4 | 3 | **Most interpretable** |
| Cubic Polynomial | 0.898 | 0.089 | -122.6 | 4 | Flexible but complex |
| Quadratic Polynomial | 0.862 | 0.103 | -116.6 | 3 | Good compromise |
| Logarithmic | 0.829 | 0.115 | -112.9 | 2 | Simple, parsimonious |
| Power Law (log-log) | 0.810 | 0.121 | -110.0 | 2 | Theoretically grounded |
| **Linear** | **0.518** | **0.193** | **-84.9** | **2** | **INADEQUATE** |

**Key Insights**:
1. All nonlinear models achieve R² > 0.80
2. Linear model R² = 0.52 (30-40 percentage points worse)
3. ΔAIC = 37.5 between best model and linear (decisive difference)
4. Top models are statistically equivalent (ΔAIC < 2)

**Visual Comparison**: See `eda/analyst_2/visualizations/04_all_functional_forms.png`

---

## 4. Recommended Functional Forms for Bayesian Modeling

### Tier 1: Primary Candidates (Must Test)

#### 1. Asymptotic Exponential Model
```
Y = α - β * exp(-γ * x)
```
**Fitted**: Y = 2.565 - 1.019 × exp(-0.204 × x)

**Strengths**:
- Theoretically motivated (saturation processes, learning curves)
- Clear parameter interpretation:
  - α = 2.565: asymptote (upper limit of Y)
  - β = 1.019: amplitude (difference from min to max)
  - γ = 0.204: rate (speed of saturation)
- Excellent fit (R²=0.889)
- Only 3 parameters

**Bayesian Advantages**:
- Interpretable priors (e.g., asymptote from data range)
- Handles uncertainty in saturation level
- Natural for mechanistic interpretation

#### 2. Piecewise Linear Model (Broken Stick)
```
Y = β₀ + β₁*x                    if x ≤ τ
Y = β₀ + β₁*τ + β₂*(x - τ)      if x > τ
```
**Fitted**: Breakpoint τ = 9.5, slope before = 0.078, slope after ≈ 0

**Strengths**:
- Best empirical fit (R²=0.904)
- Captures regime shift explicitly
- Simple interpretation (two linear segments)

**Bayesian Advantages**:
- Can infer breakpoint location (uncertainty in τ)
- Tests hypothesis of sharp vs smooth transition
- Hierarchical variants possible

#### 3. Quadratic Polynomial
```
Y = β₀ + β₁*x + β₂*x²
```
**Fitted**: Y = 1.75 + 0.086x - 0.002x²

**Strengths**:
- Good fit (R²=0.862)
- Simple, standard approach
- Captures curvature with 3 parameters

**Bayesian Advantages**:
- Easy to implement (linear in parameters)
- Gaussian priors straightforward
- Fast inference

### Tier 2: Alternative Approaches

#### 4. Log-Log Model (Power Law)
```
log(Y) = α + β * log(x)
Equivalent to: Y = exp(α) * x^β
```
**Fitted**: Y = 1.798 × x^0.121 (R²=0.81)

**Strengths**:
- Transforms to linear problem (r=0.92 after log-log transform)
- Theoretically grounded (power law common in nature)
- Robust to outliers

**Bayesian Advantages**:
- Can use Gaussian likelihood on log-scale
- Simple parameter interpretation (power law exponent)

#### 5. Gaussian Process (Consider if time permits)
**Strengths**:
- Maximum flexibility
- Quantifies uncertainty everywhere
- No parametric form assumed

**Cautions**:
- Computationally intensive for small N
- Requires covariance function selection
- Less interpretable parameters

---

## 5. Prior Elicitation Guidance

Based on EDA findings, suggested priors for Bayesian models:

| Parameter | Symbol | Domain Knowledge | Suggested Prior | Rationale |
|-----------|--------|-----------------|----------------|-----------|
| Asymptote | α | Y plateaus at ~2.5-2.6 | Normal(2.55, 0.1) | Observed plateau |
| Minimum Y | Y₀ | Extrapolate to x→0: ~1.6 | Normal(1.65, 0.2) | Back-extrapolation |
| Rate | γ | Transition over ~10 x-units | Gamma(2, 10) | E[γ]=0.2 |
| Residual SD | σ | Pure error ~0.075-0.12 | Half-Cauchy(0, 0.15) | Replicates |
| Breakpoint | τ | Visual break at x=9-10 | Normal(9.5, 1.5) | Segmented analysis |
| Poly coefs | β₁, β₂ | Positive β₁, negative β₂ | Normal(0, 1) weakly informative | Standard |

**Prior Predictive Check Goals**:
- Y should stay in [1.0, 3.5] (observed range ± buffer)
- Should generate saturation patterns (not monotone unbounded)
- Asymptote should be achievable within x range

---

## 6. Likelihood Recommendations

### Primary: Gaussian Likelihood
```
Y ~ Normal(μ(x), σ)
```
**Rationale**:
- Residuals approximately normal after accounting for nonlinearity
- Variance appears constant across x range (homoscedastic)
- Simple, standard approach

**Prior for σ**: Half-Cauchy(0, 0.15) based on pure error ~0.075

### Alternative: Student-t Likelihood (Robust)
```
Y ~ Student-t(ν, μ(x), σ)
```
**Rationale**:
- Robust to potential outliers (e.g., x=31.5)
- Heavy tails provide insurance
- Can estimate degrees of freedom ν or fix at ν=4-7

**Prior for ν**: Gamma(2, 0.1) if estimated (weakly favors ν=20)

### Consider if Needed: Heteroscedastic Models
If posterior predictive checks show variance issues:
```
Y ~ Normal(μ(x), σ(x))
σ(x) = σ₀ * |x|^α  or  σ(x) = σ₀ * exp(α*x)
```
**Note**: Current evidence suggests this is unlikely to be necessary.

---

## 7. Model Validation Strategy

### Simulation-Based Calibration
For each model:
1. Simulate synthetic data from known parameters
2. Recover parameters via MCMC
3. Check parameter recovery (within 95% credible intervals)
4. Verify saturation pattern is reproducible

**Target**: >95% coverage for all parameters

### Posterior Predictive Checks
Focus on:
1. **Saturation pattern**: Model should predict plateau at high x
2. **Replicated x-values**: Check prediction at 6 x-values with replicates
3. **Residual structure**: No systematic patterns
4. **Variance**: Consistent across x range

**Metrics**:
- Proportion of data in posterior predictive intervals (target: ~95%)
- Visual checks of Y_rep vs Y_obs
- Residual plots (should be homoscedastic, centered at 0)

### Model Comparison
- **LOO-CV** (via PSIS-LOO): Assess out-of-sample predictive accuracy
- **ΔELPD**: Compare models, account for SE
- **Pareto-k diagnostics**: Check for influential observations (especially x=31.5)

**Decision Rule**:
- If |ΔELPD| > 2×SE: Prefer model with higher ELPD
- If |ΔELPD| < 2×SE: Prefer simpler/more interpretable model

---

## 8. Expected Challenges & Mitigation

### Challenge 1: Nonlinear Parameter Inference
**Issue**: Asymptotic model requires non-centered parameterization or good initialization
**Mitigation**:
- Use OLS fits to initialize MCMC
- Consider non-centered parameterization
- Monitor R-hat and effective sample size

### Challenge 2: Extrapolation Uncertainty
**Issue**: Limited data for x > 20 (only 3 observations)
**Mitigation**:
- Informative priors on asymptote
- Wide posterior predictive intervals for high x
- Emphasize in interpretation

### Challenge 3: Model Selection Ambiguity
**Issue**: Multiple models may fit equally well (ΔAIC < 2)
**Mitigation**:
- Use LOO-CV for out-of-sample comparison
- Prefer simpler/more interpretable if predictive accuracy equivalent
- Report model averaging if appropriate

### Challenge 4: Breakpoint Uncertainty (Piecewise Model)
**Issue**: Exact location of τ may be uncertain
**Mitigation**:
- Put prior on τ: Normal(9.5, 1.5)
- Report posterior distribution of τ
- Check sensitivity to prior choice

---

## 9. Falsification Criteria

A Bayesian model will be considered **adequate** if:

1. ✓ **Convergence**: R-hat < 1.01 for all parameters
2. ✓ **Fit**: Captures saturation pattern visually
3. ✓ **Performance**: Posterior predictive R² > 0.85
4. ✓ **Validation**: >90% of observed Y in 95% posterior predictive intervals
5. ✓ **Residuals**: No systematic patterns in posterior predictive residuals
6. ✓ **LOO**: No Pareto-k values > 0.7 (no highly influential points)

A model will be **rejected** if:
- ✗ Fails to converge after tuning (R-hat > 1.01)
- ✗ Systematic residual patterns remain
- ✗ Poor out-of-sample prediction (LOO-R² < 0.75)
- ✗ Fails to capture saturation (predictions unbounded)

---

## 10. Modeling Recommendations Summary

### Must Do:
1. **Test 3 model classes minimum**:
   - Asymptotic exponential (interpretable, theoretically motivated)
   - Piecewise linear (best empirical fit)
   - Quadratic polynomial (simple, flexible)

2. **Use weakly informative priors** based on EDA insights (see Section 5)

3. **Run full validation pipeline** for each model:
   - Prior predictive checks
   - Simulation-based calibration
   - MCMC diagnostics (R-hat, ESS, trace plots)
   - Posterior predictive checks
   - LOO-CV

4. **Compare models** using ELPD with SE

### Nice to Have:
5. Log-log model (transforms to linear problem)
6. Student-t likelihood (robust alternative)
7. Gaussian Process (maximum flexibility)

### Don't Do:
- ✗ Simple linear regression (demonstrably inadequate, R²=0.52)
- ✗ Ignore saturation pattern
- ✗ Extrapolate far beyond x=31.5 without strong priors

---

## 11. Key Visualizations Reference

**Summary Figures** (single-page overviews):
- `eda/analyst_1/visualizations/00_comprehensive_summary.png`
- `eda/analyst_2/visualizations/00_SUMMARY_comprehensive.png`

**Relationship Structure**:
- `eda/analyst_1/visualizations/01_scatter_with_smoothers.png` - Multiple smoothing methods
- `eda/analyst_1/visualizations/03_segmented_relationship.png` - Regime analysis
- `eda/analyst_2/visualizations/10_segmentation_analysis.png` - Breakpoint evidence

**Model Comparison**:
- `eda/analyst_1/visualizations/05_model_comparison.png` - 5 models with metrics
- `eda/analyst_2/visualizations/04_all_functional_forms.png` - 7 models side-by-side
- `eda/analyst_2/visualizations/05_top_models_comparison.png` - Top 3 with residuals

**Diagnostics**:
- `eda/analyst_1/visualizations/04_residual_diagnostics.png` - 4-panel linear residuals
- `eda/analyst_2/visualizations/06_residual_comparison.png` - Top models residuals

**Transformations**:
- `eda/analyst_2/visualizations/11_transformations.png` - All transformations tested

---

## 12. Conclusion

This dataset provides **clear, robust evidence** for a nonlinear saturation relationship between x and Y. The data quality is excellent and suitable for Bayesian modeling.

**Key Takeaways**:
1. ✅ Nonlinear model is necessary (linear inadequate)
2. ✅ Saturation pattern is evident (rapid increase → plateau)
3. ✅ Multiple functional forms plausible (test 3+)
4. ✅ Informative priors possible from EDA
5. ✅ Validation strategy clear (PPCs, LOO, calibration)

**Next Phase**: Design and test Bayesian models incorporating saturation mechanisms. Expect best models to achieve R² > 0.85 with well-calibrated uncertainty.

---

## Appendix: Analyst Documentation

**Detailed Findings**:
- Analyst 1: `eda/analyst_1/findings.md`
- Analyst 2: `eda/analyst_2/findings.md`
- Synthesis: `eda/synthesis.md`

**Code**:
- Analyst 1: `eda/analyst_1/code/` (7 scripts)
- Analyst 2: `eda/analyst_2/code/` (9 scripts)

**All Visualizations**:
- Analyst 1: `eda/analyst_1/visualizations/` (8 plots)
- Analyst 2: `eda/analyst_2/visualizations/` (14 plots)
