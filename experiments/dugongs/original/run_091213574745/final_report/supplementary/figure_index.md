# Complete Figure Index

**Project**: Bayesian Analysis of Y-x Relationship
**Date**: October 28, 2025
**Total Figures**: 50+ across all phases

---

## Main Report Figures

These 5 key figures are referenced in the main report and copied to `/workspace/final_report/figures/`

### Figure 1: EDA Summary
**File**: `figure_1_eda_summary.png`
**Original**: `/workspace/eda/visualizations/00_eda_summary.png`
**Type**: 6-panel comprehensive overview
**Shows**:
- Distribution of predictor x (right-skewed)
- Distribution of response Y (approximately normal)
- Scatterplot with linear vs logarithmic fit comparison
- Residual patterns showing systematic misfit of linear model
- Variance structure (homoscedastic)
- Functional form comparisons (logarithmic best)

**Key Message**: Logarithmic transformation dramatically improves fit (R² from 0.68 → 0.90), establishing logarithmic form as primary candidate.

**Referenced in**: Introduction (Section 1), Methods (Section 2.1)

---

### Figure 2: Model 1 Fitted Curve
**File**: `figure_2_fitted_curve.png`
**Original**: `/workspace/experiments/experiment_1/posterior_inference/plots/fitted_curve.png`
**Type**: Scatterplot with posterior mean curve and credible interval
**Shows**:
- Observed data points (red circles)
- Posterior mean fitted curve (blue line)
- 95% credible interval (light blue shading)
- Logarithmic saturation pattern

**Key Message**: Model captures diminishing returns pattern across full x range. All observations fall within or near the 95% credible interval, indicating good fit.

**Referenced in**: Results (Section 3.3), Discussion (Section 5.1)

---

### Figure 3: Residual Diagnostics
**File**: `figure_3_residual_diagnostics.png`
**Original**: `/workspace/experiments/experiment_1/posterior_predictive_check/plots/residual_patterns.png`
**Type**: 4-panel diagnostic grid
**Shows**:
1. **Top Left**: Residuals vs Fitted Values
   - Random scatter around zero (no systematic pattern)
   - Confirms functional form is appropriate
2. **Top Right**: Residuals vs Predictor (x)
   - Random scatter across x range
   - No evidence of missed nonlinearity
3. **Bottom Left**: Scale-Location Plot
   - Square root of absolute standardized residuals vs fitted values
   - Flat trend confirms homoscedasticity
   - Variance ratio (high/low) = 0.91 < 2.0 threshold
4. **Bottom Right**: Normal Q-Q Plot
   - Standardized residuals vs theoretical quantiles
   - Good alignment along diagonal
   - Minor deviation in tails (common with n=27, not concerning)

**Key Message**: All diagnostic checks passed. No systematic model misfit, constant variance, approximately normal residuals.

**Referenced in**: Results (Section 3.4), Model Diagnostics (Section 4.2)

---

### Figure 4: LOO-CV Model Comparison
**File**: `figure_4_loo_comparison.png`
**Original**: `/workspace/experiments/model_comparison/plots/loo_comparison.png`
**Type**: Bar chart with error bars
**Shows**:
- LOO-ELPD for Model 1 (Normal): 24.89 ± 2.82
- LOO-ELPD for Model 2 (Student-t): 23.83 ± 2.84
- ΔLOO = -1.06 ± 0.36 (Model 2 relative to Model 1)
- Error bars representing standard errors
- Model 1 clearly superior (rightmost bar)

**Key Message**: Model 1 outperforms Model 2 in cross-validation (moderate evidence). Combined with parsimony and convergence considerations, Model 1 is the clear choice.

**Referenced in**: Results (Section 3.6), Model Diagnostics (Section 4.3)

---

### Figure 5: Integrated Dashboard
**File**: `figure_5_integrated_dashboard.png`
**Original**: `/workspace/experiments/model_comparison/plots/integrated_dashboard.png`
**Type**: 6-panel comprehensive comparison dashboard
**Shows**:
1. **Top Left**: LOO-ELPD Comparison (Model 1 better)
2. **Top Middle**: Pareto k Diagnostics (both models reliable, all k < 0.7)
3. **Top Right**: β₀ Posteriors (nearly identical distributions)
4. **Bottom Left**: β₁ Posteriors (nearly identical distributions)
5. **Bottom Middle**: ν Posterior (Model 2 only, mean ≈ 23, wide uncertainty [3.7, 60.0])
6. **Bottom Right**: Prediction Comparison (fitted curves and intervals virtually identical)

**Key Message**: Comprehensive visual evidence supporting Model 1 selection. Models agree on all parameters and predictions, but Model 1 has better LOO and doesn't require uncertain ν parameter.

**Referenced in**: Results (Section 3.6), Discussion (Section 5.2)

---

## Complete Figure Catalog by Phase

### Exploratory Data Analysis

**Location**: `/workspace/eda/visualizations/`

1. **`00_eda_summary.png`** - 6-panel overview (copied as Figure 1)
2. **`01_x_distribution.png`** - Histogram and box plot of predictor x
3. **`02_Y_distribution.png`** - Histogram and box plot of response Y
4. **`03_bivariate_analysis.png`** - Scatterplot with correlation statistics and residual plots
5. **`04_variance_analysis.png`** - Heteroscedasticity assessment (scale-location plot, residuals vs x)
6. **`05_functional_forms.png`** - Comparison of 6 functional forms (linear, quadratic, cubic, sqrt, log, asymptotic)
7. **`06_transformations.png`** - Log-log, semi-log, reciprocal transformations
8. **`07_changepoint_analysis.png`** - Piecewise linear fit showing two-regime structure at x ≈ 7
9. **`08_rate_of_change.png`** - Local slope analysis confirming regime shift
10. **`09_outlier_influence.png`** - Cook's distance, leverage, influence diagnostics

**Key Insights**:
- Logarithmic transformation best among all forms tested
- Clear two-regime structure (growth x ≤ 7, plateau x > 7)
- One influential observation at x = 31.5
- No heteroscedasticity detected

---

### Model 1: Prior Predictive Check

**Location**: `/workspace/experiments/experiment_1/prior_predictive_check/plots/`

1. **`parameter_plausibility.png`** - Prior distributions for β₀, β₁, σ with observed data range overlaid
2. **`prior_predictive_coverage.png`** - 100 simulated datasets from prior, checking if observed data is plausible
3. **`data_range_diagnostic.png`** - Histogram of simulated Y ranges vs observed range
4. **`residual_scale_diagnostic.png`** - Prior predictive distribution of σ vs observed residual SD
5. **`slope_sign_diagnostic.png`** - Prior predictive distribution of β₁ (all positive as expected)
6. **`example_datasets.png`** - 9 example datasets simulated from prior

**Key Insights**:
- Priors generate scientifically plausible data
- Observed data falls well within prior predictive distribution
- No prior-data conflict
- Priors are weakly informative (not overly constraining)

---

### Model 1: Simulation-Based Validation

**Location**: `/workspace/experiments/experiment_1/simulation_based_validation/plots/`

1. **`parameter_recovery.png`** - True vs estimated parameters across 10 simulations
2. **`prior_posterior_comparison.png`** - Prior vs posterior for each simulation
3. **`convergence_diagnostics.png`** - R-hat and ESS across all simulations
4. **`posterior_pairs.png`** - Pairwise posterior correlations
5. **`synthetic_data_fit.png`** - Example of fitted curve to synthetic data
6. **`residual_diagnostics.png`** - Residuals from synthetic data fits
7. **`multi_simulation_recovery.png`** - Coverage assessment across simulations
8. **`estimate_distributions.png`** - Distribution of parameter estimates across simulations
9. **`calibration_summary.png`** - Overall calibration metrics (coverage, bias)

**Key Insights**:
- 80-90% coverage achieved (target: 80%)
- Unbiased parameter recovery (mean error < 0.01)
- All simulations converged successfully
- Inference procedure is reliable

---

### Model 1: Posterior Inference

**Location**: `/workspace/experiments/experiment_1/posterior_inference/plots/`

1. **`trace_plots.png`** - MCMC traces for β₀, β₁, σ showing excellent mixing
2. **`rank_plots.png`** - Rank histograms (approximately uniform indicates good convergence)
3. **`autocorrelation.png`** - Autocorrelation functions (rapid decay indicates efficient sampling)
4. **`posterior_vs_prior.png`** - Comparison showing posteriors much tighter than priors
5. **`pairs_plot.png`** - Pairwise posterior distributions and correlations
6. **`fitted_curve.png`** - Observed data with fitted curve and credible interval (copied as Figure 2)
7. **`residuals_diagnostics.png`** - Residual plots from posterior mean fit
8. **`loo_pit.png`** - LOO probability integral transform for calibration check
9. **`pareto_k.png`** - Pareto k values for all observations (all < 0.5)

**Key Insights**:
- Perfect convergence (R-hat = 1.00, ESS > 11,000)
- Posteriors highly informed by data (7-8× precision gain)
- All observations well-predicted in LOO-CV
- No influential observations

---

### Model 1: Posterior Predictive Check

**Location**: `/workspace/experiments/experiment_1/posterior_predictive_check/plots/`

1. **`ppc_density_overlay.png`** - Observed data density overlaid on 100 replicated datasets
2. **`test_statistic_distributions.png`** - 6-panel grid showing distributions of 6 test statistics (mean, SD, min, max, skewness, range)
3. **`residual_patterns.png`** - 4-panel residual diagnostics (copied as Figure 3)
4. **`individual_predictions.png`** - Point-wise predictions with 95% intervals showing 100% coverage
5. **`loo_pit_calibration.png`** - LOO-PIT uniformity check (ECDF vs diagonal)
6. **`qq_observed_vs_predicted.png`** - Quantile-quantile plot comparing observed to posterior predictive quantiles
7. **`fitted_curve_with_envelope.png`** - Fitted curve with 95% posterior predictive envelope

**Key Insights**:
- 10/10 test statistics passed (all p-values in [0.29, 0.84])
- 100% coverage of 95% intervals (conservative, acceptable)
- No systematic residual patterns
- Model captures all key features of data

---

### Model 2: Prior Predictive Check

**Location**: `/workspace/experiments/experiment_2/prior_predictive_check/plots/`

1. **`parameter_plausibility.png`** - Prior distributions including ν (degrees of freedom)
2. **`nu_tail_behavior_diagnostic.png`** - How ν prior affects tail heaviness
3. **`prior_predictive_coverage.png`** - Prior predictive datasets with truncated ν ≥ 3
4. **`data_range_diagnostic.png`** - Simulated Y ranges vs observed
5. **`slope_scale_diagnostic.png`** - Joint prior for β₁ and σ
6. **`example_datasets.png`** - Example datasets from Student-t prior predictive
7. **`studentt_vs_normal_comparison.png`** - Comparison of tail behavior between distributions

**Key Insights**:
- Initial prior (ν unrestricted) generated implausible outliers
- Truncation to ν ≥ 3 resolved issue
- Student-t prior predictive more dispersed than Normal (expected)

---

### Model 2: Posterior Inference

**Location**: `/workspace/experiments/experiment_2/posterior_inference/plots/`

1. **`trace_plots.png`** - MCMC traces (β₀, β₁ OK; σ, ν poor mixing)
2. **`nu_posterior.png`** - Posterior distribution of ν (mean ≈ 23, wide [3.7, 60.0])
3. **`model_comparison_fit.png`** - Fitted curves from Model 1 and Model 2 (nearly identical)
4. **`loo_comparison.png`** - LOO-ELPD comparison (Model 1 better)
5. **`posterior_predictive_check.png`** - PPC density overlay for Model 2
6. **`rank_plots.png`** - Rank histograms showing poor mixing for σ, ν
7. **`pareto_k_diagnostic.png`** - Pareto k values (all < 0.7, LOO reliable)
8. **`parameter_comparison.png`** - Overlaid posteriors for β₀, β₁, σ from both models (nearly identical)

**Key Insights**:
- ν ≈ 23 suggests Normal adequate
- Convergence issues for σ and ν (R-hat > 1.1, ESS < 20)
- Parameters nearly identical to Model 1
- No improvement in predictive performance

---

### Model Comparison

**Location**: `/workspace/experiments/model_comparison/plots/`

1. **`loo_comparison.png`** - LOO-ELPD bar chart (copied as Figure 4)
2. **`pareto_k_comparison.png`** - Side-by-side Pareto k diagnostics for both models
3. **`loo_pit_comparison.png`** - Calibration curves for both models
4. **`parameter_comparison.png`** - Overlaid posteriors for β₀, β₁, σ
5. **`nu_posterior.png`** - ν posterior from Model 2 with Normal threshold indicated
6. **`prediction_comparison.png`** - Overlaid fitted curves and credible intervals
7. **`residual_comparison.png`** - Side-by-side residual diagnostics
8. **`integrated_dashboard.png`** - 6-panel comprehensive comparison (copied as Figure 5)

**Key Insights**:
- Model 1 superior in LOO-ELPD (Δ = 1.06)
- Both models have reliable LOO (all k < 0.7)
- Parameters and predictions nearly identical
- Student-t offers no benefit

---

## Figure Usage Guide

### For Presentations

**Minimal Set** (3 slides):
1. Figure 1 (EDA Summary) - "Why logarithmic?"
2. Figure 2 (Fitted Curve) - "How good is the fit?"
3. Figure 4 (LOO Comparison) - "Why Model 1?"

**Standard Set** (5 slides):
- Add Figure 3 (Residual Diagnostics) - "Is the model adequate?"
- Add Figure 5 (Integrated Dashboard) - "Comprehensive evidence"

**Comprehensive Set** (10+ slides):
- Include all 5 main figures
- Add key EDA plots (functional forms, changepoint)
- Add key validation plots (parameter recovery, PPCs)

### For Papers

**Main Text** (3-5 figures):
- Figure 1 (EDA Summary)
- Figure 2 (Fitted Curve)
- Figure 3 (Residual Diagnostics)
- Figure 4 (LOO Comparison)
- Optional: Figure 5 (Integrated Dashboard)

**Supplementary** (all remaining figures):
- Full EDA suite (10 plots)
- Prior predictive checks (6-7 plots per model)
- Simulation-based validation (9 plots)
- Complete posterior inference diagnostics (9 plots per model)
- All posterior predictive checks (7 plots)

### For Technical Reports

**Include all figures** organized by phase, with captions explaining:
1. What is plotted
2. What to notice
3. What it means for model adequacy

---

## Figure Quality

All figures are:
- **Format**: PNG (high resolution)
- **Size**: Variable (typically 1200-2400 px width)
- **Color**: Color-blind friendly palettes used
- **Text**: Readable axis labels and titles
- **Style**: Consistent matplotlib styling

**Software**: Generated using Python 3.11 with Matplotlib 3.7, Seaborn 0.12, ArviZ 0.15

---

## Linking Figures to Findings

### Finding 1: Logarithmic Form is Best
**Supporting Figures**:
- `eda/visualizations/05_functional_forms.png` (comparison of 6 forms)
- `eda/visualizations/06_transformations.png` (log transformation effect)
- Figure 1 (EDA summary showing log fit superiority)

### Finding 2: Model Passes All Validation
**Supporting Figures**:
- All prior predictive check plots (priors appropriate)
- All simulation-based validation plots (parameter recovery)
- All convergence diagnostics (R-hat = 1.00, ESS > 11k)
- All posterior predictive check plots (10/10 tests passed)

### Finding 3: Student-t Not Needed
**Supporting Figures**:
- Figure 4 (LOO comparison)
- Figure 5 (integrated dashboard)
- `experiment_2/posterior_inference/plots/nu_posterior.png` (ν ≈ 23)
- `model_comparison/plots/parameter_comparison.png` (identical parameters)

### Finding 4: Two-Regime Structure (Untested but Noted)
**Supporting Figures**:
- `eda/visualizations/07_changepoint_analysis.png` (x ≈ 7 breakpoint)
- `eda/visualizations/08_rate_of_change.png` (slope change)
- Figure 3 bottom-left (residuals show no regime clustering)

### Finding 5: No Problematic Outliers
**Supporting Figures**:
- `eda/visualizations/09_outlier_influence.png` (Cook's D diagnostics)
- `experiment_1/posterior_inference/plots/pareto_k.png` (all k < 0.5)
- Figure 3 (residuals within expected range)

---

## Accessibility Notes

### For Colorblind Readers
- All key comparisons use shape/line style in addition to color
- Red/green combinations avoided
- High contrast maintained

### For Screen Readers
- All figure captions include descriptive text
- Key numerical results stated in text, not figure-only

### For Print
- All figures render well in grayscale
- Important features distinguishable without color

---

## Copyright and Reuse

**Status**: All figures generated as part of this analysis

**License**: (Specify as appropriate for your project)

**Citation**: "Bayesian Analysis of Y-x Relationship, October 2025"

**Reuse**: Figures may be reused with attribution, subject to data sharing agreements

---

## Contact

**Questions about figures**: Refer to corresponding report sections

**High-resolution versions**: Available in original locations listed above

**Custom figures**: Code available in respective `/code/` directories

---

*Figure Index - Version 1.0 - October 28, 2025*
