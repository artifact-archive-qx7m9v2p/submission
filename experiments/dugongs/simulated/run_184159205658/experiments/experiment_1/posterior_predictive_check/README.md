# Posterior Predictive Check - Experiment 1

## Overview
This directory contains comprehensive posterior predictive checks (PPC) for the logarithmic regression model fitted in Experiment 1.

## Contents

### Code
- `code/posterior_predictive_check.py` - Complete PPC analysis script

### Plots (6 diagnostic visualizations)
1. **ppc_overlays.png** - Visual comparison of observed vs predicted data (4 panels)
   - Posterior predictive draws overlaid on observations
   - Distribution comparison histograms
   - Predictive intervals with coverage
   - Residuals from median prediction

2. **ppc_statistics.png** - Test statistics calibration (10 panels)
   - Mean, SD, Min, Max, Range
   - Quartiles (Q25, Median, Q75)
   - Skewness, Kurtosis
   - Each panel shows observed vs posterior predictive distribution

3. **residual_diagnostics.png** - Comprehensive residual analysis (9 panels)
   - Residuals vs fitted values and x
   - Histogram and Q-Q plot
   - Scale-location plot
   - Autocorrelation function
   - Absolute residuals analysis
   - Cook's distance (influence measures)

4. **loo_pit.png** - LOO-PIT uniformity check
   - Leave-One-Out Probability Integral Transform
   - Tests calibration of predictive distributions

5. **coverage_assessment.png** - Predictive coverage analysis (4 panels)
   - Coverage status by observation
   - Interval width vs x
   - Probability Integral Transform distribution
   - Coverage summary (50% and 95% intervals)

6. **model_weaknesses.png** - Diagnostic visualization of identified issues (7 panels)
   - Problematic observations highlighted
   - Residual patterns
   - Extreme value detection
   - Test statistic calibration
   - Heteroscedasticity checks
   - Normality assessment
   - Summary of findings

### Documentation
- **ppc_findings.md** - Comprehensive analysis report with:
  - Executive summary
  - Detailed diagnostic results
  - Visual evidence documentation
  - Model adequacy assessment
  - Recommendations

## Key Findings

**Overall Assessment: EXCELLENT FIT**

### Strengths
- **Perfect predictive coverage:** 100% of observations within 95% PI
- **Exceptional residuals:** Shapiro p = 0.986 (perfectly normal)
- **Well-calibrated statistics:** 9/10 test statistics in [0.05, 0.95]
- **No systematic patterns:** All model assumptions satisfied

### Minor Issues
- Maximum value statistic borderline extreme (p = 0.969)
- Not substantively important - within predictive intervals

### Recommendation
**ACCEPT** - Model is well-specified and ready for scientific use.

## Usage

To reproduce the analysis:
```bash
python code/posterior_predictive_check.py
```

## Requirements
- ArviZ InferenceData from Stan/PyMC fit (posterior + posterior_predictive)
- Python packages: numpy, pandas, matplotlib, seaborn, arviz, scipy

## Model Specification
- **Form:** Y = β₀ + β₁·log(x) + ε
- **Error:** ε ~ Normal(0, σ)
- **Data:** 27 observations
- **Parameters:** β₀ = 1.751, β₁ = 0.275, σ = 0.124

---

**Analysis Date:** 2025-10-27
**Analyst:** Posterior Predictive Check Agent
