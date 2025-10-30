# EDA Analyst 1 - Distributional Characteristics and Sample Size Effects

## Overview
This directory contains the complete exploratory data analysis focused on:
- Distribution of success rates
- Sample size effects
- Overdispersion assessment
- Outlier detection

**Analyst:** EDA Analyst 1
**Date:** 2025-10-30
**Data Source:** `/workspace/data/data_analyst_1.csv`

---

## Key Findings (Executive Summary)

### Main Discovery: STRONG OVERDISPERSION
- **Dispersion parameter (φ) = 3.505** (should be 1.0 under binomial)
- **250% more variance** than expected under simple binomial model
- **68.8% of total variance** is between-group heterogeneity
- **Chi-squared test: p < 0.0001** (strong rejection of homogeneity)

### Modeling Recommendation
**DO NOT use standard binomial GLM**. Use hierarchical models:
1. **RECOMMENDED:** Beta-binomial model
2. **ALTERNATIVE:** GLMM with random intercepts
3. **SIMPLE:** Quasi-binomial GLM

### Outliers Identified
- **5 of 12 groups (41.7%)** flagged as outliers
- **Group 8:** Most extreme (z = 3.94, success_rate = 0.144)
- **Group 1:** Zero successes in 47 trials (needs verification)

---

## Directory Structure

```
eda/analyst_1/
├── README.md                          # This file
├── findings.md                        # Executive summary report
├── eda_log.md                         # Detailed exploration log
├── summary_statistics.csv             # Comprehensive statistics table
├── code/                              # Reproducible analysis scripts
│   ├── 01_initial_exploration.py
│   ├── 02_distribution_analysis.py
│   ├── 03_sample_size_effects.py
│   ├── 04_overdispersion_analysis.py
│   ├── 05_outlier_detection.py
│   └── 06_summary_table.py
└── visualizations/                    # All plots (PNG, 300 DPI)
    ├── success_rate_distribution.png
    ├── sample_size_distribution.png
    ├── sample_size_effects.png
    ├── funnel_plot.png
    └── outlier_detection.png
```

---

## Files Guide

### Reports (Start Here)

1. **`findings.md`** - Main report with all results
   - Executive summary
   - Statistical tests
   - Visualizations interpretation
   - Modeling recommendations
   - ~7,000 words, comprehensive

2. **`eda_log.md`** - Detailed exploration process
   - Step-by-step analysis
   - Hypothesis testing
   - Alternative explanations
   - Questions for further investigation

3. **`summary_statistics.csv`** - Data table
   - All 12 groups with statistics
   - Success rates with 95% Wilson CIs
   - Standardized residuals (z-scores)
   - Sortable/filterable

### Visualizations

All saved as high-resolution PNG (300 DPI) in `/visualizations/`

1. **`success_rate_distribution.png`** (4 panels)
   - Histogram of success rates
   - Box plot
   - Q-Q plot for normality
   - Kernel density estimate
   - **Shows:** Approximate normality of success rates

2. **`sample_size_distribution.png`** (2 panels)
   - Histogram of sample sizes
   - Bar chart by group
   - **Shows:** Wide range (47 to 810 trials)

3. **`sample_size_effects.png`** (2 panels)
   - Scatter: success_rate vs n_trials with CI bands
   - Residuals vs sample size
   - **Shows:** No correlation (r ≈ 0), violation of funnel pattern

4. **`funnel_plot.png`** (2 panels) ⭐ KEY DIAGNOSTIC
   - Standard funnel plot
   - Precision-based funnel plot
   - **Shows:** Overdispersion, 5 outliers outside 95% limits

5. **`outlier_detection.png`** (6 panels)
   - Standardized residuals by group
   - Box plot with outliers
   - Deviance residuals
   - Cook's distance
   - Q-Q plot with outliers highlighted
   - Outliers in spatial context
   - **Shows:** 5 outliers identified by multiple methods

### Code (Fully Reproducible)

All Python scripts in `/code/` directory, numbered in execution order:

1. **`01_initial_exploration.py`**
   - Load data, check structure
   - Summary statistics
   - Missing value check

2. **`02_distribution_analysis.py`**
   - Distribution plots (histogram, KDE, box plot, Q-Q)
   - Normality tests (Shapiro-Wilk, Anderson-Darling)
   - Skewness and kurtosis

3. **`03_sample_size_effects.py`**
   - Correlation tests (Pearson, Spearman)
   - Variance stratification by sample size
   - Scatter plots with confidence bands
   - Residual analysis

4. **`04_overdispersion_analysis.py`** ⭐ KEY ANALYSIS
   - Chi-squared test for homogeneity
   - Dispersion parameter (φ)
   - Variance component decomposition
   - Quasi-likelihood dispersion
   - Funnel plots

5. **`05_outlier_detection.py`**
   - Z-score method
   - IQR method
   - Modified z-score (MAD-based)
   - Deviance residuals
   - Cook's distance
   - Multi-panel diagnostics

6. **`06_summary_table.py`**
   - Comprehensive statistics table
   - Wilson confidence intervals
   - Overdispersion metrics
   - Export to CSV

**To reproduce:** Run scripts 01-06 in order from `/workspace/`

---

## Key Statistics at a Glance

| Metric | Value |
|--------|-------|
| **Data** | |
| Number of groups | 12 |
| Total trials | 2,814 |
| Total successes | 208 |
| Pooled success rate | 0.0739 (7.39%) |
| **Distribution** | |
| Mean success rate | 0.0737 |
| Median success rate | 0.0669 |
| Std deviation | 0.0384 |
| Range | 0.000 to 0.144 |
| CV | 0.521 (high variability) |
| **Overdispersion** | |
| Dispersion parameter (φ) | **3.505** ⚠️ |
| Variance ratio | **3.21** ⚠️ |
| Chi-squared p-value | **<0.0001** ⚠️ |
| % variance from overdispersion | **68.8%** ⚠️ |
| **Outliers** | |
| Groups with \|z\| > 1.96 | 4 (33%) |
| Groups with \|z\| > 2.576 | 1 (8%) |
| Most extreme outlier | Group 8 (z=3.94) |

---

## Interpretation Summary

### What We Learned

1. **Groups are heterogeneous** - they don't all have the same underlying success rate
2. **Simple binomial model inadequate** - standard GLM will give wrong inference
3. **Hierarchical model needed** - beta-binomial or GLMM recommended
4. **Sample size doesn't bias results** - no systematic relationship
5. **Several clear outliers** - especially Group 8 (high) and Group 1 (zero rate)

### What This Means for Modeling

**DON'T:**
- Fit `glm(y ~ 1, family=binomial)` - assumes φ=1 (observed φ=3.5!)
- Use standard errors from binomial GLM - they're ~2x too small
- Ignore heterogeneity - it explains 69% of variance

**DO:**
- Fit beta-binomial or GLMM with random intercepts
- Account for overdispersion in all inference
- Consider group-specific estimates with shrinkage
- Investigate causes of heterogeneity (covariates?)

### Questions Raised

1. **Why is Group 1 zero?** - Data error or genuine difference?
2. **Why is Group 8 so high?** - Different conditions? Population?
3. **What explains heterogeneity?** - Need group-level covariates
4. **Is this temporal or spatial?** - Unknown from current data

---

## Modeling Next Steps

### Immediate (Required)
1. **Verify Group 1 data** - 0/47 is unusual
2. **Fit beta-binomial model** - quantify heterogeneity
3. **Compare models** - beta-binomial vs GLMM vs quasi-binomial
4. **Posterior predictive check** - validate chosen model

### Further Analysis
1. **Collect covariates** - group characteristics that might explain variation
2. **Shrinkage estimation** - obtain empirical Bayes estimates
3. **Sensitivity analysis** - impact of outliers on inference
4. **Prediction intervals** - for new groups

### Recommended Software

**R:**
```r
# Beta-binomial
library(VGAM)
fit <- vglm(cbind(r, n-r) ~ 1, family=betabinomial, data=df)

# GLMM
library(lme4)
fit <- glmer(cbind(r, n-r) ~ (1|group), family=binomial, data=df)

# Quasi-binomial
fit <- glm(cbind(r, n-r) ~ 1, family=quasibinomial, data=df)
```

**Python:**
```python
# Beta-binomial (Bayesian)
import pymc as pm
with pm.Model() as model:
    alpha = pm.HalfNormal('alpha', sigma=10)
    beta = pm.HalfNormal('beta', sigma=10)
    p = pm.Beta('p', alpha=alpha, beta=beta, shape=n_groups)
    y = pm.Binomial('y', n=n_trials, p=p, observed=r_successes)
```

---

## Contact & Reproducibility

**All paths are absolute from `/workspace/`**

- Data: `/workspace/data/data_analyst_1.csv`
- Code: `/workspace/eda/analyst_1/code/*.py`
- Output: `/workspace/eda/analyst_1/`

**Dependencies:**
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- pathlib (standard library)

**Python version:** 3.x compatible

**Random seeds:** Not applicable (deterministic analysis)

---

## Quick Start

### For Data Scientists
1. Read **`findings.md`** (executive summary)
2. Look at **`funnel_plot.png`** (key diagnostic)
3. Check **`summary_statistics.csv`** (all group stats)
4. Fit beta-binomial model with your preferred software

### For Statisticians
1. Read **`eda_log.md`** (detailed methodology)
2. Review all visualizations in `/visualizations/`
3. Examine code in `/code/` for reproducibility
4. Check dispersion parameter calculation in `04_overdispersion_analysis.py`

### For Collaborators
1. Note Group 1 data quality issue (zero successes)
2. Review outlier list in **`findings.md`** section 5
3. Consider collecting group-level covariates
4. Discuss modeling approach (beta-binomial recommended)

---

## Citation

If using this analysis, please cite:

> EDA Analyst 1. (2025). Distributional Characteristics and Sample Size Effects Analysis.
> Exploratory Data Analysis Report. /workspace/eda/analyst_1/

---

**Analysis complete:** 2025-10-30
**Status:** Ready for modeling
**Recommendation:** Beta-binomial or GLMM with random intercepts
