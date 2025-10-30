# EDA Analyst 1: Distributional Characteristics and Variance Patterns

**Focus**: Distribution of success rates, variance structure, outlier detection, and sample size effects

**Analyst**: Analyst 1
**Dataset**: `/workspace/data/data_analyst_1.csv` (12 groups, 2814 total trials)
**Date**: 2025-10-30

---

## Quick Start

**Main Finding**: STRONG EVIDENCE for heterogeneity - groups have genuinely different success rates (variance ratio = 2.78, p < 0.001).

**Key Recommendation**: Use hierarchical models (beta-binomial, mixed-effects, or Bayesian) to account for between-group variance.

---

## Directory Structure

```
/workspace/eda/analyst_1/
├── README.md                    (this file)
├── findings.md                  (main report - START HERE)
├── eda_log.md                   (detailed exploration process)
├── code/                        (reproducible analysis scripts)
│   ├── 01_initial_exploration.py
│   ├── 02_distribution_analysis.py
│   ├── 03_sample_size_relationship.py
│   ├── 04_funnel_plot.py
│   ├── 05_variance_analysis.py
│   ├── 06_group_specific_analysis.py
│   ├── 07_hypothesis_testing.py
│   └── 08_summary_visualization.py
└── visualizations/              (all plots)
    ├── 00_summary_dashboard.png      (overview of all findings)
    ├── 01_distribution_overview.png
    ├── 02_sample_size_relationship.png
    ├── 03_funnel_plot.png
    ├── 04_variance_analysis.png
    └── 05_group_specific_analysis.png
```

---

## Key Files

### 1. Main Report
**File**: `findings.md`
**Description**: Comprehensive 11-section report with all findings, statistical tests, and modeling recommendations.

**Key Sections**:
- Executive Summary
- Distribution Analysis
- Evidence for Heterogeneity
- Outlier Analysis
- Sample Size Effects
- Hypothesis Testing
- Modeling Recommendations

### 2. Exploration Log
**File**: `eda_log.md`
**Description**: Detailed process log showing iterative exploration across 6 rounds of analysis.

### 3. Summary Dashboard
**File**: `visualizations/00_summary_dashboard.png`
**Description**: Single comprehensive figure showing all key findings at a glance.

---

## Key Findings

### 1. Heterogeneity is Real
- **Variance ratio**: 2.78 (empirical variance is 2.78x larger than expected)
- **Chi-square test**: χ² = 39.52, p = 0.000043 (strongly reject homogeneity)
- **64% of variance** is between-group differences (not sampling noise)

### 2. Two Extreme Outliers
- **Group 8**: 13.95% success rate (z = +4.03, p = 0.000057)
- **Group 4**: 4.20% success rate (z = -3.09, p = 0.002)
- Both have large sample sizes, so cannot be dismissed as noise

### 3. Not Explained by Sample Size
- Correlation between n_trials and success_rate: r = -0.34 (p = 0.28)
- Heterogeneity reflects true underlying differences, not measurement artifacts

### 4. Model Comparison
| Model | AIC | BIC | Verdict |
|-------|-----|-----|---------|
| Homogeneous (pooled) | 90.63 | 91.11 | REJECTED |
| Heterogeneous (group-specific) | 76.36 | 82.18 | BEST (Δ AIC = -14.3) |

---

## Visualizations Guide

### `00_summary_dashboard.png` - START HERE
Comprehensive overview showing:
- Distribution of success rates
- Funnel plot with outliers
- Variance decomposition pie chart
- Z-scores for all groups
- Model comparison table
- Statistical evidence summary
- Modeling recommendations

### `01_distribution_overview.png`
Six-panel analysis:
- Histogram of success rates
- Box plot with outliers
- Q-Q plot for normality
- Histogram of sample sizes
- Box plot of sample sizes
- Kernel density estimate

**Key insight**: Right-skewed distribution, 1 outlier (Group 8)

### `02_sample_size_relationship.png`
Two panels:
- Linear scale scatter plot with 95% and 99.7% CI bands
- Log scale version for clarity

**Key insight**: 3 groups outside 95% CI (Groups 2, 4, 8). No correlation with sample size.

### `03_funnel_plot.png`
Two funnel plot variants:
- Success rate vs precision (1/SE)
- Success rate vs √n

**Key insight**: 25% of groups outside funnel (5x excess), including 2 high-precision outliers.

### `04_variance_analysis.png`
Four-panel variance decomposition:
- Observed vs expected standard errors
- Standardized residuals vs sample size
- Expected variance vs observed squared deviations
- Q-Q plot of standardized residuals

**Key insight**: Variance ratio = 2.78, standardized residuals have SD = 1.87 (should be 1.0).

### `05_group_specific_analysis.png`
Four-panel group profiles:
- Z-scores by group (color-coded by severity)
- Success rates with confidence intervals
- Sample size vs deviation magnitude
- Distribution of z-scores vs N(0,1)

**Key insight**: Clear separation between extreme outliers, moderate outliers, and typical groups.

---

## Reproducibility

All analyses can be reproduced by running the Python scripts in order:

```bash
cd /workspace/eda/analyst_1/code

# Run all analyses
python 01_initial_exploration.py
python 02_distribution_analysis.py
python 03_sample_size_relationship.py
python 04_funnel_plot.py
python 05_variance_analysis.py
python 06_group_specific_analysis.py
python 07_hypothesis_testing.py
python 08_summary_visualization.py
```

**Dependencies**: pandas, numpy, matplotlib, seaborn, scipy

---

## Modeling Recommendations

### RECOMMENDED: Hierarchical Models

1. **Beta-Binomial Model**
   - Directly accounts for overdispersion
   - Estimates pooled rate and dispersion parameter
   - Good fit for heterogeneous binomial data

2. **Hierarchical Logistic Regression**
   - Group-specific random intercepts
   - Estimates population mean and between-group variance
   - Natural shrinkage for small-sample groups

3. **Bayesian Hierarchical Model**
   - Flexible prior specification
   - Full posterior distributions
   - Can incorporate covariates

### NOT RECOMMENDED

- **Simple pooled binomial**: Strongly rejected (p < 0.001)
- **Fixed-effects only**: Ignores hierarchical structure, overfits

---

## Statistical Tests Summary

| Test | Statistic | P-value | Conclusion |
|------|-----------|---------|------------|
| Chi-square (homogeneity) | χ² = 39.52 | p < 0.001 | Reject H0 |
| Shapiro-Wilk (normality) | W = 0.883 | p = 0.096 | Borderline |
| Likelihood ratio (H3 vs H1) | LR = 36.27 | p < 0.001 | Favor H3 |
| Pearson correlation (n vs rate) | r = -0.34 | p = 0.278 | Not significant |

---

## Group-Level Summary

| Group | n_trials | Success Rate | Z-Score | Status |
|-------|----------|--------------|---------|--------|
| 8 | 215 | 13.95% | +4.03 | **Extreme High** |
| 4 | 810 | 4.20% | -3.09 | **Extreme Low** |
| 2 | 148 | 12.84% | +2.81 | Moderate High |
| 1-12 (others) | 47-360 | 3.09%-12.77% | -1.50 to +1.56 | Typical |

**Pooled rate**: 6.97% (196/2814)

---

## Contact & Questions

For questions about this analysis or to request additional analyses, refer to:
- Full report: `findings.md`
- Detailed log: `eda_log.md`
- Code files: `code/*.py`

---

**End of README**
