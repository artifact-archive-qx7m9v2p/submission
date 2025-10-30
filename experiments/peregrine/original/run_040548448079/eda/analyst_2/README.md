# EDA Analyst 2: Distributional Properties & Variance Structure

## Overview
This directory contains a comprehensive analysis of the distributional properties and variance structure of time-ordered count data (n=40 observations).

**Analyst Focus**: Count distribution, Poisson vs Negative Binomial, overdispersion, variance-mean relationship, outlier detection

---

## Quick Start: Key Findings

### Main Result
**Use Negative Binomial distribution (NOT Poisson)**
- Variance/Mean ratio: **67.99** (Poisson assumes 1.0)
- ΔAIC: **-2417** (overwhelming evidence for NB)
- Dispersion parameter: r = 1.634 (α = 0.612)

### Data Quality
- **Excellent**: No missing values, no outliers
- All 40 observations should be used

### Variance Structure
- **Heteroscedastic**: Variance changes over time
- **Non-linear**: Variance ∝ Mean^1.67 (not linear)
- **Time-varying dispersion**: 1.27 to 7.52 across periods

---

## Directory Structure

```
/workspace/eda/analyst_2/
├── README.md                    # This file
├── findings.md                  # Main comprehensive report (READ THIS FIRST)
├── eda_log.md                   # Detailed exploration log
├── code/                        # Reproducible analysis scripts
│   ├── 00_summary_statistics.py      # Quick reference summary
│   ├── 01_initial_exploration.py     # Basic statistics
│   ├── 02_distribution_plots.py      # Distribution visualizations
│   ├── 03_variance_mean_analysis.py  # Variance-mean relationship
│   ├── 04_distribution_fitting.py    # Poisson vs NB comparison
│   ├── 05_outlier_analysis.py        # Outlier detection
│   └── 06_dispersion_temporal.py     # Temporal dispersion patterns
└── visualizations/              # All plots (7 files)
    ├── distribution_overview.png
    ├── count_histogram.png
    ├── variance_mean_analysis.png
    ├── distribution_fitting.png
    ├── outlier_analysis.png
    ├── temporal_dispersion_rolling.png
    └── temporal_periods_comparison.png
```

---

## File Guide

### Reports (Start Here)

1. **findings.md** (MAIN REPORT)
   - Comprehensive 13-section report
   - All evidence, recommendations, and implications
   - Reference for modeling decisions
   - ~100 pages of detailed analysis

2. **eda_log.md** (DETAILED PROCESS)
   - Hypothesis testing workflow
   - Iterative exploration process
   - Alternative hypotheses considered
   - Discovery timeline

### Code (Reproducible)

All scripts are standalone and can be run independently:

```bash
# Quick summary
python /workspace/eda/analyst_2/code/00_summary_statistics.py

# Full analysis pipeline
python /workspace/eda/analyst_2/code/01_initial_exploration.py
python /workspace/eda/analyst_2/code/02_distribution_plots.py
python /workspace/eda/analyst_2/code/03_variance_mean_analysis.py
python /workspace/eda/analyst_2/code/04_distribution_fitting.py
python /workspace/eda/analyst_2/code/05_outlier_analysis.py
python /workspace/eda/analyst_2/code/06_dispersion_temporal.py
```

### Visualizations (7 Figures)

| File | Description | Key Insight |
|------|-------------|-------------|
| `distribution_overview.png` | 4-panel distribution summary | Overall shape, skewness, temporal context |
| `count_histogram.png` | Detailed frequency histogram | Right-skewed, platykurtic distribution |
| `variance_mean_analysis.png` | Variance-mean relationship | Power law: Var ∝ Mean^1.67 |
| `distribution_fitting.png` | Poisson vs NB comparison | NB fits well, Poisson fails |
| `outlier_analysis.png` | 6-panel outlier diagnostics | No outliers detected |
| `temporal_dispersion_rolling.png` | Rolling window dispersion | Time-varying dispersion patterns |
| `temporal_periods_comparison.png` | Period-by-period analysis | 6-fold variation in dispersion |

---

## Key Numbers (Quick Reference)

### Distributional Properties
- **Mean**: 109.45
- **Variance**: 7441.74
- **Variance/Mean**: 67.99
- **Skewness**: 0.602
- **Kurtosis**: -1.233
- **Range**: [19, 272]

### Model Comparison
- **Poisson AIC**: 2872.13
- **NB AIC**: 455.51
- **ΔAIC**: -2416.62
- **Δ Log-likelihood**: +1209.31

### Dispersion Parameters
- **r (size)**: 1.634
- **α (overdispersion)**: 0.612
- **Power law exponent**: 1.667

### Data Quality
- **Missing values**: 0
- **Outliers**: 0
- **Influential points**: 3 (legitimate)

---

## Modeling Recommendations

### PRIMARY MODEL
```
Y_t ~ NegBinomial(μ_t, r)
log(μ_t) = β₀ + β₁ × year_t
r ≈ 1.6
```

### ALTERNATIVE (if needed)
```
Y_t ~ NegBinomial(μ_t, r_t)
log(μ_t) = β₀ + β₁ × year_t
log(r_t) = γ₀ + γ₁ × year_t
```

### DO NOT USE
- ❌ Poisson regression (fundamentally wrong)
- ❌ Zero-inflated models (no zeros in data)

---

## Analysis Workflow

This analysis followed a systematic hypothesis-testing approach:

### Round 1: Distribution Family
- **Hypothesis**: Poisson is appropriate
- **Test**: Variance/mean ratio
- **Result**: REJECTED (ratio = 68, not 1)
- **Conclusion**: Need overdispersed model

### Round 2: Variance-Mean Relationship
- **Hypothesis**: Variance scales linearly with mean
- **Test**: Power law regression
- **Result**: REJECTED (exponent = 1.67, not 1.0)
- **Conclusion**: Non-linear variance structure

### Round 3: Temporal Stability
- **Hypothesis**: Constant dispersion over time
- **Test**: Period-specific variance/mean ratios
- **Result**: REJECTED (range 1.27 to 7.52)
- **Conclusion**: Heteroscedastic, time-varying

### Round 4: Data Quality
- **Hypothesis**: Outliers drive overdispersion
- **Test**: Multiple outlier detection methods
- **Result**: REJECTED (0 outliers found)
- **Conclusion**: Overdispersion is genuine

---

## Integration with Other Analysts

### Questions for Collaborators

**Time Series Analyst**:
- Is there autocorrelation in residuals?
- Could temporal dependence explain some overdispersion?
- Should we model AR errors?

**Covariate Analyst**:
- Are there unmeasured variables causing overdispersion?
- Could covariates reduce dispersion parameter?
- What α do you observe after including predictors?

**Model Specialist**:
- NB1 vs NB2 parameterization preference?
- Cross-validation strategy given n=40?
- Prior sensitivity for Bayesian approach?

### Key Contributions to Team

1. **Definitive distribution family**: Negative Binomial
2. **Quantified overdispersion**: α = 0.612
3. **Identified heteroscedasticity**: Time-varying dispersion
4. **Validated data quality**: No issues, use all 40 obs
5. **Variance-mean relationship**: Power law with exponent 1.67

---

## Technical Notes

### Software Requirements
```python
pandas
numpy
scipy
matplotlib
seaborn
```

### Computation Time
- Total runtime: ~2 minutes
- All analyses run on standard laptop
- No special hardware needed

### Reproducibility
- Fixed random seed: None needed (no simulation)
- Deterministic analyses only
- Results exactly reproducible

---

## Citation

If using these findings, reference:
- **Dataset**: data/data_analyst_2.csv (n=40)
- **Analysis date**: 2025-10-29
- **Analyst**: EDA Specialist 2 (Distributional Properties)
- **Output**: /workspace/eda/analyst_2/

---

## Contact / Questions

For questions about:
- **Distribution family choice**: See findings.md Section 1
- **Variance structure**: See findings.md Section 2
- **Outliers**: See findings.md Section 4
- **Model recommendations**: See findings.md Section 6
- **Specific figures**: See findings.md Section 11

---

## Version History

- **v1.0** (2025-10-29): Initial comprehensive analysis
  - 7 visualizations created
  - 6 analysis scripts
  - 2 comprehensive reports
  - All hypothesis tests completed

---

## Next Steps

1. **Review findings.md** for comprehensive results
2. **Check visualizations** for key patterns
3. **Implement NB model** per recommendations
4. **Validate assumptions** using diagnostic checklist (findings.md Section 10)
5. **Consider time-varying dispersion** if standard NB insufficient
6. **Coordinate with other analysts** on final model specification

---

**Bottom Line**: Strong, unambiguous evidence for Negative Binomial distribution with time-varying dispersion. Data quality is excellent. Ready for modeling.
