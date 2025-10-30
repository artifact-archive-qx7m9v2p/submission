# Exploratory Data Analysis: Binomial Dataset

This directory contains a comprehensive exploratory data analysis of the binomial dataset located at `/workspace/data/data.csv`.

## Quick Start

For a quick overview of findings, run:
```bash
python /workspace/eda/code/00_summary.py
```

## Directory Structure

```
eda/
├── README.md                    # This file
├── eda_report.md               # Comprehensive analysis report (PRIMARY OUTPUT)
├── eda_log.md                  # Detailed exploration log with process notes
├── code/                       # Reproducible analysis scripts
│   ├── 00_summary.py          # Quick summary (run this first!)
│   ├── 01_initial_exploration.py
│   ├── 02_overdispersion_analysis.py
│   ├── 03_visualization.py
│   └── 04_pattern_analysis.py
└── visualizations/             # 8 publication-quality plots
    ├── sample_size_distribution.png
    ├── proportion_distribution.png
    ├── proportion_vs_trial.png
    ├── proportion_vs_sample_size.png
    ├── standardized_residuals.png
    ├── comprehensive_comparison.png
    ├── qq_plot.png
    └── funnel_plot.png
```

## Key Findings

### Primary Finding: Strong Overdispersion
- **Chi-square test**: χ² = 38.56, df = 11, **p < 0.001**
- **Dispersion parameter**: φ = 3.51 (variance is 3.5x larger than expected)
- **Conclusion**: Simple binomial model with constant probability is **REJECTED**

### Data Characteristics
- **Observations**: 12 binomial trials
- **Total sample size**: 2,814 trials
- **Total successes**: 208
- **Pooled proportion**: 0.0739 (7.39%)
- **Proportion range**: [0.000, 0.144]

### Pattern Analysis Results
- **Temporal trend**: NO (p = 0.199)
- **Sample size effect**: NO (p = 0.787)
- **Distinct groups**: YES (p = 0.012)
- **Outliers**: Trial 1 (0% success) and Trial 8 (14.4% success)

## Model Recommendations

### Do NOT Use
- ✗ Simple Binomial(n, p) with constant p → **Strongly rejected by data**

### Recommended Models

1. **Beta-Binomial Model** (PRIMARY RECOMMENDATION)
   - Naturally accounts for overdispersion
   - φ parameter should be around 3-4
   - Prior: Beta(2, 25) for probability, Gamma(2, 0.5) for dispersion

2. **Two-Component Mixture Model** (SENSITIVITY ANALYSIS)
   - May capture two distinct probability groups
   - Low group: p ≈ 0.05
   - High group: p ≈ 0.11

3. **Hierarchical Binomial Model** (MOST FLEXIBLE)
   - Trial-specific probabilities
   - Borrows strength across observations

## Documentation

### Main Report
**File**: `eda_report.md`
- Executive summary
- Data quality assessment
- Overdispersion analysis
- Pattern detection results
- Modeling recommendations
- Prior specifications
- Validation steps

### Detailed Log
**File**: `eda_log.md`
- Round-by-round exploration process
- Hypothesis testing details
- Alternative explanations considered
- Robust vs tentative findings

## Visualizations

All plots are saved in `visualizations/` directory:

1. **sample_size_distribution.png**: Shows heterogeneity in sample sizes
2. **proportion_distribution.png**: Success proportions across trials
3. **proportion_vs_trial.png**: Checks for temporal trends (none found)
4. **proportion_vs_sample_size.png**: Tests size-dependence (none found)
5. **standardized_residuals.png**: Visual evidence of overdispersion
6. **comprehensive_comparison.png**: 4-panel comparison of observed vs expected
7. **qq_plot.png**: Normality check (departure confirms overdispersion)
8. **funnel_plot.png**: Classic overdispersion diagnostic (points outside funnel)

## Reproducibility

All analyses are fully reproducible:
- Python 3 with standard scientific stack (pandas, numpy, scipy, matplotlib, seaborn)
- No proprietary software required
- All scripts are self-contained and documented
- Random seeds set where applicable

### Running Individual Scripts

```bash
# Initial exploration and data quality
python /workspace/eda/code/01_initial_exploration.py

# Overdispersion tests
python /workspace/eda/code/02_overdispersion_analysis.py

# Generate all visualizations
python /workspace/eda/code/03_visualization.py

# Pattern analysis and hypothesis testing
python /workspace/eda/code/04_pattern_analysis.py
```

## Statistical Tests Performed

1. **Chi-square goodness of fit** → Reject constant p model (p < 0.001)
2. **Dispersion parameter estimation** → φ = 3.51
3. **Pearson and Spearman correlations** → No temporal or size effects
4. **T-tests** → Evidence for two groups (p = 0.012)
5. **IQR outlier detection** → Trials 1 and 8 identified
6. **Runs test** → Sequence appears random (p = 0.226)
7. **Levene test** → Variance homogeneity across groups (p = 0.853)

## Key Takeaways

1. **Strong overdispersion** (3.5x expected variance) is the dominant feature
2. **No systematic biases** (temporal, sample-size-dependent)
3. **Possible group structure** (low-probability vs high-probability trials)
4. **Beta-Binomial or mixture models required** for valid inference
5. **Simple binomial model will underestimate uncertainty** by ~2x
6. **Pooled estimate is 0.074** but with substantial heterogeneity

## Contact

For questions or additional analyses, refer to:
- **Comprehensive report**: `/workspace/eda/eda_report.md`
- **Detailed log**: `/workspace/eda/eda_log.md`
- **Summary output**: Run `python /workspace/eda/code/00_summary.py`

---

**Analysis completed**: 2025-10-30
**Total runtime**: ~5 minutes
**Files generated**: 14 (5 code scripts, 8 visualizations, 3 reports)
