# EDA Analyst 2: Hierarchical Structure Analysis

## Quick Start

**Main findings document**: `findings.md` - Comprehensive report with all results and recommendations

**Detailed exploration log**: `eda_log.md` - Step-by-step analysis process and intermediate findings

## Directory Structure

```
/workspace/eda/analyst_2/
├── findings.md                          # Main findings report (START HERE)
├── eda_log.md                          # Detailed analysis log
├── README.md                           # This file
├── code/                               # Reproducible analysis scripts
│   ├── 01_initial_exploration.py       # Basic statistics, CI calculations
│   ├── 02_caterpillar_plots.py        # Caterpillar plots, CI overlap analysis
│   ├── 03_variance_decomposition.py   # ICC, variance components, shrinkage
│   ├── 04_clustering_analysis.py      # Distribution tests, clustering
│   ├── 05_summary_visualization.py    # Comprehensive summary figure
│   ├── group_data_with_ci.csv         # Data + confidence intervals
│   ├── hierarchical_analysis.csv      # + shrinkage factors
│   └── clustering_analysis.csv        # + clustering assignments
└── visualizations/                     # All plots (PNG format)
    ├── hierarchical_summary.png       # COMPREHENSIVE 6-panel summary (RECOMMENDED)
    ├── caterpillar_plot_sorted.png    # Groups sorted by success rate with CIs
    ├── caterpillar_plot_by_id.png     # Groups in original order
    ├── shrinkage_visualization.png    # Arrows showing shrinkage potential
    ├── pooling_comparison.png         # Three pooling strategies compared
    ├── distribution_analysis.png      # 4-panel distribution diagnostics
    ├── clustering_analysis.png        # Clustering attempts (2 panels)
    └── regression_to_mean.png         # Sample size vs deviation from pooled
```

## Key Results at a Glance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overdispersion ratio** | 5.06x | Variance 5x larger than expected |
| **Chi-square p-value** | 0.000063 | Strong evidence for heterogeneity |
| **ICC** | 0.727 | 72.7% of variance is between-group |
| **Average shrinkage** | 85.6% | Substantial information pooling |
| **Normality test** | p = 0.496 | Supports normal prior |

## Main Recommendation

**Use hierarchical/multilevel binomial model with partial pooling**

Rationale:
- Strong overdispersion (5x)
- High ICC (72.7%)
- Continuous distribution (not discrete clusters)
- Handles unbalanced design naturally
- Stabilizes small-sample estimates

## Visualizations Guide

**For presentations**: Use `hierarchical_summary.png` (6-panel comprehensive overview)

**For detailed exploration**:
- `caterpillar_plot_sorted.png` - See all group rates with uncertainty
- `pooling_comparison.png` - Understand shrinkage visually
- `distribution_analysis.png` - Check distributional assumptions

## Code Execution

All scripts are self-contained and can be run independently:

```bash
# From /workspace directory
python eda/analyst_2/code/01_initial_exploration.py
python eda/analyst_2/code/02_caterpillar_plots.py
python eda/analyst_2/code/03_variance_decomposition.py
python eda/analyst_2/code/04_clustering_analysis.py
python eda/analyst_2/code/05_summary_visualization.py
```

## Data Source

Input data: `/workspace/data/data_analyst_2.csv`

Structure:
- 12 groups
- Variables: group, n_trials, r_successes, success_rate
- Total: 2,814 trials, 208 successes
- Pooled rate: 7.39%
