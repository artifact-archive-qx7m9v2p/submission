# Exploratory Data Analysis: X-Y Relationship

**Date:** 2025-10-27
**Dataset:** `/workspace/data/data.csv`
**Sample Size:** N = 27 observations

## Quick Links

- **[EDA Report](eda_report.md)** - Comprehensive analysis report with all findings and recommendations
- **[EDA Log](eda_log.md)** - Detailed exploration process with intermediate findings
- **[Summary Visualization](visualizations/eda_summary.png)** - One-page visual summary

## Key Findings

1. **Relationship:** Strong, nonlinear positive association (r=0.72)
2. **Pattern:** Diminishing marginal returns / saturation behavior
3. **Data Quality:** Excellent - complete, normal residuals, constant variance
4. **Recommended Model:** Bayesian logarithmic regression (R²=0.83)

## Directory Structure

```
eda/
├── README.md                           # This file
├── eda_report.md                       # Main comprehensive report
├── eda_log.md                          # Detailed exploration log
├── code/                               # Reproducible analysis scripts
│   ├── 01_initial_exploration.py      # Data quality & univariate stats
│   ├── 02_univariate_analysis.py      # Distribution analysis
│   ├── 03_bivariate_analysis.py       # Relationship & model comparison
│   ├── 04_residual_diagnostics.py     # Residual analysis & tests
│   └── 05_summary_visualization.py    # Summary plot generation
└── visualizations/                     # All plots (300 DPI)
    ├── eda_summary.png                # One-page summary ⭐
    ├── distribution_x.png             # Predictor distribution
    ├── distribution_Y.png             # Response distribution
    ├── distribution_comparison.png    # Comparative distributions
    ├── scatter_relationship.png       # Main scatterplots with fits
    ├── advanced_patterns.png          # Residuals & segmentation
    ├── model_comparison.png           # Four model types compared
    ├── residual_diagnostics.png       # 6-panel diagnostic suite
    └── heteroscedasticity_analysis.png # Variance structure

```

## Recommended Next Steps

1. Implement Bayesian logarithmic regression:
   ```
   Y ~ Normal(β₀ + β₁·log(x), σ²)
   ```

2. Use recommended priors:
   - β₀ ~ Normal(1.7, 0.5)
   - β₁ ~ Normal(0.3, 0.2)
   - σ ~ HalfNormal(0.2)

3. Perform posterior predictive checks

4. Compare with quadratic alternative via WAIC/LOO-CV

5. Report predictions with credible intervals

## Code Reproducibility

All code is self-contained and can be re-run:

```bash
cd /workspace/eda/code
python 01_initial_exploration.py
python 02_univariate_analysis.py
python 03_bivariate_analysis.py
python 04_residual_diagnostics.py
python 05_summary_visualization.py
```

**Requirements:**
- Python 3.x
- pandas, numpy, scipy, matplotlib, seaborn

## Contact

For questions about this analysis, refer to detailed findings in `eda_report.md`.
