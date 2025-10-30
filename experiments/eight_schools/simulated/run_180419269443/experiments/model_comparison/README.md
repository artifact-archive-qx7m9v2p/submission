# Model Comparison: 8 Schools Meta-Analysis

## Overview

This directory contains a comprehensive comparison of 4 Bayesian models for the 8 Schools SAT coaching study, evaluated using Leave-One-Out Cross-Validation (LOO-CV).

## Quick Summary

**KEY FINDING:** All four models show statistically equivalent predictive performance (all ΔELPD < 2×SE), with posterior mean estimates ranging from 8.58 to 10.40. Model choice has minimal impact on substantive conclusions.

**RECOMMENDATION:** Complete Pooling model for primary analysis (interpretability), with sensitivity analyses using all four models.

## Models Compared

1. **Hierarchical** - Partial pooling with hyperpriors (μ, τ, θ)
2. **Complete Pooling** - Single global mean (μ only)
3. **Skeptical** - Hierarchical with skeptical priors
4. **Enthusiastic** - Hierarchical with enthusiastic priors

## Results Summary

| Model | Rank | ELPD | SE | ΔELPD | Weight | p_loo |
|-------|------|------|----|----|--------|-------|
| Skeptical | 1 | -63.87 | 2.73 | 0.00 | 64.9% | 1.00 |
| Enthusiastic | 2 | -63.96 | 2.81 | 0.09 | 35.1% | 1.20 |
| Complete Pooling | 3 | -64.12 | 2.87 | 0.25 | 0.0% | 1.18 |
| Hierarchical | 4 | -64.46 | 2.21 | 0.59 | 0.0% | 2.11 |

**Statistical Equivalence:** All ΔELPD < 2×SE → No model significantly better

**Parsimony:** Skeptical model simplest (p_loo = 1.00)

**Robustness:** μ estimates range 8.58-10.40 (1.83 spread, < posterior SD ~4)

## Directory Structure

```
model_comparison/
├── README.md                          # This file
├── comparison_report.md               # Comprehensive 13-section report
├── recommendation.md                  # Final model selection decision
├── code/
│   └── model_comparison_analysis.py   # Complete analysis script
├── diagnostics/
│   ├── loo_comparison_full.csv        # LOO comparison table
│   ├── calibration_metrics.json       # Coverage and calibration stats
│   └── predictive_metrics.csv         # RMSE, MAE, bias, coverage
└── plots/
    ├── loo_comparison.png             # ELPD comparison with error bars
    ├── model_weights.png              # Stacking weights
    ├── pareto_k_diagnostics.png       # Reliability diagnostics
    └── predictive_performance.png     # 5-panel dashboard
```

## Key Documents

### 1. comparison_report.md (Primary Document)

**Comprehensive 13-section analysis covering:**
- Executive summary
- LOO-CV comparison with statistical equivalence tests
- Pareto k diagnostics (all reliable)
- Calibration assessment (100% coverage, slightly conservative)
- Predictive metrics (RMSE, MAE, bias)
- Posterior parameter estimates (μ, τ)
- Model selection decision with rationale
- Practical reporting guidance
- Template language for publications
- Interpretation and key insights

**Length:** ~50 pages equivalent
**Audience:** Statistical researchers and practitioners

### 2. recommendation.md (Decision Document)

**Executive recommendation with:**
- Primary model: Complete Pooling
- Alternative: Skeptical Priors
- Dual recommendation structure for different audiences
- Decision criteria applied transparently
- Visual evidence summary
- Implementation guidance for final reports
- Template language for abstract, methods, results, discussion

**Length:** ~20 pages equivalent
**Audience:** Decision-makers and authors

## Visualizations

### plots/loo_comparison.png
LOO-CV comparison showing all models within error bars, confirming statistical equivalence.

**Key Message:** No model clearly superior

### plots/model_weights.png
Stacking weights concentrate on Skeptical (65%) and Enthusiastic (35%).

**Key Message:** Model averaging favors these two, but Complete Pooling competitive

### plots/pareto_k_diagnostics.png
All models show reliable LOO estimates (k < 0.7 for all observations).

**Key Message:** LOO comparison is valid and trustworthy

### plots/predictive_performance.png
5-panel dashboard showing:
- Panel A: ELPD rankings
- Panel B: Stacking weights
- Panel C: RMSE/MAE comparison
- Panel D: Interval coverage (calibration)
- Panel E: Posterior predictive vs observed

**Key Message:** Models perform similarly across all criteria

## Reproducibility

### To reproduce this analysis:

```bash
cd /workspace/experiments/model_comparison/code
python model_comparison_analysis.py
```

**Requirements:**
- PyMC 5.x
- ArviZ 0.18+
- NumPy, Pandas, Matplotlib, Seaborn
- All model NetCDF files in expected locations

### Model locations:
- Hierarchical: `experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Complete Pooling: `experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`
- Skeptical: `experiments/experiment_4/experiment_4a_skeptical/posterior_inference/diagnostics/posterior_inference.netcdf`
- Enthusiastic: `experiments/experiment_4/experiment_4b_enthusiastic/posterior_inference/diagnostics/posterior_inference.netcdf`

## Key Insights

### 1. Model Complexity Doesn't Help
More complex models (Hierarchical, p_loo=2.11) don't predict better than simpler models (Skeptical, p_loo=1.00).

### 2. Prior Sensitivity is Modest
Skeptical (μ=8.58) vs Enthusiastic (μ=10.40) differ by 1.83, small relative to posterior SD (~4).

### 3. Pooling is Appropriate
Complete pooling performs as well as hierarchical models, suggesting limited between-study heterogeneity or insufficient data to estimate it.

### 4. Data Dominate Priors
All models converge to similar μ despite different priors, indicating data informativeness.

### 5. Small Sample Uncertainty
With J=8, all models show wide credible intervals; more data needed for precision.

## Recommendations by Priority

| Priority | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Pure prediction | Skeptical | Best LOO + highest weight |
| Parsimony | Skeptical | Lowest p_loo = 1.00 |
| **Interpretability** | **Complete Pooling** | **Single parameter, simple** |
| Conservative | Skeptical | Lower μ, skeptical priors |
| Flexibility | Stacking | Combines Skeptical + Enthusiastic |
| Full modeling | Hierarchical | Estimates τ, allows shrinkage |

**Our choice:** Interpretability (Complete Pooling) with Skeptical as sensitivity check

## Usage for Publications

### For Methods Section:
See `comparison_report.md` Section 8.1 for template language

### For Results Section:
```
The estimated overall coaching effect was robust across model specifications, 
ranging from 8.6 to 10.4 points (mean ~10 points) with substantial uncertainty 
(SD ~4 points). We report results from the complete pooling model for 
interpretability, with μ = 10.0 ± 4.1 (95% CI: [2.3, 17.9]).
```

### For Discussion Section:
```
Model comparison via LOO-CV revealed that no single model clearly outperformed 
others in predictive accuracy, suggesting that with J=8 studies and large 
within-study variance, the data do not strongly favor hierarchical structure 
over complete pooling.
```

## References

**Methods:**
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis*, 13(3), 917-1007.

**Software:**
- PyMC 5.x: Probabilistic programming in Python
- ArviZ 0.18: Bayesian model diagnostics and visualization

## Contact

**Analysis performed by:** Claude (Model Assessment Specialist)
**Date:** 2025-10-28
**Framework:** Anthropic Claude Agent SDK

## License

Analysis code and reports are provided for educational and research purposes.

---

**Bottom Line:** Model choice matters little for this dataset. The key finding is robust: coaching programs show a positive but uncertain average effect of ~10 SAT points. Simpler models (Complete Pooling) suffice given J=8 and large within-study variance.

