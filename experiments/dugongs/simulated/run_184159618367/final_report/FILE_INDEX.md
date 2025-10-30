# Complete File Index
## Bayesian Modeling Project: Y-x Relationship

**Navigation Guide for All Project Files**

---

## Final Report (THIS DIRECTORY)

### Main Documents

| File | Purpose | Audience | Length | Read Time |
|------|---------|----------|--------|-----------|
| `README.md` | Directory guide | All | 5 pages | 10 min |
| `EXECUTIVE_SUMMARY.md` | Non-technical summary | Executives, stakeholders | 6 pages | 10 min |
| `QUICK_REFERENCE.md` | Practitioner's cheat sheet | Applied scientists | 1 page | 5 min |
| `report.md` | Comprehensive report | Scientists, reviewers | 52 pages | 2-3 hours |
| `FILE_INDEX.md` | This file - complete navigation | All | 4 pages | 5 min |

### Supplementary Materials

| File | Purpose | Audience | Length | Read Time |
|------|---------|----------|--------|-----------|
| `supplementary/technical_details.md` | Implementation details | Statisticians, replicators | 25 pages | 1-2 hours |

### Figures (All in `figures/`)

| File | Description | Reference in Report |
|------|-------------|---------------------|
| `main_model_fit.png` | Power law fit with credible intervals | Figure 1 (main report) |
| `parameter_posteriors.png` | Posterior distributions (α, β, σ) | Figure 2 (main report) |
| `convergence_diagnostics.png` | MCMC trace plots | Figure 3 (main report) |
| `residual_diagnostics.png` | Residual analysis on log scale | Figure 4 (supplement) |
| `prediction_intervals.png` | Coverage diagnostic | Figure 5 (supplement) |
| `model_comparison_loo.png` | LOO cross-validation comparison | Figure 6 (supplement) |
| `scale_comparison.png` | Log-log vs original scale | Figure 7 (supplement) |

---

## Experiment Files

### Winner Model: Experiment 3 (Log-Log Power Law)

**Base Directory**: `/workspace/experiments/experiment_3/`

#### Core Documentation

| File | Content |
|------|---------|
| `metadata.md` | Model specification, priors, falsification criteria |
| `prior_predictive_check/findings.md` | Prior validation results |
| `prior_predictive_check/RECOMMENDATION.md` | Prior revision decision |
| `posterior_inference/inference_summary.md` | Parameter estimates, convergence diagnostics |
| `posterior_inference/diagnostics/convergence_report.md` | Detailed MCMC diagnostics |
| `posterior_predictive_check/ppc_findings.md` | Model adequacy assessment |
| `model_critique/decision.md` | ACCEPT/REVISE/REJECT decision |
| `model_critique/critique_summary.md` | Comprehensive critique |
| `model_critique/improvement_priorities.md` | Potential improvements |

#### Data and Results

| File | Content |
|------|---------|
| `posterior_inference/diagnostics/posterior_inference.netcdf` | **InferenceData** (all posterior samples) |
| `posterior_inference/diagnostics/parameter_summary.csv` | Tabulated parameter estimates |
| `prior_predictive_check/ppc_results.json` | Prior predictive check results |
| `posterior_predictive_check/ppc_results.json` | Posterior predictive check results |

#### Code

| Directory | Contains |
|-----------|----------|
| `prior_predictive_check/code/` | Prior predictive check scripts |
| `posterior_inference/code/` | Model fitting (PyMC implementation) |
| `posterior_predictive_check/code/` | PPC analysis scripts |
| `model_critique/code/` | LOO diagnostics and critique |

#### Visualizations (22 total)

**Prior Predictive Checks** (6 plots in `prior_predictive_check/plots/`):
- `parameter_plausibility.png` - Prior parameter distributions
- `prior_predictive_coverage.png` - Prior trajectory coverage
- `behavior_diagnostics.png` - Prior behavior assessment
- `heavy_tail_diagnostics.png` - Tail behavior checks
- `pointwise_plausibility.png` - Point-wise prior checks
- `prior_revision_comparison.png` - v1 vs v2 comparison

**Posterior Inference** (7 plots in `posterior_inference/plots/`):
- `trace_plots.png` - MCMC traces (convergence)
- `rank_plots.png` - Rank uniformity (mixing)
- `posterior_distributions.png` - Parameter posteriors
- `pairs_plot.png` - Joint posteriors and correlations
- `posterior_predictive_check.png` - Model fit to data
- `power_law_fit.png` - Power law curve with data
- `residual_analysis.png` - Residual diagnostics

**Posterior Predictive Checks** (8 plots in `posterior_predictive_check/plots/`):
- `overlay_ppc.png` - Observed vs replicated data
- `coverage_diagnostic.png` - Spatial coverage across x
- `residual_plot_log_scale.png` - Log-scale residuals
- `qq_plot_residuals.png` - Normality check
- `replicate_performance.png` - Fit at replicated x-values
- `summary_statistics_comparison.png` - PPC test statistics
- `scale_comparison.png` - Log-log vs original scale
- `r_squared_distribution.png` - Posterior predictive R²

**Model Critique** (1 plot in `model_critique/`):
- `pareto_k_diagnostic.png` - LOO Pareto k values
- `critique_summary_visual.png` - Visual summary of critique

---

### Alternative Model: Experiment 1 (Asymptotic Exponential)

**Base Directory**: `/workspace/experiments/experiment_1/`

#### Key Files

| File | Content |
|------|---------|
| `posterior_inference/diagnostics/posterior_inference.netcdf` | InferenceData for Exp1 |
| `posterior_inference/inference_summary.md` | Parameter estimates and diagnostics |
| `posterior_predictive_check/ppc_findings.md` | PPC assessment |
| `model_critique/decision.md` | ACCEPT decision (but not selected) |

#### Visualizations (11 total)

**Posterior Inference** (5 plots in `posterior_inference/plots/`):
- `convergence_overview.png`
- `model_fit.png`
- `posterior_distributions.png`
- `posterior_predictive_checks.png`
- `convergence_metrics.png`

**Posterior Predictive Checks** (6 plots in `posterior_predictive_check/plots/`):
- `ppc_overlay.png`
- `predictive_intervals.png`
- `residual_diagnostics.png`
- `test_statistics.png`
- `observed_vs_predicted.png`
- `distribution_comparison.png`

**Model Critique** (1 plot):
- `loo_diagnostics.png`

---

## Model Comparison

**Directory**: `/workspace/experiments/model_comparison/`

### Files

| File | Content |
|------|---------|
| `comparison_report.md` | Comprehensive model comparison (detailed) |
| `comparison_summary.csv` | Tabulated comparison metrics |
| `code/model_comparison.py` | Comparison analysis code |

### Visualizations (6 plots in `plots/`)

| File | Description |
|------|-------------|
| `loo_comparison.png` | ELPD comparison (primary decision plot) |
| `integrated_comparison_dashboard.png` | 9-panel comprehensive comparison |
| `model_fits_comparison.png` | Side-by-side model fits |
| `pareto_k_comparison.png` | Pareto k diagnostics comparison |
| `residual_analysis_comparison.png` | Residual patterns comparison |
| `parameter_comparison.png` | Parameter posterior comparison |

---

## Planning and Assessment

**Directory**: `/workspace/experiments/`

### Key Reports

| File | Purpose | Length |
|------|---------|--------|
| `experiment_plan.md` | Model design and validation strategy | 8 pages |
| `adequacy_assessment.md` | Final adequacy decision | 15 pages |
| `log.md` | Experiment log (if exists) | Variable |

---

## Exploratory Data Analysis

**Directory**: `/workspace/eda/`

### Main Report

| File | Content |
|------|---------|
| `eda_report.md` | Comprehensive EDA synthesis | 10 pages |
| `synthesis.md` | Dual analyst synthesis (if separate) | Variable |

### Analyst Reports

**Analyst 1** (`/workspace/eda/analyst_1/`):
- `findings.md` - EDA findings
- `visualizations/` - 8 plots including:
  - `00_comprehensive_summary.png`
  - `01_scatter_with_smoothers.png`
  - `03_segmented_relationship.png`
  - `04_residual_diagnostics.png`
  - `05_model_comparison.png`

**Analyst 2** (`/workspace/eda/analyst_2/`):
- `findings.md` - EDA findings
- `visualizations/` - 14 plots including:
  - `00_SUMMARY_comprehensive.png`
  - `04_all_functional_forms.png`
  - `05_top_models_comparison.png`
  - `10_segmentation_analysis.png`
  - `11_transformations.png`

---

## Data

**Location**: `/workspace/data/data.csv`

**Contents**:
- 27 observations
- Columns: `x`, `Y`
- No missing values
- 6 x-values with replicates

**Statistics**:
- x: range [1.0, 31.5], mean = 10.94, SD = 7.87
- Y: range [1.71, 2.63], mean = 2.32, SD = 0.28

---

## Navigation by Task

### I Want to Understand the Results

**Start Here**:
1. `/workspace/final_report/EXECUTIVE_SUMMARY.md` (5 min)
2. `/workspace/final_report/report.md` Section 1-2 (20 min)
3. `/workspace/final_report/figures/main_model_fit.png` (visual)

**Then Explore**:
- Section 7 (Scientific Interpretation)
- Section 9 (Limitations)

### I Want to Use the Model for Prediction

**Start Here**:
1. `/workspace/final_report/QUICK_REFERENCE.md` (5 min)
2. Load: `/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf`

**Reference**:
- `/workspace/final_report/report.md` Section 10.2 (Recommendations for Prediction)

### I Want to Validate the Analysis

**Check These**:
1. `/workspace/experiments/experiment_3/posterior_inference/inference_summary.md` (convergence)
2. `/workspace/experiments/experiment_3/posterior_predictive_check/ppc_findings.md` (model adequacy)
3. `/workspace/experiments/model_comparison/comparison_report.md` (model selection)
4. `/workspace/experiments/adequacy_assessment.md` (final decision)

**Review Figures**:
- All 22 plots in `/workspace/experiments/experiment_3/*/plots/`

### I Want to Replicate the Analysis

**Follow This Order**:
1. `/workspace/final_report/supplementary/technical_details.md` (implementation)
2. Code in `/workspace/experiments/experiment_3/*/code/`
3. Data: `/workspace/data/data.csv`

**Verification**:
- `/workspace/final_report/supplementary/technical_details.md` Section 11 (Reproducibility)

### I Want to Understand Model Comparison

**Read These**:
1. `/workspace/experiments/model_comparison/comparison_report.md` (detailed)
2. `/workspace/final_report/report.md` Section 4 (summary)

**View Figures**:
- `/workspace/final_report/figures/model_comparison_loo.png`
- `/workspace/experiments/model_comparison/plots/integrated_comparison_dashboard.png`

### I Want to Extend the Analysis

**Review**:
1. `/workspace/final_report/report.md` Section 10.4 (Future Work)
2. `/workspace/experiments/experiment_3/model_critique/improvement_priorities.md`
3. `/workspace/final_report/supplementary/technical_details.md` Section 13 (Future Enhancements)

---

## Quick Access: Most Important Files

### Top 5 Must-Read Documents

1. **`/workspace/final_report/EXECUTIVE_SUMMARY.md`** - Start here (everyone)
2. **`/workspace/final_report/report.md`** - Complete story (scientists)
3. **`/workspace/experiments/experiment_3/posterior_inference/inference_summary.md`** - Parameter estimates
4. **`/workspace/experiments/model_comparison/comparison_report.md`** - Why this model won
5. **`/workspace/experiments/adequacy_assessment.md`** - Final adequacy decision

### Top 5 Must-See Visualizations

1. **`/workspace/final_report/figures/main_model_fit.png`** - The model in action
2. **`/workspace/final_report/figures/model_comparison_loo.png`** - Why we chose this model
3. **`/workspace/final_report/figures/convergence_diagnostics.png`** - Validation evidence
4. **`/workspace/experiments/experiment_3/posterior_predictive_check/plots/coverage_diagnostic.png`** - Model adequacy
5. **`/workspace/experiments/model_comparison/plots/integrated_comparison_dashboard.png`** - Complete comparison

### Top 3 Key Data Files

1. **`/workspace/data/data.csv`** - Original data
2. **`/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf`** - Posterior samples
3. **`/workspace/experiments/model_comparison/comparison_summary.csv`** - Comparison metrics

---

## File Counts Summary

| Category | Count |
|----------|-------|
| **Final Report Documents** | 5 markdown files |
| **Final Report Figures** | 7 PNG files |
| **Experiment 3 Documentation** | 10+ markdown files |
| **Experiment 3 Visualizations** | 22 PNG files |
| **Experiment 1 Documentation** | 6+ markdown files |
| **Experiment 1 Visualizations** | 11 PNG files |
| **Model Comparison Visualizations** | 6 PNG files |
| **EDA Visualizations** | 22 PNG files (8+14) |
| **Code Files** | 15+ Python scripts |
| **Data Files** | 1 CSV, 2 NetCDF (InferenceData) |

**Total Files**: ~100+ files across entire project

---

## Search Tips

### Find by Keyword

**"Power Law"**:
- Main report Section 5, 7
- Experiment 3 metadata
- QUICK_REFERENCE

**"Convergence"**:
- Experiment 3 inference_summary
- Experiment 3 convergence_report
- Main report Section 6.1

**"LOO" or "Cross-Validation"**:
- Model comparison report
- Main report Section 4.1
- Technical details Section 5

**"Limitations"**:
- Main report Section 9
- Adequacy assessment Section 10.4
- EXECUTIVE_SUMMARY

**"Prediction"**:
- QUICK_REFERENCE
- Main report Section 10.2
- Technical details Section 6

**"Prior"**:
- Experiment 3 metadata
- Prior predictive check findings
- Technical details Section 2

---

## File Formats

| Format | Purpose | Count |
|--------|---------|-------|
| `.md` (Markdown) | Documentation, reports | ~40 files |
| `.png` (PNG images) | Visualizations | ~50 files |
| `.py` (Python) | Analysis code | ~15 files |
| `.csv` (CSV) | Tabulated data/results | ~5 files |
| `.netcdf` (NetCDF) | InferenceData objects | 2 files |
| `.json` (JSON) | Structured results | ~5 files |
| `.stan` (Stan code) | Model specification | 1 file (reference) |

---

## Last Updated

**Date**: October 27, 2025
**Project Status**: COMPLETE
**Model Status**: ADEQUATE
**Documentation**: FINAL

---

**Navigation Help**: If you can't find what you need, start with the README in the relevant directory or consult this index.
