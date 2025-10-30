# Complete File Index

**Bayesian Time Series Count Modeling Project**
**Date:** October 29, 2025
**Total Files:** ~100+ across all directories
**Total Size:** ~65 MB

---

## Final Report Directory (`/workspace/final_report/`)

### Main Documents

| File | Size | Purpose | Read if... |
|------|------|---------|-----------|
| **report.md** | 150 KB | Comprehensive 30-page analysis | Need complete technical details |
| **executive_summary.md** | 20 KB | 2-page overview | Need quick summary |
| **quick_reference.md** | 8 KB | 1-page cheat sheet | Applying model or need quick facts |
| **README.md** | 12 KB | Navigation and overview | First-time reader |
| **FILE_INDEX.md** | This file | Complete file catalog | Looking for specific file |

### Figures Directory (`/workspace/final_report/figures/`)

| File | Size | Description |
|------|------|-------------|
| **exp1_ppc_dashboard.png** | 1.1 MB | 12-panel comprehensive model diagnostics (Exp 1) |
| **exp1_residual_diagnostics.png** | 714 KB | 6-panel residual analysis showing ACF(1)=0.686 |
| **exp1_fitted_values.png** | 99 KB | Data with model trend and 95% credible intervals |
| **exp1_trace_plots.png** | 2.4 MB | MCMC convergence diagnostics (4 chains) |
| **exp1_posteriors.png** | 218 KB | Parameter posterior distributions vs. priors |
| **exp3_trace_plots.png** | 1.1 MB | MCMC convergence for AR(1) model |
| **exp3_residual_diagnostics.png** | 713 KB | Residual analysis for Exp 3 |
| **exp3_ar_parameters.png** | 101 KB | AR(1) coefficient (ρ) and innovation SD (σ_η) |
| **acf_comparison_exp1_vs_exp3.png** | 206 KB | **CRITICAL:** Shows zero ACF improvement |
| **parameter_comparison.png** | 87 KB | Exp 1 vs Exp 3 parameter posteriors |
| **loo_comparison.png** | 35 KB | LOO cross-validation model comparison |

**Total figures:** 13 files, ~6.8 MB

### Supplementary Directory (`/workspace/final_report/supplementary/`)

| File | Size | Purpose |
|------|------|---------|
| **reproducibility.md** | 25 KB | Complete reproduction guide with environment, installation, verification |
| **parameter_interpretation_guide.md** | 35 KB | Detailed guide for each parameter with examples |

---

## Data (`/workspace/data/`)

| File | Size | Description |
|------|------|-------------|
| **data.csv** | 2 KB | Original data: 40 observations of (year, C) |

**Format:** CSV with header
**Variables:** year (float64, standardized), C (int64, counts)

---

## Exploratory Data Analysis (`/workspace/eda/`)

### Reports

| File | Size | Content |
|------|------|---------|
| **eda_report.md** | 20 KB | Comprehensive EDA findings |
| **eda_log.md** | 8 KB | Process documentation |

### Visualizations (`/workspace/eda/visualizations/`)

| File | Description |
|------|-------------|
| **timeseries_plot.png** | Time series showing growth pattern |
| **count_distribution.png** | Histogram with density overlay |
| **scatter_with_smoothing.png** | Scatter with linear, polynomial, LOWESS fits |
| **residual_diagnostics.png** | 4-panel residual analysis (linear model) |
| **variance_analysis.png** | Mean-variance relationship, overdispersion |
| **boxplot_by_period.png** | Distribution by time quartile |
| **autocorrelation_plot.png** | ACF showing temporal dependence |
| **log_transformation_analysis.png** | Log-scale analysis |

**Total:** 8 files, ~10 MB

---

## Experiment 1: Negative Binomial Quadratic (RECOMMENDED)

**Location:** `/workspace/experiments/experiment_1/`

### Root Files

| File | Content |
|------|---------|
| **metadata.md** | Experiment specification and summary |

### Prior Predictive Check (`prior_predictive_check/`)

| Directory | Content |
|-----------|---------|
| **code/** | Python scripts for PPC |
| **plots/** | 6 visualizations (prior predictive distribution, max values, etc.) |
| **FINDINGS.md** | Results showing β₂ prior needed adjustment |

### Simulation-Based Calibration (`simulation_based_validation/`)

| Directory | Content |
|-----------|---------|
| **code/** | 5 Python scripts for SBC workflow |
| **plots/** | 7 diagnostic plots (rank statistics, coverage) |
| **metrics/** | SBC coverage statistics |
| **sbc_results.md** | Validation findings (85% coverage for φ at 95%) |

### Posterior Inference (`posterior_inference/`)

| Directory/File | Content |
|----------------|---------|
| **code/** | `fit_model_pymc.py` - Main fitting script |
| **diagnostics/** | **posterior_inference.netcdf** (InferenceData, 1.9 MB) |
| **diagnostics/** | `summary_table.csv` - Parameter estimates |
| **diagnostics/** | `convergence_metrics.json` - R̂, ESS, divergences |
| **diagnostics/** | `convergence_report.md` - Detailed diagnostics |
| **plots/** | 5 diagnostic plots (trace, rank, posteriors, pairs, energy) |
| **inference_summary.md** | Complete posterior analysis |

### Posterior Predictive Check (`posterior_predictive_check/`)

| Directory/File | Content |
|----------------|---------|
| **code/** | PPC scripts (dashboard, residuals, test statistics) |
| **plots/** | 6 diagnostic plots including critical ACF(1)=0.686 |
| **ppc_findings.md** | Comprehensive PPC results |
| **SUMMARY.md** | Executive summary of PPC |
| **DECISION.md** | REJECT for temporal, ACCEPT for trend |

### Model Critique (`model_critique/`)

| File | Content |
|------|---------|
| **DECISION.md** | Final adequacy decision |
| **README.md** | Critique summary |
| **SUMMARY.md** | Findings recap |

**Experiment 1 total:** ~30 files, ~25 MB

---

## Experiment 3: Latent AR(1) Negative Binomial (NOT RECOMMENDED)

**Location:** `/workspace/experiments/experiment_3/`

**Structure:** Similar to Experiment 1 (prior_predictive_check/ → SBC → posterior_inference/ → PPC)

### Key Differences

| Aspect | Detail |
|--------|--------|
| **Parameters** | 46 (6 structural + 40 latent states) |
| **InferenceData** | Larger due to latent states |
| **AR parameters** | ρ = 0.84, σ_η = 0.09 |
| **Critical finding** | ACF(1) = 0.690 (no improvement vs. 0.686) |

### Unique Files

| File | Content |
|------|---------|
| **posterior_inference/plots/ar1_parameters.png** | AR(1) posteriors |
| **posterior_inference/plots/parameter_comparison_exp1_vs_exp3.png** | Cross-experiment |
| **posterior_inference/plots/fitted_values.png** | Time series fit |
| **posterior_predictive_check/plots/acf_comparison_exp1_vs_exp3.png** | **KEY EVIDENCE** |
| **posterior_predictive_check/SUMMARY.txt** | Failure analysis |

**Experiment 3 total:** ~30 files, ~20 MB

---

## Experiment Summary Files (`/workspace/experiments/`)

| File | Size | Content |
|------|------|---------|
| **experiment_plan.md** | 14 KB | Original experimental design |
| **iteration_log.md** | 6.5 KB | Modeling journey documentation |
| **adequacy_assessment.md** | 28 KB | Final decision with full rationale |

### Designer Proposals (`/workspace/experiments/designer_*/`)

| Directory | Content |
|-----------|---------|
| **designer_1/** | Parametric models proposal (quadratic, exponential) |
| **designer_2/** | Flexible models proposal (GP, splines) |
| **designer_3/** | Temporal models proposal (AR, state-space) |

**Note:** Designer proposals informed experiment plan but are not part of final deliverables.

---

## File Organization by Purpose

### For Understanding Results

**Start here:**
1. `/workspace/final_report/executive_summary.md`
2. `/workspace/final_report/quick_reference.md`

**Then if needed:**
3. `/workspace/final_report/report.md` (Sections 1, 4, 6, 7, 8)

**Supporting:**
4. `/workspace/final_report/figures/exp1_ppc_dashboard.png`
5. `/workspace/final_report/figures/acf_comparison_exp1_vs_exp3.png`

### For Reproduction

**Essential:**
1. `/workspace/final_report/supplementary/reproducibility.md`
2. `/workspace/data/data.csv`
3. `/workspace/experiments/experiment_1/posterior_inference/code/fit_model_pymc.py`

**Verification:**
4. `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
5. `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md`

### For Application

**Must read:**
1. `/workspace/final_report/quick_reference.md` (use cases)
2. `/workspace/final_report/supplementary/parameter_interpretation_guide.md`
3. `/workspace/final_report/report.md` Section 7 (limitations)

**Reference:**
4. `/workspace/experiments/experiment_1/posterior_inference/diagnostics/summary_table.csv`
5. `/workspace/final_report/figures/exp1_fitted_values.png`

### For Publication

**Methods:**
1. `/workspace/final_report/report.md` Section 3
2. `/workspace/experiments/experiment_plan.md`

**Results:**
3. `/workspace/final_report/report.md` Section 4.1
4. `/workspace/experiments/experiment_1/posterior_inference/inference_summary.md`

**Figures:**
5. `/workspace/final_report/figures/` (select 3-5 key plots)

**Limitations:**
6. `/workspace/final_report/report.md` Section 7

**Required disclosure:**
7. `/workspace/final_report/quick_reference.md` (template text)

### For Learning

**Workflow tutorial:**
1. `/workspace/eda/eda_report.md` (exploration)
2. `/workspace/experiments/experiment_1/` (complete pipeline)
3. `/workspace/final_report/report.md` Section 6.3 (lessons)

**Methodology:**
4. `/workspace/experiments/experiment_1/prior_predictive_check/FINDINGS.md`
5. `/workspace/experiments/experiment_1/simulation_based_validation/sbc_results.md`
6. `/workspace/experiments/adequacy_assessment.md`

**Negative result case study:**
7. `/workspace/experiments/experiment_3/posterior_predictive_check/SUMMARY.txt`
8. `/workspace/final_report/figures/acf_comparison_exp1_vs_exp3.png`

---

## Most Important Files (Top 10)

| Rank | File | Why Important |
|------|------|---------------|
| 1 | `/workspace/final_report/report.md` | Complete analysis documentation |
| 2 | `/workspace/final_report/executive_summary.md` | Quick overview of everything |
| 3 | `/workspace/data/data.csv` | The actual data |
| 4 | `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf` | Full posterior for recommended model |
| 5 | `/workspace/final_report/figures/acf_comparison_exp1_vs_exp3.png` | Shows why complex model failed |
| 6 | `/workspace/experiments/adequacy_assessment.md` | Decision rationale |
| 7 | `/workspace/final_report/supplementary/reproducibility.md` | How to reproduce |
| 8 | `/workspace/final_report/quick_reference.md` | Daily reference |
| 9 | `/workspace/final_report/figures/exp1_ppc_dashboard.png` | Comprehensive diagnostics |
| 10 | `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md` | Limitation details |

---

## File Naming Conventions

### Prefixes
- `exp1_*` - Experiment 1 (recommended model)
- `exp3_*` - Experiment 3 (complex model)
- `eda_*` - Exploratory data analysis
- `ppc_*` - Posterior predictive check

### Suffixes
- `*_dashboard.png` - Multi-panel comprehensive visualization
- `*_diagnostics.png` - Diagnostic plots
- `*_comparison.png` - Cross-model or cross-experiment comparison
- `*_summary.md` - Executive summary of analysis
- `*_findings.md` - Detailed findings
- `*_report.md` - Comprehensive documentation

### Key Terms
- **InferenceData** - ArviZ format containing full posterior samples
- **summary_table.csv** - Parameter estimates with HDI
- **convergence_metrics.json** - R̂, ESS, divergences
- **metadata.md** - Experiment specification

---

## Missing Files (Intentionally Not Created)

**The following were NOT created per strategic decisions:**

1. **Experiment 2 (Negative Binomial Exponential)**
   - Reason: Skipped per minimum attempt policy
   - Same model class as Exp 1, would have identical temporal issues
   - Documented in iteration_log.md

2. **Additional temporal models**
   - Reason: Diminishing returns after Exp 3 failure
   - Two failures on same metric = stop criterion met
   - Future directions documented in report

3. **Sensitivity analyses**
   - Reason: Core findings robust, priors not dominating posteriors
   - Could be added in future work if needed

4. **Model diagnostics for skipped Experiment 2**
   - Reason: Not fitted per strategic decision
   - Linear vs. quadratic comparison could be future work

---

## File Access by Research Question

### "What did you find?"
- **Quick:** `/workspace/final_report/executive_summary.md`
- **Detailed:** `/workspace/final_report/report.md` Sections 4, 6

### "Why did the complex model fail?"
- **Visual proof:** `/workspace/final_report/figures/acf_comparison_exp1_vs_exp3.png`
- **Analysis:** `/workspace/experiments/experiment_3/posterior_predictive_check/SUMMARY.txt`
- **Discussion:** `/workspace/final_report/report.md` Section 6.1.4

### "Can I use this model?"
- **Decision tree:** `/workspace/final_report/quick_reference.md`
- **Use cases:** `/workspace/final_report/report.md` Section 6.2.1
- **Limitations:** `/workspace/final_report/report.md` Section 7

### "How do I interpret parameters?"
- **Quick:** `/workspace/final_report/quick_reference.md`
- **Detailed:** `/workspace/final_report/supplementary/parameter_interpretation_guide.md`
- **Examples:** `/workspace/final_report/report.md` Section 4.1.3

### "How do I reproduce this?"
- **Guide:** `/workspace/final_report/supplementary/reproducibility.md`
- **Code:** `/workspace/experiments/experiment_1/posterior_inference/code/fit_model_pymc.py`
- **Verification:** Checklist in reproducibility.md

### "What were the data characteristics?"
- **Report:** `/workspace/eda/eda_report.md`
- **Visual:** `/workspace/eda/visualizations/timeseries_plot.png`
- **Summary:** `/workspace/final_report/report.md` Section 2

---

## Archival Recommendations

**For long-term preservation, archive:**

**Essential (minimum viable):**
- `/workspace/final_report/` (entire directory)
- `/workspace/data/data.csv`
- `/workspace/experiments/experiment_1/posterior_inference/`
- `/workspace/experiments/adequacy_assessment.md`

**Recommended (full archive):**
- `/workspace/` (entire directory except temp files)

**Archive format:** tar.gz or zip
**Estimated size:** ~65 MB compressed

**Create archive:**
```bash
cd /workspace
tar -czf bayesian_analysis_2025-10-29.tar.gz \
  final_report/ \
  data/ \
  eda/ \
  experiments/experiment_1/ \
  experiments/experiment_3/ \
  experiments/adequacy_assessment.md \
  experiments/iteration_log.md
```

---

## Quick File Lookup

**I need...**

- **Parameter estimates** → `/workspace/experiments/experiment_1/posterior_inference/diagnostics/summary_table.csv`
- **Full posterior samples** → `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- **Model equation** → `/workspace/final_report/quick_reference.md`
- **Limitations** → `/workspace/final_report/report.md` Section 7
- **Figures for presentation** → `/workspace/final_report/figures/`
- **Code to reproduce** → `/workspace/experiments/experiment_1/posterior_inference/code/fit_model_pymc.py`
- **Why complex model failed** → `/workspace/final_report/figures/acf_comparison_exp1_vs_exp3.png`
- **Convergence diagnostics** → `/workspace/experiments/experiment_1/posterior_inference/diagnostics/convergence_report.md`
- **Residual ACF** → `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md`
- **LOO comparison** → `/workspace/experiments/experiment_3/posterior_inference/diagnostics/loo_comparison.txt`

---

## Total Storage Summary

| Directory | Number of Files | Total Size | Purpose |
|-----------|----------------|------------|---------|
| `/workspace/final_report/` | 20 | ~8 MB | Final deliverables |
| `/workspace/data/` | 1 | 2 KB | Original data |
| `/workspace/eda/` | 12 | ~12 MB | Exploration |
| `/workspace/experiments/experiment_1/` | ~35 | ~25 MB | Recommended model |
| `/workspace/experiments/experiment_3/` | ~35 | ~20 MB | Complex model |
| **Total** | **~100** | **~65 MB** | Complete project |

---

**INDEX Version:** 1.0
**Date:** October 29, 2025
**Status:** Complete

**For questions about file locations or organization, see `/workspace/final_report/README.md`**
