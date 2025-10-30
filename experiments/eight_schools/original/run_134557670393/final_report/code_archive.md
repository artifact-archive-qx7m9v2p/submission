# Code Archive Guide
## Bayesian Meta-Analysis Project

**Date**: October 28, 2025
**Purpose**: Navigation guide to all code and analysis files
**Project**: Rigorous Bayesian Workflow for Meta-Analysis

---

## Project Structure

```
/workspace/
├── data/
│   └── data.csv                          # Original dataset (8 studies)
├── eda/                                  # Phase 1: Exploratory Data Analysis
│   ├── analyst_1/                        # Distributions & Heterogeneity perspective
│   ├── analyst_2/                        # Uncertainty & Patterns perspective
│   ├── analyst_3/                        # Structure & Context perspective
│   └── eda_report.md                     # Synthesized EDA findings
├── experiments/                          # Phases 2-5: Model Development
│   ├── experiment_plan.md                # Phase 2: Model design specifications
│   ├── experiment_1/                     # Model 1: Hierarchical Normal
│   ├── model_assessment/                 # Phase 4: Assessment & comparison
│   └── adequacy_assessment.md            # Phase 5: Final adequacy decision
└── final_report/                         # Phase 6: Publication outputs
    ├── report.md                         # Main comprehensive report
    ├── executive_summary.md              # 2-page stakeholder summary
    ├── key_figures.md                    # Figure guide and captions
    ├── model_specification.md            # Complete mathematical specification
    ├── code_archive.md                   # This document
    └── figures/                          # Key visualizations
```

---

## Phase 1: Exploratory Data Analysis

### Analyst #1: Distributions & Heterogeneity

**Directory**: `/workspace/eda/analyst_1/`

**Code Files**:
- `eda_analysis.py` (317 lines)
  - Data loading and basic statistics
  - Classical meta-analysis (DerSimonian-Laird)
  - Heterogeneity statistics (I², Q test, tau²)
  - Simulation for heterogeneity paradox
  - Publication bias tests (Egger, Begg)
  - Influence analysis (leave-one-out)

**Key Functions**:
- `load_data()`: Read and validate CSV
- `classical_meta_analysis()`: Pooled effect, I², tau²
- `heterogeneity_paradox_simulation()`: Demonstrate I²=0% with low power
- `publication_bias_tests()`: Egger regression, Begg correlation
- `leave_one_out_analysis()`: Influence diagnostics

**Outputs**:
- `findings.md`: Detailed analysis report
- `visualizations/`: 10 plots including forest plot, funnel plot, heterogeneity paradox

**Runtime**: ~2 minutes

### Analyst #2: Uncertainty & Patterns

**Directory**: `/workspace/eda/analyst_2/`

**Code Files**:
- `eda_patterns.py` (289 lines)
  - Signal-to-noise ratios
  - Precision weighting effects
  - Temporal/ordering patterns
  - Weighted vs unweighted pooling

**Key Functions**:
- `calculate_snr()`: Effect size / SE ratios
- `precision_analysis()`: Inverse-variance weighting
- `correlation_analysis()`: Effect vs precision, temporal trends

**Outputs**:
- `findings.md`: Patterns and uncertainty analysis
- `visualizations/`: 6 plots including precision-effect scatter, SNR analysis

**Runtime**: ~1.5 minutes

### Analyst #3: Structure & Context

**Directory**: `/workspace/eda/analyst_3/`

**Code Files**:
- `eda_structure.py` (271 lines)
  - Data quality checks
  - Sample size adequacy assessment
  - Study sequence analysis
  - Meta-analysis standards comparison

**Key Functions**:
- `data_quality_checks()`: Missing values, outliers, implausible values
- `assess_sample_size()`: Power analysis, minimum J requirements
- `sequence_analysis()`: Temporal ordering, progression patterns

**Outputs**:
- `findings.md`: Quality and structure assessment
- `visualizations/`: 6 plots including comprehensive summary, boxplots

**Runtime**: ~1.5 minutes

### EDA Synthesis

**File**: `/workspace/eda/eda_report.md` (459 lines)
- Synthesizes findings from all three analysts
- Identifies key patterns: I²=0% paradox, Study 1 influence, borderline significance
- Recommends Bayesian hierarchical model
- 30+ visualizations across analysts

---

## Phase 2: Model Design

**File**: `/workspace/experiments/experiment_plan.md` (645 lines)

**Content**:
- Synthesis of 3 independent model designers (9 proposed models)
- Selection of 4 models for implementation
- Complete mathematical specifications
- Pre-specified falsification criteria
- Validation pipeline description
- Sampling configurations

**Key Sections**:
- Model 1: Bayesian Hierarchical (Normal) - PRIMARY
- Model 2: Robust Hierarchical (Student-t)
- Model 3: Fixed-Effects
- Model 4: Precision-Stratified
- Falsification criteria for each model
- Iteration strategy

**Output**: Rigorous experiment plan with pre-specified decision rules

---

## Phase 3: Model Development - Experiment 1

### Directory Structure

```
/workspace/experiments/experiment_1/
├── prior_predictive_check/
├── simulation_based_validation/
├── posterior_inference/
├── posterior_predictive_check/
└── model_critique/
```

### Stage 1: Prior Predictive Check

**Directory**: `/workspace/experiments/experiment_1/prior_predictive_check/`

**Code**: `prior_predictive_check.py` (243 lines)
- Samples from prior distributions
- Generates synthetic datasets
- Checks if priors generate plausible data
- Identifies heavy tail issue (3% extreme)

**Key Functions**:
- `generate_prior_samples()`: Sample mu, tau from priors
- `prior_predictive_dataset()`: Generate y_i from prior predictive
- `check_plausibility()`: Assess if synthetic data reasonable

**Outputs**:
- `findings.md`: Prior predictive assessment
- `prior_samples.csv`: 10,000 prior samples
- `plots/`: Prior distributions, predictive datasets

**Decision**: CONDITIONAL PASS (3% heavy tail acceptable)

**Runtime**: ~1 minute

### Stage 2: Simulation-Based Validation

**Directory**: `/workspace/experiments/experiment_1/simulation_based_validation/`

**Code**: `simulation_validation.py` (326 lines)
- Simulation-based calibration (SBC)
- Parameter recovery assessment
- Coverage analysis for mu and tau
- Identifies well-calibrated inference

**Key Functions**:
- `simulate_dataset()`: Generate data from known parameters
- `fit_to_simulation()`: Fit model to synthetic data
- `assess_recovery()`: Compare posterior to true values
- `coverage_analysis()`: Check credible interval coverage

**Outputs**:
- `recovery_metrics.md`: Coverage statistics
- `simulation_results.csv`: 100 simulation replications
- `plots/`: Recovery plots, coverage diagnostics

**Key Results**:
- mu recovery: 90% coverage for 90% CI (excellent)
- tau recovery: 95% coverage for 90% CI (slightly over-covered, conservative)

**Decision**: PASS

**Runtime**: ~15 minutes (100 replications)

### Stage 3: Posterior Inference

**Directory**: `/workspace/experiments/experiment_1/posterior_inference/`

**Code**: `fit_model_pymc.py` (387 lines)
- PyMC model implementation
- NUTS sampling with 4 chains
- Convergence diagnostics
- Posterior summaries and visualizations

**Model Implementation**:
```python
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=50)
    tau = pm.HalfCauchy('tau', beta=5)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)
    y_rep = pm.Normal('y_rep', mu=theta, sigma=sigma, shape=J)
```

**Key Functions**:
- `build_model()`: Construct PyMC model
- `run_sampling()`: Execute NUTS with specified config
- `convergence_diagnostics()`: R-hat, ESS, divergences
- `posterior_summaries()`: Mean, median, CI for all parameters

**Outputs**:
- `inference_summary.md`: Posterior estimates, diagnostics
- `diagnostics/posterior_inference.netcdf`: ArviZ InferenceData object
- `posterior_samples.csv`: 4000 samples × all parameters
- `plots/`: 14 diagnostic plots (trace, forest, pair, convergence)

**Key Results**:
- R-hat = 1.000 (all parameters)
- ESS > 2000 (all parameters)
- 0 divergences
- Runtime: 40 seconds

**Decision**: PERFECT CONVERGENCE

**Runtime**: ~1 minute (including plotting)

### Stage 4: Posterior Predictive Check

**Directory**: `/workspace/experiments/experiment_1/posterior_predictive_check/`

**Code**: `posterior_predictive_check.py` (312 lines)
- Generate replicated datasets from posterior
- Compare observed to replicated data
- Compute p-values for each study
- Check for outliers

**Key Functions**:
- `generate_ppc_samples()`: Sample y_rep from posterior predictive
- `compute_ppc_pvalues()`: P(y_rep ≥ y_obs | posterior)
- `identify_outliers()`: Studies outside 95% PPI

**Outputs**:
- `ppc_findings.md`: Posterior predictive assessment
- `ppc_results.csv`: P-values and intervals for each study
- `plots/`: 5 PPC visualizations

**Key Results**:
- 0/8 studies outside 95% PPI (excellent fit)
- Minimum p-value: 0.244 (Study 1)
- All p-values > 0.05 threshold

**Decision**: PASS (falsification criterion: reject if >1 outlier)

**Runtime**: ~30 seconds

### Stage 5: Model Critique

**Directory**: `/workspace/experiments/experiment_1/model_critique/`

**Code**: `falsification_tests.py` (441 lines)
- Apply all 4 pre-specified falsification criteria
- Leave-one-out cross-validation
- Shrinkage diagnostics
- Prior-posterior checks

**Key Functions**:
- `test_ppc_outliers()`: Count studies outside 95% PPI
- `test_loo_stability()`: Max |Delta mu| across leave-one-out
- `test_convergence()`: R-hat, ESS, divergence thresholds
- `test_extreme_shrinkage()`: |theta_i - y_i| vs 3*sigma_i
- `test_prior_posterior_conflict()`: Tail probability comparison
- `test_identifiability()`: Posterior density variation

**Outputs**:
- `critique_summary.md`: Comprehensive critique
- `decision.md`: ACCEPT decision with rationale (380 lines)
- `falsification_results.json`: Structured test results
- `improvement_priorities.md`: Notes on optional enhancements
- `plots/`: 4 diagnostic plots (LOO, shrinkage, prior-posterior, Pareto k)

**Key Results**:
- Test 1 (PPC outliers): 0/8, threshold 1 → PASS
- Test 2 (LOO stability): max Delta = 2.09, threshold 5.0 → PASS
- Test 3 (Convergence): R-hat=1.00, ESS>2000, 0 div → PASS
- Test 4 (Shrinkage): max = 18.75, threshold 45.0 → PASS
- No revision criteria triggered

**Decision**: **ACCEPT MODEL**

**Runtime**: ~2 minutes

---

## Phase 4: Model Assessment

**Directory**: `/workspace/experiments/model_assessment/`

**Code**: `model_assessment.py` (489 lines)
- LOO-CV diagnostics (Pareto k, ELPD)
- LOO-PIT calibration assessment
- Predictive performance metrics (RMSE, MAE)
- Interval coverage analysis
- Study-level diagnostics

**Key Functions**:
- `compute_loo_cv()`: Leave-one-out cross-validation
- `pareto_k_diagnostics()`: Assess LOO reliability
- `loo_pit_calibration()`: Uniformity test
- `predictive_metrics()`: RMSE, MAE vs baseline
- `interval_coverage()`: 50% and 90% CI coverage rates

**Outputs**:
- `assessment_report.md`: Comprehensive assessment (586 lines)
- `loo_results.csv`: Study-level LOO diagnostics
- `calibration_metrics.json`: LOO-PIT statistics
- `assessment_summary.json`: All metrics structured
- `plots/`: 6 assessment visualizations

**Key Results**:
- All Pareto k < 0.7 (excellent LOO reliability)
- LOO-PIT KS test p = 0.975 (well-calibrated)
- RMSE improvement: 8.7% vs baseline
- **Limitation identified**: 90% CI undercoverage (75% observed)

**Decision**: **ADEQUATE** with documented limitation

**Runtime**: ~3 minutes

---

## Phase 5: Adequacy Assessment

**File**: `/workspace/experiments/adequacy_assessment.md` (905 lines)

**Content**:
- Synthesis across all 4 phases (EDA, Design, Development, Assessment)
- Verification of PPL compliance (PyMC, MCMC, InferenceData, log_likelihood)
- Adequacy criteria evaluation (13 criteria, all passed)
- Strengths and limitations analysis
- Comparison to alternative models (why not implemented)
- Known limitations and appropriate use cases
- Final recommendation

**Key Sections**:
1. Executive summary
2. PPL compliance check
3. Modeling journey (models attempted, improvements)
4. Current model performance
5. Decision: ADEQUATE (with rationale)
6. Strengths (6 major strengths)
7. Limitations (6 limitations with severity)
8. Scientific validity assessment
9. Comparison to alternatives
10. Recommended model
11. Appropriate use cases
12. Next steps

**Decision**: **ADEQUATE** - proceed to final reporting

**Confidence**: HIGH for adequacy, MODERATE to HIGH for optimality

---

## Phase 6: Final Reporting

**Directory**: `/workspace/final_report/`

**Main Documents**:

1. **`report.md`** (23 pages, ~15,000 words)
   - Complete publication-ready report
   - 8 main sections + references + supplementary
   - Comprehensive methodology, results, discussion
   - Honest limitations and recommendations

2. **`executive_summary.md`** (2 pages)
   - Standalone summary for stakeholders
   - Key findings, conclusions, recommendations
   - Critical takeaways for different audiences

3. **`key_figures.md`**
   - Guide to 7 primary figures + supplementary
   - Detailed captions explaining what each shows
   - Interpretation guidance
   - Figure usage recommendations

4. **`model_specification.md`**
   - Complete mathematical specification
   - Prior justifications
   - Non-centered parameterization details
   - Sampling configuration
   - Reproducibility checklist

5. **`code_archive.md`** (this document)
   - Navigation guide to all code
   - File-by-file descriptions
   - Key functions and outputs
   - Runtime estimates

**Figures** (`figures/` subdirectory):
- fig1_forest_plot.png
- fig2_posterior_distributions.png
- fig3_prior_posterior_tau.png
- fig4_posterior_predictive.png
- fig5_loo_diagnostics.png
- fig6_calibration.png
- fig7_shrinkage.png

---

## Key Code Files Summary

### Most Important Files (For Reproduction)

**Data**:
- `/workspace/data/data.csv` (8 studies, 3 columns)

**Model Fitting**:
- `/workspace/experiments/experiment_1/posterior_inference/code/fit_model_pymc.py`
  - Complete PyMC implementation
  - Run this to reproduce main results

**Validation**:
- `/workspace/experiments/experiment_1/simulation_based_validation/simulation_validation.py`
  - Parameter recovery validation

**Assessment**:
- `/workspace/experiments/model_assessment/model_assessment.py`
  - LOO-CV and calibration

**Complete Workflow** (sequential execution):
1. EDA (3 analysts in parallel): `eda/analyst_*/eda_*.py`
2. Model design: Read `experiment_plan.md`
3. Prior predictive: `experiment_1/prior_predictive_check/prior_predictive_check.py`
4. Simulation: `experiment_1/simulation_based_validation/simulation_validation.py`
5. Fitting: `experiment_1/posterior_inference/code/fit_model_pymc.py`
6. PPC: `experiment_1/posterior_predictive_check/posterior_predictive_check.py`
7. Critique: `experiment_1/model_critique/falsification_tests.py`
8. Assessment: `model_assessment/model_assessment.py`

**Total Runtime**: ~30-40 minutes for complete workflow

---

## Dependencies and Environment

### Python Requirements

**Core**:
- Python >= 3.11
- PyMC >= 5.26.1
- ArviZ >= 0.19.0
- NumPy >= 1.26.0
- Pandas >= 2.2.0

**Visualization**:
- Matplotlib >= 3.8.0
- Seaborn >= 0.13.0

**Statistical**:
- SciPy >= 1.11.0
- Statsmodels >= 0.14.0

**Utilities**:
- tqdm (progress bars)
- json, csv (data I/O)

### Installation

```bash
pip install pymc==5.26.1 arviz==0.19.0 numpy pandas matplotlib seaborn scipy statsmodels tqdm
```

Or use `requirements.txt` if provided.

### Hardware Requirements

**Minimal**:
- CPU: Any modern processor
- RAM: 8 GB (16 GB recommended)
- Storage: 1 GB (for outputs)
- GPU: Not required

**Runtime**: All analyses run in minutes on standard hardware

---

## Data Files

### Input Data

**File**: `/workspace/data/data.csv`

**Format**:
```csv
study_id,y,sigma
1,28,15
2,8,10
3,-3,16
4,7,11
5,-1,9
6,1,11
7,18,10
8,12,18
```

**Columns**:
- `study_id`: Integer 1-8 (study identifier)
- `y`: Float (observed effect size)
- `sigma`: Float > 0 (known standard error)

**Rows**: 8 (one per study)

**Size**: < 1 KB

### Output Data

**Posterior Samples**:
- `/workspace/experiments/experiment_1/posterior_inference/posterior_samples.csv`
- 4000 rows × (2 + 8 + 1) = 11 columns
- Columns: mu, tau, theta[1:8], log_likelihood
- Size: ~1 MB

**InferenceData** (Preferred):
- `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- NetCDF format (ArviZ standard)
- Includes: posterior, posterior_predictive, log_likelihood, sample_stats, observed_data
- Size: ~2 MB

**LOO Results**:
- `/workspace/experiments/model_assessment/loo_results.csv`
- 8 rows (one per study)
- Columns: study, y_obs, sigma, theta_mean, theta_sd, CI_90_lower, CI_90_upper, residual, std_residual, pareto_k, loo_pit
- Size: < 1 KB

---

## Visualization Code

All plots generated using custom functions in respective analysis scripts.

**Key Plotting Functions**:

**EDA**:
- `plot_forest()`: Forest plot with CIs
- `plot_funnel()`: Funnel plot for publication bias
- `plot_heterogeneity_paradox()`: Simulation demonstration

**Posterior Inference**:
- `plot_trace()`: MCMC trace plots
- `plot_forest_shrinkage()`: Observed vs posterior estimates
- `plot_posterior_distributions()`: Density plots for mu, tau
- `plot_pair()`: Joint distribution of mu, tau

**Validation**:
- `plot_recovery()`: Posterior vs true parameters
- `plot_coverage()`: Calibration diagnostics

**Assessment**:
- `plot_pareto_k()`: LOO reliability
- `plot_loo_pit()`: Calibration uniformity
- `plot_predicted_vs_observed()`: Model fit

**Common Settings**:
- Figure size: (10, 8) or (12, 6) for panels
- DPI: 100 (high-resolution)
- Color palette: Matplotlib defaults or Seaborn (colorblind-safe)
- Fonts: Standard matplotlib (adjustable)

---

## Reproducibility Notes

### Random Seeds

**All analyses use seed = 12345** for reproducibility:
- EDA: Simulations, bootstrap
- Simulation validation: Data generation, sampling
- Posterior inference: MCMC sampling
- Posterior predictive: y_rep generation

**Setting seed**:
```python
import numpy as np
np.random.seed(12345)

# For PyMC
pm.sample(..., random_seed=12345)
```

### Version Control

**Recommended**:
- Git repository for code
- Track all `.py` and `.md` files
- Exclude large binary files (`.netcdf`, plots)
- Include `requirements.txt`

**Git ignore**:
```
*.netcdf
*.csv
*.png
*.jpg
__pycache__/
.ipynb_checkpoints/
```

### Computational Reproducibility

**Expected Results** (Monte Carlo error):
- mu posterior mean: 7.75 ± 0.1
- mu 95% CI: [-1.2, 16.5] ± 0.2
- tau posterior mean: 2.86 ± 0.2
- tau 95% CI: [0.1, 11.3] ± 0.5
- R-hat: 1.000 ± 0.001
- ESS: 2000 ± 100

**Verification**: Rerun should match these values within stated tolerances.

---

## Testing and Quality Assurance

### Automated Tests (Recommended)

**Unit Tests**:
- Data loading and validation
- Prior and likelihood functions
- Posterior summary calculations
- Diagnostic computations

**Integration Tests**:
- End-to-end workflow execution
- Convergence verification
- Falsification criteria evaluation

**Regression Tests**:
- Compare outputs to reference results
- Ensure reproducibility across runs

### Manual Checks

**Code Review Checklist**:
- [ ] All functions documented
- [ ] Error handling present
- [ ] Edge cases considered
- [ ] Random seeds set
- [ ] Outputs saved correctly
- [ ] Plots generated and saved

**Analysis Review Checklist**:
- [ ] Data quality verified
- [ ] Model correctly specified
- [ ] Priors justified
- [ ] Convergence achieved
- [ ] Diagnostics run
- [ ] Limitations documented

---

## Common Issues and Troubleshooting

### Installation Issues

**Problem**: PyMC installation fails
**Solution**: Use conda instead of pip
```bash
conda install -c conda-forge pymc
```

**Problem**: ArviZ version mismatch
**Solution**: Ensure ArviZ >= 0.19.0
```bash
pip install --upgrade arviz
```

### Runtime Issues

**Problem**: MCMC sampling slow
**Solution**:
- Check for divergences (increase `target_accept`)
- Verify non-centered parameterization used
- Reduce number of samples if testing

**Problem**: Out of memory
**Solution**:
- Reduce number of chains (try 2 instead of 4)
- Reduce samples (try 500 instead of 1000)
- Close other applications

**Problem**: Convergence failure (R-hat > 1.01)
**Solution**:
- Increase warmup iterations (try 2000)
- Increase `target_accept` to 0.99
- Check for model misspecification

### Analysis Issues

**Problem**: Different posterior estimates
**Cause**: Monte Carlo error, different random seed
**Solution**: Results should match within tolerances (see Reproducibility Notes)

**Problem**: Divergences appear
**Cause**: Posterior geometry issues
**Solution**: Already using non-centered parameterization; increase `target_accept`

**Problem**: LOO Pareto k > 0.7
**Cause**: Influential observations
**Solution**: Model already handles this (all k < 0.7 in our analysis)

---

## Citation and Licensing

### Code Attribution

**Primary Author**: Bayesian Modeling Workflow Team
**Date**: October 28, 2025
**License**: (Specify open-source license if applicable, e.g., MIT, Apache 2.0)

### Software Citations

**PyMC**:
Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.

**ArviZ**:
Kumar, R., et al. (2019). ArviZ a unified library for exploratory analysis of Bayesian models in Python. *Journal of Open Source Software*, 4(33), 1143.

**NumPy, SciPy, Matplotlib, Pandas**: See respective documentation

---

## Contact and Support

**Questions about code**: Refer to inline comments in individual files

**Questions about methodology**: See `/workspace/final_report/report.md` Section 3 (Methods)

**Questions about interpretation**: See `/workspace/final_report/executive_summary.md`

**Reproducibility issues**: Check Reproducibility Notes section above

---

## Summary Table: All Code Files

| File | Lines | Purpose | Runtime |
|------|-------|---------|---------|
| `eda/analyst_1/eda_analysis.py` | 317 | Heterogeneity analysis | 2 min |
| `eda/analyst_2/eda_patterns.py` | 289 | Uncertainty patterns | 1.5 min |
| `eda/analyst_3/eda_structure.py` | 271 | Quality checks | 1.5 min |
| `experiment_1/prior_predictive_check/prior_predictive_check.py` | 243 | Prior validation | 1 min |
| `experiment_1/simulation_based_validation/simulation_validation.py` | 326 | Parameter recovery | 15 min |
| `experiment_1/posterior_inference/code/fit_model_pymc.py` | 387 | Model fitting | 1 min |
| `experiment_1/posterior_predictive_check/posterior_predictive_check.py` | 312 | PPC | 30 sec |
| `experiment_1/model_critique/falsification_tests.py` | 441 | Falsification tests | 2 min |
| `model_assessment/model_assessment.py` | 489 | LOO-CV, calibration | 3 min |
| **Total** | **3075** | **Full workflow** | **~30 min** |

---

**Code Archive Prepared**: October 28, 2025
**Status**: Complete
**Purpose**: Navigation and reproducibility guide for Bayesian meta-analysis project
