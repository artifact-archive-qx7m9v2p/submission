# Reproducibility Guide: Eight Schools Bayesian Analysis

**Project:** Eight Schools Meta-Analysis
**Date:** October 28, 2025
**Status:** Complete and Reproducible

---

## Overview

This guide provides complete information for reproducing all analyses in the Eight Schools Bayesian modeling project. All code, data, and computational environment specifications are documented here.

---

## Quick Start

### Minimum Requirements
- Python 3.x
- PyMC 5.26.1
- ArviZ 0.22.0
- Standard scientific Python stack (NumPy, Pandas, Matplotlib, SciPy)

### Installation
```bash
# Create virtual environment (recommended)
python -m venv eight_schools_env
source eight_schools_env/bin/activate  # On Windows: eight_schools_env\Scripts\activate

# Install requirements
pip install pymc==5.26.1 arviz==0.22.0 pandas numpy matplotlib seaborn scipy
```

### Run Complete Analysis
```bash
# Navigate to project root
cd /workspace

# Run EDA
cd eda/code
python 01_initial_exploration.py
python 02_visualizations.py

# Run models
cd ../../experiments/experiment_1/posterior_inference/code
python fit_model.py

cd ../../../experiment_2/posterior_inference/code
python fit_model.py

# Run comparison
cd ../../../model_comparison/code
python comprehensive_assessment_v2.py
```

---

## Data

### Source Data

**File:** `/workspace/data/data.csv`

**Format:** CSV with 3 columns
- `school`: School identifier (1-8)
- `y`: Observed treatment effect
- `sigma`: Known standard error

**Content:**
```csv
school,y,sigma
1,28,15
2,8,10
3,-3,16
4,7,11
5,-1,9
6,1,11
7,18,10
8,12,18
```

**Provenance:**
- Original source: Rubin (1981)
- Standard dataset in Bayesian literature
- Included in Gelman et al. (2013) BDA3
- No preprocessing required

**Data Quality:**
- No missing values
- No duplicates
- All values within plausible ranges
- Standard errors are known (not estimated)

---

## Software Environment

### Operating System
- **Platform:** Linux
- **Version:** 6.14.0-33-generic
- **Architecture:** x86_64

### Python Environment

**Python Version:** 3.x (system)

**Required Packages:**

| Package | Version | Purpose |
|---------|---------|---------|
| pymc | 5.26.1 | Probabilistic programming, MCMC sampling |
| arviz | 0.22.0 | Diagnostics, visualization, InferenceData |
| numpy | (system) | Numerical computing |
| pandas | (system) | Data manipulation |
| matplotlib | (system) | Plotting |
| seaborn | (system) | Statistical visualization |
| scipy | (system) | Statistical functions |

**Installation via pip:**
```bash
pip install pymc==5.26.1 arviz==0.22.0 pandas numpy matplotlib seaborn scipy
```

**Installation via conda:**
```bash
conda install -c conda-forge pymc=5.26.1 arviz=0.22.0 pandas numpy matplotlib seaborn scipy
```

### Dependencies

PyMC 5.26.1 automatically installs:
- PyTensor (computational backend)
- NumPy (>= 1.15.0)
- SciPy (>= 1.4.1)
- Typing-extensions

ArviZ 0.22.0 automatically installs:
- xarray (for InferenceData)
- netCDF4 (for file I/O)
- Matplotlib, Pandas, NumPy

---

## Computational Resources

### Hardware Requirements

**Minimum:**
- CPU: 1 core (functional but slow)
- RAM: 2 GB
- Disk: 100 MB

**Recommended:**
- CPU: 4+ cores (for parallel chains)
- RAM: 8 GB
- Disk: 1 GB (for all outputs)

**Used in This Analysis:**
- CPU: Multi-core (4 parallel chains)
- RAM: Sufficient for 8000 samples
- Runtime: < 1 minute total compute time

### Parallelization

**MCMC Sampling:**
- 4 independent chains run in parallel
- One chain per CPU core
- Linear speedup with cores (4x faster than sequential)

**Configuration in PyMC:**
```python
with pm.Model() as model:
    # ... model specification ...
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,  # Number of parallel chains
        cores=4,   # Number of CPU cores
        random_seed=42
    )
```

---

## Random Seeds

### Reproducibility via Seeds

All analyses use **fixed random seed = 42** for reproducibility.

**Set at script initialization:**
```python
import numpy as np
import pymc as pm

# Set global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

**In PyMC sampling:**
```python
idata = pm.sample(
    draws=1000,
    tune=1000,
    random_seed=RANDOM_SEED  # Ensures reproducible MCMC samples
)
```

**Expected behavior:**
- Identical posterior samples across runs
- Identical convergence diagnostics
- Identical visualizations

**Note:** Exact reproducibility requires:
- Same PyMC version (5.26.1)
- Same random seed (42)
- Same sampling configuration (draws, tune, chains)

---

## File Organization

### Project Structure

```
/workspace/
├── data/
│   └── data.csv                          # Source data
│
├── eda/                                  # Exploratory Data Analysis
│   ├── eda_report.md                     # Main EDA report
│   ├── EXECUTIVE_SUMMARY.md              # EDA summary
│   ├── visualizations/                   # 6 diagnostic plots
│   │   ├── forest_plot.png
│   │   ├── distribution_analysis.png
│   │   ├── effect_vs_uncertainty.png
│   │   ├── precision_analysis.png
│   │   ├── school_profiles.png
│   │   └── heterogeneity_diagnostics.png
│   └── code/                             # EDA scripts
│       ├── 01_initial_exploration.py
│       ├── 02_visualizations.py
│       └── 03_hypothesis_testing.py
│
├── experiments/
│   ├── experiment_plan.md                # Synthesized model design
│   │
│   ├── experiment_1/                     # Hierarchical model
│   │   ├── metadata.md
│   │   ├── prior_predictive_check/
│   │   │   └── findings.md
│   │   ├── simulation_based_validation/
│   │   │   └── recovery_metrics.md
│   │   ├── posterior_inference/
│   │   │   ├── inference_summary.md
│   │   │   ├── code/fit_model.py
│   │   │   ├── diagnostics/
│   │   │   │   ├── posterior_inference.netcdf  # InferenceData
│   │   │   │   └── convergence_diagnostics.txt
│   │   │   └── plots/                   # 7 diagnostic plots
│   │   ├── posterior_predictive_check/
│   │   │   └── ppc_findings.md
│   │   └── model_critique/
│   │       ├── critique_summary.md
│   │       └── decision.md
│   │
│   ├── experiment_2/                     # Complete pooling model
│   │   ├── metadata.md
│   │   └── posterior_inference/
│   │       ├── inference_summary.md
│   │       ├── code/fit_model.py
│   │       ├── diagnostics/
│   │       │   ├── posterior_inference.netcdf  # InferenceData
│   │       │   └── convergence_diagnostics.txt
│   │       └── plots/                   # 6 diagnostic plots
│   │
│   └── model_comparison/                 # Model comparison
│       ├── comparison_report.md
│       ├── recommendation.md
│       ├── ASSESSMENT_SUMMARY.md
│       ├── code/comprehensive_assessment_v2.py
│       ├── loo_comparison.csv
│       └── figures/                      # 4 comparison plots
│           ├── loo_comparison_plot.png
│           ├── pareto_k_comparison.png
│           ├── prediction_comparison.png
│           └── pointwise_loo_comparison.png
│
├── final_report/                         # Final report (this phase)
│   ├── report.md                         # Main comprehensive report
│   ├── executive_summary.md              # 2-3 page summary
│   ├── README.md                         # Report navigation
│   ├── figures/                          # All key visualizations
│   └── supplementary/
│       ├── technical_appendix.md
│       ├── model_development.md
│       └── reproducibility.md            # This file
│
└── log.md                                # Project timeline
```

### Key Files

**Data:**
- `/workspace/data/data.csv` - Original Eight Schools dataset

**EDA Outputs:**
- `/workspace/eda/eda_report.md` - 713 lines, comprehensive analysis
- `/workspace/eda/visualizations/*.png` - 6 publication-quality plots

**Model Outputs:**
- Experiment 1: `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Experiment 2: `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`

**Comparison:**
- `/workspace/experiments/model_comparison/comparison_report.md` - Detailed comparison
- `/workspace/experiments/model_comparison/loo_comparison.csv` - LOO results table

**Final Report:**
- `/workspace/final_report/report.md` - Main report (15-25 pages)
- `/workspace/final_report/executive_summary.md` - Summary (2-3 pages)

---

## Reproducing Each Phase

### Phase 1: Exploratory Data Analysis

**Scripts:**
```bash
cd /workspace/eda/code

# Initial exploration and classical meta-analysis
python 01_initial_exploration.py

# Generate visualizations
python 02_visualizations.py

# Hypothesis testing
python 03_hypothesis_testing.py
```

**Expected Runtime:** 1-2 minutes
**Outputs:**
- `eda_report.md`
- 6 PNG plots in `visualizations/`

**Key Findings to Verify:**
- I² = 0%
- Q test p = 0.696
- tau² = 0
- Pooled mean ≈ 7.69

### Phase 2: Model Design

**No code to run** - synthesis of designer proposals

**Review:**
- `/workspace/experiments/experiment_plan.md`

### Phase 3: Experiment 1 (Hierarchical Model)

**Script:**
```bash
cd /workspace/experiments/experiment_1/posterior_inference/code
python fit_model.py
```

**Expected Runtime:** 15-20 seconds
**Outputs:**
- `posterior_inference.netcdf` (2.6 MB)
- Convergence diagnostics (text file)
- 7 diagnostic plots

**Key Results to Verify:**
- μ ≈ 7.36 ± 4.32
- τ ≈ 3.58 ± 3.15
- R-hat = 1.000 (all parameters)
- 0 divergences

### Phase 4: Experiment 2 (Complete Pooling)

**Script:**
```bash
cd /workspace/experiments/experiment_2/posterior_inference/code
python fit_model.py
```

**Expected Runtime:** 1-2 seconds
**Outputs:**
- `posterior_inference.netcdf` (758 KB)
- Convergence diagnostics (text file)
- 6 diagnostic plots

**Key Results to Verify:**
- μ ≈ 7.55 ± 4.00
- R-hat = 1.000
- 0 divergences

### Phase 5: Model Comparison

**Script:**
```bash
cd /workspace/experiments/model_comparison/code
python comprehensive_assessment_v2.py
```

**Expected Runtime:** 5-10 seconds
**Outputs:**
- `comparison_report.md`
- `loo_comparison.csv`
- 4 comparison plots

**Key Results to Verify:**
- ΔELPD = 0.21 ± 0.11
- ΔELPD < 2×SE (models equivalent)
- Complete pooling selected

---

## InferenceData Files

### Format: NetCDF

All posterior samples stored as ArviZ InferenceData in NetCDF format.

**Files:**
- Experiment 1: `posterior_inference.netcdf` (2.6 MB)
- Experiment 2: `posterior_inference.netcdf` (758 KB)

### Loading InferenceData

```python
import arviz as az

# Load posterior samples
idata = az.from_netcdf('posterior_inference.netcdf')

# Access posterior group
print(idata.posterior)

# Access specific parameter
mu_samples = idata.posterior['mu'].values  # shape: (chains, draws)

# Compute summary statistics
summary = az.summary(idata, var_names=['mu', 'tau'])
print(summary)
```

### InferenceData Groups

**Experiment 1 (Hierarchical):**
- `posterior`: μ, τ, θ[1:8], η[1:8]
- `posterior_predictive`: Not saved
- `log_likelihood`: log p(y_i | θ) for LOO-CV
- `observed_data`: y[1:8]

**Experiment 2 (Complete Pooling):**
- `posterior`: μ
- `log_likelihood`: log p(y_i | μ) for LOO-CV
- `observed_data`: y[1:8]

---

## Verification Checklist

### Data Integrity
- [ ] `data.csv` has 8 rows, 3 columns
- [ ] No missing values
- [ ] y ranges from -3 to 28
- [ ] sigma ranges from 9 to 18

### EDA Results
- [ ] I² = 0.0%
- [ ] Q = 4.71, p = 0.696
- [ ] tau² = 0.00
- [ ] Pooled mean ≈ 7.69 ± 4.07
- [ ] 6 plots generated

### Experiment 1 (Hierarchical)
- [ ] R-hat = 1.000 for all parameters
- [ ] ESS > 400 for all parameters
- [ ] 0 divergences
- [ ] μ ≈ 7.36 ± 4.32
- [ ] τ ≈ 3.58 ± 3.15
- [ ] LOO ELPD ≈ -30.73 ± 1.04

### Experiment 2 (Complete Pooling)
- [ ] R-hat = 1.000
- [ ] ESS > 400
- [ ] 0 divergences
- [ ] μ ≈ 7.55 ± 4.00
- [ ] LOO ELPD ≈ -30.52 ± 1.12

### Model Comparison
- [ ] ΔELPD = 0.21 ± 0.11
- [ ] ΔELPD < 2×SE (0.22)
- [ ] Complete pooling selected
- [ ] All Pareto k < 0.7

---

## Common Issues and Solutions

### Issue 1: PyMC Version Mismatch

**Symptom:** Different posterior estimates or convergence issues

**Solution:**
```bash
# Check version
python -c "import pymc; print(pymc.__version__)"

# Should print: 5.26.1

# If wrong version, reinstall
pip uninstall pymc
pip install pymc==5.26.1
```

### Issue 2: Random Seed Not Working

**Symptom:** Different results on each run

**Check:**
```python
# Verify seed is set in script
import numpy as np
np.random.seed(42)

# Verify seed passed to sampler
idata = pm.sample(..., random_seed=42)
```

**Note:** Even with same seed, different PyMC versions may give different samples.

### Issue 3: Slow Sampling

**Symptom:** Sampling takes > 1 minute

**Solutions:**
- Reduce draws: `draws=500` instead of 1000
- Reduce chains: `chains=2` instead of 4
- Check CPU usage (should use all cores)

### Issue 4: InferenceData Load Failure

**Symptom:** Cannot load .netcdf files

**Solution:**
```bash
# Install netCDF4
pip install netCDF4

# Try loading
python -c "import arviz as az; idata = az.from_netcdf('posterior_inference.netcdf')"
```

### Issue 5: Plotting Errors

**Symptom:** Matplotlib errors when generating plots

**Solution:**
```bash
# Install plotting backends
pip install matplotlib seaborn

# Set backend if needed
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

---

## Performance Benchmarks

### Expected Runtimes

| Task | Expected Runtime | Hardware |
|------|------------------|----------|
| EDA scripts | 1-2 minutes | Any modern CPU |
| Hierarchical fit | 15-20 seconds | 4 cores |
| Complete pooling fit | 1-2 seconds | 4 cores |
| Model comparison | 5-10 seconds | Any |
| Total analysis | < 1 minute | 4 cores |

### Memory Usage

| Component | Memory Usage |
|-----------|--------------|
| Hierarchical samples | ~50 MB (in memory) |
| Complete pooling samples | ~20 MB (in memory) |
| InferenceData files | 2.6 MB + 758 KB (on disk) |
| Plots | ~10 MB (on disk) |
| Total project | < 100 MB |

---

## Testing Reproduction

### Quick Test (5 minutes)

```bash
# 1. Load data
cd /workspace
python -c "import pandas as pd; df = pd.read_csv('data/data.csv'); print(df)"

# 2. Fit complete pooling model (fastest)
cd experiments/experiment_2/posterior_inference/code
python fit_model.py

# 3. Check convergence
python -c "import arviz as az; idata = az.from_netcdf('../diagnostics/posterior_inference.netcdf'); print(az.summary(idata))"
```

**Expected output:** μ ≈ 7.55, R-hat = 1.0

### Full Test (10 minutes)

Run all scripts in sequence:
1. EDA (2 min)
2. Experiment 1 (20 sec)
3. Experiment 2 (2 sec)
4. Comparison (10 sec)

---

## Citation and Attribution

### Data Source
```
Rubin, D. B. (1981). Estimation in parallel randomized experiments.
Journal of Educational Statistics, 6(4), 377-401.
```

### Software Citation

**PyMC:**
```
PyMC Development Team (2024). PyMC: Bayesian Modeling in Python.
Version 5.26.1. https://www.pymc.io
```

**ArviZ:**
```
Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. A. (2019).
ArviZ a unified library for exploratory analysis of Bayesian models
in Python. Journal of Open Source Software, 4(33), 1143.
```

### This Analysis
```
Eight Schools Bayesian Meta-Analysis. (2025).
Complete Bayesian workflow with model comparison.
```

---

## Support and Contact

### Questions About Reproduction

If you encounter issues reproducing this analysis:

1. **Check versions:** Ensure PyMC 5.26.1, ArviZ 0.22.0
2. **Check seed:** Verify random_seed=42 in all scripts
3. **Check data:** Verify data.csv is identical
4. **Review this guide:** All steps documented above

### Known Limitations

**Approximate Reproducibility:**
- Different hardware may have minor numerical differences
- Different OS may affect random number generation
- These should be < 1% difference in estimates

**Exact Reproducibility Requires:**
- Same PyMC version (5.26.1)
- Same random seed (42)
- Same sampling configuration

---

## Archive and Preservation

### Long-Term Storage

**Recommended archive contents:**
```
eight_schools_analysis.zip
├── data/data.csv
├── code/ (all .py scripts)
├── results/ (InferenceData files)
├── plots/ (all figures)
├── reports/ (markdown files)
└── reproducibility.md (this file)
```

**Minimum for reproduction:**
- Data file
- Code scripts
- This reproducibility guide
- Software versions

---

## License and Sharing

**Data:** Public domain (Rubin 1981)
**Code:** Available for educational and research use
**Reports:** Available for reference and citation

---

## Version History

**Version 1.0 (2025-10-28)**
- Initial analysis complete
- All phases documented
- Reproducibility verified

---

**Document Created:** October 28, 2025
**Last Updated:** October 28, 2025
**Status:** Complete and Tested

**END OF REPRODUCIBILITY GUIDE**
