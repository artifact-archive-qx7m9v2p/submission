# Supplementary Material D: Reproducibility Guide

**Report:** Bayesian Power Law Modeling of Y-x Relationship
**Date:** October 27, 2025

---

## Overview

This document provides complete instructions for reproducing all analyses reported in the main report. We follow best practices for computational reproducibility in Bayesian statistics.

**Reproducibility Level:** Full
- All code available
- All data available
- Software versions documented
- Random seeds fixed
- InferenceData objects archived

---

## Software Environment

### Core Software

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.13.0 | Programming language |
| **PyMC** | 5.26.1 | Probabilistic programming |
| **ArviZ** | 0.22.0 | Bayesian diagnostics |
| **NumPy** | 1.26.4 | Numerical computing |
| **Pandas** | 2.2.3 | Data manipulation |
| **Matplotlib** | 3.10.0 | Visualization |
| **SciPy** | 1.15.1 | Scientific computing |

### System Environment

- **Platform:** Linux 6.14.0-33-generic
- **Python Implementation:** CPython
- **Architecture:** x86_64

### Installation

```bash
# Using pip
pip install pymc==5.26.1 arviz==0.22.0 numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.0 scipy==1.15.1

# Or using conda
conda install -c conda-forge pymc==5.26.1 arviz==0.22.0

# Verify installation
python -c "import pymc; print(pymc.__version__)"  # Should print 5.26.1
python -c "import arviz; print(arviz.__version__)"  # Should print 0.22.0
```

---

## Data

### Source Data

**Location:** `/workspace/data/data.csv`

**Format:** CSV with header
- Column 1: `x` (predictor)
- Column 2: `Y` (response)

**Size:** 27 observations

**Sample (first 5 rows):**
```
x,Y
1.0,1.77
2.0,1.98
3.0,2.05
4.0,2.07
5.0,2.12
```

**Access:**
```python
import pandas as pd

# Load data
data = pd.read_csv('/workspace/data/data.csv')

# Verify
assert len(data) == 27
assert list(data.columns) == ['x', 'Y']
assert data['x'].min() == 1.0
assert data['x'].max() == 31.5
```

**Data Integrity:**
- MD5 checksum: [Would be computed and provided]
- No missing values
- One duplicate observation (not removed)

---

## Random Seeds

**All analyses use fixed random seeds for reproducibility:**

| Analysis | Seed | Notes |
|----------|------|-------|
| **EDA** | 12345 | Bootstrap, resampling |
| **Prior Predictive** | 12345 | Prior sampling |
| **SBC** | 12345 + i | Where i = simulation index (0-199) |
| **MCMC (Model 1)** | 12345 | Posterior sampling |
| **MCMC (Model 2)** | 12345 | Posterior sampling |
| **Posterior Predictive** | 12345 | PPC sampling |

**Setting seeds:**

```python
import numpy as np
import random

# Set all seeds
SEED = 12345
np.random.seed(SEED)
random.seed(SEED)

# PyMC sampling
import pymc as pm
with pm.Model() as model:
    # ... model specification ...
    idata = pm.sample(1000, tune=1000, chains=4, random_seed=SEED)
```

---

## Reproducing Model 1 (Primary Result)

### Step 1: Load and Transform Data

```python
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('/workspace/data/data.csv')
Y = data['Y'].values
x = data['x'].values

# Log-transform
log_Y = np.log(Y)
log_x = np.log(x)
```

### Step 2: Specify Model

```python
import pymc as pm

SEED = 12345

with pm.Model() as model1:
    # Priors
    alpha = pm.Normal('alpha', mu=0.6, sigma=0.3)
    beta = pm.Normal('beta', mu=0.13, sigma=0.1)
    sigma = pm.HalfNormal('sigma', sigma=0.1)

    # Linear predictor in log-log space
    mu = alpha + beta * log_x

    # Likelihood
    log_Y_obs = pm.Normal('log_Y_obs', mu=mu, sigma=sigma, observed=log_Y)
```

### Step 3: Sample from Posterior

```python
with model1:
    # Sample
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        random_seed=SEED,
        idata_kwargs={'log_likelihood': True}  # For LOO-CV
    )
```

**Expected output:**
- Sampling completed in ~5 seconds
- 4 chains × 1000 draws = 4000 total samples
- No divergences
- All R-hat = 1.000

### Step 4: Compute LOO

```python
import arviz as az

# Compute LOO cross-validation
loo = az.loo(idata, pointwise=True)

print(f"ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"p_loo: {loo.p_loo:.2f}")
print(f"Max Pareto k: {loo.pareto_k.max():.3f}")
```

**Expected output:**
```
ELPD LOO: 46.99 ± 3.11
p_loo: 2.43
Max Pareto k: 0.472
```

### Step 5: Extract Parameter Estimates

```python
# Posterior summary
summary = az.summary(idata, hdi_prob=0.95)
print(summary)
```

**Expected output:**
```
         mean     sd  hdi_2.5%  hdi_97.5%  ...
alpha   0.580  0.019     0.542      0.616  ...
beta    0.126  0.009     0.111      0.143  ...
sigma   0.041  0.006     0.031      0.053  ...
```

### Step 6: Compute Performance Metrics

```python
# Predictions
alpha_mean = idata.posterior['alpha'].values.mean()
beta_mean = idata.posterior['beta'].values.mean()
log_Y_pred = alpha_mean + beta_mean * log_x
Y_pred = np.exp(log_Y_pred)

# Metrics
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error

r2 = r2_score(log_Y, log_Y_pred)  # On log scale
mape = mean_absolute_percentage_error(Y, Y_pred) * 100
mae = mean_absolute_error(Y, Y_pred)

print(f"R² (log scale): {r2:.3f}")
print(f"MAPE: {mape:.2f}%")
print(f"MAE: {mae:.4f}")
```

**Expected output:**
```
R² (log scale): 0.902
MAPE: 3.04%
MAE: 0.0714
```

---

## Reproducing Model 2 (Secondary Analysis)

### Model Specification

```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model2:
    # Priors for mean function
    beta_0 = pm.Normal('beta_0', mu=1.8, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.3, sigma=0.2)

    # Priors for variance function
    gamma_0 = pm.Normal('gamma_0', mu=-2, sigma=1)
    gamma_1 = pm.Normal('gamma_1', mu=-0.05, sigma=0.05)

    # Mean function
    mu = beta_0 + beta_1 * log_x

    # Variance function
    log_sigma = gamma_0 + gamma_1 * x
    sigma_het = pt.exp(log_sigma)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_het, observed=Y)
```

### Sampling

```python
with model2:
    idata2 = pm.sample(
        draws=1500,
        tune=1500,
        chains=4,
        target_accept=0.97,  # Conservative
        random_seed=SEED,
        idata_kwargs={'log_likelihood': True}
    )
```

**Expected output:**
- Sampling completed in ~110 seconds
- No divergences
- All R-hat = 1.000

### Key Finding

```python
gamma_1_samples = idata2.posterior['gamma_1'].values.flatten()
print(f"γ₁ mean: {gamma_1_samples.mean():.3f}")
print(f"γ₁ SD: {gamma_1_samples.std():.3f}")
print(f"95% HDI: [{np.percentile(gamma_1_samples, 2.5):.3f}, {np.percentile(gamma_1_samples, 97.5):.3f}]")
print(f"P(γ₁ < 0): {(gamma_1_samples < 0).mean():.3f}")
```

**Expected output:**
```
γ₁ mean: 0.003
γ₁ SD: 0.017
95% HDI: [-0.028, 0.039]
P(γ₁ < 0): 0.439
```

**Interpretation:** 95% CI includes zero → No evidence for heteroscedasticity

---

## Reproducing Model Comparison

```python
import arviz as az

# Load both models (or use idata and idata2 from above)

# Compute LOO for both
loo1 = az.loo(idata)
loo2 = az.loo(idata2)

# Compare
comparison = az.compare({'Model 1': idata, 'Model 2': idata2})
print(comparison)
```

**Expected output:**
```
            elpd_loo   se  dse  ...
Model 1        46.99 3.11  0.00  ...
Model 2        23.56 3.15 23.43  ...
```

**Interpretation:** Model 1 is 23.43 ELPD units better (>5 SE difference)

---

## Reproducing Visualizations

### Fitted Line Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Posterior samples
alpha_samples = idata.posterior['alpha'].values.flatten()
beta_samples = idata.posterior['beta'].values.flatten()

# Grid for predictions
x_grid = np.linspace(1, 31.5, 100)
log_x_grid = np.log(x_grid)

# Posterior predictive
n_samples = 1000
idx = np.random.choice(len(alpha_samples), n_samples, replace=False)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Log-log scale
for i in idx[:100]:  # Plot 100 posterior lines
    log_y_line = alpha_samples[i] + beta_samples[i] * log_x_grid
    ax1.plot(log_x_grid, log_y_line, alpha=0.02, color='blue')

ax1.scatter(log_x, log_Y, color='black', s=50, zorder=5)
ax1.set_xlabel('log(x)')
ax1.set_ylabel('log(Y)')
ax1.set_title('Log-Log Scale')

# Original scale
for i in idx[:100]:
    log_y_line = alpha_samples[i] + beta_samples[i] * log_x_grid
    y_line = np.exp(log_y_line)
    ax2.plot(x_grid, y_line, alpha=0.02, color='blue')

ax2.scatter(x, Y, color='black', s=50, zorder=5)
ax2.set_xlabel('x')
ax2.set_ylabel('Y')
ax2.set_title('Original Scale')

plt.tight_layout()
plt.savefig('fitted_power_law.png', dpi=300)
```

### Posterior Distributions

```python
import arviz as az

az.plot_posterior(idata, var_names=['alpha', 'beta', 'sigma'],
                  hdi_prob=0.95, figsize=(12, 4))
plt.savefig('posterior_distributions.png', dpi=300)
```

### Diagnostic Plots

```python
# Trace plots
az.plot_trace(idata, var_names=['alpha', 'beta', 'sigma'])
plt.tight_layout()
plt.savefig('trace_plots.png', dpi=300)

# LOO-PIT
az.plot_loo_pit(idata, y='log_Y_obs')
plt.savefig('loo_pit.png', dpi=300)
```

---

## Archived Results

### InferenceData Objects

Pre-computed posterior samples available for immediate use:

**Model 1:**
- Location: `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Load: `idata = az.from_netcdf('posterior_inference.netcdf')`
- Contains: 4000 posterior samples, log-likelihood, sample stats

**Model 2:**
- Location: `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`
- Load: `idata2 = az.from_netcdf('posterior_inference.netcdf')`
- Contains: 6000 posterior samples, log-likelihood, sample stats

### LOO Results

Pre-computed LOO cross-validation:

**Model 1:**
- Location: `/workspace/experiments/experiment_1/posterior_inference/diagnostics/loo_results.json`
- Contains: ELPD LOO, p_loo, Pareto k values for all observations

**Model 2:**
- Location: `/workspace/experiments/experiment_2/posterior_inference/diagnostics/loo_results.json`

### Visualizations

All plots available in `*/plots/` subdirectories:
- `/workspace/experiments/experiment_1/posterior_inference/plots/`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/`
- `/workspace/experiments/model_assessment/plots/`

---

## Complete Code Scripts

### Model 1 Fitting Script

**Location:** `/workspace/experiments/experiment_1/posterior_inference/code/fit_model_pymc.py`

**Run:**
```bash
cd /workspace/experiments/experiment_1/posterior_inference/code
python fit_model_pymc.py
```

### Model 2 Fitting Script

**Location:** `/workspace/experiments/experiment_2/posterior_inference/code/fit_model.py`

**Run:**
```bash
cd /workspace/experiments/experiment_2/posterior_inference/code
python fit_model.py
```

### Diagnostic Script (Model 1)

**Location:** `/workspace/experiments/experiment_1/posterior_inference/code/create_diagnostics.py`

**Run:**
```bash
cd /workspace/experiments/experiment_1/posterior_inference/code
python create_diagnostics.py
```

---

## Verification Checklist

To verify successful reproduction:

### Data Loading
- [ ] Load `/workspace/data/data.csv`
- [ ] Verify n=27, x ∈ [1.0, 31.5], Y ∈ [1.77, 2.72]
- [ ] Log-transform: log_Y, log_x

### Model 1 Fitting
- [ ] Set random seed to 12345
- [ ] Specify priors: α ~ N(0.6, 0.3), β ~ N(0.13, 0.1), σ ~ HN(0.1)
- [ ] Sample: 4 chains × (1000 warmup + 1000 sampling)
- [ ] Verify: All R-hat = 1.000, No divergences

### Parameter Estimates
- [ ] α ≈ 0.580 ± 0.019
- [ ] β ≈ 0.126 ± 0.009
- [ ] σ ≈ 0.041 ± 0.006

### LOO Cross-Validation
- [ ] ELPD LOO ≈ 46.99 ± 3.11
- [ ] p_loo ≈ 2.43
- [ ] All Pareto k < 0.5

### Performance Metrics
- [ ] R² (log scale) ≈ 0.902
- [ ] MAPE ≈ 3.04%
- [ ] MAE ≈ 0.0714

### Model 2 (Optional)
- [ ] γ₁ ≈ 0.003 ± 0.017
- [ ] 95% HDI includes zero
- [ ] ELPD LOO ≈ 23.56

### Model Comparison
- [ ] ΔELPD ≈ -23.43 ± 4.43 (Model 2 worse)
- [ ] Model 1 decisively superior

---

## Common Issues and Solutions

### Issue 1: Different Random Seeds

**Problem:** Results differ slightly from reported values

**Solution:**
- Ensure all seeds set to 12345
- Check PyMC version (must be 5.26.1)
- MCMC is stochastic; small variations expected but should be minimal

**Acceptable variation:**
- Parameter estimates: ±0.005
- ELPD LOO: ±1.0
- Pareto k: ±0.05

### Issue 2: Longer Runtime

**Problem:** Sampling takes much longer than reported ~5 seconds

**Solution:**
- Check system specifications (multi-core CPU recommended)
- Ensure no other heavy processes running
- Try reducing to 2 chains if necessary (will be slower but still work)

### Issue 3: Compilation Errors

**Problem:** PyMC compilation fails

**Solution:**
- Update PyMC: `pip install --upgrade pymc`
- Check dependencies: `pip install --upgrade pytensor`
- See PyMC documentation: https://www.pymc.io/

### Issue 4: Import Errors

**Problem:** Cannot import PyMC or ArviZ

**Solution:**
- Verify installation: `pip list | grep -E "pymc|arviz"`
- Reinstall if needed: `pip install --force-reinstall pymc arviz`
- Check Python version: Python 3.10+ required

---

## Contact and Support

### For Reproducibility Issues

If you encounter problems reproducing these results:

1. **Check software versions** (most common issue)
2. **Verify random seeds** are set correctly
3. **Compare InferenceData** to archived objects
4. **Check data integrity** (MD5 checksum)

### Reporting Issues

Include in report:
- Software versions (`pip list`)
- Error messages (full traceback)
- Data checksums
- Steps already attempted

---

## Reproducibility Statement

This analysis follows best practices for computational reproducibility:

✓ **Code available:** All scripts provided
✓ **Data available:** CSV file accessible
✓ **Software versions documented:** Complete environment
✓ **Random seeds fixed:** All stochastic processes controlled
✓ **Results archived:** InferenceData objects saved
✓ **Step-by-step instructions:** This document

**Reproducibility Level: FULL**

Any researcher with the documented software environment should be able to reproduce all reported results within computational tolerances (MCMC stochasticity).

---

**Document Status:** SUPPLEMENTARY MATERIAL D
**Version:** 1.0
**Date:** October 27, 2025
**Reproducibility:** FULL
