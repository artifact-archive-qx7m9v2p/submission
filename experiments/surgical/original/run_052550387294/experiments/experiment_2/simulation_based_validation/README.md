# Simulation-Based Calibration: Hierarchical Logit Model

**Experiment**: 2 (Hierarchical Logit Model)
**Date**: 2025-10-30
**Status**: **FAILED** ❌
**Inference Method**: MAP + Laplace Approximation

---

## Overview

This directory contains simulation-based calibration (SBC) results for the Hierarchical Logit Model. SBC tests whether the model can recover known parameters from synthetic data, validating both the model and inference method.

**Result**: The validation **FAILED** due to severe miscalibration when using MAP + Laplace approximation for inference.

---

## Directory Structure

```
simulation_based_validation/
├── README.md                    # This file
├── findings.md                  # Executive summary and decision (START HERE)
├── recovery_metrics.md          # Detailed quantitative analysis
├── code/
│   ├── hierarchical_logit.stan  # Stan model (compilation failed)
│   ├── run_sbc.py              # SBC with Stan (not used)
│   ├── run_sbc_scipy.py        # SBC with MAP + Laplace (used)
│   └── visualize_sbc.py        # Diagnostic plots
├── results/
│   ├── sbc_results.csv         # Raw SBC results (150 iterations)
│   ├── sbc_summary.json        # Quantitative metrics
│   └── sbc_log.txt             # Execution log
└── plots/
    ├── sbc_rank_histograms.png              # Calibration diagnostic (KEY)
    ├── coverage_calibration.png             # Coverage failures (KEY)
    ├── parameter_recovery_scatter.png       # Bias assessment
    ├── posterior_contraction.png            # Overconfidence evidence
    ├── parameter_space_identifiability.png  # Joint coverage
    ├── zscore_distribution.png              # Non-normality
    └── sbc_comprehensive_summary.png        # Integrated overview
```

---

## Quick Start

### 1. Read the Findings
Start with **`findings.md`** for executive summary and decision.

### 2. View Key Plots
- **`plots/sbc_rank_histograms.png`**: Shows extreme non-uniformity (calibration failure)
- **`plots/coverage_calibration.png`**: Shows 98% failure rate for σ parameter

### 3. Detailed Metrics
Read **`recovery_metrics.md`** for comprehensive quantitative analysis.

---

## Key Results

### Coverage (Target: 95%)

| Parameter | Coverage | Status |
|-----------|----------|--------|
| μ_logit | 40.7% | **FAIL** ❌ |
| σ | 2.0% | **CATASTROPHIC FAIL** ❌❌ |

### Bias

| Parameter | Bias | RMSE | Status |
|-----------|------|------|--------|
| μ_logit | -0.016 | 0.354 | Small bias, but poor uncertainty |
| σ | +1.045 | 1.088 | Large overestimation |

### Rank Statistics (uniformity test)

| Parameter | χ² | p-value | Status |
|-----------|-----|---------|--------|
| μ_logit | 567.6 | <0.0001 | Non-uniform ❌ |
| σ | 2770.8 | <0.0001 | Extreme non-uniformity ❌ |

---

## Why Did It Fail?

### Root Cause: Laplace Approximation Inadequacy

The Laplace approximation (normal approximation to posterior) is fundamentally inappropriate for hierarchical models:

1. **Funnel geometry**: Hierarchical posteriors are not normal
2. **High dimensionality**: 14 parameters (μ_logit, σ, 12 η's)
3. **Boundary constraints**: σ > 0 creates non-normal shape
4. **Overconfidence**: Hessian-based covariance underestimates uncertainty

### Contributing Factor: Weak Identifiability

With only N=12 trials:
- Limited information about σ (scale parameter)
- Similar to φ failure in Beta-Binomial (Experiment 1)
- Both location and scale parameters affected

---

## What This Means

### If we fit this model to real data using current approach:

**μ_logit**:
- Point estimates may be reasonable
- But 95% CIs will miss truth **60% of the time**
- Overconfident precision claims

**σ**:
- Will overestimate by ~1.0 on average
- 95% CIs will miss truth **98% of the time**
- Completely unreliable for scientific inference

**Scientific impact**: Invalid conclusions about overdispersion and uncertainty

---

## Required Actions

### 1. STOP (Immediate)
**Do not fit real data** with current inference method

### 2. Implement Full MCMC
- Install PyMC, Stan (with compiler), or Numpyro
- Implement proper MCMC sampling
- Laplace approximation is not sufficient

### 3. Re-Run SBC
- Validate with full MCMC
- Check if calibration improves
- Only proceed to real data if validation passes

### 4. If MCMC Still Fails
- Consider model simplification
- Use informative priors on σ
- Accept N=12 limitation and report uncertainty
- Collect more data (N >> 12)

---

## Comparison to Beta-Binomial

| Metric | Beta-Binomial | Hierarchical Logit | Better |
|--------|--------------|-------------------|--------|
| Location coverage | 96.6% ✓ | 40.7% ❌ | Beta-Binomial |
| Scale coverage | 45.6% ❌ | 2.0% ❌❌ | Beta-Binomial |
| Parameters | 2 | 14 | Beta-Binomial |
| Inference | MAP + Laplace | MAP + Laplace | - |

**Conclusion**: Hierarchical structure + Laplace approximation performs worse than Beta-Binomial. Need MCMC for hierarchical models.

---

## How to Reproduce

### Run SBC (if needed)
```bash
cd /workspace/experiments/experiment_2/simulation_based_validation/code
python run_sbc_scipy.py  # Takes ~30-60 minutes
```

### Create Plots
```bash
python visualize_sbc.py  # Takes ~2 minutes
```

### View Results
```bash
# Quantitative summary
cat ../results/sbc_summary.json

# Raw results
head ../results/sbc_results.csv
```

---

## Technical Details

### Model Specification

**Likelihood**:
```
r_i ~ Binomial(n_i, θ_i)
logit(θ_i) = μ_logit + σ·η_i
η_i ~ Normal(0, 1)
```

**Priors**:
```
μ_logit ~ Normal(-2.53, 1)
σ ~ HalfNormal(0, 1)
η_i ~ Normal(0, 1)
```

**Parameters**: μ_logit, σ, η_1, ..., η_12 (14 total)

### SBC Configuration

- **Iterations**: 150
- **Success rate**: 100% (optimization converged)
- **Posterior samples**: 4000 per iteration (from Laplace approximation)
- **Sample sizes**: n = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360]

### Inference Method

**MAP Estimation**:
- L-BFGS-B optimization
- Negative log posterior minimization
- Multiple random initializations

**Laplace Approximation**:
- Hessian computed at MAP via finite differences
- Covariance = inverse Hessian
- Sample from multivariate normal

**Limitations**:
- Assumes posterior is approximately normal
- Fails for hierarchical models
- Underestimates uncertainty

---

## Diagnostic Plots Explanation

### 1. Rank Histograms (`sbc_rank_histograms.png`)
- **Expected**: Uniform (flat) histogram
- **Observed**:
  - μ_logit: Bimodal (U-shaped)
  - σ: Spike at rank 0
- **Interpretation**: Severe miscalibration

### 2. Coverage Calibration (`coverage_calibration.png`)
- **Expected**: 95% green intervals, 5% red
- **Observed**:
  - μ_logit: ~60% red
  - σ: ~98% red
- **Interpretation**: Intervals too narrow (overconfident)

### 3. Parameter Recovery (`parameter_recovery_scatter.png`)
- Shows true vs estimated values
- Error bars = 95% CI
- Ideal = points on diagonal
- **Observed**:
  - μ_logit: Near diagonal but with large scatter
  - σ: Systematic overestimation

### 4. Posterior Contraction (`posterior_contraction.png`)
- Shows learning from data
- **Observed**: Excessive contraction (overconfidence)
- μ_logit: 92% narrower than prior
- σ: 64% narrower than prior

### 5. Parameter Space (`parameter_space_identifiability.png`)
- Shows joint parameter coverage
- **Observed**: Nearly all red (poor joint coverage)
- Suggests identifiability issues

### 6. Z-Score Distribution (`zscore_distribution.png`)
- **Expected**: Standard normal if well-calibrated
- **Observed**: Extreme deviations
- **Interpretation**: Non-normal posterior (Laplace fails)

---

## References

### SBC Methodology
- Talts et al. (2018). "Validating Bayesian Inference Algorithms with Simulation-Based Calibration"
- Säilynoja et al. (2022). "Graphical Test for Discrete Uniformity and its Applications in Goodness of Fit Evaluation and Multiple Sample Comparison"

### Hierarchical Models
- Gelman & Hill (2007). "Data Analysis Using Regression and Multilevel/Hierarchical Models"
- Betancourt & Girolami (2015). "Hamiltonian Monte Carlo for Hierarchical Models"

### Why Laplace Approximation Fails
- Neal (2003). "Slice Sampling"
- Betancourt (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo"

---

## Contact

For questions about this validation:
- See `findings.md` for executive summary
- See `recovery_metrics.md` for detailed analysis
- Check plots in `plots/` directory

**Status**: Validation FAILED - requires MCMC implementation and re-validation

---

## Version History

- **2025-10-30**: Initial SBC with MAP + Laplace approximation
  - Result: FAILED
  - Required action: Implement MCMC

---

**IMPORTANT**: Do not proceed to model fitting (Phase 3) until validation passes with proper MCMC inference.
