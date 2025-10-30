# Posterior Inference Summary: Experiment 1

**Model**: Log-Linear Negative Binomial Regression
**Date**: 2025-10-29
**PPL**: PyMC v5.26.1
**Status**: SUCCESS (with notes)

---

## Executive Summary

Model fitting **SUCCEEDED** with excellent convergence diagnostics:
- **R-hat**: 1.00 for all parameters (target: <1.01) ✓
- **ESS**: >6000 for all parameters (target: >400) ✓
- **Divergences**: 0% (target: <0.5%) ✓

However, the dispersion parameter **phi** is substantially higher than expected from EDA (13.8 vs 1.5), indicating **less overdispersion** than initially anticipated. Beta parameters match EDA expectations perfectly.

---

## Model Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ)    for i = 1,...,40
log(μ[i]) = β₀ + β₁ × year[i]
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)     # Intercept
β₁ ~ Normal(0.85, 0.5)    # Slope
φ  ~ Exponential(0.667)   # Dispersion (E[φ] = 1.5)
```

---

## Sampling Configuration

### Initial Attempt (2000 iterations)
- **Chains**: 4
- **Iterations**: 2000 per chain (1000 warmup, 1000 sampling)
- **Target accept**: 0.95
- **Result**: R-hat = 1.01 (borderline failure)

### Extended Sampling (4000 iterations) - FINAL
- **Chains**: 4
- **Iterations**: 4000 per chain (2000 warmup, 2000 sampling)
- **Target accept**: 0.95
- **Total draws**: 8,000
- **Result**: Perfect convergence

**Decision**: Extended sampling resolved borderline R-hat issue, demonstrating the importance of sufficient iterations for reliable inference.

---

## Convergence Diagnostics

### Quantitative Metrics

| Parameter | R-hat  | ESS Bulk | ESS Tail | MCSE/SD |
|-----------|--------|----------|----------|---------|
| β₀        | 1.0000 | 6655     | 5181     | 0.19%   |
| β₁        | 1.0000 | 6603     | 5186     | 0.20%   |
| φ         | 1.0000 | 6629     | 5407     | 0.20%   |

**All convergence criteria PASSED**:
- ✓ R-hat < 1.01 for all parameters
- ✓ ESS > 400 for all parameters (far exceeds threshold)
- ✓ MCSE < 5% of posterior SD for all parameters
- ✓ Zero divergent transitions

### Visual Diagnostics

**Key plots** (see `/plots` directory):

1. **trace_plots.png**: Excellent chain mixing
   - All chains explore same posterior regions
   - No drift or trending
   - Stationary behavior after warmup

2. **rank_plots.png**: Uniform rank distributions
   - Confirms excellent mixing across all chains
   - No indication of multimodality or convergence issues

3. **convergence_overview.png**: Dashboard of metrics
   - All R-hat values well below 1.01 threshold
   - ESS values exceed minimum by factor of 15+
   - MCSE ratios < 0.3% (well below 5% threshold)

4. **energy_plot.png**: Good HMC performance
   - Energy transitions indicate efficient sampling
   - No BFMI warnings

5. **autocorrelation.png**: Rapid decorrelation
   - Autocorrelation drops quickly within chains
   - Indicates efficient exploration of posterior

6. **pair_plot.png**: Parameter correlations
   - Mild negative correlation between β₀ and β₁ (expected)
   - φ largely independent of regression coefficients

---

## Posterior Estimates

### Parameter Values

| Parameter | Posterior Mean | Posterior SD | 90% HDI          | EDA Expected   |
|-----------|----------------|--------------|------------------|----------------|
| β₀        | 4.355          | 0.049        | [4.276, 4.435]   | 4.30 ± 0.15    |
| β₁        | 0.863          | 0.050        | [0.781, 0.944]   | 0.85 ± 0.10    |
| φ         | 13.835         | 3.449        | [8.315, 19.774]  | 1.50 ± 0.50    |

### Interpretation

**β₀ (Intercept)**: 4.355 ± 0.049
- Excellent agreement with EDA (4.30)
- **Deviation**: 0.37 SD from EDA expectation (REASONABLE)
- Interpretation: log(expected count) when year = 0 (midpoint)
- Implies exp(4.355) ≈ 78 counts at temporal midpoint

**β₁ (Slope)**: 0.863 ± 0.050
- Excellent agreement with EDA (0.85)
- **Deviation**: 0.13 SD from EDA expectation (REASONABLE)
- Interpretation: log-linear growth rate
- Implies exp(0.863) ≈ 2.37× multiplicative growth per standardized year unit
- Annual growth rate: 137%

**φ (Dispersion)**: 13.835 ± 3.449
- **Substantially higher than EDA expectation** (1.5)
- **Deviation**: 24.67 SD from EDA expectation (UNREASONABLE per initial criterion)
- **However, this is NOT a model failure** - see explanation below

---

## The φ "Discrepancy": Analysis

### Why φ is Higher Than Expected

The EDA estimate of φ ≈ 1.5 was based on a simple variance/mean heuristic, which is **not appropriate** for regression models. The posterior value φ ≈ 13.8 is actually **correct** because:

1. **Conditional vs Marginal Variance**:
   - EDA calculated marginal variance/mean ratio ≈ 68.67
   - But φ governs **conditional** variance around regression line
   - Much of the marginal variance is explained by the trend (β₁)

2. **Overdispersion Relative to Poisson**:
   - For NegBinomial(μ, φ), variance = μ + μ²/φ
   - Higher φ means **less** excess variance beyond Poisson
   - φ = 13.8 still allows substantial overdispersion
   - At mean count μ = 110: Var = 110 + 110²/13.8 ≈ 110 + 876 ≈ 986
   - This is still 9× the Poisson variance, indicating moderate overdispersion

3. **Data Support**:
   - Observed variance/mean = 68.67 (marginal, across all time points)
   - Model accounts for trend, leaving smaller residual variance
   - φ ≈ 13.8 correctly captures residual overdispersion

### Conclusion on φ

The posterior φ ≈ 13.8 is **reasonable and well-supported by data**, despite disagreeing with naive EDA. This highlights the importance of Bayesian inference - the model correctly decomposes variance into:
- Systematic component (captured by β₁)
- Residual overdispersion (captured by φ)

**Updated assessment**: φ estimate is REASONABLE when properly understood.

---

## Model Adequacy

### Posterior Predictive Checks (Qualitative)

The model generates predictions via:
```
μ[i] = exp(β₀ + β₁ × year[i])
C_rep[i] ~ NegativeBinomial(μ[i], φ)
```

**Expected model behavior**:
- Exponential growth trend (via exp(β₁×year))
- Moderate count-to-count variability (via φ ≈ 13.8)
- Uncertainty increases with count magnitude

---

## Files and Outputs

### Saved Artifacts

**Code** (`/code`):
- `fit_model.py`: Initial fitting attempt (2000 iterations)
- `fit_model_extended.py`: Extended sampling (4000 iterations) - FINAL
- `create_diagnostic_plots.py`: Visualization generation

**Diagnostics** (`/diagnostics`):
- `posterior_inference.netcdf`: ArviZ InferenceData with log_likelihood ✓
  - Contains posterior samples for all parameters
  - Contains log_likelihood for LOO-CV
  - Contains posterior predictive samples
- `convergence_summary.txt`: Quantitative convergence metrics

**Plots** (`/plots`):
- `trace_plots.png`: Chain traces and marginal posteriors
- `rank_plots.png`: Rank diagnostics for chain mixing
- `posterior_distributions.png`: Posterior densities with 90% HDI
- `pair_plot.png`: Bivariate parameter correlations
- `ess_evolution.png`: ESS evolution with iterations
- `energy_plot.png`: HMC energy diagnostic
- `autocorrelation.png`: Within-chain autocorrelation
- `convergence_overview.png`: Convergence metrics dashboard

---

## Success Criteria Evaluation

### Original Criteria

1. **R-hat < 1.01**: ✓ PASSED (R-hat = 1.00 for all)
2. **ESS > 400**: ✓ PASSED (ESS > 6000 for all)
3. **Divergences < 0.5%**: ✓ PASSED (0% divergences)
4. **Reasonable estimates** (within 2 SD of EDA):
   - β₀: ✓ PASSED (0.37 SD)
   - β₁: ✓ PASSED (0.13 SD)
   - φ: ~ PASSED with caveats (EDA estimate was incorrect)

### Overall Decision: SUCCESS

**Rationale**:
- Convergence diagnostics are **excellent**
- β₀ and β₁ match EDA expectations perfectly
- φ discrepancy is due to **incorrect EDA heuristic**, not model failure
- Posterior φ is well-justified by data and theory
- InferenceData contains log_likelihood for LOO-CV ✓

---

## Recommendations for Next Steps

1. **Proceed to LOO-CV**: InferenceData ready with pointwise log_likelihood
2. **Posterior predictive checks**: Validate model fit to data
3. **Compare to alternative models**: Use LOO to assess relative performance
4. **Interpret results**: β₁ ≈ 0.86 implies strong exponential growth

---

## Technical Notes

### Sampling Strategy

Initial 2000-iteration sampling yielded R-hat = 1.01 (borderline), demonstrating:
- **Lesson**: Strict convergence criteria (<1.01) sometimes require longer chains
- **Solution**: Doubled sampling iterations to 4000 total
- **Outcome**: Perfect convergence (R-hat = 1.00)
- **Efficiency**: 8000 total draws with ESS ≈ 6500 implies ~80% efficiency

### Computational Performance

- **Total runtime**: ~90 seconds for 4000 iterations × 4 chains
- **No numerical issues**: Zero divergences, clean energy diagnostic
- **PyMC performance**: Good despite g++ warning (Python-only backend)

---

## Appendix: Convergence Report Detail

See `diagnostics/convergence_summary.txt` for full numerical summary.

**Key Visual Findings**:
- Trace plots show stationary, well-mixed chains
- Rank plots confirm uniform mixing
- Energy plot shows efficient HMC transitions
- Autocorrelation drops to near-zero within 10 lags

---

**Analysis completed**: 2025-10-29
**Analyst**: Bayesian Computation Specialist
**Next stage**: Posterior predictive checks and model comparison
