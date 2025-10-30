# Designer 1: Parsimonious Bayesian Models

**Philosophy**: Simplicity, Interpretability, Computational Efficiency

**Author**: Bayesian Modeling Strategist (Parsimony Track)

**Date**: 2025-10-29

---

## Overview

This directory contains **parsimonious Bayesian models** designed based on EDA findings from `/workspace/eda/eda_report.md`. The focus is on:

1. **Simplicity**: Minimal parameters, maximum interpretability
2. **Robustness**: Computationally stable, well-identified models
3. **Baseline quality**: Strong reference points for comparison with complex models

---

## Files in This Directory

### Design Documents
- **`proposed_models.md`**: Comprehensive model design with falsification criteria (main document, 600+ lines)
- **`model_specifications.md`**: Quick reference guide for model specs and Stan implementation
- **`README.md`**: This file

### Stan Models
- **`model1_linear.stan`**: Log-Linear Negative Binomial (3 parameters)
- **`model2_quadratic.stan`**: Quadratic Negative Binomial (4 parameters)

### Analysis Scripts
- **`fit_models.py`**: Python script to fit models, run diagnostics, and compare via LOO-CV

### Results (created after running)
- **`results/`**: Directory containing fitted models, plots, and diagnostics

---

## Model Summary

### Model 1: Log-Linear Negative Binomial (RECOMMENDED BASELINE)

**Mathematical Form**:
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i]
```

**Parameters**: 3 (β₀, β₁, φ)

**Interpretation**: Constant exponential growth rate

**Expected Values**:
- β₀ ≈ 4.3 (log-scale intercept)
- β₁ ≈ 0.85 (134% annual growth)
- φ ≈ 1.5 (severe overdispersion)

**When to Use**:
- Always fit as baseline
- Prefer if ΔELPD < 2 compared to Model 2

**When to Reject**:
- LOO-CV shows ΔELPD > 4 vs. Model 2
- Systematic curvature in residuals
- Poor variance calibration

---

### Model 2: Quadratic Negative Binomial

**Mathematical Form**:
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```

**Parameters**: 4 (β₀, β₁, β₂, φ)

**Interpretation**: Accelerating/decelerating growth

**Expected Values**:
- β₀ ≈ 4.3
- β₁ ≈ 0.85
- β₂ ≈ 0.3 (acceleration term)
- φ ≈ 1.5

**When to Use**:
- Model 1 shows systematic residual patterns
- LOO-CV favors quadratic (ΔELPD > 2)
- EDA suggests regime shift

**When to Reject**:
- β₂ 95% CI includes 0 AND |β₂| < 0.1
- ΔELPD < 2 vs. Model 1
- >0.5% divergent transitions

---

## Quick Start

### Prerequisites
```bash
pip install cmdstanpy arviz numpy pandas matplotlib seaborn
```

### Run Analysis
```bash
cd /workspace/experiments/designer_1
python fit_models.py
```

### Expected Output
1. Model convergence diagnostics
2. Posterior predictive checks
3. LOO-CV comparison
4. Plots saved to `results/`
5. InferenceData objects saved as `.nc` files

---

## EDA Key Findings (Summary)

From `/workspace/eda/eda_report.md`:

1. **Severe overdispersion**: Var/Mean ≈ 70 (φ ≈ 1.5)
   - **Implication**: Negative Binomial required (NOT Poisson)

2. **Functional form debate**:
   - **Linear**: R² = 0.92, simple exponential growth
   - **Quadratic**: R² = 0.96, captures acceleration

3. **No temporal autocorrelation**: Residual r = 0.14 (p = 0.37)
   - **Implication**: No need for ARIMA components

4. **Heteroscedasticity**: Variance changes over time
   - **Implication**: May need time-varying dispersion (future work)

5. **Excellent data quality**: No outliers, no zeros, no missing values

---

## Model Comparison Strategy

### LOO-CV Decision Rules

| ΔELPD | Interpretation | Recommendation |
|-------|----------------|----------------|
| < 2 | Statistically equivalent | Choose simpler model (Model 1) |
| 2-4 | Weak evidence | Consider interpretability |
| > 4 | Strong evidence | Use better model |

### Convergence Criteria

All of the following must be met:
- R̂ < 1.01 for all parameters
- ESS_bulk > 400
- ESS_tail > 400
- Divergences < 0.5%
- No hitting max treedepth
- BFMI > 0.3

### Posterior Predictive Checks

Key diagnostics:
1. **Variance-to-mean ratio**: Should be ~70 ± 20
2. **Calibration**: 90% PI should contain ~90% of observations
3. **Residuals**: No systematic patterns vs. fitted or time
4. **Extreme values**: Can generate counts up to 269

---

## Falsification Criteria

### Model 1 (Linear)

**Abandon if**:
1. LOO-CV: ΔELPD > 4 vs. Model 2
2. Residuals show clear U-shape or inverted-U
3. Variance-to-mean ratio outside [50, 90]
4. <80% of observations in 90% prediction intervals

### Model 2 (Quadratic)

**Abandon if**:
1. β₂ 95% CI includes 0 AND |β₂| < 0.1
2. LOO-CV: ΔELPD < 2 vs. Model 1
3. >10% of observations have Pareto-k > 0.7
4. >0.5% divergent transitions

### Both Models

**Reconsider everything if**:
1. Prior-posterior conflict (posteriors >3 SD from EDA)
2. Extreme parameter values (β₀ > 8, β₁ > 2, φ < 0.3 or > 10)
3. Computational pathology (>5% divergences, ESS < 100)
4. Posterior predictive variance < 20 or > 200

---

## Computational Expectations

| Model | Parameters | Warmup | Sampling | Total Time |
|-------|-----------|--------|----------|------------|
| Model 1 | 3 | ~5s | ~5s | ~10s |
| Model 2 | 4 | ~10s | ~5s | ~15s |

*Estimated for 4 chains × 1000 iterations on standard CPU*

---

## Escape Routes

If all proposed models fail, consider:

1. **Time-varying dispersion**: Model heteroscedasticity explicitly
2. **Changepoint model**: Explicit regime shift at year ≈ -0.21
3. **Mixture model**: Two latent processes
4. **Student-t errors**: Heavier tails
5. **Gaussian Process**: Nonparametric trend
6. **Data re-examination**: Recording errors, batch effects, hidden covariates

---

## Key Principles

### Parsimony First
- Start with simplest model (Model 1)
- Add complexity only if evidence supports it
- ΔELPD < 2 → choose simpler model

### Falsification Mindset
- Each model has explicit rejection criteria
- Plan for failure, not just success
- Switching models = learning, not failure

### Interpretability Matters
- Prefer models with clear scientific meaning
- β₁ = growth rate is more interpretable than complex functions
- Communicate results to domain experts

### Computational Stability
- Well-identified models converge reliably
- Avoid over-parameterization
- Monitor diagnostics obsessively

---

## Integration with Other Designers

This is **Designer 1 (Parsimony Track)**. Other designers may propose:
- **More complex models** (GP, hierarchical, changepoint)
- **Different likelihood families** (Student-t, mixture models)
- **Nonparametric approaches** (splines, wavelets)

**Use these models as baselines for comparison**:
- Does added complexity improve ELPD by >2?
- Are complex models more interpretable?
- Do they reveal insights not captured here?

**Philosophy**: Complexity is justified only if it improves understanding or prediction meaningfully.

---

## Contact & Questions

For questions about model design philosophy, refer to:
- **`proposed_models.md`**: Detailed justification and falsification criteria
- **`model_specifications.md`**: Quick reference for implementation
- **EDA Report**: `/workspace/eda/eda_report.md`

---

## Version History

- **2025-10-29**: Initial design (Models 1-2)
- Future: Add Model 3 (heteroscedastic) if needed

---

## License

This analysis is part of an adaptive Bayesian modeling experiment. All code is provided as-is for educational and research purposes.

---

**End of README**
