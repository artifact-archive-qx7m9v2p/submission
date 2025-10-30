# Experiment 1: Negative Binomial Quadratic Regression

**Date:** 2025-10-29
**Status:** In Progress
**Model Class:** Parametric GLM (Baseline)

---

## Model Specification

### Likelihood
```
C_i ~ NegativeBinomial(μ_i, φ)
```
Where:
- `C_i`: Count observation at time i
- `μ_i`: Expected count (mean parameter)
- `φ`: Dispersion parameter (larger → less overdispersion)

### Link Function
```
log(μ_i) = β₀ + β₁·year_i + β₂·year_i²
```

### Prior Distributions
```
β₀ ~ Normal(4.7, 0.3)    # Intercept: log(109) ≈ 4.7 [ADJUSTED: 0.5→0.3]
β₁ ~ Normal(0.8, 0.2)    # Linear growth rate [ADJUSTED: 0.3→0.2]
β₂ ~ Normal(0.3, 0.1)    # Acceleration term [ADJUSTED: 0.2→0.1, CRITICAL]
φ ~ Gamma(2, 0.5)        # Dispersion parameter
```

**Prior Adjustment Note (2025-10-29):**
Initial prior predictive check revealed priors were too vague, generating extreme predictions (max >40,000 vs observed max=272). Tightened all coefficient priors, especially β₂ (0.2→0.1), which is critical because the quadratic term combined with exp() link can create explosive growth. Adjusted priors still weakly informative but prevent unrealistic trajectories.

---

## Rationale

### Why This Model?
1. **Negative Binomial:** Handles extreme overdispersion (Var/Mean = 68)
2. **Quadratic trend:** Captures accelerating growth (6× rate increase)
3. **Log link:** Ensures positive predictions, multiplicative effects
4. **Weakly informative priors:** Centered on EDA findings, allow data to dominate

### Expected Strengths
- Parsimonious (4 parameters for n=40)
- Interpretable parameters (intercept, growth, acceleration)
- Handles non-linearity and overdispersion
- Computationally stable

### Expected Weaknesses
- Assumes constant dispersion φ (may vary with time)
- May show residual autocorrelation (ACF 0.3-0.6)
- Quadratic may extrapolate poorly outside data range

---

## Success Criteria

### Convergence
- R̂ < 1.01 for all parameters
- ESS > 400 for all parameters
- Divergent transitions < 1%

### Predictive Performance
- Posterior predictive coverage: 85-98%
- LOO-ELPD competitive with alternatives
- Pareto-k diagnostics: < 5% problematic (k > 0.7)

### Residual Diagnostics
- No systematic patterns in residuals vs fitted
- Residual ACF(1) < 0.6 (acceptable threshold)
- QQ plot approximately linear

---

## Failure Criteria (REJECT if 2+ occur)

1. **Convergence failure:** R̂ > 1.05 or ESS < 100 despite tuning
2. **Systematic residual bias:** U-shaped or S-shaped pattern
3. **Poor coverage:** < 75% of observations in 95% posterior intervals
4. **High residual ACF:** ACF(1) > 0.8 (temporal models needed)
5. **Extreme outliers:** > 5 observations with LOO Pareto-k > 1.0

---

## Falsification Tests

### Prior Predictive Check
- Do priors generate realistic count trajectories?
- Range: Should cover 10-500 counts
- Shape: Should allow various growth patterns

### Simulation-Based Calibration
- Generate synthetic data from model
- Fit model to synthetic data
- Check parameter recovery (rank statistics)

### Posterior Predictive Check
- Plot observed vs posterior predicted counts
- Coverage analysis: % of obs in 50%, 80%, 95% intervals
- Residual diagnostics

### LOO Cross-Validation
- Compute ELPD and SE
- Check Pareto-k diagnostics
- Compare to alternative models

---

## Implementation Details

**Software:** Stan (CmdStanPy)
**Sampling:** 4 chains, 2000 iterations (1000 warmup, 1000 sampling)
**Adaptation:** adapt_delta = 0.95 (if divergences occur)

**Files:**
- Stan model: `experiments/experiment_1/posterior_inference/code/model.stan`
- Fitting script: `experiments/experiment_1/posterior_inference/code/fit_model.py`
- InferenceData: `experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`

---

## Next Steps

1. ✅ Metadata created
2. ⏳ Prior predictive check
3. ⏳ Simulation-based validation
4. ⏳ Posterior inference
5. ⏳ Posterior predictive check
6. ⏳ Model critique

---

## Notes

This is the baseline parametric model. All other models will be compared against this. Even if it shows some residual autocorrelation, it may be "good enough" depending on magnitude and practical implications.
