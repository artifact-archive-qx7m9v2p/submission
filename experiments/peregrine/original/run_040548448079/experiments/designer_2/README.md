# Designer 2: Smooth Nonlinear Models

**Focus**: Flexible trend models without discrete changepoints

## Philosophy

This design explores whether the apparent structural break at observation 17 can be explained by smooth, continuous acceleration rather than a discrete regime change. We propose three model classes with increasing flexibility:

1. **Polynomial Regression** - Parametric baseline (quadratic/cubic)
2. **Gaussian Process Regression** - Maximum flexibility, captures arbitrary smooth functions
3. **Penalized B-Spline Regression** - Semi-parametric, local flexibility with shrinkage

## Critical Insight

**These models are designed to FAIL if a discrete break exists.** Success means we correctly identify smooth acceleration. Failure means discrete changepoint models are correct.

## Key Files

### Core Documents
- `proposed_models.md` - Full mathematical specifications, priors, falsification criteria
- `falsification_protocol.md` - Systematic testing to determine model adequacy
- `implementation_guide.md` - Step-by-step code with PyMC/Stan examples
- `model_summary.md` - Quick reference guide

### Expected Outputs (after implementation)
- Model traces (`*_trace.nc`)
- Diagnostic plots (`*_residuals.png`, `*_derivative.png`)
- Comparison tables (`model_comparison.csv`)
- Final decision (`final_decision.txt`)

## Quick Start

1. **Read EDA findings**: `/workspace/eda/eda_report.md`
2. **Review models**: `proposed_models.md`
3. **Follow implementation**: `implementation_guide.md`
4. **Apply falsification tests**: `falsification_protocol.md`
5. **Make decision**: Compare to Designer 1's changepoint models

## Decision Criteria

| Evidence | Conclusion |
|----------|------------|
| LOO-ELPD within 10 of changepoint | Smooth sufficient |
| LOO-ELPD 10-20 worse | Borderline |
| LOO-ELPD >20 worse | Discrete break real |

Additional red flags:
- GP lengthscale < 0.2 (trying to capture discontinuity)
- Derivative jump > 1.0 at observation 17
- Residual ACF(1) > 0.5 (autocorrelation not captured)

## Expected Outcome

**Most likely**: Smooth models fail, discrete break confirmed.

**Reasoning**: 730% growth rate increase at observation 17 suggests true regime shift, not smooth acceleration.

**Value**: Systematic falsification builds confidence in changepoint models (Designer 1).

## Model Specifications Summary

### All Models Include
- Negative Binomial likelihood (variance/mean = 67.99)
- Log link function
- AR(1) autocorrelation structure
- Priors informed by EDA

### Model 1: Polynomial
```
log(μ_t) = β₀ + β₁×year + β₂×year² + β₃×year³ + ε_t
ε_t ~ AR(1)
```

### Model 2: Gaussian Process
```
log(μ_t) = f(year_t) + ε_t
f ~ GP(β₀ + β₁×year, k(x,x'))
k(x,x') = σ²_f × exp(-ρ²×(x-x')²)
```

### Model 3: B-Spline
```
log(μ_t) = Σⱼ βⱼ × Bⱼ(year_t) + ε_t
β ~ MVN(0, Σ_β) with smoothing penalty
```

## Implementation Status

- [ ] Polynomial model fitted
- [ ] GP model fitted
- [ ] Spline model fitted
- [ ] Convergence diagnostics
- [ ] LOO comparison
- [ ] Residual analysis
- [ ] First derivative test
- [ ] Leave-future-out CV
- [ ] Comparison to changepoint models
- [ ] Final decision documented

## Contact

Model Designer 2 (Smooth Nonlinear Specialist)
Date: 2025-10-29

---

**Remember**: The goal is finding truth, not defending smooth models. If discrete break is real, our job is to discover and document that fact.
