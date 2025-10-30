# Parametric Bayesian Models - Designer 1

**Status**: Design Phase Complete
**Date**: 2025-10-28
**Analyst**: Parametric Models Specialist

---

## Quick Start

**Main Documents**:
- **`proposed_models.md`** - Comprehensive model specifications (29KB, ~7000 words)
- **`experiment_plan.md`** - Concise experiment protocol (15KB, ~3500 words)

**Data**:
- Dataset: `/workspace/data/data.csv` (n=27)
- EDA Report: `/workspace/eda/eda_report.md`

---

## Three Model Classes Proposed

### 1. Logarithmic (BASELINE)
```
Y ~ Normal(β₀ + β₁*log(x), σ)
```
- **Parameters**: 3
- **EDA Performance**: R²=0.897, RMSE=0.087
- **Interpretation**: Smooth saturation, diminishing returns
- **Strength**: Simplest, proven EDA fit
- **Weakness**: Cannot capture sharp regime transition

### 2. Piecewise Linear (MOST FLEXIBLE)
```
Y ~ Normal(μ, σ) where:
  μ = α₁ + β₁*x     if x ≤ τ (changepoint)
  μ = α₂ + β₂*x     if x > τ
```
- **Parameters**: 4-5 (with/without continuity constraint)
- **EDA Performance**: F=22.4, p<0.0001 for regime shift at x≈7
- **Interpretation**: Sharp threshold, two operating regimes
- **Strength**: Captures regime structure identified in EDA
- **Weakness**: Changepoint location uncertain, more parameters

### 3. Asymptotic (MOST MECHANISTIC)
```
Y ~ Normal(a - b*exp(-c*x), σ)
```
- **Parameters**: 4
- **EDA Performance**: R²=0.889, RMSE=0.090
- **Interpretation**: Exponential approach to biochemical/physical maximum
- **Strength**: Mechanistic interpretation, smooth saturation
- **Weakness**: Nonlinear parameters, sampling challenges

---

## Model Comparison Strategy

**Primary**: Leave-One-Out Cross-Validation (LOO-CV)
- Metric: ELPD_loo (higher is better)
- Decision: Δ ELPD > 10 = strong preference

**Secondary**: Posterior Predictive Checks (PPCs)
- 6 test statistics: mean, SD, range, regime differences, smoothness, outliers
- Failure: p-value < 0.05 or > 0.95

**Robustness**: Student-t likelihood variants for outlier at x=31.5

---

## Critical Questions to Answer

1. **Is saturation smooth or sharp?**
   - Smooth → Logarithmic or Asymptotic wins
   - Sharp → Piecewise wins

2. **Do we need robust likelihood?**
   - If Student-t ν < 10 → Heavy tails present
   - If Student-t ν > 30 → Normal sufficient

3. **Can we identify changepoint with n=27?**
   - Tight τ posterior → Clear threshold
   - Flat τ posterior → Changepoint not identifiable

4. **How sensitive to x=31.5 outlier?**
   - Refit without outlier
   - If estimates shift >10% → Single point drives inference

---

## Falsification Criteria

### When to Abandon Each Model

**Logarithmic**:
- Residuals cluster into two groups (low-x vs high-x)
- Piecewise has Δ ELPD > 10
- PPC for local slope change is extreme (p < 0.05)

**Piecewise**:
- Changepoint posterior is flat (not identifiable)
- Slopes not different: P(β₁ > β₂) < 0.9
- Discontinuity at changepoint > 0.2 units
- Worse LOO than simpler logarithmic

**Asymptotic**:
- Asymptote a < max(Y) (can't reach data)
- Rate parameter c → 0 or → ∞
- Persistent divergences >1%, ESS < 100
- Residuals show two-regime structure

### When to Abandon Parametric Approach Entirely

- All models have multiple Pareto k > 0.7
- Systematic PPC failures across all models
- Residual patterns suggest missing covariates
- Non-parametric alternatives perform much better

---

## Implementation Phases

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Fit 4 baseline models (Normal likelihood) | 2-4 hours |
| 2 | Fit robust variants (Student-t likelihood) | 1-2 hours |
| 3 | Deep dive on top 2 models (PPCs, sensitivity) | 3-6 hours |
| 4 | Final report & recommendations | 1-2 hours |

**Total**: 7-14 hours

**Fast track**: Fit Log/Normal + Piecewise/Normal only → 1-2 hours

---

## Key Design Principles

1. **Falsification mindset**: Design models to fail informatively
2. **Adversarial testing**: Stress tests designed to break assumptions
3. **Honesty**: Report all models, not just best; acknowledge uncertainty
4. **Decision points**: Clear criteria for switching model classes
5. **Escape routes**: Alternative models if all primary models fail

---

## Expected Outcomes

**Scenario A**: Logarithmic wins → Smooth saturation, use log-transformed predictor

**Scenario B**: Piecewise wins → Sharp threshold at x≈7, investigate physical mechanism

**Scenario C**: Asymptotic wins → Biochemical saturation, estimate asymptote and rate

**Scenario D**: Models equivalent → Structural uncertainty, use model averaging

**Scenario E**: All fail → Switch to non-parametric (GP, splines)

---

## Prior Justification Philosophy

With n=27, priors matter. All priors are:
- **Weakly informative**: Constrain to plausible ranges, but let data dominate
- **Conservative**: Allow for larger effects/variances than EDA suggests
- **Justified**: Based on EDA findings, domain knowledge, or computational stability

**Prior sensitivity analysis**: Required for best model to ensure robustness

---

## Success Criteria

### Minimum Standards
- R-hat < 1.05, ESS > 100, divergences < 1%
- At least one model passes all PPCs
- LOO diagnostics acceptable (Pareto k < 0.7 for most points)

### Ideal Standards
- R-hat < 1.01, ESS > 400, 0 divergences
- Clear winner (Δ ELPD > 10)
- All PPCs pass, prior-insensitive, outlier-insensitive

### Failure Criteria
- Persistent divergences >5%
- Non-convergence after 10k iterations
- Multiple Pareto k > 0.7 across all models
- Systematic PPC failures (p < 0.01) across all models

---

## Files to be Generated

### Reports
- `phase1_initial_fits.md` - First comparison
- `phase2_robust_likelihood.md` - Student-t analysis
- `phase3_detailed_analysis.md` - PPCs, sensitivity
- `final_model_recommendation.md` - Conclusions

### Code
- `model_logarithmic.stan` (or `.py` for PyMC)
- `model_piecewise.stan`
- `model_asymptotic.stan`
- `fit_all_models.py` - Master script
- `posterior_analysis.py` - Diagnostics, PPCs, LOO
- `sensitivity_analysis.py` - Prior, outlier sensitivity

### Visualizations
- `posterior_predictive_overlay.png`
- `residual_vs_x.png`
- `residual_vs_fitted.png`
- `trace_plots.png`
- `pairs_plot.png`
- `ppc_test_statistics.png`
- `loo_comparison.png`

---

## Contact Points with Other Specialists

- **Non-parametric designer**: Share LOO results; if all parametric models fail, hand off
- **Main agent**: Report convergence issues, recommend model class switches
- **Domain expert** (if available): Validate parameter interpretability

---

## Red Flags to Watch

| Red Flag | Action |
|----------|--------|
| Changepoint at prior boundary | Expand prior range |
| Asymptotic `a` < max(Y) | Reject model |
| Student-t ν → 1 | Check data quality |
| All models similar LOO | Consider non-parametric |
| High parameter correlation >0.95 | Reparameterize |

---

## Final Philosophy

> "All models are wrong, but some are useful." - George Box

These models are **competing hypotheses**, not truths. Success is:
- Finding which model fails least badly
- Understanding how and why models break
- Quantifying uncertainty honestly
- Knowing when to abandon the approach

**The data decides. Not the analyst.**

---

**Status**: Ready for implementation in Stan/PyMC
**Next Step**: Code models, fit to data, compare via LOO-CV
