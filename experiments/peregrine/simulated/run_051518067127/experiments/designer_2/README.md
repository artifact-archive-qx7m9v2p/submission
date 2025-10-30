# Designer 2: Transformed Continuous & Temporal Structure Models

**Focus**: Log-transformation approaches with sophisticated temporal dependence modeling
**Date**: 2025-10-30
**Framework**: Bayesian inference with Stan/CmdStanPy

---

## Quick Summary

This directory contains **three competing Bayesian model classes** that leverage the excellent log-scale fit (R²=0.937) while handling severe autocorrelation (ACF=0.971) and regime shifts (7.8× growth).

### Model Rankings (by temporal structure capability)

1. **AR(1) Log-Normal with Regime-Switching** - Most balanced approach
2. **Gaussian Process on Log-Scale** - Most flexible for discovering patterns
3. **Hierarchical Period Model** - Explicit regime heterogeneity

---

## Three Competing Hypotheses

| Hypothesis | Model Class | Key Assumption | Falsified By |
|------------|-------------|----------------|--------------|
| Smooth nonstationary process | Gaussian Process | Continuous temporal correlation decay | Abrupt changepoints, length-scale → 0 |
| Fixed-parameter growth + AR errors | AR(1) Log-Normal | Constant autocorrelation, regime-varying variance | Residual ACF > 0.3, time-varying phi |
| Discrete regime shifts | Hierarchical by Period | Different DGP per regime | Tau parameters → 0, wrong boundaries |

---

## Key Features of All Models

### Shared Elements
- **Transformation**: log(C) to handle exponential growth
- **Mean structure**: Quadratic trend (alpha + beta_1*year + beta_2*year²)
- **Priors**: Weakly informative, centered on EDA findings
- **Diagnostics**: LOO-CV with log_lik, residual ACF, posterior predictive checks

### Critical Differences
- **Model 1**: Single AR(1) parameter, regime-specific variances only
- **Model 2**: No explicit autocorrelation parameter (GP captures all temporal structure)
- **Model 3**: Everything varies by regime (intercept, slope, AR, variance)

---

## Falsification Criteria (When to Abandon Each Model)

### Model 1: AR(1) Log-Normal
- [ ] Residual ACF[1] > 0.3 (AR(1) insufficient)
- [ ] Regime variances all overlap (no regime effect)
- [ ] phi 95% CI includes 0 (no autocorrelation)
- [ ] Prior-posterior conflict
- [ ] LOO shows systematic bias in late regime

### Model 2: Gaussian Process
- [ ] Length-scale rho → ∞ or → 0 (GP adds no value)
- [ ] Marginal SD eta ≈ 0 (trend explains everything)
- [ ] Poor mixing (ESS < 100)
- [ ] Over-smoothing (misses sharp transitions)
- [ ] LOO-CV worse than AR(1) by >4 ELPD

### Model 3: Hierarchical
- [ ] All tau parameters ≈ 0 (no heterogeneity)
- [ ] Any regime diverges (Rhat > 1.05)
- [ ] Overfitting (LOO much worse than in-sample)
- [ ] Boundary artifacts (jumps within regimes)
- [ ] Phi_regime too heterogeneous (model too complex)

---

## Implementation Priority

### Phase 1: Baseline (Week 1)
1. Fit Model 1 (AR1 Log-Normal) - establishes baseline
2. Check convergence and residual diagnostics
3. If successful, proceed to Phase 2
4. If fails, pivot to alternative approaches

### Phase 2: Comparison (Week 1-2)
1. Fit Model 3 (Hierarchical) - tests regime hypothesis
2. Fit Model 2 (GP) - most flexible, last resort
3. Compare all via LOO-CV
4. Select best model based on ELPD + diagnostics

### Phase 3: Validation (Week 2)
1. Sensitivity analysis (priors, regime boundaries)
2. Stress tests (remove last regime, shuffle, cross-validation)
3. Prior/posterior predictive checks
4. Finalize model selection

---

## Expected Computational Requirements

| Model | Runtime (4 chains × 2000 iter) | Memory | Convergence Risk |
|-------|--------------------------------|--------|------------------|
| Model 1 (AR1) | 1-2 minutes | Low | Low |
| Model 2 (GP) | 2-5 minutes | Medium | Medium |
| Model 3 (Hierarchical) | 2-3 minutes | Medium | High (weak ID) |

---

## Decision Points for Major Pivots

### Pivot 1: Abandon Log-Transform
**Trigger**: Back-transformed predictions systematically biased OR early period underdispersion poorly modeled

**Action**: Switch to count-scale models (Negative Binomial, Poisson-lognormal)

### Pivot 2: Abandon AR Structure
**Trigger**: phi → 1.0 (unit root) OR residuals still show strong ACF

**Action**: Switch to state-space models or differencing

### Pivot 3: Abandon Exponential Growth
**Trigger**: Out-of-sample predictions collapse OR growth unsustainable

**Action**: Add saturation effects (logistic, Gompertz)

### Pivot 4: Abandon All Models
**Trigger**: All models fail convergence OR poor residual diagnostics

**Action**: Observation-level random effects, Student-t errors, or mixture models

---

## Files in This Directory

### Documentation
- `proposed_models.md` - Full model specifications (THIS IS THE MAIN DOCUMENT)
- `README.md` - This summary file

### Stan Code (to be created)
- `model_1_ar1_lognormal.stan` - AR(1) with regime-specific variances
- `model_2_gp_logscale.stan` - Gaussian Process with Matern 3/2 kernel
- `model_3_hierarchical_ar1.stan` - Hierarchical regime model

### Python Scripts (to be created)
- `prior_predictive_checks.py` - Visualize prior distributions
- `fit_models.py` - Main fitting script for all three models
- `compare_models.py` - LOO-CV comparison and model selection
- `diagnostics.py` - Residual analysis and convergence checks

### Results (to be created)
- `results/convergence_summary.txt` - Rhat, ESS, divergences
- `results/loo_comparison.csv` - ELPD differences
- `results/posterior_plots/` - Parameter posteriors
- `results/diagnostics/` - Residual ACF, QQ-plots, PPC

---

## Critical Philosophy

**This is not a checklist to complete.** It's a hypothesis-testing framework.

### What Success Looks Like
- Discovering which temporal structure fits best
- Recognizing model failure early
- Pivoting based on evidence
- Honest uncertainty quantification

### What Success Does NOT Look Like
- Fitting all three models regardless of diagnostics
- Choosing model with best ELPD even if diagnostics fail
- Ignoring prior-posterior conflicts
- Completing tasks without learning

**Remember**: The goal is truth, not task completion. If evidence suggests these models are wrong, **say so and propose alternatives**.

---

## Quick Start Guide

### 1. Read the EDA Report
```bash
cat /workspace/eda/eda_report.md
```

Key findings:
- Exponential growth: 2.37× per year
- Severe overdispersion: variance/mean = 70.43
- Strong autocorrelation: ACF[1] = 0.971
- Regime shifts: Early (underdispersed), Middle (overdispersed), Late (moderate)

### 2. Read the Full Model Proposals
```bash
cat /workspace/experiments/designer_2/proposed_models.md
```

This contains:
- Complete Bayesian specifications with priors
- Falsification criteria for each model
- Stan code templates
- Implementation plans
- Stress tests

### 3. Start with Model 1
- Simplest, most interpretable
- Good balance of flexibility and parsimony
- Fast convergence
- If this fails, others likely will too

### 4. Compare Models
- LOO-CV (ELPD differences)
- Residual diagnostics
- Scientific plausibility
- Only proceed with model that passes ALL checks

### 5. Report Findings
- Don't hide failures
- Document what you learned
- Propose next steps if models fail

---

## Contact and Coordination

**Designer 2 Focus**: Temporal structure, log-transformation, continuous approaches

**Complementary Approaches** (other designers):
- Designer 1: Count-scale models (Negative Binomial, Poisson variants)
- Designer 3: Other alternative approaches

**Integration Point**: Compare LOO-CV across all designers to find globally best model

---

**Last Updated**: 2025-10-30
**Status**: Awaiting implementation
**Next Steps**: Prior predictive checks, then fit Model 1
