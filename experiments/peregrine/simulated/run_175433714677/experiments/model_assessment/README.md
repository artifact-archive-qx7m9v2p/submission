# Model Assessment: Dual Model Rejection Analysis

**Date**: 2025-10-29
**Status**: BOTH MODELS REJECTED
**Analyst**: Model Assessment Specialist

---

## Overview

This directory contains a comprehensive assessment of two fitted Bayesian Negative Binomial models, both of which were REJECTED despite excellent computational properties. This is an atypical assessment—we do not select a "best" model, but rather document **why both failed** and what this reveals about the data structure.

---

## Key Findings

### 1. Both Models Are Computationally Sound

- **Perfect convergence**: R = 1.0000 for all parameters
- **High ESS**: >6,000 effective samples (target: >400)
- **Zero divergences**: No sampling issues
- **Reliable LOO**: All Pareto k < 0.5 (no problematic observations)

### 2. Both Models Are Statistically Inadequate

**Model 1 (Log-Linear)**:
- Systematic residual curvature (quadratic coef = -5.22)
- 4.17× MAE degradation from early to late period
- Overestimates dispersion (prediction intervals too wide)
- Fails 3 of 4 posterior predictive checks

**Model 2 (Quadratic)**:
- β₂ non-significant (95% CI: [-0.051, 0.173])
- No improvement over Model 1 (ΔELPD = 0.45 ± 0.93)
- Residual curvature WORSENED (coef = -11.99)
- Late-period fit DEGRADED (MAE ratio 4.56 vs 4.17)
- Fails 5 of 6 posterior predictive checks

### 3. Models Are Equivalent in Predictive Performance

- **ΔELPD**: 0.45 ± 0.93 (difference < 2 SE)
- **Interpretation**: Models are indistinguishable in predictive accuracy
- **Conclusion**: Adding quadratic term provides NO improvement

### 4. Common Failure Mode

**Polynomial functional form is inappropriate**:
- Linear growth: Assumes constant exponential growth (violated)
- Quadratic growth: Assumes polynomial acceleration (not supported)
- Data structure: Requires different model class (changepoint, GP, etc.)

---

## Files in This Directory

### Reports

1. **`assessment_report.md`** (COMPREHENSIVE, 15,000 words)
   - Full quantitative analysis
   - LOO-CV comparison results
   - Diagnostic summaries for both models
   - Critical analysis of failure modes
   - Methodological lessons learned

2. **`recommendations.md`** (DETAILED, 8,000 words)
   - Next steps given both models rejected
   - 6 alternative approaches prioritized
   - Implementation roadmap
   - Falsification criteria for each approach
   - Risk assessment and success metrics

3. **`README.md`** (this file)
   - Quick overview and navigation

### Data

4. **`loo_comparison.csv`**
   - Quantitative comparison table
   - ELPD, SE, p_loo, Pareto k stats
   - Decision and rejection reasons

5. **`comparison_results.json`**
   - Detailed comparison metrics
   - Model-specific diagnostics
   - Pareto k statistics
   - Structured for programmatic access

### Visualizations

All plots in `/plots/` subdirectory:

6. **`loo_comparison.png`**
   - LOO-CV ELPD comparison with error bars
   - Shows models are equivalent (bars overlap)

7. **`pareto_k_diagnostics.png`**
   - Pareto k values for each observation (both models)
   - All k < 0.5 (excellent LOO reliability)

8. **`parameter_comparison.png`**
   - Posterior distributions for all parameters
   - β₂ clearly overlaps zero (non-significant)
   - Shared parameters (β₀, β₁, φ) nearly identical

9. **`convergence_summary.png`**
   - Convergence metrics dashboard
   - R, ESS comparison
   - Shows: CONVERGED but REJECTED

### Code

10. **`code/comprehensive_assessment.py`**
    - Assessment analysis script
    - Loads both models
    - Computes LOO-CV
    - Generates all plots
    - Saves metrics
    - Fully reproducible (seed=42)

---

## Quick Start

### View Key Findings

```bash
# Read the executive summary
head -100 assessment_report.md

# Check LOO comparison
cat loo_comparison.csv

# View plots
open plots/*.png
```

### Run Assessment

```bash
# Reproduce all analysis
python code/comprehensive_assessment.py

# Output: 4 plots + 2 data files + console summary
```

### Next Steps

```bash
# Read recommendations for alternative models
cat recommendations.md

# Priority: Implement changepoint model (Model 3b)
# See recommendations.md → "Implementation Roadmap" section
```

---

## Assessment Summary Table

| Aspect | Model 1 (Linear) | Model 2 (Quadratic) | Verdict |
|--------|------------------|---------------------|---------|
| **Convergence** | R=1.00, ESS=6603 | R=1.00, ESS=7783 | Both EXCELLENT |
| **ELPD** | -174.61 ± 4.80 | -175.06 ± 5.21 | Equivalent |
| **p_loo** | 1.51 | 2.12 | Model 1 simpler |
| **Pareto k >0.7** | 0/40 | 0/40 | Both reliable |
| **Residual curve** | -5.22 | -11.99 | Model 2 WORSE |
| **MAE ratio** | 4.17 | 4.56 | Model 2 WORSE |
| **Decision** | REJECTED | REJECTED | Both fail |
| **Reason** | Curvature | β₂ non-sig | Different |

---

## Key Visualizations Explained

### 1. LOO Comparison Plot

**What it shows**: ELPD estimates with standard errors

**Finding**: Error bars completely overlap → models equivalent

**Implication**: Adding quadratic term buys nothing

### 2. Pareto k Diagnostics

**What it shows**: LOO reliability for each observation

**Finding**: All k < 0.5 (excellent)

**Implication**: Comparison is trustworthy, no influential points

### 3. Parameter Comparison

**What it shows**: Posterior distributions for β₀, β₁, β₂, φ

**Finding**:
- β₀, β₁, φ nearly identical between models
- β₂ clearly includes zero

**Implication**: Model 2 = Model 1 + noise

### 4. Convergence Summary

**What it shows**: R, ESS, decision status

**Finding**: Both CONVERGED, both REJECTED

**Implication**: Computational success ≠ statistical adequacy

---

## Critical Questions Answered

### 1. Are both models computationally sound?

**YES** - Perfect convergence, reliable LOO, no numerical issues

### 2. Are both models statistically inadequate?

**YES** - Both fail posterior predictive checks, systematic residual patterns

### 3. What is the common failure mode?

**Polynomial functional form inappropriate** - Data require different model class

### 4. What alternatives should be explored?

**See recommendations.md**:
1. Changepoint models (HIGH PRIORITY)
2. Gaussian processes (MEDIUM PRIORITY)
3. Time-varying coefficients (MEDIUM PRIORITY)
4. Missing covariates (EXPLORATORY)

### 5. Is there any reason to prefer one model?

**Model 1 (by parsimony)** - Simpler, slightly better ELPD

**But NEITHER should be used** for inference or prediction

---

## Methodological Lessons

### 1. Convergence ≠ Adequacy

Perfect R and ESS do not guarantee a good model. Posterior predictive checks are essential.

### 2. R² is Insufficient

Model 2 had R² = 0.96 in EDA, yet Bayesian analysis found no improvement. Point estimates mislead.

### 3. LOO-CV Prevents Overfitting

EDA predicted ΔELPD > 10, but LOO-CV found ΔELPD ≈ 0. Cross-validation catches complexity that doesn't help.

### 4. Falsification Frameworks Work

Pre-registered criteria successfully identified both models as inadequate. The framework guided next steps.

### 5. Model Comparison is Essential

Without comparing to Model 1, we might have accepted Model 2 based on convergence alone. Comparison revealed equivalence.

---

## What We've Learned About the Data

### The Pattern

1. **Early period**: Model fits well (MAE = 6.3)
2. **Late period**: Model fails (MAE = 26.5)
3. **Transition**: Systematic, not random

### Implications

1. **Growth is non-polynomial**: Neither linear nor quadratic works
2. **Possible regime change**: Different dynamics early vs late
3. **Missing structure**: Changepoint, GP, or covariates needed
4. **Non-stationary process**: Parameters may change over time

### Scientific Value

Even rejected models provide insight:
- **What doesn't work**: Polynomial growth
- **Where it breaks**: Late period (years >0.98)
- **How it breaks**: Systematic underprediction
- **Why it breaks**: Wrong functional form

---

## Reproducibility

### Data Sources

- Model 1: `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Model 2: `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`

### Software

- Python 3.13
- PyMC 5.26.1
- ArviZ 0.20.0
- NumPy, Pandas, Matplotlib

### Random Seed

All analyses use `seed=42` for reproducibility

### Runtime

Assessment script completes in ~30 seconds on standard hardware

---

## Related Documentation

### Model 1 (Log-Linear)

- Metadata: `/workspace/experiments/experiment_1/metadata.md`
- Inference summary: `/workspace/experiments/experiment_1/posterior_inference/RESULTS.md`
- PPC findings: `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md`
- Critique: `/workspace/experiments/experiment_1/model_critique/critique_summary.md`
- **Decision**: `/workspace/experiments/experiment_1/model_critique/decision.md`

### Model 2 (Quadratic)

- Metadata: `/workspace/experiments/experiment_2/metadata.md`
- Inference summary: `/workspace/experiments/experiment_2/posterior_inference/inference_summary.md`
- **No full PPC** (only comparison to Model 1)

---

## Citation

If using this assessment methodology:

```
Model Assessment Framework for Dual Rejection
Bayesian Negative Binomial Models (2025)
LOO-CV comparison with posterior predictive checks
ArviZ-based diagnostics
```

---

## Contact

For questions about this assessment:
- Methodology: See `assessment_report.md` → "Methodological Lessons"
- Next steps: See `recommendations.md` → "Implementation Roadmap"
- Technical details: See `code/comprehensive_assessment.py`

---

## Final Verdict

**BOTH MODELS REJECTED**

**Reason**: Polynomial functional form is fundamentally inadequate for this data structure

**Next Step**: Implement changepoint model (Model 3b with sigmoid transition)

**Expected Outcome**: ΔELPD = 15-25 (strong improvement) if regime change hypothesis is correct

**Timeline**: 1-2 weeks for full alternative model exploration

---

**Assessment completed**: 2025-10-29
**Status**: Documentation complete, ready for next modeling iteration
**Files**: 10 files (3 reports, 2 data, 4 plots, 1 code)
