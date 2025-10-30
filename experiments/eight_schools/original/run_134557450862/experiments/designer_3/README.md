# Designer 3: Robust Models and Alternative Specifications

## Quick Start

This directory contains **Designer 3's independent model design** for the Eight Schools meta-analysis, focusing on **robustness and sensitivity analysis**.

### Key Files

1. **proposed_models.md** (32 KB) - Main design document with full mathematical specifications
2. **model_comparison_matrix.md** (10 KB) - Quick reference tables and decision trees
3. **implementation_roadmap.md** (17 KB) - Step-by-step implementation guide
4. **README.md** (this file) - Overview and quick navigation

### Core Philosophy

**Critical Question:** Do we need robust models at all?

The EDA shows NO outliers, NO heterogeneity, and perfect consistency with normal assumptions. This design **tests whether robustness is necessary** rather than assuming it is.

**Key Principle:** If robust models converge to normal models, that's EVIDENCE that normality is appropriate (not a failure).

## Three Model Classes Proposed

### 1. Student-t Models (Heavy-Tailed Alternatives)
- **Model 1A:** Student-t data distribution
- **Model 1B:** Student-t random effects
- **Model 1C:** Double Student-t (both data and effects)
- **Falsification:** If nu_posterior > 30, abandon and use normal models

### 2. Mixture Models (Outlier Detection)
- **Model 2A:** Outlier indicator model (inflate variance for outliers)
- **Model 2B:** Latent class model (two subgroups)
- **Model 2C:** Dirichlet process (fully nonparametric)
- **Falsification:** If posterior favors K=1 cluster, abandon and use simple hierarchical

### 3. Prior Sensitivity Analysis
- **Grid search:** 6 mu priors × 6 tau priors = 36 models
- **Metrics:** Relative_Sensitivity = (max - min) / posterior_SD
- **Threshold:** If Rel_Sens < 0.5, conclusions are robust to prior choices

## Expected Outcomes

**Most Likely (85% confidence):** Robust models converge to normal
- nu_posterior > 30 (Student-t effectively normal)
- All p_i < 0.2 (no outliers detected)
- Relative_Sensitivity < 0.5 (low prior dependence)
- **Conclusion:** Normal hierarchical model is appropriate

**Possible (10% confidence):** Mild robustness benefits
- nu_posterior = 10-30 (borderline heavy tails)
- Some p_i = 0.2-0.5 (uncertain outliers)
- Relative_Sensitivity = 0.5-1.0 (moderate prior dependence)
- **Conclusion:** Report range, use conservative priors

**Unlikely (5% confidence):** Strong robustness needed
- nu_posterior < 10 (very heavy tails)
- Some p_i > 0.7 (clear outliers)
- Relative_Sensitivity > 1.0 (high prior dependence)
- **Conclusion:** EDA missed something important, investigate

## Implementation Priority

1. **Phase 1:** Normal baseline (30 min) - ALWAYS DO
2. **Phase 2:** Student-t and outlier models (45 min) - ALWAYS DO
3. **Phase 3:** Extended robustness (60 min) - CONDITIONAL (if Phase 2 finds issues)
4. **Phase 4:** Prior sensitivity (45 min) - ALWAYS DO
5. **Phase 5:** Synthesis and reporting (30 min) - ALWAYS DO

**Total time:** 2-3 hours with parallelization

## Decision Points

### After Phase 1 (Baseline)
- **Green:** R-hat < 1.01, ESS > 400, divergences < 1% → CONTINUE
- **Yellow:** Marginal diagnostics → FIX parameterization, then continue
- **Red:** Poor convergence → STOP, reconsider model structure

### After Phase 2 (Core Robustness)
- **If nu > 30 AND p_i < 0.2:** Robustness NOT needed → SKIP Phase 3
- **If nu < 10 OR p_i > 0.7:** Robustness NEEDED → PRIORITY Phase 3
- **If borderline:** Complete Phase 3 for thoroughness

### After Phase 4 (Prior Sensitivity)
- **If Rel_Sens < 0.5:** Robust to priors → Report with weak informative prior
- **If Rel_Sens > 1.0:** Sensitive to priors → Report range, justify prior choice

## Key Contributions (vs Other Designers)

**Designer 3's Unique Focus:**
- Systematic distributional robustness (Student-t, mixtures)
- Quantitative sensitivity metrics (Relative_Sensitivity)
- Decision thresholds and falsification criteria
- Critical perspective: "Do we need robustness at all?"

**Expected Overlap:**
- All designers will likely propose hierarchical models
- Student-t may be proposed by multiple designers
- Prior sensitivity should be common

**Value of Multiple Designers:**
- If we converge → strong evidence for conclusions
- If we diverge → reveals genuine uncertainty
- Either outcome is scientifically valuable

## Critical Red Flags

**Abandon ALL models if:**
1. Prior-posterior conflict across all model classes
2. Extreme parameter values (tau > 50, nu < 3)
3. Computational breakdown in simple models
4. Posterior predictive checks fail universally
5. Inconsistent conclusions across independent analyses

**These suggest fundamental model misspecification, not just wrong distribution.**

## Quick Reference

### When to Use Each Model

| Scenario | Recommended Model | Evidence |
|----------|-------------------|----------|
| Standard case (as EDA suggests) | Normal hierarchical | All robustness checks pass |
| Suspected outliers | Outlier indicator (2A) | Some p_i > 0.5 |
| Heavy tails | Student-t (1A) | nu_posterior < 20 |
| Unknown structure | Prior sensitivity | Always, as validation |
| High uncertainty | Report range | Relative_Sensitivity > 1.0 |

### Interpretation Thresholds

**Student-t degrees of freedom (nu):**
- nu > 30: Effectively normal
- nu = 10-30: Borderline heavy tails
- nu < 10: Strong evidence for heavy tails

**Outlier probability (p_i):**
- p_i < 0.2: Not an outlier
- p_i = 0.2-0.7: Uncertain
- p_i > 0.7: Likely outlier

**Prior sensitivity (Relative_Sensitivity):**
- < 0.5: Low sensitivity (robust)
- 0.5-1.0: Moderate sensitivity
- > 1.0: High sensitivity (prior dominates)

## Files to Generate (After Implementation)

```
code/
├── baseline_normal_model.py
├── model_1a_student_t.py
├── model_2a_outlier_indicators.py
├── fit_prior_grid.py
└── comparison_plots.py

results/
├── baseline_results.md
├── model_comparison.csv
├── prior_sensitivity_results.csv
└── robustness_report.md

figures/
├── comparison_forest.png
├── prior_sensitivity_heatmap.png
└── shrinkage_comparison.png
```

## Integration Strategy

### With Other Designers
1. Compare model classes proposed
2. Identify overlap (likely hierarchical + Student-t common)
3. Focus detailed analysis on areas of disagreement
4. Use agreement as validation of conclusions

### Final Synthesis
- If all designers agree: Report consensus with robustness validation
- If designers disagree: Investigate if due to distributional assumptions
- If fundamental disagreement: Report range, acknowledge uncertainty

## Contact and Questions

**Designer:** Designer 3 (Independent - Robust Models Specialist)
**Design Date:** 2025-10-28
**Status:** Design Complete - Ready for Implementation

**For questions about:**
- Mathematical specifications → see `proposed_models.md`
- Decision thresholds → see `model_comparison_matrix.md`
- Implementation steps → see `implementation_roadmap.md`
- Quick reference → this file

---

**Remember:** The goal is finding truth, not proving robustness is needed. Negative results (robustness not needed) are equally valuable as positive results (robustness matters).

**Success = Learning whether assumptions matter, not always preferring complex models.**
