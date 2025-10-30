# Model Selection Recommendation: Eight Schools Analysis

**Date:** 2025-10-28
**Decision:** Use **Complete Pooling Model** for final inference

---

## Decision Summary

After comprehensive LOO-CV comparison of the hierarchical and complete pooling models, we recommend the **complete pooling model** (single μ parameter) for final inference in the Eight Schools analysis.

---

## Quantitative Justification

### LOO-CV Comparison

| Criterion | Hierarchical | Complete Pooling | Advantage |
|-----------|--------------|------------------|-----------|
| **ELPD** | -30.73 ± 1.04 | -30.52 ± 1.12 | Pooling (+0.21) |
| **ΔELPD** | 0.21 ± 0.11 | - | Not significant* |
| **p_loo** | 1.03 | 0.64 | Pooling (simpler) |
| **Akaike weight** | 0.000 | 1.000 | Pooling (decisive) |
| **Max Pareto k** | 0.634 | 0.285 | Pooling (more reliable) |
| **RMSE** | 8.98 | 9.84 | Hierarchical (-0.86) |
| **MAE** | 6.99 | 7.75 | Hierarchical (-0.76) |

*Statistical significance threshold: |ΔELPD| > 2×SE = 0.22. Observed difference (0.21) does not meet threshold.

### Key Findings

1. **No significant predictive difference:** ΔELPD = 0.21 ± 0.11 (not significant)
2. **Simpler model:** Complete pooling uses 0.64 effective parameters vs 1.03 for hierarchical
3. **Better diagnostics:** All Pareto k < 0.5 for pooling vs 3/8 > 0.5 for hierarchical
4. **Parsimony principle:** When models perform equivalently, choose the simpler one

---

## Key Visual Evidence

### 1. LOO Comparison (`loo_comparison_plot.png`)
Shows overlapping confidence intervals for ELPD estimates, confirming models are statistically equivalent.

### 2. Pareto k Diagnostics (`pareto_k_comparison.png`)
Complete pooling model shows excellent k values (all < 0.5), while hierarchical has 3 observations with k ∈ [0.5, 0.7].

### 3. Prediction Comparison (`prediction_comparison.png`)
Four-panel comparison showing:
- Similar posterior means between models
- Similar prediction errors
- Nearly identical uncertainty quantification
- Strong agreement in predictions (scatter near diagonal)

**Interpretation:** Models make essentially the same predictions, confirming LOO-CV results.

### 4. Pointwise LOO (`pointwise_loo_comparison.png`)
Complete pooling performs better in 6/8 schools, with no systematic pattern favoring hierarchical model.

---

## Rationale

### Statistical Rationale

1. **LOO-CV equivalence:** The "gold standard" for Bayesian model comparison shows no meaningful difference
2. **Effective parameters:** Hierarchical model uses only ~1 effective parameter despite having 10 nominal parameters, indicating complete shrinkage
3. **Model weights:** Akaike weights place 100% weight on complete pooling
4. **Diagnostic reliability:** Complete pooling has more reliable LOO estimates (better Pareto k values)

### Scientific Rationale

1. **No heterogeneity detected:** Multiple lines of evidence (EDA I² = 0%, hierarchical model shrinkage, LOO equivalence) all point to absence of between-school variation
2. **Limited data:** 8 observations insufficient to reliably estimate 8 school-specific effects
3. **Large uncertainty:** Within-school SEs (9-18) swamp any potential between-school differences
4. **Occam's razor:** Adding complexity without improving predictions violates parsimony

### Practical Rationale

1. **Simplicity:** Easier to explain and communicate ("single treatment effect")
2. **Efficiency:** Faster computation, fewer parameters to monitor
3. **Clarity:** Directly reflects what data support (no heterogeneity)
4. **Reproducibility:** Simpler model is easier to reproduce and extend

---

## What This Means for Inference

### School-Specific Estimates

**Do not** report school-specific estimates from the hierarchical model. The data do not support reliable estimation of school-specific effects.

**Do** report the pooled estimate for all schools:
- **Pooled treatment effect:** μ = 7.55
- **95% Credible Interval:** [-0.21, 15.31]
- **Standard Deviation:** 4.00

### Interpretation

"There is no evidence for differences in treatment effects across schools. The best estimate for all schools is a treatment effect of μ = 7.55 ± 4.00. Large uncertainty reflects limited data (n=8) and substantial within-school variability, not heterogeneity of effects."

---

## When Might We Reconsider?

The hierarchical model might be preferred if:

1. **More data:** With n_schools > 30-50, hierarchical models can better detect heterogeneity
2. **Strong prior beliefs:** Domain expertise suggests schools should differ
3. **Covariate information:** School characteristics that might explain heterogeneity
4. **Conservative approach:** Explicit desire to acknowledge structural hierarchy even without evidence

For the current Eight Schools dataset with 8 observations, **none of these conditions apply**.

---

## Recommended Reporting

### Main Text

> "We compared a hierarchical model allowing for between-school variation with a complete pooling model assuming a common treatment effect. Leave-one-out cross-validation showed the models were statistically indistinguishable (ΔELPD = 0.21 ± 0.11), with the simpler complete pooling model having superior diagnostic properties (all Pareto k < 0.5 vs 3/8 observations with k > 0.5 for the hierarchical model). The hierarchical model's effective parameter count (p_loo = 1.03) indicated complete shrinkage, consistent with absence of heterogeneity. We therefore report a pooled treatment effect of μ = 7.55 (95% CI: [-0.21, 15.31])."

### Supplementary Material

Report both models for transparency:

| Model | μ estimate | 95% CI | ELPD | p_loo |
|-------|------------|---------|------|-------|
| **Complete Pooling (primary)** | **7.55** | **[-0.21, 15.31]** | **-30.52 ± 1.12** | **0.64** |
| Hierarchical (sensitivity) | 7.36 | [-0.97, 15.69] | -30.73 ± 1.04 | 1.03 |

Difference in point estimates (0.19) is negligible compared to posterior uncertainty (±4.00).

---

## Implementation

### For Phase 5 (Model Adequacy)

Use the **Complete Pooling model** posterior samples from:
- `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`

### Final Estimates

Primary parameter of interest:
- **μ:** 7.55 ± 4.00 (treatment effect, all schools)

Report uncertainty appropriately:
- Point estimate: 7.55
- 95% CI: [-0.21, 15.31]
- Posterior SD: 4.00

### Graphical Abstract

Use `prediction_comparison.png` to show:
- Both models make similar predictions
- All schools estimated near pooled mean
- Large uncertainty dominates small between-school differences

---

## Conclusion

The complete pooling model is the clear choice for the Eight Schools analysis based on:
1. Statistical equivalence in predictive performance
2. Superior simplicity and diagnostic properties
3. Scientific interpretation aligned with data
4. Practical advantages in communication and computation

This decision reflects best practices in Bayesian workflow:
- Use cross-validation for model comparison
- Check diagnostic reliability
- Apply parsimony when appropriate
- Connect statistical decisions to scientific interpretation

**Decision:** ACCEPT complete pooling model as the primary model for inference.

---

## References

- Model comparison code: `/workspace/experiments/model_comparison/code/comprehensive_assessment_v2.py`
- Detailed report: `/workspace/experiments/model_comparison/comparison_report.md`
- Figures: `/workspace/experiments/model_comparison/figures/`
- Data: `/workspace/experiments/model_comparison/loo_comparison.csv`
