# Experiment 2: Complete Pooling (Fixed Effect) Model

**Model Type:** Bayesian Fixed Effect Meta-Analysis
**Parameterization:** Single parameter (no hierarchy)
**Probabilistic Programming Language:** PyMC
**Status:** In Progress

---

## Model Specification

### Mathematical Formulation

```
Data Model:
  y_i ~ Normal(mu, sigma_i)    for i = 1, ..., 8
  where sigma_i are KNOWN measurement errors

Prior:
  mu ~ Normal(0, 25)           [weakly informative]
```

### Rationale

**Complete Pooling Assumption:**
- All schools share EXACTLY the same true treatment effect
- The 31-point range in observed effects (from -3 to 28) is entirely due to measurement error
- Maximum parsimony (1 parameter vs 10 in hierarchical model)

**Why This Model is Justified:**
- EDA shows no heterogeneity: Q p=0.696, I²=0%, tau²=0
- Variance ratio < 1 (observed variance less than expected under noise)
- Model 1 had p_eff≈1 (effective parameter count near 1)
- Hierarchical model effectively reduces to this

**Normal(0, 25) Prior on mu:**
- Weakly informative: covers [-50, 50] at 2 SD
- Slightly more diffuse than Model 1 (SD=25 vs 20) for fair comparison
- Data dominate: pooled SE=4.07 << prior SD=25

### Expected Behavior (Given EDA)

**Posterior predictions:**
- **mu:** Concentrated around 7.7 (precision-weighted mean), SD ≈ 4.1
- Posterior should match classical meta-analysis pooled estimate
- All schools covered within prediction intervals (large sigma_i)

**Comparison with Model 1:**
- Should have nearly identical mu posterior
- Should have similar or better LOO (more parsimonious)
- Prediction intervals similar (Model 1 tau≈0 anyway)

---

## Falsification Criteria

This model will be ABANDONED if:

1. **Posterior predictive p-value < 0.05** for max(|y_i - mu|/sigma_i)
   - Current max z-score is 1.35 (School 1)
   - If model is correct, seeing z > 1.35 should happen ~35% of time

2. **More than 2 schools outside 95% posterior predictive intervals**
   - Expected: 0.4 schools (8 × 0.05)
   - 3+ schools outside: systematic underfitting

3. **LOO-PIT shows U-shape** (underdispersed)
   - Indicates model too confident

4. **Systematic residual pattern by precision**
   - Plot (y_i - mu) vs 1/sigma_i
   - If correlation detected: heterogeneity signal missed

5. **LOO substantially worse than Model 1**
   - If ΔELPD > 5: hierarchical structure is needed
   - Expected: Similar or better LOO (parsimony advantage)

---

## Expected Results

**Most Likely Outcome:**
- mu ≈ N(7.7, 4.1) (same as Model 1 mu posterior)
- LOO ELPD ≈ -30 to -31 (similar to Model 1: -30.73)
- All posterior predictive checks pass
- p_eff = 1 (exactly 1 parameter)
- **Recommendation:** Use this model for parsimony

**Alternative Outcome (Model 1 preferred):**
- LOO worse by >3 ELPD
- Posterior predictive checks fail
- Residuals show systematic patterns
- **Recommendation:** Hierarchical structure is needed

---

## Validation Pipeline

1. ⏳ **Prior Predictive Check:** Quick check (prior is simple)
2. ⏳ **Posterior Inference:** Fit to data (trivial, <1 min)
3. ⏳ **Posterior Predictive Check:** Validate fit
4. ⏳ **Model Critique:** Compare with Model 1

(Skipping simulation-based validation for this simple model - only 1 parameter to estimate)

---

**Created:** 2025-10-28
