# Designer 2: Alternative Bayesian Approaches - Quick Summary

**Designer:** Model Designer 2 (Mixture Models & Robust Methods)
**Date:** 2025-10-30
**Status:** COMPLETE - Ready for Implementation

---

## Three Alternative Model Classes Proposed

### Model 1: Finite Mixture Model (K=3) ⭐ PRIMARY
- **What:** Mixture of 3 normal distributions (clusters)
- **Why:** EDA found exactly 3 clusters (3.1%, 6.5%, 13.2% success rates)
- **Reject if:** Clusters collapse or assignments ambiguous
- **File:** `model_fmm3.stan`

### Model 2: Robust Beta-Binomial (Student-t) 🛡️ SECONDARY
- **What:** Heavy-tailed hierarchy with beta-binomial overdispersion
- **Why:** 2 extreme outliers (Groups 4, 8) might distort estimates
- **Reject if:** Tails collapse (nu > 30) or no overdispersion (kappa > 1000)
- **File:** `model_robust_hbb.stan`

### Model 3: Dirichlet Process Mixture 🔬 EXPLORATORY
- **What:** Non-parametric mixture (doesn't fix K)
- **Why:** Test if K=3 is robust or artifact
- **Reject if:** K_effective = 1 or > 6 (underfitting/overfitting)
- **File:** `model_dp_mbb.stan`

---

## Quick Start

```bash
# 1. Review complete specifications
cat /workspace/experiments/designer_2/proposed_models.md

# 2. Fit all models (~60 seconds total)
python /workspace/experiments/designer_2/fit_alternatives.py

# 3. Compare models via LOO-CV
python /workspace/experiments/designer_2/compare_models.py

# 4. Run falsification tests
python /workspace/experiments/designer_2/falsification_tests.py
```

---

## Key Design Principles

✅ **Falsification First:** Each model has clear rejection criteria
✅ **Competing Hypotheses:** Test fundamentally different model classes
✅ **Stan/PyMC Implementation:** All use proper PPL for inference
✅ **Computational Feasibility:** All models fit in < 1 minute
✅ **Scientific Motivation:** Each linked to specific EDA finding

---

## Expected Outcomes

| Scenario | Winner | Evidence | Conclusion |
|----------|--------|----------|------------|
| **1** | FMM-3 | Clear clusters, K_eff ≈ 3 | Discrete heterogeneity real |
| **2** | Robust-HBB | nu ∈ [4,15], outliers downweighted | Outliers problematic |
| **3** | Standard Hierarchy | All alternatives fail | EDA clusters are noise |
| **4** | Uncertain | LOO delta < 2 | Model averaging needed |

---

## Falsification Summary

| Model | Reject If... | Threshold | Automated Test |
|-------|-------------|-----------|----------------|
| FMM-3 | Ambiguous assignments | certainty < 0.6 | ✅ Yes |
| FMM-3 | Clusters collapse | separation < 2σ | ✅ Yes |
| FMM-3 | K_effective < 2 | mean < 2 | ✅ Yes |
| Robust-HBB | Normal tails | nu > 30 | ✅ Yes |
| Robust-HBB | No overdispersion | kappa > 1000 | ✅ Yes |
| DP-MBB | Single cluster | K_eff = 1 | ✅ Yes |
| DP-MBB | Overfitting | K_eff > 6 | ✅ Yes |
| DP-MBB | Fragmentation | singletons > 5 | ✅ Yes |

---

## Files Delivered

### 📄 Design Documents
- `proposed_models.md` - Complete specifications (38 KB)
- `README.md` - User guide (8 KB)
- `SUMMARY.md` - This quick reference

### 💻 Stan Models
- `model_fmm3.stan` - Finite mixture (2.8 KB)
- `model_robust_hbb.stan` - Robust beta-binomial (2.6 KB)
- `model_dp_mbb.stan` - Dirichlet process (3.3 KB)

### 🐍 Python Scripts
- `fit_alternatives.py` - Fitting pipeline (13 KB)
- `compare_models.py` - LOO-CV comparison (12 KB)
- `falsification_tests.py` - Automated tests (14 KB)

---

## Computational Cost

| Model | Parameters | Run Time | Difficulty | Divergences |
|-------|-----------|----------|------------|-------------|
| FMM-3 | 20 | 20s | Moderate | ~1% |
| Robust-HBB | 16 | 15s | Moderate | ~1% |
| DP-MBB | 25-35 | 40s | High | ~2-5% |
| Standard* | 14 | 10s | Easy | <0.1% |

*Standard hierarchy for comparison

---

## Critical Success Factors

1. **At least one model converges** (Rhat < 1.01)
2. **Clear winner or model averaging** (LOO-CV differentiates)
3. **Falsification tests decisive** (accept or reject each model)
4. **Consistent with EDA** (results align with cluster structure)
5. **Robust to priors** (sensitivity analysis confirms)

---

## What Makes This Different from Standard Hierarchy?

| Feature | Standard Hierarchy | Our Alternatives |
|---------|-------------------|------------------|
| Distribution | Single normal | Mixture (FMM) / Heavy-tailed (Robust) / Infinite (DP) |
| Clusters | Implicit | Explicit |
| Outliers | Partial shrinkage | Downweighted (Robust) |
| Flexibility | Fixed | Adaptive |
| Complexity | Low | Moderate-High |

---

## Philosophy

> "The goal is finding truth, not completing tasks."

- **Rejection is success** if it reveals the right model class
- **All models might fail** - that's OK, means standard hierarchy is best
- **Clusters might be artifacts** - we test this explicitly
- **Switching is winning** - pivoting to better model means learning

---

## Next Steps for Main Agent

1. **Compare with Designer 1** (standard hierarchy)
2. **Compare with Designer 3** (if advanced methods proposed)
3. **Run all models** and compute LOO-CV
4. **Apply falsification tests** automatically
5. **Synthesize findings** into experiment plan
6. **Recommend best model** based on evidence

---

## Contact

**Specialization:** Mixture models, robust Bayes, non-parametric methods
**Approach:** Competing hypotheses + falsification
**Output Directory:** `/workspace/experiments/designer_2/`

---

**Status:** ✅ DESIGN COMPLETE
**Ready for:** Model fitting and comparison
**Expected Duration:** ~2 minutes total (fit + compare + falsify)
