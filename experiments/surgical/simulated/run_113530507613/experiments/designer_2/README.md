# Alternative Bayesian Model Specifications
## Model Designer 2: Mixture Models and Robust Approaches

**Designer:** Model Designer 2 (Alternative Approaches)
**Date:** 2025-10-30
**Status:** Design Complete, Ready for Implementation

---

## Overview

This directory contains alternative Bayesian model specifications for binomial data with strong heterogeneity. All models differ fundamentally from standard hierarchical approaches and are motivated by specific EDA findings.

**Key EDA Findings that Motivate These Models:**
- 3 distinct clusters identified (K=3 optimal)
- 2 extreme outliers (Groups 4, 8)
- Strong heterogeneity (ICC = 0.42)
- Variance ratio = 2.78 (observed >> expected)

---

## Proposed Models

### 1. Finite Mixture Model (K=3) - **PRIMARY**
**File:** `model_fmm3.stan`

**Motivation:** EDA identified exactly 3 clusters with distinct success rates (3.1%, 6.5%, 13.2%). This model explicitly represents that structure.

**Key Feature:** Mixture of 3 normal distributions on logit scale with ordered constraint for identifiability.

**Falsification Criteria:**
- Reject if cluster assignments ambiguous (certainty < 0.6)
- Reject if clusters collapse (mu_k overlap)
- Reject if K_effective < 2

**Expected Result:** If clusters are real, should see clear assignments with >70% certainty for most groups.

---

### 2. Robust Beta-Binomial with Student-t - **SECONDARY**
**File:** `model_robust_hbb.stan`

**Motivation:** Two extreme outliers (Groups 4, 8) might distort standard hierarchical estimates. Heavy-tailed Student-t distribution provides robustness.

**Key Feature:** Student-t distribution for group effects + beta-binomial likelihood for overdispersion.

**Falsification Criteria:**
- Reject if nu > 30 (tails collapse to normal)
- Reject if kappa > 1000 (no overdispersion)

**Expected Result:** If outliers are problematic, should see nu ∈ [4, 15] (moderately heavy tails).

---

### 3. Dirichlet Process Mixture - **EXPLORATORY**
**File:** `model_dp_mbb.stan`

**Motivation:** Non-parametric approach that doesn't commit to K=3. Tests whether cluster structure is robust to model specification.

**Key Feature:** Infinite mixture model with stick-breaking construction (truncated at K_max=10).

**Falsification Criteria:**
- Reject if K_effective = 1 (no clustering)
- Reject if K_effective > 6 (overfitting)
- Reject if high fragmentation (many singleton clusters)

**Expected Result:** If EDA is correct, K_effective should concentrate around 3.

---

## Files in This Directory

### Core Design Documents
- **`proposed_models.md`** - Complete model specifications with math, justification, and falsification criteria
- **`README.md`** - This file

### Stan Model Files
- **`model_fmm3.stan`** - Finite mixture model implementation
- **`model_robust_hbb.stan`** - Robust beta-binomial implementation
- **`model_dp_mbb.stan`** - Dirichlet process mixture implementation

### Python Implementation Scripts
- **`fit_alternatives.py`** - Fit all three models with diagnostic checks
- **`compare_models.py`** - LOO-CV comparison and posterior predictive checks
- **`falsification_tests.py`** - Automated falsification criteria testing

### Output Directory (Generated After Fitting)
- **`results/`** - Model fits, diagnostics, and comparison results
  - `fmm3/` - Finite mixture results
  - `robust_hbb/` - Robust beta-binomial results
  - `dp_mbb/` - Dirichlet process results
  - `comparison/` - Model comparison plots and tables
  - `falsification/` - Falsification test results

---

## How to Use

### Step 1: Review Design
```bash
# Read complete model specifications
cat proposed_models.md
```

### Step 2: Fit Models
```bash
# Fit all three alternative models
python fit_alternatives.py
```

**Expected Runtime:**
- FMM-3: ~20 seconds
- Robust-HBB: ~15 seconds
- DP-MBB: ~40 seconds (most complex)

### Step 3: Compare Models
```bash
# Compare using LOO-CV
python compare_models.py
```

**Output:**
- LOO-CV comparison table
- Pareto-k diagnostics
- Posterior predictive check plots
- Model stacking weights

### Step 4: Falsification Tests
```bash
# Run automated falsification tests
python falsification_tests.py
```

**Output:**
- Pass/fail for each model
- Detailed diagnostics
- Recommendations

---

## Expected Outcomes

### Scenario 1: Mixture Model Wins
**Evidence:**
- FMM-3 has best LOO-ELPD (delta > 3)
- Clear cluster assignments (prob > 0.7)
- K_effective ≈ 3 in DP-MBB

**Conclusion:** Discrete heterogeneity is real. Groups belong to 3 distinct subpopulations.

**Action:** Use FMM-3 for inference and prediction.

---

### Scenario 2: Robust Model Wins
**Evidence:**
- Robust-HBB has best LOO-ELPD
- nu ∈ [4, 15] (moderately heavy tails)
- Outliers less influential than in normal hierarchy

**Conclusion:** Continuous heterogeneity with extreme outliers.

**Action:** Use Robust-HBB for inference; report outlier-robust estimates.

---

### Scenario 3: Standard Hierarchy Wins
**Evidence:**
- FMM clusters collapse (mu_k similar)
- Robust-HBB has nu > 30 (normal tails)
- DP-MBB has K_effective = 1

**Conclusion:** EDA clusters are sampling artifacts. Simple hierarchy is sufficient.

**Action:** Use standard hierarchical model from Designer 1.

---

### Scenario 4: Model Uncertainty
**Evidence:**
- Multiple models similar (LOO delta < 2)

**Conclusion:** Cluster structure is weak/ambiguous.

**Action:** Use model averaging (Bayesian Model Averaging with stacking weights).

---

## Falsification Philosophy

**Critical Principle:** We WANT to find out if models are wrong. Rejection is success if it reveals truth.

**For Each Model:**
1. **Clear falsification criteria** - What evidence would make us reject it?
2. **Automated tests** - Scripts check criteria automatically
3. **Conservative thresholds** - Rather reject too eagerly than accept bad model
4. **Alternative escape routes** - If all fail, what's next?

**Red Flags (Reconsider Everything):**
- All models fail posterior predictive checks
- Prior-data conflict across all specifications
- Extreme parameter values (e.g., kappa > 10,000)
- Inconsistent results across data subsets

---

## Model Comparison Summary

| Feature | FMM-3 | Robust-HBB | DP-MBB |
|---------|-------|------------|--------|
| **Complexity** | Moderate | Moderate | High |
| **Parameters** | 20 | 16 | 25-35 |
| **Run Time** | 2x | 1.5x | 4x |
| **Interpretability** | High | Moderate | Moderate |
| **Best If...** | Discrete clusters real | Heavy outliers | Uncertain K |
| **Falsification** | Clusters collapse | nu > 30 | K=1 or K>6 |

---

## Key Takeaways

1. **All three models are scientifically motivated** by specific EDA findings
2. **Each has different strengths** - FMM for clusters, Robust for outliers, DP for flexibility
3. **Computational cost increases** with model flexibility
4. **Success = finding the right model**, not forcing all to work
5. **Be ready to abandon all** if standard hierarchy wins (means EDA patterns were noise)

---

## Connection to Other Designers

**Designer 1 (Standard Hierarchy):** Provides baseline for comparison. If all alternatives fail falsification, Designer 1's model wins.

**Designer 3 (Advanced Methods):** May propose complementary approaches (GP, state-space, etc.). Should compare if alternative perspectives needed.

**Synthesis:** Main agent will integrate findings and recommend best approach based on LOO-CV and falsification tests.

---

## References

**Statistical Methods:**
- Finite Mixture Models: Frühwirth-Schnatter (2006)
- Dirichlet Process: Ferguson (1973), Escobar & West (1995)
- Beta-Binomial: Skellam (1948), Williams (1975)
- Model Selection: Vehtari et al. (2017) - LOO-CV for Bayesian models

**Implementation:**
- Stan: Carpenter et al. (2017)
- ArviZ: Kumar et al. (2019)
- CmdStanPy: Stan Development Team

---

## Contact & Questions

**Designer:** Model Designer 2 (Alternative Approaches)
**Specialization:** Mixture models, robust methods, non-parametric Bayes
**Philosophy:** Test multiple competing hypotheses; let data choose

**For Questions:**
- Model specification: See `proposed_models.md` Section 1-3
- Falsification criteria: See `proposed_models.md` Section X.5
- Implementation: See Python scripts with detailed comments

---

**Last Updated:** 2025-10-30
**Status:** Ready for implementation and testing
**Next Step:** Run `fit_alternatives.py` to begin model fitting
