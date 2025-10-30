# Model Designer 3: Robust and Alternative Models
## Executive Summary

**Date:** 2025-10-30  
**Designer:** Model Designer 3 (Robust Models Specialist)  
**Challenge:** 12 groups with 42% outlier rate, severe overdispersion (φ ≈ 3.5-5.1)

---

## What We Built

Three complementary **robust Bayesian models** that handle outliers better than standard hierarchical approaches:

1. **Student-t Hierarchical Model** - Heavy-tailed continuous variation
2. **Horseshoe Prior Model** - Sparse effects with automatic outlier detection  
3. **Mixture Model** - Discrete latent subgroups

All models implemented in Stan with complete validation framework.

---

## Why These Models?

### The Problem with Standard Models

The EDA revealed:
- **5 of 12 groups are outliers** (Groups 2, 5, 8, 11, plus Group 1 with zero successes)
- **Group 8 is extreme:** z=3.94, rate=14.4% vs mean 7.6%
- **Standard Normal priors:** Vulnerable to outlier contamination

**Normal hierarchical model issues:**
- Population mean (μ) inflated by Group 8
- All groups shrink by same factor (~85%)
- Can't distinguish "real outliers" from tail events

### Our Solutions

**Student-t:** Polynomial tails (not exponential) → outliers 1000× more plausible  
**Horseshoe:** Adaptive shrinkage → outliers shrink <50%, normals shrink >95%  
**Mixture:** Explicit clusters → separate "outlier population" from "normal population"

---

## Key Files

```
experiments/designer_3/
├── experiment_plan.md                  # This document - comprehensive plan
├── proposed_models.md                  # Detailed mathematical specifications
├── model_comparison_table.md           # Quick reference comparison
├── README.md                           # Usage instructions
├── SUMMARY.md                          # This executive summary
│
├── student_t_hierarchical.stan         # Model 1: Heavy-tailed effects
├── horseshoe_hierarchical.stan         # Model 2: Sparse effects
├── mixture_hierarchical.stan           # Model 3: Latent clusters
│
└── fit_robust_models.py                # Python script to fit all models
```

---

## Quick Start

### Install Dependencies
```bash
pip install cmdstanpy arviz numpy pandas matplotlib seaborn scipy
```

### Fit Models
```bash
cd /workspace/experiments/designer_3

# Fit all three models and compare
python fit_robust_models.py --model all

# Or fit individually
python fit_robust_models.py --model student_t
python fit_robust_models.py --model horseshoe
python fit_robust_models.py --model mixture
```

### Expected Runtime
- Student-t: ~2-5 minutes
- Horseshoe: ~5-10 minutes  
- Mixture: ~10-20 minutes
- **Total: ~30-40 minutes** for all three + diagnostics

---

## Model Recommendations

### Primary: Student-t Hierarchical

**Use as default robust model**

✓ Best balance of robustness and simplicity  
✓ Well-studied, easy to explain  
✓ Automatically detects if robustness needed (via ν posterior)

**Abandon if:**
- ν > 50 (use Normal hierarchical instead)
- ν < 2 (switch to mixture model)
- Posterior predictive fails

### Sensitivity: Horseshoe Prior

**Test sparsity hypothesis**

✓ Better if most groups truly identical  
✓ Automatic outlier identification  
✓ Often improves out-of-sample prediction

**Abandon if:**
- All λ_i similar (no sparsity detected)
- Computational issues
- Interpretation unclear

### Exploratory: Mixture Model

**Only if discrete clusters suspected**

⚠️ N=12 is small for reliable clustering  
⚠️ High risk of overinterpretation  
⚠️ Label switching issues

**Abandon if:**
- π → 0 or 1 (no mixture)
- μ_1 ≈ μ_2 (no separation)
- Cluster assignments ambiguous

---

## Falsification Criteria

### Built-in Model Rejection Rules

Each model has **clear posterior signals** for when it's wrong:

| Model | Reject if... | Posterior Diagnostic |
|-------|-------------|---------------------|
| Student-t | ν > 50 | Data doesn't need heavy tails |
| Student-t | ν < 2 | Need discrete mixture |
| Horseshoe | All λ ≈ 0.5 | No sparsity |
| Horseshoe | τ >> τ₀ | Too many effects |
| Mixture | π < 0.1 or > 0.9 | No mixture |
| Mixture | μ₂ - μ₁ < 0.5 | No separation |

### Red Flags (All Models)

Stop and reconsider if:
- Posterior predictive checks fail
- High Pareto k (> 0.7) for multiple groups
- Computational pathologies (divergences, non-convergence)
- Prior-posterior conflict

---

## Expected Results

### For Our Dataset

Based on EDA (12 groups, mean rate 7.6%, Group 8 at 14.4%):

**Student-t Model:**
- ν ≈ 8-15 (moderate heavy tails)
- μ ≈ -2.6 on logit scale (≈ 6.9% probability)
- Group 8: α₈ ≈ +1.5, less shrinkage than Normal
- Group 1: α₁ ≈ -1.2, moderate shrinkage to ~1-2%

**Horseshoe Model:**
- 3-5 groups "active" (λ > 0.5)
- Groups 2, 8, 11 likely active
- Others heavily shrunk (λ < 0.2)
- Better LOO-CV if sparsity real

**Mixture Model:**
- Cluster 1 (normal): 8-9 groups, μ₁ ≈ -2.8 (6.5%)
- Cluster 2 (outlier): 3-4 groups, μ₂ ≈ -1.9 (13%)
- Mixing proportion: π ≈ 0.7 (70% normal, 30% outlier)
- Groups 2, 8, 11 in outlier cluster with high probability

---

## Validation Framework

### Posterior Predictive Checks

Model must reproduce:
1. ✓ Overdispersion (φ ≈ 3.5-5.1)
2. ✓ Outlier frequency (5 of 12 outside 95% limits)
3. ✓ Zero counts (Group 1: 0/47)
4. ✓ Range of rates (0% to 14.4%)

### Stress Tests

**Test 1:** Exclude Group 8  
→ Robust model's μ should be stable

**Test 2:** Simulation-based calibration  
→ Can recover known parameters?

**Test 3:** Prior sensitivity  
→ Conclusions robust to prior changes?

**Test 4:** Cross-validation  
→ 90% of groups within 50% prediction interval?

---

## Success Criteria

### This modeling succeeds if:

1. ✓ At least one model validates
2. ✓ Outlier uncertainty quantified
3. ✓ Population mean robustly estimated
4. ✓ Falsification criteria clear
5. ✓ Results interpretable

### Finding robustness unnecessary = SUCCESS

If posterior ν → ∞ (Student-t reduces to Normal), that's **valuable information:**
- Data cleaner than suspected
- Standard model sufficient
- Complexity not warranted

**Parsimony is a feature, not a bug.**

---

## Model Selection Guide

### LOO-CV Decision Rules

- **ΔLOO < 2:** Models equivalent → choose simplest (Student-t)
- **ΔLOO ∈ [2, 10]:** Some evidence → check interpretability
- **ΔLOO > 10:** Clear winner → use best model
- **Pareto k > 0.7:** LOO unreliable → investigate influential points

### Interpretation Priority

1. **Posterior diagnostics** (Rhat, ESS, divergences)
2. **Falsification checks** (ν, λ, π posteriors)
3. **Posterior predictive** (can reproduce patterns?)
4. **LOO-CV** (out-of-sample prediction)
5. **Scientific plausibility** (does it make sense?)

---

## Common Pitfalls

### Pitfall 1: Overconfidence with N=12

**Problem:** 12 groups is small for complex models  
**Solution:** Be cautious about strong claims, report uncertainty

### Pitfall 2: Ignoring computational warnings

**Problem:** Divergences often indicate model misspecification  
**Solution:** Don't just increase adapt_delta, investigate cause

### Pitfall 3: Forcing complexity

**Problem:** Using mixture model when simpler model fits  
**Solution:** Accept that data may not need complexity

### Pitfall 4: Misinterpreting sparsity

**Problem:** Thinking λᵢ < 0.2 means "group i is irrelevant"  
**Solution:** It means "group i is near population mean"

---

## Theoretical Foundations

### Why Student-t for Robustness?

Normal: P(|z| > 4) ≈ 10⁻⁴ (exponential tails)  
Student-t(ν=5): P(|z| > 4) ≈ 10⁻³ (polynomial tails)

**Practical impact:**
- Group 8 (z=3.94) is 10× more plausible under Student-t
- Less shrinkage, less contamination of μ

### Why Horseshoe for Sparsity?

Normal prior: All groups shrink by constant factor  
Horseshoe: Adaptive shrinkage per group

**Key insight:**
- Small effects: λ → 0, shrinkage → 100%
- Large effects: λ → ∞, shrinkage → 0%
- Data determines which regime each group is in

### Why Mixture for Clusters?

Single component: Outliers are tail events  
Mixture: Outliers are different population

**Philosophical shift:**
- Not "how extreme is Group 8?"
- But "which population does Group 8 belong to?"

---

## Computational Notes

### Sampling Strategy

All models use:
- 4 chains (check convergence)
- adapt_delta = 0.95 (control divergences)
- max_treedepth = 12 (allow complex geometry)
- Non-centered parameterization (avoid funnels)

**Iterations:**
- Student-t: 2000 (1000 warmup)
- Horseshoe: 3000 (1500 warmup) - Cauchy priors slower
- Mixture: 4000 (2000 warmup) - marginalizing expensive

### Diagnostics to Check

1. **Rhat < 1.01** for all parameters
2. **ESS > 400** for key parameters
3. **No divergences** (or very few < 1%)
4. **Trace plots** show good mixing
5. **Pairs plots** check for degeneracies

---

## Alternative Approaches

### If All Models Fail

**Backup Plan A: Negative Binomial**
- Different overdispersion mechanism
- Useful if beta-binomial family inadequate

**Backup Plan B: Zero-Inflated Model**
- If Group 1 (zero count) is systematically different
- Two processes: zero vs non-zero, then count

**Backup Plan C: Data Investigation**
- Verify Group 8 data (z=3.94 is very extreme)
- Check for measurement errors
- Consider temporal stability

---

## Key Insights

### 1. Robustness ≠ Complexity

Student-t adds one parameter (ν) but provides substantial robustness.  
If ν → ∞, model automatically simplifies to Normal.

### 2. Sparsity is Testable

Horseshoe directly tests "are most groups identical?"  
Posterior λ distribution answers this question.

### 3. Discrete Clusters Require Evidence

With N=12, be skeptical of mixture models.  
Need clear separation (π away from 0/1, μ₂ >> μ₁).

### 4. LOO-CV Favors Prediction

Model with best LOO may not be most interpretable.  
Balance predictive accuracy with scientific understanding.

### 5. Outliers ≠ Errors

Robust models protect against outliers, but:
- Can't fix data quality issues
- Can't determine if outliers are errors
- Should always report sensitivity

---

## Documentation Structure

### For Quick Reference
→ `model_comparison_table.md` (side-by-side comparison)

### For Understanding
→ `proposed_models.md` (detailed specifications)

### For Implementation
→ `README.md` (usage instructions)

### For Planning
→ `experiment_plan.md` (comprehensive strategy)

### For Overview
→ `SUMMARY.md` (this document)

---

## Final Recommendations

### Start Here

1. **Fit Student-t model** (primary robust model)
2. **Check ν posterior** (is robustness needed?)
3. **Validate with posterior predictive checks**
4. **If passes:** Use it. Report ν to justify robustness.
5. **If fails:** Fit alternatives (Horseshoe, Mixture)

### Report This

**Minimum reporting:**
- Which model(s) fit
- Posterior diagnostics (Rhat, ESS)
- Key parameters (μ, σ, ν or λ or π)
- Falsification checks
- Posterior predictive results

**Full reporting:**
- All three models + comparison
- LOO-CV results
- Stress tests (exclude Group 8)
- Sensitivity analyses
- Limitations and assumptions

---

## Contact

**Model Designer:** Designer 3 (Robust Models Specialist)  
**Role:** Alternative approaches and robustness features  
**Focus:** Handling outlier-heavy datasets

**Related Designers:**
- Designer 1: Standard hierarchical models
- Designer 2: Advanced model classes
- Designer 3: This work (robust models)

---

## Philosophy

> "A good robust model should tell you when robustness is unnecessary."

If Student-t posterior shows ν → ∞, that's **success** - it means:
- Standard Normal hierarchical is sufficient
- Data doesn't need heavy tails
- We learned something valuable

> "The goal is truth, not task completion."

If all models fail validation, that's **important** - it means:
- Data generation process not captured
- Need different model class
- Or data quality issues exist

> "Switching model classes is success, not failure."

If we abandon Student-t for Mixture based on posteriors, that's **learning** - it means:
- We responded to evidence
- We didn't force a predetermined model
- Science is working as intended

---

**End of Summary**

All files in: `/workspace/experiments/designer_3/`  
Ready to run: `python fit_robust_models.py --model all`
