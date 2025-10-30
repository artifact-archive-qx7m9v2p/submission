# Designer 2: Complete File Index

## Overview
This directory contains complete specifications and implementations for hierarchical binomial models with random effects, designed to address severe overdispersion (φ ≈ 3.5-5.1) through group-level random effects on the logit scale.

---

## File Organization

### 📋 Documentation (Start Here)

1. **`INDEX.md`** (this file)
   - Navigation guide to all files
   - Quick reference for what to read when

2. **`QUICK_START.md`** ⭐ START HERE
   - 5-minute quick start guide
   - Essential commands and expected outputs
   - Troubleshooting common issues
   - **Read this first if you want to run the models immediately**

3. **`README.md`**
   - Comprehensive overview of all three models
   - Design decisions and justifications
   - Usage instructions and examples
   - Success criteria and failure modes
   - **Read this for complete understanding**

4. **`proposed_models.md`** 📚 DETAILED DESIGN
   - Full mathematical specifications
   - Detailed prior justifications
   - Falsification criteria for each model
   - Expected posterior behavior
   - Critical thinking about model failures
   - **Read this to understand the theoretical foundation**

5. **`MODEL_DECISION_TREE.md`** 🗺️ DECISION GUIDE
   - Flowchart for model selection
   - Decision points and thresholds
   - When to pivot to alternative models
   - Success/failure criteria matrix
   - **Read this when deciding which model to use**

---

### 💻 Stan Model Implementations

6. **`model1_centered.stan`**
   - Centered parameterization: logit(p_i) = μ + α_i
   - Standard hierarchical model structure
   - **Status:** Baseline, may have computational issues
   - **Use:** Demonstrating importance of reparameterization

7. **`model2_noncentered.stan`** ⭐ RECOMMENDED
   - Non-centered: logit(p_i) = μ + σ·z_i
   - Eliminates funnel geometry
   - **Status:** Primary model for inference
   - **Use:** Most likely to converge reliably

8. **`model3_robust.stan`**
   - Student-t priors: z_i ~ Student-t(ν, 0, 1)
   - Heavy tails for outlier robustness
   - **Status:** Sensitivity analysis
   - **Use:** Assessing impact of Group 8 outlier

---

### 🐍 Python Implementation Scripts

9. **`fit_models.py`** ⭐ MAIN FITTING SCRIPT
   - Fits all three models using CmdStanPy
   - Comprehensive MCMC diagnostics
   - Posterior summaries and checks
   - **Command:** `python fit_models.py --model 2`
   - **Runtime:** 2-5 minutes per model

10. **`model_comparison.py`**
    - Compares fitted models
    - LOO-CV analysis
    - Creates visualizations
    - Summary reports
    - **Command:** `python model_comparison.py`
    - **Runtime:** 2-3 minutes

---

## Reading Guide by Use Case

### 🎯 "I just want to fit models and get results"
1. Read: `QUICK_START.md` (5 min)
2. Run: `python fit_models.py --model 2`
3. Check: Diagnostics in console output
4. Review: `results/*_summary.csv`

**Total time:** 15 minutes

---

### 🔬 "I want to understand the design"
1. Read: `README.md` (15 min)
2. Read: `proposed_models.md` (30 min)
3. Review: Stan files to see implementation
4. Check: Mathematical derivations in proposed_models.md

**Total time:** 1 hour

---

### 🤔 "I'm deciding which model to use"
1. Read: `MODEL_DECISION_TREE.md` (10 min)
2. Run: `python fit_models.py --model all`
3. Run: `python model_comparison.py`
4. Review: `visualizations/comparison_report.txt`
5. Check: LOO-CV results and diagnostic comparisons

**Total time:** 30 minutes

---

### 🐛 "Something went wrong"
1. Check: `QUICK_START.md` troubleshooting section
2. Review: Diagnostic output from `fit_models.py`
3. Check: `MODEL_DECISION_TREE.md` red flags section
4. Read: Falsification criteria in `proposed_models.md`
5. Consider: Switching to beta-binomial (Designer 1)

---

### 📊 "I need to compare with other designers"
1. Ensure: All models fitted successfully
2. Run: `python model_comparison.py`
3. Compare: LOO-CV scores across designers
4. Check: Stacking weights
5. Review: Predictive performance metrics

---

## Directory Structure (After Running)

```
experiments/designer_2/
│
├── Documentation
│   ├── INDEX.md                      (this file)
│   ├── QUICK_START.md               (5-min guide)
│   ├── README.md                    (comprehensive)
│   ├── proposed_models.md           (detailed design)
│   └── MODEL_DECISION_TREE.md       (decision guide)
│
├── Stan Models
│   ├── model1_centered.stan         (baseline)
│   ├── model2_noncentered.stan      (recommended)
│   └── model3_robust.stan           (outlier-robust)
│
├── Python Scripts
│   ├── fit_models.py                (main fitting)
│   └── model_comparison.py          (comparison & viz)
│
├── Results (created after fitting)
│   ├── M1_*_summary.csv
│   ├── M1_*_diagnostics.json
│   ├── M1_*_draws.csv
│   ├── M2_*_summary.csv
│   ├── M2_*_diagnostics.json
│   ├── M2_*_draws.csv
│   ├── M3_*_summary.csv
│   ├── M3_*_diagnostics.json
│   └── M3_*_draws.csv
│
├── Visualizations (created after comparison)
│   ├── comparison_report.txt        (summary text)
│   ├── population_parameters_comparison.png
│   ├── group_posteriors_comparison.png
│   ├── shrinkage_comparison.png
│   ├── overdispersion_check.png
│   ├── group1_zero_count_shrinkage.png
│   ├── nu_posterior.png             (if M3 fitted)
│   └── diagnostic_comparison.csv
│
└── Stan Output (temporary files)
    └── model_*_*.csv
```

---

## Key Features of This Design

### ✅ Strengths
- **Principled shrinkage:** No ad-hoc corrections for zero counts
- **Interpretable:** Clear parameter meanings (μ, σ, α_i)
- **Flexible:** Can add group-level covariates
- **Robust:** Three variants for different scenarios
- **Well-tested:** Non-centered parameterization for reliability

### ⚠️ Limitations
- **More complex:** 12 parameters vs 2 for beta-binomial
- **Computationally intensive:** Requires MCMC sampling
- **Assumes continuous distribution:** May fail if discrete subgroups
- **Logit scale:** Between-group SD not directly interpretable

---

## Model Summary Table

| Model | Parameterization | Prior on σ | Strengths | When to Use |
|-------|------------------|------------|-----------|-------------|
| **M1** | Centered | Half-Cauchy(0,1) | Standard, interpretable | Demonstrating funnel problem |
| **M2** | Non-centered | Half-Normal(0,1) | Efficient sampling | **Primary analysis** |
| **M3** | Non-centered + Student-t | Half-Student-t(3,0,1) | Outlier robust | **Sensitivity analysis** |

---

## Critical Design Decisions

### 1. Why Non-Centered (M2)?
- High ICC (0.73) → strong shrinkage → funnel geometry in centered
- Non-centered eliminates σ-α correlation
- Expected 10-100× reduction in divergences

### 2. Why Logit Link?
- Natural for probabilities [0, 1]
- Group effects plausibly normal on logit scale
- Standard in binomial regression

### 3. How is Group 1 (0/47) Handled?
- **No continuity correction**
- Hierarchical shrinkage pulls toward μ
- Expected posterior: 1-3% (low but not zero)

### 4. Why Robust Model (M3)?
- Group 8 is extreme (z = 3.94)
- Heavy tails allow outliers without distortion
- Posterior ν tells if it was necessary

---

## Expected Results

### If Models Succeed:

**Population parameters:**
- μ ≈ -2.5 (7.5% on probability scale)
- σ ≈ 0.9 (consistent with ICC = 0.73)
- φ ≈ 3.5-5.1 (reproduces overdispersion)

**Group-specific:**
- Group 1: Shrink from 0% to ~2%
- Group 8: Shrink from 14.4% to ~11-12%
- Small-n groups shrink more

**Diagnostics:**
- Rhat < 1.01, ESS > 400, Divergences < 1%

### If Models Fail:

**Computational failure:** Switch to beta-binomial
**Statistical failure:** Can't reproduce φ → Try mixture
**Outlier problems:** Multiple Pareto k > 0.7 → Finite mixture

---

## Quick Reference Commands

```bash
# Fit recommended model (M2)
python fit_models.py --model 2

# Fit all models
python fit_models.py --model all

# Fit with higher adapt_delta (if divergences)
python fit_models.py --model 2 --adapt_delta 0.95

# Compare models
python model_comparison.py

# Check diagnostics quickly
grep "PASS\|FAIL" results/*_diagnostics.json
```

---

## Contact Points with Other Designers

### vs Designer 1 (Beta-Binomial)
- **Compare:** LOO-CV scores
- **Trade-off:** Complexity vs simplicity
- **Use hierarchical if:** Group-specific effects important
- **Use beta-binomial if:** Similar performance, more parsimonious

### vs Designer 3 (Alternative Approaches)
- **Compare:** LOO-CV and predictive checks
- **Consider:** Stacking weights if models competitive
- **Document:** Why hierarchical binomial chosen/rejected

---

## Success Criteria Checklist

Before declaring this approach successful:

- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 for all parameters
- [ ] Divergences < 1%
- [ ] Posterior φ ∈ [3.0, 6.0]
- [ ] σ ∈ [0.5, 1.5]
- [ ] Group 1 posterior ≈ 1-3%
- [ ] LOO Pareto k < 0.7 for all groups
- [ ] Posterior predictive matches observed
- [ ] Compared to other designers
- [ ] Results scientifically interpretable

---

## Citation

If using this model design:

```
Hierarchical binomial model with non-centered parameterization
addressing severe overdispersion (φ ≈ 3.5-5.1) through group-level
random effects. Handles zero counts via shrinkage without ad-hoc
corrections. Implemented in Stan via CmdStanPy.
```

Key references:
- Gelman & Hill (2007) for hierarchical models
- Stan manual for non-centered parameterization
- Williams (1982) for overdispersion in binomial data

---

## Final Recommendations

1. **Start with M2** (non-centered) - most likely to work
2. **Check diagnostics carefully** - don't trust results without verification
3. **Compare to Designer 1** - beta-binomial might be simpler
4. **Document failures** - negative results are valuable
5. **Be ready to pivot** - finding model inadequacy is success

**Remember:** The goal is finding truth, not completing tasks.

---

## Version Information

**Created:** 2025-10-30
**Designer:** Model Designer 2 (Hierarchical Binomial Focus)
**Data:** 12 groups, binomial trials, severe overdispersion
**Framework:** Bayesian hierarchical models via Stan/CmdStanPy

---

## Questions?

1. **What to read first?** → `QUICK_START.md`
2. **Model not converging?** → `MODEL_DECISION_TREE.md` troubleshooting
3. **Understanding the theory?** → `proposed_models.md`
4. **Which model to use?** → `MODEL_DECISION_TREE.md`
5. **Comparing designers?** → Run `model_comparison.py`

---

**Good luck with the analysis! Remember: principled inference > completing predetermined plans.**
