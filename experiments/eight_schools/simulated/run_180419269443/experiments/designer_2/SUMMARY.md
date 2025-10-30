# Designer 2: Quick Reference Summary

## Mission
Design robust Bayesian models that **challenge standard assumptions** and plan for failure.

---

## Three Models at a Glance

| Model | File | Key Feature | When It Wins | When It Fails |
|-------|------|-------------|--------------|---------------|
| **1. t-Distribution** | `model_1_t_distribution.stan` | Heavy tails, robust to outliers | nu < 20 (data prefer heavy tails) | nu > 50 (normal adequate) |
| **2. Mixture** | `model_2_mixture.py` | Two subpopulations | 0.1 < pi < 0.9, mu_1 ≠ mu_2 | pi → 0 or 1 (collapses) |
| **3. Dirichlet Process** | `model_3_dirichlet_process.py` | Data-driven cluster count | K_eff clearly > 1 | K_eff = 1 or = J (extremes) |

---

## Decision Tree (Start Here)

```
1. Fit Model 1 (t-distribution)
   │
   ├─ nu > 50? → STOP: Use standard normal (Designer 1's model)
   ├─ 20 < nu < 50? → REPORT: Mild heavy tails, use Model 1
   └─ nu < 20? → Continue to Model 2
       │
       2. Fit Model 2 (Mixture)
          │
          ├─ pi → 0 or 1? → STOP: Single cluster confirmed
          ├─ Genuine mixture? → REPORT: Model 2, investigate clusters
          └─ Suggests >2 clusters? → Continue to Model 3
              │
              3. Fit Model 3 (Dirichlet Process)
                 │
                 ├─ K_eff ≈ 1? → Use hierarchical normal
                 ├─ K_eff ≈ 2? → Validate with Model 2
                 └─ K_eff > 3? → Question pooling assumption
```

---

## Expected Outcomes (Given EDA: I²=2.9%)

| Model | Prediction | If Wrong, Then... |
|-------|------------|-------------------|
| Model 1 | nu ≈ 30-50 | If nu < 20: Heavy tails matter |
| Model 2 | pi → 0 or 1 | If mixture: Hidden subgroups exist |
| Model 3 | K_eff ≈ 1-2 | If K_eff > 3: Complex heterogeneity |

**Most likely:** All models agree → Standard normal adequate
**Exciting:** Any model reveals structure → Important finding

---

## Falsification Criteria (Why I'd Abandon Each Model)

### Model 1 (t-Distribution)
- ✗ Divergent transitions >5% after tuning
- ✗ nu < 3 or nu > 100 (unrealistic)
- ✗ LOO worse than normal model
- ✗ Posterior predictive checks fail

### Model 2 (Mixture)
- ✗ Label switching unsolvable
- ✗ pi posterior uniform (unidentified)
- ✗ Cluster assignments make no sense
- ✗ LOO worse than hierarchical

### Model 3 (Dirichlet Process)
- ✗ Each study in separate cluster (K=J)
- ✗ Concentration at extremes (alpha → 0 or ∞)
- ✗ Unstable across chains
- ✗ No better than simpler models

---

## Comparison to Designer 1

| Aspect | Designer 1 | Designer 2 (This) |
|--------|------------|-------------------|
| **Philosophy** | Classical foundations | Adversarial robustness |
| **Models** | Normal hierarchical, common effect | t, mixture, Dirichlet Process |
| **Focus** | Establish baseline | Test assumptions |
| **Risk** | May miss outliers | May overfit |
| **Best when** | Data well-behaved | Uncertainty about assumptions |

**Together:** Comprehensive assessment from two angles

---

## Key Metrics for Model Selection

### 1. Leave-One-Out CV (LOO-CV)
- **Primary criterion:** Highest ELPD wins
- **Rule:** If ΔELPD < 2 SE, choose simpler model
- **Tools:** `arviz.loo()` in Python, `loo` package in R

### 2. Posterior Predictive Checks
- **Test statistics:** mean, SD, min, max, range
- **Good:** p-values in [0.05, 0.95]
- **Bad:** Systematic failures (p < 0.01 or p > 0.99)

### 3. Convergence Diagnostics
- **R-hat:** < 1.01 for all parameters
- **ESS:** > 400 for all parameters
- **Divergences:** < 1% after tuning

---

## Warning Signs (Red Flags)

### Computational
- Divergent transitions persist despite tuning
- R-hat > 1.05
- ESS < 100
- Chains don't mix

### Statistical
- Extreme parameter values (tau > 100, nu < 3)
- Posterior at prior mode (data uninformative)
- Wide posteriors despite strong data
- Poor predictive performance

### Scientific
- Results change >50% when removing any study
- Model predictions contradict observed data
- Cluster assignments nonsensical
- Conclusions qualitatively different across models

**If multiple red flags → STOP and reconsider entire approach**

---

## Sensitivity Analyses (Required)

### For Each Model
1. **Prior sensitivity:** Vary hyperpriors by 2-3x
2. **Influence:** Remove Studies 4 and 5 separately
3. **Computational:** 4 chains, different seeds

### Across Models
1. **Compare LOO-CV:** Which model predicts best?
2. **Compare posteriors:** Similar mu estimates?
3. **Compare predictions:** Overlapping intervals?

---

## How to Interpret Results

### Scenario 1: All Models Agree
- **Conclusion:** Data clearly favor one structure
- **Action:** Report agreed-upon model
- **Confidence:** High

### Scenario 2: Models Disagree Slightly
- **Conclusion:** Data consistent with multiple structures
- **Action:** Report most parsimonious model + sensitivity
- **Confidence:** Moderate

### Scenario 3: Models Disagree Strongly
- **Conclusion:** Data insufficient to distinguish
- **Action:** Report range of estimates, emphasize uncertainty
- **Confidence:** Low

### Scenario 4: All Models Fail
- **Conclusion:** Fundamental misspecification
- **Action:** Reconsider problem formulation
- **Confidence:** Very low, need more data

---

## Quick Implementation Guide

### Step 1: Model 1 (Start here)
```bash
# Compile Stan model
cmdstan_model = CmdStanModel('model_1_t_distribution.stan')

# Fit
fit = cmdstan_model.sample(data=data, chains=4, iter_sampling=2000)

# Check nu
nu_mean = fit.stan_variable('nu').mean()
```

**Decision:**
- nu > 50 → Stop, use normal
- 20-50 → Report Model 1
- < 20 → Continue to Model 2

### Step 2: Model 2 (Conditional)
```python
# Fit mixture
trace, model = fit_mixture_model(y, sigma)

# Check mixing proportion
pi_mean = trace.posterior['pi'].mean()
```

**Decision:**
- pi ≈ 0 or 1 → Single cluster
- 0.1 < pi < 0.9 → Genuine mixture
- Check if mu_1 ≠ mu_2

### Step 3: Model 3 (Advanced)
```python
# Fit Dirichlet Process
trace, model = fit_dp_mixture(y, sigma, K=10)

# Check effective clusters
k_eff = compute_effective_clusters(trace)
```

**Decision:**
- K_eff ≈ 1 → Hierarchical normal
- K_eff ≈ 2 → Validate with Model 2
- K_eff > 3 → Complex heterogeneity

---

## Files and Their Purpose

| File | Purpose | Status |
|------|---------|--------|
| `proposed_models.md` | **Comprehensive design document** (28 KB) | ✅ Complete |
| `model_1_t_distribution.stan` | **Stan code for heavy-tailed model** | ✅ Ready |
| `model_2_mixture.py` | **PyMC code for mixture model** | ✅ Ready |
| `model_3_dirichlet_process.py` | **PyMC code for DP model** | ✅ Ready |
| `README.md` | **Detailed documentation** (12 KB) | ✅ Complete |
| `SUMMARY.md` | **Quick reference (this file)** | ✅ Complete |

---

## What Makes This Design Different?

### 1. Adversarial Mindset
- Explicitly tries to **break** assumptions
- Plans for failure, not just success
- Falsification criteria for each model

### 2. Multiple Exit Points
- Can abandon model classes if evidence suggests
- Decision points throughout
- Clear escalation path (Model 1 → 2 → 3)

### 3. Explicit Predictions
- States what we expect to find
- Defines what would surprise us
- Plans for being wrong

### 4. Robustness Focus
- Heavy-tailed distributions
- Mixture models for subgroups
- Non-parametric alternatives

### 5. Honest Uncertainty
- Reports when models disagree
- Emphasizes limitations
- Doesn't claim certainty where none exists

---

## Success = Finding Truth, Not Completing Tasks

**Good outcomes:**
- ✅ Discover assumptions are wrong → Pivot quickly
- ✅ Find all models agree → Confidence in conclusion
- ✅ Identify limitations → Honest reporting

**Bad outcomes:**
- ❌ Fit models without checking assumptions
- ❌ Ignore evidence contradicting approach
- ❌ Report results without sensitivity analyses
- ❌ Claim certainty inappropriately

---

## Contact Points with Designer 1

### Likely Agreement
- Pooled effect estimate (mu) should be similar
- If I²=2.9% holds, low heterogeneity
- No evidence of outliers (yet)

### Possible Disagreement
- Uncertainty quantification (Designer 2 more conservative)
- Robustness claims (Designer 2 emphasizes)
- Model complexity (Designer 2 explores more options)

### Synthesis Strategy
- Compare posteriors for mu and tau
- Compare LOO-CV across all models (both designers)
- Identify best-performing model
- Report consensus + areas of uncertainty

---

## Timeline and Phases

| Phase | Duration | Goal | Deliverable |
|-------|----------|------|-------------|
| **Phase 1** | Week 1 | Fit & evaluate Model 1 | Convergence, LOO-CV, decision |
| **Phase 2** | Week 2 | Conditional on Phase 1 | Model 2 if needed |
| **Phase 3** | Week 3 | Advanced/optional | Model 3 if warranted |
| **Phase 4** | Week 4 | Synthesis & reporting | Final recommendation |

**Key:** Can stop at any phase if model is adequate

---

## Final Checklist

Before reporting results, verify:

- [ ] Convergence: R-hat < 1.01, ESS > 400
- [ ] Diagnostics: No divergences, chains mix well
- [ ] Validation: LOO-CV computed, PPC performed
- [ ] Sensitivity: Prior and influence analyses done
- [ ] Comparison: Evaluated against Designer 1's models
- [ ] Interpretation: Results make scientific sense
- [ ] Documentation: All decisions justified
- [ ] Limitations: Honest about uncertainties

---

**Bottom Line:** Start with Model 1. Only proceed if data demand it. Be ready to stop early if simple model suffices. Falsification is success, not failure.

---

**Files:** `/workspace/experiments/designer_2/`
**Next Step:** Implement Model 1 and evaluate
**Decision Point:** After Model 1 convergence diagnostics
