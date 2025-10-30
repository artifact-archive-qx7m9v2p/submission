# Model Decision Tree: Hierarchical Binomial Strategy

## Decision Flow for Model Selection and Validation

```
START: Fit Model 2 (Non-centered, Normal priors)
│
├─ MCMC DIAGNOSTICS
│  │
│  ├─ Rhat > 1.01? → FAIL: Convergence issue
│  │  └─ Action: Run longer, increase adapt_delta
│  │
│  ├─ ESS < 400? → FAIL: Poor mixing
│  │  └─ Action: Run longer chains (4000+ iterations)
│  │
│  ├─ Divergences > 1%? → FAIL: Geometry problem
│  │  └─ Action: Increase adapt_delta to 0.95-0.99
│  │     └─ Still failing? → Already non-centered, so problem is MODEL CLASS
│  │        └─ Switch to: Beta-binomial (Designer 1)
│  │
│  └─ All pass? → PROCEED to Statistical Checks
│
├─ STATISTICAL VALIDATION
│  │
│  ├─ Posterior φ not in [3.0, 6.0]? → FAIL: Can't reproduce overdispersion
│  │  └─ Action: Model inadequate for data
│  │     └─ Switch to: Beta-binomial or Mixture model
│  │
│  ├─ Group 1 posterior = 0%? → FAIL: Shrinkage not working
│  │  └─ Action: Check priors, verify model specification
│  │     └─ If persists: Fundamental model problem
│  │
│  ├─ σ posterior < 0.1 or > 2.5? → FAIL: Prior-posterior conflict
│  │  └─ Action: Model misspecified
│  │     └─ ICC = 0.73 implies σ ≈ 0.9, extreme values are red flag
│  │
│  ├─ Pareto k > 0.7 for >3 groups? → FAIL: Influential outliers
│  │  └─ Action: Consider mixture model (discrete subgroups)
│  │
│  └─ All pass? → PROCEED to Model Refinement
│
├─ MODEL REFINEMENT
│  │
│  ├─ Fit Model 3 (Robust, Student-t priors)
│  │  │
│  │  ├─ Check posterior ν:
│  │  │  ├─ ν > 30? → Heavy tails NOT necessary
│  │  │  │  └─ Use Model 2 (simpler)
│  │  │  │
│  │  │  └─ ν < 10? → Heavy tails ARE important
│  │  │     └─ Use Model 3 for robustness
│  │  │
│  │  ├─ Compare LOO-CV:
│  │  │  ├─ M3 better by >4 elpd? → Use M3
│  │  │  ├─ M2 better by >4 elpd? → Use M2
│  │  │  └─ Difference <4 elpd? → Use M2 (simpler)
│  │  │
│  │  └─ Group 8 handling:
│  │     ├─ M3 shrinks less than M2? → Heavy tails helping
│  │     └─ M3 same as M2? → Normal adequate
│  │
│  └─ Fit Model 1 (Centered) [OPTIONAL]
│     └─ Purpose: Demonstrate non-centered advantage
│        ├─ Should have more divergences than M2
│        └─ Posteriors should match M2 (if both converge)
│
├─ CROSS-DESIGNER COMPARISON
│  │
│  ├─ Compare to Designer 1 (Beta-binomial) via LOO-CV
│  │  ├─ Beta-binomial better? → Use simpler model
│  │  ├─ Hierarchical binomial better? → Use M2 or M3
│  │  └─ Equivalent? → Prefer beta-binomial (more parsimonious)
│  │
│  └─ Compare to Designer 3 (Alternative approaches)
│     └─ Use LOO stacking weights for final inference
│
└─ FINAL RECOMMENDATION
   │
   ├─ SUCCESS: M2 or M3 works well
   │  └─ Use for: Group-specific inference, shrinkage estimates
   │
   ├─ PARTIAL SUCCESS: Works but beta-binomial simpler
   │  └─ Document: Both adequate, prefer simpler
   │
   └─ FAILURE: Cannot reproduce key data features
      └─ Document why, pivot to alternatives
         ├─ Try: Mixture models (if discrete groups)
         ├─ Try: Different likelihood (if binomial wrong)
         └─ Try: Structural model (if covariates available)
```

## Key Decision Points

### Decision 1: Is Non-Centered Working?
**Check:** Divergences < 1%, ESS > 400, Rhat < 1.01
**If NO:** Problem is model class (hierarchical binomial), not parameterization
**Action:** Switch to beta-binomial

### Decision 2: Can Model Reproduce Overdispersion?
**Check:** Posterior predictive φ ≈ 3.5-5.1
**If NO:** Model structure inadequate
**Action:** Try beta-binomial or mixture

### Decision 3: Are Heavy Tails Necessary?
**Check:** Posterior ν from Model 3
**If ν > 30:** Normal priors adequate, use M2
**If ν < 10:** Outliers problematic, use M3

### Decision 4: Which Designer's Model is Best?
**Check:** LOO-CV comparison across all designers
**Use:** Stacking weights or best elpd_loo
**Report:** All models if differences <4 elpd

## Falsification Trigger Matrix

| Observation | Threshold | Implication | Action |
|-------------|-----------|-------------|---------|
| Divergences | >2% (M2) | Geometry problem | Switch to beta-binomial |
| Max Rhat | >1.05 | Convergence failure | Run longer or abandon |
| Posterior φ | <2.5 or >7.0 | Can't reproduce overdispersion | Different model class |
| σ posterior | <0.2 or >2.0 | Prior-posterior conflict | Model misspecified |
| Group 1 posterior | =0% | Shrinkage not working | Check specification |
| Pareto k | >0.7 for >3 groups | Influential outliers | Try mixture model |
| LOO elpd | 10+ worse than beta-binomial | Overparameterized | Use simpler model |

## Success Criteria (All Must Pass)

✓ **Computational:**
- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 for all parameters
- [ ] Divergences < 1%
- [ ] BFMI > 0.3

✓ **Statistical:**
- [ ] Posterior φ ∈ [3.0, 6.0]
- [ ] σ posterior ∈ [0.5, 1.5]
- [ ] μ posterior ≈ -2.5 ± 0.5
- [ ] LOO Pareto k < 0.7 for all groups

✓ **Scientific:**
- [ ] Group 1 posterior ≈ 1-3% (not 0%)
- [ ] Group 8 appropriately shrunk (to ~11-13%)
- [ ] Small-n groups shrink more than large-n
- [ ] Posterior predictive visually matches observed

✓ **Comparative:**
- [ ] Within 4 elpd of best model
- [ ] Or provides unique scientific insight
- [ ] Computational cost justified

## Red Flags (Stop and Investigate)

🚩 **STOP if:**
- M2 fails to converge (non-centered should work!)
- σ posterior is extreme (<0.1 or >2.5)
- Multiple divergences even with adapt_delta=0.99
- Group 1 posterior stuck at exactly 0%
- All posteriors identical (complete pooling)
- Posterior predictive can't generate zero counts

🚩 **RECONSIDER if:**
- Beta-binomial is >10 elpd better
- Mixture model clearly evident in residuals
- Results are scientifically implausible
- Other designers have much better models

## Escape Routes (When to Pivot)

### Pivot to Beta-Binomial if:
- [ ] Computational issues persist across all parameterizations
- [ ] Need more parsimonious model (2 vs 12 parameters)
- [ ] LOO strongly favors beta-binomial
- [ ] No interest in group-specific effects

### Pivot to Mixture Model if:
- [ ] Multiple Pareto k > 0.7
- [ ] Evidence for discrete subgroups (not continuous)
- [ ] Residual analysis shows clustering
- [ ] Group 8 seems fundamentally different

### Pivot to Structural Model if:
- [ ] Group-level covariates become available
- [ ] Can explain WHY groups differ
- [ ] Interest in predictive model for new groups
- [ ] Current models inadequate

### Abandon Bayesian Hierarchical if:
- [ ] Frequentist methods perform much better
- [ ] Computational cost not justified
- [ ] Priors dominating data (small sample)
- [ ] Groups are incomparable (no hierarchy)

## Time Allocation Strategy

**Phase 1: Quick Check (30 min)**
- Fit M2 only
- Check diagnostics
- Verify basic posteriors
- Decision: Proceed or abandon?

**Phase 2: Thorough Validation (2 hours)**
- Fit all three models
- Posterior predictive checks
- LOO-CV comparison
- Visualizations

**Phase 3: Cross-Comparison (1 hour)**
- Compare to other designers
- Stacking weights
- Final model selection

**Phase 4: Sensitivity (optional, 2 hours)**
- Alternative priors
- Outlier removal
- Different subsets

## Expected Outcomes

### Best Case: M2 Works Perfectly
- All diagnostics pass
- Reproduces key features
- Interpretable posteriors
- Use for inference

### Good Case: M3 Needed for Robustness
- M2 has issues with outliers
- M3 resolves via heavy tails
- Posterior ν < 20
- Use M3 for inference

### Acceptable Case: Works But Beta-Binomial Simpler
- Both models fit well
- Similar predictive performance
- Beta-binomial more parsimonious
- Document and defer to simpler

### Poor Case: Computational Issues
- M2 doesn't converge well
- Geometry problems
- Already non-centered
- Abandon hierarchical binomial

### Worst Case: Statistical Failure
- Can't reproduce φ
- Posteriors unreasonable
- Prior-posterior conflict
- Model class is wrong

## Final Checklist Before Declaring Success

- [ ] All MCMC diagnostics pass
- [ ] Posterior predictive reproduces φ ≈ 3.5-5.1
- [ ] Group 1 gets reasonable posterior (1-3%)
- [ ] σ consistent with ICC = 0.73
- [ ] LOO Pareto k < 0.7 for all groups
- [ ] Results scientifically interpretable
- [ ] Compared to other designers
- [ ] Documented any issues or limitations
- [ ] Ready to report with appropriate uncertainty

## Remember

**Success is finding truth, not completing tasks.**

If hierarchical binomial fails, that's valuable information about the data generation process. Document why, pivot to better models, and report findings honestly.

The goal is reliable inference with appropriate uncertainty, not defending a predetermined approach.
