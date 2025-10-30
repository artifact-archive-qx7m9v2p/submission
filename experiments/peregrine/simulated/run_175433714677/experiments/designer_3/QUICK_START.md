# Quick Start Guide: Alternative Models

**Time investment**: 1 hour total | **Difficulty**: Medium | **Priority**: Model 2 → Model 1 → Model 3

---

## TL;DR

Test three alternatives to standard Negative Binomial GLM:

1. **Hierarchical Gamma-Poisson**: Explicit random effects for overdispersion
2. **Student-t Regression**: Robust regression on log-counts
3. **COM-Poisson**: Flexible dispersion (optional, computationally expensive)

**Expected outcome**: Either validate NegBin or find a better alternative.

---

## 5-Minute Overview

### What's the Problem?

EDA recommends Negative Binomial GLM for overdispersed counts (Var/Mean ≈ 70). But:
- NegBin assumes specific variance-mean relationship
- Ignores potential time-varying dispersion
- May not be robust to extreme values

### What Are We Testing?

Three fundamentally different perspectives on overdispersion:
- **Random heterogeneity** (Model 1)
- **Heavy-tailed residuals** (Model 2)
- **Flexible variance-mean relationship** (Model 3)

### What's the Goal?

Find the model that best explains the data, OR validate that standard NegBin is robust.

---

## Implementation Checklist

### Pre-Flight (5 min)

- [ ] Read this Quick Start
- [ ] Review EDA findings: `/workspace/eda/eda_report.md`
- [ ] Verify data available: `/workspace/data.json`
- [ ] Create output directory: `mkdir -p experiments/designer_3/fits`

### Model 2: Student-t (15 min)

**Why first?** Fastest, simplest, establishes baseline

- [ ] Copy Stan code from `implementation_guide.md` (lines 143-210)
- [ ] Save as `model_2_studentt.stan`
- [ ] Copy Python code (lines 212-285)
- [ ] Run fit (2 min runtime)
- [ ] Check convergence: Rhat < 1.01
- [ ] Check ν parameter: If ν > 30, Student-t may be unnecessary
- [ ] Save LOO-CV result

### Model 1: Hierarchical (20 min)

**Why second?** Moderate complexity, interpretable

- [ ] Copy Stan code from `implementation_guide.md` (lines 23-95)
- [ ] Save as `model_1_hierarchical.stan`
- [ ] Copy Python code (lines 97-140)
- [ ] Run fit (5 min runtime)
- [ ] Check convergence: Rhat < 1.01, no divergences
- [ ] Check random effects: Plot λ[i] vs year[i]
- [ ] Save LOO-CV result

### Model 3: COM-Poisson (30 min, OPTIONAL)

**Why last?** Most complex, computationally expensive

- [ ] Copy Stan code from `implementation_guide.md` (lines 258-330)
- [ ] Save as `model_3_compoisson.stan`
- [ ] Copy Python code (lines 332-380)
- [ ] Run fit (20 min runtime, may be longer)
- [ ] Check convergence: Rhat < 1.01
- [ ] Check ν parameter: If ν ≈ 1, data is Poisson (contradicts EDA)
- [ ] Save LOO-CV result

**Skip if**: Time constrained or Models 1-2 already show clear winner

### Comparison (15 min)

- [ ] Run LOO-CV comparison (code in `implementation_guide.md` lines 420-455)
- [ ] Create comparison table
- [ ] Check Var/Mean recovery for all models (must be ≈ 70)
- [ ] Generate visualizations
- [ ] Write summary report

---

## Decision Rules (Copy-Paste Ready)

### If ΔELPD > 4 (Clear Winner)

```
Winner has substantially better predictive accuracy.
Use winner for final analysis.
Report why it's better (check falsification_plan.md for interpretation).
```

### If ΔELPD < 2 (Tied)

```
Models are statistically equivalent.
Use simplest model (standard NegBin or winner of Model 1 vs 2).
Report robustness: "Results consistent across distributional assumptions."
```

### If All Fail Var/Mean Check

```
All models fail to recover Var/Mean ≈ 70.
Diagnosis: Homogeneous dispersion assumption is wrong.
Action: Pivot to time-varying dispersion model.
See falsification_plan.md lines 320-350 for next steps.
```

### If ν > 30 (Model 2)

```
Data is approximately Normal, not heavy-tailed.
Student-t robustness is unnecessary.
Stick with count models (Model 1 or standard NegBin).
```

### If λ[i] correlates with year[i] (Model 1)

```
Random effects show time pattern (not random!).
This indicates time-varying dispersion.
Action: Fit heteroscedastic model with log(φ[i]) = γ₀ + γ₁×year[i].
```

### If ν ≈ 1 (Model 3)

```
Data is consistent with Poisson (ν = 1).
This contradicts EDA finding of severe overdispersion.
Action: Check model implementation or data preprocessing.
```

---

## Emergency Troubleshooting

| Problem | Solution |
|---------|----------|
| **Rhat > 1.01** | Increase warmup to 2000, check trace plots |
| **Divergences** | Increase adapt_delta to 0.98 or 0.99 |
| **Low ESS < 400** | Increase max_treedepth to 12, run longer chains |
| **Runtime > 1 hour** | Only applies to Model 3; consider skipping |
| **Models give wildly different β₁** | Data is weakly informative; report uncertainty |
| **LOO-CV fails** | Check for extreme Pareto k values; model misspecification |

---

## What to Report

### Minimum (if time constrained)

1. Which models converged
2. LOO-CV comparison table
3. Winner (or tie) with justification
4. Var/Mean recovery check

### Complete (recommended)

1. All of above, plus:
2. Parameter estimates for all models
3. Posterior predictive check plots
4. Falsification tests (from `falsification_plan.md`)
5. Interpretation of why winner is better
6. Limitations and caveats

---

## File Map

| Need | File | Section |
|------|------|---------|
| **Model specifications** | `proposed_models.md` | Full mathematical details |
| **Stan code** | `implementation_guide.md` | Copy-paste ready code |
| **Rejection criteria** | `falsification_plan.md` | "I will abandon if..." |
| **Overview** | `README.md` | Big picture, philosophy |
| **This guide** | `QUICK_START.md` | Practical steps |

---

## Common Questions

### "Do I have to fit all three models?"

No. Priority order:
1. **Must do**: Model 2 (baseline)
2. **Should do**: Model 1 (interpretable alternative)
3. **Nice to have**: Model 3 (if time permits)

### "What if I only have 30 minutes?"

Fit Model 2 only. Compare to standard NegBin from other designers.

### "What if all models perform similarly?"

**Good!** It means results are robust. Report: "Growth rate estimate (β₁ ≈ 0.85) is consistent across Student-t, Hierarchical, and standard NegBin models."

### "What if my model doesn't converge?"

1. Check `falsification_plan.md` for that model's troubleshooting
2. Try escape routes (reparameterization, simpler version)
3. If all else fails, document failure mode and move on
4. **Failure is informative!** It tells us something about the data

### "How do I know if I'm done?"

When you can answer:
- Which model has best LOO-CV? (or are they tied?)
- Does winner recover Var/Mean ≈ 70?
- What does this tell us about the data generation process?
- Should we use this model or stick with standard NegBin?

---

## Success Metrics

**Experiment succeeds if**:

✓ At least 2 models converge
✓ LOO-CV distinguishes models (or shows robustness)
✓ Winner passes posterior predictive checks
✓ Results are interpretable

**Experiment fails (but teaches us) if**:

✗ All models fail Var/Mean check → Need time-varying dispersion
✗ All models give same LOO-CV → Distributional choice doesn't matter
✗ Parameters inconsistent → Data is weakly informative
✗ Computational failures → Model/data pathology

---

## Time Budget

| Task | Time | Cumulative |
|------|------|------------|
| Pre-flight | 5 min | 5 min |
| Fit Model 2 | 15 min | 20 min |
| Fit Model 1 | 20 min | 40 min |
| Fit Model 3 (optional) | 30 min | 70 min |
| Comparison & reporting | 15 min | 55 min (or 85 min with Model 3) |

**Minimum viable**: 40 minutes (Models 1-2 only)
**Complete**: 85 minutes (all three models)

---

## Key Take-Home Messages

1. **Standard NegBin is not wrong**: We're testing if alternatives are better
2. **Failure is success**: Learning a model doesn't work is valuable
3. **Robustness matters**: Consistent results across models = strong findings
4. **Honest uncertainty**: If data doesn't distinguish models, say so
5. **Interpretation > Fit**: Understanding why a model works matters more than fit statistics

---

## Next Steps After Completion

1. **Compare to other designers**: Do they reach same conclusions?
2. **Sensitivity analysis**: Test with different priors
3. **If winner found**: Use for predictions and interpretation
4. **If tie**: Use simplest model, report robustness
5. **If all fail**: Pivot to time-varying dispersion models

---

**Bottom Line**: This experiment tests whether we can do better than standard NegBin. If yes, great! If no, NegBin is validated from multiple perspectives. Either outcome is useful.

**Ready?** Start with Model 2 in `implementation_guide.md` (line 143).

**Questions?** See `README.md` for philosophy or `falsification_plan.md` for rejection criteria.
