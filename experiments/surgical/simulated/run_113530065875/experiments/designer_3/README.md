# Designer #3: Practical Bayesian Modeling Strategy
## Computational Efficiency, Interpretability, and Robustness

**Designer Role**: Skeptical Practitioner - "Will it actually work?"
**Focus**: Practical considerations, computational efficiency, model selection strategy
**Date**: 2025-10-30

---

## Quick Start

**TL;DR**: Use Model 1 (Hierarchical Binomial with non-centered parameterization). It will work. Spend your time on interpretation, not model tweaking.

**Expected Results**:
- Sampling: <1 minute
- All diagnostics pass
- Clear group-specific estimates
- Interpretable shrinkage patterns

---

## Document Index

### 1. Main Document: `proposed_models.md` (22,000 words)
**Comprehensive modeling strategy with three models**

**Contents**:
- Model 1: Hierarchical Binomial (Non-Centered) ⭐ RECOMMENDED
- Model 2: Beta-Binomial (Simple Alternative)
- Model 3: Robust Hierarchical (Escalation Only)
- Detailed specifications with Stan/PyMC code
- Practical advantages and disadvantages
- Computational profiles
- Falsification criteria
- Model comparison strategy
- Red/green flags for iteration
- Resource estimates

**Key Sections**:
- Executive Summary (page 1)
- Model Proposals (pages 2-15)
- Decision Tree (page 16)
- Red/Green Flags (pages 17-18)
- Resource Estimates (page 19-20)
- Implementation Checklist (page 21-22)

---

### 2. Quick Reference: `executive_summary.md` (4,000 words)
**Distilled essentials for rapid decision-making**

**Contents**:
- Bottom-line recommendation
- Three-model strategy table
- 30-second decision flow
- Copy-paste ready Stan code for Model 1
- Expected results and pass/fail criteria
- Practical checklist
- Expected parameter estimates
- Communication strategy
- FAQs
- Quick reference matrix

**Use this when**:
- You need to make a quick decision
- You want the main points without details
- You're explaining the approach to stakeholders

---

### 3. Implementation: `implementation_guide.md` (8,000 words)
**Working code examples for immediate use**

**Contents**:
- Setup (R and Python)
- Data loading code
- Complete Stan models (all three)
- R fitting and diagnostic code
- Python/PyMC implementations
- LOO comparison code
- Visualization functions
- Complete workflow scripts
- Quick diagnostic functions
- Troubleshooting solutions
- Final checklist

**Use this when**:
- Ready to implement models
- Need copy-paste code
- Want working examples
- Debugging issues

---

### 4. Decision Guide: `decision_flowchart.md` (5,000 words)
**Visual decision trees for every situation**

**Contents**:
- Main decision flow (ASCII flowchart)
- Diagnostic decision trees:
  - Convergence issues
  - Divergent transitions
  - LOO Pareto k
  - Posterior predictive checks
- Model comparison tree
- Prior sensitivity tree
- Troubleshooting common problems
- Stopping rules (when to stop vs escalate vs abandon)
- Final decision matrix

**Use this when**:
- Facing a specific decision point
- Diagnostics fail and need guidance
- Choosing between models
- Troubleshooting issues

---

## The Three Models

### Model 1: Hierarchical Binomial (Non-Centered) ⭐

**When to use**: Default choice, need group-specific estimates
**Complexity**: Medium (14 parameters)
**Time**: 1-2 hours total
**Confidence**: 90% this will work perfectly

**Strengths**:
- Partial pooling optimizes bias-variance tradeoff
- Automatic shrinkage (small groups borrow strength)
- Well-understood, extensively documented
- Fast convergence with non-centered parameterization

**Weaknesses**:
- Normal distribution has light tails (may miss outliers)
- Assumes symmetric heterogeneity
- Requires understanding of logit scale

**Expected for this dataset**:
- μ ≈ -2.4 (pooled rate ~8%)
- τ ≈ 0.4 (moderate heterogeneity)
- Group 4 (n=810) shrinks ~20%
- Group 10 (n=97) shrinks ~65%
- All diagnostics pass

---

### Model 2: Beta-Binomial

**When to use**: Only care about population mean, want simplicity
**Complexity**: Low (2 parameters)
**Time**: 1 hour total
**Confidence**: 100% will converge, 70% will be sufficient

**Strengths**:
- Extremely simple and fast
- Marginal model (no group parameters)
- Easy to explain to anyone
- Robust convergence

**Weaknesses**:
- No group-specific estimates (dealbreaker for many)
- Treats all groups identically
- Can't identify outliers
- Weaker predictions

**Expected for this dataset**:
- μ ≈ 0.08 (8% mean rate)
- φ ≈ 10 (concentration)
- Implied overdispersion ≈ 3.6
- Will be outperformed by Model 1 in LOO

**When acceptable**:
- Research question is population-level only
- Stakeholders don't care about individual groups
- Speed is paramount

---

### Model 3: Robust Hierarchical (Student-t)

**When to use**: Only if Model 1 fails specific diagnostics
**Complexity**: High (15 parameters)
**Time**: 2 hours total
**Confidence**: 80% will improve over Model 1 if outliers present

**Strengths**:
- Heavy tails accommodate outliers
- ν parameter learns tail heaviness from data
- Same interpretation as Model 1
- Better LOO for extreme groups

**Weaknesses**:
- 2-5x slower than Model 1
- More divergences likely
- One extra parameter to monitor
- May not improve predictions with only 12 groups

**Expected for this dataset**:
- If ν <5: Heavy tails justified, use this
- If ν >10: No evidence for heavy tails, use Model 1
- Likely: ν ≈ 6-8 (borderline, Model 1 probably sufficient)

**Escalation triggers**:
- Model 1 has multiple Pareto k >0.7
- Model 1 PPC fails for outliers
- Domain knowledge suggests heavy-tailed heterogeneity

---

## Decision Logic

### Start Here
```
Do you need group-specific estimates?
  ├─ NO  → Model 2 (Beta-binomial)
  └─ YES → Model 1 (Hierarchical)
            ├─ Diagnostics pass? → DONE ✓
            └─ Diagnostics fail? → Try Model 3
```

### Diagnostics Pass If
- ✓ Rhat <1.01 for all parameters
- ✓ ESS >400 for μ, τ
- ✓ Divergences <1%
- ✓ Pareto k <0.7 for all groups
- ✓ PPC captures observed variance (~3.6× binomial)
- ✓ Shrinkage follows expected pattern (small→more, large→less)

### Model Comparison
- **ΔLOO <4**: Models tied, choose simpler
- **ΔLOO 4-10**: Weak preference for better
- **ΔLOO >10**: Strong preference for better

### Stop Iterating When
- All diagnostics pass
- Results interpretable and plausible
- Stakeholders understand outputs
- Prior sensitivity <10%

---

## Key Recommendations

### Computational
1. **Always use non-centered parameterization** for hierarchical models
2. **Start with adapt_delta=0.8**, increase to 0.95 if divergences occur
3. **4 chains, 2000 iterations** is sufficient for 12 groups
4. **Expect <2 minutes** for Model 1 on modern laptop

### Statistical
1. **LOO is your friend** - trust Pareto k diagnostics
2. **PPC for key features** - does model capture variance?
3. **Don't over-tune** - good enough beats perfect
4. **Compare to baselines** - pooled and unpooled models

### Practical
1. **Start with Model 1** - it's the right default
2. **Fit all three quickly** - comparison validates choice
3. **Visualize shrinkage** - stakeholders understand this
4. **Report uncertainty** - credible intervals, not just point estimates

### Communication
1. **For statisticians**: "Hierarchical binomial with partial pooling"
2. **For domain experts**: "Groups share information, but each has its own rate"
3. **For executives**: "Success rates vary 3-14%, we account for sample sizes"

---

## Expected Timeline

### First-Time Implementation
- **Setup + data prep**: 30 minutes
- **Model 1 coding**: 30 minutes
- **Model 1 fitting**: 1 minute
- **Diagnostics**: 20 minutes
- **Model 2 comparison**: 15 minutes
- **Visualization**: 30 minutes
- **Write-up**: 1.5 hours
- **Total**: 3-4 hours

### Experienced User
- **Everything above**: 1-2 hours

### If Problems Arise
- **Troubleshooting**: +30 minutes
- **Model 3 fitting**: +30 minutes
- **Prior sensitivity**: +30 minutes
- **Maximum total**: 6 hours

---

## Red Flags: When to Stop and Reconsider

### Computational 🚨
- Divergences >5% after tuning
- Rhat >1.05 after 4000 iterations
- Sampling time >10 minutes for 12 groups

### Statistical 🚩
- Multiple Pareto k >0.7
- Posterior predictive p-value <0.01 or >0.99
- Prior-posterior conflict

### Scientific ⚠️
- Implausible parameter values (τ >2, any p >0.3)
- Shrinkage pathology (large groups shrink more than small)
- Unstable conclusions across runs

**If any of these occur**: See `decision_flowchart.md` for specific guidance

---

## Green Flags: Success Indicators

### Computational ✅
- Rhat <1.01, ESS >400, Divergences <1%
- Sampling completes in expected time
- Trace plots show good mixing

### Statistical ✅
- LOO clearly favors hierarchical (ΔLOO >10 vs pooled)
- All Pareto k <0.7
- PPC passes for key features

### Scientific ✅
- Parameters in expected ranges (μ ≈ -2.4, τ ≈ 0.4)
- Shrinkage logical (small→more, large→less)
- Outliers identified but not excluded

### Practical ✅
- Results stable across runs
- Stakeholders understand outputs
- Conclusions actionable

**If all green flags present**: Accept model and report results. Don't chase perfection.

---

## Files in This Directory

1. **README.md** (this file) - Overview and navigation
2. **proposed_models.md** - Complete modeling strategy (MAIN DOCUMENT)
3. **executive_summary.md** - Quick reference guide
4. **implementation_guide.md** - Copy-paste code examples
5. **decision_flowchart.md** - Visual decision trees

---

## Data Context (from EDA)

- **Structure**: 12 groups, 2,814 trials, 196 successes (6.97%)
- **Overdispersion**: Strong (φ=3.59, ICC=0.56, p<0.0001)
- **Range**: 3.1% to 14.0% success rates (4.5-fold difference)
- **Outliers**: Groups 2, 4, 8 identified
- **Sample sizes**: 47 to 810 per group (16-fold range)
- **Exchangeability**: Confirmed (no covariates needed)

**Implication**: Hierarchical model is necessary and justified. Pooled model empirically rejected. Unpooled model wastes information.

---

## Model Selection Summary

| Question | Answer | Model |
|----------|--------|-------|
| Need group estimates? | Yes | Model 1 or 3 |
| Need group estimates? | No | Model 2 |
| Outliers problematic? | Yes | Model 3 |
| Outliers problematic? | No | Model 1 |
| Want simplest? | Yes | Model 2 |
| Want best predictions? | Yes | Model 1 or 3 |
| Limited computation? | Yes | Model 2 |

**Default**: Start with Model 1, validate with diagnostics, escalate to Model 3 only if needed.

---

## Falsification Criteria

### I will abandon Model 1 if:
- ❌ Divergences persist >5% despite adapt_delta=0.99
- ❌ Multiple groups have Pareto k >0.7
- ❌ Posterior predictive fails to capture variance
- ❌ Large groups shrink >40% (pathological)

### I will abandon Model 3 if:
- ❌ Posterior ν >10 (no evidence for heavy tails → use Model 1)
- ❌ LOO not better than Model 1
- ❌ Computational issues persist

### I will abandon all models if:
- ❌ None converge despite extensive tuning
- ❌ All fail posterior predictive checks
- ❌ Results contradict known scientific facts
- **Action**: Reconsider data quality, model assumptions, or consult domain expert

---

## Success Definition

**Success** = Answering the scientific question with defensible methodology, not achieving perfect diagnostics.

**Acceptable outcome**:
- Diagnostics in acceptable range (not perfect)
- Model captures key data features
- Results interpretable and actionable
- Stakeholders trust the analysis

**Not required**:
- Every diagnostic in "ideal" range
- Multiple model classes exhausted
- Extensive sensitivity analyses (unless uncertainty is high)

---

## Contact Points with Other Designers

### Designer #1 (Theoretical Focus)
- May propose more complex models (e.g., GP, mixture)
- My contribution: Computational feasibility check
- Integration: "Here's the simplest model that works"

### Designer #2 (Substantive Focus)
- May propose domain-informed priors or structure
- My contribution: Implementation practicality
- Integration: "Here's how to fit your model efficiently"

### Synthesis
- Show that practical models often outperform complex ones
- Demonstrate value of simplicity in communication
- Validate that computational efficiency matters

---

## Philosophy

**Bayesian modeling is a tool for understanding, not an end in itself.**

Principles:
1. Start simple, add complexity only when justified
2. Validate ruthlessly, but don't chase perfection
3. Communicate clearly > Model sophistication
4. "Good enough" is often truly good enough
5. Failed diagnostics are information, not failure

**The goal**: Answer the research question in a way that:
- Stakeholders understand
- Results are reproducible
- Methods are defensible
- Conclusions are actionable

---

## Quick Troubleshooting

**"Sampling is slow"** → Check non-centered parameterization
**"Divergent transitions"** → Increase adapt_delta to 0.95
**"Low ESS for tau"** → OK if Rhat is fine and tau near 0
**"High Pareto k"** → Try Model 3 (robust)
**"PPC fails"** → Check variance, try robust model
**"Results implausible"** → Check data quality first
**"Can't decide between models"** → Choose simpler if ΔLOO <4

See `decision_flowchart.md` for detailed guidance.

---

## Final Recommendation

**For this dataset**:

1. **Fit Model 1** (Hierarchical, non-centered)
2. **Check diagnostics** (should pass)
3. **Compute LOO** (should be best)
4. **Report results** (don't over-iterate)

**Expected outcome**: 1-2 hours to complete analysis with clear, interpretable results.

**If complications arise**: Consult `decision_flowchart.md` for specific guidance.

**Remember**: The perfect model is the one that answers the question and gets used. Ship results, iterate if needed, but don't let perfect be the enemy of good.

---

## Document Statistics

- **Total words**: ~40,000 across all documents
- **Code examples**: 15+ complete implementations
- **Decision trees**: 8 detailed flowcharts
- **Practical tips**: 50+ specific recommendations
- **Time investment**: ~6 hours of design and documentation
- **Expected user time savings**: 4-8 hours (amortized over multiple analyses)

---

**All files created**: 2025-10-30
**Designer**: #3 (Practical/Computational Focus)
**Status**: Complete and ready for implementation
